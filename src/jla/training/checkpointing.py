# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Crash-consistent training checkpoints.

Layout under the checkpoint directory::

    step_00000100.incomplete      present while step 100 is being written
    step_00000100/
        model.safetensors         parameters        (payload_format "safetensors")
        optimizer.safetensors     optimizer state
        data_state.json           data position, typed RNG key, config hash
        manifest.json             commit record, written last
    latest                        "step=100" pointer (convenience only)

Commit protocol: the ``.incomplete`` marker is published first, payloads are
written to temporary files, fsynced and renamed, then the data-state sidecar
and finally the manifest (sizes of every artifact) are published, the marker
is removed, ``latest`` is updated and old checkpoints are pruned. Discovery
trusts only manifests whose artifacts all exist with the recorded sizes, so a
crash at any point leaves the previous checkpoint as the latest.

Payload formats:

* ``safetensors``: one file per tree, for single-process runs (every array is
  fully addressable);
* ``sharded``: every process writes the shards it owns (``replica_id == 0``)
  to ``model.pNNNNN-of-MMMMM.safetensors`` plus an index; no process ever
  transfers data it does not hold, and restore reassembles any target
  sharding from the chunks;
* ``orbax``: ``orbax.checkpoint.StandardCheckpointer`` (optional dependency).

Single-process saves take a synchronous host snapshot (safe against buffer
donation) and write on one background worker; multi-process saves are
synchronous and separated by barriers.

Restores are strict: every leaf must be present with its exact shape and
dtype (no silent casts, no partial optimizer state).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jla.distributed.init import sync_processes
from jla.distributed.sharding import path_names

__all__ = [
    "CheckpointIntegrity",
    "CheckpointManager",
    "Restored",
    "flatten_arrays",
    "load_pytree",
    "save_pytree",
    "shutdown_checkpointing",
    "submit_checkpoint_work",
    "verify_checkpoint",
    "wait_for_pending_checkpoints",
]

logger = logging.getLogger(__name__)

SCHEMA = "jla.checkpoint"
VERSION = 1
_STEP_DIR = re.compile(r"^step_(\d{8,})$")
_FORMATS = ("safetensors", "sharded", "orbax")

# ---------------------------------------------------------------------------
# Background writer: one worker, so saves complete in submission order.
# ---------------------------------------------------------------------------

_EXECUTOR: ThreadPoolExecutor | None = None
_PENDING: list[Future] = []
_LOCK = threading.Lock()


def submit_checkpoint_work(fn: Callable[[], Any]) -> Future:
    """Run ``fn`` on the single checkpoint worker; :func:`wait_for_pending_checkpoints` awaits it."""
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jla-ckpt")
        future = _EXECUTOR.submit(fn)
        _PENDING.append(future)
    return future


def wait_for_pending_checkpoints(timeout: float | None = None) -> None:
    """Block until all submitted checkpoint work is done; re-raises the first failure."""
    if timeout is not None and timeout < 0:
        raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        with _LOCK:
            if not _PENDING:
                return
            future = _PENDING[0]
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        try:
            future.result(timeout=remaining)
        finally:
            if future.done():
                with _LOCK:
                    if _PENDING and _PENDING[0] is future:
                        _PENDING.pop(0)


def shutdown_checkpointing(timeout: float | None = None) -> None:
    """Flush pending work and release the worker (a later save creates a new one)."""
    global _EXECUTOR
    wait_for_pending_checkpoints(timeout)
    with _LOCK:
        executor, _EXECUTOR = _EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some network filesystems reject directory fsync
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _read_json(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path} is unreasonably large")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{path} contains duplicate key {key!r}")
            out[key] = value
        return out

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _save_safetensors(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, str]) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        save_file(dict(arrays), tmp, metadata=dict(metadata))
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pytree flattening and host snapshots
# ---------------------------------------------------------------------------


def flatten_arrays(tree: Any) -> dict[str, Any]:
    """``{dotted.key.path: array}`` for every array leaf; colliding paths are rejected."""
    flat: dict[str, Any] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)):
            continue
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.extended):
            raise TypeError(f"{'.'.join(path_names(path))}: typed PRNG keys are stored in the data-state sidecar")
        key = ".".join(path_names(path))
        if not key:
            raise ValueError("checkpoint array paths must not be empty")
        if key in flat:
            raise ValueError(f"pytree contains colliding checkpoint key {key!r}")
        flat[key] = leaf
    return flat


def _host_copies(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Owned host copies of fully addressable arrays (taken before the step reuses donated buffers)."""
    for value in arrays.values():
        if isinstance(value, jax.Array):
            if not value.is_fully_addressable:
                raise ValueError(
                    "payload_format 'safetensors' needs fully addressable arrays; "
                    "use 'sharded' or 'orbax' for multi-process runs"
                )
            value.copy_to_host_async()
    return {key: np.array(value, copy=True) for key, value in arrays.items()}


def _shard_chunks(arrays: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """This process's unique shards (``replica_id == 0``) and their global index."""
    chunks: dict[str, np.ndarray] = {}
    index: dict[str, Any] = {}
    for name, value in arrays.items():
        entry = {"shape": list(value.shape), "dtype": str(jnp.dtype(value.dtype)), "chunks": []}
        if isinstance(value, jax.Array):
            pieces = [(shard.index, shard.data) for shard in value.addressable_shards if shard.replica_id == 0]
            for _, data in pieces:
                data.copy_to_host_async()
        elif jax.process_index() == 0:
            pieces = [(tuple(slice(0, d) for d in value.shape), value)]
        else:
            pieces = []
        for number, (slices, data) in enumerate(pieces):
            key = f"{name}@{number}"
            start = [s.indices(dim)[0] for s, dim in zip(slices, value.shape)]
            chunks[key] = np.array(data, copy=True)
            entry["chunks"].append({"key": key, "start": start, "shape": list(chunks[key].shape)})
        index[name] = entry
    return chunks, index


# ---------------------------------------------------------------------------
# Restoring arrays into target shardings
# ---------------------------------------------------------------------------


def _target_spec(leaf: Any) -> tuple[tuple[int, ...], np.dtype, Any]:
    sharding = getattr(leaf, "sharding", None)
    if sharding is None:
        raise ValueError("restore templates need arrays (or ShapeDtypeStructs) with a sharding")
    return tuple(leaf.shape), np.dtype(leaf.dtype), sharding


def _check_keys(available: set[str], targets: Mapping[str, Any], what: str, strict: bool) -> None:
    missing = sorted(set(targets) - available)
    unexpected = sorted(available - set(targets))
    if strict and (missing or unexpected):
        raise ValueError(f"{what} checkpoint does not match its template: missing={missing[:10]} unexpected={unexpected[:10]}")
    if missing:
        logger.warning("%s checkpoint lacks %d leaves (kept from the template): %s", what, len(missing), missing[:5])
    if unexpected:
        logger.warning("%s checkpoint has %d unexpected leaves (ignored): %s", what, len(unexpected), unexpected[:5])


def _check_leaf(name: str, shape: tuple[int, ...], dtype: str, target_shape: tuple[int, ...], target_dtype: np.dtype) -> None:
    if tuple(shape) != target_shape:
        raise ValueError(f"{name}: checkpoint shape {tuple(shape)} != expected {target_shape}")
    if np.dtype(dtype) != target_dtype:
        raise ValueError(f"{name}: checkpoint dtype {dtype} != expected {target_dtype} (no implicit casts)")


def _restore_from_files(files: list[Path], targets: Mapping[str, Any], what: str, strict: bool, chunked: bool) -> dict[str, jax.Array]:
    from safetensors import safe_open

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework="numpy")) for path in files]
        # name -> list of (handle, key, start, shape); a plain file has one chunk per leaf.
        layout: dict[str, list[tuple[Any, str, tuple[int, ...], tuple[int, ...]]]] = {}
        meta: dict[str, tuple[tuple[int, ...], str]] = {}
        for path, handle in zip(files, handles):
            if chunked:
                index = _read_json(path.with_suffix(".index.json"))
                for name, entry in index.items():
                    meta[name] = (tuple(entry["shape"]), entry["dtype"])
                    for chunk in entry["chunks"]:
                        layout.setdefault(name, []).append((handle, chunk["key"], tuple(chunk["start"]), tuple(chunk["shape"])))
            else:
                for key in handle.keys():
                    tensor = handle.get_slice(key)
                    shape = tuple(tensor.get_shape())
                    meta[key] = (shape, _safetensors_dtype(tensor))
                    layout[key] = [(handle, key, (0,) * len(shape), shape)]
        _check_keys(set(meta), targets, what, strict)
        restored = {}
        for name, leaf in targets.items():
            if name not in meta:
                continue
            shape, dtype, sharding = _target_spec(leaf)
            _check_leaf(f"{what}.{name}", meta[name][0], meta[name][1], shape, dtype)
            restored[name] = jax.make_array_from_callback(shape, sharding, _region_reader(name, layout[name], shape, dtype))
        return restored


def _safetensors_dtype(tensor: Any) -> str:
    names = {
        "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
        "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
        "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
    }  # fmt: skip
    code = tensor.get_dtype()
    if code not in names:
        raise ValueError(f"unsupported safetensors dtype {code}")
    return names[code]


def _region_reader(
    name: str,
    chunks: list[tuple[Any, str, tuple[int, ...], tuple[int, ...]]],
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> Callable[[tuple[slice, ...]], np.ndarray]:
    def read(index: tuple[slice, ...]) -> np.ndarray:
        if not shape:  # scalars are stored as one chunk
            handle, key, _, _ = chunks[0]
            return np.asarray(handle.get_tensor(key), dtype=dtype).reshape(())
        bounds = [s.indices(dim)[:2] for s, dim in zip(index, shape)]
        region = np.empty(tuple(hi - lo for lo, hi in bounds), dtype=dtype)
        covered = 0
        for handle, key, start, chunk_shape in chunks:
            overlap = [
                (max(lo, c0), min(hi, c0 + cn), c0, lo)
                for (lo, hi), c0, cn in zip(bounds, start, chunk_shape)
            ]
            if any(a >= b for a, b, _, _ in overlap):
                continue
            source = tuple(slice(a - c0, b - c0) for a, b, c0, _ in overlap)
            target = tuple(slice(a - lo, b - lo) for a, b, _, lo in overlap)
            region[target] = handle.get_slice(key)[source]
            covered += int(np.prod([b - a for a, b, _, _ in overlap]))
        if covered != region.size:
            raise ValueError(f"{name}: checkpoint chunks cover {covered} of {region.size} requested elements")
        return region

    return read


# ---------------------------------------------------------------------------
# Single-file pytree payloads
# ---------------------------------------------------------------------------


def save_pytree(path: str | Path, tree: Any, *, metadata: Mapping[str, str] | None = None) -> Path:
    """Write every array leaf of ``tree`` to one safetensors file (atomic, synchronous)."""
    destination = Path(path)
    _save_safetensors(destination, _host_copies(flatten_arrays(tree)), metadata or {})
    return destination


def load_pytree(path: str | Path, template: Any, *, strict: bool = True) -> Any:
    """Load ``path`` into a template pytree whose array leaves carry target shardings.

    Shapes and dtypes must match exactly. With ``strict=False`` leaves missing
    from the file keep their template values and extra file entries are ignored.
    """
    targets = flatten_arrays(template)
    restored = _restore_from_files([Path(path)], targets, "pytree", strict, chunked=False)
    return _unflatten_into(template, restored)


def _unflatten_into(template: Any, restored: Mapping[str, Any]) -> Any:
    def replace(path: tuple[Any, ...], leaf: Any) -> Any:
        return restored.get(".".join(path_names(path)), leaf)

    return jax.tree_util.tree_map_with_path(replace, template)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointIntegrity:
    """Result of :func:`verify_checkpoint`."""

    path: Path
    valid: bool
    step: int | None = None
    payload_format: str | None = None
    artifact_bytes: int = 0
    sha256: dict[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def require_valid(self) -> CheckpointIntegrity:
        if not self.valid:
            raise ValueError(f"invalid checkpoint {self.path}: {'; '.join(self.errors) or 'unknown failure'}")
        return self


def _validated_manifest(step_dir: Path) -> dict[str, Any]:
    manifest = _read_json(step_dir / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise ValueError("unknown manifest schema/version")
    if manifest.get("complete") is not True:
        raise ValueError("manifest is not committed")
    match = _STEP_DIR.match(step_dir.name)
    if match is None or manifest.get("step") != int(match.group(1)):
        raise ValueError("manifest step does not match its directory")
    if manifest.get("payload_format") not in _FORMATS:
        raise ValueError(f"unknown payload format {manifest.get('payload_format')!r}")
    for artifact in manifest.get("artifacts", []):
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path escapes the checkpoint: {relative}")
        target = step_dir / relative
        if not target.exists():
            raise ValueError(f"missing artifact {relative}")
        if target.is_file() and target.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {relative}")
    return manifest


def verify_checkpoint(step_dir: str | Path, *, hashes: bool = False) -> CheckpointIntegrity:
    """Validate a committed step directory without a model; optionally hash every artifact."""
    root = Path(step_dir)
    try:
        manifest = _validated_manifest(root)
        files = [root / artifact["path"] for artifact in manifest["artifacts"]]
        digests = {str(path.relative_to(root)): _sha256(path) for path in files if hashes and path.is_file()}
        return CheckpointIntegrity(
            path=root,
            valid=True,
            step=manifest["step"],
            payload_format=manifest["payload_format"],
            artifact_bytes=sum(path.stat().st_size for path in files if path.is_file()),
            sha256=digests,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return CheckpointIntegrity(path=root, valid=False, errors=(f"{type(exc).__name__}: {exc}",))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Restored:
    """State loaded by :meth:`CheckpointManager.restore`."""

    step: int
    model: Any
    opt_state: Any
    data_state: dict[str, Any]
    rng_key: jax.Array
    manifest: dict[str, Any]


def _orbax_module() -> Any:
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:
        raise ImportError("payload_format 'orbax' needs the optional 'orbax-checkpoint' package") from exc
    return ocp


def _key_payload(key: jax.Array) -> dict[str, Any]:
    data = jax.random.key_data(key)
    host = np.asarray(data if data.is_fully_addressable else data.addressable_data(0))
    return {"impl": str(jax.random.key_impl(key)), "shape": list(host.shape), "data": host.astype(np.uint32).ravel().tolist()}


def _key_from_payload(payload: Mapping[str, Any]) -> jax.Array:
    data = np.asarray(payload["data"], dtype=np.uint32).reshape(tuple(payload["shape"]))
    return jax.random.wrap_key_data(jnp.asarray(data), impl=payload["impl"])


class CheckpointManager:
    """Saves and restores ``(model, opt_state, data state, RNG key)`` under one directory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        keep: int = 3,
        asynchronous: bool = True,
        payload_format: str = "auto",
    ) -> None:
        if keep < 0:
            raise ValueError("keep must be >= 0 (0 keeps every checkpoint)")
        self.directory = Path(directory)
        self.keep = int(keep)
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        if payload_format == "auto":
            if self.process_count == 1:
                payload_format = "safetensors"
            else:
                try:
                    _orbax_module()
                    payload_format = "orbax"
                except ImportError:
                    payload_format = "sharded"
        if payload_format not in _FORMATS:
            raise ValueError(f"payload_format must be one of {('auto', *_FORMATS)}, got {payload_format!r}")
        if payload_format == "orbax":
            _orbax_module()
        if payload_format == "safetensors" and self.process_count > 1:
            raise ValueError("payload_format 'safetensors' is single-process; use 'sharded' or 'orbax'")
        self.payload_format = payload_format
        # Multi-process saves are collective and stay on the main thread.
        self.asynchronous = bool(asynchronous) and self.process_count == 1 and payload_format != "orbax"
        if self.process_index == 0:
            self.directory.mkdir(parents=True, exist_ok=True)

    # -- discovery ---------------------------------------------------------

    def step_dir(self, step: int) -> Path:
        return self.directory / f"step_{step:08d}"

    def committed_steps(self) -> list[int]:
        """Steps whose manifest and artifacts validate, oldest first."""
        if not self.directory.is_dir():
            return []
        steps = []
        for entry in self.directory.iterdir():
            match = _STEP_DIR.match(entry.name)
            if match is None or not entry.is_dir():
                continue
            try:
                _validated_manifest(entry)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                if (entry / "manifest.json").exists():
                    logger.warning("ignoring invalid checkpoint %s: %s", entry, exc)
                continue
            steps.append(int(match.group(1)))
        return sorted(steps)

    def latest_step(self) -> int | None:
        steps = self.committed_steps()
        return steps[-1] if steps else None

    # -- saving ------------------------------------------------------------

    def save(
        self,
        step: int,
        *,
        model: Any,
        opt_state: Any,
        data_state: Mapping[str, Any],
        rng_key: jax.Array,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Checkpoint ``step``; returns once the state is snapshotted (async) or committed (sync)."""
        if step < 0:
            raise ValueError("step must be >= 0")
        wait_for_pending_checkpoints()  # one save in flight; keeps commits ordered
        step_dir = self.step_dir(step)
        marker = self.directory / f"{step_dir.name}.incomplete"
        primary = self.process_index == 0
        if primary:
            _atomic_write_json(marker, {"step": step, "started_unix_ns": time.time_ns()})
            if step_dir.exists():  # re-saving a step: drop the old commit first
                (step_dir / "manifest.json").unlink(missing_ok=True)
                shutil.rmtree(step_dir)
            step_dir.mkdir(parents=True)
        sync_processes(f"jla-ckpt-start-{step}")
        trees = {"model": flatten_arrays(eqx.filter(model, eqx.is_array)), "optimizer": flatten_arrays(opt_state)}
        sidecar = {"data_state": dict(data_state), "rng": _key_payload(rng_key), "metadata": dict(metadata or {})}
        header = {"format": SCHEMA, "step": str(step)}

        if self.payload_format == "orbax":
            ocp = _orbax_module()
            for kind, arrays in trees.items():
                checkpointer = ocp.StandardCheckpointer()
                checkpointer.save((step_dir / f"{kind}.orbax").absolute(), dict(arrays))
                checkpointer.wait_until_finished()
                checkpointer.close()
            sync_processes(f"jla-ckpt-written-{step}")
            if primary:
                self._commit(step, sidecar)
            sync_processes(f"jla-ckpt-committed-{step}")
            return

        if self.payload_format == "safetensors":
            snapshot = {kind: _host_copies(arrays) for kind, arrays in trees.items()}

            def write() -> None:
                for kind, arrays in snapshot.items():
                    _save_safetensors(step_dir / f"{kind}.safetensors", arrays, {**header, "kind": kind})
                self._commit(step, sidecar)

        else:
            snapshot_chunks = {kind: _shard_chunks(arrays) for kind, arrays in trees.items()}
            suffix = f"p{self.process_index:05d}-of-{self.process_count:05d}"

            def write() -> None:
                for kind, (chunks, index) in snapshot_chunks.items():
                    _save_safetensors(step_dir / f"{kind}.{suffix}.safetensors", chunks, {**header, "kind": kind})
                    _atomic_write_json(step_dir / f"{kind}.{suffix}.index.json", index)
                if self.process_count == 1:
                    self._commit(step, sidecar)

        if self.asynchronous:
            submit_checkpoint_work(write)
            return
        write()
        if self.process_count > 1:
            sync_processes(f"jla-ckpt-written-{step}")
            if primary:
                self._commit(step, sidecar)
            sync_processes(f"jla-ckpt-committed-{step}")

    def _commit(self, step: int, sidecar: Mapping[str, Any]) -> None:
        step_dir = self.step_dir(step)
        _atomic_write_json(step_dir / "data_state.json", sidecar)
        artifacts = []
        for path in sorted(step_dir.iterdir()):
            if path.name == "manifest.json" or path.name.startswith("."):
                continue
            size = path.stat().st_size if path.is_file() else 0
            artifacts.append({"path": path.name, "size_bytes": size})
        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "complete": True,
            "step": step,
            "created_unix_ns": time.time_ns(),
            "payload_format": self.payload_format,
            "process_count": self.process_count,
            "artifacts": artifacts,
        }
        _atomic_write_json(step_dir / "manifest.json", manifest)
        (self.directory / f"{step_dir.name}.incomplete").unlink(missing_ok=True)
        _atomic_write_bytes(self.directory / "latest", f"step={step}\npath={step_dir.name}\n".encode())
        self._prune(step)
        logger.info("committed checkpoint %s", step_dir)

    def _prune(self, newest: int) -> None:
        if self.keep == 0:
            return
        committed = self.committed_steps()
        for step in committed[: -self.keep]:
            if step != newest:
                shutil.rmtree(self.step_dir(step), ignore_errors=True)
        # Leftovers of interrupted saves older than the newest commit.
        for marker in self.directory.glob("step_*.incomplete"):
            match = _STEP_DIR.match(marker.name.removesuffix(".incomplete"))
            if match and int(match.group(1)) < newest:
                shutil.rmtree(self.directory / marker.name.removesuffix(".incomplete"), ignore_errors=True)
                marker.unlink(missing_ok=True)

    # -- restoring -----------------------------------------------------------

    def restore(self, step: int | None = None, *, model: Any, opt_state: Any, strict: bool = True) -> Restored:
        """Load a committed step into templates whose array leaves carry the target shardings.

        ``model`` and ``opt_state`` are typically the freshly initialized,
        already placed state; the restored arrays take their shardings.
        """
        wait_for_pending_checkpoints()
        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError(f"no committed checkpoint in {self.directory}")
        step_dir = self.step_dir(step)
        manifest = _validated_manifest(step_dir)
        fmt = manifest["payload_format"]
        restored = {}
        for kind, template in (("model", eqx.filter(model, eqx.is_array)), ("optimizer", opt_state)):
            targets = flatten_arrays(template)
            if fmt == "orbax":
                restored[kind] = self._restore_orbax(step_dir / f"{kind}.orbax", targets)
            elif fmt == "safetensors":
                restored[kind] = _restore_from_files([step_dir / f"{kind}.safetensors"], targets, kind, strict, chunked=False)
            else:
                files = sorted(step_dir.glob(f"{kind}.p*-of-*.safetensors"))
                if not files:
                    raise FileNotFoundError(f"no {kind} shards in {step_dir}")
                restored[kind] = _restore_from_files(files, targets, kind, strict, chunked=True)
        sidecar = _read_json(step_dir / "data_state.json")
        params = _unflatten_into(eqx.filter(model, eqx.is_array), restored["model"])
        return Restored(
            step=step,
            model=eqx.combine(params, eqx.filter(model, eqx.is_array, inverse=True)),
            opt_state=_unflatten_into(opt_state, restored["optimizer"]),
            data_state=sidecar["data_state"],
            rng_key=_key_from_payload(sidecar["rng"]),
            manifest=manifest,
        )

    @staticmethod
    def _restore_orbax(path: Path, targets: Mapping[str, Any]) -> dict[str, jax.Array]:
        ocp = _orbax_module()
        abstract = {}
        for name, leaf in targets.items():
            shape, dtype, sharding = _target_spec(leaf)
            abstract[name] = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        checkpointer = ocp.StandardCheckpointer()
        try:
            restored = checkpointer.restore(path.absolute(), abstract)
        finally:
            checkpointer.close()
        for name, value in restored.items():
            _check_leaf(name, value.shape, str(value.dtype), abstract[name].shape, np.dtype(abstract[name].dtype))
        return dict(restored)
