# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compile preflight, sharding audits and small runtime utilities.

:func:`compile_preflight` lowers and compiles a jitted callable ahead of time
and reports each stage (lowering, compilation, memory and cost analyses)
separately; the compiled executable is returned so the first training step
does not compile again. The sharding helpers summarize where every array leaf
lives and write a reproducible ``sharding_dump.json``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import operator
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jla.training.performance import compiled_cost_bytes_accessed

__all__ = [
    "audit_parameter_sharding",
    "compile_preflight",
    "compiled_memory_summary",
    "merge_xla_flags",
    "serializable_tree_summary",
    "sharding_mismatches",
    "tree_sharding_summary",
    "write_sharding_dump",
]


def _exact_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return int(parsed)


def _flag_key(flag: str) -> str:
    return flag.split("=", 1)[0].strip()


def _flag_parts(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    return [str(part).strip() for part in value if str(part).strip()]


def _dedupe_last(flags: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for flag in reversed(flags):
        key = _flag_key(flag)
        if key and key not in seen:
            seen.add(key)
            output.append(flag)
    output.reverse()
    return output


def merge_xla_flags(
    existing: str | Sequence[str] | None,
    new_flags: str | Sequence[str] | None,
    *,
    force_keys: set[str] | frozenset[str] | None = None,
) -> str:
    """Merge XLA flag strings; existing values win unless their key is in ``force_keys``.

    Within each input the last occurrence of a key wins.
    """
    forced = {_flag_key(key) for key in (force_keys or ())}
    output = [flag for flag in _dedupe_last(_flag_parts(existing)) if _flag_key(flag) not in forced]
    keys = {_flag_key(flag) for flag in output}
    for flag in _dedupe_last(_flag_parts(new_flags)):
        if _flag_key(flag) not in keys:
            output.append(flag)
            keys.add(_flag_key(flag))
    return " ".join(output)


def compiled_memory_summary(stats: Any) -> dict[str, int | float]:
    """Device/host memory totals derived from a compiled executable's ``memory_analysis``.

    Missing fields count as zero (the analysis is best-effort); malformed or
    negative fields raise.
    """
    if stats is None:
        return {}

    def get(name: str) -> int:
        value = stats.get(name, 0) if isinstance(stats, Mapping) else getattr(stats, name, 0)
        if value is None:
            return 0
        if isinstance(value, bool):
            raise TypeError(f"memory field {name} must be an integer, not bool")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"memory field {name} must be >= 0, got {parsed}")
        return parsed

    gib = float(1 << 30)
    argument, output = get("argument_size_in_bytes"), get("output_size_in_bytes")
    temp, alias = get("temp_size_in_bytes"), get("alias_size_in_bytes")
    device_total = max(argument + output + temp - alias, 0)
    peak = get("peak_memory_in_bytes") or device_total
    host_total = max(
        get("host_argument_size_in_bytes")
        + get("host_output_size_in_bytes")
        + get("host_temp_size_in_bytes")
        - get("host_alias_size_in_bytes"),
        0,
    )
    return {
        "argument_size_in_bytes": argument,
        "output_size_in_bytes": output,
        "temp_size_in_bytes": temp,
        "alias_size_in_bytes": alias,
        "total_size_in_bytes": device_total,
        "total_size_gib": device_total / gib,
        "peak_memory_in_bytes": peak,
        "peak_memory_gib": peak / gib,
        "host_total_size_in_bytes": host_total,
        "total_including_host_size_in_bytes": device_total + host_total,
        "generated_code_size_gib": get("generated_code_size_in_bytes") / gib,
    }


def _leaf_bytes(leaf: Any) -> int:
    if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.extended):  # typed PRNG keys
        return 0
    return _leaf_numel(leaf) * np.dtype(leaf.dtype).itemsize


def _signature(value: Any) -> dict[str, Any]:
    """Array-leaf count and bytes of one argument, plus its first few leaf shapes."""
    leaves = [leaf for leaf in jax.tree.leaves(value) if hasattr(leaf, "shape") and hasattr(leaf, "dtype")]
    return {
        "array_leaves": len(leaves),
        "bytes": sum(_leaf_bytes(leaf) for leaf in leaves),
        "first_leaves": [{"shape": [int(d) for d in leaf.shape], "dtype": str(leaf.dtype)} for leaf in leaves[:8]],
    }


def _analysis(compiled: Any, method: str) -> Any:
    owner, seen = compiled, set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        fn = getattr(owner, method, None)
        if callable(fn):
            return fn()
        owner = getattr(owner, "compiled", None)
    raise AttributeError(f"{type(compiled).__name__} exposes no {method}()")


def compile_preflight(
    name: str,
    fn: Any,
    *args: Any,
    compiler_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Lower and compile ``fn(*args)`` ahead of time.

    Returns ``(report, compiled)``; ``compiled`` is ``None`` unless
    ``report["status"] == "compiled"``. Statuses: ``skipped`` (no ``lower``),
    ``lower_failed``, ``compile_failed``, ``compiled``. Memory and cost
    analyses are best-effort and recorded separately from compile success.
    """
    report: dict[str, Any] = {
        "name": name,
        "inputs": [_signature(arg) for arg in args],
        "compiler_options": serializable_tree_summary(dict(compiler_options or {})),
    }
    lower = getattr(fn, "lower", None)
    if not callable(lower):
        return {**report, "status": "skipped", "reason": "callable has no lower()"}, None
    started = time.perf_counter()
    try:
        lowered = lower(*args)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {**report, "status": "lower_failed", "error": f"{type(exc).__name__}: {exc}", "total_seconds": elapsed}, None
    lowered_at = time.perf_counter()
    try:
        compiled = lowered.compile(dict(compiler_options)) if compiler_options else lowered.compile()
    except Exception as exc:
        finished = time.perf_counter()
        return {
            **report,
            "status": "compile_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "lower_seconds": lowered_at - started,
            "total_seconds": finished - started,
        }, None
    compiled_at = time.perf_counter()
    try:
        memory: dict[str, Any] = compiled_memory_summary(_analysis(compiled, "memory_analysis"))
    except Exception as exc:  # backend dependent; compilation itself succeeded
        memory = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        raw_cost = _analysis(compiled, "cost_analysis")
        cost: Any = serializable_tree_summary(raw_cost)
        bytes_accessed = compiled_cost_bytes_accessed(raw_cost)
    except Exception as exc:
        cost, bytes_accessed = {"error": f"{type(exc).__name__}: {exc}"}, None
    return {
        **report,
        "status": "compiled",
        "lower_seconds": lowered_at - started,
        "compile_seconds": compiled_at - lowered_at,
        "total_seconds": compiled_at - started,
        "memory": memory,
        "cost_analysis": cost,
        # Compiler estimate of logical bytes, not a hardware measurement.
        "cost_analysis_bytes_accessed": bytes_accessed,
    }, compiled


def _leaf_numel(leaf: Any) -> int:
    return int(math.prod(int(d) for d in getattr(leaf, "shape", ()) or ()))


def _is_replicated(sharding: Any) -> bool:
    if sharding is None:
        return True
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return bool(getattr(sharding, "is_fully_replicated", False))
    return all(part is None for part in tuple(spec))


def _path(path: tuple[Any, ...]) -> str:
    return jax.tree_util.keystr(path)


def audit_parameter_sharding(tree: Any, *, min_elements: int = 4096, top_k: int = 20) -> dict[str, Any]:
    """Replicated element fraction and the largest replicated leaves."""
    min_elements = _exact_int("min_elements", min_elements, minimum=1)
    top_k = _exact_int("top_k", top_k, minimum=0)
    total = replicated = 0
    leaves: list[dict[str, Any]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
            continue
        size = _leaf_numel(leaf)
        total += size
        sharding = getattr(leaf, "sharding", None)
        if _is_replicated(sharding):
            replicated += size
            if size >= min_elements:
                leaves.append(
                    {"path": _path(path), "shape": tuple(leaf.shape), "dtype": str(leaf.dtype), "elements": size, "sharding": str(sharding)}
                )
    leaves.sort(key=lambda item: item["elements"], reverse=True)
    return {
        "total_elements": total,
        "replicated_elements": replicated,
        "replicated_fraction": replicated / total if total else 0.0,
        "min_elements": min_elements,
        "large_replicated_leaves": leaves[:top_k] if top_k else leaves,
    }


def _sharding_json(sharding: Any) -> dict[str, Any]:
    if sharding is None:
        return {"kind": "none", "spec": None}
    mesh = getattr(sharding, "mesh", None)
    return {
        "kind": type(sharding).__name__,
        "spec": str(getattr(sharding, "spec", None)),
        "mesh": None if mesh is None else {str(k): int(v) for k, v in mesh.shape.items()},
        "memory_kind": str(getattr(sharding, "memory_kind", None)),
    }


def tree_sharding_summary(tree: Any, *, min_elements: int = 1, top_k: int = 0) -> list[dict[str, Any]]:
    """JSON-safe ``{path, shape, dtype, elements, sharding}`` records, largest first."""
    min_elements = _exact_int("min_elements", min_elements, minimum=1)
    records = [
        {
            "path": _path(path),
            "shape": tuple(int(d) for d in leaf.shape),
            "dtype": str(leaf.dtype),
            "elements": _leaf_numel(leaf),
            "sharding": _sharding_json(getattr(leaf, "sharding", None)),
        }
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype") and _leaf_numel(leaf) >= min_elements
    ]
    records.sort(key=lambda item: item["elements"], reverse=True)
    return records[:top_k] if top_k else records


def _equivalent(left: Any, right: Any, ndim: int) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return bool(left.is_equivalent_to(right, ndim))
    except (AttributeError, TypeError, ValueError):
        return left == right


def sharding_mismatches(left: Any, right: Any, *, min_elements: int = 1, top_k: int = 20) -> list[dict[str, Any]]:
    """Leaves of two same-structure trees whose shape, dtype or sharding differ."""
    min_elements = _exact_int("min_elements", min_elements, minimum=1)
    if jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(right):
        return [{"path": "<tree>", "reason": "tree-structure-mismatch"}]
    out = []
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(left)[0], jax.tree.leaves(right)):
        if not (hasattr(a, "shape") and hasattr(b, "shape")):
            continue
        reasons = []
        if tuple(a.shape) != tuple(b.shape):
            reasons.append("shape")
        if jnp.dtype(a.dtype) != jnp.dtype(b.dtype):
            reasons.append("dtype")
        if not _equivalent(getattr(a, "sharding", None), getattr(b, "sharding", None), len(a.shape)):
            reasons.append("sharding")
        elements = max(_leaf_numel(a), _leaf_numel(b))
        if reasons and elements >= min_elements:
            out.append(
                {
                    "path": _path(path),
                    "reason": ",".join(reasons),
                    "elements": elements,
                    "left": {"shape": tuple(a.shape), "dtype": str(a.dtype), "sharding": _sharding_json(getattr(a, "sharding", None))},
                    "right": {"shape": tuple(b.shape), "dtype": str(b.dtype), "sharding": _sharding_json(getattr(b, "sharding", None))},
                }
            )
    out.sort(key=lambda item: item["elements"], reverse=True)
    return out[:top_k] if top_k else out


def write_sharding_dump(
    output_dir: str | Path,
    *,
    mesh: Any,
    model: Any,
    optimizer: Any,
    batch_sharding: Any = None,
    audit: Mapping[str, Any] | None = None,
    min_elements: int = 4096,
) -> Path:
    """Atomically write ``sharding_dump.json`` describing the mesh and every large leaf."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = serializable_tree_summary(
        {
            "mesh": {"shape": {str(k): int(v) for k, v in mesh.shape.items()}, "axis_names": [str(a) for a in mesh.axis_names]},
            "batch_sharding": _sharding_json(batch_sharding),
            "model": tree_sharding_summary(model, min_elements=min_elements),
            "optimizer": tree_sharding_summary(optimizer, min_elements=min_elements),
            "audit": dict(audit or {}),
        }
    )
    destination = directory / "sharding_dump.json"
    fd, tmp = tempfile.mkstemp(prefix=".sharding_dump.", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return destination


def serializable_tree_summary(value: Any) -> Any:
    """Bounded, deterministic JSON-safe summary; arrays become shape/dtype metadata, never values."""
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return serializable_tree_summary(value.item())
    if isinstance(value, Enum):
        return serializable_tree_summary(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": type(value).__name__, "size_bytes": len(value)}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"kind": type(value).__name__, "shape": [int(d) for d in value.shape], "dtype": str(value.dtype), "elements": _leaf_numel(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: serializable_tree_summary(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): serializable_tree_summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable_tree_summary(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [serializable_tree_summary(item) for item in sorted(value, key=lambda item: (type(item).__name__, repr(item)))]
    return str(value)
