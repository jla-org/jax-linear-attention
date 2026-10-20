# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``jax.distributed`` initialization and host-level coordination.

Settings are explicit by default. Launcher environment discovery
(:func:`discover_cluster_env`) runs only when the caller asks for it, e.g. with
``runtime.cluster_env = "slurm"``; it understands torchrun, SLURM and Open MPI
variables and assigns one accelerator per local process when a launcher starts
several processes per host.

All coordination helpers (:func:`process_any`, :func:`sync_processes`) are
collectives: every process must call them in the same order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "ClusterKind",
    "DistributedInit",
    "discover_cluster_env",
    "initialize_distributed",
    "process_any",
    "process_any_host",
    "resolve_distributed_init",
    "shutdown_distributed",
    "sync_processes",
]

ClusterKind = Literal["auto", "slurm", "openmpi", "torchrun"]


@dataclass(frozen=True)
class DistributedInit:
    """Arguments for :func:`jax.distributed.initialize`."""

    coordinator_address: str | None
    num_processes: int
    process_id: int
    local_device_ids: tuple[int, ...] | None = None
    cluster_detection_method: str | None = None
    initialization_timeout: int = 300
    heartbeat_timeout_seconds: int = 100
    shutdown_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if not 0 <= self.process_id < self.num_processes:
            raise ValueError(f"process_id must be in [0, {self.num_processes}), got {self.process_id}")
        if self.coordinator_address is None and self.cluster_detection_method is None:
            raise ValueError("coordinator_address is required without cluster detection")
        if self.coordinator_address is not None:
            _validate_address(self.coordinator_address)
        for name in ("initialization_timeout", "heartbeat_timeout_seconds", "shutdown_timeout_seconds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.local_device_ids is not None:
            ids = self.local_device_ids
            if any(i < 0 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"local_device_ids must be unique and non-negative, got {ids}")

    def initialize_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_processes": self.num_processes,
            "process_id": self.process_id,
            "initialization_timeout": self.initialization_timeout,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
        }
        if self.coordinator_address is not None:
            kwargs["coordinator_address"] = self.coordinator_address
        if self.local_device_ids is not None:
            kwargs["local_device_ids"] = list(self.local_device_ids)
        if self.cluster_detection_method is not None:
            kwargs["cluster_detection_method"] = self.cluster_detection_method
        return kwargs


def _validate_address(address: str) -> str:
    host, sep, port_text = address.rpartition(":")
    if not sep or not host or not port_text:
        raise ValueError(f"coordinator address must be host:port, got {address!r}")
    if host.startswith("[") != host.endswith("]"):
        raise ValueError(f"malformed bracketed IPv6 host in {address!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"coordinator port must be an integer, got {port_text!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"coordinator port must be in [1, 65535], got {port}")
    return address


def _env_int(env: Mapping[str, str], *names: str) -> int | None:
    for name in names:
        raw = env.get(name)
        if raw is not None and raw.strip():
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return None


def _env_str(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        raw = env.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


def _detect_kind(env: Mapping[str, str]) -> ClusterKind | None:
    if _env_str(env, "SLURM_PROCID", "SLURM_JOB_ID"):
        return "slurm"
    if _env_str(env, "OMPI_COMM_WORLD_RANK"):
        return "openmpi"
    if _env_str(env, "WORLD_SIZE", "RANK"):
        return "torchrun"
    return None


def discover_cluster_env(env: Mapping[str, str], kind: ClusterKind = "auto") -> dict[str, Any]:
    """Read launcher variables into partial :class:`DistributedInit` settings.

    Returns a mapping with any of ``coordinator_address``, ``num_processes``,
    ``process_id``, ``local_device_ids`` and ``cluster_detection_method``.
    SLURM and Open MPI rely on JAX's own cluster detection for the coordinator
    unless ``MASTER_ADDR``/``MASTER_PORT`` are present.
    """
    resolved = _detect_kind(env) if kind == "auto" else kind
    if resolved is None:
        raise ValueError("cluster_env='auto' found no SLURM, Open MPI or torchrun variables")
    names = {
        "torchrun": (("WORLD_SIZE",), ("RANK",), ("LOCAL_RANK",), ("LOCAL_WORLD_SIZE",)),
        "slurm": (("SLURM_NTASKS",), ("SLURM_PROCID",), ("SLURM_LOCALID",), ("SLURM_NTASKS_PER_NODE",)),
        "openmpi": (
            ("OMPI_COMM_WORLD_SIZE",),
            ("OMPI_COMM_WORLD_RANK",),
            ("OMPI_COMM_WORLD_LOCAL_RANK",),
            ("OMPI_COMM_WORLD_LOCAL_SIZE",),
        ),
    }[resolved]
    world, rank, local_rank_names, local_world_names = names
    out: dict[str, Any] = {}
    num_processes = _env_int(env, *world)
    process_id = _env_int(env, *rank)
    local_rank = _env_int(env, *local_rank_names)
    local_world = _env_int(env, *local_world_names)
    if resolved == "torchrun" and process_id is None and local_world in (None, 1):
        # NODE_RANK is a process rank only for one-process-per-node launches.
        process_id = _env_int(env, "NODE_RANK")
    if num_processes is not None:
        out["num_processes"] = num_processes
    if process_id is not None:
        out["process_id"] = process_id
    master_addr, master_port = _env_str(env, "MASTER_ADDR"), _env_str(env, "MASTER_PORT")
    if master_addr and master_port:
        out["coordinator_address"] = _validate_address(f"{master_addr}:{master_port}")
    elif resolved in ("slurm", "openmpi"):
        out["cluster_detection_method"] = "slurm" if resolved == "slurm" else "ompi"
    if local_rank is not None and local_world is not None and local_world > 1:
        visible = _env_str(env, "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        visible_count = len([part for part in visible.split(",") if part.strip()]) if visible else None
        if visible_count == 1:
            out["local_device_ids"] = (0,)  # launcher already narrowed visibility to one device
        elif visible_count is not None and local_rank >= visible_count:
            raise ValueError(f"local rank {local_rank} is outside {visible_count} visible local devices")
        else:
            out["local_device_ids"] = (local_rank,)
    return out


def resolve_distributed_init(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: Sequence[int] = (),
    cluster_env: Literal["none", "auto", "slurm", "openmpi", "torchrun"] = "none",
    env: Mapping[str, str] | None = None,
    initialization_timeout: int = 300,
    heartbeat_timeout_seconds: int = 100,
    shutdown_timeout_seconds: int = 300,
) -> DistributedInit:
    """Merge explicit settings with optional launcher discovery (explicit values win)."""
    discovered: dict[str, Any] = {}
    if cluster_env != "none":
        if env is None:
            raise ValueError("cluster_env discovery needs the environment mapping to read")
        discovered = discover_cluster_env(env, cluster_env)
    settings = {
        "coordinator_address": coordinator_address or discovered.get("coordinator_address"),
        "num_processes": num_processes if num_processes is not None else discovered.get("num_processes"),
        "process_id": process_id if process_id is not None else discovered.get("process_id"),
        "local_device_ids": tuple(local_device_ids) or discovered.get("local_device_ids"),
        "cluster_detection_method": None if coordinator_address else discovered.get("cluster_detection_method"),
    }
    missing = [name for name in ("num_processes", "process_id") if settings[name] is None]
    if settings["coordinator_address"] is None and settings["cluster_detection_method"] is None:
        missing.insert(0, "coordinator_address")
    if missing:
        raise ValueError("missing distributed settings: " + ", ".join(missing))
    return DistributedInit(
        initialization_timeout=initialization_timeout,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        **settings,
    )


def initialize_distributed(spec: DistributedInit) -> bool:
    """Call ``jax.distributed.initialize`` once; returns False if already initialized."""
    import jax

    if jax.distributed.is_initialized():
        return False
    jax.distributed.initialize(**spec.initialize_kwargs())
    return True


def shutdown_distributed() -> None:
    import jax

    if jax.distributed.is_initialized():
        jax.distributed.shutdown()


def process_any(flag: bool) -> bool:
    """Logical OR of ``flag`` over all processes (identity on one process)."""
    import jax

    if jax.process_count() <= 1:
        return bool(flag)
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(np.asarray(flag, dtype=np.uint8), tiled=False)
    return bool(np.asarray(gathered, dtype=np.uint8).any())


def sync_processes(name: str) -> None:
    """Barrier across processes (no-op on one process)."""
    import jax

    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(name)


_HOST_ANY_KEYS: list[str] = []  # this process's keys of the last process_any_host calls (deleted two calls later)
_HOST_ANY_CALLS = [0]  # calls so far: every process makes the same calls in the same order
_HOST_ANY_WARNED: list[bool] = []  # the fallback to process_any is logged once


def process_any_host(flag: bool, *, timeout_seconds: float = 600.0) -> bool:
    """Logical OR of ``flag`` over all processes through the distributed runtime's key-value store.

    :func:`process_any` is a device collective: it queues behind the work already dispatched to the devices, so it
    waits for that work to finish. This one runs on the hosts only (each process writes its flag under the number
    of the call and reads every other process's), so it returns while the devices are still busy: the trainer's
    stop check uses it after dispatching a step ahead (``train.dispatch_ahead``). Like :func:`process_any` it is a
    collective: every process makes the same calls in the same order. Identity on one process; a process without
    the runtime's client (no ``jax.distributed.initialize``) uses :func:`process_any`.
    """
    import jax

    if jax.process_count() <= 1:
        return bool(flag)
    try:  # the runtime's client is not public API: fall back to the device collective where it moved
        from jax._src import distributed

        client = getattr(distributed.global_state, "client", None)
    except (ImportError, AttributeError):
        client = None
    if client is not None and not all(hasattr(client, name) for name in
                                      ("key_value_set", "blocking_key_value_get", "key_value_delete")):
        client = None
    if client is None:
        if not _HOST_ANY_WARNED:
            _HOST_ANY_WARNED.append(True)
            import logging

            logging.getLogger(__name__).warning(
                "process_any_host: no key-value client of jax.distributed in this process or JAX version (%s); "
                "using the device collective process_any, which waits for the work already dispatched",
                jax.__version__)
        return process_any(flag)
    _HOST_ANY_CALLS[0] += 1
    prefix = f"jla/process_any_host/{_HOST_ANY_CALLS[0]}/"
    own = prefix + str(jax.process_index())
    client.key_value_set(own, "1" if flag else "0")
    timeout_ms = int(timeout_seconds * 1000)
    values = [client.blocking_key_value_get(prefix + str(p), timeout_ms) for p in range(jax.process_count())]
    # A process that finished this call has read every key of the previous one, so every process has finished the
    # call before that: its keys can go.
    _HOST_ANY_KEYS.append(own)
    while len(_HOST_ANY_KEYS) > 2:
        client.key_value_delete(_HOST_ANY_KEYS.pop(0))
    return any(value == "1" for value in values)
