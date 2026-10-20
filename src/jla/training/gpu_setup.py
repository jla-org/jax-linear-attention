# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accelerator inventory before JAX starts, opt-in XLA tuning tiers and capacity planning.

XLA reads ``XLA_FLAGS`` when its backend initializes, so GPU tuning must be
decided before the first JAX device call. :func:`detect_prejax_nvidia` reads
``nvidia-smi`` and applies CUDA's visibility rules to the visibility string
the caller passes (``CUDA_VISIBLE_DEVICES`` of the launch): ordering, UUID
prefixes (only when unique), MIG instances (whole-GPU capacity is never
reported for a partition) and the stop-at-first-invalid-token behaviour.
Unknown SKUs stay ``GENERIC_GPU`` rather than being guessed.

Tuning tiers (:func:`xla_flags_for`): ``0`` none, ``1`` latency-hiding
scheduler, ``2`` adds cuBLASLt, while-loop double buffering and (multi-GPU)
pipelined collectives.
"""

from __future__ import annotations

import csv
import functools
import logging
import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "DeviceSpecs",
    "GPUProfile",
    "detect_device",
    "detect_prejax_nvidia",
    "get_optimal_batch_size",
    "nvidia_inventory",
    "nvidia_specs_from_inventory",
    "query_free_vram_gb",
    "visible_nvidia_inventory",
    "xla_flags_for",
]

logger = logging.getLogger(__name__)


class GPUProfile(Enum):
    CPU = "cpu"
    GENERIC_GPU = "generic_gpu"
    A10 = "a10"
    ADA = "ada"
    A100 = "a100"
    H100 = "h100"
    H200 = "h200"
    B200 = "b200"
    B300 = "b300"
    RTX_PRO_6000 = "rtx_pro_6000"
    TPU = "tpu"


@dataclass
class DeviceSpecs:
    """Visible accelerator inventory (minimum capacity across devices)."""

    name: str
    platform: str
    profile: GPUProfile
    memory_gb: float
    num_devices: int
    compute_capability: tuple[int, int] = (0, 0)
    sm_count: int = 0
    has_nvlink: bool = False
    memory_bandwidth_gb_s: float = 0.0
    bf16_tflops: float = 0.0
    fp8_tflops: float = 0.0
    heterogeneous: bool = False
    is_partitioned: bool = False


# Used only after an exact SKU match.
_PROFILE_METADATA: dict[GPUProfile, dict[str, Any]] = {
    GPUProfile.A10: {"cc": (8, 6), "bandwidth": 600.0, "bf16": 125.0},
    GPUProfile.ADA: {"cc": (8, 9), "bandwidth": 1008.0, "bf16": 330.0},
    GPUProfile.A100: {"cc": (8, 0), "bandwidth": 1935.0, "bf16": 312.0},
    GPUProfile.H100: {"cc": (9, 0), "bandwidth": 3350.0, "bf16": 989.0, "fp8": 1979.0},
    GPUProfile.H200: {"cc": (9, 0), "bandwidth": 4800.0, "bf16": 989.0, "fp8": 1979.0},
    GPUProfile.B200: {"cc": (10, 0), "bandwidth": 8000.0, "bf16": 2250.0, "fp8": 4500.0},
    GPUProfile.B300: {"cc": (10, 3), "bandwidth": 8000.0, "bf16": 2500.0, "fp8": 5000.0},
    GPUProfile.RTX_PRO_6000: {"cc": (12, 0), "bandwidth": 1597.0, "bf16": 1000.0, "fp8": 2000.0},
}
_NVIDIA_FIELDS = ("index", "uuid", "name", "memory.total", "memory.free", "multiprocessor_count", "compute_cap")
_UNAVAILABLE = {"-", "n/a", "na", "not available", "not supported", "unknown error"}
_MIG_PARENT = re.compile(r"^MIG-(GPU-[0-9a-fA-F-]+)(?:/\d+/\d+)?$", re.I)


def _clean(value: str) -> str:
    cleaned = value.strip()
    return "" if cleaned.strip("[]").strip().lower() in _UNAVAILABLE else cleaned


def _run_nvidia_query(query_fields: tuple[str, ...]) -> list[dict[str, str]]:
    """One machine-readable ``nvidia-smi --query-gpu`` call (empty on failure)."""
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(query_fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    rows = []
    for values in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(values) != len(query_fields):
            continue
        row = {name: cleaned for name, value in zip(query_fields, values) if (cleaned := _clean(value))}
        if row:
            rows.append(row)
    return rows


@functools.lru_cache(maxsize=1)
def nvidia_inventory() -> tuple[dict[str, str], ...]:
    """All physical GPUs; optional fields are dropped progressively on old drivers."""
    queries = (
        _NVIDIA_FIELDS,
        tuple(f for f in _NVIDIA_FIELDS if f != "compute_cap"),
        ("index", "uuid", "name", "memory.total", "memory.free"),
        ("index", "uuid", "name"),
    )
    try:
        for query in queries:
            rows = _run_nvidia_query(query)
            if rows:
                return tuple(rows)
    except (OSError, subprocess.SubprocessError):
        pass
    return ()


def _unique_uuid(token: str, inventory: tuple[dict[str, str], ...]) -> dict[str, str] | None:
    lowered = token.lower()
    matches = [row for row in inventory if row.get("uuid", "").lower().startswith(lowered)]
    return matches[0] if len(matches) == 1 else None


def visible_nvidia_inventory(visible: str | None, inventory: tuple[dict[str, str], ...] | None = None) -> tuple[dict[str, str], ...]:
    """Inventory rows in CUDA enumeration order for a ``CUDA_VISIBLE_DEVICES`` value."""
    rows = nvidia_inventory() if inventory is None else inventory
    if visible is None or visible.strip().lower() == "all":
        return rows
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    tokens = visible.split(",")
    for position, raw in enumerate(tokens):
        token = raw.strip()
        if not token or token.lower() in ("-1", "none", "void"):
            break
        match: dict[str, str] | None
        if token.isdigit():
            match = next((row for row in rows if row.get("index") == token), None)
        elif token.upper().startswith("MIG-"):
            parent_match = _MIG_PARENT.fullmatch(token)
            parent = _unique_uuid(parent_match.group(1), rows) if parent_match else None
            match = None
            if parent is not None:
                match = {k: v for k, v in parent.items() if k not in ("memory.total", "memory.free", "multiprocessor_count")}
                match.update({"uuid": token, "mig.partition": "1"})
            if position + 1 < len(tokens):
                logger.warning("CUDA enumerates a single MIG instance; identifiers after %r are ignored", token)
        else:
            match = _unique_uuid(token, rows)
        if match is None:
            logger.warning("GPU visibility enumeration stopped at unresolved identifier %r", token)
            break
        identity = match.get("uuid") or match.get("index", token)
        if identity not in seen:
            seen.add(identity)
            selected.append(match)
        if token.upper().startswith("MIG-"):
            break
    return tuple(selected)


@functools.lru_cache(maxsize=16)
def _detect_nvlink(indices: tuple[str, ...]) -> bool:
    labels = tuple(f"GPU{index}" for index in indices if index.isdigit())
    if len(labels) < 2:
        return False
    try:
        result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    header = next((tokens for tokens in lines if all(label in tokens for label in labels)), None)
    if header is None:
        return False
    table = {tokens[0]: tokens for tokens in lines if tokens[0] in labels}
    for source in labels:
        row = table.get(source, [])
        for target in labels:
            column = header.index(target) + 1
            if source != target and column < len(row) and re.fullmatch(r"NV\d+", row[column], re.I):
                return True
    return False


def _classify(name: str) -> GPUProfile:
    lowered = name.lower()
    patterns = (
        (GPUProfile.RTX_PRO_6000, ("rtx pro 6000",)),
        (GPUProfile.B300, ("gb300", "b300")),
        (GPUProfile.B200, ("gb200", "b200")),
        (GPUProfile.H200, ("h200",)),
        (GPUProfile.H100, ("h100",)),
        (GPUProfile.A100, ("a100",)),
        (GPUProfile.A10, ("a10",)),
        (GPUProfile.ADA, ("l40", "l4", "rtx 4090", "rtx 6000 ada")),
    )
    for profile, tokens in patterns:
        if any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered) for token in tokens):
            return profile
    return GPUProfile.GENERIC_GPU


def _capability(value: str | None) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", value or "")
    return (int(match.group(1)), int(match.group(2) or 0)) if match else (0, 0)


def _positive(row: Mapping[str, str], name: str, scale: float = 1.0) -> float | None:
    try:
        value = float(row.get(name, "")) / scale
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def nvidia_specs_from_inventory(rows: tuple[dict[str, str], ...], *, num_devices: int) -> DeviceSpecs:
    """Specs for visible rows: heterogeneous sets report the minimum capacity and a generic profile."""
    if not rows:
        raise ValueError("at least one NVIDIA inventory row is required")
    names = tuple(row.get("name", "Unknown NVIDIA GPU") for row in rows)
    capabilities = tuple(_capability(row.get("compute_cap")) for row in rows)
    partitioned = any(row.get("mig.partition") == "1" for row in rows)
    heterogeneous = len({(n.lower(), c, r.get("memory.total", "")) for n, c, r in zip(names, capabilities, rows)}) > 1
    profiles = {_classify(name) for name in names}
    profile = profiles.pop() if len(profiles) == 1 and not heterogeneous else GPUProfile.GENERIC_GPU
    metadata = _PROFILE_METADATA.get(profile, {})
    known = [c for c in capabilities if c != (0, 0)]
    capability = metadata.get("cc") or (min(known) if len(known) == len(rows) else (0, 0))
    bandwidth = float(metadata.get("bandwidth", 0.0))
    if profile == GPUProfile.A100 and all("sxm" in name.lower() for name in names):
        bandwidth = 2039.0
    bf16, fp8 = float(metadata.get("bf16", 0.0)), float(metadata.get("fp8", 0.0))
    if partitioned:
        bandwidth = bf16 = fp8 = 0.0
    memories = [_positive(row, "memory.total", 1024.0) for row in rows]
    sm_counts = [_positive(row, "multiprocessor_count") for row in rows]
    unique_names = tuple(dict.fromkeys(names))
    if heterogeneous:
        logger.warning("heterogeneous GPUs visible (%s); reporting minimum capacity", ", ".join(unique_names))
    return DeviceSpecs(
        name=unique_names[0] if len(unique_names) == 1 else "Heterogeneous NVIDIA GPUs (" + ", ".join(unique_names) + ")",
        platform="gpu",
        profile=profile,
        memory_gb=round(min(memories), 2) if all(m is not None for m in memories) else 0.0,  # type: ignore[type-var]
        num_devices=num_devices,
        compute_capability=capability,
        sm_count=int(min(sm_counts)) if all(s is not None for s in sm_counts) else 0,  # type: ignore[type-var]
        has_nvlink=False if partitioned else _detect_nvlink(tuple(row.get("index", "") for row in rows)),
        memory_bandwidth_gb_s=bandwidth,
        bf16_tflops=bf16,
        fp8_tflops=fp8,
        heterogeneous=heterogeneous,
        is_partitioned=partitioned,
    )


def detect_prejax_nvidia(visible: str | None) -> DeviceSpecs | None:
    """NVIDIA specs without importing JAX (``None`` when no GPU is visible)."""
    rows = visible_nvidia_inventory(visible)
    return nvidia_specs_from_inventory(rows, num_devices=len(rows)) if rows else None


def detect_device(visible: str | None = None) -> DeviceSpecs:
    """Specs of the devices JAX sees (initializes the JAX backend)."""
    import jax

    devices = jax.devices()
    first = devices[0]
    platform = str(first.platform).lower()
    kind = str(getattr(first, "device_kind", "") or "")
    if platform == "cpu":
        return DeviceSpecs(name="CPU", platform="cpu", profile=GPUProfile.CPU, memory_gb=0.0, num_devices=len(devices))
    if platform == "tpu":
        return DeviceSpecs(name=kind or "TPU", platform="tpu", profile=GPUProfile.TPU, memory_gb=0.0, num_devices=len(devices))
    rows = visible_nvidia_inventory(visible)
    if rows:
        return nvidia_specs_from_inventory(rows, num_devices=len(devices))
    profile = _classify(kind)
    metadata = _PROFILE_METADATA.get(profile, {})
    return DeviceSpecs(
        name=kind or "GPU",
        platform="gpu",
        profile=profile,
        memory_gb=0.0,
        num_devices=len(devices),
        compute_capability=metadata.get("cc", (0, 0)),
        memory_bandwidth_gb_s=float(metadata.get("bandwidth", 0.0)),
        bf16_tflops=float(metadata.get("bf16", 0.0)),
        fp8_tflops=float(metadata.get("fp8", 0.0)),
    )


def xla_flags_for(specs: DeviceSpecs | None, level: int, *, command_buffer: str | None = None) -> list[str]:
    """XLA GPU flags for tuning tier ``level``; ``command_buffer=""`` disables command buffers."""
    if level not in (0, 1, 2):
        raise ValueError(f"tuning level must be 0, 1 or 2, got {level}")
    flags: list[str] = []
    if specs is not None and specs.platform == "gpu":
        if level >= 1:
            flags.append("--xla_gpu_enable_latency_hiding_scheduler=true")
        if level >= 2:
            flags += ["--xla_gpu_enable_cublaslt=true", "--xla_gpu_enable_while_loop_double_buffering=true"]
            if specs.num_devices > 1:
                flags += [
                    "--xla_gpu_enable_pipelined_all_reduce=true",
                    "--xla_gpu_enable_pipelined_all_gather=true",
                    "--xla_gpu_enable_pipelined_reduce_scatter=true",
                ]
    if command_buffer is not None:
        flags.append(f"--xla_gpu_enable_command_buffer={command_buffer}")
    return flags


def query_free_vram_gb(visible: str | None) -> float | None:
    """Minimum free memory over visible whole GPUs; ``None`` when telemetry is incomplete."""
    rows = visible_nvidia_inventory(visible)
    if not rows or any(row.get("mig.partition") == "1" for row in rows):
        return None
    try:
        live = _run_nvidia_query(("index", "uuid", "memory.free"))
    except (OSError, subprocess.SubprocessError):
        return None
    free = []
    for row in rows:
        uuid = row.get("uuid", "").lower()
        current = next(
            (c for c in live if (uuid and c.get("uuid", "").lower() == uuid) or (not uuid and c.get("index") == row.get("index"))),
            None,
        )
        if current is None:
            return None
        try:
            value = float(current.get("memory.free", "")) / 1024.0
        except ValueError:
            return None
        if not math.isfinite(value) or value < 0:
            return None
        free.append(value)
    return min(free)


_OPTIMIZER_STATE_BYTES = {"adamw": 8.0, "nadamw": 8.0, "ademamix": 12.0, "lion": 4.0, "sgd": 4.0, "adafactor": 4.0, "muon": 8.0}


def get_optimal_batch_size(
    specs: DeviceSpecs,
    model_params_m: float,
    seq_len: int,
    *,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    optimizer: str = "adamw",
    param_dtype_bytes: int = 4,
    activation_checkpointing: bool = True,
    fsdp_shards: int = 1,
    reserve_gb: float | None = None,
    usable_memory_fraction: float = 0.85,
    memory_gb: float | None = None,
) -> dict[str, int]:
    """Rough per-device microbatch that fits (``training`` and ``inference``).

    A planning estimate, not an OOM guarantee: parameters, fp32 gradients and
    optimizer state are sharded over ``fsdp_shards``; activations use
    ``hidden_size * num_layers`` per token; logits are assumed chunked (one
    ``vocab_size`` row per token). Zero means persistent state alone does not fit.
    """
    if optimizer not in _OPTIMIZER_STATE_BYTES:
        raise ValueError(f"unsupported optimizer {optimizer!r}")
    if not 0.0 < usable_memory_fraction <= 1.0:
        raise ValueError("usable_memory_fraction must be in (0, 1]")
    for name, value in (("seq_len", seq_len), ("hidden_size", hidden_size), ("num_layers", num_layers), ("fsdp_shards", fsdp_shards)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    capacity = specs.memory_gb if memory_gb is None else memory_gb
    if not math.isfinite(capacity) or capacity <= 0:
        return {"training": 0, "inference": 0}
    reserve = max(1.0, min(capacity * 0.05, 8.0)) if reserve_gb is None else reserve_gb
    params = model_params_m * 1e6
    state_gb = params * (param_dtype_bytes + 4.0 + _OPTIMIZER_STATE_BYTES[optimizer]) / fsdp_shards / 1e9
    weights_gb = params * param_dtype_bytes / fsdp_shards / 1e9
    usable = capacity * usable_memory_fraction
    per_token = hidden_size * num_layers * 2 * (2.0 if activation_checkpointing else 10.0) + vocab_size * 2
    per_sample_gb = max(seq_len * per_token / 1e9, 0.05)
    inference_sample_gb = max(seq_len * (hidden_size * num_layers * 2 * 0.2 + vocab_size * 2) / 1e9, 0.05)
    train_room = usable - state_gb - reserve
    infer_room = usable - weights_gb - reserve
    return {
        "training": max(0, min(int(train_room / per_sample_gb), 256)) if train_room > 0 else 0,
        "inference": max(0, min(int(infer_room / inference_sample_gb), 1024)) if infer_room > 0 else 0,
    }
