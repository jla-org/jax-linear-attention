# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Process start-up driven by :class:`jla.config.RuntimeConfig`.

Call :func:`launch` once, before anything touches a JAX device:

1. XLA flags (tuning tier for the visible NVIDIA GPUs, determinism, explicit
   extra flags) are merged into ``XLA_FLAGS``; flags already present in the
   environment win, so a launcher can always override a config;
2. JAX options: PRNG implementation, matmul precision, persistent
   compilation cache;
3. ``jax.distributed.initialize`` when ``runtime.distributed`` is set
   (environment discovery only with ``runtime.cluster_env``).

Nothing else in jla reads or writes the environment.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jla.distributed.init import DistributedInit, initialize_distributed, resolve_distributed_init
from jla.training.gpu_setup import detect_prejax_nvidia, xla_flags_for
from jla.training.runtime import merge_xla_flags

__all__ = ["LaunchInfo", "configure_jax", "launch", "prepare_xla_flags", "xla_flag_list"]

logger = logging.getLogger(__name__)

DETERMINISTIC_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


@dataclass(frozen=True)
class LaunchInfo:
    xla_flags: str
    distributed: DistributedInit | None


def xla_flag_list(runtime: Any, *, visible_devices: str | None) -> list[str]:
    """Flags implied by ``runtime`` for the NVIDIA GPUs visible under ``visible_devices``."""
    flags: list[str] = []
    if runtime.xla_tuning_level:
        specs = detect_prejax_nvidia(visible_devices)
        if specs is None:
            logger.warning("runtime.xla_tuning_level=%d but no NVIDIA GPU is visible; no tuning flags", runtime.xla_tuning_level)
        flags += xla_flags_for(specs, runtime.xla_tuning_level)
    if runtime.deterministic:
        flags.append(DETERMINISTIC_FLAG)
    flags += list(runtime.xla_flags)
    return flags


def prepare_xla_flags(runtime: Any, environ: MutableMapping[str, str]) -> str:
    """Merge config flags into ``environ["XLA_FLAGS"]`` (existing keys win); returns the result."""
    flags = xla_flag_list(runtime, visible_devices=environ.get("CUDA_VISIBLE_DEVICES"))
    merged = merge_xla_flags(environ.get("XLA_FLAGS", ""), flags)
    if merged:
        environ["XLA_FLAGS"] = merged
    warn_slow_xla_flags(merged)
    return merged


def configure_jax(runtime: Any) -> None:
    """JAX options that must be set before the first computation."""
    import jax

    jax.config.update("jax_default_prng_impl", runtime.prng_impl)
    if runtime.matmul_precision != "default":
        jax.config.update("jax_default_matmul_precision", runtime.matmul_precision)
    if runtime.compile_cache_dir:
        cache = Path(runtime.compile_cache_dir).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def launch(runtime: Any, environ: MutableMapping[str, str] | None = None) -> LaunchInfo:
    """Apply ``runtime`` to this process (see the module docstring)."""
    env = os.environ if environ is None else environ
    flags = prepare_xla_flags(runtime, env)
    configure_jax(runtime)
    spec = None
    if runtime.distributed:
        spec = resolve_distributed_init(
            coordinator_address=runtime.coordinator_address,
            num_processes=runtime.num_processes,
            process_id=runtime.process_id,
            local_device_ids=runtime.local_device_ids,
            cluster_env=runtime.cluster_env,
            env=env if runtime.cluster_env != "none" else None,
            initialization_timeout=runtime.initialization_timeout,
        )
        initialize_distributed(spec)
    return LaunchInfo(xla_flags=flags, distributed=spec)


def warn_slow_xla_flags(flags: str) -> bool:
    """Log a warning for XLA flags measured to slow training down badly; True if one was found.

    ``--xla_gpu_autotune_level=0`` skips GEMM and fusion autotuning. It saves a few seconds of
    compile time, and it made the 1.3B Mamba-2 train step 15x slower on an H100
    (benchmarks/fp8_1p3b.md, "XLA settings at 1.3B").
    """
    levels = [part.split("=", 1)[1] for part in flags.split() if part.startswith("--xla_gpu_autotune_level=")]
    if levels and levels[-1].strip() == "0":
        logger.warning(
            "XLA_FLAGS has --xla_gpu_autotune_level=0: GEMMs and fusions run untuned "
            "(15x slower 1.3B train steps on H100, benchmarks/fp8_1p3b.md)"
        )
        return True
    return False
