# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared timing helpers for the linear-attention kernel benchmarks.

The metric is GPU kernel time per call: the sum of the CUDA kernel durations
of one call, measured with CUPTI (``jax.experimental.mosaic.gpu.profiler`` for
JAX, ``torch.profiler`` for the torch baselines), reported as the median over
``iterations`` calls after a warmup. It excludes host dispatch and idle gaps,
so XLA, Mosaic GPU, Triton and torch code are compared on the same footing.

When a GPU is shared with other jobs, kernels run in
time slices of the GPU, so absolute times are inflated by an unknown amount;
each result records ``nvidia-smi`` utilization and memory before and after,
and the drivers interleave implementations so ratios are measured in the same
time window.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

ITERATIONS = 30
WARMUP = 3


def gpu_snapshot() -> list[dict[str, Any]]:
    """``nvidia-smi`` utilization/memory of the visible GPUs (all GPUs if unset)."""
    query = "index,utilization.gpu,memory.used,memory.total"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    keep = None if not visible else {int(i) for i in visible.split(",") if i.strip().isdigit()}
    rows = []
    for line in out.strip().splitlines():
        idx, util, used, total = (int(x.strip()) for x in line.split(","))
        if keep is None or idx in keep:
            rows.append(dict(gpu=idx, util_pct=util, mem_used_mib=used, mem_total_mib=total))
    return rows


def _measure_each(fn: Callable[..., Any], args, iterations: int, aggregate: bool) -> list:
    """CUPTI timings of ``iterations`` calls, one call per measurement.

    ``profiler.measure(fn, iterations=n)`` keeps the outputs of all ``n`` calls
    alive until it returns, so a large forward+backward ran out of memory in the
    harness rather than in the kernels; here each call's outputs are freed
    before the next call.
    """
    from jax.experimental.mosaic.gpu import profiler

    timings = []
    for _ in range(iterations):
        out, t = profiler.measure(fn, aggregate=aggregate)(*args)
        jax.block_until_ready(out)
        del out
        if t is not None:
            timings.append(t)
    return timings


def median_kernel_ms(fn: Callable[..., Any], *args, iterations: int = ITERATIONS, warmup: int = WARMUP) -> dict:
    """Median CUPTI kernel time (ms) of ``fn(*args)`` (``fn`` should be jitted)."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = _measure_each(fn, args, iterations, aggregate=True)
    return dict(median_ms=statistics.median(times), min_ms=min(times), iterations=len(times))


def kernel_breakdown_ms(fn: Callable[..., Any], *args, iterations: int = 10) -> dict[str, float]:
    """Median per-kernel time (ms) of ``fn(*args)`` by kernel name."""
    jax.block_until_ready(fn(*args))
    runs = _measure_each(fn, args, iterations, aggregate=False)
    per: dict[str, list[float]] = {}
    for run in runs:
        agg: dict[str, float] = {}
        for name, t in run:
            name = name.split("(")[0]
            agg[name] = agg.get(name, 0.0) + t
        for name, t in agg.items():
            per.setdefault(name, []).append(t)
    return {name: statistics.median(ts) for name, ts in sorted(per.items(), key=lambda kv: -max(kv[1]))}


def vjp_fn(fn: Callable[..., Any], argnums: tuple[int, ...]) -> Callable[..., Any]:
    """``(args, cotangent) -> (output, gradients)`` of the first output of ``fn``.

    The output is returned so that the forward kernels run: the gradients alone
    do not need it, and XLA would otherwise drop the output computation (in a
    model the next layer consumes it, and the torch baselines always run it).
    """

    def run(*args_and_ct):
        *args, ct = args_and_ct

        def primal(*diff):
            full = list(args)
            for i, a in zip(argnums, diff):
                full[i] = a
            return fn(*full)[0]

        out, pullback = jax.vjp(primal, *(args[i] for i in argnums))
        return out, pullback(ct)

    return jax.jit(run)


def environment() -> dict[str, Any]:
    device = jax.devices()[0]
    return dict(
        jax=jax.__version__, device=device.device_kind, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def error_summary(text: str, limit: int = 300) -> str:
    """The last non-empty line of a failed baseline's output (the exception), without stack frames or paths."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else "no output")[:limit]


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)


def bf16_inputs(seed: int, batch: int, seq_len: int, heads: int, key_dim: int, value_dim: int, *,
                kv_heads: int | None = None, delta: bool) -> dict[str, jax.Array]:
    """FLA-style benchmark inputs: l2-normalized keys, logsigmoid gates, sigmoid betas."""
    kv_heads = kv_heads or heads
    ks = jax.random.split(jax.random.key(seed), 6)
    f32 = jnp.float32
    q = jax.random.normal(ks[0], (batch, seq_len, kv_heads, key_dim), f32)
    k = jax.random.normal(ks[1], (batch, seq_len, kv_heads, key_dim), f32)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    v = jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), f32)
    g = jax.nn.log_sigmoid(jax.random.normal(ks[3], (batch, seq_len, heads), f32))
    out = dict(q=q.astype(jnp.bfloat16), k=k.astype(jnp.bfloat16), v=v.astype(jnp.bfloat16), g=g)
    if delta:
        out["beta"] = jax.nn.sigmoid(jax.random.normal(ks[4], (batch, seq_len, heads), f32))
    out["do"] = jax.random.normal(ks[5], (batch, seq_len, heads, value_dim), f32).astype(jnp.bfloat16)
    return out
