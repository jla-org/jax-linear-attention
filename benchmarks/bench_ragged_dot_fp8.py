# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 against bf16 forward grouped GEMMs at the hybrid presets' MoE shapes (benchmarks/ragged_dot.md).

    python benchmarks/bench_ragged_dot_fp8.py --tokens 16384 --out <json>

Per shape (the expert up and down projections of ``qwen3-next-1p3b`` and ``kimi-linear-1p3b``, group
sizes from top-k routing of random logits with a per-expert bias): the bf16 forward on the shipped
default tiling (``PERSISTENT``, weights ``[G, K, N]``), the same kernel reading ``[G, N, K]`` weights, and
:func:`jla.ops.mosaic_gpu.ragged_dot_fp8.ragged_dot_fp8` (e4m3 operands, ``[G, N, K]`` weights), in
rotated rounds; median CUPTI kernel ms per call and TFLOP/s of the useful ``2 routes K N``. The
quantization passes an FP8 MoE layer would add are not included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _measure_each, environment, gpu_snapshot  # noqa: E402

from jla.ops.mosaic_gpu import ragged_dot as rd  # noqa: E402
from jla.ops.mosaic_gpu.ragged_dot_fp8 import ragged_dot_fp8  # noqa: E402

# name: (experts, top_k, d_model, d_ff)
LAYERS = {"qwen3-next-1p3b": (64, 10, 1024, 256), "kimi-linear-1p3b": (32, 8, 1024, 512)}


def _group_sizes(tokens: int, experts: int, top_k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(tokens, experts)) + rng.normal(scale=0.5, size=(1, experts))
    top = np.argsort(-logits, axis=1)[:, :top_k]
    return np.bincount(top.ravel(), minlength=experts).astype(np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = {"environment": environment(), "tokens": args.tokens, "gpu_before": gpu_snapshot(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": {}}
    for name, (experts, top_k, d_model, d_ff) in LAYERS.items():
        sizes = jnp.asarray(_group_sizes(args.tokens, experts, top_k))
        routes = args.tokens * top_k
        for gemm, (k, n) in {"up": (d_model, 2 * d_ff), "down": (d_ff, d_model)}.items():
            kx, kw = jax.random.split(jax.random.key(len(report["cases"])))
            x = jax.random.normal(kx, (routes, k), jnp.float32)
            w = jax.random.normal(kw, (experts, k, n), jnp.float32) * k ** -0.5
            x16, w16 = x.astype(jnp.bfloat16), w.astype(jnp.bfloat16)
            w16_t = jnp.swapaxes(w16, 1, 2)
            x8, w8_t = x.astype(jnp.float8_e4m3fn), jnp.swapaxes(w, 1, 2).astype(jnp.float8_e4m3fn)
            fns = {
                "bf16_persistent": (jax.jit(lambda a, b, s: rd._ragged_dot_kernel(
                    a, b, s, transpose_rhs=False, cfg=rd.PERSISTENT)), (x16, w16, sizes)),
                "bf16_persistent_nk": (jax.jit(lambda a, b, s: rd._ragged_dot_kernel(
                    a, b, s, transpose_rhs=True, cfg=rd.PERSISTENT)), (x16, w16_t, sizes)),
                "fp8_e4m3": (jax.jit(ragged_dot_fp8), (x8, w8_t, sizes)),
            }
            times = {v: [] for v in fns}
            names = list(fns)
            for r in range(args.rounds):
                order = names[r % len(names):] + names[:r % len(names)]
                for v in order:
                    fn, fargs = fns[v]
                    for _ in range(3):
                        jax.block_until_ready(fn(*fargs))
                    times[v] += [float(t) for t in _measure_each(fn, fargs, args.iterations, aggregate=True)]
            ref = np.asarray(fns["bf16_persistent"][0](*fns["bf16_persistent"][1]), np.float32)
            got = np.asarray(fns["fp8_e4m3"][0](*fns["fp8_e4m3"][1]), np.float32)
            flops = 2.0 * routes * k * n
            case = {"shape": {"routes": routes, "k": k, "n": n, "experts": experts},
                    "median_ms": {v: float(np.median(t)) for v, t in times.items()},
                    "tflops": {v: flops / (float(np.median(t)) * 1e-3) / 1e12 for v, t in times.items()},
                    "fp8_rel_error_vs_bf16": float(np.linalg.norm(got - ref) / np.linalg.norm(ref))}
            case["fp8_speedup"] = case["median_ms"]["bf16_persistent"] / case["median_ms"]["fp8_e4m3"]
            report["cases"][f"{name}/{gemm}"] = case
            print(json.dumps({f"{name}/{gemm}": case}), flush=True)
    report["gpu_after"] = gpu_snapshot()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
