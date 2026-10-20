# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time one FP8 matmul (forward plus both backward GEMMs) in several spellings of the scaled dot.

    python benchmarks/bench_fp8_gemm.py [M K N]   # default 16384 2048 8192 (the 1B MLP shape)

Each spelling replaces ``jla.layers.fp8._scaled_dot``; ``direct_out_dtype`` is the shipped one.
Prints one JSON row per (spelling, recipe): time, speed against bfloat16, the GEMM kernels XLA
picked (cuBLASLt FP8 calls and their output types) and the largest difference from the first
spelling's result, relative to the output's max. ``benchmarks/fp8.md`` reports these rows. Run it on a Hopper GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

from jla.layers import fp8 as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_fp8 import _hlo_text, gemm_summary  # noqa: E402


def f32_then_scale(a, a_scale, b, b_scale, dims, out_dtype, precision):
    out = lax.dot_general(a, b, dims, precision=precision, preferred_element_type=jnp.float32)
    return (out * (a_scale * b_scale)).astype(out_dtype)


def direct_out_dtype(a, a_scale, b, b_scale, dims, out_dtype, precision):  # the shipped form
    out = lax.dot_general(a, b, dims, precision=precision, preferred_element_type=out_dtype)
    return out * (a_scale * b_scale).astype(out_dtype)


def direct_f32_scale(a, a_scale, b, b_scale, dims, out_dtype, precision):
    out = lax.dot_general(a, b, dims, precision=precision, preferred_element_type=out_dtype)
    return (out.astype(jnp.float32) * (a_scale * b_scale)).astype(out_dtype)


def dequant_then_dot(a, a_scale, b, b_scale, dims, out_dtype, precision):
    a = a.astype(out_dtype) * a_scale.astype(out_dtype)
    b = b.astype(out_dtype) * b_scale.astype(out_dtype)
    return lax.dot_general(a, b, dims, precision=precision, preferred_element_type=out_dtype)


SPELLINGS = {
    "direct_out_dtype": direct_out_dtype,
    "f32_then_scale": f32_then_scale,
    "direct_f32_scale": direct_f32_scale,
    "dequant_then_dot": dequant_then_dot,
}


def _time(fn, *args, iters: int = 30) -> float:
    for _ in range(4):  # compile + 3 warmups
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return 1e3 * (time.perf_counter() - start) / iters


def main() -> None:
    m, k, n = (int(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (16384, 2048, 8192)
    kx, kw, kg = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(kx, (m, k), jnp.bfloat16)
    w = 0.02 * jax.random.normal(kw, (k, n), jnp.float32)
    g = jax.random.normal(kg, (m, n), jnp.bfloat16)

    @jax.jit
    def bf16(x, w, g):
        y, vjp = jax.vjp(lambda a, b: a @ b.astype(jnp.bfloat16), x, w)
        return y, vjp(g)

    t_bf16 = _time(bf16, x, w, g)
    print(json.dumps({"shape": [m, k, n], "variant": "bf16", "ms": round(t_bf16, 3)}), flush=True)
    shipped, first = F._scaled_dot, {}
    try:
        for name, impl in SPELLINGS.items():
            F._scaled_dot = impl
            for recipe in ("delayed", "current"):
                # A delayed state with recorded amaxes (inputs ~4, weights ~0.1, gradients ~4).
                state = None if recipe == "current" else F.Fp8State(
                    jnp.float32(1.0), jnp.zeros(16).at[0].set(4.0), jnp.float32(1.0), jnp.zeros(16).at[0].set(0.1),
                    jnp.float32(1.0), jnp.zeros(16).at[0].set(4.0))

                @jax.jit
                def fwd_bwd(x, w, g, state, recipe=recipe):
                    y, vjp = jax.vjp(lambda a, b, s: F.fp8_dot(a, b, s, recipe=recipe), x, w, state)
                    return y, vjp(g)

                summary = gemm_summary(_hlo_text(fwd_bwd.lower(x, w, g, state).compile()))
                ms = _time(fwd_bwd, x, w, g, state)
                y, grads = fwd_bwd(x, w, g, state)
                outs = [t.astype(jnp.float32) for t in (y, grads[0], grads[1])]
                ref = first.setdefault(recipe, outs)
                err = [float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30)) for a, b in zip(outs, ref)]
                print(json.dumps({
                    "shape": [m, k, n], "variant": name, "recipe": recipe, "ms": round(ms, 3),
                    "vs_bf16": round(t_bf16 / ms, 3), "rel_err_vs_first": [round(e, 5) for e in err],
                    "f8_gemms": summary.get("__cublas$lt$matmul$f8", 0),
                    "out_types": summary["f8_cublaslt_out_types"],
                    "other_gemms": {key: v for key, v in summary.items() if "gemm" in key or "matmul" in key},
                }), flush=True)
    finally:
        F._scaled_dot = shipped


if __name__ == "__main__":
    main()
