# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time chunked log-linear Mamba-2 (``log_linear_simple_gla``) against chunked ``simple_gla``.

    python benchmarks/bench_log_linear.py --batch 1 --seq-len 2048 --heads 24 --dk 128 --dv 64

Both are plain-JAX chunked ops (no Mosaic GPU kernel yet); bf16 inputs, float32 state.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jla import ops
from jla.ops.log_linear import num_levels


def _time(fn, *args, iters: int = 10) -> float:
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    b, t, h, dk, dv = args.batch, args.seq_len, args.heads, args.dk, args.dv
    ks = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(ks[0], (b, t, h, dk), jnp.bfloat16)
    k = jax.random.normal(ks[1], (b, t, h, dk), jnp.bfloat16)
    v = jax.random.normal(ks[2], (b, t, h, dv), jnp.bfloat16)
    g = -jax.nn.softplus(jax.random.normal(ks[3], (b, t, h)))
    lam = jax.nn.softplus(jax.random.normal(ks[4], (b, t, h, num_levels(t))))
    opts = dict(impl="chunked", chunk_size=args.chunk_size, precision=jax.lax.Precision.DEFAULT)

    def log_fwd(q, k, v, g, lam):
        return ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, **opts)[0]

    def lin_fwd(q, k, v, g, lam):
        return ops.linear_attention("simple_gla", q, k, v, g=g, **opts)[0]

    results = {"shape": dict(vars(args)), "levels": num_levels(t)}
    for name, fwd in (("log_linear_simple_gla", log_fwd), ("simple_gla", lin_fwd)):
        f = jax.jit(fwd)
        fb = jax.jit(jax.grad(lambda *a, fwd=fwd: jnp.sum(fwd(*a).astype(jnp.float32)), argnums=(0, 1, 2, 3)))
        results[name] = {"fwd_ms": _time(f, q, k, v, g, lam), "fwd_bwd_ms": _time(fb, q, k, v, g, lam)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
