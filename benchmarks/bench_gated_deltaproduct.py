# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaProduct forward + backward: Mosaic GPU (interleaved GDN kernel) vs chunked JAX.

    python benchmarks/bench_gated_deltaproduct.py --batch 4 --seq 2048 --heads 8 --num-householder 2
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jla import ops
from jla.ops import reference as ref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-householder", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--impls", default="mosaic_gpu,chunked")
    args = parser.parse_args()
    B, T, H, K, n = args.batch, args.seq, args.heads, args.dim, args.num_householder
    ks = jax.random.split(jax.random.key(0), 6)
    x = dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(jnp.bfloat16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T * n, H, K))).astype(jnp.bfloat16),
        v=jax.random.normal(ks[2], (B, T * n, H, K)).astype(jnp.bfloat16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 1.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (B, T * n, H))),
    )
    do = jax.random.normal(ks[5], (B, T, H, K)).astype(jnp.bfloat16)
    results = {"shape": dict(B=B, T=T, H=H, K=K, V=K, num_householder=n), "ms_fwd_bwd": {}}
    for impl in args.impls.split(","):
        def loss(p, impl=impl):
            o, _ = ops.linear_attention("gated_deltaproduct", p["q"], p["k"], p["v"], g=p["g"], beta=p["beta"],
                                        impl=impl, use_qk_l2norm=True)
            return jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32))

        step = jax.jit(jax.grad(loss))
        jax.block_until_ready(step(x))
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            jax.block_until_ready(step(x))
            times.append(1e3 * (time.perf_counter() - start))
        results["ms_fwd_bwd"][impl] = float(np.median(times))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
