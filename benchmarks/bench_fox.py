# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FoX attention (the XLA path) against cuDNN flash attention without a forget gate.

Times forward and forward + backward (gradients of q, k, v and the log forget gates)
of ``jla.layers.forgetting_attention.forgetting_attention`` and, as the speed target
for a fused FoX kernel, ``jax.nn.dot_product_attention(implementation="cudnn")`` on
the same causal shape::

    python benchmarks/bench_fox.py --batch 4 --seq 2048 --heads 12 --dim 64
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jla.layers.forgetting_attention import forgetting_attention


def _ms(fn, *args, iters: int) -> float:
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    args = parser.parse_args()
    b, t, h, d = args.batch, args.seq, args.heads, args.dim
    keys = jax.random.split(jax.random.key(0), 4)
    q, k, v = (jax.random.normal(kk, (b, t, h, d), jnp.bfloat16) for kk in keys[:3])
    log_f = jax.nn.log_sigmoid(jax.random.normal(keys[3], (b, t, h), jnp.float32) + 3.0)

    fox_fwd = jax.jit(forgetting_attention)
    fox_bwd = jax.jit(jax.grad(lambda *a: forgetting_attention(*a).astype(jnp.float32).sum(), argnums=(0, 1, 2, 3)))

    def sdpa(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")

    sdpa_fwd = jax.jit(sdpa)
    sdpa_bwd = jax.jit(jax.grad(lambda *a: sdpa(*a).astype(jnp.float32).sum(), argnums=(0, 1, 2)))

    causal_flops = 4.0 * b * h * d * t * (t + 1) / 2.0  # q.k^T and p.v over the causal half
    rows = {
        "fox_xla_fwd_ms": _ms(fox_fwd, q, k, v, log_f, iters=args.iters),
        "fox_xla_fwd_bwd_ms": _ms(fox_bwd, q, k, v, log_f, iters=args.iters),
        "cudnn_sdpa_fwd_ms": _ms(sdpa_fwd, q, k, v, iters=args.iters),
        "cudnn_sdpa_fwd_bwd_ms": _ms(sdpa_bwd, q, k, v, iters=args.iters),
    }
    result = {
        "shape": {"batch": b, "seq": t, "heads": h, "kv_heads": h, "dim": d, "dtype": "bfloat16"},
        "device": jax.devices()[0].device_kind,
        **rows,
        "fox_xla_fwd_tflops": causal_flops / rows["fox_xla_fwd_ms"] / 1e9,
        "cudnn_sdpa_fwd_tflops": causal_flops / rows["cudnn_sdpa_fwd_ms"] / 1e9,
        "note": "provisional: GPU shared with test runs",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
