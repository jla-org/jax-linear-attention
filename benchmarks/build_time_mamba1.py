# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build time of the Mamba-1 kernels: ``lower()`` and ``compile()`` of one forward + backward per backward path.

    python benchmarks/build_time_mamba1.py --batch 8 --seq-len 2048 --dim 1536 --d-state 16 \\
        --backwards kernel,kernel_v1

Lowering (Mosaic GPU's Python-side emission of the kernels to MLIR) is what dominates. The paths run in one
process in the order given, so a later path may reuse the forward kernel's cached trace; run one path per
process for the cold number.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jla import ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--backwards", default="kernel,kernel_v1")
    args = parser.parse_args()
    b, t, d, n = args.batch, args.seq_len, args.dim, args.d_state
    bf16 = jnp.bfloat16
    x = dict(q=jnp.zeros((b, t, n), bf16), k=jnp.zeros((b, t, n), bf16), v=jnp.zeros((b, t, d), bf16),
             dt=jnp.zeros((b, t, d), bf16), a=-jnp.ones((d, n), jnp.float32), d=jnp.ones((d,), jnp.float32),
             z=jnp.zeros((b, t, d), bf16), dt_bias=jnp.zeros((d,), jnp.float32))
    for backward in args.backwards.split(","):
        def loss(q, k, v, dt, a, d, z, dt_bias, backward=backward):
            o, _ = ops.linear_attention("mamba1", q, k, v, dt=dt, a=a, d=d, z=z, dt_bias=dt_bias, dt_softplus=True,
                                        impl="mosaic_gpu", backward=backward)
            return jnp.sum(o.astype(jnp.float32))

        fn = jax.jit(jax.grad(loss, argnums=tuple(range(8))))
        t0 = time.perf_counter()
        lowered = fn.lower(*x.values())
        t1 = time.perf_counter()
        lowered.compile()
        t2 = time.perf_counter()
        print(json.dumps(dict(backward=backward, shape=[b, t, d, n], lower_s=round(t1 - t0, 1),
                              compile_s=round(t2 - t1, 1))), flush=True)


if __name__ == "__main__":
    main()
