# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU (JAX 0.11.2, H100): cross-warp reductions in two warpgroups at the same time.

``x.sum(axis=0)`` of a WGMMA-layout tile sums over rows that live in different warps, so Pallas
lowers it through an SMEM scratch region (``CompilerParams.reduction_scratch_bytes``). There is
one such region per kernel: two warpgroups that reduce at the same time share it, and the
column sums come out wrong without an error. (The all-axes reduction of a strided layout refuses
multiple Pallas threads with NotImplementedError; this path has no such check.) Reductions over
columns (``axis=1``) stay inside a warp and are fine.

Workaround in ``gated_delta_rule_bwd.chunk_grads``: no ``axis=0`` sums and no full sums of
row vectors in warpgroup code. Column sums of ``X`` are row sums of a transposed product
computed directly (one extra 64x64 matmul each), and the one full sum goes through a small
SMEM vector read back in the column layout.
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from jax import lax

WGMMA = plgpu.Layout.WGMMA
ROUNDS = 16


@functools.partial(
    plgpu.kernel, out_type=jax.ShapeDtypeStruct((2, 64), jnp.float32), num_threads=2, thread_name="wg",
    scratch_types=dict(o_s=plgpu.SMEM((2, 64), jnp.float32)),
)
def kernel(o_ref, o_s):
    wg = lax.axis_index("wg")
    for w in range(2):

        @pl.when(wg == w)
        def _():
            rows = plgpu.broadcasted_iota(jnp.int32, (64, 64), 0, layout=WGMMA).astype(jnp.float32)
            cols = plgpu.broadcasted_iota(jnp.int32, (64, 64), 1, layout=WGMMA).astype(jnp.float32)
            acc = None
            for i in range(ROUNDS):  # many reductions, so the two warpgroups overlap
                part = (rows * (w + 1) + cols * (i + 1)).sum(axis=0)
                acc = part if acc is None else acc + part
            o_s.at[w][...] = acc
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_s.at[w], o_ref.at[w])
            plgpu.wait_smem_to_gmem(0)


got = np.asarray(jax.jit(kernel)())
r, c = np.arange(64.0)[:, None], np.arange(64.0)[None, :]
want = np.stack([sum((r * (w + 1) + c * (i + 1)).sum(0) for i in range(ROUNDS)) for w in range(2)])
err = np.abs(got - want).max()
print(f"max |column sum error| = {err:.3g} (of max {np.abs(want).max():.3g}):",
      "wrong results" if err > 1e-3 else "correct (no overlap this run)")
