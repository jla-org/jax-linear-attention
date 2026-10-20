# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU (JAX 0.11.2, H100): row-shifted views of an untransformed SMEM buffer.

    python inferred_swizzle_views.py [warpgroup|lane]

A [64, 64] f32 tile is stored at rows 16..79 of an untransformed [80, 72] SMEM buffer and read
back through row/column-offset views with ``plgpu.load(..., layout=WGMMA)``. Under
``LoweringSemantics.Warpgroup`` layout inference gives the buffer a swizzle, and the offset views
return permuted data without an error; ``Lane`` keeps the buffer untransformed and every view is
exact.
"""

import functools
import sys

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

C, K, ROWS, PITCH = 64, 64, 80, 72
CASES = ((16, 0, 64), (15, 0, 64), (11, 0, 64), (11, 32, 32), (1, 8, 32))  # (row0, col0, ncols)
semantics = sys.argv[1] if len(sys.argv) > 1 else "warpgroup"
params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane if semantics == "lane"
                              else plgpu.LoweringSemantics.Warpgroup)


@functools.partial(plgpu.kernel, out_type=tuple(jax.ShapeDtypeStruct((C, n), jnp.float32) for _, _, n in CASES),
                   scratch_types=dict(buf=plgpu.SMEM((ROWS, PITCH), jnp.float32),
                                      src=plgpu.SMEM((C, K), jnp.float32), bar=plgpu.Barrier()),
                   compiler_params=params)
def kernel(x_ref, *outs, buf, src, bar):
    plgpu.copy_gmem_to_smem(x_ref, src, bar)
    plgpu.barrier_wait(bar)
    buf.at[pl.ds(16, C), pl.ds(0, K)][...] = plgpu.load(src, layout=plgpu.Layout.WGMMA, optimized=False)
    for o_ref, (r0, c0, n) in zip(outs, CASES):
        o_ref[...] = plgpu.load(buf.at[pl.ds(r0, C), pl.ds(c0, n)], layout=plgpu.Layout.WGMMA, optimized=False)


x = jnp.arange(C * K, dtype=jnp.float32).reshape(C, K)
expected = np.full((ROWS, PITCH), np.nan, np.float32)
expected[16:16 + C, :K] = np.asarray(x)
for out, (r0, c0, n) in zip(jax.jit(kernel)(x), CASES):
    want = expected[r0:r0 + C, c0:c0 + n]
    valid = np.isfinite(want)
    print(f"{semantics}: view rows {r0}..{r0 + C - 1}, cols {c0}..{c0 + n - 1}: "
          f"{'exact' if np.array_equal(np.asarray(out)[valid], want[valid]) else 'WRONG DATA'}")
