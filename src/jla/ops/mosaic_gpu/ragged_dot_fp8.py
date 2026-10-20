# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2025 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py (JAX 0.11.2), through
# jla.ops.mosaic_gpu.ragged_dot (the persistent aligned schedule).
"""FP8 forward grouped GEMM on Mosaic GPU (H100): a measurement kernel for the MoE's FP8 path.

``ragged_dot_fp8(lhs [M, K], rhs_t [G, N, K], group_sizes [G])`` computes
``out[r] = lhs[r] @ rhs_t[g(r)]^T`` with float8 (e4m3 or e5m2) operands, float32 accumulation and a
bfloat16 output, on the persistent aligned schedule of :mod:`jla.ops.mosaic_gpu.ragged_dot`. WGMMA
takes 8-bit operands only K-major, so the weights come as ``[G, N, K]``. Dequantization (the product of
the two per-tensor scales) is left to the caller, which XLA fuses into the consumer. Rows past
``sum(group_sizes)`` are not written (every row must be in a group).

Not differentiable and not used by the MoE layer: it measures what an FP8 forward / ``d_lhs`` GEMM
gains at the shipped MoE shapes (``benchmarks/bench_ragged_dot_fp8.py``, ``benchmarks/ragged_dot.md``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from jla.ops.mosaic_gpu._common import is_hopper, num_sms
from jla.ops.mosaic_gpu.ragged_dot import ROW_ALIGN, _row_tile_table

__all__ = ["FP8_DTYPES", "ragged_dot_fp8", "supports_fp8"]

Array = jax.Array

FP8_DTYPES = (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2))
BLOCK_M, BLOCK_N, BLOCK_K, STEPS = 128, 128, 128, 4  # 128-byte rows of 8-bit operands


def supports_fp8(lhs, rhs_t, group_sizes) -> bool:
    """Whether :func:`ragged_dot_fp8` takes these operands on this device."""
    if not is_hopper() or getattr(lhs, "ndim", None) != 2 or getattr(rhs_t, "ndim", None) != 3:
        return False
    if jnp.dtype(lhs.dtype) not in FP8_DTYPES or jnp.dtype(rhs_t.dtype) not in FP8_DTYPES:
        return False
    m, k = lhs.shape
    g, n, k2 = rhs_t.shape
    return (k == k2 and group_sizes.shape == (g,) and m >= BLOCK_M and m % ROW_ALIGN == 0
            and k % BLOCK_K == 0 and n % BLOCK_N == 0)


def ragged_dot_fp8(lhs: Array, rhs_t: Array, group_sizes: Array) -> Array:
    """``lhs [M, K]`` (8-bit) x ``rhs_t [G, N, K]`` (8-bit) -> ``[M, N]`` bfloat16, float32 accumulation."""
    if not supports_fp8(lhs, rhs_t, group_sizes):
        raise ValueError(f"ragged_dot_fp8 does not take lhs {lhs.shape} {lhs.dtype}, rhs_t {rhs_t.shape} "
                         f"{rhs_t.dtype}: it needs an H100, float8 operands, M >= {BLOCK_M} and a multiple of "
                         f"{ROW_ALIGN}, K a multiple of {BLOCK_K} and N of {BLOCK_N}")
    m, k = lhs.shape
    g, n, _ = rhs_t.shape
    tiles = pl.cdiv(m, BLOCK_M) + g - 1
    grid_n = n // BLOCK_N
    k_steps = k // BLOCK_K
    sms = num_sms()
    table, count = _row_tile_table(group_sizes.astype(jnp.int32), BLOCK_M, tiles)

    def body(table_gmem, count_gmem, lhs_gmem, rhs_gmem, o_gmem):
        sm = lax.axis_index("sm")
        work = count_gmem[0] * grid_n
        mine = lax.div(jnp.maximum(work - sm, 0) + (sms - 1), jnp.int32(sms))

        def item(i):
            w = sm + i * sms
            return lax.div(w, jnp.int32(grid_n)), lax.rem(w, jnp.int32(grid_n))

        def lhs_map(i, kk):
            return pl.multiple_of(table_gmem[1, item(i)[0]], ROW_ALIGN), kk

        def rhs_map(i, kk):
            mi, ni = item(i)
            return table_gmem[0, mi], ni, kk

        def scope(acc_ref, o_smem):
            def step(idx, lhs_smem, rhs_smem):
                i, kk = idx

                @pl.when(kk == 0)
                def _():
                    acc_ref[...] = jnp.zeros((BLOCK_M, BLOCK_N), jnp.float32)

                plgpu.wgmma(acc_ref, lhs_smem, rhs_smem.transpose((1, 0)))  # B K-major, as 8-bit WGMMA needs

                @pl.when(kk == k_steps - 1)
                def _():
                    mi, ni = item(i)
                    acc = acc_ref[...]
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    o_smem[...] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()
                    row0 = pl.multiple_of(table_gmem[1, mi], ROW_ALIGN)
                    start, size = table_gmem[2, mi], table_gmem[3, mi]
                    columns = pl.ds(ni * BLOCK_N, BLOCK_N)
                    smem_start = start
                    remaining = min(BLOCK_M, m)
                    while remaining > 0:  # TMA boxes are static: a power-of-two ladder of row slices
                        rows = 1 << (remaining.bit_length() - 1)
                        remaining //= 2

                        @pl.when(size & rows != 0)
                        def _():
                            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(smem_start, rows)],
                                                    o_gmem.at[pl.ds(row0 + smem_start, rows), columns])

                        smem_start += size & rows

            @pl.when(mine > 0)
            def _():
                plgpu.emit_pipeline(
                    step,
                    grid=(mine, k_steps),
                    in_specs=[
                        plgpu.BlockSpec((pl.Element(BLOCK_M), BLOCK_K), lhs_map, delay_release=1),
                        plgpu.BlockSpec((None, BLOCK_N, BLOCK_K), rhs_map, delay_release=1),
                    ],
                    max_concurrent_steps=STEPS,
                )(lhs_gmem, rhs_gmem)

            plgpu.wait_smem_to_gmem(0)

        pl.run_scoped(scope, plgpu.ACC((BLOCK_M, BLOCK_N)), plgpu.SMEM((BLOCK_M, BLOCK_N), jnp.bfloat16))

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(table, count, lhs, rhs_t)
