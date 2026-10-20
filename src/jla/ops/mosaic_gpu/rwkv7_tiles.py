# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tile helpers of the RWKV-7 per-chunk kernels (:mod:`rwkv7_prep`, :mod:`rwkv7_bwd`).

One warpgroup works on ``[64, 64]`` register tiles in the WGMMA layout. These are the RWKV-7 kernels'
own copies of the small helpers that :mod:`jla.ops.mosaic_gpu.vector_gate` keeps private (same code
and numerics), so a change to the vector-gate kernels' internals cannot break RWKV-7.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

Array = jax.Array

__all__ = ["COL", "ROW", "WG", "Regs", "cols", "iota", "kernel", "load", "matmul", "matmuls", "rows", "split_bf16",
           "swizzled"]

WG = plgpu.Layout.WGMMA
ROW = plgpu.Layout.WGMMA.reduce(1)  # one value per row of a WGMMA-layout array
COL = plgpu.Layout.WGMMA.reduce(0)  # one value per column
_F32 = jnp.float32
_BF16 = jnp.bfloat16


def kernel(name: str, body, **kwargs):
    """``plgpu.kernel`` with Lane semantics (SMEM buffers keep exactly the transforms given) and fast math."""
    params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane, approx_math=True)
    return plgpu.kernel(body, kernel_name=name, compiler_params=params, **kwargs)


def swizzled(shape: tuple[int, ...], dtype=_BF16) -> plgpu.SMEM:
    """SMEM tiled and swizzled for its minor dimension (WGMMA operands, conflict-free loads)."""
    itemsize = jnp.dtype(dtype).itemsize
    swizzle = plgpu.find_swizzle(shape[-1] * itemsize * 8)
    transforms = (plgpu.TilingTransform((8, swizzle // itemsize)), plgpu.SwizzleTransform(swizzle))
    return plgpu.SMEM(shape, dtype, transforms=transforms)


def iota(shape: tuple[int, int], dim: int) -> Array:
    return plgpu.broadcasted_iota(jnp.int32, shape, dim, layout=WG)


def load(ref, layout=WG) -> Array:
    return plgpu.load(ref, layout=layout, optimized=False)


def rows(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-row vector (``ROW`` layout) along the columns."""
    return lax.broadcast_in_dim(x, shape, [0])


def cols(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-column vector (``COL`` layout) along the rows."""
    return lax.broadcast_in_dim(x, shape, [1])


class Regs(NamedTuple):
    """A bf16 WGMMA-layout register array used as the left operand of a matmul."""

    value: Array


def matmul(shape: tuple[int, int], *pairs, init: Array | None = None) -> Array:
    """``init + sum(a @ b for a, b in pairs)`` on the tensor cores (f32 accumulator).

    ``a`` is an SMEM ref (or a transposed view) or a :class:`Regs` operand, ``b`` an SMEM ref (or a
    transposed view). Reading the result waits for the products, so the operands' SMEM may be
    rewritten afterwards.
    """
    pairs = tuple((a.value if isinstance(a, Regs) else a, b) for a, b in pairs)
    if init is None:

        def scoped(acc):
            for a, b in pairs:
                plgpu.wgmma(acc, a, b)
            return acc[...]

        return pl.run_scoped(scoped, plgpu.ACC(shape, _F32))

    def body(acc):
        for a, b in pairs:
            plgpu.wgmma(acc, a, b)

    return pl.run_state(body)(plgpu.ACC.init(init))


def matmuls(shape: tuple[int, int], *groups) -> tuple[Array, ...]:
    """Independent products issued together, one f32 accumulator per group of ``(a, b)`` pairs (as in
    :func:`matmul`), so their tensor-core chains overlap; reading the results waits for all of them."""
    groups = tuple(tuple((a.value if isinstance(a, Regs) else a, b) for a, b in group) for group in groups)

    def scoped(*accs):
        for acc, pairs in zip(accs, groups):
            for a, b in pairs:
                plgpu.wgmma(acc, a, b)
        return tuple(acc[...] for acc in accs)

    return pl.run_scoped(scoped, *(plgpu.ACC(shape, _F32) for _ in groups))


def split_bf16(x: Array) -> tuple[Array, Array]:
    """``x ~= hi + lo`` with both parts bf16 (about 16 significant bits together)."""
    hi = x.astype(_BF16)
    return hi, (x - hi.astype(_F32)).astype(_BF16)
