# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py (the four
# intra-chunk DPLR matrices from operands gated relative to an in-chunk offset, on the tensor cores) and
# fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py (the WY factors W and U from the inverse).
"""The forward per-chunk work of the RWKV-7 bounded mode in one Mosaic GPU kernel (H100).

:func:`jla.ops.mosaic_gpu.rwkv7._operands` builds the state-pass operands in XLA: the four decayed
score matrices of each 32-token chunk, the WY inverse ``T = (I - A_ab)^-1``, ``W = T (a exp(G_prev))``
and ``U_wy = T A_ak V``, and the 64-row operand tiles ``[k rows; b rows]`` of the vector-gate state
kernels. :func:`prep` computes the same tiles in one kernel, one CTA per two chunks (64 tokens) and
head, without writing any score matrix or factor to HBM:

* the factors ``q exp(G - G_m)``, ``a exp(G_prev - G_m)``, ``k exp(G_m - G)`` and ``b exp(G_m - G)``
  through each chunk's middle token ``m`` (the bounded mode: every exponent is at most ``16 |lower_bound|``)
  are split into bf16 ``hi + lo`` parts, and each of the four ``[64, 64]`` score products is three
  tensor-core products (``hi hi + hi lo + lo hi``, f32 accumulation: the ``BF16X3`` numerics of the
  XLA path); the two chunks of a CTA form block-diagonal matrices, masked to their chunk, the causal
  (``A_q*``) or strictly causal (``A_a*``) triangle and the token's document;
* ``T = (I - A_ab)^-1`` for the block-diagonal ``A_ab`` of the two chunks (32 x 32 strictly lower blocks):
  the series ``(I + A)(I + A^2)(I + A^4)`` on its 8 x 8 diagonal blocks, doubled to 16 and 32 rows by
  ``T <- T + T A_off T``, eight split-bf16 products (see :func:`_inverse`);
* ``W``, ``A_ak V`` and ``U_wy`` as split-bf16 tensor-core products, and the output tiles written by
  TMA straight into the state kernels' 64-row layout.

Two warpgroups share the CTA (one warpgroup per SM left most issue slots idle behind dependent register
work, as in :mod:`jla.ops.mosaic_gpu.rwkv7_bwd`): warpgroup 0 splits the right factors into SMEM for both,
then runs the ``a`` rows (``A_ak``, ``A_ab``), the inverse, ``W`` and ``U``; warpgroup 1, once the right
factors are in SMEM (an mbarrier), the ``q`` rows (``A_qk``, ``A_qb`` folded onto each chunk's columns)
and the ``kg`` and ``qg`` tiles. The products and their order are those of one warpgroup, so the outputs
are bit for bit the same.

The chunk-local cumulative log decay ``G`` (f32, from XLA) is read once; ``G_prev`` is ``G`` shifted by
one row inside each chunk (zero at a chunk start). Head dims 64 only.
"""

from __future__ import annotations

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu.rwkv7_tiles import COL as _COL
from jla.ops.mosaic_gpu.rwkv7_tiles import ROW as _ROW
from jla.ops.mosaic_gpu.rwkv7_tiles import WG as _WG
from jla.ops.mosaic_gpu.rwkv7_tiles import Regs as _Regs
from jla.ops.mosaic_gpu.rwkv7_tiles import cols as _cols
from jla.ops.mosaic_gpu.rwkv7_tiles import iota as _iota
from jla.ops.mosaic_gpu.rwkv7_tiles import kernel as _kernel
from jla.ops.mosaic_gpu.rwkv7_tiles import load as _load
from jla.ops.mosaic_gpu.rwkv7_tiles import matmul as _mm
from jla.ops.mosaic_gpu.rwkv7_tiles import rows as _rows
from jla.ops.mosaic_gpu.rwkv7_tiles import split_bf16 as _split_bf16
from jla.ops.mosaic_gpu.rwkv7_tiles import swizzled as _swizzled

Array = jax.Array

__all__ = ["DIM", "TOKENS", "prep"]

CHUNK = 32  # tokens per chunk (jla.ops.mosaic_gpu.rwkv7.CHUNK)
TOKENS = 2 * CHUNK  # tokens per CTA: two chunks
DIM = 64  # the head dims the kernel supports
_SHIFT = 8  # leading rows of the G buffer: G_prev is G read one row up
_F32 = jnp.float32
_BF16 = jnp.bfloat16


def _x3(left: Array, r_hi, r_lo, *, transpose: bool) -> Array:
    """``left @ R^T`` (``transpose``) or ``left @ R`` for f32 ``left [64, 64]`` and ``R = r_hi + r_lo`` in SMEM."""
    l_hi, l_lo = _split_bf16(left)
    hi, lo = (r_hi.T, r_lo.T) if transpose else (r_hi, r_lo)
    return _mm((TOKENS, DIM), (_Regs(l_hi), hi), (_Regs(l_hi), lo), (_Regs(l_lo), hi))


def _store_split(x: Array, hi_smem, lo_smem) -> None:
    hi, lo = _split_bf16(x)
    hi_smem[...] = hi
    lo_smem[...] = lo


def _inverse(a: Array, hi_smem, lo_smem) -> Array:
    """``(I - A)^-1`` for the strictly lower ``A [64, 64]`` whose two diagonal 32 x 32 blocks are its only
    nonzeros: 8 split-bf16 products.

    The 8 x 8 diagonal blocks are ``(I + A)(I + A^2)(I + A^4)``; the inverse is then doubled to 16 and 32
    rows by ``T <- T + T A_off T`` (``A_off``: ``A``'s block below the diagonal blocks of ``T``, so ``T_21 =
    T_22 A_21 T_11``). Every factor is bounded by the inverse itself. The series over all 32 rows, ``(I +
    A)(I + A^2) ... (I + A^16)``, is not: with ``A_ab`` near ``-a`` on every entry (consecutive ``kk``
    nearly parallel, ``a`` large, slow decay) its powers grow like ``C(30, 15) a^16`` (``4e6`` at ``a =
    0.8``) and cancel to a ``T`` of order one, which the ~16-bit split products cannot resolve: ``W`` and
    ``U`` came out wrong by more than their magnitude and the state pass overflowed (NaN losses in
    training, benchmarks/rwkv7.md).
    """
    rows, cols = _iota((TOKENS, TOKENS), 0), _iota((TOKENS, TOKENS), 1)
    block = jnp.where((rows >> 3) == (cols >> 3), a, 0.0)  # the 8 x 8 diagonal blocks: block^8 = 0
    t, power = jnp.where(rows == cols, 1.0, 0.0) + block, block
    for _ in range(2):
        _store_split(power, hi_smem, lo_smem)
        plgpu.commit_smem()
        power = _x3(power, hi_smem, lo_smem, transpose=False)  # block^(2^i)
        _store_split(t, hi_smem, lo_smem)
        plgpu.commit_smem()
        t = t + _x3(power, hi_smem, lo_smem, transpose=False)  # (I + block^(2^i)) t
    for shift in (4, 5):  # 16- then 32-row blocks
        off = jnp.where(jnp.logical_and((rows >> shift) == (cols >> shift),
                                        (rows >> (shift - 1)) != (cols >> (shift - 1))), a, 0.0)
        _store_split(off, hi_smem, lo_smem)
        plgpu.commit_smem()
        t_off = _x3(t, hi_smem, lo_smem, transpose=False)  # T A_off
        _store_split(t, hi_smem, lo_smem)
        plgpu.commit_smem()
        t = t + _x3(t_off, hi_smem, lo_smem, transpose=False)  # T + T A_off T
    return t


# XLA's GPU cumulative sum over 32 tokens (a reduce-window it rewrites into blocks of 16): each block
# summed in order from zero, the second plus the first's total. chunk_decays sums the same way.
_BLOCK = 16
DECAY_COLUMNS = 128  # (head, channel) columns per chunk_decays CTA: one per thread of its warpgroup


def chunk_decays(w: Array, flags: Array, *, no_carry: float) -> tuple[Array, Array]:
    """``(G, G_carry)`` of the log decay ``w [B, T, H, 64]`` (f32 or bf16, ``T`` a multiple of 32): the
    cumulative sum inside each 32-token chunk ``G [B, T, H, 64]`` f32, bit for bit
    ``jnp.cumsum(w.astype(f32).reshape(B, T / 32, 32, H, 64), axis=2)`` on the GPU (in the order of
    ``_BLOCK``), and ``G_carry [B, T / 32, H, 64]`` f32: ``G`` at each chunk's last token, or
    ``no_carry`` where that token's ``carry_ok`` (``flags [B, 3, T]`` row 0, see :func:`prep`) is 0.
    One pass over ``w`` instead of XLA's three (block totals, their offsets, the sums) and the slice.
    """
    batch, seq_len, heads, key_dim = w.shape
    width = heads * key_dim
    if seq_len % CHUNK or width % DECAY_COLUMNS or CHUNK != 2 * _BLOCK:
        raise ValueError(f"chunk_decays needs T % {CHUNK} == 0 and H * K % {DECAY_COLUMNS} == 0, got {w.shape}")
    line = plgpu.Layout.WG_STRIDED((DECAY_COLUMNS,), vec_size=1)

    def body(w_ref, f_ref, g_ref, gc_ref, w_smem, g_smem, gc_smem, bar):
        col, n, b = lax.axis_index("col"), lax.axis_index("n"), lax.axis_index("b")
        cols = pl.ds(col * DECAY_COLUMNS, DECAY_COLUMNS)
        tok = pl.ds(n * CHUNK, CHUNK)
        plgpu.copy_gmem_to_smem(w_ref.at[b, tok, cols], w_smem, bar)
        plgpu.barrier_wait(bar)
        zero = plgpu.layout_cast(jnp.zeros((DECAY_COLUMNS,), _F32), line)
        first = zero
        for t in range(_BLOCK):
            first = first + _load(w_smem.at[t], line).astype(_F32)
            g_smem.at[t][...] = first
        second = zero
        for t in range(_BLOCK, CHUNK):
            second = second + _load(w_smem.at[t], line).astype(_F32)
            g_smem.at[t][...] = second + first
        keep = f_ref[b, 0, n * CHUNK + CHUNK - 1]  # 1.0 or 0.0; with a finite G the blend is the select
        gc_smem[...] = (second + first) * keep + no_carry * (1.0 - keep)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(g_smem, g_ref.at[b, tok, cols])
        plgpu.copy_smem_to_gmem(gc_smem, gc_ref.at[b, n, cols])
        plgpu.wait_smem_to_gmem(0)

    out = (jax.ShapeDtypeStruct((batch, seq_len, width), _F32),
           jax.ShapeDtypeStruct((batch, seq_len // CHUNK, width), _F32))
    tile = (CHUNK, DECAY_COLUMNS)
    scratch = (plgpu.SMEM(tile, w.dtype), plgpu.SMEM(tile, _F32), plgpu.SMEM((DECAY_COLUMNS,), _F32),
               plgpu.Barrier(num_arrivals=1))
    kern = _kernel("rwkv7_decays", body, out_type=out, grid=(width // DECAY_COLUMNS, seq_len // CHUNK, batch),
                   grid_names=("col", "n", "b"), scratch_types=scratch)
    g, g_carry = kern(w.reshape(batch, seq_len, width), flags)
    return g.reshape(w.shape), g_carry.reshape(batch, seq_len // CHUNK, heads, key_dim)


def prep(q: Array, k: Array, v: Array, a: Array, b: Array, g: Array, flags: Array, *, scale: float,
         save: bool = False) -> tuple[Array, ...]:
    """``(kg, w_rows, u_rows, qg, aqk)`` ``[B, H, 2 T, 64]`` bf16, the tiles of ``rwkv7._operands``; with
    ``save`` also ``T_hi, T_lo, Ak_hi, Ak_lo`` ``[B, H, T, 64]`` bf16, the residuals of the backward: the
    split-bf16 parts (``x ~= hi + lo``) of ``T`` and ``A_ak`` over the 64 tokens of each pair of chunks,
    block diagonal (row ``t`` holds row ``t % 32`` of its chunk's ``32 x 32`` matrix in the columns of
    that chunk, zeros in the other 32), the operands :mod:`jla.ops.mosaic_gpu.rwkv7_bwd` loads.

    ``q, k, v`` ``[B, T, H, 64]`` bf16, ``a, b`` ``[B, T, H, 64]`` f32 or bf16 (one dtype; bf16 tiles are
    read as their f32 values), ``g`` ``[B, T, H, 64]`` f32 (the
    cumulative log decay inside each 32-token chunk) and ``flags [B, 3, T]`` f32: per token, 1.0 where
    its document is the one of the token before its chunk (``carry_ok``), 1.0 where it is the document
    of its chunk's last token (``same_last``), and its document id. ``T`` is a multiple of 64.
    """
    batch, seq_len, heads, key_dim = q.shape
    if a.dtype != b.dtype or a.dtype not in (_F32, _BF16):
        raise ValueError(f"prep takes a and b of one dtype, f32 or bf16, got {a.dtype} and {b.dtype}")
    if seq_len % TOKENS or key_dim != DIM or v.shape[-1] != DIM:
        raise ValueError(f"prep needs T % {TOKENS} == 0 and head dims {DIM}, got {q.shape} and {v.shape}")
    out = jax.ShapeDtypeStruct((batch, heads, 2 * seq_len, DIM), _BF16)  # tested on the H100 only
    saved = jax.ShapeDtypeStruct((batch, heads, seq_len, TOKENS), _BF16)
    shape = (TOKENS, DIM)

    def body(q_ref, k_ref, v_ref, a_ref, b_ref, g_ref, f_ref, kg_ref, w_ref, u_ref, qg_ref, aqk_ref, *rest):
        saved_refs, scratch = (rest[:4], rest[4:]) if save else ((), rest)
        (q_smem, k_smem, v_smem, a_smem, b_smem, g_smem, f_smem, r1_hi, r1_lo, r2_hi, r2_lo, i_hi, i_lo, square,
         st_kk, st_kb, st_w, st_u, st_q, st_aqk, zeros, bar, bar_right, bar_free, bar_done) = scratch
        wg = lax.axis_index("wg")
        pair, h, bb = lax.axis_index("p"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(pair * TOKENS, TOKENS)

        @pl.when(wg == 0)
        def _loads():
            plgpu.copy_gmem_to_smem(q_ref.at[bb, tok, h], q_smem, bar)
            plgpu.copy_gmem_to_smem(k_ref.at[bb, tok, h], k_smem, bar)
            plgpu.copy_gmem_to_smem(v_ref.at[bb, tok, h], v_smem, bar)
            plgpu.copy_gmem_to_smem(a_ref.at[bb, tok, h], a_smem, bar)
            plgpu.copy_gmem_to_smem(b_ref.at[bb, tok, h], b_smem, bar)
            plgpu.copy_gmem_to_smem(g_ref.at[bb, tok, h], g_smem.at[pl.ds(_SHIFT, TOKENS)], bar)
            plgpu.copy_gmem_to_smem(f_ref.at[bb, :, tok], f_smem, bar)

        def second_rows() -> Array:  # rows of the pair's second chunk
            return _iota(shape, 0) >= CHUNK

        def at_token(t: int) -> Array:  # G at token t of the CTA, per channel, broadcast to every row
            return _cols(_load(g_smem.at[_SHIFT + t], _COL), shape)

        def g_now() -> Array:
            return _load(g_smem.at[pl.ds(_SHIFT, TOKENS)])

        def g_before() -> Array:  # G_prev: G one token earlier, zero at a chunk start
            start = (_iota(shape, 0) & (CHUNK - 1)) == 0
            return jnp.where(start, 0.0, _load(g_smem.at[pl.ds(_SHIFT - 1, TOKENS)]))

        def g_mid() -> Array:  # G at each chunk's middle token (the pivot of the bounded mode's factors)
            return jnp.where(second_rows(), at_token(TOKENS - CHUNK // 2 - 1), at_token(CHUNK // 2 - 1))

        def flag(i: int) -> Array:  # flags row i per token, broadcast along the channels
            return _rows(_load(f_smem.at[i], _ROW), shape)

        def keep(min_offset: int) -> Array:  # same chunk and document, row - column >= min_offset
            doc_rows = _rows(_load(f_smem.at[2], _ROW), (TOKENS, TOKENS))
            doc_cols = _cols(_load(f_smem.at[2], _COL), (TOKENS, TOKENS))
            rr, cc = _iota((TOKENS, TOKENS), 0), _iota((TOKENS, TOKENS), 1)
            offset = jnp.where((rr >> 5) == (cc >> 5), rr - cc, -1)
            return jnp.where(doc_rows == doc_cols, offset, -1) >= min_offset

        def store_rows(pairs) -> None:  # TMA stores: per chunk c, 32 upper rows then 32 lower rows
            for c in range(2):  # chunk c of the pair: rows (2 pair + c) 64 .. of the 64-row layout
                src = pl.ds(c * CHUNK, CHUNK)
                top = pl.ds((2 * pair + c) * TOKENS, CHUNK)
                bottom = pl.ds((2 * pair + c) * TOKENS + CHUNK, CHUNK)
                for out_ref, upper, lower in pairs:
                    if upper is not None:
                        plgpu.copy_smem_to_gmem(upper.at[src], out_ref.at[bb, h, top])
                    if lower is not None:
                        plgpu.copy_smem_to_gmem(lower.at[src], out_ref.at[bb, h, bottom])

        @pl.when(wg == 0)
        def _inverse_and_wy():
            plgpu.barrier_wait(bar)
            # the right factors k exp(G_m - G) (r1) and b exp(G_m - G) (r2), read by both warpgroups' products
            mid = g_mid()
            e_k = jnp.exp(mid - g_now())
            _store_split(_load(k_smem).astype(_F32) * e_k, r1_hi, r1_lo)
            _store_split(_load(b_smem).astype(_F32) * e_k, r2_hi, r2_lo)
            plgpu.commit_smem()
            plgpu.barrier_arrive(bar_right)
            left_a = _load(a_smem).astype(_F32) * jnp.exp(g_before() - mid)
            s_ak = _x3(left_a, r1_hi, r1_lo, transpose=True)
            s_ab = _x3(left_a, r2_hi, r2_lo, transpose=True)
            strict = keep(1)
            a_ak = jnp.where(strict, s_ak, 0.0)
            t_val = _inverse(jnp.where(strict, s_ab, 0.0), i_hi, i_lo)  # (I - A_ab)^-1, block diagonal

            _store_split(_load(a_smem).astype(_F32) * jnp.exp(g_before()) * flag(0), i_hi, i_lo)  # a exp(G_prev) carry
            plgpu.commit_smem()
            t_hi, t_lo = _split_bf16(t_val)
            w_val = _mm(shape, (_Regs(t_hi), i_hi), (_Regs(t_hi), i_lo), (_Regs(t_lo), i_hi))
            ak_hi, ak_lo = _split_bf16(a_ak)
            av_val = _mm(shape, (_Regs(ak_hi), v_smem), (_Regs(ak_lo), v_smem))
            _store_split(av_val, i_hi, i_lo)
            plgpu.commit_smem()
            u_val = _mm(shape, (_Regs(t_hi), i_hi), (_Regs(t_hi), i_lo), (_Regs(t_lo), i_hi))
            st_w[...] = (w_val * -1.0).astype(_BF16)
            st_u[...] = u_val.astype(_BF16)
            if save:  # r1 and r2 stage the split T and A_ak once warpgroup 1's products have read them
                plgpu.barrier_wait(bar_free)
                r1_hi[...], r1_lo[...], r2_hi[...], r2_lo[...] = t_hi, t_lo, ak_hi, ak_lo
            plgpu.commit_smem()
            store_rows(((w_ref, None, st_w), (u_ref, None, st_u)))
            if save:
                for out_ref, src in zip(saved_refs, (r1_hi, r1_lo, r2_hi, r2_lo)):
                    plgpu.copy_smem_to_gmem(src, out_ref.at[bb, h, tok])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _outputs():
            zeros[...] = plgpu.layout_cast(jnp.zeros(shape, _BF16), _WG)
            plgpu.barrier_wait(bar)
            left_q = _load(q_smem).astype(_F32) * scale * jnp.exp(g_now() - g_mid())
            plgpu.barrier_wait(bar_right)
            s_qk = _x3(left_q, r1_hi, r1_lo, transpose=True)
            s_qb = _x3(left_q, r2_hi, r2_lo, transpose=True)
            plgpu.barrier_arrive(bar_free)
            causal = keep(0)

            # The output scores [A_qk | A_qb] of each q row over its own chunk's 32 + 32 columns: the
            # block-diagonal products folded onto 32 columns (the other chunk's block is zero).
            def fold(val: Array) -> Array:  # [64, 64] block diagonal -> [64, 32]: each row's own chunk block
                square[...] = val
                plgpu.commit_smem()  # every thread's part is written (and, below, read) before the next use
                folded = _load(square.at[:, pl.ds(0, CHUNK)]) + _load(square.at[:, pl.ds(CHUNK, CHUNK)])
                plgpu.commit_smem()
                return folded

            for half, val in enumerate((jnp.where(causal, s_qk, 0.0), jnp.where(causal, s_qb, 0.0))):
                st_aqk.at[:, pl.ds(half * CHUNK, CHUNK)][...] = fold(val).astype(_BF16)
            g_val = g_now()
            last = jnp.where(second_rows(), at_token(TOKENS - 1), at_token(CHUNK - 1))
            to_end = jnp.exp(last - g_val) * flag(1)
            st_kk[...] = (_load(k_smem).astype(_F32) * to_end).astype(_BF16)
            st_kb[...] = (_load(b_smem).astype(_F32) * to_end).astype(_BF16)
            st_q[...] = (_load(q_smem).astype(_F32) * scale * jnp.exp(g_val) * flag(0)).astype(_BF16)
            plgpu.commit_smem()
            store_rows(((kg_ref, st_kk, st_kb), (w_ref, zeros, None), (u_ref, v_smem, None), (qg_ref, st_q, zeros),
                        (aqk_ref, st_aqk, zeros)))
            plgpu.wait_smem_to_gmem(0)

        # Both warpgroups finish together (one exiting while the other issues WGMMAs is unsafe on sm_90).
        plgpu.barrier_arrive(bar_done)
        plgpu.barrier_wait(bar_done)

    scratch = (
        _swizzled(shape), _swizzled(shape), _swizzled(shape),  # q, k, v (v is also a wgmma operand)
        *(plgpu.SMEM(shape, _F32) if x.dtype == _F32 else _swizzled(shape) for x in (a, b)),  # a, b (f32 or bf16)
        plgpu.SMEM((_SHIFT + TOKENS, DIM), _F32),  # G at rows 8.., so row 7.. is G one token earlier
        plgpu.SMEM((3, TOKENS), _F32),  # flags
        _swizzled(shape), _swizzled(shape), _swizzled(shape), _swizzled(shape),  # the split right factors
        _swizzled(shape), _swizzled(shape),  # warpgroup 0's split operands (the inverse, W and U)
        plgpu.SMEM((TOKENS, TOKENS), _F32),  # the fold buffer
        _swizzled(shape), _swizzled(shape), _swizzled(shape), _swizzled(shape), _swizzled(shape),  # staging
        plgpu.SMEM((TOKENS, TOKENS), _BF16),  # [A_qk | A_qb] staging (written by column halves)
        plgpu.SMEM(shape, _BF16),  # zeros
        plgpu.Barrier(num_arrivals=7),  # the loads
        plgpu.Barrier(num_arrivals=1),  # the right factors are in SMEM
        plgpu.Barrier(num_arrivals=1),  # warpgroup 1's score products have read them
        plgpu.Barrier(num_arrivals=2),  # both warpgroups done
    )
    kern = _kernel(
        "rwkv7_prep_save" if save else "rwkv7_prep", body,
        out_type=(out,) * 5 + ((saved,) * 4 if save else ()),
        grid=(seq_len // TOKENS, heads, batch), grid_names=("p", "h", "b"), scratch_types=scratch,
        num_threads=2, thread_name="wg",
    )
    return kern(q, k, v, a, b, g, flags)
