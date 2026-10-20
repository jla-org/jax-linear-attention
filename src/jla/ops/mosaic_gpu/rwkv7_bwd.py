# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py (the score
# gradients from operands gated relative to an in-chunk offset), wy_fast_bwd.py (the WY backward through
# T = (I - A_ab)^-1, W and U), chunk_o_bwd.py and chunk_h_bwd.py (the gradients of the state reads and
# writes from the per-chunk states and state gradients).
"""The backward per-chunk work of the RWKV-7 bounded mode in one Mosaic GPU kernel (H100).

:func:`jla.ops.mosaic_gpu.rwkv7._kernel_path` runs the forward as :func:`rwkv7_prep.prep` (``save``) plus
the vector-gate state pass, and the backward as the reversed state pass
(:func:`vector_gate_bwd.bwd_state`: the state gradient ``dS`` leaving every chunk and ``dv_new``) plus
:func:`bwd`, which turns everything per chunk into the input gradients in one kernel, one CTA per two
chunks (64 tokens) and head, without writing any intermediate to HBM:

* the gradients of the state-pass operands from the saved state ``S`` entering each chunk, its
  gradient ``dS`` and ``dv_new`` (bf16 tensor-core products of bf16 values: exact): ``d(q exp(G)) =
  do S^T``, ``dW = dU S^T``, ``d(k to_end) = v dS^T``, ``d(b to_end) = U dS^T`` and the output score
  gradients ``dA_qk = do v^T``, ``dA_qb = do U^T``;
* the WY backward with the forward's saved ``T`` and ``A_ak``: ``dT = dW (a exp(G_prev))^T + dU (A_ak
  v)^T``, ``T^T dW``, ``T^T dU``, ``dA_ak = (T^T dU) v^T``, ``dv += A_ak^T T^T dU`` and ``dA_ab = T^T dT
  T^T``;
* the score backward through the bounded mode's factors ``left = [q exp(G - G_m); a exp(G_prev - G_m)]``
  and ``right = [k exp(G_m - G); b exp(G_m - G)]`` (every exponent at most ``16 |lower_bound|``):
  ``d_left = dA right`` and ``d_right = dA^T left`` for the four masked score gradients ``dA``;
* ``dw`` as the reverse cumulative sum of the per-token log-decay gradient inside each chunk (a product
  with a triangular ones matrix), plus the chunk's carry gradient.

Two warpgroups share the CTA (its per-chunk work is a few thousand dependent register operations, so one
warpgroup per SM left the SM's issue slots mostly idle): warpgroup 0 runs the WY backward and the ``a``
rows of the score backward (``dv``, ``da``, the ``G_prev`` gradient and the ``a`` rows' ``d_right``),
warpgroup 1 at the same time the ``q`` rows (``dq``) and the state-write products, then adds warpgroup 0's
``d_right`` (handed over in SMEM) for ``dk``, ``db`` and ``dw``. Each has its own operand staging buffers.

Every product with an f32 operand is three bf16 tensor-core products of split operands (``hi hi + hi
lo + lo hi``, f32 accumulation: the ``BF16X3`` numerics of the XLA path); ``T`` and ``A_ak`` arrive split.
Head dims 64 only.
"""

from __future__ import annotations

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu.rwkv7_tiles import (COL, ROW, Regs, cols, iota, kernel, load, matmul, matmuls, rows, split_bf16,
                                             swizzled)

Array = jax.Array

__all__ = ["bwd"]

CHUNK = 32  # tokens per chunk (jla.ops.mosaic_gpu.rwkv7.CHUNK)
TOKENS = 2 * CHUNK  # tokens per CTA: two chunks
DIM = 64  # the head dims the kernel supports
_SHIFT = 2  # leading rows of the G buffer (the two chunks' carry gradients): G_prev is G read one row up
_F32 = jnp.float32
_BF16 = jnp.bfloat16
_SHAPE = (TOKENS, DIM)


def _store_split(x: Array, hi_smem, lo_smem) -> None:
    hi, lo = split_bf16(x)
    hi_smem[...] = hi
    lo_smem[...] = lo


def bwd(q: Array, k: Array, v: Array, a: Array, b: Array, g: Array, flags: Array, saved: tuple[Array, ...],
        states: Array, d_states: Array, do: Array, v_new: Array, dv_new: Array, d_carry: Array, *,
        scale: float, grad_dtype=_F32) -> tuple[Array, ...]:
    """``(dq, dk, dv)`` ``[B, T, H, 64]`` bf16 and ``(da, db, dw)`` ``[B, T, H, 64]`` ``grad_dtype`` (f32 or
    bf16: the f32 values rounded to bf16 in the kernel, as XLA would round them).

    ``q, k, v, a, b, g, flags`` are the inputs of :func:`rwkv7_prep.prep` and ``saved`` its four saved
    ``[B, H, T, 64]`` arrays (``T`` and ``A_ak``, split and block diagonal). With ``N = T / 32`` chunks and
    the state pass on 64-row chunks ``[k rows; b rows]``: ``states`` ``[B, H, N, 64, 64]`` bf16 is the
    state entering each chunk, ``d_states`` (same shape) the gradient of the state leaving it, ``v_new``
    and ``dv_new`` ``[B, H, 2 T, 64]`` bf16 the state pass's ``v_new = [v; U]`` and its gradient, ``do``
    ``[B, T, H, 64]`` bf16 the output cotangent and ``d_carry`` ``[B, N, H, 64]`` f32 the gradient of each
    chunk's carry log-decay (``G`` at its last token; zero for a chunk with a segment start).
    """
    batch, seq_len, heads, key_dim = q.shape
    if seq_len % TOKENS or key_dim != DIM or v.shape[-1] != DIM:
        raise ValueError(f"bwd needs T % {TOKENS} == 0 and head dims {DIM}, got {q.shape} and {v.shape}")
    out_bf16 = jax.ShapeDtypeStruct((batch, seq_len, heads, DIM), _BF16)
    out_g = jax.ShapeDtypeStruct((batch, seq_len, heads, DIM), grad_dtype)
    if a.dtype != b.dtype or a.dtype not in (_F32, _BF16) or grad_dtype not in (_F32, _BF16):
        raise ValueError(f"bwd takes a, b of one dtype and a grad_dtype, each f32 or bf16, not {a.dtype}, {b.dtype}, "
                         f"{grad_dtype}")
    bf16_ab, bf16_grads = a.dtype == _BF16, grad_dtype == _BF16

    def body(q_ref, k_ref, v_ref, a_ref, b_ref, g_ref, f_ref, th_ref, tl_ref, ah_ref, al_ref, s_ref, ds_ref,
             do_ref, vn_ref, dvn_ref, dc_ref, dq_ref, dk_ref, dv_ref, da_ref, db_ref, dw_ref, *scratch):
        (q_s, k_s, v_s, a_s, b_s, t_union, ak_union, s_union, ds_s, do_s, un_s, dv_s, stage0, stage1, g_s, f_s,
         bar_wy, bar_rest, bar_dvt, bar_acc, bar_done, *gp) = scratch
        gp_s = gp[0] if bf16_ab else a_s  # the G_prev gradient: in a's f32 buffer, or its own beside bf16 a
        (th_s, tl_s), t_f32 = t_union  # db is staged where T was
        (ah_s, al_s), ak_f32 = ak_union  # da where A_ak was
        s_s, s_f32 = s_union  # dw where the two states were
        if bf16_grads:  # bf16 staging in the first bf16 tile of the same buffers
            da_stage, db_stage, dw_stage = ah_s, th_s, s_s.at[0]
        else:
            da_stage, db_stage, dw_stage = ak_f32, t_f32, s_f32
        # warpgroup 0's d_right of the a rows, handed to warpgroup 1: acc_k where T was, acc_b where da was staged
        acc_k_s, acc_b_s = t_f32, ak_f32
        wg = lax.axis_index("wg")
        pair, h, bb = lax.axis_index("p"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(pair * TOKENS, TOKENS)

        @pl.when(wg == 0)
        def _loads():
            # What both warpgroups start with (bar_wy); the rest (bar_rest) arrives while they compute.
            for src, dst in ((th_ref, th_s), (tl_ref, tl_s), (ah_ref, ah_s), (al_ref, al_s)):
                plgpu.copy_gmem_to_smem(src.at[bb, h, tok], dst, bar_wy)
            for src, dst in ((v_ref, v_s), (a_ref, a_s)):
                plgpu.copy_gmem_to_smem(src.at[bb, tok, h], dst, bar_wy)
            plgpu.copy_gmem_to_smem(g_ref.at[bb, tok, h], g_s.at[pl.ds(_SHIFT, TOKENS)], bar_wy)
            plgpu.copy_gmem_to_smem(f_ref.at[bb, :, tok], f_s, bar_wy)
            for c in range(2):  # chunk c of the pair: its state and its 64 state-pass rows [k rows; b rows]
                n = 2 * pair + c
                half = pl.ds(c * CHUNK, CHUNK)
                plgpu.copy_gmem_to_smem(s_ref.at[bb, h, n], s_s.at[c], bar_wy)
                plgpu.copy_gmem_to_smem(dvn_ref.at[bb, h, pl.ds(n * TOKENS + CHUNK, CHUNK)], dv_s.at[half], bar_wy)
            for src, dst in ((q_ref, q_s), (k_ref, k_s), (b_ref, b_s), (do_ref, do_s)):
                plgpu.copy_gmem_to_smem(src.at[bb, tok, h], dst, bar_rest)
            for c in range(2):
                n = 2 * pair + c
                half = pl.ds(c * CHUNK, CHUNK)
                plgpu.copy_gmem_to_smem(ds_ref.at[bb, h, n], ds_s.at[c], bar_rest)
                plgpu.copy_gmem_to_smem(vn_ref.at[bb, h, pl.ds(n * TOKENS + CHUNK, CHUNK)], un_s.at[half], bar_rest)
                plgpu.copy_gmem_to_smem(dc_ref.at[bb, n, h], g_s.at[c], bar_rest)

        # Few 64 x 64 f32 tiles are live at once in either warpgroup (each is 32 registers per thread):
        # every output is finished as early as it can be, and cheap factors are recomputed from SMEM where
        # they are used instead of being kept.
        def iotas() -> tuple[Array, Array]:
            return iota(_SHAPE, 0), iota(_SHAPE, 1)

        def first_rows() -> Array:  # rows of the pair's first chunk
            return iota(_SHAPE, 0) < CHUNK

        def keep(min_offset: int) -> Array:  # same chunk and document, row - column >= min_offset
            row, col = iotas()
            same_doc = rows(load(f_s.at[2], ROW), _SHAPE) == cols(load(f_s.at[2], COL), _SHAPE)
            offset = jnp.where(jnp.logical_and((row >> 5) == (col >> 5), same_doc), row - col, -1)
            return offset >= min_offset

        def g_tok(t: int) -> Array:  # G at token t of the CTA, per channel, broadcast to every row
            return cols(load(g_s.at[_SHIFT + t], COL), _SHAPE)

        def g_now() -> Array:
            return load(g_s.at[pl.ds(_SHIFT, TOKENS)])

        def g_before() -> Array:  # G_prev: G one token earlier, zero at a chunk start
            start = (iota(_SHAPE, 0) & (CHUNK - 1)) == 0
            return jnp.where(start, 0.0, load(g_s.at[pl.ds(_SHIFT - 1, TOKENS)]))

        def g_mid() -> Array:  # G at each chunk's middle token (the pivot of the bounded mode's factors)
            return jnp.where(first_rows(), g_tok(CHUNK // 2 - 1), g_tok(TOKENS - CHUNK // 2 - 1))

        def carry_ok() -> Array:
            return rows(load(f_s.at[0], ROW), _SHAPE)

        def chunk_pairs(x: Array, ref) -> tuple:  # row t of the bf16 x times ref.at[chunk of t]^T
            first = first_rows()
            return (Regs(jnp.where(first, x, 0.0)), ref.at[0].T), (Regs(jnp.where(first, 0.0, x)), ref.at[1].T)

        def store_rights(st) -> None:  # right_k = k exp(G_m - G) in st[0:2], right_b = b exp(G_m - G) in st[2:4]
            e_k = jnp.exp(g_mid() - g_now())
            _store_split(load(k_s).astype(_F32) * e_k, st[0], st[1])
            _store_split(load(b_s).astype(_F32) * e_k, st[2], st[3])
            plgpu.commit_smem()

        def times_right(st, ct1: Array, ct2: Array) -> Array:  # ct1 right_k + ct2 right_b
            c1h, c1l = split_bf16(ct1)  # one split pair of register operands at a time (fewer registers)
            part = matmul(_SHAPE, (Regs(c1h), st[0]), (Regs(c1h), st[1]), (Regs(c1l), st[0]))
            c2h, c2l = split_bf16(ct2)
            return matmul(_SHAPE, (Regs(c2h), st[2]), (Regs(c2h), st[3]), (Regs(c2l), st[2]), init=part)

        def times_left(st, ct: Array, init: Array | None = None) -> Array:  # ct^T left (the left factor in st[0:2])
            _store_split(ct, st[2], st[3])
            plgpu.commit_smem()
            return matmul(_SHAPE, (st[2].T, st[0]), (st[2].T, st[1]), (st[3].T, st[0]), init=init)

        @pl.when(wg == 0)
        def _wy_and_a_rows():
            p1h, p1l, p2h, p2l = stage0
            plgpu.barrier_wait(bar_wy)
            # --- WY backward: W = T ae (ae = a exp(G_prev) carry_ok), U_wy = T (A_ak v), T = (I - A_ab)^-1 ---
            _store_split(load(a_s).astype(_F32) * jnp.exp(g_before()) * carry_ok(), p2h, p2l)  # ae
            a_v, d_w = matmuls(_SHAPE, ((ah_s, v_s), (al_s, v_s)),  # A_ak v
                               chunk_pairs(load(dv_s), s_s))  # dW = dU S^T (w rows = -W, v_new = u - w S)
            _store_split(a_v, p1h, p1l)
            plgpu.commit_smem()
            dw_hi, dw_lo = split_bf16(d_w)
            d_t, y_u = matmuls(_SHAPE, ((Regs(dw_hi), p2h.T), (Regs(dw_hi), p2l.T), (Regs(dw_lo), p2h.T),
                                        (dv_s, p1h.T), (dv_s, p1l.T)),  # dT = dW ae^T + dU (A_ak v)^T
                               ((th_s.T, dv_s), (tl_s.T, dv_s)))  # T^T dU = d(A_ak v)
            # dU (dv_new's b rows) is read: load dv_new's k rows (dU_top, dv's start) into its buffer
            for c in range(2):
                n = 2 * pair + c
                plgpu.copy_gmem_to_smem(dvn_ref.at[bb, h, pl.ds(n * TOKENS, CHUNK)], dv_s.at[pl.ds(c * CHUNK, CHUNK)],
                                        bar_dvt)
            p2h[...] = dw_hi
            p2l[...] = dw_lo
            plgpu.commit_smem()
            y_w = matmul(_SHAPE, (th_s.T, p2h), (th_s.T, p2l), (tl_s.T, p2h))  # T^T dW = d(ae)
            yu_hi, yu_lo = split_bf16(y_u)
            ct_ak = jnp.where(keep(1), matmul(_SHAPE, (Regs(yu_hi), v_s.T), (Regs(yu_lo), v_s.T)), 0.0)
            p1h[...] = yu_hi
            p1l[...] = yu_lo
            plgpu.commit_smem()
            plgpu.barrier_wait(bar_dvt)
            d_v = matmul(_SHAPE, (ah_s.T, p1h), (ah_s.T, p1l), (al_s.T, p1h), init=load(dv_s).astype(_F32))
            dv_s[...] = d_v.astype(_BF16)  # dv = dU_top + A_ak^T T^T dU, staged where dU_top was
            dt_hi, dt_lo = split_bf16(d_t)
            p2h[...] = dt_hi
            p2l[...] = dt_lo
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dv_s, dv_ref.at[bb, tok, h])
            z = matmul(_SHAPE, (th_s.T, p2h), (th_s.T, p2l), (tl_s.T, p2h))  # T^T dT
            z_hi, z_lo = split_bf16(z)
            ct_ab = jnp.where(keep(1), matmul(_SHAPE, (Regs(z_hi), th_s.T), (Regs(z_hi), tl_s.T),
                                              (Regs(z_lo), th_s.T)), 0.0)

            # --- the a rows of the score backward: d_left_a = dA_ak right_k + dA_ab right_b; then da ---
            plgpu.barrier_wait(bar_rest)
            store_rights(stage0)
            d_left_a = times_right(stage0, ct_ak, ct_ab)
            g_prev = g_before()
            d_a = y_w * (jnp.exp(g_prev) * carry_ok())
            e_a = jnp.exp(g_prev - g_mid())
            d_a = d_a + d_left_a * e_a
            da_stage[...] = d_a.astype(grad_dtype)
            a_val = load(a_s).astype(_F32)
            _store_split(a_val * e_a, p1h, p1l)  # left_a = a exp(G_prev - G_m), for d_right below
            # The gradient of G_prev, kept in a's buffer (a is not read again) for warpgroup 1's dw.
            gp_s[...] = jnp.where((iota(_SHAPE, 0) & (CHUNK - 1)) == 0, 0.0, d_a * a_val)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(da_stage, da_ref.at[bb, tok, h])
            # d_right from the a rows: dA_ak^T left_a, dA_ab^T left_a, handed to warpgroup 1 in SMEM
            acc_k_s[...] = times_left(stage0, ct_ak)
            acc_b = times_left(stage0, ct_ab)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # da has left its staging buffer
            acc_b_s[...] = acc_b
            plgpu.commit_smem()
            plgpu.barrier_arrive(bar_acc)
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _q_rows_and_rest():
            r1h, r1l, r2h, r2l = stage1
            plgpu.barrier_wait(bar_wy)
            plgpu.barrier_wait(bar_rest)
            # --- the q rows: d(q exp(G) carry_ok) = do S^T, dA_qk = do v^T, dA_qb = do U^T; then dq ---
            causal = keep(0)
            ct_qk = jnp.where(causal, matmul(_SHAPE, (do_s, v_s.T)), 0.0)  # do v_new^T on the output scores
            ct_qb = jnp.where(causal, matmul(_SHAPE, (do_s, un_s.T)), 0.0)
            store_rights(stage1)
            d_left_q = times_right(stage1, ct_qk, ct_qb)
            d_qg = matmul(_SHAPE, *chunk_pairs(load(do_s), s_s))
            g_val = g_now()
            d_q = d_qg * (jnp.exp(g_val) * carry_ok()) + d_left_q * jnp.exp(g_val - g_mid())  # dq / scale
            do_s[...] = (d_q * scale).astype(_BF16)  # dq, staged where do was
            q_val = load(q_s).astype(_F32) * scale
            dg_rest = d_q * q_val  # the gradient of G (more terms below)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(do_s, dq_ref.at[bb, tok, h])
            _store_split(q_val * jnp.exp(g_now() - g_mid()), r1h, r1l)  # left_q
            acc_bq = times_left(stage1, ct_qb)
            acc_kq = times_left(stage1, ct_qk)

            # --- the state writes: kg = [k to_end; b to_end], to_end = exp(G_last - G) same_last; then dk, db ---
            g_val = g_now()
            last = jnp.where(first_rows(), g_tok(CHUNK - 1), g_tok(TOKENS - 1))
            to_end = jnp.exp(last - g_val) * rows(load(f_s.at[1], ROW), _SHAPE)
            e_k = jnp.exp(g_mid() - g_val)
            d_kk = matmul(_SHAPE, *chunk_pairs(load(v_s), ds_s))  # v_new dS^T on the k rows (v_new = v)
            k_val = load(k_s).astype(_F32)
            dk_q = d_kk * to_end + acc_kq * e_k  # dk without the a rows' d_right
            y_kb = d_kk * k_val * to_end  # -dG from to_end; + its chunk sum at G_last
            dg_rest = dg_rest - acc_kq * k_val * e_k
            d_bb = matmul(_SHAPE, *chunk_pairs(load(un_s), ds_s))  # and on the b rows (v_new = U)
            b_val = load(b_s).astype(_F32)
            db_q = d_bb * to_end + acc_bq * e_k
            y_kb = y_kb + d_bb * b_val * to_end
            dg_rest = dg_rest - acc_bq * b_val * e_k
            # --- dw: reverse cumulative sum inside each chunk of dG + (dG_prev shifted) + G_last terms ---
            # dw_t = sum_{s >= t} (dG_s + dGp_s) - dGp_t + sum_{s < t} y_kb_s + d_carry (sums inside the chunk).
            # What does not need warpgroup 0 is done before waiting for it: y_kb's split and the masks.
            _store_split(y_kb, r2h, r2l)
            row, col = iotas()
            same_chunk = (row >> 5) == (col >> 5)
            upper = jnp.where(jnp.logical_and(same_chunk, col >= row), 1.0, 0.0).astype(_BF16)
            lower = jnp.where(jnp.logical_and(same_chunk, col < row), 1.0, 0.0).astype(_BF16)
            plgpu.barrier_wait(bar_acc)  # warpgroup 0's acc_k, acc_b and G_prev gradient are in SMEM
            acc_k = load(acc_k_s)
            un_s[...] = (dk_q + acc_k * e_k).astype(_BF16)  # dk, staged where U was
            dg_rest = dg_rest - acc_k * load(k_s).astype(_F32) * e_k
            acc_b = load(acc_b_s)
            db_stage[...] = (db_q + acc_b * e_k).astype(grad_dtype)  # where acc_k was (read above)
            dg_rest = dg_rest - acc_b * load(b_s).astype(_F32) * e_k
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(un_s, dk_ref.at[bb, tok, h])
            plgpu.copy_smem_to_gmem(db_stage, db_ref.at[bb, tok, h])
            dgp = load(gp_s)
            _store_split(dg_rest + dgp, r1h, r1l)
            plgpu.commit_smem()
            d_w = matmul(_SHAPE, (Regs(upper), r1h), (Regs(upper), r1l), (Regs(lower), r2h), (Regs(lower), r2l))
            carry = jnp.where(first_rows(), cols(load(g_s.at[0], COL), _SHAPE), cols(load(g_s.at[1], COL), _SHAPE))
            dw_stage[...] = (d_w - dgp + carry).astype(grad_dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dw_stage, dw_ref.at[bb, tok, h])
            plgpu.wait_smem_to_gmem(0)

        # Both warpgroups finish together (one exiting while the other issues WGMMAs is unsafe on sm_90).
        plgpu.barrier_arrive(bar_done)
        plgpu.barrier_wait(bar_done)

    pair_tiles = (swizzled(_SHAPE), swizzled(_SHAPE))
    scratch = (
        swizzled(_SHAPE), swizzled(_SHAPE), swizzled(_SHAPE),  # q, k, v (v is also a wgmma operand)
        # a, b (with f32 a, a's buffer later holds the G_prev gradient)
        *((swizzled(_SHAPE), swizzled(_SHAPE)) if bf16_ab else (plgpu.SMEM(_SHAPE, _F32), plgpu.SMEM(_SHAPE, _F32))),
        plgpu.RefUnion(pair_tiles, plgpu.SMEM(_SHAPE, _F32)),  # T (hi, lo), later acc_k, then the db staging
        plgpu.RefUnion(pair_tiles, plgpu.SMEM(_SHAPE, _F32)),  # A_ak (hi, lo), later the da staging, then acc_b
        plgpu.RefUnion(swizzled((2, DIM, DIM)), plgpu.SMEM(_SHAPE, _F32)),  # the two states, later dw
        swizzled((2, DIM, DIM)),  # the two state gradients
        swizzled(_SHAPE), swizzled(_SHAPE),  # do (later dq), U (later dk)
        swizzled(_SHAPE),  # dv_new on the b rows (dU), then on the k rows (dU_top), then the dv staging
        tuple(swizzled(_SHAPE) for _ in range(4)),  # warpgroup 0's two split operand pairs
        tuple(swizzled(_SHAPE) for _ in range(4)),  # warpgroup 1's
        # rows 0, 1: the two chunks' carry gradients; G at rows 2.., so rows 1.. are G one token earlier (row 1
        # is read at a chunk start only, where G_prev is zero)
        plgpu.SMEM((_SHIFT + TOKENS, DIM), _F32),
        plgpu.SMEM((3, TOKENS), _F32),  # flags
        plgpu.Barrier(num_arrivals=12),  # what the WY backward and the q rows start with
        plgpu.Barrier(num_arrivals=10),  # the rest
        plgpu.Barrier(num_arrivals=2),  # dU_top
        plgpu.Barrier(num_arrivals=1),  # warpgroup 0's d_right handed to warpgroup 1
        plgpu.Barrier(num_arrivals=2),  # both warpgroups done
        *((plgpu.SMEM(_SHAPE, _F32),) if bf16_ab else ()),  # the G_prev gradient (in a's buffer with f32 a)
    )
    kern = kernel(
        "rwkv7_bwd", body, out_type=(out_bf16, out_bf16, out_bf16, out_g, out_g, out_g),
        grid=(seq_len // TOKENS, heads, batch), grid_names=("p", "h", "b"), scratch_types=scratch,
        num_threads=2, thread_name="wg",
    )
    return kern(q, k, v, a, b, g, flags, *saved, states, d_states, do, v_new, dv_new, d_carry)
