# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/generalized_delta_rule/dplr/chunk.py,
# fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py, wy_fast_fwd.py, chunk_h_fwd.py, chunk_o_fwd.py,
# chunk_h_bwd.py and fla/ops/rwkv7/chunk.py.
"""RWKV-7 (``kind="rwkv7"``, the DPLR delta rule) with its serial passes on Mosaic GPU kernels.

The recurrence (see :mod:`jla.ops.rwkv7`) is ``S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t^T S_{t-1})
+ k_t v_t^T`` and ``o_t = scale q_t^T S_t``. Per chunk of ``C = 32`` tokens with ``S`` the state
entering it, FLA's DPLR algorithm gives the low-rank reads ``U = W S + U_wy`` (``W = T (a
exp(G_{t-1}))``, ``U_wy = T A_ak V``, ``T = (I - A_ab)^-1``), the output ``o = (q exp(G)) S + A_qk V
+ A_qb U`` and the next state ``exp(G_last) S + (k exp(G_last - G))^T V + (b exp(G_last - G))^T U``.

That is the vector-gate state pass of :mod:`jla.ops.mosaic_gpu.vector_gate` on a 64-row chunk whose
rows are the chunk's 32 ``k`` rows followed by its 32 ``b`` rows: keys ``[k_end; b_end]``, values
``v_new = [V; U] = [V; U_wy] - [0; -W] S``, queries ``[q exp(G); 0]`` and scores ``[[A_qk, A_qb];
[0, 0]]``. So the serial pass forward (:func:`vector_gate.state_output`, the state carried in
registers by one warpgroup while a second computes the output) and backward
(:func:`vector_gate_bwd.bwd_state`, the reversed state-gradient pass) run on the existing H100
kernels unchanged. Everything parallel over chunks runs in XLA and is differentiated by JAX: the
four decayed score matrices (8-token sub-chunks: exact pairwise decays inside each diagonal
8 x 8 block, with a hand-written VJP, a factorisation through the token before the row's
sub-chunk elsewhere, so every exponent is at most zero and no ``[B, H, N, C, C, K]`` tensor is
formed; or, with ``lower_bound``, one dense product per chunk), the WY inverse
(:func:`_unit_lower_inverse`, batched matmuls) and the state-pass operands; the products of the backward
through the state pass (``dq``-, ``dw``-, ``dk``-side and the score gradients) are batched XLA
matmuls on the saved per-chunk states. In the bounded mode with its default products and head dims 64
(the RWKV-7 layers' call) the per-chunk work runs on Mosaic GPU kernels end to end instead
(:func:`_kernel_path`): the forward in :mod:`jla.ops.mosaic_gpu.rwkv7_prep`, which saves ``T`` and
``A_ak``, and the backward in :mod:`jla.ops.mosaic_gpu.rwkv7_bwd` after the reversed state pass, with
only ``dv_o = A^T do``, the carry gradient and padding in XLA. (:func:`_prep_operands`, the same forward
kernel with the per-chunk backward as the XLA VJP of :func:`_operands`, is kept as a reference:
``_run(..., kernel_backward=False)``.)

``segment_ids`` restart the state with exact masks (as :func:`jla.ops.rwkv7.rwkv7_chunked`): the
state pass sees a zero carry decay for a chunk with a segment start. The serial pass uses bf16
tensor-core operands with f32 accumulation and an f32 state, like the KDA kernels.

Registered for ``impl="auto"``: calls that :func:`supports_rwkv7` accepts (bf16 ``q, k, v``, no
``chunk_size`` or 32, ...) run here, everything else falls back to the chunked op. The RWKV-7 layer
(:mod:`jla.layers.rwkv7`) passes ``r, k, v`` in its compute dtype, so a bf16 layer with ``chunk_size=32``
runs here and one with its default ``chunk_size=16`` runs the chunked op.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops import register
from jla.ops.mosaic_gpu import rwkv7_bwd, rwkv7_prep
from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_bwd as vgb
from jla.ops.mosaic_gpu.vector_gate_ops import HEAD_DIMS, _forward_state_config, _pieces, _split_state_config, is_hopper
from jla.ops.rwkv7 import _check, segment_resets

Array = jax.Array

__all__ = ["BF16X3", "BOUNDED_PRECISION", "CHUNK", "rwkv7", "supports_rwkv7", "unsupported_reason"]

CHUNK = vg.CHUNK // 2  # 32 tokens: one chunk is one 64-row chunk [k rows; b rows] of the state kernels
SUB_CHUNK = 8
_F32 = jnp.float32
_BF16 = jnp.bfloat16
_HIGHEST = lax.Precision.HIGHEST
# f32 products as three bf16 tensor-core products of split operands (hi hi + hi lo + lo hi, f32
# accumulation): about 16 significant bits per operand instead of 24, on the tensor cores.
BF16X3 = lax.DotAlgorithmPreset.BF16_BF16_F32_X3
# The bounded mode's products when ``precision=None`` (see :func:`rwkv7`).
BOUNDED_PRECISION = BF16X3
# Carry log-decay of a chunk that contains a segment start: exp() of it is exactly zero in f32.
_NO_CARRY = -1.0e4


# ---------------------------------------------------------------------------------------------
# the serial pass: vector_gate.state_output forward, vector_gate_bwd.bwd_state backward
# ---------------------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _state_pass(cfg: vg.Config, kg, w, u, g_last, h0, qg, aqk):
    """``(o [B, 2T, H, V] bf16, final [B, H, K, V] f32)`` of :func:`vector_gate.state_output`."""
    return vg.state_output(_forward_state_config(cfg, fused=False), kg, w, u, g_last, h0, qg, aqk, save=False)


def _state_pass_fwd(cfg: vg.Config, kg, w, u, g_last, h0, qg, aqk):
    o, final, states, v_new = vg.state_output(
        _forward_state_config(cfg, fused=False), kg, w, u, g_last, h0, qg, aqk, save=True
    )
    return (o, final), (kg, w, g_last, h0, qg, aqk, states, v_new)


def _state_pass_bwd(cfg: vg.Config, residuals, cotangents):
    """Reverse pass on the kernel (``dS``, ``dv_new``, ``dh0``), the per-chunk products in XLA."""
    kg, w, g_last, h0, qg, aqk, states, v_new = residuals
    do, d_final = cotangents
    c, nc = vg.CHUNK, cfg.num_chunks
    b_, h_, vd = cfg.batch, cfg.heads, cfg.value_dim
    do = do.astype(_BF16)
    mm = functools.partial(jnp.einsum, preferred_element_type=_F32)
    do_c = jnp.swapaxes(do, 1, 2).reshape(b_, h_, nc, c, vd)  # [B, H, N, 2C, V]
    a_c = aqk.reshape(b_, h_, nc, c, c)
    dv_o = mm("bhnti,bhntv->bhniv", a_c, do_c).astype(_BF16).reshape(b_, h_, nc * c, vd)
    dh, dv_new, dh0 = vgb.bwd_state(cfg, qg, kg, w, g_last, do, dv_o, d_final.astype(_F32))
    vn_c = v_new.reshape(b_, h_, nc, c, vd)
    dvn_c = dv_new.reshape(b_, h_, nc, c, vd)
    d_qg = mm("bhntv,bhnkv->bhntk", do_c, states)  # o = qg S + A v_new
    d_w = -mm("bhntv,bhnkv->bhntk", dvn_c, states)  # v_new = u - w S
    d_kg = mm("bhntv,bhnkv->bhntk", vn_c, dh)  # S' = exp(G_last) S + kg^T v_new
    d_aqk = mm("bhntv,bhniv->bhnti", do_c, vn_c)
    d_gl = jnp.exp(jnp.swapaxes(g_last, 1, 2)) * mm("bhnkv,bhnkv->bhnk", states, dh)
    flat = lambda x, like: x.reshape(like.shape).astype(like.dtype)  # noqa: E731
    return (
        flat(d_kg, kg), flat(d_w, w), dv_new.astype(_BF16), jnp.swapaxes(d_gl, 1, 2).astype(_F32),
        None if h0 is None else dh0, flat(d_qg, qg), flat(d_aqk, aqk),
    )


_state_pass.defvjp(_state_pass_fwd, _state_pass_bwd)


# ---------------------------------------------------------------------------------------------
# the parallel part: decayed score matrices, WY solve, state-pass operands (XLA)
# ---------------------------------------------------------------------------------------------


def _block_decay(gxb: Array, gb: Array, strict: bool) -> Array:
    """``exp(gx_tk - g_ik)`` on the causal (``strict``: strictly causal) part of each diagonal block."""
    t = jnp.arange(gxb.shape[-2])
    keep = (t[:, None] > t[None, :]) if strict else (t[:, None] >= t[None, :])
    return jnp.exp(jnp.where(keep[:, :, None], gxb[..., :, None, :] - gb[..., None, :, :], -jnp.inf))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _diag_blocks(xb: Array, gxb: Array, yb: Array, gb: Array, strict: bool, precision) -> Array:
    """``D[..., t, i] = sum_k x_tk y_ik exp(gx_tk - g_ik)`` on the diagonal ``8 x 8`` blocks (exact)."""
    return jnp.einsum("...tk,...tik,...ik->...ti", xb, _block_decay(gxb, gb, strict), yb, precision=precision)


def _diag_blocks_fwd(xb, gxb, yb, gb, strict, precision):
    return _diag_blocks(xb, gxb, yb, gb, strict, precision), (xb, gxb, yb, gb)


def _diag_blocks_bwd(strict, precision, residuals, ct):
    # Two contractions with the recomputed decay; the log-decay gradients follow from them
    # (d gx_tk = x_tk dx_tk, d g_ik = -y_ik dy_ik). Autodiff instead forms the cotangent of the
    # [..., 8, 8, K] decay and XLA fused it with every other score gradient into one slow loop.
    xb, gxb, yb, gb = residuals
    decay = _block_decay(gxb, gb, strict)
    dx = jnp.einsum("...ti,...tik,...ik->...tk", ct, decay, yb, precision=precision)
    dy = jnp.einsum("...ti,...tik,...tk->...ik", ct, decay, xb, precision=precision)
    dx, dy = lax.optimization_barrier((dx, dy))
    return dx, xb * dx, dy, -yb * dy


_diag_blocks.defvjp(_diag_blocks_fwd, _diag_blocks_bwd)


def _decayed_scores(x: Array, gx: Array, y: Array, g: Array, *, strict: bool, precision) -> Array:
    """``R[t, i] = sum_c x_tc y_ic exp(gx_tc - g_ic)`` for ``i < t`` (``strict``) or ``i <= t``.

    ``x, gx, y, g`` are ``[..., C, K]`` f32; ``g`` is the chunk-local cumulative log decay (at most
    zero, non-increasing) and ``gx_t`` is ``g`` at token ``t`` (or ``t - 1`` with ``strict``), so every
    exponent formed is at most zero. Rows of sub-chunk ``I`` reach the columns of earlier
    sub-chunks through ``g`` at the token before ``I``; the diagonal ``8 x 8`` blocks use exact
    pairwise differences.
    """
    *lead, c, _ = x.shape
    s = SUB_CHUNK
    nb = c // s

    def blocks(z):
        return z.reshape(*lead, nb, s, z.shape[-1])

    xb, gxb, yb, gb = blocks(x), blocks(gx), blocks(y), blocks(g)
    exact = _HIGHEST if precision is None else precision
    diag = _diag_blocks(xb, gxb, yb, gb, strict, exact)  # [..., nb, s, s]
    ends = gb[..., -1, :]  # [..., nb, K]: g at the last token of each sub-chunk
    ref = jnp.concatenate([jnp.zeros_like(ends[..., :1, :]), ends[..., :-1, :]], axis=-2)  # before sub-chunk I
    rows = xb * jnp.exp(gxb - ref[..., :, None, :])  # [..., nb, s, K]
    col_block = jnp.arange(c) // s
    before = col_block[None, :] < jnp.arange(nb)[:, None]  # [nb, C]: column i precedes sub-chunk I
    cols = y[..., None, :, :] * jnp.exp(
        jnp.where(before[:, :, None], ref[..., :, None, :] - g[..., None, :, :], -jnp.inf)
    )  # [..., nb, C, K]
    if precision is None:  # bf16 tensor-core operands, f32 accumulation (the vector-gate kernels' numerics)
        off = jnp.einsum("...ntk,...nik->...nti", rows.astype(_BF16), cols.astype(_BF16), preferred_element_type=_F32)
    else:
        off = jnp.einsum("...ntk,...nik->...nti", rows, cols, precision=precision)  # [..., nb, s, C]
    own = col_block[None, :] == jnp.arange(nb)[:, None]  # [nb, C]
    full = jnp.where(own[:, None, :], jnp.tile(diag, (1,) * (diag.ndim - 1) + (nb,)), off)
    return full.reshape(*lead, c, c)


def _causal(c: int, strict: bool) -> Array:
    t = jnp.arange(c)
    return (t[:, None] > t[None, :]) if strict else (t[:, None] >= t[None, :])


def _bounded_f32_factors(q: Array, a: Array, k: Array, b: Array, g_inc: Array, g_exc: Array):
    """``exp(G - G_m), exp(G_prev - G_m), exp(G_m - G)`` and the stacked factors of :func:`_bounded_f32_scores`."""
    c = q.shape[-2]
    mid = g_inc[..., c // 2 - 1 : c // 2, :]
    e_q, e_a, e_k = jnp.exp(g_inc - mid), jnp.exp(g_exc - mid), jnp.exp(mid - g_inc)
    left = jnp.concatenate([q * e_q, a * e_a], axis=-2)  # [..., 2C, K]
    right = jnp.concatenate([k * e_k, b * e_k], axis=-2)  # [..., 2C, K]
    return e_q, e_a, e_k, left, right


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _bounded_f32_scores(q: Array, a: Array, k: Array, b: Array, g_inc: Array, g_exc: Array, precision):
    """The four bounded-mode score matrices of a chunk from ONE f32 product (``precision``: ``HIGHEST``
    or :data:`BF16X3`).

    ``[q exp(G - G_m); a exp(G_prev - G_m)] [k exp(G_m - G); b exp(G_m - G)]^T`` is ``[[A_qk, A_qb],
    [A_ak, A_ab]]`` before the causal (``A_q*``) and strictly causal (``A_a*``) masks: one ``[2C, K] x
    [K, 2C]`` product per chunk instead of four ``[C, K] x [K, C]`` ones. The VJP recomputes the factors
    (autodiff would save them) and gives the pivot ``G_m`` no gradient: it cancels from every score.
    """
    _, _, _, left, right = _bounded_f32_factors(q, a, k, b, g_inc, g_exc)
    full = jnp.einsum("...tk,...ik->...ti", left, right, precision=precision)
    c = q.shape[-2]
    causal, strict = _causal(c, False), _causal(c, True)
    return (jnp.where(causal, full[..., :c, :c], 0.0), jnp.where(causal, full[..., :c, c:], 0.0),
            jnp.where(strict, full[..., c:, :c], 0.0), jnp.where(strict, full[..., c:, c:], 0.0))


def _bounded_f32_scores_fwd(q, a, k, b, g_inc, g_exc, precision):
    return _bounded_f32_scores(q, a, k, b, g_inc, g_exc, precision), (q, a, k, b, g_inc, g_exc)


def _bounded_f32_scores_bwd(precision, residuals, cotangents):
    q, a, k, b, g_inc, g_exc = residuals
    c = q.shape[-2]
    e_q, e_a, e_k, left, right = _bounded_f32_factors(q, a, k, b, g_inc, g_exc)
    causal, strict = _causal(c, False), _causal(c, True)
    ct_qk, ct_qb, ct_ak, ct_ab = cotangents
    ct = jnp.concatenate([
        jnp.concatenate([jnp.where(causal, ct_qk, 0.0), jnp.where(causal, ct_qb, 0.0)], axis=-1),
        jnp.concatenate([jnp.where(strict, ct_ak, 0.0), jnp.where(strict, ct_ab, 0.0)], axis=-1),
    ], axis=-2)  # [..., 2C, 2C]
    d_left = jnp.einsum("...ti,...ik->...tk", ct, right, precision=precision)
    d_right = jnp.einsum("...ti,...tk->...ik", ct, left, precision=precision)
    dl_q, dl_a, dr_k, dr_b = d_left[..., :c, :], d_left[..., c:, :], d_right[..., :c, :], d_right[..., c:, :]
    d_g_inc = dl_q * left[..., :c, :] - dr_k * right[..., :c, :] - dr_b * right[..., c:, :]
    return dl_q * e_q, dl_a * e_a, dr_k * e_k, dr_b * e_k, d_g_inc, dl_a * left[..., c:, :]


_bounded_f32_scores.defvjp(_bounded_f32_scores_fwd, _bounded_f32_scores_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _scores_given(q, a, k, b, g_inc, g_exc, a_ak, precision):
    """:func:`_bounded_f32_scores` in the backward of :func:`_prep_operands`, with the kernel's ``A_ak``.

    There only ``A_ak`` is used as a value (by ``A_ak V``): ``A_qk``/``A_qb`` feed the score tiles, whose
    recomputed values are discarded, and ``A_ab`` the inverse, which is also given (:func:`_inverse_given`).
    So those three are zeros and the product is skipped; the VJP is :func:`_bounded_f32_scores`' (it
    recomputes the factors from the inputs and needs no score value).
    """
    zero = jnp.zeros_like(a_ak)
    return zero, zero, a_ak, zero


def _scores_given_fwd(q, a, k, b, g_inc, g_exc, a_ak, precision):
    return _scores_given(q, a, k, b, g_inc, g_exc, a_ak, precision), (q, a, k, b, g_inc, g_exc, a_ak)


def _scores_given_bwd(precision, residuals, cotangents):
    *inputs, a_ak = residuals
    return (*_bounded_f32_scores_bwd(precision, tuple(inputs), cotangents), jnp.zeros_like(a_ak))


_scores_given.defvjp(_scores_given_fwd, _scores_given_bwd)


# Most negative ``lower_bound`` of the bounded mode: 16 tokens of decay stay within e^40.
MIN_LOWER_BOUND = -2.5


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _to_chunks(x: Array, c: int) -> Array:
    """``[B, N C, H, D]`` to ``[B, H, N, C, D]``."""
    batch, seq_len, heads, dim = x.shape
    return x.reshape(batch, seq_len // c, c, heads, dim).transpose(0, 3, 1, 2, 4)


def _to_chunks_fwd(x: Array, c: int):
    return _to_chunks(x, c), None


def _to_chunks_bwd(c: int, _, ct: Array):
    # The barrier materializes the summed cotangent in the chunk layout before the transpose: XLA
    # otherwise fuses every gradient term of every input into one loop fusion that reads its ~30
    # operands through the transpose (22 of 23 ms of the backward at B=8, T=4096, H=12 on H100).
    ct = lax.optimization_barrier(ct)
    batch, heads, n, _, dim = ct.shape
    return (ct.transpose(0, 2, 3, 1, 4).reshape(batch, n * c, heads, dim),)


_to_chunks.defvjp(_to_chunks_fwd, _to_chunks_bwd)


def _series_inverse(a: Array, precision) -> Array:
    """``(I - A)^-1 = sum_{m < C} A^m = (I + A)(I + A^2)(I + A^4) ...`` for strictly lower ``A [..., C, C]``
    (nilpotent): ``2 log2(C) - 2`` batched matmuls."""
    t = jnp.eye(a.shape[-1], dtype=a.dtype) + a
    power = a
    for _ in range((a.shape[-1] - 1).bit_length() - 1):
        power = jnp.einsum("...ij,...jk->...ik", power, power, precision=precision)
        t = t + jnp.einsum("...ij,...jk->...ik", power, t, precision=precision)
    return t


def _block_inverse(a: Array, precision) -> Array:
    """:func:`_unit_lower_inverse`'s value: the two diagonal half blocks inverted recursively in one batch,
    ``T_21 = T_22 A_21 T_11``, and :func:`_series_inverse` from 8 x 8 blocks down. Every factor is bounded
    by the inverse; the series on a larger block is not (its powers grow like ``C(C - 2, C / 2 - 1) a^(C/2)``
    when ``A`` is near ``-a`` on every entry, as for RWKV-7 with nearly parallel ``kk``)."""
    c = a.shape[-1]
    if c % 2 or c < 16:
        return _series_inverse(a, precision)
    h = c // 2
    diag = _block_inverse(jnp.stack([a[..., :h, :h], a[..., h:, h:]], axis=-3), precision)  # [..., 2, h, h]
    t11, t22 = diag[..., 0, :, :], diag[..., 1, :, :]
    t21 = jnp.einsum("...ij,...jk->...ik", t22, a[..., h:, :h], precision=precision)
    t21 = jnp.einsum("...ij,...jk->...ik", t21, t11, precision=precision)
    top = jnp.concatenate([t11, jnp.zeros_like(t11)], axis=-1)
    return jnp.concatenate([top, jnp.concatenate([t21, t22], axis=-1)], axis=-2)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _unit_lower_inverse(a: Array, precision) -> Array:
    """``(I - A)^-1`` for strictly lower ``A [..., C, C]`` (a batched triangular solve is several times
    slower on H100 for 32 x 32 blocks): :func:`_block_inverse`, fewer and smaller batched products than
    the series on the whole block, and stable. The VJP is the inverse's, ``dA = T^T dT T^T`` (two products
    instead of the transposed chain)."""
    return _block_inverse(a, precision)


def _unit_lower_inverse_fwd(a: Array, precision):
    t = _unit_lower_inverse(a, precision)
    return t, t


def _unit_lower_inverse_bwd(precision, t: Array, ct: Array):
    # T = (I - A)^-1 gives dT = T dA T. Only the strictly lower part of A is an input (the series is the
    # inverse only for a strictly lower A).
    d = jnp.einsum("...ji,...jk->...ik", t, ct, precision=precision)  # T^T ct
    d = jnp.einsum("...ij,...kj->...ik", d, t, precision=precision)  # T^T ct T^T
    return (jnp.tril(d, -1),)


_unit_lower_inverse.defvjp(_unit_lower_inverse_fwd, _unit_lower_inverse_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _inverse_given(a: Array, t: Array, precision) -> Array:
    """``(I - A)^-1`` given as ``t`` (the kernel's, in the backward of :func:`_prep_operands`), with the
    VJP of :func:`_unit_lower_inverse` in ``a`` (it needs only ``t``)."""
    return t


def _inverse_given_fwd(a: Array, t: Array, precision):
    return t, t


def _inverse_given_bwd(precision, t: Array, ct: Array):
    return _unit_lower_inverse_bwd(precision, t, ct)[0], jnp.zeros_like(t)


_inverse_given.defvjp(_inverse_given_fwd, _inverse_given_bwd)


def _doc_ids(segment_ids: Array | None, batch: int, seq_len: int) -> Array:
    """``[B, T]`` int32 document index of every token (0 without ``segment_ids``)."""
    if segment_ids is None:
        return jnp.zeros((batch, seq_len), jnp.int32)
    return jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)


def _operands(q, k, v, w, a, b, scale, doc, precision, bounded, saved=None):
    """The state-pass operands ``(kg, w, u, g_last, qg, aqk)`` (64-row chunks) and ``n_chunks``
    (``doc``: :func:`_doc_ids`). ``saved = (T, A_ak)`` (bounded mode, the backward of
    :func:`_prep_operands`) replaces the score product and the inverse by the kernel's values."""
    batch, seq_len, heads, _ = q.shape
    c = CHUNK
    n = -(-seq_len // c)
    pad = n * c - seq_len

    # Padding tokens have k = v = a = b = 0 and w = 0, so they leave the state unchanged.
    def prep(x):
        return _to_chunks(jnp.pad(x.astype(_F32), ((0, 0), (0, pad), (0, 0), (0, 0))), c)  # [B, H, N, C, D]

    q_c = prep(q) * scale
    k_c, v_c, w_c, a_c, b_c = (prep(x) for x in (k, v, w, a, b))
    doc = jnp.pad(doc, ((0, 0), (0, pad)), mode="edge").reshape(batch, n, c)
    doc_in = jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), doc[:, :-1, -1]], axis=1)  # doc of the token before
    carry_ok = (doc == doc_in[:, :, None])[:, None, :, :, None].astype(_F32)  # [B, 1, N, C, 1]
    same = (doc[:, :, :, None] == doc[:, :, None, :])[:, None]  # [B, 1, N, C, C]
    same_last = (doc == doc[:, :, -1:])[:, None, :, :, None].astype(_F32)

    g_inc = jnp.cumsum(w_c, axis=3)  # G_t = sum_{j <= t} w_j inside the chunk
    g_exc = jnp.concatenate([jnp.zeros_like(g_inc[..., :1, :]), g_inc[..., :-1, :]], axis=3)  # G_{t-1}
    if bounded:  # the four matrices from one product per chunk
        precision = BOUNDED_PRECISION if precision is None else precision
        if saved is None:
            four = _bounded_f32_scores(q_c, a_c, k_c, b_c, g_inc, g_exc, precision)
        else:
            four = _scores_given(q_c, a_c, k_c, b_c, g_inc, g_exc, saved[1], precision)
        a_qk, a_qb, a_ak, a_ab = (jnp.where(same, x, 0.0) for x in four)
    else:
        scores = functools.partial(_decayed_scores, g=g_inc, precision=precision)
        a_qk = jnp.where(same, scores(q_c, g_inc, k_c, strict=False), 0.0)
        a_qb = jnp.where(same, scores(q_c, g_inc, b_c, strict=False), 0.0)
        a_ak = jnp.where(same, scores(a_c, g_exc, k_c, strict=True), 0.0)
        a_ab = jnp.where(same, scores(a_c, g_exc, b_c, strict=True), 0.0)
        precision = _HIGHEST if precision is None else precision  # the WY products stay f32 by default

    # WY: U = A_ab U + (a exp(G_{t-1})) S + A_ak V  =>  U = W S + U_wy.
    t_ab = _unit_lower_inverse(a_ab, precision) if saved is None else _inverse_given(a_ab, saved[0], precision)
    a_v = jnp.einsum("...ti,...iv->...tv", a_ak, v_c, precision=precision)
    key_dim = a_c.shape[-1]
    wu = jnp.einsum("...ti,...iz->...tz", t_ab,  # W = T (a exp(G_{t-1})) and U_wy = T A_ak V in one product
                    jnp.concatenate([a_c * jnp.exp(g_exc) * carry_ok, a_v], axis=-1), precision=precision)
    w_wy, u_wy = wu[..., :key_dim], wu[..., key_dim:]
    q_dec = q_c * jnp.exp(g_inc) * carry_ok
    g_last = g_inc[..., -1, :]  # [B, H, N, K]
    to_end = jnp.exp(g_last[..., None, :] - g_inc) * same_last  # to the chunk end, last document only
    g_carry = jnp.where(carry_ok[..., -1, :] > 0, g_last, _NO_CARRY)

    def rows64(top, bottom):  # two [B, H, N, C, D] halves -> [B, H, N * 2C, D] bf16
        x = jnp.concatenate([top, bottom], axis=3)
        return x.reshape(batch, heads, n * 2 * c, x.shape[-1]).astype(_BF16)

    kg = rows64(k_c * to_end, b_c * to_end)
    w_rows = rows64(jnp.zeros_like(w_wy), -w_wy)
    u_rows = rows64(v_c, u_wy)
    qg = rows64(q_dec, jnp.zeros_like(q_dec))
    aqk = rows64(jnp.concatenate([a_qk, a_qb], axis=-1), jnp.zeros((batch, heads, n, c, 2 * c), _F32))
    return (kg, w_rows, u_rows, jnp.swapaxes(g_carry, 1, 2), qg, aqk), n


# ---------------------------------------------------------------------------------------------
# the bounded mode's forward per-chunk work on a Mosaic GPU kernel (rwkv7_prep)
# ---------------------------------------------------------------------------------------------


def _uses_prep_kernel(key_dim: int, value_dim: int, precision, bounded: bool) -> bool:
    """The bounded mode with its default products and head dims 64 builds its operands in one kernel."""
    return bounded and precision is None and key_dim == value_dim == rwkv7_prep.DIM


def _prep_flags(doc: Array, n_chunks: int) -> Array:
    """``[B, 3, N C]`` f32 per token: ``carry_ok``, ``same_last`` and the document index (see ``prep``)."""
    batch, seq_len = doc.shape
    d = jnp.pad(doc, ((0, 0), (0, n_chunks * CHUNK - seq_len)), mode="edge").reshape(batch, n_chunks, CHUNK)
    d_in = jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), d[:, :-1, -1]], axis=1)  # doc of the token before
    flags = jnp.stack([d == d_in[:, :, None], d == d[:, :, -1:], d], axis=1).astype(_F32)  # [B, 3, N, C]
    return flags.reshape(batch, 3, n_chunks * CHUNK)


def _prep_operands_impl(q, k, v, w, a, b, doc_f, scale, save=False):
    """:func:`_operands` (bounded mode, ``BF16X3``-class products) with the per-chunk work in one kernel;
    with ``save`` also the kernel's ``(T, A_ak)`` ``[B, H, N, C, C]``."""
    batch, seq_len, heads, key_dim = q.shape
    n = -(-seq_len // CHUNK)
    n2 = n + n % 2  # the kernel takes chunks in pairs; a padded last chunk is computed and dropped
    pad = n2 * CHUNK - seq_len

    def padded(x, dtype):
        return jnp.pad(x.astype(dtype), ((0, 0), (0, pad), (0, 0), (0, 0)))

    g = jnp.cumsum(padded(w, _F32).reshape(batch, n2, CHUNK, heads, key_dim), axis=2)
    flags = _prep_flags(doc_f.astype(jnp.int32), n2)
    tiles = rwkv7_prep.prep(padded(q, _BF16), padded(k, _BF16), padded(v, _BF16), padded(a, _F32),
                            padded(b, _F32), g.reshape(batch, n2 * CHUNK, heads, key_dim), flags, scale=scale,
                            save=save)
    kg, w_rows, u_rows, qg, aqk = (x[:, :, : n * vg.CHUNK] for x in tiles[:5])
    carry_last = flags[:, 0].reshape(batch, n2, CHUNK)[:, :n, -1]  # [B, N]
    g_carry = jnp.where(carry_last[:, :, None, None] > 0, g[:, :n, -1], _NO_CARRY)  # [B, N, H, K]
    operands = (kg, w_rows, u_rows, g_carry, qg, aqk)
    if not save:
        return operands
    return operands, (_folded(*tiles[5:7], n), _folded(*tiles[7:9], n))


def _folded(hi: Array, lo: Array, n: int) -> Array:
    """``[B, H, N, C, C]`` f32 per-chunk matrices from the kernel's split, block-diagonal ``[B, H, T, 2C]``."""
    batch, heads, seq_len, _ = hi.shape
    x = (hi.astype(_F32) + lo.astype(_F32)).reshape(batch, heads, seq_len // (2 * CHUNK), 2, CHUNK, 2, CHUNK)
    blocks = jnp.stack([x[:, :, :, 0, :, 0], x[:, :, :, 1, :, 1]], axis=3)  # [B, H, pairs, 2, C, C]
    return blocks.reshape(batch, heads, seq_len // CHUNK, CHUNK, CHUNK)[:, :, :n]


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _prep_operands(q, k, v, w, a, b, doc_f, scale):
    """The kernel's operands. The VJP is that of the XLA :func:`_operands`, with the kernel's ``T`` and
    ``A_ak`` (saved by the forward) in place of its score product and inverse."""
    return _prep_operands_impl(q, k, v, w, a, b, doc_f, scale)


def _prep_operands_fwd(q, k, v, w, a, b, doc_f, scale):
    operands, saved = _prep_operands_impl(q, k, v, w, a, b, doc_f, scale, save=True)
    return operands, (q, k, v, w, a, b, doc_f, saved)


def _prep_operands_bwd(scale, residuals, cotangents):
    *inputs, doc_f, saved = residuals
    doc = doc_f.astype(jnp.int32)
    _, pullback = jax.vjp(lambda *x: _operands(*x, scale, doc, BOUNDED_PRECISION, True, saved)[0], *inputs)
    return (*pullback(cotangents), jnp.zeros_like(doc_f))


_prep_operands.defvjp(_prep_operands_fwd, _prep_operands_bwd)


# ---------------------------------------------------------------------------------------------
# the bounded mode with head dims 64 on kernels end to end: rwkv7_prep and the state pass forward,
# the reversed state pass and rwkv7_bwd backward
# ---------------------------------------------------------------------------------------------


def _kernel_inputs(q, k, v, w, a, b, doc_f):
    """The kernels' inputs ``(q, k, v, a, b, G, flags)``, padded to whole pairs of chunks (``a, b``
    bf16 when both are bf16, else f32; both kernels read either)."""
    batch, seq_len, heads, key_dim = q.shape
    n = -(-seq_len // CHUNK)
    n2 = n + n % 2  # the per-chunk kernels take chunks in pairs; a padded chunk is inert in the state pass
    pad = n2 * CHUNK - seq_len
    ab = _BF16 if a.dtype == b.dtype == _BF16 else _F32

    def padded(x, dtype):
        return jnp.pad(x.astype(dtype), ((0, 0), (0, pad), (0, 0), (0, 0)))

    flags = _prep_flags(doc_f.astype(jnp.int32), n2)
    # G, the cumulative log decay inside each chunk, and each chunk's carry log decay [B, N, H, K]
    if heads * key_dim % rwkv7_prep.DECAY_COLUMNS == 0:  # one kernel pass, the same bits as XLA's below
        g, g_carry = rwkv7_prep.chunk_decays(padded(w, _BF16 if w.dtype == _BF16 else _F32), flags,
                                             no_carry=_NO_CARRY)
    else:
        g = jnp.cumsum(padded(w, _F32).reshape(batch, n2, CHUNK, heads, key_dim), axis=2)
        carry_ok = flags[:, 0].reshape(batch, n2, CHUNK)[:, :, -1]
        g_carry = jnp.where(carry_ok[:, :, None, None] > 0, g[:, :, -1], _NO_CARRY)
        g = g.reshape(batch, n2 * CHUNK, heads, key_dim)
    return (padded(q, _BF16), padded(k, _BF16), padded(v, _BF16), padded(a, ab), padded(b, ab), g, flags), g_carry


def _kernel_forward(q, k, v, w, a, b, doc_f, h0, scale, save, split=False):
    """``(o, final)`` and, with ``save``, the residuals of :func:`_kernel_path_bwd`. With ``split`` (packed
    documents) the state pass cuts each row's chain at chunks where a document starts (``vg.piece_table``:
    such a chunk's carry decay is zero and its flags zero every term that crosses the start)."""
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    inputs, g_carry = _kernel_inputs(q, k, v, w, a, b, doc_f)  # g_carry [B, N, H, K]
    n2 = inputs[5].shape[1] // CHUNK
    tiles = rwkv7_prep.prep(*inputs, scale=scale, save=save)
    kg, w_rows, u_rows, qg, aqk = tiles[:5]
    cfg = vg.Config(batch=batch, seq_len=n2 * vg.CHUNK, heads=heads, key_dim=key_dim, value_dim=value_dim,
                    scale=1.0, channel_gates=False)
    # a chunk with a document start: the carry_ok flag of its last token is 0 (its carry decay _NO_CARRY)
    kill = (inputs[6][:, 0].reshape(batch, n2, CHUNK)[:, :, -1] == 0) if split else None
    state_cfg = _split_state_config(_forward_state_config(cfg, fused=False), kill)
    out = vg.state_output(state_cfg, kg, w_rows, u_rows, g_carry, h0, qg, aqk, pieces=_pieces(state_cfg, kill),
                          save=save, out_rows=CHUNK)  # each 64-row chunk's first 32 rows are outputs
    o = out[0][:, :seq_len]
    if not save:
        return (o, out[1]), None
    markers = tuple(jnp.zeros((), x.dtype) for x in (w, a, b))  # the gate dtypes of the cotangents
    return (o, out[1]), (inputs, tiles[5:], (kg, w_rows, qg, aqk, g_carry), h0, out[2], out[3], doc_f, markers,
                         kill)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _kernel_path(q, k, v, w, a, b, doc_f, h0, scale):
    """``(o, final)`` of the bounded mode with head dims 64: the forward per-chunk work in
    :func:`rwkv7_prep.prep`, the backward per-chunk work in :func:`rwkv7_bwd.bwd`."""
    return _kernel_forward(q, k, v, w, a, b, doc_f, h0, scale, save=False)[0]


def _kernel_path_fwd(q, k, v, w, a, b, doc_f, h0, scale, split=False):
    return _kernel_forward(q, k, v, w, a, b, doc_f, h0, scale, save=True, split=split)


def _kernel_path_bwd(scale, residuals, cotangents):
    inputs, saved, (kg, w_rows, qg, aqk, g_carry), h0, states, v_new, doc_f, markers, kill = residuals
    do, d_final = cotangents
    batch, seq_len = doc_f.shape
    _, padded_len, heads, key_dim = inputs[0].shape
    value_dim = inputs[2].shape[-1]
    do_p = jnp.pad(do.astype(_BF16), ((0, 0), (0, padded_len - seq_len), (0, 0), (0, 0)))
    cfg = vg.Config(batch=batch, seq_len=2 * padded_len, heads=heads, key_dim=key_dim, value_dim=value_dim,
                    scale=1.0, channel_gates=False)
    # the state pass's 64-row chunks have outputs on their first 32 rows only: bwd_state reads those rows of
    # do (zero below) and forms dv_o = Aqk^T do itself
    d_states, dv_new, dh0 = vgb.bwd_state(cfg, qg, kg, w_rows, g_carry, do_p, None, d_final.astype(_F32), aqk=aqk,
                                          do_rows=CHUNK, pieces=_pieces(cfg, kill))
    # the carry log-decay G_last of each chunk (exp(_NO_CARRY) = 0 after a segment start)
    d_carry = jnp.exp(g_carry) * jnp.einsum("bhnkv,bhnkv->bnhk", states, d_states, preferred_element_type=_F32)
    w_m, a_m, b_m = markers
    # bf16 gate cotangents come out of the kernel already rounded (the same bits as rounding its f32 values here)
    grad_dtype = _BF16 if w_m.dtype == a_m.dtype == b_m.dtype == _BF16 else _F32
    grads = rwkv7_bwd.bwd(*inputs, saved, states, d_states, do_p, v_new, dv_new, d_carry, scale=scale,
                          grad_dtype=grad_dtype)
    dq, dk, dv, da, db, dw = (x[:, :seq_len] for x in grads)
    return (dq, dk, dv, dw.astype(w_m.dtype), da.astype(a_m.dtype), db.astype(b_m.dtype), jnp.zeros_like(doc_f),
            None if h0 is None else dh0)


_kernel_path.defvjp(_kernel_path_fwd, _kernel_path_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _kernel_path_docs(q, k, v, w, a, b, doc_f, h0, scale):
    """:func:`_kernel_path` with packed documents: both state passes split at document starts (the
    same results, more chains in flight when there are few)."""
    return _kernel_forward(q, k, v, w, a, b, doc_f, h0, scale, save=False, split=True)[0]


_kernel_path_docs.defvjp(functools.partial(_kernel_path_fwd, split=True), _kernel_path_bwd)


# ---------------------------------------------------------------------------------------------
# public entry point, predicate, registration
# ---------------------------------------------------------------------------------------------

_OPTIONS = frozenset({"scale", "initial_state", "output_final_state", "cu_seqlens", "segment_ids", "chunk_size",
                      "precision", "lower_bound"})


def unsupported_reason(q: Array, k: Array, v: Array, w: Array, a: Array, b: Array, options: dict) -> str | None:
    """Why the kernels cannot run this call (``None`` if they can).

    They need sm_90, bf16 ``q, k, v`` (``w, a, b`` float32 or bfloat16), ``K, V`` in {64, 128},
    ``T >= 1``, chunk 32, integer ``segment_ids [B, T]`` or none, and no ``cu_seqlens``.
    """
    extra = sorted(set(options) - _OPTIONS)
    if extra:
        return f"unsupported options {extra}"
    if not is_hopper():
        return "the kernels need an sm_90 (Hopper) GPU"
    if options.get("cu_seqlens") is not None:
        return "the kernels take segment_ids, not cu_seqlens"
    if options.get("chunk_size", CHUNK) != CHUNK:
        return f"the kernels use chunk_size={CHUNK}"
    bound = options.get("lower_bound")
    if bound is not None and not MIN_LOWER_BOUND <= bound < 0:
        return f"lower_bound must be in [{MIN_LOWER_BOUND}, 0)"
    precision = options.get("precision")
    if not any(precision is p for p in (None, _HIGHEST, BF16X3)):
        return "precision must be None (default numerics), lax.Precision.HIGHEST or BF16X3"
    if precision is BF16X3 and bound is None:
        return "precision BF16X3 is for the bounded mode (pass lower_bound)"
    if q.ndim != 4 or v.ndim != 4 or v.shape[:3] != q.shape[:3] or any(x.shape != q.shape for x in (k, w, a, b)):
        return "q, k, w, a, b must be [B, T, H, K] and v [B, T, H, V]"
    if any(x.dtype != _BF16 for x in (q, k, v)):
        return "q, k, v must be bfloat16"
    if any(x.dtype not in (_F32, _BF16) for x in (w, a, b)):
        return "w, a, b must be float32 or bfloat16"
    batch, seq_len, heads, key_dim = q.shape
    if seq_len < 1:
        return "the sequence is empty"
    if key_dim not in HEAD_DIMS or v.shape[-1] not in HEAD_DIMS:
        return f"head dims must be in {HEAD_DIMS}"
    segment_ids = options.get("segment_ids")
    if segment_ids is not None and (
        segment_ids.shape != (batch, seq_len) or not jnp.issubdtype(segment_ids.dtype, jnp.integer)
    ):
        return "segment_ids must be an integer [B, T] array"
    state = options.get("initial_state")
    if state is not None and state.shape != (batch, heads, key_dim, v.shape[3]):
        return "initial_state must be [B, H, K, V]"
    return None


def supports_rwkv7(q, k, v, w, a, b, **options) -> bool:
    """True for the calls the kernels are tested on (see :func:`unsupported_reason`)."""
    return unsupported_reason(q, k, v, w, a, b, options) is None


def rwkv7(
    q: Array,
    k: Array,
    v: Array,
    w: Array,
    a: Array,
    b: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = CHUNK,
    precision: lax.Precision | lax.DotAlgorithmPreset | None = None,
    lower_bound: float | None = None,
) -> tuple[Array, Array | None]:
    """RWKV-7 / DPLR with the arguments and conventions of :func:`jla.ops.rwkv7.rwkv7`.

    ``q`` (RWKV's ``r``), ``k`` ``[B, T, H, K]`` and ``v [B, T, H, V]`` bf16 with ``K, V`` in
    {64, 128}; ``w`` (log decay, at most zero), ``a``, ``b`` ``[B, T, H, K]``; any ``T``.
    ``o`` has ``q``'s dtype; the final state is f32 ``[B, H, K, V]``.

    ``lower_bound`` (bounded mode, like FLA's ``safe_gate`` / ``lower_bound``; what the RWKV-7 layers
    pass): the caller guarantees ``lower_bound <= w`` everywhere (``-0.61`` covers RWKV-7's ``w =
    -0.6065 sigmoid(.)``, also after bf16 rounding), in ``[MIN_LOWER_BOUND, 0)``. Each chunk's four
    score matrices then come from one dense product through the chunk's middle token
    (:func:`_bounded_f32_scores`) instead of the exact sub-chunk blocks; the bound keeps every factor
    below ``e^(16 |lower_bound|)``. It is not checked on the values: a much smaller ``w`` overflows and
    gives wrong results. The bounded mode's products (scores, WY inverse, ``W``/``U_wy``) take f32
    operands: ``precision=None`` runs them as :data:`BOUNDED_PRECISION` (:data:`BF16X3`: three bf16
    tensor-core products of split operands, f32 accumulation), ``precision=lax.Precision.HIGHEST`` as
    f32 products. Plain bf16 operands are not offered: they lost ``dq``/``dk`` accuracy under the
    structured cotangent of the group norm after the op (11% norm error in the layer against 0.4%).

    Without ``lower_bound`` (the exact mode, any ``w <= 0``): ``precision=None`` runs the
    off-diagonal sub-chunk score products on bf16 tensor-core operands with f32 accumulation and
    every other XLA product (the exact diagonal blocks, the WY inverse and products) in f32
    (``HIGHEST``); ``precision=lax.Precision.HIGHEST`` runs all of them with f32 operands. Other
    precisions are rejected.
    """
    _check(q, k, v, w, a, b, cu_seqlens, segment_ids)
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size, lower_bound=lower_bound,
                   precision=precision)
    reason = unsupported_reason(q, k, v, w, a, b, options)
    if reason is not None:
        raise ValueError(f"mosaic_gpu rwkv7: {reason}")
    return _run(q, k, v, w, a, b, scale=1.0 if scale is None else scale, initial_state=initial_state,
                output_final_state=output_final_state, segment_ids=segment_ids, precision=precision,
                lower_bound=lower_bound)


# Which backward the bounded mode's head-dim-64 calls (the RWKV-7 layers' call) run by default: the
# rwkv7_bwd kernel (True) or the XLA VJP of the forward kernel's operands (False). Chosen by measured
# speed at the 1.3B layer shape (forward + backward 31.0 ms against 33.4 ms,
# interleaved on one H100); both are tested against each other.
KERNEL_BACKWARD = True


def _run(q, k, v, w, a, b, *, scale: float, initial_state, output_final_state: bool, segment_ids, precision,
         lower_bound, kernel_backward: bool | None = None) -> tuple[Array, Array | None]:
    """:func:`rwkv7` after validation. ``kernel_backward`` picks the backward of the bounded mode's 64-dim
    calls (``None``: :data:`KERNEL_BACKWARD`): :func:`rwkv7_bwd.bwd` after the reversed state pass
    (:func:`_kernel_path`) or the XLA VJP of :func:`_prep_operands`."""
    kernel_backward = KERNEL_BACKWARD if kernel_backward is None else kernel_backward
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    doc = _doc_ids(segment_ids, batch, seq_len)
    prep_kernel = _uses_prep_kernel(key_dim, value_dim, precision, lower_bound is not None)
    if prep_kernel and kernel_backward:
        h0 = None if initial_state is None else initial_state.astype(_F32)
        path = _kernel_path if segment_ids is None else _kernel_path_docs
        o, final = path(q, k, v, w, a, b, doc.astype(_F32), h0, scale)
        return o.astype(q.dtype), (final if output_final_state else None)
    if prep_kernel:
        operands, n = _prep_operands(q, k, v, w, a, b, doc.astype(_F32), scale), -(-seq_len // CHUNK)
    else:
        operands, n = _operands(q, k, v, w, a, b, scale, doc, precision, lower_bound is not None)
    cfg = vg.Config(batch=batch, seq_len=n * vg.CHUNK, heads=heads, key_dim=key_dim, value_dim=value_dim,
                    scale=1.0, channel_gates=False)
    kg, w_rows, u_rows, g_last, qg, aqk = operands
    h0 = None if initial_state is None else initial_state.astype(_F32)
    o, final = _state_pass(cfg, kg, w_rows, u_rows, g_last, h0, qg, aqk)
    o = o.reshape(batch, n, vg.CHUNK, heads, value_dim)[:, :, :CHUNK].reshape(batch, n * CHUNK, heads, value_dim)
    return o[:, :seq_len].astype(q.dtype), (final if output_final_state else None)


register("mosaic_gpu", "rwkv7", rwkv7, supports_rwkv7, auto=True)
