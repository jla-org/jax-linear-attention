# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/rwkv6/{fused_recurrent,chunk}.py
"""RWKV-6 (Finch) linear attention: GLA with a bonus ``u`` for the current token.

    o_t = scale * q_t (S_{t-1} + diag(u) k_t^T v_t)
    S_t = diag(exp(g_t)) S_{t-1} + k_t^T v_t

``q`` (RWKV's receptance ``r``) and ``k`` are ``[B, T, H, K]``, ``v`` is
``[B, T, H, V]``, ``g`` the per-channel log-decay ``[B, T, H, K]`` (RWKV-6's
``w = -exp(.)``, so ``g <= 0``) and ``u`` the bonus ``[H, K]``. Unlike GLA the
output reads the state before the token's own decay, and the token itself
enters the output through ``u`` instead of through the state.

``reference`` is the token-by-token oracle. ``chunked`` reuses the GLA stages
of ``jla.ops.chunked``: the in-chunk cumulative decay ``G``, the serial state
pass (the state recurrence is GLA's) and the stable sub-chunked decayed scores.
The query side reads the exclusive cumulative decay ``G_{i-1}``: the strictly
causal scores ``sum_c q_ic k_jc exp(G_{i-1,c} - G_jc)`` are the inclusive GLA
scores of the next row's query shifted down one row, so every exponential stays
at most one. The diagonal is the bonus ``(q_i . (u * k_i)) v_i``.

``segment_ids [B, T]`` restart the state wherever the id changes, as in
``jla.ops.reference``. ``cu_seqlens`` packing is not supported: pass
``segment_ids``. Every function returns ``(o, final_state)`` with ``o`` in the
dtype of ``q`` and ``final_state`` float32 (or ``None`` unless
``output_final_state``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops.chunked import (
    DEFAULT_CHUNK,
    DEFAULT_SUB_CHUNK,
    Precision,
    _from_chunks,
    _to_chunks,
    chunk_local_cumsum,
    chunk_state_pass,
    decayed_scores,
    segment_masks,
)
from jla.ops.reference import segment_resets

__all__ = ["chunked", "reference"]

Array = jax.Array

_HIGHEST = lax.Precision.HIGHEST


def _check(q: Array, k: Array, v: Array, g: Array, u: Array, cu_seqlens, segment_ids) -> None:
    if cu_seqlens is not None:
        raise NotImplementedError("rwkv6 does not support cu_seqlens; pass segment_ids")
    batch, seq_len, heads, key_dim = q.shape
    if k.shape != q.shape or g.shape != q.shape:
        raise ValueError(f"rwkv6 needs k and g shaped like q {q.shape}; got k {k.shape}, g {g.shape}")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"rwkv6 needs v [B, T, H, V] matching q {q.shape}; got {v.shape}")
    if u.shape != (heads, key_dim):
        raise ValueError(f"rwkv6 bonus u must be [H, K] = {(heads, key_dim)}; got {u.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")


def reference(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    u: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
) -> tuple[Array, Array | None]:
    """Token-by-token RWKV-6 recurrence in float32 (the oracle)."""
    _check(q, k, v, g, u, cu_seqlens, segment_ids)
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if scale is None:
        scale = key_dim**-0.5
    q = q.astype(f32) * scale
    k, v, g, u = (x.astype(f32) for x in (k, v, g, u))
    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)
    resets = None if segment_ids is None else segment_resets(segment_ids)

    def step(state: Array, token) -> tuple[Array, Array]:
        q_t, k_t, v_t, g_t, reset_t = token  # [H, K], [H, K], [H, V], [H, K], scalar bool
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
        kv = k_t[:, :, None] * v_t[:, None, :]
        o_t = jnp.einsum("hk,hkv->hv", q_t, state + u[:, :, None] * kv, precision=_HIGHEST)
        return state * jnp.exp(g_t)[:, :, None] + kv, o_t

    def run(q, k, v, g, resets, state0):
        final, o = lax.scan(step, state0, (q, k, v, g, resets))
        return o, final

    o, final = jax.vmap(run)(q, k, v, g, resets, initial_state.astype(f32))
    return o.astype(out_dtype), (final if output_final_state else None)


def chunked(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    u: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = DEFAULT_CHUNK,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunk-parallel RWKV-6 built from the GLA stages of ``jla.ops.chunked``."""
    _check(q, k, v, g, u, cu_seqlens, segment_ids)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if scale is None:
        scale = key_dim**-0.5
    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)

    # Zero padding is inert: k = 0 writes nothing and g = 0 does not decay.
    pad = (-seq_len) % chunk_size

    def prep(x):
        x = x.astype(f32)
        return _to_chunks(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)), chunk_size)

    masks = None if segment_ids is None else segment_masks(segment_ids, chunk_size, pad)
    q_c, k_c, v_c = prep(q), prep(k), prep(v)  # [B, NC, H, C, K or V]
    g_cum = chunk_local_cumsum(prep(g))  # inclusive G [B, NC, H, C, K]
    h, _, final = chunk_state_pass(k_c, None, v_c, g_cum, initial_state.astype(f32), masks=masks,
                                   precision=precision)
    # Inter-chunk: the incoming state decayed up to (not including) the token: exp(G_{i-1}).
    g_excl = jnp.concatenate([jnp.zeros_like(g_cum[..., :1, :]), g_cum[..., :-1, :]], axis=-2)
    q_inter = q_c * jnp.exp(g_excl)
    if masks is not None:
        q_inter = q_inter * masks.carry_in[..., None]
    inter = jnp.einsum("...ck,...kv->...cv", q_inter, h, precision=precision)
    # Intra-chunk, strictly causal: row i-1 of the inclusive scores of the shifted query.
    q_next = jnp.concatenate([q_c[..., 1:, :], jnp.zeros_like(q_c[..., :1, :])], axis=-2)
    scores = decayed_scores(q_next, k_c, g_cum, inclusive=True, sub_chunk=sub_chunk, precision=precision)
    scores = jnp.concatenate([jnp.zeros_like(scores[..., :1, :]), scores[..., :-1, :]], axis=-2)
    if masks is not None:
        scores = scores * masks.same
    intra = jnp.einsum("...ij,...jv->...iv", scores, v_c, precision=precision)
    # The current token's bonus.
    bonus = jnp.sum(q_c * u.astype(f32)[:, None, :] * k_c, axis=-1, keepdims=True) * v_c
    o = scale * (inter + intra + bonus)
    o = _from_chunks(o)[:, :seq_len].astype(out_dtype)
    return o, (final if output_final_state else None)
