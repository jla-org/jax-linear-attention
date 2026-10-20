# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/generalized_delta_rule/dplr/naive.py,
# fla/ops/generalized_delta_rule/dplr/chunk.py and fla/ops/rwkv7/chunk.py.
"""RWKV-7 (Goose) generalised delta rule: a diagonal-plus-low-rank (DPLR) state transition.

FLA layout and conventions (``chunk_rwkv7(r, w, k, v, a, b)``): ``q`` (RWKV's ``r``), ``k``,
``w``, ``a``, ``b`` are ``[B, T, H, K]``, ``v`` is ``[B, T, H, V]``, ``w`` is the log decay
(``w <= 0``) and the state ``S`` is ``[K, V]`` per head:

    S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T,      o_t = scale q_t^T S_t

which is ``S_t^T = S_{t-1}^T (diag(exp(w_t)) + a_t b_t^T) + v_t k_t^T`` in RWKV's ``[V, K]``
form. The RWKV-7 layer passes ``a = -kk`` and ``b = kk * a_gate`` (``kk`` the L2-normalised
key). ``scale`` defaults to 1 (RWKV-7; FLA's ``chunk_rwkv7`` default).

``segment_ids`` (int ``[B, T]``) restarts the state from zero wherever the id changes;
``initial_state`` applies to each row's first segment and the final state is the state after
each row's last token. ``cu_seqlens`` is not supported: pass ``segment_ids``.
``lower_bound`` (optional, in ``[-2.5, 0)``) promises ``w >= lower_bound`` everywhere; the
Mosaic GPU kernels use it for their bounded mode (:mod:`jla.ops.mosaic_gpu.rwkv7`), the
oracle and the chunked op compute the same function without it (they validate its range only).

* :func:`rwkv7` is the token-by-token ``lax.scan`` oracle;
* :func:`rwkv7_chunked` is FLA's DPLR chunk algorithm in plain JAX: per chunk the strictly
  causal ``A_ab``/``A_ak`` and causal ``A_qk``/``A_qb`` matrices, the WY solve
  ``(I - A_ab)^{-1}`` that gives the low-rank reads ``u_t = S_{t-1}^T a_t`` as
  ``w_wy S_0 + u_wy``, a sequential state pass over chunks and the output. Decays inside a chunk
  are exact pairwise differences of the cumulative log decay, so any ``w <= 0`` is stable
  (memory grows as ``C^2 K`` per chunk; keep ``chunk_size`` small).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["rwkv7", "rwkv7_chunked", "segment_resets"]

Array = jax.Array
_HIGHEST = lax.Precision.HIGHEST


def segment_resets(segment_ids: Array) -> Array:
    """``[B, T]`` bool: True where a token starts a new segment (never at ``t = 0``); also valid for T = 1."""
    change = segment_ids[:, 1:] != segment_ids[:, :-1]
    return jnp.concatenate([jnp.zeros((segment_ids.shape[0], 1), bool), change], axis=1)


def _check(q, k, v, w, a, b, cu_seqlens, segment_ids) -> None:
    if cu_seqlens is not None:
        raise ValueError("rwkv7 does not support cu_seqlens; pass segment_ids for packed documents")
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError(f"rwkv7 expects q [B, T, H, K] and v [B, T, H, V]; got {q.shape}, {v.shape}")
    for name, x in (("k", k), ("w", w), ("a", a), ("b", b)):
        if x.shape != q.shape:
            raise ValueError(f"rwkv7: {name} must have the shape of q {q.shape}; got {x.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"rwkv7: v must be [B, T, H, V] matching q {q.shape}; got {v.shape}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")


def _check_lower_bound(lower_bound: float | None) -> None:
    """``lower_bound`` is a promise about ``w`` (see the module docstring): ``None`` or in ``[-2.5, 0)``."""
    if lower_bound is not None and not -2.5 <= lower_bound < 0:
        raise ValueError(f"rwkv7: lower_bound must be None or in [-2.5, 0); got {lower_bound}")


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
    lower_bound: float | None = None,
) -> tuple[Array, Array | None]:
    """Token-by-token RWKV-7 / DPLR recurrence (the oracle); returns ``(o, final_state)``."""
    _check(q, k, v, w, a, b, cu_seqlens, segment_ids)
    _check_lower_bound(lower_bound)
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    scale = 1.0 if scale is None else scale
    q = q.astype(f32) * scale
    k, v, w, a, b = (x.astype(f32) for x in (k, v, w, a, b))
    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)
    resets = None if segment_ids is None else segment_resets(segment_ids)

    def step(state: Array, token) -> tuple[Array, Array]:
        q_t, k_t, v_t, w_t, a_t, b_t, reset_t = token  # [H, K] / [H, V]; reset_t scalar or None
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
        read = jnp.einsum("hk,hkv->hv", a_t, state, precision=_HIGHEST)  # a_t^T S_{t-1}
        state = (
            state * jnp.exp(w_t)[:, :, None]
            + b_t[:, :, None] * read[:, None, :]
            + k_t[:, :, None] * v_t[:, None, :]
        )
        return state, jnp.einsum("hk,hkv->hv", q_t, state, precision=_HIGHEST)

    def run(q, k, v, w, a, b, resets, state0):
        final, o = lax.scan(step, state0, (q, k, v, w, a, b, resets))
        return o, final

    o, final = jax.vmap(run)(q, k, v, w, a, b, resets, initial_state.astype(f32))
    return o.astype(out_dtype), (final if output_final_state else None)


def rwkv7_chunked(
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
    chunk_size: int = 16,
    precision: lax.Precision | None = None,
    lower_bound: float | None = None,
) -> tuple[Array, Array | None]:
    """Chunk-parallel RWKV-7 / DPLR (FLA's DPLR chunk algorithm); returns ``(o, final_state)``."""
    _check(q, k, v, w, a, b, cu_seqlens, segment_ids)
    _check_lower_bound(lower_bound)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    prec = _HIGHEST if precision is None else precision
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    scale = 1.0 if scale is None else scale
    C = chunk_size
    n_chunks = -(-seq_len // C)
    pad = n_chunks * C - seq_len

    if segment_ids is None:
        doc = jnp.zeros((batch, seq_len), jnp.int32)
    else:
        doc = jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)
    # Padding tokens have k = v = a = b = 0 and w = 0, so they leave the state unchanged.
    def prep(x):
        x = jnp.pad(x.astype(f32), ((0, 0), (0, pad), (0, 0), (0, 0)))
        return x.reshape(batch, n_chunks, C, heads, x.shape[-1]).transpose(0, 3, 1, 2, 4)  # [B, H, N, C, D]

    q_c = prep(q) * scale
    k_c, v_c, w_c, a_c, b_c = (prep(x) for x in (k, v, w, a, b))
    doc = jnp.pad(doc, ((0, 0), (0, pad)), mode="edge").reshape(batch, n_chunks, C)
    doc_in = jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), doc[:, :-1, -1]], axis=1)  # doc of the token before
    carry_ok = (doc == doc_in[:, :, None])[:, None, :, :, None].astype(f32)  # [B, 1, N, C, 1]
    same = doc[:, :, :, None] == doc[:, :, None, :]  # [B, N, C(t), C(i)]
    same_last = (doc == doc[:, :, -1:])[:, None, :, :, None].astype(f32)  # [B, 1, N, C, 1]

    g_inc = jnp.cumsum(w_c, axis=3)  # G_t = sum_{j <= t} w_j inside the chunk
    g_exc = g_inc - w_c  # G_{t-1}
    t_idx = jnp.arange(C)
    strict = (t_idx[:, None] > t_idx[None, :])[None, None, None]  # [1, 1, 1, C, C]
    causal = (t_idx[:, None] >= t_idx[None, :])[None, None, None]
    same_h = same[:, None]  # [B, 1, N, C, C]

    def pair(rows, g_rows, cols, mask):
        # sum_k rows[t, k] cols[i, k] exp(g_rows[t, k] - G_i[k]) on the mask, 0 elsewhere.
        diff = g_rows[..., :, None, :] - g_inc[..., None, :, :]  # [B, H, N, C, C, K]
        m = (mask & same_h)[..., None]
        decay = jnp.exp(jnp.where(m, diff, -jnp.inf))
        return jnp.einsum("...tk,...tik,...ik->...ti", rows, decay, cols, precision=prec)

    A_ab = pair(a_c, g_exc, b_c, strict)
    A_ak = pair(a_c, g_exc, k_c, strict)
    A_qk = pair(q_c, g_inc, k_c, causal)
    A_qb = pair(q_c, g_inc, b_c, causal)

    # WY: u = A_ab u + (a * exp(G_{t-1})) S_0 + A_ak v  =>  u = w_wy S_0 + u_wy.
    lower = jnp.eye(C, dtype=f32) - A_ab
    a_dec = a_c * jnp.exp(g_exc) * carry_ok
    w_wy = lax.linalg.triangular_solve(lower, a_dec, left_side=True, lower=True, unit_diagonal=True)
    u_wy = lax.linalg.triangular_solve(
        lower, jnp.einsum("...ti,...iv->...tv", A_ak, v_c, precision=prec),
        left_side=True, lower=True, unit_diagonal=True,
    )
    q_dec = q_c * jnp.exp(g_inc) * carry_ok
    o_intra = jnp.einsum("...ti,...iv->...tv", A_qk, v_c, precision=prec)
    g_last = g_inc[..., -1:, :]  # [B, H, N, 1, K]
    to_end = jnp.exp(g_last - g_inc) * same_last  # decay from token i to the chunk end, last document only
    k_end = k_c * to_end
    b_end = b_c * to_end
    state_decay = jnp.exp(g_last[..., 0, :]) * carry_ok[..., -1, :]  # [B, H, N, K]

    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)

    def chunk_step(state, xs):
        w_n, u_n, qd_n, oi_n, aqb_n, ke_n, be_n, v_n, sd_n = xs
        u_full = u_n + jnp.einsum("bhtk,bhkv->bhtv", w_n, state, precision=prec)
        o_n = (
            oi_n
            + jnp.einsum("bhti,bhiv->bhtv", aqb_n, u_full, precision=prec)
            + jnp.einsum("bhtk,bhkv->bhtv", qd_n, state, precision=prec)
        )
        state = (
            state * sd_n[..., None]
            + jnp.einsum("bhik,bhiv->bhkv", ke_n, v_n, precision=prec)
            + jnp.einsum("bhik,bhiv->bhkv", be_n, u_full, precision=prec)
        )
        return state, o_n

    def chunks_first(x):
        return jnp.moveaxis(x, 2, 0)

    xs = tuple(
        chunks_first(x) for x in (w_wy, u_wy, q_dec, o_intra, A_qb, k_end, b_end, v_c, state_decay)
    )
    final, o = lax.scan(chunk_step, initial_state.astype(f32), xs)  # o [N, B, H, C, V]
    o = o.transpose(1, 0, 3, 2, 4).reshape(batch, n_chunks * C, heads, value_dim)[:, :seq_len]
    return o.astype(out_dtype), (final if output_final_state else None)
