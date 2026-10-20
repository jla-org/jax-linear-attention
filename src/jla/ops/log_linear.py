# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/log_linear_attn/naive.py (Fenwick level
# structure, level_scales [B, T, H, L] layout) and Guo et al., "Log-Linear Attention",
# arXiv 2506.04761.
"""Log-linear attention: Fenwick-tree hierarchical states with per-level weights.

Positions ``p`` count from the start of each segment. For a query at ``p_t`` the
prefix ``[0, p_t]`` splits into Fenwick buckets; key ``s <= t`` sits in level

    level(t, s) = 0                       if s == t
                = 1 + msb(p_t XOR p_s)    otherwise

so level 0 is the token itself and level ``l >= 1`` holds the ``2^(l-1)`` keys
whose position first differs from ``p_t`` at bit ``l - 1`` (empty unless that bit
of ``p_t`` is set). With ``level_scales`` ``lambda [B, T, H, L]`` and
``L >= num_levels(T)``, log-linear Mamba-2 (kind ``log_linear_simple_gla``) is

    o_t = scale * sum_{s <= t} lambda_t[level(t, s)] exp(G_t - G_s) (q_t . k_s) v_s

with ``G`` the inclusive cumulative sum of ``g`` inside the segment. With every
``lambda_t[l] = 1`` this is exactly ``simple_gla``, and the levels of the final
state sum to the ``simple_gla`` final state.

``log_linear_simple_gla_reference`` runs the Fenwick recurrence token by token:
at position ``p >= 1`` with ``z`` trailing zero bits, levels ``0..z`` merge into
level ``z + 1``, every level decays by ``exp(g)`` and level 0 becomes ``k^T v``.

``log_linear_simple_gla_chunked`` is chunk-parallel: exact masked attention
inside each chunk plus the Fenwick state carried across chunk boundaries. A query
at ``p_t`` reads boundary level ``l`` with weight ``lambda_t[max(l, level(t, t0))]``
(``t0`` the last token of the previous chunk), and the boundary state maps level
``l`` to ``max(l, level(t1, t0))`` and adds the chunk's keys at their own levels.
Chunks therefore need not align with segment starts.

The state is the Fenwick state ``[B, H, L, K, V]`` (float32) after each row's last
token. ``initial_state`` and ``cu_seqlens`` are not supported (use ``segment_ids``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["log_linear_simple_gla_chunked", "log_linear_simple_gla_reference", "num_levels"]

Array = jax.Array
Precision = lax.Precision

_HIGHEST = lax.Precision.HIGHEST


def num_levels(seq_len: int) -> int:
    """Fenwick levels ``T`` tokens need: ``ceil(log2(T)) + 1``."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1; got {seq_len}")
    return (seq_len - 1).bit_length() + 1


def _level(p_query: Array, p_key: Array) -> Array:
    """Fenwick level of key position ``p_key`` for query position ``p_query`` (int32)."""
    return 32 - lax.clz(jnp.bitwise_xor(p_query, p_key).astype(jnp.int32))


def _positions(segment_ids: Array | None, batch: int, seq_len: int) -> tuple[Array, Array]:
    """Position inside the segment and run index, both int32 ``[B, T]``."""
    t = jnp.arange(seq_len, dtype=jnp.int32)
    if segment_ids is None:
        resets = jnp.zeros((batch, seq_len), dtype=bool)
    else:
        change = segment_ids[:, 1:] != segment_ids[:, :-1]
        resets = jnp.concatenate([jnp.zeros_like(change[:, :1]), change], axis=1)
    start = lax.cummax(jnp.where(resets, t[None, :], 0), axis=1)
    runs = jnp.cumsum(resets.astype(jnp.int32), axis=1)
    return (t[None, :] - start).astype(jnp.int32), runs


def _check(q, k, v, g, level_scales, initial_state, cu_seqlens, segment_ids) -> None:
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(f"q and k must be [B, T, H, K] with equal shapes; got {q.shape}, {k.shape}")
    batch, seq_len, heads, _ = q.shape
    if v.ndim != 4 or v.shape[:3] != (batch, seq_len, heads):
        raise ValueError(f"v must be [B, T, H, V] = {(batch, seq_len, heads)} + (V,); got {v.shape}")
    if g.shape != (batch, seq_len, heads):
        raise ValueError(f"g must be [B, T, H] = {(batch, seq_len, heads)}; got {g.shape}")
    if level_scales.ndim != 4 or level_scales.shape[:3] != (batch, seq_len, heads):
        raise ValueError(f"level_scales must be [B, T, H, L]; got {level_scales.shape}")
    if level_scales.shape[3] < num_levels(seq_len):
        raise ValueError(
            f"level_scales has {level_scales.shape[3]} levels; T = {seq_len} needs {num_levels(seq_len)}"
        )
    if initial_state is not None:
        raise ValueError("log-linear attention does not support initial_state")
    if cu_seqlens is not None:
        raise ValueError("log-linear attention takes segment_ids, not cu_seqlens")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")


def log_linear_simple_gla_reference(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    level_scales: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
) -> tuple[Array, Array | None]:
    """Token-by-token Fenwick recurrence (float32, HIGHEST precision); the oracle."""
    _check(q, k, v, g, level_scales, initial_state, cu_seqlens, segment_ids)
    batch, seq_len, heads, dk = q.shape
    dv = v.shape[-1]
    n_levels = level_scales.shape[-1]
    scale = dk**-0.5 if scale is None else scale
    f32 = jnp.float32
    pos, _ = _positions(segment_ids, batch, seq_len)
    levels = jnp.arange(n_levels, dtype=jnp.int32)

    def step(state, inputs):
        q_t, k_t, v_t, g_t, lam_t, p_t = inputs
        low = jnp.bitwise_and(p_t, -p_t)
        target = (lax.population_count(low - 1) + 1)[:, None]  # z + 1, [B, 1]
        merge = (levels[None, :] < target).astype(f32)
        merged = jnp.einsum("bl,bhlkv->bhkv", merge, state, precision=_HIGHEST)
        keep = (levels[None, :] > target)[:, None, :, None, None]
        put = (levels[None, :] == target)[:, None, :, None, None]
        state = jnp.where(keep, state, 0.0) + jnp.where(put, merged[:, :, None], 0.0)
        state = state * jnp.exp(g_t)[:, :, None, None, None]
        state = jnp.where((p_t > 0)[:, None, None, None, None], state, 0.0)
        state = state.at[:, :, 0].set(jnp.einsum("bhk,bhv->bhkv", k_t, v_t, precision=_HIGHEST))
        o_t = jnp.einsum("bhl,bhk,bhlkv->bhv", lam_t, q_t, state, precision=_HIGHEST)
        return state, o_t

    xs = tuple(
        jnp.moveaxis(x, 1, 0)
        for x in (q.astype(f32) * scale, k.astype(f32), v.astype(f32), g.astype(f32), level_scales.astype(f32), pos)
    )
    state0 = jnp.zeros((batch, heads, n_levels, dk, dv), f32)
    final, o = lax.scan(step, state0, xs)
    return jnp.moveaxis(o, 0, 1).astype(q.dtype), final if output_final_state else None


def log_linear_simple_gla_chunked(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    level_scales: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = 64,
    precision: Precision = _HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunk-parallel log-linear attention: masked intra-chunk attention + boundary Fenwick states."""
    _check(q, k, v, g, level_scales, initial_state, cu_seqlens, segment_ids)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    batch, seq_len, heads, dk = q.shape
    dv = v.shape[-1]
    n_levels = level_scales.shape[-1]
    scale = dk**-0.5 if scale is None else scale
    f32 = jnp.float32
    size = chunk_size
    n_chunks = -(-seq_len // size)
    pad = n_chunks * size - seq_len
    pos, runs = _positions(segment_ids, batch, seq_len)
    # Padded tokens form their own run after the last one, so they never touch real tokens.
    runs = jnp.concatenate([runs, jnp.broadcast_to(runs[:, -1:] + 1, (batch, pad))], axis=1)
    pos = jnp.pad(pos, ((0, 0), (0, pad)))

    def chunks(x):
        x = jnp.pad(x, ((0, 0), (0, n_chunks * size - x.shape[1])) + ((0, 0),) * (x.ndim - 2))
        return jnp.moveaxis(x.reshape((batch, n_chunks, size) + x.shape[2:]), 1, 0)

    xs = (
        chunks(q.astype(f32) * scale), chunks(k.astype(f32)), chunks(v.astype(f32)), chunks(g.astype(f32)),
        chunks(level_scales.astype(f32)), chunks(pos), chunks(runs),
        jnp.minimum(size - 1, seq_len - 1 - size * jnp.arange(n_chunks, dtype=jnp.int32)),
    )
    levels = jnp.arange(n_levels, dtype=jnp.int32)
    s_idx = jnp.arange(size, dtype=jnp.int32)

    def step(carry, inputs):
        state, p0, r0 = carry
        q_c, k_c, v_c, g_c, lam_c, pos_c, run_c, end = inputs
        cum = jnp.cumsum(g_c, axis=1)  # [B, C, H] log decay since the previous chunk's last token

        # Intra-chunk: exact masked attention with the per-pair level weight.
        mask = (s_idx[None, None, :] <= s_idx[None, :, None]) & (run_c[:, :, None] == run_c[:, None, :])
        onehot = jax.nn.one_hot(_level(pos_c[:, :, None], pos_c[:, None, :]), n_levels, dtype=f32)
        weight = jnp.einsum("btsl,bthl->bhts", onehot, lam_c, precision=precision)
        diff = jnp.moveaxis(cum[:, :, None, :] - cum[:, None, :, :], 3, 1)  # [B, H, t, s]
        decay = jnp.exp(jnp.where(mask[:, None], diff, -jnp.inf))
        scores = jnp.einsum("bthk,bshk->bhts", q_c, k_c, precision=precision) * weight * decay
        o = jnp.einsum("bhts,bshv->bthv", scores, v_c, precision=precision)

        # Inter-chunk: boundary level l is read with weight lambda_t[max(l, level(t, t0))].
        same0 = run_c == r0[:, None]  # [B, C]
        idx = jnp.maximum(levels[None, None, :], _level(pos_c, p0[:, None])[:, :, None])
        idx = jnp.broadcast_to(jnp.minimum(idx, n_levels - 1)[:, :, None, :], lam_c.shape)
        w_inter = jnp.take_along_axis(lam_c, idx, axis=3)  # [B, C, H, L]
        decay0 = jnp.exp(jnp.where(same0[:, :, None], cum, -jnp.inf))  # [B, C, H]
        q_state = jnp.einsum("bthk,bhlkv->bthlv", q_c, state, precision=precision)
        o = o + jnp.einsum("bthl,bthlv->bthv", w_inter * decay0[..., None], q_state, precision=precision)

        # Boundary state after token ``end`` of this chunk.
        p1 = jnp.take(pos_c, end, axis=1)
        r1 = jnp.take(run_c, end, axis=1)
        g1 = jnp.take(cum, end, axis=1)  # [B, H]
        lev1 = _level(p1, p0)[:, None]  # [B, 1]
        merged = jnp.einsum("bl,bhlkv->bhkv", (levels[None, :] <= lev1).astype(f32), state, precision=precision)
        keep = (levels[None, :] > lev1)[:, None, :, None, None]
        put = (levels[None, :] == lev1)[:, None, :, None, None]
        moved = jnp.where(keep, state, 0.0) + jnp.where(put, merged[:, :, None], 0.0)
        moved = jnp.where((r1 == r0)[:, None, None, None, None], moved * jnp.exp(g1)[:, :, None, None, None], 0.0)
        valid = (s_idx[None, :] <= end) & (run_c == r1[:, None])  # [B, C]
        own = jax.nn.one_hot(_level(p1[:, None], pos_c), n_levels, dtype=f32) * valid[..., None]
        dec_s = jnp.exp(jnp.where(valid[..., None], g1[:, None, :] - cum, -jnp.inf))  # [B, C, H]
        update = jnp.einsum("bsl,bsh,bshk,bshv->bhlkv", own, dec_s, k_c, v_c, precision=precision)
        return (moved + update, p1, r1), o

    carry0 = (
        jnp.zeros((batch, heads, n_levels, dk, dv), f32),
        jnp.zeros((batch,), jnp.int32),
        jnp.full((batch,), -1, jnp.int32),
    )
    (final, _, _), o = lax.scan(step, carry0, xs)
    o = jnp.moveaxis(o, 0, 1).reshape(batch, n_chunks * size, heads, dv)[:, :seq_len]
    return o.astype(q.dtype), final if output_final_state else None
