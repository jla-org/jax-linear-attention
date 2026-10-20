# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode cache of PaTH attention: the key-reflection recurrence of ``path_attention_reference``.

A :class:`PathCache` holds every cached key already reflected by all later tokens
(``k_j <- H_t k_j`` for each new token ``t > j``, with ``H_t = I - beta_t w_t w_t^T``),
so a query reads the cache directly: ``logit[t, j] = scale * (q_t . k~_j + G_t - G_j)``.
Each new token first reflects the cache, then writes its own (unreflected) key, then
attends over the slots whose position is at most its own. Slots are addressed by
position (``slot = position``, as a dense :class:`~jla.layers.cache.KVCache`; positions
``>= max_len`` are dropped). ``G`` is the running sum of the FoX log forget gate
(PaTH-FoX); the short conv on ``w`` carries its input state. Keys, scores and the gate
sums stay in float32, like the reference op. Prefill runs the same recurrence token by
token, so a cached prompt costs ``O(T * max_len * D)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jla.layers.norm import l2_normalize

if TYPE_CHECKING:
    from jla.layers.path_attention import PathAttention

__all__ = ["PathCache", "init_path_cache", "path_decode_step"]

Array = jax.Array


class PathCache(eqx.Module):
    keys: Array  # [B, S, Hkv, D] float32, reflected by every token processed so far
    values: Array  # [B, S, Hkv, Dv]
    positions: Array  # [B, S] int32, -1 = empty
    gate_sums: Array | None  # [B, S, H] float32: G at each cached token (forget gate only)
    gate_last: Array | None  # [B, H] float32: G after the last processed token (forget gate only)
    conv: Array | None  # [B, K - 1, Hkv * D] input state of the w conv


def init_path_cache(layer: PathAttention, batch: int, max_len: int, dtype: jnp.dtype) -> PathCache:
    cfg = layer.cfg
    if max_len < 1:
        raise ValueError(f"cache size must be >= 1; got {max_len}")
    kv_shape = (batch, max_len, cfg.num_kv_heads, cfg.head_dim)
    gate = cfg.forget_gate
    return PathCache(
        keys=jnp.zeros(kv_shape, jnp.float32),
        values=jnp.zeros(kv_shape, dtype),
        positions=jnp.full((batch, max_len), -1, jnp.int32),
        gate_sums=jnp.zeros((batch, max_len, cfg.num_heads), jnp.float32) if gate else None,
        gate_last=jnp.zeros((batch, cfg.num_heads), jnp.float32) if gate else None,
        conv=(jnp.zeros((batch, cfg.w_conv_size - 1, cfg.num_kv_heads * cfg.head_dim), dtype)
              if layer.w_conv is not None else None),
    )


def path_decode_step(layer: PathAttention, x: Array, positions: Array,
                     cache: PathCache) -> tuple[Array, PathCache]:
    """Prefill or decode ``x [B, T, D]`` at ``positions [B, T]``; returns the output and the new cache."""
    cfg = layer.cfg
    batch, length, _ = x.shape
    heads, kv_heads, dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    groups = heads // kv_heads
    f32 = jnp.float32
    q = layer.q_proj(x).reshape(batch, length, heads, dim)
    k = layer.k_proj(x).reshape(batch, length, kv_heads, dim)
    v = layer.v_proj(x).reshape(batch, length, kv_heads, dim)
    w = layer.w_proj(x)
    conv = cache.conv
    if layer.w_conv is not None:
        w, conv = layer.w_conv(w, state=conv)
    w = l2_normalize(w.reshape(batch, length, kv_heads, dim).astype(f32))
    beta = 2.0 * jax.nn.sigmoid(layer.beta_proj(x).astype(f32))
    g = None if layer.g_proj is None else jax.nn.log_sigmoid(layer.g_proj(x).astype(f32))
    if layer.q_norm is not None:
        q, k = layer.q_norm(q), layer.k_norm(k)
    scale = dim ** -0.5 if cfg.softmax_scale is None else float(cfg.softmax_scale)
    size = cache.keys.shape[1]
    rows = jnp.arange(batch)

    def token(carry, xs):
        keys, values, slots_pos, gate_sums, gate_last = carry
        q_t, k_t, v_t, w_t, b_t, g_t, p_t = xs  # [B, H, D], [B, Hkv, D] x3, [B, Hkv], [B, H] | None, [B]
        proj = jnp.einsum("bshd,bhd->bsh", keys, w_t)
        keys = keys - (b_t[:, None, :] * proj)[..., None] * w_t[:, None]  # every cached key: k <- H_t k
        slot = jnp.where((p_t >= 0) & (p_t < size), p_t, size)
        keys = keys.at[rows, slot].set(k_t.astype(f32), mode="drop")
        values = values.at[rows, slot].set(v_t.astype(values.dtype), mode="drop")
        slots_pos = slots_pos.at[rows, slot].set(p_t, mode="drop")
        key_heads = keys if groups == 1 else jnp.repeat(keys, groups, axis=2)  # [B, S, H, D]
        scores = jnp.einsum("bhd,bshd->bhs", q_t.astype(f32), key_heads)
        if g_t is not None:
            gate_last = gate_last + g_t
            gate_sums = gate_sums.at[rows, slot].set(gate_last, mode="drop")
            scores = scores + gate_last[:, :, None] - gate_sums.transpose(0, 2, 1)
        visible = (slots_pos >= 0) & (slots_pos <= p_t[:, None])  # [B, S]
        scores = jnp.where(visible[:, None], scores * scale, -jnp.inf)
        probs = jax.nn.softmax(scores, axis=-1)
        value_heads = values if groups == 1 else jnp.repeat(values, groups, axis=2)
        o_t = jnp.einsum("bhs,bshe->bhe", probs, value_heads.astype(f32))
        return (keys, values, slots_pos, gate_sums, gate_last), o_t

    time_major = lambda a: None if a is None else jnp.moveaxis(a, 1, 0)  # noqa: E731
    xs = tuple(time_major(a) for a in (q, k, v, w, beta, g, positions.astype(jnp.int32)))
    carry = (cache.keys, cache.values, cache.positions, cache.gate_sums, cache.gate_last)
    (keys, values, slots_pos, gate_sums, gate_last), o = lax.scan(token, carry, xs)
    o = jnp.moveaxis(o, 0, 1).astype(q.dtype).reshape(batch, length, heads * dim)
    new_cache = PathCache(keys=keys, values=values, positions=slots_pos, gate_sums=gate_sums,
                          gate_last=gate_last, conv=conv)
    return layer.o_proj(o), new_cache
