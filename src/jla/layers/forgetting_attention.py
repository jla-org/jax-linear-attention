# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/forgetting_attn.py; the attention op, the KV
# shift and the output norm are written from the FoX paper's equations (arXiv 2503.02130).
"""Forgetting Transformer (FoX) attention: causal softmax attention with a forget gate.

Every head has a data-dependent forget gate ``log f_t = logsigmoid(x_t W_f + b_f)``
(float32, ``<= 0``). Query ``i`` scores key ``j <= i`` with

    logits_ij = scale * q_i . k_j + sum_{l = j + 1}^{i} log f_l = scale * q_i . k_j + c_i - c_j,

where ``c = cumsum(log f)`` along the sequence, so older keys fade at a learned,
input-dependent rate and no positional encoding is needed. The "Pro" block options
of the paper are all switches of :class:`jla.models.config.ForgettingAttentionConfig`:
QK-norm (per-head RMSNorm), an output gate ``o * sigmoid(x W_g)``, an output norm
(per-head RMSNorm of the attention output) and the KV shift (data-dependent token
shift ``k_t <- a_t k_{t-1} + (1 - a_t) k_t`` with ``a_t = sigmoid(x_t W_a)`` per KV
head, likewise for v, applied to the projections before QK-norm).

The training path is plain XLA: one ``[B, H, T, T]`` float32 score matrix with the
``c_i - c_j`` bias, the causal mask, packed-document (``segment_ids``) masks and an
optional sliding window. Decode keeps a dense position-addressed KV cache plus the
cumulative log forget ``c`` of every cached token and the previous token's k and v
projections (for the KV shift).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import KVCache
from jla.layers.linear import Linear
from jla.layers.norm import RMSNorm
from jla.layers.rope import apply_rope

if TYPE_CHECKING:  # layers never import jla.models at runtime (it imports layers)
    from jla.models.config import ForgettingAttentionConfig

__all__ = ["ForgettingAttention", "ForgettingCache", "forgetting_attention"]

Array = jax.Array

_NEG = float(jnp.finfo(jnp.float32).min)


def _scores_and_values(q: Array, k: Array, v: Array, cum_q: Array, cum_k: Array, mask: Array,
                       scale: float) -> Array:
    """Masked softmax of ``scale * q.k + cum_q - cum_k`` applied to ``v``.

    ``q [B, Tq, H, D]``, ``k, v [B, S, Hkv, D]``, ``cum_q [B, Tq, H]``, ``cum_k [B, S, H]``
    (float32) and ``mask [B, Tq, S]`` (True = visible; every query must see a key).
    """
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    groups = heads // kv_heads
    qg = q.reshape(batch, length, kv_heads, groups, head_dim)
    logits = jnp.einsum("bihgd,bjhd->bhgij", qg, k, preferred_element_type=jnp.float32) * scale
    cq = cum_q.astype(jnp.float32).reshape(batch, length, kv_heads, groups).transpose(0, 2, 3, 1)
    ck = cum_k.astype(jnp.float32).reshape(batch, k.shape[1], kv_heads, groups).transpose(0, 2, 3, 1)
    logits = logits + (cq[..., :, None] - ck[..., None, :])
    logits = jnp.where(mask[:, None, None], logits, _NEG)
    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhgij,bjhd->bihgd", probs.astype(v.dtype), v, preferred_element_type=jnp.float32)
    return out.reshape(batch, length, heads, v.shape[-1])


def forgetting_attention(
    q: Array,
    k: Array,
    v: Array,
    log_f: Array,
    *,
    segment_ids: Array | None = None,
    sliding_window: int | None = None,
    scale: float | None = None,
) -> Array:
    """Causal FoX attention over a full sequence (XLA path).

    ``q [B, T, H, D]``, ``k, v [B, T, Hkv, D]`` (``H`` a multiple of ``Hkv``) and the log
    forget gates ``log_f [B, T, H]`` (``<= 0``, one per query head). ``segment_ids [B, T]``
    keeps every query inside its own document; ``sliding_window`` counts the visible
    tokens including the query itself. Returns ``[B, T, H, D]`` in ``q.dtype``.
    """
    batch, length, heads, head_dim = q.shape
    if k.shape[:2] != (batch, length) or v.shape[:3] != k.shape[:3] or k.shape[-1] != head_dim:
        raise ValueError(f"k, v must be [B, T, Hkv, D] matching q {q.shape}; got {k.shape}, {v.shape}")
    if heads % k.shape[2]:
        raise ValueError(f"query heads {heads} must be a multiple of KV heads {k.shape[2]}")
    if log_f.shape != (batch, length, heads):
        raise ValueError(f"log_f must be [B, T, H] = {(batch, length, heads)}; got {log_f.shape}")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding_window must be >= 1")
    scale = head_dim ** -0.5 if scale is None else float(scale)
    i = jnp.arange(length)[:, None]
    j = jnp.arange(length)[None, :]
    mask = j <= i
    if sliding_window is not None:
        mask = mask & (i - j < sliding_window)
    mask = jnp.broadcast_to(mask, (batch, length, length))
    if segment_ids is not None:
        if segment_ids.shape != (batch, length):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
        mask = mask & (segment_ids[:, :, None] == segment_ids[:, None, :])
    # Within one document c_i - c_j only sums gates of that document, so one global cumsum suffices.
    cum = jnp.cumsum(log_f.astype(jnp.float32), axis=1)
    return _scores_and_values(q, k, v, cum, cum, mask, scale).astype(q.dtype)


class ForgettingCache(eqx.Module):
    """Decode state of a FoX layer.

    ``kv`` is a dense KV cache (slot = position), ``log_cum [B, S, H]`` the cumulative log
    forget ``c`` of each cached token, ``last_cum [B, H]`` the ``c`` of the latest token
    (0 before the first) and ``shift`` the previous token's k and v projections
    ``[B, Hkv * D]`` (zeros before the first token; ``None`` without the KV shift).
    """

    kv: KVCache
    log_cum: Array
    last_cum: Array
    shift: tuple[Array, Array] | None


class ForgettingAttention(eqx.Module):
    """FoX block mixer: q/k/v/o projections, a per-head forget gate and the Pro options."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    f_proj: Linear
    o_proj: Linear
    gate_proj: Linear | None
    k_shift_proj: Linear | None
    v_shift_proj: Linear | None
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None
    o_norm: RMSNorm | None
    cfg: ForgettingAttentionConfig = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        cfg: ForgettingAttentionConfig,
        *,
        key: Array,
        std: float,
        norm_eps: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        kq, kk, kv, kf, ko, kg, ka, kb = jax.random.split(key, 8)
        heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.q_proj = Linear(d_model, heads * head_dim, key=kq, axes=("embed", "heads"), **common)
        self.k_proj = Linear(d_model, kv_heads * head_dim, key=kk, axes=("embed", "kv_heads"), **common)
        self.v_proj = Linear(d_model, kv_heads * head_dim, key=kv, axes=("embed", "kv_heads"), **common)
        self.f_proj = Linear(d_model, heads, key=kf, axes=("embed", None), use_bias=True, **common)
        self.o_proj = Linear(heads * head_dim, d_model, key=ko, axes=("heads", "embed"), **common)
        self.gate_proj = (
            Linear(d_model, heads * head_dim, key=kg, axes=("embed", "heads"), **common)
            if cfg.output_gate else None
        )
        if cfg.kv_shift:
            self.k_shift_proj = Linear(d_model, kv_heads, key=ka, axes=("embed", None), **common)
            self.v_shift_proj = Linear(d_model, kv_heads, key=kb, axes=("embed", None), **common)
        else:
            self.k_shift_proj = self.v_shift_proj = None
        if cfg.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
        else:
            self.q_norm = self.k_norm = None
        self.o_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype) if cfg.output_norm else None
        self.cfg = cfg

    def _shift(self, y: Array, alpha_proj: Linear, x: Array, previous: Array | None,
               resets: Array | None) -> Array:
        """``a_t * y_{t-1} + (1 - a_t) * y_t`` per KV head (``y_{-1}`` = ``previous`` or zeros)."""
        batch, length, width = y.shape
        heads = self.cfg.num_kv_heads
        first = jnp.zeros((batch, 1, width), y.dtype) if previous is None else previous[:, None].astype(y.dtype)
        prev = jnp.concatenate([first, y[:, :-1]], axis=1)
        if resets is not None:  # a document's first token has no predecessor
            prev = jnp.where(resets[..., None], jnp.zeros((), y.dtype), prev)
        alpha = jax.nn.sigmoid(alpha_proj(x).astype(jnp.float32))[..., None]  # [B, T, Hkv, 1]
        y_h = y.astype(jnp.float32).reshape(batch, length, heads, -1)
        prev_h = prev.astype(jnp.float32).reshape(batch, length, heads, -1)
        return (alpha * prev_h + (1.0 - alpha) * y_h).reshape(batch, length, width).astype(y.dtype)

    def _project(self, x: Array, positions: Array, segment_ids: Array | None,
                 shift: tuple[Array, Array] | None):
        batch, length, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x)
        k_raw, v_raw = self.k_proj(x), self.v_proj(x)
        k, v = k_raw, v_raw
        if cfg.kv_shift:
            resets = None
            if segment_ids is not None:
                resets = jnp.concatenate(
                    [jnp.zeros((batch, 1), bool), segment_ids[:, 1:] != segment_ids[:, :-1]], axis=1)
            prev_k, prev_v = (None, None) if shift is None else shift
            k = self._shift(k_raw, self.k_shift_proj, x, prev_k, resets)
            v = self._shift(v_raw, self.v_shift_proj, x, prev_v, resets)
        q = q.reshape(batch, length, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        v = v.reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if cfg.use_rope:
            q = apply_rope(q, positions, theta=cfg.rope_theta)
            k = apply_rope(k, positions, theta=cfg.rope_theta)
        log_f = jax.nn.log_sigmoid(self.f_proj(x).astype(jnp.float32))  # [B, T, H]
        new_shift = (k_raw[:, -1], v_raw[:, -1]) if cfg.kv_shift else None
        return q, k, v, log_f, new_shift

    def _output(self, x: Array, o: Array) -> Array:
        batch, length = x.shape[:2]
        if self.o_norm is not None:
            o = self.o_norm(o)
        o = checkpoint_name(o.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        if self.gate_proj is not None:
            gate = jax.nn.sigmoid(self.gate_proj(x).astype(jnp.float32))
            o = (o.astype(jnp.float32) * gate).astype(o.dtype)
        return self.o_proj(o)

    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        q, k, v, log_f, _ = self._project(x, positions, segment_ids, None)
        o = forgetting_attention(q, k, v, log_f, segment_ids=segment_ids,
                                 sliding_window=self.cfg.sliding_window, scale=self.cfg.softmax_scale)
        return self._output(x, o)

    def init_cache(self, batch: int, max_len: int, dtype: jnp.dtype) -> ForgettingCache:
        cfg = self.cfg
        width = cfg.num_kv_heads * cfg.head_dim
        shift = (jnp.zeros((batch, width), dtype), jnp.zeros((batch, width), dtype)) if cfg.kv_shift else None
        return ForgettingCache(
            kv=KVCache.create(batch, max_len, cfg.num_kv_heads, cfg.head_dim, dtype, ring=False),
            log_cum=jnp.zeros((batch, max_len, cfg.num_heads), jnp.float32),
            last_cum=jnp.zeros((batch, cfg.num_heads), jnp.float32),
            shift=shift,
        )

    def step(self, x: Array, positions: Array, cache: ForgettingCache) -> tuple[Array, ForgettingCache]:
        """Prefill or decode ``x [B, T, D]`` at consecutive ``positions [B, T]`` against ``cache``.

        Every row must continue right after its previous call (the forget-gate sum and
        the KV shift run through the block in order); positions past the cache size
        are dropped like :class:`KVCache` does.
        """
        cfg = self.cfg
        q, k, v, log_f, shift = self._project(x, positions, None, cache.shift)
        positions = positions.astype(jnp.int32)
        cum = cache.last_cum[:, None, :] + jnp.cumsum(log_f, axis=1)  # [B, T, H]
        kv = cache.kv.write(k, v, positions)
        size = kv.size
        rows = jnp.arange(positions.shape[0], dtype=jnp.int32)[:, None]
        slots = jnp.where((positions >= 0) & (positions < size), positions, size)
        log_cum = cache.log_cum.at[rows, slots].set(cum, mode="drop")
        kv_pos = kv.positions[:, None, :]
        q_pos = positions[:, :, None]
        mask = (kv_pos >= 0) & (kv_pos <= q_pos)
        if cfg.sliding_window is not None:
            mask = mask & (q_pos - kv_pos < cfg.sliding_window)
        scale = cfg.head_dim ** -0.5 if cfg.softmax_scale is None else float(cfg.softmax_scale)
        o = _scores_and_values(q.astype(kv.keys.dtype), kv.keys, kv.values, cum, log_cum, mask, scale)
        new_shift = None if shift is None else tuple(s.astype(c.dtype) for s, c in zip(shift, cache.shift))
        cache = ForgettingCache(kv=kv, log_cum=log_cum, last_cum=cum[:, -1], shift=new_shift)
        return self._output(x, o.astype(q.dtype)), cache
