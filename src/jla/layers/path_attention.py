# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/path_attn.py
"""PaTH attention token mixer (FLA's ``PaTHAttention`` layout, training and full-sequence calls).

q, k, v and o projections as in softmax attention (no RoPE: the Householder
reflections are the position encoding), plus per-token reflection directions
``w`` (a rank-``w_low_rank_dim`` projection, or a full one when it is ``None``,
then an optional depthwise causal conv with SiLU, then per-head L2 norm), write
strengths ``beta = 2 sigmoid(x W_beta + b)`` in ``(0, 2)`` per key head, and an
optional FoX forget gate ``g = logsigmoid(x W_g + b)`` per query head.
Cached decoding (``init_cache``, ``step``) runs the key-reflection recurrence in
``jla.layers.path_decode``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.layers.conv import ShortConv
from jla.layers.linear import Linear, LowRankLinear
from jla.layers.norm import RMSNorm, l2_normalize
from jla.ops.path_attention import path_attention

if TYPE_CHECKING:  # layers never import jla.models at runtime (it imports layers)
    from jla.models.config import PathAttentionConfig

__all__ = ["PathAttention"]

Array = jax.Array


class PathAttention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    w_proj: Linear | LowRankLinear
    w_conv: ShortConv | None
    beta_proj: Linear
    g_proj: Linear | None
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None
    o_proj: Linear
    cfg: PathAttentionConfig = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        cfg: PathAttentionConfig,
        *,
        key: Array,
        std: float,
        norm_eps: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        kq, kk, kv, kw, kc, kb, kg, ko = jax.random.split(key, 8)
        heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        q_width, kv_width = heads * head_dim, kv_heads * head_dim
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.q_proj = Linear(d_model, q_width, key=kq, axes=("embed", "heads"), **common)
        self.k_proj = Linear(d_model, kv_width, key=kk, axes=("embed", "kv_heads"), **common)
        self.v_proj = Linear(d_model, kv_width, key=kv, axes=("embed", "kv_heads"), **common)
        if cfg.w_low_rank_dim is None:
            self.w_proj = Linear(d_model, kv_width, key=kw, axes=("embed", "kv_heads"), **common)
        else:
            self.w_proj = LowRankLinear(d_model, cfg.w_low_rank_dim, kv_width, key=kw, out_axis="kv_heads",
                                        **common)
        self.w_conv = (
            ShortConv(kv_width, cfg.w_conv_size, activation=True, init="uniform", key=kc, axis="kv_heads",
                      param_dtype=param_dtype)
            if cfg.w_conv_size else None
        )
        self.beta_proj = Linear(d_model, kv_heads, key=kb, axes=("embed", None), use_bias=True, **common)
        self.g_proj = (
            Linear(d_model, heads, key=kg, axes=("embed", None), use_bias=True, **common)
            if cfg.forget_gate else None
        )
        if cfg.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
        else:
            self.q_norm = self.k_norm = None
        self.o_proj = Linear(q_width, d_model, key=ko, axes=("heads", "embed"), **common)
        self.cfg = cfg

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        cfg = self.cfg
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(x).reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        v = self.v_proj(x).reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        w = self.w_proj(x)
        if self.w_conv is not None:
            w, _ = self.w_conv(w, segment_ids=_document_index(segment_ids))
        w = l2_normalize(w.reshape(batch, length, cfg.num_kv_heads, cfg.head_dim).astype(jnp.float32))
        beta = 2.0 * jax.nn.sigmoid(self.beta_proj(x).astype(jnp.float32))
        g = None if self.g_proj is None else jax.nn.log_sigmoid(self.g_proj(x).astype(jnp.float32))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        o = path_attention(q, k, v, w, beta, g, segment_ids=segment_ids, scale=cfg.softmax_scale,
                           chunk_size=cfg.chunk_size, implementation=cfg.implementation)
        return self.o_proj(o.reshape(batch, length, -1))

    def init_cache(self, batch: int, max_len: int, dtype: jnp.dtype):
        """An empty :class:`~jla.layers.path_decode.PathCache` of ``max_len`` positions."""
        from jla.layers.path_decode import init_path_cache

        return init_path_cache(self, batch, max_len, dtype)

    def step(self, x: Array, positions: Array, cache):
        """Prefill or decode ``x [B, T, D]`` at ``positions [B, T]`` (the reference op's key-reflection scan)."""
        from jla.layers.path_decode import path_decode_step

        return path_decode_step(self, x, positions, cache)


def _document_index(segment_ids: Array | None) -> Array | None:
    """Running document number, so the w conv starts a new document at every id change, as the op does.

    ``ShortConv`` keeps a tap whose id equals the current id, so a reused id (``[0, 1, 0]``) with a middle
    document shorter than the conv would otherwise read across it.
    """
    if segment_ids is None:
        return None
    changed = (segment_ids[:, 1:] != segment_ids[:, :-1]).astype(segment_ids.dtype)
    return jnp.cumsum(jnp.concatenate([jnp.zeros_like(segment_ids[:, :1]), changed], axis=1), axis=1)
