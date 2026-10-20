# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/rwkv6.py and fla/layers/hgrn2.py
"""RWKV-6 (Finch) time mixing and HGRN2 token mixers, following FLA's layers.

``RWKV6`` (layer type ``"rwkv6"``), FLA's ``RWKV6Attention``:

* token shift ``delta_t = x_{t-1} - x_t`` (zero before the first token and at
  every ``segment_ids`` change; decode carries the last token);
* data-dependent lerp (ddlerp): ``m = tanh((x + delta mu) W_down) W_up + bias``
  gives five mixes (``r, w, k, v, g``), each projection reads ``x + delta m_*``;
* the per-channel decay is a tanh LoRA, ``w = -exp(tanh(x_w W_a) W_b + b)``;
* ``jla.ops`` kind ``rwkv6`` with the bonus ``u [H, K]`` and ``scale = 1``;
* per-head group norm (mean and variance, weight and bias), times ``swish(g)``,
  and the output projection.

``HGRN2`` (layer type ``"hgrn2"``), FLA's ``HGRN2Attention``: ``q = swish(x W_q)``,
forget gate ``g = logsigmoid(x W_f)``, key ``k = 1 - exp(g)`` (the outer-product
state expansion of HGRN2) and value ``i = x W_i``, run through the ``gla`` kind;
an RMSNorm over all heads and the output projection. FLA's optional per-layer
lower bound on the forget gate is not implemented (every layer uses the
first-layer form).

Both read ``jla.models.config.LinearAttentionConfig``: ``num_k_heads`` heads
(``num_v_heads`` must be equal), ``head_k_dim`` / ``head_v_dim``,
``gate_low_rank_dim`` (RWKV-6 decay LoRA rank, 64 by default), ``chunk_size``
and ``implementation``. Neither uses short convolutions (``conv_size`` must be
0); RWKV-6 always gates its output and HGRN2 never does, so ``output_gate`` must
be ``True`` for ``rwkv6`` and ``False`` for ``hgrn2``. Parameters follow FLA's
initialization: projections ``N(0, initializer_range)`` (truncated), zero lerp
weights, mix biases and bonus, unit norm weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.linear import Linear, truncated_normal
from jla.layers.norm import RMSNorm
from jla.ops.reference import segment_resets

if TYPE_CHECKING:
    from jla.models.config import LinearAttentionConfig

__all__ = ["HGRN2", "MIX_RANK", "RWKV6", "token_shift"]

Array = jax.Array

MIX_RANK = 32  # FLA's proj_low_rank_dim: rank of each of the five ddlerp mixes
DECAY_RANK = 64  # FLA's gate_low_rank_dim default: rank of the decay LoRA


def token_shift(x: Array, segment_ids: Array | None, last: Array | None) -> Array:
    """``x_{t-1}`` for every ``t`` of ``x [B, T, D]``.

    ``last [B, 1, D]`` (decode) or zero precedes the first token; a token that
    starts a new segment sees zero.
    """
    prev = jnp.zeros_like(x[:, :1]) if last is None else last.astype(x.dtype)
    shifted = jnp.concatenate([prev, x[:, :-1]], axis=1)
    if segment_ids is not None:
        shifted = jnp.where(segment_resets(segment_ids)[..., None], jnp.zeros((), x.dtype), shifted)
    return shifted


def _check_config(kind: str, cfg: LinearAttentionConfig, *, output_gate: bool) -> None:
    if cfg.num_v_heads != cfg.num_k_heads:
        raise ValueError(f"{kind} needs num_v_heads == num_k_heads; got {cfg.num_v_heads} and {cfg.num_k_heads}")
    if cfg.conv_size != 0:
        raise ValueError(f"{kind} uses no short convolution; set conv_size=0 (got {cfg.conv_size})")
    if cfg.output_gate != output_gate:
        raise ValueError(f"{kind} needs output_gate={output_gate} (its output gating is fixed)")


class RWKV6(eqx.Module):
    """RWKV-6 time mixing (FLA ``RWKV6Attention``)."""

    x_mu: Array  # [D] lerp weight of the ddlerp input
    x_down: Linear  # D -> 5 * MIX_RANK
    x_up: Array  # [5, MIX_RANK, D]
    x_bias: Array  # [5, D]
    r_proj: Linear
    k_proj: Linear
    v_proj: Linear
    g_proj: Linear
    w_down: Linear  # D -> decay rank
    w_up: Linear  # decay rank -> H * K, with bias
    bonus: Array  # [H, K]
    norm_weight: Array  # [H * V]
    norm_bias: Array  # [H * V]
    o_proj: Linear
    cfg: LinearAttentionConfig = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        _check_config("rwkv6", cfg, output_gate=True)
        keys = jax.random.split(key, 9)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        key_dim = cfg.num_k_heads * cfg.head_k_dim
        value_dim = cfg.num_v_heads * cfg.head_v_dim
        rank = cfg.gate_low_rank_dim or DECAY_RANK
        self.x_mu = jnp.zeros((d_model,), param_dtype)
        self.x_down = Linear(d_model, 5 * MIX_RANK, key=keys[0], axes=("embed", None), **common)
        self.x_up = truncated_normal(keys[1], (5, MIX_RANK, d_model), std, param_dtype)
        self.x_bias = jnp.zeros((5, d_model), param_dtype)
        self.r_proj = Linear(d_model, key_dim, key=keys[2], axes=("embed", "heads"), **common)
        self.k_proj = Linear(d_model, key_dim, key=keys[3], axes=("embed", "heads"), **common)
        self.v_proj = Linear(d_model, value_dim, key=keys[4], axes=("embed", "heads"), **common)
        self.g_proj = Linear(d_model, value_dim, key=keys[5], axes=("embed", "heads"), **common)
        self.w_down = Linear(d_model, rank, key=keys[6], axes=("embed", None), **common)
        self.w_up = Linear(rank, key_dim, key=keys[7], axes=(None, "heads"), use_bias=True, **common)
        self.bonus = jnp.zeros((cfg.num_k_heads, cfg.head_k_dim), param_dtype)
        self.norm_weight = jnp.ones((value_dim,), param_dtype)
        self.norm_bias = jnp.zeros((value_dim,), param_dtype)
        self.o_proj = Linear(value_dim, d_model, key=keys[8], axes=("heads", "embed"), **common)
        self.cfg = cfg
        self.d_model = d_model
        self.norm_eps = float(norm_eps)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {
            "x_mu": ("embed",), "x_up": (None, None, "embed"), "x_bias": (None, "embed"),
            "bonus": (None, None), "norm_weight": (None,), "norm_bias": (None,),
        }

    def _group_norm(self, o: Array) -> Array:
        """Per-head layer norm of ``o [B, T, H, V]`` (float32), flattened to ``[B, T, H * V]``."""
        o = o.astype(jnp.float32)
        mean = jnp.mean(o, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(o - mean), axis=-1, keepdims=True)
        o = ((o - mean) * lax.rsqrt(var + self.norm_eps)).reshape(*o.shape[:2], -1)
        return o * self.norm_weight.astype(jnp.float32) + self.norm_bias.astype(jnp.float32)

    def _forward(self, x: Array, *, segment_ids: Array | None,
                 cache: RecurrentCache | None) -> tuple[Array, RecurrentCache | None]:
        cfg = self.cfg
        if cfg.context_parallel_axis is not None and cache is None:
            raise NotImplementedError("context parallelism (context_parallel_axis) does not support rwkv6 layers")
        batch, seq_len, _ = x.shape
        heads = cfg.num_k_heads
        last = None if cache is None else cache.conv[0]
        delta = token_shift(x, segment_ids, last) - x
        mixes = jnp.tanh(self.x_down(x + delta * self.x_mu.astype(x.dtype)))
        mixes = mixes.reshape(batch, seq_len, 5, MIX_RANK)
        mixes = jnp.einsum("btnr,nrd->btnd", mixes, self.x_up.astype(mixes.dtype)) + self.x_bias.astype(mixes.dtype)

        def lerp(i: int) -> Array:
            return x + delta * mixes[:, :, i].astype(x.dtype)

        r = self.r_proj(lerp(0)).reshape(batch, seq_len, heads, cfg.head_k_dim)
        w = self.w_up(jnp.tanh(self.w_down(lerp(1))))
        k = self.k_proj(lerp(2)).reshape(batch, seq_len, heads, cfg.head_k_dim)
        v = self.v_proj(lerp(3)).reshape(batch, seq_len, heads, cfg.head_v_dim)
        gate = self.g_proj(lerp(4))
        log_decay = -jnp.exp(w.astype(jnp.float32)).reshape(batch, seq_len, heads, cfg.head_k_dim)
        options: dict = {"chunk_size": cfg.chunk_size, "g": log_decay, "u": self.bonus.astype(jnp.float32),
                         "scale": 1.0}
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if seq_len == 1:
                impl = "reference"  # a single recurrent step; chunking would pad it to a full chunk
        o, final = ops.linear_attention("rwkv6", r, k, v, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        o = self._group_norm(o) * jax.nn.silu(gate.astype(jnp.float32))
        y = self.o_proj(o.astype(gate.dtype))
        if cache is None:
            return y, None
        return y, RecurrentCache(state=final, conv=(x[:, -1:].astype(last.dtype),))

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(x, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg = self.cfg
        state = jnp.zeros((batch, cfg.num_k_heads, cfg.head_k_dim, cfg.head_v_dim), jnp.float32)
        return RecurrentCache(state=state, conv=(jnp.zeros((batch, 1, self.d_model), dtype),))

    def step(self, x: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``x [B, T, D]`` from ``cache``; returns the output and the new cache."""
        return self._forward(x, segment_ids=None, cache=cache)


class HGRN2(eqx.Module):
    """HGRN2 (Qin et al. 2024): GLA with the key tied to the forget gate (FLA ``HGRN2Attention``)."""

    q_proj: Linear
    f_proj: Linear
    i_proj: Linear
    g_norm: RMSNorm
    o_proj: Linear
    cfg: LinearAttentionConfig = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        _check_config("hgrn2", cfg, output_gate=False)
        kq, kf, ki, ko = jax.random.split(key, 4)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        forget_dim = cfg.num_k_heads * cfg.head_k_dim
        input_dim = cfg.num_v_heads * cfg.head_v_dim
        self.q_proj = Linear(d_model, forget_dim, key=kq, axes=("embed", "heads"), **common)
        self.f_proj = Linear(d_model, forget_dim, key=kf, axes=("embed", "heads"), **common)
        self.i_proj = Linear(d_model, input_dim, key=ki, axes=("embed", "heads"), **common)
        self.g_norm = RMSNorm(input_dim, eps=norm_eps, param_dtype=param_dtype)
        self.o_proj = Linear(input_dim, d_model, key=ko, axes=("heads", "embed"), **common)
        self.cfg = cfg

    def _forward(self, x: Array, *, segment_ids: Array | None,
                 cache: RecurrentCache | None) -> tuple[Array, RecurrentCache | None]:
        cfg = self.cfg
        if cfg.context_parallel_axis is not None and cache is None:
            raise NotImplementedError("context parallelism (context_parallel_axis) does not support hgrn2 layers")
        batch, seq_len, _ = x.shape
        heads = cfg.num_k_heads
        q = jax.nn.silu(self.q_proj(x).astype(jnp.float32))
        g = jax.nn.log_sigmoid(self.f_proj(x).astype(jnp.float32))
        i = self.i_proj(x)
        k = -jnp.expm1(g)  # 1 - forget gate
        q, k, g = (t.reshape(batch, seq_len, heads, cfg.head_k_dim) for t in (q.astype(i.dtype), k.astype(i.dtype), g))
        i = i.reshape(batch, seq_len, heads, cfg.head_v_dim)
        options: dict = {"chunk_size": cfg.chunk_size, "g": g}
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if seq_len == 1:
                impl = "reference"
        o, final = ops.linear_attention("gla", q, k, i, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        y = self.o_proj(self.g_norm(o.reshape(batch, seq_len, -1)))
        if cache is None:
            return y, None
        return y, RecurrentCache(state=final, conv=None)

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(x, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg = self.cfg
        state = jnp.zeros((batch, cfg.num_k_heads, cfg.head_k_dim, cfg.head_v_dim), jnp.float32)
        return RecurrentCache(state=state, conv=None)

    def step(self, x: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``x [B, T, D]`` from ``cache``; returns the output and the new cache."""
        return self._forward(x, segment_ids=None, cache=cache)


class RWKV6ChannelMix(eqx.Module):
    """RWKV-6 channel mix (FLA's ``RWKV6FeedForward``), the block's ``mlp_type="rwkv_cmix"``.

    ``delta = x_{t-1} - x_t`` (the same token shift as the time mix, over the
    normed input), ``k = relu((x + delta mu_k) W_k)^2``,
    ``out = sigmoid((x + delta mu_r) W_r) * (k W_v)``. Decode carries the last
    normed token.
    """

    key_mu: Array  # [D]
    receptance_mu: Array  # [D]
    key: Linear
    value: Linear
    receptance: Linear

    def __init__(self, d_model: int, d_ff: int, *, key: Array, std: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k1, k2, k3 = jax.random.split(key, 3)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.key_mu = jnp.zeros((d_model,), param_dtype)
        self.receptance_mu = jnp.zeros((d_model,), param_dtype)
        self.key = Linear(d_model, d_ff, key=k1, axes=("embed", "mlp"), **common)
        self.value = Linear(d_ff, d_model, key=k2, axes=("mlp", "embed"), **common)
        self.receptance = Linear(d_model, d_model, key=k3, axes=("embed", None), **common)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"key_mu": ("embed",), "receptance_mu": ("embed",)}

    def __call__(self, x: Array, segment_ids: Array | None = None, last: Array | None = None) -> Array:
        delta = token_shift(x, segment_ids, last) - x
        k = jnp.square(jax.nn.relu(self.key(x + delta * self.key_mu.astype(x.dtype))))
        r = self.receptance(x + delta * self.receptance_mu.astype(x.dtype))
        return (jax.nn.sigmoid(r.astype(jnp.float32)) * self.value(k).astype(jnp.float32)).astype(x.dtype)
