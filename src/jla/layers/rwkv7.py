# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/rwkv7.py, fla/models/rwkv7/modeling_rwkv7.py,
# fla/ops/rwkv7/fused_addcmul.py, fla/ops/rwkv7/fused_k_update.py, fla/ops/rwkv7/gate_output_correction.py.
"""RWKV-7 (Goose) time mix, channel mix and block.

Time mix (FLA ``RWKV7Attention``): token shift ``delta = x_{t-1} - x_t`` and per-channel
mixes ``x_* = x + delta * mu_*``; ``r, k, v`` projections; LoRA projections
``w = -exp(-1/2) sigmoid(w0 + tanh(x W1) W2)`` (log decay), ``a = sigmoid(a0 + x A1 A2)``,
``g = sigmoid(x G1) G2`` and, after the first layer, the value residual
``v = lerp(v, v_first, sigmoid(v0 + x V1 V2))``; ``kk = l2norm(k * k_k)`` per head,
``k = k * (1 + (a - 1) k_a)``; the DPLR recurrence with ``a = -kk, b = kk * a`` and scale 1
(``r, k, v`` in the compute dtype, as FLA passes them: in bf16 with ``chunk_size=32``, ``impl="auto"``
runs the Mosaic GPU kernels on H100); a per-head group norm; the bonus ``(r . (k * r_k)) v``; the output
gate ``g`` and ``o_proj``.

Channel mix (FLA ``RWKV7FeedForward``): token shift, ``x_k`` mix, ``value(relu(key(x))^2)``.

Block: ``x = ln0(x)`` on the first layer, ``x + att(ln1(x))``, ``x + ffn(ln2(x))`` (LayerNorms).
``segment_ids`` restart both token shifts and the recurrent state at document starts.
Decode uses :class:`RWKV7State` (the shifted token of each mix and the ``[H, K, V]`` state).

Initialisation: the mixes and per-channel vectors follow FLA / BlinkDL; ``r``, ``k``, ``v`` and the
channel-mix ``key`` draw truncated normals with FLA's orthogonal scales (``C^-1/2``, ``0.1 C^-1/2``), not
orthogonal matrices. ``init_scheme="jla"`` (jla's original initialisation; the default of these layer classes)
draws ``o_proj``, the channel-mix ``value`` and the LoRAs as truncated normals too. ``init_scheme="upstream"`` (the default of
``ModelConfig`` and ``RWKV7Config``) follows FLA
(``RWKV7Attention``, ``RWKV7PreTrainedModel``, ``LoRA``) and the official RWKV-v7 ``train_temp`` model:
``o_proj`` and the channel-mix ``value`` start at zero (a fresh block is the identity on the residual
stream), the LoRA down projections (``w1``, ``a1``, ``v1``, ``g1``) at zero and the up projections
orthogonal with gain 0.1, so at step 0 ``w = w0``, ``a = sigmoid(a0)`` and the value residual mix is
``sigmoid(v0)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.layers.linear import Linear
from jla.ops import linear_attention
from jla.ops.rwkv7 import segment_resets

__all__ = ["LayerNorm", "RWKV7Block", "RWKV7ChannelMix", "RWKV7State", "RWKV7TimeMix", "RWKV7_LOWER_BOUND",
           "rwkv7_lora_dims"]

Array = jax.Array
_LOG_DECAY_SCALE = 0.6065306597126334  # exp(-0.5)
# The log decay w = -exp(-1/2) sigmoid(.) is at least -0.6066 for every input (also in bf16), so the
# layers promise the op ``lower_bound = -0.61``: the Mosaic GPU kernel then builds each chunk's score
# matrices with one dense product (its bounded mode, FLA RWKV7Attention's ``safe_gate``), the same
# function as its exact sub-chunk mode without the [B, H, N, 4, 8, 8, K] decay tensors. The bounded
# mode's products always take f32 operands (its default runs them as split-bf16 tensor-core products,
# jla.ops.mosaic_gpu.rwkv7.BOUNDED_PRECISION): plain bf16 operands lost dq and dk accuracy under the
# cotangents of the group norm (11% norm error against 0.4%). The reference
# ignores the option and the chunked op computes the same function in f32.
RWKV7_LOWER_BOUND = -0.61
_OP_OPTIONS = {"lower_bound": RWKV7_LOWER_BOUND}


def rwkv7_lora_dims(d_model: int, head_dim: int) -> tuple[int, int, int, int]:
    """FLA's default LoRA ranks ``(decay, a, v, gate)``."""
    factor = head_dim / 64
    decay = max(32, int(round((2.5 * d_model**0.5) * factor / 32) * 32))
    a = max(32, int(round((2.5 * d_model**0.5) * factor / 32) * 32))
    v = max(32, int(round((1.7 * d_model**0.5) * factor / 32) * 32))
    gate = max(32, int(round((5 * d_model**0.5) / 32) * 32))
    return decay, a, v, gate


def _upstream_init(init_scheme: str) -> bool:
    """True for ``init_scheme="upstream"`` (FLA / BlinkDL zero and orthogonal inits), False for ``"jla"``."""
    if init_scheme not in ("jla", "upstream"):
        raise ValueError(f"unknown init_scheme {init_scheme!r}; expected 'jla' or 'upstream'")
    return init_scheme == "upstream"


class LayerNorm(eqx.Module):
    weight: Array
    bias: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-5, param_dtype: jnp.dtype = jnp.float32):
        self.weight = jnp.ones((dim,), param_dtype)
        self.bias = jnp.zeros((dim,), param_dtype)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        xf = x.astype(jnp.float32)
        mean = xf.mean(-1, keepdims=True)
        var = jnp.square(xf - mean).mean(-1, keepdims=True)
        y = (xf - mean) * jax.lax.rsqrt(var + self.eps)
        return (y * self.weight.astype(jnp.float32) + self.bias.astype(jnp.float32)).astype(x.dtype)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"weight": ("embed",), "bias": ("embed",)}


class _LoRA(eqx.Module):
    """``act(x A) B (+ bias)``; ``act`` is ``tanh``, ``sigmoid`` or none."""

    down: Linear
    up: Linear
    activation: str = eqx.field(static=True)

    def __init__(self, d_in: int, rank: int, d_out: int, *, key: Array, activation: str, use_bias: bool,
                 param_dtype: jnp.dtype, compute_dtype: str, init_scheme: str = "jla"):
        k1, k2 = jax.random.split(key)
        # init_scheme "upstream": FLA LoRA._initialize_weights / official RWKV-v7 w1..g1 = 0, ortho_init(w2..g2, 0.1):
        # the down projection starts at zero (the LoRA output is then its bias, or sigmoid(0) G for the gate), the
        # up projection orthogonal with gain 0.1 (times sqrt(rank / d_out) when rank > d_out). "jla": truncated
        # normals with std d_in^-1/2 (down) and 0.1 rank^-1/2 (up).
        gain = 0.1 * (math.sqrt(rank / d_out) if rank > d_out else 1.0)
        upstream = _upstream_init(init_scheme)
        self.down = Linear(d_in, rank, key=k1, axes=("embed", None), std=d_in**-0.5,
                           **(dict(init="zeros") if upstream else {}), param_dtype=param_dtype,
                           compute_dtype=compute_dtype)
        self.up = Linear(rank, d_out, key=k2, axes=(None, "heads"), std=0.1 * rank**-0.5, use_bias=use_bias,
                         **(dict(init="orthogonal", gain=gain) if upstream else {}), param_dtype=param_dtype,
                         compute_dtype=compute_dtype)
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        h = self.down(x)
        if self.activation == "tanh":
            h = jnp.tanh(h)
        elif self.activation == "sigmoid":
            h = jax.nn.sigmoid(h)
        return self.up(h)


class RWKV7State(NamedTuple):
    """Decode cache of one RWKV-7 block."""

    att_shift: Array  # [B, D] last input of the time mix
    ffn_shift: Array  # [B, D] last input of the channel mix
    state: Array  # [B, H, K, V] float32


def _token_shift(x: Array, segment_ids: Array | None, prev: Array | None) -> Array:
    """``x_{t-1}`` along T: ``prev`` (or zero) at ``t = 0`` and zero at every document start."""
    first = jnp.zeros_like(x[:, :1]) if prev is None else prev[:, None].astype(x.dtype)
    shifted = jnp.concatenate([first, x[:, :-1]], axis=1)
    if segment_ids is not None:
        shifted = jnp.where(segment_resets(segment_ids)[..., None], 0.0, shifted)
    return shifted


class RWKV7TimeMix(eqx.Module):
    mu: Array  # [6, D]: r, w, k, v, a, g token-shift mixes
    k_k: Array  # [D]
    k_a: Array  # [D]
    r_k: Array  # [H, K]
    r_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    w_lora: _LoRA
    a_lora: _LoRA
    v_lora: _LoRA | None
    g_lora: _LoRA
    norm_weight: Array  # [D] group norm
    norm_bias: Array  # [D]
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    layer_idx: int = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    impl: str = eqx.field(static=True)

    def __init__(self, d_model: int, *, head_dim: int, layer_idx: int, n_layers: int, key: Array,
                 lora_dims: tuple[int, int, int, int] | None = None, norm_eps: float = 1e-5,
                 chunk_size: int = 16, impl: str = "auto", param_dtype: jnp.dtype = jnp.float32,
                 compute_dtype: str = "bfloat16", init_scheme: str = "jla"):
        if d_model % head_dim:
            raise ValueError(f"d_model {d_model} must be divisible by head_dim {head_dim}")
        decay_r, a_r, v_r, g_r = lora_dims or rwkv7_lora_dims(d_model, head_dim)
        keys = jax.random.split(key, 8)
        common = dict(param_dtype=param_dtype, compute_dtype=compute_dtype)
        lora = dict(common, init_scheme=init_scheme)
        o_init = dict(init="zeros") if _upstream_init(init_scheme) else {}
        std = d_model**-0.5
        self.num_heads = d_model // head_dim
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps
        self.chunk_size = chunk_size
        self.impl = impl
        # FLA / BlinkDL initialisation of the mixes and per-channel vectors.
        ratio_0_to_1 = layer_idx / max(n_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_idx / n_layers
        n = jnp.arange(d_model, dtype=jnp.float32)
        ddd = n / d_model
        linear = n / max(d_model - 1, 1) - 0.5
        zig = ((n % head_dim) - (head_dim - 1) / 2) / max((head_dim - 1) / 2, 1e-9)
        zigzag = zig * jnp.abs(zig)
        www = -6 + 6 * (n / max(d_model - 1, 1)) ** (1 + ratio_0_to_1**0.3)
        powers = jnp.array([0.2, 0.9, 0.7, 0.7, 0.9, 0.2]) * ratio_1_to_almost0
        self.mu = (1.0 - ddd[None, :] ** powers[:, None]).astype(param_dtype)
        self.k_k = (0.71 - linear * 0.1).astype(param_dtype)
        self.k_a = jnp.full((d_model,), 1.02, param_dtype)
        self.r_k = jnp.full((self.num_heads, head_dim), -0.04, param_dtype)
        self.r_proj = Linear(d_model, d_model, key=keys[0], axes=("embed", "heads"), std=std, **common)
        self.k_proj = Linear(d_model, d_model, key=keys[1], axes=("embed", "heads"), std=0.1 * std, **common)
        self.v_proj = Linear(d_model, d_model, key=keys[2], axes=("embed", "heads"), std=std, **common)
        self.o_proj = Linear(d_model, d_model, key=keys[3], axes=("heads", "embed"), std=std, **o_init, **common)
        self.w_lora = _LoRA(d_model, decay_r, d_model, key=keys[4], activation="tanh", use_bias=True, **lora)
        self.w_lora = eqx.tree_at(lambda m: m.up.bias, self.w_lora, (www + 0.5 + zigzag * 2.5).astype(param_dtype))
        self.a_lora = _LoRA(d_model, a_r, d_model, key=keys[5], activation="none", use_bias=True, **lora)
        self.a_lora = eqx.tree_at(lambda m: m.up.bias, self.a_lora,
                                  (-0.19 + zigzag * 0.3 + linear * 0.4).astype(param_dtype))
        if layer_idx > 0:
            v_lora = _LoRA(d_model, v_r, d_model, key=keys[6], activation="none", use_bias=True, **lora)
            self.v_lora = eqx.tree_at(lambda m: m.up.bias, v_lora, (0.73 - linear * 0.4).astype(param_dtype))
        else:
            self.v_lora = None
        self.g_lora = _LoRA(d_model, g_r, d_model, key=keys[7], activation="sigmoid", use_bias=False, **lora)
        self.norm_weight = jnp.full((d_model,), ((layer_idx + 1) / n_layers) ** 0.7, param_dtype)
        self.norm_bias = jnp.zeros((d_model,), param_dtype)

    def _group_norm(self, o: Array) -> Array:
        # o [B, T, H, V] float32: per-head normalisation (nn.GroupNorm(H, D), eps = head_dim * norm_eps).
        mean = o.mean(-1, keepdims=True)
        var = jnp.square(o - mean).mean(-1, keepdims=True)
        o = (o - mean) * jax.lax.rsqrt(var + self.head_dim * self.norm_eps)
        o = o.reshape(*o.shape[:2], -1)
        return o * self.norm_weight.astype(jnp.float32) + self.norm_bias.astype(jnp.float32)

    def __call__(self, x: Array, v_first: Array | None, segment_ids: Array | None = None, *,
                 shift: Array | None = None, state: Array | None = None
                 ) -> tuple[Array, Array, Array, Array]:
        """``x [B, T, D]`` -> ``(out, v_first, last_x, final_state)``."""
        if (self.layer_idx == 0) != (v_first is None):
            raise ValueError("v_first must be None exactly on the first layer")
        batch, seq_len, d = x.shape
        f32 = jnp.float32
        xf = x.astype(f32)
        delta = _token_shift(xf, segment_ids, shift) - xf
        mu = self.mu.astype(f32)
        xr, xw, xk, xv, xa, xg = (xf + delta * mu[i] for i in range(6))
        r = self.r_proj(xr).astype(f32)
        w = -_LOG_DECAY_SCALE * jax.nn.sigmoid(self.w_lora(xw).astype(f32))
        k = self.k_proj(xk).astype(f32)
        v = self.v_proj(xv).astype(f32)
        if self.v_lora is None:
            v_first = v
        else:
            mix = jax.nn.sigmoid(self.v_lora(xv).astype(f32))
            v = v + (v_first.astype(f32) - v) * mix
        a = jax.nn.sigmoid(self.a_lora(xa).astype(f32))
        g = self.g_lora(xg).astype(f32)
        heads = (batch, seq_len, self.num_heads, self.head_dim)
        kk = (k * self.k_k.astype(f32)).reshape(heads)
        kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        k = k * (1.0 + (a - 1.0) * self.k_a.astype(f32))
        r_h, w_h, k_h, a_h, v_h = (t.reshape(heads) for t in (r, w, k, a, v))
        # The op takes r, k, v in the compute dtype, as FLA's layer passes them (the Mosaic GPU kernels
        # need bf16 r, k, v, so a bf16 layer with chunk_size=32 runs them under impl="auto"); w, a, b stay f32.
        op_dtype = jnp.dtype(self.r_proj.compute_dtype)
        o, final = linear_attention(
            "rwkv7", r_h.astype(op_dtype), k_h.astype(op_dtype), v_h.astype(op_dtype), w=w_h, a=-kk, b=kk * a_h,
            scale=1.0, impl=self.impl, **_OP_OPTIONS,
            initial_state=state, output_final_state=True, segment_ids=segment_ids,
            **({} if self.impl == "reference" else {"chunk_size": self.chunk_size}),
        )
        o = self._group_norm(o.astype(f32))
        bonus = (r_h * k_h * self.r_k.astype(f32)).sum(-1, keepdims=True) * v_h
        o = (o + bonus.reshape(batch, seq_len, d)) * g
        out = self.o_proj(o.astype(x.dtype))
        return out, v_first, x[:, -1], final


class RWKV7ChannelMix(eqx.Module):
    x_k: Array
    key: Linear
    value: Linear

    def __init__(self, d_model: int, d_ff: int, *, layer_idx: int, n_layers: int, key: Array,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16", init_scheme: str = "jla"):
        k1, k2 = jax.random.split(key)
        ratio_1_to_almost0 = 1.0 - layer_idx / n_layers
        ddd = jnp.arange(d_model, dtype=jnp.float32) / d_model
        self.x_k = (1.0 - ddd**(ratio_1_to_almost0**4)).astype(param_dtype)
        self.key = Linear(d_model, d_ff, key=k1, axes=("embed", "mlp"), std=d_model**-0.5,
                          param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.value = Linear(d_ff, d_model, key=k2, axes=("mlp", "embed"), std=d_ff**-0.5,
                            **(dict(init="zeros") if _upstream_init(init_scheme) else {}),
                            param_dtype=param_dtype, compute_dtype=compute_dtype)

    def __call__(self, x: Array, segment_ids: Array | None = None, *, shift: Array | None = None
                 ) -> tuple[Array, Array]:
        xf = x.astype(jnp.float32)
        xk = xf + (_token_shift(xf, segment_ids, shift) - xf) * self.x_k.astype(jnp.float32)
        h = jnp.square(jax.nn.relu(self.key(xk)))
        return self.value(h).astype(x.dtype), x[:, -1]


class RWKV7Block(eqx.Module):
    """``x = ln0(x)`` (first layer), ``x + att(ln1(x))``, ``x + ffn(ln2(x))``."""

    ln0: LayerNorm | None
    ln1: LayerNorm
    ln2: LayerNorm
    att: RWKV7TimeMix
    ffn: RWKV7ChannelMix

    def __init__(self, d_model: int, d_ff: int, *, head_dim: int, layer_idx: int, n_layers: int, key: Array,
                 lora_dims: tuple[int, int, int, int] | None = None, norm_eps: float = 1e-5,
                 chunk_size: int = 16, impl: str = "auto", param_dtype: jnp.dtype = jnp.float32,
                 compute_dtype: str = "bfloat16", init_scheme: str = "jla"):
        k1, k2 = jax.random.split(key)
        self.ln0 = LayerNorm(d_model, eps=norm_eps, param_dtype=param_dtype) if layer_idx == 0 else None
        self.ln1 = LayerNorm(d_model, eps=norm_eps, param_dtype=param_dtype)
        self.ln2 = LayerNorm(d_model, eps=norm_eps, param_dtype=param_dtype)
        self.att = RWKV7TimeMix(d_model, head_dim=head_dim, layer_idx=layer_idx, n_layers=n_layers, key=k1,
                                lora_dims=lora_dims, norm_eps=norm_eps, chunk_size=chunk_size, impl=impl,
                                param_dtype=param_dtype, compute_dtype=compute_dtype, init_scheme=init_scheme)
        self.ffn = RWKV7ChannelMix(d_model, d_ff, layer_idx=layer_idx, n_layers=n_layers, key=k2,
                                   param_dtype=param_dtype, compute_dtype=compute_dtype, init_scheme=init_scheme)

    def __call__(self, x: Array, v_first: Array | None, segment_ids: Array | None = None, *,
                 cache: RWKV7State | None = None) -> tuple[Array, Array, RWKV7State]:
        """``x [B, T, D]`` -> ``(x, v_first, cache after the last token)``."""
        if self.ln0 is not None:
            x = self.ln0(x)
        h = self.ln1(x)
        att, v_first, att_last, state = self.att(
            h, v_first, segment_ids,
            shift=None if cache is None else cache.att_shift,
            state=None if cache is None else cache.state,
        )
        x = x + att
        h = self.ln2(x)
        ffn, ffn_last = self.ffn(h, segment_ids, shift=None if cache is None else cache.ffn_shift)
        x = x + ffn
        return x, v_first, RWKV7State(att_last, ffn_last, state)

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RWKV7State:
        d = self.ln1.weight.shape[0]
        heads, dim = self.att.num_heads, self.att.head_dim
        return RWKV7State(jnp.zeros((batch, d), dtype), jnp.zeros((batch, d), dtype),
                          jnp.zeros((batch, heads, dim, dim), jnp.float32))

    def step(self, x: Array, v_first: Array | None, cache: RWKV7State) -> tuple[Array, Array, RWKV7State]:
        """One decode token: ``x [B, D]`` -> ``(x, v_first [B, D], cache)``."""
        y, v_first, cache = self(x[:, None], None if v_first is None else v_first[:, None], cache=cache)
        return y[:, 0], v_first[:, 0], cache


# LanguageModel integration: layer type "rwkv7" and mlp type "rwkv7_cmix" (jla.models.lm) -------------
from jax.ad_checkpoint import checkpoint_name  # noqa: E402

from jla.layers.attention import ATTENTION_CHECKPOINT_NAME  # noqa: E402
from jla.layers.cache import RecurrentCache  # noqa: E402

__all__ += ["RWKV7FeedForward", "RWKV7Mixer", "ValueResidual"]


class ValueResidual(NamedTuple):
    """The RWKV-7 value residual carried through a ``LanguageModel`` stack.

    ``v_first`` is the value of the stack's first RWKV-7 layer (``[B, T, D]``, compute
    dtype: it is a projection output, so the cast is exact) and ``started`` (bool
    scalar) says whether an earlier layer has set it.
    """

    v_first: Array
    started: Array

    @classmethod
    def empty(cls, batch: int, length: int, d_model: int, dtype: jnp.dtype) -> ValueResidual:
        return cls(jnp.zeros((batch, length, d_model), dtype), jnp.zeros((), bool))


class RWKV7Mixer(eqx.Module):
    """RWKV-7 time mix as a ``LanguageModel`` mixer (``layer_types`` entry ``"rwkv7"``).

    The computation and initialisation of :class:`RWKV7TimeMix` at the block's
    ``layer_idx``, laid out so that every layer has the same structure and
    ``scan_layers`` can stack them: every layer owns a value-residual LoRA, and the
    stack's first RWKV-7 layer is found at run time from the :class:`ValueResidual`
    carry instead of a static index. That layer passes its ``v`` on as ``v_first`` (its
    value LoRA is computed but unused, so it gets no gradient); every later layer
    mixes ``v`` towards ``v_first``. Called without a carry (``__call__``, ``step``) the
    layer acts as a first layer.

    Reads ``LinearAttentionConfig``: ``num_k_heads == num_v_heads`` heads with
    ``head_k_dim == head_v_dim`` and ``heads * head_dim == d_model`` (RWKV-7's square
    projections), ``conv_size=0``, ``output_gate=True`` (the gate LoRA),
    ``rwkv7_lora_dims`` (``None`` = FLA's defaults), ``chunk_size`` and
    ``implementation``. The group-norm epsilon is ``head_dim * norm_eps``, as in FLA.
    """

    mu: Array  # [6, D]: r, w, k, v, a, g token-shift mixes
    k_k: Array  # [D]
    k_a: Array  # [D]
    r_k: Array  # [H, K]
    r_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    w_lora: _LoRA
    a_lora: _LoRA
    v_lora: _LoRA
    g_lora: _LoRA
    norm_weight: Array  # [D] group norm
    norm_bias: Array  # [D]
    cfg: object = eqx.field(static=True)  # jla.models.config.LinearAttentionConfig
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)

    def __init__(self, d_model: int, cfg, *, layer_idx: int, n_layers: int, key: Array, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16", init_scheme: str = "jla"):
        if cfg.num_k_heads != cfg.num_v_heads or cfg.head_k_dim != cfg.head_v_dim:
            raise ValueError("rwkv7 needs num_k_heads == num_v_heads and head_k_dim == head_v_dim")
        if cfg.num_k_heads * cfg.head_k_dim != d_model:
            raise ValueError(f"rwkv7 needs num_k_heads * head_k_dim == d_model ({d_model}); "
                             f"got {cfg.num_k_heads} x {cfg.head_k_dim}")
        if cfg.conv_size != 0:
            raise ValueError(f"rwkv7 uses no short convolution; set conv_size=0 (got {cfg.conv_size})")
        if not cfg.output_gate:
            raise ValueError("rwkv7 always gates its output (the gate LoRA): set output_gate=True")
        lora_dims = cfg.rwkv7_lora_dims or rwkv7_lora_dims(d_model, cfg.head_k_dim)
        k_mix, k_v = jax.random.split(key)
        mix = RWKV7TimeMix(d_model, head_dim=cfg.head_k_dim, layer_idx=layer_idx, n_layers=n_layers, key=k_mix,
                           lora_dims=lora_dims, norm_eps=norm_eps, chunk_size=cfg.chunk_size,
                           impl=cfg.implementation, param_dtype=param_dtype, compute_dtype=compute_dtype,
                           init_scheme=init_scheme)
        for name in ("mu", "k_k", "k_a", "r_k", "r_proj", "k_proj", "v_proj", "o_proj", "w_lora", "a_lora",
                     "g_lora", "norm_weight", "norm_bias"):
            setattr(self, name, getattr(mix, name))
        if mix.v_lora is None:  # the first layer: the same initialisation as the later layers' value LoRA
            linear = jnp.arange(d_model, dtype=jnp.float32) / max(d_model - 1, 1) - 0.5
            v_lora = _LoRA(d_model, lora_dims[2], d_model, key=k_v, activation="none", use_bias=True,
                           param_dtype=param_dtype, compute_dtype=compute_dtype, init_scheme=init_scheme)
            self.v_lora = eqx.tree_at(lambda m: m.up.bias, v_lora, (0.73 - linear * 0.4).astype(param_dtype))
        else:
            self.v_lora = mix.v_lora
        self.cfg = cfg
        self.num_heads = cfg.num_k_heads
        self.head_dim = cfg.head_k_dim
        self.norm_eps = float(norm_eps)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"mu": (None, "embed"), "k_k": (None,), "k_a": (None,), "r_k": (None, None),
                "norm_weight": (None,), "norm_bias": (None,)}

    def _group_norm(self, o: Array) -> Array:
        mean = o.mean(-1, keepdims=True)
        var = jnp.square(o - mean).mean(-1, keepdims=True)
        o = (o - mean) * jax.lax.rsqrt(var + self.head_dim * self.norm_eps)
        o = o.reshape(*o.shape[:2], -1)
        return o * self.norm_weight.astype(jnp.float32) + self.norm_bias.astype(jnp.float32)

    def run(self, x: Array, *, segment_ids: Array | None = None, cache: RecurrentCache | None = None,
            value_residual: ValueResidual | None = None
            ) -> tuple[Array, RecurrentCache | None, ValueResidual | None]:
        """``x [B, T, D]`` -> ``(out, cache, value_residual)``; ``cache`` continues a decode."""
        cfg = self.cfg
        if cfg.context_parallel_axis is not None and cache is None:
            raise NotImplementedError("context parallelism (context_parallel_axis) does not support rwkv7 layers")
        batch, seq_len, d = x.shape
        f32 = jnp.float32
        xf = x.astype(f32)
        last = None if cache is None else cache.conv[0][:, 0]
        delta = _token_shift(xf, segment_ids, last) - xf
        mu = self.mu.astype(f32)
        xr, xw, xk, xv, xa, xg = (xf + delta * mu[i] for i in range(6))
        r = self.r_proj(xr).astype(f32)
        w = -_LOG_DECAY_SCALE * jax.nn.sigmoid(self.w_lora(xw).astype(f32))
        k = self.k_proj(xk).astype(f32)
        v = self.v_proj(xv).astype(f32)
        if value_residual is not None:
            started = value_residual.started
            v_first = value_residual.v_first
            mixed = v + (v_first.astype(f32) - v) * jax.nn.sigmoid(self.v_lora(xv).astype(f32))
            value_residual = ValueResidual(jnp.where(started, v_first, v.astype(v_first.dtype)), jnp.ones((), bool))
            v = jnp.where(started, mixed, v)
        a = jax.nn.sigmoid(self.a_lora(xa).astype(f32))
        g = self.g_lora(xg).astype(f32)
        heads = (batch, seq_len, self.num_heads, self.head_dim)
        kk = (k * self.k_k.astype(f32)).reshape(heads)
        kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        k = k * (1.0 + (a - 1.0) * self.k_a.astype(f32))
        r_h, w_h, k_h, a_h, v_h = (t.reshape(heads) for t in (r, w, k, a, v))
        options: dict = {"chunk_size": cfg.chunk_size, **_OP_OPTIONS}  # the dispatcher drops chunk_size for
        # the reference
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if seq_len == 1:
                impl = "reference"  # a single recurrent step; chunking would pad it to a full chunk
        # r, k, v go to the op in the compute dtype, as RWKV7TimeMix and FLA's layer pass them (the Mosaic GPU
        # kernel needs bf16 r, k, v); w, a, b stay f32.
        op_dtype = jnp.dtype(self.r_proj.compute_dtype)
        o, final = linear_attention("rwkv7", r_h.astype(op_dtype), k_h.astype(op_dtype), v_h.astype(op_dtype),
                                    w=w_h, a=-kk, b=kk * a_h, scale=1.0, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        o = self._group_norm(o.astype(f32))
        bonus = (r_h * k_h * self.r_k.astype(f32)).sum(-1, keepdims=True) * v_h
        o = (o + bonus.reshape(batch, seq_len, d)) * g
        out = self.o_proj(o.astype(x.dtype))
        if cache is not None:
            cache = RecurrentCache(state=final, conv=(x[:, -1:].astype(cache.conv[0].dtype),))
        return out, cache, value_residual

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        return self.run(x, segment_ids=segment_ids)[0]

    def first_value(self, x: Array, segment_ids: Array | None, dtype: jnp.dtype) -> Array:
        """The ``v`` that :meth:`run` passes on as ``v_first`` when this is the stack's first layer
        (``[B, T, D]`` in ``dtype``, the :class:`ValueResidual` carry's dtype)."""
        xf = x.astype(jnp.float32)
        xv = xf + (_token_shift(xf, segment_ids, None) - xf) * self.mu[3].astype(jnp.float32)
        return self.v_proj(xv).astype(jnp.float32).astype(dtype)

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        d = self.num_heads * self.head_dim
        state = jnp.zeros((batch, self.num_heads, self.head_dim, self.head_dim), jnp.float32)
        return RecurrentCache(state=state, conv=(jnp.zeros((batch, 1, d), dtype),))

    def step(self, x: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``x [B, T, D]`` from ``cache`` (as a first layer: no value residual)."""
        out, cache, _ = self.run(x, cache=cache)
        return out, cache


class RWKV7FeedForward(RWKV7ChannelMix):
    """:class:`RWKV7ChannelMix` as a ``LanguageModel`` channel mixer (``mlp_types`` entry ``"rwkv7_cmix"``).

    ``last [B, 1, D]`` is the previous normed token during decode (the block's
    ``rwkv_cmix`` cache convention).
    """

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"x_k": ("embed",)}

    def __call__(self, x: Array, segment_ids: Array | None = None, last: Array | None = None) -> Array:
        return super().__call__(x, segment_ids, shift=None if last is None else last[:, 0])[0]


__all__ += ["hoisted_value_residual"]


def hoisted_value_residual(block, x: Array, segment_ids: Array | None, dtype: jnp.dtype) -> Array:
    """``v_first`` of a scanned stack whose first block is an RWKV-7 layer, computed before the scan.

    ``block`` is the stack's first ``jla.models.lm.Block`` (``x`` its input). Inside ``lax.scan`` under
    ``model.remat``, a ``v_first`` in the loop carry is saved once per layer for the backward pass
    (``[n_layers, B, T, D]``: 6 GiB at the 1.3B preset's 16 x 4096 tokens); passed to the loop as a
    constant it is saved once. The first layer still computes its own ``v`` and does not mix; this
    adds the first layer's token shift, ``v`` mix and value projection once more. Under FP8 delayed
    scaling that value projection therefore counts two uses per step (``jla.training.fp8`` divides
    its scales and sums the two input amaxes, at most one binade of e4m3 headroom on that GEMM).
    """
    return block.mixer.first_value(block.mixer_norm(x), segment_ids, dtype)
