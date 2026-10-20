# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/{gated_deltanet,kda,gla,delta_net,simple_gla}.py
# GatedDeltaNet2: Implemented from arXiv 2605.22791 (equations only)
"""Linear-attention token mixers following FLA's layer definitions.

Every mixer projects ``x [B, T, D]`` to ``q, k [B, T, Hk, Dk]`` and
``v [B, T, Hv, Dv]``, applies a short causal convolution plus SiLU to each
(``conv_size > 0``), computes its gates, runs
``jla.ops.linear_attention`` and maps the result through an output RMSNorm
(gated by ``x W_g`` when ``output_gate``) and an output projection. With
``Hv > Hk`` (grouped value heads, Qwen3-Next) every q/k head serves
``Hv // Hk`` consecutive value heads. Gates per kind:

* ``GatedDeltaNet``: ``g = -exp(A_log) * softplus(x W_a + dt_bias)`` per value head,
  ``beta = sigmoid(x W_b)``; q and k are L2-normalized;
* ``KDA``: the same Mamba-style gate per key channel through a low-rank projection,
  sigmoid-gated output norm with a low-rank gate projection;
* ``DeltaNet``: ``beta = sigmoid(x W_b)``, no decay;
* ``GLA``: ``g = logsigmoid(x W_down W_up + b) / gate_logit_normalizer`` per key channel;
* ``SimpleGLA``: ``g = logsigmoid(x W_g + b) / gate_logit_normalizer`` per head;
* ``GatedDeltaNet2`` (arXiv 2605.22791, Eq. 10-12 and Fig. 1): the KDA-style channel-wise
  decay ``g = -exp(a) * softplus(x W_f + delta)``, a channel-wise erase gate
  ``b = sigmoid(x W_b)`` on the key channels and a write gate ``w = sigmoid(x W_w)`` on
  the value channels; ``q``, ``k``, ``g`` and ``b`` live on the key heads and are
  repeated across value-head groups, ``v`` and ``w`` stay on the value heads.

With ``allow_neg_eigval`` the delta-rule write strength (``beta``, or GDN-2's erase
gate) is scaled to ``[0, 2]``.

Parameters follow FLA's initialization: projections ``N(0, initializer_range)``
(truncated; FLA draws an untruncated normal), zero biases, unit norm weights, ``A_log = log(U(0, 16))`` (GDN) or
``log(U(1, 16))`` (KDA; GDN-2 uses the GDN form) and ``dt_bias = softplus^-1(dt)`` with ``dt`` log-uniform
in ``[1e-3, 1e-1]``. The q/k/v short convs start as the identity filter (current-token tap 1) unless
``LinearAttentionConfig.conv_init`` is ``"normal"``: ``N(0, initializer_range)`` (untruncated, no bias),
as FLA initializes every ``nn.Conv1d``. ``LanguageModel`` with ``ModelConfig.init_scheme="upstream"``
picks ``"normal"`` for ``GatedDeltaNet``, ``DeltaNet``, ``GatedDeltaProduct`` and ``KDA`` when
``conv_init`` is None (:data:`UPSTREAM_NORMAL_CONV_KINDS`); ``GLA``, ``SimpleGLA`` and ``GatedDeltaNet2``
(whose paper states no conv init) keep the identity. ``step`` continues from a :class:`RecurrentCache`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.conv import ShortConv
from jla.layers.linear import Linear, LowRankLinear
from jla.layers.norm import GatedRMSNorm, RMSNorm

if TYPE_CHECKING:
    from jla.models.config import LinearAttentionConfig

__all__ = [
    "DeltaNet", "GLA", "GatedDeltaNet", "GatedDeltaNet2", "KDA", "LinearAttentionMixer", "SimpleGLA", "make_mixer",
]

Array = jax.Array

_DELTA_KINDS = ("gated_deltanet", "kda", "deltanet", "gdn2")


def _a_log_init(key: Array, heads: int, low: float) -> Array:
    a = jax.random.uniform(key, (heads,), jnp.float32, minval=low, maxval=16.0)
    return jnp.log(jnp.maximum(a, 1e-4))


def _dt_bias_init(key: Array, size: int) -> Array:
    """Inverse softplus of ``dt`` log-uniform in ``[1e-3, 1e-1]`` (Mamba / FLA)."""
    u = jax.random.uniform(key, (size,), jnp.float32)
    dt = jnp.exp(u * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    dt = jnp.maximum(dt, 1e-4)
    return dt + jnp.log(-jnp.expm1(-dt))


def _mamba_gate(a: Array, a_log: Array, dt_bias: Array) -> Array:
    """``-exp(A_log) * softplus(a + dt_bias)`` in float32; ``A_log`` broadcasts per head."""
    return -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))


class LinearAttentionMixer(eqx.Module):
    """Projections, short convolutions, output norm/gate and output projection shared by all kinds."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    q_conv: ShortConv | None
    k_conv: ShortConv | None
    v_conv: ShortConv | None
    g_proj: Linear | LowRankLinear | None
    o_norm: GatedRMSNorm | RMSNorm
    o_proj: Linear
    cfg: LinearAttentionConfig = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def _init_common(
        self,
        kind: str,
        d_model: int,
        cfg: LinearAttentionConfig,
        key: Array,
        std: float,
        norm_eps: float,
        param_dtype: jnp.dtype,
        compute_dtype: str,
        *,
        gate_activation: str = "silu",
        low_rank_gate: int | None = None,
    ) -> None:
        kq, kk, kv, kg, ko = jax.random.split(key, 5)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        key_dim = cfg.num_k_heads * cfg.head_k_dim
        value_dim = cfg.num_v_heads * cfg.head_v_dim
        self.q_proj = Linear(d_model, key_dim, key=kq, axes=("embed", "heads"), **common)
        self.k_proj = Linear(d_model, key_dim, key=kk, axes=("embed", "heads"), **common)
        self.v_proj = Linear(d_model, value_dim, key=kv, axes=("embed", "heads"), **common)
        if cfg.conv_size > 0:
            conv = dict(activation=True, param_dtype=param_dtype, impl=cfg.conv_impl)
            kcq, kck, kcv = _conv_inits(kind, cfg, key, 5, 3, std)
            self.q_conv = ShortConv(key_dim, cfg.conv_size, **conv, **kcq)
            self.k_conv = ShortConv(key_dim, cfg.conv_size, **conv, **kck)
            self.v_conv = ShortConv(value_dim, cfg.conv_size, **conv, **kcv)
        else:
            self.q_conv = self.k_conv = self.v_conv = None
        if cfg.output_gate:
            if low_rank_gate is None:
                self.g_proj = Linear(d_model, value_dim, key=kg, axes=("embed", "heads"), **common)
            else:
                self.g_proj = LowRankLinear(d_model, low_rank_gate, value_dim, key=kg, out_axis="heads",
                                            use_bias=True, **common)
            self.o_norm = GatedRMSNorm(cfg.head_v_dim, eps=norm_eps, activation=gate_activation,
                                       param_dtype=param_dtype)
        else:
            self.g_proj = None
            self.o_norm = RMSNorm(cfg.head_v_dim, eps=norm_eps, param_dtype=param_dtype)
        self.o_proj = Linear(value_dim, d_model, key=ko, axes=("heads", "embed"), **common)
        self.cfg = cfg
        self.kind = kind

    # Kind-specific pieces ------------------------------------------------------------
    def gates(self, x: Array) -> dict[str, Array]:
        """The op's gates by name (``g``, ``beta``, ``b``, ``w``), float32, at value-head resolution."""
        raise NotImplementedError

    def _gate_input_projections(self) -> tuple:
        """Gate projections of ``x`` that :meth:`_forward` runs together with q, k, v and the output gate
        (one shared FP8 cast of ``x``, :func:`jla.layers.fp8.shared_input_projections`); their outputs are
        passed to :meth:`gates` after ``x``, in this order (none by default)."""
        return ()

    def _write_strength(self, logits: Array) -> Array:
        """``sigmoid(logits)``, in ``[0, 2]`` when ``allow_neg_eigval``."""
        strength = jax.nn.sigmoid(logits.astype(jnp.float32))
        return 2.0 * strength if self.cfg.allow_neg_eigval else strength

    # Shared forward ------------------------------------------------------------------
    def _activate(self, x: Array, conv: ShortConv | None, segment_ids, state) -> tuple[Array, Array | None]:
        if conv is not None:
            return conv(x, segment_ids=segment_ids, state=state)
        if self.kind in _DELTA_KINDS:  # FLA applies SiLU when the delta-rule mixers run without conv
            x = jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)
        return x, None

    def _heads(self, x: Array, heads: int) -> Array:
        return x.reshape(*x.shape[:2], heads, -1)

    def _repeat_k_heads(self, x: Array) -> Array:
        # Through float32 (same values), so the transpose sums the per-head gradients in float32.
        groups = self.cfg.num_v_heads // self.cfg.num_k_heads
        return x if groups == 1 else jnp.repeat(x.astype(jnp.float32), groups, axis=2).astype(x.dtype)

    def _forward(
        self,
        x: Array,
        *,
        segment_ids: Array | None,
        cache: RecurrentCache | None,
    ) -> tuple[Array, RecurrentCache | None]:
        cfg = self.cfg
        conv_state = (None, None, None) if cache is None or cache.conv is None else cache.conv
        from jla.layers.fp8 import shared_input_projections  # one FP8 cast of x for q, k, v, g, gates (training)

        q, k, v_in, g_in, *gate_in = shared_input_projections(x, self.q_proj, self.k_proj, self.v_proj, self.g_proj,
                                                               *self._gate_input_projections())
        normed = self._conv_l2norm(q, k, segment_ids, conv_state)  # None: conv here, L2 norm in the op
        if normed is None:
            q, cq = self._activate(q, self.q_conv, segment_ids, conv_state[0])
            k, ck = self._activate(k, self.k_conv, segment_ids, conv_state[1])
        else:
            (q, cq), (k, ck) = normed
        v, cv = self._activate(v_in, self.v_conv, segment_ids, conv_state[2])
        q = self._repeat_k_heads(self._heads(q, cfg.num_k_heads))
        k = self._repeat_k_heads(self._heads(k, cfg.num_k_heads))
        v = self._heads(v, cfg.num_v_heads)
        if cfg.qk_conv_barrier and cfg.qk_l2norm and self.kind in _DELTA_KINDS:
            # Without it, XLA fuses the q/k/v convolutions into the reduction of use_qk_l2norm (one slow
            # multi-output reduction kernel, ~10 ms per 1.3B layer under remat on an H100).
            q, k = jax.lax.optimization_barrier((q, k))
        # chunk_size is dropped by the dispatcher when the reference implementation runs.
        options: dict = {"chunk_size": cfg.chunk_size, **self.gates(x, *gate_in)}
        if self.kind in _DELTA_KINDS:
            options["use_qk_l2norm"] = cfg.qk_l2norm and normed is None
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if x.shape[1] == 1:
                impl = "reference"  # a single recurrent step; chunking would pad it to a full chunk
        if cfg.context_parallel_axis is not None and cache is None:
            from jla.distributed.context_parallel import context_parallel_linear_attention

            o, final = context_parallel_linear_attention(self.kind, q, k, v, axis=cfg.context_parallel_axis, **options)
        else:
            o, final = self._run_op(q, k, v, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        if self.g_proj is not None:
            o = self.o_norm(o, self._heads(g_in, cfg.num_v_heads))
        else:
            o = self.o_norm(o)
        y = self.o_proj(o.reshape(*o.shape[:2], -1))
        if cache is None:
            return y, None
        conv = None if self.q_conv is None else (cq, ck, cv)
        return y, RecurrentCache(state=final, conv=conv)

    def _run_op(self, q: Array, k: Array, v: Array, *, impl: str, **options) -> tuple[Array, Array | None]:
        """``ops.linear_attention`` on the q, k, v and options :meth:`_forward` prepared.

        ``jla.layers.lean`` (``ModelConfig.lean_remat_core``) overrides only this step, to rerun the op in
        the backward pass; the prologue above, the fused conv + L2 norm (``qk_l2norm_in_conv``) and
        ``use_qk_l2norm`` included, stays this method's caller's for every mixer.
        """
        return ops.linear_attention(self.kind, q, k, v, impl=impl, **options)

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(x, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        # Shapes come from the config only, so this also works on layer-stacked modules.
        cfg = self.cfg
        state = jnp.zeros((batch, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim), jnp.float32)
        conv = None
        if cfg.conv_size > 0:
            key_dim, value_dim = cfg.num_k_heads * cfg.head_k_dim, cfg.num_v_heads * cfg.head_v_dim
            conv = tuple(jnp.zeros((batch, cfg.conv_size - 1, c), dtype) for c in (key_dim, key_dim, value_dim))
        return RecurrentCache(state=state, conv=conv)

    def step(self, x: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``x [B, T, D]`` from ``cache``; returns the output and the new cache."""
        return self._forward(x, segment_ids=None, cache=cache)

    def _conv_l2norm(self, q: Array, k: Array, segment_ids, conv_state):
        """``((q, q_state), (k, k_state))`` with q and k L2-normalized per head inside the short-conv kernel
        (:mod:`jla.layers.conv_l2norm`), or ``None`` when that does not run for both (unfused path)."""
        cfg = self.cfg
        if not (cfg.qk_l2norm and cfg.qk_l2norm_in_conv and self.kind in _DELTA_KINDS and self.q_conv is not None):
            return None
        from jla.layers.conv_l2norm import conv_l2norm, conv_l2norm_reason

        pairs = ((self.q_conv, q, conv_state[0]), (self.k_conv, k, conv_state[1]))
        if any(conv_l2norm_reason(conv, h, cfg.head_k_dim, segment_ids=segment_ids, state=s) is not None
               for conv, h, s in pairs):
            return None
        normed = tuple(conv_l2norm(conv, h, cfg.head_k_dim, segment_ids=segment_ids, state=s) for conv, h, s in pairs)
        return None if any(pair is None for pair in normed) else normed  # None: no mesh split fits the kernels


class GatedDeltaNet(LinearAttentionMixer):
    """Gated DeltaNet (Yang et al. 2024; the linear layer of Qwen3-Next)."""

    a_proj: Linear
    b_proj: Linear
    A_log: Array
    dt_bias: Array

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, ka, kb, kA, kdt = jax.random.split(key, 5)
        self._init_common("gated_deltanet", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.a_proj = Linear(d_model, cfg.num_v_heads, key=ka, axes=("embed", None), **common)
        self.b_proj = Linear(d_model, cfg.num_v_heads, key=kb, axes=("embed", None), **common)
        self.A_log = _a_log_init(kA, cfg.num_v_heads, 0.0).astype(param_dtype)
        self.dt_bias = _dt_bias_init(kdt, cfg.num_v_heads).astype(param_dtype)

    def gates(self, x):
        g = _mamba_gate(self.a_proj(x), self.A_log, self.dt_bias)
        return {"g": g, "beta": self._write_strength(self.b_proj(x))}

    def param_axes(self):
        return {"A_log": (None,), "dt_bias": (None,)}


class KDA(LinearAttentionMixer):
    """Kimi Delta Attention (Kimi Linear): per-channel Mamba-style decay with the delta rule."""

    f_proj: LowRankLinear
    b_proj: Linear
    A_log: Array
    dt_bias: Array

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, kf, kb, kA, kdt = jax.random.split(key, 5)
        rank = cfg.gate_low_rank_dim or cfg.head_v_dim
        self._init_common("kda", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype,
                          gate_activation="sigmoid", low_rank_gate=rank)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        gate_dim = cfg.num_v_heads * cfg.head_k_dim
        self.f_proj = LowRankLinear(d_model, rank, gate_dim, key=kf, out_axis="heads", **common)
        self.b_proj = Linear(d_model, cfg.num_v_heads, key=kb, axes=("embed", None), **common)
        self.A_log = _a_log_init(kA, cfg.num_v_heads, 1.0).astype(param_dtype)
        self.dt_bias = _dt_bias_init(kdt, gate_dim).astype(param_dtype)

    def gates(self, x):
        cfg = self.cfg
        f = self._heads(self.f_proj(x), cfg.num_v_heads)  # [B, T, Hv, Dk]
        g = _mamba_gate(f, self.A_log[:, None], self.dt_bias.reshape(cfg.num_v_heads, cfg.head_k_dim))
        return {"g": g, "beta": self._write_strength(self.b_proj(x))}

    def param_axes(self):
        return {"A_log": (None,), "dt_bias": ("heads",)}


class DeltaNet(LinearAttentionMixer):
    """DeltaNet (Schlag et al. 2021; Yang et al. 2024): the delta rule without decay."""

    b_proj: Linear

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, kb = jax.random.split(key)
        self._init_common("deltanet", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        self.b_proj = Linear(d_model, cfg.num_v_heads, key=kb, axes=("embed", None), std=std,
                             param_dtype=param_dtype, compute_dtype=compute_dtype)

    def gates(self, x):
        return {"beta": self._write_strength(self.b_proj(x))}


class GLA(LinearAttentionMixer):
    """Gated linear attention (Yang et al. 2023) with a low-rank per-channel forget gate."""

    gk_proj: LowRankLinear

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, kg = jax.random.split(key)
        self._init_common("gla", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        rank = cfg.gate_low_rank_dim or 16
        self.gk_proj = LowRankLinear(d_model, rank, cfg.num_k_heads * cfg.head_k_dim, key=kg,
                                     out_axis="heads", use_bias=True, std=std,
                                     param_dtype=param_dtype, compute_dtype=compute_dtype)

    def gates(self, x):
        gk = self._heads(self.gk_proj(x), self.cfg.num_k_heads).astype(jnp.float32)
        g = jax.nn.log_sigmoid(gk) / self.cfg.gate_logit_normalizer
        return {"g": self._repeat_k_heads(g)}


class SimpleGLA(LinearAttentionMixer):
    """Linear attention with one data-dependent log-decay per head (FLA ``SimpleGatedLinearAttention``)."""

    gk_proj: Linear

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, kg = jax.random.split(key)
        self._init_common("simple_gla", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        self.gk_proj = Linear(d_model, cfg.num_v_heads, key=kg, axes=("embed", None), use_bias=True,
                              std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)

    def gates(self, x):
        g = jax.nn.log_sigmoid(self.gk_proj(x).astype(jnp.float32)) / self.cfg.gate_logit_normalizer
        return {"g": g}


class GatedDeltaNet2(LinearAttentionMixer):
    """Gated DeltaNet-2 (Hatamizadeh et al. 2026, arXiv 2605.22791), from the paper's equations.

    Gated Delta Rule-2 (Eq. 10): ``S_t = (I - k_t (b_t * k_t)^T) diag(exp g_t) S_{t-1}
    + k_t (w_t * v_t)^T``, read as ``o_t = S_t^T q_t``. Gates (Eq. 11-12, Appendix C.1):
    ``b = sigmoid(x W_b)`` over ``Hk * Dk`` key channels, ``w = sigmoid(x W_w)`` over
    ``Hv * Dv`` value channels and ``g = -exp(a) * softplus(x W_f + delta)`` with ``a``
    per key head and ``delta`` per key channel (``W_f`` low rank, as in KDA). Block
    (Fig. 1): q, k through linear, short conv, SiLU and L2 norm; v through linear,
    short conv and SiLU; output RMSNorm times ``silu(x W_g)``, then the out projection.
    """

    f_proj: LowRankLinear
    b_proj: Linear
    w_proj: Linear
    A_log: Array
    dt_bias: Array

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        if not cfg.output_gate:
            raise ValueError("GatedDeltaNet2 always gates its output norm (output_gate=True)")
        k0, kf, kb, kw, kA, kdt = jax.random.split(key, 6)
        self._init_common("gdn2", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        key_dim = cfg.num_k_heads * cfg.head_k_dim
        rank = cfg.gate_low_rank_dim or cfg.head_v_dim
        self.f_proj = LowRankLinear(d_model, rank, key_dim, key=kf, out_axis="heads", **common)
        self.b_proj = Linear(d_model, key_dim, key=kb, axes=("embed", "heads"), **common)
        self.w_proj = Linear(d_model, cfg.num_v_heads * cfg.head_v_dim, key=kw, axes=("embed", "heads"), **common)
        self.A_log = _a_log_init(kA, cfg.num_k_heads, 0.0).astype(param_dtype)
        self.dt_bias = _dt_bias_init(kdt, key_dim).astype(param_dtype)

    def _gate_input_projections(self):
        # The erase and write gate projections read the mixer input like q, k, v and the output gate: under
        # delayed-scaling FP8 they share its one e4m3 copy (bitwise the separate casts, see shared_input_projections).
        return (self.b_proj, self.w_proj)

    def _decay(self, f: Array) -> Array:
        """``g = -exp(a) * softplus(f + delta)`` float32 from the low-rank projection ``f [B, T, Hk, Dk]``."""
        cfg = self.cfg
        return _mamba_gate(f, self.A_log[:, None], self.dt_bias.reshape(cfg.num_k_heads, cfg.head_k_dim))

    def gates(self, x, b_logits: Array | None = None, w_logits: Array | None = None):
        """The gates of ``x``; ``b_logits`` / ``w_logits``: ``x W_b`` / ``x W_w`` when :meth:`_forward` ran them."""
        cfg = self.cfg
        b_logits = self.b_proj(x) if b_logits is None else b_logits
        w_logits = self.w_proj(x) if w_logits is None else w_logits
        g = self._decay(self._heads(self.f_proj(x), cfg.num_k_heads))  # [B, T, Hk, Dk]
        b = self._write_strength(self._heads(b_logits, cfg.num_k_heads))
        w = jax.nn.sigmoid(self._heads(w_logits, cfg.num_v_heads).astype(jnp.float32))
        return {"g": self._repeat_k_heads(g), "b": self._repeat_k_heads(b), "w": w}

    def param_axes(self):
        return {"A_log": (None,), "dt_bias": ("heads",)}


_MIXERS = {
    "gated_deltanet": GatedDeltaNet,
    "kda": KDA,
    "deltanet": DeltaNet,
    "gla": GLA,
    "simple_gla": SimpleGLA,
    "gdn2": GatedDeltaNet2,
}


def make_mixer(kind: str, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float, norm_eps: float,
               param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16") -> LinearAttentionMixer:
    """Build the linear-attention mixer for ``kind``."""
    if kind not in _MIXERS:
        raise ValueError(f"unknown linear-attention kind {kind!r}; expected one of {sorted(_MIXERS)}")
    return _MIXERS[kind](d_model, cfg, key=key, std=std, norm_eps=norm_eps, param_dtype=param_dtype,
                         compute_dtype=compute_dtype)


# mLSTM: the xLSTM-7B layer (arXiv 2503.13427); the cell is jla.ops "mlstm".
# Adapted from NX-AI/xlstm (Apache-2.0), xlstm/xlstm_large/model.py (mLSTMLayer layout).
class MLSTM(LinearAttentionMixer):
    """xLSTM-7B mLSTM layer.

    ``q, k [B, T, H, Dk]`` and ``v [B, T, H, Dv]`` come from plain projections (no conv
    in the xLSTM-7B layout, ``conv_size=0``); the input and forget gate pre-activations
    are per-head projections with bias, soft-capped at ``cfg.gate_soft_cap``
    (``cap * tanh(x / cap)``); the cell uses ``g = logsigmoid(f)`` and the exponential
    input gate ``exp(i)``. The output is normalised per head and multiplied by the
    sigmoid output gate ``sigmoid(x W_o)``, then projected. Gate biases: forget gate
    linspace(3, 6) over heads (as xlstm's mLSTM cell), input gate -10 (arXiv 2503.13427).
    Per-head RMSNorm stands in for xLSTM-7B's per-head LayerNorm.
    """

    i_proj: Linear
    f_proj: Linear

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        if not cfg.output_gate:
            raise ValueError("mlstm needs output_gate=True (the xLSTM sigmoid output gate)")
        k0, ki, kf = jax.random.split(key, 3)
        self._init_common("mlstm", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype,
                          gate_activation="sigmoid")
        heads = cfg.num_v_heads
        common = dict(axes=("embed", None), use_bias=True, std=std, param_dtype=param_dtype,
                      compute_dtype=compute_dtype)
        i_proj = Linear(d_model, heads, key=ki, **common)
        f_proj = Linear(d_model, heads, key=kf, **common)
        self.i_proj = eqx.tree_at(lambda lin: lin.bias, i_proj, jnp.full((heads,), -10.0, param_dtype))
        self.f_proj = eqx.tree_at(lambda lin: lin.bias, f_proj, jnp.linspace(3.0, 6.0, heads).astype(param_dtype))

    def _soft_cap(self, x: Array) -> Array:
        x = x.astype(jnp.float32)
        cap = self.cfg.gate_soft_cap
        return x if cap is None else cap * jnp.tanh(x / cap)

    def gates(self, x):
        return {"g": jax.nn.log_sigmoid(self._soft_cap(self.f_proj(x))), "i": self._soft_cap(self.i_proj(x))}

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg = self.cfg
        state = ops.reference.mlstm_empty_state(batch, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim)
        return RecurrentCache(state=state, conv=super().init_cache(batch, dtype).conv)


_MIXERS["mlstm"] = MLSTM
__all__ += ["MLSTM"]


# Gated DeltaProduct, adapted from flash-linear-attention (MIT), fla/layers/gated_deltaproduct.py.


class GatedDeltaProduct(GatedDeltaNet):
    """Gated DeltaProduct (Siems et al. 2025; FLA ``GatedDeltaProduct``).

    ``cfg.num_householder`` (``n_h``) delta-rule (Householder) steps per token
    after one decay: k, v and beta are ``n_h`` times as wide as in Gated
    DeltaNet and split per token into ``n_h`` consecutive sub-steps (FLA's
    ``(t n)`` layout); the decay and q stay per token. ``n_h = 1`` has Gated
    DeltaNet's parameters and output.
    """

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        k0, kk, kv, kb = jax.random.split(key, 4)
        super().__init__(d_model, cfg, key=k0, std=std, norm_eps=norm_eps, param_dtype=param_dtype,
                         compute_dtype=compute_dtype)
        n = cfg.num_householder
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        key_dim, value_dim = cfg.num_k_heads * cfg.head_k_dim, cfg.num_v_heads * cfg.head_v_dim
        self.k_proj = Linear(d_model, key_dim * n, key=kk, axes=("embed", "heads"), **common)
        self.v_proj = Linear(d_model, value_dim * n, key=kv, axes=("embed", "heads"), **common)
        self.b_proj = Linear(d_model, cfg.num_v_heads * n, key=kb, axes=("embed", None), **common)
        if cfg.conv_size > 0:
            conv = dict(activation=True, param_dtype=param_dtype, impl=cfg.conv_impl)
            kck, kcv = _conv_inits("gated_deltaproduct", cfg, key, 4, 2, std)
            self.k_conv = ShortConv(key_dim * n, cfg.conv_size, **conv, **kck)
            self.v_conv = ShortConv(value_dim * n, cfg.conv_size, **conv, **kcv)
        self.kind = "gated_deltaproduct"

    def _forward(
        self,
        x: Array,
        *,
        segment_ids: Array | None,
        cache: RecurrentCache | None,
    ) -> tuple[Array, RecurrentCache | None]:
        cfg, n = self.cfg, self.cfg.num_householder
        batch, seq_len = x.shape[:2]
        conv_state = (None, None, None) if cache is None or cache.conv is None else cache.conv

        def activate(h, conv, state):
            if conv is not None:
                return conv(h, segment_ids=segment_ids, state=state)
            return jax.nn.silu(h.astype(jnp.float32)).astype(h.dtype), None  # FLA's SiLU without conv

        q, cq = activate(self.q_proj(x), self.q_conv, conv_state[0])
        k, ck = activate(self.k_proj(x), self.k_conv, conv_state[1])
        v, cv = activate(self.v_proj(x), self.v_conv, conv_state[2])
        q = self._repeat_k_heads(self._heads(q, cfg.num_k_heads))
        k = self._repeat_k_heads(k.reshape(batch, seq_len * n, cfg.num_k_heads, cfg.head_k_dim))
        v = v.reshape(batch, seq_len * n, cfg.num_v_heads, cfg.head_v_dim)
        gates = self.gates(x)
        gates["beta"] = gates["beta"].reshape(batch, seq_len * n, cfg.num_v_heads)
        options: dict = {"chunk_size": cfg.chunk_size, "use_qk_l2norm": cfg.qk_l2norm, "num_householder": n,
                         **gates}
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if seq_len == 1:
                impl = "reference"  # a single recurrent step; chunking would pad it to a full chunk
        if cfg.context_parallel_axis is not None and cache is None:  # raises for kinds CP does not support
            from jla.distributed.context_parallel import context_parallel_linear_attention

            o, final = context_parallel_linear_attention(self.kind, q, k, v, axis=cfg.context_parallel_axis, **options)
        else:
            o, final = ops.linear_attention(self.kind, q, k, v, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        if self.g_proj is not None:
            o = self.o_norm(o, self._heads(self.g_proj(x), cfg.num_v_heads))
        else:
            o = self.o_norm(o)
        y = self.o_proj(o.reshape(*o.shape[:2], -1))
        if cache is None:
            return y, None
        conv = None if self.q_conv is None else (cq, ck, cv)
        return y, RecurrentCache(state=final, conv=conv)

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg, n = self.cfg, self.cfg.num_householder
        state = jnp.zeros((batch, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim), jnp.float32)
        conv = None
        if cfg.conv_size > 0:
            key_dim, value_dim = cfg.num_k_heads * cfg.head_k_dim, cfg.num_v_heads * cfg.head_v_dim
            conv = tuple(jnp.zeros((batch, cfg.conv_size - 1, c), dtype)
                         for c in (key_dim, key_dim * n, value_dim * n))
        return RecurrentCache(state=state, conv=conv)


_MIXERS["gated_deltaproduct"] = GatedDeltaProduct


# RWKV-6 and HGRN2 mixers live in their own module.
from jla.layers.rwkv6 import HGRN2, RWKV6  # noqa: E402

_MIXERS.update({"rwkv6": RWKV6, "hgrn2": HGRN2})


# Kinds whose q/k/v short convs start at N(0, initializer_range) under ModelConfig.init_scheme="upstream"
# (when conv_init is None): FLA's _init_weights (fla/models/modeling_utils.py) re-initializes every
# nn.Conv1d, ShortConvolution included, with normal(0, initializer_range) for its GatedDeltaNet, DeltaNet,
# GatedDeltaProduct and KDA models. GLA, simple GLA and mLSTM keep the identity init; GDN-2 keeps it too,
# since its paper (arXiv 2605.22791, D.5) states the init of the linear layers only.
UPSTREAM_NORMAL_CONV_KINDS = ("gated_deltanet", "deltanet", "gated_deltaproduct", "kda")


def _conv_inits(kind: str, cfg: LinearAttentionConfig, key: Array, salt: int, count: int,
                std: float) -> list[dict]:
    """``ShortConv`` init arguments for ``count`` convs of a ``kind`` mixer (``cfg.conv_init``; None = identity).

    The normal init draws from ``fold_in(key, salt)``, a key no projection uses, so every other
    parameter of the mixer keeps the values it had with the identity init.
    """
    init = cfg.conv_init or "identity"
    if init == "identity":
        return [dict(init="identity") for _ in range(count)]
    keys = jax.random.split(jax.random.fold_in(key, salt), count)
    return [dict(init="normal", key=k, std=std) for k in keys]
