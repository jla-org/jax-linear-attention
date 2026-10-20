# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Memory-lean custom VJPs for the elementwise layers between the matmuls (``ModelConfig.lean_vjp``).

Plain autodiff of the float32 elementwise chains of a block keeps their float32 intermediates for the
backward pass: at 1.3B and 4096 tokens a Gated DeltaNet block saves about 1 GiB per sequence in its three
short convolutions alone (every tap's float32 input, the masked taps, the pre-activation and the SiLU),
about 0.5 GiB in SwiGLU and 0.16 GiB in its two RMSNorms (``benchmarks/fp8_1p3b.md``). With
``remat = "none"`` that is what keeps a useful micro-batch from fitting. The functions here compute the
same forward values in the same float32 order and save only their inputs, in their own dtype, plus
per-row statistics; the backward pass recomputes the rest:

* :func:`rms_norm`: saves ``x`` and ``rstd [..., 1]`` (float32);
* :func:`gated_rms_norm`: saves ``x``, the gate input ``z`` and ``rstd [..., groups, 1]``;
* :func:`swiglu_linear`: SwiGLU followed by the down projection; saves the gate/up pre-activations
  ``[..., 2, F]`` (compute dtype) and the cast down weight, so neither the float32 gate chain nor the
  ``[..., F]`` product reaches the backward pass;
* :func:`short_conv`: the depthwise causal conv + SiLU; saves the (state-extended) input and the
  document masks and recomputes the pre-activation in the backward pass. It runs only where
  :class:`ShortConv` itself would run XLA (``conv_impl = "xla"``, or ``"auto"`` where the kernels decline):
  ShortConv's Mosaic GPU kernels and the delta-rule mixers' fused conv + SiLU + per-head L2 norm
  (``LinearAttentionConfig.qk_l2norm_in_conv``, :mod:`jla.layers.conv_l2norm`) are custom VJPs that already
  save only their inputs, and they are faster, so lean keeps them (no second conv or L2-norm VJP);
* :func:`mamba_gate`: ``-exp(A_log) * softplus(a + dt_bias)``; saves ``a`` (compute dtype) only;
* :func:`skip_gate`: Mamba-3's ``(y + D x) * silu(z)`` output epilogue (Mamba-2's ``y + D x``); saves ``y``,
  ``x`` and ``z``;
* :func:`dt_scale`: Mamba-2's SSD input ``dt * x``; saves ``x`` and ``dt``.

:func:`apply_lean` swaps a model's :class:`~jla.layers.norm.RMSNorm`, :class:`~jla.layers.norm.GatedRMSNorm`,
:class:`~jla.layers.mlp.SwiGLU`, :class:`~jla.layers.conv.ShortConv`, :class:`~jla.layers.linear_attention.KDA`,
:class:`~jla.layers.linear_attention.GatedDeltaNet2`, :class:`~jla.layers.ssm.Mamba2` and
:class:`~jla.layers.ssm.Mamba3` modules (exact types only, so subclasses keep their own forward) for subclasses
that call these functions; parameters, their tree paths and the forward values are unchanged. Gradients agree with plain autodiff to float32 rounding (the same formulas, evaluated in a
different order), see ``tests/test_lean_vjp.py``.

These are XLA-level VJPs. Fused Pallas norm-prologue / activation-epilogue GEMMs with their own VJPs
can replace them layer by layer: a module that a fused GEMM owns is simply not swapped.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.conv import ShortConv
from jla.layers.linear_attention import KDA, GatedDeltaNet, GatedDeltaNet2
from jla.layers.mlp import SwiGLU
from jla.layers.norm import GatedRMSNorm, RMSNorm
from jla.layers.ssm import Mamba2, Mamba3, _grouped_simple_gla, _repeat_groups, _split_rows, heavy_tail, mamba3_siso

__all__ = [
    "LeanGatedDeltaNet", "LeanGatedDeltaNet2", "LeanGatedRMSNorm", "LeanKDA", "LeanMamba2", "LeanMamba3", "LeanRMSNorm", "LeanShortConv",
    "LeanSwiGLU", "apply_lean", "dt_scale", "gated_rms_norm", "mamba2_ssd", "mamba_gate", "rms_norm", "short_conv",
    "skip_gate", "swiglu_linear",
]

Array = jax.Array
f32 = jnp.float32


def _sum_to(x: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``x`` over the axes that broadcasting added or expanded to reach it from ``shape``."""
    lead = x.ndim - len(shape)
    x = jnp.sum(x, axis=tuple(range(lead))) if lead else x
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and x.shape[i] != 1)
    return jnp.sum(x, axis=axes, keepdims=True) if axes else x


def _dsilu(z: Array, s: Array) -> Array:
    """``d silu(z) / dz`` from ``z`` and ``s = sigmoid(z)``."""
    return s * (1.0 + z * (1.0 - s))


# RMSNorm ---------------------------------------------------------------------------------------
def _rms_forward(x: Array, weight: Array | None, eps: float) -> tuple[Array, Array]:
    x32 = x.astype(f32)
    rstd = lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    y = x32 * rstd
    if weight is not None:
        y = y * weight.astype(f32)
    return y.astype(x.dtype), rstd


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rms_norm(x: Array, weight: Array | None, eps: float) -> Array:
    return _rms_forward(x, weight, eps)[0]


def _rms_norm_fwd(x, weight, eps):
    y, rstd = _rms_forward(x, weight, eps)
    return y, (x, weight, rstd)


def _rms_norm_bwd(eps, res, g):
    x, weight, rstd = res
    xhat = x.astype(f32) * rstd
    g32 = g.astype(f32)
    gw = g32 if weight is None else g32 * weight.astype(f32)
    dx = rstd * (gw - xhat * jnp.mean(gw * xhat, axis=-1, keepdims=True))
    dw = None if weight is None else _sum_to(g32 * xhat, weight.shape).astype(weight.dtype)
    return dx.astype(x.dtype), dw


_rms_norm.defvjp(_rms_norm_fwd, _rms_norm_bwd)


def rms_norm(x: Array, weight: Array | None, eps: float) -> Array:
    """:func:`jla.layers.norm.rms_norm` saving only ``x`` and ``rstd`` for the backward pass."""
    return _rms_norm(x, weight, float(eps))


# Gated RMSNorm ---------------------------------------------------------------------------------
def _gate(z32: Array, activation: str) -> tuple[Array, Array]:
    """``(act(z), d act / dz)`` in float32."""
    s = jax.nn.sigmoid(z32)
    if activation == "silu":
        return z32 * s, _dsilu(z32, s)
    return s, s * (1.0 - s)


def _gated_forward(x, z, weight, eps, activation, norm_before_gate, group):
    x32, z32 = x.astype(f32), z.astype(f32)
    gate = jax.nn.silu(z32) if activation == "silu" else jax.nn.sigmoid(z32)
    u = x32 if norm_before_gate else x32 * gate
    grouped = u.reshape(*x.shape[:-1], x.shape[-1] // group, group)
    rstd = lax.rsqrt(jnp.mean(grouped * grouped, axis=-1, keepdims=True) + eps)
    y = (grouped * rstd).reshape(x.shape) * weight.astype(f32)
    if norm_before_gate:
        y = y * gate
    return y.astype(x.dtype), rstd


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _gated_rms_norm(x, z, weight, eps, activation, norm_before_gate, group):
    return _gated_forward(x, z, weight, eps, activation, norm_before_gate, group)[0]


def _gated_rms_norm_fwd(x, z, weight, eps, activation, norm_before_gate, group):
    y, rstd = _gated_forward(x, z, weight, eps, activation, norm_before_gate, group)
    return y, (x, z, weight, rstd)


def _gated_rms_norm_bwd(eps, activation, norm_before_gate, group, res, g):
    x, z, weight, rstd = res
    shape = x.shape
    grouped_shape = (*shape[:-1], shape[-1] // group, group)
    x32, z32, g32, w32 = x.astype(f32), z.astype(f32), g.astype(f32), weight.astype(f32)
    gate, dgate = _gate(z32, activation)
    u = x32 if norm_before_gate else x32 * gate
    uhat = (u.reshape(grouped_shape) * rstd).reshape(shape)
    if norm_before_gate:  # y = uhat * w * act(z)
        dw = _sum_to(g32 * gate * uhat, weight.shape)
        dz = g32 * uhat * w32 * dgate
        gu = g32 * gate * w32
    else:  # y = norm(x * act(z)) * w
        dw = _sum_to(g32 * uhat, weight.shape)
        gu = g32 * w32
    gug, uhg = gu.reshape(grouped_shape), uhat.reshape(grouped_shape)
    du = (rstd * (gug - uhg * jnp.mean(gug * uhg, axis=-1, keepdims=True))).reshape(shape)
    if norm_before_gate:
        dx = du
    else:
        dx = du * gate
        dz = du * x32 * dgate
    return dx.astype(x.dtype), dz.astype(z.dtype), dw.astype(weight.dtype)


_gated_rms_norm.defvjp(_gated_rms_norm_fwd, _gated_rms_norm_bwd)


def gated_rms_norm(x: Array, z: Array, weight: Array, eps: float, *, activation: str = "silu",
                   norm_before_gate: bool = True, group_size: int | None = None) -> Array:
    """:class:`jla.layers.norm.GatedRMSNorm`'s forward saving ``x``, ``z`` and ``rstd`` only."""
    if z.shape != x.shape:
        raise ValueError(f"gate shape {z.shape} must match input shape {x.shape}")
    if activation not in ("silu", "sigmoid"):
        raise ValueError(f"unsupported gate activation {activation!r}")
    group = group_size or x.shape[-1]
    if x.shape[-1] % group:
        raise ValueError(f"group_size {group} must divide the last axis {x.shape[-1]}")
    return _gated_rms_norm(x, z, weight, float(eps), activation, bool(norm_before_gate), int(group))


# SwiGLU + down projection ----------------------------------------------------------------------
def _swiglu_product(h: Array) -> Array:
    gate, up = h[..., 0, :].astype(f32), h[..., 1, :].astype(f32)
    return (jax.nn.silu(gate) * up).astype(h.dtype)


def _matmul(a: Array, w: Array) -> Array:
    return lax.dot_general(a, w, (((a.ndim - 1,), (0,)), ((), ())))


@jax.custom_vjp
def _swiglu_linear(h: Array, w: Array) -> Array:
    return _matmul(_swiglu_product(h), w)


def _swiglu_linear_fwd(h, w):
    return _matmul(_swiglu_product(h), w), (h, w)


def _swiglu_linear_bwd(res, g):
    h, w = res
    p = _swiglu_product(h)  # recomputed: [..., F] in the compute dtype, as the forward fed the GEMM
    lead = p.ndim - 1
    dw = lax.dot_general(p, g, ((tuple(range(lead)), tuple(range(lead))), ((), ())))  # [F, D]
    dp = lax.dot_general(g, w, (((g.ndim - 1,), (1,)), ((), ()))).astype(f32)  # [..., F]
    gate, up = h[..., 0, :].astype(f32), h[..., 1, :].astype(f32)
    s = jax.nn.sigmoid(gate)
    dgate = dp * up * _dsilu(gate, s)
    dup = dp * (gate * s)
    dh = jnp.stack([dgate, dup], axis=-2).astype(h.dtype)
    return dh, dw.astype(w.dtype)


_swiglu_linear.defvjp(_swiglu_linear_fwd, _swiglu_linear_bwd)


def swiglu_linear(h: Array, w: Array) -> Array:
    """``silu(h[..., 0, :]) * h[..., 1, :] @ w`` (``h [..., 2, F]``, ``w [F, D]``, one dtype).

    The same values as :func:`jla.layers.mlp.swiglu` followed by the down projection; the backward pass
    recomputes the product from ``h`` instead of keeping it and the float32 gate chain.
    """
    if h.ndim < 2 or h.shape[-2] != 2 or w.ndim != 2 or w.shape[0] != h.shape[-1]:
        raise ValueError(f"expected h [..., 2, F] and w [F, D]; got {h.shape} and {w.shape}")
    if h.dtype != w.dtype:
        raise ValueError(f"h and w must share a dtype; got {h.dtype} and {w.dtype}")
    return _swiglu_linear(h, w)


# Short causal convolution ----------------------------------------------------------------------
def _conv_pre(x_ext: Array, weight: Array, bias: Array | None, masks: Array | None) -> Array:
    """Float32 pre-activation, accumulated tap by tap in :class:`ShortConv`'s order."""
    width = weight.shape[1]
    seq_len = x_ext.shape[1] - (width - 1)
    w = weight.astype(f32)
    y = jnp.zeros((x_ext.shape[0], seq_len, x_ext.shape[2]), f32)
    if bias is not None:
        y = y + bias.astype(f32)
    for i in range(width):
        tap = x_ext[:, i:i + seq_len].astype(f32) * w[:, i]
        if masks is not None and i < width - 1:
            tap = jnp.where(masks[i][..., None], tap, 0.0)
        y = y + tap
    return y


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _short_conv(x_ext, weight, bias, masks, activation):
    y = _conv_pre(x_ext, weight, bias, masks)
    return (jax.nn.silu(y) if activation else y).astype(x_ext.dtype)


def _short_conv_fwd(x_ext, weight, bias, masks, activation):
    return _short_conv(x_ext, weight, bias, masks, activation), (x_ext, weight, bias, masks)


def _short_conv_bwd(activation, res, g):
    x_ext, weight, bias, masks = res
    width = weight.shape[1]
    seq_len = x_ext.shape[1] - (width - 1)
    g32 = g.astype(f32)
    if activation:
        pre = _conv_pre(x_ext, weight, bias, masks)
        g32 = g32 * _dsilu(pre, jax.nn.sigmoid(pre))
    w = weight.astype(f32)
    dx = jnp.zeros(x_ext.shape, f32)
    dws = []
    for i in range(width):
        gi = g32 if masks is None or i == width - 1 else jnp.where(masks[i][..., None], g32, 0.0)
        dx = dx + jnp.pad(gi * w[:, i], ((0, 0), (i, width - 1 - i), (0, 0)))
        dws.append(jnp.sum(gi * x_ext[:, i:i + seq_len].astype(f32), axis=(0, 1)))
    dw = jnp.stack(dws, axis=1).astype(weight.dtype)
    db = None if bias is None else jnp.sum(g32, axis=(0, 1)).astype(bias.dtype)
    return dx.astype(x_ext.dtype), dw, db, None


_short_conv.defvjp(_short_conv_fwd, _short_conv_bwd)


def short_conv(x_ext: Array, weight: Array, bias: Array | None, masks: Array | None, *,
               activation: bool) -> Array:
    """Depthwise causal conv of ``x_ext [B, K - 1 + T, C]`` (the ``K - 1`` state tokens, then ``x``).

    ``weight [C, K]``, ``bias [C]`` or ``None``; ``masks [K - 1, B, T]`` (bool) lets tap ``i`` read
    a token only where it is ``True`` (same document), or ``None``. Returns ``[B, T, C]`` in
    ``x_ext.dtype``; the backward pass saves only the arguments.
    """
    return _short_conv(x_ext, weight, bias, masks, bool(activation))


# Mamba-style decay gate ------------------------------------------------------------------------
@jax.custom_vjp
def _mamba_gate(a: Array, a_log: Array, dt_bias: Array) -> Array:
    return -jnp.exp(a_log.astype(f32)) * jax.nn.softplus(a.astype(f32) + dt_bias.astype(f32))


def _mamba_gate_fwd(a, a_log, dt_bias):
    return _mamba_gate(a, a_log, dt_bias), (a, a_log, dt_bias)


def _mamba_gate_bwd(res, g):
    a, a_log, dt_bias = res
    z = a.astype(f32) + dt_bias.astype(f32)
    neg_a = -jnp.exp(a_log.astype(f32))
    g32 = g.astype(f32)
    dz = g32 * neg_a * jax.nn.sigmoid(z)
    d_a_log = _sum_to(g32 * neg_a * jax.nn.softplus(z), a_log.shape)
    return dz.astype(a.dtype), d_a_log.astype(a_log.dtype), _sum_to(dz, dt_bias.shape).astype(dt_bias.dtype)


_mamba_gate.defvjp(_mamba_gate_fwd, _mamba_gate_bwd)


def mamba_gate(a: Array, a_log: Array, dt_bias: Array) -> Array:
    """``-exp(A_log) * softplus(a + dt_bias)`` in float32, saving ``a`` (its own dtype) and the parameters."""
    return _mamba_gate(a, a_log, dt_bias)


# Mamba-3 output epilogue -----------------------------------------------------------------------
def _skip_gate_forward(y, x, z, d):
    out = y.astype(f32) + d.astype(f32)[:, None] * x.astype(f32)
    if z is not None:
        out = out * jax.nn.silu(z.astype(f32))
    return out.astype(x.dtype)


@jax.custom_vjp
def _skip_gate(y, x, z, d):
    return _skip_gate_forward(y, x, z, d)


def _skip_gate_fwd(y, x, z, d):
    return _skip_gate_forward(y, x, z, d), (y, x, z, d)


def _skip_gate_bwd(res, g):
    y, x, z, d = res
    g32, x32, d32 = g.astype(f32), x.astype(f32), d.astype(f32)[:, None]
    dz = None
    du = g32
    if z is not None:
        z32 = z.astype(f32)
        s = jax.nn.sigmoid(z32)
        dz = (g32 * (y.astype(f32) + d32 * x32) * _dsilu(z32, s)).astype(z.dtype)
        du = g32 * (z32 * s)
    dd = _sum_to(du * x32, (d.shape[0], 1)).reshape(d.shape)
    return du.astype(y.dtype), (du * d32).astype(x.dtype), dz, dd.astype(d.dtype)


_skip_gate.defvjp(_skip_gate_fwd, _skip_gate_bwd)


def skip_gate(y: Array, x: Array, z: Array | None, d: Array) -> Array:
    """Mamba-3's output epilogue ``(y + D x) * silu(z)`` (``z=None``: no gate), returned in ``x.dtype``.

    ``y, x, z [B, T, H, P]``, ``d [H]``. The same float32 math as the end of
    ``jla.ops.mosaic_gpu.mamba3_fused.mamba3_siso_fused``; saves its inputs instead of their float32 copies.
    """
    return _skip_gate(y, x, z, d)


# Mamba-2 input scaling -------------------------------------------------------------------------
@jax.custom_vjp
def _dt_scale(x, dt):
    return (x.astype(f32) * dt.astype(f32)[..., None]).astype(x.dtype)


def _dt_scale_fwd(x, dt):
    return _dt_scale(x, dt), (x, dt)


def _dt_scale_bwd(res, g):
    x, dt = res
    g32 = g.astype(f32)
    dx = (g32 * dt.astype(f32)[..., None]).astype(x.dtype)
    return dx, jnp.sum(g32 * x.astype(f32), axis=-1).astype(dt.dtype)


_dt_scale.defvjp(_dt_scale_fwd, _dt_scale_bwd)


def dt_scale(x: Array, dt: Array) -> Array:
    """``x * dt[..., None]`` in float32, returned in ``x.dtype`` (Mamba-2's SSD input); saves ``x`` and ``dt``."""
    return _dt_scale(x, dt)


def mamba2_ssd(x: Array, dt: Array, a: Array, b: Array, c: Array, *, d: Array, segment_ids: Array | None,
               impl: str, chunk_size: int) -> Array:
    """:func:`jla.layers.ssm.mamba2_ssd` (training: no initial or final state) through :func:`dt_scale`
    and :func:`skip_gate`, so neither float32 copy of ``x`` is a residual."""
    heads = x.shape[2]
    v = dt_scale(x, dt).astype(c.dtype)
    g = dt.astype(f32) * a.astype(f32)
    options = dict(scale=1.0, chunk_size=chunk_size, segment_ids=segment_ids)
    if _grouped_simple_gla(impl, c, b.astype(c.dtype), v, g, **options):
        y, _ = ops.linear_attention("simple_gla", c, b.astype(c.dtype), v, g=g, impl="mosaic_gpu", **options)
    else:
        q, k = _repeat_groups(c, heads), _repeat_groups(b, heads)
        y, _ = ops.linear_attention("simple_gla", q, k.astype(q.dtype), v, g=g, impl=impl, **options)
    return skip_gate(y, x.astype(y.dtype), None, d)


# Modules ---------------------------------------------------------------------------------------
class LeanRMSNorm(RMSNorm):
    """:class:`RMSNorm` through :func:`rms_norm`."""

    def __call__(self, x: Array) -> Array:
        return rms_norm(x, self.weight, self.eps)


class LeanGatedRMSNorm(GatedRMSNorm):
    """:class:`GatedRMSNorm` through :func:`gated_rms_norm`."""

    def __call__(self, x: Array, z: Array) -> Array:
        return gated_rms_norm(x, z, self.weight, self.eps, activation=self.activation,
                              norm_before_gate=self.norm_before_gate, group_size=self.group_size)


class LeanSwiGLU(SwiGLU):
    """:class:`SwiGLU` whose product and down projection run through :func:`swiglu_linear`."""

    def __call__(self, x: Array) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        h = jnp.einsum("...d,dcf->...cf", x.astype(dtype), self.gate_up.astype(dtype))
        down_dtype = jnp.dtype(self.down.compute_dtype)
        y = swiglu_linear(h.astype(down_dtype), self.down.weight.astype(down_dtype))
        if self.down.bias is not None:
            y = y + self.down.bias.astype(down_dtype)
        return y


class LeanShortConv(ShortConv):
    """:class:`ShortConv` through :func:`short_conv` (same outputs and decode state)."""

    def __call__(self, x: Array, *, segment_ids: Array | None = None,
                 state: Array | None = None) -> tuple[Array, Array]:
        if x.ndim != 3 or x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"expected x [B, T, {self.weight.shape[0]}]; got {x.shape}")
        if self.impl != "xla":
            # ShortConv's Mosaic GPU kernels are a custom VJP that already keeps only its inputs (x, weight, bias,
            # state, segment mask) and runs forward + backward 2.4-2.9x faster than XLA (benchmarks/short_conv.md),
            # so they keep running under lean_vjp; the XLA VJP below covers where ShortConv would run XLA
            # (impl "auto" off Hopper or on shapes the kernels decline; impl "mosaic_gpu" raises there, as ShortConv).
            fused = self._fused(x, segment_ids=segment_ids, state=state)
            if fused is not None:
                return fused
        batch, seq_len, channels = x.shape
        width = self.kernel_size
        if state is None:
            state = jnp.zeros((batch, width - 1, channels), x.dtype)
        if state.shape != (batch, width - 1, channels):
            raise ValueError(f"conv state must be {(batch, width - 1, channels)}; got {state.shape}")
        x_ext = jnp.concatenate([state.astype(x.dtype), x], axis=1)
        seg_ext = masks = None
        if segment_ids is not None:
            if segment_ids.shape != (batch, seq_len):
                raise ValueError(f"segment_ids must be {(batch, seq_len)}; got {segment_ids.shape}")
            seg_ext = jnp.concatenate([jnp.broadcast_to(segment_ids[:, :1], (batch, width - 1)), segment_ids], axis=1)
            if width > 1:
                masks = jnp.stack([seg_ext[:, i:i + seq_len] == segment_ids for i in range(width - 1)])
        y = short_conv(x_ext, self.weight, self.bias, masks, activation=self.activation)
        new_state = x_ext[:, seq_len:]
        if seg_ext is not None and width > 1:
            keep = seg_ext[:, seq_len:] == segment_ids[:, -1:]
            new_state = jnp.where(keep[..., None], new_state, jnp.zeros((), x.dtype))
        return y, new_state


def _checkpointed(fn):
    return jax.checkpoint(fn, policy=jax.checkpoint_policies.nothing_saveable)


def _core_linear_attention(kind: str, impl: str, q: Array, k: Array, v: Array, options: dict) -> Array:
    """``ops.linear_attention(...)[0]`` whose residuals (chunk states, WY blocks and, when the op normalizes
    q and k itself, the L2 norm's float32 copies) are recomputed in the backward pass: only the op's inputs
    are kept."""
    arrays = {name: value for name, value in options.items() if isinstance(value, jax.Array)}
    static = {name: value for name, value in options.items() if not isinstance(value, jax.Array)}

    def core(q, k, v, arrays):
        return ops.linear_attention(kind, q, k, v, impl=impl, **static, **arrays)[0]

    return _checkpointed(core)(q, k, v, arrays)


def _remat_core_op(mixer, q: Array, k: Array, v: Array, impl: str, options: dict):
    """``(o, None)`` through :func:`_core_linear_attention` for a training call of a ``remat_core`` mixer (no state
    in or out; ``_forward`` never gets here under context parallelism without a cache), else ``None``.

    This replaces only ``LinearAttentionMixer._run_op``: q, k and v come from the mixer's own ``_forward``,
    so with ``qk_l2norm_in_conv`` they are the fused conv + L2-norm kernels' output and ``options`` carries
    ``use_qk_l2norm=False`` (no second normalization inside the recomputed op).
    """
    if not mixer.remat_core or options.get("initial_state") is not None or options.get("output_final_state"):
        return None
    return _core_linear_attention(mixer.kind, impl, q, k, v, options), None


class LeanGatedDeltaNet(GatedDeltaNet):
    """:class:`GatedDeltaNet` whose op is recomputed in the backward pass when ``remat_core``."""

    remat_core: bool = eqx.field(static=True, default=False)

    def _run_op(self, q, k, v, *, impl, **options):
        out = _remat_core_op(self, q, k, v, impl, options)
        return super()._run_op(q, k, v, impl=impl, **options) if out is None else out


class LeanKDA(KDA):
    """:class:`KDA` with its per-channel decay gate through :func:`mamba_gate` (and the op recomputed in the
    backward pass when ``remat_core``)."""

    remat_core: bool = eqx.field(static=True, default=False)

    def gates(self, x):
        cfg = self.cfg
        f = self._heads(self.f_proj(x), cfg.num_v_heads)  # [B, T, Hv, Dk]
        g = mamba_gate(f, self.A_log[:, None], self.dt_bias.reshape(cfg.num_v_heads, cfg.head_k_dim))
        return {"g": g, "beta": self._write_strength(self.b_proj(x))}

    def _run_op(self, q, k, v, *, impl, **options):
        out = _remat_core_op(self, q, k, v, impl, options)
        return super()._run_op(q, k, v, impl=impl, **options) if out is None else out


class LeanGatedDeltaNet2(GatedDeltaNet2):
    """:class:`GatedDeltaNet2` with its per-channel decay gate through :func:`mamba_gate` (and the op recomputed
    in the backward pass when ``remat_core``)."""

    remat_core: bool = eqx.field(static=True, default=False)

    def _run_op(self, q, k, v, *, impl, **options):
        out = _remat_core_op(self, q, k, v, impl, options)
        return super()._run_op(q, k, v, impl=impl, **options) if out is None else out

    def _decay(self, f):
        cfg = self.cfg
        return mamba_gate(f, self.A_log[:, None], self.dt_bias.reshape(cfg.num_k_heads, cfg.head_k_dim))


class LeanMamba2(Mamba2):
    """:class:`Mamba2` whose SSD prologue (``dt * x``) and D skip run through :func:`mamba2_ssd`.

    Training only (decoding and log-linear Mamba-2 keep :class:`Mamba2`'s path).
    """

    def _forward(self, u: Array, *, segment_ids: Array | None, cache):
        if cache is not None or self.level_proj is not None:
            return super()._forward(u, segment_ids=segment_ids, cache=cache)
        cfg, heads = self.cfg, self.num_heads
        batch, length, _ = u.shape
        z, x = _split_rows(self.zx_proj, u)
        bc = self.bc_proj(u).reshape(batch, length, -1)
        x, _ = self.x_conv(x, segment_ids=segment_ids)
        bc, _ = self.bc_conv(bc, segment_ids=segment_ids)
        bc = bc.reshape(batch, length, 2, cfg.n_groups, cfg.d_state)
        dt = jax.nn.softplus(self.dt_proj(u).astype(jnp.float32) + self.dt_bias.astype(jnp.float32))
        a = -jnp.exp(self.A_log.astype(jnp.float32))
        y = mamba2_ssd(x.reshape(batch, length, heads, cfg.head_dim), dt, a, bc[:, :, 0], bc[:, :, 1], d=self.D,
                       segment_ids=segment_ids, impl=cfg.implementation, chunk_size=cfg.chunk_size)
        y = checkpoint_name(y, ATTENTION_CHECKPOINT_NAME)
        return self.out_proj(self.norm(y.reshape(batch, length, -1), z)), None


class LeanMamba3(Mamba3):
    """:class:`Mamba3` whose ``(y + D x) * silu(z)`` epilogue runs through :func:`skip_gate`.

    Training only: the op runs without ``d`` / ``z`` and the epilogue follows; decoding (a cache)
    keeps :class:`Mamba3`'s path. With ``remat_core`` the op (rotary, gates and SSD kernels) is
    recomputed in the backward pass instead of keeping its residuals.
    """

    remat_core: bool = eqx.field(static=True, default=False)

    def _forward(self, u: Array, *, segment_ids: Array | None, cache):
        if cache is not None:
            return super()._forward(u, segment_ids=segment_ids, cache=cache)
        cfg, heads = self.cfg, self.num_heads
        batch, length, _ = u.shape
        z, x = _split_rows(self.zx_proj, u)
        z = z.reshape(batch, length, heads, cfg.head_dim)
        x = x.reshape(batch, length, heads, cfg.head_dim)
        bc = self.bc_proj(u).reshape(batch, length, 2, cfg.n_groups, cfg.d_state)
        b, c = self.B_norm(bc[:, :, 0]), self.C_norm(bc[:, :, 1])
        scalars = self.scalar_proj(u).astype(jnp.float32)
        dt = jax.nn.softplus(scalars[..., 0, :] + self.dt_bias.astype(jnp.float32))
        a = jnp.minimum(-heavy_tail(scalars[..., 1, :]), -cfg.a_floor)

        def core(c, b, x, adt, dt, trap, q_bias, k_bias, angle_rows, segment_ids):
            angles = jnp.broadcast_to(angle_rows[:, :, None, :], (batch, length, heads, self.num_angles))
            return mamba3_siso(c, b, x, adt, dt, trap, q_bias, k_bias, angles, d=None, z=None, segment_ids=segment_ids,
                               impl=cfg.implementation, chunk_size=cfg.chunk_size)[0]

        y = (_checkpointed(core) if self.remat_core else core)(
            c, b, x, a * dt, dt, scalars[..., 2, :], self.C_bias, self.B_bias,
            self.angle_proj(u).astype(jnp.float32), segment_ids)
        y = skip_gate(y, x, None if self.norm is not None else z, self.D)
        y = checkpoint_name(y.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        if self.norm is not None:
            y = self.norm(y, z.reshape(batch, length, -1))
        return self.out_proj(y), None


_LEAN = {
    RMSNorm: LeanRMSNorm,
    GatedRMSNorm: LeanGatedRMSNorm,
    SwiGLU: LeanSwiGLU,
    ShortConv: LeanShortConv,
    GatedDeltaNet: LeanGatedDeltaNet,
    KDA: LeanKDA,
    GatedDeltaNet2: LeanGatedDeltaNet2,
    Mamba2: LeanMamba2,
    Mamba3: LeanMamba3,
}


def _swap(node: Any, remat_core: bool) -> Any:
    cls = _LEAN[type(node)]
    new = object.__new__(cls)
    for field in dataclasses.fields(node):
        object.__setattr__(new, field.name, apply_lean(getattr(node, field.name), remat_core=remat_core))
    if "remat_core" in {field.name for field in dataclasses.fields(cls)}:
        object.__setattr__(new, "remat_core", bool(remat_core))
    return new


def apply_lean(tree: Any, *, remat_core: bool = False) -> Any:
    """``tree`` (a model or part of one) with every lean-able module swapped for its lean subclass.

    Only exact types are swapped (a subclass with its own forward keeps it); arrays and tree paths
    are unchanged, so parameters, optimizer states and checkpoints carry over. ``remat_core``
    (``ModelConfig.lean_remat_core``) makes the Gated DeltaNet, KDA, GDN-2 and Mamba-3 mixers rerun
    their op in the backward pass instead of keeping its residuals (training calls only).
    """
    return jax.tree_util.tree_map(lambda n: _swap(n, remat_core) if type(n) in _LEAN else n, tree,
                                  is_leaf=lambda n: type(n) in _LEAN)
