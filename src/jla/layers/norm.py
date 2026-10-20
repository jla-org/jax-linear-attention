# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/modules/fused_norm_gate.py (gated RMSNorm semantics)
"""Normalization layers. Every norm computes in float32 and returns the input dtype."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["GatedRMSNorm", "RMSNorm", "l2_normalize", "rms_norm"]

Array = jax.Array
GateActivation = Literal["silu", "sigmoid"]


def rms_norm(x: Array, weight: Array | None, eps: float) -> Array:
    """``x / rms(x) * weight`` over the last axis in float32, returned in ``x.dtype``.

    The weight multiplies in float32 before the cast (Hugging Face casts first;
    the difference only matters for bit-level parity).
    """
    x32 = x.astype(jnp.float32)
    y = x32 * lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    if weight is not None:
        y = y * weight.astype(jnp.float32)
    return y.astype(x.dtype)


def l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    """``x / sqrt(sum(x^2) + eps)`` over the last axis in float32 (FLA ``l2norm``)."""
    x32 = x.astype(jnp.float32)
    return (x32 * lax.rsqrt(jnp.sum(x32 * x32, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class RMSNorm(eqx.Module):
    """Root-mean-square norm over the last axis with a learned scale (initialized to one)."""

    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, eps: float = 1e-6, param_dtype: jnp.dtype = jnp.float32):
        self.weight = jnp.ones((dim,), param_dtype)
        self.eps = float(eps)

    def __call__(self, x: Array) -> Array:
        return rms_norm(x, self.weight, self.eps)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"weight": (None,)}


class GatedRMSNorm(eqx.Module):
    """RMS norm with a multiplicative gate ``act(z)``.

    * ``norm_before_gate=True`` (default): ``rms_norm(x) * weight * act(z)``, FLA's
      ``FusedRMSNormGated``; ``activation="silu"`` is its ``swish`` gate (GatedDeltaNet,
      GLA) and ``"sigmoid"`` the Kimi Delta Attention gate;
    * ``norm_before_gate=False``: ``rms_norm(x * act(z)) * weight``, the Mamba-2 norm.

    ``group_size`` normalizes each consecutive group of that many channels on its
    own (state-spaces/mamba's grouped ``RMSNormGated``); ``None`` uses the whole axis.
    """

    weight: Array
    eps: float = eqx.field(static=True)
    activation: GateActivation = eqx.field(static=True)
    norm_before_gate: bool = eqx.field(static=True)
    group_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        activation: GateActivation = "silu",
        norm_before_gate: bool = True,
        group_size: int | None = None,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        if activation not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported gate activation {activation!r}")
        if group_size is not None and (group_size < 1 or dim % group_size):
            raise ValueError(f"group_size {group_size} must divide dim {dim}")
        self.weight = jnp.ones((dim,), param_dtype)
        self.eps = float(eps)
        self.activation = activation
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size

    def __call__(self, x: Array, z: Array) -> Array:
        if z.shape != x.shape:
            raise ValueError(f"gate shape {z.shape} must match input shape {x.shape}")
        x32, z32 = x.astype(jnp.float32), z.astype(jnp.float32)
        gate = jax.nn.silu(z32) if self.activation == "silu" else jax.nn.sigmoid(z32)
        if not self.norm_before_gate:
            x32 = x32 * gate
        dim = x.shape[-1]
        group = self.group_size or dim
        grouped = x32.reshape(*x.shape[:-1], dim // group, group)
        y = grouped * lax.rsqrt(jnp.mean(grouped * grouped, axis=-1, keepdims=True) + self.eps)
        y = y.reshape(x.shape) * self.weight.astype(jnp.float32)
        if self.norm_before_gate:
            y = y * gate
        return y.astype(x.dtype)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"weight": (None,)}
