# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU feed-forward block with a fused gate/up weight."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers.linear import Linear, truncated_normal

MLP_HIDDEN_CHECKPOINT_NAME = "mlp_hidden"  # the SwiGLU gate/up product, for the "names" remat policies

__all__ = ["SwiGLU", "swiglu"]

Array = jax.Array


def swiglu(gate_up: Array) -> Array:
    """``silu(gate) * up`` for ``gate_up [..., 2, F]``, computed in float32."""
    gate = gate_up[..., 0, :].astype(jnp.float32)
    up = gate_up[..., 1, :].astype(jnp.float32)
    return (jax.nn.silu(gate) * up).astype(gate_up.dtype)


class SwiGLU(eqx.Module):
    """``down(silu(x W_gate) * x W_up)``.

    The gate and up projections share one weight ``[D, 2, F]`` (one GEMM); keeping
    the pair axis separate from ``F`` means tensor parallelism over ``"mlp"`` gives
    every device matching gate and up columns.
    """

    gate_up: Array  # [D, 2, F]
    down: Linear
    compute_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        key: Array,
        std: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        k1, k2 = jax.random.split(key)
        self.gate_up = truncated_normal(k1, (d_model, 2, d_ff), std, param_dtype)
        self.down = Linear(d_ff, d_model, key=k2, axes=("mlp", "embed"), std=std,
                           param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.compute_dtype = str(jnp.dtype(compute_dtype))

    def __call__(self, x: Array) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        h = jnp.einsum("...d,dcf->...cf", x.astype(dtype), self.gate_up.astype(dtype))
        return self.down(swiglu(checkpoint_name(h, MLP_HIDDEN_CHECKPOINT_NAME)))

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"gate_up": ("embed", None, "mlp")}


class SquaredReLUMLP(eqx.Module):
    """``down(relu(x W_up) ** 2)``: the ungated squared-ReLU MLP of Nemotron-H (``relu2``)."""

    up: Linear
    down: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        key: Array,
        std: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        k1, k2 = jax.random.split(key)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.up = Linear(d_model, d_ff, key=k1, axes=("embed", "mlp"), **common)
        self.down = Linear(d_ff, d_model, key=k2, axes=("mlp", "embed"), **common)

    def __call__(self, x: Array) -> Array:
        h = self.up(x)
        r = jax.nn.relu(h.astype(jnp.float32))
        return self.down((r * r).astype(h.dtype))
