# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Linear and embedding layers with explicit parameter and compute dtypes.

Parameters are stored in the dtype they were created with (the model's
``param_dtype``); every matmul casts both operands to ``compute_dtype`` and
returns ``compute_dtype``. Each layer records the logical axis names of its
weight so the trainer can shard parameters by name.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

__all__ = ["Embedding", "Linear", "LowRankLinear", "StackedLinear", "truncated_normal"]

Array = jax.Array
LINEAR_CHECKPOINT_NAME = "linear"  # every projection output, for the "names" remat policies
Axes = tuple[str | None, ...]


def truncated_normal(key: Array, shape: Sequence[int], std: float, dtype: jnp.dtype = jnp.float32) -> Array:
    """Normal(0, ``std``) truncated at two standard deviations, sampled in float32."""
    return (std * jax.random.truncated_normal(key, -2.0, 2.0, tuple(shape), jnp.float32)).astype(dtype)


class Linear(eqx.Module):
    """``y = x @ W (+ b)`` with ``W [in, out]``.

    ``init`` sets the weight's starting values (the bias always starts at zero):
    ``"truncated_normal"`` (the default) draws ``N(0, std)`` truncated at two standard
    deviations; ``"zeros"`` starts at zero (RWKV-7's output projections and LoRA down
    projections); ``"orthogonal"`` draws a matrix with orthonormal rows or columns, whichever
    side is smaller (``torch.nn.init.orthogonal_``), times ``gain``; ``"kaiming_uniform"`` is
    PyTorch's ``nn.Linear`` default ``kaiming_uniform_(a=sqrt(5))``, ``U(-1/sqrt(in), 1/sqrt(in))``,
    times ``gain`` (Mamba's depth-scaled ``out_proj``). Only ``"truncated_normal"`` uses ``std``.
    """

    weight: Array
    bias: Array | None
    compute_dtype: str = eqx.field(static=True)
    axes: Axes = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: Array,
        axes: Axes,
        std: float,
        use_bias: bool = False,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
        init: str = "truncated_normal",
        gain: float = 1.0,
    ):
        if len(axes) != 2:
            raise ValueError(f"a linear weight needs two logical axes; got {axes}")
        self.weight = linear_init(key, (in_features, out_features), init, std=std, gain=gain, dtype=param_dtype)
        self.bias = jnp.zeros((out_features,), param_dtype) if use_bias else None
        self.compute_dtype = str(jnp.dtype(compute_dtype))
        self.axes = tuple(axes)

    def __call__(self, x: Array) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        y = lax.dot_general(x.astype(dtype), self.weight.astype(dtype), (((x.ndim - 1,), (0,)), ((), ())))
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        return checkpoint_name(y, LINEAR_CHECKPOINT_NAME)

    def param_axes(self) -> dict[str, Axes]:
        return {"weight": self.axes, "bias": (self.axes[1],)}


class StackedLinear(eqx.Module):
    """Several same-width projections of one input in a single GEMM: ``W [in, n, out]``.

    ``x [..., in] -> [..., n, out]``. Keeping the stack axis apart from ``out`` lets
    the output axis be sharded without splitting a projection across devices.
    """

    weight: Array
    compute_dtype: str = eqx.field(static=True)
    axes: Axes = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        count: int,
        out_features: int,
        *,
        key: Array,
        out_axis: str | None,
        std: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        self.weight = truncated_normal(key, (in_features, count, out_features), std, param_dtype)
        self.compute_dtype = str(jnp.dtype(compute_dtype))
        self.axes = ("embed", None, out_axis)

    def __call__(self, x: Array) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        return checkpoint_name(jnp.einsum("...d,dno->...no", x.astype(dtype), self.weight.astype(dtype)),
                               LINEAR_CHECKPOINT_NAME)

    def param_axes(self) -> dict[str, Axes]:
        return {"weight": self.axes}


class LowRankLinear(eqx.Module):
    """``x @ A @ B (+ b)``: a bottlenecked projection (FLA's gate projections)."""

    down: Linear
    up: Linear

    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        *,
        key: Array,
        out_axis: str | None,
        std: float,
        use_bias: bool = False,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        k1, k2 = jax.random.split(key)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.down = Linear(in_features, rank, key=k1, axes=("embed", None), **common)
        self.up = Linear(rank, out_features, key=k2, axes=(None, out_axis), use_bias=use_bias, **common)

    def __call__(self, x: Array) -> Array:
        return self.up(self.down(x))


class Embedding(eqx.Module):
    """Token embedding table ``[vocab, d_model]``; lookups return ``compute_dtype``."""

    weight: Array
    compute_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        key: Array,
        std: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        self.weight = truncated_normal(key, (vocab_size, dim), std, param_dtype)
        self.compute_dtype = str(jnp.dtype(compute_dtype))

    def __call__(self, ids: Array) -> Array:
        if not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError(f"token ids must be integers; got {ids.dtype}")
        return jnp.take(self.weight, ids, axis=0).astype(self.compute_dtype)

    def param_axes(self) -> dict[str, Axes]:
        return {"weight": ("vocab", "embed")}


LINEAR_INITS = ("truncated_normal", "zeros", "orthogonal", "kaiming_uniform")


def linear_init(key: Array, shape: Sequence[int], init: str, *, std: float, gain: float = 1.0,
                dtype: jnp.dtype = jnp.float32) -> Array:
    """A ``[in, out]`` weight drawn by ``init`` (see :class:`Linear`), sampled in float32."""
    shape = tuple(shape)
    if init == "truncated_normal":
        return truncated_normal(key, shape, std, dtype)
    if init == "zeros":
        return jnp.zeros(shape, dtype)
    if init == "orthogonal":
        return jax.nn.initializers.orthogonal(scale=gain)(key, shape, jnp.float32).astype(dtype)
    if init == "kaiming_uniform":
        bound = gain / shape[0] ** 0.5  # kaiming_uniform_(a=sqrt(5)): gain sqrt(1/3) * sqrt(3) / sqrt(fan_in)
        return jax.random.uniform(key, shape, jnp.float32, -bound, bound).astype(dtype)
    raise ValueError(f"unknown linear init {init!r}; expected one of {LINEAR_INITS}")
