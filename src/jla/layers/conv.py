# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Short depthwise causal convolution (FLA's ``ShortConvolution``, Mamba-2's ``conv1d``).

``y_t = act(b + sum_i w[:, i] * x_{t - (K - 1 - i)})``: tap ``K - 1`` multiplies the
current token and tap ``0`` the oldest, the PyTorch ``Conv1d`` convention used by
FLA and state-spaces/mamba. The layer carries a decode state (the last ``K - 1``
inputs, oldest first), so prefill followed by token-by-token decoding reproduces
the full-sequence output exactly. With ``segment_ids`` a tap never reads a token
from a different document; the incoming state counts as part of each row's first
segment.

``impl`` picks the implementation: ``"xla"`` (plain JAX ops, the default),
``"mosaic_gpu"`` (the fused Mosaic GPU kernels of :mod:`jla.ops.mosaic_gpu.short_conv`,
forward and backward; raises on calls they do not support) or ``"auto"`` (the kernels
when they support the call, i.e. one Hopper GPU, otherwise XLA). On several devices the
kernels run per shard of the context mesh (``short_conv.short_conv_sharded``).
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["ShortConv"]

Array = jax.Array


class ShortConv(eqx.Module):
    """Depthwise causal conv over ``[B, T, C]`` with an optional bias and SiLU.

    ``init="identity"`` starts the kernel as the identity (current-token tap 1,
    older taps 0), so a fresh linear-attention mixer behaves like its conv-free
    variant; ``init="uniform"`` is PyTorch's ``Conv1d`` default,
    ``U(-1/sqrt(K), 1/sqrt(K))`` for weight and bias (Mamba-2's ``conv_init=None``);
    ``init="normal"`` draws the weight from ``N(0, std)`` (untruncated) with a zero bias, FLA's
    ``_init_weights`` for every ``nn.Conv1d`` (its ``ShortConvolution``, used by the delta-rule
    mixers, ``std = initializer_range``).
    """

    weight: Array  # [C, K]
    bias: Array | None  # [C]
    kernel_size: int = eqx.field(static=True)
    activation: bool = eqx.field(static=True)
    axis: str | None = eqx.field(static=True)  # logical axis of the channel dimension
    impl: str = eqx.field(static=True, default="xla")  # "xla", "mosaic_gpu" or "auto"

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        activation: bool = True,
        use_bias: bool = False,
        init: Literal["identity", "uniform", "normal"] = "identity",
        key: Array | None = None,
        axis: str | None = "heads",
        param_dtype: jnp.dtype = jnp.float32,
        impl: Literal["xla", "mosaic_gpu", "auto"] = "xla",
        std: float | None = None,
    ):
        if impl not in _IMPLS:
            raise ValueError(f"unknown ShortConv impl {impl!r}; expected one of {_IMPLS}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1; got {kernel_size}")
        if init == "identity":
            self.weight = jnp.zeros((channels, kernel_size), param_dtype).at[:, -1].set(1.0)
            self.bias = jnp.zeros((channels,), param_dtype) if use_bias else None
        elif init == "uniform":
            if key is None:
                raise ValueError("init='uniform' needs a key")
            bound = 1.0 / math.sqrt(kernel_size)  # fan_in of a depthwise kernel is K
            kw, kb = jax.random.split(key)
            self.weight = jax.random.uniform(kw, (channels, kernel_size), jnp.float32, -bound, bound).astype(
                param_dtype)
            self.bias = (
                jax.random.uniform(kb, (channels,), jnp.float32, -bound, bound).astype(param_dtype)
                if use_bias else None
            )
        elif init == "normal":
            if key is None or std is None:
                raise ValueError("init='normal' needs a key and std")
            self.weight = (std * jax.random.normal(key, (channels, kernel_size), jnp.float32)).astype(param_dtype)
            self.bias = jnp.zeros((channels,), param_dtype) if use_bias else None
        else:
            raise ValueError(f"unknown conv init {init!r}")
        self.kernel_size = kernel_size
        self.activation = activation
        self.axis = axis
        self.impl = impl

    def __call__(
        self,
        x: Array,
        *,
        segment_ids: Array | None = None,
        state: Array | None = None,
    ) -> tuple[Array, Array]:
        """Returns ``(y [B, T, C], new_state [B, K - 1, C])``."""
        if x.ndim != 3 or x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"expected x [B, T, {self.weight.shape[0]}]; got {x.shape}")
        batch, seq_len, channels = x.shape
        width = self.kernel_size
        if self.impl != "xla":
            fused = self._fused(x, segment_ids=segment_ids, state=state)
            if fused is not None:
                return fused
        if state is None:
            state = jnp.zeros((batch, width - 1, channels), x.dtype)
        if state.shape != (batch, width - 1, channels):
            raise ValueError(f"conv state must be {(batch, width - 1, channels)}; got {state.shape}")
        x_ext = jnp.concatenate([state.astype(x.dtype), x], axis=1)  # [B, K - 1 + T, C]
        seg_ext = None
        if segment_ids is not None:
            if segment_ids.shape != (batch, seq_len):
                raise ValueError(f"segment_ids must be {(batch, seq_len)}; got {segment_ids.shape}")
            seg_ext = jnp.concatenate(
                [jnp.broadcast_to(segment_ids[:, :1], (batch, width - 1)), segment_ids], axis=1
            )
        w = self.weight.astype(jnp.float32)
        y = jnp.zeros((batch, seq_len, channels), jnp.float32)
        if self.bias is not None:
            y = y + self.bias.astype(jnp.float32)
        for i in range(width):
            tap = x_ext[:, i : i + seq_len].astype(jnp.float32) * w[:, i]
            if seg_ext is not None and i < width - 1:
                same = seg_ext[:, i : i + seq_len] == segment_ids
                tap = jnp.where(same[..., None], tap, 0.0)
            y = y + tap
        if self.activation:
            y = jax.nn.silu(y)
        new_state = x_ext[:, seq_len:]
        if seg_ext is not None and width > 1:
            # Entries from an earlier document must not leak into the next decode step.
            keep = seg_ext[:, seq_len:] == segment_ids[:, -1:]
            new_state = jnp.where(keep[..., None], new_state, jnp.zeros((), x.dtype))
        return y.astype(x.dtype), new_state

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"weight": (self.axis, None), "bias": (self.axis,)}

    def _fused(self, x: Array, *, segment_ids: Array | None, state: Array | None) -> tuple[Array, Array] | None:
        """The Mosaic GPU kernels' ``(y, new_state)``, or ``None`` when ``impl="auto"`` falls back to XLA."""
        from jla.ops import mosaic_gpu

        if not mosaic_gpu._HAVE_MOSAIC_GPU:  # the kernel module imports Mosaic GPU at top level
            if self.impl == "mosaic_gpu":
                raise ValueError("ShortConv impl='mosaic_gpu': this JAX build has no Mosaic GPU")
            return None
        from jla.ops.mosaic_gpu import short_conv as kernels

        batch, seq_len, channels = x.shape
        width = self.kernel_size
        if state is not None:
            if state.shape != (batch, width - 1, channels):
                raise ValueError(f"conv state must be {(batch, width - 1, channels)}; got {state.shape}")
            state = state.astype(x.dtype)
        if segment_ids is not None and segment_ids.shape != (batch, seq_len):
            raise ValueError(f"segment_ids must be {(batch, seq_len)}; got {segment_ids.shape}")
        reason = kernels.unsupported_reason(x, self.weight, segment_ids=segment_ids, state=state)
        if reason is not None and kernels.unsupported_reason(x, self.weight, segment_ids=segment_ids, state=state,
                                                             check_devices=False) is None:
            # Only the multi-device rule declined: run the kernels per shard of the context mesh (batch over the
            # data axes, channels over the axes of this conv's logical axis; jla.distributed.kernel_sharding).
            from jla.distributed.kernel_sharding import rule_axes

            y = kernels.short_conv_sharded(x, self.weight, self.bias, activation=self.activation,
                                           segment_ids=segment_ids, state=state,
                                           channel_axes=rule_axes(self.axis) if self.axis else ())
            if y is not None:
                return y, _final_state(x, state, segment_ids, width)
            reason = f"{reason}; and no split of the context mesh gives per-device shapes the kernels take"
        if reason is not None:
            if self.impl == "mosaic_gpu":
                raise ValueError(f"ShortConv impl='mosaic_gpu': {reason}")
            return None
        y = kernels.short_conv(x, self.weight, self.bias, activation=self.activation, segment_ids=segment_ids,
                               state=state)
        return y, _final_state(x, state, segment_ids, width)


_IMPLS = ("xla", "mosaic_gpu", "auto")


def _final_state(x: Array, state: Array | None, segment_ids: Array | None, width: int) -> Array:
    """The last ``width - 1`` inputs (``x_ext[:, T:]`` of the XLA path) without concatenating all of ``x``."""
    batch, seq_len, channels = x.shape
    if state is None:
        state = jnp.zeros((batch, width - 1, channels), x.dtype)
    if seq_len >= width - 1:
        tail = x[:, seq_len - (width - 1):]
    else:
        tail = jnp.concatenate([state[:, seq_len:], x], axis=1)
    if segment_ids is not None and width > 1:
        seg = segment_ids
        if seq_len >= width - 1:
            seg_tail = seg[:, seq_len - (width - 1):]
        else:
            seg_tail = jnp.concatenate([jnp.broadcast_to(seg[:, :1], (batch, width - 1 - seq_len)), seg], axis=1)
        keep = seg_tail == seg[:, -1:]
        tail = jnp.where(keep[..., None], tail, jnp.zeros((), x.dtype))
    return tail
