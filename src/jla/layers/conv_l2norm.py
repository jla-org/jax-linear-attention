# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A :class:`~jla.layers.conv.ShortConv` followed by a per-head L2 norm, fused on Hopper.

The delta-rule mixers (Gated DeltaNet, GDN-2, KDA, DeltaNet) run q and k through a short conv + SiLU
and then L2-normalize each head (``use_qk_l2norm``). :func:`conv_l2norm` does both in the Mosaic GPU
kernels of :mod:`jla.ops.mosaic_gpu.short_conv_l2norm` when the conv would run its Mosaic GPU kernel
(``impl`` ``"auto"`` or ``"mosaic_gpu"``) and those kernels support the call, and returns ``None``
otherwise, so the caller keeps the unfused path (conv, then the op's ``use_qk_l2norm``).

Under ``ModelConfig.lean_vjp`` the convs are :class:`~jla.layers.lean.LeanShortConv`; this module reads only their
parameters and ``impl``, so the fused kernels (a custom VJP that already saves only its inputs) keep producing q
and k, and lean's XLA conv VJP runs only where these kernels decline (``tests/test_lean_conv_l2norm.py``).

On several devices the kernels run per shard of the context mesh
(:func:`jla.ops.mosaic_gpu.short_conv_l2norm.short_conv_l2norm_sharded`: batch over the data axes, whole heads
over the axes of the conv's logical channel axis); :func:`conv_l2norm` returns ``None`` when no split gives
per-device shapes the kernels take.
"""

from __future__ import annotations

import jax

from jla.layers.conv import ShortConv, _final_state

__all__ = ["conv_l2norm", "conv_l2norm_reason"]

Array = jax.Array


def conv_l2norm_reason(conv: ShortConv, x: Array, head_dim: int, *, segment_ids: Array | None = None,
                       state: Array | None = None) -> str | None:
    """``None`` when :func:`conv_l2norm` runs the fused kernels for this call, otherwise why not.

    On several devices ``None`` means the global call qualifies and the context mesh can split it; whether a
    split gives per-device shapes the kernels take is decided by :func:`conv_l2norm` (``None`` if none does)."""
    if conv.impl == "xla":
        return "the conv runs on XLA (impl='xla')"
    from jla.ops import mosaic_gpu

    # The kernel module imports Mosaic GPU at top level; only the package import is guarded, so check
    # first (qk_l2norm_in_conv and conv_impl='auto' are defaults and must not break such a JAX build).
    if not mosaic_gpu._HAVE_MOSAIC_GPU:
        return "this JAX build has no Mosaic GPU (jax.experimental.pallas.mosaic_gpu)"
    from jla.ops.mosaic_gpu import short_conv_l2norm as kernels

    batch, seq_len, channels = x.shape
    if state is not None and state.shape != (batch, conv.kernel_size - 1, channels):
        return f"conv state must be {(batch, conv.kernel_size - 1, channels)}; got {state.shape}"
    if state is not None:
        state = state.astype(x.dtype)
    reason = kernels.unsupported_reason(x, conv.weight, head_dim=head_dim, segment_ids=segment_ids, state=state)
    if reason is not None and _per_shard(kernels, conv, x, head_dim, segment_ids, state):
        return None
    return reason


def _per_shard(kernels, conv: ShortConv, x: Array, head_dim: int, segment_ids: Array | None,
               state: Array | None) -> bool:
    """Only the multi-device rule declines the call, and the context mesh can split it."""
    from jla.distributed.kernel_sharding import mesh_can_split, per_device_call

    return (not per_device_call() and mesh_can_split()
            and kernels.unsupported_reason(x, conv.weight, head_dim=head_dim, segment_ids=segment_ids, state=state,
                                           check_devices=False) is None)


def conv_l2norm(conv: ShortConv, x: Array, head_dim: int, *, segment_ids: Array | None = None,
                state: Array | None = None) -> tuple[Array, Array] | None:
    """``(l2norm(conv(x)) per head_dim-channel head, new_state)`` from the fused kernels, or ``None``
    when :func:`conv_l2norm_reason` gives a reason, or on several devices when no split of the context mesh
    gives per-device shapes the kernels take (the caller then runs the conv and the norm itself)."""
    if conv_l2norm_reason(conv, x, head_dim, segment_ids=segment_ids, state=state) is not None:
        return None
    from jla.distributed.kernel_sharding import per_device_call, rule_axes
    from jla.ops.mosaic_gpu import short_conv_l2norm as kernels

    if state is not None:
        state = state.astype(x.dtype)
    if per_device_call():
        y = kernels.short_conv_l2norm(x, conv.weight, conv.bias, head_dim=head_dim, activation=conv.activation,
                                      segment_ids=segment_ids, state=state)
    else:
        y = kernels.short_conv_l2norm_sharded(x, conv.weight, conv.bias, head_dim=head_dim,
                                              activation=conv.activation, segment_ids=segment_ids, state=state,
                                              channel_axes=rule_axes(conv.axis) if conv.axis else ())
        if y is None:
            return None
    return y, _final_state(x, state, segment_ids, conv.kernel_size)
