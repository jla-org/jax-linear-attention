# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gated_delta_rule/chunk.py
"""Mosaic GPU Gated DeltaNet and DeltaNet (``g = 0``) for Hopper.

Same contract as :func:`jla.ops.reference.gated_delta_rule` and
:func:`jla.ops.reference.delta_rule`; see
:mod:`jla.ops.mosaic_gpu.scalar_gate` for the kernel pipeline and the
grouped-head extension.
"""

from __future__ import annotations

import jax

from jla.ops import register
from jla.ops.mosaic_gpu import scalar_gate_kb
from jla.ops.mosaic_gpu._common import CHUNK
from jla.ops.mosaic_gpu.scalar_gate import run, unsupported_reason

Array = jax.Array


def gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
    chunk_size: int = CHUNK,
) -> tuple[Array, Array | None]:
    """Gated DeltaNet (FLA ``chunk_gated_delta_rule``) with Mosaic GPU kernels."""
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size)
    if scalar_gate_kb.handles(q, v):  # large heads (K = 256, V up to 512)
        reason = scalar_gate_kb.unsupported_reason(q, k, v, (g, beta), options)
        if reason is not None:
            raise ValueError(f"mosaic_gpu gated_delta_rule: {reason}")
        return scalar_gate_kb.run(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                                  output_final_state=output_final_state, segment_ids=segment_ids,
                                  use_qk_l2norm=use_qk_l2norm)
    reason = unsupported_reason(q, k, v, (g, beta), options, delta=True)
    if reason is not None:
        raise ValueError(f"mosaic_gpu gated_delta_rule: {reason}")
    return run(q, k, v, g, beta, delta=True, scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, segment_ids=segment_ids, use_qk_l2norm=use_qk_l2norm)


def delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
    chunk_size: int = CHUNK,
) -> tuple[Array, Array | None]:
    """DeltaNet: the Gated DeltaNet kernels with a zero log-decay."""
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size)
    if scalar_gate_kb.handles(q, v):  # large heads (K = 256, V up to 512)
        reason = scalar_gate_kb.unsupported_reason(q, k, v, (beta,), options)
        if reason is not None:
            raise ValueError(f"mosaic_gpu delta_rule: {reason}")
        return scalar_gate_kb.run(q, k, v, None, beta, scale=scale, initial_state=initial_state,
                                  output_final_state=output_final_state, segment_ids=segment_ids,
                                  use_qk_l2norm=use_qk_l2norm)
    reason = unsupported_reason(q, k, v, (beta,), options, delta=True)
    if reason is not None:
        raise ValueError(f"mosaic_gpu delta_rule: {reason}")
    return run(q, k, v, None, beta, delta=True, scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, segment_ids=segment_ids, use_qk_l2norm=use_qk_l2norm)


def supports_gated_delta_rule(q, k, v, g, beta, **options) -> bool:
    if scalar_gate_kb.handles(q, v):
        return scalar_gate_kb.unsupported_reason(q, k, v, (g, beta), options) is None
    return unsupported_reason(q, k, v, (g, beta), options, delta=True) is None


def supports_delta_rule(q, k, v, beta, **options) -> bool:
    if scalar_gate_kb.handles(q, v):
        return scalar_gate_kb.unsupported_reason(q, k, v, (beta,), options) is None
    return unsupported_reason(q, k, v, (beta,), options, delta=True) is None


register("mosaic_gpu", "gated_deltanet", gated_delta_rule, supports_gated_delta_rule, auto=True)
register("mosaic_gpu", "deltanet", delta_rule, supports_delta_rule, auto=True)
