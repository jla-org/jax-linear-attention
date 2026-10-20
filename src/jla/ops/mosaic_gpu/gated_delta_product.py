# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gated_delta_product/chunk.py
"""Gated DeltaProduct on the Mosaic GPU Gated DeltaNet kernels.

FLA runs Gated DeltaProduct as the gated delta rule over ``T * n_h``
interleaved sub-steps (:func:`jla.ops.chunked.householder_interleave`); this
wrapper does the same with the Mosaic GPU ``gated_deltanet`` kernel, forward
and backward. ``supports`` is the kernel's predicate on the interleaved call.
"""

from __future__ import annotations

import jax

from jla.ops import linear_attention, register
from jla.ops.chunked import householder_interleave
from jla.ops.mosaic_gpu._common import CHUNK
from jla.ops.mosaic_gpu.scalar_gate import unsupported_reason
from jla.ops.reference import householder_steps, l2norm

Array = jax.Array


class _Shape:
    """Shape and dtype of an interleaved operand, for the kernel predicate (nothing is materialized)."""

    def __init__(self, shape: tuple, dtype) -> None:
        self.shape, self.dtype, self.ndim = tuple(shape), dtype, len(shape)


def _reason(q, k, v, g, beta, options: dict) -> str | None:
    options = dict(options)
    try:
        n = householder_steps(q, k, v, g, beta, options.pop("num_householder", None))
    except ValueError as err:
        return str(err)
    batch, seq_len = q.shape[:2]
    q_i = _Shape((batch, seq_len * n, *q.shape[2:]), q.dtype)
    g_i = _Shape((batch, seq_len * n, *g.shape[2:]), g.dtype)
    segment_ids = options.get("segment_ids")
    if segment_ids is not None:
        if segment_ids.shape != (batch, seq_len):
            return f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}"
        options["segment_ids"] = _Shape((batch, seq_len * n), segment_ids.dtype)
    return unsupported_reason(q_i, k, v, (g_i, beta), options, delta=True)


def gated_delta_product(
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
    num_householder: int | None = None,
    chunk_size: int = CHUNK,
) -> tuple[Array, Array | None]:
    """Gated DeltaProduct (FLA ``chunk_gated_delta_product``) with Mosaic GPU kernels."""
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size,
                   num_householder=num_householder, use_qk_l2norm=use_qk_l2norm)
    reason = _reason(q, k, v, g, beta, options)
    if reason is not None:
        raise ValueError(f"mosaic_gpu gated_delta_product: {reason}")
    n = householder_steps(q, k, v, g, beta, num_householder)
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_i, g_i, seg_i = householder_interleave(q, g, segment_ids, n)
    o, final = linear_attention(
        "gated_deltanet", q_i, k, v, g=g_i, beta=beta, impl="mosaic_gpu", scale=scale,
        initial_state=initial_state, output_final_state=output_final_state, segment_ids=seg_i,
        chunk_size=chunk_size,
    )
    return o[:, n - 1 :: n], final


def supports_gated_delta_product(q, k, v, g, beta, **options) -> bool:
    return _reason(q, k, v, g, beta, options) is None


register("mosaic_gpu", "gated_deltaproduct", gated_delta_product, supports_gated_delta_product, auto=False)
