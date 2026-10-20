# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/kda/chunk.py
"""Kimi Delta Attention (``kind="kda"``) on Hopper GPUs with Pallas Mosaic GPU kernels.

Registers ``impl="mosaic_gpu"`` for :func:`jla.ops.linear_attention`. The kernels and their
numerics are described in :mod:`jla.ops.mosaic_gpu.vector_gate`: bf16 operands on the tensor
cores with f32 accumulation, f32 decays, exact (overflow-free) intra-chunk decay for any gate.
"""

from __future__ import annotations

import jax

from jla.ops import register
from jla.ops.mosaic_gpu.vector_gate_ops import run, supports

Array = jax.Array

__all__ = ["kda", "supports_kda"]


def kda(
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
    chunk_size: int = 64,
    sub_chunk: int = 16,
) -> tuple[Array, Array | None]:
    """KDA with the arguments and conventions of :func:`jla.ops.reference.kda`.

    ``q, k [B, T, H, K]`` and ``v [B, T, H, V]`` are bf16 with ``K, V`` in ``{64, 128}``;
    ``g [B, T, H, K]`` is the log-decay and ``beta [B, T, H]`` the write strength.
    ``segment_ids [B, T]`` restarts the recurrence at every change of id; ``cu_seqlens`` is not
    supported (as in :mod:`jla.ops.chunked`).
    """
    return run("KDA", q, k, v, g, (beta,), scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, cu_seqlens=cu_seqlens, segment_ids=segment_ids,
               use_qk_l2norm=use_qk_l2norm, chunk_size=chunk_size, sub_chunk=sub_chunk)


def supports_kda(q, k, v, g, beta, **options) -> bool:
    """True for the calls the kernels are tested on (see :func:`vector_gate_ops.supports`)."""
    return supports(q, k, v, g, beta.shape == q.shape[:3], options)


register("mosaic_gpu", "kda", kda, supports_kda, auto=True)
