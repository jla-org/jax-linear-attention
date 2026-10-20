# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Implemented from arXiv 2605.22791 (equations only)
"""Gated DeltaNet-2 (``kind="gdn2"``) on Hopper GPUs with Pallas Mosaic GPU kernels.

Gated Delta Rule-2 (Eq. 10 of arXiv 2605.22791): ``S_t = (I - k_t (b_t * k_t)^T) diag(exp g_t)
S_{t-1} + k_t (w_t * v_t)^T`` with a channel-wise erase gate ``b [B, T, H, K]`` and write gate
``w [B, T, H, V]``. It runs the chunkwise WY form of Eq. 18-25 with the gate-aware backward of
Eq. 26-28 on the same kernels as KDA (:mod:`jla.ops.mosaic_gpu.vector_gate`): the erase gate
enters the read side of ``Akk`` and the ``w`` factor, the write gate the ``u`` factor. The gates
are rounded to bf16 for the kernels (their gradients are returned in the gates' dtype).
"""

from __future__ import annotations

import jax

from jla.ops import register
from jla.ops.mosaic_gpu.vector_gate_ops import run, supports

Array = jax.Array

__all__ = ["gdn2", "supports_gdn2"]


def gdn2(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    b: Array,
    w: Array,
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
    """GDN-2 with the arguments and conventions of :func:`jla.ops.reference.gdn2`.

    ``q, k [B, T, H, K]`` and ``v [B, T, H, V]`` are bf16 with ``K, V`` in ``{64, 128}``;
    ``g [B, T, H, K]`` is the log-decay, ``b [B, T, H, K]`` the erase and ``w [B, T, H, V]`` the
    write gate. ``segment_ids [B, T]`` restarts the recurrence at every change of id;
    ``cu_seqlens`` is not supported (as in :mod:`jla.ops.chunked`).
    """
    return run("GDN-2", q, k, v, g, (b, w), scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, cu_seqlens=cu_seqlens, segment_ids=segment_ids,
               use_qk_l2norm=use_qk_l2norm, chunk_size=chunk_size, sub_chunk=sub_chunk)


def supports_gdn2(q, k, v, g, b, w, **options) -> bool:
    """True for the calls the kernels are tested on (see :func:`vector_gate_ops.supports`)."""
    return supports(q, k, v, g, b.shape == q.shape and w.shape == v.shape, options)


register("mosaic_gpu", "gdn2", gdn2, supports_gdn2, auto=True)
