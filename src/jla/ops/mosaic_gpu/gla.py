# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gla/chunk.py and fla/ops/kda/chunk.py
"""Gated linear attention (``kind="gla"``) on Hopper GPUs with Pallas Mosaic GPU kernels.

GLA keeps a log-decay per key channel and writes without the delta rule:
``S_t = diag(exp g_t) S_{t-1} + k_t v_t^T`` and ``o_t = scale q_t^T S_t``.

Two pipelines compute it exactly (f32 states, bf16 tensor-core operands with f32 accumulation,
exact intra-chunk decays for any gate):

* ``pipeline="gla"`` (the default): the dedicated kernels of :mod:`jla.ops.mosaic_gpu.gla_kernels`,
  FLA's ``chunk_gla`` stage split (``Aqk`` only, a state pass fused with the output, a backward
  without any delta-rule stage);
* ``pipeline="vector_gate"``: the first version, kept selectable. GLA is the channel-gated
  recurrence of :mod:`jla.ops.mosaic_gpu.vector_gate` with a zero erase gate and a unit write gate:
  ``Akk`` is then zero, its inverse ``T`` the identity, the WY factors ``w = 0`` and ``u = v``, and
the chunk pipeline (exact intra-chunk decays, f32 states, bf16 tensor-core operands with f32
accumulation) computes GLA and its gradients exactly. The gates are constants built here, so the
gradients are those of ``q, k, v, g`` and the initial state only.

That version reuses the KDA / GDN-2 kernels unchanged and still runs the ``Akk`` / solve / WY
stages (on zeros).

The kernels are registered for ``impl="auto"``: every call :func:`supports_gla` accepts (bf16, ``K, V``
in {64, 128}, ``H * K`` a multiple of 128, ``segment_ids``, initial and final state, any ``T`` up to the
grid limit) is covered by the GPU tests, and the dedicated pipeline is 6.8x to 10.6x faster than the
chunked XLA op it replaces (``benchmarks/gla.md``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jla.ops import register
from jla.ops.mosaic_gpu import gla_kernels
from jla.ops.mosaic_gpu._common import MAX_GRID_YZ
from jla.ops.mosaic_gpu.vector_gate_ops import run, unsupported_reason

Array = jax.Array

__all__ = ["PIPELINES", "gla", "supports_gla"]

# The options of jla.ops.reference.gla plus the chunk geometry of jla.ops.chunked.gla and the
# kernel pipeline.
_GLA_OPTIONS = frozenset({"scale", "initial_state", "output_final_state", "cu_seqlens", "segment_ids",
                          "chunk_size", "sub_chunk", "pipeline"})
PIPELINES = ("gla", "vector_gate")


def _gates(q: Array, v: Array) -> tuple[Array, Array]:
    """The erase gate (zero) and write gate (one) that turn the channel-gated delta rule into GLA."""
    return jnp.zeros(q.shape, jnp.bfloat16), jnp.ones(v.shape, jnp.bfloat16)


def gla(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = 64,
    sub_chunk: int = 16,
    pipeline: str = "gla",
) -> tuple[Array, Array | None]:
    """GLA with the arguments and conventions of :func:`jla.ops.reference.gla`.

    ``q, k [B, T, H, K]`` and ``v [B, T, H, V]`` are bf16 with ``K, V`` in ``{64, 128}`` and
    ``H * K`` a multiple of 128; ``g [B, T, H, K]`` is the log-decay. ``segment_ids [B, T]``
    restarts the recurrence at every change of id; ``cu_seqlens`` is not supported (as in
    :mod:`jla.ops.chunked`). ``o`` has ``q``'s dtype, the final state is f32 ``[B, H, K, V]``.
    ``pipeline`` is ``"gla"`` (dedicated kernels) or ``"vector_gate"`` (the KDA / GDN-2 kernels).
    """
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("mosaic_gpu GLA: q, k, g must be [B, T, H, K] and v [B, T, H, V]")
    if pipeline not in PIPELINES:
        raise ValueError(f"mosaic_gpu GLA: pipeline must be one of {PIPELINES}, got {pipeline!r}")
    if pipeline == "gla":
        return _dedicated(q, k, v, g, scale=scale, initial_state=initial_state,
                          output_final_state=output_final_state, cu_seqlens=cu_seqlens, segment_ids=segment_ids,
                          chunk_size=chunk_size, sub_chunk=sub_chunk)
    erase, write = _gates(q, v)
    return run("GLA", q, k, v, g, (erase, write), scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, cu_seqlens=cu_seqlens, segment_ids=segment_ids,
               use_qk_l2norm=False, chunk_size=chunk_size, sub_chunk=sub_chunk)


def _grid_reason(q: Array) -> str | None:
    """The dedicated kernels' own launch limit (``None`` if within it): ``bwd_inter`` runs one CTA per chunk
    and 64-wide key slice on one CUDA grid dimension (at most 65535 blocks)."""
    blocks = -(-q.shape[1] // gla_kernels.CHUNK) * max(1, q.shape[-1] // 64)
    if blocks > MAX_GRID_YZ:
        return (f"T={q.shape[1]} is too long for one call (CUDA grid limit of {MAX_GRID_YZ} blocks per "
                "chunk-indexed dimension); split the sequence and chain initial_state")
    return None


def _dedicated(q, k, v, g, *, scale, initial_state, output_final_state, cu_seqlens, segment_ids, chunk_size,
               sub_chunk):
    """The ``pipeline="gla"`` body: the checks of the shared entry point, then ``chunked_gla``."""
    if cu_seqlens is not None:
        raise NotImplementedError("the Mosaic GPU GLA kernels take segment_ids, not cu_seqlens")
    if chunk_size != gla_kernels.CHUNK or sub_chunk != gla_kernels.SUB_CHUNK:
        raise ValueError(f"the Mosaic GPU GLA kernels use chunk_size={gla_kernels.CHUNK}, "
                         f"sub_chunk={gla_kernels.SUB_CHUNK} (got {chunk_size}, {sub_chunk})")
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   segment_ids=segment_ids)
    reason = unsupported_reason(q, k, v, g, True, options) or _grid_reason(q)
    if reason is not None:
        raise ValueError(f"mosaic_gpu GLA: {reason}")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final = gla_kernels.chunked_gla(q, k, v, g, scale=scale, initial_state=initial_state,
                                       segment_ids=segment_ids)
    return o, (final if output_final_state else None)


def supports_gla(q, k, v, g, **options) -> bool:
    """True for the calls the kernels are tested on (see :func:`vector_gate_ops.supports`)."""
    if set(options) - _GLA_OPTIONS:
        return False
    pipeline = options.get("pipeline", "gla")
    if pipeline not in PIPELINES:
        return False
    options = {key: value for key, value in options.items() if key != "pipeline"}
    if unsupported_reason(q, k, v, g, q.ndim == 4 and v.ndim == 4, options) is not None:
        return False
    return pipeline != "gla" or _grid_reason(q) is None


register("mosaic_gpu", "gla", gla, supports_gla, auto=True)
