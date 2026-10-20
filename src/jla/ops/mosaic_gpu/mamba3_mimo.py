# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO (``jla.ops`` kind ``mamba3_mimo``) on Mosaic GPU, with two routes (``kernel=``):

* ``"mimo"`` (default): the dedicated kernels of :mod:`jla.ops.mosaic_gpu.mamba3_mimo_kernels`.
  A chunk is ``64 / R`` tokens times ``R`` ranks (64 rows) and the head keeps one ``[K, V]`` state,
  so the rank-``R`` update is one tensor-core product per chunk: ``R`` times the work of one SISO
  recurrence over the same tokens. Chunks are ``64 / R`` tokens (``chunk_size`` must be that or
  None); ranks in :data:`~jla.ops.mosaic_gpu.mamba3_mimo_kernels.RANKS`.
* ``"siso"``: the previous route, exact by linearity. The MIMO state is the sum of ``R`` SISO
  states, ``S_t = sum_s S_t^(s)``, where ``S^(s)`` is written only by rank ``s`` (both trapezoid
  streams) under the shared decay, and ``S^(0)`` also carries the initial state. So
  ``o_{t,r} = sum_s q_{t,r} S_t^(s)`` is ``R^2`` SISO recurrences (pair ``(r, s)``: queries of rank
  ``r``, keys and values of rank ``s``), run as ``H R^2`` heads of
  :func:`jla.ops.mosaic_gpu.mamba3.mamba3` (chunks of 64 tokens) and summed over ``s`` in float32:
  ``R`` times the work of the dedicated kernels, for any rank.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jla import ops
from jla.ops.mosaic_gpu import mamba3 as siso
from jla.ops.mosaic_gpu import mamba3_mimo_kernels as kernels
from jla.ops.reference import Mamba3State

__all__ = ["KERNELS", "mamba3_mimo", "supports"]

Array = jax.Array
_F32 = jnp.float32

KERNELS = ("mimo", "siso")


def _pairs(q, k, v, g, beta, gamma, initial_state):
    """The ``H R^2`` SISO problem; head ``(h, r, s)`` pairs query rank ``r`` with key/value rank ``s``."""
    batch, length, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    wide = heads * rank * rank
    qq = jnp.broadcast_to(q[:, :, :, :, None], (batch, length, heads, rank, rank, key_dim))
    kk = jnp.broadcast_to(k[:, :, :, None], (batch, length, heads, rank, rank, key_dim))
    vv = jnp.broadcast_to(v[:, :, :, None], (batch, length, heads, rank, rank, value_dim))
    flat = lambda x: x.reshape(batch, length, wide, x.shape[-1])  # noqa: E731
    gates = [jnp.repeat(x, rank * rank, axis=2) for x in (g, beta, gamma)]
    state = None
    if initial_state is not None:
        ssm, key, value = (x.astype(_F32) for x in initial_state)
        first = (jnp.arange(rank) == 0)[:, None, None]  # only s = 0 carries the incoming state
        ssm = jnp.where(first, ssm[:, :, None, None], 0.0)  # [B, H, 1, R, K, V]
        ssm = jnp.broadcast_to(ssm, (batch, heads, rank, rank, key_dim, value_dim))
        key = jnp.broadcast_to(key[:, :, None], (batch, heads, rank, rank, key_dim))
        value = jnp.broadcast_to(value[:, :, None], (batch, heads, rank, rank, value_dim))
        state = Mamba3State(ssm.reshape(batch, wide, key_dim, value_dim), key.reshape(batch, wide, key_dim),
                            value.reshape(batch, wide, value_dim))
    return flat(qq), flat(kk), flat(vv), gates, state


def _siso_route(q, k, v, g, beta, gamma, *, scale, initial_state, output_final_state, segment_ids, chunk_size,
                stages, interpret):
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4]:
        raise ValueError(f"mamba3_mimo needs q, k [B, T, H, R, K] and v [B, T, H, R, V]; got {q.shape}, "
                         f"{k.shape}, {v.shape}")
    batch, length, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    qq, kk, vv, gates, state = _pairs(q, k, v, g, beta, gamma, initial_state)
    o, final = siso.mamba3(qq, kk, vv, *gates, scale=scale, initial_state=state,
                           output_final_state=output_final_state, segment_ids=segment_ids, chunk_size=chunk_size,
                           stages=stages, interpret=interpret)
    o = o.astype(_F32).reshape(batch, length, heads, rank, rank, value_dim).sum(axis=4).astype(q.dtype)
    if not output_final_state:
        return o, None
    ssm = final.ssm.reshape(batch, heads, rank, rank, key_dim, value_dim)[:, :, 0].sum(axis=2)
    key = final.key.reshape(batch, heads, rank, rank, key_dim)[:, :, 0]
    value = final.value.reshape(batch, heads, rank, rank, value_dim)[:, :, 0]
    return o, Mamba3State(ssm, key.astype(_F32), value.astype(_F32))


def mamba3_mimo(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    gamma: Array,
    *,
    scale: float = 1.0,
    initial_state: Mamba3State | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
    chunk_size: int | None = None,
    stages: int | None = None,
    kernel: str = "mimo",
    interpret: bool = False,
) -> tuple[Array, Mamba3State | None]:
    """Mamba-3 MIMO on Mosaic GPU; the arguments of :func:`jla.ops.reference.mamba3_mimo`.

    ``kernel="mimo"`` runs the dedicated kernels (chunks of ``64 / R`` tokens), ``kernel="siso"``
    the ``H R^2``-head SISO route (chunks of 64 tokens); ``chunk_size`` None picks the route's chunk.
    """
    common = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                  segment_ids=segment_ids, stages=stages, interpret=interpret)
    if kernel == "mimo":
        return kernels.mamba3_mimo(q, k, v, g, beta, gamma, chunk_size=chunk_size, **common)
    if kernel == "siso":
        return _siso_route(q, k, v, g, beta, gamma, chunk_size=siso.CHUNK if chunk_size is None else chunk_size,
                           **common)
    raise ValueError(f"kernel must be one of {KERNELS}; got {kernel!r}")


def _siso_supports(q: Array, k: Array, v: Array, g: Array, beta: Array, gamma: Array, *,
                   initial_state: Mamba3State | None = None, chunk_size: int | None = None, **options: Any) -> bool:
    """The SISO kernel's ``supports`` on the ``H R^2``-head problem this call becomes."""
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4]:
        return False
    if any(x.shape != q.shape[:3] for x in (g, beta, gamma)):
        return False
    batch, length, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    wide = heads * rank * rank
    sds = jax.ShapeDtypeStruct
    state = None
    if initial_state is not None:
        want = ((batch, heads, key_dim, value_dim), (batch, heads, rank, key_dim), (batch, heads, rank, value_dim))
        if not isinstance(initial_state, Mamba3State) or tuple(x.shape for x in initial_state) != want:
            return False
        state = Mamba3State(sds((batch, wide, key_dim, value_dim), initial_state.ssm.dtype),
                            sds((batch, wide, key_dim), initial_state.key.dtype),
                            sds((batch, wide, value_dim), initial_state.value.dtype))
    shape_k, shape_v, shape_g = (batch, length, wide, key_dim), (batch, length, wide, value_dim), (batch, length, wide)
    return siso.supports(sds(shape_k, q.dtype), sds(shape_k, k.dtype), sds(shape_v, v.dtype),
                         *(sds(shape_g, x.dtype) for x in (g, beta, gamma)), initial_state=state,
                         chunk_size=siso.CHUNK if chunk_size is None else chunk_size, **options)


def supports(q: Array, k: Array, v: Array, g: Array, beta: Array, gamma: Array, *, kernel: str = "mimo",
             **options: Any) -> bool:
    """The chosen route's ``supports``: the dedicated kernels' or the SISO kernel's on ``H R^2`` heads."""
    if kernel == "mimo":
        return kernels.supports(q, k, v, g, beta, gamma, **options)
    if kernel == "siso":
        return _siso_supports(q, k, v, g, beta, gamma, **options)
    return False


# Eligible for impl="auto": the dedicated kernels are GPU-tested (outputs, final states and gradients
# against the reference) for every call `supports` accepts (ranks 1, 2, 4, 8, every head-dim pair,
# segment_ids, initial states, any length, stages 1 to 4) and are about 10x faster than the chunked path.
ops.register("mosaic_gpu", "mamba3_mimo", mamba3_mimo, supports, auto=True)
