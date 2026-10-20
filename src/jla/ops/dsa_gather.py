# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA's selected-key attention by gathering the selected keys: ``O(T * top_k)`` memory and work, plain JAX.

:func:`gather_attention` has the contract of :func:`jla.ops.sparse_attention.dsa` and of the masked-dense
kernels' :func:`jla.ops.mosaic_gpu.dsa.dsa_attention`: every query head attends to the keys in its row of
``indices [B, T, K]`` (``-1`` = unused slot), and selected keys that are not causal keys of the query's
document are ignored. One difference: the indices of a row must be distinct (a top-k selection always is),
because here a repeated index counts once per slot.

Per chunk of queries the selected key and value rows are gathered (``[B, C, K, H, D]``; bf16/fp16 operands
with float32 accumulation and a float32 softmax, the probabilities rounded to the value dtype for the
``P V`` product as in the kernels), so nothing of size ``T^2`` exists and any head dims work (MLA's
576/512 included). The backward pass is JAX's: each chunk is recomputed (``jax.checkpoint``), and the
gathers' transposes scatter-add the key and value gradients into float32 buffers (one rounding at the end).
This is the selected-key formulation that a gathering kernel (the design of FlashMLA's sparse prefill)
runs, as plain XLA ops: bound by the gathers' memory traffic, so it is no match for such a kernel; see
``benchmarks/dsa.md`` for where it stands against the masked-dense kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops.sparse_attention import _check_heads, _chunked, _gather_keys, _row_chunk, valid_indices

__all__ = ["gather_attention"]

Array = jax.Array


def gather_attention(
    q: Array,
    k: Array,
    v: Array,
    indices: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
    chunk_bytes: int = 512 << 20,
) -> Array:
    """Softmax attention of every query head over its selected keys: ``o [B, T, HQ, E]`` in ``q``'s dtype.

    ``q [B, T, HQ, D]``, ``k [B, T, H, D]``, ``v [B, T, H, E]`` (``HQ`` a multiple of ``H``; query head
    ``h * G + g`` reads key/value head ``h``), ``indices [B, T, K]`` (distinct per row, ``-1`` unused; no
    gradient). A query without a valid selected key gets zeros. ``chunk_bytes`` bounds one chunk's gathered
    operands and scores (the chunk is the largest power of two of query rows within it).
    """
    _check_heads(q, k, v)
    batch, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    value_dim = v.shape[-1]
    if indices.ndim != 3 or indices.shape[:2] != (batch, length):
        raise ValueError(f"indices must be [B, T, K]; got {indices.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    top = indices.shape[-1]
    scale = dim ** -0.5 if scale is None else float(scale)
    indices = lax.stop_gradient(indices).astype(jnp.int32)
    valid = valid_indices(indices, segment_ids)
    safe = jnp.clip(indices, 0, length - 1)
    # The gathers read k and v; their gradients accumulate in these float32 copies (see _gather_keys).
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    @jax.checkpoint
    def chunk_attention(qc, ic, vc):
        rows = qc.shape[1]
        kc = _gather_keys(k_f32, k, ic, k.shape[1:])  # [B, C, K, H, D]
        vg = _gather_keys(v_f32, v, ic, v.shape[1:])  # [B, C, K, H, E]
        qg = qc.reshape(batch, rows, kv_heads, group, dim)
        s = jnp.einsum("bchgd,bckhd->bchgk", qg, kc, preferred_element_type=jnp.float32) * scale
        keep = vc[:, :, None, None, :]
        s = jnp.where(keep, s, -jnp.inf)
        m = lax.stop_gradient(jnp.max(s, axis=-1, keepdims=True))
        m = jnp.where(jnp.isfinite(m), m, 0.0)
        p = jnp.where(keep, jnp.exp(s - m), 0.0)
        total = jnp.sum(p, axis=-1, keepdims=True)
        p = (p / jnp.where(total > 0, total, 1.0)).astype(v.dtype)
        o = jnp.einsum("bchgk,bckhe->bchge", p, vg, preferred_element_type=jnp.float32)
        return o.reshape(batch, rows, q_heads, value_dim).astype(q.dtype)

    itemsize = max(k.dtype.itemsize, v.dtype.itemsize)
    per_row = top * kv_heads * (dim + value_dim) * itemsize + 3 * top * q_heads * 4
    chunk = _row_chunk(length, per_row, budget=chunk_bytes)
    return _chunked(chunk_attention, chunk, q, safe, valid)
