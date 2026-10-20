# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MoonshotAI/MoBA (MIT), moba/moba_naive.py
"""MoBA: Mixture of Block Attention (Lu et al. 2025, arXiv 2502.13189).

Every document is cut into blocks of ``block_size`` tokens counted from its first
token (MoBA's varlen layout). A query attends causally inside its own block and to
the ``top_k - 1`` earlier blocks of its document whose mean-pooled keys score
highest against it (``q . mean(k_block)``, per query head; GQA heads share their
KV head's block means). ``top_k`` counts the query's own block, as upstream does.
The gate has no parameters and passes no gradient. With ``top_k`` at least the
number of blocks the mask is exactly the causal (same-document) mask.

This is the plain JAX implementation: it materializes ``[B, H, T, T]`` gate scores
and logits, like XLA attention, and is the reference for a block-sparse kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["moba_attention", "moba_block_layout", "moba_cached_mask", "moba_mask"]

Array = jax.Array


def moba_block_layout(segment_ids: Array | None, batch: int, length: int,
                      block_size: int) -> tuple[Array, Array, Array, Array]:
    """Per-token ``(document, doc_start, local_block, block_id)``, each ``[B, T]`` int32.

    A document is a maximal run of equal ``segment_ids`` (the whole row when
    ``None``). ``local_block`` counts blocks from the document's first token and
    ``block_id`` numbers all blocks of a row consecutively.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    pos = jnp.arange(length, dtype=jnp.int32)
    if segment_ids is None:
        seg = jnp.zeros((batch, length), jnp.int32)
    else:
        if segment_ids.shape != (batch, length):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
        seg = segment_ids.astype(jnp.int32)
    starts = jnp.concatenate([jnp.ones((batch, 1), bool), seg[:, 1:] != seg[:, :-1]], axis=1)
    document = jnp.cumsum(starts.astype(jnp.int32), axis=1) - 1
    doc_start = jax.lax.cummax(jnp.where(starts, pos[None, :], 0), axis=1)
    local = pos[None, :] - doc_start
    local_block = local // block_size
    block_id = jnp.cumsum((local % block_size == 0).astype(jnp.int32), axis=1) - 1
    return document, doc_start, local_block, block_id


def _check_shapes(q: Array, k: Array, v: Array | None = None) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be [B, T, H, D]; got {q.shape} and {k.shape}")
    if k.shape[:2] != q.shape[:2] or k.shape[3] != q.shape[3] or q.shape[2] % k.shape[2]:
        raise ValueError(f"k {k.shape} does not match q {q.shape} (GQA needs H % Hkv == 0)")
    if v is not None and v.shape[:3] != k.shape[:3]:
        raise ValueError(f"v {v.shape} does not match k {k.shape}")


def moba_mask(q: Array, k: Array, *, block_size: int, top_k: int,
              segment_ids: Array | None = None) -> Array:
    """Boolean ``[B, H, T, T]`` MoBA mask: True where query ``i`` attends key ``j``."""
    _check_shapes(q, k)
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}")
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    document, doc_start, local_block, block_id = moba_block_layout(segment_ids, batch, length, block_size)
    q = jax.lax.stop_gradient(q).astype(jnp.float32)
    k = jax.lax.stop_gradient(k).astype(jnp.float32)

    # Mean-pooled keys of every block (partial last blocks average their own tokens).
    sums = jax.vmap(lambda x, i: jax.ops.segment_sum(x, i, num_segments=length))(k, block_id)
    counts = jax.vmap(lambda i: jax.ops.segment_sum(jnp.ones((length,), jnp.float32), i,
                                                    num_segments=length))(block_id)
    means = sums / jnp.maximum(counts, 1.0)[:, :, None, None]
    token_means = jnp.take_along_axis(means, block_id[:, :, None, None], axis=1)  # [B, T, Hkv, D]
    scores = jnp.einsum("bihgd,bjhd->bhgij", q.reshape(batch, length, kv_heads, group, dim), token_means)
    scores = scores.reshape(batch, heads, length, length)

    # Gate of query i for local block n of its document, read at the block's first token.
    num_blocks = -(-length // block_size)
    n = jnp.arange(num_blocks, dtype=jnp.int32)
    first = jnp.minimum(doc_start[:, :, None] + n * block_size, length - 1)  # [B, T, NB]
    past = n[None, None, :] < local_block[:, :, None]  # strictly earlier blocks of the same document
    gate = jnp.take_along_axis(scores, jnp.broadcast_to(first[:, None], (batch, heads, length, num_blocks)),
                               axis=-1)
    past = jnp.broadcast_to(past[:, None], gate.shape)
    gate = jnp.where(past, gate, -jnp.inf)
    picks = min(top_k - 1, num_blocks)
    if picks > 0:
        _, idx = jax.lax.top_k(gate, picks)  # [B, H, T, picks]
        valid = jnp.take_along_axis(past, idx, axis=-1)
        chosen = jnp.any((idx[..., None] == n) & valid[..., None], axis=-2)  # [B, H, T, NB]
    else:
        chosen = jnp.zeros(gate.shape, bool)

    key_block = jnp.broadcast_to(local_block[:, None, None, :], (batch, heads, length, length))
    selected = jnp.take_along_axis(chosen, key_block, axis=-1)
    selected = selected | (local_block[:, None, :, None] == local_block[:, None, None, :])
    same_doc = (document[:, :, None] == document[:, None, :])[:, None]
    pos = jnp.arange(length)
    causal = (pos[:, None] >= pos[None, :])[None, None]
    return selected & same_doc & causal


def moba_attention(q: Array, k: Array, v: Array, *, block_size: int, top_k: int,
                   segment_ids: Array | None = None, scale: float | None = None) -> Array:
    """MoBA over a full sequence: ``q [B, T, H, D]``, ``k, v [B, T, Hkv, D]`` -> ``[B, T, H, D]``.

    ``segment_ids [B, T]`` keeps every query inside its own document; blocks
    restart at each document's first token.
    """
    _check_shapes(q, k, v)
    mask = moba_mask(q, k, block_size=block_size, top_k=top_k, segment_ids=segment_ids)
    return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=scale, implementation="xla")


def moba_cached_mask(q: Array, keys: Array, q_positions: Array, kv_positions: Array, *,
                     block_size: int, top_k: int) -> Array:
    """Boolean ``[B, H, T, S]`` MoBA mask of queries at ``q_positions [B, T]`` over a KV cache.

    The cache holds one document per row with ``slot = position`` (``kv_positions``
    -1 = empty), so a slot's block is ``position // block_size``. Blocks before a
    query's own are complete; they are gated by the mean of their cached keys,
    which reproduces the full-sequence selection of :func:`moba_mask`.
    """
    if block_size < 1 or top_k < 1:
        raise ValueError("block_size and top_k must be >= 1")
    batch, length, heads, dim = q.shape
    size, kv_heads = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    q = jax.lax.stop_gradient(q).astype(jnp.float32)
    keys = jax.lax.stop_gradient(keys).astype(jnp.float32)
    num_blocks = -(-size // block_size)
    filled = kv_positions >= 0
    key_block = jnp.clip(kv_positions // block_size, 0, num_blocks - 1)
    weight = filled.astype(jnp.float32)
    sums = jax.vmap(lambda x, i: jax.ops.segment_sum(x, i, num_segments=num_blocks))(
        keys * weight[:, :, None, None], key_block)
    counts = jax.vmap(lambda w, i: jax.ops.segment_sum(w, i, num_segments=num_blocks))(weight, key_block)
    means = sums / jnp.maximum(counts, 1.0)[:, :, None, None]
    gate = jnp.einsum("bthgd,bnhd->bhgtn", q.reshape(batch, length, kv_heads, group, dim), means)
    gate = gate.reshape(batch, heads, length, num_blocks)
    q_block = q_positions // block_size
    n = jnp.arange(num_blocks, dtype=jnp.int32)
    past = (n[None, None, :] < q_block[:, :, None]) & (counts[:, None, :] > 0)
    past = jnp.broadcast_to(past[:, None], gate.shape)
    gate = jnp.where(past, gate, -jnp.inf)
    picks = min(top_k - 1, num_blocks)
    if picks > 0:
        _, idx = jax.lax.top_k(gate, picks)
        valid = jnp.take_along_axis(past, idx, axis=-1)
        chosen = jnp.any((idx[..., None] == n) & valid[..., None], axis=-2)  # [B, H, T, NB]
    else:
        chosen = jnp.zeros(gate.shape, bool)
    slot_block = jnp.broadcast_to(key_block[:, None, None, :], (batch, heads, length, size))
    selected = jnp.take_along_axis(chosen, slot_block, axis=-1)
    selected = selected | (key_block[:, None, None, :] == q_block[:, None, :, None])
    visible = filled[:, None, None, :] & (kv_positions[:, None, None, :] <= q_positions[:, None, :, None])
    return selected & visible
