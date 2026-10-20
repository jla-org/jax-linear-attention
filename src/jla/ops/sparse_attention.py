# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/nsa/naive.py, fla/ops/dsa/naive.py and
# fla/ops/utils/pooling.py; the DSA indexer follows deepseek-ai/DeepSeek-V3.2-Exp (MIT), inference/model.py.
"""Native Sparse Attention (NSA) and DeepSeek Sparse Attention (DSA) in plain JAX.

Both ops take ``q [B, T, HQ, D]``, ``k [B, T, H, D]``, ``v [B, T, H, E]`` (grouped
query heads, ``HQ % H == 0``) and optional ``segment_ids [B, T]``: a document is a
maximal run of equal ids, and no query sees a key of another document. Every
softmax runs in float32 over an explicit ``[T, T]`` mask, so these ops cost dense
attention; they define the math the sparse kernels must reproduce.

NSA (FLA's formulation): keys and values are mean-pooled over blocks of
``block_size`` tokens counted from each document's start. The compressed branch
attends to the blocks that are complete at the query, the selected branch attends
to the raw tokens of the ``block_count`` blocks with the highest compressed
attention probability (averaged over the query heads that share a KV head; the
document's first block, the query's own block and the one before it are always
selected), and the sliding-window branch attends to the last ``window_size``
tokens. Learned per-head gates mix the three branch outputs.

DSA: the lightning indexer scores ``I[t, s] = sum_j w[t, j] relu(scale q_idx[t, j] . k_idx[s])``
(one index key per token shared by the index heads), every query keeps its
``top_k`` best causal keys of its document, and all attention heads attend only
to those keys.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "dsa",
    "dsa_attention_target",
    "dsa_index_scores",
    "dsa_select",
    "attach_indexer_grad",
    "nsa",
]

Array = jax.Array


def _documents(segment_ids: Array | None, batch: int, length: int) -> tuple[Array, Array]:
    """Run-numbered document ids ``[B, T]`` and each token's document start ``[B, T]``."""
    t = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    if segment_ids is None:
        return jnp.zeros((batch, length), jnp.int32), jnp.zeros((batch, length), jnp.int32)
    if segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    resets = jnp.concatenate(
        [jnp.ones((batch, 1), bool), segment_ids[:, 1:] != segment_ids[:, :-1]], axis=1)
    doc = jnp.cumsum(resets.astype(jnp.int32), axis=1)
    start = jax.lax.cummax(jnp.where(resets, t, 0), axis=1)
    return doc, start


def _causal_document_mask(doc: Array) -> Array:
    """``[B, T, T]``: key ``s`` is visible to query ``t`` (same document, ``s <= t``)."""
    length = doc.shape[1]
    causal = jnp.arange(length)[:, None] >= jnp.arange(length)[None, :]
    return causal[None] & (doc[:, :, None] == doc[:, None, :])


def _check_heads(q: Array, k: Array, v: Array) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be [B, T, heads, dim]")
    if k.shape[:3] != v.shape[:3] or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"incompatible q {q.shape}, k {k.shape}, v {v.shape}")
    if k.shape[2] == 0 or q.shape[2] % k.shape[2]:
        raise ValueError(f"query heads {q.shape[2]} must be a multiple of key/value heads {k.shape[2]}")


def _attend(q: Array, k: Array, v: Array, mask: Array, scale: float) -> tuple[Array, Array]:
    """Masked GQA softmax attention in float32; rows without a visible key give zeros.

    ``mask`` is ``[B, H or 1, T, S]``. Returns ``o [B, T, HQ, E]`` and the
    probabilities ``[B, H, G, T, S]`` (query head ``h * G + g`` uses KV head ``h``).
    """
    batch, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    qg = q.astype(jnp.float32).reshape(batch, length, kv_heads, q_heads // kv_heads, dim)
    s = jnp.einsum("bthgd,bshd->bhgts", qg, k.astype(jnp.float32)) * scale
    s = jnp.where(mask[:, :, None], s, -jnp.inf)
    m = jax.lax.stop_gradient(jnp.max(s, axis=-1, keepdims=True))
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m)
    total = jnp.sum(p, axis=-1, keepdims=True)
    p = p / jnp.where(total > 0, total, 1.0)
    o = jnp.einsum("bhgts,bshe->bthge", p, v.astype(jnp.float32))
    return o.reshape(batch, length, q_heads, v.shape[-1]), p


def nsa(
    q: Array,
    k: Array,
    v: Array,
    g_cmp: Array,
    g_slc: Array,
    g_swa: Array | None,
    *,
    block_size: int,
    block_count: int,
    window_size: int = 0,
    scale: float | None = None,
    segment_ids: Array | None = None,
    return_block_starts: bool = False,
) -> Array | tuple[Array, Array]:
    """Native Sparse Attention: ``g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa``.

    Gates are ``[B, T, HQ]`` (already squashed, e.g. sigmoid). ``window_size = 0``
    disables the sliding-window branch. With ``return_block_starts`` also returns the
    selected blocks as ``[B, T, H, min(block_count, T)]`` token positions of their
    first token (``-1`` for unused slots).
    """
    _check_heads(q, k, v)
    if block_size < 1 or block_count < 1 or window_size < 0:
        raise ValueError("NSA needs block_size >= 1, block_count >= 1 and window_size >= 0")
    batch, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    for name, gate in (("g_cmp", g_cmp), ("g_slc", g_slc), ("g_swa", g_swa)):
        if gate is not None and gate.shape != (batch, length, q_heads):
            raise ValueError(f"{name} must be [B, T, HQ] = {(batch, length, q_heads)}; got {gate.shape}")
    if window_size > 0 and g_swa is None:
        raise ValueError("window_size > 0 needs the g_swa gate")
    scale = dim ** -0.5 if scale is None else float(scale)

    doc, start = _documents(segment_ids, batch, length)
    pos = jnp.arange(length, dtype=jnp.int32)[None]
    local = pos - start
    block_start = pos - local % block_size  # first token of each token's block, [B, T]
    is_start = local % block_size == 0
    visible = _causal_document_mask(doc)  # [B, T(query), T(key or slot)]

    # Compressed keys and values: block means stored at each block's first-token slot.
    def pool(x, ids):
        total = jax.ops.segment_sum(x.astype(jnp.float32), ids, num_segments=length)
        count = jax.ops.segment_sum(jnp.ones((length,), jnp.float32), ids, num_segments=length)
        return total / jnp.maximum(count, 1.0)[:, None, None]

    k_cmp = jax.vmap(pool)(k, block_start)
    v_cmp = jax.vmap(pool)(v, block_start)
    # A block is visible to the compressed branch once its last token is in the past.
    complete = visible & is_start[:, None, :] & (pos[0][:, None] - pos[0][None, :] >= block_size - 1)[None]
    o_cmp, p_cmp = _attend(q, k_cmp, v_cmp, complete[:, None], scale)

    # Block scores: compressed probabilities averaged over each KV head's query group.
    scores = jax.lax.stop_gradient(jnp.mean(p_cmp, axis=2))  # [B, H, T, T]
    slot = pos[0][None, None, :]
    own = block_start[:, :, None]
    # Forced blocks outrank every probability (<= 1); when block_count < 3 the own block
    # wins, then the previous one, then the document's first block (FLA leaves this tie open).
    previous = (slot == own - block_size) & (local[:, :, None] >= block_size)
    first = slot == start[:, :, None]
    forced = (slot == own) | previous | first
    priority = jnp.where(slot == own, 4.0, jnp.where(previous, 3.0, 2.0))
    scores = jnp.where(forced[:, None], priority[:, None], scores)
    allowed = (complete | forced)[:, None]
    n_select = min(block_count, length)
    top, slots = jax.lax.top_k(jnp.where(allowed, scores, -jnp.inf), n_select)  # [B, H, T, S]
    slots = jnp.where(jnp.isfinite(top), slots, length)
    b_idx = jnp.arange(batch)[:, None, None, None]
    h_idx = jnp.arange(kv_heads)[None, :, None, None]
    t_idx = jnp.arange(length)[None, None, :, None]
    chosen = jnp.zeros((batch, kv_heads, length, length + 1), bool).at[b_idx, h_idx, t_idx, slots].set(True)
    token_block = jnp.broadcast_to(block_start[:, None, None, :], (batch, kv_heads, length, length))
    selected = jnp.take_along_axis(chosen, token_block, axis=-1) & visible[:, None]
    o_slc, _ = _attend(q, k, v, selected, scale)

    o = o_cmp * g_cmp[..., None].astype(jnp.float32) + o_slc * g_slc[..., None].astype(jnp.float32)
    if window_size > 0:
        in_window = visible & (pos[0][:, None] - pos[0][None, :] < window_size)[None]
        o_swa, _ = _attend(q, k, v, in_window[:, None], scale)
        o = o + o_swa * g_swa[..., None].astype(jnp.float32)
    o = o.astype(q.dtype)
    if return_block_starts:
        starts = jnp.where(slots == length, -1, slots).transpose(0, 2, 1, 3)
        return o, starts
    return o


def dsa_index_scores(
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
) -> Array:
    """Lightning-indexer scores ``[B, T, T]`` in float32, ``-inf`` where the key is not visible.

    ``q_idx [B, T, HI, DI]``, ``k_idx [B, T, DI]`` (shared by the index heads),
    ``w_idx [B, T, HI]``; ``scale`` defaults to ``DI ** -0.5``.
    """
    if q_idx.ndim != 4 or k_idx.ndim != 3 or w_idx.ndim != 3:
        raise ValueError("expected q_idx [B, T, HI, DI], k_idx [B, T, DI] and w_idx [B, T, HI]")
    batch, length, heads, dim = q_idx.shape
    if k_idx.shape != (batch, length, dim) or w_idx.shape != (batch, length, heads):
        raise ValueError(f"incompatible q_idx {q_idx.shape}, k_idx {k_idx.shape}, w_idx {w_idx.shape}")
    scale = dim ** -0.5 if scale is None else float(scale)
    s = jnp.einsum("bthd,bsd->bths", q_idx.astype(jnp.float32), k_idx.astype(jnp.float32)) * scale
    scores = jnp.einsum("bths,bth->bts", jax.nn.relu(s), w_idx.astype(jnp.float32))
    doc, _ = _documents(segment_ids, batch, length)
    return jnp.where(_causal_document_mask(doc), scores, -jnp.inf)


def dsa_select(scores: Array, top_k: int) -> Array:
    """The ``min(top_k, T)`` best keys per query ``[B, T, K]``; ``-1`` marks unused slots."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    top, indices = jax.lax.top_k(jax.lax.stop_gradient(scores), min(top_k, scores.shape[-1]))
    return jnp.where(jnp.isfinite(top), indices, -1)


def _selection_mask(indices: Array, length: int) -> Array:
    batch = indices.shape[0]
    slots = jnp.where(indices < 0, length, indices)
    b_idx = jnp.arange(batch)[:, None, None]
    t_idx = jnp.arange(length)[None, :, None]
    return jnp.zeros((batch, length, length + 1), bool).at[b_idx, t_idx, slots].set(True)[..., :length]


def dsa(
    q: Array,
    k: Array,
    v: Array,
    indices: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
    return_probs: bool = False,
) -> Array | tuple[Array, Array, Array]:
    """Softmax attention of every query head over the keys in ``indices [B, T, K]`` (``-1`` = unused).

    Selected keys that are not causal keys of the query's document are ignored.
    With ``return_probs`` also returns the attention probabilities ``[B, H, G, T, T]``
    and the boolean selection mask ``[B, T, T]``.
    """
    _check_heads(q, k, v)
    batch, length = q.shape[:2]
    if indices.ndim != 3 or indices.shape[:2] != (batch, length):
        raise ValueError(f"indices must be [B, T, K]; got {indices.shape}")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    doc, _ = _documents(segment_ids, batch, length)
    selected = _selection_mask(indices, length) & _causal_document_mask(doc)
    o, p = _attend(q, k, v, selected[:, None], scale)
    o = o.astype(q.dtype)
    if return_probs:
        return o, p, selected
    return o


def dsa_attention_target(probs: Array) -> Array:
    """DeepSeek-V3.2's indexer target: attention probabilities summed over heads, L1-normalized."""
    total = jnp.sum(jax.lax.stop_gradient(probs), axis=(1, 2))
    return total / jnp.maximum(jnp.sum(total, axis=-1, keepdims=True), 1e-30)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def attach_indexer_grad(o: Array, scores: Array, target: Array, selected: Array, coef: float) -> Array:
    """Identity on ``o`` whose backward adds ``coef * grad KL(target || softmax(scores))`` to ``scores``.

    The KL runs over the selected keys of each query (``selected [B, T, T]``, float
    0/1), as in DeepSeek-V3.2's sparse training stage, and is SUMMED over queries to
    match ``jla.training.step``, which differentiates the summed token loss and then
    divides every gradient by the number of target tokens: there ``coef`` weights the
    per-query mean KL against the mean LM loss. The gradient does not depend on the
    upstream cotangent, and the reported loss does not include the KL.
    """
    return o


def _attach_fwd(o, scores, target, selected, coef):
    return o, (scores, target, selected)


def _attach_bwd(coef, residuals, g):
    scores, target, selected = residuals
    keep = selected > 0
    logits = jnp.where(keep, scores.astype(jnp.float32), -jnp.inf)
    pred = jnp.where(keep, jax.nn.softmax(logits, axis=-1), 0.0)
    d_scores = coef * (pred - jnp.where(keep, target, 0.0))
    return g, d_scores.astype(scores.dtype), jnp.zeros_like(target), jnp.zeros_like(selected)


attach_indexer_grad.defvjp(_attach_fwd, _attach_bwd)


# ----------------------------------------------------------------------------------------------
# Plain-JAX helpers of the selected-key paths (jla.ops.dsa_gather and jla.ops.mosaic_gpu.dsa)
# ----------------------------------------------------------------------------------------------


def valid_indices(indices: Array, segment_ids: Array | None = None) -> Array:
    """``[B, T, K]`` bool: the slot holds a causal key of the query's document (``-1`` and invalid: False)."""
    batch, length, _ = indices.shape
    doc, _ = _documents(segment_ids, batch, length)
    t = jnp.arange(length, dtype=jnp.int32)[None, :, None]
    safe = jnp.clip(indices, 0, length - 1)
    key_doc = jnp.take_along_axis(doc[:, None, :], safe.reshape(batch, 1, -1), axis=2).reshape(indices.shape)
    return (indices >= 0) & (indices <= t) & (key_doc == doc[:, :, None])


def _row_chunk(length: int, bytes_per_row: int, budget: int = 256 << 20) -> int:
    """Query rows per chunk so that one chunk's gathered operands stay within ``budget`` bytes."""
    rows = max(8, budget // max(bytes_per_row, 1))
    return min(length, 1 << (rows.bit_length() - 1))


def _chunked(fn, chunk: int, *xs):
    """``fn`` over row chunks of the ``[B, T, ...]`` arrays ``xs`` (``lax.map``), concatenated back to ``[B, T, ...]``."""
    batch, length = xs[0].shape[:2]
    n = -(-length // chunk)
    pad = n * chunk - length

    def split(x):
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2))
        return x.reshape(batch, n, chunk, *x.shape[2:]).swapaxes(0, 1)

    out = lax.map(lambda args: fn(*args), tuple(split(x) for x in xs))
    out = out.swapaxes(0, 1).reshape(batch, n * chunk, *out.shape[3:])
    return out[:, :length] if pad else out


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _gather_keys(table_f32: Array, table: Array, rows: Array, shape: tuple[int, ...]) -> Array:
    """``table[b, rows[b]]`` (``[B, C, K, ...]``), read from the bf16/fp16 ``table``; its backward scatter-adds
    into the float32 ``table_f32 [B, *shape]``, so a key's gradient sums every query that selected it in
    float32 (one rounding at the end) instead of in the low precision of ``table``."""
    del table_f32
    return jax.vmap(lambda t, r: t[r])(table, rows)


def _gather_keys_fwd(table_f32, table, rows, shape):
    return _gather_keys(table_f32, table, rows, shape), rows


def _gather_keys_bwd(shape, rows, ct):
    def scatter(r, c):
        return jnp.zeros(shape, jnp.float32).at[r].add(c.astype(jnp.float32))

    return jax.vmap(scatter)(rows, ct), None, None


_gather_keys.defvjp(_gather_keys_fwd, _gather_keys_bwd)
