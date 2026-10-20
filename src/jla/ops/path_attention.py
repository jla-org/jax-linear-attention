# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/path_attn/naive.py and fla/ops/path_attn/parallel.py
"""PaTH attention: softmax attention with data-dependent Householder positions (arXiv 2505.16381).

Each token ``s`` carries a Householder reflection ``H_s = I - beta_s w_s w_s^T``
(``w_s`` unit-norm, ``beta_s`` in ``(0, 2)``). Query ``i`` scores key ``j <= i`` with

    logit[i, j] = scale * (q_i^T H_i H_{i-1} ... H_{j+1} k_j + G_i - G_j)

where ``G`` is the cumulative sum of the optional log forget gate ``g`` (FoX); the
scale multiplies the gate term too, as in FLA's kernels. ``beta = 0`` gives
plain causal softmax attention (NoPE). Shapes: ``q [B, T, H, D]``,
``k, w [B, T, Hkv, D]``, ``v [B, T, Hkv, Dv]``, ``beta [B, T, Hkv]``,
``g [B, T, H]`` (or ``None``); ``H`` must be a multiple of ``Hkv``.

Two exact implementations:

* :func:`path_attention_reference`: token by token, as decoding does: every new
  token reflects all cached keys, ``k_j <- H_t k_j``, then queries them.
* :func:`path_attention_chunked`: FLA's chunked form. Inside a chunk the product
  of reflections is ``I - W^T T W_beta`` with the UT transform
  ``T = (I + strict_lower(W_beta W^T))^{-1}``; queries are transported to the
  chunk start and keys to the chunk end, and each earlier chunk ``J`` multiplies
  the query by its whole-chunk reflection ``I - W_J^T T_J W_beta,J``.

Both materialize the ``[B, H, T, T]`` scores in fp32 (no flash-style tiling yet)
and restrict queries to keys of their own document with ``segment_ids [B, T]``
(documents are contiguous runs, so every reflection between ``j`` and ``i`` lies
in the same document).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["path_attention", "path_attention_chunked", "path_attention_reference"]

Array = jax.Array


def _check(q: Array, k: Array, v: Array, w: Array, beta: Array, g: Array | None,
           segment_ids: Array | None) -> tuple[int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or w.ndim != 4:
        raise ValueError("q, k, v and w must be [B, T, heads, dim]")
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    if k.shape != (batch, length, kv_heads, dim) or w.shape != k.shape:
        raise ValueError(f"k and w must be [B, T, Hkv, {dim}]; got {k.shape} and {w.shape}")
    if v.shape[:3] != (batch, length, kv_heads):
        raise ValueError(f"v must be [B, T, {kv_heads}, Dv]; got {v.shape}")
    if heads % kv_heads:
        raise ValueError(f"query heads {heads} must be a multiple of key heads {kv_heads}")
    if beta.shape != (batch, length, kv_heads):
        raise ValueError(f"beta must be {(batch, length, kv_heads)}; got {beta.shape}")
    if g is not None and g.shape != (batch, length, heads):
        raise ValueError(f"g must be {(batch, length, heads)}; got {g.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be {(batch, length)}; got {segment_ids.shape}")
    return heads, kv_heads


def _to_query_heads(x: Array, groups: int) -> Array:
    """``[B, T, Hkv, ...] -> [B, T, H, ...]``: query head ``h`` uses key head ``h // groups``."""
    return x if groups == 1 else jnp.repeat(x, groups, axis=2)


def _softmax_output(scores: Array, v: Array, g: Array | None, segment_ids: Array | None,
                    scale: float, out_dtype) -> Array:
    """``scores [B, H, T, T]`` (fp32, the Householder-transported ``q.k``) -> ``o [B, T, H, Dv]``."""
    length = scores.shape[-1]
    if g is not None:
        cum = jnp.cumsum(g.astype(jnp.float32), axis=1).transpose(0, 2, 1)  # [B, H, T]
        scores = scores + cum[..., :, None] - cum[..., None, :]
    scores = scores * scale
    mask = jnp.tril(jnp.ones((length, length), bool))[None, None]
    if segment_ids is not None:  # a document starts wherever the id changes (ids may be reused)
        change = jnp.concatenate([jnp.zeros_like(segment_ids[:, :1], bool),
                                  segment_ids[:, 1:] != segment_ids[:, :-1]], axis=1)
        docs = jnp.cumsum(change.astype(jnp.int32), axis=1)
        mask = mask & (docs[:, :, None] == docs[:, None, :])[:, None]
    scores = jnp.where(mask, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bhij,bjhe->bihe", probs, v.astype(jnp.float32))
    return o.astype(out_dtype)


def path_attention_reference(
    q: Array,
    k: Array,
    v: Array,
    w: Array,
    beta: Array,
    g: Array | None = None,
    *,
    segment_ids: Array | None = None,
    scale: float | None = None,
) -> Array:
    """Token-by-token PaTH attention (the decoding recurrence over a key cache)."""
    heads, kv_heads = _check(q, k, v, w, beta, g, segment_ids)
    groups = heads // kv_heads
    batch, length, _, dim = q.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    f32 = jnp.float32
    qf = q.astype(f32)
    kf, wf = (_to_query_heads(x.astype(f32), groups) for x in (k, w))
    bf = _to_query_heads(beta.astype(f32), groups)

    def step(cache, xs):  # cache [B, H, T, D]: keys 0..t-1 reflected up to token t-1, zeros after
        t, q_t, k_t, w_t, b_t = xs  # [B, H, D] x3, [B, H]
        proj = jnp.einsum("bhjd,bhd->bhj", cache, w_t)
        cache = cache - (b_t[..., None] * proj)[..., None] * w_t[:, :, None, :]
        cache = lax.dynamic_update_slice_in_dim(cache, k_t[:, :, None, :], t, axis=2)
        return cache, jnp.einsum("bhd,bhjd->bhj", q_t, cache)

    xs = (jnp.arange(length), *(x.transpose(1, 0, 2, 3) for x in (qf, kf, wf)), bf.transpose(1, 0, 2))
    _, scores = lax.scan(step, jnp.zeros((batch, heads, length, dim), f32), xs)  # [T, B, H, T]
    scores = scores.transpose(1, 2, 0, 3)
    return _softmax_output(scores, _to_query_heads(v, groups), g, segment_ids, scale, v.dtype)


def path_attention_chunked(
    q: Array,
    k: Array,
    v: Array,
    w: Array,
    beta: Array,
    g: Array | None = None,
    *,
    segment_ids: Array | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
) -> Array:
    """PaTH attention through FLA's chunked UT-transform form (exact, any ``T``)."""
    heads, kv_heads = _check(q, k, v, w, beta, g, segment_ids)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    groups = heads // kv_heads
    batch, length, _, dim = q.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    f32 = jnp.float32
    c = chunk_size
    n = -(-length // c)
    pad = n * c - length

    def blocks(x):  # [B, T, H, D] -> [B, H, N, C, D]; zero padding (beta = 0 there: H = I)
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        return x.transpose(0, 2, 1, 3).reshape(batch, heads, n, c, x.shape[-1])

    qb = blocks(q.astype(f32))
    kb, wb = (blocks(_to_query_heads(x.astype(f32), groups)) for x in (k, w))
    bb = blocks(_to_query_heads(beta.astype(f32), groups)[..., None])[..., 0]  # [B, H, N, C]
    wbeta = wb * bb[..., None]

    eye = jnp.eye(c, dtype=f32)
    strict = jnp.tril(jnp.ones((c, c), bool), -1)
    incl = jnp.tril(jnp.ones((c, c), bool))
    # UT transform: T = (I + strict_lower(W_beta W^T))^{-1}, so prod_{s=c-1..0} H_s = I - W^T T W_beta.
    lower = jnp.where(strict, jnp.einsum("...id,...jd->...ij", wbeta, wb), 0.0)
    t_mat = lax.linalg.triangular_solve(eye + lower, jnp.broadcast_to(eye, lower.shape), left_side=True,
                                        lower=True, unit_diagonal=True)
    t_wbeta = t_mat @ wbeta  # [B, H, N, C, D]
    # Keys reflected to the chunk end: k_hat_j = H_{c-1} ... H_{j+1} k_j.
    t_wbk = t_mat @ jnp.where(strict, jnp.einsum("...rd,...jd->...rj", wbeta, kb), 0.0)  # [.., s, j]
    k_hat = kb - jnp.einsum("...sj,...sd->...jd", t_wbk, wb)
    # Queries reflected to the chunk start: q_hat_i^T = q_i^T H_i ... H_0.
    qw = jnp.where(incl, jnp.einsum("...id,...sd->...is", qb, wb), 0.0)
    q_hat = qb - qw @ t_wbeta
    # Intra-chunk scores q_i^T H_i ... H_{j+1} k_j for j <= i.
    local = jnp.where(incl, jnp.einsum("...id,...jd->...ij", qb, kb) - qw @ t_wbk, 0.0)
    # Whole-chunk reflection I - W^T T W_beta, applied to row vectors: x <- x - x @ h_mat.
    h_mat = jnp.einsum("...sd,...se->...de", wb, t_wbeta)  # [B, H, N, D, D]

    def shift(x, d):  # x[..., I - d, ...] at chunk I (zeros for I < d)
        return jnp.roll(x, d, axis=2)

    def body(carry, d):  # carry: queries transported back to the start of chunk I - d + 1
        q_t = carry
        k_prev = jnp.where((jnp.arange(n) >= d)[:, None, None], shift(k_hat, d), 0.0)
        block = jnp.einsum("bhnid,bhnjd->bhnij", q_t, k_prev)
        h_prev = shift(h_mat, d)
        q_t = q_t - jnp.einsum("bhnid,bhnde->bhnie", q_t, h_prev)
        return q_t, block

    if n > 1:
        _, cross = lax.scan(body, q_hat, jnp.arange(1, n))  # [N - 1, B, H, N, C, C], offset d = 1..N-1
        diag = jnp.concatenate([local[None], cross], axis=0)  # [N, B, H, N(I), C, C] by offset
    else:
        diag = local[None]
    diag = jnp.moveaxis(diag, 0, 2)  # [B, H, N(d), N(I), C, C]
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    offset = jnp.clip(rows - cols, 0, n - 1)
    full = diag[:, :, offset, jnp.broadcast_to(rows, (n, n))]  # [B, H, N(I), N(J), C, C]
    full = jnp.where((rows >= cols)[None, None, :, :, None, None], full, 0.0)
    scores = full.transpose(0, 1, 2, 4, 3, 5).reshape(batch, heads, n * c, n * c)[:, :, :length, :length]
    return _softmax_output(scores, _to_query_heads(v, groups), g, segment_ids, scale, v.dtype)


def path_attention(
    q: Array,
    k: Array,
    v: Array,
    w: Array,
    beta: Array,
    g: Array | None = None,
    *,
    segment_ids: Array | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
    implementation: str = "chunked",
) -> Array:
    """PaTH attention by ``implementation``: ``"chunked"`` (default, ``"auto"``) or ``"reference"``."""
    if implementation in ("auto", "chunked"):
        return path_attention_chunked(q, k, v, w, beta, g, segment_ids=segment_ids, scale=scale,
                                      chunk_size=chunk_size)
    if implementation == "reference":
        return path_attention_reference(q, k, v, w, beta, g, segment_ids=segment_ids, scale=scale)
    raise ValueError(f"PaTH attention has no {implementation!r} implementation (use chunked or reference)")
