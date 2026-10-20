# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Follows the chunk-stage decomposition of flash-linear-attention (MIT): fla/ops/utils/cumsum.py,
# fla/ops/common/chunk_scaled_dot_kkt.py, fla/ops/utils/solve_tril.py, fla/ops/gated_delta_rule/wy_fast.py,
# fla/ops/common/chunk_delta_h.py, fla/ops/common/chunk_o.py.
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Chunk-parallel forms of the linear-attention family in plain JAX.

The functions follow the stage decomposition of flash-linear-attention's
chunk kernels (``chunk_gated_delta_rule``, ``chunk_kda``, ``chunk_gla``), so
every fast kernel can be checked against the exact stage it replaces. Inside a
chunk ``G`` is the cumulative log-decay, one value per head (scalar gates:
simple GLA, DeltaNet, Gated DeltaNet) or one per key channel (vector gates:
GLA, KDA):

1. ``chunk_local_cumsum``: ``G`` inside each chunk;
2. ``chunk_scaled_dot_kkt``: ``A[i, j] = beta_i sum_c k_ic k_jc exp(G_ic - G_jc)`` for ``j < i``;
3. ``solve_unit_lower``: ``T = (I + A)^-1``;
4. ``recompute_w_u``: ``w = T (beta exp(G) k)`` and ``u = T (beta v)``;
5. ``chunk_state_pass``: serial over chunks, ``v_new = u - w S`` and
   ``S <- exp(G_last) S + (k exp(G_last - G))^T v_new``;
6. ``chunk_output``: ``o = scale ((q exp(G)) S + P v_new)`` with
   ``P[i, j] = sum_c q_ic k_jc exp(G_ic - G_jc)`` for ``j <= i``.

Without the delta rule (``simple_gla``, ``gla``) stages 2-4 vanish: ``w = 0``
and ``u = v``. Vector-gated scores are built from ``sub_chunk`` blocks: an
off-diagonal block uses the first decay of its row block as the reference, so
both exponential factors are at most one, and diagonal blocks use exact
per-channel differences. No exponential can overflow for any gate value.

Packed documents use ``segment_ids [B, T]`` (the state restarts wherever the
id changes, as in ``jla.ops.reference``). Three masks make the chunked form
exact: scores only connect tokens of the same segment run, a token sees the
incoming chunk state only if it belongs to the run that was active at the end
of the previous chunk (``carry_in``), and the state handed to the next chunk
keeps only the chunk's last run (``tail``). Within a run the cumulative decay
differences are unaffected by where a chunk starts, so no rescaling is needed.
``cu_seqlens`` packing is not supported here; use ``segment_ids`` or the
reference.

Gradients come from JAX autodiff. Inputs are computed in float32; ``precision``
sets the matmul precision (HIGHEST by default, so the results track
``jla.ops.reference`` closely).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops.reference import Mamba3State, l2norm, segment_resets

__all__ = [
    "chunk_local_cumsum",
    "chunk_output",
    "chunk_scaled_dot_kkt",
    "chunk_state_pass",
    "decayed_scores",
    "delta_rule",
    "gated_delta_rule",
    "gdn2",
    "gla",
    "kda",
    "mamba3",
    "recompute_w_u",
    "segment_masks",
    "simple_gla",
    "solve_unit_lower",
]

Array = jax.Array
Precision = lax.Precision | str | None

DEFAULT_CHUNK = 64
DEFAULT_SUB_CHUNK = 16


def _to_chunks(x: Array, chunk: int) -> Array:
    """``[B, T, H, ...]`` with ``T % chunk == 0`` to ``[B, NC, H, C, ...]``."""
    batch, seq_len, heads = x.shape[:3]
    x = x.reshape(batch, seq_len // chunk, chunk, heads, *x.shape[3:])
    return jnp.moveaxis(x, 2, 3)


def _from_chunks(x: Array) -> Array:
    """``[B, NC, H, C, ...]`` back to ``[B, T, H, ...]``."""
    x = jnp.moveaxis(x, 3, 2)
    batch, num_chunks, chunk, heads = x.shape[:4]
    return x.reshape(batch, num_chunks * chunk, heads, *x.shape[4:])


def _is_vector_gate(g_cum: Array, k: Array) -> bool:
    return g_cum.ndim == k.ndim


class SegmentMasks(NamedTuple):
    """Per-chunk masks for packed documents (all broadcast over heads)."""

    same: Array  # [B, NC, 1, C, C] token i and j belong to the same segment run
    carry_in: Array  # [B, NC, 1, C] token sees the state entering the chunk
    tail: Array  # [B, NC, 1, C] token belongs to the chunk's last run


def segment_masks(segment_ids: Array, chunk: int, pad: int) -> SegmentMasks:
    """Build :class:`SegmentMasks` from ``segment_ids [B, T]`` (``pad`` inert tokens appended)."""
    runs = jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)
    runs = jnp.pad(runs, [(0, 0), (0, pad)], mode="edge")  # padding joins the last run
    batch = runs.shape[0]
    runs = runs.reshape(batch, -1, chunk)[:, :, None, :]  # [B, NC, 1, C]
    # The run active when each chunk starts: token 0's run for the first chunk,
    # otherwise the previous chunk's last token.
    entering = jnp.concatenate([runs[:, :1, :, :1], runs[:, :-1, :, -1:]], axis=1)
    return SegmentMasks(
        same=runs[..., :, None] == runs[..., None, :],
        carry_in=runs == entering,
        tail=runs == runs[..., -1:],
    )


def _masked_exp(exponent: Array, mask: Array) -> Array:
    """``exp`` on ``mask``, zero elsewhere, without ever evaluating ``exp`` off the mask.

    Masking the exponent first keeps overflowing acausal entries from producing
    ``inf`` (and ``0 * inf = nan`` in matmuls or gradients).
    """
    return jnp.exp(jnp.where(mask, exponent, -jnp.inf))


def decayed_scores(
    a: Array,
    b: Array,
    g_cum: Array,
    *,
    inclusive: bool,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = None,
) -> Array:
    """Causal ``P[i, j] = sum_c a_ic b_jc exp(G_ic - G_jc)`` for each chunk.

    Args:
      a, b: ``[..., C, K]``.
      g_cum: ``[..., C]`` (one decay per head) or ``[..., C, K]`` (per channel).
      inclusive: keep the diagonal (``j <= i``) or not (``j < i``).
      sub_chunk: block size for the stable vector-gate construction.

    Returns:
      ``[..., C, C]``, zero above (or on) the diagonal.
    """
    chunk = a.shape[-2]
    causal = jnp.tril(jnp.ones((chunk, chunk), bool), 0 if inclusive else -1)
    if not _is_vector_gate(g_cum, a):
        scores = jnp.einsum("...ik,...jk->...ij", a, b, precision=precision)
        return scores * _masked_exp(g_cum[..., :, None] - g_cum[..., None, :], causal)

    if chunk % sub_chunk:
        raise ValueError(f"chunk size {chunk} must be a multiple of sub_chunk {sub_chunk}")
    *lead, _, key_dim = a.shape
    blocks = chunk // sub_chunk

    def split(x):
        return x.reshape(*lead, blocks, sub_chunk, key_dim)

    a_b, b_b, g_b = split(a), split(b), split(g_cum)
    g_ref = g_b[..., :, 0, :]  # first decay of each row block: [..., I, K]
    # Off-diagonal blocks J < I: exp(G_i - G_ref_I) <= 1 and exp(G_ref_I - G_j) <= 1.
    rows = a_b * jnp.exp(g_b - g_ref[..., :, None, :])
    below = jnp.tril(jnp.ones((blocks, blocks), bool), -1)[:, :, None, None]
    cols = b_b[..., None, :, :, :] * _masked_exp(
        g_ref[..., :, None, None, :] - g_b[..., None, :, :, :], below
    )  # [..., I, J, BC, K]
    off = jnp.einsum("...ick,...ijdk->...ijcd", rows, cols, precision=precision)
    # Diagonal blocks: exact per-channel differences.
    tri = jnp.tril(jnp.ones((sub_chunk, sub_chunk), bool), 0 if inclusive else -1)[:, :, None]
    weights = _masked_exp(g_b[..., :, None, :] - g_b[..., None, :, :], tri)  # [..., I, BC, BC, K]
    diag = jnp.einsum("...ik,...jk,...ijk->...ij", a_b, b_b, weights, precision=precision)
    on_diag = jnp.eye(blocks, dtype=bool)[:, :, None, None]
    tiles = jnp.where(on_diag, diag[..., :, None, :, :], off)  # [..., I, J, BC, BC]
    return jnp.swapaxes(tiles, -3, -2).reshape(*lead, chunk, chunk)


def chunk_local_cumsum(g: Array) -> Array:
    """Cumulative log-decay inside each chunk.

    ``g`` is ``[B, NC, H, C]`` (scalar gates) or ``[B, NC, H, C, K]`` (vector gates).
    """
    return jnp.cumsum(g, axis=3)


def chunk_scaled_dot_kkt(
    k: Array,
    beta: Array | None,
    g_cum: Array,
    *,
    erase: Array | None = None,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = None,
) -> Array:
    """Strictly lower ``A[i, j] = beta_i sum_c k_ic k_jc exp(G_ic - G_jc)`` per chunk.

    With a channel-wise erase gate ``erase`` (``[..., C, K]``, GDN-2) the read
    side uses ``erase_i * k_i`` instead of ``beta_i k_i``.
    """
    if erase is not None:
        return decayed_scores(k * erase, k, g_cum, inclusive=False, sub_chunk=sub_chunk, precision=precision)
    scores = decayed_scores(k, k, g_cum, inclusive=False, sub_chunk=sub_chunk, precision=precision)
    return scores * beta[..., :, None]


def solve_unit_lower(a: Array) -> Array:
    """``(I + A)^-1`` for strictly lower ``A`` (``[..., C, C]``)."""
    chunk = a.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(chunk, dtype=a.dtype), a.shape)
    return jax.scipy.linalg.solve_triangular(eye + a, eye, lower=True, unit_diagonal=True)


def _key_decay(g_cum: Array, k: Array) -> Array:
    """``exp(G)`` shaped to multiply ``k`` (``[..., C, K]``)."""
    decay = jnp.exp(g_cum)
    return decay if _is_vector_gate(g_cum, k) else decay[..., None]


def recompute_w_u(
    k: Array,
    v: Array,
    beta: Array | None,
    g_cum: Array,
    t: Array,
    *,
    erase: Array | None = None,
    write: Array | None = None,
    carry_in: Array | None = None,
    precision: Precision = None,
) -> tuple[Array, Array]:
    """WY factors ``w = T (beta exp(G) k)`` and ``u = T (beta v)``.

    GDN-2 passes channel-wise ``erase`` (``[..., C, K]``) and ``write``
    (``[..., C, V]``) gates instead of ``beta``: ``w = T (erase exp(G) k)`` and
    ``u = T (write v)``. With ``carry_in`` (``[..., C]`` bool), rows of tokens
    that start from a fresh state get no correction from the incoming state.
    """
    erase_scale = beta[..., None] if erase is None else erase
    write_scale = beta[..., None] if write is None else write
    rhs = k * _key_decay(g_cum, k) * erase_scale
    if carry_in is not None:
        rhs = rhs * carry_in[..., None]
    w = jnp.einsum("...ij,...jk->...ik", t, rhs, precision=precision)
    u = jnp.einsum("...ij,...jv->...iv", t, v * write_scale, precision=precision)
    return w, u


def chunk_state_pass(
    k: Array,
    w: Array | None,
    u: Array,
    g_cum: Array,
    initial_state: Array,
    *,
    masks: SegmentMasks | None = None,
    precision: Precision = None,
) -> tuple[Array, Array, Array]:
    """Carry the state across chunks.

    Args:
      k, w: ``[B, NC, H, C, K]`` (``w`` is ``None`` without the delta rule).
      u: ``[B, NC, H, C, V]``.
      g_cum: ``[B, NC, H, C]`` or ``[B, NC, H, C, K]``.
      initial_state: ``[B, H, K, V]`` float32.
      masks: packed-document masks; the state reaching the next chunk then
        holds only the chunk's last segment run.

    Returns:
      ``(h, v_new, final)``: ``h[:, n]`` is the state entering chunk ``n``
      (``[B, NC, H, K, V]``), ``v_new`` the values actually written
      (``[B, NC, H, C, V]``), and ``final`` the state after the last chunk.
    """
    vector = _is_vector_gate(g_cum, k)

    def step(state, chunk):
        k_n, w_n, u_n, g_n, tail_n, keep_n = chunk
        v_new = u_n
        if w_n is not None:
            v_new = u_n - jnp.einsum("bhck,bhkv->bhcv", w_n, state, precision=precision)
        if vector:
            g_last = g_n[..., -1, :]  # [B, H, K]
            k_decayed = k_n * jnp.exp(g_last[..., None, :] - g_n)
            decayed = state * jnp.exp(g_last)[..., :, None]
        else:
            g_last = g_n[..., -1]  # [B, H]
            k_decayed = k_n * jnp.exp(g_last[..., None] - g_n)[..., None]
            decayed = state * jnp.exp(g_last)[..., None, None]
        if tail_n is not None:
            k_decayed = k_decayed * tail_n[..., None]  # [B, 1, C, 1]
            decayed = decayed * keep_n[..., None, None]  # [B, 1, 1, 1]
        next_state = decayed + jnp.einsum("bhck,bhcv->bhkv", k_decayed, v_new, precision=precision)
        return next_state, (state, v_new)

    # The incoming state survives a chunk only if the chunk's last token still
    # belongs to the run that was active when the chunk started.
    tail = None if masks is None else masks.tail
    keep = None if masks is None else masks.carry_in[..., -1]  # [B, NC, 1]
    chunks = tuple(None if x is None else jnp.moveaxis(x, 1, 0) for x in (k, w, u, g_cum, tail, keep))
    final, (h, v_new) = lax.scan(step, initial_state, chunks)
    return jnp.moveaxis(h, 0, 1), jnp.moveaxis(v_new, 0, 1), final


def chunk_output(
    q: Array,
    k: Array,
    v_new: Array,
    h: Array,
    g_cum: Array,
    scale: float,
    *,
    masks: SegmentMasks | None = None,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = None,
) -> Array:
    """``o = scale ((q exp(G)) h + P v_new)`` per chunk (``P`` from ``decayed_scores``)."""
    q_decayed = q * _key_decay(g_cum, q)
    if masks is not None:
        q_decayed = q_decayed * masks.carry_in[..., None]
    inter = jnp.einsum("...ck,...kv->...cv", q_decayed, h, precision=precision)
    scores = decayed_scores(q, k, g_cum, inclusive=True, sub_chunk=sub_chunk, precision=precision)
    if masks is not None:
        scores = scores * masks.same
    intra = jnp.einsum("...ij,...jv->...iv", scores, v_new, precision=precision)
    return scale * (inter + intra)


def _chunked(
    q: Array,
    k: Array,
    v: Array,
    g: Array | None,
    beta: Array | None,
    *,
    delta: bool,
    scale: float | None,
    initial_state: Array | None,
    output_final_state: bool,
    cu_seqlens: Array | None,
    use_qk_l2norm: bool,
    chunk_size: int,
    sub_chunk: int,
    precision: Precision,
    segment_ids: Array | None,
    erase: Array | None = None,
    write: Array | None = None,
) -> tuple[Array, Array | None]:
    if cu_seqlens is not None:
        raise NotImplementedError(
            "chunked ops do not support cu_seqlens; pass segment_ids or use jla.ops.reference"
        )
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = key_dim**-0.5
    if g is None:
        g = jnp.zeros((batch, seq_len, heads), f32)
    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)

    # Zero padding is inert: k = 0 writes nothing and g = 0 does not decay.
    pad = (-seq_len) % chunk_size

    def prep(x):
        x = x.astype(f32)
        return _to_chunks(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)), chunk_size)

    masks = None
    if segment_ids is not None:
        if segment_ids.shape != (batch, seq_len):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
        masks = segment_masks(segment_ids, chunk_size, pad)

    q_c, k_c, v_c = prep(q), prep(k), prep(v)
    g_cum = chunk_local_cumsum(prep(g))
    if delta:
        beta_c, erase_c, write_c = (None if x is None else prep(x) for x in (beta, erase, write))
        a = chunk_scaled_dot_kkt(k_c, beta_c, g_cum, erase=erase_c, sub_chunk=sub_chunk, precision=precision)
        if masks is not None:
            a = a * masks.same
        w, u = recompute_w_u(
            k_c, v_c, beta_c, g_cum, solve_unit_lower(a), erase=erase_c, write=write_c,
            carry_in=None if masks is None else masks.carry_in, precision=precision,
        )
    else:
        w, u = None, v_c
    h, v_new, final = chunk_state_pass(
        k_c, w, u, g_cum, initial_state.astype(f32), masks=masks, precision=precision
    )
    o = chunk_output(q_c, k_c, v_new, h, g_cum, scale, masks=masks, sub_chunk=sub_chunk, precision=precision)
    o = _from_chunks(o)[:, :seq_len].astype(out_dtype)
    return o, (final if output_final_state else None)


def simple_gla(
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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked linear attention with one log-decay per head."""
    return _chunked(
        q, k, v, g, None, delta=False, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=False,
        chunk_size=chunk_size, sub_chunk=DEFAULT_SUB_CHUNK, precision=precision, segment_ids=segment_ids,
    )


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
    chunk_size: int = DEFAULT_CHUNK,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked gated linear attention with a log-decay per key channel."""
    return _chunked(
        q, k, v, g, None, delta=False, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=False,
        chunk_size=chunk_size, sub_chunk=sub_chunk, precision=precision, segment_ids=segment_ids,
    )


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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked DeltaNet."""
    return _chunked(
        q, k, v, None, beta, delta=True, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size, sub_chunk=DEFAULT_SUB_CHUNK, precision=precision, segment_ids=segment_ids,
    )


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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked Gated DeltaNet."""
    return _chunked(
        q, k, v, g, beta, delta=True, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size, sub_chunk=DEFAULT_SUB_CHUNK, precision=precision, segment_ids=segment_ids,
    )


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
    chunk_size: int = DEFAULT_CHUNK,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked Kimi Delta Attention."""
    return _chunked(
        q, k, v, g, beta, delta=True, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size, sub_chunk=sub_chunk, precision=precision, segment_ids=segment_ids,
    )


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
    chunk_size: int = DEFAULT_CHUNK,
    sub_chunk: int = DEFAULT_SUB_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked Gated DeltaNet-2: the KDA chunk form with channel-wise erase/write gates.

    Follows the chunkwise WY algorithm of arXiv 2605.22791 (Eq. 18-25): the erase
    gate enters the read side of ``A`` and the WY ``w`` factor, the write gate
    enters ``u``.
    """
    return _chunked(
        q, k, v, g, None, delta=True, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size, sub_chunk=sub_chunk, precision=precision, segment_ids=segment_ids,
        erase=b, write=w,
    )


def mamba3(
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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Mamba3State | None]:
    """Chunked Mamba-3 SISO: simple GLA with two write streams per token.

    Token ``j`` writes ``k_j^T (gamma_j v_j)`` and ``k_{j-1}^T (beta_j v_{j-1})``
    after the decay ``exp(g_j)``. Both streams share the decay and the masks, so
    the state pass runs over the two streams concatenated along the chunk axis
    and the output adds one score matrix per stream.
    """
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if initial_state is None:
        initial_state = Mamba3State(
            jnp.zeros((batch, heads, key_dim, value_dim), f32),
            jnp.zeros((batch, heads, key_dim), f32),
            jnp.zeros((batch, heads, value_dim), f32),
        )
    q32, k32, v32 = (x.astype(f32) for x in (q, k, v))
    k_prev = jnp.concatenate([initial_state.key.astype(f32)[:, None], k32[:, :-1]], axis=1)
    v_prev = jnp.concatenate([initial_state.value.astype(f32)[:, None], v32[:, :-1]], axis=1)
    masks = None
    pad = (-seq_len) % chunk_size
    if segment_ids is not None:
        if segment_ids.shape != (batch, seq_len):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
        fresh = segment_resets(segment_ids)[..., None, None]
        k_prev = jnp.where(fresh, 0.0, k_prev)  # no previous-token write across documents
        masks = segment_masks(segment_ids, chunk_size, pad)

    def prep(x):
        x = x.astype(f32)
        return _to_chunks(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)), chunk_size)

    q_c, k_c, kp_c = prep(q32), prep(k32), prep(k_prev)
    cur_c = prep(v32 * gamma.astype(f32)[..., None])
    prev_c = prep(v_prev * beta.astype(f32)[..., None])
    g_cum = chunk_local_cumsum(prep(g))

    both = functools.partial(jnp.concatenate, axis=3)
    stream_masks = None
    if masks is not None:
        stream_masks = SegmentMasks(same=masks.same, carry_in=both([masks.carry_in] * 2), tail=both([masks.tail] * 2))
    h, _, final = chunk_state_pass(
        both([k_c, kp_c]), None, both([cur_c, prev_c]), both([g_cum, g_cum]),
        initial_state.ssm.astype(f32), masks=stream_masks, precision=precision,
    )
    q_decayed = q_c * _key_decay(g_cum, q_c)
    if masks is not None:
        q_decayed = q_decayed * masks.carry_in[..., None]
    o = jnp.einsum("...ck,...kv->...cv", q_decayed, h, precision=precision)
    for keys, values in ((k_c, cur_c), (kp_c, prev_c)):
        scores = decayed_scores(q_c, keys, g_cum, inclusive=True, precision=precision)
        if masks is not None:
            scores = scores * masks.same
        o = o + jnp.einsum("...ij,...jv->...iv", scores, values, precision=precision)
    o = (scale * _from_chunks(o))[:, :seq_len].astype(out_dtype)
    if not output_final_state:
        return o, None
    return o, Mamba3State(final, k32[:, -1], v32[:, -1])


# mLSTM (xLSTM): chunkwise-parallel form derived from arXiv 2405.04517 / 2503.13427 (equations only).
def mlstm(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    i: Array,
    *,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state=None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
):
    """Chunkwise-parallel mLSTM, equal to :func:`jla.ops.reference.mlstm` (same stabiliser).

    Unrolling the recurrence, ``m_t`` is the largest log weight of every write
    that reaches token ``t``: ``b_t + m_in`` for the state entering the chunk and
    ``b_t - b_j + i_j`` for token ``j <= t`` of the chunk (``b`` the in-chunk
    cumulative log forget gate). Each chunk stabilises its scores with that
    row maximum; a ``lax.scan`` over chunks carries ``(c, n, m)``.
    """
    from jla.ops.reference import MLSTM_EMPTY_M, MLSTMState, check_mlstm_inputs, mlstm_empty_state, mlstm_floor

    check_mlstm_inputs(q, k, v, g, i, initial_state, cu_seqlens, segment_ids)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    scale = key_dim**-0.5 if scale is None else scale
    f32 = jnp.float32
    chunk = chunk_size
    pad = (-seq_len) % chunk
    n_chunks = (seq_len + pad) // chunk
    empty = MLSTM_EMPTY_M

    def chunks(x, fill=0.0):  # [B, T, H, ...] -> [B, NC, H, C, ...]
        x = jnp.pad(x.astype(f32), [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), constant_values=fill)
        x = x.reshape(batch, n_chunks, chunk, *x.shape[2:])
        return jnp.moveaxis(x, 2, 3)

    qc, kc, vc = chunks(q) * scale, chunks(k), chunks(v)
    gc, ic = chunks(g), chunks(i, fill=empty)  # padding: no decay, no write
    if segment_ids is None:
        same = jnp.ones((1, 1, 1, chunk, chunk), bool)
        carry_in = tail = jnp.ones((1, 1, 1, chunk), bool)
    else:
        same, carry_in, tail = segment_masks(segment_ids, chunk, pad)
    causal = jnp.tril(jnp.ones((chunk, chunk), bool))
    b = jnp.cumsum(gc, axis=-1)  # [B, NC, H, C]

    # Chunk summaries for the state pass (writes of the chunk's last segment run).
    w_end = jnp.where(tail, b[..., -1:] - b + ic, empty)
    m_loc = jnp.max(w_end, axis=-1)
    p_end = jnp.exp(w_end - m_loc[..., None])
    c_loc = jnp.einsum("bnhc,bnhck,bnhcv->bnhkv", p_end, kc, vc, precision=precision)
    n_loc = jnp.einsum("bnhc,bnhck->bnhk", p_end, kc, precision=precision)
    decay = jnp.where(carry_in[..., -1], b[..., -1], empty)  # [B, NC, H]: the entering state survives?

    state0 = (
        mlstm_empty_state(batch, heads, key_dim, value_dim)
        if initial_state is None
        else MLSTMState(*(x.astype(f32) for x in initial_state))
    )

    def pass_chunk(carry, xs):
        c, n, m = carry
        a, m_l, c_l, n_l = xs
        m_new = jnp.maximum(a + m, m_l)
        keep, add = jnp.exp(a + m - m_new), jnp.exp(m_l - m_new)
        new = MLSTMState(
            keep[..., None, None] * c + add[..., None, None] * c_l, keep[..., None] * n + add[..., None] * n_l, m_new
        )
        return new, carry

    xs = tuple(jnp.moveaxis(x, 1, 0) for x in (decay, m_loc, c_loc, n_loc))
    final, entering = lax.scan(pass_chunk, state0, xs)
    c_in, n_in, m_in = (jnp.moveaxis(x, 0, 1) for x in entering)  # [B, NC, H, ...]

    # Outputs: inter-chunk term from the entering state plus the stabilised intra-chunk scores.
    w0 = jnp.where(carry_in, b + m_in[..., None], empty)  # [B, NC, H, C]
    w = jnp.where(causal & same, b[..., :, None] - b[..., None, :] + ic[..., None, :], empty)
    m_t = jnp.maximum(w0, jnp.max(w, axis=-1))
    scores = jnp.einsum("bnhck,bnhdk->bnhcd", qc, kc, precision=precision) * jnp.exp(w - m_t[..., None])
    inter = jnp.exp(w0 - m_t)
    num = jnp.einsum("bnhcd,bnhdv->bnhcv", scores, vc, precision=precision)
    num = num + inter[..., None] * jnp.einsum("bnhck,bnhkv->bnhcv", qc, c_in, precision=precision)
    den = jnp.sum(scores, axis=-1) + inter * jnp.einsum("bnhck,bnhk->bnhc", qc, n_in, precision=precision)
    o = num / (jnp.maximum(jnp.abs(den), mlstm_floor(m_t)) + eps)[..., None]
    o = jnp.moveaxis(o, 3, 2).reshape(batch, n_chunks * chunk, heads, value_dim)[:, :seq_len]
    return o.astype(q.dtype), (final if output_final_state else None)


__all__ += ["mlstm"]


def mamba1(
    q: Array,
    k: Array,
    v: Array,
    dt: Array,
    a: Array,
    *,
    d: Array | None = None,
    z: Array | None = None,
    dt_bias: Array | None = None,
    dt_softplus: bool = False,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = 64,
) -> tuple[Array, Array | None]:
    """Mamba-1 selective scan (``jla.ops.reference.mamba1``) as a chunked associative scan.

    The recurrence ``h_t = a_t h_{t-1} + b_t`` with ``a_t = exp(delta_t A)`` (zero at
    a document start) and ``b_t = delta_t u_t B_t`` is a linear scan: a sequential
    ``lax.scan`` over chunks carries ``h [B, D, N]`` and each chunk runs
    ``lax.associative_scan`` over its ``chunk_size`` tokens. The chunk body is wrapped in
    ``jax.checkpoint``, so under ``jax.grad`` only the carried ``[B, D, N]`` state per chunk
    is saved and one chunk's ``[B, chunk, D, N]`` states are rebuilt in the backward pass
    (without it, the scan keeps every chunk's residuals: about ten full-sequence
    ``[B, T, D, N]`` float32 tensors). Padding tokens have ``delta = 0`` (``a = 1``,
    ``b = 0``) and leave the state unchanged.

    On a GPU only the forward pass is this scan: the gradient is the reference op's
    (:func:`jla.ops.reference.mamba1`, recomputed in the backward pass), because
    differentiating this scan on the H100 gives wrong gradients for some inputs (cause
    not found; the input is in ``tests/test_mosaic_gpu_mamba1.py``). Elsewhere the
    gradient is this scan's own.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
    options = (dt_softplus, scale, chunk_size)
    scan = _mamba1_reference_grad if _mamba1_differentiates_via_reference() else _mamba1_scan
    o, final = scan(options, q, k, v, dt, a, d, z, dt_bias, initial_state, cu_seqlens, segment_ids)
    return o, final if output_final_state else None


def _mamba1_differentiates_via_reference() -> bool:
    """True on a GPU, where :func:`mamba1` takes the reference op's gradient."""
    return jax.default_backend() == "gpu"


def _mamba1_scan(options, q, k, v, dt, a, d, z, dt_bias, initial_state, cu_seqlens, segment_ids):
    """The chunked scan of :func:`mamba1`; returns ``(o, final_state)``."""
    from jla.ops.reference import mamba1_inputs, mamba1_output

    dt_softplus, scale, chunk_size = options
    u, delta, a32, b, c = mamba1_inputs(
        q, k, v, dt, a, dt_bias=dt_bias, dt_softplus=dt_softplus, scale=scale, cu_seqlens=cu_seqlens,
        segment_ids=segment_ids, initial_state=initial_state,
    )
    batch, seq_len, dim = u.shape
    dstate = a32.shape[1]
    resets = jnp.zeros((batch, seq_len), bool) if segment_ids is None else segment_resets(segment_ids)
    chunk = min(chunk_size, seq_len)
    num_chunks = -(-seq_len // chunk)
    pad = num_chunks * chunk - seq_len

    def chunks(x: Array) -> Array:  # [B, T, ...] -> [num_chunks, B, chunk, ...]
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        x = x.reshape(batch, num_chunks, chunk, *x.shape[2:])
        return jnp.moveaxis(x, 1, 0)

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, a_r * b_l + b_r

    def body(h, xs):
        u_c, delta_c, b_c, c_c, reset_c = xs
        decay = jnp.exp(delta_c[..., None] * a32)  # [B, L, D, N]
        decay = jnp.where(reset_c[:, :, None, None], 0.0, decay)
        write = (delta_c * u_c)[..., None] * b_c
        decay_cum, write_cum = lax.associative_scan(combine, (decay, write), axis=1)
        states = decay_cum * h[:, None] + write_cum
        return states[:, -1], jnp.sum(states * c_c, axis=-1)

    state0 = (jnp.zeros((batch, dim, dstate), jnp.float32) if initial_state is None
              else initial_state.astype(jnp.float32))
    final, y = lax.scan(jax.checkpoint(body), state0, tuple(chunks(x) for x in (u, delta, b, c, resets)))
    y = jnp.moveaxis(y, 0, 1).reshape(batch, num_chunks * chunk, dim)[:, :seq_len]
    return mamba1_output(y, u, d, z, v.dtype), final


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _mamba1_reference_grad(options, q, k, v, dt, a, d, z, dt_bias, initial_state, cu_seqlens, segment_ids):
    """:func:`_mamba1_scan` with the reference op's gradient (the GPU path of :func:`mamba1`)."""
    return _mamba1_scan(options, q, k, v, dt, a, d, z, dt_bias, initial_state, cu_seqlens, segment_ids)


def _mamba1_reference_grad_fwd(options, *inputs):
    return _mamba1_scan(options, *inputs), inputs


def _mamba1_reference_grad_bwd(options, inputs, cts):
    from jla.ops.reference import mamba1 as reference_mamba1

    *diff, cu_seqlens, segment_ids = inputs
    dt_softplus, scale, _ = options

    def scan(q, k, v, dt, a, d, z, dt_bias, initial_state):
        return reference_mamba1(q, k, v, dt, a, d=d, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus, scale=scale,
                                initial_state=initial_state, output_final_state=True, cu_seqlens=cu_seqlens,
                                segment_ids=segment_ids)

    _, vjp = jax.vjp(scan, *diff)
    return (*vjp(cts), None, None)


_mamba1_reference_grad.defvjp(_mamba1_reference_grad_fwd, _mamba1_reference_grad_bwd)


# Gated DeltaProduct, adapted from flash-linear-attention (MIT),
# fla/ops/gated_delta_product/chunk.py and fla/layers/gated_deltaproduct.py.


def householder_interleave(
    q: Array, g: Array, segment_ids: Array | None, num_householder: int
) -> tuple[Array, Array, Array | None]:
    """Spread Gated DeltaProduct over a ``T * n_h`` Gated DeltaNet sequence (FLA's construction).

    Token ``t`` becomes sub-steps ``t * n_h + j``: the decay ``g_t`` sits on
    sub-step 0 (zero elsewhere), the query on sub-step ``n_h - 1`` (zero
    elsewhere) and the segment id repeats. The delta rule then runs the ``n_h``
    Householder steps with ``k``, ``v``, ``beta`` already in ``[B, T*n_h, ...]``
    layout, and ``o[:, n_h - 1 :: n_h]`` is the DeltaProduct output.
    """
    n = num_householder
    batch, seq_len = q.shape[:2]

    def spread(x: Array, slot: int) -> Array:
        out = jnp.zeros((batch, seq_len, n, *x.shape[2:]), x.dtype).at[:, :, slot].set(x)
        return out.reshape(batch, seq_len * n, *x.shape[2:])

    seg = None if segment_ids is None else jnp.repeat(segment_ids, n, axis=1)
    return spread(q, n - 1), spread(g, 0), seg


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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Array | None]:
    """Chunked Gated DeltaProduct: :func:`gated_delta_rule` on the interleaved ``T * n_h`` sequence."""
    from jla.ops.reference import householder_steps

    n = householder_steps(q, k, v, g, beta, num_householder)
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_i, g_i, seg_i = householder_interleave(q, g, segment_ids, n)
    o, final = gated_delta_rule(
        q_i, k, v, g_i, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, segment_ids=seg_i, chunk_size=chunk_size, precision=precision,
    )
    return o[:, n - 1 :: n], final


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
    chunk_size: int = DEFAULT_CHUNK,
    precision: Precision = lax.Precision.HIGHEST,
) -> tuple[Array, Mamba3State | None]:
    """Chunked Mamba-3 MIMO (``q, k [B, T, H, R, K]``, ``v [B, T, H, R, V]``; see ``reference.mamba3_mimo``).

    The :func:`mamba3` chunk form with ``R`` keys and values per token: the state
    pass runs over the ``C * R`` (token, rank) writes of both trapezoid streams, and
    the intra-chunk scores ``q_{i,r} . k_{j,s} exp(G_i - G_j)`` are causal over
    tokens (``j <= i``) and dense over ranks, because every rank of token ``i``
    reads the state after all ranks of token ``i`` were written.
    """
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4]:
        raise ValueError(f"mamba3_mimo needs q, k [B, T, H, R, K] and v [B, T, H, R, V]; got {q.shape}, "
                         f"{k.shape}, {v.shape}")
    batch, seq_len, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if initial_state is None:
        initial_state = Mamba3State(
            jnp.zeros((batch, heads, key_dim, value_dim), f32),
            jnp.zeros((batch, heads, rank, key_dim), f32),
            jnp.zeros((batch, heads, rank, value_dim), f32),
        )
    q32, k32, v32 = (x.astype(f32) for x in (q, k, v))
    k_prev = jnp.concatenate([initial_state.key.astype(f32)[:, None], k32[:, :-1]], axis=1)
    v_prev = jnp.concatenate([initial_state.value.astype(f32)[:, None], v32[:, :-1]], axis=1)
    masks = None
    pad = (-seq_len) % chunk_size
    if segment_ids is not None:
        if segment_ids.shape != (batch, seq_len):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
        fresh = segment_resets(segment_ids)[..., None, None, None]
        k_prev = jnp.where(fresh, 0.0, k_prev)  # no previous-token write across documents
        masks = segment_masks(segment_ids, chunk_size, pad)

    def prep(x):
        x = x.astype(f32)
        return _to_chunks(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)), chunk_size)

    def flat(x):  # [B, NC, H, C, R, D] -> [B, NC, H, C * R, D] (token-major, rank-minor)
        return x.reshape(*x.shape[:3], x.shape[3] * x.shape[4], x.shape[5])

    q_c, k_c, kp_c = prep(q32), prep(k32), prep(k_prev)  # [B, NC, H, C, R, K]
    cur_c = prep(v32 * gamma.astype(f32)[..., None, None])  # [B, NC, H, C, R, V]
    prev_c = prep(v_prev * beta.astype(f32)[..., None, None])
    g_cum = chunk_local_cumsum(prep(g))  # [B, NC, H, C]
    g_sub = jnp.repeat(g_cum, rank, axis=3)  # [B, NC, H, C * R]

    both = functools.partial(jnp.concatenate, axis=3)
    stream_masks = None
    if masks is not None:
        carry_sub, tail_sub = (jnp.repeat(m, rank, axis=-1) for m in (masks.carry_in, masks.tail))
        stream_masks = SegmentMasks(same=masks.same, carry_in=both([carry_sub] * 2), tail=both([tail_sub] * 2))
    h, _, final = chunk_state_pass(
        both([flat(k_c), flat(kp_c)]), None, both([flat(cur_c), flat(prev_c)]), both([g_sub, g_sub]),
        initial_state.ssm.astype(f32), masks=stream_masks, precision=precision,
    )
    q_decayed = q_c * jnp.exp(g_cum)[..., None, None]
    if masks is not None:
        q_decayed = q_decayed * masks.carry_in[..., None, None]
    o = jnp.einsum("...crk,...kv->...crv", q_decayed, h, precision=precision)
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size), bool))
    mask = causal if masks is None else causal & masks.same
    decay = _masked_exp(g_cum[..., :, None] - g_cum[..., None, :], mask)  # [B, NC, H, C, C]
    for keys, values in ((k_c, cur_c), (kp_c, prev_c)):
        scores = jnp.einsum("...irk,...jsk->...irjs", q_c, keys, precision=precision)
        scores = scores * decay[..., :, None, :, None]
        o = o + jnp.einsum("...irjs,...jsv->...irv", scores, values, precision=precision)
    o = (scale * _from_chunks(o))[:, :seq_len].astype(out_dtype)
    if not output_final_state:
        return o, None
    return o, Mamba3State(final, k32[:, -1], v32[:, -1])


from jla.ops.log_linear import log_linear_simple_gla_chunked as log_linear_simple_gla  # noqa: E402
