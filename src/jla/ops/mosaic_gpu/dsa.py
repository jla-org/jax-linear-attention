# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/attention_mgpu.py (through
# jla.ops.mosaic_gpu.attention); the indexer kernel follows deepseek-ai/DeepSeek-V3.2-Exp (MIT),
# inference/kernel.py (fp8_index_kernel).
"""DeepSeek Sparse Attention (DSA) on Hopper in Pallas Mosaic GPU.

Kernels for the dense plain-JAX math of :mod:`jla.ops.sparse_attention`, in query blocks so that
memory stays linear in the sequence length (one block's ``[rows, end]`` scores and mask at a time):

* The lightning indexer (:func:`index_scores` for the full ``[B, T, T]`` scores, :func:`index_topk`
  for the top-k indices of query blocks): ``I[t, s] = sum_j w[t, j] relu(scale q_idx[t, j] . k_idx[s])``
  in float32, ``-inf`` outside the causal document. Each CTA owns one ``128 x 128`` (query, key)
  tile and streams the index heads through a TMA pipeline: per head one bf16/fp16 WGMMA (f32
  accumulation, so the scores are the float32 scores of the low-precision inputs up to summation
  order), then ReLU and the per-token head weight are applied to the accumulator in registers and
  summed over heads; the ``[B, T, HI, T]`` intermediate of the plain op is never built. The grid
  enumerates only the tiles on or below the diagonal (query tile ``i`` paired with ``m - 1 - i``);
  the CTAs of the diagonal block also write the mirrored ``-inf`` tiles. ``scale > 0`` is folded
  into the head weights. The top-k is exact (the keys of ``dsa_select``): the radix-select kernel
  :func:`jla.ops.mosaic_gpu.dsa_topk.radix_top_k` (``dsa_layer``'s default) or ``lax.top_k`` on causal
  prefixes; an approximate FP8 indexer is an explicit option.
* :func:`dsa_attention` is ``dsa``: GQA softmax attention of every query over its selected keys
  (``indices [B, T, K]``, ``-1`` unused), forward and backward (``jax.custom_vjp``). Per query block
  the selection (with the causal and document masks) is scattered into a ``[B, rows, end]`` 0/1
  bf16 mask in XLA (rebuilt in the backward pass) and applied per tile inside FA3-style kernels (two
  ping-pong consumer warpgroups and a TMA producer in the forward, ``dq`` and ``dk, dv`` kernels in
  the backward). The forward never loads key tiles in the causal future; the backward loads them but
  skips their math. Exact (no approximate mode): masked keys get probability 0, a query without a
  selected key gets output 0.
* DeepSeek-V3.2's indexer loss over the selected keys: the selected scores come with the top-k, and
  their gradient (:func:`selected_index_scores`) runs on two indexer backward kernels over the same
  lower-triangle tiles as the forward (the score gradient scattered into a dense float16 block, ``dq``
  and ``dw`` per query tile and head, ``dk`` per key tile and head group); the target
  (:func:`selected_attention_target`, the main attention's probabilities summed over heads) runs on
  the indexer forward kernel in an ``exp`` mode. :func:`dsa_layer` is the whole mixer core per batch
  shard of the context mesh. :func:`sparse_index_scores` and :func:`sparse_attention_target` are the
  plain-JAX gather forms (``O(T K)``, references).

The attention kernels cost dense causal FLOPs plus one bf16 mask tile per (query, key) tile: they
do not gather the selected keys, so they win against the plain op, not against dense cuDNN
attention (see ``benchmarks/dsa.md``).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import COLS, ROWS, WGMMA, bcast_cols, bcast_rows, compiler_params, iota, is_hopper, smem
from jla.ops.mosaic_gpu.attention import _delta
from jla.ops.sparse_attention import _check_heads, _chunked, _documents, _gather_keys, _row_chunk, valid_indices

__all__ = ["dsa_attention", "dsa_layer", "index_scores", "index_topk", "select_top_k", "selected_attention_target",
           "selected_index_scores", "selection_mask", "sparse_attention_target", "sparse_index_scores",
           "supported_attention", "supported_index", "valid_indices"]

Array = jax.Array

_LOG2E = math.log2(math.e)
_M_INIT = -1e30  # finite running max: a fully masked tile leaves acc and l at zero
_EMPTY_LSE = 1e30  # log-sum-exp of a query without selected keys: every recomputed probability is 0
_IDX_TILE = 128  # query and key tile of the indexer kernel (two consumer warpgroups of 64 rows)
_ATTN_PAD = 128  # sequence padding of the attention kernels (2 * block_q of the forward)
_SINGLE_BLOCK = 8192  # sequences up to this length run as one query block
_BLOCK_ROWS = 4096  # query rows per block beyond it: one block's [rows, end] scores and mask
_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))


def _hopper_reason() -> str | None:
    if jax.default_backend() != "gpu" or not is_hopper():
        return "the Mosaic GPU DSA kernels need an sm_90 (Hopper) GPU"
    return None


def supported_index(q_idx: Array, k_idx: Array, w_idx: Array, scale: float | None = None) -> str | None:
    """``None`` when :func:`index_scores` accepts these operands, otherwise the reason it does not."""
    if q_idx.ndim != 4 or k_idx.ndim != 3 or w_idx.ndim != 3:
        return "expected q_idx [B, T, HI, DI], k_idx [B, T, DI] and w_idx [B, T, HI]"
    batch, length, heads, dim = q_idx.shape
    if k_idx.shape != (batch, length, dim) or w_idx.shape != (batch, length, heads):
        return f"incompatible q_idx {q_idx.shape}, k_idx {k_idx.shape}, w_idx {w_idx.shape}"
    if dim not in (64, 128):
        return f"index_head_dim must be 64 or 128; got {dim}"
    if jnp.dtype(q_idx.dtype) not in _DTYPES or k_idx.dtype != q_idx.dtype:
        return f"q_idx and k_idx must both be bf16 or fp16; got {q_idx.dtype} and {k_idx.dtype}"
    if scale is not None and not float(scale) > 0.0:
        return f"the kernel folds the scale into the head weights and needs scale > 0; got {scale}"
    return _hopper_reason()


def supported_attention(q: Array, k: Array, v: Array) -> str | None:
    """``None`` when :func:`dsa_attention` accepts these operands, otherwise the reason it does not."""
    try:
        _check_heads(q, k, v)
    except ValueError as err:
        return str(err)
    if k.shape != v.shape:
        return f"the kernel needs value_dim == head_dim; got k {k.shape} and v {v.shape}"
    if q.shape[3] not in (64, 128):
        return f"head_dim must be 64 or 128; got {q.shape[3]}"
    if jnp.dtype(q.dtype) not in _DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q, k and v must all be bf16 or fp16; got {q.dtype}, {k.dtype}, {v.dtype}"
    return _hopper_reason()


# ----------------------------------------------------------------------------------------------
# Lightning indexer scores
# ----------------------------------------------------------------------------------------------


def _index_call(qh: Array, kk: Array, wh: Array, doc: Array, *, heads: int, rows: int, offset: int, has_doc: bool,
                exp_scale: float | None = None, w_tiles: tuple[int, int] | None = None):
    """Scores ``[B, rows, offset + rows]`` of the queries ``offset ... offset + rows - 1`` over the keys before them.

    ``qh [B, HI * rows, DI]`` and ``wh [B, HI * rows]`` (head-major) hold the block's queries and head
    weights; ``kk [B, P, DI]`` and the document ids ``doc [B, P]`` cover at least the keys
    ``0 ... offset + rows - 1`` (global positions). ``rows`` is a multiple of ``2 * 128`` and ``offset`` of
    128. The grid pairs query tile ``i`` (``a + i + 1`` visible key tiles, ``a = offset / 128``) with tile
    ``m - 1 - i``, so every grid row has ``2 a + m + 1`` tiles; the CTAs of the last ``m`` key tiles also write
    the mirrored ``-inf`` tile above the diagonal. ``offset = 0, rows = P`` is the full ``[B, P, P]`` square.
    With ``exp_scale`` each head adds ``exp2(exp_scale * q_h . k - wh_h)`` instead of ``wh_h relu(q_h . k)``
    (``wh`` then holds the heads' log2 log-sum-exp): attention probabilities summed over the heads.
    ``w_tiles = (base, stride)`` reads head ``j``'s values of query tile ``i`` from 128-wide tile
    ``base + j * stride + i`` of ``wh [B, N]`` instead of ``j * m + i`` (no sliced copy of a small array).
    """
    batch, _, dim = kk.shape
    m = rows // _IDX_TILE  # even
    w_base, w_stride = (0, m) if w_tiles is None else w_tiles
    a = offset // _IDX_TILE
    half = _IDX_TILE // 2
    dtype = kk.dtype
    k_tile = smem((_IDX_TILE, dim), dtype)
    tiling, swizzle = k_tile.transforms

    def kernel(q_ref, k_ref, w_ref, doc_ref, out_ref, k_smem, out_smem, docq_smem, dock_smem, bars):
        row = lax.axis_index("row")
        col = lax.axis_index("col")
        b = lax.axis_index("batch")
        # 1 for the paired query tile m - 1 - row
        upper = lax.min(lax.max(col - (row + a), jnp.int32(0)), jnp.int32(1))
        qi = row + upper * (jnp.int32(m - 1) - 2 * row)  # query tile of the block
        kj = col - upper * (row + a + 1)  # key tile, 0 ... a + qi
        q0 = qi * _IDX_TILE  # first query row of the tile in the block
        qg = q0 + offset  # its global position
        k0 = kj * _IDX_TILE
        k_bar, dq_bar, dk_bar = bars

        def compute_context(pipeline):
            wg = lax.axis_index("wg")

            @pl.when(wg == 0)
            def _():
                plgpu.copy_gmem_to_smem(k_ref.at[b, pl.ds(k0, _IDX_TILE)], k_smem, k_bar)
                if has_doc:
                    plgpu.copy_gmem_to_smem(doc_ref.at[b, pl.ds(qg, _IDX_TILE)], docq_smem, dq_bar)
                    plgpu.copy_gmem_to_smem(doc_ref.at[b, pl.ds(k0, _IDX_TILE)], dock_smem, dk_bar)

            plgpu.barrier_wait(k_bar)
            if has_doc:
                plgpu.barrier_wait(dq_bar)
                plgpu.barrier_wait(dk_bar)
            o_smem = out_smem.at[wg]
            mirrored = kj != qi + a
            if a:
                mirrored = jnp.logical_and(mirrored, kj >= a)

            @pl.when(mirrored)
            def _():  # tile (kj - a, a + qi) of the block: entirely in the causal future
                o_smem[...] = plgpu.layout_cast(jnp.full((half, _IDX_TILE), -jnp.inf, jnp.float32), WGMMA)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem, out_ref.at[b, pl.ds((kj - a) * _IDX_TILE + wg * half, half), pl.ds(qg, _IDX_TILE)])
                plgpu.wait_smem_to_gmem(0)

            score = pipeline(plgpu.layout_cast(jnp.full((half, _IDX_TILE), 0.0, jnp.float32), WGMMA))
            q_pos = iota((half, _IDX_TILE), 0) + (qg + wg * half)
            k_pos = iota((half, _IDX_TILE), 1) + k0
            visible = q_pos >= k_pos
            if has_doc:
                doc_q = plgpu.load(docq_smem.at[pl.ds(wg * half, half)], layout=ROWS)
                doc_k = plgpu.load(dock_smem, layout=COLS)
                visible = visible & (bcast_rows(doc_q, (half, _IDX_TILE)) == bcast_cols(doc_k, (half, _IDX_TILE)))
            o_smem[...] = jnp.where(visible, score, -jnp.inf)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem, out_ref.at[b, pl.ds(q0 + wg * half, half), pl.ds(k0, _IDX_TILE)])
            plgpu.wait_smem_to_gmem(0)

        def body(idx, q_smem, w_smem, q_consumed, w_consumed, score):
            wg = lax.axis_index("wg")

            def matmul(acc_ref):
                plgpu.wgmma(acc_ref, q_smem.at[pl.ds(wg * half, half)], k_smem.transpose((1, 0)))
                return acc_ref[...]

            s = pl.run_scoped(matmul, plgpu.ACC((half, _IDX_TILE), jnp.float32))
            plgpu.barrier_arrive(q_consumed)
            w = plgpu.load(w_smem.at[pl.ds(wg * half, half)], layout=ROWS)
            plgpu.barrier_arrive(w_consumed)
            if exp_scale is None:
                return score + jnp.maximum(s, 0.0) * bcast_rows(w, (half, _IDX_TILE))
            return score + jnp.exp2(s * exp_scale - bcast_rows(w, (half, _IDX_TILE)))

        pipeline = plgpu.emit_pipeline_warp_specialized(
            body,
            grid=(heads,),
            max_concurrent_steps=min(3, heads),
            num_compute_wgs=2,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_context,
            in_specs=[
                plgpu.BlockSpec(block_shape=(_IDX_TILE, dim), index_map=lambda j: (j * m + qi, 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(_IDX_TILE,), index_map=lambda j: (w_base + j * w_stride + qi,)),
            ],
        )
        pipeline(q_ref.at[b], w_ref.at[b])

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((batch, rows, offset + rows), jnp.float32),
        scratch_types=[
            k_tile,
            smem((2, half, _IDX_TILE), jnp.float32),
            plgpu.SMEM((_IDX_TILE,), jnp.int32),
            plgpu.SMEM((_IDX_TILE,), jnp.int32),
            (plgpu.Barrier(), plgpu.Barrier(), plgpu.Barrier()),
        ],
        compiler_params=compiler_params(),
        grid=(m // 2, 2 * a + m + 1, batch),
        grid_names=("row", "col", "batch"),
        num_threads=3,
        thread_name="wg",
    )(qh, kk, wh, doc)


def index_scores(
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
) -> Array:
    """Lightning-indexer scores ``[B, T, T]`` (float32, ``-inf`` outside the causal document).

    Same contract as :func:`jla.ops.sparse_attention.dsa_index_scores`; ``q_idx``
    and ``k_idx`` must be bf16 or fp16 (the products are exact in the float32
    accumulator), ``w_idx`` is read in float32.
    """
    reason = supported_index(q_idx, k_idx, w_idx, scale)
    if reason is not None:
        raise ValueError(f"mosaic_gpu dsa index_scores: {reason}")
    batch, length, _, dim = q_idx.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    doc, _ = _documents(segment_ids, batch, length)
    return _index_scores(q_idx, k_idx, w_idx.astype(jnp.float32), doc.astype(jnp.int32), scale,
                         segment_ids is not None)


def _prefix_top_k(scores: Array, k: int, offset: int, row_blocks: int) -> tuple[Array, Array]:
    """``dsa_select`` of the score rows of queries ``offset ...`` (``[B, R, >= offset + R]``), ``k`` slots,
    and the selected scores (``-inf`` in unused slots).

    The queries of row block ``[start, end)`` only see keys ``< offset + end``, so ``lax.top_k``
    runs on that prefix; slots beyond a short prefix are ``-1`` exactly as in ``dsa_select``,
    and the order (descending score, lower index first among ties) is unchanged.
    """
    rows = scores.shape[1]
    parts, values, start = [], [], 0
    for i in range(row_blocks):
        end = (rows * (i + 1)) // row_blocks
        if end <= start:
            continue
        width = min(offset + end, scores.shape[2])
        kk = min(k, width)
        top, idx = lax.top_k(scores[:, start:end, :width], kk)
        idx = jnp.where(jnp.isfinite(top), idx, -1)
        if kk < k:
            idx = jnp.pad(idx, ((0, 0), (0, 0), (0, k - kk)), constant_values=-1)
            top = jnp.pad(top, ((0, 0), (0, 0), (0, k - kk)), constant_values=-jnp.inf)
        parts.append(idx)
        values.append(top)
        start = end
    return jnp.concatenate(parts, axis=1), jnp.concatenate(values, axis=1)


def select_top_k(scores: Array, top_k: int, *, row_blocks: int = 8) -> Array:
    """:func:`jla.ops.sparse_attention.dsa_select` with the same output, sorting only causal prefixes.

    The queries of row block ``[start, end)`` only see keys ``< end``, so ``lax.top_k``
    runs on ``scores[:, start:end, :end]`` (about 56% of the entries for 8 blocks).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    return _prefix_top_k(lax.stop_gradient(scores), min(top_k, scores.shape[-1]), 0, row_blocks)[0]


def _pad_index_inputs(q_idx, k_idx, w_idx, doc):
    """Pad the sequence to a multiple of ``2 * 128``: padded keys sit in the causal future of every real query."""
    length = q_idx.shape[1]
    pad = -length % (2 * _IDX_TILE)
    if pad:
        q_idx = jnp.pad(q_idx, ((0, 0), (0, pad), (0, 0), (0, 0)))
        k_idx = jnp.pad(k_idx, ((0, 0), (0, pad), (0, 0)))
        w_idx = jnp.pad(w_idx, ((0, 0), (0, pad), (0, 0)))
        doc = jnp.pad(doc, ((0, 0), (0, pad)), constant_values=-1)
    return q_idx, k_idx, w_idx, doc


def _index_block(q_idx, k_idx, w_idx, doc, *, offset: int, rows: int, has_doc: bool) -> Array:
    """Kernel scores ``[B, rows, offset + rows]`` of the padded inputs' queries ``offset ... offset + rows - 1``
    (``w_idx`` already scaled)."""
    batch, _, heads, dim = q_idx.shape
    qh = q_idx[:, offset:offset + rows].transpose(0, 2, 1, 3).reshape(batch, heads * rows, dim)
    wh = w_idx[:, offset:offset + rows].transpose(0, 2, 1).reshape(batch, heads * rows)
    return _index_call(qh, k_idx, wh, doc, heads=heads, rows=rows, offset=offset, has_doc=has_doc)


def _index_forward(q_idx, k_idx, w_idx, doc, scale: float, has_doc: bool):
    length = q_idx.shape[1]
    q_idx, k_idx, w_idx, doc = _pad_index_inputs(q_idx, k_idx, w_idx * scale, doc)
    padded = q_idx.shape[1]
    out = _index_block(q_idx, k_idx, w_idx, doc, offset=0, rows=padded, has_doc=has_doc)
    return out[:, :length, :length] if padded != length else out


def index_topk(
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    top_k: int,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
    block_rows: int | None = None,
    method: str = "sort",
    fp8: bool = False,
) -> Array:
    """``dsa_select(dsa_index_scores(...), top_k)`` without the ``[B, T, T]`` scores: ``[B, T, min(top_k, T)]``.

    The queries are processed in blocks of ``block_rows`` (a multiple of 256; default 4096, or the whole
    sequence up to 8192 tokens): the indexer kernel scores one block against the keys before its end
    (``[B, rows, end]`` float32), and an exact top-k (``lax.top_k`` on causal prefixes, as
    :func:`select_top_k`) keeps the indices. Peak memory is one block's scores instead of ``T^2``
    floats (each block waits for the previous one). No gradient flows (the indices are discrete; the
    indexer learns through :func:`selected_index_scores`). ``method="radix"`` selects the same keys with
    the radix-select kernel (:func:`jla.ops.mosaic_gpu.dsa_topk.radix_top_k`: no sort scratch), listed by
    position (two ascending runs) instead of by descending score. ``fp8=True``: the approximate FP8 indexer
    (see :func:`_index_topk`).
    """
    return _index_topk(q_idx, k_idx, w_idx, top_k, scale=scale, segment_ids=segment_ids, block_rows=block_rows,
                       method=method, fp8=fp8)[0]


_E4M3_MAX = 448.0  # largest finite float8_e4m3fn


def _fp8_quantize(x: Array, axes: tuple[int, ...]) -> tuple[Array, Array]:
    """``x / s`` in float8_e4m3fn with ``s = max |x| / 448`` over ``axes`` (kept as size-1 axes; 1 where the
    max is 0)."""
    x = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x), axis=axes, keepdims=True)
    s = jnp.where(amax > 0, amax / _E4M3_MAX, 1.0)
    return (x / s).astype(jnp.float8_e4m3fn), s


def _index_topk(q_idx, k_idx, w_idx, top_k: int, *, scale=None, segment_ids=None, block_rows=None, method="sort",
                fp8: bool = False):
    """:func:`index_topk`'s indices and the kernel scores of the selected keys (``[B, T, K]`` float32,
    ``-inf`` in unused slots), both without gradient. ``method``: ``"sort"`` (``lax.top_k`` on causal
    prefixes) or ``"radix"`` (the radix-select kernel; same keys, listed by position).

    ``fp8=True`` is an APPROXIMATE mode (DeepSeek-V3.2 runs its indexer in FP8): the indexer kernel reads
    float8_e4m3fn index queries (one scale per token and head, folded into the head weights) and keys (one
    scale per sequence), at twice the bf16 tensor-core rate, so the selection and the returned scores are
    those of the quantized scores (measured against the exact selection in ``benchmarks/dsa.md``)."""
    if method not in ("sort", "radix"):
        raise ValueError(f"method must be 'sort' or 'radix'; got {method!r}")
    reason = supported_index(q_idx, k_idx, w_idx, scale)
    if reason is not None:
        raise ValueError(f"mosaic_gpu dsa index_topk: {reason}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    batch, length, _, dim = q_idx.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    doc, _ = _documents(segment_ids, batch, length)
    q_idx, k_idx = lax.stop_gradient(q_idx), lax.stop_gradient(k_idx)
    w_idx = lax.stop_gradient(w_idx).astype(jnp.float32) * scale
    if fp8:  # relu(a b x) = a b relu(x) for scales a, b > 0
        q_idx, q_scale = _fp8_quantize(q_idx, (3,))
        k_idx, k_scale = _fp8_quantize(k_idx, (1, 2))
        w_idx = w_idx * q_scale[..., 0] * k_scale
    q_idx, k_idx, w_idx, doc = _pad_index_inputs(q_idx, k_idx, w_idx, doc.astype(jnp.int32))
    padded = q_idx.shape[1]
    block = _block_rows(padded, block_rows, 2 * _IDX_TILE)
    k = min(top_k, length)
    parts, values = [], []
    for offset in range(0, padded, block):
        rows = min(block, padded - offset)
        q_b, w_b = _after(parts[-1] if parts else None, q_idx, w_idx)
        scores = _index_block(q_b, k_idx, w_b, doc, offset=offset, rows=rows, has_doc=segment_ids is not None)
        if method == "radix":
            from jla.ops.mosaic_gpu.dsa_topk import radix_top_k

            if offset + rows > length:  # padding queries select nothing, so they cannot force the full-row passes
                scores = scores.at[:, length - offset:].set(-jnp.inf)
            idx, top = radix_top_k(scores, k, offset=offset)
        else:
            idx, top = _prefix_top_k(scores, k, offset, 8 if offset == 0 else 2)
        parts.append(idx)
        values.append(top)
    out, top = jnp.concatenate(parts, axis=1), jnp.concatenate(values, axis=1)
    return (out[:, :length], top[:, :length]) if padded != length else (out, top)


def _after(prev, *xs):
    """``xs`` unchanged, but computed only once ``prev`` (the previous query block's result) exists, so that
    XLA keeps one block's ``[rows, end]`` intermediates alive at a time (``None``: no dependency)."""
    if prev is None:
        return xs
    return lax.optimization_barrier((prev, xs))[1]


def _block_rows(padded: int, block_rows: int | None, multiple: int) -> int:
    """Rows per block: the whole (padded) sequence up to ``_SINGLE_BLOCK`` tokens, else ``_BLOCK_ROWS``."""
    if block_rows is None:
        return padded if padded <= _SINGLE_BLOCK else _BLOCK_ROWS
    if block_rows < multiple or block_rows % multiple:
        raise ValueError(f"block_rows must be a positive multiple of {multiple}; got {block_rows}")
    return block_rows


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _index_scores(q_idx, k_idx, w_idx, doc, scale: float, has_doc: bool):
    return _index_forward(q_idx, k_idx, w_idx, doc, scale, has_doc)


def _index_scores_fwd(q_idx, k_idx, w_idx, doc, scale: float, has_doc: bool):
    out = _index_forward(q_idx, k_idx, w_idx, doc, scale, has_doc)
    return out, (q_idx, k_idx, w_idx, out)


def _index_scores_bwd(scale: float, has_doc: bool, res, g):
    """Plain-JAX backward, one index head at a time (``[B, T, T]`` per step).

    The pre-activations are recomputed from the bf16/fp16 inputs with float32
    accumulation, as in the forward kernel; masked scores get no gradient. As in
    the attention backward kernels (and FlashAttention), the score gradient is
    rounded to the input dtype for the ``dq`` and ``dk`` matmuls, which accumulate
    in float32.
    """
    q_idx, k_idx, w_idx, out = res
    g = jnp.where(jnp.isfinite(out), g, 0.0)

    def head(dk, inputs):
        qj, wj = inputs  # [B, T, DI], [B, T]
        s = jnp.einsum("btd,bsd->bts", qj, k_idx, preferred_element_type=jnp.float32) * scale
        dw = jnp.sum(g * jnp.maximum(s, 0.0), axis=-1)
        ds = (jnp.where(s > 0, g * wj[..., None], 0.0) * scale).astype(q_idx.dtype)
        dq = jnp.einsum("bts,bsd->btd", ds, k_idx, preferred_element_type=jnp.float32)
        dk = dk + jnp.einsum("bts,btd->bsd", ds, qj, preferred_element_type=jnp.float32)
        return dk, (dq, dw)

    dk, (dq, dw) = lax.scan(head, jnp.zeros(k_idx.shape, jnp.float32),
                            (q_idx.transpose(2, 0, 1, 3), w_idx.transpose(2, 0, 1)))
    dq = dq.transpose(1, 2, 0, 3).astype(q_idx.dtype)
    return dq, dk.astype(k_idx.dtype), dw.transpose(1, 2, 0), None


_index_scores.defvjp(_index_scores_fwd, _index_scores_bwd)


# ----------------------------------------------------------------------------------------------
# Lightning indexer backward (the gradient of DeepSeek-V3.2's indexer KL)
# ----------------------------------------------------------------------------------------------

_DK_GROUP = 8  # index heads per CTA of the key-gradient kernel (partial sums over head groups in f32)


def _index_bwd_call(qh: Array, kk: Array, wh: Array, d: Array, dt: Array, *, heads: int, rows: int, offset: int):
    """Indexer gradients of the block's queries ``offset ... offset + rows - 1`` from the dense score gradient.

    ``qh [B, HI * rows, DI]`` and ``wh [B, HI * rows]`` (head-major, ``wh`` = head weight times the indexer
    scale); ``kk [B, W, DI]`` the keys ``0 ... W - 1`` (``W = offset + rows``); ``d [B, rows, W]`` and its
    transpose ``dt [B, W, rows]``: the block's score gradient (float16, scaled to ``[-1, 1]``; zero outside the
    selected keys). With ``S_h = q_h k^T`` (float32 accumulation) and ``P_h = d * wh_h * (S_h > 0)`` rounded
    to the input dtype (as FlashAttention rounds ``dS``), returns ``dq_h = P_h k`` (``[B, HI * rows, DI]``, input dtype),
    ``dwh_h = rowsum(d * relu(S_h))`` (``[B, HI * rows]`` float32) and the key gradient ``sum_h P_h^T q_h``
    as float32 partial sums over groups of ``_DK_GROUP`` heads (``[B, HI / _DK_GROUP, W, DI]``).

    Two kernels with 128 x 128 (query, key) tiles and two consumer warpgroups of 64 rows: ``dq`` / ``dw``
    per (query tile, head), streaming the key tiles (``k`` and ``d``); ``dk`` per (key tile, head group),
    streaming the (head, query tile) pairs (``q``, ``wh`` and ``dt``). Tiles in the causal future are not
    fetched again (the index maps clamp to the last visible tile) and their math is skipped.
    """
    batch, width, dim = kk.shape
    m = rows // _IDX_TILE
    nk = width // _IDX_TILE
    a = offset // _IDX_TILE
    half = _IDX_TILE // 2
    dtype = kk.dtype
    group = math.gcd(heads, _DK_GROUP)
    n_groups = heads // group
    row_tile = smem((_IDX_TILE, dim), dtype)
    tiling, swizzle = row_tile.transforms
    d_tile = smem((_IDX_TILE, _IDX_TILE), d.dtype)
    d_tiling, d_swizzle = d_tile.transforms

    def kernel_dq(q_ref, k_ref, w_ref, d_ref, dq_ref, dw_ref, q_smem, w_smem, dq_smem, dw_smem, bars):
        qi = lax.axis_index("q")
        h = lax.axis_index("heads")
        b = lax.axis_index("batch")
        row0 = h * rows + qi * _IDX_TILE  # first row of the tile in the head-major qh / wh
        last = qi + a  # last key tile visible to the query tile

        def compute_context(pipeline):
            wg = lax.axis_index("wg")

            @pl.when(wg == 0)
            def _():
                plgpu.copy_gmem_to_smem(q_ref.at[b, pl.ds(row0, _IDX_TILE)], q_smem, bars[0])
                plgpu.copy_gmem_to_smem(w_ref.at[b, pl.ds(row0, _IDX_TILE)], w_smem, bars[1])

            plgpu.barrier_wait(bars[0])
            plgpu.barrier_wait(bars[1])
            w = plgpu.load(w_smem.at[pl.ds(wg * half, half)], layout=ROWS)
            dq0 = plgpu.layout_cast(jnp.full((half, dim), 0.0, jnp.float32), WGMMA)
            dq, dw, _ = pipeline((dq0, w * 0.0, w))
            dq_out, dw_out = dq_smem.at[wg], dw_smem.at[wg]
            dq_out[...] = dq.astype(dtype)
            dw_out[...] = dw
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dq_out, dq_ref.at[b, pl.ds(row0 + wg * half, half)], commit_group=False)
            plgpu.copy_smem_to_gmem(dw_out, dw_ref.at[b, pl.ds(row0 + wg * half, half)], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        def body(idx, k_smem, d_smem, k_consumed, d_consumed, carry):
            (j,) = idx
            dq_acc, dw_acc, w = carry
            wg = lax.axis_index("wg")

            def visit(acc):
                dq_acc, dw_acc = acc

                def scores(acc_ref):
                    plgpu.wgmma(acc_ref, q_smem.at[pl.ds(wg * half, half)], k_smem.transpose((1, 0)))
                    return acc_ref[...]

                s = pl.run_scoped(scores, plgpu.ACC((half, _IDX_TILE), jnp.float32))
                g = plgpu.load(d_smem.at[pl.ds(wg * half, half)], layout=WGMMA).astype(jnp.float32)
                dw_acc = dw_acc + jnp.sum(g * jnp.maximum(s, 0.0), axis=1)
                p = jnp.where(s > 0.0, g * bcast_rows(w, (half, _IDX_TILE)), 0.0).astype(dtype)

                def grad_q(acc_ref):
                    plgpu.wgmma(acc_ref, p, k_smem)

                return pl.run_state(grad_q)(plgpu.ACC.init(dq_acc)), dw_acc

            dq_acc, dw_acc = lax.cond(j <= last, visit, lambda acc: acc, (dq_acc, dw_acc))
            plgpu.barrier_arrive(k_consumed)
            plgpu.barrier_arrive(d_consumed)
            return dq_acc, dw_acc, w

        pipeline = plgpu.emit_pipeline_warp_specialized(
            body,
            grid=(nk,),
            max_concurrent_steps=min(2, nk),
            num_compute_wgs=2,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_context,
            in_specs=[
                plgpu.BlockSpec(block_shape=(_IDX_TILE, dim), index_map=lambda j: (lax.min(j, last), 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(_IDX_TILE, _IDX_TILE), index_map=lambda j: (0, lax.min(j, last)),
                                transforms=[d_tiling, d_swizzle]),
            ],
        )
        pipeline(k_ref.at[b], d_ref.at[b, pl.ds(qi * _IDX_TILE, _IDX_TILE)])

    dq, dw = plgpu.kernel(
        kernel_dq,
        out_type=[jax.ShapeDtypeStruct(qh.shape, dtype), jax.ShapeDtypeStruct(wh.shape, jnp.float32)],
        scratch_types=[
            row_tile,
            plgpu.SMEM((_IDX_TILE,), jnp.float32),
            smem((2, half, dim), dtype),
            plgpu.SMEM((2, half), jnp.float32),
            (plgpu.Barrier(), plgpu.Barrier()),
        ],
        compiler_params=compiler_params(),
        grid=(m, heads, batch),
        grid_names=("q", "heads", "batch"),
        num_threads=3,
        thread_name="wg",
    )(qh, kk, wh, d)

    def kernel_dk(q_ref, w_ref, dt_ref, k_ref, dk_ref, k_smem, out_smem, bar):
        kj = lax.axis_index("k")
        grp = lax.axis_index("group")
        b = lax.axis_index("batch")
        k0 = kj * _IDX_TILE
        first = lax.max(kj - a, jnp.int32(0))  # first query tile of the block that sees the key tile

        def compute_context(pipeline):
            wg = lax.axis_index("wg")

            @pl.when(wg == 0)
            def _():
                plgpu.copy_gmem_to_smem(k_ref.at[b, pl.ds(k0, _IDX_TILE)], k_smem, bar)

            plgpu.barrier_wait(bar)
            dk = pipeline(plgpu.layout_cast(jnp.full((half, dim), 0.0, jnp.float32), WGMMA))
            o_smem = out_smem.at[wg]
            o_smem[...] = dk
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem, dk_ref.at[b, grp, pl.ds(k0 + wg * half, half)])
            plgpu.wait_smem_to_gmem(0)

        def body(idx, q_smem, w_smem, d_smem, q_consumed, w_consumed, d_consumed, dk_acc):
            _, i = idx
            wg = lax.axis_index("wg")

            def visit(dk_acc):
                def scores_t(acc_ref):
                    plgpu.wgmma(acc_ref, k_smem.at[pl.ds(wg * half, half)], q_smem.transpose((1, 0)))
                    return acc_ref[...]

                st = pl.run_scoped(scores_t, plgpu.ACC((half, _IDX_TILE), jnp.float32))
                g = plgpu.load(d_smem.at[pl.ds(wg * half, half)], layout=WGMMA).astype(jnp.float32)
                w = plgpu.load(w_smem, layout=COLS)
                pt = jnp.where(st > 0.0, g * bcast_cols(w, (half, _IDX_TILE)), 0.0).astype(dtype)

                def grad_k(acc_ref):
                    plgpu.wgmma(acc_ref, pt, q_smem)

                return pl.run_state(grad_k)(plgpu.ACC.init(dk_acc))

            dk_acc = lax.cond(i >= first, visit, lambda acc: acc, dk_acc)
            plgpu.barrier_arrive(q_consumed)
            plgpu.barrier_arrive(w_consumed)
            plgpu.barrier_arrive(d_consumed)
            return dk_acc

        def q_row(hh, i):  # tile row of head grp * group + hh, query tile max(i, first), in qh / wh
            return (grp * group + hh) * m + lax.max(i, first)

        pipeline = plgpu.emit_pipeline_warp_specialized(
            body,
            grid=(group, m),
            max_concurrent_steps=2,
            num_compute_wgs=2,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_context,
            in_specs=[
                plgpu.BlockSpec(block_shape=(_IDX_TILE, dim), index_map=lambda hh, i: (q_row(hh, i), 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(_IDX_TILE,), index_map=lambda hh, i: (q_row(hh, i),)),
                plgpu.BlockSpec(block_shape=(_IDX_TILE, _IDX_TILE), index_map=lambda hh, i: (0, lax.max(i, first)),
                                transforms=[d_tiling, d_swizzle]),
            ],
        )
        pipeline(q_ref.at[b], w_ref.at[b], dt_ref.at[b, pl.ds(k0, _IDX_TILE)])

    dk = plgpu.kernel(
        kernel_dk,
        out_type=jax.ShapeDtypeStruct((batch, n_groups, width, dim), jnp.float32),
        scratch_types=[
            row_tile,
            smem((2, half, dim), jnp.float32),
            plgpu.Barrier(),
        ],
        compiler_params=compiler_params(),
        grid=(nk, n_groups, batch),
        grid_names=("k", "group", "batch"),
        num_threads=3,
        thread_name="wg",
    )(qh, wh, dt, kk)
    return dq, dw, dk


def _dense_score_grad(g: Array, indices: Array, valid: Array, width: int) -> tuple[Array, Array, Array]:
    """The block's score gradient ``g [B, R, K]`` (at ``indices``, kept where ``valid``) scattered into dense
    float16 ``[B, R, width]`` and ``[B, width, R]`` arrays, each query row divided by ``s [B, R]``, its largest
    ``|g|`` (so every row's values lie in ``[-1, 1]``, where float16 keeps 11 significant bits whatever the
    row's magnitude; top-k indices are distinct, so ``set`` is exact). Returns ``(d, dt, s)``."""
    batch, rows, _ = indices.shape
    g = jnp.where(valid, g, 0.0)
    s = jnp.max(jnp.abs(g), axis=2)
    s = jnp.where(s > 0, s, 1.0)
    dtype = jnp.float16
    cols = jnp.where(valid, indices, width)  # out of bounds: dropped
    row_idx = jnp.broadcast_to(jnp.arange(rows, dtype=jnp.int32)[None, :, None], indices.shape)
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None, None], indices.shape)
    vals = (g / s[:, :, None]).astype(dtype)
    d = jnp.zeros((batch, rows, width), dtype).at[batch_idx, row_idx, cols].set(vals, mode="drop")
    dt = jnp.zeros((batch, width, rows), dtype).at[batch_idx, cols, row_idx].set(vals, mode="drop")
    return d, dt, s


def _index_backward(q_idx, k_idx, w_idx, indices, valid, g, *, scale: float, block: int):
    """``dq_idx, dk_idx, dw_idx`` of the selected scores from their gradient ``g [B, T, K]`` (kernels per block)."""
    batch, length, heads, dim = q_idx.shape
    pad = -length % (2 * _IDX_TILE)
    w_scaled = w_idx.astype(jnp.float32) * scale
    if pad:
        q_idx = jnp.pad(q_idx, ((0, 0), (0, pad), (0, 0), (0, 0)))
        k_idx = jnp.pad(k_idx, ((0, 0), (0, pad), (0, 0)))
        w_scaled = jnp.pad(w_scaled, ((0, 0), (0, pad), (0, 0)))
        indices = jnp.pad(indices, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
        valid = jnp.pad(valid, ((0, 0), (0, pad), (0, 0)))
        g = jnp.pad(g, ((0, 0), (0, pad), (0, 0)))
    padded = q_idx.shape[1]
    dqs, dws = [], []
    dk = jnp.zeros((batch, padded, dim), jnp.float32)
    for offset in range(0, padded, block):
        rows = min(block, padded - offset)
        width = offset + rows
        part = slice(offset, width)
        g_b, q_b = _after(dk if offset else None, g[:, part], q_idx)
        d, dt, s = _dense_score_grad(g_b, indices[:, part], valid[:, part], width)
        qh = q_b[:, part].transpose(0, 2, 1, 3).reshape(batch, heads * rows, dim)
        # wh carries each query row's scale, so the kernels' P = d * wh * (S > 0) is the unscaled
        # g * w * scale * (S > 0) and their dw (a row sum of d * relu(S)) is rescaled per row below
        wh = (w_scaled[:, part] * s[:, :, None]).transpose(0, 2, 1).reshape(batch, heads * rows)
        dq_h, dw_h, dk_g = _index_bwd_call(qh, k_idx[:, :width], wh, d, dt, heads=heads, rows=rows, offset=offset)
        dqs.append(dq_h.reshape(batch, heads, rows, dim).transpose(0, 2, 1, 3))
        dws.append(dw_h.reshape(batch, heads, rows).transpose(0, 2, 1) * s[:, :, None])
        dk = dk + jnp.pad(dk_g.sum(axis=1), ((0, 0), (0, padded - width), (0, 0)))
    dq = jnp.concatenate(dqs, axis=1) if len(dqs) > 1 else dqs[0]
    dw = jnp.concatenate(dws, axis=1) if len(dws) > 1 else dws[0]
    # d/dw of sum_h w_h relu(scale q_h . k) = scale relu(q_h . k): the kernels' dw is per unit of w * scale
    return dq[:, :length], dk[:, :length].astype(k_idx.dtype), dw[:, :length] * scale


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _selected_scores(q_idx, k_idx, w_idx, indices, valid, values, scale: float, block: int):
    del q_idx, k_idx, w_idx, indices, valid
    return values


def _selected_scores_fwd(q_idx, k_idx, w_idx, indices, valid, values, scale: float, block: int):
    return values, (q_idx, k_idx, w_idx, indices, valid)


def _selected_scores_bwd(scale: float, block: int, res, g):
    from jla.distributed.kernel_sharding import shard_layout_call

    q_idx, k_idx, w_idx, indices, valid = res

    def backward(q_idx, k_idx, w_idx, indices, valid, g):
        return _index_backward(q_idx, k_idx, w_idx, indices, valid, g, scale=scale, block=block)

    operands = dict(q_idx=q_idx, k_idx=k_idx, w_idx=w_idx, indices=indices, valid=valid,
                    g=jnp.where(valid, g, 0.0).astype(jnp.float32))
    dq, dk, dw = shard_layout_call(
        backward, operands, {name: "batch" for name in operands}, out_layouts=("batch", "batch", "batch"),
        supports=lambda q_idx, k_idx, w_idx, **_: supported_index(q_idx, k_idx, w_idx) is None)
    return dq, dk, dw.astype(w_idx.dtype), None, None, None


_selected_scores.defvjp(_selected_scores_fwd, _selected_scores_bwd)


def selected_index_scores(
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    indices: Array,
    values: Array,
    valid: Array,
    *,
    scale: float | None = None,
    block_rows: int | None = None,
) -> Array:
    """The indexer scores of the selected keys, ``values [B, T, K]`` (as returned with the indices by the
    top-k, so nothing is recomputed), differentiable in ``q_idx``, ``k_idx`` and ``w_idx`` on the kernels.

    The backward pass scatters the score gradient (kept where ``valid``) into a dense float16 block
    (each query row divided by its largest magnitude) per query block of ``block_rows`` (default as :func:`index_topk`)
    and runs the indexer backward kernels (:func:`_index_bwd_call`): ``O(T^2 HI DI)`` tensor-core work
    like the forward, ``dS`` rounded to the input dtype (as FlashAttention does) and float32
    accumulation. Same gradients as differentiating :func:`sparse_index_scores` (up to that rounding).
    """
    reason = supported_index(q_idx, k_idx, w_idx, scale)
    if reason is not None:
        raise ValueError(f"mosaic_gpu dsa selected_index_scores: {reason}")
    batch, length, _, dim = q_idx.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    block = _block_rows(length + (-length % (2 * _IDX_TILE)), block_rows, 2 * _IDX_TILE)
    return _selected_scores(q_idx, k_idx, w_idx, lax.stop_gradient(indices).astype(jnp.int32), valid,
                            lax.stop_gradient(values).astype(jnp.float32), scale, block)


# ----------------------------------------------------------------------------------------------
# Sparse attention over the selected keys
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _AttnConfig:
    scale: float
    block_q: int = 64
    block_kv: int = 64
    max_concurrent_steps: int = 2
    block_q_bwd: int = 64
    block_kv_bwd: int = 64


def selection_mask(indices: Array, length: int, segment_ids: Array | None = None, *, padded: int | None = None,
                   dtype=jnp.bool_) -> Array:
    """``[B, P, P]`` (``P = padded or T``): 1 where key ``s`` is selected by query ``t`` and is a causal key
    of its document.

    The selected indices are validated on ``[B, T, K]`` and scattered once into a
    zero mask (invalid ones are dropped), so the dense mask costs one write.
    """
    batch, _, _ = indices.shape
    padded = length if padded is None else padded
    doc, _ = _documents(segment_ids, batch, length)
    t = jnp.arange(length, dtype=jnp.int32)[None, :, None]
    safe = jnp.clip(indices, 0, length - 1)
    key_doc = jnp.take_along_axis(doc[:, None, :], safe.reshape(batch, 1, -1), axis=2).reshape(indices.shape)
    valid = (indices >= 0) & (indices <= t) & (key_doc == doc[:, :, None])
    cols = jnp.where(valid, indices, padded)  # out of bounds: dropped by the scatter
    rows = jnp.broadcast_to(t, indices.shape)
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None, None], indices.shape)
    one = jnp.ones((), dtype)
    return jnp.zeros((batch, padded, padded), dtype).at[batch_idx, rows, cols].set(one, mode="drop")


def _attn_forward(q, k, v, mask, cfg: _AttnConfig, q_offset: int = 0):
    """Output ``[B, R, HQ, D]`` and base-2 log-sum-exp ``[B, HQ, R]`` of the queries ``q_offset ... q_offset + R - 1``
    (``q [B, R, HQ, D]``) over the keys ``0 ... q_offset + R - 1`` of ``k, v [B, >= q_offset + R, H, D]``, with
    the 0/1 selection ``mask [B, R, q_offset + R]``."""
    batch, length, heads, head_dim = q.shape
    width = mask.shape[2]
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    bq, bkv = cfg.block_q, cfg.block_kv
    mcs = min(cfg.max_concurrent_steps, width // bkv)
    c = cfg.scale * _LOG2E
    layout = WGMMA

    def kernel(q_ref, k_ref, v_ref, m_ref, out_ref, lse_ref, scoped):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        wg = lax.axis_index("wg")
        smem_refs, bars, consumed, schedule = scoped
        qo_smem2, k_smem, v_smem, lse_smem2, m_smem = smem_refs
        k_bars, v_bars, q_bars, m_bars = bars
        k_consumed, v_consumed = consumed
        tile = lax.axis_index("q_seq")
        tile_base = tile * (2 * bq)  # first query row of the CTA in the block
        hi_all = lax.div(tile_base + (q_offset + 2 * bq + bkv - 1), jnp.int32(bkv))

        def sync():
            plgpu.barrier_arrive(schedule)
            plgpu.barrier_wait(schedule)

        def slot_of(step):
            return lax.rem(step, jnp.int32(mcs))

        @pl.when(wg < 2)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            qo_smem = qo_smem2.at[wg]
            q_base = tile_base + wg * bq
            hi = lax.div(q_base + (q_offset + bq + bkv - 1), jnp.int32(bkv))
            plgpu.copy_gmem_to_smem(q_ref.at[b, pl.ds(q_base, bq), h], qo_smem, q_bars.at[wg])
            plgpu.barrier_wait(q_bars.at[wg])

            m_i = plgpu.layout_cast(jnp.full((bq,), _M_INIT, jnp.float32), ROWS)
            l_i = plgpu.layout_cast(jnp.full((bq,), 0, jnp.float32), ROWS)
            acc = plgpu.layout_cast(jnp.full((bq, head_dim), 0, jnp.float32), layout)

            plgpu.barrier_wait(k_bars.at[slot_of(0)])
            pl.when(wg == 1)(sync)

            def kv_step(step, carry):
                acc, m_i, l_i = carry
                slot = slot_of(step)
                plgpu.barrier_wait(m_bars.at[slot])
                keep = plgpu.load(m_smem.at[slot, pl.ds(wg * bq, bq)], layout=layout).astype(jnp.float32) > 0.5

                def compute_qk(acc_ref):
                    plgpu.wgmma(acc_ref, qo_smem, k_smem.at[slot].transpose((1, 0)))
                    sync()
                    return acc_ref[...]

                qk = pl.run_scoped(compute_qk, plgpu.ACC((bq, bkv), jnp.float32))
                plgpu.barrier_arrive(k_consumed.at[slot])
                qk = jnp.where(keep, qk, -jnp.inf)

                m_ij = jnp.maximum(m_i, qk.max(axis=1) * c)
                alpha = jnp.exp2(m_i - m_ij)
                m_i = m_ij
                p = jnp.exp2(qk * c - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i *= alpha
                p16 = p.astype(dtype)
                l_i += p.sum(axis=1)
                acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                plgpu.barrier_arrive(schedule)
                plgpu.barrier_wait(v_bars.at[slot])
                plgpu.barrier_wait(schedule)

                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                    nxt = step + 1

                    @pl.when(nxt < hi)
                    def _():
                        plgpu.barrier_wait(k_bars.at[slot_of(nxt)])

                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_consumed.at[slot])
                return acc, m_i, l_i

            def skip_after(step, carry):
                slot = slot_of(step)
                plgpu.barrier_arrive(k_consumed.at[slot])
                plgpu.barrier_arrive(v_consumed.at[slot])
                sync()
                sync()
                return carry

            acc, m_i, l_i = lax.fori_loop(jnp.int32(0), hi, kv_step, (acc, m_i, l_i))
            lax.fori_loop(hi, hi_all, skip_after, None)
            pl.when(wg == 0)(sync)

            nonempty = l_i > 0
            acc /= lax.broadcast_in_dim(jnp.where(nonempty, l_i, 1.0), (bq, head_dim), [0])
            qo_smem[...] = acc.astype(dtype)
            lse_smem = lse_smem2.at[wg]
            lse_smem[...] = jnp.where(nonempty, m_i + jnp.log(jnp.where(nonempty, l_i, 1.0)) * _LOG2E, _EMPTY_LSE)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(qo_smem, out_ref.at[b, pl.ds(q_base, bq), h])
            plgpu.copy_smem_to_gmem(lse_smem, lse_ref.at[b, h, pl.ds(q_base, bq)])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")
            kvh = lax.div(h, jnp.int32(group))

            def load_k(step, slot):
                plgpu.copy_gmem_to_smem(k_ref.at[b, pl.ds(step * bkv, bkv), kvh], k_smem.at[slot], k_bars.at[slot])
                plgpu.copy_gmem_to_smem(m_ref.at[b, pl.ds(tile_base, 2 * bq), pl.ds(step * bkv, bkv)],
                                        m_smem.at[slot], m_bars.at[slot])

            def load_v(step, slot):
                plgpu.copy_gmem_to_smem(v_ref.at[b, pl.ds(step * bkv, bkv), kvh], v_smem.at[slot], v_bars.at[slot])

            for i in range(mcs):

                @pl.when(jnp.int32(i) < hi_all)
                def _():
                    load_k(jnp.int32(i), jnp.int32(i))
                    load_v(jnp.int32(i), jnp.int32(i))

            @pl.loop(jnp.int32(0), hi_all - mcs)
            def _(step):
                nxt = step + mcs
                slot = slot_of(step)
                plgpu.barrier_wait(k_consumed.at[slot])
                load_k(nxt, slot)
                plgpu.barrier_wait(v_consumed.at[slot])
                load_v(nxt, slot)

    def entry(q_ref, k_ref, v_ref, m_ref, out_ref, lse_ref):
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)
        scratch = (
            plgpu.SMEM((2, bq, head_dim), dtype, transforms=(tiling, swizzle)),
            plgpu.SMEM((mcs, bkv, head_dim), dtype, transforms=(tiling, swizzle)),
            plgpu.SMEM((mcs, bkv, head_dim), dtype, transforms=(tiling, swizzle)),
            plgpu.SMEM((2, bq), jnp.float32),
            plgpu.SMEM((mcs, 2 * bq, bkv), mask.dtype, transforms=(tiling, swizzle)),
        )
        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, m_ref, out_ref, lse_ref, args),
            scratch,
            (
                plgpu.Barrier(num_barriers=mcs),
                plgpu.Barrier(num_barriers=mcs),
                plgpu.Barrier(num_barriers=2),
                plgpu.Barrier(num_barriers=mcs),
            ),
            (plgpu.Barrier(num_arrivals=2, num_barriers=mcs),) * 2,
            plgpu.Barrier(num_arrivals=2),
            collective_axes="wg",
        )

    out, lse = plgpu.kernel(
        entry,
        out_type=[
            jax.ShapeDtypeStruct(q.shape, dtype),
            jax.ShapeDtypeStruct((batch, heads, length), jnp.float32),
        ],
        grid=(heads, length // (2 * bq), batch),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, mask)
    return out, lse


def _attn_backward(cfg: _AttnConfig, q, k, v, mask, out, lse, do, q_offset: int = 0):
    """``dq [B, R, HQ, D]`` and float32 ``dk, dv [B, q_offset + R, H, D]`` (the block's keys) of :func:`_attn_forward`."""
    batch, length, heads, head_dim = q.shape
    width = mask.shape[2]
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    n_q, n_kv = length // bq, width // bkv
    mcs = cfg.max_concurrent_steps
    c = cfg.scale * _LOG2E
    layout = WGMMA
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    delta = _delta(out, do)  # from the rounded cotangent the kernels read (see attention._delta)

    def kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, m_ref, dq_ref, smem_refs, bars):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        kvh = lax.div(h, jnp.int32(group))
        q_smem, do_smem, lse_smem, delta_smem = smem_refs
        q_base = lax.axis_index("q_seq") * bq

        def compute_thread(pipeline_callback):
            q_slice = (b, pl.ds(q_base, bq), h)
            plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, bars[0])
            plgpu.copy_gmem_to_smem(do_ref.at[q_slice], do_smem, bars[1])
            plgpu.copy_gmem_to_smem(lse_ref.at[b, h, pl.ds(q_base, bq)], lse_smem, bars[2])
            plgpu.copy_gmem_to_smem(delta_ref.at[b, h, pl.ds(q_base, bq)], delta_smem, bars[3])
            for bar in bars:
                plgpu.barrier_wait(bar)
            lse_v = plgpu.load(lse_smem, layout=ROWS)
            delta_v = plgpu.load(delta_smem, layout=ROWS)
            dq_acc = plgpu.layout_cast(jnp.full((bq, head_dim), 0, jnp.float32), layout)
            dq, _, _ = pipeline_callback((dq_acc, lse_v, delta_v))
            q_smem[...] = dq.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
            plgpu.wait_smem_to_gmem(0)

        def kv_body(idx, k_smem, v_smem, m_smem, k_c, v_c, m_c, carry):
            dq_acc, lse_v, delta_v = carry
            (kv_step,) = idx

            def visit(dq_acc):
                def compute_s(acc_ref):
                    plgpu.wgmma(acc_ref, q_smem, k_smem.transpose((1, 0)))
                    return acc_ref[...]

                s = pl.run_scoped(compute_s, plgpu.ACC((bq, bkv), jnp.float32))
                keep = plgpu.load(m_smem, layout=layout).astype(jnp.float32) > 0.5
                p = jnp.exp2(s * c - lax.broadcast_in_dim(lse_v, (bq, bkv), [0]))
                p = jnp.where(keep, p, 0.0)

                def compute_dp(acc_ref):
                    plgpu.wgmma(acc_ref, do_smem, v_smem.transpose((1, 0)))
                    return acc_ref[...]

                dp = pl.run_scoped(compute_dp, plgpu.ACC((bq, bkv), jnp.float32))
                ds = p * (dp - lax.broadcast_in_dim(delta_v, (bq, bkv), [0])) * cfg.scale

                def compute_dq(acc_ref):
                    plgpu.wgmma(acc_ref, ds.astype(dtype), k_smem)

                return pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))

            # Key tiles entirely in the causal future of this query tile hold no selected key.
            dq_acc = lax.cond(kv_step * bkv <= q_base + (q_offset + bq - 1), visit, lambda acc: acc, dq_acc)
            plgpu.barrier_arrive(m_c)
            plgpu.barrier_arrive(v_c)
            plgpu.barrier_arrive(k_c)
            return dq_acc, lse_v, delta_v

        pipeline = plgpu.emit_pipeline_warp_specialized(
            kv_body,
            grid=(n_kv,),
            max_concurrent_steps=min(mcs, n_kv),
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=[
                plgpu.BlockSpec(block_shape=(bkv, head_dim), index_map=lambda i: (i, 0), transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bkv, head_dim), index_map=lambda i: (i, 0), transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bq, bkv), index_map=lambda i: (0, i), transforms=[tiling, swizzle]),
            ],
        )
        pipeline(k_ref.at[b, :, kvh, :], v_ref.at[b, :, kvh, :], m_ref.at[b, pl.ds(q_base, bq), :])

    def kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, m_ref, dk_ref, dv_ref, smem_refs, bars):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        kvh = lax.div(h, jnp.int32(group))
        k_smem, v_smem = smem_refs
        kv_base = lax.axis_index("kv_seq") * bkv

        def compute_thread(pipeline_callback):
            kv_slice = (b, pl.ds(kv_base, bkv), kvh)
            plgpu.copy_gmem_to_smem(k_ref.at[kv_slice], k_smem, bars[0])
            plgpu.copy_gmem_to_smem(v_ref.at[kv_slice], v_smem, bars[1])
            for bar in bars:
                plgpu.barrier_wait(bar)
            zeros = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            dk, dv = pipeline_callback((zeros, zeros))
            k_smem[...] = dk.astype(dtype)
            v_smem[...] = dv.astype(dtype)
            plgpu.commit_smem()
            out_slice = (b, pl.ds(kv_base, bkv), h)
            plgpu.copy_smem_to_gmem(k_smem, dk_ref.at[out_slice], commit_group=False)
            plgpu.copy_smem_to_gmem(v_smem, dv_ref.at[out_slice], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        def q_body(idx, q_smem, do_smem, lse_smem, delta_smem, m_smem, q_c, do_c, lse_c, delta_c, m_c, carry):
            (q_step,) = idx

            def visit(carry):
                dk_acc, dv_acc = carry

                def compute_st(acc_ref):
                    plgpu.wgmma(acc_ref, k_smem, q_smem.transpose((1, 0)))
                    return acc_ref[...]

                st = pl.run_scoped(compute_st, plgpu.ACC((bkv, bq), jnp.float32))
                lse_v = plgpu.load(lse_smem, layout=COLS)
                keep_t = plgpu.load(m_smem.T, layout=layout).astype(jnp.float32) > 0.5
                pt = jnp.exp2(st * c - lax.broadcast_in_dim(lse_v, (bkv, bq), [1]))
                pt = jnp.where(keep_t, pt, 0.0)

                def compute(refs):
                    dv_acc_ref, dpt_ref = refs
                    plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))

                zeros = plgpu.layout_cast(jnp.full((bkv, bq), 0, jnp.float32), layout)
                dv_acc, dpt = pl.run_state(compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
                delta_v = plgpu.load(delta_smem, layout=COLS)
                dst = pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1])) * cfg.scale

                def compute_dk(acc_ref):
                    plgpu.wgmma(acc_ref, dst.astype(dtype), q_smem)

                return pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc)), dv_acc

            # Query tiles entirely before this key tile never see it (causal).
            carry = lax.cond(q_step * bq + (q_offset + bq - 1) >= kv_base, visit, lambda carry: carry, carry)
            for bar in (lse_c, m_c, do_c, delta_c, q_c):
                plgpu.barrier_arrive(bar)
            return carry

        pipeline = plgpu.emit_pipeline_warp_specialized(
            q_body,
            grid=(n_q,),
            max_concurrent_steps=min(mcs, n_q),
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=[
                plgpu.BlockSpec(block_shape=(bq, head_dim), index_map=lambda i: (i, 0), transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bq, head_dim), index_map=lambda i: (i, 0), transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bq,), index_map=lambda i: (i,)),
                plgpu.BlockSpec(block_shape=(bq,), index_map=lambda i: (i,)),
                plgpu.BlockSpec(block_shape=(bq, bkv), index_map=lambda i: (i, 0), transforms=[tiling, swizzle]),
            ],
        )
        pipeline(q_ref.at[b, :, h, :], do_ref.at[b, :, h, :], lse_ref.at[b, h, :], delta_ref.at[b, h, :],
                 m_ref.at[b, :, pl.ds(kv_base, bkv)])

    params = plgpu.CompilerParams(approx_math=True)
    q_tile = plgpu.SMEM((bq, head_dim), dtype, transforms=(tiling, swizzle))
    dq = plgpu.kernel(
        kernel_dq,
        out_type=jax.ShapeDtypeStruct(q.shape, dtype),
        scratch_types=[
            (q_tile, q_tile, plgpu.SMEM((bq,), jnp.float32), plgpu.SMEM((bq,), jnp.float32)),
            tuple(plgpu.Barrier() for _ in range(4)),
        ],
        compiler_params=params,
        grid=(heads, n_q, batch),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=2,
        thread_name="wg",
    )(q, k, v, do, lse, delta, mask)

    kv_tile = plgpu.SMEM((bkv, head_dim), dtype, transforms=(tiling, swizzle))
    per_head = jax.ShapeDtypeStruct((batch, width, heads, head_dim), dtype)
    dk, dv = plgpu.kernel(
        kernel_dkv,
        out_type=[per_head, per_head],
        scratch_types=[(kv_tile, kv_tile), tuple(plgpu.Barrier() for _ in range(2))],
        compiler_params=params,
        grid=(heads, n_kv, batch),
        grid_names=("heads", "kv_seq", "batch"),
        num_threads=2,
        thread_name="wg",
    )(q, k, v, do, lse, delta, mask)
    # float32 key/value gradients of the block's keys [B, width, H, D], summed over each GQA group
    shape = (batch, width, kv_heads, group, head_dim)
    dk = dk.reshape(shape).astype(jnp.float32).sum(axis=3)
    dv = dv.reshape(shape).astype(jnp.float32).sum(axis=3)
    return dq, dk, dv


def _block_mask(indices: Array, doc: Array, offset: int, dtype) -> Array:
    """``[B, R, offset + R]`` 0/1 selection of the queries ``offset ... offset + R - 1`` (``indices [B, R, K]``).

    A selected key counts when it is a causal key of the query's document (``doc [B, P]``, padded
    positions ``-1``); invalid, unused (``-1``) and repeated indices are dropped by the scatter.
    """
    batch, rows, _ = indices.shape
    width = offset + rows
    t = offset + jnp.arange(rows, dtype=jnp.int32)[None, :, None]
    safe = jnp.clip(indices, 0, width - 1)
    key_doc = jnp.take_along_axis(doc[:, None, :width], safe.reshape(batch, 1, -1), axis=2).reshape(indices.shape)
    valid = (indices >= 0) & (indices <= t) & (key_doc == doc[:, offset:width, None])
    cols = jnp.where(valid, indices, width)  # out of bounds: dropped
    row_idx = jnp.broadcast_to(jnp.arange(rows, dtype=jnp.int32)[None, :, None], indices.shape)
    batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None, None], indices.shape)
    return jnp.zeros((batch, rows, width), dtype).at[batch_idx, row_idx, cols].set(jnp.ones((), dtype), mode="drop")


def _blocks(padded: int, block: int) -> list[tuple[int, int]]:
    return [(offset, min(block, padded - offset)) for offset in range(0, padded, block)]


def _blocked_forward(q, k, v, indices, doc, cfg: _AttnConfig, block: int):
    outs, lses = [], []
    for offset, rows in _blocks(q.shape[1], block):
        (idx_b,) = _after(outs[-1] if outs else None, indices[:, offset:offset + rows])
        mask = _block_mask(idx_b, doc, offset, q.dtype)
        out, lse = _attn_forward(q[:, offset:offset + rows], k, v, mask, cfg, offset)
        outs.append(out)
        lses.append(lse)
    if len(outs) == 1:
        return outs[0], lses[0]
    return jnp.concatenate(outs, axis=1), jnp.concatenate(lses, axis=2)


def _blocked_backward(q, k, v, indices, doc, out, lse, do, cfg: _AttnConfig, block: int):
    dqs = []
    dk = jnp.zeros(k.shape, jnp.float32)
    dv = jnp.zeros(v.shape, jnp.float32)
    for offset, rows in _blocks(q.shape[1], block):
        part = slice(offset, offset + rows)
        (idx_b,) = _after(dk if offset else None, indices[:, part])
        mask = _block_mask(idx_b, doc, offset, q.dtype)
        dq_b, dk_b, dv_b = _attn_backward(cfg, q[:, part], k, v, mask, out[:, part], lse[:, :, part], do[:, part],
                                          offset)
        dqs.append(dq_b)
        # float32 sums over the blocks (one rounding of dk, dv at the end); the block covers keys < offset + rows
        widths = ((0, 0), (0, k.shape[1] - offset - rows), (0, 0), (0, 0))
        dk = dk + jnp.pad(dk_b, widths)
        dv = dv + jnp.pad(dv_b, widths)
    dq = dqs[0] if len(dqs) == 1 else jnp.concatenate(dqs, axis=1)
    return dq, dk.astype(k.dtype), dv.astype(v.dtype)


#: Operand layouts for :func:`jla.distributed.kernel_sharding.shard_layout_call` (lse is ``[B, HQ, T]``).
_ATTN_LAYOUTS = {"q": "token", "k": "token", "v": "token", "indices": "batch", "doc": "batch", "out": "token",
                 "lse": (0, 1), "do": "token"}


def _per_shard(fn, operands: dict, out_layouts):
    """``fn(**operands)`` per batch (and GQA head block) shard of the context mesh: the kernels run on each
    device's own rows, outside any autodiff (the custom VJP wraps the sharded calls, as the attention
    kernel's ``custom_partitioning`` does), so ``shard_map`` is never transposed."""
    from jla.distributed.kernel_sharding import shard_layout_call

    q, k = operands["q"], operands["k"]
    return shard_layout_call(fn, operands, {name: _ATTN_LAYOUTS[name] for name in operands},
                             out_layouts=out_layouts, heads=math.gcd(q.shape[2], k.shape[2]),
                             supports=lambda q, k, v, **_: supported_attention(q, k, v) is None)


def _sharded_forward(q, k, v, indices, doc, cfg: _AttnConfig, block: int):
    return _per_shard(lambda q, k, v, indices, doc: _blocked_forward(q, k, v, indices, doc, cfg, block),
                      dict(q=q, k=k, v=v, indices=indices, doc=doc), ("token", (0, 1)))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _blocked_attention(q, k, v, indices, doc, cfg: _AttnConfig, block: int):
    """Query blocks of ``block`` rows, each over the keys before its end (padded ``q, k, v``); the selection
    masks are rebuilt from ``indices`` in the backward pass instead of being kept as residuals."""
    return _sharded_forward(q, k, v, indices, doc, cfg, block)


def _blocked_attention_fwd(q, k, v, indices, doc, cfg: _AttnConfig, block: int):
    out, lse = _sharded_forward(q, k, v, indices, doc, cfg, block)
    return (out, lse), (q, k, v, indices, doc, out, lse)


def _blocked_attention_bwd(cfg: _AttnConfig, block: int, res, cotangents):
    do, _ = cotangents  # the log-sum-exp is returned under stop_gradient
    q, k, v, indices, doc, out, lse = res

    def backward(q, k, v, indices, doc, out, lse, do):
        return _blocked_backward(q, k, v, indices, doc, out, lse, do, cfg, block)

    dq, dk, dv = _per_shard(backward, dict(q=q, k=k, v=v, indices=indices, doc=doc, out=out, lse=lse,
                                           do=do.astype(q.dtype)), ("token", "token", "token"))
    return dq, dk, dv, None, None


_blocked_attention.defvjp(_blocked_attention_fwd, _blocked_attention_bwd)


def dsa_attention(
    q: Array,
    k: Array,
    v: Array,
    indices: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
    return_lse: bool = False,
    block_rows: int | None = None,
) -> Array | tuple[Array, Array]:
    """Softmax attention of every query head over the keys in ``indices [B, T, K]`` (``-1`` = unused).

    Same contract as :func:`jla.ops.sparse_attention.dsa` (selected keys that are not causal keys
    of the query's document are ignored, repeated ones count once). The queries run in blocks of
    ``block_rows`` (a multiple of 128; default: the whole sequence up to 8192 tokens, else 4096), each
    against the keys before its end with a ``[B, rows, end]`` bf16 selection mask built from its
    indices (rebuilt in the backward pass). With ``return_lse`` also returns the natural-log
    log-sum-exp ``[B, HQ, T]`` (no gradient; ``1e30 / log2(e)`` for a query without keys).
    """
    reason = supported_attention(q, k, v)
    if reason is not None:
        raise ValueError(f"mosaic_gpu dsa attention: {reason}")
    batch, length = q.shape[:2]
    if indices.ndim != 3 or indices.shape[:2] != (batch, length):
        raise ValueError(f"indices must be [B, T, K]; got {indices.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    cfg = _AttnConfig(scale=float(q.shape[-1] ** -0.5 if scale is None else scale))
    pad = -length % _ATTN_PAD
    padded = length + pad
    doc, _ = _documents(segment_ids, batch, length)
    indices = lax.stop_gradient(indices).astype(jnp.int32)
    if pad:  # padded queries select nothing and no real query selects a padded key
        widths = ((0, 0), (0, pad), (0, 0), (0, 0))
        q, k, v = (jnp.pad(x, widths) for x in (q, k, v))
        indices = jnp.pad(indices, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
        doc = jnp.pad(doc, ((0, 0), (0, pad)), constant_values=-1)
    block = _block_rows(padded, block_rows, _ATTN_PAD)
    out, lse = _blocked_attention(q, k, v, indices, doc, cfg, block)
    if pad:
        out = out[:, :length]
    if not return_lse:
        return out
    return out, lax.stop_gradient(lse[:, :, :length]) / _LOG2E


# ----------------------------------------------------------------------------------------------
# DeepSeek-V3.2's indexer loss over the selected keys (sparse training stage)
# ----------------------------------------------------------------------------------------------


def sparse_index_scores(
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    indices: Array,
    *,
    scale: float | None = None,
) -> Array:
    """Indexer scores of the selected keys only: ``[B, T, K]`` float32, ``I[t, j] = sum_h w[t, h] relu(scale
    q_idx[t, h] . k_idx[indices[t, j]])`` (slots that are not valid hold the score of key 0: mask them).

    Differentiable in ``q_idx``, ``k_idx`` and ``w_idx``: the KL gradient of DeepSeek-V3.2's sparse stage
    is nonzero only on the selected keys, so its backward costs ``O(T K HI DI)`` instead of the dense
    ``O(T^2 HI DI)``. Plain JAX: the selected index keys are gathered per chunk of queries (bf16/fp16
    operands, float32 accumulation, as in the kernel), and each chunk is recomputed in the backward
    pass (``jax.checkpoint``), so memory stays at one chunk.
    """
    batch, length, heads, dim = q_idx.shape
    top = indices.shape[-1]
    scale = dim ** -0.5 if scale is None else float(scale)
    safe = jnp.clip(lax.stop_gradient(indices), 0, length - 1)

    k_f32 = k_idx.astype(jnp.float32)  # gathers read k_idx; its gradient accumulates here in float32

    @jax.checkpoint
    def chunk_scores(qc, wc, ic):
        kc = _gather_keys(k_f32, k_idx, ic, k_idx.shape[1:])  # [B, C, K, DI]
        s = jnp.einsum("bchd,bckd->bchk", qc, kc, preferred_element_type=jnp.float32) * scale
        return jnp.einsum("bchk,bch->bck", jax.nn.relu(s), wc.astype(jnp.float32))

    chunk = _row_chunk(length, top * dim * k_idx.dtype.itemsize + 2 * top * heads * 4)
    return _chunked(chunk_scores, chunk, q_idx, w_idx, safe)


def sparse_attention_target(
    q: Array,
    k: Array,
    lse: Array,
    indices: Array,
    valid: Array,
    *,
    scale: float | None = None,
) -> Array:
    """DeepSeek-V3.2's indexer target on the selected keys: ``[B, T, K]`` float32, no gradient.

    The main attention's probabilities of the selected keys, recomputed from the kernel's natural-log
    log-sum-exp ``lse [B, HQ, T]``, summed over all query heads and L1-normalized per query
    (``dsa_attention_target`` of the dense probabilities, at the slots ``valid [B, T, K]``; zero elsewhere).
    The selected keys are gathered per chunk of queries.
    """
    batch, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    top = indices.shape[-1]
    scale = dim ** -0.5 if scale is None else float(scale)
    q, k, lse = (lax.stop_gradient(x) for x in (q, k, lse))
    safe = jnp.clip(lax.stop_gradient(indices), 0, length - 1)
    lse_t = lse.transpose(0, 2, 1)  # [B, T, HQ]

    def chunk_target(qc, lc, ic, vc):
        rows = qc.shape[1]
        kc = jax.vmap(lambda keys, r: keys[r])(k, ic)  # [B, C, K, H, D]
        qg = qc.reshape(batch, rows, kv_heads, group, dim)
        s = jnp.einsum("bchgd,bckhd->bchgk", qg, kc, preferred_element_type=jnp.float32) * scale
        p = jnp.exp(s - lc.reshape(batch, rows, kv_heads, group)[..., None])
        total = jnp.sum(jnp.where(vc[:, :, None, None, :], p, 0.0), axis=(2, 3))
        return total / jnp.maximum(jnp.sum(total, axis=-1, keepdims=True), 1e-30)

    chunk = _row_chunk(length, top * kv_heads * dim * k.dtype.itemsize + 2 * top * q_heads * 4)
    return _chunked(chunk_target, chunk, q, lse_t, safe, valid)


def selected_attention_target(
    q: Array,
    k: Array,
    lse: Array,
    indices: Array,
    valid: Array,
    *,
    scale: float | None = None,
    segment_ids: Array | None = None,
    block_rows: int | None = None,
) -> Array:
    """:func:`sparse_attention_target` on the indexer kernel: ``[B, T, K]`` float32, no gradient.

    Per query block (as :func:`index_topk`) and key/value head, the indexer forward kernel with
    ``exp_scale`` sums ``exp(scale q_h . k - lse_h)`` over the group's query heads for every causal
    (query, key) tile (one WGMMA per head, f32 accumulation); the selected keys are gathered from the
    ``[B, rows, end]`` sums, kept where ``valid`` and L1-normalized per query. Cost: the ``Q K^T``
    products of dense causal attention, instead of gathering ``top_k`` keys per query.
    """
    reason = supported_attention(q, k, k)
    if reason is not None:
        raise ValueError(f"mosaic_gpu dsa selected_attention_target: {reason}")
    batch, length, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    group = q_heads // kv_heads
    scale = dim ** -0.5 if scale is None else float(scale)
    q, k, lse = (lax.stop_gradient(x) for x in (q, k, lse))
    doc, _ = _documents(segment_ids, batch, length)
    doc = doc.astype(jnp.int32)
    lse2 = lse.astype(jnp.float32) * _LOG2E  # [B, HQ, T], log2 domain
    pad = -length % (2 * _IDX_TILE)
    indices = lax.stop_gradient(indices).astype(jnp.int32)
    if pad:
        q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
        lse2 = jnp.pad(lse2, ((0, 0), (0, 0), (0, pad)))
        doc = jnp.pad(doc, ((0, 0), (0, pad)), constant_values=-1)
        indices = jnp.pad(indices, ((0, 0), (0, pad), (0, 0)), constant_values=-1)
    padded = q.shape[1]
    lse2 = lse2.reshape(batch, q_heads * padded)  # head h, query tile i: 128-wide tile h * (P / 128) + i
    tiles = padded // _IDX_TILE
    block = _block_rows(padded, block_rows, 2 * _IDX_TILE)
    parts = []
    for offset in range(0, padded, block):
        rows = min(block, padded - offset)
        (q_b,) = _after(parts[-1] if parts else None, q)
        total = None
        for h in range(kv_heads):
            heads = slice(h * group, (h + 1) * group)
            qh = q_b[:, offset:offset + rows, heads].transpose(0, 2, 1, 3).reshape(batch, group * rows, dim)
            part = _index_call(qh, k[:, :, h], lse2, doc, heads=group, rows=rows, offset=offset,
                               has_doc=segment_ids is not None, exp_scale=scale * _LOG2E,
                               w_tiles=(h * group * tiles + offset // _IDX_TILE, tiles))
            total = part if total is None else total + part
        safe = jnp.clip(indices[:, offset:offset + rows], 0, offset + rows - 1)
        parts.append(jnp.take_along_axis(total, safe, axis=2))
    total = jnp.concatenate(parts, axis=1)[:, :length] if len(parts) > 1 else parts[0][:, :length]
    total = jnp.where(valid, total, 0.0)
    return total / jnp.maximum(jnp.sum(total, axis=-1, keepdims=True), 1e-30)


def dsa_layer(
    q: Array,
    k: Array,
    v: Array,
    q_idx: Array,
    k_idx: Array,
    w_idx: Array,
    *,
    top_k: int,
    scale: float | None = None,
    indexer_loss_coef: float = 0.0,
    segment_ids: Array | None = None,
    block_rows: int | None = None,
    topk_method: str = "radix",
    index_fp8: bool = False,
) -> Array:
    """The DSA mixer's core on the kernels: indexer top-k, selected-key attention and the indexer KL gradient.

    ``q [B, T, HQ, D]``, ``k, v [B, T, H, D]``; the indexer's ``q_idx [B, T, HI, DI]``, shared key
    ``k_idx [B, T, DI]`` and head weights ``w_idx [B, T, HI]`` (``scale`` of the indexer: ``DI ** -0.5``).
    Equal to the plain path of :class:`jla.layers.sparse_attention.DeepSeekSparseAttention`: the same
    ``min(top_k, T)`` keys per query (exact top-k of the kernel scores), exact attention over them, and
    with ``indexer_loss_coef > 0`` the gradient of ``coef * KL(target || softmax(scores))`` over the
    selected keys summed over queries, injected into the indexer inputs by
    :func:`jla.ops.sparse_attention.attach_indexer_grad` (the selected scores come from the top-k and
    their gradient runs on the indexer backward kernels, :func:`selected_index_scores`; the target is
    :func:`selected_attention_target`). ``topk_method``: ``"radix"`` (the radix-select kernel,
    :func:`jla.ops.mosaic_gpu.dsa_topk.radix_top_k`) or ``"sort"`` (``lax.top_k``); both select the same
    keys. ``index_fp8=True`` selects with the approximate FP8 indexer (see :func:`_index_topk`; the
    indexer KL then sees the FP8 scores of the selected keys, its gradient the bf16 ones). Under a
    context mesh (``jax.set_mesh``) the kernels run per batch shard (and GQA head block) through
    :func:`jla.distributed.kernel_sharding.shard_layout_call`, with the custom VJPs outside the
    ``shard_map`` calls; XLA partitions the plain-JAX parts.
    """
    from jla.distributed.kernel_sharding import shard_layout_call
    from jla.ops.sparse_attention import attach_indexer_grad

    for name, reason in (("attention", supported_attention(q, k, v)),
                         ("indexer", supported_index(q_idx, k_idx, w_idx))):
        if reason is not None:
            raise ValueError(f"mosaic_gpu dsa {name}: {reason}")
    coef = float(indexer_loss_coef)

    def topk(q_idx, k_idx, w_idx, segment_ids):
        return _index_topk(q_idx, k_idx, w_idx, top_k, segment_ids=segment_ids, block_rows=block_rows,
                           method=topk_method, fp8=index_fp8)

    indices, values = shard_layout_call(
        topk, {"q_idx": q_idx, "k_idx": k_idx, "w_idx": w_idx, "segment_ids": segment_ids},
        {"q_idx": "batch", "k_idx": "batch", "w_idx": "batch", "segment_ids": "batch"},
        out_layouts=("batch", "batch"),
        supports=lambda q_idx, k_idx, w_idx, segment_ids: supported_index(q_idx, k_idx, w_idx) is None)
    out, lse = dsa_attention(q, k, v, indices, scale=scale, segment_ids=segment_ids, return_lse=True,
                             block_rows=block_rows)
    if coef > 0:
        valid = valid_indices(indices, segment_ids)
        scores = selected_index_scores(q_idx, k_idx, w_idx, indices, values, valid, block_rows=block_rows)

        def target_fn(q, k, lse, indices, valid, segment_ids):
            return selected_attention_target(q, k, lse, indices, valid, scale=scale, segment_ids=segment_ids,
                                             block_rows=block_rows)

        operands = dict(q=q, k=k, lse=lse, indices=indices, valid=valid, segment_ids=segment_ids)
        target = shard_layout_call(target_fn, operands, {name: "batch" for name in operands}, out_layouts="batch",
                                   supports=lambda q, k, **_: supported_attention(q, k, k) is None)
        out = attach_indexer_grad(out, scores, target, valid.astype(jnp.float32), coef)
    return out
