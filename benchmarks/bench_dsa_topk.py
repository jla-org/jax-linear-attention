# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The DSA top-k: the radix-select kernel against ``lax.top_k``, alone, in ``index_topk`` and in the layer.

DeepSeek-V3.2-like shapes as ``bench_dsa_long.py``: ``B = 1``, indexer 64 heads of 128, ``top_k = 2048``,
attention 16 query heads sharing one key/value head of 128, bf16. Rows, interleaved per length on one
GPU in one run (median CUPTI kernel time, ``harness.py``), written to the JSON after every row:

* ``topk_block``: the top-k alone on the last (largest) query block of the indexer's causal scores
  (``[1, rows, T]`` float32, 4096 rows): ``radix`` (``radix_top_k``) and ``sort`` (``lax.top_k`` on causal
  prefixes, the previous path), and the indexer kernel that writes the block;
* ``indexer``: the indexer kernel over every query block without the top-k (the scores ``index_topk`` selects
  from), and ``index_topk``: indexer + top-k of the whole sequence (``index_topk(method=...)``), forward;
* ``candidates`` (untimed): per query block, the radix top-k's candidates per row after its first three
  passes (the entries its compaction keeps), mean and maximum;
* ``fp8``: ``index_topk`` with the approximate FP8 indexer (radix top-k), timed, and the recall of its
  selection against the exact one (per query with more than ``top_k`` keys: mean, minimum, 1st percentile);
* ``layer``: ``dsa_layer`` with ``topk_method`` radix and sort, forward and forward + backward (gradients of
  q, k, v and of the indexer's q, k, w; DeepSeek-V3.2's indexer KL with coefficient 1).

    python benchmarks/bench_dsa_topk.py --out benchmarks/results/dsa_topk.json [--lengths 16384,32768]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.ops.mosaic_gpu.dsa import _BLOCK_ROWS, _block_rows, _index_block, _pad_index_inputs  # noqa: E402
from jla.ops.mosaic_gpu.dsa import _prefix_top_k  # noqa: E402
from jla.ops.mosaic_gpu.dsa import dsa_layer  # noqa: E402
from jla.ops.mosaic_gpu.dsa import index_topk  # noqa: E402
from jla.ops.mosaic_gpu.dsa_topk import radix_top_k  # noqa: E402
from jla.ops.sparse_attention import _documents  # noqa: E402

Q_HEADS, KV_HEADS, HEAD_DIM = 16, 1, 128
INDEX_HEADS, INDEX_DIM, TOP_K = 64, 128, 2048


def _inputs(length: int):
    keys = jax.random.split(jax.random.key(length), 8)
    bf16 = jnp.bfloat16
    q = jax.random.normal(keys[0], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    k = jax.random.normal(keys[1], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    v = jax.random.normal(keys[2], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    do = jax.random.normal(keys[3], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    qi = jax.random.normal(keys[4], (1, length, INDEX_HEADS, INDEX_DIM), jnp.float32).astype(bf16)
    ki = jax.random.normal(keys[5], (1, length, INDEX_DIM), jnp.float32).astype(bf16)
    wi = jax.random.normal(keys[6], (1, length, INDEX_HEADS), jnp.float32) * INDEX_HEADS ** -0.5
    return q, k, v, do, qi, ki, wi


def _block_shape(length: int) -> tuple[int, int]:
    """Offset and rows of the last query block of the indexer (as ``index_topk`` splits the padded sequence)."""
    padded = length + (-length % 256)
    rows = _BLOCK_ROWS if padded > 8192 else padded
    return padded - rows, rows


def _last_block(qi, ki, wi):
    """The indexer's causal scores of the last query block ``[1, rows, T]``."""
    length = qi.shape[1]
    offset, rows = _block_shape(length)
    doc, _ = _documents(None, 1, length)
    q_p, k_p, w_p, doc = _pad_index_inputs(qi, ki, wi.astype(jnp.float32) * INDEX_DIM ** -0.5, doc.astype(jnp.int32))
    return _index_block(q_p, k_p, w_p, doc, offset=offset, rows=rows, has_doc=False)


def _indexer_all(qi, ki, wi):
    """The indexer kernel over every query block as ``index_topk`` runs it, without the top-k (each block's
    scores reduced to their last column, so the blocks are freed one by one)."""
    length = qi.shape[1]
    doc, _ = _documents(None, 1, length)
    q_p, k_p, w_p, doc = _pad_index_inputs(qi, ki, wi.astype(jnp.float32) * INDEX_DIM ** -0.5, doc.astype(jnp.int32))
    padded = q_p.shape[1]
    block = _block_rows(padded, None, 256)
    outs = []
    for offset in range(0, padded, block):
        rows = min(block, padded - offset)
        outs.append(_index_block(q_p, k_p, w_p, doc, offset=offset, rows=rows, has_doc=False)[:, :, -1])
    return jnp.concatenate(outs, axis=1)


def _candidates(scores, offset):
    """Candidates per row ``[B, R]`` after the radix top-k's first passes (the entries sharing the ``k``-th largest
    entry's 12-bit key prefix), as ``radix_top_k`` computes them before its compaction."""
    from jla.ops.mosaic_gpu import dsa_topk as t

    batch, rows, width = scores.shape
    count = dict(offset=offset, splits=t._default_splits(batch, rows, width // t._TC))
    base = jnp.full((batch, rows), -8, jnp.int32)
    hist = t._count_call(scores, base, shift=28, finite=True, **count).sum(axis=1)
    kk = jnp.minimum(jnp.int32(TOP_K), hist[:, t._BOUNDS])
    digit = t._digit(t._at_least(hist), kk)
    value = base + digit
    for p in range(1, t._COMPACT_AFTER):
        base = value * 16
        counts = t._at_least(t._count_call(scores, base, shift=28 - 4 * p, finite=False, **count))
        digit = t._digit(counts.sum(axis=1), kk)
        value = base + digit
    above, at_least = t._split_counts(counts, digit)
    return (at_least - above).sum(axis=1)


def _recall(approx, exact):
    """Per query ``[B, T]``: the fraction of the exact selection's keys that the approximate one also holds."""
    big = jnp.iinfo(jnp.int32).max
    e = jnp.sort(jnp.where(exact < 0, big, exact), axis=-1)
    a = jnp.where(approx < 0, -2, approx)
    pos = jax.vmap(jax.vmap(jnp.searchsorted))(e, a)
    found = jnp.take_along_axis(e, jnp.clip(pos, 0, e.shape[-1] - 1), axis=-1) == a
    return found.sum(-1) / jnp.maximum((exact >= 0).sum(-1), 1)


def _time(row, key, fn, args, iterations):
    try:
        row[key] = harness.median_kernel_ms(jax.jit(fn), *args, iterations=iterations)["median_ms"]
    except Exception as err:  # record the failure (e.g. out of memory), keep measuring the rest
        row[key + "_error"] = harness.error_summary(f"{type(err).__name__}: {err}")
    gc.collect()


def _grad_fn(core, n):
    def run(*args):
        *inputs, do = args
        loss = lambda *a: jnp.sum(core(*a).astype(jnp.float32) * do.astype(jnp.float32))  # noqa: E731
        return jax.grad(loss, argnums=tuple(range(n)))(*inputs)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/dsa_topk.json")
    parser.add_argument("--lengths", default="4096,16384,32768,65536,131072")
    parser.add_argument("--sort-max", type=int, default=32768, help="longest T for the lax.top_k rows")
    parser.add_argument("--layer-max", type=int, default=65536, help="longest T for the layer rows")
    parser.add_argument("--recall-max", type=int, default=32768, help="longest T for the FP8 recall measurement")
    parser.add_argument("--parts", default="topk_block,indexer,index_topk,fp8,layer")
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    before = harness.gpu_snapshot()
    rows = []

    def emit(row):
        print(row, flush=True)
        rows.append(row)
        harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before, rows=rows,
                                          note="one H100, the only timed job on it; test runs of other jobs "
                                               "may share it (provisional)"))

    for length in (int(x) for x in args.lengths.split(",")):
        iterations = max(5, min(30, (30 * 16384) // length))
        q, k, v, do, qi, ki, wi = _inputs(length)
        base = dict(T=length, B=1, HQ=Q_HEADS, H=KV_HEADS, D=HEAD_DIM, HI=INDEX_HEADS, DI=INDEX_DIM, top_k=TOP_K,
                    iterations=iterations)
        methods = ["radix"] + (["sort"] if length <= args.sort_max else [])
        if "topk_block" in parts:
            try:
                offset = _block_shape(length)[0]
                scores = jax.block_until_ready(jax.jit(_last_block)(qi, ki, wi))
            except Exception as err:
                emit(dict(base, part="topk_block", error=harness.error_summary(f"{type(err).__name__}: {err}")))
                scores = None
            if scores is not None:
                block = dict(base, rows=scores.shape[1], width=scores.shape[2], offset=offset)
                row = dict(block, part="topk_block", impl="indexer kernel (writes the block)")
                _time(row, "fwd_ms", lambda a, b, c: (_last_block(a, b, c),), (qi, ki, wi), iterations)
                emit(row)
                for method in methods:
                    row = dict(block, part="topk_block", impl=method)
                    if method == "radix":
                        fn = lambda s, offset=offset: radix_top_k(s, TOP_K, offset=offset)  # noqa: E731
                    else:
                        fn = lambda s, offset=offset: _prefix_top_k(s, TOP_K, offset, 2)  # noqa: E731
                    _time(row, "fwd_ms", fn, (scores,), iterations)
                    if method == "radix" and "fwd_ms" in row:
                        row["kernels_ms"] = harness.kernel_breakdown_ms(jax.jit(fn), scores, iterations=5)
                    emit(row)
            del scores
            gc.collect()
        if "indexer" in parts:
            row = dict(base, part="indexer", impl="mosaic_gpu (bf16), every block, no top-k")
            _time(row, "fwd_ms", lambda a, b, c: (_indexer_all(a, b, c),), (qi, ki, wi), iterations)
            emit(row)
        if "index_topk" in parts:
            for method in methods:
                row = dict(base, part="index_topk", impl=method)
                _time(row, "fwd_ms", lambda a, b, c, method=method: (index_topk(a, b, c, TOP_K, method=method),),
                      (qi, ki, wi), iterations)
                emit(row)
        if "candidates" in parts:  # untimed: the candidate counts per query block (the compaction's buffer use)
            doc, _ = _documents(None, 1, length)
            q_p, k_p, w_p, doc = _pad_index_inputs(qi, ki, wi.astype(jnp.float32) * INDEX_DIM ** -0.5,
                                                   doc.astype(jnp.int32))
            padded = q_p.shape[1]
            block = _block_rows(padded, None, 256)
            for offset in range(0, padded, block):
                rows = min(block, padded - offset)
                n = jax.jit(lambda q, kk, w, d, offset=offset, rows=rows: _candidates(
                    _index_block(q, kk, w, d, offset=offset, rows=rows, has_doc=False), offset))(q_p, k_p, w_p, doc)
                emit(dict(base, part="candidates", block_offset=offset, block_width=offset + rows,
                          max_candidates=int(n.max()), mean_candidates=float(n.mean())))
        if "fp8" in parts:
            row = dict(base, part="index_topk", impl="radix, FP8 indexer (approximate)")
            fp8 = lambda a, b, c: index_topk(a, b, c, TOP_K, method="radix", fp8=True)  # noqa: E731
            _time(row, "fwd_ms", lambda a, b, c: (fp8(a, b, c),), (qi, ki, wi), iterations)
            try:  # selection recall against the exact radix path, over the queries with more than top_k keys
                if length > args.recall_max:
                    raise ValueError(f"recall measured up to --recall-max {args.recall_max} tokens (compile time)")
                exact = jax.jit(lambda a, b, c: index_topk(a, b, c, TOP_K, method="radix"))(qi, ki, wi)
                recall = jax.jit(_recall)(jax.jit(fp8)(qi, ki, wi), exact)[:, TOP_K:]
                row.update(recall_mean=float(recall.mean()), recall_min=float(recall.min()),
                           recall_p01=float(jnp.percentile(recall, 1.0)))
                del exact, recall
            except Exception as err:
                row["recall_error"] = harness.error_summary(f"{type(err).__name__}: {err}")
            emit(row)
            gc.collect()
        if "layer" in parts and length <= args.layer_max:
            for method in methods:
                core = lambda *a, method=method: dsa_layer(*a, top_k=TOP_K, indexer_loss_coef=1.0,  # noqa: E731
                                                           topk_method=method)
                row = dict(base, part="layer", impl=method)
                _time(row, "fwd_ms", lambda *a, core=core: (core(*a),), (q, k, v, qi, ki, wi), iterations)
                _time(row, "fwdbwd_ms", _grad_fn(core, 6), (q, k, v, qi, ki, wi, do), iterations)
                emit(row)
        del q, k, v, do, qi, ki, wi
        gc.collect()
    harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before,
                                      gpu_after=harness.gpu_snapshot(), rows=rows,
                                      note="one H100, the only timed job on it; test runs of other jobs "
                                           "may share it (provisional)"))


if __name__ == "__main__":
    main()
