# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Sparse Attention at long context: the kernel path against the plain-JAX path and dense attention.

DeepSeek-V3.2-like shapes per GPU: ``B = 1``, 16 query heads sharing one key/value head (MQA, as
DeepSeek-V3.2 runs MLA for DSA, sharded over 8-way tensor parallelism) of ``D = 128`` (our kernels
take 64/128, not MLA's 576/512), bf16; a lightning indexer of 64 heads of 128 and ``top_k = 2048``
(DeepSeek-V3.2's ``index_n_heads``, ``index_head_dim``, ``index_topk``); ``T`` from 1K to 128K.
Rows, interleaved on one GPU in one run (median CUPTI kernel time, ``harness.py``), written to the
JSON after every row:

* ``layer``: the whole DSA core, forward and forward+backward (gradients of q, k, v and of the
  indexer's q, k, w; DeepSeek-V3.2's indexer KL on the selected keys with coefficient 1):
  ``mosaic_gpu`` is ``dsa_layer`` (indexer blocks + exact top-k, selected-key attention, the indexer
  backward kernels); ``mosaic_gpu (gather KL)`` the same with the plain-JAX gather form of the KL
  (``sparse_index_scores``, the previous path); ``xla`` the plain-JAX core (dense ``[T, T]`` scores,
  ``lax.top_k``, ``dsa``, dense KL) where it fits;
* ``index_topk`` (indexer + top-k, forward), ``attention`` (``dsa_attention`` on fixed indices, forward
  and forward+backward), ``indexer_kl`` (the selected scores + ``sparse_attention_target`` + KL: forward
  and the indexer backward: the kernels, and the gather forms ``sparse_index_scores`` and
  ``sparse_attention_target``);
* dense causal attention of all keys: cuDNN (``jax.nn.dot_product_attention``) and the Mosaic GPU
  flash attention.

    python benchmarks/bench_dsa_long.py --out benchmarks/results/dsa_long.json [--lengths 16384,32768]
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

from jla.ops.mosaic_gpu.attention import flash_attention  # noqa: E402
from jla.ops.mosaic_gpu.dsa import (  # noqa: E402
    _index_topk,
    dsa_attention,
    dsa_layer,
    index_topk,
    selected_attention_target,
    selected_index_scores,
    sparse_attention_target,
    sparse_index_scores,
    valid_indices,
)
from jla.ops.sparse_attention import (  # noqa: E402
    attach_indexer_grad,
    dsa,
    dsa_attention_target,
    dsa_index_scores,
    dsa_select,
)

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


def _plain_core(q, k, v, qi, ki, wi):
    scores = dsa_index_scores(qi, ki, wi)
    indices = dsa_select(scores, TOP_K)
    o, probs, selected = dsa(q, k, v, indices, return_probs=True)
    return attach_indexer_grad(o, scores, dsa_attention_target(probs), selected.astype(jnp.float32), 1.0)


def _kernel_core(q, k, v, qi, ki, wi):
    return dsa_layer(q, k, v, qi, ki, wi, top_k=TOP_K, indexer_loss_coef=1.0)


def _gather_kl_core(q, k, v, qi, ki, wi):
    """The kernel path with the plain-JAX gather form of the indexer KL (before the backward kernels)."""
    indices = index_topk(qi, ki, wi, TOP_K)
    out, lse = dsa_attention(q, k, v, indices, return_lse=True)
    valid = valid_indices(indices)
    scores = sparse_index_scores(qi, ki, wi, indices)
    target = sparse_attention_target(q, k, lse, indices, valid)
    return attach_indexer_grad(out, scores, target, valid.astype(jnp.float32), 1.0)


def _kl(scores, target, valid):
    log_q = jax.nn.log_softmax(jnp.where(valid, scores, -jnp.inf), axis=-1)
    return jnp.sum(jnp.where(valid, target * (jnp.log(jnp.maximum(target, 1e-30)) - log_q), 0.0))


def _kl_kernel(qi, ki, wi, q, k, lse, indices, values, valid):
    scores = selected_index_scores(qi, ki, wi, indices, values, valid)
    return _kl(scores, selected_attention_target(q, k, lse, indices, valid), valid)


def _kl_gather(qi, ki, wi, q, k, lse, indices, values, valid):
    del values
    scores = sparse_index_scores(qi, ki, wi, indices)
    return _kl(scores, sparse_attention_target(q, k, lse, indices, valid), valid)


def _time(row, key, fn, args, iterations):
    try:
        row[key] = harness.median_kernel_ms(jax.jit(fn), *args, iterations=iterations)["median_ms"]
    except Exception as err:  # record the failure (e.g. out of memory), keep measuring the rest
        row[key + "_error"] = harness.error_summary(f"{type(err).__name__}: {err}")
    gc.collect()


def _grad_fn(core, n):
    """Forward + backward of ``sum(core(...) * do)`` for the first ``n`` arguments."""
    def run(*args):
        *inputs, do = args
        loss = lambda *a: jnp.sum(core(*a).astype(jnp.float32) * do.astype(jnp.float32))  # noqa: E731
        return jax.grad(loss, argnums=tuple(range(n)))(*inputs)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/dsa_long.json")
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    parser.add_argument("--plain-max", type=int, default=16384, help="longest T for the plain-JAX rows")
    parser.add_argument("--gather-max", type=int, default=32768, help="longest T for the gather-form KL rows")
    parser.add_argument("--parts", default="layer,index_topk,attention,indexer_kl,dense")
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    before = harness.gpu_snapshot()
    rows = []

    def emit(row):
        print(row, flush=True)
        rows.append(row)
        harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before, rows=rows,
                                          note="one H100 of a shared node; other jobs used the other GPUs"))

    for length in (int(x) for x in args.lengths.split(",")):
        iterations = max(5, min(30, (30 * 16384) // length))
        q, k, v, do, qi, ki, wi = _inputs(length)
        base = dict(T=length, B=1, HQ=Q_HEADS, H=KV_HEADS, D=HEAD_DIM, HI=INDEX_HEADS, DI=INDEX_DIM, top_k=TOP_K,
                    iterations=iterations)
        if "layer" in parts:
            impls = [("mosaic_gpu", _kernel_core)]
            if length <= args.gather_max:
                impls.append(("mosaic_gpu (gather KL)", _gather_kl_core))
            if length <= args.plain_max:
                impls.append(("xla", _plain_core))
            for impl, core in impls:
                row = dict(base, part="layer", impl=impl)
                _time(row, "fwd_ms", lambda *a, core=core: (core(*a),), (q, k, v, qi, ki, wi), iterations)
                _time(row, "fwdbwd_ms", _grad_fn(core, 6), (q, k, v, qi, ki, wi, do), iterations)
                emit(row)
        if "index_topk" in parts:
            row = dict(base, part="index_topk", impl="mosaic_gpu")
            _time(row, "fwd_ms", lambda a, b, c: (index_topk(a, b, c, TOP_K),), (qi, ki, wi), iterations)
            row["indexer_tflop"] = length * (length + 1) / 2 * INDEX_HEADS * INDEX_DIM * 2.0 / 1e12
            emit(row)
        indices = values = None
        if parts & {"attention", "indexer_kl"}:
            try:
                topk = jax.jit(lambda a, b, c: _index_topk(a, b, c, TOP_K))
                indices, values = jax.block_until_ready(topk(qi, ki, wi))
            except Exception as err:
                emit(dict(base, part="indices", error=harness.error_summary(f"{type(err).__name__}: {err}")))
                indices = values = None
        if "attention" in parts and indices is not None:
            row = dict(base, part="attention", impl="mosaic_gpu")
            fn = lambda q, k, v, idx: dsa_attention(q, k, v, idx)  # noqa: E731
            _time(row, "fwd_ms", lambda *a: (fn(*a),), (q, k, v, indices), iterations)
            _time(row, "fwdbwd_ms", lambda q, k, v, idx, do: jax.grad(
                lambda q, k, v: jnp.sum(fn(q, k, v, idx).astype(jnp.float32) * do.astype(jnp.float32)),
                argnums=(0, 1, 2))(q, k, v), (q, k, v, indices, do), iterations)
            emit(row)
        if "indexer_kl" in parts and indices is not None:
            try:
                lse = jax.block_until_ready(jax.jit(
                    lambda q, k, v, idx: dsa_attention(q, k, v, idx, return_lse=True)[1])(q, k, v, indices))
                valid = jax.block_until_ready(jax.jit(valid_indices)(indices))
            except Exception as err:
                emit(dict(base, part="indexer_kl", error=harness.error_summary(f"{type(err).__name__}: {err}")))
                lse = None
            kls = [("mosaic_gpu", _kl_kernel)] + ([("xla (gathered keys)", _kl_gather)]
                                                  if length <= args.gather_max else [])
            for impl, kl in (kls if lse is not None else []):
                row = dict(base, part="indexer_kl", impl=impl)
                operands = (qi, ki, wi, q, k, lse, indices, values, valid)
                _time(row, "fwd_ms", lambda *a, kl=kl: (kl(*a),), operands, iterations)
                _time(row, "fwdbwd_ms", lambda *a, kl=kl: jax.grad(kl, argnums=(0, 1, 2))(*a), operands, iterations)
                emit(row)
            del lse
        del indices, values
        gc.collect()
        if "dense" in parts:
            for impl, fn in (("cudnn_dense_causal", lambda q, k, v: jax.nn.dot_product_attention(
                    q, k, v, is_causal=True, implementation="cudnn")),
                             ("mosaic_gpu_dense_causal", lambda q, k, v: flash_attention(q, k, v))):
                row = dict(base, part="dense_attention", impl=impl)
                _time(row, "fwd_ms", lambda *a, fn=fn: (fn(*a),), (q, k, v), iterations)
                _time(row, "fwdbwd_ms", _grad_fn(fn, 3), (q, k, v, do), iterations)
                emit(row)
    harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before,
                                      gpu_after=harness.gpu_snapshot(), rows=rows,
                                      note="one H100 of a shared node; other jobs used the other GPUs"))


if __name__ == "__main__":
    main()
