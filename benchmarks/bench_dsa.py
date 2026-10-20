# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Sparse Attention: the Mosaic GPU kernels against the plain-JAX ops and dense cuDNN attention.

Shapes: ``B = 1``, ``T`` in {4096, 8192}, 16 query and 4 key/value heads of ``D = 128`` (GQA group 4),
bf16; a lightning indexer of 64 heads of 128 (the ``SparseAttentionConfig`` defaults); top-k in
{512, 2048}. Parts, interleaved on one GPU in one run (median CUPTI kernel time, ``harness.py``):

* ``indexer`` (forward): ``dsa_index_scores`` (XLA) against the kernel ``index_scores``;
* ``select``: ``dsa_select`` (``lax.top_k``) on the scores, shared by both paths;
* ``attention`` (``fwd`` and ``fwdbwd`` for a fixed output cotangent): ``dsa`` (XLA) against the
  kernel ``dsa_attention`` (including the XLA selection-mask build) and dense causal attention
  (cuDNN through ``jax.nn.dot_product_attention`` and the Mosaic GPU flash attention), which do
  the work of selecting every causal key.

    python benchmarks/bench_dsa.py --out benchmarks/results/dsa.json
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.ops.mosaic_gpu.attention import flash_attention  # noqa: E402
from jla.ops.mosaic_gpu.dsa import dsa_attention, index_scores, select_top_k  # noqa: E402
from jla.ops.sparse_attention import dsa, dsa_index_scores, dsa_select  # noqa: E402

Q_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
INDEX_HEADS, INDEX_DIM = 64, 128


def _inputs(length: int):
    keys = jax.random.split(jax.random.key(length), 7)
    bf16 = jnp.bfloat16
    q = jax.random.normal(keys[0], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    k = jax.random.normal(keys[1], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    v = jax.random.normal(keys[2], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    do = jax.random.normal(keys[3], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    qi = jax.random.normal(keys[4], (1, length, INDEX_HEADS, INDEX_DIM), jnp.float32).astype(bf16)
    ki = jax.random.normal(keys[5], (1, length, INDEX_DIM), jnp.float32).astype(bf16)
    wi = jax.random.uniform(keys[6], (1, length, INDEX_HEADS), jnp.float32, 0.1, 1.0) * INDEX_HEADS ** -0.5
    return q, k, v, do, qi, ki, wi


def _measure(row, fn, args, iterations, grad_args=None, flops=None):
    try:
        fwd = harness.median_kernel_ms(jax.jit(fn), *args, iterations=iterations)
        row["fwd_ms"] = fwd["median_ms"]
        if flops:
            row["fwd_tflops"] = flops / (fwd["median_ms"] * 1e9)
        if grad_args is not None:
            both = harness.median_kernel_ms(harness.vjp_fn(fn, (0, 1, 2)), *args, grad_args,
                                            iterations=iterations)
            row["fwdbwd_ms"] = both["median_ms"]
    except Exception as err:  # record the failure (e.g. out of memory), keep measuring the rest
        row["error"] = harness.error_summary(f"{type(err).__name__}: {err}")
    print(row, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/dsa.json")
    parser.add_argument("--lengths", default="4096,8192")
    parser.add_argument("--top-k", default="512,2048")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-plain", action="store_true", help="skip the plain-JAX (XLA) rows")
    args = parser.parse_args()
    before = harness.gpu_snapshot()
    rows = []
    for length in (int(x) for x in args.lengths.split(",")):
        q, k, v, do, qi, ki, wi = _inputs(length)
        index_flops = length * (length + 1) / 2 * INDEX_HEADS * INDEX_DIM * 2.0
        base = dict(T=length, B=1, HQ=Q_HEADS, H=KV_HEADS, D=HEAD_DIM, HI=INDEX_HEADS, DI=INDEX_DIM)
        impls = [("mosaic_gpu", index_scores)] + ([] if args.skip_plain else [("xla", dsa_index_scores)])
        for impl, fn in impls:
            rows.append(_measure(dict(base, part="indexer", impl=impl), lambda a, b, c, fn=fn: (fn(a, b, c),),
                                 (qi, ki, wi), args.iterations, flops=index_flops))
        scores = jax.jit(index_scores)(qi, ki, wi)
        for top_k in (int(x) for x in args.top_k.split(",")):
            rows.append(_measure(dict(base, part="select", impl="lax.top_k", top_k=top_k),
                                 lambda s, top_k=top_k: (dsa_select(s, top_k),), (scores,), args.iterations))
            rows.append(_measure(dict(base, part="select", impl="select_top_k (causal prefixes)", top_k=top_k),
                                 lambda s, top_k=top_k: (select_top_k(s, top_k),), (scores,), args.iterations))
            idx = jax.jit(lambda s, top_k=top_k: dsa_select(s, top_k))(scores)
            pairs = sum(min(t + 1, top_k) for t in range(length))
            flops = 4.0 * Q_HEADS * pairs * HEAD_DIM
            attn = [("mosaic_gpu", lambda q, k, v, idx=idx: (dsa_attention(q, k, v, idx),))]
            if not args.skip_plain:
                attn.append(("xla", lambda q, k, v, idx=idx: (dsa(q, k, v, idx),)))
            for impl, fn in attn:
                rows.append(_measure(dict(base, part="attention", impl=impl, top_k=top_k), fn, (q, k, v),
                                     args.iterations, grad_args=do, flops=flops))
        dense_flops = 4.0 * Q_HEADS * length * (length + 1) / 2 * HEAD_DIM
        dense = [
            ("cudnn_dense_causal", lambda q, k, v: (jax.nn.dot_product_attention(
                q, k, v, is_causal=True, implementation="cudnn"),)),
            ("mosaic_gpu_dense_causal", lambda q, k, v: (flash_attention(q, k, v),)),
        ]
        for impl, fn in dense:
            rows.append(_measure(dict(base, part="attention", impl=impl, top_k=length), fn, (q, k, v),
                                 args.iterations, grad_args=do, flops=dense_flops))
    harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before,
                                      gpu_after=harness.gpu_snapshot(), rows=rows,
                                      note="one H100 of a shared node; other jobs ran on the other GPUs"))


if __name__ == "__main__":
    main()
