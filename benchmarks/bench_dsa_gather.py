# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA's selected-key attention: gathering in plain JAX (``gather_attention``) against the masked-dense kernels.

Shapes as ``bench_dsa_topk.py``: ``B = 1``, 16 query heads sharing one key/value head of 128, ``top_k = 2048``,
bf16. Each query ``t`` selects ``min(t + 1, top_k)`` distinct keys spread evenly over its causal prefix
(``floor(j (t + 1) / top_k)``), so both paths do the same selected-key work; the masked-dense kernels'
time does not depend on which keys are selected, the gathers' only through locality. Rows per length, the
two paths interleaved in one run (median CUPTI kernel time, ``harness.py``), forward and forward + backward
(gradients of q, k, v), written to the JSON after every row. ``*_useful_tflops``: the selected-key products,
``4 T top_k HQ D`` for the forward and ``3.5x`` that for forward + backward (the backward's five products
against the forward's two), per time.

    python benchmarks/bench_dsa_gather.py --out benchmarks/results/dsa_gather.json [--lengths 16384,32768]
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

from jla.ops.dsa_gather import gather_attention  # noqa: E402
from jla.ops.mosaic_gpu.dsa import dsa_attention  # noqa: E402

Q_HEADS, KV_HEADS, HEAD_DIM, TOP_K = 16, 1, 128, 2048


def _inputs(length: int):
    keys = jax.random.split(jax.random.key(length), 4)
    bf16 = jnp.bfloat16
    q = jax.random.normal(keys[0], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    k = jax.random.normal(keys[1], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    v = jax.random.normal(keys[2], (1, length, KV_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    do = jax.random.normal(keys[3], (1, length, Q_HEADS, HEAD_DIM), jnp.float32).astype(bf16)
    t = jnp.arange(length, dtype=jnp.int32)[:, None]
    j = jnp.arange(TOP_K, dtype=jnp.int32)[None, :]
    spread = (j * (t + 1)) // TOP_K  # distinct when t + 1 >= TOP_K
    idx = jnp.where(t + 1 >= TOP_K, spread, jnp.where(j <= t, j, -1))[None]
    return q, k, v, do, idx


def _time(row, key, fn, args, iterations):
    try:
        row[key] = harness.median_kernel_ms(jax.jit(fn), *args, iterations=iterations)["median_ms"]
    except Exception as err:  # record the failure (e.g. out of memory), keep measuring the rest
        row[key + "_error"] = harness.error_summary(f"{type(err).__name__}: {err}")
    gc.collect()


def _grad_fn(core):
    def run(q, k, v, idx, do):
        loss = lambda q, k, v: jnp.sum(core(q, k, v, idx).astype(jnp.float32) * do.astype(jnp.float32))  # noqa: E731
        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/dsa_gather.json")
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    parser.add_argument("--impls", default="gather,masked_dense")
    args = parser.parse_args()
    impls = {"gather": gather_attention, "masked_dense": dsa_attention}
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
        q, k, v, do, idx = _inputs(length)
        work = 4.0 * length * TOP_K * Q_HEADS * HEAD_DIM  # selected-key flops of the forward (upper bound)
        for name in args.impls.split(","):
            core = impls[name]
            row = dict(T=length, B=1, HQ=Q_HEADS, H=KV_HEADS, D=HEAD_DIM, top_k=TOP_K, iterations=iterations,
                       impl=name)
            _time(row, "fwd_ms", lambda q, k, v, idx, core=core: (core(q, k, v, idx),), (q, k, v, idx), iterations)
            _time(row, "fwdbwd_ms", _grad_fn(core), (q, k, v, idx, do), iterations)
            if "fwd_ms" in row:
                row["fwd_useful_tflops"] = work / row["fwd_ms"] / 1e9
            if "fwdbwd_ms" in row:
                row["fwdbwd_useful_tflops"] = 3.5 * work / row["fwdbwd_ms"] / 1e9
            emit(row)
        del q, k, v, do, idx
        gc.collect()
    harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=before,
                                      gpu_after=harness.gpu_snapshot(), rows=rows,
                                      note="one H100, the only timed job on it; test runs of other jobs "
                                           "may share it (provisional)"))


if __name__ == "__main__":
    main()
