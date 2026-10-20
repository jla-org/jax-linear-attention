# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock against CUPTI kernel time of the DSA layer's forward, radix and sort top-k (diagnostic).

Same inputs as ``bench_dsa_topk.py``. Per method, interleaved twice: the median and minimum wall-clock time of
``jax.block_until_ready(layer(...))`` over 20 calls, and the median CUPTI kernel time (the benchmarks' number,
which sums the kernels' durations and so would count overlapping kernels twice).

    python benchmarks/dsa_layer_wallclock.py <out.json> [T] [layer,index_topk]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

import harness  # noqa: E402
from bench_dsa_topk import TOP_K, _inputs  # noqa: E402

from jla.ops.mosaic_gpu.dsa import _index_topk, dsa_layer  # noqa: E402


def main() -> None:
    out = sys.argv[1]
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    before = harness.gpu_snapshot()
    q, k, v, _, qi, ki, wi = _inputs(length)
    args = (q, k, v, qi, ki, wi)
    parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["layer"]
    fns = {}
    for m in ("radix", "sort"):
        if "layer" in parts:
            fns["layer", m] = (jax.jit(lambda q, k, v, qi, ki, wi, m=m: dsa_layer(
                q, k, v, qi, ki, wi, top_k=TOP_K, indexer_loss_coef=1.0, topk_method=m)), args)
        if "index_topk" in parts:
            fns["index_topk", m] = (jax.jit(lambda qi, ki, wi, m=m: _index_topk(qi, ki, wi, TOP_K, method=m)),
                                    (qi, ki, wi))
    rows = []
    for part, method in [key for _ in range(2) for key in fns]:
        fn, args = fns[part, method]
        for _ in range(3):
            jax.block_until_ready(fn(*args))
        times = []
        for _ in range(20):
            start = time.perf_counter()
            jax.block_until_ready(fn(*args))
            times.append((time.perf_counter() - start) * 1e3)
        row = dict(T=length, part=part, method=method, wall_median_ms=statistics.median(times), wall_min_ms=min(times),
                   cupti_median_ms=harness.median_kernel_ms(fn, *args, iterations=10)["median_ms"])
        print(row, flush=True)
        rows.append(row)
    harness.write_json(out, dict(environment=harness.environment(), gpu_before=before,
                                 gpu_after=harness.gpu_snapshot(), rows=rows))


if __name__ == "__main__":
    main()
