# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel breakdown of the DSA layer's forward with the radix and the sort top-k (diagnostic).

Same inputs as ``bench_dsa_topk.py`` (``B = 1``, indexer 64 heads x 128, ``top_k = 2048``, attention 16 query
heads on one key/value head of 128, bf16). For each method: the layer's forward (median CUPTI kernel time),
its kernels by name (median per name over a few runs) and how many launches each name has in one run, and
the same for ``index_topk`` alone, to locate the radix select's slow mode inside the layer.

    python benchmarks/dsa_layer_breakdown.py <out.json> [T]
"""

from __future__ import annotations

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

import harness  # noqa: E402
from bench_dsa_topk import TOP_K, _inputs  # noqa: E402

from jla.ops.mosaic_gpu.dsa import _index_topk, dsa_layer  # noqa: E402


def _launches(fn, args) -> dict[str, int]:
    jax.block_until_ready(fn(*args))
    (run,) = harness._measure_each(fn, args, 1, aggregate=False)
    counts: dict[str, int] = {}
    for name, _ in run:
        name = name.split("(")[0]
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> None:
    out = sys.argv[1]
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    before = harness.gpu_snapshot()
    q, k, v, _, qi, ki, wi = _inputs(length)
    rows = []
    for method in ("radix", "sort"):
        layer = jax.jit(lambda q, k, v, qi, ki, wi, m=method: dsa_layer(q, k, v, qi, ki, wi, top_k=TOP_K,
                                                                        indexer_loss_coef=1.0, topk_method=m))
        index = jax.jit(lambda qi, ki, wi, m=method: _index_topk(qi, ki, wi, TOP_K, method=m))
        row = dict(T=length, method=method)
        row["layer_fwd_ms"] = harness.median_kernel_ms(layer, q, k, v, qi, ki, wi, iterations=10)["median_ms"]
        row["layer_kernels_ms"] = harness.kernel_breakdown_ms(layer, q, k, v, qi, ki, wi, iterations=5)
        row["layer_launches"] = _launches(layer, (q, k, v, qi, ki, wi))
        row["index_topk_ms"] = harness.median_kernel_ms(index, qi, ki, wi, iterations=10)["median_ms"]
        row["index_topk_kernels_ms"] = harness.kernel_breakdown_ms(index, qi, ki, wi, iterations=5)
        row["index_topk_launches"] = _launches(index, (qi, ki, wi))
        print({key: row[key] for key in ("method", "layer_fwd_ms", "index_topk_ms")}, flush=True)
        rows.append(row)
        del layer, index
        gc.collect()
    harness.write_json(out, dict(environment=harness.environment(), gpu_before=before,
                                 gpu_after=harness.gpu_snapshot(), rows=rows))


if __name__ == "__main__":
    main()
