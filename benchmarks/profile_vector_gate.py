# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel time of the vector-gate calls (KDA, Gated DeltaNet-2), one JSON line per shape and mode.

    python benchmarks/profile_vector_gate.py --kind kda --shapes "4,4096,16,128,128" [--docs 16] [--modes fwd,fwdbwd]

Each line: kind, shape, documents, mode, the median total kernel time and the median time of every
kernel name (CUPTI, ``harness.py``), on the inputs of ``bench_kda.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_kda  # noqa: E402
import harness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("kda", "gdn2"), default="kda")
    parser.add_argument("--shapes", default="4,4096,16,128,128", help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--docs", type=int, default=0, help="packed documents (B = 1)")
    parser.add_argument("--modes", default="fwdbwd")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    for spec in args.shapes.split(";"):
        shape = tuple(int(x) for x in spec.split(","))
        cu = bench_kda.doc_cu_seqlens(shape[1], args.docs) if args.docs else None
        fwd, fwdbwd, fargs, do = bench_kda.jax_impls(args.kind, shape, ["mosaic_gpu"], cu)["mosaic_gpu"]
        for mode in args.modes.split(","):
            fn, call = (fwd, fargs) if mode == "fwd" else (fwdbwd, (*fargs, do))
            total = harness.median_kernel_ms(fn, *call, iterations=args.iterations)
            per = harness.kernel_breakdown_ms(fn, *call, iterations=args.iterations)
            print(json.dumps(dict(kind=args.kind, shape=shape, docs=args.docs or None, mode=mode,
                                  total_ms=round(total["median_ms"], 3),
                                  kernels_ms={k: round(v, 3) for k, v in per.items()})), flush=True)


if __name__ == "__main__":
    main()
