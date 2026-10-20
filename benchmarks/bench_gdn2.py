# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Gated DeltaNet-2 kernels against the chunked XLA op and our KDA kernels.

    python benchmarks/bench_gdn2.py [--shapes ...] [--rounds 2]

Same driver and metric as ``bench_kda.py --kind gdn2``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_kda  # noqa: E402

if __name__ == "__main__":
    bench_kda.main(default_kind="gdn2")
