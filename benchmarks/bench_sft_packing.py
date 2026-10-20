# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SFT packing: best-fit-decreasing speed and row fill, and SFT row-building throughput.

Host-side (NumPy) work only; the JAX backend does not matter. Example lengths are
lognormal (chat-like: many short examples, a long tail), clipped to ``[2, seq_len + 1]``.

* ``pack_best_fit_decreasing`` on ``--examples`` lengths for each ``--seq-len``: rows,
  fill (share of row positions holding example tokens), the lower bound
  ``ceil(sum / seq_len)`` on the row count, and seconds; ``single`` (one example per
  row) for comparison.
* :class:`jla.data.sft.SftDataset` on a written corpus of ``--corpus-examples``
  examples: corpus write time, packing-plan time, and rows/s and tokens/s of
  ``__getitem__`` (what each loader worker does per row).

    python benchmarks/bench_sft_packing.py --out benchmarks/results/sft_packing.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from jla.data.sft import SftDataset, pack_best_fit_decreasing, write_sft


def _lengths(rng: np.random.Generator, count: int, median: float, sigma: float, cap: int) -> np.ndarray:
    return np.clip(rng.lognormal(np.log(median), sigma, size=count).astype(np.int64), 2, cap)


def bench_bfd(examples: int, seq_len: int, median: float, sigma: float, seed: int) -> dict:
    # Packed items are next-token positions: an example of n tokens occupies n - 1 positions.
    positions = _lengths(np.random.default_rng(seed), examples, median, sigma, seq_len + 1) - 1
    start = time.perf_counter()
    bins = pack_best_fit_decreasing(positions, seq_len)
    seconds = time.perf_counter() - start
    total = int(positions.sum())
    assert sorted(i for b in bins for i in b) == list(range(examples))
    assert max(int(positions[b].sum()) for b in bins) <= seq_len
    return {
        "examples": examples,
        "seq_len": seq_len,
        "length_median": median,
        "length_sigma": sigma,
        "positions": total,
        "bfd_rows": len(bins),
        "bfd_fill": round(total / (len(bins) * seq_len), 5),
        "lower_bound_rows": -(-total // seq_len),
        "single_rows": examples,
        "single_fill": round(total / (examples * seq_len), 5),
        "bfd_seconds": round(seconds, 3),
        "bfd_examples_per_s": round(examples / seconds),
    }


def bench_dataset(root: Path, examples: int, seq_len: int, median: float, sigma: float, rows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lengths = _lengths(rng, examples, median, sigma, 2 * seq_len)  # some examples are truncated

    def corpus():
        for n in lengths.tolist():
            tokens = rng.integers(0, 32_000, size=n, dtype=np.int64)
            mask = np.zeros(n, dtype=np.uint8)
            mask[int(rng.integers(1, n)) :] = 1
            yield tokens, mask

    start = time.perf_counter()
    write_sft(root / "corpus", corpus())
    write_seconds = time.perf_counter() - start
    out = {"corpus_examples": examples, "corpus_tokens": int(lengths.sum()), "seq_len": seq_len,
           "write_seconds": round(write_seconds, 3)}
    for packing in ("bfd", "single"):
        start = time.perf_counter()
        dataset = SftDataset(str(root / "corpus"), seq_len, packing=packing)
        plan_seconds = time.perf_counter() - start
        count = min(rows, len(dataset))
        order = np.random.default_rng(seed + 1).permutation(len(dataset))[:count]
        targets = 0
        start = time.perf_counter()
        for index in order.tolist():
            targets += int(dataset[index]["loss_mask"].sum())
        seconds = time.perf_counter() - start
        out[packing] = {
            "rows": len(dataset),
            "fill": round(dataset.fill_fraction(), 5),
            "truncated_examples": dataset.truncated,
            "plan_seconds": round(plan_seconds, 3),
            "rows_timed": count,
            "rows_per_s": round(count / seconds, 1),
            "row_tokens_per_s": round(count * seq_len / seconds),
            "targets_in_timed_rows": targets,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=int, default=1_000_000)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--median", type=float, default=600.0, help="median example length in tokens")
    parser.add_argument("--sigma", type=float, default=0.9, help="lognormal sigma of example lengths")
    parser.add_argument("--corpus-examples", type=int, default=50_000)
    parser.add_argument("--corpus-seq-len", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=2_000, help="rows built per packing mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tmp", default=None, help="directory for the temporary corpus")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = {
        "benchmark": "sft_packing",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "cpus": os.cpu_count()},
        "bfd": [bench_bfd(args.examples, s, args.median, args.sigma, args.seed) for s in args.seq_len],
    }
    for row in result["bfd"]:
        print(json.dumps(row), flush=True)
    with tempfile.TemporaryDirectory(dir=args.tmp, prefix="sft-packing-") as tmp:
        result["dataset"] = bench_dataset(
            Path(tmp), args.corpus_examples, args.corpus_seq_len, args.median, args.sigma, args.rows, args.seed
        )
    print(json.dumps(result["dataset"]), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
