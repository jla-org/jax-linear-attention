# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 grouped backward kernel (``sg_dqkwg_grouped``) at the 1.3B layer shape: its variants, interleaved.

    python benchmarks/bench_mamba2_dqkwg.py --json results/mamba2_dqkwg.json

One call of :func:`jla.ops.mosaic_gpu.scalar_gate_grouped.dqkwg_grouped` on the per-layer shape of
``mamba2-1p3b`` (16 x 4096 tokens, one B/C group of N = 128 shared by 64 value heads of P = 64, packed documents),
for every ``(splits, register_sums)`` variant, interleaved round by round on one GPU (CUPTI kernel time, median of
the rounds' medians, ``benchmarks/harness.py``). Every variant's outputs are checked against the default
(``splits = 1``, shared-memory sums) first.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import harness  # noqa: E402
from jla.ops.mosaic_gpu import _common as cm  # noqa: E402
from jla.ops.mosaic_gpu import scalar_gate_grouped as sgg  # noqa: E402

BF16 = jnp.bfloat16


def _inputs(batch, seq_len, heads, key_dim, value_dim, docs):
    ks = jax.random.split(jax.random.key(0), 8)
    c = cm.CHUNK
    q = jax.random.normal(ks[0], (batch, seq_len, 1, key_dim)).astype(BF16)
    k = (0.5 * jax.random.normal(ks[1], (batch, seq_len, 1, key_dim))).astype(BF16)
    v = jax.random.normal(ks[2], (batch, seq_len, heads, value_dim)).astype(BF16)
    do = jax.random.normal(ks[3], (batch, seq_len, heads, value_dim)).astype(BF16)
    g2 = cm.chunk_log2_decay(-jax.nn.softplus(jax.random.normal(ks[4], (batch, seq_len, heads)) - 2.0),
                             batch, seq_len, heads, 0)
    h = (0.3 * jax.random.normal(ks[5], (batch, heads, seq_len // c, value_dim, key_dim))).astype(BF16)
    dh = (0.3 * jax.random.normal(ks[6], (batch, heads, seq_len // c, value_dim, key_dim))).astype(BF16)
    seg = None
    if docs:
        rng = np.random.default_rng(0)
        ids = np.zeros((batch, seq_len), np.int32)
        for b in range(batch):
            for cut in sorted(rng.choice(np.arange(1, seq_len), size=docs, replace=False)):
                ids[b, cut:] += 1
        seg = cm.segment_vectors(jnp.asarray(ids), 0)
    return (q, k, v, g2, h, dh, do), seg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--key-dim", type=int, default=128)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--docs", type=int, default=3, help="document boundaries per row (0: no segment_ids)")
    p.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    operands, seg = _inputs(args.batch, args.seq_len, args.heads, args.key_dim, args.value_dim, args.docs)
    variants = [(s, r) for s in args.splits for r in (False, True)]

    def fn(splits, regs):
        def run(*a):
            dq, dk, dg = sgg.dqkwg_grouped(*a, scale=1.0, splits=splits, seg=seg, register_sums=regs)
            if splits > 1:  # the split partial sums, summed as scalar_gate does
                dq, dk = (x.reshape(*x.shape[:2], 1, splits, -1).sum(axis=3) for x in (dq, dk))
            return dq, dk, dg

        return jax.jit(run)

    fns = {v: fn(*v) for v in variants}
    want = jax.block_until_ready(fns[variants[0]](*operands))
    errors = {}
    for v, f in fns.items():
        got = jax.block_until_ready(f(*operands))
        errors[f"splits{v[0]}-regs{int(v[1])}"] = [
            float(jnp.max(jnp.abs(g - w)) / jnp.maximum(jnp.max(jnp.abs(w)), 1e-6)) for g, w in zip(got, want)]
    before = harness.gpu_snapshot()
    times = {v: [] for v in variants}
    for _ in range(args.rounds):
        for v, f in fns.items():
            times[v].append(harness.median_kernel_ms(f, *operands, iterations=args.iterations)["median_ms"])
    after = harness.gpu_snapshot()
    rows = []
    base = statistics.median(times[variants[0]])
    for v in variants:
        ms = statistics.median(times[v])
        name = f"splits{v[0]}-regs{int(v[1])}"
        rows.append(dict(variant=name, splits=v[0], register_sums=v[1], median_ms=ms, rounds_ms=times[v],
                         speedup_vs_default=base / ms, max_rel_err_vs_default=errors[name]))
        print(f"{name:18s} {ms:8.3f} ms  x{base / ms:5.3f}  err {max(errors[name]):.2e}", flush=True)
    if args.json:
        harness.write_json(args.json, dict(
            benchmark="mamba2_dqkwg_grouped", shape=dict(batch=args.batch, seq_len=args.seq_len, heads=args.heads,
                                                         key_dim=args.key_dim, value_dim=args.value_dim,
                                                         docs=args.docs),
            rows=rows, gpu_before=before, gpu_after=after, env=harness.environment()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
