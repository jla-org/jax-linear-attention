# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The attention backward with the FP8 q/k/v gradient epilogues against the plain backward plus XLA's passes.

    python benchmarks/bench_attention_fp8_bwd.py --json results/attention_fp8_bwd.json \
        [--batch 4 --length 4096 --heads 32 --kv-heads 8 --head-dim 64 --masks causal,documents,swa2048]

Every variant maps the same post-RoPE ``q, k, v``, forward results and output cotangent to what the q/k/v
projections' backward GEMMs read in a delayed-scaling FP8 step: the e5m2 cotangents of the three projection
outputs and their amaxes (benchmarks/fp8_1p3b.md, "The recipe and the fused paths"):

* ``plain-ranged`` / ``plain-fused``: ``jla.ops.mosaic_gpu.attention``'s ranged or fused backward kernels,
  then XLA's RoPE backward, bf16 rounding, e5m2 cast and amax (one jitted program, as in the train step);
* ``fp8-ranged`` / ``fp8-fused``: :func:`jla.ops.mosaic_gpu.attention_fp8_bwd.fp8_attention_backward`
  (the epilogues in the kernels; the fused variant adds one pass over its float32 ``dq`` sums).

The metric is CUPTI kernel time per call (median of ``--iterations`` calls after a warmup), the variants
interleaved over ``--rounds`` rounds on one GPU; ``documents`` packs 4 equal documents per row with
positions restarting in each, ``swa<N>`` is a sliding window of ``N`` tokens. Not timed: the weight-gradient
GEMMs' transposed operand, which XLA makes for either path.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness  # noqa: E402

from jla.layers.fp8 import E5M2, _amax, quantize  # noqa: E402
from jla.layers.fp8_attention import rope_tables  # noqa: E402
from jla.layers.rope import apply_rope  # noqa: E402
from jla.ops.mosaic_gpu import attention as A  # noqa: E402
from jla.ops.mosaic_gpu.attention_fp8_bwd import fp8_attention_backward  # noqa: E402

VARIANTS = ("plain-ranged", "plain-fused", "fp8-ranged", "fp8-fused")


def _case(batch, length, heads, kv_heads, head_dim, mask, seed=0):
    rng = np.random.default_rng(seed)
    q, do = (jnp.asarray(rng.standard_normal((batch, length, heads, head_dim)), jnp.bfloat16) for _ in range(2))
    k, v = (jnp.asarray(rng.standard_normal((batch, length, kv_heads, head_dim)), jnp.bfloat16) for _ in range(2))
    window = int(mask[3:]) if mask.startswith("swa") else None
    if mask == "documents":
        doc = length // 4
        seg = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32) // doc, (batch, length))
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32) % doc, (batch, length))
        contiguous = A._contiguous_ids(seg)
        runs = A._segment_runs(seg, contiguous)
        runs = jnp.concatenate([runs, A._inner_runs(runs, contiguous, True)], axis=1)
    else:
        seg = jnp.zeros((batch, length), jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
        runs = jnp.zeros((batch, 4, length), jnp.int32)
    cfg = A._Config(scale=head_dim ** -0.5, window=window if window and window < length else None,
                    has_segments=mask == "documents")
    sink = A._sink_init(jnp.zeros((heads,), jnp.float32), cfg.block_q)
    out, lse = jax.jit(lambda q, k, v: A._forward(q, k, v, seg, runs, sink, cfg))(q, k, v)
    cos, sin = rope_tables(positions, theta=10_000.0, head_dim=head_dim)
    gscale = jnp.asarray([2e-6, 2e-6, 4e-6], jnp.float32)
    return cfg, (q, k, v, seg, runs, out, lse, do, cos, sin, gscale), positions


def _plain(cfg, backward, positions):
    def run(q, k, v, seg, runs, out, lse, do, cos, sin, gscale):
        if backward == "fused":
            dq, dk, dv = A._backward_fused(cfg, q, k, v, seg, runs, out, lse, do)
        else:
            dq, dk, dv = A._backward_ranged(cfg, q, k, v, seg, runs, out, lse, do)

        def rope_vjp(x, ct):
            return jax.vjp(lambda t: apply_rope(t, positions, theta=10_000.0), x)[1](ct)[0]

        grads = (rope_vjp(q, dq), rope_vjp(k, dk), dv)
        return tuple(quantize(g, gscale[i], E5M2) for i, g in enumerate(grads)), tuple(_amax(g) for g in grads)

    return jax.jit(run)


def _fp8(cfg, backward):
    fp8_cfg = dataclasses.replace(cfg, backward=backward)

    def run(*args):
        grads = fp8_attention_backward(*args, fp8_cfg)
        return tuple(g8 for g8, _, _ in grads), tuple(jnp.max(am) for _, _, am in grads)

    return jax.jit(run)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--masks", default="causal,documents,swa2048")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--breakdown", action="store_true", help="also record the median time of every kernel")
    p.add_argument("--json", required=True)
    args = p.parse_args(argv)
    variants = args.variants.split(",")
    rows = []
    before = harness.gpu_snapshot()
    for mask in args.masks.split(","):
        cfg, operands, positions = _case(args.batch, args.length, args.heads, args.kv_heads, args.head_dim, mask)
        fns = {}
        for name in variants:
            kind, backward = name.split("-")
            fns[name] = _fp8(cfg, backward) if kind == "fp8" else _plain(cfg, backward, positions)
        times: dict[str, list[float]] = {name: [] for name in variants}
        for _ in range(args.rounds):
            for name in variants:
                timed = harness.median_kernel_ms(fns[name], *operands, iterations=args.iterations)
                times[name].append(timed["median_ms"])
        row = dict(mask=mask, **{name: min(ts) for name, ts in times.items()}, rounds=dict(times))
        if args.breakdown:  # median ms per kernel name
            row["kernels"] = {name: harness.kernel_breakdown_ms(fns[name], *operands) for name in variants}
            for name, kernels in row["kernels"].items():
                print(f"  {mask} {name}: " + ", ".join(f"{k} {t:.3f}" for k, t in kernels.items()), flush=True)
        rows.append(row)
        print(mask, " ".join(f"{name}={min(ts):.3f}" for name, ts in times.items()), flush=True)
    payload = dict(environment=harness.environment(), shape=dict(batch=args.batch, length=args.length,
                   heads=args.heads, kv_heads=args.kv_heads, head_dim=args.head_dim), iterations=args.iterations,
                   rounds=args.rounds, metric="CUPTI kernel ms per call, fastest round's median", rows=rows,
                   gpu_before=before, gpu_after=harness.gpu_snapshot())
    harness.write_json(args.json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
