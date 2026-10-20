# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-level timing of the Mamba mixer fast paths against the paths they replace.

    python benchmarks/bench_mixer_fast_paths.py --json results.json [--iterations 30]

Times forward + backward (CUPTI kernel time, ``benchmarks/harness.py``; gradients of
the input and every parameter) of one ``Mamba2`` and one ``Mamba3`` mixer of the
``mamba2-130m`` / ``mamba3-130m`` presets (d_model 768, 24 heads of 64, d_state 128,
one B/C group) on bfloat16 ``[8, 2048, 768]``, for these variants of the
``jla.layers.ssm`` flags, interleaved in the same run:

* ``old``: the previous paths (``GROUPED_BC = False``, ``SPLIT_ZX = False``);
* ``new``: both fast paths (the defaults);
* ``grouped_only`` / ``split_only``: one of them.

``nvidia-smi`` utilisation of the visible GPU is recorded before and after.
"""

from __future__ import annotations

import argparse
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from jla.layers import ssm as ssm_mod  # noqa: E402
from jla.models.config import SSMConfig  # noqa: E402

VARIANTS = {
    "old": dict(grouped=False, split=False),
    "new": dict(grouped=True, split=True),
    "grouped_only": dict(grouped=True, split=False),
    "split_only": dict(grouped=False, split=True),
}


def set_variant(variant: str) -> None:
    flags = VARIANTS[variant]
    ssm_mod.GROUPED_BC = flags["grouped"]
    ssm_mod.SPLIT_ZX = flags["split"]


def _mixer_case(kind: str, batch: int, seq_len: int):
    cls = {"mamba2": ssm_mod.Mamba2, "mamba3": ssm_mod.Mamba3}[kind]
    cfg = SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1)
    layer = cls(768, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5, compute_dtype="bfloat16")
    u = jax.random.normal(jax.random.key(1), (batch, seq_len, 768), jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(2), (batch, seq_len, 768), jnp.bfloat16)
    params, static = eqx.partition(layer, eqx.is_inexact_array)

    def build():
        def f(params, u, ct):
            def run(params, u):
                return eqx.combine(params, static)(u)

            y, pullback = jax.vjp(run, params, u)
            return y, pullback(ct)

        return jax.jit(f)

    return build, (params, u, ct)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", default=None)
    p.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--cases", default="mamba2,mamba3")
    p.add_argument("--variants", default="old,new,grouped_only,split_only")
    p.add_argument("--breakdown", action="store_true", help="also record the per-kernel times")
    args = p.parse_args(argv)
    variants = args.variants.split(",")
    results = {"environment": harness.environment(), "gpu_before": harness.gpu_snapshot(), "rows": []}
    for name in args.cases.split(","):
        build, inputs = _mixer_case(name, args.batch, args.seq_len)
        row = {"case": name, "batch": args.batch, "seq_len": args.seq_len}
        for variant in variants:
            if name == "mamba3" and variant not in ("old", "new"):
                continue  # grouped B/C is Mamba-2 only, so Mamba-3 has two variants
            set_variant(variant)
            fn = build()
            row[variant] = harness.median_kernel_ms(fn, *inputs, iterations=args.iterations)
            if args.breakdown:
                row[variant]["kernels"] = {k: round(v, 4) for k, v in
                                           list(harness.kernel_breakdown_ms(fn, *inputs).items())[:12]}
        set_variant("new")
        if "old" in row and "new" in row:
            row["speedup_new_vs_old"] = round(row["old"]["median_ms"] / row["new"]["median_ms"], 3)
        print({k: (round(v["median_ms"], 4) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        results["rows"].append(row)
    results["gpu_after"] = harness.gpu_snapshot()
    if args.json:
        harness.write_json(args.json, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
