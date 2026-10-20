# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Interleaved before/after timing of the end-to-end step with the mixer fast paths on and off.

    python benchmarks/bench_end_to_end_ab.py benchmarks/end_to_end/fw-mamba2-130m.toml \
        --variants old,new --repeats 2 --steps 15 --json result.json

Runs :mod:`bench_end_to_end_step` (the ``jla train`` step on synthetic tokens of the
config's shape) once per variant and repeat, in the order ``old, new, old, new, ...``
in one process, so both paths share the GPU's time window. Variants set the module
flags of the fast paths before the step is traced:

* ``old``: the previous paths (``ssm.GROUPED_BC = False``, ``ssm.SPLIT_ZX = False``);
* ``new``: every fast path on (the defaults);
* ``grouped_only`` / ``split_only``: one of them on.

``nvidia-smi`` utilisation of the visible GPU is recorded before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_end_to_end_step as e2e  # noqa: E402
import harness  # noqa: E402

VARIANTS = {
    "old": dict(grouped=False, split=False),
    "new": dict(grouped=True, split=True),
    "grouped_only": dict(grouped=True, split=False),
    "split_only": dict(grouped=False, split=True),
}


def _set(variant: str) -> None:
    from jla.layers import ssm

    flags = VARIANTS[variant]
    ssm.GROUPED_BC = flags["grouped"]
    ssm.SPLIT_ZX = flags["split"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config")
    p.add_argument("--variants", default="old,new")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; expected one of {sorted(VARIANTS)}")
    out = {"config": args.config, "steps": args.steps, "gpu_before": harness.gpu_snapshot(), "runs": []}
    with tempfile.TemporaryDirectory() as tmp:
        for rep in range(args.repeats):
            for variant in variants:
                _set(variant)
                path = os.path.join(tmp, f"{variant}-{rep}.json")
                extra = [x for o in args.overrides for x in ("--set", o)]
                e2e.main([args.config, "--steps", str(args.steps), "--json", path, *extra])
                with open(path) as f:
                    run = json.load(f)
                run["variant"], run["repeat"] = variant, rep
                out["runs"].append(run)
    _set("new")
    out["gpu_after"] = harness.gpu_snapshot()
    summary = {}
    for v in variants:
        runs = [r for r in out["runs"] if r["variant"] == v]
        summary[v] = dict(
            ms_per_step_median=statistics.median(r["ms_per_step_median"] for r in runs),
            ms_per_step_min=min(r["ms_per_step_min"] for r in runs),
            tok_per_s_median=statistics.median(r["tok_per_s"] for r in runs),
            loss=[r["loss"] for r in runs],
            compile_s=[r["compile_s"] for r in runs],
            device_bytes_estimate=runs[0]["device_bytes_estimate"],
        )
    if "old" in summary and "new" in summary:
        summary["speedup_new_vs_old_median"] = round(
            summary["old"]["ms_per_step_median"] / summary["new"]["ms_per_step_median"], 4)
    out["summary"] = summary
    print(json.dumps(summary, indent=1))
    if args.json:
        harness.write_json(args.json, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
