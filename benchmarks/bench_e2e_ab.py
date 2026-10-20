# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of the 1.3B end-to-end harness on one GPU (benchmarks/zero_copy_layouts.md).

    python benchmarks/bench_e2e_ab.py --arch mamba2-1p3b --data /path/to/fineweb_edu_3B \
        --b-set model.ssm.grouped_key_grads=true --rounds 2 --out-dir /path/to/ab --json summary.json

Runs ``benchmarks/bench_e2e_1p3b.py`` (``--impl kernels``, 16 x 4096 tokens, 20 steps, the first 5
untimed) alternately without (A) and with (B) the ``--b-set`` overrides, ``--rounds`` times each, in
fresh processes on the GPU this process sees, and records ``nvidia-smi`` for that GPU before and after.
The summary gives every run's tokens/s and the median ratio B / A. ``--a-tree`` runs variant A from another
checkout (its ``benchmarks/bench_e2e_1p3b.py`` with its ``src`` first on ``PYTHONPATH``), so A can be the code
before a change and B this tree (``--b-set`` may then be omitted).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _gpu() -> str:
    index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader",
                              "-i", index], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--b-set", action="append", default=[], help="override(s) that define variant B")
    p.add_argument("--a-set", action="append", default=[], help="override(s) of variant A (default: none)")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--a-tree", default=None, help="repository root whose code runs variant A (default: this one)")
    p.add_argument("--set", dest="common", action="append", default=[], help="override(s) of both variants")
    p.add_argument("--b-env", action="append", default=[], help="KEY=VALUE environment of variant B only "
                   "(e.g. XLA_PYTHON_CLIENT_PREALLOCATE=true for a split that needs one preallocated pool)")
    p.add_argument("--b-micro-batch", type=int, default=None, help="variant B's micro-batch (default --micro-batch)")
    p.add_argument("--harness-arg", action="append", default=[],
                   help="argument for bench_e2e_1p3b.py in both variants, e.g. --harness-arg=--attention=cudnn")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.b_set and not args.a_tree:
        p.error("give --b-set, --a-tree or both")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    gpu_before = _gpu()
    for r in range(1, args.rounds + 1):
        for name, sets in (("A", args.a_set), ("B", args.b_set)):
            path = out / f"{args.arch}-{name}{r}.json"
            tree = Path(args.a_tree).resolve() if (name == "A" and args.a_tree) else HERE.parent
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(tree / "src")] + [x for x in env.get("PYTHONPATH", "").split(
                os.pathsep) if x and not x.endswith("/src")])
            micro = args.micro_batch
            if name == "B":
                env.update(kv.split("=", 1) for kv in args.b_env)
                micro = args.b_micro_batch or micro
            cmd = [sys.executable, str(tree / "benchmarks" / "bench_e2e_1p3b.py"), "--arch", args.arch, "--data",
                   args.data, "--impl", "kernels", "--micro-batch", str(micro), "--steps", str(args.steps),
                   "--skip", str(args.skip), "--json", str(path), *args.harness_arg]
            for s in [*args.common, *sets]:
                cmd += ["--set", s]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tree), env=env)
            row = {"variant": name, "round": r, "overrides": [*args.common, *sets], "returncode": proc.returncode,
                   "tree": "a-tree" if tree != HERE.parent else "this tree"}
            if proc.returncode == 0 and path.exists():
                res = json.loads(path.read_text())
                row.update({k: res.get(k) for k in ("tokens_per_sec", "mfu_percent", "step_time_mean_s",
                                                      "step_time_median_s", "step_time_min_s",
                                                      "peak_memory_gib_max_device", "loss_last", "loss_decreasing",
                                                      "loss_mean_last_steps", "compile_seconds", "per_step")})
            else:  # the exception line only: no stack frames or paths
                lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
                row["error"] = (lines[-1] if lines else "no output")[:300]
            print("AB_RUN " + json.dumps(row), flush=True)
            runs.append(row)
    ok = {v: [r["tokens_per_sec"] for r in runs if r["variant"] == v and r.get("tokens_per_sec")] for v in "AB"}
    summary = {
        "arch": args.arch, "a_overrides": args.a_set, "b_overrides": args.b_set, "rounds": args.rounds,
        "runs": runs, "gpu_before": gpu_before, "gpu_after": _gpu(),
        "tokens_per_sec_median": {v: statistics.median(x) if x else None for v, x in ok.items()},
    }
    if ok["A"] and ok["B"]:
        summary["ratio_b_over_a"] = statistics.median(ok["B"]) / statistics.median(ok["A"])
    print("AB_SUMMARY " + json.dumps({k: summary[k] for k in ("arch", "tokens_per_sec_median")}
                                     | {"ratio": summary.get("ratio_b_over_a")}))
    Path(args.json).write_text(json.dumps(summary, indent=2))
    return 0 if ok["A"] and ok["B"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
