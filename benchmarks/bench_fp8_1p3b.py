# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 against bf16 end-to-end training at 1.3B on one GPU (benchmarks/fp8_1p3b.md).

    python benchmarks/bench_fp8_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B \
        --variants bf16,delayed,current,bf16 --json results/fp8_1p3b_transformer-1p3b.json

Each variant is one ``benchmarks/bench_e2e_1p3b.py`` run (``jla train`` on real data, Mosaic GPU kernels,
16 x 4096 tokens per step, 20 steps of which the first 5 are warm-up) in a fresh process, one after
another on the same GPU, with identical settings except the variant's ``--set`` overrides:

* ``bf16``: the preset as it is (FP8 off, the default); ``bf16-seed1`` the same with ``train.seed = 1``
  (another initialization), whose loss difference to ``bf16`` is the yardstick for "within bf16 noise";
* ``delayed`` / ``current``: ``model.fp8`` (``jla.layers.fp8``, per-tensor delayed or current scaling)
  on every 16-aligned projection;
* ``delayed-min512`` / ``current-min512``: the same with ``model.fp8_min_features = 512``, which keeps
  the small GEMMs (gates, low-rank factors, Mamba's dt / B-C projections) in bf16;
* ``delayed-fused`` (``-min512``): delayed scaling with ``model.fp8_fused = true``, the SwiGLU MLPs'
  fused activation + quantize + transpose passes (``jla.layers.fp8_swiglu``).

The result lists every run (tokens/s, step times, peak memory, every step's loss) and, against the
first ``bf16`` run, the tokens/s ratio and the largest per-step loss difference. A second ``bf16`` run
gives the run-to-run spread of both.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS: dict[str, list[str]] = {
    "bf16": [],
    "bf16-seed1": ["train.seed=1"],  # another initialization: the run-to-run loss spread of bf16 itself
    "delayed": ["model.fp8=delayed"],
    "current": ["model.fp8=current"],
    "delayed-min512": ["model.fp8=delayed", "model.fp8_min_features=512"],
    "current-min512": ["model.fp8=current", "model.fp8_min_features=512"],
    "delayed-fused": ["model.fp8=delayed", "model.fp8_fused=true"],
    "delayed-fused-min512": ["model.fp8=delayed", "model.fp8_fused=true", "model.fp8_min_features=512"],
}


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--variants", default="bf16,delayed,current,bf16",
                   help=f"comma-separated, from {sorted(VARIANTS)}; repeats allowed")
    p.add_argument("--impl", default="kernels")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1500, help="seconds per run")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="override for every variant")
    p.add_argument("--json", required=True)
    args = p.parse_args(argv)
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown or not names:
        p.error(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    args.variant_names = names
    return args


def _run(args, variant: str, index: int, workdir: Path) -> dict:
    out_json = workdir / f"{index:02d}-{variant}.json"
    cmd = [
        sys.executable, str(HERE / "bench_e2e_1p3b.py"), "--arch", args.arch, "--data", args.data,
        "--impl", args.impl, "--micro-batch", str(args.micro_batch), "--steps", str(args.steps),
        "--skip", str(args.skip), "--dp", "1", "--fsdp", "1", "--json", str(out_json),
    ]
    for override in [*args.overrides, *VARIANTS[variant]]:
        cmd += ["--set", override]
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        rc, tail = proc.returncode, (proc.stdout[-1500:] + proc.stderr[-2500:])
    except subprocess.TimeoutExpired as exc:
        rc, tail = -9, f"timeout after {args.timeout}s: {str(exc.stdout or '')[-500:]} {str(exc.stderr or '')[-1500:]}"
    record = {"variant": variant, "overrides": VARIANTS[variant], "rc": rc, "wall_s": round(time.time() - started, 1)}
    if out_json.exists():
        record["result"] = json.loads(out_json.read_text())
    if rc != 0:
        record["log_tail"] = tail
    print(f"[bench_fp8_1p3b] {args.arch} {variant}: rc={rc} "
          f"tok/s={record.get('result', {}).get('tokens_per_sec', float('nan')):.0f}", flush=True)
    return record


def _compare(runs: list[dict]) -> list[dict]:
    ok = [r for r in runs if r.get("rc") == 0 and r.get("result", {}).get("status") == "ok"]
    base = next((r for r in ok if r["variant"] == "bf16"), None)
    rows = []
    for r in ok:
        res = r["result"]
        row = {
            "variant": r["variant"],
            "tokens_per_sec": res["tokens_per_sec"],
            "step_time_mean_s": res["step_time_mean_s"],
            "step_time_min_s": res["step_time_min_s"],
            "step_time_max_s": res["step_time_max_s"],
            "peak_memory_gib": res["peak_memory_gib_max_device"],
            "compile_seconds": res.get("compile_seconds"),
            "loss_first": res["loss_first"],
            "loss_last": res["loss_last"],
            "loss_mean_last_steps": res["loss_mean_last_steps"],
        }
        if base is not None:
            b = base["result"]
            row["speedup_vs_bf16"] = res["tokens_per_sec"] / b["tokens_per_sec"]
            losses = [s["loss"] for s in res["per_step"]]
            ref = [s["loss"] for s in b["per_step"]]
            diffs = [abs(x - y) for x, y in zip(losses, ref)]
            row["loss_max_abs_diff_vs_bf16"] = max(diffs)
            row["loss_last5_mean_diff_vs_bf16"] = res["loss_mean_last_steps"] - b["loss_mean_last_steps"]
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    workdir = Path(tempfile.mkdtemp(prefix=f"fp8-1p3b-{args.arch}-"))
    runs = []
    for i, variant in enumerate(args.variant_names):
        runs.append(_run(args, variant, i, workdir))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"arch": args.arch, "variants": args.variant_names,
                                               "summary": _compare(runs), "runs": runs}, indent=2))
    summary = _compare(runs)
    for row in summary:
        print("FP8_1P3B " + json.dumps({"arch": args.arch, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                              for k, v in row.items()}}), flush=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    Path(args.json).write_text(json.dumps({"arch": args.arch, "variants": args.variant_names,
                                           "cuda_visible_devices": visible, "summary": summary, "runs": runs},
                                          indent=2))
    return 0 if all(r.get("rc") == 0 for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
