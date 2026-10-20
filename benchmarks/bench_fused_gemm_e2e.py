# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tokens/s at 1.3B on one GPU, XLA SwiGLU vs the fused SwiGLU GEMMs (benchmarks/fused_gemm.md).

    python benchmarks/bench_fused_gemm_e2e.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B \
        --variants xla,fused,xla,fused --json results/fused_gemm_e2e_transformer-1p3b.json

Each variant is one ``benchmarks/bench_e2e_1p3b.py`` run (``jla train`` on real data, Mosaic GPU
kernels for the mixers, 16 x 4096 tokens per step, 20 steps of which the first 5 are warm-up) in a
fresh process, one after another on the same GPU, identical except for ``model.mlp_kernel``
(``xla``: the preset as it is; ``fused``: ``mosaic_gpu``). Repeating the variants interleaved gives
the run-to-run spread. The summary lists tokens/s, MFU, step times, peak memory and losses, with the
tokens/s ratio and the largest per-step loss difference against the first ``xla`` run.
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
VARIANTS: dict[str, list[str]] = {"xla": [], "fused": ["model.mlp_kernel=mosaic_gpu"]}


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--variants", default="xla,fused,xla,fused", help=f"comma-separated, from {sorted(VARIANTS)}")
    p.add_argument("--impl", default="kernels")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1200, help="seconds per run")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="override for every variant")
    p.add_argument("--json", required=True)
    args = p.parse_args(argv)
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not names or any(v not in VARIANTS for v in names):
        p.error(f"variants must be from {sorted(VARIANTS)}; got {names}")
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
    print(f"[bench_fused_gemm_e2e] {args.arch} {variant}: rc={rc} "
          f"tok/s={record.get('result', {}).get('tokens_per_sec', float('nan')):.0f}", flush=True)
    return record


def _summary(runs: list[dict]) -> list[dict]:
    ok = [r for r in runs if r.get("rc") == 0 and r.get("result", {}).get("status") == "ok"]
    base = next((r for r in ok if r["variant"] == "xla"), None)
    rows = []
    for r in ok:
        res = r["result"]
        row = {key: res.get(key) for key in ("tokens_per_sec", "mfu_percent", "step_time_mean_s", "step_time_min_s",
                                             "step_time_max_s", "peak_memory_gib_max_device", "compile_seconds",
                                             "loss_first", "loss_last", "loss_mean_last_steps")}
        row["variant"] = r["variant"]
        if base is not None:
            b = base["result"]
            row["speedup_vs_xla"] = res["tokens_per_sec"] / b["tokens_per_sec"]
            diffs = [abs(s["loss"] - t["loss"]) for s, t in zip(res["per_step"], b["per_step"])]
            row["loss_max_abs_diff_vs_xla"] = max(diffs)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    workdir = Path(tempfile.mkdtemp(prefix=f"fused-gemm-{args.arch}-"))
    runs = []

    def dump():
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "arch": args.arch, "variants": args.variant_names,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "summary": _summary(runs), "runs": runs}, indent=2))

    for i, variant in enumerate(args.variant_names):
        runs.append(_run(args, variant, i, workdir))
        dump()
    for row in _summary(runs):
        print("FUSED_GEMM_E2E " + json.dumps({"arch": args.arch, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                  for k, v in row.items()}}), flush=True)
    return 0 if all(r.get("rc") == 0 for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
