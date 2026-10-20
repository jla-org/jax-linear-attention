# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tokens/s of a 1.3B preset with the q/k L2 norm in the op, then inside the short-conv kernel.

    python benchmarks/bench_short_conv_l2norm_e2e.py --arch kda-1p3b --data /path/to/fineweb_edu_3B \
        --json results/conv_l2norm_e2e_kda.json [--order off,on] [--rounds 1]

Runs ``benchmarks/bench_e2e_1p3b.py`` (``jla train``, 1 GPU, ``--impl kernels``, 16 x 4096 tokens per step,
document masking) once per setting of ``model.linear_attention.qk_l2norm_in_conv`` (``off`` = the norm in the
recurrent op, the previous path; ``on`` = fused into the conv kernel), one after the other on the same GPU, and
writes every result JSON plus the ratio of the medians. Presets without q/k L2 norm (``mamba2-1p3b``) run the
same code with either setting, so their pair measures the run-to-run noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SETTING = {"off": "false", "on": "true"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--order", default="off,on")
    p.add_argument("--rounds", type=int, default=1, help="repeat the whole order this many times")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--micro-batch", type=int, default=16)
    args = p.parse_args()
    order = args.order.split(",")
    if any(o not in _SETTING for o in order):
        p.error(f"--order entries must be in {sorted(_SETTING)}")
    field = "mamba" if args.arch.startswith("mamba") else "linear_attention"
    runs: dict[str, list] = {o: [] for o in order}
    for rnd in range(args.rounds):
        for setting in order:
            with tempfile.TemporaryDirectory(prefix=f"l2n-e2e-{setting}-") as tmp:
                out = Path(tmp) / "result.json"
                cmd = [sys.executable, str(HERE / "bench_e2e_1p3b.py"), "--arch", args.arch, "--data", args.data,
                       "--impl", "kernels", "--micro-batch", str(args.micro_batch), "--steps", str(args.steps),
                       "--skip", str(args.skip), "--json", str(out)]
                if field == "linear_attention":
                    cmd += ["--set", f"model.linear_attention.qk_l2norm_in_conv={_SETTING[setting]}"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                result = json.loads(out.read_text()) if out.exists() else {}
                if proc.returncode:
                    result.setdefault("status", "failed")
                    result["stderr_tail"] = proc.stderr[-3000:]
                result.pop("history", None)
                result["round"] = rnd
                runs[setting].append(result)
                summary = result.get("summary", result)
                print(json.dumps({"round": rnd, "qk_l2norm_in_conv": setting, "status": result.get("status"),
                                  "tokens_per_sec": summary.get("tokens_per_sec")}), flush=True)
    payload = {"arch": args.arch, "switch": "model.linear_attention.qk_l2norm_in_conv", "order": args.order,
               "rounds": args.rounds, "runs": runs}
    try:
        tps = {k: statistics.median((r.get("summary") or r)["tokens_per_sec"] for r in v) for k, v in runs.items()}
        payload["tokens_per_sec_median"] = tps
        payload["speedup_on_vs_off"] = tps["on"] / tps["off"]
    except (KeyError, TypeError, ZeroDivisionError, statistics.StatisticsError):
        pass
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=1, default=str))
    return 0 if all(r.get("status") == "ok" for v in runs.values() for r in v) else 1


if __name__ == "__main__":
    raise SystemExit(main())
