# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tokens/s of a 1.3B preset with the short conv on XLA, then on the Mosaic GPU kernels.

    python benchmarks/bench_short_conv_e2e.py --arch mamba2-1p3b --data /path/to/fineweb_edu_3B \
        --json results/short_conv_e2e_mamba2.json [--order xla,mosaic_gpu]

Runs ``benchmarks/bench_e2e_1p3b.py`` (``jla train``, 1 GPU, ``--impl kernels``, 16 x 4096 tokens per
step, document masking) once per conv implementation, one after the other on the same GPU, with
only ``model.<mixer>.conv_impl`` changed, and writes both result JSONs plus the ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_FIELD = {"mamba2": "ssm"}  # every other conv-using preset family keeps its convs in linear_attention


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--order", default="xla,mosaic_gpu")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--micro-batch", type=int, default=16)
    args = p.parse_args()
    field = _FIELD.get(args.arch.split("-")[0], "linear_attention")
    runs = {}
    for impl in args.order.split(","):
        with tempfile.TemporaryDirectory(prefix=f"sc-e2e-{impl}-") as tmp:
            out = os.path.join(tmp, "result.json")
            cmd = [sys.executable, str(HERE / "bench_e2e_1p3b.py"), "--arch", args.arch, "--data", args.data,
                   "--impl", "kernels", "--micro-batch", str(args.micro_batch), "--steps", str(args.steps),
                   "--skip", str(args.skip), "--json", out, "--set", f'model.{field}.conv_impl="{impl}"']
            proc = subprocess.run(cmd, capture_output=True, text=True)
            result = json.loads(Path(out).read_text()) if Path(out).exists() else {}
            if proc.returncode:
                result.setdefault("status", "failed")
                result["stderr_tail"] = proc.stderr[-3000:]
            result.pop("history", None)
            runs[impl] = result
            summary = result.get("summary", result)
            print(json.dumps({"impl": impl, "status": result.get("status"),
                              "tokens_per_sec": summary.get("tokens_per_sec")}), flush=True)
    payload = {"arch": args.arch, "conv_field": f"model.{field}.conv_impl", "order": args.order, "runs": runs}
    try:
        tps = {k: (v.get("summary") or v)["tokens_per_sec"] for k, v in runs.items()}
        payload["speedup_mosaic_gpu_vs_xla"] = tps["mosaic_gpu"] / tps["xla"]
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=1, default=str))
    return 0 if all(r.get("status") == "ok" for r in runs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
