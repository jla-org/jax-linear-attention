# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute policy x micro-batch x FP8 at 1.3B on one GPU, through the release harness (benchmarks/fp8_1p3b.md).

    python benchmarks/bench_remat_fp8_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B \
        --variant fp8:full:16x1 --variant fp8:none:4x4 --variant fp8:names=linear+kernels:8x2 \
        --variant bf16:none:4x4:lean --variant bf16:none:4x4:leancore --json results/remat_fp8_transformer.json

Each ``--variant`` is ``<dtype>:<remat>[=<name>+<name>...]:<micro batch>x<grad accum>[:<flags>]``, where the
flags are ``lean`` or ``leancore`` and/or ``noscan``, joined by ``+`` (e.g. ``fp8:none:4x4:lean+noscan``):

* ``<dtype>``: ``fp8`` (``model.fp8 = "delayed"``, ``model.fp8_min_features = 512``, ``model.fp8_fused = true``,
  the FP8 recipe of benchmarks/fp8_1p3b.md) or ``bf16`` (FP8 off, the presets' default);
* ``<remat>``: any ``model.remat`` policy (``none``, ``dots``, ``dots_with_no_batch_dims``, ``full``, ...);
  ``names=<a>+<b>`` sets ``model.remat = "names"`` with ``model.remat_names = [a, b]``;
* ``<micro batch>x<grad accum>``: ``train.micro_batch_per_replica`` x ``train.grad_accum`` (65,536 tokens per
  step at 16x1, 8x2 and 4x4 with 4096-token rows). With more than one micro-step the micro-step loop is
  rolled (``train.scan_unroll = 1``): unrolled, XLA keeps several micro-steps' activations alive
  (benchmarks/fp8_1p3b.md);
* ``lean``: ``model.lean_vjp = true``; ``leancore``: also ``model.lean_remat_core = true`` (with ``fp8``
  this needs the ``fp8_fused`` FP8 SwiGLU, which the recipe above sets);
* ``noscan``: ``model.scan_layers = false`` (the layers unrolled instead of a ``lax.scan``; under
  ``remat = "none"`` the scanned step stacks every layer's residuals with copies and launches the attention
  forward kernel twice per layer, benchmarks/fp8_1p3b.md).

Every variant is one ``benchmarks/bench_e2e_1p3b.py`` run (``jla train`` on real data, Mosaic GPU kernels for
every mixer and the FA3-style attention, XLA's default BFC allocator, 20 steps of which the first 5 are
warm-up) in a fresh process, one after another on the same GPU. The JSON is rewritten after every run,
so the runs that finished survive a dead node. Each record has tokens/s, MFU, HFU (the replayed forward
counted by ``jla.training.performance.estimate_hfu_flops_per_token``), peak and compiled memory, compile
time, every step's loss and the failure message of a run that did not finish (for example out of memory).
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
FP8 = ["model.fp8=\"delayed\"", "model.fp8_min_features=512", "model.fp8_fused=true"]
_KEEP = ("status", "tokens_per_sec", "mfu_percent", "hfu_percent", "step_time_mean_s", "step_time_median_s",
         "step_time_min_s", "step_time_max_s", "tokens_per_step", "peak_memory_gib_max_device",
         "compiled_memory_estimate_gib", "compile_seconds", "loss_first", "loss_last", "loss_mean_first_steps",
         "loss_mean_last_steps", "loss_finite", "loss_decreasing", "per_step", "gpus_before", "gpus_after",
         "micro_batch_per_replica", "grad_accum", "flops_per_token", "git_head", "source_sha256",
         "wall_seconds_incl_compile", "xla_client_env", "error", "jax_version", "device_kind")


def parse_variant(text: str) -> dict:
    """``fp8:names=linear+kernels:8x2:lean`` -> the variant's fields and its ``--set`` overrides."""
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"variant {text!r}: expected <dtype>:<remat>[=<names>]:<mb>x<ga>[:lean|:leancore]")
    dtype, policy, batch = parts[:3]
    flags = set(parts[3].split("+")) if len(parts) == 4 else set()
    if dtype not in ("fp8", "bf16"):
        raise ValueError(f"variant {text!r}: dtype must be fp8 or bf16")
    if not flags <= {"lean", "leancore", "noscan"} or {"lean", "leancore"} <= flags:
        raise ValueError(f"variant {text!r}: the fourth field is lean or leancore and/or noscan, joined by +")
    extra = "leancore" if "leancore" in flags else "lean" if "lean" in flags else ""
    remat, _, names = policy.partition("=")
    mb, _, ga = batch.partition("x")
    micro_batch, grad_accum = int(mb), int(ga or 1)
    remat_names = [n for n in names.split("+") if n]
    overrides = list(FP8) if dtype == "fp8" else []
    overrides.append(f'model.remat="{remat}"')
    if remat_names:
        overrides.append("model.remat_names=[" + ",".join(f'"{n}"' for n in remat_names) + "]")
    # Always explicit: the transformer TOMLs now set lean_vjp = true and scan_layers = false themselves, and a
    # variant without the flags means the scanned, non-lean step.
    overrides.append(f"model.lean_vjp={'true' if extra else 'false'}")
    overrides.append(f"model.lean_remat_core={'true' if extra == 'leancore' else 'false'}")
    overrides.append(f"model.scan_layers={'false' if 'noscan' in flags else 'true'}")
    if grad_accum > 1:
        overrides.append("train.scan_unroll=1")
    return {"label": text, "dtype": dtype, "remat": remat, "remat_names": remat_names, "micro_batch": micro_batch,
            "grad_accum": grad_accum, "lean_vjp": bool(extra), "lean_remat_core": extra == "leancore",
            "scan_layers": "noscan" not in flags, "overrides": overrides}


def _run(args, variant: dict, index: int, workdir: Path) -> dict:
    out = workdir / f"run{index}.json"
    cmd = [sys.executable, str(HERE / "bench_e2e_1p3b.py"), "--arch", args.arch, "--data", args.data,
           "--impl", "kernels", "--micro-batch", str(variant["micro_batch"]), "--grad-accum", str(variant["grad_accum"]),
           "--steps", str(args.steps), "--skip", str(args.skip), "--stall-timeout", str(args.stall_timeout),
           "--json", str(out)]
    for override in variant["overrides"]:
        cmd += ["--set", override]
    rec = {k: v for k, v in variant.items()}
    rec["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        rec["returncode"] = proc.returncode
        tail = (proc.stdout + proc.stderr)[-3000:]
    except subprocess.TimeoutExpired as exc:
        rec["returncode"] = None
        tail = f"timeout after {args.timeout} s: {str(exc.stdout or '')[-1000:]}{str(exc.stderr or '')[-1500:]}"
    rec["wall_s"] = time.time() - t0
    result = json.loads(out.read_text()) if out.exists() else {}
    rec.update({k: result[k] for k in _KEEP if k in result})
    if rec.get("status") != "ok":
        rec["status"] = result.get("status", "failed")
        low = (str(result.get("error", "")) + tail).lower()
        if "resource_exhausted" in low or "out of memory" in low:
            rec["status"] = "oom"
        rec["log_tail"] = tail
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--stall-timeout", type=float, default=300.0)
    p.add_argument("--json", required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    variants = [parse_variant(v) for v in args.variant]
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    env = {k: os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
                                          "XLA_PYTHON_CLIENT_MEM_FRACTION")}
    with tempfile.TemporaryDirectory(prefix="remat-fp8-") as tmp:
        for i, variant in enumerate(variants):
            rec = _run(args, variant, i, Path(tmp))
            records.append(rec)
            out_path.write_text(json.dumps({"arch": args.arch, "steps": args.steps, "skip": args.skip,
                                            "xla_client_env": env, "records": records}, indent=2))
            print("REMAT_FP8 " + json.dumps({k: rec.get(k) for k in (
                "label", "status", "tokens_per_sec", "mfu_percent", "hfu_percent", "peak_memory_gib_max_device",
                "compiled_memory_estimate_gib", "loss_last", "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
