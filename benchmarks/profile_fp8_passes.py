# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every FP8 quantize / amax / transpose pass of a profiled train step (benchmarks/fp8_1p3b.md).

    python benchmarks/profile_fp8_passes.py /path/to/profile-dir [--json out.json] [--top 60]

Reads a directory written by ``benchmarks/profile_1p3b.py --profile-dir`` (the trace, the compiled
step's HLO text and ``profile_meta.json``; no JAX needed) and lists every GPU kernel that writes an
FP8 tensor or that JAX traced from ``layers/fp8.py`` / ``layers/fp8_swiglu.py`` (casts, amax
reductions, transposed FP8 copies, the fused SwiGLU kernels), except the GEMMs. For each HLO op it
reports the phase (forward, recomputed forward under remat, backward), whether it reads a weight or
an activation (an output dim of at least ``--token-dim`` elements means tokens), what it does (cast,
amax, transpose, fused kernel), its launches per optimizer step (a pass inside the micro-step loop runs
``grad_accum`` times per step) and its device time per step. The summary splits the time by kind and
phase, and counts how often each weight quantization runs per step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench_end_to_end_step import _load_trace  # noqa: E402
from profile_1p3b import _GEMM_KERNEL_RE, _jla_frames, hlo_instructions, phase_of  # noqa: E402

_FP8_FILES = ("fp8.py", "fp8_swiglu.py", "fp8_fused.py")
_DIMS_RE = re.compile(r"\[([\d,]*)\]")
_SHAPE_RE = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.+?)\s+[a-z][\w\-]*\(")


def full_shapes(hlo_text: str) -> dict[str, str]:
    """``instruction -> its whole (tuple) result shape``; ``hlo_instructions`` keeps only the first word."""
    out: dict[str, str] = {}
    for line in hlo_text.splitlines():
        m = _SHAPE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _is_activation(shape: str, token_dim: int) -> bool:
    """A result with a token axis: one dim, or the product of all but the last dim, of at least ``token_dim``;
    or only bfloat16 partial reductions (the first stage of an activation's amax: weights are float32)."""
    if "f8e" not in shape and "bf16[" in shape:
        return True
    for dims in _dims(shape):
        if dims and (max(dims) >= token_dim or _prod(dims[:-1]) >= token_dim):
            return True
    return False


def _prod(dims: list[int]) -> int:
    out = 1
    for d in dims:
        out *= d
    return out


def _dims(shape: str) -> list[list[int]]:
    return [[int(d) for d in m.split(",") if d] for m in _DIMS_RE.findall(shape)]


def _what(kernel: str, info: dict) -> str:
    lname = kernel.lower()
    op = info.get("opcode", "")
    target = info.get("target", "").lower()
    if "pallas" in info.get("op_name", "") or "mosaic" in target or "triton" in target:
        return "fused kernel (Pallas)"
    if "transpose" in lname or op in ("transpose", "copy"):
        return "transpose / layout copy"
    shape = info.get("shape", "")
    writes_fp8 = "f8e4m3fn" in shape or "f8e5m2" in shape
    if "reduce" in lname and writes_fp8:
        return "cast + amax"
    if writes_fp8:
        return "cast"
    if "reduce" in lname or op == "reduce":
        return "amax"
    return "other"


def _call_site(info: dict) -> str:
    """The first frame outside the FP8 modules and ``linear.py`` (where the projection is called)."""
    frames = _jla_frames(info)
    for f in frames:
        base = os.path.basename(f[0])
        if base not in _FP8_FILES and base != "linear.py":
            return f"{base}:{f[1]}"
    return f"{os.path.basename(frames[0][0])}:{frames[0][1]}" if frames else "(no source)"


def fp8_passes(profile_dir: Path, hlo_text: str, steps: int, token_dim: int) -> dict:
    events = _load_trace(profile_dir)
    pnames = {e.get("pid"): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {pid for pid, n in pnames.items() if "GPU" in n or "gpu" in n or "/device:" in n}
    meta = hlo_instructions(hlo_text)
    shapes = full_shapes(hlo_text)
    per_op: dict[str, dict] = {}
    gemm_ms = 0.0
    total_ms = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0)) / 1e3
        args = e.get("args", {}) or {}
        name = e.get("name", "")
        hlo_op = args.get("hlo_op") or args.get("long_name", "").split(" ")[0].lstrip("%") or ""
        if not hlo_op or hlo_op.startswith("command_buffer"):
            dotted = re.sub(r"_(\d+)$", r".\1", name)
            hlo_op = dotted if dotted in meta else name
        info = meta.get(hlo_op) or meta.get(name) or {}
        total_ms += dur
        target = info.get("target", "").lower()
        is_gemm = (info.get("opcode") in ("dot", "convolution") or "cublas" in target or "gemm" in target
                   or info.get("contains_dot") or bool(_GEMM_KERNEL_RE.search(name)))
        if is_gemm:
            gemm_ms += dur
            continue
        files = [os.path.basename(f[0]) for f in _jla_frames(info)]
        shape = shapes.get(hlo_op, info.get("shape", ""))
        info = {**info, "shape": shape}
        writes_fp8 = "f8e4m3fn" in shape or "f8e5m2" in shape
        if not (writes_fp8 or any(f in _FP8_FILES for f in files[:3])):
            continue
        row = per_op.setdefault(hlo_op, {
            "hlo_op": hlo_op, "kernel": name[:80], "phase": phase_of(info) if info.get("op_name") else "(no metadata)",
            "shape": shape[:120],
            "what": _what(name, info), "call_site": _call_site(info),
            "source": f"{files[0]}:{_jla_frames(info)[0][1]}" if files else "",
            "kind": "activation" if _is_activation(shape, token_dim) else "weight",
            "launches": 0, "ms": 0.0})
        row["launches"] += 1
        row["ms"] += dur
    rows = []
    for row in per_op.values():
        row = dict(row)
        row["launches_per_step"] = row.pop("launches") / steps
        row["ms_per_step"] = row.pop("ms") / steps
        rows.append(row)
    rows.sort(key=lambda r: -r["ms_per_step"])
    by_kind: dict[str, float] = defaultdict(float)
    for r in rows:
        by_kind[f"{r['kind']} | {r['what']} | {r['phase']}"] += r["ms_per_step"]
    groups: dict[tuple, dict] = {}
    for r in rows:  # the same pass of every layer (unrolled layers are separate HLO ops)
        shape = re.sub(r"\{[\d,]*\}", "", r["shape"])
        g = groups.setdefault((r["phase"], r["kind"], r["what"], r["call_site"], shape),
                              {"phase": r["phase"], "kind": r["kind"], "what": r["what"], "call_site": r["call_site"],
                               "shape": shape, "hlo_ops": 0, "launches_per_step": 0.0, "ms_per_step": 0.0})
        g["hlo_ops"] += 1
        g["launches_per_step"] += r["launches_per_step"]
        g["ms_per_step"] += r["ms_per_step"]
    grouped = sorted(groups.values(), key=lambda g: -g["ms_per_step"])
    weight_launches: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["kind"] == "weight" and r["what"] in ("cast", "cast + amax", "transpose / layout copy", "amax"):
            weight_launches[f"{r['launches_per_step']:g} per step"] += 1
    return {
        "profile_steps": steps,
        "device_ms_per_step": total_ms / steps,
        "gemm_ms_per_step": gemm_ms / steps,
        "fp8_pass_ms_per_step": sum(r["ms_per_step"] for r in rows),
        "by_kind_ms_per_step": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "weight_pass_ops_by_launches": dict(weight_launches),
        "grouped": grouped,
        "passes": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("profile_dir")
    p.add_argument("--json", default=None)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--token-dim", type=int, default=16384,
                   help="an output dim at least this large marks an activation (tokens per micro-step)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    pdir = Path(args.profile_dir)
    saved = json.loads((pdir / "profile_meta.json").read_text())
    out = fp8_passes(pdir, (pdir / "train_step.hlo.txt").read_text(), saved["profile_steps"], args.token_dim)
    out["step"] = {k: saved.get("result", {}).get(k) for k in ("arch", "micro_batch", "grad_accum", "remat",
                                                               "ms_per_step_median", "tok_per_s", "mfu_percent")}
    out["step"]["wall_ms_median"] = saved.get("wall_ms_median")
    print(f"step {out['step']}")
    print(f"device {out['device_ms_per_step']:.1f} ms/step; GEMMs {out['gemm_ms_per_step']:.1f}; "
          f"FP8 passes {out['fp8_pass_ms_per_step']:.1f}")
    print("by kind | what | phase:")
    for k, v in out["by_kind_ms_per_step"].items():
        print(f"  {v:8.2f} ms  {k}")
    print("weight pass ops by launches per step:", out["weight_pass_ops_by_launches"])
    print("grouped (same pass over all layers):")
    for g in out["grouped"][: args.top]:
        print(f"  {g['ms_per_step']:7.2f} ms ops {g['hlo_ops']:<3d} launches {g['launches_per_step']:<5g} "
              f"{g['phase'][:5]:5s} {g['kind']:10s} {g['what']:24s} {g['shape'][:64]:64s} {g['call_site']}")
    print("passes:")
    for r in out["passes"][: args.top]:
        print(f"  {r['ms_per_step']:7.2f} ms x{r['launches_per_step']:<4g} {r['phase'][:5]:5s} {r['kind']:10s} "
              f"{r['what']:24s} {r['hlo_op'][:34]:34s} {r['shape'][:70]:70s} {r['call_site']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
