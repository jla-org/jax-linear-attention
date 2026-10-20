# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speed-of-light model of one profiled training step (benchmarks/profile_1p3b.md).

    python benchmarks/speed_of_light_step.py --profile-dir <dir> [--profile-dir <dir> ...] \
        --bench-json gemm_xla.json --bench-json gemm_mosaic.json --bench-json torch.json \
        --bench-json torch_clock_overlap.json --write-ceilings ceilings.json --json out.json
    python benchmarks/speed_of_light_step.py --profile-dir <dir> --ceilings ceilings.json --json out.json

``<dir>`` is a ``--profile-dir`` written by ``benchmarks/profile_1p3b.py`` or ``bench_end_to_end_step.py``
(``train_step.hlo.txt``, ``profile_meta.json`` and the ``jax.profiler`` trace, recorded with XLA command
buffers off so that every kernel carries its HLO instruction). For every HLO instruction that ran on
the GPU the script computes:

* executions per step (the product of the trip counts of the ``while`` loops around it),
* FLOPs: ``2 * |out| * |contraction|`` of every dot (cuBLAS custom calls, Triton GEMM fusions, dots
  inside fusions); a Mosaic GPU kernel is credited with the analytic recurrence/attention-core FLOPs
  of ``jla.training.performance`` (fwd 1x, bwd 2x of the forward core, shared by its kernels),
* HBM bytes: operands read plus results written, where a fusion that only slices an operand reads the
  slices, an in-place ``dynamic-update-slice`` writes only the update and cuBLAS workspaces are dropped,
* its minimum time at the measured ceilings, ``max(FLOPs / GEMM TFLOP/s at that shape, bytes / HBM GB/s)``,
* its measured time (sum of its kernels' durations in the trace / profiled steps).

The ceilings come from ``bench_speed_of_light.py`` / ``speed_of_light_torch.py`` JSONs (``--bench-json``):
per (M, K, N) GEMM problem the best bf16 and FP8 rate of any implementation, the best elementwise HBM
rate, and the sustained / burst ratio of the back-to-back GEMM loops (the power cap). An op that ran
faster than its micro-benchmark ceiling counts as its own ceiling.

It reports the step's speed of light as ``as_compiled`` (every executed op at its own ceiling,
recomputation included), ``no_remat`` (without the recomputed forward), ``ideal`` (no recomputed
forward and no logits recompute in the cross-entropy backward; elementwise passes, casts and layout
copies fused into neighbouring GEMMs or kernels; every GEMM and kernel at its ceiling, the attention
kernels executing only ``--attention-work-fraction`` of their counted causal FLOPs, the tiles that
document masking leaves), ``ideal_power_capped`` (the tensor-core work at the sustained rate) and the
two ideal steps with FP8 GEMMs, each as ms and MFU (model FLOPs / 989.4 TFLOP/s), plus the ranked gap
list in ms per step and every op of at least 0.1 ms per step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench_end_to_end_step import _load_trace, stack_frames  # noqa: E402

SPEC_TFLOPS = 989.4
DTYPE_BYTES = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2, "s32": 4, "u32": 4, "f32": 4,
               "s64": 8, "u64": 8, "f64": 8, "c64": 8, "c128": 16, "token": 0, "s4": 0.5, "u4": 0.5}
_ARRAY_RE = re.compile(r"\b([a-z]+[0-9]*(?:e[0-9]m[0-9](?:fn|fnuz)?)?)\[([0-9,]*)\]")
_HEADER_RE = re.compile(r"^(ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s+->\s+.*\{\s*$")
_INSTR_RE = re.compile(r"^\s*(ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
_NOKERNEL = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "while", "conditional", "call",
             "after-all", "partition-id", "replica-id", "opt-barrier", "add-dependency", "copy-start", "copy-done"}
_GEMM_TARGETS = ("__cublas$lt$matmul", "__cublas$gemm")


def _dtype_bytes(dtype: str) -> float:
    if dtype.startswith("f8") or dtype.startswith("f4") or dtype.startswith("f6"):
        return 1.0
    return DTYPE_BYTES.get(dtype, 4)


def shape_arrays(shape: str) -> list[tuple[str, tuple[int, ...]]]:
    return [(m.group(1), tuple(int(d) for d in m.group(2).split(",") if d)) for m in _ARRAY_RE.finditer(shape)]


def shape_bytes(shape: str) -> float:
    return sum(_dtype_bytes(t) * math.prod(dims) for t, dims in shape_arrays(shape))


def _balanced(text: str, start: int) -> int:
    """Index of the parenthesis closing the one at ``text[start]``."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def parse_instruction(line: str) -> dict | None:
    m = _INSTR_RE.match(line)
    if not m:
        return None
    root, name, rest = bool(m.group(1)), m.group(2), m.group(3)
    if rest.startswith("("):
        end = _balanced(rest, 0)
        shape, tail = rest[: end + 1], rest[end + 1:].lstrip()
    else:
        shape, _, tail = rest.partition(" ")
    om = re.match(r"([a-z][\w\-]*)\(", tail)
    if not om:
        return None
    opcode = om.group(1)
    end = _balanced(tail, om.end() - 1)
    operands = re.findall(r"%([\w.\-]+)", tail[om.end(): end])
    attrs = tail[end + 1:]
    return {"name": name, "root": root, "shape": shape, "opcode": opcode, "operands": operands, "attrs": attrs}


def parse_module(hlo_text: str) -> tuple[dict[str, list[dict]], str, dict[str, dict]]:
    """(computation -> instructions, entry computation name, instruction name -> instruction)."""
    comps: dict[str, list[dict]] = {}
    entry = ""
    current = None
    by_name: dict[str, dict] = {}
    for line in hlo_text.splitlines():
        h = _HEADER_RE.match(line)
        if h and " = " not in line.split("(")[0]:
            current = h.group(2)
            comps[current] = []
            if h.group(1):
                entry = current
            continue
        if current is None:
            continue
        if line.strip() == "}":
            current = None
            continue
        ins = parse_instruction(line)
        if ins is None:
            continue
        ins["computation"] = current
        comps[current].append(ins)
        by_name[ins["name"]] = ins
    return comps, entry, by_name


def _attr(attrs: str, key: str) -> str | None:
    m = re.search(rf"\b{key}=%?([\w.\-]+)", attrs)
    return m.group(1) if m else None


def _dims_attr(attrs: str, key: str) -> tuple[int, ...]:
    m = re.search(rf"\b{key}=\{{([0-9,]*)\}}", attrs)
    return tuple(int(x) for x in m.group(1).split(",") if x) if m else ()


def multipliers(comps: dict[str, list[dict]], entry: str) -> dict[str, float]:
    """Executions per step of each (non-fused) computation: products of while trip counts."""
    mult: dict[str, float] = {entry: 1.0}
    stack = [entry]
    while stack:
        comp = stack.pop()
        m = mult[comp]
        for ins in comps.get(comp, []):
            op, attrs = ins["opcode"], ins["attrs"]
            children: list[tuple[str, float]] = []
            if op == "while":
                trip = re.search(r'known_trip_count"\s*:\s*\{\s*"n"\s*:\s*"(\d+)"', attrs)
                n = float(trip.group(1)) if trip else 1.0
                body, cond = _attr(attrs, "body"), _attr(attrs, "condition")
                children = [(body, m * n), (cond, m * (n + 1))]
            elif op == "conditional":
                branches = re.search(r"branch_computations=\{([^}]*)\}", attrs)
                names = re.findall(r"%?([\w.\-]+)", branches.group(1)) if branches else []
                names += [x for x in (_attr(attrs, "true_computation"), _attr(attrs, "false_computation")) if x]
                children = [(b, m) for b in names]  # only the branch that runs appears in the trace
            elif op in ("call", "async-start"):
                callee = _attr(attrs, "to_apply") or _attr(attrs, "calls")
                children = [(callee, m)] if callee else []
            for child, cm in children:
                if child and child in comps and child not in mult:
                    mult[child] = cm
                    stack.append(child)
    return mult


def _dot_flops(ins: dict, names: dict[str, dict]) -> float:
    lhs = names.get(ins["operands"][0]) if ins["operands"] else None
    if lhs is None:
        return 0.0
    arrays = shape_arrays(lhs["shape"])
    if not arrays:
        return 0.0
    lhs_dims = arrays[0][1]
    contract = _dims_attr(ins["attrs"], "lhs_contracting_dims")
    out = shape_arrays(ins["shape"])
    out_elems = math.prod(out[0][1]) if out else 0
    return 2.0 * out_elems * math.prod(lhs_dims[d] for d in contract)


def computation_flops(comp: str, comps: dict[str, list[dict]], names: dict[str, dict], seen=None) -> float:
    """FLOPs of the dots in a fused computation (nested fusions included)."""
    seen = seen or set()
    if comp in seen:
        return 0.0
    seen.add(comp)
    total = 0.0
    for ins in comps.get(comp, []):
        if ins["opcode"] == "dot":
            total += _dot_flops(ins, names)
        callee = _attr(ins["attrs"], "calls")
        if callee and ins["opcode"] in ("fusion", "call"):
            total += computation_flops(callee, comps, names, seen)
    return total


def _gemm_custom_call(ins: dict, names: dict[str, dict]) -> tuple[float, tuple[int, int, int] | None]:
    cfg = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', ins["attrs"])
    lhs_batch = re.search(r'"lhs_batch_dimensions":\[([^\]]*)\]', ins["attrs"])
    lhs = names.get(ins["operands"][0]) if ins["operands"] else None
    out = shape_arrays(ins["shape"])
    if not (cfg and lhs and out):
        return 0.0, None
    contract = [int(x.strip('" ')) for x in cfg.group(1).split(",") if x.strip('" ')]
    batch = [int(x.strip('" ')) for x in lhs_batch.group(1).split(",") if x.strip('" ')] if lhs_batch else []
    lhs_dims = shape_arrays(lhs["shape"])[0][1]
    k = math.prod(lhs_dims[d] for d in contract)
    out_dims = out[0][1]
    b = math.prod(lhs_dims[d] for d in batch) if batch else 1
    mn = math.prod(out_dims) // max(1, b)
    lhs_free = [lhs_dims[d] for d in range(len(lhs_dims)) if d not in contract and d not in batch]
    m = math.prod(lhs_free) if lhs_free else 1
    n = mn // max(1, m)
    return 2.0 * math.prod(out_dims) * k, (m * b, k, n)


def fusion_bytes(ins: dict, comps: dict[str, list[dict]], names: dict[str, dict]) -> float:
    """HBM bytes of one fusion: operands (sliced reads counted as the slices, in-place buffers as 0)
    plus results (an in-place dynamic-update-slice writes its update only)."""
    callee = _attr(ins["attrs"], "calls")
    body = comps.get(callee or "", [])
    if not body:
        return sum(shape_bytes(names[o]["shape"]) for o in ins["operands"] if o in names) + shape_bytes(ins["shape"])
    users: dict[str, list[dict]] = defaultdict(list)
    for b in body:
        for o in b["operands"]:
            users[o].append(b)
    local = {b["name"]: b for b in body}

    def read_bytes(name: str, full: float, depth: int = 0) -> float:
        us = users.get(name, [])
        if not us or depth > 8:
            return full
        total = 0.0
        for u in us:
            if u["opcode"] == "bitcast":
                total += read_bytes(u["name"], full, depth + 1)
            elif u["opcode"] in ("dynamic-slice", "slice", "gather") and u["operands"][0] == name:
                total += shape_bytes(u["shape"])
            elif u["opcode"] == "dynamic-update-slice" and u["operands"][0] == name:
                total += 0.0
            else:
                return full
        return min(total, full)

    reads = 0.0
    for b in body:
        if b["opcode"] == "parameter":
            reads += read_bytes(b["name"], shape_bytes(b["shape"]))

    def write_bytes(b: dict, depth: int = 0) -> float:
        if b["opcode"] == "dynamic-update-slice" and len(b["operands"]) > 1 and b["operands"][1] in local:
            return shape_bytes(local[b["operands"][1]]["shape"])
        if b["opcode"] == "bitcast" and b["operands"] and b["operands"][0] in local and depth < 8:
            return write_bytes(local[b["operands"][0]], depth + 1)
        if b["opcode"] == "tuple":
            return sum(write_bytes(local[o], depth + 1) if o in local else 0.0 for o in b["operands"])
        return shape_bytes(b["shape"])

    root = next((b for b in body if b["root"]), body[-1])
    return reads + write_bytes(root)


def _only_layout(comp: str, comps: dict[str, list[dict]]) -> bool:
    ops = {b["opcode"] for b in comps.get(comp, [])} - {"parameter", "bitcast", "tuple", "get-tuple-element"}
    return bool(ops) and ops <= {"transpose", "copy", "reshape", "broadcast", "slice", "concatenate", "pad",
                                 "dynamic-slice", "dynamic-update-slice", "constant"}


def _only_convert(comp: str, comps: dict[str, list[dict]]) -> bool:
    ops = {b["opcode"] for b in comps.get(comp, [])} - {"parameter", "bitcast", "tuple", "get-tuple-element"}
    return "convert" in ops and ops <= {"convert", "copy", "transpose", "reshape", "broadcast", "constant"}


def phase_of(op_name: str, frames: list[tuple[str, str, str]]) -> str:
    if any("/optax/" in f[0] or os.path.basename(f[0]) == "optimizers.py" for f in frames[:3]):
        return "opt"
    if "rematted_computation" in op_name:
        return "remat"
    if "transpose(" in op_name:
        return "bwd"
    return "fwd"


def _jla_site(frames: list[tuple[str, str, str]]) -> str:
    for f in frames:
        if "/jla/" in f[0] and os.path.basename(f[0]) not in ("linear.py",):
            return f"{f[0].split('/jla/')[-1]}:{f[1]}"
    for f in frames:
        if "/jla/" in f[0]:
            return f"{f[0].split('/jla/')[-1]}:{f[1]}"
    return ""


def _gemm_tflops(ceilings: dict, mkn: tuple[int, int, int] | None, table_name: str = "gemm_bf16_tflops") -> float:
    """Measured GEMM ceiling for an (M, K, N) problem: the exact problem (or its transpose, which costs
    the same), else the nearest measured problem in log distance, else ``gemm_default_tflops``."""
    table = ceilings.get(table_name, {})
    if mkn is not None:
        m, k, n = mkn
        for key in (f"{m}x{k}x{n}", f"{n}x{k}x{m}"):
            if key in table:
                return float(table[key])
        best, dist = None, None
        for key, v in table.items():
            try:
                mm, kk, nn = (int(x) for x in key.split("x"))
            except ValueError:
                continue
            for a_m, a_n in ((m, n), (n, m)):
                d = sum(abs(math.log(a / b)) for a, b in ((a_m, mm), (k, kk), (a_n, nn)))
                if dist is None or d < dist:
                    best, dist = float(v), d
        if best is not None and dist is not None and dist < 1.5:
            return best
    if table_name != "gemm_bf16_tflops":
        return float(ceilings.get("gemm_fp8_default_tflops", 2 * SPEC_TFLOPS))
    return float(ceilings.get("gemm_default_tflops", SPEC_TFLOPS))


def build_ceilings(paths: list[str]) -> dict:
    """Ceilings from ``bench_speed_of_light.py`` (gemm, hbm) and ``speed_of_light_torch.py`` JSONs: per
    (M, K, N) GEMM problem the best bf16 TFLOP/s of any implementation, the best FP8 one, and the best
    achievable HBM bandwidth of a large elementwise pass (read + write)."""
    bf16: dict[str, float] = {}
    fp8: dict[str, float] = {}
    hbm = []
    sustained: dict[str, list[float]] = {}
    sources: dict[str, str] = {}

    def put(table, key, tf, src):
        if tf and tf > table.get(key, 0.0):
            table[key] = tf
            sources[("fp8 " if table is fp8 else "bf16 ") + key] = src

    for path in paths:
        data = json.loads(Path(path).read_text())
        for row in data.get("rows", []) + data.get("bf16", []) + data.get("fp8", []):
            if "tflops" not in row:
                continue
            if "gemm" in row:
                mm, kk, nn = row["gemm"]
            elif "lhs" in row:
                ls, out = row["lhs"], row["out"]
                kk = row["flops"] / 2 / math.prod(out)
                mm, nn = out
            else:
                continue
            key = f"{int(mm)}x{int(kk)}x{int(nn)}"
            src = f"{row.get('impl')} {row.get('layout', '')} {row.get('variant', '')}".strip()
            put(fp8 if row.get("dtype") == "fp8" else bf16, key, float(row["tflops"]), src)
        for row in data.get("rows", []) + data.get("hbm", []):
            if "gbps" in row and "read+write" in row.get("op", ""):
                hbm.append(float(row["gbps"]))
        for row in data.get("clock", []) if isinstance(data.get("clock"), list) else []:
            sustained.setdefault("fp8" if row["gemm"].startswith("fp8") else "bf16", []).append(float(row["tflops"]))
        if data.get("cmd") == "clock" and "sustained_tflops_wall" in data:
            sustained.setdefault("bf16", []).append(float(data["sustained_tflops_wall"]))
    cube = bf16.get("8192x8192x8192", max(bf16.values()) if bf16 else SPEC_TFLOPS)
    cube8 = fp8.get("8192x8192x8192", max(fp8.values()) if fp8 else 2 * SPEC_TFLOPS)
    # sustained (power-capped) / burst rate of the 8192^3 GEMM: the median over the clock runs
    scale16 = (statistics.median(sustained["bf16"]) / cube) if sustained.get("bf16") else 1.0
    scale8 = (statistics.median(sustained["fp8"]) / cube8) if sustained.get("fp8") else 1.0
    return {"gemm_bf16_tflops": bf16, "gemm_fp8_tflops": fp8, "gemm_default_tflops": cube,
            "gemm_fp8_default_tflops": cube8, "power_scale_bf16": round(min(1.0, scale16), 4),
            "power_scale_fp8": round(min(1.0, scale8), 4), "sustained_tflops": sustained,
            "kernel_tflops": cube, "hbm_gbps": max(hbm) if hbm else 3352.0, "sources": sources}


def ce_logits_recompute(rows: list[dict]) -> set[int]:
    """Indices of the backward GEMMs of the cross entropy that recompute the forward's logits: a backward
    GEMM at an ``ops/cross_entropy.py`` site with the same (M, K, N) problem (or its transpose) as a
    forward GEMM there. A fused linear cross entropy computes dlogits during the forward pass and never
    runs them, so the ideal step leaves them out (the dgrad and wgrad GEMMs stay)."""
    def is_ce_gemm(r: dict) -> bool:
        return r["category"] == "GEMM" and bool(r.get("gemm")) and "cross_entropy.py" in r.get("site", "")

    fwd = set()
    for r in rows:
        if is_ce_gemm(r) and r["phase"] == "fwd":
            m, k, n = r["gemm"]
            fwd.update({(m, k, n), (n, k, m)})
    return {i for i, r in enumerate(rows) if is_ce_gemm(r) and r["phase"] == "bwd" and tuple(r["gemm"]) in fwd}


def analyze(profile_dir: Path, ceilings: dict, top: int = 40, attention_work_fraction: float = 1.0) -> dict:
    """``attention_work_fraction``: the share of the counted causal attention-core FLOPs that a kernel
    skipping fully masked tiles executes on the profiled batch (document masking; from
    ``speed_of_light_prototypes.py --doc-stats``). Only the ideal steps use it; 1.0 charges the full
    causal work, as the MFU numerator counts it."""
    hlo_text = (profile_dir / "train_step.hlo.txt").read_text()
    meta = json.loads((profile_dir / "profile_meta.json").read_text())
    steps = int(meta["profile_steps"])
    wall_ms = float(meta.get("wall_ms_median") or 0.0)
    result = meta.get("result", {})
    tokens = int(result.get("micro_batch", 16)) * int(result.get("seq_len", 4096))
    fpt = result.get("flops_per_token", {})
    model_flops = float(fpt.get("training", 0.0)) * tokens
    attn_core_fwd = float(fpt.get("attention_core", 0.0)) * tokens
    lin_core_fwd = float(fpt.get("linear_attention_core", 0.0)) * tokens

    comps, entry, names = parse_module(hlo_text)
    mult = multipliers(comps, entry)
    frames_by_id = stack_frames(hlo_text)
    hbm = float(ceilings["hbm_gbps"]) * 1e9
    kernel_peak = float(ceilings.get("kernel_tflops", ceilings.get("gemm_default_tflops", SPEC_TFLOPS))) * 1e12

    events = _load_trace(profile_dir)
    pnames = {e.get("pid"): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {pid for pid, n in pnames.items() if "GPU" in n or "/device:" in n}
    measured: dict[str, float] = defaultdict(float)
    launches: dict[str, int] = defaultdict(int)
    kernel_name: dict[str, str] = {}
    intervals = []
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        args = e.get("args", {}) or {}
        op = args.get("hlo_op") or "(no hlo op)"
        measured[op] += float(e.get("dur", 0.0))
        launches[op] += 1
        name = e.get("name", "")
        if op not in kernel_name or (kernel_name[op].startswith("Mem") and not name.startswith("Mem")):
            kernel_name[op] = name  # the op's main kernel, not the memset/memcpy before it
        ts = float(e.get("ts", 0.0))
        intervals.append((ts, ts + float(e.get("dur", 0.0))))
    busy = 0.0
    end = None
    for s, t in sorted(intervals):
        if end is None or s > end:
            busy += (t - s)
            end = t
        elif t > end:
            busy += t - end
            end = t
    busy_ms = busy / steps / 1e3

    rows = []
    for op, us in measured.items():
        ins = names.get(op)
        ms = us / steps / 1e3
        if ins is None:
            rows.append(dict(op=op, kind="no HLO op", category="memcpy/memset/other", phase="?", site="", ms=ms,
                             flops=0.0, bytes=0.0, sol_ms=0.0, execs=launches[op] / steps, gemm=None,
                             kernel=kernel_name.get(op, "")[:70], shape=""))
            continue
        execs = mult.get(ins["computation"], 0.0) or launches[op] / steps
        md = re.search(r'op_name="([^"]*)"', ins["attrs"])
        op_name = md.group(1) if md else ""
        sid = re.search(r"stack_frame_id=(\d+)", ins["attrs"])
        frames = frames_by_id.get(int(sid.group(1)), []) if sid else []
        phase = phase_of(op_name, frames)
        site = _jla_site(frames)
        opcode, attrs = ins["opcode"], ins["attrs"]
        flops, nbytes, mkn = 0.0, 0.0, None
        target = (re.search(r'custom_call_target="([^"]+)"', attrs) or [None, ""])[1]
        if opcode == "custom-call" and target in _GEMM_TARGETS:
            flops, mkn = _gemm_custom_call(ins, names)
            out = shape_arrays(ins["shape"])
            nbytes = sum(shape_bytes(names[o]["shape"]) for o in ins["operands"] if o in names)
            nbytes += _dtype_bytes(out[0][0]) * math.prod(out[0][1]) if out else 0.0
            category = "GEMM"
        elif opcode == "custom-call":
            nbytes = sum(shape_bytes(names[o]["shape"]) for o in ins["operands"] if o in names) + shape_bytes(ins["shape"])
            category = "Mosaic GPU kernel" if "mosaic" in target else f"custom call {target}"
        elif opcode == "fusion":
            callee = _attr(attrs, "calls") or ""
            flops = computation_flops(callee, comps, names)
            nbytes = fusion_bytes(ins, comps, names)
            if flops > 0:
                category = "GEMM"
                dot = next((b for b in comps.get(callee, []) if b["opcode"] == "dot"), None)
                if dot is None:  # nested Triton GEMM: the dot lives one level down
                    for b in comps.get(callee, []):
                        inner = _attr(b["attrs"], "calls")
                        dot = next((c for c in comps.get(inner or "", []) if c["opcode"] == "dot"), None)
                        if dot:
                            break
                if dot is not None:
                    o = shape_arrays(dot["shape"])
                    lhs = names.get(dot["operands"][0])
                    if o and lhs:
                        ld = shape_arrays(lhs["shape"])[0][1]
                        k = math.prod(ld[d] for d in _dims_attr(dot["attrs"], "lhs_contracting_dims"))
                        mn = math.prod(o[0][1])
                        m = math.prod(ld) // max(1, k)
                        mkn = (m, k, mn // max(1, m))
            elif _only_convert(callee, comps):
                category = "casts (convert-only fusions)"
            elif _only_layout(callee, comps):
                category = "layout copies (transpose/copy/pad/concat fusions)"
            else:
                category = "elementwise/reduction fusions"
        elif opcode in _NOKERNEL:  # loop/branch bookkeeping copies (predicates, counters)
            category = "memcpy/memset/other"
        elif opcode in ("copy", "transpose"):
            nbytes = sum(shape_bytes(names[o]["shape"]) for o in ins["operands"] if o in names) + shape_bytes(ins["shape"])
            category = "layout copies (transpose/copy/pad/concat fusions)"
        else:
            nbytes = sum(shape_bytes(names[o]["shape"]) for o in ins["operands"] if o in names) + shape_bytes(ins["shape"])
            category = f"other ({opcode})"
        if phase == "opt" and category != "GEMM":
            category = "optimizer (AdamW, clipping)"
        gemm_peak = _gemm_tflops(ceilings, mkn) * 1e12 if category == "GEMM" else kernel_peak
        # an op that ran faster than the micro-benchmark ceiling is its own ceiling (L2 reuse, better tiles)
        sol = min(ms, max(flops * execs / gemm_peak if flops else 0.0, nbytes * execs / hbm) * 1e3)
        rows.append(dict(op=op, kind=opcode, category=category, phase=phase, site=site, ms=ms,
                         execs=execs, launches=launches[op] / steps, flops=flops * execs, bytes=nbytes * execs,
                         gemm=list(mkn) if mkn else None, sol_ms=sol, kernel=kernel_name.get(op, "")[:70],
                         shape=ins["shape"][:80]))

    # Mosaic GPU kernels: the analytic core FLOPs (softmax attention for the attention kernels, the
    # recurrence for the others), shared per phase among that group's kernels in proportion to bytes
    for phase, factor in (("fwd", 1.0), ("remat", 1.0), ("bwd", 2.0)):
        for is_attn, core in ((True, attn_core_fwd), (False, lin_core_fwd)):
            group = [r for r in rows if r["category"] == "Mosaic GPU kernel" and r["phase"] == phase
                     and ("mosaic_gpu/attention.py" in r.get("site", "")) == is_attn]
            gb = sum(r["bytes"] for r in group)
            for r in group:
                share = r["bytes"] / gb if gb else 1.0 / max(1, len(group))
                r["flops"] = core * factor * share
                r["sol_ms"] = min(r["ms"], max(r["flops"] / kernel_peak, r["bytes"] / hbm) * 1e3)
                r["attention"] = is_attn

    measured_ms = sum(r["ms"] for r in rows)
    cats: dict[str, dict] = {}
    for r in rows:
        c = cats.setdefault(r["category"], defaultdict(float))
        c["ms"] += r["ms"]
        c["sol_ms"] += r["sol_ms"]
        c["flops"] += r["flops"]
        c["bytes"] += r["bytes"]
        c[f"ms_{r['phase']}"] += r["ms"]
        c[f"sol_ms_{r['phase']}"] += r["sol_ms"]
    categories = {k: {kk: round(vv, 3) if "flops" not in kk and "bytes" not in kk else vv for kk, vv in v.items()}
                  for k, v in sorted(cats.items(), key=lambda kv: -kv[1]["ms"])}
    as_compiled = sum(r["sol_ms"] for r in rows)

    # ideal: no remat and no CE logits recompute; elementwise, casts and layout copies fused away;
    # GEMMs/kernels/optimizer at ceiling; attention kernels skip the tiles document masking empties
    fused_away = {"casts (convert-only fusions)", "layout copies (transpose/copy/pad/concat fusions)",
                  "elementwise/reduction fusions", "memcpy/memset/other"}
    recompute_ce = ce_logits_recompute(rows)
    for i in recompute_ce:
        rows[i]["ce_logits_recompute"] = True
    ideal_rows = [r for r in rows if r["phase"] != "remat" and r["category"] not in fused_away
                  and not r.get("ce_logits_recompute")]
    power16 = float(ceilings.get("power_scale_bf16", 1.0))

    def kernel_sol(r: dict, power: bool) -> float:
        """A Mosaic GPU kernel at its ceiling: attention executes ``attention_work_fraction`` of its
        counted FLOPs; power-capped, its tensor-core work runs at the sustained bf16 rate."""
        work = r["flops"] * (attention_work_fraction if r.get("attention") else 1.0)
        sol = max(work / (kernel_peak * (power16 if power else 1.0)), r["bytes"] / hbm) * 1e3
        return sol if power else min(sol, r["ms"])

    def ideal_with(fp8: bool, power: bool) -> float:
        """The ideal step with every GEMM at its (FP8) ceiling, optionally scaled down to the sustained
        power-capped rate of a back-to-back GEMM loop (``power_scale_bf16`` / ``power_scale_fp8``; the
        Mosaic GPU kernels' tensor-core work too). FP8 applies to every GEMM whose M, K and N are all
        >= 512 (the small ones stay bf16), with the quantization assumed fused into the producers'
        epilogues."""
        total = 0.0
        for r in ideal_rows:
            if r["category"] == "Mosaic GPU kernel":
                total += kernel_sol(r, power)
                continue
            if r["category"] != "GEMM" or not r.get("gemm"):
                total += r["sol_ms"]
                continue
            use8 = fp8 and min(r["gemm"]) >= 512
            peak = _gemm_tflops(ceilings, tuple(r["gemm"]), "gemm_fp8_tflops" if use8 else "gemm_bf16_tflops")
            scale = float(ceilings.get("power_scale_fp8" if use8 else "power_scale_bf16", 1.0)) if power else 1.0
            nbytes = r["bytes"] / 2 if use8 else r["bytes"]
            sol = max(r["flops"] / (peak * scale * 1e12), nbytes / hbm) * 1e3
            total += sol if (power or use8) else min(sol, r["ms"])
        return total

    ideal = ideal_with(fp8=False, power=False)
    ideal_fp8 = ideal_with(fp8=True, power=False)
    ideal_sustained = ideal_with(fp8=False, power=True)
    ideal_fp8_sustained = ideal_with(fp8=True, power=True)
    no_remat = sum(r["sol_ms"] for r in rows if r["phase"] != "remat")
    gemm_rows = [r for r in rows if r["category"] == "GEMM"]
    gemm_flops_needed = sum(r["flops"] for r in gemm_rows if r["phase"] != "remat")
    # tensor-core work the ideal step executes: its GEMMs and the kernels' core FLOPs (attention at the
    # executed fraction); the MFU bound if all of it ran at the sustained (power-capped) bf16 rate
    tensor_flops = sum(r["flops"] * (attention_work_fraction if r.get("attention") else 1.0)
                       for r in ideal_rows if r["category"] in ("GEMM", "Mosaic GPU kernel"))
    sustained_tflops = float(ceilings.get("kernel_tflops", SPEC_TFLOPS)) * power16

    gaps = []
    for cat, c in cats.items():
        for phase in ("fwd", "remat", "bwd", "opt", "?"):
            ms = c.get(f"ms_{phase}", 0.0)
            if ms <= 0:
                continue
            sol = c.get(f"sol_ms_{phase}", 0.0)
            removable = phase == "remat" or cat in fused_away
            gaps.append(dict(item=f"{cat} [{phase}]", measured_ms=round(ms, 2), sol_ms=round(sol, 2),
                             gap_ms=round(ms if removable else ms - sol, 2),
                             kind="should not exist (fuse / do not recompute)" if removable else
                             "distance from its own ceiling"))
    idle = max(0.0, wall_ms - busy_ms)
    gaps.append(dict(item="idle GPU (launch gaps, host, sync)", measured_ms=round(idle, 2), sol_ms=0.0,
                     gap_ms=round(idle, 2), kind="should not exist"))
    gaps.sort(key=lambda g: -g["gap_ms"])

    def mfu(ms: float) -> float:
        return round(100 * model_flops / (SPEC_TFLOPS * 1e12 * ms / 1e3), 1) if ms else 0.0

    every_op = [dict(op=r["op"], category=r["category"], phase=r["phase"], site=r.get("site", ""),
                     kernel=r.get("kernel", "")[:50], execs=r["execs"], gemm=r.get("gemm"),
                     gflops=round(r["flops"] / 1e9, 2), gbytes=round(r["bytes"] / 1e9, 3), ms=round(r["ms"], 3),
                     sol_ms=round(r["sol_ms"], 3))
                for r in sorted(rows, key=lambda r: -r["ms"]) if r["ms"] >= 0.1]
    top_rows = sorted(rows, key=lambda r: -(r["ms"] - r["sol_ms"]))[:top]
    for r in top_rows:
        r["ms"] = round(r["ms"], 3)
        r["sol_ms"] = round(r["sol_ms"], 3)
        r["achieved_tflops"] = round(r["flops"] / (r["ms"] / 1e3) / 1e12, 1) if r["flops"] and r["ms"] else None
        r["achieved_gbps"] = round(r["bytes"] / (r["ms"] / 1e3) / 1e9, 1) if r["ms"] else None
    gemm_shapes: dict[str, dict] = {}
    for r in gemm_rows:
        key = f"{r['phase']} {'x'.join(map(str, r['gemm'])) if r['gemm'] else '?'}"
        g = gemm_shapes.setdefault(key, defaultdict(float))
        g["ms"] += r["ms"]
        g["sol_ms"] += r["sol_ms"]
        g["flops"] += r["flops"]
        g["execs"] += r["execs"]
    gemm_table = sorted(({"gemm": k, "ms": round(v["ms"], 2), "sol_ms": round(v["sol_ms"], 2),
                          "execs_per_step": v["execs"],
                          "achieved_tflops": round(v["flops"] / (v["ms"] / 1e3) / 1e12, 1) if v["ms"] else None}
                         for k, v in gemm_shapes.items()), key=lambda g: -g["ms"])
    return {
        "arch": result.get("arch"), "tokens_per_step": tokens, "model_flops_per_step": model_flops,
        "wall_ms": round(wall_ms, 2), "busy_ms": round(busy_ms, 2), "kernel_ms_sum": round(measured_ms, 2),
        "measured_mfu_percent": mfu(wall_ms),
        "ceilings": ceilings,
        "sol_as_compiled_ms": round(as_compiled, 2), "sol_as_compiled_mfu_percent": mfu(as_compiled),
        "sol_no_remat_ms": round(no_remat, 2), "sol_no_remat_mfu_percent": mfu(no_remat),
        "sol_ideal_ms": round(ideal, 2), "sol_ideal_mfu_percent": mfu(ideal),
        "sol_ideal_power_capped_ms": round(ideal_sustained, 2),
        "sol_ideal_power_capped_mfu_percent": mfu(ideal_sustained),
        "sol_ideal_fp8_ms": round(ideal_fp8, 2), "sol_ideal_fp8_bf16_equivalent_mfu_percent": mfu(ideal_fp8),
        "sol_ideal_fp8_fp8_peak_utilization_percent": round(mfu(ideal_fp8) / 2, 1),
        "sol_ideal_fp8_power_capped_ms": round(ideal_fp8_sustained, 2),
        "sol_ideal_fp8_power_capped_bf16_equivalent_mfu_percent": mfu(ideal_fp8_sustained),
        "sol_ideal_fp8_power_capped_fp8_peak_utilization_percent": round(mfu(ideal_fp8_sustained) / 2, 1),
        "attention_work_fraction": attention_work_fraction,
        "ce_logits_recompute_ms": round(sum(rows[i]["ms"] for i in recompute_ce), 2),
        "ce_logits_recompute_sol_ms": round(sum(rows[i]["sol_ms"] for i in recompute_ce), 2),
        "tensor_flops_per_step_ideal": tensor_flops,
        "tensor_work_bound_power_capped_ms": round(tensor_flops / (sustained_tflops * 1e12) * 1e3, 2),
        "tensor_work_bound_power_capped_mfu_percent": mfu(tensor_flops / (sustained_tflops * 1e12) * 1e3),
        "gemm_only_at_ceiling_ms": round(sum(r["sol_ms"] for r in gemm_rows if r["phase"] != "remat"), 2),
        "gemm_flops_per_step_no_remat": gemm_flops_needed,
        "categories": categories, "gaps": gaps, "gemm_shapes": gemm_table[:30], "top_ops_by_gap": top_rows,
        "ops_over_0p1_ms": every_op,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile-dir", action="append", default=[])
    p.add_argument("--ceilings", default=None, help="JSON with gemm_bf16_tflops {MxKxN: TF}, gemm_default_tflops, "
                                                    "kernel_tflops, hbm_gbps")
    p.add_argument("--bench-json", action="append", default=[],
                   help="build the ceilings from these benchmark JSONs instead (written to --write-ceilings)")
    p.add_argument("--write-ceilings", default=None)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--attention-work-fraction", action="append", default=[], metavar="ARCH=F",
                   help="executed share of the counted causal attention FLOPs for that preset's profiled batch "
                        "(speed_of_light_prototypes.py --doc-stats); default 1.0")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    fractions = {}
    for item in args.attention_work_fraction:
        arch, _, value = item.partition("=")
        if not value or not 0.0 < float(value) <= 1.5:
            raise SystemExit(f"--attention-work-fraction wants ARCH=F with 0 < F <= 1.5, got {item!r}")
        fractions[arch] = float(value)
    if args.bench_json:
        ceilings = build_ceilings(args.bench_json)
        if args.write_ceilings:
            Path(args.write_ceilings).write_text(json.dumps(ceilings, indent=1))
    elif args.ceilings:
        ceilings = json.loads(Path(args.ceilings).read_text())
    else:
        raise SystemExit("need --ceilings or --bench-json")
    out = []
    for d in args.profile_dir:
        arch = json.loads((Path(d) / "profile_meta.json").read_text()).get("result", {}).get("arch")
        res = analyze(Path(d), ceilings, args.top, fractions.get(arch, 1.0))
        out.append(res)
        print(f"== {res['arch']}: wall {res['wall_ms']} ms (MFU {res['measured_mfu_percent']}%), busy {res['busy_ms']}; "
              f"SOL as compiled {res['sol_as_compiled_ms']} ms ({res['sol_as_compiled_mfu_percent']}%), "
              f"no remat {res['sol_no_remat_ms']} ms ({res['sol_no_remat_mfu_percent']}%), "
              f"ideal {res['sol_ideal_ms']} ms ({res['sol_ideal_mfu_percent']}%), GEMMs at ceiling "
              f"{res['gemm_only_at_ceiling_ms']} ms; ideal power-capped {res['sol_ideal_power_capped_ms']} ms "
              f"({res['sol_ideal_power_capped_mfu_percent']}%); ideal with FP8 GEMMs {res['sol_ideal_fp8_ms']} ms "
              f"({res['sol_ideal_fp8_bf16_equivalent_mfu_percent']}% bf16-equivalent; power-capped "
              f"{res['sol_ideal_fp8_power_capped_ms']} ms, "
              f"{res['sol_ideal_fp8_power_capped_bf16_equivalent_mfu_percent']}%); attention work fraction "
              f"{res['attention_work_fraction']}, CE logits recompute {res['ce_logits_recompute_ms']} ms; "
              f"tensor-work bound power-capped {res['tensor_work_bound_power_capped_mfu_percent']}%", flush=True)
        for g in res["gaps"][:16]:
            print(f"   {g['gap_ms']:9.2f} ms gap  {g['item'][:62]:62s} measured {g['measured_ms']:8.2f} sol {g['sol_ms']:8.2f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
