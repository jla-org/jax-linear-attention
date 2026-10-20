# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the device time of one 1.3B training step goes (benchmarks/profile_1p3b.md).

    python benchmarks/profile_1p3b.py --arch kda-1p3b --data /path/to/fineweb_edu_3B \
        --profile-dir /path/to/trace --json results/profile_1p3b_kda-1p3b.json
    python benchmarks/profile_1p3b.py --analyze-only /path/to/trace --json out.json

The step is the 1.3B harness's (``benchmarks/bench_e2e_1p3b.py``): the config
``benchmarks/end_to_end/1p3b/<arch>.toml`` with the same overrides (``--impl kernels`` puts every mixer
on its Mosaic GPU kernel), one GPU, ``[1, micro_batch, seq_len]`` tokens. The batch is the first
global batch of the CIDX corpus that the harness's ``BatchLoader`` yields (seed 0, shuffled), with
``segment_ids``/``positions``/``loss_mask`` from EOS when the config masks documents, and it is
reused for every step. The step is built as ``bench_end_to_end_step.py`` builds it (``launch``,
the model builder, ``optimizer_from_config``, ``lm_loss``, ``build_train_step``).

The trace is recorded with XLA command buffers (CUDA graphs) off (``--xla_gpu_enable_command_buffer=``),
so that every kernel, cuBLAS GEMMs included, carries its HLO instruction; ``--keep-command-buffers``
keeps XLA's default. Every kernel is mapped to its HLO instruction and from there to the chain of JAX
source frames (``stack_frame_id``). A GEMM's role is read from the first frame outside
``layers/linear.py`` (the call site of the projection), the phase from the HLO ``op_name``:
``transpose(...)`` = backward, ``rematted_computation`` = the forward recomputed under
``model.remat``, otherwise forward. Wall time per step is the host-timed median of the timed steps;
busy time is the union of all kernel intervals on the GPU over the profiled steps, and the
difference is idle time (launch gaps, host work, synchronisation).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bench_end_to_end_step import _load_trace, stack_frames  # noqa: E402  (same directory)

NO_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", help="config name under benchmarks/end_to_end/1p3b/ or a .toml path")
    p.add_argument("--data", help="CIDX prefix (without .bin/.idx) of 4096-token packed records")
    p.add_argument("--impl", choices=("kernels", "chunked", "auto"), default="kernels")
    p.add_argument("--attention", choices=("mosaic_gpu", "cudnn", "xla", "auto"), default=None,
                   help="override the softmax-attention implementation (as bench_e2e_1p3b.py --attention)")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=6, help="timed steps (host-timed, block_until_ready each)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--keep-command-buffers", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--json", default=None)
    p.add_argument("--analyze-only", default=None, help="re-analyze a saved --profile-dir (no JAX needed)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-steps per optimizer step; the first batch is repeated in each micro-step")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# HLO metadata with whole frame chains

_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
_META_RE = re.compile(r"metadata=\{([^}]*)\}")
_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^ ]+)')
_CALLS_RE = re.compile(r"calls=%?([\w.\-]+)")
_COMP_RE = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
_GEMM_KERNEL_RE = re.compile(r"gemm|cublas|cutlass|nvjet|xmma|sm90_|sm80_|ampere_|matmul", re.I)


def hlo_instructions(hlo_text: str) -> dict[str, dict]:
    """``instruction -> {opcode, shape, op_name, target, frames}``; a fusion inherits the metadata of
    the first instruction with metadata in its fused computation (and remembers whether that
    computation contains a dot)."""
    frames = stack_frames(hlo_text)
    meta: dict[str, dict] = {}
    calls: dict[str, str] = {}
    first_in_comp: dict[str, dict] = {}
    comp_has_dot: dict[str, bool] = defaultdict(bool)
    comp = None
    for line in hlo_text.splitlines():
        m = _COMP_RE.match(line)
        if m and "=" not in line.split("{")[0]:
            comp = m.group(1)
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        opcode = re.search(r"\b([a-z][\w\-]*)\(", rest)
        info: dict = {"shape": rest.split(" ")[0], "opcode": opcode.group(1) if opcode else ""}
        mm = _META_RE.search(rest)
        if mm:
            for k, v in _KV_RE.findall(mm.group(1)):
                info[k] = v.strip('"')
            sid = info.get("stack_frame_id", "")
            if sid.isdigit():
                info["frames"] = frames.get(int(sid), [])
            elif "source_file" in info:  # inline metadata (older JAX): one frame
                info["frames"] = [(info["source_file"], info.get("source_line", ""), "")]
        t = re.search(r'custom_call_target="([^"]+)"', rest)
        if t:
            info["target"] = t.group(1)
        cm = _CALLS_RE.search(rest)
        if cm:
            calls[name] = cm.group(1)
        if comp is not None and info["opcode"] in ("dot", "convolution"):
            comp_has_dot[comp] = True
        meta[name] = info
        if comp is not None and "op_name" in info and comp not in first_in_comp:
            first_in_comp[comp] = info
    for name, callee in calls.items():
        own = meta[name]
        if comp_has_dot.get(callee):
            own["contains_dot"] = True
        if "op_name" not in own and callee in first_in_comp:
            inherited = dict(first_in_comp[callee])
            inherited.update({k: own[k] for k in ("opcode", "shape", "target", "contains_dot") if k in own})
            meta[name] = inherited
    return meta


# --------------------------------------------------------------------------------------
# Categories

_ROLE_ATTRS = [  # (regex on the attribute / call text, role)
    (r"\b(o_proj|out_proj)\b", "out projection"),
    (r"\b(g_proj|gate_proj)\b", "output-gate projection"),
    (r"\b(q_proj|k_proj|v_proj|r_proj|qkv\w*|zx_proj|in_proj|bc_proj|x_proj|w_proj)\b", "in/q/k/v projections"),
    (r"\b(a_proj|b_proj|dt_proj|scalar_proj|angle_proj|level_proj|gk_proj|f_proj|\w_lora)\b",
     "gate/beta/dt/LoRA projections"),
    (r"\bgate_up\b|self\.key\(|\bself\.up\(", "MLP up/gate"),
    (r"\bself\.down\(|self\.value\(|\bdown\b", "MLP down"),
]


def _source_line(path: str, line: str, cache: dict) -> str:
    if (path, line) not in cache:
        text = ""
        try:
            lines = cache.setdefault(path, Path(path).read_text().splitlines() if Path(path).exists() else [])
            n = int(line)
            text = lines[n - 1].strip() if 0 < n <= len(lines) else ""
        except (OSError, ValueError):
            text = ""
        cache[(path, line)] = text
    return cache[(path, line)]


def _jla_frames(info: dict) -> list[tuple[str, str, str]]:
    return [f for f in info.get("frames", []) if "/jla/" in f[0] or "/optax/" in f[0]]


def gemm_role(info: dict, cache: dict) -> tuple[str, str]:
    """(role, call site) of a GEMM from its frames: the first frames outside ``layers/linear.py``."""
    frames = _jla_frames(info)
    if any(os.path.basename(f[0]) == "cross_entropy.py" or "linear_cross_entropy" in f[0] for f in frames):
        f = next(f for f in frames if "cross_entropy" in f[0])
        return "LM head (cross entropy)", f"{os.path.basename(f[0])}:{f[1]}"
    outer = [f for f in frames if os.path.basename(f[0]) != "linear.py"]
    for f in outer[:3]:
        text = _source_line(f[0], f[1], cache)
        for pattern, role in _ROLE_ATTRS:
            if re.search(pattern, text):
                return role, f"{os.path.basename(f[0])}:{f[1]} {text[:80]}"
    if outer:
        f = outer[0]
        return "other GEMM", f"{os.path.basename(f[0])}:{f[1]} {_source_line(f[0], f[1], cache)[:80]}"
    return "other GEMM", "(no source)"


def phase_of(info: dict | None) -> str:
    op = (info or {}).get("op_name", "")
    if "rematted_computation" in op:  # the forward recomputed in the backward pass (model.remat)
        return "remat"
    if "transpose(" in op:
        return "bwd"
    return "fwd"


def classify(kernel: str, info: dict | None, cache: dict) -> tuple[str, str, str]:
    """(category, phase, detail) of one kernel."""
    info = info or {}
    phase = phase_of(info)
    target = info.get("target", "").lower()
    op = info.get("op_name", "")
    frames = _jla_frames(info)
    files = [os.path.basename(f[0]) for f in frames]
    funcs = " ".join(f[2] for f in frames[:4]).lower()
    src = f"{files[0]}:{frames[0][1]}" if frames else ""
    if any("optax" in f[0] or os.path.basename(f[0]) in ("optimizers.py",) for f in frames[:2]) or "cond/branch" in op:
        return "optimizer update", "opt", src
    lname = kernel.lower()
    if "mosaic" in target or "mosaic" in lname or "pallas" in op:
        mfiles = [f for f in frames if "/ops/mosaic_gpu/" in f[0]]
        mfile = os.path.basename(mfiles[0][0]) if mfiles else (files[0] if files else "?")
        if mfile == "attention.py" or "attention.py" in files[:2]:
            return "softmax attention (Mosaic GPU)", phase, mfile
        if "cross_entropy" in mfile:
            return "cross entropy (Mosaic GPU kernel)", phase, mfile
        if mfile == "ragged_dot.py":  # dropless MoE experts (jla.layers.moe with ragged_dot_impl="mosaic_gpu")
            return "MoE expert grouped GEMM (Mosaic GPU)", phase, mfile
        return "recurrent kernels (Mosaic GPU)", phase, f"{mfile}:{kernel[:60]}"
    if "cudnn" in target or "fmha" in lname or "flash" in lname:
        return "softmax attention (cuDNN/XLA)", phase, kernel[:60]
    is_gemm = (info.get("opcode") in ("dot", "convolution") or "cublas" in target or "gemm" in target
               or info.get("contains_dot") or bool(_GEMM_KERNEL_RE.search(kernel)))
    if is_gemm:
        role, site = gemm_role(info, cache)
        if "moe.py" in files:  # the MoE block: router, shared experts (SwiGLU), capacity-path experts
            return "GEMM: MoE (router, shared expert, XLA experts)", phase, site
        return f"GEMM: {role}", phase, site
    if "moe.py" in files[:3]:  # routing, top-k, sort, gather/scatter of the routes, activations, combine
        return "MoE routing, dispatch and activations (non-GEMM)", phase, src
    if "cross_entropy.py" in files[:2]:
        return "cross entropy (non-GEMM)", phase, src
    if kernel.startswith(("memcpy", "memset", "Memcpy", "Memset")):
        return "memcpy/memset", phase, kernel[:40]
    if "step.py" in files[:1] or (files and files[0] == "step.py"):
        return "step glue (loss, grad accumulate, clip)", phase, src
    if info.get("opcode") in ("copy", "transpose") or "copy" in lname or "transpose" in lname:
        return "layout copies/transposes", phase, src
    first = files[0] if files else ""
    if first == "linear.py":  # casts, biases, and small-N dots XLA rewrites into reductions (e.g. dt_proj)
        role, site = gemm_role(info, cache)
        return "projection fusions (casts, bias, small dots)", phase, f"{role}: {site}"
    if first == "norm.py" or "norm" in funcs and first in ("rwkv7.py",) and "group_norm" in funcs:
        return "norms", phase, src
    if first == "conv.py":
        return "short conv", phase, src
    if first == "rope.py" or "rotary" in funcs or "rope" in funcs:
        return "rotary", phase, src
    if "l2" in funcs or "l2norm" in op:
        return "l2norm", phase, src
    if "cumsum" in op or "cumsum" in lname or "reduce_window" in op:
        return "cumsums", phase, src
    if first in ("mlp.py",):
        return "MLP activation (SwiGLU etc.)", phase, src
    if first in ("ssm.py", "linear_attention.py", "rwkv7.py", "attention.py", "scalar_gate.py", "vector_gate_ops.py",
                 "mamba3.py", "mamba3_fused.py", "kda.py", "gdn2.py", "gated_delta_rule.py") or "/ops/" in (frames[0][0] if frames else ""):
        return "mixer glue (gates, activations, layout)", phase, src
    if first in ("lm.py", "embedding.py"):
        return "embedding/residual/final norm", phase, src
    if first:
        return f"other ({first})", phase, src
    return "other (no metadata)", phase, kernel[:40]


def _union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    start = None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def analyze(profile_dir: Path, hlo_text: str, steps: int, top: int) -> dict:
    events = _load_trace(profile_dir)
    pnames = {e.get("pid"): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {pid for pid, n in pnames.items() if "GPU" in n or "gpu" in n or "/device:" in n}
    meta = hlo_instructions(hlo_text)
    cache: dict = {}
    per_cat_phase: dict[tuple[str, str], float] = defaultdict(float)
    per_detail: dict[tuple[str, str, str], float] = defaultdict(float)
    per_op: dict[str, float] = defaultdict(float)
    op_rows: dict[str, tuple] = {}
    intervals = []
    unmatched = 0.0
    total = 0.0
    n_kernels = 0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0))
        ts = float(e.get("ts", 0.0))
        args = e.get("args", {}) or {}
        name = e.get("name", "")
        hlo_op = args.get("hlo_op") or args.get("long_name", "").split(" ")[0].lstrip("%") or ""
        if not hlo_op or hlo_op.startswith("command_buffer"):
            dotted = re.sub(r"_(\d+)$", r".\1", name)
            hlo_op = dotted if dotted in meta else name
        info = meta.get(hlo_op) or meta.get(name)
        if info is None:
            unmatched += dur
        cat, phase, detail = classify(name, info, cache)
        per_cat_phase[(cat, phase)] += dur
        per_detail[(cat, phase, detail)] += dur
        per_op[hlo_op] += dur
        op_rows[hlo_op] = (name, cat, phase, detail, (info or {}).get("shape", ""))
        intervals.append((ts, ts + dur))
        total += dur
        n_kernels += 1
    span = (max(e for _, e in intervals) - min(s for s, _ in intervals)) if intervals else 0.0
    busy = _union_us(intervals)
    cats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (cat, phase), v in per_cat_phase.items():
        cats[cat][phase] += v / steps / 1e3
        cats[cat]["total"] += v / steps / 1e3
    ordered = dict(sorted(((k, dict(v)) for k, v in cats.items()), key=lambda kv: -kv[1]["total"]))
    phases: dict[str, float] = defaultdict(float)
    for (cat, phase), v in per_cat_phase.items():
        phases[phase] += v / steps / 1e3
    details = [{"category": c, "phase": p, "detail": d, "ms_per_step": v / steps / 1e3}
               for (c, p, d), v in sorted(per_detail.items(), key=lambda kv: -kv[1])[: top * 3]]
    top_ops = [{"hlo_op": k, "kernel": op_rows[k][0][:80], "category": op_rows[k][1], "phase": op_rows[k][2],
                "detail": op_rows[k][3][:100], "shape": op_rows[k][4][:60], "ms_per_step": v / steps / 1e3}
               for k, v in sorted(per_op.items(), key=lambda kv: -kv[1])[:top]]
    return {
        "profile_steps": steps,
        "kernels_per_step": n_kernels / steps,
        "kernel_ms_per_step": total / steps / 1e3,
        "busy_ms_per_step": busy / steps / 1e3,
        "span_ms_per_step": span / steps / 1e3,
        "unmatched_kernel_ms_per_step": unmatched / steps / 1e3,
        "phases_ms_per_step": dict(phases),
        "categories_ms_per_step": ordered,
        "details": details,
        "top_ops": top_ops,
    }


def print_breakdown(out: dict, wall_ms: float | None) -> None:
    print(f"wall {wall_ms} ms/step; span {out['span_ms_per_step']:.1f}; busy {out['busy_ms_per_step']:.1f}; "
          f"kernel sum {out['kernel_ms_per_step']:.1f}; kernels/step {out['kernels_per_step']:.0f}; "
          f"unmatched {out['unmatched_kernel_ms_per_step']:.2f}")
    print("phases:", {k: round(v, 1) for k, v in out["phases_ms_per_step"].items()})
    for cat, v in out["categories_ms_per_step"].items():
        parts = " ".join(f"{p}={v.get(p, 0.0):7.2f}" for p in ("fwd", "remat", "bwd", "opt"))
        print(f"  {cat:46s} {v['total']:8.2f} ms  {parts}")
    print("details:")
    for row in out["details"][:60]:
        print(f"  {row['ms_per_step']:8.2f} ms  {row['category'][:38]:38s} {row['phase']:5s} {row['detail'][:110]}")
    print("top ops:")
    for row in out["top_ops"]:
        print(f"  {row['ms_per_step']:8.2f} ms  {row['category'][:30]:30s} {row['phase']:5s} {row['hlo_op'][:30]:30s} "
              f"{row['shape'][:36]:36s} {row['detail'][:60]}")


# --------------------------------------------------------------------------------------


def _first_batch(cfg, micro_batch: int, seq_len: int) -> dict:
    """The first global batch of the harness's loader (``BatchLoader``, seed ``train.seed``), as ``[1, B, T]``."""
    import numpy as np

    from jla.data.datasets import TokenWindowDataset
    from jla.data.loader import BatchLoader
    from jla.data.packing import DocumentFields

    dataset = TokenWindowDataset(cfg.data.paths[0], seq_len, exact_packed_records=cfg.data.exact_packed_records)
    transform = DocumentFields(cfg.data.eos_id) if cfg.data.document_masking else None
    loader = BatchLoader(dataset, micro_batch, seed=getattr(cfg.train, "seed", 0), shuffle=True, num_workers=0,
                         transform=transform)
    batch = next(loader)
    loader.close()
    return {k: np.asarray(v)[None] for k, v in batch.items()}


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.analyze_only:
        pdir = Path(args.analyze_only)
        saved = json.loads((pdir / "profile_meta.json").read_text())
        out = analyze(pdir, (pdir / "train_step.hlo.txt").read_text(), saved["profile_steps"], args.top)
        print_breakdown(out, saved.get("wall_ms_median"))
        if args.json:
            result = {**saved.get("result", {}), "profile": out}
            Path(args.json).write_text(json.dumps(result, indent=2))
        return 0
    if not (args.arch and args.data):
        raise SystemExit("--arch and --data are required unless --analyze-only is given")
    if not args.keep_command_buffers:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + NO_COMMAND_BUFFERS).strip()

    import bench_e2e_1p3b as harness

    from jla.config import load_config
    from jla.launch_env import launch

    config_path = harness._config_path(args.arch)
    overrides = [f'data.paths=["{args.data}"]', f"train.seq_len={args.seq_len}",
                 f"train.micro_batch_per_replica={args.micro_batch}", f"train.grad_accum={args.grad_accum}", "mesh.dp=1",
                 "mesh.fsdp=1", *args.overrides]
    cfg = load_config(str(config_path), overrides)
    mixer_impl, attention_impl = harness._IMPLS[args.impl]
    attention_impl = args.attention or attention_impl
    cfg = dataclasses.replace(cfg, model=harness._with_impl(cfg.model, mixer_impl, attention_impl))
    launch(cfg.runtime)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jla.models import build_model, parameter_count
    from jla.training.optimizers import optimizer_from_config
    from jla.training.performance import estimate_flops_per_token
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss

    train = cfg.train
    schedule = create_schedule(
        cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps,
        warmup_steps=cfg.optim.warmup_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
        stable_fraction=cfg.optim.stable_fraction, decay_type=cfg.optim.decay_type, d_model=cfg.model.d_model,
    )
    optimizer = optimizer_from_config(cfg.optim, schedule)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_jit(lambda k: build_model(cfg.model, key=k))(key)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = jax.jit(optimizer.init)(params)
    step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(train)), scan_unroll=train.scan_unroll)
    batch = {k: jnp.asarray(np.repeat(v, args.grad_accum, axis=0))
             for k, v in _first_batch(cfg, args.micro_batch, args.seq_len).items()}
    tokens = args.micro_batch * args.seq_len * args.grad_accum

    t0 = time.perf_counter()
    compiled = step.lower(model, opt_state, batch, key).compile()
    step.use_compiled(compiled)
    compile_s = time.perf_counter() - t0
    xla_compiled = compiled if hasattr(compiled, "as_text") else compiled.compiled
    hlo_text = xla_compiled.as_text()
    try:
        mem = xla_compiled.memory_analysis()
        estimate = int(getattr(mem, "temp_size_in_bytes", 0) + getattr(mem, "argument_size_in_bytes", 0))
    except Exception:  # optional on some backends
        estimate = 0
    for _ in range(args.warmup):
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
    jax.block_until_ready(metrics)
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        jax.block_until_ready(metrics)
        times.append(time.perf_counter() - t)
    wall = statistics.median(times) * 1e3
    flops = estimate_flops_per_token(cfg.model, args.seq_len)
    counts = parameter_count(cfg.model)
    stats = jax.devices()[0].memory_stats() or {}
    result = {
        "arch": config_path.stem, "impl": args.impl, "mixer_implementation": mixer_impl,
        "attention_implementation": attention_impl, "micro_batch": args.micro_batch, "seq_len": args.seq_len,
        "grad_accum": args.grad_accum,
        "document_masking": bool(cfg.data.document_masking), "remat": cfg.model.remat,
        "command_buffers": "xla default" if args.keep_command_buffers else "off",
        "compile_s": round(compile_s, 1), "ms_per_step_median": round(wall, 2),
        "ms_per_step_min": round(min(times) * 1e3, 2), "ms_per_step_all": [round(t * 1e3, 2) for t in times],
        "tok_per_s": round(tokens / (wall / 1e3)), "loss": float(metrics.get("loss", float("nan"))),
        "compiled_bytes_estimate": estimate, "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "parameters": counts.total, "flops_per_token": flops.as_dict(),
        "device": str(jax.devices()[0].device_kind), "jax_version": jax.__version__,
    }
    result["mfu_percent"] = round(100 * flops.training * result["tok_per_s"] / 989e12, 2)
    print("PROFILE_STEP " + json.dumps({k: result[k] for k in ("arch", "ms_per_step_median", "tok_per_s", "loss",
                                                                 "compile_s", "mfu_percent")}), flush=True)
    if args.profile_dir:
        pdir = Path(args.profile_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "train_step.hlo.txt").write_text(hlo_text)
        with jax.profiler.trace(str(pdir), create_perfetto_trace=True):
            for _ in range(args.profile_steps):
                model, opt_state, metrics, key = step(model, opt_state, batch, key)
            jax.block_until_ready(metrics)
        (pdir / "profile_meta.json").write_text(json.dumps({"profile_steps": args.profile_steps,
                                                            "wall_ms_median": wall, "result": result}))
        out = analyze(pdir, hlo_text, args.profile_steps, args.top)
        result["profile"] = out
        print_breakdown(out, wall)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
