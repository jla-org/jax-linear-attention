# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time and profile one ``jla train`` step of an end-to-end config on synthetic tokens.

    python benchmarks/bench_end_to_end_step.py benchmarks/end_to_end/fw-mamba2-130m.toml \
        --steps 20 --profile-dir /path/to/trace --json result.json [--set key=value ...]

The step is built as ``jla train`` builds it on one device: the same config file,
``launch(cfg.runtime)`` (XLA tuning flags), the model builder, ``optimizer_from_config``,
``lm_loss`` and ``build_train_step``. The batch is uniform random token ids of the
config's shape (``[grad_accum, micro_batch, seq_len]``), so no data pipeline runs.
The result is the median ms/step and tok/s over ``--steps`` timed steps after warm-up.

With ``--profile-dir`` the script also records a profiler trace of ``--profile-steps``
steps (Perfetto JSON), maps every device kernel to its HLO instruction in the compiled
step and from there to the JAX source line that produced it (inline ``source_file`` metadata,
or ``stack_frame_id`` resolved through the FileNames/FileLocations/StackFrames tables that
JAX 0.11 prints in the HLO), and prints the device time per category (Mosaic GPU kernels,
matmuls, short conv, norms, cross entropy, mixer glue, optimizer, other), the time per
source line and the top kernels. cuBLAS kernels launched inside command buffers carry no
HLO op in the trace and stay under "matmuls".
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", nargs="?")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--json", default=None)
    p.add_argument("--analyze-only", default=None, help="re-analyze a saved --profile-dir without running")
    p.add_argument("--ce-shape", action="append", default=[], help="with --analyze-only: CE-only HLO shape substrings")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Trace analysis (no JAX needed)

_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
_META_RE = re.compile(r'metadata=\{([^}]*)\}')
_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^ ]+)')
_CALLS_RE = re.compile(r"calls=%?([\w.\-]+)")
_COMP_RE = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")


_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
_ROW_RE = re.compile(r"^(\d+)\s+(.*)$")
_ITEM_RE = re.compile(r"(\w+)=(\d+)")


def stack_frames(hlo_text: str) -> dict[int, list[tuple[str, str, str]]]:
    """``stack_frame_id -> [(file, line, function), ...]``, innermost frame first.

    JAX 0.11 prints source locations as four tables after the module header instead of inline
    ``source_file``/``source_line``: FileNames and FunctionNames (``id "string"``), FileLocations
    (``id {file_name_id= function_name_id= line= ...}``) and StackFrames
    (``id {file_location_id= parent_frame_id=}``). The printed ``parent_frame_id`` is the parent's
    id plus one (frame 1, the outermost, prints 1 = no parent), so the parent is ``printed - 1``.
    """
    tables: dict[str, dict[int, str]] = {t: {} for t in _TABLES}
    section = None
    for line in hlo_text.splitlines():
        s = line.strip()
        if s in _TABLES:
            section = s
            continue
        if section is None:
            continue
        m = _ROW_RE.match(s)
        if not m:
            section = None
            continue
        tables[section][int(m.group(1))] = m.group(2)
    files = {k: v.strip('"') for k, v in tables["FileNames"].items()}
    funcs = {k: v.strip('"') for k, v in tables["FunctionNames"].items()}
    locs = {k: {a: int(b) for a, b in _ITEM_RE.findall(v)} for k, v in tables["FileLocations"].items()}
    frames = {k: {a: int(b) for a, b in _ITEM_RE.findall(v)} for k, v in tables["StackFrames"].items()}
    out: dict[int, list[tuple[str, str, str]]] = {}
    for fid in frames:
        chain, cur = [], fid
        while cur in frames and len(chain) < 256:
            loc = locs.get(frames[cur].get("file_location_id", -1), {})
            chain.append((files.get(loc.get("file_name_id", -1), ""), str(loc.get("line", "")),
                          funcs.get(loc.get("function_name_id", -1), "")))
            parent = frames[cur].get("parent_frame_id", 0) - 1
            if parent <= 0 or parent >= cur:  # 0 = no parent; ids only grow towards callees
                break
            cur = parent
        out[fid] = chain
    return out


def _source_from_frames(chain: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    """The innermost frame in this package or optax, else the innermost frame."""
    for frame in chain:
        if "/jla/" in frame[0] or "/optax/" in frame[0]:
            return frame
    return chain[0] if chain else ("", "", "")


def hlo_metadata(hlo_text: str) -> dict[str, dict[str, str]]:
    """``instruction name -> {op_name, source_file, source_line, opcode}`` from compiled HLO text.

    ``source_file``/``source_line`` come from inline metadata or from ``stack_frame_id``
    (see :func:`stack_frames`). A fusion without metadata of its own inherits the metadata of
    the first instruction with metadata inside its fused computation.
    """
    frames = stack_frames(hlo_text)
    meta: dict[str, dict[str, str]] = {}
    calls: dict[str, str] = {}
    first_in_comp: dict[str, dict[str, str]] = {}
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
        info: dict[str, str] = {"shape": rest.split(" ")[0]}
        opcode = re.search(r"\b([a-z][\w\-]*)\(", rest)
        info["opcode"] = opcode.group(1) if opcode else ""
        mm = _META_RE.search(rest)
        if mm:
            for k, v in _KV_RE.findall(mm.group(1)):
                info[k] = v.strip('"')
            if "source_file" not in info and info.get("stack_frame_id", "").isdigit():
                f, ln, fn = _source_from_frames(frames.get(int(info["stack_frame_id"]), []))
                if f:
                    info["source_file"], info["source_line"], info["function"] = f, ln, fn
        cm = _CALLS_RE.search(rest)
        if cm:
            calls[name] = cm.group(1)
        if "custom_call_target" in rest:
            t = re.search(r'custom_call_target="([^"]+)"', rest)
            if t:
                info["target"] = t.group(1)
        meta[name] = info
        if comp is not None and "op_name" in info and comp not in first_in_comp:
            first_in_comp[comp] = info
    for name, callee in calls.items():
        if "op_name" not in meta[name] and callee in first_in_comp:
            inherited = dict(first_in_comp[callee])
            inherited["opcode"] = meta[name].get("opcode", "")
            inherited["shape"] = meta[name].get("shape", "")
            meta[name] = inherited
    return meta


_GEMM_RE = re.compile(r"gemm|cublas|cutlass|nvjet|xmma|sm90_|ampere_|matmul|dot", re.I)


def categorize(kernel: str, info: dict[str, str] | None, ce_shapes: tuple[str, ...] = ()) -> str:
    """Coarse sink of one kernel. ``ce_shapes`` are substrings of HLO result shapes that only the
    LM-head cross entropy produces (``"N,tile"``, ``"f32[N]"``, ``"V,"``), used when the HLO has
    ``stack_frame_id`` metadata instead of source lines."""
    src = os.path.basename((info or {}).get("source_file", ""))
    op = (info or {}).get("op_name", "")
    target = (info or {}).get("target", "")
    lname = kernel.lower()
    if "mosaic" in target.lower() or "mosaic" in lname or "pallas" in op or "mosaic" in op:
        return "mosaic_gpu kernels"
    is_gemm = bool(_GEMM_RE.search(kernel)) or (info or {}).get("opcode") in ("dot", "convolution") or "__cublas" in target
    shape = (info or {}).get("shape", "")
    is_ce_shape = bool(ce_shapes) and any(c in shape for c in ce_shapes)
    if src == "cross_entropy.py" or (not src and is_gemm and is_ce_shape):
        return "cross entropy matmuls" if is_gemm else "cross entropy (softmax, scan glue)"
    if is_gemm:
        return "matmuls (projections, MLP)"
    if src == "conv.py":
        return "short conv1d"
    if src == "norm.py":
        return "norms"
    if src in ("optimizers.py", "transform.py", "alias.py", "_adam.py", "numerics.py", "update.py", "clipping.py", "base.py") or "optax" in (info or {}).get("source_file", ""):
        return "optimizer"
    if src in ("ssm.py", "linear_attention.py"):
        return "mixer glue (gates, layout, D-skip)"
    if src in ("mlp.py",):
        return "mlp elementwise"
    if src in ("step.py",):
        return "step glue (grad accumulate, norm)"
    if "copy" in lname or "transpose" in lname or (info or {}).get("opcode") in ("copy", "transpose"):
        return "copies/transposes"
    if src:
        return f"other ({src})"
    if "cond/branch" in op:  # the optimizer update runs under lax.cond (skip on non-finite steps)
        return "optimizer"
    if is_ce_shape:
        return "cross entropy (softmax, scan glue)"
    if "broadcast" in kernel and op.endswith("reshape"):
        return "broadcasts (e.g. groups -> heads)"
    if "transpose(jvp" in op or "rematted_computation" in op:
        return "backward + remat elementwise/reductions"
    if "jvp(" in op:
        return "forward elementwise/reductions"
    if kernel.startswith("memcpy"):
        return "copies/transposes"
    return "other (no metadata)"


def _load_trace(profile_dir: Path):
    files = sorted(profile_dir.rglob("perfetto_trace.json.gz")) or sorted(profile_dir.rglob("*.trace.json.gz"))
    if not files:
        raise FileNotFoundError(f"no perfetto_trace.json.gz under {profile_dir}")
    with gzip.open(files[-1], "rt") as f:
        value = json.load(f)
    return value if isinstance(value, list) else value.get("traceEvents", [])


def analyze(profile_dir: Path, hlo_text: str, steps: int, top: int, ce_shapes: tuple[str, ...] = ()) -> dict:
    events = _load_trace(profile_dir)
    pnames = {e.get("pid"): e.get("args", {}).get("name", "") for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {pid for pid, n in pnames.items() if "GPU" in n or "gpu" in n or "/device:" in n}
    meta = hlo_metadata(hlo_text)
    per_kernel = defaultdict(float)
    per_cat = defaultdict(float)
    per_source = defaultdict(float)
    kernel_info = {}
    total = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0))
        args = e.get("args", {}) or {}
        hlo_op = args.get("hlo_op") or args.get("long_name", "").split(" ")[0].lstrip("%") or ""
        name = e.get("name", "")
        if not hlo_op or hlo_op.startswith("command_buffer"):  # a CUDA graph: the kernel name is the HLO op
            dotted = re.sub(r"_(\d+)$", r".\1", name)  # kernel loop_add_fusion_3 = HLO loop_add_fusion.3
            hlo_op = dotted if dotted in meta else name
        info = meta.get(hlo_op) or meta.get(name)
        cat = categorize(name, info, ce_shapes)
        key = (hlo_op or name)
        per_kernel[key] += dur
        if info and info.get("source_file"):
            phase = "bwd" if ("transpose(jvp" in info.get("op_name", "") or "rematted_computation" in info.get("op_name", "")) else "fwd"
            per_source[f"{os.path.basename(info['source_file'])}:{info.get('source_line', '')} "
                       f"{info.get('function', '')} [{phase}] {cat}"] += dur
        kernel_info[key] = (name, cat, (info or {}).get("source_file", ""), (info or {}).get("source_line", ""),
                            (info or {}).get("shape", ""), (info or {}).get("op_name", ""))
        per_cat[cat] += dur
        total += dur
    us_per_step = {k: v / steps for k, v in per_cat.items()}
    rows = sorted(us_per_step.items(), key=lambda kv: -kv[1])
    top_rows = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:top]
    return {
        "device_ms_per_step": total / steps / 1e3,
        "categories_ms_per_step": {k: v / 1e3 for k, v in rows},
        "sources_ms_per_step": {k: v / steps / 1e3 for k, v in sorted(per_source.items(), key=lambda kv: -kv[1])[:top]},
        "top_ops": [
            {"hlo_op": k, "kernel": kernel_info[k][0], "category": kernel_info[k][1],
             "source": f"{os.path.basename(kernel_info[k][2])}:{kernel_info[k][3]}", "ms_per_step": v / steps / 1e3,
             "shape": kernel_info[k][4], "op_name": kernel_info[k][5].replace("jit(train_step)/", "")[-90:]}
            for k, v in top_rows
        ],
    }


def _print_breakdown(breakdown: dict, wall_ms: float | None) -> None:
    wall = "" if wall_ms is None else f" (wall {wall_ms:.2f} ms/step)"
    print(f"device time {breakdown['device_ms_per_step']:.2f} ms/step{wall}")
    for k, v in breakdown["categories_ms_per_step"].items():
        print(f"  {k:40s} {v:8.2f} ms  {100 * v / breakdown['device_ms_per_step']:5.1f}%")
    print("top source lines (file:line function [fwd|bwd] category):")
    for k, v in breakdown.get("sources_ms_per_step", {}).items():
        print(f"  {v:7.3f} ms  {k[:110]}")
    print("top ops:")
    for row in breakdown["top_ops"]:
        print(f"  {row['ms_per_step']:7.3f} ms  {row['category'][:26]:26s} {row['hlo_op'][:34]:34s} {row['shape'][:34]:34s} {row['op_name'][-70:]}")


# --------------------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.analyze_only:  # re-read a saved trace and HLO dump; no JAX
        pdir = Path(args.analyze_only)
        saved = json.loads((pdir / "profile_meta.json").read_text()) if (pdir / "profile_meta.json").exists() else {}
        steps = saved.get("profile_steps", args.profile_steps)
        ce_shapes = tuple(args.ce_shape) or tuple(saved.get("ce_shapes", ()))
        out = analyze(pdir, (pdir / "train_step.hlo.txt").read_text(), steps, args.top, ce_shapes)
        _print_breakdown(out, None)
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=2))
        return 0
    if not args.config:
        raise SystemExit("a config file is required unless --analyze-only is given")
    from jla.config import load_config
    from jla.launch_env import launch

    cfg = load_config(args.config, args.overrides)
    launch(cfg.runtime)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jla.models import build_model
    from jla.training.optimizers import optimizer_from_config
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

    shape = (train.grad_accum, train.micro_batch_per_replica, train.seq_len)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.model.vocab_size, size=(shape[0], shape[1], shape[2] + 1), dtype=np.int32)
    batch = {"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])}
    tokens = int(np.prod(shape))

    t0 = time.perf_counter()
    lowered = step.lower(model, opt_state, batch, key)
    compiled = lowered.compile()
    step.use_compiled(compiled)
    compile_s = time.perf_counter() - t0
    xla_compiled = compiled if hasattr(compiled, "as_text") else compiled.compiled  # equinox wraps it
    hlo_text = xla_compiled.as_text()
    try:
        mem = xla_compiled.memory_analysis()
        peak = getattr(mem, "temp_size_in_bytes", 0) + getattr(mem, "argument_size_in_bytes", 0)
    except Exception:  # memory analysis is optional on some backends
        peak = 0

    for _ in range(args.warmup):
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
    jax.block_until_ready(metrics)
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        jax.block_until_ready(metrics)
        times.append(time.perf_counter() - t)
    ms = statistics.median(times) * 1e3
    loss = float(metrics["loss"]) if "loss" in metrics else float("nan")
    result = {
        "config": args.config, "overrides": args.overrides, "shape": list(shape),
        "vocab_size": cfg.model.vocab_size,
        "compile_s": round(compile_s, 1), "ms_per_step_median": round(ms, 2),
        "ms_per_step_min": round(min(times) * 1e3, 2), "tok_per_s": round(tokens / (ms / 1e3)),
        "loss": loss, "device_bytes_estimate": int(peak), "steps": args.steps,
        "device": str(jax.devices()[0].device_kind),
    }
    print(json.dumps({k: result[k] for k in ("config", "shape", "ms_per_step_median", "tok_per_s", "loss", "compile_s")}))

    if args.profile_dir:
        pdir = Path(args.profile_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "train_step.hlo.txt").write_text(hlo_text)
        jax.block_until_ready(metrics)
        with jax.profiler.trace(str(pdir), create_perfetto_trace=True):
            for _ in range(args.profile_steps):
                model, opt_state, metrics, key = step(model, opt_state, batch, key)
            jax.block_until_ready(metrics)
        n_rows = shape[1] * shape[2]  # one micro batch, flattened for the LM head
        tile = min(train.ce_vocab_chunk, cfg.model.vocab_size)
        ce_shapes = (f"{n_rows},{tile}]", f"f32[{n_rows}]", f"[{cfg.model.vocab_size},")
        (pdir / "profile_meta.json").write_text(json.dumps({"profile_steps": args.profile_steps, "ce_shapes": ce_shapes}))
        breakdown = analyze(pdir, hlo_text, args.profile_steps, args.top, ce_shapes)
        result["profile"] = breakdown
        _print_breakdown(breakdown, ms)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
