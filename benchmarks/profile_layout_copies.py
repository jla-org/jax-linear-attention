# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layout traffic of one 1.3B training step: every transpose/copy/pad/concatenate/slice XLA runs.

    python benchmarks/profile_layout_copies.py --arch mamba2-1p3b --data /path/to/fineweb_edu_3B \
        --profile-dir /path/to/trace --json out.json [--set model.ssm.grouped_key_grads=true]
    python benchmarks/profile_layout_copies.py --analyze-only /path/to/trace --json out.json

The step is the 1.3B harness's (``benchmarks/bench_e2e_1p3b.py``): its config, ``--impl kernels``
(every mixer on its Mosaic GPU kernel), one GPU, the first ``[micro_batch, seq_len]`` batch of the
CIDX corpus with ``segment_ids``/``positions``/``loss_mask`` from EOS, reused for every step. XLA
command buffers are off while profiling, so every kernel carries its HLO instruction.

Every device kernel is mapped to its HLO instruction. A fusion is *pure data movement* when its
fused computation holds only ``transpose``, ``copy``, ``pad``, ``concatenate``, ``slice``,
``dynamic-slice``, ``dynamic-update-slice``, ``reshape``, ``bitcast``, ``broadcast`` and
``convert`` (plus parameters, constants and tuples); it is reported under its most significant
movement op. Fusions that compute and also transpose, pad or concatenate are reported as
``compute+<op>``. Each row carries the JAX source line (first frame inside ``jla``), the phase
(``fwd``, ``bwd`` = ``transpose(jvp(...))``, ``remat`` = the forward recomputed under
``model.remat``) and its ms per step.
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

from bench_end_to_end_step import _load_trace, hlo_metadata  # noqa: E402  (same directory)

NO_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
MOVEMENT = ("transpose", "concatenate", "pad", "dynamic-update-slice", "dynamic-slice", "slice", "copy",
            "broadcast", "convert", "reshape", "bitcast")  # most significant first
INERT = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast", "iota"}
_COMP_START = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
_INSTR = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
_OPCODE = re.compile(r"\b([a-z][\w\-]*)\(")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", help="config name under benchmarks/end_to_end/1p3b/ or a .toml path")
    p.add_argument("--data", help="CIDX prefix (without .bin/.idx) of 4096-token packed records")
    p.add_argument("--impl", choices=("kernels", "chunked", "auto"), default="kernels")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=6, help="host-timed steps (block_until_ready each)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--json", default=None)
    p.add_argument("--analyze-only", default=None, help="re-analyze a saved --profile-dir (no JAX needed)")
    return p.parse_args(argv)


def fused_opcodes(hlo_text: str) -> dict[str, list[str]]:
    """``computation name -> opcodes of its instructions``."""
    comps: dict[str, list[str]] = {}
    current = None
    for line in hlo_text.splitlines():
        m = _COMP_START.match(line)
        if m and "=" not in line.split("{")[0]:
            current = m.group(1)
            comps[current] = []
            continue
        if current is None:
            continue
        m = _INSTR.match(line)
        if m:
            op = _OPCODE.search(m.group(2).split(" metadata=")[0])
            if op:
                comps[current].append(op.group(1))
    return comps


def layout_class(info: dict | None, hlo_line: str, comps: dict[str, list[str]]) -> str | None:
    """``"<op>"`` for a pure data-movement kernel, ``"compute+<op>"`` for a computing fusion that also
    transposes/pads/concatenates, ``None`` otherwise."""
    opcode = (info or {}).get("opcode", "")
    if opcode in MOVEMENT:
        return opcode
    if opcode != "fusion":
        return None
    m = re.search(r"calls=%?([\w.\-]+)", hlo_line)
    ops = [o for o in comps.get(m.group(1), []) if o not in INERT] if m else []
    if not ops:
        return None
    moving = [o for o in MOVEMENT if o in ops]
    if all(o in MOVEMENT for o in ops):
        return moving[0] if moving else None
    strong = [o for o in ("transpose", "concatenate", "pad", "dynamic-update-slice") if o in ops]
    return f"compute+{strong[0]}" if strong else None


def phase_of(info: dict | None) -> str:
    op = (info or {}).get("op_name", "")
    if "rematted_computation" in op or "remat" in op.split("/")[-1]:
        return "remat"
    if "transpose(" in op:
        return "bwd"
    return "fwd"


def analyze(profile_dir: Path, hlo_text: str, steps: int, top: int) -> dict:
    events = _load_trace(profile_dir)
    pnames = {e.get("pid"): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu_pids = {pid for pid, n in pnames.items() if "GPU" in n or "gpu" in n or "/device:" in n}
    meta = hlo_metadata(hlo_text)
    comps = fused_opcodes(hlo_text)
    lines = {}
    for line in hlo_text.splitlines():
        m = _INSTR.match(line)
        if m:
            lines[m.group(1)] = m.group(2)
    total = 0.0
    per_class: dict[str, float] = defaultdict(float)
    per_row: dict[tuple, float] = defaultdict(float)
    per_file: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0))
        total += dur
        args = e.get("args", {}) or {}
        name = e.get("name", "")
        hlo_op = args.get("hlo_op") or args.get("long_name", "").split(" ")[0].lstrip("%") or ""
        if not hlo_op or hlo_op.startswith("command_buffer"):
            dotted = re.sub(r"_(\d+)$", r".\1", name)
            hlo_op = dotted if dotted in meta else name
        info = meta.get(hlo_op) or meta.get(name)
        cls = layout_class(info, lines.get(hlo_op, ""), comps)
        if cls is None:
            continue
        src = f"{os.path.basename((info or {}).get('source_file', '?'))}:{(info or {}).get('source_line', '')}"
        fn = (info or {}).get("function", "")
        phase = phase_of(info)
        per_class[cls] += dur
        per_file[os.path.basename((info or {}).get("source_file", "?"))] += dur
        per_row[(cls, phase, src, fn, (info or {}).get("shape", "")[:60], hlo_op)] += dur
    rows = sorted(per_row.items(), key=lambda kv: -kv[1])
    pure = sum(v for k, v in per_class.items() if not k.startswith("compute+"))
    return {
        "device_ms_per_step": total / steps / 1e3,
        "pure_movement_ms_per_step": pure / steps / 1e3,
        "compute_with_movement_ms_per_step": (sum(per_class.values()) - pure) / steps / 1e3,
        "by_class_ms_per_step": {k: v / steps / 1e3 for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])},
        "by_source_file_ms_per_step": {k: v / steps / 1e3 for k, v in sorted(per_file.items(), key=lambda kv: -kv[1])},
        "rows": [{"class": k[0], "phase": k[1], "source": k[2], "function": k[3], "shape": k[4], "hlo_op": k[5],
                  "ms_per_step": v / steps / 1e3} for k, v in rows[:top]],
    }


def print_report(out: dict, wall_ms: float | None) -> None:
    wall = "" if wall_ms is None else f", wall {wall_ms:.1f} ms/step"
    print(f"device busy {out['device_ms_per_step']:.1f} ms/step{wall}; pure data movement "
          f"{out['pure_movement_ms_per_step']:.2f} ms, computing fusions with movement "
          f"{out['compute_with_movement_ms_per_step']:.2f} ms")
    for k, v in out["by_class_ms_per_step"].items():
        print(f"  {k:28s} {v:8.2f} ms")
    for k, v in out["by_source_file_ms_per_step"].items():
        print(f"  file {k:23s} {v:8.2f} ms")
    for r in out["rows"]:
        print(f"  {r['ms_per_step']:7.3f} ms {r['class'][:22]:22s} {r['phase']:5s} {r['source'][:28]:28s} "
              f"{r['function'][:26]:26s} {r['shape'][:44]}")


def _first_batch(cfg, micro_batch: int, seq_len: int) -> dict:
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
        print_report(out, None)
        if args.json:
            Path(args.json).write_text(json.dumps({"profile": out}, indent=2))
        return 0
    if not (args.arch and args.data):
        raise SystemExit("--arch and --data are required unless --analyze-only is given")
    if args.profile_dir:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + NO_COMMAND_BUFFERS).strip()
    from bench_e2e_1p3b import _IMPLS, _config_path, _with_impl

    from jla.config import load_config
    from jla.launch_env import launch

    overrides = [f'data.paths=["{args.data}"]', f"train.seq_len={args.seq_len}",
                 f"train.micro_batch_per_replica={args.micro_batch}", "train.grad_accum=1", "mesh.dp=1",
                 "mesh.fsdp=1", *args.overrides]
    cfg = load_config(str(_config_path(args.arch)), overrides)
    mixer_impl, attention_impl = _IMPLS[args.impl]
    cfg = dataclasses.replace(cfg, model=_with_impl(cfg.model, mixer_impl, attention_impl))
    launch(cfg.runtime)
    if args.profile_dir:  # launch() sets the tuning flags; keep command buffers off for the trace
        flags = os.environ.get("XLA_FLAGS", "")
        if NO_COMMAND_BUFFERS not in flags:
            os.environ["XLA_FLAGS"] = (flags + " " + NO_COMMAND_BUFFERS).strip()

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jla.models import build_model
    from jla.training.optimizers import optimizer_from_config
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss

    train = cfg.train
    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps,
                               warmup_steps=cfg.optim.warmup_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                               stable_fraction=cfg.optim.stable_fraction, decay_type=cfg.optim.decay_type,
                               d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_jit(lambda k: build_model(cfg.model, key=k))(key)
    opt_state = jax.jit(optimizer.init)(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(train)), scan_unroll=train.scan_unroll)
    batch = {k: jnp.asarray(v) for k, v in _first_batch(cfg, args.micro_batch, args.seq_len).items()}
    tokens = args.micro_batch * args.seq_len

    t0 = time.perf_counter()
    compiled = step.lower(model, opt_state, batch, key).compile()
    step.use_compiled(compiled)
    compile_s = time.perf_counter() - t0
    xla_compiled = compiled if hasattr(compiled, "as_text") else compiled.compiled
    hlo_text = xla_compiled.as_text()
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
    stats = jax.local_devices()[0].memory_stats() or {}
    result = {
        "arch": args.arch, "impl": args.impl, "overrides": args.overrides, "micro_batch": args.micro_batch,
        "seq_len": args.seq_len, "compile_s": round(compile_s, 1), "ms_per_step_median": round(ms, 2),
        "ms_per_step_all": [round(t * 1e3, 2) for t in times], "tokens_per_sec": round(tokens / (ms / 1e3)),
        "loss": float(metrics["loss"]), "peak_gib": int(stats.get("peak_bytes_in_use", 0)) / 2**30,
        "command_buffers": not bool(args.profile_dir), "device": str(jax.devices()[0].device_kind),
        "jax_version": jax.__version__,
    }
    print("LAYOUT_RESULT " + json.dumps({k: result[k] for k in ("arch", "overrides", "ms_per_step_median",
                                                                 "tokens_per_sec", "loss", "compile_s")}))
    if args.profile_dir:
        pdir = Path(args.profile_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "train_step.hlo.txt").write_text(hlo_text)
        with jax.profiler.trace(str(pdir), create_perfetto_trace=True):
            for _ in range(args.profile_steps):
                model, opt_state, metrics, key = step(model, opt_state, batch, key)
            jax.block_until_ready(metrics)
        (pdir / "profile_meta.json").write_text(json.dumps({"profile_steps": args.profile_steps}))
        result["profile"] = analyze(pdir, hlo_text, args.profile_steps, args.top)
        print_report(result["profile"], ms)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
