# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dissect the compiled 1.3B training step on a ``(dp, fsdp)`` mesh (benchmarks/multi_gpu.md).

    # build and compile the step as `jla train` does, then analyse it (CPU: fake host devices)
    python benchmarks/fsdp_step_hlo.py build --arch gdn-hd128-1p3b --dp 1 --fsdp 8 --micro-batch 16 \
        --impl chunked --platform cpu --out results/fsdp_hlo/gdn-hd128-fsdp8
    # analyse an optimized HLO dump (XLA_FLAGS=--xla_dump_to=DIR --xla_dump_hlo_as_text) of a real run
    python benchmarks/fsdp_step_hlo.py analyze DIR/module_0123.jit_train_step.sm_9.0_gpu_after_optimizations.txt \
        --global-batch 128 --micro-batch 16 --seq-len 4096 --out results/fsdp_hlo/gdn-hd128-fsdp8-gpu

``build`` creates the mesh, the sharded model and optimizer state and the train step with the same
functions as :class:`jla.training.trainer.Trainer` (logical-axis rules, batch sharding, parameter and
optimizer-state pins), feeds random packed-document batches and lowers and compiles the step. It
writes the per-device optimized HLO and ``report.json``.

``analyze`` reads optimized HLO text and reports, per training step (while-loop trip counts
applied): every all-gather / reduce-scatter / all-reduce / all-to-all / collective-permute with its
dtype, operand bytes, group size and source line; the totals per kind and dtype; and every
materialized array or kernel operand whose per-device shape still carries the global batch
(``[B_global, T, ...]`` or ``B_global * T`` rows), i.e. work that is not split along the batch.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

_DTYPE_BYTES = {
    "pred": 1, "s4": 0.5, "u4": 0.5, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "s64": 8, "u64": 8,
    "bf16": 2, "f16": 2, "f32": 4, "f64": 8, "c64": 8, "c128": 16,
}
_SHAPE = re.compile(r"\b([a-z]+[0-9]*[a-z0-9]*)\[([0-9,]*)\]")
_COLLECTIVES = {
    "all-gather": "all-gather", "all-gather-start": "all-gather",
    "all-reduce": "all-reduce", "all-reduce-start": "all-reduce",
    "reduce-scatter": "reduce-scatter",
    "all-to-all": "all-to-all", "ragged-all-to-all": "all-to-all",
    "collective-permute": "collective-permute", "collective-permute-start": "collective-permute",
    "collective-broadcast": "collective-broadcast",
}
_CALLEE = re.compile(r"\b(body|condition|to_apply|calls|true_computation|false_computation)=%?([\w.\-]+)")
_BRANCHES = re.compile(r"\b(?:branch_computations|called_computations)=\{([^}]*)\}")
_TRIP = re.compile(r'"known_trip_count":\{"n":"(\d+)"\}')
_GROUPS_IOTA = re.compile(r"replica_groups=\[(\d+),(\d+)\]")
_GROUPS_LIST = re.compile(r"replica_groups=\{\{([0-9,]*)\}")
_PAIRS = re.compile(r"source_target_pairs=\{([^}]*)\}")
_OP_NAME = re.compile(r'op_name="([^"]*)"')
_SOURCE = re.compile(r'source_file="([^"]*)" source_line=(\d+)')


def _dtype_bytes(dtype: str) -> float:
    if dtype.startswith("f8"):
        return 1
    return _DTYPE_BYTES.get(dtype, 0)


def _shapes(text: str) -> list[tuple[str, tuple[int, ...]]]:
    out = []
    for dtype, dims in _SHAPE.findall(text):
        if dtype in _DTYPE_BYTES or dtype.startswith("f8"):
            out.append((dtype, tuple(int(d) for d in dims.split(",") if d)))
    return out


def _nbytes(shapes: list[tuple[str, tuple[int, ...]]]) -> float:
    return sum(_dtype_bytes(dtype) * math.prod(dims) for dtype, dims in shapes)


def _split_instruction(line: str):
    """``(name, shape_text, opcode, operand_text, attrs)`` of one HLO instruction line, or None."""
    stripped = line.strip()
    if stripped.startswith("ROOT "):
        stripped = stripped[5:]
    if " = " not in stripped:
        return None
    name, rest = stripped.split(" = ", 1)
    name = name.lstrip("%")
    if rest.startswith("("):
        depth = 0
        for i, ch in enumerate(rest):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                break
        shape, rest = rest[: i + 1], rest[i + 1 :].lstrip()
    else:
        if " " not in rest:
            return None
        shape, rest = rest.split(" ", 1)
    paren = rest.find("(")
    if paren <= 0:
        return None
    opcode = rest[:paren]
    if not re.fullmatch(r"[a-z][a-z0-9\-]*", opcode):
        return None
    depth = 0
    for j in range(paren, len(rest)):
        depth += rest[j] == "("
        depth -= rest[j] == ")"
        if depth == 0:
            break
    return name, shape, opcode, rest[paren + 1 : j], rest[j + 1 :]


def parse_hlo(text: str) -> dict:
    """Computations of an HLO module: ``{name: {"entry": bool, "instructions": [...]}}``."""
    comps: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped.endswith("{") and (stripped.startswith("%") or stripped.startswith("ENTRY")):
                head = stripped[6:] if stripped.startswith("ENTRY") else stripped
                name = head.split(" ", 1)[0].lstrip("%")
                current = {"entry": stripped.startswith("ENTRY"), "instructions": []}
                comps[name] = current
            continue
        if stripped == "}":
            current = None
            continue
        parts = _split_instruction(stripped)
        if parts is not None:
            name, shape, opcode, operands, attrs = parts
            current["instructions"].append(
                {"name": name, "shape": shape, "opcode": opcode, "operands": operands, "attrs": attrs}
            )
    return comps


def multiplicities(comps: dict) -> tuple[dict[str, int], set[str]]:
    """Executions of each computation per run of the entry (while trip counts multiplied), fusion bodies."""
    fused: set[str] = set()
    edges: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for cname, comp in comps.items():
        for ins in comp["instructions"]:
            trip = 1
            if ins["opcode"] == "while":
                hit = _TRIP.search(ins["attrs"])
                trip = int(hit.group(1)) if hit else 1
            for key, callee in _CALLEE.findall(ins["attrs"]):
                factor = trip if key in ("body", "condition") else 1
                edges[cname].append((callee, factor))
                if ins["opcode"] == "fusion" and key == "calls":
                    fused.add(callee)
            for group in _BRANCHES.findall(ins["attrs"]):
                for callee in re.findall(r"%?([\w.\-]+)", group):
                    edges[cname].append((callee, 1))
    mult: dict[str, int] = collections.defaultdict(int)
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:  # post-order DFS, then reverse = topological order
        if name in seen:
            return
        seen.add(name)
        for callee, _ in edges.get(name, ()):
            visit(callee)
        order.append(name)

    entries = [name for name, comp in comps.items() if comp["entry"]]
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    for name in entries:
        visit(name)
        mult[name] = 1
    for name in reversed(order):
        for callee, factor in edges.get(name, ()):
            mult[callee] += mult[name] * factor
    return dict(mult), fused


def _group_size(attrs: str, default: int) -> int:
    hit = _GROUPS_IOTA.search(attrs)
    if hit:
        return int(hit.group(2))
    hit = _GROUPS_LIST.search(attrs)
    if hit:
        return len([x for x in hit.group(1).split(",") if x])
    return default


def _where(attrs: str) -> str:
    name = _OP_NAME.search(attrs)
    source = _SOURCE.search(attrs)
    text = name.group(1) if name else ""
    if source:
        path = source.group(1)
        marker = "/src/jla/"
        path = path[path.index(marker) + 1 :] if marker in path else os.path.basename(path)
        text += f" @ {path}:{source.group(2)}"
    return text[-220:]


def analyze_text(text: str, *, global_batch: int, micro_batch: int, seq_len: int, devices: int) -> dict:
    comps = parse_hlo(text)
    mult, fused = multiplicities(comps)
    collectives = []
    batch_arrays = []
    tokens_global = global_batch * seq_len
    for cname, comp in comps.items():
        times = mult.get(cname, 0)
        if times == 0:
            continue
        for ins in comp["instructions"]:
            kind = _COLLECTIVES.get(ins["opcode"])
            if kind is not None:
                # Results: the instruction's shape (an async "-start" holds (operands..., results...)).
                result_shapes = _shapes(ins["shape"])
                if ins["opcode"] == "all-gather-start" and len(result_shapes) >= 2:
                    result_shapes = result_shapes[len(result_shapes) // 2 :]
                elif ins["opcode"] == "collective-permute-start" and len(result_shapes) >= 2:
                    result_shapes = result_shapes[:1]
                group = _group_size(ins["attrs"], devices)
                if kind == "collective-permute":
                    pairs = _PAIRS.search(ins["attrs"])
                    group = len(re.findall(r"\{", pairs.group(1))) if pairs else group
                out_bytes = _nbytes(result_shapes)
                in_bytes = {"all-gather": out_bytes / max(group, 1), "reduce-scatter": out_bytes * group}.get(kind, out_bytes)
                wire = {
                    "all-gather": out_bytes * (group - 1) / max(group, 1),
                    "reduce-scatter": in_bytes * (group - 1) / max(group, 1),
                    "all-reduce": 2 * in_bytes * (group - 1) / max(group, 1),
                    "all-to-all": in_bytes * (group - 1) / max(group, 1),
                }.get(kind, in_bytes)
                collectives.append({
                    "kind": kind,
                    "opcode": ins["opcode"],
                    "dtypes": sorted({d for d, _ in result_shapes}),
                    "result_shapes": [f"{d}[{','.join(map(str, s))}]" for d, s in result_shapes][:8],
                    "operand_bytes": in_bytes,
                    "result_bytes": out_bytes,
                    "group_size": group,
                    "per_step": times,
                    "wire_bytes_per_step": wire * times,
                    "computation": cname,
                    "where": _where(ins["attrs"]),
                })
            if cname in fused or global_batch <= micro_batch:
                continue
            shapes = _shapes(ins["shape"])
            if ins["opcode"] == "custom-call":
                shapes = shapes + _shapes(ins["operands"])
            for dtype, dims in shapes:
                carries = any(dims[i] == global_batch and dims[i + 1] == seq_len for i in range(len(dims) - 1))
                carries = carries or tokens_global in dims
                if carries:
                    batch_arrays.append({
                        "opcode": ins["opcode"],
                        "shape": f"{dtype}[{','.join(map(str, dims))}]",
                        "bytes": _dtype_bytes(dtype) * math.prod(dims),
                        "per_step": times,
                        "computation": cname,
                        "where": _where(ins["attrs"]),
                    })
    totals: dict[str, dict] = collections.defaultdict(lambda: {"count_per_step": 0, "operand_bytes_per_step": 0.0,
                                                               "wire_bytes_per_step": 0.0})
    for c in collectives:
        key = f"{c['kind']} {'/'.join(c['dtypes'])}"
        totals[key]["count_per_step"] += c["per_step"]
        totals[key]["operand_bytes_per_step"] += c["operand_bytes"] * c["per_step"]
        totals[key]["wire_bytes_per_step"] += c["wire_bytes_per_step"]
    collectives.sort(key=lambda c: c["wire_bytes_per_step"], reverse=True)
    batch_arrays.sort(key=lambda a: a["bytes"] * a["per_step"], reverse=True)
    loops = []
    for cname, comp in comps.items():
        for ins in comp["instructions"]:
            if ins["opcode"] == "while":
                hit = _TRIP.search(ins["attrs"])
                body = re.search(r"\bbody=%?([\w.\-]+)", ins["attrs"])
                loops.append({"computation": cname, "trip_count": int(hit.group(1)) if hit else None,
                              "body": body.group(1) if body else None, "per_step": mult.get(cname, 0)})
    return {
        "computations": len(comps),
        "collective_totals": {k: dict(v) for k, v in sorted(totals.items())},
        "wire_gib_per_step": sum(c["wire_bytes_per_step"] for c in collectives) / 2**30,
        "collectives": collectives[:80],
        "collective_count": len(collectives),
        "global_batch_arrays": batch_arrays[:60],
        "global_batch_array_count": len(batch_arrays),
        "while_loops": loops,
    }


def _print_summary(report: dict) -> None:
    print(f"wire GiB per step (ring estimate, one device): {report['wire_gib_per_step']:.2f}")
    for key, value in report["collective_totals"].items():
        print(f"  {key:32s} x{value['count_per_step']:6d}/step  operands {value['operand_bytes_per_step'] / 2**30:8.2f} GiB"
              f"  wire {value['wire_bytes_per_step'] / 2**30:8.2f} GiB")
    print("largest collectives (wire bytes per step):")
    for c in report["collectives"][:25]:
        print(f"  {c['kind']:18s} {','.join(c['result_shapes'][:3]):60s} g={c['group_size']} x{c['per_step']:4d}"
              f" {c['wire_bytes_per_step'] / 2**30:7.3f} GiB  {c['where'][-110:]}")
    print(f"arrays carrying the global batch: {report['global_batch_array_count']}")
    for a in report["global_batch_arrays"][:25]:
        print(f"  {a['opcode']:14s} {a['shape']:36s} x{a['per_step']:4d} {a['bytes'] / 2**20:9.1f} MiB  {a['where'][-110:]}")


# -- build: the trainer's step on random packed batches -------------------------------------------------------


def _random_batch(rng, grad_accum: int, batch: int, seq_len: int, vocab: int, eos_id: int, docs: int):
    import numpy as np

    from jla.data.packing import DocumentFields

    tokens = rng.integers(0, vocab - 1, size=(grad_accum * batch, seq_len + 1), dtype=np.int64)
    tokens[tokens == eos_id] = eos_id - 1
    for row in tokens:  # about ``docs`` documents per row, like the FineWeb-Edu records
        row[rng.choice(seq_len, size=max(docs - 1, 0), replace=False)] = eos_id
    sample = {"input_ids": tokens[:, :-1].astype(np.int32), "labels": tokens[:, 1:].astype(np.int32)}
    sample = DocumentFields(eos_id)(sample)
    return {k: np.asarray(v).reshape(grad_accum, batch, seq_len) for k, v in sample.items()}


def build(args) -> int:
    devices_wanted = args.dp * args.fsdp
    if args.platform == "cpu":
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count={devices_wanted}".strip()
        os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(HERE))
    from bench_e2e_1p3b import _IMPLS, _config_path, _with_impl

    from jla.config import load_config
    from jla.launch_env import launch

    overrides = [
        f"train.seq_len={args.seq_len}",
        f"train.micro_batch_per_replica={args.micro_batch}",
        f"train.grad_accum={args.grad_accum}",
        f"mesh.dp={args.dp}",
        f"mesh.fsdp={args.fsdp}",
        *args.overrides,
    ]
    cfg = load_config(str(_config_path(args.arch)), overrides)
    mixer_impl, attention_impl = _IMPLS[args.impl]
    import dataclasses

    model_cfg = _with_impl(cfg.model, mixer_impl, args.attention or attention_impl)
    if args.layers:  # the first ``layers`` blocks of the preset (scan_layers: the same loop body)
        layer_types = None if model_cfg.layer_types is None else model_cfg.layer_types[: args.layers]
        mlp_types = None if model_cfg.mlp_types is None else model_cfg.mlp_types[: args.layers]
        model_cfg = model_cfg.replace(n_layers=args.layers, layer_types=layer_types, mlp_types=mlp_types)
    cfg = dataclasses.replace(cfg, model=model_cfg)
    launch(cfg.runtime)

    import contextlib

    import equinox as eqx
    import jax
    import numpy as np

    from jla.distributed.kernel_sharding import use_rules
    from jla.distributed.mesh import build_mesh, plan_from_config
    from jla.distributed.sharding import grouped_batch_sharding, opt_state_shardings, resolve_rules
    from jla.training.optimizers import optimizer_from_config
    from jla.training.runtime import compile_preflight
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss
    from jla.training.trainer import Trainer, _replicated_key

    devices = jax.devices()[:devices_wanted]
    if len(devices) != devices_wanted:
        raise SystemExit(f"need {devices_wanted} devices, have {len(jax.devices())}")
    plan = plan_from_config(cfg.mesh, len(devices))
    mesh = build_mesh(plan, devices, topology_aware=cfg.mesh.topology_aware)
    rules = resolve_rules(cfg.mesh.rules)
    trainer = Trainer(cfg, devices=devices)
    init_key, train_key = jax.random.split(jax.random.key(cfg.train.seed))
    started = time.perf_counter()
    model = trainer._init_model(mesh, rules, init_key)
    params = eqx.filter(model, eqx.is_inexact_array)
    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps,
                               warmup_steps=cfg.optim.warmup_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                               stable_fraction=cfg.optim.stable_fraction, decay_type=cfg.optim.decay_type,
                               d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    param_tree = jax.tree.map(lambda leaf: leaf.sharding, params)
    opt_shardings = opt_state_shardings(jax.eval_shape(optimizer.init, params), params, param_tree, mesh)
    opt_state = jax.jit(optimizer.init, out_shardings=opt_shardings)(params)
    rng = _replicated_key(train_key, mesh)
    batch_sharding = grouped_batch_sharding(mesh, rules)
    global_batch = cfg.train.micro_batch_per_replica * plan.data_parallel_size
    host = _random_batch(np.random.default_rng(0), cfg.train.grad_accum, global_batch, cfg.train.seq_len,
                         cfg.model.vocab_size, cfg.data.eos_id or cfg.model.vocab_size - 1, args.docs)
    batch = {k: jax.make_array_from_callback(v.shape, batch_sharding, lambda idx, v=v: v[idx]) for k, v in host.items()}
    step_fn = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(cfg.train)), batch_sharding=batch_sharding,
                               param_shardings=param_tree, opt_shardings=opt_shardings, scan_unroll=cfg.train.scan_unroll)
    setup_s = time.perf_counter() - started
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        stack.enter_context(jax.set_mesh(mesh))
        stack.enter_context(use_rules(rules))
        report, compiled = compile_preflight("train_step", step_fn, model, opt_state, batch, rng)
    if compiled is None:
        print(json.dumps({k: report.get(k) for k in ("status", "error")}, indent=2))
        (out / "report.json").write_text(json.dumps({"status": report["status"], "error": report.get("error")}, indent=2))
        return 1
    text = getattr(compiled, "compiled", compiled).as_text()  # equinox's Compiled wraps jax.stages.Compiled
    (out / "step.hlo.txt").write_text(text)
    analysis = analyze_text(text, global_batch=global_batch, micro_batch=cfg.train.micro_batch_per_replica,
                            seq_len=cfg.train.seq_len, devices=len(devices))
    cost = report.get("cost_analysis") or {}
    result = {
        "arch": args.arch, "impl": args.impl, "platform": args.platform, "mesh": plan.axis_sizes,
        "layers": cfg.model.n_layers, "micro_batch": cfg.train.micro_batch_per_replica, "global_batch": global_batch,
        "seq_len": cfg.train.seq_len, "grad_accum": cfg.train.grad_accum, "overrides": args.overrides,
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "setup_seconds": setup_s,
        "compile_seconds": report.get("compile_seconds"), "memory": report.get("memory"),
        "flops_per_device": cost.get("flops") if isinstance(cost, dict) else None,
        "bytes_accessed_per_device": report.get("cost_analysis_bytes_accessed"),
        "jax_version": jax.__version__, **analysis,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"compiled in {report.get('compile_seconds'):.1f}s; memory {json.dumps(report.get('memory'))}")
    print(f"flops per device {result['flops_per_device']}")
    _print_summary(result)
    return 0


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(a, b) for a, b in merged]


def _length(intervals: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in intervals)


def _intersection(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def trace_overlap(path: str) -> dict:
    """Per GPU of a ``jax.profiler`` trace (``*.xplane.pb``): kernel busy time, NCCL time, and how much of
    the NCCL time no compute kernel overlaps (exposed communication)."""
    from jax.profiler import ProfileData

    files = sorted(Path(path).rglob("*.xplane.pb"))
    if not files:
        raise SystemExit(f"no *.xplane.pb under {path}")
    data = ProfileData.from_file(str(files[-1]))
    devices = []
    for plane in data.planes:
        if not plane.name.startswith("/device:GPU:"):
            continue
        comm: list[tuple[int, int]] = []
        comp: list[tuple[int, int]] = []
        comm_by_name: dict[str, int] = collections.defaultdict(int)
        for line in plane.lines:
            if not line.name.startswith("Stream"):
                continue  # "XLA Modules" / "XLA Ops" / "Steps" lines repeat the stream events
            for event in line.events:
                start = int(event.start_ns)
                interval = (start, start + int(event.duration_ns))
                name = event.name
                if "nccl" in name.lower():
                    comm.append(interval)
                    comm_by_name[name.split("_", 2)[0][:40]] += interval[1] - interval[0]
                elif not name.startswith("Memcpy") and not name.startswith("Memset"):
                    comp.append(interval)
        if not comp and not comm:
            continue
        comm_u, comp_u = _union(comm), _union(comp)
        everything = _union(comm_u + comp_u)
        window = everything[-1][1] - everything[0][0]
        overlapped = _intersection(comm_u, comp_u)
        devices.append({
            "device": plane.name,
            "window_ms": window / 1e6,
            "compute_busy_ms": _length(comp_u) / 1e6,
            "nccl_ms": _length(comm_u) / 1e6,
            "nccl_overlapped_ms": overlapped / 1e6,
            "nccl_exposed_ms": (_length(comm_u) - overlapped) / 1e6,
            "idle_ms": (window - _length(everything)) / 1e6,
            "nccl_kernels": len(comm),
        })
    return {"trace": files[-1].name, "devices": devices}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("trace", help="compute / NCCL overlap per GPU of a jax.profiler trace directory")
    t.add_argument("profile_dir")
    t.add_argument("--out", required=True)
    b = sub.add_parser("build", help="build and compile the trainer's step, then analyse it")
    b.add_argument("--arch", required=True)
    b.add_argument("--impl", choices=("kernels", "chunked", "auto"), default="kernels")
    b.add_argument("--attention", choices=("mosaic_gpu", "cudnn", "xla", "auto"), default=None)
    b.add_argument("--dp", type=int, default=1)
    b.add_argument("--fsdp", type=int, default=8)
    b.add_argument("--micro-batch", type=int, default=16)
    b.add_argument("--grad-accum", type=int, default=1)
    b.add_argument("--seq-len", type=int, default=4096)
    b.add_argument("--layers", type=int, default=0, help="override n_layers (0: the preset's)")
    b.add_argument("--docs", type=int, default=4, help="documents per packed row of the random batch")
    b.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    b.add_argument("--set", dest="overrides", action="append", default=[], help="config override a.b=value")
    b.add_argument("--out", required=True)
    a = sub.add_parser("analyze", help="analyse an optimized HLO text dump")
    a.add_argument("hlo")
    a.add_argument("--global-batch", type=int, required=True)
    a.add_argument("--micro-batch", type=int, required=True)
    a.add_argument("--seq-len", type=int, default=4096)
    a.add_argument("--devices", type=int, default=8)
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.mode == "build":
        return build(args)
    if args.mode == "trace":
        report = trace_overlap(args.profile_dir)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace_overlap.json").write_text(json.dumps(report, indent=2))
        for d in report["devices"]:
            print(f"{d['device']}: window {d['window_ms']:.1f} ms, compute {d['compute_busy_ms']:.1f}, NCCL {d['nccl_ms']:.1f}"
                  f" (exposed {d['nccl_exposed_ms']:.1f}), idle {d['idle_ms']:.1f}, {d['nccl_kernels']} NCCL kernels")
        return 0
    text = Path(args.hlo).read_text()
    report = analyze_text(text, global_batch=args.global_batch, micro_batch=args.micro_batch, seq_len=args.seq_len,
                          devices=args.devices)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"hlo": Path(args.hlo).name, **report}, indent=2))
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
