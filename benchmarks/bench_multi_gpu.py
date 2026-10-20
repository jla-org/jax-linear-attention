# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU validation of the training stack (see ``benchmarks/multi_gpu.md``).

One process trains one mesh layout and writes a JSON record; ``compare`` reads the records:

    # one layout per process; a layout is "1gpu" or axis sizes joined by "-" (dp8, fsdp8, dp4-tp2, fsdp4-dp2)
    python benchmarks/bench_multi_gpu.py run --preset gdn-hybrid-125m --layout fsdp8 --steps 16 --save-at 8 \
        --out runs/mgpu/gdn/fsdp8 --json runs/mgpu/gdn/fsdp8.json
    # resume that run's step-8 checkpoint under another layout (the same LR schedule, 16 steps in all)
    python benchmarks/bench_multi_gpu.py run --preset gdn-hybrid-125m --layout dp8 --steps 16 \
        --resume-from runs/mgpu/gdn/fsdp8 --resume-step 8 --out runs/mgpu/gdn/resume-dp8 --json ...
    # N jax.distributed processes with one GPU each (the parent never touches a GPU)
    python benchmarks/bench_multi_gpu.py multiprocess --num-processes 8 --coordinator auto \
        --preset gdn-hybrid-125m --layout fsdp8 --steps 16 --out runs/mgpu/gdn/mp8 --json runs/mgpu/gdn/mp8.json
    python benchmarks/bench_multi_gpu.py compare --dir runs/mgpu --json summary.json   # CPU only
    python benchmarks/bench_multi_gpu.py collectives --json collectives.json           # NCCL bus bandwidth

Every run uses the same global batch whatever its layout (``--global-batch`` rows of ``--seq-len``),
synthetic data and a cosine schedule over ``--steps``. ``--record-kernels`` records the shapes the
Mosaic GPU kernels are traced with (the per-device shapes show that a kernel runs per shard).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_AXIS = re.compile(r"^(dp|fsdp|tp)(\d+)$")


def parse_layout(layout: str) -> dict[str, int]:
    """``"1gpu"`` -> one device; ``"fsdp4-dp2"`` -> ``{"dp": 2, "fsdp": 4, "tp": 1}``."""
    sizes = {"dp": 1, "fsdp": 1, "tp": 1}
    if layout == "1gpu":
        return sizes
    for part in layout.split("-"):
        match = _AXIS.match(part)
        if match is None or sizes[match.group(1)] != 1:
            raise ValueError(f"bad layout {layout!r}: expected '1gpu' or parts like dp8, fsdp4, tp2")
        sizes[match.group(1)] = int(match.group(2))
    return sizes


def device_count(layout: str) -> int:
    return math.prod(parse_layout(layout).values())


def model_config(args: argparse.Namespace) -> Any:
    from jla.models import get_preset

    model = get_preset(args.preset).replace(compute_dtype=args.compute_dtype)
    if args.sliding_window:
        model = model.replace(attention=dataclasses.replace(model.attention, sliding_window=args.sliding_window))
    if args.attention_impl:
        model = model.replace(attention=dataclasses.replace(model.attention, implementation=args.attention_impl))
    if args.linear_impl and model.linear_attention is not None:
        model = model.replace(linear_attention=dataclasses.replace(model.linear_attention, implementation=args.linear_impl))
    if args.ssm_impl and model.ssm is not None:
        model = model.replace(ssm=dataclasses.replace(model.ssm, implementation=args.ssm_impl))
    return model


def experiment_config(args: argparse.Namespace, out: Path, *, runtime: dict | None = None) -> Any:
    from jla.config import DataConfig, ExperimentConfig, MeshConfig, OptimConfig, RuntimeConfig, TrainConfig

    mesh = parse_layout(args.layout)
    replicas = mesh["dp"] * mesh["fsdp"]
    if args.global_batch % (replicas * args.grad_accum):
        raise ValueError(f"global batch {args.global_batch} is not divisible by {replicas} replicas x {args.grad_accum} microbatches")
    train = TrainConfig(
        max_steps=args.steps,
        seq_len=args.seq_len,
        micro_batch_per_replica=args.global_batch // (replicas * args.grad_accum),
        grad_accum=args.grad_accum,
        log_every=1,
        eval_every=0,
        final_eval=False,
        save_every=args.save_at,
        save_final=False,
        keep_checkpoints=0,
        output_dir=str(out),
        ce_compute_dtype="float32" if args.compute_dtype == "float32" else "bfloat16",
        z_loss=1e-4,
        checkpoint_format=args.checkpoint_format,
        resume="required" if args.resume_from else "never",
    )
    optim = OptimConfig(
        name="adamw", lr=args.lr, schedule="cosine", warmup_steps=args.warmup, schedule_steps=args.steps,
        weight_decay=0.1, b2=0.95, max_grad_norm=1.0,
    )
    # --doc-length N: packed documents (an EOS every N tokens; segment ids and positions from it).
    docs = {"eos_id": 0, "synthetic_doc_length": args.doc_length, "document_masking": True} if args.doc_length else {}
    data = DataConfig(source="synthetic", num_workers=0, synthetic_samples=65_536, **docs)
    return ExperimentConfig(
        model=model_config(args), name=f"mgpu-{args.preset}-{args.layout}", optim=optim, train=train, data=data,
        mesh=MeshConfig(**mesh), runtime=RuntimeConfig(**(runtime or {})),
    )


@contextlib.contextmanager
def record_kernel_shapes(seen: list[list[Any]]):
    """Record ``(kernel, q shape)`` for every trace of a Mosaic GPU kernel behind the sharding wrappers."""
    from jla.distributed import kernel_sharding

    original_kernel, original_layout = kernel_sharding.shard_kernel_call, kernel_sharding.shard_layout_call

    def kernel_call(fn, q, k, v, gates, options, **kwargs):
        def spy(q, k, v, *gates, **options):
            seen.append([kwargs.get("kind") or getattr(fn, "__name__", "kernel"), list(q.shape)])
            return fn(q, k, v, *gates, **options)

        return original_kernel(spy, q, k, v, gates, options, **kwargs)

    def layout_call(fn, operands, layouts, **kwargs):
        name = "attention" if "segment_ids" in operands and "trap" not in operands else "mamba3_fused"

        def spy(**arrays):
            seen.append([name, list(arrays["q"].shape)])
            return fn(**arrays)

        return original_layout(spy, operands, layouts, **kwargs)

    kernel_sharding.shard_kernel_call, kernel_sharding.shard_layout_call = kernel_call, layout_call
    try:
        yield
    finally:
        kernel_sharding.shard_kernel_call, kernel_sharding.shard_layout_call = original_kernel, original_layout


def _peak_memory_gib(devices: list[Any]) -> list[float]:
    peaks = []
    for device in devices:
        stats = device.memory_stats() or {}
        peaks.append(round(stats.get("peak_bytes_in_use", 0) / 2**30, 2))
    return peaks


def _summary(result: Any, skip: int) -> dict[str, Any]:
    history = result.history
    timed = [record for record in history if record["step"] > skip] or history
    return {
        "steps": [int(record["step"]) for record in history],
        "losses": [float(record["loss"]) for record in history],
        "grad_norms": [float(record["grad_norm"]) for record in history],
        "median_tokens_per_sec": statistics.median(float(record["tokens_per_sec"]) for record in timed),
        "median_step_ms": 1000.0 * statistics.median(float(record["step_time"]) for record in timed),
        "median_mfu_percent": statistics.median(float(record["mfu_percent"]) for record in timed),
    }


def _gpu_utilisation() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or shutil.which("nvidia-smi") is None:
        return []
    query = ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader", "-i", visible]
    proc = subprocess.run(query, capture_output=True, text=True, check=False)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _train(args: argparse.Namespace, config: Any, devices: list[Any]) -> tuple[Any, list[list[Any]], dict[str, Any]]:
    import jax

    from jla.training import trainer as trainer_module

    seen: list[list[Any]] = []
    collectives: dict[str, Any] = {}
    original_preflight = trainer_module.compile_preflight

    def preflight(name, fn, *inputs, **kwargs):  # the compiled train step's collectives, from its HLO
        report, compiled = original_preflight(name, fn, *inputs, **kwargs)
        if compiled is not None and name == "train_step":
            inner = compiled
            while not hasattr(inner, "as_text") and hasattr(inner, "compiled"):  # equinox wraps jax.stages.Compiled
                inner = inner.compiled
            try:
                collectives.update(collective_stats(inner.as_text()))
            except Exception as exc:  # telemetry only; never fail the run
                collectives["error"] = f"{type(exc).__name__}: {exc}"
        return report, compiled

    precision = jax.default_matmul_precision("highest") if args.compute_dtype == "float32" else contextlib.nullcontext()
    recording = record_kernel_shapes(seen) if args.record_kernels else contextlib.nullcontext()
    trainer_module.compile_preflight = preflight
    try:
        with precision, recording:
            result = trainer_module.Trainer(config, devices=devices).run()
    finally:
        trainer_module.compile_preflight = original_preflight
    unique = sorted({json.dumps(entry) for entry in seen})
    return result, [json.loads(entry) for entry in unique], collectives


_COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")
_SHAPE = re.compile(r"\b(pred|s8|u8|s16|u16|s32|u32|s64|u64|f16|bf16|f32|f64|f8e4m3fn|f8e5m2)\[([0-9,]*)\]")
_BYTES = {"pred": 1, "s8": 1, "u8": 1, "f8e4m3fn": 1, "f8e5m2": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2,
          "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8}


def collective_stats(hlo: str) -> dict[str, Any]:
    """Count and result bytes of every collective in an optimized HLO module, by op and dtype."""
    stats: dict[str, dict[str, float]] = {}
    for line in hlo.splitlines():
        if " = " not in line:
            continue
        rhs = line.split(" = ", 1)[1]
        match = next((m for name in _COLLECTIVES if (m := re.search(rf"\s({name})(-start)?\(", " " + rhs))), None)
        if match is None:
            continue
        op, start = match.group(1), match.group(2)
        shapes = _SHAPE.findall(rhs.split(op, 1)[0])
        if start and op == "all-gather":  # (operands..., results...)
            shapes = shapes[len(shapes) // 2:]
        for dtype, dims in shapes:  # combined collectives return a tuple
            entry = stats.setdefault(f"{op}:{dtype}", {"count": 0, "mib": 0.0})
            entry["count"] += 1
            entry["mib"] = round(entry["mib"] + _BYTES.get(dtype, 4) * math.prod(int(d) for d in dims.split(",") if d) / 2**20, 3)
    return stats


def _prepare_resume(args: argparse.Namespace, out: Path) -> None:
    """Copy one committed step of ``--resume-from``'s checkpoints into ``out`` (the run resumes from it)."""
    source = Path(args.resume_from) / "checkpoints" / f"step_{args.resume_step:08d}"
    if not (source / "manifest.json").exists():
        raise FileNotFoundError(f"no committed checkpoint at {source}")
    shutil.copytree(source, out / "checkpoints" / source.name)


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if args.resume_from:
        _prepare_resume(args, out)
    config = experiment_config(args, out, runtime={"xla_tuning_level": args.xla_tuning_level})
    from jla.launch_env import launch

    launch(config.runtime)  # XLA flags of the tuning tier, before JAX touches a device
    count = device_count(args.layout)
    devices = jax.devices()
    if len(devices) < count:
        raise RuntimeError(f"layout {args.layout} needs {count} devices; {len(devices)} visible")
    started = time.perf_counter()
    result, kernels, collectives = _train(args, config, devices[:count])
    report = {
        "mode": "run", "preset": args.preset, "layout": args.layout, "devices": count,
        "mesh": parse_layout(args.layout), "compute_dtype": args.compute_dtype, "seq_len": args.seq_len,
        "global_batch": args.global_batch, "sliding_window": args.sliding_window or None,
        "attention_impl": args.attention_impl or "auto", "linear_impl": args.linear_impl or "auto",
        "ssm_impl": args.ssm_impl or "auto", "resume_from_step": args.resume_step if args.resume_from else None,
        "resume_source": Path(args.resume_from).name.removesuffix(".run") if args.resume_from else None,
        "xla_tuning_level": args.xla_tuning_level,
        "grad_accum": args.grad_accum, "doc_length": args.doc_length or None,
        **_summary(result, args.skip), "wall_seconds": time.perf_counter() - started,
        "peak_memory_gib": _peak_memory_gib(devices[:count]), "kernel_traces": kernels,
        "collectives": collectives,
        "jax_version": jax.__version__, "device_kind": devices[0].device_kind,
    }
    print(f"[run] {args.preset} {args.layout}: losses {report['losses'][:2]} ... {report['losses'][-1]:.5f}, "
          f"{report['median_tokens_per_sec']:,.0f} tok/s, {report['wall_seconds']:.0f} s", flush=True)
    return report


def cmd_worker(args: argparse.Namespace) -> dict[str, Any]:
    """One process of ``multiprocess`` (started by the parent with one visible GPU)."""
    from jla.launch_env import launch

    distributed = {"distributed": True, "coordinator_address": args.coordinator, "num_processes": args.num_processes,
                   "process_id": args.process_id, "initialization_timeout": 300, "xla_tuning_level": args.xla_tuning_level}
    out = Path(args.out)
    config = experiment_config(args, out, runtime=distributed)
    launch(config.runtime)
    import jax

    if jax.process_count() != args.num_processes or len(jax.local_devices()) != 1:
        raise RuntimeError(f"expected {args.num_processes} processes with one GPU each; got "
                           f"{jax.process_count()} and {jax.local_devices()}")
    started = time.perf_counter()
    result, kernels, collectives = _train(args, config, jax.devices())
    report = {
        "mode": "worker", "preset": args.preset, "layout": args.layout, "process_index": jax.process_index(),
        "process_count": jax.process_count(), "devices": len(jax.devices()), **_summary(result, args.skip),
        "wall_seconds": time.perf_counter() - started, "peak_memory_gib": _peak_memory_gib(jax.local_devices()),
        "kernel_traces": kernels, "collectives": collectives,
    }
    if args.save_at:
        manifest = out / "checkpoints" / f"step_{args.save_at:08d}" / "manifest.json"
        report["checkpoint_manifest"] = json.loads(manifest.read_text()) if manifest.exists() else None
    jax.distributed.shutdown()
    return report


def _auto_coordinator() -> str:
    """``host:port`` of this host that accepts TCP: the outbound IPv4 address and a free ephemeral port.

    Some hosts refuse TCP to 127.0.0.1, or to the IPv4 address on ports below the ephemeral range (seen on
    one 8xH100 node), so the address and the port are both probed before the workers start.
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # no packet is sent; picks the outbound interface
        candidates = [probe.getsockname()[0], "127.0.0.1"]
    except OSError:
        candidates = ["127.0.0.1"]
    finally:
        probe.close()
    for host in candidates:
        server = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
        try:
            server.bind(("::" if ":" in host else "0.0.0.0", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.create_connection((host, port), timeout=2):
                return f"{host}:{port}"  # the port is free again once the probe closes
        except OSError:
            continue
        finally:
            server.close()
    raise RuntimeError("no local TCP address accepts connections; pass --coordinator HOST:PORT")


def cmd_multiprocess(args: argparse.Namespace) -> dict[str, Any]:
    """Parent: ``--num-processes`` one-GPU ``jax.distributed`` workers on this host (no JAX here)."""
    visible = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) < args.num_processes:
        raise RuntimeError(f"multiprocess needs {args.num_processes} visible GPUs, CUDA_VISIBLE_DEVICES={visible}")
    coordinator = _auto_coordinator() if args.coordinator in ("", "auto") else args.coordinator
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    forwarded = [f"--preset={args.preset}", f"--layout={args.layout}", f"--steps={args.steps}",
                 f"--seq-len={args.seq_len}", f"--global-batch={args.global_batch}", f"--grad-accum={args.grad_accum}", f"--compute-dtype={args.compute_dtype}",
                 f"--lr={args.lr}", f"--warmup={args.warmup}", f"--skip={args.skip}", f"--save-at={args.save_at}",
                 f"--checkpoint-format={args.checkpoint_format}", f"--sliding-window={args.sliding_window}",
                 f"--attention-impl={args.attention_impl}", f"--linear-impl={args.linear_impl}",
                 f"--ssm-impl={args.ssm_impl}", f"--xla-tuning-level={args.xla_tuning_level}",
                 f"--doc-length={args.doc_length}"] + (["--record-kernels"] if args.record_kernels else [])
    procs = []
    started = time.perf_counter()
    try:
        for rank in range(args.num_processes):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible[rank]}
            cmd = [sys.executable, __file__, "worker", f"--coordinator={coordinator}",
                   f"--num-processes={args.num_processes}", f"--process-id={rank}", f"--out={out / 'run'}",
                   f"--json={out / f'worker{rank}.json'}", *forwarded]
            log = open(out / f"worker{rank}.log", "w")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        deadline = time.monotonic() + args.worker_timeout
        while any(proc.poll() is None for proc, _ in procs) and time.monotonic() < deadline:
            if any(proc.poll() not in (None, 0) for proc, _ in procs):
                break  # one worker failed: the others would wait for it in a collective
            time.sleep(1.0)
    finally:
        for proc, log in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            log.close()
    codes = [proc.returncode for proc, _ in procs]
    if any(codes):
        tails = {rank: (out / f"worker{rank}.log").read_text()[-4000:] for rank in range(args.num_processes)}
        raise RuntimeError(f"worker exit codes {codes}: {json.dumps(tails, indent=1)}")
    workers = [json.loads((out / f"worker{rank}.json").read_text()) for rank in range(args.num_processes)]
    report = {
        "mode": "multiprocess", "preset": args.preset, "layout": args.layout, "num_processes": args.num_processes,
        "coordinator_port_probe": args.coordinator in ("", "auto"),
        "compute_dtype": args.compute_dtype, "seq_len": args.seq_len, "global_batch": args.global_batch,
        "doc_length": args.doc_length or None,
        "workers_agree": all(worker["losses"] == workers[0]["losses"] for worker in workers),
        "wall_seconds": time.perf_counter() - started, **{key: workers[0][key] for key in (
            "steps", "losses", "grad_norms", "median_tokens_per_sec", "median_step_ms", "median_mfu_percent",
            "kernel_traces")},
        "peak_memory_gib": [worker["peak_memory_gib"][0] for worker in workers],
        "checkpoint_manifest": {key: workers[0].get("checkpoint_manifest", {}).get(key)
                                for key in ("payload_format", "process_count")} if workers[0].get("checkpoint_manifest") else None,
    }
    return report


def _compare(reference: list[float], other: list[float], reference_norms: list[float] | None = None,
             other_norms: list[float] | None = None) -> dict[str, float]:
    """Loss differences per step and, given the gradient norms, the first step's relative norm difference.

    The first loss is a forward pass only; the first gradient norm is the step's backward pass (the later
    steps mix in reduction-order noise from every earlier update)."""
    diffs = [abs(a - b) for a, b in zip(reference, other, strict=True)]
    out = {
        "first_abs_loss_diff": diffs[0],
        "max_abs_loss_diff": max(diffs),
        "max_rel_loss_diff": max(d / abs(a) for d, a in zip(diffs, reference)),
        "final_abs_loss_diff": diffs[-1],
    }
    if reference_norms and other_norms:
        out["first_rel_grad_norm_diff"] = abs(other_norms[0] - reference_norms[0]) / abs(reference_norms[0])
    return out


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Loss agreement and throughput of every record under ``--dir`` against its preset's 1-GPU run."""
    records = [(path.parent.name, json.loads(path.read_text())) for path in sorted(Path(args.dir).glob("*/*.json"))]
    by_preset: dict[str, dict[str, Any]] = {}
    for group, record in records:  # one directory per model configuration
        if record.get("mode") not in ("run", "multiprocess"):
            continue
        tag = record["layout"] if record["mode"] == "run" else f"mp{record['num_processes']}-{record['layout']}"
        if record.get("grad_accum", 1) > 1:
            tag = f"{tag}-accum{record['grad_accum']}"
        if record.get("resume_from_step") is not None:  # e.g. resume-fsdp8-to-dp8
            tag = f"resume-{record.get('resume_source') or args.resume_source}-to-{tag}"
        by_preset.setdefault(group, {})[tag] = record
    summary: dict[str, Any] = {}
    for preset, runs in sorted(by_preset.items()):
        one = runs.get("1gpu")
        rows: dict[str, Any] = {}
        for tag, record in sorted(runs.items()):
            devices = record.get("devices") or record.get("num_processes") or 1
            row = {
                "devices": devices,
                "final_loss": record["losses"][-1],
                "median_tokens_per_sec": round(record["median_tokens_per_sec"]),
                "tokens_per_sec_per_gpu": round(record["median_tokens_per_sec"] / devices),
                "median_step_ms": round(record["median_step_ms"], 2),
                "median_mfu_percent": round(record["median_mfu_percent"], 2),
                "peak_memory_gib_max": max(record.get("peak_memory_gib") or [0.0]),
                "local_kernel_shapes": sorted({json.dumps(entry) for entry in record.get("kernel_traces", [])}),
            }
            if one is not None and tag != "1gpu" and not tag.startswith("resume-"):
                row["vs_1gpu"] = _compare(one["losses"], record["losses"], one.get("grad_norms"), record.get("grad_norms"))
                row["throughput_per_gpu_vs_1gpu"] = round(row["tokens_per_sec_per_gpu"] / one["median_tokens_per_sec"], 3)
            single = runs.get(record["layout"]) if record["mode"] == "multiprocess" else None
            if single is not None:  # N processes against one process with N devices, same layout
                row["vs_single_process"] = _compare(single["losses"], record["losses"], single.get("grad_norms"),
                                                    record.get("grad_norms"))
                row["workers_agree"] = record["workers_agree"]
            rows[tag] = row
        # Resumes: the next losses against the uninterrupted run that wrote the checkpoint.
        for tag, record in runs.items():
            if not tag.startswith("resume-"):
                continue
            source = runs.get(record.get("resume_source") or args.resume_source)
            if source is None:
                continue
            start = record["resume_from_step"]
            reference = [loss for step, loss in zip(source["steps"], source["losses"]) if step > start]
            norms = [norm for step, norm in zip(source["steps"], source.get("grad_norms") or []) if step > start]
            rows[tag]["vs_uninterrupted"] = _compare(reference, record["losses"], norms, record.get("grad_norms"))
            rows[tag]["resumed_steps"] = record["steps"]
        summary[preset] = rows
    report: dict[str, Any] = {"mode": "compare", "resume_source": args.resume_source, "presets": summary}
    if args.records:  # every record in full (per-step losses, kernel traces, collectives)
        report["records"] = by_preset
    if args.markdown:
        Path(args.markdown).write_text(_markdown(summary))
    return report


def _fmt(value: float) -> str:
    return "0" if value == 0 else f"{value:.1e}"


def _markdown(summary: dict[str, Any]) -> str:
    """Tables of the ``compare`` summary: layouts against 1 GPU, then resumes and multi-process runs."""
    lines = []
    for preset, rows in summary.items():
        lines += [f"### {preset}", "",
                  "| run | GPUs | final loss | first-step loss diff | first-step grad-norm rel diff | max abs (rel) loss diff | tokens/s | tokens/s per GPU | per GPU vs 1 GPU | step ms | MFU % | peak GiB |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        order = sorted(rows, key=lambda t: (t.startswith("resume-"), rows[t]["devices"], t.startswith("mp"), t))
        for tag in order:
            row = rows[tag]
            key = next((k for k in ("vs_single_process", "vs_uninterrupted", "vs_1gpu") if k in row), None)
            diff = row.get(key) if key else None
            ref = {"vs_single_process": "vs 1 process", "vs_uninterrupted": "vs the uninterrupted run", "vs_1gpu": "vs 1 GPU"}.get(key, "")
            diff_text = "-" if diff is None else f"{_fmt(diff['max_abs_loss_diff'])} ({_fmt(diff['max_rel_loss_diff'])}) {ref}"
            first = "-" if diff is None else _fmt(diff["first_abs_loss_diff"])
            norm = "-" if diff is None or "first_rel_grad_norm_diff" not in diff else _fmt(diff["first_rel_grad_norm_diff"])
            scale = row.get("throughput_per_gpu_vs_1gpu")
            lines.append(
                f"| {tag} | {row['devices']} | {row['final_loss']:.5f} | {first} | {norm} | {diff_text} | {row['median_tokens_per_sec']:,} | "
                f"{row['tokens_per_sec_per_gpu']:,} | {'-' if scale is None else f'{scale:.2f}'} | {row['median_step_ms']} | "
                f"{row['median_mfu_percent']} | {row['peak_memory_gib_max']} |")
        lines.append("")
    return "\n".join(lines)


def _time(fn: Any, *inputs: Any, calls: int = 20) -> float:
    import jax

    jax.block_until_ready(fn(*inputs))
    jax.block_until_ready(fn(*inputs))
    started = time.perf_counter()
    for _ in range(calls):
        result = fn(*inputs)
    jax.block_until_ready(result)
    return 1000.0 * (time.perf_counter() - started) / calls


def cmd_collectives(args: argparse.Namespace) -> dict[str, Any]:
    """Bus bandwidth of all-reduce, all-gather and reduce-scatter over every visible GPU (NCCL via XLA)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("x",))
    n = len(devices)
    results = {}
    for mib in (16, 256):
        elements = mib * 2**20 // 4
        x = jax.device_put(jnp.ones((n, elements), jnp.float32), NamedSharding(mesh, P("x")))
        ops = {
            "all_reduce": (lambda a: jax.lax.psum(a, "x"), P("x"), (n - 1) / n * 2),
            "all_gather": (lambda a: jax.lax.all_gather(a, "x", tiled=True), P(), (n - 1) / n),
            "reduce_scatter": (lambda a: jax.lax.psum_scatter(a, "x", scatter_dimension=1, tiled=True), P("x"), (n - 1) / n),
        }
        for name, (body, out_spec, factor) in ops.items():
            fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P("x"), out_specs=out_spec, check_vma=False))
            ms = _time(fn, x)
            # nccl-tests convention: bus bandwidth = factor * bytes / time; bytes = one device's full buffer
            size = elements * 4 * (n if name == "all_gather" else 1)
            results[f"{name}_{mib}MiB"] = {"ms": ms, "bus_GBps": factor * size / (ms * 1e-3) / 1e9}
            print(f"[collectives] {name} {mib} MiB: {ms:.3f} ms, bus {results[f'{name}_{mib}MiB']['bus_GBps']:.1f} GB/s", flush=True)
    return {"mode": "collectives", "devices": n, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("run", "worker", "multiprocess", "compare", "collectives"))
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--layout", default="1gpu")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--save-at", type=int, default=0, help="checkpoint every this many steps (0: never)")
    parser.add_argument("--resume-from", default="", help="a run directory whose checkpoint this run resumes")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1, help="microbatches per step (the global batch stays the same)")
    parser.add_argument("--compute-dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--sliding-window", type=int, default=0, help="override attention.sliding_window")
    parser.add_argument("--attention-impl", default="", help="override attention.implementation")
    parser.add_argument("--linear-impl", default="", help="override linear_attention.implementation")
    parser.add_argument("--ssm-impl", default="", help="override ssm.implementation")
    parser.add_argument("--checkpoint-format", default="auto", choices=("auto", "safetensors", "sharded", "orbax"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--skip", type=int, default=5, help="steps excluded from the throughput median")
    parser.add_argument("--record-kernels", action="store_true")
    parser.add_argument("--xla-tuning-level", type=int, default=0, help="runtime.xla_tuning_level (0 off, 1, 2)")
    parser.add_argument("--doc-length", type=int, default=0,
                        help="packed documents: an EOS (id 0) every this many tokens, with document masking (0: off)")
    parser.add_argument("--coordinator", default="auto", help="HOST:PORT of process 0, or auto (probe this host)")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--dir", default="runs/multi-gpu", help="compare: directory of <name>/<record>.json files")
    parser.add_argument("--resume-source", default="fsdp8", help="compare: the run whose checkpoint was resumed")
    parser.add_argument("--markdown", default="", help="compare: also write the tables to this file")
    parser.add_argument("--records", action="store_true", help="compare: include every record in the JSON")
    parser.add_argument("--out", default="runs/multi-gpu/run")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    commands = {"run": cmd_run, "worker": cmd_worker, "multiprocess": cmd_multiprocess, "compare": cmd_compare,
                "collectives": cmd_collectives}
    report = commands[args.mode](args)
    if args.mode in ("run", "multiprocess", "collectives"):
        report["gpus"] = _gpu_utilisation()
    if args.mode == "run":
        report["save_at"] = args.save_at
    text = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
    if args.mode != "worker":
        print("RESULT " + json.dumps(report)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
