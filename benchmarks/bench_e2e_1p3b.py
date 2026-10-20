# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end training throughput of the 1.3B presets on real data (benchmarks/end_to_end_1p3b.md).

    python benchmarks/bench_e2e_1p3b.py --arch kda-1p3b --data /path/to/fineweb_edu_3B \
        --impl kernels --json results/kda-1p3b-1gpu.json

``--micro-batch`` / ``--grad-accum`` override the config's split (default: the config's own).

The run is ``jla train`` itself: the config ``benchmarks/end_to_end/1p3b/<arch>.toml`` (or a
path) goes through ``load_config`` with the overrides below and :class:`jla.training.trainer.Trainer`
builds the mesh, the sharded model and optimizer state, the CIDX data pipeline (pre-packed
4096-token records, ``segment_ids``/``positions`` from EOS for document masking, EOS targets
out of the loss) and the compiled step. Every step is logged (``train.log_every = 1``), so each
step's wall time ends when its metrics reach the host; checkpoints and evaluation are off. The
devices then wait while the host logs and dispatches the next step (the 8-GPU transformer's GPUs are
idle 157 ms per step, benchmarks/multi_gpu.md) unless ``train.dispatch_ahead`` is on (the
transformer-1p3b TOML turns it on with ``train.update_guard = "select"``, without which it gains nothing on
several GPUs). With that option the last step's metrics are read without the lag of the others, so its time comes
out short and the summary leaves it out (``_summary(drop_last=True)``).
Compilation happens before step 1 (compile preflight) and is reported separately; steps
1..``--skip`` are warm-up and the remaining ``--steps - --skip`` steps are timed.

``--impl kernels`` runs every mixer on its Mosaic GPU kernel (linear attention, Mamba and the
FA3-style softmax attention: ``implementation = "mosaic_gpu"``; a shape a kernel does not support
fails loudly). ``--impl chunked`` runs the plain-JAX chunked linear-attention / SSM ops with
cuDNN attention (``--attention`` overrides the attention implementation in either mode).
``--impl auto`` keeps the configs' ``auto`` dispatch.

Reported (also per step in the JSON): tokens/s, tokens/s/GPU, MFU from
``jla.training.performance.estimate_flops_per_token`` (dense bf16 peak of the device), peak
device memory (``memory_stats()["peak_bytes_in_use"]``, max over local devices), the compiled
step's memory estimate, and the loss of every step (finite and decreasing is checked).

``--allocator auto`` (the default) leaves XLA's default allocator (BFC) on every mesh, as ``jla train``
does; ``cuda_async`` and ``bfc`` set ``XLA_PYTHON_CLIENT_ALLOCATOR``, and a value already in the
environment wins over ``auto``. Multi-GPU runs used ``cuda_async`` while each GPU of an 8-GPU run
peaked at its 1-GPU memory (BFC with preallocation off then ran out of memory on GPU 0 only, and the
other seven GPUs waited in their first collective forever). With the activations split along the batch
each GPU peaks at 15-30 GiB and BFC fits; on the same 8-GPU step the two allocators measured the same
(``benchmarks/multi_gpu.md``). The watchdog (``--stall-timeout``) turns a stall into a
failure record: no log record for that many seconds writes ``status = "failed"`` to ``--json`` and
exits with code 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "end_to_end" / "1p3b"
HYBRID_CONFIG_DIR = HERE / "end_to_end" / "hybrids_1p3b"  # Kimi-Linear-like and Qwen3-Next-like (end_to_end_1p3b.md)
_IMPLS = {
    # mode: (linear attention / SSM implementation, attention implementation)
    "kernels": ("mosaic_gpu", "mosaic_gpu"),
    "chunked": ("chunked", "cudnn"),
    "auto": ("auto", "auto"),
}


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True, help="config name under benchmarks/end_to_end/1p3b/ or a .toml path")
    p.add_argument("--data", required=True, help="CIDX prefix (without .bin/.idx) of 4096-token packed records")
    p.add_argument("--impl", choices=sorted(_IMPLS), default="kernels")
    p.add_argument("--attention", choices=("mosaic_gpu", "cudnn", "xla", "auto"), default=None,
                   help="override the softmax-attention implementation")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="sequences per data-parallel replica per micro-step (default: the config's)")
    p.add_argument("--grad-accum", type=int, default=None, help="micro-steps per optimizer step (default: the config's)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--dp", type=int, default=1, help="mesh.dp (replicated parameters)")
    p.add_argument("--fsdp", type=int, default=-1, help="mesh.fsdp (-1 = the remaining devices)")
    p.add_argument("--steps", type=int, default=20, help="total optimizer steps")
    p.add_argument("--skip", type=int, default=5, help="warm-up steps excluded from the timing")
    p.add_argument("--output-dir", default=None, help="trainer output dir (default: a temporary directory)")
    p.add_argument("--json", default=None, help="write the result JSON here")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="extra config override a.b=value")
    p.add_argument("--allocator", choices=("auto", "cuda_async", "bfc"), default="auto",
                   help="XLA GPU allocator: auto = XLA's default (BFC) on every mesh")
    p.add_argument("--stall-timeout", type=float, default=900.0,
                   help="fail the run when no log record arrives for this many seconds (0 = off)")
    args = p.parse_args(argv)
    if not 0 <= args.skip < args.steps:
        p.error("need 0 <= --skip < --steps")
    return args


_XLA_ENV = ("XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def _allocator_choice(allocator: str) -> str | None:
    """The ``XLA_PYTHON_CLIENT_ALLOCATOR`` value to set (``None``: leave XLA's default, BFC)."""
    return None if allocator == "auto" else allocator


class _Heartbeat(logging.Handler):
    """Notes the time of every log record (the trainer logs every step and the compile preflight)."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.last = time.monotonic()
        self.message = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.last = time.monotonic()
        try:
            self.message = record.getMessage()[:300]
        except Exception:  # noqa: BLE001 - a bad format string must not kill the run
            self.message = str(record.msg)[:300]


def _start_watchdog(timeout: float, heartbeat: _Heartbeat, on_stall) -> None:
    """A daemon thread: when ``heartbeat`` is silent for ``timeout`` seconds, ``on_stall(idle)`` then exit 3."""

    def watch() -> None:
        while True:
            time.sleep(max(1.0, min(30.0, timeout / 10)))
            idle = time.monotonic() - heartbeat.last
            if idle > timeout:
                try:
                    on_stall(idle)
                finally:
                    sys.stderr.flush()
                    os._exit(3)

    threading.Thread(target=watch, name="e2e-stall-watchdog", daemon=True).start()


def _config_path(arch: str) -> Path:
    path = Path(arch)
    if path.suffix == ".toml":
        return path
    path = CONFIG_DIR / f"{arch}.toml"
    if not path.exists() and (HYBRID_CONFIG_DIR / f"{arch}.toml").exists():
        return HYBRID_CONFIG_DIR / f"{arch}.toml"
    if not path.exists():
        names = sorted(p.stem for d in (CONFIG_DIR, HYBRID_CONFIG_DIR) for p in d.glob("*.toml"))
        raise SystemExit(f"no config {path}; available: {', '.join(names)}")
    return path


def _with_impl(model, mixer_impl: str, attention_impl: str):
    """``model`` with every mixer's ``implementation`` set (configs a stack does not use included)."""
    changes = {"attention": dataclasses.replace(model.attention, implementation=attention_impl)}
    if model.linear_attention is not None:
        changes["linear_attention"] = dataclasses.replace(model.linear_attention, implementation=mixer_impl)
    if model.ssm is not None:
        changes["ssm"] = dataclasses.replace(model.ssm, implementation=mixer_impl)
    if model.local_attention is not None:
        changes["local_attention"] = dataclasses.replace(model.local_attention, implementation=attention_impl)
    if model.mla is not None:  # MLA layers run the same attention implementation (unpadded heads on the kernel)
        changes["mla"] = dataclasses.replace(model.mla, implementation=attention_impl)
    return model.replace(**changes)


def _git_describe() -> str:
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _source_sha256() -> str:
    """SHA-256 over the package sources (``src/jla/**/*.py``, path and content): identifies trees without git."""
    import hashlib

    digest = hashlib.sha256()
    root = HERE.parent / "src" / "jla"
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _portable(text: str) -> str:
    """``text`` without this machine's paths: the repository root and the Python install prefixes."""
    import sysconfig

    prefixes = {str(HERE.parent) + os.sep, *(p + os.sep for p in sysconfig.get_paths().values())}
    for prefix in sorted(prefixes, key=len, reverse=True):
        text = text.replace(prefix, "")
    return text


def _clip(text: str, limit: int = 2000) -> str:
    """``text`` cut to about ``limit`` characters, keeping its start and its end (a traceback's cause)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters ...]\n{text[-half:]}"


def _gpu_snapshot() -> list[dict[str, str]]:
    """``nvidia-smi`` utilisation and memory of the visible GPUs only (no host names)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    query = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    keep = None if visible is None else {v.strip() for v in visible.split(",") if v.strip()}
    for line in out.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 5 and (keep is None or fields[0] in keep):
            rows.append(dict(zip(("index", "name", "utilization", "memory_used", "memory_total"), fields)))
    return rows


def _summary(history: list[dict], skip: int, drop_last: bool = False) -> dict:
    """Timing of the steps after ``skip``. ``drop_last`` (runs with ``train.dispatch_ahead``) leaves out the last
    step: a step dispatched ahead is timed from the arrival of the previous step's metrics, which the host reads
    only after its call that dispatches the next step returns, late in that next step; so every step's time is
    one step period except the last one's (the remainder after that call, a few tens of ms), and with it the mean
    would come out up to 1/timed steps too short (benchmarks/multi_gpu.md)."""
    timed = [h for h in history if h["step"] > skip]
    if drop_last and len(timed) > 1:
        timed = timed[:-1]
    times = [float(h["step_time"]) for h in timed]
    mean = statistics.fmean(times)
    first = history[0]
    tokens_per_step = first["tokens_per_sec"] * first["step_time"]
    return {
        "timed_steps": len(timed),
        "step_time_mean_s": mean,
        "step_time_median_s": statistics.median(times),
        "step_time_min_s": min(times),
        "step_time_max_s": max(times),
        "tokens_per_step": round(tokens_per_step),
        # Aggregate rates over the timed window: total tokens / total time.
        "tokens_per_sec": tokens_per_step / mean,
        "tokens_per_sec_per_gpu": tokens_per_step / mean / max(1, int(first["_devices"])),
        "mfu_percent": statistics.fmean(float(h["mfu_percent"]) * float(h["step_time"]) for h in timed) / mean,
        "hfu_percent": statistics.fmean(float(h["hfu_percent"]) * float(h["step_time"]) for h in timed) / mean,
    }


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # Before JAX creates its GPU client: the allocator is read once, at backend initialisation.
    choice = _allocator_choice(args.allocator)
    if choice is not None and (args.allocator != "auto" or "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = choice
    xla_env = {k: os.environ.get(k) for k in _XLA_ENV}
    from jla.config import load_config
    from jla.launch_env import launch

    config_path = _config_path(args.arch)
    output_dir = args.output_dir
    if output_dir is None:
        import tempfile

        output_dir = tempfile.mkdtemp(prefix=f"e2e-{config_path.stem}-")
    overrides = [
        f'data.paths=["{args.data}"]',
        f"train.seq_len={args.seq_len}",
        *([f"train.micro_batch_per_replica={args.micro_batch}"] if args.micro_batch is not None else []),
        *([f"train.grad_accum={args.grad_accum}"] if args.grad_accum is not None else []),
        f"train.max_steps={args.steps}",
        "train.log_every=1",
        "train.eval_every=0",
        "train.final_eval=false",
        "train.save_every=0",
        "train.save_final=false",
        "train.sharding_audit=false",
        f'train.output_dir="{output_dir}"',
        f"mesh.dp={args.dp}",
        f"mesh.fsdp={args.fsdp}",
        *args.overrides,
    ]
    cfg = load_config(str(config_path), overrides)
    mixer_impl, attention_impl = _IMPLS[args.impl]
    attention_impl = args.attention or attention_impl
    cfg = dataclasses.replace(cfg, model=_with_impl(cfg.model, mixer_impl, attention_impl))
    logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    launch(cfg.runtime)

    import jax
    import numpy as np

    from jla.models import parameter_count
    from jla.training.performance import estimate_flops_per_token
    from jla.training.trainer import Trainer

    devices = jax.devices()
    gpus_before = _gpu_snapshot()
    started = time.time()

    def write_failure(error: str) -> None:
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps({
                "arch": config_path.stem, "impl": args.impl, "mixer_implementation": mixer_impl,
                "attention_implementation": attention_impl, "mesh": {"dp": args.dp, "fsdp": args.fsdp if args.fsdp > 0 else len(devices) // args.dp, "tp": 1},
                "devices": len(devices), "status": "failed", "error": _clip(_portable(error)),
                "xla_client_env": xla_env, "jax_version": jax.__version__, "git_head": _git_describe(),
            }, indent=2))

    if args.stall_timeout > 0:
        heartbeat = _Heartbeat()
        logging.getLogger().addHandler(heartbeat)

        def on_stall(idle: float) -> None:
            error = (f"stalled: no log record for {idle:.0f} s (--stall-timeout {args.stall_timeout:g}); the last one: "
                     f"{heartbeat.message!r}. A device that ran out of memory leaves the others waiting in a "
                     "collective: look for 'ran out of memory' in the log.")
            print(f"E2E_ERROR {error}", file=sys.stderr)
            write_failure(error)

        _start_watchdog(args.stall_timeout, heartbeat, on_stall)
    try:
        result = Trainer(cfg).run()
    except Exception as exc:  # record the failure (e.g. a shape a kernel rejects), then fail
        write_failure(f"{type(exc).__name__}: {exc}")
        raise
    wall = time.time() - started
    gpus_after = _gpu_snapshot()

    history = [dict(h, _devices=len(devices)) for h in result.history]
    if len(history) != args.steps or [h["step"] for h in history] != list(range(1, args.steps + 1)):
        raise RuntimeError(f"expected one log record per step 1..{args.steps}; got {[h['step'] for h in history]}")
    losses = [float(h["loss"]) for h in history]
    summary = _summary(history, args.skip, drop_last=bool(getattr(cfg.train, "dispatch_ahead", False)))
    peaks = []
    for d in jax.local_devices():
        stats = d.memory_stats() or {}
        peaks.append(int(stats.get("peak_bytes_in_use", 0)))
    preflight_path = Path(output_dir) / "compile_preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    counts = parameter_count(cfg.model)
    flops = estimate_flops_per_token(cfg.model, cfg.train.seq_len)
    window = max(1, min(5, args.skip))
    early = statistics.fmean(losses[:window])
    late = statistics.fmean(losses[-window:])
    out = {
        "status": "ok",
        "arch": config_path.stem,
        "config": str(config_path.relative_to(HERE.parent)) if config_path.is_relative_to(HERE.parent) else str(config_path),
        "impl": args.impl,
        "mixer_implementation": mixer_impl,
        "attention_implementation": attention_impl,
        "mesh": {"dp": args.dp, "fsdp": args.fsdp if args.fsdp > 0 else len(devices) // args.dp, "tp": 1},
        "devices": len(devices),
        "device_kind": str(devices[0].device_kind),
        "seq_len": cfg.train.seq_len,
        "micro_batch_per_replica": cfg.train.micro_batch_per_replica,
        "grad_accum": cfg.train.grad_accum,
        "global_batch_sequences": summary["tokens_per_step"] // cfg.train.seq_len,
        "document_masking": cfg.data.document_masking,
        "parameters": {"total": counts.total, "embedding": counts.embedding, "non_embedding": counts.total - counts.embedding},
        "flops_per_token": flops.as_dict(),
        "steps": args.steps,
        "skipped_warmup_steps": args.skip,
        **summary,
        "peak_memory_gib_max_device": max(peaks) / 2**30 if peaks else None,
        "compiled_memory_estimate_gib": (preflight.get("memory") or {}).get("total_size_gib"),
        "compile_seconds": preflight.get("compile_seconds"),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_first_steps": early,
        "loss_mean_last_steps": late,
        "loss_finite": all(math.isfinite(x) for x in losses),
        "loss_decreasing": late < early,
        "per_step": [
            {k: h[k] for k in ("step", "loss", "grad_norm", "step_time", "tokens_per_sec", "mfu_percent") if k in h}
            for h in history
        ],
        "stop_reason": result.stop_reason,
        "wall_seconds_incl_compile": wall,
        "xla_client_env": xla_env,
        "gpus_before": gpus_before,
        "gpus_after": gpus_after,
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_head": _git_describe(),
        "source_sha256": _source_sha256(),
        "data": {"prefix_name": Path(args.data).name},
        "overrides": args.overrides,
    }
    line = {k: out[k] for k in ("arch", "impl", "devices", "tokens_per_sec", "tokens_per_sec_per_gpu", "mfu_percent",
                                "step_time_mean_s", "peak_memory_gib_max_device", "loss_last", "loss_decreasing")}
    print("E2E_RESULT " + json.dumps(line))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
    if not out["loss_finite"]:
        print("E2E_ERROR non-finite loss", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
