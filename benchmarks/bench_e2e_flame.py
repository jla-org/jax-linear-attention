# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time fla-org/flame's trainer on the 1.3B end-to-end configs (benchmarks/end_to_end_1p3b.md).

    python benchmarks/bench_e2e_flame.py --arch kda-1p3b --data /path/to/fineweb_edu_3B --nproc 8 \
        --micro-batch 16 --json results/flame-kda-1p3b-8gpu.json --jla-python /path/to/jax/bin/python

Runs in a PyTorch environment with FLA, flame and torchtitan (``end_to_end/1p3b/flame/setup_flame_env.sh``).
The run is flame's own ``flame.train.main``: FLA's model from ``end_to_end/1p3b/flame/<arch>.json``,
flame's ``parallelize_fla`` (per-block ``torch.compile``, FSDP2 with bf16 parameters and fp32 reductions,
activation checkpointing), torchtitan's AdamW, LR schedule and gradient clipping, logging every step.
Three things are replaced, nothing in the step itself:

* the data: flame's HF dataset/tokenizer pipeline is swapped for :class:`CidxBatches`, which draws the
  same global batches as the jla harness (``jla.data`` TokenWindowDataset over the CIDX records, the
  seeded BatchLoader permutation) and hands each rank its contiguous rows. With ``--varlen`` (document
  masking) every rank's rows are flattened into one sequence with ``cu_seqlens`` at every EOS and every
  row start, as flame's varlen collator does;
* the tokenizer: the corpus is pre-tokenized, so ``AutoTokenizer`` returns a stub with the vocabulary;
* rendezvous: ``init_process_group`` gets a ``file://`` store (some hosts refuse loopback TCP), and the
  runner starts one process per GPU itself instead of torchrun.

FLA shifts labels inside the model (target ``t`` = label ``t + 1``), so the labels are the inputs with
every target after an EOS and every target that would cross a row set to -100: the loss covers the
4095 in-record targets of each 4096-token record (the jla harness also scores the record's last token
against the next record's first token, and also drops EOS targets).

Timing follows ``bench_e2e_1p3b.py``: every step is logged (the log line syncs the GPU), step ``k`` ends
when its log call starts, steps ``1..--skip`` (step 1 includes torch.compile) are excluded and the other
steps are timed. MFU uses jla's analytic FLOPs of the matching preset (``--jla-python``: an
interpreter with JAX; ``estimate_flops_per_token(...).training``, 989 TFLOP/s dense bf16 per H100),
so both trainers are scored with the same FLOPs. Writes one JSON per run. The JSON's peak memory covers
the timed steps (the per-step peaks and the all-steps peak are kept too), ``fla_backend_dispatch`` holds
FLA's logged backend decisions, ``versions.commits`` every commit of the environment, and a step that
flame skipped on a NaN/Inf gradient norm (``skip_nan_inf``) fails the run. Paths in the JSON are
tree-relative. ``--from-raw`` rebuilds a JSON from an earlier run's dump folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAME_DIR = HERE / "end_to_end" / "1p3b" / "flame"
JLA_SRC = HERE.parent / "src"
H100_DENSE_BF16_TFLOPS = 989.0  # jla.training.performance DeviceProfile("H100")
# FLA layers whose varlen path ignores cu_seqlens (fla/layers/mamba2.py passes seq_idx=None): those
# run without document masking, one 4096-token row per sequence.
NO_VARLEN = {"mamba2-1p3b"}
# flame config -> jla preset with the same shapes (benchmarks/end_to_end/1p3b/<preset>.toml).
JLA_PRESET = {
    "transformer-1p3b": "transformer-1p3b",
    "transformer-swa-1p3b": "transformer-swa-1p3b",
    "gdn-1p3b": "gdn-1p3b",
    "gdn-hd128-1p3b": "gdn-hd128-1p3b",
    "kda-1p3b": "kda-1p3b",
    "mamba2-1p3b": "mamba2-1p3b",
    "mamba3-1p3b": "mamba3-1p3b",
}
_PACKAGES = ("torch", "triton", "flash-linear-attention", "fla-core", "flame", "torchtitan", "transformers",
             "flash-qla", "flash-kda", "tilelang", "flash-attn-3", "mamba-ssm", "causal-conv1d")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True, help="config name under end_to_end/1p3b/flame/ or a .json path")
    p.add_argument("--data", required=True, help="CIDX prefix (without .bin/.idx) of 4096-token packed records")
    p.add_argument("--nproc", type=int, default=1, help="GPUs (one process each) on this host")
    p.add_argument("--micro-batch", type=int, default=16, help="4096-token records per GPU per step")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="flame micro-steps per optimizer step (each takes micro-batch / grad-accum records)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--skip", type=int, default=5, help="warm-up steps excluded from the timing (step 1 compiles)")
    p.add_argument("--varlen", choices=("auto", "on", "off"), default="auto",
                   help="document masking via cu_seqlens; auto = on unless the FLA layer ignores cu_seqlens")
    p.add_argument("--ac", choices=("none", "full", "selective"), default="full",
                   help="flame activation checkpointing (jla's default remat is 'full')")
    p.add_argument("--no-compile", action="store_true", help="disable flame's per-block torch.compile")
    p.add_argument("--fla-backends", choices=("default", "triton"), default="default",
                   help="default = FLA's dispatch (FlashQLA/TileLang/FlashKDA where installed); triton = "
                        "FLA_DISABLE_BACKEND_DISPATCH=1")
    p.add_argument("--fa3-shim", action="store_true",
                   help="serve fla.layers.attn's flash_attn imports from FlashAttention-3 (end_to_end/1p3b/flame/fa3_shim)")
    p.add_argument("--flame-src", default=None,
                   help="the flame checkout (flame.train imports `custom_models` from its root, as train.sh runs it)")
    p.add_argument("--seed", type=int, default=0, help="data order seed (jla train.seed)")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--eos-id", type=int, default=128001)
    p.add_argument("--jla-python", default=None, help="python with JAX for jla's analytic FLOPs (CPU only)")
    p.add_argument("--jla-preset", default=None, help="jla preset for the FLOPs (default: the matching preset)")
    p.add_argument("--jla-src", default=str(JLA_SRC), help="jla source tree for the FLOPs (default: this tree)")
    p.add_argument("--dump-dir", default=None, help="flame job dump folder (default: a temporary directory)")
    p.add_argument("--cache-dir", default=None, help="torch.compile / Triton cache directory")
    p.add_argument("--json", default=None)
    p.add_argument("--set", dest="flame_args", action="append", default=[],
                   help="extra flame argument, e.g. --set=--training.fsdp_reshard_after_forward=always")
    p.add_argument("--from-raw", action="store_true",
                   help="do not run: rebuild the result JSON from --dump-dir of an earlier run (same arguments)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--init-file", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 0 <= args.skip < args.steps:
        p.error("need 0 <= --skip < --steps")
    if args.nproc < 1 or args.micro_batch < 1 or args.grad_accum < 1 or args.micro_batch % args.grad_accum:
        p.error("--nproc, --micro-batch, --grad-accum must be >= 1 and --grad-accum must divide --micro-batch")
    return args


def _model_config(arch: str) -> Path:
    path = Path(arch)
    if path.suffix == ".json":
        return path
    path = FLAME_DIR / f"{arch}.json"
    if not path.exists():
        names = sorted(p.stem for p in FLAME_DIR.glob("*.json"))
        raise SystemExit(f"no flame config {path}; available: {', '.join(names)}")
    return path


def _varlen(args) -> bool:
    if args.varlen == "auto":
        return Path(args.arch).stem not in NO_VARLEN
    return args.varlen == "on"


# --------------------------------------------------------------------------------------
# Data: the jla harness's global batches, as flame batches


class CidxBatches:
    """flame-style batches of the jla harness's records (``input_ids``/``labels``, ``cu_seqlens`` if varlen).

    Optimizer step ``g`` uses the records the jla ``BatchLoader`` yields as its ``g``-th global batch (same
    dataset, seed and shuffle); this rank takes rows ``[rank * rows, (rank + 1) * rows)`` and serves them as
    ``accum`` micro-batches of ``rows // accum`` records (flame calls ``next`` once per micro-step).
    """

    def __init__(self, prefix: str, *, seq_len: int, rows: int, rank: int, world: int, seed: int,
                 shuffle: bool, eos_id: int, varlen: bool, accum: int = 1) -> None:
        if rows % accum:
            raise ValueError(f"{rows} records per step do not split into {accum} micro-batches")
        if str(JLA_SRC) not in sys.path:
            sys.path.insert(0, str(JLA_SRC))
        from jla.data.datasets import TokenWindowDataset  # numpy only, no JAX
        from jla.data.loader import BatchLoader

        self.dataset = TokenWindowDataset(prefix, seq_len, exact_packed_records=True)
        self.loader = BatchLoader(self.dataset, rows * world, rows=(rank * rows, (rank + 1) * rows), seed=seed,
                                  shuffle=shuffle, num_workers=0)
        self.seq_len, self.rows, self.eos_id, self.varlen, self.accum = seq_len, rows, eos_id, varlen, accum
        self._pending: list = []
        self.first_next_time: float | None = None
        self.batches = 0
        self.token_checksum = 0  # sum of input ids of every batch this rank drew (identity check)

    def __iter__(self):
        return self

    def __next__(self):
        import numpy as np
        import torch

        if self.first_next_time is None:
            torch.cuda.synchronize()
            self.first_next_time = time.perf_counter()
        if not self._pending:
            step_rows = np.ascontiguousarray(next(self.loader)["input_ids"], dtype=np.int64)  # [rows, T]
            self.batches += 1
            self.token_checksum += int(step_rows.sum())
            self._pending = list(np.split(step_rows, self.accum))
        tokens = self._pending.pop(0)
        x = torch.from_numpy(tokens)
        if self.varlen:
            x = x.reshape(1, -1)
        labels = x.clone()
        flat_in, flat_lab = x.reshape(-1), labels.view(-1)
        after_eos = torch.zeros_like(flat_in, dtype=torch.bool)
        after_eos[1:] = flat_in[:-1] == self.eos_id
        row_start = torch.zeros_like(after_eos)
        row_start[:: self.seq_len] = True
        flat_lab[after_eos] = -100  # targets of EOS positions (the next document's first token)
        flat_lab[row_start] = -100  # in varlen rows are concatenated: no target crosses a row
        out = {"input_ids": x, "labels": labels}
        if self.varlen:
            starts = torch.nonzero(after_eos | row_start).flatten()
            out["cu_seqlens"] = torch.cat([starts, torch.tensor([flat_in.numel()])]).to(torch.int32)
        return out

    def state_dict(self):
        return {"batches": self.batches}

    def load_state_dict(self, state):
        if state and state.get("batches", 0):
            raise RuntimeError("CidxBatches does not resume")


# --------------------------------------------------------------------------------------
# Worker: one GPU, flame.train.main


def _patch_fla_mamba3_dtensor_init(captured: dict) -> None:
    """FLA main's Mamba-3 ``_init_weights`` copies a plain tensor into ``dt_bias``, which FSDP2 has already made a
    DTensor, so on several GPUs every rank fails in ``aten.copy_`` ("got mixed torch.Tensor and DTensor"). FLA's
    Mamba-2 init converts its value first (``tensor_to_dtensor``: replicated from rank 0, then redistributed to
    the parameter's placements). This does the same for Mamba-3's ``dt_bias``, with FLA's formula, and leaves the
    rest of FLA's init unchanged (the other Mamba-3 parameters are filled in place, which DTensors support)."""
    import math

    import torch
    from torch.distributed.tensor import DTensor, Replicate

    try:
        from fla.models.mamba3 import modeling_mamba3 as m3
    except ImportError:
        return
    original = m3.Mamba3PreTrainedModel._init_weights

    def _init_weights(self, module, *a, **kw):
        p = getattr(module, "dt_bias", None) if isinstance(module, m3.Mamba3) else None
        if isinstance(p, DTensor) and p.device.type != "meta" and not getattr(p, "_is_hf_initialized", False):
            cfg = self.config
            dt = torch.exp(torch.rand(cfg.num_heads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
                           + math.log(cfg.dt_min)).clamp(min=cfg.dt_init_floor)
            inv_dt = (dt + torch.log(-torch.expm1(-dt))).to(device=p.device, dtype=p.dtype)
            value = DTensor.from_local(inv_dt, device_mesh=p.device_mesh, placements=[Replicate()] * len(p.placements),
                                       run_check=True).redistribute(p.device_mesh, p.placements)
            with torch.no_grad():
                p.copy_(value)
            p._is_hf_initialized = True  # FLA's init then skips its plain-tensor copy (and keeps the rest)
        return original(self, module, *a, **kw)

    m3.Mamba3PreTrainedModel._init_weights = _init_weights
    captured.setdefault("fla_patches", []).append("mamba3 dt_bias init as a DTensor under FSDP2 (as FLA's Mamba-2 init)")


def _worker(args) -> int:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if args.fa3_shim:
        sys.path.insert(0, str(FLAME_DIR / "fa3_shim"))
    if args.flame_src:  # flame.train imports `custom_models` from the root of the flame checkout
        sys.path.append(args.flame_src)
    import torch
    import torch.distributed as dist

    original_init = dist.init_process_group

    def init_process_group(*a, **kw):
        kw.setdefault("init_method", f"file://{args.init_file}")
        kw.setdefault("rank", rank)
        kw.setdefault("world_size", world)
        return original_init(*a, **kw)

    dist.init_process_group = init_process_group

    import flame.train as flame_train
    from flame.config_manager import JobConfig
    from torchtitan.components import metrics as tt_metrics
    from torchtitan.tools.logging import init_logger

    model_config = _model_config(args.arch)
    vocab = int(json.loads(model_config.read_text())["vocab_size"])
    varlen = _varlen(args)
    captured: dict = {"records": [], "loader": None}

    class _Tokenizer:
        vocab_size = vocab

        def __repr__(self) -> str:
            return f"pre-tokenized CIDX corpus (vocab {vocab})"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_a, **_kw):
            return _Tokenizer()

    def build_dataloader(*, rank, world_size, batch_size, seq_len, context_len, varlen, **_kw):
        if seq_len * batch_size % context_len:
            raise ValueError("flame batch is not a whole number of records")
        loader = CidxBatches(args.data, seq_len=context_len, rows=seq_len * batch_size // context_len * args.grad_accum,
                             rank=rank, world=world_size, seed=args.seed, shuffle=not args.no_shuffle,
                             eos_id=args.eos_id, varlen=varlen, accum=args.grad_accum)
        captured["loader"] = loader
        return loader

    original_counts = flame_train.get_nparams_and_flops

    def get_nparams_and_flops(model, config, seq_len):
        n, flops = original_counts(model, config, seq_len)
        embed = sum(p.numel() for name, p in model.named_parameters() if "embed" in name or "lm_head" in name)
        captured.update(params=int(n), embedding_params=int(embed), flame_flops_per_token=float(flops))
        return n, flops

    original_log = tt_metrics.MetricsProcessor.log

    def log(self, *a, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        step = int(a[0] if a else kw["step"])
        loss = a[1] if len(a) > 1 else kw.get("global_avg_loss")
        extra = kw.get("extra_metrics") or (a[3] if len(a) > 3 else None) or {}
        # torchtitan resets the peak memory stats at the end of every log call, so these are this step's peaks
        captured["records"].append({"step": step, "end": now, "loss": float(loss),
                                    "grad_norm": float(extra.get("optimizer/grad_norm", float("nan"))),
                                    # flame's skip_nan_inf counter (cumulative): a skipped step does no update
                                    "skipped_steps": int(extra.get("optimizer/skipped_step", 0) or 0),
                                    "peak_allocated": torch.cuda.max_memory_allocated(),
                                    "peak_reserved": torch.cuda.max_memory_reserved()})
        captured["peak_allocated"] = max(captured.get("peak_allocated", 0), torch.cuda.max_memory_allocated())
        captured["peak_reserved"] = max(captured.get("peak_reserved", 0), torch.cuda.max_memory_reserved())
        return original_log(self, *a, **kw)

    flame_train.AutoTokenizer = _AutoTokenizer
    flame_train.build_dataset = lambda **_kw: None
    flame_train.build_dataloader = build_dataloader
    flame_train.get_nparams_and_flops = get_nparams_and_flops
    tt_metrics.MetricsProcessor.log = log

    per_micro_step = args.micro_batch // args.grad_accum
    if varlen:
        geometry = ["--training.batch_size", "1", "--training.seq_len", str(per_micro_step * args.seq_len),
                    "--training.varlen"]
    else:
        geometry = ["--training.batch_size", str(per_micro_step), "--training.seq_len", str(args.seq_len)]
    geometry += ["--training.gradient_accumulation_steps", str(args.grad_accum)]
    flame_argv = [
        "--job.config_file", str(FLAME_DIR / "job.toml"),
        "--job.dump_folder", args.dump_dir,
        "--model.config", str(model_config),
        *geometry,
        "--training.context_len", str(args.seq_len),
        "--training.steps", str(args.steps),
        "--activation_checkpoint.mode", args.ac,
        *([] if args.no_compile else ["--training.compile"]),
        *args.flame_args,
    ]
    init_logger()
    config = JobConfig()
    config.parse_args(flame_argv)
    if args.no_compile:
        config.training.compile = False
    if world > 1:
        _patch_fla_mamba3_dtensor_init(captured)
    flame_train.main(config)  # resets the peak memory stats before its training loop

    peak = torch.tensor([captured.get("peak_allocated", 0), captured.get("peak_reserved", 0)], dtype=torch.float64,
                        device="cuda")
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    # per-step peaks, max over ranks; row s-1 = step s (fixed shape on every rank, whatever each logged)
    step_peak = torch.zeros(args.steps, 2, dtype=torch.float64, device="cuda")
    for r in captured["records"]:
        if 1 <= r["step"] <= args.steps:
            step_peak[r["step"] - 1, 0] = float(r.pop("peak_allocated"))
            step_peak[r["step"] - 1, 1] = float(r.pop("peak_reserved"))
    dist.all_reduce(step_peak, op=dist.ReduceOp.MAX)
    loader = captured["loader"]
    checksum = torch.tensor([loader.token_checksum if loader else 0], dtype=torch.float64, device="cuda")
    dist.all_reduce(checksum)
    if rank == 0:
        raw = {
            "records": captured["records"],
            "loop_start": loader.first_next_time if loader else None,
            "batches_drawn_per_rank": loader.batches if loader else 0,
            "input_token_checksum_all_ranks": int(checksum.item()),
            "peak_allocated_gib_max_rank": peak[0].item() / 2**30,
            "peak_reserved_gib_max_rank": peak[1].item() / 2**30,
            "per_step_peak_gib_max_rank": (step_peak / 2**30).tolist(),  # [allocated, reserved] per step
            "params": captured.get("params"),
            "embedding_params": captured.get("embedding_params"),
            "flame_flops_per_token": captured.get("flame_flops_per_token"),
            "device_name": torch.cuda.get_device_name(),
            "flame_argv": flame_argv,
            "fla_patches": captured.get("fla_patches", []),
            "varlen": varlen,
        }
        Path(args.dump_dir, "bench_raw.json").write_text(json.dumps(raw, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    return 0


# --------------------------------------------------------------------------------------
# Launcher


def _jla_flops(args) -> dict | None:
    preset = args.jla_preset or JLA_PRESET.get(Path(args.arch).stem)
    if args.jla_python is None or preset is None:
        return None
    code = (
        "import json, sys\n"
        "from jla.models import get_preset, parameter_count\n"
        "from jla.training.performance import estimate_flops_per_token\n"
        "cfg = get_preset(sys.argv[1]); n = parameter_count(cfg)\n"
        "print(json.dumps({'preset': sys.argv[1], 'flops_per_token': estimate_flops_per_token(cfg, int(sys.argv[2])).as_dict(),"
        " 'parameters': {'total': n.total, 'embedding': n.embedding}}))\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONPATH=args.jla_src)
    out = subprocess.run([args.jla_python, "-c", code, preset, str(args.seq_len)], capture_output=True, text=True,
                         env=env, timeout=600)
    if out.returncode:
        raise RuntimeError(f"jla FLOPs for {preset} failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _versions() -> dict:
    from importlib import metadata

    found = {}
    for name in _PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    commits = Path(sys.prefix) / "FLAME_ENV_COMMITS.txt"
    if commits.exists():  # every "<project> <commit or version> (...)" line before the pip freeze
        lines = commits.read_text().splitlines()
        end = next((i for i, line in enumerate(lines) if line.startswith("-- freeze")), len(lines))
        found["commits"] = [line.split(" (")[0].strip() for line in lines[:end] if line.strip()]
    return found


def _fla_dispatch(log_path: Path) -> dict:
    """FLA's backend-dispatch decisions logged in a rank log ("[FLA Backend] <op> -> <backend>[ rejected: why]").

    FLA logs each decision once per process. An op with no line has no other backend registered for that call
    and runs FLA's own Triton kernel; a rejected backend also falls back to Triton.
    """
    lines: list[str] = []
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {"log_lines": [], "served": {}, "rejected": {}}
    for line in text.splitlines():
        i = line.find("[FLA Backend]")
        if i >= 0:
            msg = line[i + len("[FLA Backend]"):].strip()
            if msg not in lines:
                lines.append(msg)
    served, rejected = {}, {}
    for msg in lines:
        op, _, rest = msg.partition(" -> ")
        backend, sep, why = rest.partition(" rejected: ")
        if sep:
            rejected[op.strip()] = f"{backend.strip()}: {why.strip()}"
        else:
            served[op.strip()] = backend.strip()
    return {"log_lines": lines, "served": served, "rejected": rejected}


def _portable(value: str, dump_dir: str) -> str:
    """Tree-relative paths in the result JSON (no host paths): this tree's root -> '', dump dir -> <dump-dir>."""
    for d in {str(Path(dump_dir).resolve()), str(dump_dir)}:
        value = value.replace(d, "<dump-dir>")
    value = value.replace(str(HERE.parent) + os.sep, "")
    i = value.find("/benchmarks/end_to_end/")  # a config of another checkout (e.g. --from-raw of an older run)
    return value[i + 1:] if value.startswith("/") and i >= 0 else value


def _gpu_snapshot() -> list[dict[str, str]]:
    """``nvidia-smi`` utilisation and memory of the visible GPUs only (no host names)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    query = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keep = None if visible is None else {v.strip() for v in visible.split(",") if v.strip()}
    rows = []
    for line in out.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 5 and (keep is None or fields[0] in keep):
            rows.append(dict(zip(("index", "name", "utilization", "memory_used", "memory_total"), fields)))
    return rows


def _launch(args) -> int:
    dump_dir = args.dump_dir or tempfile.mkdtemp(prefix=f"flame-{Path(args.arch).stem}-")
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    flops = _jla_flops(args)
    init_file = Path(dump_dir) / f"pg-{uuid.uuid4().hex}"
    argv = [a for a in sys.argv[1:]]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", WORLD_SIZE=str(args.nproc),
               LOCAL_WORLD_SIZE=str(args.nproc), PYTHONDONTWRITEBYTECODE="1")
    if args.fla_backends == "triton":
        env["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    if args.cache_dir:
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(args.cache_dir) / "inductor"))
        env.setdefault("TRITON_CACHE_DIR", str(Path(args.cache_dir) / "triton"))
    if args.from_raw:  # summarise an earlier run's dump dir (bench_raw.json, rank0.log) without running it
        gpus_before = gpus_after = []
        codes, wall = [0], None
    else:
        gpus_before = _gpu_snapshot()
        started = time.time()
        procs = []
        for rank in range(args.nproc):
            worker_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
            cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--worker", "--init-file", str(init_file),
                   "--dump-dir", dump_dir]
            log = open(Path(dump_dir) / f"rank{rank}.log", "w")
            procs.append((subprocess.Popen(cmd, env=worker_env, stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for proc, log in procs:
            codes.append(proc.wait())
            log.close()
        wall = time.time() - started
        gpus_after = _gpu_snapshot()
    if any(codes):
        tail = (Path(dump_dir) / f"rank{codes.index(next(c for c in codes if c))}.log").read_text()[-4000:]
        print(f"E2E_ERROR worker exit codes {codes}\n{tail}", file=sys.stderr)
        return 1
    raw = json.loads((Path(dump_dir) / "bench_raw.json").read_text())
    records = raw["records"]
    if [r["step"] for r in records] != list(range(1, args.steps + 1)):
        raise RuntimeError(f"expected one log record per step 1..{args.steps}; got {[r['step'] for r in records]}")
    ends = [raw["loop_start"]] + [r["end"] for r in records]
    for r, t0, t1 in zip(records, ends[:-1], ends[1:]):
        r["step_time"] = t1 - t0
    timed = [r["step_time"] for r in records if r["step"] > args.skip]
    mean = statistics.fmean(timed)
    tokens_per_step = args.micro_batch * args.seq_len * args.nproc
    tps = tokens_per_step / mean
    training_flops = flops["flops_per_token"]["training"] if flops else None
    peak = H100_DENSE_BF16_TFLOPS * 1e12 * args.nproc
    losses = [r["loss"] for r in records]
    window = max(1, min(5, args.skip))
    step_peaks = raw.get("per_step_peak_gib_max_rank") or []  # [allocated, reserved] GiB per step, max rank
    timed_peaks = [p for i, p in enumerate(step_peaks) if i + 1 > args.skip]
    skipped = max((r.get("skipped_steps", 0) for r in records), default=0)
    grad_norm_finite = all(math.isfinite(r["grad_norm"]) for r in records)
    out = {
        "trainer": "flame",
        "arch": Path(args.arch).stem,
        "config": str(_model_config(args.arch).relative_to(HERE.parent)) if _model_config(args.arch).is_relative_to(HERE.parent) else str(_model_config(args.arch)),
        "devices": args.nproc,
        "device_kind": raw["device_name"],
        "seq_len": args.seq_len,
        "micro_batch_per_gpu": args.micro_batch,
        "grad_accum": args.grad_accum,
        "global_batch_sequences": args.micro_batch * args.nproc,
        "tokens_per_step": tokens_per_step,
        "document_masking": raw["varlen"],
        "activation_checkpointing": args.ac,
        "torch_compile": not args.no_compile,
        "fla_backends": args.fla_backends,
        "attention_kernel": "flash-attn-3 via FA2-API shim" if args.fa3_shim else "flash_attn (FA2) if installed",
        "fla_patches": raw.get("fla_patches", []),
        "parameters": {"total": raw["params"], "embedding": raw["embedding_params"],
                       "non_embedding": (raw["params"] or 0) - (raw["embedding_params"] or 0)},
        "jla_flops": flops,
        "steps": args.steps,
        "skipped_warmup_steps": args.skip,
        "timed_steps": len(timed),
        "step_time_mean_s": mean,
        "step_time_median_s": statistics.median(timed),
        "step_time_min_s": min(timed),
        "step_time_max_s": max(timed),
        "step1_time_incl_compile_s": records[0]["step_time"],
        "tokens_per_sec": tps,
        "tokens_per_sec_per_gpu": tps / args.nproc,
        "mfu_percent": 100.0 * training_flops * tps / peak if training_flops else None,
        "mfu_percent_flame_formula": 100.0 * raw["flame_flops_per_token"] * tps / peak if raw["flame_flops_per_token"] else None,
        # timed steps only (step > --skip): the warm-up steps include compile and kernel autotuning, whose
        # peak depends on the cache state; the all-steps peak is kept as *_incl_warmup
        "peak_memory_allocated_gib_max_gpu": max((p[0] for p in timed_peaks), default=None),
        "peak_memory_reserved_gib_max_gpu": max((p[1] for p in timed_peaks), default=None),
        "peak_memory_allocated_gib_max_gpu_incl_warmup": raw["peak_allocated_gib_max_rank"],
        "peak_memory_reserved_gib_max_gpu_incl_warmup": raw["peak_reserved_gib_max_rank"],
        "optimizer_skipped_steps": skipped,
        "grad_norm_finite": grad_norm_finite,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_first_steps": statistics.fmean(losses[:window]),
        "loss_mean_last_steps": statistics.fmean(losses[-window:]),
        "loss_finite": all(math.isfinite(x) for x in losses),
        "loss_decreasing": statistics.fmean(losses[-window:]) < statistics.fmean(losses[:window]),
        "input_token_checksum": raw["input_token_checksum_all_ranks"],
        "per_step": [{k: r.get(k) for k in ("step", "loss", "grad_norm", "skipped_steps", "step_time")}
                     | ({"peak_allocated_gib": step_peaks[i][0], "peak_reserved_gib": step_peaks[i][1]}
                        if i < len(step_peaks) else {})
                     for i, r in enumerate(records)],
        "fla_backend_dispatch": _fla_dispatch(Path(dump_dir) / "rank0.log"),
        "wall_seconds_incl_compile": wall,
        "gpus_before": gpus_before,
        "gpus_after": gpus_after,
        "versions": _versions(),
        "python": platform.python_version(),
        "data": {"prefix_name": Path(args.data).name, "seed": args.seed, "shuffle": not args.no_shuffle,
                 "eos_id": args.eos_id},
        "flame_argv": [_portable(str(a), dump_dir) for a in raw["flame_argv"]],
    }
    line = {k: out[k] for k in ("arch", "devices", "document_masking", "tokens_per_sec", "tokens_per_sec_per_gpu",
                                "mfu_percent", "step_time_mean_s", "peak_memory_allocated_gib_max_gpu", "loss_last",
                                "loss_decreasing")}
    print("E2E_RESULT " + json.dumps(line))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
    if skipped:  # job.toml keeps flame's skip_nan_inf: a skipped (cheaper) step must not pass as a timed one
        print(f"E2E_ERROR flame skipped {skipped} optimizer step(s) on a NaN/Inf gradient norm", file=sys.stderr)
        return 1
    return 0 if out["loss_finite"] and grad_norm_finite else 1


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return _worker(args) if args.worker else _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
