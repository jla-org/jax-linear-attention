# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experiment configuration: frozen dataclasses loaded strictly from TOML or JSON.

An experiment file has one table per section::

    name = "tiny-synthetic"

    [model]
    preset = "tiny"        # optional: start from jla.models.get_preset(...)
    d_model = 128          # field overrides, nested tables for sub-configs

    [optim]  [train]  [data]  [mesh]  [runtime]

Loading is strict: unknown keys, wrong types, booleans where numbers are
expected, NaN/Inf and (for JSON) duplicate keys are rejected with the dotted
path of the offending value. ``--set a.b.c=value`` overrides are applied to the
raw mapping before validation, so they obey the same rules. Values are parsed
as TOML (``3e-4``, ``true``, ``"text"``, ``[1, 2]``); bare words are strings and
``none`` clears an optional field.

Nothing here reads environment variables: the resolved config is the complete
description of a run, and :meth:`ExperimentConfig.config_hash` identifies it in
checkpoint manifests.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jla.models.config import ModelConfig

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "MeshConfig",
    "OptimConfig",
    "RuntimeConfig",
    "TrainConfig",
    "apply_overrides",
    "config_from_mapping",
    "dump_toml",
    "load_config",
    "load_mapping",
    "parse_override",
]

MAX_CONFIG_BYTES = 4 * 1024 * 1024

OptimizerName = Literal["adamw", "nadamw", "adafactor", "ademamix", "lion", "sgd", "muon"]
ScheduleName = Literal["cosine", "wsd", "inv_sqrt", "constant", "noam", "trapezoidal"]
MomentDtype = Literal["float32", "bfloat16", "float16", "param"]
CheckpointFormat = Literal["auto", "safetensors", "sharded", "orbax"]


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer, learning-rate schedule and gradient clipping."""

    name: OptimizerName = "adamw"
    lr: float = 3e-4
    schedule: ScheduleName = "cosine"
    warmup_steps: int = 0
    schedule_steps: int = 0  # 0 = train.max_steps
    min_lr_ratio: float = 0.1
    stable_fraction: float = 0.8  # wsd / trapezoidal
    decay_type: Literal["linear", "cosine", "sqrt"] = "cosine"  # wsd decay shape
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0  # 0 disables clipping
    mu_dtype: MomentDtype = "float32"  # AdamW/NAdamW first-moment storage
    nu_dtype: MomentDtype = "float32"  # AdamW/NAdamW second-moment storage
    embedding_lr_scale: float = 1.0
    muon_ns_steps: int = 5
    muon_momentum: float = 0.95
    muon_adam_lr: float | None = None  # LR of Muon's Adam fallback leaves; None = lr
    muon_consistent_rms: float = 0.2  # 0 = optax's shape-dependent scaling
    trainable_patterns: tuple[str, ...] = ()  # regexes; empty = everything trains
    skip_spikes: bool = False
    spike_interval: int = 128
    spike_scaling_factor: float = 6.0
    multi_steps: int = 1  # optimizer-level accumulation across step calls

    def __post_init__(self) -> None:
        _require(self.lr > 0, "optim.lr must be > 0")
        _require(self.warmup_steps >= 0, "optim.warmup_steps must be >= 0")
        _require(self.schedule_steps >= 0, "optim.schedule_steps must be >= 0")
        _require(0.0 <= self.min_lr_ratio <= 1.0, "optim.min_lr_ratio must be in [0, 1]")
        _require(0.0 <= self.stable_fraction <= 1.0, "optim.stable_fraction must be in [0, 1]")
        _require(0.0 <= self.b1 < 1.0 and 0.0 <= self.b2 < 1.0, "optim.b1/b2 must be in [0, 1)")
        _require(self.eps > 0, "optim.eps must be > 0")
        _require(self.weight_decay >= 0, "optim.weight_decay must be >= 0")
        _require(self.max_grad_norm >= 0, "optim.max_grad_norm must be >= 0")
        _require(self.embedding_lr_scale >= 0, "optim.embedding_lr_scale must be >= 0")
        _require(self.muon_ns_steps >= 1, "optim.muon_ns_steps must be >= 1")
        _require(0.0 <= self.muon_momentum < 1.0, "optim.muon_momentum must be in [0, 1)")
        _require(self.muon_consistent_rms >= 0, "optim.muon_consistent_rms must be >= 0")
        _require(self.spike_interval >= 2, "optim.spike_interval must be >= 2")
        _require(self.spike_scaling_factor > 0, "optim.spike_scaling_factor must be > 0")
        _require(self.multi_steps >= 1, "optim.multi_steps must be >= 1")
        for pattern in self.trainable_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"optim.trainable_patterns: invalid regex {pattern!r}: {exc}") from exc


@dataclass(frozen=True)
class TrainConfig:
    """Batch geometry, loss, loop cadence, checkpointing and diagnostics."""

    seed: int = 0
    max_steps: int = 1000
    seq_len: int = 2048
    micro_batch_per_replica: int = 8  # sequences per data-parallel replica (dp * fsdp) per microbatch
    grad_accum: int = 1  # microbatches per optimizer step (lax.scan inside the step)
    label_smoothing: float = 0.0
    z_loss: float = 0.0  # coefficient on the mean of logsumexp**2
    ce_vocab_chunk: int = 4096
    ce_compute_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    aux_loss_cap: float = 100.0  # clip finite model aux losses to +-cap; 0 disables
    scan_unroll: int = 0  # 0 = unroll the microbatch scan when grad_accum <= 4
    log_every: int = 10
    # Dispatch the next step before reading a log step's metrics (not at eval, save or profiler boundaries), so
    # the devices do not wait for the host's logging and dispatch; exact. A stop request (signal, time limit,
    # telemetry halt) then takes effect one step later. With several processes the stop check after a dispatched
    # step runs on the hosts (jla.distributed.init.process_any_host), so it does not wait for that step.
    # Off by default; on in the transformer-1p3b TOML. On one H100 it gains
    # 0.3-1.3% for transformer-1p3b in FP8 and nothing for gdn-hd128-1p3b; on several GPUs the host waits for each
    # step's rejected-step lax.cond, so it gains only with update_guard = "select" (benchmarks/end_to_end_1p3b.md,
    # benchmarks/multi_gpu.md). With it on, the last logged step's time comes out short (its metrics
    # are read without the lag of the others), so time such runs without their last step.
    dispatch_ahead: bool = False
    eval_every: int = 0  # 0 disables periodic evaluation
    eval_batches: int = 20
    final_eval: bool = True
    save_every: int = 0  # 0 disables periodic checkpoints
    save_final: bool = True
    keep_checkpoints: int = 3  # 0 keeps every committed checkpoint
    async_checkpoint: bool = True
    checkpoint_format: CheckpointFormat = "auto"
    resume: Literal["auto", "required", "never"] = "auto"
    output_dir: str = "runs/experiment"
    time_limit_minutes: float = 0.0  # graceful checkpoint + stop after this wall time; 0 disables
    profile_start_step: int = 0  # 0 disables the profiler window
    profile_steps: int = 0
    graph_health: bool = False  # parse the profiler trace for graph-health counters
    compile_preflight: bool = True
    sharding_audit: bool = True
    fail_on_replicated_params: bool = False
    replicated_params_tolerance: float = 0.05
    max_consecutive_skips: int = 20  # halt after this many rejected steps in a row; 0 disables
    # Metric sinks of the primary process (jla.training.loggers); wandb/tensorboard are optional extras.
    loggers: tuple[Literal["jsonl", "wandb", "tensorboard"], ...] = ("jsonl",)
    wandb_project: str = ""  # required with "wandb"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline"] = "online"  # offline: files under <output_dir>/wandb, sync later
    tensorboard_dir: str = ""  # "" = <output_dir>/tensorboard
    wandb_run_id: str = ""  # set: resumes of this training continue one W&B run (wandb.init id, resume="allow")
    ce_impl: Literal["xla", "mosaic_gpu", "auto"] = "xla"  # LM-head cross entropy (jla.ops.cross_entropy)
    # Form the LM-head gradient in the forward pass (jla.ops.cross_entropy.linear_cross_entropy_objective):
    # no logits recompute in the backward pass; needs ce_impl "mosaic_gpu" or "auto" (XLA keeps the recompute).
    # MTP heads take it too (jla.training.step._mtp_objective), with "auto": a call the kernels decline uses XLA.
    ce_grad_in_forward: bool = False
    ce_row_chunk: int = 8192  # rows per block of the forward-pass gradient (transient 4 * rows * vocab bytes)
    # With ce_grad_in_forward: the LM head's (and MTP heads') two gradient GEMMs in FP8 (e5m2 x e4m3). Lossy:
    # +0.02 loss after 600 steps of transformer-1p3b for a 1-4% faster step (benchmarks/cross_entropy.md); not
    # recommended.
    ce_fp8: bool = False
    # Collect once and gc.freeze() the surviving Python objects before the first step (unfrozen when the loop ends),
    # so the first full collection after compiling, a host stall of a few tenths of a second at a step that varies
    # from run to run, runs before the loop instead of inside it. Needs compile_preflight (without it the step
    # compiles after the freeze; jla.training.trainer._frozen_gc). Host-side only: the steps, their results and
    # the steady-state step time are unchanged.
    gc_freeze: bool = True
    # "step": on several devices the micro-steps run per device (whole weights, FP8 copies and transposes made
    # once) and the gradients are reduce-scattered once per step instead of once per micro-step, with the loss
    # sums and FP8 amax maxima in one collective each (jla.training.grad_reduce). Each device computes one GPU's
    # micro-steps; only the order of the float32 gradient sums changes. Every mesh axis of size > 1 must split the
    # batch; refused with MoE, MTP, dropout, FP8 current scaling, the DSA indexer loss, Mamba-1 layers and, on GPU,
    # cuDNN or "auto" attention (set model.*.implementation = "mosaic_gpu" or "xla"). Memory: about 10 bytes per
    # parameter on every device whatever the FSDP size (the whole float32 weights and gradient accumulator, the FP8
    # copies): about 30 GB at 3B, and 7B does not fit on an 80 GB GPU, so it is for models up to about 3B (a warning
    # is logged above half of the device's memory). benchmarks/multi_gpu.md
    grad_reduce: Literal["micro_step", "step"] = "micro_step"
    # How a rejected step (non-finite loss or gradient, no valid targets) keeps the parameters and optimizer state:
    # "cond" (lax.cond around the update) or "select" (the update computed and every leaf selected elementwise: the
    # same rule without that conditional, so on several GPUs the host does not wait for the step's gradient norm
    # before dispatching the next step; optim.skip_spikes keeps its own lax.cond; on GPU the update fuses
    # differently, a few float32 ulps). On in the transformer-1p3b TOML. benchmarks/multi_gpu.md
    update_guard: Literal["cond", "select"] = "cond"

    def __post_init__(self) -> None:
        _require(len(set(self.loggers)) == len(self.loggers), "train.loggers must not repeat a logger")
        _require(
            "wandb" not in self.loggers or bool(self.wandb_project.strip()),
            "train.loggers 'wandb' needs train.wandb_project",
        )
        for name in ("max_steps", "seq_len", "micro_batch_per_replica", "grad_accum", "ce_vocab_chunk"):
            _require(getattr(self, name) >= 1, f"train.{name} must be >= 1")
        for name in (
            "scan_unroll",
            "log_every",
            "eval_every",
            "save_every",
            "keep_checkpoints",
            "profile_start_step",
            "profile_steps",
            "max_consecutive_skips",
        ):
            _require(getattr(self, name) >= 0, f"train.{name} must be >= 0")
        _require(self.eval_batches >= 1, "train.eval_batches must be >= 1")
        _require(0.0 <= self.label_smoothing < 1.0, "train.label_smoothing must be in [0, 1)")
        _require(self.z_loss >= 0, "train.z_loss must be >= 0")
        _require(self.aux_loss_cap >= 0, "train.aux_loss_cap must be >= 0")
        _require(self.time_limit_minutes >= 0, "train.time_limit_minutes must be >= 0")
        _require(
            0.0 <= self.replicated_params_tolerance <= 1.0,
            "train.replicated_params_tolerance must be in [0, 1]",
        )
        _require(bool(self.output_dir.strip()), "train.output_dir must be non-empty")
        _require(
            (self.profile_start_step == 0) == (self.profile_steps == 0),
            "train.profile_start_step and train.profile_steps must both be set or both be 0",
        )
        _require(
            not self.graph_health or self.profile_steps > 0,
            "train.graph_health parses the profiler trace and needs a profiler window",
        )
        _require(self.ce_row_chunk >= 1, "train.ce_row_chunk must be >= 1")
        _require(not self.ce_fp8 or self.ce_grad_in_forward, "train.ce_fp8 needs train.ce_grad_in_forward")
        _require(
            not self.ce_grad_in_forward or self.ce_impl != "xla",
            "train.ce_grad_in_forward needs train.ce_impl 'mosaic_gpu' or 'auto' (the XLA op recomputes the logits)",
        )


@dataclass(frozen=True)
class DataConfig:
    """Training and evaluation token sources."""

    source: Literal["synthetic", "cidx", "stream", "sft"] = "synthetic"
    paths: tuple[str, ...] = ()  # CIDX prefixes (without .bin/.idx); several = weighted blend
    weights: tuple[float, ...] = ()  # blend weights; empty = equal
    eval_path: str | None = None  # separate CIDX prefix for evaluation
    eval_holdout_records: int = 0  # or hold out the tail records of the (single) train path
    exact_packed_records: bool = False  # one CIDX document per context (pre-packed corpora)
    shuffle: bool = True
    num_workers: int = 2  # forked collation workers; 0 = synchronous
    prefetch: int = 4  # host batches in flight
    device_prefetch: int = 2  # device-resident step batches staged ahead
    eos_id: int | None = None
    document_masking: bool = False  # segment_ids/positions from eos_id; EOS-position targets leave the loss
    synthetic_kind: Literal["markov", "uniform"] = "markov"
    synthetic_follow_prob: float = 0.9
    synthetic_doc_length: int = 0  # > 0 inserts eos_id every N tokens
    synthetic_samples: int = 65_536
    # data.source = "stream": raw text read, tokenized and packed on the fly (jla.data.streaming).
    # Tokenization runs on the device-prefetch thread; num_workers/prefetch apply to cidx/synthetic.
    stream_files: tuple[str, ...] = ()  # local .jsonl / .parquet files, read in this order
    hf_dataset: str = ""  # or a Hugging Face dataset, streamed (optional 'datasets' extra)
    hf_name: str | None = None  # its configuration name
    hf_split: str = "train"
    hf_data_files: tuple[str, ...] = ()
    text_field: str = "text"
    tokenizer: str = ""  # "byte" (UTF-8 bytes, needs eos_id >= 256) or a Hugging Face tokenizer name/path
    tokenizer_local_files_only: bool = False
    packing: Literal["concat_split", "first_fit"] = "concat_split"
    pack_bins: int = 0  # first_fit bins per flush; 0 = this process's rows per batch
    stream_read_ahead: int = 1024  # documents tokenized per tokenizer call (at most 8 MiB of text)
    # data.source = "sft": SFT corpora (CIDX + loss-mask sidecar, see jla.data.sft); several paths are pooled.
    sft_packing: Literal["bfd", "single"] = "bfd"  # whole examples per row (best-fit decreasing) or one per row
    sft_overlong: Literal["truncate", "drop"] = "truncate"  # examples longer than seq_len + 1 tokens
    sft_completion_only: bool = True  # loss on the corpus' masked (assistant) targets only; False = every token
    sft_pad_id: int = 0  # token id filling unused row positions (never a target)

    def __post_init__(self) -> None:
        if self.source == "stream":
            _require(
                bool(self.stream_files) != bool(self.hf_dataset),
                "data.source = 'stream' needs exactly one of data.stream_files and data.hf_dataset",
            )
            _require(bool(self.tokenizer.strip()), "data.source = 'stream' needs data.tokenizer")
            _require(
                self.tokenizer != "byte" or (self.eos_id is not None and self.eos_id >= 256),
                "data.tokenizer = 'byte' needs data.eos_id >= 256 (ids 0-255 are bytes)",
            )
            _require(
                all(path.endswith((".jsonl", ".parquet")) for path in self.stream_files),
                "data.stream_files entries must end in .jsonl or .parquet",
            )
            _require(
                not (self.hf_dataset and self.shuffle),
                "data.shuffle permutes data.stream_files per epoch; set data.shuffle = false with data.hf_dataset",
            )
            _require(self.pack_bins >= 0, "data.pack_bins must be >= 0")
            _require(self.stream_read_ahead >= 1, "data.stream_read_ahead must be >= 1")
            _require(bool(self.text_field), "data.text_field must be non-empty")
        else:
            _require(
                not self.stream_files and not self.hf_dataset,
                "data.stream_files and data.hf_dataset are only used with data.source = 'stream'",
            )
        if self.source == "cidx":
            _require(bool(self.paths), "data.paths is required for data.source = 'cidx'")
        elif self.source != "sft":  # sft paths are checked in the sft block below
            _require(not self.paths, "data.paths is only used with data.source = 'cidx' or 'sft'")
        _require(
            not self.weights or len(self.weights) == len(self.paths),
            "data.weights must be empty or match data.paths",
        )
        _require(all(w > 0 for w in self.weights), "data.weights must be > 0")
        _require(self.eval_holdout_records >= 0, "data.eval_holdout_records must be >= 0")
        _require(
            not (self.eval_holdout_records and self.eval_path),
            "data.eval_path and data.eval_holdout_records are mutually exclusive",
        )
        _require(
            not self.eval_holdout_records or len(self.paths) == 1,
            "data.eval_holdout_records needs exactly one CIDX train path",
        )
        _require(self.num_workers >= 0, "data.num_workers must be >= 0")
        _require(self.prefetch >= 1 and self.device_prefetch >= 1, "data prefetch depths must be >= 1")
        _require(self.eos_id is None or self.eos_id >= 0, "data.eos_id must be >= 0")
        _require(
            not self.document_masking or self.eos_id is not None,
            "data.document_masking requires data.eos_id",
        )
        _require(0.0 <= self.synthetic_follow_prob <= 1.0, "data.synthetic_follow_prob must be in [0, 1]")
        _require(self.synthetic_doc_length >= 0, "data.synthetic_doc_length must be >= 0")
        _require(
            not self.synthetic_doc_length or self.eos_id is not None,
            "data.synthetic_doc_length requires data.eos_id",
        )
        _require(self.synthetic_samples >= 1, "data.synthetic_samples must be >= 1")
        _require(self.sft_pad_id >= 0, "data.sft_pad_id must be >= 0")
        if self.source == "sft":
            _require(bool(self.paths), "data.paths is required for data.source = 'sft'")
            _require(not self.weights, "data.weights are not used with data.source = 'sft' (all paths are pooled)")
            _require(
                not self.document_masking and not self.exact_packed_records,
                "data.source = 'sft' rows carry their own segment_ids, positions and loss_mask; "
                "leave data.document_masking and data.exact_packed_records off",
            )


@dataclass(frozen=True)
class MeshConfig:
    """Device mesh ``(dp, fsdp, tp)`` and logical-axis sharding rules.

    ``fsdp = -1`` takes every device not used by the other axes. ``dcn_*``
    sizes split each axis across slices/hosts (MaxText-style hybrid mesh).
    ``rules`` override the default logical-axis rules, e.g.
    ``[mesh.rules] experts = ["fsdp"]``; an empty list replicates that axis.
    """

    dp: int = 1
    fsdp: int = -1
    tp: int = 1
    dcn_dp: int = 1
    dcn_fsdp: int = 1
    dcn_tp: int = 1
    topology_aware: bool = True
    rules: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # Expert parallelism: MoE experts sharded over an "ep" mesh axis that also splits the batch
    # (set model.moe.expert_parallel_axis = "ep" for the all-to-all dispatch).
    ep: int = 1
    dcn_ep: int = 1
    # Pipeline parallelism (jla.distributed.pipeline): pp stages over the scanned layer groups,
    # GPipe schedule with pp_microbatches microbatches per forward pass (0 = one per stage).
    pp: int = 1
    dcn_pp: int = 1
    pp_microbatches: int = 0

    def __post_init__(self) -> None:
        _require(self.dp >= 1 and self.tp >= 1, "mesh.dp and mesh.tp must be >= 1")
        _require(self.fsdp >= 1 or self.fsdp == -1, "mesh.fsdp must be >= 1 or -1")
        for name in ("dcn_dp", "dcn_fsdp", "dcn_tp"):
            _require(getattr(self, name) >= 1, f"mesh.{name} must be >= 1")
        _require(self.ep >= 1 and self.dcn_ep >= 1, "mesh.ep and mesh.dcn_ep must be >= 1")
        _require(self.ep % self.dcn_ep == 0, f"mesh.ep={self.ep} is not divisible by mesh.dcn_ep={self.dcn_ep}")
        names = [name for name, _ in self.rules]
        _require(len(set(names)) == len(names), "mesh.rules names must be unique")
        for name, axes in self.rules:
            unknown = [axis for axis in axes if axis not in ("dp", "fsdp", "tp", "ep", "pp")]
            _require(not unknown, f"mesh.rules.{name}: unknown mesh axes {unknown}")
        _require(self.pp >= 1 and self.dcn_pp >= 1, "mesh.pp and mesh.dcn_pp must be >= 1")
        _require(self.pp % self.dcn_pp == 0, "mesh.pp must be divisible by mesh.dcn_pp")
        _require(self.pp_microbatches >= 0, "mesh.pp_microbatches must be >= 0 (0 = one per stage)")
        _require(self.pp > 1 or not any("pp" in axes for _, axes in self.rules), "mesh.rules use pp but mesh.pp = 1")


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-level settings applied before JAX initializes (see ``jla.launch_env``)."""

    xla_tuning_level: int = 0  # 0 off, 1 latency hiding, 2 + pipelined collectives
    xla_flags: tuple[str, ...] = ()  # extra XLA flags, merged after the tuning tier
    compile_cache_dir: str = ""  # persistent compilation cache; "" disables
    deterministic: bool = False  # XLA deterministic ops (and XLA attention): bitwise-reproducible GPU runs
    prng_impl: Literal["threefry2x32", "rbg", "unsafe_rbg"] = "threefry2x32"
    matmul_precision: Literal["default", "high", "highest"] = "default"
    distributed: bool = False  # call jax.distributed.initialize
    cluster_env: Literal["none", "auto", "slurm", "openmpi", "torchrun"] = "none"  # opt-in env discovery
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] = ()
    initialization_timeout: int = 300
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def __post_init__(self) -> None:
        _require(self.xla_tuning_level in (0, 1, 2), "runtime.xla_tuning_level must be 0, 1 or 2")
        _require(self.initialization_timeout >= 1, "runtime.initialization_timeout must be >= 1")
        _require(
            self.cluster_env == "none" or self.distributed,
            "runtime.cluster_env needs runtime.distributed = true",
        )
        for flag in self.xla_flags:
            _require(flag.startswith("--"), f"runtime.xla_flags entries must start with '--': {flag!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that defines a training run."""

    model: ModelConfig
    name: str = "experiment"
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "name must be non-empty")
        schedule_steps = self.optim.schedule_steps or self.train.max_steps
        _require(
            schedule_steps >= self.train.max_steps,
            "optim.schedule_steps must be 0 or >= train.max_steps",
        )
        _require(
            self.optim.warmup_steps <= schedule_steps,
            "optim.warmup_steps must be <= the schedule length",
        )
        _require(
            self.optim.schedule not in ("inv_sqrt", "noam") or self.optim.warmup_steps > 0,
            f"optim.schedule = {self.optim.schedule!r} needs optim.warmup_steps > 0",
        )
        if self.data.eos_id is not None:
            _require(
                self.data.eos_id < self.model.vocab_size,
                "data.eos_id must be < model.vocab_size",
            )
        if self.data.source == "cidx" and (self.train.eval_every or self.train.final_eval):
            _require(
                self.data.eval_path is not None or self.data.eval_holdout_records > 0,
                "evaluation (train.eval_every / train.final_eval) needs data.eval_path or "
                "data.eval_holdout_records",
            )
        if self.data.source == "stream" and (self.train.eval_every or self.train.final_eval):
            _require(
                self.data.eval_path is not None,
                "evaluation with data.source = 'stream' needs data.eval_path (a CIDX corpus, see `jla tokenize`)",
            )
        if self.data.source == "sft":
            _require(self.data.sft_pad_id < self.model.vocab_size, "data.sft_pad_id must be < model.vocab_size")
            _require(
                self.data.eval_path is not None
                or self.data.eval_holdout_records > 0
                or not (self.train.eval_every or self.train.final_eval),
                "evaluation (train.eval_every / train.final_eval) needs data.eval_path or "
                "data.eval_holdout_records",
            )
        expert_axis = getattr(getattr(self.model, "moe", None), "expert_parallel_axis", None)
        if expert_axis is not None:
            _require(
                expert_axis == "ep" and self.mesh.ep > 1,
                f"model.moe.expert_parallel_axis = {expert_axis!r} needs the 'ep' mesh axis with mesh.ep > 1 "
                f"(got mesh.ep = {self.mesh.ep}); unset it to train without expert parallelism",
            )

    @property
    def schedule_steps(self) -> int:
        return self.optim.schedule_steps or self.train.max_steps

    def replace(self, **changes: Any) -> ExperimentConfig:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible mapping (``None`` kept as ``null``)."""
        return _to_plain(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def to_toml(self) -> str:
        """TOML text that :func:`load_config` reads back to an equal config."""
        return dump_toml(self.to_dict())

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form; stable across processes and runs."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_json() + "\n" if destination.suffix == ".json" else self.to_toml()
        destination.write_text(text, encoding="utf-8")
        return destination


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load a ``.toml`` or ``.json`` experiment file and apply ``--set`` overrides."""
    return config_from_mapping(load_mapping(path), overrides)


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Read one bounded TOML/JSON object with strict values."""
    source = Path(path)
    size = source.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"config {source} exceeds {MAX_CONFIG_BYTES} bytes")
    text = source.read_bytes().decode("utf-8")
    suffix = source.suffix.lower()
    if suffix == ".toml":
        try:
            value: Any = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"invalid TOML in {source}: {exc}") from exc
    elif suffix == ".json":
        try:
            value = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    else:
        raise ValueError(f"config files must be .toml or .json, got {source}")
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a table/object at the top level")
    return value


def config_from_mapping(raw: Mapping[str, Any], overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Validate a raw mapping (as read from a file) into an :class:`ExperimentConfig`."""
    data = apply_overrides(raw, overrides)
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(ExperimentConfig)})
    if unknown:
        raise ValueError(f"unknown top-level key(s) {unknown}")
    if "model" not in data:
        raise ValueError("the [model] section is required")
    model = _model_from_mapping(data["model"])
    rest = {key: value for key, value in data.items() if key != "model"}
    return _from_mapping(ExperimentConfig, rest, "", preset={"model": model})


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a deep copy of ``raw`` with ``a.b.c=value`` overrides applied."""
    data = copy.deepcopy(dict(raw))
    for text in overrides:
        keys, value = parse_override(text)
        node = data
        for depth, key in enumerate(keys[:-1]):
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"--set {text}: {'.'.join(keys[: depth + 1])} is not a table")
            node = child
        node[keys[-1]] = value
    return data


def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``a.b.c=value`` into keys and a TOML-parsed value."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like section.key=value")
    path, raw_value = text.split("=", 1)
    keys = [key.strip() for key in path.split(".")]
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in keys):
        raise ValueError(f"override {text!r} has an invalid key path")
    raw_value = raw_value.strip()
    if raw_value.lower() in ("none", "null"):
        return keys, None
    try:
        value = tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError:
        if not re.fullmatch(r"[A-Za-z0-9_./:+-]+", raw_value):
            raise ValueError(f"override {text!r}: cannot parse value {raw_value!r}") from None
        value = raw_value  # bare word, e.g. data.source=cidx
    return keys, value


def _model_from_mapping(raw: Any) -> ModelConfig:
    if not isinstance(raw, Mapping):
        raise TypeError("[model] must be a table")
    values = dict(raw)
    preset = values.pop("preset", None)
    if preset is None:
        return _from_mapping(ModelConfig, values, "model")
    if not isinstance(preset, str):
        raise TypeError("model.preset must be a string")
    from jla.models import get_preset

    return _override(get_preset(preset), values, "model")


# ---------------------------------------------------------------------------
# Strict dataclass conversion
# ---------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite number {value!r} is not allowed")


def _type_hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _from_mapping(
    cls: type,
    raw: Any,
    path: str,
    *,
    preset: Mapping[str, Any] | None = None,
) -> Any:
    if not isinstance(raw, Mapping):
        raise TypeError(f"[{path or '<root>'}] must be a table, got {type(raw).__name__}")
    hints = _type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    if unknown:
        where = f"[{path}]" if path else "the top level"
        raise ValueError(f"unknown key(s) {unknown} in {where}; expected a subset of {names}")
    kwargs = dict(preset or {})
    for name, value in raw.items():
        kwargs[name] = _convert(value, hints[name], _join(path, name))
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"[{path or '<root>'}]: {exc}") from exc


def _override(base: Any, raw: Any, path: str) -> Any:
    """Apply a mapping of field overrides to an existing dataclass instance."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"[{path}] must be a table")
    hints = _type_hints(type(base))
    names = [f.name for f in dataclasses.fields(base)]
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in [{path}]; expected a subset of {names}")
    changes = {}
    for name, value in raw.items():
        current = getattr(base, name)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            changes[name] = _override(current, value, _join(path, name))
        else:
            changes[name] = _convert(value, hints[name], _join(path, name))
    return dataclasses.replace(base, **changes)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            if type(None) in args:
                return None
            raise TypeError(f"{path} may not be none")
        options = [arg for arg in args if arg is not type(None)]
        errors = []
        for option in options:
            try:
                return _convert(value, option, path)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError("; ".join(errors))
    if value is None:
        raise TypeError(f"{path} may not be none")
    if origin is Literal:
        if not any(value == arg and type(value) is type(arg) for arg in args):
            raise ValueError(f"{path} must be one of {list(args)}, got {value!r}")
        return value
    if origin is tuple:
        if isinstance(value, Mapping) and len(args) == 2 and args[1] is Ellipsis:
            # A tuple of (name, value) pairs may be written as a table.
            value = sorted(value.items())
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be an array, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(item, args[0], f"{path}[{i}]") for i, item in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{path} must have {len(args)} entries, got {len(value)}")
        return tuple(_convert(item, arg, f"{path}[{i}]") for i, (item, arg) in enumerate(zip(value, args)))
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        return _from_mapping(annotation, value, path)
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string, got {value!r}")
        return value
    raise TypeError(f"{path}: unsupported field type {annotation!r}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            if f.name == "rules" and isinstance(value, MeshConfig):
                out[f.name] = {name: list(axes) for name, axes in item}
            else:
                out[f.name] = _to_plain(item)
        return out
    if isinstance(value, (tuple, list)):
        return [_to_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"cannot serialize non-finite value {value!r}")
    return value


_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def dump_toml(data: Mapping[str, Any]) -> str:
    """Serialize nested mappings of scalars/arrays to TOML; ``None`` values are omitted."""
    lines: list[str] = []

    def key_text(key: str) -> str:
        return key if _BARE_KEY.fullmatch(key) else json.dumps(key)

    def value_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"TOML output cannot contain {value!r}")
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, tuple)):
            if any(item is None or isinstance(item, Mapping) for item in value):
                raise ValueError("TOML arrays may only hold scalars or arrays here")
            return "[" + ", ".join(value_text(item) for item in value) + "]"
        raise TypeError(f"cannot write {type(value).__name__} to TOML")

    def emit(prefix: str, table: Mapping[str, Any]) -> None:
        scalars = [(k, v) for k, v in table.items() if v is not None and not isinstance(v, Mapping)]
        tables = [(k, v) for k, v in table.items() if isinstance(v, Mapping)]
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars:
            lines.append(f"{key_text(key)} = {value_text(value)}")
        if prefix or scalars:
            lines.append("")
        for key, value in tables:
            emit(f"{prefix}.{key_text(key)}" if prefix else key_text(key), value)

    emit("", data)
    return "\n".join(lines).rstrip() + "\n"
