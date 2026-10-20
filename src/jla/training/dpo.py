# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MaxText (Apache-2.0), src/maxtext/configs/post_train/dpo.yml (options and defaults:
# beta, label smoothing, max prompt length), and from TRL (Apache-2.0), trl/trainer/dpo_trainer.py
# (the sigmoid, IPO and hinge losses and the reward metrics).
"""Direct Preference Optimization (DPO, Rafailov et al. 2023) on the compiled train step.

    jla dpo --config configs/post_train/dpo-tiny.toml --set dpo.beta=0.1

The experiment file is a normal ``jla train`` file plus a ``[dpo]`` table (:class:`DPOConfig`).
The policy starts from ``dpo.init_checkpoint`` (a ``jla`` checkpoint) or, without one, from the
seeded initialization; the frozen reference model is a copy of that starting point.

One optimizer step is :func:`jla.training.step.build_train_step` on a
:class:`PolicyAndReference` module whose ``trainable`` filter selects the policy only, so the
reference enters the executable as a traced, sharded input with no gradient, optimizer state or
weight decay, and its log-probabilities are computed in the same step. Batches hold ``2P`` rows
per microbatch: chosen and rejected sequences of each pair interleaved (see
:mod:`jla.data.preference`). Per-sequence log-probabilities sum
:func:`jla.ops.token_logprobs.linear_token_logprobs` over response tokens (prompt and padding
are masked), so ``[N, V]`` logits are never materialized. With
``h = (log pi(c) - log pi(r)) - (log ref(c) - log ref(r))`` the per-pair losses are

* ``sigmoid``: ``-(1 - eps) log sigmoid(beta h) - eps log sigmoid(-beta h)`` (``eps`` =
  ``label_smoothing``, conservative DPO; ``log 2`` when the policy equals the reference);
* ``ipo``: ``(h - 1 / (2 beta))**2`` on length-normalized log-probabilities (Azar et al. 2023);
* ``hinge``: ``max(0, 1 - beta h)`` (SLiC).

The step's ``loss`` is the mean over pairs (its ``tokens`` metric counts valid pairs); metrics
add ``reward_chosen``/``reward_rejected`` (``beta * (log pi - log ref)``), ``reward_margin``,
``reward_accuracy`` (chosen reward > rejected reward), the policy log-probabilities and the
response length. Checkpoints hold the policy (loadable like any ``jla`` model checkpoint), the
optimizer state, the data position and the RNG key; a resume rebuilds the reference from the
same source and continues the identical data stream.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jla.config import DataConfig, ExperimentConfig, _from_mapping, apply_overrides, config_from_mapping, dump_toml, load_mapping

__all__ = [
    "DPOConfig",
    "DPOResult",
    "DPOTrainer",
    "PolicyAndReference",
    "build_dpo_eval_step",
    "dpo_config_hash",
    "dpo_loss",
    "load_dpo_config",
    "load_model_weights",
    "main",
    "preference_terms",
    "sequence_logps",
]

logger = logging.getLogger(__name__)

LossType = Literal["sigmoid", "ipo", "hinge"]
IGNORE_INDEX = -100
# DataConfig fields the DPO loader uses; every other [data] field must keep its default.
_DATA_FIELDS_USED = ("shuffle", "num_workers", "prefetch", "device_prefetch")


@dataclass(frozen=True)
class DPOConfig:
    """The ``[dpo]`` table of a ``jla dpo`` experiment file."""

    beta: float = 0.1
    loss_type: LossType = "sigmoid"
    label_smoothing: float = 0.0  # conservative DPO (sigmoid loss only)
    train_path: str | None = None  # preference JSONL; None = synthetic toy pairs
    eval_path: str | None = None  # None with synthetic data = held-out synthetic pairs
    tokenizer: str | None = None  # HuggingFace tokenizer for text records (needs `transformers`)
    max_prompt_len: int = 0  # 0 = train.seq_len // 2; longer prompts keep their last tokens
    pad_id: int = 0
    eos_id: int | None = None  # appended to every response when set
    init_checkpoint: str | None = None  # jla checkpoint (run dir/checkpoints, step dir or .safetensors)
    synthetic_pairs: int = 4096

    def __post_init__(self) -> None:
        if not (self.beta > 0 and math.isfinite(self.beta)):
            raise ValueError("dpo.beta must be a finite number > 0")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("dpo.label_smoothing must be in [0, 0.5)")
        if self.label_smoothing and self.loss_type != "sigmoid":
            raise ValueError("dpo.label_smoothing applies to loss_type = 'sigmoid' only")
        if self.max_prompt_len < 0 or self.pad_id < 0 or (self.eos_id is not None and self.eos_id < 0):
            raise ValueError("dpo.max_prompt_len, dpo.pad_id and dpo.eos_id must be >= 0")
        if self.synthetic_pairs < 1:
            raise ValueError("dpo.synthetic_pairs must be >= 1")
        if self.eval_path is not None and self.train_path is None:
            raise ValueError("dpo.eval_path needs dpo.train_path (synthetic runs evaluate on held-out synthetic pairs)")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _check_supported(config: ExperimentConfig, dpo: DPOConfig) -> None:
    train = config.train
    problems = []
    if train.label_smoothing:
        problems.append("train.label_smoothing (token smoothing) does not apply; use dpo.label_smoothing")
    if train.z_loss:
        problems.append("train.z_loss is not part of the DPO objective")
    if train.ce_grad_in_forward:
        problems.append("train.ce_grad_in_forward (and ce_fp8) apply to the LM cross entropy; jla dpo scores "
                        "token log-probabilities (jla.ops.token_logprobs)")
    if config.model.fp8 == "delayed":  # the step fails to lower with the frozen reference's delayed-scaling state
        problems.append("model.fp8 = 'delayed' is not supported by jla dpo yet; use 'current' or 'off'")
    if train.profile_steps or train.graph_health:
        problems.append("the profiler window (train.profile_*) is not supported by jla dpo")
    if train.time_limit_minutes:
        problems.append("train.time_limit_minutes is not supported by jla dpo")
    if train.dispatch_ahead:
        problems.append("train.dispatch_ahead is not supported by jla dpo (its loop reads every log step first)")
    if getattr(train, "grad_reduce", "micro_step") != "micro_step":
        problems.append("train.grad_reduce must be 'micro_step' for jla dpo (its step reduces the gradients per "
                        "micro-step)")
    if getattr(train, "update_guard", "cond") != "cond":
        problems.append("train.update_guard must be 'cond' for jla dpo (its step guards the update with lax.cond)")
    if config.model.mtp is not None:
        problems.append("models with MTP heads are not supported by jla dpo (the heads would only decay)")
    if config.model.embed_dropout or config.model.resid_dropout:
        problems.append("model dropout must be 0 (the policy would score sequences differently from its reference)")
    # TrainConfig.loggers comes with the wandb/tensorboard sinks; jla dpo writes metrics.jsonl only.
    if tuple(getattr(train, "loggers", ("jsonl",))) != ("jsonl",):
        problems.append("train.loggers other than 'jsonl' are not supported by jla dpo")
    defaults = DataConfig()
    changed = [f.name for f in dataclasses.fields(DataConfig) if f.name not in _DATA_FIELDS_USED and getattr(config.data, f.name) != getattr(defaults, f.name)]
    if changed:
        problems.append(f"[data] {changed}: preference data comes from [dpo] (train_path/eval_path)")
    if dpo.train_path is not None and dpo.eval_path is None and (train.eval_every or train.final_eval):
        problems.append("evaluation (train.eval_every / train.final_eval) needs dpo.eval_path")
    if dpo.max_prompt_len and dpo.max_prompt_len >= train.seq_len:
        problems.append(f"dpo.max_prompt_len must be < train.seq_len = {train.seq_len}")
    for name, value in (("pad_id", dpo.pad_id), ("eos_id", dpo.eos_id)):
        if value is not None and value >= config.model.vocab_size:
            problems.append(f"dpo.{name} must be < model.vocab_size")
    if problems:
        raise ValueError("unsupported DPO configuration: " + "; ".join(problems))


def load_dpo_config(path: str | Path, overrides: Sequence[str] = ()) -> tuple[ExperimentConfig, DPOConfig]:
    """Read an experiment file with a ``[dpo]`` table; ``--set dpo.beta=...`` overrides work too."""
    raw = apply_overrides(load_mapping(path), overrides)
    dpo_raw = raw.pop("dpo", {})
    config = config_from_mapping(raw)
    dpo = _from_mapping(DPOConfig, dpo_raw, "dpo")
    _check_supported(config, dpo)
    return config, dpo


def dpo_config_toml(config: ExperimentConfig, dpo: DPOConfig) -> str:
    return dump_toml({**config.to_dict(), "dpo": dpo.to_dict()})


def dpo_config_hash(config: ExperimentConfig, dpo: DPOConfig) -> str:
    canonical = json.dumps({"experiment": config.to_dict(), "dpo": dpo.to_dict()}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model wrapper and loss
# ---------------------------------------------------------------------------


class PolicyAndReference(eqx.Module):
    """The trained policy and its frozen reference (same architecture)."""

    policy: Any
    reference: Any


#: ``trainable`` filter for :func:`jla.training.step.build_train_step`: the policy's float arrays.
TRAINABLE = PolicyAndReference(policy=eqx.is_inexact_array, reference=False)


def sequence_logps(
    model: Any,
    batch: Mapping[str, jax.Array],
    key: jax.Array | None,
    *,
    inference: bool,
    vocab_chunk: int = 4096,
    compute_dtype: Any = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array, Any]:
    """``(sum of response-token log-probs [R], response tokens [R], LMOutput)`` for ``[R, T]`` rows."""
    from jla.ops.token_logprobs import linear_token_logprobs

    out = model(
        batch["input_ids"],
        positions=batch.get("positions"),
        segment_ids=batch.get("segment_ids"),
        key=key,
        inference=inference,
    )
    targets = jnp.where(batch["loss_mask"].astype(bool), batch["labels"].astype(jnp.int32), IGNORE_INDEX)
    hidden = out.hidden
    logp = linear_token_logprobs(
        hidden.reshape(-1, hidden.shape[-1]),
        model.lm_head_weight(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        vocab_chunk=vocab_chunk,
        compute_dtype=compute_dtype,
    ).reshape(targets.shape)
    counts = jnp.sum(targets != IGNORE_INDEX, axis=-1).astype(jnp.float32)
    return jnp.sum(logp, axis=-1), counts, out


def preference_terms(
    policy_logps: jax.Array,
    reference_logps: jax.Array,
    counts: jax.Array,
    *,
    beta: float,
    loss_type: str,
    label_smoothing: float = 0.0,
) -> dict[str, jax.Array]:
    """Summed DPO loss and metrics of ``[2P]`` interleaved (chosen, rejected) rows.

    A pair counts (``tokens``) when both its rows have response tokens; ``*_sum`` entries are
    sums over counted pairs.
    """
    if policy_logps.shape[0] % 2:
        raise ValueError(f"rows must hold whole (chosen, rejected) pairs; got {policy_logps.shape[0]} rows")
    pi = policy_logps.astype(jnp.float32).reshape(-1, 2)
    ref = reference_logps.astype(jnp.float32).reshape(-1, 2)
    n = counts.astype(jnp.float32).reshape(-1, 2)
    # Rewards and log-prob metrics use the summed log-probabilities for every loss type (TRL's
    # metrics); only the IPO loss sees length-normalized ones.
    pi_sum, ref_sum = pi, ref
    if loss_type == "ipo":
        pi, ref = pi / jnp.maximum(n, 1.0), ref / jnp.maximum(n, 1.0)
    h = (pi[:, 0] - pi[:, 1]) - (ref[:, 0] - ref[:, 1])
    if loss_type == "sigmoid":
        losses = -(1.0 - label_smoothing) * jax.nn.log_sigmoid(beta * h) - label_smoothing * jax.nn.log_sigmoid(-beta * h)
    elif loss_type == "ipo":
        losses = (h - 1.0 / (2.0 * beta)) ** 2
    elif loss_type == "hinge":
        losses = jax.nn.relu(1.0 - beta * h)
    else:
        raise ValueError(f"unknown DPO loss_type {loss_type!r}")
    chosen = beta * (pi_sum[:, 0] - ref_sum[:, 0])
    rejected = beta * (pi_sum[:, 1] - ref_sum[:, 1])
    w = ((n[:, 0] > 0) & (n[:, 1] > 0)).astype(jnp.float32)
    return {
        "loss_sum": jnp.sum(w * losses),
        "tokens": jnp.sum(w),
        "reward_chosen_sum": jnp.sum(w * chosen),
        "reward_rejected_sum": jnp.sum(w * rejected),
        "reward_margin_sum": jnp.sum(w * (chosen - rejected)),
        "reward_accuracy_sum": jnp.sum(w * (chosen > rejected).astype(jnp.float32)),
        "logp_chosen_sum": jnp.sum(w * pi_sum[:, 0]),
        "logp_rejected_sum": jnp.sum(w * pi_sum[:, 1]),
        "response_tokens_sum": jnp.sum(w * (n[:, 0] + n[:, 1])),
    }


def dpo_loss(
    dpo: DPOConfig,
    *,
    vocab_chunk: int = 4096,
    compute_dtype: str = "bfloat16",
    aux_loss_cap: float = 100.0,
) -> Callable[[PolicyAndReference, Mapping[str, jax.Array], jax.Array], tuple[jax.Array, dict[str, jax.Array]]]:
    """``loss_fn(pair, micro_batch, key)`` for :func:`jla.training.step.build_train_step`.

    ``objective_sum = sum of pair losses + aux_loss * pairs`` (the policy's weighted MoE
    auxiliary loss, capped like the LM loss), normalized by the step's total pairs.
    """
    dtype = jnp.dtype(compute_dtype)

    def loss_fn(pair: PolicyAndReference, batch: Mapping[str, jax.Array], key: jax.Array):
        policy_logps, counts, out = sequence_logps(pair.policy, batch, key, inference=False, vocab_chunk=vocab_chunk, compute_dtype=dtype)
        reference_logps, _, _ = sequence_logps(pair.reference, batch, None, inference=True, vocab_chunk=vocab_chunk, compute_dtype=dtype)
        terms = preference_terms(
            policy_logps,
            jax.lax.stop_gradient(reference_logps),
            counts,
            beta=dpo.beta,
            loss_type=dpo.loss_type,
            label_smoothing=dpo.label_smoothing,
        )
        aux_loss = jnp.asarray(out.aux_loss, jnp.float32)
        if aux_loss_cap > 0:
            aux_loss = jnp.where(jnp.isfinite(aux_loss), jnp.clip(aux_loss, -aux_loss_cap, aux_loss_cap), aux_loss)
        objective = terms["loss_sum"] + aux_loss * terms["tokens"]
        return objective, {**terms, "aux_loss": aux_loss}

    return loss_fn


def build_dpo_eval_step(
    dpo: DPOConfig,
    *,
    vocab_chunk: int = 4096,
    compute_dtype: str = "bfloat16",
    batch_sharding: jax.sharding.Sharding | None = None,
) -> Callable[[PolicyAndReference, dict[str, jax.Array]], dict[str, jax.Array]]:
    """Jitted DPO terms of one ``[2P, T]`` batch (policy and reference in inference mode)."""
    dtype = jnp.dtype(compute_dtype)

    def eval_step(pair: PolicyAndReference, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        if batch_sharding is not None:
            batch = {name: jax.lax.with_sharding_constraint(value, batch_sharding) for name, value in batch.items()}
        policy_logps, counts, _ = sequence_logps(pair.policy, batch, None, inference=True, vocab_chunk=vocab_chunk, compute_dtype=dtype)
        reference_logps, _, _ = sequence_logps(pair.reference, batch, None, inference=True, vocab_chunk=vocab_chunk, compute_dtype=dtype)
        return preference_terms(
            policy_logps, reference_logps, counts, beta=dpo.beta, loss_type=dpo.loss_type, label_smoothing=dpo.label_smoothing
        )

    return eqx.filter_jit(eval_step)


# ---------------------------------------------------------------------------
# Weights, data
# ---------------------------------------------------------------------------


def load_model_weights(path: str | Path, template: Any) -> Any:
    """Load model weights into ``template`` (its arrays give the target shardings).

    ``path``: a ``.safetensors`` file, a committed ``step_NNNNNNNN`` directory, or a checkpoint
    directory (its latest committed step). Shapes and dtypes must match exactly.
    """
    from jla.training import checkpointing as ckpt

    source = Path(path)
    arrays = eqx.filter(template, eqx.is_array)
    if source.is_file():
        restored = ckpt.load_pytree(source, arrays)
    else:
        if not source.is_dir():
            raise FileNotFoundError(f"init checkpoint {source} does not exist")
        step_dir = source
        if not (source / "manifest.json").exists():
            for candidate in (source, source / "checkpoints"):
                steps = ckpt.CheckpointManager(candidate, asynchronous=False).committed_steps() if candidate.is_dir() else []
                if steps:
                    step_dir = ckpt.CheckpointManager(candidate, asynchronous=False).step_dir(steps[-1])
                    break
            else:
                raise FileNotFoundError(f"no committed checkpoint under {source}")
        integrity = ckpt.verify_checkpoint(step_dir).require_valid()
        if integrity.payload_format == "safetensors":
            restored = ckpt.load_pytree(step_dir / "model.safetensors", arrays)
        elif integrity.payload_format == "sharded":
            files = sorted(step_dir.glob("model.p*-of-*.safetensors"))
            flat = ckpt._restore_from_files(files, ckpt.flatten_arrays(arrays), "model", True, chunked=True)
            restored = ckpt._unflatten_into(arrays, flat)
        else:
            raise ValueError(f"init checkpoint format {integrity.payload_format!r} is not supported (use safetensors or sharded)")
    return eqx.combine(restored, eqx.filter(template, eqx.is_array, inverse=True))


def _copy_arrays(tree: Any) -> Any:
    """Distinct buffers with the same shardings (policy and reference are both donated to the step)."""
    return jax.tree.map(lambda x: jax.device_put(x, x.sharding, may_alias=False) if eqx.is_array(x) else x, tree)


def _datasets(config: ExperimentConfig, dpo: DPOConfig, encode: Any) -> tuple[Any, Any]:
    from jla.data.preference import PreferenceDataset, SyntheticPreferenceDataset, hf_encoder, load_preference_jsonl

    seq_len, vocab = config.train.seq_len, config.model.vocab_size
    max_prompt_len = dpo.max_prompt_len or seq_len // 2
    if dpo.train_path is None:
        common = {"seed": config.train.seed, "max_prompt_len": max_prompt_len, "pad_id": dpo.pad_id}
        train = SyntheticPreferenceDataset(vocab, seq_len, num_pairs=dpo.synthetic_pairs, **common)
        held_out = SyntheticPreferenceDataset(vocab, seq_len, num_pairs=dpo.synthetic_pairs, offset=dpo.synthetic_pairs, **common)
        return train, held_out
    if encode is None and dpo.tokenizer is not None:
        encode = hf_encoder(dpo.tokenizer)
    options = {"max_prompt_len": max_prompt_len, "pad_id": dpo.pad_id, "eos_id": dpo.eos_id, "vocab_size": vocab}
    train = PreferenceDataset(load_preference_jsonl(dpo.train_path, encode=encode), seq_len, **options)
    held_out = None
    if dpo.eval_path is not None:
        held_out = PreferenceDataset(load_preference_jsonl(dpo.eval_path, encode=encode), seq_len, **options)
    return train, held_out


def _data_fingerprint(config: ExperimentConfig, dpo: DPOConfig) -> str:
    source: Any = "synthetic"
    if dpo.train_path is not None:
        stat = Path(dpo.train_path).stat()
        source = [str(Path(dpo.train_path).resolve()), stat.st_size]
    payload = {
        "source": source,
        "tokenizer": dpo.tokenizer,
        "max_prompt_len": dpo.max_prompt_len,
        "pad_id": dpo.pad_id,
        "eos_id": dpo.eos_id,
        "synthetic_pairs": dpo.synthetic_pairs,
        "shuffle": config.data.shuffle,
        "seq_len": config.train.seq_len,
        "seed": config.train.seed,
        "vocab_size": config.model.vocab_size,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class DPOResult:
    step: int
    pair: PolicyAndReference
    opt_state: Any
    rng_key: jax.Array
    data_state: Any
    stop_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)
    eval_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def policy(self) -> Any:
        return self.pair.policy


class DPOTrainer:
    """Runs DPO for an experiment config and a :class:`DPOConfig`.

    ``train.micro_batch_per_replica`` counts preference *pairs* per data-parallel replica (each
    pair is two rows); ``train.seq_len`` is the row length. ``model_builder``/``logical_axes_fn``/
    ``devices`` are as for :class:`jla.training.trainer.Trainer`; ``encode(text, add_special)``
    replaces the HuggingFace tokenizer for text records.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dpo: DPOConfig,
        *,
        model_builder: Callable[..., Any] | None = None,
        logical_axes_fn: Callable[[Any], Any] | None = None,
        devices: list[Any] | None = None,
        encode: Callable[[str, bool], list[int]] | None = None,
    ) -> None:
        from jla.training.trainer import Trainer

        _check_supported(config, dpo)
        self._lm = Trainer(config, model_builder=model_builder, logical_axes_fn=logical_axes_fn, devices=devices)
        self.config = self._lm.config  # with runtime.deterministic applied
        self.dpo = dpo
        self.encode = encode
        self.devices = self._lm.devices
        self.primary = jax.process_index() == 0
        self.output_dir = Path(config.train.output_dir)

    def _log(self, message: str, *args: Any) -> None:
        if self.primary:
            logger.info(message, *args)

    def run(self) -> DPOResult:
        from jla.data.loader import BatchLoader
        from jla.data.pipeline import DataState, DevicePrefetcher
        from jla.data.preference import PairRows
        from jla.distributed.init import process_any
        from jla.distributed.kernel_sharding import use_rules as kernel_sharding_rules
        from jla.distributed.mesh import build_mesh, plan_from_config
        from jla.distributed.sharding import (
            grouped_batch_sharding,
            make_global_batch,
            opt_state_shardings,
            process_row_range,
            replicated,
            resolve_rules,
        )
        from jla.training.checkpointing import CheckpointManager, wait_for_pending_checkpoints
        from jla.training.metrics_sink import MetricSink, atomic_write_json
        from jla.training.optimizers import _path_string, optimizer_from_config
        from jla.training.runtime import audit_parameter_sharding, compile_preflight, write_sharding_dump
        from jla.training.schedulers import create_schedule, create_schedule_fn
        from jla.training.step import build_train_step
        from jla.training.trainer import _frozen_gc
        from jla.training.telemetry import TelemetryCollector, TelemetryConfig

        cfg, dpo, train = self.config, self.dpo, self.config.train
        started = time.time()
        if self.primary:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_from_config(cfg.mesh, len(self.devices))
        mesh = build_mesh(plan, self.devices, topology_aware=cfg.mesh.topology_aware)
        rules = resolve_rules(cfg.mesh.rules)
        pairs_per_batch = train.micro_batch_per_replica * plan.data_parallel_size
        batch_shape = (train.grad_accum, 2 * pairs_per_batch, train.seq_len)
        batch_sharding = grouped_batch_sharding(mesh, rules)
        eval_sharding = grouped_batch_sharding(mesh, rules, grouped=False)
        rows = process_row_range(batch_sharding, batch_shape, axis=1)
        if rows[0] % 2 or rows[1] % 2:
            raise ValueError(f"process rows {rows} split a preference pair")
        pair_rows = (rows[0] // 2, rows[1] // 2)
        config_hash = dpo_config_hash(cfg, dpo)

        init_key, train_key = jax.random.split(jax.random.key(train.seed))
        policy = self._lm._init_model(mesh, rules, init_key)
        if dpo.init_checkpoint is not None:
            policy = load_model_weights(dpo.init_checkpoint, policy)
            self._log("policy and reference initialized from %s", dpo.init_checkpoint)
        sharding_report = self._lm._check_params(eqx.filter(policy, eqx.is_inexact_array), mesh)
        pair = PolicyAndReference(policy=policy, reference=_copy_arrays(policy))
        params = eqx.filter(pair, TRAINABLE)
        if cfg.optim.trainable_patterns:
            # optim.trainable_patterns see the paths of this module ("policy/..."); a pattern that
            # matches nothing would freeze the whole policy silently.
            paths = [_path_string(path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
            unmatched = [p for p in cfg.optim.trainable_patterns if not any(re.search(p, text) for text in paths)]
            if unmatched:
                example = paths[0] if paths else "policy/..."
                raise ValueError(f"optim.trainable_patterns {unmatched} match no policy parameter; DPO parameter paths look like {example!r}")
        param_count = sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))
        self._log("DPO policy: %.2fM trainable parameters; %d pairs x %d tokens x %d microbatches per step",
                  param_count / 1e6, pairs_per_batch, train.seq_len, train.grad_accum)

        schedule_args = {
            "warmup_steps": cfg.optim.warmup_steps,
            "min_lr_ratio": cfg.optim.min_lr_ratio,
            "stable_fraction": cfg.optim.stable_fraction,
            "decay_type": cfg.optim.decay_type,
            "d_model": cfg.model.d_model,
        }
        schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps, **schedule_args)
        host_lr = create_schedule_fn(cfg.optim.schedule, max_steps=cfg.schedule_steps, **schedule_args)
        optimizer = optimizer_from_config(cfg.optim, schedule)
        param_sharding_tree = jax.tree.map(lambda leaf: leaf.sharding, params)
        opt_shardings = opt_state_shardings(jax.eval_shape(optimizer.init, params), params, param_sharding_tree, mesh)
        opt_state = jax.jit(optimizer.init, out_shardings=opt_shardings)(params)
        rng = jax.device_put(train_key, replicated(mesh))

        manager = CheckpointManager(
            self.output_dir / "checkpoints",
            keep=train.keep_checkpoints,
            asynchronous=train.async_checkpoint,
            payload_format=train.checkpoint_format,
        )
        fingerprint = _data_fingerprint(cfg, dpo)
        reference_source = dpo.init_checkpoint or f"seed:{train.seed}"
        start_step = batches_consumed = tokens_seen = 0
        latest = manager.latest_step()
        if train.resume == "never" and latest is not None:
            raise FileExistsError(f"{manager.directory} already holds step {latest}; use another output_dir or resume")
        if train.resume == "required" and latest is None:
            raise FileNotFoundError(f"train.resume = 'required' but {manager.directory} has no committed checkpoint")
        if latest is not None:
            restored = manager.restore(latest, model=pair.policy, opt_state=opt_state, strict=True)
            saved_reference = _read_metadata(manager.step_dir(latest)).get("reference")
            if saved_reference != reference_source:
                raise ValueError(f"cannot resume: the reference model was {saved_reference!r}, now {reference_source!r}")
            state = DataState.from_dict(restored.data_state)
            state.require_compatible(global_batch_size=pairs_per_batch, seq_len=train.seq_len, fingerprint=fingerprint)
            pair = PolicyAndReference(policy=restored.model, reference=pair.reference)
            opt_state = restored.opt_state
            rng = jax.device_put(restored.rng_key, replicated(mesh))
            start_step, batches_consumed, tokens_seen = state.step, state.batches_consumed, state.tokens_seen
            self._log("resumed DPO from step %d (%d batches consumed)", start_step, batches_consumed)
        if start_step >= train.max_steps:
            raise ValueError(f"checkpoint step {start_step} already reaches train.max_steps={train.max_steps}")
        tokens_per_step = int(np.prod(batch_shape))

        def data_state(step: int) -> Any:
            return DataState(
                step=step,
                batches_consumed=batches_consumed + (step - start_step) * train.grad_accum,
                tokens_seen=tokens_seen + (step - start_step) * tokens_per_step,
                global_batch_size=pairs_per_batch,
                seq_len=train.seq_len,
                grad_accum=train.grad_accum,
                fingerprint=fingerprint,
            )

        step_fn = build_train_step(
            optimizer,
            dpo_loss(dpo, vocab_chunk=train.ce_vocab_chunk, compute_dtype=train.ce_compute_dtype, aux_loss_cap=train.aux_loss_cap),
            batch_sharding=batch_sharding,
            param_shardings=param_sharding_tree,
            opt_shardings=opt_shardings,
            scan_unroll=train.scan_unroll,
            trainable=TRAINABLE,
        )
        eval_fn = build_dpo_eval_step(dpo, vocab_chunk=train.ce_vocab_chunk, compute_dtype=train.ce_compute_dtype, batch_sharding=eval_sharding)
        train_set, eval_set = _datasets(cfg, dpo, self.encode)
        if len(train_set) < pairs_per_batch:
            raise ValueError(f"{len(train_set)} training pairs cannot fill one batch of {pairs_per_batch} pairs")
        loader = BatchLoader(
            train_set,
            pairs_per_batch,
            rows=pair_rows,
            seed=train.seed,
            shuffle=cfg.data.shuffle,
            num_workers=cfg.data.num_workers,
            prefetch=cfg.data.prefetch,
            skip_batches=batches_consumed,
            transform=PairRows(),
        )
        prefetch = DevicePrefetcher(loader, grad_accum=train.grad_accum, sharding=batch_sharding, global_shape=batch_shape, depth=cfg.data.device_prefetch)
        sink = MetricSink(self.output_dir / "metrics.jsonl", run_id=f"{cfg.name}-dpo-{config_hash[:12]}") if self.primary else None
        telemetry = TelemetryCollector(TelemetryConfig(max_consecutive_skips=train.max_consecutive_skips))
        history: list[dict[str, Any]] = []
        eval_history: list[dict[str, Any]] = []

        def run_eval(step: int, pair: PolicyAndReference) -> None:
            if eval_set is None:
                return
            available = len(eval_set) // pairs_per_batch
            if available == 0:
                raise ValueError(f"evaluation set of {len(eval_set)} pairs has no complete batch of {pairs_per_batch}")
            eval_loader = BatchLoader(eval_set, pairs_per_batch, rows=pair_rows, shuffle=False, transform=PairRows())
            totals: dict[str, jax.Array] = {}
            try:
                for _ in range(min(train.eval_batches, available)):
                    batch = make_global_batch(next(eval_loader), eval_sharding, (2 * pairs_per_batch, train.seq_len))
                    for name, value in eval_fn(pair, batch).items():
                        totals[name] = totals.get(name, 0.0) + value
            finally:
                eval_loader.close()
            host = {name: float(value) for name, value in jax.device_get(totals).items()}
            pairs = max(host.pop("tokens"), 1.0)
            metrics = {f"eval_{name.removesuffix('_sum')}": value / pairs for name, value in host.items()}
            metrics["eval_pairs"] = pairs
            eval_history.append({"step": step, **metrics})
            self._log("eval step %d | loss %.4f | accuracy %.3f | margin %.3f", step, metrics["eval_loss"],
                      metrics["eval_reward_accuracy"], metrics["eval_reward_margin"])
            if sink is not None:
                sink.append("eval", step=step, metrics=metrics)

        def save(step: int, pair: PolicyAndReference, opt_state: Any, rng: jax.Array) -> None:
            manager.save(
                step,
                model=pair.policy,
                opt_state=opt_state,
                data_state=data_state(step).to_dict(),
                rng_key=rng,
                metadata={"config_hash": config_hash, "name": cfg.name, "objective": "dpo", "reference": reference_source},
            )

        step = start_step
        last_saved = start_step
        stop_reason = "max_steps"
        pending: list[dict[str, jax.Array]] = []
        # As in jla train, the steps trace and run under the mesh and the kernel-sharding rules: kernels
        # XLA cannot partition (Mosaic GPU custom calls) run per shard instead of on gathered inputs, and
        # shard_map layers (expert parallelism, per-device MoE dispatch) see the mesh.
        # jax.set_mesh takes effect when called, so it is entered inside the try (closed in finally).
        contexts = contextlib.ExitStack()
        try:
            contexts.enter_context(jax.set_mesh(mesh))
            contexts.enter_context(kernel_sharding_rules(rules))
            batch = next(prefetch)
            if train.compile_preflight:
                report, compiled = compile_preflight("dpo_step", step_fn, pair, opt_state, batch, rng)
                if self.primary:
                    atomic_write_json(self.output_dir / "compile_preflight.json", report)
                if compiled is None:
                    raise RuntimeError(f"DPO step preflight {report['status']}: {report.get('error', report.get('reason'))}")
                step_fn.use_compiled(compiled)
            if train.sharding_audit and self.primary:
                policy_params = eqx.filter(pair.policy, eqx.is_inexact_array)
                audit = {**audit_parameter_sharding(policy_params), **sharding_report}
                write_sharding_dump(self.output_dir, mesh=mesh, model=policy_params, optimizer=opt_state, batch_sharding=batch_sharding, audit=audit)
            if self.primary:
                (self.output_dir / "config.toml").write_text(dpo_config_toml(cfg, dpo), encoding="utf-8")
            contexts.enter_context(_frozen_gc(train.gc_freeze))  # train.gc_freeze, as in jla train
            window_start = time.perf_counter()
            while step < train.max_steps:
                pair, opt_state, metrics, rng = step_fn(pair, opt_state, batch, rng)
                step += 1
                pending.append(metrics)
                is_log = train.log_every and step % train.log_every == 0
                is_eval = train.eval_every and step % train.eval_every == 0
                is_save = train.save_every and step % train.save_every == 0
                if step == train.max_steps or is_log or is_eval or is_save:
                    host = jax.device_get(pending)
                    elapsed = time.perf_counter() - window_start
                    first_step = step - len(host) + 1
                    for offset, step_metrics in enumerate(host):
                        telemetry_metrics = telemetry.on_step(first_step + offset, step_metrics)
                        if telemetry_metrics and sink is not None:
                            sink.append("telemetry", step=first_step + offset, metrics=telemetry_metrics)
                    window = {name: float(np.mean([float(m[name]) for m in host])) for name in host[-1]}
                    window["skipped_steps"] = float(sum(not bool(m["step_applied"]) for m in host))
                    window["skip_reason"] = float(int(np.bitwise_or.reduce([int(m["skip_reason"]) for m in host])))
                    window["lr"] = cfg.optim.lr * host_lr(step - 1)
                    window["step_time"] = elapsed / len(host)
                    window["pairs"] = window.pop("tokens")
                    history.append({"step": step, **window})
                    if sink is not None:
                        sink.append("train", step=step, metrics=window)
                    self._log("step %d/%d | loss %.4f | accuracy %.3f | margin %.3f | lr %.3e | grad %.3f | %.0f ms/step | skipped %d",
                              step, train.max_steps, window["loss"], window["reward_accuracy"], window["reward_margin"],
                              window["lr"], window["grad_norm"], 1000.0 * window["step_time"], int(window["skipped_steps"]))
                    pending.clear()
                    if process_any(telemetry.should_halt()):
                        stop_reason = f"halt: {telemetry.halt_reason}" if telemetry.should_halt() else "stop requested by another process"
                        self._log("stopping at step %d: %s", step, stop_reason)
                        break
                    if is_eval:
                        run_eval(step, pair)
                    if is_save:
                        save(step, pair, opt_state, rng)
                        last_saved = step
                    window_start = time.perf_counter()
                if step < train.max_steps:
                    batch = next(prefetch)
            if stop_reason == "max_steps" and train.final_eval and not (eval_history and eval_history[-1]["step"] == step):
                run_eval(step, pair)
            if (train.save_final or stop_reason != "max_steps") and step > last_saved:
                save(step, pair, opt_state, rng)
        finally:
            prefetch.close()
            wait_for_pending_checkpoints()
            contexts.close()
        if telemetry.should_halt() and self.primary:
            telemetry.recorder.dump(self.output_dir / "flight_recorder.json", reason=telemetry.halt_reason)
        if sink is not None:
            sink.append("end", step=step, metrics={"stop_reason": stop_reason, "seconds": time.time() - started})
        return DPOResult(
            step=step,
            pair=pair,
            opt_state=opt_state,
            rng_key=rng,
            data_state=data_state(step),
            stop_reason=stop_reason,
            history=history,
            eval_history=eval_history,
        )


def _read_metadata(step_dir: Path) -> dict[str, Any]:
    sidecar = json.loads((step_dir / "data_state.json").read_text(encoding="utf-8"))
    return dict(sidecar.get("metadata", {}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``jla dpo --config FILE [--set SECTION.KEY=VALUE ...] [--dry-run]``."""
    parser = argparse.ArgumentParser(prog="jla dpo", description="Direct preference optimization (experiment file + [dpo] table).")
    parser.add_argument("--config", required=True, help="experiment file (.toml or .json) with a [dpo] table")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the resolved config, then exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    config, dpo = load_dpo_config(args.config, args.set)
    if args.dry_run:
        sys.stdout.write(dpo_config_toml(config, dpo))
        print(f"# config_hash = {dpo_config_hash(config, dpo)}")
        return 0
    logging.basicConfig(level=config.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from jla.launch_env import launch

    launch(config.runtime)
    result = DPOTrainer(config, dpo).run()
    last = result.history[-1] if result.history else {}
    summary = {
        "step": result.step,
        "stop_reason": result.stop_reason,
        "loss": last.get("loss"),
        "reward_accuracy": last.get("reward_accuracy"),
        "eval": result.eval_history[-1] if result.eval_history else None,
    }
    print(json.dumps(summary))
    return 0 if result.stop_reason == "max_steps" else 1
