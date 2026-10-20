# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The compiled training step: one donated ``eqx.filter_jit`` executable.

``step(model, opt_state, batch, key) -> (model, opt_state, metrics, key)``

* ``batch`` arrays are ``[grad_accum, B, T]``; microbatches run in a
  ``lax.scan`` that accumulates float32 gradients and loss sums;
* gradients are normalized by the total number of valid target tokens of the
  whole step (token-weighted, independent of how tokens split across
  microbatches);
* a step whose loss, token count, auxiliary losses or gradient norm is
  non-finite, or whose batch has no valid targets, is rejected inside the
  executable with ``lax.cond``: parameters *and* optimizer state come back
  unchanged and ``metrics["skip_reason"]`` holds a bitmask of ``STEP_SKIP_*``;
* updated parameters and optimizer state are cast back to their stored dtypes;
* parameter and optimizer-state shardings are pinned so the step's outputs
  feed the next call without resharding or recompiling.

The language-model loss (:func:`lm_loss`) is linear cross entropy on
``LMOutput.hidden`` and ``model.lm_head_weight()`` (logits are never
materialized), plus z-loss, the model's auxiliary loss and the MTP loss.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jla.distributed.sharding import constrain
from jla.training.fp8 import (accumulate_gradients, apply_fp8_state, attach_fp8_weights, fp8_state_mask,
                               prepare_fp8_weights, split_fp8_state)
from jla.training.mixed_precision import restore_dtypes

__all__ = [
    "BATCH_KEYS",
    "STEP_SKIP_EMPTY_WEIGHTED_BATCH",
    "STEP_SKIP_NONE",
    "STEP_SKIP_NONFINITE_AUX",
    "STEP_SKIP_NONFINITE_GRAD",
    "STEP_SKIP_NONFINITE_LOSS",
    "STEP_SKIP_NONFINITE_WEIGHT",
    "LossConfig",
    "TrainStep",
    "build_eval_step",
    "build_train_step",
    "evaluate",
    "lm_loss",
    "resolve_scan_unroll",
    "validate_batch",
]

STEP_SKIP_NONE = 0
STEP_SKIP_NONFINITE_LOSS = 1 << 0
STEP_SKIP_NONFINITE_WEIGHT = 1 << 1
STEP_SKIP_NONFINITE_AUX = 1 << 2
STEP_SKIP_NONFINITE_GRAD = 1 << 3
STEP_SKIP_EMPTY_WEIGHTED_BATCH = 1 << 4

#: Keys a batch may carry; ``input_ids`` and ``labels`` are required.
BATCH_KEYS: tuple[str, ...] = ("input_ids", "labels", "loss_mask", "segment_ids", "positions")
_INTEGER_KEYS = ("input_ids", "labels", "segment_ids", "positions")
#: Auxiliary loss terms that enter the objective (checked for finiteness).
_OBJECTIVE_EXTRAS = ("aux_loss", "mtp_loss", "z_loss_sum")
#: Names the loss and step report themselves; model and MTP metrics may not reuse them.
_RESERVED_METRICS = frozenset(
    {"loss", "loss_sum", "z_loss", "z_loss_sum", "tokens", "aux_loss", "mtp_loss", "grad_norm", "step_applied", "skip_reason", "grad_accum", "spike_skip_reason"}
)

LossFn = Callable[[Any, Mapping[str, jax.Array], jax.Array], tuple[jax.Array, dict[str, jax.Array]]]


@dataclass(frozen=True)
class LossConfig:
    """Settings of :func:`lm_loss`."""

    label_smoothing: float = 0.0
    z_loss: float = 0.0
    vocab_chunk: int = 4096
    compute_dtype: str = "bfloat16"
    aux_loss_cap: float = 100.0
    ignore_index: int = -100
    impl: str = "xla"  # passed to the cross entropy as ``impl=`` unless "xla"
    # Gradient formed in the forward pass (jla.ops.cross_entropy.linear_cross_entropy_objective), by blocks of
    # ``row_chunk`` rows; needs impl "mosaic_gpu" or "auto".
    grad_in_forward: bool = False
    row_chunk: int = 8192
    fp8: bool = False  # with grad_in_forward: the two LM-head gradient GEMMs in FP8

    @classmethod
    def from_train_config(cls, train: Any) -> LossConfig:
        return cls(
            label_smoothing=train.label_smoothing,
            z_loss=train.z_loss,
            vocab_chunk=train.ce_vocab_chunk,
            compute_dtype=train.ce_compute_dtype,
            aux_loss_cap=train.aux_loss_cap,
            impl=getattr(train, "ce_impl", "xla"),
            grad_in_forward=getattr(train, "ce_grad_in_forward", False),
            row_chunk=getattr(train, "ce_row_chunk", 8192),
            fp8=getattr(train, "ce_fp8", False),
        )


def _cross_entropy_fn(cross_entropy: Callable[..., Any] | None) -> Callable[..., Any]:
    if cross_entropy is not None:
        return cross_entropy
    from jla.ops.cross_entropy import linear_cross_entropy

    return linear_cross_entropy


def _masked_targets(batch: Mapping[str, jax.Array], ignore_index: int) -> jax.Array:
    targets = batch["labels"].astype(jnp.int32)
    if "loss_mask" in batch:
        targets = jnp.where(batch["loss_mask"].astype(bool), targets, ignore_index)
    return targets


def _forward(model: Any, batch: Mapping[str, jax.Array], key: jax.Array | None, inference: bool) -> Any:
    return model(
        batch["input_ids"],
        positions=batch.get("positions"),
        segment_ids=batch.get("segment_ids"),
        key=key,
        inference=inference,
    )


def lm_loss(config: LossConfig, *, cross_entropy: Callable[..., Any] | None = None,
            forward: Callable[..., Any] | None = None) -> LossFn:
    """Language-model loss for :func:`build_train_step`.

    Returns ``(objective_sum, aux)`` for one microbatch, where
    ``objective_sum = CE_sum + z_loss * lse2_sum + (aux_loss + mtp_loss) * tokens``;
    the model's ``metrics`` and the MTP metrics are passed through (averaged
    over microbatches). ``cross_entropy`` defaults to
    :func:`jla.ops.cross_entropy.linear_cross_entropy`. ``forward(model, batch,
    key, inference=...)`` replaces the plain model call (e.g. the pipelined
    forward of :func:`jla.distributed.pipeline.make_pipeline_forward`).
    ``config.grad_in_forward`` uses
    :func:`jla.ops.cross_entropy.linear_cross_entropy_objective` instead, which forms the
    LM-head gradient in the forward pass (no logits recompute; not with ``cross_entropy``).
    """
    if config.grad_in_forward and (cross_entropy is not None or config.impl == "xla"):
        raise ValueError("LossConfig.grad_in_forward needs impl 'mosaic_gpu' or 'auto' and the built-in cross entropy")
    if config.fp8 and not config.grad_in_forward:
        raise ValueError("LossConfig.fp8 needs grad_in_forward")
    ce_fn = _cross_entropy_fn(cross_entropy)
    compute_dtype = jnp.dtype(config.compute_dtype)
    run_forward = _forward if forward is None else forward

    def loss_fn(model: Any, batch: Mapping[str, jax.Array], key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        out = run_forward(model, batch, key, inference=False)
        targets = _masked_targets(batch, config.ignore_index)
        hidden = out.hidden
        if config.grad_in_forward:
            from jla.ops.cross_entropy import linear_cross_entropy_objective

            ce = linear_cross_entropy_objective(
                hidden.reshape(-1, hidden.shape[-1]),
                model.lm_head_weight(),
                targets.reshape(-1),
                z_loss_weight=config.z_loss,
                ignore_index=config.ignore_index,
                vocab_chunk=config.vocab_chunk,
                row_chunk=config.row_chunk,
                label_smoothing=config.label_smoothing,
                compute_dtype=compute_dtype,
                impl=config.impl,
                fp8=config.fp8,
            )
        else:
            ce = ce_fn(
                hidden.reshape(-1, hidden.shape[-1]),
                model.lm_head_weight(),
                targets.reshape(-1),
                ignore_index=config.ignore_index,
                vocab_chunk=config.vocab_chunk,
                label_smoothing=config.label_smoothing,
                compute_dtype=compute_dtype,
                **({} if config.impl == "xla" else {"impl": config.impl}),
            )
        tokens = jnp.asarray(ce.num_valid, jnp.float32)
        aux_loss = jnp.asarray(out.aux_loss, jnp.float32)
        if config.aux_loss_cap > 0:
            # Keep non-finite values visible so the step rejects them.
            aux_loss = jnp.where(jnp.isfinite(aux_loss), jnp.clip(aux_loss, -config.aux_loss_cap, config.aux_loss_cap), aux_loss)
        # MTP sees the same document layout as the trunk, so no label crosses a packed boundary.
        # A model whose mtp_loss takes aux_loss_cap (LanguageModel) clips its MoE refiner aux loss like the trunk's.
        mtp_cap = {"aux_loss_cap": config.aux_loss_cap} if _takes_keyword(model.mtp_loss, "aux_loss_cap") else {}
        if config.grad_in_forward and _takes_keyword(model.mtp_loss, "objective"):
            # The MTP heads' cross entropy forms its gradient in the forward pass too (no z-loss there).
            mtp_cap["objective"] = _mtp_objective(config)
        mtp_loss, mtp_metrics = model.mtp_loss(
            out, batch["input_ids"], targets, positions=batch.get("positions"), segment_ids=batch.get("segment_ids"),
            **mtp_cap,
        )
        mtp_loss = jnp.asarray(mtp_loss, jnp.float32)
        loss_sum = jnp.asarray(ce.loss_sum, jnp.float32)
        z_loss_sum = jnp.asarray(ce.z_loss_sum, jnp.float32)
        # With grad_in_forward the two sums carry no gradient and the objective carries both terms' gradient.
        ce_objective = ce.objective_sum if config.grad_in_forward else loss_sum + config.z_loss * z_loss_sum
        objective = ce_objective + (aux_loss + mtp_loss) * tokens
        aux = {
            "loss_sum": loss_sum,
            "z_loss_sum": z_loss_sum,
            "tokens": tokens,
            "aux_loss": aux_loss,
            "mtp_loss": mtp_loss,
        }
        for name, value in {**out.metrics, **mtp_metrics}.items():
            if name in _RESERVED_METRICS or name.endswith("_sum"):
                raise ValueError(f"model metric {name!r} collides with a step metric name")
            aux[name] = jnp.asarray(value, jnp.float32)
        return objective, aux

    return loss_fn


def resolve_scan_unroll(scan_unroll: int, grad_accum: int) -> int:
    """``0`` unrolls the microbatch scan for ``grad_accum <= 4``; result is in ``[1, grad_accum]``."""
    if scan_unroll < 0 or grad_accum < 1:
        raise ValueError("scan_unroll must be >= 0 and grad_accum >= 1")
    if scan_unroll == 0:
        scan_unroll = grad_accum if grad_accum <= 4 else 1
    return max(1, min(scan_unroll, grad_accum))


def validate_batch(batch: Mapping[str, Any], batch_shape: tuple[int, ...] | None = None) -> None:
    """Check keys, shapes and dtypes of a batch (works on arrays and tracers)."""
    if not isinstance(batch, Mapping):
        raise TypeError(f"batch must be a mapping, got {type(batch).__name__}")
    missing = [key for key in ("input_ids", "labels") if key not in batch]
    if missing:
        raise KeyError(f"batch is missing keys {missing}")
    unexpected = sorted(set(batch) - set(BATCH_KEYS))
    if unexpected:
        raise KeyError(f"batch has unsupported keys {unexpected}; allowed {list(BATCH_KEYS)}")
    shape = tuple(batch["input_ids"].shape) if batch_shape is None else tuple(batch_shape)
    for key, value in batch.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"batch[{key!r}] has shape {tuple(value.shape)}, expected {shape}")
        dtype = jnp.dtype(value.dtype)
        if key in _INTEGER_KEYS and not jnp.issubdtype(dtype, jnp.integer):
            raise TypeError(f"batch[{key!r}] must be an integer array, got {dtype}")
        if key == "loss_mask" and not (jnp.issubdtype(dtype, jnp.integer) or dtype == jnp.bool_ or jnp.issubdtype(dtype, jnp.floating)):
            raise TypeError(f"batch['loss_mask'] must be boolean or numeric, got {dtype}")


class TrainStep:
    """``step(model, opt_state, batch, key)``; model, optimizer state and key are donated.

    :meth:`lower` gives the AOT lowering (used by the compile preflight) and
    :meth:`use_compiled` swaps in the resulting executable so the first call
    does not compile again.
    """

    def __init__(self, jitted: Any) -> None:
        self._jitted = jitted
        self._compiled: Any = None

    def __call__(self, model: Any, opt_state: Any, batch: dict[str, jax.Array], key: jax.Array):
        run = self._jitted if self._compiled is None else self._compiled
        return run(batch, model, opt_state, key)

    def lower(self, model: Any, opt_state: Any, batch: dict[str, jax.Array], key: jax.Array) -> Any:
        return self._jitted.lower(batch, model, opt_state, key)

    def use_compiled(self, compiled: Any) -> None:
        self._compiled = compiled


def build_train_step(
    optimizer: optax.GradientTransformation,
    loss_fn: LossFn,
    *,
    batch_sharding: jax.sharding.Sharding | None = None,
    param_shardings: Any = None,
    opt_shardings: Any = None,
    scan_unroll: int = 0,
    donate: bool = True,
    trainable: Any = eqx.is_inexact_array,
    grad_reduce: str = "micro_step",
    update_guard: str = "cond",
) -> TrainStep:
    """Compile-once train step (see the module docstring).

    ``loss_fn(model, micro_batch, key) -> (objective_sum, aux)``: ``aux`` maps
    names to float scalars and must contain ``loss_sum`` and ``tokens`` (the
    number of valid targets). Keys ending in ``_sum`` are reported divided by
    the step's total tokens (without the suffix); other keys are averaged over
    microbatches. ``param_shardings``/``opt_shardings`` must match the
    trainable part of the model and the optimizer state.
    ``trainable`` (an Equinox filter spec: a callable or a prefix tree of
    callables/booleans) selects the leaves that are differentiated and updated;
    the rest of the model (for example a frozen DPO reference model) enters the
    step as traced inputs and is returned unchanged.
    ``grad_reduce``: ``"micro_step"`` (the partitioner reduces every micro-step's
    gradient) or ``"step"``: on several devices the micro-steps run per device
    in a ``shard_map`` and the gradients are reduced once per step
    (:mod:`jla.training.grad_reduce`; needs ``batch_sharding`` and
    ``param_shardings``, and every mesh axis of size > 1 splitting the batch).
    ``update_guard``: how a rejected step keeps the parameters and optimizer
    state: ``"cond"`` (``lax.cond`` around the update) or ``"select"`` (the
    update computed and every leaf selected elementwise: the same rule; on GPU
    the update fuses differently, a few float32 ulps).
    """
    from jla.training.grad_reduce import GRAD_REDUCE_MODES, reduce_axes

    if grad_reduce not in GRAD_REDUCE_MODES:
        raise ValueError(f"grad_reduce must be one of {GRAD_REDUCE_MODES}; got {grad_reduce!r}")
    if update_guard not in ("cond", "select"):
        raise ValueError(f"update_guard must be 'cond' or 'select'; got {update_guard!r}")
    reduce_over = reduce_axes(batch_sharding) if grad_reduce == "step" else ()
    tx = optax.with_extra_args_support(optimizer)

    def train_step(batch: dict[str, jax.Array], model: Any, opt_state: Any, key: jax.Array):
        validate_batch(batch)
        if batch["input_ids"].ndim != 3:
            raise ValueError(f"train batches are [grad_accum, B, T]; got {batch['input_ids'].shape}")
        if batch_sharding is not None:
            batch = {name: jax.lax.with_sharding_constraint(value, batch_sharding) for name, value in batch.items()}
        params, static = eqx.partition(model, trainable)
        grad_accum = batch["input_ids"].shape[0]
        # Delayed-scaling FP8 weights quantized once for the whole step, outside the micro-step loop
        # (jla.training.fp8.prepare_fp8_weights; None without FP8 or with ModelConfig.fp8_weight_once = false).
        # Transposed copies for weights on one device (also in a process that sees several).
        fp8_weights = (prepare_fp8_weights(model, transposed=_on_one_device(param_shardings))
                       if fp8_state_mask(params) is not None else None)

        def micro_loss(p: Any, micro: Mapping[str, jax.Array], micro_key: jax.Array):
            with _attention_batch_partition(batch_sharding):
                return loss_fn(attach_fp8_weights(eqx.combine(p, static), fp8_weights), micro, micro_key)

        first = {name: value[0] for name, value in batch.items()}
        _, aux_shapes = jax.eval_shape(micro_loss, params, first, key)
        for name in ("loss_sum", "tokens"):
            if name not in aux_shapes:
                raise KeyError(f"loss_fn aux must contain {name!r}")
        for name, shape in aux_shapes.items():
            if shape.shape != ():
                raise ValueError(f"loss_fn aux[{name!r}] must be a scalar, got shape {shape.shape}")

        def body(carry, micro):
            grads_acc, aux_acc, carry_key = carry
            carry_key, micro_key = jax.random.split(carry_key)
            (_, aux), grads = jax.value_and_grad(micro_loss, has_aux=True)(params, micro, micro_key)
            grads_acc = accumulate_gradients(grads_acc, grads, fp8_mask)
            aux_acc = {name: aux_acc[name] + jnp.asarray(value, jnp.float32) for name, value in aux.items()}
            return (grads_acc, aux_acc, carry_key), None

        fp8_mask = fp8_state_mask(params)  # FP8 delayed-scaling state (jla.training.fp8), None if absent
        if reduce_over:  # grad_reduce="step": the micro-steps per device, one reduction (jla.training.grad_reduce)
            from jla.training.grad_reduce import accumulate_per_device

            grads, totals, key = accumulate_per_device(
                loss_fn, params, static, batch, key, axes=reduce_over, batch_sharding=batch_sharding,
                param_shardings=param_shardings, fp8_mask=fp8_mask, aux_names=tuple(aux_shapes),
                unroll=min(resolve_scan_unroll(scan_unroll, grad_accum), grad_accum),
            )
        else:
            grads0 = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
            aux0 = {name: jnp.zeros((), jnp.float32) for name in aux_shapes}
            (grads, totals, key), _ = jax.lax.scan(
                body,
                (grads0, aux0, key),
                batch,
                length=grad_accum,
                unroll=min(resolve_scan_unroll(scan_unroll, grad_accum), grad_accum),
            )

        grads, fp8_values = split_fp8_state(grads, fp8_mask)
        tokens = totals["tokens"]
        inv_tokens = 1.0 / jnp.maximum(tokens, 1.0)
        grads = jax.tree.map(lambda g: g * inv_tokens, grads)
        grad_norm = optax.tree.norm(grads)
        loss = totals["loss_sum"] * inv_tokens

        loss_finite = jnp.isfinite(totals["loss_sum"])
        tokens_finite = jnp.isfinite(tokens)
        extras = [totals[name] for name in _OBJECTIVE_EXTRAS if name in totals]
        aux_finite = jnp.all(jnp.isfinite(jnp.stack(extras))) if extras else jnp.asarray(True)
        grad_finite = jnp.isfinite(grad_norm)
        empty = tokens_finite & (tokens <= 0)
        applied = loss_finite & tokens_finite & aux_finite & grad_finite & ~empty
        skip_reason = (
            jnp.where(loss_finite, 0, STEP_SKIP_NONFINITE_LOSS)
            | jnp.where(tokens_finite, 0, STEP_SKIP_NONFINITE_WEIGHT)
            | jnp.where(aux_finite, 0, STEP_SKIP_NONFINITE_AUX)
            | jnp.where(grad_finite, 0, STEP_SKIP_NONFINITE_GRAD)
            | jnp.where(empty, STEP_SKIP_EMPTY_WEIGHTED_BATCH, 0)
        ).astype(jnp.int32)

        def apply_update(operand):
            p, state, g = operand
            updates, new_state = tx.update(g, state, p, loss=loss)
            # Keep parameter and optimizer-state dtypes fixed (fp32 updates must not
            # promote bf16 leaves, which would change the executable's signature).
            return restore_dtypes(optax.apply_updates(p, updates), p), restore_dtypes(new_state, state)

        def skip_update(operand):
            p, state, _ = operand
            return p, state

        # Rejecting only the gradient is not enough: AdamW would still move its
        # moments and apply decoupled weight decay. Keep both untouched.
        if update_guard == "select":
            # The same rule as an elementwise select of every updated leaf (fused into the update), so the program
            # has no conditional of its own for the host to wait on (optim.skip_spikes keeps its lax.cond;
            # benchmarks/multi_gpu.md).
            updated_params, updated_state = apply_update((params, opt_state, grads))
            new_params = jax.tree.map(lambda new, old: jnp.where(applied, new, old), updated_params, params)
            new_opt_state = jax.tree.map(lambda new, old: jnp.where(applied, new, old), updated_state, opt_state)
        else:
            new_params, new_opt_state = jax.lax.cond(applied, apply_update, skip_update, (params, opt_state, grads))
        new_params = apply_fp8_state(new_params, params, fp8_values, fp8_mask, applied)
        if param_shardings is not None:
            new_params = constrain(new_params, param_shardings)
        if opt_shardings is not None:
            new_opt_state = constrain(new_opt_state, opt_shardings)

        metrics: dict[str, jax.Array] = {
            "loss": loss,
            "grad_norm": grad_norm,
            "tokens": tokens,
            "step_applied": applied,
            "skip_reason": skip_reason,
            "grad_accum": jnp.asarray(grad_accum, jnp.int32),
        }
        for name, value in totals.items():
            if name in ("loss_sum", "tokens"):
                continue
            if name.endswith("_sum"):
                metrics[name[: -len("_sum")]] = value * inv_tokens
            else:
                metrics[name] = value / grad_accum
        if isinstance(new_opt_state, Mapping) and "skip_reason" in new_opt_state:
            metrics["spike_skip_reason"] = new_opt_state["skip_reason"]
        return eqx.combine(new_params, static), new_opt_state, metrics, key

    # The batch is the first argument so "all-except-first" donates exactly the state.
    return TrainStep(eqx.filter_jit(train_step, donate="all-except-first" if donate else "none"))


def build_eval_step(
    config: LossConfig,
    *,
    cross_entropy: Callable[..., Any] | None = None,
    batch_sharding: jax.sharding.Sharding | None = None,
    forward: Callable[..., Any] | None = None,
) -> Callable[[Any, dict[str, jax.Array]], dict[str, jax.Array]]:
    """Jitted evaluation of one ``[B, T]`` batch: summed CE (no smoothing) and token count."""
    ce_fn = _cross_entropy_fn(cross_entropy)
    compute_dtype = jnp.dtype(config.compute_dtype)
    run_forward = _forward if forward is None else forward

    def eval_step(model: Any, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        validate_batch(batch)
        if batch_sharding is not None:
            batch = {name: jax.lax.with_sharding_constraint(value, batch_sharding) for name, value in batch.items()}
        out = run_forward(model, batch, None, inference=True)
        targets = _masked_targets(batch, config.ignore_index)
        ce = ce_fn(
            out.hidden.reshape(-1, out.hidden.shape[-1]),
            model.lm_head_weight(),
            targets.reshape(-1),
            ignore_index=config.ignore_index,
            vocab_chunk=config.vocab_chunk,
            label_smoothing=0.0,
            compute_dtype=compute_dtype,
        )
        return {"loss_sum": jnp.asarray(ce.loss_sum, jnp.float32), "tokens": jnp.asarray(ce.num_valid, jnp.float32)}

    return eqx.filter_jit(eval_step)


def evaluate(
    eval_step: Callable[[Any, dict[str, jax.Array]], dict[str, jax.Array]],
    model: Any,
    batches: Iterable[dict[str, jax.Array]],
) -> dict[str, float]:
    """Token-weighted loss over ``batches`` with a single host transfer at the end."""
    loss_sum = jnp.zeros((), jnp.float32)
    tokens = jnp.zeros((), jnp.float32)
    count = 0
    for batch in batches:
        out = eval_step(model, batch)
        loss_sum = loss_sum + out["loss_sum"]
        tokens = tokens + out["tokens"]
        count += 1
    if count == 0:
        raise ValueError("evaluation received no batches")
    loss_host, tokens_host = (float(x) for x in jax.device_get((loss_sum, tokens)))
    mean = loss_host / max(tokens_host, 1.0)
    return {
        "eval_loss": mean,
        "eval_perplexity": math.exp(min(mean, 30.0)),
        "eval_tokens": tokens_host,
        "eval_batches": float(count),
    }


def _attention_batch_partition(batch_sharding: Any):
    """Tell attention layers traced in the loss which mesh axes shard the batch (``[grad_accum, B, T]``
    batches: the ``B`` entry of the spec), so cuDNN's size limit is checked per device, not on the global
    shapes GSPMD traces see (``jla.layers.attention.batch_partition``)."""
    import contextlib

    from jla.layers.attention import batch_partition

    spec = getattr(batch_sharding, "spec", None)
    mesh = getattr(batch_sharding, "mesh", None)
    if spec is None or mesh is None or len(spec) < 2 or spec[1] is None:
        return contextlib.nullcontext()
    return batch_partition(mesh, spec[1])


def _mtp_objective(config: LossConfig) -> Callable[..., Any]:
    """The forward-gradient cross entropy for the MTP heads (``LossConfig.grad_in_forward``), without z-loss.

    Called per depth as ``objective(hidden, weight, targets, vocab_chunk=..., compute_dtype=...)`` with the
    MTP head's compute dtype (the model's). The heads share the main head's rows, width and weight, so the
    kernels take them whenever they take the main head, except for a compute dtype they do not run (a
    float32 model): those calls keep the XLA op (``impl="auto"``) instead of raising.
    """
    from jla.ops.cross_entropy import linear_cross_entropy_objective

    impl = "auto" if config.impl == "mosaic_gpu" else config.impl

    def objective(hidden, weight, targets, *, vocab_chunk, compute_dtype):
        return linear_cross_entropy_objective(
            hidden, weight, targets, z_loss_weight=0.0, ignore_index=-100, vocab_chunk=vocab_chunk,
            row_chunk=config.row_chunk, compute_dtype=compute_dtype, impl=impl, fp8=config.fp8,
        )

    return objective


def _takes_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Whether ``fn`` accepts the keyword argument ``name`` (custom models' ``mtp_loss`` may not)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _on_one_device(shardings: Any) -> bool | None:
    """Whether every sharding in ``shardings`` (the train step's parameter shardings) names one device; None
    when none is given (:func:`jla.training.fp8.prepare_fp8_weights` then asks how many devices the process sees)."""
    leaves = [s for s in jax.tree.leaves(shardings) if isinstance(s, jax.sharding.Sharding)]
    return all(len(s.device_set) == 1 for s in leaves) if leaves else None
