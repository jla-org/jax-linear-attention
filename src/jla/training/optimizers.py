# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validated Optax optimizer construction.

* AdamW/NAdamW run their moment math in float32 while storing ``mu``/``nu`` in
  independently chosen dtypes;
* Muon (``optax.contrib.muon``) orthogonalizes hidden matrices (the last two
  axes of every eligible leaf; leading axes such as scanned layers or experts
  are batch axes) and uses its Adam fallback elsewhere;
* Adafactor, AdEMAMix, Lion and SGD are available with the same decay masks;
* weight decay skips vectors/scalars and norm, bias, ``dt_bias`` and ``A_log``
  parameters;
* ``skip_step_on_spikes`` rejects non-finite or statistically anomalous steps
  without poisoning its running statistics;
* regex freezing, ``optax.MultiSteps`` and an embedding LR multiplier.

Gradients are promoted to float32 before any optimizer math.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

__all__ = [
    "SPIKE_SKIP_GRAD_NORM_SPIKE",
    "SPIKE_SKIP_LOSS_SPIKE",
    "SPIKE_SKIP_NONE",
    "SPIKE_SKIP_NONFINITE_GRAD_NORM",
    "SPIKE_SKIP_NONFINITE_LOSS",
    "clip_by_global_norm",
    "create_optimizer",
    "decay_mask",
    "muon_eligible",
    "optimizer_from_config",
    "skip_step_on_spikes",
]

_NO_DECAY_TOKENS: tuple[str, ...] = ("norm", "bias", "dt_bias", "a_log")
_MUON_INELIGIBLE_TOKENS: tuple[str, ...] = (
    "embed",
    "lm_head",
    "unembed",
    "norm",
    "bias",
    "router",
    "conv",
    "dt_bias",
    "a_log",
)
_EMBEDDING_TOKENS: tuple[str, ...] = ("embed", "lm_head")
_MOMENT_DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


def _none_leaf(value: Any) -> bool:
    return value is None


# Bitwise reasons recorded by ``skip_step_on_spikes`` for the latest update.
SPIKE_SKIP_NONE: int = 0
SPIKE_SKIP_NONFINITE_LOSS: int = 1 << 0
SPIKE_SKIP_LOSS_SPIKE: int = 1 << 1
SPIKE_SKIP_NONFINITE_GRAD_NORM: int = 1 << 2
SPIKE_SKIP_GRAD_NORM_SPIKE: int = 1 << 3


def _exact_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return int(parsed)


def _finite_float(name: str, value: Any, *, minimum: float | None = None, strict_minimum: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        invalid = parsed <= minimum if strict_minimum else parsed < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}, got {parsed}")
    return parsed


def _validate_betas(betas: Sequence[float]) -> tuple[float, float]:
    if isinstance(betas, (str, bytes)) or len(betas) != 2:
        raise ValueError("betas must contain exactly two coefficients")
    b1 = _finite_float("betas[0]", betas[0], minimum=0.0)
    b2 = _finite_float("betas[1]", betas[1], minimum=0.0)
    if b1 >= 1.0 or b2 >= 1.0:
        raise ValueError(f"betas must be in [0, 1), got {(b1, b2)!r}")
    return b1, b2


def _path_string(path: tuple[Any, ...]) -> str:
    return jax.tree_util.keystr(path, simple=True, separator="/").lower()


def _is_array(leaf: Any) -> bool:
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _path_has(path: tuple[Any, ...], tokens: tuple[str, ...]) -> bool:
    text = _path_string(path)
    return any(token in text for token in tokens)


def _stacked_layer_axes(params: Any, path: tuple[Any, ...]) -> int:
    """1 for leaves of a scanned model's ``blocks`` (stacked over layers, ``config.scan_layers``), else 0."""
    scanned = bool(getattr(getattr(params, "config", None), "scan_layers", False))
    head = path[0] if path else None
    return int(scanned and isinstance(head, jax.tree_util.GetAttrKey) and head.name == "blocks")


# Leaves that stack per-channel vectors into one array rather than holding a matrix: RWKV-7's six
# token-shift mixes ``mu [6, D]`` and its per-head bonus ``r_k [H, K]``. Like vectors, they get no
# weight decay and no Muon (BlinkDL's RWKV-7 recipe decays only the ``.weight`` matrices).
_VECTOR_LEAF_NAMES: frozenset[str] = frozenset({"mu", "r_k"})


def _is_vector_leaf(path: tuple[Any, ...]) -> bool:
    last = path[-1] if path else None
    return isinstance(last, jax.tree_util.GetAttrKey) and last.name in _VECTOR_LEAF_NAMES


def decay_mask(params: Any) -> Any:
    """True for leaves that receive weight decay: matrices that are not norm/bias/dt_bias/A_log.

    The rank of a scanned model's stacked block leaves is judged per layer, so a stacked
    vector is not a matrix and both layer layouts train the same way.
    """

    def should_decay(path: tuple[Any, ...], leaf: Any) -> bool:
        return (_is_array(leaf) and len(leaf.shape) - _stacked_layer_axes(params, path) >= 2
                and not _path_has(path, _NO_DECAY_TOKENS) and not _is_vector_leaf(path))

    return jax.tree_util.tree_map_with_path(should_decay, params, is_leaf=_none_leaf)


def muon_eligible(path: tuple[Any, ...], leaf: Any, layer_axes: int = 0) -> bool:
    """Hidden matrices: ndim >= 2, both matrix axes >= 2, not embeddings/heads/norms/routers/convs.

    ``layer_axes`` leading axes stack layers (scanned blocks) and do not count towards the rank.
    """
    if not _is_array(leaf) or len(leaf.shape) - layer_axes < 2 or min(leaf.shape[-2:]) < 2:
        return False
    return not _path_has(path, _MUON_INELIGIBLE_TOKENS) and not _is_vector_leaf(path)


def _muon_dimension_numbers(params: Any) -> Any:
    def numbers(path: tuple[Any, ...], leaf: Any) -> Any:
        if not muon_eligible(path, leaf, _stacked_layer_axes(params, path)):
            return None
        ndim = len(leaf.shape)
        return optax.contrib.MuonDimensionNumbers(reduction_axis=ndim - 2, output_axis=ndim - 1)

    return jax.tree_util.tree_map_with_path(numbers, params, is_leaf=_none_leaf)


def _embedding_mask(params: Any) -> Any:
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: _is_array(leaf) and _path_has(path, _EMBEDDING_TOKENS),
        params,
        is_leaf=_none_leaf,
    )


def _trainable_label_fn(patterns: Sequence[str]) -> Callable[[Any], Any]:
    compiled = tuple(re.compile(pattern) for pattern in patterns)

    def labels(params: Any) -> Any:
        return jax.tree_util.tree_map_with_path(
            lambda path, _leaf: (
                "trainable" if any(p.search(_path_string(path)) for p in compiled) else "frozen"
            ),
            params,
            is_leaf=_none_leaf,
        )

    return labels


def _promote_updates_to_fp32() -> optax.GradientTransformation:
    """Promote sub-fp32 floating gradients before accumulation and optimizer math."""

    def init_fn(_params: Any) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(updates: Any, state: optax.EmptyState, params: Any = None):
        del params

        def promote(leaf: Any) -> Any:
            if not _is_array(leaf):
                return leaf
            dtype = jnp.dtype(leaf.dtype)
            if jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < 4:
                return leaf.astype(jnp.float32)
            return leaf

        return jax.tree.map(promote, updates, is_leaf=_none_leaf), state

    return optax.GradientTransformation(init_fn, update_fn)


def clip_by_global_norm(max_norm: float, *, eps: float = 1e-6) -> optax.GradientTransformation:
    """Global-norm clipping that leaves the tree untouched when the norm is non-finite.

    Non-finite steps are rejected by the train step itself; scaling every leaf
    by a NaN coefficient here would only spread the corruption.
    """
    max_norm = _finite_float("max_norm", max_norm, minimum=0.0)
    eps = _finite_float("eps", eps, minimum=0.0, strict_minimum=True)

    def init_fn(_params: Any) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(updates: Any, state: optax.EmptyState, params: Any = None):
        del params
        norm = optax.tree.norm(updates)
        clip = jnp.isfinite(norm) & (norm > max_norm)
        coefficient = jnp.where(clip, max_norm / jnp.maximum(norm, eps), 1.0)

        def scale(leaf: Any) -> Any:
            if not _is_array(leaf):
                return leaf
            return jnp.where(clip, leaf * coefficient.astype(leaf.dtype), leaf)

        return jax.tree.map(scale, updates, is_leaf=_none_leaf), state

    return optax.GradientTransformation(init_fn, update_fn)


def _rolling_stats(values: jax.Array, count: jax.Array, interval: int) -> tuple[jax.Array, jax.Array]:
    valid = jnp.minimum(count, interval)
    mask = jnp.arange(interval, dtype=count.dtype) < valid
    mean = jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(valid, 1)
    variance = jnp.sum(jnp.where(mask, jnp.square(values - mean), 0.0)) / jnp.maximum(valid - 1, 1)
    return mean, jnp.sqrt(variance)


def skip_step_on_spikes(
    inner: optax.GradientTransformation,
    *,
    interval: int = 128,
    scaling_factor: float = 6.0,
) -> optax.GradientTransformationExtraArgs:
    """Skip non-finite or anomalous (> mean + k * std) loss / gradient-norm steps.

    Rejected observations never enter the rolling history. ``loss`` is an
    optional extra argument of ``update``; the gradient norm is computed from
    the incoming updates unless ``grad_norm`` is passed. The state's
    ``skip_reason`` is a bitwise combination of the ``SPIKE_SKIP_*`` constants.
    """
    interval = _exact_int("interval", interval, minimum=2)
    scaling_factor = _finite_float("scaling_factor", scaling_factor, minimum=0.0, strict_minimum=True)
    guarded = optax.with_extra_args_support(inner)
    min_history = max(2, interval // 2)

    def init_fn(params: Any) -> dict[str, Any]:
        return {
            "inner_state": guarded.init(params),
            "losses": jnp.zeros((interval,), dtype=jnp.float32),
            "grad_norms": jnp.zeros((interval,), dtype=jnp.float32),
            "loss_count": jnp.zeros((), dtype=jnp.int32),
            "grad_count": jnp.zeros((), dtype=jnp.int32),
            "is_skipped": jnp.asarray(False),
            "skip_reason": jnp.asarray(SPIKE_SKIP_NONE, dtype=jnp.int32),
        }

    def update_fn(updates: Any, state: dict[str, Any], params: Any = None, **extra_args: Any):
        loss_value = extra_args.pop("loss", None)
        grad_value = extra_args.pop("grad_norm", None)
        reason = jnp.asarray(SPIKE_SKIP_NONE, dtype=jnp.int32)

        loss = None
        if loss_value is not None:
            loss = jnp.asarray(loss_value, dtype=jnp.float32)
            if loss.ndim != 0:
                raise ValueError(f"loss must be scalar, got shape {loss.shape}")
            mean, std = _rolling_stats(state["losses"], state["loss_count"], interval)
            finite = jnp.isfinite(loss)
            ok = (state["loss_count"] < min_history) | (loss <= mean + scaling_factor * std)
            reason = reason | jnp.where(finite, SPIKE_SKIP_NONE, SPIKE_SKIP_NONFINITE_LOSS)
            reason = reason | jnp.where(finite & ~ok, SPIKE_SKIP_LOSS_SPIKE, SPIKE_SKIP_NONE)

        grad = jnp.asarray(optax.tree.norm(updates) if grad_value is None else grad_value, dtype=jnp.float32)
        if grad.ndim != 0:
            raise ValueError(f"grad_norm must be scalar, got shape {grad.shape}")
        mean, std = _rolling_stats(state["grad_norms"], state["grad_count"], interval)
        finite = jnp.isfinite(grad)
        ok = (state["grad_count"] < min_history) | (grad <= mean + scaling_factor * std)
        reason = reason | jnp.where(finite, SPIKE_SKIP_NONE, SPIKE_SKIP_NONFINITE_GRAD_NORM)
        reason = reason | jnp.where(finite & ~ok, SPIKE_SKIP_GRAD_NORM_SPIKE, SPIKE_SKIP_NONE)
        reason = reason.astype(jnp.int32)
        accepted = reason == SPIKE_SKIP_NONE

        def apply(_: None):
            return guarded.update(updates, state["inner_state"], params, **extra_args)

        def skip(_: None):
            zeros = jax.tree.map(lambda leaf: None if leaf is None else jnp.zeros_like(leaf), updates, is_leaf=_none_leaf)
            return zeros, state["inner_state"]

        new_updates, inner_state = jax.lax.cond(accepted, apply, skip, operand=None)

        losses, loss_count = state["losses"], state["loss_count"]
        if loss is not None:
            losses = jnp.where(accepted, losses.at[loss_count % interval].set(loss), losses)
            loss_count = loss_count + accepted.astype(jnp.int32)
        grad_norms = jnp.where(accepted, state["grad_norms"].at[state["grad_count"] % interval].set(grad), state["grad_norms"])
        grad_count = state["grad_count"] + accepted.astype(jnp.int32)
        return new_updates, {
            "inner_state": inner_state,
            "losses": losses,
            "grad_norms": grad_norms,
            "loss_count": loss_count,
            "grad_count": grad_count,
            "is_skipped": ~accepted,
            "skip_reason": reason,
        }

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _moment_dtype(mode: str, param: Any) -> Any:
    if mode == "param":
        return jnp.dtype(param.dtype)
    if mode not in _MOMENT_DTYPES:
        raise ValueError(f"moment dtype must be one of {sorted(_MOMENT_DTYPES) + ['param']}, got {mode!r}")
    return jnp.dtype(_MOMENT_DTYPES[mode])


def _scale_by_adam_fp32(
    *,
    b1: float,
    b2: float,
    eps: float,
    mu_dtype: str,
    nu_dtype: str,
    nesterov: bool,
) -> optax.GradientTransformation:
    """Adam normalization with float32 math and separately typed stored moments."""
    _moment_dtype(mu_dtype, jnp.zeros((), jnp.float32))
    _moment_dtype(nu_dtype, jnp.zeros((), jnp.float32))

    def zeros(param: Any, mode: str) -> Any:
        if param is None or not _is_array(param):
            return param
        return jnp.zeros(param.shape, dtype=_moment_dtype(mode, param))

    def init_fn(params: Any) -> optax.ScaleByAdamState:
        return optax.ScaleByAdamState(
            count=jnp.zeros((), dtype=jnp.int32),
            mu=jax.tree.map(lambda p: zeros(p, mu_dtype), params, is_leaf=_none_leaf),
            nu=jax.tree.map(lambda p: zeros(p, nu_dtype), params, is_leaf=_none_leaf),
        )

    def update_fn(updates: Any, state: optax.ScaleByAdamState, params: Any = None):
        del params
        beta1 = jnp.float32(b1)
        beta2 = jnp.float32(b2)
        mu = jax.tree.map(
            lambda g, m: None if g is None else beta1 * m.astype(jnp.float32) + (1 - beta1) * g.astype(jnp.float32),
            updates,
            state.mu,
            is_leaf=_none_leaf,
        )
        nu = jax.tree.map(
            lambda g, v: None
            if g is None
            else beta2 * v.astype(jnp.float32) + (1 - beta2) * jnp.square(g.astype(jnp.float32)),
            updates,
            state.nu,
            is_leaf=_none_leaf,
        )
        count = optax.safe_increment(state.count)
        count_f = count.astype(jnp.float32)
        correction1 = -jnp.expm1(count_f * jnp.log(beta1))
        correction2 = -jnp.expm1(count_f * jnp.log(beta2))
        if nesterov:
            next_correction1 = -jnp.expm1((count_f + 1.0) * jnp.log(beta1))
            mu_hat = jax.tree.map(
                lambda m, g: None
                if m is None
                else beta1 * m / next_correction1 + (1 - beta1) * g.astype(jnp.float32) / correction1,
                mu,
                updates,
                is_leaf=_none_leaf,
            )
        else:
            mu_hat = jax.tree.map(lambda m: None if m is None else m / correction1, mu, is_leaf=_none_leaf)
        normalized = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v / correction2) + jnp.float32(eps)),
            mu_hat,
            nu,
            is_leaf=_none_leaf,
        )
        new_state = optax.ScaleByAdamState(
            count=count,
            mu=jax.tree.map(lambda m, s: None if m is None else m.astype(s.dtype), mu, state.mu, is_leaf=_none_leaf),
            nu=jax.tree.map(lambda v, s: None if v is None else v.astype(s.dtype), nu, state.nu, is_leaf=_none_leaf),
        )
        return normalized, new_state

    return optax.GradientTransformation(init_fn, update_fn)


class _DecayState(NamedTuple):
    count: jax.Array


def _decay_muon_adam_leaves(learning_rate: optax.ScalarOrSchedule, weight_decay: float) -> optax.GradientTransformation:
    """Decoupled decay for the leaves Muon routes to its internal Adam."""

    def init_fn(_params: Any) -> _DecayState:
        return _DecayState(jnp.zeros((), dtype=jnp.int32))

    def update_fn(updates: Any, state: _DecayState, params: Any = None):
        if params is None:
            raise ValueError("Muon weight decay requires params in optimizer.update()")
        lr = learning_rate(state.count) if callable(learning_rate) else learning_rate
        mask = decay_mask(params)
        adam_leaf = jax.tree_util.tree_map_with_path(
            lambda path, leaf: not muon_eligible(path, leaf, _stacked_layer_axes(params, path)), params,
            is_leaf=_none_leaf,
        )

        def decay(update: Any, param: Any, should_decay: bool, is_adam: bool) -> Any:
            if update is None or not should_decay or not is_adam:
                return update
            return update - jnp.asarray(lr, jnp.float32) * weight_decay * param.astype(update.dtype)

        decayed = jax.tree.map(decay, updates, params, mask, adam_leaf, is_leaf=_none_leaf)
        return decayed, _DecayState(optax.safe_increment(state.count))

    return optax.GradientTransformation(init_fn, update_fn)


def _as_gradient_transformation(tx: Any) -> optax.GradientTransformation:
    converter = getattr(tx, "gradient_transformation", None)
    return converter() if callable(converter) else tx


def create_optimizer(
    *,
    lr: float = 1e-4,
    schedule: optax.Schedule | None = None,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    max_grad_norm: float = 1.0,
    mu_dtype: str = "float32",
    nu_dtype: str = "float32",
    gradient_accumulation_steps: int = 1,
    embedding_lr_scale: float = 1.0,
    muon_ns_steps: int = 5,
    muon_momentum: float = 0.95,
    muon_adam_lr: float | None = None,
    muon_consistent_rms: float | None = 0.2,
    trainable_parameter_patterns: Sequence[str] = (),
    skip_spike_steps: bool = False,
    spike_interval: int = 128,
    spike_scaling_factor: float = 6.0,
) -> optax.GradientTransformation:
    """Build an Optax optimizer.

    ``schedule`` maps the optimizer step to an absolute learning rate; ``lr``
    is used when it is ``None`` and also sets the peak of Muon's Adam fallback
    (scaled by ``muon_adam_lr / lr`` when given). ``muon_consistent_rms=None``
    keeps optax's shape-dependent Muon scaling.
    """
    lr = _finite_float("lr", lr, minimum=0.0, strict_minimum=True)
    weight_decay = _finite_float("weight_decay", weight_decay, minimum=0.0)
    eps = _finite_float("eps", eps, minimum=0.0, strict_minimum=True)
    max_grad_norm = _finite_float("max_grad_norm", max_grad_norm, minimum=0.0)
    embedding_lr_scale = _finite_float("embedding_lr_scale", embedding_lr_scale, minimum=0.0)
    gradient_accumulation_steps = _exact_int("gradient_accumulation_steps", gradient_accumulation_steps, minimum=1)
    muon_ns_steps = _exact_int("muon_ns_steps", muon_ns_steps, minimum=1)
    muon_momentum = _finite_float("muon_momentum", muon_momentum, minimum=0.0)
    if muon_momentum >= 1.0:
        raise ValueError("muon_momentum must be in [0, 1)")
    if muon_consistent_rms is not None:
        muon_consistent_rms = _finite_float(
            "muon_consistent_rms", muon_consistent_rms, minimum=0.0, strict_minimum=True
        )
    b1, b2 = _validate_betas(betas)
    for pattern in trainable_parameter_patterns:
        re.compile(pattern)
    if not isinstance(optimizer_type, str) or not optimizer_type.strip():
        raise TypeError("optimizer_type must be a non-empty string")
    kind = optimizer_type.strip().lower()
    if kind not in ("adamw", "nadamw") and (mu_dtype, nu_dtype) != ("float32", "float32"):
        raise ValueError(f"mu_dtype/nu_dtype apply to adamw and nadamw only, not {kind!r}")
    learning_rate: optax.ScalarOrSchedule = lr if schedule is None else schedule
    mask = decay_mask if weight_decay else None

    if kind in ("adamw", "nadamw"):
        base = optax.chain(
            _scale_by_adam_fp32(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype, nu_dtype=nu_dtype, nesterov=kind == "nadamw"),
            optax.add_decayed_weights(weight_decay, mask=mask),
            optax.scale_by_learning_rate(learning_rate),
        )
    elif kind == "adafactor":
        base = optax.adafactor(
            learning_rate=learning_rate,
            min_dim_size_to_factor=128,
            decay_rate=0.8,
            multiply_by_parameter_scale=False,
            clipping_threshold=None,
            momentum=None,
            dtype_momentum=jnp.float32,
            weight_decay_rate=weight_decay or None,
            weight_decay_mask=mask,
        )
    elif kind == "ademamix":
        base = optax.contrib.ademamix(
            learning_rate=learning_rate, b1=b1, b2=b2, mu_dtype=jnp.float32, weight_decay=weight_decay, mask=mask
        )
    elif kind == "lion":
        base = optax.lion(
            learning_rate=learning_rate, b1=b1, b2=b2, mu_dtype=jnp.float32, weight_decay=weight_decay, mask=mask
        )
    elif kind == "sgd":
        # Decay after momentum and before LR scaling (sign and scale stay correct).
        transforms = [optax.trace(decay=b1, nesterov=b1 > 0.0, accumulator_dtype=jnp.float32)]
        if weight_decay:
            transforms.append(optax.add_decayed_weights(weight_decay, mask=decay_mask))
        transforms.append(optax.scale_by_learning_rate(learning_rate))
        base = optax.chain(*transforms)
    elif kind == "muon":
        if muon_adam_lr is None:
            adam_learning_rate: optax.ScalarOrSchedule = learning_rate
        else:
            ratio = _finite_float("muon_adam_lr", muon_adam_lr, minimum=0.0, strict_minimum=True) / lr
            adam_learning_rate = (lambda count: schedule(count) * ratio) if schedule is not None else lr * ratio
        base = optax.contrib.muon(
            learning_rate=learning_rate,
            ns_steps=muon_ns_steps,
            beta=muon_momentum,
            weight_decay=weight_decay,
            weight_decay_mask=mask,
            mu_dtype=jnp.float32,
            nesterov=True,
            consistent_rms=muon_consistent_rms,
            adam_b1=b1,
            adam_b2=b2,
            adam_weight_decay=0.0,
            adam_learning_rate=adam_learning_rate,
            muon_weight_dimension_numbers=_muon_dimension_numbers,
        )
        if weight_decay:
            base = optax.chain(base, _decay_muon_adam_leaves(adam_learning_rate, weight_decay))
    else:
        raise ValueError(
            f"unknown optimizer {optimizer_type!r}; expected adamw, nadamw, adafactor, ademamix, lion, sgd or muon"
        )

    if embedding_lr_scale != 1.0:
        base = optax.chain(base, optax.masked(optax.scale(embedding_lr_scale), _embedding_mask))
    if max_grad_norm > 0.0:
        base = optax.chain(clip_by_global_norm(max_grad_norm), base)
    if trainable_parameter_patterns:
        base = optax.multi_transform(
            {"trainable": base, "frozen": optax.set_to_zero()},
            _trainable_label_fn(trainable_parameter_patterns),
        )
    if gradient_accumulation_steps > 1:
        base = _as_gradient_transformation(
            optax.MultiSteps(
                base, every_k_schedule=gradient_accumulation_steps, use_grad_mean=True, accumulator_dtype=jnp.float32
            )
        )
    base = optax.chain(_promote_updates_to_fp32(), base)
    if skip_spike_steps:
        base = skip_step_on_spikes(base, interval=spike_interval, scaling_factor=spike_scaling_factor)
    return base


def optimizer_from_config(config: Any, schedule: optax.Schedule) -> optax.GradientTransformation:
    """Optimizer for a :class:`jla.config.OptimConfig` and an absolute-LR schedule."""
    return create_optimizer(
        lr=config.lr,
        schedule=schedule,
        optimizer_type=config.name,
        weight_decay=config.weight_decay,
        betas=(config.b1, config.b2),
        eps=config.eps,
        max_grad_norm=config.max_grad_norm,
        mu_dtype=config.mu_dtype,
        nu_dtype=config.nu_dtype,
        gradient_accumulation_steps=config.multi_steps,
        embedding_lr_scale=config.embedding_lr_scale,
        muon_ns_steps=config.muon_ns_steps,
        muon_momentum=config.muon_momentum,
        muon_adam_lr=config.muon_adam_lr,
        muon_consistent_rms=config.muon_consistent_rms or None,
        trainable_parameter_patterns=config.trainable_patterns,
        skip_spike_steps=config.skip_spikes,
        spike_interval=config.spike_interval,
        spike_scaling_factor=config.spike_scaling_factor,
    )
