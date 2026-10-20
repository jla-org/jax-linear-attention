# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validated learning-rate schedules with matching device and host variants.

``create_schedule`` returns an absolute-LR Optax schedule (traceable, used by
the optimizer); ``create_schedule_fn`` returns the same curve as a host-side
multiplier of the peak LR (used for logging without touching the device).
Both share defaults and phase arithmetic so they cannot drift apart.

Families: ``wsd`` (warmup-stable-decay), ``cosine``, ``inv_sqrt``,
``constant``, ``noam`` and ``trapezoidal``; every family supports linear
warmup over ``warmup_steps``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

__all__ = [
    "SUPPORTED_SCHEDULE_TYPES",
    "SchedulePlan",
    "cosine_schedule",
    "cosine_schedule_fn",
    "create_schedule",
    "create_schedule_fn",
    "create_schedule_plan",
    "inv_sqrt_schedule",
    "inv_sqrt_schedule_fn",
    "noam_schedule",
    "noam_schedule_fn",
    "schedule_boundary_steps",
    "schedule_phase_steps",
    "trapezoidal_schedule",
    "trapezoidal_schedule_fn",
    "wsd_schedule",
    "wsd_schedule_fn",
]

_VALID_DECAY_TYPES = ("linear", "cosine", "sqrt")
SUPPORTED_SCHEDULE_TYPES = ("cosine", "wsd", "noam", "constant", "trapezoidal", "inv_sqrt", "inverse_sqrt")


def _exact_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _nonnegative_int(name: str, value: int) -> int:
    result = _exact_int(name, value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _positive_int(name: str, value: int) -> int:
    result = _exact_int(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _finite_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not valid:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {comparator}, got {value!r}")
    return result


def _ratio(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return result


def _validate_phase_steps(**steps: int) -> None:
    for name, value in steps.items():
        _nonnegative_int(name, value)


def _validate_decay_type(decay_type: str) -> str:
    if not isinstance(decay_type, str):
        raise TypeError("decay_type must be a string")
    normalized = decay_type.strip().lower()
    if normalized not in _VALID_DECAY_TYPES:
        raise ValueError(f"decay_type must be one of {_VALID_DECAY_TYPES}, got {decay_type!r}")
    return normalized


def _schedule_key(schedule_type: str) -> str:
    if not isinstance(schedule_type, str) or not schedule_type.strip():
        raise TypeError("schedule_type must be a non-empty string")
    key = schedule_type.strip().lower()
    if key not in SUPPORTED_SCHEDULE_TYPES:
        raise ValueError(f"unknown schedule_type={schedule_type!r}; expected one of {SUPPORTED_SCHEDULE_TYPES}")
    return "inv_sqrt" if key == "inverse_sqrt" else key


def _host_step(step: int) -> int:
    return _nonnegative_int("step", step)


@dataclass(frozen=True)
class SchedulePlan:
    """Host metadata describing the phases of a schedule."""

    schedule_type: str
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    total_steps: int
    min_lr_ratio: float
    decay_type: str
    peak_lr: float

    def __post_init__(self) -> None:
        schedule_type = _schedule_key(self.schedule_type)
        warmup_steps = _nonnegative_int("warmup_steps", self.warmup_steps)
        stable_steps = _nonnegative_int("stable_steps", self.stable_steps)
        decay_steps = _nonnegative_int("decay_steps", self.decay_steps)
        total_steps = _positive_int("total_steps", self.total_steps)
        if warmup_steps + stable_steps + decay_steps != total_steps:
            raise ValueError("schedule phases must sum to total_steps")
        min_lr_ratio = _ratio("min_lr_ratio", self.min_lr_ratio)
        peak_lr = _finite_float("peak_lr", self.peak_lr, allow_zero=True)
        decay_type = _validate_decay_type(self.decay_type) if schedule_type == "wsd" else str(self.decay_type)
        # Canonicalize aliases and scalar types so equal plans compare equal.
        object.__setattr__(self, "schedule_type", schedule_type)
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "stable_steps", stable_steps)
        object.__setattr__(self, "decay_steps", decay_steps)
        object.__setattr__(self, "total_steps", total_steps)
        object.__setattr__(self, "min_lr_ratio", min_lr_ratio)
        object.__setattr__(self, "decay_type", decay_type)
        object.__setattr__(self, "peak_lr", peak_lr)

    @property
    def boundary_steps(self) -> tuple[int, ...]:
        steps = {0, self.warmup_steps, self.warmup_steps + self.stable_steps, self.total_steps}
        return tuple(sorted(step for step in steps if 0 <= step <= self.total_steps))

    def as_dict(self) -> dict[str, float | int | str | tuple[int, ...]]:
        return {
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "stable_steps": self.stable_steps,
            "decay_steps": self.decay_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "decay_type": self.decay_type,
            "peak_lr": self.peak_lr,
            "boundary_steps": self.boundary_steps,
        }


def _select(predicate: jax.Array, on_true: jax.Array, on_false: jax.Array) -> jax.Array:
    return jnp.where(predicate, on_true, on_false)


def _warmup_scale(step: jax.Array, warmup_steps: int) -> jax.Array:
    return jnp.minimum(1.0, (step + 1.0) / max(warmup_steps, 1))


def _decay_fraction(progress: jax.Array, decay_type: str) -> jax.Array:
    if decay_type == "cosine":
        return 0.5 * (1.0 - jnp.cos(jnp.pi * progress))
    if decay_type == "sqrt":
        return jnp.sqrt(progress)
    return progress


def _constant_schedule(learning_rate: float) -> optax.Schedule:
    value = jnp.asarray(learning_rate, jnp.float32)

    def schedule(step):
        del step
        return value

    return schedule


def wsd_schedule(
    peak_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
    decay_type: str = "cosine",
) -> optax.Schedule:
    """Warmup, constant peak, then decay to ``min_lr_ratio * peak_lr``."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    _validate_phase_steps(warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    decay_type = _validate_decay_type(decay_type)
    stable_end = warmup_steps + stable_steps
    peak = jnp.asarray(peak_lr, jnp.float32)
    floor = jnp.asarray(min_lr_ratio, jnp.float32)
    one = jnp.asarray(1.0, jnp.float32)

    def schedule(step):
        current = jnp.maximum(jnp.asarray(step, jnp.float32), 0.0)
        warmup = _warmup_scale(current, warmup_steps)
        if decay_steps == 0:
            post_warmup = _select(current < stable_end, one, floor)
        else:
            progress = jnp.clip((current - stable_end) / decay_steps, 0.0, 1.0)
            decay = one - _decay_fraction(progress, decay_type) * (one - floor)
            post_warmup = _select(current < stable_end, one, decay)
        return peak * _select(current < warmup_steps, warmup, post_warmup)

    return schedule


def cosine_schedule(peak_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> optax.Schedule:
    """Linear warmup then cosine decay from ``peak_lr`` to ``min_lr_ratio * peak_lr``."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
    max_steps = _positive_int("max_steps", max_steps)
    if warmup_steps > max_steps:
        raise ValueError("warmup_steps must be <= max_steps")
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    decay_steps = max(max_steps - warmup_steps, 1)
    peak = jnp.asarray(peak_lr, jnp.float32)
    floor = jnp.asarray(min_lr_ratio, jnp.float32)

    def schedule(step):
        current = jnp.maximum(jnp.asarray(step, jnp.float32), 0.0)
        warmup = _warmup_scale(current, warmup_steps)
        progress = jnp.clip((current - warmup_steps) / decay_steps, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay = floor + (1.0 - floor) * cosine
        return peak * _select(current < warmup_steps, warmup, decay)

    return schedule


def trapezoidal_schedule(
    peak_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> optax.Schedule:
    """Warmup, constant, then ``1 - sqrt(progress)`` cooldown."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    _validate_phase_steps(warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    stable_end = warmup_steps + stable_steps
    peak = jnp.asarray(peak_lr, jnp.float32)
    floor = jnp.asarray(min_lr_ratio, jnp.float32)
    one = jnp.asarray(1.0, jnp.float32)

    def schedule(step):
        current = jnp.maximum(jnp.asarray(step, jnp.float32), 0.0)
        warmup = _warmup_scale(current, warmup_steps)
        if decay_steps == 0:
            post_warmup = _select(current < stable_end, one, floor)
        else:
            progress = jnp.clip((current - stable_end) / decay_steps, 0.0, 1.0)
            decay = floor + (one - floor) * (one - jnp.sqrt(progress))
            post_warmup = _select(current < stable_end, one, decay)
        return peak * _select(current < warmup_steps, warmup, post_warmup)

    return schedule


def noam_schedule(peak_lr: float, d_model: int = 512, warmup_steps: int = 4000, factor: float = 1.0) -> optax.Schedule:
    """Transformer (Noam) schedule rescaled so its maximum equals ``peak_lr``."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    d_model = _positive_int("d_model", d_model)
    warmup_steps = _positive_int("warmup_steps", warmup_steps)
    factor = _finite_float("factor", factor)
    inv_sqrt_model = float(d_model) ** -0.5
    warmup_power = float(warmup_steps) ** -1.5
    raw_peak = factor * inv_sqrt_model * float(warmup_steps) ** -0.5
    multiplier = peak_lr / raw_peak

    def schedule(step):
        current = jnp.maximum(jnp.asarray(step, jnp.float32), 1.0)
        raw = factor * inv_sqrt_model * jnp.minimum(jax.lax.rsqrt(current), current * warmup_power)
        return jnp.asarray(multiplier, jnp.float32) * raw

    return schedule


def inv_sqrt_schedule(peak_lr: float, warmup_steps: int, min_lr_ratio: float = 0.1) -> optax.Schedule:
    """Linear warmup, then ``peak * sqrt(warmup / step)`` floored at ``min_lr_ratio``."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    warmup_steps = _positive_int("warmup_steps", warmup_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    peak = jnp.asarray(peak_lr, jnp.float32)
    floor = jnp.asarray(peak_lr * min_lr_ratio, jnp.float32)

    def schedule(step):
        current = jnp.maximum(jnp.asarray(step, jnp.float32), 0.0)
        warmup = peak * (current + 1.0) / warmup_steps
        decay = peak * jax.lax.rsqrt(jnp.maximum(current, 1.0) / warmup_steps)
        # The floor bounds the decay only; warmup always ramps from zero.
        return _select(current < warmup_steps, warmup, jnp.maximum(decay, floor))

    return schedule


def schedule_phase_steps(
    schedule_type: str = "wsd",
    *,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    stable_fraction: float = 0.8,
) -> tuple[int, int, int]:
    """``(warmup, stable, decay)`` step counts of a schedule family."""
    warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
    total_steps = _positive_int("total_steps", total_steps)
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps must be <= total_steps")
    stable_fraction = _ratio("stable_fraction", stable_fraction)
    key = _schedule_key(schedule_type)
    if key == "constant":
        return 0, total_steps, 0
    post_warmup = total_steps - warmup_steps
    if key in ("noam", "cosine", "inv_sqrt"):
        return warmup_steps, 0, post_warmup
    stable_steps = int(post_warmup * stable_fraction)
    return warmup_steps, stable_steps, post_warmup - stable_steps


def schedule_boundary_steps(
    schedule_type: str = "wsd",
    *,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    stable_fraction: float = 0.8,
) -> tuple[int, ...]:
    warmup, stable, _ = schedule_phase_steps(
        schedule_type, warmup_steps=warmup_steps, total_steps=total_steps, stable_fraction=stable_fraction
    )
    return tuple(sorted({0, warmup, warmup + stable, total_steps}))


def create_schedule(
    peak_lr: float,
    schedule_type: str = "wsd",
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    stable_fraction: float = 0.8,
    decay_type: str = "cosine",
    d_model: int = 512,
) -> optax.Schedule:
    """Absolute-LR schedule of any supported family."""
    peak_lr = _finite_float("peak_lr", peak_lr, allow_zero=True)
    warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
    total_steps = _positive_int("total_steps", total_steps)
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps must be <= total_steps")
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    stable_fraction = _ratio("stable_fraction", stable_fraction)
    key = _schedule_key(schedule_type)
    if key in ("wsd", "trapezoidal"):
        _, stable_steps, decay_steps = schedule_phase_steps(
            key, warmup_steps=warmup_steps, total_steps=total_steps, stable_fraction=stable_fraction
        )
        if key == "wsd":
            return wsd_schedule(peak_lr, warmup_steps, stable_steps, decay_steps, min_lr_ratio, decay_type)
        return trapezoidal_schedule(peak_lr, warmup_steps, stable_steps, decay_steps, min_lr_ratio)
    if key == "cosine":
        return cosine_schedule(peak_lr, warmup_steps, total_steps, min_lr_ratio)
    if key == "noam":
        return noam_schedule(peak_lr, d_model, warmup_steps)
    if key == "inv_sqrt":
        return inv_sqrt_schedule(peak_lr, warmup_steps, min_lr_ratio)
    return _constant_schedule(peak_lr)


def create_schedule_plan(
    peak_lr: float,
    schedule_type: str = "wsd",
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    stable_fraction: float = 0.8,
    decay_type: str = "cosine",
) -> SchedulePlan:
    key = _schedule_key(schedule_type)
    warmup, stable, decay = schedule_phase_steps(
        key, warmup_steps=warmup_steps, total_steps=total_steps, stable_fraction=stable_fraction
    )
    return SchedulePlan(
        schedule_type=key,
        warmup_steps=warmup,
        stable_steps=stable,
        decay_steps=decay,
        total_steps=total_steps,
        min_lr_ratio=_ratio("min_lr_ratio", min_lr_ratio),
        decay_type=_validate_decay_type(decay_type) if key == "wsd" else str(decay_type),
        peak_lr=_finite_float("peak_lr", peak_lr, allow_zero=True),
    )


def wsd_schedule_fn(
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
    decay_type: str = "cosine",
) -> Callable[[int], float]:
    _validate_phase_steps(warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    decay_type = _validate_decay_type(decay_type)

    def schedule(step: int) -> float:
        step = _host_step(step)
        if step < warmup_steps:
            return min(1.0, (step + 1) / max(warmup_steps, 1))
        stable_end = warmup_steps + stable_steps
        if step < stable_end:
            return 1.0
        if decay_steps == 0:
            return min_lr_ratio
        progress = min(max((step - stable_end) / decay_steps, 0.0), 1.0)
        if decay_type == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif decay_type == "sqrt":
            progress = math.sqrt(progress)
        return 1.0 - progress * (1.0 - min_lr_ratio)

    return schedule


def cosine_schedule_fn(warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1) -> Callable[[int], float]:
    warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
    max_steps = _positive_int("max_steps", max_steps)
    if warmup_steps > max_steps:
        raise ValueError("warmup_steps must be <= max_steps")
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)

    def schedule(step: int) -> float:
        step = _host_step(step)
        if step < warmup_steps:
            return min(1.0, (step + 1) / max(warmup_steps, 1))
        decay_steps = max(max_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule


def noam_schedule_fn(d_model: int = 512, warmup_steps: int = 4000, factor: float = 1.0) -> Callable[[int], float]:
    d_model = _positive_int("d_model", d_model)
    warmup_steps = _positive_int("warmup_steps", warmup_steps)
    factor = _finite_float("factor", factor)
    peak = factor * d_model**-0.5 * warmup_steps**-0.5

    def schedule(step: int) -> float:
        current = max(_host_step(step), 1)
        raw = factor * d_model**-0.5 * min(current**-0.5, current * warmup_steps**-1.5)
        return raw / peak

    return schedule


def trapezoidal_schedule_fn(
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    _validate_phase_steps(warmup_steps=warmup_steps, stable_steps=stable_steps, decay_steps=decay_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)

    def schedule(step: int) -> float:
        step = _host_step(step)
        if step < warmup_steps:
            return min(1.0, (step + 1) / max(warmup_steps, 1))
        stable_end = warmup_steps + stable_steps
        if step < stable_end:
            return 1.0
        if decay_steps == 0:
            return min_lr_ratio
        progress = min(max((step - stable_end) / decay_steps, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(progress))

    return schedule


def inv_sqrt_schedule_fn(warmup_steps: int, min_lr_ratio: float = 0.1) -> Callable[[int], float]:
    warmup_steps = _positive_int("warmup_steps", warmup_steps)
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)

    def schedule(step: int) -> float:
        step = _host_step(step)
        if step < warmup_steps:
            return min(1.0, (step + 1) / warmup_steps)
        return max(min_lr_ratio, math.sqrt(warmup_steps / max(step, 1)))

    return schedule


def create_schedule_fn(
    scheduler_type: str = "wsd",
    warmup_steps: int = 2000,
    max_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    stable_fraction: float = 0.8,
    decay_type: str = "cosine",
    d_model: int = 512,
) -> Callable[[int], float]:
    """Host multiplier of the peak LR matching :func:`create_schedule`."""
    warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
    max_steps = _positive_int("max_steps", max_steps)
    if warmup_steps > max_steps:
        raise ValueError("warmup_steps must be <= max_steps")
    min_lr_ratio = _ratio("min_lr_ratio", min_lr_ratio)
    stable_fraction = _ratio("stable_fraction", stable_fraction)
    key = _schedule_key(scheduler_type)
    if key in ("wsd", "trapezoidal"):
        _, stable_steps, decay_steps = schedule_phase_steps(
            key, warmup_steps=warmup_steps, total_steps=max_steps, stable_fraction=stable_fraction
        )
        if key == "wsd":
            return wsd_schedule_fn(warmup_steps, stable_steps, decay_steps, min_lr_ratio, decay_type)
        return trapezoidal_schedule_fn(warmup_steps, stable_steps, decay_steps, min_lr_ratio)
    if key == "cosine":
        return cosine_schedule_fn(warmup_steps, max_steps, min_lr_ratio)
    if key == "noam":
        return noam_schedule_fn(d_model, warmup_steps)
    if key == "inv_sqrt":
        return inv_sqrt_schedule_fn(warmup_steps, min_lr_ratio)

    def constant(step: int) -> float:
        _host_step(step)
        return 1.0

    return constant
