# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device and host learning-rate schedules agree at every phase boundary."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.training.schedulers import (
    SchedulePlan,
    create_schedule,
    create_schedule_fn,
    create_schedule_plan,
    schedule_boundary_steps,
    wsd_schedule,
)


@pytest.mark.parametrize("schedule_type", ["constant", "cosine", "wsd", "noam", "trapezoidal", "inv_sqrt", "inverse_sqrt"])
@pytest.mark.parametrize("warmup_steps", [4, 20])
def test_host_and_device_schedules_match(schedule_type: str, warmup_steps: int) -> None:
    peak, total, stable = 0.25, 20, 0.6
    host = create_schedule_fn(schedule_type, warmup_steps=warmup_steps, max_steps=total, min_lr_ratio=0.2, stable_fraction=stable, d_model=16)
    device = create_schedule(peak, schedule_type, warmup_steps=warmup_steps, total_steps=total, min_lr_ratio=0.2, stable_fraction=stable, d_model=16)
    probes = set(schedule_boundary_steps(schedule_type, warmup_steps=warmup_steps, total_steps=total, stable_fraction=stable))
    probes |= {1, warmup_steps - 1, total - 1, total + 5}
    for step in sorted(probes):
        assert float(jax.jit(device)(jnp.asarray(step, jnp.int32))) == pytest.approx(peak * host(step), rel=2e-6, abs=2e-7)


def test_wsd_phases() -> None:
    schedule = wsd_schedule(1.0, warmup_steps=2, stable_steps=3, decay_steps=5, min_lr_ratio=0.1, decay_type="linear")
    values = np.asarray([float(schedule(step)) for step in range(12)])
    np.testing.assert_allclose(values[:2], [0.5, 1.0], rtol=1e-6)
    np.testing.assert_allclose(values[2:5], 1.0, rtol=1e-6)
    np.testing.assert_allclose(values[5], 1.0, rtol=1e-6)
    np.testing.assert_allclose(values[10:], 0.1, rtol=1e-6)
    assert np.all(np.diff(values[5:11]) < 0)


def test_schedule_plan_canonicalizes_scalars() -> None:
    plan = SchedulePlan(
        schedule_type="WSD",
        warmup_steps=np.int64(2),
        stable_steps=np.int64(6),
        decay_steps=np.int64(2),
        total_steps=np.int64(10),
        min_lr_ratio=np.float64(0.1),
        decay_type="COSINE",
        peak_lr=np.float64(1e-3),
    )
    assert plan.schedule_type == "wsd" and plan.decay_type == "cosine"
    assert type(plan.total_steps) is int and type(plan.peak_lr) is float
    assert plan.boundary_steps == (0, 2, 8, 10)
    assert create_schedule_plan(1e-3, "inverse_sqrt", warmup_steps=10, total_steps=100).schedule_type == "inv_sqrt"
    with pytest.raises(ValueError, match="sum to total_steps"):
        SchedulePlan("wsd", 1, 1, 1, 10, 0.1, "cosine", 1e-3)


@pytest.mark.parametrize("name", ["min_lr_ratio", "stable_fraction"])
def test_ratios_reject_booleans(name: str) -> None:
    kwargs: dict[str, Any] = {name: True}
    with pytest.raises(ValueError, match=name):
        create_schedule_plan(1e-3, total_steps=10, warmup_steps=1, **kwargs)


def test_invalid_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown schedule_type"):
        create_schedule(1e-3, "p6")
    with pytest.raises(ValueError, match="warmup_steps must be <="):
        create_schedule(1e-3, "cosine", warmup_steps=11, total_steps=10)
    with pytest.raises(ValueError, match="warmup_steps must be > 0"):
        create_schedule(1e-3, "inv_sqrt", warmup_steps=0, total_steps=10)
