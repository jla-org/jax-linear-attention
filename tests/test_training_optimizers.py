# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer construction, decay masks, spike guard, Muon scaling, freezing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.training.optimizers import (
    SPIKE_SKIP_GRAD_NORM_SPIKE,
    SPIKE_SKIP_LOSS_SPIKE,
    SPIKE_SKIP_NONE,
    SPIKE_SKIP_NONFINITE_GRAD_NORM,
    SPIKE_SKIP_NONFINITE_LOSS,
    clip_by_global_norm,
    create_optimizer,
    decay_mask,
    muon_eligible,
    skip_step_on_spikes,
)


def _array_leaves(tree: Any) -> Iterable[jax.Array]:
    return (leaf for leaf in jax.tree.leaves(tree) if hasattr(leaf, "dtype"))


def _assert_finite(tree: Any) -> None:
    for leaf in _array_leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _assert_equal(left: Any, right: Any) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(("name", "nesterov"), [("adamw", False), ("nadamw", True)])
def test_adam_variants_match_optax_one_step(name: str, nesterov: bool) -> None:
    params = jnp.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], jnp.float32)
    grads = jnp.array([[0.2, -0.4, 0.1], [-0.3, 0.6, -0.2]], jnp.float32)
    reference = optax.adamw(3e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, nesterov=nesterov)
    actual = create_optimizer(lr=3e-4, weight_decay=0.1, optimizer_type=name, betas=(0.9, 0.95), max_grad_norm=0.0)
    expected, _ = reference.update(grads, reference.init(params), params)
    got, _ = actual.update(grads, actual.init(params), params)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("name", ["adamw", "nadamw", "adafactor", "ademamix", "lion", "sgd", "muon"])
def test_every_optimizer_runs_eagerly_and_under_jit(name: str) -> None:
    params = {
        "dense": {"kernel": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16.0, "bias": jnp.zeros((4,))},
        "layers": {"stacked": jnp.ones((2, 4, 4), jnp.float32)},
        "optional": None,
    }
    grads = jax.tree.map(lambda p: None if p is None else jnp.full_like(p, 0.1), params, is_leaf=lambda x: x is None)
    optimizer = create_optimizer(lr=1e-3, weight_decay=0.01, optimizer_type=name, max_grad_norm=1.0)
    state = optimizer.init(params)
    eager, eager_state = optimizer.update(grads, state, params)
    jitted, jit_state = jax.jit(optimizer.update)(grads, state, params)
    assert eager["optional"] is None and jitted["optional"] is None
    for tree in (eager, eager_state, jitted, jit_state):
        _assert_finite(tree)
    for a, b in zip(_array_leaves(eager), _array_leaves(jitted), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_decay_mask_skips_vectors_norms_biases_and_ssm_parameters() -> None:
    params = {
        "attn": {"wq": jnp.ones((4, 4)), "q_norm": {"weight": jnp.ones((4, 4))}},
        "mixer": {"A_log": jnp.ones((2, 4)), "dt_bias": jnp.ones((2, 4)), "out_bias": jnp.ones((4, 4))},
        "embed": jnp.ones((8, 4)),
        "scale": jnp.ones((4,)),
    }
    mask = decay_mask(params)
    assert mask["attn"]["wq"] and mask["embed"]
    assert not mask["attn"]["q_norm"]["weight"] and not mask["scale"]
    assert not mask["mixer"]["A_log"] and not mask["mixer"]["dt_bias"] and not mask["mixer"]["out_bias"]
    reference = optax.add_decayed_weights(0.2, mask=decay_mask)
    zeros = jax.tree.map(jnp.zeros_like, params)
    updates, _ = reference.update(zeros, reference.init(params), params)
    np.testing.assert_allclose(updates["attn"]["wq"], 0.2)
    np.testing.assert_allclose(updates["scale"], 0.0)


def test_muon_eligibility_covers_hidden_and_stacked_matrices() -> None:
    path = lambda *names: tuple(jax.tree_util.DictKey(n) for n in names)  # noqa: E731
    assert muon_eligible(path("mlp", "gate_proj"), jnp.ones((4, 8)))
    assert muon_eligible(path("layers", "wq"), jnp.ones((3, 4, 8)))
    for name in ("embed", "lm_head", "router", "conv", "q_norm"):
        assert not muon_eligible(path(name), jnp.ones((4, 8)))
    assert not muon_eligible(path("w"), jnp.ones((8,)))


def test_multisteps_applies_the_mean_on_schedule() -> None:
    params = {"weight": jnp.zeros((2,))}
    optimizer = create_optimizer(
        lr=0.3, weight_decay=0.0, optimizer_type="sgd", betas=(0.0, 0.95), max_grad_norm=0.0, gradient_accumulation_steps=3
    )
    state = optimizer.init(params)
    emitted = []
    for value in (1.0, 2.0, 3.0):
        updates, state = optimizer.update({"weight": jnp.full((2,), value)}, state, params)
        emitted.append(updates["weight"])
    np.testing.assert_array_equal(emitted[0], 0.0)
    np.testing.assert_array_equal(emitted[1], 0.0)
    np.testing.assert_allclose(emitted[2], -0.6, atol=1e-7)


def test_regex_freezing_and_none_leaves() -> None:
    params = {"trainable": {"w": jnp.ones((2, 2))}, "frozen": {"w": jnp.ones((2, 2))}, "optional": None}
    optimizer = create_optimizer(
        lr=0.1, weight_decay=0.0, optimizer_type="sgd", betas=(0.0, 0.95), max_grad_norm=0.0, trainable_parameter_patterns=(r"^trainable",)
    )
    updates, _ = optimizer.update(jax.tree.map(jnp.ones_like, params), optimizer.init(params), params)
    np.testing.assert_allclose(updates["trainable"]["w"], -0.1)
    np.testing.assert_array_equal(updates["frozen"]["w"], 0.0)
    assert updates["optional"] is None


def test_low_precision_moments_keep_fp32_math() -> None:
    params = {"weight": jnp.ones((2, 2), jnp.bfloat16)}
    grads = {"weight": jnp.full((2, 2), 0.25, jnp.bfloat16)}
    optimizer = create_optimizer(lr=1e-3, weight_decay=0.0, max_grad_norm=0.0, mu_dtype="bfloat16", nu_dtype="float32")
    state = optimizer.init(params)
    moments = sorted(str(leaf.dtype) for leaf in _array_leaves(state) if leaf.shape == (2, 2))
    assert moments == ["bfloat16", "float32"]
    updates, next_state = optimizer.update(grads, state, params)
    assert updates["weight"].dtype == jnp.float32
    _assert_finite((updates, next_state))
    with pytest.raises(ValueError, match="moment dtype"):
        create_optimizer(mu_dtype="float8")
    with pytest.raises(ValueError, match="adamw and nadamw only"):
        create_optimizer(optimizer_type="lion", mu_dtype="bfloat16")


def test_clipping_scales_finite_norms_and_ignores_nonfinite() -> None:
    clip = clip_by_global_norm(1.0)
    big = {"a": jnp.full((4,), 2.0)}
    clipped, _ = clip.update(big, clip.init(big))
    np.testing.assert_allclose(float(optax.tree.norm(clipped)), 1.0, rtol=1e-6)
    bad = {"a": jnp.asarray([jnp.nan, 1.0])}
    passthrough, _ = clip.update(bad, clip.init(bad))
    np.testing.assert_array_equal(np.isnan(passthrough["a"]), [True, False])


def test_spike_guard_rejects_nan_without_loss_under_jit() -> None:
    params = {"w": jnp.ones((2, 2)), "optional": None}
    bad = {"w": jnp.full((2, 2), jnp.nan), "optional": None}
    guard = skip_step_on_spikes(optax.adam(1e-2), interval=4, scaling_factor=2.0)
    initial = guard.init(params)
    updates, state = jax.jit(guard.update)(bad, initial, params)
    np.testing.assert_array_equal(updates["w"], 0.0)
    assert updates["optional"] is None
    assert bool(state["is_skipped"]) and int(state["skip_reason"]) == SPIKE_SKIP_NONFINITE_GRAD_NORM
    assert int(state["loss_count"]) == 0 and int(state["grad_count"]) == 0
    _assert_equal(state["inner_state"], initial["inner_state"])


def test_spike_guard_rejects_nan_loss_and_keeps_history() -> None:
    params = jnp.ones((3,))
    guard = skip_step_on_spikes(optax.adam(1e-2), interval=4, scaling_factor=2.0)
    state = guard.init(params)
    accepted, state = guard.update(jnp.ones((3,)), state, params, loss=1.0)
    assert bool(jnp.any(accepted != 0)) and int(state["skip_reason"]) == SPIKE_SKIP_NONE
    rejected, after = guard.update(jnp.ones((3,)), state, params, loss=jnp.asarray(jnp.nan))
    np.testing.assert_array_equal(rejected, 0.0)
    assert int(after["skip_reason"]) == SPIKE_SKIP_NONFINITE_LOSS
    assert int(after["loss_count"]) == int(state["loss_count"])
    _assert_equal(after["inner_state"], state["inner_state"])


def test_spike_guard_reports_statistical_spikes() -> None:
    params = jnp.zeros((1,))
    guard = skip_step_on_spikes(optax.sgd(0.1), interval=4, scaling_factor=2.0)
    state = guard.init(params)
    for _ in range(2):
        _, state = guard.update(jnp.ones((1,)), state, params, loss=1.0)
    updates, state = guard.update(jnp.ones((1,)), state, params, loss=10.0)
    np.testing.assert_array_equal(updates, 0.0)
    assert int(state["skip_reason"]) == SPIKE_SKIP_LOSS_SPIKE
    updates, state = guard.update(jnp.full((1,), 100.0), state, params, loss=1.0)
    assert int(state["skip_reason"]) == SPIKE_SKIP_GRAD_NORM_SPIKE
    assert int(state["loss_count"]) == 2 and int(state["grad_count"]) == 2


def test_create_optimizer_wires_the_spike_guard() -> None:
    params = {"w": jnp.ones((2,))}
    optimizer = create_optimizer(lr=1e-3, weight_decay=0.0, max_grad_norm=0.0, skip_spike_steps=True, spike_interval=4)
    updates, state = optimizer.update({"w": jnp.full((2,), jnp.inf)}, optimizer.init(params), params)
    np.testing.assert_array_equal(updates["w"], 0.0)
    assert int(state["skip_reason"]) == SPIKE_SKIP_NONFINITE_GRAD_NORM


def _muon_rms(consistent_rms: Any = ...) -> list[float]:
    values = []
    for index, shape in enumerate(((8, 32), (16, 32), (32, 32), (32, 16), (32, 8))):
        params = {"weight": jnp.zeros(shape)}
        grads = {"weight": jax.random.normal(jax.random.key(index + 123), shape)}
        kwargs = {} if consistent_rms is ... else {"muon_consistent_rms": consistent_rms}
        optimizer = create_optimizer(lr=1.0, weight_decay=0.0, optimizer_type="muon", max_grad_norm=0.0, **kwargs)
        updates, _ = optimizer.update(grads, optimizer.init(params), params)
        values.append(float(jnp.sqrt(jnp.mean(jnp.square(updates["weight"])))))
    return values


def test_muon_defaults_to_shape_consistent_rms() -> None:
    default, legacy = _muon_rms(), _muon_rms(None)
    np.testing.assert_allclose(default, _muon_rms(0.2), rtol=1e-6)
    assert max(default) / min(default) < 1.25 < 1.8 < max(legacy) / min(legacy)
    with pytest.raises(ValueError, match="muon_consistent_rms"):
        create_optimizer(optimizer_type="muon", muon_consistent_rms=0.0)


def test_unknown_optimizer_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown optimizer"):
        create_optimizer(optimizer_type="adamw_8bit")
