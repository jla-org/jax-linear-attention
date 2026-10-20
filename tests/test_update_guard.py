# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``update_guard = "select"`` (``train.update_guard``): the rejected-step rule without a conditional.

The update is computed and every parameter and optimizer-state leaf selected elementwise, so an applied step gives
the values of ``lax.cond`` (bit for bit on CPU; on GPU to a few float32 ulps, as XLA fuses the update differently)
and a rejected one (non-finite loss) leaves the parameters and AdamW's
moments and step count untouched; the compiled step has no ``conditional``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from _toy_model import TOY_MODEL, build_toy
from jla.training.step import (
    STEP_SKIP_EMPTY_WEIGHTED_BATCH,
    STEP_SKIP_NONFINITE_LOSS,
    LossConfig,
    build_train_step,
    lm_loss,
)


def _batch(seed: int, shape=(2, 4, 16)):
    tokens = np.random.default_rng(seed).integers(0, TOY_MODEL.vocab_size, size=(*shape[:2], shape[2] + 1))
    return {"input_ids": jnp.asarray(tokens[..., :-1], jnp.int32), "labels": jnp.asarray(tokens[..., 1:], jnp.int32)}


def _setup(guard: str):
    model = build_toy(TOY_MODEL, key=jax.random.key(0))
    optimizer = optax.adamw(1e-2, weight_decay=0.1)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype="float32")), update_guard=guard,
                            donate=False)
    return model, opt_state, step


def _arrays(tree):
    return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def test_select_applies_the_same_update_as_cond_and_rejects_the_same_steps() -> None:
    results = {}
    for guard in ("cond", "select"):
        model, opt_state, step = _setup(guard)
        key = jax.random.key(1)
        model, opt_state, applied, key = step(model, opt_state, _batch(0), key)
        poisoned = eqx.tree_at(lambda m: m.proj, model, model.proj.at[0, 0].set(jnp.nan))
        before = _arrays((poisoned, opt_state))
        kept_model, kept_state, rejected, _ = step(poisoned, opt_state, _batch(1), key)
        assert bool(applied["step_applied"]) and not bool(rejected["step_applied"])
        assert int(rejected["skip_reason"]) & STEP_SKIP_NONFINITE_LOSS
        for expected, actual in zip(before, _arrays((kept_model, kept_state)), strict=True):
            np.testing.assert_array_equal(actual, expected)  # NaN where it was, untouched elsewhere
        results[guard] = (_arrays((model, opt_state)), float(applied["loss"]), float(applied["grad_norm"]))
    # Bit for bit on CPU; on GPU XLA fuses the AdamW update into other kernels around the select (a few float32
    # ulps in about 1% of the elements). The loss and gradient norm come before the update: equal either way.
    exact = jax.default_backend() == "cpu"
    for a, b in zip(results["cond"][0], results["select"][0], strict=True):
        if exact:
            np.testing.assert_array_equal(b, a)
        else:
            np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-8)
    assert results["select"][1:] == results["cond"][1:]


def test_select_compiles_no_conditional() -> None:
    model, opt_state, step = _setup("select")
    compiled = step.lower(model, opt_state, _batch(0), jax.random.key(1)).compile()
    text = getattr(compiled, "compiled", compiled).as_text()  # equinox wraps the jax Compiled
    assert " conditional(" not in text


def test_unknown_guard_is_refused() -> None:
    with pytest.raises(ValueError, match="update_guard"):
        build_train_step(optax.sgd(0.1), lm_loss(LossConfig()), update_guard="maybe")


@pytest.mark.parametrize("guard", ["cond", "select"])
def test_a_batch_without_targets_is_rejected_and_keeps_the_state(guard) -> None:
    """The other rejection rules apply the same way: a batch whose targets are all masked out is skipped under both
    guards, with the parameters and optimizer state kept bit for bit."""
    model, opt_state, step = _setup(guard)
    batch = dict(_batch(2))
    batch["loss_mask"] = jnp.zeros_like(batch["labels"])
    before = _arrays((model, opt_state))
    kept_model, kept_state, metrics, _ = step(model, opt_state, batch, jax.random.key(3))
    assert not bool(metrics["step_applied"])
    assert int(metrics["skip_reason"]) & STEP_SKIP_EMPTY_WEIGHTED_BATCH
    for expected, actual in zip(before, _arrays((kept_model, kept_state)), strict=True):
        np.testing.assert_array_equal(actual, expected)
