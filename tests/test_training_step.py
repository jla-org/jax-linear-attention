# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The compiled train step: accumulation, normalization, rejection of unsafe updates, eval."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from _toy_model import TOY_MODEL, ToyLM, build_toy
from jla.training.step import (
    STEP_SKIP_EMPTY_WEIGHTED_BATCH,
    STEP_SKIP_NONE,
    STEP_SKIP_NONFINITE_GRAD,
    STEP_SKIP_NONFINITE_LOSS,
    LossConfig,
    build_eval_step,
    build_train_step,
    evaluate,
    lm_loss,
    resolve_scan_unroll,
    validate_batch,
)


class TinyRegressor(eqx.Module):
    weight: jax.Array


def _squared_loss(model, batch, key):
    del key
    inputs = batch["input_ids"].astype(jnp.float32)
    targets = batch["labels"].astype(jnp.float32)
    mask = batch.get("loss_mask", jnp.ones_like(inputs)).astype(jnp.float32)
    loss_sum = jnp.sum(jnp.square(model.weight * inputs - targets) * mask)
    return loss_sum, {
        "loss_sum": loss_sum,
        "tokens": jnp.sum(mask),
        "aux_loss": jnp.asarray(2.0, jnp.float32),
        "probe_mean": jnp.mean(inputs),
        "probe_count_sum": jnp.asarray(inputs.size, jnp.float32),
    }


def _batch(mask=None):
    batch = {"input_ids": jnp.asarray([[[1, 2]], [[3, 4]]], jnp.int32), "labels": jnp.zeros((2, 1, 2), jnp.int32)}
    if mask is not None:
        batch["loss_mask"] = mask
    return batch


def _regressor(optimizer, loss_fn=_squared_loss):
    model = TinyRegressor(jnp.asarray(1.0, jnp.float32))
    return model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)), build_train_step(optimizer, loss_fn)


def test_token_weighted_accumulation_updates_once_and_normalizes_metrics() -> None:
    model, opt_state, step = _regressor(optax.sgd(0.1))
    key = jax.random.key(7)
    before_key = np.asarray(jax.random.key_data(key)).copy()
    model, _, metrics, new_key = step(model, opt_state, _batch(), key)
    # sum(x^2) = 30, d/dw = 60; normalized by 4 tokens -> grad 15, w = 1 - 0.1 * 15.
    np.testing.assert_allclose(np.asarray(model.weight), -0.5, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["loss"]), 7.5, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["aux_loss"]), 2.0, rtol=1e-6)
    # Plain keys average over microbatches; *_sum keys divide by total tokens.
    np.testing.assert_allclose(float(metrics["probe_mean"]), 2.5, rtol=1e-6)
    np.testing.assert_allclose(float(metrics["probe_count"]), 1.0, rtol=1e-6)
    assert bool(metrics["step_applied"]) and int(metrics["skip_reason"]) == STEP_SKIP_NONE
    assert float(metrics["tokens"]) == 4.0 and int(metrics["grad_accum"]) == 2
    assert not np.array_equal(np.asarray(jax.random.key_data(new_key)), before_key)


def test_empty_weighted_batch_skips_adamw_moments_and_weight_decay() -> None:
    model, opt_state, step = _regressor(optax.adamw(0.1, weight_decay=0.5))
    before = [np.asarray(x).copy() for x in jax.tree.leaves((model, opt_state))]
    model, opt_state, metrics, _ = step(model, opt_state, _batch(jnp.zeros((2, 1, 2), jnp.float32)), jax.random.key(0))
    for expected, actual in zip(before, jax.tree.leaves((model, opt_state)), strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))
    assert not bool(metrics["step_applied"])
    assert int(metrics["skip_reason"]) == STEP_SKIP_EMPTY_WEIGHTED_BATCH


def test_nonfinite_loss_is_rejected_with_its_origin() -> None:
    def nan_loss(model, batch, key):
        value = model.weight * jnp.asarray(jnp.nan, jnp.float32)
        return value, {"loss_sum": value, "tokens": jnp.asarray(1.0, jnp.float32)}

    model, opt_state, step = _regressor(optax.sgd(0.1), nan_loss)
    model, _, metrics, _ = step(model, opt_state, _batch(), jax.random.key(0))
    assert float(model.weight) == 1.0
    reason = int(metrics["skip_reason"])
    assert reason & STEP_SKIP_NONFINITE_LOSS and reason & STEP_SKIP_NONFINITE_GRAD
    assert not bool(metrics["step_applied"])


def test_loss_fn_must_report_tokens() -> None:
    def no_tokens(model, batch, key):
        return model.weight, {"loss_sum": model.weight}

    model, opt_state, step = _regressor(optax.sgd(0.1), no_tokens)
    with pytest.raises(KeyError, match="tokens"):
        step(model, opt_state, _batch(), jax.random.key(0))


def test_validate_batch_schema_and_dtypes() -> None:
    validate_batch(_batch(), (2, 1, 2))
    with pytest.raises(KeyError, match="missing"):
        validate_batch({"input_ids": _batch()["input_ids"]})
    with pytest.raises(KeyError, match="unsupported"):
        validate_batch({**_batch(), "extra": jnp.ones((2, 1, 2))})
    with pytest.raises(TypeError, match="integer"):
        validate_batch({**_batch(), "input_ids": _batch()["input_ids"].astype(jnp.float32)})
    with pytest.raises(ValueError, match="shape"):
        validate_batch(_batch(), (1, 1, 2))
    assert resolve_scan_unroll(0, 4) == 4 and resolve_scan_unroll(0, 8) == 1 and resolve_scan_unroll(16, 3) == 3


def _lm_batch(seed: int, shape=(2, 4, 16)):
    tokens = np.random.default_rng(seed).integers(0, TOY_MODEL.vocab_size, size=(*shape[:2], shape[2] + 1))
    return {"input_ids": jnp.asarray(tokens[..., :-1], jnp.int32), "labels": jnp.asarray(tokens[..., 1:], jnp.int32)}


def _lm_step(model_config=TOY_MODEL):
    model = build_toy(model_config, key=jax.random.key(0))
    optimizer = optax.adamw(1e-2, weight_decay=0.1)
    params = eqx.filter(model, eqx.is_inexact_array)
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype="float32")))
    return model, optimizer.init(params), step


def test_nonfinite_parameter_injection_leaves_state_untouched() -> None:
    model, opt_state, step = _lm_step()
    model, opt_state, metrics, key = step(model, opt_state, _lm_batch(0), jax.random.key(1))
    assert bool(metrics["step_applied"])
    poisoned = eqx.tree_at(lambda m: m.proj, model, model.proj.at[0, 0].set(jnp.nan))
    before = [np.asarray(x).copy() for x in jax.tree.leaves(eqx.filter((poisoned, opt_state), eqx.is_array))]
    new_model, new_opt_state, metrics, _ = step(poisoned, opt_state, _lm_batch(1), key)
    assert not bool(metrics["step_applied"])
    assert int(metrics["skip_reason"]) & STEP_SKIP_NONFINITE_LOSS
    after = jax.tree.leaves(eqx.filter((new_model, new_opt_state), eqx.is_array))
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_array_equal(expected, np.asarray(actual))


def test_bf16_parameters_stay_bf16_after_fp32_updates() -> None:
    import dataclasses

    model, opt_state, step = _lm_step(dataclasses.replace(TOY_MODEL, param_dtype="bfloat16"))
    model, opt_state, metrics, _ = step(model, opt_state, _lm_batch(0), jax.random.key(0))
    assert bool(metrics["step_applied"])
    assert {leaf.dtype for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))} == {jnp.dtype(jnp.bfloat16)}


def test_step_matches_a_reference_gradient_over_the_whole_batch() -> None:
    model, opt_state, step = _lm_step()
    batch = _lm_batch(3)
    loss_fn = lm_loss(LossConfig(compute_dtype="float32"))

    def full_loss(params):
        m = eqx.combine(params, eqx.filter(model, eqx.is_inexact_array, inverse=True))
        flat = {k: v.reshape(1, -1, v.shape[-1])[0] for k, v in batch.items()}
        objective, aux = loss_fn(m, flat, jax.random.key(0))
        return objective / aux["tokens"]

    params = eqx.filter(model, eqx.is_inexact_array)
    expected_loss, grads = jax.value_and_grad(full_loss)(params)
    reference = optax.adamw(1e-2, weight_decay=0.1)
    updates, _ = reference.update(grads, reference.init(params), params)
    expected = optax.apply_updates(params, updates)
    new_model, _, metrics, _ = step(model, opt_state, batch, jax.random.key(0))
    np.testing.assert_allclose(float(metrics["loss"]), float(expected_loss), rtol=1e-5)
    for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(eqx.filter(new_model, eqx.is_inexact_array)), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-6)


def test_eval_step_is_jitted_and_token_weighted() -> None:
    model = build_toy(TOY_MODEL, key=jax.random.key(0))
    eval_step = build_eval_step(LossConfig(compute_dtype="float32"))
    batches = [{k: v[0] for k, v in _lm_batch(i).items()} for i in range(3)]
    batches[1]["loss_mask"] = jnp.zeros((4, 16), jnp.bool_).at[:, :4].set(True)
    metrics = evaluate(eval_step, model, batches)
    assert metrics["eval_batches"] == 3.0 and metrics["eval_tokens"] == 4 * 16 * 2 + 4 * 4
    expected_sum = 0.0
    for batch in batches:
        logits = model.logits(model(batch["input_ids"]).hidden).reshape(-1, TOY_MODEL.vocab_size)
        nll = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, batch["labels"].reshape(-1, 1), -1)[:, 0]
        mask = batch.get("loss_mask", jnp.ones_like(batch["labels"], bool)).reshape(-1)
        expected_sum += float(jnp.sum(jnp.where(mask, nll, 0.0)))
    np.testing.assert_allclose(metrics["eval_loss"], expected_sum / metrics["eval_tokens"], rtol=1e-5)
    with pytest.raises(ValueError, match="no batches"):
        evaluate(eval_step, model, [])


def test_mtp_and_moe_metrics_pass_through_the_step() -> None:
    from jla.models import build_model, get_preset

    for preset, expected in (("tiny-hybrid", ("mtp/loss", "mtp/loss_1", "mtp/loss_2")), ("tiny-moe", ("moe/balance_loss",))):
        config = get_preset(preset)
        model = build_model(config, key=jax.random.key(0))
        optimizer = optax.adamw(1e-3)
        step = build_train_step(optimizer, lm_loss(LossConfig()))
        segment_ids = jnp.broadcast_to(jnp.repeat(jnp.arange(2, dtype=jnp.int32), 8), (2, 2, 16))
        batch = {**_lm_batch(5, (2, 2, 16)), "segment_ids": segment_ids}  # token ids < 128 <= vocab
        _, _, metrics, _ = step(model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)), batch, jax.random.key(1))
        assert bool(metrics["step_applied"])
        for name in expected:
            assert np.isfinite(float(metrics[name])) and float(metrics[name]) > 0, (preset, name)
        if preset == "tiny-hybrid":
            assert float(metrics["mtp_loss"]) == pytest.approx(0.3 * float(metrics["mtp/loss"]), rel=1e-5)
        else:
            assert float(metrics["aux_loss"]) > 0


def test_model_metrics_may_not_shadow_step_metrics() -> None:
    class Shadowing(ToyLM):
        def __call__(self, input_ids, **kwargs):
            out = super().__call__(input_ids, **kwargs)
            return out._replace(metrics={"loss": jnp.zeros(())})

    model = Shadowing(TOY_MODEL, key=jax.random.key(0))
    optimizer = optax.sgd(0.1)
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype="float32")))
    with pytest.raises(ValueError, match="collides"):
        step(model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)), _lm_batch(0), jax.random.key(0))
