# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 training (jla.layers.fp8, jla.training.fp8): matmuls against a NumPy emulation, model
conversion, loss tracking against bf16, the delayed-scaling state in the train step, and
checkpoint/resume of that state."""

from __future__ import annotations

import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest

from _toy_model import experiment
from jla.config import config_from_mapping
from jla.layers import fp8 as F
from jla.models import build_model, get_preset, logical_axes
from jla.training.fp8 import fp8_state_mask, without_fp8_state
from jla.training.mixed_precision import MixedPrecisionPolicy, check_param_dtypes, is_fp8_capable
from jla.training.step import LossConfig, build_train_step, lm_loss
from jla.training.trainer import Trainer

E4M3_MAX, E5M2_MAX = 448.0, 57344.0
_DEF = re.compile(r"^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*([a-z0-9]+)\[")  # HLO "%name = type[" definitions
_DOT = re.compile(r"\sdot\((%[\w.\-]+),\s*(%[\w.\-]+)\)")


# -- NumPy reference (ml_dtypes rounding, float64 GEMMs) ---------------------------------------


def _q(x, scale, dtype) -> np.ndarray:
    """Values of ``clip(x / scale)`` rounded to the FP8 ``dtype`` (float64, still divided by scale)."""
    bound = float(ml_dtypes.finfo(dtype).max)
    scaled = np.asarray(x, np.float32) / np.float32(scale)
    return np.clip(scaled, -bound, bound).astype(dtype).astype(np.float64)


def _reference(x, w, g, sx, sw, sg):
    """Forward and both backward GEMMs of x [..., K] @ w [K, *R] with the given scales."""
    qx, qw, qg = _q(x, sx, ml_dtypes.float8_e4m3fn), _q(w, sw, ml_dtypes.float8_e4m3fn), _q(g, sg, ml_dtypes.float8_e5m2)
    nb, nr = qx.ndim - 1, qw.ndim - 1
    y = np.tensordot(qx, qw, axes=([nb], [0])) * sx * sw
    dx = np.tensordot(qg, qw, axes=(list(range(nb, nb + nr)), list(range(1, 1 + nr)))) * sg * sw
    dw = np.tensordot(qx, qg, axes=(list(range(nb)), list(range(nb)))) * sx * sg
    return y, dx, dw


def _close(actual, expected, rel=2e-3) -> None:
    expected = np.asarray(expected, np.float64)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    np.testing.assert_allclose(np.asarray(actual, np.float64), expected, rtol=0, atol=rel * scale)


def _operands(w_shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 8, w_shape[0])).astype(np.float32)
    w = (0.1 * rng.standard_normal(w_shape)).astype(np.float32)
    g = rng.standard_normal((2, 8, *w_shape[1:])).astype(np.float32)
    return x, w, g


def test_scale_quantize_and_history_helpers() -> None:
    np.testing.assert_allclose(F.scale_from_amax(jnp.float32(8.96), E4M3_MAX, 1.0), 0.02, rtol=1e-6)
    for bad in (0.0, np.inf, np.nan):
        assert float(F.scale_from_amax(jnp.float32(bad), E4M3_MAX, 0.5)) == 0.5
        # A scale is never 0 (or not finite), even from a zeroed state: it falls back to 1.
        assert float(F.scale_from_amax(jnp.float32(0.0), E4M3_MAX, bad)) == 1.0
    q = F.quantize(jnp.array([1e6, -1e6, 0.3]), jnp.float32(1.0), F.E4M3)
    assert q.dtype == F.E4M3
    np.testing.assert_array_equal(np.asarray(q, np.float32)[:2], [448.0, -448.0])  # saturates, never NaN
    np.testing.assert_array_equal(F.push_amax(jnp.array([3.0, 2.0, 1.0]), jnp.float32(9.0)), [9.0, 3.0, 2.0])
    state = F.Fp8State.init(16, (3,))
    assert state.input_scale.shape == (3,) and state.grad_amax_history.shape == (3, 16)
    with pytest.raises(ValueError, match="at least one"):
        F.Fp8State.init(0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("w_shape", [(32, 48), (32, 2, 24)], ids=["linear", "stacked"])
def test_delayed_fp8_dot_matches_numpy_emulation(w_shape, dtype: str) -> None:
    x, w, g = _operands(w_shape)
    # Activations and output gradients arrive in the compute dtype; the reference sees the same values.
    x, g = (np.asarray(jnp.asarray(v, dtype), np.float32) for v in (x, g))
    history = 8
    state = F.Fp8State.init(history)
    state = eqx.tree_at(
        lambda s: (s.input_amax_history, s.kernel_amax_history, s.grad_amax_history),
        state,
        (jnp.zeros(history).at[:2].set([3.0, 6.0]), jnp.zeros(history).at[0].set(0.5), jnp.zeros(history).at[3].set(7.0)),
    )
    sx, sw, sg = 6.0 / E4M3_MAX, 0.5 / E4M3_MAX, 7.0 / E5M2_MAX  # max of each history / fp8 max
    y, vjp = jax.vjp(lambda a, b, s: F.fp8_dot(a, b, s, recipe="delayed"),
                     jnp.asarray(x, dtype), jnp.asarray(w), state)
    dx, dw, new_state = vjp(jnp.asarray(g, dtype))
    y_ref, dx_ref, dw_ref = _reference(x, w, g, sx, sw, sg)
    # y and dx come out in the compute dtype, dW in the weight's float32.
    assert y.dtype == jnp.dtype(dtype) and dx.dtype == jnp.dtype(dtype) and dw.dtype == jnp.float32
    # bfloat16 output: 2^-9 rounding of each value plus 2^-9 rounding of the scale product.
    rel = 2e-3 if dtype == "float32" else 1e-2
    _close(y, y_ref, rel)
    _close(dx, dx_ref, rel)
    _close(dw, dw_ref)
    # The "gradient" of the state is its next value: the scales used now, histories shifted by one.
    np.testing.assert_allclose([new_state.input_scale, new_state.kernel_scale, new_state.grad_scale], [sx, sw, sg], rtol=1e-6)
    for name, value in (("input", x), ("kernel", w), ("grad", g)):
        old = np.asarray(getattr(state, f"{name}_amax_history"))
        new = np.asarray(getattr(new_state, f"{name}_amax_history"))
        np.testing.assert_allclose(new[0], np.max(np.abs(value)), rtol=1e-6)
        np.testing.assert_array_equal(new[1:], old[:-1])


@pytest.mark.parametrize("w_shape", [(32, 48), (32, 2, 24)], ids=["linear", "stacked"])
def test_current_fp8_dot_matches_numpy_emulation(w_shape) -> None:
    x, w, g = _operands(w_shape, seed=1)
    sx, sw, sg = (float(np.max(np.abs(v))) / m for v, m in ((x, E4M3_MAX), (w, E4M3_MAX), (g, E5M2_MAX)))
    y, vjp = jax.vjp(lambda a, b: F.fp8_dot(a, b, None, recipe="current"), jnp.asarray(x), jnp.asarray(w))
    dx, dw = vjp(jnp.asarray(g))
    y_ref, dx_ref, dw_ref = _reference(x, w, g, sx, sw, sg)
    _close(y, y_ref)
    _close(dx, dx_ref)
    _close(dw, dw_ref)
    # FP8 is an approximation of the exact product: e4m3 keeps ~2^-4 relative error per operand.
    exact = np.tensordot(x.astype(np.float64), w.astype(np.float64), axes=([2], [0]))
    assert np.max(np.abs(np.asarray(y) - exact)) < 0.1 * np.max(np.abs(exact))


def test_fp8_dot_rejects_bad_arguments() -> None:
    x, w = jnp.ones((4, 32)), jnp.ones((32, 16))
    with pytest.raises(ValueError, match="needs the matmul's Fp8State"):
        F.fp8_dot(x, w, None, recipe="delayed")
    with pytest.raises(ValueError, match="keeps no state"):
        F.fp8_dot(x, w, F.Fp8State.init(4), recipe="current")
    with pytest.raises(ValueError, match="unknown FP8 recipe"):
        F.fp8_dot(x, w, None, recipe="mxfp8")
    with pytest.raises(ValueError, match="contracts"):
        F.fp8_dot(x, jnp.ones((16, 16)), None, recipe="current")
    with pytest.raises(ValueError, match="fp8 must be"):
        get_preset("tiny").replace(fp8="e4m3")
    with pytest.raises(ValueError, match="fp8_amax_history"):
        get_preset("tiny").replace(fp8="delayed", fp8_amax_history=0)
    with pytest.raises(ValueError, match="compute_dtype"):
        get_preset("tiny").replace(fp8="current", compute_dtype="float16")
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        F.fp8_dot(x.astype(jnp.float16), w, None, recipe="current")


# -- model conversion -----------------------------------------------------------------------------


@pytest.mark.parametrize("scan_layers", [True, False])
def test_build_model_converts_projections_and_keeps_parameters(scan_layers: bool) -> None:
    base = get_preset("tiny").replace(scan_layers=scan_layers)
    key = jax.random.key(0)
    ref = build_model(base, key=key)
    model = build_model(base.replace(fp8="delayed", fp8_amax_history=32), key=key)
    modules = F.fp8_modules(model)
    # Per block: q, k, v, o projections, the SwiGLU gate/up GEMM and its down projection.
    assert len(modules) == 6 * (1 if scan_layers else base.n_layers)
    assert type(model.embed) is type(ref.embed) and not F.fp8_modules(model.embed)
    for expected, actual in zip(jax.tree.leaves(eqx.filter(ref, eqx.is_array)),
                                jax.tree.leaves(without_fp8_state(eqx.filter(model, eqx.is_array))), strict=True):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    states = [m.state for m in modules]
    lead = (base.n_layers,) if scan_layers else ()
    assert all(s.input_scale.shape == lead and s.input_amax_history.shape == (*lead, 32) for s in states)
    axes = logical_axes(model)
    is_axes = lambda x: isinstance(x, tuple) and all(a is None or isinstance(a, str) for a in x)  # noqa: E731
    flat_axes = jax.tree_util.tree_flatten_with_path(axes, is_leaf=is_axes)[0]
    history_axes = {a for path, a in flat_axes if "amax_history" in jax.tree_util.keystr(path)}
    assert history_axes == {(("layers", None) if scan_layers else (None,))}
    current = build_model(base.replace(fp8="current"), key=key)
    assert all(m.state is None for m in F.fp8_modules(current)) and len(F.fp8_modules(current)) == len(modules)
    assert fp8_state_mask(eqx.filter(current, eqx.is_inexact_array)) is None
    out = model(jnp.zeros((2, 16), jnp.int32))
    assert out.hidden.shape == (2, 16, base.d_model) and bool(jnp.all(jnp.isfinite(out.hidden)))


def test_policy_records_the_fp8_switch_and_dtype_check_skips_its_state() -> None:
    cfg = get_preset("tiny").replace(fp8="delayed", param_dtype="bfloat16")
    policy = MixedPrecisionPolicy.from_model_config(cfg)
    assert policy.fp8 == "delayed" and policy.as_dict()["fp8"] == "delayed"
    assert MixedPrecisionPolicy().fp8 == "off"
    with pytest.raises(ValueError, match="fp8 must be"):
        MixedPrecisionPolicy(fp8="on")
    with pytest.raises(TypeError, match="ModelConfig.fp8"):
        MixedPrecisionPolicy(compute_dtype=jnp.float8_e4m3fn)
    model = build_model(cfg, key=jax.random.key(0))
    check_param_dtypes(eqx.filter(model, eqx.is_inexact_array), "bfloat16")  # the float32 state is exempt


def test_experiment_config_switch_defaults_off() -> None:
    assert config_from_mapping({"model": {"preset": "tiny"}}).model.fp8 == "off"
    config = config_from_mapping({"model": {"preset": "tiny", "fp8": "delayed", "fp8_amax_history": 16}})
    assert config.model.fp8 == "delayed" and config.model.fp8_amax_history == 16
    assert config_from_mapping({"model": {"preset": "tiny"}}, ["model.fp8=current"]).model.fp8 == "current"
    with pytest.raises((TypeError, ValueError)):
        config_from_mapping({"model": {"preset": "tiny", "fp8": "e4m3"}})


# -- the train step -----------------------------------------------------------------------------------


def _batches(steps: int, grad_accum: int = 1, batch: int = 8, seq: int = 32, vocab: int = 512, seed: int = 0):
    """A learnable stream: each sequence follows ``t -> (5 t + 3) mod vocab`` from a random start."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        start = rng.integers(0, vocab, size=(grad_accum, batch, 1))
        ids = [start]
        for _ in range(seq):
            ids.append((5 * ids[-1] + 3) % vocab)
        ids = np.concatenate(ids, axis=-1).astype(np.int32)
        out.append({"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])})
    return out


def _run(cfg, batches, *, lr: float = 3e-3, model=None):
    model = build_model(cfg, key=jax.random.key(0)) if model is None else model
    optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype=cfg.compute_dtype)), donate=False)
    key = jax.random.key(1)
    losses = []
    for batch in batches:
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
    return model, np.array(losses), metrics


#: Stated tolerance of FP8 against bf16 on the tiny model (40 AdamW steps, loss 6.25 -> 2.57). The
#: largest gaps measured on this test are 0.008 / 0.010 (delayed / current) on an H100 and 0.010 on
#: CPU, so the bound (0.076 at the end, 0.11 at the start) leaves a 5-7x margin.
LOSS_ATOL = 0.05
LOSS_RTOL = 0.01


@pytest.mark.parametrize("recipe", ["delayed", "current"])
def test_fp8_loss_tracks_bf16(recipe: str) -> None:
    base = get_preset("tiny")  # compute dtype bfloat16
    batches = _batches(40)
    _, bf16, _ = _run(base, batches)
    _, fp8, metrics = _run(base.replace(fp8=recipe), batches)
    assert bf16[-1] < bf16[0] - 2.0, bf16  # the task is learnable
    diff = np.abs(fp8 - bf16)
    assert np.all(diff <= LOSS_ATOL + LOSS_RTOL * bf16), (recipe, np.round(fp8, 3), np.round(bf16, 3))
    assert bool(metrics["step_applied"]) and np.isfinite(float(metrics["grad_norm"]))


def _states(model) -> list[F.Fp8State]:
    return [m.state for m in F.fp8_modules(model)]


@pytest.mark.parametrize("preset", ["tiny-hybrid", "tiny-moe", "tiny-mamba2"])
def test_fp8_trains_hybrid_moe_ssm_and_mtp_presets(preset: str) -> None:
    """Gated DeltaNet + attention + MTP heads, an MoE shared expert, Mamba-2 stacked projections."""
    cfg = get_preset(preset).replace(fp8="delayed", fp8_amax_history=4)
    model = build_model(cfg, key=jax.random.key(0))
    assert F.fp8_modules(model)
    logical_axes(model)
    trained, losses, metrics = _run(cfg, _batches(2), model=model)
    assert bool(metrics["step_applied"]) and np.all(np.isfinite(losses)), losses
    for state in _states(trained):
        for name in ("input", "kernel"):
            assert np.all(np.asarray(getattr(state, f"{name}_amax_history"))[..., :2] > 0)
        assert np.all(np.isfinite(np.asarray(state.grad_scale)))


def test_delayed_state_advances_once_per_step_with_the_max_over_microbatches() -> None:
    cfg = get_preset("tiny").replace(fp8="delayed", fp8_amax_history=4)
    init = build_model(cfg, key=jax.random.key(0))
    (b1,), (b2,) = _batches(1, seed=1), _batches(1, seed=2)
    both = {name: jnp.concatenate([b1[name], b2[name]]) for name in b1}
    one, _, _ = _run(cfg, [b1], model=init)
    two, _, _ = _run(cfg, [b2], model=init)
    accum, _, metrics = _run(cfg, [both], model=init)
    assert int(metrics["grad_accum"]) == 2
    for s0, s1, s2, s12 in zip(_states(init), _states(one), _states(two), _states(accum), strict=True):
        for name in ("input", "kernel", "grad"):
            h1, h2, h12 = (np.asarray(getattr(s, f"{name}_amax_history")) for s in (s1, s2, s12))
            np.testing.assert_allclose(h12[..., 0], np.maximum(h1[..., 0], h2[..., 0]), rtol=1e-6)
            np.testing.assert_array_equal(h12[..., 1:], np.asarray(getattr(s0, f"{name}_amax_history"))[..., :-1])
            assert np.all(h12[..., 0] > 0)
            # First step: empty history, so the scale stays at its initial 1.
            np.testing.assert_array_equal(np.asarray(getattr(s12, f"{name}_scale")), 1.0)
    # Second step: scales come from the recorded amaxes.
    after, _, _ = _run(cfg, _batches(1, seed=3), model=accum)
    for s12, s3 in zip(_states(accum), _states(after), strict=True):
        for name, bound in (("input", E4M3_MAX), ("kernel", E4M3_MAX), ("grad", E5M2_MAX)):
            expected = np.max(np.asarray(getattr(s12, f"{name}_amax_history")), axis=-1) / bound
            np.testing.assert_allclose(np.asarray(getattr(s3, f"{name}_scale")), expected, rtol=1e-6)
            np.testing.assert_array_equal(np.asarray(getattr(s3, f"{name}_amax_history"))[..., 1:],
                                          np.asarray(getattr(s12, f"{name}_amax_history"))[..., :-1])


def test_rejected_step_keeps_fp8_state_and_state_never_reaches_the_optimizer() -> None:
    cfg = get_preset("tiny").replace(fp8="delayed", fp8_amax_history=4)
    init = build_model(cfg, key=jax.random.key(0))
    (batch,) = _batches(1)
    empty = {**batch, "labels": jnp.full_like(batch["labels"], -100)}
    skipped, _, metrics = _run(cfg, [empty], model=init)
    assert not bool(metrics["step_applied"])
    for before, after in zip(jax.tree.leaves(_states(init)), jax.tree.leaves(_states(skipped)), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    # With weight decay and a large learning rate the state still holds exactly the recorded values.
    optimizer = optax.adamw(1.0, weight_decay=0.5)
    opt_state = optimizer.init(eqx.filter(init, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig()), donate=False)
    model, opt_state, metrics, _ = step(init, opt_state, batch, jax.random.key(1))
    assert bool(metrics["step_applied"])
    for state in _states(model):
        np.testing.assert_array_equal(np.asarray(state.input_scale), 1.0)
        assert np.all(np.asarray(state.input_amax_history)[..., 1:] == 0.0)
    moments = [leaf for leaf in jax.tree.leaves(opt_state) if hasattr(leaf, "shape") and leaf.shape[-1:] == (4,)]
    assert moments and all(not np.any(np.asarray(leaf)) for leaf in moments)  # zero "gradients" for the state


def test_unused_and_repeated_matmuls_advance_their_state_once() -> None:
    """No cotangent keeps the state (it is not zeroed); n uses in one loss advance it once."""
    from jla.training.fp8 import apply_fp8_state, split_fp8_state

    rng = np.random.default_rng(0)
    x1, x2 = (jnp.asarray(rng.standard_normal((4, 32)), jnp.float32) for _ in range(2))
    w = jnp.asarray(0.1 * rng.standard_normal((32, 16)), jnp.float32)
    hist = jnp.array([2.0, 4.0, 0.0, 0.0], jnp.float32)
    start = F.Fp8State(jnp.float32(0.5), hist, jnp.float32(0.5), hist, jnp.float32(0.5), hist)
    params = {"once": start, "twice": start, "unused": start, "w": w}

    def loss(p):
        once = F.fp8_dot(x1, p["w"], p["once"], recipe="delayed")
        twice = F.fp8_dot(x1, p["w"], p["twice"], recipe="delayed") + F.fp8_dot(x2, p["w"], p["twice"], recipe="delayed")
        unused = jax.lax.stop_gradient(F.fp8_dot(x1, p["w"], p["unused"], recipe="delayed"))
        return jnp.sum(once**2) + jnp.sum(twice**2) + 0.0 * jnp.sum(unused)

    mask = fp8_state_mask(params)
    _, values = split_fp8_state(jax.grad(loss)(params), mask)
    assert not any(np.any(np.asarray(leaf)) for leaf in jax.tree.leaves(values["unused"]))  # a zero "gradient"
    out = apply_fp8_state(params, params, values, mask, jnp.bool_(True))
    ax1, ax2, aw = (float(jnp.max(jnp.abs(t))) for t in (x1, x2, w))
    for name, uses in (("once", 1), ("twice", 2)):
        state = out[name]
        np.testing.assert_allclose(state.input_scale, 4.0 / E4M3_MAX, rtol=1e-6)  # the scale the forward used
        np.testing.assert_allclose(state.kernel_scale, 4.0 / E4M3_MAX, rtol=1e-6)
        np.testing.assert_allclose(state.grad_scale, 4.0 / E5M2_MAX, rtol=1e-6)
        expected_x = ax1 if uses == 1 else ax1 + ax2  # n uses record the sum of their amaxes
        np.testing.assert_allclose(state.input_amax_history, [expected_x, 2.0, 4.0, 0.0], rtol=1e-6)
        np.testing.assert_allclose(state.kernel_amax_history, [uses * aw, 2.0, 4.0, 0.0], rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(state.grad_amax_history)[1:], [2.0, 4.0, 0.0])
        assert np.asarray(state.grad_amax_history)[0] > 0
    for name in ("unused",):
        for got, want in zip(jax.tree.leaves(out[name]), jax.tree.leaves(start), strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    kept = apply_fp8_state(params, params, values, mask, jnp.bool_(False))  # a rejected step
    for got, want in zip(jax.tree.leaves(kept), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_model_applied_twice_in_one_loss_does_not_compound_its_scales() -> None:
    """A loss that runs the model on two halves of the batch (as a pipeline's microbatch loop does)."""
    cfg = get_preset("tiny").replace(fp8_amax_history=16)
    single = lm_loss(LossConfig(compute_dtype=cfg.compute_dtype))

    def two_halves(model, batch, key):
        (la, xa), (lb, xb) = (single(model, {k: v[i::2] for k, v in batch.items()}, key) for i in (0, 1))
        return la + lb, {k: xa[k] + xb[k] for k in xa}

    def run(recipe, loss_fn, steps=12):
        model = build_model(cfg.replace(fp8=recipe), key=jax.random.key(0))
        optimizer = optax.adamw(3e-3, weight_decay=0.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        step = build_train_step(optimizer, loss_fn, donate=False)
        key, losses = jax.random.key(1), []
        for batch in _batches(steps):
            model, opt_state, metrics, key = step(model, opt_state, batch, key)
            assert bool(metrics["step_applied"])
            losses.append(float(metrics["loss"]))
        return model, np.array(losses)

    once, _ = run("delayed", single)
    twice, losses = run("delayed", two_halves)
    _, bf16 = run("off", two_halves)
    # Looser than LOSS_ATOL / LOSS_RTOL: the summed amaxes give up to 2x larger scales here.
    assert np.all(np.abs(losses - bf16) <= 0.15 + 0.05 * bf16), (np.round(losses, 3), np.round(bf16, 3))
    for a, b in zip(_states(once), _states(twice), strict=True):
        for name in ("input", "kernel"):
            ratio = np.asarray(getattr(b, f"{name}_scale")) / np.asarray(getattr(a, f"{name}_scale"))
            assert np.all(ratio <= 2.5), (name, ratio)  # the sum of two amaxes, not 2 ** steps


def test_dsa_indexer_without_gradient_keeps_its_fp8_state() -> None:
    """indexer_loss_coef=0 gives the lightning indexer's FP8 projections no cotangent."""
    import dataclasses

    base = get_preset("tiny-dsa")
    cfg = base.replace(fp8="delayed", fp8_amax_history=4,
                       sparse_attention=dataclasses.replace(base.sparse_attention, indexer_loss_coef=0.0))
    init = build_model(cfg, key=jax.random.key(0))
    trained, losses, metrics = _run(cfg, _batches(3), model=init)
    assert bool(metrics["step_applied"]) and np.all(np.isfinite(losses)), losses
    flat = jax.tree_util.tree_flatten_with_path(trained, is_leaf=lambda n: isinstance(n, F.Fp8State))[0]
    states = {jax.tree_util.keystr(path): s for path, s in flat if isinstance(s, F.Fp8State)}
    index = [k for k in states if "index_" in k]
    assert index and len(index) < len(states)
    for key, state in states.items():
        for name in ("input", "kernel", "grad"):
            scale = np.asarray(getattr(state, f"{name}_scale"))
            assert np.all(np.isfinite(scale) & (scale > 0)), (key, name, scale)
            history = np.asarray(getattr(state, f"{name}_amax_history"))
            if key in index:  # never reached by a cotangent: still the initial state
                np.testing.assert_array_equal(scale, 1.0)
                np.testing.assert_array_equal(history, 0.0)
            elif name != "grad":
                assert np.all(history[..., :3] > 0), (key, name)


# -- checkpoint / resume ----------------------------------------------------------------------------------


def _experiment(path: Path, max_steps: int):
    model = get_preset("tiny").replace(fp8="delayed", fp8_amax_history=16)
    return experiment(str(path), model=model, train={"max_steps": max_steps, "log_every": 1, "final_eval": False},
                      optim={"schedule_steps": 4})


def test_fp8_state_survives_checkpoint_and_resume(tmp_path: Path) -> None:
    full = Trainer(_experiment(tmp_path / "full", 4)).run()
    first = Trainer(_experiment(tmp_path / "split", 2)).run()
    second = Trainer(_experiment(tmp_path / "split", 4)).run()
    assert first.step == 2 and second.step == 4
    for state in _states(full.model):
        # One recorded amax per applied step: 4, not 2, so the resumed run restored the first two.
        assert np.all(np.count_nonzero(np.asarray(state.grad_amax_history), axis=-1) == 4)
        assert np.all(np.asarray(state.input_scale) != 1.0)
    exact = jax.default_backend() == "cpu"
    same = np.testing.assert_array_equal if exact else (lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5))
    for expected, actual in zip(jax.tree.leaves(_states(full.model)), jax.tree.leaves(_states(second.model)), strict=True):
        same(np.asarray(expected), np.asarray(actual))
    for expected, actual in zip(jax.tree.leaves(eqx.filter(full.model, eqx.is_array)),
                                jax.tree.leaves(eqx.filter(second.model, eqx.is_array)), strict=True):
        same(np.asarray(expected), np.asarray(actual))


# -- GPU: the FP8 dots become cuBLASLt FP8 GEMMs ---------------------------------------------------------


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or not is_fp8_capable(str(jax.devices()[0].device_kind)),
    reason="needs an FP8-capable NVIDIA GPU",
)
@pytest.mark.parametrize("recipe", ["delayed", "current"])
def test_compiled_train_step_uses_fp8_gemms_on_gpu(recipe: str) -> None:
    cfg = get_preset("tiny").replace(fp8=recipe)
    model = build_model(cfg, key=jax.random.key(0))
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig()), donate=False)
    (batch,) = _batches(1)
    compiled = step.lower(model, opt_state, batch, jax.random.key(1)).compile()
    text = (compiled if hasattr(compiled, "as_text") else compiled.compiled).as_text()  # equinox wraps it
    # FP8 GEMMs are cuBLASLt FP8 calls or (small shapes) Triton GEMM fusions of a dot of two FP8 values.
    types = {m.group(1): m.group(2) for m in map(_DEF.match, text.splitlines()) if m}
    fused = [m.groups() for m in map(_DOT.search, text.splitlines())
             if m and all(types.get(name, "").startswith("f8") for name in m.groups())]
    cublas = text.count('custom_call_target="__cublas$lt$matmul$f8"')
    # 6 FP8 matmuls per block, one forward and two backward GEMMs each; XLA may merge dots that share
    # an operand (q/k/v), so the count is a lower bound.
    assert len(fused) + cublas >= 12, (len(fused), cublas)
    assert any("f8e5m2" in types[name] for pair in fused for name in pair) or cublas
