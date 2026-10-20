# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vocabulary-tiled LM-head cross entropy against naive full-logits cross entropy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.cross_entropy import cross_entropy_from_logits, linear_cross_entropy

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


N, D, V = 37, 16, 1000


def _inputs(seed=0, ignore_every=5):
    ks = jax.random.split(jax.random.key(seed), 3)
    hidden = jax.random.normal(ks[0], (N, D))
    weight = 0.5 * jax.random.normal(ks[1], (V, D))
    targets = jax.random.randint(ks[2], (N,), 0, V)
    targets = jnp.where(jnp.arange(N) % ignore_every == 0, -100, targets)
    return hidden, weight, targets


def _naive(hidden, weight, targets, **kw):
    logits = jnp.einsum("nd,vd->nv", hidden, weight, precision="highest")
    return cross_entropy_from_logits(logits, targets, **kw)


def _objective(fn, z_weight=0.1, **kw):
    def f(hidden, weight, targets):
        out = fn(hidden, weight, targets, **kw)
        return out.loss_sum + z_weight * out.z_loss_sum
    return f


@pytest.mark.parametrize("vocab_chunk", [128, 333, 4096])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_matches_naive_values_and_gradients(vocab_chunk, label_smoothing):
    hidden, weight, targets = _inputs()
    kw = dict(label_smoothing=label_smoothing)
    got = linear_cross_entropy(hidden, weight, targets, vocab_chunk=vocab_chunk, compute_dtype=jnp.float32, **kw)
    want = _naive(hidden, weight, targets, **kw)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=1e-5)
    assert float(got.num_valid) == float(jnp.sum(targets != -100))
    tiled = _objective(lambda h, w, t, **k: linear_cross_entropy(h, w, t, vocab_chunk=vocab_chunk,
                                                                 compute_dtype=jnp.float32, **k), **kw)
    g_got = jax.grad(tiled, argnums=(0, 1))(hidden, weight, targets)
    g_want = jax.grad(_objective(_naive, **kw), argnums=(0, 1))(hidden, weight, targets)
    for a, b in zip(g_got, g_want):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_z_loss_gradient_alone():
    hidden, weight, targets = _inputs(1)
    zf = lambda fn: lambda h, w: fn(h, w, targets).z_loss_sum  # noqa: E731
    tiled = zf(lambda h, w, t: linear_cross_entropy(h, w, t, vocab_chunk=256, compute_dtype=jnp.float32))
    for a, b in zip(jax.grad(tiled, (0, 1))(hidden, weight), jax.grad(zf(_naive), (0, 1))(hidden, weight)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_bf16_compute_is_close():
    hidden, weight, targets = _inputs(2)
    got = linear_cross_entropy(hidden, weight, targets, vocab_chunk=256)
    want = _naive(hidden, weight, targets)
    np.testing.assert_allclose(got.loss_sum, want.loss_sum, rtol=2e-2)
    g = jax.grad(_objective(lambda h, w, t: linear_cross_entropy(h, w, t, vocab_chunk=256)), (0, 1))(
        hidden, weight, targets)
    g_want = jax.grad(_objective(_naive), (0, 1))(hidden, weight, targets)
    for a, b in zip(g, g_want):
        assert a.dtype == b.dtype
        np.testing.assert_allclose(a, b, rtol=5e-2, atol=5e-2 * float(jnp.max(jnp.abs(b))))


def test_ignored_rows_are_inert_even_when_not_finite():
    hidden, weight, targets = _inputs(3, ignore_every=2)
    hidden = jnp.where((targets == -100)[:, None], jnp.nan, hidden)
    f = _objective(lambda h, w, t: linear_cross_entropy(h, w, t, vocab_chunk=128, compute_dtype=jnp.float32))
    value, grads = jax.value_and_grad(f, (0, 1))(hidden, weight, targets)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in grads)
    np.testing.assert_array_equal(grads[0][targets == -100], 0.0)
    none_valid = linear_cross_entropy(hidden, weight, jnp.full((N,), -100), vocab_chunk=128)
    assert float(none_valid.loss_sum) == 0.0 and float(none_valid.num_valid) == 0.0


def test_backward_never_builds_full_logits():
    hidden, weight, targets = _inputs(4)
    f = _objective(lambda h, w, t: linear_cross_entropy(h, w, t, vocab_chunk=128, compute_dtype=jnp.float32))
    text = jax.jit(jax.grad(f, (0, 1))).lower(hidden, weight, targets).as_text()
    assert f"tensor<{N}x{V}x" not in text and f"tensor<{N}x1024x" not in text
    assert f"tensor<{N}x128xf32>" in text


def test_input_validation():
    hidden, weight, targets = _inputs()
    with pytest.raises(ValueError, match="hidden"):
        linear_cross_entropy(hidden[None], weight, targets)
    with pytest.raises(ValueError, match="targets"):
        linear_cross_entropy(hidden, weight, targets[:-1])
    with pytest.raises(TypeError):
        linear_cross_entropy(hidden, weight, targets.astype(jnp.float32))
    with pytest.raises(ValueError, match="label_smoothing"):
        linear_cross_entropy(hidden, weight, targets, label_smoothing=1.0)
