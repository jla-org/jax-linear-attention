# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Norms, RoPE, linear/embedding layers and SwiGLU against their closed forms."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.linear import Embedding, Linear, LowRankLinear, truncated_normal
from jla.layers.mlp import SwiGLU
from jla.layers.norm import GatedRMSNorm, RMSNorm, l2_normalize, rms_norm
from jla.layers.rope import apply_rope, rotary_dim

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=1e-5, atol=1e-5)


def _x(shape, seed=0, dtype=jnp.float32):
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32).astype(dtype)


def test_rms_norm_matches_formula_and_keeps_dtype():
    x = _x((3, 5, 16))
    weight = jnp.linspace(0.5, 1.5, 16)
    norm = eqx.tree_at(lambda n: n.weight, RMSNorm(16, eps=1e-6), weight)
    want = np.asarray(x) / np.sqrt(np.mean(np.asarray(x) ** 2, -1, keepdims=True) + 1e-6) * np.asarray(weight)
    np.testing.assert_allclose(norm(x), want, **TOL)
    y = RMSNorm(16)(x.astype(jnp.bfloat16))
    assert y.dtype == jnp.bfloat16
    # bf16 input is normalized in float32: matches the f32 result rounded once.
    np.testing.assert_allclose(y.astype(jnp.float32), rms_norm(x.astype(jnp.bfloat16).astype(jnp.float32),
                                                               None, 1e-6), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_gated_rms_norm(activation):
    x, z = _x((2, 7, 8), 1), _x((2, 7, 8), 2)
    out = GatedRMSNorm(8, activation=activation)(x, z)
    gate = jax.nn.silu(z) if activation == "silu" else jax.nn.sigmoid(z)
    np.testing.assert_allclose(out, rms_norm(x, None, 1e-6) * gate, **TOL)
    with pytest.raises(ValueError, match="gate shape"):
        GatedRMSNorm(8)(x, z[..., :4])
    with pytest.raises(ValueError, match="activation"):
        GatedRMSNorm(8, activation="relu")


def test_l2_normalize():
    x = _x((4, 32), 3)
    y = l2_normalize(x)
    np.testing.assert_allclose(jnp.linalg.norm(y, axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(l2_normalize(jnp.zeros((2, 8))), 0.0)


def _hf_rope(x, positions, theta, dim):
    """Hugging Face rotate_half reference (numpy)."""
    x = np.asarray(x, np.float64)
    inv = 1.0 / theta ** (np.arange(0, dim, 2) / dim)
    freqs = np.asarray(positions, np.float64)[..., None] * inv
    emb = np.concatenate([freqs, freqs], -1)[:, :, None, :]
    rot, rest = x[..., :dim], x[..., dim:]
    half = dim // 2
    rotated = np.concatenate([-rot[..., half:], rot[..., :half]], -1)
    return np.concatenate([rot * np.cos(emb) + rotated * np.sin(emb), rest], -1)


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_rope_matches_hugging_face_with_per_sample_positions(fraction):
    x = _x((2, 6, 3, 16), 4)
    positions = jnp.array([[0, 1, 2, 3, 4, 5], [7, 3, 100, 0, 9, 2]], jnp.int32)
    out = apply_rope(x, positions, theta=10_000.0, rotary_fraction=fraction)
    want = _hf_rope(x, positions, 10_000.0, rotary_dim(16, fraction))
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)
    if fraction < 1.0:
        np.testing.assert_array_equal(out[..., 4:], x[..., 4:])


def test_rope_scores_depend_only_on_relative_position():
    q, k = _x((1, 1, 1, 32), 5), _x((1, 1, 1, 32), 6)

    def score(pq, pk):
        rq = apply_rope(q, jnp.array([[pq]]), theta=500.0)
        rk = apply_rope(k, jnp.array([[pk]]), theta=500.0)
        return float(jnp.sum(rq * rk))

    np.testing.assert_allclose(score(10, 3), score(107, 100), rtol=1e-4)
    with pytest.raises(ValueError, match="even"):
        rotary_dim(16, 0.2)


def test_linear_and_embedding_dtypes():
    key = jax.random.key(7)
    layer = Linear(8, 12, key=key, axes=("embed", "mlp"), std=0.02, use_bias=True)
    assert layer.weight.dtype == jnp.float32 and layer.weight.shape == (8, 12)
    y = layer(_x((2, 3, 8), 8))
    assert y.dtype == jnp.bfloat16 and y.shape == (2, 3, 12)
    f32 = Linear(8, 12, key=key, axes=("embed", "mlp"), std=0.02, compute_dtype="float32")
    x = _x((2, 3, 8), 8)
    np.testing.assert_allclose(f32(x), x @ f32.weight, **TOL)
    bf16_params = Linear(8, 12, key=key, axes=("embed", "mlp"), std=0.02, param_dtype=jnp.bfloat16)
    assert bf16_params.weight.dtype == jnp.bfloat16
    low = LowRankLinear(8, 2, 12, key=key, out_axis="heads", std=0.02, use_bias=True, compute_dtype="float32")
    np.testing.assert_allclose(low(x), x @ low.down.weight @ low.up.weight + low.up.bias, **TOL)
    emb = Embedding(50, 8, key=key, std=0.02)
    ids = jnp.array([[1, 2], [49, 0]])
    assert emb(ids).dtype == jnp.bfloat16
    np.testing.assert_allclose(emb(ids).astype(jnp.float32), emb.weight[ids], rtol=1e-2, atol=1e-3)
    with pytest.raises(TypeError):
        emb(ids.astype(jnp.float32))


def test_truncated_normal_init_scale():
    w = truncated_normal(jax.random.key(9), (256, 256), 0.02)
    assert float(jnp.max(jnp.abs(w))) <= 0.04 + 1e-6
    # Truncation at 2 sigma shrinks the std by ~12%.
    assert 0.015 < float(jnp.std(w)) < 0.02


def test_swiglu_formula():
    mlp = SwiGLU(8, 16, key=jax.random.key(10), std=0.1, compute_dtype="float32")
    x = _x((3, 8), 11)
    gate = x @ mlp.gate_up[:, 0]
    up = x @ mlp.gate_up[:, 1]
    np.testing.assert_allclose(mlp(x), (jax.nn.silu(gate) * up) @ mlp.down.weight, **TOL)
