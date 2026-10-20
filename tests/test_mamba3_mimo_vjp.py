# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO: the custom VJP of the per-head C/B expansion + bias + halves rotation (``_expand_rotate``).

The oracle is plain JAX autodiff of the expression the layer computed before the custom VJP (repeat to the heads in
float32, add the bias, ``rotate_halves``, cast), on small float32 shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jla.layers.mamba3_mimo as mimo
from jla.layers.mamba3_mimo import mamba3_mimo, rotate_halves


def _plain(x, bias, angle, dtype):
    """The pre-custom-VJP expression, differentiated by JAX."""
    heads, groups = bias.shape[0], x.shape[3]
    x32 = jnp.repeat(x.astype(jnp.float32), heads // groups, axis=3)
    return rotate_halves(jnp.moveaxis(x32, 2, 3) + bias.astype(jnp.float32), angle[:, :, :, None]).astype(dtype)


def _inputs(seed, groups, num, batch=2, length=19, rank=2, heads=4, n=8):
    ks = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(ks[0], (batch, length, rank, groups, n))
    bias = 1.0 + 0.1 * jax.random.normal(ks[1], (heads, rank, n))
    angle = jax.random.uniform(ks[2], (batch, length, heads, num), maxval=2 * np.pi)
    ct = jax.random.normal(ks[3], (batch, length, heads, rank, n))
    return x, bias, angle, ct


def _assert_close(got, want, rtol):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * np.max(np.abs(want)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(("groups", "num"), [(1, 2), (2, 4), (4, 1)])
def test_forward_is_the_plain_expression_bit_for_bit(dtype, groups, num):
    x, bias, angle, _ = _inputs(0, groups, num)
    want = _plain(x, bias, angle, dtype)
    got = mimo._expand_rotate(x, bias, angle, dtype)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    primal, _ = jax.vjp(lambda *a: mimo._expand_rotate(*a, dtype), x, bias, angle)  # the custom VJP's forward
    np.testing.assert_array_equal(np.asarray(primal, np.float32), np.asarray(want, np.float32))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])  # the output (and so the cotangent) dtype
@pytest.mark.parametrize(("groups", "num"), [(1, 2), (2, 4), (4, 1)])
def test_gradients_equal_plain_autodiff(dtype, groups, num):
    x, bias, angle, ct = _inputs(1, groups, num)
    ct = ct.astype(dtype)
    _, plain_vjp = jax.vjp(lambda *a: _plain(*a, dtype), x, bias, angle)
    _, custom_vjp = jax.vjp(lambda *a: mimo._expand_rotate(*a, dtype), x, bias, angle)
    want, got = plain_vjp(ct), custom_vjp(ct)
    for name, g, w in zip(("x", "bias", "angle"), got, want):
        assert g.shape == w.shape and g.dtype == w.dtype, name
    _assert_close(got[0], want[0], 1e-6)  # C/B: the per-head sum of the inverse-rotated cotangent
    _assert_close(got[1], want[1], 1e-6)  # bias: its sum over tokens
    _assert_close(got[2], want[2], 1e-6)  # angles


def test_core_gradients_equal_plain_autodiff(monkeypatch):
    batch, length, rank, groups, heads, n, p, num = 2, 20, 2, 1, 2, 8, 4, 2
    ks = jax.random.split(jax.random.key(2), 12)
    q = jax.random.normal(ks[0], (batch, length, rank, groups, n))
    k = jax.random.normal(ks[1], (batch, length, rank, groups, n))
    v = jax.random.normal(ks[2], (batch, length, heads, p))
    dt = jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads)))
    adt = -dt * (1.0 + jax.random.uniform(ks[4], (batch, length, heads)))
    trap = jax.random.normal(ks[5], (batch, length, heads))
    q_bias, k_bias = (1.0 + 0.1 * jax.random.normal(ks[i], (heads, rank, n)) for i in (6, 7))
    angles = jax.random.normal(ks[8], (batch, length, heads, num))
    mimo_x, mimo_z, mimo_o = jax.random.normal(ks[9], (3, heads, rank, p)) * 0.5 + 1.0
    z = jax.random.normal(ks[10], (batch, length, heads, p))
    w = jax.random.normal(ks[11], (batch, length, heads, p))
    args = (q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_x)

    def loss(*a):
        y, _ = mamba3_mimo(*a, mimo_z=mimo_z, mimo_o=mimo_o, d=jnp.array([0.5, -1.0]), z=z, impl="chunked",
                           chunk_size=8)
        return jnp.sum(y * w)

    argnums = tuple(range(len(args)))
    got_loss, got = jax.value_and_grad(loss, argnums=argnums)(*args)
    monkeypatch.setattr(mimo, "_expand_rotate", _plain)
    want_loss, want = jax.value_and_grad(loss, argnums=argnums)(*args)
    assert float(got_loss) == float(want_loss)
    names = ("q", "k", "v", "adt", "dt", "trap", "q_bias", "k_bias", "angles", "mimo_x")
    for name, g, wnt in zip(names, got, want):
        assert g.shape == wnt.shape and g.dtype == wnt.dtype, name
        _assert_close(g, wnt, 1e-6)
