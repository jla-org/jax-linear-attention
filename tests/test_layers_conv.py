# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Short causal convolution: formula, causality, decode-state parity and document isolation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.conv import ShortConv


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=1e-5, atol=1e-5)


def _conv(channels=6, width=4, activation=True, seed=0):
    conv = ShortConv(channels, width, activation=activation)
    weight = jax.random.normal(jax.random.key(seed), (channels, width))
    return eqx.tree_at(lambda c: c.weight, conv, weight)


def _reference(x, weight, activation):
    x = np.asarray(x, np.float64)
    w = np.asarray(weight, np.float64)
    batch, length, channels = x.shape
    width = w.shape[1]
    y = np.zeros_like(x)
    for t in range(length):
        for i in range(width):
            src = t - (width - 1 - i)
            if src >= 0:
                y[:, t] += w[:, i] * x[:, src]
    return y / (1 + np.exp(-y)) if activation else y  # silu


def _x(shape, seed=1):
    return jax.random.normal(jax.random.key(seed), shape)


def test_identity_init_passes_tokens_through():
    x = _x((2, 9, 6))
    y, _ = ShortConv(6, 4, activation=False)(x)
    np.testing.assert_allclose(y, x, **TOL)
    y, _ = ShortConv(6, 4)(x)
    np.testing.assert_allclose(y, jax.nn.silu(x), **TOL)


def test_matches_reference_and_is_causal():
    conv = _conv()
    x = _x((2, 11, 6))
    y, _ = conv(x)
    np.testing.assert_allclose(y, _reference(x, conv.weight, True), **TOL)
    x2 = x.at[:, 7:].set(_x((2, 4, 6), 5))
    np.testing.assert_allclose(conv(x2)[0][:, :7], y[:, :7], **TOL)


def test_state_makes_prefill_and_decode_match_the_full_pass():
    conv = _conv(activation=False)
    x = _x((3, 10, 6))
    full, full_state = conv(x)
    first, state = conv(x[:, :6])
    outs = [first]
    for t in range(6, 10):
        y, state = conv(x[:, t : t + 1], state=state)
        outs.append(y)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)
    np.testing.assert_allclose(state, full_state, **TOL)
    np.testing.assert_allclose(full_state, x[:, -3:], **TOL)


def test_segments_never_mix_and_state_drops_old_documents():
    conv = _conv()
    lengths = [3, 1, 5]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    x = _x((1, 9, 6), 2)
    packed, state = conv(x, segment_ids=seg)
    start = 0
    for n in lengths:
        alone, _ = conv(x[:, start : start + n])
        np.testing.assert_allclose(packed[:, start : start + n], alone, **TOL)
        start += n
    # The last document has 5 tokens, so all three state entries belong to it.
    np.testing.assert_allclose(state, x[:, -3:], **TOL)
    short_last = jnp.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 1]], jnp.int32)
    _, state = conv(x, segment_ids=short_last)
    np.testing.assert_allclose(state[:, 0], 0.0)
    np.testing.assert_allclose(state[:, 1:], x[:, -2:], **TOL)
