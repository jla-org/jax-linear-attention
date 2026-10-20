# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaProduct mixer: FLA layout, n_h = 1 is Gated DeltaNet, documents, decode parity, model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.linear_attention import GatedDeltaProduct, make_mixer
from jla.models import get_preset
from jla.models.config import LinearAttentionConfig


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


D = 32
TOL = dict(rtol=2e-4, atol=2e-4)


def _mixer(kind, seed=0, **kw):
    base = dict(num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, chunk_size=16, allow_neg_eigval=True)
    base.update(kw)
    mixer = make_mixer(kind, D, LinearAttentionConfig(**base), key=jax.random.key(seed), std=0.2, norm_eps=1e-6,
                       compute_dtype="float32")
    keep = {n: getattr(mixer, n) for n in ("A_log", "dt_bias")}
    leaves, treedef = jax.tree.flatten(mixer)
    keys = jax.random.split(jax.random.key(seed + 100), len(leaves))
    leaves = [0.3 * jax.random.normal(k, x.shape, x.dtype) if eqx.is_inexact_array(x) else x
              for k, x in zip(keys, leaves)]
    mixer = jax.tree.unflatten(treedef, leaves)
    for name, value in keep.items():
        mixer = eqx.tree_at(lambda m, name=name: getattr(m, name), mixer, value)
    return mixer


def _x(batch=2, length=20, seed=1):
    return jax.random.normal(jax.random.key(seed), (batch, length, D))


def test_projection_widths():
    mixer = _mixer("gated_deltaproduct", num_householder=3)
    assert isinstance(mixer, GatedDeltaProduct)
    assert mixer.q_proj.weight.size == D * 16
    assert mixer.k_proj.weight.size == D * 16 * 3
    assert mixer.v_proj.weight.size == D * 32 * 3
    assert mixer.b_proj.weight.size == D * 4 * 3
    assert mixer.a_proj.weight.size == D * 4


@pytest.mark.parametrize("conv_size", [4, 0])
def test_one_householder_equals_gated_deltanet(conv_size):
    gdn = _mixer("gated_deltanet", conv_size=conv_size)
    dp = _mixer("gated_deltaproduct", conv_size=conv_size, num_householder=1)
    dp = jax.tree.unflatten(jax.tree.structure(dp), jax.tree.leaves(gdn))  # same parameters
    x = _x()
    seg = jnp.asarray([[0] * 7 + [1] * 13, [0] * 20], jnp.int32)
    np.testing.assert_array_equal(np.asarray(dp(x)), np.asarray(gdn(x)))
    np.testing.assert_array_equal(np.asarray(dp(x, seg)), np.asarray(gdn(x, seg)))


@pytest.mark.parametrize("n", [2, 3])
def test_chunked_matches_reference(n):
    x = _x()
    ref = _mixer("gated_deltaproduct", num_householder=n, implementation="reference")
    chunked = _mixer("gated_deltaproduct", num_householder=n, implementation="chunked")
    np.testing.assert_allclose(chunked(x), ref(x), **TOL)


@pytest.mark.parametrize("conv_size", [4, 0])
def test_documents_are_independent(conv_size):
    mixer = _mixer("gated_deltaproduct", num_householder=2, conv_size=conv_size)
    x = _x(1, 24)
    seg = jnp.asarray([[0] * 9 + [1] * 15], jnp.int32)
    packed = mixer(x, seg)
    np.testing.assert_allclose(packed[:, :9], mixer(x[:, :9]), **TOL)
    np.testing.assert_allclose(packed[:, 9:], mixer(x[:, 9:]), **TOL)


@pytest.mark.parametrize("conv_size", [4, 0])
def test_prefill_then_decode_matches_the_full_pass(conv_size):
    mixer = _mixer("gated_deltaproduct", num_householder=2, conv_size=conv_size, implementation="chunked")
    x = _x(2, 14, seed=3)
    full = mixer(x)
    cache = mixer.init_cache(2, jnp.float32)
    out, cache = mixer.step(x[:, :5], cache)
    outs = [out]
    for t in range(5, 14):
        out, cache = mixer.step(x[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)


def test_tiny_preset_trains_one_step():
    from jla.models import LanguageModel

    config = get_preset("tiny-deltaproduct")
    assert config.linear_attention.num_householder == 2
    model = LanguageModel(config, key=jax.random.key(0))
    tokens = jax.random.randint(jax.random.key(1), (2, 16), 0, config.vocab_size)

    def loss(m):
        hidden = m(tokens, segment_ids=jnp.asarray([[0] * 6 + [1] * 10] * 2, jnp.int32)).hidden
        return jnp.mean(hidden.astype(jnp.float32) ** 2)

    value, grads = eqx.filter_value_and_grad(loss)(model)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
