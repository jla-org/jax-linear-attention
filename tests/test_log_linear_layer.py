# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Log-linear Mamba-2 mixer (``SSMConfig.log_linear_levels``) and its presets."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.ssm import log_linear_mamba2_ssd, mamba2_ssd
from jla.models import build_model, get_preset
from jla.ops.log_linear import num_levels


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _ssd_inputs(batch=2, length=24, heads=4, head_dim=8, groups=2, d_state=6):
    ks = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(ks[0], (batch, length, heads, head_dim))
    dt = jax.nn.softplus(jax.random.normal(ks[1], (batch, length, heads)) - 1.0)
    a = -jnp.exp(jax.random.normal(ks[2], (heads,)))
    b = jax.random.normal(ks[3], (batch, length, groups, d_state))
    c = jax.random.normal(ks[4], (batch, length, groups, d_state))
    seg = jnp.asarray(np.repeat(np.array([[0] * 10 + [1] * (length - 10)]), batch, axis=0).astype(np.int32))
    return x, dt, a, b, c, seg


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_constant_level_scales_scale_the_ssd(impl):
    """Identity: level_scales == c everywhere gives c * Mamba-2's SSD core (no D skip)."""
    x, dt, a, b, c, seg = _ssd_inputs()
    scales = jnp.full(x.shape[:3] + (num_levels(x.shape[1]) + 2,), 0.7)
    y_log, _ = log_linear_mamba2_ssd(x, dt, a, b, c, level_scales=scales, segment_ids=seg, impl=impl, chunk_size=8)
    y_lin, _ = mamba2_ssd(x, dt, a, b, c, segment_ids=seg, impl="reference")
    np.testing.assert_allclose(np.asarray(y_log), 0.7 * np.asarray(y_lin), rtol=1e-4, atol=1e-4)


def test_tiny_preset_chunked_matches_reference():
    cfg = get_preset("tiny-loglinear-mamba2").replace(compute_dtype="float32")
    assert cfg.ssm.log_linear_levels == 10
    ids = jax.random.randint(jax.random.key(1), (2, 40), 0, cfg.vocab_size)
    seg = jnp.asarray(np.array([[0] * 17 + [1] * 23, [3] * 40], np.int32))
    out = build_model(cfg, key=jax.random.key(0))(ids, segment_ids=seg).hidden
    ref_cfg = cfg.replace(ssm=dataclasses.replace(cfg.ssm, implementation="reference"))
    ref = build_model(ref_cfg, key=jax.random.key(0))(ids, segment_ids=seg).hidden
    assert np.all(np.isfinite(np.asarray(out)))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_tiny_preset_trains_and_rejects_decoding():
    cfg = get_preset("tiny-loglinear-mamba2").replace(compute_dtype="float32")
    model = build_model(cfg, key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(2), (2, 32), 0, cfg.vocab_size)

    def loss(m):
        return jnp.mean(m(ids).hidden ** 2)

    grads = jax.grad(loss)(model)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if hasattr(leaf, "dtype")]
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    plain = build_model(get_preset("tiny-mamba2").replace(compute_dtype="float32"), key=jax.random.key(0))
    count = lambda m: sum(x.size for x in jax.tree_util.tree_leaves(eqx_arrays(m)))  # noqa: E731
    heads = cfg.ssm.num_heads(cfg.d_model)
    extra = cfg.n_layers * (cfg.d_model * heads * 10 + heads * 10)
    assert count(model) - count(plain) == extra


def eqx_arrays(module):
    import equinox as eqx

    return eqx.filter(module, eqx.is_array)


def test_levels_must_cover_the_sequence():
    cfg = get_preset("tiny-loglinear-mamba2").replace(compute_dtype="float32")
    cfg = cfg.replace(ssm=dataclasses.replace(cfg.ssm, log_linear_levels=3))
    model = build_model(cfg, key=jax.random.key(0))
    with pytest.raises(ValueError, match="levels"):
        model(jnp.zeros((1, 16), jnp.int32))
