# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plumbing of the fused short conv + L2 norm (any backend): the switch, the fallback and the op's flag."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from jla.layers.conv import ShortConv
from jla.layers.conv_l2norm import conv_l2norm, conv_l2norm_reason
from jla.models.config import LinearAttentionConfig
from jla.ops.mosaic_gpu._common import is_hopper


def test_default_on():
    assert LinearAttentionConfig(num_k_heads=1, num_v_heads=1, head_k_dim=64, head_v_dim=64).qk_l2norm_in_conv


def test_xla_conv_never_fuses():
    conv = ShortConv(512, 4, impl="xla")
    x = jnp.zeros((1, 8, 512), jnp.bfloat16)
    assert conv_l2norm_reason(conv, x, 128) is not None
    assert conv_l2norm(conv, x, 128) is None


@pytest.mark.skipif(is_hopper(), reason="checks the fallback off Hopper")
def test_falls_back_off_hopper():
    conv = ShortConv(512, 4, impl="auto")
    assert conv_l2norm(conv, jnp.zeros((1, 8, 512), jnp.bfloat16), 128) is None


@pytest.mark.parametrize("in_conv", [False, True])
def test_mixer_flag_reaches_the_op(in_conv, monkeypatch):
    """The op normalizes q and k exactly when the conv kernel did not (no second normalisation)."""
    from jla import ops
    from jla.layers import linear_attention as la_mod

    seen = []
    real = ops.linear_attention

    def spy(kind, q, k, v, **options):
        seen.append(options.get("use_qk_l2norm"))
        return real(kind, q, k, v, **options)

    monkeypatch.setattr(la_mod.ops, "linear_attention", spy)
    cfg = LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=128, head_v_dim=128,
                                implementation="chunked", qk_l2norm_in_conv=in_conv)
    mixer = la_mod.make_mixer("gated_deltanet", 256, cfg, key=jax.random.PRNGKey(0), std=0.02, norm_eps=1e-6)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 64, 256), jnp.float32).astype(jnp.bfloat16)
    y = mixer(x)
    assert y.shape == x.shape
    fused = in_conv and conv_l2norm_reason(mixer.q_conv, jnp.zeros((2, 64, 512), jnp.bfloat16), 128) is None
    assert seen == [not fused]
    off = dataclasses.replace(cfg, qk_l2norm=False)
    seen.clear()
    la_mod.make_mixer("gated_deltanet", 256, off, key=jax.random.PRNGKey(0), std=0.02, norm_eps=1e-6)(x)
    assert seen == [False]


def test_without_mosaic_gpu_the_defaults_fall_back(monkeypatch):
    """A JAX build without Mosaic GPU: the fused conv + L2 norm and the conv kernel decline before importing their
    kernel modules (qk_l2norm_in_conv and conv_impl='auto' are defaults), and impl='mosaic_gpu' fails loudly."""
    from jla.ops import mosaic_gpu

    monkeypatch.setattr(mosaic_gpu, "_HAVE_MOSAIC_GPU", False)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 512), jnp.float32).astype(jnp.bfloat16)
    auto = ShortConv(512, 4, impl="auto", init="uniform", key=jax.random.PRNGKey(1))
    assert "no Mosaic GPU" in conv_l2norm_reason(auto, x, 128)
    assert conv_l2norm(auto, x, 128) is None
    xla = ShortConv(512, 4, impl="xla", init="uniform", key=jax.random.PRNGKey(1))
    for got, want in zip(auto(x), xla(x)):
        assert bool(jnp.array_equal(got, want))
    with pytest.raises(ValueError, match="no Mosaic GPU"):
        ShortConv(512, 4, impl="mosaic_gpu")(x)
