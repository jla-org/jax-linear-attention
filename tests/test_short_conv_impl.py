# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``ShortConv(impl=...)`` plumbing on any backend (the kernels themselves: test_mosaic_gpu_short_conv.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.conv import ShortConv, _final_state
from jla.models import build_model, get_preset


def test_unknown_impl_raises():
    with pytest.raises(ValueError, match="impl"):
        ShortConv(128, 4, impl="cuda")


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("segments", [False, True])
@pytest.mark.parametrize("width", [2, 4])
def test_final_state_matches_xla_path(seq_len, segments, width):
    ks = jax.random.split(jax.random.PRNGKey(seq_len * 7 + width), 3)
    x = jax.random.normal(ks[0], (2, seq_len, 128), jnp.float32).astype(jnp.bfloat16)
    state = jax.random.normal(ks[1], (2, width - 1, 128), jnp.float32).astype(jnp.bfloat16)
    seg = jnp.cumsum(jax.random.bernoulli(ks[2], 0.4, (2, seq_len)).astype(jnp.int32), axis=1) if segments else None
    conv = ShortConv(128, width, impl="xla")
    for s in (None, state):
        _, expected = conv(x, segment_ids=seg, state=s)
        np.testing.assert_array_equal(np.asarray(_final_state(x, s, seg, width), np.float32),
                                      np.asarray(expected, np.float32))


def test_auto_falls_back_off_hopper():
    """Off an sm_90 GPU (or with several devices) ``auto`` is the XLA path, bit for bit."""
    from jla.ops.mosaic_gpu.short_conv import unsupported_reason

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 256), jnp.float32).astype(jnp.bfloat16)
    conv = ShortConv(256, 4, use_bias=True, init="uniform", key=jax.random.PRNGKey(1), impl="xla")
    auto = ShortConv(256, 4, use_bias=True, init="uniform", key=jax.random.PRNGKey(1), impl="auto")
    if unsupported_reason(x, conv.weight) is None:
        pytest.skip("the kernels support this call here (GPU test file covers it)")
    np.testing.assert_array_equal(np.asarray(auto(x)[0], np.float32), np.asarray(conv(x)[0], np.float32))


def test_release_configs_default_to_auto():
    from jla.models.config import LinearAttentionConfig, SSMConfig

    assert SSMConfig().conv_impl == "auto"
    assert LinearAttentionConfig(num_k_heads=1, num_v_heads=1, head_k_dim=64, head_v_dim=64).conv_impl == "auto"


@pytest.mark.parametrize("preset,attr", [("mamba2-1p3b", "ssm"), ("kda-1p3b", "linear_attention")])
def test_config_field_reaches_the_convs(preset, attr):
    import dataclasses

    cfg = get_preset(preset)
    cfg = cfg.replace(n_layers=1, layer_types=cfg.layer_types[:1], vocab_size=256,
                      mlp_types=None if cfg.mlp_types is None else cfg.mlp_types[:1],
                      **{attr: dataclasses.replace(getattr(cfg, attr), conv_impl="mosaic_gpu")})
    model = build_model(cfg, key=jax.random.PRNGKey(0))
    convs = [leaf for leaf in jax.tree_util.tree_leaves(model, is_leaf=lambda m: isinstance(m, ShortConv))
             if isinstance(leaf, ShortConv)]
    assert convs and all(c.impl == "mosaic_gpu" for c in convs)
