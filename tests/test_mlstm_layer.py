# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mLSTM layer and presets: chunked = reference inside the layer, packed documents, prefill + decode."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.linear_attention import MLSTM, make_mixer
from jla.models import build_model, get_preset, list_presets, parameter_count


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _mixer(impl: str, **changes):
    cfg = get_preset("tiny-mlstm")
    la = dataclasses.replace(cfg.linear_attention, implementation=impl, **changes)
    return make_mixer("mlstm", cfg.d_model, la, key=jax.random.key(0), std=0.2, norm_eps=1e-6,
                      param_dtype=jnp.float32, compute_dtype="float32")


def test_layer_structure_and_init():
    layer = _mixer("chunked")
    assert isinstance(layer, MLSTM)
    np.testing.assert_allclose(layer.i_proj.bias, -10.0)
    np.testing.assert_allclose(layer.f_proj.bias, np.linspace(3.0, 6.0, layer.cfg.num_v_heads))
    assert layer.q_conv is None and layer.o_norm.activation == "sigmoid"
    with pytest.raises(ValueError, match="output_gate"):
        _mixer("chunked", output_gate=False)


@pytest.mark.parametrize("soft_cap", [15.0, None])
def test_chunked_matches_reference_in_the_layer(soft_cap):
    x = jax.random.normal(jax.random.key(1), (2, 37, 64))
    seg = jnp.repeat(jnp.array([[0] * 20 + [1] * 17, [0] * 37]), 1, axis=0)
    want = _mixer("reference", gate_soft_cap=soft_cap)(x, segment_ids=seg)
    got = _mixer("chunked", gate_soft_cap=soft_cap)(x, segment_ids=seg)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    # Packed documents: the second document alone gives the same outputs.
    alone = _mixer("reference", gate_soft_cap=soft_cap)(x[:1, 20:])
    np.testing.assert_allclose(want[:1, 20:], alone, rtol=1e-4, atol=1e-5)


def test_layer_prefill_then_decode():
    layer = _mixer("chunked")
    x = jax.random.normal(jax.random.key(2), (2, 21, 64))
    full = layer(x)
    cache = layer.init_cache(2, jnp.float32)
    y, cache = layer.step(x[:, :9], cache)
    outs = [y]
    for t in range(9, 21):
        y, cache = layer.step(x[:, t : t + 1], cache)
        outs.append(y)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("scan", [True, False])
def test_lm_prefill_then_decode_matches_the_forward_pass(scan):
    cfg = get_preset("tiny-mlstm").replace(compute_dtype="float32", scan_layers=scan)
    model = build_model(cfg, key=jax.random.key(5))
    ids = jax.random.randint(jax.random.key(6), (2, 14), 0, cfg.vocab_size)
    full = model.logits(model(ids).hidden)
    pos = jnp.broadcast_to(jnp.arange(14, dtype=jnp.int32), (2, 14))
    step = jax.jit(lambda m, c, i, p: m.decode_step(c, i, p))
    cache = model.init_cache(2, 20)
    logits, cache = step(model, cache, ids[:, :6], pos[:, :6])
    outs = [logits]
    for t in range(6, 14):
        logits, cache = step(model, cache, ids[:, t : t + 1], pos[:, t : t + 1])
        outs.append(logits)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=2e-4)


def test_presets():
    assert {"tiny-mlstm", "xlstm-125m"} <= set(list_presets())
    assert 115e6 < parameter_count(get_preset("xlstm-125m")).total < 135e6
    tiny = build_model(get_preset("tiny-mlstm"), key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (2, 16), 0, 512)
    assert bool(jnp.all(jnp.isfinite(tiny.logits(tiny(ids).hidden))))
