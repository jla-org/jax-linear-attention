# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production architecture presets (Qwen3-Next, Nemotron-H, Granite-4.0-H, GPT-OSS, Gemma 3).

Attention sinks are checked against a token-by-token NumPy oracle; the presets must
build, take a finite training step and decode exactly like their forward pass.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.attention import Attention, attention
from jla.layers.mlp import SquaredReLUMLP
from jla.models import build_model, get_preset, logical_axes, parameter_count
from jla.models.presets import NEMOTRON_H_8B_PATTERN, PRODUCTION_PRESETS
from jla.ops.cross_entropy import linear_cross_entropy
from jla.training.performance import estimate_flops_per_token


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


TINY = [name for name in PRODUCTION_PRESETS if name.startswith("tiny")]
REAL = [name for name in PRODUCTION_PRESETS if not name.startswith("tiny")]


def _naive_sink_attention(q, k, v, sinks, *, window=None, segment_ids=None, scale=None):
    """Loop oracle: ``o_t = sum_j e^{s_tj} v_j / (e^{sink_h} + sum_j e^{s_tj})`` over visible ``j``."""
    q, k, v = (np.asarray(a, np.float64) for a in (q, k, v))
    sinks = None if sinks is None else np.asarray(sinks, np.float64)
    batch, length, heads, head_dim = q.shape
    group = heads // k.shape[2]
    scale = head_dim ** -0.5 if scale is None else scale
    out = np.zeros_like(q)
    for b in range(batch):
        for h in range(heads):
            kh = h // group
            for t in range(length):
                js = [j for j in range(t + 1)
                      if (window is None or t - j < window)
                      and (segment_ids is None or segment_ids[b, j] == segment_ids[b, t])]
                weights = np.exp(np.array([q[b, t, h] @ k[b, j, kh] * scale for j in js]))
                denom = weights.sum() + (0.0 if sinks is None else np.exp(sinks[h]))
                out[b, t, h] = (weights[:, None] * v[b, js, kh]).sum(0) / denom
    return out


def _qkv(seed=0, batch=2, length=9, heads=4, kv_heads=2, head_dim=8):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (batch, length, heads, head_dim))
    k = jax.random.normal(ks[1], (batch, length, kv_heads, head_dim))
    v = jax.random.normal(ks[2], (batch, length, kv_heads, head_dim))
    return q, k, v, jax.random.normal(ks[3], (heads,))


@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scale", [None, 0.3])
def test_sink_attention_matches_a_naive_reference(window, packed, scale):
    q, k, v, sinks = _qkv()
    seg = jnp.array([[0] * 4 + [1] * 5, [0] * 9]) if packed else None
    got = attention(q, k, v, segment_ids=seg, sliding_window=window, scale=scale, sinks=sinks)
    want = _naive_sink_attention(q, k, v, sinks, window=window, scale=scale,
                                 segment_ids=None if seg is None else np.asarray(seg))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_vanishing_sinks_recover_plain_attention():
    q, k, v, _ = _qkv(seed=1)
    got = attention(q, k, v, sinks=jnp.full((4,), -1e4), sliding_window=4, implementation="xla")
    np.testing.assert_allclose(got, _naive_sink_attention(q, k, v, None, window=4), rtol=1e-5, atol=1e-5)


def test_sinks_refuse_the_cudnn_backend():
    q, k, v, sinks = _qkv()
    with pytest.raises(ValueError, match="XLA"):
        attention(q, k, v, sinks=sinks, implementation="cudnn")


def test_sink_gradients_match_finite_differences():
    q, k, v, sinks = _qkv(seed=2)

    def f(s):
        return jnp.sum(attention(q, k, v, sinks=s, sliding_window=5) ** 2)

    grad = jax.grad(f)(sinks)
    eps = 1e-3
    for h in range(4):
        e = jnp.zeros(4).at[h].set(eps)
        fd = (f(sinks + e) - f(sinks - e)) / (2 * eps)
        np.testing.assert_allclose(grad[h], fd, rtol=2e-2, atol=1e-4)


def test_squared_relu_mlp_matches_numpy():
    mlp = SquaredReLUMLP(8, 16, key=jax.random.key(0), std=0.5, compute_dtype="float32")
    x = jax.random.normal(jax.random.key(1), (3, 8))
    up, down = np.asarray(mlp.up.weight, np.float64), np.asarray(mlp.down.weight, np.float64)
    want = np.maximum(np.asarray(x, np.float64) @ up, 0.0) ** 2 @ down
    np.testing.assert_allclose(mlp(x), want, rtol=1e-5, atol=1e-5)


def test_published_layer_patterns():
    nem = get_preset("nemotron-h-600m")
    assert len(NEMOTRON_H_8B_PATTERN) == 52
    assert nem.resolved_layer_types.count("mamba2") == 24 and nem.resolved_layer_types.count("attention") == 4
    assert nem.resolved_mlp_types.count("relu2") == 24
    oss = get_preset("gpt-oss-180m")
    assert oss.resolved_layer_types[:2] == ("local_attention", "attention") and oss.local_attention.sliding_window == 128
    assert oss.attention.sinks and oss.local_attention.sinks and oss.moe.top_k == 4
    gem = get_preset("gemma-3-750m").resolved_layer_types
    assert gem[:6] == ("local_attention",) * 5 + ("attention",) and gem.count("attention") == 4
    gra = get_preset("granite-4.0-h-125m").resolved_layer_types
    assert [i for i, t in enumerate(gra) if t == "attention"] == [5]
    qn = get_preset("qwen3-next-300m")
    assert qn.resolved_layer_types[:4] == ("gated_deltanet",) * 3 + ("attention",)
    assert qn.attention.output_gate and qn.attention.rotary_fraction == 0.25


@pytest.mark.parametrize("name", REAL)
def test_real_presets_build_and_have_flops(name):
    cfg = get_preset(name)
    count = parameter_count(cfg)
    flops = estimate_flops_per_token(cfg, seq_len=4096)
    assert count.active <= count.total and flops.forward > 0


def _f32(name: str, **changes):
    cfg = get_preset(name).replace(compute_dtype="float32", **changes)
    if cfg.moe is not None:
        cfg = cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None))
    return cfg


def _sinks_of(model):
    return [b.mixer.sinks for b in model.blocks if isinstance(b.mixer, Attention) and b.mixer.sinks is not None]


def _randomize_sinks(model, key):
    sinks = _sinks_of(model)
    if not sinks:
        return model
    new = [jax.random.normal(jax.random.fold_in(key, i), s.shape, s.dtype) for i, s in enumerate(sinks)]
    return eqx.tree_at(_sinks_of, model, new)


def _loss(model, ids):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    return ce.loss_sum / ce.num_valid + out.aux_loss


@pytest.mark.parametrize("name", TINY)
def test_tiny_presets_take_a_training_step(name):
    cfg = _f32(name)
    model = _randomize_sinks(build_model(cfg, key=jax.random.key(1)), jax.random.key(3))
    logical_axes(model)  # every new parameter (the sinks) declares its axes
    ids = jax.random.randint(jax.random.key(2), (2, 16), 0, cfg.vocab_size)
    loss, grads = eqx.filter_value_and_grad(_loss)(model, ids)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert np.isfinite(float(loss)) and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    for g in _sinks_of(grads):
        assert float(jnp.max(jnp.abs(g))) > 0
    stepped = eqx.apply_updates(model, jax.tree.map(lambda g: -1e-2 * g, eqx.filter(grads, eqx.is_array)))
    assert float(_loss(stepped, ids)) < float(loss)


@pytest.mark.parametrize("name", TINY)
@pytest.mark.parametrize("scan", [True, False])
def test_tiny_presets_decode_like_the_forward_pass(name, scan):
    cfg = _f32(name, scan_layers=scan)
    model = _randomize_sinks(build_model(cfg, key=jax.random.key(5)), jax.random.key(7))
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


def test_deterministic_mode_moves_local_attention_to_xla():
    """``runtime.deterministic`` must route the local layers of a local/global stack to XLA too."""
    from jla.config import ExperimentConfig, RuntimeConfig
    from jla.training.trainer import deterministic_config

    model = get_preset("tiny-gemma-3")
    assert model.local_attention is not None and model.local_attention.implementation != "xla"
    out = deterministic_config(ExperimentConfig(model=model, runtime=RuntimeConfig(deterministic=True)))
    assert out.model.attention.implementation == "xla"
    assert out.model.local_attention.implementation == "xla"
    assert out.model.local_attention.sliding_window == model.local_attention.sliding_window
    # Global attention already on XLA: the local config is still moved.
    pinned = model.replace(attention=dataclasses.replace(model.attention, implementation="xla"))
    out = deterministic_config(ExperimentConfig(model=pinned, runtime=RuntimeConfig(deterministic=True)))
    assert out.model.local_attention.implementation == "xla"
    # Off: unchanged.
    off = ExperimentConfig(model=model)
    assert deterministic_config(off) is off
