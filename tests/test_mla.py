# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-head latent attention: an independent float64 NumPy oracle, the absorbed decode
path, gradients, the MLA presets (one CPU train step each) and the FLOP model."""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.layers.mla import MLA
from jla.models import build_model, get_preset, parameter_count
from jla.models.config import MLAConfig
from jla.training.performance import estimate_flops_per_token
from jla.training.step import LossConfig, build_train_step, lm_loss

MLA_PRESETS = ["tiny-mla", "deepseek-v3-tiny", "kimi-linear-tiny"]


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


# Independent oracle ----------------------------------------------------------------------------
def _w(linear) -> np.ndarray:
    return np.asarray(linear.weight, np.float64)


def _rms(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * np.asarray(weight, np.float64)


def _rope(x, positions, theta):
    """Half-rotation RoPE of ``x [B, T, H, D]``: ``(x1 cos - x2 sin, x2 cos + x1 sin)``."""
    dim = x.shape[-1]
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = np.asarray(positions, np.float64)[..., None] * inv
    cos, sin = np.cos(ang)[:, :, None], np.sin(ang)[:, :, None]
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def naive_mla(m: MLA, x, positions, segment_ids=None):
    """DeepSeek-V3's naive MLA (per-head keys and values) in float64, one query at a time."""
    cfg = m.cfg
    x = np.asarray(x, np.float64)
    batch, length, _ = x.shape
    heads, nope, rope, dv, rank = (cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim,
                                   cfg.kv_lora_rank)
    eps = m.kv_a_norm.eps
    if cfg.q_lora_rank:
        q = _rms(x @ _w(m.q_a_proj), m.q_a_norm.weight, eps) @ _w(m.q_b_proj)
    else:
        q = x @ _w(m.q_proj)
    q = q.reshape(batch, length, heads, nope + rope)
    q_nope, q_pe = q[..., :nope], q[..., nope:]
    kv = x @ _w(m.kv_a_proj)
    c, k_pe = _rms(kv[..., :rank], m.kv_a_norm.weight, eps), kv[..., rank:]
    if cfg.use_rope:
        q_pe = _rope(q_pe, positions, cfg.rope_theta)
        k_pe = _rope(k_pe[:, :, None], positions, cfg.rope_theta)[:, :, 0]
    kvb = (c @ _w(m.kv_b_proj)).reshape(batch, length, heads, nope + dv)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    scale = (nope + rope) ** -0.5
    out = np.zeros((batch, length, heads, dv))
    for b in range(batch):
        for t in range(length):
            visible = np.arange(length) <= t
            if segment_ids is not None:
                visible &= np.asarray(segment_ids)[b] == np.asarray(segment_ids)[b, t]
            if cfg.sliding_window is not None:
                visible &= t - np.arange(length) < cfg.sliding_window
            s = (np.einsum("hd,shd->hs", q_nope[b, t], k_nope[b]) + np.einsum("he,se->hs", q_pe[b, t], k_pe[b]))
            s = np.where(visible[None], s * scale, -np.inf)
            p = np.exp(s - s.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            out[b, t] = np.einsum("hs,shd->hd", p, v[b])
    return out.reshape(batch, length, heads * dv) @ _w(m.o_proj)


# Layer ----------------------------------------------------------------------------------------
BASE = MLAConfig(num_heads=4, q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
                 v_head_dim=12, implementation="xla")
VARIANTS = {
    "lora_q": BASE,
    "full_q_nope": dataclasses.replace(BASE, q_lora_rank=None, use_rope=False),
    "v_wider": dataclasses.replace(BASE, v_head_dim=40),
    "window": dataclasses.replace(BASE, sliding_window=4),
}


def _layer(cfg: MLAConfig, d_model=32, seed=0) -> MLA:
    m = MLA(d_model, cfg, key=jax.random.key(seed), std=0.2, norm_eps=1e-6, compute_dtype="float32")
    # Non-trivial norm weights so the oracle checks where they apply.
    m = eqx.tree_at(lambda l: l.kv_a_norm.weight, m, 1.0 + 0.1 * jnp.arange(cfg.kv_lora_rank, dtype=jnp.float32))
    return m


def _x(batch=2, length=12, d_model=32, seed=1):
    return jax.random.normal(jax.random.key(seed), (batch, length, d_model), jnp.float32)


def _positions(batch, length):
    return jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_full_forward_matches_the_naive_reference(variant):
    m = _layer(VARIANTS[variant])
    x, pos = _x(), _positions(2, 12)
    out = jax.jit(lambda m, x, p: m(x, p))(m, x, pos)
    assert out.shape == (2, 12, 32) and out.dtype == jnp.float32
    np.testing.assert_allclose(out, naive_mla(m, x, pos), rtol=1e-4, atol=1e-5)


def test_packed_documents_match_the_naive_reference():
    m = _layer(BASE)
    seg = jnp.asarray([[0] * 5 + [1] * 4 + [2] * 3, [0] * 12], jnp.int32)
    pos = jnp.asarray([list(range(5)) + list(range(4)) + list(range(3)), list(range(12))], jnp.int32)
    x = _x()
    out = m(x, pos, seg)
    np.testing.assert_allclose(out, naive_mla(m, x, pos, seg), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("max_len", [20, 6])
def test_absorbed_decode_matches_the_full_forward(variant, max_len):
    cfg = VARIANTS[variant]
    if max_len < 12 and cfg.sliding_window is None:
        pytest.skip("a dense cache shorter than the sequence drops positions")
    m = _layer(cfg)
    x, pos = _x(), _positions(2, 12)
    reference = naive_mla(m, x, pos)
    cache = m.init_cache(2, max_len, jnp.float32)
    assert cache.keys.shape[-1] == cfg.kv_lora_rank and cache.values.shape[-1] == cfg.qk_rope_head_dim
    assert cache.ring == (cfg.sliding_window is not None and cfg.sliding_window < max_len)
    step = jax.jit(lambda m, c, x, p: m.step(x, p, c))
    out, cache = step(m, cache, x[:, :5], pos[:, :5])
    outs = [out]
    for t in range(5, 12):
        out, cache = step(m, cache, x[:, t : t + 1], pos[:, t : t + 1])
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), reference, rtol=1e-4, atol=1e-5)


def test_gradients_are_finite_and_reach_every_weight():
    m = _layer(BASE)
    x, pos = _x(), _positions(2, 12)
    grads = eqx.filter_grad(lambda m: jnp.sum(m(x, pos) ** 2))(m)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert all(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_softmax_scale_and_yarn_mscale():
    assert BASE.scale == pytest.approx(24 ** -0.5)
    yarn = MLAConfig(num_heads=2, yarn_factor=40.0, yarn_mscale=1.0)
    assert yarn.scale == pytest.approx(192 ** -0.5 * (0.1 * math.log(40.0) + 1.0) ** 2)
    assert dataclasses.replace(yarn, softmax_scale=0.5).scale == 0.5
    with pytest.raises(ValueError):
        MLAConfig(num_heads=2, qk_rope_head_dim=7)
    with pytest.raises(ValueError):
        MLAConfig(num_heads=2, q_lora_rank=0)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
@pytest.mark.parametrize("v_head_dim", [128, 192])
def test_cudnn_matches_xla_at_deepseek_head_dims(v_head_dim):
    cfg = MLAConfig(num_heads=4, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=128, qk_rope_head_dim=64,
                    v_head_dim=v_head_dim, implementation="cudnn")
    m = MLA(128, cfg, key=jax.random.key(0), std=0.05, norm_eps=1e-6, compute_dtype="bfloat16")
    x, pos = _x(2, 64, 128).astype(jnp.bfloat16), _positions(2, 64)
    seg = jnp.asarray(np.repeat([0, 1], 32)[None].repeat(2, 0), jnp.int32)
    xla = MLA(128, dataclasses.replace(cfg, implementation="xla"), key=jax.random.key(0), std=0.05, norm_eps=1e-6,
              compute_dtype="bfloat16")  # same key: same weights
    for s in (None, seg):
        a = jax.jit(lambda m, x: m(x, pos, s))(m, x).astype(jnp.float32)
        b = jax.jit(lambda m, x: m(x, pos, s))(xla, x).astype(jnp.float32)
        np.testing.assert_allclose(a, b, rtol=3e-2, atol=3e-2)
    lowered = jax.jit(lambda m, x: m(x, pos)).lower(m, x).as_text()
    assert "cudnn" in lowered.lower() or "custom_call" in lowered


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
@pytest.mark.parametrize("window", [None, 24])
def test_cudnn_takes_the_value_head_dim_unpadded(window):
    """cuDNN's MLA mode (q/k 192, v 128): no zero-padded values; output and every gradient match XLA."""
    from jla.layers.mla import _cudnn_takes_value_head_dim

    probe = jnp.zeros((2, 64, 4, 192), jnp.bfloat16)
    if not _cudnn_takes_value_head_dim(probe, probe[..., :128]):
        pytest.skip("cuDNN < 9.10 or pre-Hopper GPU: the padded path")
    cfg = MLAConfig(num_heads=4, q_lora_rank=64, kv_lora_rank=64, qk_nope_head_dim=128, qk_rope_head_dim=64,
                    v_head_dim=128, implementation="cudnn", sliding_window=window)
    kw = dict(key=jax.random.key(1), std=0.05, norm_eps=1e-6, compute_dtype="bfloat16")
    m, xla = MLA(128, cfg, **kw), MLA(128, dataclasses.replace(cfg, implementation="xla"), **kw)
    x, pos = _x(2, 64, 128).astype(jnp.bfloat16), _positions(2, 64)
    seg = jnp.asarray(np.repeat([0, 1], 32)[None].repeat(2, 0), jnp.int32)
    for s in (None, seg):
        lowered = jax.jit(lambda m, x: m(x, pos, s)).lower(m, x).as_text()
        assert "stablehlo.pad" not in lowered and "custom_call" in lowered

        def loss(m, x):
            return jnp.sum(m(x, pos, s).astype(jnp.float32) ** 2)

        (la, ga), (lb, gb) = (eqx.filter_jit(eqx.filter_value_and_grad(loss))(mod, x) for mod in (m, xla))
        np.testing.assert_allclose(float(la), float(lb), rtol=3e-2)
        leaves_a = jax.tree.leaves(eqx.filter(ga, eqx.is_array))
        leaves_b = jax.tree.leaves(eqx.filter(gb, eqx.is_array))
        assert len(leaves_a) == len(leaves_b) > 0
        for a, b in zip(leaves_a, leaves_b):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            scale = max(float(np.abs(b).max()), 1e-6)
            assert float(np.abs(a - b).max()) <= 5e-2 * scale


# Presets ----------------------------------------------------------------------------------------
def _ids(batch=2, length=14, seed=6, vocab=512):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, vocab)


@pytest.mark.parametrize("name", MLA_PRESETS)
@pytest.mark.parametrize("scan", [True, False])
def test_preset_prefill_then_decode_matches_the_forward_pass(name, scan):
    model = build_model(get_preset(name).replace(compute_dtype="float32", scan_layers=scan), key=jax.random.key(5))
    ids = _ids()
    full = model.logits(model(ids).hidden)
    pos = _positions(2, 14)
    step = jax.jit(lambda m, c, i, p: m.decode_step(c, i, p))
    cache = model.init_cache(2, 20)
    logits, cache = step(model, cache, ids[:, :6], pos[:, :6])
    outs = [logits]
    for t in range(6, 14):
        logits, cache = step(model, cache, ids[:, t : t + 1], pos[:, t : t + 1])
        outs.append(logits)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("name", MLA_PRESETS)
def test_preset_packed_documents_match_separate_runs(name):
    model = build_model(get_preset(name).replace(compute_dtype="float32"), key=jax.random.key(0))
    lengths = [7, 3, 6]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = _ids(1, 16, seed=2)
    packed = model(ids, positions=pos, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        np.testing.assert_allclose(packed[:, start : start + n], model(ids[:, start : start + n]).hidden,
                                   rtol=1e-4, atol=1e-4)
        start += n


@pytest.mark.parametrize("name", MLA_PRESETS)
def test_preset_runs_a_train_step(name):
    config = get_preset(name)
    model = build_model(config, key=jax.random.key(0))
    optimizer = optax.adamw(1e-3)
    step = build_train_step(optimizer, lm_loss(LossConfig()))
    tokens = np.random.default_rng(3).integers(0, config.vocab_size, size=(1, 2, 17))
    batch = {"input_ids": jnp.asarray(tokens[..., :-1], jnp.int32),
             "labels": jnp.asarray(tokens[..., 1:], jnp.int32)}
    new_model, _, metrics, _ = step(model, optimizer.init(eqx.filter(model, eqx.is_inexact_array)), batch,
                                    jax.random.key(1))
    assert bool(metrics["step_applied"]) and np.isfinite(float(metrics["loss"]))
    before = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    after = jax.tree.leaves(eqx.filter(new_model, eqx.is_inexact_array))
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    if config.moe is not None:
        assert float(metrics["aux_loss"]) > 0
    if config.mtp is not None:
        assert np.isfinite(float(metrics["mtp/loss"])) and float(metrics["mtp/loss"]) > 0


def test_kimi_linear_layouts():
    for name, n in (("kimi-linear-tiny", 4), ("kimi-linear-1b", 16)):
        cfg = get_preset(name)
        assert cfg.resolved_layer_types == ("kda", "kda", "kda", "mla") * (n // 4)
        assert cfg.resolved_mlp_types == ("dense",) + ("moe",) * (n - 1)
        assert cfg.mla.use_rope is False and cfg.mla.q_lora_rank is None and cfg.moe.router == "sigmoid"
    counts = parameter_count(get_preset("kimi-linear-1b"))
    assert 0.8e9 <= counts.total <= 1.1e9 and 0.2e9 <= counts.active <= 0.3e9, counts
    ds = get_preset("deepseek-v3-tiny")
    assert ds.resolved_mlp_types[0] == "dense" and ds.mtp.depth == 1 and ds.moe.num_shared_experts == 1


# FLOPs ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q_lora_rank", [None, 24])
def test_mla_flops_match_the_layer_weights(q_lora_rank):
    cfg = dataclasses.replace(BASE, q_lora_rank=q_lora_rank, sliding_window=None)
    model_cfg = get_preset("tiny-mla").replace(d_model=32, mla=cfg, mlp_types=("none",) * 2)
    m = _layer(cfg)
    matmul_params = sum(int(l.weight.size) for l in (m.q_proj, m.q_a_proj, m.q_b_proj, m.kv_a_proj, m.kv_b_proj,
                                                     m.o_proj) if l is not None)
    seq = 256
    flops = estimate_flops_per_token(model_cfg, seq_len=seq)
    assert flops.attention_projections == pytest.approx(2 * 2 * matmul_params)  # two MLA layers
    context = (seq + 1) / 2
    core = 2 * cfg.num_heads * (cfg.qk_head_dim + cfg.v_head_dim) * context
    assert flops.attention_core == pytest.approx(2 * core)
    assert flops.linear_attention_core == 0 and flops.mlp == 0


# Config loading ---------------------------------------------------------------------------------
def test_mla_layers_load_from_a_mapping():
    from jla.config import _model_from_mapping

    raw = {
        "vocab_size": 512, "d_model": 64, "n_layers": 2, "d_ff": 128,
        "attention": {"num_heads": 4, "num_kv_heads": 2, "head_dim": 16},
        "mla": {"num_heads": 4, "q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 8, "v_head_dim": 16, "sliding_window": 8},
        "layer_types": ["mla", "mla"],
    }
    cfg = _model_from_mapping(raw)
    assert isinstance(cfg.mla, MLAConfig) and cfg.mla.sliding_window == 8
    assert cfg.resolved_layer_types == ("mla", "mla")
    assert _model_from_mapping({"preset": "kimi-linear-tiny", "d_ff": 64}).mla.use_rope is False
    with pytest.raises((ValueError, TypeError), match="mla settings"):
        _model_from_mapping({k: v for k, v in raw.items() if k != "mla"})
