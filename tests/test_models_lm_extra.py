# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-7 in LanguageModel, MiniMax-01 lightning attention, the Qwen3-Next shared-expert gate and MTP."""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.lightning_attention import LightningAttention, lightning_decay_rates
from jla.layers.moe import MoE
from jla.layers.rwkv7 import RWKV7Mixer, RWKV7TimeMix, ValueResidual
from jla.models import build_model, get_preset, parameter_count
from jla.models.config import LinearAttentionConfig, MoEConfig, MTPConfig
from jla.ops.cross_entropy import linear_cross_entropy
from jla.training.performance import estimate_flops_per_token


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=1e-4, atol=1e-4)


def _f32(name: str, **changes):
    cfg = get_preset(name).replace(compute_dtype="float32", **changes)
    if cfg.moe is not None:
        cfg = cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None))
    return cfg


def _perturbed(module, scale=0.05, seed=11):
    """Every float parameter moved off its initial value."""
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _loss(model, ids, *, with_mtp=True):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    mtp = model.mtp_loss(out, ids, targets)[0] if with_mtp else 0.0
    return ce.loss_sum / ce.num_valid + out.aux_loss + mtp


# RWKV-7 ---------------------------------------------------------------------------------------
_RWKV7_LA = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=16, head_v_dim=16, conv_size=0,
                                  qk_l2norm=False, chunk_size=16, rwkv7_lora_dims=(8, 8, 8, 8))
_SHARED = ("mu", "k_k", "k_a", "r_k", "r_proj", "k_proj", "v_proj", "o_proj", "w_lora", "a_lora", "g_lora",
           "norm_weight", "norm_bias")


def _mixer_and_time_mix(layer_idx: int, impl: str):
    """An RWKV7Mixer (perturbed) and the standalone RWKV7TimeMix carrying the same weights."""
    la = dataclasses.replace(_RWKV7_LA, implementation=impl)
    key = jax.random.key(layer_idx)
    mixer = _perturbed(RWKV7Mixer(32, la, layer_idx=layer_idx, n_layers=3, key=key, norm_eps=1e-5,
                                  compute_dtype="float32"), seed=layer_idx)
    tm = RWKV7TimeMix(32, head_dim=16, layer_idx=layer_idx, n_layers=3, key=jax.random.split(key)[0],
                      lora_dims=(8, 8, 8, 8), norm_eps=1e-5, chunk_size=16, impl=impl, compute_dtype="float32")
    names = _SHARED + (("v_lora",) if layer_idx > 0 else ())
    tm = eqx.tree_at(lambda m: [getattr(m, n) for n in names], tm, [getattr(mixer, n) for n in names])
    return mixer, tm


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_rwkv7_mixer_matches_the_time_mix_with_value_residual(impl):
    x0 = jax.random.normal(jax.random.key(1), (2, 21, 32))
    x1 = jax.random.normal(jax.random.key(2), (2, 21, 32))
    seg = jnp.asarray(np.repeat([[0, 1, 2], [4, 4, 5]], [5, 9, 7], axis=1), jnp.int32)
    m0, t0 = _mixer_and_time_mix(0, impl)
    m1, t1 = _mixer_and_time_mix(1, impl)
    vr = ValueResidual.empty(2, 21, 32, jnp.float32)
    y0, _, vr = m0.run(x0, segment_ids=seg, value_residual=vr)
    want0, v_first, _, _ = t0(x0, None, seg)
    np.testing.assert_allclose(y0, want0, **TOL)
    np.testing.assert_allclose(vr.v_first, v_first, **TOL)
    assert bool(vr.started)
    y1, _, vr1 = m1.run(x1, segment_ids=seg, value_residual=vr)
    want1, _, _, _ = t1(x1, v_first, seg)
    np.testing.assert_allclose(y1, want1, **TOL)
    np.testing.assert_array_equal(vr1.v_first, vr.v_first)  # later layers pass v_first on unchanged
    # Without a carry the mixer acts as a first layer.
    first = RWKV7TimeMix(32, head_dim=16, layer_idx=0, n_layers=3, key=jax.random.key(0), lora_dims=(8, 8, 8, 8),
                         norm_eps=1e-5, chunk_size=16, impl=impl, compute_dtype="float32")
    first = eqx.tree_at(lambda m: [getattr(m, n) for n in _SHARED], first, [getattr(m1, n) for n in _SHARED])
    np.testing.assert_allclose(m1(x1, seg), first(x1, None, seg)[0], **TOL)


def test_rwkv7_first_layer_value_lora_gets_no_gradient():
    m0, _ = _mixer_and_time_mix(0, "chunked")
    x = jax.random.normal(jax.random.key(3), (1, 9, 32))

    def f(m):
        y, _, vr = m.run(x, value_residual=ValueResidual.empty(1, 9, 32, jnp.float32))
        return jnp.sum(y**2) + jnp.sum(vr.v_first**2)

    grads = eqx.filter_grad(f)(m0)
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree.leaves(eqx.filter(grads.v_lora, eqx.is_array)))
    assert float(jnp.max(jnp.abs(grads.v_proj.weight))) > 0


@pytest.mark.parametrize("name", ["tiny-rwkv7", "tiny-minimax-01"])
def test_scan_and_loop_match_outputs_and_gradients(name):
    scanned = build_model(_f32(name, scan_layers=True), key=jax.random.key(3))
    looped = build_model(_f32(name, scan_layers=False), key=jax.random.key(3))
    period = len(scanned.blocks)

    def restack(blocks):
        return tuple(jax.tree.map(lambda *xs: jnp.stack(xs), *blocks[j::period]) for j in range(period))

    def assert_close(a, b, scaled_atol=0.0, **tol):
        la, lb = (jax.tree.leaves(eqx.filter(t, eqx.is_array)) for t in (a, b))
        assert len(la) == len(lb)
        for x, y in zip(la, lb):
            leaf_tol = dict(tol or TOL)
            leaf_tol["atol"] = max(leaf_tol["atol"], scaled_atol * float(np.max(np.abs(y), initial=0.0)))
            np.testing.assert_allclose(x, y, **leaf_tol)

    assert_close(scanned.blocks, restack(looped.blocks))  # depth-dependent initial values agree too
    if name == "tiny-rwkv7":  # zero o_proj and value make every block the identity at init: move them all
        looped = eqx.tree_at(lambda m: m.blocks, looped, _perturbed(looped.blocks))
        arrays, static = eqx.partition(scanned.blocks, eqx.is_array)
        moved = jax.tree.leaves(eqx.filter(restack(looped.blocks), eqx.is_array))
        scanned = eqx.tree_at(lambda m: m.blocks, scanned,
                              eqx.combine(jax.tree.unflatten(jax.tree.structure(arrays), moved), static))
    ids = jax.random.randint(jax.random.key(4), (2, 19), 0, 512)
    np.testing.assert_allclose(scanned(ids).hidden, looped(ids).hidden, **TOL)
    g_scan = eqx.filter_grad(_loss)(scanned, ids)
    g_loop = eqx.filter_grad(_loss)(looped, ids)
    # GPU float32 gradients are not bitwise reproducible even within one layout (two calls of the same
    # tiny-minimax-01 model differ by up to ~1e-5 in the tied embedding's gradient, max |g| ~ 3), so on
    # GPU each leaf's absolute floor is 2e-5 of its largest entry; CPU keeps 1e-6.
    scaled = 0.0 if jax.default_backend() == "cpu" else 2e-5
    if name == "tiny-rwkv7":  # moved weights: the chunked op's float32 rounding (~3e-6 against the reference,
        scaled = max(scaled, 1e-3)  # see the remat test) reorders between the layouts; seen: one embedding
        # gradient element of 32768 off by 2e-6 on CPU. A wrong carry or v_first is off by order 1.
    assert_close(g_scan.blocks, restack(g_loop.blocks), scaled_atol=scaled, rtol=1e-4, atol=1e-6)
    assert_close(g_scan.embed, g_loop.embed, scaled_atol=scaled, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("policy", ["full", "save_attention"])
@pytest.mark.parametrize("scan", [True, False])
def test_rwkv7_remat_preserves_outputs_and_gradients(policy, scan):
    # The token-by-token op makes remat bitwise neutral; the chunked op's own float32 rounding
    # (~3e-6 against the reference here) is reordered by remat in the unscanned layout.
    def cfg(remat):
        c = _f32("tiny-rwkv7", remat=remat, scan_layers=scan)
        return c.replace(linear_attention=dataclasses.replace(c.linear_attention, implementation="reference"))

    # Perturbed (same seed, same tree): with the zero o_proj and value every block is the identity at init.
    base = _perturbed(build_model(cfg("none"), key=jax.random.key(4)))
    remat = _perturbed(build_model(cfg(policy), key=jax.random.key(4)))
    ids = jax.random.randint(jax.random.key(5), (2, 17), 0, 512)
    np.testing.assert_allclose(remat(ids).hidden, base(ids).hidden, rtol=1e-6, atol=1e-6)
    g_base, g_remat = (eqx.filter_grad(_loss)(m, ids) for m in (base, remat))
    for a, b in zip(jax.tree.leaves(eqx.filter(g_remat, eqx.is_array)), jax.tree.leaves(eqx.filter(g_base, eqx.is_array))):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_rwkv7_value_residual_reaches_later_layers():
    model = _perturbed(build_model(_f32("tiny-rwkv7", scan_layers=False), key=jax.random.key(5)))
    ids = jax.random.randint(jax.random.key(6), (1, 12), 0, 512)
    # Switching the last layer's value-residual mix off (sigmoid(-30) ~ 0) changes the output.
    mixed_off = eqx.tree_at(lambda m: m.blocks[2].mixer.v_lora.up.bias, model,
                            jnp.full_like(model.blocks[2].mixer.v_lora.up.bias, -30.0))
    assert not np.allclose(mixed_off(ids).hidden, model(ids).hidden, atol=1e-5)
    # The first layer's value feeds the last layer: d(out of layer 3) / d(layer-1 v_proj) through v_first.
    grads = eqx.filter_grad(lambda m: jnp.sum(m(ids).hidden ** 2))(model)
    off = eqx.filter_grad(lambda m: jnp.sum(m(ids).hidden ** 2))(mixed_off)
    assert not np.allclose(grads.blocks[0].mixer.v_proj.weight, off.blocks[0].mixer.v_proj.weight, atol=1e-7)


@pytest.mark.parametrize("name", ["tiny-rwkv7", "tiny-minimax-01", "tiny-qwen3-next"])
def test_cpu_train_steps_reduce_the_loss(name):
    model = build_model(_f32(name), key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (4, 32), 0, 512)

    @eqx.filter_jit
    def step(m):
        loss, grads = eqx.filter_value_and_grad(_loss)(m, ids)
        return eqx.apply_updates(m, jax.tree.map(lambda g: -0.5 * g, grads)), loss, grads

    losses = []
    for i in range(6):
        model, loss, grads = step(model)
        losses.append(float(loss))
        if i == 0:
            leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
            assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert all(np.isfinite(losses)) and losses[-1] < losses[0] - 0.05, losses


# MiniMax-01 lightning attention ------------------------------------------------------------------
def test_lightning_decay_rates_follow_the_minimax_schedule():
    for heads, layer, layers in ((64, 0, 80), (64, 41, 80), (8, 15, 16), (4, 3, 8)):
        rates = np.asarray(lightning_decay_rates(heads, layer, layers), np.float64)
        alibi = np.array([2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)])  # ALiBi slopes, 2^a heads
        factor = 1.0 - layer / (layers - 1 + 1e-5) + 1e-5
        np.testing.assert_allclose(rates, alibi * factor, rtol=1e-5)
    assert float(lightning_decay_rates(64, 79, 80)[0]) < 1e-4  # the deepest layer barely decays


def _lightning(seed=0, heads=2, dim=8, d=24, layer_idx=1, n_layers=4, impl="auto", chunk=16, dtype="float32"):
    cfg = LinearAttentionConfig(num_k_heads=heads, num_v_heads=heads, head_k_dim=dim, head_v_dim=dim, conv_size=0,
                                qk_l2norm=False, chunk_size=chunk, implementation=impl)
    layer = LightningAttention(d, cfg, layer_idx=layer_idx, n_layers=n_layers, key=jax.random.key(seed), std=0.2,
                               compute_dtype=dtype)
    return eqx.tree_at(lambda m: m.norm.weight, layer, 1.0 + 0.3 * jax.random.normal(jax.random.key(9), (heads * dim,)))


def _naive_lightning(layer, x, segment_ids=None):
    """float64 loop of MiniMaxLightningAttention: S = exp(-s) S + k^T v, o = q S (no scale)."""
    f = lambda a: np.asarray(a, np.float64)  # noqa: E731
    x = f(x)
    cfg = layer.cfg
    heads, dim = cfg.num_v_heads, cfg.head_v_dim
    batch, length, _ = x.shape
    z = x @ f(layer.qkv_proj.weight)
    qkv = (z / (1.0 + np.exp(-z))).reshape(batch, length, heads, 3, dim)
    decay = np.exp(-f(layer.decay_rate))
    o = np.zeros((batch, length, heads, dim))
    for b in range(batch):
        for h in range(heads):
            s = np.zeros((dim, dim))
            for t in range(length):
                if segment_ids is not None and t > 0 and segment_ids[b, t] != segment_ids[b, t - 1]:
                    s = np.zeros((dim, dim))
                s = decay[h] * s + np.outer(qkv[b, t, h, 1], qkv[b, t, h, 2])
                o[b, t, h] = qkv[b, t, h, 0] @ s
    o = o.reshape(batch, length, -1)
    o = o / np.sqrt((o**2).mean(-1, keepdims=True) + 1e-6) * f(layer.norm.weight)
    gate = 1.0 / (1.0 + np.exp(-(x @ f(layer.gate_proj.weight))))
    return (gate * o) @ f(layer.o_proj.weight)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("packed", [False, True])
def test_lightning_layer_matches_a_naive_loop(impl, packed):
    layer = _lightning(impl=impl)
    x = jax.random.normal(jax.random.key(2), (2, 37, 24))
    seg = jnp.asarray(np.repeat([[0, 1, 2], [3, 3, 3]], [11, 17, 9], axis=1), jnp.int32) if packed else None
    want = _naive_lightning(layer, x, None if seg is None else np.asarray(seg))
    np.testing.assert_allclose(layer(x, seg), want, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
@pytest.mark.parametrize("dim", [64, 128])
def test_lightning_on_the_mosaic_gpu_kernel_matches_the_reference(dim):
    """bf16 lightning layers dispatch ``simple_gla`` to the Mosaic GPU kernel under ``auto``."""
    from jla.ops.mosaic_gpu.scalar_gate import supports_simple_gla

    common = dict(seed=5, heads=2, dim=dim, d=96, layer_idx=2, n_layers=8, chunk=64, dtype="bfloat16")
    fast, ref = _lightning(impl="auto", **common), _lightning(impl="reference", **common)
    x = jax.random.normal(jax.random.key(6), (2, 200, 96)).astype(jnp.bfloat16)
    seg = jnp.asarray(np.repeat([[0, 1, 2], [3, 3, 4]], [37, 100, 63], axis=1), jnp.int32)
    q = jnp.zeros((2, 200, 2, dim), jnp.bfloat16)
    g = jnp.zeros((2, 200, 2), jnp.float32)
    assert supports_simple_gla(q, q, q, g, scale=1.0, chunk_size=64, segment_ids=seg)

    def rel(a, b):
        a, b = (np.asarray(t, np.float32) for t in (a, b))
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert rel(fast(x, seg), ref(x, seg)) < 1e-2

    def loss(m, x):
        return jnp.sum(m(x, seg).astype(jnp.float32) ** 2)

    grad = eqx.filter_grad(lambda mx: loss(*mx))  # the layer's int32 depth leaf is not differentiable
    gm_fast, gx_fast = grad((fast, x))
    gm_ref, gx_ref = grad((ref, x))
    assert rel(gx_fast, gx_ref) < 2e-2
    assert rel(gm_fast.qkv_proj.weight, gm_ref.qkv_proj.weight) < 2e-2


def test_lightning_decay_is_fixed_and_decode_matches():
    layer = _lightning(seed=3)
    x = jax.random.normal(jax.random.key(4), (2, 19, 24))
    grads = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(layer)
    assert grads.layer_index is None  # the depth is an int32 leaf: not a parameter, no gradient
    assert layer.layer_index.dtype == jnp.int32
    np.testing.assert_array_equal(layer.decay_rate, lightning_decay_rates(2, 1, 4))
    assert float(jnp.max(jnp.abs(grads.qkv_proj.weight))) > 0
    full = layer(x)
    cache = layer.init_cache(2, jnp.float32)
    y, cache = layer.step(x[:, :7], cache)
    outs = [y]
    for t in range(7, 19):
        y, cache = layer.step(x[:, t : t + 1], cache)
        outs.append(y)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)


def test_lightning_config_fails_loudly():
    common = dict(layer_idx=0, n_layers=2, key=jax.random.key(0), std=0.02)
    base = dict(num_k_heads=2, num_v_heads=2, head_k_dim=8, head_v_dim=8, conv_size=0, qk_l2norm=False)
    for change, match in (({"conv_size": 4}, "conv_size"), ({"qk_l2norm": True}, "qk_l2norm"),
                          ({"output_gate": False}, "output_gate"), ({"num_v_heads": 4}, "num_v_heads")):
        with pytest.raises(ValueError, match=match):
            LightningAttention(16, LinearAttentionConfig(**{**base, **change}), **common)


def test_minimax_preset_layout():
    big = get_preset("minimax-01-1b")
    types = big.resolved_layer_types
    assert [i for i, t in enumerate(types) if t == "attention"] == [7, 15]
    assert big.moe.num_experts == 32 and big.moe.top_k == 2 and big.moe.router == "softmax"
    assert big.moe.normalize_top_k and big.moe.num_shared_experts == 0 and big.moe.aux_loss_coef == 2e-3
    assert big.attention.rotary_fraction == 0.5 and big.attention.rope_theta == 1e7 and not big.attention.qk_norm
    assert big.linear_attention.head_v_dim == 128 and not big.tie_embeddings
    count = parameter_count(big)
    assert 0.9e9 < count.total < 1.2e9, count


# Qwen3-Next: shared-expert gate and MTP ---------------------------------------------------------
def test_shared_expert_gate_matches_a_manual_combination():
    cfg = MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, num_shared_experts=1, shared_expert_gate=True)
    moe = MoE(16, cfg, key=jax.random.key(0), std=0.3, compute_dtype="float32")
    ungated = eqx.tree_at(lambda m: m.shared_gate, moe, None)
    x = jax.random.normal(jax.random.key(1), (3, 5, 16))
    flat = np.asarray(x, np.float64).reshape(-1, 16)
    gate = 1.0 / (1.0 + np.exp(-(flat @ np.asarray(moe.shared_gate, np.float64))))
    shared = np.asarray(moe.shared(x.reshape(-1, 16)), np.float64)
    routed = np.asarray(ungated(x)[0], np.float64).reshape(-1, 16) - shared
    want = (routed + gate * shared).reshape(x.shape)
    np.testing.assert_allclose(moe(x)[0], want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(moe.reference(x), want, rtol=1e-5, atol=1e-5)
    g = jax.grad(lambda m: jnp.sum(m(x)[0] ** 2))(moe)
    assert float(jnp.max(jnp.abs(g.shared_gate))) > 0
    with pytest.raises(ValueError, match="shared_expert_gate"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, shared_expert_gate=True)


def test_qwen3_next_mtp_is_a_moe_decoder_layer_with_its_aux_loss():
    cfg = _f32("tiny-qwen3-next")
    assert cfg.mtp == MTPConfig(depth=1, kind="deepseek", mlp_type="moe") and cfg.moe.shared_expert_gate
    model = build_model(cfg, key=jax.random.key(0))
    assert isinstance(model.mtp.blocks[0].mlp, MoE) and model.mtp.blocks[0].mlp.shared_gate is not None
    ids = jax.random.randint(jax.random.key(1), (2, 16), 0, 512)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((2, 1), -100)], axis=1)

    def mtp_loss(m):
        return m.mtp_loss(m(ids), ids, targets)

    (loss, metrics), grads = eqx.filter_value_and_grad(mtp_loss, has_aux=True)(model)
    assert set(metrics) == {"mtp/loss", "mtp/loss_1", "mtp/moe_aux_loss"}
    np.testing.assert_allclose(loss, 0.3 * metrics["mtp/loss"] + metrics["mtp/moe_aux_loss"], rtol=1e-6)
    assert float(metrics["mtp/moe_aux_loss"]) > 0
    assert float(jnp.max(jnp.abs(grads.mtp.blocks[0].mlp.router))) > 0
    # Active parameters leave out the MTP block's unselected experts too.
    count = parameter_count(cfg)
    moe = cfg.moe
    inactive = (sum(t == "moe" for t in cfg.resolved_mlp_types) + 1) * (moe.num_experts - moe.top_k) * 3 * 64 * 16
    assert count.active == count.total - inactive


@pytest.mark.parametrize("name", ["qwen3-next-300m"])
def test_mtp_preset_flops_match_parameters(name):
    """The shared FLOP test skips MTP presets: check the trunk and the MTP head separately."""
    cfg = get_preset(name)
    trunk = cfg.replace(mtp=None)
    counts, flops = parameter_count(trunk), estimate_flops_per_token(trunk, seq_len=2048)
    matmul = flops.attention_projections + flops.linear_attention_projections + flops.mlp
    assert matmul == pytest.approx(2 * (counts.active - counts.embedding), rel=0.02)
    full = parameter_count(cfg)
    mtp_params = full.active - counts.active
    mtp = estimate_flops_per_token(cfg, seq_len=2048).mtp - cfg.mtp.depth * flops.lm_head  # the head is shared
    attention_core = estimate_flops_per_token(cfg, seq_len=2048).attention_core / cfg.resolved_layer_types.count(
        "attention")
    assert mtp - attention_core == pytest.approx(2 * mtp_params, rel=0.02)


def test_new_real_presets_have_a_sane_size():
    rwkv = parameter_count(get_preset("rwkv7-0.1b"))
    assert 0.12e9 < rwkv.total < 0.14e9 and rwkv.embedding == 50304 * 768
    mm = parameter_count(get_preset("minimax-01-1b"))
    assert 0.15e9 < mm.active < 0.25e9
    assert math.isclose(estimate_flops_per_token(get_preset("rwkv7-0.1b"), 4096).training,
                        3 * estimate_flops_per_token(get_preset("rwkv7-0.1b"), 4096).forward)


# Regression tests --------------------------------------------------------------------------------
def _lightning_rates(model, cfg):
    """``[n_lightning, H]`` decay rates of every lightning layer, in depth order (either layout)."""
    period = len(model.blocks)
    out = []
    for i, t in enumerate(cfg.resolved_layer_types):
        if t != "lightning":
            continue
        rates = model.blocks[i % period].mixer.decay_rate
        out.append(rates[i // period] if cfg.scan_layers else rates)
    return jnp.stack(out)


@pytest.mark.parametrize("optimizer_type", ["adamw", "muon"])
@pytest.mark.parametrize("scan", [True, False])
def test_template_optimizer_never_moves_the_lightning_decay_schedule(scan, optimizer_type):
    import optax

    from jla.training.optimizers import create_optimizer

    cfg = _f32("tiny-minimax-01", scan_layers=scan)
    model = build_model(cfg, key=jax.random.key(0))
    heads = cfg.linear_attention.num_v_heads
    want = jnp.stack([lightning_decay_rates(heads, i, cfg.n_layers)
                      for i, t in enumerate(cfg.resolved_layer_types) if t == "lightning"])
    np.testing.assert_array_equal(_lightning_rates(model, cfg), want)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt = create_optimizer(lr=3e-2, weight_decay=0.1, optimizer_type=optimizer_type)  # OptimConfig's decay
    state = opt.init(params)
    for _ in range(3):
        updates, state = opt.update(jax.tree.map(jnp.ones_like, params), state, params)
        params = optax.apply_updates(params, updates)
    trained = eqx.combine(params, static)
    np.testing.assert_array_equal(_lightning_rates(trained, cfg), want)
    assert not np.allclose(trained.blocks[0].mixer.qkv_proj.weight, model.blocks[0].mixer.qkv_proj.weight)


def test_lightning_decay_rates_stay_float32_with_bf16_parameters():
    cfg = get_preset("tiny-minimax-01").replace(param_dtype="bfloat16")
    model = build_model(cfg, key=jax.random.key(0))
    assert model.blocks[0].mixer.qkv_proj.weight.dtype == jnp.bfloat16
    rates = _lightning_rates(model, cfg)
    assert rates.dtype == jnp.float32
    np.testing.assert_array_equal(rates[0], lightning_decay_rates(cfg.linear_attention.num_v_heads, 0, cfg.n_layers))
    # Integer leaves are not parameters.
    count = parameter_count(cfg)
    floats = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))
    assert count.total == floats


@pytest.mark.parametrize("name", ["tiny-rwkv7", "tiny-minimax-01", "tiny-qwen3-next"])
def test_optimizer_masks_are_the_same_in_the_scanned_and_looped_layouts(name):
    import optax

    from jla.training.optimizers import _muon_dimension_numbers, decay_mask

    def masks(scan):
        params = eqx.filter(build_model(_f32(name, scan_layers=scan), key=jax.random.key(0)), eqx.is_inexact_array)
        muon = jax.tree.map(lambda n: n is not None, _muon_dimension_numbers(params),
                            is_leaf=lambda n: n is None or isinstance(n, optax.contrib.MuonDimensionNumbers))
        return params, decay_mask(params), muon

    scanned, decay_scan, muon_scan = masks(True)
    _, decay_loop, muon_loop = masks(False)
    period = len(scanned.blocks)
    for i in range(get_preset(name).n_layers):
        j = i % period
        assert jax.tree.leaves(decay_scan.blocks[j]) == jax.tree.leaves(decay_loop.blocks[i]), (name, i)
        assert jax.tree.leaves(muon_scan.blocks[j]) == jax.tree.leaves(muon_loop.blocks[i]), (name, i)
    if name == "tiny-rwkv7":  # stacked [L, D] vectors and the mu [6, D] / r_k [H, K] vector sets: no decay, no Muon
        mixer, muon_mixer = decay_scan.blocks[0].mixer, muon_scan.blocks[0].mixer
        assert not any((mixer.k_k, mixer.k_a, mixer.mu, mixer.r_k))
        assert not any((muon_mixer.k_k, muon_mixer.k_a, muon_mixer.mu, muon_mixer.r_k))
        assert mixer.r_proj.weight and muon_mixer.r_proj.weight


@pytest.mark.parametrize("policy", ["full", "save_attention"])
@pytest.mark.parametrize("scan", [True, False])
@pytest.mark.parametrize("name", ["tiny-rwkv7", "tiny-minimax-01"])
def test_remat_with_the_default_ops_preserves_outputs_and_gradients(name, policy, scan):
    """The default (chunked) RWKV-7 and lightning ops under remat; the chunked RWKV-7 op's float32
    rounding (~3e-6) is reordered by remat in the unscanned layout, hence the forward tolerance."""
    base = build_model(_f32(name, remat="none", scan_layers=scan), key=jax.random.key(4))
    remat = build_model(_f32(name, remat=policy, scan_layers=scan), key=jax.random.key(4))
    ids = jax.random.randint(jax.random.key(5), (2, 17), 0, 512)
    np.testing.assert_allclose(remat(ids).hidden, base(ids).hidden, rtol=1e-5, atol=1e-5)
    g_base, g_remat = (eqx.filter_grad(_loss)(m, ids) for m in (base, remat))
    for a, b in zip(jax.tree.leaves(eqx.filter(g_remat, eqx.is_array)), jax.tree.leaves(eqx.filter(g_base, eqx.is_array))):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_shared_expert_gate_init_is_independent_of_the_other_weights():
    cfg = MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, num_shared_experts=1, shared_expert_gate=True)
    moe = MoE(16, cfg, key=jax.random.key(0), std=0.3, compute_dtype="float32")
    gate = np.asarray(moe.shared_gate[:, 0])
    vectors = [*np.asarray(moe.shared.down.weight), *np.asarray(moe.shared.gate_up).reshape(16, -1).T,
               *np.asarray(moe.router).T, *np.asarray(moe.down).reshape(-1, 16),
               *np.moveaxis(np.asarray(moe.gate_up), 1, -1).reshape(-1, 16)]
    assert all(abs(np.corrcoef(gate, v)[0, 1]) < 0.95 for v in vectors)
    # The other parameters keep the initialization they have without the gate.
    plain = MoE(16, dataclasses.replace(cfg, shared_expert_gate=False), key=jax.random.key(0), std=0.3,
                compute_dtype="float32")
    for a, b in zip(jax.tree.leaves(eqx.filter(plain, eqx.is_array)),
                    jax.tree.leaves(eqx.filter(eqx.tree_at(lambda m: m.shared_gate, moe, None), eqx.is_array))):
        np.testing.assert_array_equal(a, b)


def test_chained_mtp_rejects_a_moe_refiner():
    with pytest.raises(ValueError, match="chained"):
        MTPConfig(depth=2, kind="chained", mlp_type="moe")
    assert MTPConfig(depth=2, kind="chained").mlp_type == "dense"


def test_mtp_moe_aux_loss_gets_the_train_steps_cap():
    from jla.training.step import LossConfig, lm_loss

    model = build_model(_f32("tiny-qwen3-next"), key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (2, 16), 0, 512)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((2, 1), -100)], axis=1)
    out = model(ids)
    free, metrics = model.mtp_loss(out, ids, targets)
    aux = float(metrics["mtp/moe_aux_loss"])
    cap = 0.5 * aux
    capped, capped_metrics = model.mtp_loss(out, ids, targets, aux_loss_cap=cap)
    np.testing.assert_allclose(free, 0.3 * metrics["mtp/loss"] + aux, rtol=1e-6)
    np.testing.assert_allclose(capped, 0.3 * metrics["mtp/loss"] + cap, rtol=1e-6)
    np.testing.assert_allclose(capped_metrics["mtp/moe_aux_loss"], aux, rtol=1e-6)  # the metric is unclipped
    # Through the train step's loss.
    batch = {"input_ids": ids, "labels": targets}
    _, step_aux = lm_loss(LossConfig(compute_dtype="float32", aux_loss_cap=cap))(model, batch, jax.random.key(2))
    np.testing.assert_allclose(step_aux["mtp_loss"], capped, rtol=1e-5)
    _, step_free = lm_loss(LossConfig(compute_dtype="float32", aux_loss_cap=0.0))(model, batch, jax.random.key(2))
    np.testing.assert_allclose(step_free["mtp_loss"], free, rtol=1e-5)


def test_minimax_trains_and_resumes_through_the_trainer(tmp_path):
    """The lightning layers' int32 depth leaves go through the trainer's sharded init, weight decay,
    checkpointing and resume, and the decay schedule comes back unchanged."""
    from _toy_model import experiment

    from jla.training.trainer import Trainer

    def run(steps):
        return Trainer(experiment(str(tmp_path / "minimax"), model="tiny-minimax-01",
                                  train={"max_steps": steps, "log_every": 2, "save_every": 2, "final_eval": False},
                                  optim={"weight_decay": 0.1, "schedule_steps": 4})).run()

    first = run(2)
    second = run(4)  # resumes from the step-2 checkpoint
    assert first.step == 2 and second.step == 4
    assert np.isfinite([r["loss"] for r in first.history + second.history]).all()
    cfg = second.model.config
    want = jnp.stack([lightning_decay_rates(cfg.linear_attention.num_v_heads, i, cfg.n_layers)
                      for i, t in enumerate(cfg.resolved_layer_types) if t == "lightning"])
    np.testing.assert_array_equal(_lightning_rates(second.model, cfg), want)
