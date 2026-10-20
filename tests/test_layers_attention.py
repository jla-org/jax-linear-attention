# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GQA attention: explicit oracle, causality, documents, GQA = MHA, KV caches and cuDNN parity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.attention import Attention, resolve_attention_implementation
from jla.layers.norm import rms_norm
from jla.layers.rope import apply_rope
from jla.models.config import AttentionConfig

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=2e-5, atol=2e-5)
D = 32
on_gpu = pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")


def _layer(seed=0, compute_dtype="float32", **cfg) -> Attention:
    base = dict(num_heads=4, num_kv_heads=2, head_dim=8)
    base.update(cfg)
    return Attention(D, AttentionConfig(**base), key=jax.random.key(seed), std=0.2, norm_eps=1e-6,
                     compute_dtype=compute_dtype)


def _x(batch=2, length=12, seed=1, dtype=jnp.float32):
    return jax.random.normal(jax.random.key(seed), (batch, length, D)).astype(dtype)


def _positions(batch, length):
    return jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))


def _oracle(layer: Attention, x, positions, segment_ids=None):
    cfg = layer.cfg
    batch, length, _ = x.shape
    heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    q = (x @ layer.q_proj.weight).reshape(batch, length, heads, head_dim)
    k = (x @ layer.k_proj.weight).reshape(batch, length, kv_heads, head_dim)
    v = (x @ layer.v_proj.weight).reshape(batch, length, kv_heads, head_dim)
    if cfg.qk_norm:
        q, k = rms_norm(q, layer.q_norm.weight, layer.q_norm.eps), rms_norm(k, layer.k_norm.weight, layer.k_norm.eps)
    if cfg.use_rope:
        q = apply_rope(q, positions, theta=cfg.rope_theta, rotary_fraction=cfg.rotary_fraction)
        k = apply_rope(k, positions, theta=cfg.rope_theta, rotary_fraction=cfg.rotary_fraction)
    k, v = jnp.repeat(k, heads // kv_heads, axis=2), jnp.repeat(v, heads // kv_heads, axis=2)
    scale = cfg.softmax_scale or head_dim**-0.5
    scores = jnp.einsum("bthd,bshd->bhts", q, k, precision="highest") * scale
    t = jnp.arange(length)
    mask = t[:, None] >= t[None, :]
    if cfg.sliding_window is not None:
        mask = mask & (t[:, None] - t[None, :] < cfg.sliding_window)
    mask = jnp.broadcast_to(mask, (batch, 1, length, length))
    if segment_ids is not None:
        mask = mask & (segment_ids[:, None, :, None] == segment_ids[:, None, None, :])
    probs = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
    o = jnp.einsum("bhts,bshd->bthd", probs, v, precision="highest").reshape(batch, length, -1)
    if cfg.output_gate:
        o = o * jax.nn.sigmoid(x @ layer.gate_proj.weight)
    return o @ layer.o_proj.weight


CONFIGS = [
    dict(),
    dict(qk_norm=False, rotary_fraction=0.5, output_gate=True),
    dict(sliding_window=4, rope_theta=1000.0),
    dict(use_rope=False, num_kv_heads=1, softmax_scale=0.2),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_explicit_oracle(cfg):
    layer = _layer(**cfg)
    x = _x()
    pos = _positions(2, 12)
    np.testing.assert_allclose(layer(x, pos), _oracle(layer, x, pos), **TOL)


def test_dtypes_follow_compute_dtype():
    layer = _layer(compute_dtype="bfloat16")
    out = layer(_x(), _positions(2, 12))
    assert out.dtype == jnp.bfloat16 and out.shape == (2, 12, D)
    assert layer.q_proj.weight.dtype == jnp.float32


@pytest.mark.parametrize("cfg", CONFIGS)
def test_future_tokens_never_change_the_past(cfg):
    layer = _layer(**cfg)
    x = _x()
    pos = _positions(2, 12)
    x2 = x.at[:, 8:].set(_x(2, 4, seed=9))
    np.testing.assert_allclose(layer(x2, pos)[:, :8], layer(x, pos)[:, :8], **TOL)


@pytest.mark.parametrize("cfg", CONFIGS[:3])
def test_packed_documents_match_separate_runs(cfg):
    layer = _layer(**cfg)
    lengths = [5, 2, 7]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    x = _x(1, 14, seed=3)
    packed = layer(x, pos, seg)
    np.testing.assert_allclose(packed, _oracle(layer, x, pos, seg), **TOL)
    start = 0
    for n in lengths:
        alone = layer(x[:, start : start + n], _positions(1, n))
        np.testing.assert_allclose(packed[:, start : start + n], alone, **TOL)
        start += n


def test_gqa_equals_mha_with_repeated_kv_heads():
    gqa = _layer(num_heads=4, num_kv_heads=2)
    mha = _layer(num_heads=4, num_kv_heads=4)

    def repeat_heads(w):  # [D, Hkv * Dh] -> [D, H * Dh] with each kv head repeated per group
        return jnp.repeat(w.reshape(D, 2, 8), 2, axis=1).reshape(D, 32)

    mha = eqx.tree_at(
        lambda m: (m.q_proj.weight, m.k_proj.weight, m.v_proj.weight, m.o_proj.weight),
        mha,
        (gqa.q_proj.weight, repeat_heads(gqa.k_proj.weight), repeat_heads(gqa.v_proj.weight), gqa.o_proj.weight),
    )
    x, pos = _x(), _positions(2, 12)
    np.testing.assert_allclose(gqa(x, pos), mha(x, pos), **TOL)


@pytest.mark.parametrize("window", [None, 5])
def test_prefill_then_decode_matches_the_full_pass(window):
    layer = _layer(sliding_window=window, output_gate=True)
    x = _x(2, 16, seed=4)
    full = layer(x, _positions(2, 16))
    cache = layer.init_cache(2, 24, jnp.float32)
    assert cache.ring == (window is not None) and cache.size == (window or 24)
    pos = _positions(2, 16)
    out, cache = layer.step(x[:, :7], pos[:, :7], cache)
    outs = [out]
    for t in range(7, 16):
        out, cache = layer.step(x[:, t : t + 1], pos[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)


def test_ring_cache_prefill_longer_than_the_window():
    layer = _layer(sliding_window=3)
    x = _x(1, 13, seed=6)
    pos = _positions(1, 13)
    full = layer(x, pos)
    cache = layer.init_cache(1, 32, jnp.float32)
    out, cache = layer.step(x[:, :11], pos[:, :11], cache)  # 11 tokens through a 3-slot ring
    np.testing.assert_allclose(out, full[:, :11], **TOL)
    np.testing.assert_array_equal(np.sort(np.asarray(cache.positions[0])), [8, 9, 10])
    out, cache = layer.step(x[:, 11:], pos[:, 11:], cache)
    np.testing.assert_allclose(out, full[:, 11:], **TOL)


@pytest.mark.parametrize("window", [None, 4])
def test_rows_decode_at_their_own_positions(window):
    layer = _layer(sliding_window=window)
    x = _x(2, 8, seed=7)
    cache = layer.init_cache(2, 16, jnp.float32)
    # Row 0 has a 6-token prompt, row 1 a 3-token prompt; position -1 marks padding (never cached).
    prompt_pos = jnp.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, -1, -1, -1]], jnp.int32)
    _, cache = layer.step(x[:, :6], prompt_pos, cache)
    assert int(jnp.sum(cache.positions[1] >= 0)) == 3
    nxt = jnp.stack([x[0, 6], x[1, 3]])[:, None]
    out, _ = layer.step(nxt, jnp.array([[6], [3]], jnp.int32), cache)
    want0 = layer(x[:1, :7], _positions(1, 7))[0, 6]
    want1 = layer(x[1:, :4], _positions(1, 4))[0, 3]
    np.testing.assert_allclose(out[:, 0], jnp.stack([want0, want1]), **TOL)


def test_implementation_choice_happens_at_call_time():
    q = jnp.zeros((1, 8, 4, 64), jnp.bfloat16)
    expected = "cudnn" if jax.default_backend() == "gpu" else "xla"
    assert resolve_attention_implementation("auto", q, q) == expected
    assert resolve_attention_implementation("auto", q.astype(jnp.float32), q) == "xla"
    assert resolve_attention_implementation("auto", q[:, :7], q[:, :7]) == "xla"  # odd length
    assert resolve_attention_implementation("xla", q, q) == "xla"
    with pytest.raises(ValueError):
        resolve_attention_implementation("flash", q, q)


@on_gpu
@pytest.mark.parametrize("cfg", [dict(), dict(sliding_window=33), dict(output_gate=True, rotary_fraction=0.5)])
@pytest.mark.parametrize("packed", [False, True])
def test_cudnn_matches_xla_forward_and_backward(cfg, packed):
    common = dict(num_heads=8, num_kv_heads=2, head_dim=64, **cfg)
    width = 256
    cudnn_layer = Attention(width, AttentionConfig(implementation="cudnn", **common), key=jax.random.key(0),
                            std=0.05, norm_eps=1e-6)
    xla_layer = Attention(width, AttentionConfig(implementation="xla", **common), key=jax.random.key(0),
                          std=0.05, norm_eps=1e-6)
    x = jax.random.normal(jax.random.key(1), (2, 128, width), jnp.bfloat16)
    pos = _positions(2, 128)
    seg = jnp.asarray(np.repeat([0, 1, 2], [50, 30, 48])[None].repeat(2, 0), jnp.int32) if packed else None

    def loss(layer, x):
        return jnp.sum(layer(x, pos, seg).astype(jnp.float32) ** 2)

    out_c = jax.jit(lambda x: cudnn_layer(x, pos, seg))(x)
    out_x = jax.jit(lambda x: xla_layer(x, pos, seg))(x)
    np.testing.assert_allclose(out_c.astype(jnp.float32), out_x.astype(jnp.float32), rtol=2e-2, atol=2e-2)
    g_c = jax.jit(jax.grad(lambda x: loss(cudnn_layer, x)))(x).astype(jnp.float32)
    g_x = jax.jit(jax.grad(lambda x: loss(xla_layer, x)))(x).astype(jnp.float32)
    scale = float(jnp.max(jnp.abs(g_x)))
    np.testing.assert_allclose(g_c / scale, g_x / scale, atol=2e-2)
    hlo = jax.jit(lambda x: cudnn_layer(x, pos, seg)).lower(x).as_text()
    assert "cudnn" in hlo.lower()
