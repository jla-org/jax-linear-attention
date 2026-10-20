# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MoonshotAI/MoBA (MIT), moba/moba_naive.py (the NumPy oracle ``_naive_moba``)
"""MoBA (Mixture of Block Attention): exactness, selection against a naive oracle, gradients, wiring, FLOPs."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.attention import attention
from jla.models import AttentionConfig, build_model, get_preset
from jla.ops.moba import moba_attention, moba_cached_mask, moba_mask
from jla.training.performance import _moba_core_flops, estimate_flops_per_token


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _naive_moba(q, k, v, seg, block, top_k, scale=None):
    """NumPy float64 port of MoBA's ``moba_attn_varlen_naive``, one document at a time.

    Returns the output ``[B, T, H, D]`` and the attended mask ``[B, H, T, T]``.
    """
    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    batch, length, heads, dim = q.shape
    group = heads // k.shape[2]
    k, v = np.repeat(k, group, axis=2), np.repeat(v, group, axis=2)
    scale = dim ** -0.5 if scale is None else scale
    out = np.zeros_like(q)
    mask = np.zeros((batch, heads, length, length), bool)
    for b in range(batch):
        bounds = [0] + [t for t in range(1, length) if seg[b, t] != seg[b, t - 1]] + [length]
        for s, e in zip(bounds[:-1], bounds[1:]):
            n = e - s
            num_blocks = -(-n // block)
            means = np.stack([k[b, s + i * block : s + min(n, (i + 1) * block)].mean(0) for i in range(num_blocks)])
            gate = np.einsum("shd,nhd->hsn", q[b, s:e], means)
            for i in range(num_blocks):
                gate[:, : (i + 1) * block, i] = -np.inf  # this block and later ones: not gated
                gate[:, i * block : (i + 1) * block, i] = np.inf  # own block: always attended
            idx = np.argsort(-gate, axis=-1, kind="stable")[..., : min(top_k, num_blocks)]
            need = np.zeros(gate.shape, bool)
            np.put_along_axis(need, idx, True, axis=-1)
            tok = np.repeat(need, block, axis=-1)[:, :, :n] & np.tril(np.ones((n, n), bool))
            mask[b, :, s:e, s:e] = tok
            logits = np.where(tok, np.einsum("xhd,yhd->hxy", q[b, s:e], k[b, s:e]) * scale, -np.inf)
            p = np.exp(logits - logits.max(-1, keepdims=True))
            out[b, s:e] = np.einsum("hxy,yhd->xhd", p / p.sum(-1, keepdims=True), v[b, s:e])
    return out, mask


def _inputs(batch=2, length=37, heads=4, kv_heads=2, dim=8, seed=0, dtype=jnp.float32):
    kq, kk, kv = jax.random.split(jax.random.key(seed), 3)
    q = jax.random.normal(kq, (batch, length, heads, dim), dtype)
    k = jax.random.normal(kk, (batch, length, kv_heads, dim), dtype)
    v = jax.random.normal(kv, (batch, length, kv_heads, dim), dtype)
    return q, k, v


def _segments(batch, length):
    seg = np.zeros((batch, length), np.int32)
    seg[0, 11:] = 1  # row 0: documents of 11, 17 and 9 tokens
    seg[0, 28:] = 2
    if batch > 1:
        seg[1, 30:] = 5  # row 1: documents of 30 and 7 tokens
    return jnp.asarray(seg)


@pytest.mark.parametrize("packed", [False, True])
def test_topk_covering_all_blocks_is_exact_causal_attention(packed):
    q, k, v = _inputs()
    seg = _segments(2, 37) if packed else None
    mask = moba_mask(q, k, block_size=5, top_k=8, segment_ids=seg)  # 37 tokens = 8 blocks
    pos = np.arange(37)
    ids = np.zeros((2, 37)) if seg is None else np.asarray(seg)
    dense = (pos[:, None] >= pos[None, :]) & (ids[:, :, None] == ids[:, None, :])
    np.testing.assert_array_equal(np.asarray(mask), np.broadcast_to(dense[:, None], mask.shape))
    out = moba_attention(q, k, v, block_size=5, top_k=8, segment_ids=seg)
    ref = attention(q, k, v, segment_ids=seg, implementation="xla")
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=0, atol=1e-6)


@pytest.mark.parametrize("block,top_k", [(4, 1), (4, 2), (5, 3), (3, 4), (16, 2), (1, 3)])
@pytest.mark.parametrize("packed", [False, True])
def test_selection_and_output_match_naive_moba(block, top_k, packed):
    q, k, v = _inputs(seed=block * 10 + top_k)
    seg = _segments(2, 37) if packed else jnp.zeros((2, 37), jnp.int32)
    mask = moba_mask(q, k, block_size=block, top_k=top_k, segment_ids=seg if packed else None)
    out = moba_attention(q, k, v, block_size=block, top_k=top_k, segment_ids=seg if packed else None)
    ref_out, ref_mask = _naive_moba(q, k, v, np.asarray(seg), block, top_k)
    np.testing.assert_array_equal(np.asarray(mask), ref_mask)
    np.testing.assert_allclose(np.asarray(out), ref_out, rtol=1e-5, atol=1e-5)
    if 1 < top_k < -(-37 // block):  # genuinely sparse: some earlier blocks are skipped
        assert not ref_mask.all(where=np.tril(np.ones((37, 37), bool)))


def test_each_query_attends_at_most_top_k_blocks():
    q, k, v = _inputs(length=64, seed=3)
    mask = np.asarray(moba_mask(q, k, block_size=8, top_k=3))
    blocks_seen = mask.reshape(2, 4, 64, 8, 8).any(-1).sum(-1)  # [B, H, T]
    expected = np.minimum(3, np.arange(64) // 8 + 1)
    np.testing.assert_array_equal(blocks_seen, np.broadcast_to(expected, blocks_seen.shape))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gradients_are_finite(dtype):
    q, k, v = _inputs(dtype=dtype, seed=7)
    seg = _segments(2, 37)
    loss = lambda q, k, v: jnp.sum(
        moba_attention(q, k, v, block_size=4, top_k=2, segment_ids=seg).astype(jnp.float32) ** 2)
    grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    for g, x in zip(grads, (q, k, v)):
        assert g.shape == x.shape and g.dtype == x.dtype
        assert np.isfinite(np.asarray(g, np.float32)).all()
        assert float(jnp.abs(g.astype(jnp.float32)).sum()) > 0


def test_gradients_match_dense_attention_when_every_block_is_selected():
    q, k, v = _inputs(seed=11)
    seg = _segments(2, 37)
    moba = lambda q, k, v: jnp.sum(jnp.sin(moba_attention(q, k, v, block_size=6, top_k=7, segment_ids=seg)))
    dense = lambda q, k, v: jnp.sum(jnp.sin(attention(q, k, v, segment_ids=seg, implementation="xla")))
    for g, r in zip(jax.grad(moba, argnums=(0, 1, 2))(q, k, v), jax.grad(dense, argnums=(0, 1, 2))(q, k, v)):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-5, atol=1e-6)


def test_tiny_moba_model_trains_and_reduces_to_dense_attention_with_all_blocks():
    cfg = get_preset("tiny-moba").replace(compute_dtype="float32")
    ids = jax.random.randint(jax.random.key(1), (2, 16), 0, 512)
    seg = jnp.asarray(np.repeat([[0] * 9 + [1] * 7], 2, axis=0))
    model = build_model(cfg, key=jax.random.key(0))

    def loss(m):
        return jnp.mean(m(ids, segment_ids=seg).hidden ** 2)

    value, grads = jax.jit(lambda m: eqx.filter_value_and_grad(loss)(m))(model)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))

    full = cfg.replace(attention=dataclasses.replace(cfg.attention, moba_top_k=4))  # 16 tokens = 4 blocks
    dense = cfg.replace(attention=dataclasses.replace(cfg.attention, moba_block_size=None))
    h_full = build_model(full, key=jax.random.key(0))(ids, segment_ids=seg).hidden
    h_dense = build_model(dense, key=jax.random.key(0))(ids, segment_ids=seg).hidden
    h_sparse = model(ids, segment_ids=seg).hidden
    np.testing.assert_allclose(np.asarray(h_full), np.asarray(h_dense), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(h_sparse), np.asarray(h_dense), atol=1e-4)


def test_moba_hybrid_preset_shape():
    cfg = get_preset("moba-hybrid-125m")
    assert cfg.attention.moba_block_size == 512 and cfg.attention.moba_top_k == 3
    assert "attention" in cfg.resolved_layer_types and "gated_deltanet" in cfg.resolved_layer_types


@pytest.mark.parametrize("scan", [True, False])
def test_prefill_then_decode_matches_the_forward_pass(scan):
    cfg = get_preset("tiny-moba").replace(compute_dtype="float32", scan_layers=scan)
    model = build_model(cfg, key=jax.random.key(5))
    ids = jax.random.randint(jax.random.key(6), (2, 14), 0, 512)
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


def test_cached_mask_matches_the_full_sequence_mask():
    q, k, _ = _inputs(length=29, seed=13)
    size = 40  # cache longer than the sequence: empty slots must stay invisible
    keys = jnp.zeros((2, size, 2, 8)).at[:, :29].set(k)
    kv_pos = jnp.where(jnp.arange(size) < 29, jnp.arange(size), -1)[None].repeat(2, 0)
    q_pos = jnp.broadcast_to(jnp.arange(29), (2, 29))
    for block, top_k in [(4, 2), (5, 3), (3, 1), (8, 9)]:
        full = np.asarray(moba_mask(q, k, block_size=block, top_k=top_k))
        cached = np.asarray(moba_cached_mask(q, keys, q_pos, kv_pos, block_size=block, top_k=top_k))
        np.testing.assert_array_equal(cached[..., :29], full)
        assert not cached[..., 29:].any()


def test_config_rejects_bad_moba_options():
    base = dict(num_heads=4, num_kv_heads=2, head_dim=16)
    with pytest.raises(ValueError):
        AttentionConfig(**base, moba_block_size=0)
    with pytest.raises(ValueError):
        AttentionConfig(**base, moba_block_size=4, moba_top_k=0)
    with pytest.raises(ValueError, match="sliding_window"):
        AttentionConfig(**base, moba_block_size=4, sliding_window=8)
    with pytest.raises(ValueError):
        moba_attention(*_inputs(), block_size=0, top_k=2)


@pytest.mark.parametrize("seq_len,block,top_k", [(37, 5, 3), (64, 8, 1), (100, 7, 20), (2048, 512, 3)])
def test_moba_core_flops_match_a_per_token_count(seq_len, block, top_k):
    width = 6 * 128
    keys = sum(t % block + 1 + min(top_k - 1, t // block) * block for t in range(seq_len))
    gates = sum(t // block for t in range(seq_len))
    expected = (4.0 * width * keys + 2.0 * width * gates) / seq_len
    assert _moba_core_flops(width, seq_len, block, top_k) == pytest.approx(expected, rel=1e-12)
    if top_k >= -(-seq_len // block):  # every block selected: the dense causal context
        assert keys == seq_len * (seq_len + 1) // 2


def test_moba_preset_costs_less_attention_than_its_dense_twin_at_long_context():
    cfg = get_preset("moba-hybrid-125m")
    dense = cfg.replace(attention=dataclasses.replace(cfg.attention, moba_block_size=None))
    sparse, full = estimate_flops_per_token(cfg, 32768), estimate_flops_per_token(dense, 32768)
    assert sparse.attention_projections == full.attention_projections
    assert sparse.attention_core < 0.2 * full.attention_core
