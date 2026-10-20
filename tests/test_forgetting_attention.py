# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Forgetting Transformer (FoX) attention against independent float64 loop oracles.

The oracles sum the log forget gates over ``(j, i]`` explicitly for every query/key
pair (no cumsum difference), and rebuild the layer (KV shift, QK-norm, output norm
and gate) from its weights in NumPy.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.layers.attention import attention
from jla.layers.forgetting_attention import ForgettingAttention, forgetting_attention
from jla.models import build_model, get_preset, logical_axes
from jla.models.config import ForgettingAttentionConfig
from jla.ops.cross_entropy import linear_cross_entropy


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def naive_fox(q, k, v, log_f, segment_ids=None, window=None, scale=None):
    """O(T^2) loops in float64: q [B,T,H,D], k/v [B,T,Hkv,D], log_f [B,T,H]."""
    q, k, v, log_f = (np.asarray(a, np.float64) for a in (q, k, v, log_f))
    batch, length, heads, dim = q.shape
    groups = heads // k.shape[2]
    scale = dim ** -0.5 if scale is None else scale
    seg = np.zeros((batch, length), np.int64) if segment_ids is None else np.asarray(segment_ids)
    out = np.zeros(q.shape[:3] + (v.shape[-1],))
    for b in range(batch):
        for h in range(heads):
            hk = h // groups
            for i in range(length):
                js = [j for j in range(i + 1) if seg[b, j] == seg[b, i] and (window is None or i - j < window)]
                logits = np.array([scale * q[b, i, h] @ k[b, j, hk] + log_f[b, j + 1 : i + 1, h].sum()
                                   for j in js])
                p = np.exp(logits - logits.max())
                p /= p.sum()
                out[b, i, h] = sum(pj * v[b, j, hk] for pj, j in zip(p, js))
    return out


def _qkvf(batch=2, length=13, heads=4, kv_heads=2, dim=8, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (batch, length, heads, dim), jnp.float32)
    k = jax.random.normal(ks[1], (batch, length, kv_heads, dim), jnp.float32)
    v = jax.random.normal(ks[2], (batch, length, kv_heads, dim), jnp.float32)
    log_f = jax.nn.log_sigmoid(2.0 * jax.random.normal(ks[3], (batch, length, heads), jnp.float32) + 1.0)
    return q, k, v, log_f


SEGMENTS = np.array([[0] * 5 + [1] * 3 + [2] * 5, [0] * 13])


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("segments", [False, True])
@pytest.mark.parametrize("window", [None, 4])
def test_op_matches_naive_loops(kv_heads, segments, window):
    q, k, v, log_f = _qkvf(kv_heads=kv_heads)
    seg = jnp.asarray(SEGMENTS, jnp.int32) if segments else None
    got = jax.jit(lambda *a: forgetting_attention(*a, segment_ids=seg, sliding_window=window))(q, k, v, log_f)
    want = naive_fox(q, k, v, log_f, segment_ids=seg, window=window)
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-5, atol=2e-5)


def test_op_custom_scale_and_gradients_match_naive():
    q, k, v, log_f = _qkvf(batch=1, length=7, seed=3)
    got = forgetting_attention(q, k, v, log_f, scale=0.3)
    np.testing.assert_allclose(np.asarray(got), naive_fox(q, k, v, log_f, scale=0.3), rtol=2e-5, atol=2e-5)
    # Gradient w.r.t. the gates against central finite differences of the float64 oracle.
    w = np.asarray(jax.random.normal(jax.random.key(9), got.shape), np.float64)
    grad = jax.grad(lambda f: jnp.sum(forgetting_attention(q, k, v, f) * w))(log_f)
    base = np.asarray(log_f, np.float64)
    for idx in [(0, 3, 1), (0, 6, 2), (0, 1, 0)]:
        eps = 1e-5
        hi, lo = base.copy(), base.copy()
        hi[idx] += eps
        lo[idx] -= eps
        fd = (np.sum(naive_fox(q, k, v, hi) * w) - np.sum(naive_fox(q, k, v, lo) * w)) / (2 * eps)
        np.testing.assert_allclose(float(grad[idx]), fd, rtol=1e-3, atol=1e-4)


def test_open_forget_gate_is_plain_causal_attention():
    q, k, v, _ = _qkvf(seed=4)
    zero = jnp.zeros(q.shape[:3], jnp.float32)
    seg = jnp.asarray(SEGMENTS, jnp.int32)
    got = forgetting_attention(q, k, v, zero, segment_ids=seg, sliding_window=6)
    want = attention(q, k, v, segment_ids=seg, sliding_window=6, implementation="xla")
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_op_rejects_bad_shapes():
    q, k, v, log_f = _qkvf()
    with pytest.raises(ValueError):
        forgetting_attention(q, k, v, log_f[:, :, :2])
    with pytest.raises(ValueError):
        forgetting_attention(q, k[:, :, :1].repeat(3, axis=2), v[:, :, :1].repeat(3, axis=2), log_f)
    with pytest.raises(ValueError):
        ForgettingAttentionConfig(num_heads=4, num_kv_heads=3, head_dim=8)
    with pytest.raises(ValueError):
        ForgettingAttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8, implementation="mosaic_gpu")


# Layer --------------------------------------------------------------------------------------

def _layer(cfg: ForgettingAttentionConfig, d_model=32, seed=0) -> ForgettingAttention:
    layer = ForgettingAttention(d_model, cfg, key=jax.random.key(seed), std=0.02, norm_eps=1e-6,
                                compute_dtype="float32")
    params, static = eqx.partition(layer, eqx.is_array)
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed + 100), len(leaves))
    leaves = [x + 0.15 * jax.random.normal(kk, x.shape, x.dtype) for x, kk in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(tree, leaves), static)


def _rms(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def naive_layer(layer: ForgettingAttention, x, segment_ids=None):
    """The FoX block mixer rebuilt from its weights in float64 NumPy."""
    cfg = layer.cfg
    f64 = lambda a: np.asarray(a, np.float64)  # noqa: E731
    x = f64(x)
    batch, length, _ = x.shape
    hq, hk, dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    seg = np.zeros((batch, length), np.int64) if segment_ids is None else np.asarray(segment_ids)
    q = x @ f64(layer.q_proj.weight)
    k = x @ f64(layer.k_proj.weight)
    v = x @ f64(layer.v_proj.weight)
    log_f = _log_sigmoid(x @ f64(layer.f_proj.weight) + f64(layer.f_proj.bias))
    if cfg.kv_shift:
        def shift(y, w_alpha):
            alpha = _sigmoid(x @ f64(w_alpha))  # [B, T, Hkv]
            y_h = y.reshape(batch, length, hk, dim)
            out = np.empty_like(y_h)
            for b in range(batch):
                for t in range(length):
                    prev = y_h[b, t - 1] if t > 0 and seg[b, t - 1] == seg[b, t] else np.zeros((hk, dim))
                    out[b, t] = alpha[b, t][:, None] * prev + (1 - alpha[b, t][:, None]) * y_h[b, t]
            return out.reshape(batch, length, hk * dim)
        k, v = shift(k, layer.k_shift_proj.weight), shift(v, layer.v_shift_proj.weight)
    q = q.reshape(batch, length, hq, dim)
    k = k.reshape(batch, length, hk, dim)
    v = v.reshape(batch, length, hk, dim)
    if cfg.qk_norm:
        q, k = _rms(q, f64(layer.q_norm.weight)), _rms(k, f64(layer.k_norm.weight))
    o = naive_fox(q, k, v, log_f, segment_ids=seg, window=cfg.sliding_window, scale=cfg.softmax_scale)
    if cfg.output_norm:
        o = _rms(o, f64(layer.o_norm.weight))
    o = o.reshape(batch, length, hq * dim)
    if cfg.output_gate:
        o = o * _sigmoid(x @ f64(layer.gate_proj.weight))
    return o @ f64(layer.o_proj.weight)


PRO = ForgettingAttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8)
LAYER_CASES = {
    "pro": PRO,
    "plain": dataclasses.replace(PRO, qk_norm=False, output_gate=False, output_norm=False, kv_shift=False),
    "mha_window": dataclasses.replace(PRO, num_kv_heads=4, sliding_window=5),
}


@pytest.mark.parametrize("name", sorted(LAYER_CASES))
@pytest.mark.parametrize("segments", [False, True])
def test_layer_matches_naive_layer(name, segments):
    layer = _layer(LAYER_CASES[name])
    x = jax.random.normal(jax.random.key(1), (2, 13, 32), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(13, dtype=jnp.int32), (2, 13))
    seg = jnp.asarray(SEGMENTS, jnp.int32) if segments else None
    got = layer(x, pos, seg)
    np.testing.assert_allclose(np.asarray(got), naive_layer(layer, x, seg), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", sorted(LAYER_CASES))
def test_layer_prefill_then_decode_matches_full_sequence(name):
    layer = _layer(LAYER_CASES[name], seed=2)
    x = jax.random.normal(jax.random.key(3), (2, 11, 32), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(11, dtype=jnp.int32), (2, 11))
    full = layer(x, pos)
    cache = layer.init_cache(2, 16, jnp.float32)
    step = jax.jit(lambda m, h, p, c: m.step(h, p, c))
    out, cache = step(layer, x[:, :4], pos[:, :4], cache)
    outs = [out]
    for t in range(4, 11):
        out, cache = step(layer, x[:, t : t + 1], pos[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(np.asarray(jnp.concatenate(outs, axis=1)), np.asarray(full), rtol=1e-4, atol=1e-4)


# Model --------------------------------------------------------------------------------------

def _model(scan=True, **fox_changes):
    cfg = get_preset("tiny-fox").replace(compute_dtype="float32", scan_layers=scan)
    if fox_changes:
        cfg = cfg.replace(forgetting_attention=dataclasses.replace(cfg.forgetting_attention, **fox_changes))
    return build_model(cfg, key=jax.random.key(5))


def _ids(batch=2, length=14, seed=6):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, 512)


@pytest.mark.parametrize("scan", [True, False])
@pytest.mark.parametrize("changes", [{}, {"sliding_window": 5}, {"use_rope": True, "kv_shift": False}])
def test_model_prefill_then_decode_matches_the_forward_pass(scan, changes):
    model = _model(scan, **changes)
    ids = _ids()
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


def test_model_packed_documents_match_separate_runs_and_causality():
    model = _model()
    lengths = [7, 3, 6]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = _ids(1, 16, seed=2)
    packed = model(ids, positions=pos, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        alone = model(ids[:, start : start + n]).hidden
        np.testing.assert_allclose(packed[:, start : start + n], alone, rtol=1e-4, atol=1e-4)
        start += n
    ids2 = ids.at[:, 9:].set(_ids(1, 7, seed=7))
    np.testing.assert_allclose(model(ids2).hidden[:, :9], model(ids).hidden[:, :9], rtol=1e-4, atol=1e-4)


def test_logical_axes_cover_every_fox_parameter():
    for scan in (True, False):
        model = build_model(get_preset("tiny-fox").replace(scan_layers=scan), key=jax.random.key(0))
        params = eqx.filter(model, eqx.is_array)
        pairs = {}
        jax.tree_util.tree_map_with_path(
            lambda path, x, ax: pairs.__setitem__(jax.tree_util.keystr(path), (x.shape, ax)), params,
            logical_axes(model),
        )
        assert len(pairs) == len(jax.tree.leaves(params))
        assert any("f_proj" in path for path in pairs) and any("k_shift_proj" in path for path in pairs)
        for path, (shape, ax) in pairs.items():
            assert isinstance(ax, tuple) and len(ax) == len(shape), path


def test_cpu_train_steps_reduce_the_loss():
    model = build_model(get_preset("tiny-fox"), key=jax.random.key(0))
    ids = _ids(4, 32, seed=11)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((4, 1), -100)], axis=1)

    def loss_fn(m):
        out = m(ids)
        ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), m.lm_head_weight(),
                                  targets.reshape(-1), compute_dtype=jnp.float32)
        return ce.loss_sum / ce.num_valid

    opt = optax.adam(3e-3)
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(m, s):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, s = opt.update(grads, s, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), s, loss, grads

    losses = []
    for i in range(8):
        model, state, loss, grads = train_step(model, state)
        losses.append(float(loss))
        if i == 0:
            leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
            assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
            f_grad = grads.blocks[0].mixer.f_proj.weight
            assert float(jnp.abs(f_grad).max()) > 0  # the forget gate is trained
    assert np.isfinite(losses).all() and losses[-1] < losses[0] - 0.1, losses
