# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-6 op (reference and chunked) and the RWKV-6 / HGRN2 layers against naive float64 loops."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.layers.rwkv6 import HGRN2, MIX_RANK, RWKV6
from jla.models import build_model, get_preset
from jla.models.config import LinearAttentionConfig
from jla.ops import rwkv6 as rwkv6_ops
from jla.ops.cross_entropy import linear_cross_entropy


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


# Naive float64 oracles -------------------------------------------------------------------
def naive_rwkv6(q, k, v, w, u, scale, h0=None, segment_ids=None):
    """FLA's fused_recurrent_rwkv6 loop: o_t = scale q_t (S + u k_t^T v_t); S <- exp(w_t) S + k_t^T v_t."""
    q, k, v, w, u = (np.asarray(x, np.float64) for x in (q, k, v, w, u))
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    o = np.zeros((batch, seq_len, heads, value_dim))
    final = np.zeros((batch, heads, key_dim, value_dim))
    for b in range(batch):
        for h in range(heads):
            s = np.zeros((key_dim, value_dim)) if h0 is None else np.asarray(h0, np.float64)[b, h].copy()
            for t in range(seq_len):
                if segment_ids is not None and t > 0 and segment_ids[b, t] != segment_ids[b, t - 1]:
                    s = np.zeros((key_dim, value_dim))
                kv = np.outer(k[b, t, h], v[b, t, h])
                o[b, t, h] = scale * q[b, t, h] @ (s + u[h][:, None] * kv)
                s = s * np.exp(w[b, t, h])[:, None] + kv
            final[b, h] = s
    return o, final


def naive_gla(q, k, v, g, scale):
    """GLA loop: S <- diag(exp(g_t)) S + k_t^T v_t; o_t = scale q_t S."""
    q, k, v, g = (np.asarray(x, np.float64) for x in (q, k, v, g))
    batch, seq_len, heads, key_dim = q.shape
    o = np.zeros((*q.shape[:3], v.shape[-1]))
    for b in range(batch):
        for h in range(heads):
            s = np.zeros((key_dim, v.shape[-1]))
            for t in range(seq_len):
                s = s * np.exp(g[b, t, h])[:, None] + np.outer(k[b, t, h], v[b, t, h])
                o[b, t, h] = scale * q[b, t, h] @ s
    return o


def _inputs(batch=2, seq_len=45, heads=2, key_dim=16, value_dim=8, seed=0, decay_spread=1.0):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = jax.random.normal(ks[0], (batch, seq_len, heads, key_dim))
    k = jax.random.normal(ks[1], (batch, seq_len, heads, key_dim))
    v = jax.random.normal(ks[2], (batch, seq_len, heads, value_dim))
    w = -jnp.exp(decay_spread * jax.random.normal(ks[3], (batch, seq_len, heads, key_dim)) - 1.0)
    u = jax.random.normal(ks[4], (heads, key_dim))
    h0 = jax.random.normal(ks[5], (batch, heads, key_dim, value_dim))
    return q, k, v, w, u, h0


def _segments(batch=2, seq_len=45):
    # Row 0: runs of 7, 20, 18 (crossing chunk boundaries); row 1: a reused id after a change.
    row0 = np.repeat([0, 1, 2], [7, 20, 18])
    row1 = np.repeat([3, 5, 3, 4], [16, 1, 12, 16])
    return jnp.asarray(np.stack([row0, row1])[:batch, :seq_len], jnp.int32)


# The op ------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_h0", [False, True])
@pytest.mark.parametrize("use_segments", [False, True])
def test_reference_matches_naive(use_h0, use_segments):
    q, k, v, w, u, h0 = _inputs()
    seg = _segments() if use_segments else None
    o, final = rwkv6_ops.reference(q, k, v, w, u, scale=0.5, initial_state=h0 if use_h0 else None,
                                   output_final_state=True, segment_ids=seg)
    o_ref, final_ref = naive_rwkv6(q, k, v, w, u, 0.5, h0 if use_h0 else None,
                                   None if seg is None else np.asarray(seg))
    np.testing.assert_allclose(o, o_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(final, final_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("use_segments", [False, True])
def test_chunked_matches_naive(chunk_size, use_segments):
    q, k, v, w, u, h0 = _inputs(seed=chunk_size)
    seg = _segments() if use_segments else None
    o, final = rwkv6_ops.chunked(q, k, v, w, u, initial_state=h0, output_final_state=True, segment_ids=seg,
                                 chunk_size=chunk_size)
    o_ref, final_ref = naive_rwkv6(q, k, v, w, u, 16**-0.5, h0, None if seg is None else np.asarray(seg))
    np.testing.assert_allclose(o, o_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(final, final_ref, rtol=1e-4, atol=1e-4)


def test_chunked_strong_decays_stay_finite_and_exact():
    # Log-decays down to about -e^9 per step: exp underflows to zero but never overflows.
    q, k, v, w, u, h0 = _inputs(seed=3, decay_spread=3.0)
    o, final = rwkv6_ops.chunked(q, k, v, w, u, initial_state=h0, output_final_state=True)
    o_ref, final_ref = naive_rwkv6(q, k, v, w, u, 16**-0.5, h0)
    assert np.isfinite(np.asarray(o)).all() and np.isfinite(np.asarray(final)).all()
    # In-chunk cumulative decays reach about -1e4 here, so float32 differences G_i - G_j lose
    # about 1e-3 of absolute precision (the cumulative-sum chunk form of GLA shares this).
    np.testing.assert_allclose(o, o_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(final, final_ref, rtol=2e-3, atol=2e-3)


def test_chunked_gradients_match_reference():
    q, k, v, w, u, h0 = _inputs(seq_len=37, seed=4)
    seg = _segments(2, 37)
    cot = jax.random.normal(jax.random.key(9), v.shape)
    cot_s = jax.random.normal(jax.random.key(10), h0.shape)

    def loss(fn, *args):
        o, final = fn(*args[:5], initial_state=args[5], output_final_state=True, segment_ids=seg)
        return jnp.sum(o * cot) + jnp.sum(final * cot_s)

    args = (q, k, v, w, u, h0)
    ref = jax.grad(lambda *a: loss(rwkv6_ops.reference, *a), argnums=tuple(range(6)))(*args)
    got = jax.grad(lambda *a: loss(lambda *x, **o: rwkv6_ops.chunked(*x, chunk_size=16, **o), *a),
                   argnums=tuple(range(6)))(*args)
    for name, a, b in zip("q k v w u h0".split(), got, ref):
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-4, err_msg=name)


def test_dispatcher_runs_rwkv6_and_fails_closed():
    q, k, v, w, u, _ = _inputs(seq_len=20)
    o_ref, _ = naive_rwkv6(q, k, v, w, u, 16**-0.5)
    for impl in ("auto", "reference", "chunked"):
        o, final = ops.linear_attention("rwkv6", q, k, v, g=w, u=u, impl=impl, chunk_size=16)
        np.testing.assert_allclose(o, o_ref, rtol=1e-4, atol=1e-4, err_msg=impl)
        assert final is None
    assert "reference" in ops.available("rwkv6") and "chunked" in ops.available("rwkv6")
    with pytest.raises(ValueError, match="missing"):
        ops.linear_attention("rwkv6", q, k, v, g=w)
    with pytest.raises(ValueError, match="unexpected"):
        ops.linear_attention("gla", q, k, v, g=w, u=u)
    with pytest.raises(NotImplementedError):
        ops.linear_attention("rwkv6", q[:1], k[:1], v[:1], g=w[:1], u=u, cu_seqlens=jnp.array([0, 9, 20]))
    with pytest.raises(ValueError, match="bonus"):
        rwkv6_ops.reference(q, k, v, w, u[:1])


# The layers ----------------------------------------------------------------------------
def _layer_cfg(kind: str, **changes) -> LinearAttentionConfig:
    base = dict(num_k_heads=2, num_v_heads=2, head_k_dim=16, head_v_dim=8, conv_size=0,
                output_gate=kind == "rwkv6", gate_low_rank_dim=8 if kind == "rwkv6" else None, chunk_size=16)
    return LinearAttentionConfig(**{**base, **changes})


def _randomize(module, names, key):
    """Replace zero/one-initialized arrays by random ones so the test exercises them."""
    for i, name in enumerate(names):
        leaf = getattr(module, name)
        value = 0.3 * jax.random.normal(jax.random.fold_in(key, i), leaf.shape, leaf.dtype)
        module = eqx.tree_at(lambda m, n=name: getattr(m, n), module, value)
    return module


def _np(x):
    return np.asarray(x, np.float64)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def test_rwkv6_layer_matches_naive():
    d = 32
    layer = RWKV6(d, _layer_cfg("rwkv6"), key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="float32")
    layer = _randomize(layer, ["x_mu", "x_bias", "bonus", "norm_weight", "norm_bias"], jax.random.key(1))
    x = jax.random.normal(jax.random.key(2), (2, 21, d))
    got = layer(x)

    xs = _np(x)
    batch, seq_len, _ = xs.shape
    shifted = np.concatenate([np.zeros_like(xs[:, :1]), xs[:, :-1]], axis=1)
    delta = shifted - xs
    z = np.tanh((xs + delta * _np(layer.x_mu)) @ _np(layer.x_down.weight)).reshape(batch, seq_len, 5, MIX_RANK)
    mix = np.einsum("btnr,nrd->btnd", z, _np(layer.x_up)) + _np(layer.x_bias)
    inp = [xs + delta * mix[:, :, i] for i in range(5)]
    r = (inp[0] @ _np(layer.r_proj.weight)).reshape(batch, seq_len, 2, 16)
    w = np.tanh(inp[1] @ _np(layer.w_down.weight)) @ _np(layer.w_up.weight) + _np(layer.w_up.bias)
    k = (inp[2] @ _np(layer.k_proj.weight)).reshape(batch, seq_len, 2, 16)
    v = (inp[3] @ _np(layer.v_proj.weight)).reshape(batch, seq_len, 2, 8)
    g = inp[4] @ _np(layer.g_proj.weight)
    o, _ = naive_rwkv6(r, k, v, -np.exp(w).reshape(batch, seq_len, 2, 16), _np(layer.bonus), 1.0)
    mean = o.mean(-1, keepdims=True)
    var = ((o - mean) ** 2).mean(-1, keepdims=True)
    o = ((o - mean) / np.sqrt(var + 1e-5)).reshape(batch, seq_len, -1)
    o = (o * _np(layer.norm_weight) + _np(layer.norm_bias)) * _silu(g)
    want = o @ _np(layer.o_proj.weight)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_hgrn2_layer_matches_naive():
    d = 32
    cfg = _layer_cfg("hgrn2", head_v_dim=16)
    layer = HGRN2(d, cfg, key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="float32")
    layer = eqx.tree_at(lambda m: m.g_norm.weight, layer, 1.0 + 0.3 * jax.random.normal(jax.random.key(1), (d,)))
    x = jax.random.normal(jax.random.key(2), (2, 21, d))
    got = layer(x)

    xs = _np(x)
    batch, seq_len, _ = xs.shape
    q = _silu(xs @ _np(layer.q_proj.weight)).reshape(batch, seq_len, 2, 16)
    f = 1.0 / (1.0 + np.exp(-(xs @ _np(layer.f_proj.weight)))).reshape(batch, seq_len, 2, 16)
    i = (xs @ _np(layer.i_proj.weight)).reshape(batch, seq_len, 2, 16)
    o = naive_gla(q, 1.0 - f, i, np.log(f), 16**-0.5).reshape(batch, seq_len, -1)
    o = o / np.sqrt((o**2).mean(-1, keepdims=True) + 1e-5) * _np(layer.g_norm.weight)
    want = o @ _np(layer.o_proj.weight)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("kind", ["rwkv6", "hgrn2"])
def test_layer_decode_matches_forward(kind):
    cls = RWKV6 if kind == "rwkv6" else HGRN2
    layer = cls(32, _layer_cfg(kind), key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="float32")
    x = jax.random.normal(jax.random.key(3), (2, 19, 32))
    full = layer(x)
    cache = layer.init_cache(2, jnp.float32)
    y, cache = layer.step(x[:, :7], cache)
    outs = [y]
    for t in range(7, 19):
        y, cache = layer.step(x[:, t : t + 1], cache)
        outs.append(y)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["rwkv6", "hgrn2"])
def test_layer_config_fails_loudly(kind):
    cls = RWKV6 if kind == "rwkv6" else HGRN2
    common = dict(key=jax.random.key(0), std=0.02, norm_eps=1e-5)
    with pytest.raises(ValueError, match="conv_size"):
        cls(32, _layer_cfg(kind, conv_size=4), **common)
    with pytest.raises(ValueError, match="output_gate"):
        cls(32, _layer_cfg(kind, output_gate=kind != "rwkv6"), **common)
    with pytest.raises(ValueError, match="num_v_heads"):
        cls(32, _layer_cfg(kind, num_v_heads=4), **common)


# Training on CPU -------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["tiny-rwkv6", "tiny-hgrn2"])
def test_train_steps_reduce_the_loss(name):
    model = build_model(get_preset(name).replace(compute_dtype="float32"), key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (4, 32), 0, 512)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((4, 1), -100)], axis=1).reshape(-1)

    def loss_fn(m):
        hidden = m(ids).hidden
        ce = linear_cross_entropy(hidden.reshape(-1, hidden.shape[-1]), m.lm_head_weight(), targets,
                                  compute_dtype=jnp.float32)
        return ce.loss_sum / ce.num_valid

    @eqx.filter_jit
    def step(m):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates = jax.tree.map(lambda g: -0.5 * g, grads)
        return eqx.apply_updates(m, updates), loss

    losses = []
    for _ in range(8):
        model, loss = step(model)
        losses.append(float(loss))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0] - 0.05, losses


def _naive_cmix(cm, x, segment_ids):
    """float64 numpy RWKV6FeedForward (FLA fla/layers/rwkv6.py) with resets at segment starts."""
    x = np.asarray(x, np.float64)
    seg = np.asarray(segment_ids)
    prev = np.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    prev[:, 1:][seg[:, 1:] != seg[:, :-1]] = 0.0
    delta = prev - x
    f = lambda a: np.asarray(a, np.float64)
    k = np.maximum((x + delta * f(cm.key_mu)) @ f(cm.key.weight), 0.0) ** 2
    r = (x + delta * f(cm.receptance_mu)) @ f(cm.receptance.weight)
    return 1.0 / (1.0 + np.exp(-r)) * (k @ f(cm.value.weight))


def test_channel_mix_matches_naive():
    from jla.layers.rwkv6 import RWKV6ChannelMix

    cm = RWKV6ChannelMix(32, 56, key=jax.random.key(0), std=0.2, compute_dtype="float32")
    cm = eqx.tree_at(lambda m: (m.key_mu, m.receptance_mu), cm,
                     (jax.random.uniform(jax.random.key(1), (32,)), jax.random.uniform(jax.random.key(2), (32,))))
    x = jax.random.normal(jax.random.key(3), (2, 13, 32))
    seg = jnp.asarray(np.repeat([[0, 1, 0, 2]], [4, 3, 5, 1], axis=1).repeat(2, 0), jnp.int32)
    np.testing.assert_allclose(cm(x, seg), _naive_cmix(cm, x, seg), rtol=1e-4, atol=1e-5)
    # decode: token by token with the carried last token equals the full pass without segments
    full, last, outs = cm(x), jnp.zeros((2, 1, 32)), []
    for t in range(13):
        outs.append(cm(x[:, t : t + 1], None, last))
        last = x[:, t : t + 1]
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-5, atol=1e-6)


def _perturbed(model, scale=0.05):
    """Every float parameter moved off its init (lerp mus, mix biases, bonus, norm biases included)."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


@pytest.mark.parametrize("name", ["tiny-rwkv6", "tiny-hgrn2"])
@pytest.mark.parametrize("scan", [True, False])
def test_randomized_model_decode_and_packing(name, scan):
    cfg = get_preset(name).replace(compute_dtype="float32", scan_layers=scan)
    if name == "tiny-rwkv6":
        assert cfg.resolved_mlp_types == ("rwkv_cmix",) * cfg.n_layers
    model = _perturbed(build_model(cfg, key=jax.random.key(5)))
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
    lengths = [5, 1, 8]
    seg = jnp.asarray(np.repeat([0, 1, 0], lengths)[None], jnp.int32)
    ppos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    packed = model(ids[:1], positions=ppos, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        alone = model(ids[:1, start : start + n]).hidden
        np.testing.assert_allclose(packed[:, start : start + n], alone, rtol=1e-4, atol=1e-4)
        start += n
