# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""NSA and DSA ops and layers against independent token-by-token NumPy references.

The NSA reference follows FLA's naive math (fla/ops/nsa/naive.py, varlen documents);
the DSA reference follows fla/ops/dsa/naive.py and DeepSeek-V3.2's indexer.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models import build_model, get_preset
from jla.ops.sparse_attention import attach_indexer_grad, dsa, dsa_index_scores, dsa_select, nsa
from jla.training.performance import estimate_flops_per_token

TOL = dict(rtol=2e-4, atol=2e-5)


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # A fixture, not a module-level jax.config.update, so other test modules keep the default precision.
    with jax.default_matmul_precision("highest"):
        yield


def _docs(seg_row):
    """[(start, length)] runs of equal ids."""
    runs, start = [], 0
    for t in range(1, len(seg_row) + 1):
        if t == len(seg_row) or seg_row[t] != seg_row[t - 1]:
            runs.append((start, t - start))
            start = t
    return runs


def _softmax_attend(q, keys, values, scale):
    s = keys @ q * scale
    p = np.exp(s - s.max())
    return (p / p.sum()) @ values


def _np_nsa(q, k, v, g_cmp, g_slc, g_swa, seg, block_size, block_count, window, scale):
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    o = np.zeros((B, T, HQ, v.shape[-1]))
    blocks = {}
    for b in range(B):
        for start, n in _docs(seg[b]):
            kd, vd, qd = k[b, start:start + n], v[b, start:start + n], q[b, start:start + n]
            nb = -(-n // block_size)
            kc = np.stack([kd[c * block_size:(c + 1) * block_size].mean(0) for c in range(nb)])  # [nb, H, D]
            vc = np.stack([vd[c * block_size:(c + 1) * block_size].mean(0) for c in range(nb)])
            for t in range(n):
                vis = [c for c in range(nb) if (c + 1) * block_size - 1 <= t]
                own = t // block_size
                for h in range(H):
                    score = np.zeros(nb)
                    for g in range(G):
                        hq = h * G + g
                        if vis:
                            s = np.array([kc[c, h] @ qd[t, hq] * scale for c in vis])
                            p = np.exp(s - s.max())
                            p /= p.sum()
                            score[vis] += p / G
                    forced = {0: 2.0}
                    if own >= 1:
                        forced[own - 1] = 3.0
                    forced[own] = 4.0
                    for c, pr in forced.items():
                        score[c] = pr
                    allowed = sorted(set(vis) | set(forced), key=lambda c: (-score[c], c))
                    chosen = set(allowed[:block_count])
                    blocks[(b, start + t, h)] = {start + c * block_size for c in chosen}
                    sel = [u for u in range(t + 1) if u // block_size in chosen]
                    win = [u for u in range(t + 1) if t - u < window]
                    for g in range(G):
                        hq = h * G + g
                        out = g_slc[b, start + t, hq] * _softmax_attend(qd[t, hq], kd[sel, h], vd[sel, h], scale)
                        if vis:
                            out = out + g_cmp[b, start + t, hq] * _softmax_attend(qd[t, hq], kc[vis, h], vc[vis, h], scale)
                        if window > 0:
                            out = out + g_swa[b, start + t, hq] * _softmax_attend(qd[t, hq], kd[win, h], vd[win, h], scale)
                        o[b, start + t, hq] = out
    return o, blocks


def _np_dense(q, k, v, seg, scale, window=None):
    B, T, HQ, _ = q.shape
    G = HQ // k.shape[2]
    o = np.zeros((B, T, HQ, v.shape[-1]))
    for b in range(B):
        for start, n in _docs(seg[b]):
            for t in range(n):
                keys = [u for u in range(t + 1) if window is None or t - u < window]
                for hq in range(HQ):
                    kk = k[b, start:start + n, hq // G][keys]
                    vv = v[b, start:start + n, hq // G][keys]
                    o[b, start + t, hq] = _softmax_attend(q[b, start + t, hq], kk, vv, scale)
    return o


def _inputs(seed, B=2, T=40, HQ=4, H=2, D=8):
    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((B, T, h, D)).astype(np.float32) for h in (HQ, H, H))
    gates = [rng.uniform(0.1, 0.9, (B, T, HQ)).astype(np.float32) for _ in range(3)]
    seg = np.zeros((B, T), np.int32)
    seg[1, 13:] = 1  # second row: documents of 13, 17 and 10 tokens (not block aligned)
    seg[1, 30:] = 2
    return q, k, v, gates, seg


@pytest.mark.parametrize("window", [0, 5])
def test_nsa_matches_the_naive_reference_with_documents(window):
    q, k, v, (gc, gs, gw), seg = _inputs(0)
    bs, count, scale = 4, 4, 8 ** -0.5
    ref, ref_blocks = _np_nsa(q, k, v, gc, gs, gw, seg, bs, count, window, scale)
    out, starts = nsa(*map(jnp.asarray, (q, k, v, gc, gs, gw)), block_size=bs, block_count=count,
                      window_size=window, scale=scale, segment_ids=jnp.asarray(seg), return_block_starts=True)
    np.testing.assert_allclose(np.asarray(out), ref, **TOL)
    starts = np.asarray(starts)
    for (b, t, h), expected in ref_blocks.items():
        assert {int(s) for s in starts[b, t, h] if s >= 0} == expected, (b, t, h)


def test_nsa_selecting_every_block_is_causal_attention():
    q, k, v, _, seg = _inputs(1)
    ones, zeros = (jnp.full(q.shape[:3], x, jnp.float32) for x in (1.0, 0.0))
    out = nsa(*map(jnp.asarray, (q, k, v)), zeros, ones, None, block_size=4, block_count=64,
              segment_ids=jnp.asarray(seg))
    np.testing.assert_allclose(np.asarray(out), _np_dense(q, k, v, seg, 8 ** -0.5), **TOL)


def test_nsa_window_branch_is_sliding_window_attention():
    q, k, v, _, seg = _inputs(2)
    ones, zeros = (jnp.full(q.shape[:3], x, jnp.float32) for x in (1.0, 0.0))
    out = nsa(*map(jnp.asarray, (q, k, v)), zeros, zeros, ones, block_size=4, block_count=2, window_size=6,
              segment_ids=jnp.asarray(seg))
    np.testing.assert_allclose(np.asarray(out), _np_dense(q, k, v, seg, 8 ** -0.5, window=6), **TOL)


def _np_index_scores(qi, ki, wi, seg, scale):
    B, T = qi.shape[:2]
    out = np.full((B, T, T), -np.inf)
    for b in range(B):
        for start, n in _docs(seg[b]):
            for t in range(n):
                for s in range(t + 1):
                    sim = qi[b, start + t] @ ki[b, start + s] * scale  # [HI]
                    out[b, start + t, start + s] = float(np.sum(wi[b, start + t] * np.maximum(sim, 0.0)))
    return out


def test_dsa_indexer_scores_and_top_k():
    rng = np.random.default_rng(3)
    _, _, _, _, seg = _inputs(3)
    qi = rng.standard_normal((2, 40, 3, 8)).astype(np.float32)
    ki = rng.standard_normal((2, 40, 8)).astype(np.float32)
    wi = rng.uniform(0.1, 1.0, (2, 40, 3)).astype(np.float32)
    ref = _np_index_scores(qi, ki, wi, seg, 0.5)
    scores = dsa_index_scores(jnp.asarray(qi), jnp.asarray(ki), jnp.asarray(wi), scale=0.5,
                              segment_ids=jnp.asarray(seg))
    np.testing.assert_allclose(np.asarray(scores), ref, **TOL)
    idx = np.asarray(dsa_select(scores, 6))
    for b in range(2):
        for t in range(40):
            finite = np.sort(ref[b, t][np.isfinite(ref[b, t])])[::-1][:6]
            chosen = idx[b, t][idx[b, t] >= 0]
            assert len(chosen) == len(finite) == len(set(chosen.tolist()))
            np.testing.assert_allclose(np.sort(ref[b, t][chosen])[::-1], finite, **TOL)


def test_dsa_attends_exactly_the_selected_keys():
    q, k, v, _, seg = _inputs(4)
    rng = np.random.default_rng(4)
    idx = rng.integers(-1, 40, (2, 40, 5))
    out = np.asarray(dsa(*map(jnp.asarray, (q, k, v)), jnp.asarray(idx), segment_ids=jnp.asarray(seg)))
    G = q.shape[2] // k.shape[2]
    for b in range(2):
        doc = np.cumsum(np.r_[1, seg[b, 1:] != seg[b, :-1]])
        for t in range(40):
            keys = sorted({int(s) for s in idx[b, t] if 0 <= s <= t and doc[s] == doc[t]})
            for hq in range(q.shape[2]):
                if keys:
                    ref = _softmax_attend(q[b, t, hq], k[b, keys, hq // G], v[b, keys, hq // G], 8 ** -0.5)
                else:
                    ref = np.zeros(v.shape[-1])
                np.testing.assert_allclose(out[b, t, hq], ref, **TOL)


def test_dsa_top_k_covering_all_keys_is_causal_attention():
    q, k, v, _, seg = _inputs(5)
    rng = np.random.default_rng(5)
    scores = dsa_index_scores(jnp.asarray(rng.standard_normal((2, 40, 2, 8)), jnp.float32),
                              jnp.asarray(rng.standard_normal((2, 40, 8)), jnp.float32),
                              jnp.ones((2, 40, 2)), segment_ids=jnp.asarray(seg))
    out = dsa(*map(jnp.asarray, (q, k, v)), dsa_select(scores, 64), segment_ids=jnp.asarray(seg))
    np.testing.assert_allclose(np.asarray(out), _np_dense(q, k, v, seg, 8 ** -0.5), **TOL)


def test_indexer_grad_is_the_kl_gradient():
    rng = np.random.default_rng(6)
    scores = jnp.asarray(rng.standard_normal((2, 5, 5)), jnp.float32)
    selected = jnp.asarray(np.tril(rng.uniform(size=(2, 5, 5)) < 0.6) | np.eye(5, dtype=bool)[None], jnp.float32)
    raw = jnp.where(selected > 0, jnp.asarray(rng.uniform(0.1, 1.0, (2, 5, 5)), jnp.float32), 0.0)
    target = raw / raw.sum(-1, keepdims=True)

    def kl(s):
        logq = jax.nn.log_softmax(jnp.where(selected > 0, s, -jnp.inf), axis=-1)
        terms = jnp.where(selected > 0, target * (jnp.log(jnp.where(target > 0, target, 1.0)) - logq), 0.0)
        return 0.7 * jnp.sum(terms)  # summed over queries, like the train step's summed objective

    o = jnp.asarray(rng.standard_normal((2, 5, 3)), jnp.float32)
    g_o, g_s = jax.grad(lambda o, s: jnp.sum(attach_indexer_grad(o, s, target, selected, 0.7) * 2.0),
                        argnums=(0, 1))(o, scores)
    np.testing.assert_allclose(np.asarray(g_o), 2.0)
    np.testing.assert_allclose(np.asarray(g_s), np.asarray(jax.grad(kl)(scores)), rtol=1e-5, atol=1e-6)


def test_indexer_kl_has_the_mean_weight_under_the_train_step():
    """jla.training.step divides the summed objective's gradient by the token count."""
    from jla.layers.sparse_attention import DeepSeekSparseAttention

    cfg = get_preset("tiny-dsa")
    layer = DeepSeekSparseAttention(cfg.d_model, cfg.attention, cfg.sparse_attention, key=jax.random.key(0),
                                    std=0.5, norm_eps=cfg.norm_eps, compute_dtype="float32")
    x = jax.random.normal(jax.random.key(2), (2, 16, cfg.d_model), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(16), (2, 16))
    n = 2 * 16

    def objective(m):  # trainer convention: d(sum of token losses) / tokens
        return jnp.sum(m(x, pos) * jax.random.normal(jax.random.key(3), (2, 16, cfg.d_model)))

    g = jax.grad(objective)(layer)
    got = jax.tree.map(lambda a: a / n, g).index_q_proj

    def mean_kl(m):  # the intended indexer loss: coef * per-query mean KL
        scores = m.index_scores(x, pos)
        q, k, v = m.attn._project(x, pos)
        _, probs, selected = dsa(q, k, v, dsa_select(scores, cfg.sparse_attention.top_k),
                                 scale=m.attn.cfg.softmax_scale, return_probs=True)
        target = jax.lax.stop_gradient(probs.sum((1, 2)))
        target = target / target.sum(-1, keepdims=True)
        keep = selected
        logq = jax.nn.log_softmax(jnp.where(keep, scores, -jnp.inf), axis=-1)
        terms = jnp.where(keep & (target > 0), target * (jnp.log(jnp.where(target > 0, target, 1.0)) - logq), 0.0)
        return cfg.sparse_attention.indexer_loss_coef * jnp.sum(terms) / n

    want = jax.grad(mean_kl)(layer).index_q_proj
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-7)


def test_sparse_layer_types_load_from_config_mappings():
    from jla.config import _model_from_mapping

    cfg = _model_from_mapping({"preset": "tiny-nsa", "layer_types": ["nsa", "dsa"]})
    assert cfg.layer_types == ("nsa", "dsa")
    model = build_model(cfg.replace(compute_dtype="float32"), key=jax.random.key(0))
    assert np.isfinite(np.asarray(model(jnp.zeros((1, 8), jnp.int32)).hidden)).all()
    with pytest.raises(ValueError, match="plain JAX only"):
        build_model(cfg.replace(attention=dataclasses.replace(cfg.attention, implementation="cudnn")),
                    key=jax.random.key(0))


@pytest.mark.parametrize("name", ["tiny-nsa", "tiny-dsa"])
def test_sparse_models_train_and_keep_documents_apart(name):
    cfg = get_preset(name).replace(compute_dtype="float32")
    model = build_model(cfg, key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (2, 32), 0, cfg.vocab_size)
    seg = jnp.asarray(np.r_[np.zeros(11), np.ones(21)].astype(np.int32))[None].repeat(2, 0)
    pos = jnp.asarray(np.r_[np.arange(11), np.arange(21)].astype(np.int32))[None].repeat(2, 0)
    packed = model(ids, positions=pos, segment_ids=seg).hidden
    np.testing.assert_allclose(packed[:, :11], model(ids[:, :11]).hidden, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(packed[:, 11:], model(ids[:, 11:]).hidden, rtol=1e-4, atol=1e-5)

    def loss(m):
        logits = m.logits(m(ids).hidden).astype(jnp.float32)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), ids[:, 1:, None], -1))

    value, grads = jax.value_and_grad(loss)(model)
    leaves = jax.tree.leaves(grads)
    assert np.isfinite(float(value)) and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)

    def grads_of(tree, field):
        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
        return [np.asarray(x) for path, x in flat if field in jax.tree_util.keystr(path)]

    if name == "tiny-dsa":  # the indexer learns from the KL loss only, the rest from the LM loss
        assert sum(np.abs(g).sum() for g in grads_of(grads, "index_q_proj")) > 0
        frozen_cfg = cfg.replace(sparse_attention=dataclasses.replace(cfg.sparse_attention, indexer_loss_coef=0.0))
        g0 = jax.grad(loss)(build_model(frozen_cfg, key=jax.random.key(0)))
        assert sum(np.abs(g).sum() for g in grads_of(g0, "index_q_proj")) == 0.0
        for a, b in zip(grads_of(g0, "attn.q_proj"), grads_of(grads, "attn.q_proj"), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
    else:
        assert sum(np.abs(g).sum() for g in grads_of(grads, "branch_gate_proj")) > 0


@pytest.mark.parametrize("name", ["tiny-nsa", "tiny-dsa"])
def test_sparse_flops_count_the_extra_projections(name):
    cfg = get_preset(name)
    dense = cfg.replace(layer_types=None, sparse_attention=None)
    sparse_flops, dense_flops = (estimate_flops_per_token(c, seq_len=4096) for c in (cfg, dense))
    s = cfg.sparse_attention
    extra = 3 * cfg.attention.num_heads if name == "tiny-nsa" else (
        s.index_heads * s.index_head_dim + s.index_head_dim + s.index_heads)
    assert sparse_flops.attention_projections - dense_flops.attention_projections == pytest.approx(
        2.0 * cfg.d_model * extra * cfg.n_layers)
    assert sparse_flops.linear_attention_core == 0 and sparse_flops.attention_core > 0
    if name == "tiny-nsa":  # sparse attention is cheaper than dense at long context
        assert sparse_flops.attention_core < dense_flops.attention_core


@pytest.mark.parametrize("name", ["tiny-nsa", "tiny-dsa"])
@pytest.mark.parametrize("scan", [True, False])
def test_sparse_models_have_logical_axes_and_run_in_bf16(name, scan):
    import equinox as eqx

    from jla.models import LOGICAL_AXIS_NAMES, logical_axes

    model = build_model(get_preset(name).replace(scan_layers=scan), key=jax.random.key(0))
    params = eqx.filter(model, eqx.is_array)
    pairs = {}
    jax.tree_util.tree_map_with_path(
        lambda path, x, ax: pairs.__setitem__(jax.tree_util.keystr(path), (x.shape, ax)), params, logical_axes(model))
    assert len(pairs) == len(jax.tree.leaves(params))
    for path, (shape, ax) in pairs.items():
        assert isinstance(ax, tuple) and len(ax) == len(shape), path
        assert set(ax) <= set(LOGICAL_AXIS_NAMES) | {None}, path
    ids = jax.random.randint(jax.random.key(2), (2, 24), 0, model.config.vocab_size)
    hidden = jax.jit(lambda m, x: m(x).hidden)(model, ids)  # default bfloat16 compute
    assert hidden.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(hidden.astype(jnp.float32))))
    with pytest.raises(NotImplementedError):
        model.init_cache(2, 32)
