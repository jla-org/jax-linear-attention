# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PaTH attention: the chunked UT-transform op and the token-by-token op against an explicit oracle.

The oracle multiplies the Householder matrices ``H_s = I - beta_s w_s w_s^T`` out
in float64 numpy for every (query, key) pair, independent of both implementations.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.config import config_from_mapping
from jla.models import LOGICAL_AXIS_NAMES, build_model, get_preset, logical_axes, parameter_count
from jla.models.config import PathAttentionConfig
from jla.ops.path_attention import path_attention, path_attention_chunked, path_attention_reference
from jla.training.performance import estimate_flops_per_token
from jla.training.step import LossConfig, build_train_step, lm_loss


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _oracle(q, k, v, w, beta, g, segment_ids, scale):
    q, k, v, w, beta = (np.asarray(x, np.float64) for x in (q, k, v, w, beta))
    batch, length, heads, dim = q.shape
    groups = heads // k.shape[2]
    out = np.zeros((batch, length, heads, v.shape[-1]))
    for b in range(batch):
        cum = None if g is None else np.cumsum(np.asarray(g, np.float64)[b], axis=0)  # [T, H]
        seg = None if segment_ids is None else np.asarray(segment_ids)[b]
        for h in range(heads):
            hk = h // groups
            for i in range(length):
                logits = np.full(length, -np.inf)
                prod = np.eye(dim)  # H_i H_{i-1} ... H_{j+1}
                for j in range(i, -1, -1):
                    if seg is None or np.all(seg[j : i + 1] == seg[i]):
                        s = q[b, i, h] @ prod @ k[b, j, hk]
                        if cum is not None:
                            s += cum[i, h] - cum[j, h]
                        logits[j] = scale * s
                    prod = prod @ (np.eye(dim) - beta[b, j, hk] * np.outer(w[b, j, hk], w[b, j, hk]))
                p = np.exp(logits - logits.max())
                out[b, i, h] = (p / p.sum()) @ v[b, :, hk]
    return out


def _inputs(batch=2, length=37, heads=4, kv_heads=2, dim=8, dv=8, seed=0, gate=True, beta_scale=2.0):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = jax.random.normal(ks[0], (batch, length, heads, dim))
    k = jax.random.normal(ks[1], (batch, length, kv_heads, dim))
    v = jax.random.normal(ks[2], (batch, length, kv_heads, dv))
    w = jax.random.normal(ks[3], (batch, length, kv_heads, dim))
    w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)
    beta = beta_scale * jax.nn.sigmoid(jax.random.normal(ks[4], (batch, length, kv_heads)))
    g = jax.nn.log_sigmoid(2.0 + jax.random.normal(ks[5], (batch, length, heads))) if gate else None
    return q, k, v, w, beta, g


def _segments(batch, length):
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=3, replace=False))
        rows.append(np.searchsorted(cuts, np.arange(length), side="right"))
    return jnp.asarray(np.stack(rows), jnp.int32)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("chunk", [4, 16, 64])
def test_chunked_matches_the_householder_oracle(gate, packed, chunk):
    q, k, v, w, beta, g = _inputs(gate=gate)
    seg = _segments(2, 37) if packed else None
    out = path_attention_chunked(q, k, v, w, beta, g, segment_ids=seg, chunk_size=chunk)
    np.testing.assert_allclose(out, _oracle(q, k, v, w, beta, g, seg, 8 ** -0.5), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_reference_matches_the_householder_oracle(gate, packed):
    q, k, v, w, beta, g = _inputs(gate=gate, seed=1)
    seg = _segments(2, 37) if packed else None
    out = path_attention_reference(q, k, v, w, beta, g, segment_ids=seg, scale=0.3)
    np.testing.assert_allclose(out, _oracle(q, k, v, w, beta, g, seg, 0.3), rtol=2e-5, atol=2e-5)


def test_reused_segment_ids_start_new_documents():
    q, k, v, w, beta, g = _inputs(batch=1, length=20, seed=2)
    seg = jnp.asarray([[0] * 6 + [1] * 7 + [0] * 7], jnp.int32)  # id 0 again for the third document
    runs = jnp.asarray([[0] * 6 + [1] * 7 + [2] * 7], jnp.int32)
    out = path_attention_chunked(q, k, v, w, beta, g, segment_ids=seg, chunk_size=8)
    np.testing.assert_allclose(out, _oracle(q, k, v, w, beta, g, runs, 8 ** -0.5), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("impl", ["chunked", "reference"])
def test_zero_beta_is_plain_causal_attention(impl):
    q, k, v, w, beta, _ = _inputs(seed=4, gate=False)
    seg = _segments(2, 37)
    out = path_attention(q, k, v, w, jnp.zeros_like(beta), segment_ids=seg, chunk_size=16, implementation=impl)
    mask = (seg[:, :, None] == seg[:, None, :])[:, None]
    expected = jax.nn.dot_product_attention(q, k, v, mask=mask, is_causal=True, implementation="xla")
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)


def test_bfloat16_inputs_stay_close_to_float32():
    q, k, v, w, beta, g = _inputs(seed=5)
    ref = path_attention_chunked(q, k, v, w, beta, g, chunk_size=16)
    low = path_attention_chunked(*(x.astype(jnp.bfloat16) for x in (q, k, v)), w, beta, g, chunk_size=16)
    assert low.dtype == jnp.bfloat16
    np.testing.assert_allclose(low.astype(jnp.float32), ref, atol=0.06)


def test_gradients_are_finite_and_match_the_oracle_directional_derivative():
    q, k, v, w, beta, g = _inputs(batch=1, length=21, seed=6)
    xs = (q, k, v, w, beta, g)
    dirs = tuple(jax.random.normal(jax.random.key(10 + i), x.shape) for i, x in enumerate(xs))
    cot = jax.random.normal(jax.random.key(99), v.shape[:2] + (q.shape[2], v.shape[3]))

    def f(*args):
        return jnp.sum(path_attention_chunked(*args, chunk_size=8) * cot)

    grads = jax.grad(f, argnums=tuple(range(6)))(*xs)
    assert all(bool(jnp.all(jnp.isfinite(gr))) for gr in grads)
    analytic = sum(float(jnp.sum(gr * d)) for gr, d in zip(grads, dirs))
    eps = 1e-4

    def oracle_loss(sign):
        moved = [np.asarray(x, np.float64) + sign * eps * np.asarray(d, np.float64) for x, d in zip(xs, dirs)]
        return float(np.sum(_oracle(*moved[:5], moved[5], None, 8 ** -0.5) * np.asarray(cot, np.float64)))

    numeric = (oracle_loss(1.0) - oracle_loss(-1.0)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=2e-3, abs=2e-3)
    ref_grads = jax.grad(lambda *a: jnp.sum(path_attention_reference(*a) * cot), argnums=tuple(range(6)))(*xs)
    for a, b in zip(grads, ref_grads):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def test_bad_shapes_and_implementations_fail_loudly():
    q, k, v, w, beta, g = _inputs(length=8)
    with pytest.raises(ValueError, match="beta"):
        path_attention_chunked(q, k, v, w, beta[..., :1], g)
    with pytest.raises(ValueError, match="no 'mosaic_gpu'"):
        path_attention(q, k, v, w, beta, g, implementation="mosaic_gpu")
    with pytest.raises(ValueError, match="multiple"):
        PathAttentionConfig(num_heads=3, num_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="path_attention settings"):
        get_preset("tiny-path").replace(path_attention=None)


def _ids(batch=2, length=16, seed=1):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, 512)


def _f32(**changes):
    return get_preset("tiny-path").replace(compute_dtype="float32", **changes)


def test_tiny_path_forward_causality_and_packed_documents():
    model = build_model(_f32(), key=jax.random.key(0))
    ids = _ids()
    out = jax.jit(lambda m, i: m(i))(model, ids)
    assert out.hidden.shape == (2, 16, 64) and bool(jnp.all(jnp.isfinite(out.hidden)))
    ids2 = ids.at[:, 9:].set(_ids(2, 7, seed=7))
    np.testing.assert_allclose(model(ids2).hidden[:, :9], out.hidden[:, :9], rtol=1e-4, atol=1e-4)
    lengths = [7, 3, 6]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    one = _ids(1, 16, seed=2)
    packed = model(one, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        np.testing.assert_allclose(packed[:, start : start + n], model(one[:, start : start + n]).hidden,
                                   rtol=1e-4, atol=1e-4)
        start += n


def test_reused_segment_ids_start_new_documents_in_the_layer():
    # The layer called directly (the LM renumbers ids by runs first): the middle document (one token) is
    # shorter than the 4-tap w conv and the third reuses id 0, so a conv that compared ids alone would mix
    # the first document's w into the third one's (token 9 on).
    path = dataclasses.replace(get_preset("tiny-path").path_attention, w_conv_size=4)
    layer = build_model(_f32(path_attention=path, scan_layers=False), key=jax.random.key(0)).blocks[0].mixer
    x = jax.random.normal(jax.random.key(4), (1, 16, 64))
    seg = jnp.asarray([[0] * 7 + [1] + [0] * 8], jnp.int32)
    packed = layer(x, segment_ids=seg)
    for start, stop in [(0, 7), (7, 8), (8, 16)]:
        np.testing.assert_allclose(packed[:, start:stop], layer(x[:, start:stop]), rtol=1e-4, atol=1e-4)


def test_path_configs_round_trip_through_toml_and_mix_with_attention():
    import tomllib

    for model in [{"preset": "tiny-path"}, {"preset": "tiny-path", "layer_types": ["path", "attention"]}]:
        cfg = config_from_mapping({"model": model})
        again = config_from_mapping(tomllib.loads(cfg.to_toml()))
        assert again.model == cfg.model
    assert again.model.layer_types == ("path", "attention")
    hybrid = build_model(again.model.replace(compute_dtype="float32"), key=jax.random.key(0))
    assert bool(jnp.all(jnp.isfinite(hybrid(_ids()).hidden)))


def test_default_matmul_precision_stays_close_to_the_exact_reference():
    q, k, v, w, beta, g = _inputs(length=96, dim=16, dv=16, seed=5)
    exact = path_attention_reference(q, k, v, w, beta, g)  # "highest" from the autouse fixture
    with jax.default_matmul_precision("default"):  # what training uses (TF32 on GPU)
        fast = jax.jit(lambda *a: path_attention_chunked(*a, chunk_size=32))(q, k, v, w, beta, g)
    np.testing.assert_allclose(fast, exact, rtol=1e-2, atol=1e-2)


def test_layer_reference_and_chunked_agree_and_scan_matches_loop():
    chunked = build_model(_f32(), key=jax.random.key(1))
    cfg = _f32()
    ref_cfg = cfg.replace(path_attention=PathAttentionConfig(
        **{**cfg.path_attention.__dict__, "implementation": "reference"}))
    reference = build_model(ref_cfg, key=jax.random.key(1))
    looped = build_model(_f32(scan_layers=False), key=jax.random.key(1))
    ids = _ids(seed=3)
    np.testing.assert_allclose(chunked(ids).hidden, reference(ids).hidden, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(chunked(ids).hidden, looped(ids).hidden, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("scan", [True, False])
def test_logical_axes_cover_every_parameter(scan):
    model = build_model(_f32(scan_layers=scan), key=jax.random.key(0))
    def is_axes(x):
        return isinstance(x, tuple) and all(a is None or isinstance(a, str) for a in x)

    params = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    axes = [a for a in jax.tree.leaves(logical_axes(model), is_leaf=is_axes) if is_axes(a)]
    assert len(axes) == len(params) > 0
    assert [len(a) for a in axes] == [p.ndim for p in params]
    assert set().union(*axes) <= set(LOGICAL_AXIS_NAMES) | {None}
    assert any(a and a[0] == "layers" for a in axes) == scan


def _perturbed(model, scale=0.05):
    """Every float parameter moved off its init (gate biases, norm weights, conv taps included)."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


@pytest.mark.parametrize("scan", [True, False])
@pytest.mark.parametrize("forget_gate", [True, False])
def test_prefill_then_decode_matches_the_forward_pass(scan, forget_gate):
    """The decode cache (jla.layers.path_decode) runs the reference op's key-reflection recurrence."""
    cfg = _f32(scan_layers=scan)
    cfg = cfg.replace(path_attention=dataclasses.replace(cfg.path_attention, forget_gate=forget_gate))
    model = _perturbed(build_model(cfg, key=jax.random.key(5)))
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


def test_cpu_train_step_lowers_the_loss():
    model = build_model(get_preset("tiny-path"), key=jax.random.key(0))
    optimizer = optax.adamw(1e-2, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype="float32")), donate=False)
    tokens = np.random.default_rng(0).integers(0, 512, size=(1, 2, 33))
    batch = {"input_ids": jnp.asarray(tokens[..., :-1], jnp.int32),
             "labels": jnp.asarray(tokens[..., 1:], jnp.int32)}
    key = jax.random.key(1)
    losses = []
    for _ in range(4):
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        assert bool(metrics["step_applied"])
        losses.append(float(metrics["loss"]))
    assert np.all(np.isfinite(losses)) and losses[-1] < losses[0]


@pytest.mark.parametrize("name", ["tiny-path", "path-125m"])
def test_path_presets_flops_and_size(name):
    cfg = get_preset(name)
    counts = parameter_count(cfg)
    flops = estimate_flops_per_token(cfg, seq_len=2048)
    matmul = flops.attention_projections + flops.linear_attention_projections + flops.mlp
    assert matmul == pytest.approx(2 * (counts.active - counts.embedding), rel=0.02 if name == "path-125m" else 0.05)
    assert flops.attention_core > 0 and flops.linear_attention_core > 0
    if name == "path-125m":
        assert 100e6 <= counts.total <= 150e6
