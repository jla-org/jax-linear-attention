# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO: the ``mamba3_mimo`` op (reference, chunked), the layer and its presets.

The oracles here are independent float64 NumPy loops written from state-spaces/mamba's
``mamba3_MIMO_step_ref`` (tests/ops/tilelang/test_mamba3_mimo.py), with the halves
rotation of the official MIMO kernels, never the code under test.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import test_models_lm as lm_tests
from jla import ops
from jla.layers.mamba3_mimo import Mamba3MIMO, mamba3_mimo
from jla.layers.ssm import Mamba3
from jla.models import build_model, get_preset
from jla.models.config import SSMConfig
from jla.ops import chunked, reference
from jla.ops.reference import Mamba3State
from jla.training.performance import estimate_flops_per_token


def _inputs(seed, batch=2, length=19, heads=3, rank=2, key_dim=8, value_dim=6):
    ks = jax.random.split(jax.random.key(seed), 9)
    q = jax.random.normal(ks[0], (batch, length, heads, rank, key_dim)) * 0.5
    k = jax.random.normal(ks[1], (batch, length, heads, rank, key_dim)) * 0.5
    v = jax.random.normal(ks[2], (batch, length, heads, rank, value_dim))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads)))
    beta = jax.random.uniform(ks[4], (batch, length, heads))
    gamma = jax.random.uniform(ks[5], (batch, length, heads))
    state = Mamba3State(
        jax.random.normal(ks[6], (batch, heads, key_dim, value_dim)),
        jax.random.normal(ks[7], (batch, heads, rank, key_dim)),
        jax.random.normal(ks[8], (batch, heads, rank, value_dim)),
    )
    return q, k, v, g, beta, gamma, state


def _segments(batch, length):
    ids = np.zeros((batch, length), np.int32)
    ids[0, 5:] = 1
    ids[0, 12:] = 2
    if batch > 1:
        ids[1, 17:] = 1
    return jnp.asarray(ids)


def _naive_op(q, k, v, g, beta, gamma, state=None, segment_ids=None):
    """float64 loop of the MIMO recurrence (official step reference, the core only)."""
    q, k, v, g, beta, gamma = (np.asarray(x, np.float64) for x in (q, k, v, g, beta, gamma))
    batch, length, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    out = np.zeros((batch, length, heads, rank, value_dim))
    finals = []
    for b in range(batch):
        if state is None:
            s = np.zeros((heads, key_dim, value_dim))
            kp, vp = np.zeros((heads, rank, key_dim)), np.zeros((heads, rank, value_dim))
        else:
            s, kp, vp = (np.asarray(x[b], np.float64) for x in state)
        for t in range(length):
            if segment_ids is not None and t > 0 and segment_ids[b, t] != segment_ids[b, t - 1]:
                s, kp, vp = np.zeros_like(s), np.zeros_like(kp), np.zeros_like(vp)
            for h in range(heads):
                prev = sum(np.outer(kp[h, r], vp[h, r]) for r in range(rank))
                cur = sum(np.outer(k[b, t, h, r], v[b, t, h, r]) for r in range(rank))
                s[h] = math.exp(g[b, t, h]) * s[h] + beta[b, t, h] * prev + gamma[b, t, h] * cur
                for r in range(rank):
                    out[b, t, h, r] = q[b, t, h, r] @ s[h]
            kp, vp = k[b, t].copy(), v[b, t].copy()
        finals.append((s, kp, vp))
    return out, tuple(np.stack(x) for x in zip(*finals))


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
def test_reference_matches_the_naive_recurrence(with_state, segmented):
    q, k, v, g, beta, gamma, state = _inputs(0)
    seg = _segments(*q.shape[:2]) if segmented else None
    state = state if with_state else None
    o, final = reference.mamba3_mimo(q, k, v, g, beta, gamma, initial_state=state, output_final_state=True,
                                     segment_ids=seg)
    want, want_final = _naive_op(q, k, v, g, beta, gamma, state, None if seg is None else np.asarray(seg))
    np.testing.assert_allclose(o, want, rtol=1e-5, atol=1e-5)
    for got, ref in zip(final, want_final):
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_rank_one_is_the_siso_recurrence(impl):
    q, k, v, g, beta, gamma, state = _inputs(1, rank=1)
    seg = _segments(*q.shape[:2])
    siso_state = Mamba3State(state.ssm, state.key[:, :, 0], state.value[:, :, 0])
    common = dict(output_final_state=True, segment_ids=seg, impl=impl, chunk_size=16)
    o1, f1 = ops.linear_attention("mamba3_mimo", q, k, v, g=g, beta=beta, gamma=gamma, initial_state=state,
                                  **common)
    o0, f0 = ops.linear_attention("mamba3", q[:, :, :, 0], k[:, :, :, 0], v[:, :, :, 0], g=g, beta=beta,
                                  gamma=gamma, initial_state=siso_state, **common)
    np.testing.assert_allclose(o1[:, :, :, 0], o0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f1.ssm, f0.ssm, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(f1.key[:, :, 0], f0.key)
    np.testing.assert_array_equal(f1.value[:, :, 0], f0.value)


@pytest.mark.parametrize(("length", "chunk", "rank"), [(19, 16, 2), (64, 16, 4), (40, 32, 3), (5, 16, 2)])
@pytest.mark.parametrize("segmented", [False, True])
def test_chunked_matches_reference(length, chunk, rank, segmented):
    q, k, v, g, beta, gamma, state = _inputs(2, length=length, rank=rank)
    seg = _segments(*q.shape[:2]) if segmented else None
    common = dict(initial_state=state, output_final_state=True, segment_ids=seg)
    o_ref, f_ref = reference.mamba3_mimo(q, k, v, g, beta, gamma, **common)
    o, f = chunked.mamba3_mimo(q, k, v, g, beta, gamma, chunk_size=chunk, **common)
    np.testing.assert_allclose(o, o_ref, rtol=2e-5, atol=2e-5)
    for got, ref in zip(f, f_ref):
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)


def test_chunked_gradients_match_reference():
    q, k, v, g, beta, gamma, state = _inputs(3, length=21, rank=2)
    seg = _segments(*q.shape[:2])
    w = jax.random.normal(jax.random.key(9), q.shape[:4] + (v.shape[-1],))

    def loss(fn, *args):
        o, f = fn(*args, initial_state=state, output_final_state=True, segment_ids=seg)
        return jnp.sum(o * w) + jnp.sum(f.ssm ** 2) * 0.1

    argnums = tuple(range(6))
    want = jax.grad(lambda *a: loss(reference.mamba3_mimo, *a), argnums)(q, k, v, g, beta, gamma)
    got = jax.grad(lambda *a: loss(lambda *x, **o: chunked.mamba3_mimo(*x, chunk_size=16, **o), *a), argnums)(
        q, k, v, g, beta, gamma)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


def _rotate_halves_np(x, angles):
    num, half = angles.shape[-1], x.shape[-1] // 2
    out = x.copy()
    c, s = np.cos(angles), np.sin(angles)
    out[..., :num] = x[..., :num] * c - x[..., half : half + num] * s
    out[..., half : half + num] = x[..., :num] * s + x[..., half : half + num] * c
    return out


def _naive_core(q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_x, mimo_z, mimo_o, d, z):
    """float64 loop of ``mamba3_MIMO_step_ref`` (no fused norm) with the halves rotation."""
    q, k, v, adt, dt, trap, angles, z = (np.asarray(x, np.float64) for x in (q, k, v, adt, dt, trap, angles, z))
    q_bias, k_bias, mimo_x, mimo_z, mimo_o, d = (np.asarray(x, np.float64)
                                                 for x in (q_bias, k_bias, mimo_x, mimo_z, mimo_o, d))
    batch, length, rank, groups, key_dim = q.shape
    heads, value_dim = v.shape[2], v.shape[3]
    rep = heads // groups
    out = np.zeros((batch, length, heads, value_dim))
    for b in range(batch):
        angle = np.zeros((heads, angles.shape[-1]))
        s = np.zeros((heads, key_dim, value_dim))
        kp, vp = np.zeros((heads, rank, key_dim)), np.zeros((heads, rank, value_dim))
        for t in range(length):
            angle = angle + np.tanh(angles[b, t]) * dt[b, t][:, None] * math.pi
            for h in range(heads):
                qh = _rotate_halves_np(q[b, t, :, h // rep] + q_bias[h], angle[h])
                kh = _rotate_halves_np(k[b, t, :, h // rep] + k_bias[h], angle[h])
                vh = v[b, t, h][None] * mimo_x[h]
                lam = 1.0 / (1.0 + math.exp(-trap[b, t, h]))
                alpha = math.exp(adt[b, t, h])
                beta, gamma = (1 - lam) * dt[b, t, h] * alpha, lam * dt[b, t, h]
                s[h] = (alpha * s[h] + beta * sum(np.outer(kp[h, r], vp[h, r]) for r in range(rank))
                        + gamma * sum(np.outer(kh[r], vh[r]) for r in range(rank)))
                y = qh @ s[h] + d[h] * vh
                zr = z[b, t, h][None] * mimo_z[h]
                y = y * zr / (1.0 + np.exp(-zr))
                out[b, t, h] = np.sum(y * mimo_o[h], axis=0)
                kp[h], vp[h] = kh, vh
    return out


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_core_matches_the_official_step_reference(impl):
    batch, length, rank, groups, heads, n, p, angles_n = 2, 20, 2, 1, 2, 8, 4, 2
    ks = jax.random.split(jax.random.key(4), 12)
    q = jax.random.normal(ks[0], (batch, length, rank, groups, n))
    k = jax.random.normal(ks[1], (batch, length, rank, groups, n))
    v = jax.random.normal(ks[2], (batch, length, heads, p))
    dt = jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads)))
    adt = -dt * (1.0 + jax.random.uniform(ks[4], (batch, length, heads)))
    trap = jax.random.normal(ks[5], (batch, length, heads))
    q_bias, k_bias = (1.0 + 0.1 * jax.random.normal(ks[i], (heads, rank, n)) for i in (6, 7))
    angles = jax.random.normal(ks[8], (batch, length, heads, angles_n))
    mimo_x, mimo_z, mimo_o = (jax.random.normal(ks[9], (3, heads, rank, p)) * 0.5 + 1.0)
    d = jnp.array([0.5, -1.0])
    z = jax.random.normal(ks[10], (batch, length, heads, p))
    y, _ = mamba3_mimo(q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_x, mimo_z=mimo_z, mimo_o=mimo_o,
                       d=d, z=z, impl=impl, chunk_size=16)
    want = _naive_core(q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_x, mimo_z, mimo_o, d, z)
    np.testing.assert_allclose(y, want, rtol=1e-4, atol=1e-4)


def _layer(outproj_norm=False, implementation="chunked", seed=0):
    cfg = SSMConfig(d_state=16, head_dim=16, n_groups=2, chunk_size=16, mimo_rank=2, outproj_norm=outproj_norm,
                    implementation=implementation)
    return Mamba3MIMO(64, cfg, key=jax.random.key(seed), std=0.2, norm_eps=1e-5, compute_dtype="float32")


@pytest.mark.parametrize("outproj_norm", [False, True])
def test_layer_prefill_then_decode_matches_the_full_pass(outproj_norm):
    layer = _layer(outproj_norm)
    x = jax.random.normal(jax.random.key(1), (2, 23, 64))
    full = layer(x)
    cache = layer.init_cache(2, jnp.float32)
    out, cache = layer.step(x[:, :7], cache)
    outs = [out]
    for t in range(7, 23):
        out, cache = layer.step(x[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=1e-4)


def test_layer_chunked_matches_reference_and_respects_documents():
    x = jax.random.normal(jax.random.key(2), (2, 30, 64))
    seg = jnp.asarray(np.repeat([[0] * 11 + [1] * 19], 2, axis=0))
    ref = _layer(implementation="reference")(x, segment_ids=seg)
    out = _layer(implementation="chunked")(x, segment_ids=seg)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    layer = _layer()
    second = layer(x[:, 11:])
    np.testing.assert_allclose(out[:, 11:], second, rtol=1e-4, atol=1e-4)


def test_rank_one_config_keeps_the_siso_layer_and_validates():
    tiny = get_preset("tiny-mamba3")
    assert tiny.ssm.mimo_rank == 1
    def mixers(model):
        kinds = (Mamba3, Mamba3MIMO)
        return [m for m in jax.tree.leaves(model, is_leaf=lambda x: isinstance(x, kinds)) if isinstance(m, kinds)]

    siso = mixers(build_model(tiny, key=jax.random.key(0)))
    assert siso and all(type(m) is Mamba3 for m in siso)
    mimo = mixers(build_model(get_preset("tiny-mamba3-mimo"), key=jax.random.key(0)))
    assert mimo and all(type(m) is Mamba3MIMO for m in mimo)
    with pytest.raises(ValueError):
        SSMConfig(mimo_rank=0)
    with pytest.raises(ValueError):
        build_model(get_preset("tiny-mamba2").replace(ssm=SSMConfig(d_state=16, head_dim=16, n_groups=2,
                                                                    chunk_size=16, mimo_rank=2)),
                    key=jax.random.key(0))
    q, _, v, g, beta, gamma, _ = _inputs(0)
    with pytest.raises(ValueError):  # SISO-shaped (4-D) inputs are refused, not broadcast
        reference.mamba3_mimo(q[:, :, :, 0], q[:, :, :, 0], v[:, :, :, 0], g, beta, gamma)


def test_flops_scale_with_the_rank():
    siso = estimate_flops_per_token(get_preset("mamba3-130m"), seq_len=2048)
    mimo = estimate_flops_per_token(get_preset("mamba3-mimo-130m"), seq_len=2048)
    assert mimo.linear_attention_core > siso.linear_attention_core
    assert mimo.linear_attention_projections > siso.linear_attention_projections


@pytest.mark.parametrize("scan", [True, False])
def test_tiny_preset_decode_matches_forward(scan):
    lm_tests.test_prefill_then_decode_matches_the_forward_pass("tiny-mamba3-mimo", scan)


@pytest.mark.parametrize("scan", [True, False])
def test_tiny_preset_logical_axes(scan):
    lm_tests.test_logical_axes_cover_every_parameter("tiny-mamba3-mimo", scan, True)


def test_tiny_preset_packed_documents_and_causality():
    lm_tests.test_packed_documents_match_separate_runs("tiny-mamba3-mimo")
    lm_tests.test_future_tokens_never_change_the_past("tiny-mamba3-mimo")
