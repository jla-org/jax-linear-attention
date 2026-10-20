# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear: token-by-token reference vs dual form, segment resets, state continuation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops import ttt


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 comparisons run at ~1e-4: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


def _inputs(B=2, T=40, H=2, K=8, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)
    q = jax.random.normal(ks[0], (B, T, H, K)) * 0.5
    k = jax.random.normal(ks[1], (B, T, H, K)) * 0.5
    v = jax.random.normal(ks[2], (B, T, H, K))
    eta = jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H))) * 0.5
    opts = dict(
        ln_weight=1.0 + 0.1 * jax.random.normal(ks[4], (H, K)),
        ln_bias=0.1 * jax.random.normal(ks[5], (H, K)),
        w0=0.3 * jax.random.normal(ks[6], (H, K, K)),
        b0=0.1 * jax.random.normal(ks[7], (H, K)),
    )
    return q, k, v, eta, opts


def _naive(q, k, v, eta, *, ln_weight, ln_bias, w0, b0, mini_batch_size, eps=1e-6):
    """Literal mini-batch GD per (row, head) in NumPy-style Python loops (no segments)."""
    B, T, H, K = q.shape

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / jnp.sqrt(var + eps) * g + b

    out = np.zeros(q.shape, np.float32)
    for bi in range(B):
        for h in range(H):
            W, bb = w0[h], b0[h]
            for start in range(0, T, mini_batch_size):
                gws, gbs = [], []
                for i, t in enumerate(range(start, min(start + mini_batch_size, T))):
                    def loss(W_, b_):
                        z = k[bi, t, h] @ W_ + b_
                        return 0.5 * jnp.sum((ln(z, ln_weight[h], ln_bias[h]) - (v[bi, t, h] - k[bi, t, h])) ** 2)
                    dW, db = jax.grad(loss, argnums=(0, 1))(W, bb)
                    gws.append(eta[bi, t, h] * dW)
                    gbs.append(eta[bi, t, h] * db)
                    a = 1.0 / (i + 1)
                    Wt = W - a * sum(gws)
                    bt = bb - a * sum(gbs)
                    out[bi, t, h] = np.asarray(q[bi, t, h] + ln(q[bi, t, h] @ Wt + bt, ln_weight[h], ln_bias[h]))
                W, bb = Wt, bt
    return out


@pytest.mark.parametrize("T,bs", [(16, 4), (21, 8)])
def test_reference_matches_naive_loops(T, bs):
    q, k, v, eta, opts = _inputs(B=1, T=T, H=2, K=4)
    want = _naive(q, k, v, eta, mini_batch_size=bs, **opts)
    got, _ = ttt.ttt_linear_reference(q, k, v, eta, mini_batch_size=bs, **opts)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("T,bs", [(32, 8), (40, 16), (37, 16), (5, 16), (24, 1)])
def test_dual_matches_reference(T, bs):
    q, k, v, eta, opts = _inputs(T=T)
    ref, _ = ttt.ttt_linear_reference(q, k, v, eta, mini_batch_size=bs, **opts)
    dual, _ = ttt.ttt_linear_dual(q, k, v, eta, mini_batch_size=bs, **opts)
    np.testing.assert_allclose(np.asarray(dual), np.asarray(ref), rtol=2e-4, atol=2e-4)


def test_dual_matches_reference_with_segments_and_token_scale():
    q, k, v, eta, opts = _inputs(T=40)
    seg = jnp.array([[0] * 5 + [1] * 13 + [2] * 3 + [3] * 19, [7] * 16 + [8] * 24], jnp.int32)
    scale = jnp.linspace(1.0, 0.2, 8)
    kw = dict(mini_batch_size=8, token_scale=scale, segment_ids=seg, **opts)
    ref, _ = ttt.ttt_linear_reference(q, k, v, eta, **kw)
    dual, _ = ttt.ttt_linear_dual(q, k, v, eta, **kw)
    np.testing.assert_allclose(np.asarray(dual), np.asarray(ref), rtol=2e-4, atol=2e-4)


def test_aligned_packing_matches_separate_runs_and_unaligned_does_not_leak():
    q, k, v, eta, opts = _inputs(B=1, T=32)
    seg = jnp.array([[0] * 16 + [1] * 16], jnp.int32)
    packed, _ = ttt.ttt_linear_dual(q, k, v, eta, mini_batch_size=8, segment_ids=seg, **opts)
    first, _ = ttt.ttt_linear_dual(q[:, :16], k[:, :16], v[:, :16], eta[:, :16], mini_batch_size=8, **opts)
    second, _ = ttt.ttt_linear_dual(q[:, 16:], k[:, 16:], v[:, 16:], eta[:, 16:], mini_batch_size=8, **opts)
    np.testing.assert_allclose(np.asarray(packed), np.concatenate([first, second], axis=1), rtol=2e-4, atol=2e-4)
    # Unaligned boundary at 13: perturbing the first document leaves the second unchanged.
    seg = jnp.array([[0] * 13 + [1] * 19], jnp.int32)
    base, _ = ttt.ttt_linear_dual(q, k, v, eta, mini_batch_size=8, segment_ids=seg, **opts)
    v2 = v.at[:, :13].add(1.0)
    pert, _ = ttt.ttt_linear_dual(q, k, v2, eta, mini_batch_size=8, segment_ids=seg, **opts)
    np.testing.assert_allclose(np.asarray(pert[:, 13:]), np.asarray(base[:, 13:]), rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(pert[:, :13]), np.asarray(base[:, :13]))


@pytest.mark.parametrize("split", [8, 11, 1])
def test_reference_state_continuation(split):
    q, k, v, eta, opts = _inputs(T=30)
    full, final = ttt.ttt_linear_reference(q, k, v, eta, mini_batch_size=8, output_final_state=True, **opts)
    a, st = ttt.ttt_linear_reference(q[:, :split], k[:, :split], v[:, :split], eta[:, :split],
                                     mini_batch_size=8, output_final_state=True, **opts)
    b, st2 = ttt.ttt_linear_reference(q[:, split:], k[:, split:], v[:, split:], eta[:, split:],
                                      mini_batch_size=8, initial_state=st, output_final_state=True, **opts)
    np.testing.assert_allclose(np.concatenate([a, b], axis=1), np.asarray(full), rtol=1e-5, atol=1e-5)
    for x, y in zip(st2, final):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-5)


def test_dual_gradients_finite_and_match_reference():
    q, k, v, eta, opts = _inputs(T=24)

    def loss(fn, q, k, v, eta, opts):
        o, _ = fn(q, k, v, eta, mini_batch_size=8, **opts)
        return jnp.sum(o * jnp.cos(jnp.arange(o.size).reshape(o.shape) * 0.1))

    args = (q, k, v, eta, opts)
    g_dual = jax.grad(lambda *a: loss(ttt.ttt_linear_dual, *a), argnums=(0, 1, 2, 3, 4))(*args)
    g_ref = jax.grad(lambda *a: loss(ttt.ttt_linear_reference, *a), argnums=(0, 1, 2, 3, 4))(*args)
    for x, y in zip(jax.tree_util.tree_leaves(g_dual), jax.tree_util.tree_leaves(g_ref)):
        assert np.all(np.isfinite(np.asarray(x)))
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-3, atol=2e-3)


def test_dual_bfloat16_inputs_and_dispatch():
    q, k, v, eta, opts = _inputs(T=32)
    o32, _ = ttt.ttt_linear(q, k, v, eta, mini_batch_size=16, **opts)
    o16, _ = ttt.ttt_linear(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), eta,
                            mini_batch_size=16, **opts)
    assert o16.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(o16, np.float32), np.asarray(o32), rtol=0.1, atol=0.1)
    with pytest.raises(ValueError):
        ttt.ttt_linear_dual(q, k, v, eta, mini_batch_size=16, output_final_state=True, **opts)
    with pytest.raises(ValueError):
        ttt.ttt_linear(q, k, v, eta, impl="mosaic_gpu", **opts)
    _, st = ttt.ttt_linear(q, k, v, eta, mini_batch_size=16, output_final_state=True, **opts)
    assert isinstance(st, ttt.TTTState)
