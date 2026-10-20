# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mLSTM op: scan reference and chunked form against an independent float64 NumPy oracle."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import chunked, reference as ref

TOL = dict(rtol=2e-4, atol=2e-4)


def _naive(q, k, v, f_pre, i_pre, *, scale, eps, state=None, resets=None, stabilised=True):
    """The xLSTM equations transcribed directly (arXiv 2405.04517 Eq. 19-27), float64 loops.

    ``stabilised=False`` runs the unstabilised true-state form ``q C / max(|q n|, 1)``
    (only meaningful with ``eps = 0``).
    """
    q, k, v, f_pre, i_pre = (np.asarray(x, np.float64) for x in (q, k, v, f_pre, i_pre))
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    o = np.zeros((batch, seq_len, heads, value_dim))
    c_out = np.zeros((batch, heads, key_dim, value_dim))
    n_out = np.zeros((batch, heads, key_dim))
    m_out = np.zeros((batch, heads))
    for b in range(batch):
        for h in range(heads):
            if state is None:
                c, n, m = np.zeros((key_dim, value_dim)), np.zeros(key_dim), -np.inf
            else:
                c = np.array(state[0][b, h], np.float64)
                n = np.array(state[1][b, h], np.float64)
                m = float(state[2][b, h])
                if not stabilised:
                    c, n, m = c * np.exp(m), n * np.exp(m), 0.0
            for t in range(seq_len):
                if resets is not None and resets[b, t]:
                    c, n, m = np.zeros((key_dim, value_dim)), np.zeros(key_dim), (-np.inf if stabilised else 0.0)
                log_f = -np.log1p(np.exp(-f_pre[b, t, h]))
                i_t = i_pre[b, t, h]
                qs = scale * q[b, t, h]
                kv = np.outer(k[b, t, h], v[b, t, h])
                if stabilised:
                    m_new = max(log_f + m, i_t)
                    fg, ig = np.exp(log_f + m - m_new), np.exp(i_t - m_new)
                    c, n, m = fg * c + ig * kv, fg * n + ig * k[b, t, h], m_new
                    o[b, t, h] = qs @ c / (max(abs(qs @ n), np.exp(-m)) + eps)
                else:
                    c = np.exp(log_f) * c + np.exp(i_t) * kv
                    n = np.exp(log_f) * n + np.exp(i_t) * k[b, t, h]
                    o[b, t, h] = qs @ c / max(abs(qs @ n), 1.0)
            c_out[b, h], n_out[b, h], m_out[b, h] = c, n, m
    return o, (c_out, n_out, m_out)


def _inputs(seed, B=2, T=45, H=2, K=8, V=6, gate_scale=3.0, gate_shift=0.0):
    ks = jax.random.split(jax.random.key(seed), 10)
    q = jax.random.normal(ks[0], (B, T, H, K))
    k = jax.random.normal(ks[1], (B, T, H, K))
    v = jax.random.normal(ks[2], (B, T, H, V))
    f_pre = gate_scale * jax.random.normal(ks[3], (B, T, H)) + 2.0
    i_pre = gate_scale * jax.random.normal(ks[4], (B, T, H)) + gate_shift
    state = ref.MLSTMState(
        0.3 * jax.random.normal(ks[5], (B, H, K, V)),
        0.3 * jax.random.normal(ks[6], (B, H, K)),
        jax.random.normal(ks[7], (B, H)),
    )
    seg = jnp.cumsum(jax.random.bernoulli(ks[8], 0.08, (B, T)).astype(jnp.int32), axis=1)
    cot = jax.random.normal(ks[9], (B, T, H, V))
    return q, k, v, f_pre, i_pre, state, seg, cot


def _resets(seg):
    return np.asarray(ref.segment_resets(seg))


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("with_segments", [False, True])
def test_reference_and_chunked_match_naive(with_state, with_segments):
    q, k, v, f_pre, i_pre, state, seg, _ = _inputs(0)
    state = state if with_state else None
    seg = seg if with_segments else None
    scale = q.shape[-1] ** -0.5
    want_o, want_s = _naive(q, k, v, f_pre, i_pre, scale=scale, eps=1e-6, state=state,
                            resets=None if seg is None else _resets(seg))
    g = jax.nn.log_sigmoid(f_pre)
    for fn, kw in ((ref.mlstm, {}), (chunked.mlstm, dict(chunk_size=16))):
        o, s = fn(q, k, v, g, i_pre, initial_state=state, output_final_state=True, segment_ids=seg, **kw)
        np.testing.assert_allclose(o, want_o, **TOL)
        for got, want in zip(s, want_s):
            np.testing.assert_allclose(got, want, **TOL)


def test_matches_unstabilised_true_form():
    """With eps = 0 the stabilised outputs equal q C / max(|q n|, 1) of the true state."""
    q, k, v, f_pre, i_pre, state, seg, _ = _inputs(1)
    scale = q.shape[-1] ** -0.5
    want, _ = _naive(q, k, v, f_pre, i_pre, scale=scale, eps=0.0, state=state, resets=_resets(seg), stabilised=False)
    g = jax.nn.log_sigmoid(f_pre)
    for fn, kw in ((ref.mlstm, {}), (chunked.mlstm, dict(chunk_size=8))):
        o, _ = fn(q, k, v, g, i_pre, eps=0.0, initial_state=state, segment_ids=seg, **kw)
        np.testing.assert_allclose(o, want, **TOL)


@pytest.mark.parametrize("delta", [-7.0, 9.0])
def test_invariant_to_the_stabiliser(delta):
    """Shifting m by delta with c, n rescaled by exp(-delta) is the same true state: same outputs."""
    q, k, v, f_pre, i_pre, state, seg, _ = _inputs(2)
    g = jax.nn.log_sigmoid(f_pre)
    shifted = ref.MLSTMState(state.c * jnp.exp(-delta), state.n * jnp.exp(-delta), state.m + delta)
    for fn, kw in ((ref.mlstm, {}), (chunked.mlstm, dict(chunk_size=16))):
        o1, _ = fn(q, k, v, g, i_pre, eps=0.0, initial_state=state, segment_ids=seg, **kw)
        o2, _ = fn(q, k, v, g, i_pre, eps=0.0, initial_state=shifted, segment_ids=seg, **kw)
        np.testing.assert_allclose(o1, o2, **TOL)


def test_large_gates_stay_finite_and_exact():
    """Input pre-activations far beyond exp() range of float32 (|i| up to ~90)."""
    q, k, v, f_pre, i_pre, state, seg, cot = _inputs(3, gate_scale=30.0, gate_shift=20.0)
    scale = q.shape[-1] ** -0.5
    want_o, _ = _naive(q, k, v, f_pre, i_pre, scale=scale, eps=1e-6, resets=_resets(seg))
    g = jax.nn.log_sigmoid(f_pre)
    o_ref, _ = ref.mlstm(q, k, v, g, i_pre, segment_ids=seg)
    o_chk, _ = chunked.mlstm(q, k, v, g, i_pre, segment_ids=seg, chunk_size=16)
    np.testing.assert_allclose(o_ref, want_o, **TOL)
    np.testing.assert_allclose(o_chk, want_o, **TOL)

    def loss(fn, *args, **kw):
        return jnp.sum(fn(*args, segment_ids=seg, **kw)[0] * cot)

    grads = jax.grad(lambda *a: loss(chunked.mlstm, *a, chunk_size=16), argnums=(0, 1, 2, 3, 4))(q, k, v, g, i_pre)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in grads)


@pytest.mark.parametrize("chunk_size", [8, 16])
def test_chunked_gradients_match_reference(chunk_size):
    q, k, v, f_pre, i_pre, state, seg, cot = _inputs(4, T=37)
    g = jax.nn.log_sigmoid(f_pre)

    def loss(fn, q, k, v, g, i, c0, n0, m0, **kw):
        o, s = fn(q, k, v, g, i, initial_state=ref.MLSTMState(c0, n0, m0), output_final_state=True,
                  segment_ids=seg, **kw)
        return jnp.sum(o * cot) + jnp.sum(jnp.exp(s.m)[..., None, None] * s.c)

    args = (q, k, v, g, i_pre, *state)
    want = jax.grad(lambda *a: loss(ref.mlstm, *a), argnums=tuple(range(8)))(*args)
    got = jax.grad(lambda *a: loss(chunked.mlstm, *a, chunk_size=chunk_size), argnums=tuple(range(8)))(*args)
    for name, a, b in zip(("q", "k", "v", "g", "i", "c0", "n0", "m0"), got, want):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=name)


def test_decode_parity():
    """Prefill a prefix, then decode token by token from the carried state: same as one pass."""
    q, k, v, f_pre, i_pre, state, _, _ = _inputs(5, T=24)
    g = jax.nn.log_sigmoid(f_pre)
    full, _ = ref.mlstm(q, k, v, g, i_pre, initial_state=state)
    split = 11
    o, s = chunked.mlstm(q[:, :split], k[:, :split], v[:, :split], g[:, :split], i_pre[:, :split],
                         initial_state=state, output_final_state=True, chunk_size=8)
    outs = [o]
    for t in range(split, q.shape[1]):
        sl = slice(t, t + 1)
        fn = ref.mlstm if t % 2 else chunked.mlstm
        o, s = fn(q[:, sl], k[:, sl], v[:, sl], g[:, sl], i_pre[:, sl], initial_state=s, output_final_state=True)
        outs.append(o)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)


def test_dispatcher():
    q, k, v, f_pre, i_pre, _, seg, _ = _inputs(6, T=20)
    g = jax.nn.log_sigmoid(f_pre)
    want, _ = ref.mlstm(q, k, v, g, i_pre, segment_ids=seg)
    assert ops.available("mlstm")[-2:] == ("chunked", "reference")
    for impl in ("auto", "chunked", "reference"):
        o, _ = ops.linear_attention("mlstm", q, k, v, g=g, i=i_pre, impl=impl, segment_ids=seg, chunk_size=8)
        np.testing.assert_allclose(o, want, **TOL)
    with pytest.raises(ValueError, match="cu_seqlens"):
        ops.linear_attention("mlstm", q[:1], k[:1], v[:1], g=g[:1], i=i_pre[:1], impl="reference",
                             cu_seqlens=jnp.array([0, 5, 20]))
    with pytest.raises(ValueError, match="gates"):
        ops.linear_attention("mlstm", q, k, v, g=g, impl="reference")


def test_bf16_inputs_scale_q_after_the_float32_cast():
    """bf16 inputs give the float32 function of their values, up to the bf16 output rounding.

    At ``K = 128`` the scale ``K ** -0.5`` is not a power of two; scaling ``q`` in bf16 rounded it
    before the float32 math, which on this input (rows where ``|q n|`` nearly cancels and the
    ``exp(-m)`` floor is tiny) moved both ops' outputs by 23 % of max |o|.
    """
    ks = jax.random.split(jax.random.key(3), 6)
    shape = (1, 1024, 2, 128)
    q, k, v = (jax.random.normal(key, shape).astype(jnp.bfloat16) for key in ks[:3])
    g = jax.nn.log_sigmoid(3.0 * jax.random.normal(ks[3], shape[:3]) + 2.0)
    i_pre = 3.0 * jax.random.normal(ks[4], shape[:3])
    want, _ = ref.mlstm(*(x.astype(jnp.float32) for x in (q, k, v)), g, i_pre)
    want = np.asarray(want)
    for name, fn in (("reference", ref.mlstm), ("chunked", chunked.mlstm)):
        got, _ = fn(q, k, v, g, i_pre)
        assert got.dtype == jnp.bfloat16
        err = float(np.abs(np.asarray(got, np.float32) - want).max() / np.abs(want).max())
        assert err <= 1e-2, f"{name}: max error {err:.3e} of max |o|"


def test_single_token_with_segment_ids():
    """``T = 1`` with ``segment_ids``: ``segment_resets`` keeps its ``[B, T]`` shape (no reset at ``t = 0``)."""
    q, k, v, f_pre, i_pre, _, _, _ = _inputs(7, T=1)
    g = jax.nn.log_sigmoid(f_pre)
    seg = jnp.zeros(q.shape[:2], jnp.int32)
    assert ref.segment_resets(seg).shape == q.shape[:2]
    want, _ = ref.mlstm(q, k, v, g, i_pre)
    for fn in (ref.mlstm, chunked.mlstm):
        got, _ = fn(q, k, v, g, i_pre, segment_ids=seg)
        np.testing.assert_allclose(got, want, **TOL)
