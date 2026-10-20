# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/log_linear_attn/naive.py (the oracle's level masks)
"""Log-linear attention: reference and chunked ops against an independent dense oracle.

The oracle builds the dense ``[T, T]`` matrix ``H[t, s] = exp(G_t - G_s) *
sum_l lambda_t[l] * mask_l(p_t, p_s)`` in float64 numpy from FLA's bit
conditions (``construct_level_mask``), with positions ``p`` restarting at every
segment, and returns ``o_t = scale * sum_s H[t, s] (q_t . k_s) v_s``.

Identity tested (reduction to the plain linear version): with
``lambda_t[l] = 1`` for every ``t, l`` the Fenwick buckets of ``t`` partition
``{s <= t}`` in its segment, so ``o`` equals ``simple_gla(q, k, v, g)`` and the
levels of the final Fenwick state sum to the ``simple_gla`` final state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import log_linear

jax.config.update("jax_enable_x64", False)


def _level_mask(level: int, p_t: int, p_s: int) -> bool:
    if level == 0:
        return p_t == p_s
    half = 1 << (level - 1)
    base = p_t - (p_t % half)
    return (p_t % (1 << level) >= half) and (p_s + half >= base) and (p_s < base)


def _oracle(q, k, v, g, lam, scale, segment_ids=None):
    q, k, v, g, lam = (np.asarray(x, np.float64) for x in (q, k, v, g, lam))
    batch, seq_len, heads, _ = q.shape
    n_levels = lam.shape[-1]
    o = np.zeros(v.shape, np.float64)
    for b in range(batch):
        ids = np.zeros(seq_len, np.int64) if segment_ids is None else np.asarray(segment_ids[b])
        run = np.concatenate([[0], np.cumsum(ids[1:] != ids[:-1])])
        start = np.zeros(seq_len, np.int64)
        for t in range(1, seq_len):
            start[t] = start[t - 1] if run[t] == run[t - 1] else t
        pos = np.arange(seq_len) - start
        for h in range(heads):
            cum = np.cumsum(g[b, :, h])
            for t in range(seq_len):
                for s in range(t + 1):
                    if run[s] != run[t]:
                        continue
                    weight = sum(lam[b, t, h, l] for l in range(n_levels) if _level_mask(l, pos[t], pos[s]))
                    o[b, t, h] += scale * weight * np.exp(cum[t] - cum[s]) * (q[b, t, h] @ k[b, s, h]) * v[b, s, h]
    return o


def _inputs(seed, batch=2, seq_len=13, heads=2, dk=4, dv=3, extra_levels=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(keys[0], (batch, seq_len, heads, dk))
    k = jax.random.normal(keys[1], (batch, seq_len, heads, dk))
    v = jax.random.normal(keys[2], (batch, seq_len, heads, dv))
    g = -jax.nn.softplus(jax.random.normal(keys[3], (batch, seq_len, heads)))
    n_levels = log_linear.num_levels(seq_len) + extra_levels
    lam = jax.random.uniform(keys[4], (batch, seq_len, heads, n_levels), minval=0.1, maxval=2.0)
    return q, k, v, g, lam


def _segments(batch, seq_len):
    ids = np.zeros((batch, seq_len), np.int32)
    ids[0, 5:] = 1
    ids[0, 9:] = 2
    if batch > 1:
        ids[1, 3:] = 7
        ids[1, 4:] = 3
    return jnp.asarray(ids)


def test_num_levels():
    assert [log_linear.num_levels(t) for t in (1, 2, 3, 4, 5, 8, 9, 2048)] == [1, 2, 3, 3, 4, 4, 5, 12]


@pytest.mark.parametrize("use_segments", [False, True])
@pytest.mark.parametrize("seq_len", [1, 8, 13])
def test_reference_matches_oracle(seq_len, use_segments):
    q, k, v, g, lam = _inputs(0, seq_len=seq_len)
    seg = _segments(2, seq_len) if use_segments and seq_len > 9 else None
    o, _ = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, scale=0.5,
                                segment_ids=seg, impl="reference")
    np.testing.assert_allclose(np.asarray(o), _oracle(q, k, v, g, lam, 0.5, seg), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_segments", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 8, 16])
def test_chunked_matches_oracle(chunk_size, use_segments):
    q, k, v, g, lam = _inputs(1, seq_len=13, extra_levels=1)
    seg = _segments(2, 13) if use_segments else None
    o, _ = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, scale=0.5,
                                segment_ids=seg, impl="chunked", chunk_size=chunk_size)
    np.testing.assert_allclose(np.asarray(o), _oracle(q, k, v, g, lam, 0.5, seg), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_segments", [False, True])
@pytest.mark.parametrize("chunk_size", [4, 8, 64])
def test_chunked_matches_reference_long(chunk_size, use_segments):
    seq_len = 70
    q, k, v, g, lam = _inputs(2, batch=2, seq_len=seq_len, heads=3, dk=8, dv=4)
    seg = None
    if use_segments:
        ids = np.zeros((2, seq_len), np.int32)
        ids[0, 21:] = 1
        ids[0, 50:] = 2
        ids[1, 33:] = 5
        seg = jnp.asarray(ids)
    kw = dict(g=g, level_scales=lam, segment_ids=seg, output_final_state=True)
    o_ref, s_ref = ops.linear_attention("log_linear_simple_gla", q, k, v, impl="reference", **kw)
    o_chk, s_chk = ops.linear_attention("log_linear_simple_gla", q, k, v, impl="chunked", chunk_size=chunk_size, **kw)
    np.testing.assert_allclose(np.asarray(o_chk), np.asarray(o_ref), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(s_chk), np.asarray(s_ref), rtol=1e-4, atol=1e-4)
    assert s_ref.shape == (2, 3, lam.shape[-1], 8, 4)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("use_segments", [False, True])
def test_unit_lambda_reduces_to_simple_gla(impl, use_segments):
    seq_len = 37
    q, k, v, g, lam = _inputs(3, seq_len=seq_len, heads=2, dk=8, dv=5)
    seg = None
    if use_segments:
        ids = np.zeros((2, seq_len), np.int32)
        ids[0, 11:] = 1
        ids[1, 30:] = 4
        seg = jnp.asarray(ids)
    ones = jnp.ones_like(lam)
    opts = {} if impl == "reference" else {"chunk_size": 8}
    o, state = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=ones, segment_ids=seg,
                                    output_final_state=True, impl=impl, **opts)
    o_lin, s_lin = ops.reference.simple_gla(q, k, v, g, segment_ids=seg, output_final_state=True)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_lin), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(state.sum(axis=2)), np.asarray(s_lin), rtol=1e-4, atol=1e-4)


def test_level_zero_only_is_diagonal():
    q, k, v, g, lam = _inputs(4, seq_len=12)
    only0 = jnp.zeros_like(lam).at[..., 0].set(1.0)
    o, _ = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=only0, scale=1.0,
                                impl="chunked", chunk_size=4)
    expected = jnp.einsum("bthk,bthk->bth", q, k)[..., None] * v
    np.testing.assert_allclose(np.asarray(o), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_gradients_match_reference():
    q, k, v, g, lam = _inputs(5, seq_len=19, dk=4, dv=3)
    seg = jnp.asarray(np.repeat(np.array([[0] * 7 + [1] * 12]), 2, axis=0).astype(np.int32))

    def loss(impl, *args):
        opts = {"chunk_size": 4} if impl == "chunked" else {}
        o, _ = ops.linear_attention("log_linear_simple_gla", *args[:3], g=args[3], level_scales=args[4],
                                    segment_ids=seg, impl=impl, **opts)
        return jnp.sum(o * jnp.cos(o))

    grads_ref = jax.grad(lambda *a: loss("reference", *a), argnums=(0, 1, 2, 3, 4))(q, k, v, g, lam)
    grads_chk = jax.grad(lambda *a: loss("chunked", *a), argnums=(0, 1, 2, 3, 4))(q, k, v, g, lam)
    for a, b in zip(grads_chk, grads_ref):
        assert np.all(np.isfinite(np.asarray(a)))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-4)


def test_auto_dispatch_and_bf16():
    q, k, v, g, lam = _inputs(6, seq_len=16)
    o_auto, _ = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, chunk_size=8)
    o_ref, _ = ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, impl="reference")
    np.testing.assert_allclose(np.asarray(o_auto), np.asarray(o_ref), rtol=1e-4, atol=1e-4)
    o16, _ = ops.linear_attention("log_linear_simple_gla", q.astype(jnp.bfloat16), k.astype(jnp.bfloat16),
                                  v.astype(jnp.bfloat16), g=g, level_scales=lam, chunk_size=8)
    assert o16.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(o16, np.float32), np.asarray(o_ref), rtol=5e-2, atol=5e-2)


def test_rejects_unsupported_calls():
    q, k, v, g, lam = _inputs(7, seq_len=9)
    for impl in ("reference", "chunked"):
        with pytest.raises(ValueError, match="levels"):
            ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam[..., :3], impl=impl)
        with pytest.raises(ValueError, match="initial_state"):
            ops.linear_attention("log_linear_simple_gla", q, k, v, g=g, level_scales=lam, impl=impl,
                                 initial_state=jnp.zeros((2, 2, 4, 3)))
    with pytest.raises(ValueError, match="cu_seqlens"):
        ops.linear_attention("log_linear_simple_gla", q[:1], k[:1], v[:1], g=g[:1], level_scales=lam[:1],
                             impl="reference", cu_seqlens=jnp.array([0, 4, 9]))
    with pytest.raises(ValueError, match="level_scales"):
        ops.linear_attention("simple_gla", q, k, v, g=g, level_scales=lam)
