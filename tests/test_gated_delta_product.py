# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaProduct: reference, chunked and Mosaic GPU paths.

Every path is checked against an independent float64 NumPy loop of the
recurrence (FLA ``naive_recurrent_gated_delta_product``: the decay once per
token, then ``n_h`` delta-rule steps, then the read-out), with packed documents
(``segment_ids``), an initial state and the final state. ``n_h = 1`` must equal
Gated DeltaNet exactly. The Mosaic GPU tests (forward and gradients against
autodiff of the reference) need an H100.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import chunked, reference as ref

F32, BF16 = jnp.float32, jnp.bfloat16


def _naive(q, k, v, g, beta, n, scale, h0=None, segment_ids=None):
    """float64 token loop: S = exp(g_t) S; n delta steps; o_t = scale q_t S."""
    q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    o = np.zeros((batch, seq_len, heads, value_dim))
    final = np.zeros((batch, heads, key_dim, value_dim))
    for b in range(batch):
        state = np.zeros((heads, key_dim, value_dim)) if h0 is None else np.asarray(h0[b], np.float64).copy()
        for t in range(seq_len):
            if segment_ids is not None and t > 0 and segment_ids[b, t] != segment_ids[b, t - 1]:
                state = np.zeros((heads, key_dim, value_dim))
            state = state * np.exp(g[b, t])[:, None, None]
            for j in range(n):
                kk, vv, bb = k[b, t * n + j], v[b, t * n + j], beta[b, t * n + j]
                for h in range(heads):
                    state[h] = state[h] + bb[h] * np.outer(kk[h], vv[h] - kk[h] @ state[h])
            for h in range(heads):
                o[b, t, h] = scale * q[b, t, h] @ state[h]
        final[b] = state
    return o, final


def _inputs(seed, n, B=2, T=13, H=2, K=8, V=6, dtype=F32):
    ks = jax.random.split(jax.random.key(seed), 8)
    return dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(dtype),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T * n, H, K))).astype(dtype),
        v=jax.random.normal(ks[2], (B, T * n, H, V)).astype(dtype),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 1.0),
        beta=2.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (B, T * n, H))),  # allow_neg_eigval range
        h0=0.1 * jax.random.normal(ks[5], (B, H, K, V)),
        do=jax.random.normal(ks[6], (B, T, H, V)),
        dht=jax.random.normal(ks[7], (B, H, K, V)),
    )


def _segments(B, T):
    rows = []
    for b in range(B):
        cuts = {0, T // 3 + b, T // 3 + b + 1, (2 * T) // 3}
        rows.append(np.cumsum([1 if t in cuts else 0 for t in range(T)]))
    return jnp.asarray(np.stack(rows), jnp.int32)


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_matches_naive(impl, n, packed):
    x = _inputs(n, n)
    seg = _segments(2, 13) if packed else None
    extra = {"chunk_size": 8} if impl == "chunked" else {}
    o, s = ops.linear_attention(
        "gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"], impl=impl,
        initial_state=x["h0"], output_final_state=True, segment_ids=seg, num_householder=n, **extra,
    )
    o_want, s_want = _naive(x["q"], x["k"], x["v"], x["g"], x["beta"], n, 8**-0.5, x["h0"],
                            None if seg is None else np.asarray(seg))
    _assert_close(o, o_want, 1e-5, "o")
    _assert_close(s, s_want, 1e-5, "final state")


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_infers_num_householder_and_l2norm(impl):
    x = _inputs(7, 2)
    q, k = x["q"] * 3.0, x["k"] * 2.0
    o, _ = ops.linear_attention("gated_deltaproduct", q, k, x["v"], g=x["g"], beta=x["beta"], impl=impl,
                                use_qk_l2norm=True, scale=0.5)
    o_want, _ = _naive(ref.l2norm(q), ref.l2norm(k), x["v"], x["g"], x["beta"], 2, 0.5)
    _assert_close(o, o_want, 1e-5, "o")


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("packed", [False, True])
def test_one_householder_equals_gated_deltanet(impl, packed):
    x = _inputs(3, 1, T=37)
    seg = _segments(2, 37) if packed else None
    kw = dict(initial_state=x["h0"], output_final_state=True, segment_ids=seg, use_qk_l2norm=True, impl=impl)
    o, s = ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"],
                                num_householder=1, **kw)
    o_gdn, s_gdn = ops.linear_attention("gated_deltanet", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"], **kw)
    np.testing.assert_array_equal(np.asarray(o), np.asarray(o_gdn))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(s_gdn))


@pytest.mark.parametrize("n", [2, 3])
def test_chunked_gradients_match_reference(n):
    x = _inputs(11 + n, n, T=21)
    seg = _segments(2, 21)
    names = ("q", "k", "v", "g", "beta", "h0")

    def loss(impl, p):
        o, s = ops.linear_attention("gated_deltaproduct", p["q"], p["k"], p["v"], g=p["g"], beta=p["beta"],
                                    impl=impl, initial_state=p["h0"], output_final_state=True, segment_ids=seg,
                                    use_qk_l2norm=True, **({"chunk_size": 16} if impl == "chunked" else {}))
        return jnp.sum(o * x["do"]) + jnp.sum(s * x["dht"])

    params = {name: x[name] for name in names}
    got = jax.grad(lambda p: loss("chunked", p))(params)
    want = jax.grad(lambda p: loss("reference", p))(params)
    for name in names:
        _assert_close(got[name], want[name], 1e-4, f"d{name}")


def test_interleave_layout():
    q = jnp.arange(2 * 3 * 1 * 1, dtype=F32).reshape(2, 3, 1, 1) + 1
    g = -jnp.arange(2 * 3 * 1, dtype=F32).reshape(2, 3, 1) - 1
    seg = jnp.asarray([[0, 0, 1], [2, 3, 3]], jnp.int32)
    q_i, g_i, seg_i = chunked.householder_interleave(q, g, seg, 2)
    np.testing.assert_array_equal(np.asarray(q_i[0, :, 0, 0]), [0, 1, 0, 2, 0, 3])
    np.testing.assert_array_equal(np.asarray(g_i[0, :, 0]), [-1, 0, -2, 0, -3, 0])
    np.testing.assert_array_equal(np.asarray(seg_i), [[0, 0, 0, 0, 1, 1], [2, 2, 3, 3, 3, 3]])


def test_dispatch_and_validation():
    assert {"reference", "chunked"} <= set(ops.available("gated_deltaproduct"))
    x = _inputs(5, 2)
    o_auto, _ = ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"])
    o_chunk, _ = ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"],
                                      impl="chunked")
    np.testing.assert_array_equal(np.asarray(o_auto), np.asarray(o_chunk))
    for impl in ("reference", "chunked"):
        with pytest.raises(ValueError, match="k must be"):
            ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"],
                                 impl=impl, num_householder=3)
        with pytest.raises(ValueError, match="g must be"):
            ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"][:, :-1],
                                 beta=x["beta"], impl=impl)
        with pytest.raises((NotImplementedError, ValueError)):
            ops.linear_attention("gated_deltaproduct", x["q"][:1], x["k"][:1], x["v"][:1], g=x["g"][:1],
                                 beta=x["beta"][:1], impl=impl, cu_seqlens=jnp.asarray([0, 5, 13]))


# ----------------------------------------------------------------------------- Mosaic GPU


def _on_hopper() -> bool:
    try:
        from jla.ops.mosaic_gpu import _common
    except ImportError:
        return False
    return _common.is_hopper()


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


@gpu
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_mosaic_gpu_matches_reference(n, packed):
    """Values, final state and every gradient against autodiff of the reference."""
    B, T = 2, 192
    x = _inputs(20 + n, n, B=B, T=T, H=2, K=128, V=128, dtype=BF16)
    seg = _segments(B, T) if packed else None
    names = ("q", "k", "v", "g", "beta", "h0")

    def loss(impl, p):
        o, s = ops.linear_attention("gated_deltaproduct", p["q"], p["k"], p["v"], g=p["g"], beta=p["beta"],
                                    impl=impl, initial_state=p["h0"], output_final_state=True, segment_ids=seg,
                                    use_qk_l2norm=True)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    params = {name: x[name] for name in names}
    (_, (o, s)), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(params)
    ref_params = {name: (x[name].astype(F32) if name in ("q", "k", "v") else x[name]) for name in names}
    (_, (o_ref, s_ref)), want = jax.jit(jax.value_and_grad(lambda p: loss("reference", p), has_aux=True))(ref_params)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
def test_mosaic_gpu_one_householder_equals_gated_deltanet():
    x = _inputs(31, 1, B=2, T=128, H=2, K=128, V=128, dtype=BF16)
    seg = _segments(2, 128)
    kw = dict(impl="mosaic_gpu", initial_state=x["h0"], output_final_state=True, segment_ids=seg)
    o, s = ops.linear_attention("gated_deltaproduct", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"], **kw)
    o_gdn, s_gdn = ops.linear_attention("gated_deltanet", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"], **kw)
    np.testing.assert_array_equal(np.asarray(o.astype(F32)), np.asarray(o_gdn.astype(F32)))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(s_gdn))


def test_mosaic_gpu_supports_mirrors_kernel():
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    from jla.ops.mosaic_gpu import gated_delta_product as gdp
    from jla.ops.mosaic_gpu import gated_delta_rule as gdr

    x = _inputs(41, 2, B=1, T=64, H=2, K=128, V=128, dtype=BF16)
    q_i, g_i, _ = chunked.householder_interleave(x["q"], x["g"], None, 2)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    assert gdp.supports_gated_delta_product(*args) == gdr.supports_gated_delta_rule(
        q_i, x["k"], x["v"], g_i, x["beta"])
    assert not gdp.supports_gated_delta_product(*args, num_householder=3)
    assert not gdp.supports_gated_delta_product(*args, cu_seqlens=jnp.asarray([0, 64]))
    assert not gdp.supports_gated_delta_product(*args, chunk_size=32)


@gpu
@pytest.mark.parametrize("case", ["grouped_k64", "long", "fp16_n4"])
def test_mosaic_gpu_more_shapes(case):
    """Grouped key heads and head dim 64, a long sequence, float16 with n_h = 4."""
    B, T, Hk, H, K, V, n, dtype = {
        "grouped_k64": (1, 128, 1, 2, 64, 64, 2, BF16),
        "long": (1, 1024, 2, 2, 128, 128, 2, BF16),
        "fp16_n4": (2, 64, 2, 2, 128, 64, 4, jnp.float16),
    }[case]
    ks = jax.random.split(jax.random.key(50), 8)
    x = dict(
        q=jax.random.normal(ks[0], (B, T, Hk, K)).astype(dtype),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T * n, Hk, K))).astype(dtype),
        v=jax.random.normal(ks[2], (B, T * n, H, V)).astype(dtype),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 1.0),
        beta=2.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (B, T * n, H))),
        h0=0.1 * jax.random.normal(ks[5], (B, H, K, V)),
    )
    do = jax.random.normal(ks[6], (B, T, H, V))
    dht = jax.random.normal(ks[7], (B, H, K, V))
    names = tuple(x)

    def loss(impl, p):
        q, k = p["q"], p["k"]
        if impl == "reference":  # the reference takes one key head per value head
            q, k = (jnp.repeat(a.astype(F32), H // Hk, axis=2) for a in (q, k))
        o, s = ops.linear_attention("gated_deltaproduct", q, k, p["v"], g=p["g"], beta=p["beta"], impl=impl,
                                    initial_state=p["h0"], output_final_state=True, use_qk_l2norm=True)
        return jnp.sum(o.astype(F32) * do) + jnp.sum(s * dht), (o, s)

    (_, (o, s)), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(x)
    ref_x = {name: (x[name].astype(F32) if name == "v" else x[name]) for name in names}
    (_, (o_ref, s_ref)), want = jax.jit(jax.value_and_grad(lambda p: loss("reference", p), has_aux=True))(ref_x)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")
