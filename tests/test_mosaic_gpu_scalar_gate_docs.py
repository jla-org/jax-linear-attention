# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Document-aware map passes of the scalar-gate sequence split (``scalar_gate_docs``).

The walk tables on any backend; on an H100 the walked map passes against the
full walks of ``gated_delta_rule_bwd`` (the same values), and the split with
packed documents (aligned, unaligned, sparse and dense document starts,
initial and final state, bfloat16 and float16 inputs) end to end against
chunked autodiff and against the previous full-walk path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import gated_delta_rule_bwd as gb
from jla.ops.mosaic_gpu import gated_delta_rule_kernels as gk
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu import scalar_gate_docs as sgd

gpu = pytest.mark.skipif(not cm.is_hopper(), reason="needs an H100 (compute capability 9.0)")

F32, BF16 = jnp.float32, jnp.bfloat16
C = cm.CHUNK
HIGHEST = lax.Precision.HIGHEST
DELTA_GATES = {"deltanet": ("beta",), "gated_deltanet": ("g", "beta")}


def _inputs(seed, B, T, H, K=128, V=128):
    ks = jax.random.split(jax.random.key(seed), 8)
    return dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, H, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H))),
        h0=0.1 * jax.random.normal(ks[5], (B, H, K, V)),
        do=jax.random.normal(ks[6], (B, T, H, V)),
        dht=jax.random.normal(ks[7], (B, H, K, V)),
    )


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _starts_to_ids(B, T, starts_per_row):
    rows = np.zeros((B, T), np.int32)
    for b, starts in enumerate(starts_per_row):
        rows[b] = np.searchsorted(np.sort(np.asarray(starts, np.int64)), np.arange(T), side="right")
    return jnp.asarray(rows)


def _segment_ids(B, T, pattern):
    """Document layouts for ``T = 1024`` (16 chunks; 4 pieces of 4 chunks)."""
    if pattern == "aligned":  # 128-token documents: every second chunk starts clean
        return _starts_to_ids(B, T, [range(128, T, 128)] * B)
    if pattern == "unaligned":  # starts inside chunks, some pieces without one
        return _starts_to_ids(B, T, [[100, 300, 333, 650, 900], [37, 700]][:B])
    if pattern == "long":  # a start only in the last piece / in the first chunk of piece 1
        return _starts_to_ids(B, T, [[1000], [266]][:B])
    if pattern == "tiny":  # many short documents (mean length 5)
        rng = np.random.default_rng(0)
        return _starts_to_ids(B, T, [rng.choice(np.arange(1, T), size=T // 5, replace=False) for _ in range(B)])
    raise ValueError(pattern)


def test_walk_tables():
    """Forward: from the last kill chunk; backward: up to the first; unused pieces walk nothing."""
    T, cp = 1024, 4
    seg = cm.segment_vectors(_segment_ids(2, T, "unaligned"), 0)
    fwd, bwd = (np.asarray(x).reshape(2, cp, 2) for x in sgd.walk_tables(seg, 2, T, cp, need_last=False))
    # Row 0 starts at tokens 100, 300, 333, 650, 900 -> chunks 1, 4, 5, 10, 14 kill the incoming state
    # (their last token is in a new document). Pieces: chunks 0-3, 4-7, 8-11, 12-15.
    np.testing.assert_array_equal(fwd[0], [[1, 3], [1, 3], [2, 2], [2, 0]])
    np.testing.assert_array_equal(bwd[0], [[0, 0], [0, 1], [0, 3], [0, 3]])
    # Row 1 starts at 37 (chunk 0) and 700 (chunk 10).
    np.testing.assert_array_equal(fwd[1], [[0, 4], [0, 4], [2, 2], [0, 0]])
    np.testing.assert_array_equal(bwd[1], [[0, 0], [0, 4], [0, 3], [0, 4]])
    fwd, bwd = (np.asarray(x).reshape(1, cp, 2) for x in sgd.walk_tables(None, 1, T, cp, need_last=False))
    np.testing.assert_array_equal(fwd[0], [[0, 4], [0, 4], [0, 4], [0, 0]])
    np.testing.assert_array_equal(bwd[0], [[0, 0], [0, 4], [0, 4], [0, 4]])
    # A final-state gradient flows back through the last piece's map, so it is walked too.
    fwd, _ = (np.asarray(x).reshape(2, cp, 2) for x in sgd.walk_tables(seg, 2, T, cp, need_last=True))
    np.testing.assert_array_equal(fwd[:, -1], [[2, 2], [0, 4]])


def _kernel_gates(x, pad=0):
    B, T, H = x["g"].shape
    return cm.chunk_log2_decay(x["g"], B, T, H, pad), cm.head_major(x["beta"], pad)


@gpu
@pytest.mark.parametrize("pattern", ["none", "aligned", "unaligned", "long", "tiny"])
def test_map_passes_match_full_walks(pattern):
    """The walked passes give the full walks' maps (forward) and start gradients (backward)."""
    B, T, H, cp = 2, 1024, 2, 4
    x = _inputs(21, B, T, H)
    g2, bt = _kernel_gates(x)
    seg = None if pattern == "none" else cm.segment_vectors(_segment_ids(B, T, pattern), 0)
    q, k, v, do = x["q"], x["k"], x["v"], x["do"].astype(BF16)
    scale = 128 ** -0.5

    @jax.jit
    def both(q, k, v, g2, bt, do):
        t0 = gk.kkt_fwd(k, bt, seg=seg)
        _, full_fwd = gb.state_fwd_wy(k, v, t0, g2, bt, None, store_states=False, out_final=True, aug=True,
                                      seg=seg, cp=cp)
        walk_fwd = sgd.map_fwd_wy(k, v, t0, g2, bt, seg=seg, cp=cp, skip_fresh=False)
        walk_fwd_no_last = sgd.map_fwd_wy(k, v, t0, g2, bt, seg=seg, cp=cp, need_last=False, skip_fresh=False)
        walk_fwd_fresh = sgd.map_fwd_wy(k, v, t0, g2, bt, seg=seg, cp=cp)  # the default skips fresh successors
        _, _, full_bwd = gb.state_bwd_wy(q, k, do, t0, g2, bt, None, scale=scale, want_dh0=True,
                                         store_grads=False, seg=seg, cp=cp)
        walk_bwd = sgd.map_bwd_wy(q, k, do, t0, g2, bt, scale=scale, seg=seg, cp=cp)
        return full_fwd, walk_fwd, walk_fwd_no_last, full_bwd, walk_bwd, walk_fwd_fresh

    full_fwd, walk_fwd, walk_fwd_no_last, full_bwd, walk_bwd, walk_fwd_fresh = (
        np.asarray(a).reshape(B, cp, *a.shape[1:]) for a in both(q, k, v, g2, bt, do))
    # Without a final-state gradient the scans read every piece's map but the last one's, and every
    # reverse term but the first one's.
    np.testing.assert_allclose(walk_fwd, full_fwd, rtol=0, atol=1e-6)
    np.testing.assert_allclose(walk_fwd_no_last[:, :-1], full_fwd[:, :-1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(walk_bwd[:, 1:], full_bwd[:, 1:], rtol=0, atol=1e-6)
    # A piece whose successor starts a document at its first token holds [0 | 0] by default (nothing reads
    # its map); every other piece is the walk without the skip, bit for bit.
    fresh = np.asarray(sgd.fresh_next(seg, B, T, cp))
    np.testing.assert_array_equal(walk_fwd_fresh[~fresh], walk_fwd[~fresh])
    np.testing.assert_array_equal(walk_fwd_fresh[fresh], 0)


def _run(kind, p, segment_ids, cp, walk):
    return sg.run(p["q"], p["k"], p["v"], p.get("g"), p.get("beta"), delta=True, scale=None,
                  initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids, cp=cp, walk=walk)


def _value_and_grads(f, x, names):
    def loss(p):
        o, s = f(p)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    (_, (o, s)), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))({n: x[n] for n in names})
    return o, s, grads


@gpu
@pytest.mark.parametrize("pattern", ["aligned", "unaligned", "long"])
@pytest.mark.parametrize("kind", sorted(DELTA_GATES))
def test_split_with_documents_matches_chunked_and_full_walk(kind, pattern):
    """Values, final state and every gradient against chunked autodiff and the previous full walks."""
    B, T, H, cp = 2, 1024, 2, 4
    x = _inputs(22, B, T, H)
    segment_ids = _segment_ids(B, T, pattern)
    names = ("q", "k", "v") + DELTA_GATES[kind] + ("h0",)

    def chunked(p):
        gates = {n: p[n] for n in DELTA_GATES[kind]}
        return ops.linear_attention(kind, *(p[n].astype(F32) for n in ("q", "k", "v")), impl="chunked",
                                    initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids,
                                    precision=HIGHEST, **gates)

    o, s, got = _value_and_grads(lambda p: _run(kind, p, segment_ids, cp, True), x, names)
    o_full, s_full, full = _value_and_grads(lambda p: _run(kind, p, segment_ids, cp, False), x, names)
    o_ref, s_ref, want = _value_and_grads(chunked, x, names)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    _assert_close(o, o_full, 1e-6, "o vs full walk")
    _assert_close(s, s_full, 1e-6, "final state vs full walk")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")
        _assert_close(got[name], full[name], 1e-6, f"d{name} vs full walk")


@gpu
def test_ragged_split_with_documents_and_grouped_heads():
    """A padded length (300 tokens in 2 pieces of 192), documents and key heads shared by two value heads."""
    B, T, H, cp = 1, 300, 2, 2
    x = _inputs(23, B, T, H, 64, 64)
    x["q"], x["k"] = x["q"][:, :, :1], x["k"][:, :, :1]
    segment_ids = _starts_to_ids(B, T, [[70, 150, 151, 250]])
    names = ("q", "k", "v", "g", "beta", "h0")
    o, s, got = _value_and_grads(lambda p: _run("gated_deltanet", p, segment_ids, cp, True), x, names)
    o_full, s_full, full = _value_and_grads(lambda p: _run("gated_deltanet", p, segment_ids, cp, False), x, names)
    _assert_close(o, o_full, 1e-6, "o")
    _assert_close(s, s_full, 1e-6, "final state")
    for name in names:
        _assert_close(got[name], full[name], 1e-6, f"d{name}")


@gpu
@pytest.mark.parametrize("pattern", ["none", "unaligned"])
def test_split_without_final_state_skips_the_last_map(pattern):
    """Without a final state the last piece's map is not walked; values and gradients stay those of the full walks."""
    B, T, H, cp = 2, 1024, 2, 4
    x = _inputs(24, B, T, H)
    segment_ids = None if pattern == "none" else _segment_ids(B, T, pattern)
    names = ("q", "k", "v", "g", "beta", "h0")

    def f(walk):
        def call(p):
            o, _ = sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None, initial_state=p["h0"],
                          output_final_state=False, segment_ids=segment_ids, cp=cp, walk=walk)
            return o

        def loss(p):
            o = call(p)
            return jnp.sum(o.astype(F32) * x["do"]), o

        return jax.jit(jax.value_and_grad(loss, has_aux=True))({n: x[n] for n in names})

    (_, o), got = f(True)
    (_, o_full), full = f(False)
    _assert_close(o, o_full, 1e-6, "o")
    for name in names:
        _assert_close(got[name], full[name], 1e-6, f"d{name}")


@gpu
@pytest.mark.parametrize("pattern", ["none", "unaligned", "tiny"])
def test_simple_gla_map_passes_match_full_walks(pattern):
    """simple GLA: the walked local end states and start gradients against the full state passes."""
    from jla.ops.mosaic_gpu import scalar_gate_kernels as sk

    B, T, H, cp = 2, 1024, 2, 4
    x = _inputs(25, B, T, H, 128, 64)
    g2 = cm.chunk_log2_decay(x["g"], B, T, H, 0)
    seg = None if pattern == "none" else cm.segment_vectors(_segment_ids(B, T, pattern), 0)
    q, k, v, do = x["q"], x["k"], x["v"], x["do"].astype(BF16)

    @jax.jit
    def both(q, k, v, g2, do):
        _, _, full_fwd = sk.state_fwd(k, None, v, g2, None, u_head_major=False, out_final=True, store_states=False,
                                      seg=seg, cp=cp)
        walk_fwd = sgd.map_fwd_gla(k, v, g2, seg=seg, cp=cp)
        dvl = sk.dv_local(q, k, g2, do, scale=1.0, seg=seg)
        _, _, full_bwd = sk.state_bwd(q, k, None, g2, do, dvl, None, scale=1.0, want_dh0=True, store_grads=False,
                                      seg=seg, cp=cp)
        walk_bwd = sgd.map_bwd_gla(q, k, g2, do, scale=1.0, seg=seg, cp=cp)
        return full_fwd, walk_fwd, full_bwd, walk_bwd

    full_fwd, walk_fwd, full_bwd, walk_bwd = (np.asarray(a).reshape(B, cp, *a.shape[1:])
                                              for a in both(q, k, v, g2, do))
    np.testing.assert_allclose(walk_fwd, full_fwd, rtol=0, atol=1e-6)
    np.testing.assert_allclose(walk_bwd[:, 1:], full_bwd[:, 1:], rtol=0, atol=1e-6)


@gpu
@pytest.mark.parametrize("final", [True, False], ids=["final_state", "no_final_state"])
@pytest.mark.parametrize("pattern", ["unaligned", "long"])
def test_simple_gla_split_with_documents(pattern, final):
    """simple GLA with documents: values and gradients against chunked autodiff and the full walks."""
    B, T, H, cp = 2, 1024, 2, 4
    x = _inputs(26, B, T, H, 128, 64)
    segment_ids = _segment_ids(B, T, pattern)
    names = ("q", "k", "v", "g", "h0")

    def mosaic(walk):
        def call(p):
            return sg.run(p["q"], p["k"], p["v"], p["g"], None, delta=False, scale=1.0, initial_state=p["h0"],
                          output_final_state=final, segment_ids=segment_ids, cp=cp, walk=walk)
        return call

    def chunked(p):
        return ops.linear_attention("simple_gla", *(p[n].astype(F32) for n in ("q", "k", "v")), g=p["g"],
                                    impl="chunked", scale=1.0, initial_state=p["h0"], output_final_state=final,
                                    segment_ids=segment_ids, precision=HIGHEST)

    def grads(f):
        def loss(p):
            o, s = f(p)
            extra = jnp.sum(s * x["dht"]) if final else 0.0
            return jnp.sum(o.astype(F32) * x["do"]) + extra, o

        return jax.jit(jax.value_and_grad(loss, has_aux=True))({n: x[n] for n in names})

    (_, o), got = grads(mosaic(True))
    (_, o_full), full = grads(mosaic(False))
    (_, o_ref), want = grads(chunked)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(o, o_full, 1e-6, "o vs full walk")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")
        _assert_close(got[name], full[name], 1e-6, f"d{name} vs full walk")


@gpu
@pytest.mark.parametrize("kind", ["deltanet", "gated_deltanet", "simple_gla"])
def test_float16_split_with_documents(kind):
    """float16 inputs (in ``SUPPORTED_DTYPES``): walked passes against the full walks and chunked autodiff."""
    B, T, H, cp = 2, 1024, 2, 4
    delta = kind != "simple_gla"
    x = _inputs(27, B, T, H, 128, 128 if delta else 64)
    for n in ("q", "k", "v"):
        x[n] = x[n].astype(jnp.float16)
    segment_ids = _segment_ids(B, T, "unaligned")
    gates = DELTA_GATES.get(kind, ("g",))
    names = ("q", "k", "v") + gates + ("h0",)
    scale = None if delta else 1.0

    def mosaic(walk):
        def call(p):
            return sg.run(p["q"], p["k"], p["v"], p.get("g"), p.get("beta"), delta=delta, scale=scale,
                          initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids, cp=cp, walk=walk)
        return call

    def chunked(p):
        return ops.linear_attention(kind, *(p[n].astype(F32) for n in ("q", "k", "v")), impl="chunked", scale=scale,
                                    initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids,
                                    precision=HIGHEST, **{n: p[n] for n in gates})

    o, s, got = _value_and_grads(mosaic(True), x, names)
    o_full, s_full, full = _value_and_grads(mosaic(False), x, names)
    o_ref, s_ref, want = _value_and_grads(chunked, x, names)
    assert o.dtype == jnp.float16
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    _assert_close(o, o_full, 1e-6, "o vs full walk")
    _assert_close(s, s_full, 1e-6, "final state vs full walk")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")
        _assert_close(got[name], full[name], 1e-6, f"d{name} vs full walk")
