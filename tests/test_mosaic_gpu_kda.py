# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU kernels of the vector-gate family, through KDA (``kind="kda"``).

Each forward and backward kernel stage against ``jla.ops.chunked`` (or its autodiff), the
tensor-core path of bounded chunks against the exact path, KDA end to end against
``jla.ops.reference`` (with and without the wy pass), gradients against autodiff of the
chunked op (any ``T``, ``K, V`` in {64, 128}, initial state or none, packed documents up to 32
starts per chunk, the in-op l2norm, huge decays), the ``kda-hybrid-125m`` training shape, the
``supports`` predicate and the dispatcher. The kernel tests need an H100; the interpreter tests
(the whole pipeline in the Mosaic GPU interpreter) and the ``supports`` tests run anywhere.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import (
    BF16,
    C,
    F32,
    assert_close,
    check_decays,
    check_gradients,
    check_state_with_t,
    chunks,
    config,
    dense_segments,
    from_heads,
    gate_kwargs,
    inputs,
    interpret_loss,
    kernel_gates,
    oracle_forward,
    to_heads,
)

from jla import ops
from jla.ops import chunked, reference as ref
from jla.ops.mosaic_gpu import kda as mg_kda
from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_bwd as vgb
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

ON_HOPPER = vgo.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
# The interpreter runs every kernel op through host callbacks: on the CPU backend only (slow on a GPU).
cpu_only = pytest.mark.skipif(jax.default_backend() != "cpu", reason="interpreter tests run on the CPU backend")


def _stage_inputs(kind, x):
    """Kernel-layout inputs of the stages: G, marks, g_last and the bounded flags from the cumsum kernel."""
    cfg = config(kind, x)
    g_cum, marks, g_last, bounded = jax.jit(lambda g: vg.chunk_cumsum(cfg, g))(x["g"])
    return cfg, g_cum, marks, g_last, bounded


# ----------------------------------------------------------------------------- cumsum


@pytest.mark.parametrize("interpret", [False, True], ids=["gpu", "interpret"])
def test_cumsum_kernels_match_jnp(interpret):
    if not interpret and not ON_HOPPER:
        pytest.skip("needs an H100 (compute capability 9.0)")
    B, T, H, K = 2, 128, 3, 128
    g = -jax.nn.softplus(jax.random.normal(jax.random.key(0), (B, T, H, K)))
    g = g.at[1, 70, 2, 5].set(-100.0)  # one 16-token span (chunk 1, head 2) decays by more than BOUNDED_SPAN
    cfg = vg.Config(batch=B, seq_len=T, heads=H, key_dim=K, value_dim=K, scale=1.0, channel_gates=False,
                    interpret=interpret)
    g_cum, marks, g_last, bounded = jax.jit(lambda g: vg.chunk_cumsum(cfg, g))(g)
    want = chunked.chunk_local_cumsum(chunks(g))  # [B, NC, H, C, K]
    np.testing.assert_allclose(chunks(g_cum), want, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(marks, chunked._from_chunks(want)[:, 7::8], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(g_last, want[:, :, :, -1], rtol=1e-6, atol=1e-5)
    ends = want[:, :, :, vg.SUB_CHUNK - 1 :: vg.SUB_CHUNK]  # [B, NC, H, 4, K]
    starts = jnp.concatenate([jnp.zeros_like(ends[:, :, :, :1]), ends[:, :, :, :-1]], axis=3)
    want_bounded = jnp.all(starts - ends <= vg.BOUNDED_SPAN, axis=(3, 4)).astype(jnp.int32)  # one head per group
    np.testing.assert_array_equal(bounded, want_bounded)
    assert int(bounded[1, 1, 2]) == 0 and int(bounded.sum()) == bounded.size - 1
    rev = jax.jit(lambda x: vg.chunk_reverse_cumsum(cfg, x))(g)
    want_rev = jnp.flip(jnp.cumsum(jnp.flip(chunks(g), 3), axis=3), 3)
    np.testing.assert_allclose(chunks(rev), want_rev, rtol=1e-6, atol=1e-5)


def test_cumsum_rejects_widths_off_the_scan_tile():
    cfg = vg.Config(batch=1, seq_len=64, heads=1, key_dim=64, value_dim=64, scale=1.0, channel_gates=False)
    with pytest.raises(ValueError, match="multiple of 128"):
        vg.chunk_cumsum(cfg, jnp.zeros((1, 64, 1, 64)))


# ----------------------------------------------------------------------------- forward stages


@gpu
@pytest.mark.parametrize("dims", [(128, 128), (64, 64), (128, 64)], ids=["k128v128", "k64v64", "k128v64"])
@pytest.mark.parametrize("huge", [False, True], ids=["plain", "huge_decay"])
def test_forward_stages_match_chunked(dims, huge):
    K, V = dims
    x = inputs(0, "kda", B=1, T=256, H=2, K=K, V=V, huge=huge)
    want = oracle_forward("kda", x)
    cfg, g_cum, marks, g_last, bounded = _stage_inputs("kda", x)
    assert int(bounded.min()) == int(bounded.max()) == (0 if huge else 1)  # the exact or the factored path
    erase, write = kernel_gates("kda", x)
    aqk, akk = jax.jit(lambda *a: vg.intra(cfg, *a))(x["q"], x["k"], g_cum, marks, erase, bounded)
    assert_close(from_heads(aqk), want["aqk"], 1e-2, "Aqk")
    assert_close(from_heads(akk), want["akk"], 5e-3, "Akk")
    check_decays(cfg, x, want, (g_cum, marks, erase, bounded), aqk, akk)
    t = jax.jit(lambda a: vg.solve(cfg, a))(to_heads(want["akk"], F32))
    assert_close(from_heads(t), want["t"], 1e-2, "T")
    w, u, kg, qg = jax.jit(lambda *a: vg.wy(cfg, *a))(to_heads(want["t"]), x["q"], x["k"], x["v"], g_cum, marks,
                                                    erase, write)
    for name, got in (("w", w), ("u", u), ("kg", kg), ("qg", qg)):
        assert_close(from_heads(got), want[name], 1e-2, name)
    state_inputs = (*(to_heads(want[n]) for n in ("kg", "w", "u")), g_last, x["h0"], to_heads(want["qg"]),
                    to_heads(want["aqk"]))
    for block_v in (32, 64, 128)[: 2 + (V == 128)]:  # every value tile of the fused state / output pass
        scfg = dataclasses.replace(cfg, block_v=block_v)
        o, final, h, v_new = jax.jit(lambda *a: vg.state_output(scfg, *a, save=True))(*state_inputs)
        assert_close(chunks(o), want["o"], 1e-2, "o")
        assert_close(final, want["final"], 1e-2, "final")
        assert_close(jnp.swapaxes(h.astype(F32), 1, 2), want["h"], 1e-2, "h")
        assert_close(from_heads(v_new), want["v_new"], 1e-2, "v_new")
        o2, final2 = jax.jit(lambda *a: vg.state_output(scfg, *a, save=False))(*state_inputs)
        np.testing.assert_array_equal(np.asarray(o2, np.float32), np.asarray(o, np.float32))
        np.testing.assert_array_equal(np.asarray(final2), np.asarray(final))
        check_state_with_t(scfg, x, want, g_last, write)


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("key_dim", [128, 64])
def test_intra_paths_agree_and_mix(kind, key_dim):
    """One call with bounded and unbounded chunks (a -100 log-decay in chunk 2) against the
    oracle, and the factored path of every bounded chunk against the exact path on the same chunk."""
    x = inputs(3, kind, B=1, T=256, H=2, K=key_dim, V=128)
    x["g"] = x["g"].at[0, 130].set(-100.0)
    want = oracle_forward(kind, x)
    cfg, g_cum, marks, _, bounded = _stage_inputs(kind, x)
    assert bounded[0, :, 0].tolist() == [1, 1, 0, 1]
    erase, _ = kernel_gates(kind, x)
    run = jax.jit(lambda *a: vg.intra(cfg, *a))
    aqk, akk = run(x["q"], x["k"], g_cum, marks, erase, bounded)
    assert_close(from_heads(aqk), want["aqk"], 1e-2, "Aqk")
    assert_close(from_heads(akk), want["akk"], 5e-3, "Akk")
    aqk_x, akk_x = run(x["q"], x["k"], g_cum, marks, erase, jnp.zeros_like(bounded))  # exact path throughout
    assert_close(from_heads(aqk), from_heads(aqk_x), 1e-2, "Aqk paths")
    assert_close(from_heads(akk), from_heads(akk_x), 5e-3, "Akk paths")
    exact_chunk = (slice(None), slice(None), slice(128, 192))  # chunk 2 takes the exact path in both calls
    np.testing.assert_array_equal(np.asarray(aqk[exact_chunk], np.float32), np.asarray(aqk_x[exact_chunk], np.float32))


@gpu
def test_state_without_initial_state_starts_from_zero():
    x = inputs(1, "kda", B=1, T=128, H=2)
    x["h0"] = jnp.zeros_like(x["h0"])
    want = oracle_forward("kda", x)
    cfg, _, _, g_last, _ = _stage_inputs("kda", x)
    state_inputs = (*(to_heads(want[n]) for n in ("kg", "w", "u")), g_last, None, to_heads(want["qg"]),
                    to_heads(want["aqk"]))
    o, final, h, v_new = jax.jit(lambda *a: vg.state_output(cfg, *a, save=True))(*state_inputs)
    assert_close(chunks(o), want["o"], 1e-2, "o")
    assert_close(from_heads(v_new), want["v_new"], 1e-2, "v_new")
    assert_close(final, want["final"], 1e-2, "final")
    np.testing.assert_array_equal(np.asarray(h[:, :, 0], np.float32), 0.0)


# ----------------------------------------------------------------------------- backward stages


@gpu
@pytest.mark.parametrize("huge", [False, True], ids=["plain", "huge_decay"])
def test_backward_stages_match_chunked(huge):
    x = inputs(2, "kda", B=1, T=192, H=2, K=128, V=128, huge=huge)
    want = oracle_forward("kda", x)
    cfg, g_cum, marks, g_last, _ = _stage_inputs("kda", x)
    erase, write = kernel_gates("kda", x)
    scale = cfg.scale
    do = x["do"].astype(BF16)
    doc, gc, qc, kc = chunks(do), want["gc"], chunks(x["q"]), chunks(x["k"])

    # bwd_dav: dAqk = tril(do v_new^T), dv_o = Aqk^T do, qg = scale q exp(G).
    v_new, aqk = to_heads(want["v_new"]), to_heads(want["aqk"])
    daqk, dv_o, qg = jax.jit(lambda *a: vgb.bwd_dav(cfg, *a))(x["q"], g_cum, do, v_new, aqk)
    causal = jnp.tril(jnp.ones((C, C), bool))
    vnc, aqkc = from_heads(v_new), from_heads(aqk)
    assert_close(from_heads(daqk), jnp.where(causal, jnp.einsum("...iv,...jv->...ij", doc, vnc), 0.0), 1e-2, "dAqk")
    assert_close(from_heads(dv_o), jnp.einsum("...ij,...iv->...jv", aqkc, doc), 1e-2, "dv_o")
    qg_ref = scale * qc * jnp.exp(gc)
    assert_close(from_heads(qg), qg_ref, 1e-2, "qg")

    # bwd_state: the reverse pass dS_n = exp(G_last) dS_{n+1} + qg^T do - w^T dv_new.
    kg, w = to_heads(want["kg"]), to_heads(want["w"])
    d_final = x["dht"]
    dh, dv_new, dh0 = jax.jit(lambda *a: vgb.bwd_state(cfg, *a))(qg, kg, w, g_last, do, dv_o, d_final)
    kgc, wc, qgc, dvoc = from_heads(kg), from_heads(w), from_heads(qg), from_heads(dv_o)
    ds = d_final
    dvn_ref, dh_ref = [], []
    for n in reversed(range(cfg.num_chunks)):
        dh_ref.append(ds)
        dvn = dvoc[:, n] + jnp.einsum("bhck,bhkv->bhcv", kgc[:, n], ds)
        dvn_ref.append(dvn)
        ds = (jnp.exp(gc[:, n, :, -1])[..., None] * ds + jnp.einsum("bhck,bhcv->bhkv", qgc[:, n], doc[:, n])
              - jnp.einsum("bhck,bhcv->bhkv", wc[:, n], dvn))
    assert_close(from_heads(dv_new), jnp.stack(dvn_ref[::-1], 1), 1e-2, "dv_new")
    assert_close(jnp.swapaxes(dh.astype(F32), 1, 2), jnp.stack(dh_ref[::-1], 1), 1e-2, "dh")
    assert_close(dh0, ds, 1e-2, "dh0")

    # bwd_dakk: dAkk = -T^T dT T^T (strictly lower) from the partial sums of dT.
    t = to_heads(want["t"])
    dt = jax.random.normal(jax.random.key(3), (cfg.batch, cfg.heads, 2, cfg.seq_len, C))
    dakk = jax.jit(lambda *a: vgb.bwd_dakk(cfg, *a))(t, dt)
    tc = from_heads(t)
    dtc = from_heads(dt[:, :, 0]) + from_heads(dt[:, :, 1])
    strict = jnp.tril(jnp.ones((C, C), bool), -1)
    dakk_ref = -jnp.einsum("...ji,...jk,...lk->...il", tc, dtc, tc)
    assert_close(from_heads(dakk), jnp.where(strict, dakk_ref, 0.0), 1e-2, "dAkk")

    # bwd_intra: the vjp of the score matrices (zero partial gradients).
    beta_c = chunks(x["gates"][0])
    dakk_up = jnp.where(strict, jax.random.normal(jax.random.key(4), want["akk"].shape), 0.0)
    daqk_up = from_heads(daqk)

    def scores(q, k, g, beta):
        aqk = scale * chunked.decayed_scores(q, k, g, inclusive=True, precision=jax.lax.Precision.HIGHEST)
        akk = chunked.chunk_scaled_dot_kkt(k, beta, g, precision=jax.lax.Precision.HIGHEST)
        return aqk, akk

    _, vjp = jax.vjp(scores, qc, kc, gc, beta_c)
    dq_ref, dk_ref, dg_ref, dbeta_ref = vjp((daqk_up, dakk_up))
    zeros = jnp.zeros(x["q"].shape, F32)
    gate_part = jnp.zeros((*x["q"].shape[:3], 2), F32)
    dq, dk, dg, dbeta = jax.jit(lambda *a: vgb.bwd_intra(cfg, *a))(
        x["q"], x["k"], g_cum, marks, erase, daqk, to_heads(dakk_up, F32), zeros, zeros, zeros, gate_part)
    assert_close(chunks(dq), dq_ref, 1e-2, "dq")
    assert_close(chunks(dk), dk_ref, 1e-2, "dk")
    assert_close(chunks(dg), dg_ref, 1e-2, "dG")
    assert_close(chunks(dbeta), dbeta_ref, 1e-2, "dbeta")


# ----------------------------------------------------------------------------- end to end


@gpu
@pytest.mark.parametrize(
    "shape",
    [(2, 200, 2, 128, 128), (1, 100, 2, 64, 64), (1, 256, 2, 64, 128), (1, 130, 2, 128, 64), (2, 10, 2, 64, 64),
     (3, 130, 4, 128, 128), (3, 200, 4, 64, 64)],
    ids=["k128v128_t200", "k64v64_t100", "k64v128_t256", "k128v64_t130", "k64v64_t10",
         "bh12_t_in_state_pass", "k64_bh12_t_in_state_pass"],  # B H > 8: the forward without the wy pass
)
def test_kda_matches_reference(shape):
    B, T, H, K, V = shape
    x = inputs(5, "kda", B, T, H, K, V)
    beta = x["gates"][0]
    o, final = ops.linear_attention("kda", x["q"], x["k"], x["v"], g=x["g"], beta=beta, impl="mosaic_gpu",
                                    initial_state=x["h0"], output_final_state=True)
    q32, k32, v32 = (x[n].astype(F32) for n in ("q", "k", "v"))
    o_ref, final_ref = ref.kda(q32, k32, v32, x["g"], beta, initial_state=x["h0"], output_final_state=True)
    assert o.shape == (B, T, H, V) and o.dtype == BF16
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")
    o2, none = mg_kda.kda(x["q"], x["k"], x["v"], x["g"], beta, scale=0.5)
    assert none is None and o2.shape == o.shape


@gpu
@pytest.mark.parametrize(
    "case",
    [
        dict(shape=(2, 200, 2, 128, 128)),
        dict(shape=(1, 256, 2, 128, 128), huge=True),
        dict(shape=(2, 300, 2, 128, 128), packed=True),
        dict(shape=(1, 256, 2, 64, 64), no_state=True),
        dict(shape=(1, 192, 2, 64, 128), huge=True, packed=True),
        dict(shape=(1, 130, 2, 128, 64), l2=True, packed=True, no_state=True),
    ],
    ids=["t200", "huge_decay", "packed", "k64_no_state", "k64v128_huge_packed", "l2norm_packed"],
)
def test_kda_gradients_match_chunked(case):
    check_gradients("kda", 6, **case)


@gpu
def test_kda_hybrid_125m_training_shape():
    """``kda-hybrid-125m`` (6 heads of 128) at B=8, T=2048 as the model calls it: packed documents,
    the in-op l2norm, no initial state."""
    check_gradients("kda", 8, (8, 2048, 6, 128, 128), packed=True, no_state=True, l2=True)


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("shape", [(3, 130, 4, 128, 128), (1, 130, 2, 128, 128)], ids=["bh12_in_intra", "bh2_xla"])
def test_l2norm_matches_reference(kind, shape):
    """``use_qk_l2norm`` with raw ``q, k``: scaled inside intra on the forward without the wy pass
    (B H > 8) and by XLA otherwise, against the reference's l2norm."""
    B, T, H, K, V = shape
    x = inputs(11, kind, B, T, H, K, V)
    ks = jax.random.split(jax.random.key(12), 2)
    q = (3.0 * jax.random.normal(ks[0], x["q"].shape)).astype(BF16)
    k = (0.2 * jax.random.normal(ks[1], x["k"].shape)).astype(BF16)
    o, final = ops.linear_attention(kind, q, k, x["v"], g=x["g"], impl="mosaic_gpu", initial_state=x["h0"],
                                    output_final_state=True, use_qk_l2norm=True, **gate_kwargs(kind, x["gates"]))
    oracle = ref.kda if kind == "kda" else ref.gdn2
    o_ref, final_ref = oracle(*(a.astype(F32) for a in (q, k, x["v"])), x["g"], *x["gates"], initial_state=x["h0"],
                              output_final_state=True, use_qk_l2norm=True)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
def test_dense_segments_match_reference(kind):
    """Up to 32 segment starts in one chunk (``dense_segments``): their -128 log-decay offsets add up
    to -4096 in ``G``, where the f32 spacing is 4.9e-4. The forward (B H = 12: without the wy pass)
    against the sequential f32 reference at the bf16 tolerance of the other tests, and gradients."""
    B, T, H, K, V = 3, 256, 4, 128, 128
    x = inputs(9, kind, B, T, H, K, V)
    seg = dense_segments(B, T)
    o, final = ops.linear_attention(kind, x["q"], x["k"], x["v"], g=x["g"], impl="mosaic_gpu", initial_state=x["h0"],
                                    output_final_state=True, segment_ids=seg, **gate_kwargs(kind, x["gates"]))
    q32, k32, v32 = (x[n].astype(F32) for n in ("q", "k", "v"))
    oracle = ref.kda if kind == "kda" else ref.gdn2
    o_ref, final_ref = oracle(q32, k32, v32, x["g"], *x["gates"], initial_state=x["h0"], output_final_state=True,
                              segment_ids=seg)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")
    check_gradients(kind, 9, (B, T, H, K, V), packed="dense")


# ----------------------------------------------------------------------------- supports and dispatch


def _call(B=1, T=128, H=2, K=128, V=128, dtype=BF16):
    q = jnp.zeros((B, T, H, K), dtype)
    return (q, q, jnp.zeros((B, T, H, V), dtype), jnp.zeros((B, T, H, K)), jnp.zeros((B, T, H)))


def test_supports_kda_is_strict():
    q, k, v, g, beta = _call()
    assert mg_kda.supports_kda(q, k, v, g, beta) == ON_HOPPER
    assert mg_kda.supports_kda(q, k, v, g, beta, segment_ids=jnp.zeros((1, 128), jnp.int32),
                               use_qk_l2norm=True, initial_state=jnp.zeros((1, 2, 128, 128))) == ON_HOPPER
    rejected = [
        (_call(dtype=F32), {}),
        (_call(K=32), {}),
        (_call(V=256), {}),
        (_call(H=1, K=64, V=64), {}),  # H * K must be a multiple of 128
        ((q, k, v, g, jnp.zeros((1, 128, 2, 1))), {}),
        ((q, k, v, g, beta), dict(cu_seqlens=jnp.array([0, 128]))),
        ((q, k, v, g, beta), dict(chunk_size=32)),
        ((q, k, v, g, beta), dict(segment_ids=jnp.zeros((1, 64), jnp.int32))),
        ((q, k, v, g, beta), dict(initial_state=jnp.zeros((1, 2, 128, 64)))),
        ((q, k, v, g, beta), dict(unknown_option=1)),
    ]
    for args, options in rejected:
        assert not mg_kda.supports_kda(*args, **options), options


def test_kda_rejects_cu_seqlens_and_other_chunk_sizes():
    q, k, v, g, beta = _call()
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        mg_kda.kda(q, k, v, g, beta, cu_seqlens=jnp.array([0, 128]))
    with pytest.raises(ValueError, match="chunk_size"):
        mg_kda.kda(q, k, v, g, beta, chunk_size=32)


@gpu
def test_auto_dispatch_uses_the_kernels_and_falls_back():
    x = inputs(9, "kda", B=1, T=128, H=2)
    beta = x["gates"][0]
    direct, _ = mg_kda.kda(x["q"], x["k"], x["v"], x["g"], beta)
    auto, _ = ops.linear_attention("kda", x["q"], x["k"], x["v"], g=x["g"], beta=beta)
    np.testing.assert_array_equal(np.asarray(auto, np.float32), np.asarray(direct, np.float32))
    q32 = x["q"].astype(F32)  # not supported by the kernels: impl="auto" runs the chunked op
    fallback, _ = ops.linear_attention("kda", q32, x["k"].astype(F32), x["v"].astype(F32), g=x["g"], beta=beta)
    want, _ = chunked.kda(q32, x["k"].astype(F32), x["v"].astype(F32), x["g"], beta)
    np.testing.assert_allclose(fallback, want, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- interpreter


@cpu_only
@pytest.mark.parametrize("shape", [(1, 64, 2, 64, 64), (1, 100, 1, 128, 128)], ids=["k64", "k128_t100"])
def test_kda_pipeline_in_the_interpreter(shape):
    """Forward and backward kernels in the Mosaic GPU interpreter, against chunked autodiff."""
    x = inputs(10, "kda", *shape)
    args = (x["q"], x["k"], x["v"], x["g"], x["gates"], x["h0"])
    got = jax.grad(interpret_loss("kda", x["do"]), (0, 1, 2, 3, 4, 5))(*args)

    def loss_ref(q, k, v, g, gates, h0):
        o, final = chunked.kda(q.astype(F32), k.astype(F32), v.astype(F32), g, gates[0], initial_state=h0,
                               output_final_state=True)
        return jnp.sum(o * x["do"]) + jnp.sum(final)

    want = jax.grad(loss_ref, (0, 1, 2, 3, 4, 5))(*args)
    for i, (a, b) in enumerate(zip(jax.tree.leaves(got), jax.tree.leaves(want))):
        assert_close(a, b, 1.5e-2, f"grad{i}")
