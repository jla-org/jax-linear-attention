# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Mosaic GPU kernels of the vector-gate family, through Gated DeltaNet-2 (``kind="gdn2"``).

The channel-wise erase gate ``b`` (read side of ``Akk`` and ``w``) and write gate ``w`` (``u``)
in each forward stage and in the score-matrix backward against ``jla.ops.chunked``, GDN-2 end
to end against ``jla.ops.reference``, gradients against autodiff of the chunked op, the
``gdn2-hybrid-125m`` training shape, agreement with KDA when both gates equal ``beta``, the
``supports`` predicate and the pipeline in the Mosaic GPU interpreter.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from _vector_gate import (
    F32,
    assert_close,
    check_decays,
    check_gradients,
    check_state_with_t,
    chunks,
    config,
    from_heads,
    inputs,
    interpret_loss,
    kernel_gates,
    oracle_forward,
    to_heads,
)

from jla import ops
from jla.ops import chunked, reference as ref
from jla.ops.mosaic_gpu import gdn2 as mg_gdn2
from jla.ops.mosaic_gpu import kda as mg_kda
from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_bwd as vgb
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

ON_HOPPER = vgo.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
# The interpreter runs every kernel op through host callbacks: on the CPU backend only (slow on a GPU).
cpu_only = pytest.mark.skipif(jax.default_backend() != "cpu", reason="interpreter tests run on the CPU backend")
HIGHEST = jax.lax.Precision.HIGHEST


@gpu
@pytest.mark.parametrize("dims", [(128, 128), (64, 128)], ids=["k128v128", "k64v128"])
@pytest.mark.parametrize("huge", [False, True], ids=["plain", "huge_decay"])
def test_gdn2_forward_stages_match_chunked(dims, huge):
    K, V = dims
    x = inputs(0, "gdn2", B=1, T=192, H=2, K=K, V=V, huge=huge)
    want = oracle_forward("gdn2", x)
    cfg = config("gdn2", x)
    g_cum, marks, g_last, bounded = jax.jit(lambda g: vg.chunk_cumsum(cfg, g))(x["g"])
    assert int(bounded.min()) == int(bounded.max()) == (0 if huge else 1)  # the exact or the factored path
    erase, write = kernel_gates("gdn2", x)
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
    for block_v in (32, 64):  # the widest tile whose write-gate tiles fit next to T in SMEM
        check_state_with_t(dataclasses.replace(cfg, block_v=block_v), x, want, g_last, write)


@gpu
def test_gdn2_score_backward_matches_chunked():
    """bwd_intra with the channel-wise erase gate: the vjp of ``(scale P(q, k), Akk(b * k, k))``."""
    x = inputs(1, "gdn2", B=1, T=128, H=2, K=128, V=128)
    want = oracle_forward("gdn2", x)
    cfg = config("gdn2", x)
    g_cum, marks, *_ = jax.jit(lambda g: vg.chunk_cumsum(cfg, g))(x["g"])
    erase, _ = kernel_gates("gdn2", x)
    qc, kc, ec = chunks(x["q"]), chunks(x["k"]), chunks(x["gates"][0])
    causal = jnp.tril(jnp.ones((vg.CHUNK, vg.CHUNK), bool))
    ks = jax.random.split(jax.random.key(2), 2)
    daqk = jnp.where(causal, jax.random.normal(ks[0], want["aqk"].shape), 0.0)
    dakk = jnp.where(jnp.tril(causal, -1), jax.random.normal(ks[1], want["akk"].shape), 0.0)

    def scores(q, k, g, e):
        aqk = cfg.scale * chunked.decayed_scores(q, k, g, inclusive=True, precision=HIGHEST)
        return aqk, chunked.chunk_scaled_dot_kkt(k, None, g, erase=e, precision=HIGHEST)

    _, vjp = jax.vjp(scores, qc, kc, want["gc"], ec)
    dq_ref, dk_ref, dg_ref, de_ref = vjp((daqk, dakk))
    zeros = jnp.zeros(x["q"].shape, F32)
    dq, dk, dg, d_erase = jax.jit(lambda *a: vgb.bwd_intra(cfg, *a))(
        x["q"], x["k"], g_cum, marks, erase, to_heads(daqk, F32), to_heads(dakk, F32), zeros, zeros, zeros, zeros)
    assert d_erase.dtype == erase.dtype
    assert_close(chunks(dq), dq_ref, 1e-2, "dq")
    assert_close(chunks(dk), dk_ref, 1e-2, "dk")
    assert_close(chunks(dg), dg_ref, 1e-2, "dG")
    assert_close(chunks(d_erase), de_ref, 1e-2, "d_erase")


@gpu
@pytest.mark.parametrize(
    "shape",
    [(2, 200, 2, 128, 128), (1, 100, 2, 64, 64), (1, 130, 2, 128, 64), (3, 130, 4, 128, 128)],
    ids=["k128v128_t200", "k64v64_t100", "k128v64_t130", "bh12_t_in_state_pass"],  # B H > 8: no wy pass
)
def test_gdn2_matches_reference(shape):
    B, T, H, K, V = shape
    x = inputs(3, "gdn2", B, T, H, K, V)
    b, w = x["gates"]
    o, final = ops.linear_attention("gdn2", x["q"], x["k"], x["v"], g=x["g"], b=b, w=w, impl="mosaic_gpu",
                                    initial_state=x["h0"], output_final_state=True)
    q32, k32, v32 = (x[n].astype(F32) for n in ("q", "k", "v"))
    o_ref, final_ref = ref.gdn2(q32, k32, v32, x["g"], b, w, initial_state=x["h0"], output_final_state=True)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")


@gpu
@pytest.mark.parametrize(
    "case",
    [
        dict(shape=(2, 200, 2, 128, 128)),
        dict(shape=(1, 256, 2, 128, 128), huge=True, packed=True),
        dict(shape=(1, 256, 2, 64, 64), no_state=True),
        dict(shape=(2, 200, 4, 64, 128), l2=True, packed=True, no_state=True),
    ],
    ids=["t200", "huge_decay_packed", "k64_no_state", "k64v128_l2norm_packed"],
)
def test_gdn2_gradients_match_chunked(case):
    check_gradients("gdn2", 4, **case)


@gpu
def test_gdn2_hybrid_125m_training_shape():
    """``gdn2-hybrid-125m`` (6 heads of 128) at B=8, T=2048: packed documents, in-op l2norm."""
    check_gradients("gdn2", 5, (8, 2048, 6, 128, 128), packed=True, no_state=True, l2=True)


@gpu
def test_gdn2_with_gates_equal_to_beta_is_kda():
    x = inputs(6, "kda", B=1, T=192, H=2, K=128, V=64)
    beta = x["gates"][0].astype(jnp.bfloat16).astype(F32)
    b = jnp.broadcast_to(beta[..., None], x["q"].shape)
    w = jnp.broadcast_to(beta[..., None], x["v"].shape)

    def loss(fn, *gates):
        def run(q, k, v, g):
            o, s = fn(q, k, v, g, *gates, initial_state=x["h0"], output_final_state=True)
            return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s)

        return run

    args = (x["q"], x["k"], x["v"], x["g"])
    got = jax.grad(loss(mg_gdn2.gdn2, b, w), (0, 1, 2, 3))(*args)
    want = jax.grad(loss(mg_kda.kda, beta), (0, 1, 2, 3))(*args)
    for i, (a, c) in enumerate(zip(got, want)):
        assert_close(a, c, 5e-3, f"grad{i}")


def test_supports_gdn2_is_strict():
    B, T, H, K, V = 1, 128, 2, 128, 64
    q = jnp.zeros((B, T, H, K), jnp.bfloat16)
    v = jnp.zeros((B, T, H, V), jnp.bfloat16)
    g, b, w = jnp.zeros((B, T, H, K)), jnp.zeros((B, T, H, K)), jnp.zeros((B, T, H, V))
    assert mg_gdn2.supports_gdn2(q, q, v, g, b, w) == ON_HOPPER
    assert mg_gdn2.supports_gdn2(q, q, v, g, b, w, segment_ids=jnp.zeros((B, T), jnp.int32)) == ON_HOPPER
    assert not mg_gdn2.supports_gdn2(q, q, v, g, w, w)  # erase gate must be [B, T, H, K]
    assert not mg_gdn2.supports_gdn2(q, q, v, g, b, b)  # write gate must be [B, T, H, V]
    assert not mg_gdn2.supports_gdn2(q.astype(F32), q, v, g, b, w)
    assert not mg_gdn2.supports_gdn2(q, q, v, g, b, w, cu_seqlens=jnp.array([0, T]))


@cpu_only
@pytest.mark.parametrize("shape", [(1, 128, 2, 64, 64), (1, 100, 1, 128, 64)], ids=["k64", "k128v64_t100"])
def test_gdn2_pipeline_in_the_interpreter(shape):
    x = inputs(7, "gdn2", *shape)
    args = (x["q"], x["k"], x["v"], x["g"], x["gates"], x["h0"])
    got = jax.grad(interpret_loss("gdn2", x["do"]), (0, 1, 2, 3, 4, 5))(*args)

    def loss_ref(q, k, v, g, gates, h0):
        o, final = chunked.gdn2(q.astype(F32), k.astype(F32), v.astype(F32), g, *gates, initial_state=h0,
                                output_final_state=True)
        return jnp.sum(o * x["do"]) + jnp.sum(final)

    want = jax.grad(loss_ref, (0, 1, 2, 3, 4, 5))(*args)
    for i, (a, b) in enumerate(zip(jax.tree.leaves(got), jax.tree.leaves(want))):
        assert_close(a, b, 1.5e-2, f"grad{i}")
