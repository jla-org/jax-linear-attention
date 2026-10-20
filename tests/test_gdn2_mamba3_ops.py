# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Gated DeltaNet-2 and Mamba-3 ops: closed-form reductions, chunked vs reference, dispatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import chunked, reference as ref

TOL = dict(rtol=2e-4, atol=2e-4)


def _inputs(seed: int, B: int = 2, T: int = 70, H: int = 2, K: int = 32, V: int = 16):
    ks = jax.random.split(jax.random.key(seed), 10)
    x = dict(
        q=jax.random.normal(ks[0], (B, T, H, K)),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, H, K))),
        v=jax.random.normal(ks[2], (B, T, H, V)),
        g_head=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        g_key=-jax.nn.softplus(jax.random.normal(ks[4], (B, T, H, K)) - 2.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H))),
        gamma=jax.nn.sigmoid(jax.random.normal(ks[6], (B, T, H))),
        b=jax.nn.sigmoid(jax.random.normal(ks[7], (B, T, H, K))),
        w=jax.nn.sigmoid(jax.random.normal(ks[8], (B, T, H, V))),
    )
    cot = jax.random.normal(ks[9], (B, T, H, V))
    return x, cot


def _segments(B: int, T: int) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    rows[0, 20:45] = 1
    rows[0, 45:] = 2
    rows[1, 31:] = 5
    rows[1, 50:64] = 0  # an id reused after a different one starts a new document
    return jnp.asarray(rows)


# ---------------------------------------------------------------- GDN-2 reductions


def test_gdn2_with_tied_scalar_gates_is_kda():
    x, _ = _inputs(0)
    K, V = x["q"].shape[-1], x["v"].shape[-1]
    b = jnp.broadcast_to(x["beta"][..., None], x["beta"].shape + (K,))
    w = jnp.broadcast_to(x["beta"][..., None], x["beta"].shape + (V,))
    got, s_got = ref.gdn2(x["q"], x["k"], x["v"], x["g_key"], b, w, output_final_state=True)
    want, s_want = ref.kda(x["q"], x["k"], x["v"], x["g_key"], x["beta"], output_final_state=True)
    np.testing.assert_allclose(got, want, **TOL)
    np.testing.assert_allclose(s_got, s_want, **TOL)


def test_gdn2_with_scalar_decay_and_tied_gates_is_gated_deltanet():
    x, _ = _inputs(1)
    K, V = x["q"].shape[-1], x["v"].shape[-1]
    g = jnp.broadcast_to(x["g_head"][..., None], x["q"].shape)
    b = jnp.broadcast_to(x["beta"][..., None], x["beta"].shape + (K,))
    w = jnp.broadcast_to(x["beta"][..., None], x["beta"].shape + (V,))
    got, _ = ref.gdn2(x["q"], x["k"], x["v"], g, b, w)
    want, _ = ref.gated_delta_rule(x["q"], x["k"], x["v"], x["g_head"], x["beta"])
    np.testing.assert_allclose(got, want, **TOL)


def test_gdn2_without_erase_is_gla_on_gated_values():
    x, _ = _inputs(2)
    got, _ = ref.gdn2(x["q"], x["k"], x["v"], x["g_key"], jnp.zeros_like(x["b"]), x["w"])
    want, _ = ref.gla(x["q"], x["k"], x["w"] * x["v"], x["g_key"])
    np.testing.assert_allclose(got, want, **TOL)


# --------------------------------------------------------------- Mamba-3 reductions


def test_mamba3_without_previous_term_is_simple_gla():
    x, _ = _inputs(3)
    got, _ = ref.mamba3(x["q"], x["k"], x["v"], x["g_head"], jnp.zeros_like(x["beta"]), x["gamma"])
    want, _ = ref.simple_gla(x["q"], x["k"], x["gamma"][..., None] * x["v"], x["g_head"], scale=1.0)
    np.testing.assert_allclose(got, want, **TOL)


def test_mamba3_is_simple_gla_on_an_interleaved_sequence():
    """Each token = (previous-key write with decay g_t) then (current write with no decay)."""
    x, _ = _inputs(4, T=33)
    q, k, v, g, beta, gamma = (x[n] for n in ("q", "k", "v", "g_head", "beta", "gamma"))
    k_prev = jnp.concatenate([jnp.zeros_like(k[:, :1]), k[:, :-1]], axis=1)
    v_prev = jnp.concatenate([jnp.zeros_like(v[:, :1]), v[:, :-1]], axis=1)

    def interleave(a, b):
        return jnp.stack([a, b], axis=2).reshape(a.shape[0], -1, *a.shape[2:])

    o2, s2 = ref.simple_gla(
        interleave(q, q), interleave(k_prev, k), interleave(beta[..., None] * v_prev, gamma[..., None] * v),
        interleave(g, jnp.zeros_like(g)), scale=1.0, output_final_state=True,
    )
    o, state = ref.mamba3(q, k, v, g, beta, gamma, output_final_state=True)
    np.testing.assert_allclose(o, o2[:, 1::2], **TOL)
    np.testing.assert_allclose(state.ssm, s2, **TOL)
    np.testing.assert_allclose(state.key, k[:, -1], **TOL)
    np.testing.assert_allclose(state.value, v[:, -1], **TOL)


def test_mamba3_state_chains_across_segments():
    x, _ = _inputs(5, T=40)
    names = ("q", "k", "v", "g_head", "beta", "gamma")
    whole, s_whole = ref.mamba3(*(x[n] for n in names), output_final_state=True)
    first, s_mid = ref.mamba3(*(x[n][:, :23] for n in names), output_final_state=True)
    second, s_end = ref.mamba3(*(x[n][:, 23:] for n in names), initial_state=s_mid, output_final_state=True)
    np.testing.assert_allclose(jnp.concatenate([first, second], axis=1), whole, **TOL)
    for a, b in zip(s_end, s_whole):
        np.testing.assert_allclose(a, b, **TOL)


def test_mamba3_segments_match_separate_runs():
    x, _ = _inputs(6)
    seg = np.asarray(_segments(2, 70))
    names = ("q", "k", "v", "g_head", "beta", "gamma")
    packed, _ = ref.mamba3(*(x[n] for n in names), segment_ids=jnp.asarray(seg))
    for row in range(2):
        starts = [0] + [t for t in range(1, 70) if seg[row, t] != seg[row, t - 1]]
        for start, end in zip(starts, starts[1:] + [70]):
            piece = [x[n][row : row + 1, start:end] for n in names]
            o, _ = ref.mamba3(*piece)
            np.testing.assert_allclose(packed[row : row + 1, start:end], o, **TOL)


# ------------------------------------------------------------- chunked vs reference


def _run(impl, name, x, **kw):
    if name == "gdn2":
        return impl.gdn2(x["q"], x["k"], x["v"], x["g_key"], x["b"], x["w"], **kw)
    return impl.mamba3(x["q"], x["k"], x["v"], x["g_head"], x["beta"], x["gamma"], **kw)


def _initial(name, x, seed=7):
    B, _, H, K = x["q"].shape
    V = x["v"].shape[-1]
    ks = jax.random.split(jax.random.key(seed), 3)
    ssm = 0.1 * jax.random.normal(ks[0], (B, H, K, V))
    if name == "gdn2":
        return ssm
    return ref.Mamba3State(ssm, jax.random.normal(ks[1], (B, H, K)), jax.random.normal(ks[2], (B, H, V)))


def _close(a, b, **tol):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, **tol)


@pytest.mark.parametrize("name", ["gdn2", "mamba3"])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("packed", [False, True])
def test_chunked_matches_reference(name, chunk_size, packed):
    x, _ = _inputs(8)
    kw = dict(initial_state=_initial(name, x), output_final_state=True)
    if packed:
        kw["segment_ids"] = _segments(2, 70)
    o_ref, s_ref = _run(ref, name, x, **kw)
    o, s = _run(chunked, name, x, chunk_size=chunk_size, **kw)
    np.testing.assert_allclose(o, o_ref, **TOL)
    _close(s, s_ref, **TOL)


@pytest.mark.parametrize("name", ["gdn2", "mamba3"])
@pytest.mark.parametrize("packed", [False, True])
def test_chunked_gradients_match_reference(name, packed):
    x, cot = _inputs(9, T=40)
    cot = cot[:, :40]
    used = ["q", "k", "v", "g_key", "b", "w"] if name == "gdn2" else ["q", "k", "v", "g_head", "beta", "gamma"]
    kw = dict(initial_state=_initial(name, x), output_final_state=True)
    if packed:
        kw["segment_ids"] = _segments(2, 70)[:, :40]

    def loss(impl, values, **extra):
        o, s = _run(impl, name, {**x, **dict(zip(used, values))}, **kw, **extra)
        return jnp.sum(o * cot) + sum(jnp.sum(leaf * leaf) for leaf in jax.tree.leaves(s))

    values = [x[n] for n in used]
    want = jax.grad(lambda v: loss(ref, v))(values)
    got = jax.grad(lambda v: loss(chunked, v, chunk_size=16))(values)
    for label, a, b in zip(used, got, want):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=label)


def test_gdn2_chunked_survives_huge_decays():
    x, _ = _inputs(10, T=64)
    spikes = jax.random.bernoulli(jax.random.key(11), 0.1, x["g_key"].shape)
    x["g_key"] = jnp.where(spikes, -60.0, x["g_key"])
    o_ref, s_ref = _run(ref, "gdn2", x, output_final_state=True)
    o, s = _run(chunked, "gdn2", x, output_final_state=True, chunk_size=64)
    assert np.isfinite(np.asarray(o)).all()
    np.testing.assert_allclose(o, o_ref, **TOL)
    np.testing.assert_allclose(s, s_ref, **TOL)


# ----------------------------------------------------------------------- dispatch


def test_dispatch_routes_new_kinds_and_checks_gates():
    x, _ = _inputs(12, T=32)
    gdn2_gates = dict(g=x["g_key"], b=x["b"], w=x["w"])
    o_ref, _ = ops.linear_attention("gdn2", x["q"], x["k"], x["v"], impl="reference", **gdn2_gates)
    o_auto, _ = ops.linear_attention("gdn2", x["q"], x["k"], x["v"], chunk_size=16, **gdn2_gates)
    np.testing.assert_allclose(o_auto, o_ref, **TOL)
    m3_gates = dict(g=x["g_head"], beta=x["beta"], gamma=x["gamma"])
    o_ref, _ = ops.linear_attention("mamba3", x["q"], x["k"], x["v"], impl="reference", **m3_gates)
    o_ch, _ = ops.linear_attention("mamba3", x["q"], x["k"], x["v"], impl="chunked", chunk_size=16, **m3_gates)
    np.testing.assert_allclose(o_ch, o_ref, **TOL)
    with pytest.raises(ValueError, match="missing"):
        ops.linear_attention("gdn2", x["q"], x["k"], x["v"], g=x["g_key"], b=x["b"])
    with pytest.raises(ValueError, match="unexpected"):
        ops.linear_attention("gated_deltanet", x["q"], x["k"], x["v"], g=x["g_head"], beta=x["beta"], w=x["w"])


def test_unqualified_kernels_are_not_auto_selected():
    """A kernel registered without auto=True runs only when requested by name."""
    x, _ = _inputs(13, T=16)
    calls = []

    def fake(q, k, v, g, **options):
        calls.append(options)
        return ref.gla(q, k, v, g, **options)

    ops.available("gla")  # load the kernels first: the lazy import would replace the fake with the real gla kernel
    previous = ops._REGISTRY["mosaic_gpu"].get("gla")
    try:
        ops.register("mosaic_gpu", "gla", fake)
        ops.linear_attention("gla", x["q"], x["k"], x["v"], g=x["g_key"])
        assert not calls  # "auto" skipped it and used the chunked path
        ops.linear_attention("gla", x["q"], x["k"], x["v"], g=x["g_key"], impl="mosaic_gpu")
        assert len(calls) == 1
        ops.register("mosaic_gpu", "gla", fake, auto=True)
        ops.linear_attention("gla", x["q"], x["k"], x["v"], g=x["g_key"])
        assert len(calls) == 2
    finally:
        if previous is None:
            ops._REGISTRY["mosaic_gpu"].pop("gla", None)
        else:
            ops._REGISTRY["mosaic_gpu"]["gla"] = previous
