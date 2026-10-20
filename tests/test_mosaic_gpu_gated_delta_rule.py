# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU scalar-gate kernels (simple GLA / Mamba-2 SSD, DeltaNet, Gated DeltaNet).

Each kernel stage against the matching ``jla.ops.chunked`` stage (or the
autodiff of the chunked stages for the backward kernels; the gate-free-inverse
backward kernels against the tested kernels of the previous path), the ops end to end
against ``jla.ops.reference``, gradients against autodiff of
``jla.ops.chunked``, packed documents (``segment_ids``) and the sequence
split (values and gradients against chunked), the Mamba-2 training shape under
remat, and the ``supports`` predicates and split policy. The kernel tests need
an H100; the ``supports`` and policy tests also run on CPU. The kernels cannot
run in the JAX 0.11.2 Mosaic GPU interpreter (no ``wgmma`` on an initialized
accumulator or with a register operand, no ``inline_mgpu``), so there are no
interpreter tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import chunked, reference as ref
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import gated_delta_rule as gdr
from jla.ops.mosaic_gpu import gated_delta_rule_bwd as gb
from jla.ops.mosaic_gpu import gated_delta_rule_fused_bwd as gfb
from jla.ops.mosaic_gpu import gated_delta_rule_kernels as gk
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu import scalar_gate_fused as sf
from jla.ops.mosaic_gpu import scalar_gate_kernels as sk

ON_HOPPER = cm.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")

F32, BF16 = jnp.float32, jnp.bfloat16
C = cm.CHUNK
HIGHEST = lax.Precision.HIGHEST
GATES = {"simple_gla": ("g",), "deltanet": ("beta",), "gated_deltanet": ("g", "beta")}


def _inputs(seed, B=1, T=128, H=2, K=128, V=128, Hk=None):
    """bf16 q/k/v (unit-norm k), float32 gates, an initial state and output cotangents."""
    Hk = Hk or H
    ks = jax.random.split(jax.random.key(seed), 9)
    return dict(
        q=jax.random.normal(ks[0], (B, T, Hk, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, Hk, K))).astype(BF16),
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


def _chunks(x):
    """User layout ``[B, T, H, ...]`` (T a multiple of the chunk) to float32 chunks ``[B, NC, H, C, ...]``."""
    return chunked._to_chunks(jnp.asarray(x, F32), C)


def _hm_chunks(x):
    """Head-major ``[B, H, T, D]`` to chunks ``[B, NC, H, C, D]``."""
    B, H, T, D = x.shape
    return jnp.moveaxis(x.astype(F32).reshape(B, H, T // C, C, D), 1, 2)


def _states(x):
    """Per-chunk transposed states ``[B, H, NC, V, K]`` to ``[B, NC, H, K, V]``."""
    return jnp.swapaxes(jnp.moveaxis(x.astype(F32), 2, 1), -1, -2)


def _kernel_gates(x):
    B, T, H = x["g"].shape
    return cm.chunk_log2_decay(x["g"], B, T, H, 0), cm.head_major(x["beta"], 0)


def _wy_oracle(x):
    """Chunked ``(t, w, u)`` from the bf16 inputs in float32."""
    kc, vc, bc = _chunks(x["k"]), _chunks(x["v"]), _chunks(x["beta"])
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    t = chunked.solve_unit_lower(chunked.chunk_scaled_dot_kkt(kc, bc, gc, precision=HIGHEST))
    w, u = chunked.recompute_w_u(kc, vc, bc, gc, t, precision=HIGHEST)
    return t, w, u


# ----------------------------------------------------------------------------- forward stages


@gpu
@pytest.mark.parametrize("value_dim", [128, 64])
def test_intra_fwd_matches_chunked_wy_factors(value_dim):
    x = _inputs(0, B=1, T=256, H=2, K=128, V=value_dim)
    g2, bt = _kernel_gates(x)
    t, w_neg, u = jax.jit(lambda *a: gk.intra_fwd(*a, None))(x["k"], x["v"], g2, bt)
    t_ref, w_ref, u_ref = _wy_oracle(x)
    _assert_close(_hm_chunks(t), t_ref, 4e-3, "T")
    _assert_close(_hm_chunks(w_neg), -w_ref, 1e-2, "w_neg")
    _assert_close(_hm_chunks(u), u_ref, 1e-2, "u")
    # The backward's recompute from the saved T reproduces the forward factors exactly.
    _, w2, u2 = jax.jit(lambda *a: gk.intra_fwd(*a))(x["k"], x["v"], g2, bt, t)
    np.testing.assert_array_equal(np.asarray(w2, np.float32), np.asarray(w_neg, np.float32))
    np.testing.assert_array_equal(np.asarray(u2, np.float32), np.asarray(u, np.float32))


@gpu
@pytest.mark.parametrize("delta", [False, True], ids=["simple_gla", "delta"])
def test_state_pass_and_fused_output_match_chunked(delta):
    x = _inputs(1, B=1, T=256, H=2, K=128, V=128)
    g2, bt = _kernel_gates(x)
    qc, kc = _chunks(x["q"]), _chunks(x["k"])
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    if delta:
        _, w_neg, u = jax.jit(lambda *a: gk.intra_fwd(*a, None))(x["k"], x["v"], g2, bt)
        wc, uc = -_hm_chunks(w_neg), _hm_chunks(u)
    else:
        w_neg, u = None, x["v"]
        wc, uc = None, _chunks(x["v"])
    h0t = jnp.swapaxes(x["h0"], -1, -2)
    h, vn, ht = jax.jit(lambda *a: sk.state_fwd(*a, u_head_major=delta, out_final=True))(x["k"], w_neg, u, g2, h0t)
    h_ref, vn_ref, ht_ref = chunked.chunk_state_pass(kc, wc, uc, gc, x["h0"], precision=HIGHEST)
    _assert_close(_states(h), h_ref, 1e-2, "h")
    _assert_close(jnp.swapaxes(ht, -1, -2), ht_ref, 1e-2, "final state")
    if delta:
        _assert_close(_hm_chunks(vn), vn_ref, 1e-2, "v_new")
    scale = 128 ** -0.5
    o, ht2 = jax.jit(lambda *a: sf.fused_fwd(*a, scale=scale, u_head_major=delta, out_final=True,
                                            out_dtype=BF16))(x["q"], x["k"], w_neg, u, g2, h0t)
    o_ref = chunked._from_chunks(chunked.chunk_output(qc, kc, vn_ref, h_ref, gc, scale, precision=HIGHEST))
    _assert_close(o, o_ref, 1e-2, "o")
    _assert_close(jnp.swapaxes(ht2, -1, -2), ht_ref, 1e-2, "fused final state")


# ---------------------------------------------------------------------------- backward stages


def _composition(delta, scale):
    """``loss(q, k, w, u, g_cum, h0)`` of the chunked state pass + output (``w`` is FLA's ``w``)."""

    def loss(q, k, w, u, gc, h0, do, dht):
        h, v_new, final = chunked.chunk_state_pass(k, w if delta else None, u, gc, h0, precision=HIGHEST)
        o = chunked.chunk_output(q, k, v_new, h, gc, scale, precision=HIGHEST)
        return jnp.sum(o * do) + jnp.sum(final * dht)

    return loss


@gpu
@pytest.mark.parametrize("delta", [False, True], ids=["simple_gla", "delta"])
def test_backward_state_stages_match_chunked_autodiff(delta):
    """dv_local + reverse state pass + dqkwg against autodiff of the chunked state pass and output."""
    x = _inputs(2, B=1, T=256, H=2, K=128, V=128)
    scale = 0.3
    g2, bt = _kernel_gates(x)
    if delta:
        _, w_neg, u = jax.jit(lambda *a: gk.intra_fwd(*a, None))(x["k"], x["v"], g2, bt)
        wc, uc = -_hm_chunks(w_neg), _hm_chunks(u)
    else:
        w_neg, u = None, x["v"]
        wc, uc = jnp.zeros_like(_chunks(x["q"])), _chunks(x["v"])
    do = x["do"].astype(BF16)
    h0t = jnp.swapaxes(x["h0"], -1, -2)
    dht_t = jnp.swapaxes(x["dht"], -1, -2)

    @jax.jit
    def kernels(q, k, w_neg, u, g2, h0t, do, dht_t):
        h, vn, _ = sk.state_fwd(k, w_neg, u, g2, h0t, u_head_major=delta, out_final=False)
        dvl = sk.dv_local(q, k, g2, do, scale=scale)
        dh, dvn, dh0 = sk.state_bwd(q, k, w_neg, g2, do, dvl, dht_t, scale=scale, want_dh0=True)
        if delta:
            dq, dk, dw, dgc = sk.dqkwg(q, k, vn, g2, h, dh, do, dvn, scale=scale, vn_head_major=True,
                                       dq_dtype=F32, dk_dtype=F32)
        else:
            dq, dk, dw, dgc = sk.dqkwg(q, k, u, g2, h, dh, do, None, scale=scale, vn_head_major=False,
                                       dq_dtype=F32, dk_dtype=F32)
        return dq, dk, dvn, dw, dgc, dh0

    dq, dk, dvn, dw, dgc, dh0 = kernels(x["q"], x["k"], w_neg, u, g2, h0t, do, dht_t)
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    grads = jax.grad(_composition(delta, scale), argnums=(0, 1, 2, 3, 4, 5))(
        _chunks(x["q"]), _chunks(x["k"]), wc, uc, gc, x["h0"], _chunks(do), x["dht"])
    dq_r, dk_r, dw_r, du_r, dg_r, dh0_r = grads
    _assert_close(_chunks(dq), dq_r, 1e-2, "dq")
    _assert_close(_chunks(dk), dk_r, 1e-2, "dk (state and output paths)")
    _assert_close(_chunks(dvn), du_r, 1e-2, "dv_new")
    _assert_close(jnp.swapaxes(dh0, -1, -2), dh0_r, 1e-2, "dh0")
    _assert_close(_hm_chunks(dgc[..., None])[..., 0], dg_r, 1e-2, "dG")
    if delta:
        _assert_close(_hm_chunks(dw), -dw_r, 1e-2, "dw_neg")


@gpu
def test_wy_bwd_matches_chunked_autodiff():
    x = _inputs(3, B=1, T=128, H=2, K=128, V=128)
    g2, bt = _kernel_gates(x)
    ks = jax.random.split(jax.random.key(30), 3)
    dw_neg = (0.1 * jax.random.normal(ks[0], (1, 2, 128, 128))).astype(BF16)  # head-major
    du = (0.1 * jax.random.normal(ks[1], (1, 128, 2, 128))).astype(BF16)  # user layout
    dk_part = jax.random.normal(ks[2], (1, 128, 2, 128))
    t, _, _ = jax.jit(lambda *a: gk.intra_fwd(*a, None))(x["k"], x["v"], g2, bt)
    dk, dv, dbeta, dgc = jax.jit(lambda *a: gk.wy_bwd(*a, dk_dtype=F32))(x["k"], x["v"], g2, bt, t, dw_neg, du,
                                                                          dk_part)

    def wy(k, v, beta, g):
        kc, vc, bc = _chunks(k), _chunks(v), _chunks(beta)
        gc = chunked.chunk_local_cumsum(_chunks(g))
        t = chunked.solve_unit_lower(chunked.chunk_scaled_dot_kkt(kc, bc, gc, precision=HIGHEST))
        return chunked.recompute_w_u(kc, vc, bc, gc, t, precision=HIGHEST)

    _, vjp = jax.vjp(wy, x["k"].astype(F32), x["v"].astype(F32), x["beta"], x["g"])
    dk_r, dv_r, db_r, dg_r = vjp((-_hm_chunks(dw_neg), _chunks(du)))
    dg = lax.cumsum(dgc.reshape(1, 2, 2, C), axis=3, reverse=True).reshape(1, 2, 128)
    _assert_close(dk - dk_part, dk_r, 1e-2, "dk")
    _assert_close(dv, dv_r, 1e-2, "dv")
    _assert_close(jnp.swapaxes(dbeta, 1, 2), db_r, 1e-2, "dbeta")
    _assert_close(jnp.swapaxes(dg, 1, 2), dg_r, 1e-2, "dg")


# ------------------------------------------------- gate-free inverse (FlashQLA-style) stages


@gpu
@pytest.mark.parametrize("key_dim", [128, 64])
def test_kkt_fwd_gate_free_inverse_and_decay(key_dim):
    x = _inputs(6, B=1, T=256, H=2, K=key_dim, V=64)
    g2, bt = _kernel_gates(x)
    t0 = jax.jit(lambda k, b: gk.kkt_fwd(k, b))(x["k"], bt)
    kc, bc = _chunks(x["k"]), _chunks(x["beta"])
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    t0_ref = chunked.solve_unit_lower(chunked.chunk_scaled_dot_kkt(kc, bc, jnp.zeros_like(gc), precision=HIGHEST))
    t_ref = chunked.solve_unit_lower(chunked.chunk_scaled_dot_kkt(kc, bc, gc, precision=HIGHEST))
    _assert_close(_hm_chunks(t0), t0_ref, 4e-3, "T0")
    _assert_close(_hm_chunks(gk.decay_inverse(t0, g2)), t_ref, 4e-3, "T = T0 * exp(G_i - G_j)")


@gpu
def test_fused_fwd_wy_matches_chunked():
    x = _inputs(7, B=2, T=512, H=2, K=128, V=128)  # 8 chunks: twice the 4-stage input ring
    g2, bt = _kernel_gates(x)
    qc, kc = _chunks(x["q"]), _chunks(x["k"])
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    t_ref, w_ref, u_ref = _wy_oracle(x)
    h_ref, vn_ref, ht_ref = chunked.chunk_state_pass(kc, w_ref, u_ref, gc, x["h0"], precision=HIGHEST)
    scale = 128 ** -0.5
    o_ref = chunked._from_chunks(chunked.chunk_output(qc, kc, vn_ref, h_ref, gc, scale, precision=HIGHEST))
    t0 = gk.kkt_fwd(x["k"], bt)
    o, ht = jax.jit(lambda *a: sf.fused_fwd_wy(*a, scale=scale, out_final=True, out_dtype=BF16))(
        x["q"], x["k"], x["v"], t0, g2, bt, jnp.swapaxes(x["h0"], -1, -2))
    _assert_close(o, o_ref, 1.5e-2, "o")
    _assert_close(jnp.swapaxes(ht, -1, -2), ht_ref, 1e-2, "final state")


@gpu
def test_state_fwd_wy_states_and_piece_maps():
    """The prepare-h pass against chunked, and the augmented pass's maps rebuild the piece starts."""
    x = _inputs(8, B=2, T=512, H=2, K=128, V=128)
    g2, bt = _kernel_gates(x)
    kc = _chunks(x["k"])
    gc = chunked.chunk_local_cumsum(_chunks(x["g"]))
    _, w_ref, u_ref = _wy_oracle(x)
    h_ref, _, _ = chunked.chunk_state_pass(kc, w_ref, u_ref, gc, x["h0"], precision=HIGHEST)
    t0 = gk.kkt_fwd(x["k"], bt)
    h0t = jnp.swapaxes(x["h0"], -1, -2)
    h, _ = jax.jit(lambda *a: gb.state_fwd_wy(*a))(x["k"], x["v"], t0, g2, bt, h0t)
    _assert_close(_states(h), h_ref, 1e-2, "h")
    cp = 4
    _, end = jax.jit(lambda *a: gb.state_fwd_wy(*a, None, store_states=False, out_final=True, aug=True, cp=cp))(
        x["k"], x["v"], t0, g2, bt)
    end = end.reshape(2, cp, 2, 256, 128)
    s = h0t
    for i in range(1, cp):  # S_start(i) = S_start(i - 1) M + E (transposed)
        s = jnp.einsum("bhvk,bhkj->bhvj", s, end[:, i - 1, :, :128], precision=HIGHEST) + end[:, i - 1, :, 128:]
        _assert_close(jnp.swapaxes(s, -1, -2), h_ref[:, 2 * i], 1e-2, f"start of piece {i}")


@gpu
@pytest.mark.parametrize("segments", [False, True], ids=["dense", "segments"])
def test_gate_free_backward_kernels_match_previous_path(segments):
    """state_fwd_wy + state_bwd_wy + chunk_grads against the materialized-WY kernels (tested above)."""
    x = _inputs(9, B=2, T=512, H=2, K=128, V=128)
    g2, bt = _kernel_gates(x)
    seg = cm.segment_vectors(_segment_ids(2, 512, "tiny"), 0) if segments else None
    q, k, v, do = x["q"], x["k"], x["v"], x["do"].astype(BF16)
    scale = 128 ** -0.5

    @jax.jit
    def previous(q, k, v, g2, bt, do):
        t, w, u = gk.intra_fwd(k, v, g2, bt, None, seg=seg)
        h, vn, _ = sk.state_fwd(k, w, u, g2, None, u_head_major=True, out_final=False, seg=seg)
        dvl = sk.dv_local(q, k, g2, do, scale=scale, seg=seg)
        dh, dvn, _ = sk.state_bwd(q, k, w, g2, do, dvl, None, scale=scale, want_dh0=False, seg=seg)
        dq, dk_part, dw, dgc = sk.dqkwg(q, k, vn, g2, h, dh, do, dvn, scale=scale, vn_head_major=True,
                                       dq_dtype=BF16, dk_dtype=F32, seg=seg)
        dk, dv, db, dg2 = gk.wy_bwd(k, v, g2, bt, t, dw, dvn, dk_part, dk_dtype=BF16, seg=seg)
        return h, dh, dq, dk, dv, db, dgc + dg2

    @jax.jit
    def gate_free(q, k, v, g2, bt, do):
        t0 = gk.kkt_fwd(k, bt, seg=seg)
        h, _ = gb.state_fwd_wy(k, v, t0, g2, bt, None, seg=seg)
        dh, dv, _ = gb.state_bwd_wy(q, k, do, t0, g2, bt, None, scale=scale, want_dh0=False, seg=seg)
        dq, dk, db, dg = gb.chunk_grads(q, k, v, do, t0, g2, bt, h, dh, scale=scale, seg=seg)
        return h, dh, dq, dk, dv, db, dg

    want = previous(q, k, v, g2, bt, do)
    got = gate_free(q, k, v, g2, bt, do)
    for name, a, b in zip(("h", "dh", "dq", "dk", "dv", "dbeta", "dG"), got, want):
        _assert_close(a, b, 2e-2, name)


@gpu
def test_previous_delta_path_matches_gate_free_path():
    """The kept materialized-WY path (``wy=False``) against the default, values and gradients."""
    x = _inputs(10, B=1, T=320, H=2, K=128, V=128)

    def loss(wy, p):
        o, s = sg.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None, initial_state=p["h0"],
                      output_final_state=True, wy=wy)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])

    params = {n: x[n] for n in ("q", "k", "v", "g", "beta", "h0")}
    want = jax.jit(jax.grad(lambda p: loss(False, p)))(params)
    got = jax.jit(jax.grad(lambda p: loss(True, p)))(params)
    for name in params:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


# ------------------------------------------------------------------------------- end to end

E2E = [  # kind, B, T, H, K, V, Hk, with_h0, scale
    ("simple_gla", 2, 256, 3, 128, 128, None, True, None),
    ("simple_gla", 2, 200, 3, 128, 64, None, False, 1.0),
    ("simple_gla", 1, 130, 4, 64, 64, 2, True, None),
    ("simple_gla", 1, 100, 4, 128, 64, 1, True, 1.0),
    ("simple_gla", 1, 128, 2, 64, 128, None, False, None),
    ("gated_deltanet", 2, 256, 3, 128, 128, None, True, None),
    ("gated_deltanet", 2, 200, 3, 128, 64, None, False, None),
    ("gated_deltanet", 1, 130, 4, 64, 64, 2, True, None),
    ("gated_deltanet", 1, 128, 2, 64, 128, None, True, None),
    ("deltanet", 1, 192, 2, 128, 128, None, True, None),
    ("deltanet", 1, 100, 2, 64, 64, None, False, None),
]


def _call(kind, impl, x, *, h0, scale, final=True, kv_heads=None):
    q, k, v = x["q"], x["k"], x["v"]
    if impl != "mosaic_gpu":
        if kv_heads is not None and kv_heads != v.shape[2]:
            reps = v.shape[2] // kv_heads
            q, k = jnp.repeat(q, reps, axis=2), jnp.repeat(k, reps, axis=2)
        q, k, v = (a.astype(F32) for a in (q, k, v))
    gates = {name: x[name] for name in GATES[kind]}
    extra = dict(precision=HIGHEST) if impl == "chunked" else {}
    return ops.linear_attention(kind, q, k, v, impl=impl, scale=scale, initial_state=h0,
                                output_final_state=final, **gates, **extra)


def _case_id(case):
    kind, B, T, H, K, V, Hk, h0, scale = case
    return f"{kind}-B{B}T{T}H{H}K{K}V{V}" + (f"-Hk{Hk}" if Hk else "") + ("-h0" if h0 else "") + (
        f"-s{scale}" if scale else "")


@gpu
@pytest.mark.parametrize("case", E2E, ids=_case_id)
def test_end_to_end_matches_reference(case):
    kind, B, T, H, K, V, Hk, with_h0, scale = case
    x = _inputs(4, B, T, H, K, V, Hk)
    h0 = x["h0"] if with_h0 else None
    o, s = jax.jit(lambda x, h0: _call(kind, "mosaic_gpu", x, h0=h0, scale=scale))(x, h0)
    o_ref, s_ref = _call(kind, "reference", x, h0=h0, scale=scale, kv_heads=Hk)
    assert o.dtype == BF16 and s.dtype == F32
    _assert_close(o, o_ref, 1.5e-2, "o")
    _assert_close(s, s_ref, 1e-2, "final state")


@gpu
@pytest.mark.parametrize("case", E2E, ids=_case_id)
def test_gradients_match_chunked_autodiff(case):
    kind, B, T, H, K, V, Hk, with_h0, scale = case
    x = _inputs(5, B, T, H, K, V, Hk)
    names = ("q", "k", "v") + GATES[kind] + (("h0",) if with_h0 else ())

    def loss(impl, params):
        y = dict(x, **params)
        o, s = _call(kind, impl, y, h0=y["h0"] if with_h0 else None, scale=scale, kv_heads=Hk)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])

    params = {name: x[name] for name in names}
    got = jax.jit(jax.grad(lambda p: loss("mosaic_gpu", p)))(params)
    want = jax.jit(jax.grad(lambda p: loss("chunked", p)))(params)
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
@pytest.mark.parametrize("kind", sorted(GATES))
def test_float16_with_initial_state_and_no_final_state(kind):
    x = _inputs(10, 1, 17, 2, 64, 64)
    x = dict(x, **{n: x[n].astype(jnp.float16) for n in ("q", "k", "v")})
    names = ("q", "k", "v", "h0") + GATES[kind]

    def loss(impl, params):
        o, s = _call(kind, impl, dict(x, **params), h0=params["h0"], scale=None, final=False)
        assert s is None
        return jnp.sum(o.astype(F32) * x["do"]), o

    params = {name: x[name] for name in names}
    (_, o), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(params)
    (_, o_ref), want = jax.jit(jax.value_and_grad(lambda p: loss("chunked", p), has_aux=True))(params)
    assert o.dtype == jnp.float16
    _assert_close(o, o_ref, 1e-2, "o")
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
def test_qk_l2norm_matches_reference():
    x = _inputs(6, 1, 128, 2, 128, 128)
    q, k = x["q"] * 3, x["k"] * 3
    o, _ = gdr.gated_delta_rule(q, k, x["v"], x["g"], x["beta"], use_qk_l2norm=True)
    o_ref, _ = ref.gated_delta_rule(q.astype(F32), k.astype(F32), x["v"].astype(F32), x["g"], x["beta"],
                                    use_qk_l2norm=True)
    _assert_close(o, o_ref, 1.5e-2, "o")


@gpu
def test_mamba2_training_shape_value_and_grad_under_remat():
    """Mamba-2 130M layer shape: d_state 128, head dim 64, 24 heads, one B/C group repeated, scale 1."""
    B, T, H, K, V = 8, 2048, 24, 128, 64
    x = _inputs(7, B, T, 1, K, V)
    q, k = jnp.repeat(x["q"], H, axis=2), jnp.repeat(x["k"], H, axis=2)
    ks = jax.random.split(jax.random.key(70), 3)
    v = jax.random.normal(ks[0], (B, T, H, V)).astype(BF16)
    g = -jax.nn.softplus(jax.random.normal(ks[1], (B, T, H)))
    do = jax.random.normal(ks[2], (B, T, H, V))

    @jax.checkpoint
    def core(q, k, v, g):
        return ops.linear_attention("simple_gla", q, k, v, g=g, impl="mosaic_gpu", scale=1.0)[0]

    loss, grads = jax.jit(jax.value_and_grad(lambda *a: jnp.sum(core(*a).astype(F32) * do), argnums=(0, 1, 2, 3)))(
        q, k, v, g)
    assert np.isfinite(float(loss))
    heads = slice(0, 2)  # spot-check two heads against chunked autodiff

    def chunked_loss(q, k, v, g):
        o, _ = ops.linear_attention("simple_gla", q.astype(F32), k.astype(F32), v.astype(F32), g=g,
                                    impl="chunked", scale=1.0)
        return jnp.sum(o * do[:, :, heads])

    want = jax.jit(jax.grad(chunked_loss, argnums=(0, 1, 2, 3)))(q[:, :, heads], k[:, :, heads], v[:, :, heads],
                                                                  g[:, :, heads])
    for name, a, b in zip("qkvg", grads, want):
        _assert_close(a[:, :, heads], b, 2e-2, f"d{name}")


# ------------------------------------------------- packed documents and the sequence split


def _segment_ids(B: int, T: int, pattern: str) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    if pattern == "edges":
        rows[0, 64:] = 1  # a document starting exactly at a chunk edge
        rows[0, 128:] = 2
        rows[0, 191:] = 3  # ... and one on a chunk's last token
        if B > 1:
            rows[1, 63:] = 5
            rows[1, 100:130] = 0  # an id reused after a different one starts a new document
            rows[1, 130:] = 5
    elif pattern == "tiny":  # many short documents (mean length 5)
        rng = np.random.default_rng(0)
        for b in range(B):
            starts = np.sort(rng.choice(np.arange(1, T), size=T // 5, replace=False))
            rows[b] = np.searchsorted(starts, np.arange(T), side="right")
    return jnp.asarray(rows)


def _check_against_chunked(kind, x, *, segment_ids, cp, tol=2e-2):
    """Values, final state and every gradient of ``scalar_gate.run`` against chunked autodiff."""
    names = ("q", "k", "v") + GATES[kind] + ("h0",)

    def mosaic(p):
        return sg.run(p["q"], p["k"], p["v"], p.get("g"), p.get("beta"), delta=kind != "simple_gla", scale=None,
                      initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids, cp=cp)

    def chunked(p):
        gates = {n: p[n] for n in GATES[kind]}
        return ops.linear_attention(kind, *(p[n].astype(F32) for n in ("q", "k", "v")), impl="chunked",
                                    initial_state=p["h0"], output_final_state=True, segment_ids=segment_ids,
                                    precision=HIGHEST, **gates)

    def loss(f, p):
        o, s = f(p)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    params = {n: x[n] for n in names}
    (_, (o, s)), got = jax.jit(jax.value_and_grad(lambda p: loss(mosaic, p), has_aux=True))(params)
    (_, (o_ref, s_ref)), want = jax.jit(jax.value_and_grad(lambda p: loss(chunked, p), has_aux=True))(params)
    _assert_close(o, o_ref, tol, "o")
    _assert_close(s, s_ref, tol, "final state")
    for name in names:
        _assert_close(got[name], want[name], tol, f"d{name}")


@gpu
@pytest.mark.parametrize("cp", [1, 4])
@pytest.mark.parametrize("pattern", ["edges", "tiny"])
@pytest.mark.parametrize("kind", sorted(GATES))
def test_segment_ids_match_chunked(kind, pattern, cp):
    """Packed documents: the initial state reaches only each row's first document."""
    x = _inputs(12, 2, 512, 2, 128, 128)
    _check_against_chunked(kind, x, segment_ids=_segment_ids(2, 512, pattern), cp=cp)


@gpu
@pytest.mark.parametrize("cp", [2, 4])
@pytest.mark.parametrize("kind", sorted(GATES))
def test_sequence_split_matches_chunked(kind, cp):
    x = _inputs(13, 1, 512, 2, 128, 64)
    _check_against_chunked(kind, x, segment_ids=None, cp=cp)


@gpu
@pytest.mark.parametrize("kind", sorted(GATES))
def test_sequence_split_with_ragged_length_and_segments(kind):
    x = _inputs(14, 1, 300, 3, 64, 64)  # 300 tokens are padded to 2 pieces of 192
    _check_against_chunked(kind, x, segment_ids=_segment_ids(1, 300, "tiny"), cp=2)


def test_pieces_fill_the_gpu(monkeypatch):
    monkeypatch.setattr(sg, "num_sms", lambda: 132)
    assert sg.pieces(1, 8, 128, 4096, delta=False) == 8  # 16 CTAs -> one wave of 128
    assert sg.pieces(1, 16, 128, 16384, delta=False) == 4
    assert sg.pieces(1, 8, 128, 4096, delta=True) == 4  # more than 4 pieces need 32 chunks each
    assert sg.pieces(1, 8, 128, 16384, delta=True) == 8  # the delta rule stops at 8 pieces
    assert sg.pieces(1, 16, 128, 16384, delta=True) == 4
    assert sg.pieces(1, 2, 128, 65536, delta=True) == 8
    assert sg.pieces(1, 48, 64, 16384, delta=False) == 1  # more than 32 CTAs: no split
    assert sg.pieces(8, 32, 128, 2048, delta=False) == 1
    assert sg.pieces(1, 8, 128, 512, delta=False) == 1  # pieces would be shorter than 8 chunks
    assert sg.pieces(1, 8, 128, 1024, delta=False) == 2


# ------------------------------------------------------------------------------ supports


def _call_args(kind, x):
    return (x["q"], x["k"], x["v"]) + tuple(x[name] for name in GATES[kind])


SUPPORTS = {"simple_gla": sg.supports_simple_gla, "deltanet": gdr.supports_delta_rule,
            "gated_deltanet": gdr.supports_gated_delta_rule}


@pytest.mark.parametrize("kind", sorted(GATES))
def test_supports_rejects_unsupported_calls(kind, monkeypatch):
    monkeypatch.setattr(sg, "is_hopper", lambda: True)  # check the shape/option rules on any backend
    supports = SUPPORTS[kind]
    x = _inputs(8, 1, 100, 2, 128, 64)
    args = _call_args(kind, x)
    assert supports(*args)
    assert supports(*args, scale=1.0, initial_state=x["h0"], output_final_state=True, chunk_size=64)
    assert supports(*_call_args(kind, _inputs(8, 1, 100, 4, 64, 64, Hk=2)))  # grouped value heads
    assert not supports(*args, cu_seqlens=jnp.array([0, 50, 100]))
    assert supports(*args, segment_ids=jnp.zeros((1, 100), jnp.int32))
    assert not supports(*args, segment_ids=jnp.zeros((1, 100), F32))  # ids must be integers
    assert not supports(*args, segment_ids=jnp.zeros((1, 99), jnp.int32))
    assert not supports(*args, chunk_size=32)
    assert not supports(*args, precision=HIGHEST)
    assert not supports(*args, unknown_option=1)
    assert not supports(*args, initial_state=x["h0"][..., :32])
    assert supports(*args, use_qk_l2norm=True) == (kind != "simple_gla")
    f32 = dict(x, q=x["q"].astype(F32), k=x["k"].astype(F32), v=x["v"].astype(F32))
    assert not supports(*_call_args(kind, f32))
    assert not supports(*_call_args(kind, dict(x, v=x["v"].astype(jnp.float16))))  # mixed dtypes
    assert not supports(*_call_args(kind, _inputs(8, 1, 100, 2, 96, 64)))  # head dim 96
    assert not supports(*_call_args(kind, _inputs(8, 1, 100, 3, 64, 64, Hk=2)))  # 3 heads over 2 key heads
    assert not supports(*_call_args(kind, dict(x, g=x["g"][:, :50], beta=x["beta"][:, :50])))


@pytest.mark.skipif(ON_HOPPER, reason="checks the fallback without an H100")
@pytest.mark.parametrize("kind", sorted(GATES))
def test_without_hopper_supports_is_false_and_auto_falls_back(kind):
    x = _inputs(9, 1, 64, 1, 64, 64)
    args = _call_args(kind, x)
    assert not SUPPORTS[kind](*args)
    with pytest.raises(ValueError, match="Hopper"):
        ops.linear_attention(kind, *args[:3], impl="mosaic_gpu", **{n: x[n] for n in GATES[kind]})
    o, _ = ops.linear_attention(kind, *args[:3], **{n: x[n] for n in GATES[kind]})  # auto: chunked
    o_ref, _ = ops.linear_attention(kind, *args[:3], impl="chunked", **{n: x[n] for n in GATES[kind]})
    np.testing.assert_array_equal(np.asarray(o, np.float32), np.asarray(o_ref, np.float32))


@pytest.mark.parametrize("kind", sorted(GATES))
def test_registered_under_mosaic_gpu(kind):
    ops.available(kind)  # loads the kernels
    assert "mosaic_gpu" in ops.available(kind)


@gpu
@pytest.mark.parametrize("kind", sorted(GATES))
def test_auto_picks_the_kernels_on_hopper(kind):
    x = _inputs(11, 1, 64, 2, 64, 64)
    args, gates = (x["q"], x["k"], x["v"]), {n: x[n] for n in GATES[kind]}
    o_auto, _ = jax.jit(lambda a, g: ops.linear_attention(kind, *a, **g))(args, gates)
    o_kernel, _ = jax.jit(lambda a, g: ops.linear_attention(kind, *a, impl="mosaic_gpu", **g))(args, gates)
    np.testing.assert_array_equal(np.asarray(o_auto, np.float32), np.asarray(o_kernel, np.float32))
    seg = jnp.zeros((1, 64), jnp.int32).at[:, 40:].set(1)
    o_seg, _ = ops.linear_attention(kind, *args, segment_ids=seg, **gates)
    o_seg_kernel, _ = ops.linear_attention(kind, *args, impl="mosaic_gpu", segment_ids=seg, **gates)
    np.testing.assert_array_equal(np.asarray(o_seg, np.float32), np.asarray(o_seg_kernel, np.float32))


# ----------------------------------------------------- fused backward (one CTA per (b, h))

FUSED = [  # head dim, key heads, segment pattern
    (128, None, None),
    (128, None, "tiny"),
    (64, 1, None),
    (64, None, "edges"),
]


@gpu
@pytest.mark.parametrize("case", FUSED, ids=lambda c: f"d{c[0]}-hk{c[1]}-{c[2]}")
def test_fused_backward_matches_split_kernels(case):
    """gated_delta_rule_fused_bwd against state_bwd_wy + chunk_grads, with dS_T and dS_0."""
    dim, kv_heads, pattern = case
    x = _inputs(15, B=2, T=448, H=2, K=dim, V=dim, Hk=kv_heads)
    g2, bt = _kernel_gates(x)
    seg = None if pattern is None else cm.segment_vectors(_segment_ids(2, 448, pattern), 0)
    q, k, v, do = x["q"], x["k"], x["v"], x["do"].astype(BF16)
    dht, h0t = jnp.swapaxes(x["dht"], -1, -2), jnp.swapaxes(x["h0"], -1, -2)
    scale = dim ** -0.5

    @jax.jit
    def both(q, k, v, g2, bt, do, dht, h0t):
        t0 = gk.kkt_fwd(k, bt, seg=seg)
        h, _ = gb.state_fwd_wy(k, v, t0, g2, bt, h0t, seg=seg)
        dh, dv, dh0 = gb.state_bwd_wy(q, k, do, t0, g2, bt, dht, scale=scale, want_dh0=True, seg=seg)
        dq, dk, db, dg = gb.chunk_grads(q, k, v, do, t0, g2, bt, h, dh, scale=scale, seg=seg)
        fused = gfb.fused_bwd(q, k, v, do, t0, g2, bt, h, dht, scale=scale, want_dh0=True, seg=seg)
        return (dq, dk, dv, db, dg, dh0), fused

    want, got = both(q, k, v, g2, bt, do, dht, h0t)
    for name, a, b in zip(("dq", "dk", "dv", "dbeta", "dG", "dS0"), got, want):
        _assert_close(a, b, 1e-2, name)
    # Without dS_T and dS_0.
    t0 = gk.kkt_fwd(k, bt, seg=seg)
    h, _ = gb.state_fwd_wy(k, v, t0, g2, bt, None, seg=seg)
    got = gfb.fused_bwd(q, k, v, do, t0, g2, bt, h, None, scale=scale, want_dh0=False, seg=seg)
    dh, dv, _ = gb.state_bwd_wy(q, k, do, t0, g2, bt, None, scale=scale, want_dh0=False, seg=seg)
    want = gb.chunk_grads(q, k, v, do, t0, g2, bt, h, dh, scale=scale, seg=seg)
    assert got[5] is None
    for name, a, b in zip(("dq", "dk", "dv", "dbeta", "dG"), got[:5], (want[0], want[1], dv, want[2], want[3])):
        _assert_close(a, b, 1e-2, f"{name} (no dS_T)")


def _check_fused_against_chunked(kind, x, *, segment_ids, kv_heads, fused_bwd, with_h0=True, tol=2e-2):
    """Values, final state and gradients of ``scalar_gate.run(..., fused_bwd=...)`` against chunked autodiff."""
    names = ("q", "k", "v") + GATES[kind] + (("h0",) if with_h0 else ())

    def mosaic(p):
        return sg.run(p["q"], p["k"], p["v"], p.get("g"), p.get("beta"), delta=True, scale=None,
                      initial_state=p.get("h0"), output_final_state=True, segment_ids=segment_ids,
                      fused_bwd=fused_bwd)

    def chunked(p):
        q, k = p["q"], p["k"]
        if kv_heads is not None:
            reps = p["v"].shape[2] // kv_heads
            q, k = jnp.repeat(q, reps, axis=2), jnp.repeat(k, reps, axis=2)
        gates = {n: p[n] for n in GATES[kind]}
        return ops.linear_attention(kind, q.astype(F32), k.astype(F32), p["v"].astype(F32), impl="chunked",
                                    initial_state=p.get("h0"), output_final_state=True, segment_ids=segment_ids,
                                    precision=HIGHEST, **gates)

    def loss(f, p):
        o, s = f(p)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    params = {n: x[n] for n in names}
    (_, (o, s)), got = jax.jit(jax.value_and_grad(lambda p: loss(mosaic, p), has_aux=True))(params)
    (_, (o_ref, s_ref)), want = jax.jit(jax.value_and_grad(lambda p: loss(chunked, p), has_aux=True))(params)
    _assert_close(o, o_ref, tol, "o")
    _assert_close(s, s_ref, tol, "final state")
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], tol, f"d{name}")


FUSED_E2E = [  # kind, B, T, H, head dim, key heads, segment pattern, initial state
    ("gated_deltanet", 2, 200, 3, 128, None, None, True),
    ("gated_deltanet", 2, 512, 2, 128, None, "edges", True),
    ("gated_deltanet", 1, 130, 4, 64, 2, None, False),
    ("gated_deltanet", 2, 300, 2, 64, None, "tiny", True),
    ("deltanet", 1, 192, 2, 128, None, None, True),
    ("deltanet", 1, 100, 2, 64, None, "tiny", False),
]


@gpu
@pytest.mark.parametrize("case", FUSED_E2E, ids=lambda c: "-".join(map(str, c)))
def test_fused_backward_gradients_match_chunked_autodiff(case):
    kind, B, T, H, dim, kv_heads, pattern, with_h0 = case
    x = _inputs(16, B, T, H, dim, dim, kv_heads)
    segment_ids = None if pattern is None else _segment_ids(B, T, pattern)
    _check_fused_against_chunked(kind, x, segment_ids=segment_ids, kv_heads=kv_heads, fused_bwd=True,
                                 with_h0=with_h0)


@gpu
def test_default_backward_stays_split_when_batch_times_heads_fills_the_gpu(monkeypatch):
    """At ``B * H >= #SMs`` the backward stays split: the fused kernel is opt-in (``run``'s ``fused_bwd=False``)."""

    def refuse(*args, **kwargs):
        raise AssertionError("the default backward called the fused kernel")

    monkeypatch.setattr(sg.gfb, "fused_bwd", refuse)
    B, T, dim = 2, 130, 64
    H = -(-sg.num_sms() // B)
    x = _inputs(17, B, T, H, dim, dim)
    _check_fused_against_chunked("gated_deltanet", x, segment_ids=None, kv_heads=None, fused_bwd=False)


@gpu
@pytest.mark.parametrize("kind", ["gated_deltanet", "deltanet"])
def test_fused_backward_float16_one_partial_chunk(kind):
    x = _inputs(19, 1, 17, 2, 64, 64)
    x = dict(x, **{n: x[n].astype(jnp.float16) for n in ("q", "k", "v")})
    _check_fused_against_chunked(kind, x, segment_ids=None, kv_heads=None, fused_bwd=True)


def test_fused_backward_policy():
    """``fused_bwd=True`` fails loudly where the fused kernel does not apply (checked before any kernel runs)."""
    x = _inputs(18, 1, 64, 1, 64, 64)
    common = dict(scale=None, initial_state=None, output_final_state=False, fused_bwd=True)
    with pytest.raises(ValueError, match="fused_bwd"):  # the materialized WY factors
        sg.run(x["q"], x["k"], x["v"], x["g"], x["beta"], delta=True, wy=False, **common)
    with pytest.raises(ValueError, match="fused_bwd"):  # no delta rule
        sg.run(x["q"], x["k"], x["v"], x["g"], None, delta=False, **common)
    with pytest.raises(ValueError, match="fused_bwd"):  # a sequence split
        sg.run(x["q"], x["k"], x["v"], x["g"], x["beta"], delta=True, cp=2, **common)
    y = _inputs(18, 1, 64, 1, 64, 128)
    with pytest.raises(ValueError, match="fused_bwd"):  # K != V
        sg.run(y["q"], y["k"], y["v"], y["g"], y["beta"], delta=True, **common)
