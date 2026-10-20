# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated Mosaic GPU Mamba-3 MIMO kernels (``jla.ops.mosaic_gpu.mamba3_mimo_kernels``).

The gate-row algebra test runs anywhere (plain JAX); the kernel tests need an H100 and compare
outputs, final states and gradients against ``jla.ops.reference.mamba3_mimo`` (itself checked
against an independent float64 loop in tests/test_mamba3_mimo.py).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import mamba3 as m3
from jla.ops.mosaic_gpu import mamba3_mimo as mimo
from jla.ops.mosaic_gpu import mamba3_mimo_kernels as mk

ON_HOPPER = m3._on_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")

F32, BF16 = jnp.float32, jnp.bfloat16
HIGHEST = lax.Precision.HIGHEST
NAMES = ("q", "k", "v", "g", "beta", "gamma")


def _inputs(seed: int, B=2, T=150, H=2, R=4, K=64, V=64, dt_max=0.1, spikes=False):
    """Mamba-3-like inputs: ``g = A dt``, trapezoid weights; q, k scaled so R ranks sum to O(1)."""
    ks = jax.random.split(jax.random.key(seed), 7)
    q, k = ((jax.random.normal(x, (B, T, H, R, K)) / math.sqrt(R)).astype(BF16) for x in ks[:2])
    v = jax.random.normal(ks[2], (B, T, H, R, V)).astype(BF16)
    dt = jnp.exp(jax.random.uniform(ks[3], (B, T, H), minval=math.log(1e-3), maxval=math.log(dt_max)))
    a = -jax.random.uniform(ks[4], (B, T, H), minval=1.0, maxval=16.0)
    g = a * dt
    if spikes:
        g = jnp.where(jax.random.bernoulli(ks[6], 0.05, g.shape), -60.0, g)
    lam = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H)))
    return dict(q=q, k=k, v=v, g=g, beta=(1.0 - lam) * dt * jnp.exp(g), gamma=lam * dt)


def _initial(seed: int, B: int, H: int, R: int, K: int, V: int) -> ref.Mamba3State:
    ks = jax.random.split(jax.random.key(seed), 3)
    return ref.Mamba3State(
        0.1 * jax.random.normal(ks[0], (B, H, K, V)),
        jax.random.normal(ks[1], (B, H, R, K)) / math.sqrt(R),
        jax.random.normal(ks[2], (B, H, R, V)),
    )


def _segments(B: int, T: int) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    rows[0, 20:70] = 1
    rows[0, 70:128] = 2  # a reset at a chunk boundary for every rank (64 / R divides 64)
    rows[0, 128:] = 3
    if B > 1:
        rows[1, 63:] = 5  # a reset on a chunk's last token for R <= 2
        rows[1, 100:130] = 0  # an id reused after a different one starts a new document
    return jnp.asarray(rows)


def _args(x):
    return tuple(x[n] for n in NAMES)


def _f32(x):
    return {n: a.astype(F32) for n, a in x.items()}


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _core_ref(x, rank, initial=None, segment_ids=None, scale=1.0):
    """The kernels' algebra in float32 on rank-major 64-row tiles: ``(o [B, T, H, R, V], final)``."""
    B, T, H, _, K = x["q"].shape
    V = x["v"].shape[-1]
    ct = m3.CHUNK // rank
    NC = -(-T // ct)
    gates = mk.prepare_gates(x["g"], x["beta"], x["gamma"], rank=rank, num_chunks=NC, segment_ids=segment_ids)
    h0 = jnp.zeros((B, H, K, V), F32) if initial is None else mk._initial_mid_state(initial, x["g"], x["beta"])

    def tiles(a):  # [B, T, H, R, D] -> [NC, B, H, R * ct, D] (row r * ct + i)
        a = jnp.pad(a.astype(F32), [(0, 0), (0, NC * ct - T), (0, 0), (0, 0), (0, 0)])
        a = a.reshape(B, NC, ct, H, rank, -1).transpose(1, 0, 3, 4, 2, 5)
        return a.reshape(NC, B, H, rank * ct, -1)

    tok = jnp.arange(m3.CHUNK) % ct
    same_token, earlier = tok[:, None] == tok[None, :], tok[None, :] < tok[:, None]

    def step(state, xs):
        qn, kn, vn, gn = xs
        e, a, c, gam, qs, w, run, run_col = (gn[..., r, :] for r in (
            m3.ROW_E, m3.ROW_A, m3.ROW_C, m3.ROW_GAMMA, m3.ROW_QS, m3.ROW_W, m3.ROW_RUN, m3.ROW_RUN_COL))
        dec = gn[..., m3.ROW_DEC, :1]
        mask = earlier & (run[..., :, None] == run_col[..., None, :])
        decay = jnp.exp(jnp.where(mask, e[..., :, None] - a[..., None, :], -jnp.inf))
        weight = jnp.where(same_token, gam[..., :, None], decay * c[..., None, :])
        scores = jnp.einsum("bhik,bhjk->bhij", qn, kn, precision=HIGHEST)
        o = qs[..., None] * jnp.einsum("bhik,bhkv->bhiv", qn, state, precision=HIGHEST)
        o = o + jnp.einsum("bhij,bhjv->bhiv", scores * weight, vn, precision=HIGHEST)
        new = dec[..., None] * state + jnp.einsum("bhjk,bhjv->bhkv", kn * w[..., None], vn, precision=HIGHEST)
        return new, o

    final, o = lax.scan(step, h0, (tiles(x["q"]), tiles(x["k"]), tiles(x["v"]), jnp.moveaxis(gates, 2, 0)))
    o = o.reshape(NC, B, H, rank, ct, V).transpose(1, 0, 4, 2, 3, 5).reshape(B, NC * ct, H, rank, V)
    return o[:, :T] * scale, final


# ------------------------------------------------------------------ algebra (plain JAX, no kernels)


@pytest.mark.parametrize("rank", [1, 2, 4, 8])
@pytest.mark.parametrize("packed", [False, True])
def test_gate_rows_and_tile_algebra_reproduce_the_reference(rank, packed):
    """``prepare_gates`` + the kernels' rank-major tile algebra (f32) equal the token recurrence."""
    x = _f32(_inputs(0, B=2, T=150, H=2, R=rank, K=16, V=8, dt_max=1.0, spikes=True))
    initial = _initial(1, 2, 2, rank, 16, 8)
    seg = _segments(2, 150) if packed else None
    o_ref, s_ref = ref.mamba3_mimo(*_args(x), initial_state=initial, output_final_state=True, segment_ids=seg,
                                   scale=0.7)
    o, final = _core_ref(x, rank, initial, seg, scale=0.7)
    _assert_close(o, o_ref, 1e-4, "o")
    _assert_close(final, s_ref.ssm, 1e-4, "final")


def test_gate_rows_have_no_overflow_for_huge_decays():
    x = _f32(_inputs(2, B=1, T=128, H=1, R=4, K=16, V=8, dt_max=1.0, spikes=True))
    x["g"] = x["g"].at[0, 5:70, 0].set(-300.0)
    gates = mk.prepare_gates(x["g"], x["beta"], x["gamma"], rank=4, num_chunks=8)
    assert np.isfinite(np.asarray(gates)).all()
    assert float(jnp.max(gates[:, :, :, m3.ROW_QS])) <= 1.0 and float(jnp.max(gates[:, :, :, m3.ROW_DEC])) <= 1.0


def test_supports_and_registration():
    x = _inputs(3, T=64)
    if ON_HOPPER:
        assert mk.supports(*_args(x)) and mimo.supports(*_args(x))
        assert mimo.supports(*_args(x), chunk_size=16) and mimo.supports(*_args(x), kernel="siso")
    assert not mk.supports(*_args(x), chunk_size=64)  # chunks are 64 / R = 16 tokens at R = 4
    assert not mk.supports(x["q"].astype(F32), *_args(x)[1:])  # bf16 only
    assert not mk.supports(*_args(_inputs(3, T=64, R=3)))  # 64 / 3 is not a whole tile
    assert not mk.supports(*_args(_inputs(3, T=64, K=32)))
    assert not mimo.supports(*_args(x), kernel="other")
    assert not mk.supports(*_args(x), cu_seqlens=jnp.array([0, 64]))
    assert ops._REGISTRY["mosaic_gpu"]["mamba3_mimo"][0] is mimo.mamba3_mimo
    assert ops._REGISTRY["mosaic_gpu"]["mamba3_mimo"][2] is True  # impl="auto"
    with pytest.raises(ValueError, match="kernel must be one of"):
        mimo.mamba3_mimo(*_args(x), kernel="other")
    with pytest.raises(ValueError, match="ranks"):
        mk.mamba3_mimo(*_args(_inputs(3, T=64, R=3)))


# ------------------------------------------------------------------------- GPU: end to end

E2E_CASES = [
    # rank, (K, V), T, initial state, segments; T % (64 / R) != 0 for the odd lengths
    (2, (64, 64), 150, False, False),
    (2, (128, 64), 150, True, True),
    (4, (64, 128), 150, True, True),
    (4, (128, 64), 256, False, False),
    (4, (64, 64), 37, True, False),
    (1, (128, 64), 100, True, True),
    (8, (64, 64), 100, True, True),
]


@gpu
@pytest.mark.parametrize("rank,dims,T,with_initial,packed", E2E_CASES)
def test_end_to_end_matches_reference(rank, dims, T, with_initial, packed):
    K, V = dims
    x = _inputs(20, T=T, R=rank, K=K, V=V)
    initial = _initial(21, 2, 2, rank, K, V) if with_initial else None
    kw = dict(initial_state=initial, output_final_state=True, segment_ids=_segments(2, T) if packed else None)
    o, s = ops.linear_attention("mamba3_mimo", x["q"], x["k"], x["v"], g=x["g"], beta=x["beta"], gamma=x["gamma"],
                                impl="mosaic_gpu", **kw)
    o_ref, s_ref = ref.mamba3_mimo(*_args(_f32(x)), **kw)
    assert o.dtype == BF16 and o.shape == x["v"].shape and s.ssm.dtype == F32
    _assert_close(o, o_ref, 2e-2, "o")
    for got_leaf, want_leaf, name in zip(s, s_ref, ("ssm", "key", "value")):
        _assert_close(got_leaf, want_leaf, 2e-2, f"state.{name}")


@gpu
def test_auto_dispatch_runs_the_kernel_and_falls_back_when_unsupported():
    x = _inputs(26, T=100, R=4)
    seg = _segments(2, 100)
    kw = dict(g=x["g"], beta=x["beta"], gamma=x["gamma"], segment_ids=seg)
    o_auto, _ = ops.linear_attention("mamba3_mimo", x["q"], x["k"], x["v"], impl="auto", chunk_size=16, **kw)
    o_kernel, _ = mimo.mamba3_mimo(*_args(x), segment_ids=seg)
    np.testing.assert_array_equal(np.asarray(o_auto, np.float32), np.asarray(o_kernel, np.float32))
    # A chunk the kernel does not use (32 tokens at R = 4) goes to the chunked path.
    o_chunk, _ = ops.linear_attention("mamba3_mimo", x["q"], x["k"], x["v"], impl="auto", chunk_size=32, **kw)
    o_want, _ = ops.linear_attention("mamba3_mimo", x["q"], x["k"], x["v"], impl="chunked", chunk_size=32, **kw)
    np.testing.assert_array_equal(np.asarray(o_chunk, np.float32), np.asarray(o_want, np.float32))


@gpu
def test_matches_the_siso_route_and_strong_decays_stay_finite():
    x = _inputs(22, T=256, R=4, K=128, V=64, dt_max=1.0, spikes=True)
    o, s = mimo.mamba3_mimo(*_args(x), output_final_state=True)
    o_siso, s_siso = mimo.mamba3_mimo(*_args(x), output_final_state=True, kernel="siso")
    o_ref, s_ref = ref.mamba3_mimo(*_args(_f32(x)), output_final_state=True)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s.ssm, s_ref.ssm, 2e-2, "ssm")
    _assert_close(o, o_siso, 2e-2, "o vs siso route")
    _assert_close(s.ssm, s_siso.ssm, 2e-2, "ssm vs siso route")


@gpu
def test_state_chains_across_calls():
    x = _inputs(23, T=320, R=2)
    whole, s_whole = mimo.mamba3_mimo(*_args(x), output_final_state=True)
    first = {n: a[:, :130] for n, a in x.items()}
    second = {n: a[:, 130:] for n, a in x.items()}
    o1, s_mid = mimo.mamba3_mimo(*_args(first), output_final_state=True)
    o2, s_end = mimo.mamba3_mimo(*_args(second), initial_state=s_mid, output_final_state=True)
    _assert_close(jnp.concatenate([o1, o2], axis=1), whole, 2e-2, "o")
    _assert_close(s_end.ssm, s_whole.ssm, 2e-2, "ssm")


@gpu
@pytest.mark.parametrize("stages", [1, 3, 4])
def test_pipeline_depth_does_not_change_results(stages):
    x = _inputs(24, T=320, R=4)
    o2, s2 = mimo.mamba3_mimo(*_args(x), output_final_state=True, stages=2)
    o, s = mimo.mamba3_mimo(*_args(x), output_final_state=True, stages=stages)
    np.testing.assert_array_equal(np.asarray(o, np.float32), np.asarray(o2, np.float32))
    np.testing.assert_array_equal(np.asarray(s.ssm), np.asarray(s2.ssm))


@gpu
def test_scale_option():
    x = _inputs(25, T=128, R=2)
    o, _ = mimo.mamba3_mimo(*_args(x), scale=0.125)
    o_ref, _ = ref.mamba3_mimo(*_args(_f32(x)), scale=0.125)
    _assert_close(o, o_ref, 2e-2, "o")


# --------------------------------------------------------------------------- GPU: gradients

GRAD_CASES = [
    # rank, (K, V), T, segments
    (2, (64, 64), 150, True),
    (2, (128, 64), 200, False),
    (4, (128, 64), 150, True),
    (4, (64, 128), 200, False),
    (1, (64, 64), 150, True),
    (8, (64, 64), 150, True),
]


@gpu
@pytest.mark.parametrize("rank,dims,T,packed", GRAD_CASES)
def test_gradients_match_reference_autodiff(rank, dims, T, packed):
    K, V = dims
    x = _inputs(30, T=T, R=rank, K=K, V=V)
    initial = _initial(31, 2, 2, rank, K, V)
    seg = _segments(2, T) if packed else None
    ks = jax.random.split(jax.random.key(32), 4)
    cot = jax.random.normal(ks[0], x["v"].shape)
    c_ssm, c_key, c_val = (jax.random.normal(kk, a.shape) for kk, a in zip(ks[1:], initial))

    def loss(fn, inputs, init):
        o, s = fn(*_args(inputs), initial_state=init, output_final_state=True, segment_ids=seg)
        return (jnp.sum(o.astype(F32) * cot) + jnp.sum(s.ssm * c_ssm) + jnp.sum(s.key * c_key)
                + jnp.sum(s.value * c_val))

    got_x, got_s = jax.grad(lambda inp, init: loss(mimo.mamba3_mimo, inp, init), argnums=(0, 1))(x, initial)
    want_x, want_s = jax.grad(lambda inp, init: loss(ref.mamba3_mimo, inp, init), argnums=(0, 1))(_f32(x), initial)
    for n in NAMES:
        _assert_close(got_x[n], want_x[n], 3e-2, f"d{n}")
    for name, a, b in zip(("ssm", "key", "value"), got_s, want_s):
        _assert_close(a, b, 3e-2, f"d initial.{name}")


@gpu
def test_gradients_with_strong_decays_are_finite():
    x = _inputs(33, T=192, R=4, dt_max=1.0, spikes=True)

    def loss(fn, inp):
        return jnp.sum(fn(*_args(inp))[0].astype(F32) ** 2)

    got = jax.grad(lambda inp: loss(mimo.mamba3_mimo, inp))(x)
    want = jax.grad(lambda inp: loss(ref.mamba3_mimo, inp))(_f32(x))
    for n in NAMES:
        _assert_close(got[n], want[n], 3e-2, f"d{n}")


@gpu
def test_training_shape_compiles():
    """A Mamba-3 MIMO layer shape (B=4, T=2048, H=24, R=4, K=128, V=64) fits shared memory."""
    x = _inputs(34, B=4, T=2048, H=24, R=4, K=128, V=64)

    def loss(inp):
        return jnp.sum(mimo.mamba3_mimo(*_args(inp))[0].astype(F32))

    grads = jax.jit(jax.grad(loss))(x)
    assert all(np.isfinite(np.asarray(grads[n], np.float32)).all() for n in NAMES)
