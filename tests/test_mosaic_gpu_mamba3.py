# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU Mamba-3 kernels: kernel stages vs the chunked oracle, end to end vs reference and
chunked, gradients vs chunked autodiff, and dispatch.

The algebra test and the GPU-interpreter tests run on CPU (``JAX_PLATFORMS=cpu``); the rest needs an H100.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import chunked, reference as ref
from jla.ops.mosaic_gpu import mamba3 as m3

ON_HOPPER = m3._on_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
cpu = pytest.mark.skipif(jax.default_backend() != "cpu", reason="GPU-interpreter test (run with JAX_PLATFORMS=cpu)")

F32, BF16 = jnp.float32, jnp.bfloat16
HIGHEST = lax.Precision.HIGHEST


def _inputs(seed: int, B=2, T=200, H=3, K=128, V=64, dt_max=0.1, spikes=False):
    """Mamba-3-like inputs: unit-RMS q/k, ``g = A dt``, trapezoid weights; q, k, v rounded to bf16."""
    ks = jax.random.split(jax.random.key(seed), 7)
    q, k = (jax.random.normal(x, (B, T, H, K)).astype(BF16) for x in ks[:2])
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    dt = jnp.exp(jax.random.uniform(ks[3], (B, T, H), minval=math.log(1e-3), maxval=math.log(dt_max)))
    a = -jax.random.uniform(ks[4], (B, T, H), minval=1.0, maxval=16.0)
    g = a * dt
    if spikes:
        g = jnp.where(jax.random.bernoulli(ks[6], 0.05, g.shape), -60.0, g)
    lam = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H)))
    return dict(q=q, k=k, v=v, g=g, beta=(1.0 - lam) * dt * jnp.exp(g), gamma=lam * dt)


def _initial(seed: int, B: int, H: int, K: int, V: int) -> ref.Mamba3State:
    ks = jax.random.split(jax.random.key(seed), 3)
    return ref.Mamba3State(
        0.1 * jax.random.normal(ks[0], (B, H, K, V)),
        jax.random.normal(ks[1], (B, H, K)),
        jax.random.normal(ks[2], (B, H, V)),
    )


def _segments(B: int, T: int) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    rows[0, 20:70] = 1
    rows[0, 70:128] = 2  # a reset exactly at a chunk boundary
    rows[0, 128:] = 3
    if B > 1:
        rows[1, 63:] = 5  # a reset on a chunk's last token
        rows[1, 100:130] = 0  # an id reused after a different one starts a new document
    return jnp.asarray(rows)


def _args(x):
    return (x["q"], x["k"], x["v"], x["g"], x["beta"], x["gamma"])


def _f32(x):
    return {n: a.astype(F32) for n, a in x.items()}


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _core_ref(q, k, v, gates, h0, *, scale=1.0, segments=False):
    """The kernels' chunked algebra in float32 (``(o, final, states)``; see ``m3.forward``)."""
    B, T, H, K = q.shape
    V, NC = v.shape[-1], T // m3.CHUNK

    def chunks(x):  # [B, T, H, D] -> [NC, B, H, C, D]
        return x.astype(F32).reshape(B, NC, m3.CHUNK, H, -1).transpose(1, 0, 3, 2, 4)

    rows_i = jnp.arange(m3.CHUNK)[:, None]
    cols_j = jnp.arange(m3.CHUNK)[None, :]

    def step(state, xs):
        qn, kn, vn, gn = xs
        rows = (m3.ROW_E, m3.ROW_A, m3.ROW_C, m3.ROW_GAMMA, m3.ROW_QS, m3.ROW_W, m3.ROW_RUN, m3.ROW_RUN_COL, m3.ROW_DEC)
        e, a, c, gam, qs, w, run, run_col, dec = (gn[..., r, :] for r in rows)
        mask = cols_j < rows_i
        if segments:
            mask = mask & (run[..., :, None] == run_col[..., None, :])
        decay = jnp.exp(jnp.where(mask, e[..., :, None] - a[..., None, :], -jnp.inf))
        weight = jnp.where(rows_i == cols_j, gam[..., :, None], decay * c[..., None, :])
        scores = jnp.einsum("bhik,bhjk->bhij", qn, kn, precision=HIGHEST)
        o = qs[..., None] * jnp.einsum("bhik,bhkv->bhiv", qn, state, precision=HIGHEST)
        o = o + jnp.einsum("bhij,bhjv->bhiv", scores * weight, vn, precision=HIGHEST)
        new = dec[..., :1, None] * state + jnp.einsum("bhjk,bhjv->bhkv", kn * w[..., None], vn, precision=HIGHEST)
        return new, (o, state)

    final, (o, states) = lax.scan(step, h0.astype(F32), (chunks(q), chunks(k), chunks(v), jnp.moveaxis(gates, 2, 0)))
    o = o.transpose(1, 0, 3, 2, 4).reshape(B, T, H, V) * scale
    return o, final, jnp.moveaxis(states, 0, 2)


def _kernel_inputs(x, initial=None, segment_ids=None, dtype=BF16):
    B, T, H, K = x["q"].shape
    NC = -(-T // m3.CHUNK)
    gates = m3.prepare_gates(x["g"], x["beta"], x["gamma"], num_chunks=NC, segment_ids=segment_ids)
    if initial is None:
        h0 = jnp.zeros((B, H, K, x["v"].shape[-1]), F32)
    else:
        h0 = m3._initial_mid_state(initial, x["g"], x["beta"])
    pad = [(0, 0), (0, NC * m3.CHUNK - T), (0, 0), (0, 0)]
    q, k, v = (jnp.pad(x[n].astype(dtype), pad) for n in ("q", "k", "v"))
    return q, k, v, gates, h0


def _compare_backward(got, want, tol):
    for name, a, b in zip(("dq", "dk", "dv"), got[:3], want[:3]):
        _assert_close(a, b, tol, name)
    for r in (m3.ROW_E, m3.ROW_A, m3.ROW_C, m3.ROW_GAMMA, m3.ROW_QS, m3.ROW_W, m3.ROW_DEC):
        _assert_close(got[3][:, :, :, r], want[3][:, :, :, r], tol, f"dgates row {r}")
    _assert_close(got[4], want[4], tol, "dh0")


def _config(x, *, segments=False, interpret=False, stages=2, scale=1.0):
    B, T, H, K = x["q"].shape
    NC = -(-T // m3.CHUNK)
    return m3.Config(
        batch=B, heads=H, num_chunks=NC, key_dim=K, value_dim=x["v"].shape[-1], scale=scale,
        segments=segments, stages=min(stages, NC), interpret=interpret,
    )


# ------------------------------------------------------------------ algebra (CPU, no kernels)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_initial", [False, True])
def test_gate_rows_and_chunk_algebra_reproduce_the_reference(packed, with_initial):
    """``prepare_gates`` + the kernels' chunk algebra (f32) equal the token-by-token recurrence."""
    x = _f32(_inputs(0, B=2, T=150, H=2, K=16, V=8, dt_max=1.0, spikes=True))
    initial = _initial(1, 2, 2, 16, 8) if with_initial else None
    seg = _segments(2, 150) if packed else None
    o_ref, s_ref = ref.mamba3(*_args(x), initial_state=initial, output_final_state=True, segment_ids=seg, scale=0.7)
    q, k, v, gates, h0 = _kernel_inputs(x, initial, seg, dtype=F32)
    o, final, _ = _core_ref(q, k, v, gates, h0, scale=0.7, segments=packed)
    _assert_close(o[:, :150], o_ref, 1e-4, "o")
    _assert_close(final, s_ref.ssm, 1e-4, "final")


def test_gate_rows_have_no_overflow_for_huge_decays():
    x = _f32(_inputs(2, B=1, T=128, H=1, K=16, V=8, dt_max=1.0, spikes=True))
    x["g"] = x["g"].at[0, 5:70, 0].set(-300.0)
    gates = m3.prepare_gates(x["g"], x["beta"], x["gamma"], num_chunks=2)
    assert np.isfinite(np.asarray(gates)).all()
    assert float(jnp.max(gates[:, :, :, m3.ROW_QS])) <= 1.0 and float(jnp.max(gates[:, :, :, m3.ROW_DEC])) <= 1.0


# ------------------------------------------------------------- kernels in the GPU interpreter


@cpu
@pytest.mark.parametrize("packed", [False, True])
def test_forward_kernel_interpret_matches_chunk_algebra(packed):
    x = _inputs(3, B=1, T=128, H=2, K=64, V=64)
    initial = _initial(4, 1, 2, 64, 64)
    seg = None
    if packed:
        rows = np.zeros((1, 128), np.int32)
        rows[0, 30:], rows[0, 64:], rows[0, 100:] = 1, 2, 3  # resets inside and at the chunk boundary
        seg = jnp.asarray(rows)
    q, k, v, gates, h0 = _kernel_inputs(x, initial, seg)
    cfg = _config(x, segments=packed, interpret=True)
    o, final, states = m3.forward(cfg, q, k, v, gates, h0, save_states=True)
    o_want, final_want, states_want = _core_ref(q, k, v, gates, h0, segments=packed)
    _assert_close(o, o_want, 1e-2, "o")
    _assert_close(final, final_want, 1e-2, "final")
    _assert_close(states, states_want, 1e-2, "states")


@cpu
def test_backward_kernel_interpret_matches_autodiff_of_chunk_algebra():
    x = _inputs(5, B=1, T=128, H=1, K=64, V=64)
    initial = _initial(6, 1, 1, 64, 64)
    q, k, v, gates, h0 = _kernel_inputs(x, initial)
    cfg = _config(x, interpret=True)
    _, _, states = m3.forward(cfg, q, k, v, gates, h0, save_states=True)
    ks = jax.random.split(jax.random.key(7), 2)
    do = jax.random.normal(ks[0], (1, 128, 1, 64)).astype(BF16)
    dfinal = jax.random.normal(ks[1], (1, 1, 64, 64))
    got = m3.backward(cfg, q, k, v, do, gates, states, dfinal)

    def f(q, k, v, gates, h0):
        o, final, _ = _core_ref(q, k, v, gates, h0)
        return o, final

    _, vjp = jax.vjp(f, q.astype(F32), k.astype(F32), v.astype(F32), gates, h0)
    _compare_backward(got, vjp((do.astype(F32), dfinal)), 2e-2)


# ------------------------------------------------------------------------------ GPU: stages

DIMS = list(m3.SUPPORTED_DIMS)


@gpu
@pytest.mark.parametrize("dims", DIMS)
def test_forward_states_match_the_chunked_state_pass(dims):
    """The states the forward kernel saves are ``chunked.chunk_state_pass``'s states, shifted to ``S~``."""
    K, V = dims
    x = _inputs(10, K=K, V=V, T=256)
    initial = _initial(11, 2, 3, K, V)
    q, k, v, gates, h0 = _kernel_inputs(x, initial)
    _, final, states = m3.forward(_config(x), q, k, v, gates, h0, save_states=True)

    # chunked.mamba3's stage calls: two write streams (k, gamma v) and (k_prev, beta v_prev).
    xf = _f32(x)
    k_prev = jnp.concatenate([initial.key[:, None], xf["k"][:, :-1]], axis=1)
    v_prev = jnp.concatenate([initial.value[:, None], xf["v"][:, :-1]], axis=1)

    def prep(a):
        B, T, H = a.shape[:3]
        return jnp.moveaxis(a.reshape(B, T // 64, 64, H, *a.shape[3:]), 2, 3)

    g_cum = chunked.chunk_local_cumsum(prep(xf["g"]))
    both = lambda a, b: jnp.concatenate([a, b], axis=3)  # noqa: E731
    h, _, s_final = chunked.chunk_state_pass(
        both(prep(xf["k"]), prep(k_prev)), None,
        both(prep(xf["v"] * xf["gamma"][..., None]), prep(v_prev * xf["beta"][..., None])),
        both(g_cum, g_cum), initial.ssm, precision=HIGHEST,
    )
    # S~ entering chunk n = exp(g_{64n}) S_{64n-1} + beta_{64n} k_{64n-1}^T v_{64n-1}.
    first = slice(0, None, 64)
    g0 = jnp.moveaxis(xf["g"][:, first], 1, 2)[..., None, None]
    b0 = jnp.moveaxis(xf["beta"][:, first], 1, 2)[..., None, None]
    kp, vp = jnp.moveaxis(k_prev[:, first], 1, 2), jnp.moveaxis(v_prev[:, first], 1, 2)
    want = jnp.exp(g0) * jnp.moveaxis(h, 1, 2) + b0 * kp[..., :, None] * vp[..., None, :]
    _assert_close(states, want, 2e-2, "states")
    _assert_close(final, s_final, 2e-2, "final")


@gpu
@pytest.mark.parametrize("dims,packed", [(d, False) for d in DIMS] + [((128, 64), True)])
def test_backward_kernel_matches_autodiff_of_chunk_algebra(dims, packed):
    K, V = dims
    T = 192
    x = _inputs(12, K=K, V=V, T=T)
    initial = _initial(13, 2, 3, K, V)
    seg = _segments(2, T) if packed else None
    q, k, v, gates, h0 = _kernel_inputs(x, initial, seg)
    cfg = _config(x, segments=packed)
    _, _, states = m3.forward(cfg, q, k, v, gates, h0, save_states=True)
    ks = jax.random.split(jax.random.key(14), 2)
    do = jax.random.normal(ks[0], (2, T, 3, V)).astype(BF16)
    dfinal = jax.random.normal(ks[1], (2, 3, K, V))
    got = m3.backward(cfg, q, k, v, do, gates, states, dfinal)

    def f(q, k, v, gates, h0):
        o, final, _ = _core_ref(q, k, v, gates, h0, segments=packed)
        return o, final

    _, vjp = jax.vjp(f, q.astype(F32), k.astype(F32), v.astype(F32), gates, h0)
    _compare_backward(got, vjp((do.astype(F32), dfinal)), 2e-2)


# ------------------------------------------------------------------------- GPU: end to end

E2E_CASES = [
    # (K, V), T, initial state, segments: T % 64 != 0 with the previous key/value, every head-dim pair
    ((128, 64), 200, True, False),
    ((128, 64), 200, True, True),
    ((64, 128), 200, True, True),
    ((64, 64), 200, True, True),
    ((128, 64), 256, False, False),
    ((64, 128), 256, False, False),
]


@gpu
@pytest.mark.parametrize("dims,T,with_initial,packed", E2E_CASES)
def test_end_to_end_matches_reference_and_chunked(dims, T, with_initial, packed):
    K, V = dims
    x = _inputs(20, K=K, V=V, T=T)
    initial = _initial(21, 2, 3, K, V) if with_initial else None
    kw = dict(initial_state=initial, output_final_state=True, segment_ids=_segments(2, T) if packed else None)
    o, s = m3.mamba3(*_args(x), **kw)
    xf = _f32(x)
    o_ref, s_ref = ref.mamba3(*_args(xf), **kw)
    o_ch, s_ch = chunked.mamba3(*_args(xf), **kw)
    assert o.dtype == BF16 and s.ssm.dtype == F32
    for want, label in ((o_ref, "reference"), (o_ch, "chunked")):
        _assert_close(o, want, 2e-2, f"o vs {label}")
    for got_leaf, want_leaf, name in zip(s, s_ref, ("ssm", "key", "value")):
        _assert_close(got_leaf, want_leaf, 2e-2, f"state.{name}")
    _assert_close(s.ssm, s_ch.ssm, 2e-2, "state.ssm vs chunked")


@gpu
@pytest.mark.parametrize("T", [1, 37, 65])
def test_short_sequences(T):
    """One partial chunk (the second output warpgroup idles) and a nearly empty last chunk."""
    x = _inputs(26, B=1, T=T, H=2, K=128, V=64)
    initial = _initial(27, 1, 2, 128, 64)
    kw = dict(initial_state=initial, output_final_state=True)
    o, s = m3.mamba3(*_args(x), **kw)
    o_ref, s_ref = ref.mamba3(*_args(_f32(x)), **kw)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s.ssm, s_ref.ssm, 2e-2, "ssm")
    names = ("q", "k", "v", "g", "beta", "gamma")

    def loss(fn, inputs):
        out, state = fn(*(inputs[n] for n in names), **kw)
        return jnp.sum(out.astype(F32) ** 2) + jnp.sum(state.ssm ** 2)

    got = jax.grad(lambda inp: loss(m3.mamba3, inp))(x)
    want = jax.grad(lambda inp: loss(chunked.mamba3, inp))(_f32(x))
    for n in names:
        _assert_close(got[n], want[n], 3e-2, f"d{n}")


@gpu
def test_strong_decays_stay_finite_and_accurate():
    x = _inputs(22, T=256, dt_max=1.0, spikes=True)
    o, s = m3.mamba3(*_args(x), output_final_state=True)
    o_ref, s_ref = ref.mamba3(*_args(_f32(x)), output_final_state=True)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s.ssm, s_ref.ssm, 2e-2, "ssm")


@gpu
def test_state_chains_across_calls():
    x = _inputs(23, T=320)
    whole, s_whole = m3.mamba3(*_args(x), output_final_state=True)
    first = {n: a[:, :130] for n, a in x.items()}
    second = {n: a[:, 130:] for n, a in x.items()}
    o1, s_mid = m3.mamba3(*_args(first), output_final_state=True)
    o2, s_end = m3.mamba3(*_args(second), initial_state=s_mid, output_final_state=True)
    _assert_close(jnp.concatenate([o1, o2], axis=1), whole, 2e-2, "o")
    _assert_close(s_end.ssm, s_whole.ssm, 2e-2, "ssm")


@gpu
@pytest.mark.parametrize("stages", [1, 3, 4])
def test_pipeline_depth_does_not_change_results(stages):
    x = _inputs(24, T=320)
    o2, s2 = m3.mamba3(*_args(x), output_final_state=True)
    o, s = m3.mamba3(*_args(x), output_final_state=True, stages=stages)
    np.testing.assert_array_equal(np.asarray(o, np.float32), np.asarray(o2, np.float32))
    np.testing.assert_array_equal(np.asarray(s.ssm), np.asarray(s2.ssm))


@gpu
def test_scale_option():
    x = _inputs(25, T=128)
    o, _ = m3.mamba3(*_args(x), scale=0.125)
    o_ref, _ = ref.mamba3(*_args(_f32(x)), scale=0.125)
    _assert_close(o, o_ref, 2e-2, "o")


# --------------------------------------------------------------------------- GPU: gradients

GRAD_CASES = [((128, 64), False), ((128, 64), True), ((64, 128), False), ((64, 64), True)]


@gpu
@pytest.mark.parametrize("dims,packed", GRAD_CASES)
def test_gradients_match_chunked_autodiff(dims, packed):
    K, V = dims
    T = 200
    x = _inputs(30, K=K, V=V, T=T)
    initial = _initial(31, 2, 3, K, V)
    seg = _segments(2, T) if packed else None
    ks = jax.random.split(jax.random.key(32), 4)
    cot = jax.random.normal(ks[0], (2, T, 3, V))
    c_ssm, c_key, c_val = (jax.random.normal(kk, a.shape) for kk, a in zip(ks[1:], initial))
    names = ("q", "k", "v", "g", "beta", "gamma")

    def loss(fn, cast, inputs, init):
        args = [cast(inputs[n]) for n in names]
        o, s = fn(*args, initial_state=init, output_final_state=True, segment_ids=seg)
        return (jnp.sum(o.astype(F32) * cot) + jnp.sum(s.ssm * c_ssm) + jnp.sum(s.key * c_key)
                + jnp.sum(s.value * c_val))

    grad = jax.grad(lambda inp, init: loss(m3.mamba3, lambda a: a, inp, init), argnums=(0, 1))
    got_x, got_s = grad(x, initial)
    want_fn = jax.grad(lambda inp, init: loss(chunked.mamba3, lambda a: a.astype(F32), inp, init), argnums=(0, 1))
    want_x, want_s = want_fn(_f32(x), initial)
    for n in names:
        _assert_close(got_x[n], want_x[n], 3e-2, f"d{n}")
    for name, a, b in zip(("ssm", "key", "value"), got_s, want_s):
        _assert_close(a, b, 3e-2, f"d initial.{name}")


@gpu
def test_gradients_with_strong_decays_are_finite():
    x = _inputs(33, T=192, dt_max=1.0, spikes=True)

    def loss(inp):
        o, _ = m3.mamba3(*_args(inp))
        return jnp.sum(o.astype(F32) ** 2)

    grads = jax.grad(loss)(x)
    want = jax.grad(lambda inp: jnp.sum(chunked.mamba3(*_args(inp))[0] ** 2))(_f32(x))
    for n in ("q", "k", "v", "g", "beta", "gamma"):
        _assert_close(grads[n], want[n], 3e-2, f"d{n}")


@gpu
def test_training_shape_compiles_under_remat_and_value_and_grad():
    """The ``mamba3-130m`` layer shape (B=8, T=2048, H=24, K=128, V=64) fits shared memory."""
    x = _inputs(34, B=8, T=2048, H=24, K=128, V=64)

    @jax.checkpoint
    def loss(inp):
        o, _ = m3.mamba3(*_args(inp))
        return jnp.mean(o.astype(F32) ** 2)

    value, grads = jax.jit(jax.value_and_grad(loss))(x)
    assert np.isfinite(float(value))
    for n in ("q", "k", "v", "g", "beta", "gamma"):
        assert np.isfinite(np.asarray(grads[n], np.float32)).all(), n


# ----------------------------------------------------------------------------- dispatch


def test_supports_is_strict():
    x = _inputs(40, B=1, T=64, H=1, K=64, V=64)
    args = _args(x)
    assert m3.supports(*args, interpret=True)
    assert m3.supports(*args, initial_state=_initial(0, 1, 1, 64, 64), segment_ids=jnp.zeros((1, 64), jnp.int32),
                       interpret=True)
    assert not m3.supports(*_args(_f32(x)), interpret=True)  # float32 operands stay exact in chunked
    assert not m3.supports(*args, cu_seqlens=jnp.array([0, 64]), interpret=True)
    assert not m3.supports(*args, chunk_size=32, interpret=True)
    assert not m3.supports(*args, precision=HIGHEST, interpret=True)
    assert not m3.supports(*args, stages=5, interpret=True)
    assert not m3.supports(*args, unknown_option=1, interpret=True)
    assert not m3.supports(*args, segment_ids=jnp.zeros((1, 64), F32), interpret=True)
    assert not m3.supports(*args, initial_state=_initial(0, 1, 1, 64, 32), interpret=True)
    for K, V in ((32, 64), (128, 128)):
        y = _inputs(40, B=1, T=64, H=1, K=K, V=V)
        assert not m3.supports(*_args(y), interpret=True)
    assert m3.supports(*args) == ON_HOPPER
    with pytest.raises(ValueError, match="chunk_size"):
        m3.mamba3(*args, chunk_size=32)
    with pytest.raises(ValueError, match=r"\(K, V\)"):
        m3.mamba3(*_args(_inputs(40, B=1, T=64, H=1, K=128, V=128)))


def test_auto_dispatch_falls_back_where_the_kernel_does_not_apply():
    x = _f32(_inputs(41, B=1, T=96, H=2, K=64, V=64))
    gates = dict(g=x["g"], beta=x["beta"], gamma=x["gamma"])
    o_auto, _ = ops.linear_attention("mamba3", x["q"], x["k"], x["v"], **gates)
    o_ref, _ = ops.linear_attention("mamba3", x["q"], x["k"], x["v"], impl="reference", **gates)
    _assert_close(o_auto, o_ref, 1e-4, "f32 auto")  # f32 runs chunked (exact)
    assert "mosaic_gpu" in ops.available("mamba3")


@gpu
@pytest.mark.parametrize("impl", ["mosaic_gpu", "auto"])
def test_dispatch_runs_the_kernel(impl):
    x = _inputs(42, T=128)
    gates = dict(g=x["g"], beta=x["beta"], gamma=x["gamma"])
    o_dispatched, _ = ops.linear_attention("mamba3", x["q"], x["k"], x["v"], impl=impl, **gates)
    o_kernel, _ = m3.mamba3(*_args(x))
    np.testing.assert_array_equal(np.asarray(o_dispatched, np.float32), np.asarray(o_kernel, np.float32))
