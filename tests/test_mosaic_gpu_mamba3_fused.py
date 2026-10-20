# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused Mamba-3 SISO layer core on Mosaic GPU vs the layer's XLA core (``ssm.mamba3_siso``):
values, final states and gradients of every input, grouped ``B/C``, packed documents, initial
states with the angle state; the layer integration and the ``supports_fused`` predicate.

The interpreter test runs on CPU (``JAX_PLATFORMS=cpu``); the rest needs an H100.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import ssm
from jla.models.config import SSMConfig
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import mamba3 as m3
from jla.ops.mosaic_gpu import mamba3_fused as fused

ON_HOPPER = m3._on_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
cpu = pytest.mark.skipif(jax.default_backend() != "cpu", reason="GPU-interpreter test (run with JAX_PLATFORMS=cpu)")

F32, BF16 = jnp.float32, jnp.bfloat16
NAMES = ("q", "k", "v", "adt", "dt", "trap", "q_bias", "k_bias", "angles", "d", "z")


def _inputs(seed, B=2, T=200, H=4, G=1, K=128, V=64, S=None, dt_max=0.1, gated=True):
    """Layer-like inputs: unit-RMS ``C, B`` per group, ``ADT = A DT``, biases near one; ``C, B, x, z`` bf16."""
    S = S or K // 4
    ks = jax.random.split(jax.random.key(seed), 11)

    def rms(x):
        return x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)

    dt = jnp.exp(jax.random.uniform(ks[3], (B, T, H), minval=math.log(1e-3), maxval=math.log(dt_max)))
    a = -jax.random.uniform(ks[4], (B, T, H), minval=1.0, maxval=16.0)
    x = dict(
        q=rms(jax.random.normal(ks[0], (B, T, G, K))).astype(BF16),
        k=rms(jax.random.normal(ks[1], (B, T, G, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        adt=a * dt,
        dt=dt,
        trap=jax.random.normal(ks[5], (B, T, H)),
        q_bias=1.0 + 0.3 * jax.random.normal(ks[6], (H, K)),
        k_bias=1.0 + 0.3 * jax.random.normal(ks[7], (H, K)),
        angles=jax.random.normal(ks[8], (B, T, H, S)),
        d=1.0 + 0.1 * jax.random.normal(ks[9], (H,)),
        z=jax.random.normal(ks[10], (B, T, H, V)).astype(BF16),
    )
    if not gated:
        x["d"], x["z"] = None, None
    return x


def _initial(seed, B, H, K, V, S):
    ks = jax.random.split(jax.random.key(seed), 4)
    return (
        jax.random.uniform(ks[0], (B, H, S), minval=0.0, maxval=2 * math.pi),
        ref.Mamba3State(
            0.1 * jax.random.normal(ks[1], (B, H, K, V)),
            jax.random.normal(ks[2], (B, H, K)),
            jax.random.normal(ks[3], (B, H, V)),
        ),
    )


def _segments(B, T):
    rows = np.zeros((B, T), np.int32)
    rows[0, 20:70] = 1
    rows[0, 70:128] = 2  # a reset exactly at a chunk boundary
    rows[0, 128:] = 3
    if B > 1:
        rows[1, 63:] = 5  # a reset on a chunk's last token
        rows[1, 100:130] = 0  # an id reused after a different one starts a new document
    return jnp.asarray(rows)


def _f32(x):
    return {n: None if a is None else a.astype(F32) for n, a in x.items()}


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _assert_angles_close(got, want, tol, name):
    diff = np.asarray(got, np.float64) - np.asarray(want, np.float64)
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    assert np.abs(diff).max() <= tol, f"{name}: angle error {np.abs(diff).max():.3e}"
    assert (np.asarray(got) >= 0).all() and (np.asarray(got) < 2 * math.pi + 1e-6).all(), name


def _call(fn, x, **kw):
    return fn(*(x[n] for n in NAMES[:9]), d=x["d"], z=x["z"], **kw)


def _reference(x, **kw):
    """The layer's XLA core on float32 copies of the inputs (the oracle)."""
    return _call(ssm.mamba3_siso, _f32(x), impl="chunked", **kw)


# ------------------------------------------------------------------------------- values

VALUE_CASES = [  # (K, V), S, H, G, T, packed, with_initial, gated
    ((128, 64), 32, 4, 1, 200, True, True, True),
    ((128, 64), 32, 4, 2, 256, False, False, True),
    ((128, 64), 64, 3, 1, 192, True, True, False),
    ((64, 128), 16, 4, 4, 130, True, True, True),
    ((64, 64), 32, 2, 1, 100, False, True, True),
]


@gpu
@pytest.mark.parametrize("dims,S,H,G,T,packed,with_initial,gated", VALUE_CASES)
def test_fused_matches_the_layer_core(dims, S, H, G, T, packed, with_initial, gated):
    K, V = dims
    x = _inputs(1, T=T, H=H, G=G, K=K, V=V, S=S, gated=gated)
    kw = dict(output_final_state=True, segment_ids=_segments(2, T) if packed else None,
              initial_state=_initial(2, 2, H, K, V, S) if with_initial else None)
    assert _call(fused.supports_fused, x, **kw)
    y, (angle, state) = _call(fused.mamba3_siso_fused, x, **kw)
    y_ref, (angle_ref, state_ref) = _reference(x, **kw)
    assert y.dtype == BF16 and state.ssm.dtype == F32 and angle.dtype == F32
    _assert_close(y, y_ref, 2e-2, "y")
    _assert_angles_close(angle, angle_ref, 1e-4, "final angle")
    for got, want, name in zip(state, state_ref, ("ssm", "key", "value")):
        _assert_close(got, want, 2e-2, f"state.{name}")


@gpu
def test_large_angles_and_strong_decays():
    x = _inputs(3, T=320, H=2, dt_max=1.0)
    kw = dict(output_final_state=True)
    y, (angle, state) = _call(fused.mamba3_siso_fused, x, **kw)
    y_ref, (angle_ref, state_ref) = _reference(x, **kw)
    _assert_close(y, y_ref, 2e-2, "y")
    _assert_angles_close(angle, angle_ref, 1e-3, "final angle")
    _assert_close(state.ssm, state_ref.ssm, 2e-2, "ssm")


@gpu
@pytest.mark.parametrize("T", [1, 37])
def test_short_sequences_and_bf16_angles(T):
    """One partial chunk; bf16 angles are scanned in float32 and get a bf16 gradient."""
    x = _inputs(14, T=T, H=2, G=1)
    x["angles"] = x["angles"].astype(BF16)
    kw = dict(initial_state=_initial(15, 2, 2, 128, 64, 32), output_final_state=True)
    y, (angle, state) = _call(fused.mamba3_siso_fused, x, **kw)
    y_ref, (angle_ref, state_ref) = _reference(x, **kw)
    _assert_close(y, y_ref, 2e-2, "y")
    _assert_angles_close(angle, angle_ref, 1e-4, "angle")
    _assert_close(state.ssm, state_ref.ssm, 2e-2, "ssm")

    def loss(fn, inputs, **options):
        y, (angle, state) = fn(*(inputs[n] for n in NAMES[:9]), d=inputs["d"], z=inputs["z"], **kw, **options)
        return jnp.sum(y.astype(F32) ** 2) + jnp.sum(jnp.sin(angle)) + jnp.sum(state.ssm ** 2)

    got = jax.grad(lambda inp: loss(fused.mamba3_siso_fused, inp))(x)
    want = jax.grad(lambda inp: loss(ssm.mamba3_siso, inp, impl="chunked"))(_f32(x))
    for n in NAMES:
        assert got[n].dtype == x[n].dtype, n
        _assert_close(got[n], want[n], 3e-2, f"d{n}")


@gpu
def test_state_chains_across_calls():
    x = _inputs(4, T=300, H=2)
    kw = dict(output_final_state=True)
    y, (angle, state) = _call(fused.mamba3_siso_fused, x, **kw)
    head = {n: None if a is None or a.ndim < 3 else a[:, :170] for n, a in x.items()}
    tail = {n: None if a is None or a.ndim < 3 else a[:, 170:] for n, a in x.items()}
    for part in (head, tail):
        part.update(q_bias=x["q_bias"], k_bias=x["k_bias"], d=x["d"])
    y1, mid = _call(fused.mamba3_siso_fused, head, **kw)
    y2, (angle2, state2) = _call(fused.mamba3_siso_fused, tail, initial_state=mid, **kw)
    _assert_close(jnp.concatenate([y1, y2], axis=1), y, 2e-2, "y")
    _assert_angles_close(angle2, angle, 1e-4, "angle")
    _assert_close(state2.ssm, state.ssm, 2e-2, "ssm")


# ---------------------------------------------------------------------------- gradients

GRAD_CASES = [  # (K, V), S, H, G, T, packed, gated
    ((128, 64), 32, 4, 1, 200, True, True),
    ((128, 64), 32, 4, 2, 128, False, True),
    ((128, 64), 64, 2, 2, 150, False, False),
    ((64, 128), 16, 4, 4, 130, True, True),
    ((64, 64), 16, 4, 2, 192, True, True),
    ((64, 64), 32, 2, 1, 128, True, True),
]


@gpu
@pytest.mark.parametrize("dims,S,H,G,T,packed,gated", GRAD_CASES)
def test_gradients_match_the_layer_core(dims, S, H, G, T, packed, gated):
    K, V = dims
    x = _inputs(5, T=T, H=H, G=G, K=K, V=V, S=S, gated=gated)
    initial = _initial(6, 2, H, K, V, S)
    seg = _segments(2, T) if packed else None
    ks = jax.random.split(jax.random.key(7), 6)
    cot = jax.random.normal(ks[0], (2, T, H, V))
    c_angle = jax.random.normal(ks[1], (2, H, S))
    c_state = [jax.random.normal(kk, a.shape) for kk, a in zip(ks[2:], initial[1])]

    def loss(fn, inputs, init, **kw):
        y, (angle, state) = fn(*(inputs[n] for n in NAMES[:9]), d=inputs["d"], z=inputs["z"], initial_state=init,
                               output_final_state=True, segment_ids=seg, **kw)
        total = jnp.sum(y.astype(F32) * cot) + jnp.sum(jnp.sin(angle) * c_angle)
        return total + sum(jnp.sum(a * c) for a, c in zip(state, c_state))

    diff = [n for n in NAMES if x[n] is not None]

    def grads(fn, inputs, **kw):
        def f(sub, init):
            return loss(fn, {**inputs, **sub}, init, **kw)

        return jax.grad(f, argnums=(0, 1))({n: inputs[n] for n in diff}, initial)

    got_x, got_s = grads(fused.mamba3_siso_fused, x)
    want_x, want_s = grads(ssm.mamba3_siso, _f32(x), impl="chunked")
    for n in diff:
        assert got_x[n].dtype == x[n].dtype, n
        _assert_close(got_x[n], want_x[n], 3e-2, f"d{n}")
    _assert_close(got_s[0], want_s[0], 3e-2, "d initial angle")
    for name, a, b in zip(("ssm", "key", "value"), got_s[1], want_s[1]):
        _assert_close(a, b, 3e-2, f"d initial.{name}")


@gpu
def test_gradients_with_large_angles_match_the_unfused_kernels(monkeypatch):
    """``dt`` up to 1. The angle gradient sums many rounded bf16 ``q . k`` terms, so against float32
    it is only bf16-accurate (as for the layer's XLA glue around the same recurrence kernels); against
    that unfused path the fused core agrees tightly."""
    x = _inputs(8, T=256, H=2, dt_max=1.0)

    def loss(fn, inputs, **kw):
        y, _ = fn(*(inputs[n] for n in NAMES[:9]), d=inputs["d"], z=inputs["z"], **kw)
        return jnp.sum(y.astype(F32) ** 2)

    got = jax.grad(lambda inp: loss(fused.mamba3_siso_fused, inp))(x)
    want = jax.grad(lambda inp: loss(ssm.mamba3_siso, inp, impl="chunked"))(_f32(x))
    monkeypatch.setattr(ssm, "_fused_core", lambda: None)  # the XLA glue around the recurrence kernels
    unfused = jax.grad(lambda inp: loss(ssm.mamba3_siso, inp, impl="mosaic_gpu"))(x)
    for n in NAMES:
        _assert_close(got[n], unfused[n], 5e-3, f"d{n} vs unfused kernels")
        _assert_close(got[n], want[n], 1e-1 if n == "angles" else 3e-2, f"d{n} vs float32")


@gpu
def test_training_shape_under_remat_and_value_and_grad():
    """The ``mamba3-130m`` layer core (B=8, T=2048, H=24, one group, K=128, V=64, S=32)."""
    x = _inputs(9, B=8, T=2048, H=24, G=1, K=128, V=64)

    @jax.checkpoint
    def loss(inp):
        y, _ = _call(fused.mamba3_siso_fused, inp)
        return jnp.mean(y.astype(F32) ** 2)

    value, grads = jax.jit(jax.value_and_grad(loss))(x)
    assert np.isfinite(float(value))
    for n in NAMES:
        assert np.isfinite(np.asarray(grads[n], np.float32)).all(), n


# --------------------------------------------------------------------- layer integration


def _layer(**kw):
    cfg = SSMConfig(**{**dict(d_state=128, expand=2, head_dim=64, n_groups=1, chunk_size=64), **kw})
    return ssm.Mamba3(256, cfg, key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="bfloat16")


@gpu
@pytest.mark.parametrize("outproj_norm", [True, False])
def test_layer_uses_the_fused_core_and_matches_chunked(monkeypatch, outproj_norm):
    calls = []
    original = fused.mamba3_siso_fused

    def spy(*args, **kw):
        calls.append(kw.get("segment_ids") is not None)
        return original(*args, **kw)

    monkeypatch.setattr(fused, "mamba3_siso_fused", spy)
    layer = _layer(implementation="mosaic_gpu", outproj_norm=outproj_norm)
    reference_layer = _layer(implementation="chunked", outproj_norm=outproj_norm)  # same key: same parameters
    u = jax.random.normal(jax.random.key(1), (2, 200, 256))
    seg = _segments(2, 200)

    def loss(model):
        return jnp.sum(model(u, segment_ids=seg).astype(F32) ** 2)

    value, grads = jax.value_and_grad(loss)(layer)
    want_value, want_grads = jax.value_and_grad(loss)(reference_layer)
    assert calls == [True]
    _assert_close(value, want_value, 2e-2, "loss")
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    want_leaves = jax.tree.leaves(eqx.filter(want_grads, eqx.is_inexact_array))
    for i, (a, b) in enumerate(zip(leaves, want_leaves)):
        _assert_close(a, b, 5e-2, f"grad leaf {i}")


@gpu
def test_layer_auto_and_fallback():
    x = _inputs(10, T=128, H=2)
    y_mosaic, _ = _call(ssm.mamba3_siso, x, impl="mosaic_gpu")
    y_fused, _ = _call(fused.mamba3_siso_fused, x)
    np.testing.assert_array_equal(np.asarray(y_mosaic, np.float32), np.asarray(y_fused, np.float32))
    y_auto, _ = _call(ssm.mamba3_siso, x, impl="auto")
    if fused.AUTO:
        np.testing.assert_array_equal(np.asarray(y_auto, np.float32), np.asarray(y_fused, np.float32))
    # float32 operands are not fused: the layer's XLA core runs, exactly as with impl="chunked"
    xf = _f32(x)
    y_f32, _ = _call(ssm.mamba3_siso, xf, impl="auto")
    y_chunked, _ = _call(ssm.mamba3_siso, xf, impl="chunked")
    _assert_close(y_f32, y_chunked, 1e-5, "f32 auto")


# ------------------------------------------------------------------------- supports, CPU


def test_supports_fused_is_strict():
    x = _inputs(11, B=1, T=64, H=2, G=1, K=64, V=64, S=16)

    def ok(inputs=x, **kw):
        return _call(fused.supports_fused, inputs, interpret=True, **kw)

    assert ok()
    assert ok(segment_ids=jnp.zeros((1, 64), jnp.int32), initial_state=_initial(0, 1, 2, 64, 64, 16),
              output_final_state=True)
    assert ok(inputs={**x, "d": None, "z": None})
    assert not ok(inputs=_f32(x))  # float32 operands stay exact in the XLA core
    assert not ok(chunk_size=32)
    assert not ok(stages=5)
    assert not ok(unknown_option=1)
    assert not ok(segment_ids=jnp.zeros((1, 64), F32))
    assert not ok(initial_state=_initial(0, 1, 2, 64, 32, 16))
    for kw in (dict(S=8), dict(K=64, S=64), dict(K=128, S=16), dict(K=128, V=128), dict(K=32, V=64, S=8)):
        assert not ok(inputs=_inputs(11, B=1, T=64, H=2, **{"K": 64, "V": 64, **kw}))
    assert not ok(inputs={**x, "q": x["q"][:, :, :1].repeat(3, axis=2), "k": x["k"][:, :, :1].repeat(3, axis=2)})
    assert _call(fused.supports_fused, x) == ON_HOPPER
    with pytest.raises(ValueError, match="chunk_size"):
        _call(fused.mamba3_siso_fused, x, chunk_size=32)


@cpu
def test_interpret_matches_the_layer_core():
    x = _inputs(12, B=1, T=96, H=2, G=1, K=64, V=64, S=16)
    kw = dict(segment_ids=_segments(1, 96), initial_state=_initial(13, 1, 2, 64, 64, 16), output_final_state=True)
    y, (angle, state) = _call(fused.mamba3_siso_fused, x, interpret=True, **kw)
    y_ref, (angle_ref, state_ref) = _reference(x, **kw)
    _assert_close(y, y_ref, 2e-2, "y")
    _assert_angles_close(angle, angle_ref, 1e-4, "angle")
    _assert_close(state.ssm, state_ref.ssm, 2e-2, "ssm")
