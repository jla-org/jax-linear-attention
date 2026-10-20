# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU mLSTM (the simple-GLA mapping of :mod:`jla.ops.mosaic_gpu.mlstm`, opt-in).

Values, final states and gradients against the stabilised scan reference
(float32 on the same bf16 or fp16 inputs), with initial states (also as a plain
tuple), packed documents, odd and short lengths, large gate magnitudes, the
xLSTM-7B head dims (K = 256 as two key blocks, V = 512) and both value-block
widths; a bf16 mLSTM layer with ``implementation="mosaic_gpu"`` against the
chunked layer (packed documents, parameter gradients, prefill then decode);
dispatch (``auto`` keeps the chunked op); the stabiliser scan against the
reference recurrence; the ``supports`` predicate (also on CPU).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import mlstm as mm

ON_HOPPER = cm.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
BF16, F16, F32 = jnp.bfloat16, jnp.float16, jnp.float32
TOL = 2e-2  # max error / max |reference| (bf16 operands, float32 accumulation)
# Gradient bounds per key dim: K = 256 sums two bf16-rounded 128-wide key halves (measured up to 6.1e-2).
GRAD_TOL = {64: TOL, 128: TOL, 256: 4 * TOL}


def _inputs(seed, B, T, H, K, V, *, gate_scale=3.0, state=False, segments=False, dtype=BF16):
    ks = jax.random.split(jax.random.key(seed), 10)
    x = dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(dtype),
        k=jax.random.normal(ks[1], (B, T, H, K)).astype(dtype),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(dtype),
        g=jax.nn.log_sigmoid(gate_scale * jax.random.normal(ks[3], (B, T, H)) + 2.0),
        i=gate_scale * jax.random.normal(ks[4], (B, T, H)),
        cot=jax.random.normal(ks[5], (B, T, H, V)),
    )
    if state:
        x["initial_state"] = ref.MLSTMState(
            0.3 * jax.random.normal(ks[6], (B, H, K, V)),
            0.3 * jax.random.normal(ks[7], (B, H, K)),
            jax.random.normal(ks[8], (B, H)),
        )
    if segments:
        x["segment_ids"] = jnp.cumsum(jax.random.bernoulli(ks[9], 0.02, (B, T)).astype(jnp.int32), axis=1)
    return x


def _err(got, want) -> float:
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.isfinite(got).all(), "non-finite values"
    return float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-6)


def _check(got, want, tol, name):
    err = _err(got, want)
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| exceeds {tol}"


def _options(x):
    return {key: x[key] for key in ("initial_state", "segment_ids") if key in x}


def _reference(x, **kw):
    q, k, v = (x[n].astype(F32) for n in ("q", "k", "v"))
    return ref.mlstm(q, k, v, x["g"], x["i"], output_final_state=True, **_options(x), **kw)


def _kernel(x, **kw):
    return mm.mlstm(x["q"], x["k"], x["v"], x["g"], x["i"], output_final_state=True, **_options(x), **kw)


def _numerator_denominator(x, eps=1e-6):
    """Float32 scan oracle of ``q c_t`` and ``D_t = max(|q n_t|, exp(-m_t)) + eps`` (so ``o = num / D``)."""
    q, k, v = (x[n].astype(F32) for n in ("q", "k", "v"))
    batch, seq_len, heads, key_dim = q.shape
    state = x.get("initial_state") or ref.mlstm_empty_state(batch, heads, key_dim, v.shape[-1])
    seg = x.get("segment_ids")
    resets = jnp.zeros((batch, seq_len), bool) if seg is None else ref.segment_resets(seg)

    def step(carry, xs):
        c, n, m = carry
        q_t, k_t, v_t, g_t, i_t, r = xs
        c, n = jnp.where(r[:, None, None, None], 0.0, c), jnp.where(r[:, None, None], 0.0, n)
        m = jnp.where(r[:, None], ref.MLSTM_EMPTY_M, m)
        m_new = jnp.maximum(g_t + m, i_t)
        f, w = jnp.exp(g_t + m - m_new), jnp.exp(i_t - m_new)
        c = f[..., None, None] * c + w[..., None, None] * k_t[..., :, None] * v_t[..., None, :]
        n = f[..., None] * n + w[..., None] * k_t
        q_t = q_t * key_dim**-0.5
        num = jnp.einsum("bhk,bhkv->bhv", q_t, c, precision=jax.lax.Precision.HIGHEST)
        den = jnp.maximum(jnp.abs(jnp.sum(q_t * n, -1)), jnp.exp(jnp.minimum(-m_new, 80.0))) + eps
        return (c, n, m_new), (num, den)

    xs = tuple(jnp.moveaxis(a.astype(F32), 1, 0) for a in (q, k, v, x["g"], x["i"])) + (resets.T,)
    _, (num, den) = jax.lax.scan(step, tuple(a.astype(F32) for a in state), xs)
    return jnp.moveaxis(num, 0, 1), jnp.moveaxis(den, 0, 1)


def _check_forward(x, tol=TOL, **kw):
    o, state = _kernel(x, **kw)
    o_ref, state_ref = _reference(x)
    num_ref, den_ref = _numerator_denominator(x)
    assert o.dtype == x["q"].dtype
    # The numerator q c carries the kernels' bf16 operand rounding; dividing by a small |q n| amplifies it
    # (rows where q is nearly orthogonal to n), so o is checked through o * D against q c, and directly
    # against the reference with a bound that includes those rows (benchmarks/mlstm.md).
    _check(o.astype(F32) * den_ref[..., None], num_ref, tol, "o * denominator")
    _check(o, o_ref, 5 * tol, "o")
    _check(state.c, state_ref.c, tol, "final c")
    _check(state.n, state_ref.n, tol, "final n")
    np.testing.assert_allclose(np.asarray(state.m), np.asarray(state_ref.m), rtol=0, atol=1e-4)


@gpu
@pytest.mark.parametrize("shape", [(2, 200, 2, 64, 64), (1, 256, 2, 128, 128), (1, 192, 2, 256, 512),
                                   (2, 130, 1, 64, 256)])
def test_forward_matches_reference(shape):
    _check_forward(_inputs(0, *shape))


@gpu
@pytest.mark.parametrize("shape", [(2, 257, 2, 64, 128), (1, 200, 2, 256, 512)])
def test_initial_state_and_segments(shape):
    _check_forward(_inputs(1, *shape, state=True, segments=True))


@gpu
def test_value_block_64():
    _check_forward(_inputs(2, 1, 192, 2, 256, 512, state=True), value_block=64)


@gpu
def test_large_gates_and_long_sequence():
    # Input gates up to ~|40| (exp(i) far outside bf16-safe scaling without the stabiliser), strong forgetting.
    _check_forward(_inputs(3, 1, 1024, 2, 128, 128, gate_scale=12.0, state=True, segments=True))


@gpu
@pytest.mark.parametrize("shape", [(1, 192, 2, 256, 512), (2, 130, 2, 128, 128)])
def test_fp16(shape):
    _check_forward(_inputs(10, *shape, state=True, segments=True, dtype=F16))


@gpu
@pytest.mark.parametrize("seq_len", [1, 17, 63, 64, 65])
def test_short_sequences(seq_len):
    _check_forward(_inputs(11, 2, seq_len, 2, 128, 128, state=True, segments=True))


@gpu
def test_plain_tuple_initial_state():
    x = _inputs(12, 1, 136, 2, 64, 128, state=True)
    o, s = _kernel(x)
    o2, s2 = _kernel(dict(x, initial_state=tuple(x["initial_state"])))
    np.testing.assert_array_equal(np.asarray(o, np.float32), np.asarray(o2, np.float32))
    for a, b in zip(s, s2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@gpu
def test_empty_initial_state_equals_none():
    x = _inputs(4, 1, 160, 2, 64, 64)
    o, s = _kernel(x)
    o2, s2 = _kernel(dict(x, initial_state=ref.mlstm_empty_state(1, 2, 64, 64)))
    np.testing.assert_array_equal(np.asarray(o, np.float32), np.asarray(o2, np.float32))
    np.testing.assert_array_equal(np.asarray(s.c), np.asarray(s2.c))


def _grads(fn, x, with_state):
    names = ("q", "k", "v", "g", "i") + (("initial_state",) if with_state else ())

    def loss(*diff):
        y = dict(x, **dict(zip(names, diff)))
        o, state = fn(y)
        total = jnp.sum(o.astype(F32) * y["cot"])
        if with_state:  # the final state is an output too
            total = total + jnp.sum(jnp.sin(state.c)) + jnp.sum(jnp.cos(state.n))
        return total

    return names, jax.grad(loss, argnums=tuple(range(len(names))))(*(x[n] for n in names))


@gpu
@pytest.mark.parametrize("shape,state,segments", [((1, 160, 2, 64, 64), False, False),
                                                  ((2, 200, 1, 128, 128), True, True),
                                                  ((1, 128, 1, 256, 512), True, False)])
def test_gradients_match_reference(shape, state, segments):
    x = _inputs(5, *shape, state=state, segments=segments)
    names, got = _grads(_kernel, x, state)
    _, want = _grads(_reference, x, state)
    # Gradients pass through 1 / D like the outputs (see _check_forward); K = 256 also carries the bf16
    # rounding of its two key halves, hence the looser bound there (GRAD_TOL).
    tol = GRAD_TOL[shape[3]]
    for name, a, b in zip(names, got, want):
        if name == "initial_state":
            for part, aa, bb in zip(("c0", "n0", "m0"), a, b):
                _check(aa, bb, tol, f"d{part}")
        else:
            _check(a, b, tol, f"d{name}")


@gpu
@pytest.mark.parametrize("key_dim", mm.KEY_DIMS)
@pytest.mark.parametrize("value_dim", mm.VALUE_DIMS)
def test_every_supported_head_dim(key_dim, value_dim):
    """Every (K, V) that ``supports`` accepts, with an initial state and packed documents."""
    _check_forward(_inputs(9, 1, 136, 1, key_dim, value_dim, state=True, segments=True))


@gpu
def test_dispatch():
    x = _inputs(6, 1, 128, 2, 64, 64)
    o, _ = ops.linear_attention("mlstm", x["q"], x["k"], x["v"], g=x["g"], i=x["i"], impl="mosaic_gpu")
    o_ref, _ = _reference(x)
    _check(o, o_ref, TOL, "o")
    # Opt-in only (registered with auto=False): auto keeps the chunked op.
    o_auto, _ = ops.linear_attention("mlstm", x["q"], x["k"], x["v"], g=x["g"], i=x["i"])
    o_chunked, _ = ops.linear_attention("mlstm", x["q"], x["k"], x["v"], g=x["g"], i=x["i"], impl="chunked")
    np.testing.assert_array_equal(np.asarray(o_auto, np.float32), np.asarray(o_chunked, np.float32))


def _layer(impl, d_model=512, heads=2, key_dim=128, value_dim=256):
    from jla.layers.linear_attention import make_mixer
    from jla.models import get_preset

    cfg = get_preset("tiny-mlstm")
    la = dataclasses.replace(cfg.linear_attention, implementation=impl, num_k_heads=heads, num_v_heads=heads,
                             head_k_dim=key_dim, head_v_dim=value_dim, chunk_size=64)
    return make_mixer("mlstm", d_model, la, key=jax.random.key(0), std=0.05, norm_eps=1e-6,
                      param_dtype=F32, compute_dtype="bfloat16")


@gpu
def test_layer_with_mosaic_gpu_implementation():
    """A bf16 mLSTM layer (xLSTM-7B gate init) on this path against the same layer on the chunked op."""
    import equinox as eqx

    ours, chunked = _layer("mosaic_gpu"), _layer("chunked")
    x = jax.random.normal(jax.random.key(1), (2, 300, 512))
    seg = jnp.concatenate([jnp.zeros((2, 130), jnp.int32), jnp.ones((2, 170), jnp.int32)], axis=1)
    _check(ours(x, segment_ids=seg), chunked(x, segment_ids=seg), TOL, "layer output")

    # Prefill from an empty cache (T > 1: this path, initial and final state), then decode steps.
    cache = ours.init_cache(2, BF16)
    y, cache = ours.step(x[:, :70], cache)
    assert cache.state.c.dtype == F32 and cache.state.n.dtype == F32
    outs = [y]
    for t in range(70, 80):
        y, cache = ours.step(x[:, t : t + 1], cache)
        outs.append(y)
    _check(jnp.concatenate(outs, axis=1), chunked(x[:, :80]), TOL, "prefill + decode")

    cot = jax.random.normal(jax.random.key(3), (2, 300, 512))

    def loss(layer):
        return jnp.sum(layer(x, segment_ids=seg).astype(F32) * cot)

    got = eqx.filter_grad(loss)(ours)
    want = eqx.filter_grad(loss)(chunked)
    leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(got, eqx.is_inexact_array))
    wanted = dict(jax.tree_util.tree_leaves_with_path(eqx.filter(want, eqx.is_inexact_array)))
    assert leaves
    for path, a in leaves:
        _check(a, wanted[path], 2 * TOL, f"d{jax.tree_util.keystr(path)}")


def test_stabilizer_matches_recurrence():
    ks = jax.random.split(jax.random.key(7), 4)
    g = jax.nn.log_sigmoid(4.0 * jax.random.normal(ks[0], (2, 300, 3)) + 1.0)
    i = 8.0 * jax.random.normal(ks[1], (2, 300, 3))
    m0 = jax.random.normal(ks[2], (2, 3))
    resets = ref.segment_resets(jnp.cumsum(jax.random.bernoulli(ks[3], 0.05, (2, 300)).astype(jnp.int32), 1))

    def step(m, xs):
        g_t, i_t, r = xs
        m = jnp.where(r[:, None], ref.MLSTM_EMPTY_M, m)
        m = jnp.maximum(g_t + m, i_t)
        return m, m

    _, want = jax.lax.scan(step, m0, (jnp.moveaxis(g, 1, 0), jnp.moveaxis(i, 1, 0), resets.T))
    got = mm.stabilizer(g, i, m0, resets)
    np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.moveaxis(want, 0, 1)), rtol=0, atol=1e-4)


def test_supports():
    x = _inputs(8, 1, 64, 2, 64, 64)
    args = (x["q"], x["k"], x["v"], x["g"], x["i"])
    assert mm.supports_mlstm(*args) == ON_HOPPER
    assert not mm.supports_mlstm(*args, chunk_size=32)
    assert not mm.supports_mlstm(*args, cu_seqlens=jnp.array([0, 64]))
    assert not mm.supports_mlstm(*args, use_qk_l2norm=True)
    assert not mm.supports_mlstm(*(a.astype(F32) if a.ndim == 4 else a for a in args))
    y = _inputs(8, 1, 64, 2, 96, 64)
    assert not mm.supports_mlstm(y["q"], y["k"], y["v"], y["g"], y["i"])
    if ON_HOPPER:
        assert mm.supports_mlstm(*args, segment_ids=jnp.zeros((1, 64), jnp.int32), eps=1e-5,
                                 initial_state=ref.mlstm_empty_state(1, 2, 64, 64), output_final_state=True)
