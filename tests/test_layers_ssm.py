# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), tests/ops/triton/test_mamba3_siso.py (mamba3_siso_step_ref)
"""Mamba-2 and Mamba-3 SISO: official reference semantics, causality, documents, decode parity."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.ssm import (
    Mamba2,
    Mamba3,
    cumulative_angle,
    heavy_tail,
    mamba2_ssd,
    mamba3_siso,
    rotate_pairs,
)
from jla.models.config import SSMConfig


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=2e-4, atol=2e-4)
D_MODEL = 32


def _np(x):
    return np.asarray(x, np.float64)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(x))


def _rms(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _cfg(**kw) -> SSMConfig:
    base = dict(d_state=8, expand=2, head_dim=8, n_groups=2, chunk_size=16)
    base.update(kw)
    return SSMConfig(**base)


def _randomize(module, seed, keep=()):
    """Random values for every float array except ``keep`` (so identity-like inits hide nothing)."""
    kept = {name: getattr(module, name) for name in keep}
    leaves, treedef = jax.tree.flatten(module)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [0.3 * jax.random.normal(k, x.shape, x.dtype) if eqx.is_inexact_array(x) else x
              for k, x in zip(keys, leaves)]
    module = jax.tree.unflatten(treedef, leaves)
    for name, value in kept.items():
        module = eqx.tree_at(lambda m, name=name: getattr(m, name), module, value)
    return module


def _mamba2(seed=0, **kw) -> Mamba2:
    layer = Mamba2(D_MODEL, _cfg(**kw), key=jax.random.key(seed), std=0.2, norm_eps=1e-5, compute_dtype="float32")
    return _randomize(layer, seed + 50, keep=("A_log", "dt_bias"))


def _mamba3(seed=0, **kw) -> Mamba3:
    layer = Mamba3(D_MODEL, _cfg(**kw), key=jax.random.key(seed), std=0.2, norm_eps=1e-5, compute_dtype="float32")
    return _randomize(layer, seed + 50, keep=("dt_bias",))


def _x(batch=2, length=20, seed=1):
    return jax.random.normal(jax.random.key(seed), (batch, length, D_MODEL))


# ------------------------------------------------------------------ official reference semantics


def _causal_conv(x, weight, bias):
    """Depthwise causal conv, PyTorch layout: ``weight [C, K]``, tap ``K - 1`` is the current token."""
    length, width = x.shape[1], weight.shape[1]
    y = np.broadcast_to(bias, x.shape).copy()
    for i in range(width):
        shift = width - 1 - i
        y[:, shift:] += weight[:, i] * x[:, : length - shift]
    return y


def _ssd_recurrence(x, dt, a, b, c, d):
    """Direct Mamba-2 recurrence: ``h_t = exp(dt A) h + dt B x^T``, ``y = C h + D x`` (heads share groups)."""
    batch, length, heads, head_dim = x.shape
    reps = heads // b.shape[2]
    b, c = np.repeat(b, reps, axis=2), np.repeat(c, reps, axis=2)
    h = np.zeros((batch, heads, b.shape[-1], head_dim))
    y = np.zeros_like(x)
    for t in range(length):
        h = np.exp(dt[:, t] * a)[..., None, None] * h + (dt[:, t, :, None, None] * b[:, t, :, :, None]
                                                         * x[:, t, :, None, :])
        y[:, t] = np.einsum("bhn,bhnp->bhp", c[:, t], h) + d[:, None] * x[:, t]
    return y, h


def mamba3_siso_step_ref(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d=None, z=None, states=None):
    """NumPy port of state-spaces/mamba's ``mamba3_siso_step_ref`` (official layouts).

    ``q, k [B, T, G, N]``, ``v [B, T, H, P]``, ``adt, dt, trap [B, H, T]``,
    ``q_bias, k_bias [H, N]``, ``angles [B, T, H, S]``; states are
    ``(angle [B, H, S], ssm [B, H, P, N], k [B, H, N], v [B, H, P])``.
    """
    batch, length, heads, head_dim = v.shape
    num = angles.shape[-1]
    angles = np.tanh(angles) * math.pi
    q = np.repeat(q, heads // q.shape[2], axis=2)
    k = np.repeat(k, heads // k.shape[2], axis=2)
    if states is None:
        angle = np.zeros((batch, heads, num))
        ssm = np.zeros((batch, heads, head_dim, q.shape[-1]))
        k_state = np.zeros((batch, heads, q.shape[-1]))
        v_state = np.zeros((batch, heads, head_dim))
    else:
        angle, ssm, k_state, v_state = (np.array(s) for s in states)

    def rotary(x, cos, sin):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        pad = pairs.shape[-2] - cos.shape[-1]
        cos = np.concatenate([cos, np.ones(cos.shape[:-1] + (pad,))], -1)
        sin = np.concatenate([sin, np.zeros(sin.shape[:-1] + (pad,))], -1)
        first = pairs[..., 0] * cos - pairs[..., 1] * sin
        second = pairs[..., 0] * sin + pairs[..., 1] * cos
        return np.stack([first, second], -1).reshape(x.shape)

    outs = []
    for t in range(length):
        q_t, k_t, v_t = q[:, t] + q_bias, k[:, t] + k_bias, v[:, t]
        angle = angle + angles[:, t] * dt[:, :, t, None]
        angle = angle - 2 * math.pi * np.floor(angle / (2 * math.pi))
        q_rot = rotary(q_t, np.cos(angle), np.sin(angle))
        k_rot = rotary(k_t, np.cos(angle), np.sin(angle))
        lam = _sigmoid(trap[:, :, t])
        alpha = np.exp(adt[:, :, t])
        beta, gamma = (1 - lam) * dt[:, :, t] * alpha, lam * dt[:, :, t]
        ssm = (alpha[..., None, None] * ssm + beta[..., None, None] * (k_state[:, :, None, :] * v_state[..., None])
               + gamma[..., None, None] * (k_rot[:, :, None, :] * v_t[..., None]))
        out = np.einsum("bhpn,bhn->bhp", ssm, q_rot)
        if d is not None:
            out = out + d[None, :, None] * v_t
        if z is not None:
            out = out * _silu(z[:, t])
        outs.append(out)
        k_state, v_state = k_rot, v_t
    return np.stack(outs, 1), (angle, ssm, k_state, v_state)


def _mamba3_inputs(seed, batch=2, length=19, groups=2, heads=4, n=8, p=6, angles=2):
    ks = jax.random.split(jax.random.key(seed), 12)
    return dict(
        q=jax.random.normal(ks[0], (batch, length, groups, n)),
        k=jax.random.normal(ks[1], (batch, length, groups, n)),
        v=jax.random.normal(ks[2], (batch, length, heads, p)),
        adt=-jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads))),
        dt=jax.nn.softplus(jax.random.normal(ks[4], (batch, length, heads))),
        trap=jax.random.normal(ks[5], (batch, length, heads)),
        q_bias=1.0 + 0.1 * jax.random.normal(ks[6], (heads, n)),
        k_bias=1.0 + 0.1 * jax.random.normal(ks[7], (heads, n)),
        angles=2.0 * jax.random.normal(ks[8], (batch, length, heads, angles)),
        d=jax.random.normal(ks[9], (heads,)),
        z=jax.random.normal(ks[10], (batch, length, heads, p)),
    )


def _to_official(inp):
    t = lambda x: np.moveaxis(_np(x), 1, 2)  # noqa: E731  [B, T, H] -> [B, H, T]
    return (_np(inp["q"]), _np(inp["k"]), _np(inp["v"]), t(inp["adt"]), t(inp["dt"]), t(inp["trap"]),
            _np(inp["q_bias"]), _np(inp["k_bias"]), _np(inp["angles"]))


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("gated", [True, False])
def test_mamba3_core_matches_the_official_step_reference(impl, gated):
    inp = _mamba3_inputs(0)
    z = inp["z"] if gated else None
    y, (angle, state) = mamba3_siso(**{k: inp[k] for k in inp if k not in ("d", "z")}, d=inp["d"], z=z,
                                    output_final_state=True, impl=impl, chunk_size=16)
    want, (w_angle, w_ssm, w_k, w_v) = mamba3_siso_step_ref(*_to_official(inp), d=_np(inp["d"]),
                                                             z=None if z is None else _np(z))
    np.testing.assert_allclose(y, want, **TOL)
    np.testing.assert_allclose(angle, w_angle, **TOL)
    np.testing.assert_allclose(jnp.swapaxes(state.ssm, -1, -2), w_ssm, **TOL)  # ours is [K, V] = [N, P]
    np.testing.assert_allclose(state.key, w_k, **TOL)
    np.testing.assert_allclose(state.value, w_v, **TOL)


def test_mamba3_core_continues_from_a_state_like_the_reference():
    inp = _mamba3_inputs(1, length=23)
    split = 9
    head = {k: (v[:, :split] if v.ndim > 2 and k not in ("q_bias", "k_bias") else v) for k, v in inp.items()}
    tail = {k: (v[:, split:] if v.ndim > 2 and k not in ("q_bias", "k_bias") else v) for k, v in inp.items()}
    common = dict(output_final_state=True, impl="chunked", chunk_size=16)
    core = lambda x, **kw: mamba3_siso(**{k: x[k] for k in x if k not in ("d", "z")}, d=x["d"], z=x["z"],  # noqa
                                       **common, **kw)
    first, state = core(head)
    second, _ = core(tail, initial_state=state)
    want, _ = mamba3_siso_step_ref(*_to_official(inp), d=_np(inp["d"]), z=_np(inp["z"]))
    np.testing.assert_allclose(jnp.concatenate([first, second], axis=1), want, **TOL)
    # Feeding the reference's own states reproduces the tail as well.
    _, ref_state = mamba3_siso_step_ref(*_to_official(head), d=_np(head["d"]), z=_np(head["z"]))
    tail_ref, _ = mamba3_siso_step_ref(*_to_official(tail), d=_np(tail["d"]), z=_np(tail["z"]), states=ref_state)
    np.testing.assert_allclose(second, tail_ref, **TOL)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_mamba2_core_matches_the_direct_ssd_recurrence(impl):
    ks = jax.random.split(jax.random.key(2), 6)
    x = jax.random.normal(ks[0], (2, 21, 4, 6))
    dt = jax.nn.softplus(jax.random.normal(ks[1], (2, 21, 4)))
    a = -jnp.exp(jax.random.normal(ks[2], (4,)))
    b = jax.random.normal(ks[3], (2, 21, 2, 8))
    c = jax.random.normal(ks[4], (2, 21, 2, 8))
    d = jax.random.normal(ks[5], (4,))
    y, state = mamba2_ssd(x, dt, a, b, c, d=d, output_final_state=True, impl=impl, chunk_size=16)
    want, h = _ssd_recurrence(_np(x), _np(dt), _np(a), _np(b), _np(c), _np(d))
    np.testing.assert_allclose(y, want, **TOL)
    np.testing.assert_allclose(state, h, **TOL)


def test_mamba2_layer_follows_the_official_module():
    layer = _mamba2(implementation="reference")
    u = _x(2, 13, seed=3)
    cfg, heads = layer.cfg, layer.num_heads
    un = _np(u)
    zx = np.einsum("btd,dno->btno", un, _np(layer.zx_proj.weight))
    z, x = zx[..., 0, :], zx[..., 1, :]
    bc = np.einsum("btd,dno->btno", un, _np(layer.bc_proj.weight)).reshape(2, 13, -1)
    x = _silu(_causal_conv(x, _np(layer.x_conv.weight), _np(layer.x_conv.bias)))
    bc = _silu(_causal_conv(bc, _np(layer.bc_conv.weight), _np(layer.bc_conv.bias))).reshape(2, 13, 2, 2, 8)
    dt = _softplus(un @ _np(layer.dt_proj.weight) + _np(layer.dt_bias))
    y, _ = _ssd_recurrence(x.reshape(2, 13, heads, cfg.head_dim), dt, -np.exp(_np(layer.A_log)),
                           bc[:, :, 0], bc[:, :, 1], _np(layer.D))
    gated = (y.reshape(2, 13, -1) * _silu(z)).reshape(2, 13, cfg.n_groups, -1)  # norm(y * silu(z)) per group
    normed = _rms(gated, 1e-5).reshape(2, 13, -1) * _np(layer.norm.weight)
    np.testing.assert_allclose(layer(u), normed @ _np(layer.out_proj.weight), **TOL)


@pytest.mark.parametrize("outproj_norm", [False, True])
def test_mamba3_layer_follows_the_official_module(outproj_norm):
    layer = _mamba3(implementation="reference", outproj_norm=outproj_norm)
    u = _x(2, 11, seed=4)
    cfg, heads = layer.cfg, layer.num_heads
    un = _np(u)
    zx = np.einsum("btd,dno->btno", un, _np(layer.zx_proj.weight))
    z = zx[..., 0, :].reshape(2, 11, heads, cfg.head_dim)
    x = zx[..., 1, :].reshape(2, 11, heads, cfg.head_dim)
    bc = np.einsum("btd,dno->btno", un, _np(layer.bc_proj.weight)).reshape(2, 11, 2, cfg.n_groups, cfg.d_state)
    b = _rms(bc[:, :, 0], 1e-5) * _np(layer.B_norm.weight)
    c = _rms(bc[:, :, 1], 1e-5) * _np(layer.C_norm.weight)
    s = np.einsum("btd,dno->btno", un, _np(layer.scalar_proj.weight))
    dt = _softplus(s[..., 0, :] + _np(layer.dt_bias))
    neg, pos = np.minimum(s[..., 1, :], 0), np.maximum(s[..., 1, :], 0)
    a = np.minimum(-(pos + 1 / (1 - neg)), -cfg.a_floor)
    angles = np.broadcast_to((un @ _np(layer.angle_proj.weight))[:, :, None, :], (2, 11, heads, layer.num_angles))
    t = lambda v: np.moveaxis(v, 1, 2)  # noqa: E731
    y, _ = mamba3_siso_step_ref(c, b, x, t(a * dt), t(dt), t(s[..., 2, :]), _np(layer.C_bias), _np(layer.B_bias),
                                angles, d=_np(layer.D), z=None if outproj_norm else z)
    y = y.reshape(2, 11, -1)
    if outproj_norm:  # rmsnorm(y) * w * silu(z) per head
        y = (_rms(y.reshape(2, 11, heads, -1), 1e-5).reshape(2, 11, -1) * _np(layer.norm.weight)
             * _silu(z.reshape(2, 11, -1)))
    np.testing.assert_allclose(layer(u), y @ _np(layer.out_proj.weight), **TOL)


# ------------------------------------------------------------------ helpers


def test_heavy_tail_rotation_and_angle_scan():
    x = jnp.array([-3.0, -0.5, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(heavy_tail(x), [0.25, 1 / 1.5, 1.0, 1.5, 3.0], rtol=1e-6)
    v = jax.random.normal(jax.random.key(5), (3, 8))
    ang = jax.random.normal(jax.random.key(6), (3, 2))
    rotated = rotate_pairs(v, ang)
    complex_v = np.asarray(v[:, 0:4:2]) + 1j * np.asarray(v[:, 1:4:2])
    want = complex_v * np.exp(1j * np.asarray(ang))
    np.testing.assert_allclose(rotated[:, 0:4:2], want.real, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rotated[:, 1:4:2], want.imag, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(rotated[:, 4:], v[:, 4:])
    # 100k steps of +0.1 rad: the segmented mod-2pi scan stays exact where a plain f32 cumsum drifts.
    inc = jnp.full((1, 100_000, 1), 0.1)
    angle = cumulative_angle(inc)
    step = np.float64(np.float32(0.1))  # the increment as stored in float32
    exact = np.mod(step * np.arange(1, 100_001, dtype=np.float64), 2 * math.pi)
    err = np.abs(np.angle(np.exp(1j * (np.asarray(angle[0, :, 0]) - exact))))
    assert err.max() < 1e-3
    seg = jnp.asarray([[0, 0, 1, 1, 1]], jnp.int32)
    out = cumulative_angle(jnp.ones((1, 5, 1)), segment_ids=seg, initial=jnp.full((1, 1), 2.0))
    np.testing.assert_allclose(out[0, :, 0], [3.0, 4.0, 1.0, 2.0, 3.0], rtol=1e-6)


# ------------------------------------------------------------------ layer properties


LAYERS = {"mamba2": _mamba2, "mamba3": _mamba3}


@pytest.mark.parametrize("kind", LAYERS)
def test_shapes_dtypes_and_parameters(kind):
    cls = Mamba2 if kind == "mamba2" else Mamba3
    layer = cls(D_MODEL, _cfg(), key=jax.random.key(0), std=0.02, norm_eps=1e-5)
    out = layer(_x().astype(jnp.bfloat16))
    assert out.shape == (2, 20, D_MODEL) and out.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(out.astype(jnp.float32))).all()
    dt = np.asarray(jax.nn.softplus(layer.dt_bias))
    assert (dt >= 1e-3 * 0.999).all() and (dt <= 0.1 * 1.001).all()
    np.testing.assert_array_equal(layer.D, 1.0)
    if kind == "mamba2":
        a = np.exp(np.asarray(layer.A_log))
        assert (a >= 1.0).all() and (a <= 16.0).all()
        bound = 1 / math.sqrt(layer.cfg.conv_size)
        assert np.abs(np.asarray(layer.x_conv.weight)).max() <= bound and layer.x_conv.bias is not None
    else:
        np.testing.assert_array_equal(layer.B_bias, 1.0)
        np.testing.assert_array_equal(layer.C_bias, 1.0)
        assert layer.num_angles == layer.cfg.d_state // 4  # rope_fraction 0.5 of d_state, two channels per angle


@pytest.mark.parametrize("kind", LAYERS)
def test_future_tokens_never_change_the_past(kind):
    layer = LAYERS[kind](implementation="chunked")
    x = _x()
    x2 = x.at[:, 12:].set(_x(2, 8, seed=7))
    np.testing.assert_allclose(layer(x2)[:, :12], layer(x)[:, :12], **TOL)


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_packed_documents_match_separate_runs(kind, impl):
    layer = LAYERS[kind](implementation=impl)
    lengths = [6, 3, 11]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    x = _x(1, 20, seed=8)
    packed = layer(x, seg)
    start = 0
    for n in lengths:
        np.testing.assert_allclose(packed[:, start : start + n], layer(x[:, start : start + n]), **TOL)
        start += n


@pytest.mark.parametrize("kind", LAYERS)
def test_chunked_matches_reference(kind):
    x = _x()
    np.testing.assert_allclose(LAYERS[kind](implementation="chunked")(x), LAYERS[kind](implementation="reference")(x),
                               **TOL)


@pytest.mark.parametrize("kind", LAYERS)
def test_prefill_then_decode_matches_the_full_pass(kind):
    layer = LAYERS[kind](implementation="chunked")
    x = _x(2, 15, seed=9)
    full = layer(x)
    cache = layer.init_cache(2, jnp.float32)
    out, cache = layer.step(x[:, :6], cache)
    outs = [out]
    for t in range(6, 15):
        out, cache = layer.step(x[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)
    if kind == "mamba3":  # the carried state holds the last token's rotated key and its value
        assert cache.angle.shape == (2, layer.num_heads, layer.num_angles)
        assert cache.state.key.shape == (2, layer.num_heads, layer.cfg.d_state)


def test_config_validation():
    with pytest.raises(ValueError, match="rope_fraction"):
        SSMConfig(rope_fraction=0.25)
    with pytest.raises(ValueError, match="multiple of head_dim"):
        SSMConfig(head_dim=48).num_heads(64)
    with pytest.raises(ValueError, match="groups"):
        SSMConfig(head_dim=16, n_groups=3).num_heads(32)
