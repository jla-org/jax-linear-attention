# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/ops/selective_scan_interface.py (selective_scan_ref),
# mamba_ssm/modules/mamba_simple.py
"""Mamba-1: the ``mamba1`` op, the layer and the presets against independent numpy ports of the official code."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla import ops
from jla.layers.mamba1 import Mamba1
from jla.models import LOGICAL_AXIS_NAMES, build_model, get_preset, logical_axes, parameter_count
from jla.models.config import SSMConfig
from jla.ops.cross_entropy import linear_cross_entropy

TOL = dict(rtol=1e-4, atol=1e-5)


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def selective_scan_naive(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, h0=None):
    """float64 numpy port of ``selective_scan_ref`` (torch layout: ``u, delta, z [B, D, L]``,
    ``B, C [B, N, L]`` or ``[B, G, N, L]``), extended with an initial state ``h0 [B, D, N]``."""
    u = u.astype(np.float64)
    delta = delta.astype(np.float64)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = _softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    x = np.zeros((batch, dim, dstate)) if h0 is None else np.asarray(h0, np.float64)
    delta_a = np.exp(np.einsum("bdl,dn->bdln", delta, A))
    if B.ndim == 3:
        delta_b_u = np.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        B = np.repeat(B, dim // B.shape[1], axis=1)
        delta_b_u = np.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if C.ndim == 4:
        C = np.repeat(C, dim // C.shape[1], axis=1)
    ys = []
    for i in range(u.shape[2]):
        x = delta_a[:, :, i] * x + delta_b_u[:, :, i]
        ys.append(np.einsum("bdn,bn->bd", x, C[:, :, i]) if C.ndim == 3 else
                  np.einsum("bdn,bdn->bd", x, C[:, :, :, i]))
    out = np.stack(ys, axis=2)
    if D is not None:
        out = out + u * D[:, None]
    if z is not None:
        out = out * _silu(z)
    return out, x


def _inputs(seed=0, batch=2, length=37, dim=12, dstate=5, groups=None):
    rng = np.random.default_rng(seed)
    bc_shape = (batch, length, dstate) if groups is None else (batch, length, groups, dstate)
    return dict(
        q=rng.standard_normal(bc_shape).astype(np.float32),
        k=rng.standard_normal(bc_shape).astype(np.float32),
        v=rng.standard_normal((batch, length, dim)).astype(np.float32),
        dt=(0.5 * rng.standard_normal((batch, length, dim))).astype(np.float32),
        a=-np.exp(rng.uniform(-1.0, 1.5, (dim, dstate))).astype(np.float32),
        d=rng.standard_normal(dim).astype(np.float32),
        z=rng.standard_normal((batch, length, dim)).astype(np.float32),
        dt_bias=rng.uniform(-3.0, -1.0, dim).astype(np.float32),
        h0=rng.standard_normal((batch, dim, dstate)).astype(np.float32),
    )


def _torch_layout(x):
    """``[B, T, ...]`` -> ``[B, ..., T]``."""
    return np.moveaxis(x, 1, -1)


def _naive(inp, sl=slice(None), h0=None, **flags):
    t = lambda x: _torch_layout(x[:, sl])  # noqa: E731
    return selective_scan_naive(t(inp["v"]), t(inp["dt"]), inp["a"], t(inp["k"]), t(inp["q"]), inp["d"],
                                t(inp["z"]), delta_bias=inp["dt_bias"], delta_softplus=True, h0=h0, **flags)


def _op(inp, impl, **options):
    j = {key: jnp.asarray(value) for key, value in inp.items()}
    return ops.linear_attention("mamba1", j["q"], j["k"], j["v"], dt=j["dt"], a=j["a"], d=j["d"], z=j["z"],
                                dt_bias=j["dt_bias"], dt_softplus=True, impl=impl, **options)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("groups", [None, 3])
def test_op_matches_selective_scan_ref(impl, groups):
    inp = _inputs(groups=groups)
    want, last = _naive(inp, h0=inp["h0"])
    o, final = _op(inp, impl, initial_state=jnp.asarray(inp["h0"]), output_final_state=True, chunk_size=16)
    np.testing.assert_allclose(np.asarray(o), np.moveaxis(want, -1, 1), **TOL)
    np.testing.assert_allclose(np.asarray(final), last, **TOL)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_op_without_optional_inputs_matches_selective_scan_ref(impl):
    inp = _inputs(seed=3)
    t = _torch_layout
    want, _ = selective_scan_naive(t(inp["v"]), t(inp["dt"]), inp["a"], t(inp["k"]), t(inp["q"]))
    j = {key: jnp.asarray(value) for key, value in inp.items()}
    o, final = ops.linear_attention("mamba1", j["q"], j["k"], j["v"], dt=j["dt"], a=j["a"], impl=impl)
    assert final is None
    np.testing.assert_allclose(np.asarray(o), np.moveaxis(want, -1, 1), **TOL)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_packed_segments_match_separate_runs(impl):
    inp = _inputs(seed=1)
    lengths = [10, 4, 23]
    seg = np.repeat(np.arange(3), lengths)[None].repeat(2, axis=0)
    o, final = _op(inp, impl, segment_ids=jnp.asarray(seg, jnp.int32), initial_state=jnp.asarray(inp["h0"]),
                   output_final_state=True, chunk_size=16)
    start = 0
    for i, n in enumerate(lengths):
        want, last = _naive(inp, slice(start, start + n), h0=inp["h0"] if i == 0 else None)
        np.testing.assert_allclose(np.asarray(o[:, start : start + n]), np.moveaxis(want, -1, 1), **TOL)
        start += n
    np.testing.assert_allclose(np.asarray(final), last, **TOL)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_state_carries_across_calls(impl):
    inp = _inputs(seed=2)
    want, last = _naive(inp)
    first = {key: (value[:, :20] if value.ndim == 3 and value.shape[1] == 37 else value) for key, value in inp.items()}
    rest = {key: (value[:, 20:] if value.ndim == 3 and value.shape[1] == 37 else value) for key, value in inp.items()}
    o1, h = _op(first, impl, output_final_state=True, chunk_size=16)
    o2, final = _op(rest, impl, initial_state=h, output_final_state=True, chunk_size=16)
    np.testing.assert_allclose(np.concatenate([o1, o2], axis=1), np.moveaxis(want, -1, 1), **TOL)
    np.testing.assert_allclose(np.asarray(final), last, **TOL)


def test_chunked_gradients_match_the_reference():
    inp = _inputs(seed=4, length=29)
    names = ("q", "k", "v", "dt", "a", "d", "z", "dt_bias", "h0")
    w = jnp.asarray(np.random.default_rng(9).standard_normal((2, 29, 12)), jnp.float32)

    def loss(impl, *args):
        x = dict(zip(names, args))
        o, h = ops.linear_attention("mamba1", x["q"], x["k"], x["v"], dt=x["dt"], a=x["a"], d=x["d"], z=x["z"],
                                    dt_bias=x["dt_bias"], dt_softplus=True, initial_state=x["h0"],
                                    output_final_state=True, impl=impl, chunk_size=8)
        return jnp.sum(o * w) + jnp.sum(h**2)

    args = [jnp.asarray(inp[name]) for name in names]
    grads = {impl: jax.grad(lambda *a, impl=impl: loss(impl, *a), argnums=tuple(range(len(names))))(*args)
             for impl in ("reference", "chunked")}
    for name, g_ref, g_chunk in zip(names, grads["reference"], grads["chunked"]):
        np.testing.assert_allclose(np.asarray(g_chunk), np.asarray(g_ref), rtol=2e-4, atol=2e-5, err_msg=name)


@pytest.mark.parametrize("groups", [None, 3])
@pytest.mark.parametrize("optional", [True, False])
def test_chunked_reference_gradient_path(monkeypatch, groups, optional):
    """The GPU path of ``chunked.mamba1`` (chunked forward, the reference op's gradient), forced on here."""
    from jla.ops import chunked

    monkeypatch.setattr(chunked, "_mamba1_differentiates_via_reference", lambda: True)
    inp = _inputs(seed=5, length=29, groups=groups)
    names = ("q", "k", "v", "dt", "a", "d", "z", "dt_bias", "h0") if optional else ("q", "k", "v", "dt", "a")
    seg = jnp.asarray(np.repeat([[0] * 11 + [1] * 18], 2, axis=0), jnp.int32)
    w = jnp.asarray(np.random.default_rng(9).standard_normal((2, 29, 12)), jnp.float32)

    def loss(impl, *args):
        x = dict(zip(names, args))
        o, h = ops.linear_attention("mamba1", x["q"], x["k"], x["v"], dt=x["dt"], a=x["a"], d=x.get("d"),
                                    z=x.get("z"), dt_bias=x.get("dt_bias"), dt_softplus=True,
                                    initial_state=x.get("h0"), output_final_state=True, segment_ids=seg,
                                    impl=impl, chunk_size=8)
        return jnp.sum(o * w) + jnp.sum(h**2)

    args = [jnp.asarray(inp[name]) for name in names]
    outs = {impl: jax.jit(lambda *a, impl=impl: loss(impl, *a))(*args) for impl in ("reference", "chunked")}
    np.testing.assert_allclose(np.asarray(outs["chunked"]), np.asarray(outs["reference"]), rtol=1e-5)
    grads = {impl: jax.jit(jax.grad(lambda *a, impl=impl: loss(impl, *a), argnums=tuple(range(len(names)))))(*args)
             for impl in ("reference", "chunked")}
    for name, g_ref, g_chunk in zip(names, grads["reference"], grads["chunked"]):
        np.testing.assert_allclose(np.asarray(g_chunk), np.asarray(g_ref), rtol=1e-5, atol=1e-5, err_msg=name)


def test_dispatcher_fails_loudly():
    inp = {key: jnp.asarray(value) for key, value in _inputs().items()}
    with pytest.raises(ValueError, match="missing"):
        ops.linear_attention("mamba1", inp["q"], inp["k"], inp["v"], dt=inp["dt"])
    with pytest.raises(ValueError, match="cu_seqlens"):
        ops.linear_attention("mamba1", inp["q"], inp["k"], inp["v"], dt=inp["dt"], a=inp["a"], impl="reference",
                             cu_seqlens=jnp.asarray([0, 37], jnp.int32))
    with pytest.raises(ValueError, match="a must be"):
        ops.linear_attention("mamba1", inp["q"], inp["k"], inp["v"], dt=inp["dt"], a=inp["a"][:3], impl="reference")
    assert ops.available("mamba1")[-2:] == ("chunked", "reference")


def _conv_naive(x, weight, bias):
    """torch ``conv1d(groups=C, padding=K - 1)[..., :T]`` on ``x [B, T, C]``, ``weight [C, K]``."""
    width = weight.shape[1]
    out = np.zeros_like(x) + bias
    for t in range(x.shape[1]):
        for j in range(width):
            s = t - (width - 1) + j
            if s >= 0:
                out[:, t] += weight[:, j] * x[:, s]
    return out


def _layer_naive(layer: Mamba1, u):
    """float64 numpy port of ``Mamba.forward`` (``mamba_simple.py``, the non-fused path)."""
    p = lambda x: np.asarray(x, np.float64)  # noqa: E731
    w_in = p(layer.in_proj.weight)
    x, z = u @ w_in[:, 0], u @ w_in[:, 1]
    x = _silu(_conv_naive(x, p(layer.conv.weight), p(layer.conv.bias)))
    x_dbl = x @ p(layer.x_proj.weight)
    rank, n = layer.dt_rank, layer.cfg.d_state
    dt = x_dbl[..., :rank] @ p(layer.dt_proj.weight)
    b, c = x_dbl[..., rank : rank + n], x_dbl[..., rank + n :]
    t = _torch_layout
    y, _ = selective_scan_naive(t(x), t(dt), -np.exp(p(layer.A_log)), t(b), t(c), p(layer.D), t(z),
                                delta_bias=p(layer.dt_bias), delta_softplus=True)
    return np.moveaxis(y, -1, 1) @ p(layer.out_proj.weight)


def _layer(impl="auto", **changes):
    cfg = SSMConfig(d_state=8, head_dim=16, chunk_size=16, implementation=impl, **changes)
    return Mamba1(32, cfg, key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="float32")


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_layer_matches_the_official_module(impl):
    layer = _layer(impl)
    u = np.random.default_rng(5).standard_normal((2, 21, 32)).astype(np.float32)
    with jax.default_matmul_precision("highest"):  # no TF32 on GPU: the oracle is float64
        out = np.asarray(layer(jnp.asarray(u)))
    np.testing.assert_allclose(out, _layer_naive(layer, u.astype(np.float64)), rtol=2e-4, atol=2e-5)


def test_layer_official_init():
    layer = _layer(dt_scale=2.0)
    assert layer.dt_rank == 2 and layer.x_proj.weight.shape == (64, 2 + 16)
    bound = 2.0 / np.sqrt(2)
    weight = np.asarray(layer.dt_proj.weight)
    assert weight.shape == (2, 64) and np.abs(weight).max() <= bound and np.abs(weight).max() > 0.5 * bound
    assert layer.dt_proj.bias is None
    dt = _softplus(np.asarray(layer.dt_bias, np.float64))
    assert dt.min() >= 0.001 * (1 - 1e-4) and dt.max() <= 0.1 * (1 + 1e-4)
    np.testing.assert_allclose(np.exp(np.asarray(layer.A_log)), np.tile(np.arange(1, 9), (64, 1)), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(layer.D), np.ones(64))
    with pytest.raises(ValueError, match="n_groups"):
        Mamba1(32, SSMConfig(d_state=8, head_dim=16, n_groups=2), key=jax.random.key(0), std=0.02, norm_eps=1e-5)


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_layer_prefill_then_decode_matches_the_forward_pass(impl):
    layer = _layer(impl)
    u = jnp.asarray(np.random.default_rng(6).standard_normal((2, 13, 32)), jnp.float32)
    full = layer(u)
    cache = layer.init_cache(2, jnp.float32)
    out, cache = layer.step(u[:, :5], cache)
    outs = [out]
    for t in range(5, 13):
        out, cache = layer.step(u[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(np.asarray(jnp.concatenate(outs, axis=1)), np.asarray(full), **TOL)


def _f32(name: str, **changes):
    cfg = get_preset(name).replace(compute_dtype="float32", **changes)
    if cfg.moe is not None:
        cfg = cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None))
    return cfg


def _ids(batch=2, length=16, seed=1):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, 512)


@pytest.mark.parametrize("name", ["tiny-mamba1", "tiny-jamba"])
@pytest.mark.parametrize("scan", [True, False])
def test_lm_prefill_then_decode_matches_the_forward_pass(name, scan):
    model = build_model(_f32(name, scan_layers=scan), key=jax.random.key(5))
    ids = _ids(2, 14, seed=6)
    # No TF32 on GPU: the full pass and the decode steps multiply matrices of different shapes, whose TF32
    # roundings differ by more than the tolerance at a few of the tiny-jamba logits.
    with jax.default_matmul_precision("highest"):
        full = model.logits(model(ids).hidden)
        pos = jnp.broadcast_to(jnp.arange(14, dtype=jnp.int32), (2, 14))
        step = jax.jit(lambda m, c, i, p: m.decode_step(c, i, p))
        cache = model.init_cache(2, 20)
        logits, cache = step(model, cache, ids[:, :6], pos[:, :6])
        outs = [logits]
        for t in range(6, 14):
            logits, cache = step(model, cache, ids[:, t : t + 1], pos[:, t : t + 1])
            outs.append(logits)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("name", ["tiny-mamba1", "tiny-jamba"])
def test_lm_packed_documents_match_separate_runs(name):
    cfg = _f32(name)
    cfg = cfg.replace(attention=dataclasses.replace(cfg.attention, implementation="xla"))
    model = build_model(cfg, key=jax.random.key(0))
    lengths = [7, 3, 6]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = _ids(1, 16, seed=2)
    with jax.default_matmul_precision("highest"):  # no TF32 on GPU: packed and separate runs tile differently
        packed = model(ids, positions=pos, segment_ids=seg).hidden
        start = 0
        for n in lengths:
            alone = model(ids[:, start : start + n]).hidden
            np.testing.assert_allclose(packed[:, start : start + n], alone, rtol=1e-4, atol=1e-5)
            start += n


@pytest.mark.parametrize("name", ["tiny-mamba1", "tiny-jamba"])
@pytest.mark.parametrize("scan", [True, False])
def test_lm_logical_axes_cover_every_parameter(name, scan):
    model = build_model(get_preset(name).replace(scan_layers=scan), key=jax.random.key(0))
    pairs = {}
    params = eqx.filter(model, eqx.is_array)
    jax.tree_util.tree_map_with_path(
        lambda path, x, ax: pairs.__setitem__(jax.tree_util.keystr(path), (x.shape, ax)), params,
        logical_axes(model),
    )
    assert len(pairs) == len(jax.tree.leaves(params))
    for path, (shape, ax) in pairs.items():
        assert isinstance(ax, tuple) and len(ax) == len(shape), path
        assert set(ax) <= set(LOGICAL_AXIS_NAMES) | {None}, path
    assert any(path.endswith(".mixer.A_log") for path in pairs)


def test_mamba1_130m_matches_the_official_layer_layout():
    # state-spaces/mamba-130m: 24 layers of in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj and a norm.
    # The per-layer count matches the official model exactly; the preset's vocabulary is the project-wide
    # 50304 instead of the official 50280 (50277 padded), so the total is 24 * 768 = 18,432 larger.
    d, inner, n, rank = 768, 1536, 16, 48
    per_layer = (d * 2 * inner + inner * 4 + inner + inner * (rank + 2 * n) + rank * inner + inner
                 + inner * n + inner + inner * d + d)
    count = parameter_count(get_preset("mamba1-130m"))
    assert count.total == 24 * per_layer + 50304 * d + d
    assert count.total - (24 * per_layer + 50280 * d + d) == 18_432


def test_mamba1_rejects_mamba2_and_mamba3_only_options():
    from jla.layers.mamba1 import Mamba1
    from jla.models.config import SSMConfig

    for options in (dict(outproj_norm=True), dict(a_floor=1e-3), dict(rope_fraction=1.0), dict(a_init_range=(1.0, 8.0))):
        with pytest.raises(ValueError, match="does not use the Mamba-2/3 options"):
            Mamba1(32, SSMConfig(d_state=4, head_dim=16, **options), key=jax.random.PRNGKey(0), std=0.02, norm_eps=1e-5)


def _loss(model, ids):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    return ce.loss_sum / ce.num_valid + out.aux_loss


@pytest.mark.parametrize("name", ["tiny-mamba1", "tiny-jamba"])
def test_cpu_train_steps_reduce_the_loss(name):
    model = build_model(get_preset(name), key=jax.random.key(0))
    ids = _ids(4, 32, seed=3)
    opt = optax.adam(3e-3)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, state):
        loss, grads = eqx.filter_value_and_grad(_loss)(model, ids)
        updates, state = opt.update(grads, state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, loss

    losses = []
    for _ in range(8):
        model, state, loss = train_step(model, state)
        losses.append(float(loss))
    assert all(np.isfinite(losses)) and losses[-1] < losses[0] - 0.1, losses
