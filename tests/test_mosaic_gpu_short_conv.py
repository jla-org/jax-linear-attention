# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU short conv (jla.ops.mosaic_gpu.short_conv) against the XLA ``ShortConv``.

Every case runs the kernels in bfloat16 and compares the output and the gradients of x, the
weight, the bias and the initial state with an float32 oracle (the XLA path on the same values
upcast to float32), and with the XLA path in bfloat16 (the kernels may not be less accurate).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.conv import ShortConv
from jla.ops.mosaic_gpu._common import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="needs an sm_90 GPU")


def _segments(key, batch, seq_len):
    """Random packed documents (ids non-decreasing, several boundaries per row, one at t = 1)."""
    starts = jax.random.bernoulli(key, 0.08, (batch, seq_len)).at[:, 1].set(True).at[:, 0].set(False)
    return jnp.cumsum(starts.astype(jnp.int32), axis=1)


def _inputs(seed, batch, seq_len, channels, width, *, bias=True, state=False, segments=False):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    x = jax.random.normal(ks[0], (batch, seq_len, channels), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(ks[1], (channels, width), jnp.float32) * 0.5
    b = jax.random.normal(ks[2], (channels,), jnp.float32) * 0.5 if bias else None
    s = jax.random.normal(ks[3], (batch, width - 1, channels), jnp.float32).astype(jnp.bfloat16) if state else None
    seg = _segments(ks[4], batch, seq_len) if segments else None
    dy = jax.random.normal(ks[5], (batch, seq_len, channels), jnp.float32).astype(jnp.bfloat16)
    return x, w, b, s, seg, dy


def eqx_set(conv, w, b):
    conv = eqx.tree_at(lambda c: c.weight, conv, w)
    if b is not None:
        conv = eqx.tree_at(lambda c: c.bias, conv, b)
    return conv


def _run(impl, x, w, b, s, seg, dy, *, activation, dtype):
    """``(y, new_state, dx, dw, db, dstate)`` of ``ShortConv`` with ``impl`` on inputs cast to ``dtype``."""
    x = x.astype(dtype)
    s = None if s is None else s.astype(dtype)

    def loss(x, w, b, s):
        conv = eqx_set(ShortConv(w.shape[0], w.shape[1], activation=activation, use_bias=b is not None, impl=impl),
                       w, b)
        y, new_state = conv(x, segment_ids=seg, state=s)
        return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32)), (y, new_state)

    grads, (y, new_state) = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True))(x, w, b, s)
    return (y, new_state, *grads)


def _err(a, ref):
    if a is None:
        assert ref is None
        return 0.0
    a, ref = np.asarray(a, np.float32), np.asarray(ref, np.float32)
    assert a.shape == ref.shape
    return float(np.max(np.abs(a - ref)) / max(float(np.max(np.abs(ref))), 1e-6))


_NAMES = ("y", "new_state", "dx", "dw", "db", "dstate")


def _check(x, w, b, s, seg, dy, *, activation=True):
    ours = _run("mosaic_gpu", x, w, b, s, seg, dy, activation=activation, dtype=jnp.bfloat16)
    xla = _run("xla", x, w, b, s, seg, dy, activation=activation, dtype=jnp.bfloat16)
    oracle = _run("xla", x, w, b, s, seg, dy, activation=activation, dtype=jnp.float32)
    report = {}
    for name, o, xl, ref in zip(_NAMES, ours, xla, oracle):
        e_ours, e_xla = _err(o, ref), _err(xl, ref)
        report[name] = (e_ours, e_xla)
        # bf16 tensors: a few bf16 ulps of the largest value; f32 weight gradients: summation order only
        limit = 1e-4 if name in ("dw", "db") else 1.6e-2
        assert e_ours <= max(limit, 1.5 * e_xla), (name, report)
    np.testing.assert_array_equal(np.asarray(ours[1]), np.asarray(xla[1]))  # the state is a copy of inputs
    return report


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("segments", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_matches_xla_odd_length(width, segments, state):
    _check(*_inputs(width + 10 * segments + 100 * state, 2, 37, 256, width, state=state, segments=segments))


@pytest.mark.parametrize("activation,bias", [(False, False), (True, False), (False, True)])
def test_activation_and_bias_options(activation, bias):
    _check(*_inputs(7, 2, 50, 384, 4, bias=bias, segments=True), activation=activation)


@pytest.mark.parametrize("channels", [256, 1024, 1536, 2048, 3072, 4096])  # the 1.3B presets' convs
def test_preset_channel_counts(channels):
    _check(*_inputs(channels, 2, 300, channels, 4, state=True, segments=True))


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 64, 65, 127, 128, 129, 191])
def test_block_edges(seq_len):
    """Shorter than the state, the block sizes (64 forward, 128 backward) and the shifted last block."""
    _check(*_inputs(seq_len, 3, seq_len, 512, 4, state=True, segments=seq_len > 2))


def test_long_sequence_many_documents():
    x, w, b, s, _, dy = _inputs(11, 2, 4096, 2048, 4)
    seg = jnp.cumsum(jax.random.bernoulli(jax.random.PRNGKey(3), 0.01, (2, 4096)).astype(jnp.int32), axis=1)
    _check(x, w, b, s, seg, dy)


def test_prefill_then_decode_matches_full_sequence():
    """Prefill with the kernels, then decode token by token (each step a T = 1 kernel call)."""
    x, w, b, _, _, _ = _inputs(5, 2, 24, 256, 4)
    conv = eqx_set(ShortConv(256, 4, use_bias=True, impl="mosaic_gpu"), w, b)
    full, _ = conv(x)
    y, state = conv(x[:, :16])
    outs = [y]
    for t in range(16, 24):
        y, state = conv(x[:, t:t + 1], state=state)
        outs.append(y)
    np.testing.assert_array_equal(np.asarray(jnp.concatenate(outs, axis=1), np.float32),
                                  np.asarray(full, np.float32))


def test_documents_do_not_leak():
    """Changing one document's tokens leaves every other document's outputs unchanged."""
    x, w, b, _, _, _ = _inputs(9, 1, 96, 256, 4)
    seg = jnp.concatenate([jnp.zeros((1, 40), jnp.int32), jnp.ones((1, 56), jnp.int32)], axis=1)
    conv = eqx_set(ShortConv(256, 4, use_bias=True, impl="mosaic_gpu"), w, b)
    y0, _ = conv(x, segment_ids=seg)
    y1, _ = conv(x.at[:, :40].multiply(-3.0), segment_ids=seg)
    np.testing.assert_array_equal(np.asarray(y0[:, 40:], np.float32), np.asarray(y1[:, 40:], np.float32))


def test_auto_and_unsupported():
    from jla.ops.mosaic_gpu.short_conv import unsupported_reason

    x = jnp.zeros((1, 8, 200), jnp.bfloat16)  # 200 channels: not a multiple of 128
    w = jnp.zeros((200, 4), jnp.float32)
    assert unsupported_reason(x, w) is not None
    conv = ShortConv(200, 4, impl="auto")
    y, _ = conv(x)  # falls back to XLA
    assert y.shape == x.shape
    with pytest.raises(ValueError, match="mosaic_gpu"):
        ShortConv(200, 4, impl="mosaic_gpu")(x)
    assert unsupported_reason(jnp.zeros((1, 8, 256), jnp.float32), jnp.zeros((256, 4))) is not None  # f32 input
    assert unsupported_reason(jnp.zeros((1, 8, 256), jnp.bfloat16), jnp.zeros((256, 5))) is not None  # width 5


@pytest.mark.parametrize("preset,field", [("mamba2-1p3b", "ssm"), ("kda-1p3b", "linear_attention"),
                                          ("gdn-hd128-1p3b", "linear_attention"), ("gdn2-1p3b", "linear_attention")])
def test_model_step_matches_xla(preset, field):
    """One layer of each release preset with packed documents: loss and every gradient, conv on XLA vs kernels."""
    import dataclasses

    from jla.models import build_model, get_preset

    base = get_preset(preset)
    cfgs = {}
    for impl in ("xla", "mosaic_gpu"):
        cfgs[impl] = base.replace(
            n_layers=1, layer_types=base.layer_types[:1], vocab_size=512,
            mlp_types=None if base.mlp_types is None else base.mlp_types[:1],
            **{field: dataclasses.replace(getattr(base, field), conv_impl=impl, **(
                {"qk_l2norm_in_conv": False} if field == "linear_attention" else {}))})  # fused: *_l2norm.py
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 192), 0, 512)
    seg = jnp.cumsum(jax.random.bernoulli(jax.random.PRNGKey(2), 0.02, (2, 192)).astype(jnp.int32), axis=1)
    probe = jax.random.normal(jax.random.PRNGKey(3), (2, 192, base.d_model), jnp.float32)

    def loss(model):
        out = model(ids, segment_ids=seg)
        return jnp.sum(out.hidden.astype(jnp.float32) * probe)

    results = {}
    for impl, cfg in cfgs.items():
        model = build_model(cfg, key=jax.random.PRNGKey(0))
        results[impl] = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)
    (l_x, g_x), (l_k, g_k) = results["xla"], results["mosaic_gpu"]
    assert abs(float(l_k) - float(l_x)) <= 2e-2 * abs(float(l_x)) + 1e-3
    leaves_x = jax.tree_util.tree_leaves(eqx.filter(g_x, eqx.is_inexact_array))
    leaves_k = jax.tree_util.tree_leaves(eqx.filter(g_k, eqx.is_inexact_array))
    assert len(leaves_x) == len(leaves_k)
    for a, b in zip(leaves_k, leaves_x):
        assert _err(a, b) <= 5e-2, (a.shape, _err(a, b))


def test_reused_segment_ids_follow_equality():
    """Ids are compared for equality, as in the XLA path (a reused id ``[0, 1, 0]`` keeps its taps)."""
    x, w, b, s, _, dy = _inputs(13, 2, 70, 256, 4, state=True)
    seg = jax.random.randint(jax.random.PRNGKey(4), (2, 70), 0, 3)
    _check(x, w, b, s, seg, dy)


def test_float16_inputs():
    x, w, b, s, seg, dy = _inputs(17, 2, 45, 512, 3, state=True, segments=True)
    ours = _run("mosaic_gpu", x, w, b, s, seg, dy, activation=True, dtype=jnp.float16)
    oracle = _run("xla", x, w, b, s, seg, dy, activation=True, dtype=jnp.float32)
    assert ours[0].dtype == jnp.float16 and ours[2].dtype == jnp.float16
    for name, o, ref in zip(_NAMES, ours, oracle):
        assert _err(o, ref) <= (1e-4 if name in ("dw", "db") else 2e-3), name


@pytest.mark.parametrize("batch,block_t", [(8, 256), (4, 128), (1, 64)])
def test_backward_block_sizes(batch, block_t):
    """Grids large enough for the 256- and 128-token backward blocks (ragged T: shifted last block)."""
    from jla.ops.mosaic_gpu import short_conv as kernels

    x, w, b, s, seg, dy = _inputs(batch, batch, 4133, 2048, 4, state=True, segments=True)
    plan = kernels._plan(x, 4, True, True, True, backward=True)
    assert plan.block_t == block_t, plan
    _check(x, w, b, s, seg, dy)
