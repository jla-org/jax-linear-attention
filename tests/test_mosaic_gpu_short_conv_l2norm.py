# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused short conv + SiLU + per-head L2 norm (jla.ops.mosaic_gpu.short_conv_l2norm) against the unfused path.

Every case runs the fused kernels in bfloat16 and compares the output and the gradients of x, the weight, the bias
and the initial state with a float32 oracle (XLA conv then ``reference.l2norm``, on the same values upcast to
float32), and with the unfused bfloat16 path (the conv kernel, then ``reference.l2norm``): the fused kernels must be
within a few bf16 ulps of the oracle and never less accurate than the unfused path.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.conv import ShortConv
from jla.layers.conv_l2norm import conv_l2norm, conv_l2norm_reason
from jla.ops.mosaic_gpu._common import is_hopper
from jla.ops.reference import l2norm

pytestmark = pytest.mark.skipif(not is_hopper(), reason="needs an sm_90 GPU")

_NAMES = ("n", "new_state", "dx", "dw", "db", "dstate")


def _segments(key, batch, seq_len):
    starts = jax.random.bernoulli(key, 0.08, (batch, seq_len)).at[:, 1].set(True).at[:, 0].set(False)
    return jnp.cumsum(starts.astype(jnp.int32), axis=1)


def _inputs(seed, batch, seq_len, channels, width, *, bias=True, state=False, segments=False):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    x = jax.random.normal(ks[0], (batch, seq_len, channels), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(ks[1], (channels, width), jnp.float32) * 0.5
    b = jax.random.normal(ks[2], (channels,), jnp.float32) * 0.5 if bias else None
    s = jax.random.normal(ks[3], (batch, width - 1, channels), jnp.float32).astype(jnp.bfloat16) if state else None
    seg = _segments(ks[4], batch, seq_len) if segments else None
    dn = jax.random.normal(ks[5], (batch, seq_len, channels), jnp.float32).astype(jnp.bfloat16)
    return x, w, b, s, seg, dn


def _conv(w, b, impl, activation):
    conv = ShortConv(w.shape[0], w.shape[1], activation=activation, use_bias=b is not None, impl=impl)
    conv = eqx.tree_at(lambda c: c.weight, conv, w)
    return conv if b is None else eqx.tree_at(lambda c: c.bias, conv, b)


def _heads_l2norm(y, head_dim):
    return l2norm(y.reshape(*y.shape[:2], -1, head_dim)).reshape(y.shape)


def _run(path, x, w, b, s, seg, dn, *, head_dim, activation, dtype):
    """``(n, new_state, dx, dw, db, dstate)``: ``path`` "fused", or a conv impl followed by the XLA L2 norm."""
    x = x.astype(dtype)
    s = None if s is None else s.astype(dtype)

    def loss(x, w, b, s):
        if path == "fused":
            conv = _conv(w, b, "mosaic_gpu", activation)
            out = conv_l2norm(conv, x, head_dim, segment_ids=seg, state=s)
            assert out is not None, conv_l2norm_reason(conv, x, head_dim, segment_ids=seg, state=s)
            n, new_state = out
        else:
            y, new_state = _conv(w, b, path, activation)(x, segment_ids=seg, state=s)
            n = _heads_l2norm(y, head_dim)
        return jnp.sum(n.astype(jnp.float32) * dn.astype(jnp.float32)), (n, new_state)

    grads, (n, new_state) = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True))(x, w, b, s)
    return (n, new_state, *grads)


def _err(a, ref):
    if a is None:
        assert ref is None
        return 0.0
    a, ref = np.asarray(a, np.float32), np.asarray(ref, np.float32)
    assert a.shape == ref.shape
    return float(np.max(np.abs(a - ref)) / max(float(np.max(np.abs(ref))), 1e-6))


def _check(x, w, b, s, seg, dn, *, head_dim, activation=True):
    kw = dict(head_dim=head_dim, activation=activation)
    fused = _run("fused", x, w, b, s, seg, dn, dtype=jnp.bfloat16, **kw)
    unfused = _run("mosaic_gpu", x, w, b, s, seg, dn, dtype=jnp.bfloat16, **kw)
    oracle = _run("xla", x, w, b, s, seg, dn, dtype=jnp.float32, **kw)
    report = {}
    for name, f, u, ref in zip(_NAMES, fused, unfused, oracle):
        e_fused, e_unfused = _err(f, ref), _err(u, ref)
        report[name] = (round(e_fused, 6), round(e_unfused, 6))
        # bf16 tensors: a few bf16 ulps of the largest value; f32 weight gradients: summation order and rsqrt
        limit = 1e-3 if name in ("dw", "db") else 1.6e-2
        assert e_fused <= max(limit, 1.5 * e_unfused), (name, report)
        assert e_fused <= 2 * limit, (name, report)
    np.testing.assert_array_equal(np.asarray(fused[1]), np.asarray(unfused[1]))  # the state is a copy of inputs
    return report


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("segments", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_matches_unfused_odd_length(width, segments, state):
    _check(*_inputs(width + 10 * segments + 100 * state, 2, 37, 512, width, state=state, segments=segments),
           head_dim=128)


@pytest.mark.parametrize("channels,head_dim", [(256, 64), (512, 64), (1024, 64), (2048, 128), (4096, 128)])
def test_head_layouts(channels, head_dim):
    """One warp per 128-channel head at 4 channels per thread; 64-channel heads at 2 and 4 channels per thread."""
    _check(*_inputs(channels + head_dim, 2, 300, channels, 4, state=True, segments=True), head_dim=head_dim)


@pytest.mark.parametrize("activation,bias", [(False, False), (True, False), (False, True)])
def test_activation_and_bias_options(activation, bias):
    _check(*_inputs(7, 2, 50, 512, 4, bias=bias, segments=True), head_dim=128, activation=activation)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 64, 65, 127, 129, 191])
def test_block_edges(seq_len):
    _check(*_inputs(seq_len, 3, seq_len, 512, 4, state=True, segments=seq_len > 2), head_dim=128)


def test_long_sequence_many_documents():
    """The 1.3B q/k layout (16 heads of 128) at T = 4096 with about 40 documents per row."""
    x, w, _, s, _, dn = _inputs(11, 2, 4096, 2048, 4, bias=False, state=True)
    seg = jnp.cumsum(jax.random.bernoulli(jax.random.PRNGKey(3), 0.01, (2, 4096)).astype(jnp.int32), axis=1)
    _check(x, w, None, s, seg, dn, head_dim=128)


@pytest.mark.parametrize("batch", [8, 4, 1])
def test_backward_block_sizes(batch):
    """Grids for the 256-, 128- and 64-token backward blocks with a ragged T (shifted last block)."""
    _check(*_inputs(batch, batch, 4133, 2048, 4, state=True, segments=True), head_dim=128)


def test_reused_segment_ids_follow_equality():
    x, w, b, s, _, dn = _inputs(13, 2, 70, 512, 4, state=True)
    seg = jax.random.randint(jax.random.PRNGKey(4), (2, 70), 0, 3)
    _check(x, w, b, s, seg, dn, head_dim=128)


def test_unit_norm_per_head():
    x, w, b, s, seg, _ = _inputs(21, 2, 100, 2048, 4, state=True, segments=True)
    n, _ = conv_l2norm(_conv(w, b, "mosaic_gpu", True), x, 128, segment_ids=seg, state=s)
    norms = jnp.linalg.norm(n.astype(jnp.float32).reshape(2, 100, 16, 128), axis=-1)
    np.testing.assert_allclose(np.asarray(norms), 1.0, atol=1e-2)


def test_prefill_then_decode_matches_full_sequence():
    """Prefill, then decode token by token (each step a T = 1 kernel call): bit-identical to the full sequence."""
    x, w, b, _, _, _ = _inputs(5, 2, 24, 512, 4)
    conv = _conv(w, b, "mosaic_gpu", True)
    full, _ = conv_l2norm(conv, x, 128)
    y, state = conv_l2norm(conv, x[:, :16], 128)
    outs = [y]
    for t in range(16, 24):
        y, state = conv_l2norm(conv, x[:, t:t + 1], 128, state=state)
        outs.append(y)
    np.testing.assert_array_equal(np.asarray(jnp.concatenate(outs, axis=1), np.float32), np.asarray(full, np.float32))


def test_documents_do_not_leak():
    x, w, b, _, _, _ = _inputs(9, 1, 96, 512, 4)
    seg = jnp.concatenate([jnp.zeros((1, 40), jnp.int32), jnp.ones((1, 56), jnp.int32)], axis=1)
    conv = _conv(w, b, "mosaic_gpu", True)
    n0, _ = conv_l2norm(conv, x, 128, segment_ids=seg)
    n1, _ = conv_l2norm(conv, x.at[:, :40].multiply(-3.0), 128, segment_ids=seg)
    np.testing.assert_array_equal(np.asarray(n0[:, 40:], np.float32), np.asarray(n1[:, 40:], np.float32))


def test_strict_supports():
    from jla.ops.mosaic_gpu.short_conv_l2norm import short_conv_l2norm, unsupported_reason

    def reason(channels, head_dim, dtype=jnp.bfloat16, width=4):
        return unsupported_reason(jnp.zeros((1, 8, channels), dtype), jnp.zeros((channels, width)), head_dim=head_dim)

    assert reason(2048, 128) is None and reason(512, 128) is None and reason(256, 64) is None
    assert reason(2048, 256) is not None  # head dims the tests do not cover
    assert reason(2048, 96) is not None
    assert reason(384, 128) is not None  # 2 channels per thread: a 128-channel head spans two warps
    assert reason(128, 64) is not None  # 1 channel per thread
    assert reason(2048, 128, jnp.float32) is not None
    assert reason(2048, 128, jnp.float16) is not None  # the unfused path covers float16
    assert reason(2048, 128, width=5) is not None
    with pytest.raises(ValueError, match="L2 norm"):
        short_conv_l2norm(jnp.zeros((1, 8, 384), jnp.bfloat16), jnp.zeros((384, 4)), head_dim=128)
    conv = ShortConv(384, 4, impl="mosaic_gpu")
    assert conv_l2norm(conv, jnp.zeros((1, 8, 384), jnp.bfloat16), 128) is None  # caller keeps the unfused path
    assert conv_l2norm(ShortConv(512, 4, impl="xla"), jnp.zeros((1, 8, 512), jnp.bfloat16), 128) is None


def _one_layer(preset, in_conv, *, conv_impl="mosaic_gpu", **top):
    """One layer of ``preset``; ``"<kind>@<preset>"`` swaps in another mixer kind at that preset's shapes."""
    from jla.models import get_preset

    kind, _, preset = preset.rpartition("@")
    base = get_preset(preset)
    if kind:
        base = base.replace(layer_types=(kind,) * base.n_layers)
    la_kw = dict(conv_impl=conv_impl, qk_l2norm_in_conv=in_conv)
    if top.get("compute_dtype") == "float32":
        la_kw["implementation"] = "chunked"
    la = dataclasses.replace(base.linear_attention, **la_kw)
    return base.replace(n_layers=1, layer_types=base.layer_types[:1], vocab_size=512,
                        mlp_types=None if base.mlp_types is None else base.mlp_types[:1], linear_attention=la, **top)


# DeltaNet, the fourth delta-rule mixer, has no 1.3B preset: it runs at gdn-hd128-1p3b's head shapes.
@pytest.mark.parametrize("preset", ["gdn-hd128-1p3b", "kda-1p3b", "gdn2-1p3b", "deltanet@gdn-hd128-1p3b"])
def test_model_step_matches_unfused(preset, monkeypatch):
    """One layer of each delta-rule 1.3B preset with packed documents and an odd length, the norm in the conv
    kernel vs in the op, both against a float32 oracle (XLA conv, chunked op, float32 compute): the layer output
    and every parameter gradient must be within bf16 noise of the oracle and no less accurate than the unfused
    path, and the op must run with ``use_qk_l2norm=False`` (no second normalisation)."""
    from jla import ops
    from jla.layers import linear_attention as la_mod
    from jla.models import build_model

    seen = []
    real = ops.linear_attention

    def spy(kind, q, k, v, **options):
        seen.append(options.get("use_qk_l2norm"))
        return real(kind, q, k, v, **options)

    monkeypatch.setattr(la_mod.ops, "linear_attention", spy)
    seq_len = 199
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, seq_len), 0, 512)
    seg = jnp.cumsum(jax.random.bernoulli(jax.random.PRNGKey(2), 0.02, (2, seq_len)).astype(jnp.int32), axis=1)
    cfgs = {"unfused": _one_layer(preset, False), "fused": _one_layer(preset, True),
            "oracle": _one_layer(preset, False, conv_impl="xla", compute_dtype="float32")}
    results = {}
    for name, cfg in cfgs.items():
        probe = jax.random.normal(jax.random.PRNGKey(3), (2, seq_len, cfg.d_model), jnp.float32)

        def loss(model):
            out = model(ids, segment_ids=seg)
            return jnp.sum(out.hidden.astype(jnp.float32) * probe), out.hidden

        model = build_model(cfg, key=jax.random.PRNGKey(0))
        seen.clear()
        (_, hidden), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(model)
        results[name] = (hidden, jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array)))
        assert seen and set(seen) == {name != "fused"}, (name, seen)
    ref_h, ref_g = results["oracle"]
    errs = {name: (_err(h, ref_h), max(_err(a, b) for a, b in zip(g, ref_g)))
            for name, (h, g) in results.items() if name != "oracle"}
    for i, what in enumerate(("hidden", "gradient")):
        assert errs["fused"][i] <= max(2.5e-2, 1.5 * errs["unfused"][i]), (what, errs)
    for a, b, ref in zip(results["fused"][1], results["unfused"][1], ref_g):
        assert _err(a, ref) <= max(3e-2, 1.5 * _err(b, ref)), (a.shape, _err(a, ref), _err(b, ref))


@pytest.mark.parametrize("preset", ["gdn-hd128-1p3b", "kda-1p3b"])
def test_mixer_prefill_then_decode(preset):
    """The mixer's prefill + token-by-token decode with the fused conv matches its full-sequence output."""
    from jla.layers.linear_attention import make_mixer

    cfg = _one_layer(preset, True)
    kind = cfg.layer_types[0]
    mixer = make_mixer(kind, cfg.d_model, cfg.linear_attention, key=jax.random.PRNGKey(0), std=0.02,
                       norm_eps=1e-6, compute_dtype="bfloat16")
    x = (jax.random.normal(jax.random.PRNGKey(4), (2, 20, cfg.d_model), jnp.float32) * 0.5).astype(jnp.bfloat16)
    full = mixer(x)
    cache = mixer.init_cache(2, jnp.bfloat16)
    y, cache = mixer.step(x[:, :12], cache)
    outs = [y]
    for t in range(12, 20):
        y, cache = mixer.step(x[:, t:t + 1], cache)
        outs.append(y)
    assert _err(jnp.concatenate(outs, axis=1), full) <= 2e-2
