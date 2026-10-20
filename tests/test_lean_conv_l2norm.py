# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``lean_vjp`` (and ``lean_remat_core``) together with the fused short conv + per-head L2 norm.

Both are on in the ``gdn-hd128-1p3b`` and ``kda-1p3b`` end-to-end configs. The fused kernels
(``qk_l2norm_in_conv``, :mod:`jla.layers.conv_l2norm`) are a custom VJP that already saves only their
inputs, so under ``lean_vjp`` they must keep producing q and k: lean's own XLA conv VJP may run only for
v, and the op (recomputed in the backward pass or not) must get ``use_qk_l2norm=False``. Off Hopper the
kernels cannot run, so the CPU tests stand an XLA conv + L2 norm in for them (the same values) to check
the routing and the values and gradients against the plain, unfused model. The GPU test checks that the
compiled training step of a lean model contains the fused kernels, forward and backward.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.layers import conv_l2norm as conv_l2norm_mod
from jla.layers import lean
from jla.layers.conv import ShortConv
from jla.models import build_model, get_preset
from jla.models.config import LinearAttentionConfig, hybrid_pattern
from jla.ops.mosaic_gpu._common import is_hopper
from jla.ops.reference import l2norm


def _gdn():
    return get_preset("tiny-hybrid").replace(mtp=None)


def _kda():
    la = LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16, chunk_size=16)
    return get_preset("tiny-hybrid").replace(linear_attention=la, layer_types=hybrid_pattern(4, "kda", 2), mtp=None)


def _gdn2():
    return get_preset("tiny-gdn2")


_MODELS = {"gated_deltanet": _gdn, "kda": _kda, "gdn2": _gdn2}


def _with(cfg, **la_changes):
    return cfg.replace(linear_attention=dataclasses.replace(cfg.linear_attention, **la_changes))


def _stand_in(monkeypatch, calls: list):
    """Make :func:`jla.layers.conv_l2norm.conv_l2norm` run (an XLA conv, then the L2 norm per head in
    float32: the kernels' values) wherever the mixer asks for it, recording the conv module types."""

    def reason(conv, x, head_dim, *, segment_ids=None, state=None):
        return None

    def fused(conv, x, head_dim, *, segment_ids=None, state=None):
        calls.append(type(conv))
        y, new_state = ShortConv.__call__(conv, x, segment_ids=segment_ids, state=state)  # never the lean override
        heads = y.reshape(*y.shape[:-1], y.shape[-1] // head_dim, head_dim)
        return l2norm(heads.astype(jnp.float32)).reshape(y.shape).astype(x.dtype), new_state

    monkeypatch.setattr(conv_l2norm_mod, "conv_l2norm_reason", reason)
    monkeypatch.setattr(conv_l2norm_mod, "conv_l2norm", fused)


def _spy_op(monkeypatch, flags: list):
    real = ops.linear_attention

    def spy(kind, q, k, v, **options):
        flags.append(options.get("use_qk_l2norm"))
        return real(kind, q, k, v, **options)

    monkeypatch.setattr(ops, "linear_attention", spy)


def _spy_lean_conv(monkeypatch, calls: list):
    real = lean.short_conv

    def spy(*args, **kwargs):
        calls.append(args[0].shape[-1])
        return real(*args, **kwargs)

    monkeypatch.setattr(lean, "short_conv", spy)


def _loss_and_grads(model, ids, seg, r):
    def loss(m):
        return jnp.sum(m(ids, segment_ids=seg).hidden.astype(jnp.float32) * r)

    return eqx.filter_value_and_grad(loss)(model)


def _fresh_step():
    """A newly jitted :func:`_loss_and_grads`: equinox shares one compilation cache per function, and a cache
    hit would skip the trace the spies record."""
    return eqx.filter_jit(lambda *args: _loss_and_grads(*args))


@pytest.mark.skipif(is_hopper(), reason="the XLA stand-in replaces kernels that run on Hopper")
@pytest.mark.parametrize("core", [False, True])
@pytest.mark.parametrize("name", sorted(_MODELS))
def test_lean_keeps_fused_conv_l2norm(name, core, monkeypatch):
    """Lean + fused conv/L2 norm: q and k come only from the fused path, lean's conv VJP runs for v alone,
    the op never normalizes again, and loss and gradients equal the plain unfused model's."""
    cfg = _MODELS[name]().replace(compute_dtype="float32", remat="none")
    plain_cfg = _with(cfg, qk_l2norm_in_conv=False)
    lean_cfg = _with(cfg, qk_l2norm_in_conv=True).replace(lean_vjp=True, lean_remat_core=core)
    key = jax.random.key(0)
    plain, model = build_model(plain_cfg, key=key), build_model(lean_cfg, key=key)
    mixers = [blk.mixer for blk in model.blocks] if isinstance(model.blocks, (list, tuple)) else None
    if mixers is not None:
        assert all(getattr(m, "remat_core", core) is core for m in mixers if hasattr(m, "q_conv"))
    batch, length = 2, 40
    ids = jax.random.randint(jax.random.key(1), (batch, length), 0, cfg.vocab_size)
    seg = jnp.asarray(np.repeat(np.array([[0, 1], [3, 4]]), [13, 27], axis=1), jnp.int32)
    r = jax.random.normal(jax.random.key(2), (batch, length, cfg.d_model), jnp.float32)

    plain_flags: list = []
    _spy_op(monkeypatch, plain_flags)
    with jax.default_matmul_precision("highest"):
        l0, g0 = _fresh_step()(plain, ids, seg, r)
    assert plain_flags and set(plain_flags) == {True}, plain_flags

    fused_calls: list = []
    lean_flags: list = []
    lean_convs: list = []
    _stand_in(monkeypatch, fused_calls)
    _spy_op(monkeypatch, lean_flags)
    _spy_lean_conv(monkeypatch, lean_convs)
    with jax.default_matmul_precision("highest"):
        l1, g1 = _fresh_step()(model, ids, seg, r)
    la = cfg.linear_attention
    key_dim, value_dim = la.num_k_heads * la.head_k_dim, la.num_v_heads * la.head_v_dim
    # The fused path ran for q and k of every traced delta-rule mixer, on the lean conv modules' parameters...
    assert fused_calls and set(fused_calls) == {lean.LeanShortConv}, fused_calls
    # ...lean's XLA conv VJP only for v (one per mixer; a q/k-width call would be a second conv of q or k)...
    assert lean_convs == [value_dim] * (len(fused_calls) // 2), (lean_convs, key_dim, value_dim)
    # ...and the op (recomputed or not) never normalized q and k again.
    assert lean_flags and set(lean_flags) == {False}, lean_flags

    np.testing.assert_allclose(float(l0), float(l1), rtol=1e-5)
    flat0 = jax.tree_util.tree_leaves_with_path(eqx.filter(g0, eqx.is_array))
    flat1 = jax.tree_util.tree_leaves_with_path(eqx.filter(g1, eqx.is_array))
    assert [jax.tree_util.keystr(p) for p, _ in flat0] == [jax.tree_util.keystr(p) for p, _ in flat1]
    for (path, a), (_, b) in zip(flat0, flat1):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        err = float(np.max(np.abs(a - b))) / max(1e-6, float(np.max(np.abs(a))))
        assert err < 1e-4, f"{jax.tree_util.keystr(path)}: relative error {err:.2e}"


@pytest.mark.skipif(is_hopper(), reason="checks the CPU fallback")
@pytest.mark.parametrize("core", [False, True])
def test_lean_without_kernels_normalizes_in_the_op(core, monkeypatch):
    """Where the fused kernels decline (here: no Hopper GPU), lean runs its conv VJP on q, k and v and the op
    normalizes q and k, as without lean."""
    cfg = _gdn().replace(compute_dtype="float32", remat="none", lean_vjp=True, lean_remat_core=core)
    flags: list = []
    convs: list = []
    _spy_op(monkeypatch, flags)
    _spy_lean_conv(monkeypatch, convs)
    model = build_model(cfg, key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (2, 32), 0, cfg.vocab_size)
    _fresh_step()(model, ids, None, jnp.ones((2, 32, cfg.d_model), jnp.float32))
    la = cfg.linear_attention
    key_dim, value_dim = la.num_k_heads * la.head_k_dim, la.num_v_heads * la.head_v_dim
    assert flags and set(flags) == {True}, flags
    assert convs and len(convs) % 3 == 0 and sorted(set(convs)) == sorted({key_dim, value_dim}), convs


def _gpu_cfg(kind: str):
    # 4 x 128 = 512 channels: at head_dim 128 the fused kernels take multiples of 512 channels (1.3B: 2048).
    la = LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=128, head_v_dim=128, chunk_size=64,
                               implementation="mosaic_gpu", conv_impl="mosaic_gpu")
    return get_preset("tiny-hybrid").replace(
        d_model=256, linear_attention=la, layer_types=hybrid_pattern(4, kind, 2), mtp=None, remat="none",
        compute_dtype="bfloat16")


@pytest.mark.skipif(not is_hopper(), reason="needs the Mosaic GPU kernels (one Hopper GPU)")
@pytest.mark.parametrize("core", [False, True])
@pytest.mark.parametrize("kind", ["gated_deltanet", "kda"])
def test_compiled_lean_step_runs_fused_kernels(kind, core):
    """The compiled loss + gradient of a lean model (the 1.3B end-to-end settings: no remat, lean VJPs, the
    op rerun when ``core``) contains the fused conv + L2-norm kernels forward and backward, the plain conv
    kernels for v, and matches the plain model's gradients within bfloat16 rounding."""
    cfg = _gpu_cfg(kind)
    batch, length = 2, 512
    ids = jax.random.randint(jax.random.key(1), (batch, length), 0, cfg.vocab_size)
    seg = jnp.broadcast_to((jnp.arange(length, dtype=jnp.int32) >= 200).astype(jnp.int32), (batch, length))
    r = jax.random.normal(jax.random.key(2), (batch, length, cfg.d_model), jnp.float32)
    la = cfg.linear_attention
    probe = jnp.zeros((batch, length, la.num_k_heads * la.head_k_dim), jnp.bfloat16)
    assert conv_l2norm_mod.conv_l2norm_reason(ShortConv(probe.shape[-1], la.conv_size, impl="auto"), probe,
                                              la.head_k_dim, segment_ids=seg) is None
    results = []
    for use_lean in (False, True):
        model = build_model(cfg.replace(lean_vjp=use_lean, lean_remat_core=use_lean and core), key=jax.random.key(0))
        step = _fresh_step()
        if use_lean:
            params, static = eqx.partition(model, eqx.is_array)

            def loss(params, ids, seg, r):
                return _loss_and_grads(eqx.combine(params, static), ids, seg, r)

            text = jax.jit(loss).lower(params, ids, seg, r).compile().as_text()
            for name in ("short_conv_l2norm_fwd", "short_conv_l2norm_bwd", "short_conv_fwd", "short_conv_bwd"):
                assert name in text, f"{name} missing from the compiled lean step"
        results.append(step(model, ids, seg, r))
    (l0, g0), (l1, g1) = results
    np.testing.assert_allclose(float(l0), float(l1), rtol=2e-2)
    for (path, a), (_, b) in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(g0, eqx.is_array)),
                                 jax.tree_util.tree_leaves_with_path(eqx.filter(g1, eqx.is_array))):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        err = float(np.max(np.abs(a - b))) / max(1e-6, float(np.max(np.abs(a))))
        assert err < 5e-2, f"{jax.tree_util.keystr(path)}: relative error {err:.2e}"
