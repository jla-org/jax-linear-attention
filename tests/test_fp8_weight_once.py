# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Delayed-scaling FP8 weights quantized once per optimizer step (``ModelConfig.fp8_weight_once``,
:func:`jla.training.fp8.prepare_fp8_weights`) against the per-call path of :func:`jla.layers.fp8.fp8_dot`
and :func:`jla.layers.fp8_swiglu.fp8_swiglu_mlp`: the same values, gradients and next states (bitwise on
CPU), and whole train steps with micro-steps and rematerialization that match the per-call path, also on
4 devices (FSDP, pipeline parallelism; subprocesses, as the CPU device count is fixed when JAX starts)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.layers import fp8 as F
from jla.layers import fp8_swiglu as S
from jla.models import build_model, get_preset
from jla.training.fp8 import attach_fp8_weights, prepare_fp8_weights
from jla.training.step import LossConfig, build_train_step, lm_loss

ON_GPU = jax.default_backend() == "gpu"
ROOT = Path(__file__).resolve().parents[1]


def _same(actual, expected, rel: float = 1e-2, max_fraction: float = 0.05) -> None:
    """Bitwise on CPU. On a GPU at FP8-GEMM resolution: the step's copy gives XLA the same FP8 operands in
    another HLO layout, so it may pick another GEMM algorithm (accumulation order), which moves a few bf16
    outputs by one unit in the last place: every element within ``rel`` of the maximum (one bf16 ulp of the
    largest values is 2^-8..2^-7 of them) and at most ``max_fraction`` of them different at all."""
    bf16 = jnp.dtype(getattr(actual, "dtype", jnp.float32)) == jnp.dtype(jnp.bfloat16)
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if not ON_GPU:
        np.testing.assert_array_equal(a, e)
        return
    np.testing.assert_allclose(a, e, rtol=0, atol=rel * max(float(np.max(np.abs(e))), 1e-30))
    if bf16:  # a float32 result (weight gradients) changes in its last bits almost everywhere instead
        assert np.mean(a != e) <= max_fraction, f"{np.mean(a != e):.4f} of the elements differ"


def _state(history: int, amaxes: tuple[float, float, float], prefix: tuple[int, ...] = ()) -> F.Fp8State:
    state = F.Fp8State.init(history, prefix)
    hist = lambda v: jnp.zeros((*prefix, history)).at[..., :2].set(jnp.asarray([0.5 * v, v]))  # noqa: E731
    return eqx.tree_at(lambda s: (s.input_amax_history, s.kernel_amax_history, s.grad_amax_history), state,
                       tuple(hist(v) for v in amaxes))


def _vjp(fn, args, g):
    y, pullback = jax.vjp(fn, *args)
    return y, pullback(g)


@pytest.mark.parametrize("w_shape", [(64, 96), (64, 3, 32)])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("transposed", [True, False])  # one device / several (sharded weights: no qt)
def test_prepared_dot_matches_the_per_call_dot(w_shape, jit, transposed) -> None:
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 16, 64)), jnp.bfloat16)
    w = jnp.asarray(0.2 * rng.standard_normal(w_shape), jnp.float32)
    g = jnp.asarray(rng.standard_normal((2, 16, *w_shape[1:])), jnp.bfloat16)
    st = _state(8, (4.0, 0.8, 3.0))

    def per_call(x, w, st):
        return F.fp8_dot(x, w, st, recipe="delayed")

    def once(x, w, st):  # the copy is made outside the differentiated function, as in the train step
        return F.fp8_dot_prepared(x, w, wq, st)

    module = (F.Fp8Linear(w, None, st, "bfloat16", ("embed", "mlp"), "delayed") if len(w_shape) == 2 else
              F.Fp8StackedLinear(w, st, "bfloat16", ("embed", None, "mlp"), "delayed"))
    wq = F.prepare_fp8_weight(module, transposed=transposed)
    assert wq.q.dtype == F.E4M3 and (wq.qt.shape == (*w_shape[1:], w_shape[0]) if transposed else wq.qt is None)
    run = jax.jit(_vjp, static_argnums=0) if jit else _vjp
    y0, (dx0, dw0, ds0) = run(per_call, (x, w, st), g)
    y1, (dx1, dw1, ds1) = run(once, (x, w, st), g)
    assert y1.dtype == y0.dtype and dw1.dtype == dw0.dtype
    for a, b in ((y1, y0), (dx1, dx0), (dw1, dw0)):
        _same(a, b)
    for a, b in zip(jax.tree.leaves(ds1), jax.tree.leaves(ds0), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))  # scales and amax histories
    # Independent reference: the exact product of the dequantized FP8 operands in float64 (the step's copy
    # and the per-call path quantize to the same e4m3 values; only the GEMM's rounding is left).
    xs = F.scale_from_amax(jnp.max(st.input_amax_history), 448.0, st.input_scale)
    ws = F.scale_from_amax(jnp.max(st.kernel_amax_history), 448.0, st.kernel_scale)
    xq = np.asarray(F.quantize(x, xs, F.E4M3), np.float64) * float(xs)
    np.testing.assert_array_equal(np.asarray(wq.q, np.float32), np.asarray(F.quantize(w, ws, F.E4M3), np.float32))
    ref = np.tensordot(xq, np.asarray(wq.q, np.float64) * float(ws), axes=([2], [0]))
    np.testing.assert_allclose(np.asarray(y1, np.float64), ref, rtol=0, atol=8e-3 * np.max(np.abs(ref)))


def test_prepared_copies_of_stacked_layers_match_each_layer() -> None:
    """Leading stacked-layer dims (``scan_layers``): each layer's slice of the step's copy is that layer's copy."""
    rng = np.random.default_rng(1)
    w = jnp.asarray(0.3 * rng.standard_normal((3, 48, 2, 32)), jnp.float32)
    st = _state(4, (2.0, 0.5, 1.0), prefix=(3,))
    st = eqx.tree_at(lambda s: s.kernel_amax_history, st, st.kernel_amax_history * jnp.arange(1.0, 4.0)[:, None])
    module = F.Fp8StackedLinear(w, st, "bfloat16", ("embed", None, "mlp"), "delayed")
    stacked = F.prepare_fp8_weight(module, transposed=True)
    assert stacked.q.shape == w.shape and stacked.qt.shape == (3, 2, 32, 48) and stacked.amax.shape == (3,)
    sharded = F.prepare_fp8_weight(module, transposed=False)  # several devices: the same q and amax, no qt
    assert sharded.qt is None
    np.testing.assert_array_equal(np.asarray(sharded.q, np.float32), np.asarray(stacked.q, np.float32))
    np.testing.assert_array_equal(np.asarray(sharded.amax), np.asarray(stacked.amax))
    for layer in range(3):
        st_l = jax.tree.map(lambda a: a[layer], st)  # noqa: B023
        one = F.prepare_fp8_weight(F.Fp8StackedLinear(w[layer], st_l, "bfloat16", ("embed", None, "mlp"), "delayed"),
                                   transposed=True)
        for a, b in ((stacked.q[layer], one.q), (stacked.qt[layer], one.qt), (stacked.amax[layer], one.amax)):
            np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
        scale = F.scale_from_amax(jnp.max(st_l.kernel_amax_history), 448.0, st_l.kernel_scale)
        np.testing.assert_array_equal(np.asarray(one.q, np.float32),
                                      np.asarray(F.quantize(w[layer], scale, F.E4M3), np.float32))
        np.testing.assert_array_equal(np.asarray(one.qt, np.float32),
                                      np.asarray(jnp.transpose(one.q, (1, 2, 0)), np.float32))


@pytest.mark.parametrize("impl", ["xla", "pallas"] if ON_GPU else ["xla"])
@pytest.mark.parametrize("transposed", [True, False])
def test_prepared_fused_mlp_matches_the_per_call_fused_mlp(impl, transposed) -> None:
    rng = np.random.default_rng(2)
    d, f = 64, 128
    x = jnp.asarray(rng.standard_normal((2, 64, d)), jnp.bfloat16)
    w_gu = jnp.asarray(0.2 * rng.standard_normal((d, 2, f)), jnp.float32)
    w_dn = jnp.asarray(0.1 * rng.standard_normal((f, d)), jnp.float32)
    g = jnp.asarray(rng.standard_normal((2, 64, d)), jnp.bfloat16)
    st_gu, st_dn = _state(8, (4.0, 0.8, 4.0)), _state(8, (6.0, 0.4, 4.0))
    down = F.Fp8Linear(w_dn, None, st_dn, "bfloat16", ("mlp", "embed"), "delayed")
    pw_gu, pw_dn = F.prepare_fp8_weight(F.Fp8SwiGLU(w_gu, down, st_gu, "bfloat16", "delayed", True),
                                        transposed=transposed)
    assert (pw_gu.qt is None and pw_dn.qt is None) != transposed

    def per_call(x, w_gu, w_dn, st_gu, st_dn):
        return S.fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, impl=impl)

    def once(x, w_gu, w_dn, st_gu, st_dn):
        return S.fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, impl=impl, prepared=(pw_gu, pw_dn))

    run = jax.jit(_vjp, static_argnums=0)
    y0, grads0 = run(per_call, (x, w_gu, w_dn, st_gu, st_dn), g)
    y1, grads1 = run(once, (x, w_gu, w_dn, st_gu, st_dn), g)
    _same(y1, y0)
    for a, b in zip(grads1[:3], grads0[:3], strict=True):  # dx, dW_gu, dW_dn
        _same(a, b)
    for a, b in zip(jax.tree.leaves(grads1[3:]), jax.tree.leaves(grads0[3:]), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="do not match"):
        S.fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, impl=impl, prepared=(pw_dn, pw_gu))


def test_prepare_only_delayed_weights_with_weight_once() -> None:
    base = get_preset("tiny")
    model = build_model(base.replace(fp8="delayed"), key=jax.random.key(0))
    prepared = prepare_fp8_weights(model)
    n_modules = len(F.fp8_modules(model))  # every projection, and each MLP's gate/up and down
    copies = [w for p in prepared if p is not None for w in (p if isinstance(p, tuple) else (p,))]
    assert len(copies) == n_modules and all(isinstance(w, F.Fp8Weight) for w in copies)
    attached = attach_fp8_weights(model, prepared)
    assert all(m.prepared is not None for m in F.fp8_modules(attached))
    assert all(m.prepared is None for m in F.fp8_modules(model))  # the stored model never carries copies
    assert prepare_fp8_weights(build_model(base.replace(fp8="current"), key=jax.random.key(0))) is None
    off = build_model(base.replace(fp8="delayed", fp8_weight_once=False), key=jax.random.key(0))
    assert prepare_fp8_weights(off) is None
    assert prepare_fp8_weights(build_model(base, key=jax.random.key(0))) is None
    with pytest.raises(ValueError, match="leaves"):
        attach_fp8_weights(eqx.filter(model, eqx.is_array), prepared[:-1])


@pytest.mark.parametrize("scan_layers", [False, True])
@pytest.mark.parametrize("transposed", [True, False])
def test_models_carrying_the_copies_have_logical_axes(scan_layers, transposed) -> None:
    """The pipeline-parallel forward lays out the block parameters of the model it is given (the train step's,
    with the copies attached) by ``logical_axes``: each copy has its weight's axes, ``qt`` with the contracted
    axis last, and ``amax`` only the stacked-layer axis."""
    from jla.models import logical_axes

    cfg = get_preset("tiny-hybrid").replace(fp8="delayed", scan_layers=scan_layers)
    model = build_model(cfg, key=jax.random.key(0))
    attached = attach_fp8_weights(model, prepare_fp8_weights(model, transposed=transposed))
    axes = logical_axes(attached)
    modules, module_axes = F.fp8_modules(attached), F.fp8_modules(axes)
    assert len(modules) == len(module_axes) > 0
    checked = 0
    for module, ax in zip(modules, module_axes, strict=True):
        # attach_fp8_weight puts the gate/up copy on the MLP and the down copy on its down projection, which
        # fp8_modules lists on its own.
        weight, copy = (ax.gate_up, ax.prepared) if isinstance(module, F.Fp8SwiGLU) else (ax.weight, ax.prepared)
        stacked = weight[:1] if weight[:1] == ("layers",) else ()
        k, rest = weight[len(stacked)], weight[len(stacked) + 1:]
        assert copy.q == weight and copy.amax == stacked
        assert copy.qt == ((*stacked, *rest, k) if transposed else None)
        checked += 1
    assert checked == len(modules)
    assert any(ax.weight[:1] == ("layers",) for ax in module_axes if not isinstance(ax, F.Fp8SwiGLU)) == scan_layers


def _batches(steps: int, accum: int, batch: int = 8, seq: int = 32, vocab: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        ids = [rng.integers(0, vocab, size=(accum, batch // accum, 1))]
        for _ in range(seq):
            ids.append((5 * ids[-1] + 3) % vocab)
        ids = np.concatenate(ids, axis=-1).astype(np.int32)
        out.append({"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])})
    return out


def _train(cfg, batches, scan_unroll: int):
    model = build_model(cfg, key=jax.random.key(0))
    optimizer = optax.adamw(3e-3, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype=cfg.compute_dtype)), donate=False,
                            scan_unroll=scan_unroll)
    key, losses = jax.random.key(1), []
    for batch in batches:
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
    return model, np.array(losses)


@pytest.mark.parametrize("preset,fused,remat,scan_layers,accum,unroll", [
    ("tiny", True, "none", False, 2, 1),    # the transformer-1p3b setting: unrolled layers, rolled micro-steps
    ("tiny", True, "full", True, 1, 0),     # the recurrent presets' setting: scanned layers, full remat
    ("tiny", False, "full", False, 2, 0),   # the unfused FP8 SwiGLU, unrolled micro-steps
    ("tiny-hybrid", True, "full", True, 2, 1),  # Gated DeltaNet + gated attention + MTP heads
    ("kimi-linear-tiny", True, "full", True, 2, 0),  # KDA + MLA low-rank projections + MoE (grouped GEMMs not FP8)
    ("tiny-gdn2", True, "full", True, 1, 0),  # GDN-2: the erase/write gate projections share the input's FP8 cast
])
def test_train_steps_match_the_per_call_path(preset, fused, remat, scan_layers, accum, unroll) -> None:
    cfg = get_preset(preset).replace(fp8="delayed", fp8_fused=fused, fp8_amax_history=16, remat=remat,
                                     scan_layers=scan_layers)
    batches = _batches(4, accum)
    once_model, once = _train(cfg, batches, unroll)
    call_model, per_call = _train(cfg.replace(fp8_weight_once=False), batches, unroll)
    # The two paths are different compiled programs. On a GPU they round differently; on XLA CPU it depends on
    # the host. On an AVX-512 host every case is bitwise equal (losses, every parameter and every FP8 history)
    # except tiny-hybrid, whose N(0, 0.02) Gated DeltaNet short convs (the tiny test presets keep FLA's convs,
    # tests/test_init_parity.py) give a step-4 loss 4e-6 apart (5.679586 vs 5.679608, 19 Oct 2026). On
    # GitHub's runners its steps 2 and 3 already differ (3e-6 and 3e-5 apart, 20 Oct 2026). So every backend
    # gets the GPU tolerance, and the checks below are bitwise wherever the run is.
    np.testing.assert_allclose(once, per_call, rtol=2e-3)
    bitwise = not ON_GPU and np.array_equal(once, per_call)
    a_leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(once_model, eqx.is_array))[0]
    b_leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(call_model, eqx.is_array))[0]
    init_leaves = jax.tree_util.tree_leaves(eqx.filter(build_model(cfg, key=jax.random.key(0)), eqx.is_array))
    assert [jax.tree_util.keystr(p) for p, _ in a_leaves] == [jax.tree_util.keystr(p) for p, _ in b_leaves]
    drift = {}  # GPU: mean |difference| of the two paths over the mean update, per weight matrix
    for (path, a), (_, b), init in zip(a_leaves, b_leaves, init_leaves, strict=True):
        a, b, name = np.asarray(a, np.float64), np.asarray(b, np.float64), jax.tree_util.keystr(path)
        if bitwise:
            np.testing.assert_array_equal(a, b, err_msg=name)
        elif "amax_history" not in name and "scale" not in name:
            # A last-bit GEMM difference upstream moves an e5m2 gradient (2 mantissa bits) by a whole e5m2 step
            # (12-25%) wherever it flips the rounding, and AdamW normalizes every update to about lr = 3e-3: the
            # parameters of the two paths drift apart by up to 2 lr per step, never more (losses: above).
            np.testing.assert_allclose(a, b, rtol=0, atol=2 * 3e-3 * len(batches), err_msg=name)
            if a.size >= 1024:  # ... but only where a gradient is near a rounding boundary (few of them)
                drift[name] = float(np.mean(np.abs(a - b)) / max(np.mean(np.abs(b - np.asarray(init))), 1e-12))
        else:
            # FP8 state after 4 steps of those parameters: the tiny weights (std 0.02) move by up to 2 lr per
            # step between the paths, so a later layer's activation and gradient amaxes differ by 10-60%
            # (seen on an H100). The values per op are checked above; here, the same entries are recorded, and
            # the first step's weight amaxes (the initial weights', recorded through the step's copies) are equal.
            assert np.all(np.isfinite(a)) and np.all((a > 0) == (b > 0)), name
            if "kernel_amax_history" in name:
                np.testing.assert_array_equal(a[..., len(batches) - 1], b[..., len(batches) - 1], err_msg=name)
    if not bitwise:
        # On the whole the two paths make the same updates: on an H100 their mean difference is 6-25% of the mean
        # update of a weight matrix (the largest in the linear-attention q/k projections, whose normalized outputs
        # get small gradients); independent updates (a wrong weight copy or scale) would differ by about 100%.
        assert drift and max(drift.values()) <= 0.5, sorted(drift.items(), key=lambda item: -item[1])[:4]
    hist = [np.asarray(m.state.kernel_amax_history) for m in F.fp8_modules(once_model)]
    assert all(np.all(h[..., :4] > 0) and np.all(h[..., 4:] == 0) for h in hist)  # one amax per applied step


def _projection(rng, d_in, d_out, history_scale, amax=3.0):
    w = jnp.asarray(0.2 * rng.standard_normal((d_in, d_out)), jnp.float32)
    st = _state(8, (amax, 0.8 * history_scale, 2.0 * history_scale))
    module = F.Fp8Linear(w, None, st, "bfloat16", ("embed", "heads"), "delayed")
    return F.attach_fp8_weight(module, F.prepare_fp8_weight(module))


@pytest.mark.parametrize("same_history", [True, False])
def test_shared_input_projections_match_separate_calls(same_history) -> None:
    """q/k/v-style projections of one input share one e4m3 copy of it: bitwise the separate calls when their
    input histories agree (as in training), and the largest scale when they do not."""
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((2, 32, 64)), jnp.bfloat16)
    mods = [_projection(rng, 64, n, scale, amax=3.0 if same_history else 3.0 * scale)
            for n, scale in ((64, 1.0), (32, 1.5), (32, 0.5))]
    plain = F.Fp8Linear(jnp.asarray(rng.standard_normal((64, 48)), jnp.float32), None, None, "bfloat16",
                        ("embed", "heads"), "current")  # not shared: called as usual
    gs = [jnp.asarray(rng.standard_normal((2, 32, n)), jnp.bfloat16) for n in (64, 32, 32, 48)]

    def loss(x, ws, sts, together):
        ms = [eqx.tree_at(lambda m: (m.weight, m.state), mod, (w, st)) for mod, w, st in zip(mods, ws, sts)]
        outs = (F.shared_input_projections(x, *ms, plain, None) if together else
                tuple(m(x) for m in ms) + (plain(x), None))
        assert outs[-1] is None
        return sum(jnp.sum(o.astype(jnp.float32) * g.astype(jnp.float32)) for o, g in zip(outs[:-1], gs))

    args = (x, [m.weight for m in mods], [m.state for m in mods])
    run = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)), static_argnums=3)
    (v1, (dx1, dw1, ds1)), (v0, (dx0, dw0, ds0)) = run(*args, True), run(*args, False)
    for st1, st0, mod in zip(ds1, ds0, mods, strict=True):  # each keeps its own input scale and records amax(x)
        for a, b in zip(jax.tree.leaves(st1), jax.tree.leaves(st0), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    if same_history:
        if ON_GPU:
            # XLA may pick another GEMM algorithm for either path (accumulation order, chosen by autotuning), which
            # moves bf16 outputs by about one unit in the last place, 2^-8..2^-7 of their magnitude: each output
            # within two of them (plus float32 cancellation near zero), and the loss, a sum over every output times
            # its cotangent, within one of them per term.
            shared = jax.jit(lambda x: F.shared_input_projections(x, *mods, plain))(x)
            separate = jax.jit(lambda x: tuple(m(x) for m in mods) + (plain(x),))(x)
            for a, e in zip(shared, separate, strict=True):
                a, e = np.asarray(a, np.float64), np.asarray(e, np.float64)
                np.testing.assert_array_less(np.abs(a - e), 2.0**-6 * np.abs(e) + 1e-3 * np.max(np.abs(e)) + 1e-30)
            bound = 2.0**-7 * sum(float(np.sum(np.abs(np.asarray(o, np.float64) * np.asarray(g, np.float64))))
                                  for o, g in zip(separate, gs, strict=True))
            assert abs(float(v1) - float(v0)) <= bound, (float(v1), float(v0), bound)
        else:
            _same(v1, v0)
        _same(dx1, dx0)
        for a, b in zip(dw1, dw0, strict=True):
            _same(a, b)
        return
    # Different histories (not reached in training): the shared scale is the largest delayed scale, here the
    # only one that covers x (amax ~4; the 1.5 history saturates x in a separate call). Each output is then x
    # on that e4m3 grid times the e4m3 weight: within FP8 resolution of the float32 product.
    outs = jax.jit(lambda x: F.shared_input_projections(x, *mods))(x)
    for out, mod in zip(outs, mods, strict=True):
        ws = F.scale_from_amax(jnp.max(mod.state.kernel_amax_history), 448.0, mod.state.kernel_scale)
        ref = np.tensordot(np.asarray(x, np.float64), np.asarray(mod.prepared.q, np.float64) * float(ws), axes=([2], [0]))
        np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=0, atol=0.05 * np.max(np.abs(ref)))


# -- several devices: FSDP (no transposed copies) and pipeline parallelism, through the Trainer ----------------

_MULTI_DEVICE = textwrap.dedent(
    r"""
    import json
    import sys

    import equinox as eqx
    import jax
    import numpy as np

    from _toy_model import experiment
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 4, jax.devices()
    out, cases = sys.argv[1], json.loads(sys.argv[2])
    report = {}
    for name, preset, model_kw, mesh, train_kw in cases:
        runs = {}
        for once in (True, False):
            model = get_preset(preset).replace(fp8="delayed", fp8_amax_history=8, fp8_weight_once=once, **model_kw)
            train = {"max_steps": 2, "log_every": 1, "final_eval": False, **train_kw}
            config = experiment(f"{out}/{name}-{once}", model=model, train=train, optim={"lr": 3e-3, "warmup_steps": 0},
                                mesh=mesh)
            runs[once] = Trainer(config, devices=jax.devices()).run()
        a = jax.tree_util.tree_flatten_with_path(eqx.filter(runs[True].model, eqx.is_array))[0]
        b = jax.tree_util.tree_flatten_with_path(eqx.filter(runs[False].model, eqx.is_array))[0]
        assert [jax.tree_util.keystr(p) for p, _ in a] == [jax.tree_util.keystr(p) for p, _ in b], name
        differ = [jax.tree_util.keystr(p) for (p, x), (_, y) in zip(a, b)
                  if not np.array_equal(np.asarray(x), np.asarray(y))]
        assert not differ, (name, len(differ), differ[:4])
        losses = [[h["loss"] for h in runs[once].history] for once in (True, False)]
        assert losses[0] == losses[1] and len(losses[0]) == 2 and np.all(np.isfinite(losses[0])), (name, losses)
        report[name] = {"leaves": len(a), "loss": losses[0]}
    print("FP8ONCE " + json.dumps(report))
    """
)


def _run_on_4_cpu_devices(tmp_path: Path, cases: list) -> dict:
    script = tmp_path / "fp8_weight_once_multidevice.py"
    script.write_text(_MULTI_DEVICE)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "runs"), json.dumps(cases)], env=env,
                            capture_output=True, text=True, timeout=1500, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    line = next(x for x in result.stdout.splitlines() if x.startswith("FP8ONCE "))
    return json.loads(line[len("FP8ONCE "):])


def test_weight_copies_on_4_fsdp_devices_match_the_per_call_path(tmp_path: Path) -> None:
    """FSDP over 4 devices (copies without ``qt``): parameters, FP8 states and losses bitwise the per-call path's."""
    report = _run_on_4_cpu_devices(tmp_path, [
        ["tiny", "tiny", {"fp8_fused": True, "scan_layers": False, "remat": "none"}, {"fsdp": 4},
         {"micro_batch_per_replica": 2, "grad_accum": 2}],
        ["hybrid", "tiny-hybrid", {"fp8_fused": True, "scan_layers": True, "remat": "full"}, {"fsdp": 4},
         {"micro_batch_per_replica": 2, "grad_accum": 2}],
    ])
    assert set(report) == {"tiny", "hybrid"}


def test_weight_copies_under_pipeline_parallelism(tmp_path: Path) -> None:
    """``mesh.pp = 2`` (the pipeline forward lays out the copies by their logical axes; fully manual with FSDP,
    manual over the stages with data parallelism): bitwise the per-call path."""
    report = _run_on_4_cpu_devices(tmp_path, [
        ["pp2fsdp2", "tiny", {"scan_layers": True}, {"pp": 2, "fsdp": 2},
         {"micro_batch_per_replica": 4, "grad_accum": 1}],
        ["pp2dp2", "tiny", {"scan_layers": True}, {"pp": 2, "dp": 2, "fsdp": 1},
         {"micro_batch_per_replica": 4, "grad_accum": 1}],
    ])
    assert set(report) == {"pp2fsdp2", "pp2dp2"}
