# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 q/k/v gradients from the attention backward (``AttentionConfig.fp8_grad_epilogue``).

* CPU (exact): :func:`jla.layers.fp8_attention.fp8_qkv_attention` with its XLA reference backend gives
  bitwise the plain layer's outputs, input and weight gradients and next delayed-scaling states (one custom
  VJP over the FP8 projections, RoPE and attention instead of three FP8 matmuls, ``apply_rope`` and the
  attention's own VJP), also under ``jax.checkpoint``; the RoPE tables are those of ``apply_rope``;
  ineligible calls run the plain path (silently without FP8 projections).
* CPU, 4 devices: the per-shard rule of the kernel calls, with XLA stand-ins of their signatures.
* GPU: the Mosaic GPU kernels' FP8 epilogues (both backwards) against XLA's RoPE backward, e5m2 cast and amax
  applied to the ranged kernels' ``dq, dk, dv`` (the same main loops); the transposed copies against the
  token-major ones; the fused path's zeroed ``dq`` buffer; the longest-first CTA order against the old grids
  (bitwise); and the whole layer with the kernels against the plain layer on the Mosaic GPU attention.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import fp8 as F
from jla.layers.attention import Attention
from jla.layers.fp8_attention import fp8_qkv_attention, fused_fp8_attention, fused_fp8_attention_reason, rope_tables
from jla.layers.rope import apply_rope
from jla.models.config import AttentionConfig

ON_GPU = jax.default_backend() == "gpu"
ROOT = Path(__file__).resolve().parents[1]


def _hopper() -> bool:
    if not ON_GPU:
        return False
    return str(getattr(jax.devices()[0], "compute_capability", "")).split(".")[0] == "9"


gpu_only = pytest.mark.skipif(not _hopper(), reason="needs a Hopper GPU (Mosaic GPU attention kernels)")


def _positions_and_segments(batch: int, length: int, documents: bool, seed: int = 0):
    """Positions that restart at every document (as packed training data), and the documents' ids."""
    if not documents:
        pos = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
        return pos, None
    rng = np.random.default_rng(seed)
    seg = np.zeros((batch, length), np.int32)
    pos = np.zeros((batch, length), np.int32)
    for b in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=3, replace=False))
        starts = np.concatenate([[0], cuts])
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else length
            seg[b, s:e] = 10 * i + 7
            pos[b, s:e] = np.arange(e - s)
    return jnp.asarray(pos), jnp.asarray(seg)


def _state(history: int, amaxes: tuple[float, float, float]) -> F.Fp8State:
    state = F.Fp8State.init(history)
    hist = lambda v: jnp.zeros((history,)).at[:2].set(jnp.asarray([0.5 * v, v]))  # noqa: E731
    return eqx.tree_at(lambda s: (s.input_amax_history, s.kernel_amax_history, s.grad_amax_history), state,
                       tuple(hist(v) for v in amaxes))


def _layer(*, heads=4, kv_heads=2, head_dim=64, d_model=128, window=None, implementation="xla",
           epilogue=False, bias=False, qk_norm=False, backward="ranged", grad_amax=(0.05, 0.03, 0.08)):
    """An FP8 (delayed) attention layer with the train step's weight copies attached and non-trivial scales."""
    cfg = AttentionConfig(num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, qk_norm=qk_norm,
                          sliding_window=window, implementation=implementation, fp8_grad_epilogue=epilogue,
                          qkv_bias=bias, mosaic_gpu_backward=backward)
    attn = Attention(d_model, cfg, key=jax.random.key(1), std=0.2, norm_eps=1e-6)
    attn = F.apply_fp8(attn, "delayed", amax_history=8)

    def prepared(module, amaxes):
        module = eqx.tree_at(lambda m: m.state, module, _state(8, amaxes))
        return F.attach_fp8_weight(module, F.prepare_fp8_weight(module, transposed=True))

    w_amax = lambda m: float(jnp.max(jnp.abs(m.weight)))  # noqa: E731
    return eqx.tree_at(
        lambda a: (a.q_proj, a.k_proj, a.v_proj, a.o_proj), attn,
        (prepared(attn.q_proj, (4.0, w_amax(attn.q_proj), grad_amax[0])),
         prepared(attn.k_proj, (4.0, w_amax(attn.k_proj), grad_amax[1])),
         prepared(attn.v_proj, (4.0, w_amax(attn.v_proj), grad_amax[2])),
         prepared(attn.o_proj, (1.0, w_amax(attn.o_proj), 0.5))))


def _loss_and_grads(attn, x, positions, segment_ids, g, *, fused: str | None, remat: bool = False):
    """``sum(out * g)`` and its gradients w.r.t. ``x`` and every inexact array of the layer (the FP8 states'
    "gradients" are their next values). ``fused``: None (the layer as configured) or the fused VJP's impl;
    ``remat``: the layer under ``jax.checkpoint`` (nothing saved, the forward recomputed in the backward)."""

    def loss(x, arrays):
        layer = eqx.combine(arrays, static)
        if fused is None:
            out = layer(x, positions, segment_ids)
        else:
            cfg = layer.cfg
            o = fp8_qkv_attention(x, layer.q_proj, layer.k_proj, layer.v_proj, positions, num_heads=cfg.num_heads,
                                  num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, rope_theta=cfg.rope_theta,
                                  softmax_scale=cfg.softmax_scale, sliding_window=cfg.sliding_window,
                                  segment_ids=segment_ids, impl=fused)
            out = layer._output(x, o)
        return jnp.sum(out.astype(jnp.float32) * g), out

    arrays, static = eqx.partition(attn, eqx.is_inexact_array)
    fn = jax.checkpoint(loss, policy=jax.checkpoint_policies.nothing_saveable) if remat else loss
    (value, out), grads = jax.jit(jax.value_and_grad(fn, argnums=(0, 1), has_aux=True))(x, arrays)
    return value, out, grads


def _inputs(batch, length, d_model, documents, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((batch, length, d_model)), jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((batch, length, d_model)), jnp.float32)
    positions, segment_ids = _positions_and_segments(batch, length, documents, seed)
    return x, g, positions, segment_ids


def _leaves_equal(a, b, *, exact: bool, label: str) -> None:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), label
    for i, (u, v) in enumerate(zip(la, lb, strict=True)):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        if exact:
            np.testing.assert_array_equal(u, v, err_msg=f"{label} leaf {i}")
        else:
            np.testing.assert_allclose(u, v, rtol=0, atol=2e-2 * max(float(np.max(np.abs(v))), 1e-30),
                                       err_msg=f"{label} leaf {i}")


# -- CPU: the fused VJP's XLA reference is the plain layer, bitwise ----------------------------------------------


def test_rope_tables_match_apply_rope() -> None:
    """``apply_rope`` rotates by exactly the tables' angles (half-rotation layout, restarting positions)."""
    rng = np.random.default_rng(0)
    for head_dim in (64, 128):
        x = jnp.asarray(rng.standard_normal((2, 256, 3, head_dim)), jnp.bfloat16)
        positions, _ = _positions_and_segments(2, 256, True, seed=4)
        cos, sin = rope_tables(positions, theta=10_000.0, head_dim=head_dim)
        half = head_dim // 2
        x32 = x.astype(jnp.float32)
        c, s = cos[:, :, None], sin[:, :, None]
        manual = jnp.concatenate([x32[..., :half] * c - x32[..., half:] * s,
                                  x32[..., half:] * c + x32[..., :half] * s], axis=-1).astype(x.dtype)
        expected = apply_rope(x, positions, theta=10_000.0)
        if ON_GPU:  # XLA may contract the products into FMAs in either fusion
            np.testing.assert_allclose(np.asarray(manual, np.float32), np.asarray(expected, np.float32),
                                       rtol=1e-2, atol=1e-2)
        else:
            np.testing.assert_array_equal(np.asarray(manual, np.float32), np.asarray(expected, np.float32))


@pytest.mark.parametrize("documents,window,heads,kv_heads,head_dim", [
    (False, None, 4, 2, 64),
    (True, None, 4, 2, 64),
    (True, 48, 4, 1, 64),
    (False, None, 2, 2, 128),
])
def test_reference_vjp_matches_plain_layer(documents, window, heads, kv_heads, head_dim) -> None:
    """The fused VJP (XLA attention backend) against the plain FP8 layer: bitwise on CPU."""
    attn = _layer(heads=heads, kv_heads=kv_heads, head_dim=head_dim, window=window)
    x, g, positions, segment_ids = _inputs(2, 128, 128, documents, seed=2)
    v0, o0, (dx0, dl0) = _loss_and_grads(attn, x, positions, segment_ids, g, fused=None)
    v1, o1, (dx1, dl1) = _loss_and_grads(attn, x, positions, segment_ids, g, fused="xla")
    exact = not ON_GPU
    _leaves_equal(o1, o0, exact=exact, label="output")
    _leaves_equal(v1, v0, exact=exact, label="loss")
    _leaves_equal(dx1, dx0, exact=exact, label="dx")
    _leaves_equal(dl1, dl0, exact=exact, label="layer gradients and next states")
    # Every q/k/v projection recorded a gradient amax, the largest |cotangent| of its output.
    for proj in ("q_proj", "k_proj", "v_proj"):
        new = getattr(dl1, proj).state
        assert float(new.grad_amax_history[0]) > 0, proj
        np.testing.assert_array_equal(np.asarray(new.grad_amax_history[1:]),
                                      np.asarray(getattr(attn, proj).state.grad_amax_history[:-1]))


def test_reference_vjp_under_remat_matches_plain_layer() -> None:
    """Under ``jax.checkpoint`` (the forward recomputed in the backward) the fused VJP is still the plain layer."""
    attn = _layer()
    x, g, positions, segment_ids = _inputs(2, 128, 128, True, seed=6)
    v0, o0, grads0 = _loss_and_grads(attn, x, positions, segment_ids, g, fused=None, remat=True)
    v1, o1, grads1 = _loss_and_grads(attn, x, positions, segment_ids, g, fused="xla", remat=True)
    _leaves_equal((v1, o1, grads1), (v0, o0, grads0), exact=not ON_GPU, label="remat")
    if not ON_GPU:  # on a GPU the two compilations of the XLA attention round differently (then FP8 amplifies it)
        _, _, grads_saved = _loss_and_grads(attn, x, positions, segment_ids, g, fused="xla")
        _leaves_equal(grads1, grads_saved, exact=True, label="remat vs saved residuals")


def test_ineligible_calls_run_the_plain_path(caplog) -> None:
    """``fp8_grad_epilogue`` on a call it does not cover: a reason, and the plain layer's values."""
    x, g, positions, segment_ids = _inputs(2, 128, 128, True)
    cases = {
        "bias": _layer(epilogue=True, bias=True),
        "qk_norm": _layer(epilogue=True, qk_norm=True),
        "fused backward, D = 128": _layer(epilogue=True, backward="fused", heads=2, kv_heads=2, head_dim=128),
    }
    for name, attn in cases.items():
        assert fused_fp8_attention_reason(attn, x, segment_ids) is not None, name
    plain = _layer(epilogue=False)
    flagged = _layer(epilogue=True)
    if not _hopper():  # XLA attention here: not the Mosaic GPU kernel, so the plain path runs
        assert "attention" in fused_fp8_attention_reason(flagged, x, segment_ids)
        v0, o0, grads0 = _loss_and_grads(plain, x, positions, segment_ids, g, fused=None)
        v1, o1, grads1 = _loss_and_grads(flagged, x, positions, segment_ids, g, fused=None)
        _leaves_equal((v1, o1, grads1), (v0, o0, grads0), exact=not ON_GPU, label="flagged vs plain")
    unconverted = Attention(128, AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, qk_norm=False,
                                                 fp8_grad_epilogue=True), key=jax.random.key(0), std=0.02,
                            norm_eps=1e-6)
    assert "FP8" in fused_fp8_attention_reason(unconverted, x, segment_ids)
    # Without FP8 projections (FP8 off, evaluation) the option is silent: the plain path, nothing logged.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jla.layers.fp8_attention"):
        assert fused_fp8_attention(unconverted, x, positions, segment_ids) is None
    assert not caplog.records, caplog.records


# -- GPU: the kernels' epilogues ----------------------------------------------------------------------------------


def _kernel_case(batch, length, heads, kv_heads, head_dim, mask, seed=0):
    from jla.ops.mosaic_gpu.attention import _Config, _contiguous_ids, _forward, _inner_runs, _segment_runs, _sink_init

    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, length, heads, head_dim)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((batch, length, kv_heads, head_dim)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((batch, length, kv_heads, head_dim)), jnp.bfloat16)
    do = jnp.asarray(rng.standard_normal((batch, length, heads, head_dim)), jnp.bfloat16)
    positions, segment_ids = _positions_and_segments(batch, length, mask == "documents", seed)
    window = 200 if mask == "window" else None
    cfg = _Config(scale=head_dim ** -0.5, window=window, has_segments=segment_ids is not None)
    if segment_ids is None:
        seg = jnp.zeros((batch, length), jnp.int32)
        runs = jnp.zeros((batch, 4, length), jnp.int32)
    else:
        seg = segment_ids
        contiguous = _contiguous_ids(seg)
        runs = _segment_runs(seg, contiguous)
        runs = jnp.concatenate([runs, _inner_runs(runs, contiguous, True)], axis=1)
    out, lse = jax.jit(lambda q, k, v: _forward(q, k, v, seg, runs, _sink_init(jnp.zeros((heads,)), cfg.block_q),
                                                cfg))(q, k, v)
    return cfg, q, k, v, do, seg, runs, positions, out, lse


def _values(x) -> np.ndarray:
    return np.asarray(jnp.asarray(x).astype(jnp.float32), np.float64)


@gpu_only
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,mask,backward", [
    (2, 512, 8, 2, 64, "causal", "ranged"),
    (2, 512, 8, 2, 64, "documents", "ranged"),
    (1, 640, 4, 4, 64, "window", "ranged"),
    (1, 384, 4, 2, 128, "documents", "ranged"),
    (1, 256, 2, 1, 128, "causal", "ranged"),
    (2, 512, 8, 2, 64, "causal", "fused"),
    (2, 512, 8, 2, 64, "documents", "fused"),
    (1, 640, 4, 1, 64, "window", "fused"),
])
def test_kernel_epilogues_match_xla(batch, length, heads, kv_heads, head_dim, mask, backward) -> None:
    """The FP8 epilogues against XLA's RoPE backward, e5m2 cast and amax of the ranged kernels' gradients.

    ``backward="fused"`` adds ``dq`` up in float32 in an order that depends on CTA timing (as the plain fused
    kernel), so its ``dq`` cotangent can differ from the ranged kernel's in the last bf16 place more often."""
    import dataclasses

    from jla.ops.mosaic_gpu.attention import _backward_ranged
    from jla.ops.mosaic_gpu.attention_fp8_bwd import fp8_attention_backward

    cfg, q, k, v, do, seg, runs, positions, out, lse = _kernel_case(batch, length, heads, kv_heads, head_dim, mask)
    fp8_cfg = dataclasses.replace(cfg, backward=backward)
    cos, sin = rope_tables(positions, theta=10_000.0, head_dim=head_dim)
    dq, dk, dv = jax.jit(lambda *a: _backward_ranged(cfg, *a))(q, k, v, seg, runs, out, lse, do)

    def rope_vjp(x, ct):
        return jax.vjp(lambda t: apply_rope(t, positions, theta=10_000.0), x)[1](ct)[0]

    expected_g = jax.jit(lambda: (rope_vjp(q, dq), rope_vjp(k, dk), dv))()
    # The recipe's scale for these gradients' own amaxes: the e5m2 range is used, nothing saturates.
    gscale = jnp.stack([F.scale_from_amax(F._amax(g), F._fp8_max(F.E5M2), 1.0) for g in expected_g])
    args = (q, k, v, seg, runs, out, lse, do, cos, sin, gscale)
    got = jax.jit(lambda *a: fp8_attention_backward(*a, fp8_cfg))(*args)  # the layer's call: no transposed copy
    q_differ = 2e-3 if backward == "ranged" else 2e-2
    if head_dim == 64:  # the optional transposed copies (D = 64 only: shared memory)
        with_t = jax.jit(lambda *a: fp8_attention_backward(*a, fp8_cfg, transposed=True))(*args)
        for name, (g8, gt, am), (g8_t, gt_t, am_t), g in zip("qkv", got, with_t, expected_g, strict=True):
            assert gt is None and gt_t.dtype == jnp.float8_e5m2, name
            assert gt_t.shape == (g.shape[2], g.shape[3], g.shape[0], g.shape[1]), (name, gt_t.shape)
            # The transposed copy holds exactly the token-major values.
            np.testing.assert_array_equal(_values(gt_t), _values(jnp.transpose(g8_t, (2, 3, 0, 1))), err_msg=name)
            if backward == "ranged":  # deterministic: both calls write the same values
                np.testing.assert_array_equal(_values(g8_t), _values(g8), err_msg=name)
                np.testing.assert_array_equal(np.asarray(am_t), np.asarray(am), err_msg=name)
    else:
        with pytest.raises(ValueError, match="transposed"):
            fp8_attention_backward(*args, fp8_cfg, transposed=True)
    if backward == "fused":  # the merged transposed operand of one weight-gradient GEMM (fused only, D = 64)
        grads_m, merged = jax.jit(lambda *a: fp8_attention_backward(*a, fp8_cfg, merged=True))(*args)
        assert merged.dtype == jnp.float8_e5m2 and merged.shape == (heads + 2 * kv_heads, head_dim, batch, length)
        starts = (0, heads, heads + kv_heads)
        for name, (g8_m, gt_m, am_m), start in zip("qkv", grads_m, starts, strict=True):
            assert gt_m is None, name
            rows = merged[start:start + g8_m.shape[2]]
            # The merged operand holds exactly the token-major values, the query heads first.
            np.testing.assert_array_equal(_values(rows), _values(jnp.transpose(g8_m, (2, 3, 0, 1))), err_msg=name)
            if name != "q":  # dk, dv are deterministic: the same values as the call without the operand
                np.testing.assert_array_equal(_values(g8_m), _values(got["qkv".index(name)][0]), err_msg=name)
                np.testing.assert_array_equal(np.asarray(am_m), np.asarray(got["qkv".index(name)][2]), err_msg=name)
    else:
        with pytest.raises(ValueError, match="merged"):
            fp8_attention_backward(*args, fp8_cfg, merged=True)
    for name, (g8, _, am), g, gs in zip("qkv", got, expected_g, gscale, strict=True):
        assert g8.dtype == jnp.float8_e5m2 and g8.shape == g.shape, name
        e8 = F.quantize(g, gs, F.E5M2)
        a, e = _values(g8), _values(e8)
        differ = np.mean(a != e)
        # v has no RoPE products: bitwise. q and k: XLA and the kernel may contract the RoPE products into
        # fused multiply-adds differently, which moves the bf16 cotangent (and then rarely the e5m2 value) by
        # one unit in the last place.
        if name == "v":
            assert differ == 0, (name, differ)
        else:
            assert differ <= (q_differ if name == "q" else 2e-3), (name, differ)
            # One e5m2 step (25%), or, where the two RoPE terms nearly cancel, a few float32 units of the larger
            # term: at most about 2^-22 of the 57344 the amax maps to (0.014).
            np.testing.assert_allclose(a, e, rtol=0.26, atol=2.0**-5, err_msg=name)
        amax_kernel, amax_xla = float(jnp.max(am)), float(F._amax(g))
        assert am.shape == (g.shape[0], g.shape[2], g.shape[1])
        if name == "v":
            assert amax_kernel == amax_xla, (name, amax_kernel, amax_xla)
        else:
            assert abs(amax_kernel - amax_xla) <= 2.0**-7 * amax_xla, (name, amax_kernel, amax_xla)
        # The row maxima are the per-row maxima of the cotangent the kernel quantized.
        rows = np.max(np.abs(_values(g)), axis=3).transpose(0, 2, 1)
        assert np.mean(np.abs(np.asarray(am, np.float64) - rows) > 2.0**-7 * rows) <= 2e-3, name


@gpu_only
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,mask,backward", [
    (2, 512, 8, 2, 64, "causal", "ranged"),
    (1, 384, 4, 2, 128, "documents", "ranged"),
    (2, 1024, 4, 4, 128, "causal", "ranged"),
    (2, 512, 8, 2, 64, "causal", "fused"),
    (2, 512, 8, 2, 64, "documents", "fused"),
])
def test_longest_first_grid_is_exact(batch, length, heads, kv_heads, head_dim, mask, backward) -> None:
    """The FP8 backward's longest-first CTA order (``bwd_lpt``, the default) against the old grids: the same
    bits in every FP8 gradient, its transposed copy and its row maxima (the ranged kernels' ``dq, dk, dv``; the
    fused kernel's ``dk, dv``, since its float32 ``dq`` reductions follow CTA timing)."""
    import dataclasses

    from jla.ops.mosaic_gpu.attention import _backward_ranged
    from jla.ops.mosaic_gpu.attention_fp8_bwd import fp8_attention_backward

    cfg, q, k, v, do, seg, runs, positions, out, lse = _kernel_case(batch, length, heads, kv_heads, head_dim, mask)
    fp8_cfg = dataclasses.replace(cfg, backward=backward)
    assert fp8_cfg.bwd_lpt
    cos, sin = rope_tables(positions, theta=10_000.0, head_dim=head_dim)
    dq, dk, dv = jax.jit(lambda *a: _backward_ranged(cfg, *a))(q, k, v, seg, runs, out, lse, do)

    def rope_vjp(x, ct):
        return jax.vjp(lambda t: apply_rope(t, positions, theta=10_000.0), x)[1](ct)[0]

    expected_g = jax.jit(lambda: (rope_vjp(q, dq), rope_vjp(k, dk), dv))()
    gscale = jnp.stack([F.scale_from_amax(F._amax(g), F._fp8_max(F.E5M2), 1.0) for g in expected_g])
    args = (q, k, v, seg, runs, out, lse, do, cos, sin, gscale)
    on = jax.jit(lambda *a: fp8_attention_backward(*a, fp8_cfg))(*args)
    off = jax.jit(lambda *a: fp8_attention_backward(*a, dataclasses.replace(fp8_cfg, bwd_lpt=False)))(*args)
    checked = "qkv" if backward == "ranged" else "kv"
    for name, got, want in zip("qkv", on, off, strict=True):
        if name not in checked:
            continue
        for part, (a, b) in enumerate(zip(got, want, strict=True)):
            assert (a is None) == (b is None), (name, part)
            if a is not None:
                np.testing.assert_array_equal(_values(a), _values(b), err_msg=f"d{name} part {part}")


@gpu_only
@pytest.mark.parametrize("documents,window,backward", [
    (True, None, "ranged"), (False, None, "ranged"), (True, 96, "ranged"),
    (True, None, "fused"), (False, 96, "fused"),
])
def test_fused_layer_matches_plain_layer_on_gpu(documents, window, backward) -> None:
    """The whole layer, kernels against the plain path on the Mosaic GPU attention with the same backward."""
    common = dict(heads=8, kv_heads=2, head_dim=64, d_model=256, window=window, implementation="mosaic_gpu",
                  backward=backward)
    plain = _layer(**common)
    fused = _layer(**common, epilogue=True)
    x, g, positions, segment_ids = _inputs(2, 512, 256, documents, seed=5)
    assert fused_fp8_attention_reason(fused, x, segment_ids) is None
    v0, o0, (dx0, dl0) = _loss_and_grads(plain, x, positions, segment_ids, g, fused=None)
    v1, o1, (dx1, dl1) = _loss_and_grads(fused, x, positions, segment_ids, g, fused=None)
    np.testing.assert_array_equal(_values(o1), _values(o0))  # the same forward kernels and GEMMs
    _leaves_equal(dx1, dx0, exact=False, label="dx")
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        a, b = getattr(dl1, proj), getattr(dl0, proj)
        np.testing.assert_allclose(_values(a.weight), _values(b.weight), rtol=0,
                                   atol=2e-2 * float(np.max(np.abs(_values(b.weight)))), err_msg=proj)
        sa, sb = a.state, b.state
        for field in ("input_scale", "input_amax_history", "kernel_scale", "kernel_amax_history", "grad_scale"):
            np.testing.assert_array_equal(np.asarray(getattr(sa, field)), np.asarray(getattr(sb, field)),
                                          err_msg=f"{proj}.{field}")
        ga, gb = np.asarray(sa.grad_amax_history), np.asarray(sb.grad_amax_history)
        np.testing.assert_array_equal(ga[1:], gb[1:], err_msg=proj)
        assert abs(ga[0] - gb[0]) <= 2.0**-7 * gb[0], (proj, ga[0], gb[0])


# -- the per-shard rule (CPU, 4 devices, XLA stand-ins for the two kernel calls) ---------------------------------

_SHARDED_SCRIPT = textwrap.dedent(
    r"""
    import json

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    import jla.ops.mosaic_gpu.attention as A
    import jla.ops.mosaic_gpu.attention_fp8_bwd as FB
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.layers import fp8 as F
    from jla.layers.attention import Attention
    from jla.layers.fp8_attention import fp8_qkv_attention
    from jla.models.config import AttentionConfig

    assert len(jax.devices()) == 4, jax.devices()
    seen = []

    def attention(q, k, v, seg, cfg):
        mask = (seg[:, :, None] == seg[:, None, :])[:, None] if cfg.has_segments else None
        window = None if cfg.window is None else (cfg.window - 1, 0)
        return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=cfg.scale, is_causal=True,
                                            local_window_size=window, implementation="xla")

    def forward(q, k, v, seg, runs, sink_init, cfg):  # the kernels' operands and results, per device
        seen.append(("fwd", q.shape, k.shape, seg.shape, runs.shape))
        return attention(q, k, v, seg, cfg), jnp.zeros((q.shape[0], q.shape[2], q.shape[1]), jnp.float32)

    def backward(q, k, v, seg, runs, out, lse, do, cos, sin, gscale, cfg, *, transposed=False):
        seen.append(("bwd", q.shape, k.shape, out.shape, lse.shape, cos.shape, gscale.shape))
        dq, dk, dv = jax.vjp(lambda q, k, v: attention(q, k, v, seg, cfg), q, k, v)[1](do)

        def rope_backward(g):  # the epilogue's arithmetic, with the tables
            g = g.astype(jnp.float32)
            half = g.shape[-1] // 2
            c, s = cos[:, :, None], sin[:, :, None]
            g1, g2 = g[..., :half], g[..., half:]
            return jnp.concatenate([g2 * s + g1 * c, g2 * c - g1 * s], axis=-1).astype(q.dtype)

        grads = []
        for g, gs in zip((rope_backward(dq), rope_backward(dk), dv), gscale):
            g = g.astype(jnp.float32)
            g8 = jnp.clip(g / gs, -FB.E5M2_MAX, FB.E5M2_MAX).astype(jnp.float8_e5m2)
            grads.append((g8, None, jnp.max(jnp.abs(g), axis=3).transpose(0, 2, 1)))
        return grads

    A._forward_sharded, A.supported = forward, lambda q, k, v: None
    FB.fp8_attention_backward, FB.fp8_backward_supported = backward, lambda q, k, v: None

    B, T, D_MODEL, H, HKV, D = 4, 128, 128, 4, 2, 64
    cfg = AttentionConfig(num_heads=H, num_kv_heads=HKV, head_dim=D, qk_norm=False)
    attn = F.apply_fp8(Attention(D_MODEL, cfg, key=jax.random.key(1), std=0.2, norm_eps=1e-6), "delayed",
                       amax_history=8)
    attn = eqx.tree_at(lambda a: (a.q_proj, a.k_proj, a.v_proj), attn,
                       tuple(F.attach_fp8_weight(m, F.prepare_fp8_weight(m, transposed=True))
                             for m in (attn.q_proj, attn.k_proj, attn.v_proj)))
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((B, T, D_MODEL)), jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((B, T, H, D)), jnp.float32)
    seg = jnp.asarray(np.repeat(np.arange(4), T // 4)[None].repeat(B, 0), jnp.int32)
    positions = jnp.asarray(np.tile(np.arange(T // 4), 4)[None].repeat(B, 0), jnp.int32)
    arrays, static = eqx.partition(attn, eqx.is_inexact_array)

    def loss(x, arrays):
        layer = eqx.combine(arrays, static)
        o = fp8_qkv_attention(x, layer.q_proj, layer.k_proj, layer.v_proj, positions, num_heads=H,
                              num_kv_heads=HKV, head_dim=D, rope_theta=10_000.0, segment_ids=seg,
                              impl="mosaic_gpu", backward="ranged")
        return jnp.sum(o.astype(jnp.float32) * g)

    grad = jax.value_and_grad(loss, argnums=(0, 1))
    want = jax.jit(grad)(x, arrays)
    assert ("bwd", (B, T, H, D), (B, T, HKV, D), (B, T, H, D), (B, H, T), (B, T, D // 2), (3,)) in seen, seen
    mesh = build_mesh(resolve_mesh_plan(4, dp=2, fsdp=1, tp=2), topology_aware=False)
    seen.clear()
    with jax.set_mesh(mesh):
        got = jax.jit(grad)(jax.device_put(x, NamedSharding(mesh, P("dp"))), arrays)
    # Per device: half the batch (dp) and half the heads (tp, whole GQA groups); ids, runs, cos, sin with the batch.
    assert ("fwd", (B // 2, T, H // 2, D), (B // 2, T, HKV // 2, D), (B // 2, T), (B // 2, 4, T)) in seen, seen
    assert ("bwd", (B // 2, T, H // 2, D), (B // 2, T, HKV // 2, D), (B // 2, T, H // 2, D), (B // 2, H // 2, T),
            (B // 2, T, D // 2), (3,)) in seen, seen
    (v0, (dx0, da0)), (v1, (dx1, da1)) = want, got
    np.testing.assert_allclose(float(v1), float(v0), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dx1, np.float32), np.asarray(dx0, np.float32), rtol=0,
                               atol=2e-2 * float(np.max(np.abs(np.asarray(dx0, np.float32)))))
    for proj in ("q_proj", "k_proj", "v_proj"):
        a, b = getattr(da1, proj), getattr(da0, proj)
        np.testing.assert_allclose(np.asarray(a.weight), np.asarray(b.weight), rtol=0,
                                   atol=1e-2 * float(np.max(np.abs(np.asarray(b.weight)))))
        for u, w in zip(jax.tree.leaves(a.state), jax.tree.leaves(b.state), strict=True):  # amaxes: exact maxima
            np.testing.assert_array_equal(np.asarray(u), np.asarray(w), err_msg=proj)
    print("FP8ATTNSHARD " + json.dumps({"ok": True}))
    """
)


def test_per_shard_rule_on_cpu(tmp_path) -> None:
    """Under a context mesh the kernels run per batch and head shard (``shard_layout_call``): q/k/v and their
    gradients as ``[B, T, H, D]`` tokens, ``lse`` and the row maxima as ``[B, H, T]``, the ids, runs and RoPE
    tables with the batch, the scales replicated; results as one device's call (4 CPU devices, dp 2 x tp 2,
    XLA stand-ins with the kernels' signatures)."""
    script = tmp_path / "fp8_attention_sharded.py"
    script.write_text(_SHARDED_SCRIPT)
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               XLA_FLAGS="--xla_force_host_platform_device_count=4", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    assert any(line.startswith("FP8ATTNSHARD ") for line in proc.stdout.splitlines()), proc.stdout[-2000:]


@gpu_only
def test_zeros_kernel_writes_zeros() -> None:
    """The fused path's fresh dq buffer: every element zero, at a shape whose tiles cover several CTAs."""
    from jla.ops.mosaic_gpu.attention_fp8_bwd import _zeros

    out = jax.jit(lambda: _zeros((2, 384, 6, 64)))()
    assert out.shape == (2, 384, 6, 64) and out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), np.zeros((2, 384, 6, 64), np.float32))
