# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The RWKV-7 layers promise the op RWKV-7's decay bound, so the Mosaic GPU kernel runs its bounded mode.

``w = -exp(-1/2) sigmoid(.)`` is at least ``-0.6066`` for every input, so both layers (``RWKV7TimeMix``
and the LanguageModel mixer ``RWKV7Mixer``) pass ``lower_bound = RWKV7_LOWER_BOUND`` (-0.61) and the
default precision (the bounded mode's products always take f32 operands, by default as split-bf16
tensor-core products; plain bf16 operands lost dq/dk accuracy in the layer). The CPU
tests check the promise is passed and holds at saturated decay LoRAs, and that the reference and the
chunked op accept it without changing a bit; the GPU tests run the layers on the kernel (bounded mode)
against the chunked op at the strongest decay RWKV-7 can produce, with packed documents, odd lengths, an
initial state and decode.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jla.layers.rwkv7 as rwkv7_layer
from jla.layers.rwkv7 import RWKV7_LOWER_BOUND, RWKV7Mixer, RWKV7TimeMix
from jla.models.config import LinearAttentionConfig
from jla.ops import linear_attention
from jla.ops.mosaic_gpu.vector_gate_ops import is_hopper

gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100 (compute capability 9.0)")
F32 = jnp.float32
D = 128  # 2 heads x 64
_LA = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=64, head_v_dim=64, conv_size=0,
                            qk_l2norm=False, chunk_size=32, rwkv7_lora_dims=(16, 16, 16, 16))


def _perturb(m, seed: int = 7, scale: float = 0.05):
    params, static = eqx.partition(m, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _mixer(impl: str, *, decay_bias: float | None = None, compute_dtype: str = "bfloat16") -> RWKV7Mixer:
    """A later-layer mixer, perturbed; ``decay_bias`` sets the decay LoRA's bias (large = strongest decay)."""
    la = dataclasses.replace(_LA, implementation=impl)
    m = _perturb(RWKV7Mixer(D, la, layer_idx=1, n_layers=3, key=jax.random.key(1), norm_eps=1e-5,
                            compute_dtype=compute_dtype))
    if decay_bias is not None:
        m = eqx.tree_at(lambda t: t.w_lora.up.bias, m, jnp.full((D,), decay_bias, F32))
    return m


def _spy(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    real = rwkv7_layer.linear_attention

    def spy(kind, q, k, v, **kw):
        calls.append({"lower_bound": kw.get("lower_bound"), "precision": kw.get("precision"),
                      "w_min": kw["w"].min(), "w_max": kw["w"].max()})
        return real(kind, q, k, v, **kw)

    monkeypatch.setattr(rwkv7_layer, "linear_attention", spy)
    return calls


@pytest.mark.parametrize("decay_bias", [None, 30.0, -30.0])
def test_mixer_promises_the_bound_and_it_holds(monkeypatch, decay_bias):
    calls = _spy(monkeypatch)
    m = _mixer("chunked", decay_bias=decay_bias)
    x = jax.random.normal(jax.random.key(0), (1, 40, D)).astype(jnp.bfloat16)
    m(x)
    assert len(calls) == 1 and calls[0]["lower_bound"] == RWKV7_LOWER_BOUND
    assert calls[0]["precision"] is None  # the bounded mode's accurate default (module comment)
    assert RWKV7_LOWER_BOUND <= float(calls[0]["w_min"]) <= float(calls[0]["w_max"]) <= 0.0
    if decay_bias == 30.0:  # saturated: the most negative decay RWKV-7 produces
        assert float(calls[0]["w_min"]) == pytest.approx(-np.exp(-0.5), abs=1e-6)
        # also after rounding to bf16 (the kernels accept bf16 w)
        assert float(calls[0]["w_min"].astype(jnp.bfloat16)) >= RWKV7_LOWER_BOUND


def test_time_mix_promises_the_bound(monkeypatch):
    calls = _spy(monkeypatch)
    mix = RWKV7TimeMix(D, head_dim=64, layer_idx=0, n_layers=2, key=jax.random.key(3), lora_dims=(16, 16, 16, 16),
                       chunk_size=32, impl="chunked")
    mix(jax.random.normal(jax.random.key(4), (1, 33, D)).astype(jnp.bfloat16), None)
    assert [(c["lower_bound"], c["precision"]) for c in calls] == [(RWKV7_LOWER_BOUND, None)]


def _op_inputs(seed: int, batch: int, seq_len: int, heads: int = 2, dim: int = 64):
    ks = jax.random.split(jax.random.key(seed), 7)
    shape = (batch, seq_len, heads, dim)
    kk = jax.random.normal(ks[0], shape)
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    return dict(
        q=jax.random.normal(ks[1], shape), k=0.5 * jax.random.normal(ks[2], shape),
        v=jax.random.normal(ks[3], shape), w=-np.exp(-0.5) * jax.nn.sigmoid(2 * jax.random.normal(ks[4], shape)),
        a=-kk, b=kk * jax.nn.sigmoid(jax.random.normal(ks[5], shape)),
    )


@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_reference_and_chunked_accept_the_promise_bit_for_bit(impl):
    x = _op_inputs(0, 2, 37)
    seg = jnp.asarray(np.repeat([[0, 1], [2, 2]], [20, 17], axis=1), jnp.int32)
    h0 = jax.random.normal(jax.random.key(9), (2, 2, 64, 64))
    run = lambda **kw: linear_attention("rwkv7", x["q"], x["k"], x["v"], w=x["w"], a=x["a"], b=x["b"],  # noqa: E731
                                        impl=impl, segment_ids=seg, initial_state=h0, output_final_state=True, **kw)
    for got, want in zip(run(lower_bound=RWKV7_LOWER_BOUND), run()):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    for bad in (0.0, 0.5, -3.0):
        with pytest.raises(ValueError, match="lower_bound"):
            run(lower_bound=bad)


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _grads_close(g_a, g_b, tol: float) -> None:
    leaves_a = jax.tree.leaves(eqx.filter(g_a, eqx.is_inexact_array))
    leaves_b = jax.tree.leaves(eqx.filter(g_b, eqx.is_inexact_array))
    assert len(leaves_a) == len(leaves_b) > 10
    for a, b in zip(leaves_a, leaves_b):
        if float(jnp.max(jnp.abs(b))) == 0.0:  # no gradient path (the first layer's value LoRA)
            assert float(jnp.max(jnp.abs(a))) == 0.0
            continue
        assert _rel(a, b) < tol, (a.shape, _rel(a, b))


@gpu
@pytest.mark.parametrize("decay_bias", [None, 30.0])
@pytest.mark.parametrize("seq_len", [33, 100, 257])
def test_mixer_on_the_bounded_kernel_matches_chunked(monkeypatch, decay_bias, seq_len):
    """Output and every parameter gradient, packed documents, odd lengths, at RWKV-7's strongest decay."""
    from jla import ops

    fn, supports, auto = ops._REGISTRY["mosaic_gpu"]["rwkv7"]
    seen: list = []

    def counted(*args, **kwargs):
        seen.append((kwargs.get("lower_bound"), kwargs.get("precision")))
        return fn(*args, **kwargs)

    monkeypatch.setitem(ops._REGISTRY["mosaic_gpu"], "rwkv7", (counted, supports, auto))
    x = jax.random.normal(jax.random.key(2), (2, seq_len, D)).astype(jnp.bfloat16)
    cuts = [seq_len // 3, seq_len // 3 + 1]
    seg = jnp.asarray(np.repeat([[0, 1, 2], [3, 3, 4]], [cuts[0], cuts[1], seq_len - sum(cuts)], axis=1), jnp.int32)
    kernel, chunked = _mixer("auto", decay_bias=decay_bias), _mixer("chunked", decay_bias=decay_bias)

    def loss(m):
        return jnp.sum(jnp.sin(m(x, seg).astype(F32)))

    y_k, y_c = jax.jit(kernel.__call__)(x, seg), jax.jit(chunked.__call__)(x, seg)
    assert seen and all(s == (RWKV7_LOWER_BOUND, None) for s in seen), "bounded kernel not run"
    assert _rel(y_k, y_c) < 1e-2
    _grads_close(eqx.filter_jit(eqx.filter_grad(loss))(kernel), eqx.filter_jit(eqx.filter_grad(loss))(chunked), 3e-2)


@gpu
def test_mixer_prefill_on_the_kernel_then_decode_matches_full_sequence():
    """Prefill through the bounded kernel (initial and final state), then one-token decode steps."""
    m = _mixer("auto")
    x = jax.random.normal(jax.random.key(5), (2, 75, D)).astype(jnp.bfloat16)
    full = jax.jit(m.__call__)(x)
    cache = m.init_cache(2, jnp.bfloat16)
    step = eqx.filter_jit(lambda m, x, c: m.step(x, c))
    out, cache = step(m, x[:, :40], cache)  # 40 tokens: the kernel, from the zero state
    outs = [out]
    out, cache = step(m, x[:, 40:70], cache)  # 30 tokens: the kernel, from the carried state
    outs.append(out)
    for t in range(70, 75):  # one token at a time: the reference op
        out, cache = step(m, x[:, t:t + 1], cache)
        outs.append(out)
    assert _rel(jnp.concatenate(outs, axis=1), full) < 1e-2


# ------------------------------------------------------------------------------------------------
# the f32 bounded mode's one-product scores and the WY inverse, with their hand-written VJPs
# ------------------------------------------------------------------------------------------------


def _x3_on_gpu(name: str):
    from jla.ops.mosaic_gpu import rwkv7 as mg

    if name == "highest":
        return jax.lax.Precision.HIGHEST, 1e-5, 1e-4
    if not is_hopper():
        pytest.skip("the BF16X3 dot algorithm runs on the GPU")
    return mg.BF16X3, 1e-4, 1e-3  # about 16 significant bits per operand


@pytest.mark.parametrize("product", ["highest", "bf16x3"])
@pytest.mark.parametrize("decay", ["at_min_bound", "at_rwkv7_bound", "uniform"])
def test_f32_bounded_scores_and_vjp_match_pairwise_autodiff(decay, product):
    """The four score matrices from one product per chunk (values and the VJP, through ``w``) against
    autodiff of the exact pairwise formula, at the most negative bound the op accepts, with f32
    (``HIGHEST``) and split-bf16 (``BF16X3``, the bounded mode's default) products."""
    from jla.ops.mosaic_gpu import rwkv7 as mg

    precision, value_tol, grad_tol = _x3_on_gpu(product)

    ks = jax.random.split(jax.random.key(11), 6)
    shape = (2, 3, mg.CHUNK, 64)
    q, a, k, b = (jax.random.normal(kk, shape) for kk in ks[:4])
    cts = tuple(jax.random.normal(kk, shape[:-1] + (mg.CHUNK,)) for kk in jax.random.split(ks[4], 4))
    bound = {"at_min_bound": mg.MIN_LOWER_BOUND, "at_rwkv7_bound": RWKV7_LOWER_BOUND,
             "uniform": mg.MIN_LOWER_BOUND}[decay]
    w = bound * (jnp.ones(shape) if decay != "uniform" else jax.random.uniform(ks[5], shape))
    hi = jax.lax.Precision.HIGHEST
    t = jnp.arange(mg.CHUNK)
    causal, strict = t[:, None] >= t[None, :], t[:, None] > t[None, :]

    def gates(w):
        g = jnp.cumsum(w, axis=-2)
        return g, jnp.concatenate([jnp.zeros_like(g[..., :1, :]), g[..., :-1, :]], axis=-2)

    def fused(q, a, k, b, w):
        g, g_prev = gates(w)
        return sum(jnp.sum(c * s) for c, s in zip(cts, mg._bounded_f32_scores(q, a, k, b, g, g_prev, precision)))

    def pairwise(q, a, k, b, w):
        g, g_prev = gates(w)

        def pair(x, gx, y, mask):
            diff = jnp.where(mask[:, :, None], gx[..., :, None, :] - g[..., None, :, :], -jnp.inf)
            return jnp.einsum("...tk,...tik,...ik->...ti", x, jnp.exp(diff), y, precision=hi)

        four = (pair(q, g, k, causal), pair(q, g, b, causal), pair(a, g_prev, k, strict), pair(a, g_prev, b, strict))
        return sum(jnp.sum(c * s) for c, s in zip(cts, four))

    args = (q, a, k, b, w)
    np.testing.assert_allclose(float(jax.jit(fused)(*args)), float(jax.jit(pairwise)(*args)), rtol=value_tol)
    got = jax.jit(jax.grad(fused, tuple(range(5))))(*args)
    want = jax.jit(jax.grad(pairwise, tuple(range(5))))(*args)
    for name, g_got, g_want in zip(["dq", "da", "dk", "db", "dw"], got, want):
        scale = float(jnp.max(jnp.abs(g_want)))
        err = float(jnp.max(jnp.abs(g_got - g_want)))
        assert err <= grad_tol * scale, (name, err / scale)


def test_unit_lower_inverse_vjp_matches_the_inverse():
    from jla.ops.mosaic_gpu import rwkv7 as mg

    c = mg.CHUNK
    a = jnp.tril(0.3 * jax.random.normal(jax.random.key(1), (3, c, c)), -1)
    ct = jax.random.normal(jax.random.key(2), (3, c, c))
    hi = jax.lax.Precision.HIGHEST
    eye = jnp.eye(c)
    with jax.default_matmul_precision("highest"):
        want = jax.grad(lambda a: jnp.sum(ct * jnp.linalg.inv(eye - jnp.tril(a, -1))))(a)
    got = jax.jit(jax.grad(lambda a: jnp.sum(ct * mg._unit_lower_inverse(a, hi))))(a)
    np.testing.assert_allclose(np.asarray(got), np.asarray(jnp.tril(want, -1)), rtol=1e-4,
                               atol=1e-4 * float(jnp.max(jnp.abs(want))))


# ------------------------------------------------------------------------------------------------
# the scanned LanguageModel carries no [B, T, D] value residual
# ------------------------------------------------------------------------------------------------


def test_scanned_rwkv7_stack_takes_v_first_as_a_scan_constant(monkeypatch):
    """With an RWKV-7 first layer, v_first enters lax.scan as a constant: the carry holds a scalar
    placeholder, so remat saves v_first once instead of once per layer (outputs and gradients against
    the looped stack are in tests/test_models_lm_extra.py and tests/test_models_lm.py)."""
    from jla.layers.rwkv7 import ValueResidual
    from jla.models import build_model, get_preset
    import jla.models.lm as lm

    carries = []
    real_scan = lm.lax.scan

    def spy(body, init, xs, *args, **kwargs):
        carries.append(init)
        return real_scan(body, init, xs, *args, **kwargs)

    monkeypatch.setattr(lm.lax, "scan", spy)
    cfg = get_preset("tiny-rwkv7").replace(scan_layers=True, compute_dtype="float32")
    model = build_model(cfg, key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (2, 24), 0, cfg.vocab_size)
    out = model(ids)
    residuals = [leaf for c in carries for leaf in jax.tree.leaves(c, is_leaf=lambda t: isinstance(t, ValueResidual))
                 if isinstance(leaf, ValueResidual)]
    assert residuals and all(r.v_first.shape == () for r in residuals)
    assert bool(jnp.all(jnp.isfinite(out.hidden)))


def _leaves(tree) -> list:
    return jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))


def _restacked(grads, period: int):
    """A looped model's gradients in the scanned layout (as tests/test_models_lm_extra.py restacks them)."""
    blocks = tuple(jax.tree.map(lambda *xs: jnp.stack(xs), *grads.blocks[j::period]) for j in range(period))
    return eqx.tree_at(lambda g: g.blocks, grads, blocks, is_leaf=lambda x: x is None)


@gpu
def test_scanned_bf16_rwkv7_model_on_the_kernel_matches_chunked_and_the_loop():
    """A bf16 RWKV-7 LanguageModel (heads of 64, chunk 32, packed documents, an odd length) under remat
    on the bounded kernel, with the next-token cross entropy as the loss.

    Against an f32 chunked model (the truth), the kernel's loss, hidden states and every parameter
    gradient, leaf by leaf, are as accurate as the bf16 chunked model's (the bf16 noise floor). The
    scanned stack (v_first hoisted out of the carry) against the looped stack (v_first in the carry),
    leaf by leaf: within that floor on the kernel, and within f32 rounding for the f32 chunked models.
    """
    from jla.models import build_model, get_preset
    from jla.models.config import LinearAttentionConfig
    from jla.ops.cross_entropy import linear_cross_entropy

    base = get_preset("tiny-rwkv7")

    def model(impl: str, scan: bool, dtype: str = "bfloat16"):
        la = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=64, head_v_dim=64, conv_size=0,
                                   qk_l2norm=False, chunk_size=32, rwkv7_lora_dims=(16, 16, 16, 16),
                                   implementation=impl)
        cfg = base.replace(d_model=128, d_ff=256, linear_attention=la, scan_layers=scan, remat="full",
                           compute_dtype=dtype)
        # The same weights in both layouts, every float parameter moved off its init (zero o_proj and
        # value make each block the identity at init): the looped stack's values, restacked when scanned.
        looped = _perturb(build_model(cfg.replace(scan_layers=False), key=jax.random.key(0)))
        if not scan:
            return looped
        scanned = build_model(cfg, key=jax.random.key(0))
        values = jax.tree.leaves(_restacked(eqx.filter(looped, eqx.is_array), len(scanned.blocks)))
        arrays, static = eqx.partition(scanned, eqx.is_array)
        return eqx.combine(jax.tree.unflatten(jax.tree.structure(arrays), values), static)

    ids = jax.random.randint(jax.random.key(1), (2, 101), 0, base.vocab_size)
    seg = jnp.asarray(np.repeat([[0, 1, 2], [3, 3, 4]], [30, 42, 29], axis=1), jnp.int32)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((2, 1), -100)], axis=1).reshape(-1)

    def loss(m):
        out = m(ids, segment_ids=seg)
        ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), m.lm_head_weight(), targets,
                                  compute_dtype=F32)
        return ce.loss_sum / ce.num_valid, out.hidden.astype(F32)

    runs, period = {}, len(model("chunked", True, "float32").blocks)
    for name, (impl, scan, dtype) in {
        "truth": ("chunked", True, "float32"), "truth_loop": ("chunked", False, "float32"),
        "kernel": ("auto", True, "bfloat16"), "kernel_loop": ("auto", False, "bfloat16"),
        "chunked": ("chunked", True, "bfloat16"),
    }.items():
        # f32 models with f32 matmuls: the GPU default (TF32) would leave 3e-4 between scan and loop
        with jax.default_matmul_precision("highest" if dtype == "float32" else None):
            m = model(impl, scan, dtype)
            (value, hidden), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(m)
        if not scan:
            grads = _restacked(grads, period)
        runs[name] = (float(value), hidden, _leaves(grads))
    truth, chunked = runs["truth"], runs["chunked"]
    assert len(truth[2]) == len(chunked[2]) > 10
    loss_floor = abs(chunked[0] - truth[0])
    hidden_floor = _rel(chunked[1], truth[1])
    leaf_floor = [_rel(c, t) for c, t in zip(chunked[2], truth[2])]

    def within_floor(name: str, ref: tuple) -> None:
        value, hidden, leaves = runs[name]
        assert abs(value - ref[0]) <= 1.5 * loss_floor + 1e-3 * abs(truth[0]), (name, value, ref[0])
        assert _rel(hidden, ref[1]) <= 1.5 * hidden_floor + 1e-3, (name, _rel(hidden, ref[1]), hidden_floor)
        for i, (got, want, floor) in enumerate(zip(leaves, ref[2], leaf_floor)):
            if float(jnp.max(jnp.abs(want))) == 0.0:  # no gradient path (the first layer's value LoRA)
                assert float(jnp.max(jnp.abs(got))) == 0.0, (name, i, got.shape)
                continue
            assert _rel(got, want) <= 1.5 * floor + 5e-3, (name, i, got.shape, _rel(got, want), floor)

    within_floor("kernel", truth)  # the bounded kernel adds no error beyond bf16 compute
    within_floor("kernel_loop", runs["kernel"])  # the hoisted v_first: scan = loop on the kernel
    # and in f32 (the chunked op): scan = loop to f32 rounding, leaf by leaf
    t_loop = runs["truth_loop"]
    assert abs(t_loop[0] - truth[0]) <= 1e-5 * abs(truth[0])
    assert _rel(t_loop[1], truth[1]) <= 2e-5
    for got, want in zip(t_loop[2], truth[2]):
        scale = float(jnp.max(jnp.abs(want)))
        assert float(jnp.max(jnp.abs(got - want))) <= 1e-4 * scale + 1e-7, (got.shape, scale)
