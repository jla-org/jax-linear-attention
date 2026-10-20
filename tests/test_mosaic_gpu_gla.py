# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU GLA (``kind="gla"``) against the token-by-token oracle ``jla.ops.reference.gla``.

Both pipelines (``pipeline="gla"``, the dedicated kernels and the default, and ``"vector_gate"``,
the KDA / GDN-2 kernels with constant gates): outputs, final states and every gradient (``q, k, v,
g`` and the initial state) on odd and chunk-multiple lengths, packed documents (sparse and dense
``segment_ids``), huge decays (the exact intra-chunk path; the default gates take the factored
one), the forward-only path with many heads, no initial state, the dispatcher (registered with
``auto=True``) and the ``supports`` predicate. The two pipelines agree to 1e-4; the GLA and HGRN2
layers run on the dedicated kernels (``implementation="mosaic_gpu"``) and a small GLA model trains
on them; and the dedicated kernels also run in the Mosaic GPU interpreter on the CPU backend.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import F32, assert_close, dense_segments, inputs, segments

from jla import ops
from jla.layers.linear_attention import make_mixer
from jla.layers.rwkv6 import HGRN2
from jla.models.config import LinearAttentionConfig
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import gla as mg_gla
from jla.ops.mosaic_gpu import gla_kernels
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

ON_HOPPER = vgo.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
cpu_only = pytest.mark.skipif(jax.default_backend() != "cpu", reason="interpreter tests run on the CPU backend")
pipelines = pytest.mark.parametrize("pipeline", ["gla", "vector_gate"])


def _case(seed, shape, *, huge=False):
    x = inputs(seed, "gdn2", *shape, huge=huge)
    return x["q"], x["k"], x["v"], x["g"], x["h0"], x["do"], x["dht"]


def _reference(q, k, v, g, h0, seg):
    return ref.gla(q.astype(F32), k.astype(F32), v.astype(F32), g, initial_state=h0, output_final_state=True,
                   segment_ids=seg)


@gpu
@pipelines
@pytest.mark.parametrize(
    "shape",
    [(2, 200, 2, 128, 128), (1, 100, 2, 64, 64), (1, 130, 2, 128, 64), (3, 130, 4, 128, 128), (1, 1, 2, 64, 128),
     (2, 77, 2, 64, 128), (4, 192, 12, 128, 128)],
    ids=["k128v128_t200", "k64v64_t100", "k128v64_t130", "bh12_fused_state_pass", "t1", "k64v128_t77",
         "bh48_block_v128"],
)
def test_gla_forward_matches_reference(shape, pipeline):
    q, k, v, g, h0, *_ = _case(0, shape)
    o, final = ops.linear_attention("gla", q, k, v, g=g, impl="mosaic_gpu", initial_state=h0,
                                    output_final_state=True, pipeline=pipeline)
    o_ref, final_ref = _reference(q, k, v, g, h0, None)
    assert o.dtype == q.dtype and final.dtype == F32
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")


@gpu
@pipelines
@pytest.mark.parametrize("shape", [(1, 256, 2, 128, 128), (2, 131, 2, 64, 64)], ids=["k128", "k64_t131"])
def test_gla_huge_decay_forward_matches_reference(shape, pipeline):
    """-60 log-decays on 10% of the entries: every chunk takes the exact intra-chunk path."""
    q, k, v, g, h0, *_ = _case(2, shape, huge=True)
    o, final = mg_gla.gla(q, k, v, g, initial_state=h0, output_final_state=True, pipeline=pipeline)
    o_ref, final_ref = _reference(q, k, v, g, h0, None)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")


@gpu
@pipelines
@pytest.mark.parametrize("dense", [False, True], ids=["sparse_segments", "dense_segments"])
def test_gla_segments_forward_match_reference(dense, pipeline):
    shape = (2, 300, 2, 128, 64)
    q, k, v, g, h0, *_ = _case(1, shape)
    seg = dense_segments(2, 300) if dense else segments(2, 2, 300)
    o, final = mg_gla.gla(q, k, v, g, initial_state=h0, output_final_state=True, segment_ids=seg,
                          pipeline=pipeline)
    o_ref, final_ref = _reference(q, k, v, g, h0, seg)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")


@gpu
@pipelines
@pytest.mark.parametrize(
    "case",
    [
        dict(shape=(2, 200, 2, 128, 128)),
        dict(shape=(1, 256, 2, 64, 64), packed=True),
        dict(shape=(1, 256, 2, 128, 128), huge=True, packed=True),
        dict(shape=(1, 192, 2, 128, 64), huge=True),
        dict(shape=(2, 131, 4, 64, 128), packed=True, no_state=True),
        dict(shape=(1, 77, 2, 128, 64)),
        dict(shape=(2, 192, 12, 64, 128), packed=True),  # B H = 24: 64-wide value tiles in the state pass
        dict(shape=(4, 130, 12, 128, 128), packed=True),  # B H = 48: 128-wide value tiles
        dict(shape=(1, 300, 2, 128, 64), dense=True),
    ],
    ids=["t200", "k64_packed", "huge_decay_packed", "huge_decay", "k64v128_t131_packed_no_state", "t77",
         "bh24_block_v64", "bh48_block_v128", "dense_segments"],
)
def test_gla_gradients_match_reference(case, pipeline):
    shape = case["shape"]
    q, k, v, g, h0, do, dht = _case(3, shape, huge=case.get("huge", False))
    if case.get("no_state"):
        h0 = None
    seg = segments(4, shape[0], shape[1]) if case.get("packed") else None
    if case.get("dense"):
        seg = dense_segments(shape[0], shape[1])

    def loss(fn):
        def f(q, k, v, g, h0):
            o, final = fn(q, k, v, g, h0)
            return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht)

        return f

    def kernel(q, k, v, g, h0):
        return mg_gla.gla(q, k, v, g, initial_state=h0, output_final_state=True, segment_ids=seg,
                          pipeline=pipeline)

    def oracle(q, k, v, g, h0):
        return _reference(q, k, v, g, h0, seg)

    argnums = (0, 1, 2, 3) if h0 is None else (0, 1, 2, 3, 4)
    got = jax.jit(jax.grad(loss(kernel), argnums))(q, k, v, g, h0)
    want = jax.jit(jax.grad(loss(oracle), argnums))(q, k, v, g, h0)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dh0"), got, want):
        assert_close(a, b, 2e-2, name)
    assert got[0].dtype == q.dtype and got[1].dtype == k.dtype and got[2].dtype == v.dtype
    assert got[3].dtype == g.dtype


@gpu
@pipelines
def test_gla_without_state_outputs(pipeline):
    q, k, v, g, *_ = _case(5, (1, 192, 2, 128, 128))
    o, final = mg_gla.gla(q, k, v, g, pipeline=pipeline)
    assert final is None
    o_ref, _ = _reference(q, k, v, g, None, None)
    assert_close(o, o_ref, 1.5e-2, "o")


@gpu
def test_gla_pipelines_agree():
    """The dedicated and the vector-gate pipelines agree on the output, the final state and all five
    gradients to 1e-4 of the largest magnitude: with the constant gates the vector-gate pipeline
    computes the same bf16 tile products (its delta-rule terms are exact zeros), so a regression in
    either pipeline far below the oracle tolerance fails here."""
    shape = (2, 256, 4, 128, 128)
    q, k, v, g, h0, do, dht = _case(6, shape)
    seg = segments(7, shape[0], shape[1])

    def run(pipeline):
        def f(q, k, v, g, h0):
            return mg_gla.gla(q, k, v, g, initial_state=h0, output_final_state=True, segment_ids=seg,
                              pipeline=pipeline)

        return f

    def loss(pipeline):
        def f(q, k, v, g, h0):
            o, final = run(pipeline)(q, k, v, g, h0)
            return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht)

        return f

    for name, a, b in zip(("o", "final"), jax.jit(run("gla"))(q, k, v, g, h0),
                          jax.jit(run("vector_gate"))(q, k, v, g, h0)):
        assert_close(a, b, 1e-4, name)
    new = jax.jit(jax.grad(loss("gla"), (0, 1, 2, 3, 4)))(q, k, v, g, h0)
    old = jax.jit(jax.grad(loss("vector_gate"), (0, 1, 2, 3, 4)))(q, k, v, g, h0)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dh0"), new, old):
        assert_close(a, b, 1e-4, name)


_LAYER_D = 256


def _layer(kind: str, implementation: str):
    """A GLA or HGRN2 layer whose op calls satisfy the kernels' shape rules (``H K`` a multiple of 128)."""
    if kind == "hgrn2":
        cfg = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=128, head_v_dim=64, conv_size=0,
                                    output_gate=False, implementation=implementation)
        return HGRN2(_LAYER_D, cfg, key=jax.random.key(0), std=0.05, norm_eps=1e-5)
    cfg = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=64, head_v_dim=128,
                                implementation=implementation)
    return make_mixer(kind, _LAYER_D, cfg, key=jax.random.key(0), std=0.05, norm_eps=1e-5)


@gpu
@pytest.mark.parametrize("kind", ["gla", "hgrn2"])
def test_gla_layers_on_the_kernels_match_the_reference(kind):
    """The GLA and HGRN2 layers with ``implementation="mosaic_gpu"`` (fail-closed dispatch, so the
    dedicated pipeline runs) against the same layers on the f32 oracle: the output and the gradients
    of the input and of every parameter with packed documents, and a two-call prefill through the
    recurrent cache (initial and final states, odd lengths)."""
    B, T = 2, 200
    kernel_layer, oracle_layer = _layer(kind, "mosaic_gpu"), _layer(kind, "reference")  # same key and parameters
    x = jax.random.normal(jax.random.key(1), (B, T, _LAYER_D))
    dy = jax.random.normal(jax.random.key(2), (B, T, _LAYER_D))
    seg = segments(9, B, T)

    @eqx.filter_jit
    def forward(layer, x):
        return layer(x, segment_ids=seg)

    @eqx.filter_jit
    @eqx.filter_grad
    def grads(layer_and_x):
        layer, x = layer_and_x
        return jnp.sum(layer(x, segment_ids=seg).astype(F32) * dy)

    @eqx.filter_jit
    def prefill(layer, x):
        _, cache = layer.step(x[:, :130], layer.init_cache(B, jnp.bfloat16))
        y, cache = layer.step(x[:, 130:], cache)
        return y, cache.state

    assert_close(forward(kernel_layer, x), forward(oracle_layer, x), 1.5e-2, "y")
    got = jax.tree.leaves(grads((kernel_layer, x)))
    want = jax.tree.leaves(grads((oracle_layer, x)))
    assert len(got) == len(want) > 2
    for i, (a, b) in enumerate(zip(got, want)):
        assert_close(a, b, 2e-2, f"grad{i}")
    for name, a, b in zip(("prefill y", "prefill state"), prefill(kernel_layer, x), prefill(oracle_layer, x)):
        assert_close(a, b, 1.5e-2, name)


@gpu
def test_gla_model_trains_on_the_kernels():
    """A two-layer all-GLA model (``gla-125m`` layer options with two heads, bf16 compute) on the dedicated
    kernels: the loss and every gradient of one step with packed documents against the same model in
    float32 on the chunked op, within twice the error of the bf16 model on the chunked op (plus 1e-2 of each
    gradient's max), then six SGD steps on the kernels that lower the loss.

    In bf16 the model's gradients are 1 % to 8 % of their max away from float32 on any implementation
    (the projections and norms round to bf16); the kernels add their bf16 tile operands on top (measured
    on an H100: at most 1.8x the chunked op's error per gradient)."""
    import dataclasses

    from jla.models import build_model, get_preset
    from jla.ops.cross_entropy import linear_cross_entropy

    base = get_preset("gla-125m")

    def config(impl, dtype="bfloat16"):
        la = dataclasses.replace(base.linear_attention, num_k_heads=2, num_v_heads=2, implementation=impl)
        return base.replace(vocab_size=512, d_model=256, n_layers=2, d_ff=512, layer_types=("gla",) * 2,
                            linear_attention=la, compute_dtype=dtype)

    ids = jax.random.randint(jax.random.key(1), (2, 256), 0, 512)
    seg = jnp.concatenate([jnp.zeros((2, 100), jnp.int32), jnp.ones((2, 156), jnp.int32)], axis=1)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((2, 1), -100)], axis=1)

    def loss(model):
        out = model(ids, segment_ids=seg)
        ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                                  targets.reshape(-1), compute_dtype=F32)
        return ce.loss_sum / ce.num_valid

    step = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    with jax.default_matmul_precision("highest"):
        loss_ref, grads_ref = step(build_model(config("chunked", "float32"), key=jax.random.key(0)))
    kernel_model = build_model(config("mosaic_gpu"), key=jax.random.key(0))  # fail-closed: the kernels run
    loss_k, grads_k = step(kernel_model)
    _, grads_c = step(build_model(config("chunked"), key=jax.random.key(0)))
    assert abs(float(loss_k) - float(loss_ref)) <= 1e-3 * abs(float(loss_ref)), (float(loss_k), float(loss_ref))
    leaves = [jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array)) for g in (grads_k, grads_c, grads_ref)]
    assert len(leaves[0]) == len(leaves[1]) == len(leaves[2]) > 10

    def err(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-30)

    for i, (k, c, r) in enumerate(zip(*leaves)):
        assert np.isfinite(np.asarray(k)).all(), f"grad{i}: non-finite"
        assert err(k, r) <= 2.0 * err(c, r) + 1e-2, (i, err(k, r), err(c, r))

    @eqx.filter_jit
    def sgd(model):
        value, grads = eqx.filter_value_and_grad(loss)(model)
        return eqx.apply_updates(model, jax.tree.map(lambda g: -0.5 * g, grads)), value

    losses = []
    for _ in range(6):
        kernel_model, value = sgd(kernel_model)
        losses.append(float(value))
    assert all(map(jnp.isfinite, losses)) and losses[-1] < losses[0] - 0.1, losses


@cpu_only
@pytest.mark.parametrize("packed", [False, True], ids=["t70", "t130_packed"])
def test_gla_dedicated_pipeline_in_the_interpreter(packed):
    """The dedicated kernels in the Mosaic GPU interpreter (CPU backend): output, final state and the
    gradients of ``q, k, v, g`` and the initial state against the oracle, at odd lengths."""
    shape = (1, 130, 2, 64, 64) if packed else (1, 70, 2, 64, 64)
    q, k, v, g, h0, do, dht = _case(8, shape)
    seg = segments(10, shape[0], shape[1]) if packed else None
    scale = shape[3] ** -0.5

    def kernel(q, k, v, g, h0):
        return gla_kernels.chunked_gla(q, k, v, g, scale=scale, initial_state=h0, segment_ids=seg, interpret=True)

    def oracle(q, k, v, g, h0):
        return _reference(q, k, v, g, h0, seg)

    def loss(fn):
        def f(q, k, v, g, h0):
            o, final = fn(q, k, v, g, h0)
            return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht), (o, final)

        return f

    (_, (o, final)), got = jax.value_and_grad(loss(kernel), (0, 1, 2, 3, 4), has_aux=True)(q, k, v, g, h0)
    (_, (o_ref, final_ref)), want = jax.value_and_grad(loss(oracle), (0, 1, 2, 3, 4), has_aux=True)(q, k, v, g, h0)
    assert_close(o, o_ref, 1.5e-2, "o")
    assert_close(final, final_ref, 1.5e-2, "final")
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dh0"), got, want):
        assert_close(a, b, 2e-2, name)


def test_gla_dispatch():
    """``impl="auto"`` takes the kernels for the calls ``supports_gla`` accepts (and XLA for the rest);
    ``impl="mosaic_gpu"`` fails closed."""
    B, T, H, K, V = 1, 64, 2, 128, 64
    q = jnp.zeros((B, T, H, K), jnp.bfloat16)
    v = jnp.zeros((B, T, H, V), jnp.bfloat16)
    g = jnp.zeros((B, T, H, K))
    assert mg_gla.supports_gla(q, q, v, g) == ON_HOPPER
    assert mg_gla.supports_gla(q, q, v, g, segment_ids=jnp.zeros((B, T), jnp.int32)) == ON_HOPPER
    assert mg_gla.supports_gla(q, q, v, g, pipeline="vector_gate") == ON_HOPPER
    assert not mg_gla.supports_gla(q, q, v, g, pipeline="other")
    assert not mg_gla.supports_gla(q, q, v, g, use_qk_l2norm=True)  # not a GLA option
    assert not mg_gla.supports_gla(q, q, v, g, cu_seqlens=jnp.array([0, T]))
    assert not mg_gla.supports_gla(q.astype(F32), q, v, g)
    assert not mg_gla.supports_gla(q, q, v, g[..., :64])  # g must match q
    assert not mg_gla.supports_gla(q[..., :32], q[..., :32], v, g[..., :32])  # K = 32
    o_auto, _ = ops.linear_attention("gla", q.astype(F32), q.astype(F32), v.astype(F32), g=g)
    assert o_auto.shape == v.shape
    if ON_HOPPER:  # a supported call: auto runs the dedicated kernels, bit for bit
        x = jax.random.normal(jax.random.key(3), (B, T, H, K)).astype(jnp.bfloat16)
        y = jax.random.normal(jax.random.key(4), (B, T, H, V)).astype(jnp.bfloat16)
        gate = jax.nn.log_sigmoid(jax.random.normal(jax.random.key(5), (B, T, H, K))) / 16
        auto, _ = ops.linear_attention("gla", x, x, y, g=gate)
        kernel, _ = ops.linear_attention("gla", x, x, y, g=gate, impl="mosaic_gpu")
        assert bool(jnp.all(auto == kernel))
    for pipeline in mg_gla.PIPELINES:
        with pytest.raises((ValueError, NotImplementedError)):
            mg_gla.gla(q, q, v, g, cu_seqlens=jnp.array([0, T]), pipeline=pipeline)
    with pytest.raises(ValueError):
        mg_gla.gla(q, q, v, g, pipeline="other")
