# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet / DeltaNet Mosaic GPU kernels at large heads (FLA's 1.3B shape K = 256, V = 512).

End to end against ``jla.ops.reference`` (outputs, final states), every
gradient against autodiff of ``jla.ops.chunked`` (float32, HIGHEST), packed
documents, float16, grouped value heads, odd lengths, ``use_qk_l2norm``, the
dispatcher, the fused per-chunk gradients against the XLA ones, the GDN and
DeltaNet mixers against ``chunked``, and that the K, V in {64, 128} calls still
take the default path. The kernel tests need an H100.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import gated_delta_rule as gdr
from jla.ops.mosaic_gpu import scalar_gate_kb as skb

ON_HOPPER = cm.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")

F32, BF16 = jnp.float32, jnp.bfloat16
HIGHEST = lax.Precision.HIGHEST
GATES = {"deltanet": ("beta",), "gated_deltanet": ("g", "beta")}


def _inputs(seed, B=1, T=128, H=2, K=256, V=512, Hk=None):
    Hk = Hk or H
    ks = jax.random.split(jax.random.key(seed), 9)
    return dict(
        q=jax.random.normal(ks[0], (B, T, Hk, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, Hk, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H))),
        h0=0.1 * jax.random.normal(ks[5], (B, H, K, V)),
        do=jax.random.normal(ks[6], (B, T, H, V)),
        dht=jax.random.normal(ks[7], (B, H, K, V)),
    )


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _call(kind, impl, x, *, h0, scale=None, final=True, kv_heads=None, segment_ids=None):
    q, k, v = x["q"], x["k"], x["v"]
    if impl not in ("mosaic_gpu", "auto"):
        if kv_heads is not None and kv_heads != v.shape[2]:
            reps = v.shape[2] // kv_heads
            q, k = jnp.repeat(q, reps, axis=2), jnp.repeat(k, reps, axis=2)
        q, k, v = (a.astype(F32) for a in (q, k, v))
    gates = {name: x[name] for name in GATES[kind]}
    extra = dict(precision=HIGHEST) if impl == "chunked" else {}
    if segment_ids is not None:
        extra["segment_ids"] = segment_ids
    return ops.linear_attention(kind, q, k, v, impl=impl, scale=scale, initial_state=h0,
                                output_final_state=final, **gates, **extra)


E2E = [  # kind, B, T, H, K, V, Hk, with_h0
    ("gated_deltanet", 1, 200, 2, 256, 512, None, True),
    ("gated_deltanet", 2, 128, 2, 256, 256, None, False),
    ("gated_deltanet", 1, 192, 4, 256, 512, 2, True),
    ("gated_deltanet", 1, 100, 2, 256, 128, None, True),
    ("gated_deltanet", 1, 150, 2, 128, 256, None, True),
    ("deltanet", 1, 130, 2, 256, 512, None, True),
    ("deltanet", 1, 64, 2, 256, 256, None, False),
]


def _case_id(case):
    kind, B, T, H, K, V, Hk, h0 = case
    return f"{kind}-B{B}T{T}H{H}K{K}V{V}" + (f"-Hk{Hk}" if Hk else "") + ("-h0" if h0 else "")


@gpu
@pytest.mark.parametrize("case", E2E, ids=_case_id)
def test_end_to_end_matches_reference(case):
    kind, B, T, H, K, V, Hk, with_h0 = case
    x = _inputs(4, B, T, H, K, V, Hk)
    h0 = x["h0"] if with_h0 else None
    o, s = jax.jit(lambda x, h0: _call(kind, "mosaic_gpu", x, h0=h0))(x, h0)
    o_ref, s_ref = _call(kind, "reference", x, h0=h0, kv_heads=Hk)
    assert o.dtype == BF16 and s.dtype == F32
    _assert_close(o, o_ref, 1.5e-2, "o")
    _assert_close(s, s_ref, 1e-2, "final state")


@gpu
@pytest.mark.parametrize("case", E2E, ids=_case_id)
def test_gradients_match_chunked_autodiff(case):
    kind, B, T, H, K, V, Hk, with_h0 = case
    x = _inputs(5, B, T, H, K, V, Hk)
    names = ("q", "k", "v") + GATES[kind] + (("h0",) if with_h0 else ())

    def loss(impl, params):
        y = dict(x, **params)
        o, s = _call(kind, impl, y, h0=y["h0"] if with_h0 else None, kv_heads=Hk)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])

    params = {name: x[name] for name in names}
    got = jax.jit(jax.grad(lambda p: loss("mosaic_gpu", p)))(params)
    want = jax.jit(jax.grad(lambda p: loss("chunked", p)))(params)
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


def _segment_ids(B: int, T: int, pattern: str) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    if pattern == "edges":
        rows[0, 64:] = 1  # a document starting exactly at a chunk edge
        rows[0, 128:] = 2
        rows[0, 191:] = 3  # ... and one on a chunk's last token
        if B > 1:
            rows[1, 63:] = 5
            rows[1, 100:130] = 0  # an id reused after a different one starts a new document
            rows[1, 130:] = 5
    elif pattern == "tiny":  # many short documents (mean length 5)
        rng = np.random.default_rng(0)
        for b in range(B):
            starts = np.sort(rng.choice(np.arange(1, T), size=T // 5, replace=False))
            rows[b] = np.searchsorted(starts, np.arange(T), side="right")
    return jnp.asarray(rows)


@gpu
@pytest.mark.parametrize("pattern", ["edges", "tiny"])
@pytest.mark.parametrize("kind", sorted(GATES))
def test_segment_ids_match_chunked(kind, pattern):
    """Packed documents: values, final state and every gradient against chunked autodiff."""
    B, T = 2, 320
    x = _inputs(12, B, T, 2, 256, 512)
    seg = _segment_ids(B, T, pattern)
    names = ("q", "k", "v") + GATES[kind] + ("h0",)

    def loss(impl, p):
        o, s = _call(kind, impl, dict(x, **p), h0=p["h0"], segment_ids=seg)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), (o, s)

    params = {n: x[n] for n in names}
    (_, (o, s)), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(params)
    (_, (o_ref, s_ref)), want = jax.jit(jax.value_and_grad(lambda p: loss("chunked", p), has_aux=True))(params)
    _assert_close(o, o_ref, 2e-2, "o")
    _assert_close(s, s_ref, 2e-2, "final state")
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
@pytest.mark.parametrize("kind", sorted(GATES))
def test_float16_one_partial_chunk_initial_state_no_final_state(kind):
    x = _inputs(10, 1, 17, 2, 256, 512)
    x = dict(x, **{n: x[n].astype(jnp.float16) for n in ("q", "k", "v")})
    names = ("q", "k", "v", "h0") + GATES[kind]

    def loss(impl, params):
        o, s = _call(kind, impl, dict(x, **params), h0=params["h0"], final=False)
        assert s is None
        return jnp.sum(o.astype(F32) * x["do"]), o

    params = {name: x[name] for name in names}
    (_, o), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(params)
    (_, o_ref), want = jax.jit(jax.value_and_grad(lambda p: loss("chunked", p), has_aux=True))(params)
    assert o.dtype == jnp.float16
    _assert_close(o, o_ref, 1e-2, "o")
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
def test_qk_l2norm_and_scale_match_reference():
    x = _inputs(6, 1, 128, 2, 256, 512)
    q, k = x["q"] * 3, x["k"] * 3
    o, _ = gdr.gated_delta_rule(q, k, x["v"], x["g"], x["beta"], scale=0.1, use_qk_l2norm=True)
    o_ref, _ = ref.gated_delta_rule(q.astype(F32), k.astype(F32), x["v"].astype(F32), x["g"], x["beta"],
                                    scale=0.1, use_qk_l2norm=True)
    _assert_close(o, o_ref, 1.5e-2, "o")


@gpu
def test_long_sequence_state_matches_reference():
    """Many chunks through the serial passes (state drift), FLA's head shape."""
    x = _inputs(7, 1, 2048, 1, 256, 512)
    o, s = jax.jit(lambda x: _call("gated_deltanet", "mosaic_gpu", x, h0=x["h0"]))(x)
    o_ref, s_ref = _call("gated_deltanet", "reference", x, h0=x["h0"])
    _assert_close(o, o_ref, 1.5e-2, "o")
    _assert_close(s, s_ref, 1e-2, "final state")


FWD_PATHS = [  # kind, B, T, H, K, V, Hk, with_h0, segments
    ("gated_deltanet", 2, 320, 2, 256, 512, None, True, "tiny"),
    ("gated_deltanet", 1, 200, 4, 256, 256, 2, False, None),
    ("deltanet", 1, 130, 2, 128, 256, None, True, "edges"),
]


@gpu
@pytest.mark.parametrize("case", FWD_PATHS, ids=lambda c: "-".join(map(str, c)))
def test_fused_forward_matches_materialized_forward(case):
    """The fused forward kernel against the forward through HBM (states and v_new stored, output in XLA)."""
    kind, B, T, H, K, V, Hk, with_h0, pattern = case
    x = _inputs(8, B, T, H, K, V, Hk)
    seg = None if pattern is None else _segment_ids(B, T, pattern)
    h0 = x["h0"] if with_h0 else None
    g = x["g"] if kind == "gated_deltanet" else None

    def fwd(fused):
        return jax.jit(lambda x, h0: skb.run(x["q"], x["k"], x["v"], g, x["beta"], scale=None, initial_state=h0,
                                             output_final_state=True, segment_ids=seg, fused_fwd=fused))(x, h0)

    (o, s), (o_ref, s_ref) = fwd(True), fwd(False)
    _assert_close(o, o_ref, 1e-2, "o")
    _assert_close(s, s_ref, 1e-5, "final state")


@gpu
def test_per_chunk_gradients_in_batch_slices_match_one_pass(monkeypatch):
    """Large batches run the XLA per-chunk gradients (``fused_bwd=False``) in slices of whole sequences."""
    B, T = 4, 192
    x = _inputs(9, B, T, 2, 256, 512)
    seg = _segment_ids(B, T, "edges")
    names = ("q", "k", "v", "g", "beta", "h0")

    def grads():
        def loss(p):
            o, s = skb.run(p["q"], p["k"], p["v"], p["g"], p["beta"], scale=None, initial_state=p["h0"],
                           output_final_state=True, segment_ids=seg, fused_bwd=False)
            return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])
        return jax.jit(jax.grad(loss))({n: x[n] for n in names})

    assert skb._grad_rows(B, T, 2) == B
    want = grads()
    monkeypatch.setattr(skb, "GRAD_HEAD_TOKENS", 2 * T)  # one sequence per slice
    assert skb._grad_rows(B, T, 2) == 1
    got = grads()
    for name in names:  # the same algebra; only the batched matmuls' shapes differ
        _assert_close(got[name], want[name], 5e-3, f"d{name}")


@pytest.mark.parametrize("kind", sorted(GATES))
def test_supports_and_routing(kind, monkeypatch):
    monkeypatch.setattr(skb, "is_hopper", lambda: True)
    x = _inputs(0, 1, 64, 2, 256, 512)
    gates = tuple(x[n] for n in GATES[kind])
    sup = gdr.supports_gated_delta_rule if kind == "gated_deltanet" else gdr.supports_delta_rule
    assert skb.handles(x["q"], x["v"])
    assert sup(x["q"], x["k"], x["v"], *gates)
    assert sup(x["q"], x["k"], x["v"], *gates, segment_ids=jnp.zeros((1, 64), jnp.int32), use_qk_l2norm=True)
    assert not sup(x["q"], x["k"], x["v"], *gates, cu_seqlens=jnp.array([0, 64]))
    assert not sup(x["q"], x["k"], x["v"], *gates, chunk_size=32)
    v64 = x["v"][..., :64]
    assert not skb.handles(x["q"], v64)  # (256, 64) is not covered: falls through to chunked
    small = _inputs(0, 1, 64, 2, 128, 128)
    assert not skb.handles(small["q"], small["v"])  # the default pipeline keeps K, V in {64, 128}


@gpu
@pytest.mark.parametrize("kind", sorted(GATES))
def test_auto_picks_the_kernels_on_hopper(kind):
    x = _inputs(3, 1, 128, 2, 256, 512)
    o_auto, _ = _call(kind, "auto", x, h0=None)
    o_mgpu, _ = _call(kind, "mosaic_gpu", x, h0=None)
    np.testing.assert_array_equal(np.asarray(o_auto, np.float32), np.asarray(o_mgpu, np.float32))


TRAINING = [  # kind, B, T, H, K, V, Hk, dtype, segments: the layer's training call (no initial or final state)
    ("gated_deltanet", 2, 333, 2, 256, 512, None, BF16, "tiny"),
    ("deltanet", 2, 320, 2, 256, 512, None, BF16, "edges"),
    ("gated_deltanet", 1, 257, 4, 256, 256, 2, jnp.float16, "tiny"),
    ("gated_deltanet", 2, 192, 2, 128, 256, None, BF16, "edges"),
]


@gpu
@pytest.mark.parametrize("case", TRAINING, ids=lambda c: "-".join(map(str, c[:7])) + f"-{jnp.dtype(c[7]).name}-{c[8]}")
def test_training_config_segments_without_states(case):
    """segment_ids with neither an initial nor a final state: the two-stage fused forward plus the fused backward."""
    kind, B, T, H, K, V, Hk, dtype, pattern = case
    x = _inputs(13, B, T, H, K, V, Hk)
    x = dict(x, **{n: x[n].astype(dtype) for n in ("q", "k", "v")})
    seg = _segment_ids(B, T, pattern)
    names = ("q", "k", "v") + GATES[kind]

    def loss(impl, p):
        o, s = _call(kind, impl, dict(x, **p), h0=None, final=False, kv_heads=Hk, segment_ids=seg)
        assert s is None
        return jnp.sum(o.astype(F32) * x["do"]), o

    params = {n: x[n] for n in names}
    (_, o), got = jax.jit(jax.value_and_grad(lambda p: loss("mosaic_gpu", p), has_aux=True))(params)
    (_, o_ref), want = jax.jit(jax.value_and_grad(lambda p: loss("chunked", p), has_aux=True))(params)
    assert o.dtype == dtype
    _assert_close(o, o_ref, 1.5e-2, "o")
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


@gpu
@pytest.mark.parametrize("dims", [(256, 128), (128, 256)])
def test_deltanet_gradients_other_head_dims(dims):
    """DeltaNet gradients at the head pairs routed to these kernels besides (256, 512), with segments and GQA."""
    K, V = dims
    B, T, H, Hk = 1, 201, 4, 2
    x = _inputs(14, B, T, H, K, V, Hk)
    seg = _segment_ids(B, T, "tiny")
    names = ("q", "k", "v", "beta", "h0")

    def loss(impl, p):
        o, s = _call("deltanet", impl, dict(x, **p), h0=p["h0"], kv_heads=Hk, segment_ids=seg)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])

    params = {n: x[n] for n in names}
    got = jax.jit(jax.grad(lambda p: loss("mosaic_gpu", p)))(params)
    want = jax.jit(jax.grad(lambda p: loss("chunked", p)))(params)
    for name in names:
        _assert_close(got[name], want[name], 2e-2, f"d{name}")


FUSED_BWD = [  # kind, B, T, H, K, V, Hk, with_h0, segments
    ("gated_deltanet", 2, 300, 4, 256, 512, 2, True, "tiny"),
    ("gated_deltanet", 1, 192, 2, 256, 128, None, False, None),
    ("deltanet", 1, 130, 2, 128, 256, None, True, "edges"),
]


@gpu
@pytest.mark.parametrize("case", FUSED_BWD, ids=lambda c: "-".join(map(str, c)))
def test_fused_chunk_gradients_match_xla_chunk_gradients(case):
    """The fused per-chunk gradient kernel against the previous XLA per-chunk gradients (same states)."""
    kind, B, T, H, K, V, Hk, with_h0, pattern = case
    x = _inputs(15, B, T, H, K, V, Hk)
    seg = None if pattern is None else _segment_ids(B, T, pattern)
    names = ("q", "k", "v", "beta") + (("g",) if kind == "gated_deltanet" else ()) + (("h0",) if with_h0 else ())

    def grads(fused):
        def loss(p):
            o, s = skb.run(p["q"], p["k"], p["v"], p.get("g"), p["beta"], scale=None, initial_state=p.get("h0"),
                           output_final_state=True, segment_ids=seg, fused_bwd=fused)
            return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"])
        return jax.jit(jax.grad(loss))({n: x[n] for n in names})

    got, want = grads(True), grads(False)
    for name in names:  # one algebra; the kernels round and sum in another order
        _assert_close(got[name], want[name], 1e-2, f"d{name}")


@gpu
@pytest.mark.parametrize("layer_name", ["GatedDeltaNet", "DeltaNet"])
def test_layer_at_fla_1p3b_head_shape(layer_name):
    """The mixer at flame's gdn-1p3b head shape (K = 256, V = 512, short conv, q/k L2 norm, output gate) on
    packed documents: ``implementation="mosaic_gpu"`` against ``"chunked"``, output and every gradient."""
    import dataclasses

    import equinox as eqx

    from jla.layers import linear_attention as la
    from jla.models.config import LinearAttentionConfig

    d_model, B, T = 512, 1, 333
    cfg = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=256, head_v_dim=512)
    kx, kc, kp = jax.random.split(jax.random.key(16), 3)
    x = jax.random.normal(kx, (B, T, d_model)).astype(BF16)
    ct = jax.random.normal(kc, (B, T, d_model))
    seg = _segment_ids(B, T, "edges")

    def run(impl):
        layer = getattr(la, layer_name)(d_model, dataclasses.replace(cfg, implementation=impl), key=kp, std=0.02,
                                        norm_eps=1e-6)

        def loss(args):
            layer, x = args
            out = layer(x, segment_ids=seg)
            return jnp.sum(out.astype(F32) * ct), out

        (_, out), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))((layer, x))
        return out, jax.tree.leaves(eqx.filter(grads, eqx.is_array))

    out, grads = run("mosaic_gpu")
    out_ref, grads_ref = run("chunked")
    _assert_close(out, out_ref, 2e-2, "out")
    assert len(grads) == len(grads_ref)
    for i, (got, want) in enumerate(zip(grads, grads_ref)):
        _assert_close(got, want, 3e-2, f"grad {i}")
