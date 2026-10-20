# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Mamba-1 selective-scan Mosaic GPU kernels against the reference op (H100 only)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops

mgpu = pytest.importorskip("jla.ops.mosaic_gpu.mamba1")

pytestmark = pytest.mark.skipif(not mgpu._on_hopper(), reason="needs an sm_90 GPU")

NAMES = ("q", "k", "v", "dt", "a", "d", "z", "dt_bias", "h0")


def _inputs(seed=0, batch=2, length=100, dim=256, dstate=16, dtype=jnp.bfloat16):
    ks = jax.random.split(jax.random.key(seed), 9)
    return dict(
        q=jax.random.normal(ks[0], (batch, length, dstate), jnp.float32).astype(dtype),
        k=jax.random.normal(ks[1], (batch, length, dstate), jnp.float32).astype(dtype),
        v=jax.random.normal(ks[2], (batch, length, dim), jnp.float32).astype(dtype),
        dt=(0.5 * jax.random.normal(ks[3], (batch, length, dim), jnp.float32)).astype(dtype),
        a=-jnp.exp(jax.random.uniform(ks[4], (dim, dstate), jnp.float32, -1.0, 1.5)),
        d=jax.random.normal(ks[5], (dim,), jnp.float32),
        z=jax.random.normal(ks[6], (batch, length, dim), jnp.float32).astype(dtype),
        dt_bias=jax.random.uniform(ks[7], (dim,), jnp.float32, -3.0, -1.0),
        h0=jax.random.normal(ks[8], (batch, dim, dstate), jnp.float32),
    )


def _run(impl, x, *, segment_ids=None, final=True, softplus=True, backward=None, **drop):
    kw = {k: x[k] for k in ("d", "z", "dt_bias") if k not in drop}
    if backward is not None:
        kw["backward"] = backward
    return ops.linear_attention(
        "mamba1", x["q"], x["k"], x["v"], dt=x["dt"], a=x["a"], dt_softplus=softplus,
        initial_state=None if "h0" in drop else x["h0"], output_final_state=final,
        segment_ids=segment_ids, impl=impl, **kw)


def _close(got, want, rtol, atol):
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=rtol, atol=atol)


def _check_forward(x, out_tol, **kw):
    o, f = jax.jit(lambda x: _run("mosaic_gpu", x, **kw))(x)
    o_ref, f_ref = jax.jit(lambda x: _run("reference", x, **kw))(x)
    scale = float(jnp.max(jnp.abs(o_ref.astype(jnp.float32))))
    _close(o, o_ref, 0.0, out_tol * scale)
    if f_ref is not None:
        _close(f, f_ref, 1e-4, 1e-4 * float(jnp.max(jnp.abs(f_ref))))


@pytest.mark.parametrize("length", [100, 64, 1])
def test_forward_matches_reference(length):
    x = _inputs(length=length)
    _check_forward(x, 1e-2)  # bf16 output rounding


@pytest.mark.parametrize("dstate", [4, 8, 32])
def test_forward_other_state_sizes(dstate):
    _check_forward(_inputs(dstate=dstate, length=70), 1e-2)


def test_forward_float32_is_tight():
    _check_forward(_inputs(dtype=jnp.float32, length=90), 2e-5)


def test_forward_without_optional_inputs():
    x = _inputs(length=77)
    _check_forward(x, 1e-2, d=None, z=None, dt_bias=None, h0=None, final=False)


def test_forward_without_softplus_on_whole_tiles():
    x = _inputs(length=64, dtype=jnp.float32)
    x = dict(x, dt=0.2 * jnp.abs(x["dt"]), dt_bias=jnp.full_like(x["dt_bias"], 0.01))  # delta = dt + bias > 0
    _check_forward(x, 2e-5, softplus=False)


def test_forward_segments_match_reference():
    x = _inputs(length=150, dtype=jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 40), jnp.int32), jnp.ones((2, 50), jnp.int32),
                           jnp.full((2, 60), 2, jnp.int32)], axis=1)
    seg = seg.at[1, 10:].add(3)  # a different boundary in the second row
    _check_forward(x, 2e-5, segment_ids=seg)


def test_dispatcher_takes_the_kernels_where_the_backward_kernel_runs():
    assert ops.available("mamba1")[0] == "mosaic_gpu"
    x = _inputs(length=8, dim=128)
    for dstate, impl in ((16, "mosaic_gpu"), (32, "mosaic_gpu")):  # d_state 32: two 16-state slices
        xx = _inputs(length=40, dim=128, dstate=dstate)
        auto = jax.jit(lambda xx: _run("auto", xx))(xx)
        want = jax.jit(lambda xx, impl=impl: _run(impl, xx))(xx)
        for a, b in zip(auto, want):
            assert bool(jnp.all(a == b)), (dstate, impl)
    assert mgpu.supports(x["q"], x["k"], x["v"], x["dt"], x["a"], dt_softplus=True)
    assert not mgpu.supports(x["q"], x["k"], x["v"][..., :64], x["dt"][..., :64], x["a"][:64], dt_softplus=True)
    with pytest.raises(ValueError, match="multiple of 128"):
        _run("mosaic_gpu", dict(x, v=x["v"][..., :64], dt=x["dt"][..., :64], z=x["z"][..., :64],
                                a=x["a"][:64], d=x["d"][:64], dt_bias=x["dt_bias"][:64], h0=x["h0"][:, :64]))


def _grads(impl, x, diff, segment_ids=None, drop=(), final=True, backward=None, softplus=True):
    kw = dict({k: None for k in drop}, final=final, backward=backward, softplus=softplus)

    def loss(*vals):
        xx = dict(x, **dict(zip(diff, vals)))
        o, f = _run(impl, xx, segment_ids=segment_ids, **kw)
        w = jnp.cos(jnp.arange(o.size, dtype=jnp.float32).reshape(o.shape))
        total = jnp.sum(o.astype(jnp.float32) * w)
        if f is not None:
            total = total + jnp.sum(f * jnp.sin(jnp.arange(f.size, dtype=jnp.float32).reshape(f.shape)))
        return total

    return jax.jit(jax.grad(loss, argnums=tuple(range(len(diff)))))(*(x[k] for k in diff))


def _check_grads(x, segment_ids=None, rtol=1e-3, drop=(), final=True, backward=None):
    diff = [k for k in NAMES if k not in drop]
    got = _grads("mosaic_gpu", x, diff, segment_ids, drop, final, backward)
    want = _grads("reference", x, diff, segment_ids, drop, final)
    for name, g, w in zip(diff, got, want):
        scale = float(jnp.max(jnp.abs(w.astype(jnp.float32)))) + 1e-30
        err = float(jnp.max(jnp.abs(g.astype(jnp.float32) - w.astype(jnp.float32)))) / scale
        assert err < rtol, f"d{name}: max error {err:.2e} of max |grad| {scale:.3e}"


def test_gradients_match_reference_float32():
    _check_grads(_inputs(length=100, dtype=jnp.float32), backward="reference")


def test_gradients_match_reference_bf16():
    _check_grads(_inputs(length=96), rtol=2e-2, backward="reference")


def test_gradients_with_segments():
    x = _inputs(length=150, dtype=jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 40), jnp.int32), jnp.ones((2, 110), jnp.int32)], axis=1)
    _check_grads(x, segment_ids=seg, backward="reference")


def test_gradients_without_optional_inputs():
    # Large deltas (no bias): the input on which the chunked op's GPU gradient goes wrong. Run twice
    # to catch nondeterminism.
    x = _inputs(length=45, dtype=jnp.float32)
    for _ in range(2):
        _check_grads(x, drop=("d", "z", "dt_bias", "h0"), final=False, backward="reference")


def _assert_grads_close(names, got, want, rtol):
    for name, g, w in zip(names, got, want):
        scale = float(jnp.max(jnp.abs(w.astype(jnp.float32)))) + 1e-30
        err = float(jnp.max(jnp.abs(g.astype(jnp.float32) - w.astype(jnp.float32)))) / scale
        assert err < rtol, f"d{name}: max error {err:.2e} of max |grad| {scale:.3e}"


def test_chunked_gradients_on_the_gpu():
    """The input on which differentiating the chunked scan gave ``ddt`` off by about 7e6 times its max on
    the H100 (and ``du`` off in some runs). On a GPU ``chunked`` now takes the reference op's gradient.
    Run twice to catch nondeterminism."""
    x = _inputs(length=45, dtype=jnp.float32)
    drop = ("d", "z", "dt_bias", "h0")
    diff = [k for k in NAMES if k not in drop]
    want = _grads("reference", x, diff, None, drop, False)
    for _ in range(2):
        _assert_grads_close(diff, _grads("chunked", x, diff, None, drop, False), want, 1e-5)


@pytest.mark.parametrize("dim,dstate", [(192, 16), (192, 32)])
def test_auto_gradients_where_the_kernels_decline(dim, dstate):
    """``auto`` falls back to ``chunked`` for calls the kernels decline (D not a multiple of 128): its
    gradients must be the reference op's, with every input and segments, and on large deltas."""
    x = _inputs(length=100, dim=dim, dstate=dstate, dtype=jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 30), jnp.int32), jnp.ones((2, 70), jnp.int32)], axis=1)
    assert not mgpu.supports(x["q"], x["k"], x["v"], x["dt"], x["a"], dt_softplus=True, segment_ids=seg)
    want = _grads("reference", x, list(NAMES), seg)
    for _ in range(2):
        _assert_grads_close(NAMES, _grads("auto", x, list(NAMES), seg), want, 1e-4)
    x = _inputs(length=45, dim=dim, dstate=dstate, dtype=jnp.float32)
    drop = ("d", "z", "dt_bias", "h0")
    diff = [k for k in NAMES if k not in drop]
    want = _grads("reference", x, diff, None, drop, False)
    for _ in range(2):
        _assert_grads_close(diff, _grads("auto", x, diff, None, drop, False), want, 1e-4)


def test_kernel_backward_without_softplus():
    """``supports`` accepts ``dt_softplus=False`` when T % 32 == 0, so ``auto`` runs the backward kernel
    without softplus: all gradients against the reference, with segments and an initial state."""
    x = _inputs(length=64, dim=128, dtype=jnp.float32)
    x = dict(x, dt=jax.nn.softplus(x["dt"] - 1.0), dt_bias=0.01 * jnp.abs(x["dt_bias"]))  # positive deltas
    seg = jnp.concatenate([jnp.zeros((2, 20), jnp.int32), jnp.ones((2, 44), jnp.int32)], axis=1)
    assert mgpu.supports(x["q"], x["k"], x["v"], x["dt"], x["a"], dt_softplus=False, segment_ids=seg)
    got = _grads("auto", x, list(NAMES), seg, softplus=False)
    want = _grads("reference", x, list(NAMES), seg, softplus=False)
    _assert_grads_close(NAMES, got, want, 1e-3)


def test_layer_runs_the_kernel_and_matches_the_reference():
    import equinox as eqx

    from jla.layers.mamba1 import Mamba1
    from jla.models.config import SSMConfig

    def layer(impl):  # d_inner = 2 * 64 = 128 channels: one CTA per batch row
        cfg = SSMConfig(d_state=16, head_dim=16, chunk_size=16, implementation=impl)
        return Mamba1(64, cfg, key=jax.random.key(0), std=0.2, norm_eps=1e-5, compute_dtype="float32")

    u = jax.random.normal(jax.random.key(1), (2, 70, 64), jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 30), jnp.int32), jnp.ones((2, 40), jnp.int32)], axis=1)

    def loss(model, u):
        return jnp.sum(jnp.sin(model(u, segment_ids=seg)))

    with jax.default_matmul_precision("highest"):
        got = eqx.filter_jit(eqx.filter_value_and_grad(loss))(layer("mosaic_gpu"), u)
        want = eqx.filter_jit(eqx.filter_value_and_grad(loss))(layer("reference"), u)
    np.testing.assert_allclose(float(got[0]), float(want[0]), rtol=1e-4)
    for g, w in zip(jax.tree.leaves(eqx.filter(got[1], eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(want[1], eqx.is_inexact_array))):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-3, atol=1e-4 * float(jnp.max(jnp.abs(w))))


@pytest.mark.parametrize("dtype,length,rtol", [(jnp.float32, 100, 1e-3), (jnp.bfloat16, 96, 2e-2), (jnp.float32, 33, 1e-3)])
def test_kernel_backward_matches_reference(dtype, length, rtol):
    _check_grads(_inputs(length=length, dtype=dtype), rtol=rtol, backward="kernel")


def test_kernel_backward_with_segments():
    x = _inputs(length=150, dtype=jnp.bfloat16)
    seg = jnp.concatenate([jnp.zeros((2, 40), jnp.int32), jnp.ones((2, 110), jnp.int32)], axis=1)
    _check_grads(x, segment_ids=seg, rtol=2e-2, backward="kernel")


def test_kernel_backward_without_optional_inputs():
    x = _inputs(length=45, dtype=jnp.float32)
    _check_grads(x, drop=("d", "z", "dt_bias", "h0"), final=False, backward="kernel")


@pytest.mark.parametrize("dstate", [4, 8])
def test_kernel_backward_other_state_sizes(dstate):
    _check_grads(_inputs(length=40, dstate=dstate, dtype=jnp.bfloat16), rtol=2e-2, backward="kernel")


@pytest.mark.parametrize("dtype,rtol", [(jnp.float32, 1e-3), (jnp.bfloat16, 2e-2)])
def test_kernel_backward_32_states(dtype, rtol):
    """d_state 32 runs the backward kernel as two 16-state slices (float32 partial ddt, du, dz added in XLA):
    every gradient against the reference, with packed documents and an initial state."""
    x = _inputs(length=150, dstate=32, dtype=dtype)
    seg = jnp.concatenate([jnp.zeros((2, 40), jnp.int32), jnp.ones((2, 110), jnp.int32)], axis=1)
    _check_grads(x, segment_ids=seg, rtol=rtol, backward="kernel")


def test_kernel_backward_single_segment(monkeypatch):
    """The backward kernel without the sequence split (one segment per batch row, no pre-pass), several
    tiles, packed documents: the split is chosen by shape, so force it."""
    from jla.ops.mosaic_gpu import mamba1_bwd2

    monkeypatch.setattr(mamba1_bwd2, "num_segments", lambda cfg, num_sms=132: 1)
    x = _inputs(length=150, dtype=jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 40), jnp.int32), jnp.ones((2, 110), jnp.int32)], axis=1)
    _check_grads(x, segment_ids=seg, rtol=1e-3, backward="kernel")


def test_kernel_backward_documents_at_segment_starts():
    """Documents that start exactly at a sequence segment's first token: the pre-pass's decay and adjoint
    must cut the chain there. At this shape every 32-token tile is its own segment."""
    from jla.ops.mosaic_gpu import mamba1_bwd2

    cfg = mgpu.Config(batch=2, num_chunks=5, dim=256, dstate=16, softplus=True, has_z=True, segments=True)
    assert mamba1_bwd2.num_segments(cfg) == 5
    x = _inputs(length=160, dtype=jnp.float32)
    seg = jnp.concatenate([jnp.zeros((2, 64), jnp.int32), jnp.ones((2, 96), jnp.int32)], axis=1)
    seg = seg.at[1, 32:].add(1).at[1, 96:].add(1)  # second row: documents start at tokens 32, 64 and 96
    _check_grads(x, segment_ids=seg, rtol=1e-3, backward="kernel")


def test_num_segments_fills_the_gpu():
    from jla.ops.mosaic_gpu import mamba1_bwd2

    def cfg(batch, seq_len, dim):
        return mgpu.Config(batch=batch, num_chunks=seq_len // mgpu.CHUNK, dim=dim, dstate=16, softplus=True,
                           has_z=True, segments=False)

    assert mamba1_bwd2.num_segments(cfg(1, 4096, 8192)) == 2  # 64 CTAs: two segments fill 128 of 132 SMs
    assert mamba1_bwd2.num_segments(cfg(64, 2048, 8192)) == 1  # 4096 CTAs: no split
    assert mamba1_bwd2.num_segments(cfg(1, 32, 128)) == 1  # one tile


def test_kernel_v1_backward_matches_reference():
    """The first backward kernel (``backward="kernel_v1"``), kept for comparison."""
    _check_grads(_inputs(length=96), rtol=2e-2, backward="kernel_v1")


def test_kernel_v1_backward_rejects_32_states():
    x = _inputs(length=40, dstate=32)
    with pytest.raises(ValueError, match="first backward kernel with d_state = 32"):
        _run("mosaic_gpu", x, backward="kernel_v1")


def test_layer_conv_runs_the_short_conv_kernels():
    """A bf16 Mamba-1 layer with the default ``conv_impl="auto"`` runs its conv on the fused Mosaic GPU
    short-conv kernels: the same output and gradients as the XLA conv (both with the scan kernels)."""
    import equinox as eqx

    from jla.layers.mamba1 import Mamba1
    from jla.models.config import SSMConfig

    def layer(conv_impl):  # d_inner = 2 * 64 = 128 channels
        cfg = SSMConfig(d_state=16, head_dim=16, chunk_size=16, implementation="mosaic_gpu", conv_impl=conv_impl)
        return Mamba1(64, cfg, key=jax.random.key(0), std=0.2, norm_eps=1e-5)

    assert SSMConfig().conv_impl == "auto" and layer("auto").conv.impl == "auto"
    u = jax.random.normal(jax.random.key(1), (2, 70, 64), jnp.float32).astype(jnp.bfloat16)
    seg = jnp.concatenate([jnp.zeros((2, 30), jnp.int32), jnp.ones((2, 40), jnp.int32)], axis=1)

    def loss(model, u):
        return jnp.sum(jnp.sin(model(u, segment_ids=seg).astype(jnp.float32)))

    got = eqx.filter_jit(eqx.filter_value_and_grad(loss))(layer("mosaic_gpu"), u)
    want = eqx.filter_jit(eqx.filter_value_and_grad(loss))(layer("xla"), u)
    np.testing.assert_allclose(float(got[0]), float(want[0]), rtol=1e-2)
    for g, w in zip(jax.tree.leaves(eqx.filter(got[1], eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(want[1], eqx.is_inexact_array))):
        g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
        assert np.abs(g - w).max() <= 2e-2 * np.abs(w).max(), (g.shape, float(np.abs(g - w).max()))


def test_call_sites_of_one_configuration_share_one_lowering():
    """Two calls of one shape in a traced program lower the kernel once: the kernel calls are jitted with
    their configuration static, so the module holds one Mosaic GPU custom call in a function called twice
    (lowering takes about two minutes per configuration, so a Jamba stack must not pay it per layer)."""
    x1, x2 = _inputs(seed=1, length=64), _inputs(seed=2, length=64)

    def two(x1, x2):
        return _run("mosaic_gpu", x1)[0] + _run("mosaic_gpu", x2)[0]

    text = jax.jit(two).lower(x1, x2).as_text()
    calls = [line for line in text.splitlines() if "custom_call" in line and "mosaic_gpu" in line]
    assert len(calls) == 1, [line[:200] for line in calls]
