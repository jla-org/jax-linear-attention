# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU GEMMs (jla.ops.mosaic_gpu.fused_gemm) against the XLA SwiGLU block and a float32 oracle.

Kernel outputs (``h``, the saved pre-activations, the dgrad epilogue's ``dgu``), the MLP's value
and all three gradients, enough tiles that the persistent loop wraps around the SMs, both grid
orders, float16, the strict shape checks, the model-level switch (``mlp_kernel``) and the XLA
fallback. Kernel tests need an H100; the others run anywhere.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.fused_mlp import FusedSwiGLU, apply_fused_mlp, fused_mlp_modules
from jla.layers.mlp import SwiGLU
from jla.ops.mosaic_gpu import fused_gemm as fg
from jla.ops.mosaic_gpu._common import is_hopper

ON_HOPPER = is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
# The block-level calls refuse to run when several devices are visible (no partitioning rule).
one_gpu = pytest.mark.skipif(not ON_HOPPER or jax.device_count() != 1, reason="needs exactly one visible H100")


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _params(d, f, seed=0, dtype=jnp.float32):
    ks = jax.random.split(jax.random.key(seed), 3)
    gate_up = (jax.random.normal(ks[0], (d, 2, f), jnp.float32) / np.sqrt(d)).astype(dtype)
    down = (jax.random.normal(ks[1], (f, d), jnp.float32) / np.sqrt(f)).astype(dtype)
    return gate_up, down, ks[2]


def _oracle(x, gate_up, down, dtype):
    """float32 math on the compute-dtype-rounded operands."""
    xr = x.astype(dtype).astype(jnp.float32)
    gu = jnp.einsum("md,dcf->mcf", xr, gate_up.astype(dtype).astype(jnp.float32))
    h = jax.nn.silu(gu[:, 0]) * gu[:, 1]
    return h, gu, h @ down.astype(dtype).astype(jnp.float32)


def _xla_mlp(x, gate_up, down, dtype):
    h = jnp.einsum("...d,dcf->...cf", x.astype(dtype), gate_up.astype(dtype))
    gate = h[..., 0, :].astype(jnp.float32)
    up = h[..., 1, :].astype(jnp.float32)
    a = (jax.nn.silu(gate) * up).astype(dtype)
    return jax.lax.dot_general(a, down.astype(dtype), (((a.ndim - 1,), (0,)), ((), ())))


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def test_interleave_roundtrip():
    w = jnp.arange(3 * 2 * 128, dtype=jnp.float32).reshape(3, 2, 128)
    il = fg.interleave(w)
    assert il.shape == (3, 256)
    np.testing.assert_array_equal(il[:, :64], w[:, 0, :64])  # block 0: gate features 0..63
    np.testing.assert_array_equal(il[:, 64:128], w[:, 1, :64])  # then the up features 0..63
    np.testing.assert_array_equal(il[:, 128:192], w[:, 0, 64:])
    np.testing.assert_array_equal(fg.deinterleave(il), w)
    with pytest.raises(ValueError):
        fg.interleave(jnp.zeros((3, 2, 96)))


@pytest.mark.parametrize("m,d,f,dtype,needle", [
    (100, 128, 256, jnp.bfloat16, "multiple of 128"),
    (128, 96, 256, jnp.bfloat16, "multiple of 64"),
    (128, 128, 384, jnp.bfloat16, "multiple of 256"),
    (128, 128, 256, jnp.float32, "bfloat16 or float16"),
])
def test_unsupported_shapes(m, d, f, dtype, needle):
    reason = fg.unsupported_reason(m, d, f, dtype)
    assert reason is not None and needle in reason
    gate_up, down, _ = _params(d, f)
    with pytest.raises(ValueError, match="fused SwiGLU"):
        fg.swiglu_mlp(jnp.zeros((m, d)), gate_up, down, compute_dtype=dtype)


def test_fallback_matches_swiglu_exactly():
    """Calls the kernels do not take (here: odd token counts, or no H100) run SwiGLU's XLA path."""
    mlp = SwiGLU(64, 96, key=jax.random.key(0), std=0.05)
    fused = apply_fused_mlp(mlp)
    assert isinstance(fused, FusedSwiGLU) and len(fused_mlp_modules(fused)) == 1
    x = jax.random.normal(jax.random.key(1), (3, 5, 64))
    assert not fused.uses_kernels(x.shape)
    np.testing.assert_array_equal(np.asarray(fused(x), np.float32), np.asarray(mlp(x), np.float32))


def test_config_switch_validation():
    from jla.models import get_preset

    cfg = get_preset("tiny")
    assert cfg.mlp_kernel == "xla"
    with pytest.raises(ValueError, match="mlp_kernel"):
        dataclasses.replace(cfg, mlp_kernel="triton")
    with pytest.raises(ValueError, match="fp8"):
        dataclasses.replace(cfg, mlp_kernel="mosaic_gpu", fp8="current")


@gpu
@pytest.mark.parametrize("m,d,f,cfg", [
    (256, 128, 256, fg.FWD_CONFIG),
    (1024, 512, 1536, fg.FWD_CONFIG),  # 8 x 12 = 96 tiles
    (4096, 256, 2304, fg.GemmConfig(steps=3, grid_minor=1, grid_width=4)),  # 32 x 18 tiles: wraps the SMs
    (2048, 512, 1024, fg.GemmConfig(steps=4, grid_width=16)),
    (1024, 512, 1536, fg.GemmConfig(steps=4, wg_dim=0)),  # warpgroups split the rows (64 x 256 each)
    (4096, 256, 2304, fg.GemmConfig(steps=3, grid_minor=1, grid_width=4, wg_dim=0)),
])
def test_gate_up_kernel(m, d, f, cfg):
    gate_up, down, kx = _params(d, f)
    x = jax.random.normal(kx, (m, d), jnp.float32)
    h_ref, gu_ref, _ = _oracle(x, gate_up, down, jnp.bfloat16)
    w_il = fg.interleave(gate_up.astype(jnp.bfloat16))
    h, gu = fg.swiglu_gate_up(x.astype(jnp.bfloat16), w_il, save_gu=True, cfg=cfg)
    h_only = fg.swiglu_gate_up(x.astype(jnp.bfloat16), w_il, save_gu=False, cfg=cfg)
    assert h.shape == (m, f) and gu.shape == (m, 2 * f) and h.dtype == jnp.bfloat16
    assert _rel(fg.deinterleave(gu), gu_ref) < 1e-2
    assert _rel(h, h_ref) < 1e-2
    np.testing.assert_array_equal(np.asarray(h_only, np.float32), np.asarray(h, np.float32))


@gpu
@pytest.mark.parametrize("m,d,f,cfg", [
    (256, 128, 256, fg.BWD_CONFIG),
    (4096, 256, 2304, fg.GemmConfig(steps=2, grid_minor=1, grid_width=4)),
    (2048, 512, 1024, fg.GemmConfig(steps=3, grid_width=16)),
    (1024, 512, 1536, fg.GemmConfig(steps=3, wg_dim=0)),
    (4096, 256, 2304, fg.GemmConfig(steps=2, grid_minor=1, grid_width=4, wg_dim=0)),
])
def test_down_dgrad_kernel(m, d, f, cfg):
    gate_up, down, kx = _params(d, f)
    kd, kg = jax.random.split(kx)
    dy = jax.random.normal(kd, (m, d), jnp.float32).astype(jnp.bfloat16)
    gu = (2.0 * jax.random.normal(kg, (m, 2, f), jnp.float32)).astype(jnp.bfloat16)
    g, u = gu[:, 0].astype(jnp.float32), gu[:, 1].astype(jnp.float32)
    dh = dy.astype(jnp.float32) @ down.astype(jnp.bfloat16).astype(jnp.float32).T
    sig = jax.nn.sigmoid(g)
    dg_ref = dh * u * sig * (1 + g * (1 - sig))
    du_ref = dh * g * sig
    dgu = fg.swiglu_down_dgrad(dy, down.astype(jnp.bfloat16), _interleave_rows(gu), cfg=cfg)
    got = fg.deinterleave(dgu)
    assert _rel(got[:, 0], dg_ref) < 1e-2
    assert _rel(got[:, 1], du_ref) < 1e-2


def _interleave_rows(gu):
    """``[M, 2, F]`` activations -> ``[M, 2F]`` in the kernels' interleaved column order."""
    m, _, f = gu.shape
    return gu.reshape(m, 2, f // fg.PAIR, fg.PAIR).transpose(0, 2, 1, 3).reshape(m, 2 * f)


@one_gpu
@pytest.mark.parametrize("dtype,recompute,remat", [
    (jnp.bfloat16, False, False), (jnp.float16, False, False),
    (jnp.bfloat16, True, False), (jnp.bfloat16, True, True), (jnp.bfloat16, False, True),
])
def test_mlp_value_and_gradients(dtype, recompute, remat):
    m, d, f = 512, 256, 768
    gate_up, down, kx = _params(d, f, seed=3)
    x = jax.random.normal(kx, (2, m // 2, d), jnp.float32)
    ct = jax.random.normal(jax.random.key(7), (2, m // 2, d), jnp.float32).astype(dtype)

    def loss(fn):
        return lambda x, gu, dn: jnp.sum(fn(x, gu, dn).astype(jnp.float32) * ct.astype(jnp.float32))

    def fused(x, gu, dn):
        return fg.swiglu_mlp(x, gu, dn, compute_dtype=dtype, recompute=recompute)

    if remat:  # inside a fully rematerialized function, as in a model with remat = "full"
        fused = jax.checkpoint(fused, policy=jax.checkpoint_policies.nothing_saveable)
    xla = lambda x, gu, dn: _xla_mlp(x, gu, dn, dtype)  # noqa: E731

    def oracle(x, gu, dn):
        return _oracle(x.reshape(m, d), gu, dn, dtype)[2].reshape(2, m // 2, d)

    y = jax.jit(fused)(x, gate_up, down)
    assert y.shape == x.shape and y.dtype == jnp.dtype(dtype)
    y_ref = oracle(x, gate_up, down)
    assert _rel(y, y_ref) < 1e-2
    assert _rel(jax.jit(xla)(x, gate_up, down), y_ref) < 1e-2
    g_fused = jax.jit(jax.grad(loss(fused), argnums=(0, 1, 2)))(x, gate_up, down)
    g_xla = jax.jit(jax.grad(loss(xla), argnums=(0, 1, 2)))(x, gate_up, down)
    g_ref = jax.jit(jax.grad(loss(oracle), argnums=(0, 1, 2)))(x, gate_up, down)
    for name, a, b, r in zip(("dx", "dgate_up", "ddown"), g_fused, g_xla, g_ref):
        assert a.shape == r.shape and a.dtype == r.dtype, name
        err_fused, err_xla = _rel(a, r), _rel(b, r)
        assert err_fused < 2e-2, (name, err_fused)
        assert err_fused <= 1.5 * err_xla + 2e-3, (name, err_fused, err_xla)  # no worse than the XLA path


@one_gpu
@pytest.mark.parametrize("remat", ["full", "none"])
def test_model_switch_matches_xla_model(remat):
    from jla.models import build_model, get_preset

    base = dataclasses.replace(get_preset("tiny"), d_model=128, d_ff=256, scan_layers=True, remat=remat)
    fused_cfg = dataclasses.replace(base, mlp_kernel="mosaic_gpu")
    model = build_model(base, key=jax.random.key(0))
    fused = build_model(fused_cfg, key=jax.random.key(0))
    modules = fused_mlp_modules(fused)
    assert modules and all(mod.recompute == (remat == "full") for mod in modules)
    ids = jax.random.randint(jax.random.key(1), (2, 128), 0, base.vocab_size)

    def loss(m):
        out = m(ids)
        return jnp.mean(jnp.square(m.logits(out.hidden)))

    l_ref, g_ref = eqx_value_and_grad(loss, model)
    l_fused, g_fused = eqx_value_and_grad(loss, fused)
    np.testing.assert_allclose(float(l_fused), float(l_ref), rtol=2e-2)
    ref_leaves = jax.tree_util.tree_leaves(g_ref)
    fused_leaves = jax.tree_util.tree_leaves(g_fused)
    assert len(ref_leaves) == len(fused_leaves)
    for a, b in zip(fused_leaves, ref_leaves):
        assert a.shape == b.shape
        assert _rel(a, b) < 5e-2


def eqx_value_and_grad(fn, model):
    import equinox as eqx

    return eqx.filter_jit(eqx.filter_value_and_grad(fn))(model)
