# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped simple GLA backward (``scalar_gate_grouped``): dB, dC summed over a group in the kernel.

On an H100: the kernel against the per-head ``dqkwg`` plus a float32 group sum (with and
without packed documents, both buffering modes, split groups), the whole op's gradients
against the default path and against chunked autodiff (odd lengths, initial and final
state, packed documents, several key groups), and the Mamba-2 layer switch
``SSMConfig.grouped_key_grads``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.models.config import SSMConfig
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu import scalar_gate_grouped as sgg
from jla.ops.mosaic_gpu import scalar_gate_kernels as sk

gpu = pytest.mark.skipif(not cm.is_hopper(), reason="needs an H100 (compute capability 9.0)")

F32, BF16 = jnp.float32, jnp.bfloat16
C = cm.CHUNK
HIGHEST = lax.Precision.HIGHEST


def _assert_close(got, want, tol, name=""):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def _segment_ids(B: int, T: int) -> jnp.ndarray:
    rows = np.zeros((B, T), np.int32)
    rows[0, 64:] = 1  # a document starting at a chunk edge
    rows[0, 100:] = 2  # and one inside a chunk
    if B > 1:
        rows[1, 37:] = 1
        rows[1, 38:] = 2  # a one-token document
    return jnp.asarray(rows)


def _inputs(seed, B, T, Hk, H, K, V):
    ks = jax.random.split(jax.random.key(seed), 8)
    return dict(
        q=jax.random.normal(ks[0], (B, T, Hk, K)).astype(BF16),
        k=(0.5 * jax.random.normal(ks[1], (B, T, Hk, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        h0=0.1 * jax.random.normal(ks[4], (B, H, K, V)),
        do=jax.random.normal(ks[5], (B, T, H, V)),
        dht=jax.random.normal(ks[6], (B, H, K, V)),
    )


# ------------------------------------------------------------------------- the kernel

KERNEL = [  # B, T, Hk, H, K, V, segments, splits, register_sums
    (2, 128, 1, 8, 128, 64, False, 1, False),
    (2, 128, 1, 8, 128, 64, True, 1, False),
    (2, 128, 1, 8, 128, 64, True, 1, True),
    (1, 192, 2, 8, 64, 64, True, 2, True),
    (1, 192, 2, 8, 64, 64, True, 2, False),
    (1, 128, 1, 4, 128, 128, True, 1, False),  # V = 128: one buffer per head, no prefetch
    (1, 128, 1, 4, 128, 128, True, 1, True),
    (1, 128, 2, 4, 64, 128, False, 2, False),
]


@gpu
@pytest.mark.parametrize("case", KERNEL, ids=lambda c: "B{}T{}Hk{}H{}K{}V{}-seg{}-s{}-regs{}".format(*c))
def test_kernel_matches_per_head_dqkwg_and_group_sum(case):
    B, T, Hk, H, K, V, segs, splits, register_sums = case
    ks = jax.random.split(jax.random.key(1), 8)
    q = jax.random.normal(ks[0], (B, T, Hk, K)).astype(BF16)
    k = (0.5 * jax.random.normal(ks[1], (B, T, Hk, K))).astype(BF16)
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    do = jax.random.normal(ks[3], (B, T, H, V)).astype(BF16)
    g2 = cm.chunk_log2_decay(-jax.nn.softplus(jax.random.normal(ks[4], (B, T, H)) - 2.0), B, T, H, 0)
    h = (0.3 * jax.random.normal(ks[5], (B, H, T // C, V, K))).astype(BF16)
    dh = (0.3 * jax.random.normal(ks[6], (B, H, T // C, V, K))).astype(BF16)
    seg = cm.segment_vectors(_segment_ids(B, T), 0) if segs else None
    scale = 0.7

    dq_h, dk_h, _, dg_want = jax.jit(lambda *a: sk.dqkwg(*a, None, scale=scale, vn_head_major=False, dq_dtype=F32,
                                                       dk_dtype=F32, seg=seg))(q, k, v, g2, h, dh, do)
    dq_p, dk_p, dg = jax.jit(lambda *a: sgg.dqkwg_grouped(*a, scale=scale, splits=splits, seg=seg,
                                                          register_sums=register_sums))(q, k, v, g2, h, dh, do)
    assert dq_p.shape == (B, T, Hk * splits, K) and dq_p.dtype == F32 and dg.shape == (B, H, T)

    def group_sum(x):  # per value head -> per key head, float32
        return x.reshape(B, T, Hk, H // Hk, K).sum(axis=3)

    def split_sum(x):
        return x.reshape(B, T, Hk, splits, K).sum(axis=3)

    _assert_close(split_sum(dq_p), group_sum(dq_h), 1e-5, "dq")
    _assert_close(split_sum(dk_p), group_sum(dk_h), 1e-5, "dk")
    _assert_close(dg, dg_want, 1e-5, "dg")


def test_default_splits_keeps_two_ctas_per_sm():
    assert sgg.default_splits(16, 64, 1, 64) == 1  # Mamba-2 1.3B training shape: 1024 CTAs
    s = sgg.default_splits(1, 4, 1, 64)
    assert s > 1 and 64 % s == 0
    assert sgg.default_splits(1, 1, 1, 2) == 1  # a CTA keeps at least two heads
    assert SSMConfig().grouped_key_grads is True  # on by default (benchmarks/zero_copy_layouts.md)


# ---------------------------------------------------------------------------- the op

E2E = [  # B, T, Hk, H, K, V, h0, segments, splits (None = default)
    (2, 256, 1, 8, 128, 64, True, True, None),
    (1, 200, 2, 8, 64, 64, False, True, None),  # odd T: padded to the chunk
    (1, 130, 1, 4, 128, 128, True, False, None),
    (2, 192, 1, 8, 128, 64, True, False, 4),
]


def _run(x, params, *, grouped, h0, segments, splits):
    y = dict(x, **params)
    return sg.run(y["q"], y["k"], y["v"], y["g"], None, delta=False, scale=1.0,
                  initial_state=y["h0"] if h0 else None, output_final_state=True,
                  segment_ids=segments, grouped_key_grads=grouped, key_grad_splits=splits)


@gpu
@pytest.mark.parametrize("case", E2E, ids=lambda c: "B{}T{}Hk{}H{}K{}V{}-h0{}-seg{}-s{}".format(*c))
def test_gradients_match_default_path_and_chunked_autodiff(case):
    B, T, Hk, H, K, V, with_h0, segs, splits = case
    x = _inputs(3, B, T, Hk, H, K, V)
    segments = _segment_ids(B, T) if segs else None
    names = ("q", "k", "v", "g") + (("h0",) if with_h0 else ())
    params = {n: x[n] for n in names}

    def loss(grouped, p):
        o, s = _run(x, p, grouped=grouped, h0=with_h0, segments=segments, splits=splits)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(s * x["dht"]), o

    def chunked_loss(p):
        y = dict(x, **p)
        reps = H // Hk
        q, k = (jnp.repeat(y[n].astype(F32), reps, axis=2) for n in ("q", "k"))
        o, s = ops.linear_attention("simple_gla", q, k, y["v"].astype(F32), g=y["g"], impl="chunked", scale=1.0,
                                    initial_state=y["h0"] if with_h0 else None, output_final_state=True,
                                    segment_ids=segments, precision=HIGHEST)
        return jnp.sum(o * x["do"]) + jnp.sum(s * x["dht"])

    (_, o_new), got = jax.jit(jax.value_and_grad(lambda p: loss(True, p), has_aux=True))(params)
    (_, o_old), old = jax.jit(jax.value_and_grad(lambda p: loss(False, p), has_aux=True))(params)
    want = jax.jit(jax.grad(chunked_loss))(params)
    np.testing.assert_array_equal(np.asarray(o_new, np.float32), np.asarray(o_old, np.float32))  # same forward
    for name in names:
        assert got[name].dtype == x[name].dtype, name
        # Only dq, dk change (float32 group sums instead of bf16 per-head terms); dg comes from the same
        # per-head arithmetic, dv and dh0 from the unchanged kernels.
        _assert_close(got[name], old[name], {"q": 1e-2, "k": 1e-2, "g": 1e-5}.get(name, 0.0), f"d{name} vs default")
        _assert_close(got[name], want[name], 2e-2, f"d{name} vs chunked")


@gpu
def test_grouped_key_grads_are_closer_to_float32_than_bf16_terms():
    """64 heads per group (Mamba-2 1.3B): the float32 in-kernel sum is at least as accurate as the default."""
    B, T, Hk, H, K, V = 1, 256, 1, 64, 128, 64
    x = _inputs(9, B, T, Hk, H, K, V)
    params = {n: x[n] for n in ("q", "k")}

    def loss(grouped, p):
        o, _ = _run(x, p, grouped=grouped, h0=False, segments=None, splits=None)
        return jnp.sum(o.astype(F32) * x["do"])

    def f32_loss(p):
        q, k = (jnp.repeat(p[n].astype(F32), H, axis=2) for n in ("q", "k"))
        o, _ = ops.linear_attention("simple_gla", q, k, x["v"].astype(F32), g=x["g"], impl="chunked", scale=1.0,
                                    precision=HIGHEST)
        return jnp.sum(o * x["do"])

    new = jax.jit(jax.grad(lambda p: loss(True, p)))(params)
    old = jax.jit(jax.grad(lambda p: loss(False, p)))(params)
    want = jax.jit(jax.grad(f32_loss))(params)
    for name in ("q", "k"):
        ref = np.asarray(want[name], np.float32)
        err_new = np.abs(np.asarray(new[name], np.float32) - ref).mean()
        err_old = np.abs(np.asarray(old[name], np.float32) - ref).mean()
        assert err_new <= err_old * 1.02, (name, err_new, err_old)


# ------------------------------------------------------------------------- the layer


@gpu
def test_mamba2_layer_switch_matches_default():
    from jla.layers.ssm import Mamba2

    cfg = SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1, implementation="mosaic_gpu",
                    grouped_key_grads=False)
    layer = Mamba2(256, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5)
    switched = Mamba2(256, dataclasses.replace(cfg, grouped_key_grads=True), key=jax.random.key(0), std=0.02,
                      norm_eps=1e-5)
    B, T = 2, 192
    u = jax.random.normal(jax.random.key(1), (B, T, 256)).astype(BF16)
    segments = _segment_ids(B, T)
    dy = jax.random.normal(jax.random.key(2), (B, T, 256))

    def loss(module, u):
        y = module(u, segment_ids=segments)
        return jnp.sum(y.astype(F32) * dy), y

    (_, y_old), (g_old_m, g_old_u) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(layer, u)
    (_, y_new), (g_new_m, g_new_u) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(switched, u)
    np.testing.assert_array_equal(np.asarray(y_new, np.float32), np.asarray(y_old, np.float32))
    _assert_close(g_new_u, g_old_u, 2e-2, "du")
    for path, a, b in zip(("bc_proj", "bc_conv"), (g_new_m.bc_proj.weight, g_new_m.bc_conv.weight),
                          (g_old_m.bc_proj.weight, g_old_m.bc_conv.weight)):
        _assert_close(a, b, 2e-2, f"d{path}")
