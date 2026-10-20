# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO release checks: layer chunking, the kernels' default pipeline depth, the interpreter.

* Chunking (``jla.layers.mamba3_mimo.mimo_chunk_size``): state-spaces/mamba recommends
  ``chunk_size = 64 / R`` for MIMO (64 fused token-rank rows), the tile of the Mosaic GPU MIMO kernels.
  Before the cap, an ``SSMConfig`` with the SISO default ``chunk_size=64`` sent every rank to the
  chunked XLA path under ``implementation="auto"`` (27-34x slower). The FLOPs accounting follows it.
* Gradients at the pipeline depth that layers get (2 when ``B * H > 64``) against reference autodiff.
* ``interpret=True``, which ``supports`` accepts on any platform, against the reference on CPU.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import mamba3_mimo as mm
from jla.models import get_preset
from jla.models.config import SSMConfig
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import mamba3 as m3
from jla.ops.mosaic_gpu import mamba3_mimo_kernels as mk
from jla.training.performance import estimate_flops_per_token

gpu = pytest.mark.skipif(not m3._on_hopper(), reason="needs an H100 (compute capability 9.0)")
F32, BF16 = jnp.float32, jnp.bfloat16


@pytest.mark.parametrize("rank, chunk, want", [(2, 64, 32), (4, 64, 16), (8, 64, 8), (2, 16, 16), (4, 16, 16),
                                               (3, 64, 21), (4, 32, 16)])
def test_chunk_is_capped_at_64_over_rank(rank, chunk, want):
    assert mm.mimo_chunk_size(chunk, rank) == want


def test_layer_passes_the_capped_chunk(monkeypatch):
    seen = []
    real = mm.ops.linear_attention

    def spy(kind, *args, **kwargs):
        seen.append((kind, kwargs.get("chunk_size")))
        return real(kind, *args, **kwargs)

    monkeypatch.setattr(mm.ops, "linear_attention", spy)
    cfg = SSMConfig(d_state=16, head_dim=16, mimo_rank=4, implementation="chunked")  # chunk_size 64
    u = jax.random.normal(jax.random.key(1), (1, 40, 64))
    out = mm.Mamba3MIMO(64, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5)(u)
    assert seen == [("mamba3_mimo", 16)]
    explicit = mm.Mamba3MIMO(64, dataclasses.replace(cfg, chunk_size=16), key=jax.random.key(0), std=0.02,
                             norm_eps=1e-5)(u)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(explicit))


def test_flops_use_the_capped_chunk():
    preset = get_preset("mamba3-mimo-130m")  # chunk_size 16 = 64 / 4
    wide = dataclasses.replace(preset, ssm=dataclasses.replace(preset.ssm, chunk_size=64))
    assert estimate_flops_per_token(wide, seq_len=2048) == estimate_flops_per_token(preset, seq_len=2048)


@gpu
@pytest.mark.parametrize("rank", [2, 4])
def test_default_chunk_reaches_the_kernels_under_auto(rank):
    cfg = SSMConfig(mimo_rank=rank)  # d_state 128, head_dim 64, chunk_size 64, implementation "auto"
    u = jax.random.normal(jax.random.key(2), (2, 100, 128)).astype(BF16)

    def run(implementation):
        layer = mm.Mamba3MIMO(128, dataclasses.replace(cfg, implementation=implementation), key=jax.random.key(0),
                              std=0.02, norm_eps=1e-5)
        return np.asarray(jax.jit(lambda m, x: m(x))(layer, u), np.float32)

    auto, kernels, chunked = run("auto"), run("mosaic_gpu"), run("chunked")
    np.testing.assert_array_equal(auto, kernels)  # auto picked the Mosaic GPU kernels
    assert np.abs(kernels - chunked).max() <= 2e-2 * np.abs(chunked).max()


def _inputs(seed, B, T, H, R, K, V):
    ks = jax.random.split(jax.random.key(seed), 9)
    q, k = ((jax.random.normal(x, (B, T, H, R, K)) / math.sqrt(R)).astype(BF16) for x in ks[:2])
    v = jax.random.normal(ks[2], (B, T, H, R, V)).astype(BF16)
    dt = jnp.exp(jax.random.uniform(ks[3], (B, T, H), minval=math.log(1e-3), maxval=math.log(0.1)))
    g = -jax.random.uniform(ks[4], (B, T, H), minval=1.0, maxval=16.0) * dt
    lam = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H)))
    state = ref.Mamba3State(0.1 * jax.random.normal(ks[6], (B, H, K, V)),
                            jax.random.normal(ks[7], (B, H, R, K)) / math.sqrt(R),
                            jax.random.normal(ks[8], (B, H, R, V)))
    seg = np.zeros((B, T), np.int32)
    seg[0, 37:] = 1
    seg[-1, T // 2:] = 2
    return (q, k, v, g, (1.0 - lam) * dt * jnp.exp(g), lam * dt), state, jnp.asarray(seg)


def _check_grads(fn_ours, args, state, seg, tol):
    do = jax.random.normal(jax.random.key(9), args[2].shape, F32)

    def loss(fn, *xs):
        *xs, st = xs
        o, _ = fn(*xs, initial_state=st, segment_ids=seg)
        return jnp.sum(o.astype(F32) * do)

    names = ("q", "k", "v", "g", "beta", "gamma", "state.ssm", "state.key", "state.value")
    want = jax.grad(lambda *x: loss(ref.mamba3_mimo, *x), argnums=tuple(range(7)))(
        *(a.astype(F32) for a in args), state)
    got = jax.jit(jax.grad(lambda *x: loss(fn_ours, *x), argnums=tuple(range(7))))(*args, state)
    for name, g, w in zip(names, jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
        assert np.isfinite(g).all(), name
        assert np.abs(g - w).max() <= tol * max(np.abs(w).max(), 1e-6), (name, np.abs(g - w).max(), np.abs(w).max())


@gpu
@pytest.mark.parametrize("rank, dims", [(4, (128, 64)), (2, (64, 128))])
def test_gradients_at_the_default_depth_of_wide_batches(rank, dims):
    """``B * H = 72 > 64`` selects 2 pipeline stages, the depth of real layers (the other tests use 4)."""
    args, state, seg = _inputs(3, 2, 150, 36, rank, *dims)
    _check_grads(mk.mamba3_mimo, args, state, seg, 3e-2)


def test_interpreter_matches_reference():
    """``supports(interpret=True)`` is True on any platform, so the interpreter path is tested here."""
    args, state, seg = _inputs(4, 1, 20, 1, 4, 64, 64)
    assert mk.supports(*args, initial_state=state, segment_ids=seg, interpret=True)

    def ours(*xs, **kw):
        return mk.mamba3_mimo(*xs, interpret=True, **kw)

    o, final = jax.jit(lambda *x: ours(*x, initial_state=state, segment_ids=seg, output_final_state=True))(*args)
    want, want_final = ref.mamba3_mimo(*(a.astype(F32) for a in args), initial_state=state, segment_ids=seg,
                                       output_final_state=True)
    scale = np.abs(np.asarray(want)).max()
    assert np.abs(np.asarray(o, np.float32) - np.asarray(want)).max() <= 2e-2 * scale
    for g, w in zip(final, want_final):
        assert np.abs(np.asarray(g) - np.asarray(w)).max() <= 2e-2 * max(np.abs(np.asarray(w)).max(), 1e-6)
    _check_grads(ours, args, state, seg, 3e-2)
