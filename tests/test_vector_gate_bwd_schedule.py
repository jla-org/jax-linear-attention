# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The backward kernels of the vector-gate family (KDA, GDN-2) against their reference schedule, bit for bit.

``vector_gate_bwd`` schedules ``bwd_intra`` with one copy of the slab code for both warpgroups (the
warpgroup index is traced) and every direction's four tensor-core products issued before one wait,
and ``bwd_wy`` waits for each pair of its inputs only before their first use. With
``_REFERENCE_SCHEDULE`` the kernels run the plain schedule (one code copy per warpgroup, one
wait per product, every input awaited before the first product): the outputs must be the same
bits. Kernel-level ``bwd_intra`` on random inputs with extreme decays and magnitudes, and the
backward of the op on the forward's residuals (dense, packed documents, odd ``T``, ``K, V`` in
{64, 128}, initial state, huge decays). Needs an H100. The default run keeps one whole-op case
(``docs``); the others are ``slow`` tests (``pytest --run-slow``).
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import BF16, F32, dense_segments, inputs, segments

from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_bwd as vgb
from jla.ops.mosaic_gpu import vector_gate_ops as vgo
from jla.ops.reference import segment_resets

gpu = pytest.mark.skipif(not vgo.is_hopper(), reason="needs an H100 (compute capability 9.0)")


@contextlib.contextmanager
def reference_schedule():
    old = vgb._REFERENCE_SCHEDULE
    vgb._REFERENCE_SCHEDULE = True
    try:
        yield
    finally:
        vgb._REFERENCE_SCHEDULE = old


def both_schedules(fn, *args):
    """``fn(*args)`` with the default and with the reference schedule (a fresh trace each)."""
    new = jax.jit(lambda *a: fn(*a))(*args)
    with reference_schedule():
        old = jax.jit(lambda *a: fn(*a))(*args)
    return new, old


def assert_same_bits(new, old) -> None:
    new_leaves, old_leaves = jax.tree.leaves(new), jax.tree.leaves(old)
    assert len(new_leaves) == len(old_leaves)
    for i, (a, b) in enumerate(zip(new_leaves, old_leaves)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, (i, a.dtype, b.dtype, a.shape, b.shape)
        assert np.isfinite(a.astype(np.float32)).all(), f"output {i}: non-finite values"
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=f"output {i}")


def _kernel_inputs(kind, B, T, H, K, V, *, seed, scale_m, gate_lo):
    cfg = vg.Config(batch=B, seq_len=T, heads=H, key_dim=K, value_dim=V, scale=K**-0.5, channel_gates=kind == "gdn2")
    ks = jax.random.split(jax.random.key(seed), 12)
    q = jax.random.normal(ks[0], (B, T, H, K)).astype(BF16)
    k = jax.random.normal(ks[1], (B, T, H, K)).astype(BF16)
    g = gate_lo * jax.random.uniform(ks[2], (B, T, H, K))
    g = jnp.where(jax.random.bernoulli(ks[3], 0.05, g.shape), -60.0, g)
    g_cum, marks, _, _ = jax.jit(lambda x: vg.chunk_cumsum(cfg, x))(g)
    if kind == "kda":
        erase = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, T)))
        gate_part = jax.random.normal(ks[5], (B, T, H, vgb._parts(cfg)))
    else:
        erase = jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H, K))).astype(BF16)
        gate_part = jax.random.normal(ks[5], (B, T, H, K))
    daqk = scale_m * jax.random.normal(ks[6], (B, H, T, vg.CHUNK))
    dakk = scale_m * jax.random.normal(ks[7], (B, H, T, vg.CHUNK))
    parts = [jax.random.normal(ks[8 + i], (B, T, H, K)) for i in range(3)]
    return cfg, (q, k, g_cum, marks, erase, daqk, dakk, *parts, gate_part)


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize(
    "dims,scale_m,gate_lo",
    [((128, 128), 1.0, -1.0), ((64, 64), 1.0, -1.0), ((64, 128), 1.0, -1.0), ((128, 64), 1.0, -1.0),
     ((128, 128), 1e4, -20.0)],
    ids=["k128v128", "k64v64", "k64v128", "k128v64", "k128v128_extreme"],
)
def test_bwd_intra_bits_match_reference_schedule(kind, dims, scale_m, gate_lo):
    cfg, args = _kernel_inputs(kind, 2, 192, 2, *dims, seed=dims[0] + dims[1], scale_m=scale_m, gate_lo=gate_lo)
    assert_same_bits(*both_schedules(lambda *a: vgb.bwd_intra(cfg, *a), *args))


def _prepared(kind, x, seg, h0):
    """The op's padded kernel inputs and config (as ``chunked_vector_gate`` builds them)."""
    q, k, v, g = x["q"], x["k"], x["v"], x["g"].astype(F32)
    B, T, H, K = q.shape
    pad = (-T) % vg.CHUNK

    def padded(a):
        return jnp.pad(a, [(0, 0), (0, pad)] + [(0, 0)] * (a.ndim - 2)) if pad else a

    if seg is not None:
        g = jnp.where(segment_resets(seg)[:, :, None, None], g - vgo._RESET_DECAY, g)
    if kind == "kda":
        gates = (padded(x["gates"][0].astype(F32)),)
    else:
        gates = tuple(padded(a.astype(BF16)) for a in x["gates"])
    cfg = vg.Config(batch=B, seq_len=T + pad, heads=H, key_dim=K, value_dim=v.shape[-1], scale=K**-0.5,
                    channel_gates=kind == "gdn2")
    state = x["h0"].astype(F32) if h0 else None
    cot = (padded(x["do"].astype(q.dtype)), x["dht"].astype(F32))
    return cfg, (padded(q), padded(k), padded(v), padded(g), gates, state), cot


_slow = pytest.mark.slow  # each whole-op case compiles the backward twice; pass --run-slow to run them all


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize(
    "case",
    [pytest.param(dict(B=2, T=320, H=2, K=128, V=128, seed=4, h0=True, docs="random"), id="docs"),
     pytest.param(dict(B=2, T=256, H=4, K=128, V=128), id="dense", marks=_slow),
     pytest.param(dict(B=1, T=130, H=2, K=64, V=64, seed=1), id="odd_k64", marks=_slow),
     pytest.param(dict(B=1, T=200, H=2, K=128, V=64, seed=2, h0=True), id="k128v64_state", marks=_slow),
     pytest.param(dict(B=1, T=190, H=2, K=64, V=128, seed=3, h0=True), id="k64v128_state", marks=_slow),
     pytest.param(dict(B=1, T=256, H=2, K=128, V=128, seed=5, docs="dense"), id="dense_docs", marks=_slow),
     pytest.param(dict(B=2, T=192, H=2, K=128, V=128, seed=6, h0=True, huge=True), id="huge", marks=_slow)],
)
def test_backward_bits_match_reference_schedule(kind, case):
    case = dict(case)
    seed, h0, docs, huge = case.pop("seed", 0), case.pop("h0", False), case.pop("docs", None), case.pop("huge", False)
    x = inputs(seed, kind, **case, huge=huge)
    seg = {None: None, "random": lambda: segments(seed, case["B"], case["T"]),
           "dense": lambda: dense_segments(case["B"], case["T"])}[docs]
    cfg, args, cot = _prepared(kind, x, None if seg is None else seg(), h0)
    _, residuals = jax.jit(lambda *a: vgo._forward(cfg, *a))(*args)  # one forward: the same residuals
    new, old = both_schedules(lambda r, ct: vgo._backward(cfg, r, ct), residuals, cot)
    assert_same_bits(new, old)
