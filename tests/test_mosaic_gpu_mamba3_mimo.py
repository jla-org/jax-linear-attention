# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO on the SISO Mosaic GPU kernels (``jla.ops.mosaic_gpu.mamba3_mimo``) against the reference.

Needs an H100; the oracle is ``jla.ops.reference.mamba3_mimo`` (itself checked against an
independent float64 loop in tests/test_mamba3_mimo.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import reference
from jla.ops.mosaic_gpu import mamba3 as m3
from jla.ops.reference import Mamba3State

ON_HOPPER = m3._on_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")


def _inputs(seed, batch=2, length=150, heads=2, rank=2, key_dim=64, value_dim=64):
    ks = jax.random.split(jax.random.key(seed), 9)
    bf16 = jnp.bfloat16
    q = (jax.random.normal(ks[0], (batch, length, heads, rank, key_dim)) * 0.15).astype(bf16)
    k = (jax.random.normal(ks[1], (batch, length, heads, rank, key_dim)) * 0.15).astype(bf16)
    v = jax.random.normal(ks[2], (batch, length, heads, rank, value_dim)).astype(bf16)
    g = -jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads)) - 2.0)
    beta = jax.random.uniform(ks[4], (batch, length, heads)) * 0.5
    gamma = jax.random.uniform(ks[5], (batch, length, heads)) * 0.5
    state = Mamba3State(
        jax.random.normal(ks[6], (batch, heads, key_dim, value_dim)) * 0.1,
        jax.random.normal(ks[7], (batch, heads, rank, key_dim)) * 0.15,
        jax.random.normal(ks[8], (batch, heads, rank, value_dim)),
    )
    return q, k, v, g, beta, gamma, state


@gpu
@pytest.mark.parametrize(("rank", "dims", "segmented", "with_state"), [
    (2, (64, 64), False, False), (2, (128, 64), True, True), (4, (64, 128), True, False), (3, (64, 64), False, True),
])
def test_mimo_on_siso_kernels_matches_reference(rank, dims, segmented, with_state):
    q, k, v, g, beta, gamma, state = _inputs(0, rank=rank, key_dim=dims[0], value_dim=dims[1])
    seg = None
    if segmented:
        ids = np.zeros(q.shape[:2], np.int32)
        ids[0, 40:] = 1
        ids[1, 100:] = 1
        seg = jnp.asarray(ids)
    state = state if with_state else None
    common = dict(g=g, beta=beta, gamma=gamma, initial_state=state, output_final_state=True, segment_ids=seg)
    o, f = ops.linear_attention("mamba3_mimo", q, k, v, impl="mosaic_gpu", chunk_size=64, kernel="siso", **common)
    o_ref, f_ref = ops.linear_attention("mamba3_mimo", *(x.astype(jnp.float32) for x in (q, k, v)),
                                        impl="reference", **common)
    scale = float(jnp.max(jnp.abs(o_ref)))
    np.testing.assert_allclose(np.asarray(o, np.float32), o_ref, atol=2e-2 * scale, rtol=0)
    np.testing.assert_allclose(f.ssm, f_ref.ssm, atol=2e-2 * float(jnp.max(jnp.abs(f_ref.ssm))), rtol=0)
    np.testing.assert_allclose(f.key, f_ref.key, atol=1e-6)
    np.testing.assert_allclose(f.value, f_ref.value, atol=1e-6)


@gpu
def test_mimo_kernel_stays_out_of_auto_and_refuses_unsupported_calls():
    q, k, v, g, beta, gamma, _ = _inputs(1, length=64)
    from jla.ops.mosaic_gpu import mamba3_mimo as mk

    assert mk.supports(q, k, v, g, beta, gamma, kernel="siso")
    assert not mk.supports(q.astype(jnp.float32), k, v, g, beta, gamma, kernel="siso")  # bf16 only
    assert not mk.supports(q, k, v, g, beta, gamma, chunk_size=32, kernel="siso")
    assert not mk.supports(q[:, :, :, 0], k[:, :, :, 0], v[:, :, :, 0], g, beta, gamma, kernel="siso")
    # The registered default is the dedicated kernel (tests/test_mosaic_gpu_mamba3_mimo_kernels.py), whose
    # chunks are 64 / R tokens: this route's 64-token chunks need kernel="siso" by name.
    assert not ops._REGISTRY["mosaic_gpu"]["mamba3_mimo"][1](q, k, v, g, beta, gamma, chunk_size=64)


def test_reference_kind_is_registered():
    assert "reference" in ops.available("mamba3_mimo") and "chunked" in ops.available("mamba3_mimo")
    assert reference.mamba3_mimo is not None
