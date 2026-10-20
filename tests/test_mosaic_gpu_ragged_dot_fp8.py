# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``jla.ops.mosaic_gpu.ragged_dot_fp8`` against a float64 per-group oracle on the same 8-bit values."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.mosaic_gpu._common import is_hopper
from jla.ops.mosaic_gpu.ragged_dot_fp8 import ragged_dot_fp8, supports_fp8

gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100")


def _oracle(lhs, rhs_t, sizes):
    x = np.asarray(lhs.astype(jnp.float32), np.float64)
    w = np.asarray(rhs_t.astype(jnp.float32), np.float64)
    out = np.zeros((x.shape[0], w.shape[1]))
    start = 0
    for g, s in enumerate(np.asarray(sizes)):
        out[start:start + s] = x[start:start + s] @ w[g].T
        start += s
    return out


def test_supports_rejects_16_bit_operands():
    lhs = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)
    assert not supports_fp8(lhs, rhs, jnp.zeros((2,), jnp.int32))


@gpu
@pytest.mark.parametrize("dtype_a,dtype_b", [(jnp.float8_e4m3fn, jnp.float8_e4m3fn),
                                             (jnp.float8_e5m2, jnp.float8_e4m3fn)])
@pytest.mark.parametrize("sizes", [[300, 0, 701, 23, 0, 1024, 0], [128] * 8, [4096]])
def test_matches_oracle(dtype_a, dtype_b, sizes):
    m = sum(sizes)
    k, n = 256, 384
    ka, kb = jax.random.split(jax.random.key(sum(sizes) + len(sizes)))
    lhs = (jax.random.normal(ka, (m, k), jnp.float32) * 2.0).astype(dtype_a)
    rhs_t = (jax.random.normal(kb, (len(sizes), n, k), jnp.float32) * 2.0).astype(dtype_b)
    sizes_arr = jnp.asarray(sizes, jnp.int32)
    out = jax.jit(ragged_dot_fp8)(lhs, rhs_t, sizes_arr)
    assert out.dtype == jnp.bfloat16 and out.shape == (m, n)
    ref = _oracle(lhs, rhs_t, sizes)
    err = np.max(np.abs(np.asarray(out, np.float64) - ref))
    assert err <= 1e-2 * np.max(np.abs(ref)), err  # bf16 output rounding of float32 sums of exact products


@gpu
def test_many_tiles_per_sm():
    """64 uneven groups over 40,000 rows: every SM runs several tiles through one pipeline."""
    rng = np.random.default_rng(0)
    sizes = rng.multinomial(40_000, rng.dirichlet(np.ones(64)))
    sizes[3] = sizes[40] = 0
    sizes[0] += 40_000 - sizes.sum()
    m, k, n = int(sizes.sum()), 512, 256
    ka, kb = jax.random.split(jax.random.key(1))
    lhs = jax.random.normal(ka, (m, k), jnp.float32).astype(jnp.float8_e4m3fn)
    rhs_t = jax.random.normal(kb, (64, n, k), jnp.float32).astype(jnp.float8_e4m3fn)
    out = jax.jit(ragged_dot_fp8)(lhs, rhs_t, jnp.asarray(sizes, jnp.int32))
    ref = _oracle(lhs, rhs_t, sizes)
    assert np.max(np.abs(np.asarray(out, np.float64) - ref)) <= 1e-2 * np.max(np.abs(ref))
