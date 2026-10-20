# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side helpers of the Mosaic GPU scalar-gate kernels (run on any backend)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops import chunked
from jla.ops.mosaic_gpu import _common as cm
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu import scalar_gate_fused as sf


def test_chunk_log2_decay_is_the_chunk_local_cumsum_in_log2_units():
    B, T, H, pad = 2, 100, 3, 28
    g = -jax.nn.softplus(jax.random.normal(jax.random.key(0), (B, T, H)))
    got = cm.chunk_log2_decay(g, B, T, H, pad)
    assert got.shape == (B, H, T + pad) and got.dtype == jnp.float32
    padded = jnp.pad(g, [(0, 0), (0, pad), (0, 0)])
    want = chunked.chunk_local_cumsum(chunked._to_chunks(padded, cm.CHUNK))  # [B, NC, H, C]
    want = jnp.moveaxis(want, 2, 1).reshape(B, H, T + pad) / math.log(2.0)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    # Padding is inert: the decay stays at the last real token's value.
    np.testing.assert_array_equal(got[:, :, T:], jnp.broadcast_to(got[:, :, T - 1:T], (B, H, pad)))
    np.testing.assert_array_equal(cm.chunk_log2_decay(None, B, T, H, pad), jnp.zeros((B, H, T + pad)))


def test_head_major_and_pad_time():
    x = jax.random.normal(jax.random.key(1), (2, 70, 3))
    hm = cm.head_major(x.astype(jnp.bfloat16), 58)
    assert hm.shape == (2, 3, 128) and hm.dtype == jnp.float32
    np.testing.assert_array_equal(hm[:, :, :70], jnp.swapaxes(x.astype(jnp.bfloat16).astype(jnp.float32), 1, 2))
    np.testing.assert_array_equal(hm[:, :, 70:], 0.0)
    y = jnp.ones((2, 70, 3, 4))
    assert cm.pad_time(y, 0) is y
    assert cm.pad_time(y, 58).shape == (2, 128, 3, 4)


def test_key_grad_sums_value_heads_of_a_group():
    x = jax.random.normal(jax.random.key(2), (1, 130, 6, 8))
    got = sg._key_grad(x, 100, 2, jnp.float32)
    want = x[:, :100].reshape(1, 100, 2, 3, 8).sum(axis=3)
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert sg._key_grad(x, 100, 6, jnp.bfloat16).dtype == jnp.bfloat16


def test_fused_forward_picks_the_deepest_pipeline_that_fits():
    assert sf._stages(40_000, 50_000) == 4
    assert sf._stages(50_000, 50_000) == 3
    assert sf._stages(80_000, 50_000) == 2
    with pytest.raises(ValueError, match="SMEM"):
        sf._stages(100_000, 50_000)


def test_smem_tiles_are_swizzled_for_wgmma():
    tile = cm.smem((64, 128), jnp.bfloat16)
    assert tile.shape == (64, 128)
    assert any("Swizzle" in type(t).__name__ for t in tile.transforms)
    assert cm.stage((64, 128), jnp.float32).transforms == ()


def test_is_hopper_matches_the_backend():
    device = jax.devices()[0]
    expected = device.platform == "gpu" and getattr(device, "compute_capability", None) == "9.0"
    assert cm.is_hopper() == expected
