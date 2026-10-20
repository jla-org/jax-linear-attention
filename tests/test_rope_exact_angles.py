# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE at long positions: the float32 product ``positions * inv_freq`` loses ~ulp(angle) / 2 per token
(0.06 rad at 1M positions), ``exact_angles=True`` stays within ~1e-3 rad at any int32 position."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.rope import apply_rope

THETA, DIM = 10_000.0, 128


def _truth(positions: np.ndarray) -> np.ndarray:
    inv_freq = 1.0 / (THETA ** (np.arange(0, DIM, 2, dtype=np.float64) / DIM))
    return positions.astype(np.float64)[:, None] * inv_freq[None, :]


def _rotated_angles(positions: np.ndarray, exact: bool) -> np.ndarray:
    """The angle apply_rope rotates each channel pair by, read back from rotating ``(1, 0)`` pairs."""
    half = DIM // 2
    x = jnp.concatenate([jnp.ones((1, len(positions), 1, half)), jnp.zeros((1, len(positions), 1, half))], axis=-1)
    out = np.asarray(apply_rope(x, jnp.asarray(positions, jnp.int32)[None], theta=THETA, exact_angles=exact),
                     np.float64)[0, :, 0]
    return np.arctan2(out[:, half:], out[:, :half])  # (cos, sin) of each pair


def _error(positions: np.ndarray, exact: bool) -> float:
    diff = _rotated_angles(positions, exact) - _truth(positions)
    return float(np.abs(np.angle(np.exp(1j * diff))).max())


@pytest.mark.parametrize("start", [0, 2**20 - 512, 2**24 - 512, 2**31 - 1024])
def test_exact_angles_stay_exact_at_long_positions(start):
    positions = np.arange(start, start + 1024, dtype=np.int64)
    assert _error(positions, exact=True) < 2e-3


def test_default_angles_match_the_float32_product_and_drift_at_1m():
    short = np.arange(0, 1024, dtype=np.int64)
    assert _error(short, exact=False) < 2e-3  # the exact and default angles agree at short positions
    np.testing.assert_allclose(_rotated_angles(short, False), _rotated_angles(short, True), atol=2e-3)
    assert _error(np.arange(2**20 - 1024, 2**20 + 1024, dtype=np.int64), exact=False) > 0.02


@pytest.fixture
def _highest_matmuls():  # float32 parity at ~1e-5: keep GPU runs off TF32 matmuls
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.usefixtures("_highest_matmuls")
@pytest.mark.parametrize("theta,fraction", [(500_000.0, 1.0), (1_000_000.0, 0.5)], ids=["theta5e5", "theta1e6_partial"])
def test_attention_layer_exact_angles_prefill_and_decode_match_the_full_pass(theta, fraction):
    """``AttentionConfig.rope_exact_angles`` reaches the layer: at positions past 2^20 it changes the output
    (the default float32 angles are off there), and prefill plus decode steps match the full pass."""
    from jla.layers.attention import Attention
    from jla.models.config import AttentionConfig

    def layer(exact):  # a 16-slot ring cache (window 16 > 12 tokens: full causal attention) holds any position
        cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=32, rope_theta=theta, rotary_fraction=fraction,
                              rope_exact_angles=exact, implementation="xla", sliding_window=16)
        return Attention(64, cfg, key=jax.random.PRNGKey(0), std=0.3, norm_eps=1e-6, compute_dtype="float32")

    exact, default = layer(True), layer(False)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 12, 64), jnp.float32)
    pos = (2**20 + 2**19 + 7 + jnp.arange(12, dtype=jnp.int32))[None].repeat(2, axis=0)
    full = exact(x, pos)
    assert float(jnp.abs(full - default(x, pos)).max()) > 1e-3  # the flag is not silently dropped
    cache = exact.init_cache(2, 32, jnp.float32)
    out, cache = exact.step(x[:, :5], pos[:, :5], cache)
    outs = [out]
    for t in range(5, 12):
        out, cache = exact.step(x[:, t : t + 1], pos[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=2e-5, atol=2e-5)


def test_exact_angles_reject_rope_scaling():
    """Exact angles support plain RoPE only: combining them with a frequency scaling fails loudly, both in
    ``apply_rope`` and when the attention config is built."""
    from jla.models.config import AttentionConfig, RoPEScalingConfig

    scaling = RoPEScalingConfig("linear", 2.0)
    x = jnp.ones((1, 4, 1, DIM), jnp.float32)
    with pytest.raises(ValueError, match="plain RoPE only"):
        apply_rope(x, jnp.arange(4, dtype=jnp.int32)[None], theta=THETA, scaling=scaling, exact_angles=True)
    with pytest.raises(ValueError, match="plain RoPE only"):
        AttentionConfig(num_heads=2, num_kv_heads=1, head_dim=DIM, rope_exact_angles=True, rope_scaling=scaling)
