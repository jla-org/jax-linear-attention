# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rotary position embeddings in the Hugging Face (Llama / Qwen) half-rotation layout.

The rotated part of each head is split into halves ``(x1, x2)`` and rotated as
``(x1 cos - x2 sin, x2 cos + x1 sin)`` with frequencies
``theta ** (-2 i / rotary_dim)``. With ``rotary_fraction < 1`` only the first
``rotary_dim = rotary_fraction * head_dim`` channels rotate (partial RoPE, as
in Qwen3-Next and GLM); the rest pass through unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["apply_rope", "rotary_dim"]

Array = jax.Array


def rotary_dim(head_dim: int, rotary_fraction: float) -> int:
    """Number of rotated channels; must be even and positive."""
    dim = int(head_dim * rotary_fraction)  # truncation, as Hugging Face does
    if dim <= 0 or dim % 2 or dim > head_dim:
        raise ValueError(
            f"rotary_fraction {rotary_fraction} of head_dim {head_dim} gives {dim} rotated channels; "
            "it must be a positive even number"
        )
    return dim


def _exact_angles(positions: Array, theta: float, dim: int) -> Array:
    """``positions * inv_freq`` reduced mod 2 pi, within about 1e-3 rad for any int32 position.

    The position is split into four 8-bit digits, ``p = sum_k d_k 2^(8k)``, and each digit multiplies
    ``(2^(8k) inv_freq) mod 2 pi``, computed in float64 on the host: every product is below
    ``256 * 2 pi``, so its float32 rounding stays small at any position (one float32 product of the
    position, as below, is off by up to half an ulp of the angle: 0.06 rad at 1M tokens).
    """
    inv_freq = 1.0 / (float(theta) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))  # [half]
    p = positions.astype(jnp.int32)[:, :, None, None]
    angles = jnp.zeros(p.shape[:3] + (dim // 2,), jnp.float32)
    for k in range(4):
        digit = (p >> (8 * k)) & 255 if k < 3 else p >> 24  # the top digit keeps the sign
        factor = jnp.asarray(np.mod(inv_freq * 2.0 ** (8 * k), 2 * np.pi), jnp.float32)
        angles = angles + digit.astype(jnp.float32) * factor
    return angles


def apply_rope(x: Array, positions: Array, *, theta: float, rotary_fraction: float = 1.0, scaling=None,
               exact_angles: bool = False) -> Array:
    """Rotate ``x [B, T, H, D]`` by per-token ``positions [B, T]``.

    The rotation runs in float32 and the result keeps ``x.dtype``. ``scaling`` (a
    ``RoPEScalingConfig``) rescales the frequencies (:func:`scale_inv_freq`,
    :func:`yarn_inv_freq`); YaRN also multiplies the rotated channels by ``attention_factor``.
    By default the angles are the Hugging Face float32 product ``positions * inv_freq``, whose rounding
    grows with the position (up to 0.12 rad between neighbouring tokens at positions 2^20 to 2^21,
    0.25 rad at 2^21 to 2^22; ``benchmarks/long_context.md``). ``exact_angles=True`` reduces the angles
    mod 2 pi first (within about 1e-3 rad at any int32 position), for contexts far past ~256K tokens;
    it supports plain RoPE only (``scaling=None``).
    """
    if x.ndim != 4:
        raise ValueError(f"apply_rope expects [B, T, H, D]; got {x.shape}")
    if positions.shape != x.shape[:2]:
        raise ValueError(f"positions must be [B, T] = {x.shape[:2]}; got {positions.shape}")
    if exact_angles and scaling is not None:
        raise ValueError(f"exact_angles supports plain RoPE only, not {scaling.kind!r} RoPE scaling")
    head_dim = x.shape[-1]
    dim = rotary_dim(head_dim, rotary_fraction)
    half = dim // 2
    yarn = scaling is not None and scaling.kind == "yarn"
    if exact_angles:
        angles = _exact_angles(positions, theta, dim)  # [B, T, 1, half]
    else:
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))  # [half]
        if yarn:
            inv_freq = yarn_inv_freq(theta, dim, scaling)
        elif scaling is not None:
            inv_freq = scale_inv_freq(inv_freq, scaling)
        angles = positions.astype(jnp.float32)[:, :, None, None] * inv_freq  # [B, T, 1, half]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    if yarn and scaling.attention_factor != 1.0:
        cos, sin = cos * scaling.attention_factor, sin * scaling.attention_factor
    x32 = x.astype(jnp.float32)
    x1, x2, rest = x32[..., :half], x32[..., half:dim], x32[..., dim:]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)
    return rotated.astype(x.dtype)


# Adapted from transformers (Apache-2.0), src/transformers/modeling_rope_utils.py (_compute_llama3_parameters)
def scale_inv_freq(inv_freq: Array, scaling) -> Array:
    """Hugging Face's ``"linear"`` / ``"llama3"`` RoPE frequency scaling of ``inv_freq [half]`` (float32).

    ``scaling`` is a ``jla.models.config.RoPEScalingConfig``.
    """
    import math

    if scaling.kind == "linear":
        return inv_freq / scaling.factor
    if scaling.kind != "llama3":
        raise ValueError(f"unknown RoPE scaling kind {scaling.kind!r}")
    context = float(scaling.original_max_position_embeddings)
    low, high = scaling.low_freq_factor, scaling.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = jnp.where(wavelen > context / low, inv_freq / scaling.factor, inv_freq)
    smooth = (context / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / scaling.factor + smooth * scaled
    medium = (wavelen >= context / high) & (wavelen <= context / low)
    return jnp.where(medium, smoothed, scaled)


# Adapted from transformers (Apache-2.0), src/transformers/modeling_rope_utils.py (_compute_yarn_parameters)
def yarn_inv_freq(theta: float, dim: int, scaling) -> Array:
    """YaRN frequencies ``[dim // 2]`` (float32): ``theta ** (-2 i / dim)`` divided by ``factor`` past a
    linear ramp over the channels that complete ``beta_fast`` .. ``beta_slow`` turns within
    ``original_max_position_embeddings`` tokens, kept below it."""
    import math

    def correction_dim(rotations: float) -> float:
        ratio = scaling.original_max_position_embeddings / (rotations * 2 * math.pi)
        return dim * math.log(ratio) / (2 * math.log(theta))

    low, high = correction_dim(scaling.beta_fast), correction_dim(scaling.beta_slow)
    if scaling.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001  # as transformers: avoid a zero-width ramp
    ramp = jnp.clip((jnp.arange(dim // 2, dtype=jnp.float32) - low) / (high - low), 0.0, 1.0)
    keep = 1.0 - ramp  # share of the unscaled (extrapolated) frequency
    pos_freqs = theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    return (1.0 / (scaling.factor * pos_freqs)) * (1.0 - keep) + (1.0 / pos_freqs) * keep
