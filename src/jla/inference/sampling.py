# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Next-token selection: greedy, or sampling with temperature, top-k and top-p.

The filters run in the order of Hugging Face ``generate``: the logits are divided
by the temperature, then every token below the ``top_k``-th largest logit is
removed, then nucleus (top-p) filtering keeps the most probable tokens whose
probability mass, counted before each token, stays below ``top_p`` (so the most
probable token always survives). Removed tokens get ``-inf``; ties at a cut-off
are kept.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

__all__ = ["GREEDY", "SamplingConfig", "sample_token", "warp_logits"]

Array = jax.Array


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """``temperature == 0`` is greedy decoding (argmax; the first index wins ties).

    ``top_k = 0`` and ``top_p = 1`` switch those filters off.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0.0):
            raise ValueError(f"temperature must be finite and >= 0; got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 = off); got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1]; got {self.top_p}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


GREEDY = SamplingConfig()


def warp_logits(logits: Array, config: SamplingConfig) -> Array:
    """Temperature, top-k and top-p applied to ``logits [..., V]``; float32, ``-inf`` = removed."""
    if config.greedy:
        raise ValueError("warp_logits needs temperature > 0 (temperature 0 is greedy decoding)")
    logits = logits.astype(jnp.float32) / config.temperature
    vocab = logits.shape[-1]
    if 0 < config.top_k < vocab:
        kth = jax.lax.top_k(logits, config.top_k)[0][..., -1:]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    if config.top_p < 1.0:
        ordered = -jnp.sort(-logits, axis=-1)  # descending
        probs = jax.nn.softmax(ordered, axis=-1)
        mass_before = jnp.cumsum(probs, axis=-1) - probs
        kept = mass_before < config.top_p  # the first token always has mass_before == 0
        threshold = jnp.min(jnp.where(kept, ordered, jnp.inf), axis=-1, keepdims=True)
        logits = jnp.where(logits < threshold, -jnp.inf, logits)
    return logits


def sample_token(logits: Array, key: Array, config: SamplingConfig) -> Array:
    """One token id (int32) from ``logits [V]``; ``key`` is ignored for greedy decoding."""
    if config.greedy:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return jax.random.categorical(key, warp_logits(logits, config), axis=-1).astype(jnp.int32)
