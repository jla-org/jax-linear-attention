# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-7 behind the decode interface that generation and evaluation use.

:class:`~jla.models.rwkv7.RWKV7LM` is a separate model class: ``__call__`` returns logits,
``init_cache(batch)`` takes no length and ``decode_step`` consumes one token per row.
:class:`RWKV7Decoder` gives it the :class:`~jla.models.LanguageModel` surface:
``__call__(...) -> LMOutput`` (``.hidden`` after the final LayerNorm), ``logits(hidden)``,
``init_cache(batch, max_len)`` and ``decode_step(cache, input_ids [B, T], positions [B, T])``.
A multi-token ``decode_step`` runs every block's full (chunked) forward pass from the cached
state, so a prompt prefill costs one forward pass, not ``T`` single-token steps. RWKV-7 has
no positions; ``positions`` are checked for shape and otherwise unused.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.models.lm import LMOutput
from jla.models.rwkv7 import RWKV7Config, RWKV7LM

__all__ = ["RWKV7Decoder", "decoding_model"]

Array = jax.Array


class RWKV7Decoder(eqx.Module):
    """:class:`RWKV7LM` with the :class:`~jla.models.LanguageModel` forward, logits and cache methods."""

    lm: RWKV7LM

    @property
    def config(self) -> RWKV7Config:
        return self.lm.config

    def _embed(self, input_ids: Array) -> Array:
        return self.lm.embed[input_ids].astype(jnp.dtype(self.lm.config.compute_dtype))

    def __call__(self, input_ids: Array, *, positions: Array | None = None, segment_ids: Array | None = None,
                 key: Array | None = None, inference: bool = False) -> LMOutput:
        """``input_ids [B, T]`` -> :class:`LMOutput`; documents restart where ``segment_ids`` change.

        Same computation as ``RWKV7LM.__call__`` without the head (RWKV-7 has no dropout).
        """
        del key, inference
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T]; got {input_ids.shape}")
        if positions is not None and positions.shape != input_ids.shape:
            raise ValueError(f"positions must be [B, T] = {input_ids.shape}; got {positions.shape}")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError(f"segment_ids must be [B, T] = {input_ids.shape}; got {segment_ids.shape}")
        x = self._embed(input_ids)
        v_first = None
        for block in self.lm.blocks:
            x, v_first, _ = block(x, v_first, segment_ids)
        return LMOutput(hidden=self.lm.ln_out(x), pre_norm_hidden=x, aux_loss=jnp.zeros((), jnp.float32),
                        metrics={})

    def logits(self, hidden: Array) -> Array:
        return self.lm.logits(hidden)

    def init_cache(self, batch: int, max_len: int):
        """Recurrent caches: ``max_len`` does not change their size."""
        del max_len
        return self.lm.init_cache(batch)

    def decode_step(self, cache, input_ids: Array, positions: Array):
        """``input_ids [B, T]`` from ``cache`` -> float32 logits ``[B, T, V]`` and the cache after the last token."""
        if input_ids.ndim != 2 or input_ids.shape != positions.shape:
            raise ValueError(f"input_ids and positions must both be [B, T]; got {input_ids.shape}, {positions.shape}")
        x = self._embed(input_ids)
        v_first = None
        new = []
        for block, layer_cache in zip(self.lm.blocks, cache):
            x, v_first, layer_cache = block(x, v_first, cache=layer_cache)
            new.append(layer_cache)
        return self.lm.logits(self.lm.ln_out(x)), tuple(new)


def decoding_model(model):
    """``model`` with the :class:`~jla.models.LanguageModel` decode interface (wraps :class:`RWKV7LM`)."""
    return RWKV7Decoder(model) if isinstance(model, RWKV7LM) else model
