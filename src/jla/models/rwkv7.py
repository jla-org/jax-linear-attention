# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/models/rwkv7/modeling_rwkv7.py and
# fla/models/rwkv7/configuration_rwkv7.py.
"""RWKV-7 (Goose) language model: embedding, ``n_layers`` :class:`~jla.layers.rwkv7.RWKV7Block`,
a final LayerNorm and an untied head.

The first block's value ``v_first`` feeds every later block (value residual), so the stack is
run as a Python loop over blocks. :meth:`RWKV7LM.decode_step` consumes one token per call with
:class:`~jla.layers.rwkv7.RWKV7State` caches. Presets (``RWKV7_PRESETS``): ``tiny-rwkv7-blinkdl``
(tests) and ``rwkv7-0.1b-blinkdl`` (BlinkDL's RWKV-7 0.1B: L12-D768, head size 64, vocabulary
65536, LoRA ranks 64/64/32/128).

This module keeps BlinkDL's exact layout (LayerNorms, ``ln0``, untied head). The trainable
RWKV-7 models are the ``LanguageModel`` presets ``tiny-rwkv7`` and ``rwkv7-0.1b`` in
``jla.models.presets`` (``get_preset("rwkv7-0.1b")``; layer type ``"rwkv7"``, mlp type
``"rwkv7_cmix"``), which ``jla train`` uses: the same time and channel mixes in RMSNorm
pre-norm blocks, with the tied 50,304-token vocabulary.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jla.layers.linear import truncated_normal
from jla.layers.rwkv7 import LayerNorm, RWKV7Block, RWKV7State, rwkv7_lora_dims

__all__ = ["RWKV7_PRESETS", "RWKV7Config", "RWKV7LM", "rwkv7_flops_per_token", "rwkv7_param_count"]

Array = jax.Array


@dataclasses.dataclass(frozen=True)
class RWKV7Config:
    vocab_size: int
    d_model: int
    n_layers: int
    head_dim: int = 64
    d_ff: int | None = None  # None = 4 * d_model
    lora_dims: tuple[int, int, int, int] | None = None  # (decay, a, v, gate); None = FLA defaults
    norm_eps: float = 1e-5
    chunk_size: int = 16
    impl: str = "auto"
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    initializer_range: float = 0.02
    # "upstream" (the default; FLA / BlinkDL: zero o_proj, channel-mix value and LoRA downs, orthogonal LoRA
    # ups) or "jla" (jla's original initialisation); see jla.models.config.ModelConfig.init_scheme.
    init_scheme: str = "upstream"

    def __post_init__(self) -> None:
        if self.init_scheme not in ("jla", "upstream"):
            raise ValueError(f"unknown init_scheme {self.init_scheme!r}; expected 'jla' or 'upstream'")
        if self.d_model % self.head_dim:
            raise ValueError(f"d_model {self.d_model} must be divisible by head_dim {self.head_dim}")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")

    @property
    def resolved_d_ff(self) -> int:
        return self.d_ff or 4 * self.d_model

    @property
    def resolved_lora_dims(self) -> tuple[int, int, int, int]:
        return self.lora_dims or rwkv7_lora_dims(self.d_model, self.head_dim)


RWKV7_PRESETS: dict[str, RWKV7Config] = {
    "tiny-rwkv7-blinkdl": RWKV7Config(vocab_size=256, d_model=64, n_layers=2, head_dim=16, lora_dims=(16, 16, 16, 16)),
    "rwkv7-0.1b-blinkdl": RWKV7Config(vocab_size=65536, d_model=768, n_layers=12, head_dim=64,
                                      lora_dims=(64, 64, 32, 128)),
}


class RWKV7LM(eqx.Module):
    embed: Array  # [V, D]
    blocks: tuple[RWKV7Block, ...]
    ln_out: LayerNorm
    head: Array  # [V, D]
    config: RWKV7Config = eqx.field(static=True)

    def __init__(self, config: RWKV7Config, *, key: Array):
        k_embed, k_blocks, k_head = jax.random.split(key, 3)
        dtype = jnp.dtype(config.param_dtype)
        self.embed = truncated_normal(k_embed, (config.vocab_size, config.d_model), config.initializer_range, dtype)
        keys = jax.random.split(k_blocks, config.n_layers)
        self.blocks = tuple(
            RWKV7Block(config.d_model, config.resolved_d_ff, head_dim=config.head_dim, layer_idx=i,
                       n_layers=config.n_layers, key=keys[i], lora_dims=config.resolved_lora_dims,
                       norm_eps=config.norm_eps, chunk_size=config.chunk_size, impl=config.impl,
                       param_dtype=dtype, compute_dtype=config.compute_dtype, init_scheme=config.init_scheme)
            for i in range(config.n_layers)
        )
        self.ln_out = LayerNorm(config.d_model, eps=config.norm_eps, param_dtype=dtype)
        self.head = truncated_normal(k_head, (config.vocab_size, config.d_model), config.d_model**-0.5, dtype)
        self.config = config

    def logits(self, hidden: Array) -> Array:
        dtype = jnp.dtype(self.config.compute_dtype)
        return lax.dot_general(hidden.astype(dtype), self.head.astype(dtype),
                               (((hidden.ndim - 1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    def __call__(self, input_ids: Array, *, segment_ids: Array | None = None) -> Array:
        """``input_ids [B, T]`` -> float32 logits ``[B, T, V]``; documents restart at ``segment_ids`` changes."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T]; got {input_ids.shape}")
        if segment_ids is not None and segment_ids.shape != input_ids.shape:
            raise ValueError(f"segment_ids must be [B, T] = {input_ids.shape}; got {segment_ids.shape}")
        x = self.embed[input_ids].astype(jnp.dtype(self.config.compute_dtype))
        v_first = None
        for block in self.blocks:
            x, v_first, _ = block(x, v_first, segment_ids)
        return self.logits(self.ln_out(x))

    def init_cache(self, batch: int) -> tuple[RWKV7State, ...]:
        dtype = jnp.dtype(self.config.compute_dtype)
        return tuple(block.init_cache(batch, dtype) for block in self.blocks)

    def decode_step(self, cache: tuple[RWKV7State, ...], input_ids: Array) -> tuple[Array, tuple[RWKV7State, ...]]:
        """One token per row: ``input_ids [B]`` -> float32 logits ``[B, V]`` and the new caches."""
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be [B]; got {input_ids.shape}")
        x = self.embed[input_ids].astype(jnp.dtype(self.config.compute_dtype))
        v_first = None
        new = []
        for block, layer_cache in zip(self.blocks, cache):
            x, v_first, layer_cache = block.step(x, v_first, layer_cache)
            new.append(layer_cache)
        return self.logits(self.ln_out(x)), tuple(new)


def rwkv7_param_count(config: RWKV7Config) -> int:
    """Exact parameter count of :class:`RWKV7LM` (matches ``jax.tree_util`` leaves)."""
    d, f, n, v = config.d_model, config.resolved_d_ff, config.n_layers, config.vocab_size
    rw, ra, rv, rg = config.resolved_lora_dims
    heads = d // config.head_dim
    per_layer = (
        4 * d * d  # r, k, v, o projections
        + (d * rw + rw * d + d) + (d * ra + ra * d + d) + (d * rg + rg * d)  # w, a, g LoRAs
        + 6 * d + 2 * d + heads * config.head_dim + 2 * d  # mixes, k_k, k_a, r_k, group norm
        + d + 2 * d * f  # channel mix
        + 4 * d  # ln1, ln2
    )
    v_lora = (n - 1) * (d * rv + rv * d + d)
    return n * per_layer + v_lora + 2 * d + 2 * v * d + 2 * d  # + ln0, embed, head, ln_out


def rwkv7_flops_per_token(config: RWKV7Config) -> float:
    """Training FLOPs per token (forward + backward = 3x forward).

    Matmuls: ``2`` FLOPs per weight per token (the embedding gather is free). The DPLR
    recurrence costs ``8 K V`` per head per token forward: the low-rank read ``a^T S``, the
    rank-1 writes ``b (a^T S)`` and ``k v^T`` and the output ``q^T S``.
    """
    d, f, n, v = config.d_model, config.resolved_d_ff, config.n_layers, config.vocab_size
    rw, ra, rv, rg = config.resolved_lora_dims
    heads = d // config.head_dim
    matmul = n * (4 * d * d + 2 * d * (rw + ra + rg) + 2 * d * f) + (n - 1) * 2 * d * rv + v * d
    recurrence = n * heads * 8.0 * config.head_dim * config.head_dim
    return 3.0 * (2.0 * matmul + recurrence)
