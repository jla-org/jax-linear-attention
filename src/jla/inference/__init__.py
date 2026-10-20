# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generation: batched decoding with the model's caches, sampling, checkpoint loading (``jla generate``)."""

from jla.inference.generate import (
    GenerationResult,
    default_prefill_chunk,
    generate,
    prompts_from_padded,
    supports_cache,
)
from jla.inference.loading import dropless_moe, load_model
from jla.inference.rwkv7 import RWKV7Decoder, decoding_model
from jla.inference.sampling import GREEDY, SamplingConfig, sample_token, warp_logits

__all__ = [
    "GREEDY",
    "GenerationResult",
    "RWKV7Decoder",
    "SamplingConfig",
    "decoding_model",
    "default_prefill_chunk",
    "dropless_moe",
    "generate",
    "load_model",
    "prompts_from_padded",
    "sample_token",
    "supports_cache",
    "warp_logits",
]
