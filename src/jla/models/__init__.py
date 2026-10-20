# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model zoo: dense transformers, linear-attention hybrids, MoE and MTP.

    config = get_preset("gdn-hybrid-125m")
    model = build_model(config, key=jax.random.key(0))
    out = model(input_ids, segment_ids=segment_ids)        # LMOutput
"""

from jla.models.axes import LOGICAL_AXIS_NAMES, ParameterCount, logical_axes, parameter_count
from jla.models.config import (
    AttentionConfig,
    LinearAttentionConfig,
    ModelConfig,
    MoEConfig,
    MTPConfig,
    SSMConfig,
    hybrid_pattern,
)
from jla.models.lm import Cache, LanguageModel, LMOutput, build_model, remat_policy
from jla.models.presets import get_preset, list_presets

__all__ = [
    "AttentionConfig",
    "Cache",
    "LMOutput",
    "LOGICAL_AXIS_NAMES",
    "LanguageModel",
    "LinearAttentionConfig",
    "MTPConfig",
    "ModelConfig",
    "MoEConfig",
    "ParameterCount",
    "SSMConfig",
    "build_model",
    "get_preset",
    "hybrid_pattern",
    "list_presets",
    "logical_axes",
    "parameter_count",
    "remat_policy",
]
