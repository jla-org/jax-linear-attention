# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The analytical FLOP model must agree with the presets' real parameter counts.

Every block weight is used in exactly one matmul per token, so the projection and
MLP FLOPs must be twice the active non-embedding parameters (up to norm weights,
biases and per-head scalars, which are a fraction of a percent at real sizes).
"""

from __future__ import annotations

import pytest

from jla.models import get_preset, list_presets, parameter_count
from jla.training.performance import estimate_flops_per_token, estimate_hfu_flops_per_token

REAL_PRESETS = [name for name in list_presets() if not name.startswith("tiny")]
# Softmax-attention mixers keep no recurrent state, so they add no linear_attention_core FLOPs.
SOFTMAX_LAYER_TYPES = ("attention", "local_attention", "fox", "mla", "nsa", "dsa")


@pytest.mark.parametrize("name", REAL_PRESETS)
def test_projection_flops_match_parameter_counts(name):
    config = get_preset(name)
    if config.mtp is not None:
        pytest.skip("MTP heads are counted in their own term")
    counts = parameter_count(config)
    flops = estimate_flops_per_token(config, seq_len=2048)
    matmul = flops.attention_projections + flops.linear_attention_projections + flops.mlp
    assert matmul == pytest.approx(2 * (counts.active - counts.embedding), rel=0.02)


@pytest.mark.parametrize("name", list_presets())
def test_every_preset_has_a_flop_estimate(name):
    config = get_preset(name)
    flops = estimate_flops_per_token(config, seq_len=1024)
    assert flops.forward > 0 and flops.training == pytest.approx(3 * flops.forward)
    assert estimate_hfu_flops_per_token(flops, config.remat) >= flops.training
    has_recurrence = any(t not in SOFTMAX_LAYER_TYPES for t in config.resolved_layer_types)
    assert (flops.linear_attention_core > 0) == has_recurrence
