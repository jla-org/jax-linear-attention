# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Analytical FLOPs for the ModelConfig families and MFU/HFU/MBU accounting."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jla.models import get_preset, parameter_count
from jla.models.config import AttentionConfig, LinearAttentionConfig, ModelConfig, MoEConfig, MTPConfig, hybrid_pattern
from jla.training.performance import (
    DEVICE_PROFILES,
    compiled_cost_bytes_accessed,
    device_profile_for,
    estimate_flops_per_token,
    estimate_hfu_flops_per_token,
    estimate_step_bytes,
    step_performance,
)

DENSE = ModelConfig(
    vocab_size=32_000,
    d_model=768,
    n_layers=12,
    d_ff=2048,
    attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),
)


def _matmul_flops(flops) -> float:
    return flops.forward - flops.attention_core - flops.linear_attention_core


@pytest.mark.parametrize("preset", ["transformer-125m", "gdn-hybrid-125m", "kda-hybrid-125m", "moe-125m", "gla-125m"])
def test_matmul_flops_are_twice_the_active_parameters(preset: str) -> None:
    # Every active weight is one multiply-accumulate per token (the tied head is
    # used once as the output projection); norms and gates' scalars are negligible.
    config = get_preset(preset)
    flops = estimate_flops_per_token(config, 2048)
    assert _matmul_flops(flops) == pytest.approx(2.0 * parameter_count(config).active, rel=0.02)
    assert flops.training == pytest.approx(3 * flops.forward)


def test_dense_attention_core_uses_the_causal_average_context() -> None:
    seq = 2048
    flops = estimate_flops_per_token(DENSE, seq)
    assert flops.attention_core == pytest.approx(12 * 4.0 * 12 * 64 * (seq + 1) / 2)
    assert flops.linear_attention_core == 0.0


def test_sliding_window_caps_attention_core() -> None:
    windowed = dataclasses.replace(DENSE, attention=dataclasses.replace(DENSE.attention, sliding_window=256))
    full, local = estimate_flops_per_token(DENSE, 8192), estimate_flops_per_token(windowed, 8192)
    assert local.attention_projections == full.attention_projections
    assert local.attention_core < full.attention_core / 10
    exact = (256 * 257 / 2 + (8192 - 256) * 256) / 8192
    assert local.attention_core == pytest.approx(12 * 4.0 * 12 * 64 * exact)


def test_linear_attention_hybrid_is_linear_in_sequence_length() -> None:
    hybrid = dataclasses.replace(
        DENSE,
        layer_types=hybrid_pattern(12, "gated_deltanet", 4),
        linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=8, head_k_dim=64, head_v_dim=64),
    )
    short, long = estimate_flops_per_token(hybrid, 2048), estimate_flops_per_token(hybrid, 16384)
    assert short.linear_attention_core == long.linear_attention_core > 0
    assert long.attention_core > short.attention_core
    assert long.linear_attention_projections == short.linear_attention_projections


def test_moe_counts_active_experts_only() -> None:
    moe = dataclasses.replace(DENSE, moe=MoEConfig(num_experts=16, top_k=2, d_ff_expert=512, num_shared_experts=1))
    flops = estimate_flops_per_token(moe, 1024)
    assert flops.mlp == pytest.approx(12 * (2 * 768 * 16 + 3 * 6.0 * 768 * 512))
    more_experts = dataclasses.replace(moe, moe=dataclasses.replace(moe.moe, num_experts=64))
    assert estimate_flops_per_token(more_experts, 1024).mlp - flops.mlp == pytest.approx(12 * 2 * 768 * 48)


def test_mtp_heads_add_flops() -> None:
    base = estimate_flops_per_token(DENSE, 1024)
    for kind in ("deepseek", "chained", "independent"):
        with_mtp = estimate_flops_per_token(dataclasses.replace(DENSE, mtp=MTPConfig(depth=2, kind=kind)), 1024)
        assert with_mtp.mtp > 2 * base.lm_head
        assert with_mtp.forward == pytest.approx(base.forward + with_mtp.mtp)


def test_hfu_follows_the_remat_policy() -> None:
    flops = estimate_flops_per_token(DENSE, 2048)
    assert estimate_hfu_flops_per_token(flops, "none") == flops.training
    assert estimate_hfu_flops_per_token(flops, "full") == pytest.approx(flops.training + flops.blocks)
    assert flops.training < estimate_hfu_flops_per_token(flops, "dots_with_no_batch_dims") < estimate_hfu_flops_per_token(flops, "full")
    with pytest.raises(ValueError, match="unknown remat policy"):
        estimate_hfu_flops_per_token(flops, "everything")


def test_compiled_cost_bytes_use_the_aggregate_field() -> None:
    analysis = {"flops": 1.0, "bytes accessed": np.float64(1156.0), "bytes accessed0{}": 384.0, "bytes accessedout{}": 260.0}
    assert compiled_cost_bytes_accessed(analysis) == 1156.0
    assert compiled_cost_bytes_accessed([analysis, {"bytes_accessed": 4.0}]) == 1160.0
    assert compiled_cost_bytes_accessed({"cost_analysis": analysis}) == 1156.0
    assert compiled_cost_bytes_accessed({"flops": 1.0}) is None
    with pytest.raises(ValueError, match="finite"):
        compiled_cost_bytes_accessed({"bytes accessed": float("nan")})


def test_step_performance_units_and_device_profiles() -> None:
    h100 = device_profile_for("NVIDIA H100 80GB HBM3")
    assert h100 is DEVICE_PROFILES["H100"]
    assert device_profile_for("NVIDIA A100-SXM4-80GB") is DEVICE_PROFILES["A100-SXM"]
    assert device_profile_for("cpu") is None
    perf = step_performance(
        tokens_per_step=1_000_000,
        step_time=2.0,
        flops_per_token=6e9,
        hfu_flops_per_token=8e9,
        num_devices=8,
        bytes_per_step=4e12,
        bytes_per_step_source="hand_model",
        profile=h100,
        compute_dtype="bfloat16",
    )
    assert perf.tokens_per_sec == 500_000 and perf.tokens_per_sec_per_device == 62_500
    assert perf.mfu_percent == pytest.approx(100 * 500_000 * 6e9 / (989e12 * 8))
    assert perf.hfu_percent == pytest.approx(perf.mfu_percent * 8 / 6)
    assert perf.mbu_percent == pytest.approx(100 * 2e12 / (3352e9 * 8))
    unknown = step_performance(
        tokens_per_step=10, step_time=1.0, flops_per_token=1.0, hfu_flops_per_token=1.0, num_devices=1,
        bytes_per_step=1.0, bytes_per_step_source="hand_model", profile=None, compute_dtype="float32",
    )  # fmt: skip
    assert unknown.mfu_percent == 0.0 and unknown.tokens_per_sec == 10.0


def test_step_bytes_hand_model_grows_with_remat_and_accumulation() -> None:
    kwargs = {"total_params": 1000, "optimizer_state_bytes": 8000, "tokens_per_microbatch": 64}
    base = estimate_step_bytes(config=dataclasses.replace(DENSE, remat="none"), grad_accum=1, **kwargs)
    remat = estimate_step_bytes(config=DENSE, grad_accum=1, **kwargs)
    assert estimate_step_bytes(config=DENSE, grad_accum=2, **kwargs) > remat > 0 and base > 0
