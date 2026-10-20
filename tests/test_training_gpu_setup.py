# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-JAX GPU inventory, XLA tuning tiers, launch environment and capacity planning."""

from __future__ import annotations

from unittest import mock

import pytest

from jla.config import RuntimeConfig
from jla.launch_env import DETERMINISTIC_FLAG, prepare_xla_flags, xla_flag_list
from jla.training import gpu_setup
from jla.training.gpu_setup import (
    DeviceSpecs,
    GPUProfile,
    get_optimal_batch_size,
    nvidia_specs_from_inventory,
    query_free_vram_gb,
    visible_nvidia_inventory,
    xla_flags_for,
)

INVENTORY = (
    {
        "index": "0",
        "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "name": "NVIDIA H100 80GB HBM3",
        "memory.total": "81920",
        "memory.free": "80000",
        "multiprocessor_count": "132",
        "compute_cap": "9.0",
    },
    {
        "index": "1",
        "uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "name": "NVIDIA H100 80GB HBM3",
        "memory.total": "81920",
        "memory.free": "79000",
        "multiprocessor_count": "132",
        "compute_cap": "9.0",
    },
)


def _visible(value: str | None) -> list[str]:
    return [row["index"] for row in visible_nvidia_inventory(value, INVENTORY)]


def test_cuda_visibility_rules() -> None:
    assert _visible(None) == ["0", "1"] and _visible("all") == ["0", "1"]
    assert _visible("1,0") == ["1", "0"]
    assert _visible("1,not-a-device,0") == ["1"]
    assert _visible("0,-1,1") == ["0"]
    assert _visible("GPU-1111") == ["1"]
    ambiguous = ({"index": "0", "uuid": "GPU-abc111"}, {"index": "1", "uuid": "GPU-abc222"})
    assert visible_nvidia_inventory("GPU-abc", ambiguous) == ()


def test_mig_partitions_never_report_whole_gpu_capacity() -> None:
    token = "MIG-GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/1/2"
    rows = visible_nvidia_inventory(token, INVENTORY)
    assert len(rows) == 1 and rows[0]["uuid"] == token and "memory.total" not in rows[0]
    specs = nvidia_specs_from_inventory(rows, num_devices=1)
    assert specs.is_partitioned and specs.memory_gb == 0.0 and specs.bf16_tflops == 0.0


def test_specs_profiles_heterogeneity_and_a100_bandwidth() -> None:
    with mock.patch.object(gpu_setup, "_detect_nvlink", return_value=True):
        specs = nvidia_specs_from_inventory(INVENTORY, num_devices=2)
        mixed = nvidia_specs_from_inventory(
            (INVENTORY[0], {**INVENTORY[1], "name": "NVIDIA A100-SXM4-40GB", "memory.total": "40960", "compute_cap": "8.0"}),
            num_devices=2,
        )
    assert specs.profile == GPUProfile.H100 and specs.memory_gb == 80.0 and specs.sm_count == 132 and specs.has_nvlink
    assert mixed.heterogeneous and mixed.profile == GPUProfile.GENERIC_GPU and mixed.memory_gb == 40.0
    base = {"index": "0", "uuid": "GPU-a", "memory.total": "81920", "compute_cap": "8.0"}
    pcie = nvidia_specs_from_inventory(({**base, "name": "NVIDIA A100 80GB PCIe"},), num_devices=1)
    sxm = nvidia_specs_from_inventory(({**base, "name": "NVIDIA A100-SXM4-80GB"},), num_devices=1)
    assert (pcie.memory_bandwidth_gb_s, sxm.memory_bandwidth_gb_s) == (1935.0, 2039.0)


def test_inventory_query_falls_back_to_minimal_fields() -> None:
    gpu_setup.nvidia_inventory.cache_clear()
    minimal = [{"index": "0", "uuid": "GPU-x", "name": "NVIDIA A100"}]
    try:
        with mock.patch.object(gpu_setup, "_run_nvidia_query", side_effect=lambda f: minimal if f == ("index", "uuid", "name") else []) as query:
            assert gpu_setup.nvidia_inventory() == tuple(minimal)
            assert query.call_count == 4
    finally:
        gpu_setup.nvidia_inventory.cache_clear()


def test_xla_tuning_tiers() -> None:
    h100 = DeviceSpecs("H100", "gpu", GPUProfile.H100, 80.0, 8)
    assert xla_flags_for(h100, 0) == []
    assert xla_flags_for(h100, 1) == ["--xla_gpu_enable_latency_hiding_scheduler=true"]
    tier2 = xla_flags_for(h100, 2)
    assert "--xla_gpu_enable_pipelined_all_gather=true" in tier2 and "--xla_gpu_enable_cublaslt=true" in tier2
    single = xla_flags_for(DeviceSpecs("H100", "gpu", GPUProfile.H100, 80.0, 1), 2)
    assert not any("pipelined" in flag for flag in single)
    assert xla_flags_for(None, 2) == [] and xla_flags_for(h100, 0, command_buffer="") == ["--xla_gpu_enable_command_buffer="]
    with pytest.raises(ValueError, match="tuning level"):
        xla_flags_for(h100, 3)


def test_launch_merges_config_flags_without_overriding_the_launcher() -> None:
    runtime = RuntimeConfig(xla_tuning_level=1, deterministic=True, xla_flags=("--xla_gpu_autotune_level=2",))
    specs = DeviceSpecs("H100", "gpu", GPUProfile.H100, 80.0, 8)
    with mock.patch("jla.launch_env.detect_prejax_nvidia", return_value=specs) as detect:
        flags = xla_flag_list(runtime, visible_devices="0,1")
        environ = {"XLA_FLAGS": "--xla_gpu_autotune_level=4", "CUDA_VISIBLE_DEVICES": "3"}
        merged = prepare_xla_flags(runtime, environ)
    assert flags == ["--xla_gpu_enable_latency_hiding_scheduler=true", DETERMINISTIC_FLAG, "--xla_gpu_autotune_level=2"]
    assert detect.call_args_list[-1].args == ("3",)
    assert merged.split() == ["--xla_gpu_autotune_level=4", "--xla_gpu_enable_latency_hiding_scheduler=true", DETERMINISTIC_FLAG]
    assert environ["XLA_FLAGS"] == merged


def test_free_vram_is_the_minimum_over_visible_devices() -> None:
    live = [{"index": "0", "uuid": INVENTORY[0]["uuid"], "memory.free": "40960"}, {"index": "1", "uuid": INVENTORY[1]["uuid"], "memory.free": "20480"}]
    with (
        mock.patch.object(gpu_setup, "visible_nvidia_inventory", return_value=INVENTORY),
        mock.patch.object(gpu_setup, "_run_nvidia_query", return_value=live),
    ):
        assert query_free_vram_gb(None) == 20.0


def test_batch_planning_is_monotone_and_explicit_about_zero() -> None:
    specs = DeviceSpecs("H100", "gpu", GPUProfile.H100, 16.0, 1)
    common = {"seq_len": 2048, "hidden_size": 512, "num_layers": 12}
    small = get_optimal_batch_size(specs, 10.0, vocab_size=1_000, **common)
    large = get_optimal_batch_size(specs, 10.0, vocab_size=100_000, **common)
    assert large["training"] <= small["training"] and large["inference"] <= small["inference"]
    low = get_optimal_batch_size(specs, 10.0, vocab_size=32_000, usable_memory_fraction=0.5, **common)
    high = get_optimal_batch_size(specs, 10.0, vocab_size=32_000, usable_memory_fraction=0.85, **common)
    assert low["training"] <= high["training"]
    sharded = get_optimal_batch_size(specs, 5000.0, vocab_size=32_000, fsdp_shards=8, **common)
    assert get_optimal_batch_size(specs, 5000.0, vocab_size=32_000, **common) == {"training": 0, "inference": 0}
    assert sharded["inference"] > 0
    assert get_optimal_batch_size(specs, 10.0, vocab_size=1000, memory_gb=0.0, **common) == {"training": 0, "inference": 0}
    with pytest.raises(ValueError, match="unsupported optimizer"):
        get_optimal_batch_size(specs, 10.0, vocab_size=1000, optimizer="adamw_8bit", **common)
