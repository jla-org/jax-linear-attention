# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Settings checked by the 1.3B XLA/JAX knob sweep (benchmarks/fp8_1p3b.md, "XLA settings at 1.3B")."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from jla.config import RuntimeConfig, load_config
from jla.launch_env import prepare_xla_flags, warn_slow_xla_flags

_CONFIGS = sorted((Path(__file__).resolve().parents[1] / "benchmarks" / "end_to_end" / "1p3b").glob("*.toml"))


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_configs_keep_the_measured_xla_defaults(path: Path) -> None:
    # No XLA flag measured faster than tier 1 with JAX 0.11.2's defaults on one H100.
    cfg = load_config(str(path), ['data.paths=["/data/fineweb_edu_3B"]'])
    assert cfg.runtime.xla_tuning_level == 1 and cfg.runtime.xla_flags == ()


def test_autotuning_off_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="jla.launch_env"):
        environ = {"XLA_FLAGS": "--xla_gpu_autotune_level=4"}
        prepare_xla_flags(RuntimeConfig(xla_flags=("--xla_gpu_autotune_level=0",)), environ)
        assert not caplog.records  # the launcher's value wins and keeps autotuning on
        environ = {}
        merged = prepare_xla_flags(RuntimeConfig(xla_flags=("--xla_gpu_autotune_level=0",)), environ)
    assert merged == "--xla_gpu_autotune_level=0"
    assert [r.levelname for r in caplog.records] == ["WARNING"] and "autotune_level=0" in caplog.records[0].getMessage()
    assert warn_slow_xla_flags("--xla_gpu_autotune_level=0 --xla_gpu_autotune_level=4") is False
    assert warn_slow_xla_flags("--xla_gpu_enable_latency_hiding_scheduler=true") is False
