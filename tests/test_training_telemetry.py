# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side telemetry: halts, warnings, flight recorder, distributed halt."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from jla.training.telemetry import FlightRecorder, TelemetryCollector, TelemetryConfig


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="max_consecutive_skips"):
        TelemetryConfig(max_consecutive_skips=-1)
    with pytest.raises(ValueError, match="grad_norm_warn_threshold"):
        TelemetryConfig(grad_norm_warn_threshold=math.nan)


def test_halt_is_sticky_and_first_reason_wins() -> None:
    collector = TelemetryCollector()
    collector.request_halt("first")
    collector.request_halt("second")
    assert collector.should_halt() and collector.halt_reason == "first"


def test_consecutive_rejected_steps_trigger_a_halt() -> None:
    collector = TelemetryCollector(TelemetryConfig(max_consecutive_skips=3))
    for step in (1, 2):
        collector.on_step(step, {"loss": math.nan, "grad_norm": math.nan, "step_applied": False, "skip_reason": 9})
    collector.on_step(3, {"loss": 1.0, "grad_norm": 1.0, "step_applied": True, "skip_reason": 0})
    assert not collector.should_halt()
    out = {}
    for step in (4, 5, 6):
        out = collector.on_step(step, {"loss": math.nan, "grad_norm": 1.0, "step_applied": False, "skip_reason": 1})
    assert collector.should_halt() and "3 consecutive rejected steps" in collector.halt_reason
    assert out["telemetry/skipped_steps_total"] == 5 and out["telemetry/loss_nonfinite"] == 1


def test_nonfinite_values_warn_and_optionally_halt() -> None:
    collector = TelemetryCollector(TelemetryConfig(halt_on_nonfinite=True))
    out = collector.on_step(1, {"loss": math.nan, "grad_norm": math.inf})
    assert out == {"telemetry/loss_nonfinite": 1, "telemetry/grad_norm_nonfinite": 1}
    assert collector.halt_reason == "step 1: loss is NaN"
    assert len(collector.get_warnings()) == 2 and collector.get_warnings() == []


def test_warning_buffer_is_bounded() -> None:
    collector = TelemetryCollector(TelemetryConfig(max_warnings=2, grad_norm_warn_threshold=10.0))
    for step in range(1, 4):
        collector.on_step(step, {"loss": 1.0, "grad_norm": 11.0})
    assert collector.warning_stats() == {"buffered": 2, "total": 3}


def test_flight_recorder_keeps_the_latest_records(tmp_path: Path) -> None:
    recorder = FlightRecorder(2)
    for step in range(3):
        recorder.record(step, {"loss": float(step)})
    assert [record["step"] for record in recorder.records()] == [1, 2]
    payload = json.loads(recorder.dump(tmp_path / "flight.json", reason="test").read_text())
    assert payload["reason"] == "test" and len(payload["records"]) == 2


def test_distributed_halt_uses_process_allgather(monkeypatch: pytest.MonkeyPatch) -> None:
    from jax.experimental import multihost_utils

    calls: list[np.ndarray] = []

    def fake_allgather(value: Any, *, tiled: bool = False) -> np.ndarray:
        calls.append(np.asarray(value))
        return np.asarray([np.asarray(value).item(), 1], np.uint8)

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "process_allgather", fake_allgather)
    collector = TelemetryCollector()
    assert collector.should_halt_distributed()
    assert calls[0].dtype == np.uint8 and calls[0].item() == 0
    assert collector.halt_reason == "halt requested by another process"


def test_distributed_halt_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from jax.experimental import multihost_utils

    def broken(value: Any, *, tiled: bool = False) -> np.ndarray:
        raise RuntimeError("collective unavailable")

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "process_allgather", broken)
    collector = TelemetryCollector()
    assert collector.should_halt_distributed()
    assert "synchronization failed" in collector.halt_reason


def test_disabled_collector_skips_collectives_and_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax, "process_count", lambda: (_ for _ in ()).throw(AssertionError("no collectives")))
    collector = TelemetryCollector(TelemetryConfig(enabled=False))
    assert not collector.should_halt_distributed()
    assert collector.on_step(1, {"loss": math.nan}) == {}


def test_graph_health_warnings_can_halt() -> None:
    collector = TelemetryCollector()
    out = collector.record_graph_health({"memcpy_d2d_count": 5}, ["MemcpyD2D storm"], halt=True)
    assert out == {"graph/memcpy_d2d_count": 5, "graph/warning_count": 1}
    assert collector.halt_reason == "graph health: MemcpyD2D storm"
