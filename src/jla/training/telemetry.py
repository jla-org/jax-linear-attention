# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side training health monitoring.

The compiled step already rejects non-finite updates. Telemetry watches the
host copies of step metrics (fetched once per logging window, never inside the
step) for the conditions that need an operator: runs of rejected steps,
non-finite losses or gradient norms, gradient-norm spikes and graph-health
warnings. A :class:`FlightRecorder` keeps the last records so a halt or crash
can dump the run's final moments.

``should_halt_distributed`` is a host collective; every process must call it
at the same logical boundary.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jla.training.metrics_sink import atomic_write_json

__all__ = ["FlightRecorder", "TelemetryCollector", "TelemetryConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    halt_on_nonfinite: bool = False  # halt at the first non-finite loss or gradient norm
    max_consecutive_skips: int = 20  # halt after this many rejected steps in a row; 0 disables
    grad_norm_warn_threshold: float = 100.0
    flight_recorder_size: int = 256
    max_warnings: int = 1024

    def __post_init__(self) -> None:
        if self.max_consecutive_skips < 0:
            raise ValueError("max_consecutive_skips must be >= 0")
        if not math.isfinite(self.grad_norm_warn_threshold) or self.grad_norm_warn_threshold <= 0:
            raise ValueError("grad_norm_warn_threshold must be finite and > 0")
        if self.flight_recorder_size < 1 or self.max_warnings < 1:
            raise ValueError("flight_recorder_size and max_warnings must be >= 1")


class FlightRecorder:
    """Bounded ring of the most recent per-step host metric records."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, step: int, metrics: Mapping[str, Any]) -> None:
        with self._lock:
            self._records.append({"step": int(step), **{str(k): v for k, v in metrics.items()}})

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def dump(self, path: str | Path, *, reason: str = "") -> Path:
        return atomic_write_json(path, {"reason": reason, "records": self.records()})


def _host_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TelemetryCollector:
    """Per-step host checks with a sticky, first-reason-wins halt request."""

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self.recorder = FlightRecorder(self.config.flight_recorder_size)
        self._lock = threading.RLock()
        self._halt_reason = ""
        self._warnings: deque[str] = deque(maxlen=self.config.max_warnings)
        self._warnings_total = 0
        self._consecutive_skips = 0
        self._skipped_total = 0

    # -- halt state ----------------------------------------------------------

    def request_halt(self, reason: str) -> None:
        reason = str(reason).strip() or "unspecified telemetry halt"
        with self._lock:
            if self._halt_reason:
                return
            self._halt_reason = reason
        logger.critical("HALT REQUESTED: %s", reason)

    def should_halt(self) -> bool:
        with self._lock:
            return bool(self._halt_reason)

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason

    def should_halt_distributed(self) -> bool:
        """True if any process requested a halt (collective across processes).

        A failing collective is itself a reason to stop, so it fails closed.
        """
        local = self.should_halt()
        if not self.config.enabled:
            return local
        import jax

        if jax.process_count() <= 1:
            return local
        from jax.experimental import multihost_utils

        try:
            flags = multihost_utils.process_allgather(np.asarray(int(local), dtype=np.uint8), tiled=False)
        except Exception as exc:  # any collective failure halts this process
            self.request_halt(f"distributed halt synchronization failed: {exc}")
            return True
        remote = bool(np.asarray(flags, dtype=np.uint8).any())
        if remote and not local:
            self.request_halt("halt requested by another process")
        return remote

    # -- warnings ------------------------------------------------------------

    def _warn(self, message: str) -> None:
        logger.warning(message)
        with self._lock:
            self._warnings.append(message)
            self._warnings_total += 1

    def get_warnings(self) -> list[str]:
        """Return and clear the buffered warnings."""
        with self._lock:
            out = list(self._warnings)
            self._warnings.clear()
        return out

    def warning_stats(self) -> dict[str, int]:
        with self._lock:
            buffered = len(self._warnings)
            return {"buffered": buffered, "total": self._warnings_total}

    # -- per-step checks -------------------------------------------------------

    def on_step(self, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Check one step's host metrics; returns ``telemetry/*`` values to log."""
        if not self.config.enabled:
            return {}
        self.recorder.record(step, metrics)
        out: dict[str, Any] = {}
        loss = _host_float(metrics.get("loss"))
        grad_norm = _host_float(metrics.get("grad_norm"))
        applied = metrics.get("step_applied")
        if loss is not None and not math.isfinite(loss):
            out["telemetry/loss_nonfinite"] = 1
            self._warn(f"step {step}: loss is {'NaN' if math.isnan(loss) else 'Inf'}")
            if self.config.halt_on_nonfinite:
                self.request_halt(f"step {step}: loss is {'NaN' if math.isnan(loss) else 'Inf'}")
        if grad_norm is not None:
            if not math.isfinite(grad_norm):
                out["telemetry/grad_norm_nonfinite"] = 1
                self._warn(f"step {step}: gradient norm is {'NaN' if math.isnan(grad_norm) else 'Inf'}")
                if self.config.halt_on_nonfinite:
                    self.request_halt(f"step {step}: gradient norm is non-finite")
            elif grad_norm > self.config.grad_norm_warn_threshold:
                self._warn(f"step {step}: gradient norm spike {grad_norm:.3g}")
        if applied is not None:
            with self._lock:
                if bool(applied):
                    self._consecutive_skips = 0
                else:
                    self._consecutive_skips += 1
                    self._skipped_total += 1
                consecutive, total = self._consecutive_skips, self._skipped_total
            out["telemetry/skipped_steps_total"] = total
            if not bool(applied):
                self._warn(f"step {step}: update rejected (skip_reason={int(metrics.get('skip_reason', 0))})")
                limit = self.config.max_consecutive_skips
                if limit and consecutive >= limit:
                    self.request_halt(f"{consecutive} consecutive rejected steps ending at step {step}")
        bubble = _host_float(metrics.get("pipeline/bubble_fraction"))  # pipelined steps (jla.distributed.pipeline)
        if bubble is not None:
            out["telemetry/pipeline_bubble_fraction"] = bubble
        return out

    def record_graph_health(self, metrics: Mapping[str, Any], warnings: list[str], *, halt: bool) -> dict[str, Any]:
        """Log graph-health counters; ``halt`` turns any warning into a halt request."""
        for warning in warnings:
            self._warn(f"graph health: {warning}")
        if warnings and halt:
            self.request_halt(f"graph health: {warnings[0]}")
        return {f"graph/{name}": value for name, value in metrics.items()} | {"graph/warning_count": len(warnings)}
