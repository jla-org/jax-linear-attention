# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Durable JSONL metrics and atomic run manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jla.training.metrics_sink import METRIC_SCHEMA, RUN_SCHEMA, MetricSink, atomic_write_json, json_safe, write_run_manifest


def test_metric_sink_appends_complete_strict_json_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricSink(path, run_id="run")
    first = sink.append("train", step=1, metrics={"loss": 2.5, "count": np.int64(3)})
    sink.append("eval", step=1, metrics={"loss": float("nan")})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0] == first and records[0]["schema"] == METRIC_SCHEMA
    assert records[0]["metrics"]["count"] == 3
    assert records[1]["metrics"]["loss"] == "nan"
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="step"):
        sink.append("train", step=-1)


def test_atomic_json_and_manifest_replace_complete_objects(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    atomic_write_json(path, {"generation": 1})
    write_run_manifest(path, {"run_id": "x", "generation": 2})
    payload = json.loads(path.read_text())
    assert payload["schema"] == RUN_SCHEMA and payload["generation"] == 2
    assert not list(tmp_path.glob(".*.tmp"))
    assert json_safe({"inf": float("inf"), "path": Path("a"), "tuple": (1, 2)}) == {"inf": "inf", "path": "a", "tuple": [1, 2]}
