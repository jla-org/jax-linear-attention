# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Metric loggers: JSONL by default, TensorBoard event files and W&B offline runs when selected."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _toy_model import experiment
from jla.config import TrainConfig
from jla.training.loggers import MetricLoggers, metric_loggers, scalar_items
from jla.training.trainer import Trainer


def _train_records(run: Path) -> list[dict]:
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return [r for r in records if r["event"] == "train"]


def _run(tmp_path: Path, **train) -> Path:
    out = tmp_path / "run"
    Trainer(experiment(str(out), train={"max_steps": 4, "log_every": 2, "final_eval": False, **train})).run()
    return out


def test_default_logger_is_jsonl_only(tmp_path: Path) -> None:
    assert TrainConfig().loggers == ("jsonl",)
    run = _run(tmp_path)
    assert [r["step"] for r in _train_records(run)] == [2, 4]
    assert not (run / "tensorboard").exists() and not (run / "wandb").exists()


def test_tensorboard_logger_writes_an_event_file(tmp_path: Path) -> None:
    pytest.importorskip("tensorboardX")
    accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    run = _run(tmp_path, loggers=("jsonl", "tensorboard"))
    assert list((run / "tensorboard").glob("events.out.tfevents.*"))
    events = accumulator.EventAccumulator(str(run / "tensorboard"))
    events.Reload()
    loss = events.Scalars("train/loss")
    records = _train_records(run)
    assert [e.step for e in loss] == [r["step"] for r in records] == [2, 4]
    np.testing.assert_allclose([e.value for e in loss], [r["metrics"]["loss"] for r in records], rtol=1e-6)
    assert "end/seconds" in events.Tags()["scalars"]


def test_wandb_offline_logger_writes_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):  # keep the library's own files in the test dir
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    pytest.importorskip("wandb")
    run = _run(tmp_path, loggers=("jsonl", "wandb"), wandb_project="jla-test", wandb_mode="offline")
    runs = list((run / "wandb").glob("offline-run-*"))
    assert len(runs) == 1, list((run / "wandb").iterdir())
    (log,) = runs[0].glob("run-*.wandb")
    history = [row for row in _wandb_history(log) if "train/loss" in row]
    records = _train_records(run)
    assert [row["_step"] for row in history] == [r["step"] for r in records] == [2, 4]
    assert [row["train/loss"] for row in history] == pytest.approx([r["metrics"]["loss"] for r in records], rel=1e-12)
    assert not any(key.startswith("telemetry/telemetry/") for row in _wandb_history(log) for key in row)


def test_wandb_run_id_names_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    pytest.importorskip("wandb")
    train = {"loggers": ("jsonl", "wandb"), "wandb_project": "jla-test", "wandb_mode": "offline", "wandb_run_id": "jla-run-7"}
    run = _run(tmp_path, **train)
    (directory,) = (run / "wandb").glob("offline-run-*")
    assert directory.name.endswith("-jla-run-7")
    (log,) = directory.glob("run-jla-run-7.wandb")
    assert [row["_step"] for row in _wandb_history(log) if "train/loss" in row] == [2, 4]


def _wandb_history(path: Path) -> list[dict]:
    """History rows of an offline W&B transaction log (LevelDB log format: 32 KiB blocks of framed records)."""
    import struct

    from wandb.proto import wandb_internal_pb2

    data, block, header = path.read_bytes(), 32768, 7
    assert data[:4] == b":W&B"
    position, partial, rows = header, b"", []
    while position + header <= len(data):
        left = block - position % block
        if left < header:  # block trailer
            position += left
            continue
        _, length, kind = struct.unpack("<IHB", data[position : position + header])
        if length == 0 and kind == 0:  # zero padding up to the next block
            position += left
            continue
        chunk = data[position + header : position + header + length]
        position += header + length
        if kind in (2, 3):  # first / middle fragment
            partial = chunk if kind == 2 else partial + chunk
            continue
        record = wandb_internal_pb2.Record()
        record.ParseFromString(partial + chunk if kind == 4 else chunk)
        if record.WhichOneof("record_type") == "history":
            rows.append({(item.key or "/".join(item.nested_key)): json.loads(item.value_json) for item in record.history.item})
    return rows


def test_logger_config_is_validated() -> None:
    with pytest.raises(ValueError, match="needs train.wandb_project"):
        TrainConfig(loggers=("jsonl", "wandb"))
    with pytest.raises(ValueError, match="must not repeat"):
        TrainConfig(loggers=("jsonl", "jsonl"))
    assert metric_loggers(experiment("unused", train={"loggers": ()}), "unused", run_id="x") is None
    with pytest.raises(ValueError, match="at least one sink"):
        MetricLoggers([])


def test_scalar_items_flattens_numbers_and_skips_text() -> None:
    metrics = {"loss": np.float32(2.5), "ok": True, "name": "adamw", "nested": {"a": 1, "b": np.array(3.0)}, "vec": np.ones(2)}
    assert dict(scalar_items(metrics)) == {"loss": 2.5, "ok": 1.0, "nested/a": 1.0, "nested/b": 3.0}
