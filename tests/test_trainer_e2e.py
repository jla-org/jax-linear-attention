# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end training of the ``tiny`` presets on synthetic data on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _toy_model import TOY_COMPONENTS, TOY_MODEL, experiment
from jla.training.trainer import Trainer


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tiny_model_learns_synthetic_data_and_writes_run_artifacts(tmp_path: Path) -> None:
    config = experiment(str(tmp_path / "run"), train={"eval_every": 15, "save_every": 15})
    result = Trainer(config).run()

    assert result.step == 30 and result.stop_reason == "max_steps"
    # Metric windows close at every log, eval and save boundary.
    assert [record["step"] for record in result.history] == [10, 15, 20, 30]
    losses = [record["loss"] for record in result.history]
    assert losses[-1] < losses[0] - 1.0, losses
    assert all(record["skipped_steps"] == 0 for record in result.history)
    assert all(record["tokens_per_sec"] > 0 and record["grad_norm"] > 0 for record in result.history)
    assert [entry["step"] for entry in result.eval_history] == [15, 30]
    assert result.eval_history[-1]["eval_loss"] < result.eval_history[0]["eval_loss"]
    assert result.data_state.batches_consumed == 30 * 2
    assert result.data_state.tokens_seen == 30 * 2 * 8 * 32

    run = tmp_path / "run"
    for name in ("config.toml", "run.json", "metrics.jsonl", "compile_preflight.json", "sharding_dump.json"):
        assert (run / name).is_file(), name
    events = [record["event"] for record in _records(run / "metrics.jsonl")]
    assert events.count("train") == 4 and events.count("eval") == 2 and events[-1] == "end"
    assert json.loads((run / "compile_preflight.json").read_text())["status"] == "compiled"
    assert sorted(p.name for p in (run / "checkpoints").iterdir() if p.is_dir()) == ["step_00000015", "step_00000030"]


def test_tiny_hybrid_with_mtp_and_document_masking(tmp_path: Path) -> None:
    config = experiment(
        str(tmp_path / "hybrid"),
        model="tiny-hybrid",
        train={"max_steps": 12, "log_every": 4, "label_smoothing": 0.1, "z_loss": 1e-4},
        data={"eos_id": 0, "document_masking": True, "synthetic_doc_length": 7},
    )
    result = Trainer(config).run()
    assert result.step == 12 and result.stop_reason == "max_steps"
    losses = [record["loss"] for record in result.history]
    assert np.isfinite(losses).all() and losses[-1] < losses[0], losses
    last = result.history[-1]
    assert last["mtp_loss"] > 0 and last["mtp/loss_1"] > 0 and last["mtp/loss_2"] > 0 and last["z_loss"] > 0
    # Every 7th token is EOS and its cross-document target is masked out.
    assert last["tokens"] == pytest.approx(2 * 8 * 32 * 6 / 7, rel=0.02)
    assert result.eval_history and np.isfinite(result.eval_history[-1]["eval_loss"])


def test_time_limit_stops_with_a_checkpoint(tmp_path: Path) -> None:
    config = experiment(
        str(tmp_path / "deadline"),
        train={"max_steps": 1000, "log_every": 1, "time_limit_minutes": 1e-9, "save_final": False},
    )
    result = Trainer(config).run()
    assert result.stop_reason == "time_limit" and result.step == 1
    assert (tmp_path / "deadline" / "checkpoints" / "step_00000001" / "manifest.json").is_file()


def test_required_resume_without_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no committed checkpoint"):
        Trainer(experiment(str(tmp_path / "empty"), train={"resume": "required"})).run()


def test_custom_model_builders_plug_into_the_trainer(tmp_path: Path) -> None:
    config = experiment(str(tmp_path / "toy"), model=TOY_MODEL, train={"max_steps": 4, "log_every": 2})
    result = Trainer(config, **TOY_COMPONENTS).run()
    assert result.step == 4 and result.history[-1]["hidden_rms"] > 0
