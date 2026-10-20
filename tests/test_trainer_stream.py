# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Training on a text stream: resume from the checkpointed stream position equals the uninterrupted run."""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from _toy_model import experiment
from jla.config import load_config
from jla.training.trainer import Trainer

N_STEPS, SPLIT = 8, 3
WORDS = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white"]


def _corpus(tmp_path: Path) -> tuple[str, ...]:
    rng = np.random.default_rng(0)
    paths = []
    for name, count in (("a", 30), ("b", 21)):
        lines = [json.dumps({"text": " ".join(rng.choice(WORDS, size=int(rng.integers(2, 20))))}) for _ in range(count)]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(lines) + "\n")
        paths.append(str(path))
    return tuple(paths)


def _experiment(out: Path, files: tuple[str, ...], max_steps: int, packing: str):
    data = {"source": "stream", "stream_files": files, "tokenizer": "byte", "eos_id": 256, "document_masking": True, "packing": packing}
    train = {"max_steps": max_steps, "log_every": 1, "final_eval": False, "save_every": 0}
    return experiment(str(out), train=train, optim={"schedule_steps": N_STEPS}, data=data)


def _host(tree) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


@pytest.mark.parametrize("packing", ["concat_split", "first_fit"])
def test_stream_resume_matches_the_uninterrupted_run(tmp_path: Path, packing: str) -> None:
    files = _corpus(tmp_path)
    full = Trainer(_experiment(tmp_path / "full", files, N_STEPS, packing)).run()
    first = Trainer(_experiment(tmp_path / "split", files, SPLIT, packing)).run()
    sidecar = json.loads((tmp_path / "split" / "checkpoints" / f"step_{SPLIT:08d}" / "data_state.json").read_text())
    assert sidecar["data_state"]["stream"][0]["batches"] == SPLIT * 2
    second = Trainer(_experiment(tmp_path / "split", files, N_STEPS, packing)).run()
    assert second.step == N_STEPS

    same = np.testing.assert_array_equal if jax.default_backend() == "cpu" else (
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5))
    for expected, actual in zip(_host(full.model), _host(second.model), strict=True):
        same(expected, actual)
    assert full.data_state == second.data_state  # including every shard's stream position
    assert full.data_state.stream[0]["batches"] == N_STEPS * 2
    assert full.data_state.stream[0]["cursor"]["epoch"] >= 1  # the run crossed an epoch
    same(np.array([r["loss"] for r in full.history]), np.array([r["loss"] for r in first.history + second.history]))
    losses = [r["loss"] for r in full.history]
    assert np.all(np.isfinite(losses))


def test_stream_training_evaluates_on_a_cidx_corpus(tmp_path: Path) -> None:
    from jla.data.make_data import tokenize_jsonl

    files = _corpus(tmp_path)
    tokenize_jsonl([files[1]], tmp_path / "eval", encode=lambda texts: [list(t.encode("utf-8")) for t in texts], eos_id=256)
    config = _experiment(tmp_path / "run", files, 4, "first_fit")
    config = config.replace(
        train=config.train.__class__(**{**config.train.__dict__, "final_eval": True, "eval_batches": 1, "seq_len": 16}),
        data=config.data.__class__(**{**config.data.__dict__, "eval_path": str(tmp_path / "eval")}),
    )
    result = Trainer(config).run()
    assert [entry["step"] for entry in result.eval_history] == [4]
    assert np.isfinite(result.eval_history[0]["eval_loss"]) and result.eval_history[0]["eval_tokens"] > 0


def test_example_stream_config_loads() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "tiny-stream.toml")
    assert config.data.source == "stream" and config.data.packing == "first_fit"
    assert config.train.loggers == ("jsonl",)  # runs without the optional logger packages


def test_stream_config_round_trips_through_toml(tmp_path: Path) -> None:
    config = _experiment(tmp_path / "run", _corpus(tmp_path), N_STEPS, "first_fit")
    path = config.save(tmp_path / "config.toml")
    assert load_config(path) == config
    with pytest.raises(ValueError, match="needs data.eval_path"):
        config.replace(train=config.train.__class__(**{**config.train.__dict__, "final_eval": True}))
