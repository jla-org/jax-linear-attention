# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``train.dispatch_ahead``: the next step is dispatched before a log step's metrics are read (CPU)."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from _toy_model import experiment
from jla.training.trainer import Trainer

_KEYS = ("loss", "grad_norm", "skipped_steps")


def _run(tmp_path: Path, name: str, **train):
    config = experiment(str(tmp_path / name), train={"max_steps": 9, "log_every": 1, "eval_every": 4,
                                                     "save_every": 6, **train})
    return Trainer(config).run()


def test_dispatch_ahead_runs_the_same_steps(tmp_path: Path) -> None:
    # Log every step, with eval and save steps (which do not dispatch ahead) in between: the same history,
    # evaluations, data position and final parameters, bit for bit.
    base = _run(tmp_path, "base")
    ahead = _run(tmp_path, "ahead", dispatch_ahead=True)
    assert ahead.step == base.step == 9 and ahead.stop_reason == base.stop_reason == "max_steps"
    assert [r["step"] for r in ahead.history] == [r["step"] for r in base.history] == list(range(1, 10))
    for got, want in zip(ahead.history, base.history):
        assert {k: got[k] for k in _KEYS} == {k: want[k] for k in _KEYS}
        assert got["step_time"] > 0 and got["tokens_per_sec"] > 0
    assert [e["step"] for e in ahead.eval_history] == [e["step"] for e in base.eval_history]
    assert [e["eval_loss"] for e in ahead.eval_history] == [e["eval_loss"] for e in base.eval_history]
    assert ahead.data_state.batches_consumed == base.data_state.batches_consumed
    leaves_a = jax.tree.leaves(eqx.filter(ahead.model, eqx.is_array))
    leaves_b = jax.tree.leaves(eqx.filter(base.model, eqx.is_array))
    assert len(leaves_a) == len(leaves_b)
    # Bit for bit on CPU. On a GPU two plain runs already differ in one array by ~1.5e-8 (an atomic
    # scatter-add in a gradient), and dispatching ahead stays within that.
    atol = 0.0 if jax.default_backend() == "cpu" else 1e-7
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=atol)
    for name in ("step_00000006", "step_00000009"):
        assert (tmp_path / "ahead" / "checkpoints" / name / "manifest.json").is_file(), name


def test_a_stop_takes_effect_one_step_later(tmp_path: Path) -> None:
    # The time limit is due at step 1; step 2 was already dispatched, so it runs, is logged and checkpointed.
    config = experiment(str(tmp_path / "deadline"), train={"max_steps": 1000, "log_every": 1, "dispatch_ahead": True,
                                                          "time_limit_minutes": 1e-9, "save_final": False})
    result = Trainer(config).run()
    assert result.stop_reason == "time_limit" and result.step == 2
    assert [r["step"] for r in result.history] == [1, 2]
    assert result.data_state.batches_consumed == 2 * 2
    assert (tmp_path / "deadline" / "checkpoints" / "step_00000002" / "manifest.json").is_file()
