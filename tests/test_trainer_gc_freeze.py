# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``train.gc_freeze``: the step loop runs with the post-compile objects frozen, and nothing else changes."""

from __future__ import annotations

import contextlib
import gc
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from _toy_model import experiment
from jla.training import trainer as trainer_module
from jla.training.trainer import Trainer


def _arrays(tree):
    return jax.tree.leaves(eqx.filter(tree, eqx.is_array))


def test_frozen_gc_freezes_inside_and_unfreezes_after() -> None:
    with trainer_module._frozen_gc(True):
        assert gc.get_freeze_count() > 1000  # everything alive after the collection
    assert gc.get_freeze_count() == 0
    with trainer_module._frozen_gc(False):
        assert gc.get_freeze_count() == 0
    with pytest.raises(RuntimeError), trainer_module._frozen_gc(True):
        raise RuntimeError("loop failed")
    assert gc.get_freeze_count() == 0


def test_dpo_loop_runs_frozen_only_with_the_option(tmp_path: Path, monkeypatch) -> None:
    from test_dpo import _config

    from jla.training.dpo import DPOConfig, DPOTrainer

    seen: list[tuple[bool, int]] = []
    real = trainer_module._frozen_gc

    @contextlib.contextmanager
    def spy(enabled: bool):
        with real(enabled):
            seen.append((enabled, gc.get_freeze_count()))
            yield

    monkeypatch.setattr(trainer_module, "_frozen_gc", spy)
    for flag in (True, False):
        DPOTrainer(_config(tmp_path / f"run-{flag}", max_steps=2, gc_freeze=flag), DPOConfig(beta=0.5, synthetic_pairs=64)).run()
    assert seen[0][0] is True and seen[0][1] > 1000 and seen[1] == (False, 0)
    assert gc.get_freeze_count() == 0


def test_frozen_gc_leaves_a_callers_freeze_alone() -> None:
    gc.freeze()  # the caller's own freeze (before forking workers, say)
    try:
        frozen = gc.get_freeze_count()
        assert frozen > trainer_module._FROZEN_AT_IMPORT
        with trainer_module._frozen_gc(True):
            assert gc.get_freeze_count() == frozen  # only the collection ran
        assert gc.get_freeze_count() == frozen
    finally:
        gc.unfreeze()


def test_training_is_unchanged_by_gc_freeze(tmp_path: Path, monkeypatch) -> None:
    seen: list[int] = []
    loop = Trainer._loop

    def spy(self, **kwargs):
        seen.append(gc.get_freeze_count())
        return loop(self, **kwargs)

    monkeypatch.setattr(Trainer, "_loop", spy)
    runs = {}
    for flag in (True, False):
        config = experiment(str(tmp_path / f"run-{flag}"), train={"max_steps": 6, "log_every": 2, "gc_freeze": flag})
        runs[flag] = Trainer(config).run()
    assert seen[0] > 1000 and seen[1] == 0  # the loop ran frozen only with the option on
    assert gc.get_freeze_count() == 0
    on, off = runs[True], runs[False]
    assert on.step == off.step == 6

    def same(a, b):
        if jax.default_backend() == "cpu":
            np.testing.assert_array_equal(a, b)
        else:  # GPU reductions (scatter-add atomics) differ run to run in the last bits, with or without the option
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    same(np.asarray([r["loss"] for r in on.history]), np.asarray([r["loss"] for r in off.history]))
    for a, b in zip(_arrays(on.model), _arrays(off.model), strict=True):
        same(np.asarray(a), np.asarray(b))
    for a, b in zip(_arrays(on.opt_state), _arrays(off.opt_state), strict=True):
        same(np.asarray(a), np.asarray(b))
