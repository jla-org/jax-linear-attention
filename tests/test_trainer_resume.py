# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resume equivalence: n steps == k steps + checkpoint + resume + (n - k) steps, bit for bit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from _toy_model import experiment
from jla.launch_env import DETERMINISTIC_FLAG
from jla.models import get_preset
from jla.training.trainer import Trainer

N_STEPS, SPLIT = 8, 3


def _experiment(path: Path, max_steps: int, **sections):
    train = {"max_steps": max_steps, "log_every": 1, "final_eval": False, **sections.pop("train", {})}
    optim = {"schedule_steps": N_STEPS, **sections.pop("optim", {})}
    return experiment(str(path), train=train, optim=optim, **sections)


def _host(tree) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _check_resume(tmp_path: Path, num_workers: int, *, exact: bool, runtime: dict | None = None) -> None:
    sections = {"data": {"num_workers": num_workers}}
    if runtime:
        sections["runtime"] = runtime
    full = Trainer(_experiment(tmp_path / "full", N_STEPS, **sections)).run()
    first = Trainer(_experiment(tmp_path / "split", SPLIT, **sections)).run()
    assert first.step == SPLIT
    second = Trainer(_experiment(tmp_path / "split", N_STEPS, **sections)).run()
    assert second.step == N_STEPS

    # Without XLA's deterministic mode, GPU scatter-adds (the embedding gradient) use atomics, so two
    # runs differ by an ulp whether or not they resume, and Adam's normalized updates grow that over
    # the steps. A resume bug (wrong data position or optimizer state) shows up far above 1e-3.
    same = np.testing.assert_array_equal if exact else (
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5))
    for expected, actual in zip(_host(full.model), _host(second.model), strict=True):
        same(expected, actual)
    for expected, actual in zip(_host(full.opt_state), _host(second.opt_state), strict=True):
        same(expected, actual)
    np.testing.assert_array_equal(jax.random.key_data(full.rng_key), jax.random.key_data(second.rng_key))
    assert full.data_state == second.data_state
    assert full.data_state.batches_consumed == N_STEPS * 2
    same(np.array([r["loss"] for r in full.history]), np.array([r["loss"] for r in first.history + second.history]))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resumed_run_matches_the_uninterrupted_run(tmp_path: Path, num_workers: int) -> None:
    """Bit for bit on CPU; to float rounding on GPU outside deterministic mode."""
    _check_resume(tmp_path, num_workers, exact=jax.default_backend() == "cpu")


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU determinism check")
def test_resumed_run_is_bitwise_identical_on_gpu_in_deterministic_mode(tmp_path: Path) -> None:
    """``runtime.deterministic`` (XLA's deterministic ops, XLA attention): bitwise on GPU too.

    XLA flags only apply before JAX initializes, so the check runs in a fresh process.
    """
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); "
        "import test_trainer_resume as t; "
        f"t._check_resume(Path({str(tmp_path)!r}), 0, exact=True, runtime={{'deterministic': True}}); "
        "print('BITWISE')"
    )
    env = dict(os.environ, XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " " + DETERMINISTIC_FLAG).strip())
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0 and "BITWISE" in proc.stdout, proc.stderr[-3000:]


def test_resume_rejects_a_changed_data_stream(tmp_path: Path) -> None:
    Trainer(_experiment(tmp_path / "run", SPLIT)).run()
    with pytest.raises(ValueError, match="cannot resume the data stream"):
        Trainer(_experiment(tmp_path / "run", N_STEPS, train={"seq_len": 16})).run()


def test_resume_never_refuses_an_existing_run(tmp_path: Path) -> None:
    Trainer(_experiment(tmp_path / "run", SPLIT)).run()
    with pytest.raises(FileExistsError, match="already holds step 3"):
        Trainer(_experiment(tmp_path / "run", N_STEPS, train={"resume": "never"})).run()


def test_resume_is_strict_about_parameter_dtypes(tmp_path: Path) -> None:
    Trainer(_experiment(tmp_path / "run", SPLIT)).run()
    changed = _experiment(tmp_path / "run", N_STEPS).replace(model=get_preset("tiny").replace(param_dtype="bfloat16"))
    with pytest.raises(ValueError, match="no implicit casts"):
        Trainer(changed).run()
