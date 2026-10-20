# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic batch order, cheap resume, per-host rows and worker behaviour."""

from __future__ import annotations

import signal

import numpy as np
import pytest

from jla.data import loader as loader_module
from jla.data.datasets import SyntheticDataset
from jla.data.loader import BatchLoader, index_batches


class _Failing:
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        raise IndexError(f"broken sample {index}")


def _take(loader: BatchLoader, n: int) -> list[np.ndarray]:
    try:
        return [next(loader)["input_ids"] for _ in range(n)]
    finally:
        loader.close()


def _data():
    return SyntheticDataset(32, 8, seed=0, num_samples=20)


def test_index_stream_reshuffles_each_epoch_deterministically() -> None:
    stream = index_batches(10, 5, shuffle=True, seed=0)
    epoch1 = np.concatenate([next(stream), next(stream)])
    epoch2 = np.concatenate([next(stream), next(stream)])
    assert sorted(epoch1) == sorted(epoch2) == list(range(10))
    assert not np.array_equal(epoch1, epoch2)
    again = index_batches(10, 5, shuffle=True, seed=0)
    np.testing.assert_array_equal(next(again), epoch1[:5])
    with pytest.raises(ValueError, match="cannot fill one batch"):
        next(index_batches(3, 5, shuffle=False, seed=0))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_skip_batches_resumes_exactly(num_workers: int) -> None:
    full = _take(BatchLoader(_data(), 4, seed=1, num_workers=num_workers), 9)
    resumed_loader = BatchLoader(_data(), 4, seed=1, num_workers=num_workers, skip_batches=6)
    assert resumed_loader.batches_consumed == 6
    resumed = _take(resumed_loader, 3)
    for expected, actual in zip(full[6:], resumed, strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_rows_select_this_hosts_slice_of_the_global_batch() -> None:
    whole = _take(BatchLoader(_data(), 4, seed=2), 3)
    first = _take(BatchLoader(_data(), 4, seed=2, rows=(0, 2)), 3)
    second = _take(BatchLoader(_data(), 4, seed=2, rows=(2, 4), num_workers=1), 3)
    for full, a, b in zip(whole, first, second, strict=True):
        np.testing.assert_array_equal(full, np.concatenate([a, b]))
    with pytest.raises(ValueError, match="rows"):
        BatchLoader(_data(), 4, rows=(2, 6))


def test_worker_errors_reach_the_trainer() -> None:
    with pytest.raises(IndexError, match="broken sample"):
        _take(BatchLoader(_Failing(), 4, num_workers=2), 1)


def test_collate_worker_restores_default_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(loader_module.signal, "signal", lambda signum, handler: calls.append((signum, handler)))

    class OneStopQueue:
        def get(self, *, timeout: float):
            assert timeout == 0.1
            return None

    class Unset:
        def is_set(self) -> bool:
            return False

    loader_module._collate_worker(None, None, OneStopQueue(), None, Unset())
    assert calls == [(signal.SIGTERM, signal.SIG_DFL), (signal.SIGINT, signal.SIG_DFL)]
