# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, resumable batch loader with forked collation workers.

The loader walks a stream of *global* index batches (one shuffled permutation
of the dataset per epoch, drawn from a single seeded generator) and collates
only the rows this process owns (``rows=(start, stop)``). Every process
therefore sees the same global batch sequence, each host reads only its own
samples, and resuming after ``n`` batches replays index generation without
reading any data (``skip_batches=n``).

Collation runs in worker processes (``num_workers > 0``) forked from a clean
``forkserver`` (``spawn`` where unavailable), so they never inherit JAX's
threads; datasets and transforms must therefore be picklable. Workers receive
index lists over a bounded queue and results are reassembled in order. Workers
restore the default SIGTERM/SIGINT handlers so a trainer's graceful-stop
handler never keeps them alive.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue
import signal
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

__all__ = ["BatchLoader", "collate", "index_batches"]

Transform = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def collate(dataset: Any, indices: np.ndarray, transform: Transform | None = None) -> dict[str, np.ndarray]:
    """Stack ``dataset[i]`` for ``indices`` into ``[len(indices), ...]`` arrays."""
    samples = [dataset[int(i)] for i in indices]
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {}
    for key in samples[0]:
        arrays = [np.asarray(sample[key]) for sample in samples]
        out = np.empty((len(arrays), *arrays[0].shape), dtype=arrays[0].dtype)
        for row, array in enumerate(arrays):
            if array.shape != arrays[0].shape:
                raise ValueError(f"sample field {key!r} has inconsistent shapes {array.shape} vs {arrays[0].shape}")
            out[row] = array
        batch[key] = out
    return transform(batch) if transform is not None else batch


def index_batches(n: int, batch_size: int, *, shuffle: bool, seed: int, drop_last: bool = True) -> Iterator[np.ndarray]:
    """Endless stream of index batches; epoch permutations come from one generator."""
    if n <= 0 or batch_size <= 0:
        raise ValueError("dataset size and batch size must be positive")
    if drop_last and n < batch_size:
        raise ValueError(f"dataset of {n} samples cannot fill one batch of {batch_size}")
    rng = np.random.RandomState(seed)
    while True:
        order = np.arange(n, dtype=np.int64)
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n, batch_size):
            stop = start + batch_size
            if stop > n:
                if drop_last:
                    break
                stop = n
            yield order[start:stop]


def _collate_worker(dataset: Any, transform: Transform | None, jobs: Any, results: Any, stop: Any) -> None:
    # A forked child inherits the trainer's signal handlers; restore the
    # defaults so Process.terminate() actually ends it.
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, RuntimeError, ValueError):
            pass
    while not stop.is_set():
        try:
            job = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        if job is None:
            return
        job_id, indices = job
        try:
            results.put((job_id, collate(dataset, indices, transform)))
        except Exception as exc:  # forwarded to the parent, which re-raises it
            try:
                pickle.dumps(exc)
                forwarded: Exception = exc
            except Exception:
                forwarded = RuntimeError(f"collation worker failed:\n{traceback.format_exc()}")
            results.put((job_id, forwarded))
            return


class BatchLoader:
    """Iterator of fixed-shape numpy batches for one process.

    ``batch_size`` is the *global* batch; ``rows`` selects this process's
    contiguous slice of it (default: all rows). ``batches_consumed`` counts
    global batches yielded so far, including the ``skip_batches`` replayed at
    construction.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        rows: tuple[int, int] | None = None,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        prefetch: int = 4,
        skip_batches: int = 0,
        transform: Transform | None = None,
    ) -> None:
        start, stop = (0, batch_size) if rows is None else rows
        if not 0 <= start < stop <= batch_size:
            raise ValueError(f"rows must satisfy 0 <= start < stop <= {batch_size}, got {rows}")
        if num_workers < 0 or prefetch < 1 or skip_batches < 0:
            raise ValueError("num_workers >= 0, prefetch >= 1 and skip_batches >= 0 are required")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rows = (int(start), int(stop))
        self.transform = transform
        self.num_workers = int(num_workers)
        self.prefetch = int(prefetch)
        self._indices = index_batches(len(dataset), self.batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last)
        for _ in range(skip_batches):
            next(self._indices)
        self.batches_consumed = int(skip_batches)
        self._generator = self._serial() if self.num_workers == 0 else self._parallel()

    def __iter__(self) -> BatchLoader:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        batch = next(self._generator)
        self.batches_consumed += 1
        return batch

    def close(self) -> None:
        self._generator.close()

    def __enter__(self) -> BatchLoader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _local(self, indices: np.ndarray) -> np.ndarray:
        return indices[self.rows[0] : self.rows[1]]

    def _serial(self) -> Iterator[dict[str, np.ndarray]]:
        for indices in self._indices:
            yield collate(self.dataset, self._local(indices), self.transform)

    def _parallel(self) -> Iterator[dict[str, np.ndarray]]:
        # Workers fork from a clean single-threaded server process, never from
        # the (multithreaded) JAX process itself.
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
        in_flight_target = max(self.prefetch, self.num_workers)
        jobs = ctx.Queue(maxsize=in_flight_target + self.num_workers)
        results = ctx.Queue(maxsize=in_flight_target + self.num_workers)
        stop = ctx.Event()
        workers = [
            ctx.Process(target=_collate_worker, args=(self.dataset, self.transform, jobs, results, stop), daemon=True)
            for _ in range(self.num_workers)
        ]
        for worker in workers:
            worker.start()
        next_job = next_yield = in_flight = 0
        pending: dict[int, Any] = {}
        try:
            while True:
                while in_flight < in_flight_target:
                    jobs.put((next_job, self._local(next(self._indices))))
                    next_job += 1
                    in_flight += 1
                job_id, payload = results.get()
                in_flight -= 1
                if isinstance(payload, Exception):
                    raise payload
                pending[job_id] = payload
                while next_yield in pending:
                    batch = pending.pop(next_yield)
                    next_yield += 1
                    yield batch
        finally:
            stop.set()
            for _ in workers:
                try:
                    jobs.put_nowait(None)
                except queue.Full:
                    pass
            for worker in workers:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
            for q in (jobs, results):
                # Never block shutdown on a feeder thread whose reader is gone.
                q.cancel_join_thread()
                q.close()
