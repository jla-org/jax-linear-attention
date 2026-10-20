# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Config-driven data sources, the resumable data state and device prefetch.

The position of a run in its token stream is the number of global loader
batches consumed. :class:`DataState` records it with the geometry and a
fingerprint of everything that defines the stream, so a resume with a
different corpus, sequence length or batch size fails instead of silently
reading the wrong data.

:class:`DevicePrefetcher` groups ``grad_accum`` host batches into one
``[grad_accum, B_local, T]`` step batch and assembles the global array from
this process's rows only (``jax.make_array_from_process_local_data``) on a
background thread, a few steps ahead of the device.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from jla.data.cidx import check_vocab, resolve_record_split
from jla.data.datasets import BlendedDataset, SyntheticDataset, TokenWindowDataset
from jla.data.loader import BatchLoader
from jla.data.packing import DocumentFields
from jla.data.streaming import StreamLoader, StreamSpec, shard_geometry, shard_state

__all__ = [
    "DataState",
    "DevicePrefetcher",
    "data_fingerprint",
    "eval_batches",
    "eval_dataset",
    "train_dataset",
    "train_loader",
]


@dataclass(frozen=True)
class DataState:
    """Where a run is in its data stream (saved next to every checkpoint)."""

    step: int
    batches_consumed: int
    tokens_seen: int
    global_batch_size: int
    seq_len: int
    grad_accum: int
    fingerprint: str
    stream: Any = None  # data.source = "stream": every shard's position (jla.data.streaming)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["stream"] is None:
            del payload["stream"]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataState:
        names = {f.name for f in fields(cls)}
        if not names - {"stream"} <= set(payload) <= names:
            raise ValueError(f"data state keys {sorted(payload)} do not match {sorted(names)}")
        return cls(**payload)

    def require_compatible(self, *, global_batch_size: int, seq_len: int, fingerprint: str) -> None:
        problems = []
        if self.global_batch_size != global_batch_size:
            problems.append(f"global batch {self.global_batch_size} -> {global_batch_size}")
        if self.seq_len != seq_len:
            problems.append(f"seq_len {self.seq_len} -> {seq_len}")
        if self.fingerprint != fingerprint:
            problems.append("data source/order changed")
        if problems:
            raise ValueError("cannot resume the data stream: " + "; ".join(problems))


def data_fingerprint(config: Any) -> str:
    """Hash of the settings that define the training token stream of an ExperimentConfig."""
    data = config.data
    payload = {
        "source": data.source,
        "paths": list(data.paths),
        "weights": list(data.weights),
        "eval_holdout_records": data.eval_holdout_records,
        "exact_packed_records": data.exact_packed_records,
        "shuffle": data.shuffle,
        "eos_id": data.eos_id,
        "document_masking": data.document_masking,
        "synthetic": [data.synthetic_kind, data.synthetic_follow_prob, data.synthetic_doc_length, data.synthetic_samples],
        "seq_len": config.train.seq_len,
        "seed": config.train.seed,
        "vocab_size": config.model.vocab_size,
    }
    if data.source == "stream":
        payload["stream"] = [
            list(data.stream_files),
            [data.hf_dataset, data.hf_name, data.hf_split, list(data.hf_data_files)],
            [data.text_field, data.tokenizer, data.packing, data.pack_bins],
        ]
    if data.source == "sft":  # packing settings and corpus digests (other sources keep their fingerprints)
        from jla.data.sft import sft_fingerprint

        payload["sft"] = sft_fingerprint(config)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train_dataset(config: Any) -> Any:
    """Training dataset of an ExperimentConfig."""
    data, seq_len = config.data, config.train.seq_len
    if data.source == "stream":
        return StreamSpec.from_config(config)
    if data.source == "sft":
        from jla.data.sft import sft_dataset_from_config

        return sft_dataset_from_config(config)
    if data.source == "synthetic":
        return SyntheticDataset(
            config.model.vocab_size,
            seq_len,
            seed=config.train.seed,
            kind=data.synthetic_kind,
            follow_prob=data.synthetic_follow_prob,
            num_samples=data.synthetic_samples,
            eos_id=data.eos_id,
            doc_length=data.synthetic_doc_length,
        )
    for path in data.paths:
        check_vocab(path, config.model.vocab_size)
    if len(data.paths) > 1:
        weights = data.weights or (1.0,) * len(data.paths)
        return BlendedDataset(data.paths, weights, seq_len, seed=config.train.seed)
    record_range = None
    if data.eval_holdout_records:
        record_range, _ = resolve_record_split(data.paths[0], data.eval_holdout_records)
    return TokenWindowDataset(
        data.paths[0], seq_len, record_range=record_range, exact_packed_records=data.exact_packed_records
    )


def eval_dataset(config: Any) -> Any | None:
    """Evaluation dataset, or ``None`` when the config defines no evaluation source."""
    data, seq_len = config.data, config.train.seq_len
    if data.source == "sft":
        from jla.data.sft import sft_eval_dataset_from_config

        return sft_eval_dataset_from_config(config)
    if data.source == "synthetic":  # the training chain, on samples the training set never uses
        return SyntheticDataset(
            config.model.vocab_size,
            seq_len,
            seed=config.train.seed,
            kind=data.synthetic_kind,
            follow_prob=data.synthetic_follow_prob,
            num_samples=data.synthetic_samples,
            eos_id=data.eos_id,
            doc_length=data.synthetic_doc_length,
            offset=data.synthetic_samples,
        )
    if data.eval_path is not None:
        check_vocab(data.eval_path, config.model.vocab_size)
        return TokenWindowDataset(data.eval_path, seq_len, exact_packed_records=data.exact_packed_records)
    if data.eval_holdout_records:
        _, record_range = resolve_record_split(data.paths[0], data.eval_holdout_records)
        return TokenWindowDataset(
            data.paths[0], seq_len, record_range=record_range, exact_packed_records=data.exact_packed_records
        )
    return None


def _transform(config: Any) -> DocumentFields | None:
    return DocumentFields(config.data.eos_id) if config.data.document_masking else None


def train_loader(
    config: Any,
    dataset: Any,
    *,
    global_batch_size: int,
    rows: tuple[int, int],
    skip_batches: int,
    stream_state: Any = None,
) -> BatchLoader | StreamLoader:
    """Training loader for this process's rows, resumed after ``skip_batches`` global batches.

    A stream source (``train_dataset`` returned a :class:`StreamSpec`) resumes
    from ``stream_state``, the saved positions of every shard.
    """
    if isinstance(dataset, StreamSpec):
        shard_index, shard_count = shard_geometry(rows, global_batch_size)
        return StreamLoader(
            dataset,
            rows=rows[1] - rows[0],
            shard_index=shard_index,
            shard_count=shard_count,
            state=shard_state(stream_state, shard_index),
            skip_batches=skip_batches,
            # positions of every batch the device prefetcher may hold ahead of the training step
            history=(config.data.device_prefetch + 3) * config.train.grad_accum + 8,
        )
    return BatchLoader(
        dataset,
        global_batch_size,
        rows=rows,
        seed=config.train.seed,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        prefetch=config.data.prefetch,
        skip_batches=skip_batches,
        transform=_transform(config),
    )


def eval_batches(
    config: Any,
    dataset: Any,
    *,
    global_batch_size: int,
    rows: tuple[int, int],
    num_batches: int,
) -> Iterator[dict[str, np.ndarray]]:
    """The same fixed prefix of the evaluation set on every call."""
    available = len(dataset) // global_batch_size
    if available == 0:
        raise ValueError(f"evaluation set of {len(dataset)} samples has no complete batch of {global_batch_size}")
    loader = BatchLoader(dataset, global_batch_size, rows=rows, shuffle=False, transform=_transform(config))
    try:
        for _ in range(min(num_batches, available)):
            yield next(loader)
    finally:
        loader.close()


class DevicePrefetcher:
    """Background staging of step batches ``[grad_accum, B, T]`` onto the devices."""

    def __init__(
        self,
        loader: BatchLoader,
        *,
        grad_accum: int,
        sharding: Any,
        global_shape: tuple[int, int, int],
        depth: int = 2,
    ) -> None:
        self.loader = loader
        self.grad_accum = int(grad_accum)
        self.sharding = sharding
        self.global_shape = tuple(int(d) for d in global_shape)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jla-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from jla.distributed.sharding import make_global_batch

        try:
            while not self._stop.is_set():
                group = [next(self.loader) for _ in range(self.grad_accum)]
                local = {key: np.stack([batch[key] for batch in group]) for key in group[0]}
                item: Any = make_global_batch(local, self.sharding, self.global_shape)
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # surfaced to the training thread by __next__
            self._queue.put(exc)

    def __iter__(self) -> DevicePrefetcher:
        return self

    def __next__(self) -> dict[str, Any]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, timeout: float = 60.0) -> None:
        """Stop staging and close the loader (the staging thread finishes its current batch first)."""
        self._stop.set()
        deadline = time.monotonic() + timeout
        while self._thread.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.get(timeout=0.1)
            except queue.Empty:
                pass
            self._thread.join(timeout=0.1)
        if self._thread.is_alive():
            raise TimeoutError(f"data staging thread did not stop within {timeout:.0f}s")
        self.loader.close()
