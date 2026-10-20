# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Supervised fine-tuning corpora: CIDX tokens plus a loss-mask sidecar, packed into rows.

An SFT corpus ``prefix`` is a CIDX pair (``prefix.idx``, ``prefix.bin``) holding one
document per example (a whole conversation, or prompt + completion) and a sidecar
``prefix.mask``:

* header (little endian): magic ``0x4D544653`` ("SFTM") u32, version ``1`` u32,
  document count i64, token count i64, then a 32-byte SHA-256 digest of every
  document's token and mask bytes (the corpus identity used by resume checks);
* one ``uint8`` per token: ``1`` if the token is a training *target* (an assistant
  token), ``0`` otherwise.

:class:`SftDataset` turns the examples into fixed-length rows. Every example of ``n``
tokens contributes ``n - 1`` next-token positions (inputs ``x[:-1]``, labels
``x[1:]``, ``loss_mask = mask[1:]``); no position ever predicts across two examples.
``packing = "bfd"`` places whole examples into rows by best-fit decreasing (each
example keeps its own ``segment_ids`` value and ``positions`` restart at 0, so a
document-masked model sees exactly what it would see for the example alone);
``packing = "single"`` puts one example per row. The rest of a row is padding with
``loss_mask = 0`` in a segment of its own. The packing plan (including a fixed
shuffle of the row order) is a pure function of the corpus lengths and the settings,
so every process builds the same rows and resuming needs only the loader's batch
counter; the corpus digest and the settings enter the run's data fingerprint, so a
resume against a re-tokenized corpus or changed packing fails loudly.
"""

from __future__ import annotations

import bisect
import hashlib
import mmap
import os
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jla.data.cidx import CidxWriter, MMapIndexedDataset, _fsync_directory

__all__ = [
    "MASK_MAGIC",
    "MASK_VERSION",
    "SFT_PLAN_VERSION",
    "LossMaskFile",
    "SftDataset",
    "SftWriter",
    "mask_path",
    "pack_best_fit_decreasing",
    "read_mask_digest",
    "sft_dataset_from_config",
    "sft_eval_dataset_from_config",
    "sft_fingerprint",
    "write_sft",
]

MASK_MAGIC = 0x4D544653
MASK_VERSION = 1
_MASK_HEADER = struct.Struct("<IIqq32s")
PACKING_MODES = ("bfd", "single")
OVERLONG_MODES = ("truncate", "drop")
# Version of the row plan (which examples share a row, in which order). Bump it whenever
# pack_best_fit_decreasing, the fixed row shuffle or the truncation rule changes: it is part of
# the resume fingerprint, so a run cannot silently resume onto different rows.
SFT_PLAN_VERSION = 1


def mask_path(prefix: str | Path) -> str:
    return f"{prefix}.mask"


class LossMaskFile:
    """Read-only memory map of a ``prefix.mask`` sidecar, checked against its CIDX corpus."""

    def __init__(self, prefix: str | Path, *, documents: int, tokens: int) -> None:
        path = mask_path(prefix)
        with open(path, "rb") as stream:
            header = stream.read(_MASK_HEADER.size)
        if len(header) != _MASK_HEADER.size:
            raise ValueError(f"truncated loss-mask header in {path}")
        magic, version, mask_documents, mask_tokens, digest = _MASK_HEADER.unpack(header)
        if magic != MASK_MAGIC:
            raise ValueError(f"bad magic in {path}: 0x{magic:08X} (not an SFT loss-mask sidecar)")
        if version != MASK_VERSION:
            raise ValueError(f"unsupported loss-mask version {version} in {path}")
        if (mask_documents, mask_tokens) != (documents, tokens):
            raise ValueError(
                f"{path} describes {mask_documents} documents / {mask_tokens} tokens but its CIDX corpus has "
                f"{documents} / {tokens}: the sidecar belongs to another corpus"
            )
        if os.path.getsize(path) != _MASK_HEADER.size + tokens:
            raise ValueError(f"{path} must hold exactly {tokens} mask bytes after its header")
        self.path = path
        self.digest = digest.hex()
        self._file = open(path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.data = np.frombuffer(self._mmap, dtype=np.uint8, offset=_MASK_HEADER.size, count=tokens)

    def close(self) -> None:
        self.data = None  # type: ignore[assignment]
        if getattr(self, "_mmap", None) is not None:
            try:
                self._mmap.close()
            except (BufferError, ValueError):
                pass  # a view is still alive; the mapping closes with it
            self._mmap = None  # type: ignore[assignment]
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]


def read_mask_digest(prefix: str | Path) -> str:
    """The corpus digest stored in the sidecar header (no data is read)."""
    with open(mask_path(prefix), "rb") as stream:
        header = stream.read(_MASK_HEADER.size)
    if len(header) != _MASK_HEADER.size:
        raise ValueError(f"truncated loss-mask header in {mask_path(prefix)}")
    magic, version, _, _, digest = _MASK_HEADER.unpack(header)
    if magic != MASK_MAGIC or version != MASK_VERSION:
        raise ValueError(f"{mask_path(prefix)} is not a version-{MASK_VERSION} SFT loss-mask sidecar")
    return digest.hex()


class SftWriter:
    """Stream ``(tokens, mask)`` examples into an SFT corpus; all three files appear on :meth:`commit`."""

    def __init__(self, prefix: str | Path, *, dtype: str = "int32") -> None:
        self.prefix = Path(prefix)
        self._tokens = CidxWriter(self.prefix, dtype=dtype)
        self._mask_tmp = self.prefix.parent / f".{self.prefix.name}.mask.tmp.{os.getpid()}"
        self._mask = open(self._mask_tmp, "wb", buffering=4 * 1024 * 1024)
        self._mask.write(b"\0" * _MASK_HEADER.size)  # rewritten on commit
        self._hash = hashlib.sha256()
        self._token_dtype = np.dtype(self._tokens._np_dtype)
        self._closed = False
        self.trainable_tokens = 0

    @property
    def num_documents(self) -> int:
        return self._tokens.num_documents

    @property
    def num_tokens(self) -> int:
        return self._tokens.num_tokens

    def add(self, tokens: Sequence[int] | np.ndarray, mask: Sequence[int] | np.ndarray) -> None:
        """Append one example; ``mask[i] = 1`` marks token ``i`` as a training target."""
        token_array = np.asarray(tokens)
        mask_array = np.asarray(mask)
        if mask_array.shape != token_array.shape:
            raise ValueError(f"mask shape {mask_array.shape} does not match tokens {token_array.shape}")
        if mask_array.size and not np.isin(mask_array, (0, 1)).all():
            raise ValueError("mask values must be 0 or 1")
        self._tokens.add(token_array)  # validates shape, dtype and range
        mask_bytes = mask_array.astype(np.uint8).tobytes()
        self._mask.write(mask_bytes)
        self._hash.update(token_array.astype(self._token_dtype, copy=False).tobytes())
        self._hash.update(mask_bytes)
        self.trainable_tokens += int(mask_array.sum())

    def commit(self) -> tuple[int, int]:
        if self._closed:
            raise RuntimeError("SftWriter is closed")
        try:
            self._mask.seek(0)
            self._mask.write(
                _MASK_HEADER.pack(MASK_MAGIC, MASK_VERSION, self.num_documents, self.num_tokens, self._hash.digest())
            )
            self._mask.flush()
            os.fsync(self._mask.fileno())
            self._mask.close()
            # The sidecar header names its corpus' counts, so a crash between the two renames
            # leaves a pair that fails loudly at load time instead of a silently wrong mask.
            result = self._tokens.commit()
            os.replace(self._mask_tmp, mask_path(self.prefix))
            _fsync_directory(self.prefix.parent)
        except BaseException:
            self._tokens.abort()
            self._mask_tmp.unlink(missing_ok=True)
            raise
        finally:
            self._closed = True
        return result

    def abort(self) -> None:
        if not self._closed:
            self._tokens.abort()
            self._mask.close()
            self._mask_tmp.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> SftWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()


def write_sft(
    prefix: str | Path, examples: Iterable[tuple[Sequence[int], Sequence[int]]], *, dtype: str = "int32"
) -> tuple[int, int]:
    """Write ``(tokens, mask)`` examples to ``prefix.{bin,idx,mask}``; returns ``(documents, tokens)``."""
    writer = SftWriter(prefix, dtype=dtype)
    try:
        for tokens, mask in examples:
            writer.add(tokens, mask)
    except BaseException:
        writer.abort()
        raise
    return writer.commit()


def pack_best_fit_decreasing(lengths: np.ndarray, capacity: int) -> list[list[int]]:
    """Bins of item indices with ``sum(lengths[bin]) <= capacity`` (best-fit decreasing).

    Items are placed longest first (ties by index), each into the open bin with the
    least remaining room that still fits it; deterministic for given ``lengths``.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    if lengths.size and (lengths.min() <= 0 or lengths.max() > capacity):
        raise ValueError(f"item lengths must be in [1, {capacity}]")
    order = np.argsort(-lengths, kind="stable")
    bins: list[list[int]] = []
    by_room: dict[int, list[int]] = {}  # remaining room -> open bins with exactly that room
    rooms: list[int] = []  # sorted keys of by_room
    for item in order.tolist():
        length = int(lengths[item])
        slot = bisect.bisect_left(rooms, length)
        if slot == len(rooms):
            bins.append([item])
            room = capacity - length
            target = len(bins) - 1
        else:
            room = rooms[slot]
            target = by_room[room].pop()
            if not by_room[room]:
                del by_room[room]
                rooms.pop(slot)
            bins[target].append(item)
            room -= length
        if room > 0:
            if room not in by_room:
                by_room[room] = []
                bisect.insort(rooms, room)
            by_room[room].append(target)
    return bins


def _segment_sums(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """``values[starts[i]:ends[i]].sum()`` for disjoint, increasing, non-empty spans."""
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64)
    bounds = np.empty(2 * starts.size, dtype=np.int64)
    bounds[0::2], bounds[1::2] = starts, ends
    if int(ends[-1]) == values.size:  # reduceat needs in-range indices; the last span then runs to the end
        bounds = bounds[:-1]
    return np.add.reduceat(values, bounds, dtype=np.int64)[0::2]


class SftDataset:
    """Fixed-length SFT rows over one or more SFT corpora (see the module docstring).

    ``record_range`` restricts a single corpus to a document range (train/eval
    holdout split). ``overlong`` handles examples longer than ``seq_len + 1``
    tokens: ``"truncate"`` keeps their first ``seq_len + 1`` tokens, ``"drop"``
    skips them. Examples without any trainable target (after truncation) are
    skipped. With ``completion_only = False`` every position of every example is
    trained. ``pad_id`` fills the unused end of each row.
    """

    def __init__(
        self,
        paths: str | Sequence[str],
        seq_len: int,
        *,
        record_range: tuple[int, int] | None = None,
        packing: str = "bfd",
        overlong: str = "truncate",
        completion_only: bool = True,
        pad_id: int = 0,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if packing not in PACKING_MODES:
            raise ValueError(f"packing must be one of {PACKING_MODES}, got {packing!r}")
        if overlong not in OVERLONG_MODES:
            raise ValueError(f"overlong must be one of {OVERLONG_MODES}, got {overlong!r}")
        if pad_id < 0:
            raise ValueError("pad_id must be >= 0")
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not paths:
            raise ValueError("at least one SFT corpus path is required")
        if record_range is not None and len(paths) != 1:
            raise ValueError("record_range needs exactly one corpus")
        self.paths = [str(path) for path in paths]
        self.seq_len = int(seq_len)
        self.packing = packing
        self.overlong = overlong
        self.completion_only = bool(completion_only)
        self.pad_id = int(pad_id)
        self.record_range = record_range
        self._open()
        self._plan()

    # -- corpus access (re-opened in loader workers) ------------------------------

    def _open(self) -> None:
        self.corpora = [MMapIndexedDataset(path) for path in self.paths]
        self.masks = [
            LossMaskFile(path, documents=corpus.num_documents, tokens=corpus.num_tokens)
            for path, corpus in zip(self.paths, self.corpora)
        ]

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        del state["corpora"], state["masks"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()

    # -- packing plan --------------------------------------------------------------

    def _plan(self) -> None:
        window = self.seq_len + 1
        sources, documents, lengths = [], [], []
        self.dropped_overlong = self.dropped_untrainable = self.truncated = 0
        for source, (corpus, mask) in enumerate(zip(self.corpora, self.masks)):
            start, end = (0, corpus.num_documents) if self.record_range is None else self.record_range
            if not 0 <= start < end <= corpus.num_documents:
                raise ValueError(f"record_range must satisfy 0 <= start < end <= {corpus.num_documents}; got {self.record_range}")
            offsets = corpus.offsets[start : end + 1]
            sizes = np.diff(offsets)
            keep = sizes >= 2  # one-token examples have no next-token target
            if self.overlong == "drop":
                overlong = sizes > window
                self.dropped_overlong += int((overlong & keep).sum())
                keep &= ~overlong
            used = np.minimum(sizes, window)
            if self.completion_only:
                # Targets are tokens 1 .. used-1 of every example.
                index = np.flatnonzero(keep)
                trainable = _segment_sums(mask.data, offsets[index] + 1, offsets[index] + used[index])
                untrainable = index[trainable == 0]
                self.dropped_untrainable += int(untrainable.size)
                keep[untrainable] = False
            self.truncated += int((keep & (sizes > window)).sum())
            index = np.flatnonzero(keep)
            sources.append(np.full(index.size, source, dtype=np.int32))
            documents.append(index.astype(np.int64) + start)
            lengths.append(used[index] - 1)
        self._sources = np.concatenate(sources)
        self._documents = np.concatenate(documents)
        self._positions = np.concatenate(lengths).astype(np.int64)  # next-token positions per example
        if self._positions.size == 0:
            raise ValueError(f"{self.paths}: no example with a trainable target fits seq_len={self.seq_len}")
        if self.packing == "bfd":
            bins = pack_best_fit_decreasing(self._positions, self.seq_len)
            # Rows come out longest-first; a fixed shuffle keeps any prefix of them (an evaluation
            # slice) representative of the whole set.
            bins = [bins[i] for i in np.random.default_rng(0).permutation(len(bins)).tolist()]
        else:
            bins = [[item] for item in range(self._positions.size)]
        self._bin_offsets = np.cumsum([0] + [len(b) for b in bins], dtype=np.int64)
        self._bin_items = np.fromiter((item for b in bins for item in b), dtype=np.int64, count=int(self._bin_offsets[-1]))

    @property
    def num_examples(self) -> int:
        return int(self._positions.size)

    def fill_fraction(self) -> float:
        """Share of row positions holding example tokens (the rest is padding)."""
        return float(self._positions.sum()) / float(len(self) * self.seq_len)

    def __len__(self) -> int:
        return int(self._bin_offsets.size - 1)

    def row_items(self, index: int) -> list[tuple[int, int]]:
        """``(corpus, document)`` of every example in row ``index``, in row order."""
        items = self._bin_items[self._bin_offsets[index] : self._bin_offsets[index + 1]]
        return [(int(self._sources[i]), int(self._documents[i])) for i in items]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range [0, {len(self)})")
        length = self.seq_len
        input_ids = np.full(length, self.pad_id, dtype=np.int32)
        labels = np.full(length, self.pad_id, dtype=np.int32)
        loss_mask = np.zeros(length, dtype=bool)
        segment_ids = np.zeros(length, dtype=np.int32)
        positions = np.zeros(length, dtype=np.int32)
        cursor = 0
        items = self._bin_items[self._bin_offsets[index] : self._bin_offsets[index + 1]]
        for segment, item in enumerate(items.tolist()):
            source, document = int(self._sources[item]), int(self._documents[item])
            count = int(self._positions[item])
            begin = int(self.corpora[source].offsets[document])
            tokens = self.corpora[source].get_chunk(begin, count + 1)
            stop = cursor + count
            input_ids[cursor:stop] = tokens[:-1]
            labels[cursor:stop] = tokens[1:]
            if self.completion_only:
                loss_mask[cursor:stop] = self.masks[source].data[begin + 1 : begin + count + 1] != 0
            else:
                loss_mask[cursor:stop] = True
            segment_ids[cursor:stop] = segment
            positions[cursor:stop] = np.arange(count, dtype=np.int32)
            cursor = stop
        if cursor < length:  # padding: its own segment, outside the loss
            segment_ids[cursor:] = len(items)
            positions[cursor:] = np.arange(length - cursor, dtype=np.int32)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "segment_ids": segment_ids,
            "positions": positions,
        }


# ---------------------------------------------------------------------------
# ExperimentConfig glue (used by jla.data.pipeline)
# ---------------------------------------------------------------------------


def _dataset_options(data: Any) -> dict[str, Any]:
    return {
        "packing": data.sft_packing,
        "overlong": data.sft_overlong,
        "completion_only": data.sft_completion_only,
        "pad_id": data.sft_pad_id,
    }


def sft_dataset_from_config(config: Any) -> SftDataset:
    """Training rows of an ExperimentConfig with ``data.source = "sft"``."""
    from jla.data.cidx import check_vocab, resolve_record_split

    data = config.data
    for path in data.paths:
        check_vocab(path, config.model.vocab_size)
    record_range = None
    if data.eval_holdout_records:
        record_range, _ = resolve_record_split(data.paths[0], data.eval_holdout_records)
    return SftDataset(data.paths, config.train.seq_len, record_range=record_range, **_dataset_options(data))


def sft_eval_dataset_from_config(config: Any) -> SftDataset | None:
    """Evaluation rows (``data.eval_path`` or the held-out tail), or ``None``."""
    from jla.data.cidx import check_vocab, resolve_record_split

    data = config.data
    if data.eval_path is not None:
        check_vocab(data.eval_path, config.model.vocab_size)
        return SftDataset(data.eval_path, config.train.seq_len, **_dataset_options(data))
    if data.eval_holdout_records:
        _, record_range = resolve_record_split(data.paths[0], data.eval_holdout_records)
        return SftDataset(data.paths[0], config.train.seq_len, record_range=record_range, **_dataset_options(data))
    return None


def sft_fingerprint(config: Any) -> list[Any]:
    """Settings and corpus digests that define an SFT token stream (for ``data_fingerprint``)."""
    data = config.data
    return [
        SFT_PLAN_VERSION,
        data.sft_packing,
        data.sft_overlong,
        data.sft_completion_only,
        data.sft_pad_id,
        [read_mask_digest(path) for path in data.paths],
    ]
