# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming text data (``data.source = "stream"``): read, tokenize, pack, shard and resume on the fly.

Per data shard the pipeline is::

    records   local .jsonl / .parquet files (data.stream_files) or a streamed
              Hugging Face dataset (data.hf_dataset, optional 'datasets' extra)
    shard     record r of an epoch belongs to shard r % shard_count
    tokenize  data.tokenizer: "byte" (UTF-8 bytes, ids 0-255) or a Hugging Face
              fast tokenizer; the EOS id is appended to every document
    pack      concat_split: documents back to back, rows are windows of
                seq_len + 1 tokens at stride seq_len, exactly the windows the
                pre-tokenized CIDX path reads (document masking, if enabled,
                is then derived from EOS tokens as for CIDX);
              first_fit: whole documents go into the first of pack_bins bins
                with room; a document that fits no bin emits every bin and
                starts a new set (Grain's FirstFitPackIterDataset as MaxText
                uses it). Documents longer than seq_len are split into
                seq_len chunks instead of truncated, so no target is lost.
    batch     the rows of this shard, ``rows`` at a time

A shard is a contiguous block of rows of the global batch, so processes that
hold the same rows (model parallelism across hosts) read the same stream.
Records are visited in file order; ``data.shuffle`` permutes the file order
per epoch (seeded), documents inside a file keep their order. Every record is
read by every shard but parsed and tokenized only by its owner.

Resume: :class:`StreamLoader` remembers its position after each of its recent
batches (file, byte offset or row, packer buffer, unbatched rows). The trainer
saves the position at the consumed batch count, gathered from every shard, in
the checkpoint's ``data_state`` sidecar; a resumed loader continues with the
next batch the uninterrupted run would have produced.
"""

from __future__ import annotations

import base64
import collections
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jla.data.packing import DocumentFields

__all__ = [
    "STATE_VERSION",
    "StreamLoader",
    "StreamSpec",
    "build_tokenizer",
    "document_pieces",
    "encode_bytes",
    "first_fit_row",
    "gather_shard_states",
    "read_documents",
    "shard_geometry",
    "shard_state",
    "stream_positions",
]

STATE_VERSION = 1
Encoder = Callable[[list[str]], list[Sequence[int]]]


@dataclass(frozen=True)
class StreamSpec:
    """Everything that defines one shard's token stream (built from an ExperimentConfig)."""

    seq_len: int
    vocab_size: int
    tokenizer: str
    eos_id: int | None = None
    files: tuple[str, ...] = ()
    hf_dataset: str = ""
    hf_name: str | None = None
    hf_split: str = "train"
    hf_data_files: tuple[str, ...] = ()
    text_field: str = "text"
    packing: str = "concat_split"
    pack_bins: int = 0  # first_fit bins per flush; 0 = the shard's rows per batch
    document_masking: bool = False
    shuffle: bool = False
    seed: int = 0
    read_ahead: int = 1024  # documents per tokenizer call (batched calls let fast tokenizers use their threads)
    tokenizer_local_files_only: bool = False
    read_ahead_chars: int = 8 << 20  # and at most this much text per call (bounds memory on very long documents)

    def __post_init__(self) -> None:
        if self.seq_len < 1 or self.vocab_size < 2:
            raise ValueError("seq_len >= 1 and vocab_size >= 2 are required")
        if bool(self.files) == bool(self.hf_dataset):
            raise ValueError("a stream needs exactly one of files and hf_dataset")
        if self.packing not in ("concat_split", "first_fit"):
            raise ValueError(f"packing must be 'concat_split' or 'first_fit', got {self.packing!r}")
        if self.pack_bins < 0 or self.read_ahead < 1 or self.read_ahead_chars < 1:
            raise ValueError("pack_bins >= 0, read_ahead >= 1 and read_ahead_chars >= 1 are required")
        if self.hf_dataset and self.shuffle:
            raise ValueError("shuffle permutes the order of local files; it is not supported for hf_dataset")
        for path in self.files:
            if not path.endswith((".jsonl", ".parquet")):
                raise ValueError(f"stream files must end in .jsonl or .parquet, got {path!r}")

    @classmethod
    def from_config(cls, config: Any) -> StreamSpec:
        data = config.data
        return cls(
            seq_len=config.train.seq_len,
            vocab_size=config.model.vocab_size,
            tokenizer=data.tokenizer,
            eos_id=data.eos_id,
            files=tuple(data.stream_files),
            hf_dataset=data.hf_dataset,
            hf_name=data.hf_name,
            hf_split=data.hf_split,
            hf_data_files=tuple(data.hf_data_files),
            text_field=data.text_field,
            packing=data.packing,
            pack_bins=data.pack_bins,
            document_masking=data.document_masking,
            shuffle=data.shuffle,
            seed=config.train.seed,
            read_ahead=data.stream_read_ahead,
            tokenizer_local_files_only=data.tokenizer_local_files_only,
        )


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


def encode_bytes(texts: list[str]) -> list[np.ndarray]:
    """The ``byte`` tokenizer: UTF-8 bytes as token ids 0-255."""
    return [np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32) for text in texts]


def build_tokenizer(name: str, *, eos_id: int | None, local_files_only: bool = False) -> tuple[Encoder, int]:
    """``(encode_batch, eos_id)`` for ``"byte"`` or a Hugging Face tokenizer name/path."""
    if name == "byte":
        if eos_id is None or eos_id < 256:
            raise ValueError("the byte tokenizer needs an eos_id >= 256 (ids 0-255 are bytes)")
        return encode_bytes, int(eos_id)
    if not name.strip():
        raise ValueError("a stream needs a tokenizer ('byte' or a Hugging Face tokenizer name/path)")
    from jla.data.make_data import load_hf_tokenizer

    encode, tokenizer_eos = load_hf_tokenizer(name, local_files_only=local_files_only)
    if eos_id is None:
        if tokenizer_eos is None:
            raise ValueError(f"tokenizer {name!r} has no EOS token; set data.eos_id")
        return encode, int(tokenizer_eos)
    if tokenizer_eos is not None and int(tokenizer_eos) != int(eos_id):
        raise ValueError(f"data.eos_id = {eos_id} differs from the EOS id {tokenizer_eos} of tokenizer {name!r}")
    return encode, int(eos_id)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _json_text(line: str, field: str) -> str | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = record.get(field) if isinstance(record, dict) else None
    return text if isinstance(text, str) and text else None


def _jsonl_lines(path: str, offset: int) -> Iterator[tuple[int, str]]:
    """Non-blank lines from byte ``offset`` on, each with the offset just after it."""
    with open(path, "rb") as stream:
        stream.seek(offset)
        for raw in stream:
            offset += len(raw)
            line = raw.decode("utf-8").strip()
            if line:
                yield offset, line


def _parquet_rows(path: str, row: int, field: str) -> Iterator[tuple[int, Any]]:
    """Values of column ``field`` from ``row`` on, each with the index of the next row."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("parquet stream files need the optional 'pyarrow' package") from exc
    parquet = pq.ParquetFile(path)
    if field not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no column {field!r} (columns: {parquet.schema_arrow.names})")
    start = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        if row < start + count:
            values = parquet.read_row_group(group, columns=[field]).column(0).to_pylist()
            for index in range(max(row - start, 0), count):
                yield start + index + 1, values[index]
        start += count


def _hf_examples(spec: StreamSpec, record: int) -> Iterator[tuple[int, Any]]:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("data.hf_dataset needs the optional 'datasets' package") from exc
    stream = datasets.load_dataset(
        spec.hf_dataset,
        spec.hf_name,
        split=spec.hf_split,
        data_files=list(spec.hf_data_files) or None,
        streaming=True,
    )
    if record:
        stream = stream.skip(record)
    for index, example in enumerate(stream, start=record):
        yield index + 1, example.get(spec.text_field) if isinstance(example, dict) else None


def _file_order(spec: StreamSpec, epoch: int) -> list[str]:
    files = list(spec.files)
    if spec.shuffle:
        order = np.random.default_rng([spec.seed, epoch]).permutation(len(files))
        files = [files[i] for i in order]
    return files


def _shard_texts(spec: StreamSpec, cursor: dict[str, int], shard: int, shards: int) -> Iterator[tuple[dict[str, int], str]]:
    """This shard's non-empty texts, endlessly over epochs, each with the cursor just after it.

    The cursor ``{"epoch", "file", "offset", "record"}`` names the next record:
    ``file`` indexes the epoch's file order, ``offset`` is a byte offset (jsonl)
    or row index (parquet, HF), ``record`` counts the epoch's records so far.
    """
    epoch, file_pos, offset, record = cursor["epoch"], cursor["file"], cursor["offset"], cursor["record"]
    while True:
        found = record > 0  # a resumed epoch already had records
        if spec.hf_dataset:
            sources: list[Any] = [None]
        else:
            sources = _file_order(spec, epoch)
        while file_pos < len(sources):
            path = sources[file_pos]
            if spec.hf_dataset:
                items: Iterator[tuple[int, Any]] = _hf_examples(spec, offset)
            elif path.endswith(".jsonl"):
                items = _jsonl_lines(path, offset)
            else:
                items = _parquet_rows(path, offset, spec.text_field)
            for offset, value in items:
                mine = record % shards == shard
                record += 1
                if not mine:
                    continue
                if path is not None and path.endswith(".jsonl"):
                    text = _json_text(value, spec.text_field)
                else:
                    text = value if isinstance(value, str) and value else None
                if text is None:
                    continue
                found = True
                yield {"epoch": epoch, "file": file_pos, "offset": offset, "record": record}, text
            file_pos, offset = file_pos + 1, 0
        if not found:
            raise ValueError(f"stream shard {shard} of {shards} has no non-empty documents")
        epoch, file_pos, offset, record = epoch + 1, 0, 0, 0


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


class _ConcatSplit:
    """Rows are windows of ``seq_len + 1`` tokens at stride ``seq_len`` over the concatenated documents."""

    def __init__(self, seq_len: int, buffer: np.ndarray | None = None) -> None:
        self.seq_len = seq_len
        self.buffer = np.zeros(0, np.int32) if buffer is None else buffer

    def add(self, tokens: np.ndarray) -> list[np.ndarray]:
        stream = np.concatenate([self.buffer, tokens])
        count = (stream.size - 1) // self.seq_len
        rows = [stream[i * self.seq_len : (i + 1) * self.seq_len + 1] for i in range(count)]
        self.buffer = stream[count * self.seq_len :]  # replaced, never mutated: snapshots hold references
        return rows

    def snapshot(self) -> Any:
        return self.buffer

    def encode(self, snapshot: Any) -> Any:
        return _b64(snapshot)


Piece = tuple[np.ndarray, int]  # (inputs plus the next target if it exists, number of inputs)


def document_pieces(tokens: np.ndarray, seq_len: int, eos_id: int) -> list[Piece]:
    """Split a tokenized document (ending in EOS) at EOS tokens, then into chunks of ``seq_len`` inputs.

    Each piece carries its inputs and, unless it ends its document, the token
    after its last input, so a split document keeps every next-token target.
    """
    ends = np.flatnonzero(tokens == eos_id) + 1
    if ends.size == 0 or ends[-1] != tokens.size:
        ends = np.append(ends, tokens.size)
    pieces: list[Piece] = []
    start = 0
    for end in ends.tolist():
        document = tokens[start:end]
        for begin in range(0, document.size, seq_len):
            stop = min(begin + seq_len, document.size)
            pieces.append((document[begin : min(stop + 1, document.size)], stop - begin))
        start = end
    return pieces


class _FirstFit:
    """First-fit packing of document pieces into ``bins`` rows of ``seq_len`` inputs."""

    def __init__(self, seq_len: int, bins: int, eos_id: int, state: list[list[Piece]] | None = None) -> None:
        self.seq_len, self.num_bins, self.eos_id = seq_len, bins, eos_id
        self.bins: list[list[Piece]] = [list(b) for b in state] if state is not None else [[] for _ in range(bins)]
        if len(self.bins) != bins:
            raise ValueError(f"saved first-fit state has {len(self.bins)} bins, the stream uses {bins}")
        self.used = [sum(n for _, n in b) for b in self.bins]

    def add(self, tokens: np.ndarray) -> list[tuple[Piece, ...]]:
        rows: list[tuple[Piece, ...]] = []
        for piece in document_pieces(tokens, self.seq_len, self.eos_id):
            size = piece[1]
            target = next((b for b in range(self.num_bins) if self.used[b] + size <= self.seq_len), None)
            if target is None:  # fits no bin: emit them all, start over
                rows.extend(tuple(b) for b in self.bins)
                self.bins = [[] for _ in range(self.num_bins)]
                self.used = [0] * self.num_bins
                target = 0
            self.bins[target].append(piece)
            self.used[target] += size
        return rows

    def snapshot(self) -> Any:
        return [tuple(b) for b in self.bins]

    def encode(self, snapshot: Any) -> Any:
        return [[_encode_piece(p) for p in b] for b in snapshot]


def first_fit_row(pieces: Sequence[Piece], seq_len: int, eos_id: int) -> dict[str, np.ndarray]:
    """One packed row: inputs, targets, loss mask, per-piece segment ids and positions; EOS padding."""
    input_ids = np.full(seq_len, eos_id, np.int32)
    labels = np.full(seq_len, eos_id, np.int32)
    valid = np.zeros(seq_len, bool)
    segment_ids = np.empty(seq_len, np.int32)
    positions = np.empty(seq_len, np.int32)
    cursor = 0
    for index, (tokens, size) in enumerate(pieces):
        input_ids[cursor : cursor + size] = tokens[:size]
        targets = tokens.size - 1
        labels[cursor : cursor + targets] = tokens[1:]
        valid[cursor : cursor + targets] = True
        segment_ids[cursor : cursor + size] = index
        positions[cursor : cursor + size] = np.arange(size, dtype=np.int32)
        cursor += size
    segment_ids[cursor:] = len(pieces)  # padding is its own segment
    positions[cursor:] = np.arange(seq_len - cursor, dtype=np.int32)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": valid & (input_ids != eos_id),
        "segment_ids": segment_ids,
        "positions": positions,
    }


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<i4").tobytes()).decode("ascii")


def _unb64(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text.encode("ascii")), dtype="<i4").astype(np.int32)


def _encode_piece(piece: Piece) -> list[Any]:
    return [_b64(piece[0]), int(piece[1])]


def _decode_piece(payload: Sequence[Any]) -> Piece:
    tokens, size = _unb64(payload[0]), int(payload[1])
    if not 0 < size <= tokens.size <= size + 1:
        raise ValueError("corrupt first-fit piece in the saved stream state")
    return tokens, size


@dataclass(frozen=True)
class _Snapshot:
    cursor: dict[str, int]
    packer: Any
    pending: tuple[Any, ...]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def shard_geometry(rows: tuple[int, int], global_batch_size: int) -> tuple[int, int]:
    """``(shard_index, shard_count)`` of a process's contiguous row block."""
    start, stop = rows
    local = stop - start
    if local <= 0 or global_batch_size % local or start % local:
        raise ValueError(f"rows {rows} do not tile a global batch of {global_batch_size} into equal blocks")
    return start // local, global_batch_size // local


class StreamLoader:
    """Iterator of this shard's batches ``{"input_ids", "labels", ...}: [rows, seq_len]``.

    ``batches_consumed`` counts batches produced, including the ``skip_batches``
    of a resume; ``state_at(n)`` is the JSON-ready position after batch ``n`` for
    any of the last ``history`` batches.
    """

    def __init__(
        self,
        spec: StreamSpec,
        *,
        rows: int,
        shard_index: int = 0,
        shard_count: int = 1,
        state: dict[str, Any] | None = None,
        skip_batches: int = 0,
        history: int = 64,
        encoder: tuple[Encoder, int] | None = None,
    ) -> None:
        if rows < 1 or not 0 <= shard_index < shard_count or history < 1 or skip_batches < 0:
            raise ValueError("rows >= 1, 0 <= shard_index < shard_count, history >= 1 and skip_batches >= 0 are required")
        self.spec = spec
        self.rows, self.shard_index, self.shard_count = int(rows), int(shard_index), int(shard_count)
        self._encode, self.eos_id = encoder or build_tokenizer(
            spec.tokenizer, eos_id=spec.eos_id, local_files_only=spec.tokenizer_local_files_only
        )
        if not 0 <= self.eos_id < spec.vocab_size:
            raise ValueError(f"eos id {self.eos_id} is outside the vocabulary of {spec.vocab_size}")
        self.bins = spec.pack_bins or self.rows
        self._document_fields = DocumentFields(self.eos_id) if spec.document_masking else None
        if state is None:
            if skip_batches:
                raise ValueError("resuming a stream after %d batches needs its saved position" % skip_batches)
            cursor = {"epoch": 0, "file": 0, "offset": 0, "record": 0}
            self._packer = self._new_packer(None)
            self._pending: list[Any] = []
        else:
            cursor = self._load(state, skip_batches)
        self.batches_consumed = int(skip_batches)
        self._cursor = dict(cursor)
        self._texts = _shard_texts(spec, dict(cursor), self.shard_index, self.shard_count)
        self._ready: collections.deque[tuple[dict[str, int], np.ndarray]] = collections.deque()
        self._lock = threading.Lock()
        self._history: collections.OrderedDict[int, _Snapshot] = collections.OrderedDict()
        self._history_size = int(history)
        self._remember()

    # -- state -----------------------------------------------------------------

    def _new_packer(self, packed: Any) -> Any:
        if self.spec.packing == "concat_split":
            return _ConcatSplit(self.spec.seq_len, None if packed is None else _unb64(packed))
        bins = None if packed is None else [[_decode_piece(p) for p in b] for b in packed]
        return _FirstFit(self.spec.seq_len, self.bins, self.eos_id, bins)

    def _load(self, state: dict[str, Any], skip_batches: int) -> dict[str, int]:
        expected = {
            "version": STATE_VERSION,
            "packing": self.spec.packing,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "rows": self.rows,
            "bins": self.bins,
            "batches": skip_batches,
        }
        problems = [f"{key} {state.get(key)!r} -> {value!r}" for key, value in expected.items() if state.get(key) != value]
        if problems:
            raise ValueError("cannot resume the stream: " + "; ".join(problems))
        self._packer = self._new_packer(state["packer"])
        if self.spec.packing == "concat_split":
            self._pending = [_unb64(row) for row in state["pending"]]
        else:
            self._pending = [tuple(_decode_piece(p) for p in row) for row in state["pending"]]
        cursor = {key: int(state["cursor"][key]) for key in ("epoch", "file", "offset", "record")}
        return cursor

    def _remember(self) -> None:
        snapshot = _Snapshot(dict(self._cursor), self._packer.snapshot(), tuple(self._pending))
        with self._lock:
            self._history[self.batches_consumed] = snapshot
            while len(self._history) > self._history_size:
                self._history.popitem(last=False)

    def state_at(self, batches: int) -> dict[str, Any]:
        """JSON-ready stream position after ``batches`` batches."""
        with self._lock:
            snapshot = self._history.get(int(batches))
            known = (next(iter(self._history)), next(reversed(self._history))) if self._history else None
        if snapshot is None:
            raise RuntimeError(f"stream position after batch {batches} is not in the loader history {known}")
        if self.spec.packing == "concat_split":
            pending = [_b64(row) for row in snapshot.pending]
        else:
            pending = [[_encode_piece(p) for p in row] for row in snapshot.pending]
        return {
            "version": STATE_VERSION,
            "packing": self.spec.packing,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "rows": self.rows,
            "bins": self.bins,
            "batches": int(batches),
            "cursor": dict(snapshot.cursor),
            "packer": self._packer.encode(snapshot.packer),
            "pending": pending,
        }

    # -- iteration -------------------------------------------------------------

    def _next_document(self) -> tuple[dict[str, int], np.ndarray]:
        while not self._ready:
            chunk: list[tuple[dict[str, int], str]] = []
            chars = 0
            while len(chunk) < self.spec.read_ahead and chars < self.spec.read_ahead_chars:
                chunk.append(next(self._texts))
                chars += len(chunk[-1][1])
            for (cursor, _), ids in zip(chunk, self._encode([text for _, text in chunk]), strict=True):
                if len(ids) == 0:
                    continue
                tokens = np.append(np.asarray(ids, dtype=np.int32), np.int32(self.eos_id))
                if tokens.min() < 0 or tokens.max() >= self.spec.vocab_size:
                    raise ValueError(f"token id {int(tokens.max())} is outside the vocabulary of {self.spec.vocab_size}")
                self._ready.append((cursor, tokens))
        return self._ready.popleft()

    def _assemble(self, rows: list[Any]) -> dict[str, np.ndarray]:
        if self.spec.packing == "concat_split":
            windows = np.stack(rows)
            batch = {"input_ids": windows[:, :-1], "labels": windows[:, 1:]}
            return self._document_fields(batch) if self._document_fields is not None else batch
        packed = [first_fit_row(row, self.spec.seq_len, self.eos_id) for row in rows]
        keys = ("input_ids", "labels", "loss_mask", "segment_ids", "positions")
        if self._document_fields is None:
            keys = keys[:3]
        return {key: np.stack([row[key] for row in packed]) for key in keys}

    def __iter__(self) -> StreamLoader:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        while len(self._pending) < self.rows:
            cursor, tokens = self._next_document()
            self._pending.extend(self._packer.add(tokens))
            self._cursor = cursor
        rows, self._pending = self._pending[: self.rows], self._pending[self.rows :]
        batch = self._assemble(rows)
        self.batches_consumed += 1
        self._remember()
        return batch

    def close(self) -> None:
        self._texts.close()


# ---------------------------------------------------------------------------
# Multi-process positions
# ---------------------------------------------------------------------------


def gather_shard_states(local: dict[str, Any]) -> list[dict[str, Any]]:
    """Every shard's state, ordered by shard index (an all-gather across processes)."""
    import jax

    if jax.process_count() == 1:
        states = [local]
    else:
        from jax.experimental import multihost_utils

        payload = np.frombuffer(json.dumps(local, sort_keys=True).encode("utf-8"), dtype=np.uint8)
        sizes = np.asarray(multihost_utils.process_allgather(np.array([payload.size], np.int32))).reshape(-1)
        padded = np.zeros(int(sizes.max()), np.uint8)
        padded[: payload.size] = payload
        gathered = np.asarray(multihost_utils.process_allgather(padded))
        states = [json.loads(bytes(gathered[i, : int(n)]).decode("utf-8")) for i, n in enumerate(sizes)]
    by_shard: dict[int, dict[str, Any]] = {}
    for state in states:
        previous = by_shard.setdefault(int(state["shard_index"]), state)
        if previous != state:
            raise RuntimeError(f"processes reading stream shard {state['shard_index']} disagree on its position")
    count = int(local["shard_count"])
    if sorted(by_shard) != list(range(count)):
        raise RuntimeError(f"stream positions cover shards {sorted(by_shard)}, expected 0..{count - 1}")
    return [by_shard[i] for i in range(count)]


def stream_positions(loader: Any, batches: int) -> list[dict[str, Any]] | None:
    """Saved stream positions of a training loader (``None`` for map-style loaders)."""
    if not isinstance(loader, StreamLoader):
        return None
    return gather_shard_states(loader.state_at(batches))


def shard_state(saved: Sequence[dict[str, Any]] | None, shard_index: int) -> dict[str, Any] | None:
    """This shard's entry of the saved positions."""
    if saved is None:
        return None
    if not 0 <= shard_index < len(saved):
        raise ValueError(f"the checkpoint holds {len(saved)} stream shard positions; this process reads shard {shard_index}")
    return dict(saved[shard_index])


def read_documents(path: str | Path, text_field: str = "text") -> list[str]:
    """Non-empty texts of a JSONL file in order (what a single-shard stream reads)."""
    return [text for _, line in _jsonl_lines(str(path), 0) if (text := _json_text(line, text_field)) is not None]
