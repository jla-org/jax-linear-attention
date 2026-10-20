# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CIDX: a memory-mapped token corpus format.

A corpus ``prefix`` is two files:

* ``prefix.idx`` (little endian): magic ``0x58444943`` ("CIDX") u32, version
  ``1`` u32, dtype code u8, document count i64, token count i64, then
  ``documents + 1`` int64 token offsets (``offsets[0] == 0``,
  ``offsets[-1] == tokens``, non-decreasing);
* ``prefix.bin``: the raw tokens of all documents back to back (writers
  append EOS to every document).

Reads are zero-copy views into the mapping unless a window wraps around the
end of the stream.
"""

from __future__ import annotations

import mmap
import os
import struct
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CIDX_MAGIC",
    "CIDX_VERSION",
    "DTYPE_CODES",
    "CidxWriter",
    "MMapIndexedDataset",
    "check_vocab",
    "resolve_record_split",
    "write_cidx",
]

CIDX_MAGIC = 0x58444943
CIDX_VERSION = 1
_HEADER = struct.Struct("<IIBqq")
DTYPE_CODES: dict[int, Any] = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float16,
    7: np.float32,
    8: np.float64,
    9: np.uint16,
}
_WRITER_DTYPES = {"uint16": (np.uint16, 9), "int32": (np.int32, 4)}


class MMapIndexedDataset:
    """Read-only view of a CIDX corpus."""

    def __init__(self, path: str) -> None:
        self._path = str(path)
        idx_path, bin_path = self._path + ".idx", self._path + ".bin"
        with open(idx_path, "rb") as stream:
            header = stream.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError(f"truncated index header in {idx_path}")
            magic, version, dtype_code, documents, tokens = _HEADER.unpack(header)
            if magic != CIDX_MAGIC:
                raise ValueError(f"bad magic in {idx_path}: 0x{magic:08X}")
            if version != CIDX_VERSION:
                raise ValueError(f"unsupported CIDX version {version} in {idx_path}")
            if documents <= 0 or tokens <= 0:
                raise ValueError(f"CIDX counts must be positive; got documents={documents}, tokens={tokens}")
            if dtype_code not in DTYPE_CODES:
                raise ValueError(f"unknown CIDX dtype code {dtype_code}")
            offsets = stream.read((documents + 1) * 8)
            if len(offsets) != (documents + 1) * 8:
                raise ValueError(f"truncated offset array in {idx_path}")
        self._num_documents = int(documents)
        self._num_tokens = int(tokens)
        self._dtype = np.dtype(DTYPE_CODES[dtype_code])
        self.offsets = np.frombuffer(offsets, dtype="<i8").copy()
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != tokens or np.any(np.diff(self.offsets) < 0):
            raise ValueError(f"invalid document offsets in {idx_path}: expected a monotonic range from 0 to {tokens}")
        expected = tokens * self._dtype.itemsize
        if os.path.getsize(bin_path) < expected:
            raise ValueError(f"{bin_path} is smaller than the {expected} bytes its index declares")
        self._bin_file = open(bin_path, "rb")
        self._mmap = mmap.mmap(self._bin_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._advise()
        self._data = np.frombuffer(self._mmap, dtype=self._dtype.newbyteorder("<"))[: self._num_tokens]

    def _advise(self) -> None:
        # Training reads quasi-random windows: disable sequential read-ahead and
        # ask for huge pages where the kernel supports them.
        names = ("MADV_RANDOM", "MADV_HUGEPAGE") if sys.platform.startswith("linux") else ("MADV_RANDOM",)
        for name in names:
            flag = getattr(mmap, name, None)
            if flag is None:
                continue
            try:
                self._mmap.madvise(flag)
            except (OSError, ValueError):
                pass

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_documents(self) -> int:
        return self._num_documents

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __len__(self) -> int:
        return self._num_tokens

    def get_document(self, index: int) -> np.ndarray:
        if index < 0:
            index += self._num_documents
        if not 0 <= index < self._num_documents:
            raise IndexError(f"document index {index} out of range")
        return self._data[int(self.offsets[index]) : int(self.offsets[index + 1])]

    def get_chunk(self, start: int, length: int) -> np.ndarray:
        """``length`` tokens from ``start``, wrapping around the end of the stream."""
        total = self._num_tokens
        if length <= 0:
            return self._data[0:0]
        start = int(start) % total
        if start + length <= total:
            return self._data[start : start + length]
        out = np.empty(length, dtype=self._data.dtype)
        filled = 0
        position = start
        while filled < length:
            take = min(length - filled, total - position)
            out[filled : filled + take] = self._data[position : position + take]
            filled += take
            position = 0
        return out

    def close(self) -> None:
        self._data = None  # type: ignore[assignment]
        if getattr(self, "_mmap", None) is not None:
            try:
                self._mmap.close()
            except (BufferError, ValueError):
                pass  # a view is still alive; the mapping closes with it
            self._mmap = None  # type: ignore[assignment]
        if getattr(self, "_bin_file", None) is not None:
            self._bin_file.close()
            self._bin_file = None  # type: ignore[assignment]

    def __enter__(self) -> MMapIndexedDataset:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, str]:
        return {"path": self._path}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(state["path"])


def check_vocab(path: str, vocab_size: int, *, sample_tokens: int = 1 << 22, windows: int = 64) -> None:
    """Raise if a sample of the corpus holds token ids outside ``[0, vocab_size)``.

    Reads the first ``sample_tokens`` tokens plus ``windows`` deterministic windows of the same total
    size spread over the stream. A tokenizer whose ids exceed the model vocabulary then fails at
    startup, instead of turning every training step non-finite.
    """
    with MMapIndexedDataset(path) as corpus:
        total = corpus.num_tokens
        if total == 0:
            return
        width = max(sample_tokens // windows, 1)
        starts = np.random.default_rng(0).integers(0, max(total - width, 1), size=windows)
        spans = [(0, min(total, sample_tokens))] + [(int(s), min(width, total)) for s in starts]
        for start, length in spans:
            chunk = np.asarray(corpus.get_chunk(start, length))
            low, high = int(chunk.min()), int(chunk.max())
            if low < 0 or high >= vocab_size:
                bad = low if low < 0 else high
                raise ValueError(
                    f"{path}: token id {bad} (near token {start}) is outside the model vocabulary "
                    f"[0, {vocab_size}); set model.vocab_size to the tokenizer's vocabulary size"
                )


def resolve_record_split(path: str, eval_holdout_records: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Train/eval document ranges with the last ``eval_holdout_records`` documents held out.

    Both sides keep at least two documents: a next-token window needs one
    token beyond its inputs, and that token must come from the same side.
    """
    if isinstance(eval_holdout_records, bool) or not isinstance(eval_holdout_records, (int, np.integer)):
        raise TypeError("eval_holdout_records must be an integer")
    holdout = int(eval_holdout_records)
    if holdout < 2:
        raise ValueError("eval_holdout_records must be at least 2")
    with MMapIndexedDataset(path) as corpus:
        documents = corpus.num_documents
    if holdout > documents - 2:
        raise ValueError(
            f"eval_holdout_records must leave at least 2 training records; got holdout={holdout}, documents={documents}"
        )
    boundary = documents - holdout
    return (0, boundary), (boundary, documents)


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some network filesystems reject directory fsync
    finally:
        os.close(fd)


class CidxWriter:
    """Stream documents into a CIDX corpus; files appear atomically on :meth:`commit`.

    Use as a context manager: the corpus is committed when the block exits
    normally and the temporary files are removed if it raises.
    """

    def __init__(self, prefix: str | Path, *, dtype: str = "int32", buffer_bytes: int = 16 * 1024 * 1024) -> None:
        if dtype not in _WRITER_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_WRITER_DTYPES)}, got {dtype!r}")
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self._np_dtype = np.dtype(_WRITER_DTYPES[dtype][0]).newbyteorder("<")
        self._code = _WRITER_DTYPES[dtype][1]
        self._limit = np.iinfo(self._np_dtype).max
        fd, tmp = tempfile.mkstemp(dir=self.prefix.parent, prefix=f".{self.prefix.name}.", suffix=".bin.tmp")
        self._bin_tmp = Path(tmp)
        self._bin = os.fdopen(fd, "wb", buffering=buffer_bytes)
        self._offsets = [0]
        self._closed = False

    @property
    def num_documents(self) -> int:
        return len(self._offsets) - 1

    @property
    def num_tokens(self) -> int:
        return self._offsets[-1]

    def add(self, tokens: Iterable[int] | np.ndarray) -> None:
        """Append one document (empty documents are rejected)."""
        array = np.asarray(tokens)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("a document must be a non-empty 1-D token sequence")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"tokens must be integers, got {array.dtype}")
        if int(array.min()) < 0 or int(array.max()) > self._limit:
            raise ValueError(f"token ids must be in [0, {self._limit}] for {self._np_dtype}")
        self._bin.write(array.astype(self._np_dtype, copy=False).tobytes())
        self._offsets.append(self._offsets[-1] + int(array.size))

    def commit(self) -> tuple[int, int]:
        """Write the index, fsync both files and rename them into place."""
        if self._closed:
            raise RuntimeError("CidxWriter is closed")
        if self.num_documents == 0:
            self.abort()
            raise ValueError("cannot commit a CIDX corpus without documents")
        self._bin.flush()
        os.fsync(self._bin.fileno())
        self._bin.close()
        fd, idx_tmp = tempfile.mkstemp(dir=self.prefix.parent, prefix=f".{self.prefix.name}.", suffix=".idx.tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_HEADER.pack(CIDX_MAGIC, CIDX_VERSION, self._code, self.num_documents, self.num_tokens))
                stream.write(np.asarray(self._offsets, dtype="<i8").tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(self._bin_tmp, f"{self.prefix}.bin")
            os.replace(idx_tmp, f"{self.prefix}.idx")
            _fsync_directory(self.prefix.parent)
        except BaseException:
            Path(idx_tmp).unlink(missing_ok=True)
            self._bin_tmp.unlink(missing_ok=True)
            raise
        finally:
            self._closed = True
        return self.num_documents, self.num_tokens

    def abort(self) -> None:
        if not self._closed:
            self._bin.close()
            self._bin_tmp.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> CidxWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()


def write_cidx(prefix: str | Path, documents: Iterable[Iterable[int] | np.ndarray], *, dtype: str = "int32") -> tuple[int, int]:
    """Write documents to ``prefix.{bin,idx}``; returns ``(documents, tokens)``."""
    writer = CidxWriter(prefix, dtype=dtype)
    try:
        for document in documents:
            writer.add(document)
    except BaseException:
        writer.abort()
        raise
    return writer.commit()
