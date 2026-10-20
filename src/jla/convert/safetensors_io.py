# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal safetensors reader and writer on NumPy (bfloat16 and float8 through ``ml_dtypes``).

The format: an 8-byte little-endian header length, a JSON header mapping every
tensor name to ``{"dtype", "shape", "data_offsets"}`` (offsets into the byte
buffer that follows) plus an optional ``"__metadata__"`` string map, then the
raw little-endian C-order data. Reading memory-maps the file, so a tensor is
copied only when it is used. Sharded checkpoints (``*.safetensors.index.json``
with a ``weight_map``) are read and written like Hugging Face does.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import ml_dtypes
import numpy as np

__all__ = ["SafetensorsFile", "load_checkpoint_dir", "read_safetensors", "save_checkpoint_dir", "write_safetensors"]

_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype(np.float64), "F32": np.dtype(np.float32), "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn), "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "I64": np.dtype(np.int64), "I32": np.dtype(np.int32), "I16": np.dtype(np.int16), "I8": np.dtype(np.int8),
    "U64": np.dtype(np.uint64), "U32": np.dtype(np.uint32), "U16": np.dtype(np.uint16), "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}  # fmt: skip
_CODES = {dtype: code for code, dtype in _DTYPES.items()}
_MAX_HEADER = 100 * 1024 * 1024


class SafetensorsFile(Mapping[str, np.ndarray]):
    """Read-only mapping ``name -> array`` over one memory-mapped safetensors file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"{self.path}: not a safetensors file (truncated header length)")
            (size,) = struct.unpack("<Q", raw)
            if size > _MAX_HEADER:
                raise ValueError(f"{self.path}: safetensors header of {size} bytes is too large")
            header = json.loads(stream.read(size))
        self.metadata: dict[str, str] = dict(header.pop("__metadata__", None) or {})
        self._entries: dict[str, tuple[np.dtype, tuple[int, ...], int, int]] = {}
        data_bytes = self.path.stat().st_size - 8 - size
        for name, entry in header.items():
            code = entry["dtype"]
            if code not in _DTYPES:
                raise ValueError(f"{self.path}: tensor {name!r} has unsupported dtype {code}")
            dtype, shape = _DTYPES[code], tuple(int(d) for d in entry["shape"])
            begin, end = (int(x) for x in entry["data_offsets"])
            if end - begin != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize or not 0 <= begin <= end <= data_bytes:
                raise ValueError(f"{self.path}: tensor {name!r} has inconsistent offsets {begin}..{end} for {shape} {code}")
            self._entries[name] = (dtype, shape, begin, end)
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=8 + size) if data_bytes else None

    def __getitem__(self, name: str) -> np.ndarray:
        dtype, shape, begin, end = self._entries[name]
        if begin == end:
            return np.zeros(shape, dtype)
        return np.asarray(self._data[begin:end]).view(dtype).reshape(shape)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def shape(self, name: str) -> tuple[int, ...]:
        return self._entries[name][1]


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Every tensor of ``path`` as an owned NumPy array."""
    tensors = SafetensorsFile(path)
    return {name: np.array(tensors[name]) for name in tensors}


def write_safetensors(path: str | Path, tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str] | None = None) -> Path:
    """Write ``tensors`` to ``path`` atomically (temporary file, fsync, rename)."""
    destination = Path(path)
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    arrays = []
    # Larger items first keeps every offset a multiple of its itemsize (the reference writer does the same).
    for name in sorted(tensors, key=lambda n: (-np.asarray(tensors[n]).dtype.itemsize, n)):
        array = np.ascontiguousarray(tensors[name])
        if array.dtype not in _CODES:
            raise TypeError(f"tensor {name!r}: dtype {array.dtype} cannot be stored in safetensors")
        if array.dtype.byteorder == ">":
            raise TypeError(f"tensor {name!r}: big-endian arrays are not supported")
        header[name] = {"dtype": _CODES[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
        arrays.append(array)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # 8-byte aligned data, as the reference writer pads
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(struct.pack("<Q", len(blob)))
            stream.write(blob)
            for array in arrays:
                stream.write(memoryview(array.reshape(-1).view(np.uint8)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return destination


def load_checkpoint_dir(directory: str | Path) -> dict[str, np.ndarray]:
    """Lazily mapped tensors of a Hugging Face checkpoint directory (single file or sharded).

    Reads ``model.safetensors`` or else the shards listed in ``model.safetensors.index.json`` (the
    order transformers' ``from_pretrained`` uses); the arrays are memory-mapped views (copy them
    before the files go away).
    """
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    single = directory / "model.safetensors"
    if index.exists() and not single.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = {name: SafetensorsFile(directory / name) for name in sorted(set(weight_map.values()))}
        out = {}
        for tensor, filename in weight_map.items():
            if tensor not in files[filename]:
                raise ValueError(f"{index}: {tensor!r} is not in {filename}")
            out[tensor] = files[filename][tensor]
        return out
    if single.exists():
        tensors = SafetensorsFile(single)
        return {name: tensors[name] for name in tensors}
    raise FileNotFoundError(f"{directory} has neither model.safetensors nor model.safetensors.index.json")


def save_checkpoint_dir(
    directory: str | Path,
    tensors: Mapping[str, np.ndarray],
    *,
    max_shard_bytes: int = 5 * 1024**3,
    metadata: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write ``model.safetensors`` or, above ``max_shard_bytes``, numbered shards plus an index.

    Weight files of an earlier checkpoint in ``directory`` (``model.safetensors``, the index and
    ``model-*-of-*.safetensors`` shards) are removed once the new files are written, so no loader
    can pick up stale weights.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"format": "pt", **(metadata or {})}  # transformers refuses files without format=pt
    names = sorted(tensors)
    total = sum(int(np.asarray(tensors[n]).nbytes) for n in names)
    stale = {p for p in directory.glob("model-*-of-*.safetensors")}
    stale |= {p for p in (directory / "model.safetensors", directory / "model.safetensors.index.json") if p.exists()}
    if total <= max_shard_bytes:
        paths = [write_safetensors(directory / "model.safetensors", tensors, metadata)]
        _remove_stale(stale, paths)
        return paths
    shards: list[list[str]] = [[]]
    size = 0
    for name in names:
        nbytes = int(np.asarray(tensors[name]).nbytes)
        if shards[-1] and size + nbytes > max_shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += nbytes
    count = len(shards)
    weight_map, paths = {}, []
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{count:05d}.safetensors"
        paths.append(write_safetensors(directory / filename, {n: tensors[n] for n in shard}, metadata))
        weight_map.update({n: filename for n in shard})
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    _remove_stale(stale, paths + [directory / "model.safetensors.index.json"])
    return paths


def _remove_stale(stale: set[Path], written: list[Path]) -> None:
    keep = {p.resolve() for p in written}
    for path in stale:
        if path.resolve() not in keep:
            path.unlink(missing_ok=True)
