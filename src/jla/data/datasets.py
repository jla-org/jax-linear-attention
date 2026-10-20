# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Map-style token datasets: ``dataset[i] -> {"input_ids": [T], "labels": [T]}`` (int32).

* :class:`TokenWindowDataset`: windows of ``seq_len + 1`` tokens at stride
  ``seq_len`` over a CIDX corpus, visited in a fixed pseudo-random order (the
  cubic "magic prime" permutation), optionally restricted to a document range;
  ``exact_packed_records`` reads one pre-packed document per sample;
* :class:`BlendedDataset`: weighted mixture of corpora, each sample's source
  and window chosen by a splitmix64 hash of its index;
* :class:`SyntheticDataset`: deterministic generated tokens for smoke tests
  and throughput runs (``markov`` is learnable, ``uniform`` is not).

Every sample is a pure function of its index, so any process can produce any
sample and resuming needs only the sample counter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from jla.data.cidx import MMapIndexedDataset

__all__ = [
    "BlendedDataset",
    "SyntheticDataset",
    "TokenWindowDataset",
    "compute_golden_factor",
    "compute_magic_prime",
    "parse_blend_spec",
]

_MASK64 = 0xFFFFFFFFFFFFFFFF


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def compute_magic_prime(num_tokens: int, seq_len: int) -> int:
    """Largest prime ``p < num_tokens // seq_len`` with ``p % 3 == 2``.

    For such primes ``i -> i**3 mod p`` is a bijection, so the cubic walk
    visits every window exactly once per epoch.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    candidate = num_tokens // seq_len - 1
    while candidate >= 2:
        if candidate % 3 == 2 and _is_prime(candidate):
            return candidate
        candidate -= 1
    raise ValueError(f"corpus of {num_tokens} tokens is too small for seq_len={seq_len}")


def compute_golden_factor(magic_prime: int) -> int:
    return max(1, int(magic_prime * (math.sqrt(5) - 1) / 2))


def _sample(tokens: np.ndarray) -> dict[str, np.ndarray]:
    window = tokens.astype(np.int32, copy=False)
    return {"input_ids": window[:-1], "labels": window[1:]}


class TokenWindowDataset:
    """Next-token windows over one CIDX corpus."""

    def __init__(
        self,
        path: str,
        seq_len: int,
        *,
        record_range: tuple[int, int] | None = None,
        exact_packed_records: bool = False,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.corpus = MMapIndexedDataset(path)
        self.seq_len = int(seq_len)
        self.exact_packed_records = bool(exact_packed_records)
        self._restricted = record_range is not None
        if record_range is None:
            start, end = 0, self.corpus.num_documents
        else:
            if (
                not isinstance(record_range, tuple)
                or len(record_range) != 2
                or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in record_range)
            ):
                raise TypeError("record_range must be a (start, end) integer tuple")
            start, end = (int(v) for v in record_range)
        if not 0 <= start < end <= self.corpus.num_documents:
            raise ValueError(
                f"record_range must satisfy 0 <= start < end <= {self.corpus.num_documents}; got ({start}, {end})"
            )
        self.record_range = (start, end)
        self.token_range = (int(self.corpus.offsets[start]), int(self.corpus.offsets[end]))
        total = self.token_range[1] - self.token_range[0]
        if self.exact_packed_records:
            lengths = np.diff(self.corpus.offsets[start : end + 1])
            if not np.all(lengths == self.seq_len):
                raise ValueError(f"exact packed records require every document to have length {self.seq_len}")
            # In a range view the final record only supplies the last target, so
            # no window ever borrows a label from outside the range.
            self._length = (end - start) - int(self._restricted)
            if self._length <= 0:
                raise ValueError("an exact packed record range needs at least 2 records")
            self._prime = self._factor = 0
        else:
            self._prime = compute_magic_prime(total, self.seq_len)
            self._factor = compute_golden_factor(self._prime)
            self._length = self._prime

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range [0, {self._length})")
        if self.exact_packed_records:
            window = index
        else:
            i = index + 1
            window = (self._factor * i * i * i) % self._prime
        position = self.token_range[0] + window * self.seq_len
        if self._restricted and position + self.seq_len + 1 > self.token_range[1]:
            raise RuntimeError(f"window [{position}, {position + self.seq_len + 1}) escapes {self.token_range}")
        return _sample(self.corpus.get_chunk(position, self.seq_len + 1))


def parse_blend_spec(spec: str) -> tuple[list[str], list[float]]:
    """``"a:0.7,b:0.3"`` -> (paths, weights); a ``.bin`` suffix is stripped."""
    paths: list[str] = []
    weights: list[float] = []
    for entry in (item.strip() for item in spec.split(",")):
        if ":" not in entry:
            raise ValueError(f"blend entry {entry!r} must be path:weight")
        path, weight = entry.rsplit(":", 1)
        paths.append(path[: -len(".bin")] if path.endswith(".bin") else path)
        weights.append(float(weight))
    return paths, weights


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


class BlendedDataset:
    """Weighted mixture of CIDX corpora; non-overlapping windows of ``seq_len + 1`` tokens."""

    def __init__(self, paths: Sequence[str], weights: Sequence[float], seq_len: int, *, seed: int = 0) -> None:
        if not paths or len(paths) != len(weights):
            raise ValueError("paths and weights must be non-empty and of equal length")
        w = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("blend weights must be finite and > 0")
        self.corpora = [MMapIndexedDataset(path) for path in paths]
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.cumulative = np.cumsum(w / w.sum())
        self.windows = [corpus.num_tokens // (self.seq_len + 1) for corpus in self.corpora]
        small = [path for path, n in zip(paths, self.windows) if n == 0]
        if small:
            raise ValueError(f"corpora shorter than one window of {self.seq_len + 1} tokens: {small}")
        self._length = int(sum(self.windows))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range [0, {self._length})")
        h = _splitmix64((self.seed * 0x9E3779B97F4A7C15 + index) & _MASK64)
        uniform = (h >> 11) / float(1 << 53)
        source = min(int(np.searchsorted(self.cumulative, uniform, side="right")), len(self.corpora) - 1)
        window = _splitmix64(h) % self.windows[source]
        tokens = self.corpora[source].get_chunk(window * (self.seq_len + 1), self.seq_len + 1)
        return _sample(tokens)


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    z = values.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class SyntheticDataset:
    """Deterministic generated token windows.

    ``markov``: each token follows a fixed random permutation of its
    predecessor with probability ``follow_prob`` and is uniform otherwise (a
    bigram process a small model learns within tens of steps). ``uniform``:
    i.i.d. tokens. With ``doc_length > 0`` every ``doc_length``-th token of the
    stream is ``eos_id``, which exercises document masking. ``offset`` shifts the
    sample ids: an evaluation set with ``offset = num_samples`` follows the same
    chain as the training set but shares no window with it.
    """

    _RUN = 32  # the permutation chain restarts at least every _RUN tokens

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        *,
        seed: int = 0,
        kind: str = "markov",
        follow_prob: float = 0.9,
        num_samples: int = 65_536,
        eos_id: int | None = None,
        doc_length: int = 0,
        offset: int = 0,
    ) -> None:
        if kind not in ("markov", "uniform"):
            raise ValueError(f"kind must be 'markov' or 'uniform', got {kind!r}")
        if vocab_size < 2 or seq_len < 1 or num_samples < 1 or offset < 0:
            raise ValueError("vocab_size >= 2, seq_len >= 1, num_samples >= 1 and offset >= 0 are required")
        if not 0.0 <= follow_prob <= 1.0:
            raise ValueError("follow_prob must be in [0, 1]")
        if doc_length and (eos_id is None or not 0 <= eos_id < vocab_size):
            raise ValueError("doc_length needs an eos_id in [0, vocab_size)")
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.kind = kind
        self.follow_prob = float(follow_prob)
        self.num_samples = int(num_samples)
        self.eos_id = eos_id
        self.doc_length = int(doc_length)
        self.offset = int(offset)
        self._powers: np.ndarray | None = None

    def _chain(self) -> np.ndarray:
        """``table[k][t]``: the seeded permutation applied ``k`` times to ``t`` (built lazily)."""
        if self._powers is None:
            permutation = np.random.default_rng(self.seed).permutation(self.vocab_size)
            powers = np.empty((self._RUN, self.vocab_size), dtype=np.int64)
            powers[0] = np.arange(self.vocab_size)
            for k in range(1, self._RUN):
                powers[k] = permutation[powers[k - 1]]
            self._powers = powers
        return self._powers

    def __getstate__(self) -> dict[str, object]:
        return {**self.__dict__, "_powers": None}  # rebuilt in the worker instead of pickled

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < self.num_samples:
            raise IndexError(f"index {index} out of range [0, {self.num_samples})")
        index += self.offset
        length = self.seq_len + 1
        salt = np.uint64(_splitmix64(self.seed & _MASK64))
        counters = salt ^ (np.uint64(index * length) + np.arange(length, dtype=np.uint64))
        bits = _splitmix64_array(counters)
        fresh = (bits % np.uint64(self.vocab_size)).astype(np.int64)
        if self.kind == "uniform":
            tokens = fresh
        else:
            position = np.arange(length)
            uniform = (bits >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            restart = (uniform >= self.follow_prob) | (position % self._RUN == 0)
            last = np.maximum.accumulate(np.where(restart, position, 0))
            tokens = self._chain()[position - last, fresh[last]]
        if self.doc_length:
            stream_position = index * length + np.arange(length)
            tokens = np.where(stream_position % self.doc_length == self.doc_length - 1, self.eos_id, tokens)
        return _sample(tokens)
