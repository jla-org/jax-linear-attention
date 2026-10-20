# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MaxText (Apache-2.0), src/maxtext/input_pipeline/dpo_utils.py (common-prefix
# prompt extraction, left-truncated prompts and right-truncated responses).
"""Preference pairs ``(prompt, chosen, rejected)`` for DPO-style training.

JSONL records (one object per line) in any of these forms:

* ``{"prompt_ids": [...], "chosen_ids": [...], "rejected_ids": [...]}``: token ids;
* ``{"prompt": "...", "chosen": "...", "rejected": "..."}``: text, encoded with a tokenizer
  (the prompt with the tokenizer's special tokens, e.g. BOS; responses without);
* ``{"chosen": ..., "rejected": ...}`` (text or ``*_ids``) without a prompt: full conversations
  whose longest common token prefix is the prompt (the Anthropic HH-RLHF layout).

:class:`PreferenceDataset` turns pair ``i`` into two right-padded rows of ``seq_len`` tokens,
row 0 = prompt + chosen and row 1 = prompt + rejected (+ ``eos_id`` when set)::

    input_ids [2, T]   labels [2, T] (next token)   loss_mask [2, T] (1 on response targets only)

Prompt targets and padding have ``loss_mask = 0``. Prompts longer than ``max_prompt_len`` keep
their last tokens; responses are cut to fit ``seq_len + 1`` tokens. :class:`PairRows` (the
loader transform) flattens a batch of ``P`` pairs to ``2P`` interleaved rows, so the rows of one
pair always sit next to each other (and on one device when each device holds an even number of
rows). Every sample is a pure function of its index, so resuming needs only the batch counter.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PairRows",
    "PreferenceDataset",
    "SyntheticPreferenceDataset",
    "load_preference_jsonl",
    "split_common_prefix",
]

Encoder = Callable[[str, bool], list[int]]  # (text, add_special_tokens) -> ids

MAX_RECORD_BYTES = 64 * 1024 * 1024


def split_common_prefix(chosen: Sequence[int], rejected: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """``(prompt, chosen_response, rejected_response)`` split at the longest common token prefix."""
    prefix = 0
    for a, b in zip(chosen, rejected):
        if a != b:
            break
        prefix += 1
    return list(chosen[:prefix]), list(chosen[prefix:]), list(rejected[prefix:])


def _ids(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise TypeError(f"{where} must be a list of integers")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string")
    return value


def parse_record(record: Mapping[str, Any], where: str, encode: Encoder | None) -> tuple[list[int], list[int], list[int]]:
    """One JSON record -> token ids ``(prompt, chosen, rejected)`` (see the module docstring)."""
    if not isinstance(record, Mapping):
        raise TypeError(f"{where}: each line must be a JSON object")
    if "chosen_ids" in record or "rejected_ids" in record:
        chosen = _ids(record.get("chosen_ids"), f"{where}.chosen_ids")
        rejected = _ids(record.get("rejected_ids"), f"{where}.rejected_ids")
        if "prompt_ids" in record:
            return _ids(record["prompt_ids"], f"{where}.prompt_ids"), chosen, rejected
        return split_common_prefix(chosen, rejected)
    if "chosen" not in record or "rejected" not in record:
        raise KeyError(f"{where}: needs chosen/rejected (text) or chosen_ids/rejected_ids (tokens)")
    if encode is None:
        raise ValueError(f"{where}: text records need a tokenizer (dpo.tokenizer)")
    chosen_text = _text(record["chosen"], f"{where}.chosen")
    rejected_text = _text(record["rejected"], f"{where}.rejected")
    if "prompt" in record:
        prompt = encode(_text(record["prompt"], f"{where}.prompt"), True)
        return prompt, encode(chosen_text, False), encode(rejected_text, False)
    return split_common_prefix(encode(chosen_text, True), encode(rejected_text, True))


def load_preference_jsonl(path: str | Path, *, encode: Encoder | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """All pairs of a JSONL file as int32 arrays ``(prompt, chosen, rejected)``."""
    source = Path(path)
    pairs = []
    with source.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError(f"{source}:{number}: record exceeds {MAX_RECORD_BYTES} bytes")
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{number}: invalid JSON: {exc}") from exc
            prompt, chosen, rejected = parse_record(record, f"{source}:{number}", encode)
            pairs.append(tuple(np.asarray(x, np.int32) for x in (prompt, chosen, rejected)))
    if not pairs:
        raise ValueError(f"{source} holds no preference pairs")
    return pairs


def hf_encoder(name: str, *, local_files_only: bool = False) -> Encoder:
    """Text encoder from a HuggingFace tokenizer (optional ``transformers`` dependency)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("text preference data needs the optional 'transformers' package") from exc
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)

    def encode(text: str, add_special_tokens: bool) -> list[int]:
        return list(tokenizer(text, add_special_tokens=add_special_tokens).input_ids)

    return encode


def _rows(prompt: np.ndarray, response: np.ndarray, seq_len: int, max_prompt_len: int, pad_id: int, eos_id: int | None):
    if eos_id is not None and (response.size == 0 or response[-1] != eos_id):
        response = np.concatenate([response, np.asarray([eos_id], np.int32)])
    prompt = prompt[-max_prompt_len:]
    response = response[: seq_len + 1 - prompt.size]
    tokens = np.concatenate([prompt, response])
    input_ids = np.full((seq_len,), pad_id, np.int32)
    labels = np.full((seq_len,), pad_id, np.int32)
    mask = np.zeros((seq_len,), np.int32)
    n = tokens.size - 1
    input_ids[:n] = tokens[:-1]
    labels[:n] = tokens[1:]
    mask[prompt.size - 1 : n] = 1  # targets at positions >= len(prompt) - 1 are response tokens
    return input_ids, labels, mask


class PreferenceDataset:
    """Map-style dataset of preference pairs (see the module docstring)."""

    def __init__(
        self,
        pairs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        seq_len: int,
        *,
        max_prompt_len: int,
        pad_id: int = 0,
        eos_id: int | None = None,
        vocab_size: int | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        if not 1 <= max_prompt_len <= seq_len - 1:
            raise ValueError(f"max_prompt_len must be in [1, seq_len - 1 = {seq_len - 1}], got {max_prompt_len}")
        if not pairs:
            raise ValueError("a preference dataset needs at least one pair")
        self.seq_len = int(seq_len)
        self.max_prompt_len = int(max_prompt_len)
        self.pad_id = int(pad_id)
        self.eos_id = None if eos_id is None else int(eos_id)
        self.pairs = [tuple(np.asarray(x, np.int32).reshape(-1) for x in pair) for pair in pairs]
        for index, (prompt, chosen, rejected) in enumerate(self.pairs):
            if prompt.size == 0:
                raise ValueError(f"pair {index}: empty prompt (the first response token needs a context token, e.g. BOS)")
            if (chosen.size == 0 or rejected.size == 0) and self.eos_id is None:
                raise ValueError(f"pair {index}: empty chosen or rejected response")
            if vocab_size is not None:
                for part in (prompt, chosen, rejected, np.asarray([self.pad_id] + ([self.eos_id] if self.eos_id is not None else []))):
                    if part.size and (part.min() < 0 or part.max() >= vocab_size):
                        raise ValueError(f"pair {index}: token ids must lie in [0, {vocab_size})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if not 0 <= index < len(self.pairs):
            raise IndexError(f"index {index} out of range [0, {len(self.pairs)})")
        prompt, chosen, rejected = self.pairs[index]
        rows = [_rows(prompt, r, self.seq_len, self.max_prompt_len, self.pad_id, self.eos_id) for r in (chosen, rejected)]
        return {name: np.stack([row[k] for row in rows]) for k, name in enumerate(("input_ids", "labels", "loss_mask"))}


class SyntheticPreferenceDataset(PreferenceDataset):
    """Learnable toy preferences: the chosen response continues the prompt's token pattern.

    Prompt ``p`` of random tokens; chosen = ``(p[-1] + 1 + k) % V`` for ``k = 0..`` (counting
    up from the last prompt token), rejected = random tokens. A policy that prefers counting
    gets reward accuracy 1. Deterministic in ``seed``; ``offset`` gives a disjoint split.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        *,
        num_pairs: int,
        seed: int = 0,
        offset: int = 0,
        max_prompt_len: int | None = None,
        pad_id: int = 0,
    ) -> None:
        if vocab_size < 4 or num_pairs < 1:
            raise ValueError("vocab_size >= 4 and num_pairs >= 1 are required")
        max_prompt_len = max_prompt_len or seq_len // 2
        prompt_len = max(1, min(max_prompt_len, seq_len // 2))
        response_len = max(1, seq_len + 1 - prompt_len)
        pairs = []
        for i in range(offset, offset + num_pairs):
            rng = np.random.default_rng([seed, i])
            prompt = rng.integers(1, vocab_size, prompt_len)
            chosen = (prompt[-1] + 1 + np.arange(response_len)) % vocab_size
            rejected = rng.integers(1, vocab_size, response_len)
            pairs.append((prompt, chosen, rejected))
        super().__init__(pairs, seq_len, max_prompt_len=max_prompt_len, pad_id=pad_id, vocab_size=vocab_size)


class PairRows:
    """Loader transform: ``[P, 2, T]`` pair batches -> ``[2P, T]`` rows (chosen, rejected interleaved)."""

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.reshape(-1, *value.shape[2:]) for key, value in batch.items()}


def iter_pairs(records: Iterable[Mapping[str, Any]], encode: Encoder | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """In-memory records -> int32 pairs (same rules as :func:`load_preference_jsonl`)."""
    return [
        tuple(np.asarray(x, np.int32) for x in parse_record(record, f"record {i}", encode))
        for i, record in enumerate(records)
    ]
