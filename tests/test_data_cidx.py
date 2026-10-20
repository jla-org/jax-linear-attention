# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CIDX format, record splits, datasets, packing fields and the tokenizer CLI."""

from __future__ import annotations

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest

from jla.data.cidx import CidxWriter, MMapIndexedDataset, resolve_record_split, write_cidx
from jla.data.datasets import BlendedDataset, SyntheticDataset, TokenWindowDataset, compute_magic_prime, parse_blend_spec
from jla.data.make_data import tokenize_jsonl
from jla.data.packing import DocumentFields, positions_from_segment_ids, segment_ids_from_eos


def _corpus(prefix: Path, *, records: int = 10, record_length: int = 4) -> str:
    tokens = np.arange(records * record_length).reshape(records, record_length)
    write_cidx(prefix, list(tokens))
    return str(prefix)


def _observed(dataset) -> set[int]:
    seen: set[int] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        seen.update(sample["input_ids"].tolist())
        seen.update(sample["labels"].tolist())
    return seen


def test_writer_and_reader_roundtrip(tmp_path: Path) -> None:
    documents = [[1, 2, 3], [4], [5, 6]]
    assert write_cidx(tmp_path / "c", documents, dtype="uint16") == (3, 6)
    with MMapIndexedDataset(str(tmp_path / "c")) as corpus:
        assert corpus.num_documents == 3 and corpus.num_tokens == 6 and corpus.dtype == np.uint16
        assert corpus.get_document(1).tolist() == [4] and corpus.get_document(-1).tolist() == [5, 6]
        assert corpus.get_chunk(4, 4).tolist() == [5, 6, 1, 2]  # wraps around the stream
    assert pickle.loads(pickle.dumps(MMapIndexedDataset(str(tmp_path / "c")))).num_tokens == 6
    with pytest.raises(ValueError, match="token ids"):
        write_cidx(tmp_path / "bad", [[70_000]], dtype="uint16")
    assert not (tmp_path / "bad.bin").exists()
    with pytest.raises(ValueError, match="without documents"):
        with CidxWriter(tmp_path / "empty"):
            pass


def test_tail_record_split_is_deterministic_and_validated(tmp_path: Path) -> None:
    path = _corpus(tmp_path / "corpus")
    assert resolve_record_split(path, 4) == ((0, 6), (6, 10))
    assert resolve_record_split(path, np.int64(4)) == ((0, 6), (6, 10))
    with pytest.raises(TypeError, match="integer"):
        resolve_record_split(path, True)
    for holdout in (0, 1):
        with pytest.raises(ValueError, match="at least 2"):
            resolve_record_split(path, holdout)
    with pytest.raises(ValueError, match="leave at least 2 training records"):
        resolve_record_split(path, 9)
    corrupt = Path(_corpus(tmp_path / "corrupt") + ".idx")
    payload = bytearray(corrupt.read_bytes())
    struct.pack_into("<q", payload, 25 + 2 * 8, 3)
    corrupt.write_bytes(payload)
    with pytest.raises(ValueError, match="invalid document offsets"):
        resolve_record_split(str(tmp_path / "corrupt"), 4)


def test_exact_packed_views_never_cross_the_split(tmp_path: Path) -> None:
    path = _corpus(tmp_path / "packed")
    train_range, eval_range = resolve_record_split(path, 4)
    train = TokenWindowDataset(path, 4, record_range=train_range, exact_packed_records=True)
    evaluation = TokenWindowDataset(path, 4, record_range=eval_range, exact_packed_records=True)
    assert train.token_range == (0, 24) and evaluation.token_range == (24, 40)
    assert len(train) == 5 and len(evaluation) == 3
    train_tokens, eval_tokens = _observed(train), _observed(evaluation)
    assert max(train_tokens) < min(eval_tokens)


def test_window_views_stay_inside_their_records(tmp_path: Path) -> None:
    path = _corpus(tmp_path / "flat")
    train_range, eval_range = resolve_record_split(path, 4)
    train = TokenWindowDataset(path, 2, record_range=train_range)
    evaluation = TokenWindowDataset(path, 2, record_range=eval_range)
    assert _observed(train).isdisjoint(_observed(evaluation))
    assert all(train.token_range[0] <= t < train.token_range[1] for t in _observed(train))
    windows = {int(train[i]["input_ids"][0]) for i in range(len(train))}
    assert len(windows) == len(train)  # the cubic walk is a permutation
    with pytest.raises(ValueError, match="0 <= start < end"):
        TokenWindowDataset(path, 2, record_range=(3, 3))
    with pytest.raises(TypeError, match="integer tuple"):
        TokenWindowDataset(path, 2, record_range=[0, 6])  # type: ignore[arg-type]
    assert compute_magic_prime(1000, 10) % 3 == 2


def test_blend_is_deterministic_and_weighted(tmp_path: Path) -> None:
    write_cidx(tmp_path / "a", [np.zeros(4000, np.int32) + 1])
    write_cidx(tmp_path / "b", [np.zeros(4000, np.int32) + 2])
    paths, weights = parse_blend_spec(f"{tmp_path / 'a'}.bin:3,{tmp_path / 'b'}:1")
    assert paths == [str(tmp_path / "a"), str(tmp_path / "b")] and weights == [3.0, 1.0]
    blend = BlendedDataset(paths, weights, 15, seed=3)
    sources = [int(blend[i]["input_ids"][0]) for i in range(len(blend))]
    assert sources == [int(BlendedDataset(paths, weights, 15, seed=3)[i]["input_ids"][0]) for i in range(len(blend))]
    assert 0.65 < sources.count(1) / len(sources) < 0.85


def test_synthetic_data_is_deterministic_learnable_and_picklable() -> None:
    data = SyntheticDataset(64, 128, seed=1, num_samples=16)
    first = data[3]
    np.testing.assert_array_equal(first["input_ids"], data[3]["input_ids"])
    np.testing.assert_array_equal(first["input_ids"][1:], first["labels"][:-1])
    clone = pickle.loads(pickle.dumps(data))
    np.testing.assert_array_equal(clone[3]["labels"], first["labels"])
    permutation = np.random.default_rng(1).permutation(64)
    follows = np.mean(permutation[first["input_ids"]] == first["labels"])
    assert follows > 0.7
    uniform = SyntheticDataset(64, 128, seed=1, kind="uniform", num_samples=4)
    assert np.mean(permutation[uniform[0]["input_ids"]] == uniform[0]["labels"]) < 0.2
    docs = SyntheticDataset(64, 32, seed=1, eos_id=0, doc_length=8, num_samples=4)
    assert np.sum(docs[0]["input_ids"] == 0) >= 3
    held_out = SyntheticDataset(64, 128, seed=1, num_samples=16, offset=16)
    assert not np.array_equal(held_out[3]["input_ids"], first["input_ids"])
    assert np.mean(permutation[held_out[3]["input_ids"]] == held_out[3]["labels"]) > 0.7  # same chain


def test_segment_ids_and_positions_from_eos() -> None:
    tokens = np.array([[5, 0, 7, 8, 0, 9], [1, 2, 3, 4, 5, 6]])
    segments = segment_ids_from_eos(tokens, eos_id=0)
    np.testing.assert_array_equal(segments, [[0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(positions_from_segment_ids(segments), [[0, 1, 0, 1, 2, 0], [0, 1, 2, 3, 4, 5]])
    batch = DocumentFields(0)({"input_ids": tokens, "labels": tokens})
    assert set(batch) == {"input_ids", "labels", "segment_ids", "positions", "loss_mask"}
    assert batch["segment_ids"].dtype == np.int32 and batch["positions"].dtype == np.int32
    # The target after an EOS belongs to the next document and leaves the loss.
    np.testing.assert_array_equal(batch["loss_mask"], tokens != 0)
    existing = DocumentFields(0)({"input_ids": tokens, "labels": tokens, "loss_mask": np.zeros_like(tokens, bool)})
    assert not existing["loss_mask"].any()


def test_tokenize_jsonl_streams_documents_with_eos(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    lines = [{"text": "ab"}, {"text": ""}, {"other": "x"}, "not json", {"text": "abc"}]
    source.write_text("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n")

    def encode(texts: list[str]) -> list[list[int]]:
        return [[ord(ch) for ch in text] for text in texts]

    stats = tokenize_jsonl([source], tmp_path / "out", encode=encode, eos_id=1)
    assert (stats.documents, stats.tokens) == (2, 7)
    assert (stats.skipped_empty, stats.skipped_missing_field, stats.skipped_invalid_json) == (1, 1, 1)
    with MMapIndexedDataset(str(tmp_path / "out")) as corpus:
        assert corpus.get_document(0).tolist() == [97, 98, 1] and corpus.get_document(1).tolist() == [97, 98, 99, 1]
    limited = tokenize_jsonl([source], tmp_path / "one", encode=encode, eos_id=1, max_docs=1)
    assert limited.documents == 1
    with pytest.raises(ValueError, match="EOS"):
        tokenize_jsonl([source], tmp_path / "x", encode=encode, eos_id=None)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("{}\n")
    with pytest.raises(ValueError, match="no documents"):
        tokenize_jsonl([empty], tmp_path / "none", encode=encode, eos_id=1)
    assert not (tmp_path / "none.bin").exists()


def test_vocabulary_check_rejects_ids_beyond_the_model_vocabulary(tmp_path: Path) -> None:
    from jla.data.cidx import check_vocab

    write_cidx(tmp_path / "c", [list(range(100)), [5, 7, 99]])
    check_vocab(str(tmp_path / "c"), 100)  # ids 0..99 fit a vocabulary of 100
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        check_vocab(str(tmp_path / "c"), 99)
