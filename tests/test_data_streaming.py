# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming text data: equality with the pre-tokenized CIDX path, packing, sharding and resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jla.config import DataConfig
from jla.data.datasets import TokenWindowDataset
from jla.data.loader import collate
from jla.data.make_data import tokenize_jsonl
from jla.data.packing import DocumentFields
from jla.data.pipeline import DataState
from jla.data.streaming import StreamLoader, StreamSpec, read_documents, shard_geometry

EOS = 256
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]


def _documents(count: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(count):
        words = rng.choice(WORDS, size=int(rng.integers(1, 12)))
        text = " ".join(words)
        if rng.random() < 0.2:
            text += " été 中文"  # multi-byte UTF-8
        docs.append(text)
    return docs


def _write_jsonl(path: Path, docs: list[str], *, noise: bool = False) -> Path:
    lines = []
    for index, doc in enumerate(docs):
        lines.append(json.dumps({"id": index, "text": doc}))
        if noise and index % 7 == 3:  # records every reader skips
            lines += ["", "{not json", json.dumps({"other": "x"}), json.dumps({"text": ""}), "   "]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _byte_encode(texts: list[str]) -> list[list[int]]:  # independent of jla.data.streaming.encode_bytes
    return [list(text.encode("utf-8")) for text in texts]


def _spec(files, **overrides) -> StreamSpec:
    values = {"seq_len": 16, "vocab_size": 512, "tokenizer": "byte", "eos_id": EOS, "files": tuple(map(str, files))}
    return StreamSpec(**{**values, **overrides})


def _take(loader, count: int) -> list[dict[str, np.ndarray]]:
    return [next(loader) for _ in range(count)]


def _assert_batches_equal(actual: dict, expected: dict) -> None:
    assert sorted(actual) == sorted(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _pretokenized_windows(tmp_path: Path, jsonl: Path, seq_len: int, encode=_byte_encode, eos_id: int = EOS):
    """The CIDX path: `jla tokenize` + TokenWindowDataset, as {window index: sample}."""
    prefix = tmp_path / "pretok"
    tokenize_jsonl([jsonl], prefix, encode=encode, eos_id=eos_id)
    dataset = TokenWindowDataset(str(prefix), seq_len)
    windows = {}
    for index in range(len(dataset)):
        window = (dataset._factor * (index + 1) ** 3) % dataset._prime  # the dataset's window order
        windows[window] = dataset[index]
    assert sorted(windows) == list(range(len(dataset)))
    return windows


@pytest.mark.parametrize("document_masking", [False, True])
def test_concat_split_stream_equals_the_pretokenized_path(tmp_path: Path, document_masking: bool) -> None:
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", _documents(120), noise=True)
    windows = _pretokenized_windows(tmp_path, jsonl, 16)
    rows = 4
    loader = StreamLoader(_spec([jsonl], document_masking=document_masking), rows=rows)
    transform = DocumentFields(EOS) if document_masking else None
    for b in range(len(windows) // rows):
        expected = collate({w: windows[w] for w in range(len(windows))}, np.arange(b * rows, (b + 1) * rows), transform)
        _assert_batches_equal(next(loader), expected)


def test_hf_tokenizer_stream_equals_the_pretokenized_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))  # the libraries' caches, read at their import
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from jla.data.make_data import load_hf_tokenizer

    vocab = {"[UNK]": 0, "</s>": 1, **{word: i + 2 for i, word in enumerate(WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>")
    fast.save_pretrained(tmp_path / "tokenizer")
    name = str(tmp_path / "tokenizer")

    jsonl = _write_jsonl(tmp_path / "docs.jsonl", _documents(200))
    encode, eos_id = load_hf_tokenizer(name, local_files_only=True)
    assert eos_id == 1
    windows = _pretokenized_windows(tmp_path, jsonl, 8, encode=encode, eos_id=eos_id)
    spec = StreamSpec(seq_len=8, vocab_size=64, tokenizer=name, eos_id=None, files=(str(jsonl),), tokenizer_local_files_only=True)
    loader = StreamLoader(spec, rows=2)
    assert loader.eos_id == 1
    for b in range(len(windows) // 2):
        _assert_batches_equal(next(loader), collate(windows, np.arange(2 * b, 2 * b + 2)))


def test_parquet_and_hf_datasets_sources_equal_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    docs = _documents(90, seed=3)
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", docs)
    parquet = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"text": docs, "id": list(range(len(docs)))}), parquet, row_group_size=7)
    expected = _take(StreamLoader(_spec([jsonl], packing="first_fit"), rows=3), 25)
    for batch, reference in zip(_take(StreamLoader(_spec([parquet], packing="first_fit"), rows=3), 25), expected):
        _assert_batches_equal(batch, reference)

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))  # the library's caches, read at its import
    pytest.importorskip("datasets")
    hf = StreamSpec(seq_len=16, vocab_size=512, tokenizer="byte", eos_id=EOS, hf_dataset="json", hf_data_files=(str(jsonl),), packing="first_fit")
    loader = StreamLoader(hf, rows=3)
    for batch, reference in zip(_take(loader, 25), expected):
        _assert_batches_equal(batch, reference)
    resumed = StreamLoader(hf, rows=3, state=json.loads(json.dumps(loader.state_at(11))), skip_batches=11)
    for batch, reference in zip(_take(resumed, 14), expected[11:], strict=True):
        _assert_batches_equal(batch, reference)


def _reference_first_fit(documents: list[list[int]], seq_len: int, bins: int) -> list[dict[str, np.ndarray]]:
    """Plain first fit: split documents into seq_len chunks, place each in the first bin with room."""
    pieces = []
    for doc in documents:
        for start in range(0, len(doc), seq_len):
            stop = min(start + seq_len, len(doc))
            pieces.append((doc[start:stop], doc[start + 1 : stop + 1]))
    rows, open_bins = [], [[] for _ in range(bins)]
    for piece in pieces:
        for target in open_bins:
            if sum(len(p[0]) for p in target) + len(piece[0]) <= seq_len:
                target.append(piece)
                break
        else:
            rows += open_bins
            open_bins = [[piece]] + [[] for _ in range(bins - 1)]
    out = []
    for row in rows:
        ids, labels, mask, segments, positions = [], [], [], [], []
        for segment, (inputs, targets) in enumerate(row):
            ids += inputs
            labels += targets + [EOS] * (len(inputs) - len(targets))
            mask += [True] * len(targets) + [False] * (len(inputs) - len(targets))
            segments += [segment] * len(inputs)
            positions += list(range(len(inputs)))
        pad = seq_len - len(ids)
        ids += [EOS] * pad
        labels += [EOS] * pad
        mask += [False] * pad
        segments += [len(row)] * pad
        positions += list(range(pad))
        mask = [m and token != EOS for m, token in zip(mask, ids)]
        out.append({"input_ids": ids, "labels": labels, "loss_mask": mask, "segment_ids": segments, "positions": positions})
    return out


@pytest.mark.parametrize("pack_bins", [0, 6])
def test_first_fit_matches_a_reference_packer(tmp_path: Path, pack_bins: int) -> None:
    docs = _documents(150, seed=1) + ["x" * 40, "y" * 16, "z" * 15]  # documents longer than seq_len split
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", docs)
    tokens = [list(text.encode("utf-8")) + [EOS] for text in read_documents(jsonl)]
    reference = _reference_first_fit(tokens, 16, pack_bins or 4)
    loader = StreamLoader(_spec([jsonl], packing="first_fit", pack_bins=pack_bins, document_masking=True), rows=4)
    count = len(reference) // 4 - 1  # the last bins wait for documents of the next epoch
    produced = [row for batch in _take(loader, count) for row in zip(*(batch[k] for k in sorted(batch)))]
    keys = sorted(reference[0])
    for actual, expected in zip(produced, reference[: 4 * count], strict=True):
        for key, value in zip(keys, actual):
            np.testing.assert_array_equal(value, np.asarray(expected[key]), err_msg=key)
    without_masking = _take(StreamLoader(_spec([jsonl], packing="first_fit", pack_bins=pack_bins), rows=4), count)
    assert sorted(without_masking[0]) == ["input_ids", "labels", "loss_mask"]


@pytest.mark.parametrize("pack_bins", [0, 6])
def test_first_fit_matches_grain_first_fit_pack(tmp_path: Path, pack_bins: int) -> None:
    """The packing decisions equal Grain's FirstFitPackIterDataset (MaxText's first_fit), bins unshuffled."""
    grain = pytest.importorskip("grain")
    seq_len, rows = 16, 4
    bins = pack_bins or rows
    docs = _documents(150, seed=8) + ["x" * 40, "y" * 16, "z" * 15]  # documents longer than seq_len split
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", docs)
    chunks = []  # what Grain packs: each tokenized document cut into seq_len pieces (Grain itself only truncates)
    for text in read_documents(jsonl):
        tokens = list(text.encode("utf-8")) + [EOS]
        chunks += [{"inputs": np.asarray(tokens[i : i + seq_len], np.int32)} for i in range(0, len(tokens), seq_len)]
    packed = grain.experimental.FirstFitPackIterDataset(
        grain.MapDataset.source(chunks).to_iter_dataset(),
        length_struct={"inputs": seq_len},
        num_packing_bins=bins,
        shuffle_bins=False,
    )
    expected = list(packed)
    complete = len(expected) - (len(expected) % bins or bins)  # the last flush waits for the next epoch here
    assert complete >= 3 * bins
    loader = StreamLoader(_spec([jsonl], packing="first_fit", pack_bins=pack_bins, document_masking=True), rows=rows)
    batches = _take(loader, -(-complete // rows))
    ours = {key: np.concatenate([b[key] for b in batches])[:complete] for key in ("input_ids", "segment_ids", "positions")}
    for row, reference in enumerate(expected[:complete]):
        real = reference["inputs_segment_ids"] > 0  # Grain pads with segment 0; ours gives padding its own segment
        assert 0 < real.sum() and real[: real.sum()].all()  # tokens first, then padding
        np.testing.assert_array_equal(ours["input_ids"][row][real], reference["inputs"][real], err_msg=f"row {row}")
        np.testing.assert_array_equal(ours["segment_ids"][row][real] + 1, reference["inputs_segment_ids"][real], err_msg=f"row {row}")
        np.testing.assert_array_equal(ours["positions"][row][real], reference["inputs_positions"][real], err_msg=f"row {row}")
        np.testing.assert_array_equal(ours["input_ids"][row][~real], EOS, err_msg=f"row {row}")
        assert np.all(ours["segment_ids"][row][~real] == ours["segment_ids"][row][real].max() + 1)


def test_shards_partition_the_records_deterministically(tmp_path: Path) -> None:
    docs = _documents(99, seed=2)
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", docs)
    shards, rows, seq_len = 3, 2, 16
    for shard in range(shards):
        stream = np.concatenate([np.array(list(d.encode("utf-8")) + [EOS]) for d in docs[shard::shards]])
        loader = StreamLoader(_spec([jsonl]), rows=rows, shard_index=shard, shard_count=shards)
        twin = StreamLoader(_spec([jsonl]), rows=rows, shard_index=shard, shard_count=shards)
        for b in range((stream.size - 1) // seq_len // rows):
            batch = next(loader)
            _assert_batches_equal(next(twin), batch)
            for r in range(rows):
                w = b * rows + r
                np.testing.assert_array_equal(batch["input_ids"][r], stream[w * seq_len : (w + 1) * seq_len])
                np.testing.assert_array_equal(batch["labels"][r], stream[w * seq_len + 1 : (w + 1) * seq_len + 1])
    assert shard_geometry((4, 8), 8) == (1, 2)
    assert shard_geometry((0, 8), 8) == (0, 1)
    with pytest.raises(ValueError, match="equal blocks"):
        shard_geometry((2, 5), 8)


@pytest.mark.parametrize("packing", ["concat_split", "first_fit"])
@pytest.mark.parametrize("shard_count", [1, 2])
def test_resume_mid_stream_reproduces_the_next_batches(tmp_path: Path, packing: str, shard_count: int) -> None:
    files = [
        _write_jsonl(tmp_path / "a.jsonl", _documents(23, seed=4), noise=True),
        _write_jsonl(tmp_path / "b.jsonl", _documents(17, seed=5)),
    ]
    pq = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        files.append(tmp_path / "c.parquet")
        pq.write_table(pa.table({"text": _documents(19, seed=6)}), files[-1], row_group_size=5)
    except ImportError:
        pass
    spec = _spec(files, packing=packing, pack_bins=5 if packing == "first_fit" else 0, document_masking=True, shuffle=True, read_ahead=3)
    total = 60  # several epochs, with a new file order every epoch
    for shard in range(shard_count):
        full = StreamLoader(spec, rows=3, shard_index=shard, shard_count=shard_count, history=total + 1)
        batches = _take(full, total)
        assert full.state_at(total)["cursor"]["epoch"] >= 1
        for k in (0, 1, 7, 33, 59):
            state = json.loads(json.dumps(full.state_at(k)))  # through the JSON sidecar
            resumed = StreamLoader(spec, rows=3, shard_index=shard, shard_count=shard_count, state=state, skip_batches=k)
            for expected in batches[k:]:
                _assert_batches_equal(next(resumed), expected)
            assert resumed.state_at(total) == full.state_at(total)
    with pytest.raises(ValueError, match="cannot resume the stream"):
        StreamLoader(spec, rows=4, state=json.loads(json.dumps(full.state_at(7))), skip_batches=7, shard_index=0, shard_count=shard_count)
    with pytest.raises(ValueError, match="needs its saved position"):
        StreamLoader(spec, rows=3, skip_batches=7)
    with pytest.raises(RuntimeError, match="not in the loader history"):
        StreamLoader(spec, rows=3, history=4).state_at(3)


@pytest.mark.parametrize("packing", ["concat_split", "first_fit"])
def test_read_ahead_does_not_change_the_stream(tmp_path: Path, packing: str) -> None:
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", _documents(60, seed=9), noise=True)
    expected = _take(StreamLoader(_spec([jsonl], packing=packing, document_masking=True, read_ahead=1), rows=3), 30)
    for read_ahead, chars in ((7, 8 << 20), (1000, 8 << 20), (1000, 50)):
        spec = _spec([jsonl], packing=packing, document_masking=True, read_ahead=read_ahead, read_ahead_chars=chars)
        loader = StreamLoader(spec, rows=3)
        for batch, reference in zip(_take(loader, 30), expected, strict=True):
            _assert_batches_equal(batch, reference)


def test_byte_stream_rejects_bad_ids_and_configs(tmp_path: Path) -> None:
    jsonl = _write_jsonl(tmp_path / "docs.jsonl", _documents(10))
    with pytest.raises(ValueError, match="eos id 256 is outside the vocabulary"):
        StreamLoader(_spec([jsonl], vocab_size=200), rows=1)
    too_large = (lambda texts: [[600] for _ in texts], EOS)
    with pytest.raises(ValueError, match="token id 600 is outside the vocabulary"):
        next(StreamLoader(_spec([jsonl]), rows=1, encoder=too_large))
    with pytest.raises(ValueError, match="eos_id >= 256"):
        StreamLoader(_spec([jsonl], eos_id=10), rows=1)
    with pytest.raises(ValueError, match="exactly one of data.stream_files"):
        DataConfig(source="stream", tokenizer="byte", eos_id=EOS)
    with pytest.raises(ValueError, match="needs data.tokenizer"):
        DataConfig(source="stream", stream_files=(str(jsonl),))
    with pytest.raises(ValueError, match="only used with data.source = 'stream'"):
        DataConfig(stream_files=(str(jsonl),))
    with pytest.raises(ValueError, match="set data.shuffle = false"):
        DataConfig(source="stream", hf_dataset="json", tokenizer="byte", eos_id=EOS)
    with pytest.raises(ValueError, match=".jsonl or .parquet"):
        DataConfig(source="stream", stream_files=("docs.txt",), tokenizer="byte", eos_id=EOS)


def test_train_loader_reads_and_resumes_the_shard_of_its_rows(tmp_path: Path) -> None:
    """Two hosts holding rows [0, 4) and [4, 8) of a global batch of 8: each resumes its own shard's position."""
    from _toy_model import experiment
    from jla.data.pipeline import train_dataset, train_loader

    jsonl = _write_jsonl(tmp_path / "docs.jsonl", _documents(80, seed=7))
    data = {"source": "stream", "stream_files": (str(jsonl),), "tokenizer": "byte", "eos_id": EOS, "packing": "first_fit"}
    config = experiment(str(tmp_path / "run"), train={"seq_len": 16, "final_eval": False}, data=data)
    spec = train_dataset(config)
    hosts = [train_loader(config, spec, global_batch_size=8, rows=rows, skip_batches=0) for rows in ((0, 4), (4, 8))]
    assert [(h.shard_index, h.shard_count, h.rows) for h in hosts] == [(0, 2, 4), (1, 2, 4)]
    batches = [_take(h, 9) for h in hosts]
    saved = json.loads(json.dumps([h.state_at(5) for h in hosts]))  # what the checkpoint sidecar holds
    for index, rows in enumerate(((0, 4), (4, 8))):
        resumed = train_loader(config, spec, global_batch_size=8, rows=rows, skip_batches=5, stream_state=saved)
        for expected in batches[index][5:]:
            _assert_batches_equal(next(resumed), expected)
    with pytest.raises(ValueError, match="cannot resume the stream"):
        train_loader(config, spec, global_batch_size=8, rows=(0, 8), skip_batches=5, stream_state=saved)


def test_gather_shard_states_orders_and_checks_every_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three processes, two of them reading shard 1 (model parallel across hosts), through a simulated all-gather."""
    import jax
    from jax.experimental import multihost_utils

    from jla.data.streaming import gather_shard_states

    def simulate(states: list[dict]) -> None:
        payloads = [np.frombuffer(json.dumps(s, sort_keys=True).encode("utf-8"), np.uint8) for s in states]

        def allgather(value):  # what every process receives: the stacked inputs of all processes
            value = np.asarray(value)
            if value.dtype == np.int32:
                return np.stack([np.array([p.size], np.int32) for p in payloads])
            out = np.zeros((len(payloads), value.size), np.uint8)
            for i, p in enumerate(payloads):
                out[i, : p.size] = p
            return out

        monkeypatch.setattr(jax, "process_count", lambda: len(states))
        monkeypatch.setattr(multihost_utils, "process_allgather", allgather)

    one = {"shard_index": 1, "shard_count": 2, "packer": "b" * 300}
    zero = {"shard_index": 0, "shard_count": 2, "packer": "é" * 3}
    simulate([one, zero, dict(one)])
    assert gather_shard_states(one) == [zero, one]
    simulate([one, zero, {**one, "packer": "c"}])
    with pytest.raises(RuntimeError, match="disagree"):
        gather_shard_states(one)
    simulate([one, dict(one)])
    with pytest.raises(RuntimeError, match="expected 0..1"):
        gather_shard_states(one)


def test_data_state_keeps_old_sidecars_readable() -> None:
    plain = DataState(step=3, batches_consumed=6, tokens_seen=96, global_batch_size=2, seq_len=8, grad_accum=2, fingerprint="f")
    assert "stream" not in plain.to_dict()
    assert DataState.from_dict(plain.to_dict()) == plain
    streamed = DataState(**{**plain.to_dict(), "stream": [{"shard_index": 0}]})
    assert DataState.from_dict(json.loads(json.dumps(streamed.to_dict()))) == streamed
    with pytest.raises(ValueError, match="do not match"):
        DataState.from_dict({**plain.to_dict(), "extra": 1})
