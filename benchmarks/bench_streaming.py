# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side throughput of the streaming data path (``data.source = "stream"``).

    python benchmarks/bench_streaming.py --workdir /path/to/scratch [--seconds 4] [--out results.json]

Builds a synthetic text corpus (JSONL and parquet) and a byte-level BPE
tokenizer trained on it (``tokenizers``), then measures, on one thread of
the training process, batches of ``rows x seq_len`` tokens per second:

* ``cidx``: the pre-tokenized path (``jla tokenize`` once, then
  ``TokenWindowDataset`` + ``BatchLoader`` with document masking), the upper bound;
* ``stream``: :class:`jla.data.streaming.StreamLoader` for each source,
  tokenizer and packing;
* ``grain``: MaxText's Grain recipe, records parsed and tokenized one at a
  time, then Grain's ``ConcatThenSplitIterDataset`` / ``FirstFitPackIterDataset``
  and ``batch`` (the strongest external baseline);
* ``hf-map-group``: the common Hugging Face recipe, ``datasets`` streaming
  ``.map(tokenizer, batched=True)`` and concatenate-then-split windows
  (``run_clm``'s ``group_texts``), as an external baseline.

``targets_per_s`` counts trained targets (``loss_mask``: no EOS-position or
padding targets), comparable across packings. Needs the optional
``tokenizers``, ``transformers``, ``datasets``, ``pyarrow`` and ``grain`` packages.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from jla.data.datasets import TokenWindowDataset
from jla.data.loader import BatchLoader
from jla.data.make_data import load_hf_tokenizer, tokenize_jsonl
from jla.data.packing import DocumentFields
from jla.data.streaming import StreamLoader, StreamSpec, read_documents

WORDS = 4000


def build_corpus(workdir: Path, documents: int, seed: int = 0) -> tuple[Path, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    vocabulary = np.asarray(["".join(rng.choice(letters, size=int(n))) for n in rng.integers(2, 10, size=WORDS)])
    zipf = 1.0 / np.arange(1, WORDS + 1)
    zipf /= zipf.sum()
    lengths = np.clip(rng.lognormal(mean=6.0, sigma=1.0, size=documents).astype(int), 5, 20000)  # words per document
    texts = [" ".join(vocabulary[rng.choice(WORDS, size=int(n), p=zipf)]) for n in lengths]
    jsonl = workdir / "corpus.jsonl"
    jsonl.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    parquet = workdir / "corpus.parquet"
    pq.write_table(pa.table({"text": texts}), parquet, row_group_size=1024)
    return jsonl, parquet


def build_bpe(workdir: Path, jsonl: Path, vocab_size: int) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], show_progress=False)
    backend.train_from_iterator(read_documents(jsonl), trainer)
    path = workdir / "bpe"
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>").save_pretrained(path)
    return str(path)


def timed(batches: Iterator[dict], seconds: float, eos_id: int) -> dict[str, float]:
    next(batches)  # warm-up: opens files, fills the first packer state
    count = tokens = targets = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        batch = next(batches)
        count += 1
        tokens += batch["input_ids"].size
        mask = batch["loss_mask"] if "loss_mask" in batch else batch["input_ids"] != eos_id
        targets += int(mask.sum())
    elapsed = time.perf_counter() - start
    return {"batches_per_s": count / elapsed, "tokens_per_s": tokens / elapsed, "targets_per_s": targets / elapsed}


def hf_map_group(jsonl: Path, tokenizer: str, eos_id: int, seq_len: int, rows: int) -> Iterator[dict]:
    """``datasets`` streaming + batched tokenizer map + concatenate-then-split windows of seq_len + 1."""
    import datasets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    stream = datasets.load_dataset("json", data_files=str(jsonl), split="train", streaming=True)
    stream = stream.map(
        lambda b: {"ids": tok(b["text"], add_special_tokens=False)["input_ids"]}, batched=True, remove_columns=["text"]
    )
    buffer: list[int] = []
    span = seq_len * rows + 1
    while True:
        for example in stream:
            buffer.extend(example["ids"])
            buffer.append(eos_id)
            while len(buffer) >= span:
                block = np.asarray(buffer[:span], dtype=np.int32)
                del buffer[: span - 1]
                windows = np.stack([block[r * seq_len : (r + 1) * seq_len + 1] for r in range(rows)])
                yield {"input_ids": windows[:, :-1], "labels": windows[:, 1:]}


def grain_packed(jsonl: Path, encode: Callable, eos_id: int, seq_len: int, rows: int, packing: str) -> Iterator[dict]:
    """MaxText's Grain recipe: parse and tokenize each record, pack (Grain), batch; endless.

    The JSONL lines are held in memory (file reads are not timed, unlike the
    stream). first_fit trims each document to seq_len (Grain packs only trimmed
    elements, as MaxText's TokenizeAndTrim does); concat_split keeps every token.
    """
    import grain

    lines = jsonl.read_text(encoding="utf-8").splitlines()

    def tokenize(line: str) -> dict[str, np.ndarray]:
        ids = np.asarray(list(encode([json.loads(line)["text"]])[0]) + [eos_id], np.int32)
        return {"inputs": ids[:seq_len] if packing == "first_fit" else ids}

    parent = grain.MapDataset.source(lines).repeat().map(tokenize).to_iter_dataset()
    if packing == "first_fit":
        packed = grain.experimental.FirstFitPackIterDataset(
            parent, length_struct={"inputs": seq_len}, num_packing_bins=rows, shuffle_bins=False
        )
    else:
        packed = grain.experimental.ConcatThenSplitIterDataset(parent, length_struct={"inputs": seq_len})
    for batch in packed.batch(rows, drop_remainder=True):
        inputs = batch["inputs"]
        yield {"input_ids": inputs, "loss_mask": (batch["inputs_segment_ids"] > 0) & (inputs != eos_id)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Host-side throughput of the streaming data path.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--documents", type=int, default=6000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--bpe-vocab", type=int, default=8192)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--read-ahead", default="64,1024", help="comma-separated documents per tokenizer call")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    jsonl, parquet = build_corpus(workdir, args.documents)
    bpe = build_bpe(workdir, jsonl, args.bpe_vocab)
    bpe_encode, bpe_eos = load_hf_tokenizer(bpe, local_files_only=True)

    def byte_encode(texts: list[str]) -> list[list[int]]:
        return [list(text.encode("utf-8")) for text in texts]

    tokenizers: dict[str, tuple[str, int, int, Callable]] = {
        "byte": ("byte", 256, 512, byte_encode),
        "bpe": (bpe, bpe_eos, args.bpe_vocab, bpe_encode),
    }
    read_aheads = [int(value) for value in args.read_ahead.split(",")]
    results: dict[str, dict[str, float]] = {}
    for name, (tokenizer, eos_id, vocab_size, encode) in tokenizers.items():
        prefix = workdir / f"cidx-{name}"
        started = time.perf_counter()
        stats = tokenize_jsonl([jsonl], prefix, encode=encode, eos_id=eos_id)
        offline = stats.tokens / (time.perf_counter() - started)
        loader = BatchLoader(TokenWindowDataset(str(prefix), args.seq_len), args.rows, transform=DocumentFields(eos_id))
        results[f"cidx/{name}"] = {**timed(loader, args.seconds, eos_id), "offline_tokenize_tokens_per_s": offline}
        loader.close()
        for source, path in (("jsonl", jsonl), ("parquet", parquet)):
            for packing in ("concat_split", "first_fit"):
                for read_ahead in read_aheads:
                    spec = StreamSpec(
                        seq_len=args.seq_len,
                        vocab_size=vocab_size,
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        files=(str(path),),
                        packing=packing,
                        document_masking=True,
                        read_ahead=read_ahead,
                        tokenizer_local_files_only=True,
                    )
                    stream = StreamLoader(spec, rows=args.rows)
                    results[f"stream/{name}/{source}/{packing}/read_ahead={read_ahead}"] = timed(stream, args.seconds, eos_id)
                    stream.close()
        for packing in ("concat_split", "first_fit"):
            grain_batches = grain_packed(jsonl, encode, eos_id, args.seq_len, args.rows, packing)
            results[f"grain/{name}/jsonl/{packing}"] = timed(grain_batches, args.seconds, eos_id)
    results["hf-map-group/bpe"] = timed(hf_map_group(jsonl, bpe, bpe_eos, args.seq_len, args.rows), args.seconds, bpe_eos)
    report = {
        "seq_len": args.seq_len,
        "rows": args.rows,
        "documents": args.documents,
        "read_ahead": read_aheads,
        "corpus_mib": jsonl.stat().st_size / 2**20,
        "results": results,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
