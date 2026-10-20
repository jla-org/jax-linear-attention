# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tokenize JSONL text into a CIDX corpus.

    python -m jla.cli tokenize --input train.jsonl --output data/train \\
        --tokenizer EleutherAI/gpt-neox-20b

Each JSONL line holds an object with a text field. Texts are encoded in
batches with a HuggingFace fast tokenizer (``transformers`` is an optional
dependency) without special tokens, the tokenizer's EOS id is appended to
every document, and tokens stream to disk; the ``.bin``/``.idx`` pair is
fsynced and renamed into place only when the whole input succeeded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jla.data.cidx import CidxWriter

__all__ = ["TokenizeStats", "load_hf_tokenizer", "main", "tokenize_jsonl"]

logger = logging.getLogger(__name__)

Encoder = Callable[[list[str]], list[list[int]]]

_BATCH_DOCS = 4096
_BATCH_CHARS = 16 * 1024 * 1024


@dataclass
class TokenizeStats:
    documents: int = 0
    tokens: int = 0
    eos_id: int | None = None
    skipped_invalid_json: int = 0
    skipped_missing_field: int = 0
    skipped_empty: int = 0
    seconds: float = 0.0


def load_hf_tokenizer(name: str, *, local_files_only: bool = False) -> tuple[Encoder, int | None]:
    """Fast HuggingFace tokenizer as ``(encode_batch, eos_id)``."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("tokenization needs the optional 'transformers' package") from exc
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)
    if not getattr(tokenizer, "is_fast", False):
        logger.warning("%s loaded a slow tokenizer; tokenization will be much slower", name)

    def encode(texts: list[str]) -> list[list[int]]:
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        ).input_ids

    return encode, tokenizer.eos_token_id


def tokenize_jsonl(
    input_paths: Sequence[str | Path],
    output_prefix: str | Path,
    *,
    encode: Encoder,
    eos_id: int | None,
    text_field: str = "text",
    dtype: str = "int32",
    append_eos: bool = True,
    max_docs: int | None = None,
    max_tokens: int | None = None,
) -> TokenizeStats:
    """Encode every document of ``input_paths`` into ``output_prefix.{bin,idx}``."""
    if append_eos and eos_id is None:
        raise ValueError("append_eos needs a tokenizer with an EOS token (or pass --no-append-eos)")
    stats = TokenizeStats(eos_id=eos_id if append_eos else None)
    started = time.perf_counter()
    writer = CidxWriter(output_prefix, dtype=dtype)

    def done() -> bool:
        return (max_docs is not None and stats.documents >= max_docs) or (
            max_tokens is not None and stats.tokens >= max_tokens
        )

    def flush(texts: list[str]) -> None:
        for ids in encode(texts):
            if done():
                return
            if not ids:
                stats.skipped_empty += 1
                continue
            document = list(ids) + [eos_id] if append_eos else ids
            writer.add(document)
            stats.documents += 1
            stats.tokens += len(document)

    try:
        batch: list[str] = []
        chars = 0
        for path in input_paths:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if done():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats.skipped_invalid_json += 1
                        continue
                    text = record.get(text_field) if isinstance(record, dict) else None
                    if not isinstance(text, str):
                        stats.skipped_missing_field += 1
                        continue
                    if not text:
                        stats.skipped_empty += 1
                        continue
                    batch.append(text)
                    chars += len(text)
                    if len(batch) >= _BATCH_DOCS or chars >= _BATCH_CHARS:
                        flush(batch)
                        batch, chars = [], 0
        if batch and not done():
            flush(batch)
        if stats.documents == 0:
            raise ValueError(f"no documents with a non-empty {text_field!r} field in {list(map(str, input_paths))}")
        writer.commit()
    except BaseException:
        writer.abort()
        raise
    stats.seconds = round(time.perf_counter() - started, 3)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jla tokenize", description="Tokenize JSONL into a CIDX corpus.")
    parser.add_argument("--input", required=True, nargs="+", help="JSONL file(s)")
    parser.add_argument("--output", required=True, help="output prefix (writes PREFIX.bin and PREFIX.idx)")
    parser.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer name or local path")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dtype", default="int32", choices=["uint16", "int32"])
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--format", default="text", choices=["text", "chat", "instruction"],
                        help="text: plain documents; chat/instruction: SFT corpora with loss masks (jla.data.sft_tokenize)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--format", default="text")
    if probe.parse_known_args(argv)[0].format in ("chat", "instruction"):
        from jla.data.sft_tokenize import main as sft_main

        return sft_main(argv)
    args = build_parser().parse_args(argv)
    encode, eos_id = load_hf_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
    stats = tokenize_jsonl(
        args.input,
        args.output,
        encode=encode,
        eos_id=eos_id,
        text_field=args.text_field,
        dtype=args.dtype,
        append_eos=args.append_eos,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
    )
    print(json.dumps({"output": args.output, **asdict(stats)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
