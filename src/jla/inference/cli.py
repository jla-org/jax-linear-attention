# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``jla generate``: sample completions from a checkpoint (or a randomly initialized preset).

    jla generate --run-dir runs/gdn --tokenizer EleutherAI/gpt-neox-20b \\
        --prompt "The capital of France is" --max-new-tokens 32
    jla generate --run-dir runs/gdn --tokenizer gpt2 --prompts-file prompts.txt \\
        --temperature 0.8 --top-p 0.95 --seed 1 --batch-size 16
    jla generate --preset tiny --prompt-ids "1 2 3" --prompt-ids "4 5" --max-new-tokens 8

Prompts are batched in order of length (so rows of one batch share a long prefill and
few shapes compile); the output is one JSON object per prompt in input order:
``index``, ``tokens`` (the new token ids), ``stopped`` (ended on EOS) and, with a
tokenizer, ``prompt`` and ``completion`` (decoded without the EOS token).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from jla.inference.loading import add_model_arguments, load_tokenizer, model_from_arguments

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jla generate", description=__doc__.split("\n\n")[0])
    add_model_arguments(parser)
    parser.add_argument("--tokenizer", help="Hugging Face tokenizer name or path (for text prompts)")
    parser.add_argument("--local-files-only", action="store_true", help="never download the tokenizer")
    parser.add_argument("--add-bos", action="store_true", help="prepend the tokenizer's BOS token to text prompts")
    parser.add_argument("--prompt", action="append", default=[], help="a text prompt (repeatable)")
    parser.add_argument("--prompts-file", help="text prompts, one per line (.jsonl: objects with a 'prompt' field)")
    parser.add_argument("--prompt-ids", action="append", default=[],
                        help="a prompt as space-separated token ids (repeatable; no tokenizer needed)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--top-k", type=int, default=0, help="0 = off")
    parser.add_argument("--top-p", type=float, default=1.0, help="1 = off")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--eos-id", type=int, action="append", default=None,
                        help="stop token id (repeatable; default: the tokenizer's EOS)")
    parser.add_argument("--no-eos", action="store_true", help="always generate --max-new-tokens tokens")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute the full forward pass every step instead of using decode caches")
    parser.add_argument("--prefill-chunk", type=int, default=None,
                        help="most prompt tokens per prefill step (default: sized to bound attention memory)")
    return parser


def _read_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    if path.endswith(".jsonl"):
        return [json.loads(line)["prompt"] for line in lines if line.strip()]
    return [line for line in lines if line]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from jla.inference.generate import generate
    from jla.inference.sampling import SamplingConfig

    texts = list(args.prompt) + (_read_prompts(args.prompts_file) if args.prompts_file else [])
    if texts and args.prompt_ids:
        raise SystemExit("jla generate: use text prompts or --prompt-ids, not both")
    if not texts and not args.prompt_ids:
        raise SystemExit("jla generate: no prompts (--prompt, --prompts-file or --prompt-ids)")
    tokenizer = None
    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
    elif texts:
        raise SystemExit("jla generate: text prompts need --tokenizer")
    if texts:
        prefix = [tokenizer.bos_token_id] if args.add_bos and tokenizer.bos_token_id is not None else []
        prompts = [prefix + list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    else:
        prompts = [[int(t) for t in p.split()] for p in args.prompt_ids]
    if args.no_eos:
        eos_ids: list[int] = []
    elif args.eos_id is not None:
        eos_ids = list(args.eos_id)
    else:
        eos_ids = [] if tokenizer is None or tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]
    sampling = SamplingConfig(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    model = model_from_arguments(args)
    if args.batch_size < 1:
        raise SystemExit("jla generate: --batch-size must be >= 1")
    pad_id = eos_ids[0] if eos_ids else 0
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))  # row ids keep the samples
    records: dict[int, dict] = {}
    for begin in range(0, len(order), args.batch_size):
        batch = order[begin : begin + args.batch_size]
        result = generate(model, [prompts[i] for i in batch], max_new_tokens=args.max_new_tokens, sampling=sampling,
                          seed=args.seed, eos_ids=eos_ids, pad_id=pad_id, row_ids=batch,
                          use_cache=False if args.no_cache else None, prefill_chunk=args.prefill_chunk)
        for row, i in enumerate(batch):
            tokens = result.row(row)
            record = {"index": i, "tokens": tokens, "stopped": bool(result.stopped[row])}
            if tokenizer is not None:
                record["prompt"] = texts[i] if texts else tokenizer.decode(prompts[i])
                record["completion"] = tokenizer.decode([t for t in tokens if t not in eos_ids])
            records[i] = record
    for i in range(len(prompts)):
        sys.stdout.write(json.dumps(records[i]) + "\n")
    sys.stdout.flush()
    return 0
