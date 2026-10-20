# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``jla eval``: perplexity of a text or CIDX file, or lm-evaluation-harness tasks.

    jla eval ppl --run-dir runs/gdn --cidx data/val --seq-len 2048 --document-eos 0
    jla eval ppl --run-dir runs/gdn --text wiki.txt --tokenizer gpt2 --seq-len 1024 --stride 512
    jla eval lm-eval --run-dir runs/gdn --tokenizer gpt2 --tasks hellaswag,arc_easy --batch-size 16
    jla eval lm-eval --preset tiny --tokenizer ./tok --tasks my_task --include-path ./tasks --limit 8

Both print one JSON object. ``lm-eval`` needs the optional ``lm-eval`` package.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from jla.inference.loading import add_model_arguments, load_tokenizer, model_from_arguments

__all__ = ["main"]


def _ppl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jla eval ppl", description="Perplexity of a token stream.")
    add_model_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="UTF-8 text file, tokenized as one document (needs --tokenizer)")
    source.add_argument("--cidx", help="CIDX corpus prefix (prefix.idx / prefix.bin)")
    parser.add_argument("--tokenizer", help="Hugging Face tokenizer name or path (for --text)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seq-len", type=int, required=True, help="window length in tokens")
    parser.add_argument("--stride", type=int, help="new tokens scored per window (default seq_len - 1)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, help="evaluate only the first N tokens of the stream")
    parser.add_argument("--prefix-id", type=int, help="token prepended to the stream so the first token is scored")
    parser.add_argument("--document-eos", type=int,
                        help="EOS id: mask attention/state across documents as document-masked training does")
    return parser


def _ppl(argv: Sequence[str]) -> int:
    import numpy as np

    from jla.eval.perplexity import perplexity

    args = _ppl_parser().parse_args(argv)
    extra: dict[str, float | int] = {}
    if args.text:
        if not args.tokenizer:
            raise SystemExit("jla eval ppl: --text needs --tokenizer")
        tokenizer = load_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
        with open(args.text, encoding="utf-8") as handle:
            text = handle.read()
        tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False), np.int64)
        extra["bytes"] = len(text.encode("utf-8"))
    else:
        from jla.data.cidx import MMapIndexedDataset

        with MMapIndexedDataset(args.cidx) as data:
            count = data.num_tokens if args.max_tokens is None else min(args.max_tokens, data.num_tokens)
            tokens = np.array(data.get_chunk(0, count), np.int64)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    model = model_from_arguments(args)
    result = perplexity(model, tokens, seq_len=args.seq_len, stride=args.stride, batch_size=args.batch_size,
                        prefix_id=args.prefix_id, document_eos=args.document_eos)
    report = result.as_dict()
    if args.text and args.prefix_id is not None and args.max_tokens is None:
        # Every token of the text is scored: bits per UTF-8 byte of the whole file.
        report["bits_per_byte"] = result.nll_sum / np.log(2.0) / max(1, int(extra["bytes"]))
    report.update(extra)
    print(json.dumps(report))
    return 0


def _harness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jla eval lm-eval", description="Run lm-evaluation-harness tasks.")
    add_model_arguments(parser)
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tasks", required=True, help="comma-separated task names")
    parser.add_argument("--include-path", help="directory of extra task YAML files")
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float, help="examples per task (or a fraction < 1)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048, help="context length in tokens")
    parser.add_argument("--max-gen-toks", type=int, default=256,
                        help="generation budget of generate_until requests that set none (must be below --max-length)")
    parser.add_argument("--seed", type=int, default=0, help="sampling and few-shot seed")
    parser.add_argument("--output", help="also write the full results JSON here")
    return parser


def _harness(argv: Sequence[str]) -> int:
    args = _harness_parser().parse_args(argv)
    try:
        import lm_eval
        from lm_eval.tasks import TaskManager
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("jla eval lm-eval needs the optional dependency: pip install lm-eval") from exc
    from jla.eval.lm_eval_adapter import JLALM

    tokenizer = load_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
    lm = JLALM(model_from_arguments(args), tokenizer, batch_size=args.batch_size, max_length=args.max_length,
               max_gen_toks=args.max_gen_toks, seed=args.seed)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[t for t in args.tasks.split(",") if t],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        task_manager=TaskManager(include_path=args.include_path) if args.include_path else TaskManager(),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=False,
    )
    summary = {"results": results["results"]}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)
    print(json.dumps(summary, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"ppl": _ppl, "lm-eval": _harness}
    if args and args[0] in ("-h", "--help"):
        print("usage: jla eval {ppl,lm-eval} ...\n\n"
              "  ppl      Perplexity of a token stream.\n"
              "  lm-eval  Run lm-evaluation-harness tasks.\n\n"
              "Run jla eval <command> --help for its options.")
        return 0
    if not args or args[0] not in commands:
        print("usage: jla eval {ppl,lm-eval} ...", file=sys.stderr)
        return 2
    return commands[args[0]](args[1:])
