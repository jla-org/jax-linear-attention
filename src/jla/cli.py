# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Command line interface (the ``jla`` command; ``python -m jla.cli`` is the same).

    jla train --config configs/tiny-synthetic.toml --set optim.lr=3e-4
    jla show-config --config configs/transformer-125m.toml --set train.seq_len=4096
    jla tokenize --input train.jsonl --output data/train --tokenizer EleutherAI/gpt-neox-20b
    jla generate --run-dir runs/gdn --tokenizer gpt2 --prompt "The capital of France is" --max-new-tokens 32
    jla eval ppl --run-dir runs/gdn --cidx data/val --seq-len 2048 --document-eos 0
    jla convert import --hf path/to/Qwen3-0.6B --output ckpt/qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from jla.config import load_config

__all__ = ["main"]


def _config_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--config", required=True, help="experiment file (.toml or .json)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (TOML syntax; repeatable)",
    )
    return parser


def _train(argv: Sequence[str]) -> int:
    parser = _config_parser("jla train", "Train a model from an experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the resolved config, then exit")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if args.dry_run:
        sys.stdout.write(config.to_toml())
        print(f"# config_hash = {config.config_hash()}")
        return 0
    logging.basicConfig(level=config.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from jla.launch_env import launch

    launch(config.runtime)
    from jla.training.trainer import Trainer

    result = Trainer(config).run()
    last = result.history[-1] if result.history else {}
    summary = {
        "step": result.step,
        "stop_reason": result.stop_reason,
        "loss": last.get("loss"),
        "eval": result.eval_history[-1] if result.eval_history else None,
    }
    print(json.dumps(summary))
    return 0 if result.stop_reason in ("max_steps", "time_limit", "signal") else 1


def _show_config(argv: Sequence[str]) -> int:
    args = _config_parser("jla show-config", "Print a resolved experiment config and its hash.").parse_args(argv)
    config = load_config(args.config, args.set)
    sys.stdout.write(config.to_toml())
    print(f"# config_hash = {config.config_hash()}")
    return 0


_USAGE = "usage: jla {train,show-config,tokenize,generate,eval,convert,dpo} ..."
_HELP = """
commands:
  train        Train a model from an experiment config.
  show-config  Print a resolved experiment config and its hash.
  tokenize     Tokenize JSONL into a CIDX corpus.
  generate     Sample completions from a checkpoint (or a randomly initialized preset).
  eval         ppl: perplexity of a token stream; lm-eval: lm-evaluation-harness tasks.
  convert      import, export or show-config of Hugging Face checkpoints.
  dpo          Direct preference optimization.

options:
  -h, --help   show this message and exit
  --version    print the version and exit

Run jla <command> --help for the options of a command."""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"train": _train, "show-config": _show_config}
    if args and args[0] in ("-h", "--help"):
        print(_USAGE + "\n" + _HELP)
        return 0
    if args and args[0] == "--version":
        from jla import __version__

        print(f"jla {__version__}")
        return 0
    if args and args[0] == "generate":  # jla.inference.cli
        from jla.inference.cli import main as generate_main

        return generate_main(args[1:])
    if args and args[0] == "eval":  # jla.eval.cli: ppl, lm-eval
        from jla.eval.cli import main as eval_main

        return eval_main(args[1:])
    if args and args[0] == "tokenize":
        from jla.data.make_data import main as tokenize_main

        return tokenize_main(args[1:])
    if args and args[0] == "convert":
        from jla.convert.cli import main as convert_main

        return convert_main(args[1:])
    if args and args[0] == "dpo":
        from jla.training.dpo import main as dpo_main

        return dpo_main(args[1:])
    if not args or args[0] not in commands:
        print(_USAGE + "\n(jla --help lists the commands)", file=sys.stderr)
        return 2
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
