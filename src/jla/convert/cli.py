# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``jla convert``: Hugging Face checkpoints in and out.

    jla convert import --hf path/to/Qwen3-0.6B --output ckpt/qwen3-0.6b
    jla convert export --checkpoint runs/x/step_00001000 --model-config ckpt/qwen3-0.6b/model.toml \\
        --model-type qwen3 --output hf/qwen3-ft --dtype bfloat16 --hf-config ckpt/qwen3-0.6b/hf_config.json \\
        --tokenizer-from ckpt/qwen3-0.6b
    jla convert show-config --hf path/to/Qwen3-0.6B

``import`` writes ``model.safetensors`` (the parameter pytree, loadable with
``jla.training.checkpointing.load_pytree``), ``model.toml`` (the ``[model]`` table
for an experiment config), ``hf_config.json`` (the original config, reused by
``export --hf-config`` to keep token ids and other unmapped fields) and copies of the
tokenizer and ``generation_config.json`` files (``export --tokenizer-from`` copies them back).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def _model_toml(config) -> str:
    from jla.config import _to_plain, dump_toml

    return dump_toml({"model": _to_plain(config)})


def _read_model_config(path: str):
    from jla.config import _model_from_mapping, load_config, load_mapping

    raw = load_mapping(path)
    if set(raw) == {"model"}:
        return _model_from_mapping(raw["model"])
    return load_config(path).model


def _overrides(args: argparse.Namespace) -> dict:
    out = {}
    if args.param_dtype:
        out["param_dtype"] = args.param_dtype
    if args.compute_dtype:
        out["compute_dtype"] = args.compute_dtype
    if args.no_scan_layers:
        out["scan_layers"] = False
    return out


def _import(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="jla convert import", description="Hugging Face checkpoint -> jla parameters.")
    parser.add_argument("--hf", required=True, help="directory with config.json and *.safetensors")
    parser.add_argument("--output", required=True, help="directory for model.safetensors, model.toml, hf_config.json")
    parser.add_argument("--param-dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--no-scan-layers", action="store_true", help="store blocks as a list (scan_layers=False)")
    args = parser.parse_args(argv)
    import numpy as np

    from jla.convert.hf import copy_tokenizer_files, load_hf_checkpoint
    from jla.convert.hf_config import read_hf_config
    from jla.convert.safetensors_io import write_safetensors
    from jla.training.checkpointing import flatten_arrays

    model, config = load_hf_checkpoint(args.hf, **_overrides(args))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    hf = read_hf_config(args.hf)
    # The keys of jla.training.checkpointing.save_pytree; this writer also stores bfloat16 directly.
    arrays = {name: np.asarray(value) for name, value in flatten_arrays(model).items()}
    write_safetensors(out / "model.safetensors", arrays, {"hf_model_type": hf["model_type"]})
    (out / "model.toml").write_text(_model_toml(config))
    (out / "hf_config.json").write_text(json.dumps(hf, indent=2, sort_keys=True) + "\n")
    copied = copy_tokenizer_files(args.hf, out)
    count = sum(int(value.size) for value in arrays.values())
    print(json.dumps({"model_type": hf["model_type"], "parameters": count, "output": str(out), "copied": copied}))
    return 0


def _export(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="jla convert export", description="jla parameters -> Hugging Face checkpoint.")
    parser.add_argument("--checkpoint", required=True,
                        help="model.safetensors, or a directory holding one (an import output or a trainer step_* dir)")
    parser.add_argument("--model-config", required=True, help="model.toml from import, or an experiment config")
    parser.add_argument("--model-type", required=True, help="Hugging Face model_type (llama, qwen2, qwen3, ...)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), help="default: the parameter dtype")
    parser.add_argument("--hf-config", help="original config.json whose unmapped fields are kept")
    parser.add_argument("--moe-format", choices=("hub", "fused"), default="hub")
    parser.add_argument("--max-shard-gb", type=float, default=5.0)
    parser.add_argument("--tokenizer-from", help="directory whose tokenizer and generation_config files are copied")
    args = parser.parse_args(argv)
    from jla.convert.hf import copy_tokenizer_files, save_hf_checkpoint
    from jla.convert.hf_weights import unstack_layers
    from jla.convert.safetensors_io import SafetensorsFile

    path = Path(args.checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    config = _read_model_config(args.model_config)
    tensors = SafetensorsFile(path)
    params = unstack_layers({name: tensors[name] for name in tensors}, config)
    out = save_hf_checkpoint(params, args.output, args.model_type, config=config, dtype=args.dtype,
                             base_config=args.hf_config, moe_format=args.moe_format,
                             max_shard_bytes=int(args.max_shard_gb * 1024**3))
    copied = copy_tokenizer_files(args.tokenizer_from, out) if args.tokenizer_from else []
    print(json.dumps({"model_type": args.model_type, "output": str(out), "copied": copied}))
    return 0


def _show_config(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="jla convert show-config", description="Print the [model] table of an HF config.")
    parser.add_argument("--hf", required=True, help="config.json or a directory holding one")
    args = parser.parse_args(argv)
    from jla.convert.hf_config import from_hf_config

    sys.stdout.write(_model_toml(from_hf_config(args.hf)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"import": _import, "export": _export, "show-config": _show_config}
    if args and args[0] in ("-h", "--help"):
        print("usage: jla convert {import,export,show-config} ...\n\n"
              "  import       Hugging Face checkpoint -> jla parameters.\n"
              "  export       jla parameters -> Hugging Face checkpoint.\n"
              "  show-config  Print the [model] table of an HF config.\n\n"
              "Run jla convert <command> --help for its options.")
        return 0
    if not args or args[0] not in commands:
        print("usage: jla convert {import,export,show-config} ...", file=sys.stderr)
        return 2
    return commands[args[0]](args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
