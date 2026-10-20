# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Models and tokenizers for generation and evaluation.

A model comes from a training run directory (``config.toml`` plus
``checkpoints/step_*``, as written by ``jla train``), from an experiment config and
an explicit checkpoint step directory, or from a preset with random
initialization (for tests and smoke runs; the standalone RWKV-7 presets
``tiny-rwkv7-blinkdl`` and ``rwkv7-0.1b-blinkdl`` give a
:class:`jla.inference.rwkv7.RWKV7Decoder`, while ``tiny-rwkv7`` and ``rwkv7-0.1b`` are
:class:`jla.models.LanguageModel` presets). Checkpoint loading
is strict: every parameter must be present with its exact shape and dtype.

Generation and evaluation route MoE layers dropless (:func:`dropless_moe`): with
capacity routing every token of a forward pass, padding included, competes for
expert slots, so a score would depend on the batch size and the padding.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from jla.models import LanguageModel, build_model, get_preset

__all__ = [
    "add_model_arguments",
    "dropless_moe",
    "load_checkpoint_params",
    "load_model",
    "load_tokenizer",
    "model_from_arguments",
    "resolve_step_dir",
]


def resolve_step_dir(run_dir: str | Path, step: int | None = None) -> Path:
    """``run_dir/checkpoints/step_XXXXXXXX`` of ``step`` (default: the latest committed one)."""
    from jla.training.checkpointing import _STEP_DIR, _validated_manifest

    directory = Path(run_dir) / "checkpoints"
    if step is not None:
        step_dir = directory / f"step_{step:08d}"
        _validated_manifest(step_dir)
        return step_dir
    committed = []
    if directory.is_dir():
        for entry in directory.iterdir():
            match = _STEP_DIR.match(entry.name)
            if match is None or not entry.is_dir():
                continue
            try:
                _validated_manifest(entry)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            committed.append((int(match.group(1)), entry))
    if not committed:
        raise FileNotFoundError(f"no committed checkpoint under {directory}")
    return max(committed)[1]


def load_checkpoint_params(model: LanguageModel, step_dir: str | Path) -> LanguageModel:
    """``model`` with its arrays replaced by those of the committed checkpoint ``step_dir``."""
    from jla.training.checkpointing import (
        CheckpointManager,
        _restore_from_files,
        _unflatten_into,
        _validated_manifest,
        flatten_arrays,
    )

    step_dir = Path(step_dir)
    manifest = _validated_manifest(step_dir)
    params, static = eqx.partition(model, eqx.is_array)
    targets = flatten_arrays(params)
    fmt = manifest["payload_format"]
    if fmt == "safetensors":
        restored = _restore_from_files([step_dir / "model.safetensors"], targets, "model", True, chunked=False)
    elif fmt == "sharded":
        files = sorted(step_dir.glob("model.p*-of-*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no model shards in {step_dir}")
        restored = _restore_from_files(files, targets, "model", True, chunked=True)
    elif fmt == "orbax":
        restored = CheckpointManager._restore_orbax(step_dir / "model.orbax", targets)
    else:
        raise ValueError(f"unknown checkpoint payload format {fmt!r}")
    return eqx.combine(_unflatten_into(params, restored), static)


def dropless_moe(model: LanguageModel) -> LanguageModel:
    """``model`` with dropless MoE routing and the same parameters (unchanged without capacity routing).

    Capacity routing (``moe.capacity_factor``) is a training-time load-balancing
    device: decoding already routes dropless, and evaluation and full-forward
    generation must not let rows or padding of one batch displace each other's routes.
    """
    config = model.config
    moe = getattr(config, "moe", None)
    if moe is None or moe.capacity_factor is None:
        return model
    target = config.replace(moe=dataclasses.replace(moe, capacity_factor=None))
    treedef = _DROPLESS_TREEDEFS.get(target)
    if treedef is None:  # the parameter tree of the same architecture, built abstractly
        template = eqx.filter_eval_shape(build_model, target, key=jax.random.key(0))
        leaves, treedef = jax.tree.flatten(template)
        ours = jax.tree.leaves(model)
        if len(ours) != len(leaves) or any(
            eqx.is_array(a) != isinstance(b, jax.ShapeDtypeStruct)
            or (eqx.is_array(a) and (a.shape, a.dtype) != (b.shape, b.dtype))
            for a, b in zip(ours, leaves)
        ):
            raise ValueError("the model's parameter tree differs from build_model(config): cannot switch its routing")
        _DROPLESS_TREEDEFS[target] = treedef
    return jax.tree.unflatten(treedef, jax.tree.leaves(model))


_DROPLESS_TREEDEFS: dict = {}  # dropless ModelConfig -> parameter treedef


def load_model(
    *,
    run_dir: str | Path | None = None,
    step: int | None = None,
    config: str | Path | None = None,
    overrides: tuple[str, ...] = (),
    checkpoint: str | Path | None = None,
    preset: str | None = None,
    seed: int = 0,
    compute_dtype: str | None = None,
) -> LanguageModel:
    """Build a model from exactly one source: ``run_dir``, ``config`` (+ optional ``checkpoint``) or ``preset``.

    ``preset`` (and ``config`` without ``checkpoint``) initialize randomly from ``seed``.
    ``compute_dtype`` overrides the config's compute dtype (parameters keep their dtype).
    """
    sources = [s is not None for s in (run_dir, config, preset)]
    if sum(sources) != 1:
        raise ValueError("pass exactly one of run_dir, config or preset")
    step_dir: Path | None = None
    if preset is not None:
        if checkpoint is not None or step is not None or overrides:
            raise ValueError("a preset model is randomly initialized: checkpoint, step and overrides do not apply")
        from jla.models.rwkv7 import RWKV7_PRESETS, RWKV7LM

        if preset in RWKV7_PRESETS:  # a separate model class (not built from a ModelConfig)
            from jla.inference.rwkv7 import RWKV7Decoder

            rwkv = RWKV7_PRESETS[preset]
            if compute_dtype is not None:
                rwkv = dataclasses.replace(rwkv, compute_dtype=compute_dtype)
            return RWKV7Decoder(RWKV7LM(rwkv, key=jax.random.key(seed)))
        model_config = get_preset(preset)
    else:
        from jla.config import load_config

        if run_dir is not None:
            if checkpoint is not None:
                raise ValueError("run_dir already names its checkpoints; pass step instead of checkpoint")
            model_config = load_config(Path(run_dir) / "config.toml", overrides).model
            step_dir = resolve_step_dir(run_dir, step)
        else:
            if step is not None:
                raise ValueError("step needs run_dir; with config pass checkpoint=<step directory>")
            model_config = load_config(config, overrides).model
            step_dir = None if checkpoint is None else Path(checkpoint)
    if compute_dtype is not None:
        model_config = model_config.replace(compute_dtype=compute_dtype)
    model = build_model(model_config, key=jax.random.key(seed))
    return model if step_dir is None else load_checkpoint_params(model, step_dir)


def load_tokenizer(name: str, *, local_files_only: bool = False) -> Any:
    """A Hugging Face tokenizer (``transformers`` is the optional ``tokenize`` extra)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("tokenizers need the optional dependency: pip install 'jax-linear-attention[tokenize]'") from exc
    return AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model (exactly one of --run-dir, --config, --preset)")
    group.add_argument("--run-dir", help="training output directory (config.toml + checkpoints/)")
    group.add_argument("--step", type=int, help="checkpoint step under --run-dir (default: latest)")
    group.add_argument("--config", help="experiment config file (.toml/.json)")
    group.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                       help="override a config value (with --run-dir or --config; repeatable)")
    group.add_argument("--checkpoint", help="checkpoint step directory to load with --config")
    group.add_argument("--preset", help="model preset with random initialization (tests, smoke runs; "
                       "includes the standalone RWKV-7 presets tiny-rwkv7-blinkdl and rwkv7-0.1b-blinkdl)")
    group.add_argument("--init-seed", type=int, default=0, help="initialization seed for --preset / --config")
    group.add_argument("--compute-dtype", choices=("float32", "bfloat16", "float16"),
                       help="override the model's compute dtype")


def model_from_arguments(args: argparse.Namespace) -> LanguageModel:
    return load_model(run_dir=args.run_dir, step=args.step, config=args.config, overrides=tuple(args.set),
                      checkpoint=args.checkpoint, preset=args.preset, seed=args.init_seed,
                      compute_dtype=args.compute_dtype)
