# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load Hugging Face checkpoints into :class:`LanguageModel` and save models back as HF checkpoints.

    model, config = load_hf_checkpoint("path/to/Qwen3-0.6B", compute_dtype="bfloat16")
    save_hf_checkpoint(model, "out/qwen3-ft", "qwen3", dtype="bfloat16", base_config="path/to/Qwen3-0.6B")

Nothing is downloaded: the directory must hold ``config.json`` and ``model.safetensors``
(or shards with ``model.safetensors.index.json``). Tokenizer files are not touched.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from jla.convert.hf_config import from_hf_config, read_hf_config, to_hf_config
from jla.convert.hf_weights import export_hf_state, import_hf_state, stack_layers, unstack_layers
from jla.convert.safetensors_io import load_checkpoint_dir, save_checkpoint_dir
from jla.models.config import ModelConfig
from jla.models.lm import LanguageModel

__all__ = ["TOKENIZER_FILES", "assemble_model", "copy_tokenizer_files", "load_hf_checkpoint", "model_arrays", "save_hf_checkpoint"]

_NP_DTYPES = {"float32": np.dtype(np.float32), "bfloat16": np.dtype(ml_dtypes.bfloat16), "float16": np.dtype(np.float16)}
_FP32_KEYS = (".e_score_correction_bias",)


def _flatten(tree: Any) -> dict[str, Any]:
    from jla.training.checkpointing import flatten_arrays

    return flatten_arrays(tree)


def assemble_model(config: ModelConfig, params: Mapping[str, np.ndarray]) -> LanguageModel:
    """A :class:`LanguageModel` whose every parameter comes from ``params`` (``layers.<i>.<path>`` names).

    Keys and shapes must match the model exactly; values are cast to the dtype the model
    builds each parameter with (``config.param_dtype``; float32 for the MoE router bias).
    """
    skeleton = eqx.filter_eval_shape(LanguageModel, config, key=jax.random.PRNGKey(0))
    targets = _flatten(skeleton)
    flat = stack_layers(params, config)
    missing, extra = sorted(set(targets) - set(flat)), sorted(set(flat) - set(targets))
    if missing or extra:
        raise ValueError(f"converted parameters do not match the model: missing={missing[:8]} unexpected={extra[:8]}")
    host = {}
    for name, target in targets.items():
        value = np.asarray(flat[name])
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"{name}: converted shape {tuple(value.shape)} != model shape {tuple(target.shape)}")
        host[name] = value.astype(np.dtype(target.dtype))  # an owned copy (never a view of the mapped file)
    arrays = jax.device_put(host)  # one batched host-to-device copy

    from jla.distributed.sharding import path_names

    def fill(path, leaf):
        name = ".".join(path_names(path))
        return arrays[name] if name in arrays else leaf

    model = jax.tree_util.tree_map_with_path(fill, skeleton)
    left = [leaf for leaf in jax.tree_util.tree_leaves(model) if isinstance(leaf, jax.ShapeDtypeStruct)]
    if left:
        raise ValueError(f"{len(left)} model leaves were not filled by the conversion")
    return model


def model_arrays(model: LanguageModel, dtype: str | None = None) -> dict[str, np.ndarray]:
    """The model's parameters as host arrays in the ``layers.<i>.<path>`` naming.

    ``dtype`` casts on the device before the copy to the host (the float32 MoE router
    bias excepted), which halves the transfer for a bfloat16 export.
    """
    flat = _flatten(model)
    if dtype is not None:
        target = jnp.dtype(dtype)
        flat = {k: v if k.endswith("router_bias") or v.dtype == target else v.astype(target) for k, v in flat.items()}
    return unstack_layers({k: np.asarray(v) for k, v in jax.device_get(flat).items()}, model.config)


def load_hf_checkpoint(directory: str | Path, **overrides: Any) -> tuple[LanguageModel, ModelConfig]:
    """Import ``directory`` (``config.json`` + safetensors) as ``(model, config)``.

    ``overrides`` go to :func:`from_hf_config` (``param_dtype``, ``compute_dtype``,
    ``scan_layers``, ``remat``, ...).
    """
    hf = read_hf_config(directory)
    config = from_hf_config(hf, **overrides)
    params = import_hf_state(load_checkpoint_dir(directory), config, hf["model_type"],
                             rope_interleave=bool(hf.get("rope_interleave", True)))
    return assemble_model(config, params), config


def save_hf_checkpoint(
    model: LanguageModel | Mapping[str, np.ndarray],
    directory: str | Path,
    model_type: str,
    *,
    config: ModelConfig | None = None,
    dtype: str | None = None,
    base_config: str | Path | Mapping[str, Any] | None = None,
    moe_format: str = "hub",
    max_shard_bytes: int = 5 * 1024**3,
) -> Path:
    """Write ``config.json`` and safetensors for ``model`` as a ``model_type`` checkpoint.

    ``model`` is a :class:`LanguageModel` or its :func:`model_arrays` (then pass
    ``config``). ``dtype`` defaults to the parameter dtype; ``base_config`` keeps the
    unmapped fields (token ids, ``max_position_embeddings``) of an original config.
    """
    if isinstance(model, LanguageModel):
        dtype = dtype or model.config.param_dtype
        if dtype not in _NP_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_NP_DTYPES)}; got {dtype!r}")
        config, params = model.config, model_arrays(model, dtype)
    else:
        if config is None:
            raise ValueError("pass config= when saving a parameter mapping")
        params = dict(model)
    dtype = dtype or config.param_dtype
    if dtype not in _NP_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_NP_DTYPES)}; got {dtype!r}")
    base = None if base_config is None else read_hf_config(base_config)
    hf_config = to_hf_config(config, model_type, dtype=dtype, base=base)
    state = export_hf_state(params, config, model_type, moe_format=moe_format,
                            rope_interleave=bool(hf_config.get("rope_interleave", True)))
    target = _NP_DTYPES[dtype]
    # transformers keeps DeepSeek-V3's expert-choice bias in float32 whatever the checkpoint dtype.
    state = {k: np.ascontiguousarray(np.asarray(v).astype(np.float32 if k.endswith(_FP32_KEYS) else target, copy=False))
             for k, v in state.items()}
    directory = Path(directory)
    save_checkpoint_dir(directory, state, max_shard_bytes=max_shard_bytes)
    (directory / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True) + "\n")
    return directory


# Files besides the weights that make an HF directory usable (tokenizer, chat template, generation defaults).
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
    "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json", "generation_config.json",
)


def copy_tokenizer_files(source: str | Path, destination: str | Path) -> list[str]:
    """Copy the :data:`TOKENIZER_FILES` present in ``source`` to ``destination``; returns their names."""
    import shutil

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TOKENIZER_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
            copied.append(name)
    return copied
