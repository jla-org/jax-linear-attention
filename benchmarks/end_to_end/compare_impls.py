# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one trained checkpoint with the chunked JAX ops and with the Mosaic GPU kernels.

    python compare_impls.py <config.toml> <step_dir> [num_batches]

Same weights, same evaluation batches: any loss difference beyond bf16 noise is
a bias in the kernel path, not training-trajectory drift.
"""

import dataclasses
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.config import load_config
from jla.data.pipeline import eval_batches, eval_dataset
from jla.models import build_model
from jla.training.checkpointing import load_pytree
from jla.training.step import LossConfig, build_eval_step, evaluate

config_path, step_dir = sys.argv[1], sys.argv[2]
num_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cfg = load_config(config_path)
dataset = eval_dataset(cfg)
batch = cfg.train.micro_batch_per_replica


def with_impl(model_cfg, impl):
    changes = {}
    if model_cfg.linear_attention is not None:
        changes["linear_attention"] = dataclasses.replace(model_cfg.linear_attention, implementation=impl)
    if getattr(model_cfg, "ssm", None) is not None:
        changes["ssm"] = dataclasses.replace(model_cfg.ssm, implementation=impl)
    return model_cfg.replace(**changes)


results, hidden = {}, {}
first = {k: jnp.asarray(v) for k, v in next(eval_batches(cfg, dataset, global_batch_size=batch, rows=(0, batch), num_batches=1)).items()}
for impl in ("chunked", "mosaic_gpu"):
    model_cfg = with_impl(cfg.model, impl)
    model = build_model(model_cfg, key=jax.random.key(0))
    params, static = eqx.partition(model, eqx.is_array)
    params = load_pytree(f"{step_dir}/model.safetensors", params)
    model = eqx.combine(params, static)
    step = build_eval_step(LossConfig.from_train_config(cfg.train))
    batches = (
        {k: jnp.asarray(v) for k, v in b.items()}
        for b in eval_batches(cfg, dataset, global_batch_size=batch, rows=(0, batch), num_batches=num_batches)
    )
    results[impl] = evaluate(step, model, batches)
    out = eqx.filter_jit(lambda m, ids: m(ids, inference=True))(model, first["input_ids"])
    hidden[impl] = out.hidden.astype(jnp.float32)

diff = jnp.abs(hidden["mosaic_gpu"] - hidden["chunked"])
scale = jnp.max(jnp.abs(hidden["chunked"]))
print(f"eval loss  chunked {results['chunked']['eval_loss']:.5f}  mosaic_gpu {results['mosaic_gpu']['eval_loss']:.5f}  "
      f"diff {results['mosaic_gpu']['eval_loss'] - results['chunked']['eval_loss']:+.5f}  "
      f"({int(results['chunked']['eval_tokens'])} tokens)")
print(f"final hidden: max |diff| / max |h| = {float(jnp.max(diff) / scale):.2e}, mean |diff| / mean |h| = "
      f"{float(jnp.mean(diff) / jnp.mean(jnp.abs(hidden['chunked']))):.2e}")
