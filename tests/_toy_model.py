# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared training-test helpers.

* :func:`experiment` builds a small synthetic-data :class:`ExperimentConfig`
  around a real ``jla.models`` preset (``tiny`` by default);
* :class:`ToyLM` is a minimal model with the ``jla.models.LanguageModel``
  interface (embedding, one dense residual layer, RMSNorm, tied head) for step
  and checkpoint unit tests that need exact float32 arithmetic.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.config import DataConfig, ExperimentConfig, MeshConfig, OptimConfig, RuntimeConfig, TrainConfig
from jla.models import get_preset
from jla.models.config import AttentionConfig, ModelConfig

TOY_MODEL = ModelConfig(
    vocab_size=128,
    d_model=64,
    n_layers=1,
    d_ff=128,
    attention=AttentionConfig(num_heads=2, num_kv_heads=1, head_dim=32),
    param_dtype="float32",
    compute_dtype="float32",
    remat="none",
)


class LMOutput(NamedTuple):
    hidden: jax.Array
    pre_norm_hidden: jax.Array
    aux_loss: jax.Array
    metrics: dict[str, jax.Array]


class ToyLM(eqx.Module):
    config: ModelConfig = eqx.field(static=True)
    embed: jax.Array
    proj: jax.Array
    norm: jax.Array

    def __init__(self, config: ModelConfig, *, key: jax.Array) -> None:
        embed_key, proj_key = jax.random.split(key)
        dtype = jnp.dtype(config.param_dtype)
        self.config = config
        self.embed = (jax.random.normal(embed_key, (config.vocab_size, config.d_model)) * 0.02).astype(dtype)
        self.proj = (jax.random.normal(proj_key, (config.d_model, config.d_model)) * config.d_model**-0.5).astype(dtype)
        self.norm = jnp.ones((config.d_model,), dtype)

    def __call__(self, input_ids, *, positions=None, segment_ids=None, key=None, inference=False) -> LMOutput:
        del positions, segment_ids, key, inference
        dtype = jnp.dtype(self.config.compute_dtype)
        x = jnp.take(self.embed, input_ids, axis=0).astype(dtype)
        pre = x + jnp.tanh(x @ self.proj.astype(dtype))
        h32 = pre.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(h32 * h32, axis=-1, keepdims=True) + 1e-6)
        hidden = (h32 * rms * self.norm.astype(jnp.float32)).astype(dtype)
        return LMOutput(hidden, pre, jnp.zeros((), jnp.float32), {"hidden_rms": jnp.sqrt(jnp.mean(h32 * h32))})

    def lm_head_weight(self) -> jax.Array:
        return self.embed

    def logits(self, hidden: jax.Array) -> jax.Array:
        return hidden.astype(jnp.float32) @ self.embed.astype(jnp.float32).T

    def mtp_loss(self, out, input_ids, targets, *, positions=None, segment_ids=None):
        del out, input_ids, targets, positions, segment_ids
        return jnp.zeros((), jnp.float32), {}


def build_toy(config: ModelConfig, *, key: jax.Array) -> ToyLM:
    return ToyLM(config, key=key)


def toy_logical_axes(model: Any) -> Any:
    return eqx.tree_at(lambda m: (m.embed, m.proj, m.norm), model, (("vocab", "embed"), ("embed", "mlp"), (None,)))


TOY_COMPONENTS = {"model_builder": build_toy, "logical_axes_fn": toy_logical_axes}


def experiment(output_dir: str, *, model: ModelConfig | str = "tiny", **sections: Any) -> ExperimentConfig:
    """Small synthetic-data experiment; ``sections`` maps section names to field overrides."""
    train = {
        "max_steps": 30,
        "seq_len": 32,
        "micro_batch_per_replica": 8,
        "grad_accum": 2,
        "log_every": 10,
        "eval_every": 0,
        "eval_batches": 2,
        "save_every": 0,
        "output_dir": output_dir,
        **sections.get("train", {}),
    }
    optim = {"lr": 1e-2, "warmup_steps": 2, "weight_decay": 0.0, "schedule": "cosine", **sections.get("optim", {})}
    data = {"num_workers": 0, "synthetic_samples": 4096, **sections.get("data", {})}
    mesh = {"fsdp": -1, **sections.get("mesh", {})}
    return ExperimentConfig(
        model=get_preset(model) if isinstance(model, str) else model,
        name="test",
        optim=OptimConfig(**optim),
        train=TrainConfig(**train),
        data=DataConfig(**data),
        mesh=MeshConfig(**mesh),
        runtime=RuntimeConfig(**sections.get("runtime", {})),
    )
