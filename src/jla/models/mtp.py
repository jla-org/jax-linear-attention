# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-token prediction (MTP) heads trained next to the main LM head.

Depth ``d = 1 .. depth`` predicts token ``t + d + 1`` from position ``t`` (the
main head predicts ``t + 1``). Three layouts (``MTPConfig.kind``):

* ``"deepseek"`` (DeepSeek-V3): ``h_d = Block_d(W_d [norm(h_{d-1}); norm(emb(x_{t+d}))])``,
  a per-depth output norm, and the shared LM head; ``h_0`` is the trunk's
  pre-final-norm hidden state;
* ``"chained"``: the same fusion without the per-depth block and with one shared
  output norm;
* ``"independent"`` (Gloeckle et al. 2024): ``h_d = Block_d(h_0)`` on the shared
  trunk, one shared output norm.

The refiner blocks are the model's own attention block (with RoPE, GQA and
packed-document masking). The token embedding and LM head are the main model's
and receive gradients. Each depth's loss is the mean cross entropy over its
valid labels (``linear_cross_entropy``); the MTP loss is ``loss_weight`` times
the mean over depths that have at least one valid label.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.layers.linear import Embedding, Linear
from jla.layers.norm import RMSNorm
from jla.models.config import ModelConfig, MTPConfig
from jla.models.lm import Block
from jla.ops.cross_entropy import linear_cross_entropy
from jla.distributed.activation_sharding import constrain_batch

__all__ = ["MTPHead"]

Array = jax.Array
IGNORE_INDEX = -100


class MTPHead(eqx.Module):
    # Parts a layout does not use are None (never empty tuples, which look like axes leaves).
    hidden_norms: tuple[RMSNorm, ...] | None  # deepseek / chained: one per depth
    embed_norms: tuple[RMSNorm, ...] | None
    hidden_proj: tuple[Linear, ...] | None  # the two halves of DeepSeek's [2D, D] fusion projection
    embed_proj: tuple[Linear, ...] | None
    blocks: tuple[Block, ...] | None  # deepseek / independent: one per depth
    out_norms: tuple[RMSNorm, ...]  # deepseek: one per depth; chained / independent: one shared
    cfg: MTPConfig = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key: Array):
        cfg = config.mtp
        depth, d = cfg.depth, config.d_model
        param_dtype = jnp.dtype(config.param_dtype)
        keys = jax.random.split(key, 3 * depth)
        common = dict(std=config.initializer_range, param_dtype=param_dtype, compute_dtype=config.compute_dtype)

        def norm():
            return RMSNorm(d, eps=config.norm_eps, param_dtype=param_dtype)

        def proj(i, offset):
            return Linear(d, d, key=keys[3 * i + offset], axes=("embed", None), **common)

        fuse = cfg.kind in ("deepseek", "chained")
        self.hidden_norms = tuple(norm() for _ in range(depth)) if fuse else None
        self.embed_norms = tuple(norm() for _ in range(depth)) if fuse else None
        self.hidden_proj = tuple(proj(i, 0) for i in range(depth)) if fuse else None
        self.embed_proj = tuple(proj(i, 1) for i in range(depth)) if fuse else None
        refine = cfg.kind in ("deepseek", "independent")
        self.blocks = tuple(Block(config, "attention", cfg.mlp_type, key=keys[3 * i + 2]) for i in range(depth)) \
            if refine else None
        self.out_norms = tuple(norm() for _ in range(depth if cfg.kind == "deepseek" else 1))
        self.cfg = cfg
        self.compute_dtype = config.compute_dtype

    def loss(
        self,
        hidden: Array,
        input_ids: Array,
        targets: Array,
        *,
        embed: Embedding,
        head_weight: Array,
        positions: Array,
        segment_ids: Array | None,
        block_fn: Callable[..., tuple[Array, Any]],  # (block, x, positions, segment_ids) -> (x, MoE stats)
        aux_loss_cap: float = 0.0,
        objective: Callable[..., Any] | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        """``hidden [B, T, D]`` is the trunk's pre-final-norm output; ``targets[:, t]`` is token ``t + 1``.

        ``aux_loss_cap > 0`` clips the finite MoE refiner aux loss to ``+-cap`` (the train step's
        cap on the trunk's aux loss); the metric ``mtp/moe_aux_loss`` reports the unclipped value.
        ``objective(hidden [N, D], weight, targets [N], *, vocab_chunk, compute_dtype)``, when given,
        replaces ``linear_cross_entropy`` per depth and returns a
        :class:`~jla.ops.cross_entropy.CrossEntropyObjective` (the train step passes
        ``linear_cross_entropy_objective`` with ``train.ce_grad_in_forward``: the head's gradient formed
        in the forward pass); its ``objective_sum`` (no z-loss) carries the depth loss's gradient.
        """
        cfg = self.cfg
        batch, length, dim = hidden.shape
        if input_ids.shape != (batch, length) or targets.shape != (batch, length):
            raise ValueError("input_ids and targets must match hidden [B, T]")

        def shift_left(x, shift, fill):
            return jnp.pad(x[:, shift:], ((0, 0), (0, shift)), constant_values=fill)

        # Every depth keeps all T positions (the last ``shift`` have no label and a zero
        # conditioning embedding), so the refiner sees the trunk's sequence length: an even
        # length stays on the cuDNN attention path and causality keeps the tail inert.
        h = hidden
        losses, active, metrics = [], [], {}
        moe_aux = []  # weighted MoE losses of "moe" refiner blocks (mlp_type="moe"), one per depth
        for i in range(cfg.depth):
            shift = i + 1  # position t predicts token t + shift + 1 = targets[t + shift]
            if shift >= length:
                break
            labels = shift_left(targets, shift, IGNORE_INDEX)
            valid = labels != IGNORE_INDEX
            if segment_ids is not None:
                valid = valid & (shift_left(segment_ids, shift, -1) == segment_ids)
            if cfg.kind == "independent":
                refined, stats = block_fn(self.blocks[i], hidden, positions, segment_ids)
                moe_aux.append(stats.aux_loss)
                z = self.out_norms[0](refined)
            else:
                live = (jnp.arange(length) < length - shift)[None, :, None]
                # Batch split over the data axes under a mesh, as the trunk (jla.distributed.activation_sharding).
                emb = constrain_batch(jnp.where(live, embed(shift_left(input_ids, shift, 0)), 0))  # t + shift conditions t
                h = self.hidden_proj[i](self.hidden_norms[i](h)) + self.embed_proj[i](self.embed_norms[i](emb))
                h = constrain_batch(h)
                if cfg.kind == "deepseek":
                    h, stats = block_fn(self.blocks[i], h, positions, segment_ids)
                    moe_aux.append(stats.aux_loss)
                z = self.out_norms[i if cfg.kind == "deepseek" else 0](h)
            ce = (linear_cross_entropy if objective is None else objective)(
                z.reshape(-1, dim), head_weight, jnp.where(valid, labels, IGNORE_INDEX).reshape(-1),
                vocab_chunk=cfg.vocab_chunk, compute_dtype=jnp.dtype(self.compute_dtype),
            )
            depth_loss = (ce.loss_sum if objective is None else ce.objective_sum) / jnp.maximum(ce.num_valid, 1.0)
            losses.append(depth_loss)
            active.append((ce.num_valid > 0).astype(jnp.float32))
            metrics[f"mtp/loss_{i + 1}"] = depth_loss
        if not losses:
            return jnp.zeros((), jnp.float32), metrics
        losses_arr, active_arr = jnp.stack(losses), jnp.stack(active)
        mean = jnp.sum(losses_arr * active_arr) / jnp.maximum(jnp.sum(active_arr), 1.0)
        metrics["mtp/loss"] = mean
        if cfg.mlp_type == "moe" and moe_aux:  # the refiners' balance and z-losses, averaged over depths
            aux = jnp.mean(jnp.stack(moe_aux))
            metrics["mtp/moe_aux_loss"] = aux
            if aux_loss_cap > 0:  # non-finite values stay visible so the train step rejects them
                aux = jnp.where(jnp.isfinite(aux), jnp.clip(aux, -aux_loss_cap, aux_loss_cap), aux)
            return cfg.loss_weight * mean + aux, metrics
        return cfg.loss_weight * mean, metrics
