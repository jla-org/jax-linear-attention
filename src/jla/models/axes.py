# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logical axis names of every parameter, and parameter counts.

Every module that owns arrays declares their axes in ``param_axes()``.
:func:`logical_axes` returns a pytree shaped like the model in which each array
leaf is replaced by its tuple of axis names, so
``jax.tree.map(rule, model, logical_axes(model))`` visits ``(array, axes)``
pairs. Names used:

``"layers"`` (the stacked axis of scanned blocks), ``"vocab"``, ``"embed"``,
``"heads"`` (fused query / mixer heads), ``"kv_heads"`` (fused key/value heads
of attention), ``"mlp"``, ``"experts"``, and ``None`` (not sharded).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax

from jla.models.config import ModelConfig

__all__ = ["LOGICAL_AXIS_NAMES", "ParameterCount", "logical_axes", "parameter_count"]

LOGICAL_AXIS_NAMES = ("layers", "vocab", "embed", "heads", "kv_heads", "mlp", "experts")


def _owner(root, path):
    node = root
    for entry in path[:-1]:
        if isinstance(entry, jax.tree_util.GetAttrKey):
            node = getattr(node, entry.name)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            node = node[entry.idx]
        elif isinstance(entry, jax.tree_util.DictKey):
            node = node[entry.key]
        else:
            raise TypeError(f"unsupported pytree path entry {entry!r}")
    return node


def logical_axes(model: eqx.Module) -> eqx.Module:
    """Pytree like ``model`` with each array leaf replaced by its logical axis names."""
    scanned = bool(getattr(getattr(model, "config", None), "scan_layers", False))

    def axes_of(path, leaf):
        if not (eqx.is_array(leaf) or isinstance(leaf, jax.ShapeDtypeStruct)):
            return leaf
        owner, field = _owner(model, path), path[-1]
        declared = owner.param_axes() if hasattr(owner, "param_axes") else {}
        if not isinstance(field, jax.tree_util.GetAttrKey) or field.name not in declared:
            raise TypeError(f"array at {jax.tree_util.keystr(path)} has no declared logical axes")
        axes = tuple(declared[field.name])
        stacked = scanned and isinstance(path[0], jax.tree_util.GetAttrKey) and path[0].name == "blocks"
        if stacked:
            axes = ("layers", *axes)
        if len(axes) != leaf.ndim:
            raise ValueError(f"{jax.tree_util.keystr(path)}: axes {axes} do not match shape {leaf.shape}")
        return axes

    return jax.tree_util.tree_map_with_path(axes_of, model)


class ParameterCount(NamedTuple):
    total: int
    active: int  # parameters used per token (MoE counts top_k + shared experts)
    embedding: int


def parameter_count(config: ModelConfig) -> ParameterCount:
    """Count parameters without allocating them (``jax.eval_shape``)."""
    from jla.models.lm import build_model

    shapes = jax.eval_shape(lambda: build_model(config, key=jax.random.key(0)))
    # Integer leaves (the lightning layers' depths) are not parameters.
    total = sum(int(x.size) for x in jax.tree.leaves(shapes)
                if isinstance(x, jax.ShapeDtypeStruct) and jax.numpy.issubdtype(x.dtype, jax.numpy.inexact))
    inactive = 0
    if config.moe is not None:
        moe = config.moe
        per_expert = 3 * config.d_model * moe.d_ff_expert
        num_moe_layers = sum(t == "moe" for t in config.resolved_mlp_types)
        mtp = config.mtp
        if mtp is not None and mtp.mlp_type == "moe" and mtp.kind != "chained":  # MoE refiner blocks
            num_moe_layers += mtp.depth
        inactive = num_moe_layers * (moe.num_experts - moe.top_k) * per_expert
    embedding = config.vocab_size * config.d_model * (1 if config.tie_embeddings else 2)
    return ParameterCount(total=total, active=total - inactive, embedding=embedding)
