# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MaxText (Apache-2.0), src/maxtext/layers/pipeline.py (stage-sharded scanned weights,
# a microbatch loop inside the compiled step, activation shifts between stages with lax.ppermute) and
# src/maxtext/configs/base.yml (the logits batch sharded over the stage axis outside the pipeline)
"""Pipeline parallelism over the ``pp`` mesh axis (GPipe schedule).

The scanned layer stack of :class:`jla.models.LanguageModel` stores every
parameter of the repeating group stacked over ``R = n_layers / period``
repeats. With ``pp = S`` stages the repeats split into ``S`` contiguous stages
of ``R / S`` repeats each; :func:`pipeline_rules` maps the ``"layers"`` logical
axis to the ``pp`` mesh axis so stage ``s`` stores only its own layers.

:func:`pipeline_lm_forward` replaces the layer scan of the model's forward
pass. The ``[B, T, D]`` activations after the embedding split into ``M``
microbatches (rows ``j * M + m`` form microbatch ``m``, which keeps the batch
sharding over ``dp``/``fsdp`` local). A ``shard_map`` that is manual over
``pp`` only (``dp``, ``fsdp`` and ``tp`` stay under the automatic partitioner,
so the pipeline composes with data, FSDP and tensor parallelism) runs
``M + S - 1`` iterations; at iteration ``t`` stage ``s`` applies its layers to
microbatch ``t - s`` and ``lax.ppermute`` hands the result to stage ``s + 1``.
The last stage collects the outputs; the embedding, final norm, LM head, MTP
and loss run outside the pipeline exactly as without it. With ``batch_axes``
(the mesh axes that split the batch) the last stage's outputs are
reduce-scattered over the stages (``lax.psum_scatter``) so that the final norm,
LM head, loss and MTP run sharded over ``batch_axes + (pp,)``, as MaxText's
``activation_embed_and_logits_batch`` rule does, instead of every stage
recomputing them for the whole batch; this needs ``B / prod(batch_axes)``
divisible by ``M * S`` and falls back to replicated outputs otherwise
(:func:`shards_head_over_stages`).

Given the logical-axis rules, the ``shard_map`` is manual over every mesh axis
when only ``pp`` and the batch axes are larger than one: each device runs its
stage on its own rows, every block parameter is all-gathered over its FSDP axes
layer by layer (rematerialized in the backward pass, whose transpose
reduce-scatters the gradients) and nothing inside is left to the automatic
partitioner, so custom-partitioned ops (cuDNN attention) and Pallas kernels run
as they do on one device. With ``tp > 1``, MoE layers on a split batch
(their routing statistics are global over the batch) or Mamba-1 layers (XLA
cannot compile their gradient fully manual), it is manual over ``pp`` only and
``dp``/``fsdp``/``tp`` stay automatic. ``model.remat`` applies on both paths.

Reverse-mode autodiff transposes the loop (the transpose of a ppermute is the
reverse ppermute, of the reduce-scatter an all-gather), so the backward pass
runs the same schedule in reverse: GPipe, with an idle (bubble) fraction of
``(S - 1) / (M + S - 1)`` per stage, reported as the step metric
``pipeline/bubble_fraction``.

Exactness: every mixer and dense MLP is row-independent, so the pipelined
forward computes the same function as the plain scan (same rows, same order
after the outputs are reassembled). The stages also run on the bubble
iterations, whose outputs are discarded; that adds no gradient only for layers
whose VJP is linear in the cotangent, so :func:`check_pipeline_layers` refuses
the DSA indexer loss (``sparse_attention.indexer_loss_coef > 0``) for
``pp > 1``. RWKV-7 stacks are refused: the stage loop does not carry their
value residual. MoE layers route each microbatch on its
own tokens (expert capacity, balance and router z-loss statistics per
microbatch, the statistics averaged over microbatches), so they equal the plain
model's only for ``M = 1``. Residual dropout draws a different (but
identically distributed) mask per microbatch: the key of layer ``l`` is folded
with the microbatch index. The sharded head changes only the order of the loss
and gradient reductions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

__all__ = [
    "BUBBLE_METRIC",
    "PIPELINE_AXIS",
    "bubble_fraction",
    "check_pipeline_attention",
    "check_pipeline_layers",
    "make_pipeline_forward",
    "pipeline_lm_forward",
    "pipeline_rules",
    "resolve_microbatches",
    "runs_fully_manual",
    "shards_head_over_stages",
    "stage_layout",
]

#: Mesh axis that carries the pipeline stages.
PIPELINE_AXIS = "pp"
#: Step metric holding the schedule's idle fraction.
BUBBLE_METRIC = "pipeline/bubble_fraction"
_GATHER_NAME = "pipeline_fsdp_gather"
_KEEP_ALL_BUT_GATHER = jax.checkpoint_policies.save_anything_except_these_names(_GATHER_NAME)
# Mixers whose gradient XLA cannot compile inside a fully manual shard_map (Mamba-1's
# selective scan segfaults the compiler, JAX 0.11.2): they run manual over pp only.
_PP_ONLY_MANUAL_MIXERS = ("mamba1",)
# RWKV-7 layers read the first layer's values (the value-residual carry); the stage loop does
# not thread that carry between layers, so pipelining such a stack would change the model.
_NO_VALUE_RESIDUAL = (
    "pipeline parallelism does not carry RWKV-7's value residual (v_first) between layers, so a pipelined "
    "'rwkv7' stack would compute a different model; use mesh.pp = 1 for RWKV-7 layers"
)

Array = jax.Array


def _layer_policy(model_policy: Any) -> Any:
    """Remat policy of a fully manual layer: ``model_policy`` minus the all-gathered weights.

    A ``jax.checkpoint`` nested inside another one is partially evaluated with the
    OUTER policy (JAX 0.11.2), so the wrapper that names the FSDP gathers must carry
    the model's ``remat`` policy; otherwise every activation of every layer and
    iteration is saved. ``None`` (``remat = "none"``) saves all but the gathered weights.
    """
    if model_policy is None:
        return _KEEP_ALL_BUT_GATHER

    def policy(prim: Any, *args: Any, **params: Any) -> Any:
        return _KEEP_ALL_BUT_GATHER(prim, *args, **params) and model_policy(prim, *args, **params)

    return policy


def _moe_on_split_batch(cfg: Any, mesh: Mesh, batch_axes: Sequence[str]) -> bool:
    """MoE layers with the batch split over more than one mesh axis (e.g. ``dp`` and ``fsdp``)."""
    split = [name for name in batch_axes if int(mesh.shape[name]) > 1]
    return len(split) > 1 and any(t == "moe" for t in cfg.resolved_mlp_types)


def resolve_microbatches(num_stages: int, num_microbatches: int) -> int:
    """Pipeline microbatches per forward pass; ``0`` means one per stage (``M = S``)."""
    if num_stages < 1:
        raise ValueError(f"num_stages={num_stages} must be >= 1")
    if num_microbatches < 0:
        raise ValueError(f"num_microbatches={num_microbatches} must be >= 0 (0 = one per stage)")
    return num_stages if num_microbatches == 0 else num_microbatches


def bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Idle fraction of a GPipe schedule: ``(S - 1) / (M + S - 1)`` (forward and backward alike)."""
    microbatches = resolve_microbatches(num_stages, num_microbatches)
    return (num_stages - 1) / (microbatches + num_stages - 1)


def pipeline_rules(rules: Mapping[str, Sequence[str]], axis: str = PIPELINE_AXIS) -> dict[str, tuple[str, ...]]:
    """``rules`` with the stacked ``"layers"`` axis sharded over ``axis`` (stage ``s`` keeps its layers)."""
    out = {name: tuple(axes) for name, axes in rules.items()}
    layers = out.get("layers", ())
    if axis not in layers:
        out["layers"] = (axis, *layers)
    return out


def stage_layout(config: Any, num_stages: int) -> tuple[int, int, int]:
    """``(period, repeats, repeats_per_stage)`` of a scanned model split into ``num_stages`` stages."""
    from jla.models.lm import layer_period

    if not config.scan_layers:
        raise ValueError("pipeline parallelism needs scan_layers=True (stages are slices of the layer scan)")
    period = layer_period(config)
    repeats = config.n_layers // period
    if num_stages < 1 or repeats % num_stages:
        raise ValueError(
            f"pp={num_stages} must divide the {repeats} repeats of the {period}-layer group "
            f"(n_layers={config.n_layers})"
        )
    return period, repeats, repeats // num_stages


def _microbatch(a: Array, microbatches: int) -> Array:
    """``[B, ...] -> [M, B / M, ...]``; microbatch ``m`` holds rows ``j * M + m``."""
    rows = a.shape[0] // microbatches
    return jnp.swapaxes(a.reshape(rows, microbatches, *a.shape[1:]), 0, 1)


def _unmicrobatch(a: Array) -> Array:
    """Inverse of :func:`_microbatch`."""
    return jnp.swapaxes(a, 0, 1).reshape(-1, *a.shape[2:])


def _split_stages(a: Array, num_stages: int) -> Array:
    return a.reshape(num_stages, a.shape[0] // num_stages, *a.shape[1:])


def _batch_shards(mesh: Mesh, batch_axes: Sequence[str]) -> int:
    unknown = [name for name in batch_axes if name not in mesh.shape]
    if unknown:
        raise ValueError(f"batch_axes {unknown} are not axes of the mesh {tuple(mesh.axis_names)}")
    shards = 1
    for name in batch_axes:
        shards *= int(mesh.shape[name])
    return shards


class _Dims:
    """Pytree leaf: the manual mesh axes of each dimension of one stacked block parameter."""

    def __init__(self, dims: tuple[Any, ...]):
        self.dims = dims


def _block_dims(model: Any, params: Any, mesh: Mesh, rules: Mapping[str, Sequence[str]]) -> Any:
    """``_Dims`` of every array leaf of ``params`` (``model.blocks``) under the logical ``rules``."""
    from jla.distributed.sharding import is_axes_leaf, logical_to_spec, path_names
    from jla.models import logical_axes

    flat = jax.tree_util.tree_flatten_with_path(logical_axes(model).blocks, is_leaf=is_axes_leaf)[0]
    names = {path_names(path): axes for path, axes in flat if is_axes_leaf(axes)}

    def dims(path, leaf):
        spec = tuple(logical_to_spec(names[path_names(path)], leaf.shape, mesh, rules))
        return _Dims(spec + (None,) * (leaf.ndim - len(spec)))

    return jax.tree_util.tree_map_with_path(dims, params)


def _gather_dims(leaf: Array, dims: tuple[Any, ...]) -> Array:
    """All-gather a parameter over the (FSDP) mesh axes that shard its dimensions (tiled)."""
    for dim, entry in enumerate(dims):
        if entry is not None:
            leaf = lax.all_gather(leaf, entry, axis=dim, tiled=True)
    return leaf


def _manual_batch(cfg: Any, batch: int, microbatches: int, mesh: Mesh, axis: str,
                  batch_axes: Sequence[str] | None, rules: Mapping[str, Sequence[str]] | None) -> bool:
    """Whether the pipeline ``shard_map`` can be manual over every mesh axis (see the module docstring)."""
    if rules is None or batch_axes is None:
        return False
    if any(int(size) > 1 for name, size in mesh.shape.items() if name != axis and name not in batch_axes):
        return False  # tensor parallelism stays with the automatic partitioner
    shards = _batch_shards(mesh, batch_axes)
    if shards > 1 and any(t == "moe" for t in cfg.resolved_mlp_types):
        return False  # MoE routing statistics are global over the batch shards
    if any(t in _PP_ONLY_MANUAL_MIXERS for t in cfg.resolved_layer_types):
        return False
    return batch % (microbatches * shards) == 0


def check_pipeline_attention(cfg: Any, mesh: Mesh) -> None:
    """cuDNN attention cannot run inside a multi-device pipeline on GPU (JAX 0.11.2).

    In a partially manual ``shard_map`` its custom partitioning fails to convert the
    shardings; in a fully manual one the XLA GPU compiler crashes on its backward pass.
    MLA layers call the same attention op with ``model.mla.implementation``.
    """
    if mesh.size == 1 or jax.default_backend() != "gpu":
        return
    from jla.models.lm import ATTENTION_TYPES

    for layer_type in sorted(set(cfg.resolved_layer_types) & {*ATTENTION_TYPES, "mla"}):
        if layer_type == "mla":
            attn = cfg.mla
        else:
            attn = cfg.attention if layer_type == "attention" else cfg.local_attention
        if attn.implementation in ("auto", "cudnn"):
            name = layer_type if layer_type in ("attention", "mla") else "local_attention"
            raise ValueError(
                f"pipeline parallelism on GPU needs model.{name}.implementation = 'xla' or 'mosaic_gpu' "
                f"(got {attn.implementation!r}): cuDNN attention does not compile inside the pipeline's shard_map"
            )


def check_pipeline_layers(cfg: Any, num_stages: int) -> None:
    """Refuse layers whose backward pass adds gradient whatever the cotangent.

    Every stage runs all ``M + S - 1`` iterations and discards its output on the
    ``S - 1`` bubble iterations, which zeroes their cotangents: a layer adds no
    gradient there only if its VJP is linear in the cotangent. The DSA indexer's KL
    gradient (``sparse_attention.indexer_loss_coef > 0``) is not. RWKV-7 layers are refused too:
    the stage loop does not carry their value residual.
    """
    if num_stages > 1 and "rwkv7" in cfg.resolved_layer_types:
        raise ValueError(_NO_VALUE_RESIDUAL)
    if num_stages > 1 and "dsa" in cfg.resolved_layer_types and cfg.sparse_attention.indexer_loss_coef > 0:
        raise ValueError(
            "pipeline parallelism (pp > 1) cannot train the DSA indexer: its KL gradient does not depend on the "
            "cotangent, so the pipeline's bubble iterations would add spurious indexer gradients; set "
            "model.sparse_attention.indexer_loss_coef = 0 or mesh.pp = 1"
        )


def runs_fully_manual(
    config: Any,
    batch: int,
    mesh: Mesh,
    *,
    num_microbatches: int = 0,
    batch_axes: Sequence[str] | None,
    rules: Mapping[str, Sequence[str]] | None,
    axis: str = PIPELINE_AXIS,
) -> bool:
    """Whether :func:`pipeline_lm_forward` runs the stages manual over every mesh axis for this batch."""
    microbatches = resolve_microbatches(int(mesh.shape[axis]), num_microbatches)
    return _manual_batch(config, batch, microbatches, mesh, axis, batch_axes, rules)


def shards_head_over_stages(
    batch: int,
    mesh: Mesh,
    *,
    num_microbatches: int = 0,
    batch_axes: Sequence[str] | None,
    axis: str = PIPELINE_AXIS,
    config: Any = None,
) -> bool:
    """Whether :func:`pipeline_lm_forward` runs the head and loss sharded over ``axis`` for this batch.

    True when ``batch_axes`` is given and each of the ``prod(batch_axes)`` batch shards
    holds a multiple of ``M * S`` rows (``M`` microbatches, ``S`` stages), unless the
    model ``config`` has MoE layers and the batch is split over more than one mesh axis
    (XLA's SPMD partitioner aborts on that layout, JAX 0.11.2; the head stays replicated).
    """
    if batch_axes is None:
        return False
    stages = int(mesh.shape[axis])
    microbatches = resolve_microbatches(stages, num_microbatches)
    shards = _batch_shards(mesh, batch_axes)
    if config is not None and _moe_on_split_batch(config, mesh, batch_axes):
        return False
    return batch % (shards * microbatches * stages) == 0


def pipeline_lm_forward(
    model: Any,
    input_ids: Array,
    *,
    mesh: Mesh,
    num_microbatches: int = 0,
    positions: Array | None = None,
    segment_ids: Array | None = None,
    key: Array | None = None,
    inference: bool = False,
    axis: str = PIPELINE_AXIS,
    batch_axes: Sequence[str] | None = None,
    rules: Mapping[str, Sequence[str]] | None = None,
    shard_head: bool = True,
) -> Any:
    """:meth:`LanguageModel.__call__` with the layer scan pipelined over ``mesh`` axis ``axis``.

    Returns the same :class:`jla.models.LMOutput`; ``metrics`` gains
    ``pipeline/bubble_fraction``. ``num_microbatches`` (``0`` = one per stage)
    must divide the batch. ``batch_axes`` (e.g.
    ``jla.distributed.sharding.batch_axes(mesh, rules)``) returns the hidden
    states sharded over ``batch_axes + (axis,)`` when
    :func:`shards_head_over_stages` holds, so the head and loss are not
    recomputed on every stage; otherwise they are replicated over ``axis``.
    With ``batch_axes`` and the logical-axis ``rules`` the parameters are sharded
    with, the ``shard_map`` is manual over every mesh axis when no other axis
    (``tp``) is larger than one, no MoE layer meets a split batch and every batch
    shard holds a multiple of ``M`` rows; it is manual over ``axis`` only otherwise.
    ``shard_head=False`` keeps the head and loss replicated (for comparisons).
    """
    from jla.models.lm import LMOutput, _document_ids, _dropout, _tree_add, _zero_stats, remat_policy

    cfg = model.config
    if axis not in mesh.shape:
        raise ValueError(f"mesh axes {tuple(mesh.axis_names)} have no pipeline axis {axis!r}")
    stages = int(mesh.shape[axis])
    period, _, _ = stage_layout(cfg, stages)
    check_pipeline_attention(cfg, mesh)
    if "rwkv7" in cfg.resolved_layer_types:
        raise ValueError(_NO_VALUE_RESIDUAL)
    if not inference:
        check_pipeline_layers(cfg, stages)
    if len(model.blocks) != period:
        raise ValueError(f"model has {len(model.blocks)} scanned groups, expected the {period}-layer period")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, T]; got {input_ids.shape}")
    batch, length = input_ids.shape
    microbatches = resolve_microbatches(stages, num_microbatches)
    if batch % microbatches:
        raise ValueError(f"batch {batch} is not divisible by {microbatches} pipeline microbatches")
    scatter = shard_head and shards_head_over_stages(
        batch, mesh, num_microbatches=microbatches, batch_axes=batch_axes, axis=axis, config=cfg
    )
    shards = _batch_shards(mesh, batch_axes) if scatter else 1
    manual = _manual_batch(cfg, batch, microbatches, mesh, axis, batch_axes, rules)
    rows = None if not manual or not batch_axes else batch_axes[0] if len(batch_axes) == 1 else tuple(batch_axes)
    if positions is None:
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    if positions.shape != (batch, length):
        raise ValueError(f"positions must be [B, T] = {(batch, length)}; got {positions.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    segment_ids = _document_ids(segment_ids)
    uses_dropout = not inference and (cfg.embed_dropout > 0.0 or cfg.resid_dropout > 0.0)
    if uses_dropout and key is None:
        raise ValueError("dropout is enabled: pass key= (or inference=True)")
    if not uses_dropout:
        key = None
    k_embed, k_layers = (None, None) if key is None else jax.random.split(key)

    x = _dropout(model.embed(input_ids), cfg.embed_dropout, k_embed)
    layer_keys = model._layer_keys(k_layers)
    block_fn = model._block_fn()
    params, static = eqx.partition(model.blocks, eqx.is_array)

    inputs: dict[str, Any] = {
        "params": jax.tree.map(lambda a: _split_stages(a, stages), params),
        # One copy of the microbatched inputs per stage (only stage 0 reads them): the
        # transpose of this broadcast sums the per-stage cotangents outside the shard_map.
        "xs": jnp.broadcast_to(_microbatch(x, microbatches)[None], (stages, microbatches, batch // microbatches, *x.shape[1:])),
        "pos": _microbatch(positions.astype(jnp.int32), microbatches),
    }
    # Fully manual: rows of every microbatch split over the batch axes, block parameters
    # all-gathered over their FSDP axes layer by layer (the transpose reduce-scatters the
    # gradients; the shard_map transpose sums them over the axes a parameter is replicated on).
    dims = _block_dims(model, params, mesh, rules) if manual else None
    layer_policy = _layer_policy(remat_policy(cfg.remat, cfg.remat_names, cfg.remat_offload_names))
    param_specs = P(axis) if not manual else jax.tree.map(lambda d: P(axis, None, *d.dims[1:]), dims)
    specs: dict[str, Any] = {"params": param_specs, "xs": P(axis, None, rows), "pos": P(None, rows)}
    if segment_ids is not None:
        inputs["seg"], specs["seg"] = _microbatch(segment_ids, microbatches), P(None, rows)
    if layer_keys is not None:
        inputs["keys"] = _split_stages(layer_keys.reshape(-1, period, *layer_keys.shape[1:]), stages)
        specs["keys"] = P(axis)

    def stage_fn(local: dict[str, Any]) -> tuple[Array, Any]:
        stage = lax.axis_index(axis)
        stage_params = jax.tree.map(lambda a: a[0], local["params"])
        stage_keys = local["keys"][0] if "keys" in local else None
        xs, pos, seg = local["xs"][0], local["pos"], local.get("seg")
        # Dropout: a different mask per microbatch (and per batch shard when those are manual).
        shard_index = lax.axis_index(rows) if rows is not None else None

        def make_layer(j: int) -> Callable[..., tuple[Array, Any]]:
            def layer(block_params, h, micro_pos, micro_seg, block_key):
                if manual:
                    block_params = jax.tree.map(
                        lambda a, d: checkpoint_name(_gather_dims(a, d.dims[1:]), _GATHER_NAME), block_params, dims[j]
                    )
                # The third output is the RWKV-7 value residual, None here (rwkv7 stacks are refused).
                h, stats, _ = block_fn(eqx.combine(block_params, static[j]), h, micro_pos, micro_seg, block_key)
                return h, stats

            # The backward pass gathers the weights again instead of saving them per iteration;
            # the policy keeps model.remat in force (it governs the nested block checkpoint).
            return jax.checkpoint(layer, policy=layer_policy) if manual else layer

        layers = [make_layer(j) for j in range(period)]

        def run_stage(h: Array, micro_pos: Array, micro_seg: Array | None, micro: Array) -> tuple[Array, Any]:
            def body(carry, layer_xs):
                h, acc = carry
                group_params, group_keys = layer_xs
                for j in range(period):
                    block_key = None if group_keys is None else jax.random.fold_in(group_keys[j], micro)
                    if block_key is not None and shard_index is not None:
                        block_key = jax.random.fold_in(block_key, shard_index)
                    h, s = layers[j](group_params[j], h, micro_pos, micro_seg, block_key)
                    acc = _tree_add(acc, s)
                return (h, acc), None

            (h, acc), _ = lax.scan(body, (h, _zero_stats()), (stage_params, stage_keys))
            return h, acc

        def iteration(carry, t):
            state, outputs, stats = carry
            index = t - stage  # the microbatch this stage works on at iteration t
            valid = (index >= 0) & (index < microbatches)
            micro = jnp.clip(index, 0, microbatches - 1)
            h = jnp.where(stage == 0, lax.dynamic_index_in_dim(xs, micro, keepdims=False), state)
            micro_seg = None if seg is None else lax.dynamic_index_in_dim(seg, micro, keepdims=False)
            y, s = run_stage(h, lax.dynamic_index_in_dim(pos, micro, keepdims=False), micro_seg, micro)
            stats = jax.tree.map(lambda acc, v: acc + jnp.where(valid, v, jnp.zeros_like(v)), stats, s)
            keep = valid & (stage == stages - 1)
            slot = lax.dynamic_index_in_dim(outputs, micro, keepdims=False)
            outputs = lax.dynamic_update_index_in_dim(outputs, jnp.where(keep, y, slot), micro, 0)
            # Stage s + 1 receives y at iteration t + 1 (stage 0 reads the inputs instead).
            state = lax.ppermute(y, axis, perm=[(k, k + 1) for k in range(stages - 1)]) if stages > 1 else y
            return (state, outputs, stats), None

        init = (jnp.zeros_like(xs[0]), jnp.zeros_like(xs), _zero_stats())
        (_, outputs, stats), _ = lax.scan(iteration, init, jnp.arange(microbatches + stages - 1))
        stats = jax.tree.map(lambda v: v[None], stats)
        if not scatter:
            return outputs[None], stats
        # Microbatch row j = d * (L / M) + i of batch shard d (L rows per shard). Only the
        # last stage holds outputs (the others hold zeros), so the sum is exact; stage s
        # keeps the rows i in [s * L / (M S), (s + 1) * L / (M S)) of every shard and
        # microbatch, which reassemble into contiguous [B] blocks (pp minor in the batch).
        # (When the batch axes are manual, this device's rows are already shard d's.)
        local_shards = 1 if manual else shards
        blocks = outputs.reshape(microbatches, local_shards, outputs.shape[1] // local_shards, *outputs.shape[2:])
        return lax.psum_scatter(blocks, axis, scatter_dimension=2, tiled=True), stats

    # Manual over the stage axis (inputs and outputs explicitly per stage, or integer), and
    # over every other axis too when _manual_batch allows it: custom-partitioned ops such
    # as cuDNN attention cannot run inside a partially manual region on GPU.
    run = jax.shard_map(
        stage_fn,
        mesh=mesh,
        in_specs=(specs,),
        out_specs=(P(None, rows, axis) if scatter else P(axis, None, rows), P(axis)),
        axis_names=frozenset(mesh.axis_names) if manual else frozenset({axis}),
        check_vma=False,
    )
    outputs, stage_stats = run(inputs)
    if scatter:
        # [M, shards, L / M, T, D] -> [shards, L / M, M, T, D] -> [B, T, D] (row d * L + i * M + m).
        x = jnp.moveaxis(outputs, 0, 2).reshape(batch, *outputs.shape[3:])
        x = lax.with_sharding_constraint(x, NamedSharding(mesh, P((*batch_axes, axis))))
    else:
        x = _unmicrobatch(outputs[stages - 1])
    stats = jax.tree.map(lambda v: jnp.sum(v) / microbatches, stage_stats)  # mean over microbatches

    num_moe = sum(t == "moe" for t in cfg.resolved_mlp_types)
    metrics: dict[str, Array] = {}
    aux = jnp.zeros((), jnp.float32)
    if num_moe:
        stats = jax.tree.map(lambda v: v / num_moe, stats)  # mean over MoE layers
        aux = stats.aux_loss
        metrics = {
            "moe/balance_loss": stats.balance_loss,
            "moe/router_z_loss": stats.z_loss,
            "moe/dropped_fraction": stats.dropped_fraction,
        }
    metrics[BUBBLE_METRIC] = jnp.asarray(bubble_fraction(stages, microbatches), jnp.float32)
    return LMOutput(hidden=model.final_norm(x), pre_norm_hidden=x, aux_loss=aux, metrics=metrics)


def make_pipeline_forward(
    mesh: Mesh,
    *,
    num_microbatches: int = 0,
    axis: str = PIPELINE_AXIS,
    batch_axes: Sequence[str] | None = None,
    rules: Mapping[str, Sequence[str]] | None = None,
    shard_head: bool = True,
) -> Callable[[Any, Mapping[str, Array], Array | None, bool], Any]:
    """``forward(model, batch, key, inference)`` for :func:`jla.training.step.lm_loss` (``forward=``).

    ``batch_axes`` shards the head and loss over the stages and, with the logical-axis
    ``rules`` the parameters are sharded with, lets the stages run fully manual (see
    :func:`pipeline_lm_forward`).
    """
    resolve_microbatches(int(mesh.shape.get(axis, 1)), num_microbatches)
    if batch_axes is not None:
        batch_axes = tuple(batch_axes)
        _batch_shards(mesh, batch_axes)

    def forward(model: Any, batch: Mapping[str, Array], key: Array | None, inference: bool = False) -> Any:
        return pipeline_lm_forward(
            model,
            batch["input_ids"],
            mesh=mesh,
            num_microbatches=num_microbatches,
            positions=batch.get("positions"),
            segment_ids=batch.get("segment_ids"),
            key=key,
            inference=inference,
            axis=axis,
            batch_axes=batch_axes,
            rules=rules,
            shard_head=shard_head,
        )

    return forward
