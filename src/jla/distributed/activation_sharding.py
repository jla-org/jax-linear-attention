# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch sharding constraints on activations under the context mesh.

FSDP shards every weight along its ``embed`` dim over the ``fsdp`` axis, the same axis that splits the
batch. Left alone, GSPMD's sharding propagation carries the weights' sharding into the activations:
the embedding table's ``embed`` split shards the looked-up token vectors along ``D``, the residual
stream ``[B, T, D]`` stays split along ``D`` with the whole global batch on every device, and every
projection becomes a partial matmul over ``D / fsdp`` followed by an all-reduce of a full
``[B_global, T, out]`` activation. At 1.3B on 8 GPUs that was a 22 GiB float32 all-reduce per MLP
up-projection and an 8 GiB one per 4096-row vocabulary tile of the LM head
(``benchmarks/multi_gpu.md``).

:func:`constrain_batch` pins an activation's leading (batch) dim to the mesh axes of the ``batch``
rule (the trainer's resolved rules, set with :func:`jla.distributed.kernel_sharding.use_rules`) and
leaves every other dim to the partitioner. Weights are then gathered (FSDP's all-gather) and each
device computes on its own rows. It is a no-op without a context mesh (``jax.set_mesh``), when no
batch axis of size > 1 divides the batch, and for axes that are manual (inside ``shard_map``).

Under pipeline parallelism it is also a no-op outside the stages, where the stage axis is not manual
(the LM head, its loss and the MTP head). There ``pipeline_lm_forward`` leaves the hidden states split
over the ``batch`` axes and the stage axis, so each stage computes the head for its own rows. A
constraint to the ``batch`` axes alone would make every stage compute the head for all rows of its
data shard (pp times the head's FLOPs and logits memory). Inside the stages the stage axis is manual
and the constraint applies as without pipelining.
"""

from __future__ import annotations

import jax
from jax.sharding import PartitionSpec as P

from jla.distributed.kernel_sharding import kernel_shard_axes

__all__ = ["constrain_batch"]

# jla.distributed.pipeline.PIPELINE_AXIS (not imported: the pipeline module imports the models).
_PIPELINE_AXIS = "pp"


def _between_pipeline_stages(mesh) -> bool:
    """The stage axis has size > 1 and is not manual: code outside the pipeline's stages."""
    if _PIPELINE_AXIS in set(getattr(mesh, "manual_axes", ())):
        return False
    return int(dict(zip(mesh.axis_names, mesh.axis_sizes)).get(_PIPELINE_AXIS, 1)) > 1


def constrain_batch(x: jax.Array) -> jax.Array:
    """``x [B, ...]`` with ``B`` sharded over the ``batch`` rule's mesh axes (other dims unconstrained)."""
    if not hasattr(x, "ndim") or x.ndim < 1:
        return x
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or mesh.empty or _between_pipeline_stages(mesh):
        return x
    batch_axes, _ = kernel_shard_axes(mesh, int(x.shape[0]), None)
    if not batch_axes:
        return x
    entry = batch_axes[0] if len(batch_axes) == 1 else batch_axes
    return jax.lax.with_sharding_constraint(x, P(entry, *([P.UNCONSTRAINED] * (x.ndim - 1))))
