# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One gradient reduction per optimizer step on data-parallel and FSDP meshes (``train.grad_reduce = "step"``).

The plain train step (:func:`jla.training.step.build_train_step`, ``grad_reduce = "micro_step"``) differentiates
every micro-step under the automatic partitioner. The weights are FSDP-sharded, so in every micro-step XLA
all-gathers them (or their FP8 copies, which each device then transposes at every use) and reduce-scatters the
micro-step's float32 weight gradient into the sharded accumulator, and every FP8 matmul adds small all-reduces of
its amax maxima, inside the layers. On 8 H100s the 1.3B transformer (4 micro-steps) sent 18.8 GB of float32
gradient reduce-scatters per GPU and step, and GDN's layer scan waited on 72 tiny all-reduces per step
(``benchmarks/multi_gpu.md``).

With ``grad_reduce = "step"`` the micro-step loop runs inside one ``jax.shard_map`` that is manual over every mesh
axis (every axis of size > 1 must split the batch). Each device

* receives the whole parameters once per step (one all-gather of each FSDP shard at the start of the step);
* makes the FP8 weight copies and their transposes once, as one device does
  (:func:`jla.training.fp8.prepare_fp8_weights` with ``transposed=True``);
* runs the micro-steps on its own rows exactly as one device runs them: nothing inside is left to the
  partitioner, so the kernels take the per-device shapes directly and no layer waits on a collective;
* adds its micro-steps' float32 gradients in a full-size accumulator.

Then, once per step, every gradient is reduce-scattered to its FSDP shard (and summed over the axes its parameter
is replicated on), the FP8 delayed-scaling states' new amax maxima are maximized over the devices (one all-reduce
per dtype for all of them) and the loss sums are added up (one all-reduce).

Exactness. The objective is a sum over tokens, so the devices' partial gradients add up to the gradient of the
plain step; only the float32 summation order changes (micro-steps first, then devices). The FP8 states record the
step's largest amax, the same maximum whichever way the devices and micro-steps are grouped. Configurations with a
term that is not a sum over the batch are refused (:func:`check_grad_reduce`): MoE layers (routing statistics over
the global batch), MTP heads (a mean loss times the token count), dropout (masks drawn for the device's rows, not
the global batch's), FP8 current scaling (each scale is the amax of the whole tensor) and the DSA indexer loss; so
are tensor and pipeline parallelism (axes that do not split the batch), Mamba-1 layers (XLA cannot compile their
gradient fully manual) and cuDNN attention (it does not compile inside a fully manual ``shard_map`` on GPU).

Memory. Each device holds the whole float32 parameters and the float32 gradient accumulator during the step
(8 bytes per parameter: 10.4 GB at 1.3B), plus the FP8 copies and their transposes (2 bytes per FP8 weight):
about 10 bytes per parameter on every device whatever the FSDP size (about 30 GB at 3B; 7B does not fit on an
80 GB GPU), so the mode is for models up to about 3B. :func:`accumulate_per_device` logs a warning when that
estimate exceeds half of the device's memory.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from jla.distributed.kernel_sharding import per_device_step
from jla.training.fp8 import accumulate_gradients, attach_fp8_weights, prepare_fp8_weights

__all__ = ["GRAD_REDUCE_MODES", "accumulate_per_device", "check_grad_reduce", "reduce_axes"]

_log = logging.getLogger(__name__)

#: ``train.grad_reduce`` values: reduce the gradients in every micro-step (the plain step) or once per step.
GRAD_REDUCE_MODES = ("micro_step", "step")


def reduce_axes(batch_sharding: Any) -> tuple[str, ...]:
    """The mesh axes of size > 1 of ``batch_sharding`` (``[grad_accum, B, T]`` token arrays), checked to split ``B``.

    Empty on one device (the plain step is then the same program). Raises when an axis of size > 1 does not split
    the batch (tensor or pipeline parallelism): its devices would need the partitioner inside the micro-steps.
    """
    mesh = getattr(batch_sharding, "mesh", None)
    spec = getattr(batch_sharding, "spec", None)
    if mesh is None or spec is None:
        raise ValueError("grad_reduce='step' needs the batch's NamedSharding (batch_sharding=)")
    large = tuple(name for name, size in mesh.shape.items() if int(size) > 1)
    if not large:
        return ()
    entry = tuple(spec)[1] if len(tuple(spec)) > 1 else None
    batch = () if entry is None else (entry,) if isinstance(entry, str) else tuple(entry)
    other = [name for name in large if name not in batch]
    if other:
        raise ValueError(
            f"grad_reduce='step' runs the micro-steps manual over every mesh axis, so every axis of size > 1 must "
            f"split the batch; {other} do not (batch axes {list(batch)}): use grad_reduce='micro_step' with tensor "
            "or pipeline parallelism"
        )
    return large


def check_grad_reduce(config: Any, *, platform: str | None = None) -> None:
    """Raise ``ValueError`` when the model ``config`` (a :class:`jla.models.config.ModelConfig`) has a term that
    ``grad_reduce='step'`` would not compute exactly, or a layer that does not compile fully manual (see the
    module docstring). ``platform`` (default: JAX's default backend) decides the cuDNN attention check."""
    problems: list[str] = []
    mlp_types = set(getattr(config, "resolved_mlp_types", ()) or ())
    layer_types = set(getattr(config, "resolved_layer_types", ()) or ())
    if getattr(config, "moe", None) is not None or "moe" in mlp_types:
        problems.append("MoE layers (their routing statistics are taken over the global batch)")
    if getattr(config, "mtp", None) is not None:
        problems.append("MTP heads (their loss is a mean, weighted by the step's token count)")
    if float(getattr(config, "embed_dropout", 0.0)) > 0 or float(getattr(config, "resid_dropout", 0.0)) > 0:
        problems.append("dropout (each device would draw the masks of its own rows, not the global batch's)")
    if getattr(config, "fp8", "off") == "current":
        problems.append("fp8 = 'current' (each scale is the amax of the whole tensor; 'delayed' is exact)")
    sparse = getattr(config, "sparse_attention", None)
    if "dsa" in layer_types and float(getattr(sparse, "indexer_loss_coef", 0.0)) > 0:
        problems.append("the DSA indexer loss (sparse_attention.indexer_loss_coef > 0)")
    if "mamba1" in layer_types:
        problems.append("Mamba-1 layers (XLA cannot compile their gradient fully manual)")
    if (platform or jax.default_backend()) == "gpu":
        for section, kinds in (("attention", {"attention"}), ("local_attention", {"local_attention"}),
                               ("mla", {"mla"})):
            attn = getattr(config, section, None)
            if layer_types & kinds and attn is not None and getattr(attn, "implementation", None) in ("auto", "cudnn"):
                problems.append(f"model.{section}.implementation = {attn.implementation!r} (cuDNN attention does not "
                                "compile inside a fully manual shard_map on GPU; use 'mosaic_gpu' or 'xla')")
    if problems:
        raise ValueError("train.grad_reduce = 'step' cannot run this model exactly: " + "; ".join(problems)
                         + ". Use train.grad_reduce = 'micro_step'.")


def _names(entry: Any) -> tuple[str, ...]:
    return () if entry is None else (entry,) if isinstance(entry, str) else tuple(entry)


def _reduce_gradient(grad: jax.Array, spec: P, axes: Sequence[str]) -> jax.Array:
    """This device's partial ``grad`` summed over ``axes``: scattered along the dims ``spec`` splits, summed over
    the axes it does not name (the parameter is replicated on them)."""
    scattered = {name for entry in tuple(spec) for name in _names(entry)}
    summed = tuple(name for name in axes if name not in scattered)
    if summed:
        grad = lax.psum(grad, summed)
    for dim, entry in enumerate(tuple(spec)):
        names = tuple(name for name in _names(entry) if name in axes)
        if names:
            grad = lax.psum_scatter(grad, names if len(names) > 1 else names[0], scatter_dimension=dim, tiled=True)
    return grad


def _bucketed(values: list[jax.Array], reduce: Callable[[jax.Array], jax.Array]) -> list[jax.Array]:
    """``reduce`` applied to every array of ``values`` with one collective per dtype (flattened and concatenated)."""
    out: list[jax.Array | None] = [None] * len(values)
    by_dtype: dict[Any, list[int]] = {}
    for index, value in enumerate(values):
        by_dtype.setdefault(jnp.dtype(value.dtype), []).append(index)
    for indices in by_dtype.values():
        flat = reduce(jnp.concatenate([jnp.ravel(values[i]) for i in indices]))
        offset = 0
        for i in indices:
            size = values[i].size
            out[i] = flat[offset:offset + size].reshape(values[i].shape)
            offset += size
    return out  # type: ignore[return-value]


def _warn_memory(leaves: Sequence[Any]) -> None:
    """Log a warning when the step's whole-weight copies (about 10 bytes per parameter on every device, see the
    module docstring) exceed half of the first local device's memory limit (when the device reports one)."""
    need = 10 * sum(math.prod(getattr(leaf, "shape", ())) for leaf in leaves)
    try:
        limit = (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit")
    except Exception:  # noqa: BLE001 - backends without memory statistics
        limit = None
    if limit and need > limit / 2:
        _log.warning("train.grad_reduce = 'step' keeps about 10 bytes per parameter on every device whatever the "
                     "FSDP size: %.1f GB of this device's %.1f GB before activations and optimizer state; use "
                     "'micro_step' if the step does not fit", need / 1e9, limit / 1e9)


def accumulate_per_device(
    loss_fn: Callable[[Any, Mapping[str, jax.Array], jax.Array], tuple[jax.Array, dict[str, jax.Array]]],
    params: Any,
    static: Any,
    batch: Mapping[str, jax.Array],
    key: jax.Array,
    *,
    axes: Sequence[str],
    batch_sharding: Any,
    param_shardings: Any,
    fp8_mask: Any | None,
    aux_names: Sequence[str],
    unroll: int,
) -> tuple[Any, dict[str, jax.Array], jax.Array]:
    """The micro-step loop of :func:`jla.training.step.build_train_step` with one gradient reduction per step.

    Returns ``(grads, totals, key)`` as the plain loop does: ``grads`` (float32, like ``params``) summed over
    the micro-steps and the devices and sharded like the parameters (the FP8 state leaves hold the maxima),
    ``totals`` (``aux_names``) summed over the micro-steps and, for ``tokens`` and the ``*_sum`` keys, over the
    devices (other keys averaged over them), and the key after one split per micro-step.
    """
    if param_shardings is None:
        raise ValueError("grad_reduce='step' needs the parameter shardings (param_shardings=)")
    mesh = batch_sharding.mesh
    axes = tuple(axes)
    other, rest = eqx.partition(static, eqx.is_array)
    shardings = jax.tree.leaves(param_shardings, is_leaf=lambda s: isinstance(s, jax.sharding.Sharding))
    leaves, treedef = jax.tree.flatten(params)
    if len(shardings) != len(leaves):
        raise ValueError(f"param_shardings has {len(shardings)} shardings for {len(leaves)} trainable arrays")
    _warn_memory(leaves)
    specs = [getattr(s, "spec", P()) for s in shardings]
    is_state = [False] * len(leaves) if fp8_mask is None else [bool(m) for m in jax.tree.leaves(fp8_mask)]
    if len(is_state) != len(leaves):
        raise ValueError("the FP8 state mask does not match the trainable arrays")
    grad_accum = int(batch["input_ids"].shape[0])
    sums = tuple(name for name in aux_names if name == "tokens" or name.endswith("_sum"))
    means = tuple(name for name in aux_names if name not in sums)
    batch_spec = batch_sharding.spec

    def per_device(p: Any, arrays: Any, local: Mapping[str, jax.Array], k: jax.Array):
        model = eqx.combine(p, arrays, rest)
        # The FP8 copies once per step, with their transposes: every device holds whole weights here, as one device.
        fp8_weights = prepare_fp8_weights(model, transposed=True) if fp8_mask is not None else None

        def micro_loss(q: Any, micro: Mapping[str, jax.Array], micro_key: jax.Array):
            return loss_fn(attach_fp8_weights(eqx.combine(q, arrays, rest), fp8_weights), micro, micro_key)

        def body(carry, micro):
            grads_acc, aux_acc, carry_key = carry
            carry_key, micro_key = jax.random.split(carry_key)
            (_, aux), grads = jax.value_and_grad(micro_loss, has_aux=True)(p, micro, micro_key)
            grads_acc = accumulate_gradients(grads_acc, grads, fp8_mask)
            aux_acc = {name: aux_acc[name] + jnp.asarray(value, jnp.float32) for name, value in aux.items()}
            return (grads_acc, aux_acc, carry_key), None

        grads0 = jax.tree.map(lambda a: jnp.zeros(a.shape, jnp.float32), p)
        aux0 = {name: jnp.zeros((), jnp.float32) for name in aux_names}
        (grads, totals, k), _ = lax.scan(body, (grads0, aux0, k), local, length=grad_accum, unroll=unroll)

        flat = jax.tree.leaves(grads)
        states = [i for i, s in enumerate(is_state) if s]
        for i, value in zip(states, _bucketed([flat[i] for i in states], lambda x: lax.pmax(x, axes))):
            flat[i] = value
        for i, (grad, spec) in enumerate(zip(flat, specs)):
            if not is_state[i]:
                flat[i] = _reduce_gradient(grad, spec, axes)
        reduced = {}
        if sums:
            for name, value in zip(sums, _bucketed([totals[name] for name in sums], lambda x: lax.psum(x, axes))):
                reduced[name] = value
        if means:
            for name, value in zip(means, _bucketed([totals[name] for name in means], lambda x: lax.pmean(x, axes))):
                reduced[name] = value
        return jax.tree.unflatten(treedef, flat), {name: reduced[name] for name in aux_names}, k

    grad_specs = jax.tree.unflatten(treedef, [P() if s else spec for s, spec in zip(is_state, specs)])
    run = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(P(), P(), {name: batch_spec for name in batch}, P()),
        out_specs=(grad_specs, P(), P()),
        axis_names=frozenset(mesh.axis_names),
        check_vma=False,
    )
    with per_device_step():
        return run(params, other, dict(batch), key)
