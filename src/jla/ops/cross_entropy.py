# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head cross entropy that never materializes ``[N, V]`` logits.

``linear_cross_entropy(hidden, weight, targets)`` computes the summed token
cross entropy of ``logits = hidden @ weight.T`` exactly, streaming the
vocabulary in tiles of ``vocab_chunk`` rows with an online logsumexp. The
analytic custom VJP stores only the per-token logsumexp and rebuilds one logit
tile at a time in the backward pass, so peak memory is ``O(N * vocab_chunk)``
instead of ``O(N * V)``. Matmuls take ``compute_dtype`` operands with float32
accumulation; the softmax statistics are float32 throughout.

Per valid token (``target != ignore_index``) with ``lse = logsumexp(logits)``:

    ce      = (1 - ls) * (lse - logit[target]) + ls * (lse - mean(logits))
    z_loss  = lse ** 2

where ``ls`` is the label-smoothing weight. Both are summed over valid tokens;
the caller divides by ``num_valid`` (or a global token count) and weights the
z-loss.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["IMPLS", "CrossEntropyObjective", "CrossEntropyOutput", "cross_entropy_from_logits",
           "linear_cross_entropy", "linear_cross_entropy_objective"]

Array = jax.Array


class CrossEntropyOutput(NamedTuple):
    loss_sum: Array  # f32 scalar: sum over valid tokens of (CE + label smoothing)
    z_loss_sum: Array  # f32 scalar: sum over valid tokens of logsumexp**2 (unweighted)
    num_valid: Array  # f32 scalar: number of targets != ignore_index


def _tiles(weight: Array, tile: int, dtype: jnp.dtype) -> Array:
    """``[V, D]`` -> ``[V_pad / tile, tile, D]`` in ``dtype`` (zero rows pad the last tile)."""
    vocab, dim = weight.shape
    pad = (-vocab) % tile
    w = weight.astype(dtype)
    if pad:
        w = jnp.pad(w, ((0, pad), (0, 0)))
    return w.reshape(-1, tile, dim)


def _tile_logits(hidden: Array, w_tile: Array) -> Array:
    # Under a mesh, keep the tokens split over the data axes: FSDP shards the head along D over the same
    # axis, and GSPMD would otherwise compute every tile for all tokens from D-slices and all-reduce it.
    from jla.distributed.activation_sharding import constrain_batch

    logits = lax.dot_general(constrain_batch(hidden), w_tile, (((1,), (1,)), ((), ())),
                             preferred_element_type=jnp.float32)
    return constrain_batch(logits)


def _forward(hidden, weight, targets, label_smoothing, tile, compute_dtype):
    vocab = weight.shape[0]
    tiles = _tiles(weight, tile, compute_dtype)
    h = hidden.astype(compute_dtype)
    n = h.shape[0]
    columns = jnp.arange(tile, dtype=jnp.int32)

    def step(carry, xs):
        run_max, run_sum, target_logit, logit_sum = carry
        index, w_tile = xs
        offset = index * tile
        logits = _tile_logits(h, w_tile)  # [N, tile] f32
        in_vocab = (offset + columns) < vocab
        masked = jnp.where(in_vocab, logits, -jnp.inf)
        new_max = jnp.maximum(run_max, jnp.max(masked, axis=-1))
        run_sum = run_sum * jnp.exp(run_max - new_max) + jnp.sum(jnp.exp(masked - new_max[:, None]), axis=-1)
        local = targets - offset
        hit = (local >= 0) & (local < tile)
        picked = jnp.take_along_axis(logits, jnp.clip(local, 0, tile - 1)[:, None], axis=-1)[:, 0]
        target_logit = jnp.where(hit, picked, target_logit)
        if label_smoothing:
            logit_sum = logit_sum + jnp.sum(jnp.where(in_vocab, logits, 0.0), axis=-1)
        return (new_max, run_sum, target_logit, logit_sum), None

    init = (
        jnp.full((n,), -jnp.inf, jnp.float32),
        jnp.zeros((n,), jnp.float32),
        jnp.zeros((n,), jnp.float32),
        jnp.zeros((n,), jnp.float32),
    )
    (run_max, run_sum, target_logit, logit_sum), _ = lax.scan(
        step, init, (jnp.arange(tiles.shape[0], dtype=jnp.int32), tiles)
    )
    lse = run_max + jnp.log(run_sum)
    per_token = lse - target_logit
    if label_smoothing:
        per_token = (1.0 - label_smoothing) * per_token + label_smoothing * (lse - logit_sum / vocab)
    return per_token, lse


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _ce_sums(hidden, weight, targets, valid, label_smoothing, tile, compute_dtype):
    per_token, lse = _forward(hidden, weight, targets, label_smoothing, tile, compute_dtype)
    return jnp.sum(jnp.where(valid, per_token, 0.0)), jnp.sum(jnp.where(valid, lse * lse, 0.0))


def _ce_sums_fwd(hidden, weight, targets, valid, label_smoothing, tile, compute_dtype):
    per_token, lse = _forward(hidden, weight, targets, label_smoothing, tile, compute_dtype)
    sums = (jnp.sum(jnp.where(valid, per_token, 0.0)), jnp.sum(jnp.where(valid, lse * lse, 0.0)))
    return sums, (hidden, weight, targets, valid, lse)


def _ce_sums_bwd(label_smoothing, tile, compute_dtype, residuals, cotangents):
    hidden, weight, targets, valid, lse = residuals
    g_ce, g_z = (jnp.asarray(c, jnp.float32) for c in cotangents)
    vocab, dim = weight.shape
    tiles = _tiles(weight, tile, compute_dtype)
    h = hidden.astype(compute_dtype)
    columns = jnp.arange(tile, dtype=jnp.int32)
    row_scale = valid.astype(jnp.float32)
    # d(ce)/dlogit = p - (1 - ls) onehot - ls / V;  d(lse^2)/dlogit = 2 lse p
    p_coef = (g_ce + 2.0 * g_z * lse) * row_scale  # [N]
    hot_coef = g_ce * (1.0 - label_smoothing) * row_scale
    flat_coef = g_ce * (label_smoothing / vocab) * row_scale

    def step(grad_hidden, xs):
        index, w_tile = xs
        offset = index * tile
        logits = _tile_logits(h, w_tile)
        in_vocab = (offset + columns) < vocab
        probs = jnp.where(in_vocab, jnp.exp(logits - lse[:, None]), 0.0)
        onehot = (targets - offset)[:, None] == columns[None, :]
        grad_logits = p_coef[:, None] * probs - hot_coef[:, None] * onehot
        if label_smoothing:
            grad_logits = grad_logits - flat_coef[:, None] * in_vocab
        grad_logits = grad_logits.astype(compute_dtype)
        grad_hidden = grad_hidden + lax.dot_general(
            grad_logits, w_tile, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        grad_w_tile = lax.dot_general(
            grad_logits, h, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        return grad_hidden, grad_w_tile.astype(weight.dtype)

    grad_hidden, grad_tiles = lax.scan(
        step, jnp.zeros((h.shape[0], dim), jnp.float32), (jnp.arange(tiles.shape[0], dtype=jnp.int32), tiles)
    )
    grad_weight = grad_tiles.reshape(-1, dim)[:vocab]
    return grad_hidden.astype(hidden.dtype), grad_weight, None, None


_ce_sums.defvjp(_ce_sums_fwd, _ce_sums_bwd)


def linear_cross_entropy(
    hidden: Array,
    weight: Array,
    targets: Array,
    *,
    ignore_index: int = -100,
    vocab_chunk: int = 4096,
    label_smoothing: float = 0.0,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    impl: str = "xla",
) -> CrossEntropyOutput:
    """Summed cross entropy of ``hidden [N, D] @ weight[V, D].T`` against ``targets [N]``.

    Never materializes ``[N, V]`` logits: the forward streams vocabulary tiles of
    ``vocab_chunk`` rows and the backward (an analytic custom VJP) stores only the
    per-token logsumexp. Targets must lie in ``[0, V)`` or equal ``ignore_index``.

    ``impl``: ``"xla"`` (this module), ``"mosaic_gpu"`` (the fused Hopper kernels of
    :mod:`jla.ops.mosaic_gpu.linear_cross_entropy`; raises when they cannot run the
    call) or ``"auto"`` (the kernels when they can, else ``"xla"``). The kernels run
    on one device only: with several visible devices they need the body of a
    ``shard_map`` whose mesh axes are all manual, otherwise ``"auto"`` uses XLA.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"expected hidden [N, D] and weight [V, D]; got {hidden.shape} and {weight.shape}")
    if targets.shape != hidden.shape[:1]:
        raise ValueError(f"targets must be [N] = {hidden.shape[:1]}; got {targets.shape}")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError(f"targets must be integers; got {targets.dtype}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1); got {label_smoothing}")
    if vocab_chunk < 1:
        raise ValueError(f"vocab_chunk must be >= 1; got {vocab_chunk}")
    if impl not in IMPLS:
        raise ValueError(f"unknown cross entropy impl {impl!r}; expected one of {IMPLS}")
    valid = targets != ignore_index
    safe_targets = jnp.where(valid, targets, 0).astype(jnp.int32)
    # Zero ignored rows first so a non-finite ignored activation cannot reach the gradients.
    hidden = jnp.where(valid[:, None], hidden, jnp.zeros((), hidden.dtype))
    if impl != "xla":
        fused = _mosaic_gpu_sums(hidden, weight, safe_targets, valid, impl, vocab_chunk, label_smoothing,
                                 compute_dtype)
        if fused is not None:
            return CrossEntropyOutput(*fused, jnp.sum(valid).astype(jnp.float32))
    tile = min(int(vocab_chunk), weight.shape[0])
    loss_sum, z_sum = _ce_sums(hidden, weight, safe_targets, valid, float(label_smoothing), tile,
                               jnp.dtype(compute_dtype))
    return CrossEntropyOutput(loss_sum, z_sum, jnp.sum(valid).astype(jnp.float32))


def cross_entropy_from_logits(
    logits: Array,
    targets: Array,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> CrossEntropyOutput:
    """Same sums from explicit ``logits [N, V]`` (plain autodiff; for evaluation and tests)."""
    logits = logits.astype(jnp.float32)
    valid = targets != ignore_index
    safe = jnp.where(valid, targets, 0)
    lse = jax.nn.logsumexp(logits, axis=-1)
    per_token = lse - jnp.take_along_axis(logits, safe[:, None], axis=-1)[:, 0]
    if label_smoothing:
        per_token = (1.0 - label_smoothing) * per_token + label_smoothing * (lse - jnp.mean(logits, axis=-1))
    return CrossEntropyOutput(
        jnp.sum(jnp.where(valid, per_token, 0.0)),
        jnp.sum(jnp.where(valid, lse * lse, 0.0)),
        jnp.sum(valid).astype(jnp.float32),
    )


IMPLS = ("xla", "mosaic_gpu", "auto")


def _mosaic_gpu_sums(hidden, weight, targets, valid, impl, vocab_chunk, label_smoothing, compute_dtype):
    """``(loss_sum, z_loss_sum)`` from the Mosaic GPU kernels, or ``None`` to use XLA (``impl="auto"``)."""
    try:
        from jla.ops.mosaic_gpu import linear_cross_entropy as fused
    except ImportError as err:  # a JAX build without the Mosaic GPU toolchain
        if impl == "mosaic_gpu":
            raise ValueError(f"impl='mosaic_gpu' needs Pallas Mosaic GPU: {err}") from err
        return None
    reason = fused.unsupported_reason(hidden, weight, compute_dtype)
    if reason is not None:
        if impl == "mosaic_gpu":
            raise ValueError(f"impl='mosaic_gpu' cannot run this call: {reason}")
        return None
    return fused.ce_sums(hidden, weight, targets, valid, label_smoothing=float(label_smoothing),
                         vocab_chunk=int(vocab_chunk), compute_dtype=compute_dtype)


class CrossEntropyObjective(NamedTuple):
    objective_sum: Array  # f32 scalar: loss_sum + z_loss_weight * z_loss_sum; the only output with a gradient
    loss_sum: Array  # f32 scalar: as CrossEntropyOutput.loss_sum (no gradient)
    z_loss_sum: Array  # f32 scalar: as CrossEntropyOutput.z_loss_sum (no gradient)
    num_valid: Array  # f32 scalar: number of targets != ignore_index


def linear_cross_entropy_objective(
    hidden: Array,
    weight: Array,
    targets: Array,
    *,
    z_loss_weight: float = 0.0,
    ignore_index: int = -100,
    vocab_chunk: int = 4096,
    row_chunk: int = 8192,
    label_smoothing: float = 0.0,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    impl: str = "auto",
    fp8: bool = False,
) -> CrossEntropyObjective:
    """``loss_sum + z_loss_weight * z_loss_sum`` of :func:`linear_cross_entropy` as one objective.

    Knowing the z-loss weight in the forward pass lets the Mosaic GPU kernels form the
    gradient there (:func:`jla.ops.mosaic_gpu.linear_cross_entropy.objective_sums`): the
    logits GEMM keeps its float32 tiles on chip and stores ``exp(logit - tile max)`` in
    float16 per block of ``row_chunk`` rows, one elementwise pass forms the logit gradient,
    and two GEMMs give ``dhidden`` and ``dweight``. That is three GEMMs instead of four (no
    logits recompute in the backward pass), for ``4 * row_chunk * V`` bytes of transient
    buffers. ``loss_sum``, ``z_loss_sum`` and ``num_valid`` are for logging and carry no
    gradient; differentiate ``objective_sum``. ``fp8`` (kernels only) runs the two gradient
    GEMMs in FP8 (e5m2 logit gradient, e4m3 ``hidden`` and ``weight``, per-tensor current
    scaling); the logits GEMM and the loss stay in ``compute_dtype``. The XLA path ignores it.
    ``fp8`` is lossy: the e5m2 logit gradient cost 0.02 in loss after 600 training steps of a
    1.3B transformer (``benchmarks/cross_entropy.md``).

    ``impl``: ``"auto"`` (the kernels when they can run the call, otherwise
    :func:`linear_cross_entropy` with ``impl="xla"`` and plain autodiff), ``"mosaic_gpu"``
    (raises when the kernels cannot run the call) or ``"xla"``. ``vocab_chunk`` applies to
    the XLA path.

    Several devices: under a context mesh (``jax.set_mesh``, as the trainer sets it) with
    automatic axes of size > 1 that divide ``N``, the call runs per shard in a ``shard_map``:
    the rows split over all those axes, the weight replicated (its gradient summed over the
    mesh by the transpose), one partial sum per device added up outside. Without such a mesh
    "auto" uses XLA and "mosaic_gpu" raises (the kernels have no partitioning rule).
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"expected hidden [N, D] and weight [V, D]; got {hidden.shape} and {weight.shape}")
    if targets.shape != hidden.shape[:1]:
        raise ValueError(f"targets must be [N] = {hidden.shape[:1]}; got {targets.shape}")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError(f"targets must be integers; got {targets.dtype}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1); got {label_smoothing}")
    if z_loss_weight < 0:
        raise ValueError(f"z_loss_weight must be >= 0; got {z_loss_weight}")
    if row_chunk < 1 or vocab_chunk < 1:
        raise ValueError(f"row_chunk and vocab_chunk must be >= 1; got {row_chunk} and {vocab_chunk}")
    if impl not in IMPLS:
        raise ValueError(f"unknown cross entropy impl {impl!r}; expected one of {IMPLS}")
    if impl != "xla":
        sharded = _per_shard_objective(hidden, weight, targets, z_loss_weight=z_loss_weight,
                                       ignore_index=ignore_index, vocab_chunk=vocab_chunk, row_chunk=row_chunk,
                                       label_smoothing=label_smoothing, compute_dtype=compute_dtype, impl=impl,
                                       fp8=fp8)
        if sharded is not None:
            return sharded
        valid = targets != ignore_index
        safe_targets = jnp.where(valid, targets, 0).astype(jnp.int32)
        zeroed = jnp.where(valid[:, None], hidden, jnp.zeros((), hidden.dtype))
        fused = _mosaic_gpu_objective(zeroed, weight, safe_targets, valid, impl, row_chunk, label_smoothing,
                                      z_loss_weight, compute_dtype, fp8)
        if fused is not None:
            return CrossEntropyObjective(*fused, jnp.sum(valid).astype(jnp.float32))
    out = linear_cross_entropy(hidden, weight, targets, ignore_index=ignore_index, vocab_chunk=vocab_chunk,
                               label_smoothing=label_smoothing, compute_dtype=compute_dtype, impl="xla")
    return CrossEntropyObjective(out.loss_sum + z_loss_weight * out.z_loss_sum, lax.stop_gradient(out.loss_sum),
                                 lax.stop_gradient(out.z_loss_sum), out.num_valid)


def _per_shard_objective(hidden, weight, targets, *, compute_dtype, **kw) -> CrossEntropyObjective | None:
    """The objective per shard of the context mesh's automatic axes of size > 1, or ``None``.

    ``None`` also when the kernels would not take the operands on one device (dtype, sizes,
    no Hopper GPU): the caller's XLA path (or error) then applies as without a mesh.

    Rows split over all those axes (none is left out: an axis left out would repeat the
    kernels on its devices), the weight enters replicated, and each device returns its
    partial sums as one row of a ``[devices]`` array that is summed outside the
    ``shard_map``, so the transpose needs no ``psum`` of the cotangent. Inside, every axis is
    manual and the kernels take the call as on one device. A float32 weight with a narrower
    compute dtype (and columns that split over the devices) enters replicated in the compute
    dtype, and its float32 gradient is reduce-scattered over its columns
    (:func:`_column_gradient_copy`) instead of all-reduced by the transpose.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return None
    manual = set(getattr(mesh, "manual_axes", ()))
    axes = tuple(name for name, size in mesh.shape.items() if size > 1 and name not in manual)
    count = 1
    for name in axes:
        count *= mesh.shape[name]
    if not axes or hidden.shape[0] % count:
        return None
    try:
        from jla.ops.mosaic_gpu import linear_cross_entropy as fused
    except ImportError:  # a JAX build without the Mosaic GPU toolchain
        return None
    if fused.unsupported_reason(hidden, weight, compute_dtype, devices=False) is not None:
        return None
    spec = jax.sharding.PartitionSpec
    if _gather_in_compute_dtype(weight, compute_dtype, count):
        # A float32 weight (the presets' parameters) enters in the compute dtype, so the all-gather moves half
        # the bytes; the kernels cast it to that dtype first, so they see the same values. Its float32
        # gradient is reduce-scattered over the columns (each device's share of the weight): the same float32
        # sum as the transpose's all-reduce, which XLA turned into this reduce-scatter under FSDP anyway
        # (benchmarks/multi_gpu.md).
        def body_cols(h, w_c, w_cols, t):
            w = _column_gradient_copy(w_c, w_cols, axes)
            out = linear_cross_entropy_objective(h, w, t, compute_dtype=compute_dtype, **kw)
            return tuple(jnp.reshape(x, (1,)) for x in out)

        # Pin the cast to the column split, so XLA casts each device's columns and gathers the compute
        # dtype (left alone, its sharding propagation gathers the float32 weight and casts after).
        w_c = lax.with_sharding_constraint(lax.stop_gradient(weight).astype(compute_dtype), spec(None, axes))
        parts = jax.shard_map(body_cols, mesh=mesh, in_specs=(spec(axes, None), spec(), spec(None, axes), spec(axes)),
                              out_specs=(spec(axes),) * 4, check_vma=False,
                              axis_names=frozenset(axes))(hidden, w_c, weight, targets)
        return CrossEntropyObjective(*(jnp.sum(x) for x in parts))

    def body(h, w, t):
        out = linear_cross_entropy_objective(h, w, t, compute_dtype=compute_dtype, **kw)
        return tuple(jnp.reshape(x, (1,)) for x in out)

    parts = jax.shard_map(body, mesh=mesh, in_specs=(spec(axes, None), spec(), spec(axes)),
                          out_specs=(spec(axes),) * 4, check_vma=False,
                          axis_names=frozenset(axes))(hidden, weight, targets)
    return CrossEntropyObjective(*(jnp.sum(x) for x in parts))


def _mosaic_gpu_objective(hidden, weight, targets, valid, impl, row_chunk, label_smoothing, z_loss_weight,
                          compute_dtype, fp8=False):
    """``(objective_sum, loss_sum, z_loss_sum)`` from the kernels, or ``None`` to use XLA (``impl="auto"``)."""
    try:
        from jla.ops.mosaic_gpu import linear_cross_entropy as fused
    except ImportError as err:  # a JAX build without the Mosaic GPU toolchain
        if impl == "mosaic_gpu":
            raise ValueError(f"impl='mosaic_gpu' needs Pallas Mosaic GPU: {err}") from err
        return None
    reason = fused.unsupported_reason(hidden, weight, compute_dtype)
    if reason is not None:
        if impl == "mosaic_gpu":
            raise ValueError(f"impl='mosaic_gpu' cannot run this call: {reason}")
        return None
    return fused.objective_sums(hidden, weight, targets, valid, label_smoothing=float(label_smoothing),
                                z_loss_weight=float(z_loss_weight), row_chunk=int(row_chunk),
                                compute_dtype=compute_dtype, fp8=bool(fp8))


def _gather_in_compute_dtype(weight: Array, compute_dtype, devices: int) -> bool:
    """Whether :func:`_per_shard_objective` gathers ``weight`` in ``compute_dtype`` and reduce-scatters its
    float32 gradient over the columns: a float32 weight, a narrower compute dtype and columns that split
    evenly over the ``devices`` of the row split."""
    return (jnp.dtype(weight.dtype) == jnp.float32 and jnp.dtype(compute_dtype).itemsize < 4
            and weight.shape[1] % devices == 0)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _column_gradient_copy(w_c: Array, w_cols: Array, axes: tuple[str, ...]) -> Array:
    """``w_c`` (the whole weight in the compute dtype, gathered) as float32, for use inside a ``shard_map``
    over ``axes``. Its gradient goes to ``w_cols`` (this device's float32 columns of the weight, split over
    ``axes`` in order): the float32 cotangent reduce-scattered over ``axes`` along the columns. ``w_c``
    gets none (it is the ``stop_gradient`` cast of the same weight)."""
    return w_c.astype(jnp.float32)


def _column_gradient_copy_fwd(w_c, w_cols, axes):
    return w_c.astype(jnp.float32), jnp.zeros((0,), w_cols.dtype)


def _column_gradient_copy_bwd(axes, tag, cotangent):
    grad = lax.psum_scatter(cotangent.astype(jnp.float32), axes, scatter_dimension=1, tiled=True)
    return None, grad.astype(tag.dtype)


_column_gradient_copy.defvjp(_column_gradient_copy_fwd, _column_gradient_copy_bwd)
