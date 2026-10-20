# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert parallelism for :class:`jla.layers.moe.MoE`: experts sharded over a mesh axis.

With ``MoEConfig.expert_parallel_axis = "ep"`` (and ``MeshConfig.ep > 1``) the
expert weights ``[E, ...]`` are split over the ``ep`` mesh axis (``E / ep`` local
experts per device) and the tokens over the token axes ``(dp, fsdp, ep)`` (those
present in the mesh). The router, the balance and z-losses and the capacity
decisions use the global token set exactly as without expert parallelism. Inside
``jax.shard_map`` every device then

1. packs its routes, sorted by expert, into one block of ``S`` rows per ``ep`` peer
   (``[ep, S, D]`` token rows plus each row's local expert and gate);
2. exchanges the blocks over ``ep`` with ``jax.lax.all_to_all`` (whole blocks; JAX
   0.11.2's ``jax.lax.ragged_all_to_all``, which would move only the filled rows, has no
   XLA:CPU lowering and fails with a CUDA illegal-address error on H100 even as a
   one-op program, so it is not offered);
3. runs its local experts on the received rows as grouped GEMMs (``jax.lax.ragged_dot``
   or the Mosaic GPU kernels, ``MoEConfig.ragged_dot_impl``), the gate applied before
   the down projection as in the layer without expert parallelism;
4. sends the results back with the inverse all-to-all and sums every token's routes.

``S`` defaults to the worst case, ``local_tokens * min(top_k, E / ep)`` (every route
of the device can target one peer), so dispatch never drops a route and the layer
equals the one without expert parallelism: dropless routing, or capacity routing
with the same global slot-major capacity decisions (every first choice ahead of
every second choice, tokens in global order), computed from per-device expert
counts gathered over the token axes. ``MoEConfig.expert_parallel_buffer_factor = f``
bounds the blocks to ``ceil(f * local_tokens * top_k / ep)`` rows (MaxText's
``ragged_buffer_factor``): a full block keeps its routes in slot-major order, as
capacity routing does (every first choice ahead of every second choice, then the
device's token order); the routes past it are dropped (the residual carries the
token) and counted in ``dropped_fraction`` together with capacity drops.

The dispatch needs a context mesh with the axis (``jax.set_mesh(mesh)``; the
:class:`jla.training.trainer.Trainer` sets it); a context mesh without the axis is
an error. Without any context mesh (decoding often runs without one) the layer runs
its own dispatch, which expert parallelism over one device equals. Expert weights enter the
``shard_map`` sharded over ``ep`` only, so FSDP/TP shards of their other
dimensions are all-gathered first (in the compute dtype) and their gradients
reduced back by GSPMD; parameters are summed over the token axes by the
``shard_map`` transpose.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from jla.layers.mlp import swiglu
from jla.ops.moe_permute import combine_rows, dispatch_rows, group_counts

if TYPE_CHECKING:
    from jla.layers.moe import MoE

__all__ = ["TOKEN_AXES", "block_rows", "expert_parallel_moe", "token_axes"]

Array = jax.Array

#: Mesh axes that split the tokens ahead of the expert axis, in batch-sharding order
#: (``jla.distributed.sharding.DEFAULT_RULES["batch"]``).
TOKEN_AXES: tuple[str, ...] = ("dp", "fsdp")


def token_axes(mesh: Any, axis: str) -> tuple[str, ...]:
    """Mesh axes the layer splits tokens over: ``TOKEN_AXES`` present in ``mesh``, then ``axis``."""
    return tuple(name for name in TOKEN_AXES if name in mesh.axis_names and name != axis) + (axis,)


def block_rows(local_tokens: int, top_k: int, local_experts: int, ep: int, buffer_factor: float | None) -> int:
    """Rows ``S`` of every per-peer dispatch block."""
    worst = local_tokens * min(top_k, local_experts)
    if buffer_factor is None:
        return worst
    return max(1, min(worst, math.ceil(buffer_factor * local_tokens * top_k / ep)))


def expert_parallel_moe(layer: MoE, x: Array, routing: Any, *, dropless: bool) -> tuple[Array, Array]:
    """Routed experts of ``layer`` on flat tokens ``x [N, D]`` -> ``(y [N, D], dropped_fraction)``.

    ``routing`` is the layer's global routing (``experts``/``gates`` ``[N, k]``);
    ``dropless=False`` applies ``capacity_factor`` exactly as ``MoE._capacity`` does.
    """
    cfg = layer.cfg
    axis = cfg.expert_parallel_axis
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        # Decoding often runs without a context mesh: the layer's own dispatch, which expert
        # parallelism over one device equals.
        return layer._dropless(x, routing) if dropless else layer._capacity(x, routing)
    if axis not in mesh.axis_names:
        raise ValueError(
            f"expert_parallel_axis={axis!r} is not an axis of the context mesh {tuple(mesh.axis_names)} "
            "(mesh.ep = 1 meshes have no 'ep' axis): set mesh.ep > 1 or unset expert_parallel_axis"
        )
    ep = int(mesh.shape[axis])
    experts, top_k = cfg.num_experts, cfg.top_k
    if experts % ep:
        raise ValueError(f"num_experts={experts} is not divisible by mesh axis {axis!r} of size {ep}")
    axes = token_axes(mesh, axis)
    shards = math.prod(int(mesh.shape[name]) for name in axes)
    tokens = x.shape[0]
    if tokens % shards:
        raise ValueError(
            f"expert parallelism splits tokens over mesh axes {axes} ({shards} shards); got {tokens} tokens"
        )
    local_experts = experts // ep
    rows = block_rows(tokens // shards, top_k, local_experts, ep, cfg.expert_parallel_buffer_factor)
    capacity = None
    if not dropless:
        routes = tokens * top_k
        capacity = max(1, min(routes, math.ceil(cfg.capacity_factor * routes / experts)))
    dtype = jnp.dtype(layer.compute_dtype)
    body = functools.partial(
        _dispatch,
        axis=axis,
        token_axes=axes,
        ep=ep,
        local_experts=local_experts,
        rows=rows,
        capacity=capacity,
        dtype=dtype,
        impl=cfg.ragged_dot_impl,
    )
    tok = P(axes)
    # check_vma=False: the Pallas (Mosaic GPU) grouped GEMMs do not type their outputs as varying
    # over the manual axes, and an implicit invariant-to-varying cast of them would transpose to a
    # psum of per-device cotangents. Without the check, cotangents of the replicated weights are
    # summed over the token axes (and outputs replicated over tp divided by it) by the transpose.
    y, sent = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(tok, tok, tok, P(axis), P(axis)),
        out_specs=(tok, tok),
        check_vma=False,
    )(x, routing.experts, routing.gates, layer.gate_up.astype(dtype), layer.down.astype(dtype))
    return y, 1.0 - jnp.mean(sent.astype(jnp.float32))


def _capacity_kept(experts: Array, capacity: int, token_axes: tuple[str, ...], num_experts: int) -> Array:
    """Global slot-major capacity decisions for this device's routes (``[n, k]`` -> ``[n * k]``).

    A route's queue position at its expert counts the routes of earlier slots on
    every device, of the same slot on earlier devices and of earlier local tokens.
    """
    onehot = jax.nn.one_hot(experts.T, num_experts, dtype=jnp.int32)  # [k, n, E]
    local_position = jnp.cumsum(onehot, axis=1) - onehot
    counts = lax.all_gather(jnp.sum(onehot, axis=1), token_axes)  # [P, k, E]
    shard = lax.axis_index(token_axes)
    totals = jnp.sum(counts, axis=0)  # [k, E]
    earlier_slots = jnp.cumsum(totals, axis=0) - totals
    earlier_shards = jnp.sum(jnp.where((jnp.arange(counts.shape[0]) < shard)[:, None, None], counts, 0), axis=0)
    position = local_position + (earlier_slots + earlier_shards)[:, None, :]
    position = jnp.take_along_axis(position, experts.T[..., None], axis=2)[..., 0]  # [k, n]
    return (position < capacity).T.reshape(-1)


def _exchange(blocks: Array, axis: str) -> Array:
    """Block ``p`` of ``blocks [ep, S, ...]`` to peer ``p``; returns block ``q`` from peer ``q``."""
    return lax.all_to_all(blocks, axis, 0, 0, tiled=True)


def _grouped(lhs: Array, rhs: Array, group_sizes: Array, impl: str) -> Array:
    """Grouped GEMM; rows past ``sum(group_sizes)`` (block padding) are zero."""
    if impl == "xla":
        return lax.ragged_dot(lhs, rhs, group_sizes)
    from jla.ops.mosaic_gpu import ragged_dot as mosaic_ragged_dot

    return mosaic_ragged_dot.grouped_matmul(lhs, rhs, group_sizes, impl=impl, full_groups=False)


def _dispatch(
    x: Array,
    experts: Array,
    gates: Array,
    gate_up: Array,
    down: Array,
    *,
    axis: str,
    token_axes: tuple[str, ...],
    ep: int,
    local_experts: int,
    rows: int,
    capacity: int | None,
    dtype: jnp.dtype,
    impl: str,
) -> tuple[Array, Array]:
    """Per-device body: local tokens ``[n, D]``, local experts ``[E/ep, ...]``."""
    n, d_model = x.shape
    top_k = experts.shape[1]
    num_experts = ep * local_experts
    d_ff = down.shape[1]
    expert = experts.reshape(-1)  # token-major routes [n * k]
    gate = gates.reshape(-1)
    token = jnp.repeat(jnp.arange(n, dtype=jnp.int32), top_k)
    kept = jnp.ones(expert.shape, bool) if capacity is None else _capacity_kept(experts, capacity, token_axes, num_experts)

    # (1) Pack: routes sorted by expert, so each peer's block is ordered by its local expert.
    # Token rows move with jla.ops.moe_permute's gathers (their transposes are gathers too).
    peer = expert // local_experts
    local = expert % local_experts
    routes = expert.shape[0]
    if rows < n * min(top_k, local_experts):
        # Bounded blocks: each peer's block keeps its routes in slot-major order (first choices, then
        # second choices, ...; local token order within a choice), as capacity routing does.
        priority = (jnp.arange(routes, dtype=jnp.int32) % top_k) * n + token
        by_priority = jnp.argsort(jnp.where(kept, peer * routes + priority, ep * routes), stable=True)
        queue = jnp.zeros_like(by_priority).at[by_priority].set(jnp.arange(routes, dtype=by_priority.dtype),
                                                                unique_indices=True)
        per_peer = group_counts(jnp.where(kept, peer, ep), ep)
        kept = kept & (queue - (jnp.cumsum(per_peer) - per_peer)[peer] < rows)
    order = jnp.argsort(jnp.where(kept, expert, num_experts), stable=True)
    rank = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype), unique_indices=True)
    per_peer = group_counts(jnp.where(kept, peer, ep), ep)
    position = rank - (jnp.cumsum(per_peer) - per_peer)[peer]
    sent = kept & (position < rows)
    # Row of every route in the flat [ep * S] send buffer (ep * S: not sent), and the route each row holds.
    send_rows = ep * rows
    dst = jnp.where(sent, peer * rows + position, send_rows).astype(jnp.int32)
    holder = jnp.full((send_rows + 1,), routes, jnp.int32).at[dst].set(jnp.arange(routes, dtype=jnp.int32))[:-1]
    send_x = dispatch_rows(x.astype(dtype), holder // top_k, dst, top_k).reshape(ep, rows, d_model)
    send_g = dispatch_rows(gate[:, None], holder, dst, 1).reshape(ep, rows)
    send_e = jnp.where(holder < routes, local[jnp.minimum(holder, routes - 1)], local_experts).reshape(ep, rows)

    # (2) Exchange: block q of the result came from peer q.
    recv_x = _exchange(send_x, axis)
    recv_g = _exchange(send_g, axis)
    recv_e = _exchange(send_e, axis)

    # (3) Local experts: rows sorted by local expert, padding (id = local_experts) last and in no group.
    ids = recv_e.reshape(-1)
    m = ids.shape[0]
    by_expert = jnp.argsort(ids, stable=True)
    inverse = jnp.zeros_like(by_expert).at[by_expert].set(jnp.arange(m, dtype=by_expert.dtype), unique_indices=True)
    group_sizes = group_counts(ids, local_experts)
    xs = dispatch_rows(recv_x.reshape(m, d_model), by_expert, inverse, 1)
    h = _grouped(xs, gate_up.reshape(local_experts, d_model, 2 * d_ff), group_sizes, impl).reshape(m, 2, d_ff)
    a = swiglu(h) * dispatch_rows(recv_g.reshape(m, 1), by_expert, inverse, 1).astype(dtype)
    y_sorted = _grouped(a, down, group_sizes, impl)
    y_recv = dispatch_rows(y_sorted, inverse, by_expert, 1).reshape(ep, rows, d_model)

    # (4) Return and combine every token's routes in float32 (token-major, as the dense layer).
    back = _exchange(y_recv, axis).reshape(send_rows, d_model)
    out = combine_rows(back, holder // top_k, dst, top_k)
    return out.astype(dtype), sent.reshape(n, top_k)
