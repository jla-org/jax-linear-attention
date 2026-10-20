# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gather-only token <-> expert-row permutations for mixture-of-experts layers.

An MoE layer copies every token's ``k`` routes into expert-ordered rows (dispatch) and sums
them back per token (combine). Written as ``x[rows]`` / ``y[rows].sum``, their transposes are
XLA scatter-adds, which on GPU run as atomics: in a 64-expert top-10 layer they took more time
than the grouped GEMMs. The two ops here are each other's transpose, so both directions are
gathers plus a float32 sum over the ``k`` routes of a token:

* :func:`dispatch_rows`: ``out[s] = x[src[s]]``, rows ``[S, D]`` from tokens ``[T, D]``;
* :func:`combine_rows`: ``out[t] = sum_j y[dst[t * k + j]]`` in float32, tokens ``[T, D]`` from rows.

``src [S]`` gives each row's token and ``dst [T * k]`` each route's row (token-major routes:
route ``t * k + j`` is choice ``j`` of token ``t``). Out-of-range indices are empty: ``src[s] = T``
reads a zero row (a capacity slot nobody claimed) and ``dst[r] = S`` a zero (a dropped route).
The maps must be consistent (``src[dst[t * k + j]] = t`` for every kept route, and every row
``s`` with ``src[s] < T`` is the ``dst`` of exactly one route), which the callers guarantee
by construction; then the backward passes are exact transposes.

Inside a partly manual mesh (a pipeline stage: ``shard_map`` over ``pp`` only, the other axes
left to GSPMD), XLA's SPMD partitioner (JAX 0.11.2) aborts on the gather-only backward passes
(``Check failed`` in ``ExpandDeviceGroupsWithIota`` / ``spmd_partitioner_util``, with or without
``tp``). There both ops use the earlier plain formulation instead: the same gathers
forward, through a zero pad row for the empty indices, with autodiff's scatter-add transposes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

__all__ = ["combine_rows", "dispatch_rows", "group_counts"]

Array = jax.Array


def group_counts(ids: Array, groups: int) -> Array:
    """Entries of ``ids [R]`` per group ``[groups]`` int32 (ids outside ``[0, groups)`` count nowhere).

    A reduction over ``[R, groups]`` compares rather than a scatter-add, which runs as atomics.
    """
    return jnp.sum(ids[:, None] == jnp.arange(groups, dtype=ids.dtype)[None, :], axis=0, dtype=jnp.int32)


def _take(x: Array, idx: Array) -> Array:
    """``x[idx]`` along axis 0; out-of-range indices give zero rows."""
    return jnp.take(x, idx, axis=0, mode="fill", fill_value=0)


def _partially_manual() -> bool:
    """Whether the context mesh is manual over some axes and automatic over others of size > 1.

    That is the inside of a pipeline stage (``shard_map`` over ``pp`` only, the batch and model
    axes left to GSPMD), where the gather-only backward passes abort XLA's SPMD partitioner.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return False
    manual = set(getattr(mesh, "manual_axes", ()))
    return bool(manual) and any(int(size) > 1 for name, size in zip(mesh.axis_names, mesh.axis_sizes)
                                if name not in manual)


def _padded(x: Array) -> Array:
    """``x`` with one zero row appended (index ``len(x)`` reads zeros)."""
    return jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], axis=0)


def _sum_routes(y: Array, dst: Array, top_k: int) -> Array:
    # One gather per choice, summed in order: XLA fuses them into one pass that writes [T, ...]
    # (a gather of all [T * k, ...] rows followed by a reduction materialized them in float32).
    idx = dst.reshape(-1, top_k)
    total = _take(y, idx[:, 0]).astype(jnp.float32)
    for j in range(1, top_k):
        total = total + _take(y, idx[:, j]).astype(jnp.float32)
    return total


def dispatch_rows(x: Array, src: Array, dst: Array, top_k: int) -> Array:
    """Rows ``x[src]`` (``[S, ...]``, ``x``'s dtype) whose backward is a gather-and-sum through ``dst``."""
    if _partially_manual():
        return _padded(x)[src]
    return _dispatch_rows(x, src, dst, top_k)


def combine_rows(y: Array, src: Array, dst: Array, top_k: int) -> Array:
    """Per-token float32 sums ``[T, ...]`` of the ``k`` rows ``y[dst]``; the backward gathers through ``src``."""
    if _partially_manual():
        rows = _padded(y)[dst].astype(jnp.float32)
        return rows.reshape((dst.shape[0] // top_k, top_k) + y.shape[1:]).sum(axis=1)
    return _combine_rows(y, src, dst, top_k)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _dispatch_rows(x: Array, src: Array, dst: Array, top_k: int) -> Array:
    return _take(x, src)


def _dispatch_fwd(x, src, dst, top_k):
    return _take(x, src), (src, dst, jnp.zeros((0,), x.dtype))


def _dispatch_bwd(top_k, res, g):
    src, dst, like = res
    return _sum_routes(g, dst, top_k).astype(like.dtype), None, None


_dispatch_rows.defvjp(_dispatch_fwd, _dispatch_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _combine_rows(y: Array, src: Array, dst: Array, top_k: int) -> Array:
    return _sum_routes(y, dst, top_k)


def _combine_fwd(y, src, dst, top_k):
    return _sum_routes(y, dst, top_k), (src, dst, jnp.zeros((0,), y.dtype))


def _combine_bwd(top_k, res, g):
    src, dst, like = res
    # Cast the [T, ...] float32 cotangent before gathering [S, ...] rows from it (the same values: the
    # gather only copies), so the gather reads and the cast touches k times fewer elements.
    return _take(g.astype(like.dtype), src), None, None


_combine_rows.defvjp(_combine_fwd, _combine_bwd)
