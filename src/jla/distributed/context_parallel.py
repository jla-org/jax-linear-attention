# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/cp/ (cross-shard state passing design)
"""Context parallelism for linear attention: the sequence axis sharded over a mesh axis.

    o, final = context_parallel_linear_attention(
        "gated_deltanet", q, k, v, g=g, beta=beta, mesh=mesh, axis="cp", output_final_state=True)

Inputs are global ``[B, T, ...]`` arrays (FLA layout, as for
:func:`jla.ops.linear_attention`); ``T`` is split into ``mesh.shape[axis]``
contiguous shards. Every supported recurrence is affine in the state entering a
stretch of tokens, ``S_out = M S_in + N``. Each shard

1. runs the chunked op from a zero state: its final state is ``N``;
2. computes ``M``: ``exp(sum g)`` per head (``simple_gla``) or per key channel
   (``gla``), zero when a segment starts inside the shard; for the delta-rule kinds
   the ``[K, K]`` matrix ``prod_t (I - beta_t k_t k_t^T) exp(g_t)``, obtained by
   running the same chunked WY-form op on ``v = 0`` from ``S = I``;
3. all-gathers every shard's ``(M, N)`` and scans them in sequence order to get
   its true entering state (a segment change across a shard boundary resets it);
4. reruns the chunked op from that state for its outputs and final state.

Reverse mode differentiates through ``shard_map``. Packed documents use
``segment_ids [B, T]``; ``cu_seqlens`` is not supported.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from jla.ops import KINDS, chunked
from jla.ops.reference import segment_resets

__all__ = ["CONTEXT_PARALLEL_KINDS", "context_parallel_linear_attention"]

Array = jax.Array

_CHUNKED: dict[str, Callable[..., tuple[Array, Array | None]]] = {
    "simple_gla": chunked.simple_gla,
    "gla": chunked.gla,
    "deltanet": chunked.delta_rule,
    "gated_deltanet": chunked.gated_delta_rule,
    "kda": chunked.kda,
}
_DELTA = frozenset({"deltanet", "gated_deltanet", "kda"})
CONTEXT_PARALLEL_KINDS: tuple[str, ...] = tuple(_CHUNKED)


def _apply(kind: str, m: Array, state: Array) -> Array:
    """``M S`` for the transition ``m`` of ``kind`` (state ``[B, H, K, V]``)."""
    if kind in _DELTA:
        return jnp.einsum("bhij,bhjv->bhiv", m, state, precision=lax.Precision.HIGHEST)
    if m.ndim == 3:  # gla: [B, H, K]
        return m[..., :, None] * state
    return m[..., None, None] * state  # simple_gla: [B, H]


def _shard_body(kind: str, axis: str, has_segments: bool, has_state: bool, options: dict[str, Any]):
    fn = _CHUNKED[kind]

    def body(q, k, v, gates, extras):
        batch, local_len, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        f32 = jnp.float32
        seg = extras.get("segment_ids")
        run = {**options, "segment_ids": seg, "output_final_state": True}

        # (1) N: this shard's final state from a zero entering state.
        _, n = fn(q, k, v, *gates, initial_state=None, **run)
        # (2) M: how the entering state survives the shard.
        if kind in _DELTA:
            eye = jnp.broadcast_to(jnp.eye(key_dim, dtype=f32), (batch, heads, key_dim, key_dim))
            zero_v = jnp.zeros((batch, local_len, heads, key_dim), v.dtype)
            _, m = fn(q, k, zero_v, *gates, initial_state=eye, **run)
        else:
            m = jnp.exp(jnp.sum(gates[0].astype(f32), axis=1))  # [B, H] or [B, H, K]
            if has_segments:
                fresh = jnp.any(segment_resets(seg), axis=1)  # a segment starts inside the shard
                m = jnp.where(fresh.reshape((batch,) + (1,) * (m.ndim - 1)), 0.0, m)

        # (3) Scan the shard transitions in sequence order.
        m_all = lax.all_gather(m, axis)
        n_all = lax.all_gather(n, axis)
        if has_segments:
            first = lax.all_gather(seg[:, 0], axis)  # [P, B]
            last = lax.all_gather(seg[:, -1], axis)
            boundary = jnp.concatenate([jnp.zeros_like(first[:1], dtype=bool), first[1:] != last[:-1]], axis=0)
        else:
            boundary = jnp.zeros((m_all.shape[0], batch), bool)
        state0 = extras["initial_state"].astype(f32) if has_state else jnp.zeros((batch, heads, key_dim, value_dim), f32)

        def step(state, xs):
            m_p, n_p, reset_p = xs
            state = jnp.where(reset_p[:, None, None, None], 0.0, state)
            return _apply(kind, m_p, state) + n_p, state

        _, entering = lax.scan(step, state0, (m_all, n_all, boundary))
        s_in = entering[lax.axis_index(axis)]

        # (4) Rerun from the true entering state.
        o, final = fn(q, k, v, *gates, initial_state=s_in, **run)
        return o, final[None]

    return body


def context_parallel_linear_attention(
    kind: str,
    q: Array,
    k: Array,
    v: Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    axis: str = "cp",
    batch_axis: str | tuple[str, ...] | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
    cu_seqlens: Array | None = None,
    **options: Any,
) -> tuple[Array, Array | None]:
    """Run linear-attention ``kind`` with ``T`` sharded over mesh axis ``axis``.

    Gates (``g``, ``beta`` as listed in :data:`jla.ops.KINDS`) and the remaining
    options (``scale``, ``use_qk_l2norm``, ``chunk_size``, ...) are passed by
    keyword and forwarded to the chunked op of ``kind``. ``mesh`` defaults to the
    context mesh (``jax.set_mesh``). ``batch_axis`` names the mesh axes ``B`` is
    sharded over (``None``: replicated). Returns ``(o, final_state or None)``.
    """
    if kind not in _CHUNKED:
        raise NotImplementedError(
            f"context parallelism supports {CONTEXT_PARALLEL_KINDS}; got {kind!r}"
        )
    if cu_seqlens is not None:
        raise NotImplementedError("context parallelism does not support cu_seqlens; pass segment_ids")
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or axis not in mesh.axis_names:
        raise ValueError(
            f"context parallelism needs a mesh with axis {axis!r}; pass mesh= or call inside jax.set_mesh(...)"
        )
    gate_names = KINDS[kind][1]
    missing = [name for name in gate_names if options.get(name) is None]
    if missing:
        raise ValueError(f"{kind} needs gate inputs {gate_names}; missing {missing}")
    gates = tuple(options.pop(name) for name in gate_names)
    unknown = sorted(set(options) & ({"g", "beta", "gamma", "b", "w"} - set(gate_names)))
    if unknown:
        raise ValueError(f"{kind} takes gates {gate_names}; got unexpected {unknown}")
    if q.ndim != 4:
        raise ValueError(f"q must be [B, T, H, K]; got {q.shape}")
    shards = mesh.shape[axis]
    if q.shape[1] % shards:
        raise ValueError(f"sequence length {q.shape[1]} is not divisible by mesh axis {axis!r} of size {shards}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")

    has_segments = segment_ids is not None
    has_state = initial_state is not None
    seq = P(batch_axis, axis)
    batch_only = P(batch_axis)
    body = _shard_body(kind, axis, has_segments, has_state, options)
    extras, extra_specs = {}, {}
    if has_segments:
        extras["segment_ids"], extra_specs["segment_ids"] = segment_ids, seq
    if has_state:
        extras["initial_state"], extra_specs["initial_state"] = initial_state, batch_only
    sharded = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(seq, seq, seq, tuple(seq for _ in gates), extra_specs),
        out_specs=(seq, P(axis, batch_axis)),
        check_vma=False,
    )
    o, finals = sharded(q, k, v, gates, extras)
    return o, (finals[-1] if output_final_state else None)
