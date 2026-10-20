# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-time caches: position-addressed KV caches and recurrent mixer state.

A :class:`KVCache` stores keys, values and the token position held in every
slot (``-1`` marks an empty slot). A dense cache has one slot per position
(``slot = position``, positions ``>= size`` are dropped); a ring cache for
sliding-window layers has ``window`` slots (``slot = position % window``) and
keeps the latest token per slot. Positions within one row must increase from
call to call.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["KVCache", "RecurrentCache"]

Array = jax.Array


class KVCache(eqx.Module):
    keys: Array  # [B, S, Hkv, D]
    values: Array  # [B, S, Hkv, D]
    positions: Array  # [B, S] int32, -1 = empty
    ring: bool = eqx.field(static=True)

    @classmethod
    def create(cls, batch: int, size: int, kv_heads: int, head_dim: int, dtype: jnp.dtype,
               *, ring: bool) -> KVCache:
        if size < 1:
            raise ValueError(f"cache size must be >= 1; got {size}")
        return cls(
            keys=jnp.zeros((batch, size, kv_heads, head_dim), dtype),
            values=jnp.zeros((batch, size, kv_heads, head_dim), dtype),
            positions=jnp.full((batch, size), -1, jnp.int32),
            ring=ring,
        )

    @property
    def size(self) -> int:
        return self.keys.shape[1]

    def write(self, k: Array, v: Array, positions: Array) -> KVCache:
        """Insert a block ``k, v [B, T, Hkv, D]`` at ``positions [B, T]`` (``< 0`` = skip)."""
        batch, length = positions.shape
        size = self.size
        positions = positions.astype(jnp.int32)
        rows = jnp.arange(batch, dtype=jnp.int32)[:, None]
        k = k.astype(self.keys.dtype)
        v = v.astype(self.values.dtype)
        if not self.ring:
            slots = jnp.where((positions >= 0) & (positions < size), positions, size)
            return KVCache(
                keys=self.keys.at[rows, slots].set(k, mode="drop"),
                values=self.values.at[rows, slots].set(v, mode="drop"),
                positions=self.positions.at[rows, slots].set(positions, mode="drop"),
                ring=False,
            )
        # Ring: several block tokens can share a slot; the latest one wins. Scatter
        # conflicts have no defined order on accelerators, so pick the winner with a
        # scatter-max over the block order and then do one dense replacement.
        active = positions >= 0
        slots = jnp.where(active, positions % size, 0)
        order = jnp.where(active, jnp.arange(length, dtype=jnp.int32)[None, :], -1)
        latest = jnp.full((batch, size), -1, jnp.int32).at[rows, slots].max(order)
        chosen = jnp.maximum(latest, 0)
        new_pos = positions[rows, chosen]
        update = (latest >= 0) & (new_pos >= self.positions)
        return KVCache(
            keys=jnp.where(update[..., None, None], k[rows, chosen], self.keys),
            values=jnp.where(update[..., None, None], v[rows, chosen], self.values),
            positions=jnp.where(update, new_pos, self.positions),
            ring=True,
        )


class RecurrentCache(eqx.Module):
    """Decode state of a recurrent mixer.

    ``state`` is the op state (``[B, H, K, V]`` float32, or a
    ``jla.ops.reference.Mamba3State``), ``conv`` the short-conv input states
    ``[B, K - 1, C]`` (linear attention: q, k, v; Mamba-2: x and B|C) and ``angle``
    Mamba-3's cumulative rotation angle ``[B, H, S]``.
    """

    state: Any
    conv: tuple[Array, ...] | None = None
    angle: Array | None = None
