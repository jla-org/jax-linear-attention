# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Context parallelism for causal softmax attention: the sequence axis sharded over a mesh axis.

    o = context_parallel_attention(q, k, v, mesh=mesh, axis="cp", strategy="ring",
                                   segment_ids=seg, sliding_window=None)

Inputs are global ``q [B, T, H, D]``, ``k, v [B, T, Hkv, D]`` (the layout of
:func:`jla.layers.attention.attention`), split into ``P = mesh.shape[axis]``
contiguous shards: the layout of :mod:`jla.distributed.context_parallel`, so a
hybrid model keeps ``T`` sharded through its linear-attention and softmax layers.
Masks match the single-device op: causal, ``segment_ids [B, T]`` (packed
documents), ``sliding_window`` (visible tokens including the query itself).

``strategy="ring"`` (ring attention): each device keeps its queries and passes its
key/value block around the ring with ``ppermute``; every step folds one block into
a float32 online softmax (running max, sum and output). A custom VJP saves only the
output and the log-sum-exp; the backward is a second ring pass that carries the
key/value gradients with the blocks and sends them home at the end. Blocks that no
query of a device can see are skipped, and with a sliding window the ring stops
after the ``1 + ceil((W - 1) / (T / P))`` blocks a window can reach. For global
causal attention ``load_balance=True`` reorders the sequence into the zigzag layout
(device ``i`` holds chunks ``i`` and ``2P - 1 - i`` of ``2P``) with two ``ppermute``
calls per tensor and restores the contiguous layout of the output, so every device
does the same work at every step (half a block off the diagonal). Attention sinks
(``sinks [H]``, GPT-OSS) join the softmax after the last step as one zero-valued key
per head. The block math is XLA einsums with float32 accumulation over query chunks
of ``block_q`` rows (scores ``[block_q, T / P]`` at a time; with a sliding window
narrower than the block, each chunk scores only the ``block_q + W - 1`` keys its
window can reach); ``implementation`` must be ``"auto"`` or ``"xla"``.

``strategy="ulysses"`` (DeepSpeed-Ulysses): an ``all_to_all`` turns sequence shards
into head shards (``H / P`` heads over all ``T`` tokens), the single-device
:func:`jla.layers.attention.attention` runs with any ``implementation`` (cuDNN,
Mosaic GPU, XLA) and attention sinks, and a second ``all_to_all`` turns the output
back. Needs ``H % P == 0``; KV heads are repeated as needed when ``Hkv % P != 0``.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = ["CONTEXT_PARALLEL_ATTENTION_STRATEGIES", "context_parallel_attention", "ring_steps"]

Array = jax.Array

CONTEXT_PARALLEL_ATTENTION_STRATEGIES: tuple[str, ...] = ("ring", "ulysses")
_HI = lax.Precision.HIGHEST
_F32 = jnp.float32


def _explicit_axes(mesh, spec: P) -> bool:
    """Whether ``mesh`` has Explicit axes, so arrays carry their sharding in their type.

    ``shard_map`` then needs its inputs exactly in ``spec``; the axes ``spec`` names must
    be Explicit for :func:`jax.sharding.reshard` to move them there (mixed meshes fail loudly).
    """
    types = dict(zip(mesh.axis_names, mesh.axis_types, strict=True))
    if AxisType.Explicit not in types.values():
        return False
    named = [a for entry in spec if entry is not None for a in ((entry,) if isinstance(entry, str) else entry)]
    if any(types[a] != AxisType.Explicit for a in named):
        raise ValueError(f"context parallelism on a mesh with Explicit axes needs the axes {named} to be Explicit "
                         f"too; got axis types {types}")
    return True


def _reshard(x: Array, mesh, spec: P) -> Array:
    """``x`` resharded to ``spec`` (on ``mesh`` when it is concrete, else on the context mesh)."""
    return jax.sharding.reshard(x, NamedSharding(mesh, spec) if isinstance(mesh, jax.sharding.Mesh) else spec)


def _auto_batch_axes(mesh, axis: str, batch: int) -> str | tuple[str, ...] | None:
    """The mesh axes of the default ``batch`` rule (size > 1, not ``axis``) when they divide ``batch``."""
    from jla.distributed.sharding import DEFAULT_RULES

    axes = tuple(a for a in DEFAULT_RULES["batch"] if a in mesh.axis_names and a != axis and int(mesh.shape[a]) > 1)
    if not axes or batch % math.prod(int(mesh.shape[a]) for a in axes):
        return None
    return axes[0] if len(axes) == 1 else axes


def ring_steps(seq_len: int, shards: int, sliding_window: int | None) -> int:
    """Ring steps needed: all ``shards`` blocks, or the blocks a window can reach (contiguous layout)."""
    if sliding_window is None:
        return shards
    local = seq_len // shards
    return min(shards, 1 + -(-(sliding_window - 1) // local))


# Layouts -----------------------------------------------------------------------------------------


def _positions(device: Array, local_len: int, shards: int, zigzag: bool) -> Array:
    """Global positions ``[L]`` of the tokens ``device`` holds."""
    if not zigzag:
        return device * local_len + jnp.arange(local_len, dtype=jnp.int32)
    half = local_len // 2
    r = jnp.arange(half, dtype=jnp.int32)
    return jnp.concatenate([device * half + r, (2 * shards - 1 - device) * half + r])


def _zigzag_perms(shards: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Destinations of each device's even chunk ``2d`` and odd chunk ``2d + 1`` in the zigzag layout."""

    def owner(c: int) -> int:
        return c if c < shards else 2 * shards - 1 - c

    return [(d, owner(2 * d)) for d in range(shards)], [(d, owner(2 * d + 1)) for d in range(shards)]


def _halves(x: Array) -> tuple[Array, Array]:
    half = x.shape[1] // 2
    return x[:, :half], x[:, half:]


def _to_zigzag(x: Array, axis: str, shards: int) -> Array:
    """Contiguous shard (chunks ``2d, 2d + 1``) -> zigzag shard (chunks ``d, 2P - 1 - d``), along dim 1."""
    lo, hi = _halves(x)
    even, odd = _zigzag_perms(shards)
    got_even = lax.ppermute(lo, axis, even)
    got_odd = lax.ppermute(hi, axis, odd)
    even_device = lax.axis_index(axis) % 2 == 0  # chunk d has the parity of d
    return jnp.concatenate(
        [jnp.where(even_device, got_even, got_odd), jnp.where(even_device, got_odd, got_even)], axis=1
    )


def _from_zigzag(x: Array, axis: str, shards: int) -> Array:
    """Inverse of :func:`_to_zigzag`."""
    first, second = _halves(x)
    even_device = lax.axis_index(axis) % 2 == 0
    even, odd = _zigzag_perms(shards)
    lo = lax.ppermute(jnp.where(even_device, first, second), axis, [(dst, src) for src, dst in even])
    hi = lax.ppermute(jnp.where(even_device, second, first), axis, [(dst, src) for src, dst in odd])
    return jnp.concatenate([lo, hi], axis=1)


# Blockwise attention -----------------------------------------------------------------------------


def _mask(q_pos: Array, k_pos: Array, seg_q: Array | None, seg_k: Array | None, window: int | None) -> Array:
    """Visibility ``[B | 1, 1, 1, Lq, Lk]`` (True = attend)."""
    diff = q_pos[:, None] - k_pos[None, :]
    mask = diff >= 0
    if window is not None:
        mask = mask & (diff < window)
    mask = mask[None]
    if seg_q is not None:
        mask = mask & (seg_q[:, :, None] == seg_k[:, None, :])
    return mask[:, None, None]


def _scores(qg: Array, k: Array, scale: float) -> Array:
    """``qg [B, Lq, Hkv, G, D]``, ``k [B, Lk, Hkv, D]`` -> float32 ``[B, Hkv, G, Lq, Lk]``."""
    return jnp.einsum("bqhgd,bkhd->bhgqk", qg, k, precision=_HI, preferred_element_type=_F32) * scale


def _fold(state, qg, k, v, mask, scale):
    """Fold one key/value block into the online-softmax state ``(m, l, acc)``."""
    m, l, acc = state  # [B, Hkv, G, Lq], [B, Hkv, G, Lq], [B, Hkv, G, Lq, D]
    s = jnp.where(mask, _scores(qg, k, scale), -jnp.inf)
    m_new = jnp.maximum(m, jnp.max(s, axis=-1))
    m_use = jnp.where(jnp.isneginf(m_new), 0.0, m_new)  # rows with nothing visible yet
    p = jnp.exp(s - m_use[..., None])
    alpha = jnp.exp(m - m_use)
    pv = jnp.einsum("bhgqk,bkhd->bhgqd", p.astype(v.dtype), v, precision=_HI, preferred_element_type=_F32)
    return m_new, alpha * l + jnp.sum(p, axis=-1), alpha[..., None] * acc + pv


def _block_grads(qg, k, v, do_g, lse, delta, mask, scale):
    """``dq [B, Lq, Hkv, G, D]``, ``dk, dv [B, Lk, Hkv, D]`` (float32) of one block."""
    p = jnp.where(mask, jnp.exp(_scores(qg, k, scale) - lse[..., None]), 0.0)
    dv = jnp.einsum("bhgqk,bqhgd->bkhd", p.astype(do_g.dtype), do_g, precision=_HI, preferred_element_type=_F32)
    dp = jnp.einsum("bqhgd,bkhd->bhgqk", do_g, v, precision=_HI, preferred_element_type=_F32)
    ds = p * (dp - delta[..., None])
    dq = jnp.einsum("bhgqk,bkhd->bqhgd", ds.astype(k.dtype), k, precision=_HI, preferred_element_type=_F32)
    dk = jnp.einsum("bhgqk,bqhgd->bkhd", ds.astype(qg.dtype), qg, precision=_HI, preferred_element_type=_F32)
    return dq * scale, dk * scale, dv


def _band(window: int | None, block_q: int, keys: int) -> int | None:
    """Keys ``block_q`` consecutive queries can see under ``window``; ``None`` when that is all ``keys``."""
    if window is None or block_q + window - 1 >= keys:
        return None
    return block_q + window - 1


def _band_view(k, v, k_pos, seg_k, first_q, window: int, band: int):
    """The ``band`` consecutive keys holding every key queries ``first_q, first_q + 1, ...`` can see.

    Needs consecutive key positions (the contiguous layout, the only one a window runs in).
    """
    start = jnp.clip(first_q - window + 1 - k_pos[0], 0, k.shape[1] - band)
    take = partial(lax.dynamic_slice_in_dim, start_index=start, slice_size=band, axis=1)
    seg = None if seg_k is None else take(seg_k)
    return start, take(k), take(v), lax.dynamic_slice_in_dim(k_pos, start, band), seg


def _sees(q_pos: Array, k_pos: Array, window: int) -> Array:
    """Whether some query of the consecutive positions ``q_pos`` sees some key of ``k_pos``."""
    return (q_pos[-1] >= k_pos[0]) & (q_pos[0] - k_pos[-1] < window)


def _fold_q(state, qg, k, v, q_pos, k_pos, seg_q, seg_k, window, scale, block_q):
    """:func:`_fold` over query chunks of ``block_q`` rows (scores ``[block_q, Lk]`` at a time).

    With a sliding window narrower than the block, a chunk scores only the
    ``block_q + window - 1`` keys its queries can reach.
    """
    length = qg.shape[1]
    if block_q is None or block_q >= length:
        return _fold(state, qg, k, v, _mask(q_pos, k_pos, seg_q, seg_k, window), scale)
    n, bq = length // block_q, block_q
    band = _band(window, bq, k.shape[1])
    m, l, acc = state
    b, hk, g = m.shape[:3]

    def stats(x):  # [B, Hkv, G, L] -> [n, B, Hkv, G, bq]
        return x.reshape(b, hk, g, n, bq).transpose(3, 0, 1, 2, 4)

    def unstats(x):
        return x.transpose(1, 2, 3, 0, 4).reshape(b, hk, g, length)

    accs = acc.reshape(b, hk, g, n, bq, -1).transpose(3, 0, 1, 2, 4, 5)
    qs = qg.reshape(b, n, bq, *qg.shape[2:]).swapaxes(0, 1)  # [n, B, bq, Hkv, G, D]
    segs = None if seg_q is None else seg_q.reshape(b, n, bq).swapaxes(0, 1)

    def one(xs):
        chunk_state, qc, pc, sc = xs
        if band is None:
            return _fold(chunk_state, qc, k, v, _mask(pc, k_pos, sc, seg_k, window), scale)
        _, kb, vb, kp, skb = _band_view(k, v, k_pos, seg_k, pc[0], window, band)
        fold = lambda st: _fold(st, qc, kb, vb, _mask(pc, kp, sc, skb, window), scale)  # noqa: E731
        return lax.cond(_sees(pc, kp, window), fold, lambda st: st, chunk_state)

    mo, lo, ao = lax.map(one, ((stats(m), stats(l), accs), qs, q_pos.reshape(n, bq), segs))
    return unstats(mo), unstats(lo), ao.transpose(1, 2, 3, 0, 4, 5).reshape(acc.shape)


def _grads_q(qg, k, v, do_g, lse, delta, q_pos, k_pos, seg_q, seg_k, window, scale, block_q):
    """:func:`_block_grads` over query chunks of ``block_q`` rows (key gradients summed over chunks)."""
    length = qg.shape[1]
    if block_q is None or block_q >= length:
        return _block_grads(qg, k, v, do_g, lse, delta, _mask(q_pos, k_pos, seg_q, seg_k, window), scale)
    n, bq = length // block_q, block_q
    b = qg.shape[0]
    band = _band(window, bq, k.shape[1])

    def rows(x):  # [B, L, ...] -> [n, B, bq, ...]
        return x.reshape(b, n, bq, *x.shape[2:]).swapaxes(0, 1)

    def stats(x):  # [B, Hkv, G, L] -> [n, B, Hkv, G, bq]
        return x.reshape(*x.shape[:3], n, bq).transpose(3, 0, 1, 2, 4)

    segs = None if seg_q is None else seg_q.reshape(b, n, bq).swapaxes(0, 1)

    def one(carry, xs):
        qc, doc, lc, dc, pc, sc = xs
        if band is None:
            dq, dk, dv = _block_grads(qc, k, v, doc, lc, dc, _mask(pc, k_pos, sc, seg_k, window), scale)
            return (carry[0] + dk, carry[1] + dv), dq
        start, kb, vb, kp, skb = _band_view(k, v, k_pos, seg_k, pc[0], window, band)

        def grads(_):
            return _block_grads(qc, kb, vb, doc, lc, dc, _mask(pc, kp, sc, skb, window), scale)

        def nothing(_):
            return jnp.zeros(qc.shape, _F32), jnp.zeros(kb.shape, _F32), jnp.zeros(vb.shape, _F32)

        dq, dk, dv = lax.cond(_sees(pc, kp, window), grads, nothing, None)

        def add(total, part):  # total[:, start:start + band] += part
            window_total = lax.dynamic_slice_in_dim(total, start, band, axis=1)
            return lax.dynamic_update_slice_in_dim(total, window_total + part, start, axis=1)

        return (add(carry[0], dk), add(carry[1], dv)), dq

    zeros = (jnp.zeros(k.shape, _F32), jnp.zeros(v.shape, _F32))
    xs = (rows(qg), rows(do_g), stats(lse), stats(delta), q_pos.reshape(n, bq), segs)
    (dk, dv), dq = lax.scan(one, zeros, xs)
    return dq.swapaxes(0, 1).reshape(qg.shape), dk, dv


def _auto_block_q(rows: int, target: int = 512) -> int | None:
    """Largest divisor of ``rows`` up to ``target`` (``None``: no chunking needed or no useful divisor)."""
    if rows <= target:
        return None
    best = max(d for d in range(1, target + 1) if rows % d == 0)
    return best if best >= 64 else None


def _tail(state, half: int):
    m, l, acc = state
    return m[..., half:], l[..., half:], acc[..., half:, :]


def _join(state, tail, half: int):
    return tuple(jnp.concatenate([x[..., :half], t], axis=-1) for x, t in zip(state[:2], tail[:2])) + (
        jnp.concatenate([state[2][..., :half, :], tail[2]], axis=-2),
    )


# Ring attention ----------------------------------------------------------------------------------


def _ring_core(axis: str, shards: int, steps: int, zigzag: bool, scale: float, window: int | None,
               block_q: int | None):
    """Per-device ring attention with its custom VJP (called inside ``shard_map``)."""
    forward = [(j, (j + 1) % shards) for j in range(shards)]
    home = [(j, (j - steps + 1) % shards) for j in range(shards)]

    def setup(q, k, seg_all):
        batch, length, heads, dim = q.shape
        kv_heads = k.shape[2]
        idx = lax.axis_index(axis)
        qg = q.reshape(batch, length, kv_heads, heads // kv_heads, dim)
        q_pos = _positions(idx, length, shards, zigzag)
        seg_q = None if seg_all is None else lax.dynamic_index_in_dim(seg_all, idx, 0, keepdims=False)
        return idx, qg, q_pos, seg_q

    def block_view(idx, s, seg_all, length):
        src = (idx - s) % shards
        k_pos = _positions(src, length, shards, zigzag)
        seg_k = None if seg_all is None else lax.dynamic_index_in_dim(seg_all, src, 0, keepdims=False)
        return src, k_pos, seg_k

    def fwd_step(state, qg, q_pos, seg_q, k, v, idx, s, seg_all):
        length = qg.shape[1]
        src, k_pos, seg_k = block_view(idx, s, seg_all, length)
        full = lambda st: _fold_q(st, qg, k, v, q_pos, k_pos, seg_q, seg_k, window, scale, block_q)  # noqa: E731
        if not zigzag:
            visible = q_pos[-1] >= k_pos[0]
            if window is not None:
                visible = visible & (q_pos[0] - k_pos[-1] < window)
            return lax.cond(visible, full, lambda st: st, state)
        half = length // 2
        sk = None if seg_k is None else seg_k[:, :half]

        def early_keys(st):  # source before us: every query sees only the source's first chunk
            return _fold_q(st, qg, k[:, :half], v[:, :half], q_pos, k_pos[:half], seg_q, sk, window, scale,
                           block_q)

        def late_queries(st):  # source after us: only our second chunk sees the source (both chunks)
            sq = None if seg_q is None else seg_q[:, half:]
            tail = _fold_q(_tail(st, half), qg[:, half:], k, v, q_pos[half:], k_pos, sq, seg_k, window, scale,
                           block_q)
            return _join(st, tail, half)

        branch = jnp.where(s == 0, 0, jnp.where(src < idx, 1, 2))
        return lax.switch(branch, (full, early_keys, late_queries), state)

    def bwd_step(qg, q_pos, seg_q, k, v, do_g, lse, delta, idx, s, seg_all):
        length = qg.shape[1]
        src, k_pos, seg_k = block_view(idx, s, seg_all, length)
        f32 = _F32

        def full(_):
            return _grads_q(qg, k, v, do_g, lse, delta, q_pos, k_pos, seg_q, seg_k, window, scale, block_q)

        def nothing(_):
            return (jnp.zeros(qg.shape, f32), jnp.zeros(k.shape, f32), jnp.zeros(v.shape, f32))

        if not zigzag:
            visible = q_pos[-1] >= k_pos[0]
            if window is not None:
                visible = visible & (q_pos[0] - k_pos[-1] < window)
            return lax.cond(visible, full, nothing, None)
        half = length // 2

        def early_keys(_):
            sk = None if seg_k is None else seg_k[:, :half]
            dq, dk, dv = _grads_q(qg, k[:, :half], v[:, :half], do_g, lse, delta, q_pos, k_pos[:half], seg_q, sk,
                                  window, scale, block_q)
            pad = jnp.zeros((dk.shape[0], length - half) + dk.shape[2:], f32)
            return dq, jnp.concatenate([dk, pad], axis=1), jnp.concatenate([dv, pad], axis=1)

        def late_queries(_):
            sq = None if seg_q is None else seg_q[:, half:]
            dq, dk, dv = _grads_q(qg[:, half:], k, v, do_g[:, half:], lse[..., half:], delta[..., half:],
                                  q_pos[half:], k_pos, sq, seg_k, window, scale, block_q)
            return jnp.concatenate([jnp.zeros_like(dq), dq], axis=1), dk, dv

        branch = jnp.where(s == 0, 0, jnp.where(src < idx, 1, 2))
        return lax.switch(branch, (full, early_keys, late_queries), None)

    def forward_pass(q, k, v, seg_all, sinks):
        idx, qg, q_pos, seg_q = setup(q, k, seg_all)
        batch, length, kv_heads, group, dim = qg.shape
        stats = jnp.full((batch, kv_heads, group, length), -jnp.inf, _F32)
        state = (stats, jnp.zeros_like(stats), jnp.zeros((batch, kv_heads, group, length, dim), _F32))

        def step(carry, s):
            state, k_cur, v_cur = carry
            k_next, v_next = lax.ppermute(k_cur, axis, forward), lax.ppermute(v_cur, axis, forward)
            return (fwd_step(state, qg, q_pos, seg_q, k_cur, v_cur, idx, s, seg_all), k_next, v_next), None

        (state, k_cur, v_cur), _ = lax.scan(step, (state, k, v), jnp.arange(steps - 1, dtype=jnp.int32))
        m, l, acc = fwd_step(state, qg, q_pos, seg_q, k_cur, v_cur, idx, jnp.int32(steps - 1), seg_all)
        if sinks is not None:  # one virtual key per head with logit sinks[h] and a zero value
            sink = sinks.astype(_F32).reshape(kv_heads, group)[None, :, :, None]
            m_new = jnp.maximum(m, sink)
            alpha = jnp.exp(m - m_new)
            m, l, acc = m_new, alpha * l + jnp.exp(sink - m_new), alpha[..., None] * acc
        o = (acc / l[..., None]).transpose(0, 3, 1, 2, 4).reshape(q.shape).astype(q.dtype)
        return o, m + jnp.log(l)

    @jax.custom_vjp
    def core(q, k, v, seg_all, sinks):
        return forward_pass(q, k, v, seg_all, sinks)[0]

    def core_fwd(q, k, v, seg_all, sinks):
        o, lse = forward_pass(q, k, v, seg_all, sinks)
        return o, (q, k, v, seg_all, sinks, o, lse)

    def core_bwd(res, do):
        q, k, v, seg_all, sinks, o, lse = res
        idx, qg, q_pos, seg_q = setup(q, k, seg_all)
        do_g = do.reshape(qg.shape).astype(q.dtype)
        delta = jnp.einsum("bqhgd,bqhgd->bhgq", do_g.astype(_F32), o.reshape(qg.shape).astype(_F32))
        dq = jnp.zeros(qg.shape, _F32)
        dk = jnp.zeros(k.shape, _F32)
        dv = jnp.zeros(v.shape, _F32)

        def step(carry, s):
            dq, k_cur, v_cur, dk_cur, dv_cur = carry
            ddq, ddk, ddv = bwd_step(qg, q_pos, seg_q, k_cur, v_cur, do_g, lse, delta, idx, s, seg_all)
            rotate = partial(lax.ppermute, axis_name=axis, perm=forward)
            return (dq + ddq, rotate(k_cur), rotate(v_cur), rotate(dk_cur + ddk), rotate(dv_cur + ddv)), None

        (dq, k_cur, v_cur, dk, dv), _ = lax.scan(step, (dq, k, v, dk, dv), jnp.arange(steps - 1, dtype=jnp.int32))
        ddq, ddk, ddv = bwd_step(qg, q_pos, seg_q, k_cur, v_cur, do_g, lse, delta, idx, jnp.int32(steps - 1), seg_all)
        dq, dk, dv = dq + ddq, dk + ddk, dv + ddv
        if (steps - 1) % shards:  # the accumulators of block idx - steps + 1 go back to their owner
            dk, dv = lax.ppermute(dk, axis, home), lax.ppermute(dv, axis, home)
        d_sinks = None
        if sinks is not None:  # d sinks[h] = -sum_i p_sink_i (do_i . o_i)
            sink = sinks.astype(_F32).reshape(qg.shape[2], qg.shape[3])[None, :, :, None]
            d_sinks = -jnp.sum(jnp.exp(sink - lse) * delta, axis=(0, 3)).reshape(sinks.shape).astype(sinks.dtype)
        return dq.reshape(q.shape).astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), None, d_sinks

    core.defvjp(core_fwd, core_bwd)
    return core


# Ulysses -----------------------------------------------------------------------------------------


def _ulysses_body(axis: str, shards: int, scale: float, window: int | None, implementation: str):
    from jla.layers.attention import attention

    def body(q, k, v, extras):
        heads, kv_heads = q.shape[2], k.shape[2]
        repeat = shards // math.gcd(kv_heads, shards)  # divides H / Hkv because P divides H
        if repeat > 1:
            k, v = jnp.repeat(k, repeat, axis=2), jnp.repeat(v, repeat, axis=2)
        to_heads = partial(lax.all_to_all, axis_name=axis, split_axis=2, concat_axis=1, tiled=True)
        seg = extras.get("segment_ids")
        if seg is not None:
            seg = lax.all_gather(seg, axis, axis=1, tiled=True)
        sinks = extras.get("sinks")
        if sinks is not None:
            per = heads // shards
            sinks = lax.dynamic_slice_in_dim(sinks, lax.axis_index(axis) * per, per)
        o = attention(to_heads(q), to_heads(k), to_heads(v), segment_ids=seg, sliding_window=window, scale=scale,
                      implementation=implementation, sinks=sinks)
        return lax.all_to_all(o, axis, split_axis=1, concat_axis=2, tiled=True)

    return body


# Entry point -------------------------------------------------------------------------------------


def context_parallel_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    axis: str = "cp",
    batch_axis: str | tuple[str, ...] | None = None,
    strategy: str = "ring",
    load_balance: bool = True,
    segment_ids: Array | None = None,
    sliding_window: int | None = None,
    scale: float | None = None,
    implementation: str = "auto",
    sinks: Array | None = None,
    block_q: int | None = None,
) -> Array:
    """Causal GQA softmax attention with ``T`` sharded over mesh axis ``axis``.

    Same arguments and result as :func:`jla.layers.attention.attention`, plus the
    mesh (default: the context mesh of ``jax.set_mesh``), ``batch_axis`` (mesh axes
    ``B`` is sharded over; ``None`` = replicated; ``"auto"`` = the ``dp``/``fsdp`` axes of
    :data:`jla.distributed.sharding.DEFAULT_RULES` that divide ``B``), ``strategy`` (``"ring"`` or
    ``"ulysses"``), ``load_balance`` (ring only: zigzag layout for global causal
    attention; a sliding window always runs the contiguous, truncated ring) and
    ``block_q`` (ring only: query rows per score block, bounding each step to
    ``block_q x T / P`` scores; default: the largest divisor up to 512 of the block
    rows, no chunking at or below 512 rows). On a mesh with Explicit axes (the default
    of :func:`jax.make_mesh`) the inputs are resharded into the CP layout and the output
    gets the sharding of ``q``.
    """
    if strategy not in CONTEXT_PARALLEL_ATTENTION_STRATEGIES:
        raise ValueError(f"strategy must be one of {CONTEXT_PARALLEL_ATTENTION_STRATEGIES}; got {strategy!r}")
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or axis not in mesh.axis_names:
        raise ValueError(
            f"context parallelism needs a mesh with axis {axis!r}; pass mesh= or call inside jax.set_mesh(...)"
        )
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"need q [B, T, H, D] and k, v [B, T, Hkv, D]; got {q.shape}, {k.shape}, {v.shape}")
    batch, length, heads, dim = q.shape
    kv_heads = k.shape[2]
    if heads % kv_heads:
        raise ValueError(f"num_heads {heads} must be a multiple of num_kv_heads {kv_heads}")
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding_window must be >= 1")
    shards = int(mesh.shape[axis])
    if length % shards:
        raise ValueError(f"sequence length {length} is not divisible by mesh axis {axis!r} of size {shards}")
    scale = dim ** -0.5 if scale is None else float(scale)
    if batch_axis == "auto":
        batch_axis = _auto_batch_axes(mesh, axis, batch)
    seq = P(batch_axis, axis)
    extras, extra_specs = {}, {}
    if segment_ids is not None:
        extras["segment_ids"], extra_specs["segment_ids"] = segment_ids.astype(jnp.int32), seq
    if sinks is not None:
        if sinks.shape != (heads,):
            raise ValueError(f"sinks must be [H] = ({heads},); got {sinks.shape}")
        extras["sinks"], extra_specs["sinks"] = sinks, P()
    # Explicit mesh axes: move the inputs into the CP layout and give the output the sharding of q.
    out_spec = None
    if _explicit_axes(mesh, seq):
        out_spec = jax.typeof(q).sharding.spec
        q, k, v = (_reshard(x, mesh, seq) for x in (q, k, v))
        extras = {name: _reshard(x, mesh, extra_specs[name]) for name, x in extras.items()}
    layout = (lambda o: o) if out_spec is None else (lambda o: _reshard(o, mesh, out_spec))

    if strategy == "ulysses":
        if heads % shards:
            raise ValueError(f"Ulysses needs num_heads {heads} divisible by mesh axis {axis!r} of size {shards}")
        if block_q is not None:
            raise ValueError("block_q chunks the ring's score blocks; strategy='ulysses' does not take it")
        body = _ulysses_body(axis, shards, scale, sliding_window, implementation)
        return layout(jax.shard_map(body, mesh=mesh, in_specs=(seq, seq, seq, extra_specs), out_specs=seq,
                                    check_vma=False)(q, k, v, extras))

    if implementation not in ("auto", "xla"):
        raise ValueError(
            f"ring attention runs its own blockwise XLA attention; implementation must be 'auto' or 'xla', "
            f"got {implementation!r} (strategy='ulysses' runs cuDNN or Mosaic GPU locally)"
        )
    zigzag = bool(load_balance) and sliding_window is None and shards > 1
    if zigzag and (length // shards) % 2:
        raise ValueError(
            f"load_balance needs T divisible by 2 * {shards} (zigzag chunks); got T={length}. Pass load_balance=False"
        )
    rows = length // shards // (2 if zigzag else 1)  # query rows of the smallest block a step attends
    if block_q is None:
        block_q = _auto_block_q(rows)
    elif block_q < 1 or rows % block_q:
        raise ValueError(f"block_q={block_q} must divide the {rows} query rows of a ring block")
    steps = shards if zigzag else ring_steps(length, shards, sliding_window)
    core = _ring_core(axis, shards, steps, zigzag, scale, sliding_window, block_q)

    def body(q, k, v, extras):
        seg = extras.get("segment_ids")
        if zigzag:
            q, k, v = (_to_zigzag(x, axis, shards) for x in (q, k, v))
            seg = None if seg is None else _to_zigzag(seg, axis, shards)
        seg_all = None if seg is None else lax.all_gather(seg, axis)  # [P, B, L] in device order
        o = core(q, k, v, seg_all, extras.get("sinks"))
        return _from_zigzag(o, axis, shards) if zigzag else o

    return layout(jax.shard_map(body, mesh=mesh, in_specs=(seq, seq, seq, extra_specs), out_specs=seq,
                                check_vma=False)(q, k, v, extras))
