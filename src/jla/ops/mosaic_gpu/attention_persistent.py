# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2024 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/attention_mgpu.py, through the per-tile loops of
# jla.ops.mosaic_gpu.attention (which this module reuses unchanged in a persistent tile loop)
# The persistent tile loop, its longest-processing-time-first order inside L2-sized sections of (batch, head) pairs
# (heads innermost) and the next tile's query load issued as soon as the current tile's last QK^T product has
# read the query tile follow Dao-AILab flash-attention (BSD-3-Clause), hopper/tile_scheduler.hpp
# (DynamicPersistentTileScheduler, VarlenDynamicPersistentTileScheduler), hopper/flash_fwd_kernel_sm90.h and
# hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp (design only, no code copied)
"""Persistent forward of the FA3-style Mosaic GPU attention kernel (``jla.ops.mosaic_gpu.attention``).

:func:`forward` computes exactly what ``attention._forward`` computes (each query tile runs the same
ping-pong or overlapped loop, in the same order, so outputs and ``lse`` are bitwise equal) in a
**persistent** kernel: one CTA per SM, each looping over a static list of (batch, head, query tile)
items instead of one CTA per tile. What it changes is where the work lands:

- **Order**: as FlashAttention-3's persistent schedulers, the (batch, head) pairs are cut into sections of
  consecutive pairs whose keys and values fit in 32 MiB of L2 (FA3's budget; with grouped heads the query
  heads of one key/value head stay together), and inside a section the query tiles run longest first, the
  section's heads innermost (consecutive items read the same key/value blocks, or for grouped heads the same
  key/value head). The tiles of a row are ordered by their key-block count: the last tile first for causal
  attention and windows (pure index arithmetic in the kernel), and for packed documents sorted on device from
  the segment runs (FA3's varlen scheduler sorts likewise) into a table of (tile, key-block range) by rank that
  the kernel reads one item ahead, so a tile near the end of a long document comes before the short tiles of
  the documents around it.
- **Assignment**: item slots are dealt to the CTAs in rounds of one item per CTA, alternately in CTA order and
  in reverse (a snake over a list sorted longest first balances the CTAs' total work); each section is padded
  to whole rounds (padding slots are skipped), so every section starts on a round and the reversal also
  balances the section boundaries. FA3 instead hands out tiles at run time from an atomic counter; the
  static deal needs no counter and gives every warpgroup the same item list.
- **Overlap**: the TMA warpgroup streams key/value blocks through the ring of stages continuously from one
  tile into the next (no pipeline drain and refill per tile), and loads the next tile's queries as soon as a
  consumer warpgroup's last ``QK^T`` product of the current tile has read its query tile; the output is staged
  in its own shared-memory tile, so the epilogue (normalise, store ``O`` and ``lse`` by TMA) overlaps the next
  tile's loads, and no CTA is launched, initialised or drained per tile. MLA's asymmetric heads (``Dqk = 192``,
  ``Dv = 128``) fill shared memory with their key/value tiles, so there the output goes through the query tile as
  before and the next queries load once the output store has read it.

Not built here, with the measurements in ``benchmarks/attention.md``: TMA multicast of keys and values across
a CTA cluster (FA3 uses clusters only without causal or local masks, and the forward waits for keys and values
for at most 2% of its time), and FA3's document-aligned varlen tiles (a TMA copy into tiled, swizzled shared
memory needs tile-aligned global offsets).

Supported: everything ``attention._forward`` supports with its default internal options (both loops,
two or three consumer warpgroups, causal, sliding windows, segment ids, sinks, GQA, ``D`` 64 / 128 and
192 / 128); :func:`applies` says when a configuration runs here.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu import attention as _base

__all__ = ["Schedule", "applies", "forward", "schedule", "tile_items", "tile_order"]

Array = jax.Array

_L2_KV_BYTES = 32 * 2**20  # keys and values one section may keep in L2 (FA3, hopper/tile_scheduler.hpp)


def _loops(head_dim: int, dv: int, cfg) -> tuple[bool, int]:
    """(overlapped loop, consumer warpgroups) of ``attention._forward`` for these head dims."""
    small = head_dim == 64 and dv == 64
    overlap = (small and not cfg.has_segments) if cfg.fwd_overlap is None else cfg.fwd_overlap
    wgs = (3 if overlap and small else 2) if cfg.fwd_wgs is None else cfg.fwd_wgs
    return overlap, wgs


def applies(q: Array, k: Array, v: Array, cfg) -> bool:
    """Whether :func:`forward` runs this call: the kernel's default internal variants with longest-first tiles.

    The forward's experimental variants (mbarrier turns or slot releases in the overlapped loop, keys loaded
    ahead of values) and in-order tiles (``heavy_first=False``) keep the one-CTA-per-tile kernel.
    """
    head_dim, dv = q.shape[3], v.shape[3]
    if (head_dim, dv) not in ((64, 64), (128, 128), (192, 128)) or not cfg.heavy_first or cfg.fwd_k_ahead:
        return False
    overlap, _ = _loops(head_dim, dv, cfg)
    return not overlap or (cfg.fwd_named_turn and cfg.fwd_named_consumed)


@dataclasses.dataclass(frozen=True)
class Schedule:
    """Static deal of (batch, head, query tile) items to persistent CTAs (see the module docstring)."""

    ctas: int  # CTAs launched: one per SM (fewer when there are fewer items)
    section: int  # consecutive (batch, head) pairs per L2 section (the last section may hold fewer)
    padded: int  # item slots per section, a multiple of ``ctas``
    rounds: int  # item slots per CTA
    pairs: int  # batch * heads
    n_tiles: int  # query tiles per (batch, head) pair


def schedule(*, batch: int, heads: int, kv_heads: int, n_tiles: int, length: int, head_dim: int, dv: int,
             itemsize: int, num_sms: int) -> Schedule:
    """The deal for this problem: FA3's L2 section size (a power of two of key/value heads times the group),
    widened to at least one item per CTA, and sections padded to whole rounds."""
    pairs = batch * heads
    group = heads // kv_heads
    kv_head_bytes = length * (head_dim + dv) * itemsize
    fit = 1 if kv_head_bytes >= _L2_KV_BYTES else 1 << ((_L2_KV_BYTES // kv_head_bytes).bit_length() - 1)
    ctas = max(1, min(num_sms, pairs * n_tiles))
    per_round = -(-ctas // n_tiles)  # pairs that give every CTA an item in each round of a section
    wide = -(-per_round // group) * group  # in whole groups of query heads
    section = min(pairs, max(fit * group, wide))
    sections = -(-pairs // section)
    padded = -(-(section * n_tiles) // ctas) * ctas
    return Schedule(ctas=ctas, section=section, padded=padded, rounds=sections * padded // ctas, pairs=pairs,
                    n_tiles=n_tiles)


def tile_order(runs: Array, cfg, *, batch: int, length: int, n_tiles: int, rows: int) -> Array:
    """``[B, n_tiles]`` int32: each row's query tiles of ``rows`` rows by descending key-block count (ties: the
    later tile first), the count being the blocks its TMA warpgroup loads (``[lo_all, hi_all)`` of the kernel).

    Causal attention and windows: the last tile first (causal tiles grow with their index; with a window every
    tile past the first window has the same count to within one block). Segment ids: sorted on device from the
    run bounds (``runs[:, 0]``, the first position of the run around each token; trivial for rows whose ids are
    not one run each). Built from iotas only: ``custom_partitioning`` takes no captured constants.
    """
    return tile_items(runs, cfg, batch=batch, length=length, n_tiles=n_tiles, rows=rows)[:, 0]


def tile_items(runs: Array, cfg, *, batch: int, length: int, n_tiles: int, rows: int) -> Array:
    """``[B, 3, n_tiles]`` int32: by rank ``e`` (longest first, :func:`tile_order`), the tile, its ``lo_all`` and its
    ``hi_all`` (the key-block range its TMA warpgroup loads), so the kernel reads an item with three independent
    loads that it issues one item ahead."""
    bq, bkv = cfg.block_q, cfg.block_kv
    tile = jnp.arange(n_tiles, dtype=jnp.int32)
    tile_base = tile * rows
    last = jnp.minimum(tile_base + rows - bq, length - bq)  # first row of the tile's last warpgroup inside T
    hi = jnp.broadcast_to((last + bq + bkv - 1) // bkv, (batch, n_tiles))
    lo = jnp.zeros_like(tile) if cfg.window is None else jnp.maximum(tile_base - (cfg.window - 1), 0) // bkv
    if cfg.has_segments:
        lo = jnp.maximum(lo, runs[:, 0, jnp.minimum(tile_base, length - 1)] // bkv)
        order = jnp.argsort(-((hi - lo) * n_tiles + tile), axis=1).astype(jnp.int32)
    else:
        lo = jnp.broadcast_to(lo, (batch, n_tiles))
        order = jnp.broadcast_to(n_tiles - 1 - tile, (batch, n_tiles))
    by_rank = (jnp.take_along_axis(x, order, axis=1) for x in (lo, hi))
    return jnp.stack([order, *by_rank], axis=1)


def sm_count() -> int | None:
    """The GPU's SM count (``device.core_count``), or ``None`` when the device does not report one (the
    attention dispatch then keeps one CTA per query tile unless ``fwd_persistent=True``)."""
    count = getattr(jax.devices()[0], "core_count", None)
    return int(count) if count else None


def _num_sms() -> int:
    count = sm_count()
    if count is None:
        raise ValueError("the persistent attention forward needs the GPU's SM count (device.core_count); "
                         "fwd_persistent=False runs one CTA per query tile")
    return count


def forward(q, k, v, seg, runs, sink_init, cfg):
    """``(out, lse)`` of ``attention._forward(q, k, v, seg, runs, sink_init, cfg)``, bitwise, from one persistent
    CTA per SM (``T`` a multiple of 128, as there)."""
    batch, length, heads, head_dim = q.shape
    dv = v.shape[-1]
    extra = head_dim - dv
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    bq, bkv = cfg.block_q, cfg.block_kv
    mcs = min(cfg.max_concurrent_steps, length // bkv)
    c = cfg.scale * _base._LOG2E
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    col = plgpu.Layout.WGMMA.reduce(0)
    overlap, wgs = _loops(head_dim, dv, cfg)
    small = head_dim == 64 and dv == 64
    tq = wgs * bq
    if wgs not in (2, 3) or (wgs == 3 and (not overlap or not small)):
        raise ValueError(f"fwd_wgs={wgs}: 2, or 3 with fwd_overlap and D = 64; got D = {head_dim}")
    if length % _base._PAD_MULTIPLE:
        raise ValueError(f"T = {length} must be a multiple of {_base._PAD_MULTIPLE}")
    if not applies(q, k, v, cfg):
        raise ValueError("the persistent forward runs the kernel's default internal variants only")
    if overlap and _base._CONSUMED_BARRIER + 2 * mcs > 16:  # 16 named barriers exist
        raise ValueError(f"the overlapped forward releases slots on named barriers: at most 4 stages, got {mcs}")
    n_tiles = -(-length // tq)
    sched = schedule(batch=batch, heads=heads, kv_heads=kv_heads, n_tiles=n_tiles, length=length,
                     head_dim=head_dim, dv=dv, itemsize=jnp.dtype(dtype).itemsize, num_sms=_num_sms())
    # Packed documents: the items' tiles and key-block ranges by rank, from the segment runs. Otherwise the kernel
    # derives them (the last tile first) and reads no table.
    items = (tile_items(runs, cfg, batch=batch, length=length, n_tiles=n_tiles, rows=tq) if cfg.has_segments
             else jnp.zeros((1, 3, 1), jnp.int32))
    ctas, section, padded, pairs = sched.ctas, sched.section, sched.padded, sched.pairs
    # MLA's tiles leave no room for a separate output tile: the output goes through the query tile.
    own_out = extra == 0

    def kernel(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, items_ref, out_ref, lse_ref, scoped):
        cta = lax.axis_index("cta")
        wg = lax.axis_index("wg")
        smem, bars, consumed, schedule_bar = scoped
        if extra:
            qo_smem2, qx_smem2, k_smem, kx_smem, v_smem, lse_smem2, segq_smem2, segk_smem = smem
            o_smem2 = qo_smem2
        else:
            qo_smem2, k_smem, v_smem, o_smem2, lse_smem2, segq_smem2, segk_smem = smem
            qx_smem2 = kx_smem = None
        k_bars, v_bars, q_bars, segq_bars, segk_bars, q_free = bars
        k_consumed, v_consumed = consumed

        def item(r):
            """Round ``r`` of this CTA: (valid, batch, head, query tile, lo_all, hi_all)."""
            r = jnp.minimum(r, sched.rounds - 1)  # the look-ahead past the last round reads that round again
            pos = jnp.where(lax.rem(r, 2) == 0, cta, ctas - 1 - cta)
            slot = r * ctas + pos
            s = lax.div(slot, jnp.int32(padded))
            o = lax.rem(slot, jnp.int32(padded))
            in_section = jnp.minimum(jnp.int32(section), jnp.int32(pairs) - s * section)
            valid = o < in_section * n_tiles
            o = jnp.minimum(o, in_section * n_tiles - 1)  # padding slots read a real item (unused)
            pair = s * section + lax.rem(o, in_section)
            b = lax.div(pair, jnp.int32(heads))
            h = lax.rem(pair, jnp.int32(heads))
            rank = lax.div(o, in_section)
            if cfg.has_segments:  # three independent loads (tile_items)
                # "+ 0": inside jax.shard_map a value read from a ref keeps the caller's varying-axis type while
                # the kernel's arithmetic drops it, so a fori_loop that starts at a raw load fails JAX's carry type
                # check (its index comes back without the type); the add folds away in the compiled kernel.
                tile, lo_all, hi_all = (items_ref[b, i, rank] + 0 for i in range(3))
                return valid, b, h, tile, lo_all, hi_all
            tile = n_tiles - 1 - rank  # the last tile first (tile_order)
            lo_all, _ = _base._wg_kv_range(tile * tq, cfg)
            _, hi_all = _base._wg_kv_range(jnp.minimum(tile * tq + (wgs - 1) * bq, length - bq), cfg)
            return valid, b, h, tile, lo_all, hi_all

        def doc_lo(b, lo, q_first):
            # Keys before the run of the first query row cannot be seen by any row (runs are sorted).
            if not cfg.has_segments:
                return lo
            return jnp.maximum(lo, lax.div(runs_ref[b, 0, jnp.minimum(q_first, length - 1)], jnp.int32(bkv)))

        def tile_loop(body):
            """Run ``body(b, h, tile_base, lo_all, hi_all, first_step)`` for every item of this CTA, in order;
            ``first_step`` counts the key/value blocks of the CTA's earlier items (the ring position). With packed
            documents the next item's table loads are issued before this item runs, so their latency is hidden."""
            def round_body(r, carry):
                first_step, current = carry[0], carry[1:]
                ahead = item(r + 1) if cfg.has_segments else None
                valid, b, h, tile, lo_all, hi_all = current if cfg.has_segments else item(r)
                # The multiply stays in the body: a loop-carried row offset would lose its known multiple of the
                # tile, which the query copies' tiled shared-memory layout needs.
                tile_base = tile * tq
                pl.when(valid)(lambda: body(b, h, tile_base, lo_all, hi_all, first_step))
                first_step = first_step + jnp.where(valid, hi_all - lo_all, 0)
                return (first_step, *ahead) if cfg.has_segments else (first_step,)

            start = (jnp.int32(0), *item(jnp.int32(0))) if cfg.has_segments else (jnp.int32(0),)
            lax.fori_loop(0, sched.rounds, round_body, start)

        named_consumed = overlap  # applies(): the overlapped loop releases slots on named barriers

        def release(kind, slot):  # kind 0: key slot, 1: value slot
            if named_consumed:
                _base._named_barrier("arrive", _base._CONSUMED_BARRIER + kind * mcs + slot, 128 * (wgs + 1))
            else:
                plgpu.barrier_arrive((k_consumed, v_consumed)[kind].at[slot])

        def await_release(kind, slot):
            if named_consumed:
                _base._named_barrier("sync", _base._CONSUMED_BARRIER + kind * mcs + slot, 128 * (wgs + 1))
            else:
                plgpu.barrier_wait((k_consumed, v_consumed)[kind].at[slot])

        @pl.when(wg < wgs)
        def _compute():
            plgpu.set_max_registers(232 if wgs == 2 else 160, action="increase")
            qo_smem = qo_smem2.at[wg]
            qx_smem = qx_smem2.at[wg] if extra else None
            o_smem = o_smem2.at[wg]
            lse_smem = lse_smem2.at[wg]
            plgpu.barrier_arrive(q_free.at[wg])  # the query tile is free for the first item's queries

            def free_queries():  # the last QK^T product of the tile has read the query tile
                if own_out:
                    plgpu.barrier_arrive(q_free.at[wg])

            if overlap:  # the turns pass from one tile to the next: warpgroup 0 takes the first turn once
                pl.when(wg == wgs - 1)(lambda: _base._named_barrier("arrive", _base._TURN_BARRIER, 256))

            def consume(b, h, tile_base, lo_all, hi_all, first_step):
                q_base = tile_base + wg * bq
                lo, hi = _base._wg_kv_range(q_base, cfg)
                lo = doc_lo(b, lo, q_base)
                in_sequence = q_base < length
                shift = first_step - lo_all

                def slot_of(t):
                    return lax.rem(shift + t, jnp.int32(mcs))

                def wait_queries():
                    if cfg.has_segments:
                        plgpu.barrier_wait(segq_bars.at[wg])
                    plgpu.barrier_wait(q_bars.at[wg])

                if wgs == 2:  # 128-row tiles always lie inside the (padded) sequence
                    wait_queries()
                else:
                    pl.when(in_sequence)(wait_queries)
                seg_q = plgpu.load(segq_smem2.at[wg], layout=row) if cfg.has_segments else None

                if cfg.has_sinks:  # the sink is a virtual key every query sees: start from its logit
                    m_i = plgpu.load(sink_ref.at[0, h], layout=row, optimized=False)
                    l_i = plgpu.load(sink_ref.at[1, h], layout=row, optimized=False)
                else:
                    m_i = plgpu.layout_cast(jnp.full((bq,), _base._M_INIT, jnp.float32), row)
                    l_i = plgpu.layout_cast(jnp.full((bq,), 0, jnp.float32), row)
                acc = plgpu.layout_cast(jnp.full((bq, dv), 0, jnp.float32), layout)

                def finish(acc, m_i, l_i):
                    acc /= lax.broadcast_in_dim(l_i, (bq, dv), [0])
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous tile's stores read o and lse
                    o_smem[...] = acc.astype(dtype)
                    lse_smem[...] = m_i + jnp.log(l_i) * _base._LOG2E
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(o_smem, out_ref.at[b, pl.ds(q_base, bq), h])
                    plgpu.copy_smem_to_gmem(lse_smem, lse_ref.at[b, h, pl.ds(q_base, bq)])
                    if not own_out:  # the output went through the query tile: free it once the store read it
                        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                        plgpu.barrier_arrive(q_free.at[wg])

                def mask_scores(s, t, seg_k):
                    q_pos = _base._index_tile((bq, bkv), 0, q_base, layout)
                    k_pos = _base._index_tile((bq, bkv), 1, t * bkv, layout)
                    sq = sk = None
                    if cfg.has_segments:
                        sq = lax.broadcast_in_dim(seg_q, (bq, bkv), [0])
                        sk = lax.broadcast_in_dim(seg_k, (bq, bkv), [1])
                    return jnp.where(_base._tile_mask(q_pos, k_pos, sq, sk, cfg), s, -jnp.inf)

                def mask_bounds():
                    """(inside, full): blocks [inside, full) skip the mask (attention._forward's split)."""
                    full = lax.div(q_base, jnp.int32(bkv))  # blocks entirely in the causal past
                    if cfg.has_segments:  # blocks inside the last query's document skip the mask as well
                        inside = lax.div(runs_ref[b, 2, q_base + bq - 1] + bkv - 1, jnp.int32(bkv))
                    else:
                        inside = lo
                    if cfg.window is not None:  # blocks crossing the window's left edge stay masked
                        edge = lax.div(jnp.maximum(q_base + bq - cfg.window, 0) + bkv - 1, jnp.int32(bkv))
                        inside = jnp.maximum(inside, edge)
                    return jnp.clip(inside, lo, full), full

                if overlap:
                    def take_turn():
                        _base._named_barrier("sync", _base._TURN_BARRIER + wg, 256)

                    def pass_turn():
                        _base._named_barrier("arrive", _base._TURN_BARRIER + lax.rem(wg + 1, jnp.int32(wgs)), 256)

                    def scores(s_ref, slot):
                        plgpu.wgmma(s_ref, qo_smem, k_smem.at[slot].transpose((1, 0)))
                        if extra:
                            plgpu.wgmma(s_ref, qx_smem, kx_smem.at[slot].transpose((1, 0)))

                    def seg_keys(t, masked):
                        # Read before K(t) is released: the producer refills the ids with the next key tile.
                        if not cfg.has_segments:
                            return None
                        plgpu.barrier_wait(segk_bars.at[slot_of(t)])
                        return plgpu.load(segk_smem.at[slot_of(t)], layout=col) if masked else None

                    def softmax(s, m_i, l_i, t, seg_k, masked):
                        if masked:
                            s = mask_scores(s, t, seg_k)
                        m_ij = jnp.maximum(m_i, s.max(axis=1) * c)
                        alpha = jnp.exp2(m_i - m_ij)
                        p = jnp.exp2(s * c - lax.broadcast_in_dim(m_ij, s.shape, [0]))
                        l_i = l_i * alpha
                        l_i += p.sum(axis=1)
                        return m_ij, alpha, l_i, p.astype(dtype)

                    def first(t, m_i, l_i):  # S(lo) alone; the block is masked whatever it holds
                        slot = slot_of(t)
                        seg_k = seg_keys(t, True)
                        plgpu.barrier_wait(k_bars.at[slot])
                        take_turn()

                        def issue(s_ref):
                            scores(s_ref, slot)
                            pass_turn()
                            return s_ref[...]

                        s = pl.run_scoped(issue, plgpu.ACC((bq, bkv), jnp.float32))
                        release(0, slot)
                        m_i, _, l_i, p16 = softmax(s, m_i, l_i, t, seg_k, True)
                        return m_i, l_i, p16

                    def step(t, carry, masked: bool):
                        acc, m_i, l_i, p16 = carry
                        slot, prev = slot_of(t), slot_of(t - 1)
                        seg_k = seg_keys(t, masked)
                        plgpu.barrier_wait(k_bars.at[slot])
                        take_turn()

                        def issue(o_ref, s_ref):
                            o_ref[...] = acc
                            scores(s_ref, slot)
                            plgpu.barrier_wait(v_bars.at[prev])
                            plgpu.wgmma(o_ref, p16, v_smem.at[prev])
                            pass_turn()
                            s = plgpu.wgmma_accumulator_load(s_ref, wait_n=1)
                            release(0, slot)
                            return o_ref[...], softmax(s, m_i, l_i, t, seg_k, masked)

                        o, (m_i, alpha, l_i, p16) = pl.run_scoped(
                            issue, plgpu.ACC((bq, dv), jnp.float32), plgpu.ACC((bq, bkv), jnp.float32))
                        release(1, prev)
                        acc = o * lax.broadcast_in_dim(alpha, o.shape, [0])
                        return acc, m_i, l_i, p16

                    def last(t, acc, p16):  # O += P(hi - 1) V(hi - 1); every QK^T of the tile has completed
                        free_queries()
                        slot = slot_of(t)
                        take_turn()
                        plgpu.barrier_wait(v_bars.at[slot])

                        def issue(o_ref):
                            plgpu.wgmma(o_ref, p16, v_smem.at[slot])
                            pass_turn()

                        acc = pl.run_state(issue)(plgpu.ACC.init(acc))
                        release(1, slot)
                        return acc

                    def skip(t, carry):
                        # A block only other warpgroups see: wait for it (barrier phases stay in step, and a slot
                        # is released only after its data arrived) and release it within this warpgroup's turn.
                        slot = slot_of(t)
                        take_turn()
                        if cfg.has_segments:
                            plgpu.barrier_wait(segk_bars.at[slot])
                        plgpu.barrier_wait(k_bars.at[slot])
                        release(0, slot)
                        plgpu.barrier_wait(v_bars.at[slot])
                        release(1, slot)
                        pass_turn()
                        return carry

                    def run_tile():
                        inside, full = mask_bounds()
                        lax.fori_loop(lo_all, lo, skip, None)
                        m0, l0, p16 = first(lo, m_i, l_i)
                        carry = (acc, m0, l0, p16)
                        nxt = lo + 1
                        carry = lax.fori_loop(nxt, inside, functools.partial(step, masked=True), carry)
                        carry = lax.fori_loop(jnp.maximum(inside, nxt), full, functools.partial(step, masked=False),
                                              carry)
                        carry = lax.fori_loop(jnp.maximum(full, nxt), hi, functools.partial(step, masked=True), carry)
                        acc_t, m_t, l_t, p16 = carry
                        acc_t = last(hi - 1, acc_t, p16)
                        lax.fori_loop(hi, hi_all, skip, None)
                        finish(acc_t, m_t, l_t)

                    def idle_tile():  # rows past the sequence (three warpgroups, last tile): take the turns only
                        lax.fori_loop(lo_all, hi_all, skip, None)
                        take_turn()
                        pass_turn()
                        plgpu.barrier_arrive(q_free.at[wg])  # no queries were loaded for these rows

                    if wgs == 2:
                        run_tile()
                    else:
                        pl.when(in_sequence)(run_tile)
                        pl.when(jnp.logical_not(in_sequence))(idle_tile)
                    return

                # The ping-pong loop of attention._forward (one shared schedule barrier, S and PV in separate
                # turns), with the waits of blocks skipped after this warpgroup's range made explicit: every
                # warpgroup observes every phase of every slot barrier, as the next tile needs.
                def sync():
                    plgpu.barrier_arrive(schedule_bar)
                    plgpu.barrier_wait(schedule_bar)

                plgpu.barrier_wait(k_bars.at[slot_of(lo_all)])
                pl.when(wg == 1)(sync)

                def skip_before(t, carry):
                    # A block the other warpgroup needs: keep barrier phases and the schedule in step.
                    slot = slot_of(t)
                    if cfg.has_segments:
                        plgpu.barrier_wait(segk_bars.at[slot])
                    plgpu.barrier_arrive(k_consumed.at[slot])
                    sync()
                    plgpu.barrier_wait(v_bars.at[slot])
                    sync()
                    plgpu.barrier_arrive(v_consumed.at[slot])
                    plgpu.barrier_wait(k_bars.at[slot_of(t + 1)])
                    return carry

                def kv_step(t, carry, masked: bool):
                    acc, m_i, l_i = carry
                    slot = slot_of(t)
                    if cfg.has_segments:
                        plgpu.barrier_wait(segk_bars.at[slot])
                        seg_k = plgpu.load(segk_smem.at[slot], layout=col)
                    else:
                        seg_k = None

                    def compute_qk(acc_ref):
                        plgpu.wgmma(acc_ref, qo_smem, k_smem.at[slot].transpose((1, 0)))
                        if extra:
                            plgpu.wgmma(acc_ref, qx_smem, kx_smem.at[slot].transpose((1, 0)))
                        sync()
                        return acc_ref[...]

                    qk = pl.run_scoped(compute_qk, plgpu.ACC((bq, bkv), jnp.float32))
                    plgpu.barrier_arrive(k_consumed.at[slot])
                    if own_out:
                        pl.when(t == hi - 1)(free_queries)
                    if masked:
                        qk = mask_scores(qk, t, seg_k)

                    m_ij = jnp.maximum(m_i, qk.max(axis=1) * c)
                    alpha = jnp.exp2(m_i - m_ij)
                    m_i = m_ij
                    p = jnp.exp2(qk * c - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                    acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                    l_i *= alpha
                    p16 = p.astype(dtype)
                    l_i += p.sum(axis=1)
                    acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                    plgpu.barrier_arrive(schedule_bar)
                    plgpu.barrier_wait(v_bars.at[slot])
                    plgpu.barrier_wait(schedule_bar)

                    def compute_pv(acc_ref):
                        plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                        nxt = t + 1

                        @pl.when(nxt < hi)
                        def _():
                            plgpu.barrier_wait(k_bars.at[slot_of(nxt)])

                    acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                    plgpu.barrier_arrive(v_consumed.at[slot])
                    return acc, m_i, l_i

                def skip_after(t, carry):
                    slot = slot_of(t)
                    if cfg.has_segments:
                        plgpu.barrier_wait(segk_bars.at[slot])
                    plgpu.barrier_wait(k_bars.at[slot])
                    plgpu.barrier_arrive(k_consumed.at[slot])
                    sync()
                    plgpu.barrier_wait(v_bars.at[slot])
                    plgpu.barrier_arrive(v_consumed.at[slot])
                    sync()
                    return carry

                lax.fori_loop(lo_all, lo, skip_before, None)
                inside, full = mask_bounds()
                carry = (acc, m_i, l_i)
                carry = lax.fori_loop(lo, inside, functools.partial(kv_step, masked=True), carry)
                carry = lax.fori_loop(inside, full, functools.partial(kv_step, masked=False), carry)
                carry = lax.fori_loop(full, hi, functools.partial(kv_step, masked=True), carry)
                acc, m_i, l_i = carry
                lax.fori_loop(hi, hi_all, skip_after, None)
                pl.when(wg == 0)(sync)
                finish(acc, m_i, l_i)

            tile_loop(consume)
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == wgs)
        def _memory():
            # 3 x 160 + 32 = 4 x 128 registers per thread: the item list's index arithmetic needs more than 24.
            plgpu.set_max_registers(40 if wgs == 2 else 32, action="decrease")

            def load_queries(b, h, tile_base):
                for w in range(wgs):
                    q_base = tile_base + w * bq

                    def copies(w=w, q_base=q_base):
                        q_rows = (b, pl.ds(q_base, bq), h)
                        if extra:  # two copies arrive on the query barrier (num_arrivals=2)
                            plgpu.copy_gmem_to_smem(q_ref.at[(*q_rows, pl.ds(0, dv))], qo_smem2.at[w], q_bars.at[w])
                            plgpu.copy_gmem_to_smem(q_ref.at[(*q_rows, pl.ds(dv, extra))], qx_smem2.at[w],
                                                    q_bars.at[w])
                        else:
                            plgpu.copy_gmem_to_smem(q_ref.at[q_rows], qo_smem2.at[w], q_bars.at[w])
                        if cfg.has_segments:
                            plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(q_base, bq)], segq_smem2.at[w],
                                                    segq_bars.at[w])

                    plgpu.barrier_wait(q_free.at[w])  # every warpgroup frees its query tile once per item
                    if wgs == 2:
                        copies()
                    else:
                        pl.when(q_base < length)(copies)

            def produce(b, h, tile_base, lo_all, hi_all, first_step):
                kvh = lax.div(h, jnp.int32(group))

                def body(t, carry):
                    ring = first_step + t - lo_all
                    slot = lax.rem(ring, jnp.int32(mcs))
                    s = (b, pl.ds(t * bkv, bkv), kvh)
                    refill = ring >= mcs
                    pl.when(refill)(lambda: await_release(0, slot))
                    if extra:  # two copies arrive on the key barrier (num_arrivals=2)
                        plgpu.copy_gmem_to_smem(k_ref.at[(*s, pl.ds(0, dv))], k_smem.at[slot], k_bars.at[slot])
                        plgpu.copy_gmem_to_smem(k_ref.at[(*s, pl.ds(dv, extra))], kx_smem.at[slot], k_bars.at[slot])
                    else:
                        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[slot], k_bars.at[slot])
                    if cfg.has_segments:
                        plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(t * bkv, bkv)], segk_smem.at[slot],
                                                segk_bars.at[slot])
                    pl.when(refill)(lambda: await_release(1, slot))
                    plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[slot], v_bars.at[slot])
                    # The tile's first key/value block is on its way: now the queries (FA3's order).
                    pl.when(t == lo_all)(lambda: load_queries(b, h, tile_base))
                    return carry

                lax.fori_loop(lo_all, hi_all, body, None)

            tile_loop(produce)

    def entry(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, items_ref, out_ref, lse_ref):
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)

        def tiles(*shape):
            return plgpu.SMEM(shape, dtype, transforms=(tiling, swizzle))

        if extra:  # query and key tiles split into their first dv channels and the extra ones
            qk_tiles = (tiles(wgs, bq, dv), tiles(wgs, bq, extra), tiles(mcs, bkv, dv), tiles(mcs, bkv, extra))
            v_out = (tiles(mcs, bkv, dv),)
        else:
            qk_tiles = (tiles(wgs, bq, head_dim), tiles(mcs, bkv, head_dim))
            v_out = (tiles(mcs, bkv, dv), tiles(wgs, bq, dv))
        smem = (
            *qk_tiles,
            *v_out,
            plgpu.SMEM((wgs, bq), jnp.float32),
            plgpu.SMEM((wgs, bq), jnp.int32),
            plgpu.SMEM((mcs, bkv), jnp.int32),
        )
        copies = 2 if extra else 1  # TMA copies per query and per key tile
        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, items_ref, out_ref, lse_ref,
                                 args),
            smem,
            (
                plgpu.Barrier(num_arrivals=copies, num_barriers=mcs),
                plgpu.Barrier(num_barriers=mcs),
                plgpu.Barrier(num_arrivals=copies, num_barriers=wgs),
                plgpu.Barrier(num_barriers=wgs),
                plgpu.Barrier(num_barriers=mcs),
                plgpu.Barrier(num_barriers=wgs),  # q_free: a consumer warpgroup's query tile may be refilled
            ),
            (plgpu.Barrier(num_arrivals=wgs, num_barriers=mcs),) * 2,
            plgpu.Barrier(num_arrivals=2),  # the ping-pong loop's schedule barrier
            collective_axes="wg",
        )

    out, lse = plgpu.kernel(
        entry,
        out_type=[
            jax.ShapeDtypeStruct((batch, length, heads, dv), dtype),
            jax.ShapeDtypeStruct((batch, heads, length), jnp.float32),
        ],
        grid=(ctas,),
        grid_names=("cta",),
        num_threads=wgs + 1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, seg, runs, sink_init, items)
    return out, lse
