# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact radix-select top-k of score rows on Hopper (Pallas Mosaic GPU), for the DSA indexer.

:func:`radix_top_k` returns, for every row of ``scores [B, R, W]`` (float32; ``-inf`` marks keys that
cannot be selected), its ``k`` largest finite entries with ties broken toward the lower column: the set
that DeepSeek-V3.2's reference (``index_score.topk(...)`` in torch) and ``lax.top_k`` select, as column
indices (in two runs of ascending columns, see below), ``-1`` in the slots a row cannot fill (fewer
than ``k`` finite entries), with their scores (``-inf`` there). ``+inf`` and NaN entries count as ``-inf``:
they are never selected (``dsa_select`` and the sort path instead let them take a slot, which comes back
``-1``; the selections agree whenever every score is finite or ``-inf``). Besides the scores it needs
``O(rows x (k + capacity))`` memory, ``capacity = max(2048, W / 16)`` slots per row by default (so the
candidate buffer grows with the row width ``W`` beyond 32K columns); ``lax.top_k`` (a segmented radix sort
of each row here) needs scratch of several times the score block.

Algorithm: a most-significant-digit radix select on the order-preserving 32-bit integer key of each
float (``-0.0`` keyed as ``+0.0``), 4 bits per pass, 8 passes. Pass ``p`` counts per row the entries
whose key, shifted right by ``28 - 4 p``, falls in each of the 16 bins of the bucket that holds the
``k``-th largest entry, or above it (one Mosaic GPU kernel per pass: a CTA per 64 rows and column split
streams ``[64, 128]`` tiles from global memory; each entry adds one to an 8-bit field of one of five
packed counters, which are summed along the rows per tile); XLA adds the splits and picks the next
digit. After 3 passes (a 12-bit key prefix) a compaction kernel writes the entries above the prefix
(all of them selected, fewer than ``k``) to the output and the entries that share it (the candidates,
typically a few percent of a row) to a buffer of ``capacity`` slots per row, so that passes 4 to 8
read only the candidates (if any row has more than ``capacity``, they read the full rows instead).
After 8 passes the key of the ``k``-th largest entry is exact, and the per-split counts of the last
pass say how many entries above it and equal to it each split holds. The selection kernel then
writes every candidate above that key and, in column order, the first ``k - (count above)`` equal to
it. In both writing kernels the in-tile exclusive prefix counts of the flagged entries are WGMMAs
against a strictly upper-triangular ones matrix (exact in float32), and each thread stores its
flagged entries at their slots (``plgpu.inline_mgpu``). Each row thus lists the entries above the
prefix in ascending columns, then the selected candidates in ascending columns. With ``offset``
(causal score blocks, as the DSA indexer writes them: query row ``r`` sees keys ``< offset + r + 1``),
only the tiles on or before each 64-row group's diagonal are read.
"""

from __future__ import annotations

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith, memref, vector

from jla.ops.mosaic_gpu._common import ROWS, WGMMA, bcast_rows, compiler_params, iota, matmul, num_sms, smem

__all__ = ["radix_top_k", "score_key"]

Array = jax.Array

_TR = 64  # rows per CTA (one warpgroup, the WGMMA layout's 64 rows)
_TC = 128  # columns per tile
_BOUNDS = 17  # boundaries base + 0 ... base + 16 of a 16-way bucket
_PASSES = 8  # 4 bits per pass
_COMPACT_AFTER = 3  # passes over the full rows before the candidates (equal 12-bit key prefix) are compacted
_CAPACITY = 2048  # least candidates kept per row (more in any row: every pass runs over the full rows)
_INT_MAX = 2**31 - 1


def score_key(x: Array) -> Array:
    """Order-preserving int32 key of float32 ``x`` (signed comparison; ``-0.0`` and ``+0.0`` share key 0).

    Works in XLA and inside the kernels; NaN keys sort above ``+inf`` (positive NaN) or below ``-inf``.
    """
    bits = lax.bitcast_convert_type(x, jnp.int32)
    key = jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)
    return jnp.where(x == 0.0, jnp.int32(0), key)


def _selectable(x: Array) -> Array:
    """``x`` with ``+inf`` and NaN replaced by ``-inf`` (never selected), applied to every score tile a kernel reads."""
    return jnp.where(x < jnp.inf, x, -jnp.inf)


def _visible_tiles(r0, offset: int | None, n_tiles: int):
    """Column tiles that hold keys visible to the rows ``r0 ... r0 + 63`` (keys ``< offset + r0 + 64``)."""
    if offset is None:
        return jnp.int32(n_tiles)
    return lax.min(lax.div(r0 + (offset + _TR + _TC - 1), jnp.int32(_TC)), jnp.int32(n_tiles))


def _count_call(scores: Array, base: Array, *, shift: int, offset: int | None, splits: int, finite: bool) -> Array:
    """Per-split histograms ``[B, splits, 17 (+1), R]``: entry ``t`` counts the entries whose key ``>> shift`` equals
    ``base + t`` (``t = 0 ... 15``), entry 16 those above ``base + 15``; with ``finite`` entry 17 counts the entries
    ``> -inf``.

    Per ``[64, 128]`` tile each entry's bin ``t`` adds ``1 << 8 (t % 4)`` to packed counter ``t // 4`` (four 8-bit
    fields; a row of a tile adds at most 128 to a field), so a tile costs five row sums instead of seventeen.
    The next tile's load is issued before the current tile's arithmetic.
    """
    batch, rows, width = scores.shape
    n_tiles = width // _TC
    per = -(-n_tiles // splits)
    n_out = _BOUNDS + int(finite)
    shape = (_TR, _TC)

    def kernel(s_ref, base_ref, out_ref, out_smem):
        rt = lax.axis_index("rt")
        sp = lax.axis_index("sp")
        b = lax.axis_index("batch")
        r0 = rt * _TR
        lo = sp * per
        hi = lax.min(lo + per, _visible_tiles(r0, offset, n_tiles))
        base_v = plgpu.load(base_ref.at[b, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        bb = bcast_rows(base_v, shape)

        def load(j):
            j = lax.min(j, jnp.int32(n_tiles - 1))  # a valid tile (the prefetch past the end is not used)
            return _selectable(plgpu.load(s_ref.at[b, pl.ds(r0, _TR), pl.ds(j * _TC, _TC)], layout=WGMMA, optimized=False))

        def tile(j, carry):
            x, hist = carry
            x_next = load(j + 1)
            v = score_key(x)
            if shift:
                v = lax.shift_right_arithmetic(v, jnp.int32(shift))
            # bin in the bucket: -1 below it, 16 above it (bb + 15 cannot overflow: bb <= 2**31 - 16)
            t = jnp.where(v < bb, jnp.int32(-1), jnp.where(v > bb + 15, jnp.int32(16), v - bb))
            one = lax.shift_left(jnp.int32(1), lax.shift_left(t & 3, jnp.int32(3)))
            group = lax.shift_right_arithmetic(t, jnp.int32(2))  # -1 below the bucket: no group
            new = list(hist)
            for g in range(5):
                packed = jnp.sum(jnp.where(group == g, one, jnp.int32(0)), axis=1)
                for f in range(4 if g < 4 else 1):
                    field = lax.shift_right_logical(packed, jnp.int32(8 * f)) & 255 if f else packed & 255
                    new[4 * g + f] = new[4 * g + f] + field
            if finite:
                new[_BOUNDS] = new[_BOUNDS] + jnp.sum(jnp.where(x > -jnp.inf, jnp.int32(1), jnp.int32(0)), axis=1)
            return x_next, tuple(new)

        zero = plgpu.layout_cast(jnp.full((_TR,), 0, jnp.int32), ROWS)
        _, hist = lax.fori_loop(lo, hi, tile, (load(lo), (zero,) * n_out))
        for i, h in enumerate(hist):
            out_smem.at[i][...] = h
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out_smem, out_ref.at[b, sp, :, pl.ds(r0, _TR)])
        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((batch, splits, n_out, rows), jnp.int32),
        scratch_types=[plgpu.SMEM((n_out, _TR), jnp.int32)],
        compiler_params=compiler_params(),
        grid=(rows // _TR, splits, batch),
        grid_names=("rt", "sp", "batch"),
    )(scores, base)


def _scatter(keep, pos, x, rows, cols, b, idx_ref, val_ref) -> None:
    """``idx_ref[b, rows, pos] = cols`` and ``val_ref[b, rows, pos] = x`` where ``keep != 0`` (each thread its own
    elements of the WGMMA-layout arrays)."""

    @plgpu.inline_mgpu(arg_types=(WGMMA, WGMMA, WGMMA, WGMMA, WGMMA, plgpu.Layout.WG_SPLAT, plgpu.RefType(),
                                  plgpu.RefType()))
    def store(ctx, keep, pos, x, rows, cols, b, idx_ref, val_ref):
        del ctx
        index = ir.IndexType.get()
        i32 = ir.IntegerType.get_signless(32)
        zero = arith.constant(i32, 0)
        b_i = arith.index_cast(index, b.registers.item())
        arrays = (keep, pos, x, rows, cols)
        for reg_idx in np.ndindex(keep.registers.shape):
            regs = [a.registers[reg_idx] for a in arrays]
            if isinstance(regs[0].type, ir.VectorType):
                n = ir.VectorType(regs[0].type).shape[0]
                elems = [[vector.extract(r, dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([e]))
                          for r in regs] for e in range(n)]
            else:
                elems = [regs]
            for kv, pv, xv, rv, cv in elems:
                with mgpu.utils.when(arith.cmpi(arith.CmpIPredicate.ne, kv, zero)):
                    where = [b_i, arith.index_cast(index, rv), arith.index_cast(index, pv)]
                    memref.store(cv, idx_ref, where)
                    memref.store(xv, val_ref, where)

    store(keep, pos, x, rows, cols, b, idx_ref, val_ref)


def _select_call(scores: Array, star: Array, take: Array, pos0: Array, *, k: int, offset: int | None,
                 splits: int) -> tuple[Array, Array]:
    """Columns and scores ``[B, R, k]`` of the entries with key ``> star`` and the first ``take`` (per split) with key
    ``== star``, at slots ``pos0`` (per split) onward in column order (slots no entry fills are not written)."""
    batch, rows, width = scores.shape
    n_tiles = width // _TC
    per = -(-n_tiles // splits)
    shape = (_TR, _TC)

    def kernel(s_ref, star_ref, take_ref, pos_ref, idx_ref, val_ref, u_smem):
        rt = lax.axis_index("rt")
        sp = lax.axis_index("sp")
        b = lax.axis_index("batch")
        r0 = rt * _TR
        lo = sp * per
        hi = lax.min(lo + per, _visible_tiles(r0, offset, n_tiles))
        # strictly upper-triangular ones: (F @ U)[r, c] = number of flags before column c in row r
        u_smem[...] = jnp.where(iota((_TC, _TC), 0) < iota((_TC, _TC), 1), 1.0, 0.0).astype(jnp.float16)
        plgpu.commit_smem()
        star_v = plgpu.load(star_ref.at[b, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        take_v = plgpu.load(take_ref.at[b, sp, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        pos_v = plgpu.load(pos_ref.at[b, sp, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        st = bcast_rows(star_v, shape)
        tk = bcast_rows(take_v, shape)
        row_ids = iota(shape, 0) + r0
        col_ids = iota(shape, 1)

        def load(j):
            j = lax.min(j, jnp.int32(n_tiles - 1))  # a valid tile (the prefetch past the end is not used)
            return _selectable(plgpu.load(s_ref.at[b, pl.ds(r0, _TR), pl.ds(j * _TC, _TC)], layout=WGMMA, optimized=False))

        def tile(j, carry):
            x, pos_v, ties_v = carry
            x_next = load(j + 1)
            key = score_key(x)
            gt = key > st
            eq = key == st
            before_eq = matmul(jnp.where(eq, 1.0, 0.0).astype(jnp.float16), u_smem, shape)
            sel = gt | (eq & (bcast_rows(ties_v, shape) + before_eq.astype(jnp.int32) < tk))
            before_sel = matmul(jnp.where(sel, 1.0, 0.0).astype(jnp.float16), u_smem, shape)
            pos = bcast_rows(pos_v, shape) + before_sel.astype(jnp.int32)
            keep = jnp.where(sel & (pos < k), jnp.int32(1), jnp.int32(0))
            _scatter(keep, pos, x, row_ids, col_ids + j * _TC, b, idx_ref, val_ref)
            n_sel = jnp.sum(jnp.where(sel, jnp.int32(1), jnp.int32(0)), axis=1)
            n_eq = jnp.sum(jnp.where(eq, jnp.int32(1), jnp.int32(0)), axis=1)
            return x_next, pos_v + n_sel, ties_v + n_eq

        zero = plgpu.layout_cast(jnp.full((_TR,), 0, jnp.int32), ROWS)
        lax.fori_loop(lo, hi, tile, (load(lo), pos_v, zero))

    return plgpu.kernel(
        kernel,
        out_type=[jax.ShapeDtypeStruct((batch, rows, k), jnp.int32),
                  jax.ShapeDtypeStruct((batch, rows, k), jnp.float32)],
        scratch_types=[smem((_TC, _TC), jnp.float16)],
        compiler_params=compiler_params(),
        grid=(rows // _TR, splits, batch),
        grid_names=("rt", "sp", "batch"),
    )(scores, star, take, pos0)


def _compact_call(scores: Array, prefix: Array, pos_above: Array, pos_cand: Array, *, shift: int, k: int,
                  capacity: int, offset: int | None, splits: int) -> tuple[Array, Array, Array, Array]:
    """One pass that splits each row by the key prefix ``key >> shift`` against ``prefix [B, R]``: entries above it
    (all selected) go to ``[B, R, k]`` columns/scores at slots ``pos_above`` (per split) onward, entries equal to it
    (the candidates) to ``[B, R, capacity]`` columns/scores at slots ``pos_cand`` onward, both in column order
    (slots past ``k`` or ``capacity`` are dropped; slots no entry fills are not written)."""
    batch, rows, width = scores.shape
    n_tiles = width // _TC
    per = -(-n_tiles // splits)
    shape = (_TR, _TC)

    def kernel(s_ref, pre_ref, pa_ref, pc_ref, idx_ref, val_ref, cidx_ref, cval_ref, u_smem):
        rt = lax.axis_index("rt")
        sp = lax.axis_index("sp")
        b = lax.axis_index("batch")
        r0 = rt * _TR
        lo = sp * per
        hi = lax.min(lo + per, _visible_tiles(r0, offset, n_tiles))
        u_smem[...] = jnp.where(iota((_TC, _TC), 0) < iota((_TC, _TC), 1), 1.0, 0.0).astype(jnp.float16)
        plgpu.commit_smem()
        pre = bcast_rows(plgpu.load(pre_ref.at[b, pl.ds(r0, _TR)], layout=ROWS, optimized=False), shape)
        pa_v = plgpu.load(pa_ref.at[b, sp, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        pc_v = plgpu.load(pc_ref.at[b, sp, pl.ds(r0, _TR)], layout=ROWS, optimized=False)
        row_ids = iota(shape, 0) + r0
        col_ids = iota(shape, 1)

        def load(j):
            j = lax.min(j, jnp.int32(n_tiles - 1))
            return _selectable(plgpu.load(s_ref.at[b, pl.ds(r0, _TR), pl.ds(j * _TC, _TC)], layout=WGMMA, optimized=False))

        def tile(j, carry):
            x, pa_v, pc_v = carry
            x_next = load(j + 1)
            v = lax.shift_right_arithmetic(score_key(x), jnp.int32(shift))
            cols = col_ids + j * _TC
            counts = []
            for flags, pos_v, cap, refs in ((v > pre, pa_v, k, (idx_ref, val_ref)),
                                            (v == pre, pc_v, capacity, (cidx_ref, cval_ref))):
                before = matmul(jnp.where(flags, 1.0, 0.0).astype(jnp.float16), u_smem, shape)
                pos = bcast_rows(pos_v, shape) + before.astype(jnp.int32)
                keep = jnp.where(flags & (pos < cap), jnp.int32(1), jnp.int32(0))
                _scatter(keep, pos, x, row_ids, cols, b, *refs)
                counts.append(pos_v + jnp.sum(jnp.where(flags, jnp.int32(1), jnp.int32(0)), axis=1))
            return x_next, counts[0], counts[1]

        lax.fori_loop(lo, hi, tile, (load(lo), pa_v, pc_v))

    return plgpu.kernel(
        kernel,
        out_type=[jax.ShapeDtypeStruct((batch, rows, k), jnp.int32),
                  jax.ShapeDtypeStruct((batch, rows, k), jnp.float32),
                  jax.ShapeDtypeStruct((batch, rows, capacity), jnp.int32),
                  jax.ShapeDtypeStruct((batch, rows, capacity), jnp.float32)],
        scratch_types=[smem((_TC, _TC), jnp.float16)],
        compiler_params=compiler_params(),
        grid=(rows // _TR, splits, batch),
        grid_names=("rt", "sp", "batch"),
    )(scores, prefix, pos_above, pos_cand)


def _default_splits(batch: int, rows: int, n_tiles: int) -> int:
    """Column splits so that the grid covers about four CTAs per SM."""
    ctas = batch * (rows // _TR)
    return int(max(1, min(n_tiles, -(-4 * num_sms() // ctas))))


def _at_least(hist: Array) -> Array:
    """Counts ``c_i = sum_{t >= i} hist_t`` (``i = 0 ... 16``) of per-bin histograms ``[..., 17 (+1), R]``."""
    bins = hist[..., :_BOUNDS, :]
    return jnp.flip(jnp.cumsum(jnp.flip(bins, axis=-2), axis=-2), axis=-2)


def _digit(total: Array, kk: Array) -> Array:
    """The largest ``i <= 15`` with ``total[:, i] >= kk`` (``total [B, 17+, R]`` non-increasing in ``i``)."""
    return jnp.sum((total[:, 1:16] >= kk[:, None]).astype(jnp.int32), axis=1)


def _finish(scores: Array, value: Array, kk: Array, start: Array, *, k: int, count: dict) -> tuple[Array, Array]:
    """Passes ``_COMPACT_AFTER ... 7`` from the key prefix ``value`` of the ``kk``-th largest entry, then the selection
    kernel, its slots starting at ``start [B, R]`` (raw buffers: slots no entry fills are not written)."""
    counts = digit = None
    for p in range(_COMPACT_AFTER, _PASSES):
        base = value * 16
        counts = _at_least(_count_call(scores, base, shift=28 - 4 * p, finite=False, **count))  # per split
        digit = _digit(counts.sum(axis=1), kk)
        value = base + digit
    # value is the key of the kk-th largest entry; split s holds `above` entries above it and `ties` equal to it
    above, at_least = _split_counts(counts, digit)
    ties = at_least - above
    need = kk - above.sum(axis=1)
    take = jnp.clip(need[:, None] - (jnp.cumsum(ties, axis=1) - ties), 0, ties)
    chosen = above + take
    pos0 = jnp.cumsum(chosen, axis=1) - chosen + start[:, None]
    star = jnp.where(kk > 0, value, jnp.int32(_INT_MAX))
    take = jnp.where(kk[:, None] > 0, take, 0)
    idx, val = _select_call(scores, star, take.astype(jnp.int32), pos0.astype(jnp.int32), k=k, **count)
    return idx, val


def _split_counts(counts: Array, digit: Array) -> tuple[Array, Array]:
    """Per split ``[B, S, R]``: entries above the digit's bin and entries at or above it (``counts [B, S, 17, R]``
    from :func:`_at_least`)."""
    at = lambda i: jnp.take_along_axis(counts, i[:, None, None, :], axis=2)[:, :, 0]  # noqa: E731
    return at(digit + 1), at(digit)


def radix_top_k(scores: Array, k: int, *, offset: int | None = None, splits: int | None = None,
                capacity: int | None = None) -> tuple[Array, Array]:
    """The ``k`` largest finite entries of each row of ``scores [B, R, W]`` (float32): ``(indices, values)``,
    each ``[B, R, k]``.

    Ties at the ``k``-th value are broken toward the lower column (as ``torch.topk`` and ``lax.top_k``
    choose). ``+inf`` and NaN entries are treated as ``-inf`` (never selected). Each row lists its entries in two runs of ascending columns (those above the ``k``-th
    value's 12-bit key prefix, then the rest), followed by ``-1`` (values ``-inf``) in the slots a row
    cannot fill. ``R`` must be a multiple of 64 and ``W`` of 128. With ``offset``, row ``r`` only sees
    columns ``< offset + r + 1`` (a causal block of the DSA indexer): the kernels skip the tiles after
    each 64-row group's diagonal, which must hold ``-inf`` or are ignored. ``capacity`` (a multiple of
    128; default ``max(2048, W / 16)``): candidates per row kept after the first three passes; when a row
    has more, the remaining passes run over the full rows instead (same result). No gradient.
    """
    if scores.ndim != 3 or scores.dtype != jnp.float32:
        raise ValueError(f"radix_top_k needs float32 scores [B, R, W]; got {scores.dtype} {scores.shape}")
    batch, rows, width = scores.shape
    if rows % _TR or width % _TC:
        raise ValueError(f"radix_top_k needs R % {_TR} == 0 and W % {_TC} == 0; got R = {rows}, W = {width}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if capacity is None:  # the candidates are ~1% of a row on average, up to ~4% in the worst rows measured
        capacity = max(_CAPACITY, -(-width // 16 // _TC) * _TC)
    if capacity < _TC or capacity % _TC:
        raise ValueError(f"capacity must be a positive multiple of {_TC}; got {capacity}")
    n_tiles = width // _TC
    splits = _default_splits(batch, rows, n_tiles) if splits is None else int(splits)
    if not 1 <= splits <= n_tiles:
        raise ValueError(f"splits must be in [1, {n_tiles}]; got {splits}")
    scores = lax.stop_gradient(scores)
    count = dict(offset=offset, splits=splits)

    base = jnp.full((batch, rows), -8, jnp.int32)  # pass 0: key >> 28 lies in [-8, 7]
    hist = _count_call(scores, base, shift=28, finite=True, **count).sum(axis=1)
    kk = jnp.minimum(jnp.int32(k), hist[:, _BOUNDS])  # finite entries are the candidates
    digit = _digit(_at_least(hist), kk)
    value = base + digit
    for p in range(1, _COMPACT_AFTER):
        base = value * 16
        counts = _at_least(_count_call(scores, base, shift=28 - 4 * p, finite=False, **count))  # per split
        digit = _digit(counts.sum(axis=1), kk)
        value = base + digit
    # value: the 12-bit key prefix of the kk-th largest entry; per split, entries with a larger prefix and equal ones
    above, at_least = _split_counts(counts, digit)
    cand = at_least - above
    n_above, n_cand = above.sum(axis=1), cand.sum(axis=1)

    def compact(_):
        shift = 32 - 4 * _COMPACT_AFTER
        idx_a, val_a, cidx, cval = _compact_call(scores, value, (jnp.cumsum(above, axis=1) - above).astype(jnp.int32),
                                                 (jnp.cumsum(cand, axis=1) - cand).astype(jnp.int32), shift=shift, k=k,
                                                 capacity=capacity, **count)
        live = jnp.arange(capacity, dtype=jnp.int32)[None, None, :] < n_cand[:, :, None]
        cval = jnp.where(live, cval, -jnp.inf)  # the candidates' scores, -inf in empty slots
        cidx = jnp.where(live, cidx, 0)
        cand_count = dict(offset=None, splits=_default_splits(batch, rows, capacity // _TC))
        slot, val_c = _finish(cval, value, kk - n_above, n_above, k=k, count=cand_count)
        idx_c = jnp.take_along_axis(cidx, jnp.clip(slot, 0, capacity - 1), axis=2)
        first = jnp.arange(k, dtype=jnp.int32)[None, None, :] < n_above[:, :, None]
        return jnp.where(first, idx_a, idx_c), jnp.where(first, val_a, val_c)

    def full(_):
        return _finish(scores, value, kk, jnp.zeros_like(kk), k=k, count=count)

    idx, val = lax.cond(jnp.max(n_cand) <= capacity, compact, full, None)
    filled = jnp.arange(k, dtype=jnp.int32)[None, None, :] < kk[:, :, None]
    return jnp.where(filled, idx, -1), jnp.where(filled, val, -jnp.inf)
