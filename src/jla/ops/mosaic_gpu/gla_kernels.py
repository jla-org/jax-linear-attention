# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gla/chunk.py, fla/ops/common/chunk_h.py
"""Dedicated Mosaic GPU (H100) pipeline for GLA: ``S_t = diag(exp g_t) S_{t-1} + k_t v_t^T``.

The stage split of FLA's ``chunk_gla`` on the building blocks of :mod:`vector_gate` (chunks of
``C = 64`` tokens, ``G`` the chunk-local cumulative log-decay ``[C, K]`` in f32):

* ``chunk_cumsum`` (shared with KDA / GDN-2): ``G``, its value at every 8th token, at each chunk's
  last token and the per-chunk ``bounded`` flag;
* :func:`intra`: ``Aqk = scale * tril(q exp(G) . k exp(-G))`` with the sub-chunk-stable
  construction of :func:`vector_gate.intra` (every exponent at most zero, exact 8x8 diagonal
  blocks; a ``bounded`` chunk runs every block on the tensor cores), plus the decayed tiles
  ``qg = scale q exp(G)`` and ``kg = k exp(G_last - G)``. Only ``Aqk``: GLA has no ``Akk``, no
  triangular solve and no WY factors (``w = 0``, ``u = v``);
* :func:`state_output`: the serial pass over chunks fused with the output, one CTA of two
  warpgroups per ``(b, h, value tile)``. Warpgroup 0 keeps ``S`` in f32 registers and runs
  ``S <- exp(G_last) S + kg^T v`` on TMA-fed double buffers, warpgroup 1 computes
  ``o = qg S + Aqk v`` from the state warpgroup 0 publishes;
* backward: :func:`bwd_dav` (``dAqk = tril(do v^T)``, ``dv_o = Aqk^T do``), :func:`bwd_state`
  (serial, reversed: ``dv = dv_o + kg dS_{n+1}`` complete, ``dS_n = exp(G_last) dS_{n+1} + qg^T
  do``), :func:`bwd_inter` (the inter-chunk ``dq, dk, dG`` terms) and :func:`bwd_intra` (the
  gradients through ``Aqk``, the same sub-chunk construction, accumulated onto the inter-chunk
  terms), then the chunk-local reverse cumulative sum of ``dG``.

Inputs keep the FLA layout ``[B, T, H, D]``; intermediates are ``[B, H, T, D]`` and states
``[B, H, NC, K, V]`` (bf16) as in :mod:`vector_gate`. ``T`` is a multiple of ``C`` here;
:func:`chunked_gla` pads.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu._common import finish_together
from jla.ops.mosaic_gpu.vector_gate import (
    CHUNK,
    SCAN_WIDTH,
    SUB_CHUNK,
    _BF16,
    _COL,
    _F32,
    _HALF,
    _KC,
    _MARKS,
    _NUM_SUB,
    _PAD,
    _ROW,
    _WG,
    Config,
    _Regs,
    _cols,
    _factored_blocks,
    _iota,
    _key_tiles,
    _kernel,
    _load,
    _masked_exp,
    _matmul,
    _per_block,
    _rows,
    _slab_smem,
    _split_slabs,
    _store_factored,
    _swizzled,
    _wgmma,
)
from jla.ops.mosaic_gpu.vector_gate_bwd import _SHIFT, _bf16_values, _tile16
from jla.ops.reference import segment_resets

Array = jax.Array

# ---------------------------------------------------------------------------------------------
# intra: Aqk and the decayed tiles of one chunk
# ---------------------------------------------------------------------------------------------


def _slab(slab, q_smem, k_smem, g_smem):
    """``(cols, g, q, k)`` (f32) of one ``_KC``-column slab; ``slab`` may be traced."""
    cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
    g = _load(g_smem.at[:, cols])
    k = _load(k_smem.at[:, cols]).astype(_F32)
    q = _load(q_smem.at[:, cols]).astype(_F32)
    return cols, g, q, k


def _below_diagonal(cfg: Config, refs, x_smem, stage) -> list[Array]:
    """``[C, 16]`` column blocks ``J = 0..2`` of ``P`` restricted to the rows of blocks ``> J``
    (the queries' half of :func:`vector_gate._below_diagonal`): the decay factors through the end of
    block ``J``, every exponent at most zero."""
    q_smem, k_smem, g_smem, marks_smem = refs
    shape = (CHUNK, _KC)
    blocks = _iota(shape, 0) >> 4
    tile = (CHUNK, SUB_CHUNK)

    def scoped(*accs):
        def slab_step(slab, carry):
            cols, g, q, k = _slab(slab, q_smem, k_smem, g_smem)
            ends = [_load(marks_smem.at[2 * blk + 1, cols], _COL) for blk in range(_NUM_SUB)]
            x_smem.at[:, cols][...] = (k * jnp.exp(_per_block(blocks, ends, shape) - g)).astype(_BF16)
            qy = q * _masked_exp(g - _per_block(blocks, [None, *ends[:-1]], shape), blocks >= 1)
            plgpu.commit_smem()
            one = plgpu.layout_cast(jnp.ones((_KC,), _F32), _COL)
            for col_blk in range(_NUM_SUB - 1):
                factors = [None] * (col_blk + 1) + [one] + [
                    jnp.exp(ends[row_blk - 1] - ends[col_blk]) for row_blk in range(col_blk + 2, _NUM_SUB)
                ]
                factor = _per_block(blocks, factors, shape)
                rhs = x_smem.at[pl.ds(SUB_CHUNK * col_blk, SUB_CHUNK), cols].T
                _wgmma(cfg, accs[col_blk], _Regs((qy * factor).astype(_BF16)), rhs, stage)
            plgpu.wgmma_wait(0)  # no wgmma in flight across the loop back-edge
            return carry

        lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)
        return [acc[...] for acc in accs]

    return pl.run_scoped(scoped, *[plgpu.ACC(tile, _F32) for _ in range(_NUM_SUB - 1)])


def _half_blocks(cfg: Config, refs, x2_smem, out_smem, stage) -> None:
    """Lower-left 8x8 quadrant of every diagonal sub-chunk block of ``P`` on the tensor cores,
    factored through the cumulative decay at the block's 8th token; the ``[C, C]`` product goes to
    ``out_smem`` (only those quadrants are meaningful)."""
    q_smem, k_smem, g_smem, marks_smem = refs
    shape = (CHUNK, _KC)
    rows = _iota(shape, 0)
    blocks = rows >> 4
    second = (rows & (SUB_CHUNK - 1)) >= _HALF

    def scoped(acc):
        def slab_step(slab, carry):
            cols, g, q, k = _slab(slab, q_smem, k_smem, g_smem)
            mids = [_load(marks_smem.at[2 * blk, cols], _COL) for blk in range(_NUM_SUB)]
            mid = _per_block(blocks, mids, shape)
            decay = jnp.exp(jnp.where(second, g - mid, mid - g))
            x2_smem.at[:, cols][...] = jnp.where(second, 0.0, k * decay).astype(_BF16)
            plgpu.commit_smem()
            _wgmma(cfg, acc, _Regs(jnp.where(second, q * decay, 0.0).astype(_BF16)), x2_smem.at[:, cols].T, stage)
            plgpu.wgmma_wait(0)
            return carry

        lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)
        out_smem[...] = acc[...]

    pl.run_scoped(scoped, plgpu.ACC((CHUNK, CHUNK), _F32))


def _diag_tiles(cfg: Config, slabs: range, refs, kpad, epad) -> Array:
    """Exact 8x8 diagonal blocks inside each sub-chunk, summed over the key ``slabs``: entry
    ``(i, i % 16 - d)`` (``d < 8``, ``i % 8 >= d``) of the ``[C, 16]`` tile is ``sum_c q_ic
    k_{i-d,c} e^{G_ic - G_{i-d,c}}``, the decay a running product of the per-token ``e^{g_t}``."""
    q_smem, k_smem, g_smem, _ = refs
    c = CHUNK
    tile = (c, SUB_CHUNK)
    rows = pl.ds(_PAD, c)
    data = pl.ds(0, _KC)
    offset = _iota(tile, 0) & (SUB_CHUNK - 1)
    col = _iota(tile, 1)
    zero = plgpu.layout_cast(jnp.zeros(tile, _F32), _WG)
    ones = plgpu.layout_cast(jnp.ones((c, _KC), _F32), _WG)

    def slab_step(slab, diag):
        cols, g, q, k = _slab(slab, q_smem, k_smem, g_smem)
        kpad.at[rows, data][...] = _load(k_smem.at[:, cols])
        epad.at[rows, data][...] = g
        g_prev = _load(epad.at[pl.ds(_PAD - 1, c), data])
        epad.at[rows, data][...] = jnp.exp(g - g_prev)
        diag = diag + jnp.where(col == offset, _rows(jnp.sum(q * k, axis=1), tile), 0.0)

        def offset_step(d, carry):
            decay, diag = carry
            decay = decay * _load(epad.at[pl.ds(_PAD + 1 - d, c), data])
            weighted = _load(kpad.at[pl.ds(_PAD - d, c), data]).astype(_F32) * decay
            at_d = jnp.logical_and(col == offset - d, (offset & (_HALF - 1)) >= d)
            return decay, diag + jnp.where(at_d, _rows(jnp.sum(q * weighted, axis=1), tile), 0.0)

        _, diag = lax.fori_loop(1, _HALF, offset_step, (ones, diag))
        return diag

    return lax.fori_loop(slabs.start, slabs.stop, slab_step, zero)


def _decayed_tiles(cfg: Config, q_smem, k_smem, g_smem, marks_smem, qg_smem, kg_smem) -> None:
    """``qg_smem <- scale q exp(G)`` and ``kg_smem <- k exp(G_last - G)``, one slab at a time."""
    shape = (CHUNK, _KC)

    def slab_step(slab, carry):
        cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
        g = _load(g_smem.at[:, cols])
        last = _cols(_load(marks_smem.at[_MARKS - 1, cols], _COL), shape)
        kg_smem.at[:, cols][...] = (_load(k_smem.at[:, cols]).astype(_F32) * jnp.exp(last - g)).astype(_BF16)
        qg_smem.at[:, cols][...] = (_load(q_smem.at[:, cols]).astype(_F32) * jnp.exp(g) * cfg.scale).astype(_BF16)
        return carry

    lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)


def intra(cfg: Config, q: Array, k: Array, g_cum: Array, g_marks: Array, bounded: Array) -> tuple[Array, Array, Array]:
    """``(Aqk, qg, kg)``: ``Aqk = scale * tril(q exp(G) . k exp(-G))`` ``[B, H, T, C]`` and the
    decayed tiles ``qg = scale q exp(G)``, ``kg = k exp(G_last - G)`` ``[B, H, T, K]``, all bf16.

    ``q, k [B, T, H, K]`` bf16, ``g_cum``, ``g_marks`` and ``bounded`` from
    :func:`vector_gate.chunk_cumsum`. A ``bounded`` chunk: warpgroup 0 runs every block of ``Aqk``
    on the tensor cores (:func:`vector_gate._factored_blocks`) while warpgroup 1 forms ``qg`` and
    ``kg``. Otherwise warpgroup 0 computes the blocks below the diagonal sub-chunk blocks and
    warpgroup 1 the lower-left quadrants of the diagonal blocks; each computes the exact 8x8
    diagonal blocks for part of the key channels, warpgroup 0 assembles ``Aqk`` and warpgroup 1
    then forms ``qg`` and ``kg``.
    """
    c, kd = CHUNK, cfg.key_dim
    tile = (c, SUB_CHUNK)
    out_a = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _BF16)
    out_d = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, kd), _BF16)
    wg_slabs = _split_slabs(kd)
    if bounded.shape != (cfg.batch, cfg.num_chunks, cfg.heads * kd // SCAN_WIDTH):
        raise ValueError(f"bounded must be chunk_cumsum's flag, got shape {bounded.shape}")

    def body(q_ref, k_ref, g_ref, gm_ref, ok_ref, aqk_ref, qg_ref, kg_ref, *scratch):
        (q_smem, k_smem, g_smem, marks_smem, x_smem, x2_smem, qg_smem, kpad, epad, half_smem, diag_smem,
         aqk_smem, load_bar, done_bar, finish) = scratch[:15]
        stage = scratch[15] if cfg.interpret else None  # register operands, interpret mode only
        refs = (q_smem, k_smem, g_smem, marks_smem)
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        wg = lax.axis_index("wg")
        tok = pl.ds(n * c, c)
        factored = ok_ref[b, n, (h * kd) // SCAN_WIDTH] != 0

        def store_decays():  # warpgroup 1, into buffers only it writes
            _decayed_tiles(cfg, q_smem, k_smem, g_smem, marks_smem, qg_smem, x2_smem)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(qg_smem, qg_ref.at[b, h, tok])
            plgpu.copy_smem_to_gmem(x2_smem, kg_ref.at[b, h, tok])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, load_bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem, load_bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, load_bar)
            plgpu.copy_gmem_to_smem(gm_ref.at[b, pl.ds(n * _MARKS, _MARKS), h], marks_smem, load_bar)

        plgpu.barrier_wait(load_bar)
        slab_pair = pl.ds(0, _KC), pl.ds(_KC, _KC)  # per-slab right operands of the factored path

        @pl.when(jnp.logical_and(factored, wg == 0))
        def _():
            parts = _factored_blocks(cfg, (q_smem, k_smem, None, g_smem, marks_smem), False,
                                     x_smem.at[:, slab_pair[0]], x_smem.at[:, slab_pair[1]],
                                     None if stage is None else stage.at[0])
            _store_factored(parts, aqk_smem, strict=False, row_scale=None, scale=cfg.scale)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(aqk_smem, aqk_ref.at[b, h, tok])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(jnp.logical_and(factored, wg == 1))
        def _():
            store_decays()

        @pl.when(jnp.logical_and(jnp.logical_not(factored), wg == 1))
        def _():
            _half_blocks(cfg, refs, x2_smem, half_smem, None if stage is None else stage.at[1])
            diag_smem[...] = _diag_tiles(cfg, wg_slabs[1], refs, kpad.at[1], epad.at[1])
            plgpu.barrier_arrive(done_bar)
            store_decays()  # x2_smem is free again: _half_blocks has run

        @pl.when(jnp.logical_and(jnp.logical_not(factored), wg == 0))
        def _():
            below = _below_diagonal(cfg, refs, x_smem, None if stage is None else stage.at[0])
            if wg_slabs[0]:
                diag = _diag_tiles(cfg, wg_slabs[0], refs, kpad.at[0], epad.at[0])
            else:
                diag = jnp.zeros(tile, _F32)
            trow = _iota(tile, 0)
            blocks = trow >> 4
            quadrant = jnp.logical_and((trow & (SUB_CHUNK - 1)) >= _HALF, _iota(tile, 1) < _HALF)
            plgpu.barrier_wait(done_bar)
            diag = diag + _load(diag_smem)
            for col_blk in range(_NUM_SUB):
                cols = pl.ds(SUB_CHUNK * col_blk, SUB_CHUNK)
                quad = jnp.where(quadrant, _load(half_smem.at[:, cols]), 0.0)
                off = below[col_blk] if col_blk < _NUM_SUB - 1 else jnp.zeros(tile, _F32)
                val = jnp.where(blocks == col_blk, diag + quad, off)
                aqk_smem.at[:, cols][...] = (val * cfg.scale).astype(_BF16)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(aqk_smem, aqk_ref.at[b, h, tok])
            plgpu.wait_smem_to_gmem(0)

        finish_together(finish)  # no warpgroup exits while the other may still issue wgmmas

    scratch = (
        _slab_smem((c, kd), q.dtype),
        _slab_smem((c, kd), k.dtype),
        _slab_smem((c, kd), _F32),
        plgpu.SMEM((_MARKS, kd), _F32),
        _slab_smem((c, kd), _BF16),  # right operands of warpgroup 0
        _slab_smem((c, kd), _BF16),  # right operands of warpgroup 1, then kg
        _slab_smem((c, kd), _BF16),  # qg
        plgpu.SMEM((2, _PAD + c, _KC + 8), _BF16),  # padded pitch: conflict-free row-shifted reads
        plgpu.SMEM((2, _PAD + c, _KC + 8), _F32),
        plgpu.SMEM((c, c), _F32),  # diagonal-block quadrants (warpgroup 1)
        plgpu.SMEM((c, SUB_CHUNK), _F32),  # warpgroup 1's exact diagonal tiles
        plgpu.SMEM((c, c), _BF16),  # Aqk
        plgpu.Barrier(num_arrivals=4),
        plgpu.Barrier(num_arrivals=1),  # exact path: warpgroup 1's tiles are in SMEM
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    if cfg.interpret:
        scratch += (_slab_smem((2, c, _KC), _BF16),)
    kern = _kernel(
        cfg, "gla_intra", body, out_type=(out_a, out_d, out_d), grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(q, k, g_cum, g_marks, bounded)


# ---------------------------------------------------------------------------------------------
# state_output: the serial pass over chunks, fused with the output
# ---------------------------------------------------------------------------------------------


def state_config(cfg: Config) -> Config:
    """Value tile of :func:`state_output`, one serial chain per ``(b, h, value tile)``: 32 wide
    with few chains in flight (latency-bound), the widest tile with many (bandwidth-bound), the
    rule measured for :func:`vector_gate.state_output` on H100."""
    bh = cfg.batch * cfg.heads
    if bh <= 16:
        block_v = 32
    elif bh <= 32 or cfg.value_dim < 128:
        block_v = 64
    else:
        block_v = 128
    return dataclasses.replace(cfg, block_v=min(block_v, cfg.value_dim))


def state_output(
    cfg: Config, kg: Array, v: Array, g_last: Array, h0: Array | None, qg: Array, aqk: Array, *, save: bool
) -> tuple[Array, ...]:
    """``(o, final)``, plus the states ``h [B, H, NC, K, V]`` (bf16, entering each chunk) with
    ``save``: one CTA of two warpgroups per ``(b, h, block_v value columns)``.

    Warpgroup 0 carries ``S`` in f32 registers and runs ``S <- exp(G_last) S + kg^T v``; it
    publishes the bf16 state entering each chunk to warpgroup 1, which computes ``o = qg S + Aqk v``
    while warpgroup 0 updates the state. Each warpgroup double-buffers its own TMA loads (both read
    the chunk's values). ``kg, qg [B, H, T, K]``, ``aqk [B, H, T, C]`` and ``v [B, T, H, V]`` bf16,
    ``g_last [B, NC, H, K]`` f32, ``h0 [B, H, K, V]`` f32 or ``None`` (zero). ``o`` is
    ``[B, T, H, V]`` bf16 and ``final [B, H, K, V]`` f32.
    """
    c, kd, vd, bv, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.block_v, cfg.num_chunks
    out = [jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads, vd), _BF16),
           jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32)]
    if save:
        out.append(jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd, vd), _BF16))
    has_h0 = h0 is not None

    def body(kg_ref, v_ref, gl_ref, qg_ref, a_ref, *refs):
        h0_ref = refs[0] if has_h0 else None
        outs = refs[has_h0:has_h0 + len(out)]
        (kg_smem, v_smem, gl_smem, s_smem, qg_smem, a_smem, vo_smem, o_smem, bars, in_bars, ready, free,
         finish) = refs[has_h0 + len(out):]
        o_ref, final_ref = outs[:2]
        vt, h, b = lax.axis_index("v"), lax.axis_index("h"), lax.axis_index("b")
        wg = lax.axis_index("wg")
        vcols = pl.ds(vt * bv, bv)

        @pl.when(wg == 0)
        def _():
            def fetch(n, slot):
                tok = pl.ds(n * c, c)
                bar = bars.at[slot]
                plgpu.copy_gmem_to_smem(kg_ref.at[b, h, tok], kg_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h, vcols], v_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(gl_ref.at[b, n, h], gl_smem.at[slot], bar)

            fetch(0, 0)
            if has_h0:
                s0 = _load(h0_ref.at[b, h, :, vcols])
            else:
                s0 = plgpu.layout_cast(jnp.zeros((kd, bv), _F32), _WG)

            def step(n, s):
                slot = lax.rem(n, 2)
                plgpu.commit_smem()

                @pl.when(n + 1 < nc)
                def _():
                    fetch(n + 1, 1 - slot)

                plgpu.barrier_wait(bars.at[slot])

                @pl.when(n >= 1)
                def _():
                    plgpu.barrier_wait(free)  # warpgroup 1 is done with the previous state

                if save:
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous state's store read it
                s_smem[...] = s.astype(_BF16)
                plgpu.commit_smem()
                if save:
                    plgpu.copy_smem_to_gmem(s_smem, outs[2].at[b, h, n, :, vcols])
                plgpu.barrier_arrive(ready)
                decay = jnp.exp(_load(gl_smem.at[slot], _ROW))
                s = s * _rows(decay, (kd, bv))
                return _matmul(cfg, (kd, bv), (kg_smem.at[slot].T, v_smem.at[slot]), init=s)

            s = lax.fori_loop(0, nc, step, s0)
            final_ref.at[b, h, :, vcols][...] = s
            plgpu.barrier_wait(free)  # observe warpgroup 1's last arrival
            if save:
                plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _():
            def fetch(n, slot):
                tok = pl.ds(n * c, c)
                bar = in_bars.at[slot]
                plgpu.copy_gmem_to_smem(qg_ref.at[b, h, tok], qg_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(a_ref.at[b, h, tok], a_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h, vcols], vo_smem.at[slot], bar)

            fetch(0, 0)

            def step(n, carry):
                slot = lax.rem(n, 2)

                @pl.when(n + 1 < nc)
                def _():
                    fetch(n + 1, 1 - slot)

                plgpu.barrier_wait(in_bars.at[slot])
                plgpu.barrier_wait(ready)
                o = _matmul(cfg, (c, bv), (qg_smem.at[slot], s_smem), (a_smem.at[slot], vo_smem.at[slot]))
                plgpu.barrier_arrive(free)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's store read o_smem
                o_smem[...] = o.astype(_BF16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_smem, o_ref.at[b, pl.ds(n * c, c), h, vcols])
                return carry

            lax.fori_loop(0, nc, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(finish)  # warpgroup 1 must not exit while warpgroup 0 updates the state

    scratch = (
        _key_tiles(cfg, (2, c, kd)),  # kg
        _swizzled((2, c, bv)),  # v (warpgroup 0)
        plgpu.SMEM((2, kd), _F32),  # cumulative decay at each chunk's last token
        _swizzled((kd, bv)),  # state entering the chunk (bf16)
        _key_tiles(cfg, (2, c, kd)),  # qg
        _key_tiles(cfg, (2, c, c)),  # Aqk
        _swizzled((2, c, bv)),  # v (warpgroup 1)
        _swizzled((c, bv)),  # o
        plgpu.Barrier(num_arrivals=3, num_barriers=2),
        plgpu.Barrier(num_arrivals=3, num_barriers=2),
        plgpu.Barrier(num_arrivals=1),  # state published by warpgroup 0
        plgpu.Barrier(num_arrivals=1),  # state consumed by warpgroup 1
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    kern = _kernel(
        cfg, "gla_state_output", body, out_type=tuple(out), grid=(vd // bv, cfg.heads, cfg.batch),
        grid_names=("v", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(kg, v, g_last, qg, aqk, *(() if h0 is None else (h0,)))


# ---------------------------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------------------------


def bwd_dav(cfg: Config, do: Array, v: Array, aqk: Array) -> tuple[Array, Array]:
    """``(dAqk, dv_o)``: ``dAqk = tril(do v^T)`` ``[B, H, T, C]`` f32 (the gradient of the stored
    ``Aqk``) and ``dv_o = Aqk^T do`` ``[B, H, T, V]`` bf16; ``do, v [B, T, H, V]``."""
    c, vd = CHUNK, cfg.value_dim
    shape_a = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _F32)
    shape_v = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, vd), _BF16)

    def body(do_ref, v_ref, a_ref, da_ref, dv_ref, do_smem, v_smem, a_smem, da_smem, dv_smem, bar):
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        plgpu.copy_gmem_to_smem(do_ref.at[b, tok, h], do_smem, bar)
        plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h], v_smem, bar)
        plgpu.copy_gmem_to_smem(a_ref.at[b, h, tok], a_smem, bar)
        plgpu.barrier_wait(bar)
        causal = _iota((c, c), 1) <= _iota((c, c), 0)
        da_smem[...] = jnp.where(causal, _matmul(cfg, (c, c), (do_smem, v_smem.T)), 0.0)
        dv_smem[...] = _matmul(cfg, (c, vd), (a_smem.T, do_smem)).astype(_BF16)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(da_smem, da_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(dv_smem, dv_ref.at[b, h, tok])
        plgpu.wait_smem_to_gmem(0)

    scratch = (
        _swizzled((c, vd), do.dtype),
        _swizzled((c, vd), v.dtype),
        _swizzled((c, c)),
        _swizzled((c, c), _F32),
        _swizzled((c, vd)),
        plgpu.Barrier(num_arrivals=3),
    )
    kern = _kernel(
        cfg, "gla_bwd_dav", body, out_type=(shape_a, shape_v), grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), scratch_types=scratch,
    )
    return kern(do, v, aqk)


def bwd_state(
    cfg: Config, qg: Array, kg: Array, g_last: Array, do: Array, dv_o: Array, d_final: Array, v_dtype
) -> tuple[Array, Array, Array]:
    """``(dh, dv, dh0)``: serial and reversed over the chunks, ``dh [B, H, NC, K, V]`` bf16 holds
    ``dS_{n+1}`` for chunk ``n``, ``dv = dv_o + kg dS_{n+1}`` ``[B, T, H, V]`` (``v_dtype``) is the
    complete value gradient and ``dh0 [B, H, K, V]`` f32 the initial-state gradient, with
    ``dS_n = exp(G_last) dS_{n+1} + qg^T do`` and ``dS_NC = d_final``."""
    c, kd, vd, bv, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.block_v, cfg.num_chunks
    out_h = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd, vd), _BF16)
    out_v = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads, vd), v_dtype)
    out_s = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32)

    def body(qg_ref, kg_ref, gl_ref, do_ref, dvo_ref, df_ref, dh_ref, dv_ref, dh0_ref, *scratch):
        qg_smem, kg_smem, gl_smem, do_smem, dvo_smem, ds_smem, dv_smem, f_smem, bars, f_bar = scratch
        vt, h, b = lax.axis_index("v"), lax.axis_index("h"), lax.axis_index("b")
        vcols = pl.ds(vt * bv, bv)

        def fetch(n, slot):
            tok = pl.ds(n * c, c)
            bar = bars.at[slot]
            plgpu.copy_gmem_to_smem(qg_ref.at[b, h, tok], qg_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(kg_ref.at[b, h, tok], kg_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(gl_ref.at[b, n, h], gl_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, tok, h, vcols], do_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(dvo_ref.at[b, h, tok, vcols], dvo_smem.at[slot], bar)

        plgpu.copy_gmem_to_smem(df_ref.at[b, h, :, vcols], f_smem, f_bar)
        fetch(nc - 1, 0)
        plgpu.barrier_wait(f_bar)
        ds0 = _load(f_smem)

        def step(i, ds):
            n = nc - 1 - i
            slot = lax.rem(i, 2)
            plgpu.commit_smem()

            @pl.when(n > 0)
            def _():
                fetch(n - 1, 1 - slot)

            plgpu.barrier_wait(bars.at[slot])
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's stores read the tiles
            ds_smem[...] = ds.astype(_BF16)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(ds_smem, dh_ref.at[b, h, n, :, vcols], commit_group=False)
            dv = _load(dvo_smem.at[slot]).astype(_F32) + _matmul(cfg, (c, bv), (kg_smem.at[slot], ds_smem))
            dv_smem[...] = dv.astype(dv_smem.dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dv_smem, dv_ref.at[b, pl.ds(n * c, c), h, vcols], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            decayed = ds * _rows(jnp.exp(_load(gl_smem.at[slot], _ROW)), (kd, bv))
            return _matmul(cfg, (kd, bv), (qg_smem.at[slot].T, do_smem.at[slot]), init=decayed)

        ds = lax.fori_loop(0, nc, step, ds0)
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
        f_smem[...] = ds
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(f_smem, dh0_ref.at[b, h, :, vcols])
        plgpu.wait_smem_to_gmem(0)

    scratch = (
        _key_tiles(cfg, (2, c, kd)),
        _key_tiles(cfg, (2, c, kd)),
        plgpu.SMEM((2, kd), _F32),
        _swizzled((2, c, bv), do.dtype),
        _swizzled((2, c, bv)),
        _swizzled((kd, bv)),
        _swizzled((c, bv), v_dtype),
        _swizzled((kd, bv), _F32),
        plgpu.Barrier(num_arrivals=5, num_barriers=2),
        plgpu.Barrier(num_arrivals=1),
    )
    kern = _kernel(
        cfg, "gla_bwd_state", body, out_type=(out_h, out_v, out_s), grid=(vd // bv, cfg.heads, cfg.batch),
        grid_names=("v", "h", "b"), scratch_types=scratch,
    )
    return kern(qg, kg, g_last, do, dv_o, d_final)


_KP = 64  # key-channel slice of one bwd_inter CTA


def bwd_inter(
    cfg: Config, qg: Array, k: Array, v: Array, g_cum: Array, g_marks: Array, h: Array, dh: Array, do: Array
) -> tuple[Array, Array, Array]:
    """The inter-chunk gradients ``(dq, dk, dG)`` ``[B, T, H, K]`` f32 (``dG`` per token, before the
    reverse cumulative sum), through ``o += qg S_n`` and ``S_{n+1} = exp(G_last) S_n + kg^T v`` with
    ``kg = k exp(G_last - G)``. ``h``/``dh`` are :func:`state_output`'s states and
    :func:`bwd_state`'s ``dS_{n+1}``. One CTA per chunk and 64-wide slice of the key channels.
    """
    c, kd, vd, kp = CHUNK, cfg.key_dim, cfg.value_dim, _KP
    parts = kd // kp
    b_, h_, t_ = cfg.batch, cfg.heads, cfg.seq_len
    out = tuple(jax.ShapeDtypeStruct((b_, t_, h_, kd), _F32) for _ in range(3))

    def body(qg_ref, k_ref, v_ref, g_ref, gm_ref, h_ref, dh_ref, do_ref, dq_ref, dk_ref, dg_ref, *scratch):
        qg_smem, k_smem, g_smem, last_smem, h_smem, dh_smem, do_smem, v_smem, bar = scratch
        p, n, hh, b = lax.axis_index("p"), lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        kcols = pl.ds(p * kp, kp)
        plgpu.copy_gmem_to_smem(qg_ref.at[b, hh, tok, kcols], qg_smem, bar)
        plgpu.copy_gmem_to_smem(k_ref.at[b, tok, hh, kcols], k_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, tok, hh, kcols], g_smem, bar)
        plgpu.copy_gmem_to_smem(gm_ref.at[b, n * _MARKS + _MARKS - 1, hh, kcols], last_smem, bar)
        plgpu.copy_gmem_to_smem(h_ref.at[b, hh, n, kcols], h_smem, bar)
        plgpu.copy_gmem_to_smem(dh_ref.at[b, hh, n, kcols], dh_smem, bar)
        plgpu.copy_gmem_to_smem(do_ref.at[b, tok, hh], do_smem, bar)
        plgpu.copy_gmem_to_smem(v_ref.at[b, tok, hh], v_smem, bar)
        plgpu.barrier_wait(bar)
        kshape = (c, kp)
        # o += qg S_n: dq = scale exp(G) (do S_n^T), dG += qg (do S_n^T).
        dqs = _matmul(cfg, kshape, (do_smem, h_smem.T))
        g = _load(g_smem)
        last = _load(last_smem, _COL)
        to_end = jnp.exp(_cols(last, kshape) - g)
        dq_ref.at[b, tok, hh, kcols][...] = dqs * jnp.exp(g) * cfg.scale
        dg = _load(qg_smem).astype(_F32) * dqs
        # S_{n+1} = exp(G_last) S_n + kg^T v: dkg = v dS^T, dk = dkg exp(G_last - G).
        dkg = _matmul(cfg, kshape, (v_smem, dh_smem.T))
        kgv = _load(k_smem).astype(_F32) * to_end
        dk_ref.at[b, tok, hh, kcols][...] = dkg * to_end
        dg = dg - kgv * dkg
        state_term = _matmul(cfg, (kp, kp), (h_smem, dh_smem.T))  # diagonal: sum_v S dS per channel
        diag = jnp.where(_iota((kp, kp), 0) == _iota((kp, kp), 1), state_term, 0.0)
        d_last = jnp.exp(last) * jnp.sum(diag, axis=0) + jnp.sum(kgv * dkg, axis=0)
        dg = jnp.where(_iota(kshape, 0) == c - 1, dg + _cols(d_last, kshape), dg)
        dg_ref.at[b, tok, hh, kcols][...] = dg

    scratch = (
        _swizzled((c, kp)),
        _swizzled((c, kp), k.dtype),
        _swizzled((c, kp), _F32),
        plgpu.SMEM((kp,), _F32),
        _swizzled((kp, vd)),
        _swizzled((kp, vd)),
        _swizzled((c, vd), do.dtype),
        _swizzled((c, vd), v.dtype),
        plgpu.Barrier(num_arrivals=8),
    )
    kern = _kernel(
        cfg, "gla_bwd_inter", body, out_type=out, grid=(parts, cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("p", "n", "h", "b"), scratch_types=scratch,
    )
    return kern(qg, k, v, g_cum, g_marks, h, dh, do)


def bwd_intra(
    cfg: Config, q: Array, k: Array, g_cum: Array, g_marks: Array, daqk: Array, dq_part: Array, dk_part: Array,
    dg_part: Array,
) -> tuple[Array, Array, Array]:
    """Gradients through ``Aqk = scale P(q, k)`` added to :func:`bwd_inter`'s partial gradients:
    ``(dq, dk, dG)`` ``[B, T, H, K]`` (dtypes of ``q``, ``k``; ``dG`` f32 per token).

    The queries' half of :func:`vector_gate_bwd.bwd_intra`: for ``M = scale dAqk``, ``dq_i = sum_j
    M_ij k_j e^{G_i - G_j}`` and ``dk_j = sum_i M_ij q_i e^{G_i - G_j}`` from the below-diagonal
    blocks and the diagonal blocks' lower-left quadrants on the tensor cores (every exponent at most
    zero) for the pairs at least 8 tokens apart, and the band of pairs at most 7 tokens apart exactly
    in f32 (running products of ``e^{g_t}``; the same split as ``vector_gate_bwd``, see its
    ``matrix_pass``: the two directions' terms of ``dG`` cancel in its reverse cumulative sum only if
    they are rounded alike). Warpgroup 0 stages the bf16 copies of ``M`` without the band, warpgroup 1
    the band's coefficients; then each warpgroup takes half of the key channels.
    """
    c, kd = CHUNK, cfg.key_dim
    b_, h_, t_ = cfg.batch, cfg.heads, cfg.seq_len
    shape = (c, _KC)
    full = (c, c)
    count = kd // _KC
    wg_slabs = (range(0, count // 2), range(count // 2, count))
    out = tuple(jax.ShapeDtypeStruct((b_, t_, h_, kd), dtype) for dtype in (q.dtype, k.dtype, _F32))

    def body(q_ref, k_ref, g_ref, gm_ref, daqk_ref, dqp_ref, dkp_ref, dgp_ref, dq_ref, dk_ref, dg_ref, *scratch):
        (q_smem, k_smem, g_smem, marks_smem, m_bf, mq_bf, coef_smem, x_smem, x2_smem, y_union, kpad, epad, apad,
         bar, sync, finish) = scratch
        y_smem, m_f32 = y_union  # the f32 upstream matrix is only needed before the slabs
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        wg = lax.axis_index("wg")
        tok = pl.ds(n * c, c)

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(daqk_ref.at[b, h, tok], m_f32, bar)
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem, bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, bar)
            plgpu.copy_gmem_to_smem(gm_ref.at[b, pl.ds(n * _MARKS, _MARKS), h], marks_smem, bar)

        plgpu.barrier_wait(bar)
        frow = _iota(full, 0)
        fcol = _iota(full, 1)
        band = frow - fcol < _HALF  # pairs at most 7 tokens apart, computed exactly in the slab loop

        @pl.when(wg == 0)
        def _():
            quadrant = jnp.logical_and(
                jnp.logical_and(frow >> 4 == fcol >> 4, (frow & (SUB_CHUNK - 1)) >= _HALF),
                (fcol & (SUB_CHUNK - 1)) < _HALF,
            )
            m = _load(m_f32) * cfg.scale
            m_bf[...] = jnp.where(band, 0.0, m).astype(_BF16)
            mq_bf[...] = jnp.where(jnp.logical_and(quadrant, jnp.logical_not(band)), m, 0.0).astype(_BF16)

        @pl.when(wg == 1)
        def _():
            m = _load(m_f32) * cfg.scale
            # coef[0, d, i] = M[i, i - d] and coef[1, d, j] = M[j + d, j] = coef[0, d, j + d] (entries
            # past the chunk are never used).
            for d in range(_HALF):
                coef_smem.at[0, d, pl.ds(0, c)][...] = jnp.sum(jnp.where(fcol == frow - d, m, 0.0), axis=1)
            plgpu.commit_smem()
            for d in range(_HALF):
                coef_smem.at[1, d, pl.ds(0, c)][...] = _load(coef_smem.at[0, d, pl.ds(d, c)], _ROW)

        plgpu.commit_smem()
        plgpu.barrier_arrive(sync)
        plgpu.barrier_wait(sync)

        rows = _iota(shape, 0)
        blocks = rows >> 4
        offset = rows & (SUB_CHUNK - 1)
        second = offset >= _HALF
        data = pl.ds(0, _KC)
        ones = plgpu.layout_cast(jnp.ones(shape, _F32), _WG)

        def run_slabs(w, slabs):
            x_w, x2_w, y_w = x_smem.at[w], x2_smem.at[w], y_smem.at[w]
            kpad_w, epad_w, apad_w = kpad.at[w], epad.at[w], apad.at[w]

            def matrix_pass(a, kf, to_end, past, half):
                """``(da, db)`` of ``M`` on this slab (``a`` = the queries)."""
                for cb in range(_NUM_SUB - 1):
                    y_w.at[cb][...] = (a * past[cb]).astype(_BF16)
                y_w.at[_NUM_SUB - 1][...] = jnp.where(second, a * half, 0.0).astype(_BF16)
                apad_w.at[pl.ds(0, c), data][...] = a.astype(_BF16)
                plgpu.commit_smem()
                # Row direction: da_i = sum_J e^{G_i - G_end(J)} (M_iJ X_J), then the quadrants.
                da = None
                for cb in range(_NUM_SUB - 1):
                    blk = pl.ds(SUB_CHUNK * cb, SUB_CHUNK)
                    term = past[cb] * _matmul(cfg, shape, (m_bf.at[:, blk], x_w.at[blk]))
                    da = term if da is None else da + term
                da = da + jnp.where(second, _matmul(cfg, shape, (mq_bf, x2_w)) * half, 0.0)
                # Column direction: db_j = e^{G_end(J) - G_j} sum_i M_ij a_i e^{G_i - G_end(J)}, quadrants.
                db = None
                for cb in range(_NUM_SUB - 1):
                    term = jnp.where(blocks == cb, _matmul(cfg, shape, (m_bf.T, y_w.at[cb])), 0.0)
                    db = term if db is None else db + term
                quad = _matmul(cfg, shape, (mq_bf.T, y_w.at[_NUM_SUB - 1]))
                db = db * to_end + jnp.where(second, 0.0, quad * half)
                # Exact band: running products of e^{g_t}, row- and column-shifted.
                row_coef, col_coef = coef_smem.at[0], coef_smem.at[1]
                on_diag = _rows(_load(row_coef.at[0, pl.ds(0, c)], _ROW), shape)
                da = da + on_diag * kf
                db = db + on_diag * a

                def offset_step(d, carry):
                    back, fwd, da, db = carry
                    back = back * _load(epad_w.at[pl.ds(_SHIFT + 1 - d, c), data])
                    fwd = fwd * _load(epad_w.at[pl.ds(_SHIFT + d, c), data])
                    b_prev = _load(kpad_w.at[pl.ds(_SHIFT - d, c), data]).astype(_F32)
                    a_next = _load(apad_w.at[pl.ds(d, c), data]).astype(_F32)
                    da = da + jnp.where(rows >= d, _rows(_load(row_coef.at[d, pl.ds(0, c)], _ROW), shape)
                                        * b_prev * back, 0.0)
                    db = db + jnp.where(rows + d <= c - 1, _rows(_load(col_coef.at[d, pl.ds(0, c)], _ROW), shape)
                                        * a_next * fwd, 0.0)
                    return back, fwd, da, db

                _, _, da, db = lax.fori_loop(1, _HALF, offset_step, (ones, ones, da, db))
                return da, db

            def slab_step(slab, carry):
                cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
                g = _load(g_smem.at[:, cols])
                # q and k as bf16 values (exact for bf16 inputs): the row direction multiplies by them in
                # f32 and the column direction reads them back from the bf16 shift buffers.
                kf = _bf16_values(_load(k_smem.at[:, cols]))
                ends = [_load(marks_smem.at[2 * blk + 1, cols], _COL) for blk in range(_NUM_SUB)]
                mids = [_load(marks_smem.at[2 * blk, cols], _COL) for blk in range(_NUM_SUB)]
                to_end = jnp.exp(_per_block(blocks, ends, shape) - g)  # e^{G_end(block) - G}
                # past[J] = e^{G - G_end(J)} on the rows of the blocks after J (zero elsewhere).
                past = [_masked_exp(g - _cols(ends[cb], shape), blocks > cb) for cb in range(_NUM_SUB - 1)]
                mid = _per_block(blocks, mids, shape)
                half = jnp.exp(jnp.where(second, g - mid, mid - g))

                plgpu.commit_smem()  # every warp is done with the previous slab's buffers
                x_w[...] = (kf * to_end).astype(_BF16)
                x2_w[...] = jnp.where(second, 0.0, kf * half).astype(_BF16)
                kpad_w.at[pl.ds(_SHIFT, c), data][...] = kf.astype(_BF16)
                epad_w.at[pl.ds(_SHIFT, c), data][...] = g
                g_prev = _load(epad_w.at[pl.ds(_SHIFT - 1, c), data])
                epad_w.at[pl.ds(_SHIFT, c), data][...] = jnp.exp(g - g_prev)

                qf = _bf16_values(_load(q_smem.at[:, cols]))
                da_q, db_q = matrix_pass(qf, kf, to_end, past, half)
                dq_ref.at[b, tok, h, cols][...] = (_load(dqp_ref.at[b, tok, h, cols]) + da_q).astype(dq_ref.dtype)
                dk_ref.at[b, tok, h, cols][...] = (_load(dkp_ref.at[b, tok, h, cols]) + db_q).astype(dk_ref.dtype)
                dg_ref.at[b, tok, h, cols][...] = _load(dgp_ref.at[b, tok, h, cols]) + qf * da_q - kf * db_q
                return carry

            lax.fori_loop(slabs.start, slabs.stop, slab_step, None)

        for w, slabs in enumerate(wg_slabs):

            @pl.when(wg == w)
            def _():
                run_slabs(w, slabs)

        finish_together(finish)  # no warpgroup exits while the other may still issue wgmmas

    scratch = (
        _slab_smem((c, kd), q.dtype),
        _slab_smem((c, kd), k.dtype),
        _slab_smem((c, kd), _F32),
        plgpu.SMEM((_MARKS, kd), _F32),
        _tile16((c, c)),  # bf16 scale dAqk without the band
        _tile16((c, c)),  # the same, only the lower-left quadrants of the diagonal blocks
        plgpu.SMEM((2, _HALF, c + _HALF), _F32),  # exact band coefficients
        _tile16((2, c, _KC)),
        _tile16((2, c, _KC)),
        plgpu.RefUnion(_tile16((2, _NUM_SUB, c, _KC)), plgpu.SMEM((c, c), _F32)),  # y tiles / f32 dAqk
        # Row-shift buffers (keys, per-token decays, the queries); the padded pitch keeps the
        # shifted loads free of bank conflicts.
        plgpu.SMEM((2, _SHIFT + c, _KC + 8), _BF16),
        plgpu.SMEM((2, _SHIFT + c + _HALF, _KC + 8), _F32),
        plgpu.SMEM((2, c + _HALF, _KC + 8), _BF16),
        plgpu.Barrier(num_arrivals=5),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    kern = _kernel(
        cfg, "gla_bwd_intra", body, out_type=out, grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(q, k, g_cum, g_marks, daqk, dq_part, dk_part, dg_part)


# ---------------------------------------------------------------------------------------------
# the differentiable op
# ---------------------------------------------------------------------------------------------


def _run_forward(cfg: Config, q, k, v, g, h0, *, save: bool):
    """``(o, final)`` and, with ``save``, the residuals of the backward (nothing is recomputed)."""
    g_cum, marks, g_last, bounded = vg.chunk_cumsum(cfg, g)
    aqk, qg, kg = intra(cfg, q, k, g_cum, marks, bounded)
    outs = state_output(state_config(cfg), kg, v, g_last, h0, qg, aqk, save=save)
    if not save:
        return outs, None
    o, final, states = outs
    return (o, final), (q, k, v, g_cum, marks, g_last, aqk, qg, kg, states, h0)


def _backward(cfg: Config, residuals, cotangents):
    q, k, v, g_cum, marks, g_last, aqk, qg, kg, states, h0 = residuals
    do, d_final = cotangents
    do = do.astype(q.dtype)
    daqk, dv_o = bwd_dav(cfg, do, v, aqk)
    d_states, dv, dh0 = bwd_state(cfg, qg, kg, g_last, do, dv_o, d_final.astype(_F32), v.dtype)
    dq_part, dk_part, dg_part = bwd_inter(cfg, qg, k, v, g_cum, marks, states, d_states, do)
    dq, dk, dg_tok = bwd_intra(cfg, q, k, g_cum, marks, daqk, dq_part, dk_part, dg_part)
    # G is the chunk-local cumulative sum of g, so dg is the chunk-local reverse cumulative sum.
    dg = vg.chunk_reverse_cumsum(cfg, dg_tok)
    return dq, dk, dv, dg, (None if h0 is None else dh0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _op(cfg: Config, q, k, v, g, h0):
    return _run_forward(cfg, q, k, v, g, h0, save=False)[0]  # no gradient: nothing to save


_op.defvjp(lambda cfg, q, k, v, g, h0: _run_forward(cfg, q, k, v, g, h0, save=True), _backward)

# The extra log-decay of a segment's first token, as in vector_gate_ops: exp(-128) is zero in f32
# and every exponent in the kernels is a cumulative difference (at most zero), so no term crosses a
# segment start; chunks with a start take intra's exact path.
_RESET_DECAY = 128.0


def chunked_gla(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    *,
    scale: float,
    initial_state: Array | None,
    segment_ids: Array | None = None,
    interpret: bool = False,
) -> tuple[Array, Array]:
    """``(o, final_state)`` of GLA for the FLA layout with any ``T`` (padded to the chunk here; zero
    padding is inert): ``q, k [B, T, H, K]`` and ``v [B, T, H, V]`` bf16, ``g [B, T, H, K]`` the
    log-decay. ``o`` has ``q``'s dtype, the final state is f32 ``[B, H, K, V]``. ``segment_ids
    [B, T]`` restarts the recurrence at every change of id (the state entering a segment is zero)."""
    batch, seq_len, heads, key_dim = q.shape
    pad = (-seq_len) % CHUNK

    def padded(x):
        return jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)) if pad else x

    g = g.astype(_F32)
    if segment_ids is not None:
        g = jnp.where(segment_resets(segment_ids)[:, :, None, None], g - _RESET_DECAY, g)
    if initial_state is not None:
        initial_state = initial_state.astype(_F32)
    cfg = Config(
        batch=batch, seq_len=seq_len + pad, heads=heads, key_dim=key_dim, value_dim=v.shape[-1],
        scale=float(scale), channel_gates=False, interpret=interpret,
    )
    o, final = _op(cfg, padded(q), padded(k), padded(v), padded(g), initial_state)
    return o[:, :seq_len], final
