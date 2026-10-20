# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/kda/chunk_intra.py, fla/ops/kda/wy_fast.py,
# fla/ops/common/chunk_delta_h.py, fla/ops/gla/chunk.py, fla/ops/kda/chunk_bwd.py, fla/ops/utils/solve_tril.py
# Adapted from FlashKDA (MIT), csrc/smxx/fwd_kernel1.cuh: bounded decays factored on the tensor cores.
# The channel-wise erase/write gates (GDN-2) are implemented from arXiv 2605.22791 (equations only).
"""Mosaic GPU (H100) kernels for delta-rule linear attention with a log-decay per key channel.

One set of kernels serves KDA (scalar write strength ``beta``) and Gated
DeltaNet-2 (channel-wise erase gate ``b [.., K]`` and write gate ``w [.., V]``).
Per chunk of ``C = 64`` tokens, with ``G`` the chunk-local cumulative log-decay
(``[C, K]``, float32) and the per-token gates folded into ``erase`` (``beta_i``
or ``b_i``, multiplying the key on the read side) and ``write`` (``beta_i`` or
``w_i``, multiplying the value):

* ``chunk_cumsum``: ``G``, its value at every 8th token and at each chunk's last token, and a
  per-chunk flag: no 16-token span decays by more than ``BOUNDED_SPAN``;
* ``intra``: ``Aqk = scale * tril(q exp(G) . k exp(-G))`` and the strictly lower
  ``Akk[i, j] = sum_c (erase_i k_i)_c k_jc exp(G_ic - G_jc)``;
* ``solve``: ``T = (I + Akk)^-1``; ``wy``: ``w = T (erase exp(G) k)``, ``u = T (write v)``,
  the end-of-chunk keys ``kg = k exp(G_last - G)`` and ``qg = scale q exp(G)``;
* ``state_output``: serial over chunks, ``v_new = u - w S`` and ``S <- exp(G_last) S + kg^T v_new``,
  with ``o = scale (q exp(G)) S + Aqk v_new`` on a second warpgroup.

The score matrices use 16-token sub-chunks like ``jla.ops.chunked.decayed_scores``: an
off-diagonal block factors the decay through a reference inside the chunk so that every
exponent is at most zero, and the diagonal blocks are exact (running products of the per-token
decays ``exp(g_t)``, no factorisation). No exponential can overflow for any gate value. A chunk
whose 16-token spans decay by at most ``BOUNDED_SPAN`` (any gate bounded below by -5 per token,
such as FlashKDA's) instead factors its 32-token diagonal blocks through their middle token, every
factor ``e^x`` with ``|x| <= BOUNDED_SPAN``, and runs every block on the tensor cores (the bounded
chunk idea of FlashKDA, MIT, without its bound on the gate: other chunks take the exact path).

Layouts: inputs keep the FLA layout ``[B, T, H, D]``; per-chunk intermediates are
``[B, H, T, D]`` (contiguous ``[C, D]`` tiles) and states ``[B, H, NC, K, V]``. ``T`` must be a
multiple of ``C``; the public wrappers pad.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import finish_together

Array = jax.Array

CHUNK = 64
SUB_CHUNK = 16
_NUM_SUB = CHUNK // SUB_CHUNK

_WG = plgpu.Layout.WGMMA
_ROW = plgpu.Layout.WGMMA.reduce(1)  # one value per row of a WGMMA-layout array
_COL = plgpu.Layout.WGMMA.reduce(0)  # one value per column
_F32 = jnp.float32
_BF16 = jnp.bfloat16


@dataclasses.dataclass(frozen=True)
class Config:
    """Static shape and option set shared by the kernels of one call."""

    batch: int
    seq_len: int  # padded to a multiple of CHUNK
    heads: int
    key_dim: int
    value_dim: int
    scale: float
    channel_gates: bool  # GDN-2 (erase [.., K], write [.., V]) instead of a scalar beta
    block_v: int = 64  # value-dimension tile of the state kernels
    interpret: bool = False
    l2norm: bool = False  # intra scales q and k to unit L2 norm (the forward without wy)
    fuse_solve: bool = False  # opt-in: intra also computes T = (I + Akk)^-1 (no separate solve; not when interpreting)

    @property
    def num_chunks(self) -> int:
        return self.seq_len // CHUNK


def _kernel(cfg: Config, name: str, body, **kwargs):
    # Lane semantics: SMEM buffers keep exactly the transforms given here. Warpgroup semantics
    # infers tilings/swizzles for untransformed scratch, which breaks row-shifted views.
    params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane, approx_math=True)
    interpret = plgpu.InterpretGPUParams() if cfg.interpret else None
    return plgpu.kernel(body, interpret=interpret, kernel_name=name, compiler_params=params, **kwargs)


def _swizzled(shape: tuple[int, ...], dtype=_BF16) -> plgpu.SMEM:
    """SMEM tiled and swizzled for its minor dimension (WGMMA operands, conflict-free loads)."""
    itemsize = jnp.dtype(dtype).itemsize
    swizzle = plgpu.find_swizzle(shape[-1] * itemsize * 8)
    transforms = (plgpu.TilingTransform((8, swizzle // itemsize)), plgpu.SwizzleTransform(swizzle))
    return plgpu.SMEM(shape, dtype, transforms=transforms)


def _iota(shape: tuple[int, int], dim: int) -> Array:
    return plgpu.broadcasted_iota(jnp.int32, shape, dim, layout=_WG)


def _load(ref, layout=_WG) -> Array:
    return plgpu.load(ref, layout=layout, optimized=False)


def _rows(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-row vector (``_ROW`` layout) along the columns."""
    return lax.broadcast_in_dim(x, shape, [0])


def _cols(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-column vector (``_COL`` layout) along the rows."""
    return lax.broadcast_in_dim(x, shape, [1])


def _masked_exp(exponent: Array, mask: Array) -> Array:
    return jnp.exp(jnp.where(mask, exponent, -jnp.inf))


class _Regs(NamedTuple):
    """A bf16 WGMMA-layout register array used as the left operand of a matmul."""

    value: Array


def _matmul(cfg: Config, shape: tuple[int, int], *pairs, init: Array | None = None) -> Array:
    """``init + sum(a @ b for a, b in pairs)`` on the tensor cores (f32 accumulator).

    ``a`` is an SMEM ref or a :class:`_Regs` operand, ``b`` an SMEM ref.
    """
    regs = [i for i, (a, _) in enumerate(pairs) if isinstance(a, _Regs)]
    if cfg.interpret and regs:
        # The GPU interpreter's wgmma reads both operands from SMEM: stage register operands.
        def staged(*bufs):
            staged_pairs = list(pairs)
            for i, buf in zip(regs, bufs):
                buf[...] = pairs[i][0].value
                staged_pairs[i] = (buf, pairs[i][1])
            plgpu.commit_smem()
            return _matmul(cfg, shape, *staged_pairs, init=init)

        return pl.run_scoped(staged, *(_swizzled(pairs[i][0].value.shape) for i in regs))
    pairs = tuple((a.value if isinstance(a, _Regs) else a, b) for a, b in pairs)
    if cfg.interpret and init is not None:
        # The interpreter cannot run ``pl.run_state`` on an initialised accumulator.
        return init + _matmul(cfg, shape, *pairs)

    if init is None:

        def scoped(acc):
            for a, b in pairs:
                plgpu.wgmma(acc, a, b)
            return acc[...]

        return pl.run_scoped(scoped, plgpu.ACC(shape, _F32))

    def body(acc):
        for a, b in pairs:
            plgpu.wgmma(acc, a, b)

    return pl.run_state(body)(plgpu.ACC.init(init))


# ---------------------------------------------------------------------------------------------
# intra: decayed score matrices of one chunk
# ---------------------------------------------------------------------------------------------

_KC = 32  # column slab width inside the intra-chunk kernels
_PAD = SUB_CHUNK  # leading rows of the row-shift buffers


def _slab_smem(shape: tuple[int, ...], dtype) -> plgpu.SMEM:
    """SMEM tiled in ``[8, _KC]`` tiles (swizzled), so every ``_KC``-column slab is tile aligned."""
    swizzle = _KC * jnp.dtype(dtype).itemsize
    return plgpu.SMEM(shape, dtype, transforms=(plgpu.TilingTransform((8, _KC)), plgpu.SwizzleTransform(swizzle)))


def _wgmma(cfg: Config, acc, a, b, stage=None) -> None:
    """``acc += a @ b`` with ``a`` an SMEM ref or a :class:`_Regs` operand.

    The GPU interpreter's wgmma reads both operands from SMEM, so in interpret mode a register
    ``a`` goes through ``stage`` (an SMEM buffer of its shape owned by the calling warpgroup: the
    interpreter cannot give warpgroups that run different code their own ``run_scoped`` buffers).
    """
    if not isinstance(a, _Regs):
        plgpu.wgmma(acc, a, b)
    elif not cfg.interpret:
        plgpu.wgmma(acc, a.value, b)
    else:
        stage[...] = a.value
        plgpu.commit_smem()
        plgpu.wgmma(acc, stage, b)


def _per_block(blocks: Array, values: list[Array | None], shape: tuple[int, int]) -> Array:
    """Row ``i`` takes ``values[blocks[i]]`` (a per-column ``_COL`` vector; ``None`` is zero)."""
    out = None
    for blk in reversed(range(len(values))):
        val = values[blk]
        val = jnp.zeros(shape, _F32) if val is None else _cols(val, shape)
        out = val if out is None else jnp.where(blocks == blk, val, out)
    return out


_MARKS = CHUNK // 8  # cumulative decay at every 8th token: indices 2J + 1 end sub-chunk J, 2J its middle
_HALF = SUB_CHUNK // 2


# ---------------------------------------------------------------------------------------------
# cumsum: chunk-local cumulative log-decay (forward) and its transpose (backward)
# ---------------------------------------------------------------------------------------------

SCAN_WIDTH = 128  # (head, channel) columns per scan CTA: one per thread of the warpgroup
# Largest decay of a 16-token span (from the token before it) for which :func:`intra` factors the
# 32-token diagonal blocks through their 16th token: every factor is then e^x with |x| <= 80, a
# normal f32/bf16 number. FlashKDA's gate (lower bound -5 per token) always qualifies.
BOUNDED_SPAN = 80.0
# Extra log-decay of a segment's first token (see vector_gate_ops.chunked_vector_gate): exp(-128) is zero in f32.
RESET_DECAY = 128.0


def _scan(cfg: Config, x: Array, *, reverse: bool, resets: Array | None = None) -> tuple[Array, ...]:
    """Sequential f32 scan of ``x [B, T, H, K]`` over the tokens of each chunk, one thread per
    ``(head, channel)`` column. Forward: ``(G, marks, g_last, bounded)``; reverse: the suffix sums.
    ``resets [B, T]`` f32 (1.0 where a segment starts, else 0.0): those tokens' ``x`` less
    :data:`RESET_DECAY` first (``x - 128.0``, as the same subtraction in XLA would give)."""
    b_, t_, h_, kd = x.shape
    width = h_ * kd
    if width % SCAN_WIDTH:
        raise ValueError(f"heads * key_dim must be a multiple of {SCAN_WIDTH}, got {width}")
    line = plgpu.Layout.WG_STRIDED((SCAN_WIDTH,), vec_size=1)
    step_rows = CHUNK // _MARKS
    out = [jax.ShapeDtypeStruct((b_, t_, width), _F32)]
    if not reverse:
        out += [jax.ShapeDtypeStruct((b_, t_ // step_rows, width), _F32),
                jax.ShapeDtypeStruct((b_, cfg.num_chunks, width), _F32),
                jax.ShapeDtypeStruct((b_, cfg.num_chunks, width // SCAN_WIDTH), jnp.int32)]

    def body(x_ref, *refs):
        r_ref, refs = (refs[0], refs[1:]) if resets is not None else (None, refs)
        outs, (x_smem, marks_smem, bar) = refs[: len(out)], refs[len(out):]
        col, n, b = lax.axis_index("col"), lax.axis_index("n"), lax.axis_index("b")
        cols = pl.ds(col * SCAN_WIDTH, SCAN_WIDTH)
        tok = pl.ds(n * CHUNK, CHUNK)
        plgpu.copy_gmem_to_smem(x_ref.at[b, tok, cols], x_smem, bar)
        plgpu.barrier_wait(bar)

        def step(i, acc):
            group = _MARKS - 1 - i if reverse else i
            for r in reversed(range(step_rows)) if reverse else range(step_rows):
                row = x_smem.at[group * step_rows + r]
                val = _load(row, line)
                if r_ref is not None:
                    val = val - RESET_DECAY * r_ref[b, n * CHUNK + group * step_rows + r]
                acc = acc + val
                row[...] = acc
            if not reverse:
                marks_smem.at[group][...] = acc
            return acc

        lax.fori_loop(0, _MARKS, step, plgpu.layout_cast(jnp.zeros((SCAN_WIDTH,), _F32), line))
        if not reverse:
            # Each thread reads back its own column. Spans end at tokens 15, 31, 47, 63; G = 0 before
            # the chunk. A NaN fails the test, as does a segment start (its extra -128).
            ends = [_load(marks_smem.at[2 * blk + 1], line) for blk in range(_NUM_SUB)]
            span = -ends[0]
            for blk in range(1, _NUM_SUB):
                span = jnp.maximum(span, ends[blk - 1] - ends[blk])
            outs[3][b, n, col] = jnp.where(jnp.max(span) <= BOUNDED_SPAN, 1, 0).astype(jnp.int32)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(x_smem, outs[0].at[b, tok, cols])
        if not reverse:
            plgpu.copy_smem_to_gmem(marks_smem, outs[1].at[b, pl.ds(n * _MARKS, _MARKS), cols])
            plgpu.copy_smem_to_gmem(marks_smem.at[_MARKS - 1], outs[2].at[b, n, cols])
        plgpu.wait_smem_to_gmem(0)

    scratch = (plgpu.SMEM((CHUNK, SCAN_WIDTH), _F32), plgpu.SMEM((_MARKS, SCAN_WIDTH), _F32),
               plgpu.Barrier(num_arrivals=1))
    kern = _kernel(
        cfg, "vg_rev_cumsum" if reverse else "vg_cumsum", body, out_type=tuple(out),
        grid=(width // SCAN_WIDTH, cfg.num_chunks, b_), grid_names=("col", "n", "b"), scratch_types=scratch,
    )
    results = kern(x.reshape(b_, t_, width), *(() if resets is None else (resets,)))
    return (*(r.reshape(b_, -1, h_, kd) for r in results[:3]), *results[3:])


def chunk_cumsum(cfg: Config, g: Array, resets: Array | None = None) -> tuple[Array, Array, Array, Array]:
    """``(G, marks, g_last, bounded)`` for the log-decay ``g [B, T, H, K]`` f32 (at most zero): the
    chunk-local cumulative sum ``G [B, T, H, K]``, its value at every 8th token ``[B, T / 8, H, K]``
    and at the last token of each chunk ``[B, NC, H, K]``, and ``bounded [B, NC, H K / 128]`` int32:
    1 where no 16-token span of the chunk decays by more than :data:`BOUNDED_SPAN` in any of the
    128 ``(head, channel)`` columns of the group. With ``resets`` ``[B, T]`` f32 (1.0 at a segment's first
    token) those tokens' log decays are less :data:`RESET_DECAY` (see :func:`_scan`)."""
    return _scan(cfg, g, reverse=False, resets=resets)


def chunk_reverse_cumsum(cfg: Config, dg: Array) -> Array:
    """``out_t = sum_{s >= t} dg_s`` within each chunk (the transpose of :func:`chunk_cumsum`'s ``G``)."""
    return _scan(cfg, dg, reverse=True)[0]


def _slab_inputs(cfg: Config, slab, q_smem, k_smem, e_smem, g_smem):
    """``(cols, g, q, k, kr)`` for one ``_KC``-column slab (f32; ``kr`` = read-side keys).

    ``slab`` may be traced (a loop index): code is emitted once per loop, which keeps the kernels
    small enough for the instruction cache.
    """
    cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
    g = _load(g_smem.at[:, cols])
    k = _load(k_smem.at[:, cols]).astype(_F32)
    q = _load(q_smem.at[:, cols]).astype(_F32)
    kr = k * _load(e_smem.at[:, cols]).astype(_F32) if cfg.channel_gates else k
    return cols, g, q, k, kr


def _below_diagonal(cfg: Config, refs, x_smem, stage) -> list[Array]:
    """Blocks below the diagonal sub-chunk blocks, on the tensor cores.

    Returns ``[P_J, Q_J for J in 0..2]``: ``[C, 16]`` column block ``J`` of ``P`` / ``Q`` restricted to
    rows of blocks ``> J``. The decay factors through the end of block ``J``: ``X_j = k_j
    e^{G_end(J) - G_j}`` is the right operand and ``q_i e^{G_i - G_end(J)}`` the left one, built as
    ``q_i e^{G_i - G_start(I)}`` times the per-channel ``e^{G_start(I) - G_end(J)}`` (all exponents
    are at most zero).
    """
    q_smem, k_smem, e_smem, g_smem, marks_smem = refs
    c = CHUNK
    shape = (c, _KC)
    blocks = _iota(shape, 0) >> 4
    tile = (c, SUB_CHUNK)

    def scoped(*accs):
        def slab_step(slab, carry):
            cols, g, q, k, kr = _slab_inputs(cfg, slab, q_smem, k_smem, e_smem, g_smem)
            ends = [_load(marks_smem.at[2 * blk + 1, cols], _COL) for blk in range(_NUM_SUB)]
            x_smem.at[:, cols][...] = (k * jnp.exp(_per_block(blocks, ends, shape) - g)).astype(_BF16)
            decay_in = _masked_exp(g - _per_block(blocks, [None, *ends[:-1]], shape), blocks >= 1)
            qy, ky = q * decay_in, kr * decay_in
            plgpu.commit_smem()
            one = plgpu.layout_cast(jnp.ones((_KC,), _F32), _COL)
            for col_blk in range(_NUM_SUB - 1):
                factors = [None] * (col_blk + 1) + [one] + [
                    jnp.exp(ends[row_blk - 1] - ends[col_blk]) for row_blk in range(col_blk + 2, _NUM_SUB)
                ]
                factor = _per_block(blocks, factors, shape)
                rhs = x_smem.at[pl.ds(SUB_CHUNK * col_blk, SUB_CHUNK), cols].T
                _wgmma(cfg, accs[2 * col_blk], _Regs((qy * factor).astype(_BF16)), rhs, stage)
                _wgmma(cfg, accs[2 * col_blk + 1], _Regs((ky * factor).astype(_BF16)), rhs, stage)
            # Accumulators allocated outside a loop must not have wgmmas in flight across its
            # back-edge (the results came out corrupted without this wait).
            plgpu.wgmma_wait(0)
            return carry

        lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)
        return [acc[...] for acc in accs]

    return pl.run_scoped(scoped, *[plgpu.ACC(tile, _F32) for _ in range(2 * (_NUM_SUB - 1))])


def _half_blocks(cfg: Config, refs, x2_smem, out_smem, stage) -> None:
    """Lower-left 8x8 quadrant of every diagonal sub-chunk block, on the tensor cores.

    The decay factors through the cumulative decay at the block's 8th token (``G_mid``):
    ``k_j e^{G_mid - G_j}`` for the first half and ``q_i e^{G_i - G_mid}`` for the second half, both
    at most one. Writes the full ``[C, C]`` products (only those quadrants are meaningful) to
    ``out_smem[0]`` (``q``) and ``out_smem[1]`` (read-side keys).
    """
    q_smem, k_smem, e_smem, g_smem, marks_smem = refs
    c = CHUNK
    shape = (c, _KC)
    rows = _iota(shape, 0)
    blocks = rows >> 4
    second = (rows & (SUB_CHUNK - 1)) >= _HALF

    def scoped(acc_q, acc_k):
        def slab_step(slab, carry):
            cols, g, q, k, kr = _slab_inputs(cfg, slab, q_smem, k_smem, e_smem, g_smem)
            mids = [_load(marks_smem.at[2 * blk, cols], _COL) for blk in range(_NUM_SUB)]
            mid = _per_block(blocks, mids, shape)
            decay = jnp.exp(jnp.where(second, g - mid, mid - g))
            x2_smem.at[:, cols][...] = jnp.where(second, 0.0, k * decay).astype(_BF16)
            plgpu.commit_smem()
            rhs = x2_smem.at[:, cols].T
            _wgmma(cfg, acc_q, _Regs(jnp.where(second, q * decay, 0.0).astype(_BF16)), rhs, stage)
            _wgmma(cfg, acc_k, _Regs(jnp.where(second, kr * decay, 0.0).astype(_BF16)), rhs, stage)
            plgpu.wgmma_wait(0)  # no wgmma in flight across the loop back-edge (see _below_diagonal)
            return carry

        lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)
        out_smem[0] = acc_q[...]
        out_smem[1] = acc_k[...]

    pl.run_scoped(scoped, plgpu.ACC((c, c), _F32), plgpu.ACC((c, c), _F32))


def _diag_tiles(cfg: Config, slabs: range, refs, kpad, epad) -> tuple[Array, Array]:
    """Exact 8x8 diagonal blocks inside each sub-chunk, summed over the key-dimension ``slabs``.

    Returns ``[C, 16]`` tiles whose entry ``(i, i % 16 - d)`` (``d < 8``, ``i % 8 >= d``) is
    ``sum_c q_ic k_{i-d,c} e^{G_ic - G_{i-d,c}}`` (and the same with the read-side keys, ``d >= 1``).
    ``e^{G_i - G_{i-d}}`` is the running product of the per-token decays ``e^{g_t} = e^{G_t -
    G_{t-1}}``, read row-shifted from the padded buffers ``kpad`` / ``epad``; no exponential of a
    cumulative difference is formed (the decay at a group's first token is never used).
    """
    q_smem, k_smem, e_smem, g_smem, _ = refs
    c = CHUNK
    tile = (c, SUB_CHUNK)
    rows = pl.ds(_PAD, c)
    data = pl.ds(0, _KC)
    offset = _iota(tile, 0) & (SUB_CHUNK - 1)
    col = _iota(tile, 1)
    zero = plgpu.layout_cast(jnp.zeros(tile, _F32), _WG)
    ones = plgpu.layout_cast(jnp.ones((c, _KC), _F32), _WG)

    def slab_step(slab, carry):
        diag_q, diag_k = carry
        cols, g, q, k, kr = _slab_inputs(cfg, slab, q_smem, k_smem, e_smem, g_smem)
        kpad.at[rows, data][...] = _load(k_smem.at[:, cols])
        epad.at[rows, data][...] = g
        g_prev = _load(epad.at[pl.ds(_PAD - 1, c), data])
        epad.at[rows, data][...] = jnp.exp(g - g_prev)
        diag_q = diag_q + jnp.where(col == offset, _rows(jnp.sum(q * k, axis=1), tile), 0.0)

        def offset_step(d, carry):
            decay, diag_q, diag_k = carry
            decay = decay * _load(epad.at[pl.ds(_PAD + 1 - d, c), data])
            weighted = _load(kpad.at[pl.ds(_PAD - d, c), data]).astype(_F32) * decay
            at_d = jnp.logical_and(col == offset - d, (offset & (_HALF - 1)) >= d)
            diag_q = diag_q + jnp.where(at_d, _rows(jnp.sum(q * weighted, axis=1), tile), 0.0)
            diag_k = diag_k + jnp.where(at_d, _rows(jnp.sum(kr * weighted, axis=1), tile), 0.0)
            return decay, diag_q, diag_k

        _, diag_q, diag_k = lax.fori_loop(1, _HALF, offset_step, (ones, diag_q, diag_k))
        return diag_q, diag_k

    return lax.fori_loop(slabs.start, slabs.stop, slab_step, (zero, zero))


_BLOCK = 2 * SUB_CHUNK  # diagonal blocks of the factored path


def _factored_blocks(cfg: Config, refs, keys: bool, xd_smem, xo_smem, stage) -> list[Array]:
    """Every block of a ``bounded`` chunk (see :func:`chunk_cumsum`) on the tensor cores.

    Each 32-token diagonal block factors the decay through its 16th token ``m``: the left operand
    ``x_i e^{G_i - G_m}`` (``x`` = ``q``, or the read-side keys if ``keys``) and the right operand
    ``k_j e^{G_m - G_j}`` (``xd_smem``) have exponents in ``[-BOUNDED_SPAN, BOUNDED_SPAN]``. The
    lower-left 32x32 block factors through token 31: ``x_i e^{G_i - G_31}`` and ``k_j e^{G_31 - G_j}``
    (``xo_smem``), both at most one. Returns ``[C, 32]`` f32 products for columns 0-31 and 32-63 of
    the diagonal blocks (meaningful in their own block's rows; entries above the diagonal or across
    blocks may overflow and are discarded by the caller) and the lower-left block (rows 32-63).
    """
    q_smem, k_smem, e_smem, g_smem, marks_smem = refs
    shape = (CHUNK, _KC)
    second = _iota(shape, 0) >= _BLOCK
    halves = pl.ds(0, _BLOCK), pl.ds(_BLOCK, _BLOCK)

    def scoped(acc0, acc1, acc_off):
        def slab_step(slab, carry):
            cols, g, q, k, kr = _slab_inputs(cfg, slab, q_smem, k_smem, e_smem, g_smem)
            m15, m31, m47 = (_cols(_load(marks_smem.at[i, cols], _COL), shape) for i in (1, 3, 5))
            mid = jnp.where(second, m47, m15)
            to_31 = jnp.exp(jnp.where(second, g - m31, m31 - g))  # at most one in both halves
            xd_smem[...] = (k * jnp.exp(mid - g)).astype(_BF16)
            xo_smem[...] = (k * to_31).astype(_BF16)  # rows 0-31 are the lower-left block's right operand
            plgpu.commit_smem()
            x = kr if keys else q
            left = _Regs((x * jnp.exp(g - mid)).astype(_BF16))
            _wgmma(cfg, acc0, left, xd_smem.at[halves[0]].T, stage)
            _wgmma(cfg, acc1, left, xd_smem.at[halves[1]].T, stage)
            _wgmma(cfg, acc_off, _Regs(jnp.where(second, x * to_31, 0.0).astype(_BF16)), xo_smem.at[halves[0]].T,
                   stage)
            plgpu.wgmma_wait(0)  # no wgmma in flight across the loop back-edge (see _below_diagonal)
            return carry

        lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)
        return [acc0[...], acc1[...], acc_off[...]]

    return pl.run_scoped(scoped, *[plgpu.ACC((CHUNK, _BLOCK), _F32) for _ in range(3)])


def _store_factored(parts: list[Array], out_smem, *, strict: bool, row_scale: Array | None, scale: float) -> None:
    """Assemble :func:`_factored_blocks`' products into the ``[C, C]`` lower triangle ``out_smem``
    (diagonal included unless ``strict``), times ``row_scale`` (per-row) and ``scale``."""
    p0, p1, p_off = parts
    shape = (CHUNK, _BLOCK)
    rows, cols = _iota(shape, 0), _iota(shape, 1)
    second = rows >= _BLOCK
    for half, val in enumerate((
        jnp.where(second, p_off, jnp.where(rows > cols if strict else rows >= cols, p0, 0.0)),
        jnp.where(jnp.logical_and(second, rows > cols + _BLOCK if strict else rows >= cols + _BLOCK), p1, 0.0),
    )):
        if row_scale is not None:
            val = val * row_scale
        out_smem.at[:, pl.ds(half * _BLOCK, _BLOCK)][...] = (val * scale).astype(out_smem.dtype)


def _split_slabs(key_dim: int) -> tuple[range, range]:
    """Diagonal-pass slabs of warpgroup 0 and warpgroup 1."""
    count = key_dim // _KC
    first = (count + 2) // 4
    return range(0, first), range(first, count)


_L2_EPS = 1e-6  # that of jla.ops.reference.l2norm


def _normalize_rows(cfg: Config, x_smem) -> None:
    """``x <- x rsqrt(sum(x^2) + eps)`` per token in f32, in place (``reference.l2norm``)."""

    def square_step(slab, acc):
        x = _load(x_smem.at[:, pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)]).astype(_F32)
        return acc + jnp.sum(x * x, axis=1)

    zero = plgpu.layout_cast(jnp.zeros((CHUNK,), _F32), _ROW)
    inv = _rows(lax.rsqrt(lax.fori_loop(0, cfg.key_dim // _KC, square_step, zero) + _L2_EPS), (CHUNK, _KC))

    def scale_step(slab, carry):
        cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
        x_smem.at[:, cols][...] = (_load(x_smem.at[:, cols]).astype(_F32) * inv).astype(_BF16)
        return carry

    lax.fori_loop(0, cfg.key_dim // _KC, scale_step, None)


def _decay_queries(cfg: Config, q_smem, g_smem) -> None:
    """``q_smem <- scale q exp(G)`` (``qg``), in place, one slab at a time."""

    def slab_step(slab, carry):
        cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
        decay = jnp.exp(_load(g_smem.at[:, cols]))
        q_smem.at[:, cols][...] = (_load(q_smem.at[:, cols]).astype(_F32) * decay * cfg.scale).astype(_BF16)
        return carry

    lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)


def _decay_keys(cfg: Config, k_smem, g_smem, marks_smem, e_smem, kg_smem) -> None:
    """``k_smem <- -erase exp(G) k`` (``-kd``, in place) and ``kg_smem <- k exp(G_last - G)``."""
    shape = (CHUNK, _KC)
    beta = None if cfg.channel_gates else _rows(_load(e_smem, _ROW), shape)

    def slab_step(slab, carry):
        cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
        g = _load(g_smem.at[:, cols])
        k = _load(k_smem.at[:, cols]).astype(_F32)
        last = _cols(_load(marks_smem.at[_MARKS - 1, cols], _COL), shape)
        kg_smem.at[:, cols][...] = (k * jnp.exp(last - g)).astype(_BF16)
        erase = _load(e_smem.at[:, cols]).astype(_F32) if cfg.channel_gates else beta
        k_smem.at[:, cols][...] = (-(k * jnp.exp(g) * erase)).astype(_BF16)
        return carry

    lax.fori_loop(0, cfg.key_dim // _KC, slab_step, None)


def intra(
    cfg: Config, q: Array, k: Array, g_cum: Array, g_marks: Array, erase: Array, bounded: Array, *,
    decays: bool = False,
    solve: bool = False,
) -> tuple[Array, ...]:
    """``(Aqk, Akk)``: ``Aqk = scale * P`` (bf16) and ``Akk`` (f32), each ``[B, H, T, C]``.

    With ``solve`` the second output is ``T = (I + Akk)^-1`` (bf16, bit for bit :func:`solve`'s
    result) instead of ``Akk``: warpgroup 1 inverts the ``Akk`` assembled in SMEM, so ``Akk`` never
    reaches HBM and :func:`solve` does not run (not in the interpreter, whose wgmma cannot take a
    register operand in a warpgroup branch).

    ``P = tril(q exp(G) . k exp(-G))`` and ``Akk`` is its strictly lower analogue with the read-side
    keys: ``beta_i * sum_c k_ic k_jc e^{..}`` (KDA) or ``sum_c (b_i k_i)_c k_jc e^{..}`` (GDN-2).
    Inputs: ``q, k [B, T, H, K]`` bf16; ``g_cum [B, T, H, K]`` f32 (chunk-local cumulative
    log-decay) and ``g_marks [B, T / 8, H, K]`` (its value at every 8th token); ``erase`` is
    ``beta [B, H, T]`` f32 or ``b [B, T, H, K]``; ``bounded`` is :func:`chunk_cumsum`'s flag.
    With ``decays``, also ``(qg, -kd, kg)`` ``[B, H, T, K]`` bf16: ``qg = scale q exp(G)``, the
    negated read-side keys decayed from the chunk start ``kd = erase exp(G) k`` and ``kg = k
    exp(G_last - G)``, the inputs of :func:`state_output` with ``t`` (the forward without :func:`wy`).

    A ``bounded`` chunk runs every block on the tensor cores (:func:`_factored_blocks`): warpgroup
    0 computes and stores ``Aqk``, warpgroup 1 ``Akk``. Otherwise warpgroup 0 computes the blocks
    below the diagonal sub-chunk blocks and warpgroup 1 the lower-left quadrants of the diagonal
    blocks (both on the tensor cores); each then computes the exact 8x8 diagonal blocks (running
    products of the per-token decays) for part of the key channels, and warpgroup 0 assembles both.
    The decayed tiles overwrite ``q`` (warpgroup 0) and ``k`` (warpgroup 1, once warpgroup 0 has
    read ``k`` for the last time) in SMEM. With ``cfg.l2norm`` both tiles are first scaled to unit
    L2 norm per token in SMEM (warpgroup 0 ``q``, warpgroup 1 ``k``), as ``reference.l2norm``.
    """
    c, kd = CHUNK, cfg.key_dim
    tile = (c, SUB_CHUNK)
    if solve and cfg.interpret:
        raise ValueError("intra(solve=True) does not run in the interpreter: use solve() after intra()")
    out_qk = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _BF16)
    out_kk = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _BF16 if solve else _F32)
    out_d = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, kd), _BF16)
    outs = (out_qk, out_kk) + ((out_d,) * 3 if decays else ())
    wg_slabs = _split_slabs(kd)
    if bounded.shape != (cfg.batch, cfg.num_chunks, cfg.heads * kd // SCAN_WIDTH):
        raise ValueError(f"bounded must be chunk_cumsum's flag, got shape {bounded.shape}")

    def body(q_ref, k_ref, g_ref, gm_ref, e_ref, ok_ref, aqk_ref, akk_ref, *rest):
        decay_refs, scratch = (rest[:3], rest[3:]) if decays else ((), rest)
        q_smem, k_smem, g_smem, marks_smem, e_smem, x_smem, x2_smem, kpad, epad = scratch[:9]
        half_smem, diag_smem, aqk_smem, load_bar, done_bar, k_free, normed, finish = scratch[9:17]
        stage = scratch[17] if cfg.interpret else None  # register operands, interpret mode only
        akk_ready, hi_smem, lo_smem = scratch[17:20] if solve else (None,) * 3
        akk_smem = half_smem.at[1]  # assembled in place
        refs = (q_smem, k_smem, e_smem, g_smem, marks_smem)
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        wg = lax.axis_index("wg")
        tok = pl.ds(n * c, c)
        factored = ok_ref[b, n, (h * kd) // SCAN_WIDTH] != 0

        def store_queries():  # warpgroup 0, once warpgroup 1 no longer reads q
            if decays:
                _decay_queries(cfg, q_smem, g_smem)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(q_smem, decay_refs[0].at[b, h, tok])

        def keys_read():  # warpgroup 0 has read k for the last time
            if decays:
                plgpu.barrier_arrive(k_free)

        def store_keys(wait: bool = True):  # warpgroup 1
            if decays:
                plgpu.barrier_wait(k_free)
                _decay_keys(cfg, k_smem, g_smem, marks_smem, e_smem, x2_smem)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(k_smem, decay_refs[1].at[b, h, tok])
                plgpu.copy_smem_to_gmem(x2_smem, decay_refs[2].at[b, h, tok])
                if wait:
                    plgpu.wait_smem_to_gmem(0)

        def store_t():  # warpgroup 1, once Akk is complete in akk_smem
            # The solve's scratch reuses buffers that both paths are done with by now.
            t_val = _unit_lower_inverse(cfg, akk_smem, (diag_smem.at[1], half_smem.at[0], hi_smem, lo_smem))
            hi_smem[...] = t_val.astype(_BF16)  # the last product has read hi_smem
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(hi_smem, akk_ref.at[b, h, tok])

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, load_bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem, load_bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, load_bar)
            plgpu.copy_gmem_to_smem(gm_ref.at[b, pl.ds(n * _MARKS, _MARKS), h], marks_smem, load_bar)
            if cfg.channel_gates:
                plgpu.copy_gmem_to_smem(e_ref.at[b, tok, h], e_smem, load_bar)
            else:
                plgpu.copy_gmem_to_smem(e_ref.at[b, h, tok], e_smem, load_bar)

        plgpu.barrier_wait(load_bar)
        if cfg.l2norm:  # warpgroup 0 scales q, warpgroup 1 k; both then read both

            @pl.when(wg == 0)
            def _():
                _normalize_rows(cfg, q_smem)

            @pl.when(wg == 1)
            def _():
                _normalize_rows(cfg, k_smem)

            plgpu.barrier_arrive(normed)
            plgpu.barrier_wait(normed)
        slab_pair = pl.ds(0, _KC), pl.ds(_KC, _KC)  # per-slab right operands of the factored path

        @pl.when(jnp.logical_and(factored, wg == 0))
        def _():
            parts = _factored_blocks(cfg, refs, False, x_smem.at[:, slab_pair[0]], x_smem.at[:, slab_pair[1]],
                                     None if stage is None else stage.at[0])
            keys_read()
            _store_factored(parts, aqk_smem, strict=False, row_scale=None, scale=cfg.scale)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(aqk_smem, aqk_ref.at[b, h, tok])
            store_queries()
            plgpu.wait_smem_to_gmem(0)

        @pl.when(jnp.logical_and(factored, wg == 1))
        def _():
            parts = _factored_blocks(cfg, refs, True, x2_smem.at[:, slab_pair[0]], x2_smem.at[:, slab_pair[1]],
                                     None if stage is None else stage.at[1])
            beta = None if cfg.channel_gates else _rows(_load(e_smem, _ROW), (c, _BLOCK))
            _store_factored(parts, akk_smem, strict=True, row_scale=beta, scale=1.0)
            plgpu.commit_smem()
            if solve:
                store_keys(wait=False)  # the key stores drain during the solve
                store_t()
            else:
                plgpu.copy_smem_to_gmem(akk_smem, akk_ref.at[b, h, tok])
                store_keys()
            plgpu.wait_smem_to_gmem(0)

        @pl.when(jnp.logical_and(jnp.logical_not(factored), wg == 1))
        def _():
            _half_blocks(cfg, refs, x2_smem, half_smem, None if stage is None else stage.at[1])
            diag_q, diag_k = _diag_tiles(cfg, wg_slabs[1], refs, kpad.at[1], epad.at[1])
            diag_smem[0] = diag_q
            diag_smem[1] = diag_k
            plgpu.barrier_arrive(done_bar)
            store_keys(wait=not solve)
            if solve:
                plgpu.barrier_wait(akk_ready)  # warpgroup 0 has assembled Akk
                store_t()
                plgpu.wait_smem_to_gmem(0)

        @pl.when(jnp.logical_and(jnp.logical_not(factored), wg == 0))
        def _():
            below = _below_diagonal(cfg, refs, x_smem, None if stage is None else stage.at[0])
            if wg_slabs[0]:
                diags = _diag_tiles(cfg, wg_slabs[0], refs, kpad.at[0], epad.at[0])
            else:
                diags = (jnp.zeros(tile, _F32),) * 2
            keys_read()
            trow = _iota(tile, 0)
            blocks = trow >> 4
            quadrant = jnp.logical_and((trow & (SUB_CHUNK - 1)) >= _HALF, _iota(tile, 1) < _HALF)
            beta = None if cfg.channel_gates else _rows(_load(e_smem, _ROW), tile)
            plgpu.barrier_wait(done_bar)
            for idx, out_smem in enumerate((aqk_smem, akk_smem)):
                diag = diags[idx] + _load(diag_smem.at[idx])
                for col_blk in range(_NUM_SUB):
                    cols = pl.ds(SUB_CHUNK * col_blk, SUB_CHUNK)
                    quad = jnp.where(quadrant, _load(half_smem.at[idx, :, cols]), 0.0)
                    off = below[2 * col_blk + idx] if col_blk < _NUM_SUB - 1 else jnp.zeros(tile, _F32)
                    val = jnp.where(blocks == col_blk, diag + quad, off)
                    if idx == 0:
                        out_smem.at[:, cols][...] = (val * cfg.scale).astype(_BF16)
                    else:
                        out_smem.at[:, cols][...] = val if beta is None else val * beta
            plgpu.commit_smem()
            if solve:
                plgpu.barrier_arrive(akk_ready)
            plgpu.copy_smem_to_gmem(aqk_smem, aqk_ref.at[b, h, tok])
            if not solve:
                plgpu.copy_smem_to_gmem(akk_smem, akk_ref.at[b, h, tok])
            store_queries()
            plgpu.wait_smem_to_gmem(0)

        finish_together(finish)  # no warpgroup exits while the other may still issue wgmmas

    e_scratch = _slab_smem((c, kd), erase.dtype) if cfg.channel_gates else plgpu.SMEM((c,), _F32)
    scratch = (
        _slab_smem((c, kd), q.dtype),
        _slab_smem((c, kd), k.dtype),
        _slab_smem((c, kd), _F32),
        plgpu.SMEM((_MARKS, kd), _F32),
        e_scratch,
        _slab_smem((c, kd), _BF16),
        _slab_smem((c, kd), _BF16),
        plgpu.SMEM((2, _PAD + c, _KC + 8), _BF16),  # padded pitch: conflict-free row-shifted reads
        plgpu.SMEM((2, _PAD + c, _KC + 8), _F32),
        plgpu.SMEM((2, c, c), _F32),
        plgpu.SMEM((2, c, SUB_CHUNK), _F32),
        plgpu.SMEM((c, c), _BF16),
        plgpu.Barrier(num_arrivals=5),
        plgpu.Barrier(num_arrivals=1),  # exact path: warpgroup 1's diagonal tiles are in SMEM
        plgpu.Barrier(num_arrivals=1),  # decays: warpgroup 0 no longer reads k
        plgpu.Barrier(num_arrivals=2),  # l2norm: both tiles are scaled
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    if cfg.interpret:
        scratch += (_slab_smem((2, c, _KC), _BF16),)
    if solve:  # exact path: Akk assembled (by warpgroup 0); the solve's split-bf16 operands (and T)
        scratch += (plgpu.Barrier(num_arrivals=1), _swizzled((c, c)), _swizzled((c, c)))
    kern = _kernel(
        cfg, "vg_intra", body, out_type=outs, grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(q, k, g_cum, g_marks, erase, bounded)


# ---------------------------------------------------------------------------------------------
# solve: T = (I + Akk)^-1; wy: the WY factors, end-of-chunk keys and decayed queries
# ---------------------------------------------------------------------------------------------


def _split_bf16(x: Array) -> tuple[Array, Array]:
    """``x ~= hi + lo`` with both parts bf16 (about 16 significant bits together)."""
    hi = x.astype(_BF16)
    return hi, (x - hi.astype(_F32)).astype(_BF16)


def _precise_matmul(cfg: Config, x: Array, y: Array, hi_smem, lo_smem) -> Array:
    """``x @ y`` for f32 ``[C, C]`` register arrays from bf16 tensor-core products of split operands."""
    y_hi, y_lo = _split_bf16(y)
    hi_smem[...] = y_hi
    lo_smem[...] = y_lo
    plgpu.commit_smem()
    x_hi, x_lo = _split_bf16(x)
    return _matmul(cfg, (CHUNK, CHUNK), (_Regs(x_hi), hi_smem), (_Regs(x_hi), lo_smem), (_Regs(x_lo), hi_smem))


def _solve_scratch() -> tuple[plgpu.SMEM, ...]:
    c = CHUNK
    return (
        plgpu.SMEM((c, SUB_CHUNK), _F32),
        plgpu.SMEM((c, c), _F32),
        _swizzled((c, c)),
        _swizzled((c, c)),
    )


def _unit_lower_inverse(cfg: Config, a_smem, scratch) -> Array:
    """``(I + A)^-1`` for the strictly lower f32 ``[C, C]`` ``A`` in ``a_smem`` (f32 registers).

    The four 16x16 diagonal blocks are inverted exactly in f32 by right-looking elimination (all
    blocks at once, one pivot per step); the blocks are then merged twice, 16 -> 32 -> 64, with
    ``inv([[P, 0], [L, Q]]) = D - D L D`` for ``D = diag(inv(P), inv(Q))`` (the merge of FLA's ``solve_tril``).
    """
    s_smem, d1_smem, hi_smem, lo_smem = scratch
    c = CHUNK
    full = (c, c)
    tile = (c, SUB_CHUNK)
    trow = _iota(tile, 0)
    offset = trow & (SUB_CHUNK - 1)
    tblocks = trow >> 4
    eye = jnp.where(_iota(tile, 1) == offset, 1.0, 0.0)

    # Diagonal blocks stacked as [C, 16]: adiag[i, j] = A[i, 16 * block(i) + j].
    adiag = None
    for blk in range(_NUM_SUB):
        part = _load(a_smem.at[:, pl.ds(SUB_CHUNK * blk, SUB_CHUNK)])
        adiag = part if adiag is None else jnp.where(tblocks == blk, part, adiag)
    s_smem[...] = eye
    s = eye
    tcol = _iota(tile, 1)
    for j in range(SUB_CHUNK - 1):  # unrolled: static pivots, coefficients taken from registers
        # Row j of every block is final: S_i -= A[i, j] S_j for the rows below it.
        pivot = _per_block(tblocks, [_load(s_smem.at[SUB_CHUNK * blk + j], _COL) for blk in range(_NUM_SUB)], tile)
        coef = _rows(jnp.sum(jnp.where(tcol == j, adiag, 0.0), axis=1), tile)
        s = jnp.where(offset > j, s - coef * pivot, s)
        s_smem[...] = s
    for blk in range(_NUM_SUB):
        d1_smem.at[:, pl.ds(SUB_CHUNK * blk, SUB_CHUNK)][...] = jnp.where(tblocks == blk, s, 0.0)
    d1 = _load(d1_smem)
    a = _load(a_smem)
    rblocks = _iota(full, 0) >> 4
    cblocks = _iota(full, 1) >> 4
    # 16 -> 32: the (1, 0) and (3, 2) blocks.
    l1 = jnp.where(jnp.logical_and(rblocks - cblocks == 1, (rblocks & 1) == 1), a, 0.0)
    d2 = d1 - _precise_matmul(cfg, d1, _precise_matmul(cfg, l1, d1, hi_smem, lo_smem), hi_smem, lo_smem)
    # 32 -> 64: the lower-left 32x32 quadrant.
    l2 = jnp.where(jnp.logical_and(rblocks >= 2, cblocks < 2), a, 0.0)
    return d2 - _precise_matmul(cfg, d2, _precise_matmul(cfg, l2, d2, hi_smem, lo_smem), hi_smem, lo_smem)




def solve(cfg: Config, akk: Array) -> Array:
    """``T = (I + Akk)^-1`` ``[B, H, T, C]`` bf16 from :func:`intra`'s f32 ``Akk``, one chunk per CTA
    (about 60 KB of SMEM, so three CTAs share an SM)."""
    c = CHUNK
    out = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _BF16)

    def body(a_ref, t_ref, a_smem, t_smem, bar, *scratch):
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        plgpu.copy_gmem_to_smem(a_ref.at[b, h, tok], a_smem, bar)
        plgpu.barrier_wait(bar)
        t_smem[...] = _unit_lower_inverse(cfg, a_smem, scratch).astype(_BF16)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(t_smem, t_ref.at[b, h, tok])
        plgpu.wait_smem_to_gmem(0)

    scratch = (plgpu.SMEM((c, c), _F32), _swizzled((c, c)), plgpu.Barrier(num_arrivals=1), *_solve_scratch())
    kern = _kernel(
        cfg, "vg_solve", body, out_type=out, grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), scratch_types=scratch,
    )
    return kern(akk)


def wy(
    cfg: Config, t: Array, q: Array, k: Array, v: Array, g_cum: Array, g_marks: Array, erase: Array,
    write: Array,
) -> tuple[Array, Array, Array, Array]:
    """``(w, u, kg, qg)``: the WY factors ``w = T (erase exp(G) k)`` ``[B, H, T, K]`` and
    ``u = T (write v)`` ``[B, H, T, V]``, the end-of-chunk keys ``kg = k exp(G_last - G)`` and the
    left factor of the inter-chunk output ``qg = scale q exp(G)`` ``[B, H, T, K]``, all bf16.

    ``t`` is :func:`solve`'s ``T``, ``g_marks`` as in :func:`intra`; ``erase``/``write`` are
    ``beta [B, H, T]`` for KDA (pass the same array twice) or GDN-2's ``b [B, T, H, K]`` and
    ``w [B, T, H, V]`` (read straight from HBM). The results overwrite the input tiles in SMEM
    (about 90 KB per CTA, so two CTAs share an SM).
    """
    c, kd, vd = CHUNK, cfg.key_dim, cfg.value_dim
    shape_k = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, kd), _BF16)
    shape_v = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, vd), _BF16)
    if any(x.dtype != _BF16 for x in (q, k, v)):
        raise ValueError(f"wy expects bf16 q, k and v, got {q.dtype}, {k.dtype} and {v.dtype}")

    def body(t_ref, q_ref, k_ref, v_ref, g_ref, ge_ref, e_ref, wr_ref, w_ref, u_ref, kg_ref, qg_ref, *scratch):
        t_smem, q_smem, k_smem, v_smem, g_union, last_smem, beta_smem, bar = scratch
        g_smem, kg_smem = g_union  # kg replaces G once G is in registers
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        plgpu.copy_gmem_to_smem(t_ref.at[b, h, tok], t_smem, bar)
        plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, bar)
        plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem, bar)
        plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h], v_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, bar)
        plgpu.copy_gmem_to_smem(ge_ref.at[b, n * _MARKS + _MARKS - 1, h], last_smem, bar)
        if not cfg.channel_gates:
            plgpu.copy_gmem_to_smem(e_ref.at[b, h, tok], beta_smem, bar)
        plgpu.barrier_wait(bar)
        gf = _load(g_smem)
        decay = jnp.exp(gf)
        to_end = jnp.exp(_cols(_load(last_smem, _COL), (c, kd)) - gf)
        plgpu.commit_smem()  # every thread has read G
        if cfg.channel_gates:
            erase_v = _load(e_ref.at[b, tok, h]).astype(_F32)
            write_v = _load(wr_ref.at[b, tok, h]).astype(_F32)
        else:
            beta = _load(beta_smem, _ROW)
            erase_v, write_v = _rows(beta, (c, kd)), _rows(beta, (c, vd))
        kf = _load(k_smem).astype(_F32)
        kg_smem[...] = (kf * to_end).astype(_BF16)
        k_smem[...] = (kf * decay * erase_v).astype(_BF16)  # the right operands overwrite the inputs
        q_smem[...] = (_load(q_smem).astype(_F32) * decay * cfg.scale).astype(_BF16)
        v_smem[...] = (_load(v_smem).astype(_F32) * write_v).astype(_BF16)
        tm = _Regs(_load(t_smem))
        plgpu.commit_smem()
        w_val = _matmul(cfg, (c, kd), (tm, k_smem))
        u_val = _matmul(cfg, (c, vd), (tm, v_smem))
        plgpu.commit_smem()  # both products have read their right operands
        k_smem[...] = w_val.astype(_BF16)
        v_smem[...] = u_val.astype(_BF16)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(k_smem, w_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(v_smem, u_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(kg_smem, kg_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(q_smem, qg_ref.at[b, h, tok])
        plgpu.wait_smem_to_gmem(0)

    scratch = (
        _swizzled((c, c)),
        _swizzled((c, kd)),
        _swizzled((c, kd)),
        _swizzled((c, vd)),
        plgpu.RefUnion(_swizzled((c, kd), _F32), _swizzled((c, kd))),
        plgpu.SMEM((kd,), _F32),
        plgpu.SMEM((c,), _F32),
        plgpu.Barrier(num_arrivals=6 if cfg.channel_gates else 7),
    )
    kern = _kernel(
        cfg, "vg_wy", body, out_type=(shape_k, shape_v, shape_k, shape_k), grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), scratch_types=scratch,
    )
    return kern(t, q, k, v, g_cum, g_marks, erase, write)


# ---------------------------------------------------------------------------------------------
# state_output: the serial pass over chunks, fused with the output
# ---------------------------------------------------------------------------------------------


def _key_tiles(cfg: Config, shape: tuple[int, ...]) -> plgpu.SMEM:
    """bf16 operand ``[.., C, K]`` or ``[.., C, C]`` of the state kernels whose WGMMA partner is a
    ``block_v``-wide value-side tile: both need the same swizzle (that of a ``block_v`` row)."""
    swizzle = min(2 * cfg.block_v, 128)
    transforms = (plgpu.TilingTransform((8, swizzle // 2)), plgpu.SwizzleTransform(swizzle))
    return plgpu.SMEM(shape, _BF16, transforms=transforms)


def state_output(
    cfg: Config,
    kg: Array,
    w: Array,
    u: Array,
    g_last: Array,
    h0: Array | None,
    qg: Array,
    aqk: Array,
    *,
    save: bool,
    t: Array | None = None,
    write: Array | None = None,
    pieces: Array | None = None,
    out_rows: int = CHUNK,
) -> tuple[Array, ...]:
    """``(o, final)``, plus ``(h, v_new)`` for the backward with ``save``: the serial pass over the
    chunks fused with the output, one CTA of two warpgroups per ``(b, h, block_v value columns)``.

    With ``pieces`` (:func:`piece_table`, int32 ``[B * P, 4]``) each row's chain is cut at chunks
    where a segment starts, and one CTA per ``(b, piece, h, block_v columns)`` walks its piece: the
    same results, with ``P`` times the chains in flight (see :func:`piece_table`). With ``out_rows``
    below the chunk only the first ``out_rows`` rows of each chunk are outputs (RWKV-7's 64-row chunks:
    32 key rows, then 32 rows of its b-terms): ``o`` is ``[B, NC out_rows, H, V]``.

    Warpgroup 0 carries the state ``S`` in f32 registers: ``v_new = u - w S`` and
    ``S <- exp(G_last) S + kg^T v_new``. It publishes the state entering each chunk and the chunk's
    ``v_new`` (bf16 SMEM) to warpgroup 1, which computes ``o = qg S + Aqk v_new`` while warpgroup 0
    updates the state. With ``save`` both also go to ``h [B, H, NC, K, V]`` and
    ``v_new [B, H, T, V]`` (bf16).

    ``kg, w, qg`` are ``[B, H, T, K]`` and ``u`` ``[B, H, T, V]`` bf16, ``aqk [B, H, T, C]`` bf16,
    ``g_last [B, NC, H, K]`` f32 and ``h0 [B, H, K, V]`` f32 (``None``: a zero initial state).
    ``o`` is ``[B, T, H, V]`` bf16 and ``final [B, H, K, V]`` f32.

    With ``t`` (:func:`solve`'s ``T``) warpgroup 0 forms ``v_new = T (write v - kd S)`` itself, the
    WY products of :func:`wy` folded into the serial pass (FlashKDA's recurrence does the same with
    its 16-token inverse): ``w`` is then :func:`intra`'s ``-kd``, ``u`` the values ``v [B, T, H, V]``
    and ``write`` the write gate, ``beta [B, H, T]`` f32 or ``w [B, T, H, V]`` bf16.
    """
    c, kd, vd, bv, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.block_v, cfg.num_chunks
    fused = t is not None
    if fused != (write is not None):
        raise ValueError("state_output takes t and write together")
    out = [jax.ShapeDtypeStruct((cfg.batch, nc * out_rows, cfg.heads, vd), _BF16),
           jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32)]
    if save:
        out += [jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd, vd), _BF16),
                jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, vd), _BF16)]
    has_h0 = h0 is not None
    split = pieces is not None
    n_pieces = pieces.shape[0] // cfg.batch if split else 1
    extra_in = has_h0 + 2 * fused + split

    def stored(i, skip, store) -> None:  # chunk i of the walk stores its results unless it is skipped
        if split:
            pl.when(i >= skip)(store)
        else:
            store()

    def body(kg_ref, w_ref, u_ref, gl_ref, qg_ref, a_ref, *refs):
        h0_ref = refs[0] if has_h0 else None
        t_ref, wr_ref = refs[has_h0:has_h0 + 2] if fused else (None, None)
        outs = refs[extra_in:extra_in + len(out)]
        scratch = refs[extra_in + len(out):]
        (kg_smem, w_smem, u_smem, gl_smem, s_smem, vn_smem, qg_smem, a_smem, o_smem,
         bars, in_bars, ready, free, finish) = scratch[:14]
        t_smem, wr_smem = scratch[14:] if fused else (None, None)
        o_ref, final_ref = outs[:2]
        vt, h, bp = lax.axis_index("v"), lax.axis_index("h"), lax.axis_index("b")
        if split:  # walk chunks first .. first + count - 1; the first `skip` of them store nothing
            piece_ref = refs[extra_in - 1]
            b, piece = lax.div(bp, n_pieces), lax.rem(bp, n_pieces)
            first, count = piece_ref[bp, 0], piece_ref[bp, 1]
            skip, last_piece = piece_ref[bp, 2], piece_ref[bp, 3]
        else:
            b, piece, first, count, skip, last_piece = bp, 0, 0, nc, 0, 1
        wg = lax.axis_index("wg")
        vcols = pl.ds(vt * bv, bv)

        def chunk(i):  # the chunk of the walk's step i
            return first + i if split else i

        @pl.when(wg == 0 if not split else (wg == 0) & (count > 0))  # an empty piece walks nothing
        def _():
            def fetch(i, slot):
                n = chunk(i)
                tok = pl.ds(n * c, c)
                bar = bars.at[slot]
                plgpu.copy_gmem_to_smem(kg_ref.at[b, h, tok], kg_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(w_ref.at[b, h, tok], w_smem.at[slot], bar)
                if fused:
                    plgpu.copy_gmem_to_smem(u_ref.at[b, tok, h, vcols], u_smem.at[slot], bar)
                    plgpu.copy_gmem_to_smem(t_ref.at[b, h, tok], t_smem.at[slot], bar)
                    gate = wr_ref.at[b, tok, h, vcols] if cfg.channel_gates else wr_ref.at[b, h, tok]
                    plgpu.copy_gmem_to_smem(gate, wr_smem.at[slot], bar)
                else:
                    plgpu.copy_gmem_to_smem(u_ref.at[b, h, tok, vcols], u_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(gl_ref.at[b, n, h], gl_smem.at[slot], bar)

            fetch(0, 0)
            if has_h0:
                s0 = _load(h0_ref.at[b, h, :, vcols])
                if split:  # a later piece starts in a segment's first chunk: no state enters it
                    s0 = s0 * (piece == 0).astype(_F32)
            else:
                s0 = plgpu.layout_cast(jnp.zeros((kd, bv), _F32), _WG)

            def step(i, s):
                n = chunk(i)
                slot = lax.rem(i, 2)
                plgpu.commit_smem()

                @pl.when(i + 1 < count)
                def _():
                    fetch(i + 1, 1 - slot)

                plgpu.barrier_wait(bars.at[slot])

                @pl.when(i >= 1)
                def _():
                    plgpu.barrier_wait(free)  # warpgroup 1 is done with the previous chunk

                if save:
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's stores read them
                s_smem[...] = s.astype(_BF16)
                plgpu.commit_smem()
                if save:
                    stored(i, skip, lambda: plgpu.copy_smem_to_gmem(s_smem, outs[2].at[b, h, n, :, vcols],
                                                                    commit_group=False))
                if fused:
                    if cfg.channel_gates:
                        gate = _load(wr_smem.at[slot]).astype(_F32)
                    else:
                        gate = _rows(_load(wr_smem.at[slot], _ROW), (c, bv))
                    # write v + (-kd) S accumulates onto write v (an accumulator read as the addend
                    # of a multiply-add came out as zeros); it goes through vn_smem, which
                    # warpgroup 1 is done with (``free``).
                    written = _load(u_smem.at[slot]).astype(_F32) * gate
                    vn_smem[...] = _matmul(cfg, (c, bv), (w_smem.at[slot], s_smem), init=written).astype(_BF16)
                    plgpu.commit_smem()
                    v_new = _matmul(cfg, (c, bv), (t_smem.at[slot], vn_smem))
                else:
                    recalled = _matmul(cfg, (c, bv), (w_smem.at[slot], s_smem))
                    v_new = _load(u_smem.at[slot]).astype(_F32) - recalled
                vn_smem[...] = v_new.astype(_BF16)
                plgpu.commit_smem()
                if save:
                    def store_v_new():
                        plgpu.copy_smem_to_gmem(vn_smem, outs[3].at[b, h, pl.ds(n * c, c), vcols],
                                                commit_group=False)
                        plgpu.commit_smem_to_gmem_group()

                    stored(i, skip, store_v_new)
                plgpu.barrier_arrive(ready)
                decay = jnp.exp(_load(gl_smem.at[slot], _ROW))
                s = s * _rows(decay, (kd, bv))
                return _matmul(cfg, (kd, bv), (kg_smem.at[slot].T, vn_smem), init=s)

            s = lax.fori_loop(0, count, step, s0)

            def store_final():
                final_ref.at[b, h, :, vcols][...] = s

            if split:  # one piece per row (its last non-empty one) writes the final state
                pl.when(last_piece == 1)(store_final)
            else:
                store_final()
            plgpu.barrier_wait(free)  # observe warpgroup 1's last arrival
            if save:
                plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1 if not split else (wg == 1) & (count > 0))
        def _():
            def fetch(i, slot):
                tok = pl.ds(chunk(i) * c, c)
                plgpu.copy_gmem_to_smem(qg_ref.at[b, h, tok], qg_smem.at[slot], in_bars.at[slot])
                plgpu.copy_gmem_to_smem(a_ref.at[b, h, tok], a_smem.at[slot], in_bars.at[slot])

            fetch(0, 0)

            def step(i, carry):
                n = chunk(i)
                slot = lax.rem(i, 2)

                @pl.when(i + 1 < count)
                def _():
                    fetch(i + 1, 1 - slot)

                plgpu.barrier_wait(in_bars.at[slot])
                plgpu.barrier_wait(ready)

                def output():
                    o = _matmul(cfg, (c, bv), (qg_smem.at[slot], s_smem), (a_smem.at[slot], vn_smem))
                    plgpu.barrier_arrive(free)
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's store read o_smem
                    o_smem[...] = o.astype(_BF16)
                    plgpu.commit_smem()
                    rows = o_smem if out_rows == c else o_smem.at[pl.ds(0, out_rows)]
                    plgpu.copy_smem_to_gmem(rows, o_ref.at[b, pl.ds(n * out_rows, out_rows), h, vcols])

                stored(i, skip, output)
                if split:  # a skipped chunk's output is another piece's
                    pl.when(i < skip)(lambda: plgpu.barrier_arrive(free))
                return carry

            lax.fori_loop(0, count, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(finish)  # warpgroup 1 must not exit while warpgroup 0 updates the state

    scratch = (
        _key_tiles(cfg, (2, c, kd)),  # kg
        _key_tiles(cfg, (2, c, kd)),  # w
        _swizzled((2, c, bv)),  # u
        plgpu.SMEM((2, kd), _F32),  # cumulative decay at each chunk's last token
        _swizzled((kd, bv)),  # state entering the chunk (bf16)
        _swizzled((c, bv)),  # v_new
        _key_tiles(cfg, (2, c, kd)),  # qg
        _key_tiles(cfg, (2, c, c)),  # Aqk
        _swizzled((c, bv)),  # o
        plgpu.Barrier(num_arrivals=6 if fused else 4, num_barriers=2),
        plgpu.Barrier(num_arrivals=2, num_barriers=2),
        plgpu.Barrier(num_arrivals=1),  # chunk published by warpgroup 0
        plgpu.Barrier(num_arrivals=1),  # chunk consumed by warpgroup 1
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    if fused:
        gate_tiles = _swizzled((2, c, bv)) if cfg.channel_gates else plgpu.SMEM((2, c), _F32)
        scratch += (_key_tiles(cfg, (2, c, c)), gate_tiles)  # T, write gate
    kern = _kernel(
        cfg, "vg_state_output", body, out_type=tuple(out), grid=(vd // bv, cfg.heads, cfg.batch * n_pieces),
        grid_names=("v", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(kg, w, u, g_last, qg, aqk, *(() if h0 is None else (h0,)), *((t, write) if fused else ()),
                *((pieces,) if split else ()))


# One CTA of the state pass per SM (its tiles take most of the shared memory); H100 has 132 SMs.
STATE_PASS_SLOTS = 132
MAX_PIECES = 16


def num_pieces(cfg: Config) -> int:
    """Pieces per row for the state pass with packed documents: enough chains for one CTA per SM."""
    chains = (cfg.value_dim // cfg.block_v) * cfg.heads * cfg.batch
    return max(1, min(MAX_PIECES, STATE_PASS_SLOTS // chains, cfg.num_chunks))


def piece_table(kill: Array, n_pieces: int) -> Array:
    """The pieces of each row's serial chain, int32 ``[B * n_pieces, 4]``: (first chunk, chunk count,
    chunks skipped, writes the final state).

    ``kill [B, NC]`` marks the chunks where a segment starts. Such a chunk lets no state through:
    every factor of a term that crosses the segment start is exactly zero (KDA and GDN-2: the exp of
    an exponent at most -128, the reset decay of
    :func:`jla.ops.mosaic_gpu.vector_gate_ops.chunked_vector_gate`; RWKV-7: the exp(-1e4) carry decay
    and 0/1 document flags), so the state leaving the chunk, and the outputs of its tokens from the
    segment start on, are the same for any (finite) state entering it. A piece therefore
    starts at a kill chunk from a zero state, computes that chunk only for the state it passes on
    (it is skipped: the previous piece, which also walks it, stores its outputs), and walks on to
    the next piece's first chunk. Piece ``p`` starts at the first kill chunk at or after
    ``p NC / n_pieces`` (repeats dropped; piece 0 starts at chunk 0); unused pieces walk nothing.
    Every chunk's results are stored by exactly one piece, with the products of the serial pass.
    """
    batch, nc = kill.shape
    idx = jnp.arange(nc, dtype=jnp.int32)
    next_kill = lax.cummin(jnp.where(kill, idx, nc), axis=1, reverse=True)  # [B, NC]
    targets = (jnp.arange(1, n_pieces, dtype=jnp.int32) * nc) // n_pieces  # >= 1
    starts = next_kill[:, targets]  # [B, P - 1], nondecreasing, in [1, NC]
    repeat = starts == jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), starts[:, :-1]], axis=1)
    starts = jnp.sort(jnp.where(repeat, nc, starts), axis=1)
    first = jnp.concatenate([jnp.zeros((batch, 1), jnp.int32), starts], axis=1)  # [B, P]
    following = jnp.concatenate([first[:, 1:], jnp.full((batch, 1), nc, jnp.int32)], axis=1)
    live = first < nc
    count = jnp.where(live, jnp.minimum(following, nc - 1) - first + 1, 0)
    skip = (live & (jnp.arange(n_pieces) > 0)).astype(jnp.int32)
    last = (live & (following >= nc)).astype(jnp.int32)
    return jnp.stack([first, count, skip, last], axis=-1).reshape(batch * n_pieces, 4).astype(jnp.int32)
