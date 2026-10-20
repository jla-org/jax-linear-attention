# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2025 The JAX Authors.
# Portions Copyright 2025 DeepMind Technologies Limited.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/ragged_dot_mgpu.py,
# jax/experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.py and
# jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py (the warp-specialized pipeline; JAX 0.11.2),
# and from Tokamax (Apache-2.0), tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py and
# pallas_mosaic_gpu_common.py (the aligned tile schedule, skipped empty tiles, stratified grid).
"""Grouped GEMM (ragged dot) for mixture-of-experts on Mosaic GPU (H100, sm_90a).

``ragged_dot(lhs [M, K], rhs [G, K, N], group_sizes [G])`` computes
``out[r] = lhs[r] @ rhs[g(r)]`` where rows are sorted by group (group ``g`` owns
``group_sizes[g]`` consecutive rows), the semantics of ``jax.lax.ragged_dot``.
Rows past ``sum(group_sizes)`` are zero (``full_groups=True`` promises there are
none and skips that mask). Operands are bfloat16 or float16 with float32
accumulation; the output has the operand dtype.

Kernels:

* forward and ``d_lhs``: a persistent grouped GEMM over (row tile, column tile)
  pairs; ``d_lhs = ragged_dot(d_out, rhs^T)`` reuses it with the transposed weight
  read straight from ``[G, K, N]``. Two row schedules (:class:`TileConfig`):

  - ``"block_aligned"`` (:data:`BLOCK_ALIGNED`): the JAX example's schedule (row
    tiles aligned to ``block_m``, one extra tile per group boundary, a per-tile search over
    the groups, every tile runs its contraction even without rows);
  - ``"aligned"`` (:data:`ALIGNED`): each group's tiles start at its first row
    rounded down to a multiple of 8, so a group of ``s`` rows takes at most
    ``ceil((s + 7) / block_m)`` tiles instead of one extra tile per group boundary;
    tiles without rows (empty groups, the tail of the grid) are skipped, and the grid
    runs the tiles that certainly have rows first. A ``[5, G]`` per-group table from
    XLA lets a tile find its group with one compare per group. With ``warpgroups=2``
    (:data:`WARP_SPECIALIZED`) a TMA producer warpgroup feeds two consumer warpgroups;
  - ``"persistent"`` (:data:`PERSISTENT`, the default; :data:`PERSISTENT_WS` opt-in): the aligned
    schedule with one TMA pipeline per SM over all of its (tile, k) steps, so the next tile's
    loads are in flight during the current tile's epilogue;

* ``d_rhs``: the transposed ragged dot ``d_rhs[g] = lhs[rows g]^T @ d_out[rows g]``,
  which masks the rows of neighbouring groups in the first and last row block of
  every group and leaves empty groups at zero.

:func:`grouped_matmul` is the dispatch the MoE layer uses: ``impl="mosaic_gpu"``
runs the kernels when :func:`supports` accepts the shapes and falls back to
``jax.lax.ragged_dot`` otherwise (with a warning); ``impl="auto"`` (the MoE default) does the
same silently; ``impl="xla"`` always uses ``jax.lax.ragged_dot``.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from jla.ops.mosaic_gpu._common import is_hopper, num_sms

__all__ = ["ALIGNED", "BLOCK_ALIGNED", "CONFIGS", "IMPLS", "PERSISTENT", "PERSISTENT_WS", "WARP_SPECIALIZED", "TileConfig",
           "grouped_matmul", "ragged_dot", "select_config", "supports", "transposed_ragged_dot"]

Array = jax.Array

IMPLS = ("xla", "mosaic_gpu", "auto")
DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
BLOCK_M = 128  # rows of an output tile (forward / d_lhs) of the block-aligned configuration; the minimum M
BLOCK_N = 128  # columns of an output tile
BLOCK_K = 64  # contraction block
STEPS = 4  # pipeline depth
ALIGN = 128  # K and N must be multiples of this (covers every tile of the three GEMMs)
MAX_GROUPS = 512  # the block-aligned group search is unrolled per group
ROW_ALIGN = 8  # first row of an aligned-schedule tile: a multiple of this (TMA row tiling)
SMEM_BYTES = 227 * 1024  # H100 shared memory per block


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Tiling of the forward / ``d_lhs`` grouped GEMM.

    One warpgroup computes a ``block_m x block_n`` float32 tile (at most 128 registers
    per thread: ``block_m * block_n <= 16384``) over ``block_k``-wide contraction steps,
    ``steps`` of them in flight. ``grid_block_n`` column tiles are visited per row tile
    sweep (planar snake). ``schedule`` picks the row schedule (see the module docstring).
    ``TileConfig()`` is :data:`BLOCK_ALIGNED`, the block-aligned tiling (the default is :data:`PERSISTENT`).
    """

    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    block_k: int = BLOCK_K
    steps: int = STEPS
    grid_block_n: int = 2
    schedule: str = "block_aligned"
    # 2: two consumer warpgroups (each block_m x block_n of a block_m x 2 block_n tile) fed by a
    # TMA producer warpgroup (aligned schedule only).
    warpgroups: int = 1

    def __post_init__(self) -> None:
        if self.block_m not in (64, 128) or self.block_n not in (64, 128, 256) or self.block_k not in (64, 128):
            raise ValueError(f"unsupported ragged_dot tile {self}")
        if self.block_m * self.block_n > 128 * 128:
            raise ValueError(f"tile {self.block_m}x{self.block_n} exceeds one warpgroup's registers")
        if self.steps < 2 or self.grid_block_n < 1:
            raise ValueError(f"ragged_dot needs steps >= 2 and grid_block_n >= 1, got {self}")
        if self.schedule not in ("aligned", "block_aligned", "persistent"):
            raise ValueError(f"unknown ragged_dot schedule {self.schedule!r}")
        if self.warpgroups not in (1, 2) or (self.warpgroups == 2 and self.schedule == "block_aligned"):
            raise ValueError(f"ragged_dot warpgroups must be 1, or 2 with schedule 'aligned' or 'persistent': {self}")
        if self.smem_bytes() > SMEM_BYTES:
            raise ValueError(f"tile {self} needs {self.smem_bytes()} bytes of shared memory (> {SMEM_BYTES})")

    @property
    def tile_n(self) -> int:
        """Output columns of one CTA tile."""
        return self.block_n * self.warpgroups

    def smem_bytes(self, itemsize: int = 2) -> int:
        stage = (self.block_m * self.block_k + self.block_k * self.tile_n) * itemsize
        return self.steps * stage + self.block_m * self.tile_n * itemsize

    def divides(self, k: int, n: int) -> bool:
        return k % self.block_k == 0 and n % self.tile_n == 0

    def divides_both(self, k: int, n: int) -> bool:
        """Whether it tiles the forward (contraction ``k``, width ``n``) and ``d_lhs`` (``n``, ``k``)."""
        return self.divides(k, n) and self.divides(n, k)


BLOCK_ALIGNED = TileConfig()  # the JAX example's schedule: 128 x 128 x 64, 4 steps, rows aligned to block_m
ALIGNED = TileConfig(schedule="aligned", steps=5)  # aligned schedule, 128 x 128 x 64, 5 steps
# Opt-in: TMA producer + two consumer warpgroups, 128 x 256 tiles; needs N and K multiples of 256.
WARP_SPECIALIZED = TileConfig(schedule="aligned", warpgroups=2, steps=3)
# The default: the aligned schedule with one pipeline per SM across its tiles (no per-tile fill and drain).
PERSISTENT = TileConfig(schedule="persistent", steps=5)
# Opt-in: the persistent schedule with a TMA producer and two consumer warpgroups (128 x 256 tiles).
PERSISTENT_WS = TileConfig(schedule="persistent", warpgroups=2, steps=3)
CONFIGS = (PERSISTENT,)  # every tiling select_config returns (each one is GPU-tested)


def select_config(k: int, n: int) -> TileConfig:
    """Default forward / ``d_lhs`` tiling for contraction ``k`` and output width ``n``: :data:`PERSISTENT`.

    Chosen by measurement (``benchmarks/ragged_dot.md``, ``benchmarks/profile_moe.py``): the MoE
    layer forward + backward of the shipped MoE presets is faster with it than with :data:`ALIGNED`
    (the aligned schedule with per-tile pipelines), :data:`PERSISTENT_WS` and :data:`BLOCK_ALIGNED` (the
    JAX example's schedule), e.g. ``qwen3-next-300m`` 3.23 / 3.58 / 3.37 / 3.78 ms.
    """
    return CONFIGS[0]


def _group_info(sizes_ref, groups: int, tile: int, tid):
    """Group, row block and valid row range of persistent row tile ``tid`` (block-aligned schedule).

    Group ``i`` owns the row tiles ``[start_block_i + i, final_block_i + i]``: one tile per
    row block it touches, shifted by one per earlier group boundary. Tiles past the last
    group (the grid allows one extra tile per group) get an empty row range.
    """
    tile = jnp.int32(tile)
    group_end = group_start = block = group = end = jnp.array(0, dtype=jnp.int32)
    for i in range(groups):
        start = end
        end = start + sizes_ref[i]
        start_block = lax.div(start, tile)
        final_block = lax.div(end - 1, tile)
        tid_begin = start_block + i
        tid_end = final_block + 1 + i
        hit = (tid_begin <= tid) & (tid < tid_end)
        block = lax.select(hit, tid - tid_begin + start_block, block)
        group = lax.select(hit, jnp.int32(i), group)
        group_start = lax.select(hit, start, group_start)
        group_end = lax.select(hit, end, group_end)
    block_start = block * tile
    actual_start = jnp.maximum(group_start, block_start)
    actual_end = jnp.minimum(group_end, block_start + tile)
    return group, block, block_start, actual_start - block_start, actual_end - actual_start


def _store_rows(acc, o_gmem, *, row0, start_within, size, max_rows: int, columns):
    """Store rows ``[start_within, start_within + size)`` of tile ``acc`` at ``o_gmem[row0 + ...]``."""

    @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM(acc.shape, dtype=o_gmem.dtype))
    def _store(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()
        # TMA boxes have static shapes: store the dynamic row count as a binary ladder
        # of power-of-two row slices (upstream's scheme).
        smem_start = start_within
        remaining = max_rows
        while remaining > 0:
            rows = 1 << (remaining.bit_length() - 1)
            remaining //= 2

            @pl.when(size & rows != 0)
            def _():
                plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(smem_start, rows)],
                                        o_gmem.at[pl.ds(row0 + smem_start, rows), columns])

            smem_start += size & rows
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)


def _ragged_dot_kernel(lhs: Array, rhs: Array, group_sizes: Array, *, transpose_rhs: bool,
                       cfg: TileConfig = BLOCK_ALIGNED) -> Array:
    """``lhs [M, K]`` x ``rhs [G, K, N]`` (or ``[G, N, K]`` with ``transpose_rhs``) -> ``[M, N]``."""
    if cfg.schedule == "aligned" and cfg.warpgroups == 2:
        return _ragged_dot_kernel_ws(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, cfg=cfg)
    if cfg.schedule == "persistent":
        persistent = _ragged_dot_kernel_ws_persistent if cfg.warpgroups == 2 else _ragged_dot_kernel_persistent
        return persistent(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, cfg=cfg)
    if cfg.schedule == "aligned":
        return _ragged_dot_kernel_aligned(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, cfg=cfg)
    m, k = lhs.shape
    g = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    block_m, block_n, block_k = cfg.block_m, cfg.block_n, cfg.block_k
    grid_m = pl.cdiv(m, block_m) + g - 1
    grid_n = n // block_n
    grid_block_n = cfg.grid_block_n if grid_n % cfg.grid_block_n == 0 else 1

    def body(sizes_gmem, lhs_gmem, rhs_gmem, o_gmem):
        @plgpu.nd_loop((grid_m * grid_n,), collective_axes="sm")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (grid_m, grid_n), 1, grid_block_n)
            group, block, block_start, start_within_block, actual_size = _group_info(
                sizes_gmem, g, block_m, mi)

            def acc_scope(acc_ref):
                plgpu.emit_pipeline(
                    lambda _, lhs_smem, rhs_smem: plgpu.wgmma(
                        acc_ref, lhs_smem, rhs_smem.transpose((1, 0)) if transpose_rhs else rhs_smem),
                    grid=(k // block_k,),
                    in_specs=[
                        plgpu.BlockSpec((block_m, block_k), lambda kk: (block, kk), delay_release=1),
                        plgpu.BlockSpec(
                            (block_n, block_k) if transpose_rhs else (block_k, block_n),
                            (lambda kk: (ni, kk)) if transpose_rhs else (lambda kk: (kk, ni)),
                            delay_release=1,
                        ),
                    ],
                    max_concurrent_steps=cfg.steps,
                )(lhs_gmem, rhs_gmem.at[group])
                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))
            _store_rows(acc, o_gmem, row0=block_start, start_within=start_within_block, size=actual_size,
                        max_rows=min(block_m, m), columns=pl.ds(ni * block_n, block_n))

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(num_sms(),),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(group_sizes, lhs, rhs)


def _group_table(group_sizes: Array, block_m: int) -> Array:
    """``[5, G]`` int32 per group: first tile, end tile, first row, end row, first row rounded down to 8.

    Group ``g`` (rows ``[s_g, e_g)``) owns tiles ``[t_g, t_g + ceil((e_g - floor8(s_g)) / block_m))``
    (none when empty) of the aligned schedule. G-sized, so XLA computes it in a couple of tiny
    kernels; :func:`_tile_info` then finds a tile's group with one compare per group.
    """
    sizes = jnp.maximum(group_sizes.astype(jnp.int32), 0)
    ends = jnp.cumsum(sizes, dtype=jnp.int32)
    starts = ends - sizes
    first_rows = starts // ROW_ALIGN * ROW_ALIGN
    blocks = jnp.where(sizes == 0, 0, (ends - first_rows + block_m - 1) // block_m)
    tile_ends = jnp.cumsum(blocks, dtype=jnp.int32)
    return jnp.stack([tile_ends - blocks, tile_ends, starts, ends, first_rows]).astype(jnp.int32)


def _tile_info(table_ref, groups: int, block_m: int, tid):
    """Group, first row (a multiple of 8), offset of the first valid row and valid rows of tile ``tid``.

    The group is the number of groups whose tiles end at or before ``tid``; tiles past the last
    group fall in the last group's range past its rows and get no valid rows.
    """
    group = jnp.array(0, dtype=jnp.int32)
    for i in range(groups - 1):
        group = group + (table_ref[1, i] <= tid).astype(jnp.int32)
    r0 = table_ref[4, group] + (tid - table_ref[0, group]) * block_m
    lo = jnp.maximum(r0, table_ref[2, group])
    hi = jnp.minimum(r0 + block_m, table_ref[3, group])
    size = jnp.maximum(hi - lo, 0)
    return group, lax.select(size > 0, r0, jnp.int32(0)), lax.select(size > 0, lo - r0, jnp.int32(0)), size


def _ragged_dot_kernel_aligned(lhs: Array, rhs: Array, group_sizes: Array, *, transpose_rhs: bool,
                               cfg: TileConfig) -> Array:
    m, k = lhs.shape
    g = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    block_m, block_n, block_k = cfg.block_m, cfg.block_n, cfg.block_k
    tiles = pl.cdiv(m, block_m) + g - 1
    busy = pl.cdiv(m, block_m)  # with every row in a group, these tiles all have rows
    grid_n = n // block_n
    grid_block_n = cfg.grid_block_n if grid_n % cfg.grid_block_n == 0 else 1

    def body(table_gmem, lhs_gmem, rhs_gmem, o_gmem):
        def tile(m_offset: int, m_tiles: int, loop_info: plgpu.NDLoopInfo):
            mi, ni = plgpu.planar_snake(loop_info.index[0], (m_tiles, grid_n), 1, grid_block_n)
            mi = mi + m_offset
            group, row0, start_within, size = _tile_info(table_gmem, g, block_m, mi)
            row0 = pl.multiple_of(row0, ROW_ALIGN)

            @pl.when(size > 0)
            def _():
                def acc_scope(acc_ref):
                    plgpu.emit_pipeline(
                        lambda _, lhs_smem, rhs_smem: plgpu.wgmma(
                            acc_ref, lhs_smem, rhs_smem.transpose((1, 0)) if transpose_rhs else rhs_smem),
                        grid=(k // block_k,),
                        in_specs=[
                            # Element rows: the tile starts at any multiple of 8 (rows past M read zeros).
                            plgpu.BlockSpec((pl.Element(block_m), block_k), lambda kk: (row0, kk),
                                            delay_release=1),
                            plgpu.BlockSpec(
                                (block_n, block_k) if transpose_rhs else (block_k, block_n),
                                (lambda kk: (ni, kk)) if transpose_rhs else (lambda kk: (kk, ni)),
                                delay_release=1,
                            ),
                        ],
                        max_concurrent_steps=cfg.steps,
                    )(lhs_gmem, rhs_gmem.at[group])
                    return acc_ref[...]

                acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))
                _store_rows(acc, o_gmem, row0=row0, start_within=start_within, size=size,
                            max_rows=min(block_m, m), columns=pl.ds(ni * block_n, block_n))

        # Tiles that certainly have rows first, then the ones that may be empty, so that the
        # empty ones do not pile up on a few SMs.
        plgpu.nd_loop((busy * grid_n,), collective_axes="sm")(functools.partial(tile, 0, busy))
        if tiles > busy:
            plgpu.nd_loop(((tiles - busy) * grid_n,), collective_axes="sm")(
                functools.partial(tile, busy, tiles - busy))

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(num_sms(),),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(_group_table(group_sizes, block_m), lhs, rhs)


def _ragged_dot_kernel_ws(lhs: Array, rhs: Array, group_sizes: Array, *, transpose_rhs: bool,
                          cfg: TileConfig) -> Array:
    """Aligned schedule with warp specialization (JAX's ``hopper_matmul_mgpu.py`` structure).

    A producer warpgroup streams ``[block_m, block_k]`` row blocks and ``[block_k, 2 block_n]``
    weight blocks through a ``cfg.steps``-deep TMA pipeline; two consumer warpgroups each
    accumulate one ``block_m x block_n`` half and store its valid rows.
    """
    m, k = lhs.shape
    g = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    block_m, block_n, block_k = cfg.block_m, cfg.block_n, cfg.block_k
    tile_n = cfg.tile_n
    tiles = pl.cdiv(m, block_m) + g - 1
    busy = pl.cdiv(m, block_m)
    grid_n = n // tile_n
    grid_block_n = cfg.grid_block_n if grid_n % cfg.grid_block_n == 0 else 1
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(lhs.dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

    def body(table_gmem, lhs_gmem, rhs_gmem, o_gmem):
        wg = lax.axis_index("wg")
        wg_cols = pl.ds(wg * block_n, block_n)

        def rhs_view(group, ni):
            cols = pl.ds(ni * tile_n, tile_n)
            return rhs_gmem.at[group, cols, :] if transpose_rhs else rhs_gmem.at[group, :, cols]

        def get_pipeline(pipeline_body, compute_context):
            return plgpu.emit_pipeline_warp_specialized(
                pipeline_body,
                grid=(k // block_k,),
                memory_registers=40,
                in_specs=[
                    plgpu.BlockSpec((block_m, block_k), lambda kk: (0, kk), transforms=transforms,
                                    memory_space=plgpu.SMEM, delay_release=1),
                    plgpu.BlockSpec((tile_n, block_k) if transpose_rhs else (block_k, tile_n),
                                    (lambda kk: (0, kk)) if transpose_rhs else (lambda kk: (kk, 0)),
                                    transforms=transforms, memory_space=plgpu.SMEM, delay_release=1),
                ],
                wg_axis="wg",
                num_compute_wgs=2,
                max_concurrent_steps=cfg.steps,
                compute_context=compute_context,
            )

        ignore = lambda *_, **__: None  # noqa: E731 - the bodies do not change the allocations

        @functools.partial(
            pl.run_scoped,
            allocs=get_pipeline(ignore, ignore).get_allocations(lhs_gmem.at[pl.ds(0, block_m), :], rhs_view(0, 0)),
            out_smem=plgpu.SMEM((2, block_m, block_n), o_gmem.dtype),
            collective_axes="wg",
        )
        def _scope(allocs, out_smem):
            def tile(m_offset: int, m_tiles: int, loop_info: plgpu.NDLoopInfo):
                mi, ni = plgpu.planar_snake(loop_info.index[0], (m_tiles, grid_n), 1, grid_block_n)
                mi = mi + m_offset
                group, row0, start_within, size = _tile_info(table_gmem, g, block_m, mi)
                row0 = pl.multiple_of(row0, ROW_ALIGN)

                @pl.when(size > 0)
                def _():
                    def compute_context(eval_pipeline):
                        @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((block_m, block_n), jnp.float32))
                        def _acc(acc_ref):
                            eval_pipeline(acc_ref)
                            acc = acc_ref[...]
                            o_smem = out_smem.at[wg]
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the last tile's stores read it
                            o_smem[...] = acc.astype(o_smem.dtype)
                            plgpu.commit_smem()
                            columns = pl.ds(ni * tile_n + wg * block_n, block_n)
                            smem_start = start_within
                            remaining = min(block_m, m)
                            while remaining > 0:
                                rows = 1 << (remaining.bit_length() - 1)
                                remaining //= 2

                                @pl.when(size & rows != 0)
                                def _():
                                    plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(smem_start, rows)],
                                                            o_gmem.at[pl.ds(row0 + smem_start, rows), columns])

                                smem_start += size & rows

                    def mma(_, lhs_smem, rhs_smem, acc_ref):
                        b = rhs_smem.at[wg_cols].transpose((1, 0)) if transpose_rhs else rhs_smem.at[:, wg_cols]
                        plgpu.wgmma(acc_ref, lhs_smem, b)
                        plgpu.wgmma_wait(1)
                        return acc_ref

                    # Rows past M (the last tile) read zeros and are never stored.
                    get_pipeline(mma, compute_context)(lhs_gmem.at[pl.ds(row0, block_m), :], rhs_view(group, ni),
                                                       allocations=allocs)

            plgpu.nd_loop((busy * grid_n,), collective_axes="sm")(functools.partial(tile, 0, busy))
            if tiles > busy:
                plgpu.nd_loop(((tiles - busy) * grid_n,), collective_axes="sm")(
                    functools.partial(tile, busy, tiles - busy))
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(num_sms(),),
        grid_names=("sm",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(_group_table(group_sizes, block_m), lhs, rhs)


def _row_tile_table(group_sizes: Array, block_m: int, tiles: int) -> tuple[Array, Array]:
    """Aligned-schedule row tiles ``[4, tiles]`` int32 (group, first row, offset of the first valid
    row, valid rows) and ``[1]`` int32 the number of tiles with rows (a prefix), from XLA."""
    starts_t, ends_t, starts, ends, first_rows = _group_table(group_sizes, block_m)
    tid = jnp.arange(tiles, dtype=jnp.int32)
    group = jnp.sum(ends_t[None, :-1] <= tid[:, None], axis=1, dtype=jnp.int32)
    r0 = first_rows[group] + (tid - starts_t[group]) * block_m
    lo = jnp.maximum(r0, starts[group])
    size = jnp.maximum(jnp.minimum(r0 + block_m, ends[group]) - lo, 0)
    table = jnp.stack([group, jnp.where(size > 0, r0, 0), jnp.where(size > 0, lo - r0, 0), size])
    return table.astype(jnp.int32), ends_t[-1:].astype(jnp.int32)


def _ragged_dot_kernel_persistent(lhs: Array, rhs: Array, group_sizes: Array, *, transpose_rhs: bool,
                                  cfg: TileConfig) -> Array:
    """Aligned schedule with one TMA pipeline per SM across all its (tile, contraction step) pairs.

    SM ``s`` runs the work items ``s, s + P, ...`` (``P`` SMs; item ``w`` = row tile ``w // grid_n``,
    column tile ``w % grid_n``, over the row tiles that have rows) as one ``emit_pipeline`` over
    ``(items, K / block_k)`` steps: the loads of a tile's first steps are in flight while the previous
    tile's accumulator is stored (the per-tile pipelines of the other schedules fill and drain once per
    tile, which costs the most at short contractions such as a 256-wide expert's down projection).
    """
    m, k = lhs.shape
    g = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    block_m, block_n, block_k = cfg.block_m, cfg.block_n, cfg.block_k
    tiles = pl.cdiv(m, block_m) + g - 1
    grid_n = n // block_n
    k_steps = k // block_k
    sms = num_sms()
    table, count = _row_tile_table(group_sizes, block_m, tiles)

    def body(table_gmem, count_gmem, lhs_gmem, rhs_gmem, o_gmem):
        sm = lax.axis_index("sm")
        work = count_gmem[0] * grid_n
        mine = lax.div(jnp.maximum(work - sm, 0) + (sms - 1), jnp.int32(sms))

        def item(i):
            w = sm + i * sms
            return lax.div(w, jnp.int32(grid_n)), lax.rem(w, jnp.int32(grid_n))

        def lhs_map(i, kk):
            return pl.multiple_of(table_gmem[1, item(i)[0]], ROW_ALIGN), kk

        def rhs_map(i, kk):
            mi, ni = item(i)
            return (table_gmem[0, mi], ni, kk) if transpose_rhs else (table_gmem[0, mi], kk, ni)

        def scope(acc_ref, o_smem):
            def step(idx, lhs_smem, rhs_smem):
                i, kk = idx

                @pl.when(kk == 0)
                def _():
                    acc_ref[...] = jnp.zeros((block_m, block_n), jnp.float32)

                plgpu.wgmma(acc_ref, lhs_smem, rhs_smem.transpose((1, 0)) if transpose_rhs else rhs_smem)

                @pl.when(kk == k_steps - 1)
                def _():
                    mi, ni = item(i)
                    acc = acc_ref[...]
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous tile's stores read o_smem
                    o_smem[...] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()
                    row0 = pl.multiple_of(table_gmem[1, mi], ROW_ALIGN)
                    start, size = table_gmem[2, mi], table_gmem[3, mi]
                    columns = pl.ds(ni * block_n, block_n)
                    smem_start = start
                    remaining = min(block_m, m)
                    while remaining > 0:  # TMA boxes are static: a power-of-two ladder of row slices
                        rows = 1 << (remaining.bit_length() - 1)
                        remaining //= 2

                        @pl.when(size & rows != 0)
                        def _():
                            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(smem_start, rows)],
                                                    o_gmem.at[pl.ds(row0 + smem_start, rows), columns])

                        smem_start += size & rows

            @pl.when(mine > 0)
            def _():
                plgpu.emit_pipeline(
                    step,
                    grid=(mine, k_steps),
                    in_specs=[
                        # Element rows: a tile starts at any multiple of 8 (rows past M read zeros).
                        plgpu.BlockSpec((pl.Element(block_m), block_k), lhs_map, delay_release=1),
                        plgpu.BlockSpec((None, block_n, block_k) if transpose_rhs else (None, block_k, block_n),
                                        rhs_map, delay_release=1),
                    ],
                    max_concurrent_steps=cfg.steps,
                )(lhs_gmem, rhs_gmem)

            plgpu.wait_smem_to_gmem(0)

        pl.run_scoped(scope, plgpu.ACC((block_m, block_n)), plgpu.SMEM((block_m, block_n), o_gmem.dtype))

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(table, count, lhs, rhs)


def _ragged_dot_kernel_ws_persistent(lhs: Array, rhs: Array, group_sizes: Array, *, transpose_rhs: bool,
                                     cfg: TileConfig) -> Array:
    """The persistent schedule with warp specialization: a TMA producer warpgroup streams every
    (tile, contraction step) of the SM's work items through one pipeline, and two consumer
    warpgroups each accumulate and store one ``block_m x block_n`` half of a ``block_m x 2 block_n`` tile."""
    m, k = lhs.shape
    g = rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    block_m, block_n, block_k = cfg.block_m, cfg.block_n, cfg.block_k
    tile_n = cfg.tile_n
    tiles = pl.cdiv(m, block_m) + g - 1
    grid_n = n // tile_n
    k_steps = k // block_k
    sms = num_sms()
    table, count = _row_tile_table(group_sizes, block_m, tiles)
    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(lhs.dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

    def body(table_gmem, count_gmem, lhs_gmem, rhs_gmem, o_gmem):
        wg = lax.axis_index("wg")
        sm = lax.axis_index("sm")
        work = count_gmem[0] * grid_n
        mine = lax.div(jnp.maximum(work - sm, 0) + (sms - 1), jnp.int32(sms))
        wg_cols = pl.ds(wg * block_n, block_n)

        def item(i):
            w = sm + i * sms
            return lax.div(w, jnp.int32(grid_n)), lax.rem(w, jnp.int32(grid_n))

        def lhs_map(i, kk):
            return pl.multiple_of(table_gmem[1, item(i)[0]], ROW_ALIGN), kk

        def rhs_map(i, kk):
            mi, ni = item(i)
            return (table_gmem[0, mi], ni, kk) if transpose_rhs else (table_gmem[0, mi], kk, ni)

        @functools.partial(pl.run_scoped, out_smem=plgpu.SMEM((2, block_m, block_n), o_gmem.dtype),
                           collective_axes="wg")
        def _scope(out_smem):
            def compute_context(eval_pipeline):
                @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((block_m, block_n), jnp.float32))
                def _acc(acc_ref):
                    eval_pipeline(acc_ref)

            def mma(idx, lhs_smem, rhs_smem, acc_ref):
                i, kk = idx

                @pl.when(kk == 0)
                def _():
                    acc_ref[...] = jnp.zeros((block_m, block_n), jnp.float32)

                b = rhs_smem.at[wg_cols].transpose((1, 0)) if transpose_rhs else rhs_smem.at[:, wg_cols]
                plgpu.wgmma(acc_ref, lhs_smem, b)
                plgpu.wgmma_wait(1)

                @pl.when(kk == k_steps - 1)
                def _():
                    mi, ni = item(i)
                    acc = acc_ref[...]
                    o_smem = out_smem.at[wg]
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous tile's stores read it
                    o_smem[...] = acc.astype(o_smem.dtype)
                    plgpu.commit_smem()
                    row0 = pl.multiple_of(table_gmem[1, mi], ROW_ALIGN)
                    start, size = table_gmem[2, mi], table_gmem[3, mi]
                    columns = pl.ds(ni * tile_n + wg * block_n, block_n)
                    smem_start = start
                    remaining = min(block_m, m)
                    while remaining > 0:
                        rows = 1 << (remaining.bit_length() - 1)
                        remaining //= 2

                        @pl.when(size & rows != 0)
                        def _():
                            plgpu.copy_smem_to_gmem(o_smem.at[pl.ds(smem_start, rows)],
                                                    o_gmem.at[pl.ds(row0 + smem_start, rows), columns])

                        smem_start += size & rows

                return acc_ref

            pipeline = plgpu.emit_pipeline_warp_specialized(
                mma,
                grid=(mine, k_steps),
                memory_registers=40,
                in_specs=[
                    plgpu.BlockSpec((pl.Element(block_m), block_k), lhs_map, transforms=transforms,
                                    memory_space=plgpu.SMEM, delay_release=1),
                    plgpu.BlockSpec((None, tile_n, block_k) if transpose_rhs else (None, block_k, tile_n), rhs_map,
                                    transforms=transforms, memory_space=plgpu.SMEM, delay_release=1),
                ],
                wg_axis="wg",
                num_compute_wgs=2,
                max_concurrent_steps=cfg.steps,
                compute_context=compute_context,
            )

            @pl.when(mine > 0)
            def _():
                pipeline(lhs_gmem, rhs_gmem)

            plgpu.wait_smem_to_gmem(0)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((m, n), lhs.dtype),
        grid=(sms,),
        grid_names=("sm",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(table, count, lhs, rhs)


def transposed_ragged_dot(lhs: Array, rhs: Array, group_sizes: Array) -> Array:
    """``lhs [R, M]``, ``rhs [R, N]`` (rows grouped) -> ``[G, M, N]``, ``out[g] = lhs_g^T @ rhs_g``."""
    rows, m = lhs.shape
    _, n = rhs.shape
    g = group_sizes.shape[0]
    block_m, block_n, block_k = BLOCK_M, BLOCK_N, BLOCK_K
    sizes = group_sizes.astype(jnp.int32)
    ends = jnp.cumsum(sizes).astype(jnp.int32)
    starts = (ends - sizes).astype(jnp.int32)
    block_starts = (starts // block_k * block_k).astype(jnp.int32)
    block_ends = (-(ends // -block_k) * block_k).astype(jnp.int32)
    num_blocks = ((block_ends - block_starts) // block_k).astype(jnp.int32)
    grid_m = m // block_m
    grid_n = n // block_n
    grid_block_n = 2 if grid_n % 2 == 0 else 1

    swizzle = plgpu.find_swizzle(block_k * jnp.dtype(lhs.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(lhs.dtype).itemsize
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

    def body(sizes_gmem, starts_gmem, ends_gmem, num_blocks_gmem, block_starts_gmem, lhs_gmem, rhs_gmem,
             o_gmem):
        @plgpu.nd_loop((g, grid_m * grid_n), collective_axes="sm")
        def _mn_loop(loop_info: plgpu.NDLoopInfo):
            g_i = loop_info.index[0]
            m_i, n_i = plgpu.planar_snake(loop_info.index[1], (grid_m, grid_n), 1, grid_block_n)
            # The view may extend past the rows; the pipeline only reads this group's blocks.
            group_block_start = pl.multiple_of(block_starts_gmem[g_i], block_k)
            gmem_slice = pl.ds(group_block_start, rows)

            def acc_scope(acc_ref):
                def block_matmul(block_idx, lhs_smem, rhs_smem):
                    block_idx = block_idx[0]

                    @pl.when(block_idx == 0)
                    def _():
                        # First block: zero the rows of the previous group.
                        start_index = lax.rem(starts_gmem[g_i], block_k)
                        idx = plgpu.layout_cast(
                            lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0), plgpu.Layout.WGMMA)
                        lhs_smem[...] = lhs_smem[...] * (idx >= start_index).astype(lhs_smem.dtype)
                        plgpu.commit_smem()

                    @pl.when(block_idx == num_blocks_gmem[g_i] - 1)
                    def _():
                        # Last block: zero the rows of the next group (and rows past the end).
                        last_index = lax.rem(ends_gmem[g_i] - 1, block_k)
                        idx = plgpu.layout_cast(
                            lax.broadcasted_iota(jnp.int32, (block_k, block_m), 0), plgpu.Layout.WGMMA)
                        lhs_smem[...] = lhs_smem[...] * (idx <= last_index).astype(lhs_smem.dtype)
                        plgpu.commit_smem()

                    plgpu.wgmma(acc_ref, lhs_smem.transpose((1, 0)), rhs_smem)

                @pl.when(sizes_gmem[g_i] > 0)  # empty groups keep the zero accumulator
                def _():
                    plgpu.emit_pipeline(
                        block_matmul,
                        grid=(num_blocks_gmem[g_i],),
                        in_specs=[
                            plgpu.BlockSpec((block_k, block_m), lambda k_i: (k_i, m_i), delay_release=1,
                                            transforms=transforms),
                            plgpu.BlockSpec((block_k, block_n), lambda k_i: (k_i, n_i), delay_release=1,
                                            transforms=transforms),
                        ],
                        max_concurrent_steps=STEPS,
                    )(lhs_gmem.at[gmem_slice, :], rhs_gmem.at[gmem_slice, :])

                return acc_ref[...]

            acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

            @functools.partial(
                pl.run_scoped, o_smem=plgpu.SMEM((block_m, block_n), dtype=o_gmem.dtype, transforms=transforms))
            def _store(o_smem):
                o_smem[...] = acc.astype(o_smem.dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    o_smem, o_gmem.at[g_i, pl.ds(m_i * block_m, block_m), pl.ds(n_i * block_n, block_n)])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((g, m, n), lhs.dtype),
        grid=(num_sms(),),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    with jax.numpy_dtype_promotion("standard"):
        return kernel(sizes, starts, ends, num_blocks, block_starts, lhs, rhs)


def supports(lhs, rhs, group_sizes, config: TileConfig | None = None) -> bool:
    """Shapes and dtypes the kernels (forward and both gradients) handle on this device.

    With ``config``, also that the tiling divides the forward and the ``d_lhs`` GEMM.
    """
    if not is_hopper():
        return False
    if getattr(lhs, "ndim", None) != 2 or getattr(rhs, "ndim", None) != 3 or getattr(group_sizes, "ndim", None) != 1:
        return False
    if jnp.dtype(lhs.dtype) not in DTYPES or jnp.dtype(rhs.dtype) != jnp.dtype(lhs.dtype):
        return False
    if not jnp.issubdtype(group_sizes.dtype, jnp.integer):
        return False
    m, k = lhs.shape
    g, k2, n = rhs.shape
    if config is not None and not (isinstance(config, TileConfig) and config.divides_both(k, n)):
        return False
    return (k == k2 and group_sizes.shape[0] == g and 1 <= g <= MAX_GROUPS and m >= BLOCK_M
            and k % ALIGN == 0 and n % ALIGN == 0)


def _mask_rows(out: Array, group_sizes: Array) -> Array:
    rows = jnp.arange(out.shape[0], dtype=jnp.int32) < jnp.sum(group_sizes)
    return jnp.where(rows[:, None], out, jnp.zeros((), out.dtype))


def _config_for(config: TileConfig | None, k: int, n: int) -> TileConfig:
    cfg = select_config(k, n) if config is None else config
    if not cfg.divides(k, n):
        raise ValueError(f"ragged_dot tile {cfg} does not divide contraction {k} and width {n}")
    return cfg


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _ragged_dot(lhs: Array, rhs: Array, group_sizes: Array, full_groups: bool, config: TileConfig | None) -> Array:
    k, n = rhs.shape[1], rhs.shape[2]
    out = _ragged_dot_kernel(lhs, rhs, group_sizes, transpose_rhs=False, cfg=_config_for(config, k, n))
    return out if full_groups else _mask_rows(out, group_sizes)


def _ragged_dot_fwd(lhs, rhs, group_sizes, full_groups, config):
    return _ragged_dot(lhs, rhs, group_sizes, full_groups, config), (lhs, rhs, group_sizes)


def _ragged_dot_bwd(full_groups, config, residuals, d_out):
    lhs, rhs, group_sizes = residuals
    d_out = d_out.astype(lhs.dtype)
    k, n = rhs.shape[1], rhs.shape[2]
    # d_lhs contracts over n and produces k columns.
    d_lhs = _ragged_dot_kernel(d_out, rhs, group_sizes, transpose_rhs=True, cfg=_config_for(config, n, k))
    if not full_groups:
        d_lhs = _mask_rows(d_lhs, group_sizes)
    d_rhs = transposed_ragged_dot(lhs, d_out, group_sizes)  # [G, K, N]
    return d_lhs, d_rhs.astype(rhs.dtype), None


_ragged_dot.defvjp(_ragged_dot_fwd, _ragged_dot_bwd)


def ragged_dot(lhs: Array, rhs: Array, group_sizes: Array, *, full_groups: bool = False,
               config: TileConfig | None = None) -> Array:
    """Mosaic GPU ``jax.lax.ragged_dot`` (differentiable); raises when :func:`supports` rejects it.

    ``config`` tiles the forward and ``d_lhs`` kernels (default: :func:`select_config` per
    GEMM; :data:`BLOCK_ALIGNED` is the block-aligned kernel); it must divide both, checked here rather than
    when the gradient is traced.
    """
    if config is not None and not isinstance(config, TileConfig):
        raise TypeError(f"config must be a TileConfig or None, got {type(config).__name__}")
    if not supports(lhs, rhs, group_sizes):
        raise ValueError(
            f"Mosaic GPU ragged_dot does not support lhs {getattr(lhs, 'shape', None)} "
            f"{getattr(lhs, 'dtype', None)}, rhs {getattr(rhs, 'shape', None)} {getattr(rhs, 'dtype', None)}, "
            f"group_sizes {getattr(group_sizes, 'shape', None)} on {jax.devices()[0].device_kind}: it needs an "
            f"H100, bfloat16/float16 operands of one dtype, M >= {BLOCK_M}, K and N multiples of {ALIGN} "
            f"and at most {MAX_GROUPS} groups")
    if config is not None and not config.divides_both(rhs.shape[1], rhs.shape[2]):
        raise ValueError(
            f"ragged_dot tile {config} does not divide the forward GEMM (contraction {rhs.shape[1]}, width "
            f"{rhs.shape[2]}) and the d_lhs GEMM (contraction {rhs.shape[2]}, width {rhs.shape[1]})")
    # TMA tiles rows by 8: pad with rows that belong to no group and slice them off again
    # (their outputs and d_lhs rows are never read; d_rhs reads only grouped rows).
    m = lhs.shape[0]
    pad = -m % 8
    if pad:
        lhs = jnp.pad(lhs, ((0, pad), (0, 0)))
    out = _ragged_dot(lhs, rhs, group_sizes.astype(jnp.int32), bool(full_groups), config)
    return out[:m] if pad else out


def grouped_matmul(lhs: Array, rhs: Array, group_sizes: Array, *, impl: str = "xla",
                   full_groups: bool = False, config: TileConfig | None = None) -> Array:
    """Grouped GEMM by ``impl``: ``"mosaic_gpu"`` and ``"auto"`` use the kernels when supported, else
    XLA (``"mosaic_gpu"`` warns when it falls back, ``"auto"`` does not).

    An explicit ``config`` needs the kernels: it raises with ``impl="xla"`` and where the call
    would fall back to XLA, instead of being dropped.
    """
    if impl not in IMPLS:
        raise ValueError(f"unknown ragged_dot impl {impl!r}; expected one of {IMPLS}")
    if config is not None:
        if impl == "xla":
            raise ValueError("ragged_dot config tiles the Mosaic GPU kernels; impl='xla' takes no config")
        return ragged_dot(lhs, rhs, group_sizes, full_groups=full_groups, config=config)
    if impl == "auto":
        if supports(lhs, rhs, group_sizes):
            return ragged_dot(lhs, rhs, group_sizes, full_groups=full_groups)
        return lax.ragged_dot(lhs, rhs, group_sizes)
    if impl == "mosaic_gpu":
        if supports(lhs, rhs, group_sizes):
            return ragged_dot(lhs, rhs, group_sizes, full_groups=full_groups, config=config)
        warnings.warn(
            f"Mosaic GPU ragged_dot does not support lhs {lhs.shape} {lhs.dtype}, rhs {rhs.shape} {rhs.dtype} "
            f"on {jax.devices()[0].device_kind}; falling back to jax.lax.ragged_dot", stacklevel=2)
    return lax.ragged_dot(lhs, rhs, group_sizes)
