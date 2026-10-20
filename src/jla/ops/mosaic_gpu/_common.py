# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Mosaic GPU linear-attention kernels.

Layout conventions of the scalar-gate kernels (one chunk = ``CHUNK`` tokens):

* user tensors keep the FLA layout ``[B, T, H, D]`` and are read tile by tile
  with TMA (``ref.at[b, pl.ds(t0, CHUNK), h]``);
* per-token gate vectors are ``[B, H, T]`` float32, so a chunk is one
  contiguous row for a 1-D TMA copy; the decay is pre-scaled to log2 so the
  kernels use ``exp2``;
* intermediates are head-major ``[B, H, T, D]``; per-chunk states are stored
  transposed (``[B, H, NC, V, K]``), the orientation the serial kernels keep
  in registers.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jla.ops.reference import segment_resets

Array = jax.Array

CHUNK = 64
LOG2E = math.log2(math.e)
SUPPORTED_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
SUPPORTED_HEAD_DIMS = (64, 128)

WGMMA = plgpu.Layout.WGMMA
ROWS = plgpu.Layout.WGMMA.reduce(1)  # one value per row of a WGMMA-layout tile
COLS = plgpu.Layout.WGMMA.reduce(0)  # one value per column


@functools.cache
def is_hopper() -> bool:
    """True when the default JAX backend is an sm_90 (H100/H200/H800) GPU."""
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", None) == "9.0"


@functools.cache
def num_sms() -> int:
    """Streaming multiprocessors of the default device (132 on an H100 SXM if unknown)."""
    try:
        return int(getattr(jax.devices()[0], "core_count", 0) or 0) or 132
    except RuntimeError:
        return 132


def compiler_params() -> plgpu.CompilerParams:
    return plgpu.CompilerParams(approx_math=True)


def smem(shape: tuple[int, ...], dtype) -> plgpu.SMEM:
    """SMEM buffer tiled and swizzled for TMA and WGMMA (minor dim = last dim)."""
    dtype = jnp.dtype(dtype)
    swizzle = plgpu.find_swizzle(shape[-1] * dtype.itemsize * 8)
    transforms = (plgpu.TilingTransform((8, swizzle // dtype.itemsize)), plgpu.SwizzleTransform(swizzle))
    return plgpu.SMEM(shape, dtype, transforms=transforms)


def iota(shape: tuple[int, int], dim: int) -> Array:
    return plgpu.broadcasted_iota(jnp.int32, shape, dim, layout=WGMMA)


def bcast_rows(vec: Array, shape: tuple[int, int]) -> Array:
    """``vec[i]`` broadcast along row ``i`` (``vec`` in the ``ROWS`` layout)."""
    return lax.broadcast_in_dim(vec, shape, [0])


def bcast_cols(vec: Array, shape: tuple[int, int]) -> Array:
    """``vec[j]`` broadcast along column ``j`` (``vec`` in the ``COLS`` layout)."""
    return lax.broadcast_in_dim(vec, shape, [1])


def decay_matrix(g_row: Array, g_col: Array, shape: tuple[int, int], *, strict: bool,
                 runs: tuple[Array, Array] | None = None) -> Array:
    """``exp2(G_i - G_j)`` on the causal part (``j < i`` or ``j <= i``), zero elsewhere.

    With ``runs = (run_row, run_col)`` (segment run ids in the ``ROWS`` and
    ``COLS`` layouts) only pairs of one run are kept. The exponent is masked
    before ``exp2``, so acausal entries never overflow.
    """
    rows, cols = iota(shape, 0), iota(shape, 1)
    keep = rows > cols if strict else rows >= cols
    if runs is not None:
        keep = keep & same_run(runs, shape)
    expo = jnp.where(keep, bcast_rows(g_row, shape) - bcast_cols(g_col, shape), -jnp.inf)
    return jnp.exp2(expo)


def same_run(runs: tuple[Array, Array], shape: tuple[int, int]) -> Array:
    """``run_i == run_j`` for ``runs = (run_row, run_col)``."""
    return bcast_rows(runs[0], shape) == bcast_cols(runs[1], shape)


def matmul(a, b, shape: tuple[int, int], init: Array | None = None) -> Array:
    """``init + a @ b`` on the tensor cores (``a``: SMEM ref or 16-bit WGMMA registers).

    The result passes through an optimization barrier. Reading a WGMMA
    accumulator lowers to an inline ``wgmma.wait_group`` that has no data
    dependency on the accumulator registers, so LLVM may otherwise schedule
    arithmetic on the result before the wait and read stale registers (seen as
    silently zeroed products in the WY backward kernel).
    """

    def body(acc):
        plgpu.wgmma(acc, a, b)

    if init is None:
        def scoped(acc):
            plgpu.wgmma(acc, a, b)
            return acc[...]
        out = pl.run_scoped(scoped, plgpu.ACC(shape, jnp.float32))
    else:
        out = pl.run_state(body)(plgpu.ACC.init(init))
    return lax.optimization_barrier(out)


def pad_time(x: Array, pad: int, axis: int = 1) -> Array:
    if pad == 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return jnp.pad(x, widths)


def chunk_log2_decay(g: Array | None, batch: int, seq_len: int, heads: int, pad: int) -> Array:
    """Chunk-local cumulative log-decay in log2 units, ``[B, H, T + pad]`` float32."""
    padded = seq_len + pad
    if g is None:
        return jnp.zeros((batch, heads, padded), jnp.float32)
    g = pad_time(g.astype(jnp.float32), pad)  # [B, Tp, H]
    g = jnp.swapaxes(g, 1, 2).reshape(batch, heads, padded // CHUNK, CHUNK)
    return (jnp.cumsum(g, axis=-1) * LOG2E).reshape(batch, heads, padded)


def head_major(x: Array, pad: int) -> Array:
    """``[B, T, H]`` per-token scalars to padded ``[B, H, Tp]`` float32."""
    return jnp.swapaxes(pad_time(x.astype(jnp.float32), pad), 1, 2)


class Segments(NamedTuple):
    """Packed-document masks of :func:`jla.ops.chunked.segment_masks` as float32 ``[B, Tp]`` vectors.

    ``run`` is the chunk-local run id (``same[i, j] = run_i == run_j``),
    ``cin`` is 1 where a token belongs to the run active when its chunk starts
    (it sees the incoming state and gets the WY correction), ``tail`` is 1
    where a token belongs to its chunk's last run (it writes the outgoing
    state). The incoming state survives a chunk iff ``cin`` of its last token
    is 1.
    """

    run: Array
    cin: Array
    tail: Array


def segment_vectors(segment_ids: Array, pad: int) -> Segments:
    """:class:`Segments` from ``segment_ids [B, T]`` (a run starts wherever the id changes)."""
    runs = jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)
    runs = jnp.pad(runs, [(0, 0), (0, pad)], mode="edge")  # padding joins the last run
    batch, padded = runs.shape
    runs = runs.reshape(batch, padded // CHUNK, CHUNK)
    entering = jnp.concatenate([runs[:, :1, :1], runs[:, :-1, -1:]], axis=1)

    def flat(x):
        return x.astype(jnp.float32).reshape(batch, padded)

    return Segments(flat(runs - runs[:, :, :1]), flat(runs == entering), flat(runs == runs[:, :, -1:]))


def build_kernel(body, *, name: str, out_type, scratch, grid, grid_names, interpret=None, **threads):
    """``plgpu.kernel`` with this package's compiler parameters and a profiler-visible name.

    ``threads`` passes ``num_threads``/``thread_name`` for warp-specialized kernels.
    """
    return plgpu.kernel(
        body, out_type=out_type, scratch_types=scratch, grid=grid, grid_names=grid_names,
        compiler_params=compiler_params(), interpret=interpret, kernel_name=name, **threads,
    )


def finish_together(done) -> None:
    """Every warpgroup arrives at the CTA's ``done`` barrier and waits for the others.

    Warp-specialized kernels call this after their warpgroup branches (``done``
    counts one arrival per warpgroup). When one warpgroup exited while another
    still issued WGMMAs, some runs ended in ``CUDA_ERROR_ILLEGAL_INSTRUCTION``
    (sm_90, JAX 0.11.2; see ``benchmarks/README.md``).
    """
    plgpu.barrier_arrive(done)
    plgpu.barrier_wait(done)


def stage(shape: tuple[int, int], dtype) -> plgpu.SMEM:
    """Output staging tile: swizzled for 16-bit dtypes, plain for float32."""
    return smem(shape, dtype) if jnp.dtype(dtype).itemsize == 2 else plgpu.SMEM(shape, dtype)


# CUDA caps the y and z grid dimensions at 65535 blocks (x at 2**31 - 1). Pallas Mosaic GPU launches a
# grid (g0, g1, g2) as CUDA (z, y, x) and folds the leading dimensions of a longer grid into z, so a
# kernel whose grid puts the chunk index first or second runs at most 65535 chunks per call; past that
# the launch fails with CUDA_ERROR_INVALID_VALUE. ``supports`` predicates reject such calls instead.
MAX_GRID_YZ = 65535
