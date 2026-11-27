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
def is_blackwell() -> bool:
    """True on a datacenter Blackwell GPU (sm_100 B200/GB200, sm_103 B300): tcgen05 tensor cores, no WGMMA."""
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", None) in ("10.0", "10.3")


def blackwell_enabled() -> bool:
    """Blackwell kernels are opt-in (``JLA_BLACKWELL=1``) until they are qualified on hardware."""
    import os

    return is_blackwell() and os.environ.get("JLA_BLACKWELL", "") == "1"


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
    if is_blackwell():
        # A 1-CTA M=64 tcgen05 MMA splits N into two halves, and each half must hold a whole swizzle
        # atom. Tiles are used as the RHS either way round (``b``, ``b.T``) and in 64-row or 64-column
        # blocks, so the swizzle is capped at 32 elements (half a 64-wide block).
        swizzle = min(swizzle, 32 * dtype.itemsize)
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

    if is_blackwell():
        return _matmul_tcgen05(a, b, shape, init)

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


# ---------------------------------------------------------------------------------------------------------
# Blackwell (sm_100 / sm_103): the tensor-core helpers above run as tcgen05 MMAs through tensor memory (TMEM).
#
# Every Blackwell product is synchronous: its MMAs are issued, one tcgen05 commit tracks them, and the result
# is loaded back to registers before the helper returns. A 1-CTA M=64 MMA keeps its float32 accumulator in the
# TMEM half-lane layout, which loads straight into the WGMMA register layout, so the kernels' register code is
# the same as on Hopper. A register LHS (tcgen05 reads its LHS from SMEM, or from TMEM only when M=128) is
# staged through a [64, 64] SMEM tile, 64 columns of K at a time.
#
# Scratch: in a single-warpgroup kernel each product allocates its TMEM buffers and barrier with ``run_scoped``.
# A warp-specialized kernel cannot (a scoped allocation there must be made by all warpgroups together), so
# :func:`build_kernel` traces such a kernel twice: the first trace records every product's TMEM buffers in call
# order, the second hands each call site its own buffers and barrier from kernel-level scratch. Call sites
# never share buffers, so warpgroups never race on them.
# ---------------------------------------------------------------------------------------------------------
TMEM_COLS = 512
BLACKWELL_STAGING_BYTES = 24 * 1024  # SMEM the pipelines leave free for staged register operands


class _NeedPlan(Exception):
    pass


def _tmem_cols(shape, dtype) -> int:
    rows, cols = shape
    packing = 4 // jnp.dtype(dtype).itemsize
    return max(32, cols // (2 if rows == 64 else 1) // packing)


def _scratch(space, shape, dtype):
    """A float32 TMEM accumulator or a swizzled SMEM staging tile for a register LHS."""
    return plgpu.TMEM(shape, dtype) if space == "tmem" else smem(shape, dtype)


def _check_rows(shape) -> None:
    if shape[0] != 64:
        raise NotImplementedError(f"Blackwell products support 64-row tiles only; got {tuple(shape)}")


class _TcPlan:
    """Kernel-level tensor-core scratch of one warp-specialized kernel (recording pass, then providing pass).

    Buffers are pooled per warpgroup (``tc_group``): a call site takes free buffers of its kind from its
    warpgroup's pool and returns them when it finishes, so sequential products of one warpgroup share
    buffers while nested ones (a staged MMA inside :func:`mma_acc`) and other warpgroups' never do. Call
    sites outside a ``tc_group`` get a pool of their own. Both passes run the same allocation, so the
    providing pass hands every call site the buffers the recording pass counted.
    """

    def __init__(self, sizes=None, refs=None, bars=None):
        self.recording = sizes is None
        self.pools: dict = {}  # (group, kind) -> [in use, ...]
        self.sizes = sizes  # providing pass: (group, kind) -> index of its first ref / barrier
        self.refs, self.bars, self.anon = refs, bars, 0

    def _take(self, group, kind):
        pool = self.pools.setdefault((group, kind), [])
        for i, busy in enumerate(pool):
            if not busy:
                pool[i] = True
                return i
        pool.append(True)
        return len(pool) - 1

    def _ref(self, group, kind, i):
        base = self.sizes[(group, kind)]
        if kind == "bar":
            return self.bars.at[base + i]
        ref = self.refs[base + i]
        _, shape, dtype = kind
        if tuple(ref.shape) != tuple(shape) or jnp.dtype(ref.dtype) != jnp.dtype(dtype):
            raise RuntimeError(f"tensor-core scratch mismatch: {ref.shape} {ref.dtype} against {shape} {dtype}")
        return ref

    def scope(self, specs, fn, placeholder):
        group = _GROUP
        if group is None:
            group, self.anon = ("site", self.anon), self.anon + 1
        kinds = [(space, tuple(shape), jnp.dtype(dtype)) for space, shape, dtype in specs] + ["bar"]
        taken = [(kind, self._take(group, kind)) for kind in kinds]
        try:
            if self.recording:
                return placeholder()
            refs = [self._ref(group, kind, i) for kind, i in taken]
            return fn(refs[:-1], refs[-1])
        finally:
            for kind, i in taken:
                self.pools[(group, kind)][i] = False

    def layout(self):
        """Recording pass result: scratch types, and where each pool's buffers start."""
        refs, nbars, sizes, cols = [], 0, {}, 0
        for (group, kind), pool in self.pools.items():
            if kind == "bar":
                sizes[(group, kind)], nbars = nbars, nbars + len(pool)
            else:
                sizes[(group, kind)] = len(refs)
                refs += [_scratch(*kind) for _ in pool]
                if kind[0] == "tmem":
                    cols += _tmem_cols(kind[1], kind[2]) * len(pool)
        return refs, nbars, sizes, cols


_PLAN: _TcPlan | None = None
_GROUP = None
_STAGE_COLS = None  # set by tc_group(stage_cols=...) for its branch


def tc_group(name, stage_cols: int | None = None):
    """Decorator for a warpgroup's branch of a warp-specialized kernel: its Blackwell products share a pool.

    ``stage_cols`` narrows the SMEM tile that stages register operands (default :data:`STAGE_COLS`) for a
    kernel with little shared memory to spare; narrower tiles mean more, smaller synchronous MMAs.
    """
    def wrap(fn):
        def run(*args, **kwargs):
            global _GROUP, _STAGE_COLS
            outer = _GROUP, _STAGE_COLS
            _GROUP, _STAGE_COLS = name, stage_cols
            try:
                return fn(*args, **kwargs)
            finally:
                _GROUP, _STAGE_COLS = outer
        return run
    return wrap


def _tc_scope(specs, fn, placeholder):
    """``fn(refs, done)`` with scratch ``specs = [(space, shape, dtype)]`` and a tensor-core barrier."""
    if _PLAN is None:
        def scoped(done, *refs):
            return fn(list(refs), done)

        return pl.run_scoped(scoped, plgpu.Barrier(orders_tensor_core=True),
                             *(_scratch(*spec) for spec in specs))
    return _PLAN.scope(specs, fn, placeholder)


def _zeros(shape):
    return plgpu.layout_cast(jnp.zeros(shape, jnp.float32), WGMMA)


class Prestaged(NamedTuple):
    """A 16-bit register LHS whose value is also in the SMEM tile ``ref`` (Blackwell reads the tile)."""

    value: Array
    ref: object


def prestaged(value: Array, ref):
    """``value``, already stored to ``ref`` and committed: on Blackwell the MMA reads ``ref`` instead of staging."""
    return Prestaged(value, ref) if is_blackwell() else value


def stage_into(value: Array, ref):
    """On Blackwell store ``value`` to the free SMEM tile ``ref`` (committed) for use as an MMA LHS.

    The caller guarantees no other warpgroup reads or writes ``ref`` until the products using it
    return. On Hopper ``value`` is returned unchanged (WGMMA takes it from registers).
    """
    if not is_blackwell():
        return value
    ref[...] = value
    plgpu.commit_smem()
    return Prestaged(value, ref)


def _lhs(a):
    return a.ref if isinstance(a, Prestaged) else a


STAGE_COLS = 64  # register LHS operands are staged through one [64, 64] SMEM tile, 64 columns of K at a time


def _stage_cols() -> int:
    return _STAGE_COLS or STAGE_COLS


def _stage_spec(a):
    return ("smem", (a.shape[0], _stage_cols()), a.dtype)


def _staged_mma(acc, a, b, accumulate, tile, done) -> None:
    """``acc (+)= a @ b`` for a register ``a``, through ``tile``, synchronously (every earlier MMA completes too)."""
    k, cols = a.shape[1], tile.shape[1]
    if k % cols:
        raise NotImplementedError(f"Blackwell register operands need K a multiple of {cols}; got {a.shape}")
    for j in range(k // cols):
        part = slice(j * cols, (j + 1) * cols)
        tile[...] = a[:, part]
        plgpu.commit_smem()
        b_part = b if k == cols else b.at[pl.ds(j * cols, cols)]
        plgpu.tcgen05_mma(acc, tile, b_part, accumulate=accumulate or j > 0)
        plgpu.tcgen05_commit_arrive(done)
        plgpu.barrier_wait(done)


def _tc_products(jobs):
    """``[init + sum(a @ b for a, b in terms)]`` for ``jobs = [(init | None, terms, shape)]`` (float32, WGMMA)."""
    for _, _, shape in jobs:
        _check_rows(shape)
    staged = [a for _, terms, _ in jobs for a, _ in terms if isinstance(a, jax.Array)]
    specs = [("tmem", shape, jnp.float32) for _, _, shape in jobs]
    if staged:
        specs.append(_stage_spec(staged[0]))
        if any(_stage_spec(a) != specs[-1] for a in staged):
            raise NotImplementedError("Blackwell products with register operands of different dtypes")

    def fn(refs, done):
        accs, tile = refs[:len(jobs)], (refs[len(jobs)] if staged else None)
        for acc, (init, _, _) in zip(accs, jobs):
            if init is not None:
                plgpu.async_store_tmem(acc, init)
        if any(init is not None for init, _, _ in jobs):
            plgpu.commit_tmem()
        pending = False  # an asynchronous MMA not yet tracked by a commit
        for acc, (init, terms, _) in zip(accs, jobs):
            for t, (a, b) in enumerate(terms):
                accumulate = init is not None or t > 0
                if isinstance(a, jax.Array):
                    _staged_mma(acc, a, b, accumulate, tile, done)  # waits for every earlier MMA too
                    pending = False
                else:
                    plgpu.tcgen05_mma(acc, _lhs(a), b, accumulate=accumulate)
                    pending = True
        if pending:  # never commit with nothing outstanding (an empty commit stalled on B200 with >1 CTA)
            plgpu.tcgen05_commit_arrive(done)
            plgpu.barrier_wait(done)
        outs = [plgpu.async_load_tmem(acc, layout=WGMMA) for acc in accs]
        plgpu.wait_load_tmem()
        return outs

    return _tc_scope(specs, fn, lambda: [_zeros(shape) if init is None else init for init, _, shape in jobs])


def _matmul_tcgen05(a, b, shape: tuple[int, int], init: Array | None) -> Array:
    """Blackwell form of :func:`matmul`."""
    return lax.optimization_barrier(_tc_products([(init, [(a, b)], shape)])[0])


class _TmemAcc:
    """Blackwell stand-in for a WGMMA accumulator ref inside :func:`mma_acc`."""

    def __init__(self, tmem):
        self.tmem = tmem
        self.pending = False  # an asynchronous MMA into it not yet tracked by a commit


def mma(acc, a, b) -> None:
    """``acc += a @ b``: ``plgpu.wgmma`` on Hopper, a ``tcgen05`` MMA on Blackwell.

    ``acc`` is the accumulator handed to the body of :func:`mma_acc`. On Blackwell an SMEM
    LHS is read asynchronously, as with WGMMA, until the end of the accumulation (so the
    body must not overwrite it); a register LHS is staged through SMEM, and that MMA (with every
    earlier one) completes before the staging tile is released.
    """
    if not isinstance(acc, _TmemAcc):
        plgpu.wgmma(acc, a, b)
        return
    if acc.tmem is None:  # recording pass
        if isinstance(a, jax.Array):
            _tc_scope([_stage_spec(a)], None, lambda: None)
        return
    if not isinstance(a, jax.Array):
        plgpu.tcgen05_mma(acc.tmem, _lhs(a), b, accumulate=True)
        acc.pending = True
        return
    _tc_scope([_stage_spec(a)], lambda refs, done: _staged_mma(acc.tmem, a, b, True, refs[0], done), lambda: None)
    acc.pending = False


def mma_acc(body, init: Array) -> Array:
    """``init`` plus the products ``body(acc)`` issues with :func:`mma` (float32, WGMMA layout).

    On Hopper this is ``pl.run_state(body)(plgpu.ACC.init(init))``, unchanged. On Blackwell
    the accumulator is a TMEM buffer: ``init`` is stored to it, the body's MMAs run, one
    commit tracks them all, and the sum is loaded back into the WGMMA register layout.

    Blackwell rule: the body must call :func:`mma` unconditionally (Python ``if``/``for`` are fine;
    no ``pl.when``, ``lax.cond`` or ``fori_loop`` around it). Whether a final commit is needed is
    decided while tracing, so an MMA that might not run would skip the wait or issue an empty
    commit (which stalls with more than one CTA).
    """
    if not is_blackwell():
        return pl.run_state(body)(plgpu.ACC.init(init))
    _check_rows(init.shape)

    def fn(refs, done):
        plgpu.async_store_tmem(refs[0], init)
        plgpu.commit_tmem()
        acc = _TmemAcc(refs[0])
        body(acc)
        if acc.pending:
            plgpu.tcgen05_commit_arrive(done)
            plgpu.barrier_wait(done)
        out = plgpu.async_load_tmem(refs[0], layout=WGMMA)
        plgpu.wait_load_tmem()
        return lax.optimization_barrier(out)  # keep uses of ``out`` after the wait (it has no SSA link to it)

    def record():
        body(_TmemAcc(None))
        return init

    return _tc_scope([("tmem", init.shape, jnp.float32)], fn, record)


def products(*specs) -> tuple[Array, ...]:
    """Independent float32 products ``a @ b`` for ``specs = (a, b, shape)``, issued together, one wait.

    Hopper: one WGMMA group (the same instructions :func:`matmul` runs). Blackwell: one
    TMEM accumulator per product and one commit for all.
    """
    if is_blackwell():
        return lax.optimization_barrier(tuple(_tc_products([(None, [(a, b)], shape) for a, b, shape in specs])))

    def scoped(*accs):
        for acc, (a, b, _) in zip(accs, specs):
            plgpu.wgmma(acc, a, b)
        return tuple(acc[...] for acc in accs)

    return lax.optimization_barrier(pl.run_scoped(scoped, *(plgpu.ACC(shape, jnp.float32) for _, _, shape in specs)))


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

    ``threads`` passes ``num_threads``/``thread_name`` for warp-specialized kernels. On
    Blackwell a warp-specialized kernel is traced twice (see the TMEM plan above).
    """
    def make(body, scratch):
        return plgpu.kernel(
            body, out_type=out_type, scratch_types=scratch, grid=grid, grid_names=grid_names,
            compiler_params=compiler_params(), interpret=interpret, kernel_name=name, **threads,
        )

    if not (is_blackwell() and threads.get("num_threads")):
        return make(body, scratch)

    def call(*args):
        global _PLAN
        plan = _TcPlan()

        def recording_body(*refs, **kw):
            global _PLAN
            _PLAN = plan
            try:
                body(*refs, **kw)
            finally:
                _PLAN = None
            raise _NeedPlan

        try:
            make(recording_body, scratch)(*args)
        except _NeedPlan:
            pass
        if not plan.pools:
            return make(body, scratch)(*args)
        refs, nbars, sizes, cols = plan.layout()
        if cols > TMEM_COLS:
            raise NotImplementedError(f"{name}: Blackwell products need {cols} TMEM columns (> {TMEM_COLS})")
        tc_scratch = dict(_tc_refs=refs, _tc_bars=plgpu.Barrier(num_arrivals=1, num_barriers=max(nbars, 2),
                                                                 orders_tensor_core=True))

        def planned_body(*refs, _tc_refs, _tc_bars, **kw):
            global _PLAN
            _PLAN = _TcPlan(sizes, list(_tc_refs), _tc_bars)
            try:
                body(*refs, **kw)
            finally:
                _PLAN = None

        return make(planned_body, dict(scratch, **tc_scratch))(*args)

    return call


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
