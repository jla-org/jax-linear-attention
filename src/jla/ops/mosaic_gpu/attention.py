# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2024 The JAX Authors.
# Portions Copyright 2025-2026 DeepMind Technologies Limited.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/attention_mgpu.py
# Adapted from Tokamax (Apache-2.0), tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py
# Backward block ranges and warpgroup layout follow Dao-AILab flash-attention (BSD-3-Clause),
# hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp and hopper/block.h (design only, no code copied); so does the
# fused backward (dS staged in shared memory as the dQ product's operand, dQ split over the two consumer
# warpgroups, fp32 dQ added with TMA reductions and rounded afterwards)
# The batch and head custom_partitioning rule follows JAX (Apache-2.0), jax/_src/cudnn/fused_attention_stablehlo.py
# (design only, no code copied)
# The forward's intra-warpgroup overlap, turn order and 192-row tiles for D = 64 follow Dao-AILab flash-attention
# (BSD-3-Clause), hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp and hopper/tile_size.h (design only, no code copied)
# Adapted from Tokamax (Apache-2.0), tokamax/_src/mosaic_gpu.py (_bar_operation: the named barriers of _named_barrier)
# The backward kernels' CTA order follows Dao-AILab flash-attention (BSD-3-Clause), hopper/tile_scheduler.hpp
# (SingleTileBwdLPTScheduler: longest tiles first in L2-sized sections of heads; design only, no code copied)
"""FA3-style causal softmax attention for Hopper in Pallas Mosaic GPU.

``flash_attention(q, k, v, segment_ids=None, sliding_window=None, scale=None, sinks=None)``
computes causal grouped-query attention with ``q [B, T, H, D]`` and
``k, v [B, T, Hkv, D]`` (bf16 or fp16, ``D`` in {64, 128}), fp32 softmax
statistics, and the semantics of :func:`jla.layers.attention.attention`.
Multi-head latent attention's asymmetric heads are supported too: ``q, k`` of
``Dqk = 192`` and ``v`` of ``Dv = 128`` (DeepSeek-V2/V3, Kimi K2, Kimi Linear:
128 no-RoPE + 64 RoPE channels per query/key, values of 128); the output then has
``Dv`` channels. The forward keeps the first ``Dv`` query/key channels and the rest
in separate shared-memory tiles (two products per score tile), so the query tile
also stages the ``Dv``-wide output; the ranged backward kernels use ``Dqk`` for ``dq`` and
``dk`` and ``Dv`` for ``dP`` and ``dv`` (``backward="full"`` and ``"fused"`` need one head dim).
Semantics:
``sliding_window`` counts the visible tokens including the query and
``segment_ids [B, T]`` restricts every query to keys with the same id, and
``sinks [H]`` (optional) adds a per-head logit to the softmax denominator.

Forward: two consumer warpgroups (64 query rows each) and one TMA producer
warpgroup per 128-row query tile, ping-pong scheduled as in FlashAttention-3, the query
tiles with the most key blocks launched first (``heavy_first``). The overlapped loop
(``fwd_overlap``) issues ``Q K_j^T`` and ``P_{j-1} V_{j-1}`` together in step ``j``, as
FlashAttention-3 does (its softmax of block ``j`` is traced after the wait for the ``PV``
product; tracing it first measured no faster); the warpgroups
pass the turn to issue on PTX named barriers and release key and value slots to the producer
on named barriers (one phase per slot), with three consumer warpgroups (192-row tiles) for
``D = 64``. It runs by default only with ``D = 64`` and no segment ids (causal, with or
without a sliding window), where it measured faster; its arithmetic and order are those of
the ping-pong loop, so outputs and ``lse`` are bitwise equal either way.
Key blocks entirely in the causal future, entirely left of the sliding window or
entirely before the document of a warpgroup's first query are never loaded.
Document masks are applied per tile from the segment ids (no dense ``[T, T]``
mask is built). Backward (``backward="ranged"``, the default): ``dq`` and
``dk, dv`` kernels (``jax.custom_vjp``) with two consumer warpgroups and one TMA
warpgroup each, in the design of the FlashAttention-3 backward. Each tile visits
only the key (or query) block range ``[lo, hi)`` it can see, bounded by the
causal diagonal, the sliding window and the document of its first query (or last
key). In the forward and both backward kernels the mask is applied only on blocks
that cross the diagonal, the window edge or a document boundary, and grouped-query
``dk, dv`` are summed over the query heads inside the kernel. ``backward="fused"``
runs one kernel for ``dq, dk, dv`` (5 matrix products per block instead of 7, as in
the FlashAttention-3 backward): the ``dk, dv`` kernel also multiplies ``dS``, staged in
shared memory, by the key tile and adds the fp32 product to ``dq`` with TMA reductions
(``reduction_op="add"``), so ``dq`` is not bitwise reproducible across runs.
``backward="full"`` keeps the previous kernels (every block visited, one consumer
warpgroup). Without a sliding window the ranged and fused backward kernels run on a 1D grid
in FlashAttention-3's causal backward order (``bwd_lpt``): the (batch, head) pairs are cut into
sections whose queries, cotangents and ``dq`` sums fit in 40 MiB of L2, and inside a section
the tiles with the most blocks start first, every pair of the section at once. CUDA starts the
CTAs of a ``(heads, blocks, batch)`` Pallas grid head by head (Pallas gives the first axis to
CUDA's slowest-varying z), so there the longest tiles of the last heads started last and set
the kernel time (1.45x it with 8 heads of 4K tokens). Document
ranges come from the runs of equal segment ids. A row in which every id forms one
run (sorted or not, such as corpus document ids) skips blocks; a row in which some
id occurs in more than one run skips no block and masks every block, so any
segment layout stays exact.
Sequence lengths that are not a multiple of 128 are padded and the padding is
sliced off. Under ``jax.jit`` with sharded operands the forward and backward
kernels are partitioned along the batch and heads axes (``custom_partitioning``):
each device runs the kernels on its own batch rows and, under tensor parallelism,
its own heads (when every mesh axis divides ``Hkv``; otherwise the heads are
gathered before the call). A sharded sequence axis is gathered whole
first. Inside ``jax.shard_map`` the results and gradients carry their inputs'
varying mesh axes.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm

__all__ = ["flash_attention", "supported"]

Array = jax.Array

_LOG2E = math.log2(math.e)
_PAD_MULTIPLE = 128  # 2 * block_q of the forward and a multiple of every block size
_M_INIT = -1e30  # finite running max: a fully masked tile leaves acc and l at zero
_MAX_GRID_YZ = 65535  # CUDA's limit on grid y and z (see jla.ops.mosaic_gpu._common.MAX_GRID_YZ)


@dataclasses.dataclass(frozen=True)
class _Config:
    scale: float
    window: int | None
    has_segments: bool
    block_q: int = 64
    block_kv: int = 128
    max_concurrent_steps: int = 2
    block_q_bwd: int = 64
    block_kv_bwd: int = 64
    backward: str = "ranged"
    bwd_compute_wgs: int = 2
    # Key block of the ranged dq kernel (its query rows are block_q_bwd). None: 128 (1-4% faster than 64 on an H100),
    # or 64 for MLA's 192-wide keys, whose 128-key blocks do not fit in shared memory.
    block_kv_dq: int | None = None
    has_sinks: bool = False
    # dk, dv kernel: issue S^T and dP^T together, dS^T math under the dV product. None: on without segment ids and
    # without a window at q/k head dims up to 128 (1-4% faster there; slower by up to 1% with documents
    # or windows, and 14-16% slower for MLA's 192-wide q/k). See _overlap_dkv.
    overlap_dkv: bool | None = None
    heavy_first: bool = True  # forward and ranged dq: launch the query tiles with the most key blocks first
    unmask_documents: bool = True  # blocks inside one document (every id one run) skip the mask
    fused_overlap: bool = True  # backward="fused": issue S^T and dP^T together, P^T math under the dP^T product
    fwd_overlap: bool | None = None  # forward: QK^T of block j with PV of block j - 1 (None: D = 64, no segment ids)
    fwd_named_turn: bool = True  # fwd_overlap: pass the turn with PTX named barriers (else mbarriers)
    fwd_named_consumed: bool = True  # fwd_overlap: release key/value slots on named barriers (else mbarriers)
    fwd_wgs: int | None = None  # forward consumer warpgroups: 2, or 3 (D = 64; None: 3 with fwd_overlap)
    fwd_k_ahead: bool = False  # fwd_overlap: the producer loads keys one block ahead of values (FA3's order)
    fwd_persistent: bool | None = None  # forward: one persistent CTA per SM (attention_persistent); None: if it applies
    # backward="fused", D = 64: one dS^T barrier per step parity, which only the warpgroup forming that step's dq
    # waits on (the other one runs on). None: on without a sliding window (measured faster there, slower with one).
    fused_async_ds: bool | None = None
    # backward="fused": grid (key blocks, kv heads, batch) instead of (kv heads, key blocks, batch), so the CTAs that
    # run together share the q, do and dq sums of a few key/value heads in L2. None: with segment ids and D = 64.
    fused_kv_seq_first: bool | None = None
    # Backward kernels without a sliding window: a 1D grid in FA3's longest-first order of L2-sized sections
    # (_lpt_order) instead of the (heads, blocks, batch) grids, which CUDA starts head by head (the last heads'
    # longest tiles started last). backward="fused" keeps its key-block-first grid where that applies.
    bwd_lpt: bool = True


_BACKWARDS = ("ranged", "full", "fused")
# Supported (q/k head dim, v head dim) pairs: symmetric heads, and MLA's 128 no-RoPE + 64 RoPE
# query/key channels with values of 128 (DeepSeek-V2/V3, Kimi K2, Kimi Linear).
HEAD_DIMS = ((64, 64), (128, 128), (192, 128))
# Smallest default block size, read once here: callers (and tests) may swap ``_Config`` for a partial.
_MIN_BLOCK = min(_Config.block_q, _Config.block_kv, _Config.block_q_bwd, _Config.block_kv_bwd)


def supported(q: Array, k: Array, v: Array) -> str | None:
    """``None`` when the kernel accepts these operands, otherwise the reason it does not."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "q, k and v must be [B, T, H, D]"
    if (k.shape[:3] != v.shape[:3] or k.shape[0] != q.shape[0] or k.shape[1] != q.shape[1]
            or k.shape[3] != q.shape[3]):
        return f"k and v must be [B, T, Hkv, D] matching q {q.shape}; got {k.shape} and {v.shape}"
    if q.shape[2] % k.shape[2]:
        return f"num_heads {q.shape[2]} must be a multiple of num_kv_heads {k.shape[2]}"
    if (q.shape[3], v.shape[3]) not in HEAD_DIMS:
        return (f"head dims (q/k {q.shape[3]}, v {v.shape[3]}) must be one of "
                f"{', '.join(f'({a}, {b})' for a, b in HEAD_DIMS)}")
    if q.dtype not in (jnp.bfloat16, jnp.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q, k and v must all be bf16 or fp16; got {q.dtype}, {k.dtype}, {v.dtype}"
    if jax.default_backend() != "gpu":
        return "the Mosaic GPU attention kernel needs a CUDA GPU"
    capability = getattr(jax.devices()[0], "compute_capability", None)
    if capability is None or str(capability).split(".")[0] != "9":
        return f"the Mosaic GPU attention kernel needs sm_90 (Hopper); got {capability}"
    try:
        from jax._src.lib import cuda_versions
    except ImportError:
        cuda_versions = None
    version = cuda_versions.cuda_runtime_get_version() if cuda_versions is not None else None
    if version is not None and 12080 <= version < 12091:
        # Same guard as JAX's attention_mgpu.py: ptxas miscompiles causal masking there.
        return f"causal masking is miscompiled by ptxas on CUDA runtime 12.8.0-12.9.0; got {version}"
    # CUDA caps grid y and z at 65535 blocks; the kernels' grids are (heads, query or key blocks, batch).
    padded = -(-q.shape[1] // _PAD_MULTIPLE) * _PAD_MULTIPLE
    blocks = padded // _MIN_BLOCK
    if blocks > _MAX_GRID_YZ or q.shape[2] > _MAX_GRID_YZ:
        return (f"T={q.shape[1]} with {q.shape[2]} heads passes the CUDA grid limit of {_MAX_GRID_YZ} blocks "
                "per dimension; split the queries (or use a smaller sequence per call)")
    return None


_TURN_BARRIER = 5  # named barriers 5 .. 7, one per consumer warpgroup (Mosaic GPU uses 0 and 1 + warpgroup)
_CONSUMED_BARRIER = 8  # named barriers 8 .. 8 + 2 * max_concurrent_steps - 1 (keys, then values); 16 exist


def _named_barrier(op: str, barrier_id, num_threads: int) -> None:
    """PTX ``bar.sync`` / ``bar.arrive`` on named barrier ``barrier_id`` (warpgroup-uniform int32).

    A hardware barrier with no shared-memory state: ``arrive`` does not wait, ``sync`` waits until
    ``num_threads`` threads have arrived or synced. The overlapped forward passes its turns (as
    FlashAttention-3 does) and releases key/value slots on it instead of on an mbarrier, on which every
    thread of a warpgroup arrives. Adapted from Tokamax (Apache-2.0), tokamax/_src/mosaic_gpu.py
    (the dynamic-barrier-id branch of ``_bar_operation``, behind ``bar_arrive`` and ``bar_sync``).
    """
    @plgpu.inline_mgpu(arg_types=(plgpu.Layout.WG_SPLAT,))
    def bar(_, index):
        llvm.inline_asm(ir.Type.parse("!llvm.void"), [index.registers[()]], f"bar.{op} $0, {num_threads};",
                        "r", has_side_effects=True)

    bar(jnp.asarray(barrier_id, jnp.int32))


def _wg_kv_range(q_base, cfg: _Config):
    """Key-block range ``[lo, hi)`` visible to query rows ``[q_base, q_base + block_q)``."""
    bkv = jnp.int32(cfg.block_kv)
    hi = lax.div(q_base + cfg.block_q + cfg.block_kv - 1, bkv)
    if cfg.window is None:
        lo = jnp.int32(0)
    else:
        lo = lax.div(jnp.maximum(q_base - (cfg.window - 1), 0), bkv)
    return lo, hi


_L2_BWD_BYTES = 40 * 2**20  # bytes of queries, cotangents and dq sums one backward section keeps in L2 (FA3's)
# Registers of the TMA warpgroup of the two-warpgroup backward kernels, as FA3 (hopper/flash_bwd_kernel_sm90.h): the
# consumer warpgroups get 240 each instead of 232, which cut the spills of the D = 128 fused backward (1-3.5% faster).
_BWD_MEMORY_REGISTERS = 24


def _l2_section(pair_bytes: int) -> int:
    """(batch, head) pairs per section of the backward order: the largest power of two whose streamed
    bytes fit in ``_L2_BWD_BYTES``, at least one (FA3, hopper/tile_scheduler.hpp)."""
    fit = max(1, _L2_BWD_BYTES // max(1, pair_bytes))
    return 1 << (fit.bit_length() - 1)


def _lpt_order(pid, tiles: int, pairs: int, section: int):
    """``(rank, pair)`` of CTA ``pid`` on a 1D grid of ``tiles * pairs`` CTAs, longest first.

    CUDA starts CTAs in the order of their linear index, so this order decides which tiles start first: as
    FlashAttention-3's backward scheduler (``SingleTileBwdLPTScheduler``, design only) the pairs are cut into
    sections of ``section`` consecutive pairs (the last one narrower) and inside a section the CTAs run rank by
    rank with the section's pairs innermost. Rank 0 is the tile with the most blocks, so the longest tiles
    start first wherever the sequence starts a section, while the CTAs that run together read the queries,
    cotangents and ``dq`` sums of few heads (they stay in L2).
    """
    span = section * tiles
    sec = lax.div(pid, jnp.int32(span))
    inner = lax.rem(pid, jnp.int32(span))
    width = jnp.where(sec < pairs // section, jnp.int32(section), jnp.int32(max(pairs % section, 1)))
    return lax.div(inner, width), sec * section + lax.rem(inner, width)


def _index_tile(shape, dim: int, base, layout):
    """Positions ``base + i`` along axis ``dim`` of a WGMMA-layout tile: an index vector in the row (``dim = 0``) or
    column layout broadcast to the tile. The broadcast reuses the vector's registers, where a full iota tile holds one
    int32 register per element (64 per thread for a 64 x 128 tile; the masked steps then spilled)."""
    vec = plgpu.broadcasted_iota(jnp.int32, (shape[dim],), 0, layout=layout.reduce(1 - dim)) + base
    return lax.broadcast_in_dim(vec, shape, [dim])


def _tile_mask(q_pos, k_pos, seg_q, seg_k, cfg: _Config):
    """Visibility of keys ``k_pos`` from queries ``q_pos`` (int32 arrays of the tile shape)."""
    mask = q_pos >= k_pos
    if cfg.window is not None:  # q - k < window, without a difference tile (k + window repeats per key)
        mask = mask & (q_pos < k_pos + cfg.window)
    if cfg.has_segments:
        mask = mask & (seg_q == seg_k)
    return mask


def _delta(out: Array, do: Array) -> Array:
    """``[B, H, T]`` f32 row sums of ``out * do`` (the softmax backward's ``D``), exact in fp32.

    XLA (JAX 0.11.2) lowers this contraction to a multiply and a reduce; ``precision=HIGHEST`` keeps it
    fp32 should it ever become a (TF32) dot, as JAX main's ``attention_mgpu.py`` now requests.
    The kernels compute ``dP`` from the bf16/fp16 ``do``, so ``D`` must use those same values: when
    the cotangent was rounded down from fp32 in the same program, XLA (``xla_allow_excess_precision``,
    on by default) would fold this upcast into that rounding and sum the unrounded fp32 cotangent.
    Rows of ``dS`` then no longer sum to zero (``dq`` of a query that sees one key was 1e-2, not 0).
    The optimization barrier keeps the rounded values.
    """
    do = lax.optimization_barrier(do)
    return jnp.einsum("bqhd,bqhd->bhq", out.astype(jnp.float32), do.astype(jnp.float32),
                      precision=lax.Precision.HIGHEST)


def _segment_runs(seg: Array, contiguous: Array) -> Array:
    """``[B, 2, T]`` int32: first and last position of the run of equal ids around each token.

    The kernels use them as block bounds: queries see keys of their own run only when every id
    occurs in a single run, so rows with ``contiguous[b]`` False get the trivial bounds ``0`` and
    ``T - 1`` (no block is skipped). Both bounds are non-decreasing in the position.
    """
    batch, length = seg.shape
    pos = jnp.arange(length, dtype=jnp.int32)
    change = seg[:, 1:] != seg[:, :-1]
    edge = jnp.ones((batch, 1), bool)
    start = lax.cummax(jnp.where(jnp.concatenate([edge, change], axis=1), pos, 0), axis=1)
    end = lax.cummin(jnp.where(jnp.concatenate([change, edge], axis=1), pos, length - 1), axis=1,
                     reverse=True)
    start = jnp.where(contiguous[:, None], start, 0)
    end = jnp.where(contiguous[:, None], end, length - 1)
    return jnp.stack([start, end], axis=1).astype(jnp.int32)


def _contiguous_ids(seg: Array) -> Array:
    """``[B]`` bool: every id of the row forms a single run of adjacent tokens.

    Exact: the row has as many runs of equal ids as distinct ids. Monotone rows qualify, and so do
    unsorted ones such as corpus document ids ``[8812, 8812, 77, 77, 5021]``; ``[1, 1, 2, 1]``
    does not. Only these rows may skip blocks by their run bounds.
    """
    runs = jnp.sum(seg[:, 1:] != seg[:, :-1], axis=1)
    ordered = jnp.sort(seg, axis=1)
    distinct = jnp.sum(ordered[:, 1:] != ordered[:, :-1], axis=1)
    return runs == distinct


def _inner_runs(runs: Array, contiguous: Array, enabled: bool) -> Array:
    """``[B, 2, T]`` int32: run bounds that prove a block lies inside one document.

    A block of keys ``[k0, k1]`` and queries ``[q0, q1]`` with ``k1 <= q0`` is inside one run when
    ``start[q1] <= k0`` (equivalently ``end[k0] >= q1``). Only rows whose ids each form one run have such
    runs; the others (and every row when ``enabled`` is False) get the sentinels ``T`` and ``-1``,
    so their blocks always keep the mask.
    """
    length = runs.shape[2]
    ok = contiguous[:, None] & enabled
    return jnp.stack([jnp.where(ok, runs[:, 0], length), jnp.where(ok, runs[:, 1], -1)], axis=1)


def _forward(q, k, v, seg, runs, sink_init, cfg: _Config):
    batch, length, heads, head_dim = q.shape
    # Asymmetric heads (MLA): head_dim = dv + extra query/key channels. The first dv channels and the
    # extra ones sit in separate SMEM tiles (two products per score tile), so the query tile has the
    # output's width and stages it. Symmetric heads have no extra tiles.
    dv = v.shape[-1]
    extra = head_dim - dv
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    bq, bkv = cfg.block_q, cfg.block_kv
    mcs = min(cfg.max_concurrent_steps, length // bkv)
    c = cfg.scale * _LOG2E
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    col = plgpu.Layout.WGMMA.reduce(0)
    # The overlapped loop runs by default only where it measured faster: D = 64 without segment ids
    # (causal, with or without a sliding window), on 192-row tiles of three consumer warpgroups (FA3's
    # kBlockM = 192). With two warpgroups, with D = 128, and with documents (register spills) it measured
    # slower or no faster than the ping-pong loop (benchmarks/attention.md).
    small = head_dim == 64 and dv == 64
    plain = small and not cfg.has_segments
    overlap = plain if cfg.fwd_overlap is None else cfg.fwd_overlap
    # Consumer warpgroups per query tile (64 rows each): 2, or 3 (the registers of three D = 128
    # warpgroups do not fit). Three need the overlapped forward.
    wgs = (3 if overlap and small else 2) if cfg.fwd_wgs is None else cfg.fwd_wgs
    tq = wgs * bq
    if wgs not in (2, 3) or (wgs == 3 and (not overlap or not small)):
        raise ValueError(f"fwd_wgs={wgs}: 2, or 3 with fwd_overlap and D = 64; got D = {head_dim}")
    n_tiles = -(-length // tq)  # with three warpgroups the last tile may end inside the sequence
    if length % _PAD_MULTIPLE:
        raise ValueError(f"T = {length} must be a multiple of {_PAD_MULTIPLE}")
    if overlap and cfg.fwd_named_consumed and _CONSUMED_BARRIER + 2 * mcs > 16:  # 16 named barriers exist
        raise ValueError(f"the overlapped forward releases slots on named barriers: at most 4 stages, got {mcs}")

    def kernel(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, out_ref, lse_ref, scoped):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        wg = lax.axis_index("wg")
        smem, bars, consumed, schedule = scoped
        if extra:
            qo_smem2, qx_smem2, k_smem, kx_smem, v_smem, lse_smem2, segq_smem2, segk_smem = smem
        else:
            qo_smem2, k_smem, v_smem, lse_smem2, segq_smem2, segk_smem = smem
            qx_smem2 = kx_smem = None
        k_bars, v_bars, q_bars, segq_bars, segk_bars = bars
        k_consumed, v_consumed = consumed
        tile = lax.axis_index("q_seq")
        if cfg.heavy_first:  # causal tiles near the end of the sequence see the most key blocks
            tile = jnp.int32(n_tiles - 1) - tile
        tile_base = tile * tq

        def doc_lo(lo, q_first):
            # Keys before the run of the first query row cannot be seen by any row (runs are sorted).
            if not cfg.has_segments:
                return lo
            return jnp.maximum(lo, lax.div(runs_ref[b, 0, jnp.minimum(q_first, length - 1)], jnp.int32(bkv)))

        lo_all, _ = _wg_kv_range(tile_base, cfg)
        lo_all = doc_lo(lo_all, tile_base)
        _, hi_all = _wg_kv_range(jnp.minimum(tile_base + (wgs - 1) * bq, length - bq), cfg)  # last rows inside T

        def sync():
            plgpu.barrier_arrive(schedule)
            plgpu.barrier_wait(schedule)

        def slot_of(step):
            return lax.rem(step, jnp.int32(mcs))

        # fwd_named_consumed: the consumers release a key or value slot on a named barrier per slot
        # (the consumer warpgroups arrive, the producer warpgroup syncs: 128 * (wgs + 1) threads) instead of an
        # mbarrier on which all 128 threads of each warpgroup arrive. A named barrier holds one phase,
        # so a consumer releases a slot only after it waited for the slot's data (the producer has
        # then synced on that slot's previous phase).
        named_consumed = overlap and cfg.fwd_named_consumed

        def release(kind, slot):  # kind 0: key slot, 1: value slot
            if named_consumed:
                _named_barrier("arrive", _CONSUMED_BARRIER + kind * mcs + slot, 128 * (wgs + 1))
            else:
                plgpu.barrier_arrive((k_consumed, v_consumed)[kind].at[slot])

        def await_release(kind, slot):
            if named_consumed:
                _named_barrier("sync", _CONSUMED_BARRIER + kind * mcs + slot, 128 * (wgs + 1))
            else:
                plgpu.barrier_wait((k_consumed, v_consumed)[kind].at[slot])

        @pl.when(wg < wgs)
        def _compute():
            plgpu.set_max_registers(232 if wgs == 2 else 160, action="increase")
            qo_smem = qo_smem2.at[wg]
            qx_smem = qx_smem2.at[wg] if extra else None
            q_base = tile_base + wg * bq
            lo, hi = _wg_kv_range(q_base, cfg)
            lo = doc_lo(lo, q_base)
            # Three warpgroups: the last tile may end inside the sequence, and a warpgroup whose rows lie
            # past it loads nothing and only takes its turns (overlapped(..., idle=True)).
            in_sequence = q_base < length

            def load_queries():
                if extra:  # two copies arrive on the query barrier (num_arrivals=2)
                    q_rows = (b, pl.ds(q_base, bq), h)
                    plgpu.copy_gmem_to_smem(q_ref.at[(*q_rows, pl.ds(0, dv))], qo_smem, q_bars.at[wg])
                    plgpu.copy_gmem_to_smem(q_ref.at[(*q_rows, pl.ds(dv, extra))], qx_smem, q_bars.at[wg])
                else:
                    plgpu.copy_gmem_to_smem(q_ref.at[b, pl.ds(q_base, bq), h], qo_smem, q_bars.at[wg])
                if cfg.has_segments:
                    plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(q_base, bq)], segq_smem2.at[wg], segq_bars.at[wg])
                    plgpu.barrier_wait(segq_bars.at[wg])
                plgpu.barrier_wait(q_bars.at[wg])

            if wgs == 2:  # 128-row tiles always lie inside the (padded) sequence
                load_queries()
            else:
                pl.when(in_sequence)(load_queries)
            seg_q = plgpu.load(segq_smem2.at[wg], layout=row) if cfg.has_segments else None

            if cfg.has_sinks:  # the sink is a virtual key every query sees: start from its logit
                m_i = plgpu.load(sink_ref.at[0, h], layout=row, optimized=False)
                l_i = plgpu.load(sink_ref.at[1, h], layout=row, optimized=False)
            else:
                m_i = plgpu.layout_cast(jnp.full((bq,), _M_INIT, jnp.float32), row)
                l_i = plgpu.layout_cast(jnp.full((bq,), 0, jnp.float32), row)
            acc = plgpu.layout_cast(jnp.full((bq, dv), 0, jnp.float32), layout)

            def finish(acc, m_i, l_i):
                acc /= lax.broadcast_in_dim(l_i, (bq, dv), [0])
                qo_smem[...] = acc.astype(dtype)
                lse_smem = lse_smem2.at[wg]
                lse_smem[...] = m_i + jnp.log(l_i) * _LOG2E
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(qo_smem, out_ref.at[b, pl.ds(q_base, bq), h])
                plgpu.copy_smem_to_gmem(lse_smem, lse_ref.at[b, h, pl.ds(q_base, bq)])
                plgpu.wait_smem_to_gmem(0)

            def overlapped(acc, m_i, l_i, idle=False):
                """FA3's intra-warpgroup overlap with ping-pong scheduling (design of Dao-AILab
                flash-attention, hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp, no code copied).

                Step ``t`` issues ``S(t) = Q K(t)^T`` and ``O += P(t - 1) V(t - 1)`` back to back, waits
                for ``S(t)``, releases ``K(t)``, reads ``O`` (which waits for the PV product too) and runs
                the softmax of block ``t``, then rescales ``O``. The warpgroups take turns in order
                (warpgroup ``w`` waits for its turn, issues its products and passes the turn to ``w + 1``,
                on named barrier ``_TURN_BARRIER + w`` or mbarrier ``schedule[w]``), so the tensor cores
                alternate between them and each warpgroup's softmax runs under the others' products. Every warpgroup
                takes ``hi_all - lo_all + 1`` turns (blocks only the others see are passed through),
                which keeps the turns, and the slots' barrier phases, in step.
                """
                if cfg.fwd_named_turn:  # hardware named barriers: one per warpgroup's turn
                    def take_turn():
                        _named_barrier("sync", _TURN_BARRIER + wg, 256)

                    def pass_turn():
                        _named_barrier("arrive", _TURN_BARRIER + lax.rem(wg + 1, jnp.int32(wgs)), 256)

                    def start():
                        _named_barrier("arrive", _TURN_BARRIER, 256)
                else:  # mbarriers schedule[0], schedule[1] (one warpgroup arrival per phase)
                    def take_turn():
                        plgpu.barrier_wait(schedule.at[wg])

                    def pass_turn():
                        plgpu.barrier_arrive(schedule.at[lax.rem(wg + 1, jnp.int32(wgs))])

                    def start():
                        plgpu.barrier_arrive(schedule.at[0])

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
                        q_pos = _index_tile((bq, bkv), 0, q_base, layout)
                        k_pos = _index_tile((bq, bkv), 1, t * bkv, layout)
                        sq = sk = None
                        if cfg.has_segments:
                            sq = lax.broadcast_in_dim(seg_q, (bq, bkv), [0])
                            sk = lax.broadcast_in_dim(seg_k, (bq, bkv), [1])
                        s = jnp.where(_tile_mask(q_pos, k_pos, sq, sk, cfg), s, -jnp.inf)
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
                        # Only S(t) is waited for here, but o_ref is read below before the softmax of block t is
                        # traced, and an accumulator read waits for every WGMMA in flight: the softmax is
                        # emitted after the wait for P(t - 1) V(t - 1). Tracing it before the read measured no
                        # faster.
                        s = plgpu.wgmma_accumulator_load(s_ref, wait_n=1)
                        release(0, slot)
                        return o_ref[...], softmax(s, m_i, l_i, t, seg_k, masked)

                    o, (m_i, alpha, l_i, p16) = pl.run_scoped(
                        issue, plgpu.ACC((bq, dv), jnp.float32), plgpu.ACC((bq, bkv), jnp.float32))
                    release(1, prev)
                    acc = o * lax.broadcast_in_dim(alpha, o.shape, [0])
                    return acc, m_i, l_i, p16

                def last(t, acc, p16):  # O += P(hi - 1) V(hi - 1)
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
                    # A block only the other warpgroup sees (before this warpgroup's range or after it):
                    # wait for it (barrier phases stay in step, and a slot is released only after its
                    # data arrived) and release it within this warpgroup's turn, so no release runs ahead.
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

                pl.when(wg == wgs - 1)(start)  # warpgroup 0 takes the first turn
                if idle:  # rows past the sequence (three warpgroups, last tile): take the turns only
                    lax.fori_loop(lo_all, hi_all, skip, None)
                    take_turn()
                    pass_turn()
                    return

                full = lax.div(q_base, jnp.int32(bkv))  # blocks entirely in the causal past
                if cfg.has_segments:  # blocks inside the last query's document skip the mask as well
                    inside = lax.div(runs_ref[b, 2, q_base + bq - 1] + bkv - 1, jnp.int32(bkv))
                else:
                    inside = lo
                if cfg.window is not None:  # blocks crossing the window's left edge stay masked
                    edge = lax.div(jnp.maximum(q_base + bq - cfg.window, 0) + bkv - 1, jnp.int32(bkv))
                    inside = jnp.maximum(inside, edge)
                inside = jnp.clip(inside, lo, full)

                if cfg.window is not None or cfg.has_segments:  # otherwise both warpgroups start at 0
                    lax.fori_loop(lo_all, lo, skip, None)
                m_i, l_i, p16 = first(lo, m_i, l_i)
                carry = (acc, m_i, l_i, p16)
                nxt = lo + 1
                carry = lax.fori_loop(nxt, inside, functools.partial(step, masked=True), carry)
                carry = lax.fori_loop(jnp.maximum(inside, nxt), full, functools.partial(step, masked=False), carry)
                carry = lax.fori_loop(jnp.maximum(full, nxt), hi, functools.partial(step, masked=True), carry)
                acc, m_i, l_i, p16 = carry
                acc = last(hi - 1, acc, p16)
                lax.fori_loop(hi, hi_all, skip, None)
                return acc, m_i, l_i

            if overlap:
                if wgs == 2:
                    finish(*overlapped(acc, m_i, l_i))
                else:
                    pl.when(in_sequence)(lambda: finish(*overlapped(acc, m_i, l_i)))
                    pl.when(jnp.logical_not(in_sequence))(lambda: overlapped(acc, m_i, l_i, idle=True))
                return

            plgpu.barrier_wait(k_bars.at[slot_of(lo_all)])
            pl.when(wg == 1)(sync)

            def skip_before(step, carry):
                # A block the other warpgroup needs: keep barrier phases and the schedule in step.
                slot = slot_of(step)
                if cfg.has_segments:
                    plgpu.barrier_wait(segk_bars.at[slot])
                plgpu.barrier_arrive(k_consumed.at[slot])
                sync()
                plgpu.barrier_wait(v_bars.at[slot])
                sync()
                plgpu.barrier_arrive(v_consumed.at[slot])
                plgpu.barrier_wait(k_bars.at[slot_of(step + 1)])
                return carry

            def kv_step(step, carry, masked: bool):
                acc, m_i, l_i = carry
                slot = slot_of(step)
                if cfg.has_segments:
                    plgpu.barrier_wait(segk_bars.at[slot])
                    seg_k = plgpu.load(segk_smem.at[slot], layout=col)

                def compute_qk(acc_ref):
                    plgpu.wgmma(acc_ref, qo_smem, k_smem.at[slot].transpose((1, 0)))
                    if extra:
                        plgpu.wgmma(acc_ref, qx_smem, kx_smem.at[slot].transpose((1, 0)))
                    sync()
                    return acc_ref[...]

                qk = pl.run_scoped(compute_qk, plgpu.ACC((bq, bkv), jnp.float32))
                plgpu.barrier_arrive(k_consumed.at[slot])
                if masked:
                    q_pos = _index_tile((bq, bkv), 0, q_base, layout)
                    k_pos = _index_tile((bq, bkv), 1, step * bkv, layout)
                    sq = sk = None
                    if cfg.has_segments:
                        sq = lax.broadcast_in_dim(seg_q, (bq, bkv), [0])
                        sk = lax.broadcast_in_dim(seg_k, (bq, bkv), [1])
                    qk = jnp.where(_tile_mask(q_pos, k_pos, sq, sk, cfg), qk, -jnp.inf)

                m_ij = jnp.maximum(m_i, qk.max(axis=1) * c)
                alpha = jnp.exp2(m_i - m_ij)
                m_i = m_ij
                p = jnp.exp2(qk * c - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i *= alpha
                p16 = p.astype(dtype)
                l_i += p.sum(axis=1)
                acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                plgpu.barrier_arrive(schedule)
                plgpu.barrier_wait(v_bars.at[slot])
                plgpu.barrier_wait(schedule)

                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                    nxt = step + 1

                    @pl.when(nxt < hi)
                    def _():
                        plgpu.barrier_wait(k_bars.at[slot_of(nxt)])

                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_consumed.at[slot])
                return acc, m_i, l_i

            def skip_after(step, carry):
                slot = slot_of(step)
                plgpu.barrier_arrive(k_consumed.at[slot])
                plgpu.barrier_arrive(v_consumed.at[slot])
                sync()
                sync()
                return carry

            if cfg.window is not None or cfg.has_segments:  # otherwise both warpgroups start at 0
                lax.fori_loop(lo_all, lo, skip_before, None)
            carry = (acc, m_i, l_i)
            if not cfg.has_segments:
                full = lax.div(q_base, jnp.int32(bkv))  # blocks entirely in the causal past
                inside = lo
                if cfg.window is not None:  # blocks crossing the window's left edge stay masked
                    edge = lax.div(jnp.maximum(q_base + bq - cfg.window, 0) + bkv - 1, jnp.int32(bkv))
                    inside = jnp.clip(edge, lo, full)
                    carry = lax.fori_loop(lo, inside, functools.partial(kv_step, masked=True), carry)
                carry = lax.fori_loop(inside, full, functools.partial(kv_step, masked=False), carry)
                carry = lax.fori_loop(full, hi, functools.partial(kv_step, masked=True), carry)
            else:  # documents: blocks inside the last query's document skip the mask as well
                full = lax.div(q_base, jnp.int32(bkv))
                inside = lax.div(runs_ref[b, 2, q_base + bq - 1] + bkv - 1, jnp.int32(bkv))
                if cfg.window is not None:
                    edge = lax.div(jnp.maximum(q_base + bq - cfg.window, 0) + bkv - 1, jnp.int32(bkv))
                    inside = jnp.maximum(inside, edge)
                inside = jnp.clip(inside, lo, full)
                carry = lax.fori_loop(lo, inside, functools.partial(kv_step, masked=True), carry)
                carry = lax.fori_loop(inside, full, functools.partial(kv_step, masked=False), carry)
                carry = lax.fori_loop(full, hi, functools.partial(kv_step, masked=True), carry)
            acc, m_i, l_i = carry
            lax.fori_loop(hi, hi_all, skip_after, None)
            pl.when(wg == 0)(sync)
            finish(acc, m_i, l_i)

        @pl.when(wg == wgs)
        def _memory():
            plgpu.set_max_registers(40 if wgs == 2 else 24, action="decrease")
            kvh = lax.div(h, jnp.int32(group))

            def load_k(s, slot):
                if extra:  # two copies arrive on the key barrier (num_arrivals=2)
                    plgpu.copy_gmem_to_smem(k_ref.at[(*s, pl.ds(0, dv))], k_smem.at[slot], k_bars.at[slot])
                    plgpu.copy_gmem_to_smem(k_ref.at[(*s, pl.ds(dv, extra))], kx_smem.at[slot], k_bars.at[slot])
                else:
                    plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[slot], k_bars.at[slot])

            def load(step, slot):
                s = (b, pl.ds(step * bkv, bkv), kvh)
                load_k(s, slot)
                if cfg.has_segments:
                    plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(step * bkv, bkv)], segk_smem.at[slot],
                                            segk_bars.at[slot])
                plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[slot], v_bars.at[slot])

            if overlap and cfg.fwd_k_ahead:
                # FA3's producer order for the overlapped loop, which consumes K(t) and V(t - 1) in step t:
                # keys run one block ahead of values, so K(t + mcs) is loaded as soon as K(t) is released
                # (mid step t) instead of after V(t) (one step later).
                def load_keys(step):
                    slot = slot_of(step)
                    load_k((b, pl.ds(step * bkv, bkv), kvh), slot)
                    if cfg.has_segments:
                        plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(step * bkv, bkv)], segk_smem.at[slot],
                                                segk_bars.at[slot])

                def load_values(step):
                    slot = slot_of(step)
                    plgpu.copy_gmem_to_smem(v_ref.at[b, pl.ds(step * bkv, bkv), kvh], v_smem.at[slot],
                                            v_bars.at[slot])

                for i in range(mcs):
                    pl.when(lo_all + i < hi_all)(functools.partial(load_keys, lo_all + i))
                for i in range(mcs - 1):
                    pl.when(lo_all + i < hi_all)(functools.partial(load_values, lo_all + i))

                @pl.loop(lo_all, hi_all - mcs)
                def _(step):
                    await_release(0, slot_of(step))
                    load_keys(step + mcs)
                    pl.when(step > lo_all)(lambda: await_release(1, slot_of(step - 1)))
                    load_values(step + mcs - 1)

                @pl.when(hi_all - lo_all >= mcs)
                def _():
                    pl.when(hi_all - mcs - 1 >= lo_all)(lambda: await_release(1, slot_of(hi_all - mcs - 1)))
                    load_values(hi_all - 1)

                return

            for i in range(mcs):
                step = lo_all + i

                @pl.when(step < hi_all)
                def _():
                    load(step, slot_of(step))

            @pl.loop(lo_all, hi_all - mcs)
            def _(step):
                nxt = step + mcs
                slot = slot_of(step)
                s = (b, pl.ds(nxt * bkv, bkv), kvh)
                await_release(0, slot)
                load_k(s, slot)
                if cfg.has_segments:
                    plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(nxt * bkv, bkv)], segk_smem.at[slot],
                                            segk_bars.at[slot])
                await_release(1, slot)
                plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[slot], v_bars.at[slot])

    def entry(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, out_ref, lse_ref):
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)

        def tiles(*shape):
            return plgpu.SMEM(shape, dtype, transforms=(tiling, swizzle))

        if extra:  # query and key tiles split into their first dv channels and the extra ones
            qk_tiles = (tiles(wgs, bq, dv), tiles(wgs, bq, extra), tiles(mcs, bkv, dv), tiles(mcs, bkv, extra))
        else:
            qk_tiles = (tiles(wgs, bq, head_dim), tiles(mcs, bkv, head_dim))
        smem = (
            *qk_tiles,
            tiles(mcs, bkv, dv),
            plgpu.SMEM((wgs, bq), jnp.float32),
            plgpu.SMEM((wgs, bq), jnp.int32),
            plgpu.SMEM((mcs, bkv), jnp.int32),
        )
        copies = 2 if extra else 1  # TMA copies per query and per key tile
        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, seg_ref, runs_ref, sink_ref, out_ref, lse_ref, args),
            smem,
            (
                plgpu.Barrier(num_arrivals=copies, num_barriers=mcs),
                plgpu.Barrier(num_barriers=mcs),
                plgpu.Barrier(num_arrivals=copies, num_barriers=wgs),
                plgpu.Barrier(num_barriers=wgs),
                plgpu.Barrier(num_barriers=mcs),
            ),
            (plgpu.Barrier(num_arrivals=wgs, num_barriers=mcs),) * 2,
            # fwd_overlap: one turn barrier per consumer warpgroup; otherwise one shared schedule barrier
            plgpu.Barrier(num_barriers=wgs) if overlap else plgpu.Barrier(num_arrivals=2),
            collective_axes="wg",
        )

    out, lse = plgpu.kernel(
        entry,
        out_type=[
            jax.ShapeDtypeStruct((batch, length, heads, dv), dtype),
            jax.ShapeDtypeStruct((batch, heads, length), jnp.float32),
        ],
        grid=(heads, n_tiles, batch),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=wgs + 1,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q, k, v, seg, runs, sink_init)
    return out, lse


def _backward(cfg: _Config, q, k, v, seg, out, lse, do):
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    n_q, n_kv = length // bq, length // bkv
    mcs = cfg.max_concurrent_steps
    c = cfg.scale * _LOG2E
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    col = plgpu.Layout.WGMMA.reduce(0)
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    delta = _delta(out, do)
    # A window of W keys spans bq + W - 1 positions per query tile (bkv + W - 1 per key tile), so
    # each tile visits a fixed number of consecutive blocks starting at its first visible one.
    if cfg.window is None:
        n_kv_visit, n_q_visit = n_kv, n_q
    else:
        n_kv_visit = min(n_kv, (bq + cfg.window - 1 + bkv - 1) // bkv + 1)
        n_q_visit = min(n_q, (bkv + cfg.window - 1 + bq - 1) // bq + 1)

    def kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, dq_ref, smem, bars):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        kvh = lax.div(h, jnp.int32(group))
        q_smem, do_smem, lse_smem, delta_smem, segq_smem = smem
        q_base = lax.axis_index("q_seq") * bq
        if cfg.window is None:
            kv_start = jnp.int32(0)
        else:
            kv_lo = lax.div(jnp.maximum(q_base - (cfg.window - 1), 0), jnp.int32(bkv))
            kv_start = jnp.minimum(kv_lo, jnp.int32(n_kv - n_kv_visit))

        def compute_thread(pipeline_callback):
            q_slice = (b, pl.ds(q_base, bq), h)
            plgpu.copy_gmem_to_smem(q_ref.at[q_slice], q_smem, bars[0])
            plgpu.copy_gmem_to_smem(do_ref.at[q_slice], do_smem, bars[1])
            plgpu.copy_gmem_to_smem(lse_ref.at[b, h, pl.ds(q_base, bq)], lse_smem, bars[2])
            plgpu.copy_gmem_to_smem(delta_ref.at[b, h, pl.ds(q_base, bq)], delta_smem, bars[3])
            plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(q_base, bq)], segq_smem, bars[4])
            for bar in bars:
                plgpu.barrier_wait(bar)
            lse_v = plgpu.load(lse_smem, layout=row)
            delta_v = plgpu.load(delta_smem, layout=row)
            seg_q = plgpu.load(segq_smem, layout=row)
            dq_acc = plgpu.layout_cast(jnp.full((bq, head_dim), 0, jnp.float32), layout)
            dq, _, _, _ = pipeline_callback((dq_acc, lse_v, delta_v, seg_q))
            q_smem[...] = dq.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[q_slice])
            plgpu.wait_smem_to_gmem(0)

        def kv_body(idx, k_smem, v_smem, segk_smem, k_c, v_c, segk_c, carry):
            dq_acc, lse_v, delta_v, seg_q = carry
            (kv_step,) = idx

            def compute_s(acc_ref):
                plgpu.wgmma(acc_ref, q_smem, k_smem.transpose((1, 0)))
                return acc_ref[...]

            s = pl.run_scoped(compute_s, plgpu.ACC((bq, bkv), jnp.float32))
            seg_k = plgpu.load(segk_smem, layout=col)
            plgpu.barrier_arrive(segk_c)
            q_pos = plgpu.broadcasted_iota(jnp.int32, (bq, bkv), 0, layout=layout) + q_base
            k_pos = plgpu.broadcasted_iota(jnp.int32, (bq, bkv), 1, layout=layout) + (kv_start + kv_step) * bkv
            mask = _tile_mask(q_pos, k_pos, lax.broadcast_in_dim(seg_q, (bq, bkv), [0]),
                              lax.broadcast_in_dim(seg_k, (bq, bkv), [1]), cfg)
            p = jnp.exp2(s * c - lax.broadcast_in_dim(lse_v, (bq, bkv), [0]))
            p = jnp.where(mask, p, 0.0)

            def compute_dp(acc_ref):
                plgpu.wgmma(acc_ref, do_smem, v_smem.transpose((1, 0)))
                return acc_ref[...]

            dp = pl.run_scoped(compute_dp, plgpu.ACC((bq, bkv), jnp.float32))
            plgpu.barrier_arrive(v_c)
            ds = p * (dp - lax.broadcast_in_dim(delta_v, (bq, bkv), [0])) * cfg.scale

            def compute_dq(acc_ref):
                plgpu.wgmma(acc_ref, ds.astype(dtype), k_smem)

            dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
            plgpu.barrier_arrive(k_c)
            return dq_acc, lse_v, delta_v, seg_q

        pipeline = plgpu.emit_pipeline_warp_specialized(
            kv_body,
            grid=(n_kv_visit,),
            max_concurrent_steps=min(mcs, n_kv_visit),
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=[
                plgpu.BlockSpec(block_shape=(bkv, head_dim), index_map=lambda i: (kv_start + i, 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bkv, head_dim), index_map=lambda i: (kv_start + i, 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bkv,), index_map=lambda i: (kv_start + i,)),
            ],
        )
        pipeline(k_ref.at[b, :, kvh, :], v_ref.at[b, :, kvh, :], seg_ref.at[b, :])

    def kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, dk_ref, dv_ref, smem, bars):
        b = lax.axis_index("batch")
        h = lax.axis_index("heads")
        kvh = lax.div(h, jnp.int32(group))
        k_smem, v_smem, segk_smem = smem
        kv_base = lax.axis_index("kv_seq") * bkv
        if cfg.window is None:
            q_start = jnp.int32(0)
        else:
            q_start = jnp.minimum(lax.div(kv_base, jnp.int32(bq)), jnp.int32(n_q - n_q_visit))

        def compute_thread(pipeline_callback):
            kv_slice = (b, pl.ds(kv_base, bkv), kvh)
            plgpu.copy_gmem_to_smem(k_ref.at[kv_slice], k_smem, bars[0])
            plgpu.copy_gmem_to_smem(v_ref.at[kv_slice], v_smem, bars[1])
            plgpu.copy_gmem_to_smem(seg_ref.at[b, pl.ds(kv_base, bkv)], segk_smem, bars[2])
            for bar in bars:
                plgpu.barrier_wait(bar)
            seg_k = plgpu.load(segk_smem, layout=row)
            zeros = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            dk, dv, _ = pipeline_callback((zeros, zeros, seg_k))
            k_smem[...] = dk.astype(dtype)
            v_smem[...] = dv.astype(dtype)
            plgpu.commit_smem()
            out_slice = (b, pl.ds(kv_base, bkv), h)
            plgpu.copy_smem_to_gmem(k_smem, dk_ref.at[out_slice], commit_group=False)
            plgpu.copy_smem_to_gmem(v_smem, dv_ref.at[out_slice], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        def q_body(idx, q_smem, do_smem, lse_smem, delta_smem, segq_smem,
                   q_c, do_c, lse_c, delta_c, segq_c, carry):
            dk_acc, dv_acc, seg_k = carry
            (q_step,) = idx

            def compute_st(acc_ref):
                plgpu.wgmma(acc_ref, k_smem, q_smem.transpose((1, 0)))
                return acc_ref[...]

            st = pl.run_scoped(compute_st, plgpu.ACC((bkv, bq), jnp.float32))
            lse_v = plgpu.load(lse_smem, layout=col)
            plgpu.barrier_arrive(lse_c)
            seg_q = plgpu.load(segq_smem, layout=col)
            plgpu.barrier_arrive(segq_c)
            k_pos = plgpu.broadcasted_iota(jnp.int32, (bkv, bq), 0, layout=layout) + kv_base
            q_pos = plgpu.broadcasted_iota(jnp.int32, (bkv, bq), 1, layout=layout) + (q_start + q_step) * bq
            mask = _tile_mask(q_pos, k_pos, lax.broadcast_in_dim(seg_q, (bkv, bq), [1]),
                              lax.broadcast_in_dim(seg_k, (bkv, bq), [0]), cfg)
            pt = jnp.exp2(st * c - lax.broadcast_in_dim(lse_v, (bkv, bq), [1]))
            pt = jnp.where(mask, pt, 0.0)

            def compute(refs):
                dv_acc_ref, dpt_ref = refs
                plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))

            zeros = plgpu.layout_cast(jnp.full((bkv, bq), 0, jnp.float32), layout)
            dv_acc, dpt = pl.run_state(compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
            plgpu.barrier_arrive(do_c)
            delta_v = plgpu.load(delta_smem, layout=col)
            plgpu.barrier_arrive(delta_c)
            dst = pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1])) * cfg.scale

            def compute_dk(acc_ref):
                plgpu.wgmma(acc_ref, dst.astype(dtype), q_smem)

            dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
            plgpu.barrier_arrive(q_c)
            return dk_acc, dv_acc, seg_k

        pipeline = plgpu.emit_pipeline_warp_specialized(
            q_body,
            grid=(n_q_visit,),
            max_concurrent_steps=min(mcs, n_q_visit),
            num_compute_wgs=1,
            memory_registers=40,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=[
                plgpu.BlockSpec(block_shape=(bq, head_dim), index_map=lambda i: (q_start + i, 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bq, head_dim), index_map=lambda i: (q_start + i, 0),
                                transforms=[tiling, swizzle]),
                plgpu.BlockSpec(block_shape=(bq,), index_map=lambda i: (q_start + i,)),
                plgpu.BlockSpec(block_shape=(bq,), index_map=lambda i: (q_start + i,)),
                plgpu.BlockSpec(block_shape=(bq,), index_map=lambda i: (q_start + i,)),
            ],
        )
        pipeline(q_ref.at[b, :, h, :], do_ref.at[b, :, h, :], lse_ref.at[b, h, :],
                 delta_ref.at[b, h, :], seg_ref.at[b, :])

    params = plgpu.CompilerParams(approx_math=True)
    q_tile = plgpu.SMEM((bq, head_dim), dtype, transforms=(tiling, swizzle))
    dq = plgpu.kernel(
        kernel_dq,
        out_type=jax.ShapeDtypeStruct(q.shape, dtype),
        scratch_types=[
            (q_tile, q_tile, plgpu.SMEM((bq,), jnp.float32), plgpu.SMEM((bq,), jnp.float32),
             plgpu.SMEM((bq,), jnp.int32)),
            tuple(plgpu.Barrier() for _ in range(5)),
        ],
        compiler_params=params,
        grid=(heads, n_q, batch),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=2,
        thread_name="wg",
    )(q, k, v, do, lse, delta, seg)

    kv_tile = plgpu.SMEM((bkv, head_dim), dtype, transforms=(tiling, swizzle))
    per_head = jax.ShapeDtypeStruct((batch, length, heads, head_dim), dtype)
    dk, dv = plgpu.kernel(
        kernel_dkv,
        out_type=[per_head, per_head],
        scratch_types=[
            (kv_tile, kv_tile, plgpu.SMEM((bkv,), jnp.int32)),
            tuple(plgpu.Barrier() for _ in range(3)),
        ],
        compiler_params=params,
        grid=(heads, n_kv, batch),
        grid_names=("heads", "kv_seq", "batch"),
        num_threads=2,
        thread_name="wg",
    )(q, k, v, do, lse, delta, seg)
    if group > 1:
        shape = (batch, length, kv_heads, group, head_dim)
        dk = dk.reshape(shape).astype(jnp.float32).sum(axis=3).astype(dtype)
        dv = dv.reshape(shape).astype(jnp.float32).sum(axis=3).astype(dtype)
    return dq, dk, dv


def _backward_ranged(cfg: _Config, q, k, v, seg, runs, out, lse, do):
    """``dq`` and ``dk, dv`` kernels that visit only the blocks a tile can see (FA3 backward).

    A CTA runs ``bwd_compute_wgs`` consumer warpgroups of 64 rows each and one TMA warpgroup.
    A ``dq`` tile visits key blocks from the start of the window or of the first query's document
    up to the causal diagonal; a ``dk, dv`` tile visits query blocks from its diagonal up to the
    end of the window or of its last key's document, for every query head of its key/value head.
    """
    batch, length, heads, head_dim = q.shape
    head_dim_v = v.shape[-1]  # dP, dv and the cotangent use the value width (MLA: narrower than q, k)
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    wgs = cfg.bwd_compute_wgs
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    bkv_dq = cfg.block_kv_dq if cfg.block_kv_dq is not None else (128 if head_dim <= 128 else 64)
    overlap_dkv = _overlap_dkv(cfg, head_dim)
    tq, tkv = wgs * bq, wgs * bkv
    n_qb = length // bq
    window = cfg.window
    segments = cfg.has_segments
    mcs = cfg.max_concurrent_steps
    c = cfg.scale * _LOG2E
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    col = plgpu.Layout.WGMMA.reduce(0)
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    lpt = cfg.bwd_lpt and window is None  # windows: every tile sees about as many blocks (FA3 orders only causal)
    # dq: the CTAs of a section re-read the keys and values of their key/value heads, counted once per query head
    # (the longest tiles read all of them); dk, dv: the queries and cotangents of their group.
    section_dq = _l2_section(length * 2 * (head_dim + head_dim_v))
    section_dkv = _l2_section(group * length * 2 * (head_dim + head_dim_v))
    delta = _delta(out, do)

    def iota(shape, dim, base):
        return _index_tile(shape, dim, base, layout)

    def kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, dq_ref,
                  q_smems, do_smems, bars):
        if lpt:
            tile, pair = _lpt_order(lax.axis_index("tile"), length // tq, batch * heads, section_dq)
            b, h = lax.div(pair, jnp.int32(heads)), lax.rem(pair, jnp.int32(heads))
        else:
            b, h = lax.axis_index("batch"), lax.axis_index("heads")
            tile = lax.axis_index("q_seq")
        wg = lax.axis_index("wg")
        kvh = lax.div(h, jnp.int32(group))
        if cfg.heavy_first:
            tile = jnp.int32(length // tq - 1) - tile
        q0 = tile * tq
        q_base = q0 + wg * bq
        rows = pl.ds(q_base, bq)
        hi = lax.div(q0 + tq + bkv_dq - 1, jnp.int32(bkv_dq))
        lo = jnp.int32(0)
        if window is not None:
            lo = lax.div(jnp.maximum(q0 - (window - 1), 0), jnp.int32(bkv_dq))
        if segments:
            lo = jnp.maximum(lo, lax.div(runs_ref[b, 0, q0], jnp.int32(bkv_dq)))
            # Keys from here on share the document of this warpgroup's last query (runs row 2).
            doc_from = runs_ref[b, 2, jnp.minimum(q_base + bq - 1, length - 1)]
        q_smem, do_smem, bar = q_smems.at[wg], do_smems.at[wg], bars.at[wg]

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(q_ref.at[b, rows, h], q_smem, bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, rows, h], do_smem, bar)
            lse_v = plgpu.load(lse_ref.at[b, h, rows], layout=row, optimized=False)
            delta_v = plgpu.load(delta_ref.at[b, h, rows], layout=row, optimized=False)
            seg_q = plgpu.load(seg_ref.at[b, rows], layout=row, optimized=False) if segments else None
            plgpu.barrier_wait(bar)
            dq_acc = plgpu.layout_cast(jnp.full((bq, head_dim), 0, jnp.float32), layout)
            dq = pipeline_callback((dq_acc, lse_v, delta_v, seg_q))[0]
            q_smem[...] = (dq * cfg.scale).astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, dq_ref.at[b, rows, h])
            plgpu.wait_smem_to_gmem(0)

        def kv_body(idx, *args):
            *refs, carry = args
            n_in = len(refs) // 2
            smems, consumed = refs[:n_in], refs[n_in:]
            k_smem, v_smem = smems[0], smems[1]
            dq_acc, lse_v, delta_v, seg_q = carry
            k0 = (lo + idx[0]) * bkv_dq

            def probabilities(s):
                p = jnp.exp2(s * c - lax.broadcast_in_dim(lse_v, (bq, bkv_dq), [0]))
                need = k0 + bkv_dq - 1 > q_base  # the block crosses this warpgroup's diagonal
                if window is not None:
                    need = need | (q_base + bq - 1 - k0 >= window)  # or the window's left edge
                sq = sk = None
                if segments:
                    seg_k = plgpu.load(smems[2], layout=col)
                    sq = lax.broadcast_in_dim(seg_q, (bq, bkv_dq), [0])
                    sk = lax.broadcast_in_dim(seg_k, (bq, bkv_dq), [1])
                    need = need | (k0 < doc_from)  # or it may hold keys of another document

                def masked():
                    mask = _tile_mask(iota((bq, bkv_dq), 0, q_base), iota((bq, bkv_dq), 1, k0), sq, sk, cfg)
                    return jnp.where(mask, p, 0.0)

                return lax.cond(need, masked, lambda: p)

            def scores(s_ref, dp_ref):
                plgpu.wgmma(s_ref, q_smem, k_smem.transpose((1, 0)))
                plgpu.wgmma(dp_ref, do_smem, v_smem.transpose((1, 0)))
                # The probabilities are computed while the dP product is still in flight.
                p = probabilities(plgpu.wgmma_accumulator_load(s_ref, wait_n=1))
                return p, dp_ref[...]

            acc = plgpu.ACC((bq, bkv_dq), jnp.float32)
            p, dp = pl.run_scoped(scores, acc, acc)
            plgpu.barrier_arrive(consumed[1])
            if segments:
                plgpu.barrier_arrive(consumed[2])
            ds = p * (dp - lax.broadcast_in_dim(delta_v, (bq, bkv_dq), [0]))

            def compute_dq(acc_ref):
                plgpu.wgmma(acc_ref, ds.astype(dtype), k_smem)

            dq_acc = pl.run_state(compute_dq)(plgpu.ACC.init(dq_acc))
            plgpu.barrier_arrive(consumed[0])
            return dq_acc, lse_v, delta_v, seg_q

        in_specs = [
            plgpu.BlockSpec((bkv_dq, head_dim), lambda i: (lo + i, 0), transforms=(tiling, swizzle)),
            plgpu.BlockSpec((bkv_dq, head_dim_v), lambda i: (lo + i, 0), transforms=(tiling, swizzle)),
        ]
        operands = [k_ref.at[b, :, kvh, :], v_ref.at[b, :, kvh, :]]
        if segments:
            in_specs.append(plgpu.BlockSpec((bkv_dq,), lambda i: (lo + i,)))
            operands.append(seg_ref.at[b, :])
        plgpu.emit_pipeline_warp_specialized(
            kv_body,
            grid=(hi - lo,),
            max_concurrent_steps=mcs,
            num_compute_wgs=wgs,
            memory_registers=_BWD_MEMORY_REGISTERS,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=in_specs,
        )(*operands)

    def kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, dk_ref, dv_ref,
                   k_smems, v_smems, bars):
        if lpt:  # rank 0: the first key block, which the most queries see
            rank, pair = _lpt_order(lax.axis_index("tile"), length // tkv, batch * kv_heads, section_dkv)
            b, kvh = lax.div(pair, jnp.int32(kv_heads)), lax.rem(pair, jnp.int32(kv_heads))
            k0 = rank * tkv
        else:
            b, kvh = lax.axis_index("batch"), lax.axis_index("heads")
            k0 = lax.axis_index("kv_seq") * tkv
        wg = lax.axis_index("wg")
        kv_base = k0 + wg * bkv
        cols = pl.ds(kv_base, bkv)
        lo = lax.div(k0, jnp.int32(bq))
        hi = jnp.int32(n_qb)
        if window is not None:  # the last key of the tile is seen up to window - 1 positions later
            hi = jnp.minimum(hi, lax.div(k0 + tkv + window - 2, jnp.int32(bq)) + 1)
        if segments:
            hi = jnp.minimum(hi, lax.div(runs_ref[b, 1, k0 + tkv - 1], jnp.int32(bq)) + 1)
            # Queries up to here share the document of this warpgroup's first key (runs row 3).
            doc_to = runs_ref[b, 3, jnp.minimum(kv_base, length - 1)]
        k_smem, v_smem, bar = k_smems.at[wg], v_smems.at[wg], bars.at[wg]

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(k_ref.at[b, cols, kvh], k_smem, bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, cols, kvh], v_smem, bar)
            seg_k = plgpu.load(seg_ref.at[b, cols], layout=row, optimized=False) if segments else None
            plgpu.barrier_wait(bar)
            zeros_k = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            zeros_v = (zeros_k if head_dim_v == head_dim
                       else plgpu.layout_cast(jnp.full((bkv, head_dim_v), 0, jnp.float32), layout))
            dk, dv, _ = pipeline_callback((zeros_k, zeros_v, seg_k))
            k_smem[...] = (dk * cfg.scale).astype(dtype)
            v_smem[...] = dv.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(k_smem, dk_ref.at[b, cols, kvh], commit_group=False)
            plgpu.copy_smem_to_gmem(v_smem, dv_ref.at[b, cols, kvh], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        def q_body(idx, *args):
            *refs, carry = args
            n_in = len(refs) // 2
            smems, consumed = refs[:n_in], refs[n_in:]
            q_smem, do_smem, lse_smem, delta_smem = smems[:4]
            dk_acc, dv_acc, seg_k = carry
            qs = (lo + idx[0]) * bq

            def probabilities(st):
                lse_v = plgpu.load(lse_smem, layout=col)
                pt = jnp.exp2(st * c - lax.broadcast_in_dim(lse_v, (bkv, bq), [1]))
                need = qs < kv_base + bkv - 1  # the block crosses this warpgroup's diagonal
                if window is not None:
                    need = need | (qs + bq - 1 - kv_base >= window)  # or the window's edge
                sq = sk = None
                if segments:
                    seg_q = plgpu.load(smems[4], layout=col)
                    sq = lax.broadcast_in_dim(seg_q, (bkv, bq), [1])
                    sk = lax.broadcast_in_dim(seg_k, (bkv, bq), [0])
                    need = need | (qs + bq - 1 > doc_to)  # or it may hold queries of another document

                def masked():
                    mask = _tile_mask(iota((bkv, bq), 1, qs), iota((bkv, bq), 0, kv_base), sq, sk, cfg)
                    return jnp.where(mask, pt, 0.0)

                return lax.cond(need, masked, lambda: pt)

            if overlap_dkv:
                def scores(st_ref, dpt_ref):
                    plgpu.wgmma(st_ref, k_smem, q_smem.transpose((1, 0)))
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))
                    # P^T is computed while the dP^T product is still in flight.
                    return probabilities(plgpu.wgmma_accumulator_load(st_ref, wait_n=1)), dpt_ref[...]

                acc = plgpu.ACC((bkv, bq), jnp.float32)
                pt, dpt = pl.run_scoped(scores, acc, acc)
                plgpu.barrier_arrive(consumed[2])
                if segments:
                    plgpu.barrier_arrive(consumed[4])
                delta_v = plgpu.load(delta_smem, layout=col)
                plgpu.barrier_arrive(consumed[3])

                def products(refs):
                    dv_acc_ref, dk_acc_ref = refs
                    plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                    # dS^T is computed while the dV product is in flight.
                    dst = pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1]))
                    plgpu.wgmma(dk_acc_ref, dst.astype(dtype), q_smem)

                dv_acc, dk_acc = pl.run_state(products)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(dk_acc)))
                plgpu.barrier_arrive(consumed[1])
                plgpu.barrier_arrive(consumed[0])
                return dk_acc, dv_acc, seg_k

            def compute_st(acc_ref):
                plgpu.wgmma(acc_ref, k_smem, q_smem.transpose((1, 0)))
                return acc_ref[...]

            st = pl.run_scoped(compute_st, plgpu.ACC((bkv, bq), jnp.float32))
            pt = probabilities(st)
            plgpu.barrier_arrive(consumed[2])
            if segments:
                plgpu.barrier_arrive(consumed[4])

            def compute(refs):
                dv_acc_ref, dpt_ref = refs
                plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))

            zeros = plgpu.layout_cast(jnp.full((bkv, bq), 0, jnp.float32), layout)
            dv_acc, dpt = pl.run_state(compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
            plgpu.barrier_arrive(consumed[1])
            delta_v = plgpu.load(delta_smem, layout=col)
            plgpu.barrier_arrive(consumed[3])
            dst = pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1]))

            def compute_dk(acc_ref):
                plgpu.wgmma(acc_ref, dst.astype(dtype), q_smem)

            dk_acc = pl.run_state(compute_dk)(plgpu.ACC.init(dk_acc))
            plgpu.barrier_arrive(consumed[0])
            return dk_acc, dv_acc, seg_k

        def head(j):
            return kvh * group + j

        in_specs = [
            plgpu.BlockSpec((bq, None, head_dim), lambda i, j: (lo + i, head(j), 0),
                            transforms=(tiling, swizzle)),
            plgpu.BlockSpec((bq, None, head_dim_v), lambda i, j: (lo + i, head(j), 0),
                            transforms=(tiling, swizzle)),
            plgpu.BlockSpec((None, bq), lambda i, j: (head(j), lo + i)),
            plgpu.BlockSpec((None, bq), lambda i, j: (head(j), lo + i)),
        ]
        operands = [q_ref.at[b], do_ref.at[b], lse_ref.at[b], delta_ref.at[b]]
        if segments:
            in_specs.append(plgpu.BlockSpec((bq,), lambda i, j: (lo + i,)))
            operands.append(seg_ref.at[b])
        plgpu.emit_pipeline_warp_specialized(
            q_body,
            grid=(hi - lo, group),
            max_concurrent_steps=mcs,
            num_compute_wgs=wgs,
            memory_registers=_BWD_MEMORY_REGISTERS,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=in_specs,
        )(*operands)

    params = plgpu.CompilerParams(approx_math=True)

    def wg_tiles(rows, width=head_dim):
        return plgpu.SMEM((wgs, rows, width), dtype, transforms=(tiling, swizzle))

    dq = plgpu.kernel(
        kernel_dq,
        out_type=jax.ShapeDtypeStruct(q.shape, dtype),
        scratch_types=[wg_tiles(bq), wg_tiles(bq, head_dim_v), plgpu.Barrier(num_arrivals=2, num_barriers=wgs)],
        compiler_params=params,
        grid=(length // tq * heads * batch,) if lpt else (heads, length // tq, batch),
        grid_names=("tile",) if lpt else ("heads", "q_seq", "batch"),
        num_threads=wgs + 1,
        thread_name="wg",
    )(q, k, v, do, lse, delta, seg, runs)
    dk, dv = plgpu.kernel(
        kernel_dkv,
        out_type=[jax.ShapeDtypeStruct(k.shape, dtype), jax.ShapeDtypeStruct(v.shape, dtype)],
        scratch_types=[wg_tiles(bkv), wg_tiles(bkv, head_dim_v), plgpu.Barrier(num_arrivals=2, num_barriers=wgs)],
        compiler_params=params,
        grid=(length // tkv * kv_heads * batch,) if lpt else (kv_heads, length // tkv, batch),
        grid_names=("tile",) if lpt else ("heads", "kv_seq", "batch"),
        num_threads=wgs + 1,
        thread_name="wg",
    )(q, k, v, do, lse, delta, seg, runs)
    return dq, dk, dv


def _overlap_dkv(cfg: _Config, head_dim: int) -> bool:
    """``cfg.overlap_dkv`` of the ranged ``dk, dv`` kernel; None: on for causal attention without segment ids or a
    window at q/k head dims up to 128, where it measured 1-4% faster. Off for MLA's 192-wide q/k: there it made the
    ranged backward 14-16% slower (1.327 / 4.545 / 4.278 ms against 1.143 / 3.916 / 3.630 ms serial at
    (B, T, H) = (8, 2048, 16), (4, 4096, 32) and (2, 8192, 16), causal, on an H100)."""
    if cfg.overlap_dkv is not None:
        return bool(cfg.overlap_dkv)
    return cfg.window is None and not cfg.has_segments and head_dim <= 128


def _fused_async_ds(cfg: _Config) -> bool:
    """``cfg.fused_async_ds``; None: on without a sliding window."""
    return cfg.window is None if cfg.fused_async_ds is None else cfg.fused_async_ds


def _fused_kv_seq_first(cfg: _Config, head_dim: int) -> bool:
    """``cfg.fused_kv_seq_first``; None: with packed documents (segment ids) and ``D = 64``, where it was measured
    faster."""
    if cfg.fused_kv_seq_first is not None:
        return cfg.fused_kv_seq_first
    return cfg.has_segments and head_dim == 64


def _backward_fused(cfg: _Config, q, k, v, seg, runs, out, lse, do):
    """``dq, dk, dv`` in one kernel, as the FlashAttention-3 backward: 5 matrix products per block.

    The CTA layout, block ranges and masks are those of the ``dk, dv`` kernel of ``_backward_ranged``:
    a CTA owns ``2 * block_kv_bwd`` keys of one key/value head (two consumer warpgroups of 64 keys and
    one TMA warpgroup) and visits the query blocks of every query head of its group that can see them.
    Each step also forms the key tile's share of ``dq = dS K``, as FA3 does: both warpgroups stage
    their ``dS^T`` in one shared-memory tile (two buffers, so a warpgroup may run one step ahead) and
    meet at a barrier; with ``D = 128`` each warpgroup then multiplies ``dS`` (64 queries x 128 keys)
    by one 64-column half of the key tile, with ``D = 64`` the two warpgroups take the whole product
    on alternate steps (``fused_async_ds``, the default without a window: one barrier per step parity, which
    only the warpgroup forming that step's ``dq`` waits on, so the other one does not stop).
    The fp32 product is added to a zero-initialised ``[B, T, H, D]`` buffer with a TMA reduction
    (``reduction_op="add"``), issued asynchronously: the warpgroup waits only until the reduction has read
    its shared-memory tile, one step later. XLA scales and rounds ``dq``
    afterwards. ``fused_overlap`` (default) issues ``S^T`` and ``dP^T`` together and computes ``P^T``
    while ``dP^T`` is in flight, then ``dS^T`` under the ``dV`` product (2-6% faster than the serial
    order at ``D = 128``). Segment ids are read (from L1/L2) only for blocks that need the mask.
    The reductions make ``dq`` depend on the order in which CTAs add (fp32 rounding), as in FA3;
    ``backward="ranged"`` stays bitwise reproducible.
    """
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    wgs = 2
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    if (bq, bkv) != (64, 64):
        raise ValueError(f"backward='fused' needs 64-row query and key blocks; got {(bq, bkv)}")
    tkv = wgs * bkv
    n_qb = length // bq
    window = cfg.window
    segments = cfg.has_segments
    mcs = cfg.max_concurrent_steps
    c = cfg.scale * _LOG2E
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    col = plgpu.Layout.WGMMA.reduce(0)
    tiling = plgpu.TilingTransform((8, 64))
    swizzle = plgpu.SwizzleTransform(128)
    dq_cols = 64  # dq columns one warpgroup forms per product
    split = head_dim // dq_cols  # 2: one half per warpgroup and step; 1: alternate warpgroups
    # D = 64: the warpgroup that forms a step's dq waits for the other's dS^T rows; the other one does not wait
    alternate = split == 1 and _fused_async_ds(cfg)
    kv_seq_first = _fused_kv_seq_first(cfg, head_dim)
    # The key-block-first grid stays where it measured fastest (packed documents with D = 64, see _fused_kv_seq_first).
    lpt = cfg.bwd_lpt and window is None and cfg.fused_kv_seq_first is None and not kv_seq_first
    # Each (batch, key/value head) pair streams the queries and cotangents of its group (bf16) and adds to their
    # float32 dq sums.
    section = _l2_section(group * length * (2 * 2 * head_dim + 4 * head_dim))
    delta = _delta(out, do)

    def iota(shape, dim, base):
        return _index_tile(shape, dim, base, layout)

    def kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, dq_ref, dk_ref, dv_ref,
               k_tile, v_tile, ds_smem, dq_smem, bars, ds_ready, ds_full):
        if lpt:  # rank 0: the first key block, which the most queries see
            rank, pair = _lpt_order(lax.axis_index("tile"), length // tkv, batch * kv_heads, section)
            b, kvh = lax.div(pair, jnp.int32(kv_heads)), lax.rem(pair, jnp.int32(kv_heads))
            k0 = rank * tkv
        else:
            b, kvh = lax.axis_index("batch"), lax.axis_index("heads")
            k0 = lax.axis_index("kv_seq") * tkv
        wg = lax.axis_index("wg")
        kv_base = k0 + wg * bkv
        cols = pl.ds(kv_base, bkv)
        lo = lax.div(k0, jnp.int32(bq))
        hi = jnp.int32(n_qb)
        if window is not None:  # the last key of the tile is seen up to window - 1 positions later
            hi = jnp.minimum(hi, lax.div(k0 + tkv + window - 2, jnp.int32(bq)) + 1)
        if segments:
            hi = jnp.minimum(hi, lax.div(runs_ref[b, 1, k0 + tkv - 1], jnp.int32(bq)) + 1)
            # Queries up to here share the document of this warpgroup's first key (runs row 3).
            doc_to = runs_ref[b, 3, jnp.minimum(kv_base, length - 1)]
        own = pl.ds(wg * bkv, bkv)  # this warpgroup's rows of the key tile and of the shared dS^T
        k_smem, v_smem, bar = k_tile.at[own], v_tile.at[own], bars.at[wg]
        dq_buf = dq_smem.at[wg]
        c0 = wg * dq_cols if split == 2 else 0  # first dq column this warpgroup forms

        def meet():
            plgpu.barrier_arrive(ds_ready)
            plgpu.barrier_wait(ds_ready)

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(k_ref.at[b, cols, kvh], k_smem, bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, cols, kvh], v_smem, bar)
            plgpu.barrier_wait(bar)
            zeros = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            dk, dv = pipeline_callback((zeros, zeros))
            meet()  # the other warpgroup has finished reading this warpgroup's keys (its dq products)
            k_smem[...] = (dk * cfg.scale).astype(dtype)
            v_smem[...] = dv.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(k_smem, dk_ref.at[b, cols, kvh], commit_group=False)
            plgpu.copy_smem_to_gmem(v_smem, dv_ref.at[b, cols, kvh], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)  # also the last dq reductions

        def reduce_dq(part, qs, h):
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous reduction has read dq_buf
            dq_buf[...] = part
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dq_buf, dq_ref.at[b, pl.ds(qs, bq), h, pl.ds(c0, dq_cols)],
                                    reduction_op="add")

        def q_body(idx, *args):
            *refs, carry = args
            n_in = len(refs) // 2
            smems, consumed = refs[:n_in], refs[n_in:]
            q_smem, do_smem, lse_smem, delta_smem = smems[:4]
            dk_acc, dv_acc = carry
            step = idx[0] * group + idx[1]
            h = kvh * group + idx[1]
            qs = (lo + idx[0]) * bq
            ds_all = ds_smem.at[lax.rem(step, 2)]  # both warpgroups' dS^T (128 keys x 64 queries)

            def probabilities(st):
                lse_v = plgpu.load(lse_smem, layout=col)
                pt = jnp.exp2(st * c - lax.broadcast_in_dim(lse_v, (bkv, bq), [1]))
                need = qs < kv_base + bkv - 1  # the block crosses this warpgroup's diagonal
                if window is not None:
                    need = need | (qs + bq - 1 - kv_base >= window)  # or the window's edge
                if segments:
                    need = need | (qs + bq - 1 > doc_to)  # or it may hold queries of another document

                def masked():
                    sq = sk = None
                    if segments:  # read only here (cached in L1/L2): no ids held in registers or pipelined
                        seg_q = plgpu.load(seg_ref.at[b, pl.ds(qs, bq)], layout=col, optimized=False)
                        seg_k = plgpu.load(seg_ref.at[b, cols], layout=row, optimized=False)
                        sq = lax.broadcast_in_dim(seg_q, (bkv, bq), [1])
                        sk = lax.broadcast_in_dim(seg_k, (bkv, bq), [0])
                    mask = _tile_mask(iota((bkv, bq), 1, qs), iota((bkv, bq), 0, kv_base), sq, sk, cfg)
                    return jnp.where(mask, pt, 0.0)

                return lax.cond(need, masked, lambda: pt)

            def stage(dst):  # this warpgroup's dS^T rows, then wait for the other warpgroup's
                ds_all.at[own][...] = dst
                plgpu.commit_smem()
                if alternate:  # only the warpgroup that forms this step's dq waits for the other's rows
                    plgpu.barrier_arrive(ds_full.at[lax.rem(step, 2)])
                else:
                    meet()

            def dq_product(acc_ref):
                plgpu.wgmma(acc_ref, ds_all.transpose((1, 0)), k_tile.at[:, pl.ds(c0, dq_cols)])

            dq_zeros = plgpu.layout_cast(jnp.full((bq, dq_cols), 0, jnp.float32), layout)
            if cfg.fused_overlap:
                def scores(st_ref, dpt_ref):
                    plgpu.wgmma(st_ref, k_smem, q_smem.transpose((1, 0)))
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))
                    # P^T is computed while the dP^T product is still in flight.
                    return probabilities(plgpu.wgmma_accumulator_load(st_ref, wait_n=1)), dpt_ref[...]

                acc = plgpu.ACC((bkv, bq), jnp.float32)
                pt, dpt = pl.run_scoped(scores, acc, acc)
                plgpu.barrier_arrive(consumed[2])
                delta_v = plgpu.load(delta_smem, layout=col)
                plgpu.barrier_arrive(consumed[3])

                def products(refs):
                    dv_acc_ref, dk_acc_ref, dq_acc_ref = refs
                    plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                    # dS^T is computed while the dV product runs, staged while the dK product runs.
                    dst = (pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1]))).astype(dtype)
                    plgpu.wgmma(dk_acc_ref, dst, q_smem)
                    stage(dst)
                    if split == 2:
                        dq_product(dq_acc_ref)

                dv_acc, dk_acc, dq_part = pl.run_state(products)(
                    (plgpu.ACC.init(dv_acc), plgpu.ACC.init(dk_acc), plgpu.ACC.init(dq_zeros)))
                plgpu.barrier_arrive(consumed[1])
                plgpu.barrier_arrive(consumed[0])
            else:
                def compute_st(acc_ref):
                    plgpu.wgmma(acc_ref, k_smem, q_smem.transpose((1, 0)))
                    return acc_ref[...]

                pt = probabilities(pl.run_scoped(compute_st, plgpu.ACC((bkv, bq), jnp.float32)))
                plgpu.barrier_arrive(consumed[2])

                def compute(refs):
                    dv_acc_ref, dpt_ref = refs
                    plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))

                zeros = plgpu.layout_cast(jnp.full((bkv, bq), 0, jnp.float32), layout)
                dv_acc, dpt = pl.run_state(compute)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
                plgpu.barrier_arrive(consumed[1])
                delta_v = plgpu.load(delta_smem, layout=col)
                plgpu.barrier_arrive(consumed[3])
                dst = (pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1]))).astype(dtype)

                def products(refs):
                    dk_acc_ref, dq_acc_ref = refs
                    plgpu.wgmma(dk_acc_ref, dst, q_smem)
                    stage(dst)  # while the dK product runs
                    if split == 2:
                        dq_product(dq_acc_ref)

                dk_acc, dq_part = pl.run_state(products)((plgpu.ACC.init(dk_acc), plgpu.ACC.init(dq_zeros)))
                plgpu.barrier_arrive(consumed[0])

            if split == 2:
                reduce_dq(dq_part, qs, h)
            else:
                @pl.when(lax.rem(step, 2) == wg)
                def _():
                    if alternate:
                        plgpu.barrier_wait(ds_full.at[wg])  # this step's dS^T rows of both warpgroups

                    def compute_dq(acc_ref):
                        plgpu.wgmma(acc_ref, ds_all.transpose((1, 0)), k_tile)
                        return acc_ref[...]

                    reduce_dq(pl.run_scoped(compute_dq, plgpu.ACC((bq, head_dim), jnp.float32)), qs, h)

            return dk_acc, dv_acc

        def head(j):
            return kvh * group + j

        in_specs = [
            plgpu.BlockSpec((bq, None, head_dim), lambda i, j: (lo + i, head(j), 0),
                            transforms=(tiling, swizzle)),
            plgpu.BlockSpec((bq, None, head_dim), lambda i, j: (lo + i, head(j), 0),
                            transforms=(tiling, swizzle)),
            plgpu.BlockSpec((None, bq), lambda i, j: (head(j), lo + i)),
            plgpu.BlockSpec((None, bq), lambda i, j: (head(j), lo + i)),
        ]
        plgpu.emit_pipeline_warp_specialized(
            q_body,
            grid=(hi - lo, group),
            max_concurrent_steps=mcs,
            num_compute_wgs=wgs,
            memory_registers=_BWD_MEMORY_REGISTERS,
            wg_axis="wg",
            manual_consumed_barriers=True,
            compute_context=compute_thread,
            in_specs=in_specs,
        )(q_ref.at[b], do_ref.at[b], lse_ref.at[b], delta_ref.at[b])

    tile = plgpu.SMEM((tkv, head_dim), dtype, transforms=(tiling, swizzle))
    call = plgpu.kernel(
        kernel,
        out_type=(),
        scratch_types=[
            tile, tile,
            plgpu.SMEM((2, tkv, bq), dtype, transforms=(tiling, swizzle)),
            plgpu.SMEM((wgs, bq, dq_cols), jnp.float32,
                       transforms=(plgpu.TilingTransform((8, 32)), swizzle)),
            plgpu.Barrier(num_arrivals=2, num_barriers=wgs),
            plgpu.Barrier(num_arrivals=wgs),
            plgpu.Barrier(num_arrivals=wgs, num_barriers=2),  # ds_full: dS^T of step parity 0 and 1
        ],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        # Pallas gives the first grid axis to CUDA's z, which varies slowest in the order CTAs start: kv_seq_first
        # starts the first key block of every head, then the second, ...; the other grid runs head by head.
        grid=((length // tkv * kv_heads * batch,) if lpt else
              (length // tkv, kv_heads, batch) if kv_seq_first else (kv_heads, length // tkv, batch)),
        grid_names=(("tile",) if lpt else
                    ("kv_seq", "heads", "batch") if kv_seq_first else ("heads", "kv_seq", "batch")),
        num_threads=wgs + 1,
        thread_name="wg",
    )

    def run(refs):  # the kernel adds into the zeroed dq buffer and writes dk, dv whole
        call(q, k, v, do, lse, delta, seg, runs, *refs)

    # run_state keeps the buffers internal: the result is a pure value (custom_partitioning lowers no effects)
    dq_acc, dk, dv = pl.run_state(run)(
        (jnp.zeros(q.shape, jnp.float32), lax.empty(k.shape, dtype), lax.empty(v.shape, dtype)))
    dq = (dq_acc * cfg.scale).astype(dtype)
    return dq, dk, dv


def _sink_init(sinks: Array, block_q: int) -> Array:
    """``[2, H, block_q]`` f32: the running max and denominator a query starts from (log2 units)."""
    sink = sinks.astype(jnp.float32) * _LOG2E
    m0 = jnp.maximum(sink, _M_INIT)  # a -inf sink (no virtual key) starts like no sink at all
    init = jnp.stack([m0, jnp.exp2(sink - m0)])
    return jnp.broadcast_to(init[:, :, None], (2, sinks.shape[0], block_q))


def _mesh_axes(axes) -> tuple:
    """Mesh axis names of one ``PartitionSpec`` entry (``None``, a name or a tuple of names)."""
    return () if axes is None else tuple(axes) if isinstance(axes, tuple) else (axes,)


def _batch_partitioned(fn, *, cfg_first: bool, rule: str):
    """``fn`` (arrays and a static ``cfg``) partitioned along batch and heads under SPMD jit.

    A Mosaic GPU kernel is an opaque custom call: without a rule, the SPMD partitioner gathers its
    operands and every device runs the whole global batch for every head. Here every array (operand
    or result) is split along its batch factor ``b`` like the batch axis of ``q``, and along its
    heads factor (``h`` for query heads, ``kh`` for key/value heads) like the heads axis of ``q``
    (tensor parallelism), when every mesh axis divides ``Hkv``: contiguous head chunks keep the
    kernel's ``kv_head = head // (H // Hkv)`` mapping on every device. Otherwise the Shardy rule
    keeps the heads whole and they are gathered before the call. All other axes are kept whole
    (``need_replication_factors``), so each device runs the kernel on its own batch rows and heads.
    A single device lowers ``fn`` directly.
    """
    from jax.experimental.custom_partitioning import custom_partitioning
    from jax.sharding import NamedSharding, PartitionSpec

    wrapped = custom_partitioning(fn, static_argnums=(0,) if cfg_first else (fn.__code__.co_argcount - 1,))
    operands, results = ([part.split() for part in side.split(",")] for side in rule.split("->"))

    def spec_entry(sharding, dim):
        spec = getattr(sharding, "spec", None) or ()
        return spec[dim] if len(spec) > dim else None

    def plan(mesh, arg_shapes):
        """Mesh axes of the batch and the heads, shared by every operand and result."""
        q, k = arg_shapes[0], arg_shapes[1]
        batch, heads = spec_entry(q.sharding, 0), spec_entry(q.sharding, 2)
        names = _mesh_axes(heads)
        if names:
            ways = math.prod(mesh.shape[name] for name in names)
            if set(names) & set(_mesh_axes(batch)) or q.shape[2] % ways or k.shape[2] % ways:
                heads = None
        return {"b": batch, "h": heads, "kh": heads}

    def shardings(mesh, axes, factor_lists):
        return tuple(NamedSharding(mesh, PartitionSpec(*(axes.get(f) for f in fs))) for fs in factor_lists)

    def infer(cfg, mesh, arg_shapes, result_shape):
        out = shardings(mesh, plan(mesh, arg_shapes), results)
        return out if isinstance(result_shape, (tuple, list)) else out[0]

    def partition(cfg, mesh, arg_shapes, result_shape):
        axes = plan(mesh, arg_shapes)
        out_shardings = shardings(mesh, axes, results)
        if not isinstance(result_shape, (tuple, list)):
            out_shardings = out_shardings[0]
        lower = functools.partial(fn, cfg) if cfg_first else (lambda *arrays: fn(*arrays, cfg))
        return mesh, lower, out_shardings, shardings(mesh, axes, operands)

    # For Shardy the query-heads axis is the compound factor (kh g), key/value heads major and the
    # group size g kept whole, so Shardy shards the query heads only along with the key/value heads
    # (contiguous chunks of whole groups) and the partition above never refuses a head sharding
    # Shardy chose (on GPU that disagreement gave wrong gradients). The heads may be split only when
    # every mesh axis divides Hkv; otherwise Shardy would need sub-axes, which the partition cannot
    # express, so both head factors stay whole: Shardy gathers the heads before the call and the
    # kernel runs every head (correct, redundant under tensor parallelism). Size-1 factors cannot be
    # part of a compound, so with one query head per key/value head the query heads are kh itself.
    def sdy_rule(heads: tuple[str, ...], split: bool) -> tuple[str, tuple[str, ...]]:
        dims = " ".join(heads) if len(heads) == 1 else f"({' '.join(heads)})"
        rule_str = " -> ".join(", ".join(" ".join(dims if f == "h" else f for f in fs) for fs in side)
                               for side in (operands, results))
        # Every factor but the batch (and the key/value heads when split) stays whole, listed in
        # order of first appearance (Shardy's factor order).
        order = (f for fs in operands + results for x in fs for f in (heads if x == "h" else (x,)))
        free = ("b", "kh") if split else ("b",)
        return rule_str, tuple(f for f in dict.fromkeys(order) if f not in free)

    def sharding_rule(*args):  # (*static_args, mesh, operand types, result types)
        q_dims, k_dims = (_type_dims(t) for t in args[-2][:2])
        kv_heads, group = k_dims[2], q_dims[2] // k_dims[2]
        sizes = tuple(dict(getattr(args[-3], "shape", {}) or {}).values())
        if kv_heads == 1 or any(kv_heads % size for size in sizes):
            rule_str, replicated = sdy_rule(("h",), split=False)
            return rule_str, dict(need_replication_factors=replicated)
        if group == 1:
            rule_str, replicated = sdy_rule(("kh",), split=True)
            return rule_str, dict(need_replication_factors=replicated)
        rule_str, replicated = sdy_rule(("kh", "g"), split=True)
        return rule_str, dict(g=group, need_replication_factors=replicated)

    wrapped.def_partition(partition=partition, infer_sharding_from_operands=infer, sharding_rule=sharding_rule)
    return wrapped


def _type_dims(value_type) -> tuple[int, ...]:
    """Shape of an MLIR tensor type (an operand type handed to a Shardy sharding-rule callable)."""
    shape = getattr(value_type, "shape", None)
    if shape is None:
        from jax._src.lib.mlir import ir

        shape = ir.RankedTensorType(value_type).shape
    return tuple(shape)


# --- persistent forward dispatch (jla.ops.mosaic_gpu.attention_persistent) -------------------------------------
def _forward_any(q, k, v, seg, runs, sink_init, cfg: _Config):
    """``_forward``, or the same forward from one persistent CTA per SM (bitwise equal; ``cfg.fwd_persistent``:
    None runs it wherever ``attention_persistent.applies`` and the device reports its SM count, True requires it,
    False keeps one CTA per tile)."""
    from jla.ops.mosaic_gpu import attention_persistent

    if (cfg.fwd_persistent is not False and attention_persistent.applies(q, k, v, cfg)
            and (cfg.fwd_persistent or attention_persistent.sm_count() is not None)):
        return attention_persistent.forward(q, k, v, seg, runs, sink_init, cfg)
    if cfg.fwd_persistent:
        raise ValueError("fwd_persistent=True: the persistent forward runs the kernel's default internal variants")
    return _forward(q, k, v, seg, runs, sink_init, cfg)


def persistent_option(persistent_forward: bool | None) -> dict:
    """The ``_Config`` keywords for ``AttentionConfig.persistent_forward`` (None: the config's own default)."""
    if persistent_forward not in (None, True, False):
        raise ValueError(f"persistent_forward must be None, True or False; got {persistent_forward!r}")
    return {} if persistent_forward is None else {"fwd_persistent": bool(persistent_forward)}
# --- end persistent forward dispatch ---------------------------------------------------------------------------


# q, k, v, seg, runs, sink_init -> out, lse (sink_init [2, H, block_q] has no batch axis); ``e`` is the
# value head dim (MLA's values are narrower than its queries and keys)
_forward_sharded = _batch_partitioned(
    _forward_any, cfg_first=False,
    rule="b t h d, b t kh d, b t kh e, b t, b r t, s h bq -> b t h e, b h t")
# q, k, v, seg, runs, out, lse, do -> dq, dk, dv
_backward_ranged_sharded = _batch_partitioned(
    _backward_ranged, cfg_first=True,
    rule="b t h d, b t kh d, b t kh e, b t, b r t, b t h e, b h t, b t h e -> b t h d, b t kh d, b t kh e")
# q, k, v, seg, runs, out, lse, do -> dq, dk, dv (one head dim: flash_attention refuses MLA's narrower values)
_backward_fused_sharded = _batch_partitioned(
    _backward_fused, cfg_first=True,
    rule="b t h d, b t kh d, b t kh d, b t, b r t, b t h d, b h t, b t h d -> b t h d, b t kh d, b t kh d")
# q, k, v, seg, out, lse, do -> dq, dk, dv
_backward_sharded = _batch_partitioned(
    _backward, cfg_first=True,
    rule="b t h d, b t kh d, b t kh d, b t, b t h d, b h t, b t h d -> b t h d, b t kh d, b t kh d")


def _vma(x) -> frozenset:
    """Mesh axes along which ``x`` varies inside ``jax.shard_map`` (empty outside one)."""
    aval = jax.typeof(x)
    mat = getattr(aval, "manual_axis_type", None)  # JAX 0.11: ShapedArray.manual_axis_type.varying
    return frozenset(mat.varying) if mat is not None else frozenset(getattr(aval, "vma", ()))


def _varying_like(x, *like):
    """``x`` (a kernel result, typed as not varying) marked as varying like the operands ``like``."""
    missing = tuple(sorted(frozenset().union(*map(_vma, like)) - _vma(x)))
    return lax.pcast(x, missing, to="varying") if missing else x


def _grad_like(grad, primal):
    """``grad`` with the varying axes of ``primal``: summed over axes only the gradient varies along
    (a replicated input used by every shard), marked varying along the rest."""
    extra = tuple(sorted(_vma(grad) - _vma(primal)))
    if extra:
        grad = lax.psum(grad, extra)
    return _varying_like(grad, primal)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _attention(q, k, v, seg, runs, sinks, cfg: _Config):
    out = _forward_sharded(q, k, v, seg, runs, _sink_init(sinks, cfg.block_q), cfg)[0]
    return _varying_like(out, q, k, v, seg, sinks)


def _attention_fwd(q, k, v, seg, runs, sinks, cfg: _Config):
    out, lse = _forward_sharded(q, k, v, seg, runs, _sink_init(sinks, cfg.block_q), cfg)
    out, lse = (_varying_like(x, q, k, v, seg, sinks) for x in (out, lse))
    return out, (q, k, v, seg, runs, sinks, out, lse)


def _attention_bwd(cfg: _Config, res, do):
    q, k, v, seg, runs, sinks, out, lse = res
    do = do.astype(q.dtype)
    if cfg.backward == "full":
        dq, dk, dv = _backward_sharded(cfg, q, k, v, seg, out, lse, do)
    elif cfg.backward == "fused":
        dq, dk, dv = _backward_fused_sharded(cfg, q, k, v, seg, runs, out, lse, do)
    else:
        dq, dk, dv = _backward_ranged_sharded(cfg, q, k, v, seg, runs, out, lse, do)
    # The kernel gradients vary wherever any of their operands does (the cotangent included).
    dq, dk, dv = (_varying_like(x, q, k, v, seg, out, lse, do) for x in (dq, dk, dv))
    if cfg.has_sinks:
        # d lse / d sink = p_sink = exp(sink - lse), and d out / d lse contracts with do to -delta.
        delta = _delta(out, do)
        p_sink = jnp.exp2(sinks.astype(jnp.float32)[None, :, None] * _LOG2E - lse)
        dsinks = -jnp.sum(p_sink * delta, axis=(0, 2)).astype(sinks.dtype)
    else:
        dsinks = jnp.zeros_like(sinks)
    # Inside jax.shard_map the kernel results carry no varying-axis type: give each gradient its
    # input's (summing d sinks over the shards when the sinks are replicated).
    return _grad_like(dq, q), _grad_like(dk, k), _grad_like(dv, v), None, None, _grad_like(dsinks, sinks)


_attention.defvjp(_attention_fwd, _attention_bwd)


def flash_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    segment_ids: Array | None = None,
    sliding_window: int | None = None,
    scale: float | None = None,
    backward: str = "ranged",
    sinks: Array | None = None,
    persistent_forward: bool | None = None,
) -> Array:
    """Causal GQA attention ``q [B, T, H, D]``, ``k [B, T, Hkv, D]``, ``v [B, T, Hkv, Dv]`` on the Mosaic GPU kernel.

    ``(D, Dv)`` is one of ``HEAD_DIMS``: ``(64, 64)``, ``(128, 128)`` or MLA's ``(192, 128)``
    (with ``backward="ranged"`` only).

    ``backward="ranged"`` (default) runs the block-skipping two-warpgroup backward kernels
    (``dq`` and ``dk, dv``, bitwise reproducible); ``backward="fused"`` one kernel for all three
    gradients with ``dq`` added up in fp32 by TMA reductions (fewer matrix products, but not
    bitwise reproducible, see ``benchmarks/attention.md`` for where it is faster);
    ``backward="full"`` the previous kernels that visit every block. ``sinks [H]`` adds a learned
    per-head logit to every softmax denominator (GPT-OSS attention sinks, differentiable).
    ``persistent_forward`` (``AttentionConfig.persistent_forward``): ``None`` runs the forward from one
    persistent CTA per SM wherever it applies, ``True`` requires it, ``False`` keeps one CTA per query tile.
    """
    reason = supported(q, k, v)
    if reason is not None:
        raise ValueError(f"mosaic_gpu attention: {reason}")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1; got {sliding_window}")
    if backward not in _BACKWARDS:
        raise ValueError(f"mosaic_gpu attention: backward must be one of {_BACKWARDS}; got {backward!r}")
    if backward in ("full", "fused") and v.shape[-1] != q.shape[-1]:
        raise ValueError(f"mosaic_gpu attention: backward={backward!r} needs one head dim for q, k and v; "
                         "use backward='ranged' for MLA's narrower values")
    batch, length = q.shape[:2]
    if segment_ids is not None and segment_ids.shape != (batch, length):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
    if sinks is not None and sinks.shape != (q.shape[2],):
        raise ValueError(f"mosaic_gpu attention: sinks must be [H] = {(q.shape[2],)}; got {sinks.shape}")
    cfg = _Config(
        scale=float(1.0 / math.sqrt(q.shape[-1]) if scale is None else scale),
        window=None if sliding_window is None or sliding_window >= length else int(sliding_window),
        has_segments=segment_ids is not None,
        backward=backward,
        has_sinks=sinks is not None,
        **persistent_option(persistent_forward),
    )
    seg = (jnp.zeros((batch, length), jnp.int32) if segment_ids is None
           else segment_ids.astype(jnp.int32))
    contiguous = _contiguous_ids(seg) if segment_ids is not None else None
    pad = -length % _PAD_MULTIPLE
    if pad:
        widths = ((0, 0), (0, pad), (0, 0), (0, 0))
        q, k, v = (jnp.pad(x, widths) for x in (q, k, v))
        # Padded queries only see real keys or each other; real queries never see them (causal).
        # Padded rows get zero cotangents, so their (possibly narrower) block ranges do not matter.
        seg = jnp.pad(seg, ((0, 0), (0, pad)), constant_values=jnp.iinfo(jnp.int32).min)
    if segment_ids is None:
        runs = jnp.zeros((batch, 4, seg.shape[1]), jnp.int32)  # not read by the kernels
    else:
        runs = _segment_runs(seg, contiguous)
        runs = jnp.concatenate([runs, _inner_runs(runs, contiguous, cfg.unmask_documents)], axis=1)
    sinks = jnp.zeros((q.shape[2],), jnp.float32) if sinks is None else sinks.astype(jnp.float32)
    out = _attention(q, k, v, seg, runs, sinks, cfg)
    return out[:, :length] if pad else out
