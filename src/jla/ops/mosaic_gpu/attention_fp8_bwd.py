# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2024 The JAX Authors.
# Portions Copyright 2025-2026 DeepMind Technologies Limited.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/attention_mgpu.py, and Tokamax (Apache-2.0),
# tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py, through the ranged and fused backward
# kernels of jla.ops.mosaic_gpu.attention, whose main loops are copied here (their block ranges, warpgroup layout
# and fused dq reductions follow Dao-AILab flash-attention (BSD-3-Clause), hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp,
# and their longest-first CTA order follows hopper/tile_scheduler.hpp's SingleTileBwdLPTScheduler; design only, no
# code copied). The FP8 epilogues are this module's own.
"""Attention backward kernels that write the q/k/v projection gradients of an FP8 train step.

In a delayed-scaling FP8 train step (:mod:`jla.layers.fp8`) the gradients of the attention's q, k and v
projections pass through XLA passes after the attention backward: the RoPE backward of ``dq`` and ``dk``,
the bfloat16 rounding of the projection outputs' cotangents, and their e5m2 cast with the delayed gradient
scale plus the amax the recipe records (``benchmarks/fp8_1p3b.md``). :func:`fp8_attention_backward` folds
them into the epilogues of the backward kernels, which hold every finished gradient tile in registers. For
each 64-row tile a consumer warpgroup

1. rounds the tile as the plain kernels do (``dq`` and ``dk`` times the softmax scale) to the compute dtype,
2. for ``dq`` and ``dk``, applies the RoPE backward in float32 with the step's ``cos``/``sin`` rows (loaded by
   TMA when the tile starts) and rounds the result to the compute dtype again (the projection output's
   cotangent),
3. records the largest absolute value of each row (``[B, H, T]`` float32 partial maxima: their maximum is
   the amax that ``jla.layers.fp8`` records for that gradient),
4. writes ``clip(g / scale)`` in e5m2, token-major ``[B, T, H, D]`` (the input-gradient GEMM's operand) and,
   with ``transposed=True``, also ``[H, D, B, T]`` (a weight-gradient GEMM operand; transposed in shared
   memory; XLA would still merge the three weight-gradient GEMMs and concatenate such copies), or, with
   ``merged=True`` (the fused backward), all three into one ``[H + 2 Hkv, D, B, T]`` operand of the merged
   weight-gradient GEMM, which the layer issues itself (``benchmarks/fp8_1p3b.md``).

``backward="ranged"``: the ``dq`` kernel sums over the key blocks in registers (``dq`` per query block, so no
float32 accumulation buffer and no conversion pass remain) and the ``dk, dv`` kernel sums the grouped query
heads in registers, as ``jla.ops.mosaic_gpu.attention._backward_ranged``. ``backward="fused"``: the fused
kernel (5 products per block) with the epilogues for ``dk, dv``, its ``dq`` added up in float32 by TMA
reductions as ``_backward_fused`` does (into a buffer that a small kernel zeroes, instead of the copy of a
shared zeros buffer XLA makes for ``jnp.zeros``), then one pass that runs the same epilogue on those sums.
The main loops are the plain kernels' (same block ranges, masks, document and window skipping and
accumulation order), so before the epilogue each tile holds the plain kernel's values. The RoPE products may contract
into fused multiply-adds differently from XLA's fusion of the same arithmetic, so on a GPU the bf16
cotangents can differ from XLA's in the last place, rarely (``tests/test_fp8_attention.py``).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu.attention import (_BWD_MEMORY_REGISTERS, _LOG2E, _Config, _delta, _fused_async_ds,
                                           _fused_kv_seq_first, _index_tile, _l2_section, _lpt_order, _tile_mask)

__all__ = ["BACKWARDS", "E5M2_MAX", "fp8_attention_backward", "fp8_backward_supported"]

Array = jax.Array

E5M2 = jnp.float8_e5m2
E5M2_MAX = float(jnp.finfo(E5M2).max)
BACKWARDS = ("ranged", "fused")
_ROWS = 64  # rows of a consumer warpgroup's tile in every kernel here (block_q_bwd = block_kv_bwd = 64)
_N_IN = 11  # q, k, v, do, lse, delta, seg, runs, cos, sin, gscale


def fp8_backward_supported(q: Array, k: Array, v: Array) -> str | None:
    """``None`` when :func:`fp8_attention_backward` takes these (post-RoPE) operands, else the reason."""
    if q.ndim != 4 or k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[3] != q.shape[3]:
        return f"q [B, T, H, D] and k, v [B, T, Hkv, D] must share B, T and D; got {q.shape}, {k.shape}, {v.shape}"
    if q.shape[3] not in (64, 128):
        return f"head dim {q.shape[3]} must be 64 or 128"
    if q.shape[2] % k.shape[2]:
        return f"num_heads {q.shape[2]} must be a multiple of num_kv_heads {k.shape[2]}"
    if q.shape[1] % 128:
        return f"T = {q.shape[1]} must be a multiple of 128"
    if q.dtype != jnp.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q, k and v must all be bf16; got {q.dtype}, {k.dtype}, {v.dtype}"
    return None


def _smem_fp8(shape: tuple[int, ...], minor: int):
    """An e5m2 tile whose rows hold ``minor`` bytes (64 or 128), swizzled for TMA."""
    return plgpu.SMEM(shape, E5M2, transforms=(plgpu.TilingTransform((8, minor)), plgpu.SwizzleTransform(minor)))


def _epilogue_smem(wgs: int, head_dim: int, dtype, transposed: bool) -> list:
    """Each consumer warpgroup's epilogue scratch (see :func:`_epilogue`)."""
    types = [
        _smem_fp8((wgs, _ROWS, head_dim), head_dim),  # token-major e5m2 tile
        plgpu.SMEM((wgs, _ROWS), jnp.float32),  # row maxima
        plgpu.SMEM((wgs, 2, _ROWS, head_dim // 2), jnp.float32),  # the tile's cos and sin rows
        plgpu.Barrier(num_arrivals=2, num_barriers=wgs),  # their two TMA loads
    ]
    if transposed:
        tiled = (plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128))
        types += [plgpu.SMEM((wgs, head_dim, _ROWS), dtype, transforms=tiled),  # transposed compute-dtype tile
                  _smem_fp8((wgs, head_dim, _ROWS), _ROWS)]  # transposed e5m2 tile
    return types


def _grad_types(batch: int, length: int, heads: int, head_dim: int, transposed: bool) -> list:
    """The outputs for ``heads`` heads: e5m2 ``[B, T, H, D]``, row maxima ``[B, H, T]``, e5m2 ``[H, D, B, T]``."""
    types = [jax.ShapeDtypeStruct((batch, length, heads, head_dim), E5M2),
             jax.ShapeDtypeStruct((batch, heads, length), jnp.float32)]
    if transposed:
        types.append(jax.ShapeDtypeStruct((heads, head_dim, batch, length), E5M2))
    return types


def _prefetch_rope(cos_ref, sin_ref, b, rows, scratch) -> None:
    """Start the TMA loads of the tile's ``cos`` and ``sin`` rows (see :func:`_rope_rows`)."""
    _, _, rope, rope_bar, *_ = scratch
    plgpu.copy_gmem_to_smem(cos_ref.at[b, rows], rope.at[0], rope_bar)
    plgpu.copy_gmem_to_smem(sin_ref.at[b, rows], rope.at[1], rope_bar)


def _rope_rows(scratch):
    """``(cos, sin)`` float32 ``[64, D / 2]`` (WGMMA layout) of :func:`_prefetch_rope`, once they have landed."""
    _, _, rope, rope_bar, *_ = scratch
    plgpu.barrier_wait(rope_bar)
    layout = plgpu.Layout.WGMMA
    return (plgpu.load(rope.at[0], layout=layout, optimized=False),
            plgpu.load(rope.at[1], layout=layout, optimized=False))


def _epilogue(acc, *, scale, rope, gscale, dtype, b, h, rows, outs, scratch, gt_head=None) -> None:
    """Write one finished float32 gradient tile ``acc [64, D]`` (WGMMA layout) of rows ``rows``, head ``h``.

    ``rope``: ``(cos, sin)`` of :func:`_rope_rows` to apply the RoPE backward, or ``None``; ``gscale`` the
    delayed e5m2 scale; ``outs`` the e5m2 output, the row maxima and, optionally, the transposed e5m2
    output; ``scratch`` this warpgroup's epilogue tiles; ``gt_head`` the transposed tile's head in that output
    (default ``h``; the merged operand of the fused backward puts k and v after the query heads). The stores are
    issued, not awaited.
    """
    g8_ref, am_ref, *gt_ref = outs
    o8, am, _, _, *transposed = scratch
    layout = plgpu.Layout.WGMMA
    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # an earlier tile's stores have read these buffers
    x = (acc if scale is None else acc * scale).astype(dtype)  # what the plain kernels write
    if rope is not None:
        # The RoPE backward of jla.layers.rope.apply_rope (its autodiff transpose, same order of terms).
        cos, sin = rope
        half = cos.shape[1]
        x = x.astype(jnp.float32)
        x1, x2 = x[:, :half], x[:, half:]
        x = jnp.concatenate([x2 * sin + x1 * cos, x2 * cos - x1 * sin], axis=1).astype(dtype)
    g = x.astype(jnp.float32)  # the projection output's cotangent, exactly
    am[...] = jnp.max(jnp.abs(g), axis=1)
    o8[...] = jnp.clip(g / gscale, -E5M2_MAX, E5M2_MAX).astype(E5M2)
    if gt_ref:
        t16, t8 = transposed
        t16.transpose((1, 0))[...] = x  # the compute-dtype tile, transposed in shared memory
        plgpu.commit_smem()
        gt = plgpu.load(t16, layout=layout).astype(jnp.float32)
        t8[...] = jnp.clip(gt / gscale, -E5M2_MAX, E5M2_MAX).astype(E5M2)
    plgpu.commit_smem()
    plgpu.copy_smem_to_gmem(o8, g8_ref.at[b, rows, h], commit_group=False)
    plgpu.copy_smem_to_gmem(am, am_ref.at[b, h, rows], commit_group=False)
    if gt_ref:
        plgpu.copy_smem_to_gmem(transposed[1], gt_ref[0].at[h if gt_head is None else gt_head, :, b, rows],
                                commit_group=False)
    plgpu.commit_smem_to_gmem_group()


def _results(flat, n_out: int, transposed: bool) -> tuple:
    """``(g8, gt or None, am)`` per projection from a kernel's flat outputs."""
    parts = [flat[i:i + n_out] for i in range(0, len(flat), n_out)]
    return tuple((p[0], p[2] if transposed else None, p[1]) for p in parts)


def _backward_ranged_fp8(cfg: _Config, q, k, v, seg, runs, out, lse, do, cos, sin, gscale, transposed: bool):
    """The ranged ``dq`` and ``dk, dv`` kernels with the FP8 epilogues (see the module docstring)."""
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    wgs = cfg.bwd_compute_wgs
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    if (bq, bkv) != (_ROWS, _ROWS):
        raise ValueError(f"the FP8 backward epilogues need 64-row query and key blocks; got {(bq, bkv)}")
    # 64-key dq blocks: the bf16 kernel's default of 128 was not measured with these epilogues.
    bkv_dq = cfg.block_kv_dq if cfg.block_kv_dq is not None else 64
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
    delta = _delta(out, do)
    n_out = 3 if transposed else 2
    epilogue = functools.partial(_epilogue, dtype=dtype)
    lpt = cfg.bwd_lpt and window is None  # the CTA order of attention._backward_ranged
    section_dq = _l2_section(length * 2 * 2 * head_dim)
    section_dkv = _l2_section(group * length * 2 * 2 * head_dim)

    def iota(shape, dim, base):
        return _index_tile(shape, dim, base, layout)

    def kernel_dq(*refs):
        q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, cos_ref, sin_ref, gs_ref = refs[:_N_IN]
        outs = refs[_N_IN:_N_IN + n_out]
        q_smems, do_smems, bars, *epi = refs[_N_IN + n_out:]
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
        scratch = tuple(ref.at[wg] for ref in epi)

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(q_ref.at[b, rows, h], q_smem, bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, rows, h], do_smem, bar)
            _prefetch_rope(cos_ref, sin_ref, b, rows, scratch)
            lse_v = plgpu.load(lse_ref.at[b, h, rows], layout=row, optimized=False)
            delta_v = plgpu.load(delta_ref.at[b, h, rows], layout=row, optimized=False)
            seg_q = plgpu.load(seg_ref.at[b, rows], layout=row, optimized=False) if segments else None
            plgpu.barrier_wait(bar)
            dq_acc = plgpu.layout_cast(jnp.full((bq, head_dim), 0, jnp.float32), layout)
            dq = pipeline_callback((dq_acc, lse_v, delta_v, seg_q))[0]
            epilogue(dq, scale=cfg.scale, rope=_rope_rows(scratch), gscale=gs_ref[0], b=b, h=h, rows=rows,
                     outs=outs, scratch=scratch)
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
            plgpu.BlockSpec((bkv_dq, head_dim), lambda i: (lo + i, 0), transforms=(tiling, swizzle)),
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

    def kernel_dkv(*refs):
        q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, cos_ref, sin_ref, gs_ref = refs[:_N_IN]
        outs_k = refs[_N_IN:_N_IN + n_out]
        outs_v = refs[_N_IN + n_out:_N_IN + 2 * n_out]
        k_smems, v_smems, bars, *epi = refs[_N_IN + 2 * n_out:]
        if lpt:
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
        scratch = tuple(ref.at[wg] for ref in epi)

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(k_ref.at[b, cols, kvh], k_smem, bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, cols, kvh], v_smem, bar)
            _prefetch_rope(cos_ref, sin_ref, b, cols, scratch)
            seg_k = plgpu.load(seg_ref.at[b, cols], layout=row, optimized=False) if segments else None
            plgpu.barrier_wait(bar)
            zeros = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            dk, dv, _ = pipeline_callback((zeros, zeros, seg_k))
            common = dict(b=b, h=kvh, rows=cols, scratch=scratch)
            epilogue(dk, scale=cfg.scale, rope=_rope_rows(scratch), gscale=gs_ref[1], outs=outs_k, **common)
            epilogue(dv, scale=None, rope=None, gscale=gs_ref[2], outs=outs_v, **common)
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

            if bool(cfg.overlap_dkv):
                def scores(st_ref, dpt_ref):
                    plgpu.wgmma(st_ref, k_smem, q_smem.transpose((1, 0)))
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))
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
            plgpu.BlockSpec((bq, None, head_dim), lambda i, j: (lo + i, head(j), 0),
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

    def wg_tiles(rows):
        return plgpu.SMEM((wgs, rows, head_dim), dtype, transforms=(tiling, swizzle))

    epilogue_smem = _epilogue_smem(wgs, head_dim, dtype, transposed)
    operands = (q, k, v, do, lse, delta, seg, runs, cos, sin, gscale)
    gq = plgpu.kernel(
        kernel_dq,
        out_type=_grad_types(batch, length, heads, head_dim, transposed),
        scratch_types=[wg_tiles(bq), wg_tiles(bq), plgpu.Barrier(num_arrivals=2, num_barriers=wgs), *epilogue_smem],
        compiler_params=params,
        grid=(length // tq * heads * batch,) if lpt else (heads, length // tq, batch),
        grid_names=("tile",) if lpt else ("heads", "q_seq", "batch"),
        num_threads=wgs + 1,
        thread_name="wg",
    )(*operands)
    gkv = plgpu.kernel(
        kernel_dkv,
        out_type=_grad_types(batch, length, kv_heads, head_dim, transposed) * 2,
        scratch_types=[wg_tiles(bkv), wg_tiles(bkv), plgpu.Barrier(num_arrivals=2, num_barriers=wgs),
                       *epilogue_smem],
        compiler_params=params,
        grid=(length // tkv * kv_heads * batch,) if lpt else (kv_heads, length // tkv, batch),
        grid_names=("tile",) if lpt else ("heads", "kv_seq", "batch"),
        num_threads=wgs + 1,
        thread_name="wg",
    )(*operands)
    return _results([*gq, *gkv], n_out, transposed)


_DQ_HEADS_PER_CTA = 8  # heads one CTA of the dq pass streams through (they share the RoPE rows)


def _zeros(shape: tuple[int, int, int, int]) -> Array:
    """A fresh float32 ``[B, T, H, D]`` of zeros, written by TMA from one zeroed shared-memory tile per CTA.

    The fused kernel adds ``dq`` into its buffer in place. Given ``jnp.zeros``, XLA materializes one zeros
    buffer for every layer and copies it into each call's buffer (a read and a write of the whole buffer per
    call); this buffer is written once and used by one call, so it is aliased without a copy.
    """
    batch, length, heads, head_dim = shape
    rows = _ROWS

    def kernel(out_ref, tile):
        b = lax.axis_index("batch")
        block = pl.ds(lax.axis_index("t") * rows, rows)
        tile[...] = plgpu.layout_cast(jnp.full((rows, head_dim), 0, jnp.float32), plgpu.Layout.WGMMA)
        plgpu.commit_smem()

        @pl.loop(0, heads)
        def _(h):
            plgpu.copy_smem_to_gmem(tile, out_ref.at[b, block, h])

        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(shape, jnp.float32),
        scratch_types=[plgpu.SMEM((rows, head_dim), jnp.float32,
                                  transforms=(plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128)))],
        grid=(length // rows, batch),
        grid_names=("t", "batch"),
    )()


def _dq_epilogue_kernel(cfg: _Config, dq_acc, cos, sin, gscale, dtype, transposed: bool, out_refs=None):
    """The FP8 epilogue of ``dq`` on the fused kernel's float32 ``dq`` sums ``[B, T, H, D]`` (one pass).

    A CTA of two warpgroups converts 128 query rows of up to ``_DQ_HEADS_PER_CTA`` heads: each warpgroup
    loads its 64 rows' RoPE rows once and streams the heads' float32 tiles through two TMA buffers (the
    next head's tile loads while :func:`_epilogue` converts the current one).
    With ``out_refs`` (``(e5m2, row maxima, merged transposed operand)`` refs, inside ``pl.run_state``) it writes
    into them instead of returning new arrays; ``dq_acc`` may then be a ref as well.
    """
    batch, length, heads, head_dim = dq_acc.shape
    wgs = 2
    tq = wgs * _ROWS
    per_cta = math.gcd(heads, _DQ_HEADS_PER_CTA)
    n_out = 3 if transposed else 2

    def kernel(acc_ref, cos_ref, sin_ref, gs_ref, *refs):
        outs = refs[:n_out]
        acc_smems, bars, *epi = refs[n_out:]
        b = lax.axis_index("batch")
        h0 = lax.axis_index("heads") * per_cta
        wg = lax.axis_index("wg")
        rows = pl.ds(lax.axis_index("q_seq") * tq + wg * _ROWS, _ROWS)
        bufs = acc_smems.at[wg]
        scratch = tuple(ref.at[wg] for ref in epi)
        gs = gs_ref[0]

        def fetch(i, slot):
            plgpu.copy_gmem_to_smem(acc_ref.at[b, rows, h0 + i], bufs.at[slot], bars.at[wg * 2 + slot])

        fetch(0, 0)
        _prefetch_rope(cos_ref, sin_ref, b, rows, scratch)
        rope = _rope_rows(scratch)

        @pl.loop(0, per_cta)
        def _(i):
            slot = lax.rem(i, 2)
            # The other buffer was read into registers one head ago (and fenced by that epilogue's commit).
            pl.when(i + 1 < per_cta)(lambda: fetch(i + 1, 1 - slot))
            plgpu.barrier_wait(bars.at[wg * 2 + slot])
            acc = plgpu.load(bufs.at[slot], layout=plgpu.Layout.WGMMA)
            _epilogue(acc, scale=cfg.scale, rope=rope, gscale=gs, dtype=dtype, b=b, h=h0 + i, rows=rows, outs=outs,
                      scratch=scratch)

        plgpu.wait_smem_to_gmem(0)

    f32_tiles = plgpu.SMEM((wgs, 2, _ROWS, head_dim), jnp.float32,
                           transforms=(plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128)))
    call = plgpu.kernel(
        kernel,
        out_type=_grad_types(batch, length, heads, head_dim, transposed) if out_refs is None else (),
        scratch_types=[f32_tiles, plgpu.Barrier(num_barriers=2 * wgs),
                       *_epilogue_smem(wgs, head_dim, dtype, transposed)],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=(heads // per_cta, length // tq, batch),
        grid_names=("heads", "q_seq", "batch"),
        num_threads=wgs,
        thread_name="wg",
    )
    return call(dq_acc, cos, sin, gscale) if out_refs is None else call(dq_acc, cos, sin, gscale, *out_refs)


def _backward_fused_fp8(cfg: _Config, q, k, v, seg, runs, out, lse, do, cos, sin, gscale, transposed: bool,
                        merged: bool = False):
    """The fused backward (``jla.ops.mosaic_gpu.attention._backward_fused``: 5 products per block, ``dq``
    added up in float32 by TMA reductions) with the FP8 epilogues for ``dk, dv`` in its kernel and for ``dq``
    in one pass over the float32 sums (:func:`_dq_epilogue_kernel`). ``D = 64``.

    ``merged``: the transposed e5m2 tiles of q, k and v go to one ``[H + 2 Hkv, D, B, T]`` operand (the query
    heads, then the key heads, then the value heads), which the dq pass completes in place: the operand of one
    merged weight-gradient GEMM (returned last, after the three ``(e5m2, None, row maxima)``).
    """
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    group = heads // kv_heads
    dtype = q.dtype
    wgs = 2
    bq, bkv = cfg.block_q_bwd, cfg.block_kv_bwd
    if (bq, bkv) != (_ROWS, _ROWS) or head_dim != 64:
        raise ValueError(f"the fused FP8 backward needs 64-row blocks and D = 64; got {(bq, bkv)}, D = {head_dim}")
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
    dq_cols = 64  # with D = 64 the two warpgroups take the whole dq product on alternate steps
    alternate = _fused_async_ds(cfg)  # as in _backward_fused
    kv_seq_first = _fused_kv_seq_first(cfg, head_dim)
    lpt = cfg.bwd_lpt and window is None and cfg.fused_kv_seq_first is None and not kv_seq_first  # as _backward_fused
    section = _l2_section(group * length * (2 * 2 * head_dim + 4 * head_dim))
    delta = _delta(out, do)
    n_out = 3 if transposed else 2
    if merged and transposed:
        raise ValueError("merged replaces the separate transposed copies")
    n_kv = 2 if merged else n_out  # outputs per projection of the kernel (merged: one operand after them)

    def iota(shape, dim, base):
        return _index_tile(shape, dim, base, layout)

    def kernel(*refs):
        q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, seg_ref, runs_ref, cos_ref, sin_ref, gs_ref = refs[:_N_IN]
        dq_ref = refs[_N_IN]
        outs_k = refs[_N_IN + 1:_N_IN + 1 + n_kv]
        outs_v = refs[_N_IN + 1 + n_kv:_N_IN + 1 + 2 * n_kv]
        if merged:  # the merged transposed operand, after the k and v outputs
            outs_k = (*outs_k, refs[_N_IN + 1 + 2 * n_kv])
            outs_v = (*outs_v, refs[_N_IN + 1 + 2 * n_kv])
        k_tile, v_tile, ds_smem, dq_smem, bars, ds_ready, ds_full, *epi = refs[_N_IN + 1 + 2 * n_kv + merged:]
        if lpt:
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
        scratch = tuple(ref.at[wg] for ref in epi)

        def meet():
            plgpu.barrier_arrive(ds_ready)
            plgpu.barrier_wait(ds_ready)

        def compute_thread(pipeline_callback):
            plgpu.copy_gmem_to_smem(k_ref.at[b, cols, kvh], k_smem, bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, cols, kvh], v_smem, bar)
            _prefetch_rope(cos_ref, sin_ref, b, cols, scratch)
            plgpu.barrier_wait(bar)
            zeros = plgpu.layout_cast(jnp.full((bkv, head_dim), 0, jnp.float32), layout)
            dk, dv = pipeline_callback((zeros, zeros))
            # The epilogue leaves the key tile and dS^T alone (the other warpgroup's last dq product may still
            # read them), so no meeting is needed here.
            common = dict(dtype=dtype, b=b, h=kvh, rows=cols, scratch=scratch)
            _epilogue(dk, scale=cfg.scale, rope=_rope_rows(scratch), gscale=gs_ref[1], outs=outs_k,
                      gt_head=heads + kvh if merged else None, **common)
            _epilogue(dv, scale=None, rope=None, gscale=gs_ref[2], outs=outs_v,
                      gt_head=heads + kv_heads + kvh if merged else None, **common)
            plgpu.wait_smem_to_gmem(0)  # also the last dq reductions

        def reduce_dq(part, qs, h):
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous reduction has read dq_buf
            dq_buf[...] = part
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dq_buf, dq_ref.at[b, pl.ds(qs, bq), h, pl.ds(0, dq_cols)], reduction_op="add")

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

            if cfg.fused_overlap:
                def scores(st_ref, dpt_ref):
                    plgpu.wgmma(st_ref, k_smem, q_smem.transpose((1, 0)))
                    plgpu.wgmma(dpt_ref, v_smem, do_smem.transpose((1, 0)))
                    return probabilities(plgpu.wgmma_accumulator_load(st_ref, wait_n=1)), dpt_ref[...]

                acc = plgpu.ACC((bkv, bq), jnp.float32)
                pt, dpt = pl.run_scoped(scores, acc, acc)
                plgpu.barrier_arrive(consumed[2])
                delta_v = plgpu.load(delta_smem, layout=col)
                plgpu.barrier_arrive(consumed[3])

                def products(refs):
                    dv_acc_ref, dk_acc_ref = refs
                    plgpu.wgmma(dv_acc_ref, pt.astype(dtype), do_smem)
                    dst = (pt * (dpt - lax.broadcast_in_dim(delta_v, (bkv, bq), [1]))).astype(dtype)
                    plgpu.wgmma(dk_acc_ref, dst, q_smem)
                    stage(dst)

                dv_acc, dk_acc = pl.run_state(products)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(dk_acc)))
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

                def products(dk_acc_ref):
                    plgpu.wgmma(dk_acc_ref, dst, q_smem)
                    stage(dst)  # while the dK product runs

                dk_acc = pl.run_state(products)(plgpu.ACC.init(dk_acc))
                plgpu.barrier_arrive(consumed[0])

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
            *_epilogue_smem(wgs, head_dim, dtype, transposed or merged),
        ],
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=((length // tkv * kv_heads * batch,) if lpt else
              (length // tkv, kv_heads, batch) if kv_seq_first else (kv_heads, length // tkv, batch)),
        grid_names=(("tile",) if lpt else
                    ("kv_seq", "heads", "batch") if kv_seq_first else ("heads", "kv_seq", "batch")),
        num_threads=wgs + 1,
        thread_name="wg",
    )

    def run(refs):  # the kernel adds into the zeroed dq buffer and writes the k and v gradients whole
        call(q, k, v, do, lse, delta, seg, runs, cos, sin, gscale, *refs)

    if merged:
        kv_types = _grad_types(batch, length, kv_heads, head_dim, False) * 2
        q_types = _grad_types(batch, length, heads, head_dim, False)
        gt_type = jax.ShapeDtypeStruct((heads + 2 * kv_heads, head_dim, batch, length), E5M2)

        def run_merged(refs):  # the fused kernel writes the k and v rows of gt, the dq pass its q rows
            dq_ref, g8k, amk, g8v, amv, gt, g8q, amq = refs
            call(q, k, v, do, lse, delta, seg, runs, cos, sin, gscale, dq_ref, g8k, amk, g8v, amv, gt)
            _dq_epilogue_kernel(cfg, dq_ref, cos, sin, gscale, dtype, True, out_refs=(g8q, amq, gt))

        bufs = (_zeros(q.shape), *(lax.empty(t.shape, t.dtype) for t in (*kv_types, gt_type, *q_types)))
        _, g8k, amk, g8v, amv, gt, g8q, amq = pl.run_state(run_merged)(bufs)
        return ((g8q, None, amq), (g8k, None, amk), (g8v, None, amv)), gt
    kv_types = _grad_types(batch, length, kv_heads, head_dim, transposed) * 2
    # run_state keeps the buffers internal: the results are pure values (custom_partitioning lowers no effects)
    dq_acc, *gkv = pl.run_state(run)((_zeros(q.shape), *(lax.empty(t.shape, t.dtype) for t in kv_types)))
    gq = _dq_epilogue_kernel(cfg, dq_acc, cos, sin, gscale, dtype, transposed)
    return _results([*gq, *gkv], n_out, transposed)


def fp8_attention_backward(q: Array, k: Array, v: Array, segment_ids: Array, runs: Array, out: Array, lse: Array,
                           do: Array, cos: Array, sin: Array, gscale: Array, cfg: _Config, *,
                           transposed: bool = False, merged: bool = False):
    """FP8 gradients of the attention's q, k and v projections from the backward kernels (``cfg.backward``).

    ``q [B, T, H, D]`` and ``k, v [B, T, Hkv, D]`` are the attention's operands (``q``, ``k`` after RoPE),
    ``segment_ids [B, T]`` int32 (zeros without documents) and ``runs [B, 4, T]`` as
    ``jla.ops.mosaic_gpu.attention.flash_attention`` builds them, ``out``/``lse`` the forward's results,
    ``do`` the output cotangent, ``cos, sin [B, T, D / 2]`` float32 the RoPE tables of the positions, and
    ``gscale [3]`` float32 the delayed e5m2 scales of the q, k and v projections' output gradients. ``T``
    must be a multiple of 128 (no padding here); ``cfg.backward="fused"`` needs ``D = 64``. Returns, for q, k
    and v, ``(e5m2 [B, T, Hp, D], e5m2 [Hp, D, B, T] with transposed=True else None, float32 row maxima
    [B, Hp, T])``.
    ``merged=True`` (``backward="fused"`` only) returns ``(those three with None, merged)`` instead, ``merged`` the
    e5m2 ``[H + 2 Hkv, D, B, T]`` transposed gradients of q, k and v in one operand (the merged weight-gradient
    GEMM's, ``[(H + 2 Hkv) D, B T]`` once reshaped).
    """
    reason = fp8_backward_supported(q, k, v)
    if reason is not None:
        raise ValueError(f"fp8_attention_backward: {reason}")
    if cfg.has_sinks:
        raise ValueError("fp8_attention_backward: attention sinks are not supported")
    batch, length, _, head_dim = q.shape
    if cos.shape != (batch, length, head_dim // 2) or sin.shape != cos.shape:
        raise ValueError(f"cos and sin must be [B, T, D / 2] = {(batch, length, head_dim // 2)}; got {cos.shape}")
    if cfg.backward not in BACKWARDS:
        raise ValueError(f"fp8_attention_backward: backward must be one of {BACKWARDS}; got {cfg.backward!r}")
    if cfg.backward == "fused" and head_dim != 64:
        raise ValueError(f"fp8_attention_backward: backward='fused' needs D = 64; got {head_dim}")
    if transposed and head_dim != 64:  # 257 KiB of shared memory per CTA with D = 128 (227 exist)
        raise ValueError(f"fp8_attention_backward: transposed=True needs D = 64; got {head_dim}")
    if merged and (transposed or cfg.backward != "fused"):
        raise ValueError("fp8_attention_backward: merged=True needs backward='fused' and no separate transposed copies")
    fn = _backward_fused_fp8 if cfg.backward == "fused" else _backward_ranged_fp8
    extra = dict(merged=True) if merged else {}
    return fn(cfg, q, k, v, segment_ids, runs, out, lse, do.astype(q.dtype), cos.astype(jnp.float32),
              sin.astype(jnp.float32), jnp.reshape(gscale, (3,)).astype(jnp.float32), transposed, **extra)
