# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gated_delta_rule/chunk_fwd.py, fla/ops/gated_delta_rule/wy_fast.py
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py
"""Mosaic GPU intra-chunk kernels of the delta rule (DeltaNet, Gated DeltaNet).

* :func:`kkt_fwd` (the default path): the gate-free inverse
  ``T0 = (I + beta K K^T)^-1`` (strictly lower part), 16x16 diagonal blocks by
  substitution, merged 16 -> 32 -> 64 on the tensor cores (FlashQLA's
  ``kkt_solve``). The gated inverse is ``T0 * exp(G_i - G_j)``, applied by the
  consumers (:func:`scalar_gate_fused.fused_fwd_wy`,
  :mod:`gated_delta_rule_bwd`); :func:`decay_inverse` forms it in XLA.

The previous path (``scalar_gate.run(..., wy=False)``) uses:

* :func:`intra_fwd`: ``A = beta_i (k_i . k_j) exp(G_i - G_j)`` on ``j < i``,
  ``T = (I + A)^-1`` and the WY factors ``w_neg = -T (beta exp(G) k)``,
  ``u = T (beta v)`` for one chunk per CTA. The key factor is stored negated
  (``v_new = u + w_neg S``), so no kernel ever negates a WGMMA result: on
  sm_90 in JAX 0.11.2, ``-acc * x`` on a fresh accumulator produced zeros
  inside the WY backward while ``acc * -x`` was correct (see
  ``benchmarks/README.md``). ``T`` is solved exactly in float32 by forward
  substitution, one column per thread (FLA solves 16x16 blocks in float32 and
  merges them in tf32). Given a saved ``T`` the kernel only recomputes ``w``
  and ``u`` (the backward pass).
* :func:`wy_bwd`: gradients through the WY factors and the triangular inverse:
  ``dk``, ``dv``, ``dbeta`` and the ``dG`` contribution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, gpu, memref, vector
from jaxlib.mlir.dialects import math as mlir_math

from jla.ops.mosaic_gpu._common import (
    CHUNK,
    COLS,
    ROWS,
    WGMMA,
    Segments,
    bcast_cols,
    bcast_rows,
    build_kernel,
    decay_matrix,
    iota,
    matmul,
    num_sms,
    same_run,
    smem,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK
HALF = 64  # column block of the WY backward (see wy_bwd)


@plgpu.inline_mgpu(
    arg_types=(plgpu.Layout.WGMMA, plgpu.RefType(), plgpu.RefType()),
    return_type=plgpu.ShapeDtypeStruct((C, C), F32, layout=plgpu.Layout.WGMMA),
)
def _unit_lower_inverse(ctx, neg_a_t, na_ref, x_ref):
    """``X = (I + A)^-1`` for strictly lower ``A``, given ``neg_a_t = -A^T`` (float32 ``[C, C]`` registers).

    ``neg_a_t`` is staged in the plain SMEM scratch ``na_ref``, so row ``j``
    holds column ``j`` of ``-A``. Thread ``c`` of a 64-thread half of the
    warpgroup (warps 0-1 or 2-3, alternating between CTAs) then solves column
    ``c`` right-looking: with ``r = e_c``, for ``j = c, c + 1, ...``,
    ``x_jc = r_j`` and ``r_i += -A_ij x_jc`` for ``i > j``. The updates of one
    step are independent FMAs (the dependent chain is one FMA per step), the
    reads are warp-broadcast 16-byte loads, and the warp of columns ``32..63``
    starts at step 32 (``x_jc = 0`` for ``j < c``). Thread ``c`` writes column
    ``c`` of ``x_ref``. Both SMEM buffers are only touched here, so Pallas
    layout inference never sees them.
    """
    del ctx
    neg_a_t.store_untiled(na_ref, optimized=False)
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    vec4 = ir.VectorType.get([4], f32)
    cst = mgpu_utils.c
    mgpu_utils.warpgroup_barrier()  # every thread must see all of neg_a_t
    tid = arith.remui(mgpu_utils.thread_idx(), cst(128, i32))
    # Half of the CTAs solve on warps 0-1 and half on warps 2-3, so that the CTAs
    # sharing an SM spread the solve over its four sub-partitions.
    blocks = arith.addi(arith.addi(gpu.block_id(gpu.Dimension.x), gpu.block_id(gpu.Dimension.y)),
                        gpu.block_id(gpu.Dimension.z))
    base = arith.muli(arith.index_cast(i32, arith.remui(blocks, cst(2, index))), cst(C, i32))
    lane = arith.subi(tid, base)  # the column this thread solves (valid for 0 <= lane < C)
    col = arith.index_cast(index, lane)
    zero, one = cst(0.0, f32), cst(1.0, f32)

    def solve(first: int) -> None:  # columns >= first: rows < first are zero
        r = {i: arith.select(arith.cmpi(arith.CmpIPredicate.eq, lane, cst(i, i32)), one, zero)
             for i in range(first, C)}
        for i in range(first):
            memref.store(zero, x_ref, [cst(i, index), col])
        for j in range(first, C):
            x_j = r.pop(j)
            memref.store(x_j, x_ref, [cst(j, index), col])
            for i0 in range((j + 1) // 4 * 4, C, 4):
                seg = vector.load(vec4, na_ref, [cst(j, index), cst(i0, index)])
                for ii in range(4):
                    if i0 + ii > j:
                        a = vector.extract(seg, dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([ii]))
                        r[i0 + ii] = mlir_math.fma(a, x_j, r[i0 + ii])

    half = C // 2

    def lanes(lo: int, hi: int):  # lo <= lane < hi (signed: lane is negative below the base)
        return arith.andi(arith.cmpi(arith.CmpIPredicate.sge, lane, cst(lo, i32)),
                          arith.cmpi(arith.CmpIPredicate.slt, lane, cst(hi, i32)))

    with mgpu_utils.when(lanes(0, half)):
        solve(0)
    with mgpu_utils.when(lanes(half, C)):
        solve(half)
    mgpu_utils.warpgroup_barrier()  # every column must be written before x is read
    return mgpu.FragmentedArray.load_untiled(x_ref, layout=mgpu.WGMMA_LAYOUT, optimized=False)


def intra_fwd(
    k: Array,
    v: Array,
    g2: Array,
    beta: Array,
    t: Array | None,
    *,
    seg: Segments | None = None,
    interpret=None,
) -> tuple[Array, Array, Array]:
    """WY factors of each chunk.

    Args:
      k: ``[B, Tp, Hk, K]``; v: ``[B, Tp, H, V]``; g2, beta: ``[B, H, Tp]`` float32.
      t: ``None`` to solve ``T`` here, or a saved ``[B, H, Tp, C]`` inverse.
      seg: packed-document masks: ``A`` only connects tokens of one run and
        tokens outside the run entering the chunk get no ``w`` correction.

    Returns:
      ``(t [B, H, Tp, C], w_neg [B, H, Tp, K], u [B, H, Tp, V])`` in the input dtype
      (``t`` is the given one when it was passed in).

    One chunk per CTA; the key and value tiles are rescaled and then
    overwritten by the results in place, so the kernel needs about 73 KB of
    SMEM and three CTAs share an SM (one CTA's solve overlaps the others'
    copies and matmuls).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    nc = padded // C
    dtype = k.dtype
    solve = t is None
    segs = seg is not None
    out_type = [
        jax.ShapeDtypeStruct((batch, heads, padded, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, heads, padded, value_dim), dtype),
    ]
    if solve:
        out_type.insert(0, jax.ShapeDtypeStruct((batch, heads, padded, C), dtype))
    vec = plgpu.SMEM((C,), F32)
    scratch = dict(
        k_smem=smem((C, key_dim), dtype),  # k, then -beta exp(G) k, then w_neg
        v_smem=smem((C, value_dim), dtype),  # v, then beta v, then u
        g_smem=vec,
        b_smem=vec,
        t_smem=smem((C, C), dtype),
        bar=plgpu.Barrier(num_arrivals=(6 if solve else 5) + segs * (3 if solve else 1)),
    )
    if solve:
        # Row and column views of a gate vector need separate SMEM copies (layout inference).
        scratch.update(gc_smem=vec, bc_smem=vec, na_smem=plgpu.SMEM((C, C), F32), x_smem=plgpu.SMEM((C, C), F32))
    if segs:
        scratch.update(cin_r=vec)
        if solve:
            scratch.update(run_r=vec, run_c=vec)

    def body(*refs, k_smem, v_smem, g_smem, b_smem, t_smem, bar, gc_smem=None, bc_smem=None, na_smem=None,
             x_smem=None, cin_r=None, run_r=None, run_c=None):
        refs = list(refs)
        k_ref, v_ref, g_ref, beta_ref = (refs.pop(0) for _ in range(4))
        t_in_ref = None if solve else refs.pop(0)
        run_ref = refs.pop(0) if segs and solve else None
        cin_ref = refs.pop(0) if segs else None
        t_out_ref = refs.pop(0) if solve else None
        w_ref, u_ref = refs.pop(0), refs.pop(0)
        b, n, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem, bar)
        plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh], v_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem, bar)
        plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], b_smem, bar)
        if solve:
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], gc_smem, bar)
            plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], bc_smem, bar)
        else:
            plgpu.copy_gmem_to_smem(t_in_ref.at[b, hh, rows], t_smem, bar)
        if segs:
            plgpu.copy_gmem_to_smem(cin_ref.at[b, rows], cin_r, bar)
            if solve:
                plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_r, bar)
                plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_c, bar)
        plgpu.barrier_wait(bar)

        g_row = plgpu.load(g_smem, layout=ROWS)
        b_row = plgpu.load(b_smem, layout=ROWS)
        if solve:
            sq = (C, C)
            g_col = plgpu.load(gc_smem, layout=COLS)
            b_col = plgpu.load(bc_smem, layout=COLS)
            s = matmul(k_smem, k_smem.T, sq)  # symmetric, so it also serves A^T
            # -A^T[j, i] = -beta_i (k_i . k_j) exp(G_i - G_j) for i > j (row j, column i).
            upper = iota(sq, 1) > iota(sq, 0)
            if segs:
                upper = upper & same_run((plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS)), sq)
            expo = jnp.where(upper, bcast_cols(g_col, sq) - bcast_rows(g_row, sq), -jnp.inf)
            neg_a_t = s * bcast_cols(-b_col, sq) * jnp.exp2(expo)
            t_lo = _unit_lower_inverse(neg_a_t, na_smem, x_smem).astype(dtype)
            t_smem[...] = t_lo
        else:
            t_lo = plgpu.load(t_smem, layout=WGMMA)
        # FLA numerics: the right-hand sides are rounded to the input dtype; the
        # key side carries the minus sign of w_neg. Each thread rescales its own
        # fragment in place, after the KKT matmul has finished reading k.
        key_scale = -(b_row * jnp.exp2(g_row))
        if segs:
            key_scale = key_scale * plgpu.load(cin_r, layout=ROWS)
        kk = plgpu.load(k_smem, layout=WGMMA).astype(F32)
        k_smem[...] = (kk * bcast_rows(key_scale, (C, key_dim))).astype(dtype)
        vv = plgpu.load(v_smem, layout=WGMMA).astype(F32)
        v_smem[...] = (vv * bcast_rows(b_row, (C, value_dim))).astype(dtype)
        plgpu.commit_smem()
        w = matmul(t_lo, k_smem, (C, key_dim))  # complete (waited) before k_smem is overwritten
        k_smem[...] = w.astype(dtype)
        u = matmul(t_lo, v_smem, (C, value_dim))
        v_smem[...] = u.astype(dtype)
        plgpu.commit_smem()
        if solve:
            plgpu.copy_smem_to_gmem(t_smem, t_out_ref.at[b, hh, rows], commit_group=False)
        plgpu.copy_smem_to_gmem(k_smem, w_ref.at[b, hh, rows], commit_group=False)
        plgpu.copy_smem_to_gmem(v_smem, u_ref.at[b, hh, rows], commit_group=False)
        plgpu.commit_smem_to_gmem_group()
        plgpu.wait_smem_to_gmem(0)

    args = [k, v, g2, beta] + ([] if solve else [t])
    if segs:
        args += ([seg.run] if solve else []) + [seg.cin]
    outs = build_kernel(
        body, name="gdr_intra_fwd" if solve else "gdr_wu_recompute", out_type=out_type, scratch=scratch,
        grid=(batch, nc, heads), grid_names=("b", "n", "h"), interpret=interpret,
    )(*args)
    if solve:
        return tuple(outs)
    return (t, *outs)


BLOCK = 16  # diagonal blocks of the blocked inverse in kkt_fwd
# Padding of the rows of kkt_fwd's float32 solve buffers. The WGMMA layout stores 8 rows of a [64, 64]
# tile per warp instruction; at the unpadded 256-byte row stride all 8 hit the same banks (8-way conflicts).
SOLVE_PAD = 8


@plgpu.inline_mgpu(
    arg_types=(plgpu.Layout.WGMMA, plgpu.RefType(), plgpu.RefType()),
    return_type=plgpu.ShapeDtypeStruct((C, C), F32, layout=plgpu.Layout.WGMMA),
)
def _diag_block_inverse(ctx, neg_a_t, na_ref, x_ref):
    """Inverses of the 16x16 diagonal blocks of ``I + A`` (strictly lower ``A``), zero elsewhere.

    ``neg_a_t = -A^T`` as float32 ``[C, C]`` registers. As in
    :func:`_unit_lower_inverse`, but each of 64 threads solves one column of
    its diagonal block (16 steps, at most 120 FMAs and 16 live values).
    """
    del ctx
    neg_a_t.store_untiled(na_ref, optimized=False)
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    vec4 = ir.VectorType.get([4], f32)
    cst = mgpu_utils.c
    zero, one = cst(0.0, f32), cst(1.0, f32)
    mgpu.FragmentedArray.splat(zero, (C, C), layout=mgpu.WGMMA_LAYOUT).store_untiled(x_ref, optimized=False)
    mgpu_utils.warpgroup_barrier()  # neg_a_t and the zeros are visible to every thread
    tid = arith.remui(mgpu_utils.thread_idx(), cst(128, i32))
    blocks = arith.addi(arith.addi(gpu.block_id(gpu.Dimension.x), gpu.block_id(gpu.Dimension.y)),
                        gpu.block_id(gpu.Dimension.z))
    base = arith.muli(arith.index_cast(i32, arith.remui(blocks, cst(2, index))), cst(C, i32))
    lane = arith.subi(tid, base)  # column of the solved matrix (valid for 0 <= lane < C)
    active = arith.andi(arith.cmpi(arith.CmpIPredicate.sge, lane, cst(0, i32)),
                        arith.cmpi(arith.CmpIPredicate.slt, lane, cst(C, i32)))
    with mgpu_utils.when(active):
        col = arith.index_cast(index, lane)
        local = arith.remui(lane, cst(BLOCK, i32))
        start = arith.index_cast(index, arith.subi(lane, local))  # first row of this block
        r = {i: arith.select(arith.cmpi(arith.CmpIPredicate.eq, local, cst(i, i32)), one, zero)
             for i in range(BLOCK)}
        for j in range(BLOCK):
            x_j = r.pop(j)
            row_j = arith.addi(start, cst(j, index))
            memref.store(x_j, x_ref, [row_j, col])
            for i0 in range((j + 1) // 4 * 4, BLOCK, 4):
                seg = vector.load(vec4, na_ref, [row_j, arith.addi(start, cst(i0, index))])
                for ii in range(4):
                    if i0 + ii > j:
                        a = vector.extract(seg, dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([ii]))
                        r[i0 + ii] = mlir_math.fma(a, x_j, r[i0 + ii])
    mgpu_utils.warpgroup_barrier()  # every block column is written before x is read
    return mgpu.FragmentedArray.load_untiled(x_ref, layout=mgpu.WGMMA_LAYOUT, optimized=False)


def _merge_product(a: Array, b: Array, b_smem, dtype) -> Array:
    """``a @ b`` for float32 ``[C, C]`` registers on the tensor cores, operands rounded to ``dtype``.

    ``b`` is staged in the 16-bit SMEM tile ``b_smem``. One pass: three passes over
    hi/lo splits made ``T0`` 2.5x more accurate (1e-3 against 2.4e-3 relative)
    but cost 40% more time, and end to end the outputs and gradients did not
    change (the bf16 roundings downstream dominate).
    """
    b_smem[...] = b.astype(dtype)
    plgpu.commit_smem()
    return matmul(a.astype(dtype), b_smem, (C, C))


def kkt_chunks_per_cta(batch: int, heads: int, chunks: int) -> int:
    """Default chunks per CTA of :func:`kkt_fwd`: the most (up to 8) that leave at least 1.5 CTAs per SM.

    Two CTAs fit on an SM, and walking several chunks per CTA (with the next
    chunk's keys prefetched) pays only while that keeps the SMs busy: on an
    H100 at ``(B, H) = (1, 8)`` and 64 chunks, 8 chunks per CTA (64 CTAs) took
    22.9 us and 2 per CTA (256 CTAs) 10.4 us, while at 512 or more CTAs with 8
    chunks each 8 stays the fastest (``benchmarks/gdn.md``).
    """
    for cpc in (8, 4, 2):
        if chunks % cpc == 0 and batch * heads * (chunks // cpc) * 2 >= 3 * num_sms():
            return cpc
    return 1


def kkt_fwd(k: Array, beta: Array, *, seg: Segments | None = None, chunks_per_cta: int | None = None,
            interpret=None) -> Array:
    """Gate-free inverse ``T0 = (I + A0)^-1``, ``A0_ij = beta_i (k_i . k_j)`` on ``j < i``; ``[B, H, Tp, C]``.

    The decayed inverse of the gated rule is the elementwise product
    ``T_ij = T0_ij exp(G_i - G_j)`` (``i >= j``), because ``exp(G)`` enters
    ``A`` as a diagonal similarity (FlashQLA's ``kkt_solve``). The consumers
    apply that factor, which is at most 1, so nothing here depends on the
    gates. With ``seg``, ``A0`` only connects tokens of one run.

    The inverse is blocked: the 16x16 diagonal blocks are solved by forward
    substitution in float32, then merged into 32x32 and 64x64 blocks with
    ``X <- X - X L X`` (``L`` the next level's off-diagonal block of ``A0``),
    each product on the tensor cores in bf16. Each CTA walks
    ``chunks_per_cta`` chunks with the next chunk's keys prefetched and the
    stores of ``T0`` (input dtype) left in flight. The float32 buffers of the
    diagonal solve have padded rows (``SOLVE_PAD``): with 256-byte rows the
    register tiles' stores and loads hit 8-way bank conflicts, which took 30%
    of the kernel's time on an H100 (the results are the same bits).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = beta.shape[1]
    group = heads // kv_heads
    nc = padded // C
    if chunks_per_cta is None:
        chunks_per_cta = kkt_chunks_per_cta(batch, heads, nc)
    cpc = next(n for n in (chunks_per_cta, 4, 2, 1) if nc % n == 0)
    dtype = k.dtype
    segs = seg is not None
    vec2 = plgpu.SMEM((2, C), F32)
    scratch = dict(
        k_smem=smem((2, C, key_dim), dtype),
        br_smem=vec2,
        bc_smem=vec2,
        t_smem=smem((2, C, C), dtype),
        b_smem=smem((C, C), dtype),
        na_smem=plgpu.SMEM((C, C + SOLVE_PAD), F32),  # padded rows: see SOLVE_PAD
        x_smem=plgpu.SMEM((C, C + SOLVE_PAD), F32),
        bar=plgpu.Barrier(num_arrivals=3 + 2 * segs, num_barriers=2),
    )
    if segs:
        scratch.update(run_r=vec2, run_c=vec2)

    def body(*refs, k_smem, br_smem, bc_smem, t_smem, b_smem, na_smem, x_smem, bar, run_r=None, run_c=None):
        refs = list(refs)
        k_ref, beta_ref = refs.pop(0), refs.pop(0)
        run_ref = refs.pop(0) if segs else None
        (t_ref,) = refs
        b, nb, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)

        def issue(n, slot):  # n: chunk index within this CTA's run
            rows = pl.ds((nb * cpc + n) * C, C)
            plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem.at[slot], bar.at[slot])
            plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], br_smem.at[slot], bar.at[slot])
            plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], bc_smem.at[slot], bar.at[slot])
            if segs:
                plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_r.at[slot], bar.at[slot])
                plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_c.at[slot], bar.at[slot])

        issue(0, 0)
        sq = (C, C)
        ri, ci = iota(sq, 0), iota(sq, 1)

        @pl.loop(0, cpc)
        def _chunk(n):
            slot = lax.rem(n, 2)
            plgpu.barrier_wait(bar.at[slot])

            @pl.when(n + 1 < cpc)
            def _prefetch():
                plgpu.commit_smem()
                issue(n + 1, 1 - slot)

            s = matmul(k_smem.at[slot], k_smem.at[slot].T, sq)  # symmetric, so it also serves A0^T
            # A0[i, j] = beta_i S_ij on j < i, and -A0^T restricted to the diagonal 16x16 blocks.
            lower = ri > ci
            diag_upper = (ci > ri) & (lax.div(ri, BLOCK) == lax.div(ci, BLOCK))
            if segs:
                same = same_run((plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS)),
                                sq)
                lower, diag_upper = lower & same, diag_upper & same
            a0 = jnp.where(lower, s * bcast_rows(plgpu.load(br_smem.at[slot], layout=ROWS), sq), 0.0)
            neg_a_t = jnp.where(diag_upper, s * bcast_cols(-plgpu.load(bc_smem.at[slot], layout=COLS), sq), 0.0)
            x = _diag_block_inverse(neg_a_t, na_smem.at[:, :C], x_smem.at[:, :C])
            for size in (2 * BLOCK, 4 * BLOCK):  # merge 16 -> 32 -> 64
                half = size // 2
                off = ((lax.div(ri, size) == lax.div(ci, size)) & (lax.rem(ri, size) >= half)
                       & (lax.rem(ci, size) < half))
                link = _merge_product(jnp.where(off, a0, 0.0), x, b_smem, dtype)  # L X
                x = x - _merge_product(x, link, b_smem, dtype)
            plgpu.wait_smem_to_gmem(1, wait_read_only=True)  # t_smem[slot] of chunk n - 2 was sent
            t_smem.at[slot][...] = x.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(t_smem.at[slot], t_ref.at[b, hh, pl.ds((nb * cpc + n) * C, C)])

        plgpu.wait_smem_to_gmem(0)

    args = [k, beta] + ([seg.run] if segs else [])
    return build_kernel(
        body, name="gdr_kkt_fwd", out_type=jax.ShapeDtypeStruct((batch, heads, padded, C), dtype),
        scratch=scratch, grid=(batch, nc // cpc, heads), grid_names=("b", "n", "h"), interpret=interpret,
    )(*args)


def decay_inverse(t0: Array, g2: Array) -> Array:
    """``T = T0 * exp2(G_i - G_j)`` on ``i >= j`` (XLA), the inverse :func:`intra_fwd` would solve.

    ``t0`` is ``[B, H, Tp, C]`` from :func:`kkt_fwd`, ``g2`` the log2 decay ``[B, H, Tp]``.
    """
    batch, heads, padded, _ = t0.shape
    g = g2.reshape(batch, heads, padded // C, C)
    expo = g[..., :, None] - g[..., None, :]
    keep = jnp.tril(jnp.ones((C, C), bool))
    decay = jnp.exp2(jnp.where(keep, expo, -jnp.inf))
    t = t0.reshape(batch, heads, padded // C, C, C).astype(F32) * decay
    return t.reshape(batch, heads, padded, C).astype(t0.dtype)


def wy_bwd(
    k: Array,
    v: Array,
    g2: Array,
    beta: Array,
    t: Array,
    dw: Array,
    du: Array,
    dk_part: Array,
    *,
    dk_dtype,
    seg: Segments | None = None,
    interpret=None,
) -> tuple[Array, Array, Array, Array]:
    """Backward through ``w_neg = -T (beta exp(G) k)``, ``u = T (beta v)`` and ``T = (I + A)^-1``.

    Args:
      k: ``[B, Tp, Hk, K]``; v: ``[B, Tp, H, V]``; g2, beta: ``[B, H, Tp]``;
      t: ``[B, H, Tp, C]``; dw: ``[B, H, Tp, K]`` gradient of ``w_neg``;
      du: ``[B, Tp, H, V]``; dk_part: ``[B, Tp, H, K]`` float32 key gradient of
      the other stages (added here); seg: the forward's packed-document masks.

    Returns:
      ``dk [B, Tp, H, K]`` (``dk_dtype``, per value head), ``dv [B, Tp, H, V]``
      (input dtype), float32 ``dbeta [B, H, Tp]`` and ``dG [B, H, Tp]``.
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    nc = padded // C
    dtype = k.dtype
    segs = seg is not None
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dk_dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
    ]
    dk_f32 = jnp.dtype(dk_dtype) == jnp.float32
    vec = plgpu.SMEM((C,), F32)
    scratch = dict(
        k_smem=smem((C, key_dim), dtype),
        v_smem=smem((C, value_dim), dtype),
        g_smem=vec,
        gc_smem=vec,
        b_smem=vec,
        bc_smem=vec,
        t_smem=smem((C, C), dtype),
        dw_smem=smem((C, key_dim), dtype),
        du_smem=smem((C, value_dim), dtype),
        m_smem=smem((C, C), dtype),
        da_smem=smem((C, C), dtype),
        col_smem=vec,
        dk_acc=plgpu.SMEM((C, key_dim), F32),
        dv_out=smem((C, value_dim), dtype),
        db_out=vec,
        dg_out=vec,
        bar=plgpu.Barrier(num_arrivals=10 + 4 * segs),
    )
    if segs:
        scratch.update(run_r=vec, run_c=vec, cin_r=vec, cin_c=vec)

    def body(*refs, k_smem, v_smem, g_smem, gc_smem, b_smem, bc_smem, t_smem, dw_smem, du_smem, m_smem, da_smem,
             col_smem, dk_acc, dv_out, db_out, dg_out, bar, run_r=None, run_c=None, cin_r=None, cin_c=None):
        refs = list(refs)
        k_ref, v_ref, g_ref, beta_ref, t_ref, dw_ref, du_ref, dkp_ref = (refs.pop(0) for _ in range(8))
        run_ref, cin_ref = (refs.pop(0), refs.pop(0)) if segs else (None, None)
        dk_ref, dv_ref, db_ref, dg_ref = refs
        b, n, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem, bar)
        plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh], v_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], gc_smem, bar)
        plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], b_smem, bar)
        plgpu.copy_gmem_to_smem(beta_ref.at[b, hh, rows], bc_smem, bar)
        plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_smem, bar)
        plgpu.copy_gmem_to_smem(dw_ref.at[b, hh, rows], dw_smem, bar)
        plgpu.copy_gmem_to_smem(du_ref.at[b, rows, hh], du_smem, bar)
        plgpu.copy_gmem_to_smem(dkp_ref.at[b, rows, hh], dk_acc, bar)
        if segs:
            for src, dst in ((run_ref, run_r), (run_ref, run_c), (cin_ref, cin_r), (cin_ref, cin_c)):
                plgpu.copy_gmem_to_smem(src.at[b, rows], dst, bar)
        plgpu.barrier_wait(bar)

        g_row = plgpu.load(g_smem, layout=ROWS)
        g_col = plgpu.load(gc_smem, layout=COLS)
        b_row = plgpu.load(b_smem, layout=ROWS)
        b_col = plgpu.load(bc_smem, layout=COLS)
        sq = (C, C)
        sh = (C, HALF)
        gam_row = jnp.exp2(g_row)
        # The key right-hand side kbg = beta Gamma k (times carry-in with segments).
        kbg_row = b_row * gam_row
        kbg_col = b_col * jnp.exp2(g_col)
        runs = None
        if segs:
            cin_row = plgpu.load(cin_r, layout=ROWS)
            kbg_row = kbg_row * cin_row
            kbg_col = kbg_col * plgpu.load(cin_c, layout=COLS)
            runs = (plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS))
        # Value/key tiles are processed in HALF-column blocks (as FLA does with
        # BK = BV = 64): with full [C, 128] float32 tiles the kernel needed 255+
        # registers, and under that pressure the inverse-gradient products came
        # out silently zero (see benchmarks/README.md).
        cols = lambda ref, i: ref.at[:, pl.ds(i * HALF, HALF)]  # noqa: E731
        # d(beta v) = T^T du: the value gradient and its beta part.
        db = None
        for i in range(value_dim // HALF):
            dvb = matmul(t_smem.T, cols(du_smem, i), sh)
            cols(dv_out, i)[...] = (dvb * bcast_rows(b_row, sh)).astype(dtype)
            part = (dvb * plgpu.load(cols(v_smem, i), layout=WGMMA).astype(F32)).sum(axis=1)
            db = part if db is None else db + part
        # w_neg = -T kbg, so d(kbg) = -T^T dw (dw is the gradient of w_neg). dk
        # accumulates in dk_acc (preloaded with dk_part).
        tdw_k = None
        for i in range(key_dim // HALF):
            tdw = matmul(t_smem.T, cols(dw_smem, i), sh)
            part = (tdw * plgpu.load(cols(k_smem, i), layout=WGMMA).astype(F32)).sum(axis=1)
            tdw_k = part if tdw_k is None else tdw_k + part
            dk_i = plgpu.load(cols(dk_acc, i), layout=WGMMA)
            cols(dk_acc, i)[...] = dk_i + tdw * bcast_rows(-kbg_row, sh)
        if segs:
            tdw_k = tdw_k * cin_row
        db = db - tdw_k * gam_row
        dg = tdw_k * -(gam_row * b_row)
        # -dT = dw kbg^T - du (beta v)^T, scaled per column j.
        ndt = (matmul(dw_smem, k_smem.T, sq) * bcast_cols(kbg_col, sq)
               - matmul(du_smem, v_smem.T, sq) * bcast_cols(b_col, sq))
        # Through the inverse: dA = T^T (-dT) T^T on the strictly lower part, then the gate.
        strict = iota(sq, 0) > iota(sq, 1)
        m1 = matmul(jnp.where(strict, ndt, 0.0).astype(dtype), t_smem.T, sq)
        m_smem[...] = m1.astype(dtype)
        plgpu.commit_smem()
        daf = matmul(t_smem.T, m_smem, sq) * decay_matrix(g_row, g_col, sq, strict=True, runs=runs)
        # A_ij = beta_i S_ij exp(G_i - G_j), S = k k^T: to beta, G and k (both sides).
        s = matmul(k_smem, k_smem.T, sq)
        db = db + (daf * s).sum(axis=1)
        ada = s * bcast_rows(b_row, sq) * daf
        col_smem[...] = ada.sum(axis=0)
        dg = dg + ada.sum(axis=1) - plgpu.load(col_smem, layout=ROWS)
        das = (daf * bcast_rows(b_row, sq)).astype(dtype)
        da_smem[...] = das
        plgpu.commit_smem()

        for i in range(key_dim // HALF):
            def sym_update(acc, i=i):  # dk += (dA_s + dA_s^T) k
                plgpu.wgmma(acc, das, cols(k_smem, i))
                plgpu.wgmma(acc, da_smem.T, cols(k_smem, i))

            dk = pl.run_state(sym_update)(plgpu.ACC.init(plgpu.load(cols(dk_acc, i), layout=WGMMA)))
            dk = lax.optimization_barrier(dk)
            if dk_f32:
                cols(dk_acc, i)[...] = dk
            else:  # dw_smem is free by now: stage the 16-bit dk in it
                cols(dw_smem, i)[...] = dk.astype(dk_dtype)
        db_out[...] = db
        dg_out[...] = dg
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(dk_acc if dk_f32 else dw_smem, dk_ref.at[b, rows, hh], commit_group=False)
        plgpu.copy_smem_to_gmem(dv_out, dv_ref.at[b, rows, hh], commit_group=False)
        plgpu.copy_smem_to_gmem(db_out, db_ref.at[b, hh, rows], commit_group=False)
        plgpu.copy_smem_to_gmem(dg_out, dg_ref.at[b, hh, rows], commit_group=False)
        plgpu.commit_smem_to_gmem_group()
        plgpu.wait_smem_to_gmem(0)

    args = [k, v, g2, beta, t, dw, du, dk_part] + ([seg.run, seg.cin] if segs else [])
    return tuple(build_kernel(
        body, name="gdr_wy_bwd", out_type=out_type, scratch=scratch, grid=(batch, nc, heads),
        grid_names=("b", "n", "h"), interpret=interpret,
    )(*args))
