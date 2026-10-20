# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py
"""Fused Mosaic GPU backward of the delta rule from the gate-free inverse: one CTA per ``(b, h)``.

FlashQLA's ``fused_bwd`` design: the reverse state pass and the per-chunk
gradients run in one kernel that walks the chunks of a sequence in reverse and
keeps the state gradient ``dS^T`` (``[V, K]`` float32) in registers, so the
per-chunk state gradients never reach HBM, and ``Ag`` and ``P^T do`` are formed
once per chunk instead of twice. Three warpgroups share every chunk:

* WG0 issues the chunk's loads and runs the forward side of
  :func:`gated_delta_rule_bwd.chunk_grads` (``U``, ``W``, ``Ag``, ``v_new``,
  ``dP``, ``dq``), plus ``qg = scale Gamma q`` for the state update;
* WG1 runs its backward side (``dv_new``, ``dW`` = the value gradient,
  ``dAg``, the inverse, ``dbeta``, ``dk`` and the gate gradient) from the
  published ``dS'``;
* WG2 carries ``dS^T``: it publishes ``dS'^T`` (input dtype) for WG1, then
  ``dS^T <- gamma dS^T + do^T qg - (Gamma dW)^T k``.

The math is that of the split path (``state_bwd_wy`` + ``chunk_grads``), but some
intermediates are rounded to bfloat16 at different points, so the gradients are
equal only up to that rounding (on an H100, ``dq`` is bitwise equal and the other
gradients differ by up to 0.43% max-relative). ``K == V`` only (``h``, ``v`` and
``do`` share their SMEM with the float32 staging of ``dS^T``), no sequence split.

Speed: on an H100 this kernel is slower than the split path at every shape
measured (``benchmarks/gdn.md``), so :mod:`scalar_gate` runs it only on request
(``run(..., fused_bwd=True)``). Its SMEM is single-buffered at the limit
(226 KiB at ``K = V = 128``), so per chunk the TMA loads, WG0's forward side
and the tail of WG1's backward side run back to back; FlashQLA's kernel
overlaps them with a fourth (producer) warpgroup and per-buffer release
barriers, which is the next step here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

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
    finish_together,
    iota,
    matmul,
    same_run,
    smem,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK
# Registers per thread of WG0, WG1 and WG2 (multiples of 8). The launch gives every thread 168
# (65536 / 384, rounded down to 8) and setmaxnreg moves registers only within the CTA, so the
# three must sum to at most 3 * 168. WG2 holds dS^T: 128 registers. ptxas spills about 2 KiB per
# thread with any split; (160, 168, 176) and (144, 168, 192) spill less but ran 6% and 8% slower
# at (B, T, H) = (8, 2048, 32) (benchmarks/gdn.md).
REGISTERS = (168, 176, 160)
LAUNCH_REGISTERS = 168


def fused_bwd(
    q: Array,
    k: Array,
    v: Array,
    do: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    h: Array,
    dht: Array | None,
    *,
    scale: float,
    want_dh0: bool,
    seg: Segments | None = None,
    interpret=None,
) -> tuple[Array, Array, Array, Array, Array, Array | None]:
    """``dq, dk, dv, dbeta, dG`` (and ``dh0``) of the delta rule in one kernel per ``(b, h)``.

    Args:
      q, k: ``[B, Tp, Hk, K]``; v, do: ``[B, Tp, H, V]`` with ``V == K``;
      t0: ``[B, H, Tp, C]`` gate-free inverses; g2, beta: ``[B, H, Tp]`` float32;
      h: ``[B, H, NC, V, K]`` transposed states entering each chunk
        (:func:`gated_delta_rule_bwd.state_fwd_wy`); dht: ``[B, H, V, K]``
        float32 transposed gradient of the final state, or ``None``.
      want_dh0: also return the transposed gradient of the initial state.

    Returns:
      ``dq, dk [B, Tp, H, K]`` per value head and ``dv [B, Tp, H, V]`` in the
      input dtype, float32 ``dbeta, dG [B, H, Tp]`` (``dG`` before the reverse
      chunk cumsum, as :func:`gated_delta_rule_bwd.chunk_grads`) and float32
      ``dh0 [B, H, V, K]`` (``None`` without ``want_dh0``).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    if sum(REGISTERS) > 3 * LAUNCH_REGISTERS or any(n % 8 for n in REGISTERS):
        raise ValueError(f"REGISTERS {REGISTERS} must be multiples of 8 summing to at most {3 * LAUNCH_REGISTERS}")
    if value_dim != key_dim:
        raise ValueError(f"the fused backward needs K == V; got K={key_dim}, V={value_dim}")
    group = heads // kv_heads
    nc = padded // C
    dtype = q.dtype
    segs = seg is not None
    has_dht = dht is not None
    vec = plgpu.SMEM((C,), F32)
    st = (value_dim, key_dim)
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
    ]
    if want_dh0:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, value_dim, key_dim), F32))
    scratch = dict(
        q_s=smem((C, key_dim), dtype),
        k_s=smem((C, key_dim), dtype),
        # S, v (then U, then dv) and do (then dq); before and after the chunk walk, the float32 dS^T.
        big=plgpu.RefUnion([smem(st, dtype), smem((C, value_dim), dtype), smem((C, value_dim), dtype)],
                           [plgpu.SMEM(st, F32)]),
        dh_s=smem(st, dtype),  # dS'^T, written by WG2
        qg_s=smem((C, key_dim), dtype),  # scale Gamma q, written by WG0
        t_s=smem((C, C), dtype),  # T0, then dA0 * beta_i
        w_s=smem((C, value_dim), dtype),  # W, then dk
        ag_s=smem((C, C), dtype),  # Ag, then the inverse's product M
        vn_s=smem((C, value_dim), dtype),
        dvn_s=smem((C, value_dim), dtype),  # dv_new, then -Gamma dW
        dp_s=smem((C, C), dtype),
        g_r=vec,
        g_c=vec,
        b_r=vec,
        b_c=vec,
        sum_s=vec,
        dg_out=vec,
        db_out=vec,
        in_a=plgpu.Barrier(num_arrivals=8 + 4 * segs),  # k, S, v, T0 and the vectors
        in_b=plgpu.Barrier(num_arrivals=2),  # q, do
        dh_ready=plgpu.Barrier(),
        qg_ready=plgpu.Barrier(),
        ag_ready=plgpu.Barrier(),
        vn_ready=plgpu.Barrier(),
        dp_ready=plgpu.Barrier(),
        dvn_ready=plgpu.Barrier(),
        dga_ready=plgpu.Barrier(),
        do_free=plgpu.Barrier(num_arrivals=2),  # WG1 and WG2 have read do
        free=plgpu.Barrier(num_arrivals=2),  # WG1 and WG2 are done with the chunk's buffers
        done=plgpu.Barrier(num_arrivals=3),
    )
    if has_dht:
        scratch.update(stage_bar=plgpu.Barrier(), stage_free=plgpu.Barrier())
    if want_dh0:
        scratch.update(wg0_done=plgpu.Barrier())
    if segs:
        # One buffer: each scratch buffer takes a 1 KiB-aligned slot, and the kernel is near the SMEM limit.
        scratch.update(seg_v=plgpu.SMEM((4, C), F32))  # run ids (rows, columns), cin, tail

    def body(*refs, q_s, k_s, big, dh_s, qg_s, t_s, w_s, ag_s, vn_s, dvn_s, dp_s, g_r, g_c, b_r, b_c, sum_s,
             dg_out, db_out, in_a, in_b, dh_ready, qg_ready, ag_ready, vn_ready, dp_ready, dvn_ready, dga_ready,
             do_free, free, done, stage_bar=None, stage_free=None, wg0_done=None, seg_v=None):
        (h_s, v_s, do_s), (stage,) = big
        refs = list(refs)
        q_ref, k_ref, v_ref, do_ref, t_ref, g_ref, b_ref, h_ref = (refs.pop(0) for _ in range(8))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dht_ref = refs.pop(0) if has_dht else None
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref = (refs.pop(0) for _ in range(5))
        dh0_ref = refs.pop(0) if want_dh0 else None
        run_r, run_c, cin_r, tail_r = (seg_v.at[i] for i in range(4)) if segs else (None,) * 4
        b, hh = lax.axis_index("b"), lax.axis_index("h")
        hk = lax.div(hh, group)
        wg = lax.axis_index("wg")
        sq, sk = (C, C), (C, key_dim)
        # The [C, V] float32 tiles are processed in 64-column blocks to bound register use.
        vblocks = [pl.ds(i * 64, 64) for i in range(value_dim // 64)]
        sh = (C, 64)

        def gates():
            g_row = plgpu.load(g_r, layout=ROWS)
            g_col = plgpu.load(g_c, layout=COLS)
            runs = None
            gam = jnp.exp2(g_row)  # Gamma: the queries and the W correction read the incoming state
            if segs:
                runs = (plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS))
                gam = gam * plgpu.load(cin_r, layout=ROWS)
            return g_row, g_col, runs, gam

        def decay_t(g_row, g_col, runs):
            """``D^T``: ``exp(G_i - G_j)`` at row ``j``, column ``i`` for ``i >= j``."""
            keep = iota(sq, 0) <= iota(sq, 1)
            if segs:
                keep = keep & same_run(runs, sq)
            return jnp.exp2(jnp.where(keep, bcast_cols(g_col, sq) - bcast_rows(g_row, sq), -jnp.inf))

        # As in chunk_grads, no warpgroup takes a column sum (a cross-warp reduction): the
        # warpgroups would share Pallas' one reduction scratch.

        def set_registers(n):  # REGISTERS: setmaxnreg from the launch share
            if n != LAUNCH_REGISTERS:
                plgpu.set_max_registers(n, action="increase" if n > LAUNCH_REGISTERS else "decrease")

        @pl.when(wg == 0)
        def _forward_side():
            set_registers(REGISTERS[0])
            if has_dht:
                plgpu.barrier_wait(stage_free)  # WG2 has read dS^T from the staging buffer

            @pl.loop(0, nc)
            def _chunk(j):
                n = nc - 1 - j
                rows = pl.ds(n * C, C)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's dq has left do_s
                loads = [(k_ref.at[b, rows, hk], k_s), (h_ref.at[b, hh, n], h_s), (v_ref.at[b, rows, hh], v_s),
                         (t_ref.at[b, hh, rows], t_s), (g_ref.at[b, hh, rows], g_r), (g_ref.at[b, hh, rows], g_c),
                         (b_ref.at[b, hh, rows], b_r), (b_ref.at[b, hh, rows], b_c)]
                if segs:
                    loads += [(run_ref.at[b, rows], run_r), (run_ref.at[b, rows], run_c),
                              (cin_ref.at[b, rows], cin_r), (tail_ref.at[b, rows], tail_r)]
                for src, dst in loads:
                    plgpu.copy_gmem_to_smem(src, dst, in_a)
                for src, dst in ((q_ref.at[b, rows, hk], q_s), (do_ref.at[b, rows, hh], do_s)):
                    plgpu.copy_gmem_to_smem(src, dst, in_b)
                plgpu.barrier_wait(in_a)
                g_row, g_col, runs, gam = gates()
                # U = Gamma (k S) and W = v - U, then Ag and v_new = Ag W.
                for blk in vblocks:
                    u = matmul(k_s, h_s.at[blk].T, sh) * bcast_rows(gam, sh)
                    w_s.at[:, blk][...] = (plgpu.load(v_s.at[:, blk], layout=WGMMA).astype(F32) - u).astype(dtype)
                    v_s.at[:, blk][...] = u.astype(dtype)
                decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
                ag_lo = (plgpu.load(t_s, layout=WGMMA).astype(F32) * decay
                         * bcast_cols(plgpu.load(b_c, layout=COLS), sq)).astype(dtype)
                ag_s[...] = ag_lo
                plgpu.commit_smem()
                plgpu.barrier_arrive(ag_ready)
                for blk in vblocks:
                    vn_s.at[:, blk][...] = matmul(ag_lo, w_s.at[:, blk], sh).astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(vn_ready)
                plgpu.barrier_wait(in_b)
                # qg = scale Gamma q: the queries' part of the state update (WG2).
                for blk in vblocks:
                    qg = plgpu.load(q_s.at[:, blk], layout=WGMMA).astype(F32) * bcast_rows(gam * scale, sh)
                    qg_s.at[:, blk][...] = qg.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(qg_ready)
                # dP = scale (do v_new^T) * D; E = dP * (q k^T) gives the gate gradient of D.
                dp = matmul(do_s, vn_s.T, sq) * decay * scale
                dp_lo = dp.astype(dtype)
                dp_s[...] = dp_lo
                plgpu.commit_smem()
                plgpu.barrier_arrive(dp_ready)
                dg = (dp * matmul(q_s, k_s.T, sq)).sum(axis=1)
                et = matmul(vn_s, do_s.T, sq) * decay_t(g_row, g_col, runs) * matmul(k_s, q_s.T, sq)
                dg = dg - et.sum(axis=1) * scale  # E^T = (scale (v_new do^T) * D^T) * (k q^T)
                # dq = scale Gamma (do S^T) + dP k
                dq = matmul(do_s, h_s, sk) * bcast_rows(gam * scale, sk)
                dg = dg + (plgpu.load(q_s, layout=WGMMA).astype(F32) * dq).sum(axis=1)
                dq = matmul(dp_lo, k_s, sk, init=dq)
                plgpu.barrier_wait(do_free)  # WG1 and WG2 have read do
                do_s[...] = dq.astype(dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(do_s, dq_ref.at[b, rows, hh])
                dg_out[...] = dg
                plgpu.commit_smem()
                plgpu.barrier_arrive(dga_ready)
                plgpu.barrier_wait(free)  # WG1 and WG2 are done with this chunk's buffers

            plgpu.wait_smem_to_gmem(0)
            if want_dh0:
                plgpu.barrier_arrive(wg0_done)

        @pl.when(wg == 1)
        def _backward_side():
            set_registers(REGISTERS[1])

            @pl.loop(0, nc)
            def _chunk(j):
                n = nc - 1 - j
                rows = pl.ds(n * C, C)
                plgpu.barrier_wait(in_a)
                plgpu.barrier_wait(in_b)
                plgpu.barrier_wait(dh_ready)
                g_row, g_col, runs, gam = gates()
                b_col = plgpu.load(b_c, layout=COLS)
                g_last = g_r[C - 1]
                k_decay = jnp.exp2(g_last - g_row)  # Gamma_r: the keys write the outgoing state
                s_decay = jnp.exp2(g_last)
                if segs:
                    k_decay = k_decay * plgpu.load(tail_r, layout=ROWS)
                    s_decay = s_decay * seg_v[2, C - 1]
                # dv_new = scale P^T do + Gamma_r (k dS')
                pt = (matmul(k_s, q_s.T, sq) * decay_t(g_row, g_col, runs) * scale).astype(dtype)
                for blk in vblocks:
                    dvl = matmul(pt, do_s.at[:, blk], sh)
                    kds = matmul(k_s, dh_s.at[blk].T, sh)
                    dvn_s.at[:, blk][...] = (kds * bcast_rows(k_decay, sh) + dvl).astype(dtype)
                plgpu.barrier_arrive(do_free)
                plgpu.commit_smem()
                plgpu.barrier_wait(ag_ready)  # U, W and Ag
                dag = matmul(dvn_s, w_s.T, sq)  # dAg = dv_new W^T (only Ag's support is used)
                dagt = matmul(w_s, dvn_s.T, sq)  # its transpose
                dg = None
                for blk in vblocks:  # dW = Ag^T dv_new: the value gradient (staged where U was); dvn_s: -Gamma dW
                    dw = matmul(ag_s.T, dvn_s.at[:, blk], sh)
                    part = -(dw * plgpu.load(v_s.at[:, blk], layout=WGMMA).astype(F32)).sum(axis=1)  # W = v - U
                    dg = part if dg is None else dg + part
                    dvn_s.at[:, blk][...] = (dw * bcast_rows(-gam, sh)).astype(dtype)
                    v_s.at[:, blk][...] = dw.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(dvn_ready)
                plgpu.copy_smem_to_gmem(v_s, dv_ref.at[b, rows, hh])
                # Ag = T0 * D * beta_j: to beta, the gates and T0. F = dAg * Ag by rows, F^T by rows.
                decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
                t0d = plgpu.load(t_s, layout=WGMMA).astype(F32) * decay
                dg = dg + (dag * t0d * bcast_cols(b_col, sq)).sum(axis=1)
                x = dagt * plgpu.load(t_s.T, layout=WGMMA).astype(F32) * decay_t(g_row, g_col, runs)
                db = x.sum(axis=1)  # dbeta_j through Ag
                dg = dg - (x * bcast_rows(plgpu.load(b_r, layout=ROWS), sq)).sum(axis=1)
                # Through the inverse: dA0 = -T0^T dT0 T0^T on the strictly lower part (T0 has a unit diagonal).
                strict = iota(sq, 0) > iota(sq, 1)
                ndt = jnp.where(strict, dag * decay * bcast_cols(-b_col, sq), 0.0)
                ag_s[...] = matmul(ndt.astype(dtype), t_s.T, sq).astype(dtype)  # Ag was last read by the dW products
                plgpu.commit_smem()
                if segs:
                    strict = strict & same_run(runs, sq)
                da = jnp.where(strict, matmul(t_s.T, ag_s, sq), 0.0)
                db = db + (da * matmul(k_s, k_s.T, sq)).sum(axis=1)
                das = (da * bcast_rows(plgpu.load(b_r, layout=ROWS), sq)).astype(dtype)
                t_s[...] = das
                plgpu.commit_smem()
                # dk, starting from its state part Gamma_r (v_new dS'^T), whose key products feed the gate gradient.
                plgpu.barrier_wait(vn_ready)
                dks = None
                for blk in vblocks:
                    vnr = (plgpu.load(vn_s.at[:, blk], layout=WGMMA).astype(F32)
                           * bcast_rows(k_decay, sh)).astype(dtype)
                    dks = matmul(vnr, dh_s.at[blk], sk, init=dks)
                rks = (plgpu.load(k_s, layout=WGMMA).astype(F32) * dks).sum(axis=1)
                plgpu.barrier_wait(dp_ready)

                def update(acc):
                    plgpu.wgmma(acc, dvn_s, h_s)  # -Gamma (dW S^T)
                    plgpu.wgmma(acc, dp_s.T, q_s)  # dP^T q
                    plgpu.wgmma(acc, das, k_s)
                    plgpu.wgmma(acc, t_s.T, k_s)

                dk = lax.optimization_barrier(pl.run_state(update)(plgpu.ACC.init(dks)))
                w_s[...] = dk.astype(dtype)  # W was last read by the dAg product
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(w_s, dk_ref.at[b, rows, hh])
                # The last token also gets sum(rks) + gamma <S, dS'> (state decay, in 64-row blocks of
                # the transposed states). The total is summed in the column layout (within warps).
                sd = None
                for r in range(value_dim // 64):
                    blk = pl.ds(r * 64, 64)
                    part = (plgpu.load(h_s.at[blk], layout=WGMMA).astype(F32)
                            * plgpu.load(dh_s.at[blk], layout=WGMMA).astype(F32)).sum(axis=1)
                    sd = part if sd is None else sd + part
                sum_s[...] = rks + sd * s_decay
                plgpu.commit_smem()
                total = plgpu.load(sum_s, layout=COLS).sum()
                last = plgpu.broadcasted_iota(jnp.int32, (C,), 0, layout=ROWS) == C - 1
                dg = dg - rks + jnp.where(last, total, 0.0)
                plgpu.barrier_wait(dga_ready)
                dg_out[...] = plgpu.load(dg_out, layout=ROWS) + dg
                db_out[...] = db
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(dg_out, dg_ref.at[b, hh, rows], commit_group=False)
                plgpu.copy_smem_to_gmem(db_out, db_ref.at[b, hh, rows], commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # dv, dk, dG and dbeta have left SMEM
                plgpu.barrier_arrive(free)

            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 2)
        def _state():
            set_registers(REGISTERS[2])
            if has_dht:
                plgpu.copy_gmem_to_smem(dht_ref.at[b, hh], stage, stage_bar)
                plgpu.barrier_wait(stage_bar)
                ds0 = plgpu.load(stage, layout=WGMMA)
                plgpu.barrier_arrive(stage_free)
            else:
                ds0 = plgpu.layout_cast(jnp.zeros(st, F32), WGMMA)

            def step(j, ds_t):
                dh_s[...] = ds_t.astype(dtype)  # dS' of chunk nc - 1 - j
                plgpu.commit_smem()
                plgpu.barrier_arrive(dh_ready)
                plgpu.barrier_wait(in_a)
                plgpu.barrier_wait(in_b)
                s_decay = jnp.exp2(g_r[C - 1])
                if segs:
                    s_decay = s_decay * seg_v[2, C - 1]
                plgpu.barrier_wait(qg_ready)

                def with_queries(acc):  # gamma dS^T + do^T (scale Gamma q)
                    plgpu.wgmma(acc, do_s.T, qg_s)

                ds_t = lax.optimization_barrier(pl.run_state(with_queries)(plgpu.ACC.init(ds_t * s_decay)))
                plgpu.barrier_arrive(do_free)
                plgpu.barrier_wait(dvn_ready)

                def with_values(acc):  # - (Gamma dW)^T k
                    plgpu.wgmma(acc, dvn_s.T, k_s)

                ds_t = lax.optimization_barrier(pl.run_state(with_values)(plgpu.ACC.init(ds_t)))
                plgpu.barrier_arrive(free)
                plgpu.barrier_wait(free)  # WG1 has read dS' of this chunk
                return ds_t

            ds_t = lax.fori_loop(0, nc, step, ds0)
            if want_dh0:
                plgpu.barrier_wait(wg0_done)  # the staging buffer aliases WG0's inputs
                stage[...] = ds_t
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(stage, dh0_ref.at[b, hh])
                plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, v, do, t0, g2, beta, h] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([dht] if has_dht else [])
    outs = list(build_kernel(
        body, name="gdr_fused_bwd", out_type=out_type, scratch=scratch, grid=(batch, heads),
        grid_names=("b", "h"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    dq, dk, dv, db, dg = (outs.pop(0) for _ in range(5))
    dh0 = outs.pop(0) if want_dh0 else None
    return dq, dk, dv, db, dg, dh0
