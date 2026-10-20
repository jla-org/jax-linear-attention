# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py,
# flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py
"""Mosaic GPU backward of the delta rule from the gate-free inverse (Gated DeltaNet, DeltaNet).

Per chunk of ``C = 64`` tokens, with ``T0 = (I + A0)^-1`` from
:func:`gated_delta_rule_kernels.kkt_fwd`, the decay ``D_ij = exp(G_i - G_j)``
(``i >= j``), ``Gamma = exp(G)``, ``gamma = exp(G_last)`` and
``Gamma_r = exp(G_last - G)``, the forward is

    Ag = T0 * D * beta_j,   W = v - Gamma (k S),   v_new = Ag W,
    o = scale (Gamma q S + P v_new),   S' = gamma S + (Gamma_r k)^T v_new,

with ``P = (q k^T) * D``. The WY factors ``w``, ``u`` and ``v_new`` never
reach HBM. Three kernels compute the gradients:

* :func:`state_fwd_wy` (FlashQLA's ``prepare_h``): the forward state pass,
  storing the state entering every chunk. Its augmented mode yields the piece
  maps of the sequence split;
* :func:`state_bwd_wy`: the reverse state pass, warp-specialized. It stores
  the gradient of the state leaving every chunk and the value gradient
  ``dv = dW = Ag^T dv_new``;
* :func:`chunk_grads`: everything else, one chunk per CTA: ``dq``, ``dk``,
  ``dbeta`` and the gate gradient, through ``W``, ``v_new``, ``P``, ``Ag`` and
  the inverse (``dA0 = -T0^T dT0 T0^T``), recomputing ``W``, ``v_new`` and
  ``dv_new`` from the stored states.

FlashQLA fuses the last two into one kernel with one CTA per ``(b, h)``
(:mod:`gated_delta_rule_fused_bwd`: opt-in, slower than these kernels here).
Here the reverse pass runs one CTA per ``(b, h, 64 value columns)`` and the
per-chunk gradients run one CTA per chunk, so both fill the GPU at small
batch-times-heads; the price is the per-chunk state gradients in HBM (see
``benchmarks/gdn.md``).
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
from jla.ops.mosaic_gpu.scalar_gate_fused import _stages

Array = jax.Array
F32 = jnp.float32
C = CHUNK
# chunk_grads: registers per thread of WG0 (forward side), WG1 (dv_new, dW and the inverse) and WG2 (dk),
# multiples of 8 summing to at most 3 * 168 (the launch share of 384 threads; setmaxnreg moves registers
# only within the CTA).
CHUNK_GRADS_REGISTERS = (160, 232, 112)
CHUNK_GRADS_LAUNCH_REGISTERS = 168


def _products(*specs) -> tuple[Array, ...]:
    """Independent float32 products ``a @ b`` issued as one WGMMA group, with one wait for all.

    ``specs`` are ``(a, b, shape)`` as for :func:`matmul` (``a`` an SMEM ref or 16-bit
    WGMMA registers). Each product runs the instructions :func:`matmul` runs, so the
    results are the same bits; the tensor cores start the next product while the
    previous one finishes instead of draining between them.
    """

    def scoped(*accs):
        for acc, (a, b, _) in zip(accs, specs):
            plgpu.wgmma(acc, a, b)
        return tuple(acc[...] for acc in accs)

    return lax.optimization_barrier(pl.run_scoped(scoped, *(plgpu.ACC(shape, F32) for _, _, shape in specs)))


def state_fwd_wy(
    k: Array,
    v: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    h0: Array | None,
    *,
    store_states: bool = True,
    out_final: bool = False,
    aug: bool = False,
    seg: Segments | None = None,
    cp: int = 1,
    bv: int = 64,
    interpret=None,
) -> tuple[Array | None, Array | None]:
    """Forward state pass from the gate-free inverse (FlashQLA's ``prepare_h``), warp-specialized.

    WG1 builds ``Ag`` in place of ``T0``; WG0 carries ``S^T`` (``[BV, K]``
    float32) and per chunk forms ``W^T = v^T - (S^T k^T) Gamma``,
    ``v_new^T = W^T Ag^T`` and ``S^T <- gamma S^T + (Gamma_r v_new)^T k``; WG2
    streams the inputs.

    Args:
      k: ``[B, Tp, Hk, K]``; v: ``[B, Tp, H, V]`` (user layout); t0: ``[B, H, Tp, C]``;
      g2, beta: ``[B, H, Tp]`` float32; h0: ``[B * cp, H, V, K]`` float32
        transposed initial states or ``None``.
      store_states: write the state entering every chunk.
      aug: the augmented pass of the sequence split: ``K / bv`` extra value
        blocks with zero values start from the identity, the value blocks from
        zero, so the final states are ``[M^T | E^T]`` (``[B * cp, H, K + V, K]``;
        needs ``h0 is None``, ``out_final`` and not ``store_states``).
      seg, cp: packed-document masks and pieces per sequence, as in
        :func:`scalar_gate_kernels.state_fwd`.

    Returns:
      ``h [B, H, NC, V, K]`` (transposed state entering each chunk, input dtype,
      with ``store_states``) and the transposed final state of each piece,
      float32 (with ``out_final``).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    ncs = padded // C // cp
    n_id = key_dim // bv if aug else 0  # identity blocks of the augmented pass
    nv = value_dim // bv + n_id
    if aug and (h0 is not None or store_states or not out_final):
        raise ValueError("the augmented pass starts from [I | 0] and returns only the final states")
    dtype = k.dtype
    has_h0 = h0 is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = C * key_dim * isz + C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = (2 * bv * key_dim * isz if store_states else 0) + (bv * key_dim * 4 if has_h0 or out_final else 0)
    fixed += 4096 + n_vec * 1024
    stages = _stages(per_stage, fixed)
    n_copies = 3 + n_vec

    out_type = []
    if store_states:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, padded // C, value_dim, key_dim), dtype))
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, nv * bv, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        k_s=smem((stages, C, key_dim), dtype),
        v_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        g_state=vecs,
        g_rows=vecs,
        g_cols=vecs,
        b_cols=vecs,
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        ag_ready=plgpu.Barrier(num_barriers=stages),
        state_bar=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    if store_states:
        scratch.update(h_out=smem((2, bv, key_dim), dtype))
    if has_h0 or out_final:
        scratch.update(state_smem=plgpu.SMEM((bv, key_dim), F32))
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, k_s, v_s, t_s, g_state, g_rows, g_cols, b_cols, in_bar, in_free, ag_ready, state_bar, done,
             h_out=None, state_smem=None, tail_c=None, cin_c=None, run_r=None, run_c=None):
        refs = list(refs)
        k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(5))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        h_ref = refs.pop(0) if store_states else None
        ht_ref = refs.pop(0) if out_final else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        # The identity blocks read (and ignore) the first value block.
        v0 = jnp.maximum(iv - n_id, 0) * bv
        wg = lax.axis_index("wg")

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(n, slot):
                rows = pl.ds((c0 + n) * C, C)
                bar = in_bar.at[slot]
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0, bv)], v_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(min(stages, ncs)):
                issue(n, n)

            @pl.loop(stages, ncs)
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

        @pl.when(wg == 1)
        def _ag():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, ncs)
            def _chunk(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                decay = decay_matrix(plgpu.load(g_rows.at[slot], layout=ROWS),
                                     plgpu.load(g_cols.at[slot], layout=COLS), (C, C), strict=False, runs=runs)
                t0 = plgpu.load(t_s.at[slot], layout=WGMMA).astype(F32)
                ag = t0 * decay * bcast_cols(plgpu.load(b_cols.at[slot], layout=COLS), (C, C))
                t_s.at[slot][...] = ag.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(ag_ready.at[slot])
                plgpu.barrier_arrive(in_free.at[slot])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            shape = (bv, key_dim)
            if has_h0:
                plgpu.copy_gmem_to_smem(h0_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
                plgpu.barrier_wait(state_bar)
                s0 = plgpu.load(state_smem, layout=WGMMA)
            elif aug:  # rows iv * bv ... of the identity, zero for the value blocks
                eye = iota(shape, 1) == iota(shape, 0) + iv * bv
                s0 = jnp.where(eye, 1.0, 0.0).astype(F32)
            else:
                s0 = plgpu.layout_cast(jnp.zeros(shape, F32), WGMMA)
            v_keep = jnp.where(iv >= n_id, 1.0, 0.0).astype(F32) if aug else None

            def step(n, s_t):
                slot = lax.rem(n, stages)
                half = lax.rem(n, 2)
                plgpu.barrier_wait(in_bar.at[slot])
                s_lo = s_t.astype(dtype)
                if store_states:
                    plgpu.wait_smem_to_gmem(1, wait_read_only=True)  # h_out[half] of chunk n - 2 was sent
                    h_out.at[half][...] = s_lo
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(h_out.at[half], h_ref.at[b, hh, c0 + n, pl.ds(v0, bv)])
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)
                key_decay = jnp.exp2(g_col)
                k_decay = jnp.exp2(g_last - g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    key_decay = key_decay * plgpu.load(cin_c.at[slot], layout=COLS)
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                u_t = matmul(s_lo, k_s.at[slot].T, (bv, C))
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32)
                if aug:
                    v_t = v_t * v_keep
                w_t = (v_t - u_t * bcast_cols(key_decay, (bv, C))).astype(dtype)
                plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = matmul(w_t, t_s.at[slot].T, (bv, C))
                vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
                s_new = matmul(vd_t, k_s.at[slot], (bv, key_dim), init=s_t * s_decay)
                plgpu.barrier_arrive(in_free.at[slot])
                return s_new

            s_t = lax.fori_loop(0, ncs, step, s0)
            if out_final:
                state_smem[...] = s_t
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(iv * bv, bv)])
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([h0] if has_h0 else [])
    outs = list(build_kernel(
        body, name="gdr_state_fwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    h = outs.pop(0) if store_states else None
    ht = outs.pop(0) if out_final else None
    return h, ht


def state_bwd_wy(
    q: Array,
    k: Array,
    do: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    dht: Array | None,
    *,
    scale: float,
    want_dh0: bool,
    store_grads: bool = True,
    seg: Segments | None = None,
    cp: int = 1,
    bv: int = 64,
    interpret=None,
) -> tuple[Array | None, Array | None, Array | None]:
    """Reverse state pass of the delta rule from the gate-free inverse, warp-specialized.

    * WG1 builds ``Ag`` in place of ``T0`` and the intra-chunk value gradient
      ``dvl = scale P^T do`` (64 value columns);
    * WG0 carries ``dS^T`` (``[BV, K]`` float32) through the chunks in reverse:
      ``dv_new^T = dvl^T + (dS^T k^T) Gamma_r``, ``dW^T = dv_new^T Ag`` (the
      value gradient) and
      ``dS^T <- gamma dS^T + scale (do^T Gamma) q - (dW^T Gamma) k``;
    * WG2 streams ``q, k, do, T0`` and the gate vectors.

    Args:
      q, k: ``[B, Tp, Hk, K]``; do: ``[B, Tp, H, V]``; t0: ``[B, H, Tp, C]``;
      g2, beta: ``[B, H, Tp]``; dht: ``[B * cp, H, V, K]`` float32 transposed
        gradients of the state leaving each piece, or ``None``.
      store_grads: write ``dh`` and ``dv`` (else only the gradients of the
        states entering the pieces).

    Returns:
      ``dh [B, H, NC, V, K]`` (transposed gradient of the state leaving each
      chunk), ``dv [B, Tp, H, V]`` (both in the input dtype, with
      ``store_grads``) and the transposed ``dh0 [B * cp, H, V, K]`` float32
      (with ``want_dh0``).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    ncs = padded // C // cp
    nv = value_dim // bv
    dtype = q.dtype
    has_dht = dht is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = 2 * C * key_dim * isz + 2 * C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = (2 * bv * key_dim * isz + 2 * C * bv * isz if store_grads else 0)
    fixed += (bv * key_dim * 4 if has_dht or want_dh0 else 0) + 4096 + n_vec * 1024
    stages = _stages(per_stage, fixed)
    n_copies = 4 + n_vec

    out_type = []
    if store_grads:
        out_type += [jax.ShapeDtypeStruct((batch, heads, padded // C, value_dim, key_dim), dtype),
                     jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype)]
    if want_dh0:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        q_s=smem((stages, C, key_dim), dtype),
        k_s=smem((stages, C, key_dim), dtype),
        do_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        dvl_s=smem((stages, C, bv), dtype),  # written by WG1
        g_state=vecs,
        g_rows=vecs,
        g_cols=vecs,
        b_cols=vecs,
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        dvl_ready=plgpu.Barrier(num_barriers=stages),
        state_bar=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    if store_grads:
        scratch.update(dh_out=smem((2, bv, key_dim), dtype), dv_out=smem((2, C, bv), dtype))
    if has_dht or want_dh0:
        scratch.update(state_smem=plgpu.SMEM((bv, key_dim), F32))
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, q_s, k_s, do_s, t_s, dvl_s, g_state, g_rows, g_cols, b_cols, in_bar, in_free, dvl_ready,
             state_bar, done, dh_out=None, dv_out=None, state_smem=None, tail_c=None, cin_c=None, run_r=None,
             run_c=None):
        refs = list(refs)
        q_ref, k_ref, do_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(6))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dht_ref = refs.pop(0) if has_dht else None
        dh_ref, dv_ref = (refs.pop(0), refs.pop(0)) if store_grads else (None, None)
        dh0_ref = refs.pop(0) if want_dh0 else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")
        sq = (C, C)

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(i, slot):  # i: step of the reverse walk
                rows = pl.ds((c0 + ncs - 1 - i) * C, C)
                bar = in_bar.at[slot]
                plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh, pl.ds(v0, bv)], do_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for i in range(min(stages, ncs)):
                issue(i, i)

            @pl.loop(stages, ncs)
            def _refill(i):
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(i, slot)

        @pl.when(wg == 1)
        def _intra():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, ncs)
            def _chunk(i):
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                g_row = plgpu.load(g_rows.at[slot], layout=ROWS)
                g_col = plgpu.load(g_cols.at[slot], layout=COLS)
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                t0v = plgpu.load(t_s.at[slot], layout=WGMMA).astype(F32)
                ag = t0v * decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
                t_s.at[slot][...] = (ag * bcast_cols(plgpu.load(b_cols.at[slot], layout=COLS), sq)).astype(dtype)
                # P^T[j, i] = (k_j . q_i) exp(G_i - G_j) on i >= j: rows are keys, columns queries.
                keep = iota(sq, 0) <= iota(sq, 1)
                if segs:
                    keep = keep & same_run(runs, sq)
                expo = jnp.where(keep, bcast_cols(g_col, sq) - bcast_rows(g_row, sq), -jnp.inf)
                pt = matmul(k_s.at[slot], q_s.at[slot].T, sq)
                pt = (pt * jnp.exp2(expo) * scale).astype(dtype)
                dvl_s.at[slot][...] = matmul(pt, do_s.at[slot], (C, bv)).astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(dvl_ready.at[slot])
                plgpu.barrier_arrive(in_free.at[slot])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_dht:
                plgpu.copy_gmem_to_smem(dht_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
                plgpu.barrier_wait(state_bar)
                ds0 = plgpu.load(state_smem, layout=WGMMA)
            else:
                ds0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)
            st = (bv, C)

            def step(i, ds_t):
                n = c0 + ncs - 1 - i  # global chunk index
                slot = lax.rem(i, stages)
                half = lax.rem(i, 2)
                plgpu.barrier_wait(in_bar.at[slot])
                ds_lo = ds_t.astype(dtype)
                if store_grads:
                    plgpu.wait_smem_to_gmem(1, wait_read_only=True)  # dh_out/dv_out[half] of step i - 2 were sent
                    dh_out.at[half][...] = ds_lo
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)  # tokens are the columns of the [BV, C] tiles
                k_decay = jnp.exp2(g_last - g_col)
                gam = jnp.exp2(g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    gam = gam * plgpu.load(cin_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                kd = matmul(ds_lo, k_s.at[slot].T, st)  # dS^T k^T
                plgpu.barrier_wait(dvl_ready.at[slot])
                dvl_t = plgpu.load(dvl_s.at[slot].T, layout=WGMMA).astype(F32)
                dvn_lo = (kd * bcast_cols(k_decay, st) + dvl_t).astype(dtype)
                dw_t = matmul(dvn_lo, t_s.at[slot], st)  # dW^T = dv_new^T Ag
                if store_grads:
                    dv_out.at[half].T[...] = dw_t.astype(dtype)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(dh_out.at[half], dh_ref.at[b, hh, n, pl.ds(v0, bv)], commit_group=False)
                    plgpu.copy_smem_to_gmem(dv_out.at[half], dv_ref.at[b, pl.ds(n * C, C), hh, pl.ds(v0, bv)],
                                            commit_group=False)
                    plgpu.commit_smem_to_gmem_group()
                do_t = plgpu.load(do_s.at[slot].T, layout=WGMMA).astype(F32)
                do_t = (do_t * bcast_cols(gam * scale, st)).astype(dtype)
                dwn = (dw_t * bcast_cols(-gam, st)).astype(dtype)

                def update(acc):
                    plgpu.wgmma(acc, do_t, q_s.at[slot])
                    plgpu.wgmma(acc, dwn, k_s.at[slot])

                ds_new = lax.optimization_barrier(pl.run_state(update)(plgpu.ACC.init(ds_t * s_decay)))
                plgpu.barrier_arrive(in_free.at[slot])
                return ds_new

            ds_t = lax.fori_loop(0, ncs, step, ds0)
            if want_dh0:
                state_smem[...] = ds_t
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(state_smem, dh0_ref.at[bp, hh, pl.ds(v0, bv)])
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, do, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([dht] if has_dht else [])
    outs = list(build_kernel(
        body, name="gdr_state_bwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    dh, dv = (outs.pop(0), outs.pop(0)) if store_grads else (None, None)
    dh0 = outs.pop(0) if want_dh0 else None
    return dh, dv, dh0


def chunk_grads(
    q: Array,
    k: Array,
    v: Array,
    do: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    h: Array,
    dh: Array,
    *,
    scale: float,
    seg: Segments | None = None,
    interpret=None,
) -> tuple[Array, Array, Array, Array]:
    """Per-chunk ``dq``, ``dk``, ``dbeta`` and gate gradient of the delta rule; one chunk per CTA.

    WG0 recomputes ``U = Gamma (k S)``, ``W = v - U``, ``Ag`` and
    ``v_new = Ag W``, then ``dP = scale (do v_new^T) * D`` and
    ``dq = scale Gamma (do S^T) + dP k``. WG1 recomputes
    ``dv_new = scale P^T do + Gamma_r (k dS')`` and ``dW = Ag^T dv_new`` and goes
    through ``Ag`` and the inverse (``dA0 = -T0^T dT0 T0^T`` on the strictly
    lower part) before its ``dbeta`` and gate-gradient sums, so ``dA0`` reaches
    WG2 early. WG2 accumulates
    ``dk = Gamma_r (v_new dS'^T) - Gamma (dW S^T) + dP^T q + (dA0 beta_i) k + (dA0 beta_i)^T k``
    from the products the other two publish in SMEM, and the state-decay term
    ``gamma <S, dS'>`` of the gate gradient. Independent products whose results
    are live together anyway are issued as one WGMMA group (:func:`_products`).
    The results are bit for bit those of the two-warpgroup kernel this replaced
    (WG1 then also carried ``dk``: 8 of the chunk's 13 M multiply-adds). The
    kernel is compute-bound: on an H100 it takes 76% of its time without its
    loads and stores (``benchmarks/gdn.md``).

    Args:
      q, k: ``[B, Tp, Hk, K]``; v, do: ``[B, Tp, H, V]``; t0: ``[B, H, Tp, C]``;
      g2, beta: ``[B, H, Tp]`` float32; h: ``[B, H, NC, V, K]`` transposed states
      entering each chunk (:func:`state_fwd_wy`); dh: ``[B, H, NC, V, K]``
      transposed gradients of the states leaving each chunk (:func:`state_bwd_wy`).

    Returns:
      ``dq, dk [B, Tp, H, K]`` per value head in the input dtype, and float32
      ``dbeta [B, H, Tp]`` and ``dG [B, H, Tp]`` (gradient of the chunk-local
      cumulative log-decay, natural units, before the reverse chunk cumsum).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    nc = padded // C
    dtype = q.dtype
    segs = seg is not None
    vec = plgpu.SMEM((C,), F32)
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
    ]
    scratch = dict(
        q_s=smem((C, key_dim), dtype),
        k_s=smem((C, key_dim), dtype),
        v_s=smem((C, value_dim), dtype),  # v, then U = Gamma (k S)
        do_dq=plgpu.RefUnion([smem((C, value_dim), dtype)], [smem((C, key_dim), dtype)]),  # do, then dq
        h_s=smem((value_dim, key_dim), dtype),
        dh_s=smem((value_dim, key_dim), dtype),
        t_s=smem((C, C), dtype),  # T0, then dA0 * beta_i
        w_dk=plgpu.RefUnion([smem((C, value_dim), dtype)], [smem((C, key_dim), dtype)]),  # W, then dk
        ag_s=smem((C, C), dtype),
        vn_s=smem((C, value_dim), dtype),
        dvn_s=smem((C, value_dim), dtype),  # dv_new, then -Gamma dW
        dp_s=smem((C, C), dtype),
        m_s=smem((C, C), dtype),
        g_r=vec,
        g_c=vec,
        b_r=vec,
        b_c=vec,
        sum_s=vec,
        rks_s=vec,
        tot_s=vec,
        dg_out=vec,
        db_out=vec,
        in_a=plgpu.Barrier(num_arrivals=8 + 4 * segs),  # k, S, v, T0 and the vectors
        in_b=plgpu.Barrier(num_arrivals=3),  # q, do, dS'
        ag_ready=plgpu.Barrier(),
        vn_ready=plgpu.Barrier(),
        dp_ready=plgpu.Barrier(),
        do_free=plgpu.Barrier(),
        dga_ready=plgpu.Barrier(),
        dw_ready=plgpu.Barrier(),
        das_ready=plgpu.Barrier(),
        rks_ready=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    if segs:
        scratch.update(run_r=vec, run_c=vec, cin_r=vec, tail_r=vec)

    def body(*refs, q_s, k_s, v_s, do_dq, h_s, dh_s, t_s, w_dk, ag_s, vn_s, dvn_s, dp_s, m_s, g_r, g_c, b_r, b_c,
             sum_s, rks_s, tot_s, dg_out, db_out, in_a, in_b, ag_ready, vn_ready, dp_ready, do_free, dga_ready,
             dw_ready, das_ready, rks_ready, done, run_r=None, run_c=None, cin_r=None, tail_r=None):
        (do_s,), (dq_out,) = do_dq
        (w_s,), (dk_out,) = w_dk
        refs = list(refs)
        q_ref, k_ref, v_ref, do_ref, t_ref, g_ref, b_ref, h_ref, dh_ref = (refs.pop(0) for _ in range(9))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dq_ref, dk_ref, db_ref, dg_ref = refs
        b, n, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        wg = lax.axis_index("wg")
        sq, sv, sk = (C, C), (C, value_dim), (C, key_dim)
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

        # Gate gradients of the decay matrices are row sums minus column sums. No column
        # sum (a cross-warp reduction) is taken: both warpgroups would use Pallas' one
        # reduction scratch at once. Columns are summed as rows of transposed products.

        def set_registers(n):  # CHUNK_GRADS_REGISTERS: setmaxnreg from the launch share
            if n != CHUNK_GRADS_LAUNCH_REGISTERS:
                plgpu.set_max_registers(n, action="increase" if n > CHUNK_GRADS_LAUNCH_REGISTERS else "decrease")

        @pl.when(wg == 0)
        def _forward_side():
            set_registers(CHUNK_GRADS_REGISTERS[0])
            loads = [(k_ref.at[b, rows, hk], k_s), (h_ref.at[b, hh, n], h_s), (v_ref.at[b, rows, hh], v_s),
                     (t_ref.at[b, hh, rows], t_s), (g_ref.at[b, hh, rows], g_r), (g_ref.at[b, hh, rows], g_c),
                     (b_ref.at[b, hh, rows], b_r), (b_ref.at[b, hh, rows], b_c)]
            if segs:
                loads += [(run_ref.at[b, rows], run_r), (run_ref.at[b, rows], run_c), (cin_ref.at[b, rows], cin_r),
                          (tail_ref.at[b, rows], tail_r)]
            for src, dst in loads:
                plgpu.copy_gmem_to_smem(src, dst, in_a)
            for src, dst in ((q_ref.at[b, rows, hk], q_s), (do_ref.at[b, rows, hh], do_s), (dh_ref.at[b, hh, n], dh_s)):
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
            # dP = scale (do v_new^T) * D; E = dP * (q k^T) gives the gate gradient of D.
            dp, qk = _products((do_s, vn_s.T, sq), (q_s, k_s.T, sq))
            dp = dp * decay * scale
            dp_lo = dp.astype(dtype)
            dp_s[...] = dp_lo
            plgpu.commit_smem()
            plgpu.barrier_arrive(dp_ready)
            dg = (dp * qk).sum(axis=1)
            vd, kq = _products((vn_s, do_s.T, sq), (k_s, q_s.T, sq))
            et = vd * decay_t(g_row, g_col, runs) * kq
            dg = dg - et.sum(axis=1) * scale  # E^T = (scale (v_new do^T) * D^T) * (k q^T)
            # dq = scale Gamma (do S^T) + dP k
            dq = matmul(do_s, h_s, sk) * bcast_rows(gam * scale, sk)
            dg = dg + (plgpu.load(q_s, layout=WGMMA).astype(F32) * dq).sum(axis=1)
            dq = matmul(dp_lo, k_s, sk, init=dq)
            plgpu.barrier_wait(do_free)  # WG1 has read do
            dq_out[...] = dq.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dq_out, dq_ref.at[b, rows, hh])
            dg_out[...] = dg
            plgpu.commit_smem()
            plgpu.barrier_arrive(dga_ready)
            plgpu.wait_smem_to_gmem(0)

        def key_decay():
            """``Gamma_r`` (the keys write the outgoing state) and ``gamma`` (the state's own decay)."""
            g_last = g_r[C - 1]
            k_decay = jnp.exp2(g_last - plgpu.load(g_r, layout=ROWS))
            s_decay = jnp.exp2(g_last)
            if segs:
                k_decay = k_decay * plgpu.load(tail_r, layout=ROWS)
                s_decay = s_decay * cin_r[C - 1]
            return k_decay, s_decay

        @pl.when(wg == 1)
        def _backward_side():
            set_registers(CHUNK_GRADS_REGISTERS[1])
            plgpu.barrier_wait(in_a)
            plgpu.barrier_wait(in_b)
            g_row, g_col, runs, gam = gates()
            b_col = plgpu.load(b_c, layout=COLS)
            k_decay, _ = key_decay()
            # dv_new = scale P^T do + Gamma_r (k dS')
            pt = (matmul(k_s, q_s.T, sq) * decay_t(g_row, g_col, runs) * scale).astype(dtype)
            for blk in vblocks:
                dvl, kds = _products((pt, do_s.at[:, blk], sh), (k_s, dh_s.at[blk].T, sh))
                dvn_s.at[:, blk][...] = (kds * bcast_rows(k_decay, sh) + dvl).astype(dtype)
            plgpu.barrier_arrive(do_free)
            plgpu.commit_smem()
            plgpu.barrier_wait(ag_ready)  # U, W and Ag
            # dAg = dv_new W^T (only Ag's support is used) and its transpose
            dag, dagt = _products((dvn_s, w_s.T, sq), (w_s, dvn_s.T, sq))
            dg = None
            # dW = Ag^T dv_new (the value gradient); dvn_s then holds -Gamma dW
            for blk in vblocks:
                dw = matmul(ag_s.T, dvn_s.at[:, blk], sh)
                part = -(dw * plgpu.load(v_s.at[:, blk], layout=WGMMA).astype(F32)).sum(axis=1)  # W = v - U
                dg = part if dg is None else dg + part
                dvn_s.at[:, blk][...] = (dw * bcast_rows(-gam, sh)).astype(dtype)
            plgpu.commit_smem()
            plgpu.barrier_arrive(dw_ready)  # WG2: dk += -Gamma (dW S^T)
            decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
            # Through the inverse first, so WG2 gets dA0 early: dA0 = -T0^T dT0 T0^T on the strictly lower
            # part (T0 has a unit diagonal); dA0 beta_i then replaces M in m_s.
            strict = iota(sq, 0) > iota(sq, 1)
            ndt = jnp.where(strict, dag * decay * bcast_cols(-b_col, sq), 0.0)
            m_s[...] = matmul(ndt.astype(dtype), t_s.T, sq).astype(dtype)
            plgpu.commit_smem()
            if segs:
                strict = strict & same_run(runs, sq)
            da, kk = _products((t_s.T, m_s, sq), (k_s, k_s.T, sq))
            da = jnp.where(strict, da, 0.0)
            db_inv = (da * kk).sum(axis=1)
            m_s[...] = (da * bcast_rows(plgpu.load(b_r, layout=ROWS), sq)).astype(dtype)
            plgpu.commit_smem()
            plgpu.barrier_arrive(das_ready)  # WG2: dk += (dA0 beta_i) k + (dA0 beta_i)^T k
            # Ag = T0 * D * beta_j: to beta, the gates and T0. F = dAg * Ag by rows, F^T by rows.
            t0d = plgpu.load(t_s, layout=WGMMA).astype(F32) * decay
            dg = dg + (dag * t0d * bcast_cols(b_col, sq)).sum(axis=1)
            x = dagt * plgpu.load(t_s.T, layout=WGMMA).astype(F32) * decay_t(g_row, g_col, runs)
            db = x.sum(axis=1) + db_inv  # dbeta_j through Ag, then through the inverse
            dg = dg - (x * bcast_rows(plgpu.load(b_r, layout=ROWS), sq)).sum(axis=1)
            plgpu.barrier_wait(dga_ready)
            plgpu.barrier_wait(rks_ready)
            dg = dg - plgpu.load(rks_s, layout=ROWS) + plgpu.load(tot_s, layout=ROWS)
            dg_out[...] = plgpu.load(dg_out, layout=ROWS) + dg
            db_out[...] = db
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dg_out, dg_ref.at[b, hh, rows], commit_group=False)
            plgpu.copy_smem_to_gmem(db_out, db_ref.at[b, hh, rows], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 2)
        def _key_side():
            set_registers(CHUNK_GRADS_REGISTERS[2])
            plgpu.barrier_wait(in_a)
            plgpu.barrier_wait(in_b)
            k_decay, s_decay = key_decay()
            # The last token gets sum(rks) + gamma <S, dS'> (state decay, in 64-row blocks of the
            # transposed states); <S, dS'> needs only the loads, so it runs while WG0 forms v_new.
            sd = None
            for r in range(value_dim // 64):
                blk = pl.ds(r * 64, 64)
                part = (plgpu.load(h_s.at[blk], layout=WGMMA).astype(F32)
                        * plgpu.load(dh_s.at[blk], layout=WGMMA).astype(F32)).sum(axis=1)
                sd = part if sd is None else sd + part
            # dk, starting from its state part Gamma_r (v_new dS'^T), whose key products feed the gate gradient.
            plgpu.barrier_wait(vn_ready)
            vnr = [(plgpu.load(vn_s.at[:, blk], layout=WGMMA).astype(F32) * bcast_rows(k_decay, sh)).astype(dtype)
                   for blk in vblocks]

            def state_part(acc):
                for blk, a in zip(vblocks, vnr):
                    plgpu.wgmma(acc, a, dh_s.at[blk])

            dks = lax.optimization_barrier(pl.run_state(state_part)(plgpu.ACC.init(jnp.zeros(sk, F32))))
            rks = (plgpu.load(k_s, layout=WGMMA).astype(F32) * dks).sum(axis=1)

            def update(acc):
                plgpu.barrier_wait(dw_ready)
                plgpu.wgmma(acc, dvn_s, h_s)  # -Gamma (dW S^T)
                plgpu.barrier_wait(dp_ready)
                plgpu.wgmma(acc, dp_s.T, q_s)  # dP^T q
                plgpu.barrier_wait(das_ready)
                plgpu.wgmma(acc, m_s, k_s)
                plgpu.wgmma(acc, m_s.T, k_s)

            dk = lax.optimization_barrier(pl.run_state(update)(plgpu.ACC.init(dks)))
            dk_out[...] = dk.astype(dtype)  # W was last read by WG1's dAg product (before dw_ready)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dk_out, dk_ref.at[b, rows, hh])
            # The total is summed in the column layout (within warps).
            sum_s[...] = rks + sd * s_decay
            plgpu.commit_smem()
            total = plgpu.load(sum_s, layout=COLS).sum()
            last = plgpu.broadcasted_iota(jnp.int32, (C,), 0, layout=ROWS) == C - 1
            rks_s[...] = rks
            tot_s[...] = jnp.where(last, total, 0.0)
            plgpu.commit_smem()
            plgpu.barrier_arrive(rks_ready)
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, v, do, t0, g2, beta, h, dh] + ([seg.run, seg.cin, seg.tail] if segs else [])
    dq, dk, db, dg = build_kernel(
        body, name="gdr_chunk_grads", out_type=out_type, scratch=scratch, grid=(batch, nc, heads),
        grid_names=("b", "n", "h"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args)
    return dq, dk, db, dg
