# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/common/chunk_delta_h.py, fla/ops/common/chunk_o.py
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py
"""Fused, warp-specialized forward of the scalar-gate family (state pass + output).

One CTA per ``(b, h, 64-column value block)`` walks the chunks in order with
three warpgroups, the structure of FlashQLA's Hopper forward:

* WG0 (state) keeps ``S^T`` (``[BV, K]`` float32) in registers and runs only
  the serial chain: ``v_new^T = u^T + S^T w_neg^T`` (delta rule) and
  ``S^T <- gamma S^T + (Gamma_r v_new)^T k``. Per chunk it publishes ``S^T``
  and ``v_new^T`` (input dtype) in double-buffered SMEM;
* WG1 (output) computes ``P = (q k^T) * exp(G_i - G_j)`` while WG0 works, then
  ``o = scale (Gamma (q S) + P v_new)`` from the published tiles, and stores it;
* WG2 (memory) streams the chunk inputs with TMA into ``STAGES`` buffers.

Neither the per-chunk states nor ``v_new`` touch HBM (the backward recomputes
them). :func:`fused_fwd` takes precomputed WY factors (simple GLA, and the
previous delta-rule path); :func:`fused_fwd_wy` builds them inside the chunk
loop from the gate-free inverse of :func:`gated_delta_rule_kernels.kkt_fwd`
(Gated DeltaNet, DeltaNet). With ``cp > 1`` every sequence is cut into ``cp``
pieces that run in parallel CTAs, each from its own initial state (see
:mod:`scalar_gate`). Every warpgroup waits for the others before it exits
(:func:`_common.finish_together`).
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
    matmul,
    smem,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK
SMEM_BUDGET = 220 * 1024


def _stages(per_stage: int, fixed: int) -> int:
    """Deepest input pipeline (2 to 4 stages) that fits the SMEM budget."""
    for stages in (4, 3, 2):
        if stages * per_stage + fixed <= SMEM_BUDGET:
            return stages
    raise ValueError(f"fused forward needs {2 * per_stage + fixed} bytes of SMEM")


def fused_fwd(
    q: Array,
    k: Array,
    w: Array | None,
    u: Array,
    g2: Array,
    h0: Array | None,
    *,
    scale: float,
    u_head_major: bool,
    out_final: bool,
    out_dtype,
    seg: Segments | None = None,
    cp: int = 1,
    bv: int = 64,
    u_scale: Array | None = None,
    interpret=None,
) -> tuple[Array, Array | None]:
    """``o [B, Tp, H, V]`` (and the transposed final state of each piece ``[B * cp, H, V, K]`` float32).

    Arguments as in :func:`scalar_gate_kernels.state_fwd` plus ``q [B, Tp, Hk, K]``. ``u_scale [B, H, Tp]``
    (float32; simple GLA, ``u`` in user layout) scales the values per token in the kernel: it runs on
    ``(u * u_scale).astype(u.dtype)``, rounded as XLA rounds that product (Mamba-2's ``dt x`` without its
    ``[B, T, H, V]`` copy in HBM).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = u.shape[-1]
    group = heads // kv_heads
    ncs = padded // C // cp  # chunks per piece
    nv = value_dim // bv
    dtype = k.dtype
    delta = w is not None
    has_h0 = h0 is not None
    segs = seg is not None
    scaled = u_scale is not None
    if scaled and (delta or u_head_major):
        raise ValueError("u_scale needs simple GLA values in user layout")
    isz = jnp.dtype(dtype).itemsize
    per_stage = (2 + delta) * C * key_dim * isz + C * bv * isz + (3 + 5 * segs + scaled) * C * 4
    fixed = 2 * bv * key_dim * isz + 2 * C * bv * isz + 2 * C * bv * jnp.dtype(out_dtype).itemsize
    fixed += (bv * key_dim * 4 if has_h0 or out_final else 0) + 4096
    stages = _stages(per_stage, fixed)
    n_copies = 5 + (2 if delta else 1) + 5 * segs + scaled

    out_type = [jax.ShapeDtypeStruct((batch, padded, heads, value_dim), out_dtype)]
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        q_s=smem((stages, C, key_dim), dtype),
        k_s=smem((stages, C, key_dim), dtype),
        u_s=smem((stages, C, bv), dtype),
        g_state=vecs,  # WG0: column view + last decay
        g_rows=vecs,  # WG1: row view
        g_cols=vecs,  # WG1: column view
        s_pub=smem((2, bv, key_dim), dtype),
        vn_pub=smem((2, C, bv), dtype),
        o_out=smem((2, C, bv), out_dtype),
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        s_ready=plgpu.Barrier(num_barriers=2),
        s_free=plgpu.Barrier(num_barriers=2),
        state_bar=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    if delta:
        scratch.update(w_s=smem((stages, C, key_dim), dtype))
    if has_h0 or out_final:
        scratch.update(state_smem=plgpu.SMEM((bv, key_dim), F32))
    if segs:
        # WG0: tail (column view) and carry-in (last entry); WG1: run ids (row and
        # column views) and carry-in (row view).
        scratch.update(tail_c=vecs, cin_s=vecs, run_r=vecs, run_c=vecs, cin_r=vecs)
    if scaled:
        scratch.update(us_s=vecs)  # WG0: the values' per-token scale (column view)

    def body(*refs, q_s, k_s, u_s, g_state, g_rows, g_cols, s_pub, vn_pub, o_out, in_bar, in_free, s_ready,
             s_free, state_bar, done, w_s=None, state_smem=None, tail_c=None, cin_s=None, run_r=None, run_c=None,
             cin_r=None, us_s=None):
        refs = list(refs)
        q_ref, k_ref = refs.pop(0), refs.pop(0)
        w_ref = refs.pop(0) if delta else None
        u_ref, g_ref = refs.pop(0), refs.pop(0)
        us_ref = refs.pop(0) if scaled else None
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        o_ref = refs.pop(0)
        ht_ref = refs.pop(0) if out_final else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(n, slot):  # n: chunk index within the piece
                rows = pl.ds((c0 + n) * C, C)
                bar = in_bar.at[slot]
                plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_s.at[slot], bar)
                if delta:
                    plgpu.copy_gmem_to_smem(w_ref.at[b, hh, rows], w_s.at[slot], bar)
                    plgpu.copy_gmem_to_smem(u_ref.at[b, hh, rows, pl.ds(v0, bv)], u_s.at[slot], bar)
                else:
                    plgpu.copy_gmem_to_smem(u_ref.at[b, rows, hh, pl.ds(v0, bv)], u_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                if scaled:
                    plgpu.copy_gmem_to_smem(us_ref.at[b, hh, rows], us_s.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_s), (run_ref, run_r), (run_ref, run_c),
                                     (cin_ref, cin_r)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(min(stages, ncs)):
                issue(n, n)

            @pl.loop(stages, ncs)
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

            for n in range(max(ncs - stages, 0), ncs):  # observe the remaining releases
                plgpu.barrier_wait(in_free.at[n % stages])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_h0:
                plgpu.copy_gmem_to_smem(h0_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
                plgpu.barrier_wait(state_bar)
                s0 = plgpu.load(state_smem, layout=WGMMA)
            else:
                s0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

            def step(n, s_t):
                slot = lax.rem(n, stages)
                half = lax.rem(n, 2)
                plgpu.barrier_wait(in_bar.at[slot])

                @pl.when(n >= 2)
                def _():  # WG1 has finished reading the tiles published two chunks ago
                    plgpu.barrier_wait(s_free.at[half])

                s_lo = s_t.astype(dtype)
                s_pub.at[half][...] = s_lo
                u_t = plgpu.load(u_s.at[slot].T, layout=WGMMA).astype(F32)
                if scaled:  # the values (u * scale).astype(dtype), rounded as XLA rounds them
                    vn_lo = (u_t * bcast_cols(plgpu.load(us_s.at[slot], layout=COLS), (bv, C))).astype(dtype)
                    vn_pub.at[half].T[...] = vn_lo
                    vn_t = vn_lo.astype(F32)
                else:
                    vn_t = matmul(s_lo, w_s.at[slot].T, (bv, C), init=u_t) if delta else u_t
                    vn_pub.at[half].T[...] = vn_t.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(s_ready.at[half])
                g_last = g_state[slot, C - 1]
                k_decay = jnp.exp2(g_last - plgpu.load(g_state.at[slot], layout=COLS))
                s_decay = jnp.exp2(g_last)
                if segs:
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_s[slot, C - 1]
                vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
                s_new = matmul(vd_t, k_s.at[slot], (bv, key_dim), init=s_t * s_decay)
                plgpu.barrier_arrive(in_free.at[slot])
                return s_new

            s_t = lax.fori_loop(0, ncs, step, s0)
            for n in range(max(ncs - 2, 0), ncs):  # observe the last releases of the published tiles
                plgpu.barrier_wait(s_free.at[n % 2])
            if out_final:
                state_smem[...] = s_t
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(v0, bv)])
                plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _output():
            plgpu.set_max_registers(232, action="increase")

            def step(n, carry):
                slot = lax.rem(n, stages)
                half = lax.rem(n, 2)
                plgpu.barrier_wait(in_bar.at[slot])
                g_row = plgpu.load(g_rows.at[slot], layout=ROWS)
                g_col = plgpu.load(g_cols.at[slot], layout=COLS)
                q_decay = jnp.exp2(g_row)
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                    q_decay = q_decay * plgpu.load(cin_r.at[slot], layout=ROWS)
                p = matmul(q_s.at[slot], k_s.at[slot].T, (C, C))
                p_lo = (p * decay_matrix(g_row, g_col, (C, C), strict=False, runs=runs)).astype(dtype)
                plgpu.barrier_wait(s_ready.at[half])
                o = matmul(q_s.at[slot], s_pub.at[half].T, (C, bv))
                o = matmul(p_lo, vn_pub.at[half], (C, bv), init=o * bcast_rows(q_decay, (C, bv)))
                plgpu.barrier_arrive(s_free.at[half])
                plgpu.barrier_arrive(in_free.at[slot])
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)  # o_out[half] of chunk n - 2 was sent
                o_out.at[half][...] = (o * scale).astype(out_dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_out.at[half], o_ref.at[b, pl.ds((c0 + n) * C, C), hh, pl.ds(v0, bv)])
                return carry

            lax.fori_loop(0, ncs, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k] + ([w] if delta else []) + [u, g2] + ([u_scale] if scaled else [])
    args += [seg.run, seg.cin, seg.tail] if segs else []
    args += [h0] if has_h0 else []
    outs = list(build_kernel(
        body, name="sg_fused_fwd_scaled" if scaled else "sg_fused_fwd", out_type=out_type, scratch=scratch,
        grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    return outs[0], (outs[1] if out_final else None)


def fused_fwd_wy(
    q: Array,
    k: Array,
    v: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    h0: Array | None,
    *,
    scale: float,
    out_final: bool,
    out_dtype,
    seg: Segments | None = None,
    cp: int = 1,
    bv: int = 64,
    interpret=None,
) -> tuple[Array, Array | None]:
    """Delta-rule forward that applies the WY factors inside the chunk loop (FlashQLA's ``fused_fwd``).

    Takes the gate-free inverse ``t0`` of :func:`gated_delta_rule_kernels.kkt_fwd`
    instead of the WY factors, so ``w`` and ``u`` never exist in HBM:

    * WG1 builds ``Ag = T0 * exp(G_i - G_j) * beta_j`` (``i >= j``, the decayed
      inverse times beta) in place of ``T0`` in the input buffer, then the output
      ``o = scale ((Gamma q) S + P v_new)``;
    * WG0 carries ``S^T`` and per chunk forms ``W^T = v^T - (S^T k^T) Gamma``,
      ``v_new^T = W^T Ag^T`` and ``S^T <- gamma S^T + (Gamma_r v_new)^T k``, all
      three matmuls with the left operand in registers;
    * WG2 streams ``q, k, v, T0`` and the gate vectors with TMA.

    ``v`` is in the user layout ``[B, Tp, H, V]``; other arguments as in
    :func:`fused_fwd`.
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    ncs = padded // C // cp
    nv = value_dim // bv
    dtype = k.dtype
    has_h0 = h0 is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 5 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c, cin_r)
    per_stage = 2 * C * key_dim * isz + C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = 2 * bv * key_dim * isz + 2 * C * bv * isz + 2 * C * bv * jnp.dtype(out_dtype).itemsize
    fixed += (bv * key_dim * 4 if has_h0 or out_final else 0) + 4096 + n_vec * 1024
    stages = _stages(per_stage, fixed)
    n_copies = 4 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch, padded, heads, value_dim), out_dtype)]
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        q_s=smem((stages, C, key_dim), dtype),
        k_s=smem((stages, C, key_dim), dtype),
        v_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        g_state=vecs,  # WG0: column view + last decay
        g_rows=vecs,  # WG1: row view
        g_cols=vecs,  # WG1: column view
        b_cols=vecs,  # WG1: beta per column of Ag
        s_pub=smem((2, bv, key_dim), dtype),
        vn_pub=smem((2, C, bv), dtype),
        o_out=smem((2, C, bv), out_dtype),
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        ag_ready=plgpu.Barrier(num_barriers=stages),
        s_ready=plgpu.Barrier(num_barriers=2),
        s_free=plgpu.Barrier(num_barriers=2),
        state_bar=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    if has_h0 or out_final:
        scratch.update(state_smem=plgpu.SMEM((bv, key_dim), F32))
    if segs:
        # WG0: tail and carry-in (column views; carry-in's last entry is the keep flag);
        # WG1: run ids (row and column views) and carry-in (row view).
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs, cin_r=vecs)

    def body(*refs, q_s, k_s, v_s, t_s, g_state, g_rows, g_cols, b_cols, s_pub, vn_pub, o_out, in_bar, in_free,
             ag_ready, s_ready, s_free, state_bar, done, state_smem=None, tail_c=None, cin_c=None, run_r=None,
             run_c=None, cin_r=None):
        refs = list(refs)
        q_ref, k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(6))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        o_ref = refs.pop(0)
        ht_ref = refs.pop(0) if out_final else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(n, slot):  # n: chunk index within the piece
                rows = pl.ds((c0 + n) * C, C)
                bar = in_bar.at[slot]
                plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0, bv)], v_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c),
                                     (cin_ref, cin_r)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(min(stages, ncs)):
                issue(n, n)

            @pl.loop(stages, ncs)
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

            for n in range(max(ncs - stages, 0), ncs):  # observe the remaining releases
                plgpu.barrier_wait(in_free.at[n % stages])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_h0:
                plgpu.copy_gmem_to_smem(h0_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
                plgpu.barrier_wait(state_bar)
                s0 = plgpu.load(state_smem, layout=WGMMA)
            else:
                s0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

            def step(n, s_t):
                slot = lax.rem(n, stages)
                half = lax.rem(n, 2)
                plgpu.barrier_wait(in_bar.at[slot])

                @pl.when(n >= 2)
                def _():  # WG1 has finished reading the tiles published two chunks ago
                    plgpu.barrier_wait(s_free.at[half])

                s_lo = s_t.astype(dtype)
                s_pub.at[half][...] = s_lo
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)
                key_decay = jnp.exp2(g_col)  # Gamma: the correction term reads the incoming state
                k_decay = jnp.exp2(g_last - g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    cin_col = plgpu.load(cin_c.at[slot], layout=COLS)
                    key_decay = key_decay * cin_col
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                # W^T = v^T - (S^T k^T) Gamma, then v_new^T = W^T Ag^T.
                u_t = matmul(s_lo, k_s.at[slot].T, (bv, C))
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32)
                w_t = (v_t - u_t * bcast_cols(key_decay, (bv, C))).astype(dtype)
                plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = matmul(w_t, t_s.at[slot].T, (bv, C))
                vn_pub.at[half].T[...] = vn_t.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(s_ready.at[half])
                vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
                s_new = matmul(vd_t, k_s.at[slot], (bv, key_dim), init=s_t * s_decay)
                plgpu.barrier_arrive(in_free.at[slot])
                return s_new

            s_t = lax.fori_loop(0, ncs, step, s0)
            for n in range(max(ncs - 2, 0), ncs):  # observe the last releases of the published tiles
                plgpu.barrier_wait(s_free.at[n % 2])
            if out_final:
                state_smem[...] = s_t
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(v0, bv)])
                plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _output():
            plgpu.set_max_registers(232, action="increase")

            def step(n, carry):
                slot = lax.rem(n, stages)
                half = lax.rem(n, 2)
                plgpu.barrier_wait(in_bar.at[slot])
                g_row = plgpu.load(g_rows.at[slot], layout=ROWS)
                g_col = plgpu.load(g_cols.at[slot], layout=COLS)
                q_decay = jnp.exp2(g_row)
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                    q_decay = q_decay * plgpu.load(cin_r.at[slot], layout=ROWS)
                decay = decay_matrix(g_row, g_col, (C, C), strict=False, runs=runs)
                # Ag = T0 * exp(G_i - G_j) * beta_j replaces T0 in the input buffer.
                t0 = plgpu.load(t_s.at[slot], layout=WGMMA).astype(F32)
                ag = t0 * decay * bcast_cols(plgpu.load(b_cols.at[slot], layout=COLS), (C, C))
                t_s.at[slot][...] = ag.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(ag_ready.at[slot])
                p = matmul(q_s.at[slot], k_s.at[slot].T, (C, C))
                p_lo = (p * decay).astype(dtype)
                plgpu.barrier_wait(s_ready.at[half])
                o = matmul(q_s.at[slot], s_pub.at[half].T, (C, bv))
                o = matmul(p_lo, vn_pub.at[half], (C, bv), init=o * bcast_rows(q_decay, (C, bv)))
                plgpu.barrier_arrive(s_free.at[half])
                plgpu.barrier_arrive(in_free.at[slot])
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)  # o_out[half] of chunk n - 2 was sent
                o_out.at[half][...] = (o * scale).astype(out_dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_out.at[half], o_ref.at[b, pl.ds((c0 + n) * C, C), hh, pl.ds(v0, bv)])
                return carry

            lax.fori_loop(0, ncs, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([h0] if has_h0 else [])
    outs = list(build_kernel(
        body, name="gdr_fused_fwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    return outs[0], (outs[1] if out_final else None)

