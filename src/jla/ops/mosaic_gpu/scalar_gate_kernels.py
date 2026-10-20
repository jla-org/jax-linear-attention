# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/common/chunk_delta_h.py, fla/ops/common/chunk_o.py
"""Mosaic GPU kernels shared by the scalar-gate family (simple GLA, DeltaNet, Gated DeltaNet).

Per chunk of ``C = 64`` tokens, with ``G`` the chunk-local cumulative
log-decay, ``Gamma = exp(G)``, ``gamma = exp(G_last)`` and
``Gamma_r = exp(G_last - G)``:

* :func:`state_fwd` (serial over chunks): ``v_new = u + w_neg S`` (``v_new = v``
  without the delta rule; ``w_neg = -w`` is the negated WY key factor) and
  ``S <- gamma S + (Gamma_r k)^T v_new``, storing every chunk's ``S`` (the
  backward recompute; the forward pass fuses this with the output
  ``o = scale (Gamma q S + P v_new)``, ``P = (q k^T) * exp(G_i - G_j)`` on
  ``j <= i``, in :mod:`scalar_gate_fused`);
* :func:`dv_local`: ``scale P^T do``;
* :func:`state_bwd` (serial, reverse): ``dv_new = dv_local + Gamma_r k dS'`` and
  ``dS = gamma dS' + scale (Gamma q)^T do + w_neg^T dv_new``;
* :func:`dqkwg`: ``dq``, ``dk``, ``dw_neg`` and the gate gradient before the
  reverse cumsum.

No kernel negates a WGMMA accumulator: see :mod:`gated_delta_rule_kernels`.

The serial kernels hold the transposed state ``S^T`` (``[BV, K]``, float32) in
registers, so both matmuls on the critical path take their left operand from
registers and need no shared-memory round trip. They double-buffer the next
chunk's inputs with TMA while the current chunk computes. Numerics follow FLA:
bf16/fp16 operands with float32 accumulation, per-chunk states stored in the
input dtype, gates in float32.
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
    iota,
    matmul,
    same_run,
    smem,
    stage as _stage,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK


def state_fwd(
    k: Array,
    w: Array | None,
    u: Array,
    g2: Array,
    h0: Array | None,
    *,
    u_head_major: bool,
    out_final: bool,
    store_states: bool = True,
    seg: Segments | None = None,
    cp: int = 1,
    bv: int = 64,
    interpret=None,
) -> tuple[Array | None, Array | None, Array | None]:
    """Forward state pass.

    Args:
      k: ``[B, Tp, Hk, K]`` keys (``H % Hk == 0``; head ``h`` reads key head ``h // (H // Hk)``).
      w: ``[B, H, Tp, K]`` negated WY keys ``w_neg``, or ``None`` without the delta rule.
      u: values written, ``[B, H, Tp, V]`` if ``u_head_major`` else ``[B, Tp, H, V]``.
      g2: ``[B, H, Tp]`` float32 chunk-local cumulative decay in log2 units.
      h0: ``[B * cp, H, V, K]`` float32 transposed initial states or ``None``.
      store_states: write the per-chunk states and ``v_new`` (else only the final states).
      seg: packed-document masks (the state handed on keeps only each chunk's last run).
      cp: split every sequence into ``cp`` equal pieces that run in parallel,
        each from its own initial state (``h0[b * cp + i]``).

    Returns:
      ``h [B, H, NC, V, K]`` (transposed state entering each chunk, input dtype),
      ``v_new [B, H, Tp, V]`` (only with the delta rule) and the transposed final
      state of each piece ``[B * cp, H, V, K]`` float32 (only with ``out_final``).
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
    store_vn = delta and store_states

    out_type = []
    if store_states:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, padded // C, value_dim, key_dim), dtype))
    if store_vn:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, padded, value_dim), dtype))
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32))
    n_in = 3 + delta + 2 * segs
    vec2 = plgpu.SMEM((2, C), F32)
    scratch = dict(
        k_smem=smem((2, C, key_dim), dtype),
        u_smem=smem((2, C, bv), dtype),
        g_smem=vec2,
        h_out=smem((2, bv, key_dim), dtype),
        state_smem=plgpu.SMEM((bv, key_dim), F32),
        in_bar=plgpu.Barrier(num_arrivals=n_in, num_barriers=2),
        state_bar=plgpu.Barrier(),
    )
    if delta:
        scratch.update(w_smem=smem((2, C, key_dim), dtype), vn_out=smem((2, C, bv), dtype))
    if segs:
        scratch.update(tail_c=vec2, cin_s=vec2)

    def body(*refs, k_smem, u_smem, g_smem, h_out, state_smem, in_bar, state_bar, w_smem=None, vn_out=None,
             tail_c=None, cin_s=None):
        refs = list(refs)
        k_ref = refs.pop(0)
        w_ref = refs.pop(0) if delta else None
        u_ref, g_ref = refs.pop(0), refs.pop(0)
        cin_ref, tail_ref = (refs.pop(0), refs.pop(0)) if segs else (None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        h_ref = refs.pop(0) if store_states else None
        vn_ref = refs.pop(0) if store_vn else None
        ht_ref = refs.pop(0) if out_final else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        v0 = iv * bv

        def u_tile(n):
            rows = pl.ds(n * C, C)
            if u_head_major:
                return u_ref.at[b, hh, rows, pl.ds(v0, bv)]
            return u_ref.at[b, rows, hh, pl.ds(v0, bv)]

        def issue(n, slot):  # n: global chunk index
            rows = pl.ds(n * C, C)
            bar = in_bar.at[slot]
            plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem.at[slot], bar)
            if delta:
                plgpu.copy_gmem_to_smem(w_ref.at[b, hh, rows], w_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(u_tile(n), u_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem.at[slot], bar)
            if segs:
                plgpu.copy_gmem_to_smem(cin_ref.at[b, rows], cin_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(tail_ref.at[b, rows], tail_c.at[slot], bar)

        issue(c0, 0)
        if has_h0:
            plgpu.copy_gmem_to_smem(h0_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
            plgpu.barrier_wait(state_bar)
            s0 = plgpu.load(state_smem, layout=WGMMA)
        else:
            s0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

        def step(n, s_t):
            slot = lax.rem(n, 2)
            plgpu.barrier_wait(in_bar.at[slot])

            @pl.when(n + 1 < ncs)
            def _prefetch():
                plgpu.commit_smem()
                issue(c0 + n + 1, 1 - slot)

            s_lo = s_t.astype(dtype)
            if store_states:
                # The output buffers of this slot were last read by the stores of chunk n - 2.
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                h_out.at[slot][...] = s_lo
            if delta:
                u_t = plgpu.load(u_smem.at[slot].T, layout=WGMMA).astype(F32)
                vn_t = matmul(s_lo, w_smem.at[slot].T, (bv, C), init=u_t)
                if store_vn:
                    vn_out.at[slot].T[...] = vn_t.astype(dtype)
            else:
                vn_t = plgpu.load(u_smem.at[slot].T, layout=WGMMA).astype(F32)
            if store_states:
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(h_out.at[slot], h_ref.at[b, hh, c0 + n, pl.ds(v0, bv)], commit_group=False)
                if store_vn:
                    plgpu.copy_smem_to_gmem(
                        vn_out.at[slot], vn_ref.at[b, hh, pl.ds((c0 + n) * C, C), pl.ds(v0, bv)], commit_group=False
                    )
                plgpu.commit_smem_to_gmem_group()

            g_last = g_smem[slot, C - 1]
            k_decay = jnp.exp2(g_last - plgpu.load(g_smem.at[slot], layout=COLS))
            s_decay = jnp.exp2(g_last)
            if segs:
                k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                s_decay = s_decay * cin_s[slot, C - 1]
            vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
            return matmul(vd_t, k_smem.at[slot], (bv, key_dim), init=s_t * s_decay)

        s_t = lax.fori_loop(0, ncs, step, s0)
        if out_final:
            state_smem[...] = s_t
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(v0, bv)])
        plgpu.wait_smem_to_gmem(0)

    args = [k] + ([w] if delta else []) + [u, g2] + ([seg.cin, seg.tail] if segs else []) + ([h0] if has_h0 else [])
    outs = list(build_kernel(
        body, name="sg_state_fwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret,
    )(*args))
    h = outs.pop(0) if store_states else None
    vn = outs.pop(0) if store_vn else None
    ht = outs.pop(0) if out_final else None
    return h, vn, ht


def dv_local(q: Array, k: Array, g2: Array, do: Array, *, scale: float, seg: Segments | None = None,
             interpret=None) -> Array:
    """``scale P^T do`` per chunk (intra-chunk part of the value gradient); ``[B, Tp, H, V]``."""
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    nc = padded // C
    dtype = q.dtype
    segs = seg is not None
    vec = plgpu.SMEM((C,), F32)
    scratch = dict(
        q_smem=smem((C, key_dim), dtype),
        k_smem=smem((C, key_dim), dtype),
        do_smem=smem((C, value_dim), dtype),
        g_smem=vec,
        gc_smem=vec,
        o_smem=smem((C, value_dim), dtype),
        bar=plgpu.Barrier(num_arrivals=5 + 2 * segs),
    )
    if segs:
        scratch.update(run_r=vec, run_c=vec)

    def body(*refs, q_smem, k_smem, do_smem, g_smem, gc_smem, o_smem, bar, run_r=None, run_c=None):
        refs = list(refs)
        q_ref, k_ref, do_ref, g_ref = (refs.pop(0) for _ in range(4))
        run_ref = refs.pop(0) if segs else None
        (o_ref,) = refs
        b, n, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_smem, bar)
        plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem, bar)
        plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh], do_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], gc_smem, bar)
        if segs:
            plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_r, bar)
            plgpu.copy_gmem_to_smem(run_ref.at[b, rows], run_c, bar)
        plgpu.barrier_wait(bar)
        # P^T[j, i] = (k_j . q_i) exp(G_i - G_j) for j <= i: the transposed causal decay.
        sq = (C, C)
        g_row = plgpu.load(g_smem, layout=ROWS)
        g_col = plgpu.load(gc_smem, layout=COLS)
        keep = iota(sq, 0) <= iota(sq, 1)
        if segs:
            keep = keep & same_run((plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS)), sq)
        expo = jnp.where(keep, bcast_cols(g_col, sq) - bcast_rows(g_row, sq), -jnp.inf)
        pt = matmul(k_smem, q_smem.T, sq)
        pt = (pt * jnp.exp2(expo) * scale).astype(dtype)
        o_smem[...] = matmul(pt, do_smem, (C, value_dim)).astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_ref.at[b, rows, hh])
        plgpu.wait_smem_to_gmem(0)

    args = [q, k, do, g2] + ([seg.run] if segs else [])
    return build_kernel(
        body, name="sg_dv_local", out_type=jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype),
        scratch=scratch, grid=(batch, nc, heads), grid_names=("b", "n", "h"), interpret=interpret,
    )(*args)


def state_bwd(
    q: Array,
    k: Array,
    w: Array | None,
    g2: Array,
    do: Array,
    dvl: Array,
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
    """Reverse state pass.

    Args:
      q, k: ``[B, Tp, Hk, K]``; w: ``[B, H, Tp, K]`` (``w_neg``) or ``None``; g2: ``[B, H, Tp]``;
      do: ``[B, Tp, H, V]``; dvl: ``[B, Tp, H, V]`` from :func:`dv_local`;
      dht: ``[B * cp, H, V, K]`` float32 transposed gradients of the state leaving
        each piece, or ``None``.
      store_grads: write ``dh`` and ``dv_new`` (else only the gradients of the
        states entering the pieces).
      seg: packed-document masks; cp: pieces per sequence (see :func:`state_fwd`).

    Returns:
      ``dh [B, H, NC, V, K]`` (transposed gradient of the state leaving each
      chunk), ``dv_new [B, Tp, H, V]`` (the final value gradient without the
      delta rule) and the transposed ``dh0 [B * cp, H, V, K]`` float32 (only with
      ``want_dh0``).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    ncs = padded // C // cp
    nv = value_dim // bv
    dtype = q.dtype
    delta = w is not None
    has_dht = dht is not None
    segs = seg is not None
    out_type = []
    if store_grads:
        out_type += [
            jax.ShapeDtypeStruct((batch, heads, padded // C, value_dim, key_dim), dtype),
            jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype),
        ]
    if want_dh0:
        out_type.append(jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32))
    n_in = 5 + delta + 2 * segs
    vec2 = plgpu.SMEM((2, C), F32)
    scratch = dict(
        q_smem=smem((2, C, key_dim), dtype),
        k_smem=smem((2, C, key_dim), dtype),
        do_smem=smem((2, C, bv), dtype),
        dvl_smem=smem((2, C, bv), dtype),
        g_smem=vec2,
        dh_out=smem((2, bv, key_dim), dtype),
        dv_out=smem((2, C, bv), dtype),
        state_smem=plgpu.SMEM((bv, key_dim), F32),
        in_bar=plgpu.Barrier(num_arrivals=n_in, num_barriers=2),
        state_bar=plgpu.Barrier(),
    )
    if delta:
        scratch.update(w_smem=smem((2, C, key_dim), dtype))
    if segs:
        scratch.update(cin_c=vec2, tail_c=vec2)

    def body(*refs, q_smem, k_smem, do_smem, dvl_smem, g_smem, dh_out, dv_out, state_smem, in_bar, state_bar,
             w_smem=None, cin_c=None, tail_c=None):
        refs = list(refs)
        q_ref, k_ref = refs.pop(0), refs.pop(0)
        w_ref = refs.pop(0) if delta else None
        g_ref, do_ref, dvl_ref = refs.pop(0), refs.pop(0), refs.pop(0)
        cin_ref, tail_ref = (refs.pop(0), refs.pop(0)) if segs else (None, None)
        dht_ref = refs.pop(0) if has_dht else None
        dh_ref, dv_ref = (refs.pop(0), refs.pop(0)) if store_grads else (None, None)
        dh0_ref = refs.pop(0) if want_dh0 else None
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b, c0 = lax.div(bp, cp), lax.rem(bp, cp) * ncs
        hk = lax.div(hh, group)
        v0 = iv * bv

        def issue(n, slot):  # n: global chunk index
            rows = pl.ds(n * C, C)
            bar = in_bar.at[slot]
            plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem.at[slot], bar)
            if delta:
                plgpu.copy_gmem_to_smem(w_ref.at[b, hh, rows], w_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh, pl.ds(v0, bv)], do_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(dvl_ref.at[b, rows, hh, pl.ds(v0, bv)], dvl_smem.at[slot], bar)
            if segs:
                plgpu.copy_gmem_to_smem(cin_ref.at[b, rows], cin_c.at[slot], bar)
                plgpu.copy_gmem_to_smem(tail_ref.at[b, rows], tail_c.at[slot], bar)

        issue(c0 + ncs - 1, 0)
        if has_dht:
            plgpu.copy_gmem_to_smem(dht_ref.at[bp, hh, pl.ds(v0, bv)], state_smem, state_bar)
            plgpu.barrier_wait(state_bar)
            ds0 = plgpu.load(state_smem, layout=WGMMA)
        else:
            ds0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

        def step(i, ds_t):
            n = c0 + ncs - 1 - i  # global chunk index, walking backwards
            slot = lax.rem(i, 2)
            plgpu.barrier_wait(in_bar.at[slot])

            @pl.when(i + 1 < ncs)
            def _prefetch():
                plgpu.commit_smem()
                issue(n - 1, 1 - slot)

            ds_lo = ds_t.astype(dtype)
            if store_grads:
                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                dh_out.at[slot][...] = ds_lo
            g_last = g_smem[slot, C - 1]
            g_col = plgpu.load(g_smem.at[slot], layout=COLS)  # tokens are the columns of the [BV, C] tiles
            k_decay = jnp.exp2(g_last - g_col)
            q_scale = jnp.exp2(g_col) * scale
            s_decay = jnp.exp2(g_last)
            if segs:
                k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                q_scale = q_scale * plgpu.load(cin_c.at[slot], layout=COLS)
                s_decay = s_decay * cin_c[slot, C - 1]
            # dv_new^T = dv_local^T + (dS^T k^T) * Gamma_r (per token = per column)
            kd = matmul(ds_lo, k_smem.at[slot].T, (bv, C))
            dvl_t = plgpu.load(dvl_smem.at[slot].T, layout=WGMMA).astype(F32)
            dvn_t = kd * bcast_cols(k_decay, (bv, C)) + dvl_t
            dvn_lo = dvn_t.astype(dtype)
            # scale (Gamma q)^T do enters dS^T as (do^T scaled per token) q.
            do_t = plgpu.load(do_smem.at[slot].T, layout=WGMMA).astype(F32)
            do_t = (do_t * bcast_cols(q_scale, (bv, C))).astype(dtype)
            if store_grads:
                dv_out.at[slot].T[...] = dvn_lo
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(dh_out.at[slot], dh_ref.at[b, hh, n, pl.ds(v0, bv)], commit_group=False)
                plgpu.copy_smem_to_gmem(dv_out.at[slot], dv_ref.at[b, pl.ds(n * C, C), hh, pl.ds(v0, bv)],
                                        commit_group=False)
                plgpu.commit_smem_to_gmem_group()

            def update(acc):
                plgpu.wgmma(acc, do_t, q_smem.at[slot])
                if delta:
                    plgpu.wgmma(acc, dvn_lo, w_smem.at[slot])

            return pl.run_state(update)(plgpu.ACC.init(ds_t * s_decay))

        ds_t = lax.fori_loop(0, ncs, step, ds0)
        if want_dh0:
            state_smem[...] = ds_t
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(state_smem, dh0_ref.at[bp, hh, pl.ds(v0, bv)])
        plgpu.wait_smem_to_gmem(0)

    args = [q, k] + ([w] if delta else []) + [g2, do, dvl] + ([seg.cin, seg.tail] if segs else [])
    args += [dht] if has_dht else []
    outs = list(build_kernel(
        body, name="sg_state_bwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret,
    )(*args))
    dh, dvn = (outs.pop(0), outs.pop(0)) if store_grads else (None, None)
    dh0 = outs.pop(0) if want_dh0 else None
    return dh, dvn, dh0


def dqkwg(
    q: Array,
    k: Array,
    vn: Array,
    g2: Array,
    h: Array,
    dh: Array,
    do: Array,
    dvn: Array | None,
    *,
    scale: float,
    vn_head_major: bool,
    dq_dtype,
    dk_dtype,
    seg: Segments | None = None,
    interpret=None,
) -> tuple[Array, Array, Array | None, Array]:
    """Per-chunk ``dq``, ``dk``, ``dw_neg = dv_new h^T`` (delta rule) and ``dG``.

    ``dq`` and ``dk`` are ``[B, Tp, H, K]`` per value head, in ``dq_dtype`` and
    ``dk_dtype`` (float32 when the caller still sums them over key-head groups
    or adds the WY part). ``dvn`` is ``[B, Tp, H, V]``; ``dw_neg`` is
    ``[B, H, Tp, K]``. ``dG`` is the gradient with respect to the chunk-local
    cumulative log-decay (natural units) from the output and state paths; the
    caller adds the WY part and applies the reverse chunk cumsum.
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    nc = padded // C
    dtype = q.dtype
    delta = dvn is not None
    segs = seg is not None
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dq_dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dk_dtype),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
    ]
    if delta:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, padded, key_dim), dtype))
    vec = plgpu.SMEM((C,), F32)
    scratch = dict(
        q_smem=smem((C, key_dim), dtype),
        k_smem=smem((C, key_dim), dtype),
        v_smem=smem((C, value_dim), dtype),
        do_smem=smem((C, value_dim), dtype),
        # The states are read before the float32 dq/dk tiles are written: share the memory.
        states_or_out=plgpu.RefUnion(
            [smem((value_dim, key_dim), dtype), smem((value_dim, key_dim), dtype)],
            [_stage((C, key_dim), dq_dtype), _stage((C, key_dim), dk_dtype)],
        ),
        g_smem=vec,
        gc_smem=vec,
        ds_smem=smem((C, C), dtype),
        dg_out=vec,
        bar=plgpu.Barrier(num_arrivals=8 + delta + 4 * segs),
    )
    if delta:
        scratch.update(dvn_smem=smem((C, value_dim), dtype), dw_out=smem((C, key_dim), dtype))
    if segs:
        scratch.update(run_r=vec, run_c=vec, cin_r=vec, tail_r=vec)

    def body(*refs, q_smem, k_smem, v_smem, do_smem, states_or_out, g_smem, gc_smem, ds_smem, dg_out, bar,
             dvn_smem=None, dw_out=None, run_r=None, run_c=None, cin_r=None, tail_r=None):
        (h_smem, dh_smem), (dq_out, dk_out) = states_or_out
        refs = list(refs)
        q_ref, k_ref, v_ref, g_ref, h_ref, dh_ref, do_ref = (refs.pop(0) for _ in range(7))
        dvn_ref = refs.pop(0) if delta else None
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dq_ref, dk_ref, dg_ref = refs.pop(0), refs.pop(0), refs.pop(0)
        dw_ref = refs.pop(0) if delta else None
        b, n, hh = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("h")
        hk = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_smem, bar)
        plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem, bar)
        v_tile = v_ref.at[b, hh, rows] if vn_head_major else v_ref.at[b, rows, hh]
        plgpu.copy_gmem_to_smem(v_tile, v_smem, bar)
        plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh], do_smem, bar)
        plgpu.copy_gmem_to_smem(h_ref.at[b, hh, n], h_smem, bar)
        plgpu.copy_gmem_to_smem(dh_ref.at[b, hh, n], dh_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], gc_smem, bar)
        if delta:
            plgpu.copy_gmem_to_smem(dvn_ref.at[b, rows, hh], dvn_smem, bar)
        if segs:
            for src, dst in ((run_ref, run_r), (run_ref, run_c), (cin_ref, cin_r), (tail_ref, tail_r)):
                plgpu.copy_gmem_to_smem(src.at[b, rows], dst, bar)
        plgpu.barrier_wait(bar)

        g_row = plgpu.load(g_smem, layout=ROWS)
        g_col = plgpu.load(gc_smem, layout=COLS)
        g_last = g_smem[C - 1]
        shape_k = (C, key_dim)
        q_scale = jnp.exp2(g_row) * scale  # the output reads the incoming state
        k_decay = jnp.exp2(g_last - g_row)  # the key writes the outgoing state
        s_decay = jnp.exp2(g_last)
        runs = None
        if segs:
            runs = (plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS))
            q_scale = q_scale * plgpu.load(cin_r, layout=ROWS)
            k_decay = k_decay * plgpu.load(tail_r, layout=ROWS)
            s_decay = s_decay * cin_r[C - 1]
        # ds = (do v_new^T) * exp(G_i - G_j) * scale on j <= i
        ds = matmul(do_smem, v_smem.T, (C, C))
        ds_lo = (ds * decay_matrix(g_row, g_col, (C, C), strict=False, runs=runs) * scale).astype(dtype)
        ds_smem[...] = ds_lo
        dq = matmul(do_smem, h_smem, shape_k)
        dk = matmul(v_smem, dh_smem, shape_k)
        if delta:
            dw_out[...] = matmul(dvn_smem, h_smem, shape_k).astype(dtype)
        plgpu.commit_smem()
        dq = dq * bcast_rows(q_scale, shape_k)
        dq = matmul(ds_lo, k_smem, shape_k, init=dq)
        dk = dk * bcast_rows(k_decay, shape_k)
        kk = plgpu.load(k_smem, layout=WGMMA).astype(F32)
        qq = plgpu.load(q_smem, layout=WGMMA).astype(F32)
        hs = plgpu.load(h_smem, layout=WGMMA).astype(F32)
        dhs = plgpu.load(dh_smem, layout=WGMMA).astype(F32)
        dg_last = s_decay * (hs * dhs).sum(axis=1).sum() + (dk * kk).sum(axis=1).sum()
        dk = matmul(ds_smem.T, q_smem, shape_k, init=dk)
        dg = (dq * qq).sum(axis=1) - (dk * kk).sum(axis=1)
        last = plgpu.broadcasted_iota(jnp.int32, (C,), 0, layout=ROWS) == C - 1
        dg_out[...] = dg + jnp.where(last, dg_last, 0.0)
        dq_out[...] = dq.astype(dq_dtype)
        dk_out[...] = dk.astype(dk_dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(dq_out, dq_ref.at[b, rows, hh], commit_group=False)
        plgpu.copy_smem_to_gmem(dk_out, dk_ref.at[b, rows, hh], commit_group=False)
        plgpu.copy_smem_to_gmem(dg_out, dg_ref.at[b, hh, rows], commit_group=False)
        if delta:
            plgpu.copy_smem_to_gmem(dw_out, dw_ref.at[b, hh, rows], commit_group=False)
        plgpu.commit_smem_to_gmem_group()
        plgpu.wait_smem_to_gmem(0)

    args = [q, k, vn, g2, h, dh, do] + ([dvn] if delta else []) + ([seg.run, seg.cin, seg.tail] if segs else [])
    outs = list(build_kernel(
        body, name="sg_dqkwg", out_type=out_type, scratch=scratch, grid=(batch, nc, heads),
        grid_names=("b", "n", "h"), interpret=interpret,
    )(*args))
    dw = outs[3] if delta else None
    return outs[0], outs[1], dw, outs[2]
