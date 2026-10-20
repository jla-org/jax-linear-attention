# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py,
# flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py, flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py
# Adapted from flash-linear-attention (MIT), fla/ops/common/chunk_delta_h.py
"""K-blocked Mosaic GPU kernels of the delta rule for large heads (``K = 256``, ``V`` up to 512).

The kernels of :mod:`gated_delta_rule_bwd` keep the transposed state ``S^T``
(``[BV, K]`` float32) as one WGMMA accumulator and stage ``k`` as one
``[C, K]`` tile. At ``K = 256`` that is 128 registers per thread for the state
alone plus its 16-bit copy, and the fused forward's buffers no longer fit in
shared memory. Here the state is carried as ``K / 64`` register tiles
``[BV, 64]`` (FLA's ``b_h1 .. b_h4`` in ``chunk_delta_h.py``), the keys and
queries are staged as ``K / 64`` tiles ``[C, 64]``, and the 16-bit copy of the
state lives only in the shared-memory tile that is also stored to HBM and read
by the tensor cores. Every contraction over ``K`` is one chain of WGMMAs into
one accumulator, and the ``K / 64`` state tiles are updated by one group of
WGMMAs, so each chunk waits on the tensor cores three times, as at ``K = 128``.

* :func:`fused_fwd_kb`: the forward (state pass and output in one kernel, the
  structure of :func:`scalar_gate_fused.fused_fwd_wy` with single-buffered
  published tiles);
* :func:`state_fwd_kb`: the forward state pass from the gate-free inverse
  ``T0`` (the algebra of :func:`gated_delta_rule_bwd.state_fwd_wy`). It stores
  the state entering every chunk (input dtype), optionally ``v_new`` and the
  final state, which the per-chunk gradients (and the forward through HBM) in
  :mod:`scalar_gate_kb` read;
* :func:`state_bwd_kb`: the reverse state pass (the algebra of
  :func:`gated_delta_rule_bwd.state_bwd_wy`), storing the gradient of the state
  leaving every chunk and the gradient of the initial state;
* :func:`chunk_grads_kb`: the per-chunk gradients ``dq, dk, dv, dbeta`` and the
  gate gradient of every chunk (the algebra of
  :func:`gated_delta_rule_bwd.chunk_grads`), one CTA per (chunk, value head,
  128 key columns), with the stored states and state gradients streamed in
  blocks of 64 value rows.

The three state passes run one CTA per ``(b, h, 64 value columns)`` with three
warpgroups (state, intra-chunk factors or output, TMA) and no sequence split.
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
    same_run,
    smem,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK
KB = 64  # key block: the state is carried as K / KB register tiles [BV, KB]
SMEM_BUDGET = 220 * 1024


def _stages(per_stage: int, fixed: int, budget: int = SMEM_BUDGET) -> int:
    """Deepest input pipeline (1 to 3 stages) that fits the SMEM budget."""
    for stages in (3, 2, 1):
        if stages * per_stage + fixed <= budget:
            return stages
    raise ValueError(f"the K-blocked state pass needs {per_stage + fixed} bytes of SMEM")


def _wgmma_sum(pairs, shape: tuple[int, int], init: Array | None = None) -> Array:
    """``init + sum(a @ b for a, b in pairs)`` in one accumulator, waited once (see ``_common.matmul``)."""

    def body(acc):
        for a, b in pairs:
            plgpu.wgmma(acc, a, b)

    if init is None:
        def scoped(acc):
            body(acc)
            return acc[...]
        out = pl.run_scoped(scoped, plgpu.ACC(shape, F32))
    else:
        out = pl.run_state(body)(plgpu.ACC.init(init))
    return lax.optimization_barrier(out)


def _wgmma_tiles(terms, inits) -> tuple[Array, ...]:
    """``init_j + sum(a @ b for a, b in terms[j])`` for several accumulators, waited once."""

    def body(accs):
        for acc, pairs in zip(accs, terms):
            for a, b in pairs:
                plgpu.wgmma(acc, a, b)

    return lax.optimization_barrier(pl.run_state(body)(tuple(plgpu.ACC.init(x) for x in inits)))


def _tiles(scratch: dict, name: str, n: int, buf) -> None:
    for i in range(n):
        scratch[f"{name}{i}"] = buf


def _untile(scratch: dict, name: str, n: int) -> list:
    return [scratch[f"{name}{i}"] for i in range(n)]


def _check(key_dim: int, value_dim: int, bv: int) -> int:
    if key_dim % KB or value_dim % bv:
        raise ValueError(f"K={key_dim} must be a multiple of {KB} and V={value_dim} of {bv}")
    return key_dim // KB


def state_fwd_kb(
    k: Array,
    v: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    h0: Array | None,
    *,
    store_vn: bool = False,
    out_final: bool = False,
    seg: Segments | None = None,
    bv: int = 64,
    interpret=None,
) -> tuple[Array, Array | None, Array | None]:
    """Forward state pass of the delta rule with the state in ``K / 64`` register tiles.

    WG1 builds ``Ag = T0 * D * beta_j`` in place of ``T0``; WG0 carries ``S^T``
    and per chunk stores its 16-bit copy, forms ``W^T = v^T - (S^T k^T) Gamma``,
    ``v_new^T = W^T Ag^T`` and ``S^T <- gamma S^T + (Gamma_r v_new)^T k``; WG2
    streams the inputs.

    Args:
      k: ``[B, Tp, Hk, K]``; v: ``[B, Tp, H, V]`` (user layout); t0: ``[B, H, Tp, C]``;
      g2, beta: ``[B, H, Tp]`` float32; h0: ``[B, H, V, K]`` float32 transposed
        initial state or ``None``.
      store_vn: also write ``v_new``.

    Returns:
      ``h [B, H, NC, V, K]`` (transposed state entering each chunk, input dtype),
      ``v_new [B, Tp, H, V]`` (input dtype, with ``store_vn``) and the transposed
      final state ``[B, H, V, K]`` float32 (with ``out_final``).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    nk = _check(key_dim, value_dim, bv)
    group = heads // kv_heads
    nc = padded // C
    nv = value_dim // bv
    dtype = k.dtype
    has_h0 = h0 is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = C * key_dim * isz + C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = bv * key_dim * isz + (C * bv * isz if store_vn else 0)
    fixed += (bv * key_dim * 4 if has_h0 or out_final else 0) + 4096 + n_vec * 1024
    stages = _stages(per_stage, fixed)
    n_copies = nk + 2 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch, heads, nc, value_dim, key_dim), dtype)]
    if store_vn:
        out_type.append(jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype))
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        v_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        g_state=vecs,
        g_rows=vecs,
        g_cols=vecs,
        b_cols=vecs,
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        ag_ready=plgpu.Barrier(num_barriers=stages),
        state_bar=plgpu.Barrier(num_arrivals=nk),
        done=plgpu.Barrier(num_arrivals=3),
    )
    # One buffer per key block (static indices into one buffer do not lower inside run_state).
    _tiles(scratch, "k_s", nk, smem((stages, C, KB), dtype))
    _tiles(scratch, "h_out", nk, smem((bv, KB), dtype))  # the state's 16-bit copy: stored and read by the MMAs
    if store_vn:
        scratch.update(vn_out=smem((C, bv), dtype))
    if has_h0 or out_final:
        _tiles(scratch, "state_smem", nk, plgpu.SMEM((bv, KB), F32))
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, **sc):
        k_s, h_out = _untile(sc, "k_s", nk), _untile(sc, "h_out", nk)
        state_smem = _untile(sc, "state_smem", nk) if has_h0 or out_final else None
        v_s, t_s, g_state, g_rows, g_cols, b_cols = (sc[n] for n in ("v_s", "t_s", "g_state", "g_rows", "g_cols",
                                                                     "b_cols"))
        in_bar, in_free, ag_ready, state_bar, done = (sc[n] for n in ("in_bar", "in_free", "ag_ready", "state_bar",
                                                                      "done"))
        vn_out = sc.get("vn_out")
        tail_c, cin_c, run_r, run_c = (sc.get(n) for n in ("tail_c", "cin_c", "run_r", "run_c"))
        refs = list(refs)
        k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(5))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        h_ref = refs.pop(0)
        vn_ref = refs.pop(0) if store_vn else None
        ht_ref = refs.pop(0) if out_final else None
        b, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(n, slot):
                rows = pl.ds(n * C, C)
                bar = in_bar.at[slot]
                for i in range(nk):
                    plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk, pl.ds(i * KB, KB)], k_s[i].at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0, bv)], v_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(min(stages, nc)):
                issue(n, n)

            @pl.loop(stages, nc)
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

        @pl.when(wg == 1)
        def _ag():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, nc)
            def _chunk(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                decay = decay_matrix(plgpu.load(g_rows.at[slot], layout=ROWS),
                                     plgpu.load(g_cols.at[slot], layout=COLS), (C, C), strict=False, runs=runs)
                t0v = plgpu.load(t_s.at[slot], layout=WGMMA).astype(F32)
                ag = t0v * decay * bcast_cols(plgpu.load(b_cols.at[slot], layout=COLS), (C, C))
                t_s.at[slot][...] = ag.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(ag_ready.at[slot])
                plgpu.barrier_arrive(in_free.at[slot])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_h0:
                for i in range(nk):
                    plgpu.copy_gmem_to_smem(h0_ref.at[b, hh, pl.ds(v0, bv), pl.ds(i * KB, KB)], state_smem[i],
                                            state_bar)
                plgpu.barrier_wait(state_bar)
                s0 = tuple(plgpu.load(state_smem[i], layout=WGMMA) for i in range(nk))
            else:
                s0 = tuple(plgpu.layout_cast(jnp.zeros((bv, KB), F32), WGMMA) for _ in range(nk))
            st = (bv, C)

            def step(n, s_tiles):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous chunk's stores have read h_out, vn_out
                for i in range(nk):
                    h_out[i][...] = s_tiles[i].astype(dtype)
                plgpu.commit_smem()
                for i in range(nk):
                    plgpu.copy_smem_to_gmem(h_out[i], h_ref.at[b, hh, n, pl.ds(v0, bv), pl.ds(i * KB, KB)],
                                            commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)
                key_decay = jnp.exp2(g_col)
                k_decay = jnp.exp2(g_last - g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    key_decay = key_decay * plgpu.load(cin_c.at[slot], layout=COLS)
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                u_t = _wgmma_sum([(h_out[i], k_s[i].at[slot].T) for i in range(nk)], st)  # S^T k^T
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32)
                w_t = (v_t - u_t * bcast_cols(key_decay, st)).astype(dtype)
                plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = _wgmma_sum([(w_t, t_s.at[slot].T)], st)
                if store_vn:
                    vn_out.T[...] = vn_t.astype(dtype)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(vn_out, vn_ref.at[b, pl.ds(n * C, C), hh, pl.ds(v0, bv)])
                vd_t = (vn_t * bcast_cols(k_decay, st)).astype(dtype)
                s_new = _wgmma_tiles([[(vd_t, k_s[i].at[slot])] for i in range(nk)],
                                     [s_tiles[i] * s_decay for i in range(nk)])
                plgpu.barrier_arrive(in_free.at[slot])
                return tuple(s_new)

            s_t = lax.fori_loop(0, nc, step, s0)
            if out_final:
                for i in range(nk):
                    state_smem[i][...] = s_t[i]
                plgpu.commit_smem()
                for i in range(nk):
                    plgpu.copy_smem_to_gmem(state_smem[i], ht_ref.at[b, hh, pl.ds(v0, bv), pl.ds(i * KB, KB)],
                                            commit_group=False)
                plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([h0] if has_h0 else [])
    outs = list(build_kernel(
        body, name="gdr_kb_state_fwd", out_type=out_type, scratch=scratch, grid=(batch, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    h = outs.pop(0)
    vn = outs.pop(0) if store_vn else None
    ht = outs.pop(0) if out_final else None
    return h, vn, ht


def fused_fwd_kb(
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
    bv: int = 64,
    interpret=None,
) -> tuple[Array, Array | None]:
    """Fused forward of the delta rule (state pass + output) with the state in ``K / 64`` register tiles.

    The structure of :func:`scalar_gate_fused.fused_fwd_wy` (FlashQLA's ``fused_fwd``):
    WG0 carries ``S^T`` and per chunk publishes its 16-bit copy and ``v_new`` in
    shared memory; WG1 builds ``Ag`` and computes ``o = scale ((Gamma q) S + P v_new)``;
    WG2 streams ``q, k, v, T0`` and the gate vectors. The published tiles are single
    buffered (shared memory is full at ``K = 256``), so WG0 publishes chunk ``n``
    once WG1 has read chunk ``n - 1``. Neither the states nor ``v_new`` touch HBM.

    Args as in :func:`state_fwd_kb` plus ``q [B, Tp, Hk, K]``. Returns
    ``o [B, Tp, H, V]`` and the transposed final state ``[B, H, V, K]`` float32
    (with ``out_final``).
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    nk = _check(key_dim, value_dim, bv)
    group = heads // kv_heads
    nc = padded // C
    nv = value_dim // bv
    dtype = k.dtype
    has_h0 = h0 is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 5 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c, cin_r)
    per_stage = 2 * C * key_dim * isz + C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = bv * key_dim * isz + C * bv * isz + C * bv * jnp.dtype(out_dtype).itemsize
    fixed += (bv * key_dim * 4 if has_h0 or out_final else 0)
    # Two stages at K = 256 need 210 KB plus alignment (H100: 227 KB per block), so this
    # kernel's budget leaves 8 KB for barriers and padding instead of the usual slack.
    stages = _stages(per_stage, fixed, budget=219 * 1024)
    n_copies = 2 * nk + 2 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch, padded, heads, value_dim), out_dtype)]
    if out_final:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        v_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        g_state=vecs,
        g_rows=vecs,
        g_cols=vecs,
        b_cols=vecs,
        vn_pub=smem((C, bv), dtype),
        o_out=smem((C, bv), out_dtype),
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        ag_ready=plgpu.Barrier(num_barriers=stages),
        s_ready=plgpu.Barrier(),
        s_free=plgpu.Barrier(),
        state_bar=plgpu.Barrier(num_arrivals=nk),
        done=plgpu.Barrier(num_arrivals=3),
    )
    _tiles(scratch, "q_s", nk, smem((stages, C, KB), dtype))
    _tiles(scratch, "k_s", nk, smem((stages, C, KB), dtype))
    _tiles(scratch, "s_pub", nk, smem((bv, KB), dtype))  # the state's 16-bit copy, read by both warpgroups
    if has_h0 or out_final:
        _tiles(scratch, "state_smem", nk, plgpu.SMEM((bv, KB), F32))
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs, cin_r=vecs)

    def body(*refs, **sc):
        q_s, k_s, s_pub = _untile(sc, "q_s", nk), _untile(sc, "k_s", nk), _untile(sc, "s_pub", nk)
        state_smem = _untile(sc, "state_smem", nk) if has_h0 or out_final else None
        v_s, t_s, g_state, g_rows, g_cols, b_cols, vn_pub, o_out = (
            sc[n] for n in ("v_s", "t_s", "g_state", "g_rows", "g_cols", "b_cols", "vn_pub", "o_out"))
        in_bar, in_free, ag_ready, s_ready, s_free, state_bar, done = (
            sc[n] for n in ("in_bar", "in_free", "ag_ready", "s_ready", "s_free", "state_bar", "done"))
        tail_c, cin_c, run_r, run_c, cin_r = (sc.get(n) for n in ("tail_c", "cin_c", "run_r", "run_c", "cin_r"))
        refs = list(refs)
        q_ref, k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(6))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        h0_ref = refs.pop(0) if has_h0 else None
        o_ref = refs.pop(0)
        ht_ref = refs.pop(0) if out_final else None
        b, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(n, slot):
                rows = pl.ds(n * C, C)
                bar = in_bar.at[slot]
                for i in range(nk):
                    cols = pl.ds(i * KB, KB)
                    plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk, cols], q_s[i].at[slot], bar)
                    plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk, cols], k_s[i].at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0, bv)], v_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c),
                                     (cin_ref, cin_r)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(min(stages, nc)):
                issue(n, n)

            @pl.loop(stages, nc)
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_h0:
                for i in range(nk):
                    plgpu.copy_gmem_to_smem(h0_ref.at[b, hh, pl.ds(v0, bv), pl.ds(i * KB, KB)], state_smem[i],
                                            state_bar)
                plgpu.barrier_wait(state_bar)
                s0 = tuple(plgpu.load(state_smem[i], layout=WGMMA) for i in range(nk))
            else:
                s0 = tuple(plgpu.layout_cast(jnp.zeros((bv, KB), F32), WGMMA) for _ in range(nk))
            st = (bv, C)

            def step(n, s_tiles):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])

                @pl.when(n >= 1)
                def _():  # WG1 has read the tiles published for the previous chunk
                    plgpu.barrier_wait(s_free)

                for i in range(nk):
                    s_pub[i][...] = s_tiles[i].astype(dtype)
                plgpu.commit_smem()
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)
                key_decay = jnp.exp2(g_col)
                k_decay = jnp.exp2(g_last - g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    key_decay = key_decay * plgpu.load(cin_c.at[slot], layout=COLS)
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                u_t = _wgmma_sum([(s_pub[i], k_s[i].at[slot].T) for i in range(nk)], st)  # S^T k^T
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32)
                w_t = (v_t - u_t * bcast_cols(key_decay, st)).astype(dtype)
                plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = _wgmma_sum([(w_t, t_s.at[slot].T)], st)
                vn_pub.T[...] = vn_t.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(s_ready)
                vd_t = (vn_t * bcast_cols(k_decay, st)).astype(dtype)
                s_new = _wgmma_tiles([[(vd_t, k_s[i].at[slot])] for i in range(nk)],
                                     [s_tiles[i] * s_decay for i in range(nk)])
                plgpu.barrier_arrive(in_free.at[slot])
                return tuple(s_new)

            s_t = lax.fori_loop(0, nc, step, s0)
            plgpu.barrier_wait(s_free)  # observe WG1's last release
            if out_final:
                for i in range(nk):
                    state_smem[i][...] = s_t[i]
                plgpu.commit_smem()
                for i in range(nk):
                    plgpu.copy_smem_to_gmem(state_smem[i], ht_ref.at[b, hh, pl.ds(v0, bv), pl.ds(i * KB, KB)],
                                            commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _output():
            plgpu.set_max_registers(232, action="increase")
            sq = (C, C)

            def step(n, carry):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                g_row = plgpu.load(g_rows.at[slot], layout=ROWS)
                g_col = plgpu.load(g_cols.at[slot], layout=COLS)
                q_decay = jnp.exp2(g_row)
                runs = None
                if segs:
                    runs = (plgpu.load(run_r.at[slot], layout=ROWS), plgpu.load(run_c.at[slot], layout=COLS))
                    q_decay = q_decay * plgpu.load(cin_r.at[slot], layout=ROWS)
                decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
                # Ag = T0 * exp(G_i - G_j) * beta_j replaces T0 in the input buffer.
                t0v = plgpu.load(t_s.at[slot], layout=WGMMA).astype(F32)
                ag = t0v * decay * bcast_cols(plgpu.load(b_cols.at[slot], layout=COLS), sq)
                t_s.at[slot][...] = ag.astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(ag_ready.at[slot])
                p = _wgmma_sum([(q_s[i].at[slot], k_s[i].at[slot].T) for i in range(nk)], sq)
                p_lo = (p * decay).astype(dtype)
                plgpu.barrier_wait(s_ready)
                o = _wgmma_sum([(q_s[i].at[slot], s_pub[i].T) for i in range(nk)], (C, bv))
                o = _wgmma_sum([(p_lo, vn_pub)], (C, bv), init=o * bcast_rows(q_decay, (C, bv)))
                plgpu.barrier_arrive(s_free)
                plgpu.barrier_arrive(in_free.at[slot])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # o_out of chunk n - 1 was sent
                o_out[...] = (o * scale).astype(out_dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_out, o_ref.at[b, pl.ds(n * C, C), hh, pl.ds(v0, bv)])
                return carry

            lax.fori_loop(0, nc, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([h0] if has_h0 else [])
    outs = list(build_kernel(
        body, name="gdr_kb_fused_fwd", out_type=out_type, scratch=scratch, grid=(batch, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    return outs[0], (outs[1] if out_final else None)


def state_bwd_kb(
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
    seg: Segments | None = None,
    bv: int = 64,
    interpret=None,
) -> tuple[Array, Array | None]:
    """Reverse state pass of the delta rule with the state gradient in ``K / 64`` register tiles.

    * WG1 builds ``Ag`` in place of ``T0`` and ``dvl = scale P^T do``;
    * WG0 carries ``dS^T`` through the chunks in reverse, storing its 16-bit
      copy: ``dv_new^T = dvl^T + (dS^T k^T) Gamma_r``, ``dW^T = dv_new^T Ag``,
      ``dS^T <- gamma dS^T + scale (do^T Gamma) q - (dW^T Gamma) k``;
    * WG2 streams ``q, k, do, T0`` and the gate vectors.

    Args:
      q, k: ``[B, Tp, Hk, K]``; do: ``[B, Tp, H, V]``; t0: ``[B, H, Tp, C]``;
      g2, beta: ``[B, H, Tp]``; dht: ``[B, H, V, K]`` float32 transposed
        gradient of the final state, or ``None``.

    Returns:
      ``dh [B, H, NC, V, K]`` (transposed gradient of the state leaving each
      chunk, input dtype) and the transposed ``dh0 [B, H, V, K]`` float32
      (with ``want_dh0``).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    nk = _check(key_dim, value_dim, bv)
    group = heads // kv_heads
    nc = padded // C
    nv = value_dim // bv
    dtype = q.dtype
    has_dht = dht is not None
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs
    per_stage = 2 * C * key_dim * isz + 2 * C * bv * isz + C * C * isz + n_vec * C * 4
    fixed = bv * key_dim * isz + (bv * key_dim * 4 if has_dht or want_dh0 else 0) + 4096 + n_vec * 1024
    stages = _stages(per_stage, fixed)
    n_copies = 2 * nk + 2 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch, heads, nc, value_dim, key_dim), dtype)]
    if want_dh0:
        out_type.append(jax.ShapeDtypeStruct((batch, heads, value_dim, key_dim), F32))
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
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
        state_bar=plgpu.Barrier(num_arrivals=nk),
        done=plgpu.Barrier(num_arrivals=3),
    )
    _tiles(scratch, "q_s", nk, smem((stages, C, KB), dtype))
    _tiles(scratch, "k_s", nk, smem((stages, C, KB), dtype))
    _tiles(scratch, "dh_out", nk, smem((bv, KB), dtype))
    if has_dht or want_dh0:
        _tiles(scratch, "state_smem", nk, plgpu.SMEM((bv, KB), F32))
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, **sc):
        q_s, k_s, dh_out = _untile(sc, "q_s", nk), _untile(sc, "k_s", nk), _untile(sc, "dh_out", nk)
        state_smem = _untile(sc, "state_smem", nk) if has_dht or want_dh0 else None
        do_s, t_s, dvl_s, g_state, g_rows, g_cols, b_cols = (sc[n] for n in ("do_s", "t_s", "dvl_s", "g_state",
                                                                             "g_rows", "g_cols", "b_cols"))
        in_bar, in_free, dvl_ready, state_bar, done = (sc[n] for n in ("in_bar", "in_free", "dvl_ready",
                                                                       "state_bar", "done"))
        tail_c, cin_c, run_r, run_c = (sc.get(n) for n in ("tail_c", "cin_c", "run_r", "run_c"))
        refs = list(refs)
        q_ref, k_ref, do_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(6))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dht_ref = refs.pop(0) if has_dht else None
        dh_ref = refs.pop(0)
        dh0_ref = refs.pop(0) if want_dh0 else None
        b, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")
        sq = (C, C)

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(i, slot):  # i: step of the reverse walk
                rows = pl.ds((nc - 1 - i) * C, C)
                bar = in_bar.at[slot]
                for j in range(nk):
                    cols = pl.ds(j * KB, KB)
                    plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk, cols], q_s[j].at[slot], bar)
                    plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk, cols], k_s[j].at[slot], bar)
                plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh, pl.ds(v0, bv)], do_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for i in range(min(stages, nc)):
                issue(i, i)

            @pl.loop(stages, nc)
            def _refill(i):
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(i, slot)

        @pl.when(wg == 1)
        def _intra():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, nc)
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
                pt = _wgmma_sum([(k_s[j].at[slot], q_s[j].at[slot].T) for j in range(nk)], sq)
                pt = (pt * jnp.exp2(expo) * scale).astype(dtype)
                dvl_s.at[slot][...] = _wgmma_sum([(pt, do_s.at[slot])], (C, bv)).astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(dvl_ready.at[slot])
                plgpu.barrier_arrive(in_free.at[slot])

        @pl.when(wg == 0)
        def _state():
            plgpu.set_max_registers(232, action="increase")
            if has_dht:
                for j in range(nk):
                    plgpu.copy_gmem_to_smem(dht_ref.at[b, hh, pl.ds(v0, bv), pl.ds(j * KB, KB)], state_smem[j],
                                            state_bar)
                plgpu.barrier_wait(state_bar)
                ds0 = tuple(plgpu.load(state_smem[j], layout=WGMMA) for j in range(nk))
            else:
                ds0 = tuple(plgpu.layout_cast(jnp.zeros((bv, KB), F32), WGMMA) for _ in range(nk))
            st = (bv, C)

            def step(i, ds_tiles):
                n = nc - 1 - i
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous step's stores have read dh_out
                for j in range(nk):
                    dh_out[j][...] = ds_tiles[j].astype(dtype)
                plgpu.commit_smem()
                for j in range(nk):
                    plgpu.copy_smem_to_gmem(dh_out[j], dh_ref.at[b, hh, n, pl.ds(v0, bv), pl.ds(j * KB, KB)],
                                            commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                g_last = g_state[slot, C - 1]
                g_col = plgpu.load(g_state.at[slot], layout=COLS)  # tokens are the columns of the [BV, C] tiles
                k_decay = jnp.exp2(g_last - g_col)
                gam = jnp.exp2(g_col)
                s_decay = jnp.exp2(g_last)
                if segs:
                    k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                    gam = gam * plgpu.load(cin_c.at[slot], layout=COLS)
                    s_decay = s_decay * cin_c[slot, C - 1]
                kd = _wgmma_sum([(dh_out[j], k_s[j].at[slot].T) for j in range(nk)], st)  # dS^T k^T
                plgpu.barrier_wait(dvl_ready.at[slot])
                dvl_t = plgpu.load(dvl_s.at[slot].T, layout=WGMMA).astype(F32)
                dvn_lo = (kd * bcast_cols(k_decay, st) + dvl_t).astype(dtype)
                dw_t = _wgmma_sum([(dvn_lo, t_s.at[slot])], st)  # dW^T = dv_new^T Ag
                do_t = plgpu.load(do_s.at[slot].T, layout=WGMMA).astype(F32)
                do_t = (do_t * bcast_cols(gam * scale, st)).astype(dtype)
                dwn = (dw_t * bcast_cols(-gam, st)).astype(dtype)
                ds_new = _wgmma_tiles([[(do_t, q_s[j].at[slot]), (dwn, k_s[j].at[slot])] for j in range(nk)],
                                      [ds_tiles[j] * s_decay for j in range(nk)])
                plgpu.barrier_arrive(in_free.at[slot])
                return tuple(ds_new)

            ds_t = lax.fori_loop(0, nc, step, ds0)
            if want_dh0:
                for j in range(nk):
                    state_smem[j][...] = ds_t[j]
                plgpu.commit_smem()
                for j in range(nk):
                    plgpu.copy_smem_to_gmem(state_smem[j], dh0_ref.at[b, hh, pl.ds(v0, bv), pl.ds(j * KB, KB)],
                                            commit_group=False)
                plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, do, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + ([dht] if has_dht else [])
    outs = list(build_kernel(
        body, name="gdr_kb_state_bwd", out_type=out_type, scratch=scratch, grid=(batch, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args))
    dh = outs.pop(0)
    dh0 = outs.pop(0) if want_dh0 else None
    return dh, dh0


def chunk_grads_kb(
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
) -> tuple[Array, Array, Array, Array, Array]:
    """Per-chunk ``dq, dk, dv, dbeta`` and gate gradient of the delta rule at large head dims, in one kernel.

    The algebra of :func:`gated_delta_rule_bwd.chunk_grads` with the states
    streamed in blocks of 64 value rows: one CTA per (chunk, value head, 128
    key columns of ``dq``/``dk``). The CTA loads ``q, k`` and ``T0`` once, then
    for every value block ``S_r, dS'_r [64, K]`` and ``v_r, do_r [C, 64]``:

    * WG0: ``U_r = Gamma (k S_r^T)``, ``W_r = v_r - U_r``, ``v_new_r = Ag W_r``,
      and accumulates ``do v_new^T`` (and its transpose), ``do S`` on its key
      columns and ``<S, dS'>``;
    * WG1: ``dv_new_r = scale P^T do_r + Gamma_r (k dS'_r^T)``,
      ``dv_r = Ag^T dv_new_r`` (stored), and accumulates ``dv_new W^T`` (and
      its transpose), ``Gamma_r (v_new dS') - Gamma (dW S)`` on its key columns
      and the gate terms;
    * WG2 streams the blocks with TMA.

    After the last block WG0 finishes ``dq = scale Gamma (do S) + dP k`` and WG1
    goes through ``Ag`` and the inverse ``T0`` and finishes
    ``dk += dP^T q + dA k + dA^T k``, exactly as at ``K <= 128``. Every
    contraction over ``K`` runs over the CTA's ``K / 64`` key tiles, which are
    loaded rotated so the CTA's own columns are the first tiles (a sum over
    all tiles does not depend on their order). The CTAs of one chunk compute
    the value-side terms redundantly; only the first key block stores ``dv``
    and ``dbeta`` and adds the gate terms that do not depend on the key columns.
    The key blocks of one (chunk, head) are neighbours in the launch order (the
    last grid dimension is CUDA's ``x``), so the second CTA's state tiles
    usually come from L2.

    Args as in :func:`gated_delta_rule_bwd.chunk_grads`: ``q, k [B, Tp, Hk, K]``;
    ``v, do [B, Tp, H, V]``; ``t0 [B, H, Tp, C]``; ``g2, beta [B, H, Tp]``;
    ``h [B, H, NC, V, K]`` (states entering each chunk, transposed) and
    ``dh [B, H, NC, V, K]`` (gradients of the states leaving each chunk).

    Returns:
      ``dq, dk [B, Tp, H, K]`` per value head and ``dv [B, Tp, H, V]`` in the
      input dtype, ``dbeta [B, H, Tp]`` float32 and the gate gradient
      ``dG [NKS, B, H, Tp]`` float32 (one partial per key block, to be summed;
      natural units, before the reverse chunk cumsum).
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    bv = C  # value block; dq/dk are staged in the [bv, 64] state tiles after the loop
    nk = _check(key_dim, value_dim, bv)
    nks = max(1, key_dim // 128)  # key blocks of dq/dk, one CTA each
    nko = nk // nks  # key tiles owned by one CTA
    group = heads // kv_heads
    nc = padded // C
    nvb = value_dim // bv
    dtype = q.dtype
    segs = seg is not None
    vec = plgpu.SMEM((C,), F32)
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, key_dim), dtype),
        jax.ShapeDtypeStruct((batch, padded, heads, value_dim), dtype),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
        jax.ShapeDtypeStruct((nks, batch, heads, padded), F32),
    ]
    scratch = dict(
        t_s=smem((C, C), dtype),  # T0, then dA0 * beta_i (WG1)
        ag_s=smem((C, C), dtype),  # written by WG1
        dp_s=smem((C, C), dtype),  # written by WG0
        m_s=smem((C, C), dtype),
        v_s=smem((C, bv), dtype),  # v_r, then U_r (WG0)
        do_s=smem((C, bv), dtype),
        w_s=smem((C, bv), dtype),
        vn_s=smem((C, bv), dtype),
        dvn_s=smem((C, bv), dtype),
        dv_out=smem((C, bv), dtype),
        g_r=vec,
        g_c=vec,
        b_r=vec,
        b_c=vec,
        sd_s=vec,
        sum_s=vec,
        dg_out=vec,
        db_out=vec,
        in_a=plgpu.Barrier(num_arrivals=2 * nk + 5 + 4 * segs),  # q, k tiles, T0 and the vectors
        in_b=plgpu.Barrier(num_arrivals=2 * nk + 2),  # one value block: S, dS' tiles, v, do
        in_free=plgpu.Barrier(num_arrivals=2),
        wv_ready=plgpu.Barrier(),  # WG0 -> WG1: W, v_new and U of the current block
        mid_free=plgpu.Barrier(),  # WG1 -> WG0: done with the block's W, v_new, U and state tiles
        dp_ready=plgpu.Barrier(),
        dga_ready=plgpu.Barrier(),
        done=plgpu.Barrier(num_arrivals=3),
    )
    _tiles(scratch, "q_s", nk, smem((C, KB), dtype))
    _tiles(scratch, "k_s", nk, smem((C, KB), dtype))
    _tiles(scratch, "h_s", nk, smem((bv, KB), dtype))  # then dq (WG0)
    _tiles(scratch, "dh_s", nk, smem((bv, KB), dtype))  # then dk (WG1)
    if segs:
        scratch.update(run_r=vec, run_c=vec, cin_r=vec, tail_r=vec)

    def body(*refs, **sc):
        q_s, k_s = _untile(sc, "q_s", nk), _untile(sc, "k_s", nk)
        h_s, dh_s = _untile(sc, "h_s", nk), _untile(sc, "dh_s", nk)
        t_s, ag_s, dp_s, m_s, v_s, do_s, w_s, vn_s, dvn_s, dv_out = (
            sc[n] for n in ("t_s", "ag_s", "dp_s", "m_s", "v_s", "do_s", "w_s", "vn_s", "dvn_s", "dv_out"))
        g_r, g_c, b_r, b_c, sd_s, sum_s, dg_out, db_out = (
            sc[n] for n in ("g_r", "g_c", "b_r", "b_c", "sd_s", "sum_s", "dg_out", "db_out"))
        in_a, in_b, in_free, wv_ready, mid_free, dp_ready, dga_ready, done = (
            sc[n] for n in ("in_a", "in_b", "in_free", "wv_ready", "mid_free", "dp_ready", "dga_ready", "done"))
        run_r, run_c, cin_r, tail_r = (sc.get(n) for n in ("run_r", "run_c", "cin_r", "tail_r"))
        refs = list(refs)
        q_ref, k_ref, v_ref, do_ref, t_ref, g_ref, b_ref, h_ref, dh_ref = (refs.pop(0) for _ in range(9))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dq_ref, dk_ref, dv_ref, db_ref, dg_ref = refs
        hk_idx, n, b = lax.axis_index("hk"), lax.axis_index("n"), lax.axis_index("b")
        hh = lax.div(hk_idx, nks)
        kh = lax.rem(hk_idx, nks)
        hkv = lax.div(hh, group)
        rows = pl.ds(n * C, C)
        wg = lax.axis_index("wg")
        sq, sv, st = (C, C), (C, bv), (C, KB)
        first = (kh == 0).astype(F32)  # the key-independent gate terms are added by the first key block

        def key_cols(i):
            """Key columns of SMEM tile ``i`` (rotated: the CTA's own tiles come first)."""
            return pl.ds(lax.rem(kh * nko + i, nk) * KB, KB)

        def zeros(shape):
            return plgpu.layout_cast(jnp.zeros(shape, F32), WGMMA)

        def gates():
            g_row = plgpu.load(g_r, layout=ROWS)
            g_col = plgpu.load(g_c, layout=COLS)
            runs = None
            gam = jnp.exp2(g_row)
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

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")
            for i in range(nk):
                cols = key_cols(i)
                plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hkv, cols], q_s[i], in_a)
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hkv, cols], k_s[i], in_a)
            loads = [(t_ref.at[b, hh, rows], t_s), (g_ref.at[b, hh, rows], g_r), (g_ref.at[b, hh, rows], g_c),
                     (b_ref.at[b, hh, rows], b_r), (b_ref.at[b, hh, rows], b_c)]
            if segs:
                loads += [(run_ref.at[b, rows], run_r), (run_ref.at[b, rows], run_c), (cin_ref.at[b, rows], cin_r),
                          (tail_ref.at[b, rows], tail_r)]
            for src, dst in loads:
                plgpu.copy_gmem_to_smem(src, dst, in_a)

            def issue(r):
                vrows = pl.ds(r * bv, bv)
                for i in range(nk):
                    cols = key_cols(i)
                    plgpu.copy_gmem_to_smem(h_ref.at[b, hh, n, vrows, cols], h_s[i], in_b)
                    plgpu.copy_gmem_to_smem(dh_ref.at[b, hh, n, vrows, cols], dh_s[i], in_b)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, vrows], v_s, in_b)
                plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh, vrows], do_s, in_b)

            issue(0)

            @pl.loop(1, nvb)
            def _refill(r):
                plgpu.barrier_wait(in_free)
                issue(r)

        @pl.when(wg == 0)
        def _forward_side():
            plgpu.set_max_registers(232, action="increase")
            plgpu.barrier_wait(in_a)
            g_row, g_col, runs, gam = gates()
            decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
            ag_lo = (plgpu.load(t_s, layout=WGMMA).astype(F32) * decay
                     * bcast_cols(plgpu.load(b_c, layout=COLS), sq)).astype(dtype)

            def step(r, carry):
                dq_t, dp, dpt, sd = carry
                plgpu.barrier_wait(in_b)
                u = _wgmma_sum([(k_s[i], h_s[i].T) for i in range(nk)], sv) * bcast_rows(gam, sv)
                w = (plgpu.load(v_s, layout=WGMMA).astype(F32) - u).astype(dtype)
                plgpu.barrier_wait(mid_free)  # WG1 has read the previous block's W, v_new and U
                w_s[...] = w
                v_s[...] = u.astype(dtype)
                plgpu.commit_smem()
                vn_s[...] = _wgmma_sum([(ag_lo, w_s)], sv).astype(dtype)
                plgpu.commit_smem()
                plgpu.barrier_arrive(wv_ready)
                dp, dpt = _wgmma_tiles([[(do_s, vn_s.T)], [(vn_s, do_s.T)]], [dp, dpt])
                dq_t = _wgmma_tiles([[(do_s, h_s[j])] for j in range(nko)], dq_t)
                for j in range(nko):  # <S, dS'> on the CTA's key columns (rows: value rows of the block)
                    sd = sd + (plgpu.load(h_s[j], layout=WGMMA).astype(F32)
                               * plgpu.load(dh_s[j], layout=WGMMA).astype(F32)).sum(axis=1)
                plgpu.barrier_arrive(in_free)
                return tuple(dq_t), dp, dpt, sd

            init = (tuple(zeros(st) for _ in range(nko)), zeros(sq), zeros(sq),
                    plgpu.layout_cast(jnp.zeros((C,), F32), ROWS))
            dq_t, dp, dpt, sd = lax.fori_loop(0, nvb, step, init)
            # dP = scale (do v_new^T) * D; its gate gradient is the row sums minus the column sums of
            # dP * (q k^T), the column sums taken as rows of the transposed product.
            dp = dp * decay * scale
            qk = _wgmma_sum([(q_s[i], k_s[i].T) for i in range(nk)], sq)
            full = (dp * qk).sum(axis=1)
            kq = _wgmma_sum([(k_s[i], q_s[i].T) for i in range(nk)], sq)
            full = full - (dpt * decay_t(g_row, g_col, runs) * kq).sum(axis=1) * scale
            dp_lo = dp.astype(dtype)
            dp_s[...] = dp_lo
            plgpu.commit_smem()
            plgpu.barrier_arrive(dp_ready)
            # dq = scale Gamma (do S) + dP k on the CTA's key columns; the state part feeds the gate gradient.
            dq_t = [x * bcast_rows(gam * scale, st) for x in dq_t]
            part = None
            for j in range(nko):
                x = (plgpu.load(q_s[j], layout=WGMMA).astype(F32) * dq_t[j]).sum(axis=1)
                part = x if part is None else part + x
            dq_t = _wgmma_tiles([[(dp_lo, k_s[j])] for j in range(nko)], dq_t)
            plgpu.barrier_wait(mid_free)  # WG1 is done with the last block's state tiles: they stage dq
            for j in range(nko):
                h_s[j][...] = dq_t[j].astype(dtype)
            plgpu.commit_smem()
            for j in range(nko):
                plgpu.copy_smem_to_gmem(h_s[j], dq_ref.at[b, rows, hh, pl.ds((kh * nko + j) * KB, KB)],
                                        commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            dg_out[...] = full * first + part
            sd_s[...] = sd
            plgpu.commit_smem()
            plgpu.barrier_arrive(dga_ready)
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == 1)
        def _backward_side():
            plgpu.set_max_registers(232, action="increase")
            plgpu.barrier_arrive(mid_free)  # the first block may write W, v_new and U
            plgpu.barrier_wait(in_a)
            g_row, g_col, runs, gam = gates()
            b_col = plgpu.load(b_c, layout=COLS)
            g_last = g_r[C - 1]
            k_decay = jnp.exp2(g_last - g_row)  # Gamma_r: the keys write the outgoing state
            s_decay = jnp.exp2(g_last)
            if segs:
                k_decay = k_decay * plgpu.load(tail_r, layout=ROWS)
                s_decay = s_decay * cin_r[C - 1]
            decay = decay_matrix(g_row, g_col, sq, strict=False, runs=runs)
            ag_s[...] = (plgpu.load(t_s, layout=WGMMA).astype(F32) * decay * bcast_cols(b_col, sq)).astype(dtype)
            plgpu.commit_smem()
            pt = _wgmma_sum([(k_s[i], q_s[i].T) for i in range(nk)], sq)
            pt = (pt * decay_t(g_row, g_col, runs) * scale).astype(dtype)

            def step(r, carry):
                dk_t, dag, dagt, rks, dgu = carry
                plgpu.barrier_wait(in_b)
                kdh = _wgmma_sum([(k_s[i], dh_s[i].T) for i in range(nk)], sv)  # k dS'^T
                dvn = _wgmma_sum([(pt, do_s)], sv, init=kdh * bcast_rows(k_decay, sv))
                dvn_s[...] = dvn.astype(dtype)
                plgpu.commit_smem()
                dv = _wgmma_sum([(ag_s.T, dvn_s)], sv)  # dW = Ag^T dv_new: the value gradient
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous block's store has read dv_out
                dv_out[...] = dv.astype(dtype)
                plgpu.commit_smem()

                @pl.when(kh == 0)
                def _store_dv():
                    plgpu.copy_smem_to_gmem(dv_out, dv_ref.at[b, rows, hh, pl.ds(r * bv, bv)])

                plgpu.barrier_wait(wv_ready)
                dgu = dgu - (dv * plgpu.load(v_s, layout=WGMMA).astype(F32)).sum(axis=1)  # W = v - U
                dwn = (dv * bcast_rows(-gam, sv)).astype(dtype)
                vnr = (plgpu.load(vn_s, layout=WGMMA).astype(F32) * bcast_rows(k_decay, sv)).astype(dtype)
                rks = rks + (vnr.astype(F32) * kdh).sum(axis=1)  # k . (Gamma_r v_new dS'), by value block
                dag, dagt = _wgmma_tiles([[(dvn_s, w_s.T)], [(w_s, dvn_s.T)]], [dag, dagt])
                dk_t = _wgmma_tiles([[(vnr, dh_s[j]), (dwn, h_s[j])] for j in range(nko)], dk_t)
                plgpu.barrier_arrive(mid_free)
                plgpu.barrier_arrive(in_free)
                return tuple(dk_t), dag, dagt, rks, dgu

            zero_rows = plgpu.layout_cast(jnp.zeros((C,), F32), ROWS)
            init = (tuple(zeros(st) for _ in range(nko)), zeros(sq), zeros(sq), zero_rows, zero_rows)
            dk_t, dag, dagt, rks, dgu = lax.fori_loop(0, nvb, step, init)
            # Ag = T0 * D * beta_j: to beta (column sums), the gates (row minus column sums) and T0.
            b_row = plgpu.load(b_r, layout=ROWS)
            t0d = plgpu.load(t_s, layout=WGMMA).astype(F32) * decay
            full = dgu + (dag * t0d * bcast_cols(b_col, sq)).sum(axis=1)
            x = dagt * plgpu.load(t_s.T, layout=WGMMA).astype(F32) * decay_t(g_row, g_col, runs)
            db = x.sum(axis=1)
            full = full - (x * bcast_rows(b_row, sq)).sum(axis=1)
            # Through the inverse: dA0 = -T0^T dT0 T0^T on the strictly lower part (T0 has a unit diagonal).
            strict = iota(sq, 0) > iota(sq, 1)
            ndt = jnp.where(strict, dag * decay * bcast_cols(-b_col, sq), 0.0)
            m_s[...] = _wgmma_sum([(ndt.astype(dtype), t_s.T)], sq).astype(dtype)
            plgpu.commit_smem()
            if segs:
                strict = strict & same_run(runs, sq)
            da = jnp.where(strict, _wgmma_sum([(t_s.T, m_s)], sq), 0.0)
            kk = _wgmma_sum([(k_s[i], k_s[i].T) for i in range(nk)], sq)
            db = db + (da * kk).sum(axis=1)
            t_s[...] = (da * bcast_rows(b_row, sq)).astype(dtype)
            plgpu.commit_smem()
            plgpu.barrier_wait(dp_ready)  # WG0 is past its value loop and has stored dP
            dk_t = _wgmma_tiles([[(dp_s.T, q_s[j]), (t_s, k_s[j]), (t_s.T, k_s[j])] for j in range(nko)], dk_t)
            for j in range(nko):
                dh_s[j][...] = dk_t[j].astype(dtype)
            plgpu.commit_smem()
            for j in range(nko):
                plgpu.copy_smem_to_gmem(dh_s[j], dk_ref.at[b, rows, hh, pl.ds((kh * nko + j) * KB, KB)],
                                        commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            # The last token also gets sum(rks) + gamma <S, dS'> (the outgoing decays).
            plgpu.barrier_wait(dga_ready)
            sum_s[...] = rks * first + plgpu.load(sd_s, layout=ROWS) * s_decay
            plgpu.commit_smem()
            total = plgpu.load(sum_s, layout=COLS).sum()
            last = plgpu.broadcasted_iota(jnp.int32, (C,), 0, layout=ROWS) == C - 1
            dg_out[...] = plgpu.load(dg_out, layout=ROWS) + (full - rks) * first + jnp.where(last, total, 0.0)
            db_out[...] = db
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dg_out, dg_ref.at[kh, b, hh, rows])

            @pl.when(kh == 0)
            def _store_db():
                plgpu.copy_smem_to_gmem(db_out, db_ref.at[b, hh, rows])

            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, v, do, t0, g2, beta, h, dh] + ([seg.run, seg.cin, seg.tail] if segs else [])
    dq, dk, dv, db, dg = build_kernel(
        body, name="gdr_kb_chunk_grads", out_type=out_type, scratch=scratch, grid=(nc, batch, heads * nks),
        grid_names=("n", "b", "hk"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args)
    return dq, dk, dv, db, dg
