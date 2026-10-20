# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/kda/chunk_bwd.py, fla/ops/kda/wy_fast.py,
# fla/ops/kda/chunk_intra.py, fla/ops/common/chunk_delta_h.py
# The channel-wise erase/write gates (GDN-2) are implemented from arXiv 2605.22791 (equations only).
"""Backward Mosaic GPU kernels for :mod:`jla.ops.mosaic_gpu.vector_gate`.

With ``dS_{n+1}`` the gradient of the state leaving chunk ``n`` (``dS_NC`` = the final-state
cotangent), per chunk:

* ``bwd_dav``: ``dAqk = do v_new^T`` (causal), ``dv_o = Aqk^T do`` and ``qg = scale q exp(G)``;
* ``bwd_state``: serial and reversed, ``dv_new = dv_o + kg dS_{n+1}`` and
  ``dS_n = exp(G_last) dS_{n+1} + qg^T do - w^T dv_new``;
* ``bwd_wy``: the gradients through ``o = qg S_n``, ``S_{n+1} = exp(G_last) S_n + kg^T v_new``,
  ``v_new = u - w S_n``, the WY factors ``w = T kb``, ``u = T vb`` and ``T = (I + Akk)^-1``
  (``dAkk = -T^T dT T^T``);
* ``bwd_intra``: the gradients through the decayed score matrices ``Aqk`` and ``Akk`` with the
  same sub-chunk construction as the forward (exact diagonal blocks).

The caller turns the per-token ``dG`` into the gate gradient with a reverse cumulative sum.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import finish_together
from jla.ops.mosaic_gpu.vector_gate import (
    CHUNK,
    SUB_CHUNK,
    _BF16,
    _COL,
    _F32,
    _HALF,
    _KC,
    _MARKS,
    _NUM_SUB,
    _ROW,
    _WG,
    Config,
    _Regs,
    _cols,
    _iota,
    _key_tiles,
    _kernel,
    _load,
    _masked_exp,
    _matmul,
    _per_block,
    _rows,
    _slab_smem,
    _split_bf16,
    _swizzled,
)

Array = jax.Array

# The previous schedules of bwd_intra and bwd_wy: one code copy per warpgroup, one wait per
# tensor-core product, every input awaited before the first product. Their outputs are the same
# bits; tests/test_vector_gate_bwd_schedule.py compares the two.
_REFERENCE_SCHEDULE = False


def bwd_dav(cfg: Config, q: Array, g_cum: Array, do: Array, v_new: Array, aqk: Array) -> tuple[Array, Array, Array]:
    """``(dAqk, dv_o, qg)``: ``dAqk = tril(do v_new^T)`` ``[B, H, T, C]`` f32 (the gradient of the
    stored ``Aqk``), ``dv_o = Aqk^T do`` ``[B, H, T, V]`` bf16 and ``qg = scale q exp(G)``
    ``[B, H, T, K]`` bf16. ``do`` is ``[B, T, H, V]`` like ``o``.
    """
    c, kd, vd = CHUNK, cfg.key_dim, cfg.value_dim
    shape_a = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _F32)
    shape_v = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, vd), _BF16)
    shape_k = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, kd), _BF16)

    def body(q_ref, g_ref, do_ref, vn_ref, a_ref, da_ref, dv_ref, qg_ref, *scratch):
        q_smem, g_smem, do_smem, vn_smem, a_smem, da_smem, dv_smem, qg_smem, bar = scratch
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, bar)
        plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, bar)
        plgpu.copy_gmem_to_smem(do_ref.at[b, tok, h], do_smem, bar)
        plgpu.copy_gmem_to_smem(vn_ref.at[b, h, tok], vn_smem, bar)
        plgpu.copy_gmem_to_smem(a_ref.at[b, h, tok], a_smem, bar)
        plgpu.barrier_wait(bar)
        qg_smem[...] = (_load(q_smem).astype(_F32) * jnp.exp(_load(g_smem)) * cfg.scale).astype(_BF16)
        causal = _iota((c, c), 1) <= _iota((c, c), 0)
        da_smem[...] = jnp.where(causal, _matmul(cfg, (c, c), (do_smem, vn_smem.T)), 0.0)
        dv_smem[...] = _matmul(cfg, (c, vd), (a_smem.T, do_smem)).astype(_BF16)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(da_smem, da_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(dv_smem, dv_ref.at[b, h, tok])
        plgpu.copy_smem_to_gmem(qg_smem, qg_ref.at[b, h, tok])
        plgpu.wait_smem_to_gmem(0)

    scratch = (
        _swizzled((c, kd), q.dtype),
        _swizzled((c, kd), _F32),
        _swizzled((c, vd), do.dtype),
        _swizzled((c, vd)),
        _swizzled((c, c)),
        _swizzled((c, c), _F32),
        _swizzled((c, vd)),
        _swizzled((c, kd)),
        plgpu.Barrier(num_arrivals=5),
    )
    kern = _kernel(
        cfg, "vg_bwd_dav", body, out_type=(shape_a, shape_v, shape_k), grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), scratch_types=scratch,
    )
    return kern(q, g_cum, do, v_new, aqk)


def bwd_state(
    cfg: Config, qg: Array, kg: Array, w: Array, g_last: Array, do: Array, dv_o: Array | None, d_final: Array,
    *, aqk: Array | None = None, do_rows: int = CHUNK, pieces: Array | None = None,
) -> tuple[Array, Array, Array]:
    """``(dh, dv_new, dh0)``: ``dh [B, H, NC, K, V]`` bf16 holds ``dS_{n+1}`` for chunk ``n``,
    ``dv_new [B, H, T, V]`` bf16 is the gradient of ``v_new`` (and of ``u``) and ``dh0`` f32 the
    gradient of the initial state. ``d_final [B, H, K, V]`` is the final-state cotangent.

    With ``aqk`` (``[B, H, T, C]``, the forward's ``Aqk`` tiles) the kernel forms ``dv_o = Aqk^T do``
    itself (in f32, rounded to bf16 as the ``dv_o`` input is) and ``dv_o`` is ``None``; ``do`` then
    holds only the first ``do_rows`` rows of each chunk, ``[B, NC do_rows, H, V]`` (RWKV-7's 64-row
    chunks: 32 output rows, then 32 rows without outputs, whose ``do`` is zero).

    With ``pieces`` (``vector_gate.piece_table``) the reversed pass walks the forward pass's pieces,
    one CTA each: a chunk where a segment starts passes no gradient back to the state entering it from
    the state leaving it, so a piece's walk starts at its last chunk (the next piece's first) from a
    zero gradient and stores nothing there (the next piece stores that chunk); the last piece starts
    from ``d_final`` and the first writes ``dh0``. The same results as one chain per row.
    """
    c, kd, vd, bv, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.block_v, cfg.num_chunks
    own_dvo = aqk is not None
    if own_dvo == (dv_o is not None):
        raise ValueError("bwd_state takes either dv_o or aqk")
    split = pieces is not None
    n_pieces = pieces.shape[0] // cfg.batch if split else 1
    out_h = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd, vd), _BF16)
    out_v = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, vd), _BF16)
    out_s = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32)

    def stored(i, skip, store) -> None:  # step i of the walk stores its chunk's results unless skipped
        if split:
            pl.when(i >= skip)(store)
        else:
            store()

    def body(*refs):
        qg_ref, kg_ref, w_ref, gl_ref, do_ref, dvo_ref, df_ref = refs[:7]
        piece_ref = refs[7] if split else None
        dh_ref, dvn_ref, dh0_ref = refs[7 + split:10 + split]
        qg_smem, kg_smem, w_smem, gl_smem, do_smem, dvo_smem, ds_smem, dvn_smem, neg_smem, f_smem, bars, f_bar = (
            refs[10 + split:]
        )  # with aqk: dvo_ref and dvo_smem hold Aqk
        vt, h, bp = lax.axis_index("v"), lax.axis_index("h"), lax.axis_index("b")
        if split:  # walk chunks first + count - 1 down to first; a piece other than the last skips its first step
            b, piece = lax.div(bp, n_pieces), lax.rem(bp, n_pieces)
            first, count, last_piece = piece_ref[bp, 0], piece_ref[bp, 1], piece_ref[bp, 3]
            skip = 1 - last_piece
        else:
            b, piece, first, count, skip, last_piece = bp, 0, 0, nc, 0, 1
        vcols = pl.ds(vt * bv, bv)

        def chunk(i):  # the chunk of the walk's step i
            return first + count - 1 - i if split else nc - 1 - i

        def fetch(i, slot):
            n = chunk(i)
            tok = pl.ds(n * c, c)
            bar = bars.at[slot]
            plgpu.copy_gmem_to_smem(qg_ref.at[b, h, tok], qg_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(kg_ref.at[b, h, tok], kg_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(w_ref.at[b, h, tok], w_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(gl_ref.at[b, n, h], gl_smem.at[slot], bar)
            if own_dvo:
                rows = pl.ds(n * do_rows, do_rows)
                plgpu.copy_gmem_to_smem(do_ref.at[b, rows, h, vcols], do_smem.at[slot, pl.ds(0, do_rows)], bar)
                plgpu.copy_gmem_to_smem(dvo_ref.at[b, h, tok], dvo_smem.at[slot], bar)
            else:
                plgpu.copy_gmem_to_smem(do_ref.at[b, tok, h, vcols], do_smem.at[slot], bar)
                plgpu.copy_gmem_to_smem(dvo_ref.at[b, h, tok, vcols], dvo_smem.at[slot], bar)

        if own_dvo and do_rows < c:  # the rows without outputs: zero do, never loaded
            for slot in range(2):
                do_smem.at[slot][...] = plgpu.layout_cast(jnp.zeros((c, bv), do.dtype), _WG)
            plgpu.commit_smem()
        def walk():
            plgpu.copy_gmem_to_smem(df_ref.at[b, h, :, vcols], f_smem, f_bar)
            fetch(0, 0)
            plgpu.barrier_wait(f_bar)
            ds0 = _load(f_smem)
            if split:  # a piece other than the last starts in the next piece's first chunk: no gradient enters
                ds0 = ds0 * last_piece.astype(_F32)

            def step(i, ds):
                n = chunk(i)
                slot = lax.rem(i, 2)
                plgpu.commit_smem()

                @pl.when(i + 1 < count)
                def _():
                    fetch(i + 1, 1 - slot)

                plgpu.barrier_wait(bars.at[slot])
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                ds_smem[...] = ds.astype(_BF16)
                plgpu.commit_smem()
                stored(i, skip, lambda: plgpu.copy_smem_to_gmem(ds_smem, dh_ref.at[b, h, n, :, vcols],
                                                                commit_group=False))
                if own_dvo:  # dv_o = Aqk^T do, rounded to bf16
                    dvo = _matmul(cfg, (c, bv), (dvo_smem.at[slot].T, do_smem.at[slot])).astype(_BF16).astype(_F32)
                else:
                    dvo = _load(dvo_smem.at[slot]).astype(_F32)
                dv_new = dvo + _matmul(cfg, (c, bv), (kg_smem.at[slot], ds_smem))
                dvn_smem[...] = dv_new.astype(_BF16)
                neg_smem[...] = (-dv_new).astype(_BF16)
                plgpu.commit_smem()

                def store_dv_new():
                    plgpu.copy_smem_to_gmem(dvn_smem, dvn_ref.at[b, h, pl.ds(n * c, c), vcols], commit_group=False)
                    plgpu.commit_smem_to_gmem_group()

                stored(i, skip, store_dv_new)
                decayed = ds * _rows(jnp.exp(_load(gl_smem.at[slot], _ROW)), (kd, bv))
                pairs = ((qg_smem.at[slot].T, do_smem.at[slot]), (w_smem.at[slot].T, neg_smem))
                return _matmul(cfg, (kd, bv), *pairs, init=decayed)

            ds = lax.fori_loop(0, count, step, ds0)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            def store_dh0():
                f_smem[...] = ds
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(f_smem, dh0_ref.at[b, h, :, vcols])

            if split:  # the first piece ends at the row's first chunk
                pl.when(piece == 0)(store_dh0)
            else:
                store_dh0()
            plgpu.wait_smem_to_gmem(0)

        if split:
            pl.when(count > 0)(walk)
        else:
            walk()

    scratch = (
        _key_tiles(cfg, (2, c, kd)),
        _key_tiles(cfg, (2, c, kd)),
        _key_tiles(cfg, (2, c, kd)),
        plgpu.SMEM((2, kd), _F32),
        _swizzled((2, c, bv), do.dtype),
        _key_tiles(cfg, (2, c, c)) if own_dvo else _swizzled((2, c, bv)),  # Aqk (with aqk) or dv_o
        _swizzled((kd, bv)),
        _swizzled((c, bv)),
        _swizzled((c, bv)),
        _swizzled((kd, bv), _F32),
        plgpu.Barrier(num_arrivals=6, num_barriers=2),
        plgpu.Barrier(num_arrivals=1),
    )
    kern = _kernel(
        cfg, "vg_bwd_state", body, out_type=(out_h, out_v, out_s), grid=(vd // bv, cfg.heads, cfg.batch * n_pieces),
        grid_names=("v", "h", "b"), scratch_types=scratch,
    )
    return kern(qg, kg, w, g_last, do, aqk if own_dvo else dv_o, d_final, *((pieces,) if split else ()))


def _parts(cfg: Config) -> int:
    """Grid split of :func:`bwd_wy` over the key and value channels (64-wide slices)."""
    return min(cfg.key_dim, cfg.value_dim) // 64


def bwd_wy(
    cfg: Config,
    qg: Array,
    k: Array,
    v: Array,
    g_cum: Array,
    g_marks: Array,
    erase: Array,
    write: Array,
    t: Array,
    h: Array,
    dh: Array,
    do: Array,
    dv_new: Array,
    v_new: Array,
) -> tuple[Array, ...]:
    """Gradients through the inter-chunk terms and the WY factors of every chunk.

    Returns ``(dq, dk, dG, dv, dT, gate...)``: ``dq``, ``dk``, ``dG`` ``[B, T, H, K]`` f32 (the
    inter-chunk part of ``dq``; partial ``dk`` / ``dG``), ``dv [B, T, H, V]`` (complete, in the
    dtype of ``v``), ``dT [B, H, P, T, C]`` f32 (``P`` partial sums of the gradient of
    ``T = (I + Akk)^-1``), then ``dbeta [B, T, H, P]`` f32 partial sums for KDA, or the f32 partial
    ``d_erase [B, T, H, K]`` and the complete ``d_write [B, T, H, V]`` (dtype of ``write``) for
    GDN-2. Each CTA handles one ``K / P`` slice of the key channels and one ``V / P`` slice of the
    value channels of a chunk. Its inputs arrive in pairs on their own barriers, in the order of
    their first use, so the first products start while the later inputs are still in flight.
    """
    c, kd, vd = CHUNK, cfg.key_dim, cfg.value_dim
    parts = _parts(cfg)
    kp, vp = kd // parts, vd // parts
    b_, h_, t_ = cfg.batch, cfg.heads, cfg.seq_len
    out = [
        jax.ShapeDtypeStruct((b_, t_, h_, kd), _F32),
        jax.ShapeDtypeStruct((b_, t_, h_, kd), _F32),
        jax.ShapeDtypeStruct((b_, t_, h_, kd), _F32),
        jax.ShapeDtypeStruct((b_, t_, h_, vd), v.dtype),
        jax.ShapeDtypeStruct((b_, h_, parts, t_, c), _F32),
    ]
    if cfg.channel_gates:
        out += [jax.ShapeDtypeStruct((b_, t_, h_, kd), _F32), jax.ShapeDtypeStruct((b_, t_, h_, vd), write.dtype)]
    else:
        out += [jax.ShapeDtypeStruct((b_, t_, h_, parts), _F32)]

    def body(qg_ref, k_ref, v_ref, g_ref, gm_ref, e_ref, wr_ref, t_ref, h_ref, dh_ref, do_ref, dvn_ref, vn_ref,
             *refs):
        outs, scratch = refs[: len(out)], refs[len(out):]
        dq_ref, dk_ref, dg_ref, dv_ref, dt_ref = outs[:5]
        (qg_smem, k_smem, g_smem, last_smem, e_smem, wr_smem, v_smem, t_smem, h_smem, dh_smem, do_smem,
         dvn_smem, vn_smem, dw_smem, kb_smem, vb_smem, bars, bar_wr) = scratch
        p, n, hh, b = lax.axis_index("p"), lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        kcols = pl.ds(p * kp, kp)
        vcols = pl.ds(p * vp, vp)
        # The inputs in the order of their first use, two per barrier: the products start while the
        # later inputs are still in flight.
        if cfg.channel_gates:
            gate_k_src, write_src = e_ref.at[b, tok, hh, kcols], wr_ref.at[b, tok, hh, vcols]
        else:
            gate_k_src, write_src = e_ref.at[b, hh, tok], None
        groups = (
            ((do_ref.at[b, tok, hh], do_smem), (h_ref.at[b, hh, n, kcols], h_smem)),
            ((g_ref.at[b, tok, hh, kcols], g_smem), (gm_ref.at[b, n * _MARKS + _MARKS - 1, hh, kcols], last_smem)),
            ((qg_ref.at[b, hh, tok, kcols], qg_smem), (vn_ref.at[b, hh, tok], vn_smem)),
            ((dh_ref.at[b, hh, n, kcols], dh_smem), (k_ref.at[b, tok, hh, kcols], k_smem)),
            ((dvn_ref.at[b, hh, tok], dvn_smem), (gate_k_src, e_smem)),
            ((t_ref.at[b, hh, tok], t_smem), (v_ref.at[b, tok, hh, vcols], v_smem)),
        )
        for i, group in enumerate(groups):
            for src, dst in group:
                plgpu.copy_gmem_to_smem(src, dst, bars.at[i])
        if write_src is not None:
            plgpu.copy_gmem_to_smem(write_src, wr_smem, bar_wr)
        staged = not _REFERENCE_SCHEDULE

        def ready(i):
            if staged:
                plgpu.barrier_wait(bars.at[i])

        if not staged:
            for i in range(len(groups)):
                plgpu.barrier_wait(bars.at[i])
            if write_src is not None:
                plgpu.barrier_wait(bar_wr)
        kshape, vshape = (c, kp), (c, vp)
        rows_k = _iota(kshape, 0)

        # Key side (channels kcols). Inter-chunk output term: o += qg S_n.
        ready(0)
        dqs = _matmul(cfg, kshape, (do_smem, h_smem.T))  # do S_n^T
        ready(1)
        g = _load(g_smem)
        decay = jnp.exp(g)
        last = _load(last_smem, _COL)
        to_end = jnp.exp(_cols(last, kshape) - g)
        dq_ref.at[b, tok, hh, kcols][...] = dqs * decay * cfg.scale
        ready(2)
        dg = _load(qg_smem).astype(_F32) * dqs
        # State update S_{n+1} = exp(G_last) S_n + kg^T v_new with kg = k exp(G_last - G).
        ready(3)
        dkg = _matmul(cfg, kshape, (vn_smem, dh_smem.T))
        kf = _load(k_smem).astype(_F32)
        kgv = kf * to_end
        dk = dkg * to_end
        dg = dg - kgv * dkg
        state_term = _matmul(cfg, (kp, kp), (h_smem, dh_smem.T))  # diagonal: sum_v S dS per channel
        diag = jnp.where(_iota((kp, kp), 0) == _iota((kp, kp), 1), state_term, 0.0)
        d_last = jnp.exp(last) * jnp.sum(diag, axis=0) + jnp.sum(kgv * dkg, axis=0)
        dg = jnp.where(rows_k == c - 1, dg + _cols(d_last, kshape), dg)
        # v_new = u - w S_n, w = T kb with kb = erase exp(G) k.
        ready(4)
        dw = -_matmul(cfg, kshape, (dvn_smem, h_smem.T))
        erase_v = _load(e_smem).astype(_F32) if cfg.channel_gates else _rows(_load(e_smem, _ROW), kshape)
        kb = erase_v * decay * kf
        dw_smem[...] = dw.astype(_BF16)
        kb_smem[...] = kb.astype(_BF16)
        plgpu.commit_smem()
        ready(5)
        dkb = _matmul(cfg, kshape, (t_smem.T, dw_smem))
        dk_ref.at[b, tok, hh, kcols][...] = dk + dkb * erase_v * decay
        dg_ref.at[b, tok, hh, kcols][...] = dg + dkb * kb
        gate_k = dkb * kf * decay  # gradient of the erase gate
        if cfg.channel_gates:
            outs[5].at[b, tok, hh, kcols][...] = gate_k
        dt = _matmul(cfg, (c, c), (dw_smem, kb_smem.T))

        # Value side (columns vcols): u = T vb with vb = write v.
        vf = _load(v_smem).astype(_F32)
        if staged and write_src is not None:
            plgpu.barrier_wait(bar_wr)
        write_v = _load(wr_smem).astype(_F32) if cfg.channel_gates else _rows(_load(e_smem, _ROW), vshape)
        vb_smem[...] = (vf * write_v).astype(_BF16)
        plgpu.commit_smem()
        dvn_cols = dvn_smem.at[:, vcols]
        dvb = _matmul(cfg, vshape, (t_smem.T, dvn_cols))
        dv_ref.at[b, tok, hh, vcols][...] = (dvb * write_v).astype(dv_ref.dtype)
        if cfg.channel_gates:
            outs[6].at[b, tok, hh, vcols][...] = (dvb * vf).astype(outs[6].dtype)
        else:
            dbeta = jnp.sum(gate_k, axis=1) + jnp.sum(dvb * vf, axis=1)
            outs[5].at[b, tok, hh, p][...] = dbeta
        dt_ref.at[b, hh, p, tok][...] = dt + _matmul(cfg, (c, c), (dvn_cols, vb_smem.T))

    if cfg.channel_gates:
        gate_scratch = (_swizzled((c, kp), erase.dtype), _swizzled((c, vp), write.dtype))
    else:
        gate_scratch = (plgpu.SMEM((c,), _F32), plgpu.SMEM((c,), _F32))
    scratch = (
        _swizzled((c, kp)),
        _swizzled((c, kp), k.dtype),
        _swizzled((c, kp), _F32),
        plgpu.SMEM((kp,), _F32),
        *gate_scratch,
        _swizzled((c, vp), v.dtype),
        _swizzled((c, c)),
        _swizzled((kp, vd)),
        _swizzled((kp, vd)),
        _swizzled((c, vd), do.dtype),
        _swizzled((c, vd)),
        _swizzled((c, vd)),
        _swizzled((c, kp)),
        _swizzled((c, kp)),
        _swizzled((c, vp)),
        plgpu.Barrier(num_arrivals=2, num_barriers=6),  # the inputs, two per barrier in the order of use
        plgpu.Barrier(num_arrivals=1),  # the write gate (GDN-2)
    )
    kern = _kernel(
        cfg, "vg_bwd_wy", body, out_type=tuple(out), grid=(parts, cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("p", "n", "h", "b"), scratch_types=scratch,
    )
    return kern(qg, k, v, g_cum, g_marks, erase, write, t, h, dh, do, dv_new, v_new)


def bwd_dakk(cfg: Config, t: Array, dt: Array) -> Array:
    """``dAkk = -T^T dT T^T`` (strictly lower) ``[B, H, T, C]`` f32, from ``T`` and the partial
    sums ``dt [B, H, P, T, C]`` of :func:`bwd_wy`."""
    c, parts = CHUNK, _parts(cfg)
    out = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, cfg.seq_len, c), _F32)

    def body(t_ref, dt_ref, da_ref, t_smem, dt_smem, hi_smem, lo_smem, da_smem, bar):
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        tok = pl.ds(n * c, c)
        plgpu.copy_gmem_to_smem(t_ref.at[b, h, tok], t_smem, bar)
        for p in range(parts):
            plgpu.copy_gmem_to_smem(dt_ref.at[b, h, p, tok], dt_smem.at[p], bar)
        plgpu.barrier_wait(bar)
        d_t = _load(dt_smem.at[0])
        for p in range(1, parts):
            d_t = d_t + _load(dt_smem.at[p])
        full = (c, c)
        hi, lo = _split_bf16(d_t)
        z = _matmul(cfg, full, (_Regs(hi), t_smem.T), (_Regs(lo), t_smem.T))  # dT T^T
        z_hi, z_lo = _split_bf16(z)
        hi_smem[...] = z_hi
        lo_smem[...] = z_lo
        plgpu.commit_smem()
        da = _matmul(cfg, full, (t_smem.T, hi_smem), (t_smem.T, lo_smem))  # T^T dT T^T
        strict = _iota(full, 1) < _iota(full, 0)
        da_smem[...] = jnp.where(strict, -da, 0.0)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(da_smem, da_ref.at[b, h, tok])
        plgpu.wait_smem_to_gmem(0)

    scratch = (
        _swizzled((c, c)),
        plgpu.SMEM((parts, c, c), _F32),
        _swizzled((c, c)),
        _swizzled((c, c)),
        plgpu.SMEM((c, c), _F32),
        plgpu.Barrier(num_arrivals=1 + parts),
    )
    kern = _kernel(
        cfg, "vg_bwd_dakk", body, out_type=out, grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), scratch_types=scratch,
    )
    return kern(t, dt)




_SHIFT = _HALF  # leading rows of the row-shift buffers of bwd_intra (shifts of at most 7 rows)


def _bf16_values(x: Array) -> Array:
    """``x`` rounded to bf16, in f32 (unchanged for bf16 ``x``)."""
    return x.astype(_BF16).astype(_F32)


def _tile16(shape: tuple[int, ...]) -> plgpu.SMEM:
    """bf16 SMEM in ``[8, 16]`` tiles with a 32-byte swizzle: every 16-column slice is a valid
    16-deep WGMMA operand, and all WGMMA operands of :func:`bwd_intra` share this swizzle."""
    return plgpu.SMEM(shape, _BF16, transforms=(plgpu.TilingTransform((8, 16)), plgpu.SwizzleTransform(32)))


def _matmuls(cfg: Config, shape: tuple[int, int], pairs: list) -> list[Array]:
    """``[a @ b for a, b in pairs]`` on the tensor cores (f32 accumulators, SMEM operands).

    Every product is issued before the first wait; each result is the one :func:`_matmul` gives
    (the same instructions on the same operands). With ``_REFERENCE_SCHEDULE`` each product waits.
    """
    if _REFERENCE_SCHEDULE:
        return [_matmul(cfg, shape, pair) for pair in pairs]

    def scoped(*accs):
        for acc, (a, b) in zip(accs, pairs):
            plgpu.wgmma(acc, a, b)
        return [acc[...] for acc in accs]

    return pl.run_scoped(scoped, *(plgpu.ACC(shape, _F32) for _ in pairs))


def bwd_intra(
    cfg: Config,
    q: Array,
    k: Array,
    g_cum: Array,
    g_marks: Array,
    erase: Array,
    daqk: Array,
    dakk: Array,
    dq_part: Array,
    dk_part: Array,
    dg_part: Array,
    gate_part: Array,
) -> tuple[Array, Array, Array, Array]:
    """Gradients through the score matrices, added to the partial gradients of :func:`bwd_wy`.

    ``Aqk = scale P(q, k)`` and ``Akk = P'(kr, k)`` with ``P(a, b)[i, j] = sum_c a_ic b_jc
    e^{G_ic - G_jc}`` (``j <= i``; strictly lower for ``Akk``) and the read-side keys ``kr =
    erase * k`` (``erase = beta`` broadcast over channels for KDA). For an upstream ``M`` the row
    direction gives ``da_i = sum_j M_ij b_j e^{G_i - G_j}`` and the column direction
    ``db_j = sum_i M_ij a_i e^{G_i - G_j}``, both built from the same sub-chunk pieces as the
    forward: below-diagonal blocks and the diagonal blocks' lower-left quadrants on the tensor
    cores (every exponent at most zero) for the pairs at least 8 tokens apart, and the band of
    pairs at most 7 tokens apart exactly in f32 (see ``matrix_pass``).

    Returns ``(dq, dk, dG, gate)``: ``[B, T, H, K]`` each (``dq`` and ``dk`` in the dtypes of ``q``
    and ``k``, ``dG`` f32 per token, before the reverse cumulative sum), and ``gate`` = ``dbeta
    [B, T, H]`` f32 (KDA) or ``d_erase [B, T, H, K]`` in the dtype of ``erase`` (GDN-2).
    ``gate_part`` is :func:`bwd_wy`'s gate output.

    Per chunk, warpgroup ``w`` first stages upstream matrix ``w`` (``scale dAqk``, ``dAkk``): bf16
    copies without the band (all of it, and the quadrants) and the coefficients of its exact band.
    Then each warpgroup takes half of the key channels and runs both matrices on each 32-channel
    slab, sharing the slab's decay factors.

    Both warpgroups run one copy of the code (the warpgroup index is a traced value): with a copy
    per warpgroup the kernel's hot code was twice as large, and the two copies running side by side
    on every SM made it about 1.5x slower on H100. Each direction issues its four
    tensor-core products before one wait. ``_REFERENCE_SCHEDULE`` restores the previous schedule,
    whose outputs are the same bits.
    """
    c, kd = CHUNK, cfg.key_dim
    b_, h_, t_ = cfg.batch, cfg.heads, cfg.seq_len
    shape = (c, _KC)
    full = (c, c)
    count = kd // _KC
    assert count % 2 == 0, f"bwd_intra splits the {_KC}-channel key slabs evenly over two warpgroups, got K={kd}"
    half_count = count // 2  # key slabs per warpgroup
    out = [jax.ShapeDtypeStruct((b_, t_, h_, kd), dtype) for dtype in (q.dtype, k.dtype, _F32)]
    if cfg.channel_gates:
        out.append(jax.ShapeDtypeStruct((b_, t_, h_, kd), erase.dtype))
    else:
        out.append(jax.ShapeDtypeStruct((b_, t_, h_), _F32))

    def body(q_ref, k_ref, g_ref, gm_ref, e_ref, daqk_ref, dakk_ref, dqp_ref, dkp_ref, dgp_ref, gp_ref,
             dq_ref, dk_ref, dg_ref, gate_ref, *scratch):
        (q_smem, k_smem, g_smem, marks_smem, e_smem, m_bf, mq_bf, coef_smem, x_smem, x2_smem, y_union,
         kpad, epad, apad, dbeta_smem, bar, sync, finish) = scratch
        y_smem, m_f32 = y_union  # the f32 upstream matrices are only needed before the slabs
        n, h, b = lax.axis_index("n"), lax.axis_index("h"), lax.axis_index("b")
        wg = lax.axis_index("wg")
        tok = pl.ds(n * c, c)

        def per_warpgroup(fn, *per_wg):
            """``fn(w, *values)`` on warpgroup ``w``, ``values`` picked from ``per_wg`` (one pair each)."""
            if _REFERENCE_SCHEDULE:
                for w in range(2):
                    pl.when(wg == w)(functools.partial(fn, w, *(pair[w] for pair in per_wg)))
            else:
                fn(wg, *(jnp.where(wg == 0, pair[0], pair[1]) for pair in per_wg))

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(daqk_ref.at[b, h, tok], m_f32.at[0], bar)
            plgpu.copy_gmem_to_smem(dakk_ref.at[b, h, tok], m_f32.at[1], bar)
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem, bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem, bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, tok, h], g_smem, bar)
            plgpu.copy_gmem_to_smem(gm_ref.at[b, pl.ds(n * _MARKS, _MARKS), h], marks_smem, bar)
            if cfg.channel_gates:
                plgpu.copy_gmem_to_smem(e_ref.at[b, tok, h], e_smem, bar)
            else:
                plgpu.copy_gmem_to_smem(e_ref.at[b, h, tok], e_smem, bar)

        plgpu.barrier_wait(bar)

        frow = _iota(full, 0)
        fcol = _iota(full, 1)
        quadrant = jnp.logical_and(
            jnp.logical_and(frow >> 4 == fcol >> 4, (frow & (SUB_CHUNK - 1)) >= _HALF),
            (fcol & (SUB_CHUNK - 1)) < _HALF,
        )

        band = frow - fcol < _HALF  # pairs at most 7 tokens apart, computed exactly in the slab loop

        def stage(w, scale_m):
            m = _load(m_f32.at[w]) * scale_m
            m_bf.at[w][...] = jnp.where(band, 0.0, m).astype(_BF16)
            mq_bf.at[w][...] = jnp.where(jnp.logical_and(quadrant, jnp.logical_not(band)), m, 0.0).astype(_BF16)
            # coef[w, 0, d, i] = M[i, i - d] and coef[w, 1, d, j] = M[j + d, j] = coef[w, 0, d, j + d]
            # (entries past the chunk are never used).
            for d in range(_HALF):
                coef_smem.at[w, 0, d, pl.ds(0, c)][...] = jnp.sum(jnp.where(fcol == frow - d, m, 0.0), axis=1)
            plgpu.commit_smem()
            for d in range(_HALF):
                coef_smem.at[w, 1, d, pl.ds(0, c)][...] = _load(coef_smem.at[w, 0, d, pl.ds(d, c)], _ROW)

        # Upstream matrix w on warpgroup w: scale dAqk, dAkk (a traced scale multiplies by the same f32).
        per_warpgroup(stage, (cfg.scale, 1.0))

        plgpu.commit_smem()
        plgpu.barrier_arrive(sync)
        plgpu.barrier_wait(sync)

        rows = _iota(shape, 0)
        blocks = rows >> 4
        offset = rows & (SUB_CHUNK - 1)
        second = offset >= _HALF
        data = pl.ds(0, _KC)
        ones = plgpu.layout_cast(jnp.ones(shape, _F32), _WG)

        def run_slabs(w, first_slab):
            x_w, x2_w, y_w = x_smem.at[w], x2_smem.at[w], y_smem.at[w]
            kpad_w, epad_w, apad_w = kpad.at[w], epad.at[w], apad.at[w]

            def matrix_pass(mat, a, kf, to_end, past, half):
                """``(da, db)`` of upstream matrix ``mat`` on this slab (``a`` = ``q`` or ``kr``).

                Every pair term ``x_ij = M_ij a_i b_j e^{G_i - G_j}`` enters ``dG_i`` through the row
                direction and ``-dG_j`` through the column direction, and the reverse cumulative sum
                of ``dG`` relies on the two cancelling. The tensor cores round the decayed operands of
                the two directions differently (by up to 2^-9), and the gate's chain rule multiplies
                that residual by the cumulative log-decay. So every pair at most 7 tokens apart (the
                band, which carries nearly all of ``dG`` under strong decays) is computed exactly in
                f32 below, the same way in both directions; the tensor cores only see the pairs 8 or
                more tokens apart, whose decay is at least ``e^{8 g}``.
                """
                for cb in range(_NUM_SUB - 1):
                    y_w.at[cb][...] = (a * past[cb]).astype(_BF16)
                y_w.at[_NUM_SUB - 1][...] = jnp.where(second, a * half, 0.0).astype(_BF16)
                apad_w.at[pl.ds(0, c), data][...] = a.astype(_BF16)
                plgpu.commit_smem()
                m_w, mq_w = m_bf.at[mat], mq_bf.at[mat]
                blk = [pl.ds(SUB_CHUNK * cb, SUB_CHUNK) for cb in range(_NUM_SUB - 1)]
                # Row direction: da_i = sum_J e^{G_i - G_end(J)} (M_iJ X_J), then the quadrants.
                prods = _matmuls(cfg, shape, [(m_w.at[:, blk[cb]], x_w.at[blk[cb]]) for cb in range(_NUM_SUB - 1)]
                                 + [(mq_w, x2_w)])
                da = None
                for cb in range(_NUM_SUB - 1):
                    term = past[cb] * prods[cb]
                    da = term if da is None else da + term
                da = da + jnp.where(second, prods[-1] * half, 0.0)
                # Column direction: db_j = e^{G_end(J) - G_j} sum_i M_ij a_i e^{G_i - G_end(J)}, quadrants.
                prods = _matmuls(cfg, shape, [(m_w.T, y_w.at[cb]) for cb in range(_NUM_SUB - 1)]
                                 + [(mq_w.T, y_w.at[_NUM_SUB - 1])])
                db = None
                for cb in range(_NUM_SUB - 1):
                    term = jnp.where(blocks == cb, prods[cb], 0.0)
                    db = term if db is None else db + term
                db = db * to_end + jnp.where(second, 0.0, prods[-1] * half)
                # Exact band: running products of e^{g_t}, row- and column-shifted.
                row_coef, col_coef = coef_smem.at[mat, 0], coef_smem.at[mat, 1]
                on_diag = _rows(_load(row_coef.at[0, pl.ds(0, c)], _ROW), shape)
                da = da + on_diag * kf
                db = db + on_diag * a

                def offset_step(d, carry):
                    back, fwd, da, db = carry
                    back = back * _load(epad_w.at[pl.ds(_SHIFT + 1 - d, c), data])
                    fwd = fwd * _load(epad_w.at[pl.ds(_SHIFT + d, c), data])
                    b_prev = _load(kpad_w.at[pl.ds(_SHIFT - d, c), data]).astype(_F32)
                    a_next = _load(apad_w.at[pl.ds(d, c), data]).astype(_F32)
                    da = da + jnp.where(rows >= d, _rows(_load(row_coef.at[d, pl.ds(0, c)], _ROW), shape)
                                        * b_prev * back, 0.0)
                    db = db + jnp.where(rows + d <= c - 1, _rows(_load(col_coef.at[d, pl.ds(0, c)], _ROW), shape)
                                        * a_next * fwd, 0.0)
                    return back, fwd, da, db

                _, _, da, db = lax.fori_loop(1, _HALF, offset_step, (ones, ones, da, db))
                return da, db

            def slab_step(slab, dbeta):
                cols = pl.ds(pl.multiple_of(slab * _KC, _KC), _KC)
                g = _load(g_smem.at[:, cols])
                # q, k and kr as bf16 values (exact for bf16 inputs): the row direction multiplies by
                # them in f32 and the column direction reads them back from the bf16 shift buffers.
                kf = _bf16_values(_load(k_smem.at[:, cols]))
                erase_v = _load(e_smem.at[:, cols]).astype(_F32) if cfg.channel_gates else _rows(
                    _load(e_smem, _ROW), shape)
                ends = [_load(marks_smem.at[2 * blk + 1, cols], _COL) for blk in range(_NUM_SUB)]
                mids = [_load(marks_smem.at[2 * blk, cols], _COL) for blk in range(_NUM_SUB)]
                to_end = jnp.exp(_per_block(blocks, ends, shape) - g)  # e^{G_end(block) - G}
                # past[J] = e^{G - G_end(J)} on the rows of the blocks after J (zero elsewhere).
                past = [_masked_exp(g - _cols(ends[cb], shape), blocks > cb) for cb in range(_NUM_SUB - 1)]
                mid = _per_block(blocks, mids, shape)
                half = jnp.exp(jnp.where(second, g - mid, mid - g))

                plgpu.commit_smem()  # every warp is done with the previous slab's buffers
                # Shared by both matrices: keys decayed to the end of their block, keys decayed to
                # the middle of their block, and the per-token decays e^{g_t} = e^{G_t - G_{t-1}}.
                x_w[...] = (kf * to_end).astype(_BF16)
                x2_w[...] = jnp.where(second, 0.0, kf * half).astype(_BF16)
                kpad_w.at[pl.ds(_SHIFT, c), data][...] = kf.astype(_BF16)
                epad_w.at[pl.ds(_SHIFT, c), data][...] = g
                g_prev = _load(epad_w.at[pl.ds(_SHIFT - 1, c), data])
                epad_w.at[pl.ds(_SHIFT, c), data][...] = jnp.exp(g - g_prev)

                qf = _bf16_values(_load(q_smem.at[:, cols]))
                da_q, db_q = matrix_pass(0, qf, kf, to_end, past, half)
                dq_ref.at[b, tok, h, cols][...] = (_load(dqp_ref.at[b, tok, h, cols]) + da_q).astype(dq_ref.dtype)
                dg = qf * da_q - kf * db_q
                kr = _bf16_values(kf * erase_v)
                plgpu.commit_smem()  # the first pass's operand buffers may be overwritten
                da_k, db_k = matrix_pass(1, kr, kf, to_end, past, half)
                dk = _load(dkp_ref.at[b, tok, h, cols]) + db_q + erase_v * da_k + db_k
                dk_ref.at[b, tok, h, cols][...] = dk.astype(dk_ref.dtype)
                dg_ref.at[b, tok, h, cols][...] = _load(dgp_ref.at[b, tok, h, cols]) + dg + kr * da_k - kf * db_k
                if cfg.channel_gates:
                    d_erase = _load(gp_ref.at[b, tok, h, cols]) + kf * da_k
                    gate_ref.at[b, tok, h, cols][...] = d_erase.astype(gate_ref.dtype)
                    return dbeta
                return dbeta + jnp.sum(kf * da_k, axis=1)

            init = plgpu.layout_cast(jnp.zeros((c,), _F32), _ROW)
            dbeta = lax.fori_loop(first_slab, first_slab + half_count, slab_step, init)
            if not cfg.channel_gates:
                dbeta_smem.at[w][...] = dbeta

        # Warpgroup w takes the key slabs [w count/2, (w + 1) count/2).
        per_warpgroup(run_slabs, (0, half_count))

        if not cfg.channel_gates:
            plgpu.barrier_arrive(sync)
            plgpu.barrier_wait(sync)

            @pl.when(wg == 0)
            def _():
                total = _load(dbeta_smem.at[0], _ROW) + _load(dbeta_smem.at[1], _ROW)
                for p in range(_parts(cfg)):
                    total = total + _load(gp_ref.at[b, tok, h, p], _ROW)
                gate_ref.at[b, tok, h][...] = total

        finish_together(finish)  # no warpgroup exits while the other may still issue wgmmas

    e_scratch = _slab_smem((c, kd), erase.dtype) if cfg.channel_gates else plgpu.SMEM((c,), _F32)
    scratch = (
        _slab_smem((c, kd), q.dtype),
        _slab_smem((c, kd), k.dtype),
        _slab_smem((c, kd), _F32),
        plgpu.SMEM((_MARKS, kd), _F32),
        e_scratch,
        _tile16((2, c, c)),  # bf16 scale dAqk and dAkk without the band
        _tile16((2, c, c)),  # the same, only the lower-left quadrants of the diagonal blocks
        plgpu.SMEM((2, 2, _HALF, c + _HALF), _F32),  # exact band coefficients of each matrix
        _tile16((2, c, _KC)),
        _tile16((2, c, _KC)),
        plgpu.RefUnion(_tile16((2, _NUM_SUB, c, _KC)), plgpu.SMEM((2, c, c), _F32)),  # y tiles / f32 dAqk, dAkk
        # Row-shift buffers (keys, per-token decays, the current pass's left factor); the padded
        # pitch keeps the shifted loads free of bank conflicts.
        plgpu.SMEM((2, _SHIFT + c, _KC + 8), _BF16),
        plgpu.SMEM((2, _SHIFT + c + _HALF, _KC + 8), _F32),
        plgpu.SMEM((2, c + _HALF, _KC + 8), _BF16),
        plgpu.SMEM((2, c), _F32),
        plgpu.Barrier(num_arrivals=7),
        plgpu.Barrier(num_arrivals=2),
        plgpu.Barrier(num_arrivals=2),  # both warpgroups are done (finish_together)
    )
    kern = _kernel(
        cfg, "vg_bwd_intra", body, out_type=tuple(out), grid=(cfg.num_chunks, cfg.heads, cfg.batch),
        grid_names=("n", "h", "b"), num_threads=2, thread_name="wg", scratch_types=scratch,
    )
    return kern(q, k, g_cum, g_marks, erase, daqk, dakk, dq_part, dk_part, dg_part, gate_part)
