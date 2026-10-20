# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py,
# flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py,
# flash_qla/ops/gated_delta_rule/chunk/hopper/cp_fwd.py (get_warmup_chunks, as warmup_windows)
# Adapted from flash-linear-attention (MIT), fla/ops/common/chunk_delta_h.py, fla/ops/common/chunk_o.py
"""Document-aware map passes of the scalar-gate sequence split (Gated DeltaNet, DeltaNet, simple GLA).

The exact sequence split of :mod:`scalar_gate` cuts every sequence into ``cp``
pieces and stitches them with affine piece maps ``S_end = M S_start + E``:
one augmented forward pass per piece gives ``[M | E]``, and one reverse pass
per piece from a zero end gradient gives the backward's local term ``D``.
Both passes used to walk every chunk of every piece. Two facts make most of
that work unnecessary:

* A chunk whose last token is not in the run that entered the chunk (a
  document starts in it, ``cin`` of its last token is 0; a "kill" chunk)
  does not let the incoming state through: everything it passes on is
  computed from its own last run. So a piece's map, walked from the piece's
  last kill chunk instead of its first chunk, is bitwise the same (``M = 0``
  and the same ``E``), and the reverse pass from a zero end gradient gives the
  same gradient at the piece start when it starts at the piece's first kill
  chunk instead of its last chunk.
* The forward scan never reads the last piece's map (the backward reads it
  only to carry a final-state gradient), and the backward scan never reads the
  first piece's reverse term.

Each CTA therefore reads its piece's live range from a small int32 table
(:func:`walk_tables`, ``[B * cp, 2]``: first chunk within the piece, number of
chunks) and walks only those chunks; unused pieces walk none. With packed
documents of ``L`` chunks a map pass walks at most ``L`` chunks per piece
instead of ``T / (64 cp)``; without documents only the unused pieces are
skipped. The delta rule's identity blocks (the rows of ``M``) of a piece that
has a kill chunk walk nothing at all: the walk from the kill chunk zeroes them
and they stay zero, so they hold ``M = 0`` directly, and half of the forward
map pass's CTAs are idle with packed documents. The results equal the full
walks exactly, so the split stays exact.

The kernels are :func:`gated_delta_rule_bwd.state_fwd_wy` (augmented mode) and
:func:`gated_delta_rule_bwd.state_bwd_wy` (no stores, ``dh0`` only) with the
dynamic walk, and for simple GLA (whose maps are the piece decays, computed
in XLA) :func:`scalar_gate_kernels.state_fwd` / ``state_bwd`` reduced to the
local end state and start gradient; everything else is unchanged.
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
    num_sms,
    same_run,
    smem,
)
from jla.ops.mosaic_gpu.scalar_gate_fused import _stages

Array = jax.Array
F32 = jnp.float32
C = CHUNK


def fresh_next(seg: Segments | None, batch: int, padded: int, cp: int) -> Array:
    """``[B, cp]`` bool: the next piece starts a document at its first token (never for the last piece).

    That piece reads nothing of the state it is handed (its first chunk drops
    the incoming state at every token), so this piece's forward map is not
    needed: the forward map pass walks nothing for it and holds ``M = 0, E = 0``.
    """
    if seg is None:
        return jnp.zeros((batch, cp), bool)
    starts = seg.cin[:, ::padded // cp].reshape(batch, cp) == 0  # cin of every piece's first token
    return jnp.concatenate([starts[:, 1:], jnp.zeros((batch, 1), bool)], axis=1)


def walk_tables(seg: Segments | None, batch: int, padded: int, cp: int, *,
                need_last: bool = True, skip_fresh: bool = False) -> tuple[Array, Array]:
    """Live chunk ranges of the two map passes, int32 ``[B * cp, 2]`` each: (first chunk in the piece, count).

    Forward: from the piece's last kill chunk to its end (the whole piece if it
    has none); the last piece walks nothing unless ``need_last`` (its map
    carries a final-state gradient back), and with ``skip_fresh`` neither does a
    piece whose successor starts a document at its first token
    (:func:`fresh_next`). Backward (walked in reverse): from the piece start to
    its first kill chunk (the whole piece if it has none); the first piece walks
    nothing.
    """
    ncs = padded // C // cp
    idx = jnp.arange(ncs, dtype=jnp.int32)
    piece = jnp.arange(cp, dtype=jnp.int32)
    if seg is None:
        kill = jnp.zeros((batch, cp, ncs), bool)
    else:
        kill = (seg.cin[:, C - 1::C] == 0).reshape(batch, cp, ncs)
    last = jnp.max(jnp.where(kill, idx, 0), axis=-1)  # [B, cp]
    first = jnp.min(jnp.where(kill, idx, ncs - 1), axis=-1)
    skip_last = (piece == cp - 1) & (not need_last)
    if skip_fresh:
        skip_last = skip_last | fresh_next(seg, batch, padded, cp)
    fwd = jnp.stack([last, jnp.where(skip_last, 0, ncs - last)], axis=-1)
    bwd = jnp.stack([jnp.zeros_like(first), jnp.where(piece == 0, 0, first + 1)], axis=-1)
    return fwd.reshape(batch * cp, 2).astype(jnp.int32), bwd.reshape(batch * cp, 2).astype(jnp.int32)


def warmup_windows(g2: Array, cp: int, threshold: float, *, trailing: bool) -> Array:
    """Opt-in approximation of the split (FlashQLA's intra-card CP warm-up), ``[B * cp, H]`` int32.

    For every piece and head: the number of chunks, counted from the piece end
    (``trailing``, the forward) or from its start (the backward), after which
    their total log2 decay (``g2`` chunk ends) first falls below ``threshold``;
    0 where it never does, and that piece is walked exactly. With unit-norm keys
    and ``0 <= beta <= 2`` every token's transition has norm at most its decay,
    so a state (or state gradient) carried through such a window shrinks to at
    most ``2^threshold`` of its size: the map passes walk only the window and
    drop the rest (``M = 0``). FlashQLA's default threshold is ``-10`` in natural
    log units, ``-10 / ln 2`` here.
    """
    batch, heads, padded = g2.shape
    gc = g2[..., C - 1::C].reshape(batch, heads, cp, padded // C // cp)
    if trailing:
        gc = gc[..., ::-1]
    reached = jnp.cumsum(gc, axis=-1) < threshold
    win = jnp.where(reached.any(axis=-1), jnp.argmax(reached, axis=-1) + 1, 0)
    return jnp.swapaxes(win, 1, 2).reshape(batch * cp, heads).astype(jnp.int32)


def fwd_walk_table(seg: Segments | None, g2: Array, cp: int, *, need_last: bool = True, need_first: bool = True,
                   warmup: float | None = None, skip_fresh: bool = True) -> Array:
    """The forward map pass's walk table, int32 ``[B * cp, H, 4]``.

    Columns: first live chunk within the piece and the chunk count of the value
    blocks (``E``, :func:`walk_tables` with ``skip_fresh``), then the chunk count
    and initial identity flag of the identity blocks (``M``). A piece with a kill
    chunk has ``M = 0`` exactly (the walk from the kill chunk zeroes the identity
    blocks and they stay zero), and so does a piece whose successor starts a
    document at its first token (:func:`fresh_next`, nothing reads its map): their
    identity blocks walk nothing and hold zeros. Without ``need_first`` the first
    piece's ``M`` is not computed (zeros). The same table for every head, except
    with the opt-in warm-up (:func:`warmup_windows`).
    """
    batch, heads, padded = g2.shape
    walk, _ = walk_tables(seg, batch, padded, cp, need_last=need_last, skip_fresh=skip_fresh)
    ncs = padded // C // cp
    if seg is None:
        id_count, id_init = walk[:, 1], jnp.ones_like(walk[:, 1])
    else:
        killed = (seg.cin[:, C - 1::C] == 0).reshape(batch * cp, ncs).any(axis=-1)
        if skip_fresh:
            killed = killed | fresh_next(seg, batch, padded, cp).reshape(batch * cp)
        id_count, id_init = jnp.where(killed, 0, walk[:, 1]), jnp.where(killed, 0, 1)
    if not need_first:
        first = jnp.arange(batch * cp) % cp == 0
        id_count, id_init = jnp.where(first, 0, id_count), jnp.where(first, 0, id_init)
    walk = jnp.concatenate([walk, jnp.stack([id_count, id_init], axis=-1)], axis=-1).astype(jnp.int32)
    walk = jnp.broadcast_to(walk[:, None], (batch * cp, heads, 4))
    if warmup is not None:  # approximate: FlashQLA's gate-driven warm-up (warmup_windows)
        win = warmup_windows(g2, cp, warmup, trailing=True)  # [B * cp, H], 0 where the decay stays above it
        first_w = jnp.where(win > 0, jnp.maximum(walk[..., 0], ncs - win), walk[..., 0])
        count_w = jnp.where(walk[..., 1] > 0, ncs - first_w, 0)
        id_off = win > 0  # M taken as zero
        walk = jnp.stack([first_w, count_w, jnp.where(id_off, 0, walk[..., 2]), jnp.where(id_off, 0, walk[..., 3])],
                         axis=-1).astype(jnp.int32)
    return walk


def zero_maps(table: Array) -> Array:
    """``[B * cp, H]`` int32 from :func:`fwd_walk_table`: 1 where the piece's ``M`` is exactly zero."""
    return ((table[..., 2] == 0) & (table[..., 3] == 0)).astype(jnp.int32)


def map_fwd_wy(
    k: Array,
    v: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    *,
    seg: Segments | None = None,
    cp: int,
    need_last: bool = True,
    need_first: bool = True,
    warmup: float | None = None,
    skip_fresh: bool = True,
    bv: int = 64,
    interpret=None,
) -> Array:
    """Piece maps ``[M^T | E^T]`` (``[B * cp, H, K + V, K]`` float32) over each piece's live chunks.

    Same result as ``gated_delta_rule_bwd.state_fwd_wy(..., aug=True,
    out_final=True, store_states=False)``, except that without ``need_last``
    (no final-state gradient) the last piece walks nothing and holds
    ``[I | 0]``; without ``need_first`` (no initial state, whose map is the
    only reader of the first piece's ``M``) the first piece's ``M`` is not
    computed and holds zeros. With ``skip_fresh`` (the default) a piece whose
    successor starts a document at its first token walks nothing and holds
    ``[0 | 0]``: nothing reads its map (:func:`fresh_next`). With ``warmup`` (a
    log2-decay threshold, opt-in, approximate) a piece whose trailing chunks decay
    below it walks only those chunks for ``E`` and takes ``M = 0``
    (:func:`warmup_windows`). Arguments as there.
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    n_id = key_dim // bv  # identity blocks
    nv = value_dim // bv + n_id
    walk = fwd_walk_table(seg, g2, cp, need_last=need_last, need_first=need_first, warmup=warmup,
                          skip_fresh=skip_fresh)
    ncs = padded // C // cp
    pieces = cp if need_last else cp - 1  # pieces that walk; the identity blocks also skip the first without need_first
    live = pieces * (nv - n_id) + (pieces - (0 if need_first else 1)) * n_id
    # With packed documents (and the warm-up's windows) many blocks walk little or nothing, so the single
    # blocks fit one wave: with the warm-up at (1, 16384, 16), 68 us single against 99 paired with half the
    # heads long-memory, though 131 against 101 when every piece falls back to the exact walk.
    dynamic = seg is not None or warmup is not None
    if PAIRED_MAPS and n_id % 2 == 0 and nv % 2 == 0 and _use_pairs(batch * heads * live, dynamic):
        return _map_fwd_wy_paired(k, v, t0, g2, beta, walk, seg=seg, cp=cp, bv=bv, interpret=interpret)
    dtype = k.dtype
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = C * key_dim * isz + C * bv * isz + C * C * isz + n_vec * C * 4
    stages = _stages(per_stage, bv * key_dim * 4 + 4096 + n_vec * 1024)
    n_copies = 3 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch * cp, heads, nv * bv, key_dim), F32)]
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
        done=plgpu.Barrier(num_arrivals=3),
        state_smem=plgpu.SMEM((bv, key_dim), F32),
    )
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, k_s, v_s, t_s, g_state, g_rows, g_cols, b_cols, in_bar, in_free, ag_ready, done, state_smem,
             tail_c=None, cin_c=None, run_r=None, run_c=None):
        refs = list(refs)
        k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(5))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        walk_ref, ht_ref = refs
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b = lax.div(bp, cp)
        c0 = lax.rem(bp, cp) * ncs + walk_ref[bp, hh, 0]  # first live chunk
        count = jnp.where(iv < n_id, walk_ref[bp, hh, 2], walk_ref[bp, hh, 1])
        hk = lax.div(hh, group)
        v0 = jnp.maximum(iv - n_id, 0) * bv  # the identity blocks read (and ignore) the first value block
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

            for n in range(stages):
                @pl.when(count > n)
                def _(n=n):
                    issue(n, n)

            @pl.loop(stages, jnp.maximum(count, stages))
            def _refill(n):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(n, slot)

        @pl.when(wg == 1)
        def _ag():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, count)
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
            shape = (bv, key_dim)
            eye = iota(shape, 1) == iota(shape, 0) + iv * bv  # rows iv * bv ... of the identity
            s0 = jnp.where(eye, 1.0, 0.0).astype(F32) * walk_ref[bp, hh, 3].astype(F32)
            v_keep = jnp.where(iv >= n_id, 1.0, 0.0).astype(F32)

            def step(n, s_t):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                s_lo = s_t.astype(dtype)
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
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32) * v_keep
                w_t = (v_t - u_t * bcast_cols(key_decay, (bv, C))).astype(dtype)
                plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = matmul(w_t, t_s.at[slot].T, (bv, C))
                vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
                s_new = matmul(vd_t, k_s.at[slot], (bv, key_dim), init=s_t * s_decay)
                plgpu.barrier_arrive(in_free.at[slot])
                return s_new

            s_t = lax.fori_loop(0, count, step, s0)
            state_smem[...] = s_t
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(iv * bv, bv)])
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + [walk]
    (ht,) = build_kernel(
        body, name="gdr_map_fwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args)
    return ht


# The forward map pass carries two 64-row blocks of [M^T | E^T] per CTA (K, V even multiples of 64):
# half the CTAs, so the 4 (b, h) blocks of every piece fit one wave at small B * H.
PAIRED_MAPS = True
# Paired map pass: the memory warpgroup forms Ag one chunk ahead (registers 200 / 200 / 96), so both carrying
# warpgroups only walk; False lets WG1 form it before its own block (232 / 232 / 40).
AG_IN_MEMORY_WG = True
# A CTA step of the paired kernel took 1.64 times a single block's on an H100 ((1, 16384, 16), cp = 4: 105 us
# for 80 CTAs against 130 us for 160 single-block CTAs in two waves). The pairs pay only when they save a wave;
# with packed documents the identity blocks mostly walk nothing, so the single blocks fit one wave anyway
# (41 us against 61 us paired at 2048-token documents) and the paired kernel is not used.
PAIRED_STEP = 1.64


def _use_pairs(blocks: int, docs: bool) -> bool:
    """Whether ``blocks`` single-block CTAs (one per live row block) take longer than their pairs."""
    if docs:
        return False
    sms = num_sms()
    return -(-(blocks // 2) // sms) * PAIRED_STEP < -(-blocks // sms)


def _map_fwd_wy_paired(k, v, t0, g2, beta, walk, *, seg, cp, bv, interpret):
    """:func:`map_fwd_wy` with two row blocks per CTA (``walk``: its ``[B * cp, 4]`` table).

    WG0 carries block ``2 j`` and WG1 block ``2 j + 1``; WG2 streams ``k``,
    both blocks' values, ``T0`` and the gate vectors and forms ``Ag`` one chunk
    ahead (with ``AG_IN_MEMORY_WG``; otherwise WG1 forms it before its block).
    The two blocks of a CTA are both identity blocks or both
    value blocks, so they walk the same chunks. Per block the arithmetic is
    that of :func:`map_fwd_wy`, so the results are the same.
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = v.shape[-1]
    group = heads // kv_heads
    n_id = key_dim // bv
    nv = value_dim // bv + n_id
    ncs = padded // C // cp
    dtype = k.dtype
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = C * key_dim * isz + 2 * C * bv * isz + C * C * isz + n_vec * C * 4
    stages = _stages(per_stage, 2 * bv * key_dim * 4 + 4096 + n_vec * 1024)
    n_copies = 4 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch * cp, heads, nv * bv, key_dim), F32)]
    vecs = plgpu.SMEM((stages, C), F32)
    scratch = dict(
        k_s=smem((stages, C, key_dim), dtype),
        va_s=smem((stages, C, bv), dtype),
        vb_s=smem((stages, C, bv), dtype),
        t_s=smem((stages, C, C), dtype),  # T0, then Ag (written by WG1)
        g_state=vecs,
        g_rows=vecs,
        g_cols=vecs,
        b_cols=vecs,
        in_bar=plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages),
        in_free=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
        ag_ready=plgpu.Barrier(num_barriers=stages),
        done=plgpu.Barrier(num_arrivals=3),
        state_smem=plgpu.SMEM((2, bv, key_dim), F32),
    )
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, k_s, va_s, vb_s, t_s, g_state, g_rows, g_cols, b_cols, in_bar, in_free, ag_ready, done,
             state_smem, tail_c=None, cin_c=None, run_r=None, run_c=None):
        refs = list(refs)
        k_ref, v_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(5))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        walk_ref, ht_ref = refs
        bp, hh, ip = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b = lax.div(bp, cp)
        c0 = lax.rem(bp, cp) * ncs + walk_ref[bp, hh, 0]  # first live chunk
        blocks = (2 * ip, 2 * ip + 1)
        count = jnp.where(blocks[0] < n_id, walk_ref[bp, hh, 2], walk_ref[bp, hh, 1])
        hk = lax.div(hh, group)
        v0s = tuple(jnp.maximum(x - n_id, 0) * bv for x in blocks)  # identity blocks read the first value block
        wg = lax.axis_index("wg")

        def form_ag(slot):  # Ag = T0 * exp(G_i - G_j) * beta_j in place of T0
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

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(96 if AG_IN_MEMORY_WG else 40, action="decrease")

            def issue(n, slot):
                rows = pl.ds((c0 + n) * C, C)
                bar = in_bar.at[slot]
                plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0s[0], bv)], va_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh, pl.ds(v0s[1], bv)], vb_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(t_ref.at[b, hh, rows], t_s.at[slot], bar)
                for g_buf in (g_state, g_rows, g_cols):
                    plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_buf.at[slot], bar)
                plgpu.copy_gmem_to_smem(b_ref.at[b, hh, rows], b_cols.at[slot], bar)
                if segs:
                    for src, dst in ((tail_ref, tail_c), (cin_ref, cin_c), (run_ref, run_r), (run_ref, run_c)):
                        plgpu.copy_gmem_to_smem(src.at[b, rows], dst.at[slot], bar)

            for n in range(stages):
                @pl.when(count > n)
                def _(n=n):
                    issue(n, n)

            if AG_IN_MEMORY_WG:
                @pl.when(count > 0)
                def _():
                    plgpu.barrier_wait(in_bar.at[0])
                    form_ag(0)

                @pl.loop(0, count)
                def _walk(n):
                    @pl.when(n + 1 < count)
                    def _():  # the next chunk's Ag before blocking on this chunk's release
                        nxt = lax.rem(n + 1, stages)
                        plgpu.barrier_wait(in_bar.at[nxt])
                        form_ag(nxt)

                    @pl.when(n + stages < count)
                    def _():
                        slot = lax.rem(n, stages)
                        plgpu.barrier_wait(in_free.at[slot])
                        issue(n + stages, slot)
            else:
                @pl.loop(stages, jnp.maximum(count, stages))
                def _refill(n):
                    slot = lax.rem(n, stages)
                    plgpu.barrier_wait(in_free.at[slot])
                    issue(n, slot)

        def carry(which, v_s, own_ag):
            """Walk one row block: ``which`` 0 (WG0) or 1 (WG1, which forms ``Ag`` if ``own_ag``)."""
            blk = blocks[which]
            shape = (bv, key_dim)
            eye = iota(shape, 1) == iota(shape, 0) + blk * bv  # rows blk * bv ... of the identity
            s0 = jnp.where(eye, 1.0, 0.0).astype(F32) * walk_ref[bp, hh, 3].astype(F32)
            v_keep = jnp.where(blk >= n_id, 1.0, 0.0).astype(F32)

            def step(n, s_t):
                slot = lax.rem(n, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                if own_ag:
                    form_ag(slot)
                s_lo = s_t.astype(dtype)
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
                v_t = plgpu.load(v_s.at[slot].T, layout=WGMMA).astype(F32) * v_keep
                w_t = (v_t - u_t * bcast_cols(key_decay, (bv, C))).astype(dtype)
                if not own_ag:
                    plgpu.barrier_wait(ag_ready.at[slot])
                vn_t = matmul(w_t, t_s.at[slot].T, (bv, C))
                vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
                s_new = matmul(vd_t, k_s.at[slot], (bv, key_dim), init=s_t * s_decay)
                plgpu.barrier_arrive(in_free.at[slot])
                return s_new

            s_t = lax.fori_loop(0, count, step, s0)
            state_smem.at[which][...] = s_t
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(state_smem.at[which], ht_ref.at[bp, hh, pl.ds(blk * bv, bv)])
            plgpu.wait_smem_to_gmem(0)

        regs = 200 if AG_IN_MEMORY_WG else 232

        @pl.when(wg == 1)
        def _second():
            plgpu.set_max_registers(regs, action="increase")
            carry(1, vb_s, not AG_IN_MEMORY_WG)

        @pl.when(wg == 0)
        def _first():
            plgpu.set_max_registers(regs, action="increase")
            carry(0, va_s, False)

        finish_together(done)

    args = [k, v, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + [walk]
    (ht,) = build_kernel(
        body, name="gdr_map_fwd2", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv // 2),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args)
    return ht


def map_bwd_wy(
    q: Array,
    k: Array,
    do: Array,
    t0: Array,
    g2: Array,
    beta: Array,
    *,
    scale: float,
    seg: Segments | None = None,
    cp: int,
    warmup: float | None = None,
    bv: int = 64,
    interpret=None,
) -> Array:
    """Gradient at each piece start from a zero end gradient (``[B * cp, H, V, K]`` float32, transposed).

    Same result as ``gated_delta_rule_bwd.state_bwd_wy(..., dht=None,
    want_dh0=True, store_grads=False)`` for every piece but the first (whose
    term the scan never reads; it holds zeros). With ``warmup`` (opt-in,
    approximate) a piece whose leading chunks decay below it walks only those
    (:func:`warmup_windows`). Arguments as there.
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    nv = value_dim // bv
    _, walk = walk_tables(seg, batch, padded, cp)
    walk = jnp.broadcast_to(walk[:, None], (batch * cp, heads, 2))  # per head
    if warmup is not None:  # approximate: only the leading chunks that decay below the threshold
        win = warmup_windows(g2, cp, warmup, trailing=False)
        walk = walk.at[..., 1].set(jnp.where(win > 0, jnp.minimum(walk[..., 1], win), walk[..., 1]))
    ncs = padded // C // cp
    dtype = q.dtype
    segs = seg is not None
    isz = jnp.dtype(dtype).itemsize
    n_vec = 4 + 4 * segs  # g_state, g_rows, g_cols, b_cols (+ tail_c, cin_c, run_r, run_c)
    per_stage = 2 * C * key_dim * isz + 2 * C * bv * isz + C * C * isz + n_vec * C * 4
    stages = _stages(per_stage, bv * key_dim * 4 + 4096 + n_vec * 1024)
    n_copies = 4 + n_vec

    out_type = [jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32)]
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
        done=plgpu.Barrier(num_arrivals=3),
        state_smem=plgpu.SMEM((bv, key_dim), F32),
    )
    if segs:
        scratch.update(tail_c=vecs, cin_c=vecs, run_r=vecs, run_c=vecs)

    def body(*refs, q_s, k_s, do_s, t_s, dvl_s, g_state, g_rows, g_cols, b_cols, in_bar, in_free, dvl_ready, done,
             state_smem, tail_c=None, cin_c=None, run_r=None, run_c=None):
        refs = list(refs)
        q_ref, k_ref, do_ref, t_ref, g_ref, b_ref = (refs.pop(0) for _ in range(6))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        walk_ref, dh0_ref = refs
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b = lax.div(bp, cp)
        c0 = lax.rem(bp, cp) * ncs + walk_ref[bp, hh, 0]
        count = walk_ref[bp, hh, 1]
        last = c0 + count - 1  # the reverse walk starts here
        hk = lax.div(hh, group)
        v0 = iv * bv
        wg = lax.axis_index("wg")
        sq = (C, C)

        @pl.when(wg == 2)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def issue(i, slot):  # i: step of the reverse walk
                rows = pl.ds((last - i) * C, C)
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

            for i in range(stages):
                @pl.when(count > i)
                def _(i=i):
                    issue(i, i)

            @pl.loop(stages, jnp.maximum(count, stages))
            def _refill(i):
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_free.at[slot])
                issue(i, slot)

        @pl.when(wg == 1)
        def _intra():
            plgpu.set_max_registers(232, action="increase")

            @pl.loop(0, count)
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
            ds0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)
            st = (bv, C)

            def step(i, ds_t):
                slot = lax.rem(i, stages)
                plgpu.barrier_wait(in_bar.at[slot])
                ds_lo = ds_t.astype(dtype)
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
                do_t = plgpu.load(do_s.at[slot].T, layout=WGMMA).astype(F32)
                do_t = (do_t * bcast_cols(gam * scale, st)).astype(dtype)
                dwn = (dw_t * bcast_cols(-gam, st)).astype(dtype)

                def update(acc):
                    plgpu.wgmma(acc, do_t, q_s.at[slot])
                    plgpu.wgmma(acc, dwn, k_s.at[slot])

                ds_new = lax.optimization_barrier(pl.run_state(update)(plgpu.ACC.init(ds_t * s_decay)))
                plgpu.barrier_arrive(in_free.at[slot])
                return ds_new

            ds_t = lax.fori_loop(0, count, step, ds0)
            state_smem[...] = ds_t
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(state_smem, dh0_ref.at[bp, hh, pl.ds(v0, bv)])
            plgpu.wait_smem_to_gmem(0)

        finish_together(done)

    args = [q, k, do, t0, g2, beta] + ([seg.run, seg.cin, seg.tail] if segs else []) + [walk]
    (dh0,) = build_kernel(
        body, name="gdr_map_bwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret, num_threads=3, thread_name="wg",
    )(*args)
    return dh0


def map_fwd_gla(
    k: Array,
    u: Array,
    g2: Array,
    *,
    seg: Segments | None = None,
    cp: int,
    need_last: bool = True,
    bv: int = 64,
    interpret=None,
) -> Array:
    """simple GLA: each piece's end state from a zero start (``[B * cp, H, V, K]`` float32, transposed).

    Same result as ``scalar_gate_kernels.state_fwd(k, None, u, g2, None,
    u_head_major=False, out_final=True, store_states=False)`` over each piece's
    live chunks (:func:`walk_tables`); without ``need_last`` the last piece
    walks nothing and holds zeros. ``u`` is in the user layout ``[B, Tp, H, V]``.
    """
    batch, padded, kv_heads, key_dim = k.shape
    heads = g2.shape[1]
    value_dim = u.shape[-1]
    group = heads // kv_heads
    nv = value_dim // bv
    walk, _ = walk_tables(seg, batch, padded, cp, need_last=need_last)
    ncs = padded // C // cp
    dtype = k.dtype
    segs = seg is not None
    out_type = [jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32)]
    vec2 = plgpu.SMEM((2, C), F32)
    scratch = dict(
        k_smem=smem((2, C, key_dim), dtype),
        u_smem=smem((2, C, bv), dtype),
        g_smem=vec2,
        state_smem=plgpu.SMEM((bv, key_dim), F32),
        in_bar=plgpu.Barrier(num_arrivals=3 + 2 * segs, num_barriers=2),
    )
    if segs:
        scratch.update(tail_c=vec2, cin_s=vec2)

    def body(*refs, k_smem, u_smem, g_smem, state_smem, in_bar, tail_c=None, cin_s=None):
        refs = list(refs)
        k_ref, u_ref, g_ref = refs.pop(0), refs.pop(0), refs.pop(0)
        cin_ref, tail_ref = (refs.pop(0), refs.pop(0)) if segs else (None, None)
        walk_ref, ht_ref = refs
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b = lax.div(bp, cp)
        c0 = lax.rem(bp, cp) * ncs + walk_ref[bp, 0]
        count = walk_ref[bp, 1]
        hk = lax.div(hh, group)
        v0 = iv * bv

        def issue(n, slot):  # n: global chunk index
            rows = pl.ds(n * C, C)
            bar = in_bar.at[slot]
            plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(u_ref.at[b, rows, hh, pl.ds(v0, bv)], u_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem.at[slot], bar)
            if segs:
                plgpu.copy_gmem_to_smem(cin_ref.at[b, rows], cin_s.at[slot], bar)
                plgpu.copy_gmem_to_smem(tail_ref.at[b, rows], tail_c.at[slot], bar)

        @pl.when(count > 0)
        def _first():
            issue(c0, 0)

        s0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

        def step(n, s_t):
            slot = lax.rem(n, 2)
            plgpu.barrier_wait(in_bar.at[slot])

            @pl.when(n + 1 < count)
            def _prefetch():
                plgpu.commit_smem()
                issue(c0 + n + 1, 1 - slot)

            vn_t = plgpu.load(u_smem.at[slot].T, layout=WGMMA).astype(F32)
            g_last = g_smem[slot, C - 1]
            k_decay = jnp.exp2(g_last - plgpu.load(g_smem.at[slot], layout=COLS))
            s_decay = jnp.exp2(g_last)
            if segs:
                k_decay = k_decay * plgpu.load(tail_c.at[slot], layout=COLS)
                s_decay = s_decay * cin_s[slot, C - 1]
            vd_t = (vn_t * bcast_cols(k_decay, (bv, C))).astype(dtype)
            return matmul(vd_t, k_smem.at[slot], (bv, key_dim), init=s_t * s_decay)

        s_t = lax.fori_loop(0, count, step, s0)
        state_smem[...] = s_t
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(state_smem, ht_ref.at[bp, hh, pl.ds(v0, bv)])
        plgpu.wait_smem_to_gmem(0)

    args = [k, u, g2] + ([seg.cin, seg.tail] if segs else []) + [walk]
    (ht,) = build_kernel(
        body, name="sg_map_fwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret,
    )(*args)
    return ht


def map_bwd_gla(
    q: Array,
    k: Array,
    g2: Array,
    do: Array,
    *,
    scale: float,
    seg: Segments | None = None,
    cp: int,
    bv: int = 64,
    interpret=None,
) -> Array:
    """simple GLA: gradient at each piece start from a zero end gradient (``[B * cp, H, V, K]`` float32).

    Same result as ``scalar_gate_kernels.state_bwd(q, k, None, g2, do, dvl,
    None, want_dh0=True, store_grads=False)`` over each piece's live chunks
    (:func:`walk_tables`); the first piece walks nothing and holds zeros.
    Without the delta rule the state gradient needs only ``q`` and ``do``
    (``dS^T <- gamma dS^T + scale (do^T Gamma) q``), so ``dv_local`` is not read.
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    nv = value_dim // bv
    _, walk = walk_tables(seg, batch, padded, cp)
    ncs = padded // C // cp
    dtype = q.dtype
    segs = seg is not None
    out_type = [jax.ShapeDtypeStruct((batch * cp, heads, value_dim, key_dim), F32)]
    vec2 = plgpu.SMEM((2, C), F32)
    scratch = dict(
        q_smem=smem((2, C, key_dim), dtype),
        k_smem=smem((2, C, key_dim), dtype),
        do_smem=smem((2, C, bv), dtype),
        g_smem=vec2,
        state_smem=plgpu.SMEM((bv, key_dim), F32),
        in_bar=plgpu.Barrier(num_arrivals=4 + 2 * segs, num_barriers=2),
    )
    if segs:
        scratch.update(cin_c=vec2, tail_c=vec2)

    def body(*refs, q_smem, k_smem, do_smem, g_smem, state_smem, in_bar, cin_c=None, tail_c=None):
        refs = list(refs)
        q_ref, k_ref, g_ref, do_ref = (refs.pop(0) for _ in range(4))
        cin_ref, tail_ref = (refs.pop(0), refs.pop(0)) if segs else (None, None)
        walk_ref, dh0_ref = refs
        bp, hh, iv = lax.axis_index("b"), lax.axis_index("h"), lax.axis_index("v")
        b = lax.div(bp, cp)
        c0 = lax.rem(bp, cp) * ncs + walk_ref[bp, 0]
        count = walk_ref[bp, 1]
        hk = lax.div(hh, group)
        v0 = iv * bv

        def issue(n, slot):  # n: global chunk index
            rows = pl.ds(n * C, C)
            bar = in_bar.at[slot]
            plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh, pl.ds(v0, bv)], do_smem.at[slot], bar)
            if segs:
                plgpu.copy_gmem_to_smem(cin_ref.at[b, rows], cin_c.at[slot], bar)
                plgpu.copy_gmem_to_smem(tail_ref.at[b, rows], tail_c.at[slot], bar)

        @pl.when(count > 0)
        def _first():
            issue(c0 + count - 1, 0)

        ds0 = plgpu.layout_cast(jnp.zeros((bv, key_dim), F32), WGMMA)

        def step(i, ds_t):
            n = c0 + count - 1 - i  # global chunk index, walking backwards
            slot = lax.rem(i, 2)
            plgpu.barrier_wait(in_bar.at[slot])

            @pl.when(i + 1 < count)
            def _prefetch():
                plgpu.commit_smem()
                issue(n - 1, 1 - slot)

            g_last = g_smem[slot, C - 1]
            g_col = plgpu.load(g_smem.at[slot], layout=COLS)  # tokens are the columns of the [BV, C] tiles
            q_scale = jnp.exp2(g_col) * scale
            s_decay = jnp.exp2(g_last)
            if segs:
                q_scale = q_scale * plgpu.load(cin_c.at[slot], layout=COLS)
                s_decay = s_decay * cin_c[slot, C - 1]
            do_t = plgpu.load(do_smem.at[slot].T, layout=WGMMA).astype(F32)
            do_t = (do_t * bcast_cols(q_scale, (bv, C))).astype(dtype)

            def update(acc):
                plgpu.wgmma(acc, do_t, q_smem.at[slot])

            return pl.run_state(update)(plgpu.ACC.init(ds_t * s_decay))

        ds_t = lax.fori_loop(0, count, step, ds0)
        state_smem[...] = ds_t
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(state_smem, dh0_ref.at[bp, hh, pl.ds(v0, bv)])
        plgpu.wait_smem_to_gmem(0)

    args = [q, k, g2, do] + ([seg.cin, seg.tail] if segs else []) + [walk]
    (dh0,) = build_kernel(
        body, name="sg_map_bwd", out_type=out_type, scratch=scratch, grid=(batch * cp, heads, nv),
        grid_names=("b", "h", "v"), interpret=interpret,
    )(*args)
    return dh0
