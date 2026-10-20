# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), csrc/selective_scan/selective_scan_bwd_kernel.cuh
"""Backward kernels of the Mamba-1 selective scan (:mod:`jla.ops.mosaic_gpu.mamba1`), second version.

Same CTA layout as the forward (one channel per thread, the ``N`` states in registers); the CTA walks
its tiles of :data:`~jla.ops.mosaic_gpu.mamba1.CHUNK` tokens in reverse. Against the first version
(:mod:`jla.ops.mosaic_gpu.mamba1_bwd`):

* Register history. Per tile, one pass stores the state entering every sub-tile of ``history_tokens(N)``
  tokens (2 at ``N`` = 16) in shared memory; then per sub-tile (last first) one recompute keeps each
  token's entering state *and* decays in registers and the adjoint runs backwards over them. Two
  ``exp2`` per state element and token instead of three, and no per-token shared-memory history.
* ``dB, dC`` (sums over the channels of per-channel products) by warp shuffles: the ``2N`` products of a
  token (of ``32 / 2N`` tokens at ``N`` < 16) are summed over each warp's 32 channels by a butterfly
  reduce-scatter (31 shuffles; lane ``l`` ends with the warp's sum of product ``l``) and stored to a
  per-warp shared row; the CTA's four warp rows are added once per tile and written with one TMA
  copy (XLA adds the channel blocks: deterministic). Two barriers per tile instead of two per token.
* Sequence segments. A batch row of a channel block can be split into ``S`` segments that run in
  parallel CTAs (when ``D / 128 * B`` alone would leave SMs idle): a small pre-pass kernel runs the adjoint
  over each segment but the first with a zero cotangent at its end and returns it with the segment's
  decay ``exp(A * sum(delta))`` (zero across a document boundary); XLA chains them into the cotangent
  entering every segment. The adjoint is linear, so this is exact up to float32 rounding.
* ``d_state`` 32 (and any multiple of 16) runs as 16-state slices: each slice's kernel returns float32
  partial ``ddt, du, dz`` (linear in the per-state terms; the ``D`` skip only in the first slice) that
  XLA adds. Up to 16 states the kernel writes ``ddt, du, dz`` in the input dtype directly.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, memref, vector

from jla.ops.mosaic_gpu.mamba1 import BLOCK_D, CHUNK, UNROLL, Config, _softplus, _vec

__all__ = ["KERNEL_DSTATE", "backward_call", "history_tokens", "num_segments"]

_F32 = jnp.float32
_LN2 = math.log(2.0)
_WARP = 32
_WARPS = BLOCK_D // _WARP
_HALF = _WARP // 2
_SLICE = 16  # states per kernel call when d_state > 16
KERNEL_DSTATE = (4, 8, 16)  # states per kernel call (larger multiples of 16 run as slices)
_HISTORY = {4: 8, 8: 4, 16: 2}  # tokens per register sub-tile, by d_state (16: 4 spills, see history_tokens)
_LINE = plgpu.Layout.WG_STRIDED((BLOCK_D,), vec_size=1)  # thread i holds channel i
_NUM_SMS = 132  # H100 SXM (the GPU the kernels are tested on)
_PREPASS_COST = 0.2  # pre-pass time per token over the backward kernel's (measured 0.16 to 0.20 on the H100)
_MAX_SEGMENTS = 8


def history_tokens(n_state: int) -> int:
    """Tokens per register sub-tile of the backward kernel at ``n_state`` states per call.

    The sub-tile holds every token's entering state and decays in registers: at 16 states, 4 tokens need
    more than the 255 registers a thread has (ptxas spills 224 bytes, and the backward ran 1.2x slower on
    the H100 than with 2 tokens), so 16 states take 2 (and twice the shared-memory sub-tile entries,
    128 KB)."""
    return _HISTORY[n_state]


def num_segments(cfg: Config, num_sms: int = _NUM_SMS) -> int:
    """Sequence segments per (channel block, batch row): the split that minimizes the estimated time
    (waves of CTAs over ``num_sms`` SMs, one CTA per SM, plus the pre-pass on all segments but the first)."""
    ctas = (cfg.dim // BLOCK_D) * cfg.batch

    def cost(s: int) -> float:  # in units of one unsplit row's backward
        main = -(-ctas * s // num_sms) / s
        pre = -(-ctas * (s - 1) // num_sms) * _PREPASS_COST / s
        return main + pre

    best = 1
    for s in range(2, _MAX_SEGMENTS + 1):
        if cfg.num_chunks % s == 0 and cost(s) < cost(best) - 1e-6:
            best = s
    return best


def _stages(dtype) -> int:
    # Shared memory: 16-bit inputs leave room for a 2-deep prefetch, float32 ones for none.
    return 2 if jnp.dtype(dtype).itemsize == 2 else 1


def _c(value: int):
    return mgpu_utils.c(value, ir.IntegerType.get_signless(32))


def _lane():
    return arith.andi(mgpu_utils.thread_idx(), _c(_WARP - 1))


def _scalar(x):
    return vector.extract(x.registers.flat[0], dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([0]))


def _like(x, value):
    """A FragmentedArray of ``x``'s layout holding the scalar ``value`` in its one register."""
    reg = x.registers.flat[0]
    out = x.registers.copy()
    out.flat[0] = vector.broadcast(reg.type, value)
    return mgpu.FragmentedArray(_registers=out, _layout=x.layout, _is_signed=None)


def _butterfly_step(lane, half, vals):
    """One reduce-scatter step: lanes ``l`` and ``l ^ half`` add their halves of ``vals``; the lane whose
    bit ``half`` is set keeps the upper half."""
    upper = arith.cmpi(arith.CmpIPredicate.ne, arith.andi(lane, _c(half)), _c(0))
    out = []
    m = len(vals) // 2
    for i in range(m):
        lo, hi = vals[i], vals[i + m]
        send = arith.select(upper, lo, hi)
        keep = arith.select(upper, hi, lo)
        out.append(arith.addf(keep, mgpu_utils.shfl_bfly(send, half)))
    return out


@plgpu.inline_mgpu(arg_types=(_LINE, _LINE), return_type=plgpu.ShapeDtypeStruct((BLOCK_D,), _F32, layout=_LINE))
def _first_step(ctx, lo, hi):
    """The first step (lanes 16 apart) for products ``i`` and ``i + 16``, done as soon as both exist so that
    only 16 partial sums stay live."""
    del ctx
    (out,) = _butterfly_step(_lane(), _HALF, [_scalar(lo), _scalar(hi)])
    return _like(lo, out)


@plgpu.inline_mgpu(arg_types=(plgpu.RefType(),) + (_LINE,) * _HALF)
def _finish_and_store(ctx, ref, *parts):
    """The remaining four steps; lane ``l`` of warp ``w`` then holds its warp's sum of product ``l`` and
    stores it at ``ref[w, l]``."""
    del ctx
    lane = _lane()
    vals = [_scalar(p) for p in parts]
    half = _HALF // 2
    while half:
        vals = _butterfly_step(lane, half, vals)
        half //= 2
    warp = arith.andi(arith.shrui(mgpu_utils.thread_idx(), _c(5)), _c(_WARPS - 1))
    index = ir.IndexType.get()
    memref.store(vals[0], ref, [arith.index_cast(index, warp), arith.index_cast(index, lane)])


def _launch(grid, grid_names, name, body, **kwargs):
    # No automatic barriers: every thread touches only its own channel's column of the shared tiles;
    # the cross-thread hazards (refilling a tile slot, the per-tile dB/dC rows) are fenced explicitly.
    return plgpu.kernel(
        body, grid=grid, grid_names=grid_names, kernel_name=name,
        compiler_params=plgpu.CompilerParams(approx_math=True, unsafe_no_auto_barriers=True), **kwargs)


# ---------------------------------------------------------------------------------------------
# pre-pass: the adjoint over each segment from a zero cotangent, and the segment's decay
# ---------------------------------------------------------------------------------------------


def _segment_adjoints(cfg: Config, segments: int, dt, z, do, cm, keep, bias, a2t):
    """``gloc, decay [B, S - 1, N, D]``, entry ``s - 1`` for segment ``s`` (``1 .. S - 1``): the cotangent of
    the state entering the segment given a zero one at its end, and ``exp2(a2 * sum(delta))`` over the
    segment (0 if a document starts inside it). The cotangent entering segment ``s`` (the one at the end of
    segment ``s - 1``) is then ``gloc[s - 1] + decay[s - 1] * (the cotangent at the end of segment s)``."""
    n_state, nc, bsz, dim = cfg.dstate, cfg.num_chunks, cfg.batch, cfg.dim
    per = nc // segments
    stages = _stages(dt.dtype)
    tile_names = ["dt"] + (["z"] if cfg.has_z else []) + ["do"]
    names = tile_names + ["cm"] + (["keep"] if cfg.segments else []) + ["bias", "a2t"]
    arrays = dict(dt=dt, z=z, do=do, cm=cm, keep=keep, bias=bias, a2t=a2t)
    shape = jax.ShapeDtypeStruct((bsz, segments - 1, n_state, dim), _F32)
    smem = {k + "_s": plgpu.SMEM((stages, CHUNK, BLOCK_D), arrays[k].dtype) for k in tile_names}
    smem["cm_s"] = plgpu.SMEM((stages, CHUNK, n_state), _F32)
    if cfg.segments:
        smem["keep_s"] = plgpu.SMEM((stages, CHUNK), _F32)
    smem["bars"] = plgpu.Barrier(num_arrivals=len(tile_names) + 1 + int(cfg.segments), num_barriers=stages)

    def body(*refs_in, **scratch):
        refs = dict(zip(names + ["gloc", "decay"], refs_in))
        refs.update(scratch)
        bars = scratch["bars"]
        di, b, sp = lax.axis_index("d"), lax.axis_index("b"), lax.axis_index("s")
        seg = sp + 1
        cols = pl.ds(di * BLOCK_D, BLOCK_D)

        def fetch(step, slot):
            rows = pl.ds((seg * per + per - 1 - step) * CHUNK, CHUNK)
            bar = bars.at[slot]
            for k in tile_names:
                plgpu.copy_gmem_to_smem(refs[k].at[b, rows, cols], refs[k + "_s"].at[slot], bar)
            plgpu.copy_gmem_to_smem(refs["cm"].at[b, rows], refs["cm_s"].at[slot], bar)
            if cfg.segments:
                plgpu.copy_gmem_to_smem(refs["keep"].at[b, rows], refs["keep_s"].at[slot], bar)

        for i in range(min(stages, per)):
            fetch(i, i)
        bias_v = _vec(refs["bias"].at[cols])
        a2 = [_vec(refs["a2t"].at[j, cols]) for j in range(n_state)]
        zero = 0.0 * bias_v

        def tile(step, carry):
            slot = lax.rem(step, stages)
            plgpu.barrier_wait(bars.at[slot])

            def token(t, carry):
                g, sumd, kp = carry
                g = list(g)
                x = _vec(refs["dt_s"].at[slot, t]).astype(_F32) + bias_v
                delta = _softplus(x) if cfg.softplus else x
                dy = _vec(refs["do_s"].at[slot, t]).astype(_F32)
                if cfg.has_z:
                    zz = _vec(refs["z_s"].at[slot, t]).astype(_F32)
                    dy = dy * (zz * jax.nn.sigmoid(zz))
                keep_t = refs["keep_s"][slot, t] if cfg.segments else None
                for j in range(n_state):
                    a = jnp.exp2(delta * a2[j])
                    if cfg.segments:
                        a = a * keep_t
                    g[j] = (dy * refs["cm_s"][slot, t, j] + g[j]) * a
                if cfg.segments:
                    kp = kp * keep_t
                return tuple(g), sumd + delta, kp

            def group(i, carry):  # UNROLL tokens per iteration, last first
                for r in range(UNROLL):
                    carry = token(CHUNK - 1 - (i * UNROLL + r), carry)
                return carry

            carry = lax.fori_loop(0, CHUNK // UNROLL, group, carry)
            plgpu.commit_smem()  # every thread is done reading this slot

            @pl.when(step + stages < per)
            def _():
                fetch(step + stages, slot)

            return carry

        init = (tuple(zero for _ in range(n_state)), zero, zero + 1.0)
        g, sumd, kp = lax.fori_loop(0, per, tile, init)
        for j in range(n_state):
            refs["gloc"][b, sp, j, cols] = g[j]
            refs["decay"][b, sp, j, cols] = jnp.exp2(a2[j] * sumd) * kp

    call = _launch((dim // BLOCK_D, bsz, segments - 1), ("d", "b", "s"), "mamba1_bwd_segments", body,
                   out_type=(shape, shape), scratch_types=smem)
    return call(*(arrays[k] for k in names))


# ---------------------------------------------------------------------------------------------
# backward kernel
# ---------------------------------------------------------------------------------------------


def _backward_kernel(cfg: Config, segments: int, sub: int, out_dtype, dt, u, z, do, bm, cm, keep, bias, dskip,
                     a2t, states, g_in):
    n_state, nc, bsz, dim = cfg.dstate, cfg.num_chunks, cfg.batch, cfg.dim
    width = 2 * n_state
    group = _WARP // width  # tokens per row of warp sums
    nsub = CHUNK // sub
    per = nc // segments
    nblk, tp = dim // BLOCK_D, cfg.seq_len
    stages = _stages(u.dtype)
    tile_names = ["dt", "u"] + (["z"] if cfg.has_z else []) + ["do"]
    names = tile_names + ["bm", "cm"] + (["keep"] if cfg.segments else []) + ["bias", "dskip", "a2t", "states",
                                                                               "g_in"]
    arrays = dict(dt=dt, u=u, z=z, do=do, bm=bm, cm=cm, keep=keep, bias=bias, dskip=dskip, a2t=a2t,
                  states=states, g_in=g_in)
    out_names = ["ddt", "du"] + (["dz"] if cfg.has_z else []) + ["dbc", "ddskip", "dbias", "da", "dh0"]
    big = jax.ShapeDtypeStruct((bsz, tp, dim), out_dtype)
    out_types = [big, big] + ([big] if cfg.has_z else []) + [
        jax.ShapeDtypeStruct((bsz, nblk, tp * width), _F32),
        jax.ShapeDtypeStruct((bsz, segments, dim), _F32),
        jax.ShapeDtypeStruct((bsz, segments, dim), _F32),
        jax.ShapeDtypeStruct((bsz, segments, n_state, dim), _F32),
        jax.ShapeDtypeStruct((bsz, segments, n_state, dim), _F32),
    ]
    smem = {name + "_s": plgpu.SMEM((stages, CHUNK, BLOCK_D), arrays[name].dtype) for name in tile_names}
    smem["bm_s"] = plgpu.SMEM((stages, CHUNK, n_state), _F32)
    smem["cm_s"] = plgpu.SMEM((stages, CHUNK, n_state), _F32)
    if cfg.segments:
        smem["keep_s"] = plgpu.SMEM((stages, CHUNK), _F32)
    smem["subs"] = plgpu.SMEM((nsub, n_state, BLOCK_D), _F32)
    smem["part"] = plgpu.SMEM((_WARPS, CHUNK * width), _F32)  # per-warp sums of the tile's products
    smem["red"] = plgpu.SMEM((CHUNK * width,), _F32)
    n_copies = len(tile_names) + 2 + int(cfg.segments)
    smem["bars"] = plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages)
    red_line = plgpu.Layout.WG_STRIDED((CHUNK * width,), vec_size=1)

    def body(*refs_in, **scratch):
        refs = dict(zip(names + out_names, refs_in))
        refs.update(scratch)
        bars, subs, part, red = (scratch[k] for k in ("bars", "subs", "part", "red"))
        di, b, seg = lax.axis_index("d"), lax.axis_index("b"), lax.axis_index("s")
        cols = pl.ds(di * BLOCK_D, BLOCK_D)
        last = seg * per + per - 1  # the segment's last tile

        def fetch(step, slot):
            rows = pl.ds((last - step) * CHUNK, CHUNK)
            bar = bars.at[slot]
            for name in tile_names:
                plgpu.copy_gmem_to_smem(refs[name].at[b, rows, cols], refs[name + "_s"].at[slot], bar)
            for name in ("bm", "cm"):
                plgpu.copy_gmem_to_smem(refs[name].at[b, rows], refs[name + "_s"].at[slot], bar)
            if cfg.segments:
                plgpu.copy_gmem_to_smem(refs["keep"].at[b, rows], refs["keep_s"].at[slot], bar)

        for i in range(min(stages, per)):
            fetch(i, i)
        bias_v = _vec(refs["bias"].at[cols])
        dskip_v = _vec(refs["dskip"].at[cols])
        a2 = [_vec(refs["a2t"].at[j, cols]) for j in range(n_state)]
        zero = 0.0 * bias_v
        g_init = tuple(_vec(refs["g_in"].at[b, seg, j, cols]) for j in range(n_state))
        acc_init = (tuple(zero for _ in range(n_state)), zero, zero)

        def tile(step, carry):
            n = last - step
            slot = lax.rem(step, stages)
            plgpu.barrier_wait(bars.at[slot])
            tiles = {k: refs[k + "_s"].at[slot] for k in tile_names + ["bm", "cm"]}

            def token_inputs(t):
                x = _vec(tiles["dt"].at[t]).astype(_F32) + bias_v
                delta = _softplus(x) if cfg.softplus else x
                uu = _vec(tiles["u"].at[t]).astype(_F32)
                keep_t = refs["keep_s"][slot, t] if cfg.segments else None
                return x, delta, uu, keep_t

            def decays(delta, keep_t):
                out = []
                for j in range(n_state):
                    a = jnp.exp2(delta * a2[j])
                    out.append(a * keep_t if cfg.segments else a)
                return out

            def advance(h, t, dec, du):
                return [a * hj + du * tiles["bm"][t, j] for j, (a, hj) in enumerate(zip(dec, h))]

            # Pass 1: the state entering every sub-tile (shared memory, each thread its own column).
            def sub_entry(s, h):
                h = list(h)
                for j in range(n_state):
                    subs[s, j, :] = h[j]
                for r in range(sub):
                    t = s * sub + r
                    _, delta, uu, keep_t = token_inputs(t)
                    h = advance(h, t, decays(delta, keep_t), delta * uu)
                return tuple(h)

            h = tuple(_vec(refs["states"].at[b, n, j, cols]) for j in range(n_state))
            h = lax.fori_loop(0, nsub - 1, sub_entry, h)
            for j in range(n_state):
                subs[nsub - 1, j, :] = h[j]

            # Pass 2: per sub-tile (last first) recompute the entering states and decays into registers,
            # then run the adjoint backwards over them.
            def sub_back(i, carry):
                g, (da, dd, db) = carry
                g, da = list(g), list(da)
                s = nsub - 1 - i
                h = [_vec(subs.at[s, j]) for j in range(n_state)]
                hist = []
                for r in range(sub):
                    t = s * sub + r
                    x, delta, uu, keep_t = token_inputs(t)
                    du = delta * uu
                    dec = decays(delta, keep_t)
                    hist.append((x, delta, uu, du, h, dec))
                    if r + 1 < sub:
                        h = advance(h, t, dec, du)
                pending, firsts = {}, [None] * _HALF

                def put(v, val):  # product v of the current row; pair it with v ^ 16 when both exist
                    other = pending.pop(v ^ _HALF, None)
                    if other is None:
                        pending[v] = val
                    else:
                        lo, hi = (val, other) if v < _HALF else (other, val)
                        firsts[v % _HALF] = _first_step(lo, hi)

                for r in reversed(range(sub)):
                    t = s * sub + r
                    tok = n * CHUNK + t
                    x, delta, uu, du, hp, dec = hist[r]
                    hist[r] = None
                    dout = _vec(tiles["do"].at[t]).astype(_F32)
                    if cfg.has_z:
                        zz = _vec(tiles["z"].at[t]).astype(_F32)
                        sg = jax.nn.sigmoid(zz)
                        dy = dout * (zz * sg)
                    else:
                        dy = dout
                    ys = dskip_v * uu
                    ddu = zero
                    dda = zero
                    q = (r % group) * width
                    for j in range(n_state):
                        cj = tiles["cm"][t, j]
                        bj = tiles["bm"][t, j]
                        ht = dec[j] * hp[j] + du * bj
                        ys = ys + ht * cj
                        gh = dy * cj + g[j]
                        put(q + j, dy * ht)  # dC_t[j] term
                        put(q + n_state + j, gh * du)  # dB_t[j] term
                        ddu = ddu + gh * bj
                        g[j] = gh * dec[j]
                        gad = g[j] * hp[j]  # d(loss)/d(decay) times the decay
                        da[j] = da[j] + gad * delta
                        dda = dda + gad * a2[j]
                    if cfg.has_z:
                        refs["dz"][b, tok, cols] = (dout * ys * (sg * (1.0 + zz * (1.0 - sg)))).astype(out_dtype)
                    dd = dd + dy * uu
                    ddelta = dda * _LN2 + ddu * uu
                    if cfg.softplus:
                        ddelta = ddelta * jnp.where(x > 20.0, 1.0, jax.nn.sigmoid(x))
                    db = db + ddelta
                    refs["ddt"][b, tok, cols] = ddelta.astype(out_dtype)
                    refs["du"][b, tok, cols] = (dy * dskip_v + ddu * delta).astype(out_dtype)
                    if r % group == 0:  # the row (tokens t .. t + group - 1) is complete
                        row = s * (sub // group) + r // group
                        _finish_and_store(part.at[:, pl.ds(row * _WARP, _WARP)], *firsts)
                        firsts = [None] * _HALF
                return tuple(g), (tuple(da), dd, db)

            carry = lax.fori_loop(0, nsub, sub_back, carry)
            # Every thread is done with this slot and has stored its warp sums: refill the slot, add the
            # four warps' sums and write the tile's dB/dC partial sums for this channel block.
            plgpu.commit_smem()

            @pl.when(step + stages < per)
            def _():
                fetch(step + stages, slot)

            total = None
            for w in range(_WARPS):
                p = plgpu.load(part.at[w], layout=red_line, optimized=False)
                total = p if total is None else total + p
            red[...] = total
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(red, refs["dbc"].at[b, di, pl.ds(n * CHUNK * width, CHUNK * width)])
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
            return carry

        g, (da, dd, db) = lax.fori_loop(0, per, tile, (g_init, acc_init))
        plgpu.wait_smem_to_gmem(0)
        for j in range(n_state):
            refs["dh0"][b, seg, j, cols] = g[j]
            refs["da"][b, seg, j, cols] = da[j]
        refs["ddskip"][b, seg, cols] = dd
        refs["dbias"][b, seg, cols] = db

    call = _launch((nblk, bsz, segments), ("d", "b", "s"), "mamba1_bwd2", body, out_type=tuple(out_types),
                   scratch_types=smem)
    outs = list(call(*(arrays[k] for k in names)))
    ddt, du = outs[0], outs[1]
    dz = outs[2] if cfg.has_z else None
    dbc, ddskip, dbias, da, dh0 = outs[3:] if cfg.has_z else outs[2:]
    return ddt, du, dz, dbc, ddskip, dbias, da, dh0


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))  # one lowering per config, shared by every call site
def _call(cfg: Config, segments: int, sub: int, out_dtype, dt, u, z, do, bm, cm, keep, bias, dskip, a2t, states,
          dfin_t):
    if segments > 1:
        gloc, decay = _segment_adjoints(cfg, segments, dt, z, do, cm, keep, bias, a2t)
        g_in = [dfin_t]  # the cotangent at the end of the last segment, then of segment s (entering s + 1)
        for s in range(segments - 2, -1, -1):
            g_in.append(gloc[:, s] + decay[:, s] * g_in[-1])
        g_in = jnp.stack(g_in[::-1], axis=1)
    else:
        g_in = dfin_t[:, None]
    ddt, du, dz, dbc, ddskip, dbias, da, dh0 = _backward_kernel(
        cfg, segments, sub, out_dtype, dt, u, z, do, bm, cm, keep, bias, dskip, a2t, states, g_in)
    return (ddt, du, dz, dbc, jnp.sum(ddskip, axis=1), jnp.sum(dbias, axis=1), jnp.sum(da, axis=1), dh0[:, 0])


def backward_call(cfg: Config, dt, u, z, do, bm, cm, keep, bias, dskip, a2t, states, dfin_t, *, segments=None):
    """Cotangents for the padded kernel inputs of :func:`jla.ops.mosaic_gpu.mamba1._forward_call`, as
    :func:`jla.ops.mosaic_gpu.mamba1_bwd.backward_call` returns them: ``ddt, du [B, T, D]`` and ``dz`` (or
    None) in u's dtype, ``dbc [B, D / 128, T * 2N]`` (per channel block and token, ``N`` sums for ``dC`` then
    ``N`` for ``dB``), ``ddskip, dbias [B, D]``, ``da [B, N, D]`` (w.r.t. ``A``) and ``dh0 [B, N, D]``.

    ``segments`` (default :func:`num_segments`) splits each sequence into that many parallel segments; it
    must divide the number of tiles.
    """
    n_state = cfg.dstate
    if n_state in KERNEL_DSTATE:
        segs = num_segments(cfg) if segments is None else segments
        if cfg.num_chunks % segs:
            raise ValueError(f"{segs} segments do not divide {cfg.num_chunks} tiles")
        return _call(cfg, segs, history_tokens(n_state), jnp.dtype(u.dtype), dt, u, z, do, bm, cm, keep, bias, dskip,
                     a2t, states, dfin_t)
    if n_state % _SLICE:
        raise ValueError(f"d_state {n_state} is not supported by the backward kernel")
    # 16-state slices: float32 partial ddt, du, dz (the D skip only in the first slice), added here.
    scfg = dataclasses.replace(cfg, dstate=_SLICE)
    segs = num_segments(scfg) if segments is None else segments
    if cfg.num_chunks % segs:
        raise ValueError(f"{segs} segments do not divide {cfg.num_chunks} tiles")
    outs = []
    for p in range(n_state // _SLICE):
        sl = slice(p * _SLICE, (p + 1) * _SLICE)
        outs.append(_call(scfg, segs, history_tokens(_SLICE), jnp.dtype(_F32), dt, u, z, do, bm[..., sl],
                          cm[..., sl], keep, bias, dskip if p == 0 else jnp.zeros_like(dskip), a2t[sl],
                          states[:, :, sl], dfin_t[:, sl]))
    ddt = sum(o[0] for o in outs).astype(u.dtype)
    du = sum(o[1] for o in outs).astype(u.dtype)
    dz = None if z is None else sum(o[2] for o in outs).astype(u.dtype)
    bsz, nblk = cfg.batch, cfg.dim // BLOCK_D
    dbc = jnp.stack([o[3].reshape(bsz, nblk, cfg.seq_len, 2, _SLICE) for o in outs], axis=4)
    dbc = dbc.reshape(bsz, nblk, cfg.seq_len * 2 * n_state)  # per token: dC (all states), then dB
    dbias = sum(o[5] for o in outs)
    da = jnp.concatenate([o[6] for o in outs], axis=1)
    dh0 = jnp.concatenate([o[7] for o in outs], axis=1)
    return ddt, du, dz, dbc, outs[0][4], dbias, da, dh0
