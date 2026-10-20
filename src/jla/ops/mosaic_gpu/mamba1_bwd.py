# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), csrc/selective_scan/selective_scan_bwd_kernel.cuh
"""Backward kernel of the Mamba-1 selective scan (:mod:`jla.ops.mosaic_gpu.mamba1`).

Same CTA layout as the forward (one channel per thread, the ``N`` states in registers). The
CTA walks the tiles in reverse. Per tile it rebuilds the states from the state the forward saved
at the tile start: one pass stores the state entering each sub-tile of :data:`SUB` tokens, then
per sub-tile (in reverse) a recompute stores each token's entering state in a thread-private
shared-memory history and the adjoint recurrence ``g_{t-1} = exp(delta_t A) (dy_t C_t + g_t)``
runs backwards. Per token it emits ``ddt, du, dz``; ``dA, dD, d(dt_bias)`` and the initial-state
cotangent accumulate in registers. ``dB, dC`` are sums over channels: each thread writes its
``2N`` products to a ``[64, 128]`` shared tile, a row reduction (``64 / 2N`` tokens at a time)
sums the CTA's 128 channels, and XLA sums the channel blocks (deterministic; the official kernel
uses atomics).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu.mamba1 import BLOCK_D, CHUNK, Config, _kernel, _softplus, _vec

__all__ = ["SUB", "backward_call"]

SUB = 8  # tokens per sub-tile of the state history
_F32 = jnp.float32
_LN2 = math.log(2.0)
_WG = plgpu.Layout.WGMMA
_ROWS = 64  # rows of the dB/dC reduction tile


def _stages(dtype) -> int:
    # Shared memory: 16-bit inputs leave room for a 2-deep prefetch, float32 ones for none.
    return 2 if jnp.dtype(dtype).itemsize == 2 else 1


@functools.partial(jax.jit, static_argnums=0)  # one lowering per config, shared by every call site
def backward_call(cfg: Config, dt, u, z, do, bm, cm, keep, bias, dskip, a2t, states, dfin_t):
    """Cotangents for the padded kernel inputs of :func:`jla.ops.mosaic_gpu.mamba1._forward_call`.

    Jitted with ``cfg`` static, like the forward call: lowering this kernel takes about two minutes,
    and every call of one configuration in a traced program (one per Mamba-1 layer of a model that
    does not scan its layers) then shares one lowering.

    ``do [B, T, D]`` (u's dtype), ``states [B, NC, N, D]`` (from the forward), ``dfin_t [B, N, D]``.
    Returns ``ddt, du [B, T, D]`` and ``dz`` (float32, or None), ``dbc [B, D / 128, T * 2N]``
    (per channel block: for each token ``N`` sums for ``dC`` then ``N`` for ``dB``),
    ``ddskip, dbias [B, D]``, ``da [B, N, D]`` (w.r.t. ``A``) and ``dh0 [B, N, D]``.
    """
    n_state, nc, bsz, dim = cfg.dstate, cfg.num_chunks, cfg.batch, cfg.dim
    width = 2 * n_state
    if _ROWS % width or SUB % (_ROWS // width):
        raise ValueError(f"d_state {n_state} is not supported by the backward kernel")
    group = _ROWS // width  # tokens per reduction
    nblk, tp = dim // BLOCK_D, cfg.seq_len
    stages = _stages(u.dtype)
    tile_names = ["dt", "u"] + (["z"] if cfg.has_z else []) + ["do"]
    names = tile_names + ["bm", "cm"] + (["keep"] if cfg.segments else []) + ["bias", "dskip", "a2t", "states",
                                                                               "dfin"]
    arrays = dict(dt=dt, u=u, z=z, do=do, bm=bm, cm=cm, keep=keep, bias=bias, dskip=dskip, a2t=a2t,
                  states=states, dfin=dfin_t)
    out_names = ["ddt", "du"] + (["dz"] if cfg.has_z else []) + ["dbc", "ddskip", "dbias", "da", "dh0"]
    big = jax.ShapeDtypeStruct((bsz, tp, dim), _F32)
    out_types = [big, big] + ([big] if cfg.has_z else []) + [
        jax.ShapeDtypeStruct((bsz, nblk, tp * width), _F32),
        jax.ShapeDtypeStruct((bsz, dim), _F32),
        jax.ShapeDtypeStruct((bsz, dim), _F32),
        jax.ShapeDtypeStruct((bsz, n_state, dim), _F32),
        jax.ShapeDtypeStruct((bsz, n_state, dim), _F32),
    ]
    smem = {name + "_s": plgpu.SMEM((stages, CHUNK, BLOCK_D), arrays[name].dtype) for name in tile_names}
    smem["bm_s"] = plgpu.SMEM((stages, CHUNK, n_state), _F32)
    smem["cm_s"] = plgpu.SMEM((stages, CHUNK, n_state), _F32)
    if cfg.segments:
        smem["keep_s"] = plgpu.SMEM((stages, CHUNK), _F32)
    smem["subs"] = plgpu.SMEM((CHUNK // SUB, n_state, BLOCK_D), _F32)
    smem["hist"] = plgpu.SMEM((SUB, n_state, BLOCK_D), _F32)
    smem["prod"] = plgpu.SMEM((_ROWS, BLOCK_D), _F32)
    smem["red"] = plgpu.SMEM((CHUNK * width,), _F32)
    n_copies = len(tile_names) + 2 + int(cfg.segments)
    smem["bars"] = plgpu.Barrier(num_arrivals=n_copies, num_barriers=stages)

    def body(*refs_in, **scratch):
        refs = dict(zip(names + out_names, refs_in))
        refs.update(scratch)
        bars, subs, hist, prod, red = (scratch[k] for k in ("bars", "subs", "hist", "prod", "red"))
        di, b = lax.axis_index("d"), lax.axis_index("b")
        d0 = di * BLOCK_D
        cols = pl.ds(d0, BLOCK_D)

        def fetch(step, slot):
            rows = pl.ds((nc - 1 - step) * CHUNK, CHUNK)
            bar = bars.at[slot]
            for name in tile_names:
                plgpu.copy_gmem_to_smem(refs[name].at[b, rows, cols], refs[name + "_s"].at[slot], bar)
            for name in ("bm", "cm"):
                plgpu.copy_gmem_to_smem(refs[name].at[b, rows], refs[name + "_s"].at[slot], bar)
            if cfg.segments:
                plgpu.copy_gmem_to_smem(refs["keep"].at[b, rows], refs["keep_s"].at[slot], bar)

        for i in range(min(stages, nc)):
            fetch(i, i)
        bias_v = _vec(refs["bias"].at[cols])
        dskip_v = _vec(refs["dskip"].at[cols])
        a2 = [_vec(refs["a2t"].at[j, cols]) for j in range(n_state)]
        zero = 0.0 * bias_v
        g_init = tuple(_vec(refs["dfin"].at[b, j, cols]) for j in range(n_state))
        acc_init = (tuple(zero for _ in range(n_state)), zero, zero)

        def tile(step, carry):
            g, (da, dd, db) = carry
            g, da = list(g), list(da)
            n = nc - 1 - step
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

            def advance(h, t):
                _, delta, uu, keep_t = token_inputs(t)
                du = delta * uu
                return [a * hj + du * tiles["bm"][t, j] for j, (a, hj) in enumerate(zip(decays(delta, keep_t), h))]

            # Pass 1: the state entering every sub-tile.
            def sub_entry(s, h):
                h = list(h)
                for j in range(n_state):
                    subs[s, j, :] = h[j]
                for r in range(SUB):
                    h = advance(h, s * SUB + r)
                return tuple(h)

            h = tuple(_vec(refs["states"].at[b, n, j, cols]) for j in range(n_state))
            h = lax.fori_loop(0, CHUNK // SUB - 1, sub_entry, h)
            for j in range(n_state):
                subs[CHUNK // SUB - 1, j, :] = h[j]

            # Pass 2: per sub-tile (last first), recompute the history, then run the adjoint backwards.
            def sub_back(i, carry):
                g, (da, dd, db) = carry
                g, da = list(g), list(da)
                s = CHUNK // SUB - 1 - i
                h = [_vec(subs.at[s, j]) for j in range(n_state)]
                for r in range(SUB):
                    for j in range(n_state):
                        hist[r, j, :] = h[j]
                    if r + 1 < SUB:
                        h = advance(h, s * SUB + r)
                for r in reversed(range(SUB)):
                    t = s * SUB + r
                    tok = n * CHUNK + t
                    x, delta, uu, keep_t = token_inputs(t)
                    du = delta * uu
                    hp = [_vec(hist.at[r, j]) for j in range(n_state)]
                    dec = decays(delta, keep_t)
                    ht = [a * hj + du * tiles["bm"][t, j] for j, (a, hj) in enumerate(zip(dec, hp))]
                    ys = dskip_v * uu
                    for j in range(n_state):
                        ys = ys + ht[j] * tiles["cm"][t, j]
                    dout = _vec(tiles["do"].at[t]).astype(_F32)
                    if cfg.has_z:
                        zz = _vec(tiles["z"].at[t]).astype(_F32)
                        sg = jax.nn.sigmoid(zz)
                        dy = dout * (zz * sg)
                        refs["dz"][b, tok, cols] = dout * ys * (sg * (1.0 + zz * (1.0 - sg)))
                    else:
                        dy = dout
                    dd = dd + dy * uu
                    ddu = zero
                    dda = zero
                    row = (r % group) * width  # s * SUB is a multiple of group
                    for j in range(n_state):
                        gh = dy * tiles["cm"][t, j] + g[j]
                        prod[row + j, :] = dy * ht[j]  # dC_t[j] term
                        prod[row + n_state + j, :] = gh * du  # dB_t[j] term
                        ddu = ddu + gh * tiles["bm"][t, j]
                        gad = gh * hp[j] * dec[j]
                        da[j] = da[j] + gad * delta
                        dda = dda + gad * a2[j]
                        g[j] = gh * dec[j]
                    ddelta = dda * _LN2 + ddu * uu
                    if cfg.softplus:
                        ddelta = ddelta * jnp.where(x > 20.0, 1.0, jax.nn.sigmoid(x))
                    db = db + ddelta
                    refs["ddt"][b, tok, cols] = ddelta
                    refs["du"][b, tok, cols] = dy * dskip_v + ddu * delta
                    if r % group == 0:  # tokens t .. t + group - 1 are in ``prod``: sum over the channels
                        plgpu.commit_smem()
                        sums = plgpu.load(prod, layout=_WG, optimized=False).sum(axis=1)
                        red[pl.ds(t * width, _ROWS)] = sums
                        plgpu.commit_smem()
                return tuple(g), (tuple(da), dd, db)

            g, (da, dd, db) = lax.fori_loop(0, CHUNK // SUB, sub_back, (tuple(g), (tuple(da), dd, db)))
            plgpu.copy_smem_to_gmem(red, refs["dbc"].at[b, di, pl.ds(n * CHUNK * width, CHUNK * width)])
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

            @pl.when(step + stages < nc)
            def _():
                fetch(step + stages, slot)

            return tuple(g), (tuple(da), dd, db)

        g, (da, dd, db) = lax.fori_loop(0, nc, tile, (g_init, acc_init))
        plgpu.wait_smem_to_gmem(0)
        for j in range(n_state):
            refs["dh0"][b, j, cols] = g[j]
            refs["da"][b, j, cols] = da[j]
        refs["ddskip"][b, cols] = dd
        refs["dbias"][b, cols] = db

    call = _kernel(cfg, "mamba1_bwd", body, out_type=tuple(out_types), scratch_types=smem)
    outs = list(call(*(arrays[k] for k in names)))
    ddt, du = outs[0], outs[1]
    dz = outs[2] if cfg.has_z else None
    dbc, ddskip, dbias, da, dh0 = outs[3:] if cfg.has_z else outs[2:]
    return ddt, du, dz, dbc, ddskip, dbias, da, dh0
