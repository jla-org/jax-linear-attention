# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py,
# mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
"""Dedicated Mosaic GPU (H100) kernels for Mamba-3 MIMO (``jla.ops`` kind ``mamba3_mimo``).

``S_t = exp(g_t) S_{t-1} + beta_t sum_r k_{t-1,r}^T v_{t-1,r} + gamma_t sum_r k_{t,r}^T v_{t,r}`` and
``o_{t,r} = scale q_{t,r} S_t`` (``jla.ops.reference.mamba3_mimo``): one ``[K, V]`` state per head
written by ``R`` ranks per token.

Tiling (the official ``mamba3_mimo_fwd`` / ``mamba3_mimo_bwd`` scheme): a chunk holds
``C = 64 / R`` tokens and all their ranks, so every tile has ``C R = 64`` (token, rank) rows, the
SISO kernels' tile height. The rank-``R`` state update is then one ``[K, 64] x [64, V]`` tensor-core
product per chunk (``R`` keys per token contracted at once) instead of ``R`` outer-product streams,
and the whole problem costs ``R`` times one SISO recurrence over ``T`` tokens rather than ``R^2``
SISO recurrences (the route of :mod:`jla.ops.mosaic_gpu.mamba3_mimo` with ``kernel="siso"``).
Inside a tile the rows are rank-major (row ``r C + i`` is rank ``r`` of the chunk's token ``i``), so
each rank's ``[C, D]`` block is one TMA box of the free ``[B, T, H R, D]`` view of the inputs, and
the token of a row is ``row mod C``.

The kernels are :mod:`jla.ops.mosaic_gpu.mamba3`'s warp-specialized forward and backward kernels
(the shifted state ``S~_t = exp(g_{t+1}) S_t + beta_{t+1} sum_r k_{t,r}^T v_{t,r}``, the per-row
gate tile, the TMA ring and the two output / two backward compute warpgroups) with three changes:

* the score weights use the row's token: ``c_j exp(E_i - A_j)`` when token ``j`` precedes token
  ``i``, and the dense ``R x R`` same-token block ``gamma_i`` (every rank of token ``i`` reads the
  state after all of token ``i``'s ranks were written);
* every TMA load and store moves ``R`` row blocks;
* the backward kernel leaves the same-token part of the ``E`` cotangent to XLA
  (``dE -= gamma dgamma`` per row, exact per token, which is all the token-level gates see).

XLA computes the per-token gate rows over chunks of ``C`` tokens and tiles them over the ranks;
JAX autodiff maps the row gradients back to ``g``, ``beta``, ``gamma`` and the initial state.
Inputs are bfloat16 on the tensor cores with float32 accumulation; gates and decays are float32.
``R`` in :data:`RANKS`, ``(K, V)`` in :data:`SUPPORTED_DIMS`, any ``T`` (padded to the chunk; zero
padding is inert).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import finish_together
from jla.ops.mosaic_gpu.mamba3 import (
    _BWD_REGISTERS,
    _BWD_SMEM_BYTES,
    _COL,
    _FWD_MEMORY,
    _FWD_REGISTERS,
    _FWD_SMEM_BYTES,
    _FWD_STATE,
    _ROW,
    _SPLIT,
    _WG_A,
    _WG_B,
    _WG_MEMORY,
    CHUNK,
    NUM_ROWS,
    ROW_A,
    ROW_C,
    ROW_DEC,
    ROW_E,
    ROW_GAMMA,
    ROW_QS,
    ROW_RUN,
    ROW_RUN_COL,
    ROW_W,
    SUPPORTED_DIMS,
    _backward_smem,
    _cols,
    _fitting_stages,
    _forward_smem,
    _iota,
    _kernel,
    _load,
    _masked_exp,
    _matmul,
    _matmul_while,
    _memory_warpgroup,
    _on_hopper,
    _operand_smem,
    _Regs,
    _rows,
)
from jla.ops.reference import Mamba3State, segment_resets

__all__ = ["RANKS", "Config", "backward", "forward", "mamba3_mimo", "prepare_gates", "supports"]

Array = jax.Array

_F32 = jnp.float32
_BF16 = jnp.bfloat16
_HIGHEST = lax.Precision.HIGHEST

# Ranks whose chunk (64 / R tokens) is a whole number of 8-row tiles.
RANKS = (1, 2, 4, 8)


@dataclasses.dataclass(frozen=True)
class Config:
    """Static shapes and options of one kernel call (``num_chunks`` chunks of ``64 / rank`` tokens)."""

    batch: int
    heads: int
    rank: int
    num_chunks: int
    key_dim: int
    value_dim: int
    scale: float = 1.0
    segments: bool = False  # mask the scores to equal segment runs
    stages: int = 4  # TMA pipeline depth over chunks (capped by shared memory; even in the forward)
    interpret: bool = False  # run the kernels in the GPU interpreter (CPU tests)
    profile_dir: str = ""  # developer option: Mosaic GPU's per-warpgroup trace of the named phases

    @property
    def tokens(self) -> int:
        """Tokens per chunk."""
        return CHUNK // self.rank

    @property
    def seq_len(self) -> int:
        return self.num_chunks * self.tokens


def _token_of(index: Array, tokens: int) -> Array:
    """The chunk-local token of a rank-major tile row (``row mod tokens``; ``tokens`` is a power of 2)."""
    return jnp.bitwise_and(index, tokens - 1)


# ---------------------------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------------------------


def forward(
    cfg: Config, q: Array, k: Array, v: Array, gates: Array, h0: Array, *, save_states: bool
) -> tuple[Array, ...]:
    """``(o, final[, states])`` of the chunked MIMO recurrence.

    ``q, k [B, T, H R, K]`` and ``v [B, T, H R, V]`` bf16 (rank ``r`` of head ``h`` at index
    ``h R + r``) with ``T = cfg.seq_len``; ``gates [B, H, NC, NUM_ROWS, 64]`` f32
    (:func:`prepare_gates`); ``h0 [B, H, K, V]`` f32. Returns ``o [B, T, H R, V]`` bf16 (times
    ``cfg.scale``), the final ``S~`` state ``[B, H, K, V]`` f32 and, with ``save_states``, the bf16
    state entering each chunk ``[B, H, NC, K, V]``.
    """
    c, ct, rank = CHUNK, cfg.tokens, cfg.rank
    kd, vd, nc = cfg.key_dim, cfg.value_dim, cfg.num_chunks
    # Output warpgroup o handles the chunks n with n % 2 == o and owns every other ring slot.
    st = 1 if nc == 1 else _fitting_stages(min(cfg.stages, nc), _forward_smem, _FWD_SMEM_BYTES, kd, vd, even=True)
    out_types = [
        jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads * rank, vd), _BF16),
        jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32),
    ]
    if save_states:
        out_types.append(jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd, vd), _BF16))

    def body(q_ref, k_ref, v_ref, g_ref, h0_ref, o_ref, f_ref, *maybe_states, **smem):
        st_ref = maybe_states[0] if save_states else None
        q_smem, k_smem, v_smem, g_smem = smem["q"], smem["k"], smem["v"], smem["g"]
        s_smem, wv_smem, o_smem, stage = smem["s"], smem["wv"], smem["o"], smem.get("stage")
        data, consumed = smem["data"], smem["consumed"]
        state_ready, state_free = smem["state_ready"], smem["state_free"]
        wg, h, b = lax.axis_index("wg"), lax.axis_index("h"), lax.axis_index("b")

        def fetch(n, slot):
            tok = pl.ds(n * ct, ct)
            bar = data.at[slot]
            for r in range(rank):
                rows, head = pl.ds(r * ct, ct), h * rank + r
                plgpu.copy_gmem_to_smem(q_ref.at[b, tok, head], q_smem.at[slot, rows], bar)
                plgpu.copy_gmem_to_smem(k_ref.at[b, tok, head], k_smem.at[slot, rows], bar)
                plgpu.copy_gmem_to_smem(v_ref.at[b, tok, head], v_smem.at[slot, rows], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, h, n], g_smem.at[slot], bar)

        @pl.when(wg == _FWD_MEMORY)
        def _():
            _memory_warpgroup(fetch, consumed, nc, st, lambda step: step, _FWD_REGISTERS[_FWD_MEMORY])

        def output(o_wg: int):
            """Outputs o = qs * (q H) + (scores * weights) v of the chunks n = o_wg, o_wg + 2, ..."""
            plgpu.set_max_registers(_FWD_REGISTERS[o_wg], action="increase")
            sq = (c, c)
            tok_i, tok_j = _token_of(_iota(sq, 0), ct), _token_of(_iota(sq, 1), ct)
            same_token = tok_i == tok_j
            earlier = tok_j < tok_i
            o_tile = o_smem.at[o_wg]
            staging = None if stage is None else stage.at[o_wg]

            def step(j, carry):
                n = 2 * j + o_wg
                slot = lax.rem(n, st)
                with jax.named_scope("out_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                gs = g_smem.at[slot]

                def weights():
                    mask = earlier
                    if cfg.segments:
                        same = _rows(_load(gs.at[ROW_RUN], _ROW), sq) == _cols(_load(gs.at[ROW_RUN_COL], _COL), sq)
                        mask = jnp.logical_and(mask, same)
                    exponent = _rows(_load(gs.at[ROW_E], _ROW), sq) - _cols(_load(gs.at[ROW_A], _COL), sq)
                    decay = _masked_exp(exponent, mask) * _cols(_load(gs.at[ROW_C], _COL), sq)
                    return jnp.where(same_token, _rows(_load(gs.at[ROW_GAMMA], _ROW), sq), decay)

                with jax.named_scope("out_scores"):
                    scores, weight = _matmul_while(cfg, sq, ((q_smem.at[slot], k_smem.at[slot].T),), weights)
                    p = (scores * weight).astype(_BF16)
                    qs = _rows(_load(gs.at[ROW_QS], _ROW), (c, vd))
                with jax.named_scope("out_wait_state"):
                    plgpu.barrier_wait(state_ready.at[o_wg])  # S_n, published in buffer n % 2
                with jax.named_scope("out_matmuls"):
                    o = _matmul(cfg, (c, vd), (q_smem.at[slot], s_smem.at[o_wg]))

                    @pl.when(n + 2 < nc)
                    def _():
                        plgpu.barrier_arrive(state_free.at[o_wg])

                    o = _matmul(cfg, (c, vd), (_Regs(p), v_smem.at[slot]), init=o * qs, stage=staging)
                plgpu.barrier_arrive(consumed.at[slot])
                with jax.named_scope("out_store"):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # this warpgroup's last o store read o_tile
                    o_tile[...] = (o * cfg.scale).astype(_BF16)
                    plgpu.commit_smem()
                    for r in range(rank):
                        plgpu.copy_smem_to_gmem(
                            o_tile.at[pl.ds(r * ct, ct)], o_ref.at[b, pl.ds(n * ct, ct), h * rank + r],
                            commit_group=False,
                        )
                    plgpu.commit_smem_to_gmem_group()
                return carry

            lax.fori_loop(0, (nc - o_wg + 1) // 2, step, None)
            plgpu.wait_smem_to_gmem(0)

        for o_wg in (0, 1):
            pl.when(wg == o_wg)(functools.partial(output, o_wg))

        @pl.when(wg == _FWD_STATE)
        def _state():
            """The serial state pass H' = dec H + K^T (w v) over all (token, rank) rows of a chunk."""
            plgpu.set_max_registers(_FWD_REGISTERS[_FWD_STATE], action="increase")

            def publish(s, n, buf):
                s_smem.at[buf][...] = s.astype(_BF16)
                plgpu.commit_smem()
                plgpu.barrier_arrive(state_ready.at[buf])
                if save_states:
                    plgpu.copy_smem_to_gmem(s_smem.at[buf], st_ref.at[b, h, n])

            s0 = _load(h0_ref.at[b, h])
            publish(s0, 0, 0)

            def step(n, s):
                slot = lax.rem(n, st)
                with jax.named_scope("state_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                with jax.named_scope("state_wv"):
                    w = _rows(_load(g_smem.at[slot, ROW_W], _ROW), (c, vd))
                    dec = g_smem[slot, ROW_DEC, 0]
                    wv_smem[...] = (_load(v_smem.at[slot]).astype(_F32) * w).astype(_BF16)
                    plgpu.commit_smem()
                with jax.named_scope("state_update"):
                    s = _matmul(cfg, (kd, vd), (k_smem.at[slot].T, wv_smem), init=s * dec)
                plgpu.barrier_arrive(consumed.at[slot])
                nxt = n + 1
                buf = lax.rem(nxt, 2)

                @pl.when(nxt < nc)
                def _():
                    with jax.named_scope("state_wait_free"):

                        @pl.when(nxt >= 2)
                        def _():
                            plgpu.barrier_wait(state_free.at[buf])  # the output warpgroup is done with it

                        if save_states:
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    with jax.named_scope("state_publish"):
                        publish(s, nxt, buf)

                return s

            s = lax.fori_loop(0, nc, step, s0)
            f_ref.at[b, h][...] = s
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])  # no warpgroup exits while another still issues WGMMAs

    scratch = dict(
        done=plgpu.Barrier(num_arrivals=4),
        q=_operand_smem((st, c, kd)),
        k=_operand_smem((st, c, kd)),
        v=_operand_smem((st, c, vd)),
        g=plgpu.SMEM((st, NUM_ROWS, c), _F32),
        s=_operand_smem((2, kd, vd)),
        wv=_operand_smem((c, vd)),
        o=_operand_smem((2, c, vd)),
        data=plgpu.Barrier(num_arrivals=3 * rank + 1, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=2, num_barriers=st),
        state_ready=plgpu.Barrier(num_arrivals=1, num_barriers=2),
        state_free=plgpu.Barrier(num_arrivals=1, num_barriers=2),
    )
    if cfg.interpret:
        scratch["stage"] = _operand_smem((2, c, c))
    kern = _kernel(cfg, "mamba3_mimo_fwd", body, num_threads=4, out_type=tuple(out_types), scratch_types=scratch)
    return kern(q, k, v, gates, h0)


# ---------------------------------------------------------------------------------------------
# backward kernel
# ---------------------------------------------------------------------------------------------


def backward(
    cfg: Config, q: Array, k: Array, v: Array, do: Array, gates: Array, states: Array, dfinal: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """``(dq, dk, dv, dgates, dh0)`` for the forward kernel's ``(o, final)`` cotangents.

    ``do [B, T, H R, V]`` bf16 (already multiplied by ``cfg.scale``), ``states`` from
    :func:`forward`, ``dfinal [B, H, K, V]`` f32. ``dgates`` has the layout of ``gates``; the
    cotangents of a token's ``R`` rows are exact in their sum (the split of the ``E`` row over the
    ranks is not), which is what the token-level gates of :func:`prepare_gates` receive.

    Warpgroup A carries the state cotangent ``dH`` backwards over the chunks and computes ``dq`` and
    the ``E``, ``qs``, ``dec`` cotangents; warpgroup B computes the score matrices, ``dk``, ``dv`` and
    the other gate cotangents (as in :func:`jla.ops.mosaic_gpu.mamba3.backward`).
    """
    c, ct, rank = CHUNK, cfg.tokens, cfg.rank
    kd, vd, nc = cfg.key_dim, cfg.value_dim, cfg.num_chunks
    st = _fitting_stages(min(cfg.stages, nc), _backward_smem, _BWD_SMEM_BYTES, kd, vd)
    shape_k = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads * rank, kd), _BF16)
    shape_v = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads * rank, vd), _BF16)
    shape_g = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, NUM_ROWS, c), _F32)
    shape_s = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, kd, vd), _F32)
    shape_d = jax.ShapeDtypeStruct((cfg.batch, cfg.heads, nc, kd), _F32)
    rows_b, rows_a = pl.ds(0, _SPLIT), pl.ds(_SPLIT, NUM_ROWS - _SPLIT)

    def body(
        q_ref, k_ref, v_ref, do_ref, g_ref, st_ref, df_ref, dq_ref, dk_ref, dv_ref, dg_ref, dh0_ref, dd_ref, **smem
    ):
        q_smem, k_smem, v_smem, do_smem = smem["q"], smem["k"], smem["v"], smem["do"]
        g_smem, h_smem, dh_smem, qsdo_smem = smem["g"], smem["h"], smem["dh"], smem["qsdo"]
        wqt_smem, dq_smem, dk_smem, dv_smem = smem["wqt"], smem["dq"], smem["dk"], smem["dv"]
        dga_smem, dgb_smem, dd_smem, stage = smem["dga"], smem["dgb"], smem["dd"], smem.get("stage")
        data, consumed = smem["data"], smem["consumed"]
        dh_ready, dh_free, wq_ready, wq_free = smem["dh_ready"], smem["dh_free"], smem["wq_ready"], smem["wq_free"]
        wg, h, b = lax.axis_index("wg"), lax.axis_index("h"), lax.axis_index("b")
        sq = (c, c)

        def fetch(n, slot):
            tok = pl.ds(n * ct, ct)
            bar = data.at[slot]
            for r in range(rank):
                rows, head = pl.ds(r * ct, ct), h * rank + r
                for src, dst in ((q_ref, q_smem), (k_ref, k_smem), (v_ref, v_smem), (do_ref, do_smem)):
                    plgpu.copy_gmem_to_smem(src.at[b, tok, head], dst.at[slot, rows], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, h, n], g_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(st_ref.at[b, h, n], h_smem.at[slot], bar)

        def store(tile, ref, n):
            for r in range(rank):
                plgpu.copy_smem_to_gmem(
                    tile.at[pl.ds(r * ct, ct)], ref.at[b, pl.ds(n * ct, ct), h * rank + r], commit_group=False
                )

        @pl.when(wg == _WG_MEMORY)
        def _():
            _memory_warpgroup(fetch, consumed, nc, st, lambda step: nc - 1 - step, _BWD_REGISTERS[_WG_MEMORY])

        @pl.when(wg == _WG_A)
        def _state():
            """dH = dec dH' + Q^T (qs dO) in reverse, dQ = qs (dO H^T) + Wq K, and the E, qs, dec rows."""
            plgpu.set_max_registers(_BWD_REGISTERS[_WG_A], action="increase")

            def step(it, dh):
                n, slot, buf = nc - 1 - it, lax.rem(it, st), lax.rem(it, 2)
                with jax.named_scope("dstate_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                gs = g_smem.at[slot]
                qs = _load(gs.at[ROW_QS], _ROW)
                dec = g_smem[slot, ROW_DEC, 0]
                with jax.named_scope("dstate_wait_free"):

                    @pl.when(it > 0)
                    def _():
                        plgpu.barrier_wait(dh_free)  # warpgroup B has read the previous dH'

                with jax.named_scope("dstate_publish"):
                    dh_smem[...] = dh.astype(_BF16)
                    plgpu.commit_smem()
                    plgpu.barrier_arrive(dh_ready)
                with jax.named_scope("dstate_ddec"):
                    ddec_rows = (_load(h_smem.at[slot]).astype(_F32) * dh).sum(axis=1)  # rows of <H, dH'>
                with jax.named_scope("dstate_update"):
                    dov = _load(do_smem.at[slot]).astype(_F32)
                    qsdo_smem[...] = (dov * _rows(qs, (c, vd))).astype(_BF16)
                    plgpu.commit_smem()
                    dh = _matmul(cfg, (kd, vd), (q_smem.at[slot].T, qsdo_smem), init=dh * dec)
                with jax.named_scope("dstate_doh"):
                    doh = _matmul(cfg, (c, kd), (do_smem.at[slot], h_smem.at[slot].T))
                    dqs = (doh * _load(q_smem.at[slot]).astype(_F32)).sum(axis=1)
                with jax.named_scope("dstate_wait_wq"):
                    plgpu.barrier_wait(wq_ready.at[buf])
                with jax.named_scope("dstate_dq"):
                    dq = _matmul(cfg, (c, kd), (wqt_smem.at[buf].T, k_smem.at[slot]), init=doh * _rows(qs, (c, kd)))

                    @pl.when(it + 2 < nc)
                    def _():
                        plgpu.barrier_arrive(wq_free.at[buf])

                    # q . dq without its qs part; XLA removes the same-token (gamma) part.
                    de = (dq * _load(q_smem.at[slot]).astype(_F32)).sum(axis=1) - qs * dqs
                plgpu.barrier_arrive(consumed.at[slot])
                with jax.named_scope("dstate_store"):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    dq_smem[...] = dq.astype(_BF16)
                    dga_smem[ROW_E - _SPLIT, :] = de
                    dga_smem[ROW_QS - _SPLIT, :] = dqs
                    dga_smem[ROW_DEC - _SPLIT, :] = dqs * 0.0  # the dec cotangent is summed from dd_ref
                    dd_smem[...] = ddec_rows
                    plgpu.commit_smem()
                    store(dq_smem, dq_ref, n)
                    plgpu.copy_smem_to_gmem(dga_smem, dg_ref.at[b, h, n, rows_a], commit_group=False)
                    plgpu.copy_smem_to_gmem(dd_smem, dd_ref.at[b, h, n], commit_group=False)
                    plgpu.commit_smem_to_gmem_group()
                return dh

            dh = lax.fori_loop(0, nc, step, _load(df_ref.at[b, h]))
            dh0_ref.at[b, h][...] = dh
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == _WG_B)
        def _main():
            """dV = Wv^T dO + w (K dH'), dK = Wq^T Q + w (V dH'^T) and the A, c, gamma, w rows."""
            plgpu.set_max_registers(_BWD_REGISTERS[_WG_B], action="increase")
            # Transposed orientation: row j is the key/value row, column i the query row.
            tok_j, tok_i = _token_of(_iota(sq, 0), ct), _token_of(_iota(sq, 1), ct)
            same_token = tok_j == tok_i
            later = tok_i > tok_j

            def step(it, carry):
                n, slot, buf = nc - 1 - it, lax.rem(it, st), lax.rem(it, 2)
                with jax.named_scope("dmain_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                gs = g_smem.at[slot]
                c_row = _load(gs.at[ROW_C], _ROW)

                def weights():
                    mask = later
                    if cfg.segments:
                        same = _rows(_load(gs.at[ROW_RUN], _ROW), sq) == _cols(_load(gs.at[ROW_RUN_COL], _COL), sq)
                        mask = jnp.logical_and(mask, same)
                    exponent = _cols(_load(gs.at[ROW_E], _COL), sq) - _rows(_load(gs.at[ROW_A], _ROW), sq)
                    decay = _masked_exp(exponent, mask)
                    gamma = _rows(_load(gs.at[ROW_GAMMA], _ROW), sq)
                    return decay, jnp.where(same_token, gamma, decay * _rows(c_row, sq))

                with jax.named_scope("dmain_scores"):
                    p_t = _matmul(cfg, sq, (k_smem.at[slot], q_smem.at[slot].T))  # P^T[j, i] = k_j . q_i
                    dp_t, (decay_t, weight_t) = _matmul_while(
                        cfg, sq, ((v_smem.at[slot], do_smem.at[slot].T),), weights
                    )  # dP^T[j, i] = v_j . dO_i
                with jax.named_scope("dmain_weights"):
                    both = p_t * dp_t
                    dc = (decay_t * both).sum(axis=1)
                    dgamma = jnp.where(same_token, both, 0.0).sum(axis=1)
                    wv_t = (p_t * weight_t).astype(_BF16)
                    wq_t = (dp_t * weight_t).astype(_BF16)
                with jax.named_scope("dmain_wait_free"):

                    @pl.when(it >= 2)
                    def _():
                        plgpu.barrier_wait(wq_free.at[buf])  # warpgroup A's dq two chunks ago is done

                with jax.named_scope("dmain_publish"):
                    wqt_smem.at[buf][...] = wq_t
                    plgpu.commit_smem()
                    plgpu.barrier_arrive(wq_ready.at[buf])
                w = _load(gs.at[ROW_W], _ROW)
                with jax.named_scope("dmain_wait_dh"):
                    plgpu.barrier_wait(dh_ready)
                with jax.named_scope("dmain_dv"):
                    kdh = _matmul(cfg, (c, vd), (k_smem.at[slot], dh_smem))
                    dw = (kdh * _load(v_smem.at[slot]).astype(_F32)).sum(axis=1)
                    dv = _matmul(
                        cfg, (c, vd), (_Regs(wv_t), do_smem.at[slot]), init=kdh * _rows(w, (c, vd)), stage=stage
                    )
                with jax.named_scope("dmain_dk"):
                    vdh = _matmul(cfg, (c, kd), (v_smem.at[slot], dh_smem.T))

                    @pl.when(it + 1 < nc)
                    def _():
                        plgpu.barrier_arrive(dh_free)

                    dk = _matmul(
                        cfg, (c, kd), (_Regs(wq_t), q_smem.at[slot]), init=vdh * _rows(w, (c, kd)), stage=stage
                    )
                plgpu.barrier_arrive(consumed.at[slot])
                with jax.named_scope("dmain_store"):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    dk_smem[...] = dk.astype(_BF16)
                    dv_smem[...] = dv.astype(_BF16)
                    dgb_smem[ROW_A, :] = -(c_row * dc)
                    dgb_smem[ROW_C, :] = dc
                    dgb_smem[ROW_GAMMA, :] = dgamma
                    dgb_smem[ROW_W, :] = dw
                    # Mosaic stores constant (splat) vectors only with 1 or 128k elements: zero a row vector.
                    dgb_smem[ROW_RUN, :] = dc * 0.0
                    dgb_smem[ROW_RUN_COL, :] = dc * 0.0
                    plgpu.commit_smem()
                    store(dk_smem, dk_ref, n)
                    store(dv_smem, dv_ref, n)
                    plgpu.copy_smem_to_gmem(dgb_smem, dg_ref.at[b, h, n, rows_b], commit_group=False)
                    plgpu.commit_smem_to_gmem_group()
                return carry

            lax.fori_loop(0, nc, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])

    scratch = dict(
        done=plgpu.Barrier(num_arrivals=3),
        q=_operand_smem((st, c, kd)),
        k=_operand_smem((st, c, kd)),
        v=_operand_smem((st, c, vd)),
        do=_operand_smem((st, c, vd)),
        g=plgpu.SMEM((st, NUM_ROWS, c), _F32),
        h=_operand_smem((st, kd, vd)),
        dh=_operand_smem((kd, vd)),
        qsdo=_operand_smem((c, vd)),
        wqt=_operand_smem((2, c, c)),
        dq=_operand_smem((c, kd)),
        dk=_operand_smem((c, kd)),
        dv=_operand_smem((c, vd)),
        dga=plgpu.SMEM((NUM_ROWS - _SPLIT, c), _F32),
        dd=plgpu.SMEM((kd,), _F32),
        dgb=plgpu.SMEM((_SPLIT, c), _F32),
        data=plgpu.Barrier(num_arrivals=4 * rank + 2, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=2, num_barriers=st),
        dh_ready=plgpu.Barrier(num_arrivals=1),
        dh_free=plgpu.Barrier(num_arrivals=1),
        wq_ready=plgpu.Barrier(num_arrivals=1, num_barriers=2),
        wq_free=plgpu.Barrier(num_arrivals=1, num_barriers=2),
    )
    if cfg.interpret:
        scratch["stage"] = _operand_smem((c, c))
    kern = _kernel(
        cfg, "mamba3_mimo_bwd", body, num_threads=3,
        out_type=(shape_k, shape_k, shape_v, shape_g, shape_s, shape_d), scratch_types=scratch,
    )
    dq, dk, dv, dgates, dh0, dec_rows = kern(q, k, v, do, gates, states, dfinal)
    dgates = dgates.at[:, :, :, ROW_DEC, 0].add(dec_rows.sum(axis=-1))
    # The same-token block of the E cotangent: per token, gamma times the block's sum of P dP,
    # which warpgroup B already summed per key row into the gamma cotangent.
    dgates = dgates.at[:, :, :, ROW_E].add(-gates[:, :, :, ROW_GAMMA] * dgates[:, :, :, ROW_GAMMA])
    return dq, dk, dv, dgates, dh0


# ---------------------------------------------------------------------------------------------
# custom_vjp core and XLA glue
# ---------------------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _core(cfg: Config, q: Array, k: Array, v: Array, gates: Array, h0: Array) -> tuple[Array, Array]:
    o, final = forward(cfg, q, k, v, gates, h0, save_states=False)
    return o, final


def _core_fwd(cfg: Config, q, k, v, gates, h0):
    o, final, states = forward(cfg, q, k, v, gates, h0, save_states=True)
    return (o, final), (q, k, v, gates, states)


def _core_bwd(cfg: Config, residuals, cotangents):
    q, k, v, gates, states = residuals
    do, dfinal = cotangents
    do = (do.astype(_F32) * cfg.scale).astype(_BF16)
    return backward(cfg, q, k, v, do, gates, states, dfinal.astype(_F32))


_core.defvjp(_core_fwd, _core_bwd)


def _token_chunks(x: Array, num_chunks: int, tokens: int) -> Array:
    """``[B, T, H]`` to ``[B, H, NC, tokens]``, zero padded at the end."""
    pad = num_chunks * tokens - x.shape[1]
    x = jnp.pad(x, [(0, 0), (0, pad), (0, 0)])
    return jnp.moveaxis(x, 1, 2).reshape(x.shape[0], x.shape[2], num_chunks, tokens)


def prepare_gates(
    g: Array, beta: Array, gamma: Array, *, rank: int, num_chunks: int, segment_ids: Array | None = None
) -> Array:
    """Per-chunk gate rows ``[B, H, NC, NUM_ROWS, 64]`` float32 over chunks of ``64 / rank`` tokens.

    The rows of :func:`jla.ops.mosaic_gpu.mamba3.prepare_gates` computed per token and tiled over the
    ranks (column ``r C + i`` is token ``i``); the chunk decay sits in column 0 of its row.
    ``g, beta, gamma`` are ``[B, T, H]``; ``segment_ids [B, T]`` adds the packed-document masks.
    """
    tokens = CHUNK // rank
    g, beta, gamma = (x.astype(_F32) for x in (g, beta, gamma))
    batch, seq_len, _ = g.shape
    end = jnp.zeros_like(g[:, :1])
    g_next = jnp.concatenate([g[:, 1:], end], axis=1)
    beta_next = jnp.concatenate([beta[:, 1:], end], axis=1)
    a = _token_chunks(g_next, num_chunks, tokens)
    cw = _token_chunks(gamma * jnp.exp(g_next) + beta_next, num_chunks, tokens)
    gam = _token_chunks(gamma, num_chunks, tokens)
    big_a = jnp.cumsum(a, axis=-1)
    big_e = big_a - a
    last = big_a[..., -1:]
    qs = jnp.exp(big_e)
    w = cw * jnp.exp(last - big_a)
    dec = jnp.exp(last)
    if segment_ids is None:
        run = jnp.zeros_like(a)
    else:
        if segment_ids.shape != (batch, seq_len):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
        runs = jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)
        # Padding joins the last run; one more position gives every chunk the run of the next token.
        runs = jnp.pad(runs, [(0, 0), (0, num_chunks * tokens + 1 - seq_len)], mode="edge")
        nxt = runs[:, tokens::tokens]  # [B, NC]: run of the first token after each chunk
        runs = runs[:, : num_chunks * tokens].reshape(batch, num_chunks, tokens)
        carry = (runs == runs[..., :1]).astype(_F32)[:, None]
        tail = (runs == nxt[..., None]).astype(_F32)[:, None]
        keep = (runs[..., :1] == nxt[..., None]).astype(_F32)[:, None]
        qs, w, dec = qs * carry, w * tail, dec * keep
        # Chunk-local run ids: the kernels compare them only within a chunk, and float32 holds the global
        # ids exactly only below 2**24 documents.
        run = jnp.broadcast_to((runs - runs[..., :1]).astype(_F32)[:, None], a.shape)
    tile = functools.partial(jnp.tile, reps=(1, 1, 1, rank))
    dec = jnp.pad(dec, [(0, 0), (0, 0), (0, 0), (0, CHUNK - 1)])
    rows = {ROW_A: tile(big_a), ROW_C: tile(cw), ROW_GAMMA: tile(gam), ROW_W: tile(w), ROW_RUN: tile(run),
            ROW_RUN_COL: tile(run), ROW_E: tile(big_e), ROW_QS: tile(qs), ROW_DEC: dec}
    return jnp.stack([rows[r] for r in range(NUM_ROWS)], axis=-2)


def _initial_mid_state(initial_state: Mamba3State, g: Array, beta: Array) -> Array:
    """``H_0 = exp(g_0) S_0 + beta_0 sum_r k_{0,r}^T v_{0,r}`` (``[B, H, K, V]`` float32)."""
    ssm, key, value = (x.astype(_F32) for x in initial_state)
    g0, beta0 = g[:, 0].astype(_F32), beta[:, 0].astype(_F32)
    outer = jnp.einsum("bhrk,bhrv->bhkv", key, value, precision=_HIGHEST)
    return jnp.exp(g0)[..., None, None] * ssm + beta0[..., None, None] * outer


def _state_shapes(q: Array, v: Array) -> tuple[tuple[int, ...], ...]:
    batch, _, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    return (batch, heads, key_dim, value_dim), (batch, heads, rank, key_dim), (batch, heads, rank, value_dim)


def _check(q, k, v, g, beta, gamma, initial_state, segment_ids, chunk_size) -> None:
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4]:
        raise ValueError(f"mamba3_mimo needs q, k [B, T, H, R, K] and v [B, T, H, R, V]; got {q.shape}, "
                         f"{k.shape}, {v.shape}")
    for name, x in (("g", g), ("beta", beta), ("gamma", gamma)):
        if x.shape != q.shape[:3]:
            raise ValueError(f"{name} must be [B, T, H] = {q.shape[:3]}; got {x.shape}")
    _, seq_len, _, rank, key_dim = q.shape
    if rank not in RANKS:
        raise ValueError(f"the mosaic_gpu mamba3_mimo kernel supports ranks {RANKS}; got {rank}")
    if (key_dim, v.shape[-1]) not in SUPPORTED_DIMS:
        raise ValueError(f"mosaic_gpu mamba3_mimo supports (K, V) in {SUPPORTED_DIMS}; got ({key_dim}, {v.shape[-1]})")
    if chunk_size is not None and chunk_size != CHUNK // rank:
        raise ValueError(f"the mosaic_gpu mamba3_mimo kernel uses chunks of {CHUNK // rank} tokens at rank {rank} "
                         f"(64 rows); got chunk_size={chunk_size}")
    if seq_len == 0:
        raise ValueError("mosaic_gpu mamba3_mimo needs at least one token")
    if initial_state is not None:
        got = tuple(x.shape for x in initial_state)
        if got != _state_shapes(q, v):
            raise ValueError(f"initial_state shapes must be {_state_shapes(q, v)}; got {got}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")


def mamba3_mimo(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    gamma: Array,
    *,
    scale: float = 1.0,
    initial_state: Mamba3State | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
    chunk_size: int | None = None,
    stages: int | None = None,
    interpret: bool = False,
) -> tuple[Array, Mamba3State | None]:
    """Mamba-3 MIMO on the dedicated kernels; the arguments of :func:`jla.ops.reference.mamba3_mimo`.

    ``chunk_size`` (tokens per chunk) must be ``64 / R`` or None; ``stages`` (1 to 4) is the TMA
    pipeline depth over chunks, by default 4 when at most 64 CTAs (``B * H``) run, else 2.
    ``interpret`` runs the kernels in the GPU interpreter. ``q, k, v`` are computed in bfloat16.
    """
    _check(q, k, v, g, beta, gamma, initial_state, segment_ids, chunk_size)
    batch, seq_len, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    if stages is None:
        stages = 4 if batch * heads <= 64 else 2  # a deeper TMA ring pays off when few CTAs run
    if not 1 <= stages <= 4:
        raise ValueError(f"stages must be in [1, 4]; got {stages}")
    tokens = CHUNK // rank
    num_chunks = -(-seq_len // tokens)
    cfg = Config(
        batch=batch, heads=heads, rank=rank, num_chunks=num_chunks, key_dim=key_dim, value_dim=value_dim,
        scale=float(scale), segments=segment_ids is not None, stages=min(stages, num_chunks), interpret=interpret,
    )
    gates = prepare_gates(g, beta, gamma, rank=rank, num_chunks=num_chunks, segment_ids=segment_ids)
    if initial_state is None:
        h0 = jnp.zeros((batch, heads, key_dim, value_dim), _F32)
    else:
        h0 = _initial_mid_state(initial_state, g, beta)
    pad = [(0, 0), (0, cfg.seq_len - seq_len), (0, 0), (0, 0)]

    def flat(x):  # [B, T, H, R, D] -> [B, T_pad, H R, D] (a free reshape, then the chunk padding)
        return jnp.pad(x.astype(_BF16).reshape(batch, seq_len, heads * rank, x.shape[-1]), pad)

    o, final = _core(cfg, flat(q), flat(k), flat(v), gates, h0)
    o = o[:, :seq_len].reshape(batch, seq_len, heads, rank, value_dim).astype(q.dtype)
    if not output_final_state:
        return o, None
    return o, Mamba3State(final, k[:, -1].astype(_F32), v[:, -1].astype(_F32))


def supports(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    gamma: Array,
    *,
    scale: float = 1.0,
    initial_state: Mamba3State | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
    chunk_size: int | None = None,
    cu_seqlens: Array | None = None,
    precision: Any = None,
    stages: int | None = None,
    interpret: bool = False,
    **unknown: Any,
) -> bool:
    """True only for calls the kernels are tested on: H100, bf16 ``q, k, v [B, T, H, R, D]`` with
    ``R`` in :data:`RANKS` and ``(K, V)`` in :data:`SUPPORTED_DIMS`, float gates, optional
    ``segment_ids`` and ``initial_state``, chunks of ``64 / R`` tokens."""
    del output_final_state
    if unknown or cu_seqlens is not None or precision is not None:
        return False
    if not isinstance(scale, (int, float)) or (stages is not None and not 1 <= stages <= 4):
        return False
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4] or q.shape[1] == 0:
        return False
    rank = q.shape[3]
    if rank not in RANKS or (chunk_size is not None and chunk_size != CHUNK // rank):
        return False
    if (q.shape[-1], v.shape[-1]) not in SUPPORTED_DIMS or any(x.dtype != _BF16 for x in (q, k, v)):
        return False
    if any(x.shape != q.shape[:3] or not jnp.issubdtype(x.dtype, jnp.floating) for x in (g, beta, gamma)):
        return False
    if segment_ids is not None and (
        segment_ids.shape != q.shape[:2] or not jnp.issubdtype(segment_ids.dtype, jnp.integer)
    ):
        return False
    if initial_state is not None:
        if not isinstance(initial_state, Mamba3State) or tuple(x.shape for x in initial_state) != _state_shapes(q, v):
            return False
        if not all(jnp.issubdtype(x.dtype, jnp.floating) for x in initial_state):
            return False
    return interpret or _on_hopper()
