# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py,
# mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py, mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py
"""Mosaic GPU (H100) kernels for the Mamba-3 SISO recurrence (``jla.ops`` kind ``mamba3``).

``S_t = exp(g_t) S_{t-1} + beta_t k_{t-1}^T v_{t-1} + gamma_t k_t^T v_t`` and ``o_t = scale q_t S_t``
(``jla.ops.reference.mamba3``), with ``q``/``k`` the rotated, biased ``C``/``B``.

Algorithm (the official ``mamba3_siso_fwd`` / ``mamba3_siso_bwd`` structure): one CTA per
``(batch, head)`` walks the sequence in chunks of 64 tokens with the ``[K, V]`` state in a
float32 WGMMA accumulator, forward in the forward kernel and in reverse in the backward kernel
(which reads the per-chunk states that the forward kernel saved, as the official kernel does).
Each CTA is warp specialized: a memory warpgroup keeps a TMA ring of chunks in flight; in the
forward kernel one compute warpgroup carries the state and hands each chunk's entering state to
two output warpgroups that take alternate chunks; in the backward kernel one carries the state
cotangent and computes ``dq``, the other the score matrices, ``dk`` and ``dv``.
The trapezoidal previous-token term is folded into one score matrix per chunk: the official
kernel scales each key by ``gamma_t + shifted_gamma_t`` (``shifted_gamma_t = dt_{t+1}
(1 - lambda_{t+1})``), keeps the scores strictly causal, adds the diagonal through the ``q . k``
dot and folds the previous key/value of the initial state into the incoming state. This op
receives ``beta = (1 - lambda) dt exp(g)`` instead of ``dt`` and ``lambda``, so the shifted weight
would be ``beta_{t+1} / exp(g_{t+1})``, which overflows for strong decays. We move the decay's
reference point by one token instead; it is the same algebra without the division. With
``S~_t = exp(g_{t+1}) S_t + beta_{t+1} k_t^T v_t`` (the state between tokens ``t`` and ``t+1``):

    S~_t = exp(a_t) S~_{t-1} + c_t k_t^T v_t,    a_t = g_{t+1},  c_t = gamma_t exp(g_{t+1}) + beta_{t+1}
    o_t  = scale (q_t S~_{t-1} + gamma_t (q_t . k_t) v_t)

so ``S~`` is a simple-GLA state with decay ``a`` and values ``c v``; ``o`` reads it exclusively and
adds the diagonal (``gamma``) term. Inside a chunk, with ``A`` / ``E`` the inclusive / exclusive
chunk-local cumulative sums of ``a`` and ``H`` the ``S~`` state entering the chunk:

    o_i = qs_i q_i H + sum_{j<i} c_j exp(E_i - A_j) (q_i . k_j) v_j + gamma_i (q_i . k_i) v_i
    H'  = dec H + sum_j w_j k_j^T v_j

with ``qs_i = exp(E_i)``, ``w_j = c_j exp(A_last - A_j)`` and ``dec = exp(A_last)``; every exponent
is ``<= 0``. The first state is ``H_0 = exp(g_0) S_0 + beta_0 k_0^T v_0`` from the initial
``Mamba3State`` ``(S_0, k_0, v_0)``, and the last ``S~`` (with ``g_T = beta_T = 0``) is the final
state ``S_{T-1}``. Packed documents (``segment_ids``) zero ``qs`` before a reset inside the chunk,
``w`` and ``dec`` across a reset at the chunk end, and mask the scores to equal segment runs.

XLA computes the per-token gate rows (``E, A, c, gamma, qs, w``, the segment run and ``dec``) and
``H_0``; JAX autodiff maps their gradients back to ``g``, ``beta``, ``gamma`` and the initial state.
The kernels themselves sit inside one ``jax.custom_vjp``. Inputs are bfloat16 on the tensor cores
with float32 accumulation; gates and decays are float32. ``(K, V)`` in :data:`SUPPORTED_DIMS`;
any ``T`` (padded to the chunk; zero padding is inert).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla import ops
from jla.ops.mosaic_gpu._common import finish_together
from jla.ops.reference import Mamba3State, segment_resets

__all__ = ["CHUNK", "mamba3", "prepare_gates", "supports"]

Array = jax.Array

CHUNK = 64
_F32 = jnp.float32
_BF16 = jnp.bfloat16
_WG = plgpu.Layout.WGMMA
_ROW = plgpu.Layout.WGMMA.reduce(1)  # one value per row of a WGMMA-layout array
_COL = plgpu.Layout.WGMMA.reduce(0)  # one value per column

# Rows of the per-chunk gate tile [NUM_ROWS, CHUNK] (float32). The segment run is stored twice:
# Mosaic GPU's layout inference fails when one SMEM vector is loaded both per row and per column.
# The backward kernel's two compute warpgroups write the cotangent rows [0, _SPLIT) and
# [_SPLIT, NUM_ROWS) respectively.
ROW_A, ROW_C, ROW_GAMMA, ROW_W, ROW_RUN, ROW_RUN_COL, ROW_E, ROW_QS, ROW_DEC = range(9)
NUM_ROWS = 9
_SPLIT = ROW_E

# (key_dim, value_dim) pairs tested on the H100. K = V = 128 would need a [128, 128] float32
# state accumulator (128 registers per thread) next to the backward's other accumulators.
SUPPORTED_DIMS = ((128, 64), (64, 128), (64, 64))
# Shared memory per CTA on H100 (232448 bytes) minus barriers and, for the backward kernel (whose
# row reductions reserve it), the 2 KB reduction scratch.
_FWD_SMEM_BYTES = 232448 - 1024
_BWD_SMEM_BYTES = 232448 - 1024 - 2048

# Warpgroups (Pallas threads). Forward: two output warpgroups (even and odd chunks), the state
# warpgroup and the TMA warpgroup, with per-warpgroup register budgets that sum to at most 512
# (64K registers over 128-thread warpgroups). Backward: warpgroups A, B and the TMA warpgroup.
_FWD_STATE, _FWD_MEMORY = 2, 3
_FWD_REGISTERS = (168, 168, 152, 24)
_WG_A, _WG_B, _WG_MEMORY = 0, 1, 2
_BWD_REGISTERS = (232, 232, 40)


def _forward_smem(stages: int, kd: int, vd: int) -> int:
    per_stage = 2 * CHUNK * kd * 2 + CHUNK * vd * 2 + NUM_ROWS * CHUNK * 4  # q, k, v, gates
    return stages * per_stage + 2 * kd * vd * 2 + 3 * CHUNK * vd * 2  # 2 states, w * v, 2 o tiles


def _backward_smem(stages: int, kd: int, vd: int) -> int:
    # q, k, v, dO, gates and the entering state per stage
    per_stage = 2 * CHUNK * kd * 2 + 2 * CHUNK * vd * 2 + NUM_ROWS * CHUNK * 4 + kd * vd * 2
    fixed = (
        kd * vd * 2  # dH'
        + CHUNK * vd * 2  # qs * dO
        + 2 * CHUNK * CHUNK * 2  # two (dP * W)^T tiles
        + 2 * CHUNK * kd * 2 + CHUNK * vd * 2  # dq, dk, dv staging
        + NUM_ROWS * CHUNK * 4 + kd * 4  # gate cotangents, row sums of the dec cotangent
    )
    return stages * per_stage + fixed


def _fitting_stages(requested: int, smem, budget: int, kd: int, vd: int, *, even: bool = False) -> int:
    """The deepest pipeline up to ``requested`` stages whose buffers fit in ``budget`` bytes.

    ``even``: the forward kernel's two output warpgroups each own every other slot.
    """
    candidates = range(max(requested, 2) // 2 * 2, 0, -2) if even else range(requested, 0, -1)
    for stages in candidates:
        if smem(stages, kd, vd) <= budget:
            return stages
    raise ValueError(f"mamba3 kernel buffers for K={kd}, V={vd} do not fit in shared memory")


@dataclasses.dataclass(frozen=True)
class Config:
    """Static shapes and options of one kernel call."""

    batch: int
    heads: int
    num_chunks: int
    key_dim: int
    value_dim: int
    scale: float = 1.0
    segments: bool = False  # mask the scores to equal segment runs
    stages: int = 4  # TMA pipeline depth over chunks (capped by shared memory; even in the forward)
    interpret: bool = False  # run the kernels in the GPU interpreter (CPU tests)
    profile_dir: str = ""  # developer option: write Mosaic GPU's per-warpgroup trace of the named phases here

    @property
    def seq_len(self) -> int:
        return self.num_chunks * CHUNK


# ---------------------------------------------------------------------------------------------
# kernel helpers
# ---------------------------------------------------------------------------------------------


def _kernel(cfg: Config, name: str, body, *, num_threads: int, **kwargs):
    """One CTA per (head, batch), warp specialized into ``num_threads`` warpgroups."""
    interpret = plgpu.InterpretGPUParams() if cfg.interpret else None
    profile = {}
    if cfg.profile_dir:  # each chunk records at most 11 named phases per warpgroup
        profile = dict(profile_space=12 * cfg.num_chunks + 16, profile_dir=cfg.profile_dir)
    return plgpu.kernel(
        body,
        grid=(cfg.heads, cfg.batch),
        grid_names=("h", "b"),
        num_threads=num_threads,
        thread_name="wg",
        kernel_name=name,
        compiler_params=plgpu.CompilerParams(approx_math=True, **profile),
        interpret=interpret,
        **kwargs,
    )


def _memory_warpgroup(fetch, consumed, num_chunks: int, stages: int, chunk_of, registers: int) -> None:
    """Issue the TMA loads of every chunk, refilling a slot once its consumers released it.

    ``chunk_of(step)`` is the chunk loaded at pipeline step ``step``; every release is awaited.
    """
    plgpu.set_max_registers(registers, action="decrease")
    for i in range(stages):
        fetch(chunk_of(i), i)

    @pl.loop(0, num_chunks)
    def _(step):
        slot = lax.rem(step, stages)
        plgpu.barrier_wait(consumed.at[slot])

        @pl.when(step + stages < num_chunks)
        def _():
            fetch(chunk_of(step + stages), slot)


def _operand_smem(shape: tuple[int, ...]) -> plgpu.SMEM:
    """A bf16 SMEM buffer laid out for WGMMA operands and TMA (tiled and swizzled)."""
    swizzle = plgpu.find_swizzle(shape[-1] * 16)
    transforms = (plgpu.TilingTransform((8, swizzle // 2)), plgpu.SwizzleTransform(swizzle))
    return plgpu.SMEM(shape, _BF16, transforms=transforms)


def _iota(shape: tuple[int, int], dim: int) -> Array:
    return plgpu.broadcasted_iota(jnp.int32, shape, dim, layout=_WG)


def _load(ref, layout=_WG) -> Array:
    return plgpu.load(ref, layout=layout, optimized=False)


def _rows(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-row vector (``_ROW`` layout) along the columns."""
    return lax.broadcast_in_dim(x, shape, [0])


def _cols(x: Array, shape: tuple[int, int]) -> Array:
    """Broadcast a per-column vector (``_COL`` layout) along the rows."""
    return lax.broadcast_in_dim(x, shape, [1])


def _masked_exp(exponent: Array, mask: Array) -> Array:
    """``exp`` on ``mask``, zero elsewhere, never evaluating ``exp`` of a masked (possibly large) value."""
    return jnp.exp(jnp.where(mask, exponent, -jnp.inf))


class _Regs(NamedTuple):
    """A bf16 WGMMA-layout register array used as the left operand of a matmul."""

    value: Array


def _matmul(cfg: Config, shape: tuple[int, int], *pairs, init: Array | None = None, stage=None) -> Array:
    """``init + sum(a @ b for a, b in pairs)`` on the tensor cores with a float32 accumulator.

    ``a`` is an SMEM ref (possibly a transposed view) or :class:`_Regs`; ``b`` is an SMEM ref.
    ``stage`` is a bf16 SMEM buffer for register operands, needed only in the GPU interpreter
    (whose wgmma reads both operands from SMEM).
    """
    if cfg.interpret:
        staged = []
        for a, b in pairs:
            if isinstance(a, _Regs):
                stage[...] = a.value
                plgpu.commit_smem()
                a = stage
            staged.append((a, b))
        # The interpreter cannot run ``pl.run_state`` on an initialised accumulator.
        product = _wgmma_scoped(shape, staged, lambda: None)[0]
        return product if init is None else init + product
    pairs = tuple((a.value if isinstance(a, _Regs) else a, b) for a, b in pairs)
    if init is None:
        return _wgmma_scoped(shape, pairs, lambda: None)[0]

    def body(acc):
        for a, b in pairs:
            plgpu.wgmma(acc, a, b)

    return pl.run_state(body)(plgpu.ACC.init(init))


def _wgmma_scoped(shape: tuple[int, int], pairs, work):
    def scoped(acc):
        for a, b in pairs:
            plgpu.wgmma(acc, a, b)
        out = work()  # runs while the tensor cores work
        return acc[...], out

    return pl.run_scoped(scoped, plgpu.ACC(shape, _F32))


def _matmul_while(cfg: Config, shape: tuple[int, int], pairs, work):
    """``(sum(a @ b for a, b in pairs), work())``, with ``work`` (register math only) issued while
    the WGMMAs are in flight. ``a`` must be an SMEM ref."""
    if cfg.interpret:
        return _matmul(cfg, shape, *pairs), work()
    return _wgmma_scoped(shape, pairs, work)


# ---------------------------------------------------------------------------------------------
# forward kernel
# ---------------------------------------------------------------------------------------------


def forward(
    cfg: Config, q: Array, k: Array, v: Array, gates: Array, h0: Array, *, save_states: bool
) -> tuple[Array, ...]:
    """``(o, final[, states])`` of the chunked recurrence.

    ``q, k [B, T, H, K]``, ``v [B, T, H, V]`` bf16 with ``T = cfg.seq_len``; ``gates``
    ``[B, H, NC, NUM_ROWS, CHUNK]`` f32 (:func:`prepare_gates`); ``h0 [B, H, K, V]`` f32.
    Returns ``o [B, T, H, V]`` bf16 (times ``cfg.scale``), the final ``S~`` state ``[B, H, K, V]``
    f32 and, with ``save_states``, the bf16 state entering each chunk ``[B, H, NC, K, V]``.
    """
    c, kd, vd, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.num_chunks
    # Output warpgroup o handles the chunks n with n % 2 == o, and every slot of the TMA ring belongs
    # to one of them, so the depth is even (or 1 for a single chunk).
    st = 1 if nc == 1 else _fitting_stages(
        min(cfg.stages, nc), _forward_smem, _FWD_SMEM_BYTES, kd, vd, even=True
    )
    out_types = [
        jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads, vd), _BF16),
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
            tok = pl.ds(n * c, c)
            bar = data.at[slot]
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h], v_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, h, n], g_smem.at[slot], bar)

        @pl.when(wg == _FWD_MEMORY)
        def _():
            _memory_warpgroup(fetch, consumed, nc, st, lambda step: step, _FWD_REGISTERS[_FWD_MEMORY])

        def output(o_wg: int):
            """Outputs o = qs * (q H) + (scores * weights) v of the chunks n = o_wg, o_wg + 2, ..."""
            plgpu.set_max_registers(_FWD_REGISTERS[o_wg], action="increase")
            sq = (c, c)
            rows_i, cols_j = _iota(sq, 0), _iota(sq, 1)
            diag = rows_i == cols_j
            lower = cols_j < rows_i
            o_tile = o_smem.at[o_wg]
            staging = None if stage is None else stage.at[o_wg]

            def step(j, carry):
                n = 2 * j + o_wg
                slot = lax.rem(n, st)
                with jax.named_scope("out_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                gs = g_smem.at[slot]

                def weights():
                    mask = lower
                    if cfg.segments:
                        same = _rows(_load(gs.at[ROW_RUN], _ROW), sq) == _cols(_load(gs.at[ROW_RUN_COL], _COL), sq)
                        mask = jnp.logical_and(mask, same)
                    exponent = _rows(_load(gs.at[ROW_E], _ROW), sq) - _cols(_load(gs.at[ROW_A], _COL), sq)
                    decay = _masked_exp(exponent, mask) * _cols(_load(gs.at[ROW_C], _COL), sq)
                    return jnp.where(diag, _rows(_load(gs.at[ROW_GAMMA], _ROW), sq), decay)

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
                    plgpu.copy_smem_to_gmem(o_tile, o_ref.at[b, pl.ds(n * c, c), h])
                return carry

            lax.fori_loop(0, (nc - o_wg + 1) // 2, step, None)
            plgpu.wait_smem_to_gmem(0)

        for o_wg in (0, 1):
            pl.when(wg == o_wg)(functools.partial(output, o_wg))

        @pl.when(wg == _FWD_STATE)
        def _state():
            """The serial state pass H' = dec H + K^T (w v); publishes each chunk's entering state."""
            plgpu.set_max_registers(_FWD_REGISTERS[_FWD_STATE], action="increase")

            def publish(s, n, buf):
                s_smem.at[buf][...] = s.astype(_BF16)
                plgpu.commit_smem()
                plgpu.barrier_arrive(state_ready.at[buf])
                if save_states:
                    plgpu.copy_smem_to_gmem(s_smem.at[buf], st_ref.at[b, h, n])

            s0 = _load(h0_ref.at[b, h])  # once per CTA: a plain global load, no SMEM staging
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
        data=plgpu.Barrier(num_arrivals=4, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=2, num_barriers=st),
        state_ready=plgpu.Barrier(num_arrivals=1, num_barriers=2),
        state_free=plgpu.Barrier(num_arrivals=1, num_barriers=2),
    )
    if cfg.interpret:
        scratch["stage"] = _operand_smem((2, c, c))
    kern = _kernel(cfg, "mamba3_fwd", body, num_threads=4, out_type=tuple(out_types), scratch_types=scratch)
    return kern(q, k, v, gates, h0)


# ---------------------------------------------------------------------------------------------
# backward kernel
# ---------------------------------------------------------------------------------------------


def backward(
    cfg: Config, q: Array, k: Array, v: Array, do: Array, gates: Array, states: Array, dfinal: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """``(dq, dk, dv, dgates, dh0)`` for the forward kernel's ``(o, final)`` cotangents.

    ``do [B, T, H, V]`` bf16 (already multiplied by ``cfg.scale``), ``states`` from
    :func:`forward`, ``dfinal [B, H, K, V]`` f32. ``dgates`` has the layout of ``gates``: the
    cotangents of the rows ``E, A, c, gamma, qs, w`` and of ``dec`` (column 0); the run rows are zero.

    Warpgroup A carries the state cotangent ``dH`` (float32 registers) backwards over the chunks and
    computes ``dq`` and the ``E``, ``qs``, ``dec`` cotangents; warpgroup B computes the score
    matrices, ``dk``, ``dv`` and the other gate cotangents. Both reduce only along rows (within a
    warp): the ``dec`` cotangent ``<H, dH'>`` leaves the kernel as per-row partial sums that XLA
    adds up, because Mosaic GPU gives every warpgroup the same cross-warp reduction scratch.
    """
    c, kd, vd, nc = CHUNK, cfg.key_dim, cfg.value_dim, cfg.num_chunks
    st = _fitting_stages(min(cfg.stages, nc), _backward_smem, _BWD_SMEM_BYTES, kd, vd)
    shape_k = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads, kd), _BF16)
    shape_v = jax.ShapeDtypeStruct((cfg.batch, cfg.seq_len, cfg.heads, vd), _BF16)
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
            tok = pl.ds(n * c, c)
            bar = data.at[slot]
            plgpu.copy_gmem_to_smem(q_ref.at[b, tok, h], q_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(k_ref.at[b, tok, h], k_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(v_ref.at[b, tok, h], v_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, tok, h], do_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, h, n], g_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(st_ref.at[b, h, n], h_smem.at[slot], bar)

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
                gamma = _load(gs.at[ROW_GAMMA], _ROW)
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
                    dp_diag = (dov * _load(v_smem.at[slot]).astype(_F32)).sum(axis=1)  # dO_i . v_i
                    plgpu.commit_smem()
                    dh = _matmul(cfg, (kd, vd), (q_smem.at[slot].T, qsdo_smem), init=dh * dec)
                with jax.named_scope("dstate_doh"):
                    doh = _matmul(cfg, (c, kd), (do_smem.at[slot], h_smem.at[slot].T))
                    qf = _load(q_smem.at[slot]).astype(_F32)
                    dqs = (doh * qf).sum(axis=1)
                    p_diag = (qf * _load(k_smem.at[slot]).astype(_F32)).sum(axis=1)  # q_i . k_i
                with jax.named_scope("dstate_wait_wq"):
                    plgpu.barrier_wait(wq_ready.at[buf])
                with jax.named_scope("dstate_dq"):
                    dq = _matmul(cfg, (c, kd), (wqt_smem.at[buf].T, k_smem.at[slot]), init=doh * _rows(qs, (c, kd)))

                    @pl.when(it + 2 < nc)
                    def _():
                        plgpu.barrier_arrive(wq_free.at[buf])

                    # The intra-chunk E cotangent is q . dq without its qs and diagonal (gamma) parts.
                    de = (dq * _load(q_smem.at[slot]).astype(_F32)).sum(axis=1) - qs * dqs - gamma * p_diag * dp_diag
                plgpu.barrier_arrive(consumed.at[slot])
                with jax.named_scope("dstate_store"):
                    plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                    dq_smem[...] = dq.astype(_BF16)
                    dga_smem[ROW_E - _SPLIT, :] = de
                    dga_smem[ROW_QS - _SPLIT, :] = dqs
                    dga_smem[ROW_DEC - _SPLIT, :] = dqs * 0.0  # the dec cotangent is summed from dd_ref
                    dd_smem[...] = ddec_rows
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(dq_smem, dq_ref.at[b, pl.ds(n * c, c), h], commit_group=False)
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
            # Transposed orientation: row j is the key/value token, column i the query token.
            rows_j, cols_i = _iota(sq, 0), _iota(sq, 1)
            diag = rows_j == cols_i
            upper = cols_i > rows_j

            def step(it, carry):
                n, slot, buf = nc - 1 - it, lax.rem(it, st), lax.rem(it, 2)
                with jax.named_scope("dmain_wait_data"):
                    plgpu.barrier_wait(data.at[slot])
                gs = g_smem.at[slot]
                c_row = _load(gs.at[ROW_C], _ROW)

                def weights():
                    mask = upper
                    if cfg.segments:
                        same = _rows(_load(gs.at[ROW_RUN], _ROW), sq) == _cols(_load(gs.at[ROW_RUN_COL], _COL), sq)
                        mask = jnp.logical_and(mask, same)
                    exponent = _cols(_load(gs.at[ROW_E], _COL), sq) - _rows(_load(gs.at[ROW_A], _ROW), sq)
                    decay = _masked_exp(exponent, mask)
                    return decay, jnp.where(diag, _rows(_load(gs.at[ROW_GAMMA], _ROW), sq), decay * _rows(c_row, sq))

                with jax.named_scope("dmain_scores"):
                    p_t = _matmul(cfg, sq, (k_smem.at[slot], q_smem.at[slot].T))  # P^T[j, i] = k_j . q_i
                    dp_t, (decay_t, weight_t) = _matmul_while(
                        cfg, sq, ((v_smem.at[slot], do_smem.at[slot].T),), weights
                    )  # dP^T[j, i] = v_j . dO_i
                with jax.named_scope("dmain_weights"):
                    both = p_t * dp_t
                    dc = (decay_t * both).sum(axis=1)
                    dgamma = jnp.where(diag, both, 0.0).sum(axis=1)
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
                    tok = pl.ds(n * c, c)
                    plgpu.copy_smem_to_gmem(dk_smem, dk_ref.at[b, tok, h], commit_group=False)
                    plgpu.copy_smem_to_gmem(dv_smem, dv_ref.at[b, tok, h], commit_group=False)
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
        data=plgpu.Barrier(num_arrivals=6, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=2, num_barriers=st),
        dh_ready=plgpu.Barrier(num_arrivals=1),
        dh_free=plgpu.Barrier(num_arrivals=1),
        wq_ready=plgpu.Barrier(num_arrivals=1, num_barriers=2),
        wq_free=plgpu.Barrier(num_arrivals=1, num_barriers=2),
    )
    if cfg.interpret:
        scratch["stage"] = _operand_smem((c, c))
    kern = _kernel(
        cfg, "mamba3_bwd", body, num_threads=3, out_type=(shape_k, shape_k, shape_v, shape_g, shape_s, shape_d),
        scratch_types=scratch,
    )
    dq, dk, dv, dgates, dh0, dec_rows = kern(q, k, v, do, gates, states, dfinal)
    return dq, dk, dv, dgates.at[:, :, :, ROW_DEC, 0].add(dec_rows.sum(axis=-1)), dh0


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
    dq, dk, dv, dgates, dh0 = backward(cfg, q, k, v, do, gates, states, dfinal.astype(_F32))
    return dq, dk, dv, dgates, dh0


_core.defvjp(_core_fwd, _core_bwd)


def _head_chunks(x: Array, num_chunks: int) -> Array:
    """``[B, T, H]`` to ``[B, H, NC, CHUNK]``, zero padded at the end."""
    pad = num_chunks * CHUNK - x.shape[1]
    x = jnp.pad(x, [(0, 0), (0, pad), (0, 0)])
    return jnp.moveaxis(x, 1, 2).reshape(x.shape[0], x.shape[2], num_chunks, CHUNK)


def prepare_gates(g: Array, beta: Array, gamma: Array, *, num_chunks: int, segment_ids: Array | None = None) -> Array:
    """Per-chunk gate rows ``[B, H, NC, NUM_ROWS, CHUNK]`` float32 (see the module docstring).

    ``g, beta, gamma`` are ``[B, T, H]``; ``segment_ids [B, T]`` adds the packed-document masks.
    Differentiable: JAX autodiff maps the kernels' row gradients back to ``g, beta, gamma``.
    """
    g, beta, gamma = (x.astype(_F32) for x in (g, beta, gamma))
    batch, seq_len, heads = g.shape
    end = jnp.zeros_like(g[:, :1])
    g_next = jnp.concatenate([g[:, 1:], end], axis=1)
    beta_next = jnp.concatenate([beta[:, 1:], end], axis=1)
    a = _head_chunks(g_next, num_chunks)
    cw = _head_chunks(gamma * jnp.exp(g_next) + beta_next, num_chunks)
    gam = _head_chunks(gamma, num_chunks)
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
        runs = jnp.pad(runs, [(0, 0), (0, num_chunks * CHUNK + 1 - seq_len)], mode="edge")
        nxt = runs[:, CHUNK::CHUNK]  # [B, NC]: run of the first token after each chunk
        runs = runs[:, : num_chunks * CHUNK].reshape(batch, num_chunks, CHUNK)
        carry = (runs == runs[..., :1]).astype(_F32)[:, None]
        tail = (runs == nxt[..., None]).astype(_F32)[:, None]
        keep = (runs[..., :1] == nxt[..., None]).astype(_F32)[:, None]
        qs, w, dec = qs * carry, w * tail, dec * keep
        # Chunk-local run ids: the kernels compare them only within a chunk, and float32 holds the global
        # ids exactly only below 2**24 documents.
        run = jnp.broadcast_to((runs - runs[..., :1]).astype(_F32)[:, None], a.shape)
    dec = jnp.pad(dec, [(0, 0), (0, 0), (0, 0), (0, CHUNK - 1)])  # the chunk's decay sits in column 0
    rows = {ROW_A: big_a, ROW_C: cw, ROW_GAMMA: gam, ROW_W: w, ROW_RUN: run, ROW_RUN_COL: run, ROW_E: big_e,
            ROW_QS: qs, ROW_DEC: dec}
    return jnp.stack([rows[r] for r in range(NUM_ROWS)], axis=-2)


def _initial_mid_state(initial_state: Mamba3State, g: Array, beta: Array) -> Array:
    """``H_0 = exp(g_0) S_0 + beta_0 k_0^T v_0`` (``[B, H, K, V]`` float32)."""
    ssm, key, value = (x.astype(_F32) for x in initial_state)
    g0, beta0 = g[:, 0].astype(_F32), beta[:, 0].astype(_F32)
    return jnp.exp(g0)[..., None, None] * ssm + beta0[..., None, None] * key[..., :, None] * value[..., None, :]


def _check(q, k, v, g, beta, gamma, initial_state, segment_ids) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"q, k must be [B, T, H, K] and v [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}")
    for name, x in (("g", g), ("beta", beta), ("gamma", gamma)):
        if x.shape != q.shape[:3]:
            raise ValueError(f"{name} must be [B, T, H] = {q.shape[:3]}; got {x.shape}")
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if (key_dim, value_dim) not in SUPPORTED_DIMS:
        raise ValueError(f"mosaic_gpu mamba3 supports (K, V) in {SUPPORTED_DIMS}; got ({key_dim}, {value_dim})")
    if seq_len == 0:
        raise ValueError("mosaic_gpu mamba3 needs at least one token")
    if initial_state is not None:
        want = ((batch, heads, key_dim, value_dim), (batch, heads, key_dim), (batch, heads, value_dim))
        got = tuple(x.shape for x in initial_state)
        if got != want:
            raise ValueError(f"initial_state shapes must be {want}; got {got}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")


def mamba3(
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
    chunk_size: int = CHUNK,
    stages: int | None = None,
    interpret: bool = False,
) -> tuple[Array, Mamba3State | None]:
    """Mamba-3 SISO on the Mosaic GPU kernels; the arguments of :func:`jla.ops.reference.mamba3`.

    ``chunk_size`` must be 64; ``stages`` (1 to 4) is the TMA pipeline depth over chunks, capped
    per kernel by shared memory (the forward kernel rounds it to an even depth); by default 4 when
    at most 64 CTAs (``B * H``) run, else 2. ``interpret`` runs the kernels in the GPU interpreter
    (for CPU tests). ``q, k, v`` are computed in bfloat16.
    """
    if chunk_size != CHUNK:
        raise ValueError(f"mosaic_gpu mamba3 uses chunk_size={CHUNK}; got {chunk_size}")
    _check(q, k, v, g, beta, gamma, initial_state, segment_ids)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if stages is None:
        stages = 4 if batch * heads <= 64 else 2  # a deeper TMA ring pays off when few CTAs run
    if not 1 <= stages <= 4:
        raise ValueError(f"stages must be in [1, 4]; got {stages}")
    num_chunks = -(-seq_len // CHUNK)
    cfg = Config(
        batch=batch, heads=heads, num_chunks=num_chunks, key_dim=key_dim, value_dim=value_dim,
        scale=float(scale), segments=segment_ids is not None, stages=min(stages, num_chunks), interpret=interpret,
    )
    gates = prepare_gates(g, beta, gamma, num_chunks=num_chunks, segment_ids=segment_ids)
    if initial_state is None:
        h0 = jnp.zeros((batch, heads, key_dim, value_dim), _F32)
    else:
        h0 = _initial_mid_state(initial_state, g, beta)
    pad = [(0, 0), (0, cfg.seq_len - seq_len), (0, 0), (0, 0)]
    qp, kp, vp = (jnp.pad(x.astype(_BF16), pad) for x in (q, k, v))
    o, final = _core(cfg, qp, kp, vp, gates, h0)
    o = o[:, :seq_len].astype(q.dtype)
    if not output_final_state:
        return o, None
    return o, Mamba3State(final, k[:, -1].astype(_F32), v[:, -1].astype(_F32))


def _on_hopper() -> bool:
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", None) == "9.0"


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
    chunk_size: int = CHUNK,
    cu_seqlens: Array | None = None,
    precision: Any = None,
    stages: int | None = None,
    interpret: bool = False,
    **unknown: Any,
) -> bool:
    """True only for calls the kernels are tested on: H100, bf16 ``q, k, v``, ``(K, V)`` in
    :data:`SUPPORTED_DIMS`, float gates, optional ``segment_ids`` and ``initial_state``, chunk 64."""
    del output_final_state
    if unknown or cu_seqlens is not None or precision is not None or chunk_size != CHUNK:
        return False
    if not isinstance(scale, (int, float)) or (stages is not None and not 1 <= stages <= 4):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3] or q.shape[1] == 0:
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
        batch, _, heads, key_dim = q.shape
        want = ((batch, heads, key_dim, v.shape[-1]), (batch, heads, key_dim), (batch, heads, v.shape[-1]))
        if not isinstance(initial_state, Mamba3State) or tuple(x.shape for x in initial_state) != want:
            return False
        if not all(jnp.issubdtype(x.dtype, jnp.floating) for x in initial_state):
            return False
    return interpret or _on_hopper()


# Eligible for impl="auto": GPU-tested for every call `supports` accepts and 18-23x faster than chunked.
ops.register("mosaic_gpu", "mamba3", mamba3, supports, auto=True)
