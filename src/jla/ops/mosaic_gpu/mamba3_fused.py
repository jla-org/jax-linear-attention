# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/ops/triton/mamba3/angle_dt.py,
# mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py, mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py,
# mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py
"""Fused Mamba-3 SISO layer core on Mosaic GPU: :func:`mamba3_siso_fused`.

The semantics of :func:`jla.layers.ssm.mamba3_siso` (and of the official
``mamba3_siso_combined``): per-group ``C, B``, per-head biases, the data-dependent
rotation by ``theta_t = sum_{s <= t} tanh(angles_s) pi DT_s (mod 2 pi)``, the trapezoidal
recurrence (:mod:`jla.ops.mosaic_gpu.mamba3`), the ``D`` skip and the ``Z`` gate.

Kernels, following the official kernel split:

* ``angle_forward`` / ``angle_backward``: one CTA per ``(head, batch)`` walks the chunks
  (backwards for the gradient) carrying the angle (or its cotangent) and computes the
  in-chunk prefix sums exactly with three bf16 tensor-core passes (``x = hi + mid + lo``);
* ``rotary_forward``: one CTA per ``(chunk, batch, group)`` loads the group's ``C, B`` once
  and writes every head's biased, rotated ``q, k`` for the recurrence kernels;
* ``rotary_backward``: the same grid; undoes the rotation of ``dq, dk``, sums them over the
  heads of the group in the CTA (no atomics, deterministic), and emits the angle cotangent
  and per-chunk bias-gradient partials.

The kernels keep the rotated channel pairs ``(2i, 2i + 1)`` de-interleaved (``[evens | odds |
unrotated]``) so a rotation pairs two column blocks; XLA permutes ``C, B``, the biases and the
states at group size, and the recurrence is invariant to the order of the key channels.
The angle scan uses exact ``tanh`` (its errors would accumulate along the sequence). The rotary
kernels use ``sin.approx`` and ``cos.approx`` of the angle wrapped to ``[-pi, pi]``, where their absolute
error is at most 2^-20.9, far below the bf16 rounding of the rotated ``q, k`` that follows; exact
``sin``/``cos`` made those kernels 2.1x slower (measured). XLA keeps the small per-token gate math,
the ``D`` skip and the ``Z`` gate (elementwise), differentiated by JAX.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import MAX_GRID_YZ, finish_together
from jla.ops.mosaic_gpu.mamba3 import (
    CHUNK,
    SUPPORTED_DIMS,
    Config,
    _COL,
    _ROW,
    _Regs,
    _core,
    _head_chunks,
    _initial_mid_state,
    _iota,
    _load,
    _matmul,
    _memory_warpgroup,
    _on_hopper,
    _operand_smem,
    _cols,
    _rows,
    prepare_gates,
)
from jla.ops.reference import Mamba3State, segment_resets

__all__ = ["AUTO", "angle_dims", "angle_rows", "mamba3_siso_fused", "supports_fused"]

Array = jax.Array
_F32 = jnp.float32
_BF16 = jnp.bfloat16
_TWO_PI = 2.0 * math.pi

# Whether ``jla.layers.ssm.mamba3_siso(impl="auto")`` takes the fused core: only once it is
# GPU-tested for every call :func:`supports_fused` accepts and faster than the layer's XLA path.
# Qualified (tests/test_mosaic_gpu_mamba3_fused.py; benchmarks/mamba3.md).
AUTO = True


def angle_dims(key_dim: int) -> tuple[int, int]:
    """Rotated pairs per head (``angles.shape[-1]``) the kernels take: Mamba-3's rope fractions 0.5, 1.0."""
    return key_dim // 4, key_dim // 2


# Rows of the per-chunk angle tile [NUM_ANGLE_ROWS, CHUNK] (float32): the head's dt, the segment
# run (twice, as in the recurrence kernels), and three packed-document masks: CM (token continues
# the document of the token before the chunk), LS (same document as the chunk's last token), FS
# (same document as the chunk's first token).
RA_DT, RA_RUN, RA_RUN_COL, RA_CM, RA_LS, RA_FS = range(6)
NUM_ANGLE_ROWS = 6

_MEMORY_REGISTERS = 40
_ANGLE_STAGES = 4
_ROTARY_STAGES = 4
_ROTARY_BWD_STAGES = 2


@dataclasses.dataclass(frozen=True)
class RotaryConfig:
    """Static shapes of the angle and rotary kernels."""

    batch: int
    heads: int
    groups: int
    num_chunks: int
    key_dim: int
    angle_dim: int
    segments: bool = False
    interpret: bool = False

    @property
    def group_heads(self) -> int:
        return self.heads // self.groups

    @property
    def blocks(self) -> tuple[tuple[int, int], ...]:
        """``(first column, width)`` of the rotated evens, rotated odds and unrotated channels."""
        s, k = self.angle_dim, self.key_dim
        return ((0, s), (s, s)) + (((2 * s, k - 2 * s),) if k > 2 * s else ())


def _glue_kernel(rc: RotaryConfig, name: str, body, *, grid, grid_names, approx: bool = False, **kwargs):
    """A compute warpgroup (0) and a TMA warpgroup (1); ``approx`` allows PTX ``.approx`` math."""
    return plgpu.kernel(
        body,
        grid=grid,
        grid_names=grid_names,
        num_threads=2,
        thread_name="wg",
        kernel_name=name,
        compiler_params=plgpu.CompilerParams(approx_math=approx),
        interpret=plgpu.InterpretGPUParams() if rc.interpret else None,
        **kwargs,
    )


def _wrap(x: Array) -> Array:
    """The same angle in ``[-pi, pi]``."""
    return x - _TWO_PI * jnp.round(x * (1.0 / _TWO_PI))


def _split3(x: Array, parts) -> None:
    """Store float32 ``x`` as three bf16 terms whose sum is ``x`` to ~2^-24 (``parts [3, 64, N]``)."""
    hi = x.astype(_BF16)
    rest = x - hi.astype(_F32)
    mid = rest.astype(_BF16)
    parts.at[0][...] = hi
    parts.at[1][...] = mid
    parts.at[2][...] = (rest - mid.astype(_F32)).astype(_BF16)
    plgpu.commit_smem()


def _masked_sums(rc: RotaryConfig, weights: Array, parts, shape, stage) -> Array:
    """``weights @ x`` (0/1 ``weights [64, 64]`` float32) for ``x`` stored by :func:`_split3`:
    float32-exact sums over the chunk."""
    ones = _Regs(weights.astype(_BF16))
    return _matmul(rc, shape, *((ones, parts.at[i]) for i in range(3)), stage=stage)


def _triangle(shape: tuple[int, int], *, lower: bool) -> Array:
    """1 on and below (``lower``) or above the diagonal, else 0.

    Built with arithmetic: Mosaic GPU turns an iota-comparison mask into wrong 0/1 values in rows
    8-15 of every warp's 16 rows (``jnp.where(cols <= rows, 1.0, 0.0)`` or ``mask.astype``).
    """
    rows, cols = _iota(shape, 0), _iota(shape, 1)
    diff = rows - cols if lower else cols - rows
    return jnp.minimum(jnp.maximum((diff + 1).astype(_F32), 0.0), 1.0)


def _same_run(rs, shape: tuple[int, int]) -> Array:
    """1 where the row and column tokens are in the same document, else 0 (arithmetic, as above)."""
    gap = _rows(_load(rs.at[RA_RUN], _ROW), shape) - _cols(_load(rs.at[RA_RUN_COL], _COL), shape)
    return 1.0 - jnp.minimum(jnp.abs(gap), 1.0)


# ---------------------------------------------------------------------------------------------
# angle scan
# ---------------------------------------------------------------------------------------------


def angle_forward(rc: RotaryConfig, angles: Array, rows: Array, theta0: Array) -> tuple[Array, Array]:
    """``theta [B, T, H, S]`` and the last token's angle ``[B, H, S]`` (float32, in ``[-pi, pi]``).

    ``theta_t = theta0 [first document] + sum_{s <= t, same document} tanh(angles_s) pi dt_s``;
    ``rows`` from :func:`angle_rows`.
    """
    c, nc, s = CHUNK, rc.num_chunks, rc.angle_dim
    st = min(_ANGLE_STAGES, nc)
    sq, tile = (c, c), (c, s)

    def body(ang_ref, rows_ref, th0_ref, th_ref, fin_ref, **smem):
        ang_smem, rows_smem, parts, out, vec = smem["ang"], smem["rows"], smem["parts"], smem["out"], smem["vec"]
        data, consumed, vec_bar, stage = smem["data"], smem["consumed"], smem["vec_bar"], smem.get("stage")
        wg, h, b = lax.axis_index("wg"), lax.axis_index("h"), lax.axis_index("b")

        def fetch(n, slot):
            plgpu.copy_gmem_to_smem(ang_ref.at[b, pl.ds(n * c, c), h], ang_smem.at[slot], data.at[slot])
            plgpu.copy_gmem_to_smem(rows_ref.at[b, h, n], rows_smem.at[slot], data.at[slot])

        @pl.when(wg == 1)
        def _():
            _memory_warpgroup(fetch, consumed, nc, st, lambda step: step, _MEMORY_REGISTERS)

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(th0_ref.at[b, h], vec, vec_bar)
            plgpu.barrier_wait(vec_bar)
            causal = _triangle(sq, lower=True)

            def step(n, carry):
                slot = lax.rem(n, st)
                plgpu.barrier_wait(data.at[slot])
                rs = rows_smem.at[slot]
                dt = _rows(_load(rs.at[RA_DT], _ROW), tile)
                inc = jnp.tanh(_load(ang_smem.at[slot]).astype(_F32)) * (math.pi * dt)
                mask = causal * _same_run(rs, sq) if rc.segments else causal
                total = (inc * _rows(_load(rs.at[RA_LS], _ROW), tile)).sum(axis=0)  # the last document's sum
                cm = _rows(_load(rs.at[RA_CM], _ROW), tile)
                through = rows_smem[slot, RA_CM, c - 1]  # 1 if the whole chunk continues the document
                _split3(inc, parts)
                theta = _wrap(_masked_sums(rc, mask, parts, tile, stage) + cm * _cols(carry, tile))
                plgpu.barrier_arrive(consumed.at[slot])  # after the last use of the slot
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                out[...] = theta
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out, th_ref.at[b, pl.ds(n * c, c), h])
                return _wrap(total + through * carry)

            carry = lax.fori_loop(0, nc, step, _load(vec, _COL))
            vec[...] = carry
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(vec, fin_ref.at[b, h])
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])  # no warpgroup exits while another still issues WGMMAs

    scratch = dict(
        done=plgpu.Barrier(num_arrivals=2),
        ang=plgpu.SMEM((st, c, s), angles.dtype),
        rows=plgpu.SMEM((st, NUM_ANGLE_ROWS, c), _F32),
        parts=_operand_smem((3, c, s)),
        out=plgpu.SMEM((c, s), _F32),
        vec=plgpu.SMEM((s,), _F32),
        data=plgpu.Barrier(num_arrivals=2, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=1, num_barriers=st),
        vec_bar=plgpu.Barrier(num_arrivals=1),
    )
    if rc.interpret:
        scratch["stage"] = _operand_smem((c, c))
    out_type = (
        jax.ShapeDtypeStruct((rc.batch, nc * c, rc.heads, s), _F32),
        jax.ShapeDtypeStruct((rc.batch, rc.heads, s), _F32),
    )
    kern = _glue_kernel(rc, "mamba3_angle_fwd", body, grid=(rc.heads, rc.batch), grid_names=("h", "b"),
                        out_type=out_type, scratch_types=scratch)
    return kern(angles, rows, theta0)


def angle_backward(
    rc: RotaryConfig, dtheta: Array, angles: Array, rows: Array, dlast: Array
) -> tuple[Array, Array, Array]:
    """``(dangles, ddt [B, H, NC, CHUNK], dtheta0)`` from ``dtheta [B, T, H, S]`` and the cotangent
    ``dlast [B, H, S]`` of the last token's angle (a reverse, document-wise prefix sum)."""
    c, nc, s = CHUNK, rc.num_chunks, rc.angle_dim
    st = min(_ANGLE_STAGES, nc)
    sq, tile = (c, c), (c, s)

    def body(dth_ref, ang_ref, rows_ref, dl_ref, dang_ref, ddt_ref, d0_ref, **smem):
        dth_smem, ang_smem, rows_smem, parts = smem["dth"], smem["ang"], smem["rows"], smem["parts"]
        dang_out, ddt_out, vec = smem["dang"], smem["ddt"], smem["vec"]
        data, consumed, vec_bar, stage = smem["data"], smem["consumed"], smem["vec_bar"], smem.get("stage")
        wg, h, b = lax.axis_index("wg"), lax.axis_index("h"), lax.axis_index("b")

        def fetch(n, slot):
            tok = pl.ds(n * c, c)
            plgpu.copy_gmem_to_smem(dth_ref.at[b, tok, h], dth_smem.at[slot], data.at[slot])
            plgpu.copy_gmem_to_smem(ang_ref.at[b, tok, h], ang_smem.at[slot], data.at[slot])
            plgpu.copy_gmem_to_smem(rows_ref.at[b, h, n], rows_smem.at[slot], data.at[slot])

        @pl.when(wg == 1)
        def _():
            _memory_warpgroup(fetch, consumed, nc, st, lambda step: nc - 1 - step, _MEMORY_REGISTERS)

        @pl.when(wg == 0)
        def _():
            plgpu.copy_gmem_to_smem(dl_ref.at[b, h], vec, vec_bar)
            plgpu.barrier_wait(vec_bar)
            anticausal = _triangle(sq, lower=False)

            def step(it, carry):
                n, slot = nc - 1 - it, lax.rem(it, st)
                plgpu.barrier_wait(data.at[slot])
                rs = rows_smem.at[slot]
                dth = _load(dth_smem.at[slot])
                tanh = jnp.tanh(_load(ang_smem.at[slot]).astype(_F32))
                dt = _rows(_load(rs.at[RA_DT], _ROW), tile)
                mask = anticausal * _same_run(rs, sq) if rc.segments else anticausal
                first = (dth * _rows(_load(rs.at[RA_FS], _ROW), tile)).sum(axis=0)  # the first document's sum
                ls = _rows(_load(rs.at[RA_LS], _ROW), tile)
                cm0, ls0 = rows_smem[slot, RA_CM, 0], rows_smem[slot, RA_LS, 0]
                _split3(dth, parts)
                dinc = _masked_sums(rc, mask, parts, tile, stage) + ls * _cols(carry, tile)
                dang = (dinc * (math.pi * dt) * (1.0 - tanh * tanh)).astype(dang_out.dtype)
                ddt = (dinc * tanh).sum(axis=1) * math.pi
                plgpu.barrier_arrive(consumed.at[slot])  # after the last use of the slot
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                dang_out[...] = dang
                ddt_out[...] = ddt
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(dang_out, dang_ref.at[b, pl.ds(n * c, c), h], commit_group=False)
                plgpu.copy_smem_to_gmem(ddt_out, ddt_ref.at[b, h, n], commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                return cm0 * (first + ls0 * carry)

            carry = lax.fori_loop(0, nc, step, _load(vec, _COL))
            vec[...] = carry
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(vec, d0_ref.at[b, h])
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])

    scratch = dict(
        done=plgpu.Barrier(num_arrivals=2),
        dth=plgpu.SMEM((st, c, s), _F32),
        ang=plgpu.SMEM((st, c, s), angles.dtype),
        rows=plgpu.SMEM((st, NUM_ANGLE_ROWS, c), _F32),
        parts=_operand_smem((3, c, s)),
        dang=plgpu.SMEM((c, s), angles.dtype),
        ddt=plgpu.SMEM((c,), _F32),
        vec=plgpu.SMEM((s,), _F32),
        data=plgpu.Barrier(num_arrivals=3, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=1, num_barriers=st),
        vec_bar=plgpu.Barrier(num_arrivals=1),
    )
    if rc.interpret:
        scratch["stage"] = _operand_smem((c, c))
    out_type = (
        jax.ShapeDtypeStruct(angles.shape, angles.dtype),
        jax.ShapeDtypeStruct((rc.batch, rc.heads, nc, c), _F32),
        jax.ShapeDtypeStruct((rc.batch, rc.heads, s), _F32),
    )
    kern = _glue_kernel(rc, "mamba3_angle_bwd", body, grid=(rc.heads, rc.batch), grid_names=("h", "b"),
                        out_type=out_type, scratch_types=scratch)
    return kern(dtheta, angles, rows, dlast)


# ---------------------------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------------------------


def rotary_forward(rc: RotaryConfig, qp: Array, kp: Array, theta: Array, qb: Array, kb: Array) -> tuple[Array, Array]:
    """Per-head ``q = rot(C + C_bias), k = rot(B + B_bias)`` ``[B, T, H, K]`` bf16 (kernel channel order).

    ``qp, kp [B, T, G, K]`` bf16 and ``qb, kb [H, K]`` f32 in the kernel channel order.
    """
    c, nc, s, rh = CHUNK, rc.num_chunks, rc.angle_dim, rc.group_heads
    blocks = rc.blocks
    st = min(_ROTARY_STAGES, rh)

    def body(qp_ref, kp_ref, th_ref, qb_ref, kb_ref, qr_ref, kr_ref, **smem):
        srcs = [smem[f"src{i}"] for i in range(len(blocks))]
        outs = [smem[f"out{i}"] for i in range(len(blocks))]
        th_smem, bias = smem["th"], smem["bias"]
        data, consumed, src_bar = smem["data"], smem["consumed"], smem["src_bar"]
        wg, n, b, g = lax.axis_index("wg"), lax.axis_index("n"), lax.axis_index("b"), lax.axis_index("g")
        tok = pl.ds(n * c, c)

        def fetch(r, slot):
            h = g * rh + r
            plgpu.copy_gmem_to_smem(th_ref.at[b, tok, h], th_smem.at[slot], data.at[slot])
            plgpu.copy_gmem_to_smem(qb_ref.at[h], bias.at[slot, 0], data.at[slot])
            plgpu.copy_gmem_to_smem(kb_ref.at[h], bias.at[slot, 1], data.at[slot])

        @pl.when(wg == 1)
        def _():
            for t, ref in enumerate((qp_ref, kp_ref)):
                for buf, (col, width) in zip(srcs, blocks):
                    plgpu.copy_gmem_to_smem(ref.at[b, tok, g, pl.ds(col, width)], buf.at[t], src_bar)
            _memory_warpgroup(fetch, consumed, rh, st, lambda step: step, _MEMORY_REGISTERS)

        @pl.when(wg == 0)
        def _():
            plgpu.barrier_wait(src_bar)
            held = [[_load(buf.at[t]) for buf in srcs] for t in (0, 1)]  # the group's C, B tiles, in registers

            def step(r, carry):
                slot, h = lax.rem(r, st), g * rh + r
                plgpu.barrier_wait(data.at[slot])
                theta = _load(th_smem.at[slot])
                cos, sin = jnp.cos(theta), jnp.sin(theta)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous head's stores read `outs`
                for t in (0, 1):
                    bias_t = bias.at[slot, t]

                    def biased(i, t=t, bias_t=bias_t):
                        col, width = blocks[i]
                        return held[t][i].astype(_F32) + _cols(
                            _load(bias_t.at[pl.ds(col, width)], _COL), (c, width)
                        )

                    even, odd = biased(0), biased(1)
                    outs[0].at[t][...] = (even * cos - odd * sin).astype(_BF16)
                    outs[1].at[t][...] = (even * sin + odd * cos).astype(_BF16)
                    if len(blocks) == 3:
                        outs[2].at[t][...] = biased(2).astype(_BF16)
                plgpu.barrier_arrive(consumed.at[slot])
                plgpu.commit_smem()
                for t, ref in enumerate((qr_ref, kr_ref)):
                    for buf, (col, width) in zip(outs, blocks):
                        plgpu.copy_smem_to_gmem(buf.at[t], ref.at[b, tok, h, pl.ds(col, width)], commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                return carry

            lax.fori_loop(0, rh, step, None)
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])

    scratch = {}
    for i, (_, width) in enumerate(blocks):
        scratch[f"src{i}"] = _operand_smem((2, c, width))
        scratch[f"out{i}"] = _operand_smem((2, c, width))
    scratch.update(
        done=plgpu.Barrier(num_arrivals=2),
        th=plgpu.SMEM((st, c, s), _F32),
        bias=plgpu.SMEM((st, 2, rc.key_dim), _F32),
        data=plgpu.Barrier(num_arrivals=3, num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=1, num_barriers=st),
        src_bar=plgpu.Barrier(num_arrivals=2 * len(blocks)),
    )
    shape = jax.ShapeDtypeStruct((rc.batch, nc * c, rc.heads, rc.key_dim), _BF16)
    kern = _glue_kernel(rc, "mamba3_rotary_fwd", body, grid=(nc, rc.batch, rc.groups), grid_names=("n", "b", "g"),
                        approx=True, out_type=(shape, shape), scratch_types=scratch)
    return kern(qp, kp, theta, qb, kb)


def rotary_backward(
    rc: RotaryConfig, qp: Array, kp: Array, theta: Array, qb: Array, kb: Array, dq: Array, dk: Array
) -> tuple[Array, Array, Array, Array, Array]:
    """``(dC, dB [B, T, G, K], dtheta [B, T, H, S], dC_bias, dB_bias partials [B, NC, H, K])``, float32.

    ``dq, dk [B, T, H, K]`` bf16 are the cotangents of :func:`rotary_forward`'s outputs; each CTA
    sums the heads of its group, so ``dC, dB`` need no further reduction.
    """
    c, nc, s, rh = CHUNK, rc.num_chunks, rc.angle_dim, rc.group_heads
    blocks = rc.blocks
    st = min(_ROTARY_BWD_STAGES, rh)

    def body(qp_ref, kp_ref, th_ref, qb_ref, kb_ref, dq_ref, dk_ref, dqp_ref, dkp_ref, dth_ref, dqb_ref, dkb_ref,
             **smem):
        srcs = [smem[f"src{i}"] for i in range(len(blocks))]
        grads = [smem[f"grad{i}"] for i in range(len(blocks))]
        accs = [smem[f"acc{i}"] for i in range(len(blocks))]
        th_smem, bias, dth_out, part = smem["th"], smem["bias"], smem["dth"], smem["part"]
        data, consumed, src_bar = smem["data"], smem["consumed"], smem["src_bar"]
        wg, n, b, g = lax.axis_index("wg"), lax.axis_index("n"), lax.axis_index("b"), lax.axis_index("g")
        tok = pl.ds(n * c, c)

        def fetch(r, slot):
            h = g * rh + r
            bar = data.at[slot]
            plgpu.copy_gmem_to_smem(th_ref.at[b, tok, h], th_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(qb_ref.at[h], bias.at[slot, 0], bar)
            plgpu.copy_gmem_to_smem(kb_ref.at[h], bias.at[slot, 1], bar)
            for t, ref in enumerate((dq_ref, dk_ref)):
                for buf, (col, width) in zip(grads, blocks):
                    plgpu.copy_gmem_to_smem(ref.at[b, tok, h, pl.ds(col, width)], buf.at[slot, t], bar)

        @pl.when(wg == 1)
        def _():
            for t, ref in enumerate((qp_ref, kp_ref)):
                for buf, (col, width) in zip(srcs, blocks):
                    plgpu.copy_gmem_to_smem(ref.at[b, tok, g, pl.ds(col, width)], buf.at[t], src_bar)
            _memory_warpgroup(fetch, consumed, rh, st, lambda step: step, _MEMORY_REGISTERS)

        @pl.when(wg == 0)
        def _():
            for acc, (_, width) in zip(accs, blocks):
                for t in (0, 1):
                    acc.at[t][...] = jnp.zeros((c, width), _F32)
            plgpu.barrier_wait(src_bar)
            held = [[_load(buf.at[t]) for buf in srcs] for t in (0, 1)]  # the group's C, B tiles, in registers

            def step(r, carry):
                slot, h = lax.rem(r, st), g * rh + r
                plgpu.barrier_wait(data.at[slot])
                theta = _load(th_smem.at[slot])
                cos, sin = jnp.cos(theta), jnp.sin(theta)
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)  # the previous head's stores read dth, part
                dth = None
                for t in (0, 1):
                    bias_t = bias.at[slot, t]

                    def biased(i, t=t, bias_t=bias_t):
                        col, width = blocks[i]
                        return held[t][i].astype(_F32) + _cols(
                            _load(bias_t.at[pl.ds(col, width)], _COL), (c, width)
                        )

                    even, odd = biased(0), biased(1)
                    d_even = _load(grads[0].at[slot, t]).astype(_F32)
                    d_odd = _load(grads[1].at[slot, t]).astype(_F32)
                    # d theta of (even cos - odd sin, even sin + odd cos)
                    term = d_odd * (even * cos - odd * sin) - d_even * (even * sin + odd * cos)
                    dth = term if dth is None else dth + term
                    unrotated = (d_even * cos + d_odd * sin, d_odd * cos - d_even * sin)
                    if len(blocks) == 3:
                        unrotated += (_load(grads[2].at[slot, t]).astype(_F32),)
                    for i, (x, (col, width)) in enumerate(zip(unrotated, blocks)):
                        accs[i].at[t][...] = _load(accs[i].at[t]) + x
                        part.at[t, pl.ds(col, width)][...] = x.sum(axis=0)
                plgpu.barrier_arrive(consumed.at[slot])
                dth_out[...] = dth
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(dth_out, dth_ref.at[b, tok, h], commit_group=False)
                plgpu.copy_smem_to_gmem(part.at[0], dqb_ref.at[b, n, h], commit_group=False)
                plgpu.copy_smem_to_gmem(part.at[1], dkb_ref.at[b, n, h], commit_group=False)
                plgpu.commit_smem_to_gmem_group()
                return carry

            lax.fori_loop(0, rh, step, None)
            plgpu.commit_smem()
            for t, ref in enumerate((dqp_ref, dkp_ref)):
                for acc, (col, width) in zip(accs, blocks):
                    plgpu.copy_smem_to_gmem(acc.at[t], ref.at[b, tok, g, pl.ds(col, width)], commit_group=False)
            plgpu.commit_smem_to_gmem_group()
            plgpu.wait_smem_to_gmem(0)

        finish_together(smem["done"])

    scratch = {}
    for i, (_, width) in enumerate(blocks):
        scratch[f"src{i}"] = _operand_smem((2, c, width))
        scratch[f"grad{i}"] = _operand_smem((st, 2, c, width))
        scratch[f"acc{i}"] = plgpu.SMEM((2, c, width), _F32)
    scratch.update(
        done=plgpu.Barrier(num_arrivals=2),
        th=plgpu.SMEM((st, c, s), _F32),
        bias=plgpu.SMEM((st, 2, rc.key_dim), _F32),
        dth=plgpu.SMEM((c, s), _F32),
        part=plgpu.SMEM((2, rc.key_dim), _F32),
        data=plgpu.Barrier(num_arrivals=3 + 2 * len(blocks), num_barriers=st),
        consumed=plgpu.Barrier(num_arrivals=1, num_barriers=st),
        src_bar=plgpu.Barrier(num_arrivals=2 * len(blocks)),
    )
    group_shape = jax.ShapeDtypeStruct((rc.batch, nc * c, rc.groups, rc.key_dim), _F32)
    part_shape = jax.ShapeDtypeStruct((rc.batch, nc, rc.heads, rc.key_dim), _F32)
    out_type = (group_shape, group_shape, jax.ShapeDtypeStruct(theta.shape, _F32), part_shape, part_shape)
    kern = _glue_kernel(rc, "mamba3_rotary_bwd", body, grid=(nc, rc.batch, rc.groups), grid_names=("n", "b", "g"),
                        approx=True, out_type=out_type, scratch_types=scratch)
    return kern(qp, kp, theta, qb, kb, dq, dk)


# ---------------------------------------------------------------------------------------------
# custom_vjp of the rotary stage and XLA glue
# ---------------------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _rotary(rc: RotaryConfig, qp, kp, qb, kb, angles, rows, theta0):
    theta, last = angle_forward(rc, angles, rows, theta0)
    qr, kr = rotary_forward(rc, qp, kp, theta, qb, kb)
    return qr, kr, last


def _rotary_fwd(rc: RotaryConfig, qp, kp, qb, kb, angles, rows, theta0):
    theta, last = angle_forward(rc, angles, rows, theta0)
    qr, kr = rotary_forward(rc, qp, kp, theta, qb, kb)
    return (qr, kr, last), (qp, kp, qb, kb, angles, rows, theta)


def _rotary_bwd(rc: RotaryConfig, residuals, cotangents):
    qp, kp, qb, kb, angles, rows, theta = residuals
    dqr, dkr, dlast = cotangents
    dqp, dkp, dtheta, dqb, dkb = rotary_backward(rc, qp, kp, theta, qb, kb, dqr.astype(_BF16), dkr.astype(_BF16))
    dangles, ddt, dtheta0 = angle_backward(rc, dtheta, angles, rows, dlast.astype(_F32))
    drows = jnp.zeros_like(rows).at[:, :, :, RA_DT].set(ddt)
    return (dqp.astype(qp.dtype), dkp.astype(kp.dtype), dqb.sum(axis=(0, 1)), dkb.sum(axis=(0, 1)), dangles, drows,
            dtheta0)


_rotary.defvjp(_rotary_fwd, _rotary_bwd)


def _deinterleave(x: Array, num: int, axis: int = -1) -> Array:
    """Channels ``(2i, 2i + 1)``, ``i < num``, of ``axis`` reordered as ``[evens | odds | rest]``."""
    x = jnp.moveaxis(x, axis, -1)
    rot = x[..., : 2 * num].reshape(*x.shape[:-1], num, 2)
    rot = jnp.swapaxes(rot, -1, -2).reshape(*x.shape[:-1], 2 * num)
    return jnp.moveaxis(jnp.concatenate([rot, x[..., 2 * num:]], axis=-1), -1, axis)


def _interleave(x: Array, num: int, axis: int = -1) -> Array:
    """Inverse of :func:`_deinterleave`."""
    x = jnp.moveaxis(x, axis, -1)
    rot = x[..., : 2 * num].reshape(*x.shape[:-1], 2, num)
    rot = jnp.swapaxes(rot, -1, -2).reshape(*x.shape[:-1], 2 * num)
    return jnp.moveaxis(jnp.concatenate([rot, x[..., 2 * num:]], axis=-1), -1, axis)


def angle_rows(dt: Array, *, num_chunks: int, segment_ids: Array | None = None) -> Array:
    """Per-chunk angle rows ``[B, H, NC, NUM_ANGLE_ROWS, CHUNK]`` float32 from ``dt [B, T, H]``."""
    dtc = _head_chunks(dt.astype(_F32), num_chunks)
    batch, heads = dtc.shape[:2]
    if segment_ids is None:
        ones, zeros = jnp.ones_like(dtc), jnp.zeros_like(dtc)
        rows = [dtc, zeros, zeros, ones, ones, ones]
    else:
        seq_len = segment_ids.shape[1]
        runs = jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)
        runs = jnp.pad(runs, [(0, 0), (0, num_chunks * CHUNK - seq_len)], mode="edge")
        runs = runs.reshape(batch, num_chunks, CHUNK)
        # The document of the token before each chunk; the first chunk continues the initial state's.
        before = jnp.concatenate([runs[:, :1, :1], runs[:, :-1, -1:]], axis=1)

        def per_head(x):
            return jnp.broadcast_to(x.astype(_F32)[:, None], dtc.shape)

        run = per_head(runs - runs[..., :1])  # chunk-local: float32 holds global ids exactly only below 2**24
        rows = [dtc, run, run, per_head(runs == before), per_head(runs == runs[..., -1:]),
                per_head(runs == runs[..., :1])]
    return jnp.stack(rows, axis=-2)


def _check_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, initial_state, segment_ids) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"C, B must be [B, T, G, K] and x [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}")
    batch, seq_len, groups, key_dim = q.shape
    heads, value_dim = v.shape[2:]
    if heads % groups:
        raise ValueError(f"heads {heads} must be a multiple of the groups {groups}")
    for name, x in (("ADT", adt), ("DT", dt), ("trap", trap)):
        if x.shape != (batch, seq_len, heads):
            raise ValueError(f"{name} must be [B, T, H] = {(batch, seq_len, heads)}; got {x.shape}")
    for name, x in (("C_bias", q_bias), ("B_bias", k_bias)):
        if x.shape != (heads, key_dim):
            raise ValueError(f"{name} must be [H, K] = {(heads, key_dim)}; got {x.shape}")
    num = angles.shape[-1]
    if angles.shape != (batch, seq_len, heads, num) or num not in angle_dims(key_dim):
        raise ValueError(f"angles must be [B, T, H, S] with S in {angle_dims(key_dim)}; got {angles.shape}")
    if (key_dim, value_dim) not in SUPPORTED_DIMS:
        raise ValueError(f"mosaic_gpu mamba3 supports (K, V) in {SUPPORTED_DIMS}; got ({key_dim}, {value_dim})")
    if d is not None and d.shape != (heads,):
        raise ValueError(f"D must be [H] = {(heads,)}; got {d.shape}")
    if z is not None and z.shape != v.shape:
        raise ValueError(f"z must be [B, T, H, V] = {v.shape}; got {z.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
    if -(-seq_len // CHUNK) > MAX_GRID_YZ:  # the rotary kernels' grid is (chunks, batch, groups)
        raise ValueError(f"T={seq_len} needs more than {MAX_GRID_YZ} chunks of {CHUNK} tokens in one call (the "
                         "CUDA grid limit of the rotary kernels); split the sequence and chain initial_state")
    if initial_state is not None:
        angle, state = initial_state
        want = ((batch, heads, num), (batch, heads, key_dim, value_dim), (batch, heads, key_dim),
                (batch, heads, value_dim))
        got = (angle.shape,) + tuple(x.shape for x in state)
        if got != want:
            raise ValueError(f"initial_state shapes must be {want}; got {got}")


def mamba3_siso_fused(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    d: Array | None = None,
    z: Array | None = None,
    *,
    segment_ids: Array | None = None,
    initial_state: tuple[Array, Mamba3State] | None = None,
    output_final_state: bool = False,
    chunk_size: int = CHUNK,
    stages: int | None = None,
    interpret: bool = False,
) -> tuple[Array, tuple[Array, Mamba3State] | None]:
    """Mamba-3 SISO layer core: :func:`jla.layers.ssm.mamba3_siso` on Mosaic GPU kernels.

    Arguments as in the official ``mamba3_siso_combined`` (FLA layout): ``q, k [B, T, G, K]`` (``C``,
    ``B``), ``v [B, T, H, V]`` (``x``), ``adt, dt, trap [B, T, H]`` (``A DT``, ``DT``, trapezoid
    logits), ``q_bias, k_bias [H, K]``, ``angles [B, T, H, S]`` (pre-``tanh``), ``d [H]``,
    ``z [B, T, H, V]``; the state is ``(angle [B, H, S], Mamba3State)``. Returns ``y`` in
    ``v.dtype``. ``q, k, v`` are computed in bfloat16, everything else in float32.
    """
    if chunk_size != CHUNK:
        raise ValueError(f"mosaic_gpu mamba3 uses chunk_size={CHUNK}; got {chunk_size}")
    _check_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, initial_state, segment_ids)
    batch, seq_len, groups, key_dim = q.shape
    heads, value_dim = v.shape[2:]
    num = angles.shape[-1]
    num_chunks = -(-seq_len // CHUNK)
    if stages is None:
        stages = 4 if batch * heads <= 64 else 2
    if not 1 <= stages <= 4:
        raise ValueError(f"stages must be in [1, 4]; got {stages}")
    segments = segment_ids is not None
    rc = RotaryConfig(batch=batch, heads=heads, groups=groups, num_chunks=num_chunks, key_dim=key_dim,
                      angle_dim=num, segments=segments, interpret=interpret)
    cfg = Config(batch=batch, heads=heads, num_chunks=num_chunks, key_dim=key_dim, value_dim=value_dim,
                 segments=segments, stages=min(stages, num_chunks), interpret=interpret)

    def pad(x):
        return jnp.pad(x, [(0, 0), (0, num_chunks * CHUNK - seq_len)] + [(0, 0)] * (x.ndim - 2))

    dt32, adt32 = dt.astype(_F32), adt.astype(_F32)
    lam = jax.nn.sigmoid(trap.astype(_F32))
    beta, gamma = (1.0 - lam) * dt32 * jnp.exp(adt32), lam * dt32
    gates = prepare_gates(adt32, beta, gamma, num_chunks=num_chunks, segment_ids=segment_ids)
    rows = angle_rows(dt32, num_chunks=num_chunks, segment_ids=segment_ids)
    qp, kp = (pad(_deinterleave(x.astype(_BF16), num)) for x in (q, k))
    qb, kb = (_deinterleave(x.astype(_F32), num) for x in (q_bias, k_bias))
    angle_state, ssm_state = (None, None) if initial_state is None else initial_state
    theta0 = jnp.zeros((batch, heads, num), _F32) if angle_state is None else angle_state.astype(_F32)
    qr, kr, theta_last = _rotary(rc, qp, kp, qb, kb, pad(angles.astype(_F32)), rows, theta0)
    if ssm_state is None:
        h0 = jnp.zeros((batch, heads, key_dim, value_dim), _F32)
    else:
        ssm, key, value = ssm_state
        permuted = Mamba3State(
            _deinterleave(ssm.astype(_F32), num, axis=2), _deinterleave(key.astype(_F32), num), value
        )
        h0 = _initial_mid_state(permuted, adt32, beta)
    o, final = _core(cfg, qr, kr, pad(v.astype(_BF16)), gates, h0)
    y = o[:, :seq_len].astype(_F32)
    if d is not None:
        y = y + d.astype(_F32)[:, None] * v.astype(_F32)
    if z is not None:
        y = y * jax.nn.silu(z.astype(_F32))
    y = y.astype(v.dtype)
    if not output_final_state:
        return y, None
    state = Mamba3State(
        _interleave(final, num, axis=2), _interleave(kr[:, seq_len - 1].astype(_F32), num), v[:, -1].astype(_F32)
    )
    return y, (jnp.mod(theta_last, _TWO_PI), state)


def _floating(*xs) -> bool:
    return all(x is None or jnp.issubdtype(x.dtype, jnp.floating) for x in xs)


def supports_fused(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    d: Array | None = None,
    z: Array | None = None,
    *,
    segment_ids: Array | None = None,
    initial_state: tuple[Array, Mamba3State] | None = None,
    output_final_state: bool = False,
    chunk_size: int = CHUNK,
    stages: int | None = None,
    interpret: bool = False,
    **unknown: Any,
) -> bool:
    """True only for calls the fused kernels are tested on: H100, bf16 ``C, B, x``, ``(K, V)`` in
    :data:`SUPPORTED_DIMS`, ``S`` in :func:`angle_dims` (``K / 4`` or ``K / 2``), chunk 64."""
    del output_final_state
    if unknown or chunk_size != CHUNK or (stages is not None and not 1 <= stages <= 4):
        return False
    try:
        _check_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, initial_state, segment_ids)
    except (ValueError, TypeError, AttributeError):
        return False
    if q.shape[1] == 0 or any(x.dtype != _BF16 for x in (q, k, v)):
        return False
    if not _floating(adt, dt, trap, q_bias, k_bias, angles, d, z):
        return False
    if segment_ids is not None and not jnp.issubdtype(segment_ids.dtype, jnp.integer):
        return False
    if initial_state is not None:
        angle, state = initial_state
        if not isinstance(state, Mamba3State) or not _floating(angle, *state):
            return False
    return interpret or _on_hopper()
