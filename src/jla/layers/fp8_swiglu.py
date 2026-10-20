# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Design follows Transformer Engine (NVIDIA, Apache-2.0), its fused cast-transpose and activation-cast
# kernels (transformer_engine/common/transpose/, transformer_engine/common/activation/): one pass that
# applies the activation (or its gradient), records the amax, casts to FP8 with the delayed scale and
# writes the transposed FP8 copy for the weight-gradient GEMM. No code was copied or adapted, so no
# licence text is bundled for it.
"""Delayed-scaling FP8 SwiGLU MLP with fused activation, quantization and transpose passes.

The unfused path (:class:`jla.layers.fp8.Fp8SwiGLU` with ``fp8_fused = false``) runs the gate/up
GEMM, the SwiGLU and the down GEMM as two FP8 matmuls around a plain-JAX SwiGLU. XLA then spends,
per MLP and train step (H100, ``transformer-1p3b``, 65,536 tokens, ``remat = "full"``): one pass that
recomputes the SwiGLU output and quantizes it, a second one that recomputes it again for its amax
(XLA does not merge the two because the SwiGLU's sigmoid is "expensive" to duplicate), and a third
that transposes the FP8 copy for the weight-gradient GEMM; in the backward pass the SwiGLU gradient
is written in bfloat16, read again to be quantized to e5m2 with its amax, and the FP8 copy is
transposed once more (``benchmarks/fp8_1p3b.md`` lists every pass and its time).

:func:`fp8_swiglu_mlp` keeps the same matmuls, scales and delayed-scaling state semantics as the
unfused path but runs the SwiGLU inside one custom VJP with two fused passes:

* forward (re-run under rematerialization): one pass over ``h = x W_gu`` computes
  ``a = silu(gate) * up``, its amax, the e4m3 ``a`` for the down GEMM and its transposed copy for
  the down weight gradient;
* backward: one pass over ``h`` and the down GEMM's input gradient computes the SwiGLU gradient, its
  amax, the e5m2 gradient for the gate/up input-gradient GEMM and its transposed copy for the
  gate/up weight gradient.

On a single GPU both passes are Pallas Triton kernels; elsewhere (CPU, and runs on more than one
device, where XLA's SPMD partitioner cannot split a Pallas call) they are the same arithmetic in
plain JAX, and :func:`fp8_swiglu_mlp` logs which one it traced. On several GPUs under a context
mesh (the trainer's) the Pallas passes run per shard instead (:func:`_swiglu_quantize_sharded`:
tokens over the data axes, ``d_ff`` over the ``mlp`` axes, the amax reduced over the shards).

Numerics. The forward pass quantizes the float32 SwiGLU straight to e4m3 (``clip(a / scale)``,
round to nearest even) and takes its amax in float32, as Transformer Engine's activation-cast
kernels do. This is also what the jitted unfused path computes with XLA's default
``--xla_allow_excess_precision=true``: XLA drops the SwiGLU's bfloat16 rounding inside the fusion
that quantizes it, so in a jitted train step on a GPU the fused and unfused paths give bitwise-equal
outputs, input gradients and down-projection weight gradients (H100, tested; with the Pallas passes
the gate/up weight gradient differs in FP8 GEMM accumulation order only, because the transposed
copy comes from the kernel and cuBLASLt gets another layout). Run eagerly, on CPU (XLA's CPU
compiler keeps the rounding) or with that flag off, the unfused path rounds the SwiGLU to bfloat16
first, which moves 1-2% of the e4m3 values by one unit in the last place. The backward pass rounds the SwiGLU gradient to the compute dtype before its e5m2 cast,
like the unfused path, which writes that gradient to memory in the compute dtype.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from jax.ad_checkpoint import checkpoint_name

from jla.layers.fp8 import (E4M3, E5M2, Fp8State, _forward_operand, _fp8_max, _scaled_dot, push_amax, quantize,
                            scale_from_amax)
from jla.layers.linear import LINEAR_CHECKPOINT_NAME
from jla.layers.mlp import MLP_HIDDEN_CHECKPOINT_NAME

__all__ = ["fp8_swiglu_mlp", "swiglu_backward_quantize", "swiglu_quantize"]

Array = jax.Array
_log = logging.getLogger(__name__)
_LOGGED: set[tuple[int, int, int]] = set()

_BLOCK_M = 64
_BLOCK_F = 128
# Warps per program of the backward pass (4 is Pallas Triton's default; the forward keeps it). H100,
# [65536, 2, 5632]: 1.52 ms against 1.86 ms with 4 warps, the forward 0.87 ms with 4 against 0.91 ms with
# 8; 128-token or 64-feature blocks were slower for both (benchmarks/fp8_1p3b.md). Same values.
_BWD_NUM_WARPS = 8
_FWD_PRECISION = lax.Precision.DEFAULT
_BWD_PRECISION = lax.Precision.HIGHEST


# -- the two fused passes: plain-JAX reference ------------------------------------------------------


def _swiglu_f32(h: Array) -> Array:
    """``silu(gate) * up`` of ``h [M, 2, F]`` in float32 (quantized without a compute-dtype rounding)."""
    gate, up = h[:, 0, :].astype(jnp.float32), h[:, 1, :].astype(jnp.float32)
    return jax.nn.silu(gate) * up


def _swiglu_grad_values(h: Array, da: Array, dtype: Any) -> Array:
    """The gradient of ``h [M, 2, F]`` through the SwiGLU for ``da [M, F]``, rounded to ``dtype``."""
    gate, up = h[:, 0, :].astype(jnp.float32), h[:, 1, :].astype(jnp.float32)
    d = da.astype(jnp.float32)
    sig = jax.nn.sigmoid(gate)
    dgate = d * up * (sig + gate * sig * (1.0 - sig))
    dup = d * gate * sig
    return jnp.stack([dgate, dup], axis=1).astype(dtype)


def _swiglu_quantize_xla(h: Array, scale: Array):
    a = _swiglu_f32(h)
    q = quantize(a, scale, E4M3)
    return q, q.T, jnp.max(jnp.abs(a))


def _swiglu_backward_quantize_xla(h: Array, da: Array, scale: Array, dtype: Any):
    dh = _swiglu_grad_values(h, da, dtype)
    q = quantize(dh, scale, E5M2)
    return q, jnp.transpose(q, (1, 2, 0)), jnp.max(jnp.abs(dh)).astype(jnp.float32)


# -- the two fused passes: Pallas Triton kernels -----------------------------------------------------


def _fwd_kernel(h_ref, s_ref, q_ref, qt_ref, amax_ref, *, bound):
    gate = h_ref[:, 0, :].astype(jnp.float32)
    up = h_ref[:, 1, :].astype(jnp.float32)
    a = gate * jax.nn.sigmoid(gate) * up
    q = jnp.clip(a / s_ref[0], -bound, bound).astype(E4M3)
    q_ref[...] = q
    qt_ref[...] = q.T
    amax_ref[...] = jnp.full((1, 1), jnp.max(jnp.abs(a)), jnp.float32)


def _bwd_kernel(h_ref, da_ref, s_ref, q_ref, qt_ref, amax_ref, *, dtype, bound):
    gate = h_ref[:, 0, :].astype(jnp.float32)
    up = h_ref[:, 1, :].astype(jnp.float32)
    d = da_ref[...].astype(jnp.float32)
    sig = jax.nn.sigmoid(gate)
    dgate = (d * up * (sig + gate * sig * (1.0 - sig))).astype(dtype).astype(jnp.float32)
    dup = (d * gate * sig).astype(dtype).astype(jnp.float32)
    s = s_ref[0]
    qg = jnp.clip(dgate / s, -bound, bound).astype(E5M2)
    qu = jnp.clip(dup / s, -bound, bound).astype(E5M2)
    q_ref[:, 0, :] = qg
    q_ref[:, 1, :] = qu
    qt_ref[0, :, :] = qg.T
    qt_ref[1, :, :] = qu.T
    amax = jnp.maximum(jnp.max(jnp.abs(dgate)), jnp.max(jnp.abs(dup)))
    amax_ref[...] = jnp.full((1, 1), amax, jnp.float32)


def _grid(m: int, f: int) -> tuple[int, int]:
    return m // _BLOCK_M, f // _BLOCK_F


def _swiglu_quantize_pallas(h: Array, scale: Array):
    from jax.experimental import pallas as pl

    m, _, f = h.shape
    gm, gf = _grid(m, f)
    kernel = functools.partial(_fwd_kernel, bound=_fp8_max(E4M3))
    q, qt, amax = pl.pallas_call(
        kernel,
        grid=(gm, gf),
        in_specs=[
            pl.BlockSpec((_BLOCK_M, 2, _BLOCK_F), lambda i, j: (i, 0, j)),
            pl.BlockSpec((1,), lambda i, j: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((_BLOCK_M, _BLOCK_F), lambda i, j: (i, j)),
            pl.BlockSpec((_BLOCK_F, _BLOCK_M), lambda i, j: (j, i)),
            pl.BlockSpec((1, 1), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, f), E4M3),
            jax.ShapeDtypeStruct((f, m), E4M3),
            jax.ShapeDtypeStruct((gm, gf), jnp.float32),
        ],
        name="fp8_swiglu_quantize",
    )(h, jnp.reshape(scale, (1,)).astype(jnp.float32))
    return q, qt, jnp.max(amax)


def _swiglu_backward_quantize_pallas(h: Array, da: Array, scale: Array, dtype: Any):
    from jax.experimental import pallas as pl

    m, _, f = h.shape
    gm, gf = _grid(m, f)
    kernel = functools.partial(_bwd_kernel, dtype=dtype, bound=_fp8_max(E5M2))
    q, qt, amax = pl.pallas_call(
        kernel,
        grid=(gm, gf),
        in_specs=[
            pl.BlockSpec((_BLOCK_M, 2, _BLOCK_F), lambda i, j: (i, 0, j)),
            pl.BlockSpec((_BLOCK_M, _BLOCK_F), lambda i, j: (i, j)),
            pl.BlockSpec((1,), lambda i, j: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((_BLOCK_M, 2, _BLOCK_F), lambda i, j: (i, 0, j)),
            pl.BlockSpec((2, _BLOCK_F, _BLOCK_M), lambda i, j: (0, j, i)),
            pl.BlockSpec((1, 1), lambda i, j: (i, j)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, 2, f), E5M2),
            jax.ShapeDtypeStruct((2, f, m), E5M2),
            jax.ShapeDtypeStruct((gm, gf), jnp.float32),
        ],
        name="fp8_swiglu_backward_quantize",
        compiler_params=_triton_params(_BWD_NUM_WARPS),
    )(h, da, jnp.reshape(scale, (1,)).astype(jnp.float32))
    return q, qt, jnp.max(amax)


def _triton_params(num_warps: int):
    from jax.experimental.pallas import triton as pltriton

    return pltriton.CompilerParams(num_warps=num_warps, num_stages=1)


def _use_pallas(m: int, f: int, impl: str) -> bool:
    if impl == "xla":
        return False
    fits = m % _BLOCK_M == 0 and f % _BLOCK_F == 0
    if impl == "pallas":
        if not fits:
            raise ValueError(f"the fused FP8 SwiGLU kernels need tokens % {_BLOCK_M} == 0 and d_ff % {_BLOCK_F} == 0;"
                             f" got {m} tokens and d_ff {f}")
        return True
    # A Pallas call is opaque to XLA's SPMD partitioner, which would gather its operands on every
    # device of a sharded run; with more than one device "auto" keeps the plain-JAX passes, except in the
    # per-device micro-step loop of train.grad_reduce = "step" (one device's call; other fully manual
    # shard_map bodies, such as the pipeline-parallel stages, keep the plain-JAX passes).
    from jla.distributed.kernel_sharding import in_per_device_step

    use = fits and jax.default_backend() == "gpu" and (jax.device_count() == 1 or in_per_device_step())
    if not use and jax.default_backend() == "gpu" and (m, f, jax.device_count()) not in _LOGGED:
        _LOGGED.add((m, f, jax.device_count()))
        _log.warning("fp8_fused: plain-JAX SwiGLU passes instead of the Pallas kernels (%d tokens, d_ff %d, "
                     "%d devices; the kernels need one device or the per-device step of train.grad_reduce = "
                     "'step', tokens %% %d == 0 and d_ff %% %d == 0)",
                     m, f, jax.device_count(), _BLOCK_M, _BLOCK_F)
    return use


def swiglu_quantize(h: Array, scale: Array, *, impl: str = "auto"):
    """``(q [M, F], q^T [F, M], amax)``: e4m3 ``clip(swiglu(h) / scale)`` of ``h [M, 2, F]`` and its amax,
    both from the float32 SwiGLU.

    ``impl``: ``"pallas"`` (Pallas Triton, GPU), ``"xla"`` (plain JAX) or ``"auto"`` (Pallas on a
    single GPU when the shape tiles, else plain JAX; on several GPUs per shard of the context mesh,
    :func:`_swiglu_quantize_sharded`).
    """
    if _per_shard(impl):
        return _swiglu_quantize_sharded(h, scale)
    if _use_pallas(h.shape[0], h.shape[2], impl):
        return _swiglu_quantize_pallas(h, scale)
    return _swiglu_quantize_xla(h, scale)


def swiglu_backward_quantize(h: Array, da: Array, scale: Array, dtype: Any, *, impl: str = "auto"):
    """``(q [M, 2, F], q^T [2, F, M], amax)``: e5m2 ``clip(dh / scale)`` of the SwiGLU gradient ``dh``."""
    if _per_shard(impl):
        return _swiglu_backward_quantize_sharded(h, da, scale, dtype)
    if _use_pallas(h.shape[0], h.shape[2], impl):
        return _swiglu_backward_quantize_pallas(h, da, scale, dtype)
    return _swiglu_backward_quantize_xla(h, da, scale, dtype)


# -- the fused MLP ------------------------------------------------------------------------------------


def _scales(state: Fp8State, grad_max: float) -> tuple[Array, Array, Array]:
    e4m3 = _fp8_max(E4M3)
    return (
        scale_from_amax(jnp.max(state.input_amax_history), e4m3, state.input_scale),
        scale_from_amax(jnp.max(state.kernel_amax_history), e4m3, state.kernel_scale),
        scale_from_amax(jnp.max(state.grad_amax_history), grad_max, state.grad_scale),
    )


def _flat(x: Array) -> Array:
    return jnp.reshape(x, (-1, x.shape[-1]))


def _primal(x, w_gu, w_dn, st_gu, st_dn, dtypes):
    x_dtype, _, _ = dtypes
    xs, ws_gu, _ = _scales(st_gu, _fp8_max(E5M2))
    as_, ws_dn, _ = _scales(st_dn, _fp8_max(E5M2))
    qx = quantize(_flat(x), xs, E4M3)
    h = _scaled_dot(qx, xs, quantize(w_gu, ws_gu, E4M3), ws_gu, (((1,), (0,)), ((), ())), x_dtype, _FWD_PRECISION)
    a = _swiglu_f32(h)
    y = _scaled_dot(quantize(a, as_, E4M3), as_, quantize(w_dn, ws_dn, E4M3), ws_dn, (((1,), (0,)), ((), ())),
                    x_dtype, _FWD_PRECISION)
    return jnp.reshape(y, (*x.shape[:-1], w_dn.shape[-1]))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, dtypes, impl):
    return _primal(x, w_gu, w_dn, st_gu, st_dn, dtypes)


def _fwd(x, w_gu, w_dn, st_gu, st_dn, dtypes, impl, prepared=None):
    x_dtype, _, _ = dtypes
    xs, ws_gu, _ = _scales(st_gu, _fp8_max(E5M2))
    as_, ws_dn, _ = _scales(st_dn, _fp8_max(E5M2))
    xf = _flat(x)
    qx = quantize(xf, xs, E4M3)
    if prepared is None:
        qw_gu, qw_dn = quantize(w_gu, ws_gu, E4M3), quantize(w_dn, ws_dn, E4M3)
        h = _scaled_dot(qx, xs, qw_gu, ws_gu, (((1,), (0,)), ((), ())), x_dtype, _FWD_PRECISION)
        w_amaxes = jnp.max(jnp.abs(w_gu)).astype(jnp.float32), jnp.max(jnp.abs(w_dn)).astype(jnp.float32)
    else:  # the step's FP8 weight copies (jla.layers.fp8.prepare_fp8_weight): qt [2, F, D] and [D', F]
        pw_gu, pw_dn = prepared
        qw_gu, qw_dn = pw_gu.q, pw_dn.q
        operand, dims = _forward_operand(pw_gu, 2)
        h = _scaled_dot(qx, xs, operand, ws_gu, dims, x_dtype, _FWD_PRECISION)
        w_amaxes = pw_gu.amax, pw_dn.amax
    # The checkpoint names of the unfused SwiGLU (gate/up product) and of its down projection, so the
    # "names" remat policies can keep them (identity tags; the values are unchanged).
    h = checkpoint_name(h, MLP_HIDDEN_CHECKPOINT_NAME)
    qa, qat, a_amax = swiglu_quantize(h, as_, impl=impl)
    if prepared is None:
        y = _scaled_dot(qa, as_, qw_dn, ws_dn, (((1,), (0,)), ((), ())), x_dtype, _FWD_PRECISION)
    else:
        operand, dims = _forward_operand(prepared[1], 2)
        y = _scaled_dot(qa, as_, operand, ws_dn, dims, x_dtype, _FWD_PRECISION)
    y = checkpoint_name(y, LINEAR_CHECKPOINT_NAME)
    amaxes = (jnp.max(jnp.abs(xf)).astype(jnp.float32), w_amaxes[0], a_amax, w_amaxes[1])
    residuals = (qx, qw_gu, qw_dn, h, qat, (xs, ws_gu, as_, ws_dn), amaxes, st_gu, st_dn)
    return jnp.reshape(y, (*x.shape[:-1], w_dn.shape[-1])), residuals


def _next(state: Fp8State, scales: tuple[Array, Array, Array], amaxes: tuple[Array, Array, Array]) -> Fp8State:
    return Fp8State(
        input_scale=scales[0],
        input_amax_history=push_amax(state.input_amax_history, amaxes[0]),
        kernel_scale=scales[1],
        kernel_amax_history=push_amax(state.kernel_amax_history, amaxes[1]),
        grad_scale=scales[2],
        grad_amax_history=push_amax(state.grad_amax_history, amaxes[2]),
    )


def _bwd(dtypes, impl, residuals, g):
    x_dtype, gu_dtype, dn_dtype = dtypes
    qx, qw_gu, qw_dn, h, qat, (xs, ws_gu, as_, ws_dn), (x_amax, wgu_amax, a_amax, wdn_amax), st_gu, st_dn = residuals
    x_shape = (*g.shape[:-1], qw_gu.shape[0])
    e5m2 = _fp8_max(E5M2)
    gs_dn = scale_from_amax(jnp.max(st_dn.grad_amax_history), e5m2, st_dn.grad_scale)
    gs_gu = scale_from_amax(jnp.max(st_gu.grad_amax_history), e5m2, st_gu.grad_scale)
    gf = _flat(g)
    qg = quantize(gf, gs_dn, E5M2)
    # Down projection: da = g W_dn^T (bf16) and dW_dn = a^T g.
    da = _scaled_dot(qg, gs_dn, qw_dn, ws_dn, (((1,), (1,)), ((), ())), x_dtype, _BWD_PRECISION)
    dw_dn = _scaled_dot(qat, as_, qg, gs_dn, (((1,), (0,)), ((), ())), dn_dtype, _BWD_PRECISION)
    # SwiGLU gradient: e5m2, its transposed copy and its amax in one pass.
    qdh, qdht, dh_amax = swiglu_backward_quantize(h, da, gs_gu, x_dtype, impl=impl)
    dx = _scaled_dot(qdh, gs_gu, qw_gu, ws_gu, (((1, 2), (1, 2)), ((), ())), x_dtype, _BWD_PRECISION)
    m, f = qdh.shape[0], qdh.shape[2]
    dw_gu = _scaled_dot(qx.T, xs, jnp.reshape(qdht, (2 * f, m)), gs_gu, (((1,), (1,)), ((), ())), gu_dtype,
                        _BWD_PRECISION)
    dw_gu = jnp.reshape(dw_gu, qw_gu.shape)
    new_gu = _next(st_gu, (xs, ws_gu, gs_gu), (x_amax, wgu_amax, dh_amax))
    new_dn = _next(st_dn, (as_, ws_dn, gs_dn), (a_amax, wdn_amax, jnp.max(jnp.abs(gf)).astype(jnp.float32)))
    return jnp.reshape(dx, x_shape), dw_gu, dw_dn, new_gu, new_dn


_fp8_swiglu_mlp.defvjp(_fwd, _bwd)


def fp8_swiglu_mlp(x: Array, w_gu: Array, w_dn: Array, st_gu: Fp8State, st_dn: Fp8State, *, impl: str = "auto",
                   prepared: Any = None) -> Array:
    """``down(swiglu(x @ w_gu))`` with delayed-scaling FP8 GEMMs and the fused SwiGLU passes.

    ``x [..., D]`` in bfloat16 or float32, ``w_gu [D, 2, F]``, ``w_dn [F, D']``; ``st_gu`` and ``st_dn``
    are the two matmuls' :class:`~jla.layers.fp8.Fp8State` (their "gradients" are their next values,
    exactly as for :func:`jla.layers.fp8.fp8_dot`). Returns ``[..., D']`` in ``x``'s dtype.
    ``prepared``: the pair of :class:`~jla.layers.fp8.Fp8Weight` the train step made once for this optimizer
    step (:func:`jla.layers.fp8.prepare_fp8_weight`); the GEMMs then read them instead of quantizing (and
    transposing) both weights in every call. Same values, gradients and states.
    """
    if impl not in ("auto", "pallas", "xla"):
        raise ValueError(f"impl must be 'auto', 'pallas' or 'xla'; got {impl!r}")
    if x.shape[-1] != w_gu.shape[0] or w_gu.ndim != 3 or w_gu.shape[1] != 2 or w_dn.shape[0] != w_gu.shape[2]:
        raise ValueError(f"fp8_swiglu_mlp needs x [..., D], w_gu [D, 2, F], w_dn [F, D']; got "
                         f"{x.shape}, {w_gu.shape}, {w_dn.shape}")
    if jnp.dtype(x.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise TypeError(f"FP8 matmuls need a bfloat16 or float32 compute dtype; got {x.dtype}")
    if not (isinstance(st_gu, Fp8State) and isinstance(st_dn, Fp8State)):
        raise ValueError("fp8_swiglu_mlp needs both GEMMs' delayed-scaling Fp8State (fp8_fused works with "
                         "fp8 = 'delayed' only)")
    dtypes = (str(x.dtype), str(w_gu.dtype), str(w_dn.dtype))
    if prepared is not None:
        pw_gu, pw_dn = prepared
        if pw_gu.q.shape != w_gu.shape or pw_dn.q.shape != w_dn.shape:
            raise ValueError(f"prepared FP8 weights {pw_gu.q.shape}, {pw_dn.q.shape} do not match {w_gu.shape}, "
                             f"{w_dn.shape}")
        return _fp8_swiglu_mlp_prepared(x, w_gu, w_dn, pw_gu, pw_dn, st_gu, st_dn, dtypes, impl)
    return _fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, dtypes, impl)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _fp8_swiglu_mlp_prepared(x, w_gu, w_dn, pw_gu, pw_dn, st_gu, st_dn, dtypes, impl):
    return _fwd(x, w_gu, w_dn, st_gu, st_dn, dtypes, impl, (pw_gu, pw_dn))[0]


def _fwd_prepared(x, w_gu, w_dn, pw_gu, pw_dn, st_gu, st_dn, dtypes, impl):
    y, residuals = _fwd(x, w_gu, w_dn, st_gu, st_dn, dtypes, impl, (pw_gu, pw_dn))
    return y, (residuals, pw_gu, pw_dn)


def _bwd_prepared(dtypes, impl, residuals, g):
    residuals, pw_gu, pw_dn = residuals
    dx, dw_gu, dw_dn, new_gu, new_dn = _bwd(dtypes, impl, residuals, g)
    zeros = functools.partial(jax.tree.map, jnp.zeros_like)
    return dx, dw_gu, dw_dn, zeros(pw_gu), zeros(pw_dn), new_gu, new_dn


_fp8_swiglu_mlp_prepared.defvjp(_fwd_prepared, _bwd_prepared)


# -- several GPUs: the Pallas passes per shard of the context mesh -------------------------------------------


def _per_shard(impl: str) -> bool:
    """``impl="auto"`` on several GPUs under a context mesh that can split the call (the trainer's)."""
    if impl != "auto" or jax.default_backend() != "gpu":
        return False
    from jla.distributed.kernel_sharding import mesh_can_split, per_device_call

    return not per_device_call() and mesh_can_split()


def _tiles(m: int, f: int) -> bool:
    return m % _BLOCK_M == 0 and f % _BLOCK_F == 0


def _swiglu_quantize_sharded(h: Array, scale: Array):
    """:func:`swiglu_quantize` per shard of the context mesh (``jla.distributed.kernel_sharding``).

    The rule: ``h [M, 2, F]`` split along its tokens over the ``batch`` rule's mesh axes (the tokens of a
    ``[B, T, D]`` activation split along ``B``) and along ``F`` over the ``mlp`` rule's; every other mesh axis
    of size > 1 splits the tokens too. ``q`` comes back split like ``h``, ``q^T [F, M]`` with the two dims
    swapped, and the amax as one partial per device (reduced here). The pass is elementwise apart from the
    amax, so each device's block is exactly what one GPU computes for it. When no split tiles (tokens % 64,
    ``F`` % 128 per device) the plain-JAX pass runs, which XLA partitions.
    """
    from jla.distributed.kernel_sharding import SHARDS, rule_axes, shard_layout_call

    def body(h, scale):
        q, qt, amax = _swiglu_quantize_pallas(h, scale)
        return q, qt, jnp.reshape(amax, (1,))

    def unsplit(h, scale):
        q, qt, amax = _swiglu_quantize_xla(h, scale)
        return q, qt, jnp.reshape(amax, (1,))

    q, qt, amax = shard_layout_call(
        body, {"h": h, "scale": scale}, {"h": "token", "scale": "replicated"},
        out_layouts=("state", (1, 0), SHARDS), heads=h.shape[2], head_axes=rule_axes("mlp"),
        supports=lambda h, scale: _tiles(h.shape[0], h.shape[2]), unsplit=unsplit)
    return q, qt, jnp.max(amax)


def _swiglu_backward_quantize_sharded(h: Array, da: Array, scale: Array, dtype: Any):
    """:func:`swiglu_backward_quantize` per shard of the context mesh: the rule of
    :func:`_swiglu_quantize_sharded`, with ``da [M, F]`` split like ``q`` there, ``q [M, 2, F]`` like ``h`` and
    ``q^T [2, F, M]`` with the token and feature dims swapped."""
    from jla.distributed.kernel_sharding import SHARDS, rule_axes, shard_layout_call

    def body(h, da, scale):
        q, qt, amax = _swiglu_backward_quantize_pallas(h, da, scale, dtype)
        return q, qt, jnp.reshape(amax, (1,))

    def unsplit(h, da, scale):
        q, qt, amax = _swiglu_backward_quantize_xla(h, da, scale, dtype)
        return q, qt, jnp.reshape(amax, (1,))

    q, qt, amax = shard_layout_call(
        body, {"h": h, "da": da, "scale": scale}, {"h": "token", "da": "state", "scale": "replicated"},
        out_layouts=("token", (2, 1), SHARDS), heads=h.shape[2], head_axes=rule_axes("mlp"),
        supports=lambda h, da, scale: _tiles(h.shape[0], h.shape[2]), unsplit=unsplit)
    return q, qt, jnp.max(amax)
