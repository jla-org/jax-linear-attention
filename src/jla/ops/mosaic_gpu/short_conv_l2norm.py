# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright (c) 2026 Mayank Mishra.
# Adapted from FLAT (Apache-2.0), flat/depthwise_causal_convolution/pallas_implementation/{forward,backward}.py
"""Short conv + SiLU + per-head L2 norm in one Mosaic GPU kernel, forward and backward.

``n_t = l2norm(act(b + sum_i w[:, i] * x_{t - (K - 1 - i)}))`` with the L2 norm taken over each head's
``head_dim`` consecutive channels, ``l2norm(y) = y * rsqrt(sum(y^2) + 1e-6)`` as :func:`jla.ops.reference.l2norm`.
This is the q / k prologue of the delta-rule mixers (Gated DeltaNet, GDN-2, KDA): with it the normalized q and k
come straight out of the conv kernel and the recurrent kernels run with ``use_qk_l2norm=False``. Without it the
conv writes ``act(...)`` in bf16 and XLA reads it back for the norm (forward, recomputed forward and backward).

The kernels are those of :mod:`jla.ops.mosaic_gpu.short_conv` (same plan, layout, segment masks, initial state and
shifted last block) with the norm added to each row: the conv output stays in float32 registers, each head's sum of
squares is a butterfly shuffle over the ``head_dim / vec`` threads that hold the head (one warp at ``head_dim =
128``, 4 channels per thread), and the output is rounded to bf16 once. The backward recomputes the row, forms
``dy = inv * (dn - n * sum(dn * n))`` (two head sums) and continues with the conv backward of ``short_conv``. The
unfused path rounds the conv output to bf16 before the norm; this one does not, so it is the more accurate of the
two (``tests/test_mosaic_gpu_short_conv_l2norm.py``).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, vector

from jla.ops.mosaic_gpu import short_conv as sc

__all__ = ["HEAD_DIMS", "short_conv_l2norm", "short_conv_l2norm_sharded", "unsupported_reason"]

Array = jax.Array

HEAD_DIMS = (64, 128)  # the head sizes the GPU tests cover
EPS = 1e-6  # that of jla.ops.reference.l2norm
_F32 = jnp.float32
_WARP = 32


def _plan(x: Array, width: int, activation: bool, has_state: bool, has_mask: bool, head_dim: int, *,
          backward: bool) -> sc._Plan:
    """``short_conv``'s plan; a head must sit in one aligned group of at most 32 threads (one warp)."""
    p = sc._plan(x, width, activation, has_state, has_mask, backward=backward)
    lanes = head_dim // p.vec
    if head_dim % p.vec or not 1 <= lanes <= _WARP or lanes & (lanes - 1):
        raise ValueError(f"a {head_dim}-channel head at {p.vec} channels per thread does not fit one warp")
    return p


def _head_lanes(x: Array, width: int, head_dim: int) -> int:
    """Threads holding one head (the butterfly width) under both plans, or 0 when they differ or do not fit."""
    lanes = set()
    for backward in (False, True):
        vec = sc._plan(x, width, True, False, False, backward=backward).vec
        lanes.add(head_dim // vec if head_dim % vec == 0 else 0)
    (n,) = lanes if len(lanes) == 1 else (0,)
    return n if n & (n - 1) == 0 else 0


def _head_sum(line, block_c: int, lanes: int):
    """Sum of each head's channels, broadcast back to all of them: the ``lanes`` consecutive threads of a head
    (an aligned group within one warp) each add their ``vec`` registers, then exchange partial sums with
    ``log2(lanes)`` butterfly shuffles. Control flow is uniform across the warpgroup wherever this is called."""

    @plgpu.inline_mgpu(arg_types=(line,), return_type=plgpu.ShapeDtypeStruct((block_c,), _F32, layout=line))
    def head_sum(ctx, x):
        del ctx
        out = np.empty(x.registers.shape, dtype=object)
        for idx, reg in np.ndenumerate(x.registers):
            [n] = ir.VectorType(reg.type).shape
            parts = [vector.extract(reg, dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([i]))
                     for i in range(n)]
            total = parts[0]
            for part in parts[1:]:
                total = arith.addf(total, part)
            stride = 1
            while stride < lanes:
                total = arith.addf(total, mgpu_utils.shfl_bfly(total, stride))
                stride *= 2
            out[idx] = vector.broadcast(reg.type, total)
        return mgpu.FragmentedArray(_registers=out, _layout=x.layout, _is_signed=None)

    return head_sum


def _forward_call(p: sc._Plan, head_dim: int, x: Array, w_t: Array, bias: Array, state: Array | None,
                  mask: Array | None) -> Array:
    k, bt, u, bc = p.width, p.block_t, p.group, p.block_c
    t_len = p.seq_len
    line = plgpu.Layout.WG_STRIDED((bc,), vec_size=p.vec)
    head_sum = _head_sum(line, bc, head_dim // p.vec)
    dtype = x.dtype
    groups = -(-bt // u)
    guard = groups * u != bt  # the last group runs past the block

    def body(*refs):
        (x_ref, w_ref, b_ref), state_ref, mask_ref, (y_ref,) = sc._split_refs(p, refs, len(inputs))
        cb, b, tb = lax.axis_index("c"), lax.axis_index("b"), lax.axis_index("t")
        cols = pl.ds(cb * bc, bc)
        t0 = jnp.minimum(tb * bt, t_len - bt)
        w = [sc._row(w_ref.at[i, cols], line) for i in range(k)]
        bias_v = sc._row(b_ref.at[cols], line)
        prev = tuple(sc._history(p, x_ref, state_ref, b, t0 - lag, cols, line) for lag in range(1, k))

        def load_group(g):  # the group's loads are issued before any of its arithmetic
            rows = []
            for r in range(u):
                tc = jnp.minimum(t0 + g * u + r, t_len - 1)
                m = mask_ref[b, tc] if mask_ref is not None else None
                rows.append((sc._row(x_ref.at[b, tc, cols], line), m))
            return tuple(rows)

        def group_step(g, carry):
            prev, rows = carry
            if p.prefetch:  # in flight while this group computes (the last group loads itself again)
                ahead = load_group(jnp.minimum(g + 1, groups - 1))
            else:
                rows = load_group(g)
            for r, (xt, m) in enumerate(rows):
                t = t0 + g * u + r
                window = (xt, *prev)  # window[lag] = x_{t - lag}
                acc = bias_v
                for i in range(k):
                    lag = k - 1 - i
                    tap = w[i] * window[lag].astype(_F32)
                    if m is not None and lag:
                        tap = jnp.where(sc._bit(m, lag), tap, 0.0)
                    acc = acc + tap
                if p.activation:
                    acc = acc * jax.nn.sigmoid(acc)
                out = (acc * lax.rsqrt(head_sum(acc * acc) + EPS)).astype(dtype)
                if guard:
                    @pl.when(t < t0 + bt)
                    def _store(t=t, out=out):
                        y_ref.at[b, t, cols][...] = out
                else:
                    y_ref.at[b, t, cols][...] = out
                prev = window[: k - 1]
            return prev, ahead if p.prefetch else ()

        lax.fori_loop(0, groups, group_step, (prev, load_group(0) if p.prefetch else ()))

    inputs = [x, w_t, bias] + ([state] if p.has_state else []) + ([mask] if p.has_mask else [])
    kern = sc._kernel(
        body, "short_conv_l2norm_fwd", out_type=jax.ShapeDtypeStruct(x.shape, dtype),
        grid=(p.channels // bc, p.batch, p.num_blocks), grid_names=("c", "b", "t"),
    )
    return kern(*inputs)


def _backward_call(p: sc._Plan, head_dim: int, x: Array, dn: Array, w_t: Array, bias: Array,
                   state: Array | None, mask: Array | None) -> tuple[Array, Array]:
    """``(dx [B, T, C], partials [B, NB, K + 1, C] f32)`` as ``short_conv._backward_call``, from ``dn``, the
    gradient of the normalized output."""
    k, bt, u, bc = p.width, p.block_t, p.group, p.block_c
    t_len = p.seq_len
    line = plgpu.Layout.WG_STRIDED((bc,), vec_size=p.vec)
    head_sum = _head_sum(line, bc, head_dim // p.vec)
    dtype = x.dtype
    steps = bt + k - 1  # dpre rows t0 .. t0 + bt + k - 2 (the last k - 1 feed dx of the block's tail)
    groups = -(-steps // u)

    def body(*refs):
        (x_ref, dy_ref, w_ref, b_ref), state_ref, mask_ref, (dx_ref, part_ref) = sc._split_refs(
            p, refs, len(inputs))
        cb, b, tb = lax.axis_index("c"), lax.axis_index("b"), lax.axis_index("t")
        cols = pl.ds(cb * bc, bc)
        t0 = jnp.minimum(tb * bt, t_len - bt)
        own = tb * bt  # rows [own, end) count toward dw and db (the shifted last block overlaps)
        end = t0 + bt
        w = [sc._row(w_ref.at[i, cols], line) for i in range(k)]
        bias_v = sc._row(b_ref.at[cols], line)
        zero = bias_v * 0.0
        xprev = tuple(sc._history(p, x_ref, state_ref, b, t0 - lag, cols, line) for lag in range(1, k))
        dprev = tuple(zero for _ in range(1, k))  # rows before t0 only feed dx rows before t0
        mprev = tuple(mask_ref[b, jnp.maximum(t0 - lag, 0)] for lag in range(1, k)) if p.has_mask else ()
        acc_w = tuple(zero for _ in range(k))

        def load_group(g):
            rows = []
            for r in range(u):
                tc = jnp.minimum(t0 + g * u + r, t_len - 1)
                m = mask_ref[b, tc] if mask_ref is not None else None
                rows.append((sc._row(x_ref.at[b, tc, cols], line), sc._row(dy_ref.at[b, tc, cols], line), m))
            return tuple(rows)

        def group_step(g, carry):
            xprev, dprev, mprev, acc_w, acc_b, rows = carry
            if p.prefetch:
                ahead = load_group(jnp.minimum(g + 1, groups - 1))
            else:
                rows = load_group(g)
            for r, (xt, dyt, m) in enumerate(rows):
                t = t0 + g * u + r
                window = (xt, *xprev)  # window[lag] = x_{t - lag}
                taps = []
                pre = bias_v
                for i in range(k):
                    lag = k - 1 - i
                    xv = window[lag].astype(_F32)
                    if m is not None and lag:
                        xv = jnp.where(sc._bit(m, lag), xv, 0.0)
                    taps.append(xv)
                    pre = pre + w[i] * xv
                if p.activation:
                    sig = jax.nn.sigmoid(pre)
                    y = pre * sig
                else:
                    y = pre
                inv = lax.rsqrt(head_sum(y * y) + EPS)
                n = y * inv
                dn_t = dyt.astype(_F32)
                grad = inv * (dn_t - n * head_sum(dn_t * n))  # through the L2 norm
                if p.activation:
                    grad = grad * sig * (1.0 + pre * (1.0 - sig))
                grad = jnp.where(t < t_len, grad, 0.0)
                owned = jnp.where((t >= own) & (t < end), grad, 0.0)
                acc_w = tuple(acc_w[i] + owned * taps[i] for i in range(k))
                acc_b = acc_b + owned
                gwin = (grad, *dprev)  # gwin[j] = g_{t - j}
                mwin = (m, *mprev) if m is not None else ()
                dx = None
                for i in range(k):
                    term = w[i] * gwin[i]
                    lag = k - 1 - i
                    if mwin and lag:
                        term = jnp.where(sc._bit(mwin[i], lag), term, 0.0)
                    dx = term if dx is None else dx + term
                row = t - (k - 1)  # dx_row = sum_i w_i g_{row + k - 1 - i}

                @pl.when((row >= t0) & (row < end))
                def _store(row=row, dx=dx):
                    dx_ref.at[b, row, cols][...] = dx.astype(dtype)

                xprev, dprev = window[: k - 1], gwin[: k - 1]
                mprev = mwin[: k - 1] if mwin else ()
            return xprev, dprev, mprev, acc_w, acc_b, ahead if p.prefetch else ()

        init = (xprev, dprev, mprev, acc_w, zero, load_group(0) if p.prefetch else ())
        _, _, _, acc_w, acc_b, _ = lax.fori_loop(0, groups, group_step, init)
        for i in range(k):
            part_ref.at[b, tb, i, cols][...] = acc_w[i]
        part_ref.at[b, tb, k, cols][...] = acc_b

    inputs = [x, dn, w_t, bias] + ([state] if p.has_state else []) + ([mask] if p.has_mask else [])
    out_type = (jax.ShapeDtypeStruct(x.shape, dtype),
                jax.ShapeDtypeStruct((p.batch, p.num_blocks, k + 1, p.channels), _F32))
    kern = sc._kernel(body, "short_conv_l2norm_bwd", out_type=out_type,
                      grid=(p.channels // bc, p.batch, p.num_blocks), grid_names=("c", "b", "t"))
    return kern(*inputs)


def _norm_input_grad(x: Array, w_t: Array, bias: Array, state: Array, mask: Array | None, dn: Array,
                     activation: bool, head_dim: int) -> Array:
    """``d/d act`` of the first ``min(K - 1, T)`` rows (those that read the state), in XLA (tiny): the
    gradient that ``short_conv._state_grad`` takes as ``dy``."""
    k = w_t.shape[0]
    batch, _, channels = x.shape
    n = min(k - 1, x.shape[1])
    x_ext = jnp.concatenate([state.astype(_F32), x[:, :n].astype(_F32)], axis=1)
    pre = jnp.zeros((batch, n, channels), _F32) + bias
    for i in range(k):
        tap = x_ext[:, i: i + n] * w_t[i]
        if mask is not None and i < k - 1:
            tap = jnp.where(sc._bit(mask[:, :n], k - 1 - i)[..., None], tap, 0.0)
        pre = pre + tap
    y = jax.nn.silu(pre) if activation else pre
    y = y.reshape(batch, n, channels // head_dim, head_dim)
    inv = lax.rsqrt(jnp.sum(y * y, axis=-1, keepdims=True) + EPS)
    nrm = y * inv
    d = dn[:, :n].astype(_F32).reshape(nrm.shape)
    dy = inv * (d - nrm * jnp.sum(d * nrm, axis=-1, keepdims=True))
    return dy.reshape(batch, n, channels)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _conv(x, w_t, bias, state, mask, activation, head_dim):
    p = _plan(x, w_t.shape[0], activation, state is not None, mask is not None, head_dim, backward=False)
    return _forward_call(p, head_dim, x, w_t, bias, state, mask)


def _conv_fwd(x, w_t, bias, state, mask, activation, head_dim):
    return _conv(x, w_t, bias, state, mask, activation, head_dim), (x, w_t, bias, state, mask)


def _conv_bwd(activation, head_dim, res, dn):
    x, w_t, bias, state, mask = res
    k = w_t.shape[0]
    p = _plan(x, k, activation, state is not None, mask is not None, head_dim, backward=True)
    dx, parts = _backward_call(p, head_dim, x, dn.astype(x.dtype), w_t, bias, state, mask)
    sums = parts.sum(axis=(0, 1))
    dstate = None
    if state is not None:
        dy = _norm_input_grad(x, w_t, bias, state, mask, dn, activation, head_dim)
        dstate = sc._state_grad(x, w_t, bias, state, mask, dy, activation)
    return dx, sums[:k], sums[k], dstate, None


_conv.defvjp(_conv_fwd, _conv_bwd)


def unsupported_reason(x: Array, weight: Array, *, head_dim: int, segment_ids: Array | None = None,
                       state: Array | None = None, check_devices: bool = True) -> str | None:
    """``None`` when the fused kernels accept this call, otherwise why not (``check_devices=False``: on one
    device, i.e. without the multi-device rule, as :func:`short_conv_l2norm_sharded` checks its shards)."""
    reason = sc.unsupported_reason(x, weight, segment_ids=segment_ids, state=state, check_devices=check_devices)
    if reason is not None:
        return reason
    if jnp.dtype(x.dtype) != jnp.dtype(jnp.bfloat16):
        return f"x must be bfloat16 (the dtype the fused kernels are tested in); got {jnp.dtype(x.dtype)}"
    if head_dim not in HEAD_DIMS:
        return f"head_dim {head_dim} is not one of {HEAD_DIMS}"
    if x.shape[2] % head_dim:
        return f"channels {x.shape[2]} are not a whole number of {head_dim}-channel heads"
    lanes = _head_lanes(x, weight.shape[1], head_dim)
    if not 1 <= lanes <= _WARP:
        return (f"a {head_dim}-channel head spans more than one warp at {x.shape[2]} channels "
                f"(channels must be a multiple of {sc._LANES * max(1, head_dim // _WARP)})")
    return None


def short_conv_l2norm(x: Array, weight: Array, bias: Array | None = None, *, head_dim: int,
                      activation: bool = True, segment_ids: Array | None = None,
                      state: Array | None = None) -> Array:
    """``l2norm(act(b + depthwise causal conv(x)))`` per ``head_dim``-channel head (output only).

    Arguments as :func:`jla.ops.mosaic_gpu.short_conv.short_conv`; the result equals
    ``reference.l2norm(y.reshape(B, T, C // head_dim, head_dim)).reshape(B, T, C)`` of that function's ``y``
    (up to the rounding of ``y``, which is skipped here). Differentiable in ``x``, ``weight``, ``bias`` and
    ``state``. Raises ``ValueError`` for calls the kernels do not support.
    """
    reason = unsupported_reason(x, weight, head_dim=head_dim, segment_ids=segment_ids, state=state)
    if reason is not None:
        raise ValueError(f"Mosaic GPU short conv + L2 norm: {reason}")
    width = weight.shape[1]
    w_t = weight.astype(_F32).T
    b = jnp.zeros((x.shape[2],), _F32) if bias is None else bias.astype(_F32)
    mask = None if segment_ids is None else sc.lag_mask(segment_ids, width)
    return _conv(x, w_t, b, state, mask, bool(activation), int(head_dim))


def short_conv_l2norm_sharded(x: Array, weight: Array, bias: Array | None = None, *, head_dim: int,
                              activation: bool = True, segment_ids: Array | None = None,
                              state: Array | None = None, channel_axes: tuple[str, ...] | None = None) -> Array | None:
    """:func:`short_conv_l2norm` on the shards of the context mesh, for a call that may span several devices.

    The rule of :func:`jla.ops.mosaic_gpu.short_conv.short_conv_sharded`, with whole heads: the batch over the
    ``batch`` rule's mesh axes, the channels (``weight``, ``bias`` and ``state`` with them) over the
    ``channel_axes`` that divide the number of ``head_dim``-channel heads, every other mesh axis of size > 1 on
    the batch. The L2 norm is per token and head, so each device's rows and heads come out exactly as on one
    device. Returns ``None`` when there is no context mesh axis to split over or no split gives per-device
    shapes the kernels take (the caller then keeps the unfused path).
    """
    from jla.distributed.kernel_sharding import mesh_can_split, shard_layout_call

    reason = unsupported_reason(x, weight, head_dim=head_dim, segment_ids=segment_ids, state=state,
                                check_devices=False)
    if reason is not None:
        raise ValueError(f"Mosaic GPU short conv + L2 norm: {reason}")
    if not mesh_can_split():
        return None
    width = weight.shape[1]

    def body(x, weight, bias, segment_ids, state):
        w_t = weight.astype(_F32).T
        b = jnp.zeros((x.shape[2],), _F32) if bias is None else bias.astype(_F32)
        mask = None if segment_ids is None else sc.lag_mask(segment_ids, width)
        return _conv(x, w_t, b, state, mask, bool(activation), int(head_dim))

    def supports(x, weight, bias, segment_ids, state):
        return unsupported_reason(x, weight, head_dim=head_dim, segment_ids=segment_ids, state=state,
                                  check_devices=False) is None

    return shard_layout_call(
        body, {"x": x, "weight": weight, "bias": bias, "segment_ids": segment_ids, "state": state},
        {"x": "token", "weight": "channels", "bias": "channels", "segment_ids": "batch", "state": "token"},
        out_layouts="token", heads=x.shape[2] // head_dim if channel_axes else None,
        head_axes=channel_axes or None, supports=supports, unsplit=lambda **_: None)
