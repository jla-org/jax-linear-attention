# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright (c) 2026 Mayank Mishra.
# Adapted from FLAT (Apache-2.0), flat/depthwise_causal_convolution/pallas_implementation/{forward,backward}.py
"""Fused short depthwise causal convolution + SiLU on H100 (Mosaic GPU), forward and backward.

``y_t = act(b + sum_i w[:, i] * x_{t - (K - 1 - i)})`` over ``x [B, T, C]``, exactly what
:class:`jla.layers.conv.ShortConv` computes: tap ``K - 1`` multiplies the current token, an optional
initial state ``[B, K - 1, C]`` (oldest first) supplies the inputs before ``t = 0``, and with
``segment_ids`` a tap never reads a token of another document (the state counts as part of each
row's first document). Inputs and outputs keep the ``[B, T, C]`` layout the projections write, in
bfloat16 or float16; all arithmetic is float32.

Kernels. Each CTA is one warpgroup that owns ``128 * vec`` channels of one sequence and a block of
tokens. Every thread streams its ``vec`` contiguous channels along time, keeping the last ``K - 1``
inputs in registers, with loads and stores of up to 8 bytes straight from HBM (rows are loaded
``group`` at a time ahead of their use, and the backward prefetches the next group while it
computes the current one, so a thread has several loads in flight); each input is
read once and each output written once, apart from the ``K - 1`` halo rows of a block. Blocks are
independent: the last block of a sequence is shifted back to end at ``T`` (it recomputes a few rows
of its neighbour and writes the same values), so ragged ``T`` needs no padding.

The backward follows FLAT's decomposition: the SiLU gradient ``g = dy * silu'(pre)`` is formed once
in float32 from the recomputed pre-activation, ``dx`` is the sum of the gradient taps shifted
forward in time (``dx_s = sum_i w_i g_{s + K - 1 - i}``), and the weight and bias gradients are
the per-block sums ``sum_t g_t x_{t - (K - 1 - i)}`` and ``sum_t g_t`` (one f32 partial per block,
reduced in XLA), with the history rows closing the sums at the start of a sequence. Here every
block recomputes its ``K - 1`` future rows instead of FLAT's reverse sequential walk, so blocks run
in parallel; the gradient of the initial state (``K - 1`` rows) is formed in XLA.

Segment resets arrive as a per-token bit mask built in XLA (bit ``l`` of ``mask[b, t]`` set when the
token ``l`` steps back is in the same document), one scalar per row in the kernels.

The kernels are single-device custom calls without a partitioning rule; :func:`unsupported_reason`
rejects calls that may span several devices (see ``linear_cross_entropy.multi_device_reason``), and
:func:`short_conv_sharded` runs such calls per shard of the context mesh (batch over the data axes,
channels over the axes that shard them).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jla.ops.mosaic_gpu._common import is_hopper, num_sms
from jla.ops.mosaic_gpu.linear_cross_entropy import multi_device_reason

__all__ = ["DTYPES", "WIDTHS", "lag_mask", "short_conv", "short_conv_rows", "short_conv_sharded", "unsupported_reason"]

Array = jax.Array

DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
WIDTHS = (2, 3, 4)  # kernel sizes the GPU tests cover (FLA, Mamba-2, GDN, KDA use 4)
_LANES = 128  # threads of the warpgroup; a CTA covers 128 * vec channels
_F32 = jnp.float32


@dataclasses.dataclass(frozen=True)
class _Plan:
    batch: int
    seq_len: int
    channels: int
    width: int
    activation: bool
    has_state: bool
    has_mask: bool
    vec: int  # contiguous channels per thread
    block_t: int  # tokens per CTA
    group: int  # rows loaded ahead of their use
    prefetch: bool  # load the next group's rows before computing the current group
    x_offset: int = 0  # first channel of x in its last dim (a row of a stacked input, :func:`short_conv_rows`)

    @property
    def block_c(self) -> int:
        return _LANES * self.vec

    @property
    def num_blocks(self) -> int:
        return -(-self.seq_len // self.block_t)


def _plan(x: Array, width: int, activation: bool, has_state: bool, has_mask: bool, *, backward: bool,
          channels: int | None = None, x_offset: int = 0) -> _Plan:
    """Block plan, tuned on an H100 at the 1.3B presets' shapes (``benchmarks/short_conv.md``). ``channels`` and
    ``x_offset``: the conv runs on ``x[..., x_offset : x_offset + channels]`` (default: all of ``x``)."""
    batch, seq_len, width_x = x.shape
    channels = width_x if channels is None else channels
    vec = next(v for v in (4, 2, 1) if channels % (_LANES * v) == 0)  # 8 channels per thread was ~10% slower
    if backward:
        # Prefetching the next row group pays in the backward (2 CTAs per SM either way); larger blocks
        # write fewer weight-gradient partials, as long as the grid still fills the GPU.
        group, prefetch = (4 if vec == 4 else 8), True
        block_t = 64
        for candidate in (256, 128):
            if (channels // (_LANES * vec)) * batch * -(-seq_len // candidate) >= 3 * num_sms():
                block_t = candidate
                break
    else:
        # Without prefetching the forward fits one more CTA per SM, which hides its load latency better.
        group, prefetch, block_t = 8, False, 64
    block_t = min(block_t, seq_len)
    return _Plan(batch, seq_len, channels, width, activation, has_state, has_mask, vec, block_t, group, prefetch,
                 x_offset)


def _kernel(body, name: str, **kwargs):
    # Lane semantics with plain GMEM loads/stores; every thread touches only its own channels, so
    # the barriers Pallas inserts around stores (for shared SMEM) are not needed.
    params = plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Lane, unsafe_no_auto_barriers=True)
    return plgpu.kernel(body, kernel_name=name, compiler_params=params, **kwargs)


def _row(ref, line) -> Array:
    return plgpu.load(ref, layout=line)


def _bit(mask: Array, lag: int) -> Array:
    return ((mask >> lag) & 1) != 0


def _history(p: _Plan, x_ref, state_ref, b, idx, cols, line) -> Array:
    """``x_ext`` row of token ``idx`` (``-(K - 1) <= idx < T``) in the input dtype: the state or zero
    before 0. The window rows stay in that dtype (exact) and are widened to f32 where they are used."""
    xv = _row(x_ref.at[b, jnp.maximum(idx, 0), _x_cols(p, cols)], line)
    if state_ref is None:
        return jnp.where(idx >= 0, xv, jnp.zeros((), xv.dtype))
    sv = _row(state_ref.at[b, jnp.clip(idx + p.width - 1, 0, p.width - 2), cols], line)
    return jnp.where(idx >= 0, xv, sv)


def _x_cols(p: _Plan, cols):
    """The columns of ``x`` that hold the conv channels ``cols`` (shifted by ``p.x_offset``)."""
    return cols if not p.x_offset else pl.ds(cols.start + p.x_offset, cols.size)


def _split_refs(p: _Plan, refs, n_in: int):
    it = iter(refs[:n_in])
    head = [next(it) for _ in range(n_in - int(p.has_state) - int(p.has_mask))]
    state_ref = next(it) if p.has_state else None
    mask_ref = next(it) if p.has_mask else None
    return head, state_ref, mask_ref, refs[n_in:]


def _forward_call(p: _Plan, x: Array, w_t: Array, bias: Array, state: Array | None, mask: Array | None) -> Array:
    k, bt, u, bc = p.width, p.block_t, p.group, p.block_c
    t_len = p.seq_len
    line = plgpu.Layout.WG_STRIDED((bc,), vec_size=p.vec)
    dtype = x.dtype
    groups = -(-bt // u)
    guard = groups * u != bt  # the last group runs past the block

    def body(*refs):
        (x_ref, w_ref, b_ref), state_ref, mask_ref, (y_ref,) = _split_refs(p, refs, len(inputs))
        cb, b, tb = lax.axis_index("c"), lax.axis_index("b"), lax.axis_index("t")
        cols = pl.ds(cb * bc, bc)
        t0 = jnp.minimum(tb * bt, t_len - bt)
        w = [_row(w_ref.at[i, cols], line) for i in range(k)]
        bias_v = _row(b_ref.at[cols], line)
        prev = tuple(_history(p, x_ref, state_ref, b, t0 - lag, cols, line) for lag in range(1, k))

        def load_group(g):  # the group's loads are issued before any of its arithmetic
            rows = []
            for r in range(u):
                tc = jnp.minimum(t0 + g * u + r, t_len - 1)
                m = mask_ref[b, tc] if mask_ref is not None else None
                rows.append((_row(x_ref.at[b, tc, _x_cols(p, cols)], line), m))
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
                        tap = jnp.where(_bit(m, lag), tap, 0.0)
                    acc = acc + tap
                if p.activation:
                    acc = acc * jax.nn.sigmoid(acc)
                out = acc.astype(dtype)
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
    kern = _kernel(
        body, "short_conv_fwd", out_type=jax.ShapeDtypeStruct((p.batch, p.seq_len, p.channels), dtype),
        grid=(p.channels // bc, p.batch, p.num_blocks), grid_names=("c", "b", "t"),
    )
    return kern(*inputs)


def _backward_call(p: _Plan, x: Array, dy: Array, w_t: Array, bias: Array, state: Array | None,
                   mask: Array | None) -> tuple[Array, Array]:
    """``(dx [B, T, C], partials [B, NB, K + 1, C] f32)``: per-block ``dw`` rows then ``db``."""
    k, bt, u, bc = p.width, p.block_t, p.group, p.block_c
    t_len = p.seq_len
    line = plgpu.Layout.WG_STRIDED((bc,), vec_size=p.vec)
    dtype = x.dtype
    steps = bt + k - 1  # dpre rows t0 .. t0 + bt + k - 2 (the last k - 1 feed dx of the block's tail)
    groups = -(-steps // u)

    def body(*refs):
        (x_ref, dy_ref, w_ref, b_ref), state_ref, mask_ref, (dx_ref, part_ref) = _split_refs(p, refs, len(inputs))
        cb, b, tb = lax.axis_index("c"), lax.axis_index("b"), lax.axis_index("t")
        cols = pl.ds(cb * bc, bc)
        t0 = jnp.minimum(tb * bt, t_len - bt)
        own = tb * bt  # rows [own, end) count toward dw and db (the shifted last block overlaps)
        end = t0 + bt
        w = [_row(w_ref.at[i, cols], line) for i in range(k)]
        bias_v = _row(b_ref.at[cols], line)
        zero = bias_v * 0.0
        xprev = tuple(_history(p, x_ref, state_ref, b, t0 - lag, cols, line) for lag in range(1, k))
        dprev = tuple(zero for _ in range(1, k))  # rows before t0 only feed dx rows before t0
        # (their masks are never used either; loaded so every carry has its in-loop type)
        mprev = tuple(mask_ref[b, jnp.maximum(t0 - lag, 0)] for lag in range(1, k)) if p.has_mask else ()
        acc_w = tuple(zero for _ in range(k))

        def load_group(g):  # the group's loads are issued before any of its arithmetic
            rows = []
            for r in range(u):
                tc = jnp.minimum(t0 + g * u + r, t_len - 1)
                m = mask_ref[b, tc] if mask_ref is not None else None
                rows.append((_row(x_ref.at[b, tc, _x_cols(p, cols)], line), _row(dy_ref.at[b, tc, cols], line), m))
            return tuple(rows)

        def group_step(g, carry):
            xprev, dprev, mprev, acc_w, acc_b, rows = carry
            if p.prefetch:  # in flight while this group computes (the last group loads itself again)
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
                        xv = jnp.where(_bit(m, lag), xv, 0.0)
                    taps.append(xv)
                    pre = pre + w[i] * xv
                grad = dyt.astype(_F32)
                if p.activation:
                    sig = jax.nn.sigmoid(pre)
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
                        term = jnp.where(_bit(mwin[i], lag), term, 0.0)
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

    inputs = [x, dy, w_t, bias] + ([state] if p.has_state else []) + ([mask] if p.has_mask else [])
    out_type = (jax.ShapeDtypeStruct((p.batch, p.seq_len, p.channels), dtype),
                jax.ShapeDtypeStruct((p.batch, p.num_blocks, k + 1, p.channels), _F32))
    kern = _kernel(body, "short_conv_bwd", out_type=out_type,
                   grid=(p.channels // bc, p.batch, p.num_blocks), grid_names=("c", "b", "t"))
    return kern(*inputs)


def lag_mask(segment_ids: Array, width: int) -> Array:
    """``[B, T]`` int32: bit ``l`` (``0 <= l < width``) set when the token ``l`` steps back shares
    the document of token ``t``; tokens before 0 (the conv state) count as the first document."""
    batch, seq_len = segment_ids.shape
    seg = segment_ids.astype(jnp.int32)
    ext = jnp.concatenate([jnp.broadcast_to(seg[:, :1], (batch, width - 1)), seg], axis=1)
    bits = jnp.ones((batch, seq_len), jnp.int32)
    for lag in range(1, width):
        same = ext[:, width - 1 - lag: width - 1 - lag + seq_len] == seg
        bits = bits | (same.astype(jnp.int32) << lag)
    return bits


def _state_grad(x: Array, w_t: Array, bias: Array, state: Array, mask: Array | None, dy: Array,
                activation: bool) -> Array:
    """Gradient of the initial state: only outputs ``t < K - 1`` read it (tiny, in XLA)."""
    k = w_t.shape[0]
    n = min(k - 1, x.shape[1])
    x_ext = jnp.concatenate([state.astype(_F32), x[:, :n].astype(_F32)], axis=1)  # rows -(K-1) .. n-1
    keep = [None if mask is None or i == k - 1 else _bit(mask[:, :n], k - 1 - i)[..., None] for i in range(k)]
    pre = jnp.zeros((x.shape[0], n, x.shape[2]), _F32) + bias
    for i in range(k):
        tap = x_ext[:, i: i + n] * w_t[i]
        pre = pre + (tap if keep[i] is None else jnp.where(keep[i], tap, 0.0))
    grad = dy[:, :n].astype(_F32)
    if activation:
        sig = jax.nn.sigmoid(pre)
        grad = grad * sig * (1.0 + pre * (1.0 - sig))
    dstate = jnp.zeros(state.shape, _F32)
    for i in range(k):  # output row t reads x_ext row t + i: state row t + i when that is < K - 1
        for t in range(n):
            r = t + i
            if r < k - 1:
                term = grad[:, t] * w_t[i]
                if keep[i] is not None:
                    term = jnp.where(keep[i][:, t], term, 0.0)
                dstate = dstate.at[:, r].add(term)
    return dstate.astype(state.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _conv(x, w_t, bias, state, mask, activation):
    p = _plan(x, w_t.shape[0], activation, state is not None, mask is not None, backward=False)
    return _forward_call(p, x, w_t, bias, state, mask)


def _conv_fwd(x, w_t, bias, state, mask, activation):
    return _conv(x, w_t, bias, state, mask, activation), (x, w_t, bias, state, mask)


def _conv_bwd(activation, res, dy):
    x, w_t, bias, state, mask = res
    k = w_t.shape[0]
    p = _plan(x, k, activation, state is not None, mask is not None, backward=True)
    dx, parts = _backward_call(p, x, dy.astype(x.dtype), w_t, bias, state, mask)
    sums = parts.sum(axis=(0, 1))
    dstate = None if state is None else _state_grad(x, w_t, bias, state, mask, dy, activation)
    return dx, sums[:k], sums[k], dstate, None


_conv.defvjp(_conv_fwd, _conv_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _conv_rows(xs, w_t, bias, mask, activation, row):
    batch, seq_len, n, channels = xs.shape
    p = _plan(xs.reshape(batch, seq_len, n * channels), w_t.shape[0], activation, False, mask is not None,
              backward=False, channels=channels, x_offset=row * channels)
    return _forward_call(p, xs.reshape(batch, seq_len, n * channels), w_t, bias, None, mask)


def _conv_rows_fwd(xs, w_t, bias, mask, activation, row):
    return _conv_rows(xs, w_t, bias, mask, activation, row), (xs, w_t, bias, mask)


def _conv_rows_bwd(activation, row, res, dy):
    xs, w_t, bias, mask = res
    batch, seq_len, n, channels = xs.shape
    k = w_t.shape[0]
    flat = xs.reshape(batch, seq_len, n * channels)
    p = _plan(flat, k, activation, False, mask is not None, backward=True, channels=channels, x_offset=row * channels)
    dx, parts = _backward_call(p, flat, dy.astype(xs.dtype), w_t, bias, None, mask)
    sums = parts.sum(axis=(0, 1))
    # The other rows get zeros: XLA adds them to those rows' own gradients in one fusion (no copy of dx).
    dxs = jnp.stack([dx if i == row else jnp.zeros_like(dx) for i in range(n)], axis=2)
    return dxs, sums[:k], sums[k], None


_conv_rows.defvjp(_conv_rows_fwd, _conv_rows_bwd)


def short_conv_rows(xs: Array, weight: Array, bias: Array | None = None, *, row: int, activation: bool = True,
                    segment_ids: Array | None = None) -> Array | None:
    """:func:`short_conv` of ``xs[:, :, row]`` read in place from a stacked ``xs [B, T, n, C]`` (the rows of one
    stacked projection GEMM, e.g. Mamba-2's z|x rows from an FP8 projection), without copying the row out.

    One device, no initial state (training calls). The gradient of ``xs`` is ``dx`` in ``row`` and zeros in the
    other rows, which XLA folds into the other rows' gradients. Returns ``None`` when the kernels do not take the
    call (the caller then slices the row and runs :class:`jla.layers.conv.ShortConv`).
    """
    if xs.ndim != 4 or not 0 <= row < xs.shape[2]:
        raise ValueError(f"expected xs [B, T, n, C] and 0 <= row < n; got {xs.shape} and row {row}")
    batch, seq_len, n, channels = xs.shape
    view = jax.ShapeDtypeStruct((batch, seq_len, channels), xs.dtype)
    if unsupported_reason(view, weight, segment_ids=segment_ids) is not None:
        return None
    width = weight.shape[1]
    w_t = weight.astype(_F32).T
    b = jnp.zeros((channels,), _F32) if bias is None else bias.astype(_F32)
    mask = None if segment_ids is None else lag_mask(segment_ids, width)
    return _conv_rows(xs, w_t, b, mask, bool(activation), int(row))


def unsupported_reason(x: Array, weight: Array, *, segment_ids: Array | None = None,
                       state: Array | None = None, check_devices: bool = True) -> str | None:
    """``None`` when the kernels accept this call, otherwise why not (``check_devices=False``: on one device,
    i.e. without the multi-device rule; :func:`short_conv_sharded` checks the per-device shards so)."""
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[2]:
        return f"expected x [B, T, C] and weight [C, K]; got {x.shape} and {weight.shape}"
    if jnp.dtype(x.dtype) not in DTYPES:
        return f"x must be bfloat16 or float16; got {jnp.dtype(x.dtype)}"
    if weight.shape[1] not in WIDTHS:
        return f"kernel size {weight.shape[1]} is not one of {WIDTHS}"
    if x.shape[2] % _LANES:
        return f"channels {x.shape[2]} must be a multiple of {_LANES}"
    if x.shape[1] < 1 or x.shape[0] < 1:
        return "empty input"
    if state is not None and (state.shape != (x.shape[0], weight.shape[1] - 1, x.shape[2]) or state.dtype != x.dtype):
        return f"state must be {(x.shape[0], weight.shape[1] - 1, x.shape[2])} {x.dtype}; got {state.shape} {state.dtype}"
    if segment_ids is not None and segment_ids.shape != x.shape[:2]:
        return f"segment_ids must be {x.shape[:2]}; got {segment_ids.shape}"
    reason = multi_device_reason() if check_devices else None
    if reason is not None:
        return reason
    if not is_hopper():
        return "the Mosaic GPU short conv kernels need an sm_90 (Hopper) GPU"
    return None


def short_conv(x: Array, weight: Array, bias: Array | None = None, *, activation: bool = True,
               segment_ids: Array | None = None, state: Array | None = None) -> Array:
    """``act(b + depthwise causal conv(x))`` as :class:`jla.layers.conv.ShortConv` (output only).

    ``x [B, T, C]`` bf16/f16, ``weight [C, K]`` and ``bias [C]`` (any float dtype, used in f32),
    ``segment_ids [B, T]`` int, ``state [B, K - 1, C]`` in ``x.dtype``. Differentiable in ``x``,
    ``weight``, ``bias`` and ``state``. Raises ``ValueError`` for calls the kernels do not support.
    """
    reason = unsupported_reason(x, weight, segment_ids=segment_ids, state=state)
    if reason is not None:
        raise ValueError(f"Mosaic GPU short conv: {reason}")
    width = weight.shape[1]
    w_t = weight.astype(_F32).T
    b = jnp.zeros((x.shape[2],), _F32) if bias is None else bias.astype(_F32)
    mask = None if segment_ids is None else lag_mask(segment_ids, width)
    return _conv(x, w_t, b, state, mask, bool(activation))


# -- several devices: the kernels per shard of the context mesh -----------------------------------------------


def _local_conv(x: Array, weight: Array, bias: Array | None, segment_ids: Array | None, state: Array | None, *,
                activation: bool) -> Array:
    """:func:`short_conv` without its checks: one device's shard, inside ``shard_map``."""
    width = weight.shape[1]
    w_t = weight.astype(_F32).T
    b = jnp.zeros((x.shape[2],), _F32) if bias is None else bias.astype(_F32)
    mask = None if segment_ids is None else lag_mask(segment_ids, width)
    return _conv(x, w_t, b, state, mask, bool(activation))


def short_conv_sharded(x: Array, weight: Array, bias: Array | None = None, *, activation: bool = True,
                       segment_ids: Array | None = None, state: Array | None = None,
                       channel_axes: tuple[str, ...] | None = None) -> Array | None:
    """:func:`short_conv` on the shards of the context mesh, for a call that may span several devices.

    The partitioning rule (:func:`jla.distributed.kernel_sharding.shard_layout_call`): ``x [B, T, C]``,
    ``state [B, K - 1, C]`` and ``segment_ids [B, T]`` split along the batch over the ``batch`` rule's mesh axes,
    and the channels of ``x``, ``state``, ``weight [C, K]`` and ``bias [C]`` over ``channel_axes`` (the mesh
    axes that shard the conv's channels, e.g. the ``heads`` rule's; empty or ``None`` keeps them whole); every
    other mesh axis of size > 1 splits the batch too. The conv works per row and per channel, so each device
    runs the kernels on its rows and channels exactly as one device runs them on all, and ``shard_map``'s
    transpose sums the weight and bias gradients over the batch axes. Returns ``None`` when there is no
    context mesh axis to split over, or no split gives per-device shapes the kernels take (the caller then
    runs XLA). Raises ``ValueError`` for operands the kernels never take.
    """
    from jla.distributed.kernel_sharding import mesh_can_split, shard_layout_call

    reason = unsupported_reason(x, weight, segment_ids=segment_ids, state=state, check_devices=False)
    if reason is not None:
        raise ValueError(f"Mosaic GPU short conv: {reason}")
    if not mesh_can_split():
        return None

    def body(x, weight, bias, segment_ids, state):
        return _local_conv(x, weight, bias, segment_ids, state, activation=activation)

    def supports(x, weight, bias, segment_ids, state):
        return unsupported_reason(x, weight, segment_ids=segment_ids, state=state, check_devices=False) is None

    return shard_layout_call(
        body, {"x": x, "weight": weight, "bias": bias, "segment_ids": segment_ids, "state": state},
        {"x": "token", "weight": "channels", "bias": "channels", "segment_ids": "batch", "state": "token"},
        out_layouts="token", heads=x.shape[2] if channel_axes else None, head_axes=channel_axes or None,
        supports=supports, unsplit=lambda **_: None)
