# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 q/k/v projections, RoPE and Mosaic GPU attention in one VJP (``AttentionConfig.fp8_grad_epilogue``).

In a delayed-scaling FP8 train step the attention block's q, k and v projections are three FP8 matmuls
that share one e4m3 copy of their input (:func:`jla.layers.fp8.shared_input_projections`). Their backward
passes need the e5m2 gradients of the projection outputs, the amaxes the recipe records for them, and the
transposed e5m2 copies for the weight-gradient GEMMs; with the plain layer XLA builds these from the
attention backward's ``dq, dk, dv`` in separate passes (the RoPE backward, the casts and amaxes, and the
transposes of the merged weight-gradient GEMM's operand; ``benchmarks/fp8_1p3b.md``).

:func:`fp8_qkv_attention` computes the same forward as the plain layer (the three FP8 projections, RoPE,
the Mosaic GPU attention forward) under one custom VJP whose backward runs the attention backward
kernels with FP8 epilogues (:func:`jla.ops.mosaic_gpu.attention_fp8_bwd.fp8_attention_backward`, the
ranged or the fused backward as ``mosaic_gpu_backward`` says): they write the e5m2 gradients and the
row amaxes, which the projections' GEMMs read directly; with the fused backward and no multi-device mesh
axis they also write the transposed operand of one merged weight-gradient GEMM, which the layer issues (else XLA
builds that operand from the token-major gradients, as for the plain layer). The scales, the new delayed-scaling
states and the GEMMs are those of :func:`jla.layers.fp8.fp8_dot_prepared` with a shared input; ``impl="xla"`` runs the
same VJP with XLA attention and the plain path's RoPE backward and quantization (the CPU reference:
bitwise the plain layer's values, gradients and states, ``tests/test_fp8_attention.py``).

The layer takes this path when ``AttentionConfig.fp8_grad_epilogue`` is set and the call qualifies
(:func:`fused_fp8_attention`; otherwise the plain path runs, logged once per reason when the projections
are FP8): delayed-scaling FP8 q/k/v projections with the train step's weight copies and no biases, no
QK-norm, plain full RoPE, no sinks, MoBA or context parallelism, the Mosaic GPU kernel chosen for the call,
head dim 64 or 128 (64 with ``mosaic_gpu_backward = "fused"``) and ``T`` a multiple of 128. Under a context
mesh the attention kernels run per batch and head shard (:func:`jla.distributed.kernel_sharding.shard_layout_call`);
the GEMMs stay with XLA's partitioner.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from jla.layers.fp8 import (_BWD_PRECISION, E4M3, E5M2, Fp8State, _amax, _dgrad_dims, _fp8_max, _scaled_dot,
                            _shared_forward, _shares_input, _wgrad_dims, push_amax, quantize, scale_from_amax)
from jla.layers.rope import apply_rope

__all__ = ["fp8_qkv_attention", "fused_fp8_attention", "fused_fp8_attention_reason", "rope_tables"]

Array = jax.Array
_log = logging.getLogger(__name__)
_LOGGED: set[str] = set()
IMPLS = ("mosaic_gpu", "xla")


@dataclasses.dataclass(frozen=True)
class _Static:
    heads: int
    kv_heads: int
    head_dim: int
    theta: float
    softmax_scale: float | None  # AttentionConfig.softmax_scale (None: head_dim ** -0.5)
    window: int | None  # AttentionConfig.sliding_window
    has_segments: bool
    impl: str
    dtypes: tuple[str, str]  # the projections' (compute dtype, weight dtype)
    backward: str = "ranged"  # the kernels' backward (AttentionConfig.mosaic_gpu_backward)
    persistent_forward: bool | None = None  # the kernel's forward (AttentionConfig.persistent_forward)


def rope_tables(positions: Array, *, theta: float, head_dim: int) -> tuple[Array, Array]:
    """``cos, sin [B, T, head_dim / 2]`` float32: the angles of :func:`jla.layers.rope.apply_rope` (plain RoPE)."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = positions.astype(jnp.float32)[:, :, None] * inv_freq
    return jnp.cos(angles), jnp.sin(angles)


def _kernel_config(static: _Static, length: int):
    from jla.ops.mosaic_gpu.attention import _Config, persistent_option

    scale = static.softmax_scale
    window = static.window
    return _Config(
        scale=float(1.0 / math.sqrt(static.head_dim) if scale is None else scale),
        window=None if window is None or window >= length else int(window),
        has_segments=static.has_segments,
        backward=static.backward,
        **persistent_option(static.persistent_forward),
    )


def _segment_runs(seg: Array, has_segments: bool) -> Array:
    """``runs [B, 4, T]`` as ``jla.ops.mosaic_gpu.attention.flash_attention`` builds them."""
    from jla.ops.mosaic_gpu.attention import _contiguous_ids, _inner_runs, _segment_runs as runs_of

    if not has_segments:
        return jnp.zeros((seg.shape[0], 4, seg.shape[1]), jnp.int32)  # not read by the kernels
    contiguous = _contiguous_ids(seg)
    runs = runs_of(seg, contiguous)
    return jnp.concatenate([runs, _inner_runs(runs, contiguous, True)], axis=1)


def _kernel_forward(static: _Static, q, k, v, seg, runs):
    """The Mosaic GPU attention forward (``out, lse``), per batch and head shard of the context mesh."""
    from jla.distributed.kernel_sharding import shard_layout_call
    from jla.ops.mosaic_gpu.attention import _forward_sharded, _sink_init, _varying_like, supported

    def body(q, k, v, seg, runs):
        cfg = _kernel_config(static, q.shape[1])
        sinks = jnp.zeros((q.shape[2],), jnp.float32)
        out, lse = _forward_sharded(q, k, v, seg, runs, _sink_init(sinks, cfg.block_q), cfg)
        return tuple(_varying_like(x, q, k, v, seg) for x in (out, lse))

    return shard_layout_call(
        body, {"q": q, "k": k, "v": v, "seg": seg, "runs": runs},
        {"q": "token", "k": "token", "v": "token", "seg": "batch", "runs": "batch"},
        out_layouts=("token", "state"), heads=math.gcd(q.shape[2], k.shape[2]),
        supports=lambda q, k, v, seg, runs: supported(q, k, v) is None,
    )


def _merged_wgrad(static: _Static) -> bool:
    """Whether the kernels write the merged transposed operand of one q/k/v weight-gradient GEMM
    (``fp8_attention_backward(..., merged=True)``): the fused backward, with no mesh axis that could split the
    heads (the operand holds every query head before the key and value heads)."""
    if static.impl != "mosaic_gpu" or static.backward != "fused" or static.head_dim != 64:
        return False
    mesh = jax.sharding.get_abstract_mesh()
    return mesh.empty or all(int(size) == 1 for size in mesh.axis_sizes)


def _kernel_backward(static: _Static, q, k, v, seg, runs, out, lse, do, cos, sin, gscale, merged: bool = False):
    """:func:`fp8_attention_backward` per batch and head shard of the context mesh (with ``merged``, also the
    merged transposed operand ``[H + 2 Hkv, D, B, T]``, returned after the three ``(e5m2, row maxima)``)."""
    from jla.distributed.kernel_sharding import shard_layout_call
    from jla.ops.mosaic_gpu.attention import _varying_like
    from jla.ops.mosaic_gpu.attention_fp8_bwd import fp8_attention_backward, fp8_backward_supported

    def body(q, k, v, seg, runs, out, lse, do, cos, sin, gscale):
        cfg = _kernel_config(static, q.shape[1])
        extra = {"merged": True} if merged else {}
        result = fp8_attention_backward(q, k, v, seg, runs, out, lse, do, cos, sin, gscale, cfg, **extra)
        grads, gt = result if merged else (result, None)
        inputs = (q, k, v, seg, out, lse, do)
        # Flat (one layout per output): e5m2 [B, T, Hp, D] and row maxima [B, Hp, T] of q, k and v.
        flat = [x for g8, _, am in grads for x in (g8, am)] + ([gt] if merged else [])
        return tuple(_varying_like(x, *inputs) for x in flat)

    flat = shard_layout_call(
        body,
        {"q": q, "k": k, "v": v, "seg": seg, "runs": runs, "out": out, "lse": lse, "do": do, "cos": cos,
         "sin": sin, "gscale": gscale},
        {"q": "token", "k": "token", "v": "token", "seg": "batch", "runs": "batch", "out": "token",
         "lse": "state", "do": "token", "cos": "batch", "sin": "batch", "gscale": "replicated"},
        out_layouts=("token", "state") * 3 + (((2, None),) if merged else ()),  # gt: batch dim 2, heads whole
        heads=math.gcd(q.shape[2], k.shape[2]),
        supports=lambda q, k, v, **_: fp8_backward_supported(q, k, v) is None,
    )
    grads = [(flat[i], flat[i + 1]) for i in (0, 2, 4)]
    return (grads, flat[6]) if merged else grads


def _xla_attention(static: _Static, q, k, v, seg):
    """The plain layer's XLA attention (``jla.layers.attention.attention(..., implementation="xla")``)."""
    mask = None
    if static.has_segments:
        mask = (seg[:, :, None] == seg[:, None, :])[:, None]
    window = None if static.window is None else (static.window - 1, 0)
    return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=static.softmax_scale, is_causal=True,
                                        local_window_size=window, implementation="xla")


def _reference_backward(static: _Static, q, k, v, seg, positions, do, gscales):
    """The plain path's projection-output gradients: XLA attention backward, RoPE backward, e5m2 cast, amax."""
    _, attention_vjp = jax.vjp(functools.partial(_xla_attention, static, seg=seg), q, k, v)
    dq, dk, dv = attention_vjp(do)

    def rope_vjp(x, ct):  # apply_rope is linear in x: its transpose does not depend on x's value
        return jax.vjp(lambda t: apply_rope(t, positions, theta=static.theta), x)[1](ct)[0]

    return [(quantize(g, gs, E5M2), _amax(g))
            for g, gs in zip((rope_vjp(q, dq), rope_vjp(k, dk), dv), gscales, strict=True)]


@functools.partial(jax.custom_vjp, nondiff_argnums=(10,))
def _core(x, weights, prepared, states, qx, x_scale, x_amax, positions, seg, runs, static):
    return _core_fwd(x, weights, prepared, states, qx, x_scale, x_amax, positions, seg, runs, static)[0]


def _core_fwd(x, weights, prepared, states, qx, x_scale, x_amax, positions, seg, runs, static):
    batch, length = x.shape[:2]
    ys, w_scales = [], []
    for wq, state in zip(prepared, states, strict=True):  # _fp8_dot_shared's forward, per projection
        y, w_scale = _shared_forward(qx, x_scale, wq, state, static.dtypes)
        ys.append(y)
        w_scales.append(w_scale)
    q = ys[0].reshape(batch, length, static.heads, static.head_dim)
    k = ys[1].reshape(batch, length, static.kv_heads, static.head_dim)
    v = ys[2].reshape(batch, length, static.kv_heads, static.head_dim)
    q, k = apply_rope(q, positions, theta=static.theta), apply_rope(k, positions, theta=static.theta)
    if static.impl == "mosaic_gpu":
        out, lse = _kernel_forward(static, q, k, v, seg, runs)
    else:
        out, lse = _xla_attention(static, q, k, v, seg), None
    residuals = (qx, x_scale, x_amax, prepared, states, tuple(w_scales), q, k, v, out, lse, positions, seg, runs)
    return out, residuals


def _core_bwd(static, residuals, do):
    qx, x_scale, x_amax, prepared, states, w_scales, q, k, v, out, lse, positions, seg, runs = residuals
    batch, length = q.shape[:2]
    e4m3, e5m2 = _fp8_max(E4M3), _fp8_max(E5M2)
    gscales = tuple(scale_from_amax(jnp.max(s.grad_amax_history), e5m2, s.grad_scale) for s in states)
    do = do.astype(q.dtype)
    if static.impl == "mosaic_gpu":
        cos, sin = rope_tables(positions, theta=static.theta, head_dim=static.head_dim)
        merged = _merged_wgrad(static)
        result = _kernel_backward(static, q, k, v, seg, runs, out, lse, do, cos, sin, jnp.stack(gscales), merged)
        grads, gt = result if merged else (result, None)
        grads = [(g8, jnp.max(am)) for g8, am in grads]
    else:
        grads, gt = _reference_backward(static, q, k, v, seg, positions, do, gscales), None
    x_dtype, w_dtype = static.dtypes
    dxs, dws, new_states = [], [], []
    for (g8, amax), gs, wq, w_scale, state in zip(grads, gscales, prepared, w_scales, states, strict=True):
        # _fp8_dot_shared_bwd's GEMMs on the e5m2 gradient (without the kernels' merged operand XLA makes the
        # weight-gradient GEMM's transposed operand, as for the plain layer: it merges the three GEMMs and
        # concatenates their operands).
        g8 = g8.reshape(batch, length, -1)
        dxs.append(_scaled_dot(g8, gs, wq.q, w_scale, _dgrad_dims(2, 1), x_dtype, _BWD_PRECISION))
        if gt is None:
            dws.append(_scaled_dot(qx, x_scale, g8, gs, _wgrad_dims(2), w_dtype, _BWD_PRECISION))
        new_states.append(Fp8State(  # _fp8_dot_shared_bwd's, with the epilogue's amax of the gradient
            input_scale=scale_from_amax(jnp.max(state.input_amax_history), e4m3, state.input_scale),
            input_amax_history=push_amax(state.input_amax_history, x_amax),
            kernel_scale=w_scale,
            kernel_amax_history=push_amax(state.kernel_amax_history, wq.amax),
            grad_scale=gs,
            grad_amax_history=push_amax(state.grad_amax_history, amax),
        ))
    if gt is not None:
        # The three weight-gradient GEMMs as one FP8 GEMM on the kernels' merged transposed operand (XLA merges the
        # three into this GEMM anyway, but builds its operand from the token-major gradients in a separate pass),
        # each projection's columns then scaled as _scaled_dot scales them.
        gt = gt.reshape(-1, batch, length)
        both = lax.dot_general(qx, gt, (((0, 1), (1, 2)), ((), ())), precision=_BWD_PRECISION,
                               preferred_element_type=w_dtype)
        widths = [static.heads * static.head_dim] + [static.kv_heads * static.head_dim] * 2
        starts = [0, widths[0], widths[0] + widths[1]]
        dws = [both[:, s:s + n] * (x_scale * gs).astype(w_dtype)
               for s, n, gs in zip(starts, widths, gscales, strict=True)]
    # Autodiff of the plain layer adds the input gradients of v, k and q in this order.
    dx = (dxs[2] + dxs[1]) + dxs[0]
    zeros = functools.partial(jax.tree.map, jnp.zeros_like)
    return (dx, tuple(dws), zeros(prepared), tuple(new_states), zeros(qx), zeros(x_scale), zeros(x_amax),
            None, None, None)


_core.defvjp(_core_fwd, _core_bwd)


def fp8_qkv_attention(x: Array, q_proj: Any, k_proj: Any, v_proj: Any, positions: Array, *, num_heads: int,
                      num_kv_heads: int, head_dim: int, rope_theta: float, softmax_scale: float | None = None,
                      sliding_window: int | None = None, segment_ids: Array | None = None,
                      impl: str = "mosaic_gpu", backward: str = "ranged",
                      persistent_forward: bool | None = None) -> Array:
    """``attention(rope(x Wq), rope(x Wk), x Wv)`` ``[B, T, H, D]`` of the plain layer, with fused FP8 gradients.

    ``q_proj, k_proj, v_proj`` are :class:`jla.layers.fp8.Fp8Linear` projections without biases that carry
    the train step's weight copies (delayed scaling). ``impl``: ``"mosaic_gpu"`` (the kernels) or ``"xla"``
    (XLA attention, the CPU reference). ``backward`` (kernels): ``"ranged"`` (``dq`` per query block, no
    pass after the kernels) or ``"fused"`` (one kernel for ``dq, dk, dv`` with ``dq`` added up in float32,
    then one conversion pass; ``D = 64``), as ``AttentionConfig.mosaic_gpu_backward``; ``persistent_forward`` as
    ``AttentionConfig.persistent_forward``.
    """
    projections = (q_proj, k_proj, v_proj)
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}; got {impl!r}")
    if backward not in ("ranged", "fused"):
        raise ValueError(f"backward must be 'ranged' or 'fused'; got {backward!r}")
    if not all(_shares_input(p) and p.bias is None for p in projections):
        raise ValueError("fp8_qkv_attention needs delayed-scaling Fp8Linear q/k/v projections with weight copies "
                         "and no biases")
    dtypes = {(p.compute_dtype, str(p.weight.dtype)) for p in projections}
    if len(dtypes) != 1:
        raise ValueError(f"the q/k/v projections must share compute and weight dtypes; got {sorted(dtypes)}")
    compute, w_dtype = dtypes.pop()
    xc = x.astype(jnp.dtype(compute))
    batch, length = xc.shape[:2]
    e4m3 = _fp8_max(E4M3)
    # shared_input_projections' one quantization of x for the three projections.
    scales = [scale_from_amax(jnp.max(p.state.input_amax_history), e4m3, p.state.input_scale) for p in projections]
    x_scale = lax.stop_gradient(functools.reduce(jnp.maximum, scales))
    qx = lax.stop_gradient(quantize(xc, x_scale, E4M3))
    x_amax = lax.stop_gradient(_amax(xc))
    static = _Static(num_heads, num_kv_heads, head_dim, float(rope_theta), softmax_scale, sliding_window,
                     segment_ids is not None, impl, (str(xc.dtype), w_dtype), backward, persistent_forward)
    seg = (jnp.zeros((batch, length), jnp.int32) if segment_ids is None else segment_ids.astype(jnp.int32))
    runs = _segment_runs(seg, segment_ids is not None) if impl == "mosaic_gpu" else jnp.zeros((batch, 4, length),
                                                                                              jnp.int32)
    return _core(xc, tuple(p.weight for p in projections), tuple(p.prepared for p in projections),
                 tuple(p.state for p in projections), qx, x_scale, x_amax, positions, seg, runs, static)


def fused_fp8_attention_reason(attention: Any, x: Array, segment_ids: Array | None) -> str | None:
    """``None`` when :func:`fused_fp8_attention` takes this ``Attention`` layer call, else the reason it does not."""
    from jla.layers.attention import resolve_attention_implementation

    cfg = attention.cfg
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    if not all(_shares_input(p) for p in projections):
        return "the q/k/v projections are not delayed-scaling FP8 projections with the train step's weight copies"
    if any(p.bias is not None for p in projections):
        return "the q/k/v projections have biases"
    if len({p.compute_dtype for p in projections}) != 1 or projections[0].compute_dtype != "bfloat16":
        return "the q/k/v projections must compute in bfloat16"
    if attention.q_norm is not None:
        return "QK-norm sits between the projections and RoPE"
    if not cfg.use_rope or cfg.rotary_fraction != 1.0 or cfg.rope_scaling is not None or cfg.rope_exact_angles:
        return "only plain full RoPE is fused"
    if cfg.sinks or cfg.moba_block_size is not None or cfg.context_parallel_axis is not None:
        return "sinks, MoBA and context parallelism are not fused"
    if cfg.head_dim not in (64, 128):
        return f"head dim {cfg.head_dim} (64 or 128 are fused)"
    if cfg.mosaic_gpu_backward == "fused" and cfg.head_dim != 64:
        return f"mosaic_gpu_backward='fused' with head dim {cfg.head_dim}: its FP8 epilogues need D = 64"
    batch, length = x.shape[:2]
    if length % 128:
        return f"T = {length} is not a multiple of 128"
    dtype = jnp.bfloat16
    q = jax.ShapeDtypeStruct((batch, length, cfg.num_heads, cfg.head_dim), dtype)
    kv = jax.ShapeDtypeStruct((batch, length, cfg.num_kv_heads, cfg.head_dim), dtype)
    impl = resolve_attention_implementation(cfg.implementation, q, kv, v=kv, segment_ids=segment_ids)
    if impl != "mosaic_gpu":
        return f"the call runs the {impl!r} attention, not the Mosaic GPU kernel"
    return None


def fused_fp8_attention(attention: Any, x: Array, positions: Array, segment_ids: Array | None) -> Array | None:
    """The :class:`jla.layers.attention.Attention` hook: :func:`fp8_qkv_attention` for qualifying calls with
    ``AttentionConfig.fp8_grad_epilogue`` set, else ``None`` (the plain path runs; logged once per reason)."""
    cfg = attention.cfg
    if not cfg.fp8_grad_epilogue:
        return None
    if not all(_shares_input(p) for p in (attention.q_proj, attention.k_proj, attention.v_proj)):
        return None  # FP8 off, current scaling, or no weight copies (evaluation, inference): nothing to fuse
    reason = fused_fp8_attention_reason(attention, x, segment_ids)
    if reason is not None:
        if reason not in _LOGGED:
            _LOGGED.add(reason)
            _log.warning("fp8_grad_epilogue: the plain attention path runs (%s)", reason)
        return None
    return fp8_qkv_attention(x, attention.q_proj, attention.k_proj, attention.v_proj, positions,
                             num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
                             rope_theta=cfg.rope_theta, softmax_scale=cfg.softmax_scale,
                             sliding_window=cfg.sliding_window, segment_ids=segment_ids,
                             backward=cfg.mosaic_gpu_backward, persistent_forward=cfg.persistent_forward)
