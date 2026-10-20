# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/kda/chunk.py
# The channel-wise erase/write gates (GDN-2) are implemented from arXiv 2605.22791 (equations only).
"""Differentiable KDA / GDN-2 built from the Mosaic GPU kernels in :mod:`vector_gate`.

``chunked_vector_gate`` runs the forward kernels, saves what the backward needs (the per-chunk
states and WY factors, so the backward recomputes nothing) and runs the backward kernels in its
``jax.custom_vjp``. XLA only pads, casts, moves ``beta`` to the head-major layout and applies
``use_qk_l2norm`` (with its VJP), except on the forward without ``wy``, where ``intra`` does.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp

from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_bwd as vgb
from jla.ops.mosaic_gpu._common import MAX_GRID_YZ
from jla.ops.reference import l2norm, segment_resets

Array = jax.Array
_F32 = jnp.float32


@functools.cache
def is_hopper() -> bool:
    """True when the default JAX device is an sm_90 GPU (H100 / H200 / H800)."""
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", None) == "9.0"


def _to_heads(x: Array) -> Array:
    """``[B, T, H, ...]`` to ``[B, H, T, ...]``."""
    return jnp.swapaxes(x, 1, 2)


def _gate_inputs(cfg: vg.Config, gates: tuple[Array, ...]) -> tuple[Array, Array]:
    if cfg.channel_gates:
        return gates  # erase [B, T, H, K], write [B, T, H, V]
    beta = _to_heads(gates[0]).astype(_F32)  # [B, H, T]
    return beta, beta


def _forward_state_config(cfg: vg.Config, *, fused: bool) -> vg.Config:
    """Value tile of the fused state / output pass, one serial chain per ``(b, h, value tile)``.

    With few heads in flight (``B H <= 16``) the pass is latency-bound and 32-wide tiles (more,
    shorter chains) are fastest; with many it is bandwidth-bound and the widest tile, which reads
    each chunk's keys once, is fastest (measured on H100 for ``B H`` from 8 to 256). With ``fused``
    (``T`` applied in the pass) GDN-2's write-gate tiles leave SMEM for 64 columns at most.
    """
    bh = cfg.batch * cfg.heads
    if bh <= 16:
        block_v = 32
    elif bh <= 32 or cfg.value_dim < 128 or (fused and cfg.channel_gates):
        block_v = 64
    else:
        block_v = 128
    return dataclasses.replace(cfg, block_v=min(block_v, cfg.value_dim))


_WY_HEADS = 8  # forward-only calls with at most this many (batch, head) chains keep the wy pass

# With packed documents the state passes cut each row's chain at chunks where a document starts
# (vg.piece_table) when its chains alone would leave SMs idle: the same results, more chains in flight
# (KDA, H100, 1 x 16384 x 6 heads, 16 documents: the forward 0.59 -> 0.36 ms).
SPLIT_DOCUMENTS = True
# Value tile of a split forward pass: the pieces supply the chains, and 64-wide tiles beat the 32-wide
# ones of few heads (H100, KDA forward, 16 documents: 0.374 -> 0.365 ms at 6 heads, 0.769 -> 0.734 ms at 16).
SPLIT_BLOCK_V = 64


def _pieces(cfg: vg.Config, kill: Array | None) -> Array | None:
    """The state pass's piece table (``None``: one chain per row, without documents or splitting)."""
    if kill is None or not SPLIT_DOCUMENTS or cfg.interpret:
        return None
    n_pieces = vg.num_pieces(cfg)
    return None if n_pieces == 1 else vg.piece_table(kill, n_pieces)


def _split_state_config(cfg: vg.Config, kill: Array | None) -> vg.Config:
    """The forward state pass's config with documents to split at: :data:`SPLIT_BLOCK_V`-wide value
    tiles where those still leave chains to split (the same results; only the tiling changes)."""
    wide = dataclasses.replace(cfg, block_v=min(max(cfg.block_v, SPLIT_BLOCK_V), cfg.value_dim))
    if kill is None or not SPLIT_DOCUMENTS or cfg.interpret or vg.num_pieces(wide) == 1:
        return cfg
    return wide


def _run_forward(cfg: vg.Config, q, k, v, g, gates, h0, resets, *, save: bool):
    """``(o, final)`` and, with ``save``, the residuals of the backward.

    Without ``save`` (no gradient) and with more than ``_WY_HEADS`` ``(batch, head)`` chains,
    ``intra`` also writes the decayed tiles and the state pass applies ``T`` itself, so ``wy`` does
    not run (the backward needs its ``w``). Applying ``T`` adds a product to every serial step,
    which costs more than ``wy`` when few chains are in flight (H100, KDA at K = V = 128: 181 vs
    166 us at B H = 8, T = 4096; 1346 vs 1591 us at B H = 256, T = 2048). With ``cfg.l2norm``
    ``q`` and ``k`` are raw: ``intra`` scales them on that path, XLA (``l2norm``) on the others.
    """
    fused = not save and cfg.batch * cfg.heads > _WY_HEADS
    if cfg.l2norm and not fused:
        q, k = l2norm(q), l2norm(k)
        cfg = dataclasses.replace(cfg, l2norm=False)
    g_cum, marks, g_last, bounded = vg.chunk_cumsum(cfg, g, resets)  # resets: the -128 of segment starts
    kill = None if resets is None else resets.reshape(cfg.batch, -1, vg.CHUNK).max(axis=-1) > 0
    erase, write = _gate_inputs(cfg, gates)
    # With fuse_solve, intra returns T itself (warpgroup 1 inverts I + Akk in SMEM) instead of Akk.
    solve_in_intra = cfg.fuse_solve and not cfg.interpret
    if fused:
        aqk, akk, qg, neg_kd, kg = vg.intra(cfg, q, k, g_cum, marks, erase, bounded, decays=True,
                                            solve=solve_in_intra)
        t_inv = akk if solve_in_intra else vg.solve(cfg, akk)
        state_cfg = _split_state_config(_forward_state_config(cfg, fused=True), kill)
        outs = vg.state_output(state_cfg, kg, neg_kd, v, g_last, h0, qg, aqk, pieces=_pieces(state_cfg, kill),
                               save=False, t=t_inv, write=write)
        return outs, None
    aqk, akk = vg.intra(cfg, q, k, g_cum, marks, erase, bounded, solve=solve_in_intra)
    t_inv = akk if solve_in_intra else vg.solve(cfg, akk)
    w, u, kg, qg = vg.wy(cfg, t_inv, q, k, v, g_cum, marks, erase, write)
    state_cfg = _split_state_config(_forward_state_config(cfg, fused=False), kill)
    outs = vg.state_output(state_cfg, kg, w, u, g_last, h0, qg, aqk, save=save, pieces=_pieces(state_cfg, kill))
    if not save:
        return outs, None
    o, final, states, v_new = outs
    return (o, final), (q, k, v, g_cum, marks, g_last, gates, aqk, t_inv, w, kg, states, v_new, h0, kill)


def _forward(cfg: vg.Config, q, k, v, g, gates, h0, resets=None):
    if not cfg.l2norm:
        return _run_forward(cfg, q, k, v, g, gates, h0, resets, save=True)
    (q, k), l2norm_vjp = jax.vjp(lambda a, b: (l2norm(a), l2norm(b)), q, k)
    out, residuals = _run_forward(dataclasses.replace(cfg, l2norm=False), q, k, v, g, gates, h0, resets, save=True)
    return out, (residuals, l2norm_vjp)


def _backward(cfg: vg.Config, residuals, cotangents):
    if cfg.l2norm:
        residuals, l2norm_vjp = residuals
    q, k, v, g_cum, marks, g_last, gates, aqk, t_inv, w, kg, states, v_new, h0, kill = residuals
    do, d_final = cotangents
    erase, write = _gate_inputs(cfg, gates)
    do = do.astype(q.dtype)
    daqk, dv_o, qg = vgb.bwd_dav(cfg, q, g_cum, do, v_new, aqk)
    d_states, dv_new, dh0 = vgb.bwd_state(cfg, qg, kg, w, g_last, do, dv_o, d_final.astype(_F32),
                                          pieces=_pieces(cfg, kill))  # cut at document starts, walked in reverse
    wy = vgb.bwd_wy(cfg, qg, k, v, g_cum, marks, erase, write, t_inv, states, d_states, do, dv_new, v_new)
    dq_part, dk_part, dg_part, dv, dt = wy[:5]
    dakk = vgb.bwd_dakk(cfg, t_inv, dt)
    dq, dk, dg_tok, d_erase = vgb.bwd_intra(cfg, q, k, g_cum, marks, erase, daqk, dakk, dq_part, dk_part,
                                           dg_part, wy[5])
    # G is the chunk-local cumulative sum of g, so dg is the chunk-local reverse cumulative sum.
    dg = vg.chunk_reverse_cumsum(cfg, dg_tok)
    d_gates = (d_erase, wy[6]) if cfg.channel_gates else (d_erase,)
    if cfg.l2norm:
        dq, dk = l2norm_vjp((dq, dk))
    return dq, dk, dv, dg, d_gates, (None if h0 is None else dh0), None


def _primal(cfg: vg.Config, q, k, v, g, gates, h0, resets):
    return _run_forward(cfg, q, k, v, g, gates, h0, resets, save=False)[0]  # no gradient: nothing to save


_op = jax.custom_vjp(_primal, nondiff_argnums=(0,))
_op.defvjp(_forward, _backward)
# Under jax.checkpoint (ModelConfig.remat) the forward pass runs _primal, which saves nothing (with more
# than _WY_HEADS (batch, head) chains, the fused state pass without wy), and only the recomputation in the
# backward pass runs _forward for the residuals; without it the forward pass also ran _forward and threw its
# residuals away. The op's gradients are the same bit for bit (the backward reads _forward's residuals either
# way), and so are a model's with jax.checkpoint around its whole loss. The forward output is the no-gradient
# call's, which differs from _forward's in rounding only (the same products in another order, with other bf16
# intermediates: relative L2 about 4e-3 at B * H = 32, T = 4096, and slightly closer to a float32 reference;
# tests/test_gdn2_fast_paths.py checks both against the f32 chunked op). Under per-layer remat the layers
# outside the checkpoint read that output, so the gradients differ from those without remat at rounding level
# (dq up to about 1.5e-3 of its max).
# The interpreter keeps _op: JAX's remat optimization does not take forward functions with effects, and the
# interpreter runs the kernels through ordered host callbacks.
_op_remat = jax.custom_vjp(_primal, nondiff_argnums=(0,))
_op_remat.defvjp(_forward, _backward, optimize_remat=True)

# Extra log-decay of a segment's first token: exp(-128) underflows to zero in f32 (and every exponent
# in the kernels is a cumulative difference, at most zero), so no term crosses a segment start.
# Inside a segment the result is exact up to the f32 rounding of the chunk-local cumulative gate G:
# after n starts in a chunk G carries -128 n, where the f32 spacing is at most (128 n + |gate sum|)
# 2^-23 (about 1.5e-5 n), and the sequential scan rounds once per token, so a decay e^(G_i - G_j)
# between two tokens of one segment is off by at most about (i - j) half-spacings in its exponent
# (63 x 7.6e-6 n = 4.8e-4 n for a whole chunk). A chunk with a start always takes intra's exact
# path, whose 8x8 diagonal blocks multiply per-token decays and see no offset. Tested with 32 starts
# in one chunk (G offset -4096, spacing 4.9e-4): in-segment exponents off by at most 2.4e-3 (some
# pairs reach the bound), outputs as close to the sequential f32 reference as without starts
# (tests/test_mosaic_gpu_kda.py::test_dense_segments_match_reference, benchmarks/kda.md).
_RESET_DECAY = vg.RESET_DECAY


def chunked_vector_gate(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    gates: tuple[Array, ...],
    *,
    scale: float,
    initial_state: Array | None,
    segment_ids: Array | None = None,
    l2norm: bool = False,
    interpret: bool = False,
    fuse_solve: bool = False,
) -> tuple[Array, Array]:
    """``(o, final_state)`` for KDA (``gates = (beta,)``) or GDN-2 (``gates = (b, w)``).

    ``fuse_solve=True`` (opt-in) computes ``T`` inside ``intra`` instead of the separate ``solve``
    kernel: bit for bit the same results, but on H100 no faster (within 2 % of
    the separate solve's time either way; ``benchmarks/kda.md``), so the default keeps the separate
    kernel. With ``interpret`` the separate kernel always runs (the same results).

    Inputs use the FLA layout with any ``T`` (padded here to the chunk; zero padding is inert):
    ``q, k [B, T, H, K]`` and ``v [B, T, H, V]`` bf16, ``g [B, T, H, K]`` log-decay, ``beta
    [B, T, H]`` or ``b [B, T, H, K]`` and ``w [B, T, H, V]``. ``o`` has ``q``'s dtype; the final
    state is f32 ``[B, H, K, V]``. ``segment_ids [B, T]`` restarts the recurrence at every change
    of id (packed documents): the state entering a segment is zero, as in ``jla.ops.chunked``.
    ``l2norm`` scales ``q`` and ``k`` to unit L2 norm per token first (``reference.l2norm``).
    """
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    pad = (-seq_len) % vg.CHUNK
    channel_gates = len(gates) == 2

    def padded(x):
        return jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)) if pad else x

    g = g.astype(_F32)
    # 1.0 at each segment's first token: the cumsum kernel takes _RESET_DECAY off its log-decay there
    resets = None if segment_ids is None else padded(segment_resets(segment_ids).astype(_F32))
    if channel_gates:
        gates = tuple(padded(x.astype(jnp.bfloat16)) for x in gates)
    else:
        gates = (padded(gates[0].astype(_F32)),)
    if initial_state is not None:
        initial_state = initial_state.astype(_F32)
    cfg = vg.Config(
        batch=batch, seq_len=seq_len + pad, heads=heads, key_dim=key_dim, value_dim=value_dim,
        scale=float(scale), channel_gates=channel_gates, interpret=interpret, l2norm=l2norm,
        fuse_solve=fuse_solve,
    )
    op = _op if interpret else _op_remat  # KDA and GDN-2; the interpreter's callbacks are effects
    o, final = op(cfg, padded(q), padded(k), padded(v), padded(g), gates, initial_state, resets)
    return o[:, :seq_len], final


HEAD_DIMS = (64, 128)
_OPTIONS = frozenset({"scale", "initial_state", "output_final_state", "use_qk_l2norm", "cu_seqlens",
                      "segment_ids", "chunk_size", "sub_chunk"})


def unsupported_reason(q: Array, k: Array, v: Array, g: Array, gates_ok: bool, options: dict) -> str | None:
    """Why the KDA / GDN-2 kernels cannot run this call (``None`` if they can).

    ``gates_ok`` says whether the gate shapes are right. The kernels need sm_90, bf16 ``q, k, v``,
    ``K, V`` in {64, 128}, ``H * K`` a multiple of 128, ``T >= 1``, chunk 64 / sub-chunk 16, integer
    ``segment_ids [B, T]`` or none, and no ``cu_seqlens``.
    """
    extra = sorted(set(options) - _OPTIONS)
    if extra:
        return f"unsupported options {extra}"
    if not gates_ok:
        return "the gate shapes do not match q and v"
    if not is_hopper():
        return "the kernels need an sm_90 (Hopper) GPU"
    if options.get("cu_seqlens") is not None:
        return "the kernels take segment_ids, not cu_seqlens"
    if options.get("chunk_size", vg.CHUNK) != vg.CHUNK or options.get("sub_chunk", vg.SUB_CHUNK) != vg.SUB_CHUNK:
        return f"the kernels use chunk_size={vg.CHUNK}, sub_chunk={vg.SUB_CHUNK}"
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3] or g.shape != q.shape:
        return "q, k, g must be [B, T, H, K] and v [B, T, H, V]"
    if any(x.dtype != jnp.bfloat16 for x in (q, k, v)):
        return "q, k, v must be bfloat16"
    batch, seq_len, heads, key_dim = q.shape
    if seq_len < 1:
        return "the sequence is empty"
    if key_dim not in HEAD_DIMS or v.shape[-1] not in HEAD_DIMS:
        return f"head dims must be in {HEAD_DIMS}"
    if (heads * key_dim) % vg.SCAN_WIDTH:
        return f"H * K must be a multiple of {vg.SCAN_WIDTH}"
    # The per-chunk kernels put the chunk index on a CUDA grid y/z dimension; the backward WY kernel folds
    # min(K, V) / 64 channel slices into it as well.
    if -(-seq_len // vg.CHUNK) * max(1, min(key_dim, v.shape[-1]) // 64) > MAX_GRID_YZ:
        return (f"T={seq_len} is too long for one call (CUDA grid limit of {MAX_GRID_YZ} blocks per chunk-indexed "
                "dimension); split the sequence and chain initial_state")
    segment_ids = options.get("segment_ids")
    if segment_ids is not None and (
        segment_ids.shape != (batch, seq_len) or not jnp.issubdtype(segment_ids.dtype, jnp.integer)
    ):
        return "segment_ids must be an integer [B, T] array"
    state = options.get("initial_state")
    if state is not None and state.shape != (batch, heads, key_dim, v.shape[3]):
        return "initial_state must be [B, H, K, V]"
    return None


def supports(q: Array, k: Array, v: Array, g: Array, gates_ok: bool, options: dict) -> bool:
    """The ``supports`` predicate shared by the KDA and GDN-2 registrations."""
    return unsupported_reason(q, k, v, g, gates_ok, options) is None


def _gates_ok(q: Array, v: Array, gates: tuple[Array, ...]) -> bool:
    if len(gates) == 1:  # KDA: beta [B, T, H]
        return gates[0].shape == q.shape[:3]
    return len(gates) == 2 and gates[0].shape == q.shape and gates[1].shape == v.shape  # GDN-2: b, w


def run(
    name: str,
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    gates: tuple[Array, ...],
    *,
    scale: float | None,
    initial_state: Array | None,
    output_final_state: bool,
    cu_seqlens: Array | None,
    segment_ids: Array | None,
    use_qk_l2norm: bool,
    chunk_size: int,
    sub_chunk: int,
) -> tuple[Array, Array | None]:
    """The body of the public ``kda`` / ``gdn2`` entry points (``name`` is used in errors)."""
    if cu_seqlens is not None:
        raise NotImplementedError(f"the Mosaic GPU {name} kernels take segment_ids, not cu_seqlens")
    if chunk_size != vg.CHUNK or sub_chunk != vg.SUB_CHUNK:
        raise ValueError(f"the Mosaic GPU {name} kernels use chunk_size={vg.CHUNK}, sub_chunk={vg.SUB_CHUNK} "
                         f"(got {chunk_size}, {sub_chunk})")
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   segment_ids=segment_ids, use_qk_l2norm=use_qk_l2norm)
    reason = unsupported_reason(q, k, v, g, _gates_ok(q, v, gates), options)
    if reason is not None:
        raise ValueError(f"mosaic_gpu {name}: {reason}")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final = chunked_vector_gate(q, k, v, g, gates, scale=scale, initial_state=initial_state,
                                   segment_ids=segment_ids, l2norm=use_qk_l2norm)
    return o, (final if output_final_state else None)
