# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gated_delta_rule/chunk.py, fla/ops/simple_gla/chunk.py
# Adapted from FlashQLA (MIT), flash_qla/ops/gated_delta_rule/chunk/__init__.py
"""Mosaic GPU simple GLA (Mamba-2 SSD core) and the shared scalar-gate composition.

Delta rule (Gated DeltaNet, DeltaNet), FlashQLA's structure: gate cumsum (XLA)
-> the gate-free inverse ``T0`` (:func:`gated_delta_rule_kernels.kkt_fwd`, the
only saved intra-chunk factor) -> the fused forward, which forms
``v_new = (T0 * D * beta) (v - Gamma k S)`` in its chunk loop
(:func:`scalar_gate_fused.fused_fwd_wy`). The backward
(:mod:`gated_delta_rule_bwd`) recomputes the states (``state_fwd_wy``), runs
the reverse state pass (``state_bwd_wy``, which also yields ``dv``), then the
per-chunk gradients (``chunk_grads``) and the reverse gate cumsum (XLA).
``run(..., fused_bwd=True)`` replaces the last two with one kernel per
``(b, h)`` (:mod:`gated_delta_rule_fused_bwd`: exact, but slower than the split
kernels at every shape measured, so it runs only on request).
``run(..., wy=False)`` keeps the previous path, which materializes the WY
factors (``intra_fwd``) and runs the six-kernel FLA-style backward.

simple GLA (FLA ``chunk_simple_gla`` structure): the state pass fused with the
output (:func:`scalar_gate_fused.fused_fwd`); the backward runs ``state_fwd``,
``dv_local``, the reverse state pass and ``dqkwg``, then the reverse cumsum.

Packed documents (``segment_ids``) use the masks of
:func:`jla.ops.chunked.segment_masks` inside every kernel.

Sequence split. The serial state passes run one CTA per ``(b, h, 64 value
columns)``; when that leaves the GPU mostly idle (few long sequences, see
:func:`pieces`), every sequence is cut into ``cp`` pieces that run in parallel
from their own initial states (FlashQLA's intra-card context parallelism). A
piece maps its initial state affinely, ``S_end = M S_start + E``. Without the
delta rule ``M`` is the piece's total decay; with it, one extra state pass over
the augmented values ``[0 | v]`` from ``[I | 0]`` yields ``[M | E]``. A scan
over the pieces (XLA, float32) gives every piece its true initial state, and
the fused pass then runs from those. The backward splits the reverse state
pass the same way with the transposed maps ``M^T`` (one extra reverse pass per
piece from a zero end gradient). The split is exact; FlashQLA's gate-driven
warm-up, which starts each piece from zero a few chunks early, is not used.
Without an initial state the first piece's ``M`` is not computed and both
scans take the local term of their first step directly (no product with a
zero state); the forward map pass carries two row blocks per CTA when that
saves a wave (:mod:`scalar_gate_docs`). A piece whose successor starts a
document at its first token needs no forward map (its successor reads nothing
of it). With packed documents the delta rule's scans run as one Pallas Triton
kernel that skips the products of exactly-zero maps (:mod:`scalar_gate_scan`) from
8 pieces on; otherwise they stay unrolled in XLA, which is faster there.

The kernels are registered for ``impl="auto"``: ``supports`` accepts exactly
the calls the GPU tests cover (Hopper, bf16/fp16 ``q, k, v`` of one dtype,
``K, V`` in {64, 128}, any ``T``, grouped value heads, ``scale``,
``initial_state``, ``output_final_state``, ``segment_ids``, ``use_qk_l2norm``
for the delta rule). ``cu_seqlens`` and other chunk sizes fall through to
``jla.ops.chunked``.

Extension beyond the ``jla.ops`` reference interface: ``q`` and ``k`` may
have fewer heads than ``v`` (``H_v = n * H_qk``, Mamba-2 groups / GQA-style
value heads); value head ``h`` reads key head ``h // n``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops import register
from jla.ops.reference import l2norm
from jla.ops.mosaic_gpu import gated_delta_rule_bwd as gb
from jla.ops.mosaic_gpu import gated_delta_rule_fused_bwd as gfb
from jla.ops.mosaic_gpu import gated_delta_rule_kernels as gdr
from jla.ops.mosaic_gpu import scalar_gate_fused as sf
from jla.ops.mosaic_gpu import scalar_gate_kernels as sk
from jla.ops.mosaic_gpu import scalar_gate_docs as sgd
from jla.ops.mosaic_gpu import scalar_gate_grouped as sgg
from jla.ops.mosaic_gpu import scalar_gate_scan as sgs
from jla.ops.mosaic_gpu._common import (
    CHUNK,
    SUPPORTED_DTYPES,
    SUPPORTED_HEAD_DIMS,
    chunk_log2_decay,
    head_major,
    is_hopper,
    num_sms,
    pad_time,
    segment_vectors,
)
from jla.ops.mosaic_gpu._common import MAX_GRID_YZ

Array = jax.Array
F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST
# Sequence split policy, measured on an H100 (benchmarks/README.md): splitting pays only when the serial
# passes would run at most MAX_SPLIT_CTAS CTAs; each piece walks at least MIN_PIECE_CHUNKS chunks.
MAX_SPLIT_CTAS = 32
MIN_PIECE_CHUNKS = 8
# The delta rule also pays for the piece maps M (an extra state pass) and their scan: at most MAX_DELTA_PIECES
# pieces, and more than 4 only when every piece keeps at least DELTA_PIECE_CHUNKS_ABOVE_4 chunks (measured on an
# H100: 8 pieces beat 4 at (1, 16K-32K, 8 heads), lost at (1, 4K, 8); 16 heads fill the GPU with 4).
MAX_DELTA_PIECES = 8
DELTA_PIECE_CHUNKS_ABOVE_4 = 32


@dataclasses.dataclass(frozen=True)
class _Config:
    delta: bool
    gated: bool
    scale: float
    out_final: bool
    seq_len: int
    pad: int
    cp: int  # pieces per sequence for the serial state passes
    dtypes: tuple  # (q, v, g, beta, initial_state) dtypes; None for absent inputs
    interpret: Any = None
    wy: bool = True  # delta rule: the gate-free inverse path (False: the materialized WY factors)
    walk: bool = True  # split: the map passes walk only the chunks that matter (scalar_gate_docs)
    fused_bwd: bool = False  # delta rule, gate-free inverse: one backward kernel per (b, h)
    key_grad_splits: int = 0  # grouped simple GLA: > 0 sums dq, dk over a group's heads in the kernel
    warmup: float | None = None  # opt-in approximate split: log2-decay threshold of the warm-up windows
    scan_kernel: bool = True  # delta rule split: the piece scans as one kernel (scalar_gate_scan; False: XLA)
    skip_fresh: bool = True  # delta rule split: no forward map for a piece whose successor starts a document


def pieces(batch: int, heads: int, value_dim: int, seq_len: int, *, delta: bool) -> int:
    """Sequence pieces for the serial state passes (a power of two; 1 means no split).

    A sequence is split only when its passes would run at most
    ``MAX_SPLIT_CTAS`` CTAs (``B * H * V / 64``), and then into as many pieces as
    fit one wave on the GPU (at most ``MAX_DELTA_PIECES`` with the delta rule,
    and more than 4 only with ``DELTA_PIECE_CHUNKS_ABOVE_4`` chunks per piece),
    each at least ``MIN_PIECE_CHUNKS`` chunks long.
    """
    ctas = batch * heads * (value_dim // 64)
    if ctas > MAX_SPLIT_CTAS:
        return 1
    chunks = -(-seq_len // CHUNK)
    limit = min(MAX_DELTA_PIECES, num_sms() // ctas) if delta else num_sms() // ctas
    cp = 1
    while 2 * cp <= limit and chunks // (2 * cp) >= MIN_PIECE_CHUNKS:
        if delta and 2 * cp > 4 and chunks // (2 * cp) < DELTA_PIECE_CHUNKS_ABOVE_4:
            break
        cp *= 2
    return cp


def _scalars_to_user(x: Array, seq_len: int) -> Array:
    """Head-major per-token ``[B, H, Tp]`` to ``[B, T, H]``."""
    return jnp.swapaxes(x, 1, 2)[:, :seq_len]


def _key_grad(x: Array, seq_len: int, kv_heads: int, dtype) -> Array:
    """Per-value-head ``[B, Tp, H, K]`` key-side gradient to ``[B, T, Hk, K]``, summed over groups."""
    batch, padded, heads, dim = x.shape
    x = x[:, :seq_len]
    if heads != kv_heads:
        x = x.astype(jnp.float32).reshape(batch, seq_len, kv_heads, heads // kv_heads, dim).sum(axis=3)
    return x.astype(dtype)


def _piece_decay(g2: Array, seg, cp: int) -> Array:
    """Total decay of every piece without the delta rule, ``[B, cp, H]`` (0 once a document ends in it)."""
    batch, heads, padded = g2.shape
    decay = jnp.exp2(g2[..., CHUNK - 1::CHUNK].reshape(batch, heads, cp, -1).sum(-1))
    if seg is not None:
        keep = seg.cin[:, CHUNK - 1::CHUNK].reshape(batch, cp, -1).prod(-1)
        decay = decay * keep[:, None, :]
    return jnp.swapaxes(decay, 1, 2)


# Without packed documents the piece scans run in XLA (faster there, see _scan); True runs the kernel anyway.
DENSE_SCAN_KERNEL = False
# The kernel scan only from this many pieces: below it XLA's scan is short and the kernel's extra steps (the
# zero-map flags, the backward's transposed maps) cost as much as they save (measured on an H100 at 4 pieces:
# (1, 2048, 8) with 128-token documents, 0.129 against 0.121 ms forward + backward).
SCAN_KERNEL_MIN_PIECES = 8


def _scan(local, maps, init, zero, cfg: _Config, *, value_dim: int, local_row0: int, reverse: bool) -> Array:
    """The delta rule's forward (piece starts) or reverse (piece end gradients) scan, ``[B, cp, H, V, K]``.

    With packed documents (``zero`` given) the Pallas Triton kernel of :mod:`scalar_gate_scan`, which skips the
    products of the pieces whose map is exactly zero (one with a document start, or followed by a piece that
    starts with one): at (1, 16384, 8 heads) with 8 pieces, 9 us against 28 for the XLA scan when every piece
    has a start; from ``SCAN_KERNEL_MIN_PIECES`` pieces on. Without documents no map is zero, and XLA's
    unrolled scan is faster than the kernel (28 us against 47 at 8 pieces), so it stays (``zero`` is ``None``).
    """
    if cfg.scan_kernel and zero is not None and cfg.cp >= SCAN_KERNEL_MIN_PIECES:
        return sgs.piece_scan(local, maps, init, zero, value_dim=value_dim, local_row0=local_row0, reverse=reverse,
                              interpret=cfg.interpret)
    # XLA computes every product anyway (a zero map gives the local term exactly), so it takes no flags.
    return sgs.piece_scan_xla(local, maps, init, None, value_dim=value_dim, local_row0=local_row0, reverse=reverse)


def _piece_starts(kp, w, u, g2, h0t, seg, cfg: _Config, t0=None, beta=None) -> tuple[Array, Array]:
    """True initial state of every piece (transposed, ``[B * cp, H, V, K]`` float32) and the piece maps.

    The maps are ``([M^T | E^T] [B, cp, H, K + V, K], zero-map flags [B, cp, H])``
    with the delta rule and the total decays ``[B, cp, H]`` without it. With the
    gate-free inverse ``t0`` the delta rule's pass reads the values ``u`` (user
    layout) instead of the WY factors ``w, u``.
    """
    batch = kp.shape[0]
    key_dim = kp.shape[-1]
    heads, value_dim = g2.shape[1], u.shape[-1]
    cp = cfg.cp
    if cfg.delta:
        # One pass over the augmented values [0 | u] from [I | 0]: rows :K of the
        # transposed end state hold M^T, rows K: hold E^T.
        if t0 is not None and cfg.walk:
            end = sgd.map_fwd_wy(kp, u, t0, g2, beta, seg=seg, cp=cp, need_last=cfg.out_final,
                                 need_first=h0t is not None, warmup=cfg.warmup, skip_fresh=cfg.skip_fresh,
                                 interpret=cfg.interpret)
            # The pieces whose M the walk leaves exactly zero: their scan steps skip the product.
            zero = sgd.zero_maps(sgd.fwd_walk_table(seg, g2, cp, need_last=cfg.out_final,
                                                    need_first=h0t is not None, warmup=cfg.warmup,
                                                    skip_fresh=cfg.skip_fresh))
        else:
            if t0 is not None:
                _, end = gb.state_fwd_wy(kp, u, t0, g2, beta, None, store_states=False, out_final=True, aug=True,
                                         seg=seg, cp=cp, interpret=cfg.interpret)
            else:
                u_aug = jnp.concatenate([jnp.zeros(u.shape[:-1] + (key_dim,), u.dtype), u], axis=-1)
                eye = jnp.broadcast_to(jnp.eye(key_dim, dtype=F32), (batch * cp, heads, key_dim, key_dim))
                h_aug = jnp.concatenate([eye, jnp.zeros((batch * cp, heads, value_dim, key_dim), F32)], axis=2)
                _, _, end = sk.state_fwd(kp, w, u_aug, g2, h_aug, u_head_major=True, out_final=True,
                                         store_states=False, seg=seg, cp=cp, interpret=cfg.interpret)
            zero = None
            if seg is not None:  # the full walk leaves M exactly zero after a document start
                killed = (seg.cin[:, CHUNK - 1::CHUNK] == 0).reshape(batch, cp, -1).any(axis=-1)
                zero = jnp.broadcast_to(killed[:, :, None], (batch, cp, heads)).astype(jnp.int32)
        end = end.reshape(batch, cp, heads, key_dim + value_dim, key_dim)
        # Without document starts no map is zero, and the scan stays in XLA (see _scan).
        if seg is None and DENSE_SCAN_KERNEL:
            zero = jnp.zeros((batch, cp, heads), jnp.int32)
        elif seg is None or zero is None:
            zero = None
        else:
            zero = zero.reshape(batch, cp, heads)
        starts = _scan(end, end, h0t, zero, cfg, value_dim=value_dim, local_row0=key_dim, reverse=False)
        return starts.reshape(batch * cp, heads, value_dim, key_dim), (end, zero)
    elif cfg.walk:  # simple GLA: only the chunks that reach each piece end (scalar_gate_docs)
        # The maps are the piece decays, so no pass needs the last piece.
        local = sgd.map_fwd_gla(kp, u, g2, seg=seg, cp=cp, need_last=False, interpret=cfg.interpret)
        local = local.reshape(batch, cp, heads, value_dim, key_dim)
        maps = _piece_decay(g2, seg, cp)

        def advance(s, m, e):
            return s * m[..., None, None] + e
    else:
        _, _, local = sk.state_fwd(kp, None, u, g2, None, u_head_major=False, out_final=True, store_states=False,
                                   seg=seg, cp=cp, interpret=cfg.interpret)
        local = local.reshape(batch, cp, heads, value_dim, key_dim)
        maps = _piece_decay(g2, seg, cp)

        def advance(s, m, e):
            return s * m[..., None, None] + e

    # A short scan over the pieces, unrolled so XLA can fuse the elementwise steps. From a zero
    # state the first step is the local end state itself (no product, and the first map unread).
    s = jnp.zeros((batch, heads, value_dim, key_dim), F32) if h0t is None else h0t
    starts = []
    for i in range(cp):
        starts.append(s)
        if i + 1 < cp:
            s = local[:, i] if (i == 0 and h0t is None) else advance(s, maps[:, i], local[:, i])
    return jnp.stack(starts, axis=1).reshape(batch * cp, heads, value_dim, key_dim), maps


def _piece_end_grads(qp, kp, w, g2, do, dvl, dht_t, seg, maps, cfg: _Config, t0=None, beta=None) -> Array:
    """Gradient of the state leaving every piece (transposed, ``[B * cp, H, V, K]`` float32)."""
    batch = qp.shape[0]
    key_dim = qp.shape[-1]
    heads, value_dim = g2.shape[1], do.shape[-1]
    cp = cfg.cp
    # Each piece's gradient at its start from a zero gradient at its end.
    if t0 is not None and cfg.walk:
        local = sgd.map_bwd_wy(qp, kp, do, t0, g2, beta, scale=cfg.scale, seg=seg, cp=cp, warmup=cfg.warmup,
                               interpret=cfg.interpret)
    elif t0 is not None:
        _, _, local = gb.state_bwd_wy(qp, kp, do, t0, g2, beta, None, scale=cfg.scale, want_dh0=True,
                                      store_grads=False, seg=seg, cp=cp, interpret=cfg.interpret)
    elif w is None and cfg.walk:  # simple GLA
        local = sgd.map_bwd_gla(qp, kp, g2, do, scale=cfg.scale, seg=seg, cp=cp, interpret=cfg.interpret)
    else:
        _, _, local = sk.state_bwd(qp, kp, w, g2, do, dvl, None, scale=cfg.scale, want_dh0=True, store_grads=False,
                                   seg=seg, cp=cp, interpret=cfg.interpret)
    local = local.reshape(batch, cp, heads, value_dim, key_dim)
    if cfg.delta:  # dS_start^T = dS_end^T M + D^T, with the maps' rows :K holding M^T
        end, zero = maps
        ends = _scan(local, end, dht_t, zero, cfg, value_dim=value_dim, local_row0=0, reverse=True)
        return ends.reshape(batch * cp, heads, value_dim, key_dim)

    def retreat(gt, m, d):
        return gt * m[..., None, None] + d

    gt = jnp.zeros((batch, heads, value_dim, key_dim), F32) if dht_t is None else dht_t
    ends = [gt]
    for i in range(cp - 1, 0, -1):  # piece i - 1 ends where piece i starts (no product from a zero gradient)
        gt = local[:, i] if (i == cp - 1 and dht_t is None) else retreat(gt, maps[:, i], local[:, i])
        ends.append(gt)
    return jnp.stack(ends[::-1], axis=1).reshape(batch * cp, heads, value_dim, key_dim)


def _forward(q, k, v, g, beta, h0, seg, cfg: _Config, v_scale=None):
    """``v_scale`` (simple GLA without a sequence split, forward only): the values are ``(v * v_scale)`` rounded
    to ``v``'s dtype, formed in the fused kernel (:func:`scalar_gate_fused.fused_fwd`, ``u_scale``)."""
    batch, seq_len, _, _ = q.shape
    heads = v.shape[2]
    pad = cfg.pad
    qp, kp, vp = (pad_time(x, pad) for x in (q, k, v))
    g2 = chunk_log2_decay(g, batch, seq_len, heads, pad)
    # The serial kernels keep the state transposed ([V, K]); so do the saved and returned states.
    h0t = None if h0 is None else jnp.swapaxes(h0.astype(F32), -1, -2)
    starts = maps = None
    if cfg.delta:
        # The gate-free inverse is the only intra-chunk factor saved; the fused pass
        # applies the decay, beta and the WY factors in its chunk loop.
        bt = head_major(beta, pad)
        if cfg.wy:
            t = gdr.kkt_fwd(kp, bt, seg=seg, interpret=cfg.interpret)
            if cfg.cp > 1:
                starts, maps = _piece_starts(kp, None, vp, g2, h0t, seg, cfg, t0=t, beta=bt)
            o, ht = sf.fused_fwd_wy(qp, kp, vp, t, g2, bt, h0t if starts is None else starts, scale=cfg.scale,
                                    out_final=cfg.out_final, out_dtype=q.dtype, seg=seg, cp=cfg.cp,
                                    interpret=cfg.interpret)
        else:  # materialized WY factors, then the fused pass over them
            t, w, u = gdr.intra_fwd(kp, vp, g2, bt, None, seg=seg, interpret=cfg.interpret)
            if cfg.cp > 1:
                starts, maps = _piece_starts(kp, w, u, g2, h0t, seg, cfg)
            o, ht = sf.fused_fwd(qp, kp, w, u, g2, h0t if starts is None else starts, scale=cfg.scale,
                                 u_head_major=True, out_final=cfg.out_final, out_dtype=q.dtype, seg=seg, cp=cfg.cp,
                                 interpret=cfg.interpret)
    else:
        bt = t = None
        if cfg.cp > 1:
            starts, maps = _piece_starts(kp, None, vp, g2, h0t, seg, cfg)
        o, ht = sf.fused_fwd(qp, kp, None, vp, g2, h0t if starts is None else starts, scale=cfg.scale,
                             u_head_major=False, out_final=cfg.out_final, out_dtype=q.dtype, seg=seg, cp=cfg.cp,
                             u_scale=None if v_scale is None else head_major(v_scale, pad), interpret=cfg.interpret)
    if ht is not None:
        ht = jnp.swapaxes(ht.reshape(batch, cfg.cp, *ht.shape[1:])[:, -1], -1, -2)  # the last piece's end
    residuals = (qp, kp, vp, g2, bt, t, h0t, seg, starts, maps)
    return (o[:, :seq_len], ht), residuals


# Grouped simple GLA (Mamba-2's B/C groups): the per-value-head key gradients leave dqkwg in the input dtype,
# as when q, k are repeated per head, and _key_grad sums a group's heads in float32; half the bytes of
# float32 per-head gradients. False writes them in float32.
GROUPED_SIMPLE_GLA_KEY_GRADS_IN_INPUT_DTYPE = True


def _backward(cfg: _Config, residuals, cotangents):
    qp, kp, vp, g2, bt, t, h0t, seg, starts, maps = residuals
    do, dht = cotangents
    q_dtype, v_dtype, g_dtype, beta_dtype, h0_dtype = cfg.dtypes
    batch = qp.shape[0]
    kv_heads = qp.shape[2]
    do = pad_time(do.astype(qp.dtype), cfg.pad)
    if cfg.delta and cfg.wy:
        return _backward_wy(cfg, residuals, do, dht)
    if cfg.delta:
        _, w, u = gdr.intra_fwd(kp, vp, g2, bt, t, seg=seg, interpret=cfg.interpret)
    else:
        w, u = None, vp
    h, vn, _ = sk.state_fwd(kp, w, u, g2, h0t if starts is None else starts, u_head_major=cfg.delta,
                            out_final=False, seg=seg, cp=cfg.cp, interpret=cfg.interpret)
    dvl = sk.dv_local(qp, kp, g2, do, scale=cfg.scale, seg=seg, interpret=cfg.interpret)
    dht = None if (dht is None or not cfg.out_final) else jnp.swapaxes(dht.astype(F32), -1, -2)
    if cfg.cp > 1:
        dht = _piece_end_grads(qp, kp, w, g2, do, dvl, dht, seg, maps, cfg)
    dh, dvn, dh0 = sk.state_bwd(qp, kp, w, g2, do, dvl, dht, scale=cfg.scale, want_dh0=h0t is not None, seg=seg,
                                cp=cfg.cp, interpret=cfg.interpret)
    if dh0 is not None:
        dh0 = dh0.reshape(batch, cfg.cp, *dh0.shape[1:])[:, 0]  # the user's state enters the first piece
    # Key-side gradients leave the kernels in [B, Tp, H, K]; they are final in the
    # input dtype unless they still need a sum over key-head groups.
    grouped = kv_heads != vp.shape[2]
    key_dtype = F32 if grouped else qp.dtype
    dbeta = None
    if cfg.delta:
        dq, dk_part, dw, dgc = sk.dqkwg(qp, kp, vn, g2, h, dh, do, dvn, scale=cfg.scale, vn_head_major=True,
                                       dq_dtype=key_dtype, dk_dtype=F32, seg=seg, interpret=cfg.interpret)
        dk, dv, dbeta, dgc2 = gdr.wy_bwd(kp, vp, g2, bt, t, dw, dvn, dk_part, dk_dtype=key_dtype, seg=seg,
                                         interpret=cfg.interpret)
        dgc = dgc + dgc2
        dbeta = _scalars_to_user(dbeta, cfg.seq_len).astype(beta_dtype)
    elif grouped and cfg.key_grad_splits:  # the group sums leave the kernel (scalar_gate_grouped)
        dq, dk, dgc = sgg.dqkwg_grouped(qp, kp, vp, g2, h, dh, do, scale=cfg.scale, splits=cfg.key_grad_splits,
                                        seg=seg, interpret=cfg.interpret)
        if cfg.key_grad_splits > 1:
            dq, dk = (x.reshape(*x.shape[:2], kv_heads, cfg.key_grad_splits, -1).sum(axis=3) for x in (dq, dk))
        dv = dvn
    else:
        if grouped and GROUPED_SIMPLE_GLA_KEY_GRADS_IN_INPUT_DTYPE:
            key_dtype = qp.dtype
        dq, dk, _, dgc = sk.dqkwg(qp, kp, vp, g2, h, dh, do, None, scale=cfg.scale, vn_head_major=False,
                                  dq_dtype=key_dtype, dk_dtype=key_dtype, seg=seg, interpret=cfg.interpret)
        dv = dvn
    dg = None
    if cfg.gated:
        heads, padded = dgc.shape[1:]
        dgc = lax.cumsum(dgc.reshape(batch, heads, padded // CHUNK, CHUNK), axis=3, reverse=True)
        dg = _scalars_to_user(dgc.reshape(batch, heads, padded), cfg.seq_len).astype(g_dtype)
    dq = _key_grad(dq, cfg.seq_len, kv_heads, q_dtype)
    dk = _key_grad(dk, cfg.seq_len, kv_heads, q_dtype)
    dv = dv[:, :cfg.seq_len].astype(v_dtype)
    dh0 = None if dh0 is None else jnp.swapaxes(dh0, -1, -2).astype(h0_dtype)
    return dq, dk, dv, dg, dbeta, dh0, None


def _backward_wy(cfg: _Config, residuals, do, dht):
    """Backward of the delta rule from the gate-free inverse (:mod:`gated_delta_rule_bwd`)."""
    qp, kp, vp, g2, bt, t0, h0t, seg, starts, maps = residuals
    batch = qp.shape[0]
    kv_heads = qp.shape[2]
    h, _ = gb.state_fwd_wy(kp, vp, t0, g2, bt, h0t if starts is None else starts, seg=seg, cp=cfg.cp,
                           interpret=cfg.interpret)
    dht = None if (dht is None or not cfg.out_final) else jnp.swapaxes(dht.astype(F32), -1, -2)
    if cfg.fused_bwd:  # one kernel per (b, h): the reverse state pass and the per-chunk gradients
        dq, dk, dv, dbeta, dgc, dh0 = gfb.fused_bwd(qp, kp, vp, do, t0, g2, bt, h, dht, scale=cfg.scale,
                                                    want_dh0=h0t is not None, seg=seg, interpret=cfg.interpret)
        return _finish_wy(cfg, batch, kv_heads, dq, dk, dv, dbeta, dgc, dh0)
    if cfg.cp > 1:
        dht = _piece_end_grads(qp, kp, None, g2, do, None, dht, seg, maps, cfg, t0=t0, beta=bt)
    dh, dv, dh0 = gb.state_bwd_wy(qp, kp, do, t0, g2, bt, dht, scale=cfg.scale, want_dh0=h0t is not None, seg=seg,
                                  cp=cfg.cp, interpret=cfg.interpret)
    if dh0 is not None:
        dh0 = dh0.reshape(batch, cfg.cp, *dh0.shape[1:])[:, 0]  # the user's state enters the first piece
    dq, dk, dbeta, dgc = gb.chunk_grads(qp, kp, vp, do, t0, g2, bt, h, dh, scale=cfg.scale, seg=seg,
                                        interpret=cfg.interpret)
    return _finish_wy(cfg, batch, kv_heads, dq, dk, dv, dbeta, dgc, dh0)


def _finish_wy(cfg: _Config, batch, kv_heads, dq, dk, dv, dbeta, dgc, dh0):
    """The user-layout gradients from the gate-free backward's kernel outputs."""
    q_dtype, v_dtype, g_dtype, beta_dtype, h0_dtype = cfg.dtypes
    dg = None
    if cfg.gated:
        heads, padded = dgc.shape[1:]
        dgc = lax.cumsum(dgc.reshape(batch, heads, padded // CHUNK, CHUNK), axis=3, reverse=True)
        dg = _scalars_to_user(dgc.reshape(batch, heads, padded), cfg.seq_len).astype(g_dtype)
    dbeta = _scalars_to_user(dbeta, cfg.seq_len).astype(beta_dtype)
    dq = _key_grad(dq, cfg.seq_len, kv_heads, q_dtype)
    dk = _key_grad(dk, cfg.seq_len, kv_heads, q_dtype)
    dv = dv[:, :cfg.seq_len].astype(v_dtype)
    dh0 = None if dh0 is None else jnp.swapaxes(dh0, -1, -2).astype(h0_dtype)
    return dq, dk, dv, dg, dbeta, dh0, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _core(q, k, v, g, beta, h0, seg, cfg: _Config):
    out, _ = _forward(q, k, v, g, beta, h0, seg, cfg)
    return out


def _core_fwd(q, k, v, g, beta, h0, seg, cfg):
    return _forward(q, k, v, g, beta, h0, seg, cfg)


def _core_bwd(cfg, residuals, cotangents):
    return _backward(cfg, residuals, cotangents)


_core.defvjp(_core_fwd, _core_bwd)


_COMMON_OPTIONS = frozenset({"scale", "initial_state", "output_final_state", "cu_seqlens", "segment_ids",
                             "chunk_size"})


def unsupported_reason(q, k, v, gates: tuple, options: dict, *, delta: bool) -> str | None:
    """Why the Mosaic GPU scalar-gate kernels cannot run this call (``None`` if they can)."""
    allowed = _COMMON_OPTIONS | ({"use_qk_l2norm"} if delta else set())
    extra = sorted(set(options) - allowed)
    if extra:
        return f"unsupported options {extra}"
    if not is_hopper():
        return "the Mosaic GPU kernels need an sm_90 (Hopper) GPU"
    if options.get("cu_seqlens") is not None:
        return "cu_seqlens is not supported; pass segment_ids or use the reference"
    if options.get("chunk_size", CHUNK) != CHUNK:
        return f"chunk_size must be {CHUNK}"
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        return f"expected q, k [B, T, Hk, K] and v [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}"
    batch, seq_len, kv_heads, key_dim = q.shape
    heads, value_dim = v.shape[2:]
    if heads % kv_heads:
        return f"value heads {heads} must be a multiple of key heads {kv_heads}"
    if seq_len == 0:
        return "empty sequence"
    if key_dim not in SUPPORTED_HEAD_DIMS or value_dim not in SUPPORTED_HEAD_DIMS:
        return f"head dims K={key_dim}, V={value_dim} must be in {SUPPORTED_HEAD_DIMS}"
    cp = pieces(batch, heads, value_dim, seq_len, delta=delta)
    if -(-seq_len // (CHUNK * cp)) * cp > MAX_GRID_YZ or batch * cp > MAX_GRID_YZ:
        return (f"T={seq_len} needs more than {MAX_GRID_YZ} chunks of {CHUNK} tokens in one call (the CUDA grid "
                "limit of the per-chunk kernels); split the sequence and chain initial_state")
    dtypes = {jnp.dtype(x.dtype) for x in (q, k, v)}
    if len(dtypes) != 1 or dtypes.pop() not in SUPPORTED_DTYPES:
        return f"q, k, v must share a dtype in {SUPPORTED_DTYPES}; got {q.dtype}, {k.dtype}, {v.dtype}"
    for gate in gates:
        if gate.shape != (batch, seq_len, heads):
            return f"gates must be [B, T, H] = {(batch, seq_len, heads)}; got {gate.shape}"
    state = options.get("initial_state")
    if state is not None and state.shape != (batch, heads, key_dim, value_dim):
        return f"initial_state must be [B, H, K, V] = {(batch, heads, key_dim, value_dim)}; got {state.shape}"
    segment_ids = options.get("segment_ids")
    if segment_ids is not None and (segment_ids.shape != (batch, seq_len)
                                    or not jnp.issubdtype(segment_ids.dtype, jnp.integer)):
        return f"segment_ids must be integer [B, T] = {(batch, seq_len)}; got {segment_ids.dtype}{segment_ids.shape}"
    return None


def run(q, k, v, g, beta, *, delta: bool, scale, initial_state, output_final_state, segment_ids=None,
        use_qk_l2norm=False, cp: int | None = None, wy: bool = True, interpret=None, walk: bool = True,
        fused_bwd: bool = False, grouped_key_grads: bool = False, key_grad_splits: int | None = None,
        cp_warmup: float | None = None, v_scale=None, piece_scan_kernel: bool = True,
        skip_fresh_maps: bool = True):
    """Shared entry for the three kinds (inputs already validated).

    ``v_scale [B, T, H]`` (simple GLA, forward only: not differentiable) runs on the values
    ``(v * v_scale[..., None])`` rounded to ``v``'s dtype, as XLA computes that product, formed inside the fused
    forward kernel instead of in HBM (Mamba-2's ``dt x``); calls that split the sequence form them in XLA.

    ``cp`` overrides the number of sequence pieces (default: :func:`pieces`).
    ``wy=False`` selects the previous delta-rule path, which materializes the
    WY factors (kept for comparison). ``walk=False`` makes the map passes of
    the sequence split walk every chunk (the path before
    :mod:`scalar_gate_docs`, kept for comparison). ``fused_bwd=True`` runs
    the delta rule's backward as one kernel per ``(b, h)``
    (:mod:`gated_delta_rule_fused_bwd`; gate-free path, ``K == V``, no
    sequence split: ``cp`` defaults to 1). It is exact but slower than the
    default split kernels at every shape measured on an H100
    (``benchmarks/gdn.md``), so it is never chosen automatically.
    ``grouped_key_grads=True`` (simple GLA with fewer key heads than value
    heads, Mamba-2's B/C groups) sums ``dq``, ``dk`` over each group's heads
    in the backward kernel (:mod:`scalar_gate_grouped`, ``key_grad_splits``
    CTAs per group, default :func:`scalar_gate_grouped.default_splits`); other
    calls ignore it.
    ``cp_warmup`` (opt-in, approximate; default ``None`` = exact) is FlashQLA's
    intra-card CP warm-up for the delta rule's split: a natural-log decay
    threshold (FlashQLA uses ``-10``). Every piece then starts from the state
    its predecessor's last chunks build from zero, once those chunks decay below
    the threshold (their share of the incoming state is at most ``e^cp_warmup``
    for unit-norm keys), and the backward likewise; pieces whose decay never
    gets there are split exactly (:func:`scalar_gate_docs.warmup_windows`).
    ``piece_scan_kernel=False`` runs the delta rule's piece scans unrolled in XLA (the previous path) instead
    of :mod:`scalar_gate_scan`; ``skip_fresh_maps=False`` computes the forward map of a piece whose
    successor starts a document at its first token (unread; the previous path).
    """
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    batch, seq_len = q.shape[:2]
    if cp is None:
        cp = 1 if fused_bwd else pieces(batch, v.shape[2], v.shape[3], seq_len, delta=delta)
    pad = (-seq_len) % (CHUNK * cp)
    seg = None if segment_ids is None else segment_vectors(segment_ids, pad)
    dtypes = tuple(None if x is None else jnp.dtype(x.dtype) for x in (q, v, g, beta, initial_state))
    if fused_bwd and not (delta and wy and cp == 1 and k.shape[-1] == v.shape[-1]):
        raise ValueError("fused_bwd needs the delta rule's gate-free path (wy=True), cp == 1 and K == V")
    if cp_warmup is not None and not (delta and wy and walk):
        raise ValueError("cp_warmup needs the delta rule's gate-free path (wy=True) with the walked map passes")
    if cp_warmup is not None and not float(cp_warmup) < 0:  # 0 or above would drop the carried state everywhere
        raise ValueError(f"cp_warmup must be a negative natural-log decay threshold (FlashQLA uses -10), "
                         f"got {cp_warmup}")
    splits = 0
    kv_heads, heads = k.shape[2], v.shape[2]
    if grouped_key_grads and not delta and kv_heads != heads:
        splits = key_grad_splits or sgg.default_splits(batch, (seq_len + pad) // CHUNK, kv_heads, heads // kv_heads)
    cfg = _Config(delta=delta, gated=g is not None, scale=float(scale), out_final=bool(output_final_state),
                  seq_len=seq_len, pad=pad, cp=cp, dtypes=dtypes, interpret=interpret, wy=wy, walk=walk,
                  fused_bwd=fused_bwd, key_grad_splits=splits,
                  warmup=None if cp_warmup is None else float(cp_warmup) / math.log(2.0),
                  scan_kernel=bool(piece_scan_kernel), skip_fresh=bool(skip_fresh_maps))
    if v_scale is not None and (delta or cp > 1):  # the scale lives only in the unsplit simple GLA kernel
        v = (v.astype(F32) * v_scale.astype(F32)[..., None]).astype(v.dtype)
        v_scale = None
    if v_scale is not None:
        (o, ht), _ = _forward(q, k, v, g, beta, initial_state, seg, cfg, v_scale=v_scale)
    else:
        o, ht = _core(q, k, v, g, beta, initial_state, seg, cfg)
    return o, (ht if output_final_state else None)


def simple_gla(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = CHUNK,
) -> tuple[Array, Array | None]:
    """Linear attention with one log-decay per head on Hopper (Mamba-2 SSD: ``scale=1``).

    Same contract as :func:`jla.ops.reference.simple_gla`; ``q``/``k`` may
    have fewer heads than ``v`` (see the module docstring).
    """
    options = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size)
    reason = unsupported_reason(q, k, v, (g,), options, delta=False)
    if reason is not None:
        raise ValueError(f"mosaic_gpu simple_gla: {reason}")
    return run(q, k, v, g, None, delta=False, scale=scale, initial_state=initial_state,
               output_final_state=output_final_state, segment_ids=segment_ids)


def supports_simple_gla(q, k, v, g, **options) -> bool:
    return unsupported_reason(q, k, v, (g,), options, delta=False) is None


register("mosaic_gpu", "simple_gla", simple_gla, supports_simple_gla, auto=True)
