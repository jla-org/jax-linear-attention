# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/gated_delta_rule/chunk.py, fla/ops/common/chunk_o.py
"""Gated DeltaNet and DeltaNet at large head dims (FLA's 1.3B shape ``K = 256``, ``V = 512``).

The default scalar-gate pipeline (:mod:`scalar_gate`) takes ``K, V`` in
{64, 128}: its fused forward keeps the transposed state ``[64, K]`` float32 in
one warpgroup's registers next to its 16-bit copy, and its per-chunk gradient
kernel holds two ``[V, K]`` states in shared memory. This path covers
``K = 256`` with ``V`` up to 512 (and ``V = 2K`` at ``K = 128``):

* forward: the gate-free inverse ``T0`` (:func:`gated_delta_rule_kernels.kkt_fwd`,
  which does not depend on ``K`` beyond the ``k k^T`` product), then one fused
  kernel (:func:`gated_delta_rule_kb.fused_fwd_kb`: the K-blocked state pass
  and the output ``o = scale (Gamma q S + P v_new)``, nothing but ``o`` and the
  final state reaches HBM). ``run(..., fused_fwd=False)`` keeps the forward
  through HBM: :func:`gated_delta_rule_kb.state_fwd_kb` stores the state
  entering every chunk and ``v_new``, then XLA batched matmuls form the output
  (the structure of FLA's ``chunk_fwd_h`` + ``chunk_fwd_o``);
* backward: the states again (:func:`state_fwd_kb`), the K-blocked reverse
  state pass (:func:`gated_delta_rule_kb.state_bwd_kb`), then the per-chunk
  gradients ``dq, dk, dv, dbeta, dg`` of every chunk in one kernel
  (:func:`gated_delta_rule_kb.chunk_grads_kb`: the algebra of
  :func:`gated_delta_rule_bwd.chunk_grads` with the states streamed in blocks
  of 64 value rows), then the reverse gate cumsum. ``run(..., fused_bwd=False)``
  keeps the per-chunk gradients as XLA batched matmuls (same algebra and
  roundings) for comparison.

Exact (no approximation; every matmul takes 16-bit operands with float32
accumulation, as the other kernels do). Packed documents (``segment_ids``),
initial and final states, grouped value heads (``H_v = n * H_qk``), ``scale``
and ``use_qk_l2norm`` as in :mod:`scalar_gate`; there is no sequence split
(every serial pass runs one CTA per ``(b, h, 64 value columns)``).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops.reference import l2norm
from jla.ops.mosaic_gpu import gated_delta_rule_kb as kb
from jla.ops.mosaic_gpu import gated_delta_rule_kernels as gdr
from jla.ops.mosaic_gpu._common import (
    CHUNK,
    SUPPORTED_DTYPES,
    Segments,
    chunk_log2_decay,
    head_major,
    is_hopper,
    pad_time,
    segment_vectors,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK
# (K, V) head dims this path accepts (the GPU tests cover each); others go to scalar_gate or chunked.
HEAD_DIMS = frozenset({(256, 128), (256, 256), (256, 512), (128, 256)})


def handles(q: Array, v: Array) -> bool:
    """True when the call's head dims belong to this path rather than :mod:`scalar_gate`."""
    return q.ndim == 4 and v.ndim == 4 and (q.shape[-1], v.shape[-1]) in HEAD_DIMS


@dataclasses.dataclass(frozen=True)
class _Config:
    gated: bool
    scale: float
    out_final: bool
    seq_len: int
    pad: int
    dtypes: tuple  # (q, v, g, beta, initial_state) dtypes; None for absent inputs
    interpret: Any = None
    fused_fwd: bool = True  # one forward kernel (False: states and v_new through HBM, output in XLA)
    fused_bwd: bool = True  # one per-chunk gradient kernel (False: the per-chunk gradients in XLA)


def _chunks(x: Array, heads: int | None = None) -> Array:
    """User layout ``[B, Tp, H, D]`` to head-major chunks ``[B, H, NC, C, D]`` (``heads``: repeat key heads)."""
    batch, padded, h, d = x.shape
    x = jnp.transpose(x.reshape(batch, padded // C, C, h, d), (0, 3, 1, 2, 4))
    if heads is not None and heads != h:
        x = jnp.repeat(x, heads // h, axis=1)
    return x


def _from_chunks(x: Array) -> Array:
    """Head-major chunks ``[B, H, NC, C, D]`` to the user layout ``[B, Tp, H, D]``."""
    batch, h, nc, _, d = x.shape
    return jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(batch, nc * C, h, d)


def _vec_chunks(x: Array) -> Array:
    """Per-token ``[B, H, Tp]`` to ``[B, H, NC, C]``."""
    batch, h, padded = x.shape
    return x.reshape(batch, h, padded // C, C)


def _seg_chunks(seg: Segments | None, batch: int, nc: int):
    """``(run, cin, tail)`` as ``[B, 1, NC, C]`` float32 (or ``None``)."""
    if seg is None:
        return None
    return tuple(x.reshape(batch, 1, nc, C) for x in seg)


def _einsum(spec: str, a: Array, b: Array) -> Array:
    return jnp.einsum(spec, a, b, preferred_element_type=F32)


def _gates(g2c: Array, segc):
    """Decay matrix ``D [.., C, C]`` (``i >= j``, one run), ``Gamma`` and the same-run mask."""
    rows = jnp.arange(C)[:, None]
    cols = jnp.arange(C)[None, :]
    keep = rows >= cols
    same = None
    gam = jnp.exp2(g2c)
    if segc is not None:
        run, cin, _ = segc
        same = run[..., :, None] == run[..., None, :]
        keep = keep & same
        gam = gam * cin
    decay = jnp.exp2(jnp.where(keep, g2c[..., :, None] - g2c[..., None, :], -jnp.inf))
    return decay, gam, same


def _output(qp, kp, vn, h, g2, seg, scale: float, out_dtype) -> Array:
    """``o = scale (Gamma (q S) + ((q k^T) * D) v_new)`` for every chunk, ``[B, Tp, H, V]``."""
    batch, _, heads, _ = vn.shape
    nc = g2.shape[-1] // C
    q, k = _chunks(qp, heads), _chunks(kp, heads)
    decay, gam, _ = _gates(_vec_chunks(g2), _seg_chunks(seg, batch, nc))
    p = (_einsum("bhnik,bhnjk->bhnij", q, k) * decay).astype(qp.dtype)
    o = _einsum("bhnck,bhnvk->bhncv", q, h) * gam[..., None]
    o = o + _einsum("bhnij,bhnjv->bhniv", p, _chunks(vn))
    return _from_chunks((o * scale).astype(out_dtype))


def _chunk_grads(qp, kp, vp, do, t0, g2, beta, h, dh, seg, scale: float):
    """``dq, dk`` (per key head), ``dv`` (input dtypes), ``dbeta`` and ``dG`` (natural units, float32) of every chunk.

    The algebra and the 16-bit roundings of :func:`gated_delta_rule_bwd.chunk_grads`,
    for all chunks at once: from the states entering (``h``) and the gradients of
    the states leaving (``dh``) each chunk.
    """
    dtype = qp.dtype
    batch, _, heads, _ = vp.shape
    nc = g2.shape[-1] // C
    q, k = _chunks(qp, heads), _chunks(kp, heads)
    v, dout = _chunks(vp), _chunks(do)
    t0c = t0.reshape(batch, heads, nc, C, C)
    t0f = t0c.astype(F32)
    g2c, bc = _vec_chunks(g2), _vec_chunks(beta)
    segc = _seg_chunks(seg, batch, nc)
    decay, gam, same = _gates(g2c, segc)
    g_last = g2c[..., C - 1]
    k_decay = jnp.exp2(g_last[..., None] - g2c)
    s_decay = jnp.exp2(g_last)
    if segc is not None:
        _, cin, tail = segc
        k_decay = k_decay * tail
        s_decay = s_decay * cin[..., C - 1]
    # Forward side: U = Gamma (k S), W = v - U, Ag = T0 * D * beta_j, v_new = Ag W.
    u = _einsum("bhnck,bhnvk->bhncv", k, h) * gam[..., None]
    w = (v.astype(F32) - u).astype(dtype)
    u = u.astype(dtype)
    ag = (t0f * decay * bc[..., None, :]).astype(dtype)
    vn = _einsum("bhnij,bhnjv->bhniv", ag, w).astype(dtype)
    # dP = scale (do v_new^T) * D; the gate gradient of D is the row sums minus the column sums of dP * (q k^T).
    qk = _einsum("bhnik,bhnjk->bhnij", q, k)
    dp = _einsum("bhniv,bhnjv->bhnij", dout, vn) * decay * scale
    e = dp * qk
    dg = e.sum(-1) - e.sum(-2)
    dp = dp.astype(dtype)
    dq = _einsum("bhniv,bhnvk->bhnik", dout, h) * (gam * scale)[..., None]  # scale Gamma (do S^T)
    dg = dg + (q.astype(F32) * dq).sum(-1)
    dq = dq + _einsum("bhnij,bhnjk->bhnik", dp, k)
    # Backward side: dv_new = scale P^T do + Gamma_r (k dS'), dW = Ag^T dv_new (= dv).
    pt = (jnp.swapaxes(qk * decay, -1, -2) * scale).astype(dtype)
    dvn = _einsum("bhnji,bhniv->bhnjv", pt, dout) + _einsum("bhnjk,bhnvk->bhnjv", k, dh) * k_decay[..., None]
    dvn = dvn.astype(dtype)
    dag = _einsum("bhniv,bhnjv->bhnij", dvn, w)  # dAg = dv_new W^T
    dv = _einsum("bhnji,bhnjv->bhniv", ag, dvn)
    dg = dg - (dv * u.astype(F32)).sum(-1)  # W = v - U
    dwn = (dv * (-gam)[..., None]).astype(dtype)
    # Ag = T0 * D * beta_j: to beta (column sums), the gates (row minus column sums) and T0.
    f0 = dag * t0f * decay
    f = f0 * bc[..., None, :]
    dg = dg + f.sum(-1) - f.sum(-2)
    db = f0.sum(-2)
    # Through the inverse: dA0 = -T0^T dT0 T0^T on the strictly lower part (T0 has a unit diagonal).
    strict = jnp.arange(C)[:, None] > jnp.arange(C)[None, :]
    ndt = jnp.where(strict, dag * decay * (-bc)[..., None, :], 0.0).astype(dtype)
    m = _einsum("bhnij,bhnkj->bhnik", ndt, t0c).astype(dtype)  # ndt T0^T
    if same is not None:
        strict = strict & same
    da = jnp.where(strict, _einsum("bhnji,bhnjk->bhnik", t0c, m), 0.0)  # T0^T m
    db = db + (da * _einsum("bhnik,bhnjk->bhnij", k, k)).sum(-1)
    das = (da * bc[..., :, None]).astype(dtype)
    # dk = Gamma_r (v_new dS'^T) - Gamma (dW S^T) + dP^T q + dA k + dA^T k (dA = dA0 * beta_i).
    vnr = (vn.astype(F32) * k_decay[..., None]).astype(dtype)
    dks = _einsum("bhncv,bhnvk->bhnck", vnr, dh)
    rks = (k.astype(F32) * dks).sum(-1)
    dk = dks + _einsum("bhncv,bhnvk->bhnck", dwn, h) + _einsum("bhnij,bhnik->bhnjk", dp, q)
    dk = dk + _einsum("bhnij,bhnjk->bhnik", das, k) + _einsum("bhnji,bhnjk->bhnik", das, k)
    # The last token also gets sum(rks) + gamma <S, dS'> (the outgoing decays); an elementwise
    # reduction, so XLA reads the 16-bit states once without float32 copies.
    sd = (h.astype(F32) * dh.astype(F32)).sum(axis=(-2, -1))
    total = rks.sum(-1) + sd * s_decay
    last = jnp.arange(C) == C - 1
    dg = dg - rks + jnp.where(last, total[..., None], 0.0)
    # Key-side gradients summed over each key head's group in float32, then all three in the input
    # dtype: the batch slices stack 16-bit results only.
    kv_heads = qp.shape[2]
    if heads != kv_heads:
        dq, dk = (x.reshape(batch, kv_heads, heads // kv_heads, *x.shape[2:]).sum(axis=2) for x in (dq, dk))
    return dq.astype(dtype), dk.astype(dtype), dv.astype(vp.dtype), db, dg


# The per-chunk gradients run on at most this many (token, value head) pairs at once (FLA's 1.3B layer
# at B * T = 16384 tokens); larger batches go through lax.map in slices of whole sequences, which bounds
# their float32 intermediates (about 2 KB per pair at K = 256, V = 512) without losing parallelism.
GRAD_HEAD_TOKENS = 6 * 16384


def _grad_rows(batch: int, padded: int, heads: int) -> int:
    """Sequences per slice of the per-chunk gradients (the largest divisor of ``batch`` that fits)."""
    rows = max(1, GRAD_HEAD_TOKENS // (padded * heads))
    return max(r for r in range(1, min(rows, batch) + 1) if batch % r == 0)


def _key_grad(x: Array, seq_len: int, dtype) -> Array:
    """Key-head chunks ``[B, Hk, NC, C, K]`` to ``[B, T, Hk, K]``."""
    return _from_chunks(x)[:, :seq_len].astype(dtype)


def _forward(q, k, v, g, beta, h0, seg, cfg: _Config):
    batch, seq_len, _, _ = q.shape
    heads = v.shape[2]
    qp, kp, vp = (pad_time(x, cfg.pad) for x in (q, k, v))
    g2 = chunk_log2_decay(g, batch, seq_len, heads, cfg.pad)
    bt = head_major(beta, cfg.pad)
    t0 = gdr.kkt_fwd(kp, bt, seg=seg, interpret=cfg.interpret)
    h0t = None if h0 is None else jnp.swapaxes(h0.astype(F32), -1, -2)  # the kernels keep the state transposed
    if cfg.fused_fwd:
        o, ht = kb.fused_fwd_kb(qp, kp, vp, t0, g2, bt, h0t, scale=cfg.scale, out_final=cfg.out_final,
                                out_dtype=q.dtype, seg=seg, interpret=cfg.interpret)
    else:
        h, vn, ht = kb.state_fwd_kb(kp, vp, t0, g2, bt, h0t, store_vn=True, out_final=cfg.out_final, seg=seg,
                                    interpret=cfg.interpret)
        o = _output(qp, kp, vn, h, g2, seg, cfg.scale, q.dtype)
    if ht is not None:
        ht = jnp.swapaxes(ht, -1, -2)
    residuals = (qp, kp, vp, g2, bt, t0, h0t, seg)
    return (o[:, :seq_len], ht), residuals


def _backward(cfg: _Config, residuals, cotangents):
    qp, kp, vp, g2, bt, t0, h0t, seg = residuals
    do, dht = cotangents
    q_dtype, v_dtype, g_dtype, beta_dtype, h0_dtype = cfg.dtypes
    batch = qp.shape[0]
    do = pad_time(do.astype(qp.dtype), cfg.pad)
    h, _, _ = kb.state_fwd_kb(kp, vp, t0, g2, bt, h0t, seg=seg, interpret=cfg.interpret)
    dht = None if (dht is None or not cfg.out_final) else jnp.swapaxes(dht.astype(F32), -1, -2)
    dh, dh0 = kb.state_bwd_kb(qp, kp, do, t0, g2, bt, dht, scale=cfg.scale, want_dh0=h0t is not None, seg=seg,
                              interpret=cfg.interpret)
    if cfg.fused_bwd:
        dq, dk, dv, db, dgc = _fused_chunk_grads(qp, kp, vp, do, t0, g2, bt, h, dh, seg, cfg)
        return _finish_grads(cfg, batch, dq, dk, dv, db, dgc, dh0)
    rows = _grad_rows(batch, qp.shape[1], vp.shape[2])
    if rows == batch:
        dq, dk, dv, db, dgc = _chunk_grads(qp, kp, vp, do, t0, g2, bt, h, dh, seg, cfg.scale)
    else:  # bound the float32 intermediates: `rows` sequences at a time

        def one(xs):
            out = _chunk_grads(*jax.tree.map(lambda x: x[None], xs), cfg.scale)
            return jax.tree.map(lambda x: x[0], out)

        dq, dk, dv, db, dgc = lax.map(one, (qp, kp, vp, do, t0, g2, bt, h, dh, seg), batch_size=rows)
    dq = _key_grad(dq, cfg.seq_len, q_dtype)
    dk = _key_grad(dk, cfg.seq_len, q_dtype)
    dv = _from_chunks(dv)[:, :cfg.seq_len].astype(v_dtype)
    return _finish_grads(cfg, batch, dq, dk, dv, db, dgc, dh0)


def _fused_chunk_grads(qp, kp, vp, do, t0, g2, bt, h, dh, seg, cfg: _Config):
    """The per-chunk gradients from :func:`gated_delta_rule_kb.chunk_grads_kb`, in the layout of :func:`_backward`."""
    q_dtype, v_dtype = cfg.dtypes[0], cfg.dtypes[1]
    batch, padded, kv_heads, key_dim = qp.shape
    heads = vp.shape[2]
    nc = padded // C
    dq, dk, dv, db, dg = kb.chunk_grads_kb(qp, kp, vp, do, t0, g2, bt, h, dh, scale=cfg.scale, seg=seg,
                                           interpret=cfg.interpret)
    if heads != kv_heads:  # per value head: sum each key head's group in float32
        dq, dk = (x.astype(F32).reshape(batch, padded, kv_heads, heads // kv_heads, key_dim).sum(axis=3)
                  for x in (dq, dk))
    dq, dk = (x[:, :cfg.seq_len].astype(q_dtype) for x in (dq, dk))
    dv = dv[:, :cfg.seq_len].astype(v_dtype)
    return dq, dk, dv, db.reshape(batch, heads, nc, C), dg.sum(axis=0).reshape(batch, heads, nc, C)


def _finish_grads(cfg: _Config, batch: int, dq, dk, dv, db, dgc, dh0):
    """Gate and beta gradients from the per-chunk ``dbeta``, ``dG`` ``[B, H, NC, C]``; the user layout."""
    _, _, g_dtype, beta_dtype, h0_dtype = cfg.dtypes
    dg = None
    if cfg.gated:
        dg = lax.cumsum(dgc, axis=3, reverse=True)  # [B, H, NC, C]: the chunk-local cumsum's transpose
        dg = jnp.swapaxes(dg.reshape(batch, dg.shape[1], -1), 1, 2)[:, :cfg.seq_len].astype(g_dtype)
    dbeta = jnp.swapaxes(db.reshape(batch, db.shape[1], -1), 1, 2)[:, :cfg.seq_len].astype(beta_dtype)
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


_OPTIONS = frozenset({"scale", "initial_state", "output_final_state", "cu_seqlens", "segment_ids", "chunk_size",
                      "use_qk_l2norm"})


def unsupported_reason(q, k, v, gates: tuple, options: dict) -> str | None:
    """Why this path cannot run the delta-rule call (``None`` if it can)."""
    extra = sorted(set(options) - _OPTIONS)
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
    if (key_dim, value_dim) not in HEAD_DIMS:
        return f"head dims (K, V) = {(key_dim, value_dim)} must be one of {sorted(HEAD_DIMS)}"
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


def run(q, k, v, g, beta, *, scale, initial_state, output_final_state, segment_ids=None, use_qk_l2norm=False,
        interpret=None, fused_fwd: bool = True, fused_bwd: bool = True):
    """Delta rule (``g=None``: DeltaNet) at the head dims of :data:`HEAD_DIMS` (inputs already validated).

    ``fused_fwd=False`` runs the forward through HBM (:func:`state_fwd_kb` storing
    the states and ``v_new``, then the output in XLA) instead of the fused kernel
    :func:`gated_delta_rule_kb.fused_fwd_kb` (kept for comparison).
    ``fused_bwd=False`` runs the per-chunk gradients as XLA batched matmuls
    (:func:`_chunk_grads`) instead of the kernel :func:`gated_delta_rule_kb.chunk_grads_kb`.
    """
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    seq_len = q.shape[1]
    pad = (-seq_len) % CHUNK
    seg = None if segment_ids is None else segment_vectors(segment_ids, pad)
    dtypes = tuple(None if x is None else jnp.dtype(x.dtype) for x in (q, v, g, beta, initial_state))
    cfg = _Config(gated=g is not None, scale=float(scale), out_final=bool(output_final_state), seq_len=seq_len,
                  pad=pad, dtypes=dtypes, interpret=interpret, fused_fwd=fused_fwd, fused_bwd=fused_bwd)
    o, ht = _core(q, k, v, g, beta, initial_state, seg, cfg)
    return o, (ht if output_final_state else None)
