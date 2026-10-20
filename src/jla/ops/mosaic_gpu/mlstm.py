# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mLSTM (xLSTM) on the Mosaic GPU simple-GLA kernels (opt-in: ``impl="mosaic_gpu"``).

The max-stabilised mLSTM of :func:`jla.ops.reference.mlstm` is a simple GLA
(one log-decay per head) on modified inputs. With the stabiliser
``m_t = max(g_t + m_{t-1}, i_t)`` the stabilised state
``c_t = exp(-m_t) C_t`` obeys::

    c_t = exp(g'_t) c_{t-1} + k'_t^T v_t,   g'_t = g_t + m_{t-1} - m_t <= 0,
                                             k'_t = exp(i_t - m_t) k_t  (|k'| <= |k|)

and the normaliser ``n_t`` obeys the same recurrence with ``v_t = 1``. One
simple-GLA kernel pass gives the numerator ``q c_t``; the denominator ``q n_t`` is computed chunkwise in XLA from
float32-accumulated scores (see :func:`_normalizer`), and
``o_t = q c_t / (max(|q n_t|, exp(-m_t)) + eps)`` follows elementwise. The
stabiliser is the reference's own (the ``eps`` term depends on it), computed
with a max-plus ``lax.associative_scan``; ``g'`` and ``k'`` never overflow
because ``m_t`` dominates every write. A segment start (``segment_ids``)
resets ``m`` as in the reference (the kernels mask the state); the initial
state enters already decayed into the first token's stabiliser units.

Precision. The numerator carries the kernels' bf16 operand rounding (``k'``,
the intra-chunk scores, the inter-chunk state and, for ``K = 256``, each
128-wide key block's output before the float32 sum), and dividing by a small
``|q n|`` amplifies it at rows where ``q`` is nearly orthogonal to the state's
dominant key (measured in ``benchmarks/mlstm.md``). This path is therefore
registered with ``auto=False``: ``impl="auto"`` keeps the float32 chunked op.

The simple-GLA kernels take ``K, V`` blocks of 64 or 128. A larger key dim is
split into 128-wide key heads whose outputs are summed (``q c = sum over the
blocks``); the value columns become value-head groups of ``value_block``
columns per key head (the kernels' grouped value heads). Gradients: the
kernels' custom VJP (:mod:`scalar_gate`) for the simple-GLA core, autodiff for
the normaliser, the elementwise maps and the stabiliser scan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jla.ops import chunked, register
from jla.ops.mosaic_gpu import scalar_gate as sg
from jla.ops.mosaic_gpu._common import CHUNK, SUPPORTED_DTYPES, is_hopper
from jla.ops.reference import MLSTM_EMPTY_M, MLSTMState, mlstm_floor, segment_resets

Array = jax.Array
F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST
KEY_DIMS = (64, 128, 256)  # 256 runs as two 128-wide key blocks
VALUE_DIMS = (64, 128, 256, 512)
_OPTIONS = frozenset({"scale", "eps", "initial_state", "output_final_state", "cu_seqlens", "segment_ids",
                      "chunk_size"})


def stabilizer(g: Array, i: Array, m0: Array | None, resets: Array | None) -> Array:
    """The reference stabiliser ``m_t = max(g_t + m_{t-1}, i_t)`` for every token, ``[B, T, H]`` float32.

    A max-plus associative scan: token ``t`` is the map ``x -> max(x + g_t, i_t)``
    (``g_t`` replaced by the empty stabiliser at a segment start, which forgets
    ``m_{t-1}``), and ``(a1, b1)`` then ``(a2, b2)`` compose to
    ``(a1 + a2, max(b1 + a2, b2))``.
    """
    g, i = g.astype(F32), i.astype(F32)
    if resets is not None:
        g = jnp.where(resets[..., None], MLSTM_EMPTY_M, g)

    def combine(x, y):
        return x[0] + y[0], jnp.maximum(x[1] + y[0], y[1])

    total, best = lax.associative_scan(combine, (g, i), axis=1)
    start = jnp.full(g.shape[::2], MLSTM_EMPTY_M, F32) if m0 is None else m0.astype(F32)
    return jnp.maximum(start[:, None, :] + total, best)


def _blocks(key_dim: int, value_dim: int, value_block: int | None) -> tuple[int, int]:
    """(key block, value block) widths for the simple-GLA kernels."""
    kb = min(key_dim, 128)
    vb = value_block or (128 if value_dim % 128 == 0 else 64)
    if key_dim % kb or value_dim % vb or vb not in (64, 128):
        raise ValueError(f"mlstm: K={key_dim}, V={value_dim} do not split into kernel blocks (value_block={vb})")
    return kb, vb


def _simple_gla(q, k, v, g, h0, segment_ids, *, scale: float, out_final: bool):
    """The Mosaic GPU simple-GLA kernels (``q, k [B, T, Hk, K]``, ``v [B, T, H, V]``, ``o`` in the input dtype).

    Under a context mesh only this kernel call runs per shard
    (:func:`jla.distributed.kernel_sharding.shard_kernel_call`, kind ``simple_gla``: the batch over the
    ``batch`` rule's axes, the heads over ``tp`` when every operand has the same heads, else the batch
    only); the stabiliser, the denominator and the elementwise maps around it are plain XLA, which the
    SPMD partitioner splits. (Splitting the whole mLSTM call inside ``shard_map`` crashed XLA's GPU
    compiler on the gradient.)
    """
    from jla.distributed.kernel_sharding import shard_kernel_call

    def kernel(q, k, v, g, **options):
        return sg.run(q, k, v, g, None, delta=False, **options)

    def local_ok(q, k, v, g, **options) -> bool:
        local = {name: options.get(name) for name in ("initial_state", "segment_ids")}
        return sg.unsupported_reason(q, k, v, (g,), local, delta=False) is None

    options = dict(scale=scale, initial_state=h0, output_final_state=out_final, segment_ids=segment_ids)
    return shard_kernel_call(kernel, q, k, v, (g,), options, supports=local_ok, kind="simple_gla")


def unsupported_reason(q, k, v, g, i, options: dict, *, value_block: int | None = None) -> str | None:
    """Why this path cannot run the call (``None`` if it can)."""
    extra = sorted(set(options) - _OPTIONS)
    if extra:
        return f"unsupported options {extra}"
    if not is_hopper():
        return "the Mosaic GPU kernels need an sm_90 (Hopper) GPU"
    if options.get("cu_seqlens") is not None:
        return "cu_seqlens is not supported; pass segment_ids"
    if options.get("chunk_size", CHUNK) != CHUNK:
        return f"chunk_size must be {CHUNK}"
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        return f"expected q, k [B, T, H, K] and v [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}"
    batch, seq_len, heads, key_dim = q.shape
    if seq_len == 0:
        return "empty sequence"
    if key_dim not in KEY_DIMS or v.shape[-1] not in VALUE_DIMS:
        return f"head dims K={key_dim}, V={v.shape[-1]} must be in {KEY_DIMS} and {VALUE_DIMS}"
    dtypes = {jnp.dtype(x.dtype) for x in (q, k, v)}
    if len(dtypes) != 1 or dtypes.pop() not in SUPPORTED_DTYPES:
        return f"q, k, v must share a dtype in {SUPPORTED_DTYPES}; got {q.dtype}, {k.dtype}, {v.dtype}"
    for name, gate in (("g", g), ("i", i)):
        if gate.shape != (batch, seq_len, heads):
            return f"gate {name} must be [B, T, H] = {(batch, seq_len, heads)}; got {gate.shape}"
    state = options.get("initial_state")
    if state is not None:
        want = ((batch, heads, key_dim, v.shape[-1]), (batch, heads, key_dim), (batch, heads))
        if len(state) != 3 or tuple(tuple(x.shape) for x in state) != want:
            return f"initial_state must be an MLSTMState or (c, n, m) tuple with shapes {want}"
    segment_ids = options.get("segment_ids")
    if segment_ids is not None and (segment_ids.shape != (batch, seq_len)
                                    or not jnp.issubdtype(segment_ids.dtype, jnp.integer)):
        return f"segment_ids must be integer [B, T] = {(batch, seq_len)}; got {segment_ids.dtype}{segment_ids.shape}"
    # The simple-GLA kernels' own limits (the CUDA grid of their per-chunk kernels) on the shapes they get.
    try:
        kb, vb = _blocks(key_dim, v.shape[-1], value_block)
    except ValueError as err:
        return str(err)
    key_heads = heads * (key_dim // kb)
    value_heads = key_heads * (v.shape[-1] // vb)
    kernel_qk = jax.ShapeDtypeStruct((batch, seq_len, key_heads, kb), q.dtype)
    kernel_v = jax.ShapeDtypeStruct((batch, seq_len, value_heads, vb), v.dtype)
    kernel_g = jax.ShapeDtypeStruct((batch, seq_len, value_heads), F32)
    return sg.unsupported_reason(kernel_qk, kernel_qk, kernel_v, (kernel_g,), {"segment_ids": segment_ids},
                                 delta=False)


def _normalizer(q, k, g_mod, w, n_init, segment_ids, scale: float) -> tuple[Array, Array]:
    """``q_t n_t`` (scaled, ``[B, T, H]``) and the final ``n`` in float32, chunkwise in XLA.

    ``n_t = exp(g'_t) n_{t-1} + w_t k_t`` with ``w = exp(i - m)``. The scores
    ``q k^T`` are bf16 products accumulated in float32 (exact products, as the
    reference), so the denominator carries no bf16 rounding of the stabilised
    weights, the intra-chunk scores or the states; its relative error stays
    small where ``|q n|`` nearly cancels. The chunk states come from an
    associative scan over the chunks.
    """
    batch, seq_len, heads, key_dim = q.shape
    pad = (-seq_len) % CHUNK
    n_chunks = (seq_len + pad) // CHUNK

    def chunks(x):  # [B, T, H, ...] -> [B, NC, H, C, ...]; padding: no decay, no write
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return jnp.moveaxis(x.reshape(batch, n_chunks, CHUNK, *x.shape[2:]), 2, 3)

    qc, kc = chunks(q), chunks(k)
    beta = jnp.cumsum(chunks(g_mod), axis=-1)  # [B, NC, H, C], non-increasing
    wc = chunks(w)
    causal = jnp.tril(jnp.ones((CHUNK, CHUNK), bool))
    if segment_ids is None:
        same, carry_in, tail = causal, True, True
    else:
        same, carry_in, tail = chunked.segment_masks(segment_ids, CHUNK, pad)
        same = same & causal
    scores = jnp.einsum("bnhck,bnhdk->bnhcd", qc, kc, preferred_element_type=F32)
    decay = jnp.exp(jnp.where(same, beta[..., :, None] - beta[..., None, :], -jnp.inf))
    den = jnp.sum(scores * decay * wc[..., None, :], axis=-1)
    # State entering every chunk: n_out[c] = a_c n_out[c - 1] + local_c.
    local = jnp.einsum("bnhc,bnhck->bnhk", jnp.where(tail, jnp.exp(beta[..., -1:] - beta), 0.0) * wc,
                       kc.astype(F32), precision=HIGHEST)
    keep = jnp.where(carry_in[..., -1] if segment_ids is not None else True, jnp.exp(beta[..., -1]), 0.0)

    def combine(x, y):
        return x[0] * y[0], y[0][..., None] * x[1] + y[1]

    total, acc = lax.associative_scan(combine, (keep, local), axis=1)
    n_out = acc if n_init is None else acc + total[..., None] * n_init[:, None]
    start = jnp.zeros_like(n_out[:, :1]) if n_init is None else n_init[:, None]
    n_in = jnp.concatenate([start, n_out[:, :-1]], axis=1)  # [B, NC, H, K]
    inter = jnp.einsum("bnhck,bnhk->bnhc", qc.astype(F32), n_in, precision=HIGHEST)
    den = den + jnp.where(carry_in, jnp.exp(beta), 0.0) * inter
    den = jnp.moveaxis(den, 3, 2).reshape(batch, n_chunks * CHUNK, heads)[:, :seq_len]
    return scale * den, n_out[:, -1]


def mlstm(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    i: Array,
    *,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state: MLSTMState | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    chunk_size: int = CHUNK,
    value_block: int | None = None,
) -> tuple[Array, MLSTMState | None]:
    """The mLSTM cell of :func:`jla.ops.reference.mlstm` on Hopper (same contract and stabiliser).

    ``value_block`` (64 or 128; default 128 when it divides ``V``) is the width
    of the value-head blocks handed to the kernels.
    """
    options = dict(scale=scale, eps=eps, initial_state=initial_state, output_final_state=output_final_state,
                   cu_seqlens=cu_seqlens, segment_ids=segment_ids, chunk_size=chunk_size)
    reason = unsupported_reason(q, k, v, g, i, options, value_block=value_block)
    if reason is not None:
        raise ValueError(f"mosaic_gpu mlstm: {reason}")
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    scale = key_dim**-0.5 if scale is None else float(scale)
    kb, vb = _blocks(key_dim, value_dim, value_block)
    nk, nv = key_dim // kb, value_dim // vb
    dtype = q.dtype

    resets = None if segment_ids is None else segment_resets(segment_ids)
    m0 = None if initial_state is None else initial_state[2]  # MLSTMState or a plain (c, n, m) tuple
    m = stabilizer(g, i, m0, resets)  # [B, T, H]
    m_prev = jnp.concatenate([jnp.zeros_like(m[:, :1]), m[:, :-1]], axis=1)
    first = jnp.zeros((batch, seq_len), bool).at[:, 0].set(True)
    restart = first if resets is None else first | resets
    # Token 0's decay of the initial state is applied to the state itself; segment starts are masked.
    g_mod = jnp.where(restart[..., None], 0.0, g.astype(F32) + m_prev - m)
    w = jnp.exp(i.astype(F32) - m)  # write weights, <= 1
    k_mod = (w[..., None] * k.astype(F32)).astype(dtype)

    # Kernel layout: key head (h, a) = h * nk + a (128-wide key blocks); value head ((h, a), c)
    # reads value block c.
    def key_heads(x):
        return x.reshape(batch, seq_len, heads * nk, kb)

    v_ext = jnp.broadcast_to(v.reshape(batch, seq_len, heads, 1, nv, vb), (batch, seq_len, heads, nk, nv, vb))
    v_ext = v_ext.reshape(batch, seq_len, heads * nk * nv, vb)
    g_ext = jnp.broadcast_to(g_mod[..., None], (batch, seq_len, heads, nk * nv)).reshape(batch, seq_len, -1)
    h0 = n_init = None
    if initial_state is not None:
        c0, n0, m_init = (x.astype(F32) for x in initial_state)
        decay0 = jnp.exp(g[:, 0].astype(F32) + m_init - m[:, 0])  # [B, H], <= 1
        n_init = n0 * decay0[..., None]
        h0 = (c0 * decay0[..., None, None]).reshape(batch, heads, nk, kb, nv, vb)
        h0 = jnp.moveaxis(h0, 4, 3).reshape(batch, heads * nk * nv, kb, vb)

    o, ht = _simple_gla(key_heads(q), key_heads(k_mod), v_ext, g_ext, h0, segment_ids, scale=scale,
                        out_final=bool(output_final_state))
    num = o.reshape(batch, seq_len, heads, nk, value_dim).astype(F32).sum(axis=3)
    den, n_final = _normalizer(q, k, g_mod, w, n_init, segment_ids, scale)
    out = num / (jnp.maximum(jnp.abs(den), mlstm_floor(m)) + eps)[..., None]
    final = None
    if output_final_state:
        ht = ht.reshape(batch, heads, nk, nv, kb, vb)
        c = jnp.moveaxis(ht, 3, 4).reshape(batch, heads, key_dim, value_dim)
        final = MLSTMState(c, n_final, m[:, -1])
    return out.astype(dtype), final


def supports_mlstm(q, k, v, g, i, **options) -> bool:
    return unsupported_reason(q, k, v, g, i, options) is None


# auto=False: the bf16 operand rounding of the kernels leaves a per-row error tail where q is nearly
# orthogonal to the state's dominant key (benchmarks/mlstm.md); impl="auto" keeps the float32 chunked op.
register("mosaic_gpu", "mlstm", mlstm, supports_mlstm, auto=False)
