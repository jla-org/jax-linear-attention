# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Token-by-token reference recurrences for the linear-attention family.

These functions are the correctness oracles for every fast implementation in
``jla.ops``. They run a ``lax.scan`` over time in float32 with HIGHEST
matmul precision and follow the conventions of flash-linear-attention (FLA):

* ``q``/``k`` are ``[B, T, H, K]``, ``v`` is ``[B, T, H, V]``;
* ``g`` is a log-space forget gate (``g <= 0``): one value per head
  ``[B, T, H]`` or one per key channel ``[B, T, H, K]``;
* ``beta`` is the delta-rule write strength ``[B, T, H]``;
* the recurrent state ``S`` is ``[K, V]`` per head and the output is
  ``o_t = scale * q_t S_t``, where ``S_t`` already includes token ``t``;
* ``cu_seqlens`` (int ``[N + 1]``, ``cu_seqlens[-1] == T``) packs ``N``
  sequences into a batch of one; the state restarts at every sequence start,
  ``initial_state`` is then ``[N, H, K, V]`` and so is the final state;
* ``segment_ids`` (int ``[B, T]``) is the static-shape alternative for packed
  documents: the state restarts from zero wherever the id changes along ``T``,
  ``initial_state`` applies to each row's first segment and the final state is
  the state after each row's last token.

Recurrences, with ``S'`` the decayed previous state:

    simple_gla:        S_t = exp(g_t) S_{t-1} + k_t^T v_t
    gla:               S_t = diag(exp(g_t)) S_{t-1} + k_t^T v_t
    delta_rule:        S_t = S_{t-1} + beta_t k_t^T (v_t - k_t S_{t-1})
    gated_delta_rule:  S' = exp(g_t) S_{t-1};        S_t = S' + beta_t k_t^T (v_t - k_t S')
    kda:               S' = diag(exp(g_t)) S_{t-1};  S_t = S' + beta_t k_t^T (v_t - k_t S')
    gdn2:              S' = diag(exp(g_t)) S_{t-1};  S_t = S' + k_t^T (w_t v_t - (b_t k_t) S')
    mamba3:            S_t = exp(g_t) S_{t-1} + beta_t k_{t-1}^T v_{t-1} + gamma_t k_t^T v_t

Every function returns ``(o, final_state)``; ``o`` has the dtype of ``q`` and
``final_state`` is float32, or ``None`` unless ``output_final_state`` is set.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

__all__ = [
    "Mamba3State",
    "delta_rule",
    "gated_delta_rule",
    "gdn2",
    "gla",
    "kda",
    "l2norm",
    "mamba3",
    "segment_resets",
    "simple_gla",
]

Array = jax.Array
Gate = Literal["none", "scalar", "vector"]

_HIGHEST = lax.Precision.HIGHEST


def l2norm(x: Array, eps: float = 1e-6) -> Array:
    """Scale the last axis to unit L2 norm, computed in float32 (FLA ``l2norm``)."""
    x32 = x.astype(jnp.float32)
    inv = lax.rsqrt(jnp.sum(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * inv).astype(x.dtype)


def segment_resets(segment_ids: Array) -> Array:
    """``[B, T]`` bool: True where a token starts a new segment (never at ``t = 0``)."""
    change = segment_ids[:, 1:] != segment_ids[:, :-1]
    return jnp.concatenate([jnp.zeros_like(segment_ids[:, :1], dtype=bool), change], axis=1)


def _check_shapes(
    q, k, v, g, beta, gate: Gate, delta: bool, cu_seqlens, segment_ids=None, erase=None, write=None
) -> None:
    if cu_seqlens is not None and segment_ids is not None:
        raise ValueError("pass cu_seqlens or segment_ids, not both")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(f"q and k must both be [B, T, H, K]; got {q.shape} and {k.shape}")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must be [B, T, H, V] matching q; got {v.shape} for q {q.shape}")
    head_shape = q.shape[:3]
    if gate == "scalar" and g.shape != head_shape:
        raise ValueError(f"g must be [B, T, H] = {head_shape}; got {g.shape}")
    if gate == "vector" and g.shape != q.shape:
        raise ValueError(f"g must be [B, T, H, K] = {q.shape}; got {g.shape}")
    if delta and erase is None and beta.shape != head_shape:
        raise ValueError(f"beta must be [B, T, H] = {head_shape}; got {beta.shape}")
    if erase is not None and (erase.shape != q.shape or write is None or write.shape != v.shape):
        raise ValueError(
            f"b must be [B, T, H, K] = {q.shape} and w must be [B, T, H, V] = {v.shape}; "
            f"got {erase.shape} and {None if write is None else write.shape}"
        )
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("cu_seqlens packs sequences along T and requires batch size 1")


def _recurrence(
    q: Array,
    k: Array,
    v: Array,
    g: Array | None,
    beta: Array | None,
    *,
    gate: Gate,
    delta: bool,
    scale: float | None,
    initial_state: Array | None,
    output_final_state: bool,
    cu_seqlens: Array | None,
    use_qk_l2norm: bool,
    segment_ids: Array | None = None,
    erase: Array | None = None,
    write: Array | None = None,
) -> tuple[Array, Array | None]:
    _check_shapes(q, k, v, g, beta, gate, delta, cu_seqlens, segment_ids, erase, write)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    f32 = jnp.float32
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = key_dim**-0.5
    q = q.astype(f32) * scale
    k = k.astype(f32)
    v = v.astype(f32)
    g = None if g is None else g.astype(f32)
    beta = None if beta is None else beta.astype(f32)
    erase = None if erase is None else erase.astype(f32)
    write = None if write is None else write.astype(f32)

    def step(state: Array, token) -> tuple[Array, Array]:
        # q, k [H, K]; v [H, V]; g [H] or [H, K]; beta [H]; erase [H, K]; write [H, V];
        # reset_t a scalar bool. Unused entries are None.
        q_t, k_t, v_t, g_t, beta_t, erase_t, write_t, reset_t = token
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
        if gate == "scalar":
            state = state * jnp.exp(g_t)[:, None, None]
        elif gate == "vector":
            state = state * jnp.exp(g_t)[:, :, None]
        if delta and erase_t is None:
            recalled = jnp.einsum("hk,hkv->hv", k_t, state, precision=_HIGHEST)
            v_t = beta_t[:, None] * (v_t - recalled)
        elif delta:
            recalled = jnp.einsum("hk,hkv->hv", erase_t * k_t, state, precision=_HIGHEST)
            v_t = write_t * v_t - recalled
        state = state + k_t[:, :, None] * v_t[:, None, :]
        return state, jnp.einsum("hk,hkv->hv", q_t, state, precision=_HIGHEST)

    if cu_seqlens is None:
        if initial_state is None:
            initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)

        resets = None if segment_ids is None else segment_resets(segment_ids)

        def run(q, k, v, g, beta, erase, write, resets, state0):
            final, o = lax.scan(step, state0, (q, k, v, g, beta, erase, write, resets))
            return o, final

        o, final = jax.vmap(run)(q, k, v, g, beta, erase, write, resets, initial_state.astype(f32))
    else:
        cu = cu_seqlens.astype(jnp.int32)
        num_seqs = cu.shape[0] - 1
        if initial_state is None:
            initial_state = jnp.zeros((num_seqs, heads, key_dim, value_dim), f32)
        state0_all = initial_state.astype(f32)
        t = jnp.arange(seq_len, dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu[1:], t, side="right").astype(jnp.int32)
        is_start = t == cu[seq_id]
        is_end = t == cu[seq_id + 1] - 1

        def varlen_step(carry, token):
            state, finals = carry
            sid, start, end, *rest = token
            state = jnp.where(start, state0_all[sid], state)
            state, o_t = step(state, (*rest, None))
            finals = finals.at[sid].set(jnp.where(end, state, finals[sid]))
            return (state, finals), o_t

        tokens = (
            seq_id,
            is_start,
            is_end,
            q[0],
            k[0],
            v[0],
            None if g is None else g[0],
            None if beta is None else beta[0],
            None if erase is None else erase[0],
            None if write is None else write[0],
        )
        # Empty sequences never see a token, so their final state stays the initial one.
        (_, final), o = lax.scan(varlen_step, (state0_all[0], state0_all), tokens)
        o = o[None]

    return o.astype(out_dtype), (final if output_final_state else None)


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
) -> tuple[Array, Array | None]:
    """Linear attention with one log-decay per head (Mamba-2 / RetNet / Lightning family)."""
    return _recurrence(
        q, k, v, g, None, gate="scalar", delta=False, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=False, segment_ids=segment_ids,
    )


def gla(
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
) -> tuple[Array, Array | None]:
    """Gated linear attention with a log-decay per key channel."""
    return _recurrence(
        q, k, v, g, None, gate="vector", delta=False, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=False, segment_ids=segment_ids,
    )


def delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[Array, Array | None]:
    """DeltaNet: the delta-rule write with no decay."""
    return _recurrence(
        q, k, v, None, beta, gate="none", delta=True, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, segment_ids=segment_ids,
    )


def gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[Array, Array | None]:
    """Gated DeltaNet (Qwen3-Next / Qwen3.5): delta rule with one log-decay per head."""
    return _recurrence(
        q, k, v, g, beta, gate="scalar", delta=True, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, segment_ids=segment_ids,
    )


def kda(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[Array, Array | None]:
    """Kimi Delta Attention: delta rule with a log-decay per key channel."""
    return _recurrence(
        q, k, v, g, beta, gate="vector", delta=True, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, segment_ids=segment_ids,
    )


def gdn2(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    b: Array,
    w: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[Array, Array | None]:
    """Gated DeltaNet-2 (Gated Delta Rule-2, arXiv 2605.22791, Eq. 10).

    ``S_t = (I - k_t (b_t * k_t)^T) diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T``,
    with a channel-wise erase gate ``b`` (``[B, T, H, K]``) on the key side and a
    channel-wise write gate ``w`` (``[B, T, H, V]``) on the value side.
    ``b = w = beta`` recovers KDA. Implemented from the paper's equations.
    """
    return _recurrence(
        q, k, v, g, None, gate="vector", delta=True, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, segment_ids=segment_ids,
        erase=b, write=w,
    )


class Mamba3State(NamedTuple):
    """Recurrent state of the Mamba-3 SISO recurrence."""

    ssm: Array  # [B, H, K, V] float32
    key: Array  # [B, H, K] the last token's key (for the trapezoidal term)
    value: Array  # [B, H, V] the last token's value


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
) -> tuple[Array, Mamba3State | None]:
    """Mamba-3 SISO exponential-trapezoidal recurrence (state-spaces/mamba ``mamba3_siso``).

    ``S_t = exp(g_t) S_{t-1} + beta_t k_{t-1}^T v_{t-1} + gamma_t k_t^T v_t`` and
    ``o_t = scale * q_t S_t``. In Mamba-3 terms ``q``/``k`` are the rotated,
    biased ``C``/``B``, ``v`` is ``x``, ``g = A dt``, ``beta = (1 - lambda) dt exp(g)``
    and ``gamma = lambda dt``. The rotation, the ``D`` skip and the output gate
    belong to the layer. The previous-token term never crosses a segment boundary.
    """
    for name, x in (("g", g), ("beta", beta), ("gamma", gamma)):
        if x.shape != q.shape[:3]:
            raise ValueError(f"{name} must be [B, T, H] = {q.shape[:3]}; got {x.shape}")
    if k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"k must match q {q.shape} and v must be [B, T, H, V]; got {k.shape}, {v.shape}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    f32 = jnp.float32
    if initial_state is None:
        initial_state = Mamba3State(
            jnp.zeros((batch, heads, key_dim, value_dim), f32),
            jnp.zeros((batch, heads, key_dim), f32),
            jnp.zeros((batch, heads, value_dim), f32),
        )
    resets = None if segment_ids is None else segment_resets(segment_ids)

    def step(carry, token):
        state, k_prev, v_prev = carry
        q_t, k_t, v_t, g_t, beta_t, gamma_t, reset_t = token
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
            k_prev = jnp.where(reset_t, 0.0, k_prev)
            v_prev = jnp.where(reset_t, 0.0, v_prev)
        state = (
            state * jnp.exp(g_t)[:, None, None]
            + beta_t[:, None, None] * k_prev[:, :, None] * v_prev[:, None, :]
            + gamma_t[:, None, None] * k_t[:, :, None] * v_t[:, None, :]
        )
        o_t = jnp.einsum("hk,hkv->hv", q_t, state, precision=_HIGHEST)
        return (state, k_t, v_t), o_t

    def run(q, k, v, g, beta, gamma, resets, state0):
        carry, o = lax.scan(step, state0, (q, k, v, g, beta, gamma, resets))
        return o, carry

    inputs = [x.astype(f32) for x in (q * scale, k, v, g, beta, gamma)]
    state0 = tuple(x.astype(f32) for x in initial_state)
    o, (ssm, key, value) = jax.vmap(run)(*inputs, resets, state0)
    final = Mamba3State(ssm, key, value) if output_final_state else None
    return o.astype(q.dtype), final


# mLSTM (xLSTM): implemented from arXiv 2405.04517 (Eq. 19-27) and arXiv 2503.13427 (equations only).
MLSTM_EMPTY_M = -1e30  # stabiliser of the empty state: every real contribution outweighs it


class MLSTMState(NamedTuple):
    """mLSTM state in max-stabilised form; the true memory is ``exp(m) c`` and ``exp(m) n``."""

    c: Array  # [B, H, K, V] matrix memory
    n: Array  # [B, H, K] normaliser
    m: Array  # [B, H] log-space stabiliser


def mlstm_empty_state(batch: int, heads: int, head_k_dim: int, head_v_dim: int) -> MLSTMState:
    """The zero memory with the empty-state stabiliser (what a fresh sequence starts from)."""
    return MLSTMState(
        jnp.zeros((batch, heads, head_k_dim, head_v_dim), jnp.float32),
        jnp.zeros((batch, heads, head_k_dim), jnp.float32),
        jnp.full((batch, heads), MLSTM_EMPTY_M, jnp.float32),
    )


def mlstm_floor(m: Array) -> Array:
    """The normaliser floor ``exp(-m)``; the exponent is capped at 80 so ``m < -88`` cannot give inf (NaN grads)."""
    return jnp.exp(jnp.minimum(-m, 80.0))


def check_mlstm_inputs(q, k, v, g, i, initial_state, cu_seqlens, segment_ids) -> None:
    if cu_seqlens is not None:
        raise ValueError("mlstm supports segment_ids for packed documents, not cu_seqlens")
    batch, seq_len, heads, key_dim = q.shape
    if k.shape != q.shape or v.shape[:3] != (batch, seq_len, heads):
        raise ValueError(f"mlstm needs q, k [B, T, H, K] and v [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}")
    for name, gate in (("g", g), ("i", i)):
        if gate.shape != (batch, seq_len, heads):
            raise ValueError(f"mlstm gate {name} must be [B, T, H] = {(batch, seq_len, heads)}; got {gate.shape}")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
    if initial_state is not None:
        want = ((batch, heads, key_dim, v.shape[-1]), (batch, heads, key_dim), (batch, heads))
        got = tuple(tuple(x.shape) for x in initial_state)
        if len(initial_state) != 3 or got != want:
            raise ValueError(f"mlstm initial_state must be MLSTMState with shapes {want}; got {got}")


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
) -> tuple[Array, MLSTMState | None]:
    """xLSTM mLSTM cell with an exponential input gate and a max-stabiliser.

    ``g [B, T, H]`` is the log forget gate (``logsigmoid`` of the forget
    pre-activation, ``<= 0``) and ``i [B, T, H]`` the log input gate (the input
    pre-activation, the gate itself is ``exp(i)``). With ``q`` scaled by ``scale``
    (default ``K ** -0.5``)::

        m_t = max(g_t + m_{t-1}, i_t)
        c_t = exp(g_t + m_{t-1} - m_t) c_{t-1} + exp(i_t - m_t) k_t^T v_t
        n_t = exp(g_t + m_{t-1} - m_t) n_{t-1} + exp(i_t - m_t) k_t
        o_t = q_t c_t / (max(|q_t n_t|, exp(-m_t)) + eps)

    which with ``eps = 0`` equals the unstabilised ``q C / max(|q N|, 1)`` for the
    true state ``C = exp(m) c``. A sequence (and every segment of ``segment_ids``)
    starts from :func:`mlstm_empty_state`.
    """
    check_mlstm_inputs(q, k, v, g, i, initial_state, cu_seqlens, segment_ids)
    batch, seq_len, heads, key_dim = q.shape
    scale = key_dim**-0.5 if scale is None else scale
    f32 = jnp.float32
    state0 = (
        mlstm_empty_state(batch, heads, key_dim, v.shape[-1])
        if initial_state is None
        else MLSTMState(*(x.astype(f32) for x in initial_state))
    )
    resets = jnp.zeros((batch, seq_len), bool) if segment_ids is None else segment_resets(segment_ids)

    def step(carry, xs):
        c, n, m = carry
        q_t, k_t, v_t, g_t, i_t, reset = xs  # [B, H, K], [B, H, K], [B, H, V], [B, H], [B, H], [B]
        r = reset[:, None]
        c = jnp.where(r[..., None, None], 0.0, c)
        n = jnp.where(r[..., None], 0.0, n)
        m = jnp.where(r, MLSTM_EMPTY_M, m)
        m_new = jnp.maximum(g_t + m, i_t)
        forget = jnp.exp(g_t + m - m_new)
        write = jnp.exp(i_t - m_new)
        c = forget[..., None, None] * c + write[..., None, None] * k_t[..., :, None] * v_t[..., None, :]
        n = forget[..., None] * n + write[..., None] * k_t
        num = jnp.einsum("bhk,bhkv->bhv", q_t, c, precision=_HIGHEST)
        den = jnp.sum(q_t * n, axis=-1)
        o_t = num / (jnp.maximum(jnp.abs(den), mlstm_floor(m_new)) + eps)[..., None]
        return MLSTMState(c, n, m_new), o_t

    xs = tuple(jnp.moveaxis(x.astype(f32), 1, 0) for x in (q.astype(f32) * scale, k, v, g, i)) + (resets.T,)
    final, o = lax.scan(step, state0, xs)
    o = jnp.moveaxis(o, 0, 1).astype(q.dtype)
    return o, (final if output_final_state else None)


__all__ += ["MLSTM_EMPTY_M", "MLSTMState", "mlstm", "mlstm_empty_state"]


# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/ops/selective_scan_interface.py (selective_scan_ref)
def mamba1_inputs(
    q: Array,
    k: Array,
    v: Array,
    dt: Array,
    a: Array,
    *,
    dt_bias: Array | None,
    dt_softplus: bool,
    scale: float | None,
    cu_seqlens: Array | None,
    segment_ids: Array | None,
    initial_state: Array | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Checks a ``mamba1`` call; returns float32 ``(u, delta, a, b, c)`` with ``b, c [B, T, 1 or D, N]``.

    ``delta = softplus(dt + dt_bias)`` (as ``selective_scan_ref`` with ``delta_softplus``).
    """
    if cu_seqlens is not None:
        raise ValueError("mamba1 takes segment_ids for packed documents, not cu_seqlens")
    if scale is not None and scale != 1.0:
        raise ValueError("mamba1 has no output scale (C is used as is)")
    if v.ndim != 3:
        raise ValueError(f"mamba1 expects v (u) [B, T, D]; got {v.shape}")
    batch, seq_len, dim = v.shape
    if dt.shape != v.shape:
        raise ValueError(f"dt must be {v.shape}; got {dt.shape}")
    if a.ndim != 2 or a.shape[0] != dim:
        raise ValueError(f"a must be [D={dim}, N]; got {a.shape}")
    dstate = a.shape[1]

    def groups(x: Array, name: str) -> Array:
        if x.ndim == 3 and x.shape == (batch, seq_len, dstate):
            return x.astype(jnp.float32)[:, :, None, :]
        if x.ndim == 4 and x.shape[:2] == (batch, seq_len) and x.shape[3] == dstate and dim % x.shape[2] == 0:
            return jnp.repeat(x.astype(jnp.float32), dim // x.shape[2], axis=2)
        raise ValueError(f"{name} must be [B, T, N] or [B, T, G, N] with D % G == 0; got {x.shape}")

    b, c = groups(k, "k (B)"), groups(q, "q (C)")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be {(batch, seq_len)}; got {segment_ids.shape}")
    if initial_state is not None and initial_state.shape != (batch, dim, dstate):
        raise ValueError(f"initial_state must be {(batch, dim, dstate)}; got {initial_state.shape}")
    delta = dt.astype(jnp.float32)
    if dt_bias is not None:
        delta = delta + dt_bias.astype(jnp.float32)
    if dt_softplus:
        delta = jax.nn.softplus(delta)
    return v.astype(jnp.float32), delta, a.astype(jnp.float32), b, c


def mamba1_output(y: Array, u: Array, d: Array | None, z: Array | None, dtype: jnp.dtype) -> Array:
    """``(y + D u) * silu(z)`` in float32, cast to ``dtype`` (``selective_scan_ref``'s epilogue)."""
    if d is not None:
        y = y + u * d.astype(jnp.float32)
    if z is not None:
        y = y * jax.nn.silu(z.astype(jnp.float32))
    return y.astype(dtype)


def mamba1(
    q: Array,
    k: Array,
    v: Array,
    dt: Array,
    a: Array,
    *,
    d: Array | None = None,
    z: Array | None = None,
    dt_bias: Array | None = None,
    dt_softplus: bool = False,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
) -> tuple[Array, Array | None]:
    """Mamba-1 selective scan, token by token: the semantics of ``selective_scan_ref``.

    FLA layout: ``q = C``, ``k = B`` (``[B, T, N]``, or ``[B, T, G, N]`` with channel
    ``i`` reading group ``i // (D / G)``), ``v = u [B, T, D]``, ``dt [B, T, D]``,
    ``a [D, N]`` (negative), ``d [D]``, ``z [B, T, D]``, ``dt_bias [D]``. With
    ``delta = softplus(dt + dt_bias)`` (or ``dt + dt_bias`` without ``dt_softplus``):
    ``h_t = exp(delta_t A) h_{t-1} + delta_t u_t B_t``, ``y_t = h_t C_t + D u_t``,
    ``o_t = y_t silu(z_t)``. The state is ``[B, D, N]`` float32; ``segment_ids``
    zero it at every document start. Returns ``o`` in ``v.dtype``.
    """
    u, delta, a32, b, c = mamba1_inputs(
        q, k, v, dt, a, dt_bias=dt_bias, dt_softplus=dt_softplus, scale=scale, cu_seqlens=cu_seqlens,
        segment_ids=segment_ids, initial_state=initial_state,
    )
    batch, seq_len, dim = u.shape
    state0 = (jnp.zeros((batch, dim, a32.shape[1]), jnp.float32) if initial_state is None
              else initial_state.astype(jnp.float32))
    resets = jnp.zeros((batch, seq_len), bool) if segment_ids is None else segment_resets(segment_ids)

    def step(h, token):
        u_t, delta_t, b_t, c_t, reset_t = token  # [B, D], [B, D], [B, 1|D, N], [B, 1|D, N], [B]
        h = jnp.where(reset_t[:, None, None], 0.0, h)
        h = jnp.exp(delta_t[..., None] * a32) * h + (delta_t * u_t)[..., None] * b_t
        return h, jnp.sum(h * c_t, axis=-1)

    tokens = tuple(jnp.moveaxis(x, 1, 0) for x in (u, delta, b, c, resets))
    final, y = lax.scan(step, state0, tokens)
    o = mamba1_output(jnp.moveaxis(y, 0, 1), u, d, z, v.dtype)
    return o, final if output_final_state else None


# Gated DeltaProduct, adapted from flash-linear-attention (MIT),
# fla/ops/gated_delta_product/naive.py.


def householder_steps(q: Array, k: Array, v: Array, g: Array, beta: Array, num_householder: int | None) -> int:
    """Validate the Gated DeltaProduct time layout and return ``n_h``.

    ``q [B, T, Hk, K]`` and ``g [B, T, H]`` have one entry per token; ``k``,
    ``v`` and ``beta`` have ``n_h`` consecutive sub-steps per token (FLA layout
    ``[B, T * n_h, ...]``). ``num_householder=None`` infers ``n_h``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q, k, v must be 4-D; got {q.shape}, {k.shape}, {v.shape}")
    batch, seq_len = q.shape[:2]
    if num_householder is None:
        if seq_len == 0 or k.shape[1] % seq_len:
            raise ValueError(f"cannot infer num_householder from q {q.shape} and k {k.shape}")
        num_householder = k.shape[1] // seq_len
    if not isinstance(num_householder, int) or num_householder < 1:
        raise ValueError(f"num_householder must be a positive int; got {num_householder!r}")
    steps = seq_len * num_householder
    if k.shape != (batch, steps, *q.shape[2:]):
        raise ValueError(f"k must be [B, T * n_h, Hk, K] = {(batch, steps, *q.shape[2:])}; got {k.shape}")
    if v.shape[:2] != (batch, steps):
        raise ValueError(f"v must be [B, T * n_h, H, V] with T * n_h = {steps}; got {v.shape}")
    heads = v.shape[2]
    if beta.shape != (batch, steps, heads):
        raise ValueError(f"beta must be [B, T * n_h, H] = {(batch, steps, heads)}; got {beta.shape}")
    if g.shape != (batch, seq_len, heads):
        raise ValueError(f"g must be [B, T, H] = {(batch, seq_len, heads)}; got {g.shape}")
    return num_householder


def gated_delta_product(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    *,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    cu_seqlens: Array | None = None,
    segment_ids: Array | None = None,
    use_qk_l2norm: bool = False,
    num_householder: int | None = None,
) -> tuple[Array, Array | None]:
    """Gated DeltaProduct (FLA ``chunk_gated_delta_product``): ``n_h`` delta steps per token.

    ``q [B, T, H, K]``, ``g [B, T, H]``; ``k [B, T*n_h, H, K]``, ``v [B, T*n_h, H, V]``
    and ``beta [B, T*n_h, H]`` hold each token's ``n_h`` sub-steps consecutively:

        S = exp(g_t) S_{t-1};  for j < n_h: S = S + beta_tj k_tj^T (v_tj - k_tj S);  S_t = S
        o_t = scale * q_t S_t

    ``n_h = 1`` is :func:`gated_delta_rule`. ``segment_ids`` is ``[B, T]``;
    ``cu_seqlens`` is not supported (pass ``segment_ids``).
    """
    if cu_seqlens is not None:
        raise NotImplementedError("gated_delta_product does not support cu_seqlens; pass segment_ids")
    steps = householder_steps(q, k, v, g, beta, num_householder)
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"q/k heads {k.shape[2]} must equal v heads {v.shape[2]}")
    if segment_ids is not None and segment_ids.shape != (batch, seq_len):
        raise ValueError(f"segment_ids must be [B, T] = {(batch, seq_len)}; got {segment_ids.shape}")
    out_dtype = q.dtype
    f32 = jnp.float32
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = key_dim**-0.5
    q = q.astype(f32) * scale
    k = k.astype(f32).reshape(batch, seq_len, steps, heads, key_dim)
    v = v.astype(f32).reshape(batch, seq_len, steps, heads, value_dim)
    beta = beta.astype(f32).reshape(batch, seq_len, steps, heads)
    g = g.astype(f32)

    def step(state: Array, token) -> tuple[Array, Array]:
        # q [H, K]; k [n, H, K]; v [n, H, V]; g [H]; beta [n, H]; reset_t a scalar bool or None.
        q_t, k_t, v_t, g_t, beta_t, reset_t = token
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
        state = state * jnp.exp(g_t)[:, None, None]
        for j in range(steps):
            recalled = jnp.einsum("hk,hkv->hv", k_t[j], state, precision=_HIGHEST)
            update = beta_t[j][:, None] * (v_t[j] - recalled)
            state = state + k_t[j][:, :, None] * update[:, None, :]
        return state, jnp.einsum("hk,hkv->hv", q_t, state, precision=_HIGHEST)

    if initial_state is None:
        initial_state = jnp.zeros((batch, heads, key_dim, value_dim), f32)
    resets = None if segment_ids is None else segment_resets(segment_ids)

    def run(q, k, v, g, beta, resets, state0):
        final, o = lax.scan(step, state0, (q, k, v, g, beta, resets))
        return o, final

    o, final = jax.vmap(run)(q, k, v, g, beta, resets, initial_state.astype(f32))
    return o.astype(out_dtype), (final if output_final_state else None)


# Adapted from state-spaces/mamba (Apache-2.0), tests/ops/tilelang/test_mamba3_mimo.py (mamba3_MIMO_step_ref)
def mamba3_mimo(
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
) -> tuple[Array, Mamba3State | None]:
    """Mamba-3 MIMO exponential-trapezoidal recurrence of rank ``R`` (state-spaces/mamba ``mamba3_mimo``).

    ``q, k [B, T, H, R, K]`` (the rotated, biased C and B of every rank), ``v [B, T, H, R, V]``
    (x up-projected to every rank), ``g, beta, gamma [B, T, H]`` as in :func:`mamba3`:
    ``S_t = exp(g_t) S_{t-1} + beta_t sum_r k_{t-1,r}^T v_{t-1,r} + gamma_t sum_r k_{t,r}^T v_{t,r}``
    and ``o_{t,r} = scale * q_{t,r} S_t`` (``o [B, T, H, R, V]``). The state is a
    :class:`Mamba3State` whose ``key [B, H, R, K]`` and ``value [B, H, R, V]`` hold the last
    token's ranks. ``R = 1`` is :func:`mamba3`.
    """
    if q.ndim != 5 or k.shape != q.shape or v.ndim != 5 or v.shape[:4] != q.shape[:4]:
        raise ValueError(f"mamba3_mimo needs q, k [B, T, H, R, K] and v [B, T, H, R, V]; got {q.shape}, "
                         f"{k.shape}, {v.shape}")
    for name, x in (("g", g), ("beta", beta), ("gamma", gamma)):
        if x.shape != q.shape[:3]:
            raise ValueError(f"{name} must be [B, T, H] = {q.shape[:3]}; got {x.shape}")
    if segment_ids is not None and segment_ids.shape != q.shape[:2]:
        raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
    batch, _, heads, rank, key_dim = q.shape
    value_dim = v.shape[-1]
    f32 = jnp.float32
    if initial_state is None:
        initial_state = Mamba3State(
            jnp.zeros((batch, heads, key_dim, value_dim), f32),
            jnp.zeros((batch, heads, rank, key_dim), f32),
            jnp.zeros((batch, heads, rank, value_dim), f32),
        )
    resets = None if segment_ids is None else segment_resets(segment_ids)

    def step(carry, token):
        state, k_prev, v_prev = carry
        q_t, k_t, v_t, g_t, beta_t, gamma_t, reset_t = token
        if reset_t is not None:
            state = jnp.where(reset_t, 0.0, state)
            k_prev = jnp.where(reset_t, 0.0, k_prev)
            v_prev = jnp.where(reset_t, 0.0, v_prev)
        state = (
            state * jnp.exp(g_t)[:, None, None]
            + jnp.sum(beta_t[:, None, None, None] * k_prev[:, :, :, None] * v_prev[:, :, None, :], axis=1)
            + jnp.sum(gamma_t[:, None, None, None] * k_t[:, :, :, None] * v_t[:, :, None, :], axis=1)
        )
        o_t = jnp.einsum("hrk,hkv->hrv", q_t, state, precision=_HIGHEST)
        return (state, k_t, v_t), o_t

    def run(q, k, v, g, beta, gamma, resets, state0):
        carry, o = lax.scan(step, state0, (q, k, v, g, beta, gamma, resets))
        return o, carry

    inputs = [x.astype(f32) for x in (q * scale, k, v, g, beta, gamma)]
    state0 = tuple(x.astype(f32) for x in initial_state)
    o, (ssm, key, value) = jax.vmap(run)(*inputs, resets, state0)
    final = Mamba3State(ssm, key, value) if output_final_state else None
    return o.astype(q.dtype), final


from jla.ops.log_linear import log_linear_simple_gla_reference as log_linear_simple_gla  # noqa: E402
