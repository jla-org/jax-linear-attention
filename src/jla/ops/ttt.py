# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/ttt/naive.py (dual-form layer-norm backward)
"""TTT-Linear: a linear inner model trained by mini-batch gradient descent at test time.

Sun et al. 2024, "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
(arXiv 2407.04620). Every head owns an inner model ``f(x) = x + LN(x W + b)`` with
``W [K, V]`` (``K == V``), ``b [V]`` and a per-head layer norm ``(ln_weight, ln_bias)``.
Token ``s`` defines the reconstruction loss

    l_s(W, b) = 0.5 * || LN(k_s W + b) - (v_s - k_s) ||^2

Tokens are grouped into mini-batches of ``mini_batch_size`` aligned to absolute positions
(``t // mini_batch_size``). All gradients of a mini-batch are taken at the weights
``(W0, b0)`` the mini-batch starts from, and token ``t`` reads the inner model with

    W_t = W0 - a_t * sum_{s <= t in the mini-batch} eta_s * dl_s/dW      (same for b)
    o_t = q_t + LN(q_t W_t + b_t)

``eta [B, T, H]`` is the per-token inner learning rate and ``a_t = token_scale[i_t]`` a
per-position factor (``1 / (i_t + 1)`` by default, a running mean of the mini-batch's
gradients, as in the paper); ``i_t`` is ``t``'s index in its mini-batch. The last token's
``W_t`` becomes the next mini-batch's ``W0``.

``w0 [H, K, V]`` and ``b0 [H, V]`` are the (learnable) initial inner weights. With
``segment_ids [B, T]`` the inner weights restart from ``(w0, b0)`` wherever the id changes;
a segment that starts inside a mini-batch sees only its own tokens' gradients, and ``i_t``
counts from the segment start. Mini-batches stay aligned to absolute positions, so a packed
document matches a separate run exactly when it starts at a multiple of ``mini_batch_size``.

Two implementations:

* :func:`ttt_linear_reference`: token-by-token ``lax.scan``; each token's gradient comes
  from ``jax.grad`` of its loss; carries a :class:`TTTState` and so supports
  ``initial_state`` / ``output_final_state`` (decode);
* :func:`ttt_linear_dual`: the dual form, one scan step per mini-batch with
  ``[b, b]`` matmuls and the closed-form layer-norm backward; training and prefill from
  ``(w0, b0)`` only.

Layout follows FLA: ``q, k [B, T, H, K]``, ``v [B, T, H, V]``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["IMPLS", "TTTState", "default_token_scale", "ttt_linear", "ttt_linear_dual", "ttt_linear_reference"]

Array = jax.Array
_HIGHEST = lax.Precision.HIGHEST


class TTTState(NamedTuple):
    """Inner-model state between calls (all ``[B, ...]``; float32 except the counters).

    ``w [B, H, K, V]`` and ``b [B, H, V]`` are the weights the current mini-batch started
    from; ``gw``/``gb`` the ``eta``-weighted gradient sums of its tokens seen so far;
    ``count [B]`` how many of them belong to the current segment and ``phase [B]`` the
    absolute position modulo ``mini_batch_size``.
    """

    w: Array
    b: Array
    gw: Array
    gb: Array
    count: Array
    phase: Array


def default_token_scale(mini_batch_size: int) -> Array:
    """``1 / (i + 1)`` for ``i < mini_batch_size``: the running mean of the gradients."""
    return 1.0 / jnp.arange(1, mini_batch_size + 1, dtype=jnp.float32)


def _layer_norm(x: Array, weight: Array, bias: Array, eps: float) -> Array:
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * lax.rsqrt(var + eps) * weight + bias


def _check(q, k, v, eta, ln_weight, ln_bias, w0, b0, mini_batch_size, token_scale, segment_ids) -> None:
    B, T, H, K = q.shape
    if k.shape != q.shape or v.shape[:3] != (B, T, H):
        raise ValueError(f"q/k must be [B, T, H, K] and v [B, T, H, V]; got {q.shape}, {k.shape}, {v.shape}")
    if v.shape[-1] != K:
        raise ValueError("TTT-Linear needs head_k_dim == head_v_dim (the inner model is residual)")
    if eta.shape != (B, T, H):
        raise ValueError(f"eta must be [B, T, H] = {(B, T, H)}; got {eta.shape}")
    if ln_weight.shape != (H, K) or ln_bias.shape != (H, K):
        raise ValueError(f"ln_weight/ln_bias must be [H, V] = {(H, K)}")
    if w0 is not None and w0.shape != (H, K, K):
        raise ValueError(f"w0 must be [H, K, V] = {(H, K, K)}; got {w0.shape}")
    if b0 is not None and b0.shape != (H, K):
        raise ValueError(f"b0 must be [H, V] = {(H, K)}; got {b0.shape}")
    if mini_batch_size < 1:
        raise ValueError("mini_batch_size must be >= 1")
    if token_scale is not None and token_scale.shape != (mini_batch_size,):
        raise ValueError(f"token_scale must be [mini_batch_size] = {(mini_batch_size,)}")
    if segment_ids is not None and segment_ids.shape != (B, T):
        raise ValueError(f"segment_ids must be [B, T] = {(B, T)}")


def _defaults(q, w0, b0, token_scale, mini_batch_size):
    H, K = q.shape[2], q.shape[3]
    w0 = jnp.zeros((H, K, K), jnp.float32) if w0 is None else w0.astype(jnp.float32)
    b0 = jnp.zeros((H, K), jnp.float32) if b0 is None else b0.astype(jnp.float32)
    scale = default_token_scale(mini_batch_size) if token_scale is None else token_scale.astype(jnp.float32)
    return w0, b0, scale


def _resets(segment_ids: Array | None, B: int, T: int) -> Array:
    if segment_ids is None:
        return jnp.zeros((B, T), bool)
    change = segment_ids[:, 1:] != segment_ids[:, :-1]
    return jnp.concatenate([jnp.zeros((B, 1), bool), change], axis=1)


def ttt_linear_reference(
    q: Array,
    k: Array,
    v: Array,
    eta: Array,
    *,
    ln_weight: Array,
    ln_bias: Array,
    w0: Array | None = None,
    b0: Array | None = None,
    mini_batch_size: int = 16,
    token_scale: Array | None = None,
    eps: float = 1e-6,
    initial_state: TTTState | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
) -> tuple[Array, TTTState | None]:
    """Token-by-token TTT-Linear; the oracle. Returns ``(o, final_state)``."""
    _check(q, k, v, eta, ln_weight, ln_bias, w0, b0, mini_batch_size, token_scale, segment_ids)
    B, T, H, K = q.shape
    w0, b0, scale = _defaults(q, w0, b0, token_scale, mini_batch_size)
    gamma = ln_weight.astype(jnp.float32)
    beta = ln_bias.astype(jnp.float32)
    if initial_state is None:
        zw = jnp.zeros((B, H, K, K), jnp.float32)
        zb = jnp.zeros((B, H, K), jnp.float32)
        zi = jnp.zeros((B,), jnp.int32)
        initial_state = TTTState(zw + w0, zb + b0, zw, zb, zi, zi)
    state = TTTState(*(x.astype(jnp.float32) for x in initial_state[:4]),
                     initial_state.count.astype(jnp.int32), initial_state.phase.astype(jnp.int32))
    resets = _resets(segment_ids, B, T)

    def loss(w, b, kt, vt):  # summed over rows and heads: per-(row, head) gradients are independent
        z = jnp.einsum("bhk,bhkv->bhv", kt, w, precision=_HIGHEST) + b
        return 0.5 * jnp.sum(jnp.square(_layer_norm(z, gamma, beta, eps) - (vt - kt)))

    grad = jax.grad(loss, argnums=(0, 1))

    def step(st: TTTState, xs):
        qt, kt, vt, et, rt = xs  # [B, H, K] x3, [B, H], [B]
        r4, r3 = rt[:, None, None, None], rt[:, None, None]
        w = jnp.where(r4, w0[None], st.w)
        b = jnp.where(r3, b0[None], st.b)
        gw = jnp.where(r4, 0.0, st.gw)
        gb = jnp.where(r3, 0.0, st.gb)
        count = jnp.where(rt, 0, st.count)
        dw, db = grad(w, b, kt, vt)
        gw = gw + et[..., None, None] * dw
        gb = gb + et[..., None] * db
        count = count + 1
        a = scale[jnp.clip(count - 1, 0, mini_batch_size - 1)][:, None]  # [B, 1]
        wt = w - a[..., None, None] * gw
        bt = b - a[..., None] * gb
        z = jnp.einsum("bhk,bhkv->bhv", qt, wt, precision=_HIGHEST) + bt
        ot = qt + _layer_norm(z, gamma, beta, eps)
        phase = st.phase + 1
        end = phase == mini_batch_size
        e4, e3 = end[:, None, None, None], end[:, None, None]
        new = TTTState(
            jnp.where(e4, wt, w), jnp.where(e3, bt, b), jnp.where(e4, 0.0, gw), jnp.where(e3, 0.0, gb),
            jnp.where(end, 0, count), jnp.where(end, 0, phase),
        )
        return new, ot

    f32 = lambda x: jnp.moveaxis(x.astype(jnp.float32), 1, 0)  # noqa: E731
    xs = (f32(q), f32(k), f32(v), f32(eta), jnp.moveaxis(resets, 1, 0))
    final, o = lax.scan(step, state, xs)
    o = jnp.moveaxis(o, 0, 1).astype(q.dtype)
    return o, (final if output_final_state else None)


def ttt_linear_dual(
    q: Array,
    k: Array,
    v: Array,
    eta: Array,
    *,
    ln_weight: Array,
    ln_bias: Array,
    w0: Array | None = None,
    b0: Array | None = None,
    mini_batch_size: int = 16,
    token_scale: Array | None = None,
    eps: float = 1e-6,
    initial_state: TTTState | None = None,
    output_final_state: bool = False,
    segment_ids: Array | None = None,
    precision: lax.Precision | None = _HIGHEST,
) -> tuple[Array, None]:
    """Dual (mini-batch parallel) TTT-Linear from ``(w0, b0)``; returns ``(o, None)``.

    The decode state lives in :func:`ttt_linear_reference`; this form raises on
    ``initial_state`` or ``output_final_state`` instead of ignoring them.
    """
    if initial_state is not None or output_final_state:
        raise ValueError("ttt_linear_dual starts from (w0, b0) and returns no state; use the reference "
                         "implementation for initial_state / output_final_state")
    _check(q, k, v, eta, ln_weight, ln_bias, w0, b0, mini_batch_size, token_scale, segment_ids)
    B, T, H, K = q.shape
    bs = mini_batch_size
    w0, b0, scale = _defaults(q, w0, b0, token_scale, bs)
    gamma = ln_weight.astype(jnp.float32)[None, :, None, :]  # [1, H, 1, V]
    beta = ln_bias.astype(jnp.float32)[None, :, None, :]
    nt = -(-T // bs)
    pad = nt * bs - T

    # Segment bookkeeping on absolute positions: doc index, index inside the mini-batch.
    resets = _resets(segment_ids, B, T)
    doc = jnp.cumsum(resets.astype(jnp.int32), axis=1)  # [B, T], monotone per row
    t = jnp.arange(T, dtype=jnp.int32)
    seg_start = lax.cummax(jnp.where(resets, t[None, :], 0), axis=1)
    idx = t[None, :] - jnp.maximum(seg_start, (t // bs * bs)[None, :])
    a = scale[jnp.clip(idx, 0, bs - 1)]  # [B, T]

    def blocks(x, fill=0):  # [B, T, ...] -> [NT, B, bs, ...] with padding
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), constant_values=fill)
        return jnp.moveaxis(x.reshape(B, nt, bs, *x.shape[2:]), 1, 0)

    def heads(x):  # [B, T, H, D] -> [NT, B, H, bs, D]
        return jnp.swapaxes(blocks(x.astype(jnp.float32)), 2, 3)

    # Padded tokens continue the last document and carry eta = 0 (no gradient).
    doc_b = jnp.moveaxis(jnp.pad(doc, [(0, 0), (0, pad)], mode="edge").reshape(B, nt, bs), 1, 0)
    xs =(heads(q), heads(k), heads(v), jnp.swapaxes(blocks(eta.astype(jnp.float32)), 2, 3), blocks(a), doc_b)
    w_init = jnp.broadcast_to(w0[None], (B, H, K, K))
    b_init = jnp.broadcast_to(b0[None, :, None, :], (B, H, 1, K))
    causal = jnp.tril(jnp.ones((bs, bs), bool))
    mm = lambda s, x, y: jnp.einsum(s, x, y, precision=precision)  # noqa: E731

    def step(carry, x):
        w, b, dcarry = carry  # [B, H, K, V], [B, H, 1, V], [B]
        qi, ki, vi, ei, ai, di = x  # [B, H, bs, K] x3, [B, H, bs], [B, bs], [B, bs]
        cont = (di == dcarry[:, None])[:, None, :, None]  # [B, 1, bs, 1]: continues the carried doc
        kw = jnp.where(cont, mm("bhsk,bhkv->bhsv", ki, w) + b, mm("bhsk,bhkv->bhsv", ki, w_init) + b_init)
        qw = jnp.where(cont, mm("bhsk,bhkv->bhsv", qi, w) + b, mm("bhsk,bhkv->bhsv", qi, w_init) + b_init)
        # Closed-form layer-norm backward of 0.5 * ||LN(z) - (v - k)||^2 with respect to z.
        mu = jnp.mean(kw, -1, keepdims=True)
        rstd = lax.rsqrt(jnp.mean(jnp.square(kw - mu), -1, keepdims=True) + eps)
        zhat = (kw - mu) * rstd
        dzhat = (gamma * zhat + beta - (vi - ki)) * gamma
        gz = rstd * (dzhat - jnp.mean(dzhat, -1, keepdims=True)
                     - zhat * jnp.mean(dzhat * zhat, -1, keepdims=True))  # [B, H, bs, V]
        same = (di[:, :, None] == di[:, None, :]) & causal[None]  # [B, bs, bs] (t, s)
        e = jnp.where(same[:, None], ai[:, None, :, None] * ei[:, :, None, :], 0.0)  # [B, H, t, s]
        qk = mm("bhtk,bhsk->bhts", qi, ki)
        z = qw - mm("bhts,bhsv->bhtv", e * qk, gz) - mm("bhts,bhsv->bhtv", e, gz)
        o = qi + _layer_norm(z, gamma, beta, eps)
        # The last token's weights start the next mini-batch.
        el = e[:, :, -1, :]  # [B, H, s]
        last_cont = cont[:, :, -1]  # [B, 1, 1]
        w_start = jnp.where(last_cont[..., None], w, w_init)
        b_start = jnp.where(last_cont[..., None], b, b_init)
        w_new = w_start - jnp.einsum("bhs,bhsk,bhsv->bhkv", el, ki, gz, precision=precision)
        b_new = b_start - jnp.einsum("bhs,bhsv->bhv", el, gz, precision=precision)[:, :, None, :]
        return (w_new, b_new, di[:, -1]), o

    _, o = lax.scan(step, (w_init, b_init, doc[:, 0]), xs)  # o [NT, B, H, bs, V]
    o = jnp.moveaxis(o, 0, 1)  # [B, NT, H, bs, V]
    o = jnp.swapaxes(o, 2, 3).reshape(B, nt * bs, H, K)[:, :T]
    return o.astype(q.dtype), None


IMPLS = {"reference": ttt_linear_reference, "dual": ttt_linear_dual}


def ttt_linear(q: Array, k: Array, v: Array, eta: Array, *, impl: str = "auto", **options) -> tuple[Array, TTTState | None]:
    """Dispatch TTT-Linear: ``"dual"`` for stateless calls, ``"reference"`` otherwise (``"auto"``)."""
    if impl == "auto":
        stateful = options.get("initial_state") is not None or options.get("output_final_state", False)
        impl = "reference" if stateful else "dual"
    if impl not in IMPLS:
        raise ValueError(f"unknown TTT-Linear implementation {impl!r}; expected 'auto' or one of {sorted(IMPLS)}")
    if impl == "reference":
        options.pop("precision", None)
    return IMPLS[impl](q, k, v, eta, **options)
