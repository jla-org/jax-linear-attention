# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from BlinkDL RWKV-LM (Apache-2.0), RWKV-v7/train_temp/src/model.py and
# RWKV-v7/train_temp/cuda/rwkv7_clampw.cu (the formulas, transcribed into NumPy).
"""RWKV-7 layers against the official formulas (BlinkDL RWKV-LM, Apache-2.0, RWKV-v7/train_temp).

A float64 NumPy transcription of ``RWKV_Tmix_x070`` / ``RWKV_CMix_x070`` (the "slow pytorch
version" comments of ``src/model.py``) and of the ``RWKV7_CLAMPW`` CUDA recurrence
(``decay = exp(-exp(-0.5) sigmoid(w))``, ``sa = S a``, ``S = S diag(decay) + sa b^T + v k^T``,
``y = S r``), run on our layers' weights. Packed documents are checked against running each
document alone, which is how the official code (no packing) treats separate sequences; that also
covers documents of one token (FLA's packed token-shift decode bug, fla-org/flash-linear-attention
#1194).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.rwkv7 import RWKV7Block, RWKV7ChannelMix, RWKV7TimeMix

_D, _HEAD, _LAYERS, _LORA, _FF = 128, 64, 3, (32, 32, 32, 32), 256


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _shift(x):  # nn.ZeroPad2d((0, 0, 1, -1)): x_{t-1}, zero at t = 0
    return np.concatenate([np.zeros_like(x[:, :1]), x[:, :-1]], axis=1)


def _np(x):
    return np.asarray(x, np.float64)


def official_time_mix(att: RWKV7TimeMix, x: np.ndarray, v_first: np.ndarray | None):
    B, T, C = x.shape
    H, N = att.num_heads, att.head_dim
    mu = _np(att.mu)
    xx = _shift(x) - x
    xr, xw, xk, xv, xa, xg = (x + xx * mu[i] for i in range(6))

    def lora(m, z, act):
        h = z @ _np(m.down.weight)
        h = {"tanh": np.tanh, "sigmoid": _sigmoid, "none": lambda u: u}[act](h)
        return h @ _np(m.up.weight) + (0.0 if m.up.bias is None else _np(m.up.bias))

    r = xr @ _np(att.r_proj.weight)
    w = lora(att.w_lora, xw, "tanh")  # w0 + tanh(xw @ w1) @ w2
    k = xk @ _np(att.k_proj.weight)
    v = xv @ _np(att.v_proj.weight)
    if att.layer_idx == 0:
        v_first = v
    else:
        v = v + (v_first - v) * _sigmoid(lora(att.v_lora, xv, "none"))
    a = _sigmoid(lora(att.a_lora, xa, "none"))
    g = lora(att.g_lora, xg, "sigmoid")  # sigmoid(xg @ g1) @ g2
    kk = (k * _np(att.k_k)).reshape(B, T, H, N)
    kk = (kk / np.maximum(np.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)).reshape(B, T, C)  # F.normalize
    k = k * (1 + (a - 1) * _np(att.k_a))
    decay = np.exp(-0.6065306597126334 * _sigmoid(w))  # the RWKV7_CLAMPW kernel's soft clamp
    aa, bb = -kk, kk * a
    y = np.zeros((B, T, H, N))
    hd = lambda z, t: z[:, t].reshape(B, H, N)  # noqa: E731
    for bi in range(B):
        for h in range(H):
            S = np.zeros((N, N))  # [V, K]
            for t in range(T):
                sa = S @ hd(aa, t)[bi, h]
                S = S * hd(decay, t)[bi, h][None, :] + np.outer(sa, hd(bb, t)[bi, h]) + np.outer(
                    hd(v, t)[bi, h], hd(k, t)[bi, h])
                y[bi, t, h] = S @ hd(r, t)[bi, h]
    # ln_x = nn.GroupNorm(H, C, eps=64e-5) on (B T, C); head size 64 here, so FLA's eps = head_dim * 1e-5 agrees
    mean = y.mean(-1, keepdims=True)
    var = y.var(-1, keepdims=True)
    y = ((y - mean) / np.sqrt(var + 64e-5)).reshape(B, T, C) * _np(att.norm_weight) + _np(att.norm_bias)
    bonus = (r.reshape(B, T, H, N) * k.reshape(B, T, H, N) * _np(att.r_k)).sum(-1, keepdims=True)
    y = y + (bonus * v.reshape(B, T, H, N)).reshape(B, T, C)
    return (y * g) @ _np(att.o_proj.weight), v_first


def official_channel_mix(ffn: RWKV7ChannelMix, x: np.ndarray) -> np.ndarray:
    k = x + (_shift(x) - x) * _np(ffn.x_k)
    return np.maximum(k @ _np(ffn.key.weight), 0.0) ** 2 @ _np(ffn.value.weight)


def _layers(layer_idx: int, seed: int = 0, impl: str = "auto"):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed + layer_idx), 4)
    common = dict(param_dtype=jnp.float32, compute_dtype="float32")
    att = RWKV7TimeMix(_D, head_dim=_HEAD, layer_idx=layer_idx, n_layers=_LAYERS, key=k1, lora_dims=_LORA,
                       chunk_size=8, impl=impl, **common)
    ffn = RWKV7ChannelMix(_D, _FF, layer_idx=layer_idx, n_layers=_LAYERS, key=k2, **common)

    def perturb(module, key):  # every float leaf moved off its initial value (zeros included)
        leaves, treedef = jax.tree_util.tree_flatten(module)
        keys = jax.random.split(key, len(leaves))
        leaves = [x + 0.1 * jax.random.normal(kk, x.shape, x.dtype) if isinstance(x, jax.Array) else x
                  for x, kk in zip(leaves, keys)]
        return jax.tree_util.tree_unflatten(treedef, leaves)

    return perturb(att, k3), perturb(ffn, k4)


@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("impl", ["auto", "reference"])
def test_time_and_channel_mix_match_official(layer_idx, impl):
    att, ffn = _layers(layer_idx, impl=impl)
    x = np.asarray(jax.random.normal(jax.random.PRNGKey(5), (2, 19, _D)), np.float64)
    v_first = None if layer_idx == 0 else np.asarray(jax.random.normal(jax.random.PRNGKey(6), x.shape), np.float64)
    with jax.default_matmul_precision("highest"):
        vf_in = None if v_first is None else jnp.asarray(v_first, jnp.float32)
        out, vf, last, _ = att(jnp.asarray(x, jnp.float32), vf_in)
        f_out, _ = ffn(jnp.asarray(x, jnp.float32))
    want, want_vf = official_time_mix(att, x, v_first)
    scale = np.abs(want).max()
    np.testing.assert_allclose(np.asarray(out, np.float64), want, atol=2e-4 * scale, rtol=0)
    np.testing.assert_allclose(np.asarray(vf, np.float64), want_vf, atol=1e-4 * np.abs(want_vf).max(), rtol=0)
    np.testing.assert_allclose(np.asarray(last), x[:, -1], rtol=1e-6)
    f_want = official_channel_mix(ffn, x)
    np.testing.assert_allclose(np.asarray(f_out, np.float64), f_want, atol=1e-4 * np.abs(f_want).max(), rtol=0)


@pytest.mark.parametrize("docs", [(5, 1, 9, 1, 4), (1,) * 12], ids=["mixed", "one_token_each"])
def test_packed_documents_match_separate_sequences(docs):
    """Packed rows (``segment_ids``) equal each document run alone through the same blocks."""
    blocks = []
    for i in range(2):
        att, ffn = _layers(i, seed=3)
        k = jax.random.PRNGKey(10 + i)
        block = RWKV7Block(_D, _FF, head_dim=_HEAD, layer_idx=i, n_layers=_LAYERS, key=k, lora_dims=_LORA,
                           chunk_size=8, compute_dtype="float32")
        block = eqx.tree_at(lambda b: b.att, block, att)
        blocks.append(eqx.tree_at(lambda b: b.ffn, block, ffn))
    T = sum(docs)
    x = jax.random.normal(jax.random.PRNGKey(7), (1, T, _D))
    seg = jnp.asarray(np.concatenate([np.full(n, j) for j, n in enumerate(docs)]).astype(np.int32))[None]
    with jax.default_matmul_precision("highest"):
        h, vf = x, None
        for block in blocks:
            h, vf, _ = block(h, vf, seg)
        pieces, start = [], 0
        for n in docs:
            hd, vfd = x[:, start:start + n], None
            for block in blocks:
                hd, vfd, _ = block(hd, vfd)
            pieces.append(hd)
            start += n
    want = np.concatenate([np.asarray(p) for p in pieces], axis=1)
    np.testing.assert_allclose(np.asarray(h), want, atol=1e-4 * np.abs(want).max(), rtol=0)
