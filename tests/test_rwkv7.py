# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-7 / DPLR: reference vs an independent NumPy oracle, chunked vs reference, the layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.rwkv7 import RWKV7Block
from jla.ops import linear_attention
from jla.ops.rwkv7 import rwkv7, rwkv7_chunked

jax.config.update("jax_enable_x64", False)


def _naive(r, k, v, w, a, b, state0=None, segment_ids=None):
    """RWKV's own [V, K] form, per token in float64: S = S (diag(exp w) + a b^T) + v k^T; o = S r."""
    r, k, v, w, a, b = (np.asarray(x, np.float64) for x in (r, k, v, w, a, b))
    batch, seq_len, heads, key_dim = r.shape
    value_dim = v.shape[-1]
    o = np.zeros((batch, seq_len, heads, value_dim))
    finals = np.zeros((batch, heads, key_dim, value_dim))
    for bi in range(batch):
        for h in range(heads):
            S = np.zeros((value_dim, key_dim)) if state0 is None else np.asarray(state0, np.float64)[bi, h].T.copy()
            for t in range(seq_len):
                if segment_ids is not None and t > 0 and segment_ids[bi, t] != segment_ids[bi, t - 1]:
                    S = np.zeros((value_dim, key_dim))
                transition = np.diag(np.exp(w[bi, t, h])) + np.outer(a[bi, t, h], b[bi, t, h])
                S = S @ transition + np.outer(v[bi, t, h], k[bi, t, h])
                o[bi, t, h] = S @ r[bi, t, h]
            finals[bi, h] = S.T
    return o, finals


def _inputs(seed, batch=2, seq_len=37, heads=2, key_dim=16, value_dim=8, strong_decay=False, dtype=jnp.float32):
    keys = jax.random.split(jax.random.PRNGKey(seed), 7)
    shape = (batch, seq_len, heads, key_dim)
    r = jax.random.normal(keys[0], shape)
    k = 0.5 * jax.random.normal(keys[1], shape)
    v = jax.random.normal(keys[2], (batch, seq_len, heads, value_dim))
    if strong_decay:
        w = -jnp.exp(1.5 * jax.random.normal(keys[3], shape))  # some channels decay by e^-20 per token
    else:
        w = -0.6065306597126334 * jax.nn.sigmoid(2.0 * jax.random.normal(keys[3], shape))
    kk = jax.random.normal(keys[4], shape)
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    gate = jax.nn.sigmoid(jax.random.normal(keys[5], shape))
    state0 = 0.3 * jax.random.normal(keys[6], (batch, heads, key_dim, value_dim))
    cast = lambda x: x.astype(dtype)  # noqa: E731
    return tuple(map(cast, (r, k, v, w, -kk, kk * gate))), state0


@pytest.mark.parametrize("strong_decay", [False, True])
def test_reference_matches_naive(strong_decay):
    (r, k, v, w, a, b), s0 = _inputs(0, strong_decay=strong_decay)
    o, final = rwkv7(r, k, v, w, a, b, initial_state=s0, output_final_state=True)
    o_ref, final_ref = _naive(r, k, v, w, a, b, s0)
    np.testing.assert_allclose(np.asarray(o), o_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(final), final_ref, rtol=1e-4, atol=1e-4)


def test_reference_scale_and_no_state():
    (r, k, v, w, a, b), _ = _inputs(1, seq_len=9)
    o, final = rwkv7(r, k, v, w, a, b, scale=0.25)
    assert final is None
    o_ref, _ = _naive(0.25 * np.asarray(r), k, v, w, a, b)
    np.testing.assert_allclose(np.asarray(o), o_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [4, 16, 32])
@pytest.mark.parametrize("strong_decay", [False, True])
def test_chunked_matches_reference_fp32(chunk_size, strong_decay):
    (r, k, v, w, a, b), s0 = _inputs(2, seq_len=45, strong_decay=strong_decay)
    o_ref, f_ref = rwkv7(r, k, v, w, a, b, initial_state=s0, output_final_state=True)
    o, f = rwkv7_chunked(r, k, v, w, a, b, initial_state=s0, output_final_state=True, chunk_size=chunk_size)
    np.testing.assert_allclose(np.asarray(o), np.asarray(o_ref), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(f), np.asarray(f_ref), rtol=2e-4, atol=2e-4)


def test_chunked_matches_reference_bf16():
    (r, k, v, w, a, b), s0 = _inputs(3, seq_len=64, key_dim=32, value_dim=32, dtype=jnp.bfloat16)
    o_ref, f_ref = rwkv7(r, k, v, w, a, b, initial_state=s0, output_final_state=True)
    o, f = rwkv7_chunked(r, k, v, w, a, b, initial_state=s0, output_final_state=True, chunk_size=16)
    assert o.dtype == jnp.bfloat16 and f.dtype == jnp.float32
    o_naive, _ = _naive(r, k, v, w, a, b, s0)
    scale = np.abs(o_naive).max()
    assert np.abs(np.asarray(o, np.float32) - np.asarray(o_ref, np.float32)).max() < 2e-2 * scale
    assert np.abs(np.asarray(o, np.float32) - o_naive).max() < 2e-2 * scale
    np.testing.assert_allclose(np.asarray(f), np.asarray(f_ref), rtol=1e-2, atol=1e-2)


def test_segment_resets_match_naive_and_chunked():
    (r, k, v, w, a, b), s0 = _inputs(4, seq_len=40)
    seg = np.zeros((2, 40), np.int32)
    seg[0, 7:] = 1
    seg[0, 16:] = 2  # boundary exactly on a chunk edge for chunk_size 16 / 8
    seg[0, 17:] = 3  # a one-token document
    seg[1, 30:] = 5
    seg = jnp.asarray(seg)
    o_ref, f_ref = rwkv7(r, k, v, w, a, b, initial_state=s0, output_final_state=True, segment_ids=seg)
    o_naive, f_naive = _naive(r, k, v, w, a, b, s0, np.asarray(seg))
    np.testing.assert_allclose(np.asarray(o_ref), o_naive, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(f_ref), f_naive, rtol=1e-4, atol=1e-4)
    for chunk_size in (8, 16):
        o, f = rwkv7_chunked(r, k, v, w, a, b, initial_state=s0, output_final_state=True,
                             segment_ids=seg, chunk_size=chunk_size)
        np.testing.assert_allclose(np.asarray(o), o_naive, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(np.asarray(f), f_naive, rtol=2e-4, atol=2e-4)


def test_dispatcher_and_gradients():
    (r, k, v, w, a, b), s0 = _inputs(5, seq_len=24, heads=1)

    def loss(impl, r, k, v, w, a, b, s0):
        o, f = linear_attention("rwkv7", r, k, v, w=w, a=a, b=b, impl=impl, initial_state=s0,
                                output_final_state=True, **({"chunk_size": 8} if impl == "chunked" else {}))
        return jnp.sum(jnp.sin(o)) + jnp.sum(f**2)

    args = (r, k, v, w, a, b, s0)
    np.testing.assert_allclose(loss("chunked", *args), loss("reference", *args), rtol=1e-4)
    g_ref = jax.grad(lambda *x: loss("reference", *x), argnums=tuple(range(7)))(*args)
    g_chk = jax.grad(lambda *x: loss("chunked", *x), argnums=tuple(range(7)))(*args)
    for x, y in zip(g_chk, g_ref):
        assert np.all(np.isfinite(np.asarray(x)))
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-3, atol=2e-3)
    with pytest.raises(ValueError):
        linear_attention("rwkv7", r, k, v, w=w, a=a, impl="reference")
    with pytest.raises(ValueError):
        rwkv7(r, k, v, w, a, b, cu_seqlens=jnp.array([0, 24]))


def _blocks(impl, chunk_size=8):
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    return [
        RWKV7Block(64, 128, head_dim=16, layer_idx=i, n_layers=2, key=keys[i], lora_dims=(16, 16, 16, 16),
                   chunk_size=chunk_size, impl=impl, compute_dtype="float32")
        for i in range(2)
    ]


def _perturb(blocks):
    # Random (non-zero) parameters everywhere so every path matters, including o_proj.
    leaves, treedef = jax.tree_util.tree_flatten(blocks)
    keys = jax.random.split(jax.random.PRNGKey(11), len(leaves))
    leaves = [x + 0.05 * jax.random.normal(kk, x.shape, x.dtype) for x, kk in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _forward(blocks, x, segment_ids=None):
    v_first = None
    for blk in blocks:
        x, v_first, _ = blk(x, v_first, segment_ids)
    return x


def test_layer_decode_matches_full_forward_and_impls_agree():
    blocks = _perturb(_blocks("chunked"))
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 19, 64))
    with jax.default_matmul_precision("highest"):  # GPU float32 matmuls default to TF32
        full = _forward(blocks, x)
        ref_blocks = _perturb(_blocks("reference"))  # same keys, same parameters
        np.testing.assert_allclose(np.asarray(_forward(ref_blocks, x)), np.asarray(full), rtol=2e-4, atol=2e-4)
        caches = [b.init_cache(2, jnp.float32) for b in blocks]
        outs = []
        for t in range(x.shape[1]):
            h, v_first = x[:, t], None
            for i, blk in enumerate(ref_blocks):
                h, v_first, caches[i] = blk.step(h, v_first, caches[i])
            outs.append(h)
    np.testing.assert_allclose(np.asarray(jnp.stack(outs, 1)), np.asarray(full), rtol=2e-4, atol=2e-4)


def test_one_token_call_with_segment_ids():
    (r, k, v, w, a, b), s0 = _inputs(6, seq_len=1)
    seg = jnp.zeros((2, 1), jnp.int32)
    o_naive, f_naive = _naive(r, k, v, w, a, b, s0)
    for fn in (rwkv7, rwkv7_chunked):
        o, f = fn(r, k, v, w, a, b, initial_state=s0, output_final_state=True, segment_ids=seg)
        np.testing.assert_allclose(np.asarray(o), o_naive, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(f), f_naive, rtol=1e-4, atol=1e-4)
    with jax.default_matmul_precision("highest"):
        blocks = _blocks("chunked")
        x = jax.random.normal(jax.random.PRNGKey(7), (2, 1, 64))
        np.testing.assert_allclose(np.asarray(_forward(blocks, x, seg)), np.asarray(_forward(blocks, x)),
                                   rtol=1e-5, atol=1e-5)


def test_layer_segments_match_separate_documents():
    blocks = _perturb(_blocks("chunked"))
    x = jax.random.normal(jax.random.PRNGKey(4), (1, 21, 64))
    seg = jnp.asarray(np.r_[np.zeros(9), np.ones(12)].astype(np.int32))[None]
    with jax.default_matmul_precision("highest"):
        packed = _forward(blocks, x, seg)
        first = _forward(blocks, x[:, :9])
        second = _forward(blocks, x[:, 9:])
    np.testing.assert_allclose(np.asarray(packed), np.asarray(jnp.concatenate([first, second], 1)),
                               rtol=2e-4, atol=2e-4)


def test_layer_train_step_cpu():
    blocks = _blocks("auto")
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 32, 64))

    def loss(blocks):
        return jnp.mean(jnp.square(_forward(blocks, x) - 1.0))

    value, grads = jax.value_and_grad(loss)(blocks)
    assert np.isfinite(float(value))
    flat = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in flat)
    assert sum(float(jnp.abs(g).sum()) for g in flat) > 0.0

