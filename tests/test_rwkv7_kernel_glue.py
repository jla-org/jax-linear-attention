# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The work around RWKV-7's 64-dim kernels that left XLA: the chunk decays, bf16 gates and gradients.

* ``rwkv7_prep.chunk_decays`` gives ``jnp.cumsum``'s bits for the chunk-local log decay ``G`` and
  the carry select of each chunk, from f32 or bf16 ``w``.
* bf16 ``w, a, b`` go into both kernels as bf16 tiles and their gradients come out as bf16: the
  same bits as f32 inputs of the same values with the f32 gradients rounded to bf16 afterwards; so do
  bf16 ``a, b`` with an f32 ``w`` (bf16 ``a, b`` tiles, f32 ``dw``).
* With packed documents both state passes are cut at document starts: the same results as one chain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops.mosaic_gpu import rwkv7 as mg
from jla.ops.mosaic_gpu import rwkv7_prep
from jla.ops.mosaic_gpu.vector_gate_ops import is_hopper

gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100 (compute capability 9.0)")
F32, BF16 = jnp.float32, jnp.bfloat16
BOUND = -0.6065306597126334


def _segment_ids(B, T, docs, seed):
    if not docs:
        return None
    rng = np.random.default_rng(seed)
    starts = np.zeros((B, T), np.int32)
    for b in range(B):
        starts[b, rng.choice(np.arange(1, T), docs - 1, replace=False)] = 1
    return jnp.asarray(np.cumsum(starts, axis=1))


@gpu
@pytest.mark.parametrize("dtype", [F32, BF16], ids=["f32", "bf16"])
@pytest.mark.parametrize("shape", [(2, 256, 2), (1, 1024, 12), (3, 192, 4)], ids=["b2", "h12", "b3"])
def test_chunk_decays_are_cumsum_bits(dtype, shape):
    B, T, H = shape
    K = rwkv7_prep.DIM
    w = (BOUND * jax.nn.sigmoid(jax.random.normal(jax.random.key(T + H), (B, T, H, K)))).astype(dtype)
    seg = _segment_ids(B, T, 5, H)
    flags = mg._prep_flags(seg.astype(jnp.int32), T // mg.CHUNK)
    g, g_carry = jax.jit(lambda w, f: rwkv7_prep.chunk_decays(w, f, no_carry=mg._NO_CARRY))(w, flags)

    def xla(w, flags):
        g = jnp.cumsum(w.astype(F32).reshape(B, T // mg.CHUNK, mg.CHUNK, H, K), axis=2)
        carry = flags[:, 0].reshape(B, T // mg.CHUNK, mg.CHUNK)[:, :, -1]
        return g.reshape(B, T, H, K), jnp.where(carry[:, :, None, None] > 0, g[:, :, -1], mg._NO_CARRY)

    g_want, carry_want = jax.jit(xla)(w, flags)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(g_want))
    np.testing.assert_array_equal(np.asarray(g_carry), np.asarray(carry_want))


@gpu
@pytest.mark.parametrize("case", [
    dict(shape=(2, 1000, 4), docs=0, h0=True),
    dict(shape=(1, 2048, 6), docs=9, h0=True),
    dict(shape=(2, 512, 3), docs=4, h0=False),  # H * K = 192: the XLA chunk decays
], ids=["b2_odd_t", "docs", "h3_docs"])
def test_bf16_gates_are_the_f32_bits_rounded(case):
    B, T, H = case["shape"]
    K = V = rwkv7_prep.DIM
    ks = jax.random.split(jax.random.key(T), 8)
    q, k = (jax.random.normal(ks[i], (B, T, H, K)).astype(BF16) for i in range(2))
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    kk = jax.random.normal(ks[3], (B, T, H, K))
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    gates = ((BOUND * jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H, K)))).astype(BF16), (-kk).astype(BF16),
             (kk * jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H, K)))).astype(BF16))
    state = 0.1 * jax.random.normal(ks[6], (B, H, K, V)) if case["h0"] else None
    do, dht = jax.random.normal(ks[7], (B, T, H, V)), jax.random.normal(ks[6], (B, H, K, V))
    seg = _segment_ids(B, T, case["docs"], H)

    def loss(q, k, v, w, a, b, state):
        o, final = ops.linear_attention("rwkv7", q, k, v, w=w, a=a, b=b, scale=1.0, impl="mosaic_gpu",
                                        initial_state=state, output_final_state=True, segment_ids=seg,
                                        lower_bound=BOUND)
        return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht), (o, final)

    argnums = (0, 1, 2, 3, 4, 5, 6) if case["h0"] else (0, 1, 2, 3, 4, 5)
    run = jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))
    (_, out16), grads16 = run(q, k, v, *gates, state)
    (_, out32), grads32 = run(q, k, v, *(x.astype(F32) for x in gates), state)
    for got, want in zip(jax.tree.leaves((out16, grads16)), jax.tree.leaves((out32, grads32))):
        np.testing.assert_array_equal(np.asarray(got.astype(F32)), np.asarray(want.astype(got.dtype).astype(F32)))


@gpu
@pytest.mark.parametrize("case", [
    dict(shape=(1, 4096, 4), docs=16, h0=True),
    dict(shape=(2, 1000, 2), docs=5, h0=False),
    dict(shape=(1, 2048, 2), docs=40, h0=True),  # documents shorter than a chunk
], ids=["h4_docs16", "b2_odd_t", "short_docs"])
def test_document_split_is_exact(case):
    """With packed documents the forward state pass is cut at chunks where a document starts: outputs,
    final state and every gradient equal those of one chain per row (SPLIT_DOCUMENTS off)."""
    from jla.ops.mosaic_gpu import vector_gate_ops as vgo

    B, T, H = case["shape"]
    K = V = rwkv7_prep.DIM
    ks = jax.random.split(jax.random.key(T + H), 8)
    q, k = (jax.random.normal(ks[i], (B, T, H, K)).astype(BF16) for i in range(2))
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    kk = jax.random.normal(ks[3], (B, T, H, K))
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    gates = (BOUND * jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H, K))), -kk,
             kk * jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H, K))))
    state = 0.1 * jax.random.normal(ks[6], (B, H, K, V)) if case["h0"] else None
    do, dht = jax.random.normal(ks[7], (B, T, H, V)), jax.random.normal(ks[6], (B, H, K, V))
    seg = _segment_ids(B, T, case["docs"], T)

    def loss(q, k, v, w, a, b, state):
        o, final = ops.linear_attention("rwkv7", q, k, v, w=w, a=a, b=b, scale=1.0, impl="mosaic_gpu",
                                        initial_state=state, output_final_state=True, segment_ids=seg,
                                        lower_bound=BOUND)
        return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht), (o, final)

    argnums = (0, 1, 2, 3, 4, 5, 6) if case["h0"] else (0, 1, 2, 3, 4, 5)
    results = []
    for split in (True, False):
        old, vgo.SPLIT_DOCUMENTS = vgo.SPLIT_DOCUMENTS, split
        try:
            fn = jax.jit(jax.value_and_grad(lambda *a: loss(*a), argnums=argnums, has_aux=True))
            results.append(jax.tree.leaves(fn(q, k, v, *gates, state)))
            results.append(jax.tree.leaves(jax.jit(lambda *a: loss(*a)[1])(q, k, v, *gates, state)))
        finally:
            vgo.SPLIT_DOCUMENTS = old
    for got, want in zip(results[0] + results[1], results[2] + results[3]):
        np.testing.assert_array_equal(np.asarray(got.astype(F32)), np.asarray(want.astype(F32)))


@gpu
@pytest.mark.parametrize("case", [
    dict(shape=(2, 1000, 4), docs=7, h0=True),
    dict(shape=(2, 512, 3), docs=0, h0=False),  # H * K = 192: the XLA chunk decays
], ids=["b2_odd_t_docs", "h3"])
def test_bf16_a_b_with_f32_w_are_the_f32_bits_rounded(case):
    """bf16 ``a, b`` with an f32 ``w`` (the kernels' bf16 ``a, b`` tiles with f32 gate gradients): outputs, final
    state and every gradient are the bits of f32 ``a, b`` of the same values, ``da, db`` rounded to bf16
    afterwards; ``dw`` stays f32 and is bitwise equal."""
    B, T, H = case["shape"]
    K = V = rwkv7_prep.DIM
    ks = jax.random.split(jax.random.key(T + 7 * H), 9)
    q, k = (jax.random.normal(ks[i], (B, T, H, K)).astype(BF16) for i in range(2))
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    kk = jax.random.normal(ks[3], (B, T, H, K))
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    w = BOUND * jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H, K)))
    a16, b16 = (-kk).astype(BF16), (kk * jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H, K)))).astype(BF16)
    state = 0.1 * jax.random.normal(ks[6], (B, H, K, V)) if case["h0"] else None
    do, dht = jax.random.normal(ks[7], (B, T, H, V)), jax.random.normal(ks[8], (B, H, K, V))
    seg = _segment_ids(B, T, case["docs"], H)

    def loss(q, k, v, w, a, b, state):
        o, final = ops.linear_attention("rwkv7", q, k, v, w=w, a=a, b=b, scale=1.0, impl="mosaic_gpu",
                                        initial_state=state, output_final_state=True, segment_ids=seg,
                                        lower_bound=BOUND)
        return jnp.sum(o.astype(F32) * do) + jnp.sum(final * dht), (o, final)

    argnums = (0, 1, 2, 3, 4, 5, 6) if case["h0"] else (0, 1, 2, 3, 4, 5)
    run = jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))
    (_, out16), grads16 = run(q, k, v, w, a16, b16, state)
    (_, out32), grads32 = run(q, k, v, w, a16.astype(F32), b16.astype(F32), state)
    assert grads16[3].dtype == F32 and grads16[4].dtype == BF16 and grads16[5].dtype == BF16
    for got, want in zip(jax.tree.leaves((out16, grads16)), jax.tree.leaves((out32, grads32)), strict=True):
        np.testing.assert_array_equal(np.asarray(got.astype(F32)), np.asarray(want.astype(got.dtype).astype(F32)))
