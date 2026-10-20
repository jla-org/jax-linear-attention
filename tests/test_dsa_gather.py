# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``gather_attention`` (selected-key DSA attention by gathering) against the plain DSA path ``dsa`` and, on an
H100, against the masked-dense kernels (``dsa_attention``) at the benchmark's head shapes.

``jla.ops.sparse_attention.dsa`` masks dense ``[B, T, T]`` float32 scores; ``gather_attention`` gathers each
query's selected keys and never forms ``T^2``. Both must give the same output and the same gradients of
``q``, ``k`` and ``v``: with float32 inputs to float32 rounding, with bf16 inputs within bf16 noise of the
float32 math on the same (rounded) inputs. The selections cover top-k picks (``dsa_select``), arbitrary
distinct indices with future and other-document keys (which must be ignored) and ``-1`` slots, packed
documents, GQA groups, MLA-like head dims (``D != E``), rows without any valid key, and lengths that are
not a multiple of the query chunk.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.dsa_gather import gather_attention
from jla.ops.sparse_attention import dsa, dsa_index_scores, dsa_select


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _inputs(seed, batch, length, q_heads, kv_heads, dim, value_dim, dtype):
    keys = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(keys[0], (batch, length, q_heads, dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, value_dim), jnp.float32)
    do = jax.random.normal(keys[3], (batch, length, q_heads, value_dim), jnp.float32)
    return tuple(x.astype(dtype) for x in (q, k, v, do))


def _random_distinct(seed, batch, length, top, unused=0.2):
    """Distinct arbitrary keys per row (future and other-document keys included), some slots ``-1``."""
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((batch, length, length)), axis=-1)[..., :top].astype(np.int32)
    idx[rng.random(idx.shape) < unused] = -1
    return jnp.asarray(idx)


def _segments(batch, length, seed):
    rng = np.random.default_rng(seed)
    seg = np.zeros((batch, length), np.int32)
    for b in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=3, replace=False))
        for i, c in enumerate(cuts):
            seg[b, c:] = i + 1
    return jnp.asarray(seg)


def _value_and_grads(fn, q, k, v, do):
    def loss(q, k, v):
        return jnp.sum(fn(q, k, v).astype(jnp.float32) * do.astype(jnp.float32))
    out = fn(q, k, v)
    return (out,) + jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


def _compare(q, k, v, do, idx, seg, *, chunk_bytes, rtol, atol):
    ref = _value_and_grads(lambda a, b, c: dsa(a.astype(jnp.float32), b.astype(jnp.float32), c.astype(jnp.float32),
                                                idx, segment_ids=seg), q, k, v, do)
    got = _value_and_grads(lambda a, b, c: gather_attention(a, b, c, idx, segment_ids=seg, chunk_bytes=chunk_bytes),
                           q, k, v, do)
    for name, r, g in zip(("o", "dq", "dk", "dv"), ref, got):
        assert g.dtype == q.dtype, name
        r, g = np.asarray(r, np.float32), np.asarray(g, np.float32)
        scale = max(float(np.max(np.abs(r))), 1e-6)
        np.testing.assert_allclose(g, r, rtol=rtol, atol=atol * scale, err_msg=name)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("heads", [(4, 2), (3, 1)])
def test_gather_attention_matches_dsa_float32(packed, heads):
    q_heads, kv_heads = heads
    batch, length, top = 2, 77, 24
    q, k, v, do = _inputs(0, batch, length, q_heads, kv_heads, 16, 12, jnp.float32)
    seg = _segments(batch, length, 1) if packed else None
    idx = _random_distinct(2, batch, length, top)
    # a small chunk budget: several query chunks, the last one padded
    _compare(q, k, v, do, idx, seg, chunk_bytes=20 * top * kv_heads * 28 * 4, rtol=1e-5, atol=1e-6)


def test_gather_attention_on_top_k_selections_float32():
    batch, length, q_heads, kv_heads, top = 2, 96, 4, 1, 16
    q, k, v, do = _inputs(3, batch, length, q_heads, kv_heads, 32, 32, jnp.float32)
    seg = _segments(batch, length, 4)
    keys = jax.random.split(jax.random.key(5), 3)
    qi = jax.random.normal(keys[0], (batch, length, 4, 8))
    ki = jax.random.normal(keys[1], (batch, length, 8))
    wi = jax.random.normal(keys[2], (batch, length, 4))
    idx = dsa_select(dsa_index_scores(qi, ki, wi, segment_ids=seg), top)
    _compare(q, k, v, do, idx, seg, chunk_bytes=512 << 20, rtol=1e-5, atol=1e-6)


def test_gather_attention_matches_dsa_bf16_mla_dims():
    # MLA-like: the query/key head dim differs from the value head dim; many query heads on one latent head
    batch, length, q_heads, kv_heads, top = 1, 130, 8, 1, 32
    q, k, v, do = _inputs(6, batch, length, q_heads, kv_heads, 96, 64, jnp.bfloat16)
    idx = _random_distinct(7, batch, length, top, unused=0.1)
    _compare(q, k, v, do, idx, None, chunk_bytes=1 << 20, rtol=0, atol=2e-2)


def test_gather_attention_rows_without_keys_are_zero():
    batch, length, top = 1, 20, 4
    q, k, v, do = _inputs(8, batch, length, 2, 1, 8, 8, jnp.float32)
    idx = np.full((batch, length, top), -1, np.int32)
    idx[0, 5:, 0] = 3  # rows 5.. see key 3; rows 0..4 see nothing (no slot, or only the future key 3)
    idx[0, :3, 1] = 3
    out, dq, dk, dv = _value_and_grads(lambda a, b, c: gather_attention(a, b, c, jnp.asarray(idx)), q, k, v, do)
    assert np.all(np.asarray(out)[0, :5] == 0)
    np.testing.assert_allclose(np.asarray(out)[0, 5:], np.broadcast_to(np.asarray(v)[0, 3, 0], (15, 2, 8)), rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(dq))) and np.all(np.asarray(dq) == 0)  # one key per row: softmax is constant
    dk_np, dv_np = np.asarray(dk), np.asarray(dv)
    assert np.all(dk_np == 0)
    assert np.all(dv_np[0, np.arange(length) != 3] == 0)
    np.testing.assert_allclose(dv_np[0, 3, 0], np.asarray(do)[0, 5:].sum((0, 1)), rtol=1e-5, atol=1e-6)


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


@pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (the masked-dense kernels)")
def test_gather_attention_matches_the_masked_dense_kernels():
    # the benchmark's head shapes (16 query heads on one key/value head of 128, top_k 2048), bf16, packed
    # documents, three query blocks of the kernels (2048 + 2048 + the padded rest) and 132 query chunks here
    from jla.ops.mosaic_gpu.dsa import dsa_attention

    batch, length, q_heads, kv_heads, dim, top = 1, 4196, 16, 1, 128, 2048
    q, k, v, do = _inputs(10, batch, length, q_heads, kv_heads, dim, dim, jnp.bfloat16)
    seg = _segments(batch, length, 11)
    rng = np.random.default_rng(12)
    idx = np.empty((batch, length, top), np.int32)
    for t in range(length):  # distinct keys, mostly causal, some future and some unused slots
        idx[0, t] = rng.choice(min(length, t + 64 + top), size=top, replace=False) if t + 64 + top <= length \
            else rng.permutation(length)[:top]
    idx[rng.random(idx.shape) < 0.05] = -1
    idx = jnp.asarray(idx)

    def run(attention):
        return jax.jit(lambda q, k, v, do, i, s: _value_and_grads(
            lambda a, b, c: attention(a, b, c, i, segment_ids=s), q, k, v, do))(q, k, v, do, idx, seg)

    ref = run(lambda *a, **kw: dsa_attention(*a, **kw, block_rows=2048))
    got = run(lambda *a, **kw: gather_attention(*a, **kw, chunk_bytes=64 << 20))
    for name, r, g in zip(("o", "dq", "dk", "dv"), ref, got):
        r, g = np.asarray(r, np.float32), np.asarray(g, np.float32)
        np.testing.assert_allclose(g, r, rtol=0, atol=2e-2 * float(np.max(np.abs(r))), err_msg=name)


def test_gather_attention_rejects_bad_shapes():
    q, k, v, _ = _inputs(9, 1, 16, 4, 2, 8, 8, jnp.float32)
    with pytest.raises(ValueError, match="indices"):
        gather_attention(q, k, v, jnp.zeros((1, 15, 4), jnp.int32))
    with pytest.raises(ValueError, match="segment_ids"):
        gather_attention(q, k, v, jnp.zeros((1, 16, 4), jnp.int32), segment_ids=jnp.zeros((1, 15), jnp.int32))
