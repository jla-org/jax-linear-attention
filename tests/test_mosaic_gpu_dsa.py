# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU DSA kernels (``jla.ops.mosaic_gpu.dsa``) against the plain-JAX ops of ``jla.ops.sparse_attention``.

The indexer kernel is compared with ``dsa_index_scores`` on the same bf16 inputs (and must be
bit-exact on integer-valued inputs, so ``dsa_select`` picks the same keys); the selected-key
attention kernel with ``dsa`` (outputs and ``dq, dk, dv``) for GQA, packed documents, unused and
out-of-range indices, queries without keys and lengths that are not a multiple of the tiles; and
the DSA layer's ``dsa_implementation="mosaic_gpu"`` path with its ``"xla"`` path. The kernel tests
need an H100.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models.config import AttentionConfig, SparseAttentionConfig
from jla.ops.sparse_attention import dsa, dsa_index_scores, dsa_select


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


def _segments(batch: int, length: int, packed: bool):
    if not packed:
        return None
    seg = np.zeros((batch, length), np.int32)
    seg[:, length // 3:] = 1
    seg[-1, (2 * length) // 3:] = 0  # equal ids in different runs are different documents
    return jnp.asarray(seg)


def test_dsa_implementation_is_validated():
    assert SparseAttentionConfig().dsa_implementation == "auto"
    assert SparseAttentionConfig(dsa_implementation="xla").dsa_implementation == "xla"
    with pytest.raises(ValueError, match="dsa_implementation"):
        SparseAttentionConfig(dsa_implementation="cuda")


def test_selection_mask_matches_the_plain_mask():
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    from jla.ops.mosaic_gpu.dsa import selection_mask
    from jla.ops.sparse_attention import _causal_document_mask, _documents, _selection_mask

    rng = np.random.default_rng(4)
    length = 50
    idx = jnp.asarray(rng.integers(-1, length, (2, length, 9)), jnp.int32)
    seg = _segments(2, length, True)
    doc, _ = _documents(seg, 2, length)
    want = np.asarray(_selection_mask(idx, length) & _causal_document_mask(doc))
    np.testing.assert_array_equal(np.asarray(selection_mask(idx, length, seg)), want)
    padded = np.asarray(selection_mask(idx, length, seg, padded=64, dtype=jnp.bfloat16), np.float32)
    np.testing.assert_array_equal(padded[:, :length, :length], want.astype(np.float32))
    assert not padded[:, length:].any() and not padded[:, :, length:].any()


def test_select_top_k_matches_dsa_select():
    """The causal-prefix top-k returns exactly dsa_select's indices, ties and unused slots included."""
    pytest.importorskip("jax.experimental.pallas.mosaic_gpu")
    from jla.ops.mosaic_gpu.dsa import select_top_k

    rng = np.random.default_rng(9)
    for length, top_k in ((200, 16), (200, 150), (64, 300), (97, 5), (5, 3)):
        seg = _segments(2, length, True)
        qi = jnp.asarray(rng.integers(-2, 3, (2, length, 2, 8)), jnp.float32)  # integer scores: many ties
        ki = jnp.asarray(rng.integers(-2, 3, (2, length, 8)), jnp.float32)
        scores = dsa_index_scores(qi, ki, jnp.ones((2, length, 2)), scale=1.0, segment_ids=seg)
        np.testing.assert_array_equal(np.asarray(select_top_k(scores, top_k)),
                                      np.asarray(dsa_select(scores, top_k)))


@gpu
@pytest.mark.parametrize("length,heads,dim,packed", [(256, 4, 64, False), (200, 3, 128, True), (512, 8, 128, True)])
def test_index_scores_match_the_plain_op(length, heads, dim, packed):
    from jla.ops.mosaic_gpu.dsa import index_scores

    keys = jax.random.split(jax.random.key(length + heads), 3)
    batch = 2
    q = jax.random.normal(keys[0], (batch, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, dim), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.uniform(keys[2], (batch, length, heads), jnp.float32, 0.1, 1.0)
    seg = _segments(batch, length, packed)
    got = np.asarray(jax.jit(lambda *a: index_scores(*a, segment_ids=seg))(q, k, w))
    with jax.default_matmul_precision("highest"):
        want = np.asarray(jax.jit(lambda *a: dsa_index_scores(*a, segment_ids=seg))(q, k, w))
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=2e-5, atol=2e-5)


@gpu
def test_index_scores_gradients_match_the_plain_op():
    from jla.ops.mosaic_gpu.dsa import index_scores

    keys = jax.random.split(jax.random.key(5), 4)
    batch, length, heads, dim = 2, 200, 3, 64
    q = jax.random.normal(keys[0], (batch, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, dim), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.uniform(keys[2], (batch, length, heads), jnp.float32, 0.1, 1.0)
    ct = jax.random.normal(keys[3], (batch, length, length), jnp.float32)
    seg = _segments(batch, length, True)

    def objective(fn, q, k, w):
        s = fn(q, k, w, segment_ids=seg)
        return jnp.sum(jnp.where(jnp.isfinite(s), s, 0.0) * ct)

    got = jax.jit(jax.grad(lambda *a: objective(index_scores, *a), argnums=(0, 1, 2)))(q, k, w)
    f32 = (q.astype(jnp.float32), k.astype(jnp.float32), w)
    with jax.default_matmul_precision("highest"):
        want = jax.jit(jax.grad(lambda *a: objective(dsa_index_scores, *a), argnums=(0, 1, 2)))(*f32)
    for name, a, b in zip(("dq", "dk", "dw"), got, want):
        a, b = np.asarray(a, np.float32), np.asarray(b)
        err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-6)
        assert err < 1e-2, (name, err)


@gpu
def test_index_selection_is_exact_on_integer_inputs():
    """Integer q, k and dyadic w make every partial sum exact: the kernel scores are bit-equal."""
    from jla.ops.mosaic_gpu.dsa import index_scores

    rng = np.random.default_rng(7)
    batch, length, heads, dim = 2, 384, 8, 128
    q = jnp.asarray(rng.integers(-2, 3, (batch, length, heads, dim)), jnp.bfloat16)
    k = jnp.asarray(rng.integers(-2, 3, (batch, length, dim)), jnp.bfloat16)
    w = jnp.asarray(rng.integers(1, 9, (batch, length, heads)) / 8.0, jnp.float32)
    seg = _segments(batch, length, True)
    got = jax.jit(lambda *a: index_scores(*a, scale=0.5, segment_ids=seg))(q, k, w)
    with jax.default_matmul_precision("highest"):
        want = jax.jit(lambda *a: dsa_index_scores(*a, scale=0.5, segment_ids=seg))(q, k, w)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    for top_k in (16, 100):
        np.testing.assert_array_equal(np.asarray(dsa_select(got, top_k)), np.asarray(dsa_select(want, top_k)))


@gpu
def test_index_selection_matches_on_ties_free_inputs():
    """Random float inputs: rows whose k-th and (k+1)-th scores are well separated select the same keys."""
    from jla.ops.mosaic_gpu.dsa import index_scores

    keys = jax.random.split(jax.random.key(11), 3)
    batch, length, heads, dim, top_k = 1, 512, 4, 64, 32
    q = jax.random.normal(keys[0], (batch, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, dim), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.uniform(keys[2], (batch, length, heads), jnp.float32, 0.1, 1.0)
    got = np.asarray(dsa_select(index_scores(q, k, w), top_k))
    with jax.default_matmul_precision("highest"):
        scores = dsa_index_scores(q, k, w)
    want = np.asarray(dsa_select(scores, top_k))
    ref = np.sort(np.asarray(scores), axis=-1)[..., ::-1]
    with np.errstate(invalid="ignore"):  # -inf - -inf where a row has fewer than top_k + 1 visible keys
        separated = (ref[..., top_k - 1] - ref[..., top_k]) > 1e-3 * np.maximum(np.abs(ref[..., top_k - 1]), 1.0)
    separated |= ~np.isfinite(ref[..., top_k])  # fewer visible keys than top_k: no boundary
    assert separated.mean() > 0.9
    for b, t in zip(*np.nonzero(separated)):
        assert set(got[b, t].tolist()) == set(want[b, t].tolist()), (b, t)


def _attention_inputs(seed, batch, length, q_heads, kv_heads, dim, top_k):
    keys = jax.random.split(jax.random.key(seed), 5)
    q = jax.random.normal(keys[0], (batch, length, q_heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, dim), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, dim), jnp.float32).astype(jnp.bfloat16)
    do = jax.random.normal(keys[3], (batch, length, q_heads, dim), jnp.float32).astype(jnp.bfloat16)
    rng = np.random.default_rng(seed)
    idx = rng.integers(-1, length, (batch, length, top_k))  # unused (-1), duplicate and future keys
    idx[:, ::7, :] = -1  # every 7th query selects nothing
    idx[:, 5::11, 0] = np.arange(5, length, 11)  # some queries also select themselves
    return q, k, v, do, jnp.asarray(idx, jnp.int32)


@gpu
@pytest.mark.parametrize("length,q_heads,kv_heads,dim,top_k,packed", [
    (256, 4, 2, 64, 32, False),
    (200, 8, 2, 128, 48, True),
    (384, 4, 4, 128, 512, True),
])
def test_dsa_attention_matches_the_plain_op(length, q_heads, kv_heads, dim, top_k, packed):
    from jla.ops.mosaic_gpu.dsa import dsa_attention

    batch = 2
    q, k, v, do, idx = _attention_inputs(length + dim, batch, length, q_heads, kv_heads, dim, top_k)
    seg = _segments(batch, length, packed)

    def kernel(q, k, v):
        return dsa_attention(q, k, v, idx, segment_ids=seg)

    def plain(q, k, v):
        return dsa(q, k, v, idx, segment_ids=seg)

    out, vjp = jax.vjp(jax.jit(kernel), q, k, v)
    grads = vjp(do)
    f32 = [x.astype(jnp.float32) for x in (q, k, v)]
    with jax.default_matmul_precision("highest"):
        want, want_vjp = jax.vjp(jax.jit(plain), *f32)
        want_grads = want_vjp(do.astype(jnp.float32))
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(want), rtol=2e-2, atol=2e-2)
    for name, g, w in zip(("dq", "dk", "dv"), grads, want_grads):
        g, w = np.asarray(g, np.float32), np.asarray(w)
        err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-6)
        assert err < 2e-2, (name, err)
    empty = np.asarray(idx < 0).all(-1)  # rows that select nothing give zeros
    assert empty.any() and np.all(np.asarray(out, np.float32)[empty] == 0.0)


def _distinct_indices(rng, batch, length, top_k, segment_ids=None):
    """``[B, T, K]`` distinct causal keys of each query's document (``-1`` once they run out), as top-k gives."""
    seg = np.zeros((batch, length), np.int64) if segment_ids is None else np.asarray(segment_ids)
    idx = np.full((batch, length, top_k), -1, np.int64)
    for b in range(batch):
        start = 0
        for t in range(length):
            if t and seg[b, t] != seg[b, t - 1]:
                start = t
            keys = rng.permutation(np.arange(start, t + 1))[:top_k]
            idx[b, t, :len(keys)] = keys
    return jnp.asarray(idx, jnp.int32)


@gpu
def test_sparse_attention_target_matches_the_dense_target():
    from jla.ops.mosaic_gpu.dsa import dsa_attention, sparse_attention_target, valid_indices
    from jla.ops.sparse_attention import dsa_attention_target

    q, k, v, _, _ = _attention_inputs(3, 1, 192, 8, 2, 64, 40)
    seg = _segments(1, 192, True)
    idx = _distinct_indices(np.random.default_rng(3), 1, 192, 40, seg)
    _, lse = dsa_attention(q, k, v, idx, segment_ids=seg, return_lse=True)
    valid = valid_indices(idx, seg)
    with jax.default_matmul_precision("highest"):
        target = sparse_attention_target(q, k, lse, idx, valid)
        _, probs, _ = dsa(*(x.astype(jnp.float32) for x in (q, k, v)), idx, segment_ids=seg, return_probs=True)
    dense = np.asarray(dsa_attention_target(probs))
    safe = np.clip(np.asarray(idx), 0, None)
    want = np.where(np.asarray(valid), np.take_along_axis(dense, safe, axis=-1), 0.0)
    np.testing.assert_allclose(np.asarray(target), want, rtol=2e-3, atol=2e-5)


@gpu
@pytest.mark.parametrize("batch,length,q_heads,kv_heads,dim,top_k,block_rows,packed", [
    (1, 192, 8, 2, 64, 40, None, True),  # GQA, padding, packed documents
    (2, 700, 16, 1, 128, 96, 256, False),  # MQA, three query blocks
    (1, 600, 4, 4, 64, 600, 256, True),  # top_k covering every key, packed documents, blocks
])
def test_selected_attention_target_matches_the_dense_target(batch, length, q_heads, kv_heads, dim, top_k,
                                                            block_rows, packed):
    """The target on the indexer kernel (exp mode) against the dense plain-JAX target of the same selection."""
    from jla.ops.mosaic_gpu.dsa import dsa_attention, selected_attention_target, valid_indices
    from jla.ops.sparse_attention import dsa_attention_target

    q, k, v, _, _ = _attention_inputs(length, batch, length, q_heads, kv_heads, dim, top_k)
    seg = _segments(batch, length, packed)
    idx = _distinct_indices(np.random.default_rng(length), batch, length, top_k, seg)
    _, lse = dsa_attention(q, k, v, idx, segment_ids=seg, return_lse=True)
    valid = valid_indices(idx, seg)
    target = jax.jit(lambda q, k, lse: selected_attention_target(q, k, lse, idx, valid, segment_ids=seg,
                                                                 block_rows=block_rows))(q, k, lse)
    with jax.default_matmul_precision("highest"):
        _, probs, _ = dsa(*(x.astype(jnp.float32) for x in (q, k, v)), idx, segment_ids=seg, return_probs=True)
    dense = np.asarray(dsa_attention_target(probs))
    safe = np.clip(np.asarray(idx), 0, None)
    want = np.where(np.asarray(valid), np.take_along_axis(dense, safe, axis=-1), 0.0)
    np.testing.assert_allclose(np.asarray(target), want, rtol=2e-3, atol=2e-5)


@gpu
@pytest.mark.parametrize("length,offset_blocks", [(700, 256), (512, 256)])
def test_index_topk_blocks_match_dsa_select(length, offset_blocks):
    """Query blocks of the indexer give exactly dsa_select's indices (integer inputs: exact scores)."""
    from jla.ops.mosaic_gpu.dsa import index_topk

    rng = np.random.default_rng(length)
    batch, heads, dim = 2, 4, 64
    q = jnp.asarray(rng.integers(-2, 3, (batch, length, heads, dim)), jnp.bfloat16)
    k = jnp.asarray(rng.integers(-2, 3, (batch, length, dim)), jnp.bfloat16)
    w = jnp.asarray(rng.integers(1, 9, (batch, length, heads)) / 8.0, jnp.float32)
    seg = _segments(batch, length, True)
    with jax.default_matmul_precision("highest"):
        scores = dsa_index_scores(q, k, w, scale=0.5, segment_ids=seg)
    for top_k in (16, 300):
        got = jax.jit(lambda q, k, w: index_topk(q, k, w, top_k, scale=0.5, segment_ids=seg,
                                                 block_rows=offset_blocks))(q, k, w)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(dsa_select(scores, top_k)))
    got = index_topk(q, k, w, 16, scale=0.5, segment_ids=seg)  # one block
    np.testing.assert_array_equal(np.asarray(got), np.asarray(dsa_select(scores, 16)))


@gpu
@pytest.mark.parametrize("length,block_rows,packed", [(400, 128, True), (520, 256, False)])
def test_dsa_attention_query_blocks_match_the_plain_op(length, block_rows, packed):
    from jla.ops.mosaic_gpu.dsa import dsa_attention

    batch = 2
    q, k, v, do, idx = _attention_inputs(length, batch, length, 8, 2, 64, 96)
    seg = _segments(batch, length, packed)

    def kernel(q, k, v):
        return dsa_attention(q, k, v, idx, segment_ids=seg, block_rows=block_rows)

    out, vjp = jax.vjp(jax.jit(kernel), q, k, v)
    grads = vjp(do)
    f32 = [x.astype(jnp.float32) for x in (q, k, v)]
    with jax.default_matmul_precision("highest"):
        want, want_vjp = jax.vjp(lambda *a: dsa(*a, idx, segment_ids=seg), *f32)
        want_grads = want_vjp(do.astype(jnp.float32))
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(want), rtol=2e-2, atol=2e-2)
    for name, g, w in zip(("dq", "dk", "dv"), grads, want_grads):
        g, w = np.asarray(g, np.float32), np.asarray(w)
        err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-6)
        assert err < 2e-2, (name, err)


@gpu
def test_backward_uses_the_rounded_cotangent():
    """The loss is taken in fp32 against weights passed to the jitted function, so the bf16 cotangent of the
    kernel output is rounded from fp32 inside the same program: ``D = rowsum(o * do)`` must use the rounded
    values the kernels read (``attention._delta``). The gradients equal those for pre-rounded weights, and a
    query that selects only itself has ``p = 1``, so its ``dq`` is 0."""
    from jla.ops.mosaic_gpu.dsa import dsa_attention

    batch, length = 2, 256
    q, k, v, _, idx = _attention_inputs(8, batch, length, 4, 2, 64, 8)
    alone = np.zeros((batch, length), bool)
    alone[:, 3::5] = True  # these queries select only themselves
    t = np.arange(length)[None, :, None]
    idx = jnp.where(jnp.asarray(alone)[..., None], jnp.where(jnp.arange(8) == 0, jnp.asarray(t), -1), idx)
    ct = jax.random.normal(jax.random.key(9), (batch, length, 4, 64), jnp.float32)

    @jax.jit
    def grads(q, k, v, ct):
        def loss(q, k, v):
            return jnp.sum(dsa_attention(q, k, v, idx, block_rows=128).astype(jnp.float32) * ct)

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    got = grads(q, k, v, ct)
    want = grads(q, k, v, ct.astype(jnp.bfloat16).astype(jnp.float32))
    for g, w in zip(got, want):
        np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(w, np.float32), rtol=1e-5, atol=1e-5)
    assert np.abs(np.asarray(got[0], np.float32)[alone]).max() < 1e-5


def _layer_reference(q, k, v, q_idx, k_idx, w_idx, top_k, coef, seg):
    """The plain DSA layer core (dense scores, dsa_select, dsa, attach_indexer_grad), float32."""
    from jla.ops.sparse_attention import attach_indexer_grad, dsa_attention_target

    scores = dsa_index_scores(q_idx, k_idx, w_idx, segment_ids=seg)
    indices = dsa_select(scores, top_k)
    o, probs, selected = dsa(q, k, v, indices, segment_ids=seg, return_probs=True)
    if coef > 0:
        o = attach_indexer_grad(o, scores, dsa_attention_target(probs), selected.astype(jnp.float32), coef)
    return o


@gpu
@pytest.mark.parametrize("length,heads,dim,top_k,block_rows,packed", [
    (520, 16, 128, 64, 256, True),  # padding, three query blocks, two head groups, packed documents
    (300, 12, 64, 512, None, False),  # top_k above every prefix: unused slots
    (1024, 64, 128, 128, 512, False),  # DeepSeek-V3.2's indexer heads (64 x 128), two blocks
])
def test_selected_index_scores_gradients_match_the_gather_form(length, heads, dim, top_k, block_rows, packed):
    """The indexer backward kernels against the plain-JAX gather form (float32 inputs, highest precision);
    the selected scores are the top-k values and equal the gathered scores."""
    from jla.ops.mosaic_gpu.dsa import _index_topk, selected_index_scores, sparse_index_scores, valid_indices

    batch = 2
    keys = jax.random.split(jax.random.key(length + heads), 4)
    q = jax.random.normal(keys[0], (batch, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, dim), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(keys[2], (batch, length, heads), jnp.float32) * heads ** -0.5
    seg = _segments(batch, length, packed)
    idx, values = jax.jit(lambda q, k, w: _index_topk(q, k, w, top_k, segment_ids=seg, block_rows=block_rows))(q, k, w)
    valid = valid_indices(idx, seg)
    ct = jax.random.normal(keys[3], idx.shape, jnp.float32)
    f32 = (q.astype(jnp.float32), k.astype(jnp.float32), w)
    with jax.default_matmul_precision("highest"):
        gathered = jax.jit(lambda q, k, w: sparse_index_scores(q, k, w, idx))(*f32)
    v, gth = np.asarray(values), np.asarray(gathered)
    ok = np.asarray(valid)
    assert np.all(np.isneginf(v[~ok]))  # unused slots
    np.testing.assert_allclose(v[ok], gth[ok], rtol=1e-4, atol=1e-4 * np.max(np.abs(gth[ok])))

    def objective(fn):
        return lambda q, k, w: jnp.sum(jnp.where(valid, fn(q, k, w), 0.0) * ct)

    kernel = objective(lambda q, k, w: selected_index_scores(q, k, w, idx, values, valid, block_rows=block_rows))
    got = jax.jit(jax.grad(kernel, argnums=(0, 1, 2)))(q, k, w)
    with jax.default_matmul_precision("highest"):
        want = jax.jit(jax.grad(objective(lambda q, k, w: sparse_index_scores(q, k, w, idx)), argnums=(0, 1, 2)))(*f32)
    assert got[0].dtype == q.dtype and got[1].dtype == k.dtype and got[2].dtype == w.dtype
    for name, a, b in zip(("dq", "dk", "dw"), got, want):
        a, b = np.asarray(a, np.float32), np.asarray(b)
        err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-6)
        assert err < 1e-2, (name, err)


@gpu
@pytest.mark.parametrize("length,block_rows,packed", [(600, 256, True), (300, None, False)])
def test_dsa_layer_core_matches_the_plain_path(length, block_rows, packed):
    """dsa_layer (indexer blocks, top-k, attention blocks, sparse KL) against the dense plain path: outputs and the
    gradients of every input, the indexer's from the KL loss alone."""
    from jla.ops.mosaic_gpu.dsa import dsa_layer

    batch, hq, h, dim, hi, di, top_k, coef = 2, 8, 2, 64, 4, 64, 64, 1.0
    rng = np.random.default_rng(length)
    keys = jax.random.split(jax.random.key(length), 8)
    q = jax.random.normal(keys[0], (batch, length, hq, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, h, dim), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (batch, length, h, dim), jnp.float32).astype(jnp.bfloat16)
    # integer indexer inputs: exact scores, so both paths select the same keys
    q_idx = jnp.asarray(rng.integers(-3, 4, (batch, length, hi, di)), jnp.bfloat16)
    k_idx = jnp.asarray(rng.integers(-3, 4, (batch, length, di)), jnp.bfloat16)
    w_idx = jnp.asarray(rng.integers(1, 9, (batch, length, hi)) / 64.0, jnp.float32)
    ct = jax.random.normal(keys[3], (batch, length, hq, dim), jnp.float32)
    seg = _segments(batch, length, packed)
    inputs = (q, k, v, q_idx, k_idx, w_idx)

    def kernel(*a):
        o = dsa_layer(*a, top_k=top_k, indexer_loss_coef=coef, segment_ids=seg, block_rows=block_rows)
        return jnp.sum(o.astype(jnp.float32) * ct), o

    def plain(*a):
        o = _layer_reference(*a, top_k, coef, seg)
        return jnp.sum(o * ct), o

    (_, out), grads = jax.jit(jax.value_and_grad(kernel, argnums=tuple(range(6)), has_aux=True))(*inputs)
    with jax.default_matmul_precision("highest"):
        f32 = tuple(x.astype(jnp.float32) for x in inputs)
        (_, want), want_grads = jax.jit(jax.value_and_grad(plain, argnums=tuple(range(6)), has_aux=True))(*f32)
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(want), rtol=2e-2, atol=2e-2)
    # About 3x the measured errors (relative to each gradient's maximum, 18 Oct 2026: dq/dk/dv <= 5.1e-3,
    # dq_idx/dk_idx <= 3.6e-3, dw_idx <= 1.6e-4): the indexer gradients come from bf16 matmuls (the float32
    # score cotangent rounded to bf16, as in FlashAttention), dw_idx from a per-row scaled float16 cotangent.
    for name, g, w, tol in zip(("dq", "dk", "dv", "dq_idx", "dk_idx", "dw_idx"), grads, want_grads,
                               (1.5e-2, 1.5e-2, 1.5e-2, 1e-2, 1e-2, 1e-3)):
        g, w = np.asarray(g, np.float32), np.asarray(w)
        err = np.max(np.abs(g - w)) / max(np.max(np.abs(w)), 1e-6)
        assert err < tol, (name, err)


@gpu
@pytest.mark.parametrize("coef,impl", [(0.0, "mosaic_gpu"), (1.0, "mosaic_gpu"), (1.0, "auto")])
def test_dsa_layer_mosaic_gpu_matches_xla(coef, impl):
    import equinox as eqx

    from jla.layers.sparse_attention import DeepSeekSparseAttention

    attention = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64)
    base = SparseAttentionConfig(top_k=24, index_heads=4, index_head_dim=64, indexer_loss_coef=coef)
    layers = [DeepSeekSparseAttention(128, attention, dataclasses.replace(base, dsa_implementation=name),
                                      key=jax.random.key(0), std=0.3, norm_eps=1e-5, compute_dtype="bfloat16")
              for name in ("xla", impl)]
    batch, length = 2, 160
    x = jax.random.normal(jax.random.key(1), (batch, length, 128), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(length), (batch, length))
    seg = _segments(batch, length, True)
    ct = jax.random.normal(jax.random.key(2), (batch, length, 128), jnp.float32)
    q_idx, k_idx, w_idx = layers[1]._index_inputs(x, pos)
    q, k, v = layers[1].attn._project(x, pos)
    assert layers[1]._uses_kernels(q, k, v, q_idx, k_idx, w_idx)  # "auto" picks the kernels here

    with jax.default_matmul_precision("highest"):
        idx = [np.asarray(dsa_select(m.index_scores(x, pos, seg), base.top_k)) for m in layers]
    np.testing.assert_array_equal(idx[0], idx[1])  # these inputs have no near-ties at the top-k boundary

    def objective(m):
        return jnp.sum(m(x, pos, seg).astype(jnp.float32) * ct)

    with jax.default_matmul_precision("highest"):
        outs = [np.asarray(m(x, pos, seg), np.float32) for m in layers]
        grads = [eqx.filter_grad(objective)(m) for m in layers]
    # bf16 outputs of magnitude ~10: compare against the output scale (a bf16 ulp there is 0.06)
    assert np.max(np.abs(outs[1] - outs[0])) / np.max(np.abs(outs[0])) < 1e-2
    for field in ("attn", "index_q_proj", "index_k_proj", "index_w_proj", "index_k_norm_weight",
                  "index_k_norm_bias"):
        a = jax.tree.leaves(getattr(grads[0], field))
        b = jax.tree.leaves(getattr(grads[1], field))
        for ga, gb in zip(a, b, strict=True):
            ga, gb = np.asarray(ga, np.float32), np.asarray(gb, np.float32)
            scale = max(np.max(np.abs(ga)), 1e-6)
            assert np.max(np.abs(ga - gb)) / scale < 1.5e-2, field  # measured <= 5.5e-3 (18 Oct 2026)
        if field != "attn" and coef == 0.0:
            assert all(np.all(np.asarray(g) == 0) for g in b)


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="checks the CPU fallback of 'auto'")
def test_dsa_layer_auto_runs_the_plain_path_off_gpu():
    from jla.layers.sparse_attention import DeepSeekSparseAttention

    attention = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64)
    base = SparseAttentionConfig(top_k=8, index_heads=2, index_head_dim=64)
    layers = [DeepSeekSparseAttention(64, attention, dataclasses.replace(base, dsa_implementation=name),
                                      key=jax.random.key(0), std=0.3, norm_eps=1e-5, compute_dtype="bfloat16")
              for name in ("xla", "auto")]
    x = jax.random.normal(jax.random.key(1), (1, 40, 64), jnp.float32)
    pos = jnp.arange(40)[None]
    outs = [np.asarray(m(x, pos), np.float32) for m in layers]
    np.testing.assert_array_equal(outs[0], outs[1])


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
def test_dsa_layer_runs_per_batch_shard():
    """Under a context mesh (``jax.set_mesh``) with the batch sharded over two GPUs, ``dsa_layer`` runs its
    kernels on each device's own rows (no bf16 operand all-gather) and gives the single-device values and
    gradients (``jla.distributed.kernel_sharding.shard_layout_call``)."""
    import re

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from jla.ops.mosaic_gpu.dsa import dsa_layer

    mesh = Mesh(jax.devices()[:2], ("data",))
    rows = NamedSharding(mesh, PartitionSpec("data"))
    batch, length, hq, h, dim, hi, di = 4, 384, 8, 2, 64, 4, 64
    keys = jax.random.split(jax.random.key(0), 7)
    bf16 = jnp.bfloat16
    q = jax.random.normal(keys[0], (batch, length, hq, dim), jnp.float32).astype(bf16)
    k = jax.random.normal(keys[1], (batch, length, h, dim), jnp.float32).astype(bf16)
    v = jax.random.normal(keys[2], (batch, length, h, dim), jnp.float32).astype(bf16)
    q_idx = jax.random.normal(keys[3], (batch, length, hi, di), jnp.float32).astype(bf16)
    k_idx = jax.random.normal(keys[4], (batch, length, di), jnp.float32).astype(bf16)
    w_idx = jax.random.uniform(keys[5], (batch, length, hi), jnp.float32, 0.1, 1.0)
    ct = jax.random.normal(keys[6], (batch, length, hq, dim), jnp.float32)
    seg = jnp.asarray(np.repeat([[0] * 100 + [1] * 284], batch, axis=0), jnp.int32)

    def grads(q, k, v, q_idx, k_idx, w_idx, ct, seg):
        def loss(*a):
            o = dsa_layer(*a, top_k=32, indexer_loss_coef=1.0, segment_ids=seg, block_rows=256)
            return jnp.sum(o.astype(jnp.float32) * ct), o

        return jax.grad(loss, argnums=tuple(range(6)), has_aux=True)(q, k, v, q_idx, k_idx, w_idx)

    args = (q, k, v, q_idx, k_idx, w_idx, ct, seg)
    want = jax.jit(grads)(*args)
    with jax.set_mesh(mesh):
        sharded = jax.jit(grads, in_shardings=(rows,) * len(args))
        placed = [jax.device_put(x, rows) for x in args]
        hlo = sharded.lower(*placed).compile().as_text()
        got = sharded(*placed)
    gathered = [g for g in re.findall(r"all-gather[^\n]*", hlo) if "bf16[" in g]
    assert not gathered, gathered[:2]
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
        assert np.max(np.abs(g - w)) <= 1e-2 * max(np.max(np.abs(w)), 1e-6)


@gpu
def test_dsa_layer_with_top_k_covering_all_keys_is_causal_attention():
    """``top_k >= T`` selects every causal key of the document, so the same path runs DeepSeek-V3.2's dense
    warm-up stage (dense attention, indexer KL over all keys): the output is causal attention."""
    from jla.ops.mosaic_gpu.attention import flash_attention
    from jla.ops.mosaic_gpu.dsa import dsa_layer

    batch, length = 2, 300
    q, k, v, _, _ = _attention_inputs(5, batch, length, 8, 2, 64, 4)
    rng = np.random.default_rng(5)
    q_idx = jnp.asarray(rng.standard_normal((batch, length, 4, 64)), jnp.bfloat16)
    k_idx = jnp.asarray(rng.standard_normal((batch, length, 64)), jnp.bfloat16)
    w_idx = jnp.asarray(rng.uniform(0.1, 1.0, (batch, length, 4)), jnp.float32)
    seg = jnp.asarray(np.repeat([[0] * 120 + [1] * 180], batch, axis=0), jnp.int32)
    got = dsa_layer(q, k, v, q_idx, k_idx, w_idx, top_k=512, indexer_loss_coef=1.0, segment_ids=seg)
    want = flash_attention(q, k, v, segment_ids=seg)
    np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=2e-2)


@gpu
def test_selected_index_scores_keep_rows_with_small_gradients_accurate():
    """The score gradient is scaled to ``[-1, 1]`` per query row before its float16 rounding, so rows whose
    gradient is 1e-6 of the other rows' keep the accuracy of the large rows: ``dq`` and ``dw`` of each group
    of rows against the float32 gather form, relative to that group's own magnitude."""
    from jla.ops.mosaic_gpu.dsa import _index_topk, selected_index_scores, sparse_index_scores, valid_indices

    batch, length, heads, dim, top_k = 1, 1024, 8, 64, 128
    keys = jax.random.split(jax.random.key(7), 4)
    q = jax.random.normal(keys[0], (batch, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, dim), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(keys[2], (batch, length, heads), jnp.float32) * heads ** -0.5
    idx, values = jax.jit(lambda q, k, w: _index_topk(q, k, w, top_k))(q, k, w)
    valid = valid_indices(idx)
    small = np.arange(length) >= 64
    ct = jax.random.normal(keys[3], idx.shape, jnp.float32) * jnp.asarray(np.where(small, 1e-6, 1.0))[None, :, None]

    def objective(fn):
        return lambda q, k, w: jnp.sum(jnp.where(valid, fn(q, k, w), 0.0) * ct)

    kernel = objective(lambda q, k, w: selected_index_scores(q, k, w, idx, values, valid))
    got = jax.jit(jax.grad(kernel, argnums=(0, 2)))(q, k, w)
    with jax.default_matmul_precision("highest"):
        plain = objective(lambda q, k, w: sparse_index_scores(q, k, w, idx))
        want = jax.jit(jax.grad(plain, argnums=(0, 2)))(q.astype(jnp.float32), k.astype(jnp.float32), w)
    for name, a, b in zip(("dq", "dw"), got, want):
        a, b = np.asarray(a, np.float32)[0], np.asarray(b)[0]
        for rows in (small, ~small):
            err = np.max(np.abs(a[rows] - b[rows])) / np.max(np.abs(b[rows]))
            assert err < 1e-2, (name, "small rows" if rows[-1] else "large rows", err)


@gpu
def test_dsa_layer_auto_follows_an_xla_attention_pin():
    """``attention.implementation="xla"`` keeps a DSA layer with ``dsa_implementation="auto"`` on plain JAX."""
    from jla.layers.sparse_attention import DeepSeekSparseAttention

    sparse = SparseAttentionConfig(top_k=24, index_heads=4, index_head_dim=64)
    x = jax.random.normal(jax.random.key(1), (1, 160, 128), jnp.float32)
    pos = jnp.arange(160)[None]
    for implementation, kernels in (("auto", True), ("xla", False)):
        attention = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, implementation=implementation)
        layer = DeepSeekSparseAttention(128, attention, sparse, key=jax.random.key(0), std=0.3, norm_eps=1e-5,
                                        compute_dtype="bfloat16")
        q_idx, k_idx, w_idx = layer._index_inputs(x, pos)
        q, k, v = layer.attn._project(x, pos)
        assert layer._uses_kernels(q, k, v, q_idx, k_idx, w_idx) is kernels, implementation


def test_dsa_layer_rejects_attention_kernels_and_names_its_own_option():
    from jla.layers.sparse_attention import DeepSeekSparseAttention

    attention = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, implementation="cudnn")
    with pytest.raises(ValueError, match="dsa_implementation"):
        DeepSeekSparseAttention(64, attention, SparseAttentionConfig(top_k=8, index_heads=2, index_head_dim=64),
                                key=jax.random.key(0), std=0.3, norm_eps=1e-5)


@gpu
@pytest.mark.parametrize("scan,remat", [(True, "full"), (False, "none")])
def test_language_model_trains_through_the_dsa_kernels(monkeypatch, scan, remat):
    """A two-layer DSA language model with kernel-sized heads (64) and packed documents: under the default
    ``"auto"`` its layers run ``dsa_layer`` (also under scan over layers and remat), the loss matches the
    plain bf16 path, and the gradients of every parameter are as close to a float32 plain model as the
    plain bf16 path's are (largest error relative to each leaf's magnitude, over all leaves)."""
    import equinox as eqx

    from jla.models import build_model, get_preset
    from jla.ops.cross_entropy import linear_cross_entropy
    from jla.ops.mosaic_gpu import dsa as dsa_kernels

    calls = []
    dsa_layer = dsa_kernels.dsa_layer

    def counted(*args, **kwargs):
        calls.append(1)
        return dsa_layer(*args, **kwargs)

    monkeypatch.setattr(dsa_kernels, "dsa_layer", counted)
    base = get_preset("tiny-dsa")
    batch, length = 2, 300
    ids = jax.random.randint(jax.random.key(1), (batch, length), 0, base.vocab_size)
    seg = _segments(batch, length, True)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((batch, 1), -100)], axis=1).reshape(-1)

    def loss(model):
        out = model(ids, segment_ids=seg)
        ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(), targets,
                                  compute_dtype=jnp.float32)
        return ce.loss_sum / ce.num_valid + out.aux_loss

    results = {}
    for name, impl, dtype in (("f32", "xla", "float32"), ("xla", "xla", "bfloat16"), ("kernels", "auto", "bfloat16")):
        cfg = base.replace(
            attention=dataclasses.replace(base.attention, head_dim=64),
            sparse_attention=dataclasses.replace(base.sparse_attention, top_k=32, index_heads=4, index_head_dim=64,
                                                 dsa_implementation=impl),
            compute_dtype=dtype, scan_layers=scan, remat=remat)
        start = len(calls)
        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(build_model(cfg, key=jax.random.key(0)))
        assert (len(calls) > start) == (impl == "auto"), name  # the kernel path ran exactly where expected
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        results[name] = (float(value), [np.asarray(g, np.float32) for g in leaves])

    def worst(grads):
        pairs = zip(grads, results["f32"][1], strict=True)
        return max(np.max(np.abs(g - r)) / max(np.max(np.abs(r)), 1e-12) for g, r in pairs)

    assert abs(results["kernels"][0] - results["xla"][0]) <= 1e-3 * abs(results["xla"][0])
    err_kernels, err_xla = worst(results["kernels"][1]), worst(results["xla"][1])
    assert err_kernels <= 1.5 * err_xla + 1e-2, (err_kernels, err_xla)
