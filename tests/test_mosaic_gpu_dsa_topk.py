# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The radix-select top-k kernel (``jla.ops.mosaic_gpu.dsa_topk``) against an exact numpy top-k.

The reference is a stable numpy sort of each row (descending score, lower column first among equal
scores: the set that DeepSeek-V3.2's ``torch.topk`` and ``lax.top_k`` select), keeping finite entries
only (``+inf`` and NaN count as ``-inf``). Rows with heavy ties (integer scores), ``+0.0``/``-0.0``, ``-inf``
gaps, ``+inf`` and NaN entries, fewer finite entries than
``k``, ``k`` above the width and causal blocks with an ``offset`` are covered, with the candidate
compaction and with too few candidate slots (every pass over the full rows), and the DSA indexer's
``index_topk`` with ``method="radix"`` is compared with the sort path. The approximate FP8 indexer is
checked against the exact selection and in the layer. The kernel tests need an H100.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


def _reference(scores: np.ndarray, k: int) -> np.ndarray:
    """Ascending columns of each row's ``k`` largest finite entries (ties: lower column first), ``-1`` padded."""
    width = scores.shape[-1]
    order = np.argsort(-scores, axis=-1, kind="stable")[..., :k]
    finite = np.isfinite(np.take_along_axis(scores, order, axis=-1))
    chosen = np.sort(np.where(finite, order, width + 1), axis=-1)
    chosen = np.where(chosen > width, -1, chosen).astype(np.int32)
    if chosen.shape[-1] < k:
        pad = np.full(chosen.shape[:-1] + (k - chosen.shape[-1],), -1, np.int32)
        chosen = np.concatenate([chosen, pad], axis=-1)
    return chosen


def _causal(scores: np.ndarray, offset: int) -> np.ndarray:
    rows, width = scores.shape[1:]
    visible = np.arange(width)[None, :] <= offset + np.arange(rows)[:, None]
    return np.where(visible[None], scores, -np.inf).astype(np.float32)


def _check(scores: np.ndarray, k: int, **kwargs) -> None:
    from jla.ops.mosaic_gpu.dsa_topk import radix_top_k

    idx, val = jax.jit(lambda s: radix_top_k(s, k, **kwargs))(jnp.asarray(scores))
    idx, val = np.asarray(idx), np.asarray(val)
    scores = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)  # +inf and NaN are never selected
    big = np.iinfo(np.int32).max  # the slots list the keys in no global order; -1 slots come last
    ours = np.sort(np.where(idx < 0, big, idx), axis=-1)
    np.testing.assert_array_equal(np.where(ours == big, -1, ours), _reference(scores, k))
    gathered = np.take_along_axis(scores, np.maximum(idx, 0), axis=-1)
    np.testing.assert_array_equal(val, np.where(idx >= 0, gathered, -np.inf))


@gpu
@pytest.mark.parametrize("splits", [None, 1, 3])
@pytest.mark.parametrize("capacity", [128, 2048])
def test_radix_top_k_random_rows(splits, capacity):
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((2, 256, 1024)).astype(np.float32) * 10.0
    _check(scores, 100, splits=splits, capacity=capacity)
    scores = rng.standard_normal((1, 64, 8192)).astype(np.float32)
    _check(scores, 2048, capacity=capacity)  # 128: more candidates than slots, every pass reads the full rows


@gpu
def test_radix_top_k_wide_rows():
    rng = np.random.default_rng(6)
    scores = rng.standard_normal((1, 64, 65536)).astype(np.float32)
    _check(scores, 2048)  # default capacity W / 16 = 4096 candidate slots


@gpu
@pytest.mark.parametrize("capacity", [128, 2048])
def test_radix_top_k_heavy_ties_take_the_lower_columns(capacity):
    rng = np.random.default_rng(1)
    scores = rng.integers(-4, 5, (2, 128, 768)).astype(np.float32)
    _check(scores, 300, capacity=capacity)
    _check(scores, 1, capacity=capacity)
    zeros = np.zeros((1, 64, 256), np.float32)
    zeros[:, :, 1::2] = -0.0  # -0.0 ties +0.0
    _check(zeros, 37, capacity=capacity)


@gpu
def test_radix_top_k_extreme_values_and_gaps():
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((1, 128, 512)).astype(np.float32)
    scores *= np.float32(10.0) ** rng.integers(-30, 30, scores.shape).astype(np.float32)
    scores[:, :, ::7] = -np.inf
    scores[:, 3, :] = -np.inf  # a row without candidates
    scores[:, 5, :500] = -np.inf  # fewer finite entries than k
    scores[:, 9, 17] = np.finfo(np.float32).max
    scores[:, 9, 18] = -np.finfo(np.float32).max
    scores[:, 11, 40] = np.finfo(np.float32).tiny
    for capacity in (128, 2048):
        _check(scores, 64, capacity=capacity)
        _check(scores, 700, capacity=capacity)  # k above the width: every finite entry, then -1


@gpu
def test_radix_top_k_never_selects_inf_or_nan():
    rng = np.random.default_rng(7)
    scores = rng.standard_normal((1, 128, 1024)).astype(np.float32)
    scores[:, 0, 5] = np.inf  # an overflowing score: the row still gets its k largest finite entries
    scores[:, 1, ::3] = np.inf
    scores[:, 2, 7] = np.nan
    scores[:, 3, 100:400] = np.nan
    scores.view(np.int32)[:, 4, 9:20] = np.int32(-4194304)  # 0xFFC00000: NaN with the sign bit set
    scores[:, 5, :] = np.inf  # no finite entry: every slot -1
    scores[:, 6, :1000] = np.nan  # fewer finite entries than k
    scores[:, 7, 30:] = np.inf
    for capacity in (128, 2048):
        _check(scores, 64, capacity=capacity)
        _check(scores, 300, capacity=capacity)


@gpu
@pytest.mark.parametrize("offset", [0, 256])
def test_radix_top_k_causal_blocks(offset):
    rng = np.random.default_rng(3)
    rows = 256
    scores = _causal(rng.standard_normal((2, rows, offset + rows)).astype(np.float32), offset)
    _check(scores, 96, offset=offset)
    _check(scores, 2048, offset=offset)


@gpu
@pytest.mark.parametrize("length,packed", [(700, False), (1024, True), (9000, False)])
def test_index_topk_radix_matches_the_sort_path(length, packed):
    from jla.ops.mosaic_gpu.dsa import _index_topk

    keys = jax.random.split(jax.random.key(length), 3)
    heads, dim, top_k = 4, 64, 128
    q_idx = jax.random.normal(keys[0], (2, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k_idx = jax.random.normal(keys[1], (2, length, dim), jnp.float32).astype(jnp.bfloat16)
    w_idx = jax.random.normal(keys[2], (2, length, heads), jnp.float32)
    seg = None
    if packed:
        seg = np.zeros((2, length), np.int32)
        seg[:, length // 3:] = 1
        seg[-1, (2 * length) // 3:] = 2
        seg = jnp.asarray(seg)
    block = 4096 if length > 8192 else None
    run = jax.jit(lambda q, k, w, s, m: _index_topk(q, k, w, top_k, segment_ids=s, block_rows=block, method=m),
                  static_argnums=4)
    idx_sort, val_sort = (np.asarray(x) for x in run(q_idx, k_idx, w_idx, seg, "sort"))
    idx_radix, val_radix = (np.asarray(x) for x in run(q_idx, k_idx, w_idx, seg, "radix"))
    # the same keys (the sort path lists them by descending score, the radix path by column)
    big = np.iinfo(np.int32).max
    np.testing.assert_array_equal(np.sort(np.where(idx_radix < 0, big, idx_radix), axis=-1),
                                  np.sort(np.where(idx_sort < 0, big, idx_sort), axis=-1))
    np.testing.assert_array_equal(np.sort(val_radix, axis=-1), np.sort(val_sort, axis=-1))


def _recall(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Per row: the fraction of the exact selection's keys that the approximate selection also holds."""
    width = int(max(approx.max(), exact.max())) + 2
    rows = exact.reshape(-1, exact.shape[-1])
    hits = np.zeros((rows.shape[0], width), bool)
    np.put_along_axis(hits, np.where(approx.reshape(rows.shape) < 0, width - 1, approx.reshape(rows.shape)), True,
                      axis=-1)
    hits[:, width - 1] = False
    found = np.take_along_axis(hits, np.where(rows < 0, width - 1, rows), axis=-1)
    return found.sum(-1) / np.maximum((rows >= 0).sum(-1), 1)


@gpu
def test_index_topk_fp8_selects_nearly_the_exact_keys():
    from jla.ops.mosaic_gpu.dsa import _index_topk

    keys = jax.random.split(jax.random.key(5), 3)
    length, heads, dim, top_k = 2048, 16, 128, 256
    q_idx = jax.random.normal(keys[0], (1, length, heads, dim), jnp.float32).astype(jnp.bfloat16)
    k_idx = jax.random.normal(keys[1], (1, length, dim), jnp.float32).astype(jnp.bfloat16)
    w_idx = jax.random.normal(keys[2], (1, length, heads), jnp.float32) * heads ** -0.5
    run = jax.jit(lambda q, k, w, fp8: _index_topk(q, k, w, top_k, method="radix", fp8=fp8), static_argnums=3)
    exact, exact_val = (np.asarray(x) for x in run(q_idx, k_idx, w_idx, False))
    approx, approx_val = (np.asarray(x) for x in run(q_idx, k_idx, w_idx, True))
    causal = np.arange(length)[None, :, None]
    assert ((approx >= 0) == (exact >= 0)).all()  # the same number of keys per query
    assert ((approx <= causal) | (approx < 0)).all()
    recall = _recall(approx, exact)
    assert recall.mean() > 0.9, recall.mean()
    # the FP8 scores of the selected keys stay close to the exact ones (relative to each row's largest; e4m3 keeps
    # 3 mantissa bits: measured 0.105 at most here)
    scale = np.max(np.abs(np.where(exact >= 0, exact_val, 0)), axis=-1, keepdims=True)
    assert np.max(np.abs(np.sort(np.where(approx >= 0, approx_val, 0), axis=-1)
                         - np.sort(np.where(exact >= 0, exact_val, 0), axis=-1)) / scale) < 0.2


def _fp8_layers(**overrides):
    from jla.layers.sparse_attention import DeepSeekSparseAttention
    from jla.models.config import AttentionConfig, SparseAttentionConfig

    attention = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64)
    base = SparseAttentionConfig(top_k=64, index_heads=8, index_head_dim=64, **overrides)
    return [DeepSeekSparseAttention(64, attention, dataclasses.replace(base, dsa_index_fp8=fp8), key=jax.random.key(0),
                                    std=0.3, norm_eps=1e-5, compute_dtype="bfloat16") for fp8 in (False, True)]


def test_dsa_index_fp8_needs_the_kernel_path():
    _, layer = _fp8_layers(dsa_implementation="xla")
    x = jax.random.normal(jax.random.key(1), (1, 40, 64), jnp.float32)
    with pytest.raises(ValueError, match="dsa_index_fp8"):
        layer(x, jnp.arange(40)[None])


@gpu
def test_dsa_layer_with_the_fp8_indexer_stays_close_to_the_exact_layer():
    exact, fp8 = _fp8_layers(dsa_implementation="mosaic_gpu")
    x = jax.random.normal(jax.random.key(2), (2, 512, 64), jnp.float32)
    pos = jnp.broadcast_to(jnp.arange(512)[None], (2, 512))

    import equinox as eqx

    def loss(layer):
        return jnp.sum(layer(x, pos).astype(jnp.float32) ** 2)

    outs = [np.asarray(m(x, pos), np.float32) for m in (exact, fp8)]
    err = np.max(np.abs(outs[1] - outs[0])) / np.max(np.abs(outs[0]))
    mean_err = np.mean(np.abs(outs[1] - outs[0])) / np.mean(np.abs(outs[0]))
    grads = [[np.asarray(g, np.float32) for g in jax.tree.leaves(eqx.filter_grad(loss)(m))] for m in (exact, fp8)]
    grad_err = [float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30)) for a, b in zip(*grads)]
    print(f"fp8 layer: max err {err:.4f}, mean err {mean_err:.4f}, gradient rel. errors {np.round(grad_err, 4)}")
    # only keys near each query's selection threshold change; measured on H100 (19 Oct): max 0.172, mean 0.118,
    # gradients 0.014-0.142 (relative Frobenius error per parameter); the bounds keep about 1.5x of headroom
    assert err < 0.25, err
    assert mean_err < 0.18, mean_err
    assert all(np.isfinite(g).all() for g in grads[1])
    assert max(grad_err) < 0.22, grad_err
