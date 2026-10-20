# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU flash attention (``jla.ops.mosaic_gpu.attention``) against dense XLA attention.

Forward outputs and ``dq, dk, dv`` of the kernel are compared with
``jax.nn.dot_product_attention(implementation="xla")`` in fp32 on the same bf16 inputs with an
explicit dense mask (causal, sliding window, packed documents), for GQA, head dims 64 and 128
and lengths that are not a multiple of the block size. The default backward (``"ranged"``, block
skipping with two consumer warpgroups, also with the internal ``block_kv_dq``, ``overlap_dkv``,
``heavy_first`` and ``unmask_documents`` options) and the previous one (``"full"``) are both
checked, including segment ids that are sorted descending, unsorted but one run per id (block
skipping as for sorted ids) and reused across runs (no skipping), and GPT-OSS attention sinks
(with their gradient) against the layer's XLA sink path. Errors are checked globally and per row.
The forward's two loops (ping-pong, and the overlapped loop with two or three consumer warpgroups and its
internal variants) are checked against each other bitwise, outputs and gradients.
``implementation="auto"`` with packed documents picks the kernel on Hopper (checked through the
layer, values and gradients, bf16 and fp16). With the batch or the heads sharded over two GPUs
under ``jax.jit`` the kernels run per device without gathering their operands (MQA heads that do
not divide are gathered), and inside ``jax.shard_map`` they give the single-device values and
gradients. The kernel tests need an H100 (the sharding tests two); the block-bound helpers are
checked on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.attention import attention


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


def _dense_reference(q, k, v, segment_ids, sliding_window, scale):
    """fp32 attention with the mask written out position by position."""
    length = q.shape[1]
    pos = np.arange(length)
    mask = pos[:, None] >= pos[None, :]
    if sliding_window is not None:
        mask = mask & (pos[:, None] - pos[None, :] < sliding_window)
    mask = jnp.asarray(mask)[None, None]
    if segment_ids is not None:
        mask = mask & (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]
    f32 = lambda x: x.astype(jnp.float32)  # noqa: E731
    return jax.nn.dot_product_attention(f32(q), f32(k), f32(v), mask=mask, scale=scale, implementation="xla")


def _segments(rng, batch, length, docs, order="sorted"):
    """Packed documents; ``order`` ``"descending"`` reverses the ids, ``"shuffled"`` gives the
    documents distinct ids in no order (corpus-style ids: every id still one run, from 3 documents
    not monotone), ``"interleaved"`` reuses ids across runs (every id occurs in several runs, so
    the kernels must not skip blocks)."""
    ids = np.zeros((batch, length), np.int32)
    for b in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=docs - 1, replace=False))
        run = np.searchsorted(cuts, np.arange(length), side="right")
        if order == "descending":
            run = docs - 1 - run
        elif order == "shuffled":  # 0, 856, 703, 550, ...: distinct for up to 1009 documents
            run = (run * 856) % 1009
        elif order == "interleaved":
            run = run % 2
        ids[b] = run + 10 * b
    return jnp.asarray(ids)


def _inputs(seed, batch, length, heads, kv_heads, head_dim, dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(keys[0], (batch, length, heads, head_dim), jnp.float32).astype(dtype)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, head_dim), jnp.float32).astype(dtype)
    do = jax.random.normal(keys[3], (batch, length, heads, head_dim), jnp.float32)
    return q, k, v, do


def _rel_err(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    # The floor keeps exact zeros (dq of a one-token window) from turning rounding noise into failures.
    return float(np.max(np.abs(actual - expected)) / (np.max(np.abs(expected)) + 1e-3))


def _assert_matches_one_device(got, want, max_fraction=1e-3):
    """A sharded run against a separately compiled one-device run: fp32 leaves to 1e-5; bf16 leaves
    equal except for a few elements. The two programs compile the XLA-side fp32 reduction
    ``D = rowsum(o * do)`` separately, and XLA's autotuning may sum it in another order in each (as in
    ``test_head_sharded_mqa_gathers_the_heads``): a slightly different ``D`` moves a few gradient
    elements to a neighbouring bf16 value of the terms they sum (measured: up to 86 of 2,097,152
    elements). Bound: at most ``max_fraction`` of the elements differ, by less than 2e-3 of the
    largest value (bf16 resolution, the bound of that test)."""
    if got.dtype != jnp.bfloat16:
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=1e-5, atol=1e-5)
        return
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    off = ~np.isclose(g, w, rtol=1e-5, atol=1e-5)
    if not off.any():
        return
    detail = dict(differ=int(off.sum()), size=off.size, max_abs_diff=float(np.max(np.abs(g - w))),
                  max_abs=float(np.max(np.abs(w))), rel=_rel_err(g, w))
    assert off.mean() <= max_fraction and _rel_err(g, w) < 2e-3, detail


def _row_rel_err(actual, expected):
    """Worst error of one row (batch, position, head) over that row's magnitude.

    The global ``_rel_err`` is dominated by early rows that see few keys, so a key block leaked
    into (or dropped from) a few rows deep inside a long document could pass it; this metric
    catches that. The floor (5% of the median row) keeps near-zero rows from failing on noise.
    Meant for ``out`` and ``dv``: rows of ``dq`` and ``dk`` cancel to near zero, and there FA3,
    cuDNN and this kernel all reach 1.2-1.5 (benchmarks/attention.md).
    """
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    magnitude = np.abs(expected).max(axis=-1)
    floor = 0.05 * float(np.median(magnitude)) + 1e-6
    return float((np.abs(actual - expected).max(axis=-1) / (magnitude + floor)).max())


CASES = [
    # batch, length, heads, kv_heads, head_dim, sliding_window, docs
    pytest.param(2, 256, 2, 2, 128, None, 0, id="causal-d128"),
    pytest.param(1, 384, 4, 2, 64, None, 0, id="causal-gqa-d64"),
    pytest.param(1, 200, 2, 1, 128, None, 0, id="causal-odd-length-gqa"),
    pytest.param(1, 512, 2, 2, 128, 100, 0, id="swa-100"),
    pytest.param(1, 384, 2, 1, 64, 1, 0, id="swa-1"),
    # Windows with (1 - W) % 128 >= 64: the second consumer warpgroup starts one key block later
    # than the first and skips that block (the prologue path of the forward).
    pytest.param(1, 512, 2, 1, 128, 40, 0, id="swa-40-staggered-start"),
    pytest.param(2, 640, 2, 2, 64, 33, 3, id="swa-33-documents-staggered-start"),
    pytest.param(2, 384, 2, 2, 128, None, 5, id="documents"),
    pytest.param(1, 333, 4, 1, 128, 70, 4, id="swa-documents-gqa-odd"),
    # Several 128-row tiles: the ranged backward skips blocks and masks only boundary blocks.
    pytest.param(1, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(2, 768, 8, 2, 64, None, 0, id="causal-768-gqa4-d64"),
    pytest.param(1, 1024, 2, 2, 128, 200, 0, id="swa-200-1024"),
    pytest.param(1, 1024, 2, 2, 64, 129, 0, id="swa-129-1024-d64"),
    pytest.param(2, 1024, 4, 2, 128, None, 9, id="documents-9-1024-gqa2"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    # Long documents: most blocks lie inside one document and skip the mask.
    pytest.param(1, 2048, 2, 1, 128, None, 2, id="documents-2-2048-gqa2"),
    pytest.param(2, 1536, 2, 2, 64, 300, 3, id="swa-300-documents-3-1536-d64"),
]


def _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs, *,
                         order="sorted", backward=None, implementation="mosaic_gpu", dtype=jnp.bfloat16):
    rng = np.random.default_rng(length + docs)
    q, k, v, do = _inputs(length, batch, length, heads, kv_heads, head_dim, dtype)
    seg = _segments(rng, batch, length, docs, order) if docs else None
    scale = head_dim**-0.5

    def loss(fn, q, k, v):
        out = fn(q, k, v)
        return jnp.sum(out.astype(jnp.float32) * do), out

    if backward is None:  # through the layer, default backward
        kernel = lambda q, k, v: attention(  # noqa: E731
            q, k, v, segment_ids=seg, sliding_window=window, scale=scale, implementation=implementation)
    else:
        from jla.ops.mosaic_gpu.attention import flash_attention

        kernel = lambda q, k, v: flash_attention(  # noqa: E731
            q, k, v, segment_ids=seg, sliding_window=window, scale=scale, backward=backward)
    dense = lambda q, k, v: _dense_reference(q, k, v, seg, window, scale)  # noqa: E731

    grads = jax.jit(jax.grad(lambda *a: loss(kernel, *a), argnums=(0, 1, 2), has_aux=True))
    ref_grads = jax.jit(jax.grad(lambda *a: loss(dense, *a), argnums=(0, 1, 2), has_aux=True))
    (dq, dk, dv), out = grads(q, k, v)
    (rdq, rdk, rdv), ref = ref_grads(q, k, v)
    assert out.dtype == q.dtype and out.shape == q.shape
    assert _rel_err(out, ref) < 2e-2
    for name, got, want in (("dq", dq, rdq), ("dk", dk, rdk), ("dv", dv, rdv)):
        assert got.shape == want.shape, name
        assert _rel_err(got, want) < 3e-2, (name, _rel_err(got, want))
    # Per row (measured: out <= 0.006, dv <= 0.01 for this kernel, FA3 and cuDNN alike).
    assert _row_rel_err(out, ref) < 2e-2, ("out rows", _row_rel_err(out, ref))
    assert _row_rel_err(dv, rdv) < 4e-2, ("dv rows", _row_rel_err(dv, rdv))


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", CASES)
def test_forward_and_gradients_match_dense_xla(batch, length, heads, kv_heads, head_dim, window, docs):
    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs)


@gpu
@pytest.mark.parametrize("order", ["interleaved", "descending", "shuffled"])
@pytest.mark.parametrize("length,window", [(1024, None), (640, 150)])
def test_unsorted_segment_ids_stay_exact(order, length, window):
    _check_against_dense(1, length, 4, 2, 128, window, 7, order=order)


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 384, 4, 2, 64, None, 0, id="causal-gqa-d64"),
    pytest.param(1, 333, 4, 1, 128, 70, 4, id="swa-documents-gqa-odd"),
    pytest.param(1, 1024, 4, 2, 128, None, 9, id="documents-9-1024-gqa2"),
])
def test_previous_full_backward_still_matches(batch, length, heads, kv_heads, head_dim, window, docs):
    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs, backward="full")


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", CASES)
def test_fused_backward_matches_dense_xla(batch, length, heads, kv_heads, head_dim, window, docs):
    """``backward="fused"``: one kernel for ``dq, dk, dv`` (``dq`` added up in fp32 with TMA
    reductions; ``D = 128`` splits its columns over the two warpgroups, ``D = 64`` alternates them)."""
    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs, backward="fused")


@gpu
@pytest.mark.parametrize("order", ["interleaved", "descending", "shuffled"])
@pytest.mark.parametrize("length,window,head_dim", [(1024, None, 128), (640, 150, 64)])
def test_fused_backward_unsorted_segment_ids_stay_exact(order, length, window, head_dim):
    _check_against_dense(1, length, 4, 2, head_dim, window, 7, order=order, backward="fused")


def test_attention_config_picks_the_mosaic_gpu_backward():
    from jla.models.config import AttentionConfig

    assert AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64).mosaic_gpu_backward == "ranged"
    assert AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, mosaic_gpu_backward="fused")
    with pytest.raises(ValueError, match="mosaic_gpu_backward"):
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, mosaic_gpu_backward="full")


@gpu
def test_layer_routes_the_fused_backward():
    """``attention(..., mosaic_gpu_backward="fused")`` (the ``AttentionConfig`` field the layer passes)
    runs the fused backward: the same gradients as ``"ranged"`` up to the order of ``dq``'s fp32 adds,
    and one backward kernel instead of two in the compiled program."""
    q, k, v, do = _inputs(11, 2, 512, 8, 2, 64)
    seg = _segments(np.random.default_rng(11), 2, 512, 3)

    def grads(backward):
        def loss(q, k, v):
            out = attention(q, k, v, segment_ids=seg, sliding_window=200, implementation="mosaic_gpu",
                            mosaic_gpu_backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

    fused, ranged = grads("fused"), grads("ranged")
    for name, a, b in zip(("dq", "dk", "dv"), fused(q, k, v), ranged(q, k, v)):
        assert _rel_err(a, b) < 4e-3, (name, _rel_err(a, b))

    def kernels(fn):
        hlo = fn.lower(q, k, v).compile().as_text()
        return len([line for line in hlo.splitlines() if 'custom_call_target="mosaic_gpu' in line])

    assert kernels(fused) == kernels(ranged) - 1  # dq and dk, dv in one kernel


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(2, 768, 8, 2, 64, None, 0, id="causal-768-gqa4-d64"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    pytest.param(2, 1536, 2, 2, 64, 300, 3, id="swa-300-documents-3-1536-d64"),
    pytest.param(1, 200, 2, 1, 128, 1, 0, id="swa-1-odd"),
])
@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "serial"])
def test_fused_backward_equals_ranged(monkeypatch, overlap, batch, length, heads, kv_heads, head_dim, window, docs):
    """The fused kernel runs the ``dk, dv`` arithmetic of the ranged one: ``dk, dv`` are bitwise equal,
    and ``dq`` differs only by the order of its fp32 additions (within bf16 rounding). Both step orders
    of the fused kernel (``fused_overlap``) are checked."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module
    from jla.ops.mosaic_gpu.attention import flash_attention

    monkeypatch.setattr(kernel_module, "_Config", functools.partial(kernel_module._Config, fused_overlap=overlap))
    q, k, v, do = _inputs(length + 3, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length), batch, length, docs) if docs else None

    def grads(backward):
        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=window, backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)

    (fdq, fdk, fdv), (rdq, rdk, rdv) = grads("fused"), grads("ranged")
    np.testing.assert_array_equal(np.asarray(fdk, np.float32), np.asarray(rdk, np.float32))
    np.testing.assert_array_equal(np.asarray(fdv, np.float32), np.asarray(rdv, np.float32))
    assert _rel_err(fdq, rdq) < 4e-3, _rel_err(fdq, rdq)


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,window,docs", [
    pytest.param(2, 768, 8, 2, None, 0, id="causal-768-gqa4"),
    pytest.param(1, 1000, 4, 4, None, 3, id="documents-3-odd"),
    pytest.param(2, 1536, 2, 2, 300, 3, id="swa-300-documents-3"),
])
@pytest.mark.parametrize("async_ds", [True, False], ids=["async-ds", "meet"])
@pytest.mark.parametrize("kv_seq_first", [False, True], ids=["heads-first", "kv-seq-first"])
def test_fused_d64_step_options_equal_ranged(monkeypatch, async_ds, kv_seq_first, batch, length, heads, kv_heads,
                                             window, docs):
    """``backward="fused"`` with ``D = 64``: the per-parity ``dS^T`` barriers (``fused_async_ds``) and the key-block
    first grid (``fused_kv_seq_first``) change only when warpgroups and CTAs run, not the arithmetic: ``dk, dv``
    stay bitwise the ranged kernel's and ``dq`` agrees to the order of its fp32 additions."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module
    from jla.ops.mosaic_gpu.attention import flash_attention

    q, k, v, do = _inputs(length + 5, batch, length, heads, kv_heads, 64)
    seg = _segments(np.random.default_rng(length), batch, length, docs) if docs else None

    def grads(backward, **options):
        monkeypatch.setattr(kernel_module, "_Config", functools.partial(config, **options))

        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=window, backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)

    config = kernel_module._Config
    (fdq, fdk, fdv) = grads("fused", fused_async_ds=async_ds, fused_kv_seq_first=kv_seq_first)
    (rdq, rdk, rdv) = grads("ranged")
    np.testing.assert_array_equal(np.asarray(fdk, np.float32), np.asarray(rdk, np.float32))
    np.testing.assert_array_equal(np.asarray(fdv, np.float32), np.asarray(rdv, np.float32))
    assert _rel_err(fdq, rdq) < 4e-3, _rel_err(fdq, rdq)


@pytest.mark.parametrize("tiles,pairs,section", [(32, 8, 16), (32, 64, 16), (5, 7, 2), (3, 6, 4), (4, 1, 1),
                                                 (6, 5, 1)])
def test_backward_launch_order_is_a_longest_first_permutation(tiles, pairs, section):
    """``_lpt_order`` maps the 1D grid onto every (rank, pair) once; inside a section ranks never decrease, a
    section's pairs are consecutive, and the sections (the last one narrower) follow one another."""
    from jla.ops.mosaic_gpu.attention import _lpt_order

    rank, pair = (np.asarray(x) for x in _lpt_order(jnp.arange(tiles * pairs, dtype=jnp.int32), tiles, pairs,
                                                     section))
    assert sorted(zip(rank.tolist(), pair.tolist())) == [(r, p) for r in range(tiles) for p in range(pairs)]
    sec = pair // section
    assert np.all(np.diff(sec) >= 0)
    for s in np.unique(sec):
        assert np.all(np.diff(rank[sec == s]) >= 0)


@gpu
@pytest.mark.parametrize("section", [None, 1, 4], ids=["l2-sections", "one-pair-sections", "four-pair-sections"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(3, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(2, 768, 8, 2, 64, None, 0, id="causal-768-gqa4-d64"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    pytest.param(3, 1000, 2, 2, 64, None, 3, id="documents-3-odd-d64"),
    pytest.param(2, 1000, 4, 4, 128, None, 0, id="causal-odd-1000-d128"),
])
def test_backward_launch_order_is_exact(monkeypatch, section, batch, length, heads, kv_heads, head_dim, window,
                                        docs):
    """The longest-first 1D grid of the backward kernels (``bwd_lpt``, with one section or several and a narrower
    last one) changes only when CTAs start: the ranged gradients are bitwise those of the (heads, blocks, batch)
    grids, and the fused kernel's ``dk, dv`` too (its ``dq`` agrees to the order of its fp32 additions)."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module
    from jla.ops.mosaic_gpu.attention import flash_attention

    if section is not None:
        monkeypatch.setattr(kernel_module, "_l2_section", lambda pair_bytes: section)
    q, k, v, do = _inputs(length + 7, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length), batch, length, docs) if docs else None
    config = kernel_module._Config

    def grads(backward, **options):
        monkeypatch.setattr(kernel_module, "_Config", functools.partial(config, **options))

        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=window, backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)

    for new, old in zip(grads("ranged"), grads("ranged", bwd_lpt=False)):
        np.testing.assert_array_equal(np.asarray(new, np.float32), np.asarray(old, np.float32))
    (fdq, fdk, fdv), (odq, odk, odv) = grads("fused"), grads("fused", bwd_lpt=False)
    np.testing.assert_array_equal(np.asarray(fdk, np.float32), np.asarray(odk, np.float32))
    np.testing.assert_array_equal(np.asarray(fdv, np.float32), np.asarray(odv, np.float32))
    assert _rel_err(fdq, odq) < 4e-3, _rel_err(fdq, odq)


@gpu
@pytest.mark.parametrize("options", [
    pytest.param(dict(block_kv_dq=128), id="dq128"),
    pytest.param(dict(overlap_dkv=True), id="overlap"),
    pytest.param(dict(block_kv_dq=128, overlap_dkv=True), id="dq128-overlap"),
    pytest.param(dict(block_kv_dq=64, overlap_dkv=False), id="dq64-serial"),  # the earlier defaults
    pytest.param(dict(heavy_first=False), id="in-order"),
    pytest.param(dict(heavy_first=True), id="heavy-first"),
    pytest.param(dict(unmask_documents=False), id="documents-always-masked"),
])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(1, 1024, 2, 2, 64, 129, 0, id="swa-129-1024-d64"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    pytest.param(2, 1024, 4, 2, 128, None, 3, id="documents-3-1024-gqa2"),
])
def test_kernel_options_match(monkeypatch, options, batch, length, heads, kv_heads, head_dim, window, docs):
    """Internal kernel options: 128- and 64-key ``dq`` blocks, the overlapped ``dk, dv`` body (on and off), the
    query-tile launch order of the forward and ``dq`` kernels (both values) and the document-interior
    mask skipping turned off."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    monkeypatch.setattr(kernel_module, "_Config", functools.partial(kernel_module._Config, **options))
    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs, backward="ranged")


@gpu
@pytest.mark.parametrize("sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", CASES + [
    pytest.param(1, 128, 2, 1, 64, None, 0, id="one-block"),
    pytest.param(1, 4096, 2, 1, 128, 1000, 7, id="swa-1000-documents-7-4096"),
    pytest.param(1, 2048, 2, 2, 64, 300, 0, id="swa-300-2048-d64"),
])
def test_forward_overlap_equals_pingpong(monkeypatch, sinks, batch, length, heads, kv_heads, head_dim, window,
                                         docs):
    """The overlapped forward with two warpgroups (``fwd_overlap=True, fwd_wgs=2``: ``QK^T`` of block ``j``
    issued with the ``PV`` product of block ``j - 1``, softmax under that product, turn passing between the
    two warpgroups) does the arithmetic of the ping-pong forward in the same order: the output
    and every gradient through the deterministic ranged backward (which reads the forward's ``lse``)
    are bitwise equal, for every mask case, one-block sequences, long documents with a window, and
    sinks."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    original = kernel_module._Config
    q, k, v, do = _inputs(length + 5, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length + 1), batch, length, docs) if docs else None
    sink = jnp.linspace(-2.0, 4.0, heads, dtype=jnp.float32) if sinks else None

    def run(overlap):
        monkeypatch.setattr(kernel_module, "_Config", functools.partial(original, fwd_overlap=overlap, fwd_wgs=2))

        def loss(q, k, v):
            out = kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window, sinks=sink)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)

    (grads_new, out_new), (grads_old, out_old) = run(True), run(False)
    np.testing.assert_array_equal(np.asarray(out_new, np.float32), np.asarray(out_old, np.float32))
    for name, new, old in zip(("dq", "dk", "dv"), grads_new, grads_old):
        np.testing.assert_array_equal(np.asarray(new, np.float32), np.asarray(old, np.float32), err_msg=name)


@gpu
@pytest.mark.parametrize("options", [
    pytest.param(dict(fwd_named_turn=False, fwd_named_consumed=False), id="mbarriers"),
    pytest.param(dict(fwd_named_turn=False), id="mbarrier-turn"),
    pytest.param(dict(fwd_k_ahead=True), id="keys-ahead"),
    pytest.param(dict(fwd_k_ahead=True, max_concurrent_steps=3), id="keys-ahead-3-stages"),
    pytest.param(dict(max_concurrent_steps=3), id="3-stages"),
])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 128, 2, 1, 64, None, 0, id="one-block"),
    pytest.param(1, 384, 4, 2, 64, None, 0, id="causal-gqa-d64"),
    pytest.param(2, 640, 2, 2, 64, 33, 3, id="swa-33-documents-staggered-start"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    pytest.param(1, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(1, 4096, 2, 1, 128, 1000, 7, id="swa-1000-documents-7-4096"),
])
def test_forward_variants_equal_pingpong(monkeypatch, options, batch, length, heads, kv_heads, head_dim, window,
                                         docs):
    """The overlapped forward's internal variants (mbarrier turns and slot releases instead of named
    barriers, keys loaded one block ahead of values, three stages) give the ping-pong loop's output and
    gradients bitwise."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    if options.get("max_concurrent_steps", 2) > 2 and head_dim == 128:
        pytest.skip("three stages of 128 x 128 key and value tiles do not fit in shared memory with D = 128")
    original = kernel_module._Config
    q, k, v, do = _inputs(length + 9, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length + 3), batch, length, docs) if docs else None

    def run(**config):
        monkeypatch.setattr(kernel_module, "_Config", functools.partial(original, **config))

        def loss(q, k, v):
            out = kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)

    (grads_new, out_new), (grads_old, out_old) = run(fwd_overlap=True, **options), run(fwd_overlap=False)
    np.testing.assert_array_equal(np.asarray(out_new, np.float32), np.asarray(out_old, np.float32))
    for name, new, old in zip(("dq", "dk", "dv"), grads_new, grads_old):
        np.testing.assert_array_equal(np.asarray(new, np.float32), np.asarray(old, np.float32), err_msg=name)


@gpu
@pytest.mark.parametrize("options", [
    pytest.param(dict(heavy_first=False), id="in-order"),
    pytest.param(dict(heavy_first=True, fwd_k_ahead=True), id="heavy-first-keys-ahead"),
])
@pytest.mark.parametrize("sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    *(case for case in CASES if case.values[4] == 64),
    pytest.param(1, 128, 2, 1, 64, None, 0, id="one-block"),
    pytest.param(1, 200, 2, 1, 64, 50, 0, id="swa-50-odd-200"),
    pytest.param(1, 1024, 2, 2, 64, None, 5, id="documents-5-1024"),
    pytest.param(1, 4096, 4, 1, 64, 1000, 7, id="swa-1000-documents-7-4096-gqa4"),
    pytest.param(2, 1152, 8, 2, 64, None, 0, id="causal-1152-gqa4"),
])
def test_forward_three_warpgroups_equal_two(monkeypatch, options, sinks, batch, length, heads, kv_heads, head_dim,
                                            window, docs):
    """``fwd_wgs=3`` (FA3's 192-row query tiles for ``D = 64``: three consumer warpgroups take turns;
    the last tile may end inside the sequence, as at T = 128, 256, 640, 1024 and 4096, where the
    warpgroups past its end only take their turns) computes every row as the two-warpgroup forward does:
    outputs and the ranged backward's gradients are bitwise equal."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    original = kernel_module._Config
    q, k, v, do = _inputs(length + 7, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length + 2), batch, length, docs) if docs else None
    sink = jnp.linspace(-2.0, 4.0, heads, dtype=jnp.float32) if sinks else None

    def run(wgs):
        config = functools.partial(original, fwd_overlap=True, fwd_wgs=wgs, **options)
        monkeypatch.setattr(kernel_module, "_Config", config)

        def loss(q, k, v):
            out = kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window, sinks=sink)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)

    (grads_new, out_new), (grads_old, out_old) = run(3), run(2)
    np.testing.assert_array_equal(np.asarray(out_new, np.float32), np.asarray(out_old, np.float32))
    for name, new, old in zip(("dq", "dk", "dv"), grads_new, grads_old):
        np.testing.assert_array_equal(np.asarray(new, np.float32), np.asarray(old, np.float32), err_msg=name)


@gpu
@pytest.mark.parametrize("backward", ["ranged", "full", "fused"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 384, 4, 2, 64, None, 0, id="causal-gqa-d64"),
    pytest.param(1, 333, 4, 1, 128, 70, 4, id="swa-documents-gqa-odd"),
    pytest.param(2, 1024, 4, 2, 128, None, 9, id="documents-9-1024-gqa2"),
])
def test_sinks_match_the_xla_sink_path(batch, length, heads, kv_heads, head_dim, window, docs, backward):
    """GPT-OSS sinks: outputs and ``dq, dk, dv, dsinks`` against the layer's fp32 XLA sink path."""
    from jla.ops.mosaic_gpu.attention import flash_attention

    q, k, v, do = _inputs(length + 1, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length), batch, length, docs) if docs else None
    sinks = jnp.linspace(-2.0, 4.0, heads, dtype=jnp.float32)
    scale = head_dim**-0.5
    f32 = lambda x: x.astype(jnp.float32)  # noqa: E731

    def loss(fn, q, k, v, sinks):
        out = fn(q, k, v, sinks)
        return jnp.sum(out.astype(jnp.float32) * do), out

    kernel = lambda q, k, v, s: flash_attention(  # noqa: E731
        q, k, v, segment_ids=seg, sliding_window=window, scale=scale, backward=backward, sinks=s)
    xla = lambda q, k, v, s: attention(  # noqa: E731
        f32(q), f32(k), f32(v), segment_ids=seg, sliding_window=window, scale=scale, sinks=s,
        implementation="xla")
    argnums = (0, 1, 2, 3)
    got, out = jax.jit(jax.grad(lambda *a: loss(kernel, *a), argnums=argnums, has_aux=True))(q, k, v, sinks)
    want, ref = jax.jit(jax.grad(lambda *a: loss(xla, *a), argnums=argnums, has_aux=True))(q, k, v, sinks)
    assert _rel_err(out, ref) < 2e-2
    for name, g, w in zip(("dq", "dk", "dv", "dsinks"), got, want):
        assert g.shape == w.shape, name
        assert _rel_err(g, w) < 3e-2, (name, _rel_err(g, w))
    # The layer routes sinks to the kernel for implementation="mosaic_gpu" (forward only here).
    layer = jax.jit(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=window, scale=scale,
                                              sinks=sinks, implementation="mosaic_gpu"))(q, k, v)
    if backward == "ranged":
        assert _rel_err(layer, ref) < 2e-2


@pytest.mark.parametrize("order", ["sorted", "descending", "shuffled", "interleaved"])
def test_segment_run_bounds_cover_every_visible_key(order):
    """CPU: every key with a query's id lies inside the query's ``[start, end]`` run bounds, and
    the bounds are exactly the document whenever every id forms one run (sorted or not)."""
    from jla.ops.mosaic_gpu.attention import _contiguous_ids, _segment_runs

    seg = _segments(np.random.default_rng(5), 3, 300, 8, order)
    runs = np.asarray(_segment_runs(seg, _contiguous_ids(seg)))
    ids = np.asarray(seg)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            same = np.nonzero(ids[b] == ids[b, t])[0]
            assert runs[b, 0, t] <= same.min() and runs[b, 1, t] >= same.max(), (b, t)
            if order != "interleaved":
                assert (runs[b, 0, t], runs[b, 1, t]) == (same.min(), same.max()), (b, t)
        assert np.all(np.diff(runs[b, 0]) >= 0) and np.all(np.diff(runs[b, 1]) >= 0)
    contiguous = np.asarray(_contiguous_ids(seg))
    assert not np.any(contiguous) if order == "interleaved" else np.all(contiguous)
    if order == "shuffled":  # unsorted ids skip exactly the blocks sorted ids skip
        ordered = _segments(np.random.default_rng(5), 3, 300, 8, "sorted")
        assert np.array_equal(runs, np.asarray(_segment_runs(ordered, _contiguous_ids(ordered))))


def test_contiguous_ids_is_exact():
    """CPU: a row skips blocks exactly when each of its ids forms one run (number of runs equals
    the number of distinct ids), whatever the order of the ids."""
    from jla.ops.mosaic_gpu.attention import _contiguous_ids

    rows = {
        (8812, 8812, 77, 77, 5021, 5021): True,  # corpus ids, unsorted
        (3, 3, 1, 1, 1, 2): True,
        (1, 1, 2, 2, 0, 0): True,  # padding id 0 after the documents
        (5, 5, 5, 5, 5, 5): True,
        (-4, -4, 7, 7, -9, -9): True,
        (1, 1, 2, 2, 1, 1): False,  # id 1 occurs in two runs
        (5, 5, 7, 5, 5, 5): False,
        (0, 1, 0, 1, 0, 1): False,
    }
    seg = jnp.asarray(list(rows), jnp.int32)
    assert np.asarray(_contiguous_ids(seg)).tolist() == list(rows.values())
    one = jnp.zeros((2, 1), jnp.int32)  # a single token is one run
    assert np.asarray(_contiguous_ids(one)).tolist() == [True, True]


@gpu
def test_layer_mosaic_gpu_forward_matches_layer_xla():
    q, k, v, _ = _inputs(7, 1, 512, 4, 2, 128)
    seg = _segments(np.random.default_rng(3), 1, 512, 3)
    ours = jax.jit(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=96,
                                             implementation="mosaic_gpu"))(q, k, v)
    xla = jax.jit(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=96,
                                            implementation="xla"))(q, k, v)
    assert _rel_err(ours, xla) < 2e-2


def test_backward_row_sums_are_exact_fp32():
    """``D = rowsum(out * do)`` of the backward must stay exact fp32 (not a TF32 dot on GPU; JAX
    main's ``attention_mgpu.py`` now raises the precision of the same contraction)."""
    from jla.ops.mosaic_gpu.attention import _delta

    rng = np.random.default_rng(0)
    out = rng.standard_normal((2, 256, 4, 128)).astype(np.float32)
    do = rng.standard_normal((2, 256, 4, 128)).astype(np.float32)
    got = np.asarray(jax.jit(_delta)(jnp.asarray(out), jnp.asarray(do)))
    want = np.einsum("bqhd,bqhd->bhq", out.astype(np.float64), do.astype(np.float64))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


@gpu
def test_auto_picks_the_kernel_for_packed_documents_on_hopper():
    from jla.layers.attention import resolve_attention_implementation

    q = jnp.zeros((1, 256, 4, 128), jnp.bfloat16)
    k = jnp.zeros((1, 256, 2, 128), jnp.bfloat16)
    seg = jnp.zeros((1, 256), jnp.int32)
    assert resolve_attention_implementation("auto", q, k, v=k, segment_ids=seg) == "mosaic_gpu"
    odd = (q[:, :255], k[:, :255])  # the kernel pads odd lengths (the cuDNN backward rejects them)
    assert resolve_attention_implementation("auto", *odd, v=odd[1], segment_ids=seg[:, :255]) == "mosaic_gpu"
    assert resolve_attention_implementation("auto", q, k, v=k) == "cudnn"  # no documents: cuDNN
    q96, k96 = jnp.zeros((1, 256, 4, 96), jnp.bfloat16), jnp.zeros((1, 256, 2, 96), jnp.bfloat16)
    assert resolve_attention_implementation("auto", q96, k96, v=k96, segment_ids=seg) == "cudnn"  # no D = 96 kernel
    f32 = (q.astype(jnp.float32), k.astype(jnp.float32))
    assert resolve_attention_implementation("auto", *f32, v=f32[1], segment_ids=seg) == "xla"
    assert resolve_attention_implementation("cudnn", q, k, v=k, segment_ids=seg) == "cudnn"  # explicit choice kept


def test_auto_keeps_xla_for_packed_documents_without_a_gpu():
    from jla.layers.attention import resolve_attention_implementation

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only check")
    q = jnp.zeros((1, 256, 4, 128), jnp.bfloat16)
    seg = jnp.zeros((1, 256), jnp.int32)
    assert resolve_attention_implementation("auto", q, q, v=q, segment_ids=seg) == "xla"


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(2, 384, 2, 2, 128, None, 5, id="documents"),
    pytest.param(1, 333, 4, 1, 128, 70, 4, id="swa-documents-gqa-odd"),
    pytest.param(2, 1024, 4, 2, 64, None, 9, id="documents-9-1024-gqa2-d64"),
])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bf16", "fp16"])
def test_auto_with_documents_matches_dense_xla(batch, length, heads, kv_heads, head_dim, window, docs, dtype):
    """``implementation="auto"`` with ``segment_ids`` now runs the kernel (bf16 and fp16, both of
    which it routes there): values and gradients."""
    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs, implementation="auto",
                         dtype=dtype)


@gpu
@pytest.mark.parametrize("backward", ["ranged", "full", "fused"])
def test_backward_uses_the_rounded_cotangent(backward):
    """The loss is taken in fp32 against loss weights passed to the jitted function, so the bf16
    cotangent of the kernel output is rounded from fp32 inside the same program. ``D = rowsum(o * do)``
    must use the rounded values the kernels read: the gradients equal those for weights rounded
    beforehand, and a query that sees only itself (each document start) gets ``dq = 0``. Under XLA's
    default excess precision ``D`` used to sum the unrounded fp32 cotangent (``dq`` 1e-2 there)."""
    from jla.ops.mosaic_gpu.attention import flash_attention

    batch, length, heads, kv_heads, head_dim = 2, 512, 4, 1, 128
    q, k, v, do = _inputs(9, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(4), batch, length, 3, "shuffled")

    @jax.jit
    def grads(q, k, v, do):
        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=200, backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    got = grads(q, k, v, do)
    want = grads(q, k, v, do.astype(jnp.bfloat16).astype(jnp.float32))
    for g, w in zip(got, want):
        if backward == "fused":  # dq adds its tile partials in a run-dependent order (fp32 TMA reductions)
            _assert_matches_one_device(g, w)
        else:
            np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(w, np.float32), rtol=1e-5, atol=1e-5)
    ids = np.asarray(seg)
    starts = np.concatenate([np.ones((batch, 1), bool), ids[:, 1:] != ids[:, :-1]], axis=1)
    assert np.abs(np.asarray(got[0], np.float32)[starts]).max() < 1e-5


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
@pytest.mark.parametrize("backward", ["ranged", "full", "fused"])
def test_batch_sharded_call_runs_per_device(backward):
    """Under jit with the batch sharded over two GPUs the kernels run on each device's own rows
    (no operand all-gather) and give the single-device values, gradients included."""
    import re

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from jla.ops.mosaic_gpu.attention import flash_attention

    mesh = Mesh(jax.devices()[:2], ("data",))
    rows, whole = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec())
    q, k, v, do = _inputs(3, 4, 1000, 4, 2, 128)
    seg = _segments(np.random.default_rng(1), 4, 1000, 3)
    sinks = jnp.linspace(-1.0, 2.0, 4, dtype=jnp.float32)

    def grads(q, k, v, do, seg, sinks):
        def loss(q, k, v, sinks):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=300, backward=backward, sinks=sinks)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(q, k, v, sinks)

    args = (q, k, v, do, seg, sinks)
    want = jax.jit(grads)(*args)
    sharded = jax.jit(grads, in_shardings=(rows,) * 5 + (whole,))
    placed = [jax.device_put(x, rows) for x in args[:5]] + [jax.device_put(sinks, whole)]
    hlo = sharded.lower(*placed).compile().as_text()
    gathered = [g for g in re.findall(r"all-gather[^\n]*", hlo) if "bf16[" in g or "s32[" in g]
    assert not gathered, gathered[:2]
    got = sharded(*placed)
    # Each device runs the same deterministic kernels on its rows; d sinks (a sum over the
    # batch, all-reduced across devices) may round differently, and the separately compiled XLA-side
    # reductions may move a few bf16 elements by one step (_assert_matches_one_device).
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        _assert_matches_one_device(g, w)


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
@pytest.mark.parametrize("call", ["ranged", "full", "fused", "auto"])
def test_head_sharded_call_runs_per_device(call):
    """Tensor parallelism: with the heads sharded over two GPUs (GQA 4:1, batch on a size-1 axis)
    the kernels run on each device's own heads, with no operand all-gather and H/2 query heads in
    every kernel call, and give the single-device values and gradients. ``"auto"`` goes through
    the layer function with packed documents, which resolves to this kernel on Hopper."""
    import re

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from jla.ops.mosaic_gpu.attention import flash_attention

    mesh = Mesh(np.array(jax.devices()[:2]).reshape(1, 2), ("data", "model"))
    by_head = NamedSharding(mesh, PartitionSpec("data", None, "model", None))
    rows, per_head = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec("model"))
    batch, length, heads, kv_heads, head_dim = 2, 1024, 8, 2, 128
    q, k, v, do = _inputs(5, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(3), batch, length, 3)
    sinks = jnp.linspace(-1.0, 2.0, heads, dtype=jnp.float32)

    def grads(q, k, v, do, seg, sinks):
        def loss(q, k, v, sinks):
            if call == "auto":
                out = attention(q, k, v, segment_ids=seg, sliding_window=300, implementation="auto")
            else:
                out = flash_attention(q, k, v, segment_ids=seg, sliding_window=300, backward=call, sinks=sinks)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(q, k, v, sinks)

    args = (q, k, v, do, seg, sinks)
    want = jax.jit(grads)(*args)
    in_shardings = (by_head,) * 4 + (rows, per_head)
    sharded = jax.jit(grads, in_shardings=in_shardings)
    placed = [jax.device_put(x, s) for x, s in zip(args, in_shardings)]
    hlo = sharded.lower(*placed).compile().as_text()
    gathered = [g for g in re.findall(r"all-gather[^\n]*", hlo) if "bf16[" in g or "s32[" in g]
    assert not gathered, gathered[:2]
    kernels = [line for line in hlo.splitlines() if 'custom_call_target="mosaic_gpu' in line]
    assert len(kernels) >= 2, len(kernels)  # the forward and at least one backward kernel
    local, whole = (f"bf16[{batch},{length},{n},{head_dim}]" for n in (heads // 2, heads))
    assert any(local in line for line in kernels) and not any(whole in line for line in kernels)
    got = sharded(*placed)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        _assert_matches_one_device(g, w)


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
def test_head_sharded_mqa_gathers_the_heads():
    """When the tensor-parallel width does not divide ``Hkv`` (MQA, one key/value head on two GPUs)
    the query heads are gathered and every kernel call sees all of them: redundant work, but values
    and gradients match fp32 attention and a one-device run. (The gathered bf16 cotangent hid the
    fp32 rounding that ``test_backward_uses_the_rounded_cotangent`` checks, so the one-device run
    used to differ here.)"""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from jla.ops.mosaic_gpu.attention import flash_attention

    mesh = Mesh(np.array(jax.devices()[:2]).reshape(1, 2), ("data", "model"))
    by_head = NamedSharding(mesh, PartitionSpec("data", None, "model", None))
    rows = NamedSharding(mesh, PartitionSpec("data"))
    batch, length, heads, head_dim = 2, 512, 4, 128
    q, k, v, do = _inputs(9, batch, length, heads, 1, head_dim)
    seg = _segments(np.random.default_rng(4), batch, length, 3, "shuffled")

    def grads(q, k, v, do, seg):
        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=200)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.grad(loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)

    args = (q, k, v, do, seg)
    want = jax.jit(grads)(*args)
    in_shardings = (by_head, rows, rows, by_head, rows)
    sharded = jax.jit(grads, in_shardings=in_shardings)
    placed = [jax.device_put(x, s) for x, s in zip(args, in_shardings)]
    hlo = sharded.lower(*placed).compile().as_text()
    kernels = [line for line in hlo.splitlines() if 'custom_call_target="mosaic_gpu' in line]
    assert kernels and any(f"bf16[{batch},{length},{heads},{head_dim}]" in line for line in kernels)
    got = jax.block_until_ready(sharded(*placed))
    (dq, dk, dv), out = got
    dense = lambda q, k, v: _dense_reference(q, k, v, seg, 200, head_dim**-0.5)  # noqa: E731
    (rdq, rdk, rdv), ref = jax.jit(jax.grad(lambda q, k, v: (jnp.sum(dense(q, k, v) * do), dense(q, k, v)),
                                            argnums=(0, 1, 2), has_aux=True))(q, k, v)
    for name, a, b in (("out", out, ref), ("dq", dq, rdq), ("dk", dk, rdk), ("dv", dv, rdv)):
        assert _rel_err(a, b) < 2e-2, (name, _rel_err(a, b))  # against fp32 (measured <= 6e-3)
    # One device, same process: the same gradients. The two programs compile the fp32 reduction
    # D = rowsum(o * do) separately and may sum it in another order, which can move a few dq, dk
    # elements to the neighbouring bf16 value (29 of 524288 in one run, 1.95e-3 at most), so the
    # arrays agree to bf16 resolution and the queries that see one key (document starts) exactly:
    # dq = 0 in both runs. The one-device run used to give 2.5e-3 there (see _delta).
    for name, g, w in zip(("dq", "dk", "dv", "out"), jax.tree.leaves(got), jax.tree.leaves(want)):
        assert _rel_err(g, w) < 2e-3, (name, _rel_err(g, w))
    ids = np.asarray(seg)
    starts = np.concatenate([np.ones((batch, 1), bool), ids[:, 1:] != ids[:, :-1]], axis=1)
    for name, d in (("sharded", dq), ("one device", want[0][0])):
        assert np.abs(np.asarray(d, np.float32)[starts]).max() < 1e-5, name


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
@pytest.mark.parametrize("backward", ["ranged", "fused"])
def test_inside_shard_map_matches_one_device(backward):
    """Called per shard inside ``jax.shard_map`` (varying-axis types checked, as context-parallel
    code and the layer under a mesh call it): values and gradients equal one device, and the
    gradient of replicated sinks is summed over the shards."""
    from jax.sharding import Mesh, PartitionSpec

    from jla.ops.mosaic_gpu.attention import flash_attention

    mesh = Mesh(jax.devices()[:2], ("data",))
    q, k, v, do = _inputs(4, 4, 640, 4, 2, 64)
    seg = _segments(np.random.default_rng(2), 4, 640, 3)
    sinks = jnp.linspace(-1.0, 2.0, 4, dtype=jnp.float32)

    def grads(q, k, v, do, seg, sinks):
        def loss(q, k, v, sinks):
            out = flash_attention(q, k, v, segment_ids=seg, sliding_window=200, sinks=sinks, backward=backward)
            return jnp.sum(out.astype(jnp.float32) * do)

        return jax.grad(loss, argnums=(0, 1, 2, 3))(q, k, v, sinks)

    want = jax.jit(grads)(q, k, v, do, seg, sinks)
    rows, whole = PartitionSpec("data"), PartitionSpec()
    sharded = jax.jit(jax.shard_map(grads, mesh=mesh, in_specs=(rows,) * 5 + (whole,),
                                    out_specs=(rows, rows, rows, whole)))
    got = sharded(q, k, v, do, seg, sinks)
    for g, w in zip(got, want):
        _assert_matches_one_device(g, w)


def test_unsupported_inputs_fail_loudly():
    from jla.ops.mosaic_gpu.attention import flash_attention

    q = jnp.zeros((1, 128, 2, 96), jnp.bfloat16)
    with pytest.raises(ValueError, match="mosaic_gpu attention"):
        flash_attention(q, q, q)
    q = jnp.zeros((1, 128, 2, 64), jnp.float32)
    with pytest.raises(ValueError, match="mosaic_gpu attention"):
        flash_attention(q, q, q)
    if _on_hopper():
        q = jnp.zeros((1, 128, 2, 64), jnp.bfloat16)
        with pytest.raises(ValueError, match="backward must be one of"):
            flash_attention(q, q, q, backward="dense")
        with pytest.raises(ValueError, match="sinks must be"):
            flash_attention(q, q, q, sinks=jnp.zeros((3,), jnp.float32))


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,window,backward,dtype", [
    pytest.param(1, 130, 4, 1, None, "ranged", jnp.bfloat16, id="T130-one-partial-tile"),
    pytest.param(1, 1000, 4, 1, 129, "ranged", jnp.bfloat16, id="T1000-swa129"),
    pytest.param(1, 1100, 4, 2, None, "fused", jnp.bfloat16, id="T1100-fused"),
    pytest.param(1, 1999, 4, 1, 2048, "ranged", jnp.float16, id="T1999-window-past-T-fp16"),
    pytest.param(2, 640, 8, 2, 1, "ranged", jnp.float16, id="T640-swa1-fp16"),
    pytest.param(1, 2048, 32, 8, None, "fused", jnp.bfloat16, id="T2048-32x8-causal"),
    pytest.param(1, 2048, 32, 8, 1024, "ranged", jnp.bfloat16, id="T2048-32x8-swa1024"),
])
def test_default_forward_d64_matches_dense_xla(batch, length, heads, kv_heads, window, backward, dtype):
    """The shipping forward for ``D = 64`` without segment ids: ``fwd_overlap`` and ``fwd_wgs`` left at
    None pick the overlapped loop on 192-row tiles of three consumer warpgroups, with ``heavy_first`` on
    and keys not loaded ahead of values. Checked against fp32 directly (the other tests of this path are
    bitwise against the ping-pong loop) at lengths whose last 192-row tile is partly idle (130, 640, 1000,
    1100, 1999), a window reaching past ``T``, a one-token window, the 1.3B presets' 32 query / 8
    key-value heads, fp16, and both the ranged and the fused backward."""
    from jla.ops.mosaic_gpu.attention import _Config

    default = _Config(scale=1.0, window=None, has_segments=False)
    assert (default.fwd_overlap, default.fwd_wgs, default.heavy_first, default.fwd_k_ahead) == (None, None, True, False)
    _check_against_dense(batch, length, heads, kv_heads, 64, window, 0, backward=backward, dtype=dtype)
