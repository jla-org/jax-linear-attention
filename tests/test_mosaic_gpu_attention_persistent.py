# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The persistent attention forward (``jla.ops.mosaic_gpu.attention_persistent``).

On an H100 the persistent forward (one CTA per SM looping over (batch, head, query tile) items) is checked
**bitwise** against the one-CTA-per-tile forward it replaces, outputs and ``dq, dk, dv`` through the
deterministic ranged backward (which reads the forward's ``lse``), for every mask case of the dense suite
(causal, windows, packed documents with sorted, unsorted and reused ids, GQA, odd lengths, ``D`` 64 and 128,
two and three consumer warpgroups, sinks), for MLA's 192 / 128 heads, and for problems with more items than
CTAs, several sections and padding slots; and against an fp32 dense reference through the layer's default
path. On CPU: the deal covers every item exactly once and balances the CTAs' work, and the tile order of
packed documents follows the documents.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.mosaic_gpu import attention as kernel_module
from jla.ops.mosaic_gpu import attention_persistent as persistent


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


def _deal(sched: persistent.Schedule) -> list[list[tuple[int, int]]]:
    """Every CTA's (pair, rank) items in order: a NumPy mirror of the kernel's ``item``."""
    lists = []
    for cta in range(sched.ctas):
        items = []
        for r in range(sched.rounds):
            pos = cta if r % 2 == 0 else sched.ctas - 1 - cta
            section, o = divmod(r * sched.ctas + pos, sched.padded)
            in_section = min(sched.section, sched.pairs - section * sched.section)
            if o < in_section * sched.n_tiles:
                items.append((section * sched.section + o % in_section, o // in_section))
        lists.append(items)
    return lists


SHAPES = [
    # batch, heads, kv_heads, length, rows per tile, head_dim
    (1, 16, 16, 16384, 128, 128),
    (4, 16, 16, 4096, 128, 128),
    (8, 16, 16, 2048, 128, 128),
    (4, 32, 8, 4096, 192, 64),
    (1, 32, 8, 16384, 192, 64),
    (2, 2, 1, 256, 128, 128),
    (3, 6, 2, 1152, 192, 64),
    (1, 12, 4, 8192, 128, 192),
]


@pytest.mark.parametrize("batch,heads,kv_heads,length,rows,head_dim", SHAPES)
def test_deal_covers_every_item_once(batch, heads, kv_heads, length, rows, head_dim):
    n_tiles = -(-length // rows)
    dv = 128 if head_dim == 192 else head_dim
    sched = persistent.schedule(batch=batch, heads=heads, kv_heads=kv_heads, n_tiles=n_tiles, length=length,
                                head_dim=head_dim, dv=dv, itemsize=2, num_sms=132)
    assert sched.padded % sched.ctas == 0 and sched.section * sched.n_tiles <= sched.padded
    assert sched.section % (heads // kv_heads) == 0 or sched.section == sched.pairs
    items = [it for cta in _deal(sched) for it in cta]
    assert sorted(items) == [(p, e) for p in range(batch * heads) for e in range(n_tiles)]


@pytest.mark.parametrize("batch,heads,kv_heads,length,rows,head_dim", SHAPES[:5])
def test_deal_balances_causal_work(batch, heads, kv_heads, length, rows, head_dim):
    """Causal tiles (rank e holds the e-th longest tile): the most loaded CTA carries at most the mean plus
    a small part of the longest tile, and items run longest first within each CTA's section."""
    n_tiles = -(-length // rows)
    sched = persistent.schedule(batch=batch, heads=heads, kv_heads=kv_heads, n_tiles=n_tiles, length=length,
                                head_dim=head_dim, dv=head_dim, itemsize=2, num_sms=132)
    longest = -(-length // 128)
    work = lambda e: max(1, -(-(length - e * rows) // 128))  # noqa: E731 - key blocks of the e-th longest tile
    deal = _deal(sched)
    loads = np.array([sum(work(e) + 2 for _, e in cta) for cta in deal])  # + a per-tile overhead
    assert loads.max() <= loads.mean() + 0.25 * longest + 4, (loads.max(), loads.mean(), longest)
    for cta in deal:  # within a section a CTA's items come longest first
        for (p0, e0), (p1, e1) in zip(cta, cta[1:]):
            assert p0 // sched.section != p1 // sched.section or e1 >= e0


def test_tile_order_follows_the_documents():
    """Packed documents: tiles near the end of long documents first; causal: the last tile first."""
    cfg = kernel_module._Config(scale=1.0, window=None, has_segments=True)
    length = 1024
    seg = jnp.asarray(np.repeat([0, 1, 2], [640, 256, 128])[None], jnp.int32)
    runs = kernel_module._segment_runs(seg, kernel_module._contiguous_ids(seg))
    order = np.asarray(persistent.tile_order(runs, cfg, batch=1, length=length, n_tiles=8, rows=128))[0]
    # Key blocks per tile: document 0 (tiles 0-4) 1..5; document 1 (rows 640-895, from block 5): tile 5 one,
    # tile 6 two; document 2 (tile 7, from block 7): one. Longest first, ties: the later tile first.
    assert order.tolist() == [4, 3, 2, 6, 1, 7, 5, 0]
    items = np.asarray(persistent.tile_items(runs, cfg, batch=1, length=length, n_tiles=8, rows=128))[0]
    assert items[0].tolist() == order.tolist()
    assert items[1].tolist() == [0, 0, 0, 5, 0, 7, 5, 0]  # lo_all by rank: the block of the tile's document start
    assert items[2].tolist() == [5, 4, 3, 7, 2, 8, 6, 1]  # hi_all by rank: through the diagonal block
    causal = kernel_module._Config(scale=1.0, window=None, has_segments=False)
    plain = np.asarray(persistent.tile_order(None, causal, batch=2, length=length, n_tiles=8, rows=128))
    assert plain.tolist() == [list(range(7, -1, -1))] * 2


def _run(monkeypatch, persistent_on, q, k, v, do, seg, window, sinks, **options):
    config = functools.partial(kernel_module._Config, fwd_persistent=persistent_on, **options)
    monkeypatch.setattr(kernel_module, "_Config", config)

    def loss(q, k, v):
        out = kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window, sinks=sinks)
        return jnp.sum(out.astype(jnp.float32) * do), out

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2), has_aux=True))(q, k, v)


def _assert_bitwise(new, old):
    (grads_new, out_new), (grads_old, out_old) = new, old
    np.testing.assert_array_equal(np.asarray(out_new, np.float32), np.asarray(out_old, np.float32))
    for name, a, b in zip(("dq", "dk", "dv"), grads_new, grads_old):
        np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32), err_msg=name)


def _dense_cases():
    from test_mosaic_gpu_attention import CASES

    return CASES


EXTRA_CASES = [
    pytest.param(1, 128, 2, 1, 64, None, 0, id="one-block"),
    pytest.param(1, 4096, 2, 1, 128, 1000, 7, id="swa-1000-documents-7-4096"),
    pytest.param(1, 2048, 2, 2, 64, 300, 0, id="swa-300-2048-d64"),
    # More items than CTAs: several rounds, sections and padding slots.
    pytest.param(4, 2048, 16, 4, 128, None, 0, id="causal-2048-b4-h16-gqa4"),
    pytest.param(3, 1152, 24, 8, 64, None, 0, id="causal-1152-b3-h24-gqa3-d64"),
    pytest.param(2, 4096, 8, 8, 64, 700, 0, id="swa-700-4096-b2-h8-d64"),
    # Documents over several rounds: the item table is read one item ahead (a loop-carried row offset loses the
    # known tile multiple that the query copies need; this case caught it).
    pytest.param(4, 3000, 8, 2, 128, None, 9, id="documents-9-3000-b4-gqa4"),
    pytest.param(2, 2048, 32, 8, 64, None, 5, id="documents-5-2048-h32-gqa4-d64"),
    pytest.param(2, 1536, 8, 4, 64, 257, 4, id="swa-257-documents-4-1536-d64"),
]


@gpu
@pytest.mark.parametrize("sinks", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", _dense_cases() + EXTRA_CASES)
def test_persistent_forward_equals_one_cta_per_tile(monkeypatch, sinks, batch, length, heads, kv_heads, head_dim,
                                                    window, docs):
    from test_mosaic_gpu_attention import _inputs, _segments

    q, k, v, do = _inputs(length + 11, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length + 4), batch, length, docs) if docs else None
    sink = jnp.linspace(-2.0, 4.0, heads, dtype=jnp.float32) if sinks else None
    _assert_bitwise(_run(monkeypatch, True, q, k, v, do, seg, window, sink),
                    _run(monkeypatch, False, q, k, v, do, seg, window, sink))


@gpu
@pytest.mark.parametrize("options", [
    pytest.param(dict(fwd_overlap=True, fwd_wgs=2), id="overlap-2wg"),
    pytest.param(dict(fwd_overlap=False), id="pingpong"),
    pytest.param(dict(fwd_overlap=True, fwd_wgs=3), id="overlap-3wg"),
])
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(1, 384, 4, 2, 64, None, 0, id="causal-gqa-d64"),
    pytest.param(2, 640, 2, 2, 64, 33, 3, id="swa-33-documents-staggered-start"),
    pytest.param(1, 1024, 4, 2, 128, None, 0, id="causal-1024-gqa2"),
    pytest.param(1, 777, 4, 1, 128, 300, 6, id="swa-documents-777-gqa4"),
    pytest.param(2, 1152, 8, 2, 64, None, 0, id="causal-1152-gqa4"),
    pytest.param(1, 1024, 2, 2, 64, None, 5, id="documents-5-1024"),
])
def test_persistent_loops_equal_one_cta_per_tile(monkeypatch, options, batch, length, heads, kv_heads, head_dim,
                                                 window, docs):
    """Both loops of the forward (and two or three consumer warpgroups) run persistently, bitwise."""
    from test_mosaic_gpu_attention import _inputs, _segments

    if options.get("fwd_wgs") == 3 and head_dim != 64:
        pytest.skip("three consumer warpgroups need D = 64")
    q, k, v, do = _inputs(length + 13, batch, length, heads, kv_heads, head_dim)
    seg = _segments(np.random.default_rng(length + 6), batch, length, docs) if docs else None
    _assert_bitwise(_run(monkeypatch, True, q, k, v, do, seg, window, None, **options),
                    _run(monkeypatch, False, q, k, v, do, seg, window, None, **options))


@gpu
@pytest.mark.parametrize("order", ["sorted", "shuffled", "interleaved"])
@pytest.mark.parametrize("batch,length,heads,kv_heads,window,docs", [
    pytest.param(1, 1024, 4, 1, None, 0, id="causal-1024-gqa4"),
    pytest.param(2, 777, 4, 2, 300, 6, id="swa-documents-777"),
    pytest.param(2, 2048, 8, 2, None, 7, id="documents-7-2048"),
])
def test_persistent_mla_equals_one_cta_per_tile(monkeypatch, order, batch, length, heads, kv_heads, window, docs):
    """MLA's 192 / 128 heads (the output goes through the query tile) run persistently, bitwise."""
    from test_mosaic_gpu_mla_attention import _inputs, _segments

    if not docs and order != "sorted":
        pytest.skip("the id order matters only with documents")
    q, k, v, do = _inputs(length + 17, batch, length, heads, kv_heads)
    seg = _segments(np.random.default_rng(length + 8), batch, length, docs, order) if docs else None
    _assert_bitwise(_run(monkeypatch, True, q, k, v, do, seg, window, None),
                    _run(monkeypatch, False, q, k, v, do, seg, window, None))


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,head_dim,window,docs", [
    pytest.param(2, 2048, 16, 4, 128, None, 0, id="causal-2048-gqa4"),
    pytest.param(2, 3000, 8, 2, 64, None, 0, id="causal-odd-3000-gqa4-d64"),
    pytest.param(2, 2048, 8, 8, 64, 512, 0, id="swa-512-2048-d64"),
    pytest.param(2, 2048, 8, 2, 128, None, 9, id="documents-9-2048-gqa4"),
    pytest.param(2, 1536, 8, 2, 64, 300, 3, id="swa-300-documents-3-1536-d64"),
])
def test_default_path_matches_dense_reference(batch, length, heads, kv_heads, head_dim, window, docs):
    """The layer's default Mosaic GPU path (the persistent forward where it applies) against fp32 dense attention."""
    from test_mosaic_gpu_attention import _check_against_dense

    _check_against_dense(batch, length, heads, kv_heads, head_dim, window, docs)


@gpu
def test_default_forward_is_persistent(monkeypatch):
    """With the default configuration the forward runs the persistent kernel; ``fwd_persistent=False`` does not."""
    q = jnp.zeros((1, 1024, 4, 128), jnp.bfloat16)
    cfg = kernel_module._Config(scale=1.0, window=None, has_segments=False)
    assert cfg.fwd_persistent is None and persistent.applies(q, q, q, cfg)
    calls = []
    original = persistent.forward
    monkeypatch.setattr(persistent, "forward", lambda *args: calls.append(args[-1]) or original(*args))
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q)).lower(q)
    assert calls and calls[-1].fwd_persistent is None
    calls.clear()
    monkeypatch.setattr(kernel_module, "_Config", functools.partial(kernel_module._Config, fwd_persistent=False))
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q) * 2).lower(q)
    assert not calls


@gpu
def test_unknown_sm_count_keeps_one_cta_per_tile(monkeypatch):
    """A device that does not report its SM count keeps the one-CTA-per-tile forward by default instead of failing;
    ``fwd_persistent=True`` then raises."""
    q = jnp.zeros((1, 1024, 4, 128), jnp.bfloat16)
    calls = []
    original = persistent.forward
    monkeypatch.setattr(persistent, "forward", lambda *args: calls.append(args[-1]) or original(*args))
    monkeypatch.setattr(persistent, "sm_count", lambda: None)
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q) * 3).lower(q)
    assert not calls
    monkeypatch.setattr(kernel_module, "_Config", functools.partial(kernel_module._Config, fwd_persistent=True))
    with pytest.raises(ValueError, match="SM count"):
        jax.jit(lambda q: kernel_module.flash_attention(q, q, q) * 4).lower(q)


@gpu
@pytest.mark.parametrize("window,sinks", [(None, False), (200, True)], ids=["documents", "documents-swa200-sinks"])
def test_persistent_forward_inside_shard_map(window, sinks):
    """Inside ``jax.shard_map`` (varying-axis types checked) the persistent forward with packed documents traces and
    runs, forward and gradients, bitwise equal to the call outside it (one device; the first build failed to trace:
    a loop that started at a raw table load came back without the varying-axis type)."""
    from jax.sharding import Mesh, PartitionSpec

    from test_mosaic_gpu_attention import _inputs, _segments

    mesh = Mesh(jax.devices()[:1], ("data",))
    q, k, v, _ = _inputs(4, 4, 640, 4, 2, 64)
    seg = _segments(np.random.default_rng(2), 4, 640, 3)
    sink_values = jnp.linspace(-1.0, 2.0, 4, dtype=jnp.float32) if sinks else None

    def grads(q, k, v, seg):
        def loss(q, k, v):
            out = kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window, sinks=sink_values)
            return jnp.sum(out.astype(jnp.float32))

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    rows = PartitionSpec("data")
    got = jax.jit(jax.shard_map(grads, mesh=mesh, in_specs=(rows,) * 4, out_specs=(rows,) * 3))(q, k, v, seg)
    want = jax.jit(grads)(q, k, v, seg)
    for name, g, w in zip(("dq", "dk", "dv"), got, want):
        np.testing.assert_array_equal(np.asarray(g, np.float32), np.asarray(w, np.float32), err_msg=name)


def test_persistent_forward_option_and_config():
    """``AttentionConfig.persistent_forward`` (None, True, False) becomes the kernel config's ``fwd_persistent``;
    None leaves the config's own default."""
    from jla.models.config import AttentionConfig

    assert kernel_module.persistent_option(None) == {}
    assert kernel_module.persistent_option(False) == {"fwd_persistent": False}
    assert kernel_module.persistent_option(True) == {"fwd_persistent": True}
    with pytest.raises(ValueError, match="persistent_forward"):
        kernel_module.persistent_option("off")
    assert AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64).persistent_forward is None
    assert AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, persistent_forward=False).persistent_forward is False
    with pytest.raises(ValueError, match="persistent_forward"):
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, persistent_forward="off")


@pytest.mark.parametrize("value", [None, True, False])
def test_persistent_forward_reaches_the_kernel(monkeypatch, value):
    """The attention layer passes ``AttentionConfig.persistent_forward`` to ``flash_attention`` (CPU: the kernel is
    replaced by a recorder), and the FP8 gradient-epilogue path puts it in its kernel config."""
    from jla.layers.attention import Attention
    from jla.layers.fp8_attention import _kernel_config, _Static
    from jla.models.config import AttentionConfig

    seen = []

    def recorder(q, k, v, **kwargs):
        seen.append(kwargs.get("persistent_forward", "missing"))
        return jnp.zeros(q.shape[:3] + v.shape[-1:], q.dtype)

    monkeypatch.setattr(kernel_module, "flash_attention", recorder)
    cfg = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=64, qk_norm=False, implementation="mosaic_gpu",
                          persistent_forward=value)
    layer = Attention(128, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-6)
    x = jnp.ones((2, 128, 128), jnp.float32)
    layer(x, jnp.broadcast_to(jnp.arange(128), (2, 128)))
    assert seen == [value]
    static = _Static(4, 2, 64, 10_000.0, None, None, False, "mosaic_gpu", ("bfloat16", "float32"), "ranged", value)
    assert _kernel_config(static, 4096).fwd_persistent is value


@gpu
def test_persistent_forward_switch_on_the_kernel(monkeypatch):
    """``flash_attention(persistent_forward=False)`` keeps one CTA per tile; True and None run the persistent
    forward where it applies."""
    q = jnp.zeros((1, 1024, 4, 128), jnp.bfloat16)
    calls = []
    original = persistent.forward
    monkeypatch.setattr(persistent, "forward", lambda *args: calls.append(args[-1]) or original(*args))
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q, persistent_forward=False) * 5).lower(q)
    assert not calls
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q, persistent_forward=True) * 6).lower(q)
    assert calls and calls[-1].fwd_persistent is True
    calls.clear()
    jax.jit(lambda q: kernel_module.flash_attention(q, q, q, persistent_forward=None) * 7).lower(q)
    assert calls and calls[-1].fwd_persistent is None
