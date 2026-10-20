# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU grouped GEMM (ragged dot) against a naive per-group float32 oracle.

Values and both gradients (``d_lhs``, ``d_rhs``) for uneven groups, empty groups (first,
middle, last), a single group, rows not covered by any group, bfloat16 and float16; the
MoE layer's dropless path with ``ragged_dot_impl="mosaic_gpu"`` against the dense oracle;
and the fallback to ``jax.lax.ragged_dot`` where the kernels do not apply. Kernel tests
need an H100; the fallback and config tests run anywhere.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.moe import MoE
from jla.models.config import MoEConfig
from jla.ops.mosaic_gpu import ragged_dot as rd
from jla.ops.mosaic_gpu._common import is_hopper

ON_HOPPER = is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")


def _naive(lhs, rhs, group_sizes):
    """``out[r] = lhs[r] @ rhs[g(r)]`` with an explicit [M, G] membership mask, in float32."""
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    rows = jnp.arange(lhs.shape[0])[:, None]
    member = ((rows >= starts[None]) & (rows < ends[None])).astype(jnp.float32)  # [M, G]
    per_group = jnp.einsum("mk,gkn->gmn", lhs.astype(jnp.float32), rhs.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST)
    return jnp.einsum("mg,gmn->mn", member, per_group, precision=jax.lax.Precision.HIGHEST)


def _inputs(m, k, n, sizes, dtype=jnp.bfloat16, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    lhs = jax.random.normal(k1, (m, k), jnp.float32).astype(dtype)
    rhs = (jax.random.normal(k2, (len(sizes), k, n), jnp.float32) / np.sqrt(k)).astype(dtype)
    weight = jax.random.normal(k3, (m, n), jnp.float32)
    return lhs, rhs, jnp.asarray(sizes, jnp.int32), weight


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _uneven(m, g, seed, empty=()):
    """``g`` random group sizes summing to ``m`` with the groups in ``empty`` set to zero."""
    rng = np.random.default_rng(seed)
    w = rng.exponential(size=g)
    for e in empty:
        w[e] = 0.0
    sizes = np.floor(w / w.sum() * m).astype(np.int64)
    live = [i for i in range(g) if i not in empty]
    sizes[live[0]] += m - sizes.sum()
    assert sizes.sum() == m and all(sizes[e] == 0 for e in empty)
    return [int(s) for s in sizes]


CASES = {
    "uneven": (1000, 256, 384, _uneven(1000, 8, 1)),
    "empty_first_mid_last": (777, 128, 256, _uneven(777, 10, 2, empty=(0, 4, 5, 9))),
    "one_group": (300, 256, 128, [300]),
    "many_small": (1024, 128, 256, _uneven(1024, 64, 3, empty=(7, 8, 40))),
    "aligned_blocks": (512, 128, 128, [128, 0, 256, 128]),
    "partial_cover": (640, 128, 256, [100, 0, 250, 30]),  # rows 380.. are in no group
    "wide": (1000, 256, 512, _uneven(1000, 12, 4, empty=(5,))),  # admits 256-wide tiles both ways
    "wide_empty": (777, 256, 256, _uneven(777, 10, 2, empty=(0, 4, 5, 9))),
    "wide_partial": (640, 512, 256, [100, 0, 250, 30]),
}


# The block-aligned kernel, every tiling the default selection can return, and the other legal tiles.
_TILINGS = {
    "block_aligned": rd.BLOCK_ALIGNED,
    **{f"{c.block_m}x{c.block_n}x{c.block_k}s{c.steps}w{c.warpgroups}g{c.grid_block_n}{c.schedule}": c
       for c in rd.CONFIGS},
    "aligned": rd.ALIGNED,
    "aligned_s4": rd.TileConfig(schedule="aligned"),
    "m64": rd.TileConfig(schedule="aligned", block_m=64),
    "k128s3": rd.TileConfig(schedule="aligned", block_k=128, steps=3),
    "ws": rd.WARP_SPECIALIZED,
    "ws_m64": rd.TileConfig(schedule="aligned", block_m=64, warpgroups=2),
    "persistent": rd.PERSISTENT,
    "persistent_s4": rd.TileConfig(schedule="persistent"),
    "persistent_ws": rd.PERSISTENT_WS,
}
# Tilings the default (config=None) runs for the test shapes: their own entries would repeat it.
_DEFAULT_TILINGS = {rd.select_config(a, b) for _, k, n, _ in CASES.values() for a, b in ((k, n), (n, k))}
CONFIGS = {"default": None}
for _name, _cfg in _TILINGS.items():  # each distinct tiling once
    if _cfg not in _DEFAULT_TILINGS and _cfg not in CONFIGS.values():
        CONFIGS[_name] = _cfg


def test_config_axis_covers_every_selectable_tiling_once():
    tested = _DEFAULT_TILINGS | {c for c in CONFIGS.values() if c is not None}
    assert set(rd.CONFIGS) <= tested
    assert len(tested) == len(CONFIGS) - 1 + len(_DEFAULT_TILINGS)
    assert {rd.BLOCK_ALIGNED, rd.ALIGNED, rd.WARP_SPECIALIZED} <= tested


@gpu
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("name", list(CASES))
def test_forward_matches_naive(name, config):
    m, k, n, sizes = CASES[name]
    cfg = CONFIGS[config]
    if cfg is not None and not cfg.divides_both(k, n):
        pytest.skip(f"tile {config} does not divide K={k}, N={n}")
    lhs, rhs, gs, _ = _inputs(m, k, n, sizes)
    out = jax.jit(lambda a, b, g: rd.ragged_dot(a, b, g, config=cfg))(lhs, rhs, gs)
    want = _naive(lhs, rhs, gs)
    assert out.dtype == jnp.bfloat16 and out.shape == (m, n)
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(want), atol=3e-2, rtol=2e-2)
    if sum(sizes) < m:  # rows past the groups are exactly zero
        assert np.all(np.asarray(out[sum(sizes):], np.float32) == 0.0)


@gpu
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("name", list(CASES))
def test_gradients_match_naive(name, config):
    m, k, n, sizes = CASES[name]
    cfg = CONFIGS[config]
    if cfg is not None and not cfg.divides_both(k, n):
        pytest.skip(f"tile {config} does not divide K={k}, N={n}")
    lhs, rhs, gs, weight = _inputs(m, k, n, sizes, seed=5)

    def loss(fn):
        return lambda a, b: jnp.sum(fn(a, b, gs).astype(jnp.float32) * weight)

    got = jax.jit(jax.grad(loss(lambda a, b, g: rd.ragged_dot(a, b, g, config=cfg)), argnums=(0, 1)))(lhs, rhs)
    want = jax.grad(loss(_naive), argnums=(0, 1))(lhs, rhs)
    assert got[0].dtype == lhs.dtype and got[1].dtype == rhs.dtype
    assert _rel(got[0], want[0]) < 1e-2, _rel(got[0], want[0])
    assert _rel(got[1], want[1]) < 1e-2, _rel(got[1], want[1])
    # Per group as well, so an error confined to one small group cannot hide in the global norm.
    ends = np.cumsum(np.asarray(sizes))
    for gi, (lo, hi) in enumerate(zip(ends - np.asarray(sizes), ends)):
        if hi > lo:
            assert _rel(got[0][lo:hi], want[0][lo:hi]) < 2e-2, (gi, _rel(got[0][lo:hi], want[0][lo:hi]))
            assert _rel(got[1][gi], want[1][gi]) < 2e-2, (gi, _rel(got[1][gi], want[1][gi]))
    # Empty groups get an exactly zero weight gradient.
    for gi, s in enumerate(sizes):
        if s == 0:
            assert np.all(np.asarray(got[1][gi], np.float32) == 0.0)
    if sum(sizes) < m:
        assert np.all(np.asarray(got[0][sum(sizes):], np.float32) == 0.0)


# The default and the opt-in tilings on the larger cases below, where SMs run several tiles each
# (pipeline barriers and output buffers reused across tiles), with float16, full_groups and many groups.
OPT_IN = {"default": None,
          **{name: _TILINGS[name] for name in ("block_aligned", "aligned", "ws", "m64", "k128s3", "ws_m64", "persistent",
                                                      "persistent_ws") if _TILINGS[name] not in _DEFAULT_TILINGS}}


@gpu
@pytest.mark.parametrize("config", list(OPT_IN))
def test_float16_and_full_groups(config):
    m, k, n = 8192, 256, 512
    cfg = OPT_IN[config]
    if cfg is not None and not cfg.divides_both(k, n):
        pytest.skip(f"tile {config} does not divide K={k}, N={n}")
    sizes = _uneven(m, 16, 7, empty=(3,))
    lhs, rhs, gs, weight = _inputs(m, k, n, sizes, dtype=jnp.float16, seed=9)
    fn = lambda a, b: rd.ragged_dot(a, b, gs, full_groups=True, config=cfg)  # noqa: E731
    out = jax.jit(fn)(lhs, rhs)
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(_naive(lhs, rhs, gs)), atol=1e-2, rtol=1e-2)
    got = jax.jit(jax.grad(lambda a, b: jnp.sum(fn(a, b).astype(jnp.float32) * weight), argnums=(0, 1)))(lhs, rhs)
    want = jax.grad(lambda a, b: jnp.sum(_naive(a, b, gs) * weight), argnums=(0, 1))(lhs, rhs)
    assert got[0].dtype == jnp.float16 and got[1].dtype == jnp.float16
    assert _rel(got[0], want[0]) < 5e-3 and _rel(got[1], want[1]) < 5e-3


@gpu
@pytest.mark.parametrize("config", list(OPT_IN))
def test_moe_training_shape_routing(config):
    """moe-125m-like widths with real top-8 routing over 64 experts (uneven, some empty)."""
    m_tokens, d, f, e, top_k = 512, 768, 256, 64, 8
    cfg = OPT_IN[config]
    if cfg is not None and not cfg.divides_both(d, 2 * f):
        pytest.skip(f"tile {config} does not divide K={d}, N={2 * f}")
    k1, k2, k3 = jax.random.split(jax.random.key(3), 3)
    experts = jax.random.categorical(k1, jax.random.normal(k1, (e,)) * 2.0, shape=(m_tokens * top_k,))
    sizes = jnp.zeros((e,), jnp.int32).at[experts].add(1)
    lhs = jax.random.normal(k2, (m_tokens * top_k, d), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k3, (e, d, 2 * f), jnp.float32) / np.sqrt(d)).astype(jnp.bfloat16)
    fn = lambda a, b: rd.ragged_dot(a, b, sizes, full_groups=True, config=cfg)  # noqa: E731
    out = jax.jit(fn)(lhs, rhs)
    assert _rel(out, _naive(lhs, rhs, sizes)) < 1e-2
    weight = jax.random.normal(k1, out.shape, jnp.float32)
    got = jax.jit(jax.grad(lambda a, b: jnp.sum(fn(a, b).astype(jnp.float32) * weight), argnums=(0, 1)))(lhs, rhs)
    want = jax.grad(lambda a, b: jnp.sum(_naive(a, b, sizes) * weight), argnums=(0, 1))(lhs, rhs)
    assert _rel(got[0], want[0]) < 1e-2 and _rel(got[1], want[1]) < 1e-2


@gpu
@pytest.mark.parametrize("config", list(OPT_IN))
def test_many_groups_partial_cover(config):
    """256 groups (a third of them empty) and rows past the last group, several tiles per SM, against an
    exact per-group float64 oracle of the output and both gradients."""
    m, k, n, covered, g = 6144, 256, 256, 5600, 256
    cfg = OPT_IN[config]
    if cfg is not None and not cfg.divides_both(k, n):
        pytest.skip(f"tile {config} does not divide K={k}, N={n}")
    sizes = _uneven(covered, g, 12, empty=tuple(range(0, g, 3)))
    lhs, rhs, gs, weight = _inputs(m, k, n, sizes, seed=13)
    fn = lambda a, b: rd.ragged_dot(a, b, gs, config=cfg)  # noqa: E731
    out = jax.jit(fn)(lhs, rhs)
    got = jax.jit(jax.grad(lambda a, b: jnp.sum(fn(a, b).astype(jnp.float32) * weight), argnums=(0, 1)))(lhs, rhs)
    out, got = np.asarray(out, np.float32), [np.asarray(x, np.float32) for x in got]
    a, b, w = (np.asarray(x, np.float64) for x in (lhs, rhs, weight))
    want = np.zeros((m, n)), np.zeros((m, k)), np.zeros(b.shape)
    ends = np.cumsum(sizes)
    for gi, (lo, hi) in enumerate(zip(ends - np.asarray(sizes), ends)):
        want[0][lo:hi] = a[lo:hi] @ b[gi]
        want[1][lo:hi] = w[lo:hi] @ b[gi].T
        want[2][gi] = a[lo:hi].T @ w[lo:hi]
        if hi > lo:
            assert _rel(got[1][gi], want[2][gi]) < 2e-2, (gi, _rel(got[1][gi], want[2][gi]))
        else:
            assert np.all(np.asarray(got[1][gi], np.float32) == 0.0), gi
    assert _rel(out, want[0]) < 1e-2 and _rel(got[0], want[1]) < 1e-2 and _rel(got[1], want[2]) < 1e-2
    assert np.all(np.asarray(out[covered:], np.float32) == 0.0)
    assert np.all(np.asarray(got[0][covered:], np.float32) == 0.0)


def _moe_layer(impl):
    cfg = MoEConfig(num_experts=8, top_k=2, d_ff_expert=128, num_shared_experts=1, ragged_dot_impl=impl)
    return MoE(256, cfg, key=jax.random.key(0), std=0.05, compute_dtype="bfloat16")


@gpu
def test_moe_layer_mosaic_matches_dense_oracle_and_xla():
    moe = _moe_layer("mosaic_gpu")
    xla = dataclasses.replace(moe.cfg, ragged_dot_impl="xla")
    moe_xla = MoE(256, xla, key=jax.random.key(0), std=0.05, compute_dtype="bfloat16")
    x = jax.random.normal(jax.random.key(1), (2, 256, 256), jnp.float32)
    y, _ = jax.jit(lambda m, x: m(x))(moe, x)
    assert _rel(y, moe.reference(x)) < 1e-2
    y_xla, _ = jax.jit(lambda m, x: m(x))(moe_xla, x)
    assert _rel(y, y_xla) < 1e-2

    def loss(m, x):
        return jnp.sum(jnp.square(m(x)[0].astype(jnp.float32)))

    def loss_ref(m, x):
        return jnp.sum(jnp.square(m.reference(x).astype(jnp.float32)))

    g = jax.jit(jax.grad(loss, argnums=(0, 1)))(moe, x)
    g_ref = jax.grad(loss_ref, argnums=(0, 1))(moe, x)
    for name in ("gate_up", "down", "router"):
        assert _rel(getattr(g[0], name), getattr(g_ref[0], name)) < 3e-2, name
    assert _rel(g[1], g_ref[1]) < 3e-2


def test_fallback_to_xla_where_unsupported():
    """Unsupported shapes/devices (here: K not a multiple of 128) use jax.lax.ragged_dot."""
    lhs, rhs, gs, _ = _inputs(200, 96, 128, [50, 0, 150])
    assert not rd.supports(lhs, rhs, gs)
    out = rd.grouped_matmul(lhs, rhs, gs, impl="mosaic_gpu")
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.asarray(jax.lax.ragged_dot(lhs, rhs, gs), np.float32))
    with pytest.raises(ValueError, match="does not support"):
        rd.ragged_dot(lhs, rhs, gs)
    with pytest.raises(ValueError, match="unknown ragged_dot impl"):
        rd.grouped_matmul(lhs, rhs, gs, impl="triton")


def test_supports_predicate():
    lhs, rhs, gs, _ = _inputs(256, 128, 256, [128, 128])
    assert rd.supports(lhs, rhs, gs) == ON_HOPPER
    assert not rd.supports(lhs.astype(jnp.float32), rhs.astype(jnp.float32), gs)  # no f32 wgmma path
    assert not rd.supports(lhs[:100], rhs, gs)  # M below one row tile
    assert not rd.supports(lhs, rhs[:, :, :200], gs)  # N unaligned


def test_moe_config_default_and_validation():
    assert MoEConfig(num_experts=4, top_k=2, d_ff_expert=8).ragged_dot_impl == "auto"
    # "auto" is the default for capacity routing too (which does not use it); an explicit kernel is not.
    MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=1.25)
    with pytest.raises(ValueError, match="dropless"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=1.25, ragged_dot_impl="mosaic_gpu")
    with pytest.raises(ValueError, match="ragged_dot_impl"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, ragged_dot_impl="cuda")


def test_auto_impl_falls_back_silently_and_matches_xla(recwarn):
    """``impl="auto"``: the kernels where supported (H100), else jax.lax.ragged_dot without a warning."""
    lhs, rhs, gs, _ = _inputs(200, 96, 128, [50, 0, 150])  # K = 96: unsupported everywhere
    out = rd.grouped_matmul(lhs, rhs, gs, impl="auto")
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(jax.lax.ragged_dot(lhs, rhs, gs), np.float32))
    assert not [w for w in recwarn if "falling back" in str(w.message)]
    moe = _moe_layer("auto")
    assert moe._grouped_gemm_impl() == ("mosaic_gpu" if ON_HOPPER else "xla")
    x = jax.random.normal(jax.random.key(1), (2, 128, 256), jnp.float32)
    y, _ = jax.jit(lambda m, x: m(x))(moe, x)
    y_xla, _ = jax.jit(lambda m, x: m(x))(MoE(256, dataclasses.replace(moe.cfg, ragged_dot_impl="xla"),
                                              key=jax.random.key(0), std=0.05, compute_dtype="bfloat16"), x)
    assert _rel(y, y_xla) < 1e-2


@gpu
def test_config_checked_before_the_gradient():
    """A tiling that fits the forward but not d_lhs (K=384 is not a multiple of 256) fails at the call."""
    lhs, rhs, gs, _ = _inputs(512, 384, 512, [200, 0, 312])
    assert rd.WARP_SPECIALIZED.divides(384, 512) and not rd.WARP_SPECIALIZED.divides(512, 384)
    assert rd.supports(lhs, rhs, gs) and not rd.supports(lhs, rhs, gs, config=rd.WARP_SPECIALIZED)
    with pytest.raises(ValueError, match="does not divide"):
        rd.ragged_dot(lhs, rhs, gs, config=rd.WARP_SPECIALIZED)
    with pytest.raises(ValueError, match="does not divide"):
        rd.grouped_matmul(lhs, rhs, gs, impl="mosaic_gpu", config=rd.WARP_SPECIALIZED)


def test_config_is_never_dropped():
    """An explicit tiling with impl='xla', or where the call would fall back to XLA, raises."""
    lhs, rhs, gs, _ = _inputs(256, 128, 256, [128, 128])
    with pytest.raises(ValueError, match="takes no config"):
        rd.grouped_matmul(lhs, rhs, gs, impl="xla", config=rd.ALIGNED)
    small = _inputs(200, 96, 128, [50, 0, 150])[:3]  # unsupported everywhere: K not a multiple of 128
    with pytest.raises(ValueError, match="does not support"):
        rd.grouped_matmul(*small, impl="mosaic_gpu", config=rd.ALIGNED)
    with pytest.raises(TypeError, match="TileConfig"):
        rd.ragged_dot(lhs, rhs, gs, config="aligned")


def _aligned_schedule(group_sizes, block_m: int, tiles: int):
    """Oracle of the aligned schedule, per row tile: group, first row (a multiple of 8), offset of
    its first valid row, valid rows. Group ``g`` (rows ``[s_g, e_g)``) owns
    ``ceil((e_g - floor8(s_g)) / block_m)`` consecutive tiles (none when empty)."""
    sizes = np.maximum(np.asarray(group_sizes, np.int64), 0)
    ends = np.cumsum(sizes)
    starts = ends - sizes
    first_rows = starts // rd.ROW_ALIGN * rd.ROW_ALIGN
    blocks = np.where(sizes == 0, 0, (ends - first_rows + block_m - 1) // block_m)
    tile_ends = np.cumsum(blocks)
    tile_starts = tile_ends - blocks
    tid = np.arange(tiles)
    group = np.searchsorted(tile_ends[:-1], tid, side="right")
    row0 = first_rows[group] + block_m * (tid - tile_starts[group])
    lo = np.maximum(row0, starts[group])
    hi = np.minimum(row0 + block_m, ends[group])
    size = np.maximum(hi - lo, 0)
    return group, np.where(size > 0, row0, 0), np.where(size > 0, lo - row0, 0), size


@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("sizes", [
    [300], [0, 0, 300], [300, 0, 0], [1, 1, 1, 297], [128, 0, 256, 128], [100, 0, 250, 30],
    _uneven(1024, 64, 3, empty=(7, 8, 40)), _uneven(777, 10, 2, empty=(0, 4, 5, 9)), [0] * 7 + [5],
])
def test_aligned_schedule_covers_every_row_once(sizes, block_m):
    """The aligned tile table: each grouped row in exactly one tile of its group, tiles 8-aligned,
    busy tiles a prefix of the grid, and the grid bound ``ceil(M / block_m) + G - 1`` respected."""
    m = max(sum(sizes), 1) + 37  # some rows in no group
    g = len(sizes)
    tiles = -(-m // block_m) + g - 1
    group, row0, start, size = _aligned_schedule(sizes, block_m, tiles)
    ends = np.cumsum(sizes)
    starts = ends - np.asarray(sizes)
    owner = np.full(m, -1)
    for t in range(tiles):
        if size[t] == 0:
            continue
        assert row0[t] % 8 == 0 and 0 <= start[t] and start[t] + size[t] <= block_m
        lo, hi = row0[t] + start[t], row0[t] + start[t] + size[t]
        assert starts[group[t]] <= lo and hi <= ends[group[t]], (t, lo, hi, group[t])
        assert np.all(owner[lo:hi] == -1), t
        owner[lo:hi] = group[t]
    want = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)] + [np.full(m - sum(sizes), -1)])
    np.testing.assert_array_equal(owner, want)
    busy = np.nonzero(size > 0)[0]
    assert busy.size == 0 or np.array_equal(busy, np.arange(busy.size))
    assert busy.size <= tiles
    # The kernels' per-tile search over the per-group table gives the same schedule.
    table = rd._group_table(jnp.asarray(sizes, jnp.int32), block_m)
    got = jax.jit(jax.vmap(lambda t: rd._tile_info(table, g, block_m, t)))(jnp.arange(tiles, dtype=jnp.int32))
    got = [np.asarray(a) for a in got]
    np.testing.assert_array_equal(got[3], size)
    live = size > 0
    for want_col, got_col in zip((group, row0, start), got[:3]):
        np.testing.assert_array_equal(got_col[live], want_col[live])
    # The persistent schedule's per-tile table (from XLA) and its count of tiles with rows.
    table, count = rd._row_tile_table(jnp.asarray(sizes, jnp.int32), block_m, tiles)
    table = np.asarray(table)
    assert int(np.asarray(count)[0]) == busy.size
    np.testing.assert_array_equal(table[3], size)
    for want_col, got_col in zip((group, row0, start), table[:3]):
        np.testing.assert_array_equal(got_col[live], want_col[live])


def test_tile_config_validation():
    assert rd.select_config(768, 1536) in rd.CONFIGS
    assert rd.BLOCK_ALIGNED.schedule == "block_aligned" and rd.TileConfig() == rd.BLOCK_ALIGNED
    assert rd.select_config(768, 1536) == rd.PERSISTENT
    assert rd.ALIGNED.schedule == rd.WARP_SPECIALIZED.schedule == "aligned"
    with pytest.raises(ValueError, match="warpgroups must be 1"):
        rd.TileConfig(warpgroups=2)
    with pytest.raises(ValueError, match="registers"):
        rd.TileConfig(block_m=128, block_n=256)
    with pytest.raises(ValueError, match="shared memory"):
        rd.TileConfig(block_k=128, steps=6)
    with pytest.raises(ValueError, match="schedule"):
        rd.TileConfig(schedule="sorted")
    for k, n in [(128, 128), (768, 1536), (1408, 2048), (2048, 2816), (256, 768)]:
        assert rd.select_config(k, n).divides(k, n)


def test_moe_layer_mosaic_impl_off_hopper_equals_xla():
    """Off an H100 (or for unsupported shapes) the option is the XLA path, bit for bit."""
    if ON_HOPPER:
        pytest.skip("the fallback is exercised off Hopper")
    x = jax.random.normal(jax.random.key(1), (2, 64, 256), jnp.float32)
    y, _ = _moe_layer("mosaic_gpu")(x)
    y_xla, _ = _moe_layer("xla")(x)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(y_xla, np.float32))
