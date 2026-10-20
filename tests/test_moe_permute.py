# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gather-only MoE permutations (jla.ops.moe_permute) against plain indexing and its autodiff."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.moe import MoE
from jla.models.config import MoEConfig
from jla.ops.moe_permute import combine_rows, dispatch_rows


def _naive_dispatch(x, src):
    ext = jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)])
    return ext[jnp.minimum(src, x.shape[0])]


def _naive_combine(y, dst, top_k):
    ext = jnp.concatenate([y, jnp.zeros((1,) + y.shape[1:], y.dtype)])
    rows = ext[jnp.minimum(dst, y.shape[0])].astype(jnp.float32)
    return rows.reshape((dst.shape[0] // top_k, top_k) + y.shape[1:]).sum(axis=1)


def _dropless_maps(tokens, top_k, experts, seed):
    rng = np.random.default_rng(seed)
    expert = rng.integers(0, experts, tokens * top_k)
    order = np.argsort(expert, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return jnp.asarray(order // top_k, jnp.int32), jnp.asarray(rank, jnp.int32), order.size


def _capacity_maps(tokens, top_k, rows, seed):
    """Some routes dropped, some rows empty; every kept route owns one distinct row."""
    rng = np.random.default_rng(seed)
    routes = tokens * top_k
    kept = rng.random(routes) < 0.7
    kept_idx = np.flatnonzero(kept)[: rows - 3]  # leave a few rows empty
    dst = np.full(routes, rows, np.int64)
    dst[kept_idx] = rng.permutation(rows)[: kept_idx.size]
    src = np.full(rows, tokens, np.int64)
    src[dst[kept_idx]] = kept_idx // top_k
    return jnp.asarray(src, jnp.int32), jnp.asarray(dst, jnp.int32), rows


@pytest.mark.parametrize("maps", ["dropless", "capacity"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dispatch_and_combine_match_indexing_values_and_vjps(maps, dtype) -> None:
    tokens, top_k, d = 37, 3, 16
    if maps == "dropless":
        src, dst, rows = _dropless_maps(tokens, top_k, experts=5, seed=0)
    else:
        src, dst, rows = _capacity_maps(tokens, top_k, rows=60, seed=1)
    kx, ky, kg1, kg2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(kx, (tokens, d), jnp.float32).astype(dtype)
    y = jax.random.normal(ky, (rows, d), jnp.float32).astype(dtype)
    g_rows = jax.random.normal(kg1, (rows, d), jnp.float32).astype(dtype)
    g_tok = jax.random.normal(kg2, (tokens, d), jnp.float32)

    out, vjp = jax.vjp(lambda a: dispatch_rows(a, src, dst, top_k), x)
    want, want_vjp = jax.vjp(lambda a: _naive_dispatch(a, src), x)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(want, np.float32))
    tol = 1e-6 if dtype == jnp.float32 else 1e-2
    np.testing.assert_allclose(np.asarray(vjp(g_rows)[0], np.float32), np.asarray(want_vjp(g_rows)[0], np.float32),
                               rtol=tol, atol=tol)

    out, vjp = jax.vjp(lambda a: combine_rows(a, src, dst, top_k), y)
    want, want_vjp = jax.vjp(lambda a: _naive_combine(a, dst, top_k), y)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out), np.asarray(want), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(vjp(g_tok)[0], np.float32), np.asarray(want_vjp(g_tok)[0], np.float32))


@pytest.mark.parametrize("maps", ["dropless", "capacity"])
def test_partially_manual_meshes_use_plain_indexing_with_the_same_values_and_vjps(maps, monkeypatch) -> None:
    """Inside a pipeline stage (``shard_map`` over ``pp`` only) both ops index plainly (autodiff transposes)."""
    import jla.ops.moe_permute as moe_permute

    tokens, top_k, d = 37, 3, 16
    if maps == "dropless":
        src, dst, rows = _dropless_maps(tokens, top_k, experts=5, seed=0)
    else:
        src, dst, rows = _capacity_maps(tokens, top_k, rows=60, seed=1)
    x = jax.random.normal(jax.random.key(3), (tokens, d))
    y = jax.random.normal(jax.random.key(4), (rows, d))
    g_rows = jax.random.normal(jax.random.key(5), (rows, d))
    g_tok = jax.random.normal(jax.random.key(6), (tokens, d))

    def run():
        out_c, vjp_c = jax.vjp(lambda a: combine_rows(a, src, dst, top_k), y)
        out_d, vjp_d = jax.vjp(lambda a: dispatch_rows(a, src, dst, top_k), x)
        return [np.asarray(v) for v in (out_c, vjp_c(g_tok)[0], out_d, vjp_d(g_rows)[0])]

    per_choice = run()
    monkeypatch.setattr(moe_permute, "_partially_manual", lambda: True)
    for got, want in zip(run(), per_choice, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_ops_are_each_others_transpose_under_jit() -> None:
    src, dst, rows = _capacity_maps(20, 2, rows=30, seed=2)
    x = jax.random.normal(jax.random.key(1), (20, 8))
    y = jax.random.normal(jax.random.key(2), (rows, 8))
    # <dispatch(x), y> == <x, combine(y)> for the linear maps.
    lhs = jnp.vdot(jax.jit(dispatch_rows, static_argnums=3)(x, src, dst, 2), y)
    rhs = jnp.vdot(x, jax.jit(combine_rows, static_argnums=3)(y, src, dst, 2))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-5)


@pytest.mark.parametrize("capacity_factor", [None, 0.6])
def test_moe_gradients_with_dropped_routes_match_the_scatter_formulation(capacity_factor) -> None:
    """The layer against the scatter/gather formulation it replaced (drops at capacity 0.6)."""
    cfg = MoEConfig(num_experts=6, top_k=2, d_ff_expert=24, num_shared_experts=0, capacity_factor=capacity_factor)
    d = 32
    moe = MoE(d, cfg, key=jax.random.key(3), std=0.2, compute_dtype="float32")
    x = jax.random.normal(jax.random.key(4), (40, d))

    def scatter_formulation(m, x):
        routing = m._route(x)
        tokens, experts, top_k = x.shape[0], cfg.num_experts, cfg.top_k
        routes = tokens * top_k
        if capacity_factor is None:
            expert, gate = routing.experts.reshape(-1), routing.gates.reshape(-1)
            token = jnp.repeat(jnp.arange(tokens), top_k)
            order = jnp.argsort(expert, stable=True)
            sizes = jnp.zeros((experts,), jnp.int32).at[expert].add(1)
            h = jax.lax.ragged_dot(x[token[order]], m.gate_up.reshape(experts, d, -1), sizes)
            h = h.reshape(routes, 2, -1)
            from jla.layers.mlp import swiglu

            y = jax.lax.ragged_dot(swiglu(h) * gate[order][:, None], m.down, sizes)
            inverse = jnp.zeros((routes,), jnp.int32).at[order].set(jnp.arange(routes))
            return y[inverse].reshape(tokens, top_k, d).sum(axis=1)
        capacity = int(np.ceil(capacity_factor * routes / experts))
        expert, gate = routing.experts.T.reshape(-1), routing.gates.T.reshape(-1)
        token = jnp.tile(jnp.arange(tokens), top_k)
        arrival = jnp.cumsum(jax.nn.one_hot(expert, experts, dtype=jnp.int32), axis=0) - 1
        position = jnp.take_along_axis(arrival, expert[:, None], axis=1)[:, 0]
        kept = position < capacity
        slot = jnp.where(kept, position, capacity)
        buf = jnp.zeros((experts, capacity + 1, d)).at[expert, slot].add(x[token])[:, :capacity]
        w = jnp.zeros((experts, capacity + 1)).at[expert, slot].add(jnp.where(kept, gate, 0.0))[:, :capacity]
        from jla.layers.mlp import swiglu

        h = jnp.einsum("ecd,edgf->ecgf", buf, m.gate_up)
        y = jnp.einsum("ecf,efd->ecd", swiglu(h) * w[..., None], m.down)
        y = jnp.pad(y, ((0, 0), (0, 1), (0, 0)))
        return y[expert, slot].reshape(top_k, tokens, d).sum(axis=0)

    with jax.default_matmul_precision("highest"):
        out, stats = moe(x)
        if capacity_factor is not None:
            assert float(stats.dropped_fraction) > 0.05
        np.testing.assert_allclose(out, scatter_formulation(moe, x), rtol=1e-5, atol=1e-6)
        loss = lambda f: lambda m, x: jnp.sum(jnp.sin(f(m, x)))  # noqa: E731
        got = jax.grad(loss(lambda m, x: m(x)[0]), argnums=(0, 1))(moe, x)
        want = jax.grad(loss(scatter_formulation), argnums=(0, 1))(moe, x)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
