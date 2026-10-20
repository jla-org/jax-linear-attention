# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mixture of experts: dispatch paths against the dense oracle, routing, capacity and aux losses."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.mlp import swiglu
from jla.layers.moe import MoE
from jla.models.config import MoEConfig

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


D = 8


def _moe(seed=0, compute_dtype="float32", **kw) -> MoE:
    cfg = dict(num_experts=4, top_k=2, d_ff_expert=6)
    cfg.update(kw)
    return MoE(D, MoEConfig(**cfg), key=jax.random.key(seed), std=0.3, compute_dtype=compute_dtype)


def _x(shape=(2, 5, D), seed=1):
    return jax.random.normal(jax.random.key(seed), shape)


def _grads(fn, moe, x):
    return jax.grad(lambda m, x: jnp.sum(jnp.square(fn(m, x))), argnums=(0, 1))(moe, x)


def _assert_trees_close(a, b, **tol):
    for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(u, v, **tol)


@pytest.mark.parametrize("router", ["softmax", "sigmoid"])
@pytest.mark.parametrize("capacity_factor", [None, 8.0])
@pytest.mark.parametrize("shared", [0, 1])
def test_dispatch_matches_dense_oracle_values_and_gradients(router, capacity_factor, shared):
    moe = _moe(router=router, capacity_factor=capacity_factor, num_shared_experts=shared)
    x = _x()
    out, stats = moe(x)
    np.testing.assert_allclose(out, moe.reference(x), rtol=2e-5, atol=2e-6)
    assert float(stats.dropped_fraction) == 0.0
    got = _grads(lambda m, x: m(x)[0], moe, x)
    want = _grads(lambda m, x: m.reference(x), moe, x)
    _assert_trees_close(got, want, rtol=5e-5, atol=5e-6)


def test_capacity_and_dropless_agree_including_router_gradients():
    moe_cap = _moe(capacity_factor=8.0, aux_loss_coef=0.3, z_loss_coef=0.1)
    moe_free = _moe(capacity_factor=None, aux_loss_coef=0.3, z_loss_coef=0.1)
    x = _x()

    def objective(m, x):
        out, stats = m(x)
        return jnp.sum(jnp.square(out)) + stats.aux_loss

    np.testing.assert_allclose(objective(moe_cap, x), objective(moe_free, x), rtol=1e-6)
    _assert_trees_close(jax.grad(objective)(moe_cap, x), jax.grad(objective)(moe_free, x), rtol=5e-5, atol=5e-6)


def _skewed(moe: MoE) -> MoE:
    """Every token prefers experts 0 and 1 (positive inputs, column-constant router)."""
    router = jnp.broadcast_to(jnp.array([10.0, 9.0, -9.0, -10.0]), (D, 4))
    return eqx.tree_at(lambda m: m.router, moe, router)


def test_dropless_never_drops_extremely_imbalanced_routes():
    moe = _skewed(_moe(capacity_factor=None))
    x = jnp.arange(64, dtype=jnp.float32).reshape(8, D) / 64.0
    out, stats = jax.jit(lambda m, x: m(x))(moe, x)
    assert float(stats.dropped_fraction) == 0.0
    np.testing.assert_allclose(out, moe.reference(x), rtol=2e-5, atol=2e-6)
    assert float(stats.balance_loss) > 1.5  # all routes on two of four experts


def test_capacity_drops_overflow_and_stays_finite():
    moe = _skewed(_moe(capacity_factor=1.0))
    x = jnp.arange(64, dtype=jnp.float32).reshape(8, D) / 64.0
    out, stats = jax.jit(lambda m, x: m(x))(moe, x)
    # capacity = ceil(16 routes / 4 experts) = 4 slots: experts 0 and 1 keep 4 of 8 routes each.
    np.testing.assert_allclose(stats.dropped_fraction, 0.5)
    assert np.isfinite(np.asarray(out)).all()
    first_four = moe.reference(x[:4])
    np.testing.assert_allclose(out[:4], first_four, rtol=2e-5, atol=2e-6)  # earliest tokens are admitted
    np.testing.assert_allclose(out[4:], 0.0, atol=1e-7)  # later tokens lost both routes


def test_first_choices_claim_capacity_before_second_choices():
    moe = _moe(capacity_factor=1.0)  # 4 tokens x 2 routes / 4 experts -> 2 slots per expert
    x = jnp.eye(4, D)
    logits = np.full((D, 4), -5.0, np.float32)
    logits[0, :2] = logits[1, :2] = [3.0, 2.0]  # tokens 0, 1: first expert 0, second expert 1
    logits[2, :2] = logits[3, :2] = [2.0, 3.0]  # tokens 2, 3: first expert 1, second expert 0
    moe = eqx.tree_at(lambda m: m.router, moe, jnp.asarray(logits))
    out, stats = moe(x)
    np.testing.assert_allclose(stats.dropped_fraction, 0.5)
    routing = moe._route(x)
    first = routing.experts[:, 0]
    h = jnp.einsum("nd,ndgf->ngf", x, moe.gate_up[first])
    expected = routing.gates[:, :1] * jnp.einsum("nf,nfd->nd", swiglu(h), moe.down[first])
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-6)


def test_routing_options():
    x = _x((6, D), 3)
    raw = _moe(normalize_top_k=False)
    routing = raw._route(x)
    probs = jax.nn.softmax(x @ raw.router, axis=-1)
    np.testing.assert_allclose(routing.gates, jnp.sort(probs, axis=-1)[:, ::-1][:, :2], rtol=1e-6)
    norm = _moe()._route(x)
    np.testing.assert_allclose(jnp.sum(norm.gates, axis=-1), 1.0, rtol=1e-6)
    sig = _moe(router="sigmoid", normalize_top_k=False)._route(x)
    scores = jax.nn.sigmoid(x @ _moe(router="sigmoid").router)
    np.testing.assert_allclose(sig.gates, jnp.sort(scores, axis=-1)[:, ::-1][:, :2], rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(sig.probs, axis=-1), 1.0, rtol=1e-6)


def test_aux_losses():
    moe = _moe(aux_loss_coef=0.01, z_loss_coef=0.001)
    x = _x((10, D), 4)
    _, stats = moe(x)
    logits = x @ moe.router
    np.testing.assert_allclose(stats.z_loss, jnp.mean(jax.nn.logsumexp(logits, axis=-1) ** 2), rtol=1e-5)
    np.testing.assert_allclose(stats.aux_loss, 0.01 * stats.balance_loss + 0.001 * stats.z_loss, rtol=1e-6)
    uniform = eqx.tree_at(lambda m: m.router, moe, jnp.zeros_like(moe.router))
    np.testing.assert_allclose(uniform(x)[1].balance_loss, 1.0, rtol=1e-6)


def test_bf16_router_uses_bf16_operands_with_f32_logits():
    moe = _moe(compute_dtype="bfloat16")
    x = jnp.ones((2, D), jnp.bfloat16)
    jaxpr = jax.make_jaxpr(lambda v: moe(v)[0])(x).jaxpr
    router = next(e for e in jaxpr.eqns if e.primitive.name == "dot_general")
    assert tuple(v.aval.dtype for v in router.invars) == (jnp.dtype(jnp.bfloat16),) * 2
    assert router.outvars[0].aval.dtype == jnp.dtype(jnp.float32)
    assert moe(x)[0].dtype == jnp.bfloat16


def test_config_validation():
    with pytest.raises(ValueError, match="top_k"):
        MoEConfig(num_experts=4, top_k=5, d_ff_expert=8)
    with pytest.raises(ValueError, match="router"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, router="relu")
    with pytest.raises(ValueError, match="capacity_factor"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=0.0)
