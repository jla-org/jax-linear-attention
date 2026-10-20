# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped key heads (GQA in the linear-attention layers, Mamba-2 B/C groups): the gradient precision.

The layers repeat grouped ``q``/``k`` (B/C) to the value heads before the op. The repeat's
transpose sums the per-head gradients; in bfloat16 that sum lost about 2x accuracy against
upstream at 8 heads per group (``benchmarks/xcheck_scalar_gate.py``: ``mosaic_rep_bf16``
against ``mosaic_rep_f32`` and FLA). The repeat therefore goes through float32: the forward
values are bitwise the same and the gradient is the float32 sum rounded once.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import linear_attention, ssm


def _check(repeat, groups: int, heads: int) -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 96, groups, 128), jnp.float32).astype(jnp.bfloat16)
    cot = jax.random.normal(jax.random.PRNGKey(1), (2, 96, heads, 128), jnp.float32).astype(jnp.bfloat16)
    y, vjp = jax.vjp(repeat, x)
    assert y.dtype == jnp.bfloat16 and y.shape == cot.shape
    want_y = np.repeat(np.asarray(x, np.float32), heads // groups, axis=2)
    np.testing.assert_array_equal(np.asarray(y, np.float32), want_y)
    (dx,) = vjp(cot)
    assert dx.dtype == jnp.bfloat16
    exact = np.asarray(cot, np.float64).reshape(2, 96, groups, heads // groups, 128).sum(axis=3)
    got = np.asarray(dx, np.float64)
    # One rounding of a float32 sum: within one bfloat16 ulp (2^-8 relative) of the exact sum everywhere.
    np.testing.assert_array_less(np.abs(got - exact), 2.0**-8 * np.abs(exact) + 1e-30)


@pytest.mark.parametrize("groups,heads", [(1, 8), (1, 48), (2, 32)])
def test_ssm_repeat_groups_gradient_is_float32_sum(groups, heads):
    _check(lambda x: ssm._repeat_groups(x, heads), groups, heads)


@pytest.mark.parametrize("k_heads,v_heads", [(1, 16), (4, 32)])
def test_linear_attention_repeat_k_heads_gradient_is_float32_sum(k_heads, v_heads):
    mixer = SimpleNamespace(cfg=SimpleNamespace(num_k_heads=k_heads, num_v_heads=v_heads))
    _check(lambda x: linear_attention.LinearAttentionMixer._repeat_k_heads(mixer, x), k_heads, v_heads)


def _siso_inputs(groups: int, heads: int, batch: int = 1, length: int = 48, n: int = 32, p: int = 16) -> dict:
    ks = jax.random.split(jax.random.PRNGKey(4), 9)
    bf16 = jnp.bfloat16
    return dict(
        q=jax.random.normal(ks[0], (batch, length, groups, n)).astype(bf16),
        k=jax.random.normal(ks[1], (batch, length, groups, n)).astype(bf16),
        v=jax.random.normal(ks[2], (batch, length, heads, p)).astype(bf16),
        adt=-jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads))),
        dt=jax.nn.softplus(jax.random.normal(ks[4], (batch, length, heads))),
        trap=jax.random.normal(ks[5], (batch, length, heads)),
        q_bias=1.0 + 0.1 * jax.random.normal(ks[6], (heads, n)),
        k_bias=1.0 + 0.1 * jax.random.normal(ks[7], (heads, n)),
        angles=2.0 * jax.random.normal(ks[8], (batch, length, heads, n // 4)),
    )


@pytest.mark.parametrize("groups", [1, 2])
def test_mamba3_siso_group_gradient_is_float32_sum(groups):
    """Mamba-3 SISO's XLA path: the B/C gradient is the float32 sum over the heads of a group, rounded once.

    With float32 B/C the same calculation gives the unrounded sum; the bfloat16 gradient must be within one
    bfloat16 ulp of it (rounding every head's float32 gradient to bfloat16 before the sum does not hold that).
    """
    inp = _siso_inputs(groups, heads=16)
    rest = {n: x for n, x in inp.items() if n not in ("q", "k")}
    cot = jax.random.normal(jax.random.PRNGKey(3), inp["v"].shape, jnp.float32).astype(jnp.bfloat16)

    def grads(q, k):
        _, vjp = jax.vjp(lambda q, k: ssm.mamba3_siso(q, k, **rest, impl="reference")[0], q, k)
        return vjp(cot)

    got = grads(inp["q"], inp["k"])
    exact = grads(inp["q"].astype(jnp.float32), inp["k"].astype(jnp.float32))
    for g, e in zip(got, exact):
        assert g.dtype == jnp.bfloat16 and e.dtype == jnp.float32
        g, e = np.asarray(g, np.float64), np.asarray(e, np.float64)
        np.testing.assert_array_less(np.abs(g - e), 2.0**-8 * np.abs(e) + 1e-30)


def test_repeat_is_identity_without_groups():
    x = jnp.ones((1, 4, 8, 64), jnp.bfloat16)
    assert ssm._repeat_groups(x, 8) is x
    mixer = SimpleNamespace(cfg=SimpleNamespace(num_k_heads=8, num_v_heads=8))
    assert linear_attention.LinearAttentionMixer._repeat_k_heads(mixer, x) is x
