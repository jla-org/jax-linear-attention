# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Extra correctness gates for the Blackwell kernels (run with ``JLA_BLACKWELL=1`` on sm_100/sm_103).

Beyond the tolerance tests of ``test_mosaic_gpu_gated_delta_rule.py``: the result is the same bits run to run,
a NaN in one input token reaches exactly the outputs the reference's NaN reaches (no NaN is turned into an
ordinary number and no new NaN appears), and the result does not depend on what earlier calls left in memory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops.mosaic_gpu import _common as cm

pytestmark = pytest.mark.skipif(not cm.blackwell_enabled(), reason="needs a Blackwell GPU with JLA_BLACKWELL=1")

KINDS = {"simple_gla": ("g",), "deltanet": ("beta",), "gated_deltanet": ("g", "beta")}


def _inputs(seed, B=2, T=320, H=4, K=128, V=128):
    ks = jax.random.split(jax.random.key(seed), 6)
    k = jax.random.normal(ks[1], (B, T, H, K))
    return dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(jnp.bfloat16),
        k=(k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(jnp.bfloat16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(jnp.bfloat16),
        g=-jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H))),
        do=jax.random.normal(ks[5], (B, T, H, V)),
    )


def _run(kind, impl, x):
    gates = {n: x[n] for n in KINDS[kind]}

    def loss(q, k, v):
        o, _ = ops.linear_attention(kind, q, k, v, impl=impl, output_final_state=True, **gates)
        return jnp.sum(o.astype(jnp.float32) * x["do"]), o

    (_, o), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(x["q"], x["k"], x["v"])
    return [np.asarray(a.astype(jnp.float32)) for a in (o, *grads)]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_deterministic(kind):
    x = _inputs(1)
    a, b = _run(kind, "mosaic_gpu", x), _run(kind, "mosaic_gpu", x)
    for i, (u, w) in enumerate(zip(a, b)):
        assert u.tobytes() == w.tobytes(), f"output {i} differs between two identical calls"


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_nan_reaches_the_same_outputs_as_the_reference(kind):
    x = _inputs(2)
    x["v"] = x["v"].at[1, 130, 2, 5].set(jnp.nan)  # one value channel of one token of one head
    o_ref = np.asarray(ops.linear_attention(kind, *(x[n].astype(jnp.float32) for n in "qkv"), impl="chunked",
                                            **{n: x[n] for n in KINDS[kind]})[0])
    o = np.asarray(ops.linear_attention(kind, x["q"], x["k"], x["v"], impl="mosaic_gpu",
                                        **{n: x[n] for n in KINDS[kind]})[0].astype(jnp.float32))
    assert np.isnan(o_ref).any(), "the probe must put NaN into the reference output"
    assert not np.isnan(o_ref).all(), "the probe must leave most of the reference output finite"
    np.testing.assert_array_equal(np.isnan(o), np.isnan(o_ref))
    ok = ~np.isnan(o_ref)
    assert np.isfinite(o[ok]).all(), "no infinity where the reference is finite"
    err = np.linalg.norm(o[ok] - o_ref[ok]) / np.linalg.norm(o_ref[ok])
    assert err < 2e-2, f"finite outputs: relative L2 error {err:.2e}"


def test_independent_of_previous_calls():
    x = _inputs(3)
    first = _run("gated_deltanet", "mosaic_gpu", x)
    _run("gated_deltanet", "mosaic_gpu", _inputs(4, B=1, T=1000, H=8))  # a different shape in between
    again = _run("gated_deltanet", "mosaic_gpu", x)
    for u, w in zip(first, again):
        assert u.tobytes() == w.tobytes()
