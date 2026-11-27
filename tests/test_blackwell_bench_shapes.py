# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's own shapes, checked for correctness on Blackwell (``JLA_BLACKWELL=1``).

The speed evaluator (``benchmarks/bench_gated_delta_rule.py``) times shapes larger than the unit tests use; this
checks that at exactly those shapes the kernels' output and gradients agree with the plain-JAX chunked path, so a
fast but wrong kernel cannot score.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops.mosaic_gpu import _common as cm

pytestmark = pytest.mark.skipif(not cm.blackwell_enabled(), reason="needs a Blackwell GPU with JLA_BLACKWELL=1")

SHAPES = [(8, 4096, 16, 128, 128), (2, 16384, 16, 128, 128), (8, 4096, 32, 64, 64)]


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# The benchmark's gate (offset 2) decays the state ~3e-4 per 64-token chunk; offset 6 keeps ~0.85 of it, so the
# state passed between chunks is checked too.
@pytest.mark.parametrize("gate_offset", [2.0, 6.0], ids=["bench_gate", "mild_gate"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "B{}T{}H{}K{}V{}".format(*s))
def test_bench_shape_matches_chunked(shape, gate_offset):
    B, T, H, K, V = shape
    ks = jax.random.split(jax.random.key(7), 6)
    k = jax.random.normal(ks[1], (B, T, H, K))
    x = dict(q=jax.random.normal(ks[0], (B, T, H, K)).astype(jnp.bfloat16),
             k=(k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(jnp.bfloat16),
             v=jax.random.normal(ks[2], (B, T, H, V)).astype(jnp.bfloat16))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - gate_offset)
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H)))
    do = jax.random.normal(ks[5], (B, T, H, V))

    def loss(impl, q, k, v, g, beta):
        if impl == "chunked":
            q, k, v = (a.astype(jnp.float32) for a in (q, k, v))
        o, _ = ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, impl=impl)
        return jnp.sum(o.astype(jnp.float32) * do), o

    outs = {}
    for impl in ("mosaic_gpu", "chunked"):
        (_, o), grads = jax.jit(jax.value_and_grad(lambda *a, impl=impl: loss(impl, *a), argnums=(0, 1, 2, 3, 4),
                                                   has_aux=True))(x["q"], x["k"], x["v"], g, beta)
        outs[impl] = [o.astype(jnp.float32), *(gr.astype(jnp.float32) for gr in grads)]
    for name, got, want in zip(("o", "dq", "dk", "dv", "dg", "dbeta"), outs["mosaic_gpu"], outs["chunked"]):
        err = _rel(got, want)
        assert np.isfinite(np.asarray(got)).all() and np.isfinite(np.asarray(want)).all(), name
        assert float(jnp.abs(want).max()) > 0.0, f"{name}: the reference is all zeros"
        assert err < 2e-2, f"{name}: relative L2 error {err:.2e}"
