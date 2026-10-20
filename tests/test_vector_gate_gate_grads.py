# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Gradients of the softplus gate's parameters under strong decays (KDA, GDN-2, Mosaic GPU kernels).

The layers' gate is ``g = -exp(A_log) * softplus(raw + dt_bias)``, so ``dA_log = sum(dg * g)``: every
per-token error of ``dG`` (the gradient of the chunk-local cumulative gate) reaches ``A_log`` multiplied by
the cumulative log-decay, up to hundreds under strong decays. Each pair term of the score matrices enters
``dG`` twice with opposite signs (row and column direction of ``vg_bwd_intra``), so the two must be rounded
alike: when the close pairs ran on bf16 tensor cores they were not, and ``dA_log`` was 2-5 % off per head
(``ddt_bias`` 1 %), against 0.2 % for the chunked op. ``A_log = log U(1, 16)`` gives mean log-decays near
-11 per token.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

F32, BF16 = jnp.float32, jnp.bfloat16
gpu = pytest.mark.skipif(not vgo.is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _inputs(kind: str, T: int = 1000, H: int = 4, K: int = 128):
    ks = iter(jax.random.split(jax.random.key(0), 16))

    def normal(shape, scale=1.0):
        return scale * jax.random.normal(next(ks), shape, F32)

    x = dict(q=normal((1, T, H, K)), k=normal((1, T, H, K)), v=normal((1, T, H, K)), raw=normal((1, T, H, K)),
             A_log=jnp.log(jax.random.uniform(next(ks), (H,), F32, 1.0, 16.0)), dt_bias=normal((H * K,), 0.1))
    if kind == "kda":
        x["beta"] = normal((1, T, H))
    else:
        x["b"], x["w"] = normal((1, T, H, K)), normal((1, T, H, K))
    x["h0"] = normal((1, H, K, K), 0.5)
    do, dht = normal((1, T, H, K)), normal((1, H, K, K))
    for name in ("q", "k", "v"):
        x[name] = x[name].astype(BF16).astype(F32)  # every impl sees the same values
    return x, do, dht


def _grads(kind: str, impl: str, x, do, dht):
    names = list(x)
    H, K = x["q"].shape[2:]
    io = F32 if impl == "reference" else BF16

    def f(*primals):
        xs = dict(zip(names, primals))
        g = -jnp.exp(xs["A_log"])[:, None] * jax.nn.softplus(xs["raw"] + xs["dt_bias"].reshape(H, K))
        gates = {n: jax.nn.sigmoid(xs[n]) for n in (("beta",) if kind == "kda" else ("b", "w"))}
        return ops.linear_attention(kind, xs["q"].astype(io), xs["k"].astype(io), xs["v"].astype(io), impl=impl,
                                    g=g, **gates, scale=K**-0.5, initial_state=xs["h0"], output_final_state=True,
                                    use_qk_l2norm=True)

    @jax.jit
    def fwd_bwd(*primals):
        (o, final), vjp = jax.vjp(f, *primals)
        return vjp((do.astype(o.dtype), dht))

    return {n: np.asarray(gr, np.float64) for n, gr in zip(names, fwd_bwd(*[x[n] for n in names]))}


def _rel_l2(x, want):
    return float(np.linalg.norm(x - want) / np.linalg.norm(want))


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
def test_softplus_gate_parameter_grads_under_strong_decays(kind):
    x, do, dht = _inputs(kind)
    want = _grads(kind, "reference", x, do, dht)
    got = _grads(kind, "mosaic_gpu", x, do, dht)
    # Measured 17 Oct 2026 (H100): KDA dA_log 9.1e-3 (per head at most 1.9 %), ddt_bias 3.7e-3; GDN-2
    # 3.3e-3 and 3.2e-3. The kernel before the fix: KDA 2.8e-2 (4.6 %) and 9.1e-3, GDN-2 ddt_bias 1.2e-2.
    assert _rel_l2(got["A_log"], want["A_log"]) < 1.5e-2
    assert np.max(np.abs(got["A_log"] - want["A_log"]) / np.abs(want["A_log"])) < 3e-2
    assert _rel_l2(got["dt_bias"], want["dt_bias"]) < 6e-3
    for name in set(x) - {"A_log", "dt_bias"}:  # the other gradients stay at bf16 noise
        assert _rel_l2(got[name], want[name]) < 1e-2, name
