# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``intra`` with the solve fused in (``T = (I + Akk)^-1`` from SMEM, no ``vg_solve`` pass; opt-in).

``T`` must be bit for bit the separate ``solve`` of the same ``Akk`` on the factored (bounded)
path, the exact path and a mix, for KDA and GDN-2, ``K`` in {64, 128}, with and without the decayed
tiles; and the whole op with ``fuse_solve=True`` (forward without ``wy``, forward with it,
gradients) must give exactly the outputs of the default pipeline (``fuse_solve=False``, the
separate ``vg_solve``). Needs an H100.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import assert_close, config, from_heads, inputs, kernel_gates, oracle_forward

from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

pytestmark = pytest.mark.skipif(not vgo.is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _equal(got, want, name):
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32), err_msg=name)


@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("key_dim", [128, 64])
@pytest.mark.parametrize("decays", [False, True], ids=["scores", "decays"])
def test_fused_solve_is_the_separate_solve(kind, key_dim, decays):
    """Chunks 0, 1, 3 bounded (factored path), chunk 2 not (a -100 log-decay: exact path); then
    every chunk on the exact path."""
    x = inputs(3, kind, B=1, T=256, H=2, K=key_dim, V=128)
    x["g"] = x["g"].at[0, 130].set(-100.0)
    want = oracle_forward(kind, x)
    cfg = config(kind, x)
    g_cum, marks, _, bounded = jax.jit(lambda g: vg.chunk_cumsum(cfg, g))(x["g"])
    assert bounded[0, :, 0].tolist() == [1, 1, 0, 1]
    erase, _ = kernel_gates(kind, x)
    run = jax.jit(lambda *a, solve: vg.intra(cfg, *a, decays=decays, solve=solve), static_argnames="solve")
    for flags in (bounded, jnp.zeros_like(bounded)):
        sep = run(x["q"], x["k"], g_cum, marks, erase, flags, solve=False)
        fused = run(x["q"], x["k"], g_cum, marks, erase, flags, solve=True)
        t_sep = jax.jit(lambda a: vg.solve(cfg, a))(sep[1])
        assert fused[1].dtype == jnp.bfloat16 and fused[1].shape == t_sep.shape
        _equal(fused[1], t_sep, "T")
        for name, a, b in zip(("Aqk", "qg", "-kd", "kg"), (fused[0], *fused[2:]), (sep[0], *sep[2:])):
            _equal(a, b, name)
        assert_close(from_heads(fused[1]), want["t"], 1e-2, "T vs oracle")


def test_fused_solve_rejects_the_interpreter():
    x = inputs(0, "kda", B=1, T=64, H=1, K=128, V=128)
    cfg = dataclasses.replace(config("kda", x), interpret=True)
    g_cum = jnp.zeros((1, 64, 1, 128), jnp.float32)
    with pytest.raises(ValueError, match="interpreter"):
        vg.intra(cfg, x["q"], x["k"], g_cum, g_cum[:, ::8], jnp.zeros((1, 1, 64)), jnp.ones((1, 1, 1), jnp.int32),
                 solve=True)


def _op(kind, x, fuse_solve, segment_ids=None, l2norm=False):
    gates = tuple(x["gates"])

    def fn(q, k, v, g, gates, h0):
        return vgo.chunked_vector_gate(q, k, v, g, gates, scale=q.shape[-1] ** -0.5, initial_state=h0,
                                       segment_ids=segment_ids, l2norm=l2norm, fuse_solve=fuse_solve)

    return fn, (x["q"], x["k"], x["v"], x["g"], gates, x["h0"])


@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("shape", [(3, 200, 4, 128, 128), (1, 130, 2, 64, 128), (2, 300, 6, 64, 64)],
                         ids=["bh12_k128", "bh2_k64", "bh12_k64v64"])
def test_op_matches_the_separate_solve(kind, shape):
    """Forward (with and without wy) and gradients, packed documents, a huge decay: identical
    results with and without the fused solve."""
    B, T, H, K, V = shape
    x = inputs(7, kind, B, T, H, K, V)
    x["g"] = x["g"].at[:, 70].set(-100.0)
    seg = jnp.cumsum((jnp.arange(T)[None] % 97 == 0) | (jnp.arange(T)[None] == 150), axis=1).astype(jnp.int32)
    seg = jnp.broadcast_to(seg, (B, T))
    cot = jax.random.normal(jax.random.key(1), (B, T, H, V), jnp.float32)
    outs = {}
    for fuse in (False, True):
        fn, args = _op(kind, x, fuse, segment_ids=seg)
        fwd = jax.jit(fn)(*args)
        loss = jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a)[0].astype(jnp.float32) * cot) + jnp.sum(fn(*a)[1]),
                                argnums=(0, 1, 2, 3, 4, 5)))
        outs[fuse] = (fwd, loss(*args))
    for got, want in zip(jax.tree.leaves(outs[True]), jax.tree.leaves(outs[False])):
        _equal(got, want, "fused vs separate solve")


@pytest.mark.parametrize("kind", ["kda", "gdn2"])
def test_l2norm_forward_matches_the_separate_solve(kind):
    """The forward without wy with the in-intra l2norm (B H > 8)."""
    x = inputs(9, kind, 3, 130, 4, 128, 128)
    got = jax.jit(_op(kind, x, True, l2norm=True)[0])(*_op(kind, x, True, l2norm=True)[1])
    want = jax.jit(_op(kind, x, False, l2norm=True)[0])(*_op(kind, x, False, l2norm=True)[1])
    for a, b in zip(got, want):
        _equal(a, b, "l2norm forward")
