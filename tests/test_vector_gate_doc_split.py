# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KDA / GDN-2 forward state pass cut at document starts (``vector_gate.piece_table``).

With packed documents the pass walks each row's chain in pieces that start at chunks where a
document starts, from a zero state. The outputs, the final state, the saved states and the
gradients must equal those of one chain per row (``SPLIT_DOCUMENTS = False``) exactly: equal
values (``assert_array_equal``: a zero may differ in sign only), not merely close ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import F32, dense_segments, gate_kwargs, inputs, segments

from jla import ops
from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.reference import segment_resets
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

gpu = pytest.mark.skipif(not vgo.is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _layout(name: str, B: int, T: int) -> jax.Array:
    """``segment_ids [B, T]`` of a named document layout."""
    starts = np.zeros((B, T), bool)
    if name == "random":
        return segments(3, B, T)
    if name == "dense":
        return dense_segments(B, T)
    if name == "on_chunks":  # every document starts on a chunk boundary
        starts[:, 64::192] = True
    elif name == "first_and_last":  # a start in chunk 0 and one in the last chunk
        starts[:, 5] = True
        starts[:, T - 3] = True
    elif name == "one_document":
        pass
    elif name == "tail":  # all documents in the last quarter
        starts[:, 3 * T // 4::37] = True
    else:
        raise ValueError(name)
    return jnp.cumsum(jnp.asarray(starts, jnp.int32), axis=1)


def _run(kind, x, seg, *, split, grad, h0=True):
    old = vgo.SPLIT_DOCUMENTS
    vgo.SPLIT_DOCUMENTS = split
    try:
        def fwd(q, k, v, g, gates, state):
            return ops.linear_attention(kind, q, k, v, g=g, impl="mosaic_gpu", initial_state=state,
                                        output_final_state=True, segment_ids=seg, **gate_kwargs(kind, gates))

        state = x["h0"] if h0 else None
        if not grad:
            return jax.jit(fwd)(x["q"], x["k"], x["v"], x["g"], x["gates"], state)

        def loss(q, k, v, g, gates, state):
            o, final = fwd(q, k, v, g, gates, state)
            return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(final * x["dht"])

        argnums = (0, 1, 2, 3, 4, 5) if h0 else (0, 1, 2, 3, 4)
        return jax.jit(jax.value_and_grad(loss, argnums=argnums))(x["q"], x["k"], x["v"], x["g"], x["gates"], state)
    finally:
        vgo.SPLIT_DOCUMENTS = old


def _assert_same(got, want):
    got_leaves, want_leaves = jax.tree.leaves(got), jax.tree.leaves(want)
    assert len(got_leaves) == len(want_leaves)
    for i, (a, b) in enumerate(zip(got_leaves, want_leaves)):
        np.testing.assert_array_equal(np.asarray(a.astype(F32)), np.asarray(b.astype(F32)), err_msg=f"leaf {i}")


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize(
    "shape",
    [(1, 1024, 2, 128, 128), (1, 2048, 12, 128, 128), (3, 1000, 4, 64, 64), (2, 777, 3, 128, 64)],
    ids=["bh2_wy", "bh12_fused", "b3_odd_k64", "b2_t777_v64"],
)
@pytest.mark.parametrize("layout", ["random", "on_chunks", "first_and_last", "tail", "one_document"])
def test_forward_split_is_exact(kind, shape, layout):
    B, T, H, K, V = shape
    x = inputs(11, kind, B, T, H, K, V)
    seg = _layout(layout, B, T)
    _assert_same(_run(kind, x, seg, split=True, grad=False), _run(kind, x, seg, split=False, grad=False))


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("case", [
    dict(shape=(1, 1024, 2, 128, 128), layout="random", h0=True),
    dict(shape=(1, 1024, 2, 128, 128), layout="dense", h0=False),
    dict(shape=(2, 515, 2, 64, 64), layout="first_and_last", h0=True),
    dict(shape=(1, 2048, 6, 128, 128), layout="random", h0=True, huge=True),
], ids=["bh2_random", "bh2_dense_docs", "b2_odd", "bh6_huge"])
def test_training_split_is_exact(kind, case):
    """The training forward (saved states) and all gradients, split against one chain per row."""
    B, T, H, K, V = case["shape"]
    x = inputs(12, kind, B, T, H, K, V, huge=case.get("huge", False))
    seg = _layout(case["layout"], B, T)
    got = _run(kind, x, seg, split=True, grad=True, h0=case["h0"])
    want = _run(kind, x, seg, split=False, grad=True, h0=case["h0"])
    _assert_same(got, want)


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("layout", ["random", "dense"])
def test_training_residuals_are_the_same(kind, layout):
    """The training forward's outputs and residuals (``G``, states, ``v_new``, ...): the reset decay
    applied in the cumsum kernel with the split pass, against the decay applied in XLA (``g - 128``
    at segment starts) with one chain per row."""
    B, T, H, K, V = 1, 1024, 2, 128, 128
    x = inputs(13, kind, B, T, H, K, V)
    seg = _layout(layout, B, T)
    starts = segment_resets(seg)
    g = jnp.where(starts[:, :, None, None], x["g"] - vgo._RESET_DECAY, x["g"])
    gates = x["gates"] if kind == "kda" else tuple(a.astype(jnp.bfloat16) for a in x["gates"])
    cfg = vg.Config(batch=B, seq_len=T, heads=H, key_dim=K, value_dim=V, scale=K**-0.5, channel_gates=kind == "gdn2")
    out_s, res_s = jax.jit(lambda *a: vgo._forward(cfg, *a))(x["q"], x["k"], x["v"], x["g"], gates, x["h0"],
                                                              starts.astype(F32))
    out_w, res_w = jax.jit(lambda *a: vgo._forward(cfg, *a))(x["q"], x["k"], x["v"], g, gates, x["h0"])
    assert res_w[-1] is None and res_s[-1] is not None  # the chunks with a segment start, for the backward
    _assert_same((out_s, res_s[:-1]), (out_w, res_w[:-1]))


@pytest.mark.parametrize("n_pieces", [2, 5, 16])
def test_piece_table_covers_every_chunk_once(n_pieces):
    """Each chunk's results are stored by exactly one piece, pieces after the first start at a kill
    chunk, and exactly one piece per row writes the final state (pure jnp: any backend)."""
    rng = np.random.default_rng(n_pieces)
    nc = 40
    kill = rng.random((6, nc)) < np.array([0.0, 0.02, 0.1, 0.5, 1.0, 0.1])[:, None]
    kill[:, 0] = False
    kill[5, nc - 1] = True
    table = np.asarray(vg.piece_table(jnp.asarray(kill), n_pieces)).reshape(6, n_pieces, 4)
    for b in range(6):
        stored = np.zeros(nc, int)
        assert table[b, :, 3].sum() == 1
        for first, count, skip, last in table[b]:
            if count == 0:
                assert last == 0
                continue
            assert first == 0 or (kill[b, first] and skip == 1)
            stored[first + skip:first + count] += 1
            if last:
                assert first + count == nc
        np.testing.assert_array_equal(stored, 1)
