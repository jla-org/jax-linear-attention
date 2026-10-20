# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""Future gates must not change prefix outputs, nor the gradients of a loss on the prefix (KDA, GDN-2); a
kernel prefill followed by single-token reference steps (how the layers decode) matches the full pass.

The reference and the chunked op are causal bit for bit. The Mosaic GPU kernels pick a path per chunk
(``vector_gate.chunk_cumsum``'s ``bounded`` flag) and factor the decay of a bounded chunk's 32-token diagonal
blocks through their 16th token, so future gates inside the same chunk may change prefix results by rounding
(both paths are exact), never by more: bit for bit when neither happens, within bf16 noise otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla import ops
from jla.ops import reference as ref
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

F32, BF16 = jnp.float32, jnp.bfloat16
gpu = pytest.mark.skipif(not vgo.is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _inputs(kind: str, T: int = 200, H: int = 2, K: int = 128):
    ks = jax.random.split(jax.random.key(3), 8)
    x = dict(
        q=jax.random.normal(ks[0], (1, T, H, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (1, T, H, K))).astype(BF16),
        v=jax.random.normal(ks[2], (1, T, H, K)).astype(BF16),
        g=jax.nn.log_sigmoid(jax.random.normal(ks[3], (1, T, H, K))),
    )
    if kind == "kda":
        gates = dict(beta=jax.nn.sigmoid(jax.random.normal(ks[4], (1, T, H))))
    else:
        gates = dict(b=jax.nn.sigmoid(jax.random.normal(ks[4], (1, T, H, K))),
                     w=jax.nn.sigmoid(jax.random.normal(ks[5], (1, T, H, K))))
    do = jax.random.normal(ks[6], (1, T, H, K))
    return x, gates, do


def _prefix_results(kind: str, impl: str, x, gates, do, P: int):
    """Prefix outputs and the gradients of ``sum(o[:, :P] * do[:, :P])`` (prefix rows only)."""

    def loss(q, k, v, g, *gate_values):
        o, _ = ops.linear_attention(kind, q, k, v, impl=impl, g=g, **dict(zip(gates, gate_values)))
        return jnp.sum(o[:, :P].astype(F32) * do[:, :P]), o[:, :P]

    args = (x["q"], x["k"], x["v"], x["g"], *gates.values())
    (_, o), grads = jax.jit(jax.value_and_grad(loss, argnums=tuple(range(len(args))), has_aux=True))(*args)
    return [np.asarray(o.astype(F32))] + [np.asarray(gr.astype(F32))[:, :P] for gr in grads]


def _edited(x, P: int, value: float):
    return dict(x, g=x["g"].at[:, P:].set(value))


@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("impl", ["reference", "chunked"])
@pytest.mark.parametrize("value", [-4.95, -60.0, 0.0])
def test_reference_and_chunked_are_causal_bit_for_bit(kind, impl, value):
    x, gates, do = _inputs(kind, T=150)
    P = 75  # inside chunk 1 (tokens 64-127), before the 16th token of its second 32-token block
    want = _prefix_results(kind, impl, x, gates, do, P)
    got = _prefix_results(kind, impl, _edited(x, P, value), gates, do, P)
    for a, b in zip(want, got):
        np.testing.assert_array_equal(a, b)


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("P, value, bitwise", [
    (150, -4.95, True),  # the chunk stays bounded and P is past the 16th token of its 32-token block
    (150, 0.0, True),
    (150, -60.0, False),  # the edit makes the chunk unbounded: the exact path, rounded differently
    (135, -4.95, False),  # P before the 16th token: the factored block's reference token moves
])
def test_mosaic_gpu_future_gates_change_prefix_by_rounding_only(kind, P, value, bitwise):
    x, gates, do = _inputs(kind)
    want = _prefix_results(kind, "mosaic_gpu", x, gates, do, P)
    got = _prefix_results(kind, "mosaic_gpu", _edited(x, P, value), gates, do, P)
    oracle = _prefix_results(kind, "reference", {n: a.astype(F32) for n, a in x.items()}, gates, do, P)
    for a, b, r in zip(want, got, oracle):
        if bitwise:
            np.testing.assert_array_equal(a, b)
        # a leak of future tokens would move the prefix by O(1); rounding stays at bf16 noise
        assert np.abs(a - b).max() <= 1e-2 * np.abs(r).max()


@gpu
@pytest.mark.parametrize("kind", ["kda", "gdn2"])
@pytest.mark.parametrize("prefill", [64, 130])
def test_mosaic_gpu_prefill_then_decode_steps_match_the_full_pass(kind, prefill):
    """A kernel prefill's final state, then one reference step per token (as the layers decode), against
    the kernels on the whole sequence and against the f32 reference."""
    x, gates, _ = _inputs(kind, T=160)
    T = x["q"].shape[1]

    def call(impl, sl, state):
        part = {n: a[:, sl] for n, a in gates.items()}
        return ops.linear_attention(kind, x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], impl=impl, g=x["g"][:, sl],
                                    **part, initial_state=state, output_final_state=True)

    full, full_state = call("mosaic_gpu", slice(0, T), None)
    oracle, oracle_state = ops.linear_attention(kind, *(x[n].astype(F32) for n in ("q", "k", "v")), impl="reference",
                                                g=x["g"], **gates, output_final_state=True)
    outs, state = [], None
    o, state = call("mosaic_gpu", slice(0, prefill), state)
    outs.append(o)
    for t in range(prefill, T):
        o, state = call("reference", slice(t, t + 1), state)
        outs.append(o)
    stepped = np.asarray(jnp.concatenate(outs, axis=1).astype(F32))
    scale = np.abs(np.asarray(oracle)).max()
    if prefill % 64 == 0:  # whole chunks: the prefill's chunks are the full pass's, bit for bit
        np.testing.assert_array_equal(stepped[:, :prefill], np.asarray(full[:, :prefill].astype(F32)))
    assert np.abs(stepped - np.asarray(full.astype(F32))).max() <= 1e-2 * scale
    assert np.abs(stepped - np.asarray(oracle)).max() <= 1e-2 * scale
    state_scale = np.abs(np.asarray(oracle_state)).max()
    assert np.abs(np.asarray(state) - np.asarray(full_state)).max() <= 1e-2 * state_scale
    assert np.abs(np.asarray(state) - np.asarray(oracle_state)).max() <= 1e-2 * state_scale
