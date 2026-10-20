# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Closed-form checks for the token-by-token reference recurrences."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops import reference as ref

TOL = dict(rtol=1e-4, atol=1e-4)


def _inputs(seed: int = 0, B: int = 2, T: int = 37, H: int = 3, K: int = 16, V: int = 8):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = jax.random.normal(ks[0], (B, T, H, K))
    k = jax.random.normal(ks[1], (B, T, H, K)) / np.sqrt(K)
    v = jax.random.normal(ks[2], (B, T, H, V))
    g_head = -jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)))
    g_key = -jax.nn.softplus(jax.random.normal(ks[4], (B, T, H, K)))
    beta = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H)))
    return q, k, v, g_head, g_key, beta


def _call(name: str, q, k, v, g_head, g_key, beta, **kw):
    if name == "simple_gla":
        return ref.simple_gla(q, k, v, g_head, **kw)
    if name == "gla":
        return ref.gla(q, k, v, g_key, **kw)
    if name == "delta_rule":
        return ref.delta_rule(q, k, v, beta, **kw)
    if name == "gated_delta_rule":
        return ref.gated_delta_rule(q, k, v, g_head, beta, **kw)
    if name == "kda":
        return ref.kda(q, k, v, g_key, beta, **kw)
    raise AssertionError(name)


VARIANTS = ["simple_gla", "gla", "delta_rule", "gated_delta_rule", "kda"]


def test_gla_matches_parallel_form():
    q, k, v, _, g_key, _ = _inputs()
    o, _ = ref.gla(q, k, v, g_key)
    T = q.shape[1]
    scale = q.shape[-1] ** -0.5
    cum = jnp.cumsum(g_key, axis=1)
    causal = jnp.tril(jnp.ones((T, T), bool))
    diff = jnp.where(causal[None, :, :, None, None], cum[:, :, None] - cum[:, None, :], -jnp.inf)
    weights = jnp.einsum("bthk,bshk,btshk->bhts", q, k, jnp.exp(diff), precision="highest")
    expected = jnp.einsum("bhts,bshv->bthv", weights * scale, v, precision="highest")
    np.testing.assert_allclose(o, expected, **TOL)


def test_gla_without_decay_is_causal_linear_attention():
    q, k, v, *_ = _inputs(1)
    o, _ = ref.gla(q, k, v, jnp.zeros(q.shape))
    T = q.shape[1]
    scores = jnp.einsum("bthk,bshk->bhts", q, k, precision="highest") * q.shape[-1] ** -0.5
    scores = jnp.where(jnp.tril(jnp.ones((T, T), bool)), scores, 0.0)
    expected = jnp.einsum("bhts,bshv->bthv", scores, v, precision="highest")
    np.testing.assert_allclose(o, expected, **TOL)


def test_scalar_gates_are_broadcast_vector_gates():
    q, k, v, g_head, _, beta = _inputs(2)
    g_wide = jnp.broadcast_to(g_head[..., None], q.shape)
    np.testing.assert_allclose(ref.simple_gla(q, k, v, g_head)[0], ref.gla(q, k, v, g_wide)[0], **TOL)
    np.testing.assert_allclose(
        ref.gated_delta_rule(q, k, v, g_head, beta)[0], ref.kda(q, k, v, g_wide, beta)[0], **TOL
    )


def test_gated_delta_rule_without_decay_is_delta_rule():
    q, k, v, _, _, beta = _inputs(3)
    zero = jnp.zeros(q.shape[:3])
    np.testing.assert_allclose(
        ref.gated_delta_rule(q, k, v, zero, beta)[0], ref.delta_rule(q, k, v, beta)[0], **TOL
    )


def test_delta_rule_overwrites_the_value_of_a_repeated_key():
    K, V = 8, 4
    keys = jnp.concatenate([jnp.eye(K), jnp.eye(K)[:1]])  # e0..e7, then e0 again
    T = keys.shape[0]
    values = jax.random.normal(jax.random.key(4), (T, V))
    q = keys * np.sqrt(K)  # cancels the default K**-0.5 scale
    o, state = ref.delta_rule(
        q[None, :, None], keys[None, :, None], values[None, :, None], jnp.ones((1, T, 1)),
        output_final_state=True,
    )
    np.testing.assert_allclose(o[0, :, 0], values, **TOL)  # each query reads its own write
    np.testing.assert_allclose(state[0, 0, 0], values[-1], **TOL)  # e0 now holds the last value
    np.testing.assert_allclose(state[0, 0, 1:], values[1:K], **TOL)


@pytest.mark.parametrize("name", VARIANTS)
def test_final_state_chains_across_segments(name):
    q, k, v, g_head, g_key, beta = _inputs(5, T=20)
    whole, whole_state = _call(name, q, k, v, g_head, g_key, beta, output_final_state=True)
    split = 12
    head = [x[:, :split] for x in (q, k, v, g_head, g_key, beta)]
    tail = [x[:, split:] for x in (q, k, v, g_head, g_key, beta)]
    first, mid_state = _call(name, *head, output_final_state=True)
    second, end_state = _call(name, *tail, initial_state=mid_state, output_final_state=True)
    np.testing.assert_allclose(jnp.concatenate([first, second], axis=1), whole, **TOL)
    np.testing.assert_allclose(end_state, whole_state, **TOL)


@pytest.mark.parametrize("name", VARIANTS)
def test_packed_sequences_match_separate_runs(name):
    lengths = [5, 0, 9, 3]
    T = sum(lengths)
    q, k, v, g_head, g_key, beta = _inputs(6, B=1, T=T)
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    state0 = jax.random.normal(jax.random.key(7), (len(lengths), H, K, V))
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    packed, packed_state = _call(
        name, q, k, v, g_head, g_key, beta,
        initial_state=state0, output_final_state=True, cu_seqlens=cu,
    )
    start = 0
    for i, n in enumerate(lengths):
        if n == 0:
            np.testing.assert_allclose(packed_state[i], state0[i], **TOL)
            continue
        piece = [x[:, start : start + n] for x in (q, k, v, g_head, g_key, beta)]
        o, s = _call(name, *piece, initial_state=state0[i : i + 1], output_final_state=True)
        np.testing.assert_allclose(packed[:, start : start + n], o, **TOL)
        np.testing.assert_allclose(packed_state[i], s[0], **TOL)
        start += n


def test_qk_l2norm_option_matches_normalized_inputs():
    q, k, v, g_head, _, beta = _inputs(8)
    fused, _ = ref.gated_delta_rule(q, k, v, g_head, beta, use_qk_l2norm=True)
    manual, _ = ref.gated_delta_rule(ref.l2norm(q), ref.l2norm(k), v, g_head, beta)
    np.testing.assert_allclose(fused, manual, **TOL)


def test_dtypes():
    q, k, v, g_head, _, beta = _inputs(9)
    o, state = ref.gated_delta_rule(
        q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), g_head, beta,
        output_final_state=True,
    )
    assert o.dtype == jnp.bfloat16
    assert state.dtype == jnp.float32
    assert ref.gated_delta_rule(q, k, v, g_head, beta)[1] is None


def test_shape_errors():
    q, k, v, g_head, g_key, beta = _inputs(10)
    with pytest.raises(ValueError, match="g must be"):
        ref.gated_delta_rule(q, k, v, g_key, beta)
    with pytest.raises(ValueError, match="batch size 1"):
        ref.gla(q, k, v, g_key, cu_seqlens=jnp.array([0, 37], jnp.int32))


def _segment_layout():
    # Row 0: three documents; row 1 reuses id 4 after id 9, which is a new document.
    row0 = [0] * 7 + [3] * 9 + [5] * 8
    row1 = [4] * 6 + [9] * 6 + [4] * 12
    return np.array([row0, row1], np.int32)


def _runs(row):
    starts = [0] + [t for t in range(1, len(row)) if row[t] != row[t - 1]]
    return list(zip(starts, starts[1:] + [len(row)]))


@pytest.mark.parametrize("name", VARIANTS)
def test_segment_ids_match_separate_runs(name):
    seg = _segment_layout()
    B, T = seg.shape
    q, k, v, g_head, g_key, beta = _inputs(11, B=B, T=T)
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    state0 = jax.random.normal(jax.random.key(12), (B, H, K, V))
    packed, packed_state = _call(
        name, q, k, v, g_head, g_key, beta,
        initial_state=state0, output_final_state=True, segment_ids=jnp.asarray(seg),
    )
    for b in range(B):
        for i, (start, end) in enumerate(_runs(seg[b])):
            piece = [x[b : b + 1, start:end] for x in (q, k, v, g_head, g_key, beta)]
            init = state0[b : b + 1] if i == 0 else jnp.zeros((1, H, K, V))
            o, s = _call(name, *piece, initial_state=init, output_final_state=True)
            np.testing.assert_allclose(packed[b : b + 1, start:end], o, **TOL)
        np.testing.assert_allclose(packed_state[b], s[0], **TOL)  # last run's final state


def test_segment_ids_and_cu_seqlens_are_exclusive():
    q, k, v, g_head, _, beta = _inputs(13, B=1, T=8)
    with pytest.raises(ValueError, match="not both"):
        ref.gated_delta_rule(
            q, k, v, g_head, beta,
            cu_seqlens=jnp.array([0, 8], jnp.int32), segment_ids=jnp.zeros((1, 8), jnp.int32),
        )
