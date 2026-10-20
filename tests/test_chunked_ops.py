# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chunked JAX forms must match the token-by-token references, values and gradients."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops import chunked, reference as ref

TOL = dict(rtol=2e-4, atol=2e-4)
NAMES = ["simple_gla", "gla", "delta_rule", "gated_delta_rule", "kda"]


def _inputs(seed: int, B: int = 2, T: int = 100, H: int = 2, K: int = 32, V: int = 16):
    ks = jax.random.split(jax.random.key(seed), 8)
    q = jax.random.normal(ks[0], (B, T, H, K))
    k = ref.l2norm(jax.random.normal(ks[1], (B, T, H, K)))
    v = jax.random.normal(ks[2], (B, T, H, V))
    g_head = -jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)) - 2.0)
    g_key = -jax.nn.softplus(jax.random.normal(ks[4], (B, T, H, K)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H)))
    state0 = 0.1 * jax.random.normal(ks[6], (B, H, K, V))
    cotangent = jax.random.normal(ks[7], (B, T, H, V))
    return dict(q=q, k=k, v=v, g_head=g_head, g_key=g_key, beta=beta, state0=state0), cotangent


def _run(impl, name, x, **kw):
    q, k, v, beta = x["q"], x["k"], x["v"], x["beta"]
    common = dict(initial_state=x["state0"], output_final_state=True, **kw)
    if name == "simple_gla":
        return impl.simple_gla(q, k, v, x["g_head"], **common)
    if name == "gla":
        return impl.gla(q, k, v, x["g_key"], **common)
    if name == "delta_rule":
        return impl.delta_rule(q, k, v, beta, **common)
    if name == "gated_delta_rule":
        return impl.gated_delta_rule(q, k, v, x["g_head"], beta, **common)
    return impl.kda(q, k, v, x["g_key"], beta, **common)


def _used(name: str) -> list[str]:
    gate = {"simple_gla": ["g_head"], "gla": ["g_key"], "delta_rule": [],
            "gated_delta_rule": ["g_head"], "kda": ["g_key"]}[name]
    write = ["beta"] if name in ("delta_rule", "gated_delta_rule", "kda") else []
    return ["q", "k", "v", *gate, *write, "state0"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_forward_matches_reference(name, chunk_size):
    x, _ = _inputs(0)
    o_ref, s_ref = _run(ref, name, x)
    o, s = _run(chunked, name, x, chunk_size=chunk_size)
    np.testing.assert_allclose(o, o_ref, **TOL)
    np.testing.assert_allclose(s, s_ref, **TOL)


@pytest.mark.parametrize("name", NAMES)
def test_gradients_match_reference(name):
    x, cot = _inputs(1, T=48)
    used = _used(name)

    def loss(impl, values, **kw):
        o, s = _run(impl, name, {**x, **dict(zip(used, values))}, **kw)
        return jnp.sum(o * cot) + jnp.sum(s * s)

    values = [x[n] for n in used]
    want = jax.grad(lambda v: loss(ref, v))(values)
    got = jax.grad(lambda v: loss(chunked, v, chunk_size=16))(values)
    for label, a, b in zip(used, got, want):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=label)


@pytest.mark.parametrize("name", ["gla", "kda"])
def test_vector_gates_survive_huge_decays(name):
    """Per-channel decays of -60 per token must not overflow any exponential."""
    x, _ = _inputs(2, T=64)
    spikes = jax.random.bernoulli(jax.random.key(3), 0.1, x["g_key"].shape)
    x["g_key"] = jnp.where(spikes, -60.0, x["g_key"])
    o_ref, s_ref = _run(ref, name, x)
    o, s = _run(chunked, name, x, chunk_size=64)
    assert np.isfinite(np.asarray(o)).all() and np.isfinite(np.asarray(s)).all()
    np.testing.assert_allclose(o, o_ref, **TOL)
    np.testing.assert_allclose(s, s_ref, **TOL)


def test_bf16_inputs_and_qk_l2norm():
    x, _ = _inputs(4, T=64)
    bf = [x[n].astype(jnp.bfloat16) for n in ("q", "k", "v")]
    o_ref, _ = ref.gated_delta_rule(*bf, x["g_head"], x["beta"], use_qk_l2norm=True)
    o, _ = chunked.gated_delta_rule(*bf, x["g_head"], x["beta"], use_qk_l2norm=True)
    assert o.dtype == jnp.bfloat16
    np.testing.assert_allclose(o.astype(jnp.float32), o_ref.astype(jnp.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("gate", ["g_head", "g_key"])
def test_stage_identities(gate):
    """Each stage obeys its own definition, independently of the full composition."""
    x, _ = _inputs(5, B=1, T=32, H=1)
    k_c = chunked._to_chunks(x["k"], 32)
    beta_c = chunked._to_chunks(x["beta"], 32)
    g_cum = chunked.chunk_local_cumsum(chunked._to_chunks(x[gate], 32))
    a = chunked.chunk_scaled_dot_kkt(k_c, beta_c, g_cum, precision="highest")
    a00 = np.asarray(a[0, 0, 0])
    assert np.allclose(np.triu(a00), 0.0)
    # Spot-check one entry against its definition.
    g00 = np.asarray(g_cum[0, 0, 0])
    kk = np.asarray(k_c[0, 0, 0])
    i, j = 20, 3
    decay = np.exp(g00[i] - g00[j])
    want = float(x["beta"][0, i, 0]) * np.sum(kk[i] * kk[j] * decay)
    np.testing.assert_allclose(a00[i, j], want, rtol=1e-5)
    t = chunked.solve_unit_lower(a)
    # An f32 triangular solve leaves a residual of about n * eps * |A| * |T|: ~1e-6 on
    # CPU, ~1e-4 with GPU solvers.
    np.testing.assert_allclose(np.asarray((np.eye(32) + a00) @ t[0, 0, 0]), np.eye(32), atol=1e-3)


def test_packed_sequences_are_rejected():
    x, _ = _inputs(6, B=1)
    with pytest.raises(NotImplementedError):
        chunked.gated_delta_rule(
            x["q"], x["k"], x["v"], x["g_head"], x["beta"], cu_seqlens=jnp.array([0, 100], jnp.int32)
        )


def _segments(B: int, T: int, seed: int) -> jnp.ndarray:
    """Random contiguous documents per row, with one id reused after a different one."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):
        cuts = np.sort(rng.choice(np.arange(1, T), size=5, replace=False))
        ids = np.zeros(T, np.int32)
        for n, cut in enumerate(cuts):
            ids[cut:] = (n + 1) % 3  # ids cycle 0, 1, 2, 0, ... so reuse happens
        rows.append(ids)
    rows[0][16:32] = 7  # a document aligned exactly with chunk boundaries of size 16
    return jnp.asarray(np.stack(rows))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_segment_ids_match_reference(name, chunk_size):
    x, _ = _inputs(7, T=100)
    seg = _segments(2, 100, seed=8)
    o_ref, s_ref = _run(ref, name, x, segment_ids=seg)
    o, s = _run(chunked, name, x, segment_ids=seg, chunk_size=chunk_size)
    np.testing.assert_allclose(o, o_ref, **TOL)
    np.testing.assert_allclose(s, s_ref, **TOL)


@pytest.mark.parametrize("name", ["gated_delta_rule", "kda"])
def test_segment_ids_gradients_match_reference(name):
    x, cot = _inputs(9, T=48)
    seg = _segments(2, 48, seed=10)
    used = _used(name)

    def loss(impl, values, **kw):
        o, s = _run(impl, name, {**x, **dict(zip(used, values))}, segment_ids=seg, **kw)
        return jnp.sum(o * cot) + jnp.sum(s * s)

    values = [x[n] for n in used]
    want = jax.grad(lambda v: loss(ref, v))(values)
    got = jax.grad(lambda v: loss(chunked, v, chunk_size=16))(values)
    for label, a, b in zip(used, got, want):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=label)
