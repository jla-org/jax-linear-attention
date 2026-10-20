# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2 tests: Implemented from arXiv 2605.22791 (equations only)
"""Linear-attention mixers: FLA semantics, causality, documents, chunked = reference, decode parity."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.linear_attention import KDA, GatedDeltaNet, GatedDeltaNet2, make_mixer
from jla.layers.norm import l2_normalize, rms_norm
from jla.models.config import LinearAttentionConfig
from jla.ops import reference

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


KINDS = ["gated_deltanet", "kda", "gdn2", "gla", "deltanet", "simple_gla"]
D = 32
TOL = dict(rtol=2e-4, atol=2e-4)


def _cfg(**kw) -> LinearAttentionConfig:
    base = dict(num_k_heads=2, num_v_heads=4, head_k_dim=8, head_v_dim=8, chunk_size=16)
    base.update(kw)
    return LinearAttentionConfig(**base)


def _randomize(module, seed):
    """Replace every float array with random values so identity inits do not hide bugs."""
    leaves, treedef = jax.tree.flatten(module)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    new = [
        0.3 * jax.random.normal(k, x.shape, x.dtype) if eqx.is_inexact_array(x) else x
        for k, x in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new)


def _mixer(kind, seed=0, compute_dtype="float32", randomize=True, **kw):
    mixer = make_mixer(kind, D, _cfg(**kw), key=jax.random.key(seed), std=0.2, norm_eps=1e-6,
                       compute_dtype=compute_dtype)
    if randomize:
        # Keep the decay parameters in their meaningful ranges.
        keep = {n: getattr(mixer, n) for n in ("A_log", "dt_bias") if hasattr(mixer, n)}
        mixer = _randomize(mixer, seed + 100)
        for name, value in keep.items():
            mixer = eqx.tree_at(lambda m, name=name: getattr(m, name), mixer, value)
    return mixer


def _x(batch=2, length=20, seed=1):
    return jax.random.normal(jax.random.key(seed), (batch, length, D))


@pytest.mark.parametrize("kind", KINDS)
def test_shapes_and_dtypes(kind):
    mixer = _mixer(kind, compute_dtype="bfloat16", randomize=False)
    out = mixer(_x().astype(jnp.bfloat16))
    assert out.shape == (2, 20, D) and out.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(out.astype(jnp.float32))).all()


@pytest.mark.parametrize("kind", KINDS)
def test_future_tokens_never_change_the_past(kind):
    mixer = _mixer(kind, implementation="chunked")
    x = _x()
    x2 = x.at[:, 13:].set(_x(2, 7, seed=5))
    np.testing.assert_allclose(mixer(x2)[:, :13], mixer(x)[:, :13], **TOL)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("impl", ["reference", "chunked"])
def test_packed_documents_match_separate_runs(kind, impl):
    mixer = _mixer(kind, implementation=impl)
    lengths = [6, 3, 11]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    x = _x(1, 20, seed=2)
    packed = mixer(x, seg)
    start = 0
    for n in lengths:
        np.testing.assert_allclose(packed[:, start : start + n], mixer(x[:, start : start + n]), **TOL)
        start += n


@pytest.mark.parametrize("kind", KINDS)
def test_chunked_matches_reference(kind):
    x = _x()
    ref = _mixer(kind, implementation="reference")
    chunked = _mixer(kind, implementation="chunked")  # same seeds, same parameters
    np.testing.assert_allclose(chunked(x), ref(x), **TOL)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("conv_size", [4, 0])
def test_prefill_then_decode_matches_the_full_pass(kind, conv_size):
    mixer = _mixer(kind, conv_size=conv_size, implementation="chunked")
    x = _x(2, 14, seed=3)
    full = mixer(x)
    cache = mixer.init_cache(2, jnp.float32)
    out, cache = mixer.step(x[:, :5], cache)
    outs = [out]
    for t in range(5, 14):
        out, cache = mixer.step(x[:, t : t + 1], cache)
        outs.append(out)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, **TOL)


@pytest.mark.parametrize("kind", ["gla", "deltanet", "simple_gla"])
def test_variants_without_output_gate(kind):
    mixer = _mixer(kind, output_gate=False)
    assert mixer.g_proj is None
    assert mixer(_x()).shape == (2, 20, D)


def test_gated_deltanet_follows_fla_layer_definition():
    mixer = _mixer("gated_deltanet", conv_size=0, implementation="reference")
    assert isinstance(mixer, GatedDeltaNet)
    x = _x(1, 9, seed=4)
    cfg = mixer.cfg
    silu = jax.nn.silu
    q = silu(x @ mixer.q_proj.weight).reshape(1, 9, 2, 8)
    k = silu(x @ mixer.k_proj.weight).reshape(1, 9, 2, 8)
    v = silu(x @ mixer.v_proj.weight).reshape(1, 9, 4, 8)
    q, k = jnp.repeat(l2_normalize(q), 2, axis=2), jnp.repeat(l2_normalize(k), 2, axis=2)
    g = -jnp.exp(mixer.A_log) * jax.nn.softplus(x @ mixer.a_proj.weight + mixer.dt_bias)
    beta = jax.nn.sigmoid(x @ mixer.b_proj.weight)
    o, _ = reference.gated_delta_rule(q, k, v, g, beta)
    gate = (x @ mixer.g_proj.weight).reshape(1, 9, 4, 8)
    o = rms_norm(o, mixer.o_norm.weight, mixer.o_norm.eps) * silu(gate)
    want = o.reshape(1, 9, 32) @ mixer.o_proj.weight
    np.testing.assert_allclose(mixer(x), want, **TOL)
    assert cfg.num_v_heads == 4


def test_kda_follows_fla_layer_definition():
    mixer = _mixer("kda", conv_size=0, implementation="reference")
    assert isinstance(mixer, KDA)
    x = _x(1, 9, seed=6)
    silu = jax.nn.silu
    q = jnp.repeat(l2_normalize(silu(x @ mixer.q_proj.weight).reshape(1, 9, 2, 8)), 2, axis=2)
    k = jnp.repeat(l2_normalize(silu(x @ mixer.k_proj.weight).reshape(1, 9, 2, 8)), 2, axis=2)
    v = silu(x @ mixer.v_proj.weight).reshape(1, 9, 4, 8)
    f = (x @ mixer.f_proj.down.weight @ mixer.f_proj.up.weight).reshape(1, 9, 4, 8)
    g = -jnp.exp(mixer.A_log)[:, None] * jax.nn.softplus(f + mixer.dt_bias.reshape(4, 8))
    beta = jax.nn.sigmoid(x @ mixer.b_proj.weight)
    o, _ = reference.kda(q, k, v, g, beta)
    gate = (x @ mixer.g_proj.down.weight @ mixer.g_proj.up.weight + mixer.g_proj.up.bias).reshape(1, 9, 4, 8)
    o = rms_norm(o, mixer.o_norm.weight, mixer.o_norm.eps) * jax.nn.sigmoid(gate)
    np.testing.assert_allclose(mixer(x), o.reshape(1, 9, 32) @ mixer.o_proj.weight, **TOL)


@pytest.mark.parametrize("neg_eigval", [False, True])
def test_gdn2_follows_the_paper(neg_eigval):
    """Eq. 10-12 and Fig. 1 of arXiv 2605.22791: gates on key heads repeat over value-head groups."""
    mixer = _mixer("gdn2", conv_size=0, implementation="reference", allow_neg_eigval=neg_eigval)
    assert isinstance(mixer, GatedDeltaNet2)
    x = _x(1, 9, seed=7)
    silu = jax.nn.silu
    q = jnp.repeat(l2_normalize(silu(x @ mixer.q_proj.weight).reshape(1, 9, 2, 8)), 2, axis=2)
    k = jnp.repeat(l2_normalize(silu(x @ mixer.k_proj.weight).reshape(1, 9, 2, 8)), 2, axis=2)
    v = silu(x @ mixer.v_proj.weight).reshape(1, 9, 4, 8)
    f = (x @ mixer.f_proj.down.weight @ mixer.f_proj.up.weight).reshape(1, 9, 2, 8)
    g = jnp.repeat(-jnp.exp(mixer.A_log)[:, None] * jax.nn.softplus(f + mixer.dt_bias.reshape(2, 8)), 2, axis=2)
    b = jnp.repeat(jax.nn.sigmoid(x @ mixer.b_proj.weight).reshape(1, 9, 2, 8), 2, axis=2)
    b = 2.0 * b if neg_eigval else b
    w = jax.nn.sigmoid(x @ mixer.w_proj.weight).reshape(1, 9, 4, 8)
    o, _ = reference.gdn2(q, k, v, g, b, w)
    gate = (x @ mixer.g_proj.weight).reshape(1, 9, 4, 8)
    o = rms_norm(o, mixer.o_norm.weight, mixer.o_norm.eps) * silu(gate)
    np.testing.assert_allclose(mixer(x), o.reshape(1, 9, 32) @ mixer.o_proj.weight, **TOL)
    assert mixer.A_log.shape == (2,) and mixer.dt_bias.shape == (2 * 8,)  # per key head / key channel
    with pytest.raises(ValueError, match="output_gate"):
        make_mixer("gdn2", D, _cfg(output_gate=False), key=jax.random.key(0), std=0.02, norm_eps=1e-6)


def test_fla_initialization():
    gdn = make_mixer("gated_deltanet", D, _cfg(num_v_heads=64, num_k_heads=64), key=jax.random.key(0),
                     std=0.02, norm_eps=1e-6)
    a = np.exp(np.asarray(gdn.A_log))
    assert (a > 0).all() and (a <= 16).all()
    dt = np.asarray(jax.nn.softplus(gdn.dt_bias))
    np.testing.assert_array_less(0.999e-3, dt)
    np.testing.assert_array_less(dt, 0.1 + 1e-6)
    kda = make_mixer("kda", D, _cfg(), key=jax.random.key(1), std=0.02, norm_eps=1e-6)
    assert np.exp(np.asarray(kda.A_log)).min() >= 1.0
    assert kda.dt_bias.shape == (4 * 8,)
    np.testing.assert_array_equal(gdn.q_conv.weight[:, -1], 1.0)  # identity-initialized conv (the default)
    np.testing.assert_array_equal(gdn.q_conv.weight[:, :-1], 0.0)
    fla = make_mixer("gated_deltanet", D, _cfg(num_v_heads=64, num_k_heads=64, conv_init="normal"),
                     key=jax.random.key(0), std=0.02, norm_eps=1e-6)
    w = np.asarray(fla.q_conv.weight)  # FLA (opt-in): every nn.Conv1d ~ N(0, initializer_range), no bias
    assert fla.q_conv.bias is None and abs(w.std() / 0.02 - 1) < 0.25 and np.any(w[:, :-1] != 0)
    np.testing.assert_array_equal(gdn.o_norm.weight, 1.0)
    with pytest.raises(ValueError, match="unknown linear-attention kind"):
        make_mixer("mamba", D, _cfg(), key=jax.random.key(0), std=0.02, norm_eps=1e-6)
