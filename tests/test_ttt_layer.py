# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear layer and ``tiny-ttt`` model: causality, packing, decode, implementations, a CPU train step."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models import LOGICAL_AXIS_NAMES, build_model, get_preset, logical_axes
from jla.ops.cross_entropy import linear_cross_entropy

TOL = dict(rtol=1e-4, atol=2e-4)


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 comparisons run at ~1e-4: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


def _cfg(**la_changes):
    cfg = get_preset("tiny-ttt").replace(compute_dtype="float32")
    if la_changes:
        cfg = cfg.replace(linear_attention=dataclasses.replace(cfg.linear_attention, **la_changes))
    return cfg


def _ids(batch=2, length=16, seed=1):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, 512)


def _loss(model, ids):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    return ce.loss_sum / ce.num_valid


def test_future_tokens_never_change_the_past():
    model = build_model(_cfg(), key=jax.random.key(0))
    ids = _ids(length=20)
    ids2 = ids.at[:, 11:].set(_ids(2, 9, seed=7))
    np.testing.assert_allclose(model(ids2).hidden[:, :11], model(ids).hidden[:, :11], **TOL)


def test_dual_and_reference_layers_agree():
    ids = _ids(length=21)
    dual = build_model(_cfg(implementation="chunked"), key=jax.random.key(1))
    ref = build_model(_cfg(implementation="reference"), key=jax.random.key(1))
    np.testing.assert_allclose(dual(ids).hidden, ref(ids).hidden, **TOL)


def test_aligned_packed_documents_match_separate_runs():
    model = build_model(_cfg(), key=jax.random.key(0))
    lengths = [8, 16, 8]  # multiples of the mini-batch (8): exact by construction
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = _ids(1, 32, seed=2)
    packed = model(ids, positions=pos, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        np.testing.assert_allclose(packed[:, start:start + n], model(ids[:, start:start + n]).hidden, **TOL)
        start += n


@pytest.mark.parametrize("scan", [True, False])
def test_prefill_then_decode_matches_the_forward_pass(scan):
    cfg = _cfg().replace(scan_layers=scan)
    model = build_model(cfg, key=jax.random.key(5))
    ids = _ids(2, 19, seed=6)
    full = model.logits(model(ids).hidden)
    pos = jnp.broadcast_to(jnp.arange(19, dtype=jnp.int32), (2, 19))
    step = jax.jit(lambda m, c, i, p: m.decode_step(c, i, p))
    cache = model.init_cache(2, 24)
    logits, cache = step(model, cache, ids[:, :6], pos[:, :6])
    outs = [logits]
    for t in range(6, 19):
        logits, cache = step(model, cache, ids[:, t:t + 1], pos[:, t:t + 1])
        outs.append(logits)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=2e-4)


def test_cpu_train_step_reduces_the_loss():
    model = build_model(_cfg(), key=jax.random.key(3))
    ids = _ids(4, 32, seed=4)
    loss, grads = eqx.filter_value_and_grad(_loss)(model, ids)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    g_ttt = grads.blocks[0].mixer
    for name in ("w0", "b0", "ln_weight", "token_offset"):
        assert float(jnp.abs(getattr(g_ttt, name)).max()) > 0, name  # the inner-model parameters learn
    assert float(jnp.abs(g_ttt.lr_proj.weight).max()) > 0
    new = eqx.apply_updates(model, jax.tree.map(lambda g: -0.5 * g, grads))
    assert float(_loss(new, ids)) < float(loss)


@pytest.mark.parametrize("scan", [True, False])
def test_logical_axes_cover_the_ttt_parameters(scan):
    model = build_model(get_preset("tiny-ttt").replace(scan_layers=scan), key=jax.random.key(0))
    pairs = {}
    params = eqx.filter(model, eqx.is_array)
    jax.tree_util.tree_map_with_path(
        lambda path, x, ax: pairs.__setitem__(jax.tree_util.keystr(path), (x.shape, ax)), params,
        logical_axes(model),
    )
    assert len(pairs) == len(jax.tree.leaves(params))
    for path, (shape, ax) in pairs.items():
        assert isinstance(ax, tuple) and len(ax) == len(shape), path
        assert set(ax) <= set(LOGICAL_AXIS_NAMES) | {None}, path
    assert any(path.endswith(".w0") for path in pairs)
