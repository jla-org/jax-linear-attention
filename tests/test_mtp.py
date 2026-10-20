# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-token prediction: loss definition, label masking, gradients."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.norm import rms_norm
from jla.models import MTPConfig, build_model, get_preset
from jla.ops.cross_entropy import cross_entropy_from_logits

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


B, T = 2, 12


def _model(kind="deepseek", depth=2, tie=True, seed=0):
    cfg = get_preset("tiny").replace(mtp=MTPConfig(depth=depth, kind=kind), compute_dtype="float32",
                                     tie_embeddings=tie)
    return build_model(cfg, key=jax.random.key(seed))


def _batch(seed=1):
    ids = jax.random.randint(jax.random.key(seed), (B, T), 0, 512)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((B, 1), -100)], axis=1)
    return ids, targets


@pytest.mark.parametrize("kind", ["deepseek", "chained", "independent"])
def test_loss_is_finite_near_uniform_and_has_gradients(kind):
    model = _model(kind, tie=False)
    ids, targets = _batch()

    def loss_fn(m):
        loss, metrics = m.mtp_loss(m(ids), ids, targets)
        return loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    assert set(metrics) == {"mtp/loss", "mtp/loss_1", "mtp/loss_2"}
    np.testing.assert_allclose(metrics["mtp/loss"], np.log(512), rtol=0.05)
    np.testing.assert_allclose(loss, 0.3 * metrics["mtp/loss"], rtol=1e-6)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves)
    for part in (grads.mtp, grads.head):
        assert any(float(jnp.max(jnp.abs(g))) > 0 for g in jax.tree.leaves(part))
    if kind != "independent":  # conditioning tokens go through the shared embedding
        assert float(jnp.max(jnp.abs(grads.embed.weight[ids[:, 1:]]))) > 0


def test_chained_depth_one_matches_manual_computation():
    model = _model("chained", depth=1)
    ids, targets = _batch(2)
    out = model(ids)
    loss, _ = model.mtp_loss(out, ids, targets)
    mtp = model.mtp
    h = out.pre_norm_hidden[:, :-1]
    emb = model.embed.weight[ids[:, 1:]]
    fused = (rms_norm(h, mtp.hidden_norms[0].weight, 1e-6) @ mtp.hidden_proj[0].weight
             + rms_norm(emb, mtp.embed_norms[0].weight, 1e-6) @ mtp.embed_proj[0].weight)
    z = rms_norm(fused, mtp.out_norms[0].weight, 1e-6)
    ce = cross_entropy_from_logits((z @ model.embed.weight.T).reshape(-1, 512), targets[:, 1:].reshape(-1))
    np.testing.assert_allclose(loss, 0.3 * ce.loss_sum / ce.num_valid, rtol=1e-5)


def test_labels_crossing_documents_are_ignored():
    model = _model("chained", depth=2)
    ids, targets = _batch(3)
    seg = jnp.asarray(np.repeat([0, 1, 2], [5, 4, 3])[None].repeat(B, 0), jnp.int32)
    out = model(ids, segment_ids=seg)
    _, with_seg = model.mtp_loss(out, ids, targets, segment_ids=seg)
    # Same loss without segment_ids once the crossing labels are masked by hand (chained has no attention).
    masked = []
    for d in (1, 2):
        cross = jnp.concatenate([seg[:, d:] != seg[:, :-d], jnp.zeros((B, d), bool)], axis=1)
        masked.append(cross)
    for depth, cross in zip((1, 2), masked):
        t = jnp.where(jnp.roll(cross, depth, axis=1), -100, targets)
        _, manual = model.mtp_loss(out, ids, t)
        np.testing.assert_allclose(with_seg[f"mtp/loss_{depth}"], manual[f"mtp/loss_{depth}"], rtol=1e-6)


@pytest.mark.parametrize("kind", ["deepseek", "independent"])
def test_packed_documents_match_separate_runs(kind):
    """With segment_ids the refiner blocks and labels stay inside each document."""
    model = _model(kind, depth=2)
    lengths = [5, 4, 3]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = jax.random.randint(jax.random.key(5), (1, sum(lengths)), 0, 512)

    def next_tokens(x):
        return jnp.concatenate([x[:, 1:], jnp.full((1, 1), -100)], axis=1)

    # Packed targets with each document's last target ignored, as a packing pipeline produces.
    same_doc = jnp.concatenate([seg[:, 1:] == seg[:, :-1], jnp.zeros((1, 1), bool)], axis=1)
    packed_targets = jnp.where(same_doc, next_tokens(ids), -100)
    out = model(ids, positions=pos, segment_ids=seg)
    _, packed = model.mtp_loss(out, ids, packed_targets, positions=pos, segment_ids=seg)
    for depth in (1, 2):
        total, count = 0.0, 0
        start = 0
        for n in lengths:
            doc = ids[:, start : start + n]
            _, alone = model.mtp_loss(model(doc), doc, next_tokens(doc))
            labels = max(n - depth - 1, 0)  # positions whose label lies depth + 1 tokens ahead in the doc
            if labels:
                total += float(alone[f"mtp/loss_{depth}"]) * labels
                count += labels
            start += n
        np.testing.assert_allclose(packed[f"mtp/loss_{depth}"], total / count, rtol=1e-5)


def test_degenerate_inputs():
    model = _model("deepseek", depth=2)
    ids, targets = _batch(4)
    out = model(ids)
    loss, metrics = model.mtp_loss(out, ids, jnp.full_like(targets, -100))
    assert float(loss) == 0.0 and np.isfinite(float(metrics["mtp/loss_1"]))
    short = ids[:, :2]
    loss, metrics = model.mtp_loss(model(short), short, targets[:, :2])
    assert "mtp/loss_2" not in metrics  # depth 2 needs at least three tokens
    no_mtp = build_model(get_preset("tiny"), key=jax.random.key(0))
    loss, metrics = no_mtp.mtp_loss(no_mtp(ids), ids, targets)
    assert float(loss) == 0.0 and metrics == {}
    with pytest.raises(ValueError, match="depth"):
        MTPConfig(depth=0)
