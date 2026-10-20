# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``ModelConfig.fp8_min_features``: which projections ``apply_fp8`` converts, and that a model with
small GEMMs kept in bf16 still trains like bf16 and keeps the delayed-scaling semantics."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.config import config_from_mapping
from jla.layers import fp8 as F
from jla.layers.linear import Linear, StackedLinear
from jla.layers.mlp import SwiGLU
from jla.models import build_model, get_preset
from jla.training.fp8 import without_fp8_state
from jla.training.step import LossConfig, build_train_step, lm_loss


def _gemm_kn(weight, base_ndim: int) -> tuple[int, int]:
    lead = weight.ndim - base_ndim
    return weight.shape[lead], math.prod(weight.shape[lead + 1:])


def _projections(model):
    """(kind, K, N) of every FP8 module and of every bf16 Linear / StackedLinear / SwiGLU gate-up."""
    fp8, plain = [], []
    kinds = (Linear, StackedLinear, SwiGLU, F.Fp8Linear, F.Fp8StackedLinear, F.Fp8SwiGLU)

    def visit(node):
        if isinstance(node, (F.Fp8Linear,)):
            fp8.append(("linear", *_gemm_kn(node.weight, 2)))
        elif isinstance(node, F.Fp8StackedLinear):
            fp8.append(("stacked", *_gemm_kn(node.weight, 3)))
        elif isinstance(node, F.Fp8SwiGLU):
            fp8.append(("swiglu", *_gemm_kn(node.gate_up, 3)))
            jax.tree_util.tree_map(visit, node.down, is_leaf=lambda n: isinstance(n, kinds))
        elif isinstance(node, Linear):
            plain.append(("linear", *_gemm_kn(node.weight, 2)))
        elif isinstance(node, StackedLinear):
            plain.append(("stacked", *_gemm_kn(node.weight, 3)))
        elif isinstance(node, SwiGLU):
            plain.append(("swiglu", *_gemm_kn(node.gate_up, 3)))
            jax.tree_util.tree_map(visit, node.down, is_leaf=lambda n: isinstance(n, kinds))
        return node

    jax.tree_util.tree_map(visit, (model.blocks, model.mtp), is_leaf=lambda n: isinstance(n, kinds))
    return fp8, plain


@pytest.mark.parametrize("preset", ["tiny", "tiny-hybrid", "tiny-mamba2", "tiny-gdn2"])
def test_min_features_keeps_small_gemms_in_bf16(preset: str) -> None:
    base = get_preset(preset)
    key = jax.random.key(0)
    default = build_model(base.replace(fp8="delayed", fp8_amax_history=4), key=key)
    explicit = build_model(base.replace(fp8="delayed", fp8_amax_history=4, fp8_min_features=16), key=key)
    assert len(F.fp8_modules(default)) == len(F.fp8_modules(explicit))  # 16 is the default: unchanged
    fp8_all, plain_all = _projections(default)
    assert all(k % 16 == 0 and n % 16 == 0 for _, k, n in fp8_all)
    threshold = 48
    model = build_model(base.replace(fp8="delayed", fp8_amax_history=4, fp8_min_features=threshold), key=key)
    fp8, plain = _projections(model)
    assert fp8, preset
    assert all(min(k, n) >= threshold for _, k, n in fp8), fp8
    # Every aligned projection below the threshold stayed a bf16 module; nothing else changed kind.
    assert all(min(k, n) < threshold or k % 16 or n % 16 for _, k, n in plain), plain
    assert sorted(fp8 + plain) == sorted(fp8_all + plain_all)
    assert len(fp8) < len(fp8_all) or preset == "tiny-mamba2"
    # Parameters are untouched by the conversion.
    for a, b in zip(jax.tree.leaves(without_fp8_state(eqx.filter(default, eqx.is_array))),
                    jax.tree.leaves(without_fp8_state(eqx.filter(model, eqx.is_array))), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_min_features_config_and_validation() -> None:
    assert get_preset("tiny").fp8_min_features == 16
    cfg = config_from_mapping({"model": {"preset": "tiny", "fp8": "current"}}, ["model.fp8_min_features=512"])
    assert cfg.model.fp8_min_features == 512
    with pytest.raises(ValueError, match="fp8_min_features"):
        get_preset("tiny").replace(fp8="delayed", fp8_min_features=0)
    with pytest.raises(ValueError, match="min_features"):
        F.apply_fp8(build_model(get_preset("tiny"), key=jax.random.key(0)), "delayed", min_features=0)


def _batches(steps: int, batch: int = 8, seq: int = 32, vocab: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        ids = [rng.integers(0, vocab, size=(1, batch, 1))]
        for _ in range(seq):
            ids.append((5 * ids[-1] + 3) % vocab)
        ids = np.concatenate(ids, axis=-1).astype(np.int32)
        out.append({"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])})
    return out


def _train(cfg, batches):
    model = build_model(cfg, key=jax.random.key(0))
    optimizer = optax.adamw(3e-3, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype=cfg.compute_dtype)), donate=False)
    key, losses = jax.random.key(1), []
    for batch in batches:
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
    return model, np.array(losses)


@pytest.mark.parametrize("recipe", ["delayed", "current"])
def test_min_features_trains_like_bf16_and_like_full_fp8(recipe: str) -> None:
    base = get_preset("tiny")  # k/v projections are 64 -> 32: below a threshold of 48
    batches = _batches(30)
    _, bf16 = _train(base, batches)
    _, full = _train(base.replace(fp8=recipe), batches)
    trained, partial = _train(base.replace(fp8=recipe, fp8_min_features=48), batches)
    assert bf16[-1] < bf16[0] - 2.0
    for losses in (full, partial):
        assert np.all(np.abs(losses - bf16) <= 0.05 + 0.01 * bf16), (np.round(losses, 3), np.round(bf16, 3))
    if recipe == "delayed":  # the remaining FP8 matmuls recorded one amax per applied step
        for module in F.fp8_modules(trained):
            hist = np.asarray(module.state.input_amax_history)
            assert np.all(hist[..., :30] > 0) and np.all(hist[..., 30:] == 0)
