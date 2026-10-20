# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-7 language model: presets, parameter count, FLOPs, decode, CPU train step."""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models.rwkv7 import RWKV7_PRESETS, RWKV7LM, rwkv7_flops_per_token, rwkv7_param_count


def _tiny(**changes):
    return dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], compute_dtype="float32", **changes)


@pytest.mark.parametrize("name", sorted(RWKV7_PRESETS))
def test_param_count_and_flops(name):
    config = RWKV7_PRESETS[name]
    shapes = eqx.filter_eval_shape(RWKV7LM, config, key=jax.random.PRNGKey(0))
    leaves = [x for x in jax.tree_util.tree_leaves(shapes) if isinstance(x, jax.ShapeDtypeStruct)]
    n_params = sum(math.prod(x.shape) for x in leaves)
    assert n_params == rwkv7_param_count(config)
    flops = rwkv7_flops_per_token(config)
    dense = 6.0 * (n_params - config.vocab_size * config.d_model)  # the embedding is a gather
    assert abs(flops - dense) / flops < 0.02 if name != "tiny-rwkv7-blinkdl" else flops > dense


def test_rwkv7_0_1b_dims():
    config = RWKV7_PRESETS["rwkv7-0.1b-blinkdl"]
    assert (config.n_layers, config.d_model, config.head_dim, config.resolved_d_ff) == (12, 768, 64, 3072)
    assert config.resolved_lora_dims == (64, 64, 32, 128)  # BlinkDL 0.1B, also FLA's defaults for D=768
    assert 180e6 < rwkv7_param_count(config) < 200e6


def _perturbed(config):
    model = RWKV7LM(config, key=jax.random.PRNGKey(1))
    params, static = eqx.partition(model, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(2), len(leaves))
    leaves = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)


def test_decode_matches_full_forward():
    model = _perturbed(_tiny())
    ids = jax.random.randint(jax.random.PRNGKey(3), (2, 13), 0, 256)
    with jax.default_matmul_precision("highest"):  # GPU float32 matmuls default to TF32
        full = model(ids)
        cache = model.init_cache(2)
        step = eqx.filter_jit(model.decode_step)
        outs = []
        for t in range(ids.shape[1]):
            logits, cache = step(cache, ids[:, t])
            outs.append(logits)
    np.testing.assert_allclose(np.asarray(jnp.stack(outs, 1)), np.asarray(full), rtol=5e-4, atol=5e-4)


def test_reference_and_chunked_models_agree_with_segments():
    ids = jax.random.randint(jax.random.PRNGKey(4), (2, 24), 0, 256)
    seg = jnp.asarray(np.repeat([[0, 1, 2], [0, 0, 1]], 8, axis=1).astype(np.int32))
    with jax.default_matmul_precision("highest"):  # GPU float32 matmuls default to TF32
        chunked = _perturbed(_tiny(impl="chunked", chunk_size=8))(ids, segment_ids=seg)
        reference = _perturbed(_tiny(impl="reference"))(ids, segment_ids=seg)
        alone = _perturbed(_tiny(impl="reference"))(ids[:1, 8:16])
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(reference), rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(np.asarray(reference[:1, 8:16]), np.asarray(alone), rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("compute_dtype", ["bfloat16", "float32"])
def test_preset_cpu_train_step(compute_dtype):
    config = dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], compute_dtype=compute_dtype)
    model = RWKV7LM(config, key=jax.random.PRNGKey(5))
    ids = jax.random.randint(jax.random.PRNGKey(6), (2, 33), 0, 256)

    @eqx.filter_jit
    def loss_fn(model):
        logits = model(ids[:, :-1])
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(logp, ids[:, 1:, None], axis=-1))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    assert np.isfinite(float(loss)) and abs(float(loss) - math.log(256)) < 1.0
    flat = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in flat)
    if compute_dtype == "float32":  # one SGD step lowers the loss
        updated = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, eqx.filter(model, eqx.is_array), grads)
        new_loss = loss_fn(eqx.combine(updated, eqx.filter(model, eqx.is_array, inverse=True)))
        assert float(new_loss) < float(loss)
