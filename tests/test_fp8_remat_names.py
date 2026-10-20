# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 projections carry the checkpoint names of the bf16 ones, so the "names" remat policies keep them.

``Fp8Linear`` / ``Fp8StackedLinear`` outputs are named ``linear`` (as ``Linear`` / ``StackedLinear``), the
unfused ``Fp8SwiGLU`` gate/up product ``mlp_hidden`` (as ``SwiGLU``), and the fused FP8 SwiGLU
(``fp8_fused``) names the same two values inside its forward rule. The tags are identities: every policy
computes the loss, gradients and next FP8 states of ``remat = "none"``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ad_checkpoint import print_saved_residuals

from jla.layers.fp8 import Fp8Linear, Fp8State, Fp8StackedLinear
from jla.layers.fp8_swiglu import fp8_swiglu_mlp
from jla.models import build_model, get_preset
from jla.models.lm import remat_policy
from jla.ops.cross_entropy import linear_cross_entropy


def _saved(capsys, fn, *args, names=()):
    capsys.readouterr()
    print_saved_residuals(jax.checkpoint(fn, policy=remat_policy("names", names)), *args)
    return capsys.readouterr().out


def _state(lead=()):
    return Fp8State.init(16, lead)


def test_fp8_linear_outputs_are_named(capsys):
    x = jax.random.normal(jax.random.key(0), (4, 32, 64), jnp.float32)
    lin = Fp8Linear(jax.random.normal(jax.random.key(1), (64, 48)) * 0.1, None, _state(), "float32",
                    ("embed", "mlp"), "delayed")
    stacked = Fp8StackedLinear(jax.random.normal(jax.random.key(2), (64, 2, 48)) * 0.1, _state(), "float32",
                               ("embed", None, "mlp"), "delayed")

    def f(model, x):
        return jnp.sum(jnp.sin(model(x)))

    for model in (lin, stacked):
        kept = _saved(capsys, f, model, x, names=("linear",))
        assert "named 'linear'" in kept, kept
        other = _saved(capsys, f, model, x, names=("mlp_hidden",))
        assert "named 'linear'" not in other, other


def test_fused_fp8_swiglu_names_its_hidden_and_output(capsys):
    x = jax.random.normal(jax.random.key(0), (2, 32, 64), jnp.float32)
    w_gu = jax.random.normal(jax.random.key(1), (64, 2, 96)) * 0.1
    w_dn = jax.random.normal(jax.random.key(2), (96, 64)) * 0.1

    def f(x, w_gu, w_dn):
        return jnp.sum(jnp.sin(fp8_swiglu_mlp(x, w_gu, w_dn, _state(), _state(), impl="xla")))

    hidden = _saved(capsys, f, x, w_gu, w_dn, names=("mlp_hidden",))
    assert "named 'mlp_hidden'" in hidden, hidden
    assert "f32[64,2,96]" in hidden or "[64,2,96]" in hidden, hidden  # the gate/up product of 64 tokens
    out = _saved(capsys, f, x, w_gu, w_dn, names=("linear",))
    assert "named 'linear'" in out, out


def _loss(model, ids):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    return ce.loss_sum / ce.num_valid


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("preset", ["tiny", "tiny-gdn2"])
def test_fp8_named_policies_match_no_remat(preset, fused):
    base = get_preset(preset).replace(compute_dtype="float32", fp8="delayed", fp8_fused=fused, fp8_amax_history=16)
    ids = jax.random.randint(jax.random.key(0), (2, 64), 0, base.vocab_size)

    def grads(cfg):
        model = build_model(cfg, key=jax.random.key(3))
        loss, g = eqx.filter_jit(eqx.filter_value_and_grad(_loss))(model, ids)
        return loss, jax.tree.leaves(eqx.filter(g, eqx.is_array))

    with jax.default_matmul_precision("highest"):
        loss0, g0 = grads(base.replace(remat="none"))
        for remat, names in (("full", ()), ("names", ("linear",)), ("names", ("linear", "mlp_hidden", "attention")),
                             ("names", ("mlp_hidden", "kernels"))):
            loss, g = grads(base.replace(remat=remat, remat_names=names))
            np.testing.assert_allclose(loss, loss0, rtol=1e-6, atol=1e-6)
            assert len(g) == len(g0)
            for a, b in zip(g, g0):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
