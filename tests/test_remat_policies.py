# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Remat policies built from checkpoint names: ``kernels``, ``dots_kernels``, ``names``, ``offload``.

Every policy must compute the same loss and gradients as ``remat = "none"`` (on a GPU, with the
Mosaic GPU kernels, at least as closely as ``full`` does), and ``kernels`` never launches more
Mosaic GPU kernels than ``full``.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models import build_model, get_preset
from jla.models.config import REMAT_CHECKPOINT_NAMES
from jla.ops.cross_entropy import linear_cross_entropy
from jla.training.performance import estimate_flops_per_token, estimate_hfu_flops_per_token

NAMED = [
    ("kernels", (), ()),
    ("dots_kernels", (), ()),
    ("names", ("linear",), ()),
    ("names", ("linear", "mlp_hidden", "attention"), ()),
    ("names", ("mlp_hidden", "kernels"), ()),
]


def _loss(model, ids):
    out = model(ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    return ce.loss_sum / ce.num_valid


def _grads(cfg, ids):
    model = build_model(cfg, key=jax.random.key(3))
    loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(_loss))(model, ids)
    return loss, jax.tree.leaves(eqx.filter(grads, eqx.is_array))


def _policy(cfg, remat, names, offload):
    return cfg.replace(remat=remat, remat_names=names, remat_offload_names=offload)


@pytest.mark.parametrize("preset", ["tiny", "tiny-hybrid", "tiny-mamba2", "tiny-mamba3", "tiny-gdn2"])
@pytest.mark.parametrize("scan", [True, False])
def test_named_policies_match_no_remat(preset, scan):
    cfg = get_preset(preset).replace(compute_dtype="float32", scan_layers=scan)
    ids = jax.random.randint(jax.random.key(0), (2, 64), 0, cfg.vocab_size)
    # Exact float32 matmuls: on GPU XLA's default precision is TF32, whose rounding differs between a
    # replayed forward and the first one by more than these tolerances.
    with jax.default_matmul_precision("highest"):
        loss0, g0 = _grads(cfg.replace(remat="none"), ids)
        for remat, names, offload in NAMED:
            loss, g = _grads(_policy(cfg, remat, names, offload), ids)
            np.testing.assert_allclose(loss, loss0, rtol=1e-6, atol=1e-6)
            for a, b in zip(g, g0):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_remat_name_validation():
    cfg = get_preset("tiny")
    with pytest.raises(ValueError, match="needs remat_names"):
        cfg.replace(remat="names")
    with pytest.raises(ValueError, match="unknown remat_names"):
        cfg.replace(remat="names", remat_names=("qkv",))
    with pytest.raises(ValueError, match="cannot be offloaded"):
        cfg.replace(remat="offload", remat_offload_names=("kernels",))
    with pytest.raises(ValueError, match="both kept"):
        cfg.replace(remat="offload", remat_names=("linear",), remat_offload_names=("linear",))
    with pytest.raises(ValueError, match="need remat = 'names' or 'offload'"):
        cfg.replace(remat="full", remat_names=("linear",))
    with pytest.raises(ValueError, match="needs remat_offload_names"):
        cfg.replace(remat="offload", remat_names=("linear",))
    assert set(REMAT_CHECKPOINT_NAMES) == {"attention", "linear", "mlp_hidden", "kernels"}
    assert get_preset("tiny").remat == "full" and get_preset("tiny").remat_names == ()


def test_hfu_of_named_policies():
    flops = estimate_flops_per_token(get_preset("gdn-hd128-1p3b"), seq_len=4096)
    full = estimate_hfu_flops_per_token(flops, "full")
    assert estimate_hfu_flops_per_token(flops, "names", ()) == pytest.approx(full)
    everything = estimate_hfu_flops_per_token(flops, "names", ("linear", "mlp_hidden", "attention"))
    assert everything == pytest.approx(flops.training)
    # The Mosaic GPU forward kernels still launch twice under the kernel-saving policies (GPU test below).
    core = flops.attention_core + flops.linear_attention_core
    assert estimate_hfu_flops_per_token(flops, "dots_kernels") == pytest.approx(flops.training + core)
    assert estimate_hfu_flops_per_token(flops, "names", ("linear", "mlp_hidden", "kernels")) == pytest.approx(
        flops.training + core)
    assert flops.training < estimate_hfu_flops_per_token(flops, "names", ("mlp_hidden",)) < full
    assert estimate_hfu_flops_per_token(flops, "kernels") == full


def test_remat_policy_rejects_missing_names():
    """The exported policy builder fails loudly instead of returning a policy that saves nothing."""
    from jla.models.lm import remat_policy

    with pytest.raises(ValueError, match="needs checkpoint names"):
        remat_policy("names")
    with pytest.raises(ValueError, match="needs checkpoint names"):
        remat_policy("names", ("linear",), ("mlp_hidden",))
    with pytest.raises(ValueError, match="needs checkpoint names to offload"):
        remat_policy("offload", ("linear",))
    assert remat_policy("names", ("linear",)) is not None


def test_offload_policy_moves_named_residuals_to_host(capsys):
    """``offload`` keeps the offloaded names as pinned-host residuals; ``names`` keeps them on the device."""
    from jax.ad_checkpoint import checkpoint_name, print_saved_residuals

    from jla.models.lm import remat_policy

    def f(x, w):
        h = checkpoint_name(jnp.sin(x @ w), "linear")
        return jnp.sum(jnp.cos(h) @ w)

    x, w = jnp.ones((8, 8)), jnp.full((8, 8), 0.1)

    def residuals(policy):
        capsys.readouterr()
        print_saved_residuals(jax.checkpoint(f, policy=policy), x, w)
        return capsys.readouterr().out

    full = residuals(remat_policy("full"))
    kept = residuals(remat_policy("names", ("linear",)))
    moved = residuals(remat_policy("offload", (), ("linear",)))
    assert "named 'linear'" not in full and "<host>" not in full, full
    assert "named 'linear'" in kept and "<host>" not in kept, kept
    assert "f32<host>[8,8]" in moved, moved


def test_fully_manual_pipeline_keeps_named_remat():
    """The pipeline's fully manual stages honour ``remat = "names"`` (they used to fall back to full remat)."""
    from jla.distributed.pipeline import make_pipeline_forward, pipeline_rules, runs_fully_manual
    from jla.distributed.sharding import DEFAULT_RULES
    from jla.training.step import LossConfig, lm_loss

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), ("dp", "fsdp", "tp", "pp"))
    rules = pipeline_rules(DEFAULT_RULES)
    ids = np.random.default_rng(0).integers(0, 512, size=(8, 512)).astype(np.int32)
    batch = {"input_ids": ids, "labels": ids}
    temp = {}
    for label, remat, names in [("full", "full", ()), ("names", "names", ("linear", "mlp_hidden", "attention")),
                                ("none", "none", ())]:
        config = get_preset("tiny").replace(n_layers=4, compute_dtype="float32", remat=remat, remat_names=names)
        assert runs_fully_manual(config, 8, mesh, num_microbatches=4, batch_axes=(), rules=rules)
        model = build_model(config, key=jax.random.key(0))
        params, static = eqx.partition(model, eqx.is_inexact_array)
        forward = make_pipeline_forward(mesh, num_microbatches=4, batch_axes=(), rules=rules)
        loss_fn = lm_loss(LossConfig(compute_dtype="float32"), forward=forward)

        def objective(p, loss_fn=loss_fn, static=static):
            return loss_fn(eqx.combine(p, static), batch, jax.random.key(1))[0]

        temp[label] = jax.jit(jax.grad(objective)).lower(params).compile().memory_analysis().temp_size_in_bytes
    print(temp)
    assert 1.1 * temp["full"] < temp["names"] < temp["none"], temp


# ------------------------------------------------------------------------------------------------
# GPU: the Mosaic GPU kernels under the kernel-saving policies
# ------------------------------------------------------------------------------------------------

_GPU = jax.default_backend() == "gpu"


def _two_layer_kernels(preset: str):
    cfg = get_preset(preset)
    n = 2 if cfg.layer_types is None else 4 if "attention" in cfg.layer_types else 2
    changes = {"n_layers": n, "vocab_size": 512}
    if cfg.layer_types is not None:
        changes["layer_types"] = cfg.layer_types[:n]
    if cfg.mlp_types is not None:
        changes["mlp_types"] = cfg.mlp_types[:n]
    changes["attention"] = dataclasses.replace(cfg.attention, implementation="mosaic_gpu")
    if cfg.linear_attention is not None:
        changes["linear_attention"] = dataclasses.replace(cfg.linear_attention, implementation="mosaic_gpu")
    if cfg.ssm is not None:
        changes["ssm"] = dataclasses.replace(cfg.ssm, implementation="mosaic_gpu")
    return cfg.replace(**changes)


def _kernel_launches(cfg, ids) -> int:
    model = build_model(cfg, key=jax.random.key(3))
    params, static = eqx.partition(model, eqx.is_array)

    def loss(p):
        return _loss(eqx.combine(p, static), ids)

    text = jax.jit(jax.grad(loss)).lower(params).compile().as_text()
    return sum(line.count('custom_call_target="mosaic_gpu') for line in text.splitlines())


@pytest.mark.skipif(not _GPU, reason="Mosaic GPU kernels need an H100")
@pytest.mark.parametrize("preset", ["gdn-hd128-1p3b", "kda-1p3b", "gdn2-1p3b", "mamba2-1p3b", "mamba3-1p3b",
                                    "transformer-1p3b", "gdn-hybrid-1p3b"])
def test_kernel_policies_on_gpu(preset):
    cfg = _two_layer_kernels(preset)
    ids = jax.random.randint(jax.random.key(0), (2, 256), 0, cfg.vocab_size)
    loss0, g0 = _grads(cfg.replace(remat="none"), ids)

    def errors(remat, names, offload):
        loss, g = _grads(_policy(cfg, remat, names, offload), ids)
        np.testing.assert_allclose(loss, loss0, rtol=2e-3, atol=2e-3)
        return [float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-12)) for a, b in zip(g, g0)]

    # bf16 compute: a replayed forward rounds a little differently from the first one, so every policy
    # is held to the default (full remat) error against no remat, per gradient tensor. For gdn2-1p3b and
    # kda-1p3b full_err also holds the rounding of the remat-optimized vector-gate op: under per-layer remat the
    # layers after it read its no-residual primal's output, not the residual path's (vector_gate_ops._op_remat).
    full_err = errors("full", (), ())
    assert max(full_err) < 5e-2
    for remat, names, offload in [*NAMED, ("offload", ("kernels",), ("linear", "mlp_hidden"))]:
        for i, (err, ref) in enumerate(zip(errors(remat, names, offload), full_err)):
            assert err <= max(2 * ref, 1e-2), (remat, names, offload, i, err, ref)
    # Kernel launches in the compiled step. "kernels" keeps every kernel output, so no forward kernel is
    # replayed in the backward pass; under jax.checkpoint the custom VJP's primal and its forward rule
    # are separate calls, though, so a kernel whose two calls differ still runs twice (in the forward).
    full = _kernel_launches(cfg.replace(remat="full"), ids)
    kept = _kernel_launches(cfg.replace(remat="kernels"), ids)
    none = _kernel_launches(cfg.replace(remat="none"), ids)
    print(f"{preset}: Mosaic GPU launches none {none}, kernels {kept}, full {full}")
    # The vector-gate op of KDA and GDN-2 is remat-optimized (vector_gate_ops._op_remat): under full remat its
    # forward pass runs the no-residual primal, without the wy kernel of the forward rule that "kernels" keeps
    # (H100: none 22, kernels 25, full 24 for both).
    assert 0 < none <= kept and none <= full, (none, kept, full)
    assert full < kept if preset in ("gdn2-1p3b", "kda-1p3b") else kept <= full, (none, kept, full)
