# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``train.ce_grad_in_forward`` reaches the MTP heads: their cross entropy runs through
``linear_cross_entropy_objective`` (one call per depth, next to the main head's), with the same
loss and gradients as the default path. On an H100 a bf16 ``tiny-hybrid`` (two MTP heads) runs every
head on the kernels, against the XLA path and a float32 reference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models import build_model, get_preset
from jla.ops import cross_entropy as ce_module
from jla.ops.mosaic_gpu._common import is_hopper
from jla.training.step import LossConfig, lm_loss

gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _batch(vocab: int):
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, vocab, size=(2, 65))
    return {"input_ids": jnp.asarray(tokens[:, :-1], jnp.int32), "labels": jnp.asarray(tokens[:, 1:], jnp.int32),
            "loss_mask": jnp.asarray(rng.random((2, 64)) > 0.2)}


def _run(model, batch, cfg):
    loss_fn = lm_loss(cfg)
    f = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, jax.random.key(1)), has_aux=True))
    (objective, aux), grads = f(model)
    return float(objective), aux, jax.tree.leaves(eqx.filter(grads, eqx.is_array))


def _mtp_grads(model, batch, cfg):
    """Gradients of the MTP heads' own parameters (they depend on the MTP losses only)."""
    loss_fn = lm_loss(cfg)
    f = eqx.filter_jit(eqx.filter_grad(lambda m: loss_fn(m, batch, jax.random.key(1))[0]))
    return jax.tree.leaves(eqx.filter(f(model).mtp, eqx.is_array))


def test_mtp_heads_use_the_objective(monkeypatch):
    model_cfg = get_preset("tiny-hybrid").replace(compute_dtype="float32")
    assert model_cfg.mtp is not None and model_cfg.mtp.depth == 2
    model = build_model(model_cfg, key=jax.random.key(0))
    batch = _batch(model_cfg.vocab_size)
    calls = []
    original = ce_module.linear_cross_entropy_objective

    def spy(*args, **kwargs):
        calls.append(kwargs.get("z_loss_weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ce_module, "linear_cross_entropy_objective", spy)
    # A float32 cross entropy keeps "auto" on the XLA op (plain autodiff) on every device, so both runs match.
    obj_d, aux_d, g_d = _run(model, batch, LossConfig(z_loss=1e-2, compute_dtype="float32"))
    assert calls == []
    obj_f, aux_f, g_f = _run(model, batch, LossConfig(z_loss=1e-2, impl="auto", grad_in_forward=True,
                                                     compute_dtype="float32"))
    assert calls == [1e-2, 0.0, 0.0]  # the main head (with its z-loss), then one call per MTP depth
    np.testing.assert_allclose(obj_f, obj_d, rtol=1e-5)
    for key in ("loss_sum", "z_loss_sum", "mtp_loss", "mtp/loss_1", "mtp/loss_2"):
        np.testing.assert_allclose(float(aux_f[key]), float(aux_d[key]), rtol=1e-5)
    for a, b in zip(g_f, g_d):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-7)


def test_models_without_the_keyword_still_work():
    """A custom model whose ``mtp_loss`` takes no ``objective`` keyword keeps its own cross entropy."""

    class NoKeyword(eqx.Module):
        inner: eqx.Module

        def __call__(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

        def lm_head_weight(self):
            return self.inner.lm_head_weight()

        def mtp_loss(self, out, input_ids, targets, *, positions=None, segment_ids=None):
            return self.inner.mtp_loss(out, input_ids, targets, positions=positions, segment_ids=segment_ids)

    model_cfg = get_preset("tiny-hybrid").replace(compute_dtype="float32")
    model = build_model(model_cfg, key=jax.random.key(0))
    batch = _batch(model_cfg.vocab_size)
    obj_w, aux_w, _ = _run(NoKeyword(model), batch, LossConfig(impl="auto", grad_in_forward=True))
    obj_d, aux_d, _ = _run(model, batch, LossConfig())
    np.testing.assert_allclose(obj_w, obj_d, rtol=1e-5)
    np.testing.assert_allclose(float(aux_w["mtp_loss"]), float(aux_d["mtp_loss"]), rtol=1e-5)


@gpu
def test_mtp_heads_on_the_kernels_match_xla(monkeypatch):
    """bf16 ``tiny-hybrid``: the main head and both MTP heads run the forward-gradient kernels."""
    from jla.ops.mosaic_gpu import linear_cross_entropy as fused

    model_cfg = get_preset("tiny-hybrid")
    model = build_model(model_cfg, key=jax.random.key(0))
    batch = _batch(model_cfg.vocab_size)
    calls = []
    original = fused.objective_sums

    def spy(*args, **kwargs):
        calls.append(kwargs.get("z_loss_weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "objective_sums", spy)
    obj_x, aux_x, g_x = _run(model, batch, LossConfig(z_loss=1e-2))
    obj_f, aux_f, g_f = _run(model, batch, LossConfig(z_loss=1e-2, impl="mosaic_gpu", grad_in_forward=True,
                                                     row_chunk=256))
    assert calls == [1e-2, 0.0, 0.0]
    np.testing.assert_allclose(obj_f, obj_x, rtol=1e-4)
    for key in ("loss_sum", "z_loss_sum", "mtp_loss", "mtp/loss_1", "mtp/loss_2"):
        np.testing.assert_allclose(float(aux_f[key]), float(aux_x[key]), rtol=1e-4)
    flat = lambda gs: np.concatenate([np.asarray(g, np.float64).ravel() for g in gs])  # noqa: E731
    rel = lambda a, b: float(np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b)))  # noqa: E731
    # The MTP heads' own parameters see only the MTP losses: their gradients isolate the heads' kernel path.
    mtp_x = _mtp_grads(model, batch, LossConfig(z_loss=1e-2))
    mtp_f = _mtp_grads(model, batch, LossConfig(z_loss=1e-2, impl="mosaic_gpu", grad_in_forward=True, row_chunk=256))
    # Against a float32 reference (the same parameters, float32 compute and cross entropy, XLA): the bf16
    # trunk and MTP blocks amplify last-bit differences (about 5e-3 between the two bf16 runs), so the kernel
    # path must be as close to float32 as the XLA path is.
    ref_model = build_model(model_cfg.replace(compute_dtype="float32"), key=jax.random.key(0))
    ref_cfg = LossConfig(z_loss=1e-2, compute_dtype="float32")
    mtp_ref, g_ref = _mtp_grads(ref_model, batch, ref_cfg), _run(ref_model, batch, ref_cfg)[2]
    print(f"relative gradient error vs float32: MTP heads XLA {rel(mtp_x, mtp_ref):.2e} kernels "
          f"{rel(mtp_f, mtp_ref):.2e}; all parameters XLA {rel(g_x, g_ref):.2e} kernels {rel(g_f, g_ref):.2e}; "
          f"kernels vs XLA {rel(mtp_f, mtp_x):.2e} / {rel(g_f, g_x):.2e}")
    assert rel(mtp_f, mtp_ref) <= 1.25 * rel(mtp_x, mtp_ref) + 1e-3
    assert rel(g_f, g_ref) <= 1.25 * rel(g_x, g_ref) + 1e-3


@pytest.mark.skipif(not is_hopper() or jax.device_count() < 2, reason="needs two H100s")
def test_mtp_heads_on_the_kernels_match_xla_under_fsdp_on_two_gpus(monkeypatch):
    """bf16 ``tiny-hybrid`` under a two-GPU FSDP mesh: the main head and both MTP heads run the kernels per shard with
    the float32 weight gathered in bf16 and its gradient reduce-scattered (``_column_gradient_copy``), as XLA."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    gathers = []
    original = ce_module._column_gradient_copy

    def spy(*args):
        gathers.append(args[0].shape)
        return original(*args)

    monkeypatch.setattr(ce_module, "_column_gradient_copy", spy)
    model_cfg = get_preset("tiny-hybrid")
    model = build_model(model_cfg, key=jax.random.key(0))
    mesh = build_mesh(resolve_mesh_plan(2, dp=1, fsdp=2, tp=1), topology_aware=False)
    params, static = eqx.partition(model, eqx.is_array)  # replicated on the mesh, as named shardings
    model = eqx.combine(jax.device_put(params, NamedSharding(mesh, P())), static)
    rows = [_batch(model_cfg.vocab_size), _batch(model_cfg.vocab_size - 1)]  # 4 rows: 2 x 64 tokens per GPU
    batch = {key: jax.device_put(jnp.concatenate([r[key] for r in rows]), NamedSharding(mesh, P("fsdp")))
             for key in rows[0]}
    with jax.set_mesh(mesh):
        obj_x, aux_x, g_x = _run(model, batch, LossConfig(z_loss=1e-2))
        obj_f, aux_f, g_f = _run(model, batch, LossConfig(z_loss=1e-2, impl="mosaic_gpu", grad_in_forward=True,
                                                         row_chunk=256))
    assert len(gathers) == 3, gathers  # the main head and both MTP heads
    np.testing.assert_allclose(obj_f, obj_x, rtol=1e-4)
    for key in ("loss_sum", "z_loss_sum", "mtp_loss", "mtp/loss_1", "mtp/loss_2"):
        np.testing.assert_allclose(float(aux_f[key]), float(aux_x[key]), rtol=1e-4)
    flat = lambda gs: np.concatenate([np.asarray(g, np.float64).ravel() for g in gs])  # noqa: E731
    rel = float(np.linalg.norm(flat(g_f) - flat(g_x)) / np.linalg.norm(flat(g_x)))
    print(f"kernels vs XLA under FSDP 2: relative gradient difference {rel:.2e}")
    assert rel < 3e-2  # bf16 noise (about 5e-3 on one GPU); a lost or doubled weight-gradient shard is O(1)
