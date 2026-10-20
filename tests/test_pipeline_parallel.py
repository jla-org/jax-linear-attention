# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallelism (``mesh.pp``): the GPipe step equals the plain step.

The multi-device checks run in subprocesses with 8 CPU devices because the
CPU device count is fixed when JAX starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.config import MeshConfig
from jla.distributed.mesh import MeshPlan, build_mesh, parse_mesh_spec, resolve_mesh_plan
from jla.distributed.pipeline import (
    bubble_fraction,
    pipeline_lm_forward,
    pipeline_rules,
    resolve_microbatches,
    runs_fully_manual,
    shards_head_over_stages,
    stage_layout,
)
from jla.distributed.sharding import DEFAULT_RULES
from jla.models import build_model, get_preset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _full_precision_matmuls():
    with jax.default_matmul_precision("highest"):  # float32 matmuls default to TF32 on GPU
        yield


def test_bubble_fraction_and_microbatches() -> None:
    assert resolve_microbatches(4, 0) == 4 and resolve_microbatches(4, 6) == 6
    assert bubble_fraction(1, 3) == 0.0
    assert bubble_fraction(4, 0) == pytest.approx(3 / 7)
    assert bubble_fraction(2, 8) == pytest.approx(1 / 9)
    with pytest.raises(ValueError, match=">= 0"):
        resolve_microbatches(2, -1)


def test_telemetry_reports_bubble_fraction() -> None:
    from jla.training.telemetry import TelemetryCollector, TelemetryConfig

    metrics = {"loss": 1.0, "grad_norm": 1.0, "step_applied": True, "pipeline/bubble_fraction": 0.2}
    assert TelemetryCollector(TelemetryConfig()).on_step(1, metrics)["telemetry/pipeline_bubble_fraction"] == pytest.approx(0.2)
    assert "telemetry/pipeline_bubble_fraction" not in TelemetryCollector(TelemetryConfig()).on_step(1, {"loss": 1.0})


def test_stage_layout_and_rules() -> None:
    hybrid = get_preset("tiny-hybrid")  # period 2 (gated_deltanet, attention), 4 layers -> 2 repeats
    assert stage_layout(hybrid, 2) == (2, 2, 1)
    with pytest.raises(ValueError, match="must divide"):
        stage_layout(hybrid, 4)
    with pytest.raises(ValueError, match="scan_layers"):
        stage_layout(hybrid.replace(scan_layers=False), 2)
    rules = pipeline_rules(DEFAULT_RULES)
    assert rules["layers"] == ("pp",) and DEFAULT_RULES["layers"] == ()
    assert pipeline_rules(rules)["layers"] == ("pp",)


def test_mesh_plan_pipeline_axis() -> None:
    plan = resolve_mesh_plan(8, dp=2, pp=2)
    assert plan.axis_names == ("dp", "fsdp", "tp", "pp") and plan.shape == (2, 2, 1, 2)
    assert plan.data_parallel_size == 4 and plan.axis_sizes == {"dp": 2, "fsdp": 2, "tp": 1, "pp": 2}
    assert resolve_mesh_plan(8).axis_names == ("dp", "fsdp", "tp")  # pp=1 keeps the 3-axis mesh
    hybrid = resolve_mesh_plan(8, fsdp=2, pp=4, dcn_pp=2)
    assert hybrid.ici_shape == (1, 2, 1, 2) and hybrid.num_slices == 2
    with pytest.raises(ValueError, match="divisible by dcn_pp"):
        MeshPlan(pp=3, dcn_pp=2)
    with pytest.raises(ValueError, match="not divisible"):
        resolve_mesh_plan(8, pp=3)
    assert parse_mesh_spec("dp=2,pp=4") == {"dp": 2, "fsdp": 1, "tp": 1, "pp": 4}
    mesh = build_mesh(resolve_mesh_plan(1), jax.devices()[:1])
    assert mesh.axis_names == ("dp", "fsdp", "tp")
    assert MeshConfig(pp=2, pp_microbatches=4).pp == 2
    with pytest.raises(ValueError, match="pp_microbatches"):
        MeshConfig(pp_microbatches=-1)
    with pytest.raises(ValueError, match="mesh.pp = 1"):
        MeshConfig(rules=(("layers", ("pp",)),))


def test_single_stage_pipeline_matches_model_and_runs_dropout() -> None:
    config = get_preset("tiny").replace(compute_dtype="float32")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("pp",))
    ids = jax.random.randint(jax.random.key(1), (4, 16), 0, config.vocab_size)
    model = build_model(config, key=jax.random.key(0))
    out = pipeline_lm_forward(model, ids, mesh=mesh, num_microbatches=2)
    np.testing.assert_allclose(np.asarray(out.hidden), np.asarray(model(ids).hidden), rtol=1e-5, atol=1e-5)
    assert shards_head_over_stages(4, mesh, num_microbatches=2, batch_axes=())
    assert not shards_head_over_stages(4, mesh, num_microbatches=2, batch_axes=None)
    assert not shards_head_over_stages(6, mesh, num_microbatches=4, batch_axes=())
    head = pipeline_lm_forward(model, ids, mesh=mesh, num_microbatches=2, batch_axes=())  # psum_scatter path
    np.testing.assert_allclose(np.asarray(head.hidden), np.asarray(model(ids).hidden), rtol=1e-5, atol=1e-5)
    # Fully manual shard_map (all four axes of size one), with and without the sharded head.
    full = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), ("dp", "fsdp", "tp", "pp"))
    rules = pipeline_rules(DEFAULT_RULES)
    assert runs_fully_manual(config, 4, full, num_microbatches=2, batch_axes=(), rules=rules)
    assert not runs_fully_manual(config, 4, full, num_microbatches=2, batch_axes=(), rules=None)
    for shard_head in (True, False):
        manual = pipeline_lm_forward(model, ids, mesh=full, num_microbatches=2, batch_axes=(), rules=rules,
                                     shard_head=shard_head)
        np.testing.assert_allclose(np.asarray(manual.hidden), np.asarray(model(ids).hidden), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="not axes of the mesh"):
        shards_head_over_stages(4, mesh, num_microbatches=2, batch_axes=("fsdp",))
    dropout = build_model(config.replace(resid_dropout=0.1), key=jax.random.key(0))
    a = pipeline_lm_forward(dropout, ids, mesh=mesh, num_microbatches=2, key=jax.random.key(2))
    b = pipeline_lm_forward(dropout, ids, mesh=mesh, num_microbatches=2, key=jax.random.key(3))
    assert np.isfinite(np.asarray(a.hidden)).all() and not np.allclose(np.asarray(a.hidden), np.asarray(b.hidden))
    with pytest.raises(ValueError, match="dropout"):
        pipeline_lm_forward(dropout, ids, mesh=mesh)


def test_pipeline_forward_rejects_bad_layouts() -> None:
    model = build_model(get_preset("tiny"), key=jax.random.key(0))  # 2 layers
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "tp"))
    ids = jnp.zeros((4, 8), jnp.int32)
    with pytest.raises(ValueError, match="no pipeline axis"):
        pipeline_lm_forward(model, ids, mesh=mesh)
    pp_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("pp",))
    with pytest.raises(ValueError, match="not divisible"):
        pipeline_lm_forward(model, ids, mesh=pp_mesh, num_microbatches=3)


def test_pipeline_refuses_rwkv7_stacks() -> None:
    from jla.distributed.pipeline import check_pipeline_layers

    # The stage loop does not carry RWKV-7's value residual (v_first) from layer to layer.
    config = get_preset("tiny-rwkv7")
    with pytest.raises(ValueError, match="value residual"):
        check_pipeline_layers(config, 2)
    check_pipeline_layers(config, 1)  # the trainer runs pp = 1 without the pipeline
    model = build_model(config, key=jax.random.key(0))
    pp_mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("pp",))
    with pytest.raises(ValueError, match="value residual"):
        pipeline_lm_forward(model, jnp.zeros((2, 8), jnp.int32), mesh=pp_mesh, inference=True)


def test_pipeline_attention_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses
    import types

    from jla.distributed.pipeline import check_pipeline_attention

    config = get_preset("tiny")  # attention layers, implementation "auto"
    two_devices = types.SimpleNamespace(size=2)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    check_pipeline_attention(config, two_devices)  # CPU: nothing to check
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="implementation = 'xla' or 'mosaic_gpu'"):
        check_pipeline_attention(config, two_devices)
    check_pipeline_attention(config, types.SimpleNamespace(size=1))  # one device: cuDNN compiles
    xla = config.replace(attention=dataclasses.replace(config.attention, implementation="xla"))
    check_pipeline_attention(xla, two_devices)
    check_pipeline_attention(get_preset("tiny-mamba2"), two_devices)
    mla = get_preset("tiny-mla")  # MLA layers call the same attention op ("auto" picks cuDNN)
    with pytest.raises(ValueError, match="model.mla.implementation"):
        check_pipeline_attention(mla, two_devices)
    check_pipeline_attention(mla.replace(mla=dataclasses.replace(mla.mla, implementation="xla")), two_devices)


def test_pipeline_layer_guards() -> None:
    import dataclasses
    import types

    from jla.distributed.pipeline import check_pipeline_layers

    # The DSA indexer's KL gradient ignores the cotangent: bubble iterations would add to it.
    dsa = get_preset("tiny-dsa")
    with pytest.raises(ValueError, match="indexer_loss_coef"):
        check_pipeline_layers(dsa, 2)
    check_pipeline_layers(dsa, 1)  # one stage has no bubble iterations
    check_pipeline_layers(dsa.replace(sparse_attention=dataclasses.replace(dsa.sparse_attention, indexer_loss_coef=0.0)), 2)
    check_pipeline_layers(get_preset("tiny-hybrid"), 2)
    # Mamba-1 runs manual over pp only (XLA cannot compile its gradient fully manual).
    full = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), ("dp", "fsdp", "tp", "pp"))
    rules = pipeline_rules(DEFAULT_RULES)
    assert not runs_fully_manual(get_preset("tiny-mamba1"), 4, full, num_microbatches=2, batch_axes=(), rules=rules)
    assert runs_fully_manual(get_preset("tiny-mamba2"), 4, full, num_microbatches=2, batch_axes=(), rules=rules)
    # MoE with the batch split over two mesh axes keeps the head replicated (XLA aborts otherwise).
    moe = get_preset("tiny-moe")
    split = types.SimpleNamespace(shape={"dp": 2, "fsdp": 2, "tp": 1, "pp": 2})
    assert shards_head_over_stages(16, split, num_microbatches=1, batch_axes=("dp", "fsdp"))
    assert not shards_head_over_stages(16, split, num_microbatches=1, batch_axes=("dp", "fsdp"), config=moe)
    assert shards_head_over_stages(16, split, num_microbatches=1, batch_axes=("dp", "fsdp"), config=get_preset("tiny"))
    one_axis = types.SimpleNamespace(shape={"dp": 1, "fsdp": 4, "tp": 1, "pp": 2})
    assert shards_head_over_stages(16, one_axis, num_microbatches=1, batch_axes=("dp", "fsdp"), config=moe)


def test_fully_manual_pipeline_keeps_model_remat() -> None:
    """``model.remat`` governs the fully manual stages (a nested checkpoint takes the outer policy)."""
    import equinox as eqx

    from jla.distributed.pipeline import make_pipeline_forward
    from jla.training.step import LossConfig, lm_loss

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), ("dp", "fsdp", "tp", "pp"))
    rules = pipeline_rules(DEFAULT_RULES)
    ids = np.random.default_rng(0).integers(0, 512, size=(8, 512)).astype(np.int32)
    batch = {"input_ids": ids, "labels": ids}
    temp = {}
    for remat in ("full", "none"):
        config = get_preset("tiny").replace(n_layers=4, compute_dtype="float32", remat=remat)
        assert runs_fully_manual(config, 8, mesh, num_microbatches=4, batch_axes=(), rules=rules)
        model = build_model(config, key=jax.random.key(0))
        params, static = eqx.partition(model, eqx.is_inexact_array)
        forward = make_pipeline_forward(mesh, num_microbatches=4, batch_axes=(), rules=rules)
        loss_fn = lm_loss(LossConfig(compute_dtype="float32"), forward=forward)

        def objective(p, loss_fn=loss_fn, static=static):
            return loss_fn(eqx.combine(p, static), batch, jax.random.key(1))[0]

        temp[remat] = jax.jit(jax.grad(objective)).lower(params).compile().memory_analysis().temp_size_in_bytes
    assert temp["full"] < 0.5 * temp["none"], temp


STEP_SCRIPT = textwrap.dedent(
    r"""
    import dataclasses
    import json
    import sys

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.distributed.pipeline import BUBBLE_METRIC, bubble_fraction, make_pipeline_forward, pipeline_rules, runs_fully_manual, shards_head_over_stages
    from jla.distributed.sharding import DEFAULT_RULES, batch_axes, grouped_batch_sharding, param_shardings
    from jla.models import build_model, get_preset, logical_axes
    from jla.training.step import LossConfig, build_train_step, lm_loss

    assert len(jax.devices()) == 8, jax.devices()
    loss_config = LossConfig(compute_dtype="float32", z_loss=1e-3)
    T, B = 32, 8

    def make_batch(seed, lead=(), rows=B):
        rng = np.random.default_rng(seed)
        shape = (*lead, rows, T)
        ids = rng.integers(0, 512, size=shape).astype(np.int32)
        labels = rng.integers(0, 512, size=shape).astype(np.int32)
        cut = rng.integers(4, T - 4, size=(*lead, rows, 1))
        t = np.arange(T)
        segment_ids = (t >= cut).astype(np.int32)  # two packed documents per row
        positions = np.where(t >= cut, t - cut, t).astype(np.int32)
        loss_mask = (rng.random(shape) > 0.1).astype(np.int32)
        return {"input_ids": ids, "labels": labels, "segment_ids": segment_ids, "positions": positions, "loss_mask": loss_mask}

    def grads_of(model, batch, forward, mesh=None, rules=None):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        loss_fn = lm_loss(loss_config, forward=forward)
        if mesh is not None:
            params = jax.device_put(params, param_shardings(params, logical_axes(model), mesh, rules))
            sharding = grouped_batch_sharding(mesh, rules, grouped=False)
            batch = {k: jax.device_put(v, sharding) for k, v in batch.items()}

        def f(p):
            return loss_fn(eqx.combine(p, static), batch, jax.random.key(1))

        (objective, aux), grads = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
        return float(objective), {k: float(v) for k, v in aux.items()}, grads, params

    def rel_err(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-12))

    tiny = get_preset("tiny").replace(n_layers=4, compute_dtype="float32")
    hybrid = get_preset("tiny-hybrid").replace(compute_dtype="float32")  # gated_deltanet + attention + MTP
    mamba = get_preset("tiny-mamba2").replace(
        n_layers=4, layer_types=("mamba2",) * 4, mlp_types=("none",) * 4, compute_dtype="float32"
    )
    gdn2 = get_preset("tiny-gdn2").replace(compute_dtype="float32")  # GDN-2 + sliding-window attention
    moe = get_preset("tiny-moe").replace(compute_dtype="float32")
    # With 2 microbatches the load-balancing loss is a mean of per-microbatch losses, not the full-batch
    # loss, so these two cases compare with the auxiliary losses off (the M1 MoE cases keep them on).
    no_aux = dict(aux_loss_coef=0.0, z_loss_coef=0.0)
    moe_xla = moe.replace(moe=dataclasses.replace(moe.moe, ragged_dot_impl="xla", **no_aux))  # dropless, jax.lax.ragged_dot
    moe_capacity = moe.replace(moe=dataclasses.replace(moe.moe, capacity_factor=1.25, **no_aux))
    mamba1 = get_preset("tiny-mamba1").replace(compute_dtype="float32")
    dsa = get_preset("tiny-dsa").replace(compute_dtype="float32")
    dsa_no_indexer_loss = dsa.replace(sparse_attention=dataclasses.replace(dsa.sparse_attention, indexer_loss_coef=0.0))
    # (name, model, mesh axes, pipeline microbatches, mode, batch rows, head sharded, fully manual)
    # mode "pp": manual over pp only, head replicated; "head": batch_axes given (head sharded
    # when the rows allow it); "manual": batch_axes and rules given (fully manual when allowed).
    cases = [
        ("attention/pp4xfsdp2/M4", tiny, dict(fsdp=2, pp=4), 4, "pp", B, False, False),
        ("attention/dp2xfsdp2xpp2/M2", tiny, dict(dp=2, fsdp=2, pp=2), 2, "pp", B, False, False),
        ("hybrid_gdn_mtp/dp2xfsdp2xpp2/M4", hybrid, dict(dp=2, fsdp=2, pp=2), 4, "pp", B, False, False),
        ("mamba2/fsdp2xpp4/M8", mamba, dict(fsdp=2, pp=4), 8, "pp", B, False, False),
        ("gdn2_swa/tp2xfsdp2xpp2/M2", gdn2, dict(fsdp=2, tp=2, pp=2), 2, "pp", B, False, False),
        ("moe/fsdp4xpp2/M1", moe, dict(fsdp=4, pp=2), 1, "pp", B, False, False),
        # MoE with tp inside a stage: the token <-> expert-row moves index plainly there (XLA's SPMD
        # partitioner aborts on the gather-only transposes of jla.ops.moe_permute in a partly manual mesh).
        ("moe_dropless_xla/tp2xfsdp2xpp2/M2", moe_xla, dict(fsdp=2, tp=2, pp=2), 2, "pp", 16, False, False),
        ("moe_capacity/tp2xfsdp2xpp2/M2", moe_capacity, dict(fsdp=2, tp=2, pp=2), 2, "pp", 16, False, False),
        ("attention/fsdp4xpp2/M2/head", tiny, dict(fsdp=4, pp=2), 2, "head", 16, True, False),
        ("hybrid_gdn_mtp/dp2xfsdp2xpp2/M2/head", hybrid, dict(dp=2, fsdp=2, pp=2), 2, "head", 16, True, False),
        ("mamba2/fsdp2xpp4/M2/head", mamba, dict(fsdp=2, pp=4), 2, "head", 16, True, False),
        ("moe/fsdp4xpp2/M1/head", moe, dict(fsdp=4, pp=2), 1, "head", B, True, False),
        ("attention/fsdp4xpp2/M2/manual", tiny, dict(fsdp=4, pp=2), 2, "manual", 16, True, True),
        ("attention/dp2xpp4/M4/manual", tiny, dict(dp=2, pp=4), 4, "manual", B, False, True),
        ("hybrid_gdn_mtp/dp2xfsdp2xpp2/M4/manual", hybrid, dict(dp=2, fsdp=2, pp=2), 4, "manual", 16, False, True),
        ("mamba2/fsdp2xpp4/M2/manual", mamba, dict(fsdp=2, pp=4), 2, "manual", 16, True, True),
        ("gdn2_swa/tp2xfsdp2xpp2/M2/manual_falls_back", gdn2, dict(fsdp=2, tp=2, pp=2), 2, "manual", B, True, False),
        ("moe/fsdp4xpp2/M1/manual_falls_back", moe, dict(fsdp=4, pp=2), 1, "manual", B, True, False),
        # MoE on a batch split over dp and fsdp: the head stays replicated (XLA aborts otherwise).
        ("moe/dp2xfsdp2xpp2/M1/manual_head_replicated", moe, dict(dp=2, fsdp=2, pp=2), 1, "manual", 16, False, False),
        # Mamba-1 runs manual over pp only (its gradient segfaults XLA in a fully manual shard_map).
        ("mamba1/dp2xfsdp2xpp2/M2/manual_falls_back", mamba1, dict(dp=2, fsdp=2, pp=2), 2, "manual", 16, True, False),
        ("dsa_no_indexer_loss/dp2xfsdp2xpp2/M2/manual", dsa_no_indexer_loss, dict(dp=2, fsdp=2, pp=2), 2, "manual", 16, True, True),
    ]
    report = {}
    for name, config, axes, micro, mode, rows, sharded_head, fully_manual in cases:
        batch = make_batch(0, rows=rows)
        model = build_model(config, key=jax.random.key(7))
        ref_obj, ref_aux, ref_grads, _ = grads_of(model, batch, None)
        mesh = build_mesh(resolve_mesh_plan(8, **{"fsdp": 1, **axes}), jax.devices())
        rules = pipeline_rules(DEFAULT_RULES)
        row_axes = None if mode == "pp" else batch_axes(mesh, rules)
        forward_rules = rules if mode == "manual" else None
        assert shards_head_over_stages(rows, mesh, num_microbatches=micro, batch_axes=row_axes, config=config) == sharded_head, name
        assert runs_fully_manual(config, rows, mesh, num_microbatches=micro, batch_axes=row_axes, rules=forward_rules) == fully_manual, name
        forward = make_pipeline_forward(mesh, num_microbatches=micro, batch_axes=row_axes, rules=forward_rules)
        obj, aux, grads, params = grads_of(model, batch, forward, mesh, rules)
        # Stage s stores only its layers: every stacked block parameter is sharded over pp.
        stacked = jax.tree.leaves(eqx.filter(params.blocks, eqx.is_array))
        assert all(leaf.sharding.spec[0] == "pp" for leaf in stacked), [leaf.sharding.spec for leaf in stacked]
        if sharded_head:  # the hidden states leave the pipeline sharded over batch_axes + pp
            static = eqx.filter(model, eqx.is_inexact_array, inverse=True)
            ids = jax.device_put(batch["input_ids"], grouped_batch_sharding(mesh, rules, grouped=False))
            hidden = jax.jit(lambda p, i: forward(eqx.combine(p, static), {"input_ids": i}, None, True).hidden)(params, ids)
            expected = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec((*row_axes, "pp")))
            assert hidden.sharding.is_equivalent_to(expected, hidden.ndim), (name, hidden.sharding)
        assert aux[BUBBLE_METRIC] == np.float32(bubble_fraction(mesh.shape["pp"], micro)), aux
        assert aux["tokens"] == ref_aux["tokens"]
        errs = [rel_err(g, r) for g, r in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads), strict=True)]
        report[name] = {"objective_rel": abs(obj - ref_obj) / abs(ref_obj), "grad_rel_max": max(errs),
                        "aux_loss": [aux["aux_loss"], ref_aux["aux_loss"]], "mtp_loss": [aux["mtp_loss"], ref_aux["mtp_loss"]]}
        np.testing.assert_allclose(obj, ref_obj, rtol=2e-5)
        np.testing.assert_allclose(aux["aux_loss"], ref_aux["aux_loss"], rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(aux["mtp_loss"], ref_aux["mtp_loss"], rtol=2e-5, atol=1e-7)
        assert max(errs) < 1e-4, (name, errs)

    # The compiled train step (grad_accum=2, momentum, donation, pinned shardings) on a dp x fsdp x pp
    # mesh follows the plain single-device step over three updates.
    mesh = build_mesh(resolve_mesh_plan(8, dp=2, fsdp=2, pp=2), jax.devices())
    # The DSA indexer loss is refused for pp > 1 (bubble iterations would add indexer gradient).
    try:
        make_pipeline_forward(mesh, num_microbatches=2)(build_model(dsa, key=jax.random.key(0)), make_batch(0, rows=16), None)
    except ValueError as err:
        assert "indexer_loss_coef" in str(err), err
    else:
        raise AssertionError("pipelined DSA with indexer_loss_coef > 0 must be refused")
    rules = pipeline_rules(DEFAULT_RULES)
    # SGD + momentum: Adam's g / (|g| + eps) would amplify fp32 reduction-order noise on
    # near-zero gradients (the gradients themselves are compared above).
    tx = optax.sgd(0.5, momentum=0.9)

    def train(forward, mesh=None):
        model = build_model(hybrid, key=jax.random.key(3))  # fresh arrays: the step donates its inputs
        params = eqx.filter(model, eqx.is_inexact_array)
        kwargs = {}
        if mesh is not None:
            shardings = param_shardings(params, logical_axes(model), mesh, rules)
            model = eqx.combine(jax.device_put(params, shardings), eqx.filter(model, eqx.is_inexact_array, inverse=True))
            kwargs = {"batch_sharding": grouped_batch_sharding(mesh, rules), "param_shardings": shardings}
        opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        step = build_train_step(tx, lm_loss(loss_config, forward=forward), **kwargs)
        key, losses = jax.random.key(5), []
        for i in range(3):
            model, opt_state, metrics, key = step(model, opt_state, make_batch(10 + i, lead=(2,)), key)
            losses.append(float(metrics["loss"]))
        return model, losses, metrics

    ref_model, ref_losses, _ = train(None)
    pp_model, pp_losses, pp_metrics = train(make_pipeline_forward(mesh, num_microbatches=2), mesh)
    np.testing.assert_allclose(pp_losses, ref_losses, rtol=1e-5)
    assert float(pp_metrics[BUBBLE_METRIC]) == np.float32(1 / 3)
    leaf_errs = [rel_err(a, b) for a, b in zip(jax.tree.leaves(eqx.filter(pp_model, eqx.is_inexact_array)),
                                                jax.tree.leaves(eqx.filter(ref_model, eqx.is_inexact_array)), strict=True)]
    report["train_step/hybrid/dp2xfsdp2xpp2/M2"] = {"losses": pp_losses, "ref_losses": ref_losses, "param_rel_max": max(leaf_errs)}
    assert max(leaf_errs) < 1e-4, leaf_errs
    print("PIPELINE " + json.dumps(report))
    """
)


TRAINER_SCRIPT = textwrap.dedent(
    r"""
    import json
    import sys

    import equinox as eqx
    import jax
    import numpy as np

    from _toy_model import experiment
    from jla.training.checkpointing import CheckpointManager
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 8, jax.devices()
    from jla.models import get_preset

    out = sys.argv[1]
    model = get_preset("tiny").replace(n_layers=4, compute_dtype="float32")

    def run(name, devices, micro, steps=4, resume="auto", save_every=0, final_eval=False, **mesh):
        train = {
            "max_steps": steps, "log_every": 2, "micro_batch_per_replica": micro, "grad_accum": 2,
            "final_eval": final_eval, "ce_compute_dtype": "float32", "save_every": save_every, "resume": resume,
        }
        # One 4-step LR schedule for every leg, so the resumed run follows the uninterrupted one.
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0, "schedule_steps": 4}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def leaves(tree):
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))]

    single = run("single", jax.devices()[:1], 8, final_eval=True, fsdp=1)
    pp2 = run("pp2", jax.devices(), 2, final_eval=True, fsdp=4, pp=2, pp_microbatches=4)
    report = {"eval_loss": [pp2.eval_history[-1]["eval_loss"], single.eval_history[-1]["eval_loss"]]}
    # The evaluation step runs the pipelined forward too (inference=True).
    np.testing.assert_allclose(report["eval_loss"][0], report["eval_loss"][1], rtol=1e-5)
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(leaves(single.model), leaves(pp2.model), strict=True)]
    report["pp2_vs_single_max_abs"] = max(diffs)
    for a, b in zip(leaves(single.model), leaves(pp2.model), strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
    for a, b in zip(single.history, pp2.history, strict=True):
        np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)
    assert pp2.history[-1]["pipeline/bubble_fraction"] == np.float32(1 / 5), pp2.history[-1]
    assert pp2.model.blocks[0].mixer_norm.weight.sharding.spec[0] == "pp"
    assert pp2.model.embed.weight.sharding.spec == jax.sharding.PartitionSpec(None, "fsdp")

    # micro_batch_per_replica = pp_microbatches * pp: the trainer runs the head and loss sharded over pp.
    head = run("pp2_head", jax.devices(), 2, final_eval=True, fsdp=4, pp=2, pp_microbatches=1)
    for a, b in zip(leaves(single.model), leaves(head.model), strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
    for a, b in zip(single.history, head.history, strict=True):
        np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)
    np.testing.assert_allclose(head.eval_history[-1]["eval_loss"], single.eval_history[-1]["eval_loss"], rtol=1e-5)
    assert head.history[-1]["pipeline/bubble_fraction"] == np.float32(1 / 2), head.history[-1]
    report["pp2_head_sharded_vs_single_max_abs"] = max(
        float(np.max(np.abs(a - b))) for a, b in zip(leaves(single.model), leaves(head.model), strict=True)
    )

    # Resume across pipeline sizes: 2 steps at pp=2, then continue to step 4 at pp=4 (and at pp=1).
    run("resume", jax.devices(), 2, steps=2, save_every=2, fsdp=4, pp=2, pp_microbatches=4)
    resumed = run("resume", jax.devices(), 4, steps=4, resume="required", fsdp=2, pp=4, pp_microbatches=4)
    assert resumed.history[0]["step"] > 2, resumed.history
    for a, b in zip(leaves(single.model), leaves(resumed.model), strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
    assert resumed.model.blocks[0].mixer_norm.weight.sharding.spec[0] == "pp"
    report["resume_pp2_to_pp4_vs_single_max_abs"] = max(
        float(np.max(np.abs(a - b))) for a, b in zip(leaves(single.model), leaves(resumed.model), strict=True)
    )
    # The pp=2 checkpoint restores bit-exactly into the single-device (pp=1) layout.
    restored = CheckpointManager(f"{out}/pp2/checkpoints").restore(model=single.model, opt_state=single.opt_state)
    for a, b in zip(leaves(restored.model), leaves(pp2.model), strict=True):
        np.testing.assert_array_equal(a, b)
    print("PIPELINE " + json.dumps(report))
    """
)


def _run(tmp_path: Path, script_text: str) -> dict:
    script = tmp_path / "pipeline_script.py"
    script.write_text(script_text)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "runs")],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    report = json.loads(result.stdout.split("PIPELINE ", 1)[1])
    print(json.dumps(report, indent=1))
    return report


def test_pipeline_step_matches_plain_step_on_8_devices(tmp_path: Path) -> None:
    report = _run(tmp_path, STEP_SCRIPT)
    assert all(entry.get("grad_rel_max", 0.0) < 1e-4 for entry in report.values())


def test_pipeline_trainer_and_resume_across_pp_sizes(tmp_path: Path) -> None:
    report = _run(tmp_path, TRAINER_SCRIPT)
    assert report["pp2_vs_single_max_abs"] < 1e-4


EP_PP_TRAINER_SCRIPT = textwrap.dedent(
    r"""
    import dataclasses
    import json
    import sys

    import equinox as eqx
    import jax
    import numpy as np

    from _toy_model import experiment
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 8, jax.devices()
    out = sys.argv[1]
    base = get_preset("tiny-moe").replace(compute_dtype="float32")
    # No load-balancing loss: it is not linear in the tokens, so pipeline microbatches would change it.
    model = base.replace(moe=dataclasses.replace(base.moe, aux_loss_coef=0.0))
    ep_model = model.replace(moe=dataclasses.replace(model.moe, expert_parallel_axis="ep"))

    def run(name, model, devices, micro, **mesh):
        train = {"max_steps": 4, "log_every": 2, "micro_batch_per_replica": micro, "grad_accum": 2,
                 "final_eval": True, "ce_compute_dtype": "float32"}
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0, "schedule_steps": 4}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def leaves(tree):
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))]

    single = run("single", model, jax.devices()[:1], 8, fsdp=1)
    both = run("ep2_pp2", ep_model, jax.devices(), 2, fsdp=2, ep=2, pp=2, pp_microbatches=4)
    for a, b in zip(single.history, both.history, strict=True):
        np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-4)
    np.testing.assert_allclose(both.eval_history[-1]["eval_loss"], single.eval_history[-1]["eval_loss"], rtol=1e-4)
    for a, b in zip(leaves(single.model), leaves(both.model), strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)
    specs = [str(leaf.sharding.spec) for leaf in jax.tree.leaves(eqx.filter(both.model, eqx.is_inexact_array))]
    assert any("'ep'" in spec for spec in specs), specs
    assert both.model.blocks[0].mixer_norm.weight.sharding.spec[0] == "pp"
    report = {"ep2_pp2_vs_single_max_abs": max(
        float(np.max(np.abs(a - b))) for a, b in zip(leaves(single.model), leaves(both.model), strict=True))}
    print("PIPELINE " + json.dumps(report))
    """
)


def test_pipeline_trainer_with_expert_parallel_matches_single_device(tmp_path: Path) -> None:
    """Expert and pipeline parallelism in one Trainer run (fsdp 2 x ep 2 x pp 2) match one device."""
    report = _run(tmp_path, EP_PP_TRAINER_SCRIPT)
    assert report["ep2_pp2_vs_single_max_abs"] < 1e-4
