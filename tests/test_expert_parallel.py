# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert parallelism: experts sharded over an ``ep`` mesh axis, tokens exchanged with all-to-all.

The equality tests run in a subprocess with 8 CPU devices (the device count is
fixed when JAX starts): the EP layer's outputs, dropped fraction and gradients
equal the layer without EP (dropless and capacity routing), the bounded-buffer
mode drops exactly the routes that overflow a block, and Trainer runs with
``mesh.ep`` match a one-device run step for step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from jla.config import MeshConfig
from jla.distributed.expert_parallel import block_rows, token_axes
from jla.distributed.mesh import MeshPlan, parse_mesh_spec, resolve_mesh_plan
from jla.distributed.sharding import DEFAULT_RULES, batch_axes, logical_to_spec
from jla.layers.moe import MoE
from jla.models.config import MoEConfig

ROOT = Path(__file__).resolve().parents[1]


def test_mesh_plan_appends_ep_axis_only_when_used() -> None:
    plain = resolve_mesh_plan(8, dp=2, fsdp=-1)
    assert plain.axis_names == ("dp", "fsdp", "tp") and plain.shape == (2, 4, 1)
    plan = resolve_mesh_plan(8, dp=2, fsdp=-1, ep=2)
    assert plan.axis_names == ("dp", "fsdp", "tp", "ep") and plan.shape == (2, 2, 1, 2)
    assert plan.data_parallel_size == 8 and plan.axis_sizes == {"dp": 2, "fsdp": 2, "tp": 1, "ep": 2}
    assert MeshPlan(ep=4, dcn_ep=2).dcn_shape == (1, 1, 1, 2)
    with pytest.raises(ValueError, match="divisible by dp\\*tp\\*ep"):
        resolve_mesh_plan(8, ep=3)
    with pytest.raises(ValueError, match="divisible by dcn_ep"):
        MeshPlan(ep=3, dcn_ep=2)
    assert parse_mesh_spec("dp=2,ep=4") == {"dp": 2, "fsdp": 1, "tp": 1, "ep": 4}
    assert parse_mesh_spec("2,4") == {"dp": 2, "fsdp": 4, "tp": 1}
    with pytest.raises(ValueError, match="mesh.ep"):
        MeshConfig(ep=0)
    with pytest.raises(ValueError, match="mesh.dcn_ep"):
        MeshConfig(ep=3, dcn_ep=2)
    assert MeshConfig(ep=2, rules=(("experts", ("ep", "fsdp")),)).ep == 2


def test_rules_shard_experts_and_batch_over_ep() -> None:
    mesh = SimpleNamespace(shape={"dp": 1, "fsdp": 2, "tp": 1, "ep": 4})
    spec = logical_to_spec(("experts", "embed", None, "mlp"), (8, 64, 2, 32), mesh, DEFAULT_RULES)
    assert tuple(spec) == ("ep", "fsdp", None, None)
    assert batch_axes(mesh, DEFAULT_RULES) == ("fsdp", "ep")
    # A mesh without the ep axis treats it as size one (the default layout is unchanged).
    plain = SimpleNamespace(shape={"dp": 2, "fsdp": 2, "tp": 1})
    assert tuple(logical_to_spec(("experts", "embed"), (8, 64), plain, DEFAULT_RULES)) == (None, "fsdp")
    assert batch_axes(plain, DEFAULT_RULES) == ("dp", "fsdp")
    assert token_axes(SimpleNamespace(axis_names=("dp", "fsdp", "tp", "ep")), "ep") == ("dp", "fsdp", "ep")


def test_block_rows_and_config_validation() -> None:
    assert block_rows(16, 2, 4, 2, None) == 32  # every route of 16 tokens may target one peer
    assert block_rows(16, 2, 1, 8, None) == 16  # a peer owning one expert gets <= 1 route per token
    assert block_rows(16, 2, 4, 2, 1.0) == 16 and block_rows(16, 2, 4, 2, 100.0) == 32
    with pytest.raises(ValueError, match="needs expert_parallel_axis"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, expert_parallel_buffer_factor=1.0)
    with pytest.raises(ValueError, match="mesh axis name"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, expert_parallel_axis="")


def test_experiment_config_requires_an_ep_mesh_for_expert_parallelism() -> None:
    """A typo or ``mesh.ep = 1`` must not silently train without expert parallelism."""
    import dataclasses

    from jla.config import ExperimentConfig
    from jla.models import get_preset

    model = get_preset("tiny-moe")
    ep_model = model.replace(moe=dataclasses.replace(model.moe, expert_parallel_axis="ep"))
    assert ExperimentConfig(model=ep_model, mesh=MeshConfig(fsdp=2, ep=4)).mesh.ep == 4
    with pytest.raises(ValueError, match="needs the 'ep' mesh axis with mesh.ep > 1"):
        ExperimentConfig(model=ep_model, mesh=MeshConfig())
    typo = model.replace(moe=dataclasses.replace(model.moe, expert_parallel_axis="expert"))
    with pytest.raises(ValueError, match="needs the 'ep' mesh axis"):
        ExperimentConfig(model=typo, mesh=MeshConfig(ep=2))
    assert ExperimentConfig(model=model, mesh=MeshConfig(ep=2)).mesh.ep == 2  # experts sharded by GSPMD


def test_trainer_rejects_context_parallel_models(tmp_path: Path) -> None:
    """The Trainer traces under its mesh, which has no context-parallel axis: fail before training."""
    import dataclasses

    from _toy_model import experiment
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    model = get_preset("tiny-hybrid")
    cp = model.replace(linear_attention=dataclasses.replace(model.linear_attention, context_parallel_axis="fsdp"))
    config = experiment(str(tmp_path / "cp"), model=cp, train={"max_steps": 1, "final_eval": False},
                        optim={"warmup_steps": 0}, mesh={"fsdp": 1})
    with pytest.raises(NotImplementedError, match="context_parallel_axis"):
        Trainer(config, devices=jax.devices()[:1]).run()


@pytest.mark.parametrize("section", ["attention", "local_attention"])
def test_trainer_rejects_context_parallel_softmax_attention(tmp_path: Path, section: str) -> None:
    """Context-parallel softmax attention is library-only for now: the Trainer refuses it before training."""
    import dataclasses

    from _toy_model import experiment
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    model = get_preset("tiny")
    attention = dataclasses.replace(model.attention, context_parallel_axis="fsdp")
    if section == "attention":
        cp = model.replace(attention=attention)
    else:  # a stack of local_attention layers only
        cp = model.replace(local_attention=dataclasses.replace(attention, sliding_window=8),
                           layer_types=("local_attention",) * model.n_layers)
    config = experiment(str(tmp_path / section), model=cp, train={"max_steps": 1, "final_eval": False},
                        optim={"warmup_steps": 0}, mesh={"fsdp": 1})
    with pytest.raises(NotImplementedError, match=f"{section}.context_parallel_axis='fsdp'"):
        Trainer(config, devices=jax.devices()[:1]).run()


@pytest.mark.parametrize("capacity_factor", [None, 1.0])
def test_expert_parallel_without_a_mesh_runs_the_local_dispatch(capacity_factor) -> None:
    """No context mesh (decoding): the layer without EP; a context mesh without the axis: an error."""
    import dataclasses

    import numpy as np
    from jax.sharding import Mesh

    cfg = MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=capacity_factor, expert_parallel_axis="ep")
    layer = MoE(16, cfg, key=jax.random.key(0), std=0.5, compute_dtype="float32")
    base = MoE(16, dataclasses.replace(cfg, expert_parallel_axis=None), key=jax.random.key(0), std=0.5,
               compute_dtype="float32")
    x = jax.random.normal(jax.random.key(1), (2, 8, 16))
    y0, s0 = base(x)
    y1, s1 = layer(x)
    np.testing.assert_array_equal(np.asarray(y1), np.asarray(y0))
    assert float(s1.dropped_fraction) == float(s0.dropped_fraction)
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "tp"))):
        with pytest.raises(ValueError, match="not an axis of the context mesh"):
            jax.jit(lambda layer, x: layer(x))(layer, x)
    # Capacity routing may run the expert-parallel grouped GEMMs on the Mosaic GPU kernels.
    assert MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=1.0, expert_parallel_axis="ep",
                     ragged_dot_impl="mosaic_gpu").ragged_dot_impl == "mosaic_gpu"
    with pytest.raises(ValueError, match="dropless routing only"):
        MoEConfig(num_experts=4, top_k=2, d_ff_expert=8, capacity_factor=1.0, ragged_dot_impl="mosaic_gpu")


def _hopper() -> bool:
    return jax.default_backend() == "gpu" and "H100" in jax.devices()[0].device_kind


@pytest.mark.skipif(not _hopper(), reason="Mosaic GPU ragged_dot needs an H100")
def test_mosaic_ragged_dot_inside_expert_parallel_shard_map() -> None:
    """The Mosaic GPU grouped GEMM under the EP ``shard_map`` (one-device mesh)."""
    import dataclasses

    import equinox as eqx
    import numpy as np
    from jax.sharding import Mesh

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1), ("dp", "fsdp", "tp", "ep"))
    cfg = MoEConfig(num_experts=8, top_k=2, d_ff_expert=128, expert_parallel_axis="ep", ragged_dot_impl="mosaic_gpu",
                    expert_parallel_buffer_factor=1.5)
    key = jax.random.key(0)
    ep = MoE(256, cfg, key=key, std=0.05, compute_dtype="bfloat16")
    # The same kernel without EP: bfloat16 results of the XLA and Mosaic grouped GEMMs differ by a few
    # percent in the router gradient (both equally far from a float32 oracle).
    base_cfg = dataclasses.replace(cfg, expert_parallel_axis=None, expert_parallel_buffer_factor=None)
    base = MoE(256, base_cfg, key=key, std=0.05, compute_dtype="bfloat16")
    x = jax.random.normal(jax.random.key(1), (2, 512, 256), jnp.float32).astype(jnp.bfloat16)

    def loss(layer, x):
        y, stats = layer(x)
        return jnp.sum(y.astype(jnp.float32) ** 2) + stats.aux_loss, y

    with jax.set_mesh(mesh):
        (_, y1), g1 = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(ep, x)
    (_, y0), g0 = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(base, x)

    def rel(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert rel(y1, y0) < 1e-3
    for a, b in zip(jax.tree.leaves(eqx.filter(g1, eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(g0, eqx.is_inexact_array)), strict=True):
        assert rel(a, b) < 1e-3

    # Capacity routing leaves rows of the receive buffer in no group (dropped routes): the Mosaic and
    # XLA grouped GEMMs of the expert-parallel layer agree, with the same drops.
    capacity = dataclasses.replace(cfg, capacity_factor=1.0, expert_parallel_buffer_factor=None)
    outs = []
    for impl in ("mosaic_gpu", "xla"):
        layer = MoE(256, dataclasses.replace(capacity, ragged_dot_impl=impl), key=key, std=0.05, compute_dtype="bfloat16")
        with jax.set_mesh(mesh):
            (_, y), g = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(layer, x)
            _, stats = eqx.filter_jit(lambda layer, x: layer(x))(layer, x)
        outs.append((y, g, float(stats.dropped_fraction)))
    (ym, gm, dm), (yx, gx, dx) = outs
    assert dm == dx > 0.0
    assert rel(ym, yx) < 5e-3
    assert rel(gm.gate_up, gx.gate_up) < 2e-2 and rel(gm.down, gx.down) < 2e-2

    # Bounded blocks (buffer factor 0.5 < worst case, slot-major drops) on the GPU: same drops and
    # outputs with both grouped GEMMs.
    bounded = dataclasses.replace(cfg, expert_parallel_buffer_factor=0.5)
    outs = []
    for impl in ("mosaic_gpu", "xla"):
        layer = MoE(256, dataclasses.replace(bounded, ragged_dot_impl=impl), key=key, std=0.05, compute_dtype="bfloat16")
        with jax.set_mesh(mesh):
            y, stats = eqx.filter_jit(lambda layer, x: layer(x))(layer, x)
        outs.append((y, float(stats.dropped_fraction)))
    (ym, dm), (yx, dx) = outs
    assert dm == dx and 0.4 < dm < 0.6  # half the routes fit
    assert rel(ym, yx) < 5e-3


SCRIPT = textwrap.dedent(
    r"""
    import dataclasses
    import json
    import sys
    from pathlib import Path

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from _toy_model import experiment
    from jla.distributed.expert_parallel import block_rows
    from jla.distributed.sharding import DEFAULT_RULES, param_shardings
    from jla.layers.mlp import swiglu
    from jla.layers.moe import MoE
    from jla.models import get_preset, logical_axes
    from jla.models.config import MoEConfig
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 8, jax.devices()
    out = sys.argv[1]
    report = {}

    def layer_case(name, mesh_shape, cfg, *, dtype="float32", tol=1e-5, batch=8, seq=16, d_model=32,
                   call_dropless=False, base_on_mesh=True):
        # base_on_mesh=False: the reference is the layer without EP on one device (no mesh, no sharding).
        names = ("dp", "fsdp", "tp", "ep")
        mesh = Mesh(np.asarray(jax.devices()).reshape(mesh_shape), names)
        key = jax.random.key(7)
        base = MoE(d_model, dataclasses.replace(cfg, expert_parallel_axis=None, expert_parallel_buffer_factor=None),
                   key=key, std=0.5, compute_dtype=dtype)
        ep = MoE(d_model, cfg, key=key, std=0.5, compute_dtype=dtype)
        x_host = jax.random.normal(jax.random.key(1), (batch, seq, d_model), jnp.float32).astype(dtype)
        cot = jax.random.normal(jax.random.key(2), (batch, seq, d_model), jnp.float32)
        x = jax.device_put(x_host, NamedSharding(mesh, P(("dp", "fsdp", "ep"))))

        def run(layer, on_mesh=True):
            params, static = eqx.partition(layer, eqx.is_array)
            if on_mesh:
                params = jax.device_put(params, param_shardings(params, logical_axes(layer), mesh, DEFAULT_RULES))

            def loss(params, x):
                y, stats = eqx.combine(params, static)(x, dropless=call_dropless)
                return jnp.sum(y.astype(jnp.float32) * cot) + stats.aux_loss, (y, stats)

            step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
            if not on_mesh:
                (_, (y, stats)), grads = step(params, x_host)
                return y, stats, grads
            with jax.set_mesh(mesh):
                (_, (y, stats)), grads = step(params, x)
            return y, stats, grads

        def close(b, a):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            if dtype == "float32":
                np.testing.assert_allclose(b, a, rtol=tol, atol=tol * max(1.0, float(np.max(np.abs(a)))))
            else:  # bfloat16 rounding differs with the GEMM accumulation order: compare relative norms
                rel = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
                assert rel < tol, rel
            return float(np.max(np.abs(a - b)))

        y0, s0, g0 = run(base, on_mesh=base_on_mesh)
        y1, s1, g1 = run(ep)
        close(y1, y0)
        for field in ("aux_loss", "balance_loss", "z_loss", "dropped_fraction"):
            np.testing.assert_allclose(float(getattr(s1, field)), float(getattr(s0, field)), rtol=1e-6, atol=1e-7)
        flat0, flat1 = jax.tree.leaves(g0), jax.tree.leaves(g1)
        assert len(flat0) == len(flat1) >= 4, (len(flat0), len(flat1))
        err = max(close(b, a) for a, b in zip(flat0, flat1, strict=True))
        report[name] = {"max_grad_err": err, "dropped": float(s1.dropped_fraction),
                        "max_out_err": float(np.max(np.abs(np.asarray(y1, np.float32) - np.asarray(y0, np.float32))))}
        return s1

    dropless = MoEConfig(num_experts=8, top_k=2, d_ff_expert=16, num_shared_experts=1, z_loss_coef=1e-3,
                         expert_parallel_axis="ep")
    layer_case("dropless_fsdp2_ep4", (1, 2, 1, 4), dropless)
    layer_case("dropless_dp2_fsdp2_ep2", (2, 2, 1, 2), dropless)
    layer_case("dropless_ep8_sigmoid_top3", (1, 1, 1, 8),
               dataclasses.replace(dropless, router="sigmoid", top_k=3, num_shared_experts=0))
    layer_case("dropless_tp2_ep2", (1, 2, 2, 2), dropless)
    layer_case("dropless_bf16", (1, 2, 1, 4), dropless, dtype="bfloat16", tol=1e-2)
    capacity = dataclasses.replace(dropless, capacity_factor=1.0)
    stats = layer_case("capacity_fsdp2_ep4", (1, 2, 1, 4), capacity)
    assert float(stats.dropped_fraction) > 0.0, "capacity case must drop routes to test the capacity rule"
    layer_case("capacity_dp2_fsdp2_ep2", (2, 2, 1, 2), dataclasses.replace(dropless, capacity_factor=0.75))
    layer_case("capacity_fsdp2_tp2_ep2", (1, 2, 2, 2), capacity)
    layer_case("capacity_dp4_ep2", (4, 1, 1, 2), capacity)
    layer_case("capacity_top1_cf0.5", (2, 1, 1, 4), dataclasses.replace(capacity, top_k=1, capacity_factor=0.5))
    # Decoding's dropless=True override of a capacity config, under an EP mesh.
    stats = layer_case("capacity_cfg_dropless_call", (1, 2, 1, 4), capacity, call_dropless=True)
    assert float(stats.dropped_fraction) == 0.0
    # Against one device without a mesh: EP, and mesh.ep > 1 without expert_parallel_axis (the
    # experts sharded over ep by the rules, the layer partitioned by GSPMD).
    layer_case("capacity_ep8_vs_single", (1, 1, 1, 8), capacity, base_on_mesh=False)
    layer_case("gspmd_fsdp2_ep4_dropless_vs_single", (1, 2, 1, 4), dataclasses.replace(dropless, expert_parallel_axis=None),
               base_on_mesh=False)
    layer_case("gspmd_fsdp2_ep4_capacity_vs_single", (1, 2, 1, 4), dataclasses.replace(capacity, expert_parallel_axis=None),
               base_on_mesh=False)
    # A buffer factor large enough for every block is exact.
    layer_case("buffer_factor_exact", (1, 2, 1, 4), dataclasses.replace(dropless, expert_parallel_buffer_factor=8.0))
    layer_case("capacity_buffer_factor_exact", (1, 2, 1, 4), dataclasses.replace(capacity, expert_parallel_buffer_factor=8.0))

    # Bounded blocks: each (device, peer) block keeps its routes slot-major (first choices, then second
    # choices, local token order within a choice, as capacity routing) and drops the rest.
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 2, 1, 4), ("dp", "fsdp", "tp", "ep"))
    cfg = dataclasses.replace(dropless, expert_parallel_buffer_factor=0.5)
    layer = MoE(32, cfg, key=jax.random.key(7), std=0.5, compute_dtype="float32")
    x = jax.random.normal(jax.random.key(1), (8, 16, 32), jnp.float32)
    with jax.set_mesh(mesh):
        y, stats = jax.jit(lambda layer, x: layer(x))(layer, x)
    flat = x.reshape(-1, 32)
    routing = layer._route(flat)
    experts = np.asarray(routing.experts).reshape(8, -1, 2)  # [devices, n, k]
    n_local = experts.shape[1]
    rows = block_rows(n_local, 2, 2, 4, 0.5)
    peers = experts // 2
    keep = np.zeros(experts.shape, bool)
    for d in range(8):
        for p in range(4):
            queue = [(t, c) for c in range(2) for t in range(n_local) if peers[d, t, c] == p]
            for t, c in queue[:rows]:
                keep[d, t, c] = True
    assert not keep.all() and keep[..., 0].sum() > keep[..., 1].sum()
    np.testing.assert_allclose(float(stats.dropped_fraction), 1.0 - keep.mean(), rtol=1e-6)
    # The output is the dense oracle with the dropped routes' gates set to zero.
    combine = np.zeros((flat.shape[0], 8), np.float32)
    np.put_along_axis(combine, np.asarray(routing.experts), np.asarray(routing.gates) * keep.reshape(-1, 2), axis=1)
    expert_out = jnp.einsum("nef,efd->ned", swiglu(jnp.einsum("nd,edgf->negf", flat, layer.gate_up)), layer.down)
    expected = np.asarray(jnp.einsum("ne,ned->nd", jnp.asarray(combine), expert_out) + layer.shared(flat))
    np.testing.assert_allclose(np.asarray(y).reshape(-1, 32), expected, rtol=1e-5,
                               atol=1e-5 * max(1.0, float(np.max(np.abs(expected)))))
    report["buffer_factor_0.5"] = {"dropped": float(stats.dropped_fraction), "rows": rows,
                                   "first_choices_dropped": int(np.sum(~keep[..., 0])),
                                   "second_choices_dropped": int(np.sum(~keep[..., 1]))}

    # Trainer: one device vs dp/fsdp x ep meshes, dropless and capacity routing.
    def leaves(tree):
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))]

    def run(name, model, devices, micro, train_kw=None, **mesh):
        train = {"max_steps": 3, "log_every": 1, "micro_batch_per_replica": micro, "grad_accum": 2,
                 "final_eval": False, "ce_compute_dtype": "float32", **(train_kw or {})}
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def same_run(single, result):
        for a, b in zip(single.history, result.history, strict=True):
            np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)
        for a, b in zip(leaves(single.model), leaves(result.model), strict=True):
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
        return max(abs(b["loss"] - a["loss"]) / abs(a["loss"]) for a, b in zip(single.history, result.history))

    base_model = get_preset("tiny-moe").replace(compute_dtype="float32")
    for label, moe in (("dropless", base_model.moe), ("capacity", dataclasses.replace(base_model.moe, capacity_factor=1.0))):
        model = base_model.replace(moe=moe)
        ep_model = base_model.replace(moe=dataclasses.replace(moe, expert_parallel_axis="ep"))
        single = run(f"{label}-single", model, jax.devices()[:1], 8, fsdp=1)
        for mesh_name, mesh in (("fsdp2_ep4", {"fsdp": 2, "ep": 4}), ("dp2_fsdp2_ep2", {"dp": 2, "fsdp": 2, "ep": 2})):
            result = run(f"{label}-{mesh_name}", ep_model, jax.devices(), 1, **mesh)
            for a, b in zip(single.history, result.history, strict=True):
                np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)
            diffs = []
            for a, b in zip(leaves(single.model), leaves(result.model), strict=True):
                np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
                diffs.append(float(np.max(np.abs(a - b))))
            report[f"trainer_{label}_{mesh_name}"] = {"max_param_err": max(diffs),
                                                      "loss": [h["loss"] for h in result.history]}
            expert_shape = (moe.num_experts, base_model.d_model, 2)  # routed gate_up [(L,) E, D, 2, F]
            specs = {str(leaf.sharding.spec) for leaf in jax.tree.leaves(eqx.filter(result.model, eqx.is_inexact_array))
                     if leaf.ndim >= 4 and leaf.shape[-4:-1] == expert_shape}
            report[f"trainer_{label}_{mesh_name}"]["gate_up_specs"] = sorted(specs)
            assert all("'ep'" in spec for spec in specs) and specs, specs

    # Evaluation and checkpoints inside an EP run (capacity routing), against one device.
    model = base_model.replace(moe=dataclasses.replace(base_model.moe, capacity_factor=1.0))
    ep_model = base_model.replace(moe=dataclasses.replace(model.moe, expert_parallel_axis="ep"))
    train_kw = {"max_steps": 4, "eval_every": 2, "final_eval": True, "save_every": 2}
    single = run("eval-single", model, jax.devices()[:1], 8, train_kw, fsdp=1)
    result = run("eval-fsdp2_ep4", ep_model, jax.devices(), 1, train_kw, fsdp=2, ep=4)
    loss_rel = same_run(single, result)
    assert len(result.eval_history) == len(single.eval_history) >= 2
    for a, b in zip(single.eval_history, result.eval_history, strict=True):
        for key, value in a.items():
            if isinstance(value, float):
                np.testing.assert_allclose(b[key], value, rtol=1e-5, err_msg=key)
    assert list((Path(out) / "eval-fsdp2_ep4" / "checkpoints").iterdir())
    report["trainer_capacity_eval_save_fsdp2_ep4"] = {"max_loss_rel": loss_rel, "eval": result.eval_history}
    # A dense model on a mesh with ep: ep only splits the batch.
    dense = get_preset("tiny").replace(compute_dtype="float32")
    report["trainer_dense_fsdp4_ep2"] = same_run(run("dense-single", dense, jax.devices()[:1], 8, fsdp=1),
                                                 run("dense-fsdp4_ep2", dense, jax.devices(), 1, fsdp=4, ep=2))
    print("EXPERTPARALLEL " + json.dumps(report, default=float))
    """
)


def test_expert_parallel_matches_single_device_on_8_cpu_devices(tmp_path: Path) -> None:
    script = tmp_path / "expert_parallel.py"
    script.write_text(SCRIPT)
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
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    report = json.loads(result.stdout.split("EXPERTPARALLEL ", 1)[1])
    assert report["capacity_fsdp2_ep4"]["dropped"] > 0
    assert report["buffer_factor_0.5"]["dropped"] > 0
    assert report["buffer_factor_0.5"]["second_choices_dropped"] > report["buffer_factor_0.5"]["first_choices_dropped"]
    print(json.dumps(report, indent=1))
