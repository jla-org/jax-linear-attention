# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FSDP keeps activations split along the batch.

``jla.distributed.activation_sharding`` pins the residual stream and the LM-head logits to the data
axes: without it GSPMD carried the weights' ``embed`` split into the activations and all-reduced
full-batch activations after every projection (the XLA ops; the Mosaic GPU kernels' partitioning
rules anchor the batch split on their own). Runs in a subprocess with 8 CPU devices (the device count
is fixed when JAX starts).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    r"""
    import json
    import re
    import sys

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed.activation_sharding import constrain_batch
    from jla.distributed.kernel_sharding import use_rules
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.distributed.pipeline import pipeline_rules
    from jla.distributed.sharding import batch_axes, grouped_batch_sharding, opt_state_shardings, resolve_rules
    from jla.ops.cross_entropy import linear_cross_entropy
    from jla.training.optimizers import optimizer_from_config
    from jla.training.step import LossConfig, build_train_step, lm_loss
    from jla.training.trainer import Trainer, _replicated_key

    sys.path.insert(0, sys.argv[2])
    from _toy_model import experiment

    out = {}
    x = jnp.ones((16, 8, 32))
    out["no_mesh_identity"] = constrain_batch(x) is x
    rules = resolve_rules()
    mesh = build_mesh(resolve_mesh_plan(8, fsdp=8), jax.devices(), topology_aware=False)
    with jax.set_mesh(mesh), use_rules(rules):
        out["spec"] = str(jax.jit(constrain_batch)(x).sharding.spec)
        out["odd_batch_spec"] = str(jax.jit(constrain_batch)(jnp.ones((12, 8))).sharding.spec)

    # Pipeline (fsdp 4 x pp 2): outside the stages the head's rows are split over (fsdp, pp), as
    # pipeline_lm_forward leaves them; the CE's tiles must keep that split (1024 rows / 8 devices = 128).
    pp_mesh = build_mesh(resolve_mesh_plan(8, fsdp=4, pp=2), jax.devices(), topology_aware=False)
    pp_rules = pipeline_rules(rules)
    with jax.set_mesh(pp_mesh), use_rules(pp_rules):
        out["pp_identity"] = constrain_batch(x) is x
    rows = NamedSharding(pp_mesh, P((*batch_axes(pp_mesh, pp_rules), "pp")))
    hidden = jax.device_put(jnp.ones((1024, 64), jnp.float32), rows)
    head = jax.device_put(jnp.ones((1024, 64), jnp.float32), NamedSharding(pp_mesh, P(None, "fsdp")))
    labels = jax.device_put(jnp.zeros((1024,), jnp.int32), rows)
    ce = jax.jit(jax.value_and_grad(lambda h, w, t: linear_cross_entropy(h, w, t, vocab_chunk=256).loss_sum,
                                    argnums=(0, 1)))
    with jax.set_mesh(pp_mesh), use_rules(pp_rules):
        text = ce.lower(hidden, head, labels).compile().as_text()
    out["pp_ce_tile_rows"] = sorted(set(re.findall(r"f32\[(\d+),256\]", text)))

    def step_hlo(preset, batch=16, seq=48):
        config = experiment("unused", model=preset, train={"micro_batch_per_replica": batch // 8, "grad_accum": 1,
                                                          "seq_len": seq},
                            mesh={"fsdp": 8})
        trainer = Trainer(config, devices=jax.devices())
        model = trainer._init_model(mesh, rules, jax.random.key(0))
        params = eqx.filter(model, eqx.is_inexact_array)
        optimizer = optimizer_from_config(config.optim, 1e-3)
        shardings = jax.tree.map(lambda leaf: leaf.sharding, params)
        opt_sh = opt_state_shardings(jax.eval_shape(optimizer.init, params), params, shardings, mesh)
        opt_state = jax.jit(optimizer.init, out_shardings=opt_sh)(params)
        batch_sharding = grouped_batch_sharding(mesh, rules)
        step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(config.train)),
                                batch_sharding=batch_sharding, param_shardings=shardings, opt_shardings=opt_sh)
        rng = np.random.default_rng(0)
        host = {k: rng.integers(0, 500, (1, batch, seq)).astype(np.int32) for k in ("input_ids", "labels")}
        data = {k: jax.device_put(v, batch_sharding) for k, v in host.items()}
        key = _replicated_key(jax.random.key(1), mesh)
        with jax.set_mesh(mesh), use_rules(rules):
            compiled = step.lower(model, opt_state, data, key).compile()
            _, _, metrics, _ = step(model, opt_state, data, key)
        text = getattr(compiled, "compiled", compiled).as_text()
        return text, float(metrics["loss"])

    for preset in ("tiny", "tiny-hybrid", "tiny-mamba2"):
        text, loss = step_hlo(preset)
        collectives = []
        for line in text.splitlines():
            op = next((name for name in ("all-gather", "all-reduce", "reduce-scatter", "all-to-all")
                       if f" {name}(" in line or f" {name}-start(" in line), None)
            if op is not None:
                shape = line.split(" = ", 1)[1].split(f" {op}", 1)[0]
                collectives.append([op, shape])
        out[preset] = {"loss": loss, "collectives": collectives}
    print(json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def report() -> dict:
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": str(ROOT / "src"),
    }
    proc = subprocess.run([sys.executable, "-c", SCRIPT, "unused", str(ROOT / "tests")], capture_output=True,
                          text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_constrain_batch_pins_the_data_axes(report):
    assert report["no_mesh_identity"]
    assert "'fsdp'" in report["spec"].split(",")[0], report["spec"]
    assert "fsdp" not in report["odd_batch_spec"], report["odd_batch_spec"]  # 12 rows do not split over 8: unchanged


def test_pipeline_head_keeps_its_stage_sharding(report):
    # A batch-only constraint here gave [256, 256] tiles: every pp stage computed the head for all rows of its
    # fsdp shard (2x the LM-head FLOPs and logits memory per device).
    assert report["pp_identity"]
    assert report["pp_ce_tile_rows"] == ["128"], report["pp_ce_tile_rows"]


@pytest.mark.parametrize("preset", ["tiny", "tiny-hybrid", "tiny-mamba2"])
def test_no_collective_carries_the_global_batch(report, preset):
    # Global batch 16 x 48 tokens over 8 devices: per device 2 rows. A float collective of 16 rows x 48 tokens x
    # features (or 768 flattened tokens x features) moves the activations of the whole batch (the pre-fix FSDP
    # pathology). Per-token scalars ([16, 48], a few KiB) may still be reduced whole. The
    # embedding lookup all-gathers the int32 token ids, which is cheap.
    assert report[preset]["loss"] > 0
    for op, shape in report[preset]["collectives"]:
        if "f32" in shape or "bf16" in shape:
            assert not re.search(r"\[16,48,\d|\[768,\d", shape), (preset, op, shape)


def test_pipeline_axis_matches_the_pipeline_module():
    # activation_sharding keeps its own copy of the stage axis name (importing jla.distributed.pipeline there
    # would be a cycle); constrain_batch's no-op between pipeline stages keys on it.
    from jla.distributed import activation_sharding, pipeline

    assert activation_sharding._PIPELINE_AXIS == pipeline.PIPELINE_AXIS
