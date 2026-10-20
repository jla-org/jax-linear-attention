# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``train.grad_reduce = "step"`` (jla.training.grad_reduce): the same training as the per-micro-step reduction.

On 4 CPU devices (subprocesses: the CPU device count is fixed when JAX starts) the trainer with
``grad_reduce = "step"`` gives the same losses, gradient norms, parameters and FP8 states as the plain step at
FSDP 4 and at dp 2 x FSDP 2, in float32 and with FP8 delayed scaling; its compiled step has no collective inside the
micro-step loop (the plain step reduce-scatters inside it). The model checks refuse what the per-device loop would
not compute exactly. The 2-GPU versions with the Mosaic GPU kernels are in ``tests/test_grad_reduce_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jla.models import get_preset
from jla.training.grad_reduce import check_grad_reduce

ROOT = Path(__file__).resolve().parents[1]

TRAINER_SCRIPT = textwrap.dedent(
    r"""
    import json
    import sys

    import equinox as eqx
    import jax
    import numpy as np

    from _toy_model import experiment
    from jla.layers.fp8 import fp8_modules
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 4, jax.devices()
    out = sys.argv[1]

    def run(name, model, reduce, devices, micro, extra=None, **mesh):
        train = {"max_steps": 3, "log_every": 1, "micro_batch_per_replica": micro, "grad_accum": 2,
                 "final_eval": False, "ce_compute_dtype": "float32", "grad_reduce": reduce, **(extra or {})}
        # SGD + momentum: Adam's g / (|g| + eps) would amplify fp32 reduction-order noise on near-zero gradients.
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def leaves(result):
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(result.model, eqx.is_inexact_array))]

    def states(result):
        return [np.asarray(x) for m in fp8_modules(result.model) for x in jax.tree.leaves(m.state)]

    report = {}
    f32 = get_preset("tiny").replace(compute_dtype="float32")
    fp8 = f32.replace(fp8="delayed", fp8_amax_history=8)
    for label, model in (("f32", f32), ("fp8", fp8)):
        single = run(f"{label}-single", model, "micro_step", jax.devices()[:1], 8, fsdp=1)
        for mesh_name, mesh in (("fsdp4", {"fsdp": 4}), ("dp2fsdp2", {"dp": 2, "fsdp": 2})):
            plain = run(f"{label}-{mesh_name}-micro", model, "micro_step", jax.devices(), 2, **mesh)
            once = run(f"{label}-{mesh_name}-step", model, "step", jax.devices(), 2, **mesh)
            # float32: summation-order noise only. FP8: another summation order can move a value to the
            # neighbouring FP8 number (as between the plain step's layouts, tests/test_fp8_multidevice.py).
            loss_tol, norm_tol, param_tol = (1e-5, 1e-4, (1e-4, 1e-6)) if label == "f32" else (1e-3, 1e-2, (1e-2, 1e-4))
            for ref in (plain, single):
                for a, b in zip(ref.history, once.history, strict=True):
                    np.testing.assert_allclose(b["loss"], a["loss"], rtol=loss_tol)
                    np.testing.assert_allclose(b["grad_norm"], a["grad_norm"], rtol=norm_tol)
                for a, b in zip(leaves(ref), leaves(once), strict=True):
                    np.testing.assert_allclose(b, a, rtol=param_tol[0], atol=param_tol[1])
            if label == "fp8":
                # A different float32 summation order can move a value to the neighbouring FP8 number; summing
                # the amaxes over the devices instead of taking their maximum would be off by 2-4x.
                for a, b in zip(states(plain), states(once), strict=True):
                    np.testing.assert_allclose(b, a, rtol=1e-2, atol=0)
                assert all(np.count_nonzero(h) > 0 for h in states(once))
            # The reduced gradients keep the parameters' FSDP layout.
            for a, b in zip(jax.tree.leaves(eqx.filter(plain.model, eqx.is_inexact_array)),
                            jax.tree.leaves(eqx.filter(once.model, eqx.is_inexact_array)), strict=True):
                assert a.sharding.spec == b.sharding.spec, (a.sharding.spec, b.sharding.spec)
            report[f"{label}-{mesh_name}"] = max(float(np.max(np.abs(a - b)))
                                                 for a, b in zip(leaves(plain), leaves(once)))
            if label == "f32" and mesh_name == "fsdp4":
                # With the other two options of the 8-GPU runs: the rejected-step rule as a select and the next step
                # dispatched before a step's metrics are read. The same training.
                combined = run(f"{label}-{mesh_name}-step-select-ahead", model, "step", jax.devices(), 2,
                               extra={"update_guard": "select", "dispatch_ahead": True}, **mesh)
                for a, b in zip(plain.history, combined.history, strict=True):
                    np.testing.assert_allclose(b["loss"], a["loss"], rtol=loss_tol)
                for a, b in zip(leaves(plain), leaves(combined), strict=True):
                    np.testing.assert_allclose(b, a, rtol=param_tol[0], atol=param_tol[1])
    print("GRADREDUCE " + json.dumps(report))
    """
)

HLO_SCRIPT = textwrap.dedent(
    r"""
    import json
    import re

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed.mesh import build_mesh, plan_from_config
    from jla.config import MeshConfig
    from jla.distributed.sharding import grouped_batch_sharding, param_shardings, resolve_rules
    from jla.models import build_model, get_preset, logical_axes
    from jla.training.step import LossConfig, build_train_step, lm_loss

    assert len(jax.devices()) == 4, jax.devices()
    mesh = build_mesh(plan_from_config(MeshConfig(dp=1, fsdp=4, tp=1), 4), jax.devices())
    rules = resolve_rules()
    config = get_preset("tiny").replace(compute_dtype="float32", fp8="delayed", fp8_amax_history=4)
    model = build_model(config, key=jax.random.key(0))
    params = eqx.filter(model, eqx.is_inexact_array)
    shardings = param_shardings(params, logical_axes(model), mesh, rules)
    model = eqx.combine(jax.device_put(params, shardings), model)
    batch_sharding = grouped_batch_sharding(mesh, rules)
    grad_accum, rows, length = 4, 8, 16
    tokens = jax.random.randint(jax.random.key(1), (grad_accum, rows, length + 1), 0, config.vocab_size)
    batch = {"input_ids": tokens[..., :-1], "labels": tokens[..., 1:]}
    batch = jax.device_put(batch, batch_sharding)
    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)
    loss = lm_loss(LossConfig(compute_dtype="float32"))

    COLLECTIVE = re.compile(r"\b(all-reduce|reduce-scatter|all-gather|all-to-all|collective-permute)(-start)?\(")

    def computations(text):
        blocks, name, lines = {}, None, []
        for line in text.splitlines():
            head = re.match(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", line)
            if head and not line.startswith(" "):
                name, lines = head.group(1), []
            elif line.startswith("}") and name is not None:
                blocks[name] = "\n".join(lines)
                name = None
            elif name is not None:
                lines.append(line)
        return blocks

    def loop_collectives(text):
        blocks = computations(text)
        called = lambda body: set(re.findall(r"(?:body|condition|calls|to_apply|branch_computations)=\{?%?([\w.\-]+)", body))
        todo = [m for b in blocks.values() for m in re.findall(r"\bwhile\(.*?body=%?([\w.\-]+)", b)]
        seen = set()
        while todo:
            name = todo.pop()
            if name in seen or name not in blocks:
                continue
            seen.add(name)
            todo.extend(called(blocks[name]))
        found = [m.group(0) for name in seen for m in COLLECTIVE.finditer(blocks[name])]
        whole = [m.group(0) for m in COLLECTIVE.finditer(text)]
        return found, whole, len(seen)

    report = {}
    with jax.set_mesh(mesh):
        for reduce in ("micro_step", "step"):
            step = build_train_step(optimizer, loss, batch_sharding=batch_sharding, param_shardings=shardings,
                                    scan_unroll=1, grad_reduce=reduce)
            compiled = step.lower(model, opt_state, batch, jax.random.key(2)).compile()
            text = getattr(compiled, "compiled", compiled).as_text()  # equinox wraps the jax Compiled
            inside, whole, bodies = loop_collectives(text)
            report[reduce] = {"inside_loop": len(inside), "whole_step": len(whole), "loop_computations": bodies,
                              "inside_ops": sorted(set(inside))}
    print("HLO " + json.dumps(report))
    assert report["step"]["loop_computations"] > 0, report
    assert report["step"]["inside_loop"] == 0, report
    assert report["micro_step"]["inside_loop"] > 0, report

    # Tensor parallelism: an axis that does not split the batch is refused.
    from jla.training.grad_reduce import reduce_axes
    tp_mesh = build_mesh(plan_from_config(MeshConfig(dp=1, fsdp=2, tp=2), 4), jax.devices())
    try:
        reduce_axes(grouped_batch_sharding(tp_mesh, rules))
    except ValueError as err:
        assert "tp" in str(err), err
    else:
        raise AssertionError("grad_reduce='step' accepted a tensor-parallel mesh")
    assert reduce_axes(batch_sharding) == ("fsdp",)
    """
)


def _run(tmp_path: Path, name: str, script: str, marker: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(script)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run([sys.executable, str(path), str(tmp_path / "runs")], env=env, capture_output=True,
                            text=True, timeout=1200, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    assert marker in result.stdout
    return result.stdout


def test_step_reduction_trains_like_the_micro_step_reduction(tmp_path: Path) -> None:
    _run(tmp_path, "grad_reduce_trainer", TRAINER_SCRIPT, "GRADREDUCE ")


def test_step_reduction_has_no_collective_inside_the_micro_step_loop(tmp_path: Path) -> None:
    _run(tmp_path, "grad_reduce_hlo", HLO_SCRIPT, "HLO ")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: get_preset("tiny-moe"), "MoE"),
        (lambda c: get_preset("tiny-hybrid"), "MTP"),
        (lambda c: c.replace(resid_dropout=0.1), "dropout"),
        (lambda c: c.replace(fp8="current"), "current"),
        (lambda c: get_preset("tiny-mamba1"), "Mamba-1"),
        (lambda c: get_preset("tiny-dsa"), "DSA indexer loss"),
    ],
)
def test_check_refuses_models_the_per_device_loop_would_change(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        check_grad_reduce(change(get_preset("tiny")), platform="cpu")


def test_check_refuses_cudnn_attention_on_gpu_only() -> None:
    config = get_preset("tiny")
    for impl in ("cudnn", "auto"):
        cudnn = config.replace(attention=dataclasses.replace(config.attention, implementation=impl))
        check_grad_reduce(cudnn, platform="cpu")
        with pytest.raises(ValueError, match="cuDNN"):
            check_grad_reduce(cudnn, platform="gpu")
    kernel = config.replace(attention=dataclasses.replace(config.attention, implementation="mosaic_gpu"), fp8="delayed")
    check_grad_reduce(kernel, platform="gpu")


def test_check_accepts_dsa_without_its_indexer_loss() -> None:
    dsa = get_preset("tiny-dsa")
    assert float(dsa.sparse_attention.indexer_loss_coef) > 0  # the refused case above
    check_grad_reduce(dsa.replace(sparse_attention=dataclasses.replace(dsa.sparse_attention, indexer_loss_coef=0.0)),
                      platform="cpu")


def test_step_reduction_warns_above_half_of_the_device_memory(monkeypatch, caplog) -> None:
    """``"step"`` keeps about 10 bytes per parameter on every device: a warning when that exceeds half of the
    device's memory limit, none below it or when the device reports no limit."""
    import logging

    import jax
    import numpy as np

    from jla.training import grad_reduce

    class Device:
        def __init__(self, limit):
            self.limit = limit

        def memory_stats(self):
            return None if self.limit is None else {"bytes_limit": self.limit}

    for limit, params, warned in ((1000, (10, 10), True), (1000, (4, 10), False), (None, (10, 10), False)):
        monkeypatch.setattr(jax, "local_devices", lambda limit=limit: [Device(limit)])
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="jla.training.grad_reduce"):
            grad_reduce._warn_memory([np.zeros(params, np.float32)])
        assert ("10 bytes per parameter" in caplog.text) == warned, (limit, params, caplog.text)
