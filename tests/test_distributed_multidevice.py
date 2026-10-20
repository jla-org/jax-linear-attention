# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sharded training on 4 CPU devices matches one device; checkpoints move between layouts.

Runs in a subprocess because the CPU device count is fixed when JAX starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    r"""
    import json
    import sys

    import equinox as eqx
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from _toy_model import experiment
    from jla.distributed.sharding import replicated_fraction
    from jla.models import get_preset
    from jla.training.checkpointing import CheckpointManager
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 4, jax.devices()
    out = sys.argv[1]
    model = get_preset("tiny").replace(compute_dtype="float32")

    def run(name, devices, micro, fmt="auto", **mesh):
        train = {
            "max_steps": 4, "log_every": 2, "micro_batch_per_replica": micro, "grad_accum": 2,
            "final_eval": False, "checkpoint_format": fmt, "ce_compute_dtype": "float32",
        }
        # SGD + momentum: Adam's g / (|g| + eps) would amplify fp32 reduction-order
        # noise on near-zero gradients and hide real sharding bugs behind a loose tolerance.
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def leaves(tree):
        return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))]

    single = run("single", jax.devices()[:1], 8, fsdp=1)
    fsdp = run("fsdp4", jax.devices(), 2, fsdp=4)
    hybrid = run("dp2fsdp2", jax.devices(), 2, fmt="sharded", dp=2, fsdp=2)

    report = {}
    for name, result in (("fsdp4", fsdp), ("dp2fsdp2", hybrid)):
        diffs = [float(np.max(np.abs(a - b))) for a, b in zip(leaves(single.model), leaves(result.model), strict=True)]
        report[name] = max(diffs)
        for a, b in zip(leaves(single.model), leaves(result.model), strict=True):
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6)
        for a, b in zip(single.history, result.history, strict=True):
            np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)

    # Parameters and their momentum buffers are sharded over fsdp; batch rows over (dp, fsdp).
    embed = fsdp.model.embed.weight
    assert embed.sharding.spec == P(None, "fsdp"), embed.sharding.spec
    moments = [x for x in jax.tree.leaves(fsdp.opt_state) if getattr(x, "shape", None) == embed.shape]
    assert moments and all(m.sharding.spec == P(None, "fsdp") for m in moments)
    assert hybrid.model.embed.weight.sharding.spec == P(None, "fsdp")
    fraction, _ = replicated_fraction(eqx.filter(fsdp.model, eqx.is_inexact_array), embed.sharding.mesh)
    assert fraction < 0.05, fraction

    # A safetensors checkpoint of the fsdp=4 run restores into the dp=2 x fsdp=2 layout ...
    restored = CheckpointManager(f"{out}/fsdp4/checkpoints").restore(model=hybrid.model, opt_state=hybrid.opt_state)
    for a, b in zip(leaves(restored.model), leaves(fsdp.model), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(restored.opt_state), jax.tree.leaves(fsdp.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert restored.model.embed.weight.sharding == hybrid.model.embed.weight.sharding
    # ... and the per-process sharded checkpoint of the dp=2 x fsdp=2 run restores into fsdp=4.
    manifest = json.loads(open(f"{out}/dp2fsdp2/checkpoints/step_00000004/manifest.json").read())
    assert manifest["payload_format"] == "sharded"
    back = CheckpointManager(f"{out}/dp2fsdp2/checkpoints").restore(model=fsdp.model, opt_state=fsdp.opt_state)
    for a, b in zip(leaves(back.model), leaves(hybrid.model), strict=True):
        np.testing.assert_array_equal(a, b)
    assert back.model.embed.weight.sharding == fsdp.model.embed.weight.sharding
    print("MULTIDEVICE " + json.dumps(report))
    """
)


def test_four_device_sharded_training_and_checkpoint_resharding(tmp_path: Path) -> None:
    script = tmp_path / "multidevice.py"
    script.write_text(SCRIPT)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
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
    report = json.loads(result.stdout.split("MULTIDEVICE ", 1)[1])
    assert report["fsdp4"] < 1e-4 and report["dp2fsdp2"] < 1e-4
