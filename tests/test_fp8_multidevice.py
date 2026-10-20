# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP8 delayed scaling under sharding: 4 CPU devices (fsdp=4, dp=2 x fsdp=2) match one device.

The amax of a sharded tensor is a global maximum, so the scaling state must not depend on the
layout. Runs in a subprocess because the CPU device count is fixed when JAX starts.
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

    import jax
    import numpy as np

    from _toy_model import experiment
    from jla.layers.fp8 import fp8_modules
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 4, jax.devices()
    out = sys.argv[1]
    model = get_preset("tiny").replace(compute_dtype="float32", fp8="delayed", fp8_amax_history=8)

    def run(name, devices, micro, **mesh):
        train = {"max_steps": 3, "log_every": 1, "micro_batch_per_replica": micro, "grad_accum": 2,
                 "final_eval": False, "ce_compute_dtype": "float32"}
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0}
        config = experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)
        return Trainer(config, devices=devices).run()

    def states(result):
        return [np.asarray(x) for m in fp8_modules(result.model) for x in jax.tree.leaves(m.state)]

    single = run("single", jax.devices()[:1], 8, fsdp=1)
    report = {}
    for name, mesh in (("fsdp4", {"fsdp": 4}), ("dp2fsdp2", {"dp": 2, "fsdp": 2})):
        result = run(name, jax.devices(), 2, **mesh)
        # A different float32 reduction order can flip a value to the neighbouring FP8 number, so the
        # state agrees to 1e-2 (summing amaxes over devices or microbatches would be off by 2-4x).
        for a, b in zip(states(single), states(result), strict=True):
            np.testing.assert_allclose(b, a, rtol=1e-2, atol=0)
        for a, b in zip(single.history, result.history, strict=True):
            np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-3)
        report[name] = max(float(np.max(np.abs(a - b))) for a, b in zip(states(single), states(result)))
        assert all(np.count_nonzero(h) > 0 for h in states(result))
    print("FP8MULTI " + json.dumps(report))
    """
)


def test_fp8_state_is_independent_of_the_device_layout(tmp_path: Path) -> None:
    script = tmp_path / "fp8_multidevice.py"
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
    assert "FP8MULTI " in result.stdout
