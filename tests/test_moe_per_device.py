# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dropless MoE on the grouped-GEMM kernels under a multi-device mesh: each device dispatches its tokens.

Runs in a subprocess with 8 CPU devices (the device count is fixed when JAX starts). On CPU the
``"mosaic_gpu"`` grouped GEMM falls back to ``jax.lax.ragged_dot`` inside the per-device
``shard_map``; the test checks the split (the traced program holds a ``shard_map`` over the token
axes) and that outputs, router statistics and every gradient equal the one-device layer.
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
    import dataclasses
    import json
    import warnings

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.layers.moe import MoE
    from jla.models.config import MoEConfig

    warnings.simplefilter("ignore")  # the CPU fallback of the "mosaic_gpu" grouped GEMM warns
    assert len(jax.devices()) == 8
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 4, 1, 1), ("dp", "fsdp", "tp", "ep"))
    report = {}
    for name, cfg in {
        "dropless_mosaic": MoEConfig(num_experts=6, top_k=2, d_ff_expert=16, num_shared_experts=1,
                                     ragged_dot_impl="mosaic_gpu", z_loss_coef=1e-3),
        "dropless_xla": MoEConfig(num_experts=6, top_k=2, d_ff_expert=16, ragged_dot_impl="xla"),
    }.items():
        layer = MoE(32, cfg, key=jax.random.key(0), std=0.2, compute_dtype="float32")
        x = jax.random.normal(jax.random.key(1), (8, 12, 32))

        def loss(m, x):
            y, stats = m(x)
            return jnp.sum(jnp.sin(y)) + stats.aux_loss, (y, stats.balance_loss)

        step = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
        with jax.default_matmul_precision("highest"):
            (l0, (y0, b0)), g0 = step(layer, x)
            with jax.set_mesh(mesh):
                xs = jax.device_put(x, NamedSharding(mesh, P(("dp", "fsdp"))))
                (l1, (y1, b1)), g1 = step(layer, xs)
                jaxpr = str(jax.make_jaxpr(lambda m, x: m(x)[0])(layer, xs))
        diffs = [float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                 for a, b in zip(jax.tree.leaves(eqx.filter(g0, eqx.is_inexact_array)),
                                 jax.tree.leaves(eqx.filter(g1, eqx.is_inexact_array)), strict=True)]
        report[name] = dict(shard_map="shard_map" in jaxpr, y=float(np.max(np.abs(np.asarray(y0) - np.asarray(y1)))),
                            loss=abs(float(l0) - float(l1)), balance=abs(float(b0) - float(b1)), grad=max(diffs))
    print("PERDEVICE " + json.dumps(report))
    """
)


def test_dropless_kernel_path_splits_tokens_per_device_on_8_cpu_devices(tmp_path: Path) -> None:
    script = tmp_path / "per_device.py"
    script.write_text(SCRIPT)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600,
                            check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    report = json.loads(result.stdout.split("PERDEVICE ", 1)[1])
    assert report["dropless_mosaic"]["shard_map"], report
    assert not report["dropless_xla"]["shard_map"], report  # XLA partitions jax.lax.ragged_dot itself
    for name, row in report.items():
        assert row["y"] < 1e-5 and row["loss"] < 1e-4 and row["balance"] < 1e-6 and row["grad"] < 1e-4, (name, row)
