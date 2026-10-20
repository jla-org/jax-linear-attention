# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer's per-device dispatch and expert parallelism on two H100s, on the Mosaic GPU kernels.

``tests/test_moe_per_device.py`` and ``tests/test_expert_parallel.py`` check these layouts on 8 CPU
devices, where the ``"mosaic_gpu"`` grouped GEMM falls back to ``jax.lax.ragged_dot``. This file runs
the same comparisons in a subprocess that sees two GPUs, so the Mosaic GPU kernels run inside the
``shard_map``s (a fallback warning fails the test): bfloat16 layers with softmax routing and a gated
shared expert (Qwen3-Next-like) and sigmoid routing with a shared expert (Kimi-Linear-like) against the
same layer on one device without a mesh (outputs, router statistics and every gradient), and expert
parallelism over the two GPUs with dropless and bounded (``expert_parallel_buffer_factor = 2``) blocks.

The two GPUs are ``JLA_TEST_GPUS`` (e.g. ``"3,5"``) or, by default, this process's GPU and the other
GPU with the least memory in use (``nvidia-smi``). Skipped without two H100s.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _two_gpus() -> str | None:
    """Two GPU indices for the subprocess, or None when this host has no two H100s."""
    if jax.default_backend() != "gpu" or "H100" not in jax.devices()[0].device_kind:
        return None
    if os.environ.get("JLA_TEST_GPUS"):
        return os.environ["JLA_TEST_GPUS"]
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,name", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60, check=False).stdout
    rows = [r.split(",") for r in out.strip().splitlines() if "H100" in r]
    used = {r[0].strip(): float(r[1]) for r in rows}
    if len(used) < 2:
        return None
    mine = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    first = mine if mine in used else min(used, key=used.get)
    second = min((g for g in used if g != first), key=used.get)
    return f"{first},{second}"


SCRIPT = textwrap.dedent(
    r"""
    import dataclasses
    import json
    import warnings

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from jla.distributed.sharding import DEFAULT_RULES, param_shardings
    from jla.layers.moe import MoE
    from jla.models import logical_axes
    from jla.models.config import MoEConfig

    devs = jax.devices()
    assert len(devs) == 2 and all("H100" in d.device_kind for d in devs), devs
    NAMES = ("dp", "fsdp", "tp", "ep")
    report = {}

    def rel(b, a):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return float(np.linalg.norm(b - a) / max(float(np.linalg.norm(a)), 1e-30))

    def run(layer, x_host, cot, mesh):
        params, static = eqx.partition(layer, eqx.is_array)

        def loss(params, x):
            y, stats = eqx.combine(params, static)(x)
            return jnp.sum(y.astype(jnp.float32) * cot) + stats.aux_loss, (y, stats)

        step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
        if mesh is None:
            (_, (y, stats)), grads = step(params, x_host)
            return y, stats, grads, ""
        params = jax.device_put(params, param_shardings(params, logical_axes(layer), mesh, DEFAULT_RULES))
        x = jax.device_put(x_host, NamedSharding(mesh, P(("dp", "fsdp", "ep"))))
        with jax.set_mesh(mesh):
            (_, (y, stats)), grads = step(params, x)
            jaxpr = str(jax.make_jaxpr(lambda p, x: eqx.combine(p, static)(x)[0])(params, x))
        return y, stats, grads, jaxpr

    def case(name, mesh_shape, cfg, *, base=None, d_model=256, batch=4, seq=512, tol=1e-2):
        # base None: the same layer without expert parallelism on one device, no mesh.
        mesh = Mesh(np.asarray(devs).reshape(mesh_shape), NAMES)
        key = jax.random.key(7)
        x_host = jax.random.normal(jax.random.key(1), (batch, seq, d_model), jnp.float32).astype(jnp.bfloat16)
        cot = jax.random.normal(jax.random.key(2), (batch, seq, d_model), jnp.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if base is None:
                ref_cfg = dataclasses.replace(cfg, expert_parallel_axis=None, expert_parallel_buffer_factor=None)
                y0, s0, g0, _ = run(MoE(d_model, ref_cfg, key=key, std=0.05, compute_dtype="bfloat16"), x_host, cot, None)
            else:
                y0, s0, g0, _ = run(MoE(d_model, base, key=key, std=0.05, compute_dtype="bfloat16"), x_host, cot, mesh)
            y1, s1, g1, jaxpr = run(MoE(d_model, cfg, key=key, std=0.05, compute_dtype="bfloat16"), x_host, cot, mesh)
        fallbacks = [str(w.message) for w in caught if "falling back" in str(w.message)]
        assert not fallbacks, fallbacks
        row = {"shard_map": "shard_map" in jaxpr, "y_rel": rel(y1, y0),
               "grad_rel": max(rel(b, a) for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1), strict=True)),
               "n_grads": len(jax.tree.leaves(g1))}
        for field in ("aux_loss", "balance_loss", "z_loss", "dropped_fraction"):
            a, b = float(getattr(s0, field)), float(getattr(s1, field))
            row[field] = [a, b]
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6, err_msg=f"{name} {field}")
        assert row["y_rel"] < tol and row["grad_rel"] < 2 * tol, (name, row)
        report[name] = row
        return row

    qwen_like = MoEConfig(num_experts=16, top_k=4, d_ff_expert=128, num_shared_experts=1, shared_expert_gate=True,
                          z_loss_coef=1e-3, ragged_dot_impl="mosaic_gpu")
    kimi_like = MoEConfig(num_experts=16, top_k=4, d_ff_expert=128, num_shared_experts=1, router="sigmoid",
                          routed_scaling_factor=2.446, ragged_dot_impl="mosaic_gpu")
    # Per-device dispatch (FSDP 2): each GPU routes its own tokens through the Mosaic GPU kernels.
    for label, cfg in (("qwen_like", qwen_like), ("kimi_like", kimi_like)):
        row = case(f"per_device_fsdp2_{label}", (1, 2, 1, 1), cfg)
        assert row["shard_map"], row
    # Expert parallelism over the two GPUs: dropless (worst-case blocks) against one device, and
    # bounded blocks of twice the expected routes against the XLA grouped GEMM on the same mesh.
    for label, cfg in (("qwen_like", qwen_like), ("kimi_like", kimi_like)):
        ep = dataclasses.replace(cfg, expert_parallel_axis="ep")
        case(f"ep2_dropless_{label}", (1, 1, 1, 2), ep)
        bounded = dataclasses.replace(ep, expert_parallel_buffer_factor=2.0)
        case(f"ep2_bounded2_{label}_vs_xla", (1, 1, 1, 2), bounded,
             base=dataclasses.replace(bounded, ragged_dot_impl="xla"))
    print("TWOGPUS " + json.dumps(report))
    """
)


def test_moe_per_device_and_expert_parallel_on_two_gpus(tmp_path: Path) -> None:
    gpus = _two_gpus()
    if gpus is None:
        pytest.skip("needs two H100s")
    script = tmp_path / "two_gpus.py"
    script.write_text(SCRIPT)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "PYTHONPATH": str(ROOT / "src")}
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
    env.pop("JAX_PLATFORMS", None)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900,
                            check=False)
    assert result.returncode == 0, f"GPUs {gpus}\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    report = json.loads(result.stdout.split("TWOGPUS ", 1)[1].splitlines()[0])  # NCCL may print after it
    print(f"GPUs {gpus}: " + json.dumps(report, indent=1))
    assert len(report) == 6, report
