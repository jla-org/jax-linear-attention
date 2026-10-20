# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Context-parallel softmax attention on 8 CPU devices matches single-device attention.

The cases (ring and Ulysses ops, the ``Attention`` layer switch, a hybrid Gated DeltaNet
+ attention train step) live in ``tests/_context_parallel_attention_checks.py``, which also
runs unchanged on 8 GPUs. They run in a subprocess because the CPU device count is fixed
when JAX starts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_context_parallel_attention_matches_single_device() -> None:
    from _context_parallel_attention_checks import NUM_F32_CASES, failures

    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "_context_parallel_attention_checks.py")],
        env=env, capture_output=True, text=True, timeout=1500, check=False,
    )
    assert "CP_ATTENTION " in result.stdout, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    report = json.loads(result.stdout.split("CP_ATTENTION ", 1)[1].splitlines()[0])
    print(json.dumps(report, indent=1))
    assert len(report["f32"]) == NUM_F32_CASES, sorted(report["f32"])
    bad = failures(report)
    assert not bad, json.dumps(bad, indent=1)
    assert result.returncode == 0, result.stderr[-4000:]


def test_context_parallel_attention_rejects_unsupported() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from jla.distributed.context_parallel_attention import context_parallel_attention
    from jla.models import build_model, get_preset
    from jla.models.config import AttentionConfig

    mesh = Mesh(np.array(jax.devices()[:1]), ("cp",))
    q = jnp.zeros((1, 32, 4, 8))
    kv = jnp.zeros((1, 32, 2, 8))
    with pytest.raises(ValueError, match="strategy"):
        context_parallel_attention(q, kv, kv, mesh=mesh, strategy="tree")
    with pytest.raises(ValueError, match="needs a mesh"):
        context_parallel_attention(q, kv, kv, mesh=mesh, axis="sp")
    with pytest.raises(ValueError, match="sinks"):
        context_parallel_attention(q, kv, kv, mesh=mesh, sinks=jnp.zeros((3,)))
    with pytest.raises(ValueError, match="implementation"):
        context_parallel_attention(q, kv, kv, mesh=mesh, implementation="cudnn")
    with pytest.raises(ValueError, match="block_q"):
        context_parallel_attention(q, kv, kv, mesh=mesh, block_q=5)
    with pytest.raises(ValueError, match="block_q"):
        context_parallel_attention(q, kv, kv, mesh=mesh, strategy="ulysses", block_q=8)
    with pytest.raises(ValueError, match="segment_ids"):
        context_parallel_attention(q, kv, kv, mesh=mesh, segment_ids=jnp.zeros((1, 16), jnp.int32))
    with pytest.raises(ValueError, match="moba|MoBA"):
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8, moba_block_size=16, context_parallel_axis="cp")
    with pytest.raises(ValueError, match="ulysses"):
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8, implementation="cudnn", context_parallel_axis="cp")
    AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8, sinks=True, context_parallel_axis="cp")  # ring sinks: ok
    with pytest.raises(ValueError, match="context_parallel_strategy"):
        AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=8, context_parallel_strategy="tree")
    # Sliding-window layers have their own config: CP on only one of the two fails loudly.
    cfg = get_preset("tiny-gemma-3")
    with pytest.raises(ValueError, match="local_attention"):
        cfg.replace(attention=dataclasses.replace(cfg.attention, context_parallel_axis="cp"))
    cfg.replace(attention=dataclasses.replace(cfg.attention, context_parallel_axis="cp"),
                local_attention=dataclasses.replace(cfg.local_attention, context_parallel_axis="cp"))
    # Mixers that read the attention config but do not shard T fail loudly.
    for name in ("tiny-nsa", "tiny-dsa"):
        cfg = get_preset(name)
        cfg = cfg.replace(attention=dataclasses.replace(cfg.attention, context_parallel_axis="cp"))
        with pytest.raises(NotImplementedError, match="context parallelism"):
            build_model(cfg, key=jax.random.key(0))
