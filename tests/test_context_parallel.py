# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Context parallelism on 8 CPU devices matches the token-by-token reference.

Outputs, final states and gradients (of q, k, v, gates and the initial state) of
the sequence-sharded op are compared with ``jla.ops.reference`` on one device,
with packed documents whose boundaries fall inside shards and exactly on shard
boundaries. Runs in a subprocess because the CPU device count is fixed when JAX
starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    r"""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from jla.distributed.context_parallel import context_parallel_linear_attention
    from jla.layers.linear_attention import make_mixer
    from jla.models.config import LinearAttentionConfig
    from jla.ops import KINDS, reference

    assert len(jax.devices()) == 8, jax.devices()
    B, T, H, K, V = 2, 256, 2, 8, 8  # 32 tokens per shard on the 8-way axis, chunk 16
    DELTA = ("deltanet", "gated_deltanet", "kda")
    REF = {
        "simple_gla": reference.simple_gla, "gla": reference.gla, "deltanet": reference.delta_rule,
        "gated_deltanet": reference.gated_delta_rule, "kda": reference.kda,
    }
    mesh8 = Mesh(np.array(jax.devices()), ("cp",))
    mesh24 = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "cp"))

    def segments():
        # Row 0: changes inside shards and on the shard 1 -> 2 boundary (t = 64);
        # row 1: on the 0 -> 1 boundary (t = 32), right after it, and one document
        # spanning shards 1-4.
        starts = ([20, 64, 100, 200], [32, 33, 150])
        resets = np.zeros((B, T), np.int32)
        for row, ts in enumerate(starts):
            resets[row, ts] = 1
        return jnp.asarray(np.cumsum(resets, axis=1) + 7)

    def inputs(kind, seed):
        ks = jax.random.split(jax.random.key(seed), 8)
        q = jax.random.normal(ks[0], (B, T, H, K))
        k = jax.random.normal(ks[1], (B, T, H, K))
        v = jax.random.normal(ks[2], (B, T, H, V))
        gates = {}
        if kind in ("simple_gla", "gated_deltanet"):
            gates["g"] = -0.2 * jax.nn.softplus(jax.random.normal(ks[3], (B, T, H)))
        if kind in ("gla", "kda"):
            gates["g"] = -0.2 * jax.nn.softplus(jax.random.normal(ks[3], (B, T, H, K)))
        if kind in DELTA:
            gates["beta"] = jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H)))
        h0 = 0.5 * jax.random.normal(ks[5], (B, H, K, V))
        w_o = jax.random.normal(ks[6], (B, T, H, V))
        w_f = jax.random.normal(ks[7], (B, H, K, V))
        return (q, k, v, gates, h0), (w_o, w_f)

    def loss_fn(kind, impl, mesh, batch_axis, seg, use_h0, weights):
        opts = {"use_qk_l2norm": True} if kind in DELTA else {}

        def loss(q, k, v, gates, h0):
            state = h0 if use_h0 else None
            if impl == "reference":
                order = [gates[name] for name in KINDS[kind][1]]
                o, final = REF[kind](q, k, v, *order, initial_state=state, output_final_state=True,
                                     segment_ids=seg, **opts)
            else:
                o, final = context_parallel_linear_attention(
                    kind, q, k, v, mesh=mesh, axis="cp", batch_axis=batch_axis, initial_state=state,
                    output_final_state=True, segment_ids=seg, chunk_size=16, **gates, **opts)
            w_o, w_f = weights
            return jnp.sum(o * w_o) + jnp.sum(final * w_f), (o, final)

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))

    report = {}
    cases = [(kind, "cp8", True, True) for kind in REF] + [(kind, "cp8", False, False) for kind in REF]
    cases += [("gla", "dp2xcp4", True, True), ("gated_deltanet", "dp2xcp4", True, True)]
    for case, (kind, mesh_name, with_seg, use_h0) in enumerate(cases):
        mesh, batch_axis = (mesh8, None) if mesh_name == "cp8" else (mesh24, "dp")
        args, weights = inputs(kind, case)
        seg = segments() if with_seg else None
        (_, (o_ref, f_ref)), g_ref = loss_fn(kind, "reference", None, None, seg, use_h0, weights)(*args)
        (_, (o_cp, f_cp)), g_cp = loss_fn(kind, "cp", mesh, batch_axis, seg, use_h0, weights)(*args)
        errs = {"o": rel(o_cp, o_ref), "final": rel(f_cp, f_ref)}
        # Sensitivity: cutting the state at every shard boundary must change the result.
        shard_len = T // mesh.shape["cp"]
        cut = (0 if seg is None else seg) * 100 + jnp.arange(T)[None, :] // shard_len
        q, k, v, gates, h0 = args
        o_cut, _ = REF[kind](q, k, v, *[gates[n] for n in KINDS[kind][1]], initial_state=h0 if use_h0 else None,
                             segment_ids=jnp.broadcast_to(cut, (B, T)),
                             **({"use_qk_l2norm": True} if kind in DELTA else {}))
        assert rel(o_cp, o_cut) > 1e-3, (kind, mesh_name, rel(o_cp, o_cut))
        names = ["dq", "dk", "dv", "dgates", "dh0"]
        for name, a, b in zip(names, g_cp, g_ref):
            if name == "dh0" and not use_h0:
                continue
            errs[name] = max(rel(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
        report[f"{kind}/{mesh_name}/seg={with_seg}/h0={use_h0}"] = errs

    # The layer switch: a GatedDeltaNet mixer with context_parallel_axis under the
    # context mesh equals the same mixer running the token-by-token reference.
    def mixer(**kw):
        cfg = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=8, head_v_dim=8, chunk_size=16, **kw)
        return make_mixer("gated_deltanet", 32, cfg, key=jax.random.key(3), std=0.2, norm_eps=1e-6,
                          compute_dtype="float32")

    x = jax.random.normal(jax.random.key(4), (B, T, 32))
    seg = segments()

    def layer_loss(m, x):
        return jnp.sum(m(x, segment_ids=seg) ** 2)

    with jax.set_mesh(mesh8):
        cp_val, cp_dx = jax.jit(jax.value_and_grad(lambda x: layer_loss(mixer(context_parallel_axis="cp"), x)))(x)
    ref_val, ref_dx = jax.jit(jax.value_and_grad(lambda x: layer_loss(mixer(implementation="reference"), x)))(x)
    report["layer/gated_deltanet"] = {"loss": rel(cp_val, ref_val), "dx": rel(cp_dx, ref_dx)}

    # Without a mesh carrying the axis the switch fails loudly.
    try:
        layer_loss(mixer(context_parallel_axis="cp"), x)
    except ValueError as err:
        assert "context parallelism needs a mesh" in str(err), err
    else:
        raise AssertionError("context_parallel_axis without a mesh must raise")
    print("CONTEXT_PARALLEL " + json.dumps(report))
    """
)


def _run(tmp_path: Path) -> dict:
    script = tmp_path / "context_parallel.py"
    script.write_text(SCRIPT)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900, check=False
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    return json.loads(result.stdout.split("CONTEXT_PARALLEL ", 1)[1])


def test_context_parallel_matches_reference(tmp_path: Path) -> None:
    report = _run(tmp_path)
    assert len(report) == 13, sorted(report)
    bad = {case: errs for case, errs in report.items() if max(errs.values()) > 2e-4}
    assert not bad, json.dumps(bad, indent=1)
    print(json.dumps(report, indent=1))


def test_context_parallel_rejects_unsupported() -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from jla.distributed.context_parallel import context_parallel_linear_attention

    mesh = Mesh(np.array(jax.devices()[:1]), ("cp",))
    x = jnp.zeros((1, 32, 2, 8))
    with pytest.raises(NotImplementedError, match="supports"):
        context_parallel_linear_attention("gdn2", x, x, x, mesh=mesh, g=x, b=x, w=x)
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        context_parallel_linear_attention("gla", x, x, x, mesh=mesh, g=x, cu_seqlens=jnp.array([0, 32]))
    with pytest.raises(ValueError, match="missing"):
        context_parallel_linear_attention("gated_deltanet", x, x, x, mesh=mesh, g=x[..., 0])
    with pytest.raises(ValueError, match="needs a mesh"):
        context_parallel_linear_attention("gla", x, x, x, mesh=mesh, axis="sp", g=x)
