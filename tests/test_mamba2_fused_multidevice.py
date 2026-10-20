# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 fused glue on several devices: the row kernels run per shard of the context mesh.

``jla.layers.mamba2_fused`` runs its two Pallas row kernels (gated RMSNorm forward and backward) inside
``shard_layout_call`` when a multi-device call has a context mesh that can split the rows (the trainer's FSDP /
data-parallel mesh), and its plain-JAX formulas when no split divides the rows. On 4 CPU devices (the kernels in
the Pallas interpreter) every mesh layout gives the single-device values and gradients of ``gated_norm_out`` and
``gated_norm``; the per-shard call is traced with each device's rows. On several GPUs the fused Mamba-2 layer
under an FSDP mesh matches the unfused layer (run with two or more visible GPUs; skipped otherwise).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]

CPU_SCRIPT = textwrap.dedent(
    r"""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed import kernel_sharding
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.layers import mamba2_fused as fused

    assert len(jax.devices()) == 4, jax.devices()
    jax.config.update("jax_default_matmul_precision", "highest")
    calls, rows, fallbacks = [], [], []
    real_call, real_rows = kernel_sharding.shard_layout_call, fused._rows
    real_fwd_xla, real_bwd_xla = fused._norm_forward_xla, fused._norm_backward_xla

    def spy_call(*args, **kwargs):
        calls.append(1)
        return real_call(*args, **kwargs)

    def spy_rows(fn, n, *args):
        rows.append(int(args[0].shape[0]))
        return real_rows(fn, n, *args)

    def spy_fwd_xla(*args, **kwargs):
        fallbacks.append("fwd")
        return real_fwd_xla(*args, **kwargs)

    def spy_bwd_xla(*args, **kwargs):
        fallbacks.append("bwd")
        return real_bwd_xla(*args, **kwargs)

    kernel_sharding.shard_layout_call = spy_call
    fused._rows = spy_rows
    fused._norm_forward_xla, fused._norm_backward_xla = spy_fwd_xla, spy_bwd_xla

    def inputs(batch, length=6, heads=4, head_dim=8, d_model=16):
        ks = jax.random.split(jax.random.key(batch), 8)
        y = jax.random.normal(ks[0], (batch, length, heads, head_dim))
        x = jax.random.normal(ks[1], (batch, length, heads, head_dim))
        z = jax.random.normal(ks[2], (batch, length, heads * head_dim))
        d = jax.random.normal(ks[3], (heads,))
        norm_w = 1.0 + 0.3 * jax.random.normal(ks[4], (heads * head_dim,))
        out_w = jax.random.normal(ks[5], (heads * head_dim, d_model)) * 0.2
        cot = jax.random.normal(ks[6], (batch, length, d_model))
        cot_norm = jax.random.normal(ks[7], (batch, length, heads * head_dim))
        return (y, x, z, d, norm_w, out_w), cot, cot_norm

    def step(args, cot, cot_norm):
        def loss(y, x, z, d, norm_w, out_w):
            out = fused.gated_norm_out(y, x, z, d, norm_w, out_w, eps=1e-5, compute_dtype=jnp.float32,
                                       interpret=True)
            normed = fused.gated_norm(y, x, z, d, norm_w, eps=1e-5, compute_dtype=jnp.float32, interpret=True)
            return jnp.sum(out * cot) + jnp.sum(normed * cot_norm)

        return jax.jit(jax.value_and_grad(loss, argnums=tuple(range(6))))(*args)

    report = {}
    assert fused.device_layout_ok() is False  # several devices, no context mesh: the unfused layer
    for batch, split in ((8, True), (3, False)):
        args, cot, cot_norm = inputs(batch)
        calls.clear()
        rows.clear()
        want = step(args, cot, cot_norm)  # no context mesh: the kernels on the whole rows, on one device
        assert not calls and set(rows) == {batch * 6}, (calls, rows)
        for name, shape in (("fsdp4", dict(dp=1, fsdp=4, tp=1)), ("dp2fsdp2", dict(dp=2, fsdp=2, tp=1)),
                            ("dp2tp2", dict(dp=2, fsdp=1, tp=2))):
            mesh = build_mesh(resolve_mesh_plan(4, **shape), topology_aware=False)
            batch_axes = tuple(a for a in ("dp", "fsdp") if shape[a] > 1)
            spec = P(batch_axes if len(batch_axes) > 1 else batch_axes[0])

            def place(a):  # activations split along the batch (3 rows do not divide: left whole)
                return jax.device_put(a, NamedSharding(mesh, spec)) if split and a.ndim >= 3 else a

            calls.clear()
            rows.clear()
            fallbacks.clear()
            with jax.set_mesh(mesh):
                assert fused.device_layout_ok()
                assert fused.heads_split() is (shape["tp"] > 1), name  # the layer keeps its unfused glue under tp
                got = step([place(a) for a in args], place(cot), place(cot_norm))
            # forward (twice: gated_norm_out and gated_norm) and backward (twice) per shard of the mesh
            assert len(calls) == 4, (name, calls)
            if split:  # every mesh axis splits the rows: each device normalizes 48 / 4 rows
                assert batch * 6 // 4 in rows and not fallbacks, (name, rows, fallbacks)
            else:  # 18 rows do not split over 4 devices: the plain-JAX formulas, which XLA partitions
                assert sorted(fallbacks) == ["bwd", "bwd", "fwd", "fwd"], (name, fallbacks)
                assert set(rows) <= {batch * 6}, (name, rows)  # (only the global shape check of the call)
            err = 0.0
            for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
                np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5, err_msg=name)
                err = max(err, float(np.max(np.abs(np.asarray(b) - np.asarray(a)))))
            report[f"{name}-b{batch}"] = err

    # A gate read in place from a stacked z|x tensor (z_row), per shard: the values and gradients of the sliced gate.
    (y, x, z, d, norm_w, _), _, cot_norm = inputs(8)
    zs = jnp.stack([z, 0.5 * z], axis=2)

    def gate_loss(z_row, y, x, zs, d, norm_w):
        gate = zs if z_row is not None else zs[:, :, 0]
        out = fused.gated_norm(y, x, gate, d, norm_w, eps=1e-5, compute_dtype=jnp.float32, interpret=True,
                               z_row=z_row)
        return jnp.sum(out * cot_norm)

    want = jax.jit(jax.value_and_grad(lambda *a: gate_loss(None, *a), argnums=tuple(range(5))))(y, x, zs, d, norm_w)
    mesh = build_mesh(resolve_mesh_plan(4, dp=1, fsdp=4, tp=1), topology_aware=False)
    placed = [jax.device_put(a, NamedSharding(mesh, P("fsdp"))) for a in (y, x, zs)]
    calls.clear()
    rows.clear()
    with jax.set_mesh(mesh):
        got = jax.jit(jax.value_and_grad(lambda *a: gate_loss(0, *a), argnums=tuple(range(5))))(*placed, d, norm_w)
    assert len(calls) == 2 and 8 * 6 // 4 in rows, (calls, rows)
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5, err_msg="z_row")
    assert not np.any(np.asarray(got[1][2])[:, :, 1])  # the x row's gradient is the conv's, not the glue's
    report["fsdp4-z_row"] = max(float(np.max(np.abs(np.asarray(b) - np.asarray(a))))
                                for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)))
    print("FUSEDGLUE " + json.dumps(report))
    """
)


def test_fused_glue_row_kernels_per_shard_on_cpu(tmp_path):
    script = tmp_path / "fused_glue_multidevice.py"
    script.write_text(CPU_SCRIPT)
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("FUSEDGLUE "))
    report = json.loads(line.split(" ", 1)[1])
    assert len(report) == 7, report


def _gpus() -> int:
    try:
        return sum(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:
        return 0




@pytest.mark.skipif(_gpus() < 2, reason="needs two or more GPUs (an FSDP mesh over every visible GPU)")
def test_fused_layer_matches_unfused_under_fsdp_on_gpus(monkeypatch):
    """The fused layer under an FSDP mesh over every visible GPU (Mosaic GPU SSD kernels, packed documents, odd
    length, grouped B/C key gradients): its row kernels see each device's rows, and its bf16 error against a
    float32 chunked reference stays within bf16 noise of the unfused layer's, output and every gradient."""
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from test_mamba2_fused import _flat, _layer, _segments, _value_and_grads

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.layers import mamba2_fused as fused
    from jla.layers.ssm import _fused_glue_applies

    rows = []
    real_rows = fused._rows

    def spy_rows(fn, n, *args):
        rows.append(int(args[0].shape[0]))
        return real_rows(fn, n, *args)

    monkeypatch.setattr(fused, "_rows", spy_rows)
    devices = _gpus()
    batch, length, d_model = 2 * devices, 200, 128
    mesh = build_mesh(resolve_mesh_plan(devices, dp=1, fsdp=devices, tp=1), topology_aware=False)
    sharding = NamedSharding(mesh, P("fsdp"))
    u = jax.device_put(jax.random.normal(jax.random.key(3), (batch, length, d_model)), sharding)
    seg = jax.device_put(_segments(batch, length, 4), sharding)
    cot = jax.device_put(jax.random.normal(jax.random.key(5), (batch, length, d_model)), sharding)
    shape = dict(d_model=d_model, grouped_key_grads=True)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):  # an exact float32 reference
        ref = _value_and_grads(_layer(fused_glue=False, impl="chunked", compute_dtype="float32", **shape), u, seg,
                               cot)
        ref = (np.asarray(ref[0]), _flat(ref[1]), np.asarray(ref[2]))
        fused_layer, unfused_layer = _layer(fused_glue=True, **shape), _layer(fused_glue=False, **shape)
        assert _fused_glue_applies(fused_layer) and not _fused_glue_applies(unfused_layer)
        errs = {}
        for name, layer in (("fused", fused_layer), ("unfused", unfused_layer)):
            out, gp, gu = _value_and_grads(layer, u.astype(jnp.bfloat16), seg, cot)
            e = {"out": float(np.max(np.abs(np.asarray(out, np.float32) - ref[0]))),
                 "u": float(np.max(np.abs(np.asarray(gu, np.float32) - ref[2])))}
            for key, value in _flat(gp).items():
                e[key] = float(np.max(np.abs(value - ref[1][key])))
            errs[name] = e
    assert batch * length // devices in rows, rows  # each device normalizes its own rows
    scales = {"out": float(np.max(np.abs(ref[0]))), "u": float(np.max(np.abs(ref[2])))}
    scales.update({key: float(np.max(np.abs(v))) for key, v in ref[1].items()})
    for key, scale in scales.items():
        f, b = errs["fused"][key], errs["unfused"][key]
        assert f <= 2.0 * b + 0.02 * scale, f"{key}: fused err {f:.3e}, unfused err {b:.3e}, scale {scale:.3e}"
    # A mesh that splits the heads (tensor parallelism) keeps the unfused glue: its row resharding is not measured.
    tp_mesh = build_mesh(resolve_mesh_plan(devices, dp=1, fsdp=1, tp=devices), topology_aware=False)
    with jax.set_mesh(tp_mesh):
        assert fused.heads_split() and fused.device_layout_ok() and not _fused_glue_applies(fused_layer)
