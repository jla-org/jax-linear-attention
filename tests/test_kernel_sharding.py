# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU kernels run per shard of the context mesh (``jla.distributed.kernel_sharding``).

XLA cannot partition the kernels' custom calls: under ``jit`` with sharded inputs it gathers every
operand and runs the whole kernel on each device. The dispatcher wraps ``impl="mosaic_gpu"`` calls in
``shard_map`` over the context mesh, and the trainer traces its steps under its mesh. On CPU a spy
kernel (the chunked op) records the shapes it is traced with, including a ``supports`` predicate
that rejects the per-shard heads. ``shard_layout_call`` (the kernels that layers call directly: the FA3-style
attention and the fused Mamba-3 SISO core) is checked on CPU with GQA heads, replicated parameters and a
state pytree. On GPUs the real kernels are checked against one GPU.
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
    import sys

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla import ops
    from jla.distributed.kernel_sharding import kernel_shard_axes, shard_kernel_call
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.ops import chunked

    assert len(jax.devices()) == 4, jax.devices()
    report = {}
    seen = []

    def spy(q, k, v, *gates, **options):
        seen.append(tuple(q.shape))
        return chunked.gated_delta_rule(q, k, v, *gates, **options)

    ops.available("gated_deltanet")  # load the real kernels first so the spy replaces them
    ops.register("mosaic_gpu", "gated_deltanet", spy, auto=False)

    B, T, H, K = 4, 32, 4, 16
    keys = jax.random.split(jax.random.key(0), 6)
    q, k, v = (jax.random.normal(keys[i], (B, T, H, K), jnp.float32) for i in range(3))
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H))) / 4
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))
    h0 = jax.random.normal(keys[5], (B, H, K, K), jnp.float32) * 0.1
    seg = jnp.asarray(np.repeat([[0] * 20 + [1] * 12], B, axis=0), jnp.int32)

    def loss(impl, q, k, v, g, beta, h0):
        o, final = ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, impl=impl, initial_state=h0,
                                        output_final_state=True, segment_ids=seg, use_qk_l2norm=True, chunk_size=16)
        return jnp.sum(o * o) + jnp.sum(final * final), (o, final)

    def run(impl):
        grad = jax.value_and_grad(lambda *a: loss(impl, *a), argnums=(0, 1, 2, 3, 4, 5), has_aux=True)
        return jax.jit(grad)

    expected = run("chunked")(q, k, v, g, beta, h0)
    seen.clear()
    plan = resolve_mesh_plan(4, dp=2, fsdp=1, tp=2)
    mesh = build_mesh(plan, topology_aware=False)
    tok = NamedSharding(mesh, P("dp", None, "tp"))
    placed = [jax.device_put(x, tok) for x in (q, k, v)] + [jax.device_put(x, tok) for x in (g, beta)]
    placed.append(jax.device_put(h0, NamedSharding(mesh, P("dp", "tp"))))
    with jax.set_mesh(mesh):
        assert kernel_shard_axes(jax.sharding.get_abstract_mesh(), B, H) == (("dp",), ("tp",))
        assert kernel_shard_axes(jax.sharding.get_abstract_mesh(), B, 3) == (("dp",), ())
        fn = run("mosaic_gpu")
        got = fn(*placed)
        text = fn.lower(*placed).compile().as_text()
    report["local_shapes"] = sorted(set(seen))
    assert (B // 2, T, H // 2, K) in seen, seen
    report["all_gathers"] = text.count("all-gather")
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)
    assert got[0][1][0].sharding.spec == P("dp", None, "tp"), got[0][1][0].sharding.spec

    # Without a context mesh (or inside a shard_map over every axis) the kernel sees the global shapes.
    seen.clear()
    jax.block_until_ready(run("mosaic_gpu")(*placed))
    assert set(seen) == {(B, T, H, K)}, seen
    seen.clear()
    with jax.set_mesh(mesh):
        inner = jax.shard_map(lambda q, k, v, g, beta: ops.linear_attention(
            "gated_deltanet", q, k, v, g=g, beta=beta, impl="mosaic_gpu")[0],
            mesh=mesh, in_specs=(P("dp", None, "tp"),) * 5, out_specs=P("dp", None, "tp"), check_vma=False)
        jax.block_until_ready(jax.jit(inner)(*placed[:5]))
    assert seen == [(B // 2, T, H // 2, K)], seen
    # cu_seqlens (packed rows) and inputs of unknown layout are never split.
    def record(q, k, v, *gates, **options):
        seen.append(tuple(q.shape))
        return q, None

    for options in ({"cu_seqlens": jnp.asarray([0, 20, 32], jnp.int32)}, {"extra": jnp.ones((3,))}):
        seen.clear()
        with jax.set_mesh(mesh):
            jax.block_until_ready(jax.jit(
                lambda q, k, v, g, beta: shard_kernel_call(record, q, k, v, (g, beta), options)[0])(*placed[:5]))
        assert seen == [(B, T, H, K)], (options, seen)
    print("KERNELSHARD " + json.dumps(report))
    """
)

SUPPORTS_SCRIPT = textwrap.dedent(
    r"""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla import ops
    from jla.distributed.kernel_sharding import _splits, kernel_shard_axes, shard_kernel_call, use_rules
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.ops import chunked

    assert len(jax.devices()) == 4, jax.devices()
    seen = []

    # Like the vector-gate kernels: accepted globally (6 * 64 = 384), but not per tp shard (3 * 64 = 192).
    def reason(q):
        return None if (q.shape[2] * q.shape[3]) % 128 == 0 else "H * K must be a multiple of 128"

    def supports(q, k, v, g, **options):
        return reason(q) is None

    def spy(q, k, v, g, **options):
        if reason(q) is not None:
            raise ValueError(f"spy gla: {reason(q)}")
        seen.append(tuple(q.shape))
        return chunked.gla(q, k, v, g, **options)

    ops.available("gla")
    ops.register("mosaic_gpu", "gla", spy, supports, auto=False)
    B, T, H, K = 4, 32, 6, 64
    keys = jax.random.split(jax.random.key(0), 4)
    q, k, v = (jax.random.normal(keys[i], (B, T, H, K), jnp.float32) * 0.5 for i in range(3))
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H, K))) / 4
    mesh = build_mesh(resolve_mesh_plan(4, dp=2, fsdp=1, tp=2), topology_aware=False)
    placed = [jax.device_put(x, NamedSharding(mesh, P("dp", None, "tp"))) for x in (q, k, v, g)]

    def call(impl, q, k, v, g):
        return ops.linear_attention("gla", q, k, v, g=g, impl=impl, output_final_state=True, chunk_size=16)

    expected = jax.jit(lambda *a: call("chunked", *a))(q, k, v, g)
    with jax.set_mesh(mesh):
        assert kernel_shard_axes(jax.sharding.get_abstract_mesh(), B, H) == (("dp",), ("tp",))
        got = jax.jit(lambda *a: call("mosaic_gpu", *a))(*placed)
        grads = jax.jit(jax.grad(lambda *a: jnp.sum(call("mosaic_gpu", *a)[0] ** 2), argnums=(0, 1, 2, 3)))(*placed)
    want_grads = jax.grad(lambda *a: jnp.sum(call("chunked", *a)[0] ** 2), argnums=(0, 1, 2, 3))(q, k, v, g)
    # The head split is rejected on the local shapes; the batch is split over dp AND tp, so no device
    # repeats another's rows (the global shape is the eval_shape trace that finds the output layout).
    assert set(seen) == {(B, T, H, K), (B // 4, T, H, K)}, seen
    for a, b in zip(jax.tree.leaves((expected, want_grads)), jax.tree.leaves((got, grads)), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)

    # A predicate that rejects every split leaves the call unsplit (the gathered global call).
    seen.clear()
    with jax.set_mesh(mesh):
        jax.block_until_ready(jax.jit(lambda q, k, v, g: shard_kernel_call(
            spy, q, k, v, (g,), {"chunk_size": 16},
            supports=lambda q, *a, **o: q.shape[0] == B and reason(q) is None)[0])(*placed))
    assert set(seen) == {(B, T, H, K)}, seen

    # Kinds outside TOKEN_LAYOUT_KINDS (parameter-shaped gates) are never split, whatever their shapes.
    def record(q, k, v, *gates, **options):
        seen.append(tuple(q.shape))
        return q, None

    seen.clear()
    with jax.set_mesh(mesh):
        jax.block_until_ready(jax.jit(lambda q, k, v, g: shard_kernel_call(
            record, q, k, v, (g,), {}, kind="rwkv6")[0])(*placed))
    assert seen == [(B, T, H, K)], seen

    # The split axes follow the rules in use (the trainer passes its resolved rules).
    with jax.set_mesh(mesh):
        abstract = jax.sharding.get_abstract_mesh()
        with use_rules({"batch": ("fsdp",), "heads": ("tp",)}):
            assert kernel_shard_axes(abstract, B, H) == ((), ("tp",))
            # dp, which the rules leave unplaced, still splits the batch: no mesh axis is left out.
            assert _splits(abstract, B, H)[0] == (("dp",), ("tp",)), _splits(abstract, B, H)
            seen.clear()
            jax.block_until_ready(jax.jit(lambda q, k, v, g: shard_kernel_call(
                record, q, k, v, (g,), {}, kind="gla")[0])(*placed))
            assert set(seen) == {(B, T, H, K), (B // 2, T, H // 2, K)}, seen
        # Every candidate covers every axis of size > 1, or there is none (the unsplit call).
        assert _splits(abstract, B, None) == [(("dp", "tp"), ())], _splits(abstract, B, None)
        assert _splits(abstract, B, 3) == [(("dp", "tp"), ())], _splits(abstract, B, 3)
        assert _splits(abstract, 2, None) == [], _splits(abstract, 2, None)
        assert _splits(abstract, 2, H) == [(("dp",), ("tp",))], _splits(abstract, 2, H)
        assert kernel_shard_axes(abstract, B, H) == (("dp",), ("tp",))
    # An axis that no rule names (e.g. an expert-parallel "ep") splits the batch too.
    foreign = jax.sharding.AbstractMesh((2, 2, 2), ("dp", "ep", "tp"))
    assert _splits(foreign, B, H)[0] == (("dp", "ep"), ("tp",)), _splits(foreign, B, H)
    assert _splits(foreign, 2 * B, None) == [(("dp", "ep", "tp"), ())], _splits(foreign, 2 * B, None)
    assert _splits(foreign, B, None) == [], _splits(foreign, B, None)  # 4 rows over 8 devices: unsplit
    print("SUPPORTSHARD " + json.dumps({"ok": True}))
    """
)

TRAINER_SCRIPT = textwrap.dedent(
    r"""
    import dataclasses
    import json
    import sys

    import jax
    import numpy as np

    from _toy_model import experiment
    from jla import ops
    from jla.models import get_preset
    from jla.ops import chunked
    from jla.training.trainer import Trainer

    assert len(jax.devices()) == 4
    seen = []

    def spy(q, k, v, *gates, **options):
        seen.append(tuple(q.shape))
        return chunked.gated_delta_rule(q, k, v, *gates, **options)

    ops.available("gated_deltanet")
    ops.register("mosaic_gpu", "gated_deltanet", spy, auto=False)
    out = sys.argv[1]
    base = get_preset("tiny-hybrid").replace(compute_dtype="float32")

    def run(name, impl, **mesh):
        model = base.replace(linear_attention=dataclasses.replace(base.linear_attention, implementation=impl))
        train = {"max_steps": 3, "log_every": 1, "micro_batch_per_replica": 2, "grad_accum": 1, "final_eval": True,
                 "eval_batches": 1, "ce_compute_dtype": "float32"}
        optim = {"name": "sgd", "lr": 0.1, "b1": 0.9, "warmup_steps": 0}
        return Trainer(experiment(f"{out}/{name}", model=model, train=train, optim=optim, mesh=mesh)).run()

    reference = run("chunked", "chunked", dp=2, fsdp=2)
    seen.clear()
    kernel = run("kernel", "mosaic_gpu", dp=2, fsdp=2)
    local = sorted({shape for shape in seen if shape[0] == 2})
    assert local, seen  # global batch 8 split over dp * fsdp = 4 devices
    for a, b in zip(reference.history, kernel.history, strict=True):
        np.testing.assert_allclose(b["loss"], a["loss"], rtol=1e-5)
    np.testing.assert_allclose(kernel.eval_history[-1]["eval_loss"], reference.eval_history[-1]["eval_loss"], rtol=1e-5)
    # The trainer leaves no mesh behind.
    assert jax.sharding.get_abstract_mesh().empty
    print("TRAINERSHARD " + json.dumps({"local_shapes": local, "losses": [r["loss"] for r in kernel.history]}))
    """
)


def _run_cpu(tmp_path: Path, script: str, marker: str) -> dict:
    path = tmp_path / "script.py"
    path.write_text(script)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    result = subprocess.run([sys.executable, str(path), str(tmp_path / "runs")], env=env, capture_output=True,
                            text=True, timeout=900, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    return json.loads(result.stdout.split(marker + " ", 1)[1])


def test_mosaic_dispatch_runs_per_shard_of_the_context_mesh(tmp_path: Path) -> None:
    report = _run_cpu(tmp_path, CPU_SCRIPT, "KERNELSHARD")
    assert report["all_gathers"] == 0


def test_kernel_supports_is_checked_on_local_shapes(tmp_path: Path) -> None:
    assert _run_cpu(tmp_path, SUPPORTS_SCRIPT, "SUPPORTSHARD") == {"ok": True}


def test_trainer_rejects_context_parallelism(tmp_path: Path) -> None:
    import dataclasses

    from _toy_model import experiment

    from jla.models import get_preset
    from jla.training.trainer import Trainer

    base = get_preset("tiny-hybrid")
    linear = dataclasses.replace(base.linear_attention, implementation="chunked", context_parallel_axis="fsdp")
    model = base.replace(linear_attention=linear)
    with pytest.raises(NotImplementedError, match="context_parallel_axis"):
        Trainer(experiment(str(tmp_path / "cp"), model=model)).run()


def test_trainer_traces_kernels_per_shard(tmp_path: Path) -> None:
    report = _run_cpu(tmp_path, TRAINER_SCRIPT, "TRAINERSHARD")
    assert all(shape[0] == 2 for shape in map(tuple, report["local_shapes"]))


@pytest.mark.skipif(jax.default_backend() != "gpu" or len(jax.devices()) < 2, reason="needs two GPUs")
def test_mosaic_gated_deltanet_on_two_gpus_matches_one_gpu() -> None:
    """Needs a process that sees two H100s, which a one-GPU test runner never gives it; run it with e.g.
    ``CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/test_kernel_sharding.py -k two_gpus``."""
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.ops import linear_attention

    if jax.devices()[0].compute_capability != "9.0":
        pytest.skip("Mosaic GPU kernels need sm_90a")
    B, T, H, K = 4, 512, 4, 128
    keys = jax.random.split(jax.random.key(1), 5)
    q, k, v = (jax.random.normal(keys[i], (B, T, H, K), jnp.bfloat16) for i in range(3))
    g = jax.nn.log_sigmoid(jax.random.normal(keys[3], (B, T, H))) / 8
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))

    def fwd_bwd(q, k, v, g, beta):
        def loss(q, k, v, g, beta):
            o, _ = linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, impl="mosaic_gpu", use_qk_l2norm=True)
            return jnp.sum(o.astype(jnp.float32) ** 2), o
        return jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(q, k, v, g, beta)

    one = jax.jit(fwd_bwd)(*(jax.device_put(x, jax.devices()[0]) for x in (q, k, v, g, beta)))
    for spec_axes in ({"dp": 2}, {"tp": 2}):
        mesh = build_mesh(resolve_mesh_plan(2, fsdp=1, **spec_axes), jax.devices()[:2], topology_aware=False)
        tok = NamedSharding(mesh, P("dp", None, "tp"))
        placed = [jax.device_put(x, tok) for x in (q, k, v, g, beta)]
        with jax.set_mesh(mesh):
            fn = jax.jit(fwd_bwd)
            two = fn(*placed)
            text = fn.lower(*placed).compile().as_text()
        assert "all-gather" not in text, spec_axes
        (loss_one, rest_one), (loss_two, rest_two) = [(out[0][0], (out[0][1], out[1])) for out in (one, two)]
        # The loss sums 1M squares: two partial sums plus an all-reduce add in another order.
        np.testing.assert_allclose(float(loss_two), float(loss_one), rtol=1e-5)
        # Each shard runs the one-GPU kernel on its rows/heads, but not bitwise like one GPU even with the
        # same sequence pieces (the piece count depends on B * H): on H100 the XLA code around the kernel
        # fuses differently (36 of 1M bf16 outputs differed by one ulp, 3% of the float32 gate gradients
        # in their last bits), so the check is bf16-close.
        for a, b in zip(jax.tree.leaves(rest_one), jax.tree.leaves(rest_two), strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            np.testing.assert_allclose(b, a, rtol=2e-2, atol=2e-2 * float(np.abs(a).max()))


LAYOUT_SCRIPT = textwrap.dedent(
    r"""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed.kernel_sharding import shard_layout_call
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    assert len(jax.devices()) == 4, jax.devices()
    mesh = build_mesh(resolve_mesh_plan(4, dp=2, fsdp=1, tp=2), topology_aware=False)
    seen = []
    B, T, H, HKV, D = 4, 16, 4, 2, 8
    keys = jax.random.split(jax.random.key(0), 8)
    q = jax.random.normal(keys[0], (B, T, H, D), jnp.float32)
    k, v = (jax.random.normal(keys[i], (B, T, HKV, D), jnp.float32) for i in (1, 2))
    seg = jnp.asarray(np.repeat([[0] * 10 + [1] * 6], B, axis=0), jnp.int32)

    # GQA attention like the Mosaic GPU kernel's call: heads split in contiguous blocks keep their groups.
    def attn(q, k, v, segment_ids):
        seen.append(("attn", tuple(q.shape), tuple(k.shape)))
        mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]
        return jax.nn.dot_product_attention(q, k, v, mask=mask, is_causal=True, implementation="xla")

    def attn_call(q, k, v, seg, supports=None):
        return shard_layout_call(attn, {"q": q, "k": k, "v": v, "segment_ids": seg},
                                 {"q": "token", "k": "token", "v": "token", "segment_ids": "batch"},
                                 out_layouts="token", heads=2, supports=supports)

    def attn_loss(q, k, v, seg, supports=None):
        return jnp.sum(attn_call(q, k, v, seg, supports) ** 2)

    expected = jax.value_and_grad(attn_loss, argnums=(0, 1, 2))(q, k, v, seg)
    tok = NamedSharding(mesh, P("dp", None, "tp"))
    placed = [jax.device_put(x, tok) for x in (q, k, v)] + [jax.device_put(seg, NamedSharding(mesh, P("dp")))]
    seen.clear()
    with jax.set_mesh(mesh):
        fn = jax.jit(jax.value_and_grad(attn_loss, argnums=(0, 1, 2)))
        got = fn(*placed)
        out = jax.jit(attn_call)(*placed)
    assert ("attn", (B // 2, T, H // 2, D), (B // 2, T, HKV // 2, D)) in seen, seen
    assert out.sharding.spec == P("dp", None, "tp"), out.sharding.spec
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)

    # A predicate that rejects the local head split splits the batch over dp and tp; one that rejects
    # all runs unsplit.
    for supports, want in ((lambda q, k, v, segment_ids: q.shape[2] == H, (B // 4, T, H, D)),
                           (lambda q, k, v, segment_ids: q.shape == (B, T, H, D), (B, T, H, D))):
        seen.clear()
        with jax.set_mesh(mesh):
            got = jax.jit(jax.value_and_grad(lambda *a: attn_loss(*a, supports=supports), argnums=(0, 1, 2)))(*placed)
        assert {shape for _, shape, _ in seen} == {(B, T, H, D), want}, (want, seen)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(got), strict=True):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)

    # Like the fused Mamba-3 core: batch-only layouts, replicated parameters, an optional operand and a
    # state pytree. The parameters' gradients are summed over every device of the mesh.
    x = jax.random.normal(keys[3], (B, T, H, D), jnp.float32)
    w = jax.random.normal(keys[4], (H, D), jnp.float32)
    s0 = (jax.random.normal(keys[5], (B, H), jnp.float32), jax.random.normal(keys[6], (B, H, D, D), jnp.float32))

    def core(x, z, initial_state, w):
        seen.append(("core", tuple(x.shape)))
        assert z is None
        a, s = initial_state
        y = jnp.einsum("bthd,bhde->bthe", jnp.tanh(x * w + a[:, None, :, None]), s)
        # Every example on its own (a batch split is exact only for such functions).
        return y, (a + jnp.sum(y, axis=(1, 3)), s * 0.5 + 0.1 * jnp.einsum("bthd,bthe->bhde", y, x) / T)

    def core_call(x, s0, w):
        return shard_layout_call(core, {"x": x, "z": None, "initial_state": s0, "w": w},
                                 {"x": "batch", "z": "batch", "initial_state": "batch", "w": "replicated"},
                                 out_layouts=("batch", "batch"))

    def core_loss(x, s0, w):
        y, (a, s) = core_call(x, s0, w)
        return jnp.sum(y ** 2) + jnp.sum(a) + jnp.sum(s)

    want = jax.value_and_grad(core_loss, argnums=(0, 1, 2))(x, s0, w)
    rows = NamedSharding(mesh, P("dp"))
    placed = (jax.device_put(x, rows), jax.tree.map(lambda t: jax.device_put(t, rows), s0),
              jax.device_put(w, NamedSharding(mesh, P())))
    seen.clear()
    with jax.set_mesh(mesh):
        got = jax.jit(jax.value_and_grad(core_loss, argnums=(0, 1, 2)))(*placed)
    # Batch-only: the batch is split over dp AND tp (an axis left out would repeat the core on it).
    assert ("core", (B // 4, T, H, D)) in seen, seen
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)

    # Without a context mesh the function sees the global shapes.
    seen.clear()
    jax.block_until_ready(jax.jit(core_call)(x, s0, w))
    assert seen == [("core", (B, T, H, D))], seen
    print("LAYOUTSHARD " + json.dumps({"ok": True}))
    """
)


def test_layout_call_runs_per_shard_with_replicated_parameters(tmp_path: Path) -> None:
    assert _run_cpu(tmp_path, LAYOUT_SCRIPT, "LAYOUTSHARD") == {"ok": True}


@pytest.mark.skipif(jax.default_backend() != "gpu" or len(jax.devices()) < 2, reason="needs two GPUs")
def test_mosaic_attention_and_fused_mamba3_on_gpus_match_one_gpu() -> None:
    """The FA3-style attention (GQA, sliding window, documents) and the fused Mamba-3 SISO core on every
    visible GPU (batch split, plus a head split for attention when there are 4 or more; the Mamba-3 core
    splits its batch over tp as well) against one GPU: outputs and per-row gradients bitwise equal
    (attention dq and dk to bf16 rounding).
    Run with all GPUs of the machine visible, e.g.
    ``CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/test_kernel_sharding.py -k on_gpus``."""
    import json
    import math

    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.layers import ssm
    from jla.layers.attention import attention

    if jax.devices()[0].compute_capability != "9.0":
        pytest.skip("Mosaic GPU kernels need sm_90a")
    n = len(jax.devices()) // 2 * 2
    tp = 2 if n >= 4 else 1
    mesh = build_mesh(resolve_mesh_plan(n, dp=n // tp, fsdp=1, tp=tp), jax.devices()[:n], topology_aware=False)
    first = jax.devices()[0]

    # Attention: q [B, T, 8, 128], k/v [B, T, 2, 128], two documents per row, window 256.
    B, T, H, HKV, D = n, 1024, 8, 2, 128
    keys = jax.random.split(jax.random.key(2), 4)
    q = jax.random.normal(keys[0], (B, T, H, D), jnp.bfloat16)
    k, v = (jax.random.normal(keys[i], (B, T, HKV, D), jnp.bfloat16) for i in (1, 2))
    do = jax.random.normal(keys[3], (B, T, H, D), jnp.float32)
    seg = jnp.asarray(np.repeat([[0] * 600 + [1] * 424], B, axis=0), jnp.int32)

    def attn_loss(q, k, v, seg, do):
        o = attention(q, k, v, segment_ids=seg, sliding_window=256, implementation="mosaic_gpu")
        return jnp.sum(o.astype(jnp.float32) * do), o

    attn_grad = jax.value_and_grad(attn_loss, argnums=(0, 1, 2), has_aux=True)
    one = jax.jit(attn_grad)(*(jax.device_put(x, first) for x in (q, k, v, seg, do)))
    tok, rows = NamedSharding(mesh, P("dp", None, "tp")), NamedSharding(mesh, P("dp"))
    placed = [jax.device_put(x, tok) for x in (q, k, v)] + [jax.device_put(seg, rows), jax.device_put(do, tok)]
    with jax.set_mesh(mesh):
        fn = jax.jit(attn_grad)
        many = fn(*placed)
        text = fn.lower(*placed).compile().as_text()
    assert "all-gather" not in text
    assert many[0][1].sharding.spec == P("dp", None, "tp"), many[0][1].sharding.spec
    np.testing.assert_allclose(float(many[0][0]), float(one[0][0]), rtol=1e-4)
    attn_diffs = {}
    for label, a, b in zip(("o", "dq", "dk", "dv"), (one[0][1], *one[1]), (many[0][1], *many[1]), strict=True):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        attn_diffs[label] = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-30)
    print(f"ATTNSHARD n={n} tp={tp} " + json.dumps(attn_diffs))
    # Each device runs the kernel on its own rows and GQA head blocks: o and dv bitwise equal to one GPU.
    # dq and dk also use the backward's XLA-computed fp32 row sums of o * do, whose rounding is not fixed
    # across compilations (2 GPUs: 0 in one run, up to 8.3e-4 of the max in another): within 2e-3 of the max.
    assert attn_diffs["o"] == 0.0 and attn_diffs["dv"] == 0.0, attn_diffs
    assert attn_diffs["dq"] <= 2e-3 and attn_diffs["dk"] <= 2e-3, attn_diffs

    # Fused Mamba-3 SISO core: the batch split, with bias and skip parameters replicated.
    B, T, H, G, K, V = n, 512, 8, 1, 128, 64
    S = K // 4
    ks = jax.random.split(jax.random.key(3), 12)

    def rms(x):
        return x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)

    dt = jnp.exp(jax.random.uniform(ks[3], (B, T, H), minval=math.log(1e-3), maxval=math.log(0.1)))
    inputs = dict(
        q=rms(jax.random.normal(ks[0], (B, T, G, K))).astype(jnp.bfloat16),
        k=rms(jax.random.normal(ks[1], (B, T, G, K))).astype(jnp.bfloat16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(jnp.bfloat16),
        adt=-jax.random.uniform(ks[4], (B, T, H), minval=1.0, maxval=16.0) * dt, dt=dt,
        trap=jax.random.normal(ks[5], (B, T, H)),
        q_bias=1.0 + 0.3 * jax.random.normal(ks[6], (H, K)), k_bias=1.0 + 0.3 * jax.random.normal(ks[7], (H, K)),
        angles=jax.random.normal(ks[8], (B, T, H, S)), d=1.0 + 0.1 * jax.random.normal(ks[9], (H,)),
        z=jax.random.normal(ks[10], (B, T, H, V)).astype(jnp.bfloat16),
    )
    dy = jax.random.normal(ks[11], (B, T, H, V), jnp.float32)
    names = tuple(inputs)

    def ssm_loss(*args):
        x = dict(zip(names, args[:-1], strict=True))
        y, _ = ssm.mamba3_siso(x.pop("q"), x.pop("k"), x.pop("v"), x.pop("adt"), x.pop("dt"), x.pop("trap"),
                               x.pop("q_bias"), x.pop("k_bias"), x.pop("angles"), **x, impl="mosaic_gpu")
        return jnp.sum(y.astype(jnp.float32) * args[-1]), y

    ssm_grad = jax.value_and_grad(ssm_loss, argnums=tuple(range(len(names))), has_aux=True)
    args = (*inputs.values(), dy)
    one = jax.jit(ssm_grad)(*(jax.device_put(x, first) for x in args))
    replicated = NamedSharding(mesh, P())
    placed = [jax.device_put(x, replicated if name in ("q_bias", "k_bias", "d") else NamedSharding(mesh, P("dp")))
              for name, x in zip((*names, "dy"), args, strict=True)]
    with jax.set_mesh(mesh):
        fn = jax.jit(ssm_grad)
        many = fn(*placed)
        text = fn.lower(*placed).compile().as_text()
    assert text.count("all-gather") == 0 or tp > 1, "the batch split should need no gathers"
    np.testing.assert_allclose(float(many[0][0]), float(one[0][0]), rtol=1e-5)
    # Every mesh axis, tp included, splits the batch (B // n rows per device; with tp left out of the
    # specs every gradient was off by up to 1.7e-2 of its max). Each device runs the core on its own
    # rows as one GPU does on all of them, so y and every per-row gradient are bitwise equal; only the
    # gradients of the replicated parameters (q_bias, k_bias, d), summed over the devices, differ in
    # reduction order.
    y_many = many[0][1]
    diffs = {"y_shard_rows": int(y_many.sharding.shard_shape(y_many.shape)[0]), "all_gathers": text.count("all-gather")}
    for label, a, b in zip(("y", *names), (one[0][1], *one[1]), (many[0][1], *many[1]), strict=True):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        diffs[label] = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-30)
    print(f"M3SHARD n={n} tp={tp} " + json.dumps(diffs))
    for label in ("y", *names):
        limit = 1e-5 if label in ("q_bias", "k_bias", "d") else 0.0
        assert diffs[label] <= limit, (label, diffs)
