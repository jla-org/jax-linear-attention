# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exactness checks of context-parallel softmax attention on 8 devices (run as a script).

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
        python tests/_context_parallel_attention_checks.py
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tests/_context_parallel_attention_checks.py --matmul-precision highest

``tests/test_context_parallel_attention.py`` runs the first form (8 CPU devices). The
second runs the same cases on 8 GPUs, with float32 matmuls at ``"highest"`` precision
(no TF32) so the float32 tolerances mean the same thing there. Prints
``CP_ATTENTION <json report>`` and exits with status 1 when a case is outside its
tolerance (:func:`failures`).

Ring attention (zigzag and contiguous layouts, truncated rings for sliding windows,
query-chunked score blocks) and Ulysses, with and without attention sinks, are compared
with :func:`jla.layers.attention.attention` on one device: outputs and gradients of q,
k, v (and sinks), with packed documents whose boundaries fall inside shards, on shard
boundaries and on zigzag chunk boundaries, GQA, ``cp=8``, ``cp=3``, ``dp=2 x cp=4`` and
``dp=4 x cp=2`` meshes, and on a ``dp=2 x cp=4`` mesh with Explicit axes (replicated or
T-sharded inputs). The ``Attention`` layer switch and one SGD train step of a hybrid
Gated DeltaNet + attention model with MTP (both mixers sequence-sharded; also on the
documented ``jax.make_mesh(..., axis_types=Auto)`` mesh) must match the unsharded model.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys

# Tolerances on max |error| / max(1, max |reference|); train steps: per parameter leaf (see below).
F32_TOL = 2e-5
BF16_TOL = 3e-2
TRAIN_STEP_TOL = 1e-4

NUM_F32_CASES = 12 + 3 + 2 + 2 + 2 + 2 + 3 + 7 + 3


def failures(report: dict) -> dict:
    """Cases of ``report`` outside their tolerance."""
    bad = {}
    for case, errs in report["f32"].items():
        if max(errs.values()) > (TRAIN_STEP_TOL if case.startswith("train_step/") else F32_TOL):
            bad[case] = errs
    bad.update({case: errs for case, errs in report["bf16"].items() if max(errs.values()) > BF16_TOL})
    if len(report["f32"]) != NUM_F32_CASES:
        bad["case_count"] = {"expected": NUM_F32_CASES, "got": len(report["f32"])}
    return bad


def run_checks() -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.distributed.context_parallel_attention import context_parallel_attention, ring_steps
    from jla.layers.attention import Attention, attention
    from jla.models import build_model, get_preset
    from jla.models.config import AttentionConfig
    from jla.training.step import LossConfig, build_train_step, lm_loss

    assert len(jax.devices()) == 8, jax.devices()
    B, T, H, HKV, D = 2, 128, 8, 2, 16  # 16 tokens per shard on cp=8 (zigzag chunks of 8), GQA group 4
    mesh8 = Mesh(np.array(jax.devices()), ("cp",))
    mesh24 = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "cp"))
    mesh3 = Mesh(np.array(jax.devices()[:3]), ("cp",))  # an odd ring: zigzag permutations for any P
    mesh42 = Mesh(np.array(jax.devices()).reshape(4, 2), ("dp", "cp"))  # the 2-ring; Ulysses without KV repeats
    assert (ring_steps(T, 8, None), ring_steps(T, 8, 5), ring_steps(T, 8, 40), ring_steps(T, 8, 200)) == (8, 2, 4, 8)
    assert ring_steps(T, 4, 24) == 2 and ring_steps(T, 8, 17) == 2 and ring_steps(T, 8, 18) == 3

    def segments(batch=B, length=T):
        # Row 0: starts inside shards (5, 37), on cp=8 shard boundaries (16, 64) and on a zigzag
        # chunk boundary (88); row 1: one document over shards 0-5, then a start inside a shard.
        starts = ([5, 16, 37, 64, 88], [97])
        resets = np.zeros((batch, length), np.int32)
        for row in range(batch):
            resets[row, [t for t in starts[row % 2] if t < length]] = 1
        return jnp.asarray(np.cumsum(resets, axis=1) + 3)

    def inputs(seed, dtype=jnp.float32, length=T):
        ks = jax.random.split(jax.random.key(seed), 5)
        q = jax.random.normal(ks[0], (B, length, H, D), dtype)
        k = jax.random.normal(ks[1], (B, length, HKV, D), dtype)
        v = jax.random.normal(ks[2], (B, length, HKV, D), dtype)
        w = jax.random.normal(ks[3], (B, length, H, D), jnp.float32)
        sinks = 0.5 * jax.random.normal(ks[4], (H,))
        return q, k, v, w, sinks

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))

    def op_case(seed, mesh, batch_axis, strategy, load_balance, seg, window, use_sinks, dtype=jnp.float32,
                block_q=None, length=T):
        q, k, v, w, sinks = inputs(seed, dtype, length)

        def loss(fn):
            def f(q, k, v, sinks):
                o = fn(q, k, v, sinks if use_sinks else None)
                return jnp.sum(o.astype(jnp.float32) * w), o
            return jax.jit(jax.value_and_grad(f, argnums=(0, 1, 2, 3), has_aux=True))

        common = dict(segment_ids=seg, sliding_window=window)
        chunks = {} if block_q is None else {"block_q": block_q}
        ref = loss(lambda q, k, v, s: attention(q, k, v, implementation="xla", sinks=s, **common))
        cp = loss(lambda q, k, v, s: context_parallel_attention(
            q, k, v, mesh=mesh, axis="cp", batch_axis=batch_axis, strategy=strategy, load_balance=load_balance,
            sinks=s, **common, **chunks))
        (_, o_ref), g_ref = ref(q, k, v, sinks)
        (_, o_cp), g_cp = cp(q, k, v, sinks)
        errs = {"o": rel(o_cp, o_ref)}
        for name, a, b in zip(("dq", "dk", "dv", "dsinks"), g_cp, g_ref):
            if name == "dsinks" and not use_sinks:
                continue
            errs[name] = rel(a, b)
        return errs

    seg = segments()
    cases = {
        "ring/zigzag/cp8": (0, mesh8, None, "ring", True, None, None, False),
        "ring/zigzag/cp8/seg": (1, mesh8, None, "ring", True, seg, None, False),
        "ring/contiguous/cp8/seg": (2, mesh8, None, "ring", False, seg, None, False),
        "ring/window5/cp8/seg": (3, mesh8, None, "ring", True, seg, 5, False),
        "ring/window40/cp8/seg": (4, mesh8, None, "ring", True, seg, 40, False),
        "ring/window200/cp8": (5, mesh8, None, "ring", True, None, 200, False),
        "ring/zigzag/dp2xcp4/seg": (6, mesh24, "dp", "ring", True, seg, None, False),
        "ring/window24/dp2xcp4/seg": (7, mesh24, "dp", "ring", True, seg, 24, False),
        "ulysses/cp8/seg": (8, mesh8, None, "ulysses", True, seg, None, False),
        "ulysses/window40/cp8/seg": (9, mesh8, None, "ulysses", True, seg, 40, False),
        "ulysses/sinks/cp8/seg": (10, mesh8, None, "ulysses", True, seg, None, True),
        "ulysses/dp2xcp4/seg": (11, mesh24, "dp", "ulysses", True, seg, None, False),
    }
    report = {name: op_case(*args) for name, args in cases.items()}
    # Query-chunked score blocks (block_q rows at a time) in every ring branch.
    report["ring/zigzag/cp8/seg/block_q4"] = op_case(13, mesh8, None, "ring", True, seg, None, False, block_q=4)
    report["ring/window40/cp8/seg/block_q4"] = op_case(14, mesh8, None, "ring", True, seg, 40, False, block_q=4)
    report["ring/contiguous/dp2xcp4/seg/block_q8"] = op_case(15, mesh24, "dp", "ring", False, seg, None, False,
                                                              block_q=8)
    # Windows narrower than a block: each query chunk scores only block_q + W - 1 keys (a band of
    # 8 of 16 keys on cp=8, 19 of 64 on cp=2), also across the shard boundary.
    report["ring/window5/cp8/seg/block_q4"] = op_case(22, mesh8, None, "ring", True, seg, 5, False, block_q=4)
    report["ring/window12/dp4xcp2/seg/block_q8"] = op_case(23, mesh42, None, "ring", True, seg, 12, False,
                                                            block_q=8)
    seg96 = segments(length=96)  # T = 96 on 3 devices: shards of 32, zigzag chunks of 16
    report["ring/zigzag/cp3/seg"] = op_case(16, mesh3, None, "ring", True, seg96, None, False, length=96)
    report["ring/window20/cp3/seg"] = op_case(17, mesh3, None, "ring", True, seg96, 20, False, length=96)
    report["ring/sinks/cp8/seg"] = op_case(18, mesh8, None, "ring", True, seg, None, True)
    report["ring/sinks/window24/cp8/seg"] = op_case(19, mesh8, None, "ring", True, seg, 24, True)
    report["ring/zigzag/dp4xcp2/seg"] = op_case(20, mesh42, None, "ring", True, seg, None, False)
    report["ulysses/dp4xcp2/seg"] = op_case(21, mesh42, None, "ulysses", True, seg, None, False)

    # jax.make_mesh's default Explicit axes, under jax.set_mesh: the op reshards replicated or
    # T-sharded inputs into its layout (B over dp, T over cp) and returns q's sharding.
    meshx = jax.make_mesh((2, 4), ("dp", "cp"), axis_types=(AxisType.Explicit,) * 2)

    def explicit_case(seed, strategy, window, in_spec):
        q, k, v, w, _ = inputs(seed)

        def loss(fn):
            def f(q, k, v):
                o = fn(q, k, v)
                return jnp.sum(o * w), o
            return jax.jit(jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True))

        (_, o_ref), g_ref = loss(lambda q, k, v: attention(q, k, v, implementation="xla", segment_ids=seg,
                                                          sliding_window=window))(q, k, v)
        with jax.set_mesh(meshx):
            sharding = NamedSharding(meshx, in_spec)
            q, k, v, w, seg_x = (jax.device_put(x, sharding) for x in (q, k, v, w, seg))
            (_, o_cp), g_cp = loss(lambda q, k, v: context_parallel_attention(
                q, k, v, axis="cp", batch_axis="auto", strategy=strategy, segment_ids=seg_x,
                sliding_window=window))(q, k, v)
        assert o_cp.sharding.is_equivalent_to(sharding, 4), (o_cp.sharding, sharding)
        return {"o": rel(o_cp, o_ref), **{n: rel(a, b) for n, a, b in zip(("dq", "dk", "dv"), g_cp, g_ref)}}

    report["explicit/ring/zigzag/dp2xcp4/seg/replicated"] = explicit_case(24, "ring", None, P())
    report["explicit/ring/window24/dp2xcp4/seg/T_sharded"] = explicit_case(25, "ring", 24, P(None, "cp"))
    report["explicit/ulysses/dp2xcp4/seg/T_sharded"] = explicit_case(26, "ulysses", None, P(None, "cp"))
    bf16 ={"ring/zigzag/cp8/seg/bf16": op_case(12, mesh8, None, "ring", True, seg, None, False, jnp.bfloat16)}

    # The layer switch: Attention with context_parallel_axis under the context mesh.
    def layer(**kw):
        cfg = AttentionConfig(num_heads=H, num_kv_heads=HKV, head_dim=D, output_gate=True, rotary_fraction=0.5, **kw)
        return Attention(32, cfg, key=jax.random.key(3), std=0.2, norm_eps=1e-6, compute_dtype="float32")

    x = jax.random.normal(jax.random.key(4), (B, T, 32))
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))

    @eqx.filter_value_and_grad
    def layer_loss(m, x):
        return jnp.sum(m(x, positions, seg) ** 2)

    ref_val, ref_grads = eqx.filter_jit(layer_loss)(layer(), x)
    layer_cases = [("ring", None, "cp8", False), ("ring", 24, "cp8", False), ("ulysses", None, "cp8", False),
                   ("ulysses", 24, "cp8", False), ("ring", 24, "cp8", True), ("ulysses", 24, "cp8", True),
                   ("ring", None, "dp2xcp4", False)]  # dp2xcp4: the layer shards B over dp (batch_axis="auto")
    for strategy, window, mesh_name, sinks in layer_cases:
        with jax.set_mesh(mesh8 if mesh_name == "cp8" else mesh24):
            val, grads = eqx.filter_jit(layer_loss)(layer(
                context_parallel_axis="cp", context_parallel_strategy=strategy, sliding_window=window, sinks=sinks), x)
        base_val, base_grads = (ref_val, ref_grads) if window is None and not sinks else \
            eqx.filter_jit(layer_loss)(layer(sliding_window=window, sinks=sinks), x)
        errs = {"loss": rel(val, base_val)}
        errs["grads"] = max(rel(a, b) for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(base_grads),
                                                       strict=True))
        report[f"layer/{strategy}/window={window}/{mesh_name}" + ("/sinks" if sinks else "")] = errs
    try:
        layer(context_parallel_axis="cp")(x, positions, seg)
    except ValueError as err:
        assert "context parallelism needs a mesh" in str(err), err
    else:
        raise AssertionError("context_parallel_axis without a mesh must raise")

    # One SGD step of a hybrid model (Gated DeltaNet + GQA attention + MTP) with T sharded end to end.
    def leaves(tree):
        return [np.array(a, copy=True) for a in jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))]

    def train_once(cfg, mesh):
        model = build_model(cfg, key=jax.random.key(0))
        init = leaves(model)  # the step donates the model
        # A large SGD step, so that every update (-50 x gradient) dominates the float32 rounding
        # of its parameter: with lr 0.5 the RMSNorm weights (1.0) move by ~2e-4, and one ulp
        # of 1.0 would already be 3e-4 of that update.
        optimizer = optax.sgd(50.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype="float32")))
        ids = jax.random.randint(jax.random.key(5), (1, B, T), 0, cfg.vocab_size)
        batch = {"input_ids": ids, "labels": jnp.roll(ids, -1, axis=-1), "segment_ids": segments()[None]}
        with contextlib.nullcontext() if mesh is None else jax.set_mesh(mesh):
            new_model, _, metrics, _ = step(model, opt_state, batch, jax.random.key(6))
        return float(metrics["loss"]), float(metrics["grad_norm"]), [a - b for a, b in zip(leaves(new_model), init)]

    def step_errors(got, ref):
        (loss, norm, updates), (ref_loss, ref_norm, ref_updates) = got, ref
        top = max(float(np.max(np.abs(u))) for u in ref_updates)
        return {
            "loss": rel(loss, ref_loss), "grad_norm": rel(norm, ref_norm),
            # Parameter updates, per leaf relative to that leaf (floored at 1% of the largest update).
            "updates": max(float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-2 * top)
                           for a, b in zip(updates, ref_updates, strict=True)),
        }

    base = get_preset("tiny-hybrid").replace(compute_dtype="float32")
    assert {"gated_deltanet", "attention"} <= set(base.resolved_layer_types), base.resolved_layer_types
    assert base.mtp is not None and base.mtp.depth > 0
    single = train_once(base, None)
    for strategy, mesh in (("ring", mesh8), ("ulysses", mesh24)):  # tiny-hybrid has 4 heads: Ulysses on cp=4
        cfg = base.replace(
            attention=dataclasses.replace(base.attention, context_parallel_axis="cp",
                                          context_parallel_strategy=strategy),
            linear_attention=dataclasses.replace(base.linear_attention, context_parallel_axis="cp"),
        )
        report[f"train_step/hybrid/{strategy}"] = step_errors(train_once(cfg, mesh), single)
    # The documented form: jax.make_mesh with Auto axes (the model-level switch needs them).
    mesh_doc = jax.make_mesh((2, 4), ("dp", "cp"), axis_types=(AxisType.Auto,) * 2)
    report["train_step/hybrid/ulysses/make_mesh_auto"] = step_errors(train_once(cfg, mesh_doc), single)
    return {"f32": report, "bf16": bf16}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matmul-precision", default=None, help='e.g. "highest" on GPUs (no TF32)')
    args = parser.parse_args(argv)
    import jax

    if args.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    report = run_checks()
    report["platform"] = jax.devices()[0].platform
    print("CP_ATTENTION " + json.dumps(report), flush=True)
    bad = failures(report)
    if bad:
        print("FAILED " + json.dumps(bad, indent=1), file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
