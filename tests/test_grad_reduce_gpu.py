# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``grad_reduce = "step"`` on two H100s with the Mosaic GPU kernels: one GPU's micro-steps, one gradient reduction.

Small versions of the 1.3B end-to-end models (``benchmarks/end_to_end_1p3b.md``) as the 8-GPU runs use them: the
transformer with the Mosaic GPU attention, FP8 delayed scaling with the fused SwiGLU, the forward-gradient cross
entropy, no recompute and unrolled layers; Gated DeltaNet with the Mosaic GPU kernels, FP8 and full remat over a
layer scan. One optimizer step (SGD, learning rate 1, so the update is the gradient) on 8 packed rows of 512 tokens.

The GEMMs' rounding depends on their shapes (cuBLASLt picks the algorithm from the row count; FP8 GEMMs most), so
the reference is one GPU running the same calls: 4 micro-steps of 2 rows, where the 2 GPUs run 2 micro-steps of
2 rows each (micro-step ``m`` of GPU ``d`` is one GPU's micro-step ``2 m + d``). ``grad_reduce = "step"`` then gives
the same loss, gradient (to float32 summation noise: the same partial gradients added in another order) and FP8
states (the same maxima) as that GPU. The plain FSDP step (``"micro_step"``) is reported for comparison. The fused FP8
SwiGLU passes take their Pallas kernels inside the per-device step's fully manual ``shard_map`` (one GPU's call),
not outside it and not in the pipeline-parallel stages. With ``update_guard = "select"`` both reductions apply the
update of ``lax.cond`` (to float32 noise).
"""

from __future__ import annotations

import contextlib
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.config import MeshConfig
from jla.distributed.kernel_sharding import per_device_step, use_rules
from jla.distributed.mesh import build_mesh, plan_from_config
from jla.distributed.sharding import grouped_batch_sharding, param_shardings, resolve_rules
from jla.models import build_model, get_preset, logical_axes
from jla.ops.mosaic_gpu._common import is_hopper
from jla.training.fp8 import fp8_state_mask
from jla.training.step import LossConfig, build_train_step, lm_loss

pytestmark = pytest.mark.skipif(not is_hopper() or jax.device_count() < 2, reason="needs two H100s")

SEQ, ROWS = 512, 8
_FP8 = {"fp8": "delayed", "fp8_min_features": 512, "fp8_fused": True, "fp8_amax_history": 8}


def _transformer():
    base = get_preset("transformer-1p3b")
    return base.replace(
        vocab_size=8192, d_model=512, n_layers=2, d_ff=1408, remat="none", lean_vjp=True, scan_layers=False,
        attention=dataclasses.replace(base.attention, num_heads=8, num_kv_heads=2, implementation="mosaic_gpu"),
        **_FP8,
    )


def _gdn():
    base = get_preset("gdn-hd128-1p3b")
    return base.replace(
        vocab_size=8192, d_model=512, n_layers=2, d_ff=1408, layer_types=("gated_deltanet",) * 2,
        linear_attention=dataclasses.replace(base.linear_attention, num_k_heads=4, num_v_heads=4,
                                             implementation="mosaic_gpu"),
        **_FP8,
    )


def _rows(vocab: int) -> dict[str, jax.Array]:
    """8 rows of 2-3 packed documents each (positions restart and segment ids change at each document)."""
    tokens = jax.random.randint(jax.random.key(1), (ROWS, SEQ + 1), 2, vocab)
    starts = np.zeros((ROWS, SEQ), np.int32)
    for r in range(ROWS):
        starts[r, [0, 100 + 37 * r, 300 + 11 * r]] = 1
    segment = np.cumsum(starts, axis=1).astype(np.int32)
    begin = np.maximum.accumulate(np.where(starts == 1, np.arange(SEQ), 0), axis=1)
    positions = (np.arange(SEQ)[None, :] - begin).astype(np.int32)
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:], "segment_ids": jnp.asarray(segment),
            "positions": jnp.asarray(positions)}


def _one_step(config, devices: int, grad_accum: int, reduce: str, guard: str = "cond"):
    plan = plan_from_config(MeshConfig(dp=1, fsdp=devices, tp=1), devices)
    mesh = build_mesh(plan, jax.devices()[:devices])
    rules = resolve_rules()
    model = build_model(config, key=jax.random.key(0))
    params = eqx.filter(model, eqx.is_inexact_array)
    shardings = param_shardings(params, logical_axes(model), mesh, rules)
    model = eqx.combine(jax.device_put(params, shardings), model)
    batch_sharding = grouped_batch_sharding(mesh, rules)
    batch = {k: jax.device_put(v.reshape(grad_accum, ROWS // grad_accum, SEQ), batch_sharding)
             for k, v in _rows(config.vocab_size).items()}
    optimizer = optax.sgd(1.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss = lm_loss(LossConfig(impl="auto", grad_in_forward=True))
    # The context mesh also on one GPU: with two GPUs visible, kernels that take one device's call check for a
    # context mesh whose axes all have size 1 (jla.distributed.kernel_sharding.per_device_call). The fused FP8
    # SwiGLU takes its one-GPU kernels with several GPUs visible only inside per_device_step (the per-device loop
    # of grad_reduce = "step"), so the one-GPU reference is marked the same way and runs the kernels of one GPU.
    one_gpu = per_device_step() if devices == 1 else contextlib.nullcontext()
    with jax.set_mesh(mesh), use_rules(rules), one_gpu:
        step = build_train_step(optimizer, loss, batch_sharding=batch_sharding, param_shardings=shardings,
                                scan_unroll=1, donate=False, grad_reduce=reduce,
                                update_guard=guard)
        new_model, _, metrics, _ = step(model, opt_state, batch, jax.random.key(2))
    before = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    after = jax.tree.leaves(eqx.filter(new_model, eqx.is_inexact_array))
    mask = jax.tree.leaves(fp8_state_mask(eqx.filter(model, eqx.is_inexact_array)))
    grads = [np.asarray(b, np.float32) - np.asarray(a, np.float32)
             for a, b, s in zip(before, after, mask, strict=True) if not s]
    states = [np.asarray(b) for b, s in zip(after, mask, strict=True) if s]
    return {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}, grads, states


def _relative(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return max(float(np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-30)) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("name", ["transformer", "gdn"])
def test_step_reduction_is_one_gpu_with_the_gradient_reduced_once(name):
    config = {"transformer": _transformer, "gdn": _gdn}[name]()
    single, single_grads, single_states = _one_step(config, 1, 4, "micro_step")
    once, once_grads, once_states = _one_step(config, 2, 2, "step")
    plain, plain_grads, plain_states = _one_step(config, 2, 2, "micro_step")
    report = {
        "loss": (single["loss"], once["loss"], plain["loss"]),
        "grad_norm": (single["grad_norm"], once["grad_norm"], plain["grad_norm"]),
        "grad_rel_step_vs_one_gpu": _relative(single_grads, once_grads),
        "grad_rel_micro_step_vs_one_gpu": _relative(single_grads, plain_grads),
        "fp8_state_rel_step": _relative(single_states, once_states),
        "fp8_state_rel_micro_step": _relative(single_states, plain_states),
    }
    print(f"GRAD_REDUCE_GPU {name} {report}")
    assert once["step_applied"] and single["step_applied"]
    np.testing.assert_allclose(once["loss"], single["loss"], rtol=1e-6)
    np.testing.assert_allclose(once["grad_norm"], single["grad_norm"], rtol=1e-5)
    assert report["grad_rel_step_vs_one_gpu"] < 1e-4, report  # float32 sums of the same partials
    assert single_states and report["fp8_state_rel_step"] == 0.0, report  # the same maxima


@pytest.mark.parametrize("reduce", ["step", "micro_step"])
def test_select_guard_applies_the_cond_update_on_two_gpus(reduce):
    """``update_guard = "select"`` (the rejected-step rule without a conditional, which lets the host dispatch ahead
    on several GPUs) applies the update of ``lax.cond`` on two GPUs: the same loss, the same update (SGD with
    learning rate 1: the gradient) to float32 noise, the same FP8 states."""
    config = _transformer()
    cond, cond_grads, cond_states = _one_step(config, 2, 2, reduce)
    select, select_grads, select_states = _one_step(config, 2, 2, reduce, guard="select")
    rel = _relative(cond_grads, select_grads)
    print(f"UPDATE_GUARD_GPU {reduce} loss {cond['loss']!r} {select['loss']!r} update_rel {rel:.3g}")
    assert cond["step_applied"] and select["step_applied"]
    np.testing.assert_allclose(select["loss"], cond["loss"], rtol=1e-6)
    np.testing.assert_allclose(select["grad_norm"], cond["grad_norm"], rtol=1e-6)
    assert rel < 1e-5, rel  # the same update; XLA fuses it differently around the select
    assert cond_states and _relative(cond_states, select_states) == 0.0


def test_fused_fp8_swiglu_kernels_run_only_in_the_per_device_step():
    """With several GPUs the fused FP8 SwiGLU's "auto" dispatch keeps its plain-JAX passes (XLA cannot partition the
    Pallas calls), except in the per-device micro-step loop of ``grad_reduce = "step"`` (a fully manual
    ``shard_map`` traced inside ``per_device_step``), where a call is one GPU's. Other fully manual bodies, such as
    the pipeline-parallel stages, keep the plain-JAX passes."""
    from jax.sharding import PartitionSpec as P

    from jla.layers.fp8_swiglu import _use_pallas

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("fsdp",))
    seen = []

    def body(x):
        seen.append(_use_pallas(1024, 1408, "auto"))
        return x

    def run(x):
        return jax.shard_map(body, mesh=mesh, in_specs=P("fsdp"), out_specs=P("fsdp"), axis_names=frozenset({"fsdp"}),
                             check_vma=False)(x)

    jax.jit(run)(jnp.zeros(2))
    with per_device_step():
        jax.jit(run)(jnp.zeros(4))  # another shape: a new trace
        assert not _use_pallas(1024, 1408, "auto")  # two devices and no manual mesh: not one device's call
    assert seen == [False, True]
    assert not _use_pallas(1024, 1408, "auto")


def test_pipeline_parallel_fp8_run_keeps_the_plain_jax_swiglu(tmp_path, monkeypatch):
    """A two-stage pipeline-parallel FP8 run on two GPUs: its stages run in a fully manual ``shard_map`` but are not
    the per-device step, so the fused FP8 SwiGLU keeps its plain-JAX passes, as before ``grad_reduce = "step"``
    existed, and the run trains with finite losses."""
    from _toy_model import experiment

    from jla.layers import fp8_swiglu
    from jla.training.trainer import Trainer

    seen = []
    original = fp8_swiglu._use_pallas

    def spy(m, f, impl):
        seen.append(original(m, f, impl))
        return seen[-1]

    monkeypatch.setattr(fp8_swiglu, "_use_pallas", spy)
    base = _transformer()
    # Pipeline stages are slices of the layer scan.
    model = base.replace(scan_layers=True, attention=dataclasses.replace(base.attention, implementation="xla"))
    train = {"max_steps": 2, "log_every": 1, "seq_len": SEQ, "micro_batch_per_replica": 4, "grad_accum": 1,
             "final_eval": False, "save_final": False}
    config = experiment(str(tmp_path / "pp2"), model=model, train=train,
                        mesh={"fsdp": 1, "pp": 2, "pp_microbatches": 2})
    result = Trainer(config, devices=jax.devices()[:2]).run()
    assert seen and not any(seen), seen
    losses = [record["loss"] for record in result.history]
    assert len(losses) == 2 and np.all(np.isfinite(losses)), result.history
