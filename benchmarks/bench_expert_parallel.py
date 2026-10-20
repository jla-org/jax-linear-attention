# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel MoE on N GPUs: equality with the layer without EP, Trainer runs, timing.

    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/bench_expert_parallel.py --gpus 2 --out ep2.json
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmarks/bench_expert_parallel.py --gpus 8 --out ep8.json

Sections (``--sections``, all by default; results are written after each one):

``equality``
    One MoE layer, mesh ``ep = N`` (and ``fsdp = 2 x ep = N / 2`` for N >= 4):
    outputs, router statistics and gradients of the EP layer against the same
    layer without EP on the same mesh. float32 runs at the highest matmul
    precision; bfloat16 compares relative error norms, also against a float32
    oracle. Dropless and capacity routing, both ``ragged_dot_impl`` values.
``trainer_tiny``
    ``tiny-moe`` with 8 experts in float32 (highest precision): ``mesh.ep`` runs
    against ``mesh.fsdp = N`` without EP, step for step, and the final parameters.
``trainer_125m``
    The ``moe-125m`` preset (16 experts, top-2, capacity 1.25, one shared expert),
    float32 for 3 steps (losses must match) and bfloat16 for timed steps: EP
    against the FSDP baseline (experts sharded like dense weights, no EP) and the
    DP baseline (every weight replicated); and with dropless routing, FSDP and DP
    on the Mosaic GPU grouped GEMM against EP with it. ``--only-125m`` picks
    layouts, ``--repeat R`` interleaves R timed passes over them.
``timing``
    Forward + backward of one routed-expert layer at ``moe-125m`` expert shapes
    per layout: ``dp`` (experts replicated, tokens split), ``gspmd`` (experts
    sharded over ``ep``, the layer without EP partitioned by XLA), ``ep``
    (all-to-all, worst-case blocks), ``ep_f2`` and ``ep_f1.25``
    (``expert_parallel_buffer_factor``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import tempfile
import time
import traceback
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jla.config import DataConfig, ExperimentConfig, MeshConfig, OptimConfig, RuntimeConfig, TrainConfig
from jla.distributed.sharding import DEFAULT_RULES, param_shardings
from jla.layers.moe import MoE
from jla.models import get_preset, logical_axes
from jla.models.config import MoEConfig
from jla.training.trainer import Trainer

AXES = ("dp", "fsdp", "tp", "ep")
TOKENS = P(("dp", "fsdp", "ep"))


def _mesh(*, dp: int = 1, fsdp: int = 1, ep: int = 1) -> Mesh:
    devices = np.asarray(jax.devices()[: dp * fsdp * ep])
    return Mesh(devices.reshape(dp, fsdp, 1, ep), AXES)


def _place(layer: MoE, mesh: Mesh):
    params, static = eqx.partition(layer, eqx.is_array)
    return jax.device_put(params, param_shardings(params, logical_axes(layer), mesh, DEFAULT_RULES)), static


def _fwd_bwd(static, cot):
    def loss(params, x):
        y, stats = eqx.combine(params, static)(x)
        return jnp.sum(y.astype(jnp.float32) * cot) + stats.aux_loss, (y, stats)

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))


def _layer_run(layer: MoE, mesh: Mesh, x: jax.Array, cot: jax.Array):
    with jax.set_mesh(mesh):
        params, static = _place(layer, mesh)
        (_, (y, stats)), grads = _fwd_bwd(static, cot)(params, x)
    return y, stats, jax.tree.leaves(grads)


def equality(cfg: MoEConfig, dtype: str, mesh: Mesh, tokens: int = 8192, d_model: int = 256) -> dict:
    key = jax.random.key(3)
    plain = dataclasses.replace(cfg, expert_parallel_axis=None, expert_parallel_buffer_factor=None, ragged_dot_impl="xla")
    # The same grouped GEMM kernel without EP where the layer allows it (dropless routing).
    same_kernel = plain if cfg.capacity_factor is not None else dataclasses.replace(plain, ragged_dot_impl=cfg.ragged_dot_impl)
    base = MoE(d_model, same_kernel, key=key, std=0.05, compute_dtype=dtype)
    ep = MoE(d_model, cfg, key=key, std=0.05, compute_dtype=dtype)
    x32 = jax.random.normal(jax.random.key(1), (8, tokens // 8, d_model), jnp.float32)
    x = jax.device_put(x32.astype(dtype), NamedSharding(mesh, TOKENS))
    cot = jax.device_put(jax.random.normal(jax.random.key(2), x.shape, jnp.float32), NamedSharding(mesh, TOKENS))
    y0, s0, g0 = _layer_run(base, mesh, x, cot)
    y1, s1, g1 = _layer_run(ep, mesh, x, cot)
    result = {
        "out_rel_err": _rel(y1, y0),
        "grad_rel_err": max(_rel(b, a) for a, b in zip(g0, g1, strict=True)),
        "aux_diff": abs(float(s1.aux_loss) - float(s0.aux_loss)),
        "dropped": [float(s0.dropped_fraction), float(s1.dropped_fraction)],
    }
    if dtype != "float32":  # both against a float32 oracle (same weights, highest precision)
        oracle = MoE(d_model, plain, key=key, std=0.05, compute_dtype="float32")
        x_f32 = jax.device_put(x.astype(jnp.float32), NamedSharding(mesh, TOKENS))
        with jax.default_matmul_precision("highest"):
            y2, _, g2 = _layer_run(oracle, mesh, x_f32, cot)
        result["out_rel_err_vs_f32"] = {"ep": _rel(y1, y2), "no_ep": _rel(y0, y2)}
        result["grad_rel_err_vs_f32"] = {
            "ep": max(_rel(b, a) for a, b in zip(g2, g1, strict=True)),
            "no_ep": max(_rel(b, a) for a, b in zip(g2, g0, strict=True)),
        }
    return result


def section_equality(n: int) -> dict:
    report: dict[str, Any] = {}
    dropless = MoEConfig(num_experts=16, top_k=2, d_ff_expert=128, z_loss_coef=1e-3, expert_parallel_axis="ep")
    capacity = dataclasses.replace(dropless, capacity_factor=1.0)
    cases = {
        "dropless_f32_xla": (dropless, "float32"),
        "capacity_f32_xla": (capacity, "float32"),
        "dropless_bf16_xla": (dropless, "bfloat16"),
        "capacity_bf16_xla": (capacity, "bfloat16"),
        "dropless_bf16_mosaic": (dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu"), "bfloat16"),
        "capacity_bf16_mosaic": (dataclasses.replace(capacity, ragged_dot_impl="mosaic_gpu"), "bfloat16"),
        "dropless_bf16_mosaic_f2": (
            dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu", expert_parallel_buffer_factor=8.0), "bfloat16"),
    }
    meshes = {f"ep{n}": _mesh(ep=n)}
    if n >= 4:
        meshes[f"fsdp2_ep{n // 2}"] = _mesh(fsdp=2, ep=n // 2)
    for mesh_name, mesh in meshes.items():
        for name, (cfg, dtype) in cases.items():
            label = f"{mesh_name}/{name}"
            try:
                with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
                    report[label] = equality(cfg, dtype, mesh)
            except Exception:
                report[label] = "error: " + traceback.format_exc()[-1500:]
            print(label, report[label], flush=True)
    return report


#: Strict comparisons use SGD (Adam would amplify rounding noise in near-zero gradients).
STRICT_OPTIM = OptimConfig(name="sgd", lr=0.1, b1=0.9, warmup_steps=0, weight_decay=0.0, schedule="constant")
TIMED_OPTIM = OptimConfig(name="adamw", lr=3e-4, warmup_steps=0, weight_decay=0.1, schedule="constant")


def _experiment(model, mesh: dict, out_dir: str, *, steps: int, seq_len: int, micro: int, precision: str,
                optim: OptimConfig) -> ExperimentConfig:
    return ExperimentConfig(
        model=model,
        name="ep-bench",
        optim=optim,
        train=TrainConfig(max_steps=steps, seq_len=seq_len, micro_batch_per_replica=micro, grad_accum=1, log_every=1,
                          eval_every=0, save_every=0, save_final=False, final_eval=False,
                          ce_compute_dtype="float32" if model.compute_dtype == "float32" else "bfloat16",
                          output_dir=out_dir),
        data=DataConfig(num_workers=0, synthetic_samples=8192),
        mesh=MeshConfig(**mesh),
        runtime=RuntimeConfig(matmul_precision=precision),
    )


def _train(model, mesh: dict, devices: int, *, steps: int, seq_len: int, micro: int, precision: str,
           optim: OptimConfig = STRICT_OPTIM) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        config = _experiment(model, mesh, tmp, steps=steps, seq_len=seq_len, micro=micro, precision=precision,
                             optim=optim)
        context = jax.default_matmul_precision(precision) if precision != "default" else contextlib.nullcontext()
        with context:
            result = Trainer(config, devices=jax.devices()[:devices]).run()
    times = [float(h["step_time"]) for h in result.history]
    return {
        "loss": [float(h["loss"]) for h in result.history],
        "grad_norm": [float(h["grad_norm"]) for h in result.history],
        "step_time_ms": [1e3 * t for t in times],
        "median_step_ms_after_2": 1e3 * float(np.median(times[2:])) if len(times) > 2 else None,
        "tokens_per_step": devices * micro * seq_len,
        "params": [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(result.model, eqx.is_inexact_array))],
        "specs": sorted({str(leaf.sharding.spec) for leaf in jax.tree.leaves(eqx.filter(result.model, eqx.is_inexact_array))
                         if leaf.ndim >= 4}),
    }


def _strip(run: dict) -> dict:
    return {k: v for k, v in run.items() if k != "params"}


def section_trainer_tiny(n: int) -> dict:
    base = get_preset("tiny-moe").replace(compute_dtype="float32")
    report: dict[str, Any] = {}
    for label, moe in (("dropless", dataclasses.replace(base.moe, num_experts=8)),
                       ("capacity", dataclasses.replace(base.moe, num_experts=8, capacity_factor=1.0))):
        model = base.replace(moe=moe)
        ep_model = base.replace(moe=dataclasses.replace(moe, expert_parallel_axis="ep"))
        ref = _train(model, {"fsdp": n}, n, steps=5, seq_len=64, micro=2, precision="highest")
        report[f"{label}/fsdp{n}"] = _strip(ref)
        meshes = {f"ep{n}": {"fsdp": 1, "ep": n}}
        if n >= 4:
            meshes[f"fsdp2_ep{n // 2}"] = {"fsdp": 2, "ep": n // 2}
        for mesh_name, mesh in meshes.items():
            run = _train(ep_model, mesh, n, steps=5, seq_len=64, micro=2, precision="highest")
            entry = _strip(run)
            entry["max_loss_rel_diff"] = max(abs(a - b) / abs(b) for a, b in zip(run["loss"], ref["loss"], strict=True))
            entry["max_param_abs_diff"] = max(float(np.max(np.abs(a - b))) for a, b in zip(run["params"], ref["params"], strict=True))
            report[f"{label}/{mesh_name}"] = entry
            print(label, mesh_name, entry, flush=True)
    return report


def section_trainer_125m(n: int, seq_len: int, micro: int, steps: int, only: tuple[str, ...] = (),
                         repeat: int = 1, preset_name: str = "moe-125m") -> dict:
    """``only``: run just these bfloat16 layouts (and skip the float32 check); ``preset_name``: the
    model (``--trainer-preset``, default moe-125m)."""
    report: dict[str, Any] = {"preset": preset_name}
    preset = get_preset(preset_name)
    ep_moe = dataclasses.replace(preset.moe, expert_parallel_axis="ep")
    if not only:
        report.update(_check_125m_f32(n, seq_len, micro, preset, ep_moe))
    return report | _time_125m(n, seq_len, micro, steps, only, preset, ep_moe, repeat)


def _check_125m_f32(n: int, seq_len: int, micro: int, preset, ep_moe) -> dict:
    report: dict[str, Any] = {}
    # float32, highest precision: EP and the FSDP baseline take the same steps.
    f32 = preset.replace(compute_dtype="float32")
    ref = _train(f32, {"fsdp": n}, n, steps=3, seq_len=seq_len, micro=max(1, micro // 2), precision="highest")
    run = _train(f32.replace(moe=ep_moe), {"fsdp": 1, "ep": n}, n, steps=3, seq_len=seq_len, micro=max(1, micro // 2),
                 precision="highest")
    report["f32/fsdp_baseline"] = _strip(ref)
    report["f32/ep"] = _strip(run)
    report["f32/max_loss_rel_diff"] = max(abs(a - b) / abs(b) for a, b in zip(run["loss"], ref["loss"], strict=True))
    report["f32/max_param_rel_diff"] = max(_rel(a, b) for a, b in zip(run["params"], ref["params"], strict=True))
    print("125m f32", {k: v for k, v in report.items() if not isinstance(v, dict)}, flush=True)
    return report


def _time_125m(n: int, seq_len: int, micro: int, steps: int, only: tuple[str, ...], preset, ep_moe,
               repeat: int = 1) -> dict:
    report: dict[str, Any] = {}
    # bfloat16 timed runs (preset dtypes).
    layouts = {
        "fsdp_baseline": (preset, {"fsdp": n}),
        "dp_baseline": (preset, {"dp": n, "fsdp": 1}),  # every weight replicated: no weight all-gathers
        "ep": (preset.replace(moe=ep_moe), {"fsdp": 1, "ep": n}),
        "ep_mosaic_f2": (preset.replace(moe=dataclasses.replace(ep_moe, ragged_dot_impl="mosaic_gpu",
                                                                expert_parallel_buffer_factor=2.0)), {"fsdp": 1, "ep": n}),
    }
    if n >= 4:
        layouts[f"fsdp2_ep{n // 2}_mosaic_f2"] = (layouts["ep_mosaic_f2"][0], {"fsdp": 2, "ep": n // 2})
    # Dropless routing (other semantics than the preset's capacity 1.25): the strongest layer without
    # EP (Mosaic GPU grouped GEMM, FSDP) against EP with the same kernel.
    dropless = dataclasses.replace(preset.moe, capacity_factor=None, ragged_dot_impl="mosaic_gpu")
    layouts["dropless/fsdp_mosaic_baseline"] = (preset.replace(moe=dropless), {"fsdp": n})
    layouts["dropless/ep_mosaic_f2"] = (preset.replace(moe=dataclasses.replace(
        dropless, expert_parallel_axis="ep", expert_parallel_buffer_factor=2.0)), {"fsdp": 1, "ep": n})
    # The strongest layout without EP: every weight replicated, the same Mosaic GPU grouped GEMM.
    layouts["dropless/dp_mosaic_baseline"] = (preset.replace(moe=dropless), {"dp": n, "fsdp": 1})
    if n >= 4:
        layouts[f"dropless/fsdp2_ep{n // 2}_mosaic_f2"] = (layouts["dropless/ep_mosaic_f2"][0], {"fsdp": 2, "ep": n // 2})
    for rep in range(repeat):  # interleaved repeats: every layout sees the same drift in GPU load
        for name, (model, mesh) in layouts.items():
            if only and name not in only:
                continue
            key = f"bf16/{name}" if rep == 0 else f"bf16/{name}/rep{rep + 1}"
            try:
                run = _train(model, mesh, n, steps=steps, seq_len=seq_len, micro=micro, precision="default",
                             optim=TIMED_OPTIM)
                report[key] = _strip(run)
            except Exception:
                report[key] = "error: " + traceback.format_exc()[-1500:]
            print("125m bf16", key, report[key], flush=True)
    return report


def section_timing(n: int, tokens_per_gpu: int, iters: int, preset: str = "",
                   impls: tuple[str, ...] = ("xla", "mosaic_gpu")) -> dict:
    """Routed experts of moe-125m (16 experts, top-2, width 768), or of ``preset``'s MoE layer.

    ``impls``: the grouped GEMMs to time (``--timing-impls``). At shapes where XLA's ``ragged_dot``
    runs out of memory, leave it out: a device that failed inside a collective layout can leave its
    peer waiting at the all-to-all rendezvous until the process is killed.
    """
    d_model, base = 768, MoEConfig(num_experts=16, top_k=2, d_ff_expert=768)
    if preset:
        model = get_preset(preset)
        d_model = model.d_model
        base = MoEConfig(num_experts=model.moe.num_experts, top_k=model.moe.top_k, d_ff_expert=model.moe.d_ff_expert)
    report: dict[str, Any] = {"shape": {"tokens": tokens_per_gpu * n, "d_model": d_model, "experts": base.num_experts,
                                        "top_k": base.top_k, "d_ff_expert": base.d_ff_expert, "dtype": "bfloat16",
                                        "gpus": n, "preset": preset or "moe-125m"}}
    for impl in impls:
        cfg = dataclasses.replace(base, ragged_dot_impl=impl)
        ep_cfg = dataclasses.replace(cfg, expert_parallel_axis="ep")
        layouts = {
            "dp": (dict(dp=n), cfg),
            "gspmd": (dict(ep=n), cfg),
            "ep": (dict(ep=n), ep_cfg),
            "ep_f2": (dict(ep=n), dataclasses.replace(ep_cfg, expert_parallel_buffer_factor=2.0)),
            "ep_f1.25": (dict(ep=n), dataclasses.replace(ep_cfg, expert_parallel_buffer_factor=1.25)),
        }
        times: dict[str, Any] = {}
        for name, (axes, layer_cfg) in layouts.items():
            mesh = _mesh(**axes)
            layer = MoE(d_model, layer_cfg, key=jax.random.key(0), std=0.02, compute_dtype="bfloat16")
            x = jax.random.normal(jax.random.key(1), (8 * n, tokens_per_gpu // 8, d_model), jnp.float32)
            x = jax.device_put(x.astype(jnp.bfloat16), NamedSharding(mesh, TOKENS))
            cot = jax.device_put(jnp.ones(x.shape, jnp.float32), x.sharding)
            try:
                with jax.set_mesh(mesh):
                    params, static = _place(layer, mesh)
                    step = _fwd_bwd(static, cot)
                    for _ in range(3):
                        jax.block_until_ready(step(params, x))
                    samples = []
                    for _ in range(iters):
                        start = time.perf_counter()
                        jax.block_until_ready(step(params, x))
                        samples.append(time.perf_counter() - start)
                times[name] = round(1e3 * float(np.median(samples)), 3)
            except Exception as exc:  # report the failing layout and keep timing the others
                times[name] = f"error: {type(exc).__name__}: {str(exc)[:300]}"
            print("timing", impl, name, times[name], flush=True)
        report[f"ms_fwd_bwd_{impl}"] = times
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sections", default="equality,trainer_tiny,trainer_125m,timing")
    parser.add_argument("--seq-len", type=int, default=1024, help="moe-125m Trainer runs")
    parser.add_argument("--micro", type=int, default=8, help="moe-125m sequences per GPU (bfloat16 runs)")
    parser.add_argument("--steps", type=int, default=10, help="moe-125m bfloat16 steps")
    parser.add_argument("--only-125m", default="", help="comma list of moe-125m bfloat16 layouts (skips float32)")
    parser.add_argument("--repeat", type=int, default=1, help="interleaved repeats of the moe-125m bfloat16 layouts")
    parser.add_argument("--tokens-per-gpu", type=int, default=8192, help="timed layer")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--timing-preset", default="", help="time this preset's MoE layer shape instead of moe-125m's")
    parser.add_argument("--trainer-preset", default="moe-125m", help="model of the trainer_125m section")
    parser.add_argument("--timing-impls", default="xla,mosaic_gpu", help="grouped GEMMs of the timing section")
    args = parser.parse_args()
    n = args.gpus
    assert len(jax.devices()) >= n, jax.devices()
    report: dict[str, Any] = {"devices": [d.device_kind for d in jax.devices()[:n]], "gpus": n, "jax": jax.__version__}
    sections = {
        "equality": lambda: section_equality(n),
        "trainer_tiny": lambda: section_trainer_tiny(n),
        "trainer_125m": lambda: section_trainer_125m(n, args.seq_len, args.micro, args.steps,
                                                     tuple(filter(None, args.only_125m.split(","))), args.repeat,
                                                     args.trainer_preset),
        "timing": lambda: section_timing(n, args.tokens_per_gpu, args.iters, args.timing_preset,
                                         tuple(filter(None, args.timing_impls.split(",")))),
    }
    for name in args.sections.split(","):
        start = time.perf_counter()
        try:
            report[name] = sections[name]()
        except Exception:
            report[name] = "error: " + traceback.format_exc()[-3000:]
        report[f"{name}_seconds"] = round(time.perf_counter() - start, 1)
        print(name, "done", report[f"{name}_seconds"], "s", flush=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
