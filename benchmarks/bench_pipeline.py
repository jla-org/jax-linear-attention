# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline parallelism (``pp``) against FSDP/DP on the same devices: one compiled train step.

    python benchmarks/bench_pipeline.py --preset transformer-125m --batch 16 --seq 1024 \
        --layouts fsdp=2 dp=2 pp=2,M=2 pp=2,M=4 pp=2,M=8 --steps 10 --json out.json

A layout is ``axis=size`` items (``dp``, ``fsdp``, ``tp``, ``pp``) plus ``M=<microbatches>``
for the pipeline and ``head=rep`` to keep the LM head and loss replicated on every stage
(default ``head=pp``: sharded over the stages when the batch allows it); it runs on the
first ``dp * fsdp * tp * pp`` visible devices, so one invocation can time 2- and 4-device
layouts; a pipelined layout whose head ran sharded is timed again with ``head=rep`` (unless
``--no-replicated-head``). Every layout trains the same model (same init key, same batch):
parameters are sharded with the logical-axis rules (``layers`` over ``pp`` for pipelined
layouts), the step is ``build_train_step`` + ``lm_loss`` (with
``make_pipeline_forward`` when ``pp > 1``) and AdamW. Reported per layout: compile seconds,
first-step loss and gradient norm (the layouts compute the same function, so these agree up
to reduction order and bf16 rounding), the GPipe bubble fraction, the median ms/step and
tokens/s over ``--steps`` timed steps after two warm-up steps, and the peak device memory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time

import equinox as eqx
import jax
import numpy as np
import optax

from jla.distributed.mesh import build_mesh, resolve_mesh_plan
from jla.distributed.pipeline import (
    bubble_fraction,
    make_pipeline_forward,
    pipeline_rules,
    resolve_microbatches,
    runs_fully_manual,
    shards_head_over_stages,
)
from jla.distributed.sharding import DEFAULT_RULES, batch_axes, grouped_batch_sharding, opt_state_shardings, param_shardings
from jla.models import build_model, get_preset, logical_axes
from jla.training.step import LossConfig, build_train_step, lm_loss


def _layout(text: str) -> tuple[dict[str, int], int, str]:
    axes: dict[str, int] = {}
    micro, head = 0, "pp"
    for part in text.split(","):
        name, value = part.split("=")
        if name == "M":
            micro = int(value)
        elif name == "head" and value in ("pp", "rep"):
            head = value
        elif name in ("dp", "fsdp", "tp", "pp"):
            axes[name] = int(value)
        else:
            raise ValueError(f"layout {text!r}: unknown item {name!r}")
    return axes, micro, head


def _peak_gib(devices) -> float | None:
    peaks = []
    for device in devices:
        try:
            stats = device.memory_stats() or {}
        except Exception:  # noqa: BLE001 (CPU devices have no memory stats)
            return None
        if "peak_bytes_in_use" not in stats:
            return None
        peaks.append(stats["peak_bytes_in_use"])
    return round(max(peaks) / 2**30, 2) if peaks else None


def run_layout(cfg, layout: str, *, batch: int, seq: int, steps: int) -> dict:
    axes, micro, head = _layout(layout)
    count = math.prod(axes.values())
    devices = jax.devices()[:count]
    if len(devices) != count:
        raise ValueError(f"layout {layout!r} needs {count} devices; {len(jax.devices())} visible")
    plan = resolve_mesh_plan(count, **{"fsdp": 1, **axes})
    mesh = build_mesh(plan, devices)
    rules = pipeline_rules(DEFAULT_RULES) if plan.pp > 1 else dict(DEFAULT_RULES)
    model = build_model(cfg, key=jax.random.key(0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    shardings = param_shardings(params, logical_axes(model), mesh, rules)
    params = jax.device_put(params, shardings)
    tx = optax.adamw(3e-4, weight_decay=0.01)
    opt_sh = opt_state_shardings(jax.eval_shape(tx.init, params), params, shardings, mesh)
    opt_state = jax.jit(tx.init, out_shardings=opt_sh)(params)
    model = eqx.combine(params, static)
    row_axes = batch_axes(mesh, rules)
    forward = (make_pipeline_forward(mesh, num_microbatches=micro, batch_axes=row_axes, rules=rules, shard_head=head == "pp")
               if plan.pp > 1 else None)
    sharded_head = plan.pp > 1 and head == "pp" and shards_head_over_stages(batch, mesh, num_microbatches=micro, batch_axes=row_axes)
    batch_sharding = grouped_batch_sharding(mesh, rules)
    step = build_train_step(tx, lm_loss(LossConfig(), forward=forward), batch_sharding=batch_sharding,
                            param_shardings=shardings, opt_shardings=opt_sh)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(1, batch, seq + 1)).astype(np.int32)
    data = {"input_ids": jax.device_put(ids[..., :-1], batch_sharding), "labels": jax.device_put(ids[..., 1:], batch_sharding)}
    key = jax.random.key(1)
    start = time.perf_counter()
    model, opt_state, metrics, key = step(model, opt_state, data, key)
    first = {name: float(metrics[name]) for name in ("loss", "grad_norm")}
    compile_s = time.perf_counter() - start
    for _ in range(2):
        model, opt_state, metrics, key = step(model, opt_state, data, key)
    jax.block_until_ready(metrics["loss"])
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        model, opt_state, metrics, key = step(model, opt_state, data, key)
        jax.block_until_ready(metrics["loss"])
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    stages = plan.pp
    reported = float(metrics["pipeline/bubble_fraction"]) if stages > 1 else 0.0
    return {
        "layout": layout,
        "devices": count,
        "mesh": plan.axis_sizes,
        "microbatches": resolve_microbatches(stages, micro) if stages > 1 else 1,
        "bubble_fraction": round(bubble_fraction(stages, micro), 4) if stages > 1 else 0.0,
        "bubble_fraction_metric": round(reported, 4),
        "head": ("sharded over pp" if sharded_head else "replicated over pp") if stages > 1 else "data parallel",
        "manual_axes": ("all" if runs_fully_manual(cfg, batch, mesh, num_microbatches=micro, batch_axes=row_axes, rules=rules)
                        else "pp") if stages > 1 else "none",
        "compile_s": round(compile_s, 1),
        "first_loss": first["loss"],
        "first_grad_norm": first["grad_norm"],
        "ms_per_step": round(median * 1e3, 2),
        "ms_min": round(min(times) * 1e3, 2),
        "tokens_per_s": round(batch * seq / median),
        "peak_gib_per_device": _peak_gib(devices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", default="transformer-125m")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layouts", nargs="+", default=["fsdp=2", "pp=2,M=2", "pp=2,M=4", "pp=2,M=8"])
    parser.add_argument("--json", default="")
    parser.add_argument("--attention", default="", choices=("", "auto", "cudnn", "xla", "mosaic_gpu"),
                        help="attention implementation for every layout (pipelined GPU layouts need xla or mosaic_gpu)")
    parser.add_argument("--no-replicated-head", action="store_true",
                        help="do not also time head=rep for pipelined layouts whose head runs sharded over pp")
    args = parser.parse_args()
    cfg = get_preset(args.preset)
    if args.attention:
        cfg = cfg.replace(attention=dataclasses.replace(cfg.attention, implementation=args.attention))
    report = {"preset": args.preset, "attention": cfg.attention.implementation, "batch": args.batch, "seq": args.seq, "steps": args.steps,
              "device_kind": jax.devices()[0].device_kind, "visible_devices": len(jax.devices()),
              "jax": jax.__version__, "results": []}
    queue = list(args.layouts)
    while queue:
        layout = queue.pop(0)
        try:
            result = run_layout(cfg, layout, batch=args.batch, seq=args.seq, steps=args.steps)
        except Exception as exc:  # noqa: BLE001 (record the failure, keep timing the other layouts)
            result = {"layout": layout, "error": f"{type(exc).__name__}: {str(exc)[:600]}"}
        print(json.dumps(result), flush=True)
        report["results"].append(result)
        if args.json:  # rewrite after every layout so a dead node still leaves the finished rows
            with open(args.json, "w") as fh:
                json.dump(report, fh, indent=1)
        if result.get("head") == "sharded over pp" and "head=" not in layout and not args.no_replicated_head:
            queue.insert(0, f"{layout},head=rep")  # the same layout with the head on every stage


if __name__ == "__main__":
    main()
