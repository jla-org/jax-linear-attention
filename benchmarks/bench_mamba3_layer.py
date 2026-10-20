# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full Mamba-3 layer benchmark: ``jla.layers.ssm.Mamba3`` / ``jla.layers.mamba3_mimo.Mamba3MIMO`` vs the
official state-spaces/mamba ``Mamba3`` module (``mamba_ssm.modules.mamba3``), SISO and MIMO.

Both layers have bfloat16 parameters and the official defaults: ``d_state=128``, ``headdim=64``,
``expand=2``, one B/C group, ``rope_fraction=0.5``, no output norm, chunk ``64 / R``. Each framework runs
in its own process and virtualenv:

    python benchmarks/bench_mamba3_layer.py jax   --out benchmarks/results/mamba3_layer_jax.json
    python benchmarks/bench_mamba3_layer.py torch --out benchmarks/results/mamba3_layer_torch.json

``fwd`` is the layer forward; ``fwd+bwd`` adds the backward with gradients for the input and every
parameter. ``kernel_ms`` is the CUPTI device time of all kernels of one call (JAX: Mosaic
``profiler.measure``, median of ``--rounds`` calls; torch: ``torch.profiler``, mean of ``--rounds`` calls
after a wait and a warmup step); ``ms`` is wall clock, the median over ``--rounds`` rounds of ``--iters``
back-to-back calls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

# B, T, d_model, MIMO rank (1 = SISO)
SHAPES = [(8, 2048, 768, 1), (4, 4096, 1536, 1), (8, 2048, 768, 4), (4, 4096, 1536, 4), (8, 2048, 768, 2)]


def gpu_status() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", visible, "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return out.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


def _median_loop(call, sync, iters: int, rounds: int) -> float:
    times = []
    for _ in range(rounds):
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            out = call()
        sync(out)
        times.append((time.perf_counter() - start) / iters * 1e3)
    return statistics.median(times)


def _inputs(batch: int, seq_len: int, d_model: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, seq_len, d_model), dtype=np.float32)
    do = rng.standard_normal((batch, seq_len, d_model), dtype=np.float32)
    return u, do


def run_jax(args) -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.experimental.mosaic.gpu import profiler

    from jla.layers.mamba3_mimo import Mamba3MIMO
    from jla.layers.ssm import Mamba3
    from jla.models.config import SSMConfig

    def sync(out=None):
        if out is not None:
            jax.block_until_ready(out)

    results = []
    for shape in args.shapes:
        batch, seq_len, d_model, rank = shape
        row = dict(shape=dict(zip(("B", "T", "d_model", "R"), shape)), gpu_before=gpu_status())
        for impl in args.impls:
            cfg = SSMConfig(mimo_rank=rank, chunk_size=max(16, 64 // rank), implementation=impl)
            cls = Mamba3MIMO if rank > 1 else Mamba3
            layer = cls(d_model, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5, param_dtype=jnp.bfloat16)
            dyn, static = eqx.partition(layer, eqx.is_inexact_array)
            u, do = (jnp.asarray(a, jnp.bfloat16) for a in _inputs(batch, seq_len, d_model))

            def fwd(dyn, u):
                return eqx.combine(dyn, static)(u)

            def fwd_bwd(dyn, u, do):
                out, vjp = jax.vjp(fwd, dyn, u)
                return vjp(do)

            for mode, fn, ins in (("fwd", fwd, (dyn, u)), ("fwd+bwd", fwd_bwd, (dyn, u, do))):
                label = f"ours[{impl}] {mode}"
                try:
                    f = jax.jit(fn)
                    jax.block_until_ready(f(*ins))
                    wall = _median_loop(lambda: f(*ins), sync, args.iters, args.rounds)
                    kernels = [profiler.measure(f)(*ins)[1] for _ in range(args.rounds)]
                    _, breakdown = profiler.measure(f, aggregate=False)(*ins)
                    try:
                        top = sorted(breakdown, key=lambda kv: -kv[1])[:10]
                    except (TypeError, IndexError, KeyError):
                        top = str(breakdown)[:2000]
                    row[label] = dict(ms=wall, kernel_ms=statistics.median(kernels), kernels=top)
                except Exception as err:  # noqa: BLE001 - record OOMs etc.
                    row[label] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
                print(shape, label, {k: v for k, v in row[label].items() if k != "kernels"}, flush=True)
        row["gpu_after"] = gpu_status()
        results.append(row)
    return dict(framework="jax", jax=jax.__version__, device=str(jax.devices()[0]), results=results)


def run_torch(args) -> dict:
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, schedule

    from mamba_ssm.modules.mamba3 import Mamba3

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    dev = torch.device("cuda")

    def sync(out=None):
        torch.cuda.synchronize()

    def kernel_times(fn):
        active = args.rounds
        with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=active)) as prof:
            for _ in range(active + 2):
                fn()
                torch.cuda.synchronize()
                prof.step()
        # Mean time per recorded launch times launches per call: the profiler can lose whole calls' records
        # (bench_mamba3.py, kernel_times).
        totals: dict[str, tuple] = {}
        for e in prof.events():
            if e.device_type == DeviceType.CUDA:
                total, count = totals.get(e.name, (0.0, 0))
                totals[e.name] = (total + e.device_time_total / 1e3, count + 1)
        per_call = {name: total / count * -(-count // active) for name, (total, count) in totals.items()}
        return sum(per_call.values()), sorted(per_call.items(), key=lambda kv: -kv[1])[:10]

    results = []
    for shape in args.shapes:
        batch, seq_len, d_model, rank = shape
        row = dict(shape=dict(zip(("B", "T", "d_model", "R"), shape)), gpu_before=gpu_status())
        torch.manual_seed(0)
        m = Mamba3(d_model, d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=0.5, is_mimo=rank > 1,
                   mimo_rank=max(rank, 1), chunk_size=64 // rank, device=dev, dtype=torch.bfloat16)
        params = list(m.parameters())
        u, do = (torch.from_numpy(a).to(dev, torch.bfloat16) for a in _inputs(batch, seq_len, d_model))
        ug = u.clone().requires_grad_(True)

        def fwd():
            with torch.no_grad():
                return m(u)

        def fwd_bwd():
            out = m(ug)
            return torch.autograd.grad(out, [ug] + params, grad_outputs=do)

        for mode, fn in (("fwd", fwd), ("fwd+bwd", fwd_bwd)):
            label = f"official {mode}"
            try:
                for _ in range(3):
                    fn()
                kernel_ms, breakdown = kernel_times(fn)
                row[label] = dict(ms=_median_loop(fn, sync, args.iters, args.rounds), kernel_ms=kernel_ms,
                                  kernels=breakdown)
            except Exception as err:  # noqa: BLE001
                row[label] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
            print(shape, label, {k: v for k, v in row[label].items() if k != "kernels"}, flush=True)
        row["gpu_after"] = gpu_status()
        results.append(row)
        del m, params, u, do, ug
        torch.cuda.empty_cache()
    return dict(framework="torch", torch=torch.__version__, device=torch.cuda.get_device_name(), results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("framework", choices=["jax", "torch"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--shapes", type=lambda s: [tuple(int(v) for v in t.split(",")) for t in s.split(";")],
                        default=SHAPES, help='"B,T,d_model,R;..."')
    parser.add_argument("--impls", nargs="*", default=["mosaic_gpu"], help="jax: SSM implementations to time")
    parser.add_argument("--memory-fraction", type=float, default=0.25, help="torch per-process memory cap")
    args = parser.parse_args()
    result = run_jax(args) if args.framework == "jax" else run_torch(args)
    result.update(iters=args.iters, rounds=args.rounds, time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
