# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The whole Mamba-2 mixer against the official module's training path, forward and forward + backward.

The official ``mamba_ssm.modules.mamba2.Mamba2`` trains through ``mamba_split_conv1d_scan_combined``
(``use_mem_eff_path=True``, its default): the in-projection GEMM, then one autograd function running the causal
conv1d, the SSD scan, the ``D`` skip, the gated RMSNorm and the out-projection. Both sides get the same
configuration (``d_state`` 128, expand 2, head dim 64, one B/C group, chunk 256, bf16, no bias) and time one call:
the forward (``torch.no_grad`` on the official side) and the forward + backward (gradients of the input and of
every parameter).

    ours:     python benchmarks/bench_mamba2_layer.py jax --out ours.json [--src TREE]
    official: VENV_MAMBA/bin/python benchmarks/bench_mamba2_layer.py torch --out official.json

Kernel time:

* JAX: CUPTI (``harness.median_kernel_ms``), the median over ``--iterations`` calls;
* torch: ``torch.profiler`` over ``--active`` calls after a wait and a warm-up step, every kernel's mean time per
  launch times its launches per call. The profiler sometimes loses whole calls' kernel records on our nodes
  (the in-projection GEMM of a call, or every kernel of some calls), which makes a plain sum over the recorded
  kernels divided by the number of calls too small; the per-launch means are not affected. Each entry keeps the
  number of records missing against ``launches x calls`` (``dropped_records``) and the plain sum.

Both sides also report the wall clock of ``--wall-calls`` back-to-back calls (CUDA events on the torch side,
``block_until_ready`` on the JAX side), which includes launch overheads (they bound the official forward at
``(8, 2048, 768)``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

SHAPES = [(4, 4096, 2048), (16, 4096, 2048), (1, 16384, 2048), (8, 2048, 768)]
CONFIG = "d_state 128, expand 2, head dim 64, ngroups 1, chunk 256, bf16"


def parse_shapes(text: str) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in item.split(",")) for item in text.split(";") if item]


def _top(kernels: dict[str, float], n: int) -> list[list]:
    return [[name[:120], round(ms, 4)] for name, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:n]]


def run_jax(args) -> dict:
    sys.path.insert(0, os.path.join(args.src, "benchmarks"))
    sys.path.insert(0, os.path.join(args.src, "src"))
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    import harness
    from jla.layers import ssm as ssm_mod
    from jla.models.config import SSMConfig

    entries = []
    for b, t, d in args.shapes:
        cfg = SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1)
        layer = ssm_mod.Mamba2(d, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5, compute_dtype="bfloat16")
        u = jax.random.normal(jax.random.key(1), (b, t, d), jnp.bfloat16)
        ct = jax.random.normal(jax.random.key(2), (b, t, d), jnp.bfloat16)
        params, static = eqx.partition(layer, eqx.is_inexact_array)

        def run(params, u):
            return eqx.combine(params, static)(u)

        @jax.jit
        def fwdbwd(params, u, ct):
            y, pullback = jax.vjp(run, params, u)
            return y, pullback(ct)

        for mode, fn, xs in (("fwd", jax.jit(run), (params, u)), ("fwdbwd", fwdbwd, (params, u, ct))):
            e = {"shape": [b, t, d], "mode": mode, "impl": "jla", "fused_glue": ssm_mod._fused_glue_applies(layer)}
            try:
                r = harness.median_kernel_ms(fn, *xs, iterations=args.iterations)
                e.update(median_ms=r["median_ms"], min_ms=r["min_ms"], iterations=r["iterations"])
                if args.breakdown:
                    e["kernels"] = _top(harness.kernel_breakdown_ms(fn, *xs, iterations=5), args.breakdown)
                walls = []
                for _ in range(3):
                    jax.block_until_ready(fn(*xs))
                    t0 = time.perf_counter()
                    for _ in range(args.wall_calls):
                        out = fn(*xs)
                    jax.block_until_ready(out)
                    walls.append((time.perf_counter() - t0) / args.wall_calls * 1e3)
                    del out
                e["wall_ms"] = statistics.median(walls)
            except Exception as exc:  # noqa: BLE001
                e["error"] = harness.error_summary(str(exc))
            print(json.dumps({k: v for k, v in e.items() if k != "kernels"}), flush=True)
            entries.append(e)
    return {"framework": "jax", "jax": jax.__version__, "tree": args.label, "environment": harness.environment(),
            "entries": entries}


def torch_kernel_ms(step, active: int) -> dict:
    """``torch.profiler`` kernel time of one call of ``step`` over ``active`` recorded calls (see the module doc)."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, schedule

    with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=active)) as prof:
        for _ in range(active + 2):
            step()
            torch.cuda.synchronize()
            prof.step()
    per: dict[str, list] = {}
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA:
            total, count = per.setdefault(ev.name, [0.0, 0])
            per[ev.name] = [total + ev.device_time_total / 1e3, count + 1]
    launches = {name: math.ceil(count / active) for name, (_, count) in per.items()}
    kernels = {name: total / count * launches[name] for name, (total, count) in per.items()}
    return {"median_ms": sum(kernels.values()),
            "plain_sum_ms": sum(total for total, _ in per.values()) / active,
            "dropped_records": sum(launches[name] * active - count for name, (_, count) in per.items()),
            "kernels_per_call": sum(launches.values()), "kernels": kernels}


def torch_wall_ms(step, calls: int) -> float:
    import torch

    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / calls


def run_torch(args) -> dict:
    import torch
    from mamba_ssm.modules.mamba2 import Mamba2

    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    entries = []
    for b, t, d in args.shapes:
        torch.manual_seed(0)
        m = Mamba2(d, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256,
                   use_mem_eff_path=True, device="cuda", dtype=torch.bfloat16)
        u = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16)
        params = [p for p in m.parameters() if p.requires_grad]

        def fwd():
            with torch.no_grad():
                m(u)

        def fwdbwd():
            torch.autograd.grad(m(u), [u, *params], grad_outputs=do)

        for mode, step in (("fwd", fwd), ("fwdbwd", fwdbwd)):
            e = {"shape": [b, t, d], "mode": mode, "impl": "mamba2_official"}
            try:
                for _ in range(3):  # Triton autotuning and warm-up
                    step()
                torch.cuda.synchronize()
                runs = [torch_kernel_ms(step, args.active) for _ in range(args.profiles)]
                best = min(runs, key=lambda r: r["median_ms"])
                e.update(median_ms=statistics.median(r["median_ms"] for r in runs), min_ms=best["median_ms"],
                         plain_sum_ms=statistics.median(r["plain_sum_ms"] for r in runs),
                         dropped_records=sum(r["dropped_records"] for r in runs),
                         kernels_per_call=best["kernels_per_call"], calls=args.active * args.profiles)
                if args.breakdown:
                    e["kernels"] = _top(best["kernels"], args.breakdown)
                e["wall_ms"] = statistics.median(torch_wall_ms(step, args.wall_calls) for _ in range(3))
            except Exception as exc:  # noqa: BLE001
                e["error"] = repr(exc)[:300]
            print(json.dumps({k: v for k, v in e.items() if k != "kernels"}), flush=True)
            entries.append(e)
        del m
        torch.cuda.empty_cache()
    import mamba_ssm

    return {"framework": "torch", "torch": torch.__version__, "mamba_ssm": mamba_ssm.__version__,
            "path": "mamba_ssm.modules.mamba2.Mamba2(use_mem_eff_path=True) -> mamba_split_conv1d_scan_combined",
            "device": torch.cuda.get_device_name(), "entries": entries}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("framework", choices=("jax", "torch"))
    p.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="jax: the repository tree to time (default: this one)")
    p.add_argument("--label", default="", help="jax: a name for the tree, kept in the result")
    p.add_argument("--shapes", type=parse_shapes, default=SHAPES, help='"B,T,d_model;..."')
    p.add_argument("--iterations", type=int, default=30, help="jax: CUPTI-timed calls")
    p.add_argument("--active", type=int, default=10, help="torch: profiled calls per profiler session")
    p.add_argument("--profiles", type=int, default=3, help="torch: profiler sessions (median reported)")
    p.add_argument("--wall-calls", type=int, default=20, help="back-to-back calls of the wall clock")
    p.add_argument("--breakdown", type=int, default=12, help="slowest kernels kept per entry (0: none)")
    p.add_argument("--memory-fraction", type=float, default=0.0, help="torch per-process memory cap (0: none)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = run_jax(args) if args.framework == "jax" else run_torch(args)
    result.update(kind="mamba2_layer", config=CONFIG, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
