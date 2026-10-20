# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 SISO benchmark: our Mosaic GPU kernels vs ``jla.ops.chunked`` (XLA) vs the official
Triton kernel (state-spaces/mamba ``mamba3_siso_combined``).

The official function fuses the Q/K biases, the data-dependent rotation, the ``D`` skip and the
``Z`` gate. Like for like with it are ``fused`` (``jla.layers.ssm.mamba3_siso(impl="mosaic_gpu")``,
which runs ``jla.ops.mosaic_gpu.mamba3_fused``: biases, angle scan and rotation in Mosaic GPU
kernels) and ``ours+xla`` (the same layer function with that glue in XLA around the recurrence
kernels). ``ours`` times the recurrence op bare (pre-rotated ``q, k`` and the ``g, beta, gamma``
gates). Inputs come from one numpy generator, so both frameworks see identical values.

Each framework runs in its own process and virtualenv:

    python benchmarks/bench_mamba3.py jax   --out benchmarks/results/mamba3_jax.json
    python benchmarks/bench_mamba3.py torch --mamba-root /path/to/state-spaces/mamba \\
        --out benchmarks/results/mamba3_torch.json

The torch run imports ``mamba_ssm/ops/triton/mamba3`` straight from a source checkout (a package
shim skips ``mamba_ssm/__init__.py``, which needs compiled CUDA extensions). ``--dump`` writes the
official outputs and gradients for a small shape; ``jax --check <file>`` compares ours against it.

Times are milliseconds per call. ``kernel_ms`` is the CUPTI device time of all kernels of one call
(Mosaic's ``profiler.measure`` in JAX, the median of ``--rounds`` calls; ``torch.profiler`` in torch,
the mean of ``--rounds`` calls recorded after a wait and a warmup step), with a per-kernel
breakdown; ``ms`` is wall clock, the median over ``--rounds`` rounds of
``--iters`` back-to-back calls with one synchronisation per round. The GPUs are shared with other
jobs, whose time slices inflate wall clock most, so ``nvidia-smi`` utilisation is recorded next to
every run and ``kernel_ms`` measured in the same time window is the comparison to use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

SHAPES = [(4, 4096, 24, 128, 64), (1, 16384, 48, 128, 64), (8, 2048, 32, 128, 64)]
CHECK_SHAPE = (2, 512, 4, 128, 64)


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), kept as float32."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def make_inputs(batch: int, seq_len: int, heads: int, key_dim: int, value_dim: int, seed: int = 0) -> dict:
    """Raw Mamba-3 SISO inputs (float32 numpy), distributed like the official test's.

    Everything the official wrapper casts to bfloat16 (q, k, v, trap, angles, z) is already
    bfloat16-representable, so both frameworks start from identical values.
    """
    rng = np.random.default_rng(seed)

    def rms_normal(*shape):
        x = rng.standard_normal(shape, dtype=np.float32)
        return _round_bf16(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))

    def normal(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    dt = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), (batch, seq_len, heads))).astype(np.float32)
    a = -rng.uniform(1.0, 16.0, (batch, seq_len, heads)).astype(np.float32)
    return dict(
        q=rms_normal(batch, seq_len, 1, key_dim),  # C, one group shared by all heads
        k=rms_normal(batch, seq_len, 1, key_dim),  # B
        v=_round_bf16(normal(batch, seq_len, heads, value_dim)),  # x
        adt=a * dt,
        dt=dt,
        trap=_round_bf16(rng.uniform(0.0, 1.0, (batch, seq_len, heads)).astype(np.float32)),
        q_bias=normal(heads, key_dim),
        k_bias=normal(heads, key_dim),
        angles=_round_bf16(normal(batch, seq_len, heads, key_dim // 4)),
        d=np.ones((heads,), np.float32),
        z=_round_bf16(normal(batch, seq_len, heads, value_dim)),
        do=_round_bf16(normal(batch, seq_len, heads, value_dim)),
    )


def gpu_status() -> list[str]:
    """Memory and utilization of the one GPU this run uses; never other GPUs (other jobs)."""
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


# --------------------------------------------------------------------------------------- JAX


def run_jax(args) -> dict:
    import jax
    import jax.numpy as jnp
    from jax.experimental.mosaic.gpu import profiler

    from jla.layers import ssm as ssm_layer
    from jla.layers.ssm import cumulative_angle, mamba3_siso, rotate_pairs
    from jla.ops import chunked
    from jla.ops.mosaic_gpu import mamba3 as m3

    f32, bf16 = jnp.float32, jnp.bfloat16

    def to_device(raw):
        x = {n: jnp.asarray(a) for n, a in raw.items()}
        for n in ("q", "k", "v", "z", "do"):
            x[n] = x[n].astype(bf16)
        return x

    def op_inputs(x):
        """The op boundary: rotated, biased per-head q/k and the trapezoid gates."""
        heads = x["v"].shape[2]
        q = jnp.repeat(x["q"], heads, axis=2).astype(f32) + x["q_bias"]
        k = jnp.repeat(x["k"], heads, axis=2).astype(f32) + x["k_bias"]
        angle = cumulative_angle(jnp.tanh(x["angles"]) * math.pi * x["dt"][..., None])
        q, k = rotate_pairs(q, angle), rotate_pairs(k, angle)
        lam = jax.nn.sigmoid(x["trap"])
        gates = dict(g=x["adt"], beta=(1.0 - lam) * x["dt"] * jnp.exp(x["adt"]), gamma=lam * x["dt"])
        return q.astype(bf16), k.astype(bf16), gates

    def bare(impl):
        fn = m3.mamba3 if impl == "mosaic_gpu" else lambda *a, **kw: chunked.mamba3(*a, precision=None, **kw)

        def run(q, k, v, g, beta, gamma):
            return fn(q, k, v, g, beta, gamma)[0]

        return run

    def layer(impl, xla_glue=False):
        """The layer core; ``xla_glue`` traces the XLA bias/rotation/D/Z glue around the recurrence
        kernels instead of the fused Mosaic GPU core that ``impl="mosaic_gpu"`` selects."""

        def run(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z):
            saved = ssm_layer._fused_core
            if xla_glue:
                ssm_layer._fused_core = lambda: None
            try:
                return mamba3_siso(q, k, v, adt, dt, trap, q_bias, k_bias, angles, d=d, z=z, impl=impl)[0]
            finally:
                ssm_layer._fused_core = saved

        return run

    def fwd_bwd(fn):
        def run(*inputs_and_cotangent):
            *inputs, do = inputs_and_cotangent
            out, vjp = jax.vjp(fn, *inputs)
            return out, vjp(do.astype(out.dtype))

        return run

    def sync(out=None):
        if out is not None:
            jax.block_until_ready(out)

    def time_fn(fn, inputs) -> dict:
        f = jax.jit(fn)
        jax.block_until_ready(f(*inputs))
        wall = _median_loop(lambda: f(*inputs), sync, args.iters, args.rounds)
        kernels = []
        for _ in range(args.rounds):
            _, ms = profiler.measure(f)(*inputs)
            kernels.append(ms)
        _, breakdown = profiler.measure(f, aggregate=False)(*inputs)
        return dict(ms=wall, kernel_ms=statistics.median(kernels), kernels=breakdown)

    if args.check:
        return check_against_official(args, jax, jnp, to_device, op_inputs, layer, fwd_bwd)

    results = []
    for shape in args.shapes:
        x = to_device(make_inputs(*shape))
        q, k, gates = op_inputs(x)
        bare_in = (q, k, x["v"], gates["g"], gates["beta"], gates["gamma"])
        layer_in = tuple(x[n] for n in ("q", "k", "v", "adt", "dt", "trap", "q_bias", "k_bias", "angles", "d", "z"))
        row = dict(shape=dict(zip("BTHKV", shape)), gpu_before=gpu_status())
        for label, fn, inputs in (
            ("ours", bare("mosaic_gpu"), bare_in),
            ("fused", layer("mosaic_gpu"), layer_in),
            ("ours+xla", layer("mosaic_gpu", xla_glue=True), layer_in),
            ("chunked", bare("chunked"), bare_in),
            ("chunked+xla", layer("chunked"), layer_in),
        ):
            for mode, f, ins in (("fwd", fn, inputs), ("fwd+bwd", fwd_bwd(fn), inputs + (x["do"],))):
                try:
                    row[f"{label} {mode}"] = time_fn(f, ins)
                except Exception as err:  # noqa: BLE001 - record OOMs etc. and keep going
                    row[f"{label} {mode}"] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
                print(shape, label, mode, {k: v for k, v in row[f"{label} {mode}"].items() if k != "kernels"},
                      flush=True)
        row["gpu_after"] = gpu_status()
        results.append(row)
    return dict(framework="jax", jax=jax.__version__, device=str(jax.devices()[0]), results=results)


def check_against_official(args, jax, jnp, to_device, op_inputs, layer, fwd_bwd) -> dict:
    """The fused core and the XLA-glue layer vs the official kernel's dumped outputs and gradients."""
    want = dict(np.load(args.check))
    x = to_device(make_inputs(*CHECK_SHAPE))
    names = ("q", "k", "v", "adt", "dt", "trap", "q_bias", "k_bias", "angles", "d", "z")
    reports = {}
    for label, fn in (("fused", layer("mosaic_gpu")), ("ours+xla", layer("mosaic_gpu", xla_glue=True))):
        out, grads = jax.jit(fwd_bwd(fn))(*(x[n] for n in names), x["do"])
        report = reports[label] = {}
        for name, got in [("out", out)] + [(f"d{n}", g) for n, g in zip(names, grads)]:
            got, ref = np.asarray(got, np.float32), want[name].astype(np.float32)
            report[name] = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
            print(f"{label:8s} {name:10s} max |ours - official| / max |official| = {report[name]:.3e}", flush=True)
    return dict(framework="jax-check", shape=CHECK_SHAPE, relative_max_error=reports)


# ------------------------------------------------------------------------------------- torch


def import_official(mamba_root: str):
    """``mamba3_siso_combined`` from a source checkout, without running ``mamba_ssm/__init__.py``."""
    root = Path(mamba_root) / "mamba_ssm"
    for name, path in (
        ("mamba_ssm", root),
        ("mamba_ssm.ops", root / "ops"),
        ("mamba_ssm.ops.triton", root / "ops" / "triton"),
        ("mamba_ssm.ops.triton.mamba3", root / "ops" / "triton" / "mamba3"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    return mamba3_siso_combined


def run_torch(args) -> dict:
    import torch

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    official = import_official(args.mamba_root)
    dev = torch.device("cuda")

    def to_device(raw):
        x = {n: torch.from_numpy(a).to(dev) for n, a in raw.items()}
        for n in ("q", "k", "v", "z", "do"):
            x[n] = x[n].to(torch.bfloat16)
        for n in ("adt", "dt", "trap"):
            x[n] = x[n].permute(0, 2, 1).contiguous()  # official layout [B, H, T]
        return x

    names = ("q", "k", "v", "adt", "dt", "trap", "q_bias", "k_bias", "angles", "d", "z")

    def call(x):
        return official(x["q"], x["k"], x["v"], x["adt"], x["dt"], x["trap"], x["q_bias"], x["k_bias"],
                        x["angles"], D=x["d"], Z=x["z"])

    def sync(out=None):
        torch.cuda.synchronize()

    def kernel_times(fn) -> tuple[float, list, int]:
        """CUPTI (torch.profiler) device time of all kernels of one call: (ms, per-call breakdown, records lost).

        A wait and a warmup step precede ``--rounds`` recorded calls: profiling single calls missed some
        kernels (for example the angle kernel launched first). The profiler can still lose whole calls' records
        (on 20 Oct 2026 it kept the kernels of only two of five forward calls at ``(8, 2048, 32)``, so the plain sum
        over the records divided by the calls read 0.300 ms against 0.75 ms), so every kernel counts as its mean
        time per recorded launch times its launches per call; the number of missing records is returned.
        """
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile, schedule

        active = args.rounds
        with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=active)) as prof:
            for _ in range(active + 2):
                fn()
                torch.cuda.synchronize()
                prof.step()
        totals: dict[str, list] = {}
        for e in prof.events():
            if e.device_type == DeviceType.CUDA:
                total, count = totals.get(e.name, (0.0, 0))
                totals[e.name] = (total + e.device_time_total / 1e3, count + 1)
        launches = {name: -(-count // active) for name, (_, count) in totals.items()}
        per_call = {name: total / count * launches[name] for name, (total, count) in totals.items()}
        lost = sum(launches[name] * active - count for name, (_, count) in totals.items())
        return sum(per_call.values()), sorted(per_call.items(), key=lambda kv: -kv[1]), lost

    if args.dump:
        x = to_device(make_inputs(*CHECK_SHAPE))
        leaves = {n: x[n].detach().clone().requires_grad_(True) for n in names}
        out = call(leaves)
        grads = torch.autograd.grad(out, [leaves[n] for n in names], grad_outputs=x["do"])
        dump = {"out": out.detach().float().cpu().numpy()}
        for n, g in zip(names, grads):
            g = g.detach().float()
            if n in ("adt", "dt", "trap"):
                g = g.permute(0, 2, 1)
            dump[f"d{n}"] = g.cpu().numpy()
        np.savez(args.dump, **dump)
        return dict(framework="torch-dump", shape=CHECK_SHAPE, file=str(args.dump))

    results = []
    for shape in args.shapes:
        x = to_device(make_inputs(*shape))
        row = dict(shape=dict(zip("BTHKV", shape)), gpu_before=gpu_status())
        leaves = {n: x[n].detach().clone().requires_grad_(True) for n in names}

        def fwd():
            with torch.no_grad():
                return call(x)

        def fwd_bwd():
            out = call(leaves)
            return torch.autograd.grad(out, [leaves[n] for n in names], grad_outputs=x["do"])

        for mode, fn in (("fwd", fwd), ("fwd+bwd", fwd_bwd)):
            try:
                for _ in range(3):  # autotuning and warmup
                    fn()
                kernel_ms, breakdown, lost = kernel_times(fn)
                row[f"official {mode}"] = dict(
                    ms=_median_loop(fn, sync, args.iters, args.rounds), kernel_ms=kernel_ms, kernels=breakdown,
                    lost_records=lost,
                )
            except Exception as err:  # noqa: BLE001
                row[f"official {mode}"] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
            print(shape, "official", mode, {k: v for k, v in row[f"official {mode}"].items() if k != "kernels"},
                  flush=True)
        row["gpu_after"] = gpu_status()
        results.append(row)
        del x, leaves
        torch.cuda.empty_cache()
    return dict(framework="torch", torch=torch.__version__, device=torch.cuda.get_device_name(), results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("framework", choices=["jax", "torch"])
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--iters", type=int, default=10, help="back-to-back calls per round")
    parser.add_argument("--rounds", type=int, default=5, help="rounds (the median is reported)")
    parser.add_argument("--shapes", type=lambda s: [tuple(int(v) for v in t.split(",")) for t in s.split(";")],
                        default=SHAPES, help='"B,T,H,K,V;..." (default: the three benchmark shapes)')
    parser.add_argument("--mamba-root", default="", help="state-spaces/mamba source checkout (torch)")
    parser.add_argument("--memory-fraction", type=float, default=0.10, help="torch per-process memory cap")
    parser.add_argument("--dump", type=Path, help="torch: write official outputs/gradients for CHECK_SHAPE")
    parser.add_argument("--check", type=Path, help="jax: compare ours against a --dump file")
    args = parser.parse_args()
    result = run_jax(args) if args.framework == "jax" else run_torch(args)
    result.update(iters=args.iters, rounds=args.rounds, time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
