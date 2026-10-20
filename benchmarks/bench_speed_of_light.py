# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measured H100 ceilings for the speed-of-light model of the 1.3B step (benchmarks/profile_1p3b.md).

    python benchmarks/bench_speed_of_light.py clock --seconds 60 --json clock.json
    python benchmarks/bench_speed_of_light.py gemm --impl xla --json gemm_xla.json
    python benchmarks/bench_speed_of_light.py gemm --impl mosaic --budget 45 --json gemm_mosaic.json
    python benchmarks/bench_speed_of_light.py hbm --json hbm.json

``clock`` runs back-to-back bf16 8192^3 GEMMs (XLA, cuBLASLt) for ``--seconds`` while ``nvidia-smi``
samples the visible GPU every 100 ms (SM clock, max SM clock, power, power limit, temperature, the
active clock-event (throttle) reasons), and reports the sustained clock, the TFLOP/s over the whole
window and the clock that the 989.4 TFLOP/s spec implies (4096 dense bf16 FLOPs per clock per SM x 132
SMs). Nothing on the machine or GPU is changed: the script only reads ``nvidia-smi``. The sampled GPU is
the first index of ``CUDA_VISIBLE_DEVICES``, which assumes CUDA and ``nvidia-smi`` number the GPUs alike
(true for identical GPUs in PCI order).

``gemm`` times the projection and MLP GEMMs of the 1.3B presets (M = 65,536 tokens) in the three
layouts of a training step (forward ``x @ W``, dgrad ``dy @ W^T``, wgrad ``x^T @ dy``) plus 8192^3.
``--impl xla``: ``lax.dot_general`` as the model spells it (XLA picks cuBLASLt and autotunes), bf16
and FP8 (e4m3 x e4m3 -> bf16; the FP8 rows also run with both operands K-major, the layout Hopper's
FP8 tensor cores take, so the GEMM itself is timed without XLA's operand transposes).
``--impl mosaic``: JAX's Mosaic GPU Hopper matmul (``jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py``),
which reads row-major ``a[M, K]`` and ``b[K, N]``; dgrad and wgrad are timed as the same GEMM problems
with their operands pre-transposed, and each problem is tuned over tile shapes, warpgroup split,
grid order, grid tile width and 2-CTA clusters within ``--budget`` seconds. Times are CUPTI kernel
times (``harness.median_kernel_ms``), every kernel of the call included.

``hbm`` times a large elementwise pass (read + write) and a large reduction (read) at several sizes.

``epilogue`` prices a fused GEMM epilogue: ``x @ W + r`` with the residual read in the Mosaic GPU
matmul's epilogue against the same GEMM followed by a separate add.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import environment, gpu_snapshot, kernel_breakdown_ms, median_kernel_ms, write_json  # noqa: E402

SPEC_TFLOPS = 989.4  # H100 SXM dense bf16
FLOPS_PER_CLOCK_PER_SM = 4096  # dense bf16 (and 8192 for FP8)
NUM_SMS = 132
TOKENS = 65536

# (name, M, K, N) of the forward GEMM y[M, N] = x[M, K] @ W[K, N] at the 1.3B presets (d_model 2048).
PROBLEMS = [
    ("d->d (q/k/v/o/gate, GDN/KDA/attention)", TOKENS, 2048, 2048),
    ("d->512 (GQA k/v, 8 x 64)", TOKENS, 2048, 512),
    ("d->2x5632 (SwiGLU gate+up)", TOKENS, 2048, 11264),
    ("5632->d (MLP down)", TOKENS, 5632, 2048),
    ("d->4096 (Mamba z/x; LM-head vocab chunk)", TOKENS, 2048, 4096),
    ("4096->d (Mamba out_proj)", TOKENS, 4096, 2048),
    ("d->320 (Mamba B/C/dt)", TOKENS, 2048, 320),
]
CUBE = ("8192^3", 8192, 8192, 8192)

SMI_FIELDS = ["timestamp", "clocks.sm", "clocks.max.sm", "power.draw", "power.limit", "temperature.gpu"]
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
REASONS = {  # NVML clock event (throttle) reason bits
    0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}


def _gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return visible or "0"


def _smi_query(fields: list[str]) -> bool:
    out = subprocess.run(["nvidia-smi", "-i", _gpu_index(), f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return out.returncode == 0 and "Field" not in out.stdout


def _compute_apps() -> list[str]:
    try:
        uuid = subprocess.run(["nvidia-smi", "-i", _gpu_index(), "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mine = os.getpid()
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            rows.append(f"{'this process' if parts[1] == str(mine) else 'other process'}: {parts[2]} MiB")
    return rows


class Sampler:
    """``nvidia-smi -lms`` on the visible GPU in a background process."""

    def __init__(self, interval_ms: int = 100):
        reason = next((f for f in REASON_FIELDS if _smi_query([f])), None)
        self.fields = SMI_FIELDS + ([reason] if reason else [])
        self.cmd = ["nvidia-smi", "-i", _gpu_index(), f"--query-gpu={','.join(self.fields)}",
                    "--format=csv,noheader,nounits", "-lms", str(interval_ms)]
        self.rows: list[list[str]] = []
        self.marks: list[tuple[str, float]] = []

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(self.fields):
                self.rows.append(parts + [f"{time.time():.3f}"])

    def mark(self, label: str) -> None:
        self.marks.append((label, time.time()))

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=5)


def _num(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, int(p * (len(values) - 1) + 0.5))]  # noqa: E731
    return dict(n=len(values), mean=round(statistics.fmean(values), 2), median=q(0.5), p05=q(0.05), p95=q(0.95),
                min=values[0], max=values[-1])


def _summarize_samples(sampler: Sampler, t0: float, t1: float) -> dict:
    idx = {f: i for i, f in enumerate(sampler.fields)}
    rows = [r for r in sampler.rows if t0 <= float(r[-1]) <= t1]
    col = lambda f: [v for v in (_num(r[idx[f]]) for r in rows) if v is not None]  # noqa: E731
    out = {"samples": len(rows), "sm_clock_mhz": _stats(col("clocks.sm")),
           "max_sm_clock_mhz": _stats(col("clocks.max.sm")), "power_w": _stats(col("power.draw")),
           "power_limit_w": _stats(col("power.limit")), "temperature_c": _stats(col("temperature.gpu"))}
    reason_field = sampler.fields[-1] if sampler.fields[-1] in REASON_FIELDS else None
    if reason_field:
        counts: dict[str, int] = {}
        for r in rows:
            raw = r[idx[reason_field]]
            try:
                bits = int(raw, 16)
            except ValueError:
                continue
            names = [n for b, n in REASONS.items() if bits & b] or ["none"]
            key = "+".join(names)
            counts[key] = counts.get(key, 0) + 1
        out["clock_event_reasons"] = {"field": reason_field, "fraction_of_samples": {
            k: round(v / max(1, len(rows)), 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}}
    return out


def cmd_clock(args) -> dict:
    n = 8192
    ka, kb = jax.random.split(jax.random.key(0))
    a = jax.random.normal(ka, (n, n), jnp.bfloat16)
    b = (jax.random.normal(kb, (n, n), jnp.float32) / np.sqrt(n)).astype(jnp.bfloat16)
    gemm = jax.jit(lambda x, y: x @ y)
    flops = 2.0 * n ** 3
    burst = median_kernel_ms(gemm, a, b)
    idle_s = 5.0
    result = {"gemm": "bf16 8192^3 via XLA (cuBLASLt), back to back", "flops_per_call": flops,
              "spec_tflops": SPEC_TFLOPS,
              "spec_implied_sm_clock_mhz": round(SPEC_TFLOPS * 1e12 / (FLOPS_PER_CLOCK_PER_SM * NUM_SMS) / 1e6, 1),
              "flops_per_clock": FLOPS_PER_CLOCK_PER_SM * NUM_SMS,
              "burst_kernel_ms": burst, "burst_tflops": round(flops / burst["median_ms"] / 1e9, 1),
              "compute_apps_before": _compute_apps(), "gpus_before": gpu_snapshot()}
    with Sampler(args.interval_ms) as sampler:
        time.sleep(idle_s)
        sampler.mark("load start")
        t0 = time.time()
        calls = 0
        out = None
        while time.time() - t0 < args.seconds:
            for _ in range(50):
                out = gemm(a, b)
            calls += 50
            out.block_until_ready()
        t1 = time.time()
        sampler.mark("load end")
        late = median_kernel_ms(gemm, a, b, iterations=30, warmup=0)
        time.sleep(2.0)
    load = _summarize_samples(sampler, t0 + 5.0, t1)  # skip the first 5 s of ramp
    result.update({
        "seconds": round(t1 - t0, 2), "calls": calls,
        "sustained_tflops_wall": round(calls * flops / (t1 - t0) / 1e12, 1),
        "late_kernel_ms": late, "late_tflops": round(flops / late["median_ms"] / 1e9, 1),
        "idle_before": _summarize_samples(sampler, t0 - idle_s, t0 - 0.5),
        "under_load_after_5s": load,
        "compute_apps_after": _compute_apps(), "gpus_after": gpu_snapshot(),
    })
    clock = load.get("sm_clock_mhz", {}).get("median")
    if clock:
        peak = FLOPS_PER_CLOCK_PER_SM * NUM_SMS * clock * 1e6 / 1e12
        result["tensor_peak_at_sustained_clock_tflops"] = round(peak, 1)
        result["sustained_tflops_vs_peak_at_that_clock"] = round(result["sustained_tflops_wall"] / peak, 3)
    return result


# --------------------------------------------------------------------------------------------------
# GEMMs


def _layouts(m: int, k: int, n: int):
    """(layout, lhs shape, rhs shape, dimension numbers) of the forward, dgrad and wgrad GEMMs."""
    return [
        ("fwd x@W", (m, k), (k, n), (((1,), (0,)), ((), ()))),
        ("dgrad dy@W^T", (m, n), (k, n), (((1,), (1,)), ((), ()))),
        ("wgrad x^T@dy", (m, k), (m, n), (((0,), (0,)), ((), ()))),
    ]


def _out_shape(lhs, rhs, dims):
    (lc, rc), _ = dims
    return tuple(d for i, d in enumerate(lhs) if i not in lc) + tuple(d for i, d in enumerate(rhs) if i not in rc)


def _flops(lhs, dims, out) -> float:
    (lc, _), _ = dims
    return 2.0 * float(np.prod(out)) * float(np.prod([lhs[i] for i in lc]))


def _kernel_ms(fn, *args, out_bytes: float, budget_bytes: float = 6 * 2 ** 30) -> dict:
    """``harness.median_kernel_ms`` with as many iterations (3-30) as fit ``budget_bytes`` of outputs:
    the CUPTI profiler keeps every iteration's output alive until it reads the timings."""
    iterations = int(max(3, min(30, budget_bytes // max(1.0, out_bytes))))
    return median_kernel_ms(fn, *args, iterations=iterations)


def _rand(key, shape, dtype):
    return jax.random.normal(key, shape, jnp.float32).astype(dtype)


def _gemm_kernels(fn, *args) -> dict:
    breakdown = kernel_breakdown_ms(fn, *args, iterations=5)
    return {name[:90]: round(ms, 4) for name, ms in list(breakdown.items())[:4]}


def _xla_rows(problems, dtypes, out_f32: bool) -> list[dict]:
    rows = []
    key = jax.random.key(1)
    for (name, m, k, n), dtype in itertools.product(problems, dtypes):
        layouts = _layouts(m, k, n) if name != CUBE[0] else _layouts(m, k, n)[:1]
        for layout, lhs, rhs, dims in layouts:
            variants = [("as spelled", lhs, rhs, dims)]
            if dtype == "fp8":  # both operands K-major (Hopper FP8 wgmma layout): no XLA transposes
                (lc, rc), _ = dims
                tn_lhs = lhs if lc == (1,) else (lhs[1], lhs[0])
                tn_rhs = rhs if rc == (1,) else (rhs[1], rhs[0])
                variants.append(("both K-major", tn_lhs, tn_rhs, (((1,), (1,)), ((), ()))))
            for variant, ls, rs, dn in variants:
                in_dtype = jnp.float8_e4m3fn if dtype == "fp8" else jnp.bfloat16
                out_dtype = jnp.float32 if out_f32 else jnp.bfloat16
                k1, k2, key = jax.random.split(key, 3)
                a = _rand(k1, ls, in_dtype)
                b = _rand(k2, rs, in_dtype)
                fn = jax.jit(lambda x, y, dn=dn, od=out_dtype: lax.dot_general(x, y, dn, preferred_element_type=od))
                out = _out_shape(ls, rs, dn)
                fl = _flops(ls, dn, out)
                row = dict(impl="xla", problem=name, layout=layout, variant=variant, dtype=dtype,
                           out_dtype=str(jnp.dtype(out_dtype)), lhs=list(ls), rhs=list(rs), out=list(out), flops=fl)
                try:
                    t = _kernel_ms(fn, a, b, out_bytes=math.prod(out) * jnp.dtype(out_dtype).itemsize)
                    row.update(ms=round(t["median_ms"], 4), min_ms=round(t["min_ms"], 4), iterations=t["iterations"],
                               tflops=round(fl / t["median_ms"] / 1e9, 1),
                               kernels_ms=_gemm_kernels(fn, a, b))
                    gemm_ms = sum(v for kname, v in row["kernels_ms"].items()
                                  if any(s in kname.lower() for s in ("gemm", "nvjet", "cutlass", "sm90", "xmma")))
                    if gemm_ms:
                        row["gemm_kernel_tflops"] = round(fl / gemm_ms / 1e9, 1)
                except Exception as e:  # report and continue
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(json.dumps(row), flush=True)
                rows.append(row)
                del a, b
    return rows


def _mosaic_configs(m: int, n: int, k: int, base=None):
    """Tuning configs that fit the problem: tile shapes x warpgroup split x 2-CTA clusters first
    (default grid order, tile width 8, 4 stages); with ``base``, that config's grid order, grid tile
    width and pipeline depth variations."""
    from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

    D = hm.MatmulDimension

    def fits(c):
        cta_m = c.tile_m * (2 if c.wg_dimension == D.M else 1) * (2 if c.cluster_dimension == D.M else 1)
        cta_n = c.tile_n * (2 if c.wg_dimension == D.N else 1) * (2 if c.cluster_dimension == D.N else 1)
        return not (m % cta_m or n % cta_n or k % c.tile_k)

    if base is None:
        tiles = [(128, 128, D.N), (128, 128, D.M), (64, 256, D.M), (64, 256, D.N), (128, 64, D.N), (64, 128, D.M)]
        cands = [hm.TuningConfig(tile_m=tm, tile_n=tn, tile_k=64, max_concurrent_steps=4, epi_tile_n=64,
                                 epi_tile_m=64, grid_minor_dim=D.N, grid_tile_width=8, wg_dimension=wg,
                                 cluster_dimension=cl)
                 for (tm, tn, wg), cl in itertools.product(tiles, (None, D.M, D.N))]
    else:
        cands = [dataclasses.replace(base, grid_minor_dim=mi, grid_tile_width=w, max_concurrent_steps=s)
                 for mi, w, s in itertools.product((D.N, D.M), (8, 16, 4), (4, 5, 3))]
        cands = [c for c in cands if c != base]
    return [c for c in cands if fits(c)]


def _mosaic_rows(problems, budget_s: float, dtypes) -> list[dict]:
    from jax.experimental.mosaic.gpu import profiler
    from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

    rows = []
    key = jax.random.key(2)
    for (name, m, k, n), dtype in itertools.product(problems, dtypes):
        layouts = _layouts(m, k, n) if name != CUBE[0] else _layouts(m, k, n)[:1]
        for layout, lhs, rhs, dims in layouts:
            # the same GEMM problem in the kernel's native layout: a[M', K'] @ b[K', N'], row-major
            (lc, rc), _ = dims
            mm, kk = (lhs[0], lhs[1]) if lc == (1,) else (lhs[1], lhs[0])
            nn = rhs[1] if rc == (0,) else rhs[0]
            in_dtype = jnp.float8_e4m3fn if dtype == "fp8" else jnp.bfloat16
            k1, k2, key = jax.random.split(key, 3)
            a = _rand(k1, (mm, kk), in_dtype)
            b = _rand(k2, (kk, nn), in_dtype)
            fl = 2.0 * mm * nn * kk
            best = None
            tried = failed = 0
            errors: dict[str, int] = {}
            timings: list[tuple[float, object]] = []
            start = time.time()

            def run(cfg):
                nonlocal tried, failed
                tried += 1
                try:
                    fn = jax.jit(lambda x, y, cfg=cfg: hm.matmul(x, y, None, cfg))
                    jax.block_until_ready(fn(a, b))
                    _, times = profiler.measure(fn, iterations=5)(a, b)
                    timings.append((statistics.median(times), cfg))
                except Exception as e:  # SMEM overflow, unsupported layout for the dtype, ...
                    failed += 1
                    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}" if str(e) else type(e).__name__
                    errors[msg] = errors.get(msg, 0) + 1

            for cfg in _mosaic_configs(mm, nn, kk):
                if time.time() - start > budget_s:
                    break
                run(cfg)
            for _, base in sorted(timings, key=lambda t: t[0])[:2]:
                for cfg in _mosaic_configs(mm, nn, kk, base):
                    if time.time() - start > budget_s:
                        break
                    run(cfg)
            best = min(timings, key=lambda t: t[0]) if timings else None
            row = dict(impl="mosaic_gpu hopper_matmul", problem=name, layout=layout,
                       variant="native a[M,K] @ b[K,N] row-major (operands pre-transposed)", dtype=dtype,
                       out_dtype=str(jnp.dtype(in_dtype)), gemm=[mm, kk, nn], flops=fl, configs_tried=tried,
                       configs_failed=failed, tuning_s=round(time.time() - start, 1),
                       errors=dict(sorted(errors.items(), key=lambda kv: -kv[1])[:3]))
            if best is not None:
                ms, cfg = best
                fn = jax.jit(lambda x, y, cfg=cfg: hm.matmul(x, y, None, cfg))
                t = _kernel_ms(fn, a, b, out_bytes=mm * nn * jnp.dtype(in_dtype).itemsize)
                ref = lax.dot_general(a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
                got = fn(a, b).astype(jnp.float32)
                err = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
                row.update(ms=round(t["median_ms"], 4), tflops=round(fl / t["median_ms"] / 1e9, 1),
                           max_err_vs_xla_f32_rel_to_max=err,
                           config={f.name: str(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)})
            print(json.dumps(row), flush=True)
            rows.append(row)
            del a, b
    return rows


def cmd_gemm(args) -> dict:
    problems = [p for p in PROBLEMS if not args.only or any(s in p[0] for s in args.only.split(";"))]
    if args.tokens != TOKENS:  # the same projections at another micro-batch (e.g. 4 x 4096 = 16,384)
        problems = [(f"{name} @ M={args.tokens}", args.tokens, k, n) for name, _, k, n in problems]
    problems = problems + ([CUBE] if not args.no_cube else [])
    dtypes = args.dtypes.split(",")
    result = {"impl": args.impl, "xla_flags": os.environ.get("XLA_FLAGS", ""), "gpus_before": gpu_snapshot(),
              "compute_apps_before": _compute_apps()}
    if args.impl == "xla":
        result["rows"] = _xla_rows(problems, dtypes, out_f32=False)
        if args.lm_head_f32:
            result["rows"] += _xla_rows([PROBLEMS[4]], ["bf16"], out_f32=True)
    else:
        result["rows"] = _mosaic_rows(problems, args.budget, dtypes)
    result["gpus_after"] = gpu_snapshot()
    result["compute_apps_after"] = _compute_apps()
    return result


def cmd_epilogue(args) -> dict:
    """What a fused GEMM epilogue is worth: ``y = x @ W + r`` (a residual add after o_proj / MLP down) as
    (a) the GEMM alone, (b) Mosaic GPU's matmul with ``r`` read in its epilogue (``c`` operand), (c) the
    Mosaic GEMM then a separate XLA add, (d) XLA's ``dot + r`` (which cuBLASLt can take as beta = 1)."""
    from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

    D = hm.MatmulDimension
    cfg = hm.TuningConfig(tile_m=64, tile_n=256, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                          grid_minor_dim=D.N, grid_tile_width=16, wg_dimension=D.M, cluster_dimension=D.M)
    rows = []
    key = jax.random.key(3)
    for name, m, k, n in (PROBLEMS[3], PROBLEMS[0]):
        k1, k2, k3, key = jax.random.split(key, 4)
        x = _rand(k1, (m, k), jnp.bfloat16)
        w = (_rand(k2, (k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
        r = _rand(k3, (m, n), jnp.bfloat16)
        cases = {
            "GEMM alone (Mosaic GPU)": (jax.jit(lambda a, b, c: hm.matmul(a, b, None, cfg)), 0),
            "GEMM + residual in its epilogue (Mosaic GPU c operand)": (jax.jit(lambda a, b, c: hm.matmul(a, b, c, cfg)), 1),
            "GEMM (Mosaic GPU) then XLA add": (jax.jit(lambda a, b, c: hm.matmul(a, b, None, cfg) + c), 1),
            "XLA dot + add (cuBLASLt, beta=1 when XLA fuses it)": (jax.jit(lambda a, b, c: a @ b + c), 1),
        }
        ref = (x.astype(jnp.float32) @ w.astype(jnp.float32) + r.astype(jnp.float32))
        for label, (fn, uses_r) in cases.items():
            t = _kernel_ms(fn, x, w, r, out_bytes=m * n * 2)
            out = fn(x, w, r).astype(jnp.float32)
            err = float(jnp.max(jnp.abs(out - (ref if uses_r else ref - r.astype(jnp.float32)))) / jnp.max(jnp.abs(ref)))
            row = dict(problem=name, case=label, ms=round(t["median_ms"], 4),
                       tflops=round(2.0 * m * k * n / t["median_ms"] / 1e9, 1), max_err_rel_to_max=err,
                       kernels_ms=_gemm_kernels(fn, x, w, r))
            print(json.dumps(row), flush=True)
            rows.append(row)
        del x, w, r, ref
    return {"rows": rows, "config": str(cfg), "gpus": gpu_snapshot(), "compute_apps": _compute_apps()}


# --------------------------------------------------------------------------------------------------
# HBM


def cmd_hbm(args) -> dict:
    rows = []
    for mib in (256, 1024, 2048):
        n = mib * 2 ** 20 // 2
        x = jnp.ones((n // 4096, 4096), jnp.bfloat16)
        scale = jax.jit(lambda v: v * jnp.bfloat16(1.5))  # one read + one write
        red = jax.jit(lambda v: jnp.sum(v, dtype=jnp.float32))  # one read
        red_rows = jax.jit(lambda v: jnp.sum(v.astype(jnp.float32), axis=1))  # row reduction (norm-like)
        for label, fn, nbytes in (("elementwise x*1.5 (read+write)", scale, 2 * 2 * n),
                                  ("full reduction sum (read)", red, 2 * n),
                                  ("row reduction [rows,4096]->[rows] (read)", red_rows, 2 * n)):
            t = _kernel_ms(fn, x, out_bytes=2 * n if "write" in label else 4.0 * n // 4096)
            row = dict(op=label, mib=mib, bytes=nbytes, ms=round(t["median_ms"], 4), iterations=t["iterations"],
                       gbps=round(nbytes / t["median_ms"] / 1e6, 1))
            print(json.dumps(row), flush=True)
            rows.append(row)
        del x
    return {"rows": rows, "spec_gbps": 3352.0, "gpus": gpu_snapshot(), "compute_apps": _compute_apps()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clock")
    c.add_argument("--seconds", type=float, default=60.0)
    c.add_argument("--interval-ms", type=int, default=100)
    g = sub.add_parser("gemm")
    g.add_argument("--impl", choices=("xla", "mosaic"), default="xla")
    g.add_argument("--dtypes", default="bf16,fp8")
    g.add_argument("--budget", type=float, default=45.0, help="Mosaic GPU tuning seconds per problem")
    g.add_argument("--only", default="", help="';'-separated substrings of problem names")
    g.add_argument("--no-cube", action="store_true")
    g.add_argument("--tokens", type=int, default=TOKENS, help="M of the projection GEMMs (tokens per micro-batch)")
    g.add_argument("--lm-head-f32", action="store_true", help="also the f32-output LM-head chunk GEMMs")
    h = sub.add_parser("hbm")
    e = sub.add_parser("epilogue")
    for q in (c, g, h, e):
        q.add_argument("--json", default=None)
    args = p.parse_args(argv)
    result = {"cmd": args.cmd, "env": environment(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    result.update({"clock": cmd_clock, "gemm": cmd_gemm, "hbm": cmd_hbm, "epilogue": cmd_epilogue}[args.cmd](args))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, default=str)[:4000], flush=True)
    if args.json:
        write_json(args.json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
