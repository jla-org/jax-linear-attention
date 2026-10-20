# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""torch cuBLASLt reference for the speed-of-light ceilings (benchmarks/profile_1p3b.md).

    python benchmarks/speed_of_light_torch.py --json torch.json

The same GEMM problems and layouts as ``bench_speed_of_light.py gemm`` (forward ``x @ W``, dgrad
``dy @ W^T``, wgrad ``x^T @ dy`` at M = 65,536 tokens, plus 8192^3) through ``torch.mm`` in bf16
(cuBLASLt with its own heuristics; transposed operands are strided views, as autograd passes them) and
through ``torch._scaled_mm`` in FP8 (e4m3 x e4m3 -> bf16, per-tensor scales; that call needs
``a`` row-major and ``b`` column-major, so every FP8 problem is timed with both operands K-major),
with and without ``use_fast_accum``. HBM: ``copy_``, an elementwise multiply and ``sum`` at 256 MiB to
4 GiB. Times are CUDA-event times of 30 back-to-back calls after 5 warm-ups (per-call mean).

``--mode clock`` runs bf16 and FP8 8192^3 GEMMs back to back for ``--seconds`` each while ``nvidia-smi``
samples the GPU (sustained clock and power per precision); ``--mode overlap`` checks whether a
memory-bound elementwise pass hides under a GEMM on a second CUDA stream.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

TOKENS = 65536
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


def _time_ms(fn, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _gemms(name, m, k, n):
    """(layout, (M', K', N')) of the forward, dgrad and wgrad GEMMs of y[m, n] = x[m, k] @ W[k, n]."""
    rows = [("fwd x@W", (m, k, n))]
    if name != CUBE[0]:
        rows += [("dgrad dy@W^T", (m, n, k)), ("wgrad x^T@dy", (k, m, n))]
    return rows


def bf16_rows(problems) -> list[dict]:
    rows = []
    dev = "cuda"
    for name, m, k, n in problems:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(k, n, device=dev, dtype=torch.bfloat16) / k ** 0.5
        dy = torch.randn(m, n, device=dev, dtype=torch.bfloat16)
        calls = {"fwd x@W": lambda: torch.mm(x, w), "dgrad dy@W^T": lambda: torch.mm(dy, w.t()),
                 "wgrad x^T@dy": lambda: torch.mm(x.t(), dy)}
        for layout, (mm, kk, nn) in _gemms(name, m, k, n):
            ms = _time_ms(calls[layout])
            fl = 2.0 * mm * kk * nn
            row = dict(impl="torch.mm (cuBLASLt)", problem=name, layout=layout, dtype="bf16", gemm=[mm, kk, nn],
                       flops=fl, ms=round(ms, 4), tflops=round(fl / ms / 1e9, 1))
            print(json.dumps(row), flush=True)
            rows.append(row)
        del x, w, dy
        torch.cuda.empty_cache()
    return rows


def fp8_rows(problems) -> list[dict]:
    rows = []
    dev = "cuda"
    one = torch.ones((), device=dev, dtype=torch.float32)
    for name, m, k, n in problems:
        for layout, (mm, kk, nn) in _gemms(name, m, k, n):
            a = torch.randn(mm, kk, device=dev).to(torch.float8_e4m3fn)  # row-major [M', K']
            bt = torch.randn(nn, kk, device=dev).to(torch.float8_e4m3fn)  # [N', K'] row-major
            b = bt.t()  # [K', N'] column-major: both operands K-major
            fl = 2.0 * mm * kk * nn
            for fast in (False, True):
                row = dict(impl="torch._scaled_mm (cuBLASLt FP8)", problem=name, layout=layout, dtype="fp8",
                           variant="both K-major", use_fast_accum=fast, gemm=[mm, kk, nn], flops=fl)
                try:
                    ms = _time_ms(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                                                           use_fast_accum=fast))
                    row.update(ms=round(ms, 4), tflops=round(fl / ms / 1e9, 1))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(json.dumps(row), flush=True)
                rows.append(row)
            del a, bt, b
            torch.cuda.empty_cache()
    return rows


def hbm_rows() -> list[dict]:
    rows = []
    for mib in (256, 1024, 4096):
        n = mib * 2 ** 20 // 2
        x = torch.ones(n, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        for label, fn, nbytes in (("copy_ (read+write)", lambda: y.copy_(x), 4 * n),
                                  ("elementwise mul (read+write)", lambda: torch.mul(x, 1.5, out=y), 4 * n),
                                  ("sum (read)", lambda: x.sum(dtype=torch.float32), 2 * n)):
            ms = _time_ms(fn)
            row = dict(op=label, mib=mib, bytes=nbytes, ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1))
            print(json.dumps(row), flush=True)
            rows.append(row)
        del x, y
        torch.cuda.empty_cache()
    return rows


def _smi(fields: str) -> list[list[str]]:
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "-i", gpu, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]


def clock_rows(seconds: float) -> list[dict]:
    """Back-to-back bf16 and FP8 8192^3 GEMMs for ``seconds`` each while ``nvidia-smi`` samples the GPU
    every 100 ms: the sustained SM clock and power of each precision under full tensor-core load."""
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    fields = "clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
    if not _smi(fields) or "Field" in " ".join(_smi(fields)[0]):
        fields = "clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    n = 8192
    a16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16) / n ** 0.5
    a8 = a16.to(torch.float8_e4m3fn)
    b8 = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    calls = {"bf16": lambda: torch.mm(a16, b16),
             "fp8": lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)}
    rows = []
    for name, fn in calls.items():
        proc = subprocess.Popen(["nvidia-smi", "-i", gpu, f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                                 "-lms", "100"], stdout=subprocess.PIPE, text=True)
        samples: list[tuple[float, list[str]]] = []

        def read():
            for line in proc.stdout:
                samples.append((time.time(), [x.strip() for x in line.split(",")]))

        th = threading.Thread(target=read, daemon=True)
        th.start()
        t0 = time.time()
        count = 0
        while time.time() - t0 < seconds:
            for _ in range(50):
                fn()
            count += 50
            torch.cuda.synchronize()
        t1 = time.time()
        proc.terminate()
        proc.wait(timeout=10)
        th.join(timeout=5)
        steady = [s for ts, s in samples if t0 + 5 <= ts <= t1]
        num = lambda i: sorted(float(s[i]) for s in steady if len(s) > i and s[i].replace(".", "").isdigit())  # noqa: E731
        med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
        clocks, power, temp = num(0), num(1), num(2)
        reasons: dict[str, int] = {}
        for s in steady:
            if len(s) > 3:
                reasons[s[3]] = reasons.get(s[3], 0) + 1
        fl = 2.0 * n ** 3
        row = dict(gemm=f"{name} 8192^3 back to back", seconds=round(t1 - t0, 1), calls=count,
                   tflops=round(count * fl / (t1 - t0) / 1e12, 1), samples=len(steady),
                   sm_clock_mhz_median=med(clocks), sm_clock_mhz_min=clocks[0] if clocks else None,
                   sm_clock_mhz_max=clocks[-1] if clocks else None, power_w_median=med(power),
                   power_w_max=power[-1] if power else None, temperature_c_max=temp[-1] if temp else None,
                   clock_event_reasons={k: round(v / max(1, len(steady)), 3) for k, v in reasons.items()})
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def overlap_rows() -> list[dict]:
    """Can a memory-bound pass hide under a GEMM on a second CUDA stream? A = the SwiGLU gate+up GEMM
    (65536 x 2048 x 11264, bf16), B = ``silu(g) * u`` over two [65536, 5632] bf16 tensors (two
    memory-bound kernels). Times: A alone, B alone, A then B on one stream, A and B on two streams."""
    m, k, n, f = TOKENS, 2048, 11264, 5632
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / k ** 0.5
    g = torch.randn(m, f, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(m, f, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    h = torch.empty(m, f, device="cuda", dtype=torch.bfloat16)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

    def gemm():
        torch.mm(x, w, out=y)

    def elementwise():
        torch.mul(torch.nn.functional.silu(g), u, out=h)

    def serial():
        gemm()
        elementwise()

    def concurrent():
        cur = torch.cuda.current_stream()
        s1.wait_stream(cur)
        s2.wait_stream(cur)
        with torch.cuda.stream(s1):
            gemm()
        with torch.cuda.stream(s2):
            elementwise()
        cur.wait_stream(s1)
        cur.wait_stream(s2)

    rows = []
    for label, fn in (("A: GEMM alone", gemm), ("B: elementwise alone", elementwise),
                      ("A then B, one stream", serial), ("A and B, two streams", concurrent)):
        ms = _time_ms(fn, iters=20)
        rows.append(dict(case=label, ms=round(ms, 4)))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", default=None)
    p.add_argument("--mem-fraction", type=float, default=0.25)
    p.add_argument("--mode", default="ceilings", choices=("ceilings", "clock", "overlap", "clock+overlap"))
    p.add_argument("--seconds", type=float, default=30.0)
    args = p.parse_args(argv)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    problems = PROBLEMS + [CUBE]
    result = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "cublaslt_workspace": os.environ.get("CUBLASLT_WORKSPACE_SIZE", "default"),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "mode": args.mode}
    if args.mode == "ceilings":
        result["bf16"] = bf16_rows(problems)
        result["fp8"] = fp8_rows(problems)
        result["hbm"] = hbm_rows()
    if "clock" in args.mode:
        result["clock"] = clock_rows(args.seconds)
    if "overlap" in args.mode:
        result["overlap"] = overlap_rows()
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
