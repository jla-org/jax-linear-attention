# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GEMMs of the 1.3B training steps and the fused SwiGLU GEMMs (benchmarks/fused_gemm.md).

Three parts, CUPTI kernel time (median, harness.median_kernel_ms) at the 1.3B shapes
(65,536 tokens per step = 16 x 4096):

* ``gemms``: every projection GEMM of transformer-1p3b, gdn-hd128-1p3b and mamba2-1p3b in its
  three forms (forward ``x W``, dgrad ``dy W^T``, wgrad ``x^T dy``) through XLA (cuBLASLt), in
  TFLOP/s and % of the 989 TFLOP/s dense bf16 peak, next to the best GEMM measured on this GPU
  (XLA at 8192^3 and JAX's Mosaic GPU Hopper matmul example at 8192^3 and the MLP shape).
* ``passes``: the memory-bound passes adjacent to those GEMMs (RMSNorm, SwiGLU forward and
  backward, residual add), each as XLA compiles it, with its HBM bytes and GB/s.
* ``fused``: the fused SwiGLU kernels (jla.ops.mosaic_gpu.fused_gemm) against the XLA ops they
  replace, a small tuning sweep, and the whole MLP block forward and forward+backward.

    python benchmarks/bench_fused_gemm.py --parts gemms,passes,fused --json results/fused_gemm.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import itertools
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import environment, gpu_snapshot, write_json  # noqa: E402

from jla.ops.mosaic_gpu import fused_gemm as fg  # noqa: E402

PEAK = 989e12
TOKENS = 16 * 4096
D = 2048
F = 5632
BF16 = jnp.bfloat16

# (arch, projection, K, N) of y[M, N] = x[M, K] @ W[K, N]; the attention's q|k|v and Mamba-2's
# z|x and B|C projections are the separate or stacked GEMMs the layers issue.
PROJECTIONS = [
    ("transformer-1p3b", "q", D, 2048), ("transformer-1p3b", "k/v", D, 512), ("transformer-1p3b", "o", 2048, D),
    ("transformer-1p3b", "mlp gate_up", D, 2 * F), ("transformer-1p3b", "mlp down", F, D),
    ("gdn-hd128-1p3b", "q/k/v/g", D, 2048), ("gdn-hd128-1p3b", "o", 2048, D),
    ("gdn-hd128-1p3b", "mlp gate_up", D, 2 * F), ("gdn-hd128-1p3b", "mlp down", F, D),
    ("mamba2-1p3b", "z|x", D, 8192), ("mamba2-1p3b", "B|C", D, 256), ("mamba2-1p3b", "out", 4096, D),
]


def _rand(key, shape, scale=1.0):
    return (scale * jax.random.normal(jax.random.key(key), shape, jnp.float32)).astype(BF16)


def _time(fn, *args, iterations):
    """Median CUPTI kernel time over ``iterations`` calls (after 3 warm-up calls), measured in
    batches of 4: the profiler keeps every output of a batch alive, and some outputs are 2 GB."""
    import statistics

    from jax.experimental.mosaic.gpu import profiler

    fn = jax.jit(fn)
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    samples: list[float] = []
    while len(samples) < iterations:
        _, times = profiler.measure(fn, iterations=min(4, iterations - len(samples)))(*args)
        samples += [t for t in times if t is not None]
    return statistics.median(samples)


def _tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def part_gemms(iterations):
    from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm

    rows = []
    seen = {}
    for arch, proj, k, n in PROJECTIONS:
        flops = 2.0 * TOKENS * k * n
        for form in ("fwd", "dgrad", "wgrad"):
            key = (form, k, n)
            if key not in seen:
                if form == "fwd":
                    a, b = _rand(0, (TOKENS, k)), _rand(1, (k, n))
                    fn = lambda a, b: jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))  # noqa: E731
                elif form == "dgrad":
                    a, b = _rand(0, (TOKENS, n)), _rand(1, (k, n))
                    fn = lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))  # noqa: E731
                else:
                    a, b = _rand(0, (TOKENS, k)), _rand(1, (TOKENS, n))
                    fn = lambda a, b: jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())))  # noqa: E731
                seen[key] = _time(fn, a, b, iterations=iterations)
                del a, b
            ms = seen[key]
            rows.append(dict(arch=arch, projection=proj, form=form, M=TOKENS, K=k, N=n, xla_ms=ms,
                             xla_tflops=_tflops(flops, ms), xla_pct_peak=100 * _tflops(flops, ms) * 1e12 / PEAK))
            print(f"{arch:18s} {proj:12s} {form:5s} K={k:5d} N={n:5d}  {ms:7.3f} ms  "
                  f"{rows[-1]['xla_tflops']:6.1f} TFLOP/s ({rows[-1]['xla_pct_peak']:.1f}%)", flush=True)
    best = []
    cfg = hm.TuningConfig(tile_m=128, tile_n=128, tile_k=64, max_concurrent_steps=4, epi_tile_n=64, epi_tile_m=64,
                          grid_minor_dim=hm.MatmulDimension.N, grid_tile_width=8, wg_dimension=hm.MatmulDimension.N)
    for m, k, n in ((8192, 8192, 8192), (TOKENS, D, 2 * F)):
        a, b = _rand(0, (m, k)), _rand(1, (k, n))
        flops = 2.0 * m * k * n
        xla = _time(lambda a, b: jnp.dot(a, b), a, b, iterations=iterations)
        mgpu = _time(lambda a, b: hm.matmul(a, b, None, cfg), a, b, iterations=iterations)
        best.append(dict(M=m, K=k, N=n, xla_ms=xla, xla_tflops=_tflops(flops, xla),
                         jax_hopper_example_ms=mgpu, jax_hopper_example_tflops=_tflops(flops, mgpu)))
        print(f"best GEMM {m}x{k}x{n}: XLA {_tflops(flops, xla):.1f}, JAX Hopper example "
              f"{_tflops(flops, mgpu):.1f} TFLOP/s", flush=True)
        del a, b
    return dict(projections=rows, best_gemm=best)


def _rmsnorm(x, w):
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + 1e-6) * w).astype(x.dtype)


def _swiglu(gu):
    g, u = gu[:, 0].astype(jnp.float32), gu[:, 1].astype(jnp.float32)
    return (jax.nn.silu(g) * u).astype(gu.dtype)


def part_passes(iterations):
    x = _rand(0, (TOKENS, D))
    r = _rand(1, (TOKENS, D))
    w = jnp.ones((D,), jnp.float32)
    gu = _rand(2, (TOKENS, 2, F))
    dh = _rand(3, (TOKENS, F))
    act = 2 * TOKENS * D  # bytes of one [M, D] bf16 activation
    cases = [
        ("rmsnorm fwd [M, 2048]", lambda x, w: _rmsnorm(x, w), (x, w), 2 * act),
        ("rmsnorm fwd+bwd [M, 2048]", lambda x, w, r: jax.vjp(_rmsnorm, x, w)[1](r), (x, w, r), 5 * act),
        ("residual add [M, 2048]", lambda x, r: x + r, (x, r), 3 * act),
        ("swiglu fwd [M, 2, 5632]", _swiglu, (gu,), 3 * TOKENS * F * 2),
        ("swiglu bwd [M, 2, 5632]", lambda gu, dh: jax.vjp(_swiglu, gu)[1](dh), (gu, dh), 5 * TOKENS * F * 2),
    ]
    rows = []
    for name, fn, args, nbytes in cases:
        ms = _time(fn, *args, iterations=iterations)
        rows.append(dict(name=name, ms=ms, hbm_bytes=nbytes, gb_per_s=nbytes / (ms * 1e-3) / 1e9))
        print(f"{name:28s} {ms:7.3f} ms  {rows[-1]['gb_per_s']:7.0f} GB/s", flush=True)
    return rows


def _xla_gate_up(x, gate_up):
    return _swiglu(jnp.einsum("md,dcf->mcf", x, gate_up))


def _xla_gate_up_saved(x, gate_up):
    gu = jnp.einsum("md,dcf->mcf", x, gate_up)
    return _swiglu(gu), gu


def _xla_dgrad(dy, down, gu):
    dh = jax.lax.dot_general(dy, down, (((1,), (1,)), ((), ())))
    return jax.vjp(_swiglu, gu)[1](dh)[0]


def _xla_mlp(x, gate_up, down):
    a = _xla_gate_up(x, gate_up)
    return jax.lax.dot_general(a, down, (((1,), (0,)), ((), ())))


def part_fused(iterations, sweep):
    x = _rand(0, (TOKENS, D))
    gate_up = _rand(1, (D, 2, F), 0.02)
    down = _rand(2, (F, D), 0.02)
    dy = _rand(3, (TOKENS, D))
    w_il = fg.interleave(gate_up)
    _, gu_il = fg.swiglu_gate_up(x, w_il, save_gu=True)
    gu = fg.deinterleave(gu_il)
    flops_gu = 2.0 * TOKENS * D * 2 * F
    flops_dg = 2.0 * TOKENS * D * F
    out = dict(kernels=[], sweep=[], mlp=[])
    for rnd in range(2):  # interleaved rounds: XLA, fused, XLA, fused
        cases = [
            ("gate_up+swiglu", "xla", lambda x, g: _xla_gate_up(x, g), (x, gate_up), flops_gu),
            ("gate_up+swiglu", "fused", lambda x, w: fg.swiglu_gate_up(x, w, save_gu=False), (x, w_il), flops_gu),
            ("gate_up+swiglu, saves gu", "xla", _xla_gate_up_saved, (x, gate_up), flops_gu),
            ("gate_up+swiglu, saves gu", "fused", lambda x, w: fg.swiglu_gate_up(x, w, save_gu=True), (x, w_il),
             flops_gu),
            ("down dgrad+swiglu bwd", "xla", _xla_dgrad, (dy, down, gu), flops_dg),
            ("down dgrad+swiglu bwd", "fused", fg.swiglu_down_dgrad, (dy, down, gu_il), flops_dg),
        ]
        for name, impl, fn, args, flops in cases:
            ms = _time(fn, *args, iterations=iterations)
            out["kernels"].append(dict(round=rnd, op=name, impl=impl, ms=ms, gemm_tflops=_tflops(flops, ms)))
            print(f"[{rnd}] {name:26s} {impl:5s} {ms:7.3f} ms  ({_tflops(flops, ms):6.1f} GEMM TFLOP/s)", flush=True)
    if sweep:
        for wg_dim, steps, minor, width in itertools.product((0, 1), (2, 3, 4), (0, 1), (8, 16)):
            cfg = fg.GemmConfig(steps=steps, grid_minor=minor, grid_width=width, wg_dim=wg_dim)
            row = dataclasses.asdict(cfg)
            if steps >= 3:  # forward: 48 KB per stage plus 32 KB of epilogue tiles
                try:
                    row["fwd_ms"] = _time(lambda x, w, c=cfg: fg.swiglu_gate_up(x, w, save_gu=True, cfg=c),
                                          x, w_il, iterations=iterations)
                except ValueError as e:
                    row["fwd_error"] = str(e)[:120]
            if steps <= 3:  # backward: 48 KB per stage plus 64 KB of epilogue tiles
                try:
                    row["bwd_ms"] = _time(lambda d, w, g, c=cfg: fg.swiglu_down_dgrad(d, w, g, cfg=c),
                                          dy, down, gu_il, iterations=iterations)
                except ValueError as e:
                    row["bwd_error"] = str(e)[:120]
            out["sweep"].append(row)
            print("sweep", row, flush=True)
    nothing = jax.checkpoint_policies.nothing_saveable
    for rnd in range(2):
        for impl in ("xla", "fused", "fused-recompute"):
            if impl == "xla":
                fn = _xla_mlp
            else:
                fn = functools.partial(fg.swiglu_mlp, compute_dtype=BF16, recompute=impl == "fused-recompute")
            fwd = _time(fn, x, gate_up, down, iterations=iterations)

            def fwd_bwd(x, gu_w, dn, ct, fn=fn):
                y, pull = jax.vjp(fn, x, gu_w, dn)
                return y, pull(ct)

            def remat_fwd_bwd(x, gu_w, dn, ct, fn=fn):  # the block inside remat = "full"
                y, pull = jax.vjp(jax.checkpoint(fn, policy=nothing), x, gu_w, dn)
                return y, pull(ct)

            fb = _time(fwd_bwd, x, gate_up, down, dy, iterations=iterations)
            rfb = _time(remat_fwd_bwd, x, gate_up, down, dy, iterations=iterations)
            out["mlp"].append(dict(round=rnd, impl=impl, fwd_ms=fwd, fwd_bwd_ms=fb, remat_fwd_bwd_ms=rfb))
            print(f"[{rnd}] MLP block {impl:15s} fwd {fwd:7.3f} ms  fwd+bwd {fb:7.3f} ms  "
                  f"remat fwd+bwd {rfb:7.3f} ms", flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", default="gemms,passes,fused")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--sweep", action="store_true", help="also sweep the fused kernels' tuning configs")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    result = dict(env=environment(), gpu_before=gpu_snapshot(), tokens=TOKENS, d_model=D, d_ff=F,
                  fwd_config=dataclasses.asdict(fg.FWD_CONFIG), bwd_config=dataclasses.asdict(fg.BWD_CONFIG))
    parts = args.parts.split(",")
    if "gemms" in parts:
        result["gemms"] = part_gemms(args.iterations)
    if "passes" in parts:
        result["passes"] = part_passes(args.iterations)
    if "fused" in parts:
        result["fused"] = part_fused(args.iterations, args.sweep)
    result["gpu_after"] = gpu_snapshot()
    if args.json:
        write_json(args.json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
