# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Mosaic GPU mLSTM path against the chunked XLA op and NX-AI's TFLA Triton kernels.

    python benchmarks/bench_mlstm.py --torch-python VENV/bin/python

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: :func:`jla.ops.mosaic_gpu.mlstm.mlstm` (simple-GLA kernels plus the float32
  normaliser in XLA), value blocks of 128 columns; ``mosaic_gpu_vb64``: value blocks of 64;
* ``chunked``: ``impl="chunked"`` with ``precision=None`` (plain XLA), the previous path;
* ``tfla_xl128`` / ``tfla_xl256``: ``mlstm_kernels`` ``chunkwise--triton_xl_chunk`` (TFLA, mLSTMexp)
  with chunk 128 / 256; ``tfla_limit64``: ``chunkwise--triton_limit_chunk`` with chunk 64
  (``torch_mlstm.py``, in a separate torch environment with ``mlstm_kernels``, which is under the
  NXAI Community License Agreement, not a permissive licence: see ``torch_mlstm.py``).

Metric: median CUDA kernel time per call (CUPTI / torch.profiler); ``fwdbwd`` is one forward
plus the backward for a fixed output cotangent, gradients for ``q, k, v`` and both gates.
Results go to ``benchmarks/results/mlstm_<date>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_kda  # noqa: E402
import harness  # noqa: E402

from jla import ops  # noqa: E402
from jla.ops.mosaic_gpu import mlstm as mm  # noqa: E402

# xLSTM-7B heads (d 4096, 8 heads: qk 256, v 512) at 8K and 16K tokens per batch, and a 128/256-head shape.
SHAPES = [(1, 8192, 8, 256, 512), (2, 4096, 8, 256, 512), (4, 4096, 8, 128, 256)]
TORCH = {"tfla_xl128": ("chunkwise--triton_xl_chunk", 128), "tfla_xl256": ("chunkwise--triton_xl_chunk", 256),
         "tfla_limit64": ("chunkwise--triton_limit_chunk", 64)}


def inputs(shape) -> dict[str, jax.Array]:
    """bf16 ``q, k, v``, ``g = logsigmoid(3 N + 2)``, ``i = 3 N`` (float32) and a bf16 cotangent."""
    batch, seq_len, heads, key_dim, value_dim = shape
    ks = jax.random.split(jax.random.key(0), 6)
    f32, bf16 = jnp.float32, jnp.bfloat16
    return dict(
        q=jax.random.normal(ks[0], (batch, seq_len, heads, key_dim), f32).astype(bf16),
        k=jax.random.normal(ks[1], (batch, seq_len, heads, key_dim), f32).astype(bf16),
        v=jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), f32).astype(bf16),
        g=jax.nn.log_sigmoid(3.0 * jax.random.normal(ks[3], (batch, seq_len, heads), f32) + 2.0),
        i=3.0 * jax.random.normal(ks[4], (batch, seq_len, heads), f32),
        do=jax.random.normal(ks[5], (batch, seq_len, heads, value_dim), f32).astype(bf16),
    )


def jax_impls(shape, impls: list[str]):
    """``name -> (fwd_fn, fwdbwd_fn, args, cotangent)`` for the JAX implementations."""
    x = inputs(shape)
    out = {}
    for name in impls:
        if name == "mosaic_gpu":
            def fn(q, k, v, g, i):
                return mm.mlstm(q, k, v, g, i)
        elif name == "mosaic_gpu_vb64":
            def fn(q, k, v, g, i):
                return mm.mlstm(q, k, v, g, i, value_block=64)
        elif name == "chunked":
            def fn(q, k, v, g, i):
                return ops.linear_attention("mlstm", q, k, v, g=g, i=i, impl="chunked", precision=None)
        else:
            continue
        args = (x["q"], x["k"], x["v"], x["g"], x["i"])
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        out[name] = (fwd, harness.vjp_fn(fn, tuple(range(len(args)))), args, x["do"])
    return out


def run_torch(torch_python: str, shape, mode: str, iterations: int, kernel: str, chunk: int) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_mlstm.py")
    cmd = [torch_python, script, "--shape", ",".join(map(str, shape)), "--mode", mode, "--iterations", str(iterations),
           "--kernel", kernel, "--chunk", str(chunk)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + mlstm_kernels")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default="mosaic_gpu,mosaic_gpu_vb64,chunked,tfla_xl128,tfla_xl256,tfla_limit64")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--breakdown", action="store_true", help="also record per-kernel times of mosaic_gpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    impl_names = args.impls.split(",")
    modes = args.modes.split(",")
    results = dict(kind="mlstm", environment=harness.environment(), metric="median CUDA kernel ms per call",
                   rounds=args.rounds, iterations=args.iterations, entries=[], breakdown=[])
    for shape in shapes:
        jimpls = jax_impls(shape, impl_names)
        for mode in modes:
            for rnd in range(args.rounds):
                for name in impl_names:
                    before = harness.gpu_snapshot()
                    t0 = time.time()
                    if name in jimpls:
                        fwd, bwd, fargs, do = jimpls[name]
                        try:
                            if mode == "fwd":
                                r = bench_kda.median_kernel_ms(fwd, *fargs, iterations=args.iterations)
                            else:
                                r = bench_kda.median_kernel_ms(bwd, *fargs, do, iterations=args.iterations)
                        except Exception as err:  # noqa: BLE001  (record and continue)
                            r = dict(error=f"{type(err).__name__}: {str(err)[:500]}")
                    elif name in TORCH and args.torch_python:
                        r = run_torch(args.torch_python, shape, mode, args.iterations, *TORCH[name])
                    else:
                        continue
                    r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind")}
                    entry = dict(shape=shape, mode=mode, impl=name, round=rnd, wall_s=round(time.time() - t0, 1),
                                 gpu_before=before, gpu_after=harness.gpu_snapshot(), **r)
                    results["entries"].append(entry)
                    print(json.dumps({k: entry[k] for k in ("shape", "mode", "impl", "round")}
                                     | {k: entry.get(k) for k in ("median_ms", "error")}), flush=True)
            if args.breakdown and "mosaic_gpu" in jimpls:
                fwd, bwd, fargs, do = jimpls["mosaic_gpu"]
                fn, call_args = (fwd, fargs) if mode == "fwd" else (bwd, (*fargs, do))
                per = harness.kernel_breakdown_ms(fn, *call_args)
                results["breakdown"].append(dict(shape=shape, mode=mode, kernels_ms=per))
                print(json.dumps(dict(shape=shape, mode=mode, top=dict(list(per.items())[:8]))), flush=True)
    date = time.strftime("%Y%m%d_%H%M")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"mlstm_{date}.json")
    harness.write_json(out, results)
    bench_kda.summarize(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
