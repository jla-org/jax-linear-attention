# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Mosaic GPU GLA kernels against FLA Triton and XLA.

    python benchmarks/bench_gla.py --torch-python VENV/bin/python

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: ``jla.ops.linear_attention("gla", ..., impl="mosaic_gpu")``, the dedicated
  GLA pipeline (``jla.ops.mosaic_gpu.gla_kernels``);
* ``mosaic_gpu_vg``: the same with ``pipeline="vector_gate"``, the previous path (the KDA / GDN-2
  kernels with constant gates);
* ``chunked``: ``impl="chunked"`` with ``precision=None`` (plain XLA);
* ``fla``: FLA's Triton ``chunk_gla`` (``torch_gla.py``, in the torch environment of ``--torch-python``);
* ``fla_full``: the same call in a second environment (``--torch-python-full``), e.g. FLA main with its
  optional backends installed, so that FLA's own default dispatch picks the kernel.

Metric: median CUDA kernel time per call (``bench_kda.median_kernel_ms``); ``fwdbwd`` is one
forward plus the backward for a fixed output cotangent. ``--docs N`` packs every row with ``N``
documents (``T // N`` tokens each, the last one takes the remainder): ``segment_ids`` for ours,
``cu_seqlens`` over the flattened rows for FLA. Results go to ``benchmarks/results/gla_<date>.json``.
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
from jla.ops import reference as ref  # noqa: E402

SHAPES = [(1, 4096, 8, 128, 128), (4, 4096, 16, 128, 128), (8, 2048, 6, 128, 128), (8, 2048, 32, 128, 128),
          (4, 4096, 16, 64, 64)]

# name -> (impl, extra options): the dedicated Mosaic GPU pipeline (the default), the previous one
# (the KDA / GDN-2 kernels with constant gates) and plain XLA.
_JAX_IMPLS = {
    "mosaic_gpu": ("mosaic_gpu", {}),
    "mosaic_gpu_vg": ("mosaic_gpu", dict(pipeline="vector_gate")),
    "chunked": ("chunked", dict(precision=None)),
}


def inputs(shape) -> dict[str, jax.Array]:
    """bf16 ``q, k, v`` (unit-norm ``k``) and the per-channel log-decay ``g = logsigmoid(N(0, 1))``."""
    batch, seq_len, heads, key_dim, value_dim = shape
    ks = jax.random.split(jax.random.key(0), 5)
    f32, bf16 = jnp.float32, jnp.bfloat16
    return dict(
        q=jax.random.normal(ks[0], (batch, seq_len, heads, key_dim), f32).astype(bf16),
        k=ref.l2norm(jax.random.normal(ks[1], (batch, seq_len, heads, key_dim), f32)).astype(bf16),
        v=jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), f32).astype(bf16),
        g=jax.nn.log_sigmoid(jax.random.normal(ks[3], (batch, seq_len, heads, key_dim), f32)),
        do=jax.random.normal(ks[4], (batch, seq_len, heads, value_dim), f32).astype(bf16),
    )


def segment_ids(shape, docs: int) -> jax.Array | None:
    """``[B, T]`` ids of ``docs`` documents per row (``T // docs`` tokens each, the last one the rest)."""
    if docs <= 1:
        return None
    batch, seq_len = shape[:2]
    ids = jnp.minimum(jnp.arange(seq_len) // (seq_len // docs), docs - 1).astype(jnp.int32)
    return jnp.broadcast_to(ids, (batch, seq_len))


def jax_impls(shape, impls: list[str], docs: int = 0):
    """``name -> (fwd_fn, fwdbwd_fn, args, cotangent)`` for the JAX implementations."""
    x = inputs(shape)
    seg = segment_ids(shape, docs)
    out = {}
    for name in impls:
        if name not in _JAX_IMPLS:
            continue

        def fn(q, k, v, g, name=name):
            impl, kw = _JAX_IMPLS[name]
            return ops.linear_attention("gla", q, k, v, g=g, impl=impl, segment_ids=seg, **kw)

        args = (x["q"], x["k"], x["v"], x["g"])
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        out[name] = (fwd, harness.vjp_fn(fn, tuple(range(len(args)))), args, x["do"])
    return out


def run_torch(torch_python: str, shape, mode: str, iterations: int, docs: int = 0) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_gla.py")
    cmd = [torch_python, script, "--shape", ",".join(map(str, shape)), "--mode", mode, "--iterations", str(iterations),
           "--docs", str(docs)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA")
    parser.add_argument("--torch-python-full", default=None, help="a second FLA environment (impl fla_full)")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default="mosaic_gpu,fla,chunked", help="comma-separated implementations")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--docs", type=int, default=0, help="documents packed into every row (0: dense rows)")
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    impl_names = args.impls.split(",")
    modes = args.modes.split(",")
    results = dict(kind="gla", environment=harness.environment(), metric="median CUDA kernel ms per call",
                   rounds=args.rounds, iterations=args.iterations, docs=args.docs, entries=[])
    for shape in shapes:
        jimpls = jax_impls(shape, impl_names, args.docs)
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
                    elif name == "fla" and args.torch_python:
                        r = run_torch(args.torch_python, shape, mode, args.iterations, args.docs)
                    elif name == "fla_full" and args.torch_python_full:
                        r = run_torch(args.torch_python_full, shape, mode, args.iterations, args.docs)
                    else:
                        continue
                    r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind")}
                    entry = dict(shape=shape, mode=mode, impl=name, round=rnd, wall_s=round(time.time() - t0, 1),
                                 gpu_before=before, gpu_after=harness.gpu_snapshot(), **r)
                    results["entries"].append(entry)
                    print(json.dumps({k: entry[k] for k in ("shape", "mode", "impl", "round")}
                                     | {k: entry.get(k) for k in ("median_ms", "error")}), flush=True)
    date = time.strftime("%Y%m%d_%H%M")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"gla_{date}.json")
    harness.write_json(out, results)
    bench_kda.summarize(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
