# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head cross entropy: the fused Mosaic GPU kernels, the XLA op, Liger, Cut Cross-Entropy and FLA.

    python benchmarks/bench_cross_entropy.py --torch-python VENV/bin/python \\
        --shapes 16384,768,128256 16384,2048,128256 --json results/cross_entropy.json

For every ``N,D,V`` shape: GPU kernel time (CUPTI, median of ``--iterations`` calls after
3 warmups) of the forward (loss only) and of forward + backward (loss, ``dhidden`` and
``dweight``) of ``jla.ops.cross_entropy.linear_cross_entropy`` with ``impl="mosaic_gpu"``
and ``impl="xla"``, interleaved on the same GPU with the torch baselines of
``torch_cross_entropy.py`` (run in ``--torch-python``). Inputs are bfloat16 ``hidden`` and
``weight`` (as the torch baselines take them), uniform random targets, mean reduction, no
label smoothing and no z-loss. ``nvidia-smi`` utilisation of the visible GPU is recorded
before and after each shape. FLA's kinds are timed forward + backward only (its forward call
always forms the gradients). ``--cluster-ab R`` adds R interleaved rounds of the kernels
compiled with 1- and 2-CTA clusters (``_Config.cluster``). ``--objective-rows R ...`` adds
``linear_cross_entropy_objective`` (the gradient formed in the forward pass, blocks of ``R`` rows)
with ``impl="mosaic_gpu"``; its forward is the loss without differentiation (the statistics
kernel), its forward + backward the value and gradients. ``--objective-fp8-rows R ...`` adds the
same with ``fp8=True`` and ``--skip-xla`` leaves out the XLA op. ``--accuracy-rows N`` skips the
timing and instead reports, for ``N`` rows of every shape, the errors of the loss and both
gradients of each impl and baseline against float32 logits of the same bfloat16 values.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from jla.ops.cross_entropy import (  # noqa: E402
    cross_entropy_from_logits,
    linear_cross_entropy,
    linear_cross_entropy_objective,
)


def jax_fns(impl: str, chunk: int):
    def loss(h, w, t):
        out = linear_cross_entropy(h, w, t, vocab_chunk=chunk, impl=impl)
        return out.loss_sum / out.num_valid

    return jax.jit(loss), jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def objective_fns(rows: int, fp8: bool = False):
    def loss(h, w, t):
        out = linear_cross_entropy_objective(h, w, t, row_chunk=rows, impl="mosaic_gpu", fp8=fp8)
        return out.objective_sum / out.num_valid

    return jax.jit(loss), jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def inputs(n: int, d: int, v: int):
    ks = jax.random.split(jax.random.key(0), 3)
    hidden = jax.random.normal(ks[0], (n, d), jnp.float32).astype(jnp.bfloat16)
    weight = (jax.random.normal(ks[1], (v, d), jnp.float32) * (2.0 / d ** 0.5)).astype(jnp.bfloat16)
    targets = jax.random.randint(ks[2], (n,), 0, v, jnp.int32)
    return hidden, weight, targets


def accuracy(impl: str, chunk: int, hidden, weight, targets) -> dict:
    """Errors of the mean loss and its gradients against float32 logits of the same bf16 values."""
    def oracle(h, w, t):
        with jax.default_matmul_precision("highest"):
            out = cross_entropy_from_logits(h.astype(jnp.float32) @ w.astype(jnp.float32).T, t)
        return out.loss_sum / out.num_valid

    ref, (ref_dh, ref_dw) = jax.jit(jax.value_and_grad(oracle, argnums=(0, 1)))(
        hidden.astype(jnp.float32), weight.astype(jnp.float32), targets)
    loss, (dh, dw) = jax_fns(impl, chunk)[1](hidden, weight, targets)

    def rel(a, b):
        return float(jnp.linalg.norm(a.astype(jnp.float32) - b) / jnp.linalg.norm(b))

    return dict(loss_abs=abs(float(loss) - float(ref)), dhidden_rel=rel(dh, ref_dh), dweight_rel=rel(dw, ref_dw),
                bf16_floor_dhidden=rel(ref_dh.astype(jnp.bfloat16), ref_dh),
                bf16_floor_dweight=rel(ref_dw.astype(jnp.bfloat16), ref_dw), grad_dtypes=[str(dh.dtype), str(dw.dtype)])


def run_accuracy(args) -> None:
    results = []
    for spec in args.shapes:
        _, d, v = (int(x) for x in spec.split(","))
        n = args.accuracy_rows
        hidden, weight, targets = inputs(n, d, v)
        row = dict(shape=[n, d, v], impls={impl: accuracy(impl, args.chunk, hidden, weight, targets)
                                           for impl in ("mosaic_gpu", "xla")})
        if args.torch_python:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_cross_entropy.py")
            for kind in args.torch_kinds.split(","):
                cmd = [args.torch_python, script, "--kind", kind, "--shape", f"{n},{d},{v}", "--accuracy"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
                lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
                row["impls"][kind] = (json.loads(lines[-1]) if proc.returncode == 0 and lines
                                      else dict(error=harness.error_summary(proc.stderr or proc.stdout)))
        print(json.dumps(row), flush=True)
        results.append(row)
    if args.json:
        harness.write_json(args.json, dict(env=harness.environment(), accuracy_rows=args.accuracy_rows,
                                           chunk=args.chunk, results=results))


def cluster_ab(hidden, weight, targets, chunk: int, rounds: int, iterations: int) -> dict:
    """``rounds`` interleaved timings of ``mosaic_gpu`` compiled with cluster sizes 1 and 2."""
    from jla.ops.mosaic_gpu import linear_cross_entropy as fused

    saved = fused._STATS, fused._GRAD
    compiled = {}
    try:
        for cluster in (1, 2):  # the configs are read at trace time, so compile under each one
            fused._STATS, fused._GRAD = (dataclasses.replace(cfg, cluster=cluster) for cfg in saved)
            fns = jax_fns("mosaic_gpu", chunk)
            for fn in fns:
                jax.block_until_ready(fn(hidden, weight, targets))
            compiled[cluster] = fns
    finally:
        fused._STATS, fused._GRAD = saved
    out = {f"cluster{c}": {"fwd": [], "fwdbwd": []} for c in compiled}
    for _ in range(rounds):
        for cluster, (fwd, fwdbwd) in compiled.items():
            for mode, fn in (("fwd", fwd), ("fwdbwd", fwdbwd)):
                timing = harness.median_kernel_ms(fn, hidden, weight, targets, iterations=iterations)
                out[f"cluster{cluster}"][mode].append(timing["median_ms"])
    return out


def run_torch(torch_python: str, kind: str, shape, mode: str, iterations: int) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_cross_entropy.py")
    cmd = [torch_python, script, "--kind", kind, "--shape", ",".join(map(str, shape)), "--mode", mode,
           "--iterations", str(iterations)]
    # The baseline inherits this process's environment, so it runs on the same GPU.
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["16384,768,128256", "16384,1024,128256", "16384,2048,128256"])
    parser.add_argument("--chunk", type=int, default=4096, help="vocab_chunk of both JAX impls")
    parser.add_argument("--mosaic-chunks", type=int, nargs="*", default=[], help="extra mosaic_gpu vocab_chunk values")
    parser.add_argument("--torch-python", default=None)
    parser.add_argument("--torch-kinds", default="liger,cce,cce_exact,fla,fla_tf32,fla_bf16acc")
    parser.add_argument("--cluster-ab", type=int, default=0, help="interleaved rounds of cluster 1 vs 2")
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--breakdown", action="store_true", help="per-kernel times of mosaic_gpu fwd+bwd")
    parser.add_argument("--accuracy-rows", type=int, default=0, help="errors at this many rows instead of timing")
    parser.add_argument("--objective-rows", type=int, nargs="*", default=[],
                        help="row_chunk values of linear_cross_entropy_objective (forward-pass gradient)")
    parser.add_argument("--objective-fp8-rows", type=int, nargs="*", default=[],
                        help="row_chunk values of linear_cross_entropy_objective with fp8=True")
    parser.add_argument("--skip-xla", action="store_true", help="time only the Mosaic GPU paths")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.accuracy_rows:
        run_accuracy(args)
        return

    results = []
    for spec in args.shapes:
        n, d, v = (int(x) for x in spec.split(","))
        hidden, weight, targets = inputs(n, d, v)
        row = dict(shape=[n, d, v], gpu_before=harness.gpu_snapshot(), impls={})
        variants = [("mosaic_gpu", args.chunk)] + ([] if args.skip_xla else [("xla", args.chunk)])
        variants += [("mosaic_gpu", c) for c in args.mosaic_chunks]
        variants += [("objective", rows) for rows in args.objective_rows]
        variants += [("objective_fp8", rows) for rows in args.objective_fp8_rows]
        for impl, chunk in variants:
            if impl.startswith("objective"):
                name = f"{impl}_rows{chunk}"
                fwd, fwdbwd = objective_fns(chunk, fp8=impl == "objective_fp8")
            else:
                name = f"{impl}" if chunk == args.chunk else f"{impl}_chunk{chunk}"
                fwd, fwdbwd = jax_fns(impl, chunk)
            entry = dict(loss=float(fwd(hidden, weight, targets)))
            entry["fwd"] = harness.median_kernel_ms(fwd, hidden, weight, targets, iterations=args.iterations)
            entry["fwdbwd"] = harness.median_kernel_ms(fwdbwd, hidden, weight, targets, iterations=args.iterations)
            if args.breakdown and (impl.startswith("objective") or (impl == "mosaic_gpu" and chunk == args.chunk)):
                entry["fwdbwd_kernels"] = harness.kernel_breakdown_ms(fwdbwd, hidden, weight, targets)
            row["impls"][name] = entry
            print(json.dumps(dict(shape=row["shape"], impl=name, loss=entry["loss"],
                                  fwd_ms=entry["fwd"]["median_ms"], fwdbwd_ms=entry["fwdbwd"]["median_ms"])),
                  flush=True)
        if args.torch_python:
            for kind in args.torch_kinds.split(","):
                modes = ("fwdbwd",) if kind.startswith("fla") else ("fwd", "fwdbwd")
                entry = {mode: run_torch(args.torch_python, kind, (n, d, v), mode, args.iterations) for mode in modes}
                row["impls"][kind] = entry
                print(json.dumps(dict(shape=row["shape"], impl=kind, **{
                    f"{mode}_ms": e.get("median_ms", e.get("error")) for mode, e in entry.items()})), flush=True)
        if args.cluster_ab:
            row["cluster_ab"] = cluster_ab(hidden, weight, targets, args.chunk, args.cluster_ab, args.iterations)
            print(json.dumps(dict(shape=row["shape"], cluster_ab=row["cluster_ab"])), flush=True)
        row["gpu_after"] = harness.gpu_snapshot()
        ours = row["impls"]["mosaic_gpu"]
        row["speedup_fwdbwd"] = {
            k: e["fwdbwd"]["median_ms"] / ours["fwdbwd"]["median_ms"]
            for k, e in row["impls"].items() if "median_ms" in e.get("fwdbwd", {}) and k != "mosaic_gpu"}
        print(json.dumps(dict(shape=row["shape"], speedup_fwdbwd_vs=row["speedup_fwdbwd"])), flush=True)
        results.append(row)
    if args.json:
        harness.write_json(args.json, dict(env=harness.environment(), iterations=args.iterations,
                                           chunk=args.chunk, results=results))


if __name__ == "__main__":
    main()
