# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped GEMM (ragged dot) for MoE: Mosaic GPU kernels vs XLA, torch and padded einsum baselines.

Per shape: the expert up projection ``[routes, D] x [E, D, 2F]`` and down projection
``[routes, F] x [E, F, D]``, forward and forward+backward (``d_lhs``, ``d_rhs``), with
group sizes from top-k routing of random router logits (uneven, some experts can be
empty). Implementations, interleaved per case on one GPU:

* ``mosaic_gpu``: our kernels with the default tiling (``ragged_dot.select_config``, recorded
  per case in ``config_fwd`` / ``config_dlhs``);
* ``mosaic_gpu_block_aligned``: the block-aligned kernel (``ragged_dot.BLOCK_ALIGNED``, the default until 18 Oct 2026);
* ``mosaic_gpu_aligned``, ``mosaic_gpu_ws``, ``mosaic_gpu_persistent`` and ``mosaic_gpu_persistent_ws``:
  the tilings ``ragged_dot.ALIGNED``, ``ragged_dot.WARP_SPECIALIZED``, ``ragged_dot.PERSISTENT`` and
  ``ragged_dot.PERSISTENT_WS``, where they divide both the forward and the ``d_lhs`` GEMM (the column
  of the default tiling, ``PERSISTENT`` since 18 Oct 2026, times the same kernel as ``mosaic_gpu``:
  a noise check);
* ``xla``: ``jax.lax.ragged_dot`` (dense per-group products in JAX 0.11.2; ``--no-xla``);
* ``padded_einsum_cf1.25``: the capacity path's GEMM, a batched einsum over
  ``[E, ceil(1.25 routes / E), K]`` (25% padding; overflowing routes would be dropped);
* ``torch_grouped_mm_{kn,nk}``: ``torch._grouped_mm`` (CUTLASS sm90) with the weight stored
  ``[G, K, N]`` or ``[G, N, K]`` (``torch_grouped_mm.py``, run with ``--torch-python``);
  ``--grouped-gemm`` adds tgale96/grouped_gemm when that environment has it.

And the whole MoE layer (route, dispatch, experts, combine, shared experts) forward+backward:
capacity 1.25 einsum, dropless XLA, dropless Mosaic GPU (default, ``BLOCK_ALIGNED``, ``ALIGNED``,
``WARP_SPECIALIZED``, ``PERSISTENT`` and ``PERSISTENT_WS`` tilings, each where it divides a GEMM) and the same dropless layer in torch
on ``torch._grouped_mm``, eager and ``torch.compile``d. The JAX layer variants are timed in
``--layer-rounds`` interleaved rounds whose order rotates every round (layer timings drift by
more than the GEMM differences on a shared GPU); each reports the median over all its calls and
the per-round medians. GPU kernel time per call (CUPTI / torch profiler, median), see
``benchmarks/harness.py``; ``nvidia-smi`` of the visible GPU before and after every shape.
Results are written after every shape.

    python benchmarks/bench_ragged_dot.py --iterations 30 --no-xla --torch-python VENV/bin/python \\
        --grouped-gemm --shapes moe-125m,moe125m-like-f512,deepseek-v2-lite-like,deepseek-v3-like-ep8 \\
        --out benchmarks/results/ragged_dot.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import environment, error_summary, gpu_snapshot, median_kernel_ms, vjp_fn  # noqa: E402

from jla.layers.moe import MoE  # noqa: E402
from jla.models.config import MoEConfig  # noqa: E402
from jla.ops.mosaic_gpu import ragged_dot as rd  # noqa: E402

# (name, tokens, d_model, experts, top_k, d_ff_expert, shared experts)
SHAPES = [
    ("moe-125m", 8192, 768, 16, 2, 768, 1),  # the moe-125m preset's layer, 8192 tokens
    ("moe125m-like-f256", 8192, 768, 64, 8, 256, 0),
    ("moe125m-like-f512", 8192, 768, 64, 8, 512, 0),
    ("moe125m-like-f1024", 8192, 768, 64, 8, 1024, 0),
    ("deepseek-v2-lite-like", 4096, 2048, 64, 6, 1408, 2),
    ("deepseek-v3-like-ep8", 4096, 7168, 32, 8, 2048, 0),  # 32 local experts: 256 sharded over 8 GPUs
    # The shipped MoE presets' layers at 16384 tokens (8 x 2048), and Qwen3-Next-80B-A3B's under EP 8.
    ("qwen3-next-300m", 16384, 1024, 64, 10, 256, 1),
    ("kimi-linear-1b", 16384, 1024, 32, 4, 512, 1),
    ("granite-4.0-h-125m", 16384, 768, 64, 6, 256, 2),
    ("gpt-oss-180m", 16384, 768, 32, 4, 768, 0),
    ("qwen3-next-80b-ep8", 4096, 2048, 64, 10, 512, 0),  # 64 local experts: 512 sharded over 8 GPUs
]
CAPACITY = 1.25
# Other tilings timed next to the default (columns ``mosaic_gpu_<label>``). A tiling also tiles
# ``d_lhs``, so it runs only where it divides both GEMMs; the layer uses it wherever it does.
EXTRA = {
    "aligned": rd.ALIGNED,
    "ws": rd.WARP_SPECIALIZED,
    "persistent": rd.PERSISTENT,
    "persistent_ws": rd.PERSISTENT_WS,
}
LAYER_EXTRA = ("aligned", "ws", "persistent", "persistent_ws")
SKIPPED_XLA = "not run (--no-xla): XLA ragged_dot expands to dense per-group products and runs out of memory"


def _group_sizes(tokens, experts, top_k, seed=0):
    logits = jax.random.normal(jax.random.key(seed), (tokens, experts)) + 0.5 * jax.random.normal(
        jax.random.key(seed + 1), (experts,))
    _, idx = jax.lax.top_k(logits, top_k)
    return jnp.zeros((experts,), jnp.int32).at[idx.reshape(-1)].add(1)


OUTPUT_BUDGET = 3 << 30  # bytes of outputs alive at once while timing


def _kernel_times(fn, *args, iterations, warmup=3):
    """Per-call CUPTI kernel ms of ``iterations`` calls after ``warmup`` (``harness.median_kernel_ms``'s
    metric) in chunks: the profiler keeps every timed call's outputs alive, which at the large shapes
    (gigabytes of weight gradients per call) exceeds the memory cap."""
    from jax.experimental.mosaic.gpu import profiler

    out_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(jax.eval_shape(fn, *args)))
    chunk = max(1, min(iterations, OUTPUT_BUDGET // max(out_bytes, 1)))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(4 * iterations):  # bounded: a call the profiler could not time returns None
        if len(times) >= iterations:
            break
        _, t = profiler.measure(fn, iterations=min(chunk, iterations - len(times)))(*args)
        times += [x for x in (t if isinstance(t, list) else [t]) if x is not None]
    if not times:
        raise RuntimeError("the profiler timed no call")
    return times, chunk


def _chunked_kernel_ms(fn, *args, iterations):
    if OUTPUT_BUDGET // max(sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(jax.eval_shape(fn, *args))),
                            1) >= iterations:
        return median_kernel_ms(fn, *args, iterations=iterations)
    times, chunk = _kernel_times(fn, *args, iterations=iterations)
    return dict(median_ms=float(np.median(times)), min_ms=min(times), iterations=len(times), chunk=chunk)


def _timed(fn, *args, iterations):
    """Median kernel time, or the error (e.g. an XLA out-of-memory) so one case cannot stop the run."""
    try:
        return _chunked_kernel_ms(fn, *args, iterations=iterations)
    except Exception as err:  # noqa: BLE001 - recorded in the results
        return dict(error=f"{type(err).__name__}: {str(err)[:300]}")


def _run_torch(torch_python, argv, timeout=900) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_grouped_mm.py")
    try:
        proc = subprocess.run([torch_python, script, *argv], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(error=f"torch baseline timed out after {timeout} s")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def _padded_einsum(routes, k, n, experts, iterations):
    """The capacity path's expert GEMM: ``[E, C, K] x [E, K, N]`` with ``C = ceil(1.25 routes / E)``."""
    cap = math.ceil(CAPACITY * routes / experts)
    k1, k2, k3 = jax.random.split(jax.random.key(11), 3)
    lhs = jax.random.normal(k1, (experts, cap, k), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k2, (experts, k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
    ct = jax.random.normal(k3, (experts, cap, n), jnp.float32).astype(jnp.bfloat16)

    def fn(a, b):
        return (jnp.einsum("eck,ekn->ecn", a, b),)

    return (_timed(jax.jit(lambda a, b: fn(a, b)[0]), lhs, rhs, iterations=iterations),
            _timed(vjp_fn(fn, (0, 1)), lhs, rhs, ct, iterations=iterations), cap)


def _gemm_case(routes, k, n, experts, sizes, args):
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    lhs = jax.random.normal(k1, (routes, k), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k2, (experts, k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
    ct = jax.random.normal(k3, (routes, n), jnp.float32).astype(jnp.bfloat16)
    impls = {
        "mosaic_gpu": lambda a, b, g: (rd.ragged_dot(a, b, g, full_groups=True),),
        "mosaic_gpu_block_aligned": lambda a, b, g: (rd.ragged_dot(a, b, g, full_groups=True, config=rd.BLOCK_ALIGNED),),
    }
    for label, cfg in EXTRA.items():
        if cfg.divides_both(k, n):  # the tiling also runs d_lhs
            impls[f"mosaic_gpu_{label}"] = lambda a, b, g, cfg=cfg: (
                rd.ragged_dot(a, b, g, full_groups=True, config=cfg),)
    if not args.no_xla:
        impls["xla"] = lambda a, b, g: (jax.lax.ragged_dot(a, b, g),)
    row = dict(config_fwd=dataclasses.asdict(rd.select_config(k, n)),
               config_dlhs=dataclasses.asdict(rd.select_config(n, k)))
    for name, fn in impls.items():
        fwd = jax.jit(lambda a, b, g, fn=fn: fn(a, b, g)[0])
        row[f"{name}_fwd"] = _timed(fwd, lhs, rhs, sizes, iterations=args.iterations)
        if name == "xla" and args.no_xla_bwd:
            row[f"{name}_fwd_bwd"] = dict(skipped="not run (--no-xla-bwd)")
        else:
            row[f"{name}_fwd_bwd"] = _timed(vjp_fn(fn, (0, 1)), lhs, rhs, sizes, ct, iterations=args.iterations)
    if args.no_xla:
        row["xla_fwd"] = row["xla_fwd_bwd"] = dict(skipped=SKIPPED_XLA)
    fwd, fwd_bwd, cap = _padded_einsum(routes, k, n, experts, args.iterations)
    row["padded_einsum_cf1.25_fwd"], row["padded_einsum_cf1.25_fwd_bwd"] = fwd, fwd_bwd
    row["padded_capacity"] = cap
    if args.torch_python:
        argv = ["gemm", "--k", str(k), "--n", str(n), "--sizes", ",".join(str(int(s)) for s in np.asarray(sizes)),
                "--iterations", str(args.iterations)] + (["--grouped-gemm"] if args.grouped_gemm else [])
        row.update(_run_torch(args.torch_python, argv))
    flops = 2.0 * routes * k * n  # useful (dropless) FLOPs of one GEMM
    for key in list(row):
        if isinstance(row[key], dict) and "median_ms" in row[key]:
            mult = 3.0 if key.endswith("fwd_bwd") else 1.0
            row[key]["tflops"] = mult * flops / (row[key]["median_ms"] * 1e-3) / 1e12
    for mode in ("fwd", "fwd_bwd"):
        ours = row[f"mosaic_gpu_{mode}"].get("median_ms")
        for base in ("mosaic_gpu_block_aligned", *(f"mosaic_gpu_{label}" for label in EXTRA), "xla", "padded_einsum_cf1.25",
                     "torch_grouped_mm_kn", "torch_grouped_mm_nk", "grouped_gemm"):
            other = row.get(f"{base}_{mode}", {}).get("median_ms")
            if ours and other:
                row[f"speedup_{mode}_vs_{base}"] = other / ours
    return row


# A first sweep (moe-125m, moe125m-like-f512) also tried 64-row, 64 x 256 and 128-deep tiles on one
# warpgroup: all slower than 128 x 128 x 64.
SWEEP = [
    rd.BLOCK_ALIGNED,
    rd.TileConfig(schedule="aligned"),
    rd.ALIGNED,
    rd.TileConfig(schedule="aligned", steps=5, grid_block_n=4),
    rd.WARP_SPECIALIZED,
    rd.TileConfig(schedule="aligned", warpgroups=2, steps=3, grid_block_n=4),
    rd.TileConfig(schedule="aligned", block_m=64, warpgroups=2),
]


def _sweep_case(routes, k, n, experts, sizes, args):
    """Forward (``x @ w``) and ``d_lhs`` (``dy @ w^T``) kernel time of every tiling in ``SWEEP``."""
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    x = jax.random.normal(k1, (routes, k), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (experts, k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
    dy = jax.random.normal(k3, (routes, n), jnp.float32).astype(jnp.bfloat16)
    rows = []
    for cfg in SWEEP:
        for direction, lhs, contract, width in (("fwd", x, k, n), ("dlhs", dy, n, k)):
            if not cfg.divides(contract, width):
                continue
            fn = jax.jit(lambda a, b, g, cfg=cfg, t=direction == "dlhs": rd._ragged_dot_kernel(
                a, b, g, transpose_rhs=t, cfg=cfg))
            r = _timed(fn, lhs, w, sizes, iterations=args.iterations)
            rows.append(dict(direction=direction, config=dataclasses.asdict(cfg), **r))
    return rows


def _moe_with(config):
    """The MoE layer with ``config`` for every grouped GEMM it divides (the default tiling elsewhere)."""

    class _ConfigMoE(MoE):
        def _ragged_dot(self, lhs, rhs, group_sizes):
            k, n = rhs.shape[1], rhs.shape[2]
            cfg = config if config.divides_both(k, n) else None
            return rd.ragged_dot(lhs, rhs, group_sizes, full_groups=True, config=cfg)

    return _ConfigMoE


def _layer_case(tokens, d, experts, top_k, f, shared, args):
    base = MoEConfig(num_experts=experts, top_k=top_k, d_ff_expert=f, num_shared_experts=shared)
    mosaic = dataclasses.replace(base, ragged_dot_impl="mosaic_gpu")
    variants = {
        "capacity_1.25_einsum": (MoE, dataclasses.replace(base, capacity_factor=CAPACITY)),
        "dropless_mosaic_gpu": (MoE, mosaic),
        "dropless_mosaic_gpu_block_aligned": (_moe_with(rd.BLOCK_ALIGNED), mosaic),
        **{f"dropless_mosaic_gpu_{label}": (_moe_with(EXTRA[label]), mosaic) for label in LAYER_EXTRA},
    }
    if not args.no_xla:
        variants["dropless_xla"] = (MoE, base)
    x = jax.random.normal(jax.random.key(3), (tokens, d), jnp.float32).astype(jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(4), (tokens, d), jnp.float32).astype(jnp.bfloat16)
    fn = vjp_fn(lambda m, x: (m(x)[0],), (0, 1))  # one jit; each variant's class and config trace their own

    def build(name):  # one variant's parameters alive at a time (2.2 GB of float32 weights at V2-Lite)
        cls, cfg = variants[name]
        return cls(d, cfg, key=jax.random.key(0), std=0.02, param_dtype=jnp.float32, compute_dtype="bfloat16")

    names = list(variants)
    times = {name: [] for name in names}
    rounds = {name: [] for name in names}
    row = {}
    dropped = None
    for r in range(max(1, args.layer_rounds)):
        shift = r % len(names)
        for name in names[shift:] + names[:shift]:  # rotate the order: no variant always runs first
            if "error" in row.get(name, {}):
                continue
            moe = build(name)
            try:
                t, chunk = _kernel_times(fn, moe, x, ct, iterations=args.iterations)
            except Exception as err:  # noqa: BLE001 - recorded in the results
                row[name] = dict(error=f"{type(err).__name__}: {str(err)[:300]}")
                continue
            times[name] += t
            rounds[name].append(float(np.median(t)))
            if name == "capacity_1.25_einsum" and dropped is None:
                dropped = float(jax.jit(lambda m, x: m(x)[1].dropped_fraction)(moe, x))
            del moe
            row[name] = dict(median_ms=float(np.median(times[name])), min_ms=min(times[name]),
                             iterations=len(times[name]), round_median_ms=rounds[name], chunk=chunk)
    for name in names:
        if rounds[name] and "error" not in row[name]:
            row[name]["round_spread"] = max(rounds[name]) / min(rounds[name]) - 1.0
    if dropped is not None and "error" not in row["capacity_1.25_einsum"]:
        row["capacity_1.25_einsum"]["dropped_fraction"] = dropped
    if args.no_xla:
        row["dropless_xla"] = dict(skipped=SKIPPED_XLA)
    if args.torch_python:
        argv = ["layer", "--tokens", str(tokens), "--d-model", str(d), "--experts", str(experts), "--top-k",
                str(top_k), "--d-ff", str(f), "--shared", str(shared), "--iterations", str(args.iterations),
                "--compile"]
        row.update(_run_torch(args.torch_python, argv))
    ours = row["dropless_mosaic_gpu"].get("median_ms")
    for other in ("capacity_1.25_einsum", "dropless_mosaic_gpu_block_aligned", *(f"dropless_mosaic_gpu_{x}" for x in LAYER_EXTRA),
                  "dropless_xla", "torch_grouped_mm_layer_fwd_bwd", "torch_compile_grouped_mm_layer_fwd_bwd"):
        t = row.get(other, {}).get("median_ms")
        if ours and t:
            row[f"speedup_vs_{other}"] = t / ours
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    parser.add_argument("--no-layer", action="store_true")
    parser.add_argument("--layer-rounds", type=int, default=3, help="interleaved rounds of the JAX layer variants")
    parser.add_argument("--no-xla", action="store_true", help="skip jax.lax.ragged_dot entirely")
    parser.add_argument("--no-xla-bwd", action="store_true", help="skip XLA ragged_dot forward+backward only")
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch >= 2.8")
    parser.add_argument("--grouped-gemm", action="store_true", help="also time tgale96/grouped_gemm (torch env)")
    parser.add_argument("--sweep", action="store_true", help="only time the forward/d_lhs kernel per tiling in SWEEP")
    args = parser.parse_args()
    wanted = set(args.shapes.split(","))
    unknown = wanted - {s[0] for s in SHAPES}
    if unknown:
        raise SystemExit(f"unknown shapes {sorted(unknown)}")
    argv = [os.path.basename(a) if os.path.isabs(a) else a for a in sys.argv]  # no local paths in results
    result = dict(environment=environment(), command=" ".join(argv), shapes=[])
    if args.out and os.path.exists(args.out):  # keep the other shapes' entries
        with open(args.out) as fh:
            result["shapes"] = [s for s in json.load(fh).get("shapes", []) if s.get("name") not in wanted]
    for name, tokens, d, experts, top_k, f, shared in SHAPES:
        if name not in wanted:
            continue
        routes = tokens * top_k
        sizes = _group_sizes(tokens, experts, top_k)
        entry = dict(name=name, tokens=tokens, d_model=d, experts=experts, top_k=top_k, d_ff_expert=f,
                     shared_experts=shared, routes=routes, empty_groups=int(jnp.sum(sizes == 0)),
                     max_group=int(jnp.max(sizes)), min_group=int(jnp.min(sizes)),
                     measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu_before=gpu_snapshot())
        if args.sweep:  # tiling sweep only
            entry["sweep_up"] = _sweep_case(routes, d, 2 * f, experts, sizes, args)
            entry["sweep_down"] = _sweep_case(routes, f, d, experts, sizes, args)
            result["shapes"].append(entry)
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump(result, fh, indent=1)
            for g in ("sweep_up", "sweep_down"):
                for r in entry[g]:
                    c = r["config"]
                    print(name, g, r["direction"], f"{c['block_m']}x{c['block_n']}x{c['block_k']}s{c['steps']}"
                          f"g{c['grid_block_n']}", r.get("median_ms", r.get("error")), flush=True)
            continue
        entry["up"] = _gemm_case(routes, d, 2 * f, experts, sizes, args)
        entry["down"] = _gemm_case(routes, f, d, experts, sizes, args)
        if not args.no_layer and d <= 2048:
            entry["layer_fwd_bwd"] = _layer_case(tokens, d, experts, top_k, f, shared, args)
        entry["gpu_after"] = gpu_snapshot()
        result["shapes"].append(entry)
        if args.out:  # after every shape, so a timeout keeps the finished ones
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=1)
        summary = {g: {k: round(v, 3) for k, v in entry[g].items() if k.startswith("speedup")} for g in ("up", "down")}
        summary["layer_ms"] = {k: v.get("median_ms") for k, v in entry.get("layer_fwd_bwd", {}).items()
                               if isinstance(v, dict)}
        print(json.dumps(dict(name=name, **summary)), flush=True)


if __name__ == "__main__":
    main()
