# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile the MoE layer of the shipped MoE presets: forward + backward per grouped-GEMM path.

    python benchmarks/profile_moe.py --tokens 16384 --out benchmarks/results/moe_profile.json

For every preset in ``--presets`` the script builds one MoE layer exactly as the preset configures
it (``d_model``, experts, top-k, expert width, shared experts and their gate, router) and times
its forward + backward on ``--tokens`` random bf16 tokens (median CUPTI kernel ms,
``benchmarks/harness.py``) for each path:

* ``preset``: the preset as shipped (its ``capacity_factor`` and ``ragged_dot_impl``);
* ``dropless_xla``: dropless routing on ``jax.lax.ragged_dot``;
* ``dropless_mosaic_gpu``: dropless routing on the Mosaic GPU grouped GEMM (default tiling);
* ``dropless_mosaic_gpu_block_aligned`` / ``_aligned`` / ``_persistent`` / ``_persistent_ws``: the same with
  the ``BLOCK_ALIGNED`` (the default until 18 Oct 2026) / ``ALIGNED`` / ``PERSISTENT`` / ``PERSISTENT_WS``
  tiling where it divides a GEMM;
* ``capacity_1.25``: capacity routing (batched einsum over ``[E, C, D]``).

Each path also gets a per-kernel breakdown (median ms per kernel name over 10 calls), summed into
categories: grouped GEMM kernels, other matmuls, sort, gathers and scatters, and the rest.
Paths run in ``--rounds`` interleaved rounds with the order rotated every round.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _measure_each, environment, gpu_snapshot, kernel_breakdown_ms, vjp_fn  # noqa: E402

from jla.layers.moe import MoE  # noqa: E402
from jla.models import get_preset  # noqa: E402
from jla.ops.mosaic_gpu import ragged_dot as rd  # noqa: E402

PRESETS = ("qwen3-next-300m", "kimi-linear-1b", "granite-4.0-h-125m", "gpt-oss-180m", "minimax-01-1b", "moe-125m")


def _category(name: str) -> str:
    low = name.lower()
    if "ragged" in low or "mosaic" in low or low.startswith("_ragged_dot") or "transposed" in low:
        return "grouped_gemm"
    if "gemm" in low or "xmma" in low or "cutlass" in low or "cublas" in low or "dot" in low or "matmul" in low:
        return "matmul"
    if "sort" in low or "radix" in low or "cub::" in low:
        return "sort"
    if "scatter" in low:
        return "scatter"
    if "gather" in low:
        return "gather"
    return "other"


def _moe_with(config):
    class _ConfigMoE(MoE):
        def _ragged_dot(self, lhs, rhs, group_sizes):
            k, n = rhs.shape[1], rhs.shape[2]
            cfg = config if config.divides_both(k, n) and rd.supports(lhs, rhs, group_sizes) else None
            return rd.grouped_matmul(lhs, rhs, group_sizes, impl="mosaic_gpu", full_groups=True, config=cfg)

    return _ConfigMoE


def _variants(moe_cfg, only):
    dropless = dataclasses.replace(moe_cfg, capacity_factor=None)
    out = {
        "preset": (MoE, moe_cfg),
        "dropless_xla": (MoE, dataclasses.replace(dropless, ragged_dot_impl="xla")),
        "dropless_mosaic_gpu": (MoE, dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu")),
        "dropless_mosaic_gpu_block_aligned": (_moe_with(rd.BLOCK_ALIGNED), dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu")),
        "dropless_mosaic_gpu_aligned": (_moe_with(rd.ALIGNED), dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu")),
        "dropless_mosaic_gpu_persistent": (_moe_with(rd.PERSISTENT),
                                           dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu")),
        "dropless_mosaic_gpu_persistent_ws": (_moe_with(rd.PERSISTENT_WS),
                                              dataclasses.replace(dropless, ragged_dot_impl="mosaic_gpu")),
        "capacity_1.25": (MoE, dataclasses.replace(moe_cfg, capacity_factor=1.25, ragged_dot_impl="xla")),
    }
    return {k: v for k, v in out.items() if not only or k in only}


def profile_preset(name: str, tokens: int, iterations: int, rounds: int, only: tuple[str, ...]) -> dict:
    cfg = get_preset(name)
    moe_cfg = dataclasses.replace(cfg.moe, expert_parallel_axis=None, expert_parallel_buffer_factor=None)
    d = cfg.d_model
    variants = _variants(moe_cfg, only)
    x = jax.random.normal(jax.random.key(3), (tokens, d), jnp.float32).astype(jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(4), (tokens, d), jnp.float32).astype(jnp.bfloat16)
    fn = vjp_fn(lambda m, x: (m(x)[0],), (0, 1))

    def build(v):
        cls, c = variants[v]
        return cls(d, c, key=jax.random.key(0), std=0.02, param_dtype=jnp.float32, compute_dtype="bfloat16")

    entry = dict(preset=name, d_model=d, tokens=tokens, moe=dataclasses.asdict(moe_cfg), paths={},
                 measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu_before=gpu_snapshot())
    names = list(variants)
    times = {v: [] for v in names}
    for r in range(max(1, rounds)):
        shift = r % len(names)
        for v in names[shift:] + names[:shift]:
            if "error" in entry["paths"].get(v, {}):
                continue
            layer = build(v)
            try:
                for _ in range(3):
                    jax.block_until_ready(fn(layer, x, ct))
                t = _measure_each(fn, (layer, x, ct), iterations, aggregate=True)
                times[v] += [float(s) for s in t]
                row = entry["paths"].setdefault(v, {})
                row.update(median_ms=float(np.median(times[v])), min_ms=float(np.min(times[v])), calls=len(times[v]))
                if r == 0:
                    per = kernel_breakdown_ms(fn, layer, x, ct, iterations=10)
                    cats: dict[str, float] = {}
                    for kname, ms in per.items():
                        cats[_category(kname)] = cats.get(_category(kname), 0.0) + ms
                    row["categories_ms"] = {k: round(v2, 4) for k, v2 in sorted(cats.items(), key=lambda kv: -kv[1])}
                    row["top_kernels_ms"] = {k: round(v2, 4) for k, v2 in list(per.items())[:20]}
                    stats = jax.jit(lambda m, x: m(x)[1])(layer, x)
                    row["dropped_fraction"] = float(stats.dropped_fraction)
            except Exception as err:  # noqa: BLE001 - recorded in the results
                entry["paths"][v] = dict(error=f"{type(err).__name__}: {str(err)[:400]}")
            del layer
    entry["gpu_after"] = gpu_snapshot()
    base = entry["paths"].get("dropless_mosaic_gpu", {}).get("median_ms")
    if base:
        entry["time_vs_dropless_mosaic_gpu"] = {v: round(r["median_ms"] / base, 3) for v, r in entry["paths"].items()
                                                if "median_ms" in r}
    return entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--presets", default=",".join(PRESETS))
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--only", default="", help="comma list of paths")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    only = tuple(s for s in args.only.split(",") if s)
    argv = [os.path.basename(a) if os.path.isabs(a) else a for a in sys.argv]
    result = dict(environment=environment(), command=" ".join(argv), presets=[])
    if args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            keep = set(args.presets.split(","))
            result["presets"] = [e for e in json.load(fh).get("presets", []) if e.get("preset") not in keep]
    for name in args.presets.split(","):
        entry = profile_preset(name, args.tokens, args.iterations, args.rounds, only)
        result["presets"].append(entry)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=1)
        print(json.dumps(dict(preset=name, ms={v: r.get("median_ms", r.get("error")) for v, r in entry["paths"].items()},
                              cats={v: r.get("categories_ms") for v, r in entry["paths"].items()})), flush=True)


if __name__ == "__main__":
    main()
