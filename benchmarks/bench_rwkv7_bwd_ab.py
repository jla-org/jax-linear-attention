# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B of two versions of an RWKV-7 kernel module (``rwkv7_bwd.py`` or ``rwkv7_prep.py``) inside the op, on one GPU.

    python benchmarks/bench_rwkv7_bwd_ab.py --previous /path/to/previous/rwkv7_bwd.py \
        --shapes "16,4096,32,1;1,65536,32,16" --rounds 3 --iterations 20 --breakdown --json out.json
    python benchmarks/bench_rwkv7_bwd_ab.py --kernel prep --previous /path/to/previous/rwkv7_prep.py ...

For each shape ``B,T,H,documents`` (head dims 64, RWKV-7 gates, bf16 ``r, k, v``, f32 ``w, a, b``, the
layers' bounded mode), the forward + backward of :func:`jla.ops.mosaic_gpu.rwkv7.rwkv7` is compiled once
with the previous module (``--kernel``: ``bwd``, the default, or ``prep``) installed and once with this
tree's; the two compiled functions then alternate (the order flips every round) and each round records the
median CUPTI kernel time of ``--iterations`` calls. ``--breakdown`` adds the per-kernel medians (the swapped
kernel among them). ``nvidia-smi`` of the GPU is recorded before and after.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

from jla.ops.mosaic_gpu import rwkv7 as rwkv7_op  # noqa: E402
from jla.ops.mosaic_gpu import rwkv7_bwd, rwkv7_prep  # noqa: E402

CURRENT = {"bwd": rwkv7_bwd, "prep": rwkv7_prep}  # the swappable modules, by --kernel


def _gpu() -> str:
    index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader",
                              "-i", index], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _load(path: str):
    spec = importlib.util.spec_from_file_location("rwkv7_bwd_previous", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(batch: int, seq_len: int, heads: int, docs: int, seed: int = 0):
    dim = 64
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)
    shape = (batch, seq_len, heads, dim)
    q = jax.random.normal(ks[0], shape, jnp.bfloat16)
    k = jax.random.normal(ks[1], shape, jnp.bfloat16)
    v = jax.random.normal(ks[2], shape, jnp.bfloat16)
    w = -0.6065 * jax.nn.sigmoid(2.0 * jax.random.normal(ks[3], shape, jnp.float32))
    kk = jax.random.normal(ks[4], shape, jnp.float32)
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    b = kk * jax.nn.sigmoid(jax.random.normal(ks[5], shape, jnp.float32))
    do = jax.random.normal(ks[6], shape, jnp.bfloat16)
    seg = None
    if docs > 1:
        seg = jnp.broadcast_to((jnp.arange(seq_len) * docs // seq_len).astype(jnp.int32)[None], (batch, seq_len))
    return (q, k, v, w, -kk, b, do), seg


def _compiled(kind, module, args, seg):
    setattr(rwkv7_op, f"rwkv7_{kind}", module)

    def run(q, k, v, w, a, b, do):
        def primal(q, k, v, w, a, b):
            return rwkv7_op.rwkv7(q, k, v, w, a, b, lower_bound=-0.61, segment_ids=seg)[0]

        out, pullback = jax.vjp(primal, q, k, v, w, a, b)
        return out, pullback(do)

    fn = jax.jit(run)
    jax.block_until_ready(fn(*args))  # compiled now, while this module is installed
    return fn


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--kernel", choices=sorted(CURRENT), default="bwd", help="the module to swap")
    p.add_argument("--previous", required=True, help="path of the previous rwkv7_bwd.py (or rwkv7_prep.py)")
    p.add_argument("--shapes", default="16,4096,32,1;1,65536,32,16", help="B,T,H,documents; ...")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--breakdown", action="store_true")
    p.add_argument("--check", action="store_true", help="also report the max |difference| of the gradients")
    p.add_argument("--json", required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    previous = _load(args.previous)
    result = {"kernel": args.kernel, "previous": args.previous, "gpu_before": _gpu(),
              "device_kind": jax.devices()[0].device_kind, "rounds": args.rounds,
              "iterations": args.iterations, "shapes": []}
    for spec in args.shapes.split(";"):
        batch, seq_len, heads, docs = (int(x) for x in spec.split(","))
        xs, seg = _inputs(batch, seq_len, heads, docs)
        fns = {"previous": _compiled(args.kernel, previous, xs, seg),
               "new": _compiled(args.kernel, CURRENT[args.kernel], xs, seg)}
        setattr(rwkv7_op, f"rwkv7_{args.kernel}", CURRENT[args.kernel])
        rows = {name: [] for name in fns}
        for r in range(args.rounds):
            order = list(fns) if r % 2 == 0 else list(fns)[::-1]
            for name in order:
                rows[name].append(harness.median_kernel_ms(fns[name], *xs, iterations=args.iterations)["median_ms"])
        entry = {"shape": {"B": batch, "T": seq_len, "H": heads, "K": 64, "V": 64, "documents": docs},
                 "fwd_bwd_ms_per_round": rows,
                 "fwd_bwd_ms_median": {name: statistics.median(t) for name, t in rows.items()}}
        entry["speedup_new_over_previous"] = entry["fwd_bwd_ms_median"]["previous"] / entry["fwd_bwd_ms_median"]["new"]
        if args.breakdown:
            entry["breakdown_ms"] = {name: dict(list(harness.kernel_breakdown_ms(fn, *xs).items())[:8])
                                     for name, fn in fns.items()}
        if args.check:
            (_, g_prev), (_, g_new) = fns["previous"](*xs), fns["new"](*xs)
            entry["max_abs_grad_difference"] = {
                name: float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32))))
                for name, x, y in zip(("dq", "dk", "dv", "dw", "da", "db"), g_prev, g_new)}
            entry["max_abs_grad"] = {name: float(jnp.max(jnp.abs(x.astype(jnp.float32))))
                                     for name, x in zip(("dq", "dk", "dv", "dw", "da", "db"), g_prev)}
        print(json.dumps(entry), flush=True)
        result["shapes"].append(entry)
    result["gpu_after"] = _gpu()
    Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
