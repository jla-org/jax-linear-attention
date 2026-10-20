# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Context-parallel softmax attention on the local GPUs: accuracy and fwd+bwd time.

    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/bench_context_parallel_attention.py --seq 8192 32768

Compares ring attention (zigzag and contiguous layouts; with a sliding window the
ring is always contiguous and truncated) and Ulysses (cuDNN or the Mosaic GPU flash
kernel inside) over a ``("cp",)`` mesh of all visible GPUs with the single-GPU
baselines on the whole sequence: cuDNN flash attention (packed documents become a
dense ``[B, 1, T, T]`` mask) and the Mosaic GPU flash kernel (per-tile document
masks). Accuracy: outputs and q/k/v gradients against single-device float32 XLA
attention at ``"highest"`` matmul precision on the same inputs (rounded to the tested
dtype, then upcast), as the max absolute error over the max magnitude of the reference,
next to the floor (the reference itself rounded to the dtype). Time: median of ``--iters``
jitted forward+backward calls (wall clock, so collectives and waiting between GPUs count).
``--seq`` without values runs only the accuracy check.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jla.distributed.context_parallel_attention import context_parallel_attention
from jla.layers.attention import attention

VARIANTS = ("ring_zigzag", "ring_contiguous", "ulysses_cudnn", "ulysses_mosaic")
BASELINES = ("single_gpu_cudnn", "single_gpu_mosaic")


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(1e-6, float(np.max(np.abs(b)))))


def timed(fn, args, iters: int) -> float:
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, nargs="*", default=[8192, 32768], help="none: only the accuracy check")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check-seq", type=int, default=2048, help="0 skips the accuracy check")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS + BASELINES),
                        choices=list(VARIANTS + BASELINES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("cp",))
    report = {"devices": len(devices), "device_kind": devices[0].device_kind, "jax": jax.__version__,
              "shape": vars(args), "accuracy": {}, "ms_fwd_bwd": {}}

    def make(seq, dtype, seed=0):
        ks = jax.random.split(jax.random.key(seed), 4)
        q = jax.random.normal(ks[0], (args.batch, seq, args.heads, args.head_dim), dtype)
        k = jax.random.normal(ks[1], (args.batch, seq, args.kv_heads, args.head_dim), dtype)
        v = jax.random.normal(ks[2], (args.batch, seq, args.kv_heads, args.head_dim), dtype)
        w = jax.random.normal(ks[3], q.shape, dtype)
        resets = np.zeros((args.batch, seq), np.int32)
        resets[:, [seq // 3, seq // 2, (3 * seq) // 4 + 1]] = 1  # one start exactly on the 2-way shard boundary
        seg = jnp.asarray(np.cumsum(resets, axis=1))
        return q, k, v, w, seg

    def grad_fn(fn):
        def loss(q, k, v, w, seg):
            return jnp.sum(fn(q, k, v, seg).astype(jnp.float32) * w.astype(jnp.float32))
        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

    def out_fn(fn):
        return jax.jit(lambda q, k, v, seg: fn(q, k, v, seg))

    def cp(strategy, **fixed):
        return lambda q, k, v, seg, **kw: context_parallel_attention(
            q, k, v, mesh=mesh, strategy=strategy, segment_ids=seg, **fixed, **kw)

    variants = {
        "ring_zigzag": cp("ring", load_balance=True),
        "ring_contiguous": cp("ring", load_balance=False),
        "ulysses_cudnn": cp("ulysses", implementation="cudnn"),
        "ulysses_mosaic": cp("ulysses", implementation="mosaic_gpu"),
    }
    variants = {name: fn for name, fn in variants.items() if name in args.variants}

    # Accuracy on the GPUs against single-device float32 XLA attention (bf16 for the fused kernels).
    if args.check_seq:
        q, k, v, w, seg = make(args.check_seq, jnp.float32, seed=1)
        for window in (None, args.window):
            ref = lambda q, k, v, seg: attention(q, k, v, segment_ids=seg, sliding_window=window,  # noqa: E731
                                                 implementation="xla")
            for dtype in (jnp.float32, jnp.bfloat16):
                cast = [x.astype(dtype) for x in (q, k, v, w)]
                # The reference sees the same rounded inputs (upcast to float32), so the errors
                # measure each implementation, not the rounding of q, k, v, w to ``dtype``.
                exact = [x.astype(jnp.float32) for x in cast]
                with jax.default_matmul_precision("highest"):  # true float32 reference (no TF32)
                    o_ref, g_ref = out_fn(ref)(*exact[:3], seg), grad_fn(ref)(*exact, seg)
                # The floor: the reference itself rounded to ``dtype`` (every result is stored in it).
                floor = {"o": rel(o_ref.astype(dtype), o_ref),
                         **{n: rel(g.astype(dtype), g) for n, g in zip(("dq", "dk", "dv"), g_ref)}}
                report["accuracy"][f"reference_rounded/window={window}/{jnp.dtype(dtype).name}"] = floor
                print("reference_rounded", window, jnp.dtype(dtype).name, floor, flush=True)
                for name, fn in variants.items():
                    if dtype == jnp.float32 and name in ("ulysses_cudnn", "ulysses_mosaic"):
                        continue  # bf16/fp16 kernels
                    if window is not None and name == "ring_contiguous":
                        continue  # a windowed ring is always contiguous: same as ring_zigzag
                    bound = lambda q, k, v, seg, fn=fn: fn(q, k, v, seg, sliding_window=window)  # noqa: E731
                    with jax.default_matmul_precision("highest"):
                        o = out_fn(bound)(cast[0], cast[1], cast[2], seg)
                        grads = grad_fn(bound)(*cast, seg)
                    errs = {"o": rel(o, o_ref), **{n: rel(a, b) for n, a, b in zip(("dq", "dk", "dv"), grads, g_ref)}}
                    report["accuracy"][f"{name}/window={window}/{jnp.dtype(dtype).name}"] = errs
                    print(name, window, jnp.dtype(dtype).name, errs, flush=True)

    # Time: bf16 forward+backward, inputs sequence-sharded over the mesh.
    for seq_len, window in [(s, w) for s in args.seq for w in (None, args.window)]:
        q, k, v, w, seg = make(seq_len, jnp.bfloat16)
        sharded = NamedSharding(mesh, P(None, "cp"))
        sq, sk, sv, sw, sseg = (jax.device_put(x, sharded) for x in (q, k, v, w, seg))
        lq, lk, lv, lw, lseg = (jax.device_put(x, devices[0]) for x in (q, k, v, w, seg))
        runs = {}
        for name, impl in (("single_gpu_cudnn", "cudnn"), ("single_gpu_mosaic", "mosaic_gpu")):
            if name in args.variants:
                runs[name] = (lambda q, k, v, seg, impl=impl: attention(  # noqa: E731
                    q, k, v, segment_ids=seg, sliding_window=window, implementation=impl), (lq, lk, lv, lw, lseg))
        for name, fn in variants.items():
            if window is not None and name == "ring_contiguous":
                continue
            bound = lambda q, k, v, seg, fn=fn: fn(q, k, v, seg, sliding_window=window)  # noqa: E731
            runs[name] = (bound, (sq, sk, sv, sw, sseg))
        times = {}
        for name, (fn, operands) in runs.items():
            try:
                times[name] = timed(grad_fn(fn), operands, args.iters)
            except Exception as err:  # noqa: BLE001 - report out-of-memory and keep timing the others
                times[name] = f"failed: {type(err).__name__}: {str(err)[:200]}"
            print("T", seq_len, "window", window, name, times[name], flush=True)
        report["ms_fwd_bwd"][f"T={seq_len}/window={window}"] = times
        del sq, sk, sv, sw, sseg, lq, lk, lv, lw, lseg
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
