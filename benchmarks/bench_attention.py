# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Causal softmax attention: the Mosaic GPU kernel against cuDNN through ``jla.layers.attention``.

Shapes: ``T`` in {2048, 4096, 8192}, 16 query and 16 key/value heads, ``D = 128``, bf16,
``B = 16384 / T`` (16K tokens per batch). Masks: plain causal, a 1024-token sliding window
(cuDNN's native left window) and four packed documents per row (cuDNN gets the dense boolean
mask the layer builds). ``fwd`` is the forward, ``fwdbwd`` one forward plus the backward
for a fixed output cotangent; median CUPTI kernel time (``harness.py``). Implementations, run
interleaved per shape: ``mosaic_gpu`` (the kernel with its default block-skipping backward),
``mosaic_gpu_full`` (the same forward with the previous backward that visits every block) and
``cudnn``.

    python benchmarks/bench_attention.py --out benchmarks/results/attention.json
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.layers.attention import attention  # noqa: E402
from jla.ops.mosaic_gpu.attention import flash_attention  # noqa: E402

TOKENS = 16384
IMPLS = ("mosaic_gpu", "mosaic_gpu_full", "cudnn")
HEADS = 16
HEAD_DIM = 128
WINDOW = 1024
DOCS = 4


def _inputs(length: int):
    batch = TOKENS // length
    keys = jax.random.split(jax.random.key(length), 4)
    shape = (batch, length, HEADS, HEAD_DIM)
    q, k, v, do = (jax.random.normal(key, shape, jnp.float32).astype(jnp.bfloat16) for key in keys)
    starts = jnp.arange(length) // (length // DOCS)
    seg = jnp.broadcast_to(starts.astype(jnp.int32), (batch, length))
    return q, k, v, do, seg


def _causal_flops(batch: int, length: int, mode: str) -> float:
    """Useful matmul FLOPs (visible query-key pairs only) of one forward."""
    if mode == "causal":
        pairs = length * (length + 1) / 2
    elif mode == "swa":
        pairs = sum(min(t + 1, WINDOW) for t in range(length))
    else:
        doc = length // DOCS
        pairs = DOCS * doc * (doc + 1) / 2
    return 4.0 * batch * HEADS * pairs * HEAD_DIM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results/attention.json")
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--modes", default="causal,swa,documents")
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    args = parser.parse_args()
    rows = []
    gpu_before = harness.gpu_snapshot()
    for length in (int(x) for x in args.lengths.split(",")):
        q, k, v, do, seg = _inputs(length)
        batch = q.shape[0]
        for mode in args.modes.split(","):
            kwargs = dict(
                segment_ids=seg if mode == "documents" else None,
                sliding_window=WINDOW if mode == "swa" else None,
            )
            for impl in args.impls.split(","):
                def fn(q, k, v, impl=impl, kwargs=kwargs):
                    if impl == "mosaic_gpu_full":
                        return (flash_attention(q, k, v, backward="full", **kwargs),)
                    return (attention(q, k, v, implementation=impl, **kwargs),)

                row = dict(T=length, B=batch, H=HEADS, D=HEAD_DIM, mode=mode, impl=impl)
                flops = _causal_flops(batch, length, mode)
                try:
                    fwd = harness.median_kernel_ms(jax.jit(fn), q, k, v, iterations=args.iterations)
                    row["fwd_ms"] = fwd["median_ms"]
                    row["fwd_tflops"] = flops / (fwd["median_ms"] * 1e9)
                    both = harness.median_kernel_ms(harness.vjp_fn(fn, (0, 1, 2)), q, k, v, do,
                                                    iterations=args.iterations)
                    row["fwdbwd_ms"] = both["median_ms"]
                    row["fwdbwd_tflops"] = 3.5 * flops / (both["median_ms"] * 1e9)
                except Exception as err:  # record the failure, keep measuring the rest
                    row["error"] = harness.error_summary(f"{type(err).__name__}: {err}")
                print(row, flush=True)
                rows.append(row)
    harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=gpu_before,
                                      gpu=harness.gpu_snapshot(),
                                      provisional="GPU may be shared with other processes", rows=rows))


if __name__ == "__main__":
    main()
