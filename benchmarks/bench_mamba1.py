# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time the Mamba-1 selective scan: our Mosaic GPU kernel, the chunked XLA op and the official CUDA kernel.

    python benchmarks/bench_mamba1.py --batch 8 --seq-len 2048 --dim 1536 --d-state 16 \\
        --impls mosaic_gpu,mosaic_gpu:reference,chunked --torch-python /path/to/venv-mamba/bin/python

Shape defaults are one layer of state-spaces/mamba-130m (``d_inner = 1536``, ``d_state = 16``).
Metric: median CUDA kernel time per call (``harness.median_kernel_ms``; ``torch.profiler`` for
the official kernel, ``benchmarks/torch_mamba1.py``); ``fwdbwd`` is one forward plus the
backward for a fixed output cotangent. ``mosaic_gpu`` runs the op's default ``backward="auto"`` (the
Mosaic GPU backward kernel where it applies); ``mosaic_gpu:<backward>`` passes ``backward`` explicitly
(``kernel``, ``kernel_v1`` (the first backward kernel) or ``reference``, the reference op's VJP).
``mosaic_gpu:kernel@<S>`` forces the backward kernel's sequence split to ``S`` segments (default: by
shape), and ``mosaic_gpu:kernel@<S>:<H>`` (or ``@:<H>``) also its register sub-tile to ``H`` tokens.
The implementations run interleaved in one process on one GPU (the official kernel in a subprocess
with the same ``CUDA_VISIBLE_DEVICES``), with ``nvidia-smi`` utilisation recorded before and after.
Results go to ``--out`` (JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla import ops  # noqa: E402


def _inputs(batch: int, seq_len: int, dim: int, dstate: int):
    ks = jax.random.split(jax.random.key(0), 7)
    bf16 = jnp.bfloat16
    return dict(
        q=jax.random.normal(ks[0], (batch, seq_len, dstate), bf16),
        k=jax.random.normal(ks[1], (batch, seq_len, dstate), bf16),
        v=jax.random.normal(ks[2], (batch, seq_len, dim), bf16),
        dt=(0.5 * jax.random.normal(ks[3], (batch, seq_len, dim))).astype(bf16),
        a=-jnp.broadcast_to(jnp.arange(1, dstate + 1, dtype=jnp.float32), (dim, dstate)),
        d=jnp.ones((dim,), jnp.float32),
        z=jax.random.normal(ks[4], (batch, seq_len, dim), bf16),
        dt_bias=jnp.full((dim,), -3.0, jnp.float32),
    )


def run_official(torch_python: str, shape: tuple[int, ...], mode: str, iterations: int) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_mamba1.py")
    cmd = [torch_python, script, "--shape", ",".join(map(str, shape)), "--mode", mode,
           "--iterations", str(iterations)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def _force(segments: int | None, history: int | None) -> None:
    """Force (or with None restore) the backward kernel's sequence segments and register sub-tile length."""
    from jla.ops.mosaic_gpu import mamba1_bwd2 as m

    if not hasattr(m, "_defaults"):
        m._defaults = (m.num_segments, m.history_tokens)
    m.num_segments = m._defaults[0] if segments is None else (lambda cfg, num_sms=0: segments)
    m.history_tokens = m._defaults[1] if history is None else (lambda n_state: history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--impls", default="mosaic_gpu,chunked")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--torch-python", default=None, help="python of an environment with selective_scan_cuda")
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    shape = (args.batch, args.seq_len, args.dim, args.d_state)
    x = _inputs(*shape)
    names = ("q", "k", "v", "dt", "a", "d", "z", "dt_bias")
    values = [x[name] for name in names]
    ct = jnp.ones_like(x["v"])
    results = []
    gpu_before = harness.gpu_snapshot()
    for mode in args.modes.split(","):
        for impl in args.impls.split(","):
            # "mosaic_gpu:kernel" is the Mosaic GPU forward with the Mosaic GPU backward kernel.
            name, _, backward = impl.partition(":")
            backward, _, forced = backward.partition("@")
            segs, _, hist = forced.partition(":")
            extra = dict(backward=backward) if backward else {}
            _force(int(segs) if segs else None, int(hist) if hist else None)

            def fn(q, k, v, dt, a, d, z, dt_bias, name=name, extra=extra):
                return ops.linear_attention("mamba1", q, k, v, dt=dt, a=a, d=d, z=z, dt_bias=dt_bias,
                                            dt_softplus=True, impl=name, chunk_size=args.chunk_size, **extra)

            if mode == "fwd":
                timed = harness.median_kernel_ms(jax.jit(lambda *a, fn=fn: fn(*a)[0]), *values,
                                                 iterations=args.iterations)
            else:
                timed = harness.median_kernel_ms(harness.vjp_fn(fn, (0, 1, 2, 3, 4, 5, 6, 7)), *values, ct,
                                                 iterations=args.iterations)
            results.append(dict(impl=impl, mode=mode, **timed))
            print(impl, mode, f"{timed['median_ms']:.3f} ms", flush=True)
        if args.torch_python:
            row = run_official(args.torch_python, shape, mode, args.iterations)
            results.append(dict(row, impl="official", mode=mode))
            print("official", mode, row.get("median_ms", row.get("error")), flush=True)
    gpu_after = harness.gpu_snapshot()
    if args.out:
        harness.write_json(args.out, dict(
            op="mamba1", shape=dict(batch=args.batch, seq_len=args.seq_len, dim=args.dim, d_state=args.d_state,
                                    chunk_size=args.chunk_size, dtype="bfloat16"),
            environment=harness.environment(), gpu_before=gpu_before, gpu_after=gpu_after, results=results,
        ))


if __name__ == "__main__":
    main()
