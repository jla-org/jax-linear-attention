# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The official causal-conv1d CUDA kernel (Dao-AILab/causal-conv1d) as a short-conv baseline.

    python torch_short_conv.py --shape 16,4096,4096 --width 4 --mode fwdbwd [--bias] [--doc-len 1024]

``causal_conv1d_fn`` on the channel-last layout Mamba-2 uses (``x [B, T, C]`` contiguous, passed as
its ``[B, C, T]`` transpose view), SiLU, ``seq_idx`` for packed documents. Prints one JSON line:
median CUDA kernel time per call (``torch.profiler``, sum of kernel durations, the metric of
``harness.median_kernel_ms``).
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

torch.cuda.set_per_process_memory_fraction(0.20)


def kernel_ms(step, iterations: int) -> list[float]:
    from torch.profiler import ProfilerActivity, profile

    out = []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        out.append(sum(e.self_device_time_total for e in prof.key_averages()) / 1e3)
    return out


def main() -> None:
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn

    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, help="B,T,C")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--doc-len", type=int, default=0, help="packed documents of this many tokens (0: one per row)")
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    batch, seq_len, channels = (int(v) for v in args.shape.split(","))
    dev, bf16 = "cuda", torch.bfloat16
    x = torch.randn(batch, seq_len, channels, device=dev, dtype=bf16)
    w = torch.randn(channels, args.width, device=dev, dtype=torch.float32) * 0.5
    b = torch.randn(channels, device=dev, dtype=torch.float32) * 0.5 if args.bias else None
    seq_idx = None
    if args.doc_len:
        seq_idx = (torch.arange(seq_len, device=dev, dtype=torch.int32) // args.doc_len).expand(batch, -1).contiguous()
    dy = torch.randn(batch, seq_len, channels, device=dev, dtype=bf16).transpose(1, 2)
    w_bf, b_bf = w.to(bf16), None if b is None else b.to(bf16)  # the kernel wants weight in x's dtype

    def call(xv, wv, bv):
        return causal_conv1d_fn(xv.transpose(1, 2), wv, bv, seq_idx=seq_idx, activation="silu")

    if args.mode == "fwd":
        def step():
            with torch.no_grad():
                call(x, w_bf, b_bf)
    else:
        xg = x.clone().requires_grad_(True)
        wg = w_bf.clone().requires_grad_(True)
        bg = None if b_bf is None else b_bf.clone().requires_grad_(True)
        leaves = [t for t in (xg, wg, bg) if t is not None]

        def step():
            torch.autograd.grad(call(xg, wg, bg), leaves, grad_outputs=dy)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind="causal_conv1d", shape=[batch, seq_len, channels], width=args.width,
                          mode=args.mode, bias=args.bias, doc_len=args.doc_len,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          causal_conv1d=causal_conv1d.__version__, torch=torch.__version__)))


if __name__ == "__main__":
    main()
