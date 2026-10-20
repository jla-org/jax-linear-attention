# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch RWKV-7 baseline for ``bench_rwkv7.py`` (run in a torch environment with FLA).

    python torch_rwkv7.py --shape 8,4096,12,64,64 --mode fwd

Prints one JSON line: median CUDA kernel time per call of FLA ``chunk_rwkv7`` (Triton, or the backend
that FLA's default dispatch picks when the environment has one; ``torch.profiler``, the sum of
kernel durations; the metric of ``harness.median_kernel_ms``).
Inputs follow ``bench_rwkv7.inputs`` (all bf16, as an FLA model passes them).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.25)


class _BackendLog(logging.Handler):
    """Collects FLA's "[FLA Backend] op -> backend" dispatch messages."""

    def __init__(self):
        super().__init__()
        self.lines: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "[FLA Backend]" in msg:
            self.lines.add(msg)


_LOG = _BackendLog()
logging.getLogger("fla").addHandler(_LOG)
logging.getLogger("fla").setLevel(logging.INFO)


def kernel_ms(step, iterations: int) -> list[float]:
    from torch.profiler import ProfilerActivity, profile

    out = []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        # Each kernel is "self" time of exactly one profiled op, so this sums every kernel once.
        out.append(sum(e.self_device_time_total for e in prof.key_averages()) / 1e3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="B,T,H,K,V")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--safe-gate", action="store_true", help="FLA's bounded-gate mode (safe_gate=True)")
    parser.add_argument("--chunk-size", type=int, default=None, help="FLA chunk_size (default: FLA's own)")
    parser.add_argument("--lower-bound", type=float, default=None,
                        help="FLA lower_bound (the RWKV7Attention layer: -0.6065)")
    parser.add_argument("--cu-seqlens", default=None, help="packed documents: comma-separated cu_seqlens (B must be 1)")
    args = parser.parse_args()
    batch, seq_len, heads, key_dim, value_dim = (int(x) for x in args.shape.split(","))
    import fla
    from fla.ops.rwkv7 import chunk_rwkv7

    dev, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    dims = (batch, seq_len, heads, key_dim)
    r = torch.randn(*dims, device=dev, dtype=bf16, requires_grad=True)
    k = torch.randn(*dims, device=dev, dtype=bf16, requires_grad=True)
    v = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16, requires_grad=True)
    w = (-0.6065306597126334 * torch.sigmoid(torch.randn(*dims, device=dev))).to(bf16).requires_grad_()
    kk = F.normalize(torch.randn(*dims, device=dev), p=2, dim=-1)
    a = (-kk).to(bf16).requires_grad_()
    b = (kk * torch.sigmoid(torch.randn(*dims, device=dev))).to(bf16).requires_grad_()
    do = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16)

    extra = {} if args.lower_bound is None else {"lower_bound": args.lower_bound}
    cu = None
    if args.cu_seqlens:
        cu = torch.tensor([int(x) for x in args.cu_seqlens.split(",")], device=dev, dtype=torch.long)
        extra["cu_seqlens"] = cu

    def call():
        return chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, safe_gate=args.safe_gate,
                           chunk_size=args.chunk_size, **extra)[0]

    if args.mode == "fwd":
        def step():
            with torch.inference_mode():
                call()
    else:
        def step():
            torch.autograd.grad(call(), [r, w, k, v, a, b], grad_outputs=do)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind="fla_rwkv7", shape=(batch, seq_len, heads, key_dim, value_dim), mode=args.mode,
                          safe_gate=args.safe_gate, chunk_size=args.chunk_size,
                          lower_bound=args.lower_bound, packed=cu is not None,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          backends=sorted(_LOG.lines), torch=torch.__version__,
                          fla=getattr(fla, "__version__", "?"))))


if __name__ == "__main__":
    main()
