# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch GLA baseline for ``bench_gla.py`` (run in a torch environment with FLA).

    python torch_gla.py --shape 1,4096,8,128,128 --mode fwd

Prints one JSON line: median CUDA kernel time per call of FLA's ``chunk_gla`` in this environment
(``torch.profiler``, the sum of kernel durations; the metric of ``harness.median_kernel_ms``), with the
FLA version.
Inputs follow ``bench_gla.inputs``: bf16 ``q, k, v`` (unit-norm ``k``), f32 ``g = logsigmoid(N(0, 1))``.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.10)


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
    parser.add_argument("--docs", type=int, default=0, help="documents per row (bench_gla.segment_ids), as cu_seqlens")
    args = parser.parse_args()
    batch, seq_len, heads, key_dim, value_dim = (int(x) for x in args.shape.split(","))
    import fla
    from fla.ops.gla import chunk_gla

    dev, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, key_dim, device=dev, dtype=bf16, requires_grad=True)
    k = F.normalize(torch.randn(batch, seq_len, heads, key_dim, device=dev), p=2, dim=-1).to(bf16)
    k.requires_grad_()
    v = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16, requires_grad=True)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, key_dim, device=dev)).requires_grad_()
    do = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16)

    kw = {}
    if args.docs > 1:  # the rows flattened into one sequence of B * docs documents
        length = seq_len // args.docs
        bounds = [b * seq_len + d * length for b in range(batch) for d in range(args.docs)] + [batch * seq_len]
        kw["cu_seqlens"] = torch.tensor(bounds, dtype=torch.long, device=dev)
        q, k, v, g = (x.detach().reshape(1, batch * seq_len, *x.shape[2:]).requires_grad_() for x in (q, k, v, g))
        do = do.reshape(1, batch * seq_len, *do.shape[2:])

    def call():
        return chunk_gla(q=q, k=k, v=v, g=g, **kw)[0]

    if args.mode == "fwd":
        def step():
            with torch.inference_mode():
                call()
    else:
        def step():
            torch.autograd.grad(call(), [q, k, v, g], grad_outputs=do)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind="fla_gla", shape=(batch, seq_len, heads, key_dim, value_dim), mode=args.mode,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          torch=torch.__version__, fla=getattr(fla, "__version__", "?"))))


if __name__ == "__main__":
    main()
