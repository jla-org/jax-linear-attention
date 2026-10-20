# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch mLSTM baseline for ``bench_mlstm.py``: NX-AI ``mlstm_kernels`` TFLA Triton kernels.

    python torch_mlstm.py --shape 1,8192,8,256,512 --mode fwd --kernel chunkwise--triton_xl_chunk --chunk 128

Prints one JSON line: median CUDA kernel time per call (``torch.profiler``, the sum of kernel
durations; the metric of ``harness.median_kernel_ms``). Inputs: bf16 ``q, k, v`` and gate
pre-activations ``i = 3 N(0, 1)``, ``f = 3 N(0, 1) + 2`` in its ``[B, H, T, D]`` layout.

Licence: ``mlstm_kernels`` is distributed by NXAI under the NXAI Community License Agreement, which
is not a permissive open-source licence (it has attribution and commercial-use terms; read its
LICENSE before installing it). It is not a dependency of ``jla``: this script imports it only as an
optional benchmark baseline, from a separate environment where the user installed it, through its
public API (``get_mlstm_kernel``). None of its code is part of this repository.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

torch.cuda.set_per_process_memory_fraction(0.25)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="B,T,H,K,V")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    parser.add_argument("--kernel", default="chunkwise--triton_xl_chunk")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    batch, seq_len, heads, key_dim, value_dim = (int(x) for x in args.shape.split(","))
    import mlstm_kernels
    from mlstm_kernels.torch import get_mlstm_kernel

    kernel = get_mlstm_kernel(args.kernel)
    dev, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, key_dim, device=dev, dtype=bf16, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, key_dim, device=dev, dtype=bf16, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, value_dim, device=dev, dtype=bf16, requires_grad=True)
    i = (3.0 * torch.randn(batch, heads, seq_len, device=dev)).to(bf16).requires_grad_()
    f = (3.0 * torch.randn(batch, heads, seq_len, device=dev) + 2.0).to(bf16).requires_grad_()
    do = torch.randn(batch, heads, seq_len, value_dim, device=dev, dtype=bf16)

    def call():
        return kernel(q=q, k=k, v=v, i=i, f=f, return_last_states=False, chunk_size=args.chunk)

    if args.mode == "fwd":
        def step():
            with torch.no_grad():
                call()
    else:
        def step():
            torch.autograd.grad(call(), [q, k, v, i, f], grad_outputs=do)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind="mlstm_kernels", kernel=args.kernel, chunk=args.chunk,
                          shape=(batch, seq_len, heads, key_dim, value_dim), mode=args.mode,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          torch=torch.__version__, mlstm_kernels=getattr(mlstm_kernels, "__version__", "?"))))


if __name__ == "__main__":
    main()
