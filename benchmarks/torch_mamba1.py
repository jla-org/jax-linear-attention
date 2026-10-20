# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Time the official Mamba-1 selective-scan CUDA kernels (state-spaces/mamba ``selective_scan_cuda``).

    python benchmarks/torch_mamba1.py --shape 8,2048,1536,16 --mode fwd

Runs in an environment where state-spaces/mamba is built with its CUDA extension
(``MAMBA_KEEP_CUDA_BUILD=TRUE``). It calls ``selective_scan_cuda.fwd`` / ``.bwd`` exactly as
``mamba_ssm.ops.selective_scan_interface.SelectiveScanFn`` does, in the official layout
(``u, delta, z [B, D, T]`` bf16; ``B, C [B, 1, N, T]`` bf16; ``A [D, N]``, ``D``, ``delta_bias``
float32; ``delta_softplus=True``), so no autograd or layout glue is timed. ``fwdbwd`` is one
forward plus the backward for a fixed output cotangent. Prints one JSON line: the median CUDA
kernel time per call (``torch.profiler``) over ``--iterations`` calls after 3 warmups.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


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
    parser.add_argument("--shape", required=True, help="B,T,D,N")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    import selective_scan_cuda

    torch.cuda.set_per_process_memory_fraction(0.25)
    batch, seq_len, dim, dstate = (int(x) for x in args.shape.split(","))
    gen = torch.Generator(device="cuda").manual_seed(0)
    dev, bf16, f32 = "cuda", torch.bfloat16, torch.float32

    def randn(*shape, dtype=bf16, scale=1.0):
        return (scale * torch.randn(*shape, device=dev, generator=gen)).to(dtype)

    u = randn(batch, dim, seq_len)
    delta = randn(batch, dim, seq_len, scale=0.5)
    z = randn(batch, dim, seq_len)
    b = randn(batch, 1, dstate, seq_len)
    c = randn(batch, 1, dstate, seq_len)
    a = -torch.arange(1, dstate + 1, device=dev, dtype=f32).expand(dim, dstate).contiguous()
    d = torch.ones(dim, device=dev, dtype=f32)
    delta_bias = torch.full((dim,), -3.0, device=dev, dtype=f32)
    dout = randn(batch, dim, seq_len)

    def fwd():
        return selective_scan_cuda.fwd(u, delta, a, b, c, d, z, delta_bias, True)

    if args.mode == "fwd":
        step = fwd
    else:
        def step():
            out, x, *rest = fwd()
            selective_scan_cuda.bwd(u, delta, a, b, c, d, z, delta_bias, dout, x, out, None, True, False)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(impl="official", shape=[batch, seq_len, dim, dstate], mode=args.mode,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          torch=torch.__version__)))


if __name__ == "__main__":
    main()
