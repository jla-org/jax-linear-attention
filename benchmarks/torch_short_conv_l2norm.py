# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The q/k prologue as FLA runs it: short conv + SiLU, then FLA's Triton ``l2norm`` per head (baseline).

    python torch_short_conv_l2norm.py --shape 16,4096,2048 --head-dim 128 --conv fla --mode fwdbwd [--doc-len 1024]

``--conv fla``: FLA's ``causal_conv1d`` (Triton backend, its default), ``[B, T, C]``, documents as ``cu_seqlens``
over the flattened batch. ``--conv cuda``: the official causal-conv1d CUDA kernel (``causal_conv1d_fn`` on the
channel-last ``[B, C, T]`` transpose view, documents as ``seq_idx``). Both then apply
``fla.modules.norm.l2norm`` over each ``head_dim``-channel head (the norm FLA's GDN / KDA kernels run on q and
k). Width 4, SiLU, no bias, bf16. Prints one JSON line: median CUDA kernel time per call (``torch.profiler``,
sum of kernel durations, the metric of ``harness.median_kernel_ms``), fwd+bwd = output and the gradients of x
and the weight.
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
    import fla
    from fla.modules.norm.l2norm import l2norm

    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, help="B,T,C")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--conv", choices=("fla", "cuda"), required=True)
    p.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    p.add_argument("--doc-len", type=int, default=0, help="packed documents of this many tokens (0: one per row)")
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    batch, seq_len, channels = (int(v) for v in args.shape.split(","))
    dev, bf16 = "cuda", torch.bfloat16
    x = torch.randn(batch, seq_len, channels, device=dev, dtype=bf16)
    w = (torch.randn(channels, 4, device=dev, dtype=torch.float32) * 0.5).to(bf16)
    dn = torch.randn(batch, seq_len, channels, device=dev, dtype=bf16)
    heads = channels // args.head_dim
    versions = {"fla": getattr(fla, "__version__", "?"), "torch": torch.__version__}

    if args.conv == "fla":
        from fla.modules.causal_conv1d.ops import causal_conv1d

        cu = None
        if args.doc_len:
            cu = torch.arange(0, batch * seq_len + 1, args.doc_len, device=dev, dtype=torch.int32)

        def conv(xv, wv):
            if cu is None:
                return causal_conv1d(xv, wv, None, activation="silu", backend="triton")[0]
            y = causal_conv1d(xv.reshape(1, batch * seq_len, channels), wv, None, activation="silu",
                              backend="triton", cu_seqlens=cu)[0]
            return y.reshape(batch, seq_len, channels)
    else:
        import causal_conv1d as cc
        from causal_conv1d import causal_conv1d_fn

        versions["causal_conv1d"] = cc.__version__
        seq_idx = None
        if args.doc_len:
            seq_idx = (torch.arange(seq_len, device=dev, dtype=torch.int32) // args.doc_len).expand(batch, -1)
            seq_idx = seq_idx.contiguous()

        def conv(xv, wv):
            return causal_conv1d_fn(xv.transpose(1, 2), wv, None, seq_idx=seq_idx, activation="silu").transpose(1, 2)

    def call(xv, wv):
        y = conv(xv, wv)
        return l2norm(y.reshape(batch, seq_len, heads, args.head_dim)).reshape(batch, seq_len, channels)

    if args.mode == "fwd":
        def step():
            with torch.no_grad():
                call(x, w)
    else:
        xg, wg = x.clone().requires_grad_(True), w.clone().requires_grad_(True)

        def step():
            torch.autograd.grad(call(xg, wg), (xg, wg), grad_outputs=dn)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind=f"{args.conv}_conv+fla_l2norm", shape=[batch, seq_len, channels],
                          head_dim=args.head_dim, mode=args.mode, doc_len=args.doc_len,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          **versions)))


if __name__ == "__main__":
    main()
