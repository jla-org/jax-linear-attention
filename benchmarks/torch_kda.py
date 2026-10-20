# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch KDA baselines for ``bench_kda.py`` (run in a torch environment).

    python torch_kda.py --shape 1,4096,8,128,128 --mode fwd [--impl fla|fla_default|flashkda]
                        [--contract plain|raw] [--cu 0,512,...,T]

Prints one JSON line: median CUDA kernel time per call (``torch.profiler``, the sum of kernel
durations; the metric of ``harness.median_kernel_ms``).

* ``--impl fla``: FLA's Triton ``chunk_kda``, gates by keyword. ``FLA_FLASH_KDA=0`` and
  ``FLA_TILELANG=0`` keep FLA's dispatcher off its FlashKDA (inference) and TileLang (fused WY
  backward) backends, so this is always the Triton path.
* ``--impl fla_default``: the same call with FLA's default dispatch (no backend switched off): with
  FlashKDA and TileLang installed, the raw-contract forward goes to FlashKDA (it needs
  ``state_v_first=True``, which this call passes) and the backward uses the TileLang WY kernel.
* ``--impl flashkda``: MoonshotAI's FlashKDA ``flash_kda.fwd`` (CUTLASS, forward only, needs
  ``--contract raw``).
* ``--contract plain`` (default): unit-norm ``k``, ``g = logsigmoid(N(0, 1))``, ``beta =
  sigmoid(N(0, 1))``. ``--contract raw``: FlashKDA's inference contract, raw ``q, k`` (L2 norm in
  the kernel), raw gate ``g [B, T, H, K]`` bf16 with ``A_log [H]``, ``dt_bias [H, K]`` and
  ``lower_bound = -5`` (``g <- lower_bound sigmoid(exp(A_log) (g + dt_bias))``), and beta logits.
* ``--cu``: packed documents (``B = 1``), passed as ``cu_seqlens``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.25)
LOWER_BOUND = -5.0


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


def raw_inputs(batch, seq_len, heads, key_dim, value_dim) -> dict:
    """FlashKDA's inference inputs (bf16 raw tensors, f32 gate parameters)."""
    dev, bf16 = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)

    def randn(*shape, dtype=bf16):
        return torch.randn(*shape, device=dev, generator=gen).to(dtype)

    return dict(
        q=randn(batch, seq_len, heads, key_dim), k=randn(batch, seq_len, heads, key_dim),
        v=randn(batch, seq_len, heads, value_dim), g=randn(batch, seq_len, heads, key_dim),
        beta=randn(batch, seq_len, heads),
        A_log=torch.log(torch.linspace(1.0, 16.0, heads, device=dev)),
        dt_bias=0.1 * randn(heads, key_dim, dtype=torch.float32),
    )


def make_call(impl: str, contract: str, shape, cu_seqlens=None):
    batch, seq_len, heads, key_dim, value_dim = shape
    scale = key_dim**-0.5
    if contract == "raw":
        x = raw_inputs(*shape)
        if impl == "flashkda":
            import flash_kda

            out = torch.empty_like(x["v"])

            def call():
                flash_kda.fwd(x["q"], x["k"], x["v"], x["g"], x["beta"], scale, out, x["A_log"], x["dt_bias"],
                              LOWER_BOUND, cu_seqlens=cu_seqlens)
                return out

            return call, None, None
        from fla.ops.kda import chunk_kda

        def call():
            return chunk_kda(q=x["q"], k=x["k"], v=x["v"], g=x["g"], beta=x["beta"], scale=scale, A_log=x["A_log"],
                             dt_bias=x["dt_bias"], lower_bound=LOWER_BOUND, safe_gate=True, use_gate_in_kernel=True,
                             use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True,
                             cu_seqlens=cu_seqlens)[0]

        return call, None, None
    if impl == "flashkda":
        raise ValueError("flashkda needs --contract raw")
    from fla.ops.kda import chunk_kda

    dev, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, key_dim, device=dev, dtype=bf16, requires_grad=True)
    k = F.normalize(torch.randn(batch, seq_len, heads, key_dim, device=dev), p=2, dim=-1).to(bf16)
    k.requires_grad_()
    v = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16, requires_grad=True)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, key_dim, device=dev)).requires_grad_()
    beta = torch.randn(batch, seq_len, heads, device=dev).sigmoid().requires_grad_()
    do = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16)

    def call():
        return chunk_kda(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens)[0]

    return call, [q, k, v, g, beta], do


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True, help="B,T,H,K,V")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    parser.add_argument("--impl", choices=("fla", "fla_default", "flashkda"), default="fla")
    parser.add_argument("--contract", choices=("plain", "raw"), default="plain")
    parser.add_argument("--cu", default=None, help="comma-separated cu_seqlens (packed documents, B = 1)")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.impl == "fla":  # Triton only: FLA's FlashKDA and TileLang backends off
        os.environ["FLA_FLASH_KDA"] = "0"
        os.environ["FLA_TILELANG"] = "0"
    else:  # FLA's own defaults
        os.environ.pop("FLA_FLASH_KDA", None)
        os.environ.pop("FLA_TILELANG", None)
    shape = tuple(int(x) for x in args.shape.split(","))
    cu_seqlens = None
    if args.cu:
        cu_seqlens = torch.tensor([int(x) for x in args.cu.split(",")], dtype=torch.long, device="cuda")
        if shape[0] != 1 or int(cu_seqlens[-1]) != shape[1]:
            raise ValueError("--cu needs B = 1 and cu[-1] = T")
    call, inputs, do = make_call(args.impl, args.contract, shape, cu_seqlens)
    if args.mode == "fwd":
        def step():
            with torch.inference_mode():
                call()
    else:
        if inputs is None:
            raise ValueError("forward only: FlashKDA has no backward, and the raw contract is benchmarked forward")
        def step():
            torch.autograd.grad(call(), inputs, grad_outputs=do)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    print(json.dumps(dict(kind=f"{args.impl}_kda_{args.contract}", shape=shape, mode=args.mode,
                          median_ms=statistics.median(times), min_ms=min(times), iterations=len(times),
                          torch=torch.__version__, fla=version("flash-linear-attention") or version("fla-core"),
                          tilelang=version("tilelang"), flash_kda=version("flash_kda") or version("flash-kda"),
                          backends_off=args.impl == "fla", docs=None if cu_seqlens is None else len(cu_seqlens) - 1,
                          lower_bound=LOWER_BOUND if args.contract == "raw" else None)))


if __name__ == "__main__":
    main()
