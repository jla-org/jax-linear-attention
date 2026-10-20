# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch baselines for the MoE grouped GEMM benchmark (run in a torch environment).

    python torch_grouped_mm.py gemm --k 768 --n 1536 --sizes 512,0,700,...
    python torch_grouped_mm.py layer --tokens 8192 --d-model 768 --experts 16 --top-k 2 --d-ff 768 --shared 1

``gemm``: ``torch._grouped_mm(a [M, K], b [G, K, N], offs)`` (PyTorch's CUTLASS sm90 grouped
GEMM), forward and forward + backward (``d_a``, ``d_b`` through autograd), for the weight stored
``[G, K, N]`` row-major (``kn``, the JAX layout) and ``[G, N, K]`` transposed to ``[G, K, N]``
(``nk``, the column-major operand CUTLASS prefers); ``--grouped-gemm`` adds
``grouped_gemm.ops.gmm`` (tgale96/grouped_gemm, used by MegaBlocks) when it is installed.

``layer``: a dropless token-choice MoE layer forward + backward with the structure of
``jla.layers.moe.MoE`` (softmax router, top-k, renormalized gates, sort by expert, grouped
GEMM up ``[E, D, 2F]``, SwiGLU times gate, grouped GEMM down, unsort and sum, plus a dense
shared SwiGLU expert), float32 parameters cast to bfloat16 in the forward pass, as in JAX.
Eager PyTorch (``torch_grouped_mm_layer_fwd_bwd``); ``--compile`` also times the same forward
under ``torch.compile`` (Inductor fuses the elementwise work, the backward goes through
AOTAutograd: ``torch_compile_grouped_mm_layer_fwd_bwd``).

Prints one JSON line: median CUDA kernel time per call (``torch.profiler``, sum of kernel
durations, the metric of ``harness.median_kernel_ms``).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.25)  # the DeepSeek-V3-like up projection needs ~10 GiB
BF16 = torch.bfloat16


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


def _measure(step, iterations: int) -> dict:
    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        times = kernel_ms(step, iterations)
    except Exception as err:  # noqa: BLE001 - recorded in the results
        return dict(error=f"{type(err).__name__}: {str(err)[:300]}")
    return dict(median_ms=statistics.median(times), min_ms=min(times), iterations=len(times))


def gemm(args) -> dict:
    sizes = torch.tensor([int(s) for s in args.sizes.split(",")], dtype=torch.int32, device="cuda")
    m, k, n, g = int(sizes.sum()), args.k, args.n, sizes.numel()
    offs = torch.cumsum(sizes, 0).to(torch.int32)
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=BF16, requires_grad=True)
    ct = torch.randn(m, n, device="cuda", dtype=BF16)
    weights = {
        "kn": (torch.randn(g, k, n, device="cuda") / math.sqrt(k)).to(BF16),
        "nk": (torch.randn(g, n, k, device="cuda") / math.sqrt(k)).to(BF16).transpose(-2, -1),
    }
    row = {}
    for layout, b in weights.items():
        b.requires_grad_()

        def fwd(b=b):
            with torch.no_grad():
                torch._grouped_mm(a, b, offs=offs)

        def fwd_bwd(b=b):
            torch.autograd.grad(torch._grouped_mm(a, b, offs=offs), (a, b), grad_outputs=ct)

        row[f"torch_grouped_mm_{layout}_fwd"] = _measure(fwd, args.iterations)
        row[f"torch_grouped_mm_{layout}_fwd_bwd"] = _measure(fwd_bwd, args.iterations)
    if args.grouped_gemm:
        try:
            from grouped_gemm import ops as gg_ops
        except ImportError as err:
            row["grouped_gemm"] = dict(error=f"ImportError: {err}")
        else:
            b = weights["kn"].detach().clone().requires_grad_()
            batch_sizes = sizes.cpu().to(torch.int64)  # grouped_gemm takes host batch sizes

            def gg_fwd():
                with torch.no_grad():
                    gg_ops.gmm(a, b, batch_sizes)

            def gg_fwd_bwd():
                torch.autograd.grad(gg_ops.gmm(a, b, batch_sizes), (a, b), grad_outputs=ct)

            row["grouped_gemm_fwd"] = _measure(gg_fwd, args.iterations)
            row["grouped_gemm_fwd_bwd"] = _measure(gg_fwd_bwd, args.iterations)
    return row


def layer(args) -> dict:
    tokens, d, e, top_k, f, shared = args.tokens, args.d_model, args.experts, args.top_k, args.d_ff, args.shared
    torch.manual_seed(0)
    std = 0.02

    def param(*shape):
        return (torch.randn(*shape, device="cuda") * std).requires_grad_()

    params = dict(router=param(d, e), gate_up=param(e, d, 2 * f), down=param(e, f, d))
    if shared:
        params["shared_gate_up"] = param(d, 2 * shared * f)
        params["shared_down"] = param(shared * f, d)
    x = torch.randn(tokens, d, device="cuda").to(BF16).requires_grad_()
    ct = torch.randn(tokens, d, device="cuda").to(BF16)

    def swiglu(h, width):
        gate, up = h.view(-1, 2, width).float().unbind(1)
        return (F.silu(gate) * up).to(BF16)

    def forward():
        p = {name: w.to(BF16) for name, w in params.items()}
        logits = (x @ p["router"]).float()
        gates, experts = torch.softmax(logits, dim=-1).topk(top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True)
        flat = experts.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token = order // top_k
        # Expert counts with a static shape (bincount's length depends on the data under torch.compile).
        counts = torch.zeros(e, dtype=flat.dtype, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
        offs = torch.cumsum(counts, 0).to(torch.int32)
        h = torch._grouped_mm(x[token], p["gate_up"], offs=offs)
        a = swiglu(h, f) * gates.reshape(-1)[order][:, None].to(BF16)
        y_sorted = torch._grouped_mm(a, p["down"], offs=offs)
        y = torch.empty_like(y_sorted).index_copy_(0, order, y_sorted)
        out = y.view(tokens, top_k, d).float().sum(dim=1).to(BF16)
        if shared:
            out = out + swiglu(x @ p["shared_gate_up"], shared * f) @ p["shared_down"]
        return out

    leaves = [x, *params.values()]

    def fwd_bwd(forward=forward):
        torch.autograd.grad(forward(), leaves, grad_outputs=ct)

    row = dict(torch_grouped_mm_layer_fwd_bwd=_measure(fwd_bwd, args.iterations))
    if args.compile:  # the warmup calls compile it
        compiled = torch.compile(forward)
        row["torch_compile_grouped_mm_layer_fwd_bwd"] = _measure(lambda: fwd_bwd(compiled), args.iterations)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="case", required=True)
    g = sub.add_parser("gemm")
    g.add_argument("--k", type=int, required=True)
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--sizes", required=True, help="comma-separated group sizes")
    g.add_argument("--grouped-gemm", action="store_true")
    g.add_argument("--iterations", type=int, default=30)
    lay = sub.add_parser("layer")
    for name in ("tokens", "d-model", "experts", "top-k", "d-ff"):
        lay.add_argument(f"--{name}", type=int, required=True)
    lay.add_argument("--shared", type=int, default=0)
    lay.add_argument("--iterations", type=int, default=30)
    lay.add_argument("--compile", action="store_true", help="also time the layer under torch.compile")
    args = parser.parse_args()
    row = gemm(args) if args.case == "gemm" else layer(args)
    row["torch"] = torch.__version__
    print(json.dumps(row))


if __name__ == "__main__":
    main()
