# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch fused linear cross entropy baselines (run in a torch environment that has them).

    python torch_cross_entropy.py --kind liger --shape 16384,768,128256 --mode fwdbwd
    python torch_cross_entropy.py --kind liger --shape 4096,768,128256 --accuracy

Kinds:

* ``liger``: ``liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`` (Liger-Kernel);
* ``cce``: ``cut_cross_entropy.linear_cross_entropy`` with its default gradient filtering
  (``filter_eps="auto"``, which skips backward blocks whose softmax is below bf16 resolution);
* ``cce_exact``: the same with ``filter_eps=None`` (no filtering);
* ``fla``: ``fla.modules.FusedLinearCrossEntropyLoss`` (flash-linear-attention, the loss of
  ``flame`` and the FLA models) with its default ``accumulate_grad_in_fp32=True``: ``dweight``
  accumulates across row chunks in float32 through float32 GEMMs;
* ``fla_tf32``: the same with ``torch.backends.cuda.matmul.allow_tf32 = True``, so those
  float32 GEMMs run on TF32 tensor cores;
* ``fla_bf16acc``: ``accumulate_grad_in_fp32=False`` (bfloat16 ``dweight`` accumulation);
* ``torch``: ``F.cross_entropy(hidden @ weight.T)`` with full float32 logits (reference only).

Liger and FLA form the gradients inside the forward call (per chunk of rows: bfloat16 logits,
the logit gradient in place, then the two gradient GEMMs). Liger skips them when no input
requires a gradient (the ``fwd`` mode); FLA always forms them, so it has no forward-only
path and is timed in ``fwdbwd`` mode only. All take bfloat16 ``hidden [N, D]`` and
``weight [V, D]``, mean reduction, no label smoothing and no z-loss. Prints one JSON line:
median CUDA kernel time per call (``torch.profiler``, sum of kernel durations, the same metric
as ``harness.median_kernel_ms``). ``--accuracy`` instead prints the errors of the loss and of
both gradients against float32 logits of the same bfloat16 values (no TF32), next to the
error of that reference itself rounded to bfloat16 (the floor for bfloat16 gradients).
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.25)


def make_loss(kind: str):
    if kind == "liger":
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")
        return lambda h, w, t: module(w, h, t)
    if kind in ("cce", "cce_exact"):
        from cut_cross_entropy import linear_cross_entropy

        if kind == "cce":
            return lambda h, w, t: linear_cross_entropy(h, w, t, ignore_index=-100, reduction="mean")
        return lambda h, w, t: linear_cross_entropy(h, w, t, ignore_index=-100, reduction="mean", filter_eps=None)
    if kind in ("fla", "fla_tf32", "fla_bf16acc"):
        from fla.modules import FusedLinearCrossEntropyLoss

        if kind == "fla_tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
        module = FusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean",
                                             accumulate_grad_in_fp32=kind != "fla_bf16acc")
        return lambda h, w, t: module(h, t, w)
    if kind == "torch":
        return lambda h, w, t: F.cross_entropy((h @ w.T).float(), t, ignore_index=-100)
    raise ValueError(f"unknown kind {kind!r}")


def versions() -> dict:
    out = {"torch": torch.__version__}
    for name in ("liger_kernel", "cut_cross_entropy", "fla", "triton"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        out[name] = getattr(module, "__version__", None)
    return out


def accuracy(kind: str, hidden, weight, targets) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    h32 = hidden.detach().float().requires_grad_(True)
    w32 = weight.detach().float().requires_grad_(True)
    ref = F.cross_entropy(h32 @ w32.T, targets, ignore_index=-100)
    ref_dh, ref_dw = torch.autograd.grad(ref, (h32, w32))
    del h32, w32
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    loss = make_loss(kind)(h, w, targets)
    dh, dw = torch.autograd.grad(loss, (h, w))

    def rel(a, b):
        return float((a.float() - b).norm() / b.norm())

    return dict(loss_abs=abs(float(loss) - float(ref)), dhidden_rel=rel(dh, ref_dh), dweight_rel=rel(dw, ref_dw),
                bf16_floor_dhidden=rel(ref_dh.bfloat16(), ref_dh), bf16_floor_dweight=rel(ref_dw.bfloat16(), ref_dw),
                grad_dtypes=[str(dh.dtype), str(dw.dtype)])


def kernel_ms(step, iterations: int) -> list[float]:
    from torch.profiler import ProfilerActivity, profile

    out = []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        total_us = sum(e.self_device_time_total for e in prof.key_averages())
        out.append(total_us / 1e3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--shape", required=True, help="N,D,V")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), default=None)
    parser.add_argument("--accuracy", action="store_true", help="errors against float32 logits instead of timing")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    n, d, v = (int(x) for x in args.shape.split(","))
    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(n, d, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = (torch.randn(v, d, device="cuda", dtype=torch.float32, generator=gen) * (2.0 / d ** 0.5)).bfloat16()
    targets = torch.randint(0, v, (n,), device="cuda", generator=gen)
    if args.accuracy:
        print(json.dumps(dict(kind=args.kind, shape=[n, d, v], **accuracy(args.kind, hidden, weight, targets),
                              versions=versions())))
        return
    if args.mode is None:
        parser.error("--mode is required unless --accuracy is given")
    loss_fn = make_loss(args.kind)
    if args.mode == "fwd":
        def step():
            with torch.no_grad():
                loss_fn(hidden, weight, targets)
    else:
        hidden.requires_grad_(True)
        weight.requires_grad_(True)

        def step():
            torch.autograd.grad(loss_fn(hidden, weight, targets), (hidden, weight))

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    with torch.no_grad():
        loss = float(loss_fn(hidden.detach(), weight.detach(), targets))
    print(json.dumps(dict(kind=args.kind, shape=[n, d, v], mode=args.mode, median_ms=statistics.median(times),
                          min_ms=min(times), iterations=len(times), loss=loss, versions=versions())))


if __name__ == "__main__":
    main()
