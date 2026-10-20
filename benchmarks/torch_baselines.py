# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch baselines for the linear-attention benchmarks (run in a torch environment).

    python torch_baselines.py --kind fla_gdn --shape 1,4096,8,128,128 --mode fwd

Kinds:

* ``fla_gdn``: ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (FLA Triton);
* ``fla_simple_gla``: ``fla.ops.simple_gla.chunk_simple_gla`` (FLA Triton);
* ``qla_gdn``: ``flash_qla.chunk_gated_delta_rule`` (FlashQLA 0.1.3, TileLang; run in an environment
  that has it, while ``fla_gdn`` runs where it is absent so that FLA keeps its Triton path);
* ``mamba2``: ``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`` from a
  state-spaces/mamba source tree (``--mamba-root``), imported without the
  package ``__init__`` (which needs compiled CUDA extensions).

Prints one JSON line: median CUDA kernel time per call (``torch.profiler``,
sum of kernel durations, the same metric as ``harness.median_kernel_ms``).
Gates are passed to FLA by keyword (its fused_recurrent's fifth positional
argument is ``gk``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types

import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.10)


def import_mamba_ssd(root: str | None):
    """``root`` is a source tree; ``None`` uses the installed package (still without its ``__init__``)."""
    package = types.ModuleType("mamba_ssm")
    if root is None:
        import importlib.metadata
        import importlib.util

        package.__path__ = list(importlib.util.find_spec("mamba_ssm").submodule_search_locations)
        package.__version__ = importlib.metadata.version("mamba_ssm")  # the skipped __init__ sets it
    else:
        package.__path__ = [f"{root}/mamba_ssm"]
    sys.modules.setdefault("mamba_ssm", package)
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    return mamba_chunk_scan_combined


def _flat(x):
    """``[B, T, ...]`` to one varlen row ``[1, B * T, ...]``."""
    return x.reshape(1, -1, *x.shape[2:])


def _varlen(*xs):
    return tuple(_flat(x) for x in xs)


def _cu_seqlens(tokens: int, doc_len: int):
    return torch.arange(0, tokens + 1, doc_len, device="cuda", dtype=torch.long)


def make_call(kind: str, shape: tuple[int, ...], groups: int, mamba_root: str | None, chunk: int, doc_len: int = 0):
    """``(call, inputs, output shape)``. ``doc_len > 0`` packs documents of that many tokens.

    FLA then runs B * T tokens as one varlen row (``cu_seqlens``) and Mamba-2 gets ``seq_idx``.
    """
    batch, seq_len, heads, key_dim, value_dim = shape
    dev, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    if kind == "mamba2":
        ssd = import_mamba_ssd(mamba_root)
        x = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16, requires_grad=True)
        dt = F.softplus(torch.randn(batch, seq_len, heads, device=dev) - 2.0).requires_grad_()
        a = (-torch.rand(heads, device=dev) - 0.5).requires_grad_()
        b = torch.randn(batch, seq_len, groups, key_dim, device=dev, dtype=bf16, requires_grad=True)
        c = torch.randn(batch, seq_len, groups, key_dim, device=dev, dtype=bf16, requires_grad=True)
        inputs = [x, dt, a, b, c]
        seq_idx = None
        if doc_len:
            pos = torch.arange(seq_len, device=dev, dtype=torch.int32)
            seq_idx = (pos // doc_len).expand(batch, seq_len).contiguous()

        def call():
            return ssd(x, dt, a, b, c, chunk_size=chunk, seq_idx=seq_idx)

        return call, inputs, (batch, seq_len, heads, value_dim)
    q = torch.randn(batch, seq_len, heads, key_dim, device=dev, dtype=bf16, requires_grad=True)
    k = F.normalize(torch.randn(batch, seq_len, heads, key_dim, device=dev), p=2, dim=-1).to(bf16)
    k.requires_grad_()
    v = torch.randn(batch, seq_len, heads, value_dim, device=dev, dtype=bf16, requires_grad=True)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, device=dev)).requires_grad_()
    if kind == "fla_gdn":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        beta = torch.rand(batch, seq_len, heads, device=dev).sigmoid().requires_grad_()
        inputs = [q, k, v, g, beta]
        cu = _cu_seqlens(batch * seq_len, doc_len) if doc_len else None

        def call():
            if doc_len:
                o = chunk_gated_delta_rule(*_varlen(q, k, v), g=_flat(g), beta=_flat(beta), cu_seqlens=cu)[0]
                return o.reshape(batch, seq_len, heads, value_dim)
            return chunk_gated_delta_rule(q, k, v, g=g, beta=beta)[0]

        return call, inputs, (batch, seq_len, heads, value_dim)
    if kind == "qla_gdn":  # FlashQLA (TileLang), auto intra-card CP on as by default
        from flash_qla import chunk_gated_delta_rule as qla_chunk_gated_delta_rule

        beta = torch.rand(batch, seq_len, heads, device=dev).sigmoid().requires_grad_()
        inputs = [q, k, v, g, beta]
        cu = _cu_seqlens(batch * seq_len, doc_len).int() if doc_len else None

        def call():
            if doc_len:
                o = qla_chunk_gated_delta_rule(*_varlen(q, k, v), g=_flat(g), beta=_flat(beta), cu_seqlens=cu)[0]
                return o.reshape(batch, seq_len, heads, value_dim)
            return qla_chunk_gated_delta_rule(q, k, v, g=g, beta=beta)[0]

        return call, inputs, (batch, seq_len, heads, value_dim)
    if kind == "fla_simple_gla":
        from fla.ops.simple_gla import chunk_simple_gla

        inputs = [q, k, v, g]
        cu = _cu_seqlens(batch * seq_len, doc_len) if doc_len else None

        def call():
            if doc_len:
                o = chunk_simple_gla(*_varlen(q, k, v), g=_flat(g), scale=1.0, cu_seqlens=cu)[0]
                return o.reshape(batch, seq_len, heads, value_dim)
            return chunk_simple_gla(q, k, v, g=g, scale=1.0)[0]

        return call, inputs, (batch, seq_len, heads, value_dim)
    raise ValueError(f"unknown kind {kind!r}")


def kernel_ms(step, iterations: int) -> list[float]:
    from torch.profiler import ProfilerActivity, profile

    out = []
    for _ in range(iterations):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        # Each kernel is "self" time of exactly one profiled op, so this sums every kernel once.
        total_us = sum(e.self_device_time_total for e in prof.key_averages())
        out.append(total_us / 1e3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--shape", required=True, help="B,T,H,K,V")
    parser.add_argument("--mode", choices=("fwd", "fwdbwd"), required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--groups", type=int, default=1, help="Mamba-2 ngroups")
    parser.add_argument("--chunk", type=int, default=256, help="Mamba-2 chunk size")
    parser.add_argument("--mamba-root", default=None)
    parser.add_argument("--doc-len", type=int, default=0, help="pack documents of this many tokens (0: none)")
    args = parser.parse_args()
    shape = tuple(int(x) for x in args.shape.split(","))
    call, inputs, out_shape = make_call(args.kind, shape, args.groups, args.mamba_root, args.chunk, args.doc_len)
    do = torch.randn(*out_shape, device="cuda", dtype=torch.bfloat16)

    if args.mode == "fwd":
        def step():
            with torch.no_grad():
                call()
    else:
        def step():
            torch.autograd.grad(call(), inputs, grad_outputs=do)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = kernel_ms(step, args.iterations)
    print(json.dumps(dict(kind=args.kind, shape=shape, mode=args.mode, median_ms=statistics.median(times),
                          min_ms=min(times), iterations=len(times), doc_len=args.doc_len,
                          torch=torch.__version__, versions=package_versions())))


def package_versions() -> dict:
    """Versions of the baseline packages loaded in this process (FLA main reports 0.6.0)."""
    out = {}
    for name in ("fla", "flash_qla", "mamba_ssm", "triton", "tilelang"):
        module = sys.modules.get(name)
        if module is not None:
            out[name] = getattr(module, "__version__", "?")
    return out


if __name__ == "__main__":
    main()
