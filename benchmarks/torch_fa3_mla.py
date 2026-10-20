# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-3 worker for MLA's asymmetric heads (run in a torch environment with ``flash_attn_3``).

Started by ``bench_mla_attention.py``; one JSON request per line on stdin, one JSON answer per line:
``{"op": "time", "B", "T", "H", "Dqk", "Dv", "doc_lens", "iterations"}`` gives the median CUDA kernel
time (``torch.profiler``, as ``torch_fa3.py``) of the causal forward and of forward + backward for a fixed
cotangent, with ``q, k [B, T, H, Dqk]`` and ``v [B, T, H, Dv]`` (FA3 supports ``Dqk`` 192 with ``Dv`` 128;
its backward runs the 192-wide kernel with the values padded). ``doc_lens`` packs each row's documents
into one varlen batch (``cu_seqlens``).
"""

from __future__ import annotations

import json
import statistics
import sys

import numpy as np
import torch

from torch_fa3 import _fa3, _safe_module

torch.cuda.set_per_process_memory_fraction(0.25)


def _kernel_ms(step, iterations: int, required: tuple[str, ...] = (),
               max_attempts: int = 300) -> tuple[list[float], list[int]]:
    """``(kernel times of the complete captures, launch counts of every capture)``, as ``torch_fa3._kernel_ms``.

    A capture is complete when it recorded as many kernel launches as the most complete one seen and,
    for every name in ``required``, a kernel whose name contains it (case-insensitive): a step whose
    attention kernel the profiler never captured must not pass as a fast one.
    """
    from torch.profiler import ProfilerActivity, profile

    captures = []  # (kernel launches, ms)
    complete = []
    for _ in range(max_attempts):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        kernels = [e for e in prof.key_averages() if e.self_device_time_total > 0
                   and not e.key.startswith(("Memcpy", "Memset"))]
        names = " ".join(e.key.lower() for e in kernels)
        if not all(part in names for part in required):
            captures.append((0, 0.0))
            continue
        captures.append((sum(e.count for e in kernels), sum(e.self_device_time_total for e in kernels) / 1e3))
        full = max(n for n, _ in captures)
        complete = [ms for n, ms in captures if n == full and n > 0]
        if len(complete) >= iterations:
            break
    return complete, [n for n, _ in captures]


def _varlen(doc_lens, device):
    """``(cu_seqlens, longest)`` of the packed rows, built once per shape (not inside the timed call)."""
    lens = [n for row in doc_lens for n in row]
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), device=device, dtype=torch.int32)
    return cu, max(lens)


def _call(fa, q, k, v, varlen):
    batch, length, heads = q.shape[:3]
    if varlen is None:
        return fa.flash_attn_func(q, k, v, causal=True)
    cu, longest = varlen
    flat = lambda x: x.reshape(batch * length, *x.shape[2:])  # noqa: E731
    out = fa.flash_attn_varlen_func(flat(q), flat(k), flat(v), cu, cu, longest, longest, causal=True)
    return out.reshape(batch, length, heads, v.shape[-1])


def _time(req: dict) -> dict:
    fa = _fa3()
    batch, length, heads, dqk, dv = (req[n] for n in ("B", "T", "H", "Dqk", "Dv"))
    gen = torch.Generator(device="cuda").manual_seed(length)

    def randn(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen).to(torch.bfloat16)

    q, k, v = randn(batch, length, heads, dqk), randn(batch, length, heads, dqk), randn(batch, length, heads, dv)
    do = randn(batch, length, heads, dv)
    for x in (q, k, v):
        x.requires_grad_()
    doc_lens = req.get("doc_lens")
    varlen = None if doc_lens is None else _varlen(doc_lens, q.device)
    iterations = int(req.get("iterations", 30))
    result = {}

    def fwd():
        with torch.no_grad():
            _call(fa, q, k, v, varlen)

    def fwdbwd():
        torch.autograd.grad(_call(fa, q, k, v, varlen), (q, k, v), grad_outputs=do)

    # FA3's kernels are flash::FlashAttnFwdSm90 / FlashAttnBwdSm90 (hopper/flash_{fwd,bwd}_kernel_sm90.h)
    for name, step, required in (("fwd", fwd, ("fwd",)), ("fwdbwd", fwdbwd, ("fwd", "bwd"))):
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        times, counts = _kernel_ms(step, iterations, required)
        if len(times) >= iterations:
            result[f"{name}_ms"] = statistics.median(times)
            result[f"{name}_min_ms"] = min(times)
        else:  # the wall time below still measures this shape (an upper bound of the kernel time)
            result[f"{name}_kernel_error"] = (f"only {len(times)} of {iterations} profiler captures were complete; "
                                              f"launch counts seen: {sorted(set(counts))}")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            step()
        end.record()
        torch.cuda.synchronize()
        result[f"{name}_wall_ms"] = start.elapsed_time(end) / iterations
    return result


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        try:
            result = _time(req)
        except Exception as err:  # report the failure, keep serving
            result = {"error": f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"[:300]}
        result["flash_attn_3"] = getattr(_safe_module(), "__version__", None)
        result["torch"] = torch.__version__
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
