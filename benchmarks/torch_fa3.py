# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-3 baseline worker (run in a torch environment with ``flash_attn_3`` installed).

Started by ``bench_attention_fa3.py`` and ``attention_fa3_diff.py`` as a subprocess; it reads one
JSON request per line on stdin and answers with one JSON line on stdout:

* ``{"op": "time", "B", "T", "H", "Hkv", "D", "window", "doc_lens", "iterations"}``: median CUDA
  kernel time (``torch.profiler``, sum of kernel durations, the metric of
  ``harness.median_kernel_ms``) of the forward and of forward + backward for a fixed cotangent.
  Every timing also measures the GPU time of the same calls from CUDA events, one call at a time, each
  queued behind a sleep kernel (no host launch gaps; on a quiet H100 it reads 1% above complete
  captures). Only when the profiler records fewer than ``MIN_CAPTURES`` complete captures, or captures
  well below that GPU time (a kernel dropped from every capture), is the event time reported instead
  (``fwd_timer`` / ``fwdbwd_timer`` = ``"events"``, otherwise ``"profiler"``).
  ``window`` is our ``sliding_window`` (visible tokens including the query), passed to FA3 as
  ``window_size=(window - 1, 0)``; ``doc_lens`` (one list per batch row, or null) packs every row's
  documents into one varlen batch with ``cu_seqlens``.
* ``{"op": "grads", "inputs": path, "output": path, ...}``: FA3 output and ``dq, dk, dv`` for the
  arrays in an ``.npz`` (``q, k, v, do`` as float32 holding bf16 values), saved as float32.
"""

from __future__ import annotations

import json
import statistics
import sys

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(0.25)
MIN_CAPTURES = 5  # complete profiler captures a timing needs (fewer than --iterations on a busy node)
# A profiler time whose kernels add up to less than this share of the GPU time of the same calls (CUDA
# events around calls queued behind a sleep kernel, so no host launch gaps; the remainder is copies, memsets and
# gaps of a microsecond or two between kernels) missed a kernel in every capture and is not reported.
MIN_KERNEL_SHARE_OF_GPU = 0.85
QUEUE_FILL_CYCLES = 20_000_000  # sleep kernel ahead of each event-timed call (about 10 ms on an H100)


def _fa3():
    import flash_attn_interface

    return flash_attn_interface


def _cu_seqlens(doc_lens, device):
    """``cu_seqlens`` (int32 on ``device``) and the longest document of a packed batch."""
    lens = [n for row in doc_lens for n in row]
    return torch.tensor(np.concatenate([[0], np.cumsum(lens)]), device=device, dtype=torch.int32), max(lens)


def _call(fa, q, k, v, window, doc_lens, scale, packed=None):
    """FA3 causal attention of ``q [B, T, H, D]``, ``k, v [B, T, Hkv, D]``.

    ``packed`` is ``_cu_seqlens(doc_lens)`` computed in advance (a timed call then makes no
    host-to-device copy); by default it is computed here."""
    batch, length, heads, head_dim = q.shape
    window_size = (-1, -1) if window is None or window >= length else (window - 1, 0)
    if doc_lens is None:
        return fa.flash_attn_func(q, k, v, softmax_scale=scale, causal=True, window_size=window_size)
    cu, longest = _cu_seqlens(doc_lens, q.device) if packed is None else packed
    flat = lambda x: x.reshape(batch * length, *x.shape[2:])  # noqa: E731
    out = fa.flash_attn_varlen_func(flat(q), flat(k), flat(v), cu, cu, longest, longest, softmax_scale=scale,
                                    causal=True, window_size=window_size)
    return out.reshape(batch, length, heads, head_dim)


def _kernel_ms(step, iterations: int, max_attempts: int = 300) -> list[float]:
    """Kernel time of ``iterations`` complete captures of ``step``.

    On a node where other processes profile too, a capture sometimes records no kernel or only
    some of them. A capture counts only if it recorded as many kernel launches as the most complete
    capture seen; incomplete ones are retried (at most ``max_attempts`` captures in all).
    """
    from torch.profiler import ProfilerActivity, profile

    captures = []  # (kernel launches, ms)
    for _ in range(max_attempts):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        # Copies and memsets are not attention kernels: the varlen call uploads its cu_seqlens with a
        # pageable copy that the profiler records only sometimes, which made the launch count unstable.
        kernels = [e for e in prof.key_averages() if e.self_device_time_total > 0
                   and not e.key.startswith(("Memcpy", "Memset"))]
        captures.append((sum(e.count for e in kernels), sum(e.self_device_time_total for e in kernels) / 1e3))
        full = max(n for n, _ in captures)
        complete = [ms for n, ms in captures if n == full and n > 0]
        if len(complete) >= iterations or (full == 0 and len(captures) >= 20):  # or the profiler records nothing
            return complete
    return complete


def _event_ms(step, iterations: int) -> tuple[list[float], bool]:
    """GPU time of each of ``iterations`` calls of ``step`` from CUDA events, one call at a time.

    A sleep kernel runs before each call, so the call is queued before the GPU reaches it and its
    event pair spans only its GPU work, with no host launch gap (launch-bound calls would otherwise
    include one); the GPU then idles until the next call, as between profiler captures and between
    the calls ``harness.median_kernel_ms`` times, so the clock drop of sustained back-to-back load (30
    long forward + backward calls in a row read 11% slower) is not timed either. Returns the times and
    whether every call was queued before its sleep ended."""
    times, queued_in_time = [], True
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        filled = torch.cuda.Event()
        torch.cuda._sleep(QUEUE_FILL_CYCLES)
        filled.record()
        start.record()
        step()
        end.record()
        queued_in_time = queued_in_time and not filled.query()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times, queued_in_time


def _time(req: dict) -> dict:
    fa = _fa3()
    batch, length, heads, kv_heads, head_dim = (req[n] for n in ("B", "T", "H", "Hkv", "D"))
    gen = torch.Generator(device="cuda").manual_seed(length)
    shape_q = (batch, length, heads, head_dim)
    shape_kv = (batch, length, kv_heads, head_dim)

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen).to(torch.bfloat16)

    q, k, v = randn(shape_q), randn(shape_kv), randn(shape_kv)
    do = randn(shape_q)
    for x in (q, k, v):
        x.requires_grad_()
    window, doc_lens = req.get("window"), req.get("doc_lens")
    iterations = int(req.get("iterations", 30))
    packed = None if doc_lens is None else _cu_seqlens(doc_lens, q.device)
    result = {}

    def fwd():
        with torch.no_grad():
            _call(fa, q, k, v, window, doc_lens, None, packed)

    def fwdbwd():
        torch.autograd.grad(_call(fa, q, k, v, window, doc_lens, None, packed), (q, k, v), grad_outputs=do)

    for name, step in (("fwd", fwd), ("fwdbwd", fwdbwd)):
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        times = _kernel_ms(step, iterations)
        gpu, queue_filled = _event_ms(step, iterations)
        if not queue_filled:
            raise RuntimeError("the sleep kernel ended before every event-timed call was queued")
        gpu_ms = statistics.median(gpu)
        profiler_ms = statistics.median(times) if times else None
        # While other jobs on the machine profile, most captures can come back incomplete: report the
        # median of the complete ones (their count is recorded) when there are at least MIN_CAPTURES.
        # With fewer, or when the complete ones add up to less than MIN_KERNEL_SHARE_OF_GPU of the GPU
        # time (a kernel dropped from every capture, which the launch count cannot show; measured in this
        # worker under the benchmark: the forward and preprocess kernels of fwd + bwd), the GPU time from
        # events is reported instead, labelled "events"; the profiler's median is kept as "<name>_profiler_ms".
        timer = "profiler"
        if len(times) < min(iterations, MIN_CAPTURES) or profiler_ms < MIN_KERNEL_SHARE_OF_GPU * gpu_ms:
            times, timer = gpu, "events"
        # Wall time of back-to-back calls between two CUDA events (launch gaps included).
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            step()
        end.record()
        torch.cuda.synchronize()
        result[f"{name}_timer"] = timer
        result[f"{name}_ms"] = statistics.median(times)
        result[f"{name}_min_ms"] = min(times)
        result[f"{name}_captures"] = len(times)
        result[f"{name}_profiler_ms"] = profiler_ms
        result[f"{name}_events_ms"] = gpu_ms
        result[f"{name}_wall_ms"] = start.elapsed_time(end) / iterations
    return result


def _grads(req: dict) -> dict:
    fa = _fa3()
    data = np.load(req["inputs"])
    to = lambda name: torch.tensor(data[name], device="cuda").to(torch.bfloat16)  # noqa: E731
    q, k, v = (to(n).requires_grad_() for n in ("q", "k", "v"))
    do = to("do")
    out = _call(fa, q, k, v, req.get("window"), req.get("doc_lens"), req.get("scale"))
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=do)
    arrays = {name: x.detach().float().cpu().numpy()
              for name, x in (("out", out), ("dq", dq), ("dk", dk), ("dv", dv))}
    np.savez(req["output"], **arrays)
    return {"saved": req["output"]}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        try:
            result = _time(req) if req["op"] == "time" else _grads(req)
        except Exception as err:  # report the failure, keep serving
            result = {"error": f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"[:300]}
        result["flash_attn_3"] = getattr(_safe_module(), "__version__", None)
        result["torch"] = torch.__version__
        print(json.dumps(result), flush=True)


def _safe_module():
    try:
        import flash_attn_3

        return flash_attn_3
    except ImportError:
        return None


if __name__ == "__main__":
    main()
