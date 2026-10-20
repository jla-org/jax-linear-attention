# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek's production DSA kernels as baselines: DeepGEMM's indexer logits and FlashMLA's sparse prefill.

* ``indexer``: DeepGEMM's lightning-indexer logits kernel (``deep_gemm.fp8_fp4_mqa_logits``; on Hopper,
  FP8 e4m3 queries and keys with one float scale per key, float32 head weights and logits) with
  per-query key ranges ``[0, t]`` (causal, so the kernel skips the future), then ``-inf`` outside the
  range and ``torch.topk(min(top_k, end))`` per query chunk, as DeepSeek-V3.2's inference code selects.
  Queries run in chunks of ``--chunk`` rows against the causal prefix of the chunk. The FP8 casts of
  q and k are not timed (production fuses them into the projections / the indexer cache write).
* ``sparse_prefill``: FlashMLA's sparse MLA prefill (``flash_mla.flash_mla_sparse_fwd``, forward only:
  FlashMLA has no backward for it) in DeepSeek-V3.2's absorbed MQA form: ``h_q`` query heads (128),
  one latent key/value head, ``d_qk = 576``, ``d_v = 512``, ``top_k = 2048`` distinct causal indices
  per query (stratified random over the query's prefix; ``-1`` pads queries with fewer keys).

Shapes as ``bench_dsa_long.py``: ``B = 1``, indexer 64 heads of 128, ``top_k = 2048``. Timing: median
of ``--iterations`` CUDA-event times after 3 warm-ups (``ms``) and the summed CUDA kernel time per call
from ``torch.profiler`` (``kernel_ms``, kernel-only like ``harness.median_kernel_ms``; a mean over the
iterations). Their code is not part of this repository: build FlashMLA (MIT) and DeepGEMM (MIT) from
source in a PyTorch environment (see ``benchmarks/dsa.md`` for the commits measured).

    python benchmarks/bench_dsa_production.py --out x.json [--lengths 16384,32768] [--parts indexer,sparse_prefill]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess

import torch

INDEX_HEADS, INDEX_DIM, TOP_K = 64, 128, 2048
D_QK, D_V = 576, 512


def _timed(fn, iterations: int) -> dict:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
    device_us = sum(e.device_time_total for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA)
    return dict(ms=statistics.median(times), kernel_ms=device_us / 1e3 / iterations)


def _fp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 e4m3 with one float scale per row (DeepGEMM's ``per_custom_dims_cast_to_fp8(x, (0,), False)``)."""
    scale = x.float().abs().amax(dim=1).clamp(min=1e-4) / 448.0
    return (x.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def _indexer_rows(length: int, chunk: int, iterations: int) -> list[dict]:
    import deep_gemm

    gen = torch.Generator(device="cuda").manual_seed(length)
    q = torch.randn(length, INDEX_HEADS, INDEX_DIM, device="cuda", generator=gen).bfloat16()
    k = torch.randn(length, INDEX_DIM, device="cuda", generator=gen).bfloat16()
    w = torch.randn(length, INDEX_HEADS, device="cuda", generator=gen) * (INDEX_HEADS ** -0.5 * INDEX_DIM ** -0.5)
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = _fp8_rows(k)
    w = w.contiguous()  # 64 float32 heads = 256 bytes per row: already 16-byte aligned for the TMA
    del q, k
    chunks = []
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        ke = torch.arange(start + 1, end + 1, device="cuda", dtype=torch.int32)
        ks = torch.zeros_like(ke)
        cols = torch.arange(end, device="cuda", dtype=torch.int32)
        chunks.append((start, end, ks, ke, cols))

    def run(with_topk: bool):
        out = []
        for start, end, ks, ke, cols in chunks:
            logits = deep_gemm.fp8_fp4_mqa_logits(
                q=(q_fp8[start:end], None), kv=(k_fp8[:end], k_scale[:end]), weights=w[start:end],
                cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, max_seqlen_k=end)
            if with_topk:
                logits = logits.masked_fill(cols[None, :] >= ke[:, None], float("-inf"))
                out.append(logits.topk(min(TOP_K, end), dim=-1)[1])
            else:
                out.append(logits[:, :1])
        return out

    rows = []
    for with_topk in (True, False):
        row = dict(part="index_topk" if with_topk else "indexer", T=length, B=1, HI=INDEX_HEADS, DI=INDEX_DIM,
                   top_k=TOP_K, chunk=chunk, iterations=iterations,
                   impl="deepgemm fp8_fp4_mqa_logits (FP8)" + (" + mask + torch.topk" if with_topk else ", no top-k"),
                   indexer_tflop=length * (length + 1) / 2 * INDEX_HEADS * INDEX_DIM * 2.0 / 1e12)
        try:
            row.update(_timed(lambda: run(with_topk), iterations))
        except Exception as err:  # record and continue (e.g. out of memory)
            row["error"] = f"{type(err).__name__}: {str(err)[:300]}"
        print(row, flush=True)
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def _sparse_prefill_row(length: int, heads: int, iterations: int) -> dict:
    import flash_mla

    gen = torch.Generator(device="cuda").manual_seed(length + heads)
    q = (torch.randn(length, heads, D_QK, device="cuda", generator=gen) * 0.5).bfloat16()
    kv = (torch.randn(length, 1, D_QK, device="cuda", generator=gen) * 0.5).bfloat16()
    t = torch.arange(length, device="cuda")[:, None]
    j = torch.arange(TOP_K, device="cuda")[None, :]
    u = torch.rand(length, 1, device="cuda", generator=gen)
    spread = torch.floor((j + u) * (t + 1).float() / TOP_K).long()  # distinct: the stride (t + 1) / K >= 1
    indices = torch.where(t + 1 >= TOP_K, spread, torch.where(j <= t, j, -1)).int()[:, None, :].contiguous()
    del t, j, u, spread
    selected = torch.clamp(torch.arange(length, device="cuda") + 1, max=TOP_K).double().sum().item()
    row = dict(part="sparse_prefill", impl="flashmla flash_mla_sparse_fwd (bf16)", T=length, B=1, HQ=heads,
               H=1, D_QK=D_QK, D_V=D_V, top_k=TOP_K, iterations=iterations,
               tflop=2.0 * selected * heads * (D_QK + D_V) / 1e12)
    try:
        row.update(_timed(lambda: flash_mla.flash_mla_sparse_fwd(q, kv, indices, D_QK ** -0.5, D_V), iterations))
        row["tflops_per_s"] = row["tflop"] / (row["kernel_ms"] / 1e3)
    except Exception as err:
        row["error"] = f"{type(err).__name__}: {str(err)[:300]}"
    print(row, flush=True)
    del q, kv, indices
    torch.cuda.empty_cache()
    return row


def _commit(module) -> str:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:  # noqa: BLE001
        return "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    parser.add_argument("--parts", default="indexer,sparse_prefill")
    parser.add_argument("--heads", default="128", help="query heads of the sparse prefill (comma separated)")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]  # this run's GPU only
    smi = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    rows = []

    def save():
        meta = dict(torch=torch.__version__, device=torch.cuda.get_device_name(0), gpu_before=smi,
                    deep_gemm=_commit("deep_gemm"), flash_mla=_commit("flash_mla"))
        with open(args.out, "w") as f:
            json.dump(dict(environment=meta, rows=rows), f, indent=1)

    for length in (int(x) for x in args.lengths.split(",")):
        if "indexer" in parts:
            rows.extend(_indexer_rows(length, args.chunk, args.iterations))
            save()
        if "sparse_prefill" in parts:
            for heads in (int(h) for h in args.heads.split(",")):
                rows.append(_sparse_prefill_row(length, heads, args.iterations))
                save()
    save()


if __name__ == "__main__":
    main()
