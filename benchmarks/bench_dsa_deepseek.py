# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek's released DSA indexer kernel (DeepSeek-V3.2-Exp ``inference/kernel.py``, TileLang, FP8) as a baseline.

Runs DeepSeek's prefill indexer as ``Indexer.forward`` does (``act_quant`` of q and k to FP8 e4m3 with
128-wide block scales, ``fp8_index`` = per-head GEMM, ReLU, head weights, sum, then the causal mask
and ``topk(min(index_topk, end_pos))``), without the Hadamard rotation (an orthogonal transform
that does not change the cost of these kernels). Their kernel scores every key for every query (no
causal skipping) and returns a ``[B, M, N]`` float32 tensor, so long sequences run in query chunks of
``--chunk`` rows against all keys; the time is the whole chunk loop. Same shapes as
``bench_dsa_long.py`` (64 index heads of 128, top-k 2048, ``B = 1``). Timing: CUDA events, median
of ``--iterations`` after 3 warm-ups. DeepSeek's code is not part of this repository: pass the
directory of their ``inference/kernel.py`` (MIT) with ``--deepseek-inference``; it needs TileLang.

    python benchmarks/bench_dsa_deepseek.py --deepseek-inference /path/DeepSeek-V3.2-Exp/inference --out x.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

INDEX_HEADS, INDEX_DIM, TOP_K = 64, 128, 2048


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-inference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="16384,32768,65536,131072")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, args.deepseek_inference)
    from kernel import act_quant, fp8_index  # DeepSeek-V3.2-Exp inference/kernel.py (TileLang)

    torch.cuda.set_per_process_memory_fraction(0.25)
    import tilelang

    rows = []
    for length in (int(x) for x in args.lengths.split(",")):
        gen = torch.Generator(device="cuda").manual_seed(length)
        q = torch.randn(1, length, INDEX_HEADS, INDEX_DIM, device="cuda", generator=gen).bfloat16()
        k = torch.randn(1, length, INDEX_DIM, device="cuda", generator=gen).bfloat16()
        w = torch.randn(1, length, INDEX_HEADS, device="cuda", generator=gen) * INDEX_HEADS ** -0.5
        pos = torch.arange(length, device="cuda")

        def run(with_topk=True):
            q_fp8, q_scale = act_quant(q.contiguous(), 128)
            k_fp8, k_scale = act_quant(k.contiguous(), 128)
            weights = (w.unsqueeze(-1) * q_scale * INDEX_DIM ** -0.5).squeeze(-1)  # [B, M, HI]
            k_scale = k_scale.squeeze(-1)  # [B, N]
            out = []
            for start in range(0, length, args.chunk):
                end = min(start + args.chunk, length)
                score = fp8_index(q_fp8[:, start:end].contiguous(), weights[:, start:end].contiguous(),
                                  k_fp8[:, :end].contiguous(), k_scale[:, :end].contiguous())
                score = score.masked_fill(pos[None, start:end, None] < pos[None, None, :end], float("-inf"))
                out.append(score.topk(min(TOP_K, end), dim=-1)[1] if with_topk else score[..., :1])
            return out

        for with_topk in (True, False):
            impl = "deepseek fp8_index (TileLang, FP8)" + (" + torch.topk" if with_topk else ", no top-k")
            row = dict(part="index_topk" if with_topk else "indexer", impl=impl, T=length, B=1, HI=INDEX_HEADS,
                       DI=INDEX_DIM, top_k=TOP_K, chunk=args.chunk, keys="causal prefix per chunk")
            try:
                for _ in range(3):
                    run(with_topk)
                times = []
                for _ in range(args.iterations):
                    start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_ev.record()
                    run(with_topk)
                    end_ev.record()
                    torch.cuda.synchronize()
                    times.append(start_ev.elapsed_time(end_ev))
                row["fwd_ms"] = statistics.median(times)
            except Exception as err:  # record and continue (e.g. out of memory, TileLang API drift)
                row["error"] = f"{type(err).__name__}: {str(err)[:300]}"
            print(row, flush=True)
            rows.append(row)
        del q, k, w
        torch.cuda.empty_cache()
    meta = dict(torch=torch.__version__, tilelang=getattr(tilelang, "__version__", "?"),
                device=torch.cuda.get_device_name(0))
    with open(args.out, "w") as f:
        json.dump(dict(environment=meta, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
