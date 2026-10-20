# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel time, launch grid and occupancy of a torch Gated DeltaNet call (FlashQLA or FLA), one JSON line each.

    QLA_PYTHON benchmarks/profile_qla_stages.py --kind qla_gdn --shapes "1,16384,16,128,128" [--doc-len 2048]

Kernel times are ``torch.profiler`` device times (median over ``--iterations`` profiled calls); the
grid, block, registers, shared memory and estimated occupancy come from the trace's kernel events.
For FlashQLA it also prints the automatic intra-card CP decision (pieces and chunks per piece).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_baselines import make_call  # noqa: E402


def cp_split(shape, doc_len: int) -> dict:
    """FlashQLA's automatic intra-card CP decision for this call (``_auto_intra_cp_context``)."""
    try:
        from flash_qla.ops.gated_delta_rule.chunk import _auto_intra_cp_context
    except ImportError:
        return {}
    batch, seq_len, heads, key_dim, value_dim = shape
    tokens = batch * seq_len
    k = torch.empty(1 if (doc_len or batch == 1) else batch, tokens if doc_len else seq_len, heads, key_dim,
                    device="cuda", dtype=torch.bfloat16)
    v = torch.empty(*k.shape[:3], value_dim, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(0, tokens + 1, doc_len, device="cuda", dtype=torch.int32) if doc_len else None
    out = {}
    for is_train in (False, True):
        try:
            ctx = _auto_intra_cp_context(k, v, cu, 64, True, is_train)
        except TypeError:
            return {}
        info = dict(is_intra=bool(ctx.is_intra))
        if ctx.is_intra:
            lens = (ctx.intra_cp_cu_seqlens[1:] - ctx.intra_cp_cu_seqlens[:-1]).tolist()
            info.update(pieces=len(lens), max_piece_chunks=-(-max(lens) // 64), min_piece_chunks=-(-min(lens) // 64))
        out["train" if is_train else "infer"] = info
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="qla_gdn")
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--doc-len", type=int, default=0)
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--gates", default="fast", choices=("fast", "slow", "mixed"),
                        help="fast: logsigmoid(N(0, 1)); slow: per-token decay 0.999; mixed: half the heads fast, "
                             "half with decays 0.98 to 0.999 (as gdn_warmup_error.py)")
    args = parser.parse_args()
    from torch.profiler import ProfilerActivity, profile

    for spec in args.shapes.split(";"):
        shape = tuple(int(x) for x in spec.split(","))
        call, inputs, out_shape = make_call(args.kind, shape, 1, None, 64, args.doc_len)
        if args.gates != "fast":
            g = inputs[3]
            heads = g.shape[-1]
            slow = torch.log(torch.linspace(0.98, 0.999, heads - heads // 2, device=g.device))
            with torch.no_grad():
                if args.gates == "slow":
                    g.fill_(float(np.log(0.999)))
                else:
                    g[..., heads // 2:] = slow
        do = torch.randn(*out_shape, device="cuda", dtype=torch.bfloat16)
        for mode in args.modes.split(","):
            if mode == "fwd":
                def step():
                    with torch.no_grad():
                        call()
            else:
                def step():
                    torch.autograd.grad(call(), inputs, grad_outputs=do)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            per: dict[str, list[float]] = {}
            launch: dict[str, dict] = {}
            for i in range(args.iterations):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    step()
                    torch.cuda.synchronize()
                agg: dict[str, float] = {}
                for e in prof.key_averages():
                    if e.self_device_time_total > 0:
                        agg[e.key[:60]] = agg.get(e.key[:60], 0.0) + e.self_device_time_total / 1e3
                for name, t in agg.items():
                    per.setdefault(name, []).append(t)
                if i == 0:
                    with tempfile.NamedTemporaryFile(suffix=".json") as f:
                        prof.export_chrome_trace(f.name)
                        events = json.load(open(f.name)).get("traceEvents", [])
                    for ev in events:
                        if ev.get("cat") != "kernel":
                            continue
                        a = ev.get("args", {})
                        launch.setdefault(ev["name"][:60], dict(
                            grid=a.get("grid"), block=a.get("block"), regs=a.get("registers per thread"),
                            smem=a.get("shared memory"), occ=a.get("est. achieved occupancy %"),
                            blocks_per_sm=a.get("blocks per SM")))
            med = {k: round(statistics.median(v), 4) for k, v in sorted(per.items(), key=lambda kv: -max(kv[1]))}
            row = dict(kind=args.kind, shape=list(shape), doc_len=args.doc_len, mode=mode, gates=args.gates,
                       total_ms=round(sum(med.values()), 4), kernels=dict(list(med.items())[:12]),
                       launch={k: launch[k] for k in list(med)[:12] if k in launch})
            if mode == "fwd":
                row["cp"] = cp_split(shape, args.doc_len)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
