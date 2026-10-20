# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Markdown tables of benchmarks/profile_1p3b.md from the speed-of-light result JSONs.

    python benchmarks/speed_of_light_report.py --sol sol.json --gemm gemm_xla.json --gemm gemm_mosaic.json \
        --torch torch.json
    python benchmarks/speed_of_light_report.py --assemble results/speed_of_light.json \
        --part clock=clock.json --part sol=sol.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEC = 989.4


def _f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def gemm_table(gemm_jsons: list[str], torch_json: str | None) -> str:
    best: dict[tuple, dict] = {}
    for path in gemm_jsons + ([torch_json] if torch_json else []):
        data = json.loads(Path(path).read_text())
        rows = data.get("rows", []) + data.get("bf16", []) + data.get("fp8", [])
        for r in rows:
            if "tflops" not in r:
                continue
            impl = r["impl"]
            if impl == "xla":
                impl = "XLA" + (" K-major" if r.get("variant") == "both K-major" else "")
                if r.get("out_dtype") == "float32":
                    continue
            elif impl.startswith("mosaic"):
                impl = "Mosaic GPU"
            elif "scaled_mm" in impl:
                impl = "torch fast-accum" if r.get("use_fast_accum") else "torch"
            else:
                impl = "torch"
            key = (r["problem"], r["layout"])
            cell = best.setdefault(key, {})
            cell[(r["dtype"], impl)] = max(cell.get((r["dtype"], impl), 0.0), float(r["tflops"]))
    cols = [("bf16", "XLA"), ("bf16", "Mosaic GPU"), ("bf16", "torch"), ("fp8", "XLA"), ("fp8", "XLA K-major"),
            ("fp8", "torch"), ("fp8", "torch fast-accum")]
    out = ["| GEMM (M = 65,536 tokens) | layout | bf16 XLA | bf16 Mosaic GPU | bf16 torch | best bf16 (% of 989.4) | "
           "FP8 XLA as spelled | FP8 XLA K-major | FP8 torch | FP8 torch fast-accum | best FP8 (% of 1,978.9) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (problem, layout), cell in best.items():
        b16 = max([v for (d, _), v in cell.items() if d == "bf16"] or [0.0])
        b8 = max([v for (d, _), v in cell.items() if d == "fp8"] or [0.0])
        vals = [_f(cell.get(c)) for c in cols]
        out.append(f"| {problem} | {layout} | {vals[0]} | {vals[1]} | {vals[2]} | **{_f(b16)}** ({100 * b16 / SPEC:.0f}%) | "
                   f"{vals[3]} | {vals[4]} | {vals[5]} | {vals[6]} | **{_f(b8)}** ({100 * b8 / (2 * SPEC):.0f}%) |")
    return "\n".join(out)


def sol_tables(sol_json: str) -> str:
    runs = json.loads(Path(sol_json).read_text())
    out = ["| preset | measured step (MFU) | SOL as compiled | SOL, no recompute | ideal, burst ceilings | "
           "ideal, power-capped | ideal, FP8 GEMMs, burst (bf16-eq. MFU / FP8-peak util.) | "
           "ideal, FP8 GEMMs, power-capped (bf16-eq. MFU / FP8-peak util.) | tensor work at the sustained rate |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        out.append(
            f"| `{r['arch']}` | {_f(r['wall_ms'], 0)} ms ({r['measured_mfu_percent']}%) | "
            f"{_f(r['sol_as_compiled_ms'], 0)} ms ({r['sol_as_compiled_mfu_percent']}%) | "
            f"{_f(r['sol_no_remat_ms'], 0)} ms ({r['sol_no_remat_mfu_percent']}%) | "
            f"{_f(r['sol_ideal_ms'], 0)} ms ({r['sol_ideal_mfu_percent']}%) | "
            f"{_f(r['sol_ideal_power_capped_ms'], 0)} ms ({r['sol_ideal_power_capped_mfu_percent']}%) | "
            f"{_f(r['sol_ideal_fp8_ms'], 0)} ms ({r['sol_ideal_fp8_bf16_equivalent_mfu_percent']}% / "
            f"{r['sol_ideal_fp8_fp8_peak_utilization_percent']}%) | "
            f"{_f(r['sol_ideal_fp8_power_capped_ms'], 0)} ms "
            f"({r['sol_ideal_fp8_power_capped_bf16_equivalent_mfu_percent']}% / "
            f"{r['sol_ideal_fp8_power_capped_fp8_peak_utilization_percent']}%) | "
            f"{_f(r['tensor_work_bound_power_capped_ms'], 0)} ms ({r['tensor_work_bound_power_capped_mfu_percent']}%) |")
    out.append("")
    for r in runs:
        out.append(f"#### `{r['arch']}`: {_f(r['wall_ms'], 0)} ms measured, {_f(r['busy_ms'], 0)} ms GPU busy, "
                   f"GEMMs at their ceilings {_f(r['gemm_only_at_ceiling_ms'], 0)} ms\n")
        out.append("| category [phase] | measured ms | SOL ms | gap ms | kind |")
        out.append("|---|---|---|---|---|")
        for g in r["gaps"]:
            if g["gap_ms"] < 2.0:
                continue
            out.append(f"| {g['item']} | {_f(g['measured_ms'])} | {_f(g['sol_ms'])} | **{_f(g['gap_ms'])}** | {g['kind']} |")
        out.append("")
        out.append("| top ops by gap | phase | measured ms | SOL ms | GB/step | achieved | source |")
        out.append("|---|---|---|---|---|---|---|")
        for t in r["top_ops_by_gap"][:10]:
            ach = (f"{t['achieved_tflops']} TFLOP/s" if t.get("achieved_tflops") else f"{t.get('achieved_gbps')} GB/s")
            name = t.get("kernel") or t["op"]
            out.append(f"| {t['category']}: `{name[:48]}` | {t['phase']} | {_f(t['ms'])} | {_f(t['sol_ms'])} | "
                       f"{_f(t['bytes'] / 1e9)} | {ach} | {t.get('site', '')[:48]} |")
        out.append("")
    return "\n".join(out)


def roadmap_items(run: dict, ops: list[dict]) -> dict[str, float]:
    """ms per step of each roadmap item for one SOL run, from its per-op list (``ops_over_0p1_ms``)."""
    def gap(o):
        return o["ms"] - o["sol_ms"]

    def attn(o):
        return "mosaic_gpu/attention.py" in o.get("site", "")

    def ce(o):
        return "cross_entropy.py" in o.get("site", "")

    def conv(o):
        return "layers/conv.py" in o.get("site", "")

    glue = {"elementwise/reduction fusions", "casts (convert-only fusions)",
            "layout copies (transpose/copy/pad/concat fusions)", "memcpy/memset/other"}
    live = [o for o in ops if o["phase"] != "remat"]
    ce_fwd = {tuple(o["gemm"]) for o in live if o["category"] == "GEMM" and ce(o) and o["phase"] == "fwd" and o["gemm"]}
    ce_recompute = [o for o in live if o["category"] == "GEMM" and ce(o) and o["phase"] == "bwd" and o["gemm"]
                    and tuple(o["gemm"]) in ce_fwd]
    slow_remat = [o for o in ops if o["phase"] == "remat" and o["category"] == "elementwise/reduction fusions"
                  and o["gbytes"] and o["gbytes"] / (o["ms"] / 1e3) < 400]
    idle = next((g["gap_ms"] for g in run["gaps"] if g["item"].startswith("idle")), 0.0)
    items = {
        "recomputed forward (full remat), all of it": sum(o["ms"] for o in ops if o["phase"] == "remat"),
        "  of which: conv + SiLU + L2-norm remat fusion below 400 GB/s, to bandwidth": sum(gap(o) for o in slow_remat),
        "softmax attention kernels (fwd + bwd) to their ceiling": sum(gap(o) for o in live
                                                                     if o["category"] == "Mosaic GPU kernel" and attn(o)),
        "recurrent kernels (fwd + bwd) to their ceiling": sum(gap(o) for o in live
                                                             if o["category"] == "Mosaic GPU kernel" and not attn(o)),
        "short-conv backward reductions to bandwidth": sum(gap(o) for o in live if o["category"] in glue and conv(o)
                                                           and o["phase"] == "bwd"),
        "other elementwise passes, casts, layout copies (fwd + bwd), fused away": sum(
            o["ms"] for o in live if o["category"] in glue and not ce(o)) - sum(
            gap(o) for o in live if o["category"] in glue and conv(o) and o["phase"] == "bwd"),
        "LM head: logits recomputed in the CE backward": sum(o["ms"] for o in ce_recompute),
        "LM head: CE GEMMs to their ceiling + CE elementwise passes": sum(
            gap(o) for o in live if o["category"] == "GEMM" and ce(o) and o not in ce_recompute) + sum(
            o["ms"] for o in live if o["category"] in glue and ce(o)),
        "other GEMMs to their burst ceiling (fwd + bwd)": sum(gap(o) for o in live if o["category"] == "GEMM" and not ce(o)),
        "optimizer to bandwidth": sum(gap(o) for o in live if o["category"].startswith("optimizer")),
        "idle GPU (launch gaps, host)": idle,
        "FP8 GEMMs at their K-major ceilings (ideal bf16 -> ideal FP8)": run["sol_ideal_ms"] - run["sol_ideal_fp8_ms"],
    }
    return {k: round(v, 1) for k, v in items.items()}


def roadmap_table(sol_json: str, ops_json: str | None = None) -> str:
    runs = json.loads(Path(sol_json).read_text())
    ops_by_arch = json.loads(Path(ops_json).read_text()) if ops_json else {}
    rows = {}
    for r in runs:
        ops = r.get("ops_over_0p1_ms") or ops_by_arch.get(r["arch"], [])
        rows[r["arch"]] = roadmap_items(r, ops)
    archs = list(rows)
    out = ["| item (ms per step) | " + " | ".join(f"`{a}`" for a in archs) + " |",
           "|---|" + "---|" * len(archs)]
    for item in next(iter(rows.values())):
        out.append(f"| {item} | " + " | ".join(_f(rows[a][item], 0) for a in archs) + " |")
    out.append("| measured step | " + " | ".join(_f(r["wall_ms"], 0) for r in runs) + " |")
    out.append("| ideal step (bf16, burst) | " + " | ".join(_f(r["sol_ideal_ms"], 0) for r in runs) + " |")
    return "\n".join(out)


def assemble(out: str, parts: list[str]) -> None:
    """One result file from ``name=path`` parts; the SOL runs keep their top ops, the full per-op lists
    go to ``<out stem>_ops.json``."""
    result: dict = {"page": "benchmarks/profile_1p3b.md", "provisional": True}
    ops: dict = {}
    for part in parts:
        name, _, path = part.partition("=")
        data = json.loads(Path(path).read_text())
        if name.startswith("sol"):
            for i, run in enumerate(data):
                key = run["arch"] if name == "sol" else f"{name}/{i}:{run['arch']}"
                ops[key] = run.pop("ops_over_0p1_ms", [])
                run.pop("ceilings", None)
        result[name] = data
    Path(out).write_text(json.dumps(result, indent=1, default=float))
    if ops:
        Path(out).with_name(Path(out).stem + "_ops.json").write_text(json.dumps(ops, default=float))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sol", default=None)
    p.add_argument("--gemm", action="append", default=[])
    p.add_argument("--torch", default=None)
    p.add_argument("--assemble", default=None, help="write one result JSON from --part name=path entries")
    p.add_argument("--part", action="append", default=[])
    args = p.parse_args(argv)
    if args.assemble:
        assemble(args.assemble, args.part)
        return 0
    if args.gemm or args.torch:
        print(gemm_table(args.gemm, args.torch))
        print()
    if args.sol:
        print(sol_tables(args.sol))
        print()
        print(roadmap_table(args.sol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
