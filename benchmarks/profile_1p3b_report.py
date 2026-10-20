# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tables of benchmarks/profile_1p3b.md from the JSONs of profile_1p3b.py and profile_1p3b_flame.py.

    python benchmarks/profile_1p3b_report.py --ours results/profile_1p3b_*.json \
        --flame results/profile_1p3b_flame_*.json \
        [--flame-baselines results/flame-*-1gpu*.json] \
        [--ours-e2e <bench_e2e_1p3b.py JSONs>] --json results/profile_1p3b.json > tables.md

Per architecture: ms/step and % of the wall step time by category (forward, recomputed forward,
backward), the idle time (wall minus the union of kernel intervals), and the Amdahl ceiling of each
category: the end-to-end speedup if that category ran 2x faster, and if it took no time. With
``--flame-baselines``: ours against both of flame's 1-GPU modes (full activation checkpointing without
torch.compile, and the fastest valid mode, e.g. torch.compile without checkpointing), and the bf16 MFU
that a 2x or 3x step over ours and over flame's strongest mode needs. With ``--ours-e2e``: the measured
no-remat + gradient-accumulation lever against full remat on the release harness.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# fine category -> report row
_ROWS = [
    ("GEMM: in/q/k/v projections", "GEMM in/q/k/v projections"),
    ("GEMM: gate/beta/dt/LoRA projections", "GEMM gate/beta/dt/LoRA projections"),
    ("GEMM: output-gate projection", "GEMM output-gate projection"),
    ("GEMM: out projection", "GEMM out projection"),
    ("GEMM: MLP up/gate", "GEMM MLP up/gate"),
    ("GEMM: MLP down", "GEMM MLP down"),
    ("GEMM: LM head (cross entropy)", "GEMM LM head"),
    ("GEMM: LM head", "GEMM LM head"),
    ("recurrent kernels (Mosaic GPU)", "recurrent kernels"),
    ("recurrent kernels (FLA)", "recurrent kernels"),
    ("softmax attention (Mosaic GPU)", "softmax attention"),
    ("softmax attention (cuDNN/XLA)", "softmax attention"),
    ("softmax attention", "softmax attention"),
    ("cross entropy (non-GEMM)", "cross entropy (non-GEMM)"),
    ("cross entropy (Mosaic GPU kernel)", "cross entropy (non-GEMM)"),
    ("norms", "norms"),
    ("short conv", "short conv"),
    ("mixer glue (gates, activations, layout)", "mixer glue (gates, activations)"),
    ("MLP activation (SwiGLU etc.)", "MLP activation"),
    ("rotary", "rotary"),
    ("l2norm", "l2norm"),
    ("cumsums", "cumsums"),
    ("layout copies/transposes", "layout copies/transposes"),
    ("memcpy/memset", "layout copies/transposes"),
    ("embedding/residual/final norm", "embedding/residual"),
    ("projection fusions (casts, bias, small dots)", "projection fusions (casts, bias, small dots)"),
    ("optimizer update", "optimizer + grad clip"),
    ("grad norm/clip", "optimizer + grad clip"),
    ("step glue (loss, grad accumulate, clip)", "optimizer + grad clip"),
    ("elementwise/reduction (aten, triton)", "other elementwise"),
    ("FSDP casts/copies", "other elementwise"),
]
_ROW_OF = dict(_ROWS)
_ORDER = list(dict.fromkeys(r for _, r in _ROWS)) + ["other GEMM", "other elementwise", "idle (gaps, launch, host)"]


def rows_of(profile: dict, wall_ms: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for cat, v in profile["categories_ms_per_step"].items():
        row = _ROW_OF.get(cat) or ("other GEMM" if cat.startswith("GEMM") else "other elementwise")
        for phase, x in v.items():
            phase = {"recompute": "remat", "opt": "fwd"}.get(phase, phase)
            out[row][phase] += x
    out["idle (gaps, launch, host)"]["total"] = max(0.0, wall_ms - profile["busy_ms_per_step"])
    return {k: dict(v) for k, v in out.items()}


def amdahl(wall: float, t: float, factor: float) -> float:
    rest = wall - t + t / factor if factor != float("inf") else wall - t
    return wall / rest if rest > 0 else float("inf")


def table(name: str, rows: dict, wall: float, kernel_sum: float, busy: float) -> list[str]:
    lines = [f"| {name} | ms/step | % of wall | fwd | remat fwd | bwd | 2x faster | free |",
             "|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: -rows[r].get("total", 0.0)):
        v = rows[row]
        t = v.get("total", 0.0)
        if t < 0.05 * wall / 100:
            continue
        lines.append(f"| {row} | {t:.1f} | {100 * t / wall:.1f}% | {v.get('fwd', 0):.1f} | {v.get('remat', 0):.1f} | "
                     f"{v.get('bwd', 0):.1f} | {amdahl(wall, t, 2):.3f}x | {amdahl(wall, t, float('inf')):.3f}x |")
    gemm = sum(v.get("total", 0) for r, v in rows.items() if r.startswith("GEMM") or r == "other GEMM")
    remat = sum(v.get("remat", 0) for v in rows.values())
    lines.append(f"| **all GEMMs** | {gemm:.1f} | {100 * gemm / wall:.1f}% | | | | {amdahl(wall, gemm, 2):.3f}x | "
                 f"{amdahl(wall, gemm, float('inf')):.3f}x |")
    lines.append(f"| **all recomputed forward (remat)** | {remat:.1f} | {100 * remat / wall:.1f}% | | | | "
                 f"{amdahl(wall, remat, 2):.3f}x | {amdahl(wall, remat, float('inf')):.3f}x |")
    lines.append(f"| wall / busy / kernel sum | {wall:.1f} / {busy:.1f} / {kernel_sum:.1f} | | | | | | |")
    return lines


PEAK_BF16 = 989e12  # H100 SXM dense bf16 FLOP/s: the MFU denominator of profile_1p3b.py
TOKENS_PER_STEP = 65536
_LOSS_FLAG = 0.1  # a flame mode whose last loss differs from the full-AC run by more is not a valid baseline


def _step_ms(d: dict) -> float:
    return 1000.0 * (d.get("step_time_median_s") or d["step_time_mean_s"])


def flame_modes(paths: list[str]) -> dict[str, dict]:
    """Per architecture: flame's full-AC/no-compile run, its no-AC runs, and the fastest valid mode.

    Inputs are the 1-GPU per-run JSONs of ``bench_e2e_flame.py``. A no-AC run whose last loss differs
    from the full-AC run's by more than ``_LOSS_FLAG`` (RWKV-7 under torch.compile) is kept but flagged
    and never chosen as the strongest mode.
    """
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        d = json.loads(Path(path).read_text())
        if d.get("devices") != 1 or d.get("tokens_per_step") != TOKENS_PER_STEP:
            continue
        compiled = bool(d.get("torch_compile"))
        mode = (f"AC {d.get('activation_checkpointing')}, {'compile' if compiled else 'no compile'}, "
                f"accum {d.get('grad_accum')}")
        runs[d["arch"]].append({"mode": mode, "ms": _step_ms(d), "tok_per_s": d.get("tokens_per_sec"),
                                "ac": d.get("activation_checkpointing"), "compile": compiled,
                                "grad_accum": d.get("grad_accum"), "document_masking": d.get("document_masking"),
                                "loss_last": d.get("loss_last"),
                                "peak_gib": d.get("peak_memory_allocated_gib_max_gpu")
                                or d.get("peak_memory_allocated_gib_max_gpu_incl_warmup"),
                                "training_flops_per_token": ((d.get("jla_flops") or {}).get("flops_per_token")
                                                             or {}).get("training"),
                                "source": Path(path).name})
    out = {}
    for arch, rs in runs.items():
        full = next((r for r in rs if r["ac"] == "full" and not r["compile"]), None)
        for r in rs:
            r["valid"] = not (full is not None and r is not full and r["loss_last"] is not None
                              and abs(r["loss_last"] - full["loss_last"]) > _LOSS_FLAG)
        valid = [r for r in rs if r["valid"]]
        out[arch] = {"full_ac": full, "runs": rs, "strongest": min(valid, key=lambda r: r["ms"]) if valid else None}
    return out


def _mfu(flops_per_step: float, ms: float) -> float:
    return 100.0 * flops_per_step / (ms / 1000.0) / PEAK_BF16


def flame_tables(ours: dict, modes: dict, e2e: dict) -> tuple[list[str], dict]:
    """Ours vs both flame modes, and the bf16 MFU that 2x / 3x over ours and over flame's strongest need."""
    lines = ["| architecture | ours: ms/step (tok/s) | flame full AC, no compile: ms | ours vs it "
             "| flame strongest valid mode | its ms (tok/s) | ours vs it |",
             "|---|---|---|---|---|---|---|"]
    targets = ["| architecture | training TFLOP/step | ms at 100% bf16 | ours ms (MFU) | 2x over ours: ms (MFU) "
               "| 3x over ours: ms (MFU) | flame strongest ms (MFU) | 2x over it: ms (MFU) | 3x over it: ms (MFU) |",
               "|---|---|---|---|---|---|---|---|---|"]
    out: dict = {}
    for arch in sorted(set(ours) | set(modes) | set(e2e)):
        o, m = ours.get(arch), modes.get(arch) or {}
        if o and o.get("tokens_per_step", TOKENS_PER_STEP) != TOKENS_PER_STEP:  # a partial-batch profile
            o = None
        ours_ms = o["wall_ms"] if o else (e2e.get(arch) or {}).get("full16", {}).get("ms")
        ours_tps = o["tok_per_s"] if o else (e2e.get(arch) or {}).get("full16", {}).get("tok_per_s")
        flops = (o or {}).get("training_flops_per_token") or next(
            (r["training_flops_per_token"] for r in m.get("runs", []) if r["training_flops_per_token"]), None)
        full, best = m.get("full_ac"), m.get("strongest")
        rec = {"ours_ms": ours_ms, "flame_full_ac": full, "flame_strongest": best,
               "flame_runs": m.get("runs", []), "training_flops_per_token": flops}
        if ours_ms is None and best is None:
            continue
        fmt_ours = f"{ours_ms:.0f} ({ours_tps:,.0f})" if ours_ms else "–"
        fmt_full = f"{full['ms']:.0f}" if full else "–"
        r_full = f"{full['ms'] / ours_ms:.2f}x" if full and ours_ms else "–"
        if best:
            nodoc = "" if best["document_masking"] else ", no doc masking"
            fmt_mode, fmt_best = best["mode"] + nodoc, f"{best['ms']:.0f} ({best['tok_per_s']:,.0f})"
        else:
            fmt_mode = fmt_best = "–"
        r_best = f"{best['ms'] / ours_ms:.2f}x" if best and ours_ms else "–"
        flagged = [r["mode"] for r in m.get("runs", []) if not r["valid"]]
        if flagged:
            fmt_mode += f" ({'; '.join(flagged)}: loss off, not used)"
        lines.append(f"| {arch} | {fmt_ours} | {fmt_full} | {r_full} | {fmt_mode} | {fmt_best} | {r_best} |")
        if flops:
            fps = flops * TOKENS_PER_STEP
            peak_ms = 1000.0 * fps / PEAK_BF16
            cell = lambda ms: f"{ms:.0f} ({_mfu(fps, ms):.0f}%)" if ms else "–"  # noqa: E731
            need = {}
            for label, base in (("ours", ours_ms), ("flame_strongest", best and best["ms"])):
                for factor in (2, 3) if base else ():
                    need[f"{factor}x_over_{label}"] = _mfu(fps, base / factor)
            rec.update(flops_per_step=fps, ms_at_bf16_peak=peak_ms, mfu_needed=need)
            targets.append(f"| {arch} | {fps / 1e12:.0f} | {peak_ms:.0f} | {cell(ours_ms)} | "
                           f"{cell(ours_ms and ours_ms / 2)} | {cell(ours_ms and ours_ms / 3)} | "
                           f"{cell(best and best['ms'])} | {cell(best and best['ms'] / 2)} | "
                           f"{cell(best and best['ms'] / 3)} |")
        out[arch] = rec
    return lines + [""] + targets, out


def e2e_runs(paths: list[str]) -> dict[str, dict]:
    """bench_e2e_1p3b.py JSONs: per architecture, the full-remat 16 x 1 run and the no-remat 4 x 4 run."""
    out: dict[str, dict] = defaultdict(dict)
    for path in paths:
        d = json.loads(Path(path).read_text())
        if d.get("status") != "ok":
            continue
        no_remat = any(o.replace('"', "").replace("'", "") == "model.remat=none" for o in d.get("overrides", []))
        key = "noac4" if no_remat else "full16"
        out[d["arch"]][key] = {"ms": _step_ms(d), "tok_per_s": d.get("tokens_per_sec"),
                               "micro_batch": d.get("micro_batch_per_replica"), "grad_accum": d.get("grad_accum"),
                               "peak_gib": d.get("peak_memory_gib_max_device"), "loss_last": d.get("loss_last"),
                               "mfu_percent": d.get("mfu_percent"), "gpus_before": d.get("gpus_before"),
                               "source": Path(path).name}
    return dict(out)


def e2e_table(e2e: dict, modes: dict) -> list[str]:
    lines = ["| architecture | full remat: ms (tok/s, peak GiB, micro-batch x accum) | no remat: ms (tok/s, peak GiB, micro-batch x accum) "
             "| speedup | loss at step 11 (full / none) | flame strongest ms | no-remat ours vs it |",
             "|---|---|---|---|---|---|---|"]
    def cell(x):
        return (f"{x['ms']:.0f} ({x['tok_per_s']:,.0f}, {x['peak_gib']:.1f}, {x['micro_batch']} x {x['grad_accum']})"
                if x else "–")

    for arch, r in sorted(e2e.items()):
        a, b = r.get("full16"), r.get("noac4")
        best = (modes.get(arch) or {}).get("strongest")
        speedup = f"{a['ms'] / b['ms']:.2f}x" if a and b else "–"
        loss = f"{a['loss_last']:.3f} / {b['loss_last']:.3f}" if a and b else "–"
        vs = (f"{best['ms']:.0f} | {best['ms'] / b['ms']:.2f}x") if best and b else "– | –"
        lines.append(f"| {arch} | {cell(a)} | {cell(b)} | {speedup} | {loss} | {vs} |")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ours", nargs="*", default=[])
    p.add_argument("--flame", nargs="*", default=[])
    p.add_argument("--flame-baselines", nargs="*", default=[],
                   help="1-GPU per-run JSONs of bench_e2e_flame.py (both flame modes)")
    p.add_argument("--ours-e2e", nargs="*", default=[],
                   help="bench_e2e_1p3b.py JSONs: full remat 16 x 1 and no remat 4 x 4 accumulation")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    summary = {"ours": {}, "flame": {}}
    for path in args.ours:
        d = json.loads(Path(path).read_text())
        if "profile" not in d:
            continue
        wall = d["ms_per_step_median"]
        prof = d["profile"]
        rows = rows_of(prof, wall)
        summary["ours"][d["arch"]] = {"wall_ms": wall, "tok_per_s": d["tok_per_s"], "busy_ms": prof["busy_ms_per_step"],
                                      "kernel_ms": prof["kernel_ms_per_step"], "rows": rows,
                                      "mfu_percent": d.get("mfu_percent"), "compile_s": d.get("compile_s"),
                                      "peak_gib": d.get("peak_bytes_in_use", 0) / 2**30, "loss": d.get("loss"),
                                      "training_flops_per_token": (d.get("flops_per_token") or {}).get("training"),
                                      "tokens_per_step": d.get("micro_batch", 16) * d.get("seq_len", 4096)}
        tokens = summary["ours"][d["arch"]]["tokens_per_step"]
        part = "" if tokens == TOKENS_PER_STEP else f" for {tokens:,} tokens ({d.get('micro_batch')} x {d.get('seq_len')})"
        print(f"\n### {d['arch']} (ours): {wall:.1f} ms/step{part}, {d['tok_per_s']:,} tok/s, MFU {d.get('mfu_percent')}%\n")
        print("\n".join(table("category", rows, wall, prof["kernel_ms_per_step"], prof["busy_ms_per_step"])))
    for path in args.flame:
        d = json.loads(Path(path).read_text())
        prof = d["profile"]
        wall = d.get("wall_ms_median_excluding_profiled_step") or prof["span_ms_per_step"]
        rows = rows_of(prof, wall)
        arch = d["flame_result"]["arch"]
        tps = d["flame_result"].get("tokens_per_sec")
        summary["flame"][arch] = {"wall_ms": wall, "tok_per_s": tps, "busy_ms": prof["busy_ms_per_step"],
                                  "kernel_ms": prof["kernel_ms_per_step"], "rows": rows,
                                  "settings": {k: d["flame_result"].get(k) for k in (
                                      "document_masking", "activation_checkpointing", "torch_compile", "fla_backends",
                                      "attention_kernel", "grad_accum")},
                                  "fla_backend_dispatch": (d["flame_result"].get("fla_backend_dispatch") or {}).get("served")}
        print(f"\n### {arch} (flame): {wall:.1f} ms/step (host median outside the profiler window), "
              f"{tps:,.0f} tok/s over the timed steps\n")
        print("\n".join(table("category", rows, wall, prof["kernel_ms_per_step"], prof["busy_ms_per_step"])))
    print("\n### Summary\n")
    print("| architecture | trainer | ms/step | tok/s | busy ms | idle ms | GEMMs | recurrent kernels | softmax attention "
          "| other kernels and fusions | recomputed forward (part of the above) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for trainer in ("ours", "flame"):
        for arch, s in sorted(summary[trainer].items()):
            rows, wall = s["rows"], s["wall_ms"]
            gemm = sum(v.get("total", 0) for r, v in rows.items() if r.startswith("GEMM") or r == "other GEMM")
            rec = rows.get("recurrent kernels", {}).get("total", 0)
            att = rows.get("softmax attention", {}).get("total", 0)
            idle = rows.get("idle (gaps, launch, host)", {}).get("total", 0)
            other = wall - gemm - rec - att - idle
            remat = sum(v.get("remat", 0) for v in rows.values())
            pct = lambda x: f"{x:.0f} ({100 * x / wall:.0f}%)"  # noqa: E731
            print(f"| {arch} | {trainer} | {wall:.0f} | {s['tok_per_s']:,.0f} | {s['busy_ms']:.0f} | {idle:.0f} | {pct(gemm)} | "
                  f"{pct(rec)} | {pct(att)} | {pct(other)} | {pct(remat)} |")
    both = sorted(set(summary["ours"]) & set(summary["flame"]))
    for arch in both:
        o, f = summary["ours"][arch], summary["flame"][arch]
        print(f"\n### {arch}: ours vs flame, ms/step by category\n")
        print("| category | ours | flame | flame - ours |")
        print("|---|---|---|---|")
        cats = sorted(set(o["rows"]) | set(f["rows"]),
                      key=lambda r: -max(o["rows"].get(r, {}).get("total", 0), f["rows"].get(r, {}).get("total", 0)))
        for r in cats:
            a, b = o["rows"].get(r, {}).get("total", 0), f["rows"].get(r, {}).get("total", 0)
            if max(a, b) < 1:
                continue
            print(f"| {r} | {a:.1f} | {b:.1f} | {b - a:+.1f} |")
        print(f"| **step (wall)** | {o['wall_ms']:.1f} | {f['wall_ms']:.1f} | {f['wall_ms'] - o['wall_ms']:+.1f} "
              f"({f['wall_ms'] / o['wall_ms']:.2f}x) |")
    modes = flame_modes(args.flame_baselines)
    e2e = e2e_runs(args.ours_e2e)
    if modes or e2e:
        lines, rec = flame_tables(summary["ours"], modes, e2e)
        print("\n### Ours vs flame's two modes, and the bf16 MFU that 2x / 3x need\n")
        print("\n".join(lines))
        summary["versus_flame_modes"] = rec
    if e2e:
        print("\n### No remat + gradient accumulation (ours, release harness)\n")
        print("\n".join(e2e_table(e2e, modes)))
        summary["ours_e2e_remat"] = e2e
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
