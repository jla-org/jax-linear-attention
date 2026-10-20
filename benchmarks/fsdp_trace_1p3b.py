# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the per-GPU time of an FSDP training step goes (benchmarks/multi_gpu.md).

    # a profiled run of the 1.3B harness (profiler window = the last two steps; timing from the steps before)
    python benchmarks/bench_e2e_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B --dp 1 --fsdp 8 \
        --set train.profile_start_step=18 --set train.profile_steps=2 --output-dir RUN8 --json run8.json
    # the same on one GPU (--dp 1 --fsdp 1, one visible GPU), then
    python benchmarks/fsdp_trace_1p3b.py --profile RUN8/profile --result run8.json \
        --single-profile RUN1/profile --single-result run1.json --json out.json

Reads the ``jax.profiler`` trace (``*.xplane.pb``) of each run and, per GPU over the profiled steps:

- **busy / idle**: the union of every kernel interval on the GPU (all streams) against the window from the
  first kernel to the last, and the five longest idle stretches (``largest_idle_gaps``: ms, not per step, and
  where in the window they start): with the harness's ``train.log_every = 1`` the longest is the host
  dispatching the next step while the GPU waits;
- **NCCL**: the union of the NCCL kernels per collective (all-gather, reduce-scatter, all-reduce, ...), and
  how much of it no compute kernel overlaps (**exposed**). An NCCL kernel runs until every rank has joined,
  so its exposed time includes waiting for the slowest rank, not only link time;
- **compute kernels by category** (cuBLAS GEMMs, custom Mosaic GPU / Pallas kernels, XLA fusions, copies),
  summed kernel time, to compare the per-shard kernels of the multi-GPU step with the same model on one GPU.

Step times come from the harness's result JSON (``per_step``): the mean over the timed steps before the
profiler window (``--first`` .. ``--last``, default 6 .. 18), so the profiler's own cost stays out of the
timing. Every figure is per optimizer step (divided by the number of profiled steps).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fsdp_step_hlo import _intersection, _length, _union  # noqa: E402  (same directory)

_COLLECTIVE = re.compile(r"(AllGather|ReduceScatter|AllReduce|AllToAll|SendRecv|Broadcast|Reduce)", re.I)
_GEMM = re.compile(r"nvjet|gemm|cutlass|xmma|cublas|sm90_|sm80_", re.I)
_XLA_FUSION = re.compile(r"fusion|^loop_|^input_|^wrapped_|^triton_|_fusion", re.I)
_LAYOUT = re.compile(r"^copy|transpose|^bitcast", re.I)


def kernel_category(name: str) -> str:
    """Coarse category of one GPU kernel from its name."""
    low = name.lower()
    if "nccl" in low:
        m = _COLLECTIVE.search(name)
        return f"nccl {m.group(1).lower() if m else 'other'}"
    if name.startswith(("Memcpy", "Memset", "memcpy", "memset")):
        return "memcpy/memset"
    if _GEMM.search(name):
        return "GEMM (cuBLAS)"
    if _LAYOUT.search(name):
        return "layout copies/transposes"
    if _XLA_FUSION.search(name):
        return "XLA fusions"
    return "custom kernels (Mosaic GPU / Pallas / cuDNN)"


def _xplane(profile_dir: str):
    from jax.profiler import ProfileData

    files = sorted(Path(profile_dir).rglob("*.xplane.pb"))
    if not files:
        raise SystemExit(f"no *.xplane.pb under {profile_dir}")
    return ProfileData.from_file(str(files[-1]))


def device_breakdown(profile_dir: str, steps: int) -> list[dict]:
    """Per GPU plane of the trace: window, busy, idle, NCCL per collective (total and exposed), compute
    kernel time per category; every value in ms per step."""
    data = _xplane(profile_dir)
    out = []
    for plane in data.planes:
        if not plane.name.startswith("/device:GPU:"):
            continue
        comm: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        comp: list[tuple[int, int]] = []
        other: list[tuple[int, int]] = []
        per_cat: dict[str, int] = collections.defaultdict(int)
        per_kernel: dict[str, int] = collections.defaultdict(int)
        counts: dict[str, int] = collections.defaultdict(int)
        streams = set()
        for line in plane.lines:
            if not line.name.startswith("Stream"):
                continue  # "XLA Modules" / "XLA Ops" / "Steps" lines repeat the stream events
            for event in line.events:
                start = int(event.start_ns)
                interval = (start, start + int(event.duration_ns))
                cat = kernel_category(event.name)
                per_cat[cat] += interval[1] - interval[0]
                counts[cat] += 1
                streams.add(line.name)
                if cat.startswith("nccl"):
                    comm[cat].append(interval)
                elif cat == "memcpy/memset":
                    other.append(interval)
                else:
                    comp.append(interval)
                    per_kernel[re.sub(r"[._]\d+$", "", event.name)[:90]] += interval[1] - interval[0]
        if not comp and not comm:
            continue
        comp_u = _union(comp)
        comm_all = _union([iv for ivs in comm.values() for iv in ivs])
        everything = _union(comp_u + comm_all + _union(other))
        window = everything[-1][1] - everything[0][0]
        gaps = sorted(((b[0] - a[1], a[1] - everything[0][0]) for a, b in zip(everything, everything[1:])), reverse=True)
        overlapped = _intersection(comm_all, comp_u)
        nccl = {}
        for cat, ivs in sorted(comm.items()):
            u = _union(ivs)
            nccl[cat.removeprefix("nccl ")] = {
                "kernels_per_step": counts[cat] / steps,
                "ms": _length(u) / 1e6 / steps,
                "exposed_ms": (_length(u) - _intersection(u, comp_u)) / 1e6 / steps,
            }
        out.append({
            "device": plane.name.removeprefix("/device:"),
            "streams": len(streams),
            "window_ms": window / 1e6 / steps,
            "busy_ms": _length(everything) / 1e6 / steps,
            "idle_ms": (window - _length(everything)) / 1e6 / steps,
            "largest_idle_gaps": [{"ms": g / 1e6, "at_ms": at / 1e6} for g, at in gaps[:5]],
            "compute_busy_ms": _length(comp_u) / 1e6 / steps,
            "nccl_ms": _length(comm_all) / 1e6 / steps,
            "nccl_exposed_ms": (_length(comm_all) - overlapped) / 1e6 / steps,
            "nccl": nccl,
            "kernel_ms_by_category": {k: v / 1e6 / steps for k, v in sorted(per_cat.items(), key=lambda kv: -kv[1])},
            "top_compute_kernels": [{"kernel": k, "ms": v / 1e6 / steps}
                                    for k, v in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:25]],
        })
    return out


def step_timing(result_json: str, first: int, last: int) -> dict:
    """Mean step time over steps ``first..last`` of the harness's result, with tokens/s (per GPU) and MFU."""
    res = json.loads(Path(result_json).read_text())
    rows = [r for r in res["per_step"] if first <= r["step"] <= last]
    times = [float(r["step_time"]) for r in rows]
    mean = statistics.fmean(times)
    devices = int(res["devices"])
    tokens = int(res["tokens_per_step"])
    mfu = statistics.fmean(float(r["mfu_percent"]) * float(r["step_time"]) for r in rows) / mean
    return {"steps": f"{first}-{last}", "step_time_mean_s": mean, "step_time_median_s": statistics.median(times),
            "step_time_min_s": min(times), "step_time_max_s": max(times), "devices": devices,
            "tokens_per_sec": tokens / mean, "tokens_per_sec_per_gpu": tokens / mean / devices, "mfu_percent": mfu,
            "peak_memory_gib_max_device": res.get("peak_memory_gib_max_device"),
            "loss_last": res.get("loss_last"), "overrides": res.get("overrides"), "mesh": res.get("mesh")}


def _mean_over(devices: list[dict], key: str) -> float:
    return statistics.fmean(d[key] for d in devices)


def summarize(devices: list[dict]) -> dict:
    """Means (and min / max) over the GPUs of one run."""
    keys = ("window_ms", "busy_ms", "idle_ms", "compute_busy_ms", "nccl_ms", "nccl_exposed_ms")
    out = {k: {"mean": _mean_over(devices, k), "min": min(d[k] for d in devices), "max": max(d[k] for d in devices)}
           for k in keys}
    cats = sorted({c for d in devices for c in d["kernel_ms_by_category"]})
    out["kernel_ms_by_category"] = {c: statistics.fmean(d["kernel_ms_by_category"].get(c, 0.0) for d in devices)
                                    for c in cats}
    colls = sorted({c for d in devices for c in d["nccl"]})
    out["nccl"] = {c: {f: statistics.fmean(d["nccl"].get(c, {}).get(f, 0.0) for d in devices)
                       for f in ("kernels_per_step", "ms", "exposed_ms")} for c in colls}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", required=True, help="the multi-GPU run's profile directory (output_dir/profile)")
    p.add_argument("--result", required=True, help="the multi-GPU run's result JSON (bench_e2e_1p3b.py --json)")
    p.add_argument("--single-profile", default=None, help="the same config's one-GPU profile directory")
    p.add_argument("--single-result", default=None)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--first", type=int, default=6)
    p.add_argument("--last", type=int, default=18)
    p.add_argument("--json", required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    multi = device_breakdown(args.profile, args.profile_steps)
    report = {"timing": step_timing(args.result, args.first, args.last), "summary": summarize(multi), "devices": multi}
    if args.single_profile:
        single = device_breakdown(args.single_profile, args.profile_steps)
        report["single"] = {"summary": summarize(single), "devices": single}
        if args.single_result:
            report["single"]["timing"] = step_timing(args.single_result, args.first, args.last)
        a, b = report["summary"]["kernel_ms_by_category"], report["single"]["summary"]["kernel_ms_by_category"]
        report["kernel_ms_by_category_multi_minus_single"] = {c: a.get(c, 0.0) - b.get(c, 0.0)
                                                             for c in sorted(set(a) | set(b))}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2))
    t, s = report["timing"], report["summary"]
    print(f"step {1e3 * t['step_time_mean_s']:.1f} ms ({t['tokens_per_sec_per_gpu']:,.0f} tok/s per GPU, "
          f"MFU {t['mfu_percent']:.1f}%); per GPU: window {s['window_ms']['mean']:.1f} ms, busy {s['busy_ms']['mean']:.1f}, "
          f"idle {s['idle_ms']['mean']:.1f}, compute {s['compute_busy_ms']['mean']:.1f}, NCCL {s['nccl_ms']['mean']:.1f} "
          f"(exposed {s['nccl_exposed_ms']['mean']:.1f}, max {s['nccl_exposed_ms']['max']:.1f}); longest idle gap "
          f"{statistics.fmean([d['largest_idle_gaps'][0]['ms'] for d in multi if d['largest_idle_gaps']] or [0.0]):.1f} ms")
    for c, v in s["nccl"].items():
        print(f"  nccl {c:14s} {v['kernels_per_step']:6.0f} kernels  {v['ms']:7.1f} ms  exposed {v['exposed_ms']:6.1f} ms")
    for c, v in s["kernel_ms_by_category"].items():
        line = f"  {c:46s} {v:8.1f} ms"
        if "kernel_ms_by_category_multi_minus_single" in report:
            line += f"   vs 1 GPU {report['kernel_ms_by_category_multi_minus_single'][c]:+7.1f} ms"
        print(line)
    if "single" in report:
        ss = report["single"]["summary"]
        print(f"1 GPU: window {ss['window_ms']['mean']:.1f} ms, busy {ss['busy_ms']['mean']:.1f}, idle {ss['idle_ms']['mean']:.1f}"
              + (f", step {1e3 * report['single']['timing']['step_time_mean_s']:.1f} ms" if "timing" in report["single"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
