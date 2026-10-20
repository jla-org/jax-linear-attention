# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the device time of one flame (FLA) 1.3B training step goes (benchmarks/profile_1p3b.md).

    python benchmarks/profile_1p3b_flame.py --runner /path/to/bench_e2e_flame.py --json out.json -- \
        --arch kda-1p3b --data /path/to/fineweb_edu_3B --micro-batch 16 --steps 12 --skip 5 --no-compile \
        --flame-src /path/to/flame --dump-dir /path/to/dump
    python benchmarks/profile_1p3b_flame.py --analyze-only /path/to/rank0_trace.json --json out.json

Runs in the flame environment, on one GPU, the runner of the flame end-to-end baseline
(``benchmarks/bench_e2e_flame.py`` of benchmarks/end_to_end_1p3b.md: flame's own ``flame.train.main``
with the jla harness's batches; pass its path with ``--runner``) in this process (rank 0 of 1). This script
drives that runner's internals (``_parse_args``, ``_worker``, ``_launch``, ``FLAME_DIR``, ``--from-raw``), so
the two must come from the same tree; it stops with a message when the runner lacks them. It runs with
torchtitan's profiler on (``--profiling.enable_profiling``, profile step 10:
6 wait, 3 warm-up and 1 active step, ``record_shapes=True``). Two things are added for the attribution,
nothing in the step changes: every ``nn.Linear`` call runs inside a ``record_function("jla_linear:<module
path>")`` range, and the model built by ``AutoModelForCausalLM.from_config`` gets those module paths.

Each GPU kernel of the active step is matched to its launch (CUPTI correlation id) and to the stack of
CPU ops around that launch. Phase: forward (launched outside the autograd engine), backward
(inside ``autograd::engine::evaluate_function``), recompute (inside the autograd engine but under a
``jla_linear`` range, i.e. a forward re-run by activation checkpointing, or an FLA forward kernel run
there), optimizer (under ``Optimizer.step``). A backward GEMM's role is the forward role with the same
weight shape (``K x N`` of the forward ``[M, K] x [K, N]``), split by forward time when several roles share it.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runner", help="bench_e2e_flame.py of the flame baseline tree")
    p.add_argument("--json", default=None)
    p.add_argument("--analyze-only", default=None, help="a torch.profiler chrome trace (rank0_trace.json)")
    p.add_argument("--tokens", type=int, default=65536, help="tokens per micro-step (GEMM M of the projections)")
    p.add_argument("--wall-ms", type=float, default=None, help="host step time for the idle column")
    p.add_argument("--model-config", default=None,
                   help="with --analyze-only: the flame model JSON, for the nn.Linear weight shapes (needs torch + FLA)")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the runner's arguments")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Trace analysis (plain Python)

_GEMM_OPS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul", "aten::linear", "aten::_scaled_mm")
_GEMM_KERNEL = re.compile(r"nvjet|gemm|cutlass|xmma|sm90_|sm80_|cublas|splitKreduce", re.I)
_RECURRENT = re.compile(r"chunk|fused_recurrent|wy_|_wy|solve_tril|kda|delta|gdn|ssd|state_passing|bmm_chunk|"
                        r"chunk_scan|chunk_state|mamba|rwkv|dplr|gla_|recompute_w_u|prepare_wy|selective|"
                        r"merge_16x16|_intra|_inter|_dqkwg|_dhu|_dv(?:_|\b)|fwd_h|bwd_h|tilelang|main_kernel", re.I)
# PyTorch's own kernels (aten elementwise functors such as binary_internal::MulFunctor, chunk_cat, cub, thrust)
# are never recurrent kernels, whatever words their names contain
_TORCH_KERNEL = re.compile(r"\bat::|\bc10::|at_cuda_detail|\bcub::|\bthrust::")


def _role_from_label(label: str) -> str:
    name = label.split(":", 1)[1] if ":" in label else label
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "lm_head":
        return "LM head"
    if leaf in ("o_proj", "out_proj"):
        return "out projection"
    if leaf in ("g_proj", "gate_proj") and ".mlp." not in name:
        return "output-gate projection"
    if ".mlp." in name or ".ffn." in name or leaf in ("up_proj", "down_proj", "key", "value"):
        if leaf in ("down_proj", "value"):
            return "MLP down"
        return "MLP up/gate"
    if leaf in ("q_proj", "k_proj", "v_proj", "r_proj", "in_proj", "qkv_proj", "x_proj", "w_proj"):
        return "in/q/k/v projections"
    return "gate/beta/dt/LoRA projections"


def _load(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def _dims(op: dict) -> list:
    return (op.get("args") or {}).get("Input Dims") or []


def _union_us(intervals):
    total, start, end = 0.0, None, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def analyze(trace_path: Path, tokens: int, wall_ms: float | None = None, top: int = 40, vocab: int = 128256,
            weight_roles: dict[str, list[str]] | None = None) -> dict:
    """``weight_roles``: ``"in x out" -> roles`` of the model's ``nn.Linear`` modules (``linear_shapes.json``)."""
    events = _load(trace_path)
    cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    runtime = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                runtime[corr] = e
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    # The CPU ops open at each launch: one sweep per thread over ops (sorted by start, outer first) and launches.
    by_tid: dict = defaultdict(list)
    for e in cpu:
        by_tid[(e.get("pid"), e.get("tid"))].append(e)
    launches: dict = defaultdict(list)
    for corr, rt in runtime.items():
        launches[(rt.get("pid"), rt.get("tid"))].append(rt)
    open_at: dict = {}
    for k, rts in launches.items():
        ops = sorted(by_tid.get(k, []), key=lambda e: (float(e["ts"]), -float(e.get("dur", 0))))
        rts.sort(key=lambda e: float(e["ts"]))
        stack: list = []
        i = 0
        for rt in rts:
            t = float(rt["ts"])
            while i < len(ops) and float(ops[i]["ts"]) <= t:
                op = ops[i]
                while stack and float(stack[-1]["ts"]) + float(stack[-1].get("dur", 0)) < float(op["ts"]):
                    stack.pop()
                stack.append(op)
                i += 1
            while stack and float(stack[-1]["ts"]) + float(stack[-1].get("dur", 0)) < t:
                stack.pop()
            open_at[(rt.get("args") or {}).get("correlation")] = list(stack)  # outermost first
    # torchtitan exports only the active step's events, so every kernel in the trace belongs to it
    rows = [(kev, open_at.get((kev.get("args") or {}).get("correlation"), [])) for kev in kernels]
    fwd_roles: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    classified = []
    for kev, anc in rows:
        name = str(kev.get("name", ""))
        names = [str(a.get("name", "")) for a in anc]
        in_autograd = any(n.startswith("autograd::engine::evaluate_function") for n in names)
        label = next((n for n in reversed(names) if n.startswith("jla_linear:")), None)
        optimizer = any("Optimizer.step" in n or n.startswith("Optimizer.") for n in names)
        clip = any("clip_grad" in n or "_foreach_norm" in n for n in names)
        gemm_op = next((a for a in reversed(anc) if a.get("name") in _GEMM_OPS), None)
        is_gemm = bool(_GEMM_KERNEL.search(name)) or (bool(names) and names[-1] in _GEMM_OPS[:4])
        # a custom autograd Function's forward (its backward node is "<Name>Backward") or nn.Linear run again
        # inside the autograd engine = the forward recomputed by activation checkpointing
        forward_fn = any(re.fullmatch(r"[A-Z]\w*(Function|Fn|Func)", n) or n == "aten::linear" for n in names)
        if optimizer:
            phase = "opt"
        elif in_autograd and (label is not None or forward_fn or re.search(r"fwd", name)):
            phase = "recompute"
        elif in_autograd:
            phase = "bwd"
        else:
            phase = "fwd"
        dur = float(kev.get("dur", 0.0))
        innermost = names[-1] if names else ""
        detail = innermost
        attention = bool(re.search(r"flash|fmha|attn_fwd|attn_bwd|FlashAttn", name, re.I)) or any(
            "flash_attn" in n for n in names)  # FA3 kernels are CUTLASS kernels: test before the GEMM regex
        if optimizer:
            cat = "optimizer update"
        elif clip:
            cat = "grad norm/clip"
        elif attention:
            cat = "softmax attention"
        elif is_gemm:
            dims = [d for d in (_dims(gemm_op) if gemm_op else []) if isinstance(d, list)]
            if gemm_op is not None and gemm_op.get("name") in ("aten::addmm", "aten::baddbmm"):
                dims = dims[1:]  # (bias, a, b)
            dims = [d[1:] if len(d) == 3 and d[0] == 1 else d for d in dims]  # bmm with a batch of 1
            role = _role_from_label(label) if label else None
            if role is None and any(vocab in d for d in dims):  # FLA's fused linear cross entropy (chunked)
                role = "LM head"
            kn = None  # (K, N) of the forward [M, K] x [K, N]; candidates for weight gradients
            if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
                (a0, a1), (b0, b1) = dims[0], dims[1]
                if a0 == tokens and phase in ("fwd", "recompute"):  # forward [M, K] x [K, N]
                    kn = (a1, b1)
                elif a0 == tokens:  # input gradient dY [M, N] x W [N, K]
                    kn = (b1, a1)
                elif a1 == b0 == tokens:  # weight gradient dY^T [N, M] x X [M, K] (or X^T [K, M] x dY [M, N])
                    kn = (b1, a0)
            if role is not None and phase in ("fwd", "recompute") and kn is not None:
                fwd_roles[kn][role] += dur
            cat = "GEMM"
            detail = role or (f"shape {kn[0]}->{kn[1]}" if kn else f"{innermost} {dims}"[:90])
            classified.append([kev, cat, phase, detail, kn, role])
            continue
        elif re.search(r"flash|fmha|attn_fwd|attn_bwd|FlashAttn", name, re.I):
            cat = "softmax attention"
        elif re.search(r"layer_norm|rms_norm|layernorm|rmsnorm|group_norm|norm_gate|_norm_fwd|_norm_bwd|norm_kernel", name, re.I):
            cat = "norms"
        elif re.search(r"conv1d|causal_conv|short_conv", name, re.I):
            cat = "short conv"
        elif re.search(r"cross_entropy|logsumexp|_ce_|crossentropy", name, re.I):
            cat = "cross entropy (non-GEMM)"
        elif re.search(r"l2norm|l2_norm", name, re.I):
            cat = "l2norm"
        elif re.search(r"cumsum", name, re.I):
            cat = "cumsums"
        elif re.search(r"swiglu|silu|swish|sqrelu", name, re.I):
            cat = "MLP activation (SwiGLU etc.)"
        elif re.search(r"rotary|rope", name, re.I):
            cat = "rotary"
        elif _RECURRENT.search(name) and not _TORCH_KERNEL.search(name):
            cat = "recurrent kernels (FLA)"
            detail = re.sub(r"_+\d+$", "", name)[:70]
        elif str(kev.get("cat")) in ("gpu_memcpy", "gpu_memset") or re.search(r"copy|Cat|transpose", name):
            cat = "layout copies/transposes"
        elif any(n.startswith("FSDP") or "fsdp" in n.lower() for n in names):
            cat = "FSDP casts/copies"
        else:
            cat = "elementwise/reduction (aten, triton)"
        classified.append([kev, cat, phase, detail, None, None])
    # Unlabelled forward GEMMs (inside a fused autograd Function, e.g. FLA's SwiGLU-linear or Mamba-2's
    # combined op) take the role of the model's nn.Linear with that weight shape; then backward GEMMs take
    # the role of the forward GEMMs with the same weight shape.
    weight_roles = weight_roles or {}
    for row in classified:
        kev, cat, phase, detail, kn, role = row
        if cat == "GEMM" and role is None and kn and phase in ("fwd", "recompute"):
            names = weight_roles.get(f"{kn[0]}x{kn[1]}")
            if names:
                row[5] = {r: 1.0 / len(set(names)) for r in set(names)}
                for r in set(names):
                    fwd_roles[kn][r] += float(kev.get("dur", 0.0)) / len(set(names))
    for row in classified:
        kev, cat, phase, detail, kn, role = row
        if cat != "GEMM" or role is not None:
            continue
        shares = fwd_roles.get(kn) if kn else None
        if not shares and kn:
            shares = fwd_roles.get((kn[1], kn[0]))
        if not shares and kn:
            names = weight_roles.get(f"{kn[0]}x{kn[1]}") or weight_roles.get(f"{kn[1]}x{kn[0]}")
            shares = {r: 1.0 for r in set(names)} if names else None
        row[5] = dict(shares) if shares else None
    per_cat: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_detail: dict[tuple, float] = defaultdict(float)
    intervals = []
    total = 0.0
    for kev, cat, phase, detail, kn, role in classified:
        dur = float(kev.get("dur", 0.0))
        total += dur
        intervals.append((float(kev["ts"]), float(kev["ts"]) + dur))
        if cat == "GEMM":
            if isinstance(role, dict):
                s = sum(role.values())
                for r, v in role.items():
                    per_cat[f"GEMM: {r}"][phase] += dur * v / s / 1e3
                    per_cat[f"GEMM: {r}"]["total"] += dur * v / s / 1e3
                    per_detail[(f"GEMM: {r}", phase, f"{kn}")] += dur * v / s
                continue
            # a weight shape no labelled nn.Linear call has (a GEMM inside a fused autograd Function)
            r = role if isinstance(role, str) else (f"weight {min(kn)}x{max(kn)}" if kn else "unattributed")
            per_cat[f"GEMM: {r}"][phase] += dur / 1e3
            per_cat[f"GEMM: {r}"]["total"] += dur / 1e3
            per_detail[(f"GEMM: {r}", phase, f"{kn}" if kn else detail)] += dur
            continue
        per_cat[cat][phase] += dur / 1e3
        per_cat[cat]["total"] += dur / 1e3
        per_detail[(cat, phase, detail)] += dur
    span = (max(e for _, e in intervals) - min(s for s, _ in intervals)) if intervals else 0.0
    busy = _union_us(intervals)
    phases: dict[str, float] = defaultdict(float)
    for c, v in per_cat.items():
        for p, x in v.items():
            if p != "total":
                phases[p] += x
    return {
        "trace": trace_path.name,
        "profile_steps": 1,
        "kernels_per_step": len(classified),
        "kernel_ms_per_step": total / 1e3,
        "busy_ms_per_step": busy / 1e3,
        "span_ms_per_step": span / 1e3,
        "wall_ms": wall_ms,
        "phases_ms_per_step": dict(phases),
        "categories_ms_per_step": dict(sorted(((k, dict(v)) for k, v in per_cat.items()), key=lambda kv: -kv[1]["total"])),
        "details": [{"category": c, "phase": p, "detail": str(d)[:100], "ms_per_step": v / 1e3}
                    for (c, p, d), v in sorted(per_detail.items(), key=lambda kv: -kv[1])[: top * 3]],
    }


def print_breakdown(out: dict) -> None:
    print(f"span {out['span_ms_per_step']:.1f} ms; busy {out['busy_ms_per_step']:.1f}; kernel sum "
          f"{out['kernel_ms_per_step']:.1f}; kernels {out['kernels_per_step']}; wall {out['wall_ms']}")
    print("phases:", {k: round(v, 1) for k, v in out["phases_ms_per_step"].items()})
    for cat, v in out["categories_ms_per_step"].items():
        parts = " ".join(f"{p}={v.get(p, 0.0):7.2f}" for p in ("fwd", "recompute", "bwd", "opt"))
        print(f"  {cat:46s} {v['total']:8.2f} ms  {parts}")
    print("details:")
    for row in out["details"][:70]:
        print(f"  {row['ms_per_step']:8.2f} ms  {row['category'][:38]:38s} {row['phase']:9s} {row['detail'][:100]}")


# --------------------------------------------------------------------------------------
# Run (flame environment)


def _install_labels() -> None:
    import torch
    import torch.nn as nn

    original = nn.Linear.forward

    def forward(self, x):
        name = getattr(self, "_jla_profile_name", None)
        if name is None:
            return original(self, x)
        with torch.profiler.record_function(name):
            return original(self, x)

    nn.Linear.forward = forward


_LINEAR_SHAPES: dict[str, list[str]] = {}


def _name_linears(model) -> None:
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._jla_profile_name = f"jla_linear:{name}"
            key = f"{module.in_features}x{module.out_features}"
            role = _role_from_label(module._jla_profile_name)
            if role not in _LINEAR_SHAPES.setdefault(key, []):
                _LINEAR_SHAPES[key].append(role)


def run(args) -> int:
    runner = Path(args.runner).resolve()
    if not runner.is_file():
        raise SystemExit(f"--runner {runner}: no such file (the flame baseline runner benchmarks/bench_e2e_flame.py)")
    rest = [a for a in args.rest if a != "--"]
    spec = importlib.util.spec_from_file_location("bench_e2e_flame", runner)
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench_e2e_flame"] = bench
    spec.loader.exec_module(bench)
    missing = [name for name in ("_parse_args", "_worker", "_launch", "FLAME_DIR") if not hasattr(bench, name)]
    if missing:
        raise SystemExit(f"--runner {runner} lacks {', '.join(missing)}: use the bench_e2e_flame.py of the same tree")
    base = bench._parse_args(rest)
    if base.nproc != 1:
        raise SystemExit("profile one GPU: --nproc 1")
    dump = Path(base.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONDONTWRITEBYTECODE="1")
    if base.cache_dir:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(base.cache_dir) / "inductor"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(Path(base.cache_dir) / "triton"))
    if base.fla_backends == "triton":
        os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    if base.flame_src and base.flame_src not in sys.path:
        sys.path.append(base.flame_src)
    if base.fa3_shim:
        sys.path.insert(0, str(bench.FLAME_DIR / "fa3_shim"))
    _install_labels()
    import flame.train as flame_train

    auto = flame_train.AutoModelForCausalLM

    class _Auto:
        @staticmethod
        def from_config(*a, **kw):
            model = auto.from_config(*a, **kw)
            _name_linears(model)
            return model

    flame_train.AutoModelForCausalLM = _Auto
    profile = ["--set=--profiling.enable_profiling", "--set=--profiling.profile_freq=10",
               "--set=--profiling.save_traces_folder=profile_traces"]
    worker_args = bench._parse_args(rest + profile + ["--worker", "--init-file", str(dump / f"pg-{uuid.uuid4().hex}")])
    rc = bench._worker(worker_args)
    if rc:
        return rc
    summary_argv = rest + ["--from-raw"]
    sys.argv = [str(runner), *summary_argv]
    rc = bench._launch(bench._parse_args(summary_argv))
    traces = sorted(dump.glob("profile_traces/iteration_*/rank0_trace.json"))
    if not traces:
        print("PROFILE_ERROR no trace", file=sys.stderr)
        return 1
    result = json.loads(Path(base.json).read_text()) if base.json and Path(base.json).exists() else {}
    # host step times outside the profiler's warm-up and active steps (7-10)
    timed = [r["step_time"] for r in result.get("per_step", [])
             if r.get("step", 0) > base.skip and not 7 <= r.get("step", 0) <= 10]
    wall = 1e3 * sorted(timed)[len(timed) // 2] if timed else None
    tokens = base.micro_batch // base.grad_accum * base.seq_len
    (dump / "linear_shapes.json").write_text(json.dumps(_LINEAR_SHAPES, indent=2))
    out = analyze(traces[-1], tokens, wall, weight_roles=dict(_LINEAR_SHAPES))
    print_breakdown(out)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"flame_result": {k: result.get(k) for k in (
            "arch", "tokens_per_sec", "step_time_median_s", "document_masking", "activation_checkpointing",
            "torch_compile", "fla_backends", "attention_kernel", "grad_accum", "micro_batch_per_gpu",
            "peak_memory_allocated_gib_max_gpu", "fla_backend_dispatch", "versions")},
            "wall_ms_median_excluding_profiled_step": wall, "trace_dir": str(traces[-1].parent.name), "profile": out},
            indent=2))
    return rc


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.analyze_only:
        roles = None
        if args.model_config:  # the model on the meta device: only its nn.Linear shapes are needed
            shim = Path(args.model_config).parent / "fa3_shim"  # fla.layers.attn imports flash_attn
            if shim.is_dir():
                sys.path.insert(0, str(shim))
            import fla  # noqa: F401  (registers FLA's model types with transformers)
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM

            with torch.device("meta"):
                _name_linears(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_config)))
            roles = dict(_LINEAR_SHAPES)
        out = analyze(Path(args.analyze_only), args.tokens, args.wall_ms, weight_roles=roles)
        print_breakdown(out)
        if args.json:  # an existing result JSON of a run keeps its other fields; its profile is replaced
            result = json.loads(Path(args.json).read_text()) if Path(args.json).exists() else {}
            Path(args.json).write_text(json.dumps({**result, "profile": out}, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
