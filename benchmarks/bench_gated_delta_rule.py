# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the scalar-gate linear-attention kernels against FLA, XLA and MaxText/Mamba-2.

    python benchmarks/bench_gated_delta_rule.py --kind gated_deltanet --torch-python VENV/bin/python
    python benchmarks/bench_gated_delta_rule.py --kind simple_gla --torch-python VENV/bin/python \
        --mamba-root PATH/TO/state-spaces/mamba

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: ``jla.ops.linear_attention(kind, ..., impl="mosaic_gpu")``;
* ``mosaic_gpu_prev`` (gated_deltanet): the previous Mosaic GPU path, which
  materializes the WY factors and runs the six-kernel backward
  (``scalar_gate.run(..., wy=False)``);
* ``mosaic_gpu_split`` / ``mosaic_gpu_fused`` (gated_deltanet): the gate-free path with
  the backward forced to the split kernels (``state_bwd_wy`` + ``chunk_grads``, the
  default) or to the opt-in fused kernel (one CTA per ``(b, h)``, without a sequence
  split) (``scalar_gate.run(..., fused_bwd=False / True)``);
* ``mosaic_gpu_grouped`` (simple_gla): the same kernels with ``q``/``k`` shared by
  all heads (Mamba-2 ``ngroups=1``) instead of repeated;
* ``mosaic_gpu_kb_xla`` (gated_deltanet, ``K = 256`` head dims): the large-head path with
  its previous backward, the per-chunk gradients as XLA batched matmuls
  (``scalar_gate_kb.run(..., fused_bwd=False)``);
* ``mosaic_gpu_warmup`` (gated_deltanet): the default path with the opt-in approximate
  sequence split ``scalar_gate.run(..., cp_warmup=-10.0)`` (FlashQLA's warm-up; benchmarks/gdn.md);
* ``mosaic_gpu_r0`` (gated_deltanet): the earlier exact sequence split in the same process (at most 4
  pieces, the piece scans in XLA, every forward map computed:
  ``scalar_gate.run(..., piece_scan_kernel=False, skip_fresh_maps=False)``);
* ``chunked``: ``impl="chunked"`` with ``precision=None`` (plain XLA);
* ``maxtext`` (gated_deltanet): MaxText's ``jax_chunk_gated_delta_rule`` (copied in
  ``maxtext_gdn.py``);
* ``fla``: FLA 0.5.2 Triton (``chunk_gated_delta_rule`` / ``chunk_simple_gla``);
* ``flashqla`` (gated_deltanet): FlashQLA 0.1.3 ``chunk_gated_delta_rule`` (TileLang), from ``--qla-python``;
* ``mamba2`` (simple_gla): Mamba-2's Triton ``mamba_chunk_scan_combined`` (chunk 256), from
  ``--mamba-python`` (installed ``mamba_ssm``) or ``--torch-python`` with ``--mamba-root``;
* ``fla_full``: FLA's default dispatch in ``--fla-full-python``, an environment that also has
  FLA's optional backends (FlashQLA, TileLang), so FLA may route the call to them.

Metric: median CUDA kernel time per call (see ``harness.py``); ``fwdbwd`` is one
forward plus the backward for a fixed output cotangent. ``--doc-len N`` packs
documents of ``N`` tokens: ``segment_ids`` for the JAX implementations, one
``cu_seqlens`` row for FLA and ``seq_idx`` for Mamba-2 (MaxText has no packing).
Results go to ``benchmarks/results/{gdn,simple_gla}[_docsN]_<date>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import maxtext_gdn  # noqa: E402

from jla import ops  # noqa: E402

GDN_SHAPES = [(1, 4096, 8, 128, 128), (4, 4096, 16, 128, 128), (1, 16384, 16, 128, 128),
              (8, 2048, 32, 128, 128), (4, 4096, 16, 64, 64)]
SGLA_SHAPES = [(4, 4096, 24, 128, 64), (1, 16384, 48, 128, 64), (1, 4096, 8, 128, 128),
               (1, 16384, 16, 128, 128), (4, 4096, 16, 128, 128)]


def jax_impls(kind: str, shape, impls: list[str], doc_len: int = 0):
    """``name -> (fwd_fn, fwdbwd_fn, args)`` for the JAX implementations (packed documents if ``doc_len``)."""
    batch, seq_len, heads, key_dim, value_dim = shape
    segment_ids = None
    if doc_len:
        segment_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32) // doc_len, (batch, seq_len))
    delta = kind == "gated_deltanet"
    x = harness.bf16_inputs(0, batch, seq_len, heads, key_dim, value_dim, delta=delta)
    scale = None if delta else 1.0
    gate_names = ("g", "beta") if delta else ("g",)
    out = {}

    def via_ops(impl, q, k):
        def fn(q, k, v, *gates):
            kw = dict(zip(gate_names, gates))
            if impl == "chunked":
                kw["precision"] = None
            if segment_ids is not None:
                kw["segment_ids"] = segment_ids
            return ops.linear_attention(kind, q, k, v, impl=impl, scale=scale, **kw)
        args = (q, k, x["v"], *(x[n] for n in gate_names))
        return fn, args

    for name in impls:
        if name in ("mosaic_gpu", "chunked"):
            fn, args = via_ops(name, x["q"], x["k"])
        elif name == "mosaic_gpu_grouped":
            grouped = harness.bf16_inputs(0, batch, seq_len, heads, key_dim, value_dim, kv_heads=1, delta=delta)
            fn, args = via_ops("mosaic_gpu", grouped["q"], grouped["k"])
        elif name == "mosaic_gpu_prev" and delta:
            from jla.ops.mosaic_gpu import scalar_gate

            def fn(q, k, v, g, beta):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, wy=False)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        elif name in ("mosaic_gpu_split", "mosaic_gpu_fused") and delta:
            from jla.ops.mosaic_gpu import scalar_gate

            def fn(q, k, v, g, beta, fused=name == "mosaic_gpu_fused"):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, fused_bwd=fused,
                                       cp=1 if fused else None)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        elif name == "mosaic_gpu_r0" and delta:  # the earlier exact split: at most 4 pieces, the scans in XLA
            from jla.ops.mosaic_gpu import scalar_gate

            cp0 = min(scalar_gate.pieces(batch, heads, value_dim, seq_len, delta=True), 4)

            def fn(q, k, v, g, beta, cp0=cp0):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, cp=cp0,
                                       piece_scan_kernel=False, skip_fresh_maps=False)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        elif name == "mosaic_gpu_warmup" and delta:  # opt-in approximate split (FlashQLA's warm-up, e^-10)
            from jla.ops.mosaic_gpu import scalar_gate

            def fn(q, k, v, g, beta):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, cp_warmup=-10.0)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        elif name == "mosaic_gpu_kb_xla" and delta:  # K = 256 path with the previous (XLA) per-chunk gradients
            from jla.ops.mosaic_gpu import scalar_gate_kb

            def fn(q, k, v, g, beta):
                return scalar_gate_kb.run(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                                          segment_ids=segment_ids, fused_bwd=False)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        elif name == "maxtext" and not doc_len:  # no packed-document support
            def fn(q, k, v, g, beta):
                return maxtext_gdn.jax_chunk_gated_delta_rule(q, k, v, g, beta)
            args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        else:
            continue
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        bwd = harness.vjp_fn(fn, tuple(range(len(args))))
        out[name] = (fwd, bwd, args, x["do"])
    return out


def run_torch(torch_python: str, kind: str, shape, mode: str, mamba_root: str | None, iterations: int,
              doc_len: int = 0) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_baselines.py")
    cmd = [torch_python, script, "--kind", kind, "--shape", ",".join(map(str, shape)), "--mode", mode,
           "--iterations", str(iterations), "--doc-len", str(doc_len)]
    if mamba_root:
        cmd += ["--mamba-root", mamba_root]
    # The baseline inherits this process's environment, so it runs on the same GPU(s).
    env = dict(os.environ)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("gated_deltanet", "simple_gla"), required=True)
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA")
    parser.add_argument("--qla-python", default=None, help="python of an environment with FlashQLA (gated_deltanet)")
    parser.add_argument("--mamba-root", default=None, help="state-spaces/mamba source tree (simple_gla)")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default=None, help="comma-separated subset of implementations")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--doc-len", type=int, default=0, help="pack documents of this many tokens (0: none)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fla-full-python", default=None,
                        help="python of an environment with FLA and its optional backends (impl fla_full: FLA's "
                             "default dispatch)")
    parser.add_argument("--mamba-python", default=None,
                        help="python of an environment with mamba_ssm installed (impl mamba2; overrides "
                             "--torch-python for it)")
    parser.add_argument("--baseline-commit", action="append", default=[], metavar="NAME=COMMIT",
                        help="record the exact commit of a baseline in the result JSON (repeatable)")
    args = parser.parse_args()

    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    else:
        shapes = GDN_SHAPES if args.kind == "gated_deltanet" else SGLA_SHAPES
    if args.impls:
        impl_names = args.impls.split(",")
    elif args.kind == "gated_deltanet":
        impl_names = ["mosaic_gpu", "fla", "flashqla", "chunked", "maxtext"]
    else:
        impl_names = ["mosaic_gpu", "mosaic_gpu_grouped", "fla", "mamba2", "chunked"]
    modes = args.modes.split(",")
    torch_kind = {"gated_deltanet": "fla_gdn", "simple_gla": "fla_simple_gla"}[args.kind]

    results = dict(kind=args.kind, environment=harness.environment(), metric="median CUDA kernel ms per call",
                   rounds=args.rounds, iterations=args.iterations, doc_len=args.doc_len, entries=[])
    if args.baseline_commit:
        results["baseline_commits"] = dict(item.split("=", 1) for item in args.baseline_commit)
    for shape in shapes:
        jimpls = jax_impls(args.kind, shape, impl_names, args.doc_len)
        for mode in modes:
            for rnd in range(args.rounds):
                for name in impl_names:
                    before = harness.gpu_snapshot()
                    t0 = time.time()
                    if name in jimpls:
                        fwd, bwd, fargs, do = jimpls[name]
                        try:
                            if mode == "fwd":
                                r = harness.median_kernel_ms(fwd, *fargs, iterations=args.iterations)
                            else:
                                r = harness.median_kernel_ms(bwd, *fargs, do, iterations=args.iterations)
                        except Exception as err:  # noqa: BLE001  (record and continue)
                            r = dict(error=f"{type(err).__name__}: {str(err)[:500]}")
                    elif name == "flashqla" and args.qla_python:
                        r = run_torch(args.qla_python, "qla_gdn", shape, mode, None, args.iterations, args.doc_len)
                    elif name == "fla_full" and args.fla_full_python:
                        r = run_torch(args.fla_full_python, torch_kind, shape, mode, None, args.iterations,
                                      args.doc_len)
                    elif name == "mamba2" and args.mamba_python:
                        r = run_torch(args.mamba_python, "mamba2", shape, mode, args.mamba_root, args.iterations,
                                      args.doc_len)
                    elif name in ("fla", "mamba2") and args.torch_python:
                        if name == "mamba2" and not args.mamba_root:
                            continue
                        r = run_torch(args.torch_python, torch_kind if name == "fla" else "mamba2", shape, mode,
                                      args.mamba_root, args.iterations, args.doc_len)
                    else:
                        continue
                    r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind")}
                    entry = dict(shape=shape, mode=mode, impl=name, round=rnd, wall_s=round(time.time() - t0, 1),
                                 gpu_before=before, gpu_after=harness.gpu_snapshot(), **r)
                    results["entries"].append(entry)
                    print(json.dumps({k: entry[k] for k in ("shape", "mode", "impl", "round")}
                                     | {k: entry.get(k) for k in ("median_ms", "error")}), flush=True)
    date = time.strftime("%Y%m%d_%H%M")
    prefix = "gdn" if args.kind == "gated_deltanet" else "simple_gla"
    if args.doc_len:
        prefix += f"_docs{args.doc_len}"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{prefix}_{date}.json")
    harness.write_json(out, results)
    summarize(results)
    print(f"wrote {out}")


def summarize(results: dict) -> None:
    table: dict = {}
    for e in results["entries"]:
        if "median_ms" in e:
            table.setdefault((tuple(e["shape"]), e["mode"]), {}).setdefault(e["impl"], []).append(e["median_ms"])
    for (shape, mode), per in table.items():
        row = {k: statistics.median(v) for k, v in per.items()}
        base = row.get("mosaic_gpu")
        cells = "  ".join(f"{k}={v:.3f}" + (f" ({v / base:.2f}x)" if base and k != "mosaic_gpu" else "")
                          for k, v in row.items())
        print(f"{shape} {mode}: {cells}")


if __name__ == "__main__":
    main()
