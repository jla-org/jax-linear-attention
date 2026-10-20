# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet with packed documents and long single sequences: walked map passes vs full walks vs FLA/FlashQLA.

    python benchmarks/bench_gdn_packed.py --doc-lens 0,512,2048 --torch-python FLA_VENV/bin/python \
        --qla-python QLA_VENV/bin/python

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: ``jla.ops.linear_attention("gated_deltanet", ..., impl="mosaic_gpu")``
  (the sequence split's map passes walk only the chunks that matter, :mod:`scalar_gate_docs`);
* ``mosaic_gpu_full``: the same kernels with the previous full-walk map passes
  (``scalar_gate.run(..., walk=False)``);
* ``mosaic_gpu_cpN``: ours with ``N`` sequence pieces instead of :func:`scalar_gate.pieces`;
* ``fla``: FLA 0.5.2 Triton ``chunk_gated_delta_rule`` (one ``cu_seqlens`` row with documents);
* ``flashqla``: FlashQLA 0.1.3 ``chunk_gated_delta_rule`` (TileLang, its default intra-card CP).

``--doc-lens`` lists document lengths (0: no documents); ours gets ``segment_ids``.
Metric: median CUDA kernel time per call (``harness.py``). ``--breakdown`` adds
the per-kernel times of one call of ours and of the full walks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from bench_gated_delta_rule import run_torch  # noqa: E402

from jla import ops  # noqa: E402
from jla.ops.mosaic_gpu import scalar_gate  # noqa: E402

SHAPES = [(1, 16384, 16, 128, 128), (4, 4096, 16, 128, 128), (1, 4096, 8, 128, 128)]


def jax_impls(shape, impls, doc_len):
    batch, seq_len, heads, key_dim, value_dim = shape
    segment_ids = None
    if doc_len:
        segment_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32) // doc_len, (batch, seq_len))
    x = harness.bf16_inputs(0, batch, seq_len, heads, key_dim, value_dim, delta=True)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
    out = {}
    for name in impls:
        if name == "mosaic_gpu":
            def fn(q, k, v, g, beta):
                return ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, impl="mosaic_gpu",
                                            segment_ids=segment_ids)
        elif name == "mosaic_gpu_full":
            def fn(q, k, v, g, beta):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, walk=False)
        elif name.startswith("mosaic_gpu_cp"):
            def fn(q, k, v, g, beta, cp=int(name[len("mosaic_gpu_cp"):])):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=segment_ids, cp=cp)
        else:
            continue
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        bwd = harness.vjp_fn(fn, tuple(range(len(args))))
        out[name] = (fwd, bwd, args, x["do"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA")
    parser.add_argument("--qla-python", default=None, help="python of an environment with FlashQLA")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default="mosaic_gpu,mosaic_gpu_full,fla,flashqla")
    parser.add_argument("--doc-lens", default="0,512,2048")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    shapes = SHAPES if not args.shapes else [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    impl_names = args.impls.split(",")
    modes = args.modes.split(",")
    results = dict(kind="gated_deltanet", environment=harness.environment(), metric="median CUDA kernel ms per call",
                   rounds=args.rounds, iterations=args.iterations, entries=[], breakdown=[])
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   f"gdn_packed_{time.strftime('%Y%m%d_%H%M')}.json")
    for doc_len in (int(d) for d in args.doc_lens.split(",")):
        for shape in shapes:
            jimpls = jax_impls(shape, impl_names, doc_len)
            if args.breakdown:
                for name, (fwd, bwd, fargs, do) in jimpls.items():
                    for mode, fn, a in (("fwd", fwd, fargs), ("fwdbwd", bwd, (*fargs, do))):
                        if mode in modes:
                            per = harness.kernel_breakdown_ms(fn, *a)
                            results["breakdown"].append(dict(shape=shape, doc_len=doc_len, impl=name, mode=mode,
                                                             kernels_ms=per))
                            print(json.dumps(dict(shape=shape, doc_len=doc_len, impl=name, mode=mode,
                                                  kernels_ms={k: round(v, 4) for k, v in per.items()})), flush=True)
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
                            r = run_torch(args.qla_python, "qla_gdn", shape, mode, None, args.iterations, doc_len)
                        elif name == "fla" and args.torch_python:
                            r = run_torch(args.torch_python, "fla_gdn", shape, mode, None, args.iterations, doc_len)
                        else:
                            continue
                        r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind", "doc_len")}
                        entry = dict(shape=shape, doc_len=doc_len, mode=mode, impl=name, round=rnd,
                                     wall_s=round(time.time() - t0, 1), gpu_before=before,
                                     gpu_after=harness.gpu_snapshot(), **r)
                        results["entries"].append(entry)
                        print(json.dumps({k: entry[k] for k in ("shape", "doc_len", "mode", "impl", "round")}
                                         | {k: entry.get(k) for k in ("median_ms", "min_ms", "error")}), flush=True)
            harness.write_json(out, results)  # after every shape, so a timeout keeps what was measured
    harness.write_json(out, results)
    summarize(results)
    print(f"wrote {out}")


def summarize(results: dict) -> None:
    table: dict = {}
    for e in results["entries"]:
        if "median_ms" in e:
            key = (tuple(e["shape"]), e["doc_len"], e["mode"])
            table.setdefault(key, {}).setdefault(e["impl"], []).append((e["median_ms"], e.get("min_ms")))
    for (shape, doc_len, mode), per in table.items():
        row = {k: (statistics.median(m for m, _ in v), min((x for _, x in v if x is not None), default=None))
               for k, v in per.items()}
        base = row.get("mosaic_gpu")
        cells = []
        for k, (med, best) in row.items():
            cell = f"{k}={med:.3f}" + (f" ({best:.3f})" if best is not None else "")
            if base and k != "mosaic_gpu" and best is not None and base[1]:
                cell += f" [{best / base[1]:.2f}x]"
            cells.append(cell)
        print(f"{shape} docs={doc_len} {mode}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
