# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel time of our Gated DeltaNet calls (forward, forward+backward), one JSON line per call.

    python benchmarks/profile_gdn_stages.py --shapes "1,16384,16,128,128" [--doc-len 2048] [--cp 1,4,16]

Each line: shape, documents, mode, the forced piece count (``cp``, ``null`` = the default policy),
the median total kernel time and the median time of every kernel name (CUPTI, ``harness.py``).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.ops.mosaic_gpu import scalar_gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--doc-len", type=int, default=0)
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--cp", default="", help="comma-separated forced piece counts ('' = default policy)")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--gates", default="fast", choices=("fast", "slow", "mixed"),
                        help="as in profile_qla_stages.py and gdn_warmup_error.py")
    parser.add_argument("--top", type=int, default=12, help="kernels kept per row")
    parser.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE",
                        help="set a jla.ops.mosaic_gpu module constant first (a Python literal), e.g. "
                             "scalar_gate_docs.PAIRED_MAPS=False")
    parser.add_argument("--options", default="", help="extra scalar_gate.run keyword options as JSON")
    parser.add_argument("--torch", action="append", default=[], metavar="KIND=PYTHON",
                        help="also profile a torch call after each shape (qla_gdn or fla_gdn), with that python")
    args = parser.parse_args()
    for item in args.set:
        target, value = item.split("=", 1)
        module, name = target.rsplit(".", 1)
        setattr(importlib.import_module(f"jla.ops.mosaic_gpu.{module}"), name, ast.literal_eval(value))
    extra = json.loads(args.options) if args.options else {}
    cps = [int(c) for c in args.cp.split(",") if c] or [None]
    for spec in args.shapes.split(";"):
        batch, seq_len, heads, key_dim, value_dim = (int(x) for x in spec.split(","))
        x = harness.bf16_inputs(0, batch, seq_len, heads, key_dim, value_dim, delta=True)
        if args.gates == "slow":
            x["g"] = jnp.full_like(x["g"], np.log(0.999))
        elif args.gates == "mixed":
            slow = jnp.log(jnp.linspace(0.98, 0.999, heads - heads // 2))
            x["g"] = x["g"].at[..., heads // 2:].set(jnp.broadcast_to(slow, x["g"][..., heads // 2:].shape))
        seg = None
        if args.doc_len:
            seg = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32) // args.doc_len, (batch, seq_len))
        for cp in cps:
            def fn(q, k, v, g, beta, cp=cp):
                return scalar_gate.run(q, k, v, g, beta, delta=True, scale=None, initial_state=None,
                                       output_final_state=False, segment_ids=seg, cp=cp, **extra)
            call_args = (x["q"], x["k"], x["v"], x["g"], x["beta"])
            for mode in args.modes.split(","):
                if mode == "fwd":
                    f, a = jax.jit(lambda *a: fn(*a)[0]), call_args
                else:
                    f, a = harness.vjp_fn(fn, tuple(range(5))), (*call_args, x["do"])
                per = harness.kernel_breakdown_ms(f, *a, iterations=args.iterations)
                row = dict(shape=[batch, seq_len, heads, key_dim, value_dim], doc_len=args.doc_len, mode=mode,
                           gates=args.gates,
                           cp=cp, total_ms=round(sum(per.values()), 4),
                           kernels={k: round(v, 4) for k, v in list(per.items())[:args.top]}, options=extra,
                           set=args.set)
                print(json.dumps(row), flush=True)
        for spec_t in args.torch:
            kind, python = spec_t.split("=", 1)
            here = os.path.dirname(os.path.abspath(__file__))
            cmd = [python, os.path.join(here, "profile_qla_stages.py"), "--kind", kind, "--shapes", spec,
                   "--doc-len", str(args.doc_len), "--modes", args.modes, "--iterations", str(args.iterations),
                   "--gates", args.gates]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            print(proc.stdout.strip() or json.dumps(dict(kind=kind, error=proc.stderr[-400:])), flush=True)


if __name__ == "__main__":
    main()
