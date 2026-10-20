# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel time of the large-head Gated DeltaNet path (K = 256, V = 512) by stage.

    python benchmarks/bench_gdn_hd256.py --shapes "4,4096,6,256,512;16,4096,6,256,512"

The ratios against FLA come from ``bench_gated_delta_rule.py`` (interleaved on
one GPU, see ``benchmarks/gdn.md``); this script splits our forward and
forward+backward into the Mosaic GPU kernels (``gdr_kkt_fwd``,
``gdr_kb_fused_fwd`` or ``gdr_kb_state_fwd``, ``gdr_kb_state_bwd``,
``gdr_kb_chunk_grads``) and the XLA ops (the per-chunk gradients of the
previous backward, the output of the forward through HBM, layout copies), for
the default path (fused forward and fused per-chunk gradients), the previous
backward (``scalar_gate_kb.run(..., fused_bwd=False)``) and the forward through
HBM (``fused_fwd=False``), with ``nvidia-smi`` utilization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.ops.mosaic_gpu import scalar_gate_kb  # noqa: E402

MOSAIC = ("gdr_kkt_fwd", "gdr_kb_state_fwd", "gdr_kb_fused_fwd", "gdr_kb_state_bwd", "gdr_kb_chunk_grads")
PATHS = {  # name: (fused_fwd, fused_bwd)
    "default": (True, True),
    "xla_chunk_grads": (True, False),
    "hbm_fwd": (False, True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="4,4096,6,256,512;16,4096,6,256,512")
    parser.add_argument("--paths", default=",".join(PATHS), help=f"comma-separated subset of {sorted(PATHS)}")
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    results = dict(environment=harness.environment(), metric="median CUDA kernel ms per call, by kernel",
                   iterations=args.iterations, entries=[])
    for shape in shapes:
        x = harness.bf16_inputs(0, *shape, delta=True)
        fargs = (x["q"], x["k"], x["v"], x["g"], x["beta"])
        for path in args.paths.split(","):
            fused_fwd, fused_bwd = PATHS[path]

            def fn(q, k, v, g, beta, fused_fwd=fused_fwd, fused_bwd=fused_bwd):
                return scalar_gate_kb.run(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                                          fused_fwd=fused_fwd, fused_bwd=fused_bwd)

            fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
            bwd = harness.vjp_fn(fn, tuple(range(len(fargs))))
            for mode, f, call_args in (("fwd", fwd, fargs), ("fwdbwd", bwd, fargs + (x["do"],))):
                before = harness.gpu_snapshot()
                per = harness.kernel_breakdown_ms(f, *call_args, iterations=args.iterations)
                mosaic = sum(t for name, t in per.items() if name.startswith(MOSAIC))
                total = sum(per.values())
                entry = dict(shape=shape, path=path, mode=mode, total_ms=total, mosaic_gpu_ms=mosaic,
                             xla_ms=total - mosaic, top=dict(list(per.items())[:12]), gpu_before=before,
                             gpu_after=harness.gpu_snapshot())
                results["entries"].append(entry)
                print(json.dumps(dict(shape=shape, path=path, mode=mode, total_ms=round(total, 3),
                                      mosaic_gpu_ms=round(mosaic, 3), xla_ms=round(total - mosaic, 3),
                                      top3={k: round(v, 3) for k, v in list(per.items())[:3]})), flush=True)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                   f"gdn_hd256_breakdown_{time.strftime('%Y%m%d_%H%M')}.json")
    harness.write_json(out, results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
