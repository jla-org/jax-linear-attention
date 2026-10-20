# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kernel time of the q/k prologue: short conv + SiLU + per-head L2 norm, fused vs unfused (benchmarks/short_conv.md).

    python benchmarks/bench_short_conv_l2norm.py --json results.json [--batch 16] [--iterations 30]

Times, on bfloat16 ``[B, 4096, 2048]`` (the q or k of the 1.3B GDN / GDN-2 / KDA presets: 16 heads of 128, width 4,
SiLU, no bias), dense and with 1024-token documents:

* ``unfused``: the Mosaic GPU short conv, then ``reference.l2norm`` per head in XLA (the path before
  ``qk_l2norm_in_conv``), forward and forward + backward (gradients of x and the weight);
* ``fused``: :mod:`jla.ops.mosaic_gpu.short_conv_l2norm`;
* with ``--torch-python`` (an environment with FLA and causal-conv1d), FLA's prologue on the same GPU:
  FLA's Triton ``causal_conv1d`` or the causal-conv1d CUDA kernel, then FLA's Triton ``l2norm``
  (``torch_short_conv_l2norm.py``).

Median CUPTI kernel time of ``--iterations`` calls after 3 warm-ups (``benchmarks/harness.py``), with the per-kernel
breakdown, the HBM floor (x and n for the forward; x, n, dn and dx for forward + backward, at 3.35 TB/s) and
``nvidia-smi`` utilisation of the visible GPU before and after.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from jla.ops.mosaic_gpu.short_conv import short_conv  # noqa: E402
from jla.ops.mosaic_gpu.short_conv_l2norm import short_conv_l2norm  # noqa: E402
from jla.ops.reference import l2norm  # noqa: E402

HBM_BYTES_PER_S = 3.35e12


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--channels", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--torch-python", default=None, help="python of an environment with FLA and causal-conv1d")
    args = p.parse_args()
    b, t, c, d = args.batch, args.seq_len, args.channels, args.head_dim
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(ks[0], (b, t, c), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(ks[1], (c, 4), jnp.float32) * 0.5
    dn = jax.random.normal(ks[2], (b, t, c), jnp.float32).astype(jnp.bfloat16)
    docs = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32) // 1024, (b, t))

    def unfused(x, w, seg):
        y = short_conv(x, w, None, segment_ids=seg)
        return (l2norm(y.reshape(b, t, c // d, d)).reshape(b, t, c),)

    def fused(x, w, seg):
        return (short_conv_l2norm(x, w, None, head_dim=d, segment_ids=seg),)

    nbytes = b * t * c * 2
    floor = {"fwd": 2 * nbytes / HBM_BYTES_PER_S * 1e3, "fwd+bwd": 4 * nbytes / HBM_BYTES_PER_S * 1e3}
    rows = []
    before = harness.gpu_snapshot()
    for layout, seg in (("dense", None), ("docs1024", docs)):
        for name, fn in (("unfused", unfused), ("fused", fused)):
            fwd = jax.jit(lambda x, w, fn=fn, seg=seg: fn(x, w, seg))
            bwd = harness.vjp_fn(lambda x, w, fn=fn, seg=seg: fn(x, w, seg), (0, 1))
            for mode, call, call_args in (("fwd", fwd, (x, w)), ("fwd+bwd", bwd, (x, w, dn))):
                timing = harness.median_kernel_ms(call, *call_args, iterations=args.iterations)
                kernels = harness.kernel_breakdown_ms(call, *call_args)
                row = dict(path=name, layout=layout, mode=mode, shape=[b, t, c], head_dim=d, **timing,
                           floor_ms=floor[mode], fraction_of_floor=floor[mode] / timing["median_ms"],
                           kernels_ms=kernels)
                rows.append(row)
                print(f"{layout:8s} {mode:7s} {name:8s} {timing['median_ms']:.3f} ms "
                      f"(floor {floor[mode]:.3f}, {row['fraction_of_floor']:.2f})", flush=True)
        if args.torch_python:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_short_conv_l2norm.py")
            for conv in ("fla", "cuda"):
                for mode, flag in (("fwd", "fwd"), ("fwd+bwd", "fwdbwd")):
                    cmd = [args.torch_python, script, "--shape", f"{b},{t},{c}", "--head-dim", str(d), "--conv", conv,
                           "--mode", flag, "--doc-len", "1024" if seg is not None else "0",
                           "--iterations", str(args.iterations)]
                    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    name = f"{conv}_conv+fla_l2norm"
                    if run.returncode:
                        rows.append(dict(path=name, layout=layout, mode=mode,
                                         error=harness.error_summary(run.stderr)))
                        print(f"{layout:8s} {mode:7s} {name}: failed: {rows[-1]['error']}", flush=True)
                        continue
                    res = json.loads(run.stdout.strip().splitlines()[-1])
                    rows.append({**res, "path": name, "layout": layout, "mode": mode, "floor_ms": floor[mode],
                                 "fraction_of_floor": floor[mode] / res["median_ms"]})
                    print(f"{layout:8s} {mode:7s} {name} {res['median_ms']:.3f} ms", flush=True)
    after = harness.gpu_snapshot()
    harness.write_json(args.json, dict(environment=harness.environment(), gpu_before=before, gpu_after=after,
                                       rows=rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
