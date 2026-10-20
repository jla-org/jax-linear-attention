# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Short depthwise causal conv + SiLU: the Mosaic GPU kernels vs XLA vs causal-conv1d (benchmarks/short_conv.md).

    python benchmarks/bench_short_conv.py --json results.json [--torch-python /path/to/venv/bin/python]

Times the forward and the forward + backward (gradients of x, weight and bias; CUPTI kernel time,
``benchmarks/harness.py``) of ``ShortConv`` (kernel 4, SiLU) at the 1.3B presets' channel counts and
the per-GPU micro-batch of the 1.3B runs (``[16, 4096, C]``), dense and with packed documents:

* ``xla``: ``jla.layers.conv.ShortConv(impl="xla")`` under autodiff, as the layers ran it so far;
* ``mosaic_gpu``: ``ShortConv(impl="mosaic_gpu")`` (``jla.ops.mosaic_gpu.short_conv``);
* ``causal_conv1d``: the official CUDA kernel (``torch_short_conv.py`` in ``--torch-python``),
  channel-last ``[B, T, C]`` input as Mamba-2 passes it, ``seq_idx`` for the packed case.

Implementations are interleaved per case on the same GPU. Also reported: the HBM floor (every
tensor read or written once at 3.35 TB/s: x and y forward; x, y, dy and dx forward + backward).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from jla.layers.conv import ShortConv  # noqa: E402

HBM_BYTES_PER_S = 3.35e12
CASES = (  # (name, channels, bias): Mamba-2's x and B/C convs, the GDN / KDA q, k, v convs
    ("mamba2_x", 4096, True),
    ("mamba2_bc", 256, True),
    ("gdn_kda_qkv", 2048, False),
)


def _layer(channels: int, width: int, bias: bool, impl: str) -> ShortConv:
    return ShortConv(channels, width, use_bias=bias, init="uniform", key=jax.random.PRNGKey(0), impl=impl)


def _fns(conv: ShortConv, segment_ids):
    def fwd(x):
        return conv(x, segment_ids=segment_ids)[0]

    def primal(x, weight, bias):
        layer = eqx.tree_at(lambda c: c.weight, conv, weight)
        if bias is not None:
            layer = eqx.tree_at(lambda c: c.bias, layer, bias)
        return layer(x, segment_ids=segment_ids)[0]

    def fwd_bwd(x, weight, bias, dy):  # the output too, so the forward runs (as in a model)
        y, pullback = jax.vjp(primal, x, weight, bias)
        return y, pullback(dy)

    return jax.jit(fwd), jax.jit(fwd_bwd)


def _torch(python: str | None, shape, width: int, bias: bool, doc_len: int, mode: str, iterations: int):
    if not python:
        return None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_short_conv.py")
    cmd = [python, script, "--shape", ",".join(map(str, shape)), "--width", str(width), "--mode", mode,
           "--doc-len", str(doc_len), "--iterations", str(iterations)] + (["--bias"] if bias else [])
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if run.returncode:
        return dict(error=harness.error_summary(run.stderr or run.stdout))
    return json.loads(run.stdout.strip().splitlines()[-1])


def _variant(spec: str):
    """A context that overrides the kernels' block plan, e.g. ``"fbt=64,fg=8,fpre=0,bbt=128,bg=4,bvec=4"``
    (forward / backward block tokens, row group, channels per thread, next-group prefetch; any subset)."""
    import contextlib
    import dataclasses

    from jla.ops.mosaic_gpu import short_conv as kernels

    keys = {"fbt": ("block_t", False), "fg": ("group", False), "fvec": ("vec", False), "fpre": ("prefetch", False),
            "bbt": ("block_t", True), "bg": ("group", True), "bvec": ("vec", True), "bpre": ("prefetch", True)}
    changes = {True: {}, False: {}}
    for item in filter(None, spec.split(",")):
        name, value = item.split("=")
        field, backward = keys[name]
        changes[backward][field] = int(value)

    @contextlib.contextmanager
    def ctx():
        original = kernels._plan

        def plan(x, *a, backward, **kw):
            p = original(x, *a, backward=backward, **kw)
            p = dataclasses.replace(p, **changes[backward])
            p = dataclasses.replace(p, block_t=min(p.block_t, x.shape[1]))
            if x.shape[2] % p.block_c:
                raise ValueError(f"vec {p.vec} does not divide {x.shape[2]} channels")
            return p

        kernels._plan = plan
        try:
            yield
        finally:
            kernels._plan = original

    return ctx()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--doc-len", type=int, default=1024, help="document length of the packed case")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--torch-python", default=None, help="python of an environment with causal-conv1d")
    p.add_argument("--cases", default=",".join(c[0] for c in CASES))
    p.add_argument("--variants", default="", help="extra kernel block plans to time, ';'-separated (see _variant)")
    p.add_argument("--packed", default="0,1", help="which of dense (0) and packed (1) to run")
    args = p.parse_args()

    payload = dict(environment=harness.environment(), gpu_before=harness.gpu_snapshot(), rows=[])
    for name, channels, bias in CASES:
        if name not in args.cases.split(","):
            continue
        shape = (args.batch, args.seq_len, channels)
        ks = jax.random.split(jax.random.PRNGKey(channels), 2)
        x = jax.random.normal(ks[0], shape, jnp.float32).astype(jnp.bfloat16)
        dy = jax.random.normal(ks[1], shape, jnp.float32).astype(jnp.bfloat16)
        elems = x.size
        for packed in (bool(int(v)) for v in args.packed.split(",")):
            seg = None
            if packed:
                seg = jnp.broadcast_to(jnp.arange(args.seq_len, dtype=jnp.int32) // args.doc_len, shape[:2])
            row = dict(case=name, shape=list(shape), width=args.width, bias=bias,
                       doc_len=args.doc_len if packed else 0,
                       floor_fwd_ms=1e3 * 2 * elems * 2 / HBM_BYTES_PER_S,
                       floor_fwd_bwd_ms=1e3 * 5 * elems * 2 / HBM_BYTES_PER_S)
            for impl in ("xla", "mosaic_gpu"):
                conv = _layer(channels, args.width, bias, impl)
                fwd, fwd_bwd = _fns(conv, seg)
                row[f"{impl}_fwd"] = harness.median_kernel_ms(fwd, x, iterations=args.iterations)
                row[f"{impl}_fwd_bwd"] = harness.median_kernel_ms(fwd_bwd, x, conv.weight, conv.bias, dy,
                                                                  iterations=args.iterations)
            for spec in filter(None, args.variants.split(";")):
                conv = _layer(channels, args.width, bias, "mosaic_gpu")
                try:
                    with _variant(spec):
                        fwd, fwd_bwd = _fns(conv, seg)
                        row[f"variant[{spec}]_fwd"] = harness.median_kernel_ms(fwd, x, iterations=args.iterations)
                        row[f"variant[{spec}]_fwd_bwd"] = harness.median_kernel_ms(
                            fwd_bwd, x, conv.weight, conv.bias, dy, iterations=args.iterations)
                except ValueError as err:  # the plan does not fit this channel count
                    row[f"variant[{spec}]_fwd"] = row[f"variant[{spec}]_fwd_bwd"] = dict(skipped=str(err))
                    continue
                print(json.dumps({"case": name, "variant": spec,
                                  "fwd": row[f"variant[{spec}]_fwd"]["median_ms"],
                                  "fwd_bwd": row[f"variant[{spec}]_fwd_bwd"]["median_ms"]}), flush=True)
            for mode, key in (("fwd", "causal_conv1d_fwd"), ("fwdbwd", "causal_conv1d_fwd_bwd")):
                row[key] = _torch(args.torch_python, shape, args.width, bias, args.doc_len if packed else 0, mode,
                                  args.iterations)
            for mode in ("fwd", "fwd_bwd"):
                ours = row[f"mosaic_gpu_{mode}"]["median_ms"]
                row[f"speedup_vs_xla_{mode}"] = row[f"xla_{mode}"]["median_ms"] / ours
                base = row[f"causal_conv1d_{mode}"]
                if base and "median_ms" in base:
                    row[f"speedup_vs_causal_conv1d_{mode}"] = base["median_ms"] / ours
                row[f"floor_fraction_{mode}"] = row[f"floor_{mode}_ms"] / ours
            print(json.dumps({k: v for k, v in row.items() if "speedup" in k or k in ("case", "doc_len")}),
                  flush=True)
            payload["rows"].append(row)
    payload["gpu_after"] = harness.gpu_snapshot()
    harness.write_json(args.json, payload)


if __name__ == "__main__":
    main()
