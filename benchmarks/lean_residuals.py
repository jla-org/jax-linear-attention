# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Autodiff residuals of one block of each 1.3B preset under ``remat = "none"`` (benchmarks/fp8_1p3b.md).

    python benchmarks/lean_residuals.py --arch kda-1p3b --arch mamba2-1p3b --json residuals.json

For every distinct ``(layer_type, mlp_type)`` block of the preset (``benchmarks/end_to_end/1p3b/<arch>.toml``,
every mixer on its Mosaic GPU kernel as ``bench_e2e_1p3b.py --impl kernels``) it traces the block's VJP on
one packed ``[batch, seq_len]`` row (4 documents) with ``jax._src.ad_checkpoint.saved_residuals`` and
lists what the backward pass keeps: every residual's shape, dtype, bytes and producer (primitive and
source line). Nothing is compiled or run, so this needs no GPU memory. ``--lean`` repeats it with
``model.lean_vjp = true`` (``jla.layers.lean``).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _short(src: str) -> str:
    """Strip absolute paths from a residual description (``.../src/jla/x.py:12`` -> ``jla/x.py:12``)."""
    src = re.sub(r"\S*/src/(jla/)", r"\1", src)
    src = re.sub(r"\S*/(benchmarks/)", r"\1", src)
    return re.sub(r"\S*/site-packages/", "", src)


def block_residuals(model_cfg, layer_type: str, mlp_type: str, *, batch: int, seq_len: int) -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax._src.ad_checkpoint import saved_residuals

    from jla.models.lm import Block

    def make(key):
        block = Block(model_cfg, layer_type, mlp_type, key=key)
        if getattr(model_cfg, "lean_vjp", False):
            from jla.layers.lean import apply_lean

            block = apply_lean(block)
        return block

    abstract = eqx.filter_eval_shape(make, jax.random.key(0))
    is_abstract = lambda x: isinstance(x, jax.ShapeDtypeStruct)  # noqa: E731
    params, static = eqx.partition(abstract, is_abstract)
    x = jax.ShapeDtypeStruct((batch, seq_len, model_cfg.d_model), jnp.dtype(model_cfg.compute_dtype))

    def f(params, x):
        blk = eqx.combine(params, static)
        positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32) % (seq_len // 4), (batch, seq_len))
        segment_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32) // (seq_len // 4), (batch, seq_len))
        y, _ = blk(x, positions, segment_ids)
        return y

    residuals = saved_residuals(f, params, x)
    rows = []
    for aval, src in residuals:
        nbytes = int(aval.size) * aval.dtype.itemsize
        rows.append({"shape": list(aval.shape), "dtype": str(aval.dtype), "bytes": nbytes, "src": _short(src)})
    param_bytes = sum(r["bytes"] for r in rows if r["src"].startswith("from the argument params"))
    input_bytes = sum(r["bytes"] for r in rows if r["src"].startswith("from the argument x"))
    act = [r for r in rows if not r["src"].startswith("from the argument")]
    by_src: dict[str, list] = collections.defaultdict(lambda: [0, 0, set()])
    for r in act:
        entry = by_src[r["src"]]
        entry[0] += r["bytes"]
        entry[1] += 1
        entry[2].add(f"{r['dtype']}{tuple(r['shape'])}")
    grouped = sorted(({"src": k, "bytes": v[0], "count": v[1], "shapes": sorted(v[2])} for k, v in by_src.items()),
                     key=lambda e: -e["bytes"])
    by_dtype: dict[str, int] = collections.Counter()
    for r in act:
        by_dtype[r["dtype"]] += r["bytes"]
    token_bytes = batch * seq_len * model_cfg.d_model * jnp.dtype(model_cfg.compute_dtype).itemsize
    return {
        "layer_type": layer_type, "mlp_type": mlp_type, "batch": batch, "seq_len": seq_len,
        "num_residuals": len(rows), "activation_bytes": sum(r["bytes"] for r in act),
        "activation_bytes_per_hidden": sum(r["bytes"] for r in act) / token_bytes,
        "param_residual_bytes": param_bytes, "block_input_bytes": input_bytes,
        "activation_bytes_by_dtype": dict(by_dtype), "grouped": grouped, "rows": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", action="append", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lean", action="store_true", help="also trace with model.lean_vjp = true")
    p.add_argument("--json", required=True)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    import jax

    from bench_e2e_1p3b import _config_path, _with_impl
    from jla.config import load_config
    from jla.models.lm import layer_period

    out: dict = {"jax_version": jax.__version__, "device": str(jax.devices()[0].device_kind), "presets": {}}
    for arch in args.arch:
        cfg = load_config(str(_config_path(arch)), [f"train.seq_len={args.seq_len}"])
        base = _with_impl(cfg.model, "mosaic_gpu", "mosaic_gpu").replace(remat="none")
        modes = [("default", base)]
        if args.lean:
            modes.append(("lean", base.replace(lean_vjp=True)))
        period = layer_period(base)
        kinds = list(dict.fromkeys(zip(base.resolved_layer_types[:period], base.resolved_mlp_types[:period])))
        per_arch: dict = {"n_layers": base.n_layers, "d_model": base.d_model, "blocks": {}}
        for mode, mc in modes:
            for lt, mt in kinds:
                rec = block_residuals(mc, lt, mt, batch=args.batch, seq_len=args.seq_len)
                count = sum(1 for t in zip(base.resolved_layer_types, base.resolved_mlp_types) if t == (lt, mt))
                rec["blocks_in_stack"] = count
                per_arch["blocks"][f"{mode}:{lt}+{mt}"] = rec
                gib = rec["activation_bytes"] / 2**30
                print(f"\n== {arch} [{mode}] block {lt}+{mt} (x{count}): {rec['num_residuals']} residuals, "
                      f"{gib:.3f} GiB activations per {args.batch}x{args.seq_len} "
                      f"({rec['activation_bytes_per_hidden']:.1f}x the bf16 hidden state), "
                      f"stack {gib * count:.2f} GiB; by dtype "
                      + ", ".join(f"{k} {v / 2**30:.3f}" for k, v in rec["activation_bytes_by_dtype"].items()))
                for e in rec["grouped"][: args.top]:
                    print(f"  {e['bytes'] / 2**20:9.1f} MiB  x{e['count']:<3d} {', '.join(e['shapes'])[:70]:70s} "
                          f"{e['src'][:150]}")
        out["presets"][arch] = per_arch
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
