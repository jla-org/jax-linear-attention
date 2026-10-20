# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Forward variants of the Mosaic GPU attention kernel against each other (and cuDNN), interleaved.

Each variant (``--variants`` is comma-separated) is ``default``, ``cudnn`` or internal ``_Config``
options of ``jla.ops.mosaic_gpu.attention`` joined by ``+`` (``fwd_overlap=1`` is the overlapped
forward, with which ``fwd_wgs=3`` and ``fwd_k_ahead=1`` combine; ``heavy_first=1``,
``max_concurrent_steps=3``, ...); ``cudnn`` times
``jla.layers.attention.attention(..., implementation="cudnn")`` on the same inputs. Shapes are
``D:T:mask`` (``mask`` = ``causal``, ``swa<W>``, ``documents`` (four equal documents per row),
``rdocs<N>`` (N documents per row cut at seeded random positions) or ``swa<W>`` followed by either), with
16 query and key/value heads for ``D = 128`` and 32 query over 8 key/value heads for ``D = 64`` (the
transformer-1p3b attention), ``B = 16384 / T``. Every variant is compiled
first; a variant the kernel refuses (say ``fwd_wgs=3`` with ``D = 128``) is recorded and skipped. Then
``--rounds`` rounds time all variants in alternating order (median CUPTI kernel time of ``--iterations``
calls each, the whole jitted call: padding and slicing included). The outputs of all kernel variants are
compared with the first one. ``fwd_persistent=0`` is the forward of one CTA per query tile instead of the
persistent one.

    python benchmarks/bench_attention_forward.py --shapes 64:4096:causal,128:4096:causal \
        --variants default,fwd_overlap=1,heavy_first=1,fwd_overlap=1+fwd_wgs=3 --out results.json
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

import jla.ops.mosaic_gpu.attention as kernel_module  # noqa: E402
from jla.layers.attention import attention  # noqa: E402

_BOOLEAN = ("fwd_overlap", "fwd_named_turn", "fwd_named_consumed", "fwd_k_ahead", "heavy_first", "unmask_documents",
            "fwd_persistent")


def _options(variant: str) -> dict:
    if variant in ("default", "cudnn"):
        return {}
    options = {}
    for item in variant.split("+"):
        key, value = item.split("=")
        options[key] = bool(int(value)) if key in _BOOLEAN else int(value)
    return options


def _inputs(head_dim: int, length: int, mask: str):
    heads, kv_heads = (16, 16) if head_dim == 128 else (32, 8)
    batch = max(1, 16384 // length)
    keys = jax.random.split(jax.random.key(0), 3)
    shape = (batch, length)
    q = jax.random.normal(keys[0], (*shape, heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (*shape, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (*shape, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    seg = window = None
    if mask.endswith("documents"):  # documents, or swa<W>documents: a window over packed documents
        seg = jnp.broadcast_to((jnp.arange(length) // (length // 4)).astype(jnp.int32), shape)
        mask = mask[:-len("documents")] or "causal"
    elif "rdocs" in mask:  # rdocs<N> / swa<W>rdocs<N>: N documents per row cut at random (seeded) positions
        mask, docs = mask.split("rdocs")
        rng = np.random.default_rng(length)
        cuts = [np.sort(rng.choice(np.arange(1, length), int(docs) - 1, replace=False)) for _ in range(batch)]
        seg = jnp.asarray(np.stack([np.searchsorted(c, np.arange(length), side="right") for c in cuts]), jnp.int32)
        mask = mask or "causal"
    if mask.startswith("swa"):
        window = int(mask[3:])
    elif mask != "causal":
        raise ValueError(f"mask must be causal, swa<W>, documents, rdocs<N> or swa<W> before either; got {mask}")
    return (q, k, v), dict(batch=batch, heads=heads, kv_heads=kv_heads, seg=seg, window=window)


def _jitted(variant: str, seg, window):
    if variant == "cudnn":
        return jax.jit(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=window,
                                                 implementation="cudnn"))
    config = functools.partial(kernel_module._Config, **_options(variant))

    def fn(q, k, v):
        original = kernel_module._Config
        kernel_module._Config = config  # read when flash_attention builds its config (at trace time)
        try:
            return kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window)
        finally:
            kernel_module._Config = original

    return jax.jit(fn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shapes", required=True, help="comma-separated D:T:mask")
    parser.add_argument("--variants", required=True,
                        help="comma-separated variants; options within a variant joined by '+'")
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    variants = args.variants.split(",")
    rows = []
    gpu_before = harness.gpu_snapshot()

    def save():
        harness.write_json(args.out, dict(environment=harness.environment(), gpu_before=gpu_before,
                                          gpu=harness.gpu_snapshot(), variants=variants, rows=rows))

    for shape in args.shapes.split(","):
        head_dim, length, mask = shape.split(":")
        head_dim, length = int(head_dim), int(length)
        operands, info = _inputs(head_dim, length, mask)
        fns, outputs = {}, {}
        for variant in variants:
            row = dict(D=head_dim, T=length, B=info["batch"], H=info["heads"], Hkv=info["kv_heads"], mask=mask,
                       variant=variant, rounds_ms=[])
            fn = _jitted(variant, info["seg"], info["window"])
            try:
                outputs[variant] = jax.block_until_ready(fn(*operands))
                fns[variant] = fn
            except Exception as err:  # noqa: BLE001 - a refused variant is recorded, not fatal
                row["error"] = harness.error_summary(f"{type(err).__name__}: {err}")
            rows.append(row)
        reference = next((v for v in variants if v in outputs and v != "cudnn"), None)
        shape_rows = [r for r in rows if (r["D"], r["T"], r["mask"]) == (head_dim, length, mask)]
        for row in shape_rows:
            if row["variant"] in outputs and row["variant"] != "cudnn" and reference is not None:
                diff = outputs[row["variant"]].astype(jnp.float32) - outputs[reference].astype(jnp.float32)
                row["max_abs_diff_vs_" + reference] = float(jnp.max(jnp.abs(diff)))
        timed = [r for r in shape_rows if r["variant"] in fns]
        for r in range(args.rounds):
            for row in (timed if r % 2 == 0 else timed[::-1]):
                stats = harness.median_kernel_ms(fns[row["variant"]], *operands, iterations=args.iterations)
                row["rounds_ms"].append(stats["median_ms"])
        for row in timed:
            row["fwd_ms"] = min(row["rounds_ms"])  # the least disturbed round
            print({k: row[k] for k in ("D", "T", "mask", "variant", "fwd_ms", "rounds_ms")}, flush=True)
        save()
    save()


if __name__ == "__main__":
    main()
