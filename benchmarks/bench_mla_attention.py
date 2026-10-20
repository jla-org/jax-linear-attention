# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLA attention core (q/k 192 = 128 no-RoPE + 64 RoPE, v 128): Mosaic GPU kernel vs FA3 and cuDNN.

Per shape, interleaved on one GPU (benchmarks/mla.md):

* ``mosaic_gpu``: ``jla.ops.mosaic_gpu.attention.flash_attention`` on the unpadded heads (the MLA
  layer's training path, ``MLAConfig.implementation`` ``"mosaic_gpu"`` or ``"auto"`` on Hopper);
* ``cudnn_native``: JAX's cuDNN SDPA (``jax._src.cudnn.fused_attention_stablehlo``, cuDNN >= 9.10 on
  Hopper) on the unpadded heads (q/k 192, v 128); packed documents as one batch row per document
  (exact only because the documents here have equal lengths: the work of an ideal ragged call);
* ``cudnn_packed`` (documents only): the same cuDNN SDPA in its packed-sequence layout (segment offsets
  and lengths), values zero-padded to 192: JAX 0.11.2 scales the value offsets by the q/k head dim, so
  the packed layout gives wrong results (NaN) at 192/128 and is correct only with equal head dims;
* ``cudnn``: the public ``jax.nn.dot_product_attention`` wrapper, which takes one head dim: values
  zero-padded to 192, the output sliced back to 128; packed documents as a dense boolean mask;
* ``fa3``: Dao-AILab FlashAttention-3 (``hopper/``) with ``Dqk`` 192 and ``Dv`` 128 in a torch
  subprocess (``torch_fa3_mla.py``), documents through ``flash_attn_varlen_func``.

``fwd`` is the forward, ``fwdbwd`` the forward plus the backward for a fixed cotangent, as the median
CUDA kernel time of ``--iterations`` calls; every shape runs ``--rounds`` interleaved rounds (the order
alternates) and keeps the fastest round median. TFLOP/s count visible query-key pairs:
``2 (Dqk + Dv)`` FLOPs per pair forward and ``2 (3 Dqk + 2 Dv)`` backward.

    python benchmarks/bench_mla_attention.py --lengths 4096,8192 --heads 16 --masks causal,documents \
        --fa3-python <venv of benchmarks/build_fa3_mla.sh>/bin/python --out <json>

FA3's default build compiles out head dim 192, so ``fa3`` needs the build of ``build_fa3_mla.sh``.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from bench_attention_fa3 import Worker, _pairs, _wall_ms  # noqa: E402

from jax._src.cudnn import fused_attention_stablehlo as cudnn_sdpa  # noqa: E402

from jla.layers.attention import attention  # noqa: E402
from jla.ops.mosaic_gpu.attention import flash_attention  # noqa: E402

IMPLS = ("mosaic_gpu", "cudnn_native", "cudnn_packed", "cudnn", "fa3")
# The FA3 baseline build (benchmarks/build_fa3_mla.sh): FA3's default build compiles out head dim 192.
FA3_BUILD = ("Dao-AILab/flash-attention 965c9212e6be33c821ee9f2b461564c5c91a2503 hopper/, torch 2.13.0+cu129, "
             "CUDA 12.9; FLASH_ATTENTION_DISABLE_{FP8,FP16,HDIM64,HDIM96,HDIM128,HDIM256,HDIMDIFF64,SM80,PAGEDKV,"
             "APPENDKV,SPLIT,SOFTCAP}=TRUE (keeps bf16 sm90 head dim 192 and the 192/128 forward HDIMDIFF192)")


def _inputs(batch, length, heads, dqk, dv, docs):
    keys = jax.random.split(jax.random.key(length), 4)
    bf16, f32 = jnp.bfloat16, jnp.float32
    q = jax.random.normal(keys[0], (batch, length, heads, dqk), f32).astype(bf16)
    k = jax.random.normal(keys[1], (batch, length, heads, dqk), f32).astype(bf16)
    v = jax.random.normal(keys[2], (batch, length, heads, dv), f32).astype(bf16)
    do = jax.random.normal(keys[3], (batch, length, heads, dv), f32).astype(bf16)
    run = jnp.arange(length) // (length // docs)
    seg = jnp.broadcast_to(run.astype(jnp.int32), (batch, length))
    return q, k, v, do, seg


def _jax_fn(impl, seg, docs):
    scale = None  # 1 / sqrt(Dqk) everywhere

    def padded(q, k, v, **kwargs):
        dqk, dv = q.shape[-1], v.shape[-1]
        v = jnp.pad(v, [(0, 0)] * 3 + [(0, dqk - dv)])
        return attention(q, k, v, scale=q.shape[-1] ** -0.5, implementation="cudnn", **kwargs)[..., :dv]

    if impl == "mosaic_gpu":
        def fn(q, k, v):
            return (flash_attention(q, k, v, segment_ids=seg, scale=scale),)
    elif impl == "cudnn":
        def fn(q, k, v):
            return (padded(q, k, v, segment_ids=seg),)
    elif impl == "cudnn_native":
        def fn(q, k, v):
            causal = cudnn_sdpa.MaskType.CAUSAL
            if seg is None:
                return (cudnn_sdpa.dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask_type=causal),)
            batch, length = q.shape[:2]
            split = lambda x: x.reshape(batch * docs, length // docs, *x.shape[2:])  # noqa: E731
            out = cudnn_sdpa.dot_product_attention(split(q), split(k), split(v), scale=q.shape[-1] ** -0.5,
                                                   mask_type=causal)
            return (out.reshape(batch, length, *out.shape[2:]),)
    elif impl == "cudnn_packed":
        def fn(q, k, v):
            batch, length = q.shape[:2]
            doc = length // docs  # segment starts, -1 padded to M + 1 entries, and lengths, per row
            starts = jnp.concatenate([jnp.arange(docs, dtype=jnp.int32) * doc, jnp.full((1,), -1, jnp.int32)])
            offsets = jnp.broadcast_to(starts, (batch, docs + 1))
            lengths = jnp.full((batch, docs), doc, jnp.int32)
            dv = v.shape[-1]
            v = jnp.pad(v, [(0, 0)] * 3 + [(0, q.shape[-1] - dv)])
            out = cudnn_sdpa.dot_product_attention(q, k, v, None, None, lengths, lengths, offsets, offsets,
                                                   scale=q.shape[-1] ** -0.5,
                                                   mask_type=cudnn_sdpa.MaskType.PADDING_CAUSAL)
            return (out[..., :dv],)
    else:
        raise ValueError(impl)
    return fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--masks", default="causal,documents")
    parser.add_argument("--heads", default="16", help="comma-separated query (= key/value) head counts")
    parser.add_argument("--dqk", type=int, default=192)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=16384, help="tokens per batch: B = max(1, tokens // T)")
    parser.add_argument("--docs", type=int, default=4)
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--fa3-python", default=None)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    impls = args.impls.split(",")
    workers = {}
    if "fa3" in impls and args.fa3_python:
        workers["fa3"] = Worker(args.fa3_python, "torch_fa3_mla.py")
    dqk, dv = args.dqk, args.dv
    rows, versions = [], {}
    gpu_before = harness.gpu_snapshot()

    def save():
        try:
            cudnn_version = cudnn_sdpa.check_cudnn_version()
        except Exception:  # noqa: BLE001 (no cuDNN)
            cudnn_version = None
        versions["cudnn"] = dict(jax=jax.__version__, cudnn=cudnn_version)
        harness.write_json(args.out, dict(environment=harness.environment(), baselines=versions,
                                          gpu_before=gpu_before, gpu=harness.gpu_snapshot(), rows=rows))

    try:
        for heads in (int(h) for h in args.heads.split(",")):
            for length in (int(x) for x in args.lengths.split(",")):
                batch = max(1, args.tokens // length)
                q, k, v, do, seg = _inputs(batch, length, heads, dqk, dv, args.docs)
                for mask in args.masks.split(","):
                    docs = mask == "documents"
                    doc_len = length // args.docs if docs else None
                    pairs = batch * heads * _pairs(length, None, doc_len)
                    fwd_flops, bwd_flops = 2.0 * pairs * (dqk + dv), 2.0 * pairs * (3 * dqk + 2 * dv)
                    shape_rows, jitted = {}, {}
                    for impl in impls:
                        if impl == "cudnn_packed" and not docs:
                            continue
                        if impl == "fa3" and impl not in workers:
                            continue
                        shape_rows[impl] = dict(T=length, B=batch, H=heads, Dqk=dqk, Dv=dv, mask=mask,
                                                docs=args.docs if docs else 0, impl=impl, rounds=[])
                        if impl not in workers:
                            fn = _jax_fn(impl, seg if docs else None, args.docs)
                            jitted[impl] = (jax.jit(fn), harness.vjp_fn(fn, (0, 1, 2)))
                    if "mosaic_gpu" in jitted:  # every JAX path's output against the kernel's, same inputs
                        ref = jitted["mosaic_gpu"][0](q, k, v)[0].astype(jnp.float32)
                        for impl, (fwd_fn, _) in jitted.items():
                            try:
                                out = fwd_fn(q, k, v)[0].astype(jnp.float32)
                                shape_rows[impl]["out_rel_err_vs_mosaic_gpu"] = float(
                                    jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref)))
                            except Exception as err:  # noqa: BLE001 (the timing below records the failure)
                                shape_rows[impl]["out_rel_err_vs_mosaic_gpu"] = f"{type(err).__name__}"
                    for r in range(args.rounds):
                        for impl in (list(shape_rows) if r % 2 == 0 else list(shape_rows)[::-1]):
                            row = shape_rows[impl]
                            if "error" in row:
                                continue
                            if impl in workers:
                                answer = workers[impl].ask(dict(
                                    op="time", B=batch, T=length, H=heads, Dqk=dqk, Dv=dv,
                                    doc_lens=[[doc_len] * args.docs] * batch if docs else None,
                                    iterations=args.iterations))
                                for key in ("flash_attn_3", "torch"):
                                    if key in answer:
                                        versions.setdefault(impl, {})[key] = answer.pop(key)
                                versions.setdefault(impl, {})["build"] = FA3_BUILD
                                for part in ("fwd", "fwdbwd"):  # a profiler capture that lost kernels
                                    ms, wall = answer.get(f"{part}_ms"), answer.get(f"{part}_wall_ms")
                                    if ms and wall and ms < 0.5 * wall:
                                        answer[f"{part}_kernel_error"] = (
                                            f"kernel time {ms:.4f} ms < half the wall time {wall:.4f} ms: "
                                            "incomplete profiler capture, dropped")
                                        answer.pop(f"{part}_ms")
                                        answer.pop(f"{part}_min_ms", None)
                            else:
                                fwd_fn, both_fn = jitted[impl]
                                try:
                                    fwd = harness.median_kernel_ms(fwd_fn, q, k, v, iterations=args.iterations)
                                    both = harness.median_kernel_ms(both_fn, q, k, v, do, iterations=args.iterations)
                                    answer = dict(fwd_ms=fwd["median_ms"], fwd_min_ms=fwd["min_ms"],
                                                  fwdbwd_ms=both["median_ms"], fwdbwd_min_ms=both["min_ms"],
                                                  fwd_wall_ms=_wall_ms(fwd_fn, (q, k, v), args.iterations),
                                                  fwdbwd_wall_ms=_wall_ms(both_fn, (q, k, v, do), args.iterations))
                                except Exception as err:  # record the failure, keep measuring the rest
                                    answer = dict(error=harness.error_summary(f"{type(err).__name__}: {err}"))
                            if "error" in answer or not all(answer.get(f"{p}_ms") or answer.get(f"{p}_wall_ms")
                                                            for p in ("fwd", "fwdbwd")):
                                row["error"] = answer.get("error", f"no kernel time recorded: {answer}")
                                continue
                            row["rounds"].append(answer)
                    for impl, row in shape_rows.items():
                        if row["rounds"]:
                            for key in ("fwd_ms", "fwdbwd_ms", "fwd_min_ms", "fwdbwd_min_ms", "fwd_wall_ms",
                                        "fwdbwd_wall_ms"):
                                values = [a[key] for a in row["rounds"] if a.get(key)]
                                if values:  # FA3 rows may lack a kernel time (incomplete profiler captures)
                                    row[key] = min(values)
                            for part, flops in (("fwd", fwd_flops), ("fwdbwd", fwd_flops + bwd_flops)):
                                if f"{part}_ms" in row:
                                    row[f"{part}_tflops"] = flops / (row[f"{part}_ms"] * 1e9)
                                if f"{part}_wall_ms" in row:
                                    row[f"{part}_wall_tflops"] = flops / (row[f"{part}_wall_ms"] * 1e9)
                        print({key: val for key, val in row.items() if key != "rounds"}, flush=True)
                        rows.append(row)
                    save()
    finally:
        for worker in workers.values():
            worker.close()
        save()


if __name__ == "__main__":
    main()
