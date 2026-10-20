# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Softmax attention: the Mosaic GPU kernel against FlashAttention-3, cuDNN and Tokamax.

Per shape, interleaved on one GPU: ``mosaic_gpu`` (``jla.layers.attention.attention(...,
implementation="mosaic_gpu")``), ``cudnn`` (the same layer with ``implementation="cudnn"``: native
left window; packed documents as the dense boolean mask the layer builds), ``cudnn_split``
(documents only: each equal-length document as its own batch row, the same work as a ragged cuDNN
call), ``fa3`` (Dao-AILab flash-attention ``hopper/`` FA3 in a torch subprocess, ``torch_fa3.py``;
documents through ``flash_attn_varlen_func`` with ``cu_seqlens``) and ``tokamax`` (Tokamax's
Mosaic GPU attention in a JAX subprocess, ``jax_tokamax_attention.py``). ``mosaic_gpu:<backward>``
(for example ``mosaic_gpu:ranged`` or ``mosaic_gpu:fused``) calls the kernel directly with that
backward, so the backward variants can be timed against each other; ``mosaic_gpu:<backward>:overlap``
and ``mosaic_gpu:<backward>:pingpong`` also force the forward loop (``_Config.fwd_overlap`` True or False;
without it the kernel picks its default), and ``mosaic_gpu:<backward>:tiles`` runs the forward with one CTA per
query tile (``_Config.fwd_persistent=False``) instead of the persistent one. Masks: ``causal``,
``swa<W>`` (``W`` visible tokens including the query) and ``documents`` (``--docs`` equal
documents per row). ``fwd`` is the forward, ``fwdbwd`` one forward plus the backward for a fixed
cotangent, as median CUDA kernel time (CUPTI for JAX, ``torch.profiler`` for torch) of ``--iterations``
calls; every shape runs ``--rounds`` interleaved rounds (the order alternates) and reports the fastest
round median, the one least disturbed by other jobs on a shared GPU (every round is in the JSON). Where
the FA3 worker's profiler captures fail it reports CUDA-event GPU time instead (``torch_fa3.py``): a
cell takes its fastest profiler-timed round, an event-timed one only if no round has a profiler time,
and records which in ``fwd_timer`` / ``fwdbwd_timer``.
TFLOP/s count visible query-key pairs only (backward = 2.5x forward).

    python benchmarks/bench_attention_fa3.py --lengths 4096 --masks causal,swa1024,documents \
        --fa3-python <venv with flash_attn_3>/bin/python --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.layers.attention import attention  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMPLS = ("mosaic_gpu", "cudnn", "cudnn_split", "fa3", "tokamax")


class Worker:
    """A baseline subprocess speaking one JSON request / one JSON answer per line."""

    def __init__(self, python: str, script: str):
        self.proc = subprocess.Popen([python, os.path.join(HERE, script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, cwd=HERE)

    def ask(self, request: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as err:
            return {"error": f"worker failed: {err}"}
        return json.loads(line) if line else {"error": "worker exited"}

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def _window(mask: str) -> int | None:
    return int(mask[3:]) if mask.startswith("swa") else None


def _pairs(length: int, window: int | None, doc_len: int | None) -> float:
    """Visible query-key pairs of one row."""
    span = doc_len or length
    per_doc = sum(min(t + 1, window or span) for t in range(span))
    return per_doc * (length // span)


def _wall_ms(fn, args, iterations: int) -> float:
    """Cross-check of the kernel time: wall time per call of back-to-back calls (dispatch and launch
    gaps included, so it bounds the kernel time from above)."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iterations):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) * 1e3 / iterations


def _inputs(batch, length, heads, kv_heads, head_dim, docs, doc_ids="sorted"):
    keys = jax.random.split(jax.random.key(length), 4)
    bf16, f32 = jnp.bfloat16, jnp.float32
    q = jax.random.normal(keys[0], (batch, length, heads, head_dim), f32).astype(bf16)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), f32).astype(bf16)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, head_dim), f32).astype(bf16)
    do = jax.random.normal(keys[3], (batch, length, heads, head_dim), f32).astype(bf16)
    run = jnp.arange(length) // (length // docs)
    if doc_ids == "shuffled":  # corpus-style ids: distinct, one run each, in no order (0, 856, 703, ...)
        run = (run * 856) % 1009
    elif doc_ids == "repeated":  # ids reused across runs (0, 1, 0, 1): a different mask, no block skipping
        run = run % 2
    seg = jnp.broadcast_to(run.astype(jnp.int32), (batch, length))
    return q, k, v, do, seg


def _jax_fn(impl, window, seg, docs):
    if impl.startswith("mosaic_gpu:"):  # the kernel with a named backward (mosaic_gpu:ranged, mosaic_gpu:fused)
        import functools

        import jla.ops.mosaic_gpu.attention as kernel_module

        parts = impl.split(":")
        backward, forward = parts[1], (parts[2] if len(parts) > 2 else "")
        if len(parts) > 3 or forward not in ("", "overlap", "pingpong", "tiles"):
            raise ValueError(f"forward variant must be overlap, pingpong or tiles; got {impl}")

        def fn(q, k, v):
            config = kernel_module._Config
            if forward in ("overlap", "pingpong"):  # mosaic_gpu:<backward>:overlap|pingpong: the loop (traced inside)
                kernel_module._Config = functools.partial(config, fwd_overlap=forward == "overlap")
            elif forward == "tiles":  # the forward of one CTA per query tile instead of the persistent one
                kernel_module._Config = functools.partial(config, fwd_persistent=False)
            try:
                return (kernel_module.flash_attention(q, k, v, segment_ids=seg, sliding_window=window,
                                                      backward=backward),)
            finally:
                kernel_module._Config = config
        return fn
    if impl == "cudnn_split":
        def fn(q, k, v):
            batch, length = q.shape[:2]
            split = lambda x: x.reshape(batch * docs, length // docs, *x.shape[2:])  # noqa: E731
            out = attention(split(q), split(k), split(v), sliding_window=window, implementation="cudnn")
            return (out.reshape(q.shape),)
        return fn

    def fn(q, k, v):
        return (attention(q, k, v, segment_ids=seg, sliding_window=window, implementation=impl),)
    return fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="2048,4096,8192,16384,32768")
    parser.add_argument("--masks", default="causal,swa128,swa1024,swa4096,documents")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=16384, help="tokens per batch: B = max(1, tokens // T)")
    parser.add_argument("--docs", type=int, default=4)
    parser.add_argument("--doc-ids", choices=("sorted", "shuffled", "repeated"), default="sorted",
                        help="ids of the documents (mosaic_gpu and cudnn only; cudnn_split and fa3 take lengths): "
                             "sorted, shuffled (one run per id, unsorted) or repeated (reused across runs)")
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--fa3-python", default=None)
    parser.add_argument("--fa3-commit", default=None,
                        help="flash-attention commit the FA3 wheel was built from (recorded in the JSON)")
    parser.add_argument("--tokamax-python", default=None)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--rounds", type=int, default=2, help="interleaved rounds per shape (order alternates)")
    args = parser.parse_args()
    impls = args.impls.split(",")
    workers = {}
    if "fa3" in impls and args.fa3_python:
        workers["fa3"] = Worker(args.fa3_python, "torch_fa3.py")
    if "tokamax" in impls and args.tokamax_python:
        workers["tokamax"] = Worker(args.tokamax_python, "jax_tokamax_attention.py")
    heads, kv_heads, head_dim = args.heads, args.kv_heads, args.head_dim
    rows, versions = [], {}
    gpu_before = harness.gpu_snapshot()

    def save():
        if "fa3" in versions and args.fa3_commit:
            versions["fa3"]["commit"] = args.fa3_commit
        if "cudnn" in impls or "cudnn_split" in impls:
            from jax._src.cudnn import fused_attention_stablehlo as cudnn_sdpa

            try:
                versions["cudnn"] = dict(jax=jax.__version__, cudnn=cudnn_sdpa.check_cudnn_version())
            except Exception as err:  # noqa: BLE001 (no cuDNN): record why
                versions["cudnn"] = dict(jax=jax.__version__, cudnn=None, error=f"{type(err).__name__}: {err}"[:200])
        harness.write_json(args.out, dict(
            environment=harness.environment(), baselines=versions, gpu_before=gpu_before,
            gpu=harness.gpu_snapshot(), rows=rows))

    try:
        for length in (int(x) for x in args.lengths.split(",")):
            batch = max(1, args.tokens // length)
            q, k, v, do, seg = _inputs(batch, length, heads, kv_heads, head_dim, args.docs, args.doc_ids)
            for mask in args.masks.split(","):
                window = _window(mask)
                if window is not None and window >= length:
                    continue  # the same as causal
                docs = mask == "documents"
                doc_len = length // args.docs if docs else None
                flops = 4.0 * batch * heads * head_dim * _pairs(length, window, doc_len)
                shape_rows = {}
                jitted = {}
                for impl in impls:
                    if impl == "cudnn_split" and not docs:
                        continue
                    if impl in ("fa3", "tokamax") and impl not in workers:
                        continue
                    shape_rows[impl] = dict(T=length, B=batch, H=heads, Hkv=kv_heads, D=head_dim, mask=mask,
                                            window=window, docs=args.docs if docs else 0, impl=impl, rounds=[])
                    if docs and args.doc_ids != "sorted":
                        shape_rows[impl]["doc_ids"] = args.doc_ids
                    if impl not in workers:
                        fn = _jax_fn(impl, window, seg if docs else None, args.docs)
                        jitted[impl] = (jax.jit(fn), harness.vjp_fn(fn, (0, 1, 2)))
                for r in range(args.rounds):  # alternate the order so drift hits every implementation alike
                    for impl in (list(shape_rows) if r % 2 == 0 else list(shape_rows)[::-1]):
                        row = shape_rows[impl]
                        if "error" in row:
                            continue
                        if impl in workers:
                            answer = workers[impl].ask(dict(
                                op="time", B=batch, T=length, H=heads, Hkv=kv_heads, D=head_dim, window=window,
                                doc_lens=[[doc_len] * args.docs] * batch if docs else None,
                                iterations=args.iterations))
                            for key in ("flash_attn_3", "torch", "tokamax", "tokamax_git", "jax"):
                                if key in answer:
                                    versions.setdefault(impl, {})[key] = answer.pop(key)
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
                        if "error" in answer or not answer.get("fwd_ms") or not answer.get("fwdbwd_ms"):
                            row["error"] = answer.get("error", f"no kernel time recorded: {answer}")
                            continue
                        row["rounds"].append(answer)
                for impl, row in shape_rows.items():
                    if row["rounds"]:  # the fastest round median: the least disturbed by other jobs
                        for name in ("fwd", "fwdbwd"):
                            # Kernel-time rounds only; a worker's event-timed rounds count only when no round
                            # has a kernel time, and the cell's "<name>_timer" then says "events".
                            timed = [a for a in row["rounds"] if a.get(f"{name}_timer") != "events"] or row["rounds"]
                            if f"{name}_timer" in timed[0]:
                                row[f"{name}_timer"] = timed[0][f"{name}_timer"]
                            for key in (f"{name}_ms", f"{name}_min_ms", f"{name}_wall_ms"):
                                if all(key in a for a in timed):
                                    row[key] = min(a[key] for a in timed)
                        row["fwd_tflops"] = flops / (row["fwd_ms"] * 1e9)
                        row["fwdbwd_tflops"] = 3.5 * flops / (row["fwdbwd_ms"] * 1e9)
                    print({k: v for k, v in row.items() if k != "rounds"}, flush=True)
                    rows.append(row)
                save()
                if "fa3" in workers:  # a fresh FA3 worker per shape: after a few hundred profiler sessions a
                    # torch process records only empty captures (measured: from about the fifth shape on, on
                    # an idle GPU), and the worker would then report event time for every later shape
                    workers["fa3"].close()
                    workers["fa3"] = Worker(args.fa3_python, "torch_fa3.py")
    finally:
        for worker in workers.values():
            worker.close()
        save()


if __name__ == "__main__":
    main()
