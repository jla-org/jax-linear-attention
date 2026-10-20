# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the vector-gate kernels (KDA, Gated DeltaNet-2) against FLA and XLA.

    python benchmarks/bench_kda.py --kind kda --torch-python VENV/bin/python
    python benchmarks/bench_kda.py --kind gdn2

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: ``jla.ops.linear_attention(kind, ..., impl="mosaic_gpu")``;
* ``mosaic_gpu_fused``: the same kernels with the solve inside ``vg_intra`` (``fuse_solve=True``, opt-in);
* ``chunked``: ``impl="chunked"`` with ``precision=None`` (plain XLA);
* ``fla`` (KDA): FLA's Triton ``chunk_kda`` (``torch_kda.py``, in a torch environment; FLA main c6e4f00 from
  17 Oct 2026 on, 0.5.2 before), with FLA's FlashKDA and TileLang backends switched off;
* ``fla_default`` (KDA): the same call with FLA's default dispatch (``--impl-python`` points it at an environment
  with FlashKDA and TileLang installed, so FLA can hand the forward or the WY backward to them);
* ``flashkda`` (KDA, ``--contract raw``, forward only): MoonshotAI's FlashKDA ``flash_kda.fwd``;
* ``mosaic_gpu_kda`` (GDN-2): our KDA kernels on the same shape (GDN-2 has no FLA baseline we
  may use; its kernels are KDA's with the channel-wise gates, so it should cost about the same).

``--contract raw --modes fwd`` (KDA) times inference on FlashKDA's input contract (raw ``q, k``,
raw gate with ``A_log``, ``dt_bias`` and a lower bound, beta logits) for ``mosaic_gpu``, FLA's
Triton ``chunk_kda`` with its in-kernel gate, norm and sigmoid, and ``flashkda`` (MoonshotAI's
FlashKDA ``flash_kda.fwd``, forward only, K = V = 128).

MaxText has no KDA or GDN-2 implementation. Metric: median CUDA kernel time per call (see
``harness.py``); ``fwdbwd`` is one forward plus the backward for a fixed output cotangent.
Results go to ``benchmarks/results/{kda,gdn2,kda_raw}_<date>.json``.
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

from jla import ops  # noqa: E402
from jla.ops import reference as ref  # noqa: E402
from jla.ops.mosaic_gpu import vector_gate_ops as vgo  # noqa: E402

SHAPES = [(1, 4096, 8, 128, 128), (4, 4096, 16, 128, 128), (8, 2048, 6, 128, 128), (1, 16384, 16, 128, 128),
          (8, 2048, 32, 128, 128), (4, 4096, 16, 64, 64)]


def inputs(shape) -> dict[str, jax.Array]:
    """bf16 ``q, k, v`` (unit-norm ``k``), per-channel ``g = logsigmoid(N(0, 1))`` and sigmoid gates."""
    batch, seq_len, heads, key_dim, value_dim = shape
    ks = jax.random.split(jax.random.key(0), 8)
    f32, bf16 = jnp.float32, jnp.bfloat16
    return dict(
        q=jax.random.normal(ks[0], (batch, seq_len, heads, key_dim), f32).astype(bf16),
        k=ref.l2norm(jax.random.normal(ks[1], (batch, seq_len, heads, key_dim), f32)).astype(bf16),
        v=jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), f32).astype(bf16),
        g=jax.nn.log_sigmoid(jax.random.normal(ks[3], (batch, seq_len, heads, key_dim), f32)),
        beta=jax.nn.sigmoid(jax.random.normal(ks[4], (batch, seq_len, heads), f32)),
        b=jax.nn.sigmoid(jax.random.normal(ks[5], (batch, seq_len, heads, key_dim), f32)),
        w=jax.nn.sigmoid(jax.random.normal(ks[6], (batch, seq_len, heads, value_dim), f32)),
        do=jax.random.normal(ks[7], (batch, seq_len, heads, value_dim), f32).astype(bf16),
    )


def fused_solve_pipeline(q, k, v, g, gates, *, l2norm: bool = False):
    """``mosaic_gpu_fused``: the call of ``impl="mosaic_gpu"`` with the solve inside ``vg_intra``
    (``fuse_solve=True``, opt-in) instead of the separate ``vg_solve`` kernel."""
    return vgo.chunked_vector_gate(q, k, v, g, tuple(gates), scale=q.shape[-1] ** -0.5, initial_state=None,
                                   l2norm=l2norm, fuse_solve=True)


def doc_cu_seqlens(seq_len: int, docs: int) -> list[int]:
    """``cu_seqlens`` of ``docs`` packed documents with fixed pseudo-random lengths (at least 16 tokens)."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = np.sort(rng.choice(np.arange(32, seq_len - 31, 32), docs - 1, replace=False))
    cuts = base + rng.integers(0, 16, docs - 1)  # off the chunk grid; neighbours stay >= 17 tokens apart
    return [0, *(int(c) for c in cuts), seq_len]


def segment_ids_of(cu: list[int], seq_len: int) -> jax.Array:
    """``[1, T]`` int32 document ids of ``cu_seqlens``."""
    return jnp.searchsorted(jnp.asarray(cu[1:]), jnp.arange(seq_len), side="right").astype(jnp.int32)[None]


def jax_impls(kind: str, shape, impls: list[str], cu: list[int] | None = None):
    """``name -> (fwd_fn, fwdbwd_fn, args, cotangent)`` for the JAX implementations (``cu``: packed
    documents, passed as ``segment_ids``)."""
    x = inputs(shape)
    seg = None if cu is None else segment_ids_of(cu, shape[1])
    out = {}
    for name in impls:
        if name not in ("mosaic_gpu", "mosaic_gpu_fused", "chunked", "mosaic_gpu_kda"):
            continue
        if seg is not None and name == "mosaic_gpu_fused":
            continue  # fused_solve_pipeline takes no segment_ids
        op_kind = "kda" if name == "mosaic_gpu_kda" else kind
        impl = {"chunked": "chunked", "mosaic_gpu_fused": "fused"}.get(name, "mosaic_gpu")
        gate_names = ("g", "beta") if op_kind == "kda" else ("g", "b", "w")

        def fn(q, k, v, *gates, op_kind=op_kind, impl=impl, gate_names=gate_names):
            if impl == "fused":
                return fused_solve_pipeline(q, k, v, gates[0], gates[1:])
            kw = dict(zip(gate_names, gates))
            if impl == "chunked":
                kw["precision"] = None
            if seg is not None:
                kw["segment_ids"] = seg
            return ops.linear_attention(op_kind, q, k, v, impl=impl, **kw)

        args = (x["q"], x["k"], x["v"], *(x[n] for n in gate_names))
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        out[name] = (fwd, harness.vjp_fn(fn, tuple(range(len(args)))), args, x["do"])
    return out


RAW_SHAPES = [(1, 4096, 8, 128, 128), (4, 4096, 16, 128, 128), (8, 2048, 6, 128, 128), (1, 16384, 16, 128, 128),
              (8, 2048, 32, 128, 128)]
LOWER_BOUND = -5.0  # the safe-gate bound of FlashKDA's contract (torch_kda.py)


def raw_jax_impls(shape, cu: list[int] | None = None) -> dict:
    """``mosaic_gpu`` on FlashKDA's inference contract (forward only): raw ``q, k``, raw gate with
    ``A_log``, ``dt_bias`` and the -5 lower bound, beta logits. The L2 norm (``use_qk_l2norm``), the
    gate ``lower_bound sigmoid(exp(A_log) (g + dt_bias))`` and the beta sigmoid run in XLA and are
    timed with the kernels. ``cu``: packed documents (``segment_ids``; not for the fused solve)."""
    batch, seq_len, heads, key_dim, value_dim = shape
    seg = None if cu is None else segment_ids_of(cu, seq_len)
    ks = jax.random.split(jax.random.key(0), 6)
    f32, bf16 = jnp.float32, jnp.bfloat16
    args = (
        jax.random.normal(ks[0], (batch, seq_len, heads, key_dim)).astype(bf16),
        jax.random.normal(ks[1], (batch, seq_len, heads, key_dim)).astype(bf16),
        jax.random.normal(ks[2], (batch, seq_len, heads, value_dim)).astype(bf16),
        jax.random.normal(ks[3], (batch, seq_len, heads, key_dim)).astype(bf16),
        jax.random.normal(ks[4], (batch, seq_len, heads)).astype(bf16),
        jnp.log(jnp.linspace(1.0, 16.0, heads)),
        0.1 * jax.random.normal(ks[5], (heads, key_dim)),
    )

    def fn(q, k, v, g, beta, a_log, dt_bias, fused=False):
        gate = LOWER_BOUND * jax.nn.sigmoid(jnp.exp(a_log)[:, None] * (g.astype(f32) + dt_bias))
        beta = jax.nn.sigmoid(beta.astype(f32))
        if fused:
            return fused_solve_pipeline(q, k, v, gate, (beta,), l2norm=True)
        return ops.linear_attention("kda", q, k, v, g=gate, beta=beta, use_qk_l2norm=True, impl="mosaic_gpu",
                                    segment_ids=seg)

    out = {"mosaic_gpu": (jax.jit(lambda *a: fn(*a)[0]), None, args, None)}
    if seg is None:
        out["mosaic_gpu_fused"] = (jax.jit(lambda *a: fn(*a, fused=True)[0]), None, args, None)
    return out


def median_kernel_ms(fn, *args, iterations: int) -> dict:
    """``harness.median_kernel_ms`` with one call per CUPTI measurement: ``profiler.measure`` keeps
    the outputs of every call alive, and 30 sets of KDA gradients do not fit the larger shapes."""
    from jax.experimental.mosaic.gpu import profiler

    for _ in range(harness.WARMUP):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iterations):
        out, t = profiler.measure(fn, iterations=1)(*args)
        jax.block_until_ready(out)
        del out
        times.append(t)
    times = [t for t in times if t is not None]
    return dict(median_ms=statistics.median(times), min_ms=min(times), iterations=len(times))


def run_torch(torch_python: str, shape, mode: str, iterations: int, impl: str = "fla", contract: str = "plain",
              cu: list[int] | None = None) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_kda.py")
    cmd = [torch_python, script, "--shape", ",".join(map(str, shape)), "--mode", mode, "--iterations", str(iterations),
           "--impl", impl, "--contract", contract]
    if cu is not None:
        cmd += ["--cu", ",".join(map(str, cu))]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main(default_kind: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("kda", "gdn2"), default=default_kind, required=default_kind is None)
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA (KDA)")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default=None, help="comma-separated subset of implementations")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--contract", choices=("plain", "raw"), default="plain",
                        help="raw: FlashKDA's inference contract (KDA, forward only, K = V = 128)")
    parser.add_argument("--breakdown", action="store_true",
                        help="also record the per-kernel medians of each JAX forward (kernels_ms)")
    parser.add_argument("--impl-python", default=None,
                        help="per-implementation torch pythons, e.g. "
                             "fla_default=VENV/bin/python,flashkda=VENV/bin/python (default: --torch-python)")
    parser.add_argument("--docs", type=int, default=0,
                        help="pack this many documents of pseudo-random lengths into each row (B = 1): segment_ids "
                             "for jla, cu_seqlens for FLA and FlashKDA")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw = args.contract == "raw"
    if raw and (args.kind != "kda" or args.modes != "fwd"):
        raise SystemExit("--contract raw is KDA forward only: pass --kind kda --modes fwd")
    shapes = RAW_SHAPES if raw else SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    pythons = dict(item.split("=", 1) for item in args.impl_python.split(",")) if args.impl_python else {}
    if args.docs and any(shape[0] != 1 for shape in shapes):
        raise SystemExit("--docs packs documents into one row: every shape needs B = 1")
    if args.impls:
        impl_names = args.impls.split(",")
    elif raw:
        impl_names = ["mosaic_gpu", "mosaic_gpu_fused", "fla", "flashkda"]
    elif args.kind == "kda":
        impl_names = ["mosaic_gpu", "mosaic_gpu_fused", "fla", "chunked"]
    else:
        impl_names = ["mosaic_gpu", "mosaic_gpu_fused", "mosaic_gpu_kda", "chunked"]
    modes = args.modes.split(",")

    results = dict(kind=args.kind, contract=args.contract, environment=harness.environment(),
                   metric="median CUDA kernel ms per call", docs=args.docs or None,
                   rounds=args.rounds, iterations=args.iterations, entries=[])
    for shape in shapes:
        cu = doc_cu_seqlens(shape[1], args.docs) if args.docs else None
        jimpls = raw_jax_impls(shape, cu) if raw else jax_impls(args.kind, shape, impl_names, cu)
        for mode in modes:
            for rnd in range(args.rounds):
                for name in impl_names:
                    before = harness.gpu_snapshot()
                    t0 = time.time()
                    if name in jimpls:
                        fwd, bwd, fargs, do = jimpls[name]
                        try:
                            if mode == "fwd":
                                r = median_kernel_ms(fwd, *fargs, iterations=args.iterations)
                                if args.breakdown:
                                    try:
                                        r["kernels_ms"] = harness.kernel_breakdown_ms(fwd, *fargs)
                                    except Exception as err:  # noqa: BLE001  (keep the timing)
                                        r["kernels_error"] = f"{type(err).__name__}: {str(err)[:300]}"
                            else:
                                r = median_kernel_ms(bwd, *fargs, do, iterations=args.iterations)
                        except Exception as err:  # noqa: BLE001  (record and continue)
                            r = dict(error=f"{type(err).__name__}: {str(err)[:500]}")
                    elif name in ("fla", "fla_default", "flashkda") and pythons.get(name, args.torch_python):
                        if name == "flashkda" and mode != "fwd":
                            continue  # FlashKDA is forward only
                        r = run_torch(pythons.get(name, args.torch_python), shape, mode, args.iterations, name,
                                      args.contract, cu)
                    else:
                        continue
                    r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind")}
                    entry = dict(shape=shape, mode=mode, impl=name, round=rnd, wall_s=round(time.time() - t0, 1),
                                 gpu_before=before, gpu_after=harness.gpu_snapshot(), **r)
                    if cu is not None:
                        entry["cu_seqlens"] = cu
                    results["entries"].append(entry)
                    print(json.dumps({k: entry[k] for k in ("shape", "mode", "impl", "round")}
                                     | {k: entry.get(k) for k in ("median_ms", "error")}), flush=True)
    date = time.strftime("%Y%m%d_%H%M")
    stem = f"{args.kind}_raw" if raw else args.kind
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{stem}_{date}.json")
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
