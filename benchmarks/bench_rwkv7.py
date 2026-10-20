# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark RWKV-7 (``kind="rwkv7"``): the Mosaic GPU state passes against FLA Triton and XLA.

    python benchmarks/bench_rwkv7.py --torch-python VENV/bin/python

Implementations, measured round-robin on the same GPU (``rounds`` times each):

* ``mosaic_gpu``: ``jla.ops.linear_attention("rwkv7", ..., impl="mosaic_gpu")`` (default
  numerics: bf16 operands for the off-diagonal score products, f32 for the other XLA products);
* ``mosaic_gpu_highest``: the same with ``precision=HIGHEST`` (every XLA product in f32);
* ``mosaic_gpu_layer``: as the RWKV-7 layers call the op (``jla.layers.rwkv7._OP_OPTIONS``:
  ``lower_bound=-0.61``, the bounded mode: the four score matrices from one product per chunk, its
  products as split-bf16 tensor-core products, ``BF16X3``; RWKV-7's ``w >= -0.6065``, also after bf16
  rounding ``w >= -0.6075``);
* ``mosaic_gpu_bounded_highest``: the bounded mode with ``precision=HIGHEST`` (f32 products, the
  XLA path); ``mosaic_gpu_bounded`` is ``lower_bound=-0.61`` alone, the same
  call as ``mosaic_gpu_layer``;
* ``mosaic_gpu_layer_xla_bwd`` / ``mosaic_gpu_layer_kernel_bwd``: the layers' call with the head-dim-64
  kernel path's backward chosen explicitly (``rwkv7._run(..., kernel_backward=...)``): the XLA VJP of the
  forward kernel's operands, or the ``rwkv7_bwd`` Mosaic GPU kernel;
* ``chunked``: ``impl="chunked"`` (``chunk_size=16``, ``precision=DEFAULT``), our previous path;
* ``fla``: FLA ``chunk_rwkv7`` with its defaults (exact gates, ``chunk_size=16``; ``torch_rwkv7.py``
  in the ``--torch-python`` environment, FLA main's Triton kernels);
* ``fla_layer``: ``chunk_rwkv7`` as FLA's ``RWKV7Attention`` calls it (``safe_gate=True``,
  ``lower_bound=-0.6065``, ``chunk_size=64``; the match for ``mosaic_gpu_layer``);
* ``fla_safe``: ``safe_gate=True`` with FLA's default chunk (16);
* ``fla_cs32``, ``fla_cs64``: ``fla`` with ``chunk_size=32`` / ``64``;
* ``full_fla``, ``full_fla_layer``, ``full_fla_safe``: the same three in the ``--torch-python-full``
  environment (FLA main with its optional backends, so its default dispatch may pick TileLang).

A shape ``B,T,H,K,V,N`` with a sixth entry packs ``N`` documents of fixed pseudo-random lengths into
each row (``B = 1``): our ``segment_ids`` against FLA's ``cu_seqlens``.

``--errors B,T,H,K,V`` measures errors instead of time; ``--w-const L`` then sets ``w = L`` on every
token, and ``mosaic_gpu_bounded_min`` is the bounded mode at ``lower_bound = MIN_LOWER_BOUND``.

Inputs are RWKV-7's: bf16 ``r, k, v``, the log decay ``w = -0.6065 sigmoid(N(0, 1))``,
``a = -kk``, ``b = kk * sigmoid(N(0, 1))`` with ``kk`` unit-norm (all bf16, as an FLA model passes
them). Metric: median CUDA kernel time per call (``bench_kda.median_kernel_ms``); ``fwdbwd`` is one
forward plus the backward for a fixed output cotangent. Results go to
``benchmarks/results/rwkv7_<date>.json``.
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
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_kda  # noqa: E402
import harness  # noqa: E402

from jla import ops  # noqa: E402
from jla.layers.rwkv7 import _OP_OPTIONS as _LAYER_OPTIONS  # noqa: E402
from jla.ops.mosaic_gpu.rwkv7 import MIN_LOWER_BOUND  # noqa: E402

SHAPES = [(1, 4096, 12, 64, 64), (8, 4096, 12, 64, 64), (2, 4096, 32, 64, 64)]
# Below RWKV-7's log decay -0.6065 sigmoid(.) including its bf16 rounding (-0.607421875).
RWKV7_LOWER_BOUND = -0.61


def inputs(shape) -> dict[str, jax.Array]:
    batch, seq_len, heads, key_dim, value_dim = shape
    ks = jax.random.split(jax.random.key(0), 7)
    f32, bf16 = jnp.float32, jnp.bfloat16
    dims = (batch, seq_len, heads, key_dim)
    kk = jax.random.normal(ks[4], dims, f32)
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    return dict(
        q=jax.random.normal(ks[0], dims, f32).astype(bf16),
        k=jax.random.normal(ks[1], dims, f32).astype(bf16),
        v=jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), f32).astype(bf16),
        w=(-0.6065306597126334 * jax.nn.sigmoid(jax.random.normal(ks[3], dims, f32))).astype(bf16),
        a=(-kk).astype(bf16),
        b=(kk * jax.nn.sigmoid(jax.random.normal(ks[5], dims, f32))).astype(bf16),
        do=jax.random.normal(ks[6], (batch, seq_len, heads, value_dim), f32).astype(bf16),
    )


_OPTIONS = {
    "mosaic_gpu": ("mosaic_gpu", {}),
    "mosaic_gpu_highest": ("mosaic_gpu", {"precision": lax.Precision.HIGHEST}),
    "mosaic_gpu_bounded": ("mosaic_gpu", {"lower_bound": RWKV7_LOWER_BOUND}),  # the same call as mosaic_gpu_layer
    "mosaic_gpu_bounded_min": ("mosaic_gpu", {"lower_bound": MIN_LOWER_BOUND}),
    # what the RWKV-7 layers pass: the bounded mode with its default (BF16X3) products
    "mosaic_gpu_layer": ("mosaic_gpu", dict(_LAYER_OPTIONS)),
    "mosaic_gpu_bounded_highest": ("mosaic_gpu", {"lower_bound": RWKV7_LOWER_BOUND,
                                                  "precision": lax.Precision.HIGHEST}),
    "chunked": ("chunked", {"chunk_size": 16, "precision": lax.Precision.DEFAULT}),
    # the layers' call with each backward of the bounded mode's 64-dim kernel path: the XLA VJP of the
    # forward kernel's operands and the rwkv7_bwd kernel
    "mosaic_gpu_layer_xla_bwd": ("mosaic_gpu_xla_bwd", dict(_LAYER_OPTIONS)),
    "mosaic_gpu_layer_kernel_bwd": ("mosaic_gpu_kernel_bwd", dict(_LAYER_OPTIONS)),
}


def _call(impl: str, kw: dict, q, k, v, w, a, b, segment_ids, **options):
    if impl in ("mosaic_gpu_xla_bwd", "mosaic_gpu_kernel_bwd"):  # the kernel module's entry after validation
        from jla.ops.mosaic_gpu import rwkv7 as mg

        options = dict(initial_state=None, output_final_state=False) | options
        return mg._run(q, k, v, w, a, b, scale=1.0, segment_ids=segment_ids, precision=kw.get("precision"),
                       lower_bound=kw["lower_bound"], kernel_backward=impl == "mosaic_gpu_kernel_bwd", **options)
    return ops.linear_attention("rwkv7", q, k, v, w=w, a=a, b=b, scale=1.0, impl=impl, segment_ids=segment_ids,
                                **kw, **options)


def doc_bounds(seq_len: int, docs: int) -> list[int]:
    """``cu_seqlens`` of ``docs`` documents of fixed pseudo-random lengths that fill ``seq_len``."""
    import numpy as np

    cuts = np.sort(np.random.default_rng(seq_len + docs).choice(np.arange(1, seq_len), docs - 1, replace=False))
    return [0, *cuts.tolist(), seq_len]


def jax_impls(shape, impls: list[str]):
    """``name -> (fwd_fn, fwdbwd_fn, args, cotangent)`` for the JAX implementations."""
    x = inputs(shape[:5])
    segment_ids = None
    if len(shape) > 5:
        bounds = doc_bounds(shape[1], shape[5])
        segment_ids = jnp.concatenate([jnp.full((e - s,), i, jnp.int32)
                                       for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))])[None]
    out = {}
    for name in impls:
        if name not in _OPTIONS:
            continue
        impl, kw = _OPTIONS[name]

        def fn(q, k, v, w, a, b, impl=impl, kw=kw):
            return _call(impl, kw, q, k, v, w, a, b, segment_ids)

        args = tuple(x[n] for n in ("q", "k", "v", "w", "a", "b"))
        fwd = jax.jit(lambda *a, fn=fn: fn(*a)[0])
        out[name] = (fwd, harness.vjp_fn(fn, tuple(range(len(args)))), args, x["do"])
    return out


def errors(shape, impls: list[str], w_const: float | None = None) -> list[dict]:
    """Max error / max |reference| of ``o``, the final state and every gradient, per JAX implementation.

    References: the token-by-token f32 oracle (``o``, final state) and autodiff of the f32
    chunked op at ``HIGHEST`` (gradients), with an initial state and a final-state cotangent.
    ``w_const`` sets the log decay of every token (the bounded mode's worst case at its bound).
    """
    from jla.ops.rwkv7 import rwkv7 as oracle
    from jla.ops.rwkv7 import rwkv7_chunked

    x = inputs(shape)
    if w_const is not None:
        x["w"] = jnp.full_like(x["w"], w_const)
    batch, _, heads, key_dim, value_dim = shape
    h0 = 0.3 * jax.random.normal(jax.random.key(1), (batch, heads, key_dim, value_dim), jnp.float32)
    dht = jax.random.normal(jax.random.key(2), (batch, heads, key_dim, value_dim), jnp.float32)
    args = tuple(x[n] for n in ("q", "k", "v", "w", "a", "b"))
    f32 = tuple(t.astype(jnp.float32) for t in args)

    def loss(fn):
        def run(*a):
            o, final = fn(*a[:6], initial_state=a[6], output_final_state=True)
            return jnp.sum(o.astype(jnp.float32) * x["do"].astype(jnp.float32)) + jnp.sum(final * dht)
        return jax.jit(jax.grad(run, tuple(range(7))))

    def rel(got, want):
        return float(jnp.max(jnp.abs(got.astype(jnp.float32) - want)) / jnp.maximum(jnp.max(jnp.abs(want)), 1e-6))

    o_ref, final_ref = jax.jit(lambda *a: oracle(*a, initial_state=h0, output_final_state=True, scale=1.0))(*f32)
    g_ref = loss(lambda *a, **kw: rwkv7_chunked(*a, chunk_size=16, scale=1.0, **kw))(*f32, h0)
    out = []
    for name in impls:
        if name not in _OPTIONS:
            continue
        impl, kw = _OPTIONS[name]

        def fn(*a, impl=impl, kw=kw, **options):
            return _call(impl, kw, *a, None, **options)

        o, final = jax.jit(lambda *a, fn=fn: fn(*a, initial_state=h0, output_final_state=True))(*args)
        grads = loss(fn)(*args, h0)
        entry = dict(shape=shape, impl=name, w_const=w_const, o=rel(o, o_ref), final=rel(final, final_ref))
        entry |= {f"d{n}": rel(g, r) for n, g, r in zip(("r", "k", "v", "w", "a", "b", "h0"), grads, g_ref)}
        out.append(entry)
        print(json.dumps(entry), flush=True)
    return out


# name -> (environment, safe_gate, chunk_size, lower_bound)
_FLA_LAYER = (True, 64, -0.6065306597126334)  # fla/layers/rwkv7.py RWKV7Attention
_FLA = {
    "fla": ("main", False, None, None), "fla_layer": ("main", *_FLA_LAYER), "fla_safe": ("main", True, None, None),
    "fla_cs32": ("main", False, 32, None), "fla_cs64": ("main", False, 64, None),
    "full_fla": ("full", False, None, None), "full_fla_layer": ("full", *_FLA_LAYER),
    "full_fla_safe": ("full", True, None, None),
}


def run_torch(torch_python: str, shape, mode: str, iterations: int, safe_gate: bool = False,
              chunk_size: int | None = None, lower_bound: float | None = None) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_rwkv7.py")
    cmd = [torch_python, script, "--shape", ",".join(map(str, shape[:5])), "--mode", mode,
           "--iterations", str(iterations)]
    cmd += ["--safe-gate"] if safe_gate else []
    cmd += ["--chunk-size", str(chunk_size)] if chunk_size else []
    cmd += ["--lower-bound", str(lower_bound)] if lower_bound is not None else []
    cmd += ["--cu-seqlens", ",".join(map(str, doc_bounds(shape[1], shape[5])))] if len(shape) > 5 else []
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return dict(error=harness.error_summary(proc.stderr or proc.stdout))
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-python", default=None, help="python of an environment with torch + FLA")
    parser.add_argument("--torch-python-full", default=None,
                        help="python of an environment with FLA and its optional backends (the full_* rows)")
    parser.add_argument("--shapes", default=None, help="semicolon-separated B,T,H,K,V list")
    parser.add_argument("--impls", default="mosaic_gpu,mosaic_gpu_layer,mosaic_gpu_highest,fla,fla_layer,chunked")
    parser.add_argument("--modes", default="fwd,fwdbwd")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    parser.add_argument("--breakdown", action="store_true", help="also record per-kernel times of mosaic_gpu")
    parser.add_argument("--breakdown-impl", default="mosaic_gpu", help="the implementation --breakdown profiles")
    parser.add_argument("--out", default=None)
    parser.add_argument("--errors", default=None, metavar="B,T,H,K,V",
                        help="only measure each JAX implementation's error against the f32 references at this shape")
    parser.add_argument("--w-const", type=float, default=None, help="with --errors: the log decay of every token")
    args = parser.parse_args()
    if args.errors:
        shape = tuple(int(x) for x in args.errors.split(","))
        result = dict(kind="rwkv7", environment=harness.environment(), metric="max |error| / max |reference|",
                      errors=errors(shape, args.impls.split(","), args.w_const))
        if args.out:
            harness.write_json(args.out, result)
        return

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    impl_names = args.impls.split(",")
    modes = args.modes.split(",")
    results = dict(kind="rwkv7", environment=harness.environment(), metric="median CUDA kernel ms per call",
                   rounds=args.rounds, iterations=args.iterations, entries=[], breakdown=[],
                   # the JAX memory fraction the run had (fwdbwd at the 1.3B layer shape needs more than 0.25)
                   mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))
    for shape in shapes:
        jimpls = jax_impls(shape, impl_names)
        for mode in modes:
            for rnd in range(args.rounds):
                for name in impl_names:
                    before = harness.gpu_snapshot()
                    t0 = time.time()
                    if name in jimpls:
                        fwd, bwd, fargs, do = jimpls[name]
                        try:
                            if mode == "fwd":
                                r = bench_kda.median_kernel_ms(fwd, *fargs, iterations=args.iterations)
                            else:
                                r = bench_kda.median_kernel_ms(bwd, *fargs, do, iterations=args.iterations)
                        except Exception as err:  # noqa: BLE001  (record and continue)
                            r = dict(error=f"{type(err).__name__}: {str(err)[:500]}")
                    elif name in _FLA and (args.torch_python if _FLA[name][0] == "main" else args.torch_python_full):
                        env, safe_gate, chunk_size, lower_bound = _FLA[name]
                        python = args.torch_python if env == "main" else args.torch_python_full
                        r = run_torch(python, shape, mode, args.iterations, safe_gate=safe_gate,
                                      chunk_size=chunk_size, lower_bound=lower_bound)
                    else:
                        continue
                    r = {key: val for key, val in r.items() if key not in ("shape", "mode", "kind")}
                    entry = dict(shape=shape, mode=mode, impl=name, round=rnd, wall_s=round(time.time() - t0, 1),
                                 gpu_before=before, gpu_after=harness.gpu_snapshot(), **r)
                    results["entries"].append(entry)
                    print(json.dumps({k: entry[k] for k in ("shape", "mode", "impl", "round")}
                                     | {k: entry.get(k) for k in ("median_ms", "error")}), flush=True)
            if args.breakdown and args.breakdown_impl in jimpls:
                fwd, bwd, fargs, do = jimpls[args.breakdown_impl]
                fn, fn_args = (fwd, fargs) if mode == "fwd" else (bwd, (*fargs, do))
                per = harness.kernel_breakdown_ms(fn, *fn_args)
                results["breakdown"].append(dict(shape=shape, mode=mode, impl=args.breakdown_impl, kernels_ms=per))
                print(json.dumps(dict(shape=shape, mode=mode, top=dict(list(per.items())[:8]))), flush=True)
    date = time.strftime("%Y%m%d_%H%M")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"rwkv7_{date}.json")
    harness.write_json(out, results)
    bench_kda.summarize(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
