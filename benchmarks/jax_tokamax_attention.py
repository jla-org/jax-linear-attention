# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tokamax Mosaic GPU attention baseline worker (run in a JAX environment with ``tokamax`` installed).

Started by ``bench_attention_fa3.py`` as a subprocess with the same JSON-lines protocol as
``torch_fa3.py``: a ``{"op": "time", "B", "T", "H", "Hkv", "D", "window", "doc_lens",
"iterations"}`` request is answered with the median CUPTI kernel time (``harness.median_kernel_ms``)
of ``tokamax.dot_product_attention(..., implementation="mosaic")`` forward and forward + backward.
``window`` is passed as ``local_window_size=(window - 1, 0)`` with ``is_causal=True``; packed
documents (``doc_lens``) become the dense boolean mask (Tokamax has no segment-id argument).
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402


def _call(tokamax, q, k, v, window, mask):
    length = q.shape[1]
    local = None if window is None or window >= length else (window - 1, 0)
    return tokamax.dot_product_attention(q, k, v, mask=mask, is_causal=True, local_window_size=local,
                                         implementation="mosaic")


_COMPILED: dict[str, tuple] = {}  # request (without iterations) -> inputs and jitted functions


def _time(req: dict) -> dict:
    key = json.dumps({k: v for k, v in req.items() if k != "iterations"}, sort_keys=True)
    if key not in _COMPILED:
        _COMPILED[key] = _build(req)
    (q, k, v, do), fwd_fn, both_fn = _COMPILED[key]
    iterations = int(req.get("iterations", 30))
    fwd = harness.median_kernel_ms(fwd_fn, q, k, v, iterations=iterations)
    both = harness.median_kernel_ms(both_fn, q, k, v, do, iterations=iterations)
    return dict(fwd_ms=fwd["median_ms"], fwd_min_ms=fwd["min_ms"], fwdbwd_ms=both["median_ms"],
                fwdbwd_min_ms=both["min_ms"], fwd_wall_ms=_wall_ms(fwd_fn, (q, k, v), iterations),
                fwdbwd_wall_ms=_wall_ms(both_fn, (q, k, v, do), iterations))


def _wall_ms(fn, args, iterations: int) -> float:
    """Wall time per call of back-to-back calls (an upper bound on the kernel time)."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iterations):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) * 1e3 / iterations


def _build(req: dict):
    import tokamax

    batch, length, heads, kv_heads, head_dim = (req[n] for n in ("B", "T", "H", "Hkv", "D"))
    keys = jax.random.split(jax.random.key(length), 4)
    q = jax.random.normal(keys[0], (batch, length, heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    do = jax.random.normal(keys[3], (batch, length, heads, head_dim), jnp.float32).astype(jnp.bfloat16)
    mask = None
    if req.get("doc_lens") is not None:
        seg = np.stack([np.repeat(np.arange(len(row)), row) for row in req["doc_lens"]]).astype(np.int32)
        seg = jnp.asarray(seg)
        mask = (seg[:, :, None] == seg[:, None, :])[:, None]
    window = req.get("window")

    def fn(q, k, v):
        return (_call(tokamax, q, k, v, window, mask),)

    return (q, k, v, do), jax.jit(fn), harness.vjp_fn(fn, (0, 1, 2))


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        try:
            result = _time(req)
        except Exception as err:  # report the failure, keep serving
            result = {"error": harness.error_summary(f"{type(err).__name__}: {err}")}
        try:
            import tokamax

            result["tokamax"] = getattr(tokamax, "__version__", None)
            result["tokamax_git"] = getattr(tokamax, "__git_revision__", None)
        except ImportError as err:
            result["tokamax"] = f"not importable: {err}"
        result["jax"] = jax.__version__
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
