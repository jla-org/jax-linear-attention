# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the short conv and the norms of the 130M mixers spend their time.

    python benchmarks/bench_conv_norm.py --json results.json [--iterations 30]

Times forward + backward (CUPTI kernel time, ``benchmarks/harness.py``; gradients of the
input and every parameter) on bfloat16 ``[8, 2048, C]``, with the per-kernel breakdown
and the HBM floor (every tensor read or written once, at 3 TB/s):

* ``ShortConv`` (kernel 4, bias, SiLU) as the layers run it (autodiff of
  ``jla.layers.conv``), at C = 1536 (Mamba-2's x) and 256 (Mamba-2's B/C);
* two alternatives that were measured and not adopted: cuDNN's grouped (depthwise)
  convolution through ``lax.conv_general_dilated``, and a hand-written VJP that forms
  the pre-activation gradient once (``lax.optimization_barrier``) before the weight
  reductions and the shifted ``dx`` sum;
* ``rms_norm`` over 768 (the block norm) and Mamba-2's ``GatedRMSNorm`` over 1536
  (``norm_before_gate=False``, one group).

``nvidia-smi`` utilisation of the visible GPU is recorded before and after.
"""

from __future__ import annotations

import argparse
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from jla.layers.conv import ShortConv  # noqa: E402
from jla.layers.norm import GatedRMSNorm, rms_norm  # noqa: E402

HBM_BYTES_PER_S = 3.0e12


def _pre(x_ext: jax.Array, w: jax.Array, b: jax.Array, length: int) -> jax.Array:
    pre = jnp.zeros(x_ext.shape[:1] + (length,) + x_ext.shape[2:], jnp.float32) + b
    for i in range(w.shape[1]):
        pre = pre + x_ext[:, i:i + length].astype(jnp.float32) * w[:, i]
    return pre


@jax.custom_vjp
def _conv_hand_vjp(x, w, b):
    """``silu(b + causal depthwise conv(x, w))`` without segments, as ``ShortConv``."""
    x_ext = jnp.pad(x, ((0, 0), (w.shape[1] - 1, 0), (0, 0)))
    return jax.nn.silu(_pre(x_ext, w.astype(jnp.float32), b.astype(jnp.float32), x.shape[1])).astype(x.dtype)


def _conv_hand_fwd(x, w, b):
    return _conv_hand_vjp(x, w, b), (x, w, b)


def _conv_hand_bwd(res, dy):
    x, w, b = res
    width, length = w.shape[1], x.shape[1]
    w32 = w.astype(jnp.float32)
    x_ext = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
    pre = _pre(x_ext, w32, b.astype(jnp.float32), length)
    s = jax.nn.sigmoid(pre)
    dpre = lax.optimization_barrier(dy.astype(jnp.float32) * s * (1.0 + pre * (1.0 - s)))
    dw = jnp.stack([jnp.sum(dpre * x_ext[:, i:i + length].astype(jnp.float32), axis=(0, 1))
                    for i in range(width)], axis=1)
    db = jnp.sum(dpre, axis=(0, 1))
    dpre_ext = jnp.pad(dpre, ((0, 0), (0, width - 1), (0, 0)))
    dx = sum(dpre_ext[:, width - 1 - i:width - 1 - i + length] * w32[:, i] for i in range(width))
    return dx.astype(x.dtype), dw.astype(w.dtype), db.astype(b.dtype)


_conv_hand_vjp.defvjp(_conv_hand_fwd, _conv_hand_bwd)


def _vjp(f):
    def run(params, x, ct):
        y, pullback = jax.vjp(f, params, x)
        return y, pullback(ct)

    return jax.jit(run)


def _time(fn, *args, iterations: int, breakdown: bool = True) -> dict:
    out = harness.median_kernel_ms(fn, *args, iterations=iterations)
    if breakdown:
        out["kernels"] = {k: round(v, 4) for k, v in list(harness.kernel_breakdown_ms(fn, *args).items())[:8]}
    return out


def _floor_ms(tensors: int, rows: int, cols: int) -> float:
    return round(tensors * rows * cols * 2 / HBM_BYTES_PER_S * 1e3, 4)


def conv_rows(batch: int, seq_len: int, channels: int, iterations: int) -> dict:
    conv = ShortConv(channels, 4, activation=True, use_bias=True, init="uniform", key=jax.random.key(0))
    params, static = eqx.partition(conv, eqx.is_inexact_array)
    x = jax.random.normal(jax.random.key(1), (batch, seq_len, channels), jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(2), (batch, seq_len, channels), jnp.bfloat16)

    def layer(params, x):
        return eqx.combine(params, static)(x)[0]

    def cudnn(params, x):
        w = params.weight
        xp = jnp.pad(x, ((0, 0), (w.shape[1] - 1, 0), (0, 0)))
        y = lax.conv_general_dilated(xp, w.T[:, None, :].astype(x.dtype), (1,), "VALID",
                                     dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=channels,
                                     preferred_element_type=jnp.float32)
        return jax.nn.silu(y + params.bias.astype(jnp.float32)).astype(x.dtype)

    def hand(params, x):
        return _conv_hand_vjp(x, params.weight, params.bias)

    rows = batch * seq_len
    row = {"case": f"short_conv_c{channels}", "shape": [batch, seq_len, channels],
           "floor_fwd_ms": _floor_ms(2, rows, channels), "floor_fwd_bwd_ms": _floor_ms(5, rows, channels)}
    row["layer_fwd"] = _time(jax.jit(layer), params, x, iterations=iterations)
    row["layer_fwd_bwd"] = _time(_vjp(layer), params, x, ct, iterations=iterations)
    row["cudnn_fwd_bwd"] = _time(_vjp(cudnn), params, x, ct, iterations=iterations)
    row["hand_vjp_fwd_bwd"] = _time(_vjp(hand), params, x, ct, iterations=iterations)
    return row


def norm_rows(batch: int, seq_len: int, iterations: int) -> list[dict]:
    rows = batch * seq_len
    x = jax.random.normal(jax.random.key(1), (batch, seq_len, 768), jnp.bfloat16)
    ct = jax.random.normal(jax.random.key(2), (batch, seq_len, 768), jnp.bfloat16)
    w = jnp.ones((768,), jnp.float32)
    out = [{"case": "rms_norm_768", "shape": [batch, seq_len, 768], "floor_fwd_bwd_ms": _floor_ms(5, rows, 768),
            "fwd_bwd": _time(_vjp(lambda w, x: rms_norm(x, w, 1e-5)), w, x, ct, iterations=iterations)}]
    norm = GatedRMSNorm(1536, eps=1e-5, norm_before_gate=False, group_size=1536)
    params, static = eqx.partition(norm, eqx.is_inexact_array)
    ks = jax.random.split(jax.random.key(3), 3)
    y, z, ct2 = (jax.random.normal(k, (batch, seq_len, 1536), jnp.bfloat16) for k in ks)

    def gated(params, yz):
        return eqx.combine(params, static)(*yz)

    out.append({"case": "gated_rms_norm_1536", "shape": [batch, seq_len, 1536],
                "floor_fwd_bwd_ms": _floor_ms(8, rows, 1536),
                "fwd_bwd": _time(_vjp(gated), params, (y, z), ct2, iterations=iterations)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", default=None)
    p.add_argument("--iterations", type=int, default=harness.ITERATIONS)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    args = p.parse_args(argv)
    results = {"environment": harness.environment(), "gpu_before": harness.gpu_snapshot(), "rows": []}
    for channels in (1536, 256):
        row = conv_rows(args.batch, args.seq_len, channels, args.iterations)
        print({k: (round(v["median_ms"], 4) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        results["rows"].append(row)
    for row in norm_rows(args.batch, args.seq_len, args.iterations):
        print({k: (round(v["median_ms"], 4) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
        results["rows"].append(row)
    results["gpu_after"] = harness.gpu_snapshot()
    if args.json:
        harness.write_json(args.json, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
