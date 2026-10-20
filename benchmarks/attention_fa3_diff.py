# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Differential check of the Mosaic GPU attention kernel against FlashAttention-3 on identical inputs.

For every case the same bf16 ``q, k, v`` and output cotangent go through our kernel
(``jla.layers.attention.attention(..., implementation="mosaic_gpu")``), FA3 (``torch_fa3.py`` in a
torch subprocess: ``window_size=(W - 1, 0)``, packed documents as ``cu_seqlens``), cuDNN (where it
accepts the call) and an fp32 dense XLA reference with the mask written out position by position.
Reported per tensor (``out, dq, dk, dv``): max abs error and max error relative to the largest
reference value, against the fp32 reference, plus a per-row relative error (max over (batch,
position, head) rows of the row's max error over its own magnitude, floored at 5% of the median
row magnitude) that exposes a dropped or leaked key block in a single row, and ours against FA3.
Attention sinks (FA3 has none) are checked against the GPT-OSS formulation: the sink logit is
appended as an extra (unscaled) column of the logits, softmaxed and dropped.

    python benchmarks/attention_fa3_diff.py --fa3-python <venv with flash_attn_3>/bin/python --out <json>
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from bench_attention_fa3 import Worker  # noqa: E402

from jla.layers.attention import attention  # noqa: E402
from jla.ops.mosaic_gpu.attention import flash_attention  # noqa: E402

# name, batch, length, heads, kv_heads, head_dim, window, doc_lens (per row) or None, q scale
CASES = [
    ("causal-4096", 1, 4096, 4, 4, 128, None, None, 1.0),
    ("causal-odd-1000-gqa2-d64", 2, 1000, 4, 2, 64, None, None, 1.0),
    ("causal-8192-gqa2", 1, 8192, 2, 1, 128, None, None, 1.0),
    ("swa128-4096-gqa4", 1, 4096, 4, 1, 128, 128, None, 1.0),
    ("swa1-512-d64", 1, 512, 2, 2, 64, 1, None, 1.0),
    ("swa1000-odd-3001-gqa2", 1, 3001, 4, 2, 128, 1000, None, 1.0),
    ("documents-var-4096-gqa2", 2, 4096, 4, 2, 128, None, [[700, 1500, 1, 1895], [64, 4000, 32]], 1.0),
    ("documents-swa256-4096-d64", 1, 4096, 4, 4, 64, 256, [[1000, 3000, 96]], 1.0),
    ("documents-64x32-2048", 1, 2048, 2, 2, 128, None, [[32] * 64], 1.0),
    ("documents-4x512-gqa8", 1, 2048, 8, 1, 128, None, [[512] * 4], 1.0),
    ("causal-peaky-q8-2048", 1, 2048, 2, 2, 128, None, None, 8.0),
    ("documents-odd-1500-d64", 1, 1500, 2, 1, 64, 300, [[333, 1, 1166]], 1.0),
]
SINK_CASES = [
    ("sinks-causal-2048-gqa2", 1, 2048, 4, 2, 128, None, None),
    ("sinks-swa128-odd-1111-d64", 1, 1111, 4, 1, 64, 128, None),
    ("sinks-documents-2048", 1, 2048, 4, 4, 128, None, [[500, 1000, 548]]),
]


def _segments(doc_lens, length):
    if doc_lens is None:
        return None
    rows = [np.repeat(np.arange(len(row), dtype=np.int32), row) for row in doc_lens]
    assert all(len(r) == length for r in rows), doc_lens
    return jnp.asarray(np.stack(rows))


def _mask(length, window, seg):
    pos = np.arange(length)
    mask = pos[:, None] >= pos[None, :]
    if window is not None:
        mask = mask & (pos[:, None] - pos[None, :] < window)
    mask = jnp.asarray(mask)[None, None]
    if seg is not None:
        mask = mask & (seg[:, :, None] == seg[:, None, :])[:, None]
    return mask


def _dense(q, k, v, mask, scale):
    f32 = lambda x: x.astype(jnp.float32)  # noqa: E731
    return jax.nn.dot_product_attention(f32(q), f32(k), f32(v), mask=mask, scale=scale, implementation="xla")


def _gpt_oss_sinks(q, k, v, mask, sinks, scale):
    """GPT-OSS ``sdpa``: logits with the sink appended as one extra column, softmax, drop it."""
    batch, length, heads, head_dim = q.shape
    group = heads // k.shape[2]
    f32 = jnp.float32
    kk = jnp.repeat(k.astype(f32), group, axis=2)
    vv = jnp.repeat(v.astype(f32), group, axis=2)
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(f32), kk) * scale
    s = jnp.where(mask, s, -jnp.inf)
    sink = jnp.broadcast_to(sinks.astype(f32)[None, :, None, None], (batch, heads, length, 1))
    w = jax.nn.softmax(jnp.concatenate([s, sink], axis=-1), axis=-1)[..., :-1]
    return jnp.einsum("bhqk,bkhd->bqhd", w, vv)


def _grads(fn, args, do):
    def loss(*a):
        out = fn(*a)
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32)), out

    grads, out = jax.jit(jax.grad(loss, argnums=tuple(range(len(args))), has_aux=True))(*args)
    return [np.asarray(x, np.float32) for x in (out, *grads)]


def _errors(actual, expected):
    a, e = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    diff = np.abs(a - e)
    out = dict(max_abs=float(diff.max()), rel=float(diff.max() / (np.abs(e).max() + 1e-6)))
    if a.ndim == 4:  # rows = (batch, position, head)
        mag = np.abs(e).max(axis=-1)
        floor = 0.05 * float(np.median(mag)) + 1e-6
        out["row_rel"] = float((diff.max(axis=-1) / (mag + floor)).max())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--fa3-python", required=True)
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    fa3 = Worker(args.fa3_python, "torch_fa3.py")
    names = ("out", "dq", "dk", "dv")
    results = []
    tmp = tempfile.mkdtemp(prefix="attn-diff-")
    try:
        for name, batch, length, heads, kv_heads, head_dim, window, doc_lens, q_scale in CASES:
            if args.cases != "all" and name not in args.cases.split(","):
                continue
            keys = jax.random.split(jax.random.key(length + heads), 4)
            f32 = jnp.float32
            q = (q_scale * jax.random.normal(keys[0], (batch, length, heads, head_dim), f32)).astype(jnp.bfloat16)
            k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), f32).astype(jnp.bfloat16)
            v = jax.random.normal(keys[2], (batch, length, kv_heads, head_dim), f32).astype(jnp.bfloat16)
            do = jax.random.normal(keys[3], (batch, length, heads, head_dim), f32).astype(jnp.bfloat16)
            scale = head_dim**-0.5
            seg = _segments(doc_lens, length)
            mask = _mask(length, window, seg)
            ref = _grads(lambda q, k, v: _dense(q, k, v, mask, scale), (q, k, v), do)
            row = dict(case=name, B=batch, T=length, H=heads, Hkv=kv_heads, D=head_dim, window=window,
                       doc_lens=doc_lens, q_scale=q_scale, impls={})
            ours = _grads(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=window, scale=scale,
                                                    implementation="mosaic_gpu"), (q, k, v), do)
            row["impls"]["mosaic_gpu"] = {n: _errors(a, e) for n, a, e in zip(names, ours, ref)}
            if length % 2 == 0:  # the cuDNN backward rejects odd lengths
                try:
                    cudnn = _grads(lambda q, k, v: attention(q, k, v, segment_ids=seg, sliding_window=window,
                                                             scale=scale, implementation="cudnn"), (q, k, v), do)
                    row["impls"]["cudnn"] = {n: _errors(a, e) for n, a, e in zip(names, cudnn, ref)}
                except Exception as err:  # record and continue
                    row["impls"]["cudnn"] = {"error": harness.error_summary(f"{type(err).__name__}: {err}")}
            path_in, path_out = os.path.join(tmp, f"{name}-in.npz"), os.path.join(tmp, f"{name}-out.npz")
            np.savez(path_in, **{n: np.asarray(x, np.float32) for n, x in (("q", q), ("k", k), ("v", v), ("do", do))})
            answer = fa3.ask(dict(op="grads", inputs=path_in, output=path_out, window=window, doc_lens=doc_lens,
                                  scale=scale))
            if "error" in answer:
                row["impls"]["fa3"] = {"error": answer["error"]}
            else:
                data = np.load(path_out)
                theirs = [data[n] for n in names]
                row["impls"]["fa3"] = {n: _errors(a, e) for n, a, e in zip(names, theirs, ref)}
                row["mosaic_gpu_vs_fa3"] = {n: _errors(a, e) for n, a, e in zip(names, ours, theirs)}
                row["fa3_version"] = answer.get("flash_attn_3")
            print(row, flush=True)
            results.append(row)
            harness.write_json(args.out, dict(environment=harness.environment(), rows=results))
        for name, batch, length, heads, kv_heads, head_dim, window, doc_lens in SINK_CASES:
            if args.cases != "all" and name not in args.cases.split(","):
                continue
            keys = jax.random.split(jax.random.key(length + 7), 4)
            f32 = jnp.float32
            q = jax.random.normal(keys[0], (batch, length, heads, head_dim), f32).astype(jnp.bfloat16)
            k = jax.random.normal(keys[1], (batch, length, kv_heads, head_dim), f32).astype(jnp.bfloat16)
            v = jax.random.normal(keys[2], (batch, length, kv_heads, head_dim), f32).astype(jnp.bfloat16)
            do = jax.random.normal(keys[3], (batch, length, heads, head_dim), f32).astype(jnp.bfloat16)
            sinks = jnp.linspace(-3.0, 5.0, heads, dtype=f32)
            scale = head_dim**-0.5
            seg = _segments(doc_lens, length)
            mask = _mask(length, window, seg)
            ref = _grads(lambda q, k, v, s: _gpt_oss_sinks(q, k, v, mask, s, scale), (q, k, v, sinks), do)
            ours = _grads(lambda q, k, v, s: flash_attention(q, k, v, segment_ids=seg, sliding_window=window,
                                                             scale=scale, sinks=s), (q, k, v, sinks), do)
            row = dict(case=name, B=batch, T=length, H=heads, Hkv=kv_heads, D=head_dim, window=window,
                       doc_lens=doc_lens, sinks=np.asarray(sinks).tolist(), impls={})
            row["impls"]["mosaic_gpu"] = {n: _errors(a, e) for n, a, e in
                                          zip((*names, "dsinks"), ours, ref)}
            print(row, flush=True)
            results.append(row)
            harness.write_json(args.out, dict(environment=harness.environment(), rows=results))
    finally:
        fa3.close()


if __name__ == "__main__":
    main()
