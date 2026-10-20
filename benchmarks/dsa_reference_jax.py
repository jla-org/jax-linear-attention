# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JAX side of the DSA check against DeepSeek-V3.2's reference math (``dsa_reference_torch.py``).

Draws bf16 inputs (DeepSeek-V3.2-like indexer: 64 index heads of dim 128), runs the DSA core on the
Mosaic GPU kernels (``jla.ops.mosaic_gpu.dsa.dsa_layer``, query blocks of ``--block-rows``) and on the
plain-JAX path, with DeepSeek-V3.2's indexer KL loss (coefficient 1), and saves the inputs, the
selected indices, the outputs and the gradients of every input to an ``.npz`` for the torch side.

    python benchmarks/dsa_reference_jax.py --out /path/dsa_ref.npz [--length 1024 --packed]
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jla.ops.mosaic_gpu.dsa import dsa_layer, index_topk
from jla.ops.sparse_attention import attach_indexer_grad, dsa, dsa_attention_target, dsa_index_scores, dsa_select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--index-heads", type=int, default=64)
    parser.add_argument("--index-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-method", default="radix", choices=("radix", "sort"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    b, t, hq, h, d, hi, di = (args.batch, args.length, args.q_heads, args.kv_heads, args.head_dim,
                              args.index_heads, args.index_dim)

    def bf16(*shape, std=1.0):
        return jnp.asarray(rng.standard_normal(shape) * std, jnp.float32).astype(jnp.bfloat16)

    q, k, v = bf16(b, t, hq, d), bf16(b, t, h, d), bf16(b, t, h, d)
    q_idx, k_idx = bf16(b, t, hi, di), bf16(b, t, di)
    w_idx = jnp.asarray(rng.standard_normal((b, t, hi)) * hi ** -0.5, jnp.float32)  # weights_proj(x) / sqrt(HI)
    ct = jnp.asarray(rng.standard_normal((b, t, hq, d)), jnp.float32)
    seg = None
    if args.packed:
        ids = np.zeros((b, t), np.int32)
        for row in range(b):
            cuts = np.sort(rng.choice(np.arange(1, t), size=3, replace=False))
            for i, c in enumerate(cuts):
                ids[row, c:] = i + 1
        seg = jnp.asarray(ids)
    inputs = (q, k, v, q_idx, k_idx, w_idx)

    def kernel(*a):
        o = dsa_layer(*a, top_k=args.top_k, indexer_loss_coef=1.0, segment_ids=seg, block_rows=args.block_rows,
                      topk_method=args.topk_method)
        return jnp.sum(o.astype(jnp.float32) * ct), o

    def plain(*a):
        scores = dsa_index_scores(*a[3:], segment_ids=seg)
        indices = dsa_select(scores, args.top_k)
        o, probs, selected = dsa(*a[:3], indices, segment_ids=seg, return_probs=True)
        o = attach_indexer_grad(o, scores, dsa_attention_target(probs), selected.astype(jnp.float32), 1.0)
        return jnp.sum(o.astype(jnp.float32) * ct), o

    argnums = tuple(range(6))
    (_, out), grads = jax.jit(jax.value_and_grad(kernel, argnums=argnums, has_aux=True))(*inputs)
    indices = jax.jit(lambda *a: index_topk(*a, args.top_k, segment_ids=seg, block_rows=args.block_rows,
                                    method=args.topk_method))(
        q_idx, k_idx, w_idx)
    with jax.default_matmul_precision("highest"):  # float32 einsums, not TF32
        (_, out_xla), grads_xla = jax.jit(jax.value_and_grad(plain, argnums=argnums, has_aux=True))(*inputs)
    names = ("q", "k", "v", "q_idx", "k_idx", "w_idx")
    save = {name: np.asarray(x, np.float32) for name, x in zip(names, inputs)}
    save["ct"] = np.asarray(ct)
    save["segment_ids"] = np.zeros((b, t), np.int32) if seg is None else np.asarray(seg)
    save["indices"] = np.asarray(indices)
    save["out_kernel"] = np.asarray(out, np.float32)
    save["out_xla"] = np.asarray(out_xla, np.float32)
    for name, g, gx in zip(names, grads, grads_xla):
        save[f"d{name}_kernel"] = np.asarray(g, np.float32)
        save[f"d{name}_xla"] = np.asarray(gx, np.float32)
    save["top_k"] = np.asarray(args.top_k)
    np.savez(args.out, **save)
    print(f"saved {args.out}: T={t} B={b} HQ={hq} H={h} D={d} HI={hi} DI={di} top_k={args.top_k} packed={args.packed}")


if __name__ == "__main__":
    main()
