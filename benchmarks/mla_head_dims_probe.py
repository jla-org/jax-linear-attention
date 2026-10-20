# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe which (q/k head dim, v head dim) pairs ``jax.nn.dot_product_attention`` accepts per backend.

MLA has q/k head dim 192 and v head dim 128 (DeepSeek-V3); ``jla.layers.mla`` zero-pads
the narrower one. Run: ``python benchmarks/mla_head_dims_probe.py`` (GPU for cuDNN).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp


def probe(qk: int, v: int, impl: str) -> str:
    q = jnp.ones((1, 64, 4, qk), jnp.bfloat16)
    k = jnp.ones((1, 64, 4, qk), jnp.bfloat16)
    val = jnp.ones((1, 64, 4, v), jnp.bfloat16)
    try:
        out = jax.jit(lambda q, k, val: jax.nn.dot_product_attention(q, k, val, is_causal=True,
                                                                     implementation=impl))(q, k, val)
        out.block_until_ready()
        return f"ok {tuple(out.shape)}"
    except Exception as err:  # noqa: BLE001 - the probe reports every rejection
        return f"rejected: {type(err).__name__}: {str(err).splitlines()[0][:160]}"


if __name__ == "__main__":
    impls = ["xla"] + (["cudnn"] if jax.default_backend() == "gpu" else [])
    results = {f"{impl} qk={qk} v={v}": probe(qk, v, impl) for impl in impls for qk, v in
               ((192, 128), (192, 192), (128, 192), (576, 512))}
    print(json.dumps({"jax": jax.__version__, "backend": jax.default_backend(), **results}, indent=1))
