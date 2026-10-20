# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Error of the opt-in warm-up split (``scalar_gate.run(..., cp_warmup=...)``) against the exact split.

    python benchmarks/gdn_warmup_error.py --shapes "1,16384,16,128,128" [--gates fast,slow,mixed]

For each shape and gate distribution: the output, the final state and every gradient of the exact
split (the default) and of the warm-up split, each against a float32 ``chunked`` reference
(``precision=HIGHEST``), as the max absolute error over the max ``|reference|``, plus the two splits'
difference. Gates: ``fast`` = the benchmarks' ``logsigmoid(N(0, 1))`` (about -0.8 per token),
``slow`` = per-token decay 0.999 (no window reaches the threshold: exact), ``mixed`` = half the heads
fast, half with per-token decays in [0.98, 0.999] (Gated DeltaNet-like long-memory heads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla import ops  # noqa: E402
from jla.ops.mosaic_gpu import scalar_gate  # noqa: E402

F32 = jnp.float32


def gates(kind: str, batch: int, seq_len: int, heads: int):
    key = jax.random.key(7)
    fast = jax.nn.log_sigmoid(jax.random.normal(key, (batch, seq_len, heads), F32))
    if kind == "fast":
        return fast
    if kind == "slow":
        return jnp.full((batch, seq_len, heads), np.log(0.999), F32)
    decay = jnp.linspace(0.98, 0.999, heads - heads // 2)  # mixed
    slow = jnp.broadcast_to(jnp.log(decay), (batch, seq_len, heads - heads // 2))
    return jnp.concatenate([fast[..., : heads // 2], slow], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="1,16384,16,128,128")
    parser.add_argument("--gates", default="fast,slow,mixed")
    parser.add_argument("--threshold", type=float, default=-10.0)
    args = parser.parse_args()
    for spec in args.shapes.split(";"):
        batch, seq_len, heads, key_dim, value_dim = (int(x) for x in spec.split(","))
        x = harness.bf16_inputs(0, batch, seq_len, heads, key_dim, value_dim, delta=True)
        dht = jax.random.normal(jax.random.key(3), (batch, heads, key_dim, value_dim), F32)
        names = ("q", "k", "v", "g", "beta")
        for kind in args.gates.split(","):
            p0 = dict(q=x["q"], k=x["k"], v=x["v"], g=gates(kind, batch, seq_len, heads), beta=x["beta"])

            def value_and_grads(f):
                def loss(p):
                    o, s = f(p)
                    return jnp.sum(o.astype(F32) * x["do"].astype(F32)) + jnp.sum(s * dht), (o, s)
                (_, (o, s)), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(p0)
                return dict(o=o, final_state=s, **{"d" + n: g[n] for n in names})

            def ours(warmup):
                return lambda p: scalar_gate.run(p["q"], p["k"], p["v"], p["g"], p["beta"], delta=True, scale=None,
                                                 initial_state=None, output_final_state=True, cp_warmup=warmup)

            def chunked(p):
                return ops.linear_attention("gated_deltanet", *(p[n].astype(F32) for n in ("q", "k", "v")),
                                            impl="chunked", g=p["g"], beta=p["beta"], output_final_state=True,
                                            precision=lax.Precision.HIGHEST)

            ref = value_and_grads(chunked)
            exact = value_and_grads(ours(None))
            warm = value_and_grads(ours(args.threshold))

            def rel(a, b):
                a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
                return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30))

            row = dict(shape=[batch, seq_len, heads, key_dim, value_dim], gates=kind, threshold=args.threshold,
                       cp=scalar_gate.pieces(batch, heads, value_dim, seq_len, delta=True),
                       exact_vs_ref={n: rel(exact[n], ref[n]) for n in ref},
                       warmup_vs_ref={n: rel(warm[n], ref[n]) for n in ref},
                       warmup_vs_exact={n: rel(warm[n], exact[n]) for n in ref})
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
