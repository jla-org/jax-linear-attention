# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kernel time of the Gated DeltaNet backward's ``chunk_grads``, with and without its memory traffic.

    python benchmarks/profile_gdn_chunk_grads.py --shapes "1,16384,16,128,128;8,2048,32,128,128" \
        [--modes full,noload,nostore,nomem] [--registers "160,232,112;168,232,104"] [--base PATH]

``noload`` replaces every TMA load by a plain arrival on its barrier, ``nostore`` drops the TMA stores,
``nomem`` both, so the kernel computes on whatever the SMEM holds (timing only). ``--registers`` times
``gated_delta_rule_bwd.CHUNK_GRADS_REGISTERS`` splits; ``--base`` times another copy of
``gated_delta_rule_bwd.py`` (for example the tree before a change) on the same inputs and reports whether
each split's gradients equal its bit for bit. One JSON line per measurement (CUPTI, ``harness.py``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import jax
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

from jla.ops.mosaic_gpu import gated_delta_rule_bwd as gb  # noqa: E402

C = 64


def inputs(batch, seq_len, heads, key_dim, value_dim):
    ks = jax.random.split(jax.random.PRNGKey(0), 8)
    nc, bf = seq_len // C, jnp.bfloat16
    q = jax.random.normal(ks[0], (batch, seq_len, heads, key_dim), bf) * 0.1
    k = jax.random.normal(ks[1], (batch, seq_len, heads, key_dim), bf) * 0.1
    v = jax.random.normal(ks[2], (batch, seq_len, heads, value_dim), bf)
    do = jax.random.normal(ks[3], (batch, seq_len, heads, value_dim), bf)
    t0 = jax.random.normal(ks[4], (batch, heads, seq_len, C), bf) * 0.1
    g2 = jnp.broadcast_to(-0.05 * (jnp.arange(seq_len) % C + 1).astype(jnp.float32), (batch, heads, seq_len))
    beta = jax.nn.sigmoid(jax.random.normal(ks[5], (batch, heads, seq_len), jnp.float32))
    h = jax.random.normal(ks[6], (batch, heads, nc, value_dim, key_dim), bf) * 0.1
    dh = jax.random.normal(ks[7], (batch, heads, nc, value_dim, key_dim), bf) * 0.1
    return q, k, v, do, t0, g2, beta, h, dh


def without_memory(mode: str):
    """A stand-in for the ``plgpu`` module with the TMA loads and/or stores of ``mode`` removed."""
    p = types.ModuleType("plgpu_without_memory")
    p.__dict__.update(plgpu.__dict__)
    if mode in ("noload", "nomem"):
        p.copy_gmem_to_smem = lambda src, dst, barrier, **kw: plgpu.barrier_arrive(barrier)
    if mode in ("nostore", "nomem"):
        p.copy_smem_to_gmem = lambda *a, **kw: None
        p.wait_smem_to_gmem = lambda *a, **kw: None
        p.commit_smem_to_gmem_group = lambda *a, **kw: None
    return p


def timed(module, args, iterations):
    f = jax.jit(lambda *a: module.chunk_grads(*a, scale=0.088))
    out = jax.block_until_ready(f(*args))
    return out, harness.median_kernel_ms(f, *args, iterations=iterations, warmup=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--modes", default="full")
    parser.add_argument("--registers", default="")
    parser.add_argument("--base", default="")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    base = None
    if args.base:
        spec = importlib.util.spec_from_file_location("gdr_bwd_base", args.base)
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
    splits = [tuple(int(x) for x in s.split(",")) for s in args.registers.split(";") if s]
    for shape in [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]:
        xs = inputs(*shape)
        ref = None
        if base is not None:
            ref, r = timed(base, xs, args.iterations)
            print(json.dumps(dict(shape=shape, kernel="base", us=round(r["median_ms"] * 1e3, 1))), flush=True)
        for mode in args.modes.split(","):
            gb.plgpu = without_memory(mode) if mode != "full" else plgpu
            for regs in splits or [gb.CHUNK_GRADS_REGISTERS]:
                gb.CHUNK_GRADS_REGISTERS = regs
                out, r = timed(gb, xs, args.iterations)
                row = dict(shape=shape, kernel="chunk_grads", mode=mode, registers=regs,
                           us=round(r["median_ms"] * 1e3, 1))
                if ref is not None and mode == "full":
                    row["bitwise_equal_to_base"] = all(bool((a == b).all()) for a, b in zip(out, ref))
                print(json.dumps(row), flush=True)
        gb.plgpu = plgpu


if __name__ == "__main__":
    main()
