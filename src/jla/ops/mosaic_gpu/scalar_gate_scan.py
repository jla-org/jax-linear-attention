# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The delta rule's piece scans (the exact sequence split) as one small Pallas Triton kernel.

The exact split of :mod:`scalar_gate` stitches its ``cp`` pieces with the affine
piece maps ``S_end = M S_start + E`` (states transposed, ``[V, K]``): forward,
piece ``i + 1`` starts from ``S_{i+1}^T = S_i^T M_i^T + E_i^T``; backward, the
gradient leaving piece ``i - 1`` is ``G_{i-1} = G_i M_i + D_i``. Unrolled in XLA,
every step was its own float32 GEMM launch (about 4 us each on an H100: 28 us
at 8 pieces, 66 at 16, and as much again in the backward). Here one program per
``(b, h, 16 value rows)`` keeps its rows of the running state in registers and
walks every piece, with IEEE float32 products (``Precision.HIGHEST``, as the XLA
scan) over 64-column register tiles, the next step's tiles loaded while this one
multiplies (the backward reads the maps transposed once in XLA). A step whose
map is exactly zero (a piece with a document start, or one whose successor
starts a document at its first token, see
:func:`scalar_gate_docs.fwd_walk_table`) takes its local term without the
product, which is what the product gives. :mod:`scalar_gate` runs it with packed
documents (``segment_ids``) from 8 pieces on; without documents no map is zero and
XLA's unrolled scan, :func:`piece_scan_xla`, is the faster one (measured on an H100:
28 us against 47 for this kernel at 8 pieces), and below 8 pieces XLA's scan is short
enough that the kernel's extra steps cost what they save.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

Array = jax.Array
F32 = jnp.float32
HIGHEST = lax.Precision.HIGHEST
ROWS = 16  # value rows per program
COLS = 64  # column block of the register tiles
PRECISION = HIGHEST  # IEEE float32 products
NUM_WARPS = 8


def _compiler_params():
    from jax.experimental.pallas import triton as pltriton

    return pltriton.CompilerParams(num_warps=NUM_WARPS, num_stages=1)


def _kernel(skip_ref, local_ref, maps_ref, *refs, cp, heads, key_dim, local_row0, maps_row0, reverse, has_init):
    # The running rows of the state live in COLS-wide column blocks, every product is a sum of
    # [ROWS, COLS] x [COLS, COLS] tiles (small register tiles), and the next step's tiles are loaded
    # before this step's products (their latency overlaps the chain).
    init_ref = refs[0] if has_init else None
    out_ref = refs[-1]
    bh, rb = pl.program_id(0), pl.program_id(1)
    b, h = bh // heads, bh % heads
    r0 = rb * ROWS
    rows = pl.ds(r0, ROWS)
    lrows = pl.ds(local_row0 + r0, ROWS)
    nb = key_dim // COLS

    def cols(j):
        return pl.ds(j * COLS, COLS)

    def store(i, s):
        for j in range(nb):
            out_ref[b, i, h, rows, cols(j)] = s[j]

    def tile(i, j, c):  # (S A)[:, j] = sum_c S[:, c] A[c, j], A = M_i^T forward and M_i backward
        return maps_ref[b, i, h, pl.ds(maps_row0 + c * COLS, COLS), cols(j)]

    def loads(i, first):
        local = [local_ref[b, i, h, lrows, cols(j)] for j in range(nb)]
        tiles = None if first else [[tile(i, j, c) for c in range(nb)] for j in range(nb)]
        return local, tiles

    def step(i, s, local, tiles):
        if tiles is None:  # from a zero state (gradient): the local term itself
            return local

        def product():
            out = []
            for j in range(nb):
                acc = local[j]
                for c in range(nb):
                    acc = acc + jnp.dot(s[c], tiles[j][c], precision=PRECISION, preferred_element_type=F32)
                out.append(acc)
            return tuple(out)

        return list(lax.cond(skip_ref[b, i, h] != 0, lambda: tuple(local), product))

    if has_init:
        s = [init_ref[b, h, rows, cols(j)] for j in range(nb)]
    else:
        s = [jnp.zeros((ROWS, COLS), F32) for _ in range(nb)]
    # Forward: piece 0 starts from the initial state, step i gives piece i + 1's start. Backward: the last
    # piece ends with the final-state gradient, step i gives the end gradient of piece i - 1.
    order = list(range(cp - 1, 0, -1)) if reverse else list(range(cp - 1))
    first = [(i == (cp - 1 if reverse else 0)) and not has_init for i in order]
    store(cp - 1 if reverse else 0, s)
    pending = loads(order[0], first[0]) if order else None
    for n, i in enumerate(order):
        local, tiles = pending
        if n + 1 < len(order):
            pending = loads(order[n + 1], first[n + 1])
        s = step(i, s, local, tiles)
        store(i - 1 if reverse else i + 1, s)


def piece_scan(local: Array, maps: Array, init: Array | None, skip: Array, *, value_dim: int, local_row0: int = 0,
               maps_row0: int = 0, reverse: bool = False, interpret=None) -> Array:
    """Every piece's start state (forward) or end gradient (``reverse``), ``[B, cp, H, V, K]`` float32.

    ``local [B, cp, H, R, K]`` holds the local terms (``E^T`` forward, ``D^T``
    backward) in rows ``local_row0 : local_row0 + V``; ``maps [B, cp, H, R', K]``
    holds ``M^T`` in rows ``maps_row0 : maps_row0 + K`` (one array can hold both,
    as the forward map pass's ``[M^T | E^T]``). ``init [B, H, V, K]`` is the
    transposed initial state (forward) or final-state gradient (backward), ``None``
    for zero: then the first step takes its local term directly. ``skip [B, cp, H]``
    is nonzero where ``M_i`` is exactly zero. Forward, piece 0 starts from ``init``
    and piece ``i + 1`` from step ``i``; backward, the last piece ends with ``init``
    and piece ``i - 1`` with step ``i``. Same semantics as the XLA scans of
    :func:`scalar_gate._piece_starts` and ``_piece_end_grads``.
    """
    batch, cp, heads, _, key_dim = local.shape
    if value_dim % ROWS or key_dim % COLS:
        raise ValueError(f"value_dim must be a multiple of {ROWS} and key_dim of {COLS}, got {value_dim}, {key_dim}")
    if reverse:  # G M_i: the maps transposed once in XLA, so every product is a plain [S] x [A] tile product
        maps = jnp.swapaxes(maps[:, :, :, maps_row0:maps_row0 + key_dim], -1, -2)
        maps_row0 = 0
    kernel = functools.partial(_kernel, cp=cp, heads=heads, key_dim=key_dim, local_row0=local_row0,
                               maps_row0=maps_row0, reverse=reverse, has_init=init is not None)
    args = [skip.astype(jnp.int32), local, maps] + ([init.astype(F32)] if init is not None else [])
    interpret = bool(interpret)
    return pl.pallas_call(
        kernel, grid=(batch * heads, value_dim // ROWS),
        out_shape=jax.ShapeDtypeStruct((batch, cp, heads, value_dim, key_dim), F32),
        compiler_params=None if interpret else _compiler_params(), interpret=interpret,
        name="gdr_piece_scan_bwd" if reverse else "gdr_piece_scan_fwd",
    )(*args)


def piece_scan_xla(local: Array, maps: Array, init: Array | None, skip: Array | None, *, value_dim: int,
                   local_row0: int = 0, maps_row0: int = 0, reverse: bool = False) -> Array:
    """:func:`piece_scan` unrolled in XLA (one float32 GEMM per step; the scan without documents).

    With ``skip=None`` every step takes the product (exactly the previous scan).
    """
    batch, cp, heads, _, key_dim = local.shape
    loc = local[:, :, :, local_row0:local_row0 + value_dim]
    m = maps[:, :, :, maps_row0:maps_row0 + key_dim]
    keep = None if skip is None else (skip == 0)[..., None, None]

    def add(p, loc, i):
        return p + loc if keep is None else jnp.where(keep[:, i], p + loc, loc)

    s = jnp.zeros((batch, heads, value_dim, key_dim), F32) if init is None else init.astype(F32)
    out = [None] * cp
    if reverse:
        out[cp - 1] = s
        for i in range(cp - 1, 0, -1):
            if i == cp - 1 and init is None:
                s = loc[:, i]
            else:
                p = jnp.einsum("bhvk,bhjk->bhvj", s, m[:, i], precision=HIGHEST)
                s = add(p, loc[:, i], i)
            out[i - 1] = s
    else:
        for i in range(cp):
            out[i] = s
            if i + 1 < cp:
                if i == 0 and init is None:
                    s = loc[:, 0]
                else:
                    p = jnp.einsum("bhvk,bhkj->bhvj", s, m[:, i], precision=HIGHEST)
                    s = add(p, loc[:, i], i)
    return jnp.stack(out, axis=1)
