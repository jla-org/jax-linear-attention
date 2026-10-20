# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/ops/common/chunk_o.py (chunk_bwd_dqkwg)
# Packed-gradient idea (one gradient per producer tensor, no split/concat/reduction pass) from FLAT
# (Apache-2.0), flat/linear_attention/op.py linear_attention_packed_jax
"""Grouped simple GLA backward: ``dq``/``dk`` summed over a key group's value heads in the kernel.

Mamba-2 hands its ``B, C [B, T, G, N]`` to the simple-GLA kernels once per group
(``H = n G`` value heads read group ``h // n``). The per-chunk gradient kernel
:func:`jla.ops.mosaic_gpu.scalar_gate_kernels.dqkwg` runs one CTA per value head and
writes ``dq, dk`` per value head, ``[B, Tp, H, N]``; XLA then sums each group's ``n``
heads. At the 1.3B Mamba-2 shape (64 heads per group, ``B T = 65,536``) those two
per-head tensors are 1 GiB each per layer, written by the kernel and read back by
the reduction.

:func:`dqkwg_grouped` runs one CTA per ``(b, chunk, key group, split)`` instead: it
loads the chunk's ``q`` and ``k`` once, walks ``n / splits`` value heads (their
``v``, ``do``, states and decays double-buffered with TMA, the next head's tiles
loading while the current head computes), forms every head's ``dq``, ``dk`` and
gate gradient exactly as ``dqkwg`` does, and adds ``dq``, ``dk`` into float32
accumulators in shared memory (``register_sums=True``: carried in registers through
the head loop). Only ``[B, Tp, G * splits, N]`` float32 partial sums
leave the kernel (``splits = 1``: the group sums themselves). The per-head
products are those of ``dqkwg``; the sum over a group's heads is float32
throughout (``dqkwg`` + XLA round every head's term to the input dtype first,
``GROUPED_SIMPLE_GLA_KEY_GRADS_IN_INPUT_DTYPE``), so the result is at least as
accurate.
"""

from __future__ import annotations

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

from jla.ops.mosaic_gpu._common import (
    CHUNK,
    COLS,
    ROWS,
    WGMMA,
    Segments,
    bcast_rows,
    build_kernel,
    decay_matrix,
    matmul,
    num_sms,
    smem,
)

Array = jax.Array
F32 = jnp.float32
C = CHUNK


def default_splits(batch: int, chunks: int, kv_heads: int, group: int) -> int:
    """Splits of a group's heads: 1 unless that leaves fewer than two CTAs per SM (a power of two)."""
    splits = 1
    while batch * chunks * kv_heads * splits < 2 * num_sms() and group % (2 * splits) == 0 and group // splits > 2:
        splits *= 2
    return splits


def dqkwg_grouped(
    q: Array,
    k: Array,
    v: Array,
    g2: Array,
    h: Array,
    dh: Array,
    do: Array,
    *,
    scale: float,
    splits: int = 1,
    seg: Segments | None = None,
    register_sums: bool = False,
    interpret=None,
) -> tuple[Array, Array, Array]:
    """Per-chunk ``dq``, ``dk`` summed over each key group's value heads, and every head's ``dG``.

    Args:
      q, k: ``[B, Tp, Hk, K]`` (value head ``h`` reads key head ``h // (H // Hk)``).
      v, do: ``[B, Tp, H, V]`` values and output gradient.
      g2: ``[B, H, Tp]`` float32 chunk-local cumulative decay in log2 units.
      h, dh: ``[B, H, NC, V, K]`` transposed states entering each chunk and their gradients.
      splits: CTAs per key group; each sums ``H / Hk / splits`` consecutive value heads.
      register_sums: carry the float32 sums in registers through the head loop (shared memory
        only stages the final store) instead of adding every head's terms into shared memory.

    Returns:
      ``dq, dk [B, Tp, Hk * splits, K]`` float32 (row ``hk * splits + s`` holds split
      ``s`` of key head ``hk``) and ``dG [B, H, Tp]`` float32 (natural units, before the
      reverse chunk cumsum), as :func:`scalar_gate_kernels.dqkwg` without the delta rule.
    """
    batch, padded, kv_heads, key_dim = q.shape
    heads = g2.shape[1]
    value_dim = do.shape[-1]
    group = heads // kv_heads
    if heads % kv_heads or group % splits:
        raise ValueError(f"heads {heads} / key heads {kv_heads} / splits {splits} do not divide")
    per_cta = group // splits
    nc = padded // C
    dtype = q.dtype
    segs = seg is not None
    slots = 2 if value_dim == 64 else 1  # double buffering fits shared memory for V = 64
    out_type = [
        jax.ShapeDtypeStruct((batch, padded, kv_heads * splits, key_dim), F32),
        jax.ShapeDtypeStruct((batch, padded, kv_heads * splits, key_dim), F32),
        jax.ShapeDtypeStruct((batch, heads, padded), F32),
    ]
    vec = plgpu.SMEM((C,), F32)
    vecs = plgpu.SMEM((slots, C), F32)
    scratch = dict(
        q_smem=smem((C, key_dim), dtype),
        k_smem=smem((C, key_dim), dtype),
        v_smem=smem((slots, C, value_dim), dtype),
        do_smem=smem((slots, C, value_dim), dtype),
        h_smem=smem((slots, value_dim, key_dim), dtype),
        dh_smem=smem((slots, value_dim, key_dim), dtype),
        g_smem=vecs,
        gc_smem=vecs,
        ds_smem=smem((C, C), dtype),
        dg_out=vecs,
        dq_acc=plgpu.SMEM((C, key_dim), F32),
        dk_acc=plgpu.SMEM((C, key_dim), F32),
        qk_bar=plgpu.Barrier(num_arrivals=2 + 4 * segs),
        in_bar=plgpu.Barrier(num_arrivals=6, num_barriers=slots),
    )
    if segs:
        scratch.update(run_r=vec, run_c=vec, cin_r=vec, tail_r=vec)

    def body(*refs, q_smem, k_smem, v_smem, do_smem, h_smem, dh_smem, g_smem, gc_smem, ds_smem, dg_out, dq_acc,
             dk_acc, qk_bar, in_bar, run_r=None, run_c=None, cin_r=None, tail_r=None):
        refs = list(refs)
        q_ref, k_ref, v_ref, g_ref, h_ref, dh_ref, do_ref = (refs.pop(0) for _ in range(7))
        run_ref, cin_ref, tail_ref = (refs.pop(0), refs.pop(0), refs.pop(0)) if segs else (None, None, None)
        dq_ref, dk_ref, dg_ref = refs.pop(0), refs.pop(0), refs.pop(0)
        b, n, s = lax.axis_index("b"), lax.axis_index("n"), lax.axis_index("s")
        hk = lax.div(s, splits)
        h_first = hk * group + lax.rem(s, splits) * per_cta
        rows = pl.ds(n * C, C)

        def issue(j, slot):  # the tiles of the CTA's j-th value head
            hh = h_first + j
            bar = in_bar.at[slot]
            plgpu.copy_gmem_to_smem(v_ref.at[b, rows, hh], v_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(do_ref.at[b, rows, hh], do_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(h_ref.at[b, hh, n], h_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(dh_ref.at[b, hh, n], dh_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], g_smem.at[slot], bar)
            plgpu.copy_gmem_to_smem(g_ref.at[b, hh, rows], gc_smem.at[slot], bar)

        plgpu.copy_gmem_to_smem(q_ref.at[b, rows, hk], q_smem, qk_bar)
        plgpu.copy_gmem_to_smem(k_ref.at[b, rows, hk], k_smem, qk_bar)
        if segs:
            for src, dst in ((run_ref, run_r), (run_ref, run_c), (cin_ref, cin_r), (tail_ref, tail_r)):
                plgpu.copy_gmem_to_smem(src.at[b, rows], dst, qk_bar)
        issue(0, 0)
        shape_k = (C, key_dim)
        zeros = plgpu.layout_cast(jnp.zeros(shape_k, F32), WGMMA)
        if not register_sums:
            dq_acc[...] = zeros
            dk_acc[...] = zeros
        plgpu.barrier_wait(qk_bar)
        last = plgpu.broadcasted_iota(jnp.int32, (C,), 0, layout=ROWS) == C - 1

        def head_step(j, carry):
            slot = lax.rem(j, slots) if slots > 1 else 0
            plgpu.barrier_wait(in_bar.at[slot])
            if slots > 1:
                @pl.when(j + 1 < per_cta)
                def _prefetch():
                    plgpu.commit_smem()
                    issue(j + 1, 1 - slot)

            v_t, do_t, h_t, dh_t = v_smem.at[slot], do_smem.at[slot], h_smem.at[slot], dh_smem.at[slot]
            g_row = plgpu.load(g_smem.at[slot], layout=ROWS)
            g_col = plgpu.load(gc_smem.at[slot], layout=COLS)
            g_last = g_smem[slot, C - 1]
            q_scale = jnp.exp2(g_row) * scale  # the output reads the incoming state
            k_decay = jnp.exp2(g_last - g_row)  # the key writes the outgoing state
            s_decay = jnp.exp2(g_last)
            runs = None
            if segs:
                runs = (plgpu.load(run_r, layout=ROWS), plgpu.load(run_c, layout=COLS))
                q_scale = q_scale * plgpu.load(cin_r, layout=ROWS)
                k_decay = k_decay * plgpu.load(tail_r, layout=ROWS)
                s_decay = s_decay * cin_r[C - 1]
            # ds = (do v^T) * exp(G_i - G_j) * scale on j <= i
            ds = matmul(do_t, v_t.T, (C, C))
            ds_lo = (ds * decay_matrix(g_row, g_col, (C, C), strict=False, runs=runs) * scale).astype(dtype)
            ds_smem[...] = ds_lo
            dq = matmul(do_t, h_t, shape_k)
            dk = matmul(v_t, dh_t, shape_k)
            plgpu.commit_smem()
            dq = dq * bcast_rows(q_scale, shape_k)
            dq = matmul(ds_lo, k_smem, shape_k, init=dq)
            dk = dk * bcast_rows(k_decay, shape_k)
            kk = plgpu.load(k_smem, layout=WGMMA).astype(F32)
            qq = plgpu.load(q_smem, layout=WGMMA).astype(F32)
            hs = plgpu.load(h_t, layout=WGMMA).astype(F32)
            dhs = plgpu.load(dh_t, layout=WGMMA).astype(F32)
            dg_last = s_decay * (hs * dhs).sum(axis=1).sum() + (dk * kk).sum(axis=1).sum()
            dk = matmul(ds_smem.T, q_smem, shape_k, init=dk)
            dg = (dq * qq).sum(axis=1) - (dk * kk).sum(axis=1)
            # This slot's gate-gradient buffer was last read by the store of head j - slots.
            plgpu.wait_smem_to_gmem(slots - 1, wait_read_only=True)
            dg_out.at[slot][...] = dg + jnp.where(last, dg_last, 0.0)
            if register_sums:
                carry = (carry[0] + dq, carry[1] + dk)
            else:
                dq_acc[...] = plgpu.load(dq_acc, layout=WGMMA) + dq
                dk_acc[...] = plgpu.load(dk_acc, layout=WGMMA) + dk
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(dg_out.at[slot], dg_ref.at[b, h_first + j, rows])
            if slots == 1:
                @pl.when(j + 1 < per_cta)
                def _next():
                    issue(j + 1, 0)
            return carry

        sums = lax.fori_loop(0, per_cta, head_step, (zeros, zeros) if register_sums else 0)
        if register_sums:
            dq_acc[...] = sums[0]
            dk_acc[...] = sums[1]
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(dq_acc, dq_ref.at[b, rows, s], commit_group=False)
        plgpu.copy_smem_to_gmem(dk_acc, dk_ref.at[b, rows, s], commit_group=False)
        plgpu.commit_smem_to_gmem_group()
        plgpu.wait_smem_to_gmem(0)

    args = [q, k, v, g2, h, dh, do] + ([seg.run, seg.cin, seg.tail] if segs else [])
    return tuple(build_kernel(
        body, name="sg_dqkwg_grouped", out_type=out_type, scratch=scratch,
        grid=(batch, nc, kv_heads * splits), grid_names=("b", "n", "s"), interpret=interpret,
    )(*args))
