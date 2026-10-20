# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared inputs and oracles of the Mosaic GPU KDA / GDN-2 tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jla.ops import chunked, reference as ref
from jla.ops.mosaic_gpu import vector_gate as vg
from jla.ops.mosaic_gpu import vector_gate_ops as vgo

F32, BF16 = jnp.float32, jnp.bfloat16
C = vg.CHUNK
HIGHEST = lax.Precision.HIGHEST


def inputs(seed: int, kind: str, B=1, T=128, H=2, K=128, V=128, *, huge=False) -> dict:
    """bf16 ``q, k, v`` (unit-norm ``k``), f32 log-decay ``g [B, T, H, K]`` and gates, an initial
    state and cotangents. GDN-2 gates are rounded to bf16 (the kernels' precision) so the oracles
    see the same values. ``huge``: a log-decay of -60 on 10% of the (token, channel) entries."""
    ks = jax.random.split(jax.random.key(seed), 11)
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, T, H, K)) - 2.0)
    if huge:
        g = jnp.where(jax.random.bernoulli(ks[9], 0.1, g.shape), -60.0, g)
    x = dict(
        q=jax.random.normal(ks[0], (B, T, H, K)).astype(BF16),
        k=ref.l2norm(jax.random.normal(ks[1], (B, T, H, K))).astype(BF16),
        v=jax.random.normal(ks[2], (B, T, H, V)).astype(BF16),
        g=g,
        h0=0.1 * jax.random.normal(ks[6], (B, H, K, V)),
        do=jax.random.normal(ks[7], (B, T, H, V)),
        dht=jax.random.normal(ks[8], (B, H, K, V)),
    )
    if kind == "kda":
        x["gates"] = (jax.nn.sigmoid(jax.random.normal(ks[4], (B, T, H))),)
    else:
        x["gates"] = tuple(jax.nn.sigmoid(jax.random.normal(key, shape)).astype(BF16).astype(F32)
                           for key, shape in ((ks[4], (B, T, H, K)), (ks[5], (B, T, H, V))))
    return x


def gate_kwargs(kind: str, gates) -> dict:
    return {"beta": gates[0]} if kind == "kda" else {"b": gates[0], "w": gates[1]}


def segments(seed: int, B: int, T: int) -> jax.Array:
    """Packed documents: ``segment_ids [B, T]`` with resets at random tokens (some documents
    shorter than a chunk) plus a few fixed ones."""
    starts = jax.random.bernoulli(jax.random.key(seed), 0.02, (B, T)).at[:, 0].set(False)
    starts = starts.at[:, T // 3].set(True).at[0, min(5, T - 1)].set(True).at[-1, T - 2].set(True)
    return jnp.cumsum(starts.astype(jnp.int32), axis=1)


def dense_segments(B: int, T: int) -> jax.Array:
    """Many segment starts per chunk, where the -128 offsets add up in ``G`` (``T >= 256``): chunk 0
    holds 32 one-token documents and then a 31-token one, chunk 1 four-token documents, chunk 2
    eight-token ones and chunk 3 a single start."""
    starts = np.zeros((B, T), bool)
    starts[:, 1:33] = True
    starts[:, 64:128:4] = True
    starts[:, 128:192:8] = True
    starts[:, 200] = True
    return jnp.asarray(np.cumsum(starts, axis=1), jnp.int32)


def assert_close(got, want, tol: float, name: str = "") -> None:
    """Max abs error at most ``tol`` times max |want| (bf16 tensor-core tolerances)."""
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert np.isfinite(got).all(), f"{name}: non-finite values"
    scale = max(float(np.abs(want).max()), 1e-6)
    err = float(np.abs(got - want).max()) / scale
    assert err <= tol, f"{name}: max error {err:.3e} of max |ref| {scale:.3e} exceeds {tol}"


def chunks(x: jax.Array) -> jax.Array:
    """``[B, T, H, ...]`` (``T`` a multiple of the chunk) to f32 chunks ``[B, NC, H, C, ...]``."""
    return chunked._to_chunks(jnp.asarray(x, F32), C)


def from_heads(x: jax.Array) -> jax.Array:
    """Head-major ``[B, H, T, D]`` to chunks ``[B, NC, H, C, D]`` (f32)."""
    B, H, T, D = x.shape
    return jnp.moveaxis(x.astype(F32).reshape(B, H, T // C, C, D), 1, 2)


def to_heads(x: jax.Array, dtype=BF16) -> jax.Array:
    """Chunks ``[B, NC, H, C, D]`` to head-major ``[B, H, T, D]``."""
    B, NC, H, _, D = x.shape
    return jnp.moveaxis(x, 1, 2).reshape(B, H, NC * C, D).astype(dtype)


def config(kind: str, x: dict, *, interpret: bool = False) -> vg.Config:
    B, T, H, K = x["q"].shape
    return vg.Config(batch=B, seq_len=T, heads=H, key_dim=K, value_dim=x["v"].shape[-1], scale=K**-0.5,
                     channel_gates=kind == "gdn2", interpret=interpret)


def kernel_gates(kind: str, x: dict) -> tuple[jax.Array, jax.Array]:
    """``(erase, write)`` as the kernels take them."""
    if kind == "kda":
        beta = jnp.swapaxes(x["gates"][0], 1, 2).astype(F32)
        return beta, beta
    return tuple(gate.astype(BF16) for gate in x["gates"])


def oracle_forward(kind: str, x: dict) -> dict:
    """The chunked forward stages in f32 from the (bf16-valued) inputs."""
    qc, kc, vc = chunks(x["q"]), chunks(x["k"]), chunks(x["v"])
    gc = chunked.chunk_local_cumsum(chunks(x["g"]))
    scale = x["q"].shape[-1] ** -0.5
    if kind == "kda":
        beta = chunks(x["gates"][0])
        gate_args, wy_kwargs = (beta,), {}
    else:
        erase, write = (chunks(gate) for gate in x["gates"])
        gate_args, wy_kwargs = (None,), dict(erase=erase, write=write)
    aqk = scale * chunked.decayed_scores(qc, kc, gc, inclusive=True, precision=HIGHEST)
    akk = chunked.chunk_scaled_dot_kkt(kc, *gate_args, gc, erase=wy_kwargs.get("erase"), precision=HIGHEST)
    t = chunked.solve_unit_lower(akk)
    w, u = chunked.recompute_w_u(kc, vc, *gate_args, gc, t, precision=HIGHEST, **wy_kwargs)
    h, v_new, final = chunked.chunk_state_pass(kc, w, u, gc, x["h0"], precision=HIGHEST)
    o = chunked.chunk_output(qc, kc, v_new, h, gc, scale, precision=HIGHEST)
    kg = kc * jnp.exp(gc[..., -1:, :] - gc)
    qg = scale * qc * jnp.exp(gc)
    kd = (beta[..., None] if kind == "kda" else wy_kwargs["erase"]) * kc * jnp.exp(gc)  # w = T kd
    return dict(gc=gc, aqk=aqk, akk=akk, t=t, w=w, u=u, kg=kg, qg=qg, kd=kd, h=h, v_new=v_new, final=final, o=o)


def check_decays(cfg: vg.Config, x: dict, want: dict, stage_args: tuple, aqk: jax.Array, akk: jax.Array) -> None:
    """``intra(decays=True)``: the same score matrices plus the decayed tiles ``qg, kd, kg``."""
    aqk_d, akk_d, *decayed = jax.jit(lambda *a: vg.intra(cfg, *a, decays=True))(x["q"], x["k"], *stage_args)
    np.testing.assert_array_equal(np.asarray(aqk_d, np.float32), np.asarray(aqk, np.float32))
    np.testing.assert_array_equal(np.asarray(akk_d), np.asarray(akk))
    for name, got, ref in zip(("qg", "-kd", "kg"), decayed, (want["qg"], -want["kd"], want["kg"])):
        assert_close(from_heads(got), ref, 1e-2, name)


def check_state_with_t(cfg: vg.Config, x: dict, want: dict, g_last: jax.Array, write: jax.Array) -> None:
    """The state / output pass that applies ``T`` itself (``v_new = T (write v - kd S)``)."""
    t = to_heads(want["t"])
    inputs_ = (to_heads(want["kg"]), to_heads(-want["kd"]), x["v"], g_last, x["h0"], to_heads(want["qg"]),
               to_heads(want["aqk"]))
    o, final = jax.jit(lambda *a: vg.state_output(cfg, *a, save=False, t=t, write=write))(*inputs_)
    assert_close(chunks(o), want["o"], 1e-2, f"o (T in the state pass, block_v {cfg.block_v})")
    assert_close(final, want["final"], 1e-2, f"final (T in the state pass, block_v {cfg.block_v})")


def loss_fn(kind: str, impl: str, cot: jax.Array, cot_s: jax.Array, **options):
    """``sum(o * cot) + sum(final * cot_s)`` of ``ops.linear_attention`` (f32 inputs for the chunked
    oracle, which runs in f32 from the same bf16 values)."""
    from jla import ops

    def loss(q, k, v, g, gates, h0):
        if impl != "mosaic_gpu":
            q, k, v = (a.astype(F32) for a in (q, k, v))
        o, final = ops.linear_attention(kind, q, k, v, g=g, impl=impl, initial_state=h0, output_final_state=True,
                                        **gate_kwargs(kind, gates), **options)
        return jnp.sum(o.astype(F32) * cot) + jnp.sum(final * cot_s)

    return loss


def interpret_loss(kind: str, cot: jax.Array):
    """The same loss through :func:`vector_gate_ops.chunked_vector_gate` in the GPU interpreter."""

    def loss(q, k, v, g, gates, h0):
        o, final = vgo.chunked_vector_gate(q, k, v, g, gates, scale=q.shape[-1] ** -0.5, initial_state=h0,
                                           interpret=True)
        return jnp.sum(o.astype(F32) * cot) + jnp.sum(final)

    return loss


def check_gradients(kind: str, seed: int, shape, *, huge=False, packed=False, no_state=False, l2=False) -> None:
    """Gradients of the kernels against autodiff of the chunked op (``packed``: segment_ids, with
    ``"dense"`` those of :func:`dense_segments`; ``l2``: un-normalized keys and ``use_qk_l2norm``)."""
    B, T, H, K, V = shape
    x = inputs(seed, kind, B, T, H, K, V, huge=huge)
    if l2:
        x["k"] = jax.random.normal(jax.random.key(seed + 7), x["k"].shape).astype(BF16)
    options = dict(use_qk_l2norm=l2)
    if packed:
        options["segment_ids"] = dense_segments(B, T) if packed == "dense" else segments(seed, B, T)
    h0 = None if no_state else x["h0"]
    argnums = (0, 1, 2, 3, 4) if no_state else (0, 1, 2, 3, 4, 5)
    args = (x["q"], x["k"], x["v"], x["g"], x["gates"], h0)
    got = jax.jit(jax.grad(loss_fn(kind, "mosaic_gpu", x["do"], x["dht"], **options), argnums))(*args)
    want = jax.jit(jax.grad(loss_fn(kind, "chunked", x["do"], x["dht"], **options), argnums))(*args)
    names = ["dq", "dk", "dv", "dg", "dgates", "dh0"]
    for name, a, b in zip(names, got, want):
        for i, (ai, bi) in enumerate(zip(jax.tree.leaves(a), jax.tree.leaves(b))):
            assert ai.dtype == bi.dtype, (name, ai.dtype, bi.dtype)
            assert_close(ai, bi, 1.5e-2, f"{name}{i}")
