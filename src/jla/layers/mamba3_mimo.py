# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/modules/mamba3.py,
# mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py, tests/ops/tilelang/test_mamba3_mimo.py
"""Mamba-3 MIMO token mixer (state-spaces/mamba ``Mamba3(is_mimo=True)``).

MIMO of rank ``R`` (``SSMConfig.mimo_rank > 1``) replaces SISO's single
``B_t^T x_t`` write and ``C_t S_t`` read per head by ``R`` of them sharing one
state: ``in_proj`` gives ``R`` copies of ``B`` and ``C`` (layout ``(r g n)``),
``x`` and ``z`` are up-projected per rank by ``mimo_x``/``mimo_z [H, R, P]``
(``x_r = x * mimo_x[h, r]``), the exponential-trapezoidal recurrence
(``ops`` kind ``"mamba3_mimo"``) writes ``sum_r B_r^T x_r`` and reads
``y_r = C_r S_t``, then ``y_r + D x_r`` is gated by ``silu(z_r)`` (or
``rmsnorm(y_r) * silu(z_r)`` per head with ``outproj_norm``) and the ranks are
combined with ``mimo_o [H, R, P]``: ``y = sum_r y_r * mimo_o[h, r]``.

As in the official MIMO kernels (``rotate_pairwise=False``), the data-dependent
rotation pairs channel ``n`` with channel ``N / 2 + n`` for ``n < S`` (``S`` angles
per head, the same for every rank), unlike SISO's interleaved pairs. Everything
else (``A``, ``DT``, the trapezoid ``lambda``, the angle scan, B/C RMS norms and
biases) is the SISO layer's (``jla.layers.ssm``). The layer's chunks hold
``min(SSMConfig.chunk_size, 64 / R)`` tokens (:func:`mimo_chunk_size`; the official ``64 / R``).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.linear import Linear, StackedLinear
from jla.layers.norm import GatedRMSNorm, RMSNorm
from jla.layers.ssm import _dt_bias_init, cumulative_angle, heavy_tail
from jla.layers.ssm import _out_proj_init
from jla.ops.reference import Mamba3State

if TYPE_CHECKING:
    from jla.models.config import SSMConfig

__all__ = ["Mamba3MIMO", "mamba3_mimo", "mimo_chunk_size", "rotate_halves"]

Array = jax.Array
_FUSED_ROWS = 64  # token-rank rows per chunk: the official ``chunk_size = 64 / R`` and the kernels' tile


def mimo_chunk_size(chunk_size: int, rank: int) -> int:
    """Tokens per chunk of the rank-``R`` layer: ``chunk_size`` capped at ``64 / R``.

    state-spaces/mamba recommends ``64 / R`` for MIMO (64 fused token-rank rows), which is the tile of
    the Mosaic GPU kernels. The cap lets an ``SSMConfig`` with the SISO default (64) run them.
    """
    return max(1, min(chunk_size, _FUSED_ROWS // rank))


def rotate_halves(x: Array, angles: Array) -> Array:
    """Rotate channel ``n`` with channel ``N / 2 + n`` of ``x [..., N]`` by ``angles [..., S]``, ``n < S``.

    The official MIMO rotation (``rotate_pairwise=False``); the other channels pass through.
    """
    num, half = angles.shape[-1], x.shape[-1] // 2
    if num > half:
        raise ValueError(f"{num} angles cannot rotate the halves of {x.shape[-1]} channels")
    x32 = x.astype(jnp.float32)
    first, second = x32[..., :num], x32[..., half : half + num]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    out = x32.at[..., :num].set(first * cos - second * sin)
    out = out.at[..., half : half + num].set(first * sin + second * cos)
    return out.astype(x.dtype)


# The q (C) and k (B) operands of the core: C/B of every rank repeated to the heads in float32, plus the per-head
# bias, rotated, cast to the core's dtype. Plain autodiff of this expression fuses the inverse rotation into one
# reduction per layer that sums the bias gradient over tokens at ~100 GB/s (about a quarter of the 1.3B MIMO
# hybrid's training step on an H100), so it has a custom VJP: the same forward ops, and a backward that writes the
# inverse-rotated cotangent once in float32, sums it with plain reductions (bias over tokens, C/B over heads) and
# takes the angle gradient in one fusion. Same float32 math as autodiff, only summed in another order.


def _per_head(x: Array, bias: Array) -> Array:
    """``x [B, T, R, G, N]`` -> float32 ``[B, T, H, R, N]`` (head ``h`` reads group ``h // (H / G)``) + ``bias``."""
    heads, groups = bias.shape[0], x.shape[3]
    # Repeat in float32 (same values), so the transpose sums the per-head gradients in float32.
    x = x.astype(jnp.float32)
    x = x if groups == heads else jnp.repeat(x, heads // groups, axis=3)
    return jnp.moveaxis(x, 2, 3) + bias.astype(jnp.float32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _expand_rotate(x: Array, bias: Array, angle: Array, dtype: jnp.dtype) -> Array:
    """``rotate_halves(per-head x + bias, angle)`` in ``dtype``: ``x [B, T, R, G, N]``, ``bias [H, R, N]``,
    ``angle [B, T, H, S]`` -> ``[B, T, H, R, N]``."""
    return rotate_halves(_per_head(x, bias), angle[:, :, :, None]).astype(dtype)


def _expand_rotate_fwd(x, bias, angle, dtype):
    return _expand_rotate(x, bias, angle, dtype), (x, bias, angle)


def _expand_rotate_bwd(dtype, res, g):
    del dtype
    x, bias, angle = res
    f32 = jnp.float32
    batch, length, heads, rank, n = g.shape
    groups, num, half = x.shape[3], angle.shape[-1], n // 2
    g = g.astype(f32)
    cos, sin = jnp.cos(angle)[:, :, :, None], jnp.sin(angle)[:, :, :, None]
    g1, g2 = g[..., :num], g[..., half : half + num]
    # d(per-head x + bias): the cotangent rotated back, written to memory once (the barrier keeps XLA from fusing
    # it into the reductions below again).
    u = jnp.concatenate([g1 * cos + g2 * sin, g[..., num:half], g2 * cos - g1 * sin, g[..., half + num :]], axis=-1)
    u = lax.optimization_barrier(u)
    dbias = jnp.sum(u, axis=(0, 1)).astype(bias.dtype)
    dx = jnp.sum(u.reshape(batch, length, groups, heads // groups, rank, n), axis=3)  # [B, T, G, R, N]
    dx = jnp.moveaxis(dx, 2, 3).astype(x.dtype)
    # Angle: d/dtheta of (p1 cos - p2 sin, p1 sin + p2 cos), summed over the ranks, as autodiff takes it.
    p1 = _per_head(x[..., :num], bias[..., :num])
    p2 = _per_head(x[..., half : half + num], bias[..., half : half + num])
    ct_cos = jnp.sum(g1 * p1, axis=3) + jnp.sum(g2 * p2, axis=3)
    ct_sin = jnp.sum(g2 * p1, axis=3) - jnp.sum(g1 * p2, axis=3)
    dangle = (ct_sin * jnp.cos(angle) - ct_cos * jnp.sin(angle)).astype(angle.dtype)
    # The three leave together, so the bias gradient (used only by the optimizer) is summed here and not whenever
    # XLA schedules it while ``u`` waits: without this barrier the 1.3B hybrid's step no longer fit XLA's default
    # allocator.
    return lax.optimization_barrier((dx, dbias, dangle))


_expand_rotate.defvjp(_expand_rotate_fwd, _expand_rotate_bwd)


def mamba3_mimo(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    mimo_x: Array,
    *,
    mimo_z: Array | None = None,
    mimo_o: Array | None = None,
    d: Array | None = None,
    z: Array | None = None,
    segment_ids: Array | None = None,
    initial_state: tuple[Array, Mamba3State] | None = None,
    output_final_state: bool = False,
    impl: str = "auto",
    chunk_size: int = 16,
) -> tuple[Array, tuple[Array, Mamba3State] | None]:
    """Mamba-3 MIMO core with the semantics of state-spaces/mamba's ``mamba3_mimo`` (no fused norm).

    Official layout: ``q, k [B, T, R, G, N]`` (the RMS-normalized C and B of every rank,
    ``G`` groups), ``v [B, T, H, P]`` (x), ``adt, dt, trap [B, T, H]``, ``q_bias, k_bias
    [H, R, N]``, ``angles [B, T, H, S]`` (pre-``tanh``), ``mimo_x, mimo_z, mimo_o [H, R, P]``,
    ``d [H]``, ``z [B, T, H, P]`` (needs ``mimo_z``). The state is ``(angle [B, H, S],
    Mamba3State)`` with ``key [B, H, R, N]`` and ``value [B, H, R, P]``.

    Returns ``y [B, T, H, P]`` (``sum_r y_r * mimo_o[h, r]``) when ``mimo_o`` is given, else
    the per-rank ``y [B, T, H, R, P]`` (for the layer's ``outproj_norm``), in ``v.dtype``.
    """
    groups = q.shape[3]
    heads = v.shape[2]
    if k.shape != q.shape or heads % groups:
        raise ValueError(f"k must match q {q.shape} and {heads} heads must split into {groups} groups")
    if (z is None) != (mimo_z is None):
        raise ValueError("z and mimo_z go together")
    f32 = jnp.float32
    dt32, adt32 = dt.astype(f32), adt.astype(f32)
    bias_shape = (heads,) + q.shape[2:3] + q.shape[4:]  # [H, R, N]
    angle_state, ssm_state = (None, None) if initial_state is None else initial_state
    increments = jnp.tanh(angles.astype(f32)) * math.pi * dt32[..., None]
    angle = cumulative_angle(increments, segment_ids=segment_ids, initial=angle_state)
    q = _expand_rotate(q, jnp.broadcast_to(q_bias, bias_shape), angle, v.dtype)  # [B, T, H, R, N]
    k = _expand_rotate(k, jnp.broadcast_to(k_bias, bias_shape), angle, v.dtype)
    vr = (v.astype(f32)[:, :, :, None] * mimo_x.astype(f32)).astype(v.dtype)  # [B, T, H, R, P]
    lam = jax.nn.sigmoid(trap.astype(f32))
    y, final = ops.linear_attention(
        "mamba3_mimo", q, k, vr, g=adt32,
        beta=(1.0 - lam) * dt32 * jnp.exp(adt32), gamma=lam * dt32, scale=1.0, impl=impl,
        chunk_size=chunk_size, segment_ids=segment_ids, initial_state=ssm_state,
        output_final_state=output_final_state,
    )
    y = y.astype(f32)
    if d is not None:
        y = y + d.astype(f32)[:, None, None] * vr.astype(f32)
    if z is not None:
        y = y * jax.nn.silu(z.astype(f32)[:, :, :, None] * mimo_z.astype(f32))
    if mimo_o is not None:
        y = jnp.sum(y * mimo_o.astype(f32), axis=3)
    state = (angle[:, -1], final) if output_final_state else None
    return y.astype(v.dtype), state


class Mamba3MIMO(eqx.Module):
    """Mamba-3 MIMO token mixer (Lahoti et al. 2026), following state-spaces/mamba ``Mamba3(is_mimo=True)``.

    ``out_proj`` starts at ``N(0, std)`` (truncated) when ``n_residuals`` is None (``init_scheme="jla"``),
    else at ``U(-1/sqrt(inner), 1/sqrt(inner)) / sqrt(n_residuals)`` as in the SISO layer
    (:class:`jla.layers.ssm.Mamba3`; state-spaces/mamba ``MixerModel._init_weights``).
    """

    zx_proj: StackedLinear  # rows z, x of in_proj
    bc_proj: StackedLinear  # rows B, C (each R * G * N, rank-major)
    scalar_proj: StackedLinear  # rows dd_dt, dd_A, trap (one value per head each)
    angle_proj: Linear  # rows angles (shared by all heads)
    dt_bias: Array
    B_bias: Array  # [H, R, N]
    C_bias: Array  # [H, R, N]
    B_norm: RMSNorm
    C_norm: RMSNorm
    D: Array
    mimo_x: Array  # [H, R, P]
    mimo_z: Array  # [H, R, P]
    mimo_o: Array  # [H, R, P]
    norm: GatedRMSNorm | None
    out_proj: Linear
    cfg: SSMConfig = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_angles: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: SSMConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16", n_residuals: int | None = None):
        heads = cfg.num_heads(d_model)
        inner = heads * cfg.head_dim
        rank = cfg.mimo_rank
        out_init = _out_proj_init(n_residuals)
        if rank < 2:
            raise ValueError(f"Mamba3MIMO needs mimo_rank >= 2 (1 is the SISO Mamba3); got {rank}")
        if cfg.d_state % 2:
            raise ValueError(f"MIMO rotates the halves of d_state; {cfg.d_state} is odd")
        num_angles = int(cfg.d_state * cfg.rope_fraction) // 2
        if num_angles < 1:
            raise ValueError(f"d_state {cfg.d_state} with rope_fraction {cfg.rope_fraction} rotates no channel pair")
        ks = jax.random.split(key, 6)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.zx_proj = StackedLinear(d_model, 2, inner, key=ks[0], out_axis="heads", **common)
        self.bc_proj = StackedLinear(d_model, 2, rank * cfg.n_groups * cfg.d_state, key=ks[1], out_axis=None,
                                     **common)
        self.scalar_proj = StackedLinear(d_model, 3, heads, key=ks[2], out_axis=None, **common)
        self.angle_proj = Linear(d_model, num_angles, key=ks[3], axes=("embed", None), **common)
        self.dt_bias = _dt_bias_init(ks[4], heads, cfg).astype(param_dtype)
        self.B_bias = jnp.ones((heads, rank, cfg.d_state), param_dtype)
        self.C_bias = jnp.ones((heads, rank, cfg.d_state), param_dtype)
        self.B_norm = RMSNorm(cfg.d_state, eps=norm_eps, param_dtype=param_dtype)
        self.C_norm = RMSNorm(cfg.d_state, eps=norm_eps, param_dtype=param_dtype)
        self.D = jnp.ones((heads,), param_dtype)
        self.mimo_x = jnp.full((heads, rank, cfg.head_dim), 1.0 / rank, param_dtype)
        self.mimo_z = jnp.ones((heads, rank, cfg.head_dim), param_dtype)
        self.mimo_o = jnp.full((heads, rank, cfg.head_dim), 1.0 / rank, param_dtype)
        self.norm = (
            GatedRMSNorm(inner, eps=norm_eps, group_size=cfg.head_dim, param_dtype=param_dtype)
            if cfg.outproj_norm else None
        )
        self.out_proj = Linear(inner, d_model, key=ks[5], axes=("heads", "embed"), **out_init, **common)
        self.cfg = cfg
        self.num_heads = heads
        self.num_angles = num_angles
        self.rank = rank

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        three = (None, None, None)
        return {"dt_bias": (None,), "B_bias": three, "C_bias": three, "D": (None,), "mimo_x": three,
                "mimo_z": three, "mimo_o": three}

    def _forward(self, u: Array, *, segment_ids: Array | None, cache: RecurrentCache | None):
        cfg, heads, rank = self.cfg, self.num_heads, self.rank
        batch, length, _ = u.shape
        zx = self.zx_proj(u)
        z = zx[..., 0, :].reshape(batch, length, heads, cfg.head_dim)
        x = zx[..., 1, :].reshape(batch, length, heads, cfg.head_dim)
        bc = self.bc_proj(u).reshape(batch, length, 2, rank, cfg.n_groups, cfg.d_state)
        b, c = self.B_norm(bc[:, :, 0]), self.C_norm(bc[:, :, 1])
        scalars = self.scalar_proj(u).astype(jnp.float32)
        dt = jax.nn.softplus(scalars[..., 0, :] + self.dt_bias.astype(jnp.float32))
        a = jnp.minimum(-heavy_tail(scalars[..., 1, :]), -cfg.a_floor)
        angles = jnp.broadcast_to(self.angle_proj(u).astype(jnp.float32)[:, :, None, :],
                                  (batch, length, heads, self.num_angles))
        impl = "reference" if cache is not None and length == 1 else cfg.implementation
        gated = self.norm is None
        y, state = mamba3_mimo(
            c, b, x, a * dt, dt, scalars[..., 2, :], self.C_bias, self.B_bias, angles, self.mimo_x,
            mimo_z=self.mimo_z if gated else None, mimo_o=self.mimo_o if gated else None, d=self.D,
            z=z if gated else None, segment_ids=segment_ids,
            initial_state=None if cache is None else (cache.angle, cache.state),
            output_final_state=cache is not None, impl=impl, chunk_size=mimo_chunk_size(cfg.chunk_size, rank),
        )
        if not gated:  # rmsnorm(y_r) * silu(z_r) per rank, then the rank combination
            z_r = z.astype(jnp.float32)[:, :, :, None] * self.mimo_z.astype(jnp.float32)  # [B, T, H, R, P]
            per_rank = lambda t: jnp.moveaxis(t, 3, 2).reshape(batch, length, rank, -1)  # noqa: E731
            y = self.norm(per_rank(y), per_rank(z_r).astype(y.dtype))
            y = jnp.moveaxis(y.reshape(batch, length, rank, heads, cfg.head_dim), 2, 3)
            y = jnp.sum(y.astype(jnp.float32) * self.mimo_o.astype(jnp.float32), axis=3).astype(x.dtype)
        y = checkpoint_name(y.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        out = self.out_proj(y)
        return out, None if cache is None else RecurrentCache(state=state[1], angle=state[0])

    def __call__(self, u: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(u, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg, heads, rank = self.cfg, self.num_heads, self.rank
        f32 = jnp.float32
        state = Mamba3State(
            jnp.zeros((batch, heads, cfg.d_state, cfg.head_dim), f32),
            jnp.zeros((batch, heads, rank, cfg.d_state), f32),
            jnp.zeros((batch, heads, rank, cfg.head_dim), f32),
        )
        return RecurrentCache(state=state, angle=jnp.zeros((batch, heads, self.num_angles), f32))

    def step(self, u: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``u [B, T, D]`` from ``cache``."""
        return self._forward(u, segment_ids=None, cache=cache)
