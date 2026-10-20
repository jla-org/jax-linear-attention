# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/modules/mamba_simple.py,
# mamba_ssm/ops/selective_scan_interface.py
"""Mamba-1 token mixer (Gu and Dao 2023): the selective SSM of Mamba, Jamba, Falcon-Mamba and Codestral Mamba.

Follows state-spaces/mamba ``Mamba`` (``mamba_simple.py``): ``x, z = in_proj(u)``
(no bias); ``x`` goes through a depthwise causal conv with bias and SiLU;
``dt, B, C = x_proj(x)`` with ``dt`` of rank ``dt_rank = ceil(d_model / 16)``;
``delta = softplus(dt_proj.weight dt + dt_proj.bias)`` (the bias is added inside the
scan in float32, the official ``delta_bias``); ``A = -exp(A_log)`` per channel and
state (``A_log = log(1 .. N)``, S4D-real); the selective scan
``h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t``, ``y_t = C_t h_t + D x_t`` runs as
``ops.linear_attention("mamba1", q=C, k=B, v=x, dt=dt, a=A)``; then ``y * silu(z)``
and ``out_proj``. ``dt_proj`` keeps the official initialization: weights
``U(-dt_scale / sqrt(dt_rank), +dt_scale / sqrt(dt_rank))`` and a bias that is the
inverse softplus of ``dt`` log-uniform in ``[dt_min, dt_max]`` (floored at
``dt_init_floor``).

The config is :class:`jla.models.config.SSMConfig` (``d_state``, ``expand``,
``conv_size``, ``dt_*``, ``dt_rank``, ``dt_scale``, ``chunk_size``,
``implementation``, ``conv_impl``); Mamba-1 has no heads, so ``head_dim`` only has to
divide the inner width, and ``n_groups`` must be 1. ``conv_impl`` picks the conv's
implementation as for Mamba-2 (``"auto"``: the fused Mosaic GPU short-conv kernels of
:mod:`jla.ops.mosaic_gpu.short_conv` on Hopper, XLA where they decline).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.conv import ShortConv
from jla.layers.linear import Linear, StackedLinear

if TYPE_CHECKING:
    from jla.models.config import SSMConfig

__all__ = ["Mamba1", "mamba1_dt_rank"]

Array = jax.Array


def mamba1_dt_rank(d_model: int, cfg: SSMConfig) -> int:
    """``cfg.dt_rank``, or the official ``"auto"`` rank ``ceil(d_model / 16)``."""
    rank = math.ceil(d_model / 16) if cfg.dt_rank is None else cfg.dt_rank
    if rank < 1:
        raise ValueError(f"dt_rank must be >= 1; got {rank}")
    return rank


class Mamba1(eqx.Module):
    """Mamba-1 selective-SSM token mixer, following state-spaces/mamba ``Mamba``."""

    in_proj: StackedLinear  # rows x, z of the official in_proj
    conv: ShortConv  # depthwise causal conv1d on x (bias, SiLU)
    x_proj: Linear  # inner -> dt_rank + 2 d_state (dt, B, C)
    dt_proj: Linear  # dt_rank -> inner, weight only; the bias is dt_bias
    dt_bias: Array  # [inner], the official dt_proj.bias
    A_log: Array  # [inner, d_state]
    D: Array  # [inner]
    out_proj: Linear
    cfg: SSMConfig = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: SSMConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        del norm_eps  # Mamba-1 has no normalization inside the mixer
        if cfg.n_groups != 1:
            raise ValueError(f"Mamba-1 shares one B and C across all channels; n_groups must be 1, got {cfg.n_groups}")
        if cfg.dt_scale <= 0:
            raise ValueError(f"dt_scale must be positive; got {cfg.dt_scale}")
        from jla.models.config import SSMConfig as _Defaults  # runtime import; the module-level one is typing-only

        unused = [f for f in ("outproj_norm", "a_init_range", "a_floor", "rope_fraction")
                  if getattr(cfg, f) != getattr(_Defaults(), f)]
        if unused:
            raise ValueError(f"Mamba-1 does not use the Mamba-2/3 options {unused}; leave them at their defaults")
        inner = cfg.expand * d_model
        rank = mamba1_dt_rank(d_model, cfg)
        n = cfg.d_state
        ks = jax.random.split(key, 7)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.in_proj = StackedLinear(d_model, 2, inner, key=ks[0], out_axis="heads", **common)
        self.conv = ShortConv(inner, cfg.conv_size, key=ks[1], axis="heads", activation=True, use_bias=True,
                              init="uniform", param_dtype=param_dtype, impl=cfg.conv_impl)
        self.x_proj = Linear(inner, rank + 2 * n, key=ks[2], axes=("heads", None), **common)
        dt_proj = Linear(rank, inner, key=ks[3], axes=(None, "heads"), **common)
        bound = rank**-0.5 * cfg.dt_scale
        weight = jax.random.uniform(ks[4], (rank, inner), jnp.float32, -bound, bound).astype(param_dtype)
        self.dt_proj = eqx.tree_at(lambda m: m.weight, dt_proj, weight)
        u = jax.random.uniform(ks[5], (inner,), jnp.float32)
        dt = jnp.exp(u * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min))
        dt = jnp.maximum(dt, cfg.dt_init_floor)
        self.dt_bias = (dt + jnp.log(-jnp.expm1(-dt))).astype(param_dtype)
        a = jnp.broadcast_to(jnp.arange(1, n + 1, dtype=jnp.float32), (inner, n))
        self.A_log = jnp.log(a).astype(param_dtype)
        self.D = jnp.ones((inner,), param_dtype)
        self.out_proj = Linear(inner, d_model, key=ks[6], axes=("heads", "embed"), **common)
        self.cfg = cfg
        self.d_inner = inner
        self.dt_rank = rank

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"dt_bias": ("heads",), "A_log": ("heads", None), "D": ("heads",)}

    def _forward(self, u: Array, *, segment_ids: Array | None, cache: RecurrentCache | None):
        cfg = self.cfg
        length = u.shape[1]
        xz = self.in_proj(u)
        x, z = xz[..., 0, :], xz[..., 1, :]
        x, conv_state = self.conv(x, segment_ids=segment_ids, state=None if cache is None else cache.conv)
        x_dbl = self.x_proj(x)
        dt_low = x_dbl[..., : self.dt_rank]
        b = x_dbl[..., self.dt_rank : self.dt_rank + cfg.d_state]
        c = x_dbl[..., self.dt_rank + cfg.d_state :]
        dt = self.dt_proj(dt_low)
        a = -jnp.exp(self.A_log.astype(jnp.float32))
        impl = "reference" if cache is not None and length == 1 else cfg.implementation
        y, state = ops.linear_attention(
            "mamba1", c, b, x, dt=dt, a=a, d=self.D, z=z, dt_bias=self.dt_bias, dt_softplus=True,
            segment_ids=segment_ids, initial_state=None if cache is None else cache.state,
            output_final_state=cache is not None, impl=impl, chunk_size=cfg.chunk_size,
        )
        y = checkpoint_name(y, ATTENTION_CHECKPOINT_NAME)
        out = self.out_proj(y)
        return out, None if cache is None else RecurrentCache(state=state, conv=conv_state)

    def __call__(self, u: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(u, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        return RecurrentCache(
            state=jnp.zeros((batch, self.d_inner, self.cfg.d_state), jnp.float32),
            conv=jnp.zeros((batch, self.cfg.conv_size - 1, self.d_inner), dtype),
        )

    def step(self, u: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``u [B, T, D]`` from ``cache``."""
        return self._forward(u, segment_ids=None, cache=cache)
