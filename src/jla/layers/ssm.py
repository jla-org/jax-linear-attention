# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from state-spaces/mamba (Apache-2.0), mamba_ssm/modules/mamba2.py, mamba_ssm/modules/mamba3.py,
# tests/ops/triton/test_mamba3_siso.py
# Adapted from flash-linear-attention (MIT), fla/layers/log_linear_mamba2.py (the log-linear Mamba-2 option)
"""State-space token mixers: Mamba-2 (SSD) and Mamba-3 SISO.

Both follow the official state-spaces/mamba modules. The fused ``in_proj`` of the
official code is stored as role-grouped projections (rows ``[z, x]``, ``[B, C]``
and the per-head scalars), so the large ``z``/``x`` outputs can be sharded by head;
concatenating them in the official order recovers ``in_proj``.

Mamba-2 (``mamba2.py``): ``z, xBC, dt = in_proj(u)``; ``xBC`` goes through a
depthwise causal conv with bias and SiLU; ``dt = softplus(dt + dt_bias)``,
``A = -exp(A_log)``; the SSD scan ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t^T``,
``y_t = C_t h_t + D x_t`` runs as ``simple_gla(q=C, k=B, v=dt*x, g=dt*A, scale=1)``;
then ``rmsnorm(y * silu(z))`` (``norm_before_gate=False``, one group per B/C group)
and ``out_proj``.

Mamba-3 SISO (``mamba3.py``, reference ``mamba3_siso_step_ref``): ``in_proj`` gives
``[z, x, B, C, dd_dt, dd_A, trap, angles]``; ``A = min(-heavy_tail(dd_A), -A_floor)``,
``DT = softplus(dd_dt + dt_bias)``, ``ADT = A * DT``; B and C are RMS-normalized,
shifted by ``B_bias``/``C_bias`` and rotated by the cumulative data-dependent angle
``theta_t = sum_{s<=t} tanh(angles_s) * pi * DT_s (mod 2 pi)`` on interleaved pairs
of the first ``rope_fraction * d_state`` channels; the exponential-trapezoidal
recurrence uses ``lambda = sigmoid(trap)``, ``beta = (1 - lambda) DT exp(ADT)`` and
``gamma = lambda DT`` (``ops`` kind ``"mamba3"``); then the ``D`` skip, the gate
``y * silu(z)`` (or ``rmsnorm(y) * silu(z)`` per head with ``outproj_norm``) and
``out_proj``. Mamba-3 has no convolution.
"""

from __future__ import annotations

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
from jla.layers.conv import ShortConv
from jla.layers.linear import Linear, StackedLinear
from jla.layers.norm import GatedRMSNorm, RMSNorm
from jla.ops.reference import Mamba3State, segment_resets

if TYPE_CHECKING:
    from jla.models.config import SSMConfig

__all__ = [
    "Mamba2",
    "Mamba3",
    "cumulative_angle",
    "heavy_tail",
    "mamba2_ssd",
    "mamba3_siso",
    "rotate_pairs",
]

Array = jax.Array
_TWO_PI = 2.0 * math.pi


def _dt_bias_init(key: Array, size: int, cfg: SSMConfig) -> Array:
    """Inverse softplus of ``dt`` log-uniform in ``[dt_min, dt_max]``, floored at ``dt_init_floor``."""
    u = jax.random.uniform(key, (size,), jnp.float32)
    dt = jnp.exp(u * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min))
    dt = jnp.maximum(dt, cfg.dt_init_floor)
    return dt + jnp.log(-jnp.expm1(-dt))


def _repeat_groups(x: Array, heads: int) -> Array:
    """``[B, T, G, N]`` -> ``[B, T, H, N]``: head ``h`` uses group ``h // (H / G)``.

    The repeat goes through float32 (the values are unchanged), so its transpose sums the per-head
    gradients in float32 and rounds once; summed in bfloat16 they lose about 2x accuracy at 8 heads
    per group (``benchmarks/xcheck_scalar_gate.py``, ``mosaic_rep_bf16`` against ``mosaic_rep_f32``).
    """
    groups = x.shape[2]
    return x if groups == heads else jnp.repeat(x.astype(jnp.float32), heads // groups, axis=2).astype(x.dtype)


GROUPED_BC = True
"""Mamba-2: hand grouped ``B, C [B, T, G, N]`` to the Mosaic GPU simple GLA, which reads group
``h // (H / G)`` for value head ``h`` (its backward sums ``dB, dC`` over a group's heads in float32),
instead of repeating them per head in XLA, which recomputed the B/C conv for every head. False (or a
call that kernel does not take) repeats them."""


def _grouped_simple_gla(impl: str, q: Array, k: Array, v: Array, g: Array, **options) -> bool:
    """True when ``impl`` resolves to the Mosaic GPU simple GLA and it accepts grouped ``q, k``."""
    if not GROUPED_BC or impl not in ("auto", "mosaic_gpu") or q.shape[2] == v.shape[2]:
        return False
    try:
        import jax.experimental.pallas.mosaic_gpu  # noqa: F401
    except ImportError:
        return False
    from jla.ops.mosaic_gpu import scalar_gate

    return scalar_gate.supports_simple_gla(q, k, v, g, **options)


SPLIT_ZX = True
"""Compute the ``z`` and ``x`` rows of ``zx_proj`` as two GEMMs with contiguous outputs instead of one
GEMM whose ``[B, T, 2, inner]`` output is sliced (the slices cost a copy forward and a concatenation of
``dz, dx`` backward). Same parameters and values; False runs the single stacked GEMM. On the fused-glue path the
GEMMs after the first read the input through an optimization barrier: XLA's dot merger otherwise fuses GEMMs that
share their input back into one (it did at ``d_model`` 768, where the merged GEMM's two row copies and the
``dz, dx`` concatenation cost 14% of the layer's forward and 7% of its forward + backward on an H100)."""


def _split_rows(proj: StackedLinear, u: Array, *, separate: bool = False) -> tuple[Array, ...]:
    """``proj(u)[..., i, :]`` for every stacked row ``i``, each from its own GEMM (``separate``: kept apart from
    XLA's dot merger by a barrier, see :data:`SPLIT_ZX`).

    An FP8 projection (``model.fp8``: not a :class:`StackedLinear`) keeps its own scaled GEMM.
    """
    if not SPLIT_ZX or not isinstance(proj, StackedLinear):
        out = proj(u)
        return tuple(out[..., i, :] for i in range(out.shape[-2]))
    dtype = jnp.dtype(proj.compute_dtype)
    u, w = u.astype(dtype), proj.weight.astype(dtype)
    return tuple(jnp.einsum("...d,do->...o", lax.optimization_barrier(u) if separate and i else u, w[:, i])
                 for i in range(w.shape[1]))


def heavy_tail(x: Array) -> Array:
    """Mamba-3's activation for the data-dependent ``A``: ``1 + x`` for ``x >= 0``, ``1 / (1 - x)`` below."""
    return jnp.maximum(x, 0.0) + 1.0 / (1.0 - jnp.minimum(x, 0.0))


def rotate_pairs(x: Array, angles: Array) -> Array:
    """Rotate the interleaved pairs ``(2i, 2i + 1)`` of ``x [..., N]`` by ``angles [..., S]``, ``i < S``.

    Channels ``2S ..`` pass through (Mamba-3's partial rotation, ``rotate_pairwise``).
    """
    num = angles.shape[-1]
    x32 = x.astype(jnp.float32)
    rot, rest = x32[..., : 2 * num], x32[..., 2 * num :]
    pairs = rot.reshape(*rot.shape[:-1], num, 2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    first = pairs[..., 0] * cos - pairs[..., 1] * sin
    second = pairs[..., 0] * sin + pairs[..., 1] * cos
    rotated = jnp.stack([first, second], axis=-1).reshape(rot.shape)
    return jnp.concatenate([rotated, rest], axis=-1).astype(x.dtype)


def cumulative_angle(
    increments: Array,
    *,
    segment_ids: Array | None = None,
    initial: Array | None = None,
) -> Array:
    """``theta_t = (initial + sum of increments up to t within t's document) mod 2 pi``.

    ``increments [B, T, ...]``; ``initial [B, ...]`` applies to each row's first
    document. A segmented associative scan keeps every partial sum in ``[0, 2 pi)``,
    so precision does not degrade with sequence length.
    """
    inc = increments.astype(jnp.float32)
    if initial is not None:
        inc = inc.at[:, 0].add(initial.astype(jnp.float32))
    inc = jnp.mod(inc, _TWO_PI)
    if segment_ids is None:
        resets = jnp.zeros(inc.shape[:2], bool)
    else:
        resets = segment_resets(segment_ids)
    resets = jnp.broadcast_to(resets.reshape(resets.shape + (1,) * (inc.ndim - 2)), inc.shape)

    def combine(left, right):
        value_l, reset_l = left
        value_r, reset_r = right
        return jnp.where(reset_r, value_r, jnp.mod(value_l + value_r, _TWO_PI)), reset_l | reset_r

    angle, _ = lax.associative_scan(combine, (inc, resets), axis=1)
    return angle


def mamba2_ssd(
    x: Array,
    dt: Array,
    a: Array,
    b: Array,
    c: Array,
    *,
    d: Array | None = None,
    segment_ids: Array | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    impl: str = "auto",
    chunk_size: int = 64,
    grouped_key_grads: bool = False,
) -> tuple[Array, Array | None]:
    """Mamba-2 SSD scan: ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t^T x_t`` and ``y_t = C_t h_t + D x_t``.

    ``x [B, T, H, P]``, ``dt [B, T, H]`` (positive, after softplus), ``a [H]``
    (negative), ``b, c [B, T, G, N]`` with ``H % G == 0``, ``d [H]``; the state is
    ``[B, H, N, P]`` float32. Runs ``ops.linear_attention("simple_gla", q=C, k=B,
    v=dt*x, g=dt*A, scale=1.0)``. ``grouped_key_grads`` (``SSMConfig.grouped_key_grads``)
    has the Mosaic GPU backward sum ``dB, dC`` over each group's heads in its kernel.
    """
    heads = x.shape[2]
    dt32 = dt.astype(jnp.float32)
    v = (x.astype(jnp.float32) * dt32[..., None]).astype(x.dtype).astype(c.dtype)
    g = dt32 * a.astype(jnp.float32)
    options = dict(scale=1.0, chunk_size=chunk_size, segment_ids=segment_ids, initial_state=initial_state,
                   output_final_state=output_final_state)
    if grouped_key_grads and _grouped_simple_gla(impl, c, b.astype(c.dtype), v, g, **options):
        from jla.distributed.kernel_sharding import shard_kernel_call
        from jla.ops.mosaic_gpu import scalar_gate

        def grouped(q, k, v, g, chunk_size=None, **rest):
            del chunk_size  # the kernels use their own chunk length
            return scalar_gate.run(q, k, v, g, None, delta=False, grouped_key_grads=True, **rest)

        # Per shard of the context mesh, like the dispatcher's Mosaic GPU calls (jla.distributed.kernel_sharding).
        y, state = shard_kernel_call(grouped, c, b.astype(c.dtype), v, (g,), options,
                                     supports=scalar_gate.supports_simple_gla, kind="simple_gla")
    elif _grouped_simple_gla(impl, c, b.astype(c.dtype), v, g, **options):
        y, state = ops.linear_attention("simple_gla", c, b.astype(c.dtype), v, g=g, impl="mosaic_gpu", **options)
    else:
        q, k = _repeat_groups(c, heads), _repeat_groups(b, heads)
        y, state = ops.linear_attention("simple_gla", q, k.astype(q.dtype), v, g=g, impl=impl, **options)
    if d is not None:
        y = (y.astype(jnp.float32) + d.astype(jnp.float32)[:, None] * x.astype(jnp.float32)).astype(y.dtype)
    return y, state


def _fused_core():
    """``jla.ops.mosaic_gpu.mamba3_fused`` when JAX has the Mosaic GPU toolchain, else None."""
    try:
        import jax.experimental.pallas.mosaic_gpu  # noqa: F401
    except ImportError:
        return None
    from jla.ops.mosaic_gpu import mamba3_fused

    return mamba3_fused


def mamba3_siso(
    q: Array,
    k: Array,
    v: Array,
    adt: Array,
    dt: Array,
    trap: Array,
    q_bias: Array,
    k_bias: Array,
    angles: Array,
    *,
    d: Array | None = None,
    z: Array | None = None,
    segment_ids: Array | None = None,
    initial_state: tuple[Array, Mamba3State] | None = None,
    output_final_state: bool = False,
    impl: str = "auto",
    chunk_size: int = 64,
) -> tuple[Array, tuple[Array, Mamba3State] | None]:
    """Mamba-3 SISO core with the semantics of state-spaces/mamba's ``mamba3_siso_combined``.

    FLA layout: ``q, k [B, T, G, N]`` (the RMS-normalized C and B, ``G`` groups),
    ``v [B, T, H, P]`` (x), ``adt, dt, trap [B, T, H]`` (``A * DT``, ``DT``, trapezoid
    logits), ``q_bias, k_bias [H, N]``, ``angles [B, T, H, S]`` (pre-``tanh``),
    ``d [H]``, ``z [B, T, H, P]``. The state is ``(angle [B, H, S], Mamba3State)``.
    Returns ``y [B, T, H, P]`` in ``v.dtype``.

    ``impl="mosaic_gpu"`` and ``"auto"`` (while ``mamba3_fused.AUTO``) run the core's biases,
    angle scan, rotation and recurrence as Mosaic GPU kernels
    (``jla.ops.mosaic_gpu.mamba3_fused``) for the calls it supports; other calls run
    the XLA glue below around ``ops.linear_attention("mamba3", impl=impl)``.
    """
    if impl in ("auto", "mosaic_gpu"):
        fused = _fused_core()
        options = dict(d=d, z=z, segment_ids=segment_ids, initial_state=initial_state,
                       output_final_state=output_final_state, chunk_size=chunk_size)
        if (
            fused is not None
            and (impl == "mosaic_gpu" or fused.AUTO)
            and fused.supports_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, **options)
        ):
            from jla.distributed.kernel_sharding import shard_layout_call

            # Custom calls XLA cannot partition: under a mesh they run per batch shard.
            def core(q, k, v, adt, dt, trap, angles, z, segment_ids, initial_state, q_bias, k_bias, d):
                return fused.mamba3_siso_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, **{
                    **options, "d": d, "z": z, "segment_ids": segment_ids, "initial_state": initial_state})

            def core_supported(q, k, v, adt, dt, trap, angles, z, segment_ids, initial_state, q_bias, k_bias, d):
                return fused.supports_fused(q, k, v, adt, dt, trap, q_bias, k_bias, angles, **{
                    **options, "d": d, "z": z, "segment_ids": segment_ids, "initial_state": initial_state})

            operands = dict(q=q, k=k, v=v, adt=adt, dt=dt, trap=trap, angles=angles, z=z, segment_ids=segment_ids,
                            initial_state=initial_state, q_bias=q_bias, k_bias=k_bias, d=d)
            layouts = {name: "replicated" if name in ("q_bias", "k_bias", "d") else "batch" for name in operands}
            return shard_layout_call(core, operands, layouts, out_layouts="batch", supports=core_supported)
    heads = v.shape[2]
    dt32, adt32 = dt.astype(jnp.float32), adt.astype(jnp.float32)
    q = _repeat_groups(q.astype(jnp.float32), heads) + q_bias.astype(jnp.float32)
    k = _repeat_groups(k.astype(jnp.float32), heads) + k_bias.astype(jnp.float32)
    angle_state, ssm_state = (None, None) if initial_state is None else initial_state
    increments = jnp.tanh(angles.astype(jnp.float32)) * math.pi * dt32[..., None]
    angle = cumulative_angle(increments, segment_ids=segment_ids, initial=angle_state)
    q, k = rotate_pairs(q, angle), rotate_pairs(k, angle)
    lam = jax.nn.sigmoid(trap.astype(jnp.float32))
    y, final = ops.linear_attention(
        "mamba3", q.astype(v.dtype), k.astype(v.dtype), v, g=adt32, beta=(1.0 - lam) * dt32 * jnp.exp(adt32),
        gamma=lam * dt32, scale=1.0, impl=impl, chunk_size=chunk_size, segment_ids=segment_ids,
        initial_state=ssm_state, output_final_state=output_final_state,
    )
    y = y.astype(jnp.float32)
    if d is not None:
        y = y + d.astype(jnp.float32)[:, None] * v.astype(jnp.float32)
    if z is not None:
        y = y * jax.nn.silu(z.astype(jnp.float32))
    state = (angle[:, -1], final) if output_final_state else None
    return y.astype(v.dtype), state


class Mamba2(eqx.Module):
    """Mamba-2 token mixer (Dao and Gu 2024), following state-spaces/mamba ``Mamba2``.

    ``out_proj`` starts at ``N(0, std)`` (truncated) when ``n_residuals`` is None (this constructor's default;
    ``ModelConfig.init_scheme="jla"``). With ``n_residuals`` set (``init_scheme="upstream"``, the ``ModelConfig``
    default) it starts as both
    upstreams initialize it: PyTorch's ``nn.Linear`` default
    ``kaiming_uniform_(a=sqrt(5))`` = ``U(-1/sqrt(inner), 1/sqrt(inner))``, divided by
    ``sqrt(n_residuals)``, the GPT-2 depth rescale of state-spaces/mamba ``MixerModel._init_weights``
    and FLA's ``Mamba2PreTrainedModel`` (``rescale_prenorm_residual``), where ``n_residuals`` is
    ``n_layer`` for a mixer-only stack and ``2 n_layer`` when every block also has an MLP
    (``LanguageModel`` passes the number of residual branches of the whole stack). The ``in_proj`` rows
    keep ``N(0, std)`` (truncated) and the conv PyTorch's ``Conv1d``
    default ``U(-1/sqrt(K), 1/sqrt(K))``: the upstreams differ there (state-spaces keeps PyTorch's
    ``nn.Linear`` and ``Conv1d`` defaults, FLA draws both from ``N(0, initializer_range)`` with a zero
    conv bias).
    """

    zx_proj: StackedLinear  # rows z, x of in_proj
    bc_proj: StackedLinear  # rows B, C
    dt_proj: Linear  # row dt
    x_conv: ShortConv
    bc_conv: ShortConv
    A_log: Array
    dt_bias: Array
    D: Array
    norm: GatedRMSNorm
    out_proj: Linear
    cfg: SSMConfig = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    level_proj: Linear | None = None  # log-linear Mamba-2: per-head Fenwick level logits
    level_weight: Array | None = None

    def __init__(self, d_model: int, cfg: SSMConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16", n_residuals: int | None = None):
        heads = cfg.num_heads(d_model)
        inner = heads * cfg.head_dim
        bc_dim = cfg.n_groups * cfg.d_state
        out_init = _out_proj_init(n_residuals)
        ks = jax.random.split(key, 8)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.zx_proj = StackedLinear(d_model, 2, inner, key=ks[0], out_axis="heads", **common)
        self.bc_proj = StackedLinear(d_model, 2, bc_dim, key=ks[1], out_axis=None, **common)
        self.dt_proj = Linear(d_model, heads, key=ks[2], axes=("embed", None), **common)
        conv = dict(activation=True, use_bias=True, init="uniform", param_dtype=param_dtype, impl=cfg.conv_impl)
        self.x_conv = ShortConv(inner, cfg.conv_size, key=ks[3], axis="heads", **conv)
        self.bc_conv = ShortConv(2 * bc_dim, cfg.conv_size, key=ks[4], axis=None, **conv)
        low, high = cfg.a_init_range
        self.A_log = jnp.log(jax.random.uniform(ks[5], (heads,), jnp.float32, low, high)).astype(param_dtype)
        self.dt_bias = _dt_bias_init(ks[6], heads, cfg).astype(param_dtype)
        self.D = jnp.ones((heads,), param_dtype)
        self.norm = GatedRMSNorm(inner, eps=norm_eps, norm_before_gate=False, group_size=inner // cfg.n_groups,
                                 param_dtype=param_dtype)
        self.out_proj = Linear(inner, d_model, key=ks[7], axes=("heads", "embed"), **out_init, **common)
        self.cfg = cfg
        self.num_heads = heads
        levels = cfg.log_linear_levels
        if levels < 0:
            raise ValueError(f"log_linear_levels must be >= 0; got {levels}")
        self.level_proj = None if not levels else Linear(
            d_model, heads * levels, key=jax.random.fold_in(key, 8), axes=("embed", None), **common)
        self.level_weight = None if not levels else jnp.ones((heads, levels), param_dtype)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        if self.level_weight is not None:
            return {"A_log": (None,), "dt_bias": (None,), "D": (None,), "level_weight": (None, None)}
        return {"A_log": (None,), "dt_bias": (None,), "D": (None,)}

    def _forward(self, u: Array, *, segment_ids: Array | None, cache: RecurrentCache | None):
        if cache is None and _fused_glue_applies(self):
            return _mamba2_fused_forward(self, u, segment_ids), None
        cfg, heads = self.cfg, self.num_heads
        batch, length, _ = u.shape
        z, x = _split_rows(self.zx_proj, u)
        bc = self.bc_proj(u).reshape(batch, length, -1)
        conv_x, conv_bc = (None, None) if cache is None else cache.conv
        x, state_x = self.x_conv(x, segment_ids=segment_ids, state=conv_x)
        bc, state_bc = self.bc_conv(bc, segment_ids=segment_ids, state=conv_bc)
        bc = bc.reshape(batch, length, 2, cfg.n_groups, cfg.d_state)
        dt = jax.nn.softplus(self.dt_proj(u).astype(jnp.float32) + self.dt_bias.astype(jnp.float32))
        a = -jnp.exp(self.A_log.astype(jnp.float32))
        impl = "reference" if cache is not None and length == 1 else cfg.implementation
        ssd = mamba2_ssd
        if self.level_proj is not None:
            if cache is not None:
                raise ValueError("log-linear Mamba-2 has no decoding cache yet")
            logits = self.level_proj(u).astype(jnp.float32).reshape(batch, length, heads, -1)
            scales = jax.nn.softplus(self.level_weight.astype(jnp.float32) * logits)

            def ssd(*args, **options):
                return log_linear_mamba2_ssd(*args, level_scales=scales, **options)

        y, ssm = ssd(
            x.reshape(batch, length, heads, cfg.head_dim), dt, a, bc[:, :, 0], bc[:, :, 1], d=self.D,
            segment_ids=segment_ids, initial_state=None if cache is None else cache.state,
            output_final_state=cache is not None, impl=impl, chunk_size=cfg.chunk_size,
            **({"grouped_key_grads": True} if cfg.grouped_key_grads and self.level_proj is None else {}),
        )
        y = checkpoint_name(y, ATTENTION_CHECKPOINT_NAME)
        out = self.out_proj(self.norm(y.reshape(batch, length, -1), z))
        return out, None if cache is None else RecurrentCache(state=ssm, conv=(state_x, state_bc))

    def __call__(self, u: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(u, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg, heads = self.cfg, self.num_heads
        width = cfg.conv_size - 1
        bc_dim = 2 * cfg.n_groups * cfg.d_state
        return RecurrentCache(
            state=jnp.zeros((batch, heads, cfg.d_state, cfg.head_dim), jnp.float32),
            conv=(jnp.zeros((batch, width, heads * cfg.head_dim), dtype), jnp.zeros((batch, width, bc_dim), dtype)),
        )

    def step(self, u: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``u [B, T, D]`` from ``cache``."""
        return self._forward(u, segment_ids=None, cache=cache)


class Mamba3(eqx.Module):
    """Mamba-3 SISO token mixer (Lahoti et al. 2026), following state-spaces/mamba ``Mamba3``.

    ``out_proj`` starts at ``N(0, std)`` (truncated) when ``n_residuals`` is None (``init_scheme="jla"``),
    else at ``U(-1/sqrt(inner), 1/sqrt(inner)) / sqrt(n_residuals)``, as in
    state-spaces/mamba ``MixerModel._init_weights`` and FLA's ``Mamba3PreTrainedModel`` (see
    :class:`Mamba2`); the other projections keep ``N(0, std)`` (truncated), between the two
    upstreams (state-spaces: PyTorch's ``nn.Linear`` default, FLA: ``N(0, initializer_range)``).
    """

    zx_proj: StackedLinear  # rows z, x of in_proj
    bc_proj: StackedLinear  # rows B, C
    scalar_proj: StackedLinear  # rows dd_dt, dd_A, trap (one value per head each)
    angle_proj: Linear  # rows angles (shared by all heads)
    dt_bias: Array
    B_bias: Array
    C_bias: Array
    B_norm: RMSNorm
    C_norm: RMSNorm
    D: Array
    norm: GatedRMSNorm | None
    out_proj: Linear
    cfg: SSMConfig = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_angles: int = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: SSMConfig, *, key: Array, std: float, norm_eps: float,
                 param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16", n_residuals: int | None = None):
        heads = cfg.num_heads(d_model)
        inner = heads * cfg.head_dim
        rotated = int(cfg.d_state * cfg.rope_fraction)
        rotated -= rotated % 2
        if rotated < 2:
            raise ValueError(f"d_state {cfg.d_state} with rope_fraction {cfg.rope_fraction} rotates no channel pair")
        ks = jax.random.split(key, 6)
        out_init = _out_proj_init(n_residuals)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.zx_proj = StackedLinear(d_model, 2, inner, key=ks[0], out_axis="heads", **common)
        self.bc_proj = StackedLinear(d_model, 2, cfg.n_groups * cfg.d_state, key=ks[1], out_axis=None, **common)
        self.scalar_proj = StackedLinear(d_model, 3, heads, key=ks[2], out_axis=None, **common)
        self.angle_proj = Linear(d_model, rotated // 2, key=ks[3], axes=("embed", None), **common)
        self.dt_bias = _dt_bias_init(ks[4], heads, cfg).astype(param_dtype)
        self.B_bias = jnp.ones((heads, cfg.d_state), param_dtype)
        self.C_bias = jnp.ones((heads, cfg.d_state), param_dtype)
        self.B_norm = RMSNorm(cfg.d_state, eps=norm_eps, param_dtype=param_dtype)
        self.C_norm = RMSNorm(cfg.d_state, eps=norm_eps, param_dtype=param_dtype)
        self.D = jnp.ones((heads,), param_dtype)
        self.norm = (
            GatedRMSNorm(inner, eps=norm_eps, group_size=cfg.head_dim, param_dtype=param_dtype)
            if cfg.outproj_norm else None
        )
        self.out_proj = Linear(inner, d_model, key=ks[5], axes=("heads", "embed"), **out_init, **common)
        self.cfg = cfg
        self.num_heads = heads
        self.num_angles = rotated // 2

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"dt_bias": (None,), "B_bias": (None, None), "C_bias": (None, None), "D": (None,)}

    def _forward(self, u: Array, *, segment_ids: Array | None, cache: RecurrentCache | None):
        cfg, heads = self.cfg, self.num_heads
        batch, length, _ = u.shape
        z, x = _split_rows(self.zx_proj, u)
        z = z.reshape(batch, length, heads, cfg.head_dim)
        x = x.reshape(batch, length, heads, cfg.head_dim)
        bc = self.bc_proj(u).reshape(batch, length, 2, cfg.n_groups, cfg.d_state)
        b, c = self.B_norm(bc[:, :, 0]), self.C_norm(bc[:, :, 1])
        scalars = self.scalar_proj(u).astype(jnp.float32)
        dt = jax.nn.softplus(scalars[..., 0, :] + self.dt_bias.astype(jnp.float32))
        a = jnp.minimum(-heavy_tail(scalars[..., 1, :]), -cfg.a_floor)
        angles = jnp.broadcast_to(self.angle_proj(u).astype(jnp.float32)[:, :, None, :],
                                  (batch, length, heads, self.num_angles))
        impl = "reference" if cache is not None and length == 1 else cfg.implementation
        y, state = mamba3_siso(
            c, b, x, a * dt, dt, scalars[..., 2, :], self.C_bias, self.B_bias, angles, d=self.D,
            z=None if self.norm is not None else z, segment_ids=segment_ids,
            initial_state=None if cache is None else (cache.angle, cache.state),
            output_final_state=cache is not None, impl=impl, chunk_size=cfg.chunk_size,
        )
        y = checkpoint_name(y.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        if self.norm is not None:
            y = self.norm(y, z.reshape(batch, length, -1))
        out = self.out_proj(y)
        return out, None if cache is None else RecurrentCache(state=state[1], angle=state[0])

    def __call__(self, u: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(u, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg, heads = self.cfg, self.num_heads
        f32 = jnp.float32
        state = Mamba3State(
            jnp.zeros((batch, heads, cfg.d_state, cfg.head_dim), f32),
            jnp.zeros((batch, heads, cfg.d_state), f32),
            jnp.zeros((batch, heads, cfg.head_dim), f32),
        )
        return RecurrentCache(state=state, angle=jnp.zeros((batch, heads, self.num_angles), f32))

    def step(self, u: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``u [B, T, D]`` from ``cache``."""
        return self._forward(u, segment_ids=None, cache=cache)


def log_linear_mamba2_ssd(
    x: Array,
    dt: Array,
    a: Array,
    b: Array,
    c: Array,
    *,
    level_scales: Array,
    d: Array | None = None,
    segment_ids: Array | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    impl: str = "auto",
    chunk_size: int = 64,
) -> tuple[Array, Array | None]:
    """Log-linear Mamba-2 (Guo et al. 2025): :func:`mamba2_ssd` with Fenwick level weights.

    ``level_scales [B, T, H, L]`` weighs each Fenwick level of the SSD state; runs
    ``ops.linear_attention("log_linear_simple_gla", q=C, k=B, v=dt*x, g=dt*A, scale=1.0)``.
    With every scale equal to 1 it is :func:`mamba2_ssd`.
    """
    heads = x.shape[2]
    dt32 = dt.astype(jnp.float32)
    q, k = _repeat_groups(c, heads), _repeat_groups(b, heads)
    v = (x.astype(jnp.float32) * dt32[..., None]).astype(x.dtype)
    y, state = ops.linear_attention(
        "log_linear_simple_gla", q, k.astype(q.dtype), v.astype(q.dtype), g=dt32 * a.astype(jnp.float32),
        level_scales=level_scales, scale=1.0, impl=impl, chunk_size=chunk_size, segment_ids=segment_ids,
        initial_state=initial_state, output_final_state=output_final_state,
    )
    if d is not None:
        y = (y.astype(jnp.float32) + d.astype(jnp.float32)[:, None] * x.astype(jnp.float32)).astype(y.dtype)
    return y, state


def _fused_out_proj_kind(out_proj) -> str | None:
    """``"fold"`` for a plain bias-free compute-dtype ``out_proj`` (:func:`jla.layers.mamba2_fused.gated_norm_out`
    folds the norm weight into it), ``"module"`` for a bias-free FP8 ``out_proj`` (it runs as its own module after
    :func:`jla.layers.mamba2_fused.gated_norm`), else None."""
    from jla.layers.fp8 import Fp8Linear

    if getattr(out_proj, "bias", None) is not None:
        return None
    if type(out_proj) is Linear:
        return "fold"
    if type(out_proj) is Fp8Linear:
        return "module"
    return None


def _fused_glue_applies(layer: Mamba2) -> bool:
    """``SSMConfig.fused_glue`` for this layer's training calls on GPUs: one B/C group, the gated RMSNorm
    (``norm_before_gate=False``, SiLU, one group), a bias-free ``out_proj`` (compute dtype or FP8) and
    a power-of-two head dim (the Pallas Triton row kernels of :mod:`jla.layers.mamba2_fused`; any head count).
    Multi-device calls run the row kernels per shard of the context mesh; without a mesh that can split
    them they keep the unfused layer (:func:`jla.layers.mamba2_fused.device_layout_ok`), and so does a mesh
    that splits the heads (tensor parallelism; :func:`jla.layers.mamba2_fused.heads_split`)."""
    cfg, norm = layer.cfg, layer.norm
    if not getattr(cfg, "fused_glue", False):
        return False
    from jla.layers import mamba2_fused as fused

    return (layer.level_proj is None and cfg.n_groups == 1 and _fused_out_proj_kind(layer.out_proj) is not None
            and type(norm) is GatedRMSNorm and norm.activation == "silu" and not norm.norm_before_gate
            and norm.group_size in (None, layer.num_heads * cfg.head_dim)
            and fused.kernel_shape_ok(layer.num_heads, cfg.head_dim)
            and jax.default_backend() == "gpu" and fused.device_layout_ok() and not fused.heads_split())


IN_PLACE_ZX = True
"""Mamba-2 fused glue with a stacked z|x projection (an FP8 ``zx_proj``: one GEMM writing ``[B, T, 2, inner]``) and an
FP8 out-projection: the x conv and the glue kernels read their rows of the GEMM output in place
(:func:`jla.ops.mosaic_gpu.short_conv.short_conv_rows`, ``gated_norm(z_row=0)``) instead of copying both rows out
of it first (``benchmarks/end_to_end_1p3b.md``). False slices the rows out."""


def _zx_in_place(layer: Mamba2, u: Array, segment_ids: Array | None) -> bool:
    """Whether :func:`_mamba2_fused_forward` reads the z|x rows in place (:data:`IN_PLACE_ZX`): a stacked projection
    (not split by :data:`SPLIT_ZX`), an out-projection module after the glue (FP8), and an x conv the Mosaic GPU
    kernels take on one device (several devices slice the rows and run the conv per shard)."""
    if not IN_PLACE_ZX or (SPLIT_ZX and isinstance(layer.zx_proj, StackedLinear)):
        return False
    if _fused_out_proj_kind(layer.out_proj) != "module" or layer.x_conv.impl == "xla":
        return False
    try:
        import jax.experimental.pallas.mosaic_gpu  # noqa: F401
    except ImportError:
        return False
    from jla.ops.mosaic_gpu import short_conv

    inner = layer.num_heads * layer.cfg.head_dim
    view = jax.ShapeDtypeStruct((*u.shape[:2], inner), jnp.dtype(layer.zx_proj.compute_dtype))
    return short_conv.unsupported_reason(view, layer.x_conv.weight, segment_ids=segment_ids) is None


def _mamba2_fused_forward(layer: Mamba2, u: Array, segment_ids: Array | None) -> Array:
    """:meth:`Mamba2._forward` of a training call (no cache) when :func:`_fused_glue_applies`: the D skip, gate,
    RMSNorm and out-projection run as :func:`jla.layers.mamba2_fused.gated_norm_out` (an FP8 out-projection: as
    :func:`jla.layers.mamba2_fused.gated_norm` and the FP8 module), same parameters and values up to rounding
    order; the projections, convolutions, ``dt`` and the SSD kernels are :meth:`Mamba2._forward`'s. A call that
    is not differentiated forms the SSD values ``dt x`` inside the SSD kernel (``mamba2_fused.ssd_core(v_parts=...)``,
    the same values bit for bit).
    """
    from jla.layers import mamba2_fused as fused

    cfg, heads = layer.cfg, layer.num_heads
    batch, length, _ = u.shape
    zx = None  # the stacked z|x GEMM output when the conv and the glue kernels read its rows in place
    if _zx_in_place(layer, u, segment_ids):
        z = zx = layer.zx_proj(u)  # [B, T, 2, inner]: the gate is row 0, x row 1
    else:
        z, x = _split_rows(layer.zx_proj, u, separate=True)
    bc = layer.bc_proj(u).reshape(batch, length, -1)
    if zx is not None:
        from jla.ops.mosaic_gpu.short_conv import short_conv_rows

        conv = layer.x_conv
        x = short_conv_rows(zx, conv.weight, conv.bias, row=1, activation=conv.activation, segment_ids=segment_ids)
        if x is None:  # the kernels declined the stacked rows (e.g. their dtype): slice them out, as _split_rows does
            z, x = zx[..., 0, :], zx[..., 1, :]
            zx = None
    if zx is None:
        x, _ = layer.x_conv(x, segment_ids=segment_ids)
    bc, _ = layer.bc_conv(bc, segment_ids=segment_ids)
    bc = bc.reshape(batch, length, 2, cfg.n_groups, cfg.d_state)
    x = x.reshape(batch, length, heads, cfg.head_dim)
    dt = jax.nn.softplus(layer.dt_proj(u).astype(jnp.float32) + layer.dt_bias.astype(jnp.float32))
    v = (x.astype(jnp.float32) * dt[..., None]).astype(x.dtype).astype(bc.dtype)  # mamba2_ssd's SSD inputs
    g = dt * -jnp.exp(layer.A_log.astype(jnp.float32))
    y = fused.ssd_core(v, g, bc[:, :, 0], bc[:, :, 1], segment_ids=segment_ids, impl=cfg.implementation,
                       chunk_size=cfg.chunk_size, grouped_key_grads=cfg.grouped_key_grads, v_parts=(x, dt))
    y = checkpoint_name(y, ATTENTION_CHECKPOINT_NAME)
    norm, out_proj = layer.norm, layer.out_proj
    if _fused_out_proj_kind(out_proj) == "module":
        return out_proj(fused.gated_norm(y, x, z, layer.D, norm.weight, eps=norm.eps,
                                         compute_dtype=out_proj.compute_dtype, z_row=None if zx is None else 0))
    return fused.gated_norm_out(y, x, z, layer.D, norm.weight, out_proj.weight, eps=norm.eps,
                                compute_dtype=out_proj.compute_dtype)


def _out_proj_gain(n_residuals: int) -> float:
    """``1 / sqrt(n_residuals)``: the GPT-2 / Megatron depth rescale of Mamba's ``out_proj`` (state-spaces/mamba
    ``mixer_seq_simple._init_weights``, FLA ``rescale_prenorm_residual``) for a stack of ``n_residuals``
    residual branches."""
    if not isinstance(n_residuals, int) or n_residuals < 1:
        raise ValueError(f"n_residuals must be a positive int; got {n_residuals!r}")
    return n_residuals ** -0.5


def _out_proj_init(n_residuals: int | None) -> dict:
    """``Linear`` init options of Mamba's ``out_proj``: ``None`` keeps ``N(0, std)`` (truncated; the ``"jla"``
    init scheme), a count of residual branches gives ``U(-1/sqrt(inner), 1/sqrt(inner)) / sqrt(n_residuals)``
    (``"upstream"``)."""
    if n_residuals is None:
        return {}
    return dict(init="kaiming_uniform", gain=_out_proj_gain(n_residuals))
