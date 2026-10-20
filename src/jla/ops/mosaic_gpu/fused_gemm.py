# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2025 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py (JAX 0.11.2).
"""SwiGLU MLP with the activation fused into its GEMMs, on Mosaic GPU (H100, sm_90a).

``y = silu(x W_gate) * (x W_up) @ W_down`` (the :class:`jla.layers.mlp.SwiGLU` block)
as two persistent warp-specialised GEMM kernels (one TMA producer warpgroup, two
WGMMA consumer warpgroups, 128 x 256 CTA tiles, the pipeline of JAX's Hopper
matmul example) plus three cuBLAS GEMMs:

* forward (``swiglu_gate_up``): ``x [M, D] @ W_il [D, 2F]`` where ``W_il`` is the
  gate/up weight with gate and up columns interleaved in blocks of 64 features, so
  every 128-column block of a consumer warpgroup's accumulator holds the gate and
  the up pre-activations of the same 64 features. The epilogue forms
  ``h = silu(gate) * up`` in float32 registers and stores ``h [M, F]`` (and, for
  the backward pass, the pre-activations ``gu [M, 2F]``); the separate SwiGLU pass
  over ``[M, 2F]`` disappears. ``y = h @ W_down`` is a cuBLAS GEMM.
* backward (``swiglu_down_dgrad``): ``dh = dy [M, D] @ W_down^T`` with an
  epilogue that loads the matching ``gate, up`` tiles (TMA, the first one issued
  before the K loop), forms ``dgate = dh * up * silu'(gate)`` and
  ``dup = dh * silu(gate)`` and stores ``dgu [M, 2F]`` in the interleaved layout:
  neither ``dh`` nor a separate SwiGLU-backward pass touches HBM. ``dW_down``,
  ``dW_il`` and ``dx`` are cuBLAS GEMMs.
* ``swiglu_mlp(..., recompute=True)`` saves only the block's inputs and reruns the
  forward kernel in the backward pass, for blocks under full rematerialization.

Only weights are permuted (``[D, 2, F]`` <-> the interleaved ``[D, 2F]``, fused by XLA
into the compute-dtype cast); activations stay in the layouts the kernels use.
Numerics: bfloat16 or float16 operands, float32 accumulation, the activation and
its derivative in float32 from the float32 accumulator (the XLA path rounds the
pre-activations, and ``dh``, to the compute dtype first). Kernel shapes:
``M % 128 == 0``, ``D % 64 == 0``, ``F % 256 == 0``; :func:`unsupported_reason`
says why a call does not qualify. Like the cross-entropy kernels these are
single-device custom calls with no partitioning rule, so calls that may span
several devices are rejected (see :func:`jla.ops.mosaic_gpu.linear_cross_entropy.multi_device_reason`);
:func:`swiglu_mlp_sharded` runs them per shard of the context mesh (tokens over the data axes).
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from jla.ops.mosaic_gpu._common import is_hopper, num_sms
from jla.ops.mosaic_gpu.linear_cross_entropy import multi_device_reason

__all__ = [
    "DTYPES", "GemmConfig", "deinterleave", "interleave", "swiglu_down_dgrad", "swiglu_gate_up", "swiglu_mlp",
    "swiglu_mlp_sharded",
    "unsupported_reason",
]

Array = jax.Array

DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
TILE_M = 128  # rows of a CTA tile
TILE_K = 64  # contraction block per pipeline step
WG_N = 128  # accumulator columns per consumer warpgroup when the warpgroups split the columns
CTA_N = 2 * WG_N  # columns of a CTA tile
PAIR = 64  # features per interleaved block: 64 gate columns, then the 64 up columns of the same features
EPI = 64  # epilogue sub-tile (rows and columns)


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    steps: int = 4  # TMA pipeline depth
    grid_minor: int = 0  # planar_snake minor dimension (0 = rows, 1 = columns)
    grid_width: int = 8  # planar_snake tile width
    # How the two consumer warpgroups split the 128 x 256 CTA tile: 1 = by columns (128 x 128 each),
    # 0 = by rows (64 x 256 each: one m64n256 WGMMA per k-step, which reads less shared memory per FLOP).
    wg_dim: int = 1
    # Every k-step waits for its WGMMA before releasing the stage (no ``delay_release``): with one
    # pipeline per output tile, a delayed release leaves the last stage of a tile unreleased while the
    # TMA warpgroup refills every stage for the next tile (measured: wrong outputs).

    def __post_init__(self):
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1; got {self}")
        if self.grid_minor not in (0, 1) or self.wg_dim not in (0, 1) or self.grid_width < 1:
            raise ValueError(f"grid_minor and wg_dim must be 0 or 1 and grid_width >= 1; got {self}")

    def warpgroup_tile(self) -> tuple[int, int]:
        """``(rows, columns)`` of one consumer warpgroup's accumulator."""
        return (TILE_M, WG_N) if self.wg_dim == 1 else (TILE_M // 2, CTA_N)


FWD_CONFIG = GemmConfig(steps=4, grid_minor=0, grid_width=8)
BWD_CONFIG = GemmConfig(steps=3, grid_minor=0, grid_width=8)  # the epilogue's input tiles leave room for 3


def interleave(gate_up: Array) -> Array:
    """``[..., D, 2, F] -> [..., D, 2F]`` with columns ``(block j, gate|up, i) = feature 64 j + i``."""
    *lead, d, two, f = gate_up.shape
    if two != 2 or f % PAIR:
        raise ValueError(f"expected gate_up [..., D, 2, F] with F % {PAIR} == 0; got {gate_up.shape}")
    w = gate_up.reshape(*lead, d, 2, f // PAIR, PAIR)
    return jnp.swapaxes(w, -3, -2).reshape(*lead, d, 2 * f)


def deinterleave(x: Array) -> Array:
    """Inverse of :func:`interleave` on the last axis: ``[..., 2F] -> [..., 2, F]``."""
    *lead, two_f = x.shape
    f = two_f // 2
    w = x.reshape(*lead, f // PAIR, 2, PAIR)
    return jnp.swapaxes(w, -3, -2).reshape(*lead, 2, f)


def unsupported_reason(m: int, d: int, f: int, dtype, *, check_devices: bool = True) -> str | None:
    """``None`` when the kernels accept ``x [m, d]``, ``F = f`` and ``dtype``, otherwise why not
    (``check_devices=False``: on one device, as :func:`swiglu_mlp_sharded` checks its shards)."""
    if jnp.dtype(dtype) not in DTYPES:
        return f"compute dtype must be bfloat16 or float16; got {jnp.dtype(dtype)}"
    if m % TILE_M:
        return f"the number of tokens {m} must be a multiple of {TILE_M}"
    if d % TILE_K:
        return f"d_model {d} must be a multiple of {TILE_K}"
    if f % CTA_N:
        return f"d_ff {f} must be a multiple of {CTA_N}"
    reason = multi_device_reason() if check_devices else None
    if reason is not None:
        return reason
    if not is_hopper():
        return "the fused SwiGLU kernels need an sm_90 (Hopper) GPU"
    return None


def _transforms(minor: int, dtype) -> tuple:
    swizzle = plgpu.find_swizzle(minor * jnp.dtype(dtype).itemsize * 8)
    return plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle)


def _kernel(body, out_type):
    return plgpu.kernel(
        body,
        out_type=out_type,
        grid=(num_sms(),),
        grid_names=("sm",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )


def _depth(cfg: GemmConfig, k_iters: int) -> int:
    """Pipeline depth: the configured number of stages, at most the number of k-steps."""
    return min(cfg.steps, k_iters)


def _pipeline(body, compute_context, *, k_iters, steps, b_block, b_index, dtype):
    in_transforms = _transforms(TILE_K, dtype)
    return plgpu.emit_pipeline_warp_specialized(
        body,
        grid=(k_iters,),
        memory_registers=40,
        in_specs=[
            plgpu.BlockSpec((TILE_M, TILE_K), lambda k: (0, k), transforms=in_transforms),
            plgpu.BlockSpec(b_block, b_index, transforms=in_transforms),
        ],
        wg_axis="wg",
        num_compute_wgs=2,
        max_concurrent_steps=steps,
        compute_context=compute_context,
    )


# Inner jits: each distinct kernel is traced and compiled once per shape.
@functools.partial(jax.jit, static_argnames=("save_gu", "cfg"))
def swiglu_gate_up(x: Array, w_il: Array, *, save_gu: bool, cfg: GemmConfig = FWD_CONFIG):
    """``h = silu(x W_gate) * x W_up`` from the interleaved weight ``w_il [D, 2F]``.

    Returns ``h [M, F]``, and with ``save_gu`` also the pre-activations ``gu [M, 2F]``
    (interleaved like ``w_il``), all in ``x.dtype``.
    """
    m, d = x.shape
    two_f = w_il.shape[1]
    f = two_f // 2
    dtype = x.dtype
    m_iters, n_iters, k_iters = m // TILE_M, two_f // CTA_N, d // TILE_K
    steps = _depth(cfg, k_iters)
    out_transforms = _transforms(EPI, dtype)
    wg_rows, wg_cols = cfg.warpgroup_tile()

    def get_pipeline(body, compute_context):
        return _pipeline(body, compute_context, k_iters=k_iters, steps=steps, b_block=(TILE_K, CTA_N),
                         b_index=lambda k: (k, 0), dtype=dtype)

    def body(x_ref, w_ref, h_ref, *rest):
        gu_ref = rest[0] if save_gu else None
        ignore = lambda *_, **__: None  # noqa: E731  (allocations do not depend on the bodies)

        @functools.partial(
            pl.run_scoped,
            pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(x_ref, w_ref),
            out_smem=plgpu.SMEM((2, 2, EPI, EPI), dtype, transforms=out_transforms),
            collective_axes="wg",
        )
        def _scope(pipeline_allocs, out_smem):
            wg = lax.axis_index("wg")

            @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
            def _tiles(info: plgpu.NDLoopInfo):
                (lin,) = info.index
                mi, ni = plgpu.planar_snake(lin, (m_iters, n_iters), cfg.grid_minor, cfg.grid_width)
                m0 = mi * TILE_M
                # First row and first interleaved block (features 64 blk .. 64 blk + 63) of this warpgroup.
                row0 = m0 if cfg.wg_dim == 1 else m0 + wg * wg_rows
                blk0 = ni * (CTA_N // (2 * PAIR)) + (wg if cfg.wg_dim == 1 else 0)

                def compute_context(eval_pipeline):
                    @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((wg_rows, wg_cols), jnp.float32))
                    def _acc(acc_ref):
                        eval_pipeline(acc_ref)
                        acc = acc_ref[...]
                        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                        slot = 0
                        for rh in range(wg_rows // EPI):
                            rows = pl.ds(row0 + rh * EPI, EPI)
                            r = slice(rh * EPI, (rh + 1) * EPI)
                            for jj in range(wg_cols // (2 * PAIR)):
                                blk = blk0 + jj
                                gate = acc[r, jj * 2 * PAIR:jj * 2 * PAIR + PAIR]
                                up = acc[r, jj * 2 * PAIR + PAIR:(jj + 1) * 2 * PAIR]
                                stores = []
                                if save_gu:
                                    stores.append((gate, gu_ref.at[rows, pl.ds(blk * 2 * PAIR, PAIR)]))
                                    stores.append((up, gu_ref.at[rows, pl.ds(blk * 2 * PAIR + PAIR, PAIR)]))
                                stores.append((gate * jax.nn.sigmoid(gate) * up,
                                               h_ref.at[rows, pl.ds(blk * PAIR, PAIR)]))
                                for value, dst in stores:
                                    plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                                    out_smem.at[wg, slot][...] = value.astype(dtype)
                                    plgpu.commit_smem()
                                    plgpu.copy_smem_to_gmem(out_smem.at[wg, slot], dst)
                                    slot = 1 - slot

                def mma(_, x_smem, w_smem, acc_ref):
                    if cfg.wg_dim == 1:
                        plgpu.wgmma(acc_ref, x_smem, w_smem.at[:, pl.ds(wg * WG_N, WG_N)])
                    else:
                        plgpu.wgmma(acc_ref, x_smem.at[pl.ds(wg * wg_rows, wg_rows)], w_smem)
                    plgpu.wgmma_wait(0)
                    return acc_ref

                get_pipeline(mma, compute_context)(
                    x_ref.at[pl.ds(m0, TILE_M), :], w_ref.at[:, pl.ds(ni * CTA_N, CTA_N)],
                    allocations=pipeline_allocs)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    h_type = jax.ShapeDtypeStruct((m, f), dtype)
    out_type = (h_type, jax.ShapeDtypeStruct((m, two_f), dtype)) if save_gu else (h_type,)
    out = _kernel(body, out_type)(x, w_il)
    if save_gu:
        return tuple(out)
    return out[0] if isinstance(out, (tuple, list)) else out


@functools.partial(jax.jit, static_argnames=("cfg",))
def swiglu_down_dgrad(dy: Array, w_down: Array, gu: Array, *, cfg: GemmConfig = BWD_CONFIG) -> Array:
    """``dgu [M, 2F]`` (interleaved) of ``h = silu(gate) * up`` from ``dh = dy @ w_down^T``.

    ``dy [M, D]``, ``w_down [F, D]`` (the down projection's ``[in, out]`` weight) and the
    saved pre-activations ``gu [M, 2F]`` from :func:`swiglu_gate_up`.
    """
    m, d = dy.shape
    f = w_down.shape[0]
    dtype = dy.dtype
    m_iters, n_iters, k_iters = m // TILE_M, f // CTA_N, d // TILE_K
    steps = _depth(cfg, k_iters)
    tile_transforms = _transforms(EPI, dtype)
    wg_rows, wg_cols = cfg.warpgroup_tile()  # columns are features here
    subtiles = [(rh, jj) for rh in range(wg_rows // EPI) for jj in range(wg_cols // PAIR)]

    def get_pipeline(body, compute_context):
        return _pipeline(body, compute_context, k_iters=k_iters, steps=steps, b_block=(CTA_N, TILE_K),
                         b_index=lambda k: (0, k), dtype=dtype)

    def body(dy_ref, w_ref, gu_ref, dgu_ref):
        ignore = lambda *_, **__: None  # noqa: E731

        @functools.partial(
            pl.run_scoped,
            pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(dy_ref, w_ref),
            in_smem=plgpu.SMEM((2, 2, EPI, EPI), dtype, transforms=tile_transforms),
            out_smem=plgpu.SMEM((2, 2, EPI, EPI), dtype, transforms=tile_transforms),
            in_bars=plgpu.Barrier(num_arrivals=1, num_barriers=4),
            collective_axes="wg",
        )
        def _scope(pipeline_allocs, in_smem, out_smem, in_bars):
            wg = lax.axis_index("wg")

            @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
            def _tiles(info: plgpu.NDLoopInfo):
                (lin,) = info.index
                mi, ni = plgpu.planar_snake(lin, (m_iters, n_iters), cfg.grid_minor, cfg.grid_width)
                m0 = mi * TILE_M
                # First row and first interleaved block (64 features each) of this warpgroup.
                row0 = m0 if cfg.wg_dim == 1 else m0 + wg * wg_rows
                blk0 = (ni * CTA_N + (wg * WG_N if cfg.wg_dim == 1 else 0)) // PAIR

                def issue(sub):  # TMA the gate and up tiles of epilogue sub-tile ``sub``
                    rh, jj = subtiles[sub]
                    rows = pl.ds(row0 + rh * EPI, EPI)
                    col = (blk0 + jj) * 2 * PAIR
                    for part in range(2):
                        plgpu.copy_gmem_to_smem(gu_ref.at[rows, pl.ds(col + part * PAIR, PAIR)],
                                                in_smem.at[wg, part], in_bars.at[wg * 2 + part])

                def compute_context(eval_pipeline):
                    plgpu.commit_smem()  # the previous tile's generic reads of in_smem before the TMA writes
                    issue(0)  # lands while the K loop runs

                    @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC((wg_rows, wg_cols), jnp.float32))
                    def _acc(acc_ref):
                        eval_pipeline(acc_ref)
                        dh = acc_ref[...]
                        plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                        slot = 0
                        for sub, (rh, jj) in enumerate(subtiles):
                            for part in range(2):
                                plgpu.barrier_wait(in_bars.at[wg * 2 + part])
                            gate = plgpu.load(in_smem.at[wg, 0], layout=plgpu.Layout.WGMMA).astype(jnp.float32)
                            up = plgpu.load(in_smem.at[wg, 1], layout=plgpu.Layout.WGMMA).astype(jnp.float32)
                            if sub + 1 < len(subtiles):
                                plgpu.commit_smem()
                                issue(sub + 1)
                            d = dh[rh * EPI:(rh + 1) * EPI, jj * PAIR:(jj + 1) * PAIR]
                            sig = jax.nn.sigmoid(gate)
                            d_up = d * gate * sig
                            d_gate = d * up * sig * (1.0 + gate * (1.0 - sig))
                            rows = pl.ds(row0 + rh * EPI, EPI)
                            col = (blk0 + jj) * 2 * PAIR
                            for part, value in enumerate((d_gate, d_up)):
                                plgpu.wait_smem_to_gmem(1, wait_read_only=True)
                                out_smem.at[wg, slot][...] = value.astype(dtype)
                                plgpu.commit_smem()
                                plgpu.copy_smem_to_gmem(out_smem.at[wg, slot],
                                                        dgu_ref.at[rows, pl.ds(col + part * PAIR, PAIR)])
                                slot = 1 - slot

                def mma(_, dy_smem, w_smem, acc_ref):
                    if cfg.wg_dim == 1:
                        plgpu.wgmma(acc_ref, dy_smem, w_smem.at[pl.ds(wg * WG_N, WG_N)].transpose((1, 0)))
                    else:
                        plgpu.wgmma(acc_ref, dy_smem.at[pl.ds(wg * wg_rows, wg_rows)], w_smem.transpose((1, 0)))
                    plgpu.wgmma_wait(0)
                    return acc_ref

                get_pipeline(mma, compute_context)(
                    dy_ref.at[pl.ds(m0, TILE_M), :], w_ref.at[pl.ds(ni * CTA_N, CTA_N), :],
                    allocations=pipeline_allocs)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    return _kernel(body, jax.ShapeDtypeStruct((m, 2 * f), dtype))(dy, w_down, gu)


def _dot(a: Array, b: Array, contract: tuple[int, int]) -> Array:
    return lax.dot_general(a, b, (((contract[0],), (contract[1],)), ((), ())))


def _mlp_primal(x, w_il, w_down, fwd_cfg):
    return _dot(swiglu_gate_up(x, w_il, save_gu=False, cfg=fwd_cfg), w_down, (1, 0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _mlp(x, w_il, w_down, fwd_cfg, bwd_cfg):
    return _mlp_primal(x, w_il, w_down, fwd_cfg)


def _mlp_fwd(x, w_il, w_down, fwd_cfg, bwd_cfg):
    h, gu = swiglu_gate_up(x, w_il, save_gu=True, cfg=fwd_cfg)
    return _dot(h, w_down, (1, 0)), (x, w_il, w_down, h, gu)


def _mlp_bwd(fwd_cfg, bwd_cfg, residuals, dy):
    x, w_il, w_down, h, gu = residuals
    dy = dy.astype(x.dtype)
    dgu = swiglu_down_dgrad(dy, w_down, gu, cfg=bwd_cfg)
    # The same cuBLAS GEMMs (compute-dtype outputs) as XLA's transpose of the unfused block.
    return _dot(dgu, w_il, (1, 1)), _dot(x, dgu, (0, 0)), _dot(h, dy, (0, 0))


_mlp.defvjp(_mlp_fwd, _mlp_bwd)


# The same block saving only its inputs: the backward runs the forward kernel again (storing the
# pre-activations) before the dgrad kernel. Inside a fully rematerialized block the backward pass
# recomputes the forward anyway, so this costs no extra GEMM there and lets the primal pass use the
# cheaper ``h``-only forward kernel (JAX runs the forward rule in the primal pass and cannot drop one
# output of a custom call). Without rematerialization it costs one extra gate/up GEMM.
@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _mlp_recompute(x, w_il, w_down, fwd_cfg, bwd_cfg):
    return _mlp_primal(x, w_il, w_down, fwd_cfg)


def _mlp_recompute_fwd(x, w_il, w_down, fwd_cfg, bwd_cfg):
    return _mlp_primal(x, w_il, w_down, fwd_cfg), (x, w_il, w_down)


def _mlp_recompute_bwd(fwd_cfg, bwd_cfg, residuals, dy):
    x, w_il, w_down = residuals
    h, gu = swiglu_gate_up(x, w_il, save_gu=True, cfg=fwd_cfg)
    return _mlp_bwd(fwd_cfg, bwd_cfg, (x, w_il, w_down, h, gu), dy)


_mlp_recompute.defvjp(_mlp_recompute_fwd, _mlp_recompute_bwd)


def swiglu_mlp(x: Array, gate_up: Array, w_down: Array, *, compute_dtype=jnp.bfloat16, recompute: bool = False,
               fwd_cfg: GemmConfig = FWD_CONFIG, bwd_cfg: GemmConfig = BWD_CONFIG) -> Array:
    """``down(silu(x W_gate) * x W_up)`` for ``x [..., D]``, ``gate_up [D, 2, F]``, ``w_down [F, D]``.

    Same function and parameter layout as :class:`jla.layers.mlp.SwiGLU`; returns
    ``[..., D]`` in ``compute_dtype``. ``recompute=True`` saves only the inputs for the
    backward pass and runs the forward kernel again there: the right choice inside a fully
    rematerialized block (``remat = "full"``), one extra GEMM otherwise. Raises ``ValueError``
    when :func:`unsupported_reason` rejects the call.
    """
    dtype = jnp.dtype(compute_dtype)
    d, f = gate_up.shape[0], gate_up.shape[-1]
    lead = x.shape[:-1]
    m = 1
    for size in lead:
        m *= size
    if x.shape[-1] != d or gate_up.shape != (d, 2, f) or w_down.shape != (f, d):
        raise ValueError(f"expected x [..., D], gate_up [D, 2, F], w_down [F, D]; "
                         f"got {x.shape}, {gate_up.shape}, {w_down.shape}")
    reason = unsupported_reason(m, d, f, dtype)
    if reason is not None:
        raise ValueError(f"Mosaic GPU fused SwiGLU: {reason}")
    block = _mlp_recompute if recompute else _mlp
    y = block(x.reshape(m, d).astype(dtype), interleave(gate_up.astype(dtype)), w_down.astype(dtype),
              fwd_cfg, bwd_cfg)
    return y.reshape(*lead, d)


def swiglu_mlp_sharded(x: Array, gate_up: Array, w_down: Array, *, compute_dtype=jnp.bfloat16,
                       recompute: bool = False, fwd_cfg: GemmConfig = FWD_CONFIG,
                       bwd_cfg: GemmConfig = BWD_CONFIG) -> Array | None:
    """:func:`swiglu_mlp` per shard of the context mesh, for a call that may span several devices.

    The partitioning rule (:func:`jla.distributed.kernel_sharding.shard_layout_call`): the tokens of
    ``x [B, ..., D]`` split along ``B`` over the ``batch`` rule's mesh axes; the weights enter replicated, cast to
    ``compute_dtype`` and interleaved first (so FSDP gathers them in that dtype, as on the XLA path), and
    ``shard_map``'s transpose sums their gradients over the mesh. Each device runs the kernels on its tokens as
    one device does. Returns ``None`` when the rule does not apply: no context mesh axis to split over, a mesh
    axis of size > 1 outside the ``batch`` rule (``tp`` shards the MLP's features, a layout the XLA path keeps),
    or per-device shapes the kernels do not take. Raises ``ValueError`` for mismatched operands.
    """
    from jla.distributed.kernel_sharding import mesh_can_split, rule_axes, shard_layout_call

    dtype = jnp.dtype(compute_dtype)
    d, f = gate_up.shape[0], gate_up.shape[-1]
    if x.ndim < 2 or x.shape[-1] != d or gate_up.shape != (d, 2, f) or w_down.shape != (f, d):
        raise ValueError(f"expected x [B, ..., D], gate_up [D, 2, F], w_down [F, D]; "
                         f"got {x.shape}, {gate_up.shape}, {w_down.shape}")
    if not mesh_can_split():
        return None
    mesh = jax.sharding.get_abstract_mesh()
    batch_axes = set(rule_axes("batch"))
    if any(size > 1 and name not in mesh.manual_axes and name not in batch_axes for name, size in mesh.shape.items()):
        return None
    block = _mlp_recompute if recompute else _mlp

    def body(x, w_il, w_down):
        lead = x.shape[:-1]
        y = block(x.reshape(-1, d).astype(dtype), w_il, w_down, fwd_cfg, bwd_cfg)
        return y.reshape(*lead, d)

    def supports(x, w_il, w_down):
        m = 1
        for size in x.shape[:-1]:
            m *= size
        return unsupported_reason(m, d, f, dtype, check_devices=False) is None

    return shard_layout_call(
        body, {"x": x, "w_il": interleave(gate_up.astype(dtype)), "w_down": w_down.astype(dtype)},
        {"x": "batch", "w_il": "replicated", "w_down": "replicated"},
        out_layouts="batch", supports=supports, unsplit=lambda **_: None)
