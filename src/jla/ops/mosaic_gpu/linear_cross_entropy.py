# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2025 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py (JAX 0.11.2).
"""Fused LM-head cross entropy on Mosaic GPU (H100, sm_90a).

The same sums as :func:`jla.ops.cross_entropy.linear_cross_entropy` (label
smoothing, z-loss and ``ignore_index`` included), with the ``[N, V]`` logits
kept on chip:

* forward: one persistent warp-specialised GEMM kernel over ``hidden [N, D]`` x
  ``weight [V, D]^T`` (TMA producer warpgroup, two WGMMA consumer warpgroups,
  128 x 256 output tiles). Its epilogue reduces every 128 x 128 logit tile to a
  per-row max and exp-sum in registers and writes only those two numbers per
  row and tile; XLA combines them into the logsumexp. The target logit
  (``hidden . weight[target]``) and, with label smoothing, the mean logit
  (``hidden . sum_v weight[v] / V``) are row dot products done in XLA.
* backward (custom VJP storing only the per-token logsumexp): per vocabulary
  chunk the same GEMM recomputes the logits and its epilogue forms the logit
  gradient ``(g_ce + 2 g_z lse) softmax - g_ce (1 - ls) onehot - g_ce ls / V``
  in registers and stores it once in the compute dtype; ``dhidden`` and
  ``dweight`` are two cuBLAS GEMMs on that chunk. No f32 logits reach HBM, and
  the chunks run one after another, so one ``[N, chunk]`` gradient is live.
* forward-pass gradient (:func:`objective_sums`, for ``loss_sum + z * z_loss_sum`` with
  ``z`` known in the forward pass): per block of rows the GEMM's epilogue also stores
  ``exp(logit - tile max)`` in float16 and the target logit; one XLA pass forms the
  logit gradient and two cuBLAS GEMMs (bf16, or FP8 with ``fp8``) give ``dhidden`` and
  ``dweight``, which the backward pass only scales by the cotangent. Three GEMMs in all.
  Forming the gradient in the forward pass follows Liger Kernel's (BSD-2-Clause) and FLA's
  (MIT) fused linear cross entropy; here the logits stay on chip, and no code was copied.

Operands run in bfloat16 or float16 with float32 accumulation and float32
softmax statistics. Kernel shapes: ``V <= 2**24``, ``D % 64 == 0``. ``N`` is padded
to a multiple of 256 with ignored rows, and ``V`` with zero weight rows whose
columns both epilogues mask (any vocabulary size works, e.g. 50,304 or 151,936).
``_Config.cluster = 2`` runs 2-CTA clusters that share (TMA multicast) each
weight block between two row tiles; it is tested, and off by default.

The kernels are single-device custom calls with no partitioning rule: under
``jit`` with a sharded batch XLA would gather the whole batch onto every device.
:func:`unsupported_reason` therefore rejects calls that may span several
devices, i.e. when more than one device is visible, unless the call runs inside
a ``shard_map`` whose mesh axes are all manual (one device per body).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from jla.ops.mosaic_gpu._common import is_hopper, num_sms

__all__ = ["DTYPES", "EXP_DTYPE", "ce_sums", "multi_device_reason", "objective_sums", "unsupported_reason"]

Array = jax.Array

DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
TILE_M = 128  # rows of a CTA tile (each consumer warpgroup holds all of them)
TILE_N = 128  # vocabulary columns per consumer warpgroup; a CTA tile spans 2 * TILE_N
TILE_K = 64  # hidden-dimension block per pipeline step
VOCAB_ALIGN = 2 * TILE_N
_LOG2E = math.log2(math.e)


@dataclasses.dataclass(frozen=True)
class _Config:
    steps: int  # TMA pipeline depth
    grid_minor: int  # planar_snake minor dimension (0 = rows, 1 = vocabulary)
    grid_width: int  # planar_snake tile width
    cluster: int = 1  # CTAs per cluster along the rows; 2 multicasts each weight block to both


_STATS = _Config(steps=4, grid_minor=0, grid_width=8)
_GRAD = _Config(steps=3, grid_minor=0, grid_width=8)  # the 64 KB output tile leaves room for 3 stages
_STATS_EXP = _Config(steps=3, grid_minor=0, grid_width=8)  # the forward-gradient path's forward (64 KB exp tile)
# exp(logit - tile row max) is in (0, 1]: float16 keeps 11 significant bits there (bfloat16: 8).
EXP_DTYPE = jnp.dtype(jnp.float16)
ROW_ALIGN = 2 * TILE_M  # rows are padded to this, so either cluster size divides them


def unsupported_reason(hidden: Array, weight: Array, compute_dtype, *, devices: bool = True) -> str | None:
    """``None`` when the kernels accept these operands, otherwise why not.

    ``devices=False`` leaves out the multi-device check (for a caller that runs the call per shard).
    """
    if jnp.dtype(compute_dtype) not in DTYPES:
        return f"compute_dtype must be bfloat16 or float16; got {jnp.dtype(compute_dtype)}"
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        return f"expected hidden [N, D] and weight [V, D]; got {hidden.shape} and {weight.shape}"
    vocab, dim = weight.shape
    if vocab > 2**24:  # the backward compares target columns in float32
        return f"vocabulary size {vocab} must be at most 2**24"
    if dim % TILE_K:
        return f"hidden size {dim} must be a multiple of {TILE_K}"
    reason = multi_device_reason() if devices else None
    if reason is not None:
        return reason
    if not is_hopper():
        return "the Mosaic GPU cross entropy kernels need an sm_90 (Hopper) GPU"
    return None


def multi_device_reason() -> str | None:
    """``None`` when the call runs on one device, otherwise why it may not.

    One device: a single visible device, or the body of a ``shard_map`` whose mesh
    axes are all manual (or of size 1). Anything else may be partitioned by XLA,
    which would gather the operands of these custom calls onto every device.
    """
    count = jax.device_count()
    if count == 1:
        return None
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and all(size == 1 or name in mesh.manual_axes for name, size in mesh.shape.items()):
        return None
    return (f"{count} devices are visible and the call is not inside a shard_map with every mesh axis manual; "
            "the kernels have no partitioning rule, so XLA would gather a sharded batch onto every device")


def _transforms(minor: int, dtype) -> tuple:
    swizzle = plgpu.find_swizzle(minor * jnp.dtype(dtype).itemsize * 8)
    return plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle)


# An inner jit: identical calls (the backward's equal-size vocabulary chunks) share one traced
# jaxpr, and JAX lowers a jaxpr once per module, so each distinct kernel is compiled only once.
@functools.partial(jax.jit, static_argnames=("grad", "cfg", "valid", "store_exp"))
def _logit_kernel(h: Array, w: Array, vecs: Array | None, *, grad: bool, cfg: _Config, valid: int,
                  store_exp: bool = False):
    """Persistent GEMM over ``h [N, D] @ w[V, D]^T`` tiles with a cross-entropy epilogue.

    ``grad=False`` returns the per-row max and exp-sum of every 128-column tile,
    ``[V / 128, N]`` f32 each. With ``store_exp`` it also returns the target logit per
    tile (0 in the tiles without the row's target column, given as ``vecs [1, N]`` f32),
    same shape, and ``exp(logit - tile row max)`` ``[N, V]`` in :data:`EXP_DTYPE` (zero
    in padded columns). ``grad=True`` returns the logit gradient ``[N, V]`` in
    ``h.dtype`` from ``vecs [5, N]`` f32 rows (logsumexp, softmax coefficient, one-hot
    coefficient, flat coefficient, target column). Columns from ``valid`` on are
    vocabulary padding: the forward leaves them out of the max and exp-sum and the
    backward writes a zero gradient for them.
    """
    n, d = h.shape
    v = w.shape[0]
    masked = valid < v  # static: only padded calls emit the column mask
    dtype = h.dtype
    cluster = cfg.cluster
    m_iters, n_iters, k_iters = n // (TILE_M * cluster), v // VOCAB_ALIGN, d // TILE_K
    steps = min(cfg.steps, k_iters)
    in_transforms = _transforms(TILE_K, dtype)
    layout = plgpu.Layout.WGMMA
    row = plgpu.Layout.WGMMA.reduce(1)
    shape = (TILE_M, TILE_N)

    def rows(x):
        return lax.broadcast_in_dim(x, shape, [0])

    def get_pipeline(body, compute_context):
        return plgpu.emit_pipeline_warp_specialized(
            body,
            grid=(k_iters,),
            memory_registers=40,
            in_specs=[
                plgpu.BlockSpec((TILE_M, TILE_K), lambda k: (0, k), transforms=in_transforms),
                plgpu.BlockSpec((VOCAB_ALIGN, TILE_K), lambda k: (0, k), transforms=in_transforms,
                                collective_axes=("cluster",) if cluster > 1 else ()),
            ],
            wg_axis="wg",
            num_compute_wgs=2,
            max_concurrent_steps=steps,
            compute_context=compute_context,
        )

    def body(*refs):
        if grad:
            h_ref, w_ref, vec_ref, dl_ref = refs
        elif store_exp:
            h_ref, w_ref, vec_ref, max_ref, sum_ref, tgt_ref, exp_ref = refs
        else:
            h_ref, w_ref, max_ref, sum_ref = refs
        ignore = lambda *_, **__: None  # noqa: E731  (allocations do not depend on the bodies)
        scratch = dict(pipeline_allocs=get_pipeline(ignore, ignore).get_allocations(h_ref, w_ref))
        if grad:
            scratch["out_smem"] = plgpu.SMEM((2, TILE_M, TILE_N), dtype, transforms=_transforms(TILE_N, dtype))
        else:
            scratch["out_smem"] = plgpu.SMEM((2, 3 if store_exp else 2, TILE_M), jnp.float32)
        if store_exp:
            scratch["exp_smem"] = plgpu.SMEM((2, TILE_M, TILE_N), EXP_DTYPE, transforms=_transforms(TILE_N, EXP_DTYPE))

        @functools.partial(pl.run_scoped, **scratch, collective_axes="wg")
        def _scope(pipeline_allocs, out_smem, exp_smem=None):
            wg = lax.axis_index("wg")
            cta = lax.axis_index("cluster")

            @plgpu.nd_loop((m_iters * n_iters,), collective_axes="cluster_grid")
            def _tiles(info: plgpu.NDLoopInfo):
                (lin,) = info.index
                mi, ni = plgpu.planar_snake(lin, (m_iters, n_iters), cfg.grid_minor, cfg.grid_width)
                m_slice = pl.ds((mi * cluster + cta) * TILE_M, TILE_M)  # both CTAs share the weight block
                col0 = ni * VOCAB_ALIGN + wg * TILE_N  # first vocabulary column of this warpgroup

                def compute_context(eval_pipeline):
                    if grad:  # issued before the K loop; first used in the epilogue
                        lse, p_coef, hot_coef, flat_coef, target = (
                            plgpu.load(vec_ref.at[i, m_slice], layout=row, optimized=False) for i in range(5))
                    elif store_exp:
                        target = plgpu.load(vec_ref.at[0, m_slice], layout=row, optimized=False)

                    @functools.partial(pl.run_scoped, acc_ref=plgpu.ACC(shape, jnp.float32))
                    def _acc(acc_ref):
                        eval_pipeline(acc_ref)
                        logits = acc_ref[...]
                        if grad:
                            p = jnp.exp2(logits * _LOG2E - rows(lse * _LOG2E))
                            col = (plgpu.broadcasted_iota(jnp.int32, shape, 1, layout=layout) + col0)
                            hot = jnp.where(col.astype(jnp.float32) == rows(target), rows(hot_coef), 0.0)
                            dl = rows(p_coef) * p - hot - rows(flat_coef)
                            if masked:  # padded columns: exp(0 - lse) can overflow, so store exact zeros
                                dl = jnp.where(col < valid, dl, 0.0)
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                            out_smem.at[wg][...] = dl.astype(dtype)
                            plgpu.commit_smem()
                            plgpu.copy_smem_to_gmem(out_smem.at[wg], dl_ref.at[m_slice, pl.ds(col0, TILE_N)])
                        else:
                            if masked or store_exp:
                                col = (plgpu.broadcasted_iota(jnp.int32, shape, 1, layout=layout) + col0)
                            if store_exp:  # the target logit from this accumulator, consistent with the max
                                hit = col.astype(jnp.float32) == rows(target)
                                target_logit = jnp.where(hit, logits, 0.0).sum(axis=1)
                            if masked:  # the stats of a tile without real columns are dropped by _forward
                                logits = jnp.where(col < valid, logits, -jnp.inf)
                            row_max = logits.max(axis=1)
                            ex = jnp.exp2(logits * _LOG2E - rows(row_max * _LOG2E))
                            if store_exp and masked:  # a tile of padding only has row max -inf: exp gives NaN
                                ex = jnp.where(col < valid, ex, 0.0)
                            row_sum = ex.sum(axis=1)
                            part = ni * 2 + wg
                            plgpu.wait_smem_to_gmem(0, wait_read_only=True)
                            out_smem.at[wg, 0][...] = row_max
                            out_smem.at[wg, 1][...] = row_sum
                            if store_exp:
                                out_smem.at[wg, 2][...] = target_logit
                                exp_smem.at[wg][...] = ex.astype(EXP_DTYPE)
                            plgpu.commit_smem()
                            plgpu.copy_smem_to_gmem(out_smem.at[wg, 0], max_ref.at[part, m_slice],
                                                    commit_group=False)
                            plgpu.copy_smem_to_gmem(out_smem.at[wg, 1], sum_ref.at[part, m_slice],
                                                    commit_group=False)
                            if store_exp:
                                plgpu.copy_smem_to_gmem(out_smem.at[wg, 2], tgt_ref.at[part, m_slice],
                                                        commit_group=False)
                                plgpu.copy_smem_to_gmem(exp_smem.at[wg], exp_ref.at[m_slice, pl.ds(col0, TILE_N)],
                                                        commit_group=False)
                            plgpu.commit_smem_to_gmem_group()

                def mma(_, h_smem, w_smem, acc_ref):
                    plgpu.wgmma(acc_ref, h_smem, w_smem.at[pl.ds(wg * TILE_N, TILE_N)].transpose((1, 0)))
                    plgpu.wgmma_wait(0)
                    return acc_ref

                get_pipeline(mma, compute_context)(
                    h_ref.at[m_slice, :], w_ref.at[pl.ds(ni * VOCAB_ALIGN, VOCAB_ALIGN), :],
                    allocations=pipeline_allocs)

        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    if grad:
        out_type = jax.ShapeDtypeStruct((n, v), dtype)
    else:
        part = jax.ShapeDtypeStruct((2 * n_iters, n), jnp.float32)
        out_type = (part, part, part, jax.ShapeDtypeStruct((n, v), EXP_DTYPE)) if store_exp else (part, part)
    kernel = plgpu.kernel(
        body,
        out_type=out_type,
        grid=(num_sms() // cluster,),
        grid_names=("cluster_grid",),
        cluster=(cluster,),
        cluster_names=("cluster",),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )
    return kernel(h, w, vecs) if grad or store_exp else kernel(h, w)


def _forward(h: Array, w: Array, targets: Array, label_smoothing: float, vocab: int):
    """Per-token ``(ce, lse)`` f32 from the compute-dtype operands (``w`` rows from ``vocab`` on are zero)."""
    row_max, row_sum = _logit_kernel(h, w, None, grad=False, cfg=_STATS, valid=vocab)
    parts = -(-vocab // TILE_N)  # 128-column tiles holding real columns (the rest are all padding)
    row_max, row_sum = row_max[:parts], row_sum[:parts]
    top = jnp.max(row_max, axis=0)
    lse = top + jnp.log(jnp.sum(row_sum * jnp.exp(row_max - top[None, :]), axis=0))
    target_logit = jnp.einsum("nd,nd->n", h, jnp.take(w, targets, axis=0), preferred_element_type=jnp.float32)
    per_token = lse - target_logit
    if label_smoothing:  # the zero padding rows add nothing to the column sum
        mean_logit = jnp.dot(h.astype(jnp.float32), jnp.sum(w.astype(jnp.float32), axis=0)) / vocab
        per_token = (1.0 - label_smoothing) * per_token + label_smoothing * (lse - mean_logit)
    return per_token, lse


def _sums(per_token, lse, valid):
    return jnp.sum(jnp.where(valid, per_token, 0.0)), jnp.sum(jnp.where(valid, lse * lse, 0.0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _ce_sums(hidden, weight, targets, valid, label_smoothing, chunk, compute_dtype, vocab):
    h, w = hidden.astype(compute_dtype), weight.astype(compute_dtype)
    per_token, lse = _forward(h, w, targets, label_smoothing, vocab)
    return _sums(per_token, lse, valid)


def _ce_sums_fwd(hidden, weight, targets, valid, label_smoothing, chunk, compute_dtype, vocab):
    h, w = hidden.astype(compute_dtype), weight.astype(compute_dtype)
    per_token, lse = _forward(h, w, targets, label_smoothing, vocab)
    # Zero-size arrays carry the primal dtypes to the backward pass.
    tags = (jnp.zeros((0,), hidden.dtype), jnp.zeros((0,), weight.dtype))
    return _sums(per_token, lse, valid), (h, w, targets, valid, lse, tags)


def _ce_sums_bwd(label_smoothing, chunk, compute_dtype, vocab, residuals, cotangents):
    h, w, targets, valid, lse, (hidden_tag, weight_tag) = residuals
    g_ce, g_z = (jnp.asarray(c, jnp.float32) for c in cotangents)
    padded, dim = w.shape
    row_scale = valid.astype(jnp.float32)
    # d(ce)/dlogit = p - (1 - ls) onehot - ls / V;  d(lse^2)/dlogit = 2 lse p
    p_coef = (g_ce + 2.0 * g_z * lse) * row_scale
    hot_coef = g_ce * (1.0 - label_smoothing) * row_scale
    flat_coef = jnp.broadcast_to(g_ce * (label_smoothing / vocab), lse.shape) * row_scale
    grad_hidden = jnp.zeros((h.shape[0], dim), jnp.float32)
    grad_weight = jnp.zeros((padded, dim), weight_tag.dtype)
    for start in range(0, padded, chunk):
        size = min(chunk, padded - start)
        # Chain every chunk to the previous one's gradients: the chunks are independent, and
        # without the chain XLA may run all logit-gradient kernels first and keep the whole
        # [N, V] gradient alive.
        h, grad_hidden, grad_weight = lax.optimization_barrier((h, grad_hidden, grad_weight))
        w_chunk = lax.slice_in_dim(w, start, start + size)
        vecs = jnp.stack([lse, p_coef, hot_coef, flat_coef, (targets - start).astype(jnp.float32)])
        grad_logits = _logit_kernel(h, w_chunk, vecs, grad=True, cfg=_GRAD,
                                    valid=min(size, vocab - start))  # [N, size] compute dtype
        grad_hidden = grad_hidden + lax.dot_general(
            grad_logits, w_chunk, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        grad_w_chunk = lax.dot_general(grad_logits, h, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        grad_weight = lax.dynamic_update_slice_in_dim(grad_weight, grad_w_chunk.astype(weight_tag.dtype), start, 0)
    return grad_hidden.astype(hidden_tag.dtype), grad_weight, None, None


_ce_sums.defvjp(_ce_sums_fwd, _ce_sums_bwd)


def ce_sums(hidden: Array, weight: Array, targets: Array, valid: Array, *, label_smoothing: float,
            vocab_chunk: int, compute_dtype) -> tuple[Array, Array]:
    """``(sum of CE, sum of lse^2)`` over ``valid`` rows (``targets`` already in ``[0, V)``).

    ``vocab_chunk`` (rounded up to a multiple of 256) sets the vocabulary columns
    of each backward chunk, whose logit gradient ``[N, vocab_chunk]`` is the only
    ``N x V``-sized buffer. Raises ``ValueError`` when :func:`unsupported_reason`
    rejects the operands.
    """
    reason = unsupported_reason(hidden, weight, compute_dtype)
    if reason is not None:
        raise ValueError(f"Mosaic GPU linear cross entropy: {reason}")
    pad = -hidden.shape[0] % ROW_ALIGN
    if pad:  # padded rows are ignored tokens with zero activations
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        targets = jnp.pad(targets, (0, pad))
        valid = jnp.pad(valid, (0, pad))
    vocab = weight.shape[0]
    if vocab % VOCAB_ALIGN:  # zero rows; the kernels mask their columns, autodiff slices their gradient off
        weight = jnp.pad(weight, ((0, -vocab % VOCAB_ALIGN), (0, 0)))
    chunk = min(-(-max(int(vocab_chunk), 1) // VOCAB_ALIGN) * VOCAB_ALIGN, weight.shape[0])
    return _ce_sums(hidden, weight, targets.astype(jnp.int32), valid, float(label_smoothing), chunk,
                    jnp.dtype(compute_dtype), vocab)


# --- Forward-gradient path: the logit gradient formed during the forward pass (three GEMMs) ---------------------


_E4M3, _E5M2 = jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)


def _fp8_scale(amax: Array, dtype) -> Array:
    """Per-tensor dequantization scale ``amax / fp8 max`` (1 where ``amax`` is 0 or not finite).

    The recipe of :mod:`jla.layers.fp8` with ``recipe="current"``: the scale comes from the
    tensor being quantized, so it needs no state.
    """
    amax = jnp.asarray(amax, jnp.float32)
    return jnp.where(jnp.isfinite(amax) & (amax > 0), amax / float(jnp.finfo(dtype).max), 1.0).astype(jnp.float32)


def _quantize(x: Array, scale: Array, dtype) -> Array:
    bound = float(jnp.finfo(dtype).max)
    return jnp.clip(x.astype(jnp.float32) / scale, -bound, bound).astype(dtype)


def _fp8_dot(a: Array, a_scale: Array, b: Array, b_scale: Array, dims) -> Array:
    """``(a * a_scale) . (b * b_scale)`` in float32 from FP8 operands: one cuBLASLt FP8 GEMM on Hopper
    (fast accumulation off, as for the e5m2 gradient GEMMs of :mod:`jla.layers.fp8`)."""
    out = lax.dot_general(a, b, dims, precision=lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return out * (a_scale * b_scale)


def _chunk_gradient(h: Array, w: Array, targets: Array, valid: Array, label_smoothing: float, vocab: int,
                    z_weight: float, fp8: bool = False):
    """One row block: per-token ``(ce, lse)`` f32, the unit-cotangent logit gradient ``[n, V]`` and its scale.

    The kernel stores ``e = exp(logit - m)`` in :data:`EXP_DTYPE` with ``m`` the max of each row
    over its 128-column tile, next to the per-tile max, exp-sum and target logit (all from the
    same float32 accumulator). With the logsumexp from those, the gradient of
    ``ce + z_weight * lse**2`` per logit is
    ``(1 + 2 z_weight lse) exp(m - lse) e - (1 - ls) onehot(target) - ls / V`` (zero on ignored
    rows), one XLA elementwise pass that writes it in the compute dtype (scale ``None``), or with
    ``fp8`` in float8_e5m2 with a per-block scale from the bound ``|gradient| <= max |1 + 2 z lse| + 1``.
    The target column is formed from the float32 target logit instead of ``e``: ``p - 1`` cancels there.
    """
    row_max, row_sum, tile_target, ex = _logit_kernel(h, w, targets.astype(jnp.float32)[None, :], grad=False,
                                                      cfg=_STATS_EXP, valid=vocab, store_exp=True)
    parts = -(-vocab // TILE_N)  # 128-column tiles holding real columns
    top = jnp.max(row_max[:parts], axis=0)
    lse = top + jnp.log(jnp.sum(row_sum[:parts] * jnp.exp(row_max[:parts] - top[None, :]), axis=0))
    target_logit = jnp.sum(tile_target[:parts], axis=0)  # one tile holds the target; the others are 0
    per_token = lse - target_logit
    if label_smoothing:
        mean_logit = jnp.dot(h.astype(jnp.float32), jnp.sum(w.astype(jnp.float32), axis=0)) / vocab
        per_token = (1.0 - label_smoothing) * per_token + label_smoothing * (lse - mean_logit)
    row_scale = valid.astype(jnp.float32)
    p_coef = (1.0 + 2.0 * z_weight * lse) * row_scale
    hot_coef = (1.0 - label_smoothing) * row_scale
    n, padded = ex.shape
    # exp(m - lse) per row and tile; a tile of padding only has m = -inf, so 0 (and its e is 0).
    tile_coef = (jnp.exp(row_max - lse[None, :]) * p_coef[None, :]).T  # [n, padded / 128]
    grad = (ex.reshape(n, padded // TILE_N, TILE_N).astype(jnp.float32) * tile_coef[:, :, None]).reshape(n, padded)
    col = lax.broadcasted_iota(jnp.int32, (n, padded), 1)
    target_grad = p_coef * jnp.exp(target_logit - lse) - hot_coef
    grad = jnp.where(col == targets[:, None], target_grad[:, None], grad)
    if label_smoothing:
        grad = grad - jnp.where(col < vocab, ((label_smoothing / vocab) * row_scale)[:, None], 0.0)
    if fp8:
        scale = _fp8_scale(jnp.max(jnp.abs(p_coef)) + 1.0, _E5M2)
        return per_token, lse, _quantize(grad, scale, _E5M2), scale
    return per_token, lse, grad.astype(h.dtype), None


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _objective_sums(hidden, weight, targets, valid, label_smoothing, z_weight, row_chunk, compute_dtype, vocab, fp8):
    # Without differentiation (evaluation) only the statistics kernel runs.
    h, w = hidden.astype(compute_dtype), weight.astype(compute_dtype)
    per_token, lse = _forward(h, w, targets, label_smoothing, vocab)
    loss_sum, z_sum = _sums(per_token, lse, valid)
    return loss_sum + z_weight * z_sum, loss_sum, z_sum


def _objective_sums_fwd(hidden, weight, targets, valid, label_smoothing, z_weight, row_chunk, compute_dtype, vocab,
                        fp8):
    h, w = hidden.astype(compute_dtype), weight.astype(compute_dtype)
    n, dim = h.shape
    grad_weight = jnp.zeros((w.shape[0], dim), jnp.float32)
    if fp8:  # e4m3 operands with per-tensor scales from their amax (current scaling)
        h_scale = _fp8_scale(jnp.max(jnp.abs(h)), _E4M3)
        w_scale = _fp8_scale(jnp.max(jnp.abs(w)), _E4M3)
        w8 = _quantize(w, w_scale, _E4M3)
    dgrad, wgrad = (((1,), (0,)), ((), ())), (((0,), (0,)), ((), ()))  # dlogits @ w, dlogits^T @ h
    per_token, lse, grad_hidden = [], [], []
    pending = jnp.zeros((0, dim), jnp.float32)
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        # Chain each row block to the previous one's GEMMs, so one block's exp and gradient
        # buffers are live at a time (the blocks are independent and XLA could overlap them).
        h, grad_weight, pending = lax.optimization_barrier((h, grad_weight, pending))
        grad_hidden.append(pending)
        h_rows = lax.slice_in_dim(h, start, stop)
        tok, lse_rows, grad_logits, g_scale = _chunk_gradient(h_rows, w, targets[start:stop], valid[start:stop],
                                                              label_smoothing, vocab, z_weight, fp8)
        if fp8:
            pending = _fp8_dot(grad_logits, g_scale, w8, w_scale, dgrad)
            grad_weight = grad_weight + _fp8_dot(grad_logits, g_scale, _quantize(h_rows, h_scale, _E4M3), h_scale,
                                                 wgrad)
        else:
            pending = lax.dot_general(grad_logits, w, dgrad, preferred_element_type=jnp.float32)
            grad_weight = grad_weight + lax.dot_general(grad_logits, h_rows, wgrad, preferred_element_type=jnp.float32)
        per_token.append(tok)
        lse.append(lse_rows)
    grad_hidden.append(pending)
    loss_sum, z_sum = _sums(jnp.concatenate(per_token), jnp.concatenate(lse), valid)
    # Zero-size arrays carry the primal dtypes to the backward pass.
    tags = (jnp.zeros((0,), hidden.dtype), jnp.zeros((0,), weight.dtype))
    return (loss_sum + z_weight * z_sum, loss_sum, z_sum), (jnp.concatenate(grad_hidden), grad_weight, tags)


def _objective_sums_bwd(label_smoothing, z_weight, row_chunk, compute_dtype, vocab, fp8, residuals, cotangents):
    grad_hidden, grad_weight, (hidden_tag, weight_tag) = residuals
    # Only the objective carries a gradient: objective_sums stops the gradients of the two sums.
    g = jnp.asarray(cotangents[0], jnp.float32)
    return (g * grad_hidden).astype(hidden_tag.dtype), (g * grad_weight).astype(weight_tag.dtype), None, None


_objective_sums.defvjp(_objective_sums_fwd, _objective_sums_bwd)


def objective_sums(hidden: Array, weight: Array, targets: Array, valid: Array, *, label_smoothing: float,
                   z_loss_weight: float, row_chunk: int, compute_dtype, fp8: bool = False) -> tuple[Array, Array, Array]:
    """``(sum of CE + z_loss_weight * sum of lse^2, sum of CE, sum of lse^2)`` over ``valid`` rows.

    Only the first output is differentiable (the other two are returned through
    ``stop_gradient``). Under differentiation its gradient is formed during the forward
    pass, per block of ``row_chunk`` rows (rounded up to a multiple of 256): the logits
    GEMM kernel keeps the float32 logits on chip and stores ``exp(logit - tile max)`` in
    float16, one elementwise pass turns that into the logit gradient, and two cuBLAS
    GEMMs give ``dhidden`` and ``dweight``, which the backward pass scales by the
    cotangent. Three GEMMs instead of :func:`ce_sums`' four (no logits recompute), for
    ``4 * row_chunk * V`` bytes of transient buffers and float32 ``dhidden``/``dweight``
    residuals. ``fp8`` runs the two gradient GEMMs in FP8: the logit gradient in float8_e5m2
    (scaled by a bound per row block), ``hidden`` and ``weight`` in float8_e4m3fn with
    per-tensor scales from their amax (current scaling); the logits GEMM, and so the loss,
    stay in the compute dtype. Raises ``ValueError`` when :func:`unsupported_reason` rejects
    the operands.
    """
    reason = unsupported_reason(hidden, weight, compute_dtype)
    if reason is not None:
        raise ValueError(f"Mosaic GPU linear cross entropy: {reason}")
    pad = -hidden.shape[0] % ROW_ALIGN
    if pad:  # padded rows are ignored tokens with zero activations
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        targets = jnp.pad(targets, (0, pad))
        valid = jnp.pad(valid, (0, pad))
    vocab = weight.shape[0]
    if vocab % VOCAB_ALIGN:  # zero rows; the kernel masks their columns, autodiff slices their gradient off
        weight = jnp.pad(weight, ((0, -vocab % VOCAB_ALIGN), (0, 0)))
    chunk = min(-(-max(int(row_chunk), 1) // ROW_ALIGN) * ROW_ALIGN, hidden.shape[0])
    objective, loss_sum, z_sum = _objective_sums(hidden, weight, targets.astype(jnp.int32), valid,
                                                 float(label_smoothing), float(z_loss_weight), chunk,
                                                 jnp.dtype(compute_dtype), vocab, bool(fp8))
    return objective, lax.stop_gradient(loss_sum), lax.stop_gradient(z_sum)
