# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2024 The JAX Authors.
# Adapted from JAX (Apache-2.0), jax/experimental/pallas/ops/gpu/rms_norm.py (Pallas Triton row kernels of an
# RMSNorm forward and backward), state-spaces/mamba (Apache-2.0), mamba_ssm/ops/triton/layernorm_gated.py
# (RMSNormGated with norm_before_gate=False, forward and backward formulas) and flash-linear-attention (MIT),
# fla/modules/fused_norm_gate.py (gated RMSNorm backward).
"""Fused output glue of the Mamba-2 mixer: D skip, gate, RMSNorm and out-projection (``SSMConfig.fused_glue``).

The unfused layer runs ``y + D x`` (rounded to bfloat16), ``rmsnorm((y + D x) silu(z)) * w`` and the
out-projection as XLA fusions. On the 1.3B step XLA materializes the float32 product ``(y + D x) silu(z)``
in the forward pass, runs the RMSNorm's row reductions at about half the HBM bandwidth, and reduces
``dout * normalized`` over all ``B * T`` rows for the norm weight's gradient (``benchmarks/end_to_end_1p3b.md``).

:func:`gated_norm_out` (one B/C group, one RMS factor per token) runs instead:

* forward: one Pallas Triton kernel per token row reads ``y, x, z`` once and writes the bfloat16 normalized
  input ``a = rmsnorm((y + D x) silu(z))`` (without the norm weight) and the per-token RMS factor; the norm
  weight is folded into the out-projection weight, ``(a * w) @ W = a @ (w W)``, a cuBLAS GEMM;
* backward: ``da = dout (w W)^T`` and ``d(w W) = a^T dout`` (cuBLAS; the norm weight's gradient is
  ``sum_n d(w W)[c, n] W[c, n]``, so no column reduction over the rows), then one Pallas Triton kernel per row
  reads ``da, y, x, z`` and writes ``dy, dz`` and the row's per-head sums of ``dy * x`` for ``dD``.

:func:`gated_norm` is the same pair of row kernels for an out-projection that keeps its own module (an FP8
out-projection): it returns ``a * w`` and takes the norm weight's gradient as a column reduction of ``dout * a``.
:func:`ssd_core` is :func:`jla.layers.ssm.mamba2_ssd`'s kernel dispatch without the D skip; a call that is not
differentiated forms the SSD values ``dt x`` inside the Mosaic GPU SSD kernel instead of in HBM (``v_parts``).
A head count that is not a power of two (24 heads of 64 at ``d_model`` 768) runs masked versions of the row
kernels, which pad each row to the next power of two of heads.

On several devices (the trainer's FSDP / data-parallel mesh) the two row kernels run on each device's rows
(``jla.distributed.kernel_sharding.shard_layout_call``: rows split along the batch, ``D`` replicated) and the
GEMMs stay XLA's, which partitions them as it partitions the unfused layer's out-projection.

Same parameters, tree paths and forward values as the unfused layer up to rounding order (the unfused layer
rounds ``y + D x`` to bfloat16 and multiplies the norm weight before the GEMM; here ``w W`` is rounded once).
``tests/test_mamba2_fused.py`` checks the kernels (interpreted on the CPU, compiled on the GPU) and the layer
against the unfused layer, outputs and every gradient.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl

from jla import ops
from jla.layers.linear import LINEAR_CHECKPOINT_NAME

__all__ = ["device_layout_ok", "gated_norm", "gated_norm_out", "kernel_shape_ok", "ssd_core", "heads_split"]

Array = jax.Array
f32 = jnp.float32
NUM_WARPS = 8  # one 4096-channel row per program: 16 elements per thread


# SSD core (no D skip) --------------------------------------------------------------------------
def ssd_core(v: Array, g: Array, b: Array, c: Array, *, segment_ids: Array | None, impl: str, chunk_size: int,
             grouped_key_grads: bool, v_parts: tuple[Array, Array] | None = None) -> Array:
    """``y [B, T, H, P]`` of ``simple_gla(q=C, k=B, v, g, scale=1)``: :func:`jla.layers.ssm.mamba2_ssd`'s
    dispatch (grouped ``B, C`` on the Mosaic GPU kernels, else repeated per head) for a training call, without
    the D skip (:func:`gated_norm_out` adds it).

    ``v_parts = (x, dt)`` are the factors of ``v = (x * dt[..., None])`` rounded to ``v``'s dtype (``x`` in
    ``v``'s dtype). A call that is not differentiated (inference, evaluation) then runs the Mosaic GPU kernels on
    ``x`` and ``dt`` and forms ``v`` inside the fused forward kernel (``scalar_gate.run(..., v_scale=dt)``),
    so the caller's ``[B, T, H, P]`` product ``v`` becomes dead code; same values, bit for bit (a split sequence,
    ``scalar_gate.pieces > 1``, forms ``v`` in XLA). Under differentiation the call is the one on ``v``, so the
    forward + backward pass is unchanged. Single device, grouped key gradients and the Mosaic GPU path only;
    other calls ignore ``v_parts``."""
    if v_parts is not None and _fold_applies(v, g, b, c, v_parts, segment_ids, impl, chunk_size, grouped_key_grads):
        x, dt = v_parts
        return _ssd_folded(v, x, dt, g, b, c, segment_ids, impl, chunk_size, grouped_key_grads)
    return _ssd(v, g, b, c, segment_ids=segment_ids, impl=impl, chunk_size=chunk_size,
                grouped_key_grads=grouped_key_grads)


def _ssd(v, g, b, c, *, segment_ids, impl, chunk_size, grouped_key_grads):
    from jla.layers.ssm import _grouped_simple_gla, _repeat_groups

    heads = v.shape[2]
    b = b.astype(c.dtype)
    options = dict(scale=1.0, chunk_size=chunk_size, segment_ids=segment_ids, initial_state=None,
                   output_final_state=False)
    if grouped_key_grads and _grouped_simple_gla(impl, c, b, v, g, **options):
        from jla.distributed.kernel_sharding import shard_kernel_call
        from jla.ops.mosaic_gpu import scalar_gate

        def grouped(q, k, v, g, chunk_size=None, **rest):
            del chunk_size  # the kernels use their own chunk length
            return scalar_gate.run(q, k, v, g, None, delta=False, grouped_key_grads=True, **rest)

        y, _ = shard_kernel_call(grouped, c, b, v, (g,), options, supports=scalar_gate.supports_simple_gla,
                                 kind="simple_gla")
    elif _grouped_simple_gla(impl, c, b, v, g, **options):
        y, _ = ops.linear_attention("simple_gla", c, b, v, g=g, impl="mosaic_gpu", **options)
    else:
        q, k = _repeat_groups(c, heads), _repeat_groups(b, heads)
        y, _ = ops.linear_attention("simple_gla", q, k.astype(q.dtype), v, g=g, impl=impl, **options)
    return y


def _fold_applies(v, g, b, c, v_parts, segment_ids, impl, chunk_size, grouped_key_grads) -> bool:
    from jla.layers.ssm import _grouped_simple_gla

    x, dt = v_parts
    if not grouped_key_grads or jax.device_count() != 1 or x.shape != v.shape or dt.shape != v.shape[:3]:
        return False
    if not (x.dtype == v.dtype == c.dtype and jnp.issubdtype(dt.dtype, jnp.floating)):
        return False
    return _grouped_simple_gla(impl, c, b.astype(c.dtype), v, g, scale=1.0, chunk_size=chunk_size,
                               segment_ids=segment_ids, initial_state=None, output_final_state=False)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9))
def _ssd_folded(v, x, dt, g, b, c, segment_ids, impl, chunk_size, grouped_key_grads):
    from jla.ops.mosaic_gpu import scalar_gate

    del v, impl, chunk_size  # v = (x * dt) is formed in the kernel
    y, _ = scalar_gate.run(c, b.astype(c.dtype), x, g, None, delta=False, scale=1.0, initial_state=None,
                           output_final_state=False, segment_ids=segment_ids, grouped_key_grads=grouped_key_grads,
                           v_scale=dt)
    return y


def _ssd_folded_fwd(v, x, dt, g, b, c, segment_ids, impl, chunk_size, grouped_key_grads):
    def unfolded(v, g, b, c):
        return _ssd(v, g, b, c, segment_ids=segment_ids, impl=impl, chunk_size=chunk_size,
                    grouped_key_grads=grouped_key_grads)

    return jax.vjp(unfolded, v, g, b, c)


def _ssd_folded_bwd(impl, chunk_size, grouped_key_grads, pullback, dy):
    del impl, chunk_size, grouped_key_grads
    dv, dg, db, dc = pullback(dy)
    return dv, None, None, dg, db, dc, None  # x, dt get their gradients through v = x dt


_ssd_folded.defvjp(_ssd_folded_fwd, _ssd_folded_bwd)


def _pow2(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def _pow2_ceil(n: int) -> int:
    """The smallest power of two ``>= n`` (``n >= 1``)."""
    return 1 << (n - 1).bit_length()


def kernel_shape_ok(heads: int, head_dim: int) -> bool:
    """Triton blocks need power-of-two sides; one row of ``heads * head_dim`` channels per program. A head count
    that is not a power of two (24 heads of 64 at ``d_model`` 768) runs the masked row kernels, which pad the row
    to the next power of two of heads."""
    return heads >= 1 and _pow2(head_dim) and 16 <= _pow2_ceil(heads) * head_dim <= 16384


def device_layout_ok() -> bool:
    """Whether a call traced here can run the row kernels: on one device, inside a ``shard_map`` whose mesh axes
    are all manual, or on several devices under a context mesh that can split the rows (the trainer's: the kernels
    then run per shard, :func:`_per_shard`). A multi-device call without such a mesh keeps the unfused layer."""
    if jax.device_count() == 1:
        return True
    from jla.distributed.kernel_sharding import mesh_can_split, per_device_call

    return per_device_call() or mesh_can_split()


def heads_split() -> bool:
    """Whether a multi-device call traced here has a context mesh that splits the heads: an automatic axis of size
    > 1 in the ``heads`` rule (tensor parallelism). The per-shard rule splits only rows, so the head-sharded ``y``,
    ``x`` and ``z`` would be resharded into row shards and ``a`` back for the out-projection in both passes, a cost
    not measured yet: the Mamba-2 layer keeps its unfused glue under such a mesh
    (:func:`jla.layers.ssm._fused_glue_applies`); these functions still run per shard when called there."""
    if jax.device_count() == 1:
        return False
    from jla.distributed.kernel_sharding import per_device_call, rule_axes

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or per_device_call():
        return False
    heads = set(rule_axes("heads"))
    return any(size > 1 and name in heads and name not in mesh.manual_axes for name, size in mesh.shape.items())


def _per_shard() -> bool:
    """Several devices under a context mesh with an automatic axis to split the rows over."""
    if jax.device_count() == 1:
        return False
    from jla.distributed.kernel_sharding import mesh_can_split, per_device_call

    return not per_device_call() and mesh_can_split()


def _rows_per_shard(body, unsplit, operands: dict):
    """``body(**operands)`` on each device's rows of the context mesh (``jla.distributed.kernel_sharding``).

    The rule: every ``[R, ...]`` operand and output splits along its rows (the tokens of a ``[B, T, ...]``
    activation, which split along ``B``) over the ``batch`` rule's mesh axes, and every other mesh axis of size > 1
    splits the rows too; ``d [H]`` enters replicated. Each row is normalized on its own, so each device computes
    exactly what one GPU computes for its rows. When no split divides the rows the plain-JAX formulas run
    (``unsplit``), which XLA partitions.
    """
    from jla.distributed.kernel_sharding import shard_layout_call

    layouts = {name: "replicated" if name == "d" else "batch" for name in operands}
    return shard_layout_call(body, operands, layouts, out_layouts="batch", unsplit=unsplit)


def _norm_forward_xla(y, x, z, d, *, eps: float, dtype, z_row: int | None = None):
    """:func:`norm_forward`'s formulas in plain JAX (the fallback XLA partitions)."""
    z = z if z_row is None else z[:, z_row]
    y, x, z = y.astype(f32), x.astype(f32), z.astype(f32)
    g = (y + d.astype(f32)[:, None] * x) * (z * _sigmoid(z))
    rstd = lax.rsqrt(jnp.mean(g * g, axis=(1, 2)) + eps)
    return (g * rstd[:, None, None]).astype(dtype), rstd


def _norm_backward_xla(da, y, x, z, d, rstd, *, z_row: int | None = None):
    """:func:`norm_backward`'s formulas in plain JAX (the fallback XLA partitions)."""
    z = z if z_row is None else z[:, z_row]
    out_dtypes = (y.dtype, z.dtype)
    da, y, x, z = da.astype(f32), y.astype(f32), x.astype(f32), z.astype(f32)
    rstd = rstd[:, None, None]
    skip = y + d.astype(f32)[:, None] * x
    s = _sigmoid(z)
    silu = z * s
    a = skip * silu * rstd
    dg = (da - a * jnp.mean(da * a, axis=(1, 2), keepdims=True)) * rstd
    dskip = dg * silu
    dz = dg * skip * (s * (1.0 + z * (1.0 - s)))
    return dskip.astype(out_dtypes[0]), dz.astype(out_dtypes[1]), jnp.sum(dskip * x, axis=2)


def _sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def _fwd_kernel(y_ref, x_ref, z_ref, d_ref, a_ref, rstd_ref, *, eps: float, z_row: int | None = None):
    z = z_ref[...] if z_row is None else z_ref[z_row]  # z_row: row z_row of a stacked [n, H, P] block
    y, x, z = y_ref[...].astype(f32), x_ref[...].astype(f32), z.astype(f32)  # [H, P]
    skip = y + d_ref[...][:, None] * x
    g = skip * (z * _sigmoid(z))
    rstd = lax.rsqrt(jnp.sum(jnp.sum(g * g, axis=1)) / g.size + eps)
    a_ref[...] = (g * rstd).astype(a_ref.dtype)
    rstd_ref[...] = rstd


def _bwd_kernel(da_ref, y_ref, x_ref, z_ref, d_ref, rstd_ref, dy_ref, dz_ref, dd_ref, *, z_row: int | None = None):
    da = da_ref[...].astype(f32)
    z = z_ref[...] if z_row is None else z_ref[z_row]
    y, x, z = y_ref[...].astype(f32), x_ref[...].astype(f32), z.astype(f32)
    rstd = rstd_ref[...]
    skip = y + d_ref[...][:, None] * x
    s = _sigmoid(z)
    silu = z * s
    a = skip * silu * rstd
    c1 = jnp.sum(jnp.sum(da * a, axis=1)) / a.size
    dg = (da - a * c1) * rstd
    dskip = dg * silu
    dy_ref[...] = dskip.astype(dy_ref.dtype)
    dz_ref[...] = (dg * skip * (s * (1.0 + z * (1.0 - s)))).astype(dz_ref.dtype)
    dd_ref[...] = jnp.sum(dskip * x, axis=1)


# Masked row kernels for a head count that is not a power of two: flat rows of ``H * P`` channels (a stacked
# ``z`` with ``z_row``: ``n * H * P``) read and written with masked gathers over the next power of two of heads,
# zeros in the padding (they add nothing to the sums). The arithmetic is the kernels' above.
def _flat_cols(heads: int, head_dim: int):
    cols = jnp.arange(_pow2_ceil(heads) * head_dim, dtype=jnp.int32)
    return cols, cols < heads * head_dim


def _load_cols(ref, cols, mask, offset: int = 0):
    from jax.experimental.pallas import triton as pltriton

    return pltriton.load(ref.at[cols + offset if offset else cols], mask=mask, other=0.0).astype(f32)


def _store_cols(ref, cols, mask, value):
    from jax.experimental.pallas import triton as pltriton

    pltriton.store(ref.at[cols], value.astype(ref.dtype), mask=mask)


def _head_sums(v, heads: int, head_dim: int):
    """``[H_pad]`` per-head sums of a flat padded row."""
    return jnp.sum(v.reshape(_pow2_ceil(heads), head_dim), axis=1)


def _fwd_kernel_masked(y_ref, x_ref, z_ref, d_ref, a_ref, rstd_ref, *, eps: float, heads: int, head_dim: int,
                       z_row: int | None = None):
    width = heads * head_dim
    cols, mask = _flat_cols(heads, head_dim)
    y, x = _load_cols(y_ref, cols, mask), _load_cols(x_ref, cols, mask)
    z = _load_cols(z_ref, cols, mask, 0 if z_row is None else z_row * width)
    d = _load_cols(d_ref, cols // head_dim, mask)
    skip = y + d * x
    g = skip * (z * _sigmoid(z))
    rstd = lax.rsqrt(jnp.sum(_head_sums(g * g, heads, head_dim)) / width + eps)
    _store_cols(a_ref, cols, mask, g * rstd)
    rstd_ref[...] = rstd


def _bwd_kernel_masked(da_ref, y_ref, x_ref, z_ref, d_ref, rstd_ref, dy_ref, dz_ref, dd_ref, *, heads: int,
                       head_dim: int, z_row: int | None = None):
    width = heads * head_dim
    cols, mask = _flat_cols(heads, head_dim)
    da, y, x = (_load_cols(r, cols, mask) for r in (da_ref, y_ref, x_ref))
    z = _load_cols(z_ref, cols, mask, 0 if z_row is None else z_row * width)
    d = _load_cols(d_ref, cols // head_dim, mask)
    rstd = rstd_ref[...]
    skip = y + d * x
    s = _sigmoid(z)
    silu = z * s
    a = skip * silu * rstd
    c1 = jnp.sum(_head_sums(da * a, heads, head_dim)) / width
    dg = (da - a * c1) * rstd
    dskip = dg * silu
    _store_cols(dy_ref, cols, mask, dskip)
    _store_cols(dz_ref, cols, mask, dg * skip * (s * (1.0 + z * (1.0 - s))))
    h = jnp.arange(_pow2_ceil(heads), dtype=jnp.int32)
    _store_cols(dd_ref, h, h < heads, _head_sums(dskip * x, heads, head_dim))


def _compiler_params():
    from jax.experimental.pallas import triton as pltriton

    return pltriton.CompilerParams(num_warps=NUM_WARPS, num_stages=1)


def _rows(fn, n_rows_args, *args):
    """``fn`` (a per-row ``pallas_call``) over the leading row axis of the first ``n_rows_args`` arguments."""
    in_axes = (0,) * n_rows_args + (None,) * (len(args) - n_rows_args)
    return jax.vmap(fn, in_axes=in_axes)(*args)


def norm_forward(y: Array, x: Array, z: Array, d: Array, *, eps: float, dtype, interpret: bool = False,
                 z_row: int | None = None):
    """``a = rmsnorm((y + D x) silu(z))`` (no weight) in ``dtype`` and the RMS factor; ``y, x, z [R, H, P]``,
    ``d [H]``; returns ``a [R, H, P]`` and ``rstd [R]`` (float32). With ``z_row``, ``z [R, n, H, P]`` is a
    stacked tensor and the kernel reads its row ``z_row`` in place."""
    _, heads, head_dim = y.shape
    pow2 = _pow2(heads)
    kernel = (functools.partial(_fwd_kernel, eps=float(eps), z_row=z_row) if pow2 else
              functools.partial(_fwd_kernel_masked, eps=float(eps), heads=heads, head_dim=head_dim, z_row=z_row))
    call = pl.pallas_call(
        kernel, grid=(),
        out_shape=(jax.ShapeDtypeStruct((heads, head_dim) if pow2 else (heads * head_dim,), dtype),
                   jax.ShapeDtypeStruct((), f32)),
        compiler_params=None if interpret else _compiler_params(), interpret=interpret,
        name="mamba2_gated_norm_fwd")

    def body(y, x, z, d):
        if pow2:
            return _rows(call, 3, y, x, z, d)
        rows = y.shape[0]
        a, rstd = _rows(call, 3, y.reshape(rows, -1), x.reshape(rows, -1), z.reshape(rows, -1), d)
        return a.reshape(rows, heads, head_dim), rstd

    if _per_shard():  # several devices: the kernel on each device's rows
        return _rows_per_shard(body, functools.partial(_norm_forward_xla, eps=float(eps), dtype=dtype, z_row=z_row),
                               {"y": y, "x": x, "z": z, "d": d.astype(f32)})
    return body(y, x, z, d.astype(f32))


def norm_backward(da: Array, y: Array, x: Array, z: Array, d: Array, rstd: Array, *, interpret: bool = False,
                  z_row: int | None = None):
    """Backward of :func:`norm_forward` given ``da = d a``: ``dy`` (= the gradient of ``y + D x``) and ``dz`` in
    the input dtypes, and the per-row per-head sums of ``dy * x`` (``[R, H]`` float32; ``dD`` is their sum).
    With ``z_row`` (``z [R, n, H, P]``, as :func:`norm_forward`) ``dz`` is the gradient of that row, ``[R, H, P]``."""
    _, heads, head_dim = y.shape
    pow2 = _pow2(heads)
    kernel = (functools.partial(_bwd_kernel, z_row=z_row) if pow2 else
              functools.partial(_bwd_kernel_masked, heads=heads, head_dim=head_dim, z_row=z_row))
    row_shape = (heads, head_dim) if pow2 else (heads * head_dim,)
    call = pl.pallas_call(
        kernel, grid=(),
        out_shape=(jax.ShapeDtypeStruct(row_shape, y.dtype), jax.ShapeDtypeStruct(row_shape, z.dtype),
                   jax.ShapeDtypeStruct((heads,), f32)),
        compiler_params=None if interpret else _compiler_params(), interpret=interpret,
        name="mamba2_gated_norm_bwd")
    in_axes = (0, 0, 0, 0, None, 0)

    def body(da, y, x, z, d, rstd):
        if pow2:
            return jax.vmap(call, in_axes=in_axes)(da, y, x, z, d, rstd)
        rows = y.shape[0]
        dy, dz, dd = jax.vmap(call, in_axes=in_axes)(*(t.reshape(rows, -1) for t in (da, y, x, z)), d, rstd)
        return dy.reshape(rows, heads, head_dim), dz.reshape(rows, heads, head_dim), dd

    if _per_shard():  # several devices: the kernel on each device's rows
        return _rows_per_shard(body, functools.partial(_norm_backward_xla, z_row=z_row),
                               {"da": da, "y": y, "x": x, "z": z, "d": d.astype(f32), "rstd": rstd})
    return body(da, y, x, z, d.astype(f32), rstd)


# D skip + gate + RMSNorm + out-projection ------------------------------------------------------
def _out_weight(norm_w, out_w, dtype):
    """``w W``: the norm weight folded into the out-projection weight, in the compute dtype."""
    return (norm_w.astype(f32)[:, None] * out_w.astype(f32)).astype(dtype)


def _gno_forward(y, x, z, d, norm_w, out_w, eps, dtype, interpret):
    batch, length, heads, head_dim = y.shape
    rows = batch * length
    a, rstd = norm_forward(y.reshape(rows, heads, head_dim), x.reshape(rows, heads, head_dim),
                           z.reshape(rows, heads, head_dim), d, eps=eps, dtype=dtype, interpret=interpret)
    a = a.reshape(batch, length, heads * head_dim)
    out = lax.dot_general(a, _out_weight(norm_w, out_w, dtype), (((2,), (0,)), ((), ())))
    return out.astype(dtype), a, rstd


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _gated_norm_out(y, x, z, d, norm_w, out_w, eps, dtype, interpret):
    return _gno_forward(y, x, z, d, norm_w, out_w, eps, dtype, interpret)[0]


def _gated_norm_out_fwd(y, x, z, d, norm_w, out_w, eps, dtype, interpret):
    out, a, rstd = _gno_forward(y, x, z, d, norm_w, out_w, eps, dtype, interpret)
    return out, (y, x, z, d, norm_w, out_w, a, rstd)


def _gated_norm_out_bwd(eps, dtype, interpret, res, dout):
    del eps
    y, x, z, d, norm_w, out_w, a, rstd = res
    batch, length, heads, head_dim = y.shape
    rows = batch * length
    w_fold = _out_weight(norm_w, out_w, dtype)
    dout = dout.astype(dtype)
    da = lax.dot_general(dout, w_fold, (((2,), (1,)), ((), ())))  # [B, T, H * P]
    dw_fold = lax.dot_general(a, dout, (((0, 1), (0, 1)), ((), ())), preferred_element_type=f32)  # [H * P, d]
    d_out_w = (dw_fold * norm_w.astype(f32)[:, None]).astype(out_w.dtype)
    d_norm_w = jnp.sum(dw_fold * out_w.astype(f32), axis=1).astype(norm_w.dtype)
    shape3 = (rows, heads, head_dim)
    dy, dz, dd_rows = norm_backward(da.reshape(shape3), y.reshape(shape3), x.reshape(shape3), z.reshape(shape3), d,
                                    rstd, interpret=interpret)
    dy = dy.reshape(y.shape)
    dx = (dy.astype(f32) * d.astype(f32)[:, None]).astype(x.dtype)
    return dy, dx, dz.reshape(z.shape), jnp.sum(dd_rows, axis=0).astype(d.dtype), d_norm_w, d_out_w


_gated_norm_out.defvjp(_gated_norm_out_fwd, _gated_norm_out_bwd)


def gated_norm_out(y: Array, x: Array, z: Array, d: Array, norm_w: Array, out_w: Array, *, eps: float,
                   compute_dtype, interpret: bool = False) -> Array:
    """``out_proj(rmsnorm((y + D x) * silu(z)) * w)`` for one B/C group (one RMS factor per token).

    ``y, x [B, T, H, P]`` (SSD output and SSD input), ``z [B, T, H * P]``, ``d [H]``, ``norm_w [H * P]``,
    ``out_w [H * P, d_model]`` (no bias). Returns ``[B, T, d_model]`` in ``compute_dtype``; the backward pass
    keeps ``y, x, z``, the GEMM input ``a`` and the per-token RMS factor. ``interpret`` runs the Pallas kernels in
    the interpreter (CPU tests).
    """
    batch, length, heads, head_dim = y.shape
    width = heads * head_dim
    if x.shape != y.shape or z.shape != (batch, length, width) or d.shape != (heads,):
        raise ValueError(f"expected y, x [B, T, H, P], z [B, T, H * P], d [H]; got {y.shape}, {x.shape}, "
                         f"{z.shape}, {d.shape}")
    if norm_w.shape != (width,) or out_w.ndim != 2 or out_w.shape[0] != width:
        raise ValueError(f"expected norm_w [{width}] and out_w [{width}, d]; got {norm_w.shape}, {out_w.shape}")
    if not kernel_shape_ok(heads, head_dim):
        raise ValueError(f"gated_norm_out needs a power-of-two head_dim and at most 16384 padded channels; "
                         f"got {heads} x {head_dim}")
    out = _gated_norm_out(y, x, z, d, norm_w, out_w, float(eps), jnp.dtype(compute_dtype), bool(interpret))
    return checkpoint_name(out, LINEAR_CHECKPOINT_NAME)


# D skip + gate + RMSNorm, for an out-projection that keeps its own module (FP8) -----------------
def _z_rows(z, rows, heads, head_dim, z_row):
    """``z`` as kernel rows: ``[R, H, P]``, or ``[R, n, H, P]`` for a stacked ``z [B, T, n, H * P]`` (``z_row``)."""
    return z.reshape(rows, heads, head_dim) if z_row is None else z.reshape(rows, z.shape[2], heads, head_dim)


def _gn_forward(y, x, z, d, norm_w, eps, dtype, interpret, z_row=None):
    batch, length, heads, head_dim = y.shape
    rows = batch * length
    a, rstd = norm_forward(y.reshape(rows, heads, head_dim), x.reshape(rows, heads, head_dim),
                           _z_rows(z, rows, heads, head_dim, z_row), d, eps=eps, dtype=dtype, interpret=interpret,
                           z_row=z_row)
    a = a.reshape(batch, length, heads * head_dim)
    return (a.astype(f32) * norm_w.astype(f32)).astype(dtype), a, rstd


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _gated_norm(y, x, z, d, norm_w, eps, dtype, interpret, z_row):
    return _gn_forward(y, x, z, d, norm_w, eps, dtype, interpret, z_row)[0]


def _gated_norm_fwd(y, x, z, d, norm_w, eps, dtype, interpret, z_row):
    out, a, rstd = _gn_forward(y, x, z, d, norm_w, eps, dtype, interpret, z_row)
    return out, (y, x, z, d, norm_w, a, rstd)


def _gated_norm_bwd(eps, dtype, interpret, z_row, res, dout):
    del eps, dtype
    y, x, z, d, norm_w, a, rstd = res
    batch, length, heads, head_dim = y.shape
    shape3 = (batch * length, heads, head_dim)
    dout32 = dout.astype(f32)
    d_norm_w = jnp.sum(dout32 * a.astype(f32), axis=(0, 1)).astype(norm_w.dtype)
    da = (dout32 * norm_w.astype(f32)).astype(dout.dtype)
    dy, dz, dd_rows = norm_backward(da.reshape(shape3), y.reshape(shape3), x.reshape(shape3),
                                    _z_rows(z, shape3[0], heads, head_dim, z_row), d, rstd, interpret=interpret,
                                    z_row=z_row)
    dy = dy.reshape(y.shape)
    dx = (dy.astype(f32) * d.astype(f32)[:, None]).astype(x.dtype)
    dz = dz.reshape(batch, length, heads * head_dim)
    if z_row is not None:  # zeros in the other rows: XLA adds them to those rows' own gradients in one fusion
        dz = jnp.stack([dz if i == z_row else jnp.zeros_like(dz) for i in range(z.shape[2])], axis=2)
    return dy, dx, dz, jnp.sum(dd_rows, axis=0).astype(d.dtype), d_norm_w


_gated_norm.defvjp(_gated_norm_fwd, _gated_norm_bwd)


def gated_norm(y: Array, x: Array, z: Array, d: Array, norm_w: Array, *, eps: float, compute_dtype,
               interpret: bool = False, z_row: int | None = None) -> Array:
    """``rmsnorm((y + D x) * silu(z)) * w`` for one B/C group, ``[B, T, H * P]`` in ``compute_dtype``: the
    out-projection's input when the out-projection keeps its own module (an FP8 out-projection). The row kernels
    of :func:`gated_norm_out`; the norm weight's gradient is a column reduction of ``dout * a`` (two bfloat16
    reads) instead of :func:`gated_norm_out`'s weight-gradient GEMM.

    With ``z_row``, ``z [B, T, n, H * P]`` is a stacked tensor (the z|x rows of one stacked projection GEMM) whose
    row ``z_row`` is the gate, read in place by the kernels; the gradient of ``z`` has zeros in the other rows."""
    batch, length, heads, head_dim = y.shape
    width = heads * head_dim
    if z_row is not None and (z.ndim != 4 or not 0 <= z_row < z.shape[2]):
        raise ValueError(f"z_row {z_row} needs a stacked z [B, T, n, H * P] with 0 <= z_row < n; got {z.shape}")
    z_shape = (batch, length, width) if z_row is None else (batch, length, z.shape[2], width)
    if x.shape != y.shape or z.shape != z_shape or d.shape != (heads,) or norm_w.shape != (width,):
        raise ValueError(f"expected y, x [B, T, H, P], z [B, T, H * P], d [H], norm_w [H * P]; got {y.shape}, "
                         f"{x.shape}, {z.shape}, {d.shape}, {norm_w.shape}")
    if not kernel_shape_ok(heads, head_dim):
        raise ValueError(f"gated_norm needs a power-of-two head_dim and at most 16384 padded channels; "
                         f"got {heads} x {head_dim}")
    return _gated_norm(y, x, z, d, norm_w, float(eps), jnp.dtype(compute_dtype), bool(interpret),
                       None if z_row is None else int(z_row))
