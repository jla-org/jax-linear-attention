# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MaxText (Apache-2.0), src/maxtext/layers/quantizations.py (the GPU "fp8" recipe:
# Fp8DirectDotGeneralOp and its overwrite-with-gradient scaling state), and Flax (Apache-2.0),
# flax/linen/fp8_ops.py (delayed per-tensor scaling from an amax history).
"""FP8 training matmuls for Hopper-class GPUs: e4m3 inputs and weights, e5m2 gradients.

Every FP8 matmul quantizes its operands per tensor, ``q = clip(x / scale, -max, max)`` cast to
the FP8 format, multiplies the FP8 tensors with float32 accumulation and multiplies the result
by the product of the two scales. Inputs and weights use ``float8_e4m3fn`` (max 448); the
gradient of the output uses ``float8_e5m2`` (max 57344) and feeds both backward GEMMs
(``dx = g W^T`` and ``dW = x^T g``). On Hopper XLA runs these FP8 dots as cuBLASLt FP8 GEMMs
(Triton FP8 GEMM fusions for small shapes); elsewhere it upcasts them, which computes the same
quantized products (CPU emulation).

Two per-tensor scaling recipes:

* ``"delayed"`` (Transformer Engine's default and MaxText's GPU ``fp8``): the scale of each
  tensor is ``max(amax history) / fp8_max``, where the history holds the absolute maxima of
  that tensor in the previous ``amax_history`` optimizer steps (newest first). The scales and
  histories are an :class:`Fp8State` per matmul, stored in the model. The matmul's VJP returns
  their new values as their "gradient"; :mod:`jla.training.step` writes those values back
  instead of giving them to the optimizer, so the state advances once per applied step, is
  saved with the parameters and survives checkpoint/resume. Until a tensor has a recorded amax
  its scale stays at 1.
* ``"current"``: each scale comes from the amax of the tensor being quantized (one extra
  reduction per operand and no state).

:func:`apply_fp8` converts a built model: every :class:`~jla.layers.linear.Linear`,
:class:`~jla.layers.linear.StackedLinear` and the gate/up GEMM of :class:`~jla.layers.mlp.SwiGLU`
whose input and output widths are multiples of 16 (the cuBLASLt FP8 alignment) becomes its FP8
counterpart with the same parameters. Embeddings, the LM head, MoE experts and routers, and the
mixers' recurrences and attention keep the compute dtype.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jax.ad_checkpoint import checkpoint_name

from jla.layers.linear import LINEAR_CHECKPOINT_NAME, Linear, StackedLinear
from jla.layers.mlp import MLP_HIDDEN_CHECKPOINT_NAME, SwiGLU, swiglu

__all__ = [
    "E4M3",
    "E5M2",
    "FP8_RECIPES",
    "Fp8Linear",
    "Fp8StackedLinear",
    "Fp8State",
    "Fp8SwiGLU",
    "apply_fp8",
    "Fp8Weight",
    "fp8_dot",
    "fp8_dot_prepared",
    "fp8_modules",
    "prepare_fp8_weight",
    "shared_input_projections",
    "push_amax",
    "quantize",
    "scale_from_amax",
]

Array = jax.Array
Axes = tuple[str | None, ...]

E4M3 = jnp.float8_e4m3fn
E5M2 = jnp.float8_e5m2
FP8_RECIPES = ("delayed", "current")
_ALIGN = 16  # cuBLASLt FP8 GEMMs need K and N to be multiples of 16


def _fp8_max(dtype: Any) -> float:
    return float(jnp.finfo(dtype).max)


def _amax(x: Array) -> Array:
    return jnp.max(jnp.abs(x)).astype(jnp.float32)


def scale_from_amax(amax: Array, fp8_max: float, previous: Array | float) -> Array:
    """Dequantization scale ``amax / fp8_max``; ``previous`` where ``amax`` is zero or not finite,
    and 1 where ``previous`` is not a positive finite number either (a scale is never 0)."""
    amax = jnp.asarray(amax, jnp.float32)
    previous = jnp.asarray(previous, jnp.float32)
    previous = jnp.where(jnp.isfinite(previous) & (previous > 0), previous, jnp.float32(1.0))
    ok = jnp.isfinite(amax) & (amax > 0)
    return jnp.where(ok, amax / fp8_max, previous).astype(jnp.float32)


def quantize(x: Array, scale: Array, dtype: Any) -> Array:
    """``clip(x / scale)`` cast to the FP8 ``dtype`` (saturating; the division runs in float32)."""
    bound = _fp8_max(dtype)
    return jnp.clip(x.astype(jnp.float32) / scale, -bound, bound).astype(dtype)


def push_amax(history: Array, amax: Array) -> Array:
    """Shift the history one step (dropping the oldest entry) and put ``amax`` first.

    The optimization barrier materializes ``amax`` first: fused into the small concatenate (and into the train
    step's max over micro-steps), XLA recomputed the whole amax reduction for every entry of the history (75 ms
    of a 1.2 s ``transformer-1p3b`` step at grad_accum 4, benchmarks/fp8_1p3b.md). Same values.
    """
    amax = lax.optimization_barrier(jnp.reshape(amax, (1,)).astype(history.dtype))
    return jnp.concatenate([amax, history[:-1]])


class Fp8State(eqx.Module):
    """Delayed-scaling state of one FP8 matmul (float32; leading dims for stacked layers)."""

    input_scale: Array
    input_amax_history: Array
    kernel_scale: Array
    kernel_amax_history: Array
    grad_scale: Array
    grad_amax_history: Array

    @classmethod
    def init(cls, history: int, prefix: tuple[int, ...] = ()) -> Fp8State:
        if history < 1:
            raise ValueError(f"the amax history needs at least one entry; got {history}")
        ones = jnp.ones(prefix, jnp.float32)
        zeros = jnp.zeros((*prefix, history), jnp.float32)
        return cls(ones, zeros, ones, zeros, ones, zeros)

    def param_axes(self) -> dict[str, Axes]:
        return {
            "input_scale": (),
            "input_amax_history": (None,),
            "kernel_scale": (),
            "kernel_amax_history": (None,),
            "grad_scale": (),
            "grad_amax_history": (None,),
        }


# Dimension numbers of x [..., K] @ w [K, *R] and of its two backward GEMMs.
def _forward_dims(nb: int) -> Any:
    return (((nb,), (0,)), ((), ()))


def _dgrad_dims(nb: int, nr: int) -> Any:  # g [..., *R] . w [K, *R] -> [..., K]
    return ((tuple(range(nb, nb + nr)), tuple(range(1, 1 + nr))), ((), ()))


def _wgrad_dims(nb: int) -> Any:  # x [..., K] . g [..., *R] -> [K, *R]
    return ((tuple(range(nb)), tuple(range(nb))), ((), ()))


def _scaled_dot(a: Array, a_scale: Array, b: Array, b_scale: Array, dims: Any, out_dtype: Any, precision: Any) -> Array:
    """``(a * a_scale) . (b * b_scale)`` for FP8 ``a`` and ``b``: an FP8 dot (float32 accumulation
    inside the GEMM) that writes ``out_dtype``, times ``a_scale * b_scale`` in ``out_dtype``.

    This is Flax's ``Fp8DirectDotGeneralOp`` form. XLA turns it into one cuBLASLt FP8 GEMM per dot
    with a bfloat16 (or float32) output; asking the GEMM for float32 and casting afterwards writes
    twice the bytes, and dequantize-then-dot is not matched as an FP8 GEMM at all (on an H100 with
    JAX 0.11.2, forward plus both backward GEMMs at 16384 x 2048 x 8192 run at 1.38x, 1.1x and
    0.2x the speed of bfloat16; see benchmarks/fp8.md).
    """
    out = lax.dot_general(a, b, dims, precision=precision, preferred_element_type=out_dtype)
    return out * (a_scale * b_scale).astype(out_dtype)


# Forward GEMMs keep cuBLASLt's fast accumulation; the backward GEMMs (e5m2 gradients) turn it off,
# as Flax's fp8 ops do.
_FWD_PRECISION = lax.Precision.DEFAULT
_BWD_PRECISION = lax.Precision.HIGHEST


def _backward(qx: Array, qw: Array, x_scale: Array, w_scale: Array, g: Array, g_scale: Array, dtypes: tuple[str, str]):
    x_dtype, w_dtype = dtypes
    nb, nr = qx.ndim - 1, qw.ndim - 1
    qg = quantize(g, g_scale, E5M2)
    dx = _scaled_dot(qg, g_scale, qw, w_scale, _dgrad_dims(nb, nr), x_dtype, _BWD_PRECISION)
    dw = _scaled_dot(qx, x_scale, qg, g_scale, _wgrad_dims(nb), w_dtype, _BWD_PRECISION)
    return dx, dw


# -- delayed scaling --------------------------------------------------------------------


def _delayed_forward(x: Array, w: Array, state: Fp8State, dtypes: tuple[str, str]):
    e4m3_max = _fp8_max(E4M3)
    x_scale = scale_from_amax(jnp.max(state.input_amax_history), e4m3_max, state.input_scale)
    w_scale = scale_from_amax(jnp.max(state.kernel_amax_history), e4m3_max, state.kernel_scale)
    qx, qw = quantize(x, x_scale, E4M3), quantize(w, w_scale, E4M3)
    y = _scaled_dot(qx, x_scale, qw, w_scale, _forward_dims(x.ndim - 1), dtypes[0], _FWD_PRECISION)
    return y, (qx, qw, x_scale, w_scale)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _fp8_dot_delayed(x: Array, w: Array, state: Fp8State, dtypes: tuple[str, str]) -> Array:
    return _delayed_forward(x, w, state, dtypes)[0]


def _fp8_dot_delayed_fwd(x, w, state, dtypes):
    y, (qx, qw, x_scale, w_scale) = _delayed_forward(x, w, state, dtypes)
    return y, (qx, qw, x_scale, w_scale, _amax(x), _amax(w), state)


def _fp8_dot_delayed_bwd(dtypes, residuals, g):
    qx, qw, x_scale, w_scale, x_amax, w_amax, state = residuals
    g_scale = scale_from_amax(jnp.max(state.grad_amax_history), _fp8_max(E5M2), state.grad_scale)
    dx, dw = _backward(qx, qw, x_scale, w_scale, g, g_scale, dtypes)
    # The "gradient" of the state is its new value (see the module docstring).
    new_state = Fp8State(
        input_scale=x_scale,
        input_amax_history=push_amax(state.input_amax_history, x_amax),
        kernel_scale=w_scale,
        kernel_amax_history=push_amax(state.kernel_amax_history, w_amax),
        grad_scale=g_scale,
        grad_amax_history=push_amax(state.grad_amax_history, _amax(g)),
    )
    return dx, dw, new_state


_fp8_dot_delayed.defvjp(_fp8_dot_delayed_fwd, _fp8_dot_delayed_bwd)


# -- current scaling --------------------------------------------------------------------


def _current_forward(x: Array, w: Array, dtypes: tuple[str, str]):
    e4m3_max = _fp8_max(E4M3)
    x_scale = scale_from_amax(_amax(x), e4m3_max, 1.0)
    w_scale = scale_from_amax(_amax(w), e4m3_max, 1.0)
    qx, qw = quantize(x, x_scale, E4M3), quantize(w, w_scale, E4M3)
    y = _scaled_dot(qx, x_scale, qw, w_scale, _forward_dims(x.ndim - 1), dtypes[0], _FWD_PRECISION)
    return y, (qx, qw, x_scale, w_scale)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _fp8_dot_current(x: Array, w: Array, dtypes: tuple[str, str]) -> Array:
    return _current_forward(x, w, dtypes)[0]


def _fp8_dot_current_fwd(x, w, dtypes):
    return _current_forward(x, w, dtypes)


def _fp8_dot_current_bwd(dtypes, residuals, g):
    qx, qw, x_scale, w_scale = residuals
    g_scale = scale_from_amax(_amax(g), _fp8_max(E5M2), 1.0)
    return _backward(qx, qw, x_scale, w_scale, g, g_scale, dtypes)


_fp8_dot_current.defvjp(_fp8_dot_current_fwd, _fp8_dot_current_bwd)


def fp8_dot(x: Array, w: Array, state: Fp8State | None, *, recipe: str) -> Array:
    """``x [..., K] @ w [K, *R] -> [..., *R]`` in FP8; the result has ``x``'s dtype.

    ``recipe="delayed"`` needs the matmul's ``state``; ``"current"`` takes ``state=None``.
    """
    if x.shape[-1] != w.shape[0]:
        raise ValueError(f"fp8_dot contracts x [..., {x.shape[-1]}] with w [{w.shape[0]}, ...]")
    if jnp.dtype(x.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        # Scales such as amax / 57344 underflow float16, so FP8 dequantizes in bfloat16 or float32.
        raise TypeError(f"FP8 matmuls need a bfloat16 or float32 compute dtype; got {x.dtype}")
    dtypes = (str(x.dtype), str(w.dtype))
    if recipe == "delayed":
        if state is None:
            raise ValueError("delayed FP8 scaling needs the matmul's Fp8State")
        return _fp8_dot_delayed(x, w, state, dtypes)
    if recipe == "current":
        if state is not None:
            raise ValueError("current FP8 scaling keeps no state")
        return _fp8_dot_current(x, w, dtypes)
    raise ValueError(f"unknown FP8 recipe {recipe!r}; expected one of {FP8_RECIPES}")


# -- modules ----------------------------------------------------------------------------


class Fp8Linear(eqx.Module):
    """:class:`~jla.layers.linear.Linear` with an FP8 matmul (same parameters and axes)."""

    weight: Array
    bias: Array | None
    state: Fp8State | None
    compute_dtype: str = eqx.field(static=True)
    axes: Axes = eqx.field(static=True)
    recipe: str = eqx.field(static=True)
    # Delayed scaling: the train step quantizes the weight once per optimizer step (prepare_fp8_weight) and
    # attaches the FP8 copy here for its micro-steps; None (stored models, eval, inference) quantizes per call.
    prepared: Any = None
    weight_once: bool = eqx.field(static=True, default=True)  # ModelConfig.fp8_weight_once

    def __call__(self, x: Array, shared_input: Any = None) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        if self.prepared is not None:  # shared_input: see shared_input_projections
            y = fp8_dot_prepared(x.astype(dtype), self.weight, self.prepared, self.state, shared_input=shared_input)
        else:
            y = fp8_dot(x.astype(dtype), self.weight, self.state, recipe=self.recipe)
        if self.bias is not None:
            y = y + self.bias.astype(dtype)
        # The same checkpoint name as Linear, so the "names" remat policies keep FP8 projections too.
        return checkpoint_name(y, LINEAR_CHECKPOINT_NAME)

    def param_axes(self) -> dict[str, Axes]:
        return {"weight": self.axes, "bias": (self.axes[1],)}


class Fp8StackedLinear(eqx.Module):
    """:class:`~jla.layers.linear.StackedLinear` with an FP8 matmul: ``x [..., in] -> [..., n, out]``."""

    weight: Array
    state: Fp8State | None
    compute_dtype: str = eqx.field(static=True)
    axes: Axes = eqx.field(static=True)
    recipe: str = eqx.field(static=True)
    prepared: Any = None  # as Fp8Linear.prepared
    weight_once: bool = eqx.field(static=True, default=True)

    def __call__(self, x: Array, shared_input: Any = None) -> Array:
        x = x.astype(jnp.dtype(self.compute_dtype))
        if self.prepared is not None:
            y = fp8_dot_prepared(x, self.weight, self.prepared, self.state, shared_input=shared_input)
        else:
            y = fp8_dot(x, self.weight, self.state, recipe=self.recipe)
        return checkpoint_name(y, LINEAR_CHECKPOINT_NAME)  # as StackedLinear

    def param_axes(self) -> dict[str, Axes]:
        return {"weight": self.axes}


class Fp8SwiGLU(eqx.Module):
    """:class:`~jla.layers.mlp.SwiGLU` with FP8 gate/up and down GEMMs."""

    gate_up: Array  # [D, 2, F]
    down: Any  # Fp8Linear (or Linear when its widths are not FP8-aligned)
    state: Fp8State | None
    compute_dtype: str = eqx.field(static=True)
    recipe: str = eqx.field(static=True)
    # Delayed scaling only: run gate/up GEMM, SwiGLU and down GEMM as jla.layers.fp8_swiglu.fp8_swiglu_mlp
    # (fused activation + quantize + transpose passes; same state semantics, and in a jitted step on a
    # GPU the same values as the unfused path up to GEMM accumulation order; see jla.layers.fp8_swiglu).
    fused: bool = eqx.field(static=True, default=False)
    prepared: Any = None  # the gate/up weight's, as Fp8Linear.prepared (the down projection has its own)
    weight_once: bool = eqx.field(static=True, default=True)

    def __call__(self, x: Array) -> Array:
        if self.fused and isinstance(self.down, Fp8Linear) and self.down.bias is None:
            from jla.layers.fp8_swiglu import fp8_swiglu_mlp

            both = self.prepared is not None and self.down.prepared is not None
            return fp8_swiglu_mlp(x.astype(jnp.dtype(self.compute_dtype)), self.gate_up, self.down.weight,
                                  self.state, self.down.state,
                                  prepared=(self.prepared, self.down.prepared) if both else None)
        x = x.astype(jnp.dtype(self.compute_dtype))
        if self.prepared is not None:
            h = fp8_dot_prepared(x, self.gate_up, self.prepared, self.state)
        else:
            h = fp8_dot(x, self.gate_up, self.state, recipe=self.recipe)
        return self.down(swiglu(checkpoint_name(h, MLP_HIDDEN_CHECKPOINT_NAME)))  # as SwiGLU

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"gate_up": ("embed", None, "mlp")}


def _aligned(weight: Array, base_ndim: int, min_features: int = _ALIGN) -> bool:
    """The GEMM ``K`` and ``N`` of ``weight`` are multiples of 16 and both at least ``min_features``."""
    lead = weight.ndim - base_ndim
    k = weight.shape[lead]
    n = math.prod(weight.shape[lead + 1:])
    return k % _ALIGN == 0 and n % _ALIGN == 0 and min(k, n) >= min_features


def _state(weight: Array, base_ndim: int, recipe: str, history: int) -> Fp8State | None:
    if recipe == "current":
        return None
    return Fp8State.init(history, tuple(weight.shape[: weight.ndim - base_ndim]))


def apply_fp8(tree: Any, recipe: str, *, amax_history: int = 1024, min_features: int = _ALIGN,
              fused: bool = False, weight_once: bool = True) -> Any:
    """Replace the FP8-eligible projections of ``tree`` (a model or part of one) by FP8 modules.

    A projection is eligible when its GEMM's input and output widths are multiples of 16 and both
    at least ``min_features`` (the default 16 converts every aligned projection; a larger value
    keeps small GEMMs such as gates and low-rank factors in the compute dtype).
    ``fused`` (delayed scaling only) runs every SwiGLU MLP whose two GEMMs are FP8 through
    :func:`jla.layers.fp8_swiglu.fp8_swiglu_mlp`.
    Weights may carry leading stacked-layer dims (``scan_layers``); the delayed-scaling state
    gets the same leading dims. Parameters are reused unchanged.
    ``weight_once`` (delayed scaling): the train step quantizes every weight once per optimizer step
    (:func:`prepare_fp8_weight`) instead of in every forward call; same values.
    """
    if recipe not in FP8_RECIPES:
        raise ValueError(f"unknown FP8 recipe {recipe!r}; expected one of {FP8_RECIPES}")
    if min_features < 1:
        raise ValueError(f"min_features must be >= 1; got {min_features}")
    if fused and recipe != "delayed":
        raise ValueError("the fused FP8 SwiGLU passes need delayed scaling (the scales must be known before the pass)")

    def linear(node: Linear) -> Any:
        if not _aligned(node.weight, 2, min_features):
            return node
        return Fp8Linear(node.weight, node.bias, _state(node.weight, 2, recipe, amax_history),
                         node.compute_dtype, node.axes, recipe, weight_once=weight_once)

    def convert(node: Any) -> Any:
        kind = type(node)
        if kind is Linear:
            return linear(node)
        if kind is StackedLinear and _aligned(node.weight, 3, min_features):
            return Fp8StackedLinear(node.weight, _state(node.weight, 3, recipe, amax_history),
                                    node.compute_dtype, node.axes, recipe, weight_once=weight_once)
        if kind is SwiGLU:
            down = linear(node.down)
            if not _aligned(node.gate_up, 3, min_features):
                return eqx.tree_at(lambda m: m.down, node, down)
            use_fused = fused and isinstance(down, Fp8Linear) and down.bias is None
            # Deliberately per MLP and logged, not an error: fp8_fused is a model-wide switch, and the MLPs it
            # cannot fuse (a biased or unaligned down projection) keep the same FP8 recipe on the unfused path
            # (tests/test_fp8_swiglu.py::test_fused_fallbacks_are_logged).
            if fused and not use_fused:
                logging.getLogger(__name__).warning(
                    "fp8_fused: the SwiGLU MLP with gate/up weight %s keeps the unfused FP8 path (its down "
                    "projection is not FP8-aligned or has a bias)", tuple(node.gate_up.shape))
            return Fp8SwiGLU(node.gate_up, down, _state(node.gate_up, 3, recipe, amax_history),
                             node.compute_dtype, recipe, use_fused, weight_once=weight_once)
        return node

    return jax.tree_util.tree_map(convert, tree, is_leaf=lambda node: type(node) in (Linear, StackedLinear, SwiGLU))


def fp8_modules(tree: Any) -> list[Any]:
    """The FP8 modules inside ``tree`` (for tests and logging)."""
    kinds = (Fp8Linear, Fp8StackedLinear, Fp8SwiGLU)
    found: list[Any] = []

    def visit(node: Any) -> Any:
        if isinstance(node, kinds):
            found.append(node)
            if isinstance(node, Fp8SwiGLU) and isinstance(node.down, Fp8Linear):
                found.append(node.down)
        return node

    jax.tree_util.tree_map(visit, tree, is_leaf=lambda node: isinstance(node, kinds))
    return found


# -- weights quantized once per optimizer step ---------------------------------------------------


class Fp8Weight(eqx.Module):
    """One optimizer step's FP8 copy of a delayed-scaling weight ``w [K, *R]`` (:func:`prepare_fp8_weight`).

    ``q = quantize(w, scale, E4M3)`` with the matmul's delayed kernel scale (the operand of the input-gradient
    GEMM), ``qt`` the same values with ``K`` last, ``[*R, K]`` (the forward GEMM's operand: cuBLASLt FP8 GEMMs
    are TN only, so this transposed copy was otherwise made in every forward call), and ``amax`` the weight's
    absolute maximum, which the kernel amax history records. Stacked layers add the same leading dims to all three.
    ``qt`` is None on several devices (``prepare_fp8_weight(transposed=False)``): a sharded (FSDP) weight's
    copies are gathered where the GEMMs use them, and gathering one e4m3 copy and transposing it locally
    moves half the bytes of gathering both layouts. ``axes`` are the weight's logical axes; ``param_axes``
    gives each copy's, so a model carrying the copies (the pipeline-parallel forward lays out its block
    parameters by :func:`jla.models.logical_axes`) shards them like the weight.
    """

    q: Array
    qt: Array | None
    amax: Array
    axes: Axes = eqx.field(static=True)

    def param_axes(self) -> dict[str, Axes]:
        k, *rest = self.axes
        return {"q": self.axes, "qt": (*rest, k), "amax": ()}


def _forward_operand(wq: "Fp8Weight", ndim: int) -> tuple[Array, Any]:
    """The forward GEMM's weight operand and dimension numbers for ``x [..., K]`` (``ndim`` dims)."""
    if wq.qt is None:  # x [..., K] . q [K, *R]
        return wq.q, _forward_dims(ndim - 1)
    return wq.qt, (((ndim - 1,), (wq.qt.ndim - 1,)), ((), ()))  # x [..., K] . qt [*R, K]


def _prepare(w: Array, state: Fp8State, base_ndim: int, axes: Axes, transposed: bool) -> Fp8Weight:
    lead = w.ndim - base_ndim
    scale = scale_from_amax(jnp.max(state.kernel_amax_history, axis=-1), _fp8_max(E4M3), state.kernel_scale)
    q = quantize(w, jnp.reshape(scale, scale.shape + (1,) * base_ndim), E4M3)
    qt = jnp.transpose(q, (*range(lead), *range(lead + 1, w.ndim), lead)) if transposed else None
    amax = jnp.max(jnp.abs(w), axis=tuple(range(lead, w.ndim))).astype(jnp.float32)
    return Fp8Weight(q, qt, amax, tuple(axes))


def prepare_fp8_weight(module: Any, *, transposed: bool | None = None) -> Any:
    """The FP8 weight copies of one FP8 module for an optimizer step, or None where it quantizes per call.

    :class:`Fp8Linear` / :class:`Fp8StackedLinear`: an :class:`Fp8Weight`; :class:`Fp8SwiGLU`: a pair (gate/up,
    down), either entry None. Only delayed scaling with ``weight_once`` qualifies (current scaling has no state).
    ``transposed``: also make the transposed copy ``qt`` (for weights on one device; None: when the process
    sees one device). Either way the same values, gradients and states.
    """
    if transposed is None:
        transposed = jax.device_count() == 1

    def one(weight: Array, state: Any, base_ndim: int, recipe: str, once: bool, axes: Axes) -> Fp8Weight | None:
        if recipe != "delayed" or not once or not isinstance(state, Fp8State):
            return None
        return _prepare(weight, state, base_ndim, axes, transposed)

    if isinstance(module, Fp8SwiGLU):
        down = module.down
        dn = (one(down.weight, down.state, 2, down.recipe, down.weight_once, down.axes)
              if isinstance(down, Fp8Linear) else None)
        gate_up = one(module.gate_up, module.state, 3, module.recipe, module.weight_once,
                      module.param_axes()["gate_up"])
        return gate_up, dn
    if isinstance(module, Fp8Linear):
        return one(module.weight, module.state, 2, module.recipe, module.weight_once, module.axes)
    if isinstance(module, Fp8StackedLinear):
        return one(module.weight, module.state, 3, module.recipe, module.weight_once, module.axes)
    raise TypeError(f"not an FP8 module: {type(module).__name__}")


def attach_fp8_weight(module: Any, prepared: Any) -> Any:
    """``module`` with the :func:`prepare_fp8_weight` copies attached (None entries leave a weight per call)."""
    import dataclasses

    if isinstance(module, Fp8SwiGLU):
        gu, dn = prepared
        down = dataclasses.replace(module.down, prepared=dn) if dn is not None else module.down
        return dataclasses.replace(module, prepared=gu, down=down)
    return dataclasses.replace(module, prepared=prepared) if prepared is not None else module


def _prepared_forward(x: Array, wq: Fp8Weight, state: Fp8State, dtypes: tuple[str, str]):
    e4m3_max = _fp8_max(E4M3)
    x_scale = scale_from_amax(jnp.max(state.input_amax_history), e4m3_max, state.input_scale)
    w_scale = scale_from_amax(jnp.max(state.kernel_amax_history), e4m3_max, state.kernel_scale)
    qx = quantize(x, x_scale, E4M3)
    operand, dims = _forward_operand(wq, x.ndim)
    y = _scaled_dot(qx, x_scale, operand, w_scale, dims, dtypes[0], _FWD_PRECISION)
    return y, (qx, x_scale, w_scale)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _fp8_dot_prepared(x: Array, w: Array, wq: Fp8Weight, state: Fp8State, dtypes: tuple[str, str]) -> Array:
    return _prepared_forward(x, wq, state, dtypes)[0]


def _fp8_dot_prepared_fwd(x, w, wq, state, dtypes):
    y, (qx, x_scale, w_scale) = _prepared_forward(x, wq, state, dtypes)
    return y, (qx, wq, x_scale, w_scale, _amax(x), state)


def _fp8_dot_prepared_bwd(dtypes, residuals, g):
    qx, wq, x_scale, w_scale, x_amax, state = residuals
    g_scale = scale_from_amax(jnp.max(state.grad_amax_history), _fp8_max(E5M2), state.grad_scale)
    dx, dw = _backward(qx, wq.q, x_scale, w_scale, g, g_scale, dtypes)
    new_state = Fp8State(  # exactly _fp8_dot_delayed_bwd's, with the weight amax of the step's copy
        input_scale=x_scale,
        input_amax_history=push_amax(state.input_amax_history, x_amax),
        kernel_scale=w_scale,
        kernel_amax_history=push_amax(state.kernel_amax_history, wq.amax),
        grad_scale=g_scale,
        grad_amax_history=push_amax(state.grad_amax_history, _amax(g)),
    )
    return dx, dw, jax.tree.map(jnp.zeros_like, wq), new_state


_fp8_dot_prepared.defvjp(_fp8_dot_prepared_fwd, _fp8_dot_prepared_bwd)


def fp8_dot_prepared(x: Array, w: Array, wq: Fp8Weight, state: Fp8State, *, shared_input: Any = None) -> Array:
    """:func:`fp8_dot` (delayed scaling) with the weight's FP8 copy ``wq`` made once per optimizer step.

    ``w`` only receives the weight gradient (the forward and input-gradient GEMMs read ``wq``); the values,
    the gradients and the new state are those of ``fp8_dot(x, w, state, recipe="delayed")`` when ``wq`` is
    :func:`prepare_fp8_weight` of the same ``w`` and ``state``. ``shared_input``: ``x`` already quantized,
    ``(qx, scale, amax)`` from :func:`shared_input_projections`.
    """
    if x.shape[-1] != w.shape[0] or wq.q.shape != w.shape:
        raise ValueError(f"fp8_dot_prepared contracts x [..., {x.shape[-1]}] with w {w.shape} (copy {wq.q.shape})")
    if jnp.dtype(x.dtype) not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise TypeError(f"FP8 matmuls need a bfloat16 or float32 compute dtype; got {x.dtype}")
    if not isinstance(state, Fp8State):
        raise ValueError("fp8_dot_prepared needs the matmul's delayed-scaling Fp8State")
    if shared_input is not None:
        qx, x_scale, x_amax = shared_input
        if qx.shape != x.shape:
            raise ValueError(f"the shared FP8 input {qx.shape} does not match x {x.shape}")
        return _fp8_dot_shared(x, w, wq, state, qx, x_scale, x_amax, (str(x.dtype), str(w.dtype)))
    return _fp8_dot_prepared(x, w, wq, state, (str(x.dtype), str(w.dtype)))


# -- one quantization of an input that several projections read ---------------------------------


def _shared_forward(qx: Array, x_scale: Array, wq: Fp8Weight, state: Fp8State, dtypes: tuple[str, str]):
    w_scale = scale_from_amax(jnp.max(state.kernel_amax_history), _fp8_max(E4M3), state.kernel_scale)
    operand, dims = _forward_operand(wq, qx.ndim)
    return _scaled_dot(qx, x_scale, operand, w_scale, dims, dtypes[0], _FWD_PRECISION), w_scale


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _fp8_dot_shared(x, w, wq, state, qx, x_scale, x_amax, dtypes):
    return _shared_forward(qx, x_scale, wq, state, dtypes)[0]


def _fp8_dot_shared_fwd(x, w, wq, state, qx, x_scale, x_amax, dtypes):
    y, w_scale = _shared_forward(qx, x_scale, wq, state, dtypes)
    return y, (qx, wq, x_scale, w_scale, x_amax, state)


def _fp8_dot_shared_bwd(dtypes, residuals, g):
    qx, wq, x_scale, w_scale, x_amax, state = residuals
    e4m3 = _fp8_max(E4M3)
    g_scale = scale_from_amax(jnp.max(state.grad_amax_history), _fp8_max(E5M2), state.grad_scale)
    dx, dw = _backward(qx, wq.q, x_scale, w_scale, g, g_scale, dtypes)
    new_state = Fp8State(  # _fp8_dot_prepared_bwd's; the input scale this matmul's own history gives (see below)
        input_scale=scale_from_amax(jnp.max(state.input_amax_history), e4m3, state.input_scale),
        input_amax_history=push_amax(state.input_amax_history, x_amax),
        kernel_scale=w_scale,
        kernel_amax_history=push_amax(state.kernel_amax_history, wq.amax),
        grad_scale=g_scale,
        grad_amax_history=push_amax(state.grad_amax_history, _amax(g)),
    )
    zeros = functools.partial(jax.tree.map, jnp.zeros_like)
    return dx, dw, zeros(wq), new_state, zeros(qx), zeros(x_scale), zeros(x_amax)


_fp8_dot_shared.defvjp(_fp8_dot_shared_fwd, _fp8_dot_shared_bwd)


def _shares_input(module: Any) -> bool:
    return (type(module) in (Fp8Linear, Fp8StackedLinear) and module.prepared is not None
            and isinstance(module.state, Fp8State))


def shared_input_projections(x: Array, *projections: Any) -> tuple[Any, ...]:
    """``tuple(p(x) for p in projections)``, with one FP8 quantization of ``x`` for all of them.

    The delayed-scaling FP8 projections among them that carry the train step's weight copies
    (:func:`prepare_fp8_weight`) read one e4m3 copy of ``x`` (and the weight-gradient GEMMs one transposed
    copy of it) instead of one each: the attention q/k/v projections, the linear-attention q/k/v/output-gate
    projections. Their input scale is the largest of their delayed input scales. Every projection records the
    amax of the same ``x`` in its input history every step, so these scales are equal and the values,
    gradients and next states are bitwise those of separate calls; each next state keeps the input scale of
    its own history (``jla.training.fp8`` counts applications by it). Other projections (bf16, current
    scaling, no weight copies, ``None``) are called as usual.
    """
    idx = [i for i, p in enumerate(projections) if _shares_input(p)]
    dtypes = {projections[i].compute_dtype for i in idx}
    if len(idx) < 2 or len(dtypes) != 1:
        return tuple(None if p is None else p(x) for p in projections)
    xc = x.astype(jnp.dtype(dtypes.pop()))
    e4m3 = _fp8_max(E4M3)
    scales = [scale_from_amax(jnp.max(projections[i].state.input_amax_history), e4m3,
                              projections[i].state.input_scale) for i in idx]
    scale = lax.stop_gradient(functools.reduce(jnp.maximum, scales))
    shared = (lax.stop_gradient(quantize(xc, scale, E4M3)), scale, lax.stop_gradient(_amax(xc)))
    return tuple(p(xc, shared_input=shared) if i in idx else (None if p is None else p(x))
                 for i, p in enumerate(projections))
