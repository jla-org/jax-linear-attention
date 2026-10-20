# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from MaxText (Apache-2.0), src/maxtext/layers/quantizations.py (FP8 scales and amax
# histories as "_overwrite_with_gradient" variables written back by the train step).
"""Train-step handling of FP8 delayed-scaling state (:class:`jla.layers.fp8.Fp8State`).

The FP8 matmuls return the new scales and amax histories as the "gradient" of their state.
The train step (:func:`jla.training.step.build_train_step`) therefore

* combines those values over microbatches with ``max`` instead of a sum (every microbatch
  starts from the same state, so the scales agree and only the newest amax differs: the
  step records the largest amax of the step);
* zeroes them in the gradients the optimizer, the gradient norm and clipping see;
* writes them into the updated model when the step is applied, and keeps the old state when
  the step is rejected. A matmul that got no cotangent keeps its old state (its returned value
  is zero), and one used ``n`` times in one loss evaluation advances once (:func:`_next_state`).

Without FP8 state every helper is the identity (``mask`` is ``None``).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "accumulate_gradients",
    "apply_fp8_state",
    "attach_fp8_weights",
    "fp8_state_mask",
    "prepare_fp8_weights",
    "split_fp8_state",
    "without_fp8_state",
]


def _is_state(node: Any) -> bool:
    from jla.layers.fp8 import Fp8State

    return isinstance(node, Fp8State)


def fp8_state_mask(params: Any) -> Any | None:
    """Bool pytree like ``params``: ``True`` on FP8 state leaves; ``None`` when there are none."""
    mask = jax.tree.map(
        lambda node: jax.tree.map(lambda _: True, node) if _is_state(node) else False,
        params,
        is_leaf=_is_state,
    )
    return mask if any(jax.tree.leaves(mask)) else None


def accumulate_gradients(acc: Any, grads: Any, mask: Any | None) -> Any:
    """``acc + grads`` in float32; ``max`` on FP8 state leaves (their values are non-negative)."""
    if mask is None:
        return jax.tree.map(lambda a, g: a + g.astype(jnp.float32), acc, grads)
    return jax.tree.map(
        lambda a, g, m: jnp.maximum(a, g.astype(jnp.float32)) if m else a + g.astype(jnp.float32),
        acc,
        grads,
        mask,
    )


def split_fp8_state(grads: Any, mask: Any | None) -> tuple[Any, Any | None]:
    """``(grads with FP8 state leaves zeroed, the accumulated tree holding the new state)``."""
    if mask is None:
        return grads, None
    zeroed = jax.tree.map(lambda g, m: jnp.zeros_like(g) if m else g, grads, mask)
    return zeroed, grads


def _next_state(old: Any, value: Any, applied: jax.Array) -> Any:
    """The state after an optimizer step from the step's returned ``value`` (see the module docstring).

    ``value`` is the sum of what every application of the matmul returned in one loss evaluation
    (the max over microbatches). Its input scale is therefore ``uses`` times the scale the forward
    used, which the old state determines exactly:

    * ``uses == 0``: no cotangent reached the matmul (e.g. an unused head or a branch without a
      loss), so ``value`` is all zeros: the old state is kept instead of being zeroed.
    * ``uses == 1`` (the normal case): ``value`` is the new state.
    * ``uses > 1`` (the module ran several times in one forward pass): the scales are divided by
      ``uses`` and the history shifted from the old one, so the summed values never compound from
      step to step; the recorded amax is the sum of the ``uses`` amaxes, an upper bound on their max
      (at most ``uses`` times too large: it never saturates, it costs at most log2(uses) binades
      at the small end of the FP8 range).
    """
    from jla.layers.fp8 import E4M3, scale_from_amax

    used = scale_from_amax(jnp.max(old.input_amax_history, axis=-1), float(jnp.finfo(E4M3).max), old.input_scale)
    uses = jnp.round(value.input_scale.astype(jnp.float32) / used)
    take = jnp.logical_and(applied, uses >= 1)
    divisor = jnp.maximum(uses, 1.0)

    def scale(new: jax.Array, prev: jax.Array) -> jax.Array:
        return jnp.where(take, new / divisor, prev).astype(prev.dtype)

    def history(new: jax.Array, prev: jax.Array) -> jax.Array:
        shifted = jnp.concatenate([new[..., :1].astype(prev.dtype), prev[..., :-1]], axis=-1)
        return jnp.where(take[..., None], shifted, prev)

    return type(old)(
        input_scale=scale(value.input_scale, old.input_scale),
        input_amax_history=history(value.input_amax_history, old.input_amax_history),
        kernel_scale=scale(value.kernel_scale, old.kernel_scale),
        kernel_amax_history=history(value.kernel_amax_history, old.kernel_amax_history),
        grad_scale=scale(value.grad_scale, old.grad_scale),
        grad_amax_history=history(value.grad_amax_history, old.grad_amax_history),
    )


def apply_fp8_state(new_params: Any, old_params: Any, values: Any | None, mask: Any | None, applied: jax.Array) -> Any:
    """Write the new FP8 state into ``new_params`` if ``applied``, else keep ``old_params``' state.

    A matmul that got no cotangent keeps its old state, and one applied several times in one
    loss evaluation advances once (:func:`_next_state`).
    """
    if mask is None:
        return new_params
    return jax.tree.map(
        lambda new, old, value: _next_state(old, value, applied) if _is_state(old) else new,
        new_params,
        old_params,
        values,
        is_leaf=_is_state,
    )


def without_fp8_state(tree: Any) -> Any:
    """``tree`` with every :class:`~jla.layers.fp8.Fp8State` replaced by ``None``."""
    return jax.tree.map(lambda node: None if _is_state(node) else node, tree, is_leaf=_is_state)


def _is_fp8_module(node: Any) -> bool:
    from jla.layers.fp8 import Fp8Linear, Fp8StackedLinear, Fp8SwiGLU

    return isinstance(node, (Fp8Linear, Fp8StackedLinear, Fp8SwiGLU))


def prepare_fp8_weights(model: Any, *, transposed: bool | None = None) -> list[Any] | None:
    """Every delayed-scaling FP8 weight of ``model`` quantized once for an optimizer step.

    The train step calls this before its micro-step loop and :func:`attach_fp8_weights` inside it, so the
    forward and input-gradient GEMMs of every micro-step (and of a rematerialized forward) read one FP8 copy
    of each weight and of its transpose instead of re-quantizing the float32 weight each time (each copy is
    :func:`jla.layers.fp8.prepare_fp8_weight`: the same scale, values and amax as the per-call path). Returns
    one entry per leaf of ``model`` flattened down to its FP8 modules (None elsewhere), or None when no module
    qualifies (no delayed-scaling FP8 module, or ``ModelConfig.fp8_weight_once = false``). ``transposed``:
    whether to make the transposed copies (weights on one device; None: when the process sees one device).
    """
    from jla.layers.fp8 import prepare_fp8_weight

    leaves = jax.tree_util.tree_leaves(model, is_leaf=_is_fp8_module)
    prepared = [prepare_fp8_weight(leaf, transposed=transposed) if _is_fp8_module(leaf) else None
                for leaf in leaves]
    found = any(jax.tree_util.tree_leaves(p) for p in prepared if p is not None)
    return prepared if found else None


def attach_fp8_weights(model: Any, prepared: list[Any] | None) -> Any:
    """``model`` (the structure :func:`prepare_fp8_weights` saw) with the step's FP8 weight copies attached."""
    if prepared is None:
        return model
    from jla.layers.fp8 import attach_fp8_weight

    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=_is_fp8_module)
    if len(leaves) != len(prepared):
        raise ValueError(f"FP8 weight copies for {len(prepared)} leaves, model has {len(leaves)}")
    leaves = [attach_fp8_weight(leaf, p) if p is not None else leaf for leaf, p in zip(leaves, prepared)]
    return jax.tree_util.tree_unflatten(treedef, leaves)
