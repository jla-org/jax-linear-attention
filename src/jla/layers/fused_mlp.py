# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU blocks on the fused Mosaic GPU GEMMs (``ModelConfig.mlp_kernel = "mosaic_gpu"``).

:func:`apply_fused_mlp` swaps every :class:`~jla.layers.mlp.SwiGLU` of a model for a
:class:`FusedSwiGLU` holding the same parameters (same names, shapes and logical axes,
so checkpoints and sharding rules are unchanged). A call runs
:func:`jla.ops.mosaic_gpu.fused_gemm.swiglu_mlp` (SwiGLU fused into the gate/up GEMM's
epilogue, its backward into the down projection's dgrad epilogue) when
:func:`~jla.ops.mosaic_gpu.fused_gemm.unsupported_reason` accepts it (an H100, one
device, tokens % 128, d_model % 64, d_ff % 256), and the XLA path of ``SwiGLU``
otherwise (decode steps, CPU, sharded calls). A sharded call under a context mesh whose
axes all split the batch runs the kernels per shard
(:func:`~jla.ops.mosaic_gpu.fused_gemm.swiglu_mlp_sharded`); under ``tp`` it keeps XLA.
"""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers.linear import Linear
from jla.layers.mlp import MLP_HIDDEN_CHECKPOINT_NAME, SwiGLU, swiglu

__all__ = ["FusedSwiGLU", "apply_fused_mlp", "fused_mlp_modules"]

Array = jax.Array


class FusedSwiGLU(eqx.Module):
    """:class:`~jla.layers.mlp.SwiGLU` whose forward and backward run the fused Mosaic GPU GEMMs."""

    gate_up: Array  # [D, 2, F]
    down: Linear
    compute_dtype: str = eqx.field(static=True)
    # Save only the inputs and rerun the forward kernel in the backward pass (right inside a fully
    # rematerialized block, which recomputes the forward anyway; see fused_gemm.swiglu_mlp).
    recompute: bool = eqx.field(static=True, default=False)

    def uses_kernels(self, x_shape: tuple[int, ...]) -> bool:
        """Whether a call on an input of ``x_shape`` runs the fused kernels."""
        from jla.ops.mosaic_gpu import fused_gemm

        d, _, f = self.gate_up.shape
        reason = fused_gemm.unsupported_reason(math.prod(x_shape[:-1]), d, f, self.compute_dtype)
        return self.down.bias is None and reason is None

    def __call__(self, x: Array) -> Array:
        dtype = jnp.dtype(self.compute_dtype)
        if self.uses_kernels(x.shape):
            from jla.ops.mosaic_gpu import fused_gemm

            return fused_gemm.swiglu_mlp(x, self.gate_up, self.down.weight, compute_dtype=dtype,
                                         recompute=self.recompute)
        sharded = self._sharded(x)
        if sharded is not None:
            return sharded
        h = jnp.einsum("...d,dcf->...cf", x.astype(dtype), self.gate_up.astype(dtype))
        return self.down(swiglu(checkpoint_name(h, MLP_HIDDEN_CHECKPOINT_NAME)))

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"gate_up": ("embed", None, "mlp")}

    def _sharded(self, x: Array) -> Array | None:
        """The fused kernels per shard of the context mesh when only the multi-device rule declined the call
        (:func:`jla.ops.mosaic_gpu.fused_gemm.swiglu_mlp_sharded`), else ``None`` (the XLA path)."""
        from jla.ops import mosaic_gpu

        if self.down.bias is not None or x.ndim < 2 or not mosaic_gpu._HAVE_MOSAIC_GPU:
            return None
        from jla.distributed.kernel_sharding import per_device_call
        from jla.ops.mosaic_gpu import fused_gemm

        d, _, f = self.gate_up.shape
        if per_device_call() or fused_gemm.unsupported_reason(math.prod(x.shape[:-1]), d, f, self.compute_dtype,
                                                              check_devices=False) is not None:
            return None
        return fused_gemm.swiglu_mlp_sharded(x, self.gate_up, self.down.weight,
                                             compute_dtype=jnp.dtype(self.compute_dtype), recompute=self.recompute)


def apply_fused_mlp(tree: Any, *, recompute: bool = False) -> Any:
    """Replace every ``SwiGLU`` in ``tree`` (a model or part of one) by a ``FusedSwiGLU``.

    ``recompute``: the blocks are fully rematerialized (``ModelConfig.remat == "full"``).
    """

    def convert(node: Any) -> Any:
        if type(node) is SwiGLU:
            return FusedSwiGLU(node.gate_up, node.down, node.compute_dtype, recompute)
        return node

    return jax.tree_util.tree_map(convert, tree, is_leaf=lambda node: type(node) is SwiGLU)


def fused_mlp_modules(tree: Any) -> list[FusedSwiGLU]:
    """The ``FusedSwiGLU`` modules inside ``tree`` (for tests and logging)."""
    found: list[FusedSwiGLU] = []

    def visit(node: Any) -> Any:
        if isinstance(node, FusedSwiGLU):
            found.append(node)
        return node

    jax.tree_util.tree_map(visit, tree, is_leaf=lambda node: isinstance(node, FusedSwiGLU))
    return found
