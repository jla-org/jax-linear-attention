# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision policy and dtype-preserving casts.

Parameters are stored in ``param_dtype`` (float32 by default) and the model
computes in ``compute_dtype``; gradients accumulate and losses reduce in at
least float32. The train step uses :func:`restore_dtypes` after every update so
an fp32 update never silently promotes bf16 parameters (which would change the
executable's signature and double parameter memory).

Casting is explicit and leaves integer, boolean and complex leaves alone. FP8
needs scaled matmuls and is never applied by generic casting: the switch is
``ModelConfig.fp8`` (``"delayed"`` or ``"current"``, see :mod:`jla.layers.fp8`),
recorded here as ``MixedPrecisionPolicy.fp8``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "MixedPrecisionPolicy",
    "cast_floating",
    "check_param_dtypes",
    "is_fp8_capable",
    "restore_dtypes",
]

_FP8_DEVICE_MARKERS = ("H100", "H200", "GH200", "B100", "B200", "B300", "GB200", "GB300", "RTX PRO 6000")


def _floating_dtype(value: Any, name: str) -> jnp.dtype:
    try:
        dtype = jnp.dtype(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a dtype, got {value!r}") from exc
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"{name} must be a real floating dtype, got {dtype}")
    if dtype.itemsize < 2:
        raise TypeError(
            f"{name}={dtype}: FP8 is not a storage or compute dtype; FP8 training uses the scaled-matmul "
            "layers of jla.layers.fp8, switched on with ModelConfig.fp8 = 'delayed' or 'current'"
        )
    return dtype


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Storage, compute, accumulation and loss dtypes of one run."""

    param_dtype: Any = jnp.float32
    compute_dtype: Any = jnp.bfloat16
    accum_dtype: Any = jnp.float32
    loss_dtype: Any = jnp.float32
    fp8: str = "off"  # FP8 matmul recipe of the dense projections and MLPs: "off", "delayed", "current"

    def __post_init__(self) -> None:
        for name in ("param_dtype", "compute_dtype", "accum_dtype", "loss_dtype"):
            dtype = _floating_dtype(getattr(self, name), name)
            if name in ("accum_dtype", "loss_dtype") and dtype.itemsize < 4:
                raise TypeError(f"{name} must be float32 or wider, got {dtype}")
            object.__setattr__(self, name, dtype)
        if self.fp8 not in ("off", "delayed", "current"):
            raise ValueError(f"fp8 must be 'off', 'delayed' or 'current'; got {self.fp8!r}")

    @classmethod
    def from_model_config(cls, config: Any) -> MixedPrecisionPolicy:
        return cls(param_dtype=config.param_dtype, compute_dtype=config.compute_dtype, fp8=getattr(config, "fp8", "off"))

    def replace(self, **changes: Any) -> MixedPrecisionPolicy:
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in ("param_dtype", "compute_dtype", "accum_dtype", "loss_dtype", "fp8")}


def _is_real_floating(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating)


def cast_floating(tree: Any, dtype: Any) -> Any:
    """Cast real floating array leaves to ``dtype``; other leaves are unchanged."""
    target = _floating_dtype(dtype, "dtype")
    return jax.tree.map(lambda leaf: leaf.astype(target) if _is_real_floating(leaf) else leaf, tree)


def restore_dtypes(tree: Any, reference: Any) -> Any:
    """Cast each array leaf of ``tree`` to the dtype of the matching ``reference`` leaf."""
    return jax.tree.map(
        lambda leaf, ref: leaf if leaf is None or leaf.dtype == ref.dtype else leaf.astype(ref.dtype),
        tree,
        reference,
        is_leaf=lambda x: x is None,
    )


def check_param_dtypes(params: Any, expected: Any) -> None:
    """Raise if any floating parameter is not stored in ``expected`` (FP8 scaling state is float32)."""
    from jla.training.fp8 import without_fp8_state

    params = without_fp8_state(params)
    target = jnp.dtype(expected)
    wrong = [
        (jax.tree_util.keystr(path), str(leaf.dtype))
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != target
    ]
    if wrong:
        raise TypeError(f"parameters must be stored in {target}; found {wrong[:5]}")


def is_fp8_capable(device_kind: str) -> bool:
    """Whether a device kind string names an FP8-capable NVIDIA part."""
    upper = device_kind.upper()
    return any(marker in upper for marker in _FP8_DEVICE_MARKERS)
