# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logical-axis sharding rules (MaxText-style) and per-host data placement.

Models name the dimensions of every parameter (``("vocab", "embed")``,
``("embed", "mlp")``, ...; see ``jla.models.logical_axes``). A rule maps a
logical name to an ordered list of mesh axes. For each dimension the first
listed mesh axes that (a) have size > 1, (b) are not already used by another
dimension of the same array and (c) divide the dimension are chosen; anything
else is replicated. Unknown logical names fail loudly.

Optimizer-state leaves that mirror a parameter (same trailing key path and
shape) inherit that parameter's sharding; other leaves are replicated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

__all__ = [
    "DEFAULT_RULES",
    "assert_params_sufficiently_sharded",
    "batch_axes",
    "constrain",
    "grouped_batch_sharding",
    "is_axes_leaf",
    "logical_to_spec",
    "make_global_batch",
    "opt_state_shardings",
    "param_shardings",
    "path_names",
    "process_row_range",
    "replicated",
    "replicated_fraction",
    "resolve_rules",
]

#: Default logical-axis rules for the ``(dp, fsdp, tp)`` mesh (+ ``ep``: experts and batch rows;
#: mesh axes a rule names but the mesh lacks count as size one).
DEFAULT_RULES: dict[str, tuple[str, ...]] = {
    "batch": ("dp", "fsdp", "ep"),
    "vocab": ("tp",),
    "embed": ("fsdp",),
    "heads": ("tp",),
    "kv_heads": ("tp",),
    "q_heads": ("tp",),
    "v_heads": ("tp",),
    "head_dim": (),
    "kv": (),
    "mlp": ("tp",),
    "expert_mlp": ("tp",),
    "experts": ("ep",),
    "layers": (),
    "norm": (),
    "conv": (),
    "rank": (),
    "mtp": (),
}


def resolve_rules(overrides: Sequence[tuple[str, Sequence[str]]] = ()) -> dict[str, tuple[str, ...]]:
    """Default rules updated with ``(name, mesh_axes)`` overrides."""
    rules = dict(DEFAULT_RULES)
    for name, axes in overrides:
        rules[str(name)] = tuple(axes)
    return rules


def is_axes_leaf(value: Any) -> bool:
    """A logical-axes leaf: a tuple of axis names and/or ``None``."""
    return isinstance(value, tuple) and all(item is None or isinstance(item, str) for item in value)


def path_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Key path as plain strings (``GetAttrKey`` names, dict keys, sequence indices)."""
    names = []
    for entry in path:
        for attr in ("name", "key", "idx"):
            if hasattr(entry, attr):
                names.append(str(getattr(entry, attr)))
                break
        else:
            names.append(str(entry))
    return tuple(names)


def logical_to_spec(
    logical_axes: Sequence[str | None],
    shape: Sequence[int],
    mesh: Mesh,
    rules: Mapping[str, Sequence[str]],
) -> PartitionSpec:
    """PartitionSpec for one array given its logical axes."""
    if len(logical_axes) != len(shape):
        raise ValueError(f"logical axes {tuple(logical_axes)} do not match shape {tuple(shape)}")
    used: set[str] = set()
    parts: list[Any] = []
    for name, dim in zip(logical_axes, shape):
        if name is None:
            parts.append(None)
            continue
        if name not in rules:
            raise KeyError(f"no sharding rule for logical axis {name!r}; known: {sorted(rules)}")
        chosen: list[str] = []
        product = 1
        for axis in rules[name]:
            size = int(mesh.shape.get(axis, 1))
            if size == 1 or axis in used or dim % (product * size):
                continue
            chosen.append(axis)
            product *= size
        used.update(chosen)
        parts.append(None if not chosen else chosen[0] if len(chosen) == 1 else tuple(chosen))
    return PartitionSpec(*parts)


def _is_array_like(leaf: Any) -> bool:
    return isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct, np.ndarray))


def param_shardings(
    params: Any,
    axes_tree: Any,
    mesh: Mesh,
    rules: Mapping[str, Sequence[str]],
) -> Any:
    """NamedSharding for every array leaf of ``params`` (structure preserved).

    ``params`` holds arrays or ``jax.ShapeDtypeStruct`` (abstract) leaves;
    ``axes_tree`` has the same key paths with logical-axes tuples as leaves.
    """
    axes_by_path = {
        path_names(path): axes
        for path, axes in jax.tree_util.tree_flatten_with_path(axes_tree, is_leaf=is_axes_leaf)[0]
        if is_axes_leaf(axes)
    }

    def assign(path: tuple[Any, ...], leaf: Any) -> Any:
        if not _is_array_like(leaf):
            return leaf
        names = path_names(path)
        if names not in axes_by_path:
            raise KeyError(f"no logical axes for parameter {'.'.join(names)} with shape {leaf.shape}")
        try:
            spec = logical_to_spec(axes_by_path[names], leaf.shape, mesh, rules)
        except (KeyError, ValueError) as exc:
            raise type(exc)(f"parameter {'.'.join(names)}: {exc}") from None
        return NamedSharding(mesh, spec)

    return jax.tree_util.tree_map_with_path(assign, params)


def opt_state_shardings(abstract_state: Any, params: Any, shardings: Any, mesh: Mesh) -> Any:
    """Shardings for an optimizer state given the parameter shardings.

    A state leaf whose key path ends with a parameter's key path and has that
    parameter's shape (Adam moments, momentum buffers) takes the parameter's
    sharding; counters and factored statistics are replicated.
    """
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_shardings = jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, jax.sharding.Sharding))
    if len(flat_params) != len(flat_shardings):
        raise ValueError("params and shardings have different numbers of array leaves")
    index = {
        path_names(path): (tuple(leaf.shape), sharding)
        for (path, leaf), sharding in zip(flat_params, flat_shardings)
    }
    replicated_sharding = replicated(mesh)

    def assign(path: tuple[Any, ...], leaf: Any) -> Any:
        if not _is_array_like(leaf):
            return leaf
        names = path_names(path)
        for start in range(len(names)):
            hit = index.get(names[start:])
            if hit is not None and hit[0] == tuple(leaf.shape):
                return hit[1]
        return replicated_sharding

    return jax.tree_util.tree_map_with_path(assign, abstract_state)


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def batch_axes(mesh: Mesh, rules: Mapping[str, Sequence[str]] = DEFAULT_RULES) -> tuple[str, ...]:
    """Mesh axes that split the batch dimension (``rules["batch"]`` with size > 1)."""
    return tuple(axis for axis in rules["batch"] if int(mesh.shape.get(axis, 1)) > 1)


def grouped_batch_sharding(mesh: Mesh, rules: Mapping[str, Sequence[str]] = DEFAULT_RULES, *, grouped: bool = True) -> NamedSharding:
    """Sharding of token arrays ``[GA, B, T]`` (``grouped``) or ``[B, T]``."""
    axes = batch_axes(mesh, rules)
    batch = None if not axes else axes[0] if len(axes) == 1 else axes
    spec = PartitionSpec(None, batch, None) if grouped else PartitionSpec(batch, None)
    return NamedSharding(mesh, spec)


def process_row_range(sharding: jax.sharding.Sharding, global_shape: Sequence[int], axis: int) -> tuple[int, int]:
    """Contiguous ``[start, stop)`` of ``axis`` that this process must provide."""
    shape = tuple(int(dim) for dim in global_shape)
    intervals = sorted(
        {index[axis].indices(shape[axis])[:2] for index in sharding.addressable_devices_indices_map(shape).values()}
    )
    start, stop = intervals[0][0], intervals[0][0]
    for lo, hi in intervals:
        if lo > stop:
            raise ValueError(f"process data rows are not contiguous along axis {axis}: {intervals}")
        stop = max(stop, hi)
    return start, stop


def make_global_batch(
    local: Mapping[str, np.ndarray],
    sharding: jax.sharding.Sharding,
    global_shape: Sequence[int],
) -> dict[str, jax.Array]:
    """Assemble global arrays from this process's rows (no host reads the full batch)."""
    shape = tuple(int(dim) for dim in global_shape)
    return {key: jax.make_array_from_process_local_data(sharding, value, shape) for key, value in local.items()}


def _leaf_size(leaf: Any) -> int:
    return int(math.prod(leaf.shape))


def _spec_axes(spec: Any) -> set[str]:
    axes: set[str] = set()
    for part in tuple(spec or ()):
        if part is None:
            continue
        axes.update((part,) if isinstance(part, str) else (axis for axis in part if axis is not None))
    return axes


def replicated_fraction(
    params: Any,
    mesh: Mesh,
    *,
    target_axes: Sequence[str] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Fraction of parameter elements not sharded over any active ``target_axes``.

    Returns ``(fraction, leaves)`` with the replicated leaves sorted by size.
    """
    active = {
        axis
        for axis in (mesh.axis_names if target_axes is None else target_axes)
        if int(mesh.shape[axis]) > 1
    }
    unknown = set(target_axes or ()) - set(mesh.axis_names)
    if unknown:
        raise ValueError(f"target axes not in the mesh: {sorted(unknown)}")
    if not active:
        return 0.0, []
    total = 0
    replicated_elements = 0
    leaves: list[dict[str, Any]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        if not hasattr(leaf, "shape"):
            continue
        size = _leaf_size(leaf)
        total += size
        spec = getattr(getattr(leaf, "sharding", None), "spec", None)
        if _spec_axes(spec).isdisjoint(active):
            replicated_elements += size
            leaves.append({"path": ".".join(path_names(path)), "shape": tuple(leaf.shape), "size": size, "spec": str(spec)})
    leaves.sort(key=lambda item: item["size"], reverse=True)
    return (replicated_elements / total if total else 0.0), leaves


def assert_params_sufficiently_sharded(
    params: Any,
    mesh: Mesh,
    *,
    tolerance: float = 0.05,
    target_axes: Sequence[str] | None = None,
) -> None:
    """Raise when more than ``tolerance`` of parameter elements are fully replicated.

    A leaf sharded over any active target axis counts as sharded, so ordinary
    layouts where some leaves use ``tp`` and others ``fsdp`` pass.
    """
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be finite and in [0, 1]")
    fraction, leaves = replicated_fraction(params, mesh, target_axes=target_axes)
    if fraction <= tolerance:
        return
    examples = "; ".join(f"{item['path']} shape={item['shape']} spec={item['spec']}" for item in leaves[:5])
    raise AssertionError(
        f"fully replicated parameter fraction {fraction:.2%} exceeds tolerance {tolerance:.2%}: {examples}"
    )


def constrain(tree: Any, shardings: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> Any:
    """``with_sharding_constraint`` leaf-wise where a sharding is given."""
    return jax.tree.map(
        lambda leaf, sharding: leaf if sharding is None else jax.lax.with_sharding_constraint(leaf, sharding),
        tree,
        shardings,
        is_leaf=is_leaf,
    )
