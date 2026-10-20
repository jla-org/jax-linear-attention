# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh planning for ``(dp, fsdp, tp)`` meshes.

``dp`` replicates parameters across data-parallel groups, ``fsdp`` shards
parameters, gradients and optimizer state while splitting the batch (ZeRO-3
style under GSPMD), and ``tp`` splits matmuls of one example. Every axis size
is the product of its in-slice (ICI) and cross-slice (DCN) parts; a DCN part
larger than one builds a hybrid mesh with ``mesh_utils.create_hybrid_device_mesh``.

``ep`` (expert parallelism, :mod:`jla.distributed.expert_parallel`) shards MoE
experts and, like ``dp``, splits the batch; ``pp`` (pipeline stages,
:mod:`jla.distributed.pipeline`) holds stage-local layers and does not split the
batch. Each is appended after ``(dp, fsdp, tp)`` (``ep`` before ``pp``) only when
it is larger than one, so ``ep = pp = 1`` meshes keep the ``(dp, fsdp, tp)`` layout.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MESH_AXES",
    "MeshPlan",
    "OPTIONAL_MESH_AXES",
    "build_mesh",
    "device_array",
    "parse_mesh_spec",
    "plan_from_config",
    "resolve_mesh_plan",
]

MESH_AXES: tuple[str, ...] = ("dp", "fsdp", "tp")
#: Axes appended after ``MESH_AXES`` only when their size is larger than one.
OPTIONAL_MESH_AXES: tuple[str, ...] = ("ep", "pp")


@dataclass(frozen=True)
class MeshPlan:
    """Resolved mesh sizes; ``dcn_*`` parts divide the corresponding axis."""

    dp: int = 1
    fsdp: int = 1
    tp: int = 1
    dcn_dp: int = 1
    dcn_fsdp: int = 1
    dcn_tp: int = 1
    ep: int = 1
    dcn_ep: int = 1
    pp: int = 1
    dcn_pp: int = 1

    def __post_init__(self) -> None:
        for axis in MESH_AXES + OPTIONAL_MESH_AXES:
            size, dcn = getattr(self, axis), getattr(self, f"dcn_{axis}")
            if size < 1 or dcn < 1:
                raise ValueError(f"mesh axis {axis}={size} (dcn {dcn}) must be >= 1")
            if size % dcn:
                raise ValueError(f"mesh axis {axis}={size} is not divisible by dcn_{axis}={dcn}")

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Mesh axis names: ``MESH_AXES`` plus each optional axis larger than one."""
        return MESH_AXES + tuple(axis for axis in OPTIONAL_MESH_AXES if getattr(self, axis) > 1)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self, axis) for axis in self.axis_names)

    @property
    def dcn_shape(self) -> tuple[int, ...]:
        return tuple(getattr(self, f"dcn_{axis}") for axis in self.axis_names)

    @property
    def ici_shape(self) -> tuple[int, ...]:
        return tuple(size // dcn for size, dcn in zip(self.shape, self.dcn_shape))  # type: ignore[return-value]

    @property
    def num_slices(self) -> int:
        return math.prod(self.dcn_shape)

    @property
    def device_count(self) -> int:
        return math.prod(self.shape)

    @property
    def data_parallel_size(self) -> int:
        """Independent example streams: ``dp * fsdp * ep`` (``tp`` splits one example)."""
        return self.dp * self.fsdp * self.ep

    @property
    def axis_sizes(self) -> dict[str, int]:
        return dict(zip(self.axis_names, self.shape))


def parse_mesh_spec(value: str | Mapping[str, int]) -> dict[str, int]:
    """Parse ``"2,4,1"`` (positional dp,fsdp,tp) or ``"dp=2,fsdp=4"`` into axis sizes.

    Optional axes (``"dp=2,ep=4"``, ``"dp=2,pp=4"``) are named only and appear in the result when given.
    """
    axes = {"dp": 1, "fsdp": 1, "tp": 1}
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        if all("=" not in part for part in parts):
            if len(parts) > len(MESH_AXES):
                raise ValueError(f"mesh spec has {len(parts)} sizes, expected at most {len(MESH_AXES)}")
            items = list(zip(MESH_AXES, parts))
        else:
            items = []
            for part in parts:
                if "=" not in part:
                    raise ValueError(f"mesh spec mixes positional and axis=value items: {part!r}")
                name, size = (piece.strip() for piece in part.split("=", 1))
                items.append((name, size))
    seen: set[str] = set()
    for name, size in items:
        if name not in axes and name not in OPTIONAL_MESH_AXES:
            raise ValueError(f"unknown mesh axis {name!r}; expected one of {MESH_AXES + OPTIONAL_MESH_AXES}")
        if name in seen:
            raise ValueError(f"mesh axis {name!r} was specified more than once")
        seen.add(name)
        axes[name] = int(size)
    return axes


def resolve_mesh_plan(
    num_devices: int,
    *,
    dp: int = 1,
    fsdp: int = -1,
    tp: int = 1,
    dcn_dp: int = 1,
    dcn_fsdp: int = 1,
    dcn_tp: int = 1,
    ep: int = 1,
    dcn_ep: int = 1,
    pp: int = 1,
    dcn_pp: int = 1,
) -> MeshPlan:
    """Resolve axis sizes for ``num_devices``; ``fsdp=-1`` takes the remaining devices."""
    if num_devices < 1:
        raise ValueError("num_devices must be >= 1")
    for name, size in (("dp", dp), ("tp", tp), ("ep", ep), ("pp", pp)):
        if size < 1:
            raise ValueError(f"{name}={size} must be >= 1")
    if fsdp == -1:
        fixed = dp * tp * ep * pp
        if num_devices % fixed:
            raise ValueError(f"num_devices={num_devices} is not divisible by dp*tp*ep*pp={fixed}")
        fsdp = num_devices // fixed
    elif fsdp < 1:
        raise ValueError(f"fsdp={fsdp} must be >= 1 or -1")
    plan = MeshPlan(
        dp=dp,
        fsdp=fsdp,
        tp=tp,
        dcn_dp=dcn_dp,
        dcn_fsdp=dcn_fsdp,
        dcn_tp=dcn_tp,
        ep=ep,
        dcn_ep=dcn_ep,
        pp=pp,
        dcn_pp=dcn_pp,
    )
    if plan.device_count != num_devices:
        raise ValueError(f"mesh product dp*fsdp*tp*ep*pp={plan.device_count} != num_devices={num_devices}")
    return plan


def plan_from_config(config: Any, num_devices: int) -> MeshPlan:
    """Resolve a :class:`jla.config.MeshConfig` for ``num_devices``."""
    return resolve_mesh_plan(
        num_devices,
        dp=config.dp,
        fsdp=config.fsdp,
        tp=config.tp,
        dcn_dp=config.dcn_dp,
        dcn_fsdp=config.dcn_fsdp,
        dcn_tp=config.dcn_tp,
        ep=config.ep,
        dcn_ep=config.dcn_ep,
        pp=config.pp,
        dcn_pp=config.dcn_pp,
    )


def build_mesh(
    plan: MeshPlan,
    devices: Sequence[Any] | None = None,
    *,
    topology_aware: bool = True,
) -> Any:
    """Create a ``jax.sharding.Mesh`` with axes ``plan.axis_names``.

    The axes are ``("dp", "fsdp", "tp")``, then ``"ep"`` and ``"pp"`` when larger than one.
    Topology-aware placement uses ``jax.experimental.mesh_utils``; if JAX
    cannot map the requested shape onto the physical topology, devices are
    laid out in enumeration order and a ``RuntimeWarning`` names the cause.
    """
    import jax
    from jax.sharding import Mesh

    device_list = list(jax.devices() if devices is None else devices)
    if len(device_list) != plan.device_count:
        raise ValueError(f"mesh {plan.shape} needs {plan.device_count} devices, got {len(device_list)}")
    if not topology_aware and plan.num_slices == 1:
        return Mesh(np.asarray(device_list, dtype=object).reshape(plan.shape), plan.axis_names)
    return Mesh(device_array(plan, device_list), plan.axis_names)


def device_array(plan: MeshPlan, devices: Sequence[Any]) -> np.ndarray:
    """Physical device layout for ``plan`` (hybrid when DCN parts are > 1)."""
    from jax.experimental import mesh_utils

    try:
        if plan.num_slices > 1:
            return np.asarray(mesh_utils.create_hybrid_device_mesh(plan.ici_shape, plan.dcn_shape, list(devices)))
        return np.asarray(mesh_utils.create_device_mesh(plan.shape, list(devices)))
    except Exception as exc:  # mesh_utils raises backend-specific errors for unknown topologies
        warnings.warn(
            f"falling back to a lexical device mesh for shape {plan.shape}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.asarray(list(devices), dtype=object).reshape(plan.shape)
