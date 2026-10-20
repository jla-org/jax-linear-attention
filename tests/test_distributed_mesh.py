# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh planning, distributed init settings and logical-axis sharding rules (one process)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jla.distributed import init as dist
from jla.distributed.mesh import MeshPlan, build_mesh, parse_mesh_spec, resolve_mesh_plan
from jla.distributed.sharding import (
    DEFAULT_RULES,
    assert_params_sufficiently_sharded,
    grouped_batch_sharding,
    logical_to_spec,
    make_global_batch,
    opt_state_shardings,
    param_shardings,
    process_row_range,
    resolve_rules,
)
from jla.models import LOGICAL_AXIS_NAMES


def _torchrun_env(**overrides: str) -> dict[str, str]:
    env = {
        "WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_WORLD_SIZE": "4",
        "LOCAL_RANK": "3",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "12345",
    }
    return {**env, **overrides}


def test_mesh_specs_and_fsdp_fill() -> None:
    assert parse_mesh_spec("dp=2,tp=4") == {"dp": 2, "fsdp": 1, "tp": 4}
    assert parse_mesh_spec("2,4") == {"dp": 2, "fsdp": 4, "tp": 1}
    with pytest.raises(ValueError, match="more than once"):
        parse_mesh_spec("dp=2,dp=4")
    with pytest.raises(ValueError, match="mixes"):
        parse_mesh_spec("dp=2,4")
    plan = resolve_mesh_plan(8, dp=2, fsdp=-1, tp=2)
    assert plan.shape == (2, 2, 2) and plan.data_parallel_size == 4
    with pytest.raises(ValueError, match="not divisible"):
        resolve_mesh_plan(8, dp=3)
    with pytest.raises(ValueError, match="!= num_devices"):
        resolve_mesh_plan(8, dp=2, fsdp=2, tp=1)
    hybrid = resolve_mesh_plan(8, dp=2, fsdp=4, dcn_dp=2)
    assert hybrid.ici_shape == (1, 4, 1) and hybrid.num_slices == 2
    with pytest.raises(ValueError, match="divisible by dcn_fsdp"):
        MeshPlan(fsdp=3, dcn_fsdp=2)


def test_expert_and_pipeline_axes_together() -> None:
    plan = resolve_mesh_plan(8, ep=2, pp=2)
    assert plan.axis_names == ("dp", "fsdp", "tp", "ep", "pp")
    assert plan.shape == (1, 2, 1, 2, 2) and plan.data_parallel_size == 4  # pp does not split the batch


def test_single_device_mesh_and_lexical_fallback() -> None:
    mesh = build_mesh(resolve_mesh_plan(1), jax.devices()[:1])
    assert mesh.axis_names == ("dp", "fsdp", "tp") and mesh.devices.shape == (1, 1, 1)
    devices = [object() for _ in range(4)]
    with (
        patch("jax.experimental.mesh_utils.create_hybrid_device_mesh", side_effect=RuntimeError("no topology")),
        pytest.warns(RuntimeWarning, match="lexical device mesh"),
    ):
        from jla.distributed.mesh import device_array

        arranged = device_array(MeshPlan(dp=2, fsdp=2, dcn_dp=2), devices)
    assert arranged.shape == (2, 2, 1) and set(arranged.reshape(-1)) == set(devices)


def test_cluster_discovery_assigns_one_device_per_local_process() -> None:
    spec = dist.resolve_distributed_init(cluster_env="torchrun", env=_torchrun_env())
    assert (spec.num_processes, spec.process_id, spec.local_device_ids) == (8, 3, (3,))
    assert spec.coordinator_address == "127.0.0.1:12345"
    narrowed = dist.resolve_distributed_init(cluster_env="torchrun", env=_torchrun_env(CUDA_VISIBLE_DEVICES="GPU-abcd"))
    assert narrowed.local_device_ids == (0,)
    explicit = dist.resolve_distributed_init(cluster_env="torchrun", env=_torchrun_env(), local_device_ids=(1, 2))
    assert explicit.local_device_ids == (1, 2)
    slurm = dist.discover_cluster_env({"SLURM_NTASKS": "4", "SLURM_PROCID": "1", "SLURM_JOB_ID": "7"}, "auto")
    assert slurm == {"num_processes": 4, "process_id": 1, "cluster_detection_method": "slurm"}
    with pytest.raises(ValueError, match="found no SLURM"):
        dist.discover_cluster_env({}, "auto")


def test_settings_are_validated_and_env_is_never_read_implicitly() -> None:
    with pytest.raises(ValueError, match="outside 2 visible"):
        dist.resolve_distributed_init(cluster_env="torchrun", env=_torchrun_env(CUDA_VISIBLE_DEVICES="0,1"))
    with pytest.raises(ValueError, match="process_id must be"):
        dist.resolve_distributed_init(cluster_env="torchrun", env=_torchrun_env(RANK="8"))
    for address in ("localhost", "localhost:0", "localhost:65536", "host:nope"):
        with pytest.raises(ValueError):
            dist.resolve_distributed_init(coordinator_address=address, num_processes=2, process_id=0)
    multi_local = {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1", "NODE_RANK": "1", "MASTER_ADDR": "h", "MASTER_PORT": "1"}
    with pytest.raises(ValueError, match="process_id"):
        dist.resolve_distributed_init(cluster_env="torchrun", env=multi_local)
    with pytest.raises(ValueError, match="needs the environment mapping"):
        dist.resolve_distributed_init(cluster_env="slurm")
    with pytest.raises(ValueError, match="coordinator_address"):
        dist.resolve_distributed_init(num_processes=2, process_id=0)


def test_initialize_forwards_every_option() -> None:
    spec = dist.resolve_distributed_init(
        coordinator_address="localhost:12345", num_processes=2, process_id=1, local_device_ids=(0,), initialization_timeout=17
    )
    with patch.object(jax.distributed, "is_initialized", return_value=False), patch.object(jax.distributed, "initialize") as init:
        assert dist.initialize_distributed(spec)
    init.assert_called_once_with(
        num_processes=2,
        process_id=1,
        initialization_timeout=17,
        heartbeat_timeout_seconds=100,
        shutdown_timeout_seconds=300,
        coordinator_address="localhost:12345",
        local_device_ids=[0],
    )
    with patch.object(jax.distributed, "is_initialized", return_value=True):
        assert not dist.initialize_distributed(spec)
    assert dist.process_any(True) and not dist.process_any(False)


def _fake_mesh(**sizes: int) -> SimpleNamespace:
    return SimpleNamespace(shape={"dp": 1, "fsdp": 1, "tp": 1, **sizes})


def test_logical_rules_respect_divisibility_and_axis_reuse() -> None:
    rules = resolve_rules()
    assert set(LOGICAL_AXIS_NAMES) <= set(rules)
    mesh = _fake_mesh(fsdp=4, tp=2)
    assert logical_to_spec(("vocab", "embed"), (512, 64), mesh, rules) == P("tp", "fsdp")
    assert logical_to_spec(("layers", "embed", "mlp"), (2, 64, 128), mesh, rules) == P(None, "fsdp", "tp")
    assert logical_to_spec(("embed", "kv_heads"), (64, 3), mesh, rules) == P("fsdp", None)  # 3 % tp != 0
    assert logical_to_spec(("experts", "embed", "mlp"), (4, 64, 32), mesh, resolve_rules([("experts", ("fsdp",))])) == P("fsdp", None, "tp")
    assert logical_to_spec((None,), (64,), mesh, rules) == P(None)
    batch_rules = resolve_rules([("embed", ("dp", "fsdp"))])
    assert logical_to_spec(("embed",), (64,), _fake_mesh(dp=2, fsdp=4), batch_rules) == P(("dp", "fsdp"))
    with pytest.raises(KeyError, match="no sharding rule"):
        logical_to_spec(("mystery",), (4,), mesh, rules)
    with pytest.raises(ValueError, match="do not match shape"):
        logical_to_spec(("embed",), (4, 4), mesh, rules)
    assert DEFAULT_RULES["layers"] == ()


def test_param_and_optimizer_state_shardings_follow_the_parameters() -> None:
    mesh = build_mesh(resolve_mesh_plan(1), jax.devices()[:1])
    params = {"embed": jnp.zeros((8, 4)), "norm": jnp.ones((4,))}
    axes = {"embed": ("vocab", "embed"), "norm": (None,)}
    shardings = param_shardings(params, axes, mesh, resolve_rules())
    assert all(isinstance(s, NamedSharding) for s in jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, NamedSharding)))
    optimizer = optax.adamw(1e-3)
    state = opt_state_shardings(jax.eval_shape(optimizer.init, params), params, shardings, mesh)
    assert state[0].mu["embed"] is shardings["embed"] and state[0].count.spec == P()
    with pytest.raises(KeyError, match="no logical axes"):
        param_shardings({"extra": jnp.zeros(2)}, axes, mesh, resolve_rules())


def test_batch_rows_global_assembly_and_replication_audit() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("dp", "fsdp", "tp"))
    sharding = grouped_batch_sharding(mesh)
    assert sharding.spec == P(None, None, None)
    assert process_row_range(sharding, (2, 8, 16), axis=1) == (0, 8)
    local = {"input_ids": np.arange(2 * 8 * 16, dtype=np.int32).reshape(2, 8, 16)}
    batch = make_global_batch(local, sharding, (2, 8, 16))
    np.testing.assert_array_equal(np.asarray(batch["input_ids"]), local["input_ids"])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        assert_params_sufficiently_sharded({"x": jnp.ones(2)}, mesh, tolerance=1.1)
    assert_params_sufficiently_sharded({"x": jnp.ones(2)}, mesh, tolerance=0.0)  # no active axes


def test_process_any_gathers_one_flag_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from jax.experimental import multihost_utils

    gathered = np.asarray([0, 1], dtype=np.uint8)
    calls: list[tuple[np.ndarray, bool]] = []

    def fake_allgather(value: np.ndarray, *, tiled: bool = False) -> np.ndarray:
        calls.append((np.asarray(value), tiled))
        return gathered

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "process_allgather", fake_allgather)
    assert dist.process_any(False) is True
    assert calls[0][0].shape == () and calls[0][0].dtype == np.uint8 and calls[0][1] is False
    gathered[:] = 0
    assert dist.process_any(False) is False
