# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint payloads, the commit protocol and strict restores."""

from __future__ import annotations

import importlib.util
import json
from concurrent.futures import Future
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from _toy_model import TOY_MODEL, build_toy
from jla.training import checkpointing as ckpt


@pytest.fixture(autouse=True)
def _flush_checkpoint_worker():
    yield
    ckpt.shutdown_checkpointing(timeout=30.0)


def _tree():
    return {
        "weight": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "half": jnp.arange(4, dtype=jnp.bfloat16),
        "scalar": jnp.asarray(7.0, jnp.float32),
        "count": jnp.asarray(3, jnp.int32),
    }


def test_pytree_roundtrip_keeps_shapes_dtypes_and_scalars(tmp_path: Path) -> None:
    source = _tree()
    ckpt.save_pytree(tmp_path / "tree.safetensors", source, metadata={"kind": "test"})
    template = jax.tree.map(jnp.zeros_like, source)
    restored = ckpt.load_pytree(tmp_path / "tree.safetensors", template)
    for name, value in source.items():
        assert restored[name].dtype == value.dtype and restored[name].shape == value.shape
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(value))


def test_restores_are_strict_about_keys_shapes_and_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "tree.safetensors"
    ckpt.save_pytree(path, {"a": jnp.ones((2,)), "b": jnp.ones((3,))})
    with pytest.raises(ValueError, match="missing"):
        ckpt.load_pytree(path, {"a": jnp.zeros((2,)), "b": jnp.zeros((3,)), "c": jnp.zeros((1,))})
    with pytest.raises(ValueError, match="unexpected"):
        ckpt.load_pytree(path, {"a": jnp.zeros((2,))})
    with pytest.raises(ValueError, match="shape"):
        ckpt.load_pytree(path, {"a": jnp.zeros((4,)), "b": jnp.zeros((3,))})
    with pytest.raises(ValueError, match="no implicit casts"):
        ckpt.load_pytree(path, {"a": jnp.zeros((2,), jnp.bfloat16), "b": jnp.zeros((3,))})
    partial = ckpt.load_pytree(path, {"a": jnp.zeros((2,)), "c": jnp.full((1,), 5.0)}, strict=False)
    np.testing.assert_array_equal(np.asarray(partial["c"]), [5.0])


def test_flatten_rejects_colliding_paths_and_typed_keys() -> None:
    with pytest.raises(ValueError, match="colliding"):
        ckpt.flatten_arrays({"a.b": {"c": jnp.ones(1)}, "a": {"b.c": jnp.zeros(1)}})
    with pytest.raises(TypeError, match="PRNG"):
        ckpt.flatten_arrays({"key": jax.random.key(0)})


def _state():
    model = build_toy(TOY_MODEL, key=jax.random.key(0))
    optimizer = optax.adamw(1e-3)
    return model, optimizer.init(eqx.filter(model, eqx.is_inexact_array))


def _save(manager: ckpt.CheckpointManager, step: int, model, opt_state, key=None) -> None:
    manager.save(
        step,
        model=model,
        opt_state=opt_state,
        data_state={"step": step},
        rng_key=jax.random.key(step) if key is None else key,
    )


@pytest.mark.parametrize("payload_format", ["safetensors", "sharded"])
@pytest.mark.parametrize("asynchronous", [True, False])
def test_manager_roundtrip_commit_and_prune(tmp_path: Path, payload_format: str, asynchronous: bool) -> None:
    manager = ckpt.CheckpointManager(tmp_path, keep=2, asynchronous=asynchronous, payload_format=payload_format)
    model, opt_state = _state()
    for step in (1, 2, 3):
        _save(manager, step, jax.tree.map(lambda x: x + step if eqx.is_inexact_array(x) else x, model), opt_state)
    ckpt.wait_for_pending_checkpoints()
    assert manager.committed_steps() == [2, 3] and manager.latest_step() == 3
    assert (tmp_path / "latest").read_text() == "step=3\npath=step_00000003\n"
    assert not list(tmp_path.glob("*.incomplete"))
    restored = manager.restore(model=model, opt_state=opt_state)
    assert restored.step == 3 and restored.data_state == {"step": 3}
    np.testing.assert_array_equal(np.asarray(restored.model.embed), np.asarray(model.embed) + 3)
    np.testing.assert_array_equal(jax.random.key_data(restored.rng_key), jax.random.key_data(jax.random.key(3)))
    assert ckpt.verify_checkpoint(tmp_path / "step_00000003", hashes=True).require_valid().payload_format == payload_format


def test_typed_rbg_key_roundtrip(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, asynchronous=False)
    model, opt_state = _state()
    key = jax.random.split(jax.random.key(5, impl="rbg"))[1]
    _save(manager, 1, model, opt_state, key)
    restored = manager.restore(model=model, opt_state=opt_state)
    assert str(jax.random.key_impl(restored.rng_key)) == "rbg"
    np.testing.assert_array_equal(jax.random.key_data(restored.rng_key), jax.random.key_data(key))


def test_incomplete_and_corrupt_checkpoints_are_never_selected(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, keep=0, asynchronous=False)
    model, opt_state = _state()
    _save(manager, 1, model, opt_state)
    _save(manager, 2, model, opt_state)
    # Step 3 crashed mid-write: marker present, no manifest.
    (tmp_path / "step_00000003").mkdir()
    (tmp_path / "step_00000003.incomplete").write_text("{}")
    # Step 2's payload was truncated after commit.
    payload = tmp_path / "step_00000002" / "model.safetensors"
    payload.write_bytes(payload.read_bytes()[:-1])
    assert manager.committed_steps() == [1]
    assert not ckpt.verify_checkpoint(tmp_path / "step_00000002").valid
    assert manager.restore(model=model, opt_state=opt_state).step == 1


def test_resave_of_a_step_replaces_it(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, asynchronous=False)
    model, opt_state = _state()
    _save(manager, 4, model, opt_state)
    _save(manager, 4, jax.tree.map(lambda x: x * 0 if eqx.is_inexact_array(x) else x, model), opt_state)
    np.testing.assert_array_equal(np.asarray(manager.restore(model=model, opt_state=opt_state).model.embed), 0.0)


def test_optimizer_state_must_match_exactly(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, asynchronous=False)
    model, opt_state = _state()
    _save(manager, 1, model, opt_state)
    other_state = optax.sgd(0.1, momentum=0.9).init(eqx.filter(model, eqx.is_inexact_array))
    with pytest.raises(ValueError, match="optimizer checkpoint does not match"):
        manager.restore(model=model, opt_state=other_state)


def test_background_work_is_ordered_and_waitable(tmp_path: Path) -> None:
    events: list[int] = []
    futures = [ckpt.submit_checkpoint_work(lambda i=i: events.append(i)) for i in range(5)]
    assert all(isinstance(f, Future) for f in futures)
    ckpt.wait_for_pending_checkpoints(timeout=10.0)
    assert events == [0, 1, 2, 3, 4]
    failing = ckpt.submit_checkpoint_work(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        ckpt.wait_for_pending_checkpoints(timeout=10.0)
    assert failing.done()
    with pytest.raises(ValueError, match="non-negative"):
        ckpt.wait_for_pending_checkpoints(timeout=-1)


def test_manifest_records_every_artifact(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, asynchronous=False)
    model, opt_state = _state()
    _save(manager, 7, model, opt_state)
    manifest = json.loads((tmp_path / "step_00000007" / "manifest.json").read_text())
    assert manifest["complete"] and manifest["step"] == 7 and manifest["payload_format"] == "safetensors"
    assert {a["path"] for a in manifest["artifacts"]} == {"model.safetensors", "optimizer.safetensors", "data_state.json"}


@pytest.mark.skipif(importlib.util.find_spec("orbax") is not None, reason="orbax-checkpoint is installed")
def test_orbax_format_needs_the_optional_package(tmp_path: Path) -> None:
    with pytest.raises(ImportError, match="orbax-checkpoint"):
        ckpt.CheckpointManager(tmp_path, payload_format="orbax")


@pytest.mark.skipif(importlib.util.find_spec("orbax") is None, reason="orbax-checkpoint is not installed in this environment")
def test_orbax_roundtrip(tmp_path: Path) -> None:
    manager = ckpt.CheckpointManager(tmp_path, payload_format="orbax")
    model, opt_state = _state()
    _save(manager, 1, model, opt_state)
    restored = manager.restore(model=model, opt_state=opt_state)
    np.testing.assert_array_equal(np.asarray(restored.model.embed), np.asarray(model.embed))
