# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compile preflight, XLA flag merging, JSON-safe summaries and sharding audits."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.training.runtime import (
    audit_parameter_sharding,
    compile_preflight,
    compiled_memory_summary,
    merge_xla_flags,
    serializable_tree_summary,
    sharding_mismatches,
    tree_sharding_summary,
    write_sharding_dump,
)


def test_merge_xla_flags_keeps_existing_values_unless_forced() -> None:
    existing = "--a=1 --b=old --a=2"
    additions = ["--a=3", "--c=new", "--c=last"]
    assert merge_xla_flags(existing, additions) == "--b=old --a=2 --c=last"
    assert merge_xla_flags(existing, additions, force_keys={"--a"}) == "--b=old --a=3 --c=last"


def test_compiled_memory_summary_fields() -> None:
    summary = compiled_memory_summary(
        {
            "argument_size_in_bytes": 100,
            "output_size_in_bytes": 80,
            "temp_size_in_bytes": 40,
            "alias_size_in_bytes": 20,
            "peak_memory_in_bytes": 240,
            "host_argument_size_in_bytes": 30,
            "host_output_size_in_bytes": 20,
            "host_temp_size_in_bytes": 10,
            "host_alias_size_in_bytes": 5,
            "generated_code_size_in_bytes": 7,
        }
    )
    assert summary["total_size_in_bytes"] == 200 and summary["peak_memory_in_bytes"] == 240
    assert summary["host_total_size_in_bytes"] == 55 and summary["total_including_host_size_in_bytes"] == 255
    with pytest.raises(ValueError, match=">= 0"):
        compiled_memory_summary({"temp_size_in_bytes": -1})


class _FakeCompiled:
    def memory_analysis(self) -> dict[str, int]:
        return {"argument_size_in_bytes": 24, "output_size_in_bytes": 24, "temp_size_in_bytes": 8}

    def cost_analysis(self) -> dict[str, Any]:
        return {"flops": np.float64(12.0), "bytes accessed": np.float64(1024.0), "tiles": np.asarray([1, 2])}


class _FakeLowered:
    def compile(self, options: Any = None) -> _FakeCompiled:
        assert options in (None, {"xla_test_option": True})
        return _FakeCompiled()


class _FakeJitted:
    def lower(self, *args: Any) -> _FakeLowered:
        assert args == (3,)
        return _FakeLowered()


def test_compile_preflight_reports_analyses_and_returns_the_executable() -> None:
    report, compiled = compile_preflight("fake", _FakeJitted(), 3, compiler_options={"xla_test_option": True})
    assert report["status"] == "compiled" and isinstance(compiled, _FakeCompiled)
    assert report["memory"]["total_size_in_bytes"] == 56
    assert report["cost_analysis"]["flops"] == 12.0 and report["cost_analysis_bytes_accessed"] == 1024.0
    assert report["cost_analysis"]["tiles"] == {"kind": "ndarray", "shape": [2], "dtype": "int64", "elements": 2}
    json.dumps(report)


def test_compile_preflight_with_a_real_executable_and_failures() -> None:
    fn = jax.jit(lambda value: value @ value.T + 1.0)
    x = jnp.ones((4, 4), jnp.float32)
    report, compiled = compile_preflight("matmul", fn, x)
    assert report["status"] == "compiled" and report["memory"]["argument_size_in_bytes"] == 64
    np.testing.assert_allclose(np.asarray(compiled(x)), 5.0)
    json.dumps(report)
    assert compile_preflight("plain", lambda v: v, 1)[0]["status"] == "skipped"

    class Failing:
        def lower(self, *args: Any) -> None:
            raise RuntimeError("lower boom")

    failed, none = compile_preflight("failure", Failing(), 1)
    assert failed["status"] == "lower_failed" and "lower boom" in failed["error"] and none is None


@dataclasses.dataclass
class _Payload:
    name: str
    values: np.ndarray


def test_serializable_summary_never_dumps_array_values() -> None:
    summary = serializable_tree_summary(
        {"payload": _Payload("w", np.arange(1000, dtype=np.float32)), "path": Path("a.bin"), "inf": float("inf"), "bytes": b"secret", "set": {3, 1}}
    )
    assert summary["payload"]["values"] == {"kind": "ndarray", "shape": [1000], "dtype": "float32", "elements": 1000}
    assert summary["bytes"] == {"kind": "bytes", "size_bytes": 6} and summary["inf"] == "inf" and summary["set"] == [1, 3]
    assert "999." not in json.dumps(summary)


def test_sharding_audit_mismatches_and_atomic_dump(tmp_path: Path) -> None:
    left = {"large": jax.device_put(jnp.ones((32, 32), jnp.float32))}
    right = {"large": jax.device_put(jnp.ones((32, 32), jnp.bfloat16))}
    audit = audit_parameter_sharding(left, min_elements=16)
    assert audit["total_elements"] == 1024 and audit["replicated_fraction"] == pytest.approx(1.0)
    assert tree_sharding_summary(left, min_elements=16)[0]["shape"] == (32, 32)
    assert sharding_mismatches(left, right, min_elements=16)[0]["reason"] == "dtype"
    assert sharding_mismatches(left, {"other": right["large"]})[0]["reason"] == "tree-structure-mismatch"
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    destination = write_sharding_dump(tmp_path, mesh=mesh, model=left, optimizer=right, audit=audit, min_elements=16)
    dumped = json.loads(destination.read_text())
    assert dumped["mesh"]["shape"] == {"data": len(jax.devices())} and dumped["model"][0]["elements"] == 1024
    assert not list(tmp_path.glob(".sharding_dump.*"))
