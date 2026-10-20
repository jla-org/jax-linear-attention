# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Graph-health parsing of HLO text and XProf traces under resource bounds."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jla.training.graph_health import (
    GraphHealthArtifactError,
    GraphHealthMetrics,
    analyze_graph_health_path,
    analyze_xprof_trace_attribution,
    graph_health_policy,
    graph_health_release_warnings,
    graph_health_warnings,
    merge_graph_health_metrics,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH", ""))))}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=60, check=False)


def test_import_does_not_import_jax() -> None:
    result = _run("-c", "import sys, jla.training.graph_health; assert 'jax' not in sys.modules")
    assert result.returncode == 0, result.stderr


def test_text_artifact_counts_and_scan_metadata(tmp_path: Path) -> None:
    text = "%copy = copy(%arg)\n%update = dynamic-update-slice(%copy)\nhost_callback custom-call pallas_call CUDA graph\n"
    path = tmp_path / "module.hlo"
    path.write_text(text)
    metrics = analyze_graph_health_path(path)
    assert (metrics.total_events, metrics.memcpy_d2d_count, metrics.loop_dynamic_update_slice_fusion_count) == (2, 1, 1)
    assert (metrics.host_callback_count, metrics.custom_call_count, metrics.pallas_call_count) == (1, 1, 1)
    assert metrics.command_buffer_present is True
    assert metrics.artifact_files_parsed == 1 and metrics.artifact_bytes_read == len(text.encode())


def test_trace_arrays_and_command_buffer_evidence(tmp_path: Path) -> None:
    (tmp_path / "trace.json").write_text(json.dumps([{"name": "MemcpyD2D"}, {"name": "CUDA graph replay"}]))
    metrics = analyze_graph_health_path(tmp_path / "trace.json")
    assert metrics.total_events == 2 and metrics.memcpy_d2d_count == 1 and metrics.command_buffer_present is True
    (tmp_path / "plain.json").write_text(json.dumps({"traceEvents": [{"name": "kernel"}]}))
    absent = analyze_graph_health_path(tmp_path / "plain.json")
    assert absent.command_buffer_present is False
    assert any("evidence=absent" in w for w in graph_health_warnings(absent, require_command_buffer=True))
    assert any("evidence=unproven" in w for w in graph_health_warnings(GraphHealthMetrics(), require_command_buffer=True))
    unknown = GraphHealthMetrics.from_mapping({"command_buffer_present": "not-measured"})
    assert unknown.command_buffer_present is None


def test_missing_symlink_truncation_and_limits(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_graph_health_path(tmp_path / "missing")
    target = tmp_path / "target.hlo"
    target.write_text("%x = add()")
    (tmp_path / "link.hlo").symlink_to(target)
    with pytest.raises(GraphHealthArtifactError):
        analyze_graph_health_path(tmp_path / "link.hlo")
    large = tmp_path / "large" / "big.hlo"
    large.parent.mkdir()
    large.write_text("MemcpyD2D\n" * 100)
    metrics = analyze_graph_health_path(large, max_bytes_per_file=32)
    assert metrics.artifact_truncated
    assert any("lower bounds" in w for w in graph_health_warnings(metrics))
    with pytest.raises(GraphHealthArtifactError):
        analyze_graph_health_path(large, max_bytes_per_file=32, strict=True)
    many = tmp_path / "many"
    many.mkdir()
    for name in ("a", "b"):
        (many / f"{name}.hlo").write_text("%a = add()")
    limited = analyze_graph_health_path(many, max_files=1)
    assert limited.artifact_limit_reached and limited.artifact_files_parsed == 1


def test_gzip_reads_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "trace.txt.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b"MemcpyD2D\n" * 10_000)
    metrics = analyze_graph_health_path(path, max_bytes_per_file=1024)
    assert metrics.artifact_bytes_read <= 1024 and metrics.artifact_truncated


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs required")
def test_special_files_never_block(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "blocked.hlo")
    code = "import sys; from jla.training.graph_health import analyze_graph_health_path as a; assert a(sys.argv[1]).artifact_files_discovered == 0"
    result = _run("-c", code, str(tmp_path))
    assert result.returncode == 0, result.stderr


def test_policies_and_merging(tmp_path: Path) -> None:
    with pytest.raises(GraphHealthArtifactError):
        analyze_graph_health_path(tmp_path, strict=True)
    warnings = graph_health_release_warnings(GraphHealthMetrics(), policy="release")
    assert any("no non-empty" in w for w in warnings) and any("command-buffer" in w for w in warnings)
    assert graph_health_policy("future") is graph_health_policy()
    with pytest.raises(ValueError, match="unknown graph-health policy"):
        graph_health_policy("future", strict=True)
    merged = merge_graph_health_metrics(
        [
            GraphHealthMetrics(total_events=2, command_buffer_present=False, artifact_files_parsed=1),
            GraphHealthMetrics(total_events=3, command_buffer_present=True, artifact_read_error_count=1, artifact_truncated=True),
        ]
    )
    assert merged.total_events == 5 and merged.command_buffer_present is True
    assert merged.artifact_read_error_count == 1 and merged.artifact_truncated


def test_xprof_attribution_ranks_device_events(tmp_path: Path) -> None:
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "X", "pid": 1, "name": "loop_add_fusion.3", "dur": 30.0, "args": {"hlo_op": "loop_add_fusion.3", "name": "jit(step)/layer_0/add"}},
        {"ph": "X", "pid": 1, "name": "mosaic_gpu_kernel", "dur": 70.0, "args": {"hlo_op": "custom-call.1", "name": "jit(step)/layer_1/attn"}},
    ]
    (tmp_path / "trace.json").write_text(json.dumps({"traceEvents": events}))
    report = analyze_xprof_trace_attribution(tmp_path / "trace.json")
    assert report["device_event_count"] == 2 and report["device_duration_us"] == 100.0
    assert report["class_summary"]["pallas"]["percent_of_device_time"] == 70.0
    assert report["top_device_events"]["tiny_fragment"][0]["framework_op"] == "jit(step)/layer_N/add"


def test_cli_rejects_unknown_policy_in_strict_mode(tmp_path: Path) -> None:
    (tmp_path / "trace.json").write_text(json.dumps([{"name": "CUDA graph replay"}]))
    result = _run("-m", "jla.training.graph_health", str(tmp_path / "trace.json"), "--strict", "--policy", "future")
    assert result.returncode == 2 and "unknown graph-health policy" in result.stderr
    ok = _run("-m", "jla.training.graph_health", str(tmp_path / "trace.json"))
    assert ok.returncode == 0 and json.loads(ok.stdout)["metrics"]["command_buffer_present"] is True
