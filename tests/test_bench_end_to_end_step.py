# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The step profiler's HLO-metadata parser and kernel categories (benchmarks/bench_end_to_end_step.py)."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_end_to_end_step.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_end_to_end_step", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HLO = """HloModule jit_train_step

%fused_conv (p0: f32[8,2048,64]) -> f32[8,2048,64] {
  %p0 = f32[8,2048,64]{2,1,0} parameter(0)
  ROOT %mul.1 = f32[8,2048,64]{2,1,0} multiply(%p0, %p0), metadata={op_name="jit(train_step)/mul" source_file="/x/src/jla/layers/conv.py" source_line=105}
}

ENTRY %main (a: f32[8,2048,64]) -> f32[8,2048,64] {
  %a = f32[8,2048,64]{2,1,0} parameter(0)
  %loop_multiply_fusion = f32[8,2048,64]{2,1,0} fusion(%a), kind=kLoop, calls=%fused_conv
  %custom-call.7 = bf16[8,2048,64]{2,1,0} custom-call(%loop_multiply_fusion), custom_call_target="mosaic_gpu_v2", metadata={op_name="jit(train_step)/pallas_call" source_file="/x/src/jla/ops/mosaic_gpu/scalar_gate.py" source_line=10}
  %gemm.3 = f32[8,2048,64]{2,1,0} custom-call(%a), custom_call_target="__cublas$lt$matmul", metadata={op_name="jit(train_step)/dot_general" source_file="/x/src/jla/ops/cross_entropy.py" source_line=52}
  ROOT %rms = f32[8,2048,64]{2,1,0} multiply(%a, %a), metadata={op_name="jit(train_step)/mul" source_file="/x/src/jla/layers/norm.py" source_line=40}
}
"""


def test_hlo_metadata_inherits_fused_computation_source(bench):
    meta = bench.hlo_metadata(_HLO)
    assert meta["loop_multiply_fusion"]["source_file"].endswith("layers/conv.py")
    assert meta["loop_multiply_fusion"]["source_line"] == "105"
    assert meta["custom-call.7"]["target"] == "mosaic_gpu_v2"
    assert meta["gemm.3"]["source_file"].endswith("ops/cross_entropy.py")


def test_categories(bench):
    meta = bench.hlo_metadata(_HLO)
    assert bench.categorize("loop_multiply_fusion", meta["loop_multiply_fusion"]) == "short conv1d"
    assert bench.categorize("scalar_gate_fwd", meta["custom-call.7"]) == "mosaic_gpu kernels"
    assert bench.categorize("sm90_xmma_gemm_bf16", meta["gemm.3"]) == "cross entropy matmuls"
    assert bench.categorize("loop_multiply_fusion_2", meta["rms"]) == "norms"
    assert bench.categorize("mystery_kernel", None) == "other (no metadata)"


def test_categories_from_shape_and_op_name(bench):
    ce = ("16384,4096]", "f32[16384]", "[128256,")
    logits = {"shape": "f32[16384,4096]{1,0}", "op_name": "jit(train_step)/while/body/jvp()/dot_general"}
    assert bench.categorize("loop_convert_fusion", logits, ce) == "cross entropy (softmax, scan glue)"
    rowmax = {"shape": "f32[16384]{0}", "op_name": "jit(train_step)/jvp()/reduce_max"}
    assert bench.categorize("input_reduce_fusion_26", rowmax, ce) == "cross entropy (softmax, scan glue)"
    bwd = {"shape": "f32[128,1536]{1,0}", "op_name": "jit(train_step)/transpose(jvp())/checkpoint/reduce_sum"}
    assert bench.categorize("input_reduce_fusion_27", bwd, ce) == "backward + remat elementwise/reductions"
    bcast = {"shape": "bf16[8,2048,24,128]{3,2,1,0}", "op_name": "jit(train_step)/jvp()/reshape"}
    assert bench.categorize("loop_broadcast_fusion_3", bcast, ce) == "broadcasts (e.g. groups -> heads)"
    fwd = {"shape": "f32[8,2048,1536]{2,1,0}", "op_name": "jit(train_step)/while/body/jvp()/mul"}
    assert bench.categorize("loop_multiply_fusion_2", fwd, ce) == "forward elementwise/reductions"
    opt = {"shape": "f32[128256,768]{1,0}", "op_name": "jit(train_step)/cond/branch_1_fun/add"}
    assert bench.categorize("loop_add_fusion", opt, ce) == "optimizer"


def test_analyze_sums_device_time_per_step(bench, tmp_path):
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "/host:CPU"}},
        {"ph": "X", "pid": 1, "tid": 1, "name": "loop_multiply_fusion", "dur": 300.0,
         "args": {"hlo_op": "loop_multiply_fusion"}},
        {"ph": "X", "pid": 1, "tid": 1, "name": "sm90_xmma_gemm", "dur": 900.0, "args": {"hlo_op": "gemm.3"}},
        {"ph": "X", "pid": 2, "tid": 1, "name": "host_work", "dur": 5000.0, "args": {}},
    ]
    with gzip.open(tmp_path / "perfetto_trace.json.gz", "wt") as f:
        json.dump({"traceEvents": events}, f)
    out = bench.analyze(tmp_path, _HLO, steps=3, top=5)
    assert out["device_ms_per_step"] == pytest.approx(1.2 / 3)  # host events are excluded
    assert out["categories_ms_per_step"]["cross entropy matmuls"] == pytest.approx(0.3)
    assert out["categories_ms_per_step"]["short conv1d"] == pytest.approx(0.1)
    assert out["top_ops"][0]["hlo_op"] == "gemm.3"


_HLO_FRAMES = """HloModule jit_train_step, is_scheduled=true

FileNames
1 "/x/benchmarks/bench_end_to_end_step.py"
2 "/x/src/jla/layers/ssm.py"
3 "/x/site-packages/equinox/_module/_prebuilt.py"

FunctionNames
1 "main"
2 "_repeat_groups"
3 "__call__"

FileLocations
1 {file_name_id=1 function_name_id=1 line=284 end_line=284 column=21 end_column=27}
2 {file_name_id=2 function_name_id=2 line=78 end_line=78 column=11 end_column=40}
3 {file_name_id=3 function_name_id=3 line=10 end_line=10 column=1 end_column=9}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}

ENTRY %main (a: bf16[8,2048,1,128]) -> bf16[8,2048,24,128] {
  %a = bf16[8,2048,1,128]{3,2,1,0} parameter(0)
  %eqx.1 = bf16[8,2048,1,128]{3,2,1,0} negate(%a), metadata={op_name="jit(train_step)/transpose(jvp())/neg" stack_frame_id=3}
  ROOT %loop_broadcast_fusion.3 = bf16[8,2048,24,128]{3,2,1,0} broadcast(%eqx.1), metadata={op_name="jit(train_step)/jvp()/reshape" scheduling_name="loop_broadcast_fusion.3" stack_frame_id=2}
}
"""


def test_stack_frame_ids_resolve_to_source_lines(bench):
    frames = bench.stack_frames(_HLO_FRAMES)
    # printed parent_frame_id is parent + 1: frame 3 -> frame 2 -> frame 1 (no parent)
    assert [f[1:] for f in frames[3]] == [("10", "__call__"), ("78", "_repeat_groups"), ("284", "main")]
    meta = bench.hlo_metadata(_HLO_FRAMES)
    bcast = meta["loop_broadcast_fusion.3"]
    assert bcast["source_file"].endswith("layers/ssm.py") and bcast["source_line"] == "78"
    assert bcast["function"] == "_repeat_groups"
    # an op whose innermost frame is in a dependency is attributed to the innermost frame of this package
    assert meta["eqx.1"]["source_file"].endswith("layers/ssm.py")
    assert bench.categorize("loop_broadcast_fusion_3", bcast) == "mixer glue (gates, layout, D-skip)"


def test_gemm_with_cross_entropy_shape_is_a_cross_entropy_matmul(bench):
    ce = ("16384,4096]", "f32[16384]", "[128256,")
    logits = {"shape": "f32[16384,4096]{1,0}", "op_name": "jit(train_step)/transpose(jvp())/while/body/dot_general"}
    assert bench.categorize("gemm_fusion_dot_general_13", logits, ce) == "cross entropy matmuls"
    proj = {"shape": "bf16[16384,1536]{1,0}", "op_name": "jit(train_step)/jvp()/dot_general"}
    assert bench.categorize("gemm_fusion_dot_general_2", proj, ce) == "matmuls (projections, MLP)"


def test_analyze_reports_time_per_source_line(bench, tmp_path):
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "X", "pid": 1, "tid": 1, "name": "loop_broadcast_fusion_3", "dur": 600.0, "args": {}},
    ]
    with gzip.open(tmp_path / "perfetto_trace.json.gz", "wt") as f:
        json.dump({"traceEvents": events}, f)
    out = bench.analyze(tmp_path, _HLO_FRAMES, steps=2, top=5)
    assert out["categories_ms_per_step"] == {"mixer glue (gates, layout, D-skip)": pytest.approx(0.3)}
    (line,) = out["sources_ms_per_step"]
    assert line.startswith("ssm.py:78 _repeat_groups [fwd]")
