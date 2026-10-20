# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HLO accounting of benchmarks/speed_of_light_step.py on a hand-written scheduled module (CPU only)."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import speed_of_light_step as sol  # noqa: E402

HLO = """HloModule jit_train_step, is_scheduled=true

%fused_slice (param_0.1: f32[24,2048,2048], param_1.1: s32[]) -> bf16[2048,2048] {
  %param_0.1 = f32[24,2048,2048]{2,1,0} parameter(0)
  %param_1.1 = s32[] parameter(1)
  %constant.1 = s32[] constant(0)
  %dynamic-slice.1 = f32[1,2048,2048]{2,1,0} dynamic-slice(%param_0.1, %param_1.1, %constant.1, %constant.1), dynamic_slice_sizes={1,2048,2048}
  %bitcast.1 = f32[2048,2048]{1,0} bitcast(%dynamic-slice.1)
  ROOT %convert.1 = bf16[2048,2048]{1,0} convert(%bitcast.1)
}

%fused_dus (param_0.2: bf16[24,65536,2048], param_1.2: bf16[65536,2048], param_2.2: s32[]) -> bf16[24,65536,2048] {
  %param_0.2 = bf16[24,65536,2048]{2,1,0} parameter(0)
  %param_1.2 = bf16[65536,2048]{1,0} parameter(1)
  %bitcast.2 = bf16[1,65536,2048]{2,1,0} bitcast(%param_1.2)
  %param_2.2 = s32[] parameter(2)
  %constant.2 = s32[] constant(0)
  ROOT %dynamic-update-slice.2 = bf16[24,65536,2048]{2,1,0} dynamic-update-slice(%param_0.2, %bitcast.2, %param_2.2, %constant.2, %constant.2)
}

%body (p.3: (s32[], bf16[65536,2048], f32[24,2048,2048], bf16[24,65536,2048])) -> (s32[], bf16[65536,2048], f32[24,2048,2048], bf16[24,65536,2048]) {
  %p.3 = (s32[], bf16[65536,2048]{1,0}, f32[24,2048,2048]{2,1,0}, bf16[24,65536,2048]{2,1,0}) parameter(0)
  %gte.0 = s32[] get-tuple-element(%p.3), index=0
  %gte.1 = bf16[65536,2048]{1,0} get-tuple-element(%p.3), index=1
  %gte.2 = f32[24,2048,2048]{2,1,0} get-tuple-element(%p.3), index=2
  %gte.3 = bf16[24,65536,2048]{2,1,0} get-tuple-element(%p.3), index=3
  %loop_slice_fusion = bf16[2048,2048]{1,0} fusion(%gte.2, %gte.0), kind=kLoop, calls=%fused_slice, metadata={op_name="jit(train_step)/while/body/dot_general"}
  %custom-call.7 = (bf16[65536,2048]{1,0}, s8[33554432]{0}) custom-call(%gte.1, %loop_slice_fusion), custom_call_target="__cublas$lt$matmul", metadata={op_name="jit(train_step)/while/body/transpose(jvp())/dot_general"}, backend_config={"gemm_backend_config":{"dot_dimension_numbers":{"lhs_batch_dimensions":[],"lhs_contracting_dimensions":["1"],"rhs_batch_dimensions":[],"rhs_contracting_dimensions":["0"]}}}
  %gte.4 = bf16[65536,2048]{1,0} get-tuple-element(%custom-call.7), index=0
  %loop_dus_fusion = bf16[24,65536,2048]{2,1,0} fusion(%gte.3, %gte.4, %gte.0), kind=kLoop, calls=%fused_dus, metadata={op_name="jit(train_step)/while/body/checkpoint/rematted_computation/mul"}
  %constant.3 = s32[] constant(1)
  %add.3 = s32[] add(%gte.0, %constant.3)
  ROOT %tuple.3 = (s32[], bf16[65536,2048]{1,0}, f32[24,2048,2048]{2,1,0}, bf16[24,65536,2048]{2,1,0}) tuple(%add.3, %gte.4, %gte.2, %loop_dus_fusion)
}

%cond (p.4: (s32[], bf16[65536,2048], f32[24,2048,2048], bf16[24,65536,2048])) -> pred[] {
  %p.4 = (s32[], bf16[65536,2048]{1,0}, f32[24,2048,2048]{2,1,0}, bf16[24,65536,2048]{2,1,0}) parameter(0)
  %gte.5 = s32[] get-tuple-element(%p.4), index=0
  %constant.4 = s32[] constant(24)
  ROOT %compare.4 = pred[] compare(%gte.5, %constant.4), direction=LT
}

ENTRY %main (Arg_0.1: bf16[65536,2048], Arg_1.1: f32[24,2048,2048], Arg_2.1: bf16[24,65536,2048]) -> bf16[65536,2048] {
  %Arg_0.1 = bf16[65536,2048]{1,0} parameter(0)
  %Arg_1.1 = f32[24,2048,2048]{2,1,0} parameter(1)
  %Arg_2.1 = bf16[24,65536,2048]{2,1,0} parameter(2)
  %constant.5 = s32[] constant(0)
  %tuple.5 = (s32[], bf16[65536,2048]{1,0}, f32[24,2048,2048]{2,1,0}, bf16[24,65536,2048]{2,1,0}) tuple(%constant.5, %Arg_0.1, %Arg_1.1, %Arg_2.1)
  %while.5 = (s32[], bf16[65536,2048]{1,0}, f32[24,2048,2048]{2,1,0}, bf16[24,65536,2048]{2,1,0}) while(%tuple.5), condition=%cond, body=%body, backend_config={"known_trip_count":{"n":"24"}}
  %loop_add_fusion = bf16[65536,2048]{1,0} fusion(%Arg_0.1, %Arg_0.1), kind=kLoop, calls=%fused_slice, metadata={op_name="jit(train_step)/add"}
  ROOT %gte.6 = bf16[65536,2048]{1,0} get-tuple-element(%while.5), index=1
}
"""


def test_shapes_and_bytes():
    assert sol.shape_bytes("bf16[65536,2048]{1,0}") == 65536 * 2048 * 2
    assert sol.shape_bytes("(f32[4,8]{1,0}, s8[16]{0})") == 4 * 8 * 4 + 16
    assert sol.shape_bytes("f8e4m3fn[10,10]{1,0}") == 100
    assert sol.shape_bytes("pred[]") == 1


def test_parse_module_and_multipliers():
    comps, entry, names = sol.parse_module(HLO)
    assert entry == "main"
    assert {"fused_slice", "fused_dus", "body", "cond", "main"} <= set(comps)
    mult = sol.multipliers(comps, entry)
    assert mult["main"] == 1.0 and mult["body"] == 24.0 and mult["cond"] == 25.0
    gemm = names["custom-call.7"]
    assert gemm["opcode"] == "custom-call" and gemm["operands"] == ["gte.1", "loop_slice_fusion"]


def test_gemm_flops_and_problem():
    comps, entry, names = sol.parse_module(HLO)
    flops, mkn = sol._gemm_custom_call(names["custom-call.7"], names)
    assert mkn == (65536, 2048, 2048)
    assert flops == 2.0 * 65536 * 2048 * 2048


def test_fusion_bytes_slices_and_in_place_updates():
    comps, entry, names = sol.parse_module(HLO)
    # reads one f32 layer slice of the stacked weight (not all 24) and the s32 index, writes the bf16 copy
    assert sol.fusion_bytes(names["loop_slice_fusion"], comps, names) == 2048 * 2048 * 4 + 4 + 2048 * 2048 * 2
    # the in-place update reads its update (and index) and writes only the updated slice, not the buffer
    assert sol.fusion_bytes(names["loop_dus_fusion"], comps, names) == 65536 * 2048 * 2 + 4 + 65536 * 2048 * 2


def test_phase_and_ceiling_lookup():
    assert sol.phase_of("jit(f)/while/body/checkpoint/rematted_computation/mul", []) == "remat"
    assert sol.phase_of("jit(f)/transpose(jvp())/dot_general", []) == "bwd"
    assert sol.phase_of("jit(f)/dot_general", [("/x/optax/_src/transform.py", "1", "f")]) == "opt"
    ceilings = {"gemm_bf16_tflops": {"65536x2048x4096": 800.0}, "gemm_default_tflops": 700.0}
    assert sol._gemm_tflops(ceilings, (65536, 2048, 4096)) == 800.0
    assert sol._gemm_tflops(ceilings, (4096, 2048, 65536)) == 800.0  # the transposed problem costs the same
    assert sol._gemm_tflops(ceilings, (8, 8, 8)) == 700.0


def _write_profile(tmp_path: Path) -> Path:
    pdir = tmp_path / "prof"
    (pdir / "plugins" / "profile" / "run").mkdir(parents=True)
    (pdir / "train_step.hlo.txt").write_text(HLO)
    meta = {"profile_steps": 1, "wall_ms_median": 20.0,
            "result": {"arch": "toy", "micro_batch": 16, "seq_len": 4096,
                       "flops_per_token": {"training": 3 * 2.0 * 2048 * 2048 * 24, "attention_core": 0.0,
                                           "linear_attention_core": 0.0}}}
    (pdir / "profile_meta.json").write_text(json.dumps(meta))
    events = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}}]
    ts = 0.0
    for _ in range(24):  # one profiled step: 24 layer iterations
        for op, dur in (("loop_slice_fusion", 10.0), ("custom-call.7", 400.0), ("loop_dus_fusion", 200.0)):
            events.append({"ph": "X", "pid": 1, "ts": ts, "dur": dur, "name": op.replace(".", "_"),
                           "args": {"hlo_op": op}})
            ts += dur
    events.append({"ph": "X", "pid": 1, "ts": ts, "dur": 100.0, "name": "loop_add_fusion",
                   "args": {"hlo_op": "loop_add_fusion"}})
    with gzip.open(pdir / "plugins" / "profile" / "run" / "perfetto_trace.json.gz", "wt") as f:
        json.dump(events, f)
    return pdir


def test_analyze_end_to_end(tmp_path):
    pdir = _write_profile(tmp_path)
    ceilings = {"gemm_bf16_tflops": {"65536x2048x2048": 800.0}, "gemm_default_tflops": 800.0,
                "kernel_tflops": 800.0, "hbm_gbps": 3000.0}
    out = sol.analyze(pdir, ceilings)
    cats = out["categories"]
    gemm = cats["GEMM"]
    assert gemm["flops"] == pytest.approx(24 * 2.0 * 65536 * 2048 * 2048)
    assert gemm["ms"] == pytest.approx(24 * 0.4)
    # SOL of the GEMM: FLOPs at 800 TFLOP/s (above its bytes at 3 TB/s), capped by the measured time
    assert gemm["sol_ms"] == pytest.approx(min(24 * 0.4, 24 * 2.0 * 65536 * 2048 * 2048 / 800e12 * 1e3))
    remat = [g for g in out["gaps"] if g["item"].endswith("[remat]")]
    assert remat and remat[0]["gap_ms"] == pytest.approx(24 * 0.2)  # recompute counts in full
    assert out["wall_ms"] == 20.0 and out["busy_ms"] == pytest.approx(24 * 0.61 + 0.1)
    assert out["sol_ideal_ms"] <= out["sol_no_remat_ms"] <= out["sol_as_compiled_ms"]


def test_build_ceilings(tmp_path):
    gemm = {"rows": [
        {"gemm": [8192, 8192, 8192], "tflops": 800.0, "dtype": "bf16", "impl": "xla"},
        {"gemm": [8192, 8192, 8192], "tflops": 810.0, "dtype": "bf16", "impl": "mosaic"},
        {"gemm": [8192, 8192, 8192], "tflops": 900.0, "dtype": "fp8", "impl": "xla", "layout": "as spelled"},
        {"gemm": [8192, 8192, 8192], "tflops": 1600.0, "dtype": "fp8", "impl": "xla", "layout": "K-major"},
        {"gemm": [65536, 2048, 320], "tflops": 600.0, "dtype": "bf16", "impl": "xla"},
        {"op": "x * 1.5 (read+write)", "gbps": 2900.0},
        {"op": "sum (read)", "gbps": 3200.0}]}
    torch = {"clock": [{"gemm": "bf16 8192^3", "tflops": 680.0}, {"gemm": "fp8 8192^3", "tflops": 1200.0}]}
    paths = [tmp_path / "gemm.json", tmp_path / "torch.json"]
    paths[0].write_text(json.dumps(gemm))
    paths[1].write_text(json.dumps(torch))
    for i, v in enumerate((690.0, 700.0, 720.0)):
        paths.append(tmp_path / f"clock{i}.json")
        paths[-1].write_text(json.dumps({"cmd": "clock", "sustained_tflops_wall": v}))
    c = sol.build_ceilings([str(p) for p in paths])
    assert c["gemm_bf16_tflops"] == {"8192x8192x8192": 810.0, "65536x2048x320": 600.0}
    assert c["gemm_fp8_tflops"]["8192x8192x8192"] == 1600.0  # the K-major layout, not XLA's as spelled
    assert c["gemm_default_tflops"] == c["kernel_tflops"] == 810.0
    assert c["hbm_gbps"] == 2900.0  # the best read + write stream, not a read-only one
    # sustained / burst of the 8192^3 GEMM: the median of the four bf16 clock runs (695), and the FP8 one
    assert c["power_scale_bf16"] == round(695.0 / 810.0, 4)
    assert c["power_scale_fp8"] == round(1200.0 / 1600.0, 4)


def test_ideal_power_capped_and_fp8_columns(tmp_path):
    pdir = _write_profile(tmp_path)
    ceilings = {"gemm_bf16_tflops": {"65536x2048x2048": 800.0}, "gemm_fp8_tflops": {"65536x2048x2048": 1600.0},
                "gemm_default_tflops": 800.0, "kernel_tflops": 800.0, "hbm_gbps": 3000.0,
                "power_scale_bf16": 0.8, "power_scale_fp8": 0.75}
    out = sol.analyze(pdir, ceilings)
    flops = 24 * 2.0 * 65536 * 2048 * 2048  # the only op left in the ideal step: the GEMM
    # at burst rates an op that ran faster than its ceiling is its own ceiling; the other columns are not capped
    assert out["sol_ideal_ms"] == pytest.approx(min(24 * 0.4, flops / 800e12 * 1e3), abs=0.006)
    assert out["sol_ideal_power_capped_ms"] == pytest.approx(flops / (800e12 * 0.8) * 1e3, abs=0.006)
    assert out["sol_ideal_fp8_ms"] == pytest.approx(flops / 1600e12 * 1e3, abs=0.006)
    assert out["sol_ideal_fp8_power_capped_ms"] == pytest.approx(flops / (1600e12 * 0.75) * 1e3, abs=0.006)
    mfu8 = out["sol_ideal_fp8_power_capped_bf16_equivalent_mfu_percent"]
    assert out["sol_ideal_fp8_power_capped_fp8_peak_utilization_percent"] == round(mfu8 / 2, 1)
    assert out["tensor_flops_per_step_ideal"] == pytest.approx(flops)
    assert out["tensor_work_bound_power_capped_ms"] == pytest.approx(flops / (800e12 * 0.8) * 1e3, abs=0.006)
    assert out["attention_work_fraction"] == 1.0 and out["ce_logits_recompute_ms"] == 0.0


def test_ce_logits_recompute_rule():
    def gemm(phase, site, mkn):
        return {"category": "GEMM", "phase": phase, "site": site, "gemm": list(mkn)}

    rows = [gemm("fwd", "ops/cross_entropy.py:54", (65536, 2048, 4096)),    # logits
            gemm("bwd", "ops/cross_entropy.py:196", (65536, 2048, 4096)),   # logits again: recompute
            gemm("bwd", "ops/cross_entropy.py:196", (65536, 4096, 2048)),   # dgrad
            gemm("bwd", "ops/cross_entropy.py:196", (4096, 65536, 2048)),   # wgrad
            gemm("bwd", "layers/mlp.py:20", (65536, 2048, 4096)),           # same problem, not the LM head
            {"category": "elementwise/reduction fusions", "phase": "bwd", "site": "ops/cross_entropy.py:196",
             "gemm": None}]
    assert sol.ce_logits_recompute(rows) == {1}


def test_attention_work_under_document_masking():
    import speed_of_light_prototypes as proto

    t = 512
    one = proto.attention_work([[0] * t], None, 128)
    assert one["counted_pairs"] == one["exact_pairs"] == t * (t + 1) // 2
    assert one["tile_pairs"] == 10 * 128 * 128  # 4 diagonal + 6 below-diagonal tiles, all visited
    two = proto.attention_work([[0] * 256 + [1] * 256], None, 128)
    assert two["exact_pairs"] == 2 * 256 * 257 // 2 and two["documents"] == 2
    assert two["tile_pairs"] == 6 * 128 * 128  # the diagonal + (1, 0) and (3, 2); the cross-document tiles skip
    assert two["tile_fraction"] == round(6 * 128 * 128 / (t * (t + 1) / 2), 4)
    win = proto.attention_work([[0] * t], 128, 128)
    assert win["counted_pairs"] == win["exact_pairs"] == 128 * 129 // 2 + (t - 128) * 128
    assert win["tile_pairs"] == 7 * 128 * 128  # the diagonal and the first sub-diagonal only
