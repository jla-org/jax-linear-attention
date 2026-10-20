# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 1.3B step attribution (benchmarks/profile_1p3b*.py): phases, categories, GEMM roles, flame traces, Amdahl."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name: str):
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    spec = importlib.util.spec_from_file_location(name, _BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prof():
    return _load("profile_1p3b")


@pytest.fixture(scope="module")
def flame():
    return _load("profile_1p3b_flame")


@pytest.fixture(scope="module")
def report():
    return _load("profile_1p3b_report")


def _hlo(src: Path) -> str:
    """A compiled-HLO excerpt with JAX 0.11's frame tables: a q projection GEMM called from a mixer file."""
    return f"""HloModule jit_train_step

FileNames
1 "{src / 'jla' / 'layers' / 'linear.py'}"
2 "{src / 'jla' / 'layers' / 'mixer.py'}"
3 "{src / 'jla' / 'layers' / 'conv.py'}"

FunctionNames
1 "__call__"
2 "forward"

FileLocations
1 {{file_name_id=2 function_name_id=2 line=2}}
2 {{file_name_id=1 function_name_id=1 line=1}}
3 {{file_name_id=3 function_name_id=1 line=1}}

StackFrames
1 {{file_location_id=1 parent_frame_id=1}}
2 {{file_location_id=2 parent_frame_id=2}}
3 {{file_location_id=3 parent_frame_id=1}}

ENTRY %main (a: bf16[64,32]) -> bf16[64,32] {{
  %a = bf16[64,32]{{1,0}} parameter(0)
  %custom-call.1 = bf16[64,32]{{1,0}} custom-call(%a), custom_call_target="__cublas$lt$matmul", metadata={{op_name="jit(train_step)/jvp()/dot_general" stack_frame_id=2}}
  %custom-call.2 = bf16[64,32]{{1,0}} custom-call(%a), custom_call_target="__cublas$lt$matmul", metadata={{op_name="jit(train_step)/transpose(jvp())/rematted_computation/dot_general" stack_frame_id=2}}
  %custom-call.3 = bf16[64,32]{{1,0}} custom-call(%a), custom_call_target="__cublas$lt$matmul", metadata={{op_name="jit(train_step)/transpose(jvp())/dot_general" stack_frame_id=2}}
  %custom-call.4 = bf16[64,32]{{1,0}} custom-call(%a), custom_call_target="mosaic_gpu_v2", metadata={{op_name="jit(train_step)/jvp()/pallas_call" source_file="{src / 'jla' / 'ops' / 'mosaic_gpu' / 'kda.py'}" source_line=7}}
  ROOT %loop_add_fusion = bf16[64,32]{{1,0}} add(%a, %a), metadata={{op_name="jit(train_step)/transpose(jvp())/add" stack_frame_id=3}}
}}
"""


def test_gemm_roles_phases_and_categories(prof, tmp_path):
    (tmp_path / "jla" / "layers").mkdir(parents=True)
    (tmp_path / "jla" / "layers" / "linear.py").write_text("y = lax.dot_general(x, w)\n")
    (tmp_path / "jla" / "layers" / "mixer.py").write_text("# mixer\nq = self.q_proj(x)\n")
    meta = prof.hlo_instructions(_hlo(tmp_path))
    cache: dict = {}
    cat, phase, site = prof.classify("nvjet_sm90_tst_128x128", meta["custom-call.1"], cache)
    assert (cat, phase) == ("GEMM: in/q/k/v projections", "fwd") and "mixer.py:2" in site
    assert prof.classify("nvjet_sm90_tst_128x128", meta["custom-call.2"], cache)[1] == "remat"  # remat wins
    assert prof.classify("nvjet_sm90_tst_128x128", meta["custom-call.3"], cache)[1] == "bwd"
    assert prof.classify("kda_fwd_kernel", meta["custom-call.4"], cache)[0] == "recurrent kernels (Mosaic GPU)"
    assert prof.classify("loop_add_fusion", meta["loop_add_fusion"], cache)[:2] == ("short conv", "bwd")
    assert prof.classify("mystery", None, cache)[0] == "other (no metadata)"


def test_union_counts_overlaps_once(prof):
    assert prof._union_us([(0.0, 10.0), (5.0, 12.0), (20.0, 25.0)]) == 17.0
    assert prof._union_us([]) == 0.0


def test_analyze_wall_busy_and_categories(prof, tmp_path):
    src = tmp_path / "src"
    (src / "jla" / "layers").mkdir(parents=True)
    (src / "jla" / "layers" / "mixer.py").write_text("# mixer\nq = self.q_proj(x)\n")
    pdir = tmp_path / "trace"
    pdir.mkdir()
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/device:GPU:0"}},
        {"ph": "X", "pid": 1, "tid": 1, "ts": 0.0, "dur": 400.0, "name": "nvjet_gemm", "args": {"hlo_op": "custom-call.1"}},
        {"ph": "X", "pid": 1, "tid": 2, "ts": 100.0, "dur": 200.0, "name": "kda_fwd_kernel", "args": {"hlo_op": "custom-call.4"}},
        {"ph": "X", "pid": 1, "tid": 1, "ts": 1000.0, "dur": 100.0, "name": "loop_add_fusion", "args": {"hlo_op": "loop_add_fusion"}},
    ]
    with gzip.open(pdir / "perfetto_trace.json.gz", "wt") as f:
        json.dump(events, f)
    out = prof.analyze(pdir, _hlo(src), steps=1, top=10)
    assert out["kernel_ms_per_step"] == pytest.approx(0.7)
    assert out["busy_ms_per_step"] == pytest.approx(0.5)  # the kernel on stream 2 overlaps the GEMM
    cats = out["categories_ms_per_step"]
    assert cats["GEMM: in/q/k/v projections"]["fwd"] == pytest.approx(0.4)
    assert cats["short conv"]["bwd"] == pytest.approx(0.1)


def _flame_trace(path: Path) -> None:
    """A torch.profiler chrome trace: a labelled forward GEMM, its input-gradient GEMM, FA3 and a norm kernel."""
    M = 64

    def op(name, ts, dur, tid, dims=None):
        return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur,
                "args": {"Input Dims": dims or []}}

    def launch(corr, ts, tid):
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": tid, "ts": ts, "dur": 1,
                "args": {"correlation": corr}}

    def kernel(corr, name, ts, dur):
        return {"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7, "ts": ts, "dur": dur,
                "args": {"correlation": corr}}

    events = [
        {"ph": "X", "cat": "user_annotation", "name": "jla_linear:model.layers.0.attn.q_proj", "pid": 1, "tid": 1,
         "ts": 0, "dur": 50, "args": {}},
        op("aten::linear", 1, 40, 1), op("aten::mm", 2, 30, 1, [[M, 32], [32, 16]]), launch(1, 5, 1),
        op("flash_attn_3::fwd", 60, 20, 1), launch(2, 65, 1),
        op("autograd::engine::evaluate_function: MmBackward0", 100, 50, 2),
        op("MmBackward0", 101, 45, 2), op("aten::mm", 102, 30, 2, [[M, 16], [16, 32]]), launch(3, 105, 2),
        op("autograd::engine::evaluate_function: LayerNormFunctionBackward", 200, 30, 2),
        op("LayerNormFunctionBackward", 201, 25, 2), launch(4, 205, 2),
        kernel(1, "nvjet_sm90_tst_64x64", 10, 30), kernel(2, "flash::FlashAttnFwdSm90", 70, 40),
        kernel(3, "nvjet_sm90_tst_64x64", 110, 30), kernel(4, "layer_norm_bwd_kernel", 210, 10),
    ]
    path.write_text(json.dumps({"traceEvents": events}))


def test_flame_analyze_roles_and_phases(flame, tmp_path):
    trace = tmp_path / "rank0_trace.json"
    _flame_trace(trace)
    out = flame.analyze(trace, tokens=64, wall_ms=0.2)
    cats = out["categories_ms_per_step"]
    assert cats["GEMM: in/q/k/v projections"]["fwd"] == pytest.approx(0.03)
    assert cats["GEMM: in/q/k/v projections"]["bwd"] == pytest.approx(0.03)  # role from the weight shape
    assert cats["softmax attention"]["total"] == pytest.approx(0.04)  # a CUTLASS kernel, not a GEMM
    assert cats["norms"]["bwd"] == pytest.approx(0.01)
    assert out["busy_ms_per_step"] == pytest.approx(0.11)


def test_amdahl_and_rows(report):
    assert report.amdahl(100.0, 40.0, 2) == pytest.approx(100.0 / 80.0)
    assert report.amdahl(100.0, 40.0, float("inf")) == pytest.approx(100.0 / 60.0)
    profile = {"busy_ms_per_step": 90.0, "categories_ms_per_step": {
        "GEMM: MLP down": {"fwd": 10.0, "bwd": 20.0, "total": 30.0},
        "recurrent kernels (FLA)": {"recompute": 5.0, "total": 5.0}}}
    rows = report.rows_of(profile, wall_ms=100.0)
    assert rows["GEMM MLP down"]["total"] == 30.0 and rows["recurrent kernels"]["remat"] == 5.0
    assert rows["idle (gaps, launch, host)"]["total"] == pytest.approx(10.0)


def _flame_trace_unlabelled(path: Path, M: int = 64, V: int = 96) -> None:
    """Unlabelled GEMMs: FLA's fused linear cross entropy (vocabulary dims, forward and weight gradient), a fused
    Function's GEMM whose role comes from the model's weight shapes (forward and its input gradient), FLA and
    TileLang kernels, and an aten elementwise kernel whose name contains a recurrent-looking word."""

    def op(name, ts, dur, tid, dims=None):
        return {"ph": "X", "cat": "cpu_op", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur,
                "args": {"Input Dims": dims or []}}

    def launch(corr, ts, tid):
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": tid, "ts": ts, "dur": 1,
                "args": {"correlation": corr}}

    def kernel(corr, name, ts, dur):
        return {"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": 7, "ts": ts, "dur": dur,
                "args": {"correlation": corr}}

    bwd = "autograd::engine::evaluate_function: "
    events = [
        op("FusedLinearCrossEntropyFunction", 0, 100, 1), op("aten::mm", 1, 30, 1, [[M, 32], [32, V]]), launch(1, 5, 1),
        op("SwiGLULinearFunction", 200, 50, 1), op("aten::mm", 201, 30, 1, [[M, 48], [48, 32]]), launch(2, 205, 1),
        op("ChunkGatedDeltaRuleFunction", 300, 50, 1), launch(3, 305, 1), launch(4, 310, 1),
        op("aten::mul", 400, 10, 1), launch(5, 402, 1),
        op(bwd + "FusedLinearCrossEntropyFunctionBackward", 500, 60, 2),
        op("aten::addmm", 501, 40, 2, [[V, 32], [V, M], [M, 32]]), launch(6, 505, 2),
        op(bwd + "SwiGLULinearFunctionBackward", 600, 60, 2),
        op("aten::mm", 601, 40, 2, [[M, 32], [32, 48]]), launch(7, 605, 2),
        kernel(1, "nvjet_sm90_tst_128x128", 10, 30), kernel(2, "nvjet_sm90_tst_64x64", 210, 20),
        kernel(3, "chunk_gated_delta_rule_fwd_kernel_h", 310, 15), kernel(4, "main_kernel", 330, 5),
        kernel(5, "void at::native::vectorized_elementwise_kernel<4, at::native::binary_internal::MulFunctor<float>>",
               405, 4),
        kernel(6, "nvjet_sm90_tst_256x128", 510, 25), kernel(7, "nvjet_sm90_tst_64x64", 610, 12),
    ]
    path.write_text(json.dumps({"traceEvents": events}))


def test_flame_lm_head_by_vocab_dims_weight_roles_and_torch_kernels(flame, tmp_path):
    trace = tmp_path / "rank0_trace.json"
    _flame_trace_unlabelled(trace)
    out = flame.analyze(trace, tokens=64, wall_ms=1.0, vocab=96, weight_roles={"48x32": ["MLP down"]})
    cats = out["categories_ms_per_step"]
    # FLA's fused linear cross entropy: no nn.Linear label, the vocabulary dimension names the LM head
    assert cats["GEMM: LM head"]["fwd"] == pytest.approx(0.03)
    assert cats["GEMM: LM head"]["bwd"] == pytest.approx(0.025)  # the addmm weight gradient
    # a GEMM inside a fused Function: its role from the model's weight shapes; the backward GEMM inherits it
    assert cats["GEMM: MLP down"]["fwd"] == pytest.approx(0.02)
    assert cats["GEMM: MLP down"]["bwd"] == pytest.approx(0.012)
    assert cats["recurrent kernels (FLA)"]["total"] == pytest.approx(0.02)  # the FLA and TileLang kernels only
    assert cats["elementwise/reduction (aten, triton)"]["total"] == pytest.approx(0.004)  # binary_internal


def _flame_run(path: Path, arch: str, ac: str, compiled: bool, accum: int, median_s: float, loss: float,
               doc: bool = True) -> str:
    path.write_text(json.dumps({
        "arch": arch, "devices": 1, "tokens_per_step": 65536, "activation_checkpointing": ac, "torch_compile": compiled,
        "grad_accum": accum, "document_masking": doc, "step_time_median_s": median_s, "step_time_mean_s": median_s,
        "tokens_per_sec": 65536 / median_s, "loss_last": loss, "peak_memory_allocated_gib_max_gpu": 50.0,
        "jla_flops": {"flops_per_token": {"training": 9.0e9}}}))
    return str(path)


def test_flame_modes_strongest_valid_and_mfu_targets(report, tmp_path):
    paths = [_flame_run(tmp_path / "a.json", "kda-1p3b", "full", False, 1, 3.0, 7.80),
             _flame_run(tmp_path / "b.json", "kda-1p3b", "none", True, 4, 2.5, 7.81),
             _flame_run(tmp_path / "c.json", "rwkv7-1p3b", "full", False, 1, 2.8, 7.78),
             _flame_run(tmp_path / "d.json", "rwkv7-1p3b", "none", True, 8, 2.7, 9.31)]  # loss off: flagged
    modes = report.flame_modes(paths)
    assert modes["kda-1p3b"]["strongest"]["ms"] == pytest.approx(2500.0)
    assert modes["rwkv7-1p3b"]["strongest"]["ms"] == pytest.approx(2800.0)  # the flagged faster run is not used
    assert [r["valid"] for r in modes["rwkv7-1p3b"]["runs"]] == [True, False]
    ours = {"kda-1p3b": {"wall_ms": 2000.0, "tok_per_s": 32768, "training_flops_per_token": 9.0e9}}
    lines, rec = report.flame_tables(ours, modes, {})
    fps = 9.0e9 * 65536
    assert rec["kda-1p3b"]["mfu_needed"]["3x_over_ours"] == pytest.approx(100 * fps / (2.0 / 3) / 989e12)
    assert rec["kda-1p3b"]["mfu_needed"]["2x_over_flame_strongest"] == pytest.approx(100 * fps / 1.25 / 989e12)
    assert any(line.startswith("| kda-1p3b | 2000 (32,768) | 3000 | 1.50x |") for line in lines)


def test_e2e_runs_pairs_full_remat_and_no_remat(report, tmp_path):
    def run(name, accum, overrides, median_s):
        p = tmp_path / name
        p.write_text(json.dumps({"status": "ok", "arch": "gdn-hd128-1p3b", "step_time_median_s": median_s,
                                 "step_time_mean_s": median_s, "tokens_per_sec": 65536 / median_s,
                                 "micro_batch_per_replica": 16 // accum, "grad_accum": accum,
                                 "peak_memory_gib_max_device": 40.0, "loss_last": 8.0, "overrides": overrides}))
        return str(p)

    e2e = report.e2e_runs([run("a.json", 1, [], 2.0), run("b.json", 4, ['model.remat="none"'], 1.6)])
    assert e2e["gdn-hd128-1p3b"]["full16"]["ms"] == pytest.approx(2000.0)
    assert e2e["gdn-hd128-1p3b"]["noac4"]["ms"] == pytest.approx(1600.0)
    table = report.e2e_table(e2e, {})
    assert "| 1.25x |" in table[-1]
