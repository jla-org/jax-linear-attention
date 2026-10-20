# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 1.3B end-to-end harness (benchmarks/bench_e2e_1p3b.py) and its configs: protocol, presets, timing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from jla.config import load_config
from jla.models.presets import E2E_1P3B_PRESETS

_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "benchmarks" / "bench_e2e_1p3b.py"
_CONFIGS = sorted((_ROOT / "benchmarks" / "end_to_end" / "1p3b").glob("*.toml"))
_NO_REMAT = {"transformer-1p3b", "transformer-swa-1p3b"}  # benchmarks/fp8_1p3b.md


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_e2e_1p3b", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_preset_has_a_config():
    assert sorted(p.stem for p in _CONFIGS) == sorted(E2E_1P3B_PRESETS)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_config_follows_the_protocol(path):
    cfg = load_config(str(path), ['data.paths=["/data/fineweb_edu_3B"]'])
    assert cfg.model.vocab_size == 128256 and cfg.model.d_model == 2048 and not cfg.model.tie_embeddings
    assert cfg.train.seq_len == 4096
    assert cfg.data.source == "cidx" and cfg.data.exact_packed_records
    assert cfg.data.document_masking and cfg.data.eos_id == 128001
    assert cfg.optim.lr == 3e-4 and cfg.optim.warmup_steps == 5 and cfg.train.z_loss == 0.0
    # 65,536 tokens per GPU per step: full remat at 16 x 4096, or (where it measured faster in two interleaved
    # pairs, benchmarks/fp8_1p3b.md) no recompute + lean VJPs + unrolled layers at 4 x 4096 x 4.
    split = (cfg.train.micro_batch_per_replica, cfg.train.grad_accum)
    if path.stem in _NO_REMAT:
        assert split == (4, 4) and cfg.train.scan_unroll == 1
        assert cfg.model.remat == "none" and cfg.model.lean_vjp and not cfg.model.scan_layers
    else:
        assert split == (16, 1) and cfg.model.remat == "full"


def test_fla_shapes():
    """Spot checks of the FLA fields each preset mirrors (benchmarks/end_to_end_1p3b.md)."""
    t = E2E_1P3B_PRESETS["transformer-1p3b"]()
    assert (t.n_layers, t.d_ff, t.attention.num_heads, t.attention.num_kv_heads, t.attention.head_dim) == (24, 5632, 32, 8, 64)
    assert not t.attention.qk_norm and t.attention.sliding_window is None
    assert E2E_1P3B_PRESETS["transformer-swa-1p3b"]().attention.sliding_window == 2048
    g = E2E_1P3B_PRESETS["gdn-1p3b"]()
    la = g.linear_attention
    assert (g.n_layers, la.num_k_heads, la.head_k_dim, la.head_v_dim, la.conv_size) == (21, 6, 256, 512, 4)
    for name in ("kda-1p3b", "gdn2-1p3b", "gdn-hd128-1p3b"):
        la = E2E_1P3B_PRESETS[name]().linear_attention
        assert (la.num_k_heads, la.num_v_heads, la.head_k_dim, la.head_v_dim) == (16, 16, 128, 128)
    for name in ("mamba2-1p3b", "mamba3-1p3b"):
        m = E2E_1P3B_PRESETS[name]()
        assert (m.n_layers, m.ssm.d_state, m.ssm.expand, m.ssm.head_dim, m.ssm.n_groups, m.norm_eps) == (48, 128, 2, 64, 1, 1e-5)
        assert set(m.resolved_mlp_types) == {"none"}
    h = E2E_1P3B_PRESETS["gdn-hybrid-1p3b"]()
    assert [i for i, t in enumerate(h.resolved_layer_types) if t == "attention"] == [3, 7, 11, 15, 19, 23]


def test_with_impl_sets_every_mixer(bench):
    kda = bench._with_impl(E2E_1P3B_PRESETS["kda-hybrid-1p3b"](), "chunked", "cudnn")
    assert kda.linear_attention.implementation == "chunked" and kda.attention.implementation == "cudnn"
    mamba = bench._with_impl(E2E_1P3B_PRESETS["mamba3-1p3b"](), "mosaic_gpu", "mosaic_gpu")
    assert mamba.ssm.implementation == "mosaic_gpu" and mamba.linear_attention is None


def test_summary_times_only_the_steps_after_warmup(bench):
    tokens = 65536
    history = []
    for step in range(1, 21):
        t = 10.0 if step <= 5 else (1.0 if step % 2 else 3.0)  # warm-up steps are slow
        history.append({"step": step, "step_time": t, "tokens_per_sec": tokens / t, "mfu_percent": 40.0 / t,
                        "hfu_percent": 50.0 / t, "_devices": 8})
    s = bench._summary(history, skip=5)
    assert s["timed_steps"] == 15 and s["tokens_per_step"] == tokens
    mean = (7 * 1.0 + 8 * 3.0) / 15  # steps 6..20: 7 odd steps of 1 s, 8 even steps of 3 s
    assert s["step_time_mean_s"] == pytest.approx(mean)
    assert s["tokens_per_sec"] == pytest.approx(tokens / mean)
    assert s["tokens_per_sec_per_gpu"] == pytest.approx(tokens / mean / 8)
    assert s["mfu_percent"] == pytest.approx(40.0 / mean) and s["hfu_percent"] == pytest.approx(50.0 / mean)


def test_failure_text_has_no_local_paths(bench):
    import sysconfig

    site = sysconfig.get_paths()["purelib"]
    text = f'File "{_ROOT}/src/jla/layers/attention.py", line 9\nFile "{site}/jax/_src/core.py", line 3'
    assert bench._portable(text) == 'File "src/jla/layers/attention.py", line 9\nFile "jax/_src/core.py", line 3'
    long = "RuntimeError: compile failed\n" + "x" * 5000 + "\nValueError: the cause"
    clipped = bench._clip(long, 200)
    assert clipped.startswith("RuntimeError: compile failed") and clipped.endswith("ValueError: the cause")
    assert len(clipped) < 300 and bench._clip("short", 200) == "short"


def test_source_fingerprint_is_stable(bench):
    first, second = bench._source_sha256(), bench._source_sha256()
    assert first == second and len(first) == 64 and int(first, 16) >= 0


def test_allocator_auto_keeps_xla_default(bench):
    choose = bench._allocator_choice
    assert choose("auto") is None  # XLA's default (BFC) on every mesh, as jla train
    assert choose("bfc") == "bfc" and choose("cuda_async") == "cuda_async"


def test_watchdog_turns_a_stall_into_a_failure(tmp_path):
    """A silent run (e.g. seven GPUs waiting in a collective for an out-of-memory GPU 0) fails with exit code 3."""
    import subprocess
    import sys

    marker = tmp_path / "stalled.txt"
    code = (
        "import importlib.util, pathlib, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('b', {str(_PATH)!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "hb = b._Heartbeat()\n"
        f"b._start_watchdog(1.0, hb, lambda idle: pathlib.Path({str(marker)!r}).write_text(f'{{idle:.3f}}'))\n"
        "time.sleep(30)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 3, proc.stderr
    assert float(marker.read_text()) >= 1.0  # silent for at least the timeout


@pytest.mark.parametrize("name", ["transformer-1p3b", "transformer-swa-1p3b"])
def test_transformer_configs_use_the_fused_attention_backward(name):
    """benchmarks/attention.md, "Fused backward end to end": faster in bf16 for both transformer presets."""
    cfg = load_config(str(_ROOT / "benchmarks" / "end_to_end" / "1p3b" / f"{name}.toml"),
                      ['data.paths=["/data/fineweb_edu_3B"]'])
    assert cfg.model.attention.mosaic_gpu_backward == "fused"
    assert cfg.model.attention.num_heads == 32 and cfg.model.attention.head_dim == 64


# benchmarks/cross_entropy.md, "Training at 1.3B": faster on one H100, not slower on 8 GPUs (FSDP 8),
# and the 20-step loss inside the noise of repeated runs. The other configs were not measured on 8 GPUs.
_CE_GRAD_IN_FORWARD = {"transformer-1p3b", "gdn-hd128-1p3b"}
# Faster on one H100 too, but kept off (benchmarks/multi_gpu.md, "Train options for several GPUs"):
# the 8-GPU half of the rule did not clear. gdn2-1p3b (its TOML): faster on 1 and 8 GPUs, but its
# bf16 losses with the CE kernels end below the XLA cross entropy's.
_CE_GRAD_IN_FORWARD_NOT_CLEARED = {
    "kda-1p3b": "1.057x on one GPU (FP8), 0.954x in its one 8-GPU pair",
    "transformer-swa-1p3b": "1.119x on one GPU (FP8), no 8-GPU pair",
    "mamba2-1p3b": "1.028x on one GPU (FP8), loss moved more than its run-to-run spread, no 8-GPU runs",
    "gdn2-1p3b": "1.043x (bf16) and 1.052x (FP8) on one GPU, 1.052x on 8 (FP8), but with the CE kernels the "
                 "bf16 losses end about 0.02 below the XLA cross entropy's (with or without the option)",
}


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_forward_gradient_cross_entropy_where_it_measured_faster(path):
    cfg = load_config(str(path), ['data.paths=["/data/fineweb_edu_3B"]'])
    assert cfg.train.ce_grad_in_forward == (path.stem in _CE_GRAD_IN_FORWARD)
    if cfg.train.ce_grad_in_forward:
        assert cfg.train.ce_impl == "auto" and not cfg.train.ce_fp8
    if path.stem in _CE_GRAD_IN_FORWARD_NOT_CLEARED:
        assert not cfg.train.ce_grad_in_forward, _CE_GRAD_IN_FORWARD_NOT_CLEARED[path.stem]


def test_forward_gradient_cross_entropy_decisions_name_existing_tomls():
    stems = {p.stem for p in _CONFIGS}
    assert _CE_GRAD_IN_FORWARD <= stems and set(_CE_GRAD_IN_FORWARD_NOT_CLEARED) <= stems
    assert not _CE_GRAD_IN_FORWARD & set(_CE_GRAD_IN_FORWARD_NOT_CLEARED)


def test_summary_of_a_dispatch_ahead_run_leaves_out_the_last_step(bench):
    # With train.dispatch_ahead the host reads a step's metrics late in the next step, so each timed step is one
    # period (1.2 s here) except the last, timed from that late read to its own end (0.03 s).
    tokens = 65536
    history = [{"step": step, "step_time": 10.0 if step <= 5 else (0.03 if step == 20 else 1.2),
                "tokens_per_sec": 0.0, "mfu_percent": 0.0, "hfu_percent": 0.0, "_devices": 8} for step in range(1, 21)]
    history[0]["tokens_per_sec"] = tokens / 10.0
    s = bench._summary(history, skip=5, drop_last=True)
    assert s["timed_steps"] == 14 and s["step_time_mean_s"] == pytest.approx(1.2)
    assert s["tokens_per_sec"] == pytest.approx(tokens / 1.2)
    assert bench._summary(history, skip=5)["step_time_mean_s"] == pytest.approx((14 * 1.2 + 0.03) / 15)


# benchmarks/multi_gpu.md and benchmarks/fp8_1p3b.md: the transformer's
# host-side options, its one gradient reduction per step (8-GPU check, "Train options for several GPUs") and both
# transformers' FP8 attention-gradient epilogues. train.grad_reduce = "step" refuses the "auto" attention on GPUs,
# so a TOML that turns it on also sets the attention to "mosaic_gpu".
_DISPATCH_AHEAD_AND_SELECT = {"transformer-1p3b"}
_GRAD_REDUCE_STEP = {"transformer-1p3b"}
_FP8_GRAD_EPILOGUE = {"transformer-1p3b", "transformer-swa-1p3b"}


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_host_options_and_attention_epilogues_where_they_measured_faster(path):
    cfg = load_config(str(path), ['data.paths=["/data/fineweb_edu_3B"]'])
    on = path.stem in _DISPATCH_AHEAD_AND_SELECT
    assert cfg.train.dispatch_ahead == on and cfg.train.update_guard == ("select" if on else "cond")
    step = path.stem in _GRAD_REDUCE_STEP
    assert cfg.train.grad_reduce == ("step" if step else "micro_step")
    if step:
        assert cfg.model.attention.implementation == "mosaic_gpu"
    assert cfg.model.attention.fp8_grad_epilogue == (path.stem in _FP8_GRAD_EPILOGUE)
    assert cfg.model.attention.persistent_forward is None
    assert _DISPATCH_AHEAD_AND_SELECT | _GRAD_REDUCE_STEP | _FP8_GRAD_EPILOGUE <= {p.stem for p in _CONFIGS}
