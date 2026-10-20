# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ~1.3B real-hybrid presets (benchmarks/end_to_end_1p3b.md): parameter counts, layouts, configs.

``kimi-linear-1p3b`` mirrors moonshotai/Kimi-Linear-48B-A3B-Instruct and ``qwen3-next-1p3b``
Qwen/Qwen3-Next-80B-A3B-Instruct (their ``config.json`` fields are listed in the preset docstrings).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jla.config import load_config
from jla.models import build_model, get_preset, parameter_count
from jla.models.config import aux_loss_coef_from_hf
from jla.models.presets import HYBRID_1P3B_PRESETS
from jla.training.performance import estimate_flops_per_token

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = sorted((_ROOT / "benchmarks" / "end_to_end" / "hybrids_1p3b").glob("*.toml"))

# Exact counts (total, active): the untied 128,256 x 1024 embedding and head are 262.7M of each.
COUNTS = {
    "kimi-linear-1p3b": (1_196_708_544, 630_477_504),
    "qwen3-next-1p3b": (1_270_279_552, 548_334_976),
}


@pytest.mark.parametrize("name", sorted(HYBRID_1P3B_PRESETS))
def test_parameter_counts_are_exact_and_about_1p3b(name):
    counts = parameter_count(get_preset(name))
    assert (counts.total, counts.active) == COUNTS[name]
    assert 1.0e9 <= counts.total <= 1.5e9
    assert counts.embedding == 2 * 128256 * 1024


@pytest.mark.parametrize("name", sorted(HYBRID_1P3B_PRESETS))
def test_the_model_builds_with_the_counted_parameters(name):
    config = get_preset(name)
    shapes = jax.eval_shape(lambda: build_model(config, key=jax.random.key(0)))
    leaves = [x for x in jax.tree.leaves(shapes) if jnp.issubdtype(x.dtype, jnp.inexact)]
    assert sum(x.size for x in leaves) == COUNTS[name][0]


def test_kimi_linear_layout():
    c = get_preset("kimi-linear-1p3b")
    types = c.resolved_layer_types
    assert [i for i, t in enumerate(types) if t == "mla"] == [3, 7, 11, 15]  # every 4th, the last one included
    assert set(types) == {"kda", "mla"}
    assert c.resolved_mlp_types == ("dense",) + ("moe",) * 15  # first_k_dense_replace 1
    m = c.mla
    assert (m.num_heads, m.kv_lora_rank, m.q_lora_rank, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim) == (
        16, 512, None, 128, 64, 128)
    assert not m.use_rope  # mla_use_nope
    la = c.linear_attention
    assert (la.num_k_heads, la.num_v_heads, la.head_k_dim, la.head_v_dim, la.conv_size) == (16, 16, 128, 128, 4)
    moe = c.moe
    assert (moe.num_experts, moe.top_k, moe.d_ff_expert, moe.num_shared_experts, moe.router) == (32, 8, 512, 1, "sigmoid")
    assert moe.normalize_top_k and moe.routed_scaling_factor == 2.446 and not moe.shared_expert_gate
    assert moe.capacity_factor is None and moe.ragged_dot_impl == "mosaic_gpu"
    assert (c.d_model, c.d_ff, c.norm_eps, c.vocab_size, c.tie_embeddings) == (1024, 4096, 1e-5, 128256, False)


def test_qwen3_next_layout():
    c = get_preset("qwen3-next-1p3b")
    types = c.resolved_layer_types
    assert [i for i, t in enumerate(types) if t == "attention"] == [3, 7, 11, 15]  # full_attention_interval 4
    assert set(types) == {"gated_deltanet", "attention"}
    assert set(c.resolved_mlp_types) == {"moe"}  # decoder_sparse_step 1
    a = c.attention
    assert (a.num_heads, a.num_kv_heads, a.head_dim, a.output_gate, a.qk_norm) == (8, 1, 256, True, True)
    assert (a.rotary_fraction, a.rope_theta) == (0.25, 1e7)
    la = c.linear_attention
    assert (la.num_k_heads, la.num_v_heads, la.head_k_dim, la.head_v_dim, la.conv_size) == (8, 16, 128, 128, 4)
    moe = c.moe
    assert (moe.num_experts, moe.top_k, moe.d_ff_expert, moe.num_shared_experts) == (64, 10, 256, 1)
    assert moe.shared_expert_gate and moe.normalize_top_k and moe.aux_loss_coef == aux_loss_coef_from_hf(1e-3, 10)
    assert c.mtp is not None and c.mtp.depth == 1 and c.mtp.mlp_type == "moe"
    assert (c.d_model, c.vocab_size, c.tie_embeddings) == (1024, 128256, False)


def test_kimi_linear_flops_match_its_active_parameters():
    """Projection and MLP FLOPs are twice the active non-embedding parameters (MLA and KDA included)."""
    config = get_preset("kimi-linear-1p3b")
    counts = parameter_count(config)
    flops = estimate_flops_per_token(config, seq_len=4096)
    matmul = flops.attention_projections + flops.linear_attention_projections + flops.mlp
    assert matmul == pytest.approx(2 * (counts.active - counts.embedding), rel=0.02)


def test_every_hybrid_preset_has_a_config():
    assert sorted(p.stem for p in _CONFIGS) == sorted(HYBRID_1P3B_PRESETS)


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_config_follows_the_1p3b_protocol(path):
    cfg = load_config(str(path), ['data.paths=["/data/fineweb_edu_3B"]'])
    assert cfg.model.vocab_size == 128256 and cfg.model.d_model == 1024 and not cfg.model.tie_embeddings
    assert cfg.train.seq_len == 4096
    # 65,536 tokens per step on one H100; kimi-linear-1p3b needs two micro-steps to fit (71.7 GiB in one)
    micro = (8, 2) if "kimi" in path.name else (16, 1)
    assert (cfg.train.micro_batch_per_replica, cfg.train.grad_accum) == micro
    assert cfg.data.source == "cidx" and cfg.data.exact_packed_records
    assert cfg.data.document_masking and cfg.data.eos_id == 128001
    assert cfg.optim.lr == 3e-4 and cfg.optim.warmup_steps == 5 and cfg.train.z_loss == 0.0
    assert cfg.model.remat == "full"
    # The host-side train options and the FP8 attention-gradient epilogues are on only in the transformer TOMLs
    # (tests/test_bench_e2e_1p3b.py).
    assert not cfg.train.dispatch_ahead and cfg.train.update_guard == "cond" and cfg.train.grad_reduce == "micro_step"
    assert not cfg.model.attention.fp8_grad_epilogue
    # The forward-gradient cross entropy only where it cleared: faster on one H100, not slower at FSDP 8 and the
    # loss inside the noise (qwen3-next-1p3b); kimi-linear-1p3b has no interleaved 8-GPU A/B (end_to_end_1p3b.md).
    on = "qwen3-next" in path.name
    assert cfg.train.ce_grad_in_forward == on and (cfg.train.ce_impl == "auto" or not on) and not cfg.train.ce_fp8


def test_the_harness_finds_the_configs_and_sets_the_mla_implementation():
    spec = importlib.util.spec_from_file_location("bench_e2e_1p3b", _ROOT / "benchmarks" / "bench_e2e_1p3b.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    assert bench._config_path("kimi-linear-1p3b") == bench.HYBRID_CONFIG_DIR / "kimi-linear-1p3b.toml"
    model = bench._with_impl(get_preset("kimi-linear-1p3b"), "mosaic_gpu", "mosaic_gpu")
    assert model.mla.implementation == "mosaic_gpu" and model.linear_attention.implementation == "mosaic_gpu"
    model = bench._with_impl(get_preset("kimi-linear-1p3b"), "chunked", "cudnn")
    assert model.mla.implementation == "cudnn"


@pytest.mark.parametrize("name,scan,prevent_cse", [
    ("kimi-linear-tiny", True, True),  # a dense first layer: the pattern never repeats, one scan iteration
    ("tiny-hybrid", True, False),  # a repeating pattern: CSE cannot cross scan iterations
    ("tiny-hybrid", False, True),  # unscanned layers
])
def test_single_iteration_scans_prevent_cse_in_their_checkpoints(monkeypatch, name, scan, prevent_cse):
    """XLA inlines a scan of one iteration; there CSE merged the recomputed forward with the original
    and kept every layer's activations (kimi-linear-1p3b asked for 121 GiB), so such stacks checkpoint
    each block with ``prevent_cse`` like unscanned ones."""
    import equinox as eqx

    seen = []

    def checkpoint(fn, **kwargs):
        seen.append(kwargs["prevent_cse"])
        return fn

    monkeypatch.setattr(eqx, "filter_checkpoint", checkpoint)
    config = get_preset(name).replace(scan_layers=scan, remat="full")
    model = build_model(config, key=jax.random.key(0))
    model._block_fn()
    assert seen == [prevent_cse]
