# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Named model configurations.

* ``transformer-*``: Llama / Qwen3-style decoders (GQA, QK-norm, SwiGLU, RoPE);
  ``transformer-1b`` has Llama-3.2-1B shapes;
* ``gdn-hybrid-125m``: Qwen3-Next-like, three Gated DeltaNet layers per gated
  attention layer (grouped value heads, partial RoPE, sigmoid output gate);
* ``kda-hybrid-125m``: Kimi-Linear-like, three KDA layers per NoPE attention layer;
* ``gla-125m``, ``deltanet-125m``: all-linear stacks with FLA's layer defaults;
* ``moe-125m``: 16 SwiGLU experts, top-2 plus one shared expert (~125M active);
* ``mamba2-130m``, ``mamba3-130m``: state-spaces/mamba's 130M shape (24 mixer-only
  blocks, d_state 128, expand 2, head_dim 64, one B/C group; Mamba-3 SISO with
  rope_fraction 0.5);
* ``gdn2-125m``: Gated DeltaNet-2 and MLP blocks; ``gdn2-hybrid-125m``: the paper's
  hybrid cell (GDN-2, MLP, 2K sliding-window attention, MLP);
* ``tiny``, ``tiny-hybrid``, ``tiny-moe``, ``tiny-mamba2``, ``tiny-mamba3``,
  ``tiny-gdn2``: small models for tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from jla.models.config import (
    AttentionConfig,
    ForgettingAttentionConfig,
    LinearAttentionConfig,
    MLAConfig,
    ModelConfig,
    MoEConfig,
    MTPConfig,
    PathAttentionConfig,
    SparseAttentionConfig,
    SSMConfig,
    hybrid_pattern,
)
from jla.models.config import aux_loss_coef_from_hf

__all__ = ["get_preset", "list_presets"]

_VOCAB = 50304  # GPT-2/NeoX vocabulary padded to a multiple of 128
# Presets with attention (or MLA) and gated_deltanet or kda layers keep identity short convs: in the 125M hybrids FLA's
# N(0, 0.02) convs tied (GDN) or lost (KDA).
_GDN_KDA_CONV_INIT = "identity"  # 125M FineWeb-Edu A/B, 7 seeds each: benchmarks/init_parity.md
# ... except the all-linear stacks (gdn-1p3b, gdn-hd128-1p3b, kda-1p3b): no attention layers, and there FLA's convs
# won (the 125M all-linear runs on the same page). None: the default init_scheme draws N(0, 0.02).
_ALL_LINEAR_CONV_INIT = None
# The tiny test presets with those layers (tiny-hybrid, tiny-qwen3-next, kimi-linear-tiny) leave conv_init unset, so
# the default init_scheme draws FLA's N(0, 0.02) convs: every older tap is nonzero at init and the model-level tests
# (decode vs forward, packed documents) exercise the conv caches and segment resets. Guard: tests/test_init_parity.py.
_TEST_CONV_INIT = None


def _tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
    )


def _tiny_hybrid() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=4, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, output_gate=True,
                                  rotary_fraction=0.5),
        linear_attention=LinearAttentionConfig(num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                               chunk_size=16, conv_init=_TEST_CONV_INIT),
        layer_types=hybrid_pattern(4, "gated_deltanet", 2),
        mtp=MTPConfig(depth=2),
    )


def _tiny_moe() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
        moe=MoEConfig(num_experts=4, top_k=2, d_ff_expert=32, num_shared_experts=1, z_loss_coef=1e-3),
    )


def _tiny_ssm(kind: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=0,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by Mamba stacks
        ssm=SSMConfig(d_state=16, head_dim=16, n_groups=2, chunk_size=16),
        layer_types=(kind,) * 2, mlp_types=("none",) * 2, norm_eps=1e-5,
    )


def _tiny_gdn2() -> ModelConfig:
    """The Gated DeltaNet-2 hybrid cell (GDN-2, MLP, sliding-window attention, MLP), twice."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=4, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, sliding_window=8),
        linear_attention=LinearAttentionConfig(num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                               chunk_size=16),
        layer_types=hybrid_pattern(4, "gdn2", 2),
    )


def _mamba(kind: str) -> ModelConfig:
    """state-spaces/mamba's 130M shape: 24 Mamba blocks without MLPs, d_state 128, expand 2, head_dim 64."""
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=24, d_ff=0,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),  # unused by Mamba stacks
        ssm=SSMConfig(), layer_types=(kind,) * 24, mlp_types=("none",) * 24, norm_eps=1e-5,
    )


def _gdn2(hybrid: bool) -> ModelConfig:
    """Gated DeltaNet-2 with the paper's head shape (d_k = d_v = 128); the hybrid adds 2K-window attention."""
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048 if hybrid else 1536,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64, sliding_window=2048),
        linear_attention=LinearAttentionConfig(num_k_heads=6, num_v_heads=6, head_k_dim=128, head_v_dim=128),
        layer_types=hybrid_pattern(12, "gdn2", 2) if hybrid else ("gdn2",) * 12,
    )


def _transformer(d: int, layers: int, heads: int, kv_heads: int, head_dim: int, d_ff: int,
                 vocab: int = _VOCAB, theta: float = 10_000.0) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab, d_model=d, n_layers=layers, d_ff=d_ff,
        attention=AttentionConfig(num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, rope_theta=theta),
    )


def _gdn_hybrid() -> ModelConfig:
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=128, output_gate=True,
                                  rotary_fraction=0.25, rope_theta=10_000_000.0),
        linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        layer_types=hybrid_pattern(12, "gated_deltanet", 4),
    )


def _kda_hybrid() -> ModelConfig:
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=128, use_rope=False),
        linear_attention=LinearAttentionConfig(num_k_heads=6, num_v_heads=6, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        layer_types=hybrid_pattern(12, "kda", 4),
    )


def _all_linear(kind: str, la: LinearAttentionConfig) -> ModelConfig:
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),  # unused by all-linear stacks
        linear_attention=la, layer_types=(kind,) * 12,
    )


def _moe() -> ModelConfig:
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),
        moe=MoEConfig(num_experts=16, top_k=2, d_ff_expert=768, num_shared_experts=1, capacity_factor=1.25,
                      z_loss_coef=1e-3),
    )


def _tiny_deltaproduct() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
        linear_attention=LinearAttentionConfig(num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                               chunk_size=16, num_householder=2, allow_neg_eigval=True),
        layer_types=hybrid_pattern(2, "gated_deltaproduct", 2),
    )


def _tiny_mamba3_mimo() -> ModelConfig:
    """``tiny-mamba3`` with Mamba-3 MIMO of rank 2."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=0,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by Mamba stacks
        ssm=SSMConfig(d_state=16, head_dim=16, n_groups=2, chunk_size=16, mimo_rank=2),
        layer_types=("mamba3",) * 2, mlp_types=("none",) * 2, norm_eps=1e-5,
    )


def _mamba3_mimo_130m() -> ModelConfig:
    """``mamba3-130m`` as MIMO of rank 4 with chunk 16 (state-spaces/mamba: ``mimo_rank=4``, chunk ``64 / R``)."""
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=24, d_ff=0,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),  # unused by Mamba stacks
        ssm=SSMConfig(mimo_rank=4, chunk_size=16), layer_types=("mamba3",) * 24, mlp_types=("none",) * 24,
        norm_eps=1e-5,
    )


_PRESETS: dict[str, Callable[[], ModelConfig]] = {
    "tiny": _tiny,
    "tiny-hybrid": _tiny_hybrid,
    "tiny-moe": _tiny_moe,
    "tiny-mamba2": lambda: _tiny_ssm("mamba2"),
    "tiny-mamba3": lambda: _tiny_ssm("mamba3"),
    "tiny-gdn2": _tiny_gdn2,
    "mamba2-130m": lambda: _mamba("mamba2"),
    "mamba3-130m": lambda: _mamba("mamba3"),
    "gdn2-125m": lambda: _gdn2(hybrid=False),
    "gdn2-hybrid-125m": lambda: _gdn2(hybrid=True),
    "transformer-125m": lambda: _transformer(768, 12, 12, 4, 64, 2304),
    "transformer-350m": lambda: _transformer(1024, 24, 16, 8, 64, 2816),
    "transformer-1b": lambda: _transformer(2048, 16, 32, 8, 64, 8192, vocab=128256, theta=500_000.0),
    "gdn-hybrid-125m": _gdn_hybrid,
    "kda-hybrid-125m": _kda_hybrid,
    "gla-125m": lambda: _all_linear("gla", LinearAttentionConfig(
        num_k_heads=6, num_v_heads=6, head_k_dim=64, head_v_dim=128, conv_size=0, qk_l2norm=False,
        gate_low_rank_dim=16)),
    "deltanet-125m": lambda: _all_linear("deltanet", LinearAttentionConfig(
        num_k_heads=6, num_v_heads=6, head_k_dim=128, head_v_dim=128, output_gate=False)),
    "moe-125m": _moe,
    "tiny-deltaproduct": _tiny_deltaproduct,
    "gated-deltaproduct-125m": lambda: _all_linear("gated_deltaproduct", LinearAttentionConfig(
        num_k_heads=6, num_v_heads=6, head_k_dim=128, head_v_dim=128, num_householder=2, allow_neg_eigval=True)),
    "tiny-mamba3-mimo": _tiny_mamba3_mimo,
    "mamba3-mimo-130m": _mamba3_mimo_130m,
    "tiny-fox": lambda: _fox_preset(tiny=True),
    "fox-125m": lambda: _fox_preset(tiny=False),
    "tiny-loglinear-mamba2": lambda: _loglinear_mamba2(_tiny_ssm("mamba2"), levels=10),
    "loglinear-mamba2-130m": lambda: _loglinear_mamba2(_mamba("mamba2"), levels=13),
}


def _fox_preset(tiny: bool) -> ModelConfig:
    """Forgetting Transformer (FoX) Pro: QK-norm, output gate and norm, KV shift, no RoPE."""
    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=2, d_ff=128,
            attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by FoX stacks
            forgetting_attention=ForgettingAttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
            layer_types=("fox",) * 2,
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=1792,
        attention=AttentionConfig(num_heads=12, num_kv_heads=12, head_dim=64),  # unused by FoX stacks
        forgetting_attention=ForgettingAttentionConfig(num_heads=12, num_kv_heads=12, head_dim=64),
        layer_types=("fox",) * 12,
    )


def _loglinear_mamba2(config: ModelConfig, levels: int) -> ModelConfig:
    """Log-linear Mamba-2 (Guo et al. 2025): ``levels`` Fenwick levels cover ``T <= 2^(levels - 1)``."""
    return config.replace(ssm=dataclasses.replace(config.ssm, log_linear_levels=levels))


def _mlstm(d: int, layers: int, heads: int, d_ff: int, *, vocab: int = _VOCAB, chunk: int = 64) -> ModelConfig:
    """xLSTM-7B layout (arXiv 2503.13427): mLSTM layers with qk = d / 2, v = d, no conv, SwiGLU MLPs."""
    la = LinearAttentionConfig(num_k_heads=heads, num_v_heads=heads, head_k_dim=d // 2 // heads,
                               head_v_dim=d // heads, conv_size=0, qk_l2norm=False, chunk_size=chunk)
    return ModelConfig(
        vocab_size=vocab, d_model=d, n_layers=layers, d_ff=d_ff,
        attention=AttentionConfig(num_heads=heads, num_kv_heads=heads, head_dim=d // heads),  # unused
        linear_attention=la, layer_types=("mlstm",) * layers,
    )


_PRESETS["tiny-mlstm"] = lambda: _mlstm(64, 2, 2, 128, vocab=512, chunk=16)
_PRESETS["xlstm-125m"] = lambda: _mlstm(768, 12, 4, 2048)


def _tiny_moba() -> ModelConfig:
    """``tiny`` with MoBA attention: blocks of 4 tokens, each query sees its own block and one more."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, moba_block_size=4, moba_top_k=2),
    )


def _moba_hybrid() -> ModelConfig:
    """``gdn-hybrid-125m`` with MoBA attention layers (block 512, top-3: the MoBA paper's setting)."""
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=128, output_gate=True,
                                  rotary_fraction=0.25, rope_theta=10_000_000.0, moba_block_size=512,
                                  moba_top_k=3),
        linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=8, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        layer_types=hybrid_pattern(12, "gated_deltanet", 4),
    )


_PRESETS["tiny-moba"] = _tiny_moba
_PRESETS["moba-hybrid-125m"] = _moba_hybrid


def _tiny_sparse(kind: str) -> ModelConfig:
    """Two NSA or DSA blocks small enough that the sparse selection is active at short lengths."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
        sparse_attention=SparseAttentionConfig(block_size=4, block_count=4, window_size=8, top_k=8,
                                               index_heads=2, index_head_dim=16),
        layer_types=(kind,) * 2,
    )


_PRESETS["tiny-nsa"] = lambda: _tiny_sparse("nsa")
_PRESETS["tiny-dsa"] = lambda: _tiny_sparse("dsa")


def _tiny_ttt() -> ModelConfig:
    """TTT-Linear (mini-batch 8) and attention layers, for tests."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
        linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                               ttt_mini_batch_size=8),
        layer_types=("ttt_linear", "attention"),
    )


# TTT-Linear (arXiv 2407.04620): the paper's 125M head shape (12 heads of 64, mini-batch 16).
_PRESETS["tiny-ttt"] = _tiny_ttt
_PRESETS["ttt-linear-125m"] = lambda: _all_linear("ttt_linear", LinearAttentionConfig(
    num_k_heads=12, num_v_heads=12, head_k_dim=64, head_v_dim=64, ttt_mini_batch_size=16))


def get_preset(name: str) -> ModelConfig:
    """The :class:`ModelConfig` registered as ``name`` (see :func:`list_presets`)."""
    if name not in _PRESETS:
        raise KeyError(f"unknown model preset {name!r}; available: {', '.join(list_presets())}")
    return _PRESETS[name]()


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def _tiny_mamba1() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=0,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by Mamba stacks
        ssm=SSMConfig(d_state=16, head_dim=16, chunk_size=16),
        layer_types=("mamba1",) * 2, mlp_types=("none",) * 2, norm_eps=1e-5,
    )


def _mamba1_130m() -> ModelConfig:
    """state-spaces/mamba-130m: 24 Mamba-1 blocks without MLPs, d_state 16, expand 2, dt_rank 48."""
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=24, d_ff=0,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),  # unused by Mamba stacks
        ssm=SSMConfig(d_state=16), layer_types=("mamba1",) * 24, mlp_types=("none",) * 24, norm_eps=1e-5,
    )


def _tiny_jamba() -> ModelConfig:
    """Jamba's block: 8 layers, attention (no RoPE) at index 4 (Mamba-1 : attention = 7 : 1), MoE every other layer."""
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=8, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, use_rope=False),
        moe=MoEConfig(num_experts=4, top_k=2, d_ff_expert=32),
        ssm=SSMConfig(d_state=16, head_dim=16, chunk_size=16),
        layer_types=tuple("attention" if i % 8 == 4 else "mamba1" for i in range(8)),
        mlp_types=tuple("moe" if i % 2 else "dense" for i in range(8)),
    )


_PRESETS.update({
    "tiny-mamba1": _tiny_mamba1,
    "mamba1-130m": _mamba1_130m,
    "tiny-jamba": _tiny_jamba,
})


# Multi-head latent attention presets ----------------------------------------------------------
# * ``tiny-mla``: two MLA blocks (low-rank q, compressed kv latent, shared RoPE key) for tests;
# * ``deepseek-v3-tiny``: DeepSeek-V3's layout at toy width: MLA everywhere (q_lora_rank =
#   3 x kv_lora_rank, nope : rope : v head dims = 2 : 1 : 2), a leading dense layer, then
#   sigmoid-routed MoE with one shared expert (dense d_ff = 9 x d_ff_expert), MTP depth 1;
# * ``kimi-linear-tiny`` / ``kimi-linear-1b``: Kimi Linear's layout: three KDA layers per MLA
#   layer, NoPE on the MLA layers, full-rank queries, the first layer dense and sigmoid-routed
#   MoE with one shared expert after it. ``kimi-linear-1b`` has ~0.9B parameters (~0.25B active)
#   with Kimi Linear's head shapes (KDA 128 x 128, MLA nope 128 / rope 64 / v 128, rank 512).


def _tiny_mla() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by MLA stacks
        mla=MLAConfig(num_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
                      v_head_dim=16),
        layer_types=("mla",) * 2,
    )


def _deepseek_v3_tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=4, d_ff=144,
        attention=AttentionConfig(num_heads=4, num_kv_heads=4, head_dim=16),  # the MTP refiner block
        mla=MLAConfig(num_heads=4, q_lora_rank=48, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
                      v_head_dim=16),
        moe=MoEConfig(num_experts=8, top_k=2, d_ff_expert=16, num_shared_experts=1, router="sigmoid"),
        layer_types=("mla",) * 4, mlp_types=("dense",) + ("moe",) * 3,
        mtp=MTPConfig(depth=1),
    )


def _kimi_linear(*, d_model: int, n_layers: int, vocab: int, d_ff: int, heads: int, head_dim: int,
                 mla: MLAConfig, moe: MoEConfig, chunk_size: int = 64,
                 conv_init: str | None = _GDN_KDA_CONV_INIT) -> ModelConfig:
    layer_types = tuple("mla" if t == "attention" else t for t in hybrid_pattern(n_layers, "kda", 4))
    return ModelConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
        attention=AttentionConfig(num_heads=heads, num_kv_heads=heads, head_dim=head_dim, use_rope=False),
        linear_attention=LinearAttentionConfig(num_k_heads=heads, num_v_heads=heads, head_k_dim=head_dim,
                                               head_v_dim=head_dim, chunk_size=chunk_size,
                                               conv_init=conv_init),
        mla=mla, moe=moe, layer_types=layer_types, mlp_types=("dense",) + ("moe",) * (n_layers - 1),
    )


def _kimi_linear_tiny() -> ModelConfig:
    return _kimi_linear(
        d_model=64, n_layers=4, vocab=512, d_ff=256, heads=2, head_dim=16, chunk_size=16,
        mla=MLAConfig(num_heads=4, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                      use_rope=False),
        moe=MoEConfig(num_experts=8, top_k=2, d_ff_expert=32, num_shared_experts=1, router="sigmoid"),
        conv_init=_TEST_CONV_INIT,
    )


def _kimi_linear_1b() -> ModelConfig:
    return _kimi_linear(
        d_model=1024, n_layers=16, vocab=_VOCAB, d_ff=4096, heads=8, head_dim=128,
        mla=MLAConfig(num_heads=8, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                      use_rope=False),
        moe=MoEConfig(num_experts=32, top_k=4, d_ff_expert=512, num_shared_experts=1, router="sigmoid",
                      capacity_factor=1.25),
    )


_PRESETS.update({
    "tiny-mla": _tiny_mla,
    "deepseek-v3-tiny": _deepseek_v3_tiny,
    "kimi-linear-tiny": _kimi_linear_tiny,
    "kimi-linear-1b": _kimi_linear_1b,
})


# PaTH attention (arXiv 2505.16381): softmax attention with Householder position encoding.
def _path(tiny: bool) -> ModelConfig:
    if tiny:  # grouped KV heads, forget gate, QK-norm and a short chunk exercise every code path
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=2, d_ff=128,
            attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by PaTH stacks
            path_attention=PathAttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16, w_low_rank_dim=8,
                                               forget_gate=True, qk_norm=True, chunk_size=16),
            layer_types=("path",) * 2,
        )
    return ModelConfig(  # FLA's PaTHAttention defaults (MHA, rank-32 w, 3-tap w conv) at 125M
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=2048,
        attention=AttentionConfig(num_heads=12, num_kv_heads=12, head_dim=64),  # unused by PaTH stacks
        path_attention=PathAttentionConfig(num_heads=12, num_kv_heads=12, head_dim=64),
        layer_types=("path",) * 12,
    )


_PRESETS["tiny-path"] = lambda: _path(tiny=True)
_PRESETS["path-125m"] = lambda: _path(tiny=False)


# RWKV-6 and HGRN2 (jla.layers.rwkv6). The RWKV-6 stacks use RWKV's channel mix
# (mlp_type "rwkv_cmix", FLA's d_ff = 3.5 d rounded up to 32) in RMSNorm pre-norm blocks;
# HGRN2 uses the dense SwiGLU MLP, like FLA's HGRN2 models.
def _tiny_rwkv6(kind: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, d_ff=224 if kind == "rwkv6" else 128,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),
        linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                               conv_size=0, output_gate=kind == "rwkv6", qk_l2norm=False,
                                               gate_low_rank_dim=16 if kind == "rwkv6" else None,
                                               chunk_size=16),
        layer_types=(kind,) * 2,
        mlp_types=("rwkv_cmix",) * 2 if kind == "rwkv6" else None,
    )


_PRESETS.update({
    "tiny-rwkv6": lambda: _tiny_rwkv6("rwkv6"),
    "tiny-hgrn2": lambda: _tiny_rwkv6("hgrn2"),
    "rwkv6-0.1b": lambda: _all_linear("rwkv6", LinearAttentionConfig(
        num_k_heads=12, num_v_heads=12, head_k_dim=64, head_v_dim=64, conv_size=0, qk_l2norm=False,
        gate_low_rank_dim=64)).replace(d_ff=2688, mlp_types=("rwkv_cmix",) * 12),
    "hgrn2-125m": lambda: _all_linear("hgrn2", LinearAttentionConfig(
        num_k_heads=6, num_v_heads=6, head_k_dim=128, head_v_dim=128, conv_size=0, output_gate=False,
        qk_l2norm=False)),
})


# Production architecture presets ---------------------------------------------------------------
# Each has a ``tiny-*`` test variant and a ~125M-1B training variant that keeps the published
# per-head shapes, layer ratios and mixer settings of the model's Hugging Face ``config.json``
# (cited per preset) and uses this template's 50304-token vocabulary.


def _nemotron_pattern(pattern: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Blocks of a Nemotron-H ``hybrid_override_pattern`` (``M`` Mamba-2, ``*`` attention, ``-`` MLP).

    Every Nemotron-H layer is one pre-norm residual sublayer; a ``-`` joins the mixer before
    it as that block's channel mixer and a mixer followed by another mixer gets none, so
    the blocks compute exactly the published layer sequence.
    """
    layers: list[str] = []
    mlps: list[str] = []
    for symbol in pattern:
        if symbol in "M*":
            layers.append("mamba2" if symbol == "M" else "attention")
            mlps.append("none")
        elif symbol == "-" and layers and mlps[-1] == "none":
            mlps[-1] = "relu2"
        else:
            raise ValueError(f"unsupported Nemotron-H pattern {pattern!r}")
    return tuple(layers), tuple(mlps)


# nvidia/Nemotron-H-8B-Base-8K config.json: 52 layers = 24 Mamba-2, 4 attention, 24 relu2 MLPs.
NEMOTRON_H_8B_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"


def _qwen3_next(tiny: bool) -> ModelConfig:
    """Qwen3-Next (Qwen/Qwen3-Next-80B-A3B-Instruct ``config.json``).

    ``full_attention_interval`` 4 (three Gated DeltaNet layers per gated attention layer),
    attention 16 heads / 2 KV heads of 256 with a sigmoid output gate, QK-norm,
    ``partial_rotary_factor`` 0.25 and ``rope_theta`` 1e7; Gated DeltaNet with 16 key heads,
    32 value heads, head dims 128 and conv 4; an MoE in every layer (``decoder_sparse_step`` 1)
    with 512 experts of width 512, top-10, normalized top-k, one shared expert of width 512,
    ``router_aux_loss_coef`` 1e-3 (HF's balance loss divides the expert counts by tokens, ours by
    routes, so ``aux_loss_coef`` is ``top_k`` x 1e-3: 1e-2 here, 2e-3 in the top-2 tiny preset;
    :func:`~jla.models.config.aux_loss_coef_from_hf`); untied embeddings. The training variant halves ``hidden_size``
    (2048 -> 1024) and scales every width with it (8 / 1 attention heads of 256, 8 / 16 linear
    heads of 128, experts of 256), with 12 layers and 64 experts (top-10).

    The shared expert is scaled by the sigmoid ``shared_expert_gate`` (``Qwen3NextSparseMoeBlock``),
    and one multi-token-prediction layer follows the trunk (the checkpoint's ``mtp.*`` weights;
    the config sets no ``num_nextn_predict_layers``, so vLLM's ``Qwen3NextMultiTokenPredictor``
    uses its default of 1): ``fc([norm(embed(x_{t+1})); norm(h_t)])``, a gated full-attention
    decoder layer with the MoE, a final norm and the shared LM head (``MTPConfig`` kind
    ``"deepseek"`` with ``mlp_type="moe"``; the MTP loss weight 0.3 is this template's default).
    """
    mtp = MTPConfig(depth=1, kind="deepseek", mlp_type="moe")
    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=4, d_ff=0, tie_embeddings=False,
            attention=AttentionConfig(num_heads=4, num_kv_heads=1, head_dim=32, output_gate=True,
                                      rotary_fraction=0.25, rope_theta=10_000_000.0),
            linear_attention=LinearAttentionConfig(num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                                   chunk_size=16, conv_init=_TEST_CONV_INIT),
            moe=MoEConfig(num_experts=8, top_k=2, d_ff_expert=16, num_shared_experts=1,
                          aux_loss_coef=aux_loss_coef_from_hf(1e-3, top_k=2), shared_expert_gate=True),
            layer_types=hybrid_pattern(4, "gated_deltanet", 4), mtp=mtp,
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=1024, n_layers=12, d_ff=0, tie_embeddings=False,
        attention=AttentionConfig(num_heads=8, num_kv_heads=1, head_dim=256, output_gate=True,
                                  rotary_fraction=0.25, rope_theta=10_000_000.0),
        linear_attention=LinearAttentionConfig(num_k_heads=8, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        moe=MoEConfig(num_experts=64, top_k=10, d_ff_expert=256, num_shared_experts=1,
                      aux_loss_coef=aux_loss_coef_from_hf(1e-3, top_k=10), shared_expert_gate=True),
        layer_types=hybrid_pattern(12, "gated_deltanet", 4), mtp=mtp,
    )


def _nemotron_h(tiny: bool) -> ModelConfig:
    """Nemotron-H (nvidia/Nemotron-H-8B-Base-8K ``config.json``).

    ``hybrid_override_pattern`` ``NEMOTRON_H_8B_PATTERN`` (4 of 52 layers, about 8%, are
    attention); Mamba-2 with ``expand`` 2, head dim 64, ``ssm_state_size`` 128, ``n_groups`` 8,
    conv 4; attention 32 heads / 8 KV heads of 128 with no rotary embedding (the config has
    no RoPE settings) and no QK-norm; ``mlp_hidden_act`` relu2 with ``intermediate_size`` =
    5.25 x ``hidden_size``; RMSNorm eps 1e-5; untied embeddings. The training variant keeps
    the full 52-layer pattern at ``hidden_size`` 1024 (8 / 2 attention heads of 128, 32 Mamba
    heads in 8 groups, MLP width 5376).
    """
    if tiny:
        layers, mlps = _nemotron_pattern("M-M*-")
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=len(layers), d_ff=336, tie_embeddings=False, norm_eps=1e-5,
            attention=AttentionConfig(num_heads=4, num_kv_heads=1, head_dim=16, qk_norm=False, use_rope=False),
            ssm=SSMConfig(d_state=16, head_dim=16, n_groups=2, chunk_size=16),
            layer_types=layers, mlp_types=mlps,
        )
    layers, mlps = _nemotron_pattern(NEMOTRON_H_8B_PATTERN)
    return ModelConfig(
        vocab_size=_VOCAB, d_model=1024, n_layers=len(layers), d_ff=5376, tie_embeddings=False, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=8, num_kv_heads=2, head_dim=128, qk_norm=False, use_rope=False),
        ssm=SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=8),
        layer_types=layers, mlp_types=mlps,
    )


def _granite_4_h(tiny: bool) -> ModelConfig:
    """Granite-4.0-H (ibm-granite/granite-4.0-h-tiny ``config.json``).

    40 layers with attention at 1-based layers 6, 16, 26, 36 (a period of ten: five Mamba-2,
    one attention, four Mamba-2); Mamba-2 with ``expand`` 2, head dim 64, d_state 128, one
    group, conv 4; attention 12 heads / 4 KV heads of 128, ``position_embedding_type`` nope,
    ``attention_multiplier`` 0.0078125 as the softmax scale, no QK-norm; an MoE in every layer
    with 64 experts of width 512, top-6, and a shared SwiGLU of width 1024 (two always-on
    experts of 512 sum to exactly that); RMSNorm eps 1e-5; tied embeddings. The training
    variant is one ten-layer period at ``hidden_size`` 768 (6 / 2 heads of 128, 24 Mamba heads,
    experts of 256 and a shared width of 512).
    """
    period = ("mamba2",) * 5 + ("attention",) + ("mamba2",) * 4
    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=4, d_ff=0, norm_eps=1e-5,
            attention=AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=16, qk_norm=False, use_rope=False,
                                      softmax_scale=1.0 / 16),
            ssm=SSMConfig(d_state=16, head_dim=16, n_groups=1, chunk_size=16),
            moe=MoEConfig(num_experts=8, top_k=3, d_ff_expert=16, num_shared_experts=2),
            layer_types=("mamba2", "attention", "mamba2", "mamba2"),
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=10, d_ff=0, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=6, num_kv_heads=2, head_dim=128, qk_norm=False, use_rope=False,
                                  softmax_scale=0.0078125),
        ssm=SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1),
        moe=MoEConfig(num_experts=64, top_k=6, d_ff_expert=256, num_shared_experts=2),
        layer_types=period,
    )


def _gpt_oss(tiny: bool) -> ModelConfig:
    """GPT-OSS (openai/gpt-oss-20b ``config.json``).

    ``layer_types`` alternate ``sliding_attention`` (``sliding_window`` 128) and
    ``full_attention``, starting with a sliding layer; attention 64 heads / 8 KV heads of 64
    with attention sinks (a learned per-head logit in the softmax denominator), no QK-norm,
    ``rope_theta`` 150000; an MoE in every layer with 32 experts of width ``intermediate_size``
    = ``hidden_size``, ``experts_per_token`` 4 (softmax over the selected logits); RMSNorm eps
    1e-5; untied embeddings. The training variant has 12 layers at ``hidden_size`` 768
    (16 / 2 heads of 64, 32 experts of 768, top-4).
    """
    def attn(window: int | None, heads: int, kv_heads: int, head_dim: int) -> AttentionConfig:
        return AttentionConfig(num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, qk_norm=False,
                               rope_theta=150_000.0, sinks=True, sliding_window=window)

    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=4, d_ff=0, tie_embeddings=False, norm_eps=1e-5,
            attention=attn(None, 4, 1, 16), local_attention=attn(4, 4, 1, 16),
            moe=MoEConfig(num_experts=8, top_k=4, d_ff_expert=64),
            layer_types=("local_attention", "attention") * 2,
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=0, tie_embeddings=False, norm_eps=1e-5,
        attention=attn(None, 16, 2, 64), local_attention=attn(128, 16, 2, 64),
        moe=MoEConfig(num_experts=32, top_k=4, d_ff_expert=768),
        layer_types=("local_attention", "attention") * 6,
    )


def _gemma_3(tiny: bool) -> ModelConfig:
    """Gemma 3 (google/gemma-3-1b-pt ``config.json``).

    ``sliding_window_pattern`` 6: five local layers (``sliding_window`` 512 in the 1B model,
    1024 in 4B-27B; ``rope_local_base_freq`` 1e4) then one global layer (``rope_theta`` 1e6);
    QK-norm; 4 heads / 1 KV head of 256 (``query_pre_attn_scalar`` 256 = the default
    ``head_dim ** -0.5`` scale); ``hidden_size`` 1152, ``intermediate_size`` 6912, 26 layers;
    RMSNorm eps 1e-6; tied embeddings. The training variant is the 1B shape exactly with this
    template's vocabulary and SwiGLU MLPs (Gemma's MLP is GeGLU).
    """
    def attn(window: int | None, theta: float, heads: int, head_dim: int) -> AttentionConfig:
        return AttentionConfig(num_heads=heads, num_kv_heads=1, head_dim=head_dim, rope_theta=theta,
                               sliding_window=window)

    def layers(n: int) -> tuple[str, ...]:
        return tuple("attention" if (i + 1) % 6 == 0 else "local_attention" for i in range(n))

    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=6, d_ff=128,
            attention=attn(None, 1_000_000.0, 4, 16), local_attention=attn(4, 10_000.0, 4, 16),
            layer_types=layers(6),
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=1152, n_layers=26, d_ff=6912,
        attention=attn(None, 1_000_000.0, 4, 256), local_attention=attn(512, 10_000.0, 4, 256),
        layer_types=layers(26),
    )


PRODUCTION_PRESETS: dict[str, Callable[[], ModelConfig]] = {
    "tiny-qwen3-next": lambda: _qwen3_next(tiny=True),
    "qwen3-next-300m": lambda: _qwen3_next(tiny=False),
    "tiny-nemotron-h": lambda: _nemotron_h(tiny=True),
    "nemotron-h-600m": lambda: _nemotron_h(tiny=False),
    "tiny-granite-4.0-h": lambda: _granite_4_h(tiny=True),
    "granite-4.0-h-125m": lambda: _granite_4_h(tiny=False),
    "tiny-gpt-oss": lambda: _gpt_oss(tiny=True),
    "gpt-oss-180m": lambda: _gpt_oss(tiny=False),
    "tiny-gemma-3": lambda: _gemma_3(tiny=True),
    "gemma-3-750m": lambda: _gemma_3(tiny=False),
}
_PRESETS.update(PRODUCTION_PRESETS)


# RWKV-7 (jla.layers.rwkv7.RWKV7Mixer and RWKV7FeedForward) in RMSNorm pre-norm blocks, like the
# RWKV-6 presets: the time mix with its value residual from the first layer and RWKV-7's channel mix
# (mlp_type "rwkv7_cmix", d_ff = 4 d). ``rwkv7-0.1b`` has BlinkDL's RWKV-7 0.1B shape (L12-D768, head
# size 64, LoRA ranks 64/64/32/128 = FLA's defaults) with this template's tied 50304-token vocabulary and
# chunk 32, so that impl "auto" runs the Mosaic GPU kernels (document masking included; chunk 16 would run
# the chunked JAX op, about 4x slower to train); ``tiny-rwkv7`` keeps chunk 16 for the chunked-op tests.
# BlinkDL's model uses LayerNorms, an extra LayerNorm after the embedding and an untied head
# (jla.models.rwkv7.RWKV7LM keeps that exact layout).
def _rwkv7(tiny: bool) -> ModelConfig:
    """RWKV-7 time-mix and channel-mix blocks in LanguageModel, with its tied embedding and head.

    With ``init_scheme="upstream"`` (the default) every RWKV-7 block is the identity at init (zero ``o_proj`` and
    channel-mix ``value``, as in FLA and BlinkDL's code), so the tied head first predicts the input
    token: the step-0 loss is about 13 (13.3 measured at a 1024-token vocabulary) rather than ln V. It drops to ln V
    within a step or two with a short warmup and stays high for longer with a long one. Upstream RWKV-7 unties the head;
    ``tie_embeddings=False`` (as in ``rwkv7-1p3b``) does not score the input token.
    """
    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=3, d_ff=256, norm_eps=1e-5,
            attention=AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16),  # unused by RWKV-7 stacks
            linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                                   conv_size=0, qk_l2norm=False, chunk_size=16,
                                                   rwkv7_lora_dims=(16, 16, 16, 16)),
            layer_types=("rwkv7",) * 3, mlp_types=("rwkv7_cmix",) * 3,
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=12, d_ff=3072, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=12, num_kv_heads=4, head_dim=64),  # unused by RWKV-7 stacks
        linear_attention=LinearAttentionConfig(num_k_heads=12, num_v_heads=12, head_k_dim=64, head_v_dim=64,
                                               conv_size=0, qk_l2norm=False, chunk_size=32),
        layer_types=("rwkv7",) * 12, mlp_types=("rwkv7_cmix",) * 12,
    )


_PRESETS["tiny-rwkv7"] = lambda: _rwkv7(tiny=True)
_PRESETS["rwkv7-0.1b"] = lambda: _rwkv7(tiny=False)


def _minimax_01(tiny: bool) -> ModelConfig:
    """MiniMax-01 (MiniMaxAI/MiniMax-Text-01 ``config.json``).

    ``attn_type_list``: softmax attention at every 8th layer (0-based 7, 15, ..., 79), lightning
    attention elsewhere (``layer_types`` ``"lightning"``: ``simple_gla`` with the fixed per-head
    decay schedule, ``jla.layers.lightning_attention``); 64 heads of ``head_dim`` 128 in both kinds
    (``num_key_value_heads`` 8 in the softmax layers), no QK-norm, ``rotary_dim`` 64 of 128 and
    ``rope_theta`` 1e7; an MoE in every layer with ``num_local_experts`` 32, top-2 of a softmax
    router renormalized over the selected experts (Mixtral routing), ``intermediate_size`` 9216 =
    1.5 x ``hidden_size``, no shared expert (``shared_intermediate_size`` 0),
    ``router_aux_loss_coef`` 1e-3 (``aux_loss_coef`` 2e-3 = top-2 x 1e-3: HF's balance loss divides the
    expert counts by tokens, ours by routes); RMSNorm eps 1e-5; untied embeddings. The training variant
    ``minimax-01-1b`` has 16 layers (two softmax) at ``hidden_size`` 768 with 8 heads of 128 (the
    4/3 head-width ratio of 64 x 128 / 6144) and 1 KV head, 32 experts of width 768 (1.0 x hidden,
    so that the model has about 1B parameters) and top-2. Differences from MiniMax-Text-01: this
    template's pre-norm blocks (MiniMax uses ``postnorm``: the residual is the normed input scaled
    by ``layernorm_*_alpha`` = (2 L)^(1/4)) and its 50304-token vocabulary.
    """
    if tiny:
        return ModelConfig(
            vocab_size=512, d_model=64, n_layers=8, d_ff=0, tie_embeddings=False, norm_eps=1e-5,
            attention=AttentionConfig(num_heads=4, num_kv_heads=1, head_dim=16, qk_norm=False,
                                      rotary_fraction=0.5, rope_theta=10_000_000.0),
            linear_attention=LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16,
                                                   conv_size=0, qk_l2norm=False, chunk_size=16),
            moe=MoEConfig(num_experts=4, top_k=2, d_ff_expert=32, aux_loss_coef=aux_loss_coef_from_hf(1e-3, top_k=2)),
            layer_types=hybrid_pattern(8, "lightning", 8),
        )
    return ModelConfig(
        vocab_size=_VOCAB, d_model=768, n_layers=16, d_ff=0, tie_embeddings=False, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=8, num_kv_heads=1, head_dim=128, qk_norm=False,
                                  rotary_fraction=0.5, rope_theta=10_000_000.0),
        linear_attention=LinearAttentionConfig(num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128,
                                               conv_size=0, qk_l2norm=False),
        moe=MoEConfig(num_experts=32, top_k=2, d_ff_expert=768, aux_loss_coef=aux_loss_coef_from_hf(1e-3, top_k=2)),
        layer_types=hybrid_pattern(16, "lightning", 8),
    )


PRODUCTION_PRESETS["tiny-minimax-01"] = lambda: _minimax_01(tiny=True)
PRODUCTION_PRESETS["minimax-01-1b"] = lambda: _minimax_01(tiny=False)
_PRESETS.update(PRODUCTION_PRESETS)


# 1.3B end-to-end presets (benchmarks/end_to_end_1p3b.md) ------------------------------------------
# FLA's published ~1.3B shapes: fla-org/flame configs/{transformer,gated_deltanet,mamba2}_1B.json
# and, where flame has no 1B config, the defaults of FLA main (c6e4f00)
# fla/models/{kda,mamba3}/configuration_*.py, which are FLA's 1.3B shapes (hidden 2048, 24 or
# 48 layers). Every preset keeps FLA's hidden size, layer count, heads, head dims, expansions,
# norm eps and untied embeddings; the vocabulary is the Llama-3.1 tokenizer's 128,256 ids of the
# FineWeb-Edu corpus. FLA's dense MLP for ``hidden_ratio`` 4 is a SwiGLU of width
# 256 * ceil(int(2048 * 4 * 2 / 3) / 256) = 5632. Deliberate differences, field by field, are in
# benchmarks/end_to_end_1p3b.md (GQA with 8 KV heads in the transformer, as asked for a Llama-style
# model; GDN-2 from its paper; the hybrids keep this template's gdn-hybrid / kda-hybrid layouts).
_VOCAB_LLAMA3 = 128256
_FLA_1P3B_D_MODEL = 2048
_FLA_1P3B_D_FF = 5632


def _transformer_1p3b(window: int | None) -> ModelConfig:
    """flame ``transformer_1B.json``: 24 layers, 32 heads of 64, RoPE 1e4, no QK-norm; 8 KV heads (GQA)."""
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=_FLA_1P3B_D_MODEL, n_layers=24, d_ff=_FLA_1P3B_D_FF,
        tie_embeddings=False, norm_eps=1e-6,
        attention=AttentionConfig(num_heads=32, num_kv_heads=8, head_dim=64, rope_theta=10_000.0, qk_norm=False,
                                  sliding_window=window),
    )


def _linear_1p3b(kind: str, la: LinearAttentionConfig, n_layers: int = 24) -> ModelConfig:
    """``n_layers`` ``kind`` mixers, each followed by FLA's 5632-wide SwiGLU MLP."""
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=_FLA_1P3B_D_MODEL, n_layers=n_layers, d_ff=_FLA_1P3B_D_FF,
        tie_embeddings=False, norm_eps=1e-6,
        attention=AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128),  # unused by all-linear stacks
        linear_attention=la, layer_types=(kind,) * n_layers,
    )


def _mamba_1p3b(kind: str) -> ModelConfig:
    """flame ``mamba2_1B.json`` / FLA ``Mamba3Config``: 48 mixer-only blocks, d_state 128, expand 2, head 64."""
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=_FLA_1P3B_D_MODEL, n_layers=48, d_ff=0, tie_embeddings=False,
        norm_eps=1e-5,
        attention=AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128),  # unused by Mamba stacks
        ssm=SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1),
        layer_types=(kind,) * 48, mlp_types=("none",) * 48,
    )


def _hybrid_1p3b(kind: str) -> ModelConfig:
    """``gdn-hybrid-125m`` / ``kda-hybrid-125m`` at FLA's 1.3B width: 24 layers, attention every 4th."""
    if kind == "gated_deltanet":  # Qwen3-Next-like: grouped value heads, gated attention with partial RoPE
        attention = AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128, output_gate=True,
                                    rotary_fraction=0.25, rope_theta=10_000_000.0)
        la = LinearAttentionConfig(num_k_heads=8, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                                   conv_init=_GDN_KDA_CONV_INIT)
    else:  # Kimi-Linear-like: KDA with FLA's KDA head shape, NoPE attention
        attention = AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128, use_rope=False)
        la = LinearAttentionConfig(num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                                   conv_init=_GDN_KDA_CONV_INIT)
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=_FLA_1P3B_D_MODEL, n_layers=24, d_ff=_FLA_1P3B_D_FF,
        tie_embeddings=False, norm_eps=1e-6, attention=attention, linear_attention=la,
        layer_types=hybrid_pattern(24, kind, 4),
    )


# FLA's KDA head shape (16 heads of 128, expand_v 1), used by KDA, GDN-2 and the 128-dim GDN variant.
_FLA_HEADS_128 = dict(num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128)

E2E_1P3B_PRESETS: dict[str, Callable[[], ModelConfig]] = {
    "transformer-1p3b": lambda: _transformer_1p3b(None),
    "transformer-swa-1p3b": lambda: _transformer_1p3b(2048),
    # flame gated_deltanet_1B.json exactly: 21 layers, 6 heads, head_dim 256, expand_v 2 (values of 512).
    "gdn-1p3b": lambda: _linear_1p3b("gated_deltanet", LinearAttentionConfig(
        num_k_heads=6, num_v_heads=6, head_k_dim=256, head_v_dim=512, conv_init=_ALL_LINEAR_CONV_INIT), n_layers=21),
    # The same model with FLA's 128-dim head shape (24 layers), which the Mosaic GPU kernels support.
    "gdn-hd128-1p3b": lambda: _linear_1p3b("gated_deltanet", LinearAttentionConfig(
        **_FLA_HEADS_128, conv_init=_ALL_LINEAR_CONV_INIT)),
    "gdn2-1p3b": lambda: _linear_1p3b("gdn2", LinearAttentionConfig(**_FLA_HEADS_128)),
    "kda-1p3b": lambda: _linear_1p3b("kda", LinearAttentionConfig(**_FLA_HEADS_128, conv_init=_ALL_LINEAR_CONV_INIT)),
    "mamba2-1p3b": lambda: _mamba_1p3b("mamba2"),
    "mamba3-1p3b": lambda: _mamba_1p3b("mamba3"),
    "gdn-hybrid-1p3b": lambda: _hybrid_1p3b("gated_deltanet"),
    "kda-hybrid-1p3b": lambda: _hybrid_1p3b("kda"),
}
_PRESETS.update(E2E_1P3B_PRESETS)


# RWKV-7 at flame's ``rwkv7-1p3b.json`` shape (benchmarks/profile_1p3b.md): 24 layers, 32 heads of 64,
# LoRA ranks decay 64 / a 64 / v 16 / gate 128, channel mix of width 4 d (squared ReLU), untied
# 128,256-token vocabulary, chunk 32 (the Mosaic GPU kernels' chunk). flame's RWKV-7 uses LayerNorms with
# bias (and one more after the embedding); this template's RWKV-7 stacks use RMSNorm pre-norm blocks.
def _rwkv7_1p3b() -> ModelConfig:
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=_FLA_1P3B_D_MODEL, n_layers=24, d_ff=4 * _FLA_1P3B_D_MODEL,
        tie_embeddings=False, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128),  # unused by RWKV-7 stacks
        linear_attention=LinearAttentionConfig(num_k_heads=32, num_v_heads=32, head_k_dim=64, head_v_dim=64,
                                               conv_size=0, qk_l2norm=False, chunk_size=32,
                                               rwkv7_lora_dims=(64, 64, 16, 128)),
        layer_types=("rwkv7",) * 24, mlp_types=("rwkv7_cmix",) * 24,
    )


E2E_1P3B_PRESETS["rwkv7-1p3b"] = _rwkv7_1p3b
_PRESETS["rwkv7-1p3b"] = _rwkv7_1p3b


# Real hybrid layouts at ~1.3B total parameters (benchmarks/end_to_end_1p3b.md) ------------------------
# Kimi Linear and Qwen3-Next at a trainable scale, with the 1.3B end-to-end protocol's untied
# 128,256-token Llama-3.1 vocabulary. Both halve (Kimi: 2304 -> 1024, Qwen3-Next: 2048 -> 1024) the
# hidden size of the published model and keep its per-head shapes, layer pattern, routing and MoE
# granularity (expert width / hidden size); the depth and the expert count set the parameter budget.
# Configs: benchmarks/end_to_end/hybrids_1p3b/<name>.toml.
def _kimi_linear_1p3b() -> ModelConfig:
    """Kimi Linear (moonshotai/Kimi-Linear-48B-A3B-Instruct ``config.json``) at 1.20B total, 0.63B active.

    Kept from the config: three KDA layers per MLA layer with MLA last (``full_attn_layers`` every 4th,
    ending the stack), KDA heads of 128 x 128 with a 4-tap short convolution and the low-rank (rank
    ``head_dim``) decay and output gates, MLA with ``kv_lora_rank`` 512, ``qk_nope_head_dim`` 128,
    ``qk_rope_head_dim`` 64, ``v_head_dim`` 128, full-rank queries (``q_lora_rank`` null) and no RoPE
    (``mla_use_nope``), the first layer dense (``first_k_dense_replace`` 1, ``intermediate_size`` = 4 x
    hidden) and MoE after it (``moe_layer_freq`` 1): top-8 of sigmoid-scored experts, renormalized and
    scaled by ``routed_scaling_factor`` 2.446, plus one shared expert, ``rms_norm_eps`` 1e-5, untied
    embeddings. Scaled: hidden 2304 -> 1024, 32 -> 16 KDA and MLA heads, 27 -> 16 layers, 256 experts of
    1024 -> 32 experts of 512 (the ratio hidden / expert width stays about 2). Not mirrored: the
    aux-loss-free expert bias (``e_score_correction_bias``; this template has no bias update rule, so the
    MoE keeps its load-balancing loss) and the 163,840-token vocabulary.
    """
    n_layers = 16
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=1024, n_layers=n_layers, d_ff=4096, tie_embeddings=False, norm_eps=1e-5,
        attention=AttentionConfig(num_heads=16, num_kv_heads=16, head_dim=128, use_rope=False),  # unused by MLA stacks
        linear_attention=LinearAttentionConfig(num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        mla=MLAConfig(num_heads=16, kv_lora_rank=512, q_lora_rank=None, qk_nope_head_dim=128, qk_rope_head_dim=64,
                      v_head_dim=128, use_rope=False),
        moe=MoEConfig(num_experts=32, top_k=8, d_ff_expert=512, num_shared_experts=1, router="sigmoid",
                      routed_scaling_factor=2.446, ragged_dot_impl="mosaic_gpu"),
        layer_types=tuple("mla" if t == "attention" else t for t in hybrid_pattern(n_layers, "kda", 4)),
        mlp_types=("dense",) + ("moe",) * (n_layers - 1),
    )


def _qwen3_next_1p3b() -> ModelConfig:
    """Qwen3-Next (Qwen/Qwen3-Next-80B-A3B-Instruct ``config.json``) at 1.27B total, 0.55B active.

    ``qwen3-next-300m``'s training variant (half the hidden size, every width scaled with it: gated
    attention 8 / 1 heads of 256 with QK-norm and 25% partial RoPE at theta 1e7, Gated DeltaNet 8 key /
    16 value heads of 128, 64 experts of 256, top-10, normalized, a sigmoid-gated shared expert of 256,
    ``router_aux_loss_coef`` 1e-3 = ``aux_loss_coef`` 1e-2 as in ``qwen3-next-300m``, one MTP layer) with
    16 layers (three Gated DeltaNet layers per gated attention layer, ``full_attention_interval`` 4) and
    the 128,256-token vocabulary.
    """
    n_layers = 16
    return ModelConfig(
        vocab_size=_VOCAB_LLAMA3, d_model=1024, n_layers=n_layers, d_ff=0, tie_embeddings=False,
        attention=AttentionConfig(num_heads=8, num_kv_heads=1, head_dim=256, output_gate=True,
                                  rotary_fraction=0.25, rope_theta=10_000_000.0),
        linear_attention=LinearAttentionConfig(num_k_heads=8, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                                               conv_init=_GDN_KDA_CONV_INIT),
        moe=MoEConfig(num_experts=64, top_k=10, d_ff_expert=256, num_shared_experts=1,
                      aux_loss_coef=aux_loss_coef_from_hf(1e-3, top_k=10), shared_expert_gate=True,
                      ragged_dot_impl="mosaic_gpu"),
        layer_types=hybrid_pattern(n_layers, "gated_deltanet", 4),
        mtp=MTPConfig(depth=1, kind="deepseek", mlp_type="moe"),
    )


HYBRID_1P3B_PRESETS: dict[str, Callable[[], ModelConfig]] = {
    "kimi-linear-1p3b": _kimi_linear_1p3b,
    "qwen3-next-1p3b": _qwen3_next_1p3b,
}
_PRESETS.update(HYBRID_1P3B_PRESETS)
