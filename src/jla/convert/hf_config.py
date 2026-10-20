# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers (Apache-2.0): the per-family config defaults and sliding-window rules of
# models/{llama,mistral,qwen2,qwen3,phi3,mixtral,qwen3_moe,mamba2,deepseek_v3}/configuration_*.py, and
# the YaRN mscale rules and original_max_position_embeddings handling of modeling_rope_utils.py.
"""Hugging Face ``config.json`` <-> :class:`~jla.models.config.ModelConfig` for the supported families.

Families (HF ``model_type``): ``llama`` (Llama 2, 3 and 3.1+ with ``llama3`` RoPE scaling), ``mistral``,
``qwen2`` (Qwen2 / Qwen2.5, q/k/v biases), ``qwen3`` (QK-norm), ``phi3`` (fused qkv and
gate/up weights, partial rotary), ``mixtral`` and ``qwen3_moe`` (token-choice MoE), ``mamba2``
(``Mamba2ForCausalLM``) and ``deepseek_v3`` (MLA with YaRN, sigmoid MoE with group-limited
routing, the expert-choice bias, shared experts and leading dense layers). RoPE scaling:
``linear``, ``llama3`` and ``yarn``. Both
the ``transformers`` 4.x layout (``rope_theta`` / ``rope_scaling``) and the 5.x layout
(``rope_parameters``, ``layer_types``) are read. Every option without an exact
equivalent in our model raises instead of being dropped.

``router_aux_loss_coef`` (Mixtral, Qwen3-MoE) maps to ``aux_loss_coef = top_k * router_aux_loss_coef``
and back: HF's ``load_balancing_loss_func`` divides the expert counts by tokens, ours by routes
(``tokens * top_k``; :func:`jla.models.config.aux_loss_coef_from_hf`). DeepSeek-V3's ``aux_loss_alpha``
has our normalization and maps 1:1.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jla.models.config import AttentionConfig, MLAConfig, ModelConfig, MoEConfig, RoPEScalingConfig, SSMConfig
from jla.models.config import aux_loss_coef_from_hf, aux_loss_coef_to_hf

__all__ = ["ARCHITECTURES", "SUPPORTED_MODEL_TYPES", "from_hf_config", "read_hf_config", "to_hf_config"]

ARCHITECTURES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "mamba2": "Mamba2ForCausalLM",
    "deepseek_v3": "DeepseekV3ForCausalLM",
}
SUPPORTED_MODEL_TYPES = tuple(ARCHITECTURES)
MOE_MODEL_TYPES = ("mixtral", "qwen3_moe")
_DEFAULT_THETA = {"mixtral": 1_000_000.0}  # every other family defaults to 10000
_QK_NORM = ("qwen3", "qwen3_moe")
_GLOBAL_WINDOW = ("mistral", "mixtral", "phi3")  # sliding_window applies to every layer
_LAYER_WINDOW = ("qwen2", "qwen3")  # use_sliding_window + layer_types / max_window_layers
_ALL_LAYER_WINDOW = ("qwen3_moe",)  # use_sliding_window puts sliding_window on every layer (transformers 5)
_NO_WINDOW = ("llama",)  # transformers' Llama has no sliding window
# transformers' defaults when config.json lacks the key
_DEFAULT_WINDOW = {"mistral": 4096, "qwen2": 4096, "qwen3": 4096, "qwen3_moe": 4096}
_DEFAULT_MAX_WINDOW_LAYERS = 28  # qwen2 / qwen3


def read_hf_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """A config dict from a mapping, a ``config.json`` file or a checkpoint directory."""
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


def _fail(model_type: str, message: str) -> None:
    raise ValueError(f"cannot convert Hugging Face {model_type!r} config: {message}")


def _rope(hf: Mapping[str, Any], model_type: str) -> tuple[float, float, RoPEScalingConfig | None]:
    """``(rope_theta, partial_rotary_factor, scaling)``; ``linear``, ``llama3`` and ``yarn`` scaling are supported."""
    params = hf.get("rope_parameters") or {}
    scaling = hf.get("rope_scaling") or {}
    if params and not isinstance(params, Mapping):
        _fail(model_type, f"rope_parameters must be a mapping; got {params!r}")
    if params and "rope_theta" not in params and all(isinstance(v, Mapping) for v in params.values()):
        _fail(model_type, "per-layer-type rope_parameters are not supported")
    merged = _rope_settings(hf)
    rope_type = params.get("rope_type") or scaling.get("rope_type") or scaling.get("type") or "default"
    theta = float(params.get("rope_theta", hf.get("rope_theta", _DEFAULT_THETA.get(model_type, 10_000.0))))
    partial = float(params.get("partial_rotary_factor", hf.get("partial_rotary_factor", 1.0)))
    if partial != 1.0 and model_type != "phi3":
        _fail(model_type, f"partial_rotary_factor {partial} is only defined for phi3 in transformers")
    if rope_type == "default":
        return theta, partial, None
    if model_type == "phi3":  # transformers turns phi3's yarn / su scaling into longrope
        _fail(model_type, f"RoPE scaling {rope_type!r} (transformers builds longrope for phi3, which is not supported)")
    if rope_type in ("llama3", "yarn"):
        merged["original_max_position_embeddings"] = _original_length(hf, merged)
    if rope_type == "linear":
        return theta, partial, RoPEScalingConfig(kind="linear", factor=float(merged["factor"]))
    if rope_type == "llama3":
        return theta, partial, RoPEScalingConfig(
            kind="llama3", factor=float(merged["factor"]), low_freq_factor=float(merged["low_freq_factor"]),
            high_freq_factor=float(merged["high_freq_factor"]),
            original_max_position_embeddings=int(merged["original_max_position_embeddings"]),
        )
    if rope_type == "yarn":  # transformers' _compute_yarn_parameters, with its defaults
        original = int(merged.get("original_max_position_embeddings") or hf["max_position_embeddings"])
        factor = float(merged["factor"]) if merged.get("factor") is not None else hf["max_position_embeddings"] / original
        attention_factor = merged.get("attention_factor")
        if attention_factor is None:
            mscale, mscale_all = merged.get("mscale"), merged.get("mscale_all_dim")
            attention_factor = (_yarn_mscale(factor, mscale) / _yarn_mscale(factor, mscale_all) if mscale and mscale_all
                                else _yarn_mscale(factor))
        return theta, partial, RoPEScalingConfig(
            kind="yarn", factor=factor, original_max_position_embeddings=original,
            beta_fast=float(merged.get("beta_fast") or 32), beta_slow=float(merged.get("beta_slow") or 1),
            attention_factor=float(attention_factor), truncate=bool(merged.get("truncate", True)),
        )
    _fail(model_type, f"RoPE scaling {rope_type!r} is not supported (default, linear, llama3 and yarn are)")
    raise AssertionError  # unreachable


def _original_length(hf: Mapping[str, Any], merged: Mapping[str, Any]) -> int:
    """transformers' rule: the RoPE dict's ``original_max_position_embeddings``, else ``max_position_embeddings``.

    A top-level key wins only in configs that declare the field (Phi-3, whose scaling we reject);
    the other families' configs attach it after the RoPE parameters are set, so it is not read.
    """
    return int(merged.get("original_max_position_embeddings") or hf["max_position_embeddings"])


def _rope_settings(hf: Mapping[str, Any]) -> dict[str, Any]:
    """The RoPE keys of the 4.x ``rope_scaling`` and the 5.x ``rope_parameters`` in one dict (5.x wins)."""
    return {**(hf.get("rope_scaling") or {}), **(hf.get("rope_parameters") or {})}


def _yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    import math

    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _yarn_dict(scaling: RoPEScalingConfig) -> dict[str, Any]:
    return {"rope_type": "yarn", "factor": scaling.factor,
            "original_max_position_embeddings": scaling.original_max_position_embeddings,
            "beta_fast": scaling.beta_fast, "beta_slow": scaling.beta_slow,
            "attention_factor": scaling.attention_factor, "truncate": scaling.truncate}


def _rope_dicts(attn: AttentionConfig, model_type: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``(rope_scaling, rope_parameters)``: the transformers 4.x and 5.x spellings of the RoPE settings."""
    scaling = attn.rope_scaling
    legacy = None
    if scaling is not None and scaling.kind == "linear":
        legacy = {"rope_type": "linear", "factor": scaling.factor}
    elif scaling is not None and scaling.kind == "yarn":
        legacy = _yarn_dict(scaling)
    elif scaling is not None:
        legacy = {"rope_type": "llama3", "factor": scaling.factor, "low_freq_factor": scaling.low_freq_factor,
                  "high_freq_factor": scaling.high_freq_factor,
                  "original_max_position_embeddings": scaling.original_max_position_embeddings}
    params = {"rope_type": "default", **(legacy or {}), "rope_theta": attn.rope_theta}
    if model_type == "phi3":
        params["partial_rotary_factor"] = attn.rotary_fraction
    return legacy, params


def _windows(hf: Mapping[str, Any], model_type: str, n_layers: int) -> tuple[int | None, list[bool]]:
    """``(window, sliding_per_layer)`` with HF's per-family sliding-window rules."""
    window = hf["sliding_window"] if "sliding_window" in hf else _DEFAULT_WINDOW.get(model_type)
    if model_type in _NO_WINDOW:
        if hf.get("use_sliding_window") or "sliding_attention" in (hf.get("layer_types") or ()):
            _fail(model_type, "sliding windows (transformers' Llama ignores them; use mistral, qwen2 or qwen3)")
        return None, [False] * n_layers
    if model_type in _GLOBAL_WINDOW:
        return (int(window), [True] * n_layers) if window else (None, [False] * n_layers)
    if not hf.get("use_sliding_window", False) or not window:
        return None, [False] * n_layers
    if model_type in _ALL_LAYER_WINDOW:  # no layer_types / max_window_layers in this family
        return int(window), [True] * n_layers
    types = hf.get("layer_types")
    if types is None:
        start = int(hf.get("max_window_layers", _DEFAULT_MAX_WINDOW_LAYERS))
        types = ["sliding_attention" if i >= start else "full_attention" for i in range(n_layers)]
    if len(types) != n_layers or set(types) - {"sliding_attention", "full_attention"}:
        _fail(model_type, f"layer_types must list full_attention / sliding_attention per layer; got {types}")
    return int(window), [t == "sliding_attention" for t in types]


def from_hf_config(hf_config: str | Path | Mapping[str, Any], **overrides: Any) -> ModelConfig:
    """Translate a Hugging Face config into a :class:`ModelConfig`.

    ``overrides`` replace training-side fields that HF configs do not carry
    (``param_dtype``, ``compute_dtype``, ``scan_layers``, ``remat``, ...).
    """
    hf = read_hf_config(hf_config)
    model_type = hf.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported Hugging Face model_type {model_type!r}; supported: {SUPPORTED_MODEL_TYPES}")
    if hf.get("hidden_act", "silu") != "silu":
        _fail(model_type, f"hidden_act {hf.get('hidden_act')!r} (our MLPs are SwiGLU)")
    if model_type == "mamba2":
        return _mamba2_config(hf, overrides)
    if hf.get("attention_bias", False):
        _fail(model_type, "attention_bias=True (biases on the output projection are not supported)")
    if hf.get("mlp_bias", False):
        _fail(model_type, "mlp_bias=True")
    if float(hf.get("attention_dropout", 0.0) or 0.0) != 0.0:
        _fail(model_type, "attention_dropout > 0 has no equivalent")
    if float(hf.get("router_jitter_noise", 0.0) or 0.0) != 0.0:
        _fail(model_type, "router_jitter_noise > 0 has no equivalent")
    if model_type == "deepseek_v3":
        return _deepseek_v3_config(hf, overrides)
    d_model, n_layers = int(hf["hidden_size"]), int(hf["num_hidden_layers"])
    heads = int(hf["num_attention_heads"])
    kv_heads = int(hf.get("num_key_value_heads") or heads)
    head_dim = int(hf.get("head_dim") or d_model // heads)
    theta, partial, rope_scaling = _rope(hf, model_type)
    window, sliding = _windows(hf, model_type, n_layers)
    attention = AttentionConfig(
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, rope_theta=theta,
        rotary_fraction=partial, qk_norm=model_type in _QK_NORM, qkv_bias=model_type == "qwen2", rope_scaling=rope_scaling,
        sliding_window=window if all(sliding) else None,
    )
    local, layer_types = None, None
    if any(sliding) and not all(sliding):
        local = dataclasses.replace(attention, sliding_window=window)
        layer_types = tuple("local_attention" if s else "attention" for s in sliding)
    moe, mlp_types, d_ff = None, None, int(hf.get("intermediate_size") or 0)
    if model_type == "mixtral":
        moe = MoEConfig(
            num_experts=int(hf["num_local_experts"]), top_k=int(hf["num_experts_per_tok"]),
            d_ff_expert=d_ff, router="softmax", normalize_top_k=True, capacity_factor=None,
            aux_loss_coef=aux_loss_coef_from_hf(float(hf.get("router_aux_loss_coef", 0.001)),
                                                int(hf["num_experts_per_tok"])),
        )
        mlp_types = ("moe",) * n_layers
    elif model_type == "qwen3_moe":
        experts = int(hf.get("num_experts") or hf.get("num_local_experts") or 0)
        step = int(hf.get("decoder_sparse_step", 1))
        dense_only = set(hf.get("mlp_only_layers") or ())
        mlp_types = tuple(
            "moe" if i not in dense_only and experts > 0 and (i + 1) % step == 0 else "dense" for i in range(n_layers)
        )
        if "moe" in mlp_types:
            moe = MoEConfig(
                num_experts=experts, top_k=int(hf["num_experts_per_tok"]), d_ff_expert=int(hf["moe_intermediate_size"]),
                router="softmax", normalize_top_k=bool(hf.get("norm_topk_prob", False)), capacity_factor=None,
                aux_loss_coef=aux_loss_coef_from_hf(float(hf.get("router_aux_loss_coef", 0.001)),
                                                    int(hf["num_experts_per_tok"])),
            )
    dropout = {}
    if model_type == "phi3":
        dropout = dict(embed_dropout=float(hf.get("embd_pdrop", 0.0) or 0.0),
                       resid_dropout=float(hf.get("resid_pdrop", 0.0) or 0.0))
    config = ModelConfig(
        vocab_size=int(hf["vocab_size"]), d_model=d_model, n_layers=n_layers, d_ff=d_ff, attention=attention,
        moe=moe, layer_types=layer_types, mlp_types=mlp_types, local_attention=local,
        norm_eps=float(hf.get("rms_norm_eps", 1e-6)), tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
        initializer_range=float(hf.get("initializer_range", 0.02)), **dropout,
    )
    return config.replace(**overrides) if overrides else config


def _check_exportable(config: ModelConfig, model_type: str) -> None:
    def no(message: str) -> None:
        raise ValueError(f"cannot export to Hugging Face {model_type!r}: {message}")

    if model_type not in SUPPORTED_MODEL_TYPES:
        no(f"unsupported model_type; supported: {SUPPORTED_MODEL_TYPES}")
    if set(config.resolved_layer_types) - {"attention", "local_attention"}:
        no(f"layer types {sorted(set(config.resolved_layer_types))} (softmax attention only)")
    if config.mtp is not None:
        no("MTP heads have no Hugging Face equivalent in these families (set mtp=None)")
    attentions = [config.attention] + ([config.local_attention] if "local_attention" in config.resolved_layer_types else [])
    for attn in attentions:
        if attn.qk_norm != (model_type in _QK_NORM):
            no(f"qk_norm={attn.qk_norm} does not match the family")
        if attn.qkv_bias != (model_type == "qwen2"):
            no(f"qkv_bias={attn.qkv_bias} does not match the family")
        if attn.output_gate or attn.sinks or attn.moba_block_size is not None or not attn.use_rope:
            no("output gates, sinks, MoBA and NoPE are not part of these families")
        if attn.softmax_scale is not None and attn.softmax_scale != attn.head_dim ** -0.5:
            no("a non-default softmax_scale")
        if attn.rotary_fraction != 1.0 and model_type != "phi3":
            no("partial rotary (rotary_fraction < 1) exists only in phi3")
        for field in ("num_heads", "num_kv_heads", "head_dim", "rope_theta", "rotary_fraction", "qk_norm", "qkv_bias",
                      "rope_scaling"):
            if getattr(attn, field) != getattr(config.attention, field):
                no(f"local and global attention differ in {field}")
    if model_type == "phi3" and config.attention.rope_scaling is not None:
        no("RoPE scaling (transformers builds longrope for phi3)")
    if model_type in _NO_WINDOW and (config.attention.sliding_window is not None
                                     or "local_attention" in config.resolved_layer_types):
        no("sliding windows (transformers' Llama has none; export as mistral, qwen2 or qwen3)")
    if "local_attention" in config.resolved_layer_types:
        if model_type not in _LAYER_WINDOW:
            no("per-layer sliding windows exist only in qwen2 / qwen3 (qwen3_moe windows every layer or none)")
        if config.attention.sliding_window is not None or config.local_attention.sliding_window is None:
            no("mixed layers need global 'attention' layers and windowed 'local_attention' layers")
    mlp = set(config.resolved_mlp_types)
    if model_type == "mixtral" and mlp != {"moe"}:
        no("every Mixtral layer is an MoE layer")
    if model_type == "qwen3_moe" and not mlp <= {"moe", "dense"}:
        no(f"mlp types {sorted(mlp)}")
    if model_type not in MOE_MODEL_TYPES and mlp != {"dense"}:
        no(f"mlp types {sorted(mlp)} (dense SwiGLU only)")
    if config.moe is not None and "moe" in mlp:
        moe = config.moe
        if moe.router != "softmax" or moe.num_shared_experts or moe.capacity_factor is not None:
            no("only dropless softmax routing without shared experts maps to these families")
        if model_type == "mixtral" and not moe.normalize_top_k:
            no("Mixtral renormalizes the top-k gates (normalize_top_k=True)")
        if moe.router_bias or moe.num_groups != 1 or moe.top_k_groups != 1 or moe.routed_scaling_factor != 1.0:
            no("router_bias, expert groups and routed_scaling_factor are DeepSeek-V3 routing (export as deepseek_v3)")


def to_hf_config(
    config: ModelConfig,
    model_type: str,
    *,
    dtype: str = "float32",
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``config.json`` dict of ``config`` as a ``model_type`` model.

    ``base`` (e.g. the original ``config.json``) supplies fields we do not model,
    such as token ids and ``max_position_embeddings``; mapped fields override it.
    Both the 4.x (``rope_theta``) and 5.x (``rope_parameters``) RoPE keys are written.
    """
    if model_type == "mamba2":
        return _mamba2_hf_config(config, dtype, base)
    if model_type == "deepseek_v3":
        return _deepseek_v3_hf_config(config, dtype, base)
    _check_exportable(config, model_type)
    attn = config.attention
    out: dict[str, Any] = dict(base or {})
    for stale in ("rope_scaling", "rope_parameters", "layer_types", "torch_dtype", "dtype", "transformers_version"):
        out.pop(stale, None)
    legacy_scaling, rope = _rope_dicts(attn, model_type)
    out.update(
        architectures=[ARCHITECTURES[model_type]], model_type=model_type, vocab_size=config.vocab_size,
        hidden_size=config.d_model, num_hidden_layers=config.n_layers, num_attention_heads=attn.num_heads,
        num_key_value_heads=attn.num_kv_heads, head_dim=attn.head_dim, hidden_act="silu",
        rms_norm_eps=config.norm_eps, rope_theta=attn.rope_theta, rope_scaling=legacy_scaling, rope_parameters=rope,
        tie_word_embeddings=config.tie_embeddings, attention_dropout=0.0, initializer_range=config.initializer_range,
        torch_dtype=dtype, dtype=dtype,
    )
    out.setdefault("max_position_embeddings", 4096)
    out.setdefault("pad_token_id", None)  # family defaults (Phi-3: 32000) can exceed a small vocabulary
    if model_type == "phi3":
        out["partial_rotary_factor"] = attn.rotary_fraction
        out.update(embd_pdrop=config.embed_dropout, resid_pdrop=config.resid_dropout)
    if model_type in ("llama", "qwen3", "qwen3_moe"):
        out["attention_bias"] = False
    if model_type == "llama":
        out["mlp_bias"] = False
    types = config.resolved_layer_types
    if model_type in _GLOBAL_WINDOW:
        out["sliding_window"] = attn.sliding_window
    elif model_type in _ALL_LAYER_WINDOW:
        out.update(use_sliding_window=attn.sliding_window is not None, sliding_window=attn.sliding_window)
    elif "local_attention" in types or attn.sliding_window is not None:
        window = attn.sliding_window or config.local_attention.sliding_window
        sliding = [t == "local_attention" or attn.sliding_window is not None for t in types]
        out.update(use_sliding_window=True, sliding_window=window,
                   layer_types=["sliding_attention" if s else "full_attention" for s in sliding],
                   max_window_layers=sliding.index(True) if all(sliding[sliding.index(True):]) else 0)
    else:
        out.update(use_sliding_window=False, sliding_window=None)
    if model_type == "mixtral":
        out.update(num_local_experts=config.moe.num_experts, num_experts_per_tok=config.moe.top_k,
                   intermediate_size=config.moe.d_ff_expert,
                   router_aux_loss_coef=aux_loss_coef_to_hf(config.moe.aux_loss_coef, config.moe.top_k),
                   router_jitter_noise=0.0)
    else:
        out["intermediate_size"] = config.d_ff
    if model_type == "qwen3_moe":
        mlp = config.resolved_mlp_types
        moe = config.moe
        out.update(
            decoder_sparse_step=1, mlp_only_layers=[i for i, t in enumerate(mlp) if t == "dense"],
            num_experts=moe.num_experts if moe else 0, num_experts_per_tok=moe.top_k if moe else 1,
            moe_intermediate_size=moe.d_ff_expert if moe else config.d_ff,
            norm_topk_prob=bool(moe.normalize_top_k) if moe else False,
            router_aux_loss_coef=aux_loss_coef_to_hf(moe.aux_loss_coef, moe.top_k) if moe else 0.001,
        )
    return out


# Mamba-2 (``Mamba2ForCausalLM``) ------------------------------------------------------------------
# HF's PyTorch fallback (``MambaRMSNormGated``) normalizes the gated output over the whole inner width,
# while state-spaces/mamba (and our ``Mamba2``) normalizes each of the ``n_groups`` groups on its own;
# the two agree exactly for ``n_groups == 1`` only.


def _mamba2_config(hf: Mapping[str, Any], overrides: Mapping[str, Any]) -> ModelConfig:
    model_type = "mamba2"
    d_model, n_layers = int(hf["hidden_size"]), int(hf["num_hidden_layers"])
    expand, heads, head_dim = int(hf.get("expand", 2)), int(hf["num_heads"]), int(hf["head_dim"])
    if heads * head_dim != expand * d_model:
        _fail(model_type, f"num_heads * head_dim = {heads * head_dim} != expand * hidden_size = {expand * d_model}")
    if hf.get("use_bias", False):
        _fail(model_type, "use_bias=True (biases on in_proj / out_proj)")
    if not hf.get("use_conv_bias", True):
        _fail(model_type, "use_conv_bias=False")
    if not hf.get("rms_norm", True):
        _fail(model_type, "rms_norm=False")
    # transformers 5 writes non-finite floats as {"__float__": "Infinity"}
    limit = tuple(float(x["__float__"] if isinstance(x, Mapping) else x) for x in hf.get("time_step_limit", (0, "inf")))
    if limit != (0.0, float("inf")):
        _fail(model_type, f"time_step_limit {limit} (dt clamping) is not supported")
    chunk = int(hf.get("chunk_size", 256))
    ssm = SSMConfig(
        d_state=int(hf["state_size"]), expand=expand, head_dim=head_dim, n_groups=int(hf.get("n_groups", 1)),
        conv_size=int(hf.get("conv_kernel", 4)), dt_min=float(hf.get("time_step_min", 0.001)),
        dt_max=float(hf.get("time_step_max", 0.1)), dt_init_floor=float(hf.get("time_step_floor", 1e-4)),
        chunk_size=chunk if chunk >= 16 and chunk % 16 == 0 else 64,
    )
    config = ModelConfig(
        vocab_size=int(hf["vocab_size"]), d_model=d_model, n_layers=n_layers, d_ff=0,
        attention=AttentionConfig(num_heads=1, num_kv_heads=1, head_dim=head_dim),  # unused by Mamba stacks
        layer_types=("mamba2",) * n_layers, mlp_types=("none",) * n_layers, ssm=ssm,
        norm_eps=float(hf.get("layer_norm_epsilon", 1e-5)), tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
        initializer_range=float(hf.get("initializer_range", 0.02)),
    )
    return config.replace(**overrides) if overrides else config


def _mamba2_hf_config(config: ModelConfig, dtype: str, base: Mapping[str, Any] | None) -> dict[str, Any]:
    def no(message: str) -> None:
        raise ValueError(f"cannot export to Hugging Face 'mamba2': {message}")

    if set(config.resolved_layer_types) != {"mamba2"} or set(config.resolved_mlp_types) != {"none"}:
        no("every layer must be a 'mamba2' mixer without an MLP")
    ssm = config.ssm
    if ssm.log_linear_levels or ssm.mimo_rank != 1 or config.mtp is not None:
        no("log-linear Mamba-2, MIMO and MTP heads have no Hugging Face equivalent")
    out: dict[str, Any] = dict(base or {})
    for stale in ("torch_dtype", "dtype", "transformers_version"):
        out.pop(stale, None)
    out.update(
        architectures=["Mamba2ForCausalLM"], model_type="mamba2", vocab_size=config.vocab_size,
        hidden_size=config.d_model, num_hidden_layers=config.n_layers, num_heads=ssm.num_heads(config.d_model),
        head_dim=ssm.head_dim, expand=ssm.expand, state_size=ssm.d_state, n_groups=ssm.n_groups,
        conv_kernel=ssm.conv_size, use_bias=False, use_conv_bias=True, hidden_act="silu", rms_norm=True,
        layer_norm_epsilon=config.norm_eps, tie_word_embeddings=config.tie_embeddings, chunk_size=ssm.chunk_size,
        time_step_min=ssm.dt_min, time_step_max=ssm.dt_max, time_step_floor=ssm.dt_init_floor,
        time_step_limit=[0.0, float("inf")], residual_in_fp32=True, initializer_range=config.initializer_range,
        torch_dtype=dtype, dtype=dtype,
    )
    return out


# DeepSeek-V3 (``DeepseekV3ForCausalLM``) -----------------------------------------------------------
# HF normalizes the MLA latents (``q_a_layernorm``, ``kv_a_layernorm``) with eps 1e-6 whatever
# ``rms_norm_eps`` says, ours with ``norm_eps``: the two agree for ``rms_norm_eps == 1e-6`` (every
# DeepSeek-V3 release). The multi-token-prediction layers stored after the last decoder layer are
# skipped, as ``transformers`` does.


def _deepseek_v3_config(hf: Mapping[str, Any], overrides: Mapping[str, Any]) -> ModelConfig:
    model_type = "deepseek_v3"
    if hf.get("quantization_config"):
        _fail(model_type, "quantized (FP8) weights; dequantize the checkpoint to bfloat16 first")
    eps = float(hf.get("rms_norm_eps", 1e-6))
    if eps != 1e-6:
        _fail(model_type, f"rms_norm_eps {eps} (HF normalizes the MLA latents with 1e-6 regardless)")
    theta, _, scaling = _rope(hf, model_type)
    # transformers' yarn_apply_mscale: with RoPE scaling and mscale_all_dim the softmax scale gains mscale ** 2.
    mscale_all = _rope_settings(hf).get("mscale_all_dim") if scaling is not None else None
    for key, want in (("scoring_func", "sigmoid"), ("topk_method", "noaux_tc"), ("moe_layer_freq", 1)):
        if key in hf and hf[key] != want:
            _fail(model_type, f"{key}={hf[key]!r} (only {want!r} is DeepSeek-V3 routing)")
    d_model, n_layers, heads = int(hf["hidden_size"]), int(hf["num_hidden_layers"]), int(hf["num_attention_heads"])
    q_lora = hf.get("q_lora_rank")
    mla = MLAConfig(
        num_heads=heads, kv_lora_rank=int(hf["kv_lora_rank"]), q_lora_rank=int(q_lora) if q_lora else None,
        qk_nope_head_dim=int(hf["qk_nope_head_dim"]), qk_rope_head_dim=int(hf["qk_rope_head_dim"]),
        v_head_dim=int(hf["v_head_dim"]), rope_theta=theta, rope_scaling=scaling,
        yarn_factor=float(_rope_settings(hf)["factor"]) if mscale_all else 1.0,
        yarn_mscale=float(mscale_all) if mscale_all else 1.0,
    )
    first_dense = int(hf.get("first_k_dense_replace") or 0)
    mlp_types = tuple("dense" if i < first_dense else "moe" for i in range(n_layers))
    moe = None
    if "moe" in mlp_types:
        moe = MoEConfig(
            num_experts=int(hf["n_routed_experts"]), top_k=int(hf["num_experts_per_tok"]),
            d_ff_expert=int(hf["moe_intermediate_size"]), num_shared_experts=int(hf.get("n_shared_experts") or 0),
            router="sigmoid", normalize_top_k=bool(hf.get("norm_topk_prob", True)), capacity_factor=None,
            aux_loss_coef=float(hf.get("aux_loss_alpha", 1e-4)), router_bias=True,
            num_groups=int(hf.get("n_group") or 1), top_k_groups=int(hf.get("topk_group") or 1),
            routed_scaling_factor=float(hf.get("routed_scaling_factor", 1.0)),
        )
    config = ModelConfig(
        vocab_size=int(hf["vocab_size"]), d_model=d_model, n_layers=n_layers, d_ff=int(hf["intermediate_size"]),
        attention=AttentionConfig(num_heads=heads, num_kv_heads=heads, head_dim=mla.qk_head_dim),  # unused by MLA
        layer_types=("mla",) * n_layers, mla=mla, moe=moe, mlp_types=mlp_types, norm_eps=eps,
        tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
        initializer_range=float(hf.get("initializer_range", 0.02)),
    )
    return config.replace(**overrides) if overrides else config


def _deepseek_v3_hf_config(config: ModelConfig, dtype: str, base: Mapping[str, Any] | None) -> dict[str, Any]:
    def no(message: str) -> None:
        raise ValueError(f"cannot export to Hugging Face 'deepseek_v3': {message}")

    if set(config.resolved_layer_types) != {"mla"}:
        no("every layer must be an 'mla' layer")
    mla, moe, mlp = config.mla, config.moe, config.resolved_mlp_types
    first_dense = mlp.index("moe") if "moe" in mlp else len(mlp)
    if set(mlp[:first_dense]) - {"dense"} or set(mlp[first_dense:]) - {"moe"}:
        no(f"mlp types must be dense layers followed by moe layers; got {mlp}")
    if config.mtp is not None:
        no("MTP heads are not exported (set mtp=None)")
    if not mla.use_rope or mla.sliding_window is not None or mla.softmax_scale is not None:
        no("NoPE, sliding windows and a custom softmax_scale have no DeepSeek-V3 equivalent")
    if mla.yarn_factor != 1.0 and (mla.rope_scaling is None or mla.rope_scaling.factor != mla.yarn_factor):
        no("the YaRN softmax correction (yarn_factor) needs rope_scaling with the same factor")
    if config.norm_eps != 1e-6:
        no("norm_eps must be 1e-6 (HF normalizes the MLA latents with 1e-6)")
    if moe is not None and (moe.router != "sigmoid" or moe.capacity_factor is not None):
        no("DeepSeek-V3 routing is dropless sigmoid routing")
    if moe is not None and moe.shared_expert_gate:
        no("DeepSeek-V3 has no shared-expert gate (set moe.shared_expert_gate = False)")
    out: dict[str, Any] = dict(base or {})
    for stale in ("rope_scaling", "rope_parameters", "torch_dtype", "dtype", "transformers_version", "num_mtp_layers",
                  "quantization_config"):
        out.pop(stale, None)
    legacy, rope = _rope_dicts(mla, "deepseek_v3")
    if mla.yarn_factor != 1.0:
        legacy["mscale_all_dim"] = rope["mscale_all_dim"] = mla.yarn_mscale
    out.update(
        architectures=["DeepseekV3ForCausalLM"], model_type="deepseek_v3", vocab_size=config.vocab_size,
        hidden_size=config.d_model, intermediate_size=config.d_ff, num_hidden_layers=config.n_layers,
        num_attention_heads=mla.num_heads, num_key_value_heads=mla.num_heads, kv_lora_rank=mla.kv_lora_rank,
        q_lora_rank=mla.q_lora_rank, qk_nope_head_dim=mla.qk_nope_head_dim, qk_rope_head_dim=mla.qk_rope_head_dim,
        v_head_dim=mla.v_head_dim, first_k_dense_replace=first_dense, hidden_act="silu", rms_norm_eps=config.norm_eps,
        rope_theta=mla.rope_theta, rope_scaling=legacy, rope_parameters=rope,
        rope_interleave=bool(out.get("rope_interleave", True)), attention_bias=False, attention_dropout=0.0,
        tie_word_embeddings=config.tie_embeddings, initializer_range=config.initializer_range,
        num_nextn_predict_layers=0, torch_dtype=dtype, dtype=dtype,
    )
    out.setdefault("max_position_embeddings", 4096)
    if moe is not None:
        out.update(
            n_routed_experts=moe.num_experts, num_experts_per_tok=moe.top_k, moe_intermediate_size=moe.d_ff_expert,
            n_shared_experts=moe.num_shared_experts, norm_topk_prob=moe.normalize_top_k, n_group=moe.num_groups,
            topk_group=moe.top_k_groups, routed_scaling_factor=moe.routed_scaling_factor,
            aux_loss_alpha=moe.aux_loss_coef,
        )
    return out
