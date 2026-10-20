# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers (Apache-2.0): parameter names and layouts of src/transformers/models/{llama,mistral,
# qwen2,qwen3,phi3,mixtral,qwen3_moe,mamba2,deepseek_v3}/modeling_*.py and the expert fusion in conversion_mapping.py
"""Parameter name and layout mapping between Hugging Face state dicts and :class:`LanguageModel`.

HF linear weights are ``[out, in]`` and ours ``[in, out]``. Our SwiGLU keeps gate and
up in one ``[D, 2, F]`` weight (HF: ``gate_proj`` / ``up_proj``, or Phi-3's fused
``gate_up_proj`` with the gate first); MoE experts are ``gate_up [E, D, 2, F]`` and
``down [E, F, D]`` (HF hub files: one module per expert; ``transformers`` 5 in memory:
fused ``experts.gate_up_proj [E, 2F, D]`` and ``experts.down_proj [E, D, F]``, both read).
Phi-3's fused ``qkv_proj`` rows are ``[q | k | v]``. RoPE needs no permutation (our
``apply_rope`` uses the HF rotate-half layout) except for DeepSeek-V3's interleaved pairs
(see below). Layers are mapped one by one under the neutral names ``layers.<i>.<path>``;
:func:`stack_layers` / :func:`unstack_layers` move between those and the model's
(optionally scanned) ``blocks`` layout.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from jla.convert.hf_config import MOE_MODEL_TYPES, SUPPORTED_MODEL_TYPES
from jla.models.config import ModelConfig

__all__ = ["export_hf_state", "import_hf_state", "stack_layers", "unstack_layers"]

_IGNORED = re.compile(r"(^|\.)rotary_emb\.inv_freq$")


class _Taker:
    """Reads tensors from a state dict and remembers which keys were used."""

    def __init__(self, state: Mapping[str, np.ndarray]):
        self.state = state
        self.used: set[str] = set()

    def has(self, key: str) -> bool:
        return key in self.state

    def __call__(self, key: str) -> np.ndarray:
        if key not in self.state:
            raise KeyError(f"Hugging Face checkpoint lacks {key!r}")
        self.used.add(key)
        return np.asarray(self.state[key])

    def leftovers(self) -> list[str]:
        return sorted(k for k in self.state if k not in self.used and not _IGNORED.search(k))


def _layer_prefix(state: Mapping[str, np.ndarray]) -> str:
    if "model.embed_tokens.weight" in state:
        return "model."
    if "embed_tokens.weight" in state:
        return ""
    raise KeyError("no embed_tokens.weight in the checkpoint: not a decoder-only HF language model")


def _expect(name: str, array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name}: Hugging Face shape {tuple(array.shape)} != expected {tuple(shape)}")
    return array


def import_hf_state(
    state: Mapping[str, np.ndarray], config: ModelConfig, model_type: str, *, rope_interleave: bool = True
) -> dict[str, np.ndarray]:
    """HF tensors -> ``{"embed.weight", "final_norm.weight", ["head"], "layers.<i>.<path>"}`` (our layouts).

    Every checkpoint tensor must be consumed (``rotary_emb.inv_freq`` buffers excepted);
    a tied checkpoint may still carry ``lm_head.weight`` if it equals the embedding.
    ``rope_interleave`` is DeepSeek-V3's config flag (interleaved RoPE channel pairs).
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported Hugging Face model_type {model_type!r}")
    if model_type == "mamba2":
        return _import_mamba2(state, config)
    if model_type == "deepseek_v3":
        return _import_deepseek_v3(state, config, rope_interleave)
    take = _Taker(state)
    pre = _layer_prefix(state)
    d, attn = config.d_model, config.attention
    q_dim, kv_dim = attn.num_heads * attn.head_dim, attn.num_kv_heads * attn.head_dim
    out: dict[str, np.ndarray] = {
        "embed.weight": _expect("embed_tokens", take(f"{pre}embed_tokens.weight"), (config.vocab_size, d)),
        "final_norm.weight": take(f"{pre}norm.weight"),
    }
    if config.tie_embeddings:
        if take.has("lm_head.weight"):
            if not np.array_equal(take("lm_head.weight"), out["embed.weight"]):
                raise ValueError("tie_word_embeddings=True but lm_head.weight differs from embed_tokens.weight")
    else:
        out["head"] = _expect("lm_head", take("lm_head.weight"), (config.vocab_size, d))
    for i, mlp_type in enumerate(config.resolved_mlp_types):
        p, o = f"{pre}layers.{i}.", f"layers.{i}."
        out[o + "mixer_norm.weight"] = take(p + "input_layernorm.weight")
        out[o + "mlp_norm.weight"] = take(p + "post_attention_layernorm.weight")
        a = p + "self_attn."
        if model_type == "phi3":
            qkv = _expect(a + "qkv_proj", take(a + "qkv_proj.weight"), (q_dim + 2 * kv_dim, d))
            q, k, v = qkv[:q_dim], qkv[q_dim:q_dim + kv_dim], qkv[q_dim + kv_dim:]
        else:
            q, k, v = (take(a + f"{n}_proj.weight") for n in "qkv")
        out[o + "mixer.q_proj.weight"] = _expect(a + "q_proj", q, (q_dim, d)).T
        out[o + "mixer.k_proj.weight"] = _expect(a + "k_proj", k, (kv_dim, d)).T
        out[o + "mixer.v_proj.weight"] = _expect(a + "v_proj", v, (kv_dim, d)).T
        out[o + "mixer.o_proj.weight"] = _expect(a + "o_proj", take(a + "o_proj.weight"), (d, q_dim)).T
        if attn.qkv_bias:
            for n in "qkv":
                out[o + f"mixer.{n}_proj.bias"] = take(a + f"{n}_proj.bias")
        if attn.qk_norm:
            out[o + "mixer.q_norm.weight"] = take(a + "q_norm.weight")
            out[o + "mixer.k_norm.weight"] = take(a + "k_norm.weight")
        if mlp_type == "dense":
            m = p + "mlp."
            if model_type == "phi3":
                gate_up = _expect(m + "gate_up_proj", take(m + "gate_up_proj.weight"), (2 * config.d_ff, d))
                gate, up = gate_up[:config.d_ff], gate_up[config.d_ff:]
            else:
                gate, up = take(m + "gate_proj.weight"), take(m + "up_proj.weight")
            _expect(m + "gate_proj", gate, (config.d_ff, d))
            _expect(m + "up_proj", up, (config.d_ff, d))
            out[o + "mlp.gate_up"] = np.stack([gate.T, up.T], axis=1)
            out[o + "mlp.down.weight"] = _expect(m + "down_proj", take(m + "down_proj.weight"), (d, config.d_ff)).T
        elif mlp_type == "moe":
            out.update(_import_moe(take, p, o, config, model_type))
        else:
            raise ValueError(f"layer {i}: mlp type {mlp_type!r} has no Hugging Face mapping")
    leftovers = take.leftovers()
    if leftovers:
        raise ValueError(f"Hugging Face checkpoint has {len(leftovers)} unmapped tensors: {leftovers[:8]}")
    return out


def _import_moe(take: _Taker, p: str, o: str, config: ModelConfig, model_type: str) -> dict[str, np.ndarray]:
    moe, d = config.moe, config.d_model
    experts, f = moe.num_experts, moe.d_ff_expert
    hub = p + ("block_sparse_moe." if model_type == "mixtral" else "mlp.")
    fused = p + "mlp."
    names = ("w1", "w3", "w2") if model_type == "mixtral" else ("gate_proj", "up_proj", "down_proj")
    if take.has(fused + "experts.gate_up_proj"):  # transformers 5 in-memory layout (save_original_format=False)
        router = take(fused + "gate.weight")
        gate_up = _expect(fused + "experts.gate_up_proj", take(fused + "experts.gate_up_proj"), (experts, 2 * f, d))
        down = _expect(fused + "experts.down_proj", take(fused + "experts.down_proj"), (experts, d, f))
        gate, up = gate_up[:, :f], gate_up[:, f:]
    else:  # hub layout: one module per expert
        router = take(hub + "gate.weight")
        gate = np.stack([take(hub + f"experts.{e}.{names[0]}.weight") for e in range(experts)])
        up = np.stack([take(hub + f"experts.{e}.{names[1]}.weight") for e in range(experts)])
        down = np.stack([take(hub + f"experts.{e}.{names[2]}.weight") for e in range(experts)])
        _expect(hub + "experts.*.gate", gate, (experts, f, d))
        _expect(hub + "experts.*.up", up, (experts, f, d))
        _expect(hub + "experts.*.down", down, (experts, d, f))
    return {
        o + "mlp.router": _expect(hub + "gate", router, (experts, d)).T,
        o + "mlp.gate_up": np.stack([gate.transpose(0, 2, 1), up.transpose(0, 2, 1)], axis=2),
        o + "mlp.down": down.transpose(0, 2, 1),
    }


def export_hf_state(
    params: Mapping[str, np.ndarray],
    config: ModelConfig,
    model_type: str,
    *,
    moe_format: str = "hub",
    rope_interleave: bool = True,
) -> dict[str, np.ndarray]:
    """Inverse of :func:`import_hf_state` (``params`` in the ``layers.<i>.<path>`` naming).

    ``moe_format="hub"`` writes one module per expert (the layout of published
    checkpoints, which every ``transformers`` version loads); ``"fused"`` writes the
    ``transformers`` 5 in-memory layout.
    """
    if moe_format not in ("hub", "fused"):
        raise ValueError(f"moe_format must be 'hub' or 'fused'; got {moe_format!r}")
    if model_type == "mamba2":
        return _export_mamba2(params, config)
    if model_type == "deepseek_v3":
        return _export_deepseek_v3(params, config, rope_interleave, moe_format)
    attn = config.attention
    c = lambda a: np.ascontiguousarray(a)  # noqa: E731
    out = {"model.embed_tokens.weight": c(params["embed.weight"]), "model.norm.weight": c(params["final_norm.weight"])}
    if not config.tie_embeddings:
        out["lm_head.weight"] = c(params["head"])
    for i, mlp_type in enumerate(config.resolved_mlp_types):
        p, o = f"model.layers.{i}.", f"layers.{i}."
        out[p + "input_layernorm.weight"] = c(params[o + "mixer_norm.weight"])
        out[p + "post_attention_layernorm.weight"] = c(params[o + "mlp_norm.weight"])
        a = p + "self_attn."
        q, k, v = (params[o + f"mixer.{n}_proj.weight"].T for n in "qkv")
        if model_type == "phi3":
            out[a + "qkv_proj.weight"] = c(np.concatenate([q, k, v], axis=0))
        else:
            out.update({a + "q_proj.weight": c(q), a + "k_proj.weight": c(k), a + "v_proj.weight": c(v)})
        out[a + "o_proj.weight"] = c(params[o + "mixer.o_proj.weight"].T)
        if attn.qkv_bias:
            for n in "qkv":
                out[a + f"{n}_proj.bias"] = c(params[o + f"mixer.{n}_proj.bias"])
        if attn.qk_norm:
            out[a + "q_norm.weight"] = c(params[o + "mixer.q_norm.weight"])
            out[a + "k_norm.weight"] = c(params[o + "mixer.k_norm.weight"])
        if mlp_type == "dense":
            m = p + "mlp."
            gate_up = params[o + "mlp.gate_up"]
            gate, up = gate_up[:, 0].T, gate_up[:, 1].T
            if model_type == "phi3":
                out[m + "gate_up_proj.weight"] = c(np.concatenate([gate, up], axis=0))
            else:
                out.update({m + "gate_proj.weight": c(gate), m + "up_proj.weight": c(up)})
            out[m + "down_proj.weight"] = c(params[o + "mlp.down.weight"].T)
        elif mlp_type == "moe" and model_type in MOE_MODEL_TYPES:
            gate_up, down = params[o + "mlp.gate_up"], params[o + "mlp.down"]  # [E, D, 2, F], [E, F, D]
            gate, up, down = gate_up[:, :, 0].transpose(0, 2, 1), gate_up[:, :, 1].transpose(0, 2, 1), down.transpose(0, 2, 1)
            router = c(params[o + "mlp.router"].T)
            if moe_format == "fused":
                m = p + "mlp."
                out.update({m + "gate.weight": router, m + "experts.gate_up_proj": c(np.concatenate([gate, up], axis=1)),
                            m + "experts.down_proj": c(down)})
            else:
                m = p + ("block_sparse_moe." if model_type == "mixtral" else "mlp.")
                names = ("w1", "w3", "w2") if model_type == "mixtral" else ("gate_proj", "up_proj", "down_proj")
                out[m + "gate.weight"] = router
                for e in range(config.moe.num_experts):
                    out[m + f"experts.{e}.{names[0]}.weight"] = c(gate[e])
                    out[m + f"experts.{e}.{names[1]}.weight"] = c(up[e])
                    out[m + f"experts.{e}.{names[2]}.weight"] = c(down[e])
        else:
            raise ValueError(f"layer {i}: mlp type {mlp_type!r} cannot be exported as {model_type!r}")
    return out


def _period(config: ModelConfig) -> int:
    from jla.models.lm import layer_period

    return layer_period(config) if config.scan_layers else config.n_layers


def stack_layers(params: Mapping[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    """``layers.<i>.<path>`` -> the model's ``blocks.<j>.<path>`` (stacked over repeats when scanned)."""
    period = _period(config)
    out, per_layer = {}, {}
    for name, value in params.items():
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            out[name] = value
        else:
            per_layer.setdefault(match.group(2), {})[int(match.group(1))] = value
    for path, layers in per_layer.items():
        for j in range(period):
            members = list(range(j, config.n_layers, period))
            if not all(i in layers for i in members):
                continue  # this path belongs to another member of the group
            if config.scan_layers:
                out[f"blocks.{j}.{path}"] = np.stack([layers[i] for i in members])
            else:
                out[f"blocks.{j}.{path}"] = layers[j]
    return out


def unstack_layers(flat: Mapping[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    """Inverse of :func:`stack_layers`."""
    period = _period(config)
    out = {}
    for name, value in flat.items():
        match = re.fullmatch(r"blocks\.(\d+)\.(.+)", name)
        if match is None:
            out[name] = value
            continue
        j, path = int(match.group(1)), match.group(2)
        if config.scan_layers:
            members = list(range(j, config.n_layers, period))
            if value.shape[0] != len(members):
                raise ValueError(f"{name}: stacked axis {value.shape[0]} != {len(members)} repeats")
            for r, i in enumerate(members):
                out[f"layers.{i}.{path}"] = value[r]
        else:
            out[f"layers.{j}.{path}"] = value
    return out


# Mamba-2 ----------------------------------------------------------------------------------------
# HF ``in_proj`` rows are ``[z | x | B | C | dt]`` and ``conv1d`` channels ``[x | B | C]``; ours split
# them into ``zx_proj [D, 2, inner]``, ``bc_proj [D, 2, G*N]``, ``dt_proj [D, H]``, ``x_conv`` and
# ``bc_conv`` (conv weights ``[C, K]``, the same tap order as torch's padded ``Conv1d``).


def _mamba2_dims(config: ModelConfig) -> tuple[int, int, int]:
    ssm = config.ssm
    heads = ssm.num_heads(config.d_model)
    return heads * ssm.head_dim, ssm.n_groups * ssm.d_state, heads


def _import_mamba2(state: Mapping[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    take = _Taker(state)
    if "backbone.embeddings.weight" in state:
        pre = "backbone."
    elif "embeddings.weight" in state:
        pre = ""
    else:
        raise KeyError("no backbone.embeddings.weight in the checkpoint: not a Mamba2ForCausalLM checkpoint")
    d = config.d_model
    inner, bc, heads = _mamba2_dims(config)
    kernel = config.ssm.conv_size
    out = {
        "embed.weight": _expect("embeddings", take(pre + "embeddings.weight"), (config.vocab_size, d)),
        "final_norm.weight": take(pre + "norm_f.weight"),
    }
    if config.tie_embeddings:
        if take.has("lm_head.weight") and not np.array_equal(take("lm_head.weight"), out["embed.weight"]):
            raise ValueError("tie_word_embeddings=True but lm_head.weight differs from the embeddings")
    else:
        out["head"] = _expect("lm_head", take("lm_head.weight"), (config.vocab_size, d))
    for i in range(config.n_layers):
        p, o = f"{pre}layers.{i}.", f"layers.{i}."
        m = p + "mixer."
        w = _expect(m + "in_proj", take(m + "in_proj.weight"), (2 * inner + 2 * bc + heads, d))
        z, x, b, c, dt = np.split(w, np.cumsum([inner, inner, bc, bc]), axis=0)
        conv_w = _expect(m + "conv1d", take(m + "conv1d.weight"), (inner + 2 * bc, 1, kernel))[:, 0]
        conv_b = take(m + "conv1d.bias")
        out.update({
            o + "mixer_norm.weight": take(p + "norm.weight"),
            o + "mixer.zx_proj.weight": np.stack([z.T, x.T], axis=1),
            o + "mixer.bc_proj.weight": np.stack([b.T, c.T], axis=1),
            o + "mixer.dt_proj.weight": dt.T,
            o + "mixer.x_conv.weight": conv_w[:inner], o + "mixer.x_conv.bias": conv_b[:inner],
            o + "mixer.bc_conv.weight": conv_w[inner:], o + "mixer.bc_conv.bias": conv_b[inner:],
            o + "mixer.A_log": take(m + "A_log"), o + "mixer.dt_bias": take(m + "dt_bias"), o + "mixer.D": take(m + "D"),
            o + "mixer.norm.weight": take(m + "norm.weight"),
            o + "mixer.out_proj.weight": _expect(m + "out_proj", take(m + "out_proj.weight"), (d, inner)).T,
        })
    leftovers = take.leftovers()
    if leftovers:
        raise ValueError(f"Hugging Face checkpoint has {len(leftovers)} unmapped tensors: {leftovers[:8]}")
    return out


def _export_mamba2(params: Mapping[str, np.ndarray], config: ModelConfig) -> dict[str, np.ndarray]:
    c = np.ascontiguousarray
    out = {"backbone.embeddings.weight": c(params["embed.weight"]), "backbone.norm_f.weight": c(params["final_norm.weight"])}
    if not config.tie_embeddings:
        out["lm_head.weight"] = c(params["head"])
    for i in range(config.n_layers):
        p, o = f"backbone.layers.{i}.", f"layers.{i}."
        m = p + "mixer."
        zx, bcw = params[o + "mixer.zx_proj.weight"], params[o + "mixer.bc_proj.weight"]
        rows = [zx[:, 0].T, zx[:, 1].T, bcw[:, 0].T, bcw[:, 1].T, params[o + "mixer.dt_proj.weight"].T]
        conv = np.concatenate([params[o + "mixer.x_conv.weight"], params[o + "mixer.bc_conv.weight"]], axis=0)
        out.update({
            p + "norm.weight": c(params[o + "mixer_norm.weight"]),
            m + "in_proj.weight": c(np.concatenate(rows, axis=0)),
            m + "conv1d.weight": c(conv[:, None, :]),
            m + "conv1d.bias": c(np.concatenate([params[o + "mixer.x_conv.bias"], params[o + "mixer.bc_conv.bias"]])),
            m + "A_log": c(params[o + "mixer.A_log"]), m + "dt_bias": c(params[o + "mixer.dt_bias"]),
            m + "D": c(params[o + "mixer.D"]), m + "norm.weight": c(params[o + "mixer.norm.weight"]),
            m + "out_proj.weight": c(params[o + "mixer.out_proj.weight"].T),
        })
    return out


# DeepSeek-V3 ------------------------------------------------------------------------------------
# With ``rope_interleave`` (the default) HF rotates interleaved channel pairs ``(x0, x1), (x2, x3), ...``
# of each RoPE slice, which equals the rotate-half RoPE of the slice permuted to ``[x0, x2, ... | x1, x3, ...]``;
# permuting the rows that produce ``q_pe`` (in ``q_proj`` / ``q_b_proj``, per head) and ``k_pe`` (in
# ``kv_a_proj_with_mqa``) makes our rotate-half ``apply_rope`` exact. Layers past ``num_hidden_layers``
# (the MTP modules of the released checkpoints) are skipped, as ``transformers`` does.


def _deepseek_rows(config: ModelConfig, interleave: bool) -> tuple[np.ndarray, np.ndarray]:
    """Row orders that turn HF's ``q`` and ``kv_a`` projection outputs into ours."""
    mla = config.mla
    nope, rope = mla.qk_nope_head_dim, mla.qk_rope_head_dim
    perm = np.concatenate([np.arange(0, rope, 2), np.arange(1, rope, 2)]) if interleave else np.arange(rope)
    per_head = np.concatenate([np.arange(nope), nope + perm])
    q_rows = (np.arange(mla.num_heads)[:, None] * (nope + rope) + per_head[None]).reshape(-1)
    kv_rows = np.concatenate([np.arange(mla.kv_lora_rank), mla.kv_lora_rank + perm])
    return q_rows, kv_rows


def _import_deepseek_v3(state: Mapping[str, np.ndarray], config: ModelConfig, interleave: bool) -> dict[str, np.ndarray]:
    take = _Taker(state)
    pre = _layer_prefix(state)
    d, mla = config.d_model, config.mla
    heads, qk = mla.num_heads, mla.qk_head_dim
    q_rows, kv_rows = _deepseek_rows(config, interleave)
    out: dict[str, np.ndarray] = {
        "embed.weight": _expect("embed_tokens", take(f"{pre}embed_tokens.weight"), (config.vocab_size, d)),
        "final_norm.weight": take(f"{pre}norm.weight"),
    }
    if config.tie_embeddings:
        if take.has("lm_head.weight") and not np.array_equal(take("lm_head.weight"), out["embed.weight"]):
            raise ValueError("tie_word_embeddings=True but lm_head.weight differs from embed_tokens.weight")
    else:
        out["head"] = _expect("lm_head", take("lm_head.weight"), (config.vocab_size, d))
    for i, mlp_type in enumerate(config.resolved_mlp_types):
        p, o = f"{pre}layers.{i}.", f"layers.{i}."
        a, x = p + "self_attn.", o + "mixer."
        out[o + "mixer_norm.weight"] = take(p + "input_layernorm.weight")
        out[o + "mlp_norm.weight"] = take(p + "post_attention_layernorm.weight")
        if mla.q_lora_rank:
            out[x + "q_a_proj.weight"] = _expect(a + "q_a_proj", take(a + "q_a_proj.weight"), (mla.q_lora_rank, d)).T
            out[x + "q_a_norm.weight"] = take(a + "q_a_layernorm.weight")
            q_b = _expect(a + "q_b_proj", take(a + "q_b_proj.weight"), (heads * qk, mla.q_lora_rank))
            out[x + "q_b_proj.weight"] = q_b[q_rows].T
        else:
            out[x + "q_proj.weight"] = _expect(a + "q_proj", take(a + "q_proj.weight"), (heads * qk, d))[q_rows].T
        kv_a = _expect(a + "kv_a_proj_with_mqa", take(a + "kv_a_proj_with_mqa.weight"),
                       (mla.kv_lora_rank + mla.qk_rope_head_dim, d))
        out[x + "kv_a_proj.weight"] = kv_a[kv_rows].T
        out[x + "kv_a_norm.weight"] = take(a + "kv_a_layernorm.weight")
        out[x + "kv_b_proj.weight"] = _expect(a + "kv_b_proj", take(a + "kv_b_proj.weight"),
                                              (heads * (mla.qk_nope_head_dim + mla.v_head_dim), mla.kv_lora_rank)).T
        out[x + "o_proj.weight"] = _expect(a + "o_proj", take(a + "o_proj.weight"), (d, heads * mla.v_head_dim)).T
        m = p + "mlp."
        if mlp_type == "dense":
            gate = _expect(m + "gate_proj", take(m + "gate_proj.weight"), (config.d_ff, d))
            up = _expect(m + "up_proj", take(m + "up_proj.weight"), (config.d_ff, d))
            out[o + "mlp.gate_up"] = np.stack([gate.T, up.T], axis=1)
            out[o + "mlp.down.weight"] = _expect(m + "down_proj", take(m + "down_proj.weight"), (d, config.d_ff)).T
            continue
        moe = config.moe
        out.update(_import_moe(take, p, o, config, "deepseek_v3"))
        bias = _expect(m + "gate.e_score_correction_bias", take(m + "gate.e_score_correction_bias"), (moe.num_experts,))
        out[o + "mlp.router_bias"] = bias.astype(np.float32)
        if moe.num_shared_experts:
            s, width = m + "shared_experts.", moe.num_shared_experts * moe.d_ff_expert
            gate = _expect(s + "gate_proj", take(s + "gate_proj.weight"), (width, d))
            up = _expect(s + "up_proj", take(s + "up_proj.weight"), (width, d))
            out[o + "mlp.shared.gate_up"] = np.stack([gate.T, up.T], axis=1)
            out[o + "mlp.shared.down.weight"] = _expect(s + "down_proj", take(s + "down_proj.weight"), (d, width)).T
    mtp = re.compile(rf"^{re.escape(pre)}layers\.(\d+)\.")
    leftovers = [k for k in take.leftovers() if not ((hit := mtp.match(k)) and int(hit.group(1)) >= config.n_layers)]
    if leftovers:
        raise ValueError(f"Hugging Face checkpoint has {len(leftovers)} unmapped tensors: {leftovers[:8]}")
    return out


def _export_deepseek_v3(
    params: Mapping[str, np.ndarray], config: ModelConfig, interleave: bool, moe_format: str
) -> dict[str, np.ndarray]:
    c = np.ascontiguousarray
    mla = config.mla
    q_rows, kv_rows = (np.argsort(rows) for rows in _deepseek_rows(config, interleave))
    out = {"model.embed_tokens.weight": c(params["embed.weight"]), "model.norm.weight": c(params["final_norm.weight"])}
    if not config.tie_embeddings:
        out["lm_head.weight"] = c(params["head"])
    for i, mlp_type in enumerate(config.resolved_mlp_types):
        p, o = f"model.layers.{i}.", f"layers.{i}."
        a, x, m = p + "self_attn.", o + "mixer.", p + "mlp."
        out[p + "input_layernorm.weight"] = c(params[o + "mixer_norm.weight"])
        out[p + "post_attention_layernorm.weight"] = c(params[o + "mlp_norm.weight"])
        if mla.q_lora_rank:
            out[a + "q_a_proj.weight"] = c(params[x + "q_a_proj.weight"].T)
            out[a + "q_a_layernorm.weight"] = c(params[x + "q_a_norm.weight"])
            out[a + "q_b_proj.weight"] = c(params[x + "q_b_proj.weight"].T[q_rows])
        else:
            out[a + "q_proj.weight"] = c(params[x + "q_proj.weight"].T[q_rows])
        out[a + "kv_a_proj_with_mqa.weight"] = c(params[x + "kv_a_proj.weight"].T[kv_rows])
        out[a + "kv_a_layernorm.weight"] = c(params[x + "kv_a_norm.weight"])
        out[a + "kv_b_proj.weight"] = c(params[x + "kv_b_proj.weight"].T)
        out[a + "o_proj.weight"] = c(params[x + "o_proj.weight"].T)
        if mlp_type == "dense":
            gate_up = params[o + "mlp.gate_up"]
            out.update({m + "gate_proj.weight": c(gate_up[:, 0].T), m + "up_proj.weight": c(gate_up[:, 1].T),
                        m + "down_proj.weight": c(params[o + "mlp.down.weight"].T)})
            continue
        gate_up, down = params[o + "mlp.gate_up"], params[o + "mlp.down"]  # [E, D, 2, F], [E, F, D]
        gate, up, down = gate_up[:, :, 0].transpose(0, 2, 1), gate_up[:, :, 1].transpose(0, 2, 1), down.transpose(0, 2, 1)
        out[m + "gate.weight"] = c(params[o + "mlp.router"].T)
        bias = params.get(o + "mlp.router_bias")
        out[m + "gate.e_score_correction_bias"] = c(np.zeros(config.moe.num_experts, np.float32) if bias is None else bias)
        if moe_format == "fused":
            out.update({m + "experts.gate_up_proj": c(np.concatenate([gate, up], axis=1)), m + "experts.down_proj": c(down)})
        else:
            for e in range(config.moe.num_experts):
                out.update({m + f"experts.{e}.gate_proj.weight": c(gate[e]), m + f"experts.{e}.up_proj.weight": c(up[e]),
                            m + f"experts.{e}.down_proj.weight": c(down[e])})
        if config.moe.num_shared_experts:
            s, shared = m + "shared_experts.", params[o + "mlp.shared.gate_up"]
            out.update({s + "gate_proj.weight": c(shared[:, 0].T), s + "up_proj.weight": c(shared[:, 1].T),
                        s + "down_proj.weight": c(params[o + "mlp.shared.down.weight"].T)})
    return out
