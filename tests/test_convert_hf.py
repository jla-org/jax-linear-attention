# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face import/export: logit parity against ``transformers`` and exact weight round trips.

Tiny random models are built from configs on CPU (no downloads); ``torch`` and
``transformers`` are test-only dependencies and the module is skipped without them.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from jla.convert import from_hf_config, load_hf_checkpoint, save_hf_checkpoint, to_hf_config
from jla.convert.safetensors_io import load_checkpoint_dir, read_safetensors, save_checkpoint_dir, write_safetensors
from jla.models.config import MTPConfig, RoPEScalingConfig

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def _float32_matmuls():
    """True float32 matmuls on every backend (GPUs default to TF32), so the tolerance means the same everywhere."""
    with jax.default_matmul_precision("highest"):
        yield


V5 = int(transformers.__version__.split(".")[0]) >= 5
SMALL = dict(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5)


def _rope(theta: float, partial: float | None = None, scaling: dict | None = None) -> dict:
    if V5:
        params = {"rope_type": "default", **(scaling or {}), "rope_theta": theta}
        if partial is not None:
            params["partial_rotary_factor"] = partial
        return {"rope_parameters": params}
    out = {"rope_theta": theta, "rope_scaling": scaling}
    if partial is not None:
        out["partial_rotary_factor"] = partial
    return out


# Context 64 with head_dim 16 puts the 8 frequencies in all three llama3 bands (kept, smoothed, divided).
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


CASES = {
    "llama": dict(model_type="llama", **SMALL, **_rope(500_000.0)),
    "llama31_rope_scaling": dict(model_type="llama", **SMALL, **_rope(10_000.0, scaling=LLAMA3)),
    "qwen3_linear_rope": dict(model_type="qwen3", **SMALL, head_dim=16,
                              **_rope(10_000.0, scaling={"rope_type": "linear", "factor": 4.0})),
    # YaRN as Qwen2.5 long-context configs set it (attention_factor derived: 0.1 ln 4 + 1); the ramp spans channels.
    "qwen2_yarn": dict(model_type="qwen2", **SMALL, head_dim=16,
                       **_rope(10_000.0, scaling={"rope_type": "yarn", "factor": 4.0,
                                                  "original_max_position_embeddings": 32})),
    "llama_tied_head_dim": dict(model_type="llama", **{**SMALL, "num_key_value_heads": 1}, head_dim=24,
                                tie_word_embeddings=True, **_rope(10_000.0)),
    "mistral_window": dict(model_type="mistral", **SMALL, head_dim=16, sliding_window=7, **_rope(10_000.0)),
    "qwen2_bias_mixed_window": dict(model_type="qwen2", **SMALL, tie_word_embeddings=True, use_sliding_window=True,
                                    sliding_window=6, max_window_layers=1, **_rope(1_000_000.0)),
    "qwen3": dict(model_type="qwen3", **SMALL, head_dim=32, **_rope(1_000_000.0)),
    "phi3_partial_rotary": dict(model_type="phi3", **SMALL, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                **_rope(10_000.0, 0.5)),
    "mixtral": dict(model_type="mixtral", **{**SMALL, "intermediate_size": 48}, num_local_experts=4,
                    num_experts_per_tok=2, **_rope(1_000_000.0)),
    "qwen3_moe": dict(model_type="qwen3_moe", **SMALL, head_dim=16, num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=32, mlp_only_layers=[1], norm_topk_prob=False, **_rope(10_000.0)),
    "qwen3_moe_norm_step": dict(model_type="qwen3_moe", **{**SMALL, "num_hidden_layers": 4}, head_dim=16,
                                num_experts=6, num_experts_per_tok=3, moe_intermediate_size=24, decoder_sparse_step=2,
                                norm_topk_prob=True, tie_word_embeddings=True, **_rope(10_000.0)),
    # n_groups=1: HF's PyTorch gated norm ignores groups (state-spaces/mamba and ours normalize per group).
    "mamba2": dict(model_type="mamba2", vocab_size=97, hidden_size=64, num_hidden_layers=2, num_heads=8, head_dim=16,
                   expand=2, state_size=16, n_groups=1, conv_kernel=4, chunk_size=16, layer_norm_epsilon=1e-5),
    # MLA with a q LoRA and interleaved RoPE pairs; a dense first layer; 8 experts in 4 groups (2 groups per
    # token), the expert-choice bias, a shared expert and routed scaling.
    "deepseek_v3": dict(model_type="deepseek_v3", vocab_size=97, hidden_size=64, intermediate_size=96,
                        moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                        n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5, kv_lora_rank=16,
                        q_lora_rank=24, qk_rope_head_dim=8, v_head_dim=12, qk_nope_head_dim=16, n_group=4, topk_group=2,
                        num_experts_per_tok=3, first_k_dense_replace=1, norm_topk_prob=True, rms_norm_eps=1e-6,
                        max_position_embeddings=128, **_rope(10_000.0)),
    # Full-rank queries, rotate-half RoPE, no groups, unnormalized gates, two shared experts, tied embeddings.
    "deepseek_v3_plain": dict(model_type="deepseek_v3", vocab_size=97, hidden_size=64, intermediate_size=96,
                              moe_intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=2, n_shared_experts=2, n_routed_experts=6, routed_scaling_factor=1.5,
                              kv_lora_rank=32, q_lora_rank=None, qk_rope_head_dim=16, v_head_dim=16,
                              qk_nope_head_dim=8, n_group=1, topk_group=1, num_experts_per_tok=2,
                              first_k_dense_replace=0, norm_topk_prob=False, rope_interleave=False,
                              tie_word_embeddings=True, rms_norm_eps=1e-6, max_position_embeddings=128,
                              **_rope(50_000.0)),
    # DeepSeek-V3's YaRN (mscale != mscale_all_dim, so both the rotated-channel factor and the softmax
    # correction differ from 1).
    "deepseek_v3_yarn": dict(model_type="deepseek_v3", vocab_size=97, hidden_size=64, intermediate_size=96,
                             moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=4, n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=2.0,
                             kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, v_head_dim=12, qk_nope_head_dim=16,
                             n_group=2, topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1,
                             norm_topk_prob=True, rms_norm_eps=1e-6, max_position_embeddings=640,
                             **_rope(10_000.0, scaling={"rope_type": "yarn", "factor": 40.0,
                                                        "original_max_position_embeddings": 16, "beta_fast": 32,
                                                        "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 0.8})),
}
if V5:  # transformers 5 Qwen3-MoE puts the window on every layer (max_window_layers is not read)
    CASES["qwen3_moe_window"] = dict(model_type="qwen3_moe", **SMALL, head_dim=16, num_experts=4, num_experts_per_tok=2,
                                     moe_intermediate_size=32, norm_topk_prob=True, use_sliding_window=True,
                                     sliding_window=6, max_window_layers=1, **_rope(10_000.0))
TOL = 1e-4  # float32 logits; the observed error is ~1e-6 (printed by the tests)


def _eager(model_type: str) -> dict:
    return {} if model_type == "mamba2" else {"attn_implementation": "eager"}


def _hf_model(spec: dict, seed: int = 0):
    spec = dict(spec)
    model_type = spec.pop("model_type")
    cfg = transformers.AutoConfig.for_model(model_type, **spec)
    model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, **_eager(model_type))
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=gen))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):  # DeepSeek-V3's expert-choice bias (zero at init)
                b.copy_(0.2 * torch.randn(b.shape, generator=gen))
    return model.eval()


def _ids(vocab: int, batch: int = 2, length: int = 24, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=(batch, length)).astype(np.int32)


def _hf_logits(model, ids: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(ids.astype(np.int64))).logits.float().numpy()


def _our_logits(model, ids: np.ndarray) -> np.ndarray:
    out = model(jnp.asarray(ids), inference=True)
    return np.asarray(model.logits(out.hidden))


@pytest.fixture(scope="module", params=sorted(CASES))
def saved(request, tmp_path_factory):
    name = request.param
    model = _hf_model(CASES[name])
    directory = tmp_path_factory.mktemp(name) / "hf"
    model.save_pretrained(directory)
    ids = _ids(CASES[name]["vocab_size"])
    return name, directory, ids, _hf_logits(model, ids)


@pytest.mark.parametrize("scan_layers", [True, False])
def test_import_logit_parity(saved, scan_layers):
    name, directory, ids, expected = saved
    model, config = load_hf_checkpoint(directory, compute_dtype="float32", scan_layers=scan_layers)
    got = _our_logits(model, ids)
    err = float(np.max(np.abs(got - expected)))
    print(f"{name} scan={scan_layers}: max |logit diff| = {err:.2e} (logit scale {np.max(np.abs(expected)):.2f})")
    assert err < TOL, (name, err)


@pytest.mark.parametrize("name, change", [
    ("llama31_rope_scaling", {"rope_scaling": None}),
    ("qwen3_linear_rope", {"rope_scaling": None}),
    ("qwen2_yarn", {"rope_scaling": None}),
    ("mistral_window", {"sliding_window": None}),
    ("phi3_partial_rotary", {"rotary_fraction": 1.0}),
])
def test_parity_tolerance_detects_a_dropped_option(name, change):
    """Negative control: the same weights without one mapped option miss the HF logits by far more than TOL."""
    import dataclasses

    from jla.convert import assemble_model, import_hf_state, read_hf_config
    from jla.convert.safetensors_io import load_checkpoint_dir as load_dir

    model = _hf_model(CASES[name])
    ids = _ids(CASES[name]["vocab_size"])
    expected = _hf_logits(model, ids)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        config = from_hf_config(read_hf_config(tmp), compute_dtype="float32")
        params = import_hf_state(load_dir(tmp), config, CASES[name]["model_type"])
        wrong = config.replace(attention=dataclasses.replace(config.attention, **change))
        err = float(np.max(np.abs(_our_logits(assemble_model(wrong, params), ids) - expected)))
    print(f"{name} without {change}: max |logit diff| = {err:.2e}")
    assert err > 10 * TOL, (name, err)


def test_deepseek_v3_mapping_details_matter(tmp_path):
    """Negative controls: skipping the RoPE pair permutation, the expert-choice bias, the group limit or
    the routed scaling each misses the HF logits by far more than TOL."""
    import dataclasses

    from jla.convert import assemble_model, import_hf_state, read_hf_config

    model = _hf_model(CASES["deepseek_v3"])
    ids = _ids(97)
    expected = _hf_logits(model, ids)
    model.save_pretrained(tmp_path)
    config = from_hf_config(read_hf_config(tmp_path), compute_dtype="float32")
    state = load_checkpoint_dir(tmp_path)
    params = import_hf_state(state, config, "deepseek_v3")
    assert float(np.max(np.abs(_our_logits(assemble_model(config, params), ids) - expected))) < TOL
    unpermuted = import_hf_state(state, config, "deepseek_v3", rope_interleave=False)
    no_bias = {k: np.zeros_like(v) if k.endswith("router_bias") else v for k, v in params.items()}
    wrong = {
        "rope pairs not permuted": (config, unpermuted),
        "no expert-choice bias": (config, no_bias),
        "no group limit": (config.replace(moe=dataclasses.replace(config.moe, num_groups=1, top_k_groups=1)), params),
        "no routed scaling": (config.replace(moe=dataclasses.replace(config.moe, routed_scaling_factor=1.0)), params),
    }
    for what, (cfg, p) in wrong.items():
        err = float(np.max(np.abs(_our_logits(assemble_model(cfg, p), ids) - expected)))
        print(f"deepseek_v3 with {what}: max |logit diff| = {err:.2e}")
        assert err > 10 * TOL, (what, err)


def test_decode_matches_hf(saved):
    """Prefill half the prompt, then decode token by token through the caches: same logits as HF."""
    name, directory, ids, expected = saved
    model, _ = load_hf_checkpoint(directory, compute_dtype="float32")
    batch, length = ids.shape
    positions = np.broadcast_to(np.arange(length, dtype=np.int32), ids.shape)
    cache = model.init_cache(batch, length)
    split = length // 2
    logits, cache = model.decode_step(cache, jnp.asarray(ids[:, :split]), jnp.asarray(positions[:, :split]))
    pieces = [np.asarray(logits)]
    for t in range(split, length):
        logits, cache = model.decode_step(cache, jnp.asarray(ids[:, t:t + 1]), jnp.asarray(positions[:, t:t + 1]))
        pieces.append(np.asarray(logits))
    err = float(np.max(np.abs(np.concatenate(pieces, axis=1) - expected)))
    print(f"{name} decode: max |logit diff| = {err:.2e}")
    assert err < TOL, (name, err)


def test_export_round_trip_is_exact(saved, tmp_path):
    name, directory, ids, expected = saved
    original = {k: np.array(v) for k, v in load_checkpoint_dir(directory).items()}
    model, config = load_hf_checkpoint(directory, compute_dtype="float32")
    fmt = "fused" if any(k.endswith("experts.gate_up_proj") for k in original) else "hub"
    model_type = CASES[name]["model_type"]
    save_hf_checkpoint(model, tmp_path / "out", model_type, base_config=directory, moe_format=fmt)
    exported = load_checkpoint_dir(tmp_path / "out")
    assert sorted(exported) == sorted(original)
    for key, value in original.items():
        assert exported[key].dtype == value.dtype, key
        assert np.array_equal(exported[key], value), key
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "out", **_eager(model_type),
                                                                 dtype=torch.float32).eval()
    np.testing.assert_array_equal(_hf_logits(reloaded, ids), expected)
    # The exported config maps back to the same ModelConfig.
    again = from_hf_config(json.loads((tmp_path / "out" / "config.json").read_text()), compute_dtype="float32")
    assert again == config


def test_bfloat16_export_matches_torch_cast(tmp_path):
    model = _hf_model(CASES["qwen3"])
    model.save_pretrained(tmp_path / "hf")
    ours, _ = load_hf_checkpoint(tmp_path / "hf")
    save_hf_checkpoint(ours, tmp_path / "bf16", "qwen3", dtype="bfloat16", max_shard_bytes=40_000)
    assert (tmp_path / "bf16" / "model.safetensors.index.json").exists()  # sharded
    exported = load_checkpoint_dir(tmp_path / "bf16")
    reference = {k: v.to(torch.bfloat16).float().numpy() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    reference["lm_head.weight"] = model.lm_head.weight.detach().to(torch.bfloat16).float().numpy()
    assert sorted(exported) == sorted(reference)
    for key, value in exported.items():
        assert value.dtype == np.dtype(ml_dtypes.bfloat16), key
        np.testing.assert_array_equal(value.astype(np.float32), reference[key], err_msg=key)
    hf_bf16 = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "bf16", dtype=torch.bfloat16)
    for key, value in hf_bf16.state_dict().items():
        np.testing.assert_array_equal(value.float().numpy(), reference[key], err_msg=key)
    # Importing the sharded bfloat16 checkpoint: bf16 parameters, or float32 ones holding the same values.
    for param_dtype in ("bfloat16", "float32"):
        again, _ = load_hf_checkpoint(tmp_path / "bf16", param_dtype=param_dtype, compute_dtype="float32")
        assert again.embed.weight.dtype == jnp.dtype(param_dtype)
        np.testing.assert_array_equal(np.asarray(again.embed.weight, np.float32), reference["model.embed_tokens.weight"])
        ids = _ids(97, seed=2)
        with torch.no_grad():
            want = hf_bf16.float()(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
        assert float(np.max(np.abs(_our_logits(again, ids) - want))) < TOL


def test_fused_moe_layout_imports(tmp_path):
    if not V5:
        pytest.skip("transformers < 5 has no fused expert layout")
    model = _hf_model(CASES["mixtral"], seed=3)
    model.save_pretrained(tmp_path / "hf", save_original_format=False)
    keys = load_checkpoint_dir(tmp_path / "hf")
    assert any(k.endswith("experts.gate_up_proj") for k in keys)
    ids = _ids(97, seed=5)
    ours, _ = load_hf_checkpoint(tmp_path / "hf", compute_dtype="float32")
    assert float(np.max(np.abs(_our_logits(ours, ids) - _hf_logits(model, ids)))) < TOL


def test_safetensors_io_matches_reference_library(tmp_path):
    from safetensors.numpy import load_file
    from safetensors.torch import save_file as torch_save

    rng = np.random.default_rng(0)
    tensors = {"a": rng.standard_normal((3, 5)).astype(np.float32), "b": np.arange(7, dtype=np.int64),
               "c": rng.standard_normal((2, 2, 2)).astype(np.float16), "empty": np.zeros((0, 4), np.float32),
               "s": np.array(3.5, np.float64)}
    write_safetensors(tmp_path / "x.safetensors", tensors, {"format": "pt"})
    theirs = load_file(tmp_path / "x.safetensors")
    for key, value in tensors.items():
        np.testing.assert_array_equal(theirs[key], value)
    bf = torch.randn(4, 6, dtype=torch.float32).to(torch.bfloat16)
    torch_save({"w": bf, "i": torch.arange(5, dtype=torch.int32)}, tmp_path / "t.safetensors")
    mine = read_safetensors(tmp_path / "t.safetensors")
    assert mine["w"].dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(mine["w"].astype(np.float32), bf.float().numpy())
    np.testing.assert_array_equal(mine["i"], np.arange(5, dtype=np.int32))
    paths = save_checkpoint_dir(tmp_path / "sharded", tensors, max_shard_bytes=64)
    assert len(paths) > 1
    loaded = load_checkpoint_dir(tmp_path / "sharded")
    assert sorted(loaded) == sorted(tensors)
    for key, value in tensors.items():
        np.testing.assert_array_equal(loaded[key], value)
    # Both layouts in one directory (written by another tool): model.safetensors wins, as in from_pretrained.
    write_safetensors(tmp_path / "sharded" / "model.safetensors", {"only": np.ones(3, np.float32)}, {"format": "pt"})
    assert sorted(load_checkpoint_dir(tmp_path / "sharded")) == ["only"]


def test_v4_style_config_and_rejections():
    base = dict(model_type="llama", vocab_size=10, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=500000.0,
                rope_scaling=None, tie_word_embeddings=False, hidden_act="silu")
    config = from_hf_config(base)
    assert config.attention.rope_theta == 500000.0 and config.attention.head_dim == 8
    assert not config.attention.qk_norm and not config.tie_embeddings and config.norm_eps == 1e-5
    scaled = from_hf_config({**base, "rope_scaling": {**LLAMA3, "original_max_position_embeddings": 8192}})
    assert scaled.attention.rope_scaling == RoPEScalingConfig("llama3", 8.0, 1.0, 4.0, 8192)
    yarn = from_hf_config({**base, "max_position_embeddings": 4096, "rope_scaling": {"type": "yarn", "factor": 4.0}})
    import math

    assert yarn.attention.rope_scaling == RoPEScalingConfig("yarn", 4.0, original_max_position_embeddings=4096,
                                                            attention_factor=0.1 * math.log(4.0) + 1.0)
    rejected = [
        {"rope_scaling": {"type": "dynamic", "factor": 2.0}},
        {"rope_scaling": {"type": "longrope", "factor": 4.0}},
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"model_type": "gpt2"},
        {"partial_rotary_factor": 0.5},
    ]
    for change in rejected:
        with pytest.raises(ValueError):
            from_hf_config({**base, **change})
    with pytest.raises(ValueError):
        to_hf_config(config, "qwen3")  # qk_norm mismatch
    with pytest.raises(ValueError):
        to_hf_config(config.replace(mtp=MTPConfig()), "llama")


def test_unmapped_tensor_is_rejected(tmp_path):
    model = _hf_model(CASES["llama"])
    model.save_pretrained(tmp_path / "hf")
    state = {k: np.array(v) for k, v in load_checkpoint_dir(tmp_path / "hf").items()}
    state["model.layers.0.self_attn.extra.weight"] = np.zeros((2, 2), np.float32)
    save_checkpoint_dir(tmp_path / "bad", state)
    (tmp_path / "bad" / "config.json").write_text((tmp_path / "hf" / "config.json").read_text())
    with pytest.raises(ValueError, match="unmapped"):
        load_hf_checkpoint(tmp_path / "bad")


def test_cli_import_export(tmp_path, capsys):
    from jla.cli import main
    from jla.convert.cli import _read_model_config

    model = _hf_model(CASES["qwen2_bias_mixed_window"], seed=7)
    model.save_pretrained(tmp_path / "hf")
    (tmp_path / "hf" / "tokenizer.json").write_text('{"fake": "tokenizer"}')
    assert main(["convert", "import", "--hf", str(tmp_path / "hf"), "--output", str(tmp_path / "jla")]) == 0
    assert (tmp_path / "jla" / "tokenizer.json").read_text() == '{"fake": "tokenizer"}'
    config = _read_model_config(str(tmp_path / "jla" / "model.toml"))
    assert config == from_hf_config(tmp_path / "hf")
    assert main(["convert", "export", "--checkpoint", str(tmp_path / "jla"), "--model-config",
                 str(tmp_path / "jla" / "model.toml"), "--model-type", "qwen2", "--output", str(tmp_path / "out"),
                 "--hf-config", str(tmp_path / "jla" / "hf_config.json"), "--tokenizer-from", str(tmp_path / "jla")]) == 0
    assert (tmp_path / "out" / "tokenizer.json").read_text() == '{"fake": "tokenizer"}'
    if (tmp_path / "hf" / "generation_config.json").exists():
        assert (tmp_path / "out" / "generation_config.json").read_bytes() == (
            tmp_path / "hf" / "generation_config.json").read_bytes()
    original, exported = load_checkpoint_dir(tmp_path / "hf"), load_checkpoint_dir(tmp_path / "out")
    assert sorted(original) == sorted(exported)
    for key in original:
        assert np.array_equal(original[key], exported[key]), key
    assert main(["convert", "show-config", "--hf", str(tmp_path / "hf")]) == 0
    assert "[model.attention]" in capsys.readouterr().out
    # bfloat16 parameters: the stored model reloads into the trainer's template with load_pytree.
    assert main(["convert", "import", "--hf", str(tmp_path / "hf"), "--output", str(tmp_path / "bf16"),
                 "--param-dtype", "bfloat16"]) == 0
    from jla.convert import load_hf_checkpoint as load
    from jla.training.checkpointing import flatten_arrays, load_pytree

    template, _ = load(tmp_path / "hf", param_dtype="bfloat16")
    restored = load_pytree(tmp_path / "bf16" / "model.safetensors", template)
    for key, value in flatten_arrays(template).items():
        assert np.array_equal(np.asarray(flatten_arrays(restored)[key]), np.asarray(value)), key


def test_deepseek_v3_config_rejections():
    import dataclasses

    spec = dict(CASES["deepseek_v3"])
    config = from_hf_config(spec)
    assert config.resolved_mlp_types == ("dense", "moe", "moe") and config.moe.router_bias
    assert (config.moe.num_groups, config.moe.top_k_groups, config.moe.routed_scaling_factor) == (4, 2, 2.5)
    rejected = [
        {"rms_norm_eps": 1e-5},  # HF normalizes the MLA latents with 1e-6 regardless
        {"quantization_config": {"quant_method": "fp8"}},
        {"scoring_func": "softmax"},
        {"topk_method": "greedy"},
        {"attention_bias": True},
        {"rope_parameters": {"rope_type": "dynamic", "rope_theta": 10_000.0, "factor": 4.0}, "rope_scaling": None},
        {"rope_scaling": {"type": "longrope", "factor": 4.0}, "rope_parameters": None},
    ]
    for change in rejected:
        with pytest.raises(ValueError):
            from_hf_config({**spec, **change})
    with pytest.raises(ValueError):
        to_hf_config(config.replace(norm_eps=1e-5), "deepseek_v3")
    with pytest.raises(ValueError, match="shared-expert gate"):  # the export would drop mlp.shared_gate
        to_hf_config(config.replace(moe=dataclasses.replace(config.moe, shared_expert_gate=True)), "deepseek_v3")
    # The released DeepSeek-V3 RoPE (transformers 4.x spelling): YaRN frequencies, mscale == mscale_all_dim
    # (rotated channels unscaled) and the softmax correction (0.1 ln 40 + 1) ** 2.
    v3 = from_hf_config({**spec, "rope_parameters": None, "rope_theta": 10_000.0, "max_position_embeddings": 163840,
                         "rope_scaling": {"type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0,
                                          "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096}})
    assert v3.mla.rope_scaling == RoPEScalingConfig("yarn", 40.0, original_max_position_embeddings=4096)
    assert (v3.mla.yarn_factor, v3.mla.yarn_mscale) == (40.0, 1.0)
    assert np.isclose(v3.mla.scale, v3.mla.qk_head_dim ** -0.5 * (0.1 * np.log(40.0) + 1.0) ** 2)


def _hf_windows(spec: dict) -> list:
    """The window of every layer as transformers' own config object resolves it (family defaults included)."""
    spec = dict(spec)
    model_type = spec.pop("model_type")
    cfg = transformers.AutoConfig.for_model(model_type, **spec)
    window = getattr(cfg, "sliding_window", None)
    if model_type == "llama":
        return [None] * cfg.num_hidden_layers
    if model_type in ("qwen2", "qwen3"):
        return [window if t == "sliding_attention" else None for t in cfg.layer_types]
    return [window] * cfg.num_hidden_layers


def _our_windows(config) -> list:
    return [(config.local_attention if t == "local_attention" else config.attention).sliding_window
            for t in config.resolved_layer_types]


@pytest.mark.skipif(not V5, reason="the transformers 5 window rules")
def test_sliding_window_rules_match_transformers_configs():
    base = dict(SMALL, num_hidden_layers=30, **_rope(10_000.0))
    specs = [
        dict(model_type="qwen2", **base, use_sliding_window=True),  # defaults: window 4096 from layer 28 on
        dict(model_type="qwen3", **base, use_sliding_window=True, sliding_window=8, max_window_layers=5),
        dict(model_type="qwen3", **base, use_sliding_window=False, sliding_window=8, max_window_layers=5),
        dict(model_type="qwen3_moe", **base, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
             use_sliding_window=True, sliding_window=8, max_window_layers=5),  # every layer
        dict(model_type="qwen3_moe", **base, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
             use_sliding_window=True),  # default window 4096
        dict(model_type="mistral", **base),  # default window 4096
        dict(model_type="mistral", **base, sliding_window=None),
        dict(model_type="llama", **base, sliding_window=16),  # transformers' Llama ignores the key
    ]
    for spec in specs:
        assert _our_windows(from_hf_config(spec)) == _hf_windows(spec), spec
    with pytest.raises(ValueError, match="Llama"):
        from_hf_config(dict(model_type="llama", **base, use_sliding_window=True, sliding_window=16))


def test_export_rejects_what_transformers_cannot_express():
    import dataclasses

    llama = from_hf_config(CASES["llama"])
    with pytest.raises(ValueError, match="Llama"):
        to_hf_config(llama.replace(attention=dataclasses.replace(llama.attention, sliding_window=6)), "llama")
    moe = from_hf_config(CASES["qwen3_moe"])
    local = dataclasses.replace(moe.attention, sliding_window=6)
    mixed = moe.replace(layer_types=("attention", "local_attention", "local_attention"), local_attention=local)
    with pytest.raises(ValueError, match="qwen3_moe windows every layer"):
        to_hf_config(mixed, "qwen3_moe")
    assert to_hf_config(moe.replace(attention=local), "qwen3_moe")["sliding_window"] == 6  # every layer: fine
    mixtral = from_hf_config(CASES["mixtral"])
    for cfg, model_type in ((moe, "qwen3_moe"), (mixtral, "mixtral")):
        for change in (dict(router_bias=True), dict(num_groups=2, top_k_groups=1), dict(routed_scaling_factor=2.0)):
            with pytest.raises(ValueError, match="DeepSeek-V3 routing"):
                to_hf_config(cfg.replace(moe=dataclasses.replace(cfg.moe, **change)), model_type)
    phi3 = from_hf_config(CASES["phi3_partial_rotary"])
    scaled = dataclasses.replace(phi3.attention, rope_scaling=RoPEScalingConfig("linear", 2.0))
    with pytest.raises(ValueError, match="longrope"):
        to_hf_config(phi3.replace(attention=scaled), "phi3")
    yarn = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 64}
    with pytest.raises(ValueError, match="longrope"):
        from_hf_config({**CASES["phi3_partial_rotary"], **_rope(10_000.0, 0.5, scaling=yarn)})


@pytest.mark.skipif(not V5, reason="the transformers 5 RoPE parameter rules")
def test_original_max_position_embeddings_follows_transformers():
    import tempfile

    spec = dict(model_type="llama", **SMALL, **_rope(10_000.0, scaling=LLAMA3))
    cases = [spec, {**spec, "original_max_position_embeddings": 32},  # a top-level key: Llama's config ignores it
             {**spec, **_rope(10_000.0, scaling={k: v for k, v in LLAMA3.items() if k != "original_max_position_embeddings"})}]
    want = [64, 64, SMALL["max_position_embeddings"]]
    for case, value in zip(cases, want):
        ours = from_hf_config(case).attention.rope_scaling.original_max_position_embeddings
        assert transformers.AutoConfig.for_model(**case).rope_parameters["original_max_position_embeddings"] == value
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/config.json", "w") as stream:
                json.dump(case, stream)
            loaded = transformers.AutoConfig.from_pretrained(tmp)
        assert loaded.rope_parameters["original_max_position_embeddings"] == value, case
        assert ours == value, case
    # End to end with the top-level key present and with the default.
    for case in cases[1:]:
        model = _hf_model(case)
        ids = _ids(97)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            ours, config = load_hf_checkpoint(tmp, compute_dtype="float32")
            assert float(np.max(np.abs(_our_logits(ours, ids) - _hf_logits(model, ids)))) < TOL
            with open(f"{tmp}/config.json") as stream:
                exported = to_hf_config(config, "llama", base=json.load(stream))
        assert from_hf_config(exported, compute_dtype="float32") == config


def test_reexport_into_the_same_directory_leaves_no_stale_weights(tmp_path):
    model = _hf_model(CASES["qwen3"])
    model.save_pretrained(tmp_path / "hf")
    ours, _ = load_hf_checkpoint(tmp_path / "hf", compute_dtype="float32")
    out = tmp_path / "out"
    ids = _ids(97)
    for dtype, shard in (("float32", 40_000), ("bfloat16", 10**9), ("float32", 40_000)):
        save_hf_checkpoint(ours, out, "qwen3", dtype=dtype, max_shard_bytes=shard)
        files = sorted(p.name for p in out.glob("model*.safetensors*"))
        if shard == 10**9:
            assert files == ["model.safetensors"], files
        else:
            assert "model.safetensors" not in files and "model.safetensors.index.json" in files, files
            listed = set(json.loads((out / "model.safetensors.index.json").read_text())["weight_map"].values())
            assert set(files) - {"model.safetensors.index.json"} == listed, files
        mine = load_checkpoint_dir(out)
        assert {k: v.dtype for k, v in mine.items()} == {k: np.dtype(ml_dtypes.bfloat16 if dtype == "bfloat16"
                                                                     else np.float32) for k in mine}
        theirs = transformers.AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="eager")
        again, _ = load_hf_checkpoint(out, compute_dtype="float32")
        np.testing.assert_allclose(_hf_logits(theirs.eval(), ids), _our_logits(again, ids), atol=TOL, rtol=0)


def test_deepseek_v3_router_runs_in_float32_with_bfloat16_compute():
    """transformers computes DeepSeek-V3 router logits from float32 operands; so do we, whatever compute_dtype."""
    from jla.layers.moe import MoE
    from jla.models.config import MoEConfig

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((16, 64)), jnp.bfloat16)
    for router_bias in (True, False):
        cfg = MoEConfig(num_experts=8, top_k=2, d_ff_expert=32, router="sigmoid", router_bias=router_bias)
        layer = MoE(64, cfg, key=jax.random.key(0), std=0.3, compute_dtype="bfloat16")
        logits = np.asarray(layer._route(x).logits)
        exact = np.asarray(x, np.float64) @ np.asarray(layer.router, np.float64)
        bf16_router = np.asarray(jnp.asarray(layer.router, jnp.bfloat16), np.float64)
        if router_bias:
            assert np.max(np.abs(logits - exact)) < 1e-5
        else:  # other routers keep compute-dtype operands
            assert np.max(np.abs(logits - np.asarray(x, np.float64) @ bf16_router)) < 1e-4
            assert np.max(np.abs(logits - exact)) > 1e-4


@pytest.mark.parametrize("name", ["deepseek_v3_yarn", "mamba2"])
def test_cli_round_trip(name, tmp_path):
    """``jla convert import`` then ``export`` (through model.toml and model.safetensors) gives back every tensor."""
    from jla.cli import main
    from jla.convert.cli import _read_model_config

    _hf_model(CASES[name], seed=11).save_pretrained(tmp_path / "hf")
    assert main(["convert", "import", "--hf", str(tmp_path / "hf"), "--output", str(tmp_path / "jla")]) == 0
    assert _read_model_config(str(tmp_path / "jla" / "model.toml")) == from_hf_config(tmp_path / "hf")
    original = load_checkpoint_dir(tmp_path / "hf")
    fmt = "fused" if any(k.endswith("experts.gate_up_proj") for k in original) else "hub"
    assert main(["convert", "export", "--checkpoint", str(tmp_path / "jla"), "--model-config",
                 str(tmp_path / "jla" / "model.toml"), "--model-type", CASES[name]["model_type"], "--output",
                 str(tmp_path / "out"), "--hf-config", str(tmp_path / "jla" / "hf_config.json"),
                 "--moe-format", fmt]) == 0
    exported = load_checkpoint_dir(tmp_path / "out")
    assert sorted(exported) == sorted(original)
    for key, value in original.items():
        assert exported[key].dtype == value.dtype and np.array_equal(exported[key], value), key
