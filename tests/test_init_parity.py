# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Initialisation schemes and the Hugging Face MoE balance-loss coefficient.

``ModelConfig.init_scheme="upstream"`` (the default, and the matching layer options) follows the reference
implementations; ``init_scheme="jla"`` is jla's own scheme, whose parameters are pinned bit for bit:

* Mamba-2 / Mamba-3 (SISO, MIMO) ``out_proj``: ``U(+-1/sqrt(inner)) / sqrt(n_residuals)``
  (state-spaces/mamba ``mixer_seq_simple._init_weights``, FLA ``rescale_prenorm_residual``);
* delta-rule short convs (GatedDeltaNet, DeltaNet, GatedDeltaProduct, KDA): ``N(0, initializer_range)``
  (FLA ``_init_weights`` on every ``nn.Conv1d``); an explicit ``conv_init`` wins over the scheme, and the
  presets with attention and Gated DeltaNet or KDA layers set ``conv_init="identity"`` (benchmarks/init_parity.md);
  the all-linear ones keep FLA's convs, and so do the tiny test presets (model-level tests exercise the conv caches);
* RWKV-7: zero ``o_proj`` / channel-mix ``value`` / LoRA downs, orthogonal LoRA ups with gain 0.1.

The MoE ``aux_loss_coef = top_k * router_aux_loss_coef`` for HF coefficients holds in both schemes.
Every init change keeps the forward function: the same parameters give the same output.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.conv import ShortConv
from jla.layers.linear import Linear, truncated_normal
from jla.layers.linear_attention import UPSTREAM_NORMAL_CONV_KINDS, make_mixer
from jla.layers.mamba3_mimo import Mamba3MIMO
from jla.layers.moe import MoE
from jla.layers.rwkv7 import RWKV7ChannelMix, RWKV7Mixer, RWKV7TimeMix
from jla.layers.ssm import Mamba2, Mamba3
from jla.models import parameter_count
from jla.models.config import (
    LinearAttentionConfig,
    MoEConfig,
    SSMConfig,
    aux_loss_coef_from_hf,
    aux_loss_coef_to_hf,
)
from jla.models.lm import Block, LanguageModel, residual_branches
from jla.models.presets import get_preset, list_presets

KEY = jax.random.PRNGKey(0)


def _np(x):
    return np.asarray(jnp.asarray(x, jnp.float32))


# Linear ----------------------------------------------------------------------------------------
def test_linear_default_init_is_unchanged():
    lin = Linear(64, 48, key=KEY, axes=("embed", "heads"), std=0.02)
    np.testing.assert_array_equal(_np(lin.weight), _np(truncated_normal(KEY, (64, 48), 0.02)))


def test_linear_zeros_orthogonal_kaiming():
    zeros = Linear(32, 16, key=KEY, axes=("embed", None), std=0.02, init="zeros", use_bias=True)
    assert not np.any(_np(zeros.weight)) and not np.any(_np(zeros.bias))
    for shape, gain in (((16, 64), 0.1), ((64, 16), 0.1), ((32, 32), 1.0)):
        w = _np(Linear(*shape, key=KEY, axes=(None, None), std=0.02, init="orthogonal", gain=gain).weight)
        small = min(shape)
        gram = w @ w.T if shape[0] <= shape[1] else w.T @ w  # orthonormal rows or columns, whichever side is smaller
        np.testing.assert_allclose(gram, gain**2 * np.eye(small), atol=1e-5 * max(gain**2, 1e-2))
    in_f, gain = 1536, 1 / math.sqrt(24)
    w = _np(Linear(in_f, 768, key=KEY, axes=(None, None), std=0.02, init="kaiming_uniform", gain=gain).weight)
    bound = gain / math.sqrt(in_f)
    assert np.abs(w).max() <= bound
    assert abs(w.std() / (bound / math.sqrt(3)) - 1) < 0.01  # U(-b, b) has std b / sqrt(3)
    with pytest.raises(ValueError, match="unknown linear init"):
        Linear(4, 4, key=KEY, axes=(None, None), std=0.02, init="xavier")


# ShortConv ---------------------------------------------------------------------------------------
def test_short_conv_normal_init():
    conv = ShortConv(4096, 4, init="normal", key=KEY, std=0.02, use_bias=True)
    w = _np(conv.weight)
    assert abs(w.std() / 0.02 - 1) < 0.02 and abs(w.mean()) < 1e-3
    assert np.abs(w).max() > 3 * 0.02  # untruncated (FLA's nn.init.normal_)
    assert not np.any(_np(conv.bias))
    with pytest.raises(ValueError, match="needs a key and std"):
        ShortConv(8, 4, init="normal", key=KEY)


# Linear-attention short convs ---------------------------------------------------------------------
def _la_cfg(**kw):
    base = dict(num_k_heads=4, num_v_heads=4, head_k_dim=64, head_v_dim=64, conv_size=4, implementation="reference",
                conv_impl="xla", qk_l2norm_in_conv=False)
    base.update(kw)
    return LinearAttentionConfig(**base)


def _convs(mixer):
    return [c for c in (mixer.q_conv, mixer.k_conv, mixer.v_conv) if c is not None]


def _is_identity(conv):
    w = _np(conv.weight)
    return np.all(w[:, :-1] == 0) and np.all(w[:, -1] == 1)


@pytest.mark.parametrize("kind", ["gated_deltanet", "deltanet", "kda", "gated_deltaproduct"])
def test_delta_rule_convs_start_at_fla_normal(kind):
    extra = {"gate_low_rank_dim": 16} if kind == "kda" else {}
    if kind == "gated_deltaproduct":
        extra["num_householder"] = 2
    cfg = _la_cfg(conv_init="normal", **extra)  # what init_scheme="upstream" picks for these kinds
    mixer = make_mixer(kind, 256, cfg, key=KEY, std=0.02, norm_eps=1e-6)
    convs = _convs(mixer)
    assert len(convs) == 3
    for conv in convs:
        w = _np(conv.weight)
        assert conv.bias is None and abs(w.std() / 0.02 - 1) < 0.1 and not _is_identity(conv)
    # The default (conv_init None, a mixer on its own or init_scheme="jla") and conv_init="identity" keep the
    # identity filter and every other parameter as it was.
    old = make_mixer(kind, 256, _la_cfg(**extra), key=KEY, std=0.02, norm_eps=1e-6)
    assert all(_is_identity(c) for c in _convs(old))
    explicit = make_mixer(kind, 256, _la_cfg(conv_init="identity", **extra), key=KEY, std=0.02, norm_eps=1e-6)
    old_arrays, explicit_arrays = (jax.tree.leaves(eqx.filter(m, eqx.is_array)) for m in (old, explicit))
    for a, b in zip(old_arrays, explicit_arrays):
        np.testing.assert_array_equal(_np(a), _np(b))
    conv_ids = {id(c.weight) for c in convs} | {id(c.weight) for c in _convs(old)}
    new_leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(mixer, eqx.is_array))
    old_leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(old, eqx.is_array))
    assert [p for p, _ in new_leaves] == [p for p, _ in old_leaves]
    for (path, a), (_, b) in zip(new_leaves, old_leaves):
        if id(a) in conv_ids:
            continue
        np.testing.assert_array_equal(_np(a), _np(b), err_msg=jax.tree_util.keystr(path))
    # Same parameters, same output: the init option does not touch the forward function.
    swapped = eqx.tree_at(lambda m: [c.weight for c in _convs(m)], old, [c.weight for c in convs])
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 24, 256), jnp.float32)
    np.testing.assert_array_equal(_np(mixer(x)), _np(swapped(x)))


@pytest.mark.parametrize("kind", ["gla", "simple_gla", "gdn2"])
def test_other_kinds_keep_identity_convs(kind):
    extra = {"gate_low_rank_dim": 16} if kind == "gla" else {}
    mixer = make_mixer(kind, 256, _la_cfg(**extra), key=KEY, std=0.02, norm_eps=1e-6)
    assert all(_is_identity(c) for c in _convs(mixer))
    opted = make_mixer(kind, 256, _la_cfg(conv_init="normal", **extra), key=KEY, std=0.02, norm_eps=1e-6)
    assert not any(_is_identity(c) for c in _convs(opted))


def test_conv_init_validation():
    with pytest.raises(ValueError, match="conv_init"):
        _la_cfg(conv_init="uniform")


def _upstream(name, **changes):
    return dataclasses.replace(get_preset(name), init_scheme="upstream", **changes)


def _gdn_q_conv(cfg):
    model = LanguageModel(cfg, key=KEY)
    block = next(b for b in model.blocks if b.layer_type == "gated_deltanet")
    return _np(block.mixer.q_conv.weight)


def test_lm_gdn_preset_convs():
    unset = _upstream("tiny-hybrid")  # a test preset: conv_init None, the scheme decides
    assert unset.linear_attention.conv_init is None
    # conv_init="identity", as every preset with attention and gated_deltanet layers sets it
    base = dataclasses.replace(unset, linear_attention=dataclasses.replace(unset.linear_attention,
                                                                           conv_init="identity"))
    w = _gdn_q_conv(unset)
    # conv_init None: "upstream" (the default) draws N(0, initializer_range) for the stacked (scan) or
    # per-block weights, not the identity filter; "jla" keeps the identity filter
    assert abs(w.std() / unset.initializer_range - 1) < 0.2 and np.any(w[..., :-1] != 0)
    # an explicit conv_init wins over either scheme
    for cfg in (base, dataclasses.replace(base, init_scheme="jla"), dataclasses.replace(unset, init_scheme="jla")):
        w = _gdn_q_conv(cfg)
        assert np.all(w[..., :-1] == 0) and np.all(w[..., -1] == 1)
    for scheme in ("jla", "upstream"):
        w = _gdn_q_conv(dataclasses.replace(base, init_scheme=scheme, linear_attention=dataclasses.replace(
            base.linear_attention, conv_init="normal")))
        assert np.any(w[..., :-1] != 0)


# Per-family short-conv init of the presets (the 125M FineWeb-Edu A/B, benchmarks/init_parity.md): FLA's
# N(0, 0.02) convs helped DeltaNet and Gated DeltaProduct, tied for the Gated DeltaNet hybrid and lost for the
# KDA hybrid in 7 of 7 seeds, so every preset with attention and gated_deltanet or kda layers sets
# conv_init="identity". Without attention layers they helped Gated DeltaNet and KDA too (the all-linear 125M
# runs), so the all-linear presets leave conv_init unset (FLA's convs under the default scheme).
IDENTITY_CONV_PRESETS = frozenset({
    "gdn-hybrid-125m", "kda-hybrid-125m", "moba-hybrid-125m", "kimi-linear-1b",
    "qwen3-next-300m", "gdn-hybrid-1p3b",
    "kda-hybrid-1p3b", "kimi-linear-1p3b", "qwen3-next-1p3b",
})
NORMAL_CONV_PRESETS = frozenset({"tiny-deltaproduct", "deltanet-125m", "gated-deltaproduct-125m"})
ALL_LINEAR_FLA_CONV_PRESETS = frozenset({"gdn-1p3b", "gdn-hd128-1p3b", "kda-1p3b"})
# ... except the tiny test presets with those layers: they leave conv_init unset, so they draw FLA's N(0, 0.02)
# convs. With identity convs every older tap is zero at init, and the model-level tests that build these presets
# (decode vs forward, packed documents, scan vs loop) would pass with a broken conv cache or segment reset.
FLA_CONV_TEST_PRESETS = frozenset({"tiny-hybrid", "tiny-qwen3-next", "kimi-linear-tiny"})


def _presets_with(kinds):
    return {name for name in list_presets() if set(get_preset(name).resolved_layer_types) & set(kinds)}


def test_preset_conv_init_per_family():
    gdn_kda = _presets_with(("gated_deltanet", "kda"))
    assert gdn_kda == IDENTITY_CONV_PRESETS | FLA_CONV_TEST_PRESETS | ALL_LINEAR_FLA_CONV_PRESETS
    all_linear = {n for n in gdn_kda if set(get_preset(n).resolved_layer_types) <= {"gated_deltanet", "kda"}}
    assert all_linear == ALL_LINEAR_FLA_CONV_PRESETS  # every other one has attention (or MLA) layers
    assert _presets_with(("deltanet", "gated_deltaproduct")) == NORMAL_CONV_PRESETS
    for name in list_presets():
        cfg = get_preset(name)
        assert cfg.init_scheme == "upstream", name
        expected = "identity" if name in IDENTITY_CONV_PRESETS else None
        assert getattr(cfg.linear_attention, "conv_init", None) == expected, name


@pytest.mark.parametrize("name", sorted(IDENTITY_CONV_PRESETS | NORMAL_CONV_PRESETS | ALL_LINEAR_FLA_CONV_PRESETS))
def test_preset_convs_build_per_family(name):
    """The block LanguageModel builds for every delta-rule layer kind of the preset has the expected convs."""
    cfg = get_preset(name)
    kinds = sorted(set(cfg.resolved_layer_types) & set(UPSTREAM_NORMAL_CONV_KINDS))
    assert kinds
    for kind in kinds:
        convs = _convs(Block(cfg, kind, "none", key=KEY).mixer)
        assert convs, kind
        if name in IDENTITY_CONV_PRESETS:
            assert all(_is_identity(c) for c in convs), kind
        else:
            assert cfg.initializer_range == 0.02
            for conv in convs:
                w = _np(conv.weight)
                assert conv.bias is None and abs(w.std() / 0.02 - 1) < 0.1 and not _is_identity(conv), kind


def _abstract_leaves(cfg):
    model = jax.eval_shape(lambda: LanguageModel(cfg, key=KEY))
    return [(jax.tree_util.keystr(p), x.shape, x.dtype) for p, x in jax.tree_util.tree_leaves_with_path(model)
            if isinstance(x, jax.ShapeDtypeStruct)]


@pytest.mark.parametrize("name", sorted(IDENTITY_CONV_PRESETS))
def test_identity_conv_presets_keep_parameter_shapes(name):
    """Identity and N(0, 0.02) convs have the same parameters (no bias either way): same shapes and count."""
    cfg = get_preset(name)
    fla = cfg.replace(linear_attention=dataclasses.replace(cfg.linear_attention, conv_init=None))
    assert _abstract_leaves(cfg) == _abstract_leaves(fla)
    assert parameter_count(cfg) == parameter_count(fla)


def _is_qkv_conv_weight(path) -> bool:
    name = jax.tree_util.keystr(path)
    return any(f"{conv}.weight" in name for conv in ("q_conv", "k_conv", "v_conv"))


@pytest.mark.parametrize("scan_layers", [False, True])
@pytest.mark.parametrize("name", sorted(FLA_CONV_TEST_PRESETS))
def test_tiny_test_presets_keep_fla_conv_history_taps(name, scan_layers):
    """Guard for the model-level tests: every q/k/v short conv of the tiny Gated DeltaNet / KDA presets starts
    at N(0, 0.02) with nonzero older taps, and the model's output at init depends on those taps."""
    cfg = get_preset(name).replace(scan_layers=scan_layers, compute_dtype="float32")
    assert cfg.init_scheme == "upstream" and cfg.linear_attention.conv_init is None
    model = LanguageModel(cfg, key=KEY)
    params, static = eqx.partition(model, eqx.is_array)
    convs = [w for path, leaf in jax.tree_util.tree_leaves_with_path(params) if _is_qkv_conv_weight(path)
             for w in _np(leaf).reshape(-1, *leaf.shape[-2:])]  # stacked (scanned) layers split per layer
    assert len(convs) == 3 * sum(t in ("gated_deltanet", "kda") for t in cfg.resolved_layer_types)
    for w in convs:
        assert np.all(w[:, :-1] != 0) and 0.7 < w.std() / cfg.initializer_range < 1.3
    # The older taps are what a lost conv cache or a missed segment reset gets wrong: zeroing them moves the output
    # (with identity convs it would not move at all, and a cache test would pass with a broken cache).
    cut = eqx.combine(jax.tree_util.tree_map_with_path(
        lambda path, x: x.at[..., :-1].set(0) if _is_qkv_conv_weight(path) else x, params), static)
    ids = jax.random.randint(jax.random.PRNGKey(3), (2, 24), 0, cfg.vocab_size)
    assert np.abs(_np(model(ids).hidden) - _np(cut(ids).hidden)).max() > 1e-3


def test_init_scheme_validation():
    with pytest.raises(ValueError, match="init_scheme"):
        dataclasses.replace(get_preset("tiny"), init_scheme="fla")
    assert get_preset("tiny").init_scheme == "upstream"


# Mamba out_proj ----------------------------------------------------------------------------------
def _ssm(**kw):
    return SSMConfig(d_state=16, head_dim=16, chunk_size=16, implementation="reference", conv_impl="xla", **kw)


@pytest.mark.parametrize("cls,kw", [(Mamba2, {}), (Mamba3, {}), (Mamba3MIMO, {"mimo_rank": 2})])
@pytest.mark.parametrize("n_residuals", [1, 48])
def test_mamba_out_proj_kaiming_rescaled(cls, kw, n_residuals):
    layer = cls(256, _ssm(**kw), key=KEY, std=0.02, norm_eps=1e-5, n_residuals=n_residuals)
    w = _np(layer.out_proj.weight)
    inner = w.shape[0]
    bound = 1 / math.sqrt(inner) / math.sqrt(n_residuals)
    assert np.abs(w).max() <= bound * (1 + 1e-6)
    assert abs(w.std() / (bound / math.sqrt(3)) - 1) < 0.03
    # the other projections keep their truncated normal
    zx = _np(layer.zx_proj.weight)
    assert np.abs(zx).max() <= 2 * 0.02 * (1 + 1e-6) and abs(zx.std() / (0.02 * 0.8796) - 1) < 0.03
    with pytest.raises(ValueError, match="n_residuals"):
        cls(256, _ssm(**kw), key=KEY, std=0.02, norm_eps=1e-5, n_residuals=0)
    # n_residuals None (the default; init_scheme "jla") keeps the truncated normal out_proj
    w = _np(cls(256, _ssm(**kw), key=KEY, std=0.02, norm_eps=1e-5).out_proj.weight)
    assert np.abs(w).max() <= 2 * 0.02 * (1 + 1e-6) and abs(w.std() / (0.02 * 0.8796) - 1) < 0.03


@pytest.mark.parametrize("name,per_layer", [("mamba2-130m", 1), ("mamba3-130m", 1), ("tiny-mamba3-mimo", None),
                                            ("tiny-jamba", None)])
def test_lm_mamba_residual_count(name, per_layer):
    cfg = get_preset(name)
    count = residual_branches(cfg)
    assert count == sum(1 + (m != "none") for m in cfg.resolved_mlp_types)
    if per_layer is not None:
        assert count == per_layer * cfg.n_layers


def test_lm_mamba2_out_proj_scale():
    cfg = _upstream("tiny-mamba2")
    model = LanguageModel(cfg, key=KEY)
    n = residual_branches(cfg)
    for block in model.blocks:
        if block.layer_type != "mamba2":
            continue
        w = _np(block.mixer.out_proj.weight)
        inner = w.shape[-2]
        assert np.abs(w).max() <= (1 + 1e-6) / math.sqrt(inner * n)


def test_mamba_forward_same_params_same_output():
    old_w = truncated_normal(jax.random.PRNGKey(3), (256 * 2, 256), 0.02)
    for cls, kw in ((Mamba2, {}), (Mamba3, {}), (Mamba3MIMO, {"mimo_rank": 2})):
        a = cls(256, _ssm(**kw), key=KEY, std=0.02, norm_eps=1e-5, n_residuals=1)
        b = cls(256, _ssm(**kw), key=KEY, std=0.02, norm_eps=1e-5, n_residuals=24)
        a = eqx.tree_at(lambda m: m.out_proj.weight, a, old_w)
        b = eqx.tree_at(lambda m: m.out_proj.weight, b, old_w)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 256), jnp.float32)
        np.testing.assert_array_equal(_np(a(x)), _np(b(x)))


# RWKV-7 ------------------------------------------------------------------------------------------
def _check_lora(lora, d_out):
    assert not np.any(_np(lora.down.weight))
    up = _np(lora.up.weight)
    rank = up.shape[0]
    gain = 0.1 * (math.sqrt(rank / d_out) if rank > d_out else 1.0)
    gram = up @ up.T if rank <= d_out else up.T @ up
    np.testing.assert_allclose(gram, gain**2 * np.eye(min(rank, d_out)), atol=1e-6)


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_rwkv7_time_mix_upstream_init(layer_idx):
    d = 256
    tm = RWKV7TimeMix(d, head_dim=64, layer_idx=layer_idx, n_layers=4, key=KEY, impl="reference",
                      init_scheme="upstream")
    assert not np.any(_np(tm.o_proj.weight))
    for lora in (tm.w_lora, tm.a_lora, tm.g_lora) + ((tm.v_lora,) if tm.v_lora is not None else ()):
        _check_lora(lora, d)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, d), jnp.float32)
    # zero down projections: the LoRAs output exactly their bias (w0, a0, v0)
    bias = tm.w_lora.up.bias.astype(tm.w_lora.up.compute_dtype)  # the LoRA returns the compute dtype
    np.testing.assert_array_equal(_np(tm.w_lora(x)), np.broadcast_to(_np(bias), (2, 16, d)))
    v_first = None if layer_idx == 0 else x
    out = tm(x, v_first)[0]
    assert not np.any(_np(out))  # zero o_proj: the block is the identity on the residual stream
    cm = RWKV7ChannelMix(d, 4 * d, layer_idx=layer_idx, n_layers=4, key=KEY, init_scheme="upstream")
    assert not np.any(_np(cm.value.weight)) and not np.any(_np(cm(x)[0]))
    # the default ("jla") draws o_proj, value and the LoRAs as truncated normals
    jla_tm = RWKV7TimeMix(d, head_dim=64, layer_idx=layer_idx, n_layers=4, key=KEY, impl="reference")
    jla_cm = RWKV7ChannelMix(d, 4 * d, layer_idx=layer_idx, n_layers=4, key=KEY)
    for w in (jla_tm.o_proj.weight, jla_tm.w_lora.down.weight, jla_tm.w_lora.up.weight, jla_cm.value.weight):
        assert np.all(np.isfinite(_np(w))) and np.count_nonzero(_np(w)) > 0.99 * np.size(w)
    # r / k / v and the channel-mix key keep their truncated normals
    assert abs(_np(tm.k_proj.weight).std() / (0.1 * d**-0.5 * 0.8796) - 1) < 0.05


def test_rwkv7_lm_mixer_upstream_init():
    cfg = get_preset("tiny-rwkv7")
    la = cfg.linear_attention
    mixer = RWKV7Mixer(cfg.d_model, la, layer_idx=0, n_layers=cfg.n_layers, key=KEY, norm_eps=1e-5,
                       init_scheme="upstream")
    assert not np.any(_np(mixer.o_proj.weight))
    for lora in (mixer.w_lora, mixer.a_lora, mixer.v_lora, mixer.g_lora):
        _check_lora(lora, cfg.d_model)


def test_rwkv7_lm_init_scheme():
    """BlinkDL's layout (RWKV7LM) takes the same scheme from RWKV7Config.init_scheme."""
    from jla.models.rwkv7 import RWKV7LM, RWKV7Config

    cfg = RWKV7Config(vocab_size=64, d_model=128, n_layers=2, head_dim=64)
    upstream = RWKV7LM(cfg, key=KEY)  # the default
    jla = RWKV7LM(dataclasses.replace(cfg, init_scheme="jla"), key=KEY)
    for block in upstream.blocks:
        assert not np.any(_np(block.att.o_proj.weight)) and not np.any(_np(block.ffn.value.weight))
    for block in jla.blocks:
        assert np.any(_np(block.att.o_proj.weight)) and np.any(_np(block.ffn.value.weight))
    with pytest.raises(ValueError, match="init_scheme"):
        dataclasses.replace(cfg, init_scheme="fla")


def _fixed_params(module, seed):
    """Every float leaf replaced by N(0, 0.1) from a fixed key: the output no longer depends on the init."""
    params, static = eqx.partition(module, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [0.1 * jax.random.normal(k, p.shape, jnp.float32).astype(p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def test_rwkv7_forward_same_params_same_output():
    """Same parameters, same output as before the init change: the expected values below were computed
    by the code before it (CPU, float32; bit-identical to this code there) for parameters that do not
    depend on the init."""
    d = 128
    with jax.default_matmul_precision("highest"):
        tm = RWKV7TimeMix(d, head_dim=64, layer_idx=1, n_layers=4, key=jax.random.PRNGKey(0), impl="reference",
                          compute_dtype="float32")
        cm = RWKV7ChannelMix(d, 4 * d, layer_idx=1, n_layers=4, key=jax.random.PRNGKey(0), compute_dtype="float32")
        tm, cm = _fixed_params(tm, 21), _fixed_params(cm, 22)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, d), jnp.float32)
        v_first = jax.random.normal(jax.random.PRNGKey(2), (2, 16, d), jnp.float32)
        out_tm = np.asarray(tm(x, v_first)[0], np.float64)
        out_cm = np.asarray(cm(x)[0], np.float64)
    np.testing.assert_allclose(out_tm[1, 15, :6], [-0.6847943067550659, -1.1840784549713135, -0.4516778886318207,
                                                   0.35986098647117615, 0.8821699023246765, -0.2931678295135498],
                               rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose((out_tm**2).sum(), 1160.1242799971155, rtol=1e-5)
    np.testing.assert_allclose(out_cm[1, 15, :6], [-1.9236973524093628, -3.8853354454040527, -2.3918604850769043,
                                                   -3.930138349533081, 2.453497886657715, -3.3936095237731934],
                               rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose((out_cm**2).sum(), 58946.36722327351, rtol=1e-5)


# MoE balance-loss coefficient ---------------------------------------------------------------------
def test_aux_loss_coef_hf_mapping():
    assert aux_loss_coef_from_hf(1e-3, 10) == pytest.approx(1e-2, rel=1e-15)
    assert aux_loss_coef_from_hf(1e-3, 2) == 2e-3
    assert aux_loss_coef_to_hf(2e-3, 2) == 1e-3
    for c, k in ((1e-3, 10), (0.02, 8), (0.001, 6)):
        assert aux_loss_coef_to_hf(aux_loss_coef_from_hf(c, k), k) == pytest.approx(c, rel=1e-15)


def test_moe_balance_loss_is_hf_over_top_k():
    """Our balance loss times top_k equals HF's load_balancing_loss_func on the same routing."""
    cfg = MoEConfig(num_experts=8, top_k=3, d_ff_expert=16, capacity_factor=None, aux_loss_coef=1.0)
    moe = MoE(32, cfg, key=KEY, std=0.02)
    x = jax.random.normal(jax.random.PRNGKey(2), (64, 32), jnp.float32)
    _, stats = moe(x[None])
    routing = moe._route(x)
    # transformers load_balancing_loss_func: one_hot [N, k, E] mean over tokens, times the mean probability
    expert_mask = jax.nn.one_hot(routing.experts, cfg.num_experts)
    tokens_per_expert = jnp.mean(expert_mask, axis=0)  # [k, E]
    router_prob = jnp.mean(jax.nn.softmax(routing.logits, axis=-1), axis=0)
    hf = cfg.num_experts * jnp.sum(tokens_per_expert * router_prob[None])
    np.testing.assert_allclose(float(stats.balance_loss) * cfg.top_k, float(hf), rtol=1e-5)


@pytest.mark.parametrize("name,expected", [("tiny-qwen3-next", 2e-3), ("qwen3-next-300m", 1e-2),
                                           ("tiny-minimax-01", 2e-3), ("minimax-01-1b", 2e-3)])
def test_hf_coefficient_presets(name, expected):
    moe = get_preset(name).moe
    assert moe.aux_loss_coef == pytest.approx(expected, rel=1e-15)
    assert aux_loss_coef_to_hf(moe.aux_loss_coef, moe.top_k) == pytest.approx(1e-3, rel=1e-15)


def test_hf_config_maps_router_aux_loss_coef():
    from jla.convert.hf_config import from_hf_config, to_hf_config

    base = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=512,
                intermediate_size=128, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=256)
    mixtral = dict(base, model_type="mixtral", num_local_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.02)
    cfg = from_hf_config(mixtral)
    assert cfg.moe.aux_loss_coef == pytest.approx(0.04, rel=1e-15)
    assert to_hf_config(cfg, "mixtral")["router_aux_loss_coef"] == pytest.approx(0.02, rel=1e-15)
    qwen = dict(base, model_type="qwen3_moe", num_experts=8, num_experts_per_tok=6, moe_intermediate_size=32,
                norm_topk_prob=True, router_aux_loss_coef=0.001, head_dim=16)
    cfg = from_hf_config(qwen)
    assert cfg.moe.aux_loss_coef == pytest.approx(0.006, rel=1e-15)
    assert to_hf_config(cfg, "qwen3_moe")["router_aux_loss_coef"] == pytest.approx(0.001, rel=1e-15)
    # DeepSeek-V3's aux_loss_alpha has our normalization: unchanged
    assert "aux_loss_alpha" not in mixtral


# init_scheme="jla" is pinned bit for bit ------------------------------------------------------------------
# sha256 over the per-leaf sha256 (dtype, shape, bytes) of LanguageModel(get_preset(name), key=PRNGKey(0)) under
# init_scheme="jla" (JAX 0.11.2, XLA CPU): every preset whose initialisation the upstream scheme changes.
JLA_SCHEME_PARAM_DIGESTS = {
    "kimi-linear-tiny": "27444df2620c5ba761cfed19e9ea155d1ebc877e26fbe62f393e345a331332c1",
    "tiny-deltaproduct": "771b1153627c4eb70e2e2c04064a82a94b8eb0a053bfca9838e9517300c828e0",
    "tiny-gdn2": "81347eefd3f96fc8f4771cbec6038313ebfd02d11017f9785a20b58f13e9a28c",
    "tiny-granite-4.0-h": "9d7d09776abcc9402a17c96c772a8004740c11a09946446300095d9806d7425e",
    "tiny-hybrid": "6573aed9bf3ea0e6a80bbce1aba236c2b54ccd905da464d7cc6d00a5d0eb9d48",
    "tiny-jamba": "9a5cd908dad86a99cfd868600faa81d9b370571568418eaeec64dc037e9dfa72",
    "tiny-loglinear-mamba2": "3054f6eba8fa8154cc3c682e20c86f86d09f03574b960152182da711f863692b",
    "tiny-mamba2": "262c2f6d174a8fc32c01db9e844edaaed5453b6736410d87adedecfffb6296ca",
    "tiny-mamba3": "bd2efe1b10df69e553ea815380f822c9b6f9af8a1dd1ee8808022fe4aed6dba9",
    "tiny-mamba3-mimo": "00a7121009229b82b8ed96775ad0a22f1d7e908c70727e72067f2068da0643d5",
    "tiny-nemotron-h": "36edab9b9d2abd8a30c0805ad8c4824c40d35c4c583269b4a07ea7f11aeead2c",
    "tiny-qwen3-next": "e2419ffcda024d8105f05a3a73889062c5401c865918b74d3536c7dc51f5e861",
    "tiny-rwkv7": "c1191c22041b5090f7c986d066e638f31cea7767d8dc286575c57f94aa5a966a",
}


def _param_digest(model) -> str:
    total = hashlib.sha256()
    for _, leaf in jax.tree_util.tree_leaves_with_path(eqx.filter(model, eqx.is_array)):
        arr = np.asarray(leaf)
        total.update(hashlib.sha256(str(arr.dtype).encode() + str(arr.shape).encode() + arr.tobytes())
                     .hexdigest().encode())
    return total.hexdigest()


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="the digests were taken with XLA CPU's samplers"
)
@pytest.mark.parametrize("name", sorted(JLA_SCHEME_PARAM_DIGESTS))
def test_jla_scheme_parameters_are_pinned(name):
    cfg = dataclasses.replace(get_preset(name), init_scheme="jla")
    assert _param_digest(LanguageModel(cfg, key=jax.random.PRNGKey(0))) == JLA_SCHEME_PARAM_DIGESTS[name]


@pytest.mark.parametrize("name", ["tiny-hybrid", "tiny-qwen3-next", "kimi-linear-tiny"])
def test_upstream_scheme_keeps_identity_conv_presets(name):
    """With identity convs set (as every preset with attention and these layers sets them), Gated DeltaNet / KDA
    presets (no Mamba or RWKV-7 layers) get the same parameters under both schemes."""
    cfg = get_preset(name)
    cfg = dataclasses.replace(cfg, linear_attention=dataclasses.replace(cfg.linear_attention, conv_init="identity"))
    jla = LanguageModel(dataclasses.replace(cfg, init_scheme="jla"), key=KEY)
    up = LanguageModel(cfg, key=KEY)
    assert _param_digest(jla) == _param_digest(up)
    assert _param_digest(LanguageModel(get_preset(name), key=KEY)) != _param_digest(up)  # the test preset: FLA convs


@pytest.mark.parametrize("name", ["tiny-deltaproduct", "tiny-mamba2", "tiny-rwkv7"])
def test_upstream_scheme_changes_only_the_upstream_leaves(name):
    """``init_scheme="upstream"`` redraws only out_proj / short convs / RWKV-7 o_proj, value and LoRAs."""
    jla = jax.tree_util.tree_leaves_with_path(eqx.filter(
        LanguageModel(dataclasses.replace(get_preset(name), init_scheme="jla"), key=KEY), eqx.is_array))
    up = jax.tree_util.tree_leaves_with_path(eqx.filter(LanguageModel(get_preset(name), key=KEY), eqx.is_array))
    assert [p for p, _ in jla] == [p for p, _ in up]
    changed = {jax.tree_util.keystr(p) for (p, a), (_, b) in zip(jla, up)
               if not np.array_equal(_np(a), _np(b))}
    assert changed
    allowed = ("q_conv", "k_conv", "v_conv", "out_proj", "o_proj", "value", "_lora")
    assert all(any(s in path for s in allowed) for path in changed), sorted(changed)
