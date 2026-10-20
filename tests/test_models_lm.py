# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Language model: presets, causality, packed documents, scan = loop, remat, decoding, logical axes."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.models import (
    LOGICAL_AXIS_NAMES,
    AttentionConfig,
    LanguageModel,
    ModelConfig,
    build_model,
    get_preset,
    list_presets,
    logical_axes,
    parameter_count,
)
from jla.ops.cross_entropy import linear_cross_entropy

@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracles compare at ~1e-5: keep GPU runs off TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


TOL = dict(rtol=1e-4, atol=1e-4)
TINY = ["tiny", "tiny-hybrid", "tiny-moe", "tiny-mamba2", "tiny-mamba3", "tiny-gdn2"]

# Nominal size (millions): total parameters, or active parameters for MoE presets.
NOMINAL = {
    "transformer-125m": 125, "transformer-350m": 350, "transformer-1b": 1200, "gdn-hybrid-125m": 125,
    "kda-hybrid-125m": 125, "gla-125m": 125, "deltanet-125m": 125, "moe-125m": 125, "mamba2-130m": 130,
    "mamba3-130m": 130, "gdn2-125m": 125, "gdn2-hybrid-125m": 125,
    "fox-125m": 125,
    "loglinear-mamba2-130m": 130,
    "mamba3-mimo-130m": 130,
}
NOMINAL["xlstm-125m"] = 125
NOMINAL["moba-hybrid-125m"] = 125
NOMINAL["path-125m"] = 125  # PaTH attention
NOMINAL["mamba1-130m"] = 130
# RWKV-6 and HGRN2 (jla.layers.rwkv6).
TINY += ["tiny-rwkv6", "tiny-hgrn2"]
NOMINAL.update({"rwkv6-0.1b": 125, "hgrn2-125m": 125})
NOMINAL["gated-deltaproduct-125m"] = 125
NOMINAL["kimi-linear-1b"] = 250  # active parameters (~0.9B total)
NOMINAL.update({  # production architecture presets (active parameters for the MoE ones)
    "qwen3-next-300m": 300, "nemotron-h-600m": 600, "granite-4.0-h-125m": 125, "gpt-oss-180m": 180,
    "gemma-3-750m": 750,
})
NOMINAL["ttt-linear-125m"] = 125  # TTT-Linear (tests/test_ttt_layer.py covers tiny-ttt)
# RWKV-7 in LanguageModel and MiniMax-01 (tests/test_models_lm_extra.py has their layer tests).
TINY += ["tiny-rwkv7", "tiny-minimax-01"]
NOMINAL.update({"rwkv7-0.1b": 125, "minimax-01-1b": 190})  # minimax: active parameters (~1.04B total)


def _f32(name: str, **changes) -> ModelConfig:
    cfg = get_preset(name).replace(compute_dtype="float32", **changes)
    if cfg.moe is not None:  # capacity routing couples tokens; exact token-level tests use dropless routing
        cfg = cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None))
    return cfg


def _ids(batch=2, length=16, seed=1, vocab=512):
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, vocab)


def _loss(model: LanguageModel, ids, segment_ids=None):
    out = model(ids, segment_ids=segment_ids)
    targets = jnp.concatenate([ids[:, 1:], jnp.full((ids.shape[0], 1), -100)], axis=1)
    ce = linear_cross_entropy(out.hidden.reshape(-1, out.hidden.shape[-1]), model.lm_head_weight(),
                              targets.reshape(-1), compute_dtype=jnp.float32)
    mtp, _ = model.mtp_loss(out, ids, targets, segment_ids=segment_ids)
    return ce.loss_sum / ce.num_valid + out.aux_loss + mtp


@pytest.mark.parametrize("name", list_presets())
def test_every_preset_builds_with_a_sane_parameter_count(name):
    cfg = get_preset(name)
    shapes = jax.eval_shape(lambda: build_model(cfg, key=jax.random.key(0)))
    assert isinstance(shapes, LanguageModel)
    count = parameter_count(cfg)
    assert count.active <= count.total
    if name in NOMINAL:
        size = (count.active if cfg.moe is not None else count.total) / 1e6
        assert 0.8 * NOMINAL[name] <= size <= 1.2 * NOMINAL[name], (name, size)
        assert 32000 <= cfg.vocab_size <= 152064
    else:
        assert count.total < 1_000_000
    leaves = jax.tree.leaves(shapes)
    assert {x.dtype for x in leaves if jnp.issubdtype(x.dtype, jnp.inexact)} == {jnp.dtype(cfg.param_dtype)}
    # The only non-float leaves are int32 layer depths (lightning layers); they are not parameters.
    assert {x.dtype for x in leaves if not jnp.issubdtype(x.dtype, jnp.inexact)} <= {jnp.dtype(jnp.int32)}


@pytest.mark.parametrize("name", TINY)
def test_forward_shapes_and_dtypes(name):
    cfg = get_preset(name)
    model = build_model(cfg, key=jax.random.key(0))
    out = jax.jit(lambda m, i: m(i))(model, _ids())
    assert out.hidden.shape == out.pre_norm_hidden.shape == (2, 16, cfg.d_model)
    assert out.hidden.dtype == jnp.bfloat16
    assert out.aux_loss.shape == () and out.aux_loss.dtype == jnp.float32
    if cfg.moe is not None:
        assert set(out.metrics) == {"moe/balance_loss", "moe/router_z_loss", "moe/dropped_fraction"}
        assert float(out.aux_loss) > 0
    else:
        assert out.metrics == {} and float(out.aux_loss) == 0.0
    logits = model.logits(out.hidden)
    assert logits.shape == (2, 16, 512) and logits.dtype == jnp.float32
    assert np.isfinite(float(_loss(model, _ids())))


@pytest.mark.parametrize("name", TINY)
def test_future_tokens_never_change_the_past(name):
    model = _built(_f32(name), key=jax.random.key(0))
    ids = _ids()
    ids2 = ids.at[:, 9:].set(_ids(2, 7, seed=7))
    np.testing.assert_allclose(model(ids2).hidden[:, :9], model(ids).hidden[:, :9], **TOL)
    # Not vacuous: the past does reach later positions (an identity model would pass the check above).
    ids3 = ids.at[:, :9].set(_ids(2, 9, seed=8))
    assert not np.allclose(model(ids3).hidden[:, 9:], model(ids).hidden[:, 9:], **TOL)


@pytest.mark.parametrize("name", TINY)
def test_packed_documents_match_separate_runs(name):
    model = _built(_f32(name), key=jax.random.key(0))
    lengths = [7, 3, 6]
    seg = jnp.asarray(np.repeat(np.arange(3), lengths)[None], jnp.int32)
    pos = jnp.asarray(np.concatenate([np.arange(n) for n in lengths])[None], jnp.int32)
    ids = _ids(1, 16, seed=2)
    packed = model(ids, positions=pos, segment_ids=seg).hidden
    start = 0
    for n in lengths:
        alone = model(ids[:, start : start + n]).hidden
        np.testing.assert_allclose(packed[:, start : start + n], alone, **TOL)
        start += n


def _as_scanned_blocks(loop_blocks, period):
    return tuple(jax.tree.map(lambda *xs: jnp.stack(xs), *loop_blocks[j::period]) for j in range(period))


@pytest.mark.parametrize("name", ["tiny-hybrid", "tiny-moe", "tiny-mamba2", "tiny-mamba3", "tiny-gdn2"])
def test_scan_and_loop_match_outputs_and_gradients(name):
    scanned = build_model(_f32(name, scan_layers=True), key=jax.random.key(3))
    looped = build_model(_f32(name, scan_layers=False), key=jax.random.key(3))
    period = len(scanned.blocks)
    assert len(looped.blocks) == scanned.config.n_layers and period < scanned.config.n_layers
    _assert_close(scanned.blocks, _as_scanned_blocks(looped.blocks, period))
    ids = _ids()
    np.testing.assert_allclose(scanned(ids).hidden, looped(ids).hidden, **TOL)
    g_scan = eqx.filter_grad(_loss)(scanned, ids)
    g_loop = eqx.filter_grad(_loss)(looped, ids)
    _assert_close(g_scan.blocks, _as_scanned_blocks(g_loop.blocks, period), rtol=1e-4, atol=1e-6)
    _assert_close(g_scan.embed, g_loop.embed, rtol=1e-4, atol=1e-6)


def _assert_close(a, b, **tol):
    la = jax.tree.leaves(eqx.filter(a, eqx.is_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_array))
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_allclose(x, y, **(tol or TOL))


REMAT_CASES = [(p, s, "tiny-hybrid") for p in ("full", "dots", "dots_with_no_batch_dims", "save_attention")
               for s in (True, False)] + [(p, True, n) for p in ("full", "save_attention")
                                          for n in ("tiny-mamba2", "tiny-mamba3", "tiny-gdn2")]


@pytest.mark.parametrize(("policy", "scan", "name"), REMAT_CASES)
def test_remat_preserves_outputs_and_gradients(policy, scan, name):
    base = build_model(_f32(name, remat="none", scan_layers=scan), key=jax.random.key(4))
    remat = build_model(_f32(name, remat=policy, scan_layers=scan), key=jax.random.key(4))
    ids = _ids()
    np.testing.assert_allclose(remat(ids).hidden, base(ids).hidden, rtol=1e-6, atol=1e-6)
    g_base = eqx.filter_grad(_loss)(base, ids)
    g_remat = eqx.filter_grad(_loss)(remat, ids)
    _assert_close(g_remat, g_base, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", TINY + ["window"])
@pytest.mark.parametrize("scan", [True, False])
def test_prefill_then_decode_matches_the_forward_pass(name, scan):
    if name == "window":
        cfg = _f32("tiny-hybrid", scan_layers=scan)
        cfg = cfg.replace(attention=dataclasses.replace(cfg.attention, sliding_window=5))
    else:
        cfg = _f32(name, scan_layers=scan)
    model = _built(cfg, key=jax.random.key(5))
    ids = _ids(2, 14, seed=6)
    full = model.logits(model(ids).hidden)
    pos = jnp.broadcast_to(jnp.arange(14, dtype=jnp.int32), (2, 14))
    step = jax.jit(lambda m, c, i, p: m.decode_step(c, i, p))
    cache = model.init_cache(2, 20)
    logits, cache = step(model, cache, ids[:, :6], pos[:, :6])
    outs = [logits]
    for t in range(6, 14):
        logits, cache = step(model, cache, ids[:, t : t + 1], pos[:, t : t + 1])
        outs.append(logits)
    np.testing.assert_allclose(jnp.concatenate(outs, axis=1), full, rtol=1e-4, atol=2e-4)


def _axes_by_path(model):
    """``{path: (array shape, axes)}``, pairing parameters with their axes via tree prefix matching."""
    params = eqx.filter(model, lambda x: eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct))
    pairs = {}
    jax.tree_util.tree_map_with_path(
        lambda path, x, ax: pairs.__setitem__(jax.tree_util.keystr(path), (x.shape, ax)), params,
        logical_axes(model),
    )
    return pairs


@pytest.mark.parametrize("name", TINY)
@pytest.mark.parametrize("scan", [True, False])
@pytest.mark.parametrize("tie", [True, False])
def test_logical_axes_cover_every_parameter(name, scan, tie):
    model = build_model(get_preset(name).replace(scan_layers=scan, tie_embeddings=tie), key=jax.random.key(0))
    pairs = _axes_by_path(model)
    assert len(pairs) == len(jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    for path, (shape, ax) in pairs.items():
        assert isinstance(ax, tuple) and len(ax) == len(shape), path
        assert set(ax) <= set(LOGICAL_AXIS_NAMES) | {None}, path
        assert (ax[0] == "layers") == (scan and path.startswith(".blocks")), path
    assert pairs[".embed.weight"][1] == ("vocab", "embed")
    if not tie:
        assert pairs[".head"][1] == ("vocab", "embed")
    # The same axes come out of an abstract (eval_shape) model, before any parameter exists.
    abstract = jax.eval_shape(lambda: build_model(model.config, key=jax.random.key(0)))
    assert _axes_by_path(abstract) == pairs


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
def test_save_attention_keeps_the_cudnn_forward_out_of_the_backward_pass():
    ids = _ids(2, 64)

    def fwd_calls(policy):
        model = build_model(get_preset("tiny-hybrid").replace(remat=policy), key=jax.random.key(0))
        step = jax.jit(lambda m, i: eqx.filter_value_and_grad(_loss)(m, i))
        hlo = step.lower(model, ids).compile().as_text()
        return hlo.count('custom_call_target="__cudnn$fmhaSoftmax"')

    full, saved = fwd_calls("full"), fwd_calls("save_attention")
    assert saved >= 1 and full == 2 * saved  # "full" reruns every attention forward in the backward pass


def test_tied_and_untied_heads():
    tied = build_model(get_preset("tiny"), key=jax.random.key(0))
    assert tied.head is None and tied.lm_head_weight() is tied.embed.weight
    untied = build_model(get_preset("tiny").replace(tie_embeddings=False), key=jax.random.key(0))
    assert untied.lm_head_weight().shape == (512, 64)
    assert parameter_count(untied.config).total == parameter_count(tied.config).total + 512 * 64


def test_dropout():
    cfg = get_preset("tiny").replace(resid_dropout=0.1, embed_dropout=0.1)
    model = build_model(cfg, key=jax.random.key(0))
    ids = _ids()
    with pytest.raises(ValueError, match="dropout"):
        model(ids)
    a = model(ids, key=jax.random.key(1)).hidden
    b = model(ids, key=jax.random.key(2)).hidden
    assert not np.allclose(a.astype(jnp.float32), b.astype(jnp.float32))
    np.testing.assert_array_equal(model(ids, inference=True).hidden, model(ids, key=jax.random.key(1),
                                                                           inference=True).hidden)


def test_parameter_and_compute_dtypes():
    model = build_model(get_preset("tiny-hybrid").replace(param_dtype="bfloat16", compute_dtype="float32"),
                        key=jax.random.key(0))
    assert {x.dtype for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))} == {jnp.dtype(jnp.bfloat16)}
    assert model(_ids()).hidden.dtype == jnp.float32


def test_config_validation():
    attn = AttentionConfig(num_heads=4, num_kv_heads=2, head_dim=16)
    with pytest.raises(ValueError, match="layer types"):
        ModelConfig(vocab_size=8, d_model=8, n_layers=1, d_ff=8, attention=attn, layer_types=("mamba",))
    with pytest.raises(ValueError, match="remat"):
        ModelConfig(vocab_size=8, d_model=8, n_layers=1, d_ff=8, attention=attn, remat="sometimes")
    with pytest.raises(ValueError, match="linear_attention"):
        ModelConfig(vocab_size=8, d_model=8, n_layers=1, d_ff=8, attention=attn, layer_types=("kda",))
    with pytest.raises(KeyError, match="unknown model preset"):
        get_preset("gpt-5")


# 1.3B end-to-end presets (benchmarks/end_to_end_1p3b.md): exact totals, with the untied 128,256-token
# embedding and head (525M). NOMINAL is read when the tests run, so these entries can live at the end.
E2E_1P3B_TOTALS = {
    "transformer-1p3b": 1_607_567_360, "transformer-swa-1p3b": 1_607_567_360, "gdn-1p3b": 1_781_613_308,
    "gdn-hd128-1p3b": 1_861_392_128, "gdn2-1p3b": 2_073_777_536, "kda-1p3b": 1_785_206_144,
    "mamba2-1p3b": 1_766_104_064, "mamba3-1p3b": 1_781_387_264, "gdn-hybrid-1p3b": 1_747_458_368,
    "kda-hybrid-1p3b": 1_740_797_984,
}
NOMINAL.update({name: round(total / 1e6) for name, total in E2E_1P3B_TOTALS.items()})
NOMINAL["rwkv7-1p3b"] = 1760  # flame rwkv7-1p3b.json shape with the untied 128,256-token embedding and head


@pytest.mark.parametrize("name", sorted(E2E_1P3B_TOTALS))
def test_1p3b_preset_parameter_count_is_exact(name):
    assert parameter_count(get_preset(name)).total == E2E_1P3B_TOTALS[name]


# Real hybrid layouts at ~1.3B total (benchmarks/end_to_end_1p3b.md): active parameters, millions
# (exact counts in tests/test_hybrid_presets_1p3b.py).
NOMINAL.update({"kimi-linear-1p3b": 630, "qwen3-next-1p3b": 548})


def _built(cfg: ModelConfig, key) -> LanguageModel:
    """build_model, with every float parameter of an RWKV-7 model moved off its initial value. RWKV-7
    zero-initialises ``o_proj`` and the channel-mix ``value`` (as FLA and BlinkDL do), so each of its
    blocks is the identity at init and the token-mixing checks above would pass without running its
    mixer, caches or segment resets. Other models are returned as built."""
    model = build_model(cfg, key=key)
    if "rwkv7" not in cfg.resolved_layer_types:
        return model
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    leaves = [p + 0.05 * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)
