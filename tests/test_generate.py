# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generation: cached greedy decoding equals the full forward pass's argmax step by step, for every
layer family (RWKV-7 included); chunked prefill; padding sides and batching do not change any row;
sampling filters and seeds."""

from __future__ import annotations

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.inference import (
    GREEDY,
    SamplingConfig,
    generate,
    prompts_from_padded,
    sample_token,
    supports_cache,
    warp_logits,
)
from jla.inference.rwkv7 import RWKV7Decoder, decoding_model
from jla.models import build_model, get_preset
from jla.models.rwkv7 import RWKV7_PRESETS, RWKV7LM


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _f32(name: str, **changes):
    cfg = get_preset(name).replace(compute_dtype="float32", **changes)
    if cfg.moe is not None:  # capacity routing couples tokens; token-exact tests use dropless routing
        cfg = cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None))
    return cfg


def _model(name: str, seed: int = 0, *, moved: bool = True, **changes):
    """``moved``: an RWKV-7 model's weights moved off their init (see ``_rwkv7_moved``)."""
    if name in RWKV7_PRESETS:  # a separate model class, wrapped for the LanguageModel interface
        cfg = dataclasses.replace(RWKV7_PRESETS[name], compute_dtype="float32", **changes)
        lm = RWKV7LM(cfg, key=jax.random.key(seed))
        return decoding_model(_rwkv7_moved(lm) if moved else lm)
    model = build_model(_f32(name, **changes), key=jax.random.key(seed))
    return _rwkv7_moved(model) if moved and "rwkv7" in model.config.resolved_layer_types else model


def _default_model(name: str):
    """The preset in its default compute dtype (bfloat16) with its default MoE routing."""
    if name in RWKV7_PRESETS:
        return RWKV7LM(RWKV7_PRESETS[name], key=jax.random.key(0))
    return build_model(get_preset(name), key=jax.random.key(0))


def _prompts(vocab: int, lengths=(5, 9, 3), seed: int = 1) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab, n).tolist() for n in lengths]


def _check_against_forward(model, prompts, result, gap: float = 1e-4) -> None:
    """Every generated token is the argmax of the full forward pass over prompt + generated prefix."""
    for b, prompt in enumerate(prompts):
        new = result.row(b)
        seq = jnp.asarray([prompt + new], jnp.int32)
        logits = np.asarray(model.logits(model(seq, inference=True).hidden))[0]
        for i, token in enumerate(new):
            row = logits[len(prompt) + i - 1]
            best = int(row.argmax())
            assert token == best or row[best] - row[token] < gap, (b, i, token, best, row[best] - row[token])


# Every layer family with a decode cache: GQA attention, sliding windows (ring caches), sinks,
# local/global, MoE, MLA, FoX, MoBA, PaTH, the linear-attention kinds, Mamba-1/2/3 (SISO, MIMO), hybrids,
# RWKV-7 as a LanguageModel (tiny-rwkv7) and as the standalone RWKV7LM (tiny-rwkv7-blinkdl).
CACHED = [
    "tiny", "tiny-hybrid", "tiny-moe", "tiny-mamba2", "tiny-mamba3", "tiny-gdn2", "tiny-mamba1", "tiny-mla",
    "tiny-fox", "tiny-rwkv6", "tiny-hgrn2", "tiny-mlstm", "tiny-deltaproduct", "tiny-mamba3-mimo", "tiny-ttt",
    "tiny-moba", "tiny-jamba", "kimi-linear-tiny", "tiny-qwen3-next", "tiny-nemotron-h",
    "tiny-granite-4.0-h", "tiny-gpt-oss", "tiny-gemma-3", "tiny-rwkv7", "tiny-rwkv7-blinkdl", "tiny-path",
]

# The release families for the heavier sweeps: GQA attention, GDN (+ attention), Mamba-2, Mamba-3
# (SISO, MIMO), GDN-2 with sliding-window ring caches, KDA (Kimi Linear), gated attention + GDN
# (Qwen3-Next), sliding windows with sinks (GPT-OSS) and local/global windows (Gemma 3), RWKV-7 (both models).
RELEASE = [
    "tiny", "tiny-hybrid", "tiny-mamba2", "tiny-mamba3", "tiny-mamba3-mimo", "tiny-gdn2", "kimi-linear-tiny",
    "tiny-qwen3-next", "tiny-gpt-oss", "tiny-gemma-3", "tiny-rwkv7", "tiny-rwkv7-blinkdl",
]


@pytest.mark.parametrize("name", CACHED)
def test_cached_greedy_decoding_is_the_forward_argmax(name):
    model = _model(name)
    assert supports_cache(model)
    prompts = _prompts(model.config.vocab_size)
    result = generate(model, prompts, max_new_tokens=6, bucket=1)
    assert result.tokens.shape == (3, 6) and result.lengths.tolist() == [6, 6, 6]
    _check_against_forward(model, prompts, result)
    # The full-forward recompute loop (no caches) picks the same tokens.
    reference = generate(model, prompts, max_new_tokens=6, bucket=1, use_cache=False)
    np.testing.assert_array_equal(result.tokens, reference.tokens)


@pytest.mark.parametrize("name", RELEASE)
def test_long_prefill_is_the_forward_argmax(name):
    """A 96-token prefill (bucket 16) in one decode_step and in chunks of 24 (3 full chunks, then 23
    tokens): longer than conv widths, kernel chunks and sliding windows."""
    model = _model(name)
    prompts = _prompts(model.config.vocab_size, lengths=(150, 97, 131), seed=3)
    assert module_generate().default_prefill_chunk(3, 164) >= 95  # the default is one chunk here
    for chunk in (None, 24):
        result = generate(model, prompts, max_new_tokens=4, bucket=16, prefill_chunk=chunk)
        assert result.lengths.tolist() == [4, 4, 4]
        _check_against_forward(model, prompts, result)


def module_generate():
    import importlib

    return importlib.import_module("jla.inference.generate")  # the package re-exports the function `generate`


def test_default_prefill_chunk():
    chunk = module_generate().default_prefill_chunk
    assert chunk(1, 100) == 1024 and chunk(8, 2176) == 512 and chunk(1, 32768) == 512 and chunk(8, 32768) == 64
    assert chunk(64, 1 << 20) == 16  # never below 16
    with pytest.raises(ValueError, match="prefill_chunk"):
        generate(_model("tiny"), [[1, 2]], max_new_tokens=1, prefill_chunk=0)


def test_chunked_prefill_bounds_attention_memory():
    """Compile-only: with one prefill chunk, cached attention holds [B, H, T, T] float32 scores; the
    default chunk keeps them at [B, H, chunk, T] (a 16384-token prompt on the tiny transformer)."""
    module = module_generate()
    model = _model("tiny")
    batch, width, new = 1, 16384, 4
    total = width + new
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(0), i))(jnp.arange(batch, dtype=jnp.uint32))
    args = (model, jnp.zeros((batch, width), jnp.int32), jnp.full((batch,), width, jnp.int32), keys,
            jnp.zeros((0,), jnp.int32))

    def temp_bytes(chunk: int) -> int:
        compiled = module._cached_loop.lower(*args, prefill=width, total=total, max_new=new, sampling=GREEDY,
                                             pad_id=0, chunk=chunk).compile()
        stats = getattr(compiled, "compiled", compiled).memory_analysis()  # equinox wraps jax's Compiled
        if stats is None:
            pytest.skip("this backend reports no memory analysis")
        return int(stats.temp_size_in_bytes)

    default = module.default_prefill_chunk(batch, total)
    assert default == 512
    scores = batch * model.config.attention.num_heads * (width - 1) * total * 4  # one chunk's f32 scores
    chunked = temp_bytes(default)
    assert chunked * 8 < scores, (chunked, scores)
    try:
        one = temp_bytes(width - 1)
    except jax.errors.JaxRuntimeError as exc:  # GPU: the autotuner cannot even allocate the quadratic buffer
        assert "RESOURCE_EXHAUSTED" in str(exc)
    else:
        assert one >= scores, (one, scores)  # the quadratic buffer exists without chunking


NO_CACHE = ["tiny-path", "tiny-nsa", "tiny-dsa", "tiny-loglinear-mamba2"]


@pytest.mark.parametrize("name", RELEASE)
def test_default_dtype_greedy_and_sampled_generation(name):
    """The release presets in their default compute dtype (bfloat16) with their default MoE routing."""
    model = _default_model(name)
    config = model.config
    prompts = _prompts(config.vocab_size, lengths=(20, 37), seed=4)
    cfg = SamplingConfig(temperature=0.8, top_k=20, top_p=0.9)
    greedy = generate(model, prompts, max_new_tokens=4)
    sampled = generate(model, prompts, max_new_tokens=4, sampling=cfg, seed=1)
    for out in (greedy, sampled):
        assert out.lengths.tolist() == [4, 4]
        assert ((out.tokens >= 0) & (out.tokens < config.vocab_size)).all()
    np.testing.assert_array_equal(generate(model, prompts, max_new_tokens=4, sampling=cfg, seed=1).tokens,
                                  sampled.tokens)


def test_moe_capacity_routing_generates_dropless():
    """The full-forward loop and the cached loop both route dropless (decode_step always does)."""
    from jla.inference import dropless_moe

    config = get_preset("tiny-moe").replace(compute_dtype="float32")
    config = config.replace(moe=dataclasses.replace(config.moe, capacity_factor=1.0))
    model = build_model(config, key=jax.random.key(0))
    converted = dropless_moe(model)
    assert converted.config.moe.capacity_factor is None and dropless_moe(converted) is converted
    assert all(a is b for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(converted)))
    prompts = _prompts(config.vocab_size, lengths=(9, 5, 12))
    cached = generate(model, prompts, max_new_tokens=6, bucket=1)
    np.testing.assert_array_equal(cached.tokens, generate(model, prompts, max_new_tokens=6, bucket=1,
                                                          use_cache=False).tokens)
    _check_against_forward(converted, prompts, cached)


def test_unexpected_decode_errors_are_not_mistaken_for_missing_caches(monkeypatch):
    """Only missing decode support (NotImplementedError, or log-linear Mamba-2's ValueError) selects the
    full-forward loop; any other error raised while tracing a decode step propagates."""
    import importlib

    module = importlib.import_module("jla.inference.generate")  # the package re-exports the function `generate`
    model = _model("tiny-mamba2")

    def broken(self, cache, input_ids, positions):
        raise ValueError("a configuration error")

    monkeypatch.setattr(module, "_SUPPORTS_CACHE", {})
    monkeypatch.setattr(type(model), "decode_step", broken)
    with pytest.raises(ValueError, match="a configuration error"):
        supports_cache(model)


@pytest.mark.parametrize("name", ["tiny", "tiny-hybrid", "tiny-mamba2", "tiny-gdn2"])
@pytest.mark.parametrize("scan", [True, False])
def test_buckets_scan_and_batching_do_not_change_rows(name, scan):
    model = _model(name, scan_layers=scan)
    prompts = _prompts(model.config.vocab_size, lengths=(7, 12, 2, 18))
    batched = generate(model, prompts, max_new_tokens=5, bucket=4)
    exact = generate(model, prompts, max_new_tokens=5, bucket=1)
    np.testing.assert_array_equal(batched.tokens, exact.tokens)
    for b, prompt in enumerate(prompts):
        alone = generate(model, [prompt], max_new_tokens=5, bucket=1)
        assert alone.row(0) == batched.row(b)


@pytest.mark.parametrize("side", ["left", "right"])
def test_left_and_right_padded_batches(side):
    model = _model("tiny-hybrid")
    prompts = _prompts(model.config.vocab_size, lengths=(4, 10, 6))
    width = 12
    ids = np.full((3, width), 7, np.int32)
    mask = np.zeros((3, width), np.int32)
    for b, p in enumerate(prompts):
        cols = slice(width - len(p), width) if side == "left" else slice(0, len(p))
        ids[b, cols] = p
        mask[b, cols] = 1
    assert prompts_from_padded(ids, mask) == prompts
    result = generate(model, prompts_from_padded(ids, mask), max_new_tokens=6)
    for b, prompt in enumerate(prompts):
        assert generate(model, [prompt], max_new_tokens=6).row(0) == result.row(b)
    with pytest.raises(ValueError, match="contiguous"):
        prompts_from_padded(ids, np.array([[1, 0, 1] + [0] * (width - 3)] * 3))


@pytest.mark.parametrize("name", ["tiny-nsa", "tiny-dsa", "tiny-loglinear-mamba2"])
def test_layers_without_a_decode_cache_recompute_the_forward(name):
    model = _model(name)
    assert not supports_cache(model)
    prompts = _prompts(model.config.vocab_size, lengths=(6, 3))
    result = generate(model, prompts, max_new_tokens=4, bucket=1)
    _check_against_forward(model, prompts, result)


def test_eos_and_max_new_tokens_stop_rows():
    model = _model("tiny-hybrid")
    prompts = _prompts(model.config.vocab_size)
    free = generate(model, prompts, max_new_tokens=8, bucket=1)
    eos = free.row(0)[2]  # row 0 meets this token as its third new token
    out = generate(model, prompts, max_new_tokens=8, bucket=1, eos_ids=(eos,), pad_id=1)
    for b in range(3):
        row = free.row(b)
        stop = row.index(eos) + 1 if eos in row else 8
        assert out.row(b) == row[:stop]
        assert bool(out.stopped[b]) == (eos in row)
        assert out.tokens[b, stop:].tolist() == [1] * (8 - stop)
    assert out.lengths[0] <= 3 and out.stopped[0]


def test_sampling_is_reproducible_and_batch_independent():
    model = _model("tiny-hybrid")
    prompts = _prompts(model.config.vocab_size, lengths=(5, 9, 3, 7))
    cfg = SamplingConfig(temperature=1.0, top_k=50, top_p=0.9)
    a = generate(model, prompts, max_new_tokens=8, sampling=cfg, seed=3)
    b = generate(model, prompts, max_new_tokens=8, sampling=cfg, seed=3)
    np.testing.assert_array_equal(a.tokens, b.tokens)
    c = generate(model, prompts, max_new_tokens=8, sampling=cfg, seed=4)
    assert not np.array_equal(a.tokens, c.tokens)
    # Row ids, not batch composition, select the sampling stream.
    pair = generate(model, prompts[2:], max_new_tokens=8, sampling=cfg, seed=3, row_ids=[2, 3])
    np.testing.assert_array_equal(pair.tokens, a.tokens[2:])
    # The cache-free loop draws the same samples.
    ref = generate(model, prompts, max_new_tokens=8, sampling=cfg, seed=3, use_cache=False)
    np.testing.assert_array_equal(ref.tokens, a.tokens)
    # top_k = 1 is greedy.
    k1 = generate(model, prompts, max_new_tokens=8, sampling=SamplingConfig(temperature=0.7, top_k=1), seed=5)
    np.testing.assert_array_equal(k1.tokens, generate(model, prompts, max_new_tokens=8).tokens)


def _hf_style_filter(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    """Independent NumPy version of HF's TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper."""
    x = logits.astype(np.float64) / temperature
    if top_k:
        kth = np.sort(x)[-top_k]
        x = np.where(x < kth, -np.inf, x)
    if top_p < 1.0:
        order = np.argsort(-x, kind="stable")
        p = np.exp(x[order] - x[order][0])
        p /= p.sum()
        before = np.cumsum(p) - p
        removed = order[before >= top_p]
        x[removed] = -np.inf
    return x


@pytest.mark.parametrize(("temperature", "top_k", "top_p"), [(1.0, 0, 1.0), (0.7, 5, 1.0), (1.3, 0, 0.8),
                                                            (0.9, 12, 0.6), (1.0, 3, 0.05)])
def test_filters_match_an_independent_reference(temperature, top_k, top_p):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 40)).astype(np.float32) * 2.0
    cfg = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p)
    ours = np.asarray(warp_logits(jnp.asarray(logits), cfg))
    for b in range(4):
        ref = _hf_style_filter(logits[b], temperature, top_k, top_p)
        np.testing.assert_array_equal(np.isinf(ours[b]), np.isinf(ref))
        kept = ~np.isinf(ref)
        np.testing.assert_allclose(ours[b][kept], ref[kept], rtol=1e-6)


def test_sample_frequencies_follow_the_filtered_distribution():
    logits = jnp.log(jnp.asarray([0.4, 0.25, 0.15, 0.1, 0.06, 0.04]))
    cfg = SamplingConfig(temperature=1.0, top_k=5, top_p=0.85)  # keeps 0.4, 0.25, 0.15, 0.1
    n = 40000
    keys = jax.random.split(jax.random.key(0), n)
    draws = np.asarray(jax.vmap(lambda k: sample_token(logits, k, cfg))(keys))
    freq = np.bincount(draws, minlength=6) / n
    expected = np.array([0.4, 0.25, 0.15, 0.1, 0.0, 0.0]) / 0.9
    np.testing.assert_allclose(freq, expected, atol=0.01)
    assert sample_token(logits, keys[0], GREEDY) == 0


def test_rwkv7_generation_matches_its_own_decode_step():
    """A bare RWKV7LM generates (wrapped by decoding_model): its forward logits are RWKV7LM's, and every
    generated token is the argmax of RWKV7LM's own one-token decode_step fed the prompt and the generated
    prefix (prefill in chunks of 8 through the chunked kernel)."""
    cfg = dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], compute_dtype="float32")
    lm = _rwkv7_moved(RWKV7LM(cfg, key=jax.random.key(0)))
    wrapped = decoding_model(lm)
    assert isinstance(wrapped, RWKV7Decoder) and decoding_model(wrapped) is wrapped and supports_cache(lm)
    prompts = _prompts(cfg.vocab_size, lengths=(37, 21, 50), seed=5)
    seq = jnp.asarray([prompts[0]], jnp.int32)
    np.testing.assert_allclose(np.asarray(wrapped.logits(wrapped(seq).hidden)), np.asarray(lm(seq)), rtol=1e-6,
                               atol=1e-6)
    result = generate(lm, prompts, max_new_tokens=6, bucket=16, prefill_chunk=8)
    assert result.lengths.tolist() == [6, 6, 6]
    for b, prompt in enumerate(prompts):
        cache = lm.init_cache(1)
        steps = []
        for t in prompt + result.row(b):
            logits, cache = lm.decode_step(cache, jnp.asarray([t], jnp.int32))
            steps.append(np.asarray(logits)[0])
        for i, token in enumerate(result.row(b)):
            row = steps[len(prompt) + i - 1]
            assert token == row.argmax() or row.max() - row[token] < 1e-4, (b, i)
    sampled = generate(lm, prompts, max_new_tokens=6, sampling=SamplingConfig(temperature=0.9, top_p=0.9), seed=2)
    np.testing.assert_array_equal(
        sampled.tokens, generate(lm, prompts, max_new_tokens=6, sampling=SamplingConfig(temperature=0.9, top_p=0.9),
                                 seed=2, use_cache=False).tokens)


def test_invalid_arguments_fail_loudly():
    model = _model("tiny")
    with pytest.raises(ValueError):
        SamplingConfig(temperature=-1.0)
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0)
    with pytest.raises(ValueError, match="at least one token"):
        generate(model, [[]], max_new_tokens=2)
    with pytest.raises(ValueError, match="token ids"):
        generate(model, [[model.config.vocab_size]], max_new_tokens=2)


def test_generate_cli_with_prompt_ids(capsys):
    from jla.cli import main

    assert main(["generate", "--preset", "tiny", "--compute-dtype", "float32", "--prompt-ids", "1 2 3",
                 "--prompt-ids", "4 5", "--max-new-tokens", "4", "--batch-size", "1"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    model = build_model(_f32("tiny"), key=jax.random.key(0))
    expected = generate(model, [[1, 2, 3], [4, 5]], max_new_tokens=4)
    assert [line["tokens"] for line in lines] == [expected.row(0), expected.row(1)]
    assert [line["index"] for line in lines] == [0, 1]


def test_generate_cli_sorts_by_length_and_keeps_input_order(capsys):
    from jla.cli import main

    prompts = ["1 2 3 4 5 6 7 8 9", "4 5", "7 8 9 10", "11"]
    argv = ["generate", "--preset", "tiny-hybrid", "--compute-dtype", "float32", "--max-new-tokens", "3",
            "--batch-size", "2", "--temperature", "0.8", "--seed", "3"]
    for p in prompts:
        argv += ["--prompt-ids", p]
    assert main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["index"] for line in lines] == [0, 1, 2, 3]
    model = build_model(_f32("tiny-hybrid"), key=jax.random.key(0))
    rows = [[int(t) for t in p.split()] for p in prompts]
    expected = generate(model, rows, max_new_tokens=3, sampling=SamplingConfig(temperature=0.8), seed=3)
    assert [line["tokens"] for line in lines] == [expected.row(b) for b in range(4)]


@pytest.mark.parametrize("preset", ["tiny-rwkv7", "tiny-rwkv7-blinkdl"])  # LanguageModel, RWKV7LM
def test_generate_cli_rwkv7_preset(capsys, preset):
    from jla.cli import main

    assert main(["generate", "--preset", preset, "--compute-dtype", "float32", "--prompt-ids", "1 2 3",
                 "--max-new-tokens", "4"]) == 0
    (line,) = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    # The CLI builds the preset at its initial weights.
    assert line["tokens"] == generate(_model(preset, moved=False), [[1, 2, 3]], max_new_tokens=4).row(0)


@pytest.mark.parametrize("payload_format", ["safetensors", "sharded", "orbax"])
def test_checkpoint_round_trip(tmp_path, payload_format):
    """A model saved by the trainer's checkpoint manager loads back bit-exactly (run dir layout)."""
    from jla.config import ExperimentConfig
    from jla.inference.loading import load_model
    from jla.training.checkpointing import CheckpointManager, wait_for_pending_checkpoints

    if payload_format == "orbax":
        pytest.importorskip("orbax.checkpoint")
    cfg = ExperimentConfig(model=_f32("tiny-hybrid"))
    cfg.save(tmp_path / "config.toml")
    model = build_model(cfg.model, key=jax.random.key(42))
    manager = CheckpointManager(tmp_path / "checkpoints", asynchronous=False, payload_format=payload_format)
    manager.save(7, model=model, opt_state={"m": jnp.zeros(3)}, data_state={"step": 7}, rng_key=jax.random.key(0))
    wait_for_pending_checkpoints()
    loaded = load_model(run_dir=tmp_path)
    for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    other = load_model(run_dir=tmp_path, compute_dtype="bfloat16")
    assert other.config.compute_dtype == "bfloat16"
    with pytest.raises(ValueError, match="exactly one"):
        load_model(run_dir=tmp_path, preset="tiny")


def _rwkv7_moved(model, scale=0.05, seed=11):
    """Every float parameter of an RWKV-7 model moved off its initial value. RWKV-7 zero-initialises
    ``o_proj`` and the channel-mix ``value`` (as FLA and BlinkDL do), so each block is the identity at
    init and the comparisons here would pass without running its mixer, caches or segment resets."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)
