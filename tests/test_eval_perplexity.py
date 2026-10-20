# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scoring and perplexity: chunked token log-probs equal a direct log-softmax of the full logits;
windows score every token once; document masking equals evaluating documents one by one; RWKV-7;
out-of-vocabulary ids fail loudly."""

from __future__ import annotations

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.data.cidx import write_cidx
from jla.eval import perplexity, score_sequences, stream_windows, token_logprobs
from jla.inference.rwkv7 import decoding_model
from jla.models import build_model, get_preset
from jla.models.rwkv7 import RWKV7_PRESETS, RWKV7LM


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _model(name="tiny-hybrid", seed=0):
    if name in RWKV7_PRESETS:
        cfg = dataclasses.replace(RWKV7_PRESETS[name], compute_dtype="float32")
        return decoding_model(_rwkv7_moved(RWKV7LM(cfg, key=jax.random.key(seed))))
    model = build_model(get_preset(name).replace(compute_dtype="float32"), key=jax.random.key(seed))
    return _rwkv7_moved(model) if "rwkv7" in model.config.resolved_layer_types else model


def _direct_logprobs(model, ids) -> np.ndarray:
    """log p(ids[t + 1] | ids[: t + 1]) from the full [T, V] logits (NumPy float64 log-softmax)."""
    logits = np.asarray(model.logits(model(jnp.asarray([ids], jnp.int32), inference=True).hidden), np.float64)[0]
    logits = logits - logits.max(-1, keepdims=True)
    logsm = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return np.array([logsm[t, ids[t + 1]] for t in range(len(ids) - 1)])


@pytest.mark.parametrize("chunk_rows", [7, 1024])
def test_token_logprobs_match_a_direct_log_softmax(chunk_rows):
    model = _model()
    ids = jax.random.randint(jax.random.key(1), (3, 21), 0, model.config.vocab_size)
    logp, greedy = token_logprobs(model, ids, chunk_rows=chunk_rows)
    assert logp.shape == greedy.shape == (3, 20)
    logits = np.asarray(model.logits(model(ids, inference=True).hidden))
    for b in range(3):
        np.testing.assert_allclose(logp[b], _direct_logprobs(model, np.asarray(ids[b]).tolist()), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(greedy[b], logits[b, :-1].argmax(-1) == np.asarray(ids[b, 1:]))


def test_score_sequences_handles_ragged_batches():
    model = _model("tiny-mamba2")
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 512, n).tolist() for n in (2, 17, 5, 33, 9)]
    scores = score_sequences(model, seqs, batch_size=2, bucket=8)
    for s, (logp, greedy) in zip(seqs, scores):
        assert logp.shape == greedy.shape == (len(s) - 1,)
        np.testing.assert_allclose(logp, _direct_logprobs(model, s), rtol=1e-5, atol=1e-5)


def test_rwkv7_scores_match_its_own_logits():
    """A bare RWKV7LM scores (wrapped by decoding_model) exactly as a log-softmax of RWKV7LM's logits."""
    cfg = dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], compute_dtype="float32")
    lm = _rwkv7_moved(RWKV7LM(cfg, key=jax.random.key(1)))
    rng = np.random.default_rng(3)
    seqs = [rng.integers(0, cfg.vocab_size, n).tolist() for n in (40, 7, 23)]
    for s, (logp, _) in zip(seqs, score_sequences(lm, seqs, batch_size=2, bucket=16)):
        logits = np.asarray(lm(jnp.asarray([s], jnp.int32)), np.float64)[0]
        logits -= logits.max(-1, keepdims=True)
        logsm = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        np.testing.assert_allclose(logp, [logsm[t, s[t + 1]] for t in range(len(s) - 1)], rtol=1e-5, atol=1e-5)
    result = perplexity(lm, seqs[0], seq_len=64)
    assert result.tokens == 39 and result.windows == 1


def test_out_of_vocabulary_ids_fail_loudly():
    """An id at or above the vocabulary would gather NaN embeddings (and NaN every row it attends to)."""
    model = _model("tiny")
    vocab = model.config.vocab_size
    for bad in ([1, 2, vocab + 5], [vocab, 2, 3], [-1, 2]):
        with pytest.raises(ValueError, match="token ids"):
            score_sequences(model, [bad])
    with pytest.raises(ValueError, match="pad_id"):
        score_sequences(model, [[1, 2, 3]], pad_id=vocab)
    with pytest.raises(ValueError, match="token ids"):
        perplexity(model, [1, 2, vocab], seq_len=8)


def test_windows_score_every_token_once():
    for n, seq_len, stride in [(50, 8, None), (50, 8, 3), (7, 8, None), (9, 8, 1), (64, 16, 15)]:
        windows = stream_windows(n, seq_len, stride)
        scored = [j for _, end, first in windows for j in range(first, end)]
        assert scored == list(range(1, n))
        for start, end, first in windows:
            assert end - start <= seq_len and start < first
        if len(windows) > 1:  # later windows use their full context
            assert all(end - start == seq_len for start, end, _ in windows[1:])
    with pytest.raises(ValueError):
        stream_windows(10, 8, 8)


def test_perplexity_equals_window_by_window_reference():
    model = _model()
    rng = np.random.default_rng(1)
    stream = rng.integers(0, 512, 61)
    for stride in (None, 4):
        result = perplexity(model, stream, seq_len=12, stride=stride, batch_size=3)
        nll = 0.0
        for start, end, first in stream_windows(stream.size, 12, stride):
            logp = _direct_logprobs(model, stream[start:end].tolist())
            nll -= logp[first - start - 1 :].sum()
        assert result.tokens == stream.size - 1
        np.testing.assert_allclose(result.nll_sum, nll, rtol=1e-5)
        np.testing.assert_allclose(result.perplexity, np.exp(nll / (stream.size - 1)), rtol=1e-5)
    with_prefix = perplexity(model, stream[:10], seq_len=16, prefix_id=3)
    assert with_prefix.tokens == 10
    np.testing.assert_allclose(with_prefix.nll_sum, -_direct_logprobs(model, [3, *stream[:10].tolist()]).sum(),
                               rtol=1e-5)


@pytest.mark.parametrize("name", ["tiny-hybrid", "tiny-mamba3", "tiny-rwkv7", "tiny-rwkv7-blinkdl"])
def test_document_masking_equals_documents_evaluated_alone(name):
    model = _model(name)
    eos = 0
    rng = np.random.default_rng(2)
    docs = [rng.integers(1, model.config.vocab_size, n).tolist() + [eos] for n in (6, 9, 4)]
    stream = np.array([t for d in docs for t in d])
    result = perplexity(model, stream, seq_len=32, document_eos=eos)
    # Alone: every target inside a document; the target after a document's EOS is not scored.
    nll = -sum(_direct_logprobs(model, d).sum() for d in docs)
    assert result.tokens == sum(len(d) - 1 for d in docs)
    np.testing.assert_allclose(result.nll_sum, nll, rtol=1e-5)


def test_moe_capacity_routing_does_not_make_scores_batch_dependent():
    """Capacity routing would let padding and other rows take expert slots; evaluation routes dropless."""
    import dataclasses

    cfg = get_preset("tiny-moe").replace(compute_dtype="float32")
    capacity = build_model(cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=1.25)), key=jax.random.key(0))
    dropless = build_model(cfg.replace(moe=dataclasses.replace(cfg.moe, capacity_factor=None)), key=jax.random.key(0))
    rng = np.random.default_rng(4)
    seqs = [rng.integers(0, cfg.vocab_size, n).tolist() for n in (40, 25, 33, 12)]
    reference = score_sequences(dropless, seqs, batch_size=1, bucket=8)
    for batch_size in (1, 4):
        for (logp, greedy), (ref, ref_greedy) in zip(score_sequences(capacity, seqs, batch_size=batch_size, bucket=8),
                                                     reference):
            np.testing.assert_allclose(logp, ref, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(greedy, ref_greedy)
    np.testing.assert_allclose(reference[0][0], _direct_logprobs(dropless, seqs[0]), rtol=1e-5, atol=1e-5)
    stream = rng.integers(0, cfg.vocab_size, 300)
    ppl = [perplexity(capacity, stream, seq_len=32, batch_size=b).nll_sum for b in (1, 2, 8)]
    np.testing.assert_allclose(ppl, perplexity(dropless, stream, seq_len=32, batch_size=1).nll_sum, rtol=1e-6)


def test_eval_ppl_cli_on_a_cidx_corpus(tmp_path, capsys):
    from jla.cli import main

    rng = np.random.default_rng(3)
    docs = [rng.integers(1, 512, n) for n in (30, 25, 40)]
    write_cidx(tmp_path / "val", docs)
    assert main(["eval", "ppl", "--preset", "tiny", "--compute-dtype", "float32", "--cidx", str(tmp_path / "val"),
                 "--seq-len", "16", "--batch-size", "4"]) == 0
    report = json.loads(capsys.readouterr().out)
    model = build_model(get_preset("tiny").replace(compute_dtype="float32"), key=jax.random.key(0))
    expected = perplexity(model, np.concatenate(docs), seq_len=16, batch_size=4)
    assert report["tokens"] == expected.tokens == 94
    np.testing.assert_allclose(report["perplexity"], expected.perplexity, rtol=1e-6)


def _rwkv7_moved(model, scale=0.05, seed=11):
    """Every float parameter of an RWKV-7 model moved off its initial value. RWKV-7 zero-initialises
    ``o_proj`` and the channel-mix ``value`` (as FLA and BlinkDL do), so each block is the identity at
    init and the comparisons here would pass without running its mixer, caches or segment resets."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)
