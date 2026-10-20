# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""lm-eval adapter on a tiny random model with a local byte-level tokenizer and local tasks (no downloads).

Needs the optional ``lm-eval`` and ``transformers`` packages (skipped otherwise)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402

from jla.eval.lm_eval_adapter import JLALM  # noqa: E402
from jla.inference import generate  # noqa: E402
from jla.inference.rwkv7 import decoding_model  # noqa: E402
from jla.models import build_model, get_preset  # noqa: E402
from jla.models.rwkv7 import RWKV7_PRESETS, RWKV7LM  # noqa: E402

TEXT = "The quick brown fox jumps over the lazy dog. A second sentence follows here."


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _byte_tokenizer(path: Path):
    """GPT-2 style byte-level tokenizer without merges: 256 byte tokens + <|endoftext|> (id 256)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>").save_pretrained(str(path))
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok")
    tokenizer = _byte_tokenizer(path)
    model = build_model(get_preset("tiny-hybrid").replace(compute_dtype="float32"), key=jax.random.key(0))
    return model, tokenizer, path


def _direct(model, ids):
    logits = np.asarray(model.logits(model(jnp.asarray([ids], jnp.int32), inference=True).hidden), np.float64)[0]
    logits -= logits.max(-1, keepdims=True)
    logsm = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return np.array([logsm[t, ids[t + 1]] for t in range(len(ids) - 1)]), logits[:-1].argmax(-1)


def _instance(kind, args, idx=0):
    return Instance(request_type=kind, doc={}, arguments=args, idx=idx)


def test_tokenizer_round_trips(setup):
    _, tok, _ = setup
    ids = tok.encode(TEXT, add_special_tokens=False)
    assert len(ids) == len(TEXT.encode()) and tok.decode(ids) == TEXT and tok.eos_token_id == 256


def test_loglikelihood_matches_a_direct_computation(setup):
    model, tok, _ = setup
    lm = JLALM(model, tok, batch_size=3, max_length=64, max_gen_toks=8)
    pairs = [("The quick brown", " fox"), ("A", " second sentence"), ("", "hello world"), ("lazy dog. ", "A")]
    out = lm.loglikelihood([_instance("loglikelihood", p, i) for i, p in enumerate(pairs)])
    for (context, continuation), (logp, greedy) in zip(pairs, out):
        if context.endswith(" "):  # lm-eval moves trailing context spaces into the continuation
            context, continuation = context.rstrip(), " " + continuation
        ctx = tok.encode(context, add_special_tokens=False) if context else [tok.eos_token_id]
        cont = tok.encode(continuation, add_special_tokens=False)
        ref, argmax = _direct(model, ctx + cont)
        np.testing.assert_allclose(logp, ref[-len(cont):].sum(), rtol=1e-5)
        assert greedy == bool((argmax[-len(cont):] == np.asarray(cont)).all())


def test_model_args_entry_point(setup):
    """``simple_evaluate(model="jla", model_args=...)`` (string or dict) builds the same model and tokenizer.

    ``max_length`` 64 is below the default ``max_gen_toks`` (256): scoring needs no generation budget.
    """
    import lm_eval.api.registry as registry

    model, tok, path = setup
    assert registry.get_model("jla") is JLALM
    request = [_instance("loglikelihood", ("The quick brown", " fox"))]
    expected = JLALM(model, tok).loglikelihood(request)[0][0]
    lm = JLALM.create_from_arg_string(f"preset=tiny-hybrid,tokenizer={path},compute_dtype=float32,max_length=64",
                                      {"batch_size": "2", "max_batch_size": None, "device": None})
    assert lm.batch_size == 2 and lm.max_length == 64 and lm.max_gen_toks == 256
    np.testing.assert_allclose(lm.loglikelihood(request)[0][0], expected, rtol=1e-6)
    # The dict form goes through create_from_arg_obj (lm-eval's evaluator does this for dict model_args).
    lm = registry.get_model("jla").create_from_arg_obj(
        {"preset": "tiny-hybrid", "tokenizer": str(path), "compute_dtype": "float32", "max_length": 64},
        {"batch_size": 3, "max_batch_size": None, "device": None})
    assert lm.batch_size == 3 and lm.max_length == 64
    np.testing.assert_allclose(lm.loglikelihood(request)[0][0], expected, rtol=1e-6)
    with pytest.raises(ValueError, match="unknown model_args"):
        JLALM.create_from_arg_string(f"preset=tiny,tokenizer={path},nonsense=1")
    for batch_size in ("auto", "auto:4"):
        with pytest.raises(ValueError, match="integer batch_size"):
            JLALM.create_from_arg_string(f"preset=tiny,tokenizer={path}", {"batch_size": batch_size})
    with pytest.raises(ValueError, match="conflicts"):
        JLALM.create_from_arg_obj({"preset": "tiny", "tokenizer": str(path), "batch_size": 4}, {"batch_size": 2})
    with pytest.raises(ValueError, match="unsupported options"):
        JLALM.create_from_arg_string(f"preset=tiny,tokenizer={path}", {"device": "cuda:0"})


def test_lm_eval_builtin_models_stay_registered():
    """Registering ``jla`` must not hide lm-eval's own models (its registry loads them only when empty)."""
    import lm_eval.api.registry as registry

    assert registry.get_model("dummy").__name__ == "DummyLM"
    assert "hf" in registry.model_registry


def test_generation_budget_is_checked_per_request(setup):
    model, tok, _ = setup
    lm = JLALM(model, tok, max_length=16)  # default max_gen_toks 256: scoring works, long generations fail
    (logp, _), = lm.loglikelihood([_instance("loglikelihood", ("lazy", " dog"))])
    assert np.isfinite(logp)
    with pytest.raises(ValueError, match="max_gen_toks 256 must be below max_length 16"):
        lm.generate_until([_instance("generate_until", ("lazy", {"until": ["\n"]}))])
    (text,) = lm.generate_until([_instance("generate_until", ("lazy", {"until": ["\n"], "max_gen_toks": 4}))])
    ids = generate(model, [tok.encode("lazy", add_special_tokens=False)], max_new_tokens=4,
                   eos_ids=(tok.eos_token_id,), pad_id=tok.eos_token_id).row(0)
    assert text == tok.decode([t for t in ids if t != tok.eos_token_id]).split("\n")[0]


def test_generation_options_follow_lm_eval_normalization(setup):
    """lm-eval's normalize_gen_kwargs (as HFLM): max-token aliases, do_sample from temperature, HF's
    sampling defaults (temperature 1, top_k 50), num_beams 1 only, unknown options fail."""
    model, tok, _ = setup
    lm = JLALM(model, tok, max_gen_toks=7)

    def key(kwargs):
        return dict(lm._generation_key(kwargs))

    eos = "<|endoftext|>"
    assert key({"until": ["\n"]}) == {"until": ("\n", eos), "max_gen_toks": 7, "temperature": 0.0, "top_k": 0,
                                      "top_p": 1.0}
    assert key({"until": "x", "max_new_tokens": 5, "num_beams": 1})["max_gen_toks"] == 5
    assert key({"max_tokens": 4})["max_gen_toks"] == 4 and key({"max_completion_tokens": 3})["max_gen_toks"] == 3
    assert key({"max_gen_toks": 2, "max_new_tokens": 9})["max_gen_toks"] == 2  # lm-eval's priority
    assert key({"temperature": 0.7}) == {"until": (eos,), "max_gen_toks": 7, "temperature": 0.7, "top_k": 50,
                                         "top_p": 1.0}  # do_sample unset: samples when temperature > 0
    assert key({"do_sample": True})["temperature"] == 1.0
    assert key({"do_sample": False, "temperature": 0.7})["temperature"] == 0.0
    sampled = key({"do_sample": True, "temperature": 0.5, "top_k": 0, "top_p": 0.9})
    assert (sampled["temperature"], sampled["top_k"], sampled["top_p"]) == (0.5, 0, 0.9)
    with pytest.raises(ValueError, match="beam search"):
        key({"num_beams": 4})
    with pytest.raises(ValueError, match="unsupported generation options"):
        key({"repetition_penalty": 1.1})
    ids = tok.encode("Hi", add_special_tokens=False)
    assert lm.tok_decode(ids + [tok.eos_token_id]) == "Hi"  # HFLM decodes without special tokens
    assert lm.tok_decode(ids + [tok.eos_token_id], skip_special_tokens=False) == "Hi" + eos


def test_rwkv7_through_the_adapter(setup):
    _, tok, _ = setup
    small = RWKV7LM(dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], compute_dtype="float32"),
                    key=jax.random.key(0))
    with pytest.raises(ValueError, match="257 tokens"):  # vocabulary 256 < the byte tokenizer's 257
        JLALM(small, tok)
    rwkv = RWKV7LM(dataclasses.replace(RWKV7_PRESETS["tiny-rwkv7-blinkdl"], vocab_size=512, compute_dtype="float32"),
                   key=jax.random.key(0))
    lm = JLALM(rwkv, tok, max_length=64, max_gen_toks=4)
    (logp, _), = lm.loglikelihood([_instance("loglikelihood", ("The quick brown", " fox"))])
    ref, _ = _direct(decoding_model(rwkv), tok.encode("The quick brown fox", add_special_tokens=False))
    np.testing.assert_allclose(logp, ref[-4:].sum(), rtol=1e-5)
    (text,) = lm.generate_until([_instance("generate_until", ("lazy", {"until": ["\n"], "max_gen_toks": 4}))])
    ids = generate(rwkv, [tok.encode("lazy", add_special_tokens=False)], max_new_tokens=4,
                   eos_ids=(tok.eos_token_id,), pad_id=tok.eos_token_id).row(0)
    assert text == tok.decode([t for t in ids if t != tok.eos_token_id]).split("\n")[0]


def test_loglikelihood_truncates_to_max_length(setup):
    model, tok, _ = setup
    lm = JLALM(model, tok, max_length=16, max_gen_toks=4)
    assert TEXT[39:44] == " dog."
    (logp, _), = lm.loglikelihood([_instance("loglikelihood", (TEXT[:39], TEXT[39:44]))])
    ids = tok.encode(TEXT[:44], add_special_tokens=False)[-17:]  # the last max_length + 1 tokens
    ref, _ = _direct(model, ids)
    np.testing.assert_allclose(logp, ref[-5:].sum(), rtol=1e-5)


@pytest.mark.parametrize("max_length", [16, 128])
def test_loglikelihood_rolling_scores_every_token_once(setup, max_length):
    model, tok, _ = setup
    lm = JLALM(model, tok, batch_size=4, max_length=max_length, max_gen_toks=4)
    (total,) = lm.loglikelihood_rolling([_instance("loglikelihood_rolling", (TEXT,))])
    tokens = tok.encode(TEXT, add_special_tokens=False)
    if max_length >= len(tokens) + 1:
        ref, _ = _direct(model, [tok.eos_token_id] + tokens)
        np.testing.assert_allclose(total, ref.sum(), rtol=1e-5)
    else:  # lm-eval's windows: the first predicts tokens[:L] from EOS + tokens[:L-1]; later ones L tokens each
        first, _ = _direct(model, [tok.eos_token_id] + tokens[:max_length])
        expected, predicted = first.sum(), max_length
        while predicted < len(tokens):
            n = min(len(tokens) - predicted, max_length)
            end = predicted + n
            window = tokens[max(0, end - max_length - 1) : end]
            ref, _ = _direct(model, window)
            expected += ref[-n:].sum()
            predicted = end
        np.testing.assert_allclose(total, expected, rtol=1e-5)


def test_generate_until_uses_greedy_decoding_and_stop_strings(setup):
    model, tok, _ = setup
    lm = JLALM(model, tok, batch_size=2, max_length=64, max_gen_toks=12)
    contexts = ["The quick", "A second sentence", "lazy"]
    requests = [_instance("generate_until", (c, {"until": ["\n"], "max_gen_toks": 10}), i) for i, c in enumerate(contexts)]
    out = lm.generate_until(requests)
    for context, text in zip(contexts, out):
        ids = generate(model, [tok.encode(context, add_special_tokens=False)], max_new_tokens=10,
                       eos_ids=(tok.eos_token_id,), pad_id=tok.eos_token_id).row(0)
        expected = tok.decode([t for t in ids if t != tok.eos_token_id]).split("\n")[0]
        assert text == expected
    # A stop string that occurs cuts the text there.
    first = out[0]
    if len(first) > 2:
        stop = first[2]
        (cut,) = lm.generate_until([_instance("generate_until", ("The quick", {"until": [stop], "max_gen_toks": 10}))])
        assert cut == first.split(stop)[0]


def _header(task: str, data: Path) -> str:
    return (f"task: {task}\ndataset_path: json\ndataset_kwargs:\n  data_files:\n    test: {data}\n"
            "test_split: test\n")


def _write_tasks(root: Path, mc_docs, gen_docs, texts) -> None:
    root.mkdir()
    for name, docs in (("mc", mc_docs), ("gen", gen_docs), ("text", texts)):
        with open(root / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(doc) + "\n")
    (root / "jla_tiny_mc.yaml").write_text(
        _header("jla_tiny_mc", root / "mc.jsonl")
        + "output_type: multiple_choice\ndoc_to_text: question\ndoc_to_choice: choices\ndoc_to_target: answer\n"
        + "metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n")
    (root / "jla_tiny_gen.yaml").write_text(
        _header("jla_tiny_gen", root / "gen.jsonl")
        + "output_type: generate_until\ndoc_to_text: prompt\ndoc_to_target: target\n"
        + 'generation_kwargs:\n  until: ["\\n"]\n  max_gen_toks: 6\n  do_sample: false\n'
        + "process_results: !function utils.process_results\n"
        + "metric_list:\n  - metric: exact\n    aggregation: mean\n    higher_is_better: true\n")
    (root / "jla_tiny_text.yaml").write_text(
        _header("jla_tiny_text", root / "text.jsonl")
        + 'output_type: loglikelihood_rolling\ndoc_to_text: ""\ndoc_to_target: text\n'
        + "metric_list:\n  - metric: word_perplexity\n  - metric: byte_perplexity\n  - metric: bits_per_byte\n")
    (root / "utils.py").write_text(
        "def process_results(doc, results):\n    return {'exact': float(results[0] == doc['target'])}\n")


def test_simple_evaluate_on_local_tasks_through_the_cli(setup, tmp_path):
    model, tok, tok_path = setup
    lm = JLALM(model, tok, batch_size=4, max_length=64, max_gen_toks=6)
    # Multiple choice: lm-eval scores " " + choice after the question; the answer is the reference
    # argmax for 4 of 6 questions (expected acc 4/6).
    questions = [("Pick one:", ["red", "green", "blue"]), ("Colour?", ["yes", "no", "maybe"]),
                 ("Q3", ["a", "bb", "ccc"]), ("Q4", ["dog", "cat", "cow"]), ("Q5", ["up", "down", "left"]),
                 ("Q6", ["one", "two", "three"])]
    mc_docs = []
    for i, (q, choices) in enumerate(questions):
        scores = [lm.loglikelihood([_instance("loglikelihood", (q, " " + c))])[0][0] for c in choices]
        best = int(np.argmax(scores))
        mc_docs.append({"question": q, "choices": choices, "answer": best if i < 4 else (best + 1) % 3})
    # Generation: the target is our greedy completion for 2 of 3 prompts (expected exact 2/3).
    prompts = ["The quick", "A second", "lazy dog"]
    outs = lm.generate_until([_instance("generate_until", (p, {"until": ["\n"], "max_gen_toks": 6})) for p in prompts])
    gen_docs = [{"prompt": p, "target": o if i < 2 else o + "x"} for i, (p, o) in enumerate(zip(prompts, outs))]
    texts = [{"text": TEXT}, {"text": "Short one."}]
    rolling = lm.loglikelihood_rolling([_instance("loglikelihood_rolling", (t["text"],)) for t in texts])
    bpb = -sum(rolling) / sum(len(t["text"].encode()) for t in texts) / math.log(2)
    _write_tasks(tmp_path / "tasks", mc_docs, gen_docs, texts)

    env = dict(os.environ, HF_HOME=str(tmp_path / "hf"), HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    cmd = [sys.executable, "-c", "import jax, sys; jax.config.update('jax_default_matmul_precision', 'highest');"
           " from jla.cli import main; sys.exit(main(sys.argv[1:]))",
           "eval", "lm-eval", "--preset", "tiny-hybrid", "--compute-dtype", "float32", "--tokenizer", str(tok_path),
           "--local-files-only", "--tasks", "jla_tiny_mc,jla_tiny_gen,jla_tiny_text",
           "--include-path", str(tmp_path / "tasks"), "--batch-size", "4", "--max-length", "64",
           "--output", str(tmp_path / "results.json")]  # default --max-gen-toks 256 > --max-length; tasks set 6
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    results = json.loads(proc.stdout.strip().splitlines()[-1])["results"]
    assert results["jla_tiny_mc"]["acc,none"] == pytest.approx(4 / 6)
    assert results["jla_tiny_gen"]["exact,none"] == pytest.approx(2 / 3)
    assert results["jla_tiny_text"]["bits_per_byte,none"] == pytest.approx(bpb, rel=1e-5)
