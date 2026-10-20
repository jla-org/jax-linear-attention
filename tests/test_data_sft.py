# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SFT data: chat/instruction loss masks, the mask sidecar format and packed rows."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from _sft_fakes import CHATML_GENERATION, FakeChatTokenizer, expected, hf_chatml_tokenizer
from jla.data.cidx import MMapIndexedDataset
from jla.data.sft import (
    LossMaskFile,
    SftDataset,
    SftWriter,
    pack_best_fit_decreasing,
    read_mask_digest,
    write_sft,
)
from jla.data.sft_tokenize import (
    ChatTemplateError,
    chat_example,
    has_generation_blocks,
    instruction_example,
    tokenize_sft_jsonl,
    turns_without_stop_token,
)

CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "4"},
]


def _pieces(trained: dict[int, int]) -> list[tuple[str, int]]:
    """Hand-written ChatML rendering of CHAT; ``trained`` maps message index -> 1 for trained turns."""
    pieces = []
    for i, message in enumerate(CHAT):
        if message["role"] == "assistant":
            pieces.append(("<|im_start|>assistant\n", 0))
            pieces.append((message["content"] + "<|im_end|>\n", trained.get(i, 0)))
        else:
            pieces.append((f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n", trained.get(i, 0)))
    return pieces


# -- chat masks on a fake tokenizer ---------------------------------------------------


def test_chat_prefix_masks_train_assistant_turns_and_their_end_tokens() -> None:
    tok = FakeChatTokenizer()
    ids, mask = chat_example(tok, CHAT, mask_method="prefix")
    assert (ids, mask) == expected(tok.encode, _pieces({2: 1, 4: 1}))
    assert chat_example(tok, CHAT) == (ids, mask)  # auto -> prefix without generation blocks


def test_chat_last_turn_weights_roles_and_all() -> None:
    tok = FakeChatTokenizer()
    assert chat_example(tok, CHAT, loss_on="last") == expected(tok.encode, _pieces({4: 1}))
    weighted = [dict(m) for m in CHAT]
    weighted[4]["weight"] = 0
    assert chat_example(tok, weighted) == expected(tok.encode, _pieces({2: 1}))
    assert chat_example(tok, weighted, loss_on="last")[1] == [0] * len(expected(tok.encode, _pieces({}))[1])
    ids, mask = chat_example(tok, CHAT, loss_on="all")
    assert ids == expected(tok.encode, _pieces({}))[0] and mask == [1] * len(ids)
    with_tool = CHAT[:3] + [{"role": "tool", "content": "ok"}] + CHAT[3:]
    ids, mask = chat_example(tok, with_tool, train_roles=("assistant", "tool"))
    ref = expected(
        tok.encode,
        [
            ("<|im_start|>system\nBe brief.<|im_end|>\n", 0),
            ("<|im_start|>user\nHi<|im_end|>\n", 0),
            ("<|im_start|>assistant\n", 0),
            ("Hello!<|im_end|>\n", 1),
            ("<|im_start|>tool\nok<|im_end|>\n", 1),
            ("<|im_start|>user\n2+2?<|im_end|>\n", 0),
            ("<|im_start|>assistant\n", 0),
            ("4<|im_end|>\n", 1),
        ],
    )
    assert (ids, mask) == ref


def test_chat_rejects_bad_messages_and_unstable_templates() -> None:
    tok = FakeChatTokenizer()
    for bad in ([], [{"content": "x"}], [{"role": "assistant", "content": "x", "weight": 2}]):
        with pytest.raises(ValueError):
            chat_example(tok, bad)
    thinking = [dict(m) for m in CHAT]
    thinking[2]["content"] = "<think>hmm</think>Hello!"
    unstable = FakeChatTokenizer(prefix_stable=False)
    with pytest.raises(ChatTemplateError, match="prefix-stable"):
        chat_example(unstable, thinking)
    with pytest.raises(ChatTemplateError, match="generation"):
        chat_example(tok, CHAT, mask_method="generation")


def test_instruction_masks_train_completion_and_eos() -> None:
    tok = FakeChatTokenizer()
    encode = lambda texts: tok(texts)["input_ids"]  # noqa: E731
    ids, mask = instruction_example(encode, "Q: 2+2=", "4", eos_id=tok.eos_token_id)
    assert ids == tok.encode("Q: 2+2=") + tok.encode("4") + [tok.eos_token_id]
    assert mask == [0] * 7 + [1, 1]
    assert instruction_example(encode, "Q", "A", eos_id=None, append_eos=False) == (tok.encode("QA"), [0, 1])
    assert instruction_example(encode, "Q", "A", eos_id=2, loss_on="all")[1] == [1, 1, 1]


def test_tokenize_jsonl_writes_corpus_and_counts_skips(tmp_path: Path) -> None:
    tok = FakeChatTokenizer()
    lines = [
        json.dumps({"messages": CHAT}),
        "not json",
        json.dumps({"messages": [{"role": "user", "content": "only a prompt"}]}),  # no trained token
        json.dumps({"messages": "oops"}),
        json.dumps({"messages": CHAT[:3]}),
    ]
    source = tmp_path / "chat.jsonl"
    source.write_text("\n".join(lines) + "\n")
    stats = tokenize_sft_jsonl([source], tmp_path / "sft", tokenizer=tok, format="chat")
    assert (stats.documents, stats.skipped_invalid_json, stats.skipped_no_target, stats.skipped_invalid_record) == (2, 1, 1, 1)
    assert stats.mask_method == "prefix"
    corpus = MMapIndexedDataset(str(tmp_path / "sft"))
    mask = LossMaskFile(tmp_path / "sft", documents=corpus.num_documents, tokens=corpus.num_tokens)
    ids0, mask0 = chat_example(tok, CHAT)
    np.testing.assert_array_equal(corpus.get_document(0), ids0)
    np.testing.assert_array_equal(mask.data[: len(ids0)], mask0)
    assert stats.trainable_tokens == int(mask.data.sum())

    rows = tmp_path / "inst.jsonl"
    rows.write_text(json.dumps({"instruction": "Add", "input": "2+2", "output": "4"}) + "\n"
                    + json.dumps({"instruction": "Say hi", "input": "", "output": "hi"}) + "\n")
    stats = tokenize_sft_jsonl([rows], tmp_path / "inst", tokenizer=tok, format="instruction",
                               prompt_fields=("instruction", "input"), completion_field="output")
    corpus = MMapIndexedDataset(str(tmp_path / "inst"))
    assert stats.documents == 2
    np.testing.assert_array_equal(corpus.get_document(1), tok.encode("Say hi") + tok.encode("hi") + [2])

    unstable_input = tmp_path / "think.jsonl"
    thinking = [dict(m) for m in CHAT]
    thinking[2]["content"] = "<think>x</think>Hello!"
    unstable_input.write_text(json.dumps({"messages": thinking}) + "\n")
    with pytest.raises(ChatTemplateError):
        tokenize_sft_jsonl([unstable_input], tmp_path / "bad", tokenizer=FakeChatTokenizer(prefix_stable=False))
    assert not list(tmp_path.glob("bad*")) and not list(tmp_path.glob(".bad*"))
    mixed = tmp_path / "mixed.jsonl"  # skip mode: the unstable record is left out, the stable one is kept
    mixed.write_text(json.dumps({"messages": thinking}) + "\n" + json.dumps({"messages": CHAT}) + "\n")
    unstable_tok = FakeChatTokenizer(prefix_stable=False)
    stats = tokenize_sft_jsonl([mixed], tmp_path / "mixed", tokenizer=unstable_tok, on_template_error="skip")
    assert (stats.documents, stats.skipped_template_error) == (1, 1)
    np.testing.assert_array_equal(MMapIndexedDataset(str(tmp_path / "mixed")).get_document(0), chat_example(unstable_tok, CHAT)[0])
    untemplated = FakeChatTokenizer()
    untemplated.chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        tokenize_sft_jsonl([source], tmp_path / "none", tokenizer=untemplated)


def test_tokenize_requires_trained_turns_to_end_with_their_stop_token(tmp_path: Path) -> None:
    """A template that renders <|im_end|> at the start of the next message leaves it untrained."""
    late = FakeChatTokenizer(end_on_next=True)
    ids, mask = chat_example(late, CHAT)  # prefix-stable, but the trained spans are just "Hello!" and "4"
    assert late.decode([t for t, m in zip(ids, mask) if m]) == "Hello!4"
    assert turns_without_stop_token(ids, mask, late) == 2
    tok = FakeChatTokenizer()
    assert turns_without_stop_token(*chat_example(tok, CHAT), tok) == 0  # "...<|im_end|>\n": whitespace skipped
    source = tmp_path / "chat.jsonl"
    source.write_text(json.dumps({"messages": CHAT}) + "\n" + json.dumps({"messages": CHAT[:3]}) + "\n")
    with pytest.raises(ChatTemplateError, match="never learn to stop"):
        tokenize_sft_jsonl([source], tmp_path / "a", tokenizer=late)
    assert not list(tmp_path.glob("a.*"))
    # An EOS appended after the last turn is its stop token: the one-turn record passes, the two-turn one does not.
    stats = tokenize_sft_jsonl([source], tmp_path / "b", tokenizer=late, append_eos=True, on_template_error="skip")
    assert (stats.documents, stats.skipped_template_error) == (1, 1)
    assert tokenize_sft_jsonl([source], tmp_path / "c", tokenizer=late, check_stop_token=False).documents == 2
    assert tokenize_sft_jsonl([source], tmp_path / "d", tokenizer=late, loss_on="all").documents == 2
    assert tokenize_sft_jsonl([source], tmp_path / "e", tokenizer=tok).documents == 2


def test_dict_chat_templates_use_generation_only_when_every_variant_has_blocks() -> None:
    assert has_generation_blocks({"default": CHATML_GENERATION, "tool_use": CHATML_GENERATION})
    assert not has_generation_blocks({"default": CHATML_GENERATION, "tool_use": "{{ messages }}"})
    assert not has_generation_blocks({}) and not has_generation_blocks(None)


# -- a real Hugging Face tokenizer and template ---------------------------------------


def test_hf_chat_template_masks_match_the_reference() -> None:
    pytest.importorskip("transformers")
    tok = hf_chatml_tokenizer()
    encode = lambda text: tok([text], add_special_tokens=False)["input_ids"][0]  # noqa: E731
    reference = expected(encode, _pieces({2: 1, 4: 1}))
    assert chat_example(tok, CHAT, mask_method="prefix") == reference
    assert chat_example(tok, CHAT, mask_method="generation") == reference  # HF return_assistant_tokens_mask
    assert chat_example(tok, CHAT) == reference  # auto -> generation
    last = expected(encode, _pieces({4: 1}))
    assert chat_example(tok, CHAT, mask_method="generation", loss_on="last") == last
    assert chat_example(tok, CHAT, mask_method="prefix", loss_on="last") == last
    assert turns_without_stop_token(*reference, tok) == 0  # <|im_end|> is an added token of the real tokenizer
    n = len(reference[0])  # the conversation ends with "4", "<|im_end|>", "\n"
    assert turns_without_stop_token(reference[0], [int(i == n - 3) for i in range(n)], tok) == 1  # only "4"
    # One generation span per assistant message, always: an empty span would shift every later turn's mask.
    content_only = CHATML_GENERATION.replace(
        "{% generation %}{{ message['content'] + '<|im_end|>\\n' }}{% endgeneration %}",
        "{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\\n' }}",
    )
    assert content_only != CHATML_GENERATION
    content_pieces = []  # this template marks only the content: <|im_end|> stays untrained
    for text, trained in _pieces({2: 1, 4: 1}):
        content_pieces += [(text[: -len("<|im_end|>\n")], 1), ("<|im_end|>\n", 0)] if trained else [(text, 0)]
    content_masks = chat_example(tok, CHAT, mask_method="generation", chat_template=content_only)
    assert content_masks == expected(encode, content_pieces)
    assert turns_without_stop_token(*content_masks, tok) == 2  # what the tokenize check refuses
    empty_turn = [dict(m) for m in CHAT]
    empty_turn[2]["content"] = ""
    with pytest.raises(ChatTemplateError, match="generation spans"):
        chat_example(tok, empty_turn, mask_method="generation", chat_template=content_only)


def test_cli_tokenizes_chat_and_instruction_with_a_saved_tokenizer(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from jla.cli import main

    tok = hf_chatml_tokenizer()
    tok.save_pretrained(tmp_path / "tok")
    chats = tmp_path / "chats.jsonl"
    chats.write_text(json.dumps({"messages": CHAT}) + "\n")
    out = tmp_path / "chat_sft"
    assert main(["tokenize", "--format", "chat", "--input", str(chats), "--output", str(out),
                 "--tokenizer", str(tmp_path / "tok"), "--local-files-only"]) == 0
    corpus = MMapIndexedDataset(str(out))
    mask = LossMaskFile(out, documents=corpus.num_documents, tokens=corpus.num_tokens)
    encode = lambda text: tok([text], add_special_tokens=False)["input_ids"][0]  # noqa: E731
    ids, ref_mask = expected(encode, _pieces({2: 1, 4: 1}))
    np.testing.assert_array_equal(corpus.get_document(0), ids)
    np.testing.assert_array_equal(mask.data, ref_mask)

    rows = tmp_path / "inst.jsonl"
    rows.write_text(json.dumps({"prompt": "Q", "completion": "A"}) + "\n")
    out = tmp_path / "inst_sft"
    assert main(["tokenize", "--format", "instruction", "--input", str(rows), "--output", str(out),
                 "--tokenizer", str(tmp_path / "tok"), "--local-files-only"]) == 0
    corpus = MMapIndexedDataset(str(out))
    np.testing.assert_array_equal(corpus.get_document(0), encode("Q") + encode("A") + [tok.eos_token_id])


# -- the corpus format ------------------------------------------------------------------


def _random_examples(seed: int, count: int, *, low: int = 2, high: int = 40, vocab: int = 500):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(count):
        n = int(rng.integers(low, high + 1))
        tokens = rng.integers(0, vocab, size=n)
        mask = (rng.random(n) < 0.6).astype(np.uint8)
        examples.append((tokens, mask))
    return examples


def test_mask_sidecar_roundtrip_and_validation(tmp_path: Path) -> None:
    examples = _random_examples(0, 20)
    assert write_sft(tmp_path / "a", examples) == (20, sum(len(t) for t, _ in examples))
    corpus = MMapIndexedDataset(str(tmp_path / "a"))
    mask = LossMaskFile(tmp_path / "a", documents=corpus.num_documents, tokens=corpus.num_tokens)
    np.testing.assert_array_equal(mask.data, np.concatenate([m for _, m in examples]))
    digest = read_mask_digest(tmp_path / "a")
    assert digest == mask.digest
    flipped = [(t, m.copy()) for t, m in examples]
    flipped[3][1][0] ^= 1
    write_sft(tmp_path / "b", flipped)
    assert read_mask_digest(tmp_path / "b") != digest
    write_sft(tmp_path / "c", examples[:5])
    (tmp_path / "c.mask").write_bytes((tmp_path / "a.mask").read_bytes())  # sidecar of another corpus
    with pytest.raises(ValueError, match="another corpus"):
        SftDataset(str(tmp_path / "c"), 16)
    with pytest.raises(ValueError, match="mask values"):
        SftWriter(tmp_path / "d").add([1, 2], [0, 2])
    writer = SftWriter(tmp_path / "e")
    writer.add([1, 2, 3], [0, 1, 1])
    writer.abort()
    assert not list(tmp_path.glob("e.*")) and not list(tmp_path.glob(".e.*"))


# -- packing ----------------------------------------------------------------------------


def _naive_best_fit_decreasing(lengths: np.ndarray, capacity: int) -> list[int]:
    loads: list[int] = []
    for length in sorted(lengths.tolist(), reverse=True):
        fits = [i for i, load in enumerate(loads) if load + length <= capacity]
        if fits:
            best = min(fits, key=lambda i: capacity - loads[i])
            loads[best] += length
        else:
            loads.append(length)
    return sorted(loads)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_fit_decreasing_matches_a_naive_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 65, size=300)
    bins = pack_best_fit_decreasing(lengths, 64)
    assert sorted(i for b in bins for i in b) == list(range(300))
    loads = sorted(int(lengths[b].sum()) for b in bins)
    assert max(loads) <= 64
    assert loads == _naive_best_fit_decreasing(lengths, 64)
    assert pack_best_fit_decreasing(lengths, 64) == bins  # deterministic


def _check_rows(dataset: SftDataset, examples, seq_len: int, *, completion_only: bool = True) -> set[int]:
    seen: set[int] = set()
    for row in range(len(dataset)):
        sample = dataset[row]
        assert all(v.shape == (seq_len,) for v in sample.values())
        cursor = 0
        items = dataset.row_items(row)
        for segment, (_, document) in enumerate(items):
            tokens, mask = examples[document]
            tokens, mask = tokens[: seq_len + 1], mask[: seq_len + 1]
            count = len(tokens) - 1
            span = slice(cursor, cursor + count)
            np.testing.assert_array_equal(sample["input_ids"][span], tokens[:-1])
            np.testing.assert_array_equal(sample["labels"][span], tokens[1:])
            np.testing.assert_array_equal(sample["loss_mask"][span], mask[1:].astype(bool) if completion_only else True)
            np.testing.assert_array_equal(sample["segment_ids"][span], segment)
            np.testing.assert_array_equal(sample["positions"][span], np.arange(count))
            assert document not in seen
            seen.add(document)
            cursor += count
        assert not sample["loss_mask"][cursor:].any()
        np.testing.assert_array_equal(sample["segment_ids"][cursor:], len(items))
        np.testing.assert_array_equal(sample["positions"][cursor:], np.arange(seq_len - cursor))
    return seen


@pytest.mark.parametrize("packing", ["bfd", "single"])
def test_rows_reconstruct_every_example_exactly_once(tmp_path: Path, packing: str) -> None:
    examples = _random_examples(1, 120, high=50)
    examples[7] = (examples[7][0], np.zeros_like(examples[7][1]))  # no trainable target: dropped
    examples[9] = (examples[9][0][:1], examples[9][1][:1])  # one token: no target at all
    write_sft(tmp_path / "s", examples)
    seq_len = 32
    dataset = SftDataset(str(tmp_path / "s"), seq_len, packing=packing)
    seen = _check_rows(dataset, examples, seq_len)
    trainable = {i for i, (t, m) in enumerate(examples) if len(t) >= 2 and m[1 : seq_len + 1].any()}
    assert seen == trainable and 7 not in seen and 9 not in seen
    assert dataset.truncated == sum(1 for i in trainable if len(examples[i][0]) > seq_len + 1)
    if packing == "bfd":
        assert len(dataset) < len(trainable) and dataset.fill_fraction() > 0.8
    else:
        assert len(dataset) == len(trainable)
    restored = pickle.loads(pickle.dumps(dataset))  # loader workers
    for row in (0, len(dataset) - 1):
        for key, value in dataset[row].items():
            np.testing.assert_array_equal(restored[row][key], value)


def test_overlong_drop_completion_only_off_and_pooled_paths(tmp_path: Path) -> None:
    examples = _random_examples(2, 60, high=50)
    write_sft(tmp_path / "s", examples)
    seq_len = 24
    dropped = SftDataset(str(tmp_path / "s"), seq_len, overlong="drop")
    long = {i for i, (t, _) in enumerate(examples) if len(t) > seq_len + 1}
    assert long and dropped.dropped_overlong == len(long) and not (_check_rows(dropped, examples, seq_len) & long)
    everything = SftDataset(str(tmp_path / "s"), seq_len, completion_only=False)
    assert _check_rows(everything, examples, seq_len, completion_only=False) == set(range(60))
    write_sft(tmp_path / "t", examples[:10])
    pooled = SftDataset([str(tmp_path / "s"), str(tmp_path / "t")], seq_len)
    sources = [source for row in range(len(pooled)) for source, _ in pooled.row_items(row)]
    assert sources.count(1) == sum(1 for t, m in examples[:10] if m[1 : seq_len + 1].any())
