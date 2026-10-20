# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline tokenizers for the SFT tests.

* :class:`FakeChatTokenizer`: pure Python, character level, ChatML special tokens and a
  ChatML ``apply_chat_template`` (``prefix_stable=False`` drops ``<think>`` blocks from
  earlier assistant turns, as reasoning templates do);
* :func:`hf_chatml_tokenizer`: a real ``transformers`` fast tokenizer (character-level
  WordLevel model) with a ChatML Jinja template that has ``{% generation %}`` blocks;
* :func:`expected` builds the reference ``(ids, mask)`` of hand-written pieces.
"""

from __future__ import annotations

import re
from typing import Any

SPECIALS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
CHARS = [chr(i) for i in range(32, 127)] + ["\n"]

CHATML_GENERATION = (
    "{%- for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' }}{% generation %}{{ message['content'] + '<|im_end|>\\n' }}{% endgeneration %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class FakeChatTokenizer:
    """Character-level tokenizer: specials are ids 1-3, characters follow (``<unk>`` = 0).

    ``end_on_next=True`` renders each ``<|im_end|>`` at the start of the *following*
    message, as templates that attribute the end-of-turn token to the next turn do.
    """

    eos_token_id = 2  # <|im_end|>

    def __init__(self, *, prefix_stable: bool = True, end_on_next: bool = False) -> None:
        self.vocab = {"<unk>": 0, **{s: i + 1 for i, s in enumerate(SPECIALS)}}
        self.vocab.update({c: len(self.vocab) + i for i, c in enumerate(CHARS)})
        self._inverse = {i: t for t, i in self.vocab.items()}
        self.prefix_stable = prefix_stable
        self.end_on_next = end_on_next
        self.chat_template = "fake chatml"
        self._split = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

    def get_added_vocab(self) -> dict[str, int]:
        return {s: self.vocab[s] for s in SPECIALS}

    def decode(self, ids) -> str:
        return "".join(self._inverse.get(int(i), "") for i in ids)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._split.split(text):
            if piece in self.vocab and piece in SPECIALS:
                ids.append(self.vocab[piece])
            else:
                ids.extend(self.vocab.get(c, 0) for c in piece)
        return ids

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text) for text in texts]}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **_: Any):
        assert not tokenize
        last = len(messages) - 1
        text = ""
        for i, message in enumerate(messages):
            content = message["content"]
            if not self.prefix_stable and message["role"] == "assistant" and i < last:
                content = re.sub(r"<think>.*?</think>", "", content)
            if self.end_on_next:
                text += ("<|im_end|>\n" if i else "") + f"<|im_start|>{message['role']}\n{content}"
            else:
                text += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += ("<|im_end|>\n" if self.end_on_next and messages else "") + "<|im_start|>assistant\n"
        return text


def expected(encode, pieces: list[tuple[str, int]]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for text, trained in pieces:
        piece = encode(text)
        ids += piece
        mask += [trained] * len(piece)
    return ids, mask


def hf_chatml_tokenizer() -> Any:
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, **{s: i + 1 for i, s in enumerate(SPECIALS)}}
    vocab.update({c: len(vocab) + i for i, c in enumerate(CHARS)})
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    model.add_special_tokens(list(SPECIALS))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=model, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tokenizer.chat_template = CHATML_GENERATION
    return tokenizer
