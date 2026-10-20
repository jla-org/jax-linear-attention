# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tokenize chat or instruction JSONL into an SFT corpus (CIDX tokens + loss-mask sidecar).

    jla tokenize --format chat --input chats.jsonl --output data/sft \\
        --tokenizer Qwen/Qwen2.5-0.5B-Instruct
    jla tokenize --format instruction --input alpaca.jsonl --output data/sft \\
        --tokenizer EleutherAI/gpt-neox-20b --prompt-field instruction input --completion-field output

``chat`` lines hold ``{"messages": [{"role": ..., "content": ...}, ...]}`` (optionally
``"tools"``); the conversation is rendered with the tokenizer's chat template
(``tokenizer.apply_chat_template``, or ``--chat-template FILE``) and tokenized without
extra special tokens, exactly as at inference. The loss mask marks the tokens of the
trained turns:

* ``--loss-on assistant`` (default): every message whose role is in ``--train-roles``
  (default ``assistant``) and whose optional ``"weight"`` is not ``0`` (the OpenAI
  fine-tuning convention); ``last``: only the last such message; ``all``: every token.
* ``--mask-method prefix`` finds a message's tokens by rendering the conversation up to
  it: an assistant turn spans from the end of ``messages[:i]`` rendered with the
  generation prompt (so the ``assistant`` header is not trained) to the end of
  ``messages[:i + 1]``, which includes its end-of-turn token. Both renderings must be
  token prefixes of the full conversation; templates that rewrite earlier turns
  (for example by dropping reasoning blocks) fail loudly. ``generation`` uses
  Hugging Face's ``return_assistant_tokens_mask`` and needs a template with one
  non-empty ``{% generation %}`` block per assistant message (checked per record);
  ``auto`` picks ``generation`` when the template has them.
* Every trained turn must end with a special token (its end-of-turn marker, or an appended
  EOS), ignoring trailing whitespace; a template that attributes the end-of-turn token to
  the following message would otherwise never teach the model to stop
  (``--no-check-stop-token`` turns the check off).

``instruction`` lines hold a prompt and a completion field (several prompt fields are
joined by ``--prompt-separator``, empty ones skipped). The prompt and the completion
are tokenized separately, as a model sees them at inference, the tokenizer's EOS is
appended (``--no-append-eos`` to skip) and only the completion and EOS are trained.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jla.data.sft import SftWriter

__all__ = [
    "ChatTemplateError",
    "SftTokenizeStats",
    "chat_example",
    "has_generation_blocks",
    "instruction_example",
    "main",
    "tokenize_sft_jsonl",
    "turns_without_stop_token",
]

LOSS_ON = ("assistant", "last", "all")
MASK_METHODS = ("auto", "prefix", "generation")
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


class ChatTemplateError(ValueError):
    """The chat template cannot provide exact per-message token boundaries."""


@dataclass
class SftTokenizeStats:
    format: str = "chat"
    mask_method: str = ""
    documents: int = 0
    tokens: int = 0
    trainable_tokens: int = 0
    eos_id: int | None = None
    skipped_invalid_json: int = 0
    skipped_invalid_record: int = 0
    skipped_empty: int = 0
    skipped_no_target: int = 0
    skipped_template_error: int = 0  # only with on_template_error="skip"
    seconds: float = 0.0


def _encoder(tokenizer: Any) -> Callable[[list[str]], list[list[int]]]:
    def encode(texts: list[str]) -> list[list[int]]:
        out = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        return [list(ids) for ids in out["input_ids"]]

    return encode


def has_generation_blocks(template: Any) -> bool:
    """True if the template (or every named variant of a dict template) has ``{% generation %}`` blocks."""
    if isinstance(template, dict):
        return bool(template) and all(has_generation_blocks(variant) for variant in template.values())
    return isinstance(template, str) and bool(_GENERATION_TAG.search(template))


def _special_token_ids(tokenizer: Any) -> set[int]:
    """Added/special token ids (end-of-turn markers are added tokens, not merges of the base vocabulary)."""
    ids: set[int] = set()
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        ids.update(int(value) for value in added().values())
    ids.update(int(value) for value in (getattr(tokenizer, "all_special_ids", None) or ()))
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    return ids


# The heuristic (the last trained non-whitespace token must be an added token) follows
# is_chat_template_stop_token_trained in TRL (Apache-2.0), trl/chat_template_utils.py; it is
# applied here to every record's actual masks instead of to one probe conversation.
def turns_without_stop_token(
    ids: Sequence[int], mask: Sequence[int], tokenizer: Any, *, special: set[int] | None = None
) -> int:
    """Trained spans whose last non-whitespace token is not a special token (e.g. ``<|im_end|>``).

    A template that attributes an assistant turn's end-of-turn token to the *next*
    message leaves it outside the trained span, so the model never learns to stop.
    ``special`` caches the tokenizer's special ids across calls.
    """
    special = _special_token_ids(tokenizer) if special is None else special
    decode = getattr(tokenizer, "decode", None)
    missing = 0
    for start, end in _runs(mask):
        ok = False
        for position in range(end - 1, start - 1, -1):
            token = int(ids[position])
            if token in special:
                ok = True
                break
            if not (callable(decode) and decode([token]).strip() == ""):
                break  # plain content ends the turn
        missing += not ok
    return missing


def _template_of(tokenizer: Any, chat_template: str | None) -> Any:
    return chat_template if chat_template is not None else getattr(tokenizer, "chat_template", None)


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("every message must be an object with a string 'role'")
        weight = message.get("weight", 1)
        if isinstance(weight, bool) or weight not in (0, 1):
            raise ValueError(f"message weight must be 0 or 1, got {weight!r}")
    return messages


def _selected(messages: list[dict[str, Any]], loss_on: str, train_roles: Sequence[str]) -> list[int]:
    """Indices of the messages whose tokens are trained (``loss_on`` in ``assistant``/``last``)."""
    chosen = [i for i, m in enumerate(messages) if m["role"] in train_roles and m.get("weight", 1) != 0]
    if loss_on == "last":
        last = max((i for i, m in enumerate(messages) if m["role"] in train_roles), default=None)
        chosen = [last] if last is not None and last in chosen else []
    return chosen


def _runs(mask: Sequence[int]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, value in enumerate(list(mask) + [0]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    return runs


def chat_example(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: Any = None,
    mask_method: str = "auto",
    loss_on: str = "assistant",
    train_roles: Sequence[str] = ("assistant",),
    chat_template: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """``(token ids, loss mask)`` of one conversation (see the module docstring)."""
    if loss_on not in LOSS_ON:
        raise ValueError(f"loss_on must be one of {LOSS_ON}, got {loss_on!r}")
    if mask_method not in MASK_METHODS:
        raise ValueError(f"mask_method must be one of {MASK_METHODS}, got {mask_method!r}")
    messages = _validate_messages(messages)
    kwargs = dict(template_kwargs or {})
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    if tools is not None:
        kwargs["tools"] = tools
    encode = _encoder(tokenizer)

    def render(prefix: list[dict[str, Any]], generation_prompt: bool) -> str:
        text = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=generation_prompt, **kwargs)
        if not isinstance(text, str):
            raise ChatTemplateError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, not a string")
        return text

    if mask_method == "auto":
        mask_method = "generation" if has_generation_blocks(_template_of(tokenizer, chat_template)) else "prefix"

    if loss_on == "all":
        try:
            text = render(messages, False)
        except ChatTemplateError:
            raise
        except Exception as exc:
            raise ValueError(f"the chat template rejected the conversation: {exc}") from exc
        ids = encode([text])[0]
        return ids, [1] * len(ids)

    chosen = _selected(messages, loss_on, train_roles)
    if mask_method == "generation":
        if tuple(train_roles) != ("assistant",):
            raise ValueError("mask_method 'generation' marks assistant turns only; use train_roles = ('assistant',)")
        if not has_generation_blocks(_template_of(tokenizer, chat_template)):
            raise ChatTemplateError("mask_method 'generation' needs a chat template with {% generation %} blocks")
        try:
            out = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, add_generation_prompt=False, **kwargs
            )
        except Exception as exc:  # the template rejects this conversation: a data error
            raise ValueError(f"the chat template rejected the conversation: {exc}") from exc
        ids, mask = list(out["input_ids"]), [int(bool(v)) for v in out["assistant_masks"]]
        if len(ids) != len(mask) or (ids and isinstance(ids[0], list)):
            raise ChatTemplateError("expected one flat token sequence with a matching assistant mask")
        assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        runs = _runs(mask)
        # Always checked: Hugging Face stops marking a conversation's spans at the first span whose start
        # character maps to no token, which would otherwise leave later assistant turns silently untrained.
        if len(runs) != len(assistants):
            raise ChatTemplateError(
                f"the template marks {len(runs)} generation spans for {len(assistants)} assistant messages; "
                "the masks need exactly one non-empty {% generation %} block per assistant message"
            )
        if chosen == assistants:
            return ids, mask
        keep = [0] * len(ids)
        for message, (start, end) in zip(assistants, runs):
            if message in chosen:
                keep[start:end] = [1] * (end - start)
        return ids, keep

    try:
        full_text = render(messages, False)
    except ChatTemplateError:
        raise
    except Exception as exc:  # the template rejects this conversation (e.g. role order): a data error
        raise ValueError(f"the chat template rejected the conversation: {exc}") from exc
    texts = [full_text]
    for i in chosen:
        try:
            texts.append(render(messages[:i], messages[i]["role"] == "assistant") if i else "")
            texts.append(render(messages[: i + 1], False))
        except ChatTemplateError:
            raise
        except Exception as exc:  # e.g. a template that cannot render a conversation prefix
            raise ChatTemplateError(f"cannot render the conversation up to message {i}: {exc}") from exc
    for text in texts[1:]:
        if not full_text.startswith(text):
            raise ChatTemplateError(
                "the chat template is not prefix-stable (rendering a conversation prefix does not give a prefix "
                "of the full rendering); use a template with {% generation %} blocks and mask_method 'generation'"
            )
    encoded = encode(texts)
    ids = encoded[0]
    mask = [0] * len(ids)
    for k, _ in enumerate(chosen):
        start_ids, end_ids = encoded[1 + 2 * k], encoded[2 + 2 * k]
        if ids[: len(start_ids)] != start_ids or ids[: len(end_ids)] != end_ids:
            raise ChatTemplateError(
                "tokenizing a conversation prefix does not give a token prefix of the full conversation "
                "(a token merges across a turn boundary); use mask_method 'generation'"
            )
        mask[len(start_ids) : len(end_ids)] = [1] * (len(end_ids) - len(start_ids))
    return ids, mask


def instruction_example(
    encode: Callable[[list[str]], list[list[int]]],
    prompt: str,
    completion: str,
    *,
    eos_id: int | None,
    append_eos: bool = True,
    loss_on: str = "assistant",
) -> tuple[list[int], list[int]]:
    """``(token ids, loss mask)``: prompt and completion encoded separately; the completion (+ EOS) is trained."""
    if loss_on not in LOSS_ON:
        raise ValueError(f"loss_on must be one of {LOSS_ON}, got {loss_on!r}")
    if append_eos and eos_id is None:
        raise ValueError("append_eos needs a tokenizer with an EOS token (or pass --no-append-eos)")
    prompt_ids, completion_ids = encode([prompt, completion]) if prompt else ([], encode([completion])[0])
    completion_ids = list(completion_ids) + ([int(eos_id)] if append_eos else [])
    ids = list(prompt_ids) + completion_ids
    if loss_on == "all":
        return ids, [1] * len(ids)
    return ids, [0] * len(prompt_ids) + [1] * len(completion_ids)


def load_tokenizer(name: str, *, local_files_only: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("SFT tokenization needs the optional 'transformers' package") from exc
    return AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)


def tokenize_sft_jsonl(
    input_paths: Sequence[str | Path],
    output_prefix: str | Path,
    *,
    tokenizer: Any,
    format: str = "chat",
    loss_on: str = "assistant",
    train_roles: Sequence[str] = ("assistant",),
    mask_method: str = "auto",
    chat_template: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
    messages_field: str = "messages",
    tools_field: str = "tools",
    prompt_fields: Sequence[str] = ("prompt",),
    completion_field: str = "completion",
    prompt_separator: str = "\n\n",
    append_eos: bool | None = None,
    dtype: str = "int32",
    max_docs: int | None = None,
    max_tokens: int | None = None,
    on_template_error: str = "raise",
    check_stop_token: bool = True,
) -> SftTokenizeStats:
    """Tokenize every record of ``input_paths`` into ``output_prefix.{bin,idx,mask}``.

    Malformed records (invalid JSON, missing fields, bad messages) and examples without
    a trained token are skipped and counted. A record whose masks the template cannot
    give exactly (:class:`ChatTemplateError`) aborts the run, or with
    ``on_template_error="skip"`` is left out and counted in ``skipped_template_error``.
    With ``check_stop_token`` (chat format, ``loss_on`` other than ``all``) every trained
    turn must end with a special token such as ``<|im_end|>`` (trailing whitespace aside;
    an appended EOS counts), otherwise the record is a :class:`ChatTemplateError`: the
    template leaves the end-of-turn token outside the turn and the model would never
    learn to stop.
    """
    if format not in ("chat", "instruction"):
        raise ValueError(f"format must be 'chat' or 'instruction', got {format!r}")
    if on_template_error not in ("raise", "skip"):
        raise ValueError(f"on_template_error must be 'raise' or 'skip', got {on_template_error!r}")
    if loss_on not in LOSS_ON or mask_method not in MASK_METHODS:
        raise ValueError(f"loss_on must be in {LOSS_ON} and mask_method in {MASK_METHODS}")
    if format == "chat" and not _template_of(tokenizer, chat_template):
        raise ValueError("the tokenizer has no chat template; pass one with --chat-template FILE")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    append_eos = (format == "instruction") if append_eos is None else bool(append_eos)
    if append_eos and eos_id is None:
        raise ValueError("append_eos needs a tokenizer with an EOS token (or pass --no-append-eos)")
    resolved_method = mask_method
    if format == "chat" and mask_method == "auto":
        resolved_method = "generation" if has_generation_blocks(_template_of(tokenizer, chat_template)) else "prefix"
    stats = SftTokenizeStats(format=format, mask_method=resolved_method if format == "chat" else "", eos_id=eos_id if append_eos else None)
    encode = _encoder(tokenizer)
    special = _special_token_ids(tokenizer) if check_stop_token and format == "chat" else set()
    started = time.perf_counter()
    writer = SftWriter(output_prefix, dtype=dtype)

    def done() -> bool:
        return (max_docs is not None and stats.documents >= max_docs) or (max_tokens is not None and stats.tokens >= max_tokens)

    def example(record: dict[str, Any]) -> tuple[list[int], list[int]] | None:
        if format == "chat":
            ids, mask = chat_example(
                tokenizer,
                record.get(messages_field),
                tools=record.get(tools_field),
                mask_method=resolved_method,
                loss_on=loss_on,
                train_roles=train_roles,
                chat_template=chat_template,
                template_kwargs=template_kwargs,
            )
            if append_eos:
                ids, mask = ids + [int(eos_id)], mask + [mask[-1] if mask else 0]
            if check_stop_token and loss_on != "all" and turns_without_stop_token(ids, mask, tokenizer, special=special):
                raise ChatTemplateError(
                    "a trained turn does not end with an end-of-turn (special) token: the chat template puts it "
                    "outside the turn, so the model would never learn to stop; use a template whose turns end "
                    "with their end-of-turn token (--chat-template), or pass --no-check-stop-token"
                )
            return ids, mask
        parts = [record.get(name, "") for name in prompt_fields]
        completion = record.get(completion_field)
        if not all(isinstance(part, str) for part in parts) or not isinstance(completion, str):
            raise ValueError("prompt and completion fields must be strings")
        if not completion:
            return None
        prompt = prompt_separator.join(part for part in parts if part)
        return instruction_example(encode, prompt, completion, eos_id=eos_id, append_eos=append_eos, loss_on=loss_on)

    try:
        for path in input_paths:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if done():
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats.skipped_invalid_json += 1
                        continue
                    if not isinstance(record, dict):
                        stats.skipped_invalid_record += 1
                        continue
                    try:
                        result = example(record)
                    except ChatTemplateError:
                        if on_template_error == "raise":
                            raise
                        stats.skipped_template_error += 1
                        continue
                    except (ValueError, TypeError, KeyError):
                        stats.skipped_invalid_record += 1
                        continue
                    if result is None or not result[0]:
                        stats.skipped_empty += 1
                        continue
                    ids, mask = result
                    if not any(mask[1:]):  # the first token is never a target
                        stats.skipped_no_target += 1
                        continue
                    writer.add(ids, mask)
                    stats.documents += 1
                    stats.tokens += len(ids)
                    stats.trainable_tokens += int(sum(mask[1:]))
        if stats.documents == 0:
            raise ValueError(f"no usable {format} records in {list(map(str, input_paths))}")
        writer.commit()
    except BaseException:
        writer.abort()
        raise
    stats.seconds = round(time.perf_counter() - started, 3)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jla tokenize --format chat|instruction",
        description="Tokenize chat or instruction JSONL into an SFT corpus (CIDX + loss-mask sidecar).",
    )
    parser.add_argument("--format", required=True, choices=["chat", "instruction"])
    parser.add_argument("--input", required=True, nargs="+", help="JSONL file(s)")
    parser.add_argument("--output", required=True, help="output prefix (writes PREFIX.bin, PREFIX.idx, PREFIX.mask)")
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer name or local path")
    parser.add_argument("--loss-on", default="assistant", choices=list(LOSS_ON),
                        help="assistant: trained-role turns (instruction: the completion); last: the last such turn; all: every token")
    parser.add_argument("--train-roles", nargs="+", default=["assistant"], help="chat roles whose turns are trained")
    parser.add_argument("--mask-method", default="auto", choices=list(MASK_METHODS))
    parser.add_argument("--chat-template", default=None, help="Jinja file overriding the tokenizer's chat template")
    parser.add_argument("--template-kwargs", default=None, help="JSON object passed to apply_chat_template")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--tools-field", default="tools")
    parser.add_argument("--prompt-field", nargs="+", default=["prompt"], help="instruction prompt field(s)")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--prompt-separator", default="\n\n")
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None,
                        help="append the tokenizer's EOS (default: instruction yes, chat no)")
    parser.add_argument("--dtype", default="int32", choices=["uint16", "int32"])
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--on-template-error", default="raise", choices=["raise", "skip"],
                        help="a record the template cannot mask exactly: stop (default) or skip and count it")
    parser.add_argument("--check-stop-token", action=argparse.BooleanOptionalAction, default=True,
                        help="require every trained chat turn to end with a special end-of-turn token (default on)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=args.local_files_only)
    template = Path(args.chat_template).read_text(encoding="utf-8") if args.chat_template else None
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else None
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        raise ValueError("--template-kwargs must be a JSON object")
    stats = tokenize_sft_jsonl(
        args.input,
        args.output,
        tokenizer=tokenizer,
        format=args.format,
        loss_on=args.loss_on,
        train_roles=tuple(args.train_roles),
        mask_method=args.mask_method,
        chat_template=template,
        template_kwargs=template_kwargs,
        messages_field=args.messages_field,
        tools_field=args.tools_field,
        prompt_fields=tuple(args.prompt_field),
        completion_field=args.completion_field,
        prompt_separator=args.prompt_separator,
        append_eos=args.append_eos,
        dtype=args.dtype,
        max_docs=args.max_docs,
        max_tokens=args.max_tokens,
        on_template_error=args.on_template_error,
        check_stop_token=args.check_stop_token,
    )
    print(json.dumps({"output": args.output, **asdict(stats)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
