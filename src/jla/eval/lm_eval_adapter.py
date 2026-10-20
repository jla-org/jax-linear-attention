# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""lm-evaluation-harness model: ``JLALM`` (needs the optional ``lm-eval`` package).

    lm = JLALM(model, tokenizer, batch_size=8, max_length=2048)
    lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"])

or, in a Python process that imported this module, ``simple_evaluate(model="jla",
model_args="run_dir=...,tokenizer=...")`` (a string or a dict; also ``config=``,
``checkpoint=``, ``preset=``, ``step=``, ``init_seed=``, ``compute_dtype=``, ``batch_size=``,
``max_length=``, ``max_gen_toks=``, ``seed=`` (sampling), ``bucket=``). The ``lm_eval``
shell command runs in its own process and does not know ``jla``: use ``jla eval lm-eval``.

Models with MoE capacity routing are switched to dropless routing
(:func:`jla.inference.dropless_moe`), so no score depends on the batch or its padding.

* ``loglikelihood``: lm-eval's ``TemplateLM`` encoding (trailing context spaces move
  to the continuation; an empty context is the prefix token), then the full forward
  pass over ``(context + continuation)[-(max_length + 1):]``, summing the
  continuation's log-probabilities; ``is_greedy`` is True when every continuation
  token is the argmax.
* ``loglikelihood_rolling``: lm-eval's disjoint rolling windows of ``max_length``
  tokens after the prefix token (each token is scored once).
* ``generate_until``: :func:`jla.inference.generate`. Generation options go through
  lm-eval's ``normalize_gen_kwargs`` as in ``HFLM``: the ``max_new_tokens`` /
  ``max_tokens`` / ``max_completion_tokens`` aliases of ``max_gen_toks``, and
  ``do_sample`` unset means sampling exactly when ``temperature > 0``. Sampling uses
  ``temperature`` (default 1), ``top_k`` (default 50, as Hugging Face ``generate``)
  and ``top_p``; ``num_beams`` must be 1 and any other option fails loudly. Rows stop
  at the EOS token, at ``max_gen_toks`` or at the first ``until`` string (the EOS text
  included, as ``handle_stop_sequences`` does), which is cut off; completions are
  decoded without special tokens. Contexts keep their last ``max_length -
  max_gen_toks`` tokens; a request whose ``max_gen_toks`` is not below ``max_length``
  fails (as in lm-eval's ``HFLM``). Scoring requests do not depend on ``max_gen_toks``.

The prefix token is the tokenizer's BOS token when it has one, else EOS. Every
token id must be below the model's vocabulary size (checked for the tokenizer
up front). RWKV-7 models are wrapped by :func:`jla.inference.rwkv7.decoding_model`.
"""

from __future__ import annotations

from typing import Any

try:
    import lm_eval.models  # noqa: F401  (registers lm-eval's own models first; see get_model)
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model
    from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
except ModuleNotFoundError as exc:
    if exc.name != "lm_eval":
        raise
    raise ModuleNotFoundError(
        "jla.eval.lm_eval_adapter needs the optional dependency: pip install 'jax-linear-attention[eval]'",
        name="lm_eval",
    ) from exc

from jla.eval.scoring import score_sequences
from jla.inference.generate import generate
from jla.inference.loading import dropless_moe
from jla.inference.rwkv7 import decoding_model
from jla.inference.sampling import SamplingConfig

__all__ = ["JLALM"]

_GEN_KEYS = {"until", "max_gen_toks", "do_sample", "temperature", "top_k", "top_p", "num_beams"}
_HF_DEFAULT_TOP_K = 50  # transformers' GenerationConfig default when sampling


@register_model("jla")
class JLALM(TemplateLM):
    def __init__(self, model, tokenizer, *, batch_size: int = 8, max_length: int = 2048, max_gen_toks: int = 256,
                 seed: int = 0, bucket: int = 64) -> None:
        super().__init__()
        if tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer needs an EOS token")
        self.model = dropless_moe(decoding_model(model))
        vocab = self.model.config.vocab_size
        size = len(tokenizer) if hasattr(tokenizer, "__len__") else getattr(tokenizer, "vocab_size", None)
        if size is not None and int(size) > vocab:
            raise ValueError(f"the tokenizer has {int(size)} tokens but the model's vocabulary only {vocab}")
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self._max_length = int(max_length)
        self.max_gen_toks = int(max_gen_toks)
        self.seed = int(seed)
        self.bucket = int(bucket)
        if min(self.batch_size, self._max_length, self.max_gen_toks, self.bucket) < 1:
            raise ValueError("batch_size, max_length, max_gen_toks and bucket must be >= 1")

    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config: dict | None = None) -> JLALM:
        """``simple_evaluate(model="jla", model_args="key=value,...")``."""
        from lm_eval.utils import simple_parse_args_string

        return cls._from_model_args(simple_parse_args_string(arg_string), additional_config)

    @classmethod
    def create_from_arg_obj(cls, arg_dict: dict[str, Any], additional_config: dict | None = None) -> JLALM:
        """``simple_evaluate(model="jla", model_args={...})``."""
        return cls._from_model_args(dict(arg_dict), additional_config)

    @classmethod
    def _from_model_args(cls, args: dict[str, Any], additional_config: dict | None) -> JLALM:
        from jla.inference.loading import load_model, load_tokenizer

        args = dict(args)
        extra = {k: v for k, v in (additional_config or {}).items() if v is not None}
        batch_size = extra.pop("batch_size", None)  # simple_evaluate's batch_size
        if batch_size is not None:
            if isinstance(batch_size, bool) or not str(batch_size).isdigit():
                raise ValueError(f"jla needs an integer batch_size; got {batch_size!r} ('auto' is not supported)")
            if "batch_size" in args and int(args["batch_size"]) != int(batch_size):
                raise ValueError(f"batch_size {batch_size} conflicts with model_args batch_size={args['batch_size']}")
            args["batch_size"] = int(batch_size)
        if extra:  # max_batch_size (for batch_size='auto') and device (JAX selects devices itself)
            raise ValueError(f"unsupported options for jla: {sorted(extra)} (select devices with JAX_PLATFORMS "
                             "or CUDA_VISIBLE_DEVICES)")
        sources = ("run_dir", "config", "checkpoint", "preset", "step", "init_seed", "compute_dtype")
        options = ("batch_size", "max_length", "max_gen_toks", "seed", "bucket")
        unknown = set(args) - {"tokenizer", *sources, *options}
        if unknown:
            raise ValueError(f"unknown model_args for jla: {sorted(unknown)}")
        if "tokenizer" not in args:
            raise ValueError("model_args needs tokenizer=<name or path>")
        tokenizer = load_tokenizer(str(args["tokenizer"]))
        source = {k: str(args[k]) for k in ("run_dir", "config", "checkpoint", "preset") if k in args}
        model = load_model(**source, step=None if "step" not in args else int(args["step"]),
                           seed=int(args.get("init_seed", 0)), compute_dtype=args.get("compute_dtype"))
        return cls(model, tokenizer, **{k: int(args[k]) for k in options if k in args})

    # TemplateLM hooks ---------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def prefix_token_id(self) -> int:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        return self.eot_token_id if bos is None else int(bos)

    @property
    def max_length(self) -> int:
        return self._max_length

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        return list(self.tokenizer.encode(string, add_special_tokens=bool(add_special_tokens)))

    def tok_decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs) -> list[tuple[float, bool]]:
        sequences, counts = [], []
        for _, context_enc, continuation_enc in requests:
            if not continuation_enc:
                raise ValueError("empty continuation")
            if len(continuation_enc) > self._max_length:
                raise ValueError(f"continuation of {len(continuation_enc)} tokens exceeds max_length {self._max_length}")
            sequences.append((list(context_enc) + list(continuation_enc))[-(self._max_length + 1):])
            counts.append(len(continuation_enc))
        scores = score_sequences(self.model, sequences, batch_size=self.batch_size, bucket=self.bucket,
                                 pad_id=self.eot_token_id)
        results = []
        for (key, _, _), (logp, greedy), n in zip(requests, scores, counts):
            answer = (float(logp[-n:].sum()), bool(greedy[-n:].all()))
            if key is not None:
                self.cache_hook.add_partial("loglikelihood", key, answer)
            results.append(answer)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results = []
        for request in requests:
            (string,) = request.args
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.tok_encode(string), prefix_token=self.prefix_token_id,
                max_seq_len=self._max_length, context_len=1)]
            scores = self._loglikelihood_tokens([(None, ctx, cont) for ctx, cont in windows]) if windows else []
            total = float(sum(logp for logp, _ in scores))
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total)
            results.append(total)
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        results: list[str | None] = [None] * len(requests)
        groups: dict[tuple, list[int]] = {}
        for i, request in enumerate(requests):
            groups.setdefault(self._generation_key(request.args[1]), []).append(i)
        for key, indices in groups.items():
            options = dict(key)
            until, max_gen = list(options["until"]), options["max_gen_toks"]
            sampling = SamplingConfig(temperature=options["temperature"], top_k=options["top_k"],
                                      top_p=options["top_p"])
            if max_gen >= self._max_length:
                raise ValueError(f"max_gen_toks {max_gen} must be below max_length {self._max_length}")
            keep = self._max_length - max_gen
            order = sorted(indices, key=lambda i: -len(self.tok_encode(requests[i].args[0])))
            for begin in range(0, len(order), self.batch_size):
                batch = order[begin : begin + self.batch_size]
                prompts = [(self.tok_encode(requests[i].args[0]) or [self.prefix_token_id])[-keep:] for i in batch]
                out = generate(self.model, prompts, max_new_tokens=max_gen, sampling=sampling, seed=self.seed,
                               eos_ids=(self.eot_token_id,), pad_id=self.eot_token_id, row_ids=batch,
                               bucket=self.bucket)
                for row, i in enumerate(batch):
                    tokens = [t for t in out.row(row) if t != self.eot_token_id]
                    text = self.tok_decode(tokens)
                    for stop in until:
                        if stop:
                            text = text.split(stop)[0]
                    self.cache_hook.add_partial("generate_until", requests[i].args, text)
                    results[i] = text
        return results  # type: ignore[return-value]

    def _generation_key(self, gen_kwargs: dict) -> tuple:
        given = dict(gen_kwargs)
        kwargs = dict(normalize_gen_kwargs(given, self.max_gen_toks))
        unknown = set(kwargs) - _GEN_KEYS
        if unknown:
            raise ValueError(f"unsupported generation options: {sorted(unknown)}")
        if int(kwargs.get("num_beams") or 1) != 1:
            raise ValueError(f"beam search is not supported (num_beams={kwargs['num_beams']})")
        until = handle_stop_sequences(list(kwargs["until"]), eos=self.tok_decode([self.eot_token_id],
                                                                                 skip_special_tokens=False))
        temperature = 0.0
        if kwargs["do_sample"]:  # HF generate: temperature defaults to 1 when sampling
            temperature = float(given["temperature"]) if given.get("temperature") is not None else 1.0
        if temperature == 0.0:  # greedy: the filters do not matter
            top_k, top_p = 0, 1.0
        else:
            top_k = _HF_DEFAULT_TOP_K if kwargs.get("top_k") is None else int(kwargs["top_k"])
            top_p = 1.0 if kwargs.get("top_p") is None else float(kwargs["top_p"])
        return (("until", tuple(until)), ("max_gen_toks", int(kwargs["max_gen_toks"])),
                ("temperature", temperature), ("top_k", top_k), ("top_p", top_p))
