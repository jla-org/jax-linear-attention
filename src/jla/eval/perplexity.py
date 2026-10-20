# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perplexity of a token stream (a tokenized text file or a CIDX corpus).

The stream is cut into windows of ``seq_len`` tokens that score every token after
the first exactly once. The first window scores all of its targets; each later
window scores the next ``stride`` tokens (default ``seq_len - 1``: windows overlap
by one token) and fills the rest of its ``seq_len`` inputs with the preceding
tokens as context, so a smaller stride gives every scored token more context
(Hugging Face's strided perplexity). With ``prefix_id`` (for example BOS or EOS)
prepended, the first token is scored too.

With ``document_eos`` the stream is treated like document-masked training data
(``data.document_masking``): inside a window, a new document starts after every
EOS token, positions restart at 0 and no token attends to or carries state from
another document, and the targets at EOS positions (the next document's first
token) are not scored.

MoE layers route dropless (:func:`jla.inference.dropless_moe`), so the result does
not depend on ``batch_size`` or on the padding of the last batch.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from jla.data.packing import positions_from_segment_ids, segment_ids_from_eos
from jla.eval.scoring import token_logprobs
from jla.inference.loading import dropless_moe
from jla.inference.rwkv7 import decoding_model

__all__ = ["PerplexityResult", "perplexity", "stream_windows"]


@dataclasses.dataclass(frozen=True)
class PerplexityResult:
    nll_sum: float  # natural-log negative log-likelihood summed over scored tokens
    tokens: int  # scored tokens
    windows: int

    @property
    def nll_per_token(self) -> float:
        return self.nll_sum / self.tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll_per_token)

    @property
    def bits_per_token(self) -> float:
        return self.nll_per_token / math.log(2.0)

    def as_dict(self) -> dict[str, float | int]:
        return {"nll_sum": self.nll_sum, "tokens": self.tokens, "windows": self.windows,
                "nll_per_token": self.nll_per_token, "perplexity": self.perplexity,
                "bits_per_token": self.bits_per_token}


def stream_windows(num_tokens: int, seq_len: int, stride: int | None = None) -> list[tuple[int, int, int]]:
    """``(start, end, first_target)``: window ``stream[start:end]`` scores targets ``first_target .. end - 1``."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2; got {seq_len}")
    stride = seq_len - 1 if stride is None else stride
    if not 1 <= stride <= seq_len - 1:
        raise ValueError(f"stride must be in [1, seq_len - 1] = [1, {seq_len - 1}]; got {stride}")
    windows = []
    scored = 1  # stream[0] has no context
    while scored < num_tokens:
        end = min(num_tokens, seq_len if not windows else scored + stride)
        windows.append((max(0, end - seq_len), end, scored))
        scored = end
    return windows


def perplexity(
    model,
    tokens,
    *,
    seq_len: int,
    stride: int | None = None,
    batch_size: int = 8,
    prefix_id: int | None = None,
    document_eos: int | None = None,
    pad_id: int = 0,
    chunk_rows: int = 1024,
) -> PerplexityResult:
    """Perplexity of the 1-D token stream ``tokens`` (see the module docstring)."""
    model = dropless_moe(decoding_model(model))
    stream = np.asarray(tokens, np.int64).reshape(-1)
    if prefix_id is not None:
        stream = np.concatenate([np.asarray([prefix_id], np.int64), stream])
    if stream.size < 2:
        raise ValueError("perplexity needs at least 2 tokens (or 1 token and prefix_id)")
    if stream.min() < 0 or stream.max() >= model.config.vocab_size:
        raise ValueError(f"token ids must be in [0, {model.config.vocab_size})")
    windows = stream_windows(stream.size, seq_len, stride)
    nll = 0.0
    count = 0
    for begin in range(0, len(windows), batch_size):
        group = windows[begin : begin + batch_size]
        ids = np.full((batch_size, seq_len), pad_id, np.int32)  # one compiled shape for every batch
        scored = np.zeros((batch_size, seq_len - 1), bool)
        for row, (start, end, first) in enumerate(group):
            ids[row, : end - start] = stream[start:end]
            scored[row, first - start - 1 : end - start - 1] = True
        positions = segment_ids = None
        if document_eos is not None:
            seg = segment_ids_from_eos(ids, document_eos)
            segment_ids, positions = jnp.asarray(seg), jnp.asarray(positions_from_segment_ids(seg))
            scored &= ids[:, :-1] != document_eos
        logp, _ = jax.device_get(token_logprobs(model, jnp.asarray(ids), positions, segment_ids, chunk_rows=chunk_rows))
        nll -= float(np.asarray(logp, np.float64)[scored].sum())
        count += int(scored.sum())
    if count == 0:
        raise ValueError("no token was scored (every target follows an EOS?)")
    return PerplexityResult(nll_sum=nll, tokens=count, windows=len(windows))
