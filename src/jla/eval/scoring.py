# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-token log-likelihoods from the full forward pass (perplexity and lm-eval scoring).

:func:`token_logprobs` never materializes ``[B, T, V]`` logits: the LM head runs
over chunks of at most ``chunk_rows`` token rows. :func:`score_sequences` batches
ragged sequences by length and right-pads them to a multiple of ``bucket``; the
model is causal and :func:`score_sequences` routes MoE layers dropless
(:func:`jla.inference.dropless_moe`), so neither the padding nor the other rows of a
batch change any scored position.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jla.inference.loading import dropless_moe
from jla.inference.rwkv7 import decoding_model

__all__ = ["score_sequences", "token_logprobs"]

Array = jax.Array


@eqx.filter_jit
def token_logprobs(model, input_ids: Array, positions: Array | None = None, segment_ids: Array | None = None,
                   *, chunk_rows: int = 1024) -> tuple[Array, Array]:
    """``[B, T - 1]`` float32 ``log p(input_ids[:, t + 1] | input_ids[:, : t + 1])`` and argmax flags.

    The flag is True where the target is the most likely token (``argmax``, first
    index on ties), which is lm-eval's ``is_greedy``.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError(f"input_ids must be [B, T] with T >= 2; got {input_ids.shape}")
    hidden = model(input_ids, positions=positions, segment_ids=segment_ids, inference=True).hidden[:, :-1]
    targets = input_ids[:, 1:].astype(jnp.int32)
    batch, length, dim = hidden.shape
    rows = batch * length
    chunk = min(chunk_rows, rows)
    chunks = -(-rows // chunk)
    pad = chunks * chunk - rows
    h = jnp.pad(hidden.reshape(rows, dim), ((0, pad), (0, 0))).reshape(chunks, chunk, dim)
    y = jnp.pad(targets.reshape(rows), (0, pad)).reshape(chunks, chunk)

    def one(args):
        hc, yc = args
        logits = model.logits(hc)  # [chunk, V] float32
        picked = jnp.take_along_axis(logits, yc[:, None], axis=-1)[:, 0]
        return picked - jax.nn.logsumexp(logits, axis=-1), jnp.argmax(logits, axis=-1) == yc

    logp, greedy = lax.map(one, (h, y))
    return logp.reshape(-1)[:rows].reshape(batch, length), greedy.reshape(-1)[:rows].reshape(batch, length)


def score_sequences(model, sequences: Sequence[Sequence[int]], *, batch_size: int = 8, bucket: int = 64,
                    pad_id: int = 0, chunk_rows: int = 1024) -> list[tuple[np.ndarray, np.ndarray]]:
    """For every sequence ``s`` (>= 2 tokens): ``log p(s[t + 1] | s[: t + 1])`` and argmax flags, ``[len(s) - 1]``."""
    if batch_size < 1 or bucket < 1:
        raise ValueError("batch_size and bucket must be >= 1")
    model = dropless_moe(decoding_model(model))
    vocab = model.config.vocab_size
    seqs = [np.asarray(s, np.int64) for s in sequences]
    for s in seqs:
        if s.ndim != 1 or s.size < 2:
            raise ValueError("every scored sequence needs at least 2 tokens")
        if s.min() < 0 or s.max() >= vocab:  # an out-of-range id would gather NaN embeddings
            raise ValueError(f"token ids must be in [0, {vocab}); got {s.min()}..{s.max()}")
    if not 0 <= pad_id < vocab:
        raise ValueError(f"pad_id must be in [0, {vocab}); got {pad_id}")
    order = sorted(range(len(seqs)), key=lambda i: -seqs[i].size)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(seqs)
    for begin in range(0, len(order), batch_size):
        group = order[begin : begin + batch_size]
        width = -(-max(seqs[i].size for i in group) // bucket) * bucket
        ids = np.full((batch_size, width), pad_id, np.int32)  # a fixed batch keeps one compiled shape
        for row, i in enumerate(group):
            ids[row, : seqs[i].size] = seqs[i]
        logp, greedy = jax.device_get(token_logprobs(model, jnp.asarray(ids), chunk_rows=chunk_rows))
        for row, i in enumerate(group):
            n = seqs[i].size - 1
            results[i] = (np.asarray(logp[row, :n], np.float64), np.asarray(greedy[row, :n], bool))
    return results  # type: ignore[return-value]
