# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched autoregressive generation with the model's decode caches.

    result = generate(model, [[5, 6, 7], [8, 9]], max_new_tokens=32, eos_ids=(2,),
                      sampling=SamplingConfig(temperature=0.8, top_p=0.95), seed=0)
    result.row(0)        # the new tokens of prompt 0 (its EOS included when one was produced)

Prompts may have different lengths. All rows advance in lockstep over positions:
the shared prefix (the shortest prompt, rounded down to a multiple of ``bucket``) is
prefilled by :meth:`LanguageModel.decode_step` calls over chunks of at most
``prefill_chunk`` tokens (a ``lax.scan``; their logits are unused, so the compiled loop
never holds ``[B, prefill, V]`` logits), then
a jit-compiled ``lax.while_loop`` decodes one position per iteration, feeding each
row its own prompt token while ``position < length`` and its sampled token after
that. Chunking bounds the prefill memory of cached softmax attention, which scores a
chunk of queries against the whole cache (``[B, H, chunk, cache]`` scores): with one
chunk per prompt that buffer grows with the square of the prompt length. No padding
token ever reaches a KV cache, ring cache, convolution state or recurrent state, so
every layer family (softmax attention, linear attention, SSMs and hybrids) gives each
row the result of generating it alone (up to floating-point differences between batch
shapes).

Layer kinds without a decode cache (NSA, DSA, log-linear Mamba-2) and
``use_cache=False`` run the same loop by recomputing the full forward pass over the
sequence at every step. The standalone RWKV-7 model (:class:`jla.models.rwkv7.RWKV7LM`)
is wrapped by :func:`jla.inference.rwkv7.decoding_model`.

Sampling keys depend only on ``(seed, row id, position)``, so a prompt samples the
same tokens whatever batch it is in (pass ``row_ids`` to keep ids stable across calls).
MoE layers route dropless (:func:`jla.inference.dropless_moe`), as decoding always does.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jla.inference.loading import dropless_moe
from jla.inference.rwkv7 import decoding_model
from jla.inference.sampling import GREEDY, SamplingConfig, sample_token

__all__ = ["GenerationResult", "default_prefill_chunk", "generate", "prompts_from_padded", "supports_cache"]

Array = jax.Array
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    tokens: np.ndarray  # [B, max_new_tokens] int32; pad_id after each row's last token
    lengths: np.ndarray  # [B] int32 new tokens per row (an EOS that stopped the row is included)
    stopped: np.ndarray  # [B] bool: the row ended on one of eos_ids

    def row(self, index: int) -> list[int]:
        return self.tokens[index, : int(self.lengths[index])].tolist()


def prompts_from_padded(input_ids, attention_mask) -> list[list[int]]:
    """Rows of a left- or right-padded batch (``attention_mask`` 1 = token) as ragged lists.

    Each row's tokens must be one contiguous run; padding may sit on either side.
    """
    ids = np.asarray(input_ids)
    mask = np.asarray(attention_mask).astype(bool)
    if ids.ndim != 2 or ids.shape != mask.shape:
        raise ValueError(f"input_ids and attention_mask must both be [B, T]; got {ids.shape}, {mask.shape}")
    rows = []
    for b in range(ids.shape[0]):
        where = np.flatnonzero(mask[b])
        if where.size == 0:
            raise ValueError(f"row {b} has no tokens")
        if where[-1] - where[0] + 1 != where.size:
            raise ValueError(f"row {b}: attention_mask must be one contiguous run (left or right padding)")
        rows.append(ids[b, where[0] : where[-1] + 1].astype(np.int64).tolist())
    return rows


def supports_cache(model) -> bool:
    """Whether every layer of ``model`` can decode from a cache (NSA, DSA and log-linear Mamba-2 cannot).

    Traces (without running) a one-token decode step of the architecture once per
    ``model.config``. Layers without decoding raise ``NotImplementedError`` there
    (log-linear Mamba-2 a ``ValueError``, accepted only for that layer); the
    fallback to full-forward recomputation is logged as a warning. Any other error
    propagates.
    """
    model = decoding_model(model)
    known = _SUPPORTS_CACHE.get(model.config)
    if known is not None:
        return known

    def step(m):
        ids = jnp.zeros((1, 1), jnp.int32)
        return m.decode_step(m.init_cache(1, 2), ids, ids)

    try:
        eqx.filter_eval_shape(step, model)
        known = True
    except NotImplementedError as exc:
        reason, known = exc, False
    except ValueError as exc:
        if not _has_log_linear_mamba2(model.config):
            raise
        reason, known = exc, False
    if not known:
        logger.warning("no cached decoding (%s): generation recomputes the full forward pass every step", reason)
    _SUPPORTS_CACHE[model.config] = known
    return known


def _has_log_linear_mamba2(config) -> bool:
    ssm = getattr(config, "ssm", None)
    return ssm is not None and ssm.log_linear_levels > 0 and "mamba2" in config.resolved_layer_types


_SUPPORTS_CACHE: dict = {}  # ModelConfig -> bool (decode support depends on the architecture only)

# Default prefill chunk: at most this many [batch, chunk, cache length] attention scores per head.
_PREFILL_SCORES = 1 << 24


def default_prefill_chunk(batch: int, total: int) -> int:
    """A power of two in [16, 1024] with ``batch * chunk * total <= 2**24`` where possible.

    With ``H`` query heads in float32 the cached-attention scores of one chunk then take at
    most ``64 H`` MiB (768 MiB for 12 heads), whatever the prompt length.
    """
    fit = max(1, _PREFILL_SCORES // max(1, batch * total))
    return int(min(1024, max(16, 1 << (fit.bit_length() - 1))))


def generate(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    sampling: SamplingConfig = GREEDY,
    seed: int = 0,
    eos_ids: Sequence[int] = (),
    pad_id: int = 0,
    row_ids: Sequence[int] | None = None,
    use_cache: bool | None = None,
    bucket: int = 16,
    prefill_chunk: int | None = None,
) -> GenerationResult:
    """Generate up to ``max_new_tokens`` tokens after each prompt (ragged lists of token ids).

    A row stops at the first token in ``eos_ids`` (kept in its output) or after
    ``max_new_tokens`` tokens. ``use_cache=None`` uses the decode caches when every
    layer has one. ``bucket`` rounds the padded prompt width up and the prefill
    length down to multiples of itself, which bounds the number of compiled shapes
    (prompt tokens past the prefill are fed inside the decode loop). ``prefill_chunk``
    is the most tokens one prefill ``decode_step`` takes (default:
    :func:`default_prefill_chunk`); every chunk size gives the same tokens.
    """
    model = dropless_moe(decoding_model(model))
    rows = [[int(t) for t in p] for p in prompts]
    if not rows:
        raise ValueError("no prompts")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")
    if bucket < 1:
        raise ValueError(f"bucket must be >= 1; got {bucket}")
    if prefill_chunk is not None and prefill_chunk < 1:
        raise ValueError(f"prefill_chunk must be >= 1; got {prefill_chunk}")
    vocab = model.config.vocab_size
    for ids in (*rows, list(eos_ids), [pad_id]):
        if any(t < 0 or t >= vocab for t in ids):
            raise ValueError(f"token ids must be in [0, {vocab}); got {ids}")
    lengths = np.array([len(r) for r in rows], np.int32)
    if lengths.min() < 1:
        raise ValueError("every prompt needs at least one token (e.g. a BOS token)")
    ids = list(range(len(rows))) if row_ids is None else [int(i) for i in row_ids]
    if len(ids) != len(rows):
        raise ValueError(f"row_ids has {len(ids)} entries for {len(rows)} prompts")

    width = -(-int(lengths.max()) // bucket) * bucket
    prefill = max(1, int(lengths.min()) // bucket * bucket) if bucket > 1 else int(lengths.min())
    total = width + max_new_tokens
    chunk = default_prefill_chunk(len(rows), total) if prefill_chunk is None else int(prefill_chunk)
    prompt = np.full((len(rows), width), pad_id, np.int32)
    for b, r in enumerate(rows):
        prompt[b, : len(r)] = r
    base = jax.random.key(seed)
    row_keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.asarray(ids, jnp.uint32))
    eos = jnp.asarray(list(eos_ids), jnp.int32).reshape(-1)
    if use_cache is None:
        use_cache = supports_cache(model)
    common = dict(prefill=prefill, total=total, max_new=max_new_tokens, sampling=sampling, pad_id=pad_id)
    if use_cache:
        out, n_new, stopped = _cached_loop(model, jnp.asarray(prompt), jnp.asarray(lengths), row_keys, eos,
                                           chunk=min(chunk, max(1, prefill - 1)), **common)
    else:
        out, n_new, stopped = _recompute_loop(model, jnp.asarray(prompt), jnp.asarray(lengths), row_keys, eos,
                                              **common)
    return GenerationResult(tokens=np.asarray(out), lengths=np.asarray(n_new), stopped=np.asarray(stopped))


class _Progress(eqx.Module):
    out: Array  # [B, max_new] int32
    n_new: Array  # [B] int32
    done: Array  # [B] bool
    stopped: Array  # [B] bool


def _choose(t: Array, logits: Array, prompt: Array, lengths: Array, row_keys: Array,
            sampling: SamplingConfig) -> tuple[Array, Array]:
    """The token at position ``t`` of every row: its prompt token, or a sample from ``logits [B, V]``."""
    keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(row_keys, t)
    sampled = jax.vmap(lambda lg, k: sample_token(lg, k, sampling))(logits, keys)
    column = lax.dynamic_index_in_dim(prompt, jnp.minimum(t, prompt.shape[1] - 1), axis=1, keepdims=False)
    forced = t < lengths
    return jnp.where(forced, column, sampled).astype(jnp.int32), forced


def _record(p: _Progress, token: Array, forced: Array, eos: Array, max_new: int) -> _Progress:
    new = ~forced & ~p.done
    rows = jnp.arange(token.shape[0])
    out = p.out.at[rows, jnp.where(new, p.n_new, max_new)].set(token, mode="drop")
    n_new = p.n_new + new.astype(jnp.int32)
    hit = new & jnp.any(token[:, None] == eos[None, :], axis=-1)
    return _Progress(out=out, n_new=n_new, done=p.done | hit | (n_new >= max_new), stopped=p.stopped | hit)


def _start(batch: int, max_new: int, pad_id: int) -> _Progress:
    return _Progress(out=jnp.full((batch, max_new), pad_id, jnp.int32), n_new=jnp.zeros((batch,), jnp.int32),
                     done=jnp.zeros((batch,), bool), stopped=jnp.zeros((batch,), bool))


def _prefill(model, cache, prompt, length: int, chunk: int):
    """Feed ``prompt[:, :length]`` through ``decode_step`` in chunks of ``chunk`` tokens (logits unused)."""
    batch = prompt.shape[0]

    def run(cache, begin, size: int):
        ids = lax.dynamic_slice_in_dim(prompt, begin, size, axis=1)
        positions = jnp.broadcast_to(begin + jnp.arange(size, dtype=jnp.int32), (batch, size))
        return model.decode_step(cache, ids, positions)[1]  # the [B, size, V] logits are dead code

    full, rest = divmod(length, chunk)
    if full == 1:
        cache = run(cache, jnp.int32(0), chunk)
    elif full > 1:
        cache, _ = lax.scan(lambda c, i: (run(c, i * chunk, chunk), None), cache, jnp.arange(full, dtype=jnp.int32))
    if rest:
        cache = run(cache, jnp.int32(full * chunk), rest)
    return cache


@eqx.filter_jit
def _cached_loop(model, prompt, lengths, row_keys, eos, *, prefill: int, total: int, max_new: int,
                 sampling: SamplingConfig, pad_id: int, chunk: int):
    batch = prompt.shape[0]
    cache = model.init_cache(batch, total)
    if prefill > 1:  # all but the last prefill token, in chunks
        cache = _prefill(model, cache, prompt, prefill - 1, chunk)
    last = jnp.full((batch, 1), prefill - 1, jnp.int32)
    logits, cache = model.decode_step(cache, prompt[:, prefill - 1 : prefill], last)

    def cond(state):
        t, _, _, progress = state
        return (t < total) & ~jnp.all(progress.done)

    def body(state):
        t, last, cache, progress = state
        token, forced = _choose(t, last, prompt, lengths, row_keys, sampling)
        progress = _record(progress, token, forced, eos, max_new)
        logits, cache = model.decode_step(cache, token[:, None], jnp.full((batch, 1), t, jnp.int32))
        return t + 1, logits[:, -1], cache, progress

    state = (jnp.asarray(prefill, jnp.int32), logits[:, -1], cache, _start(batch, max_new, pad_id))
    _, _, _, progress = lax.while_loop(cond, body, state)
    return progress.out, progress.n_new, progress.stopped


@eqx.filter_jit
def _recompute_loop(model, prompt, lengths, row_keys, eos, *, prefill: int, total: int, max_new: int,
                    sampling: SamplingConfig, pad_id: int):
    batch, width = prompt.shape
    seq = jnp.concatenate([prompt, jnp.full((batch, total - width), pad_id, jnp.int32)], axis=1)

    def cond(state):
        t, _, progress = state
        return (t < total) & ~jnp.all(progress.done)

    def body(state):
        t, seq, progress = state
        hidden = model(seq, inference=True).hidden  # causal: the tokens after t do not matter
        last = model.logits(lax.dynamic_index_in_dim(hidden, t - 1, axis=1, keepdims=False))
        token, forced = _choose(t, last, seq, lengths, row_keys, sampling)
        progress = _record(progress, token, forced, eos, max_new)
        seq = seq.at[:, t].set(token)
        return t + 1, seq, progress

    _, _, progress = lax.while_loop(cond, body, (jnp.asarray(prefill, jnp.int32), seq, _start(batch, max_new, pad_id)))
    return progress.out, progress.n_new, progress.stopped
