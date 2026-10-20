# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-token log-probabilities of targets under an LM head, without ``[N, V]`` logits.

``linear_token_logprobs(hidden, weight, targets)`` returns, for every row,
``log softmax(hidden @ weight.T)[target]`` as float32 (0 where the target equals
``ignore_index``). The forward streams the vocabulary in tiles of ``vocab_chunk``
rows with an online logsumexp (as :func:`jla.ops.cross_entropy.linear_cross_entropy`
does); the analytic custom VJP accepts an arbitrary per-token cotangent ``g`` and
rebuilds one logit tile at a time:

    d logp_t / d logits_t = onehot(target_t) - softmax(logits_t)

so preference and policy-gradient objectives (DPO sums per sequence, GRPO ratios)
get exact gradients with ``O(N * vocab_chunk)`` peak memory.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["linear_token_logprobs", "token_logprobs_from_logits"]

Array = jax.Array


def _tiles(weight: Array, tile: int, dtype: jnp.dtype) -> Array:
    vocab, dim = weight.shape
    pad = (-vocab) % tile
    w = weight.astype(dtype)
    if pad:
        w = jnp.pad(w, ((0, pad), (0, 0)))
    return w.reshape(-1, tile, dim)


def _tile_logits(hidden: Array, w_tile: Array) -> Array:
    return lax.dot_general(hidden, w_tile, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _forward(hidden: Array, weight: Array, targets: Array, tile: int, compute_dtype: jnp.dtype) -> tuple[Array, Array]:
    vocab = weight.shape[0]
    tiles = _tiles(weight, tile, compute_dtype)
    h = hidden.astype(compute_dtype)
    n = h.shape[0]
    columns = jnp.arange(tile, dtype=jnp.int32)

    def step(carry, xs):
        run_max, run_sum, target_logit = carry
        index, w_tile = xs
        offset = index * tile
        logits = _tile_logits(h, w_tile)
        masked = jnp.where((offset + columns) < vocab, logits, -jnp.inf)
        new_max = jnp.maximum(run_max, jnp.max(masked, axis=-1))
        run_sum = run_sum * jnp.exp(run_max - new_max) + jnp.sum(jnp.exp(masked - new_max[:, None]), axis=-1)
        local = targets - offset
        hit = (local >= 0) & (local < tile)
        picked = jnp.take_along_axis(logits, jnp.clip(local, 0, tile - 1)[:, None], axis=-1)[:, 0]
        return (new_max, run_sum, jnp.where(hit, picked, target_logit)), None

    init = (jnp.full((n,), -jnp.inf, jnp.float32), jnp.zeros((n,), jnp.float32), jnp.zeros((n,), jnp.float32))
    (run_max, run_sum, target_logit), _ = lax.scan(step, init, (jnp.arange(tiles.shape[0], dtype=jnp.int32), tiles))
    lse = run_max + jnp.log(run_sum)
    return target_logit - lse, lse


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _token_logprobs(hidden, weight, targets, valid, tile, compute_dtype):
    logp, _ = _forward(hidden, weight, targets, tile, compute_dtype)
    return jnp.where(valid, logp, 0.0)


def _token_logprobs_fwd(hidden, weight, targets, valid, tile, compute_dtype):
    logp, lse = _forward(hidden, weight, targets, tile, compute_dtype)
    return jnp.where(valid, logp, 0.0), (hidden, weight, targets, valid, lse)


def _token_logprobs_bwd(tile, compute_dtype, residuals, g):
    hidden, weight, targets, valid, lse = residuals
    vocab, dim = weight.shape
    tiles = _tiles(weight, tile, compute_dtype)
    h = hidden.astype(compute_dtype)
    columns = jnp.arange(tile, dtype=jnp.int32)
    coef = jnp.where(valid, jnp.asarray(g, jnp.float32), 0.0)  # [N]

    def step(grad_hidden, xs):
        index, w_tile = xs
        offset = index * tile
        logits = _tile_logits(h, w_tile)
        probs = jnp.where((offset + columns) < vocab, jnp.exp(logits - lse[:, None]), 0.0)
        onehot = (targets - offset)[:, None] == columns[None, :]
        grad_logits = (coef[:, None] * (onehot.astype(jnp.float32) - probs)).astype(compute_dtype)
        grad_hidden = grad_hidden + lax.dot_general(
            grad_logits, w_tile, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        grad_w_tile = lax.dot_general(grad_logits, h, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        return grad_hidden, grad_w_tile.astype(weight.dtype)

    grad_hidden, grad_tiles = lax.scan(
        step, jnp.zeros((h.shape[0], dim), jnp.float32), (jnp.arange(tiles.shape[0], dtype=jnp.int32), tiles)
    )
    grad_weight = grad_tiles.reshape(-1, dim)[:vocab]
    return grad_hidden.astype(hidden.dtype), grad_weight, None, None


_token_logprobs.defvjp(_token_logprobs_fwd, _token_logprobs_bwd)


def linear_token_logprobs(
    hidden: Array,
    weight: Array,
    targets: Array,
    *,
    ignore_index: int = -100,
    vocab_chunk: int = 4096,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """``log softmax(hidden [N, D] @ weight [V, D].T)[n, targets[n]]`` as float32 ``[N]``.

    Rows whose target is ``ignore_index`` return 0 and receive no gradient. Matmuls take
    ``compute_dtype`` operands with float32 accumulation; softmax statistics are float32.
    """
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"expected hidden [N, D] and weight [V, D]; got {hidden.shape} and {weight.shape}")
    if targets.shape != hidden.shape[:1]:
        raise ValueError(f"targets must be [N] = {hidden.shape[:1]}; got {targets.shape}")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError(f"targets must be integers; got {targets.dtype}")
    if vocab_chunk < 1:
        raise ValueError(f"vocab_chunk must be >= 1; got {vocab_chunk}")
    valid = targets != ignore_index
    safe_targets = jnp.where(valid, targets, 0).astype(jnp.int32)
    # Zero ignored rows first so a non-finite ignored activation cannot reach the gradients.
    hidden = jnp.where(valid[:, None], hidden, jnp.zeros((), hidden.dtype))
    tile = min(int(vocab_chunk), weight.shape[0])
    return _token_logprobs(hidden, weight, safe_targets, valid, tile, jnp.dtype(compute_dtype))


def token_logprobs_from_logits(logits: Array, targets: Array, *, ignore_index: int = -100) -> Array:
    """Same values from explicit ``logits [N, V]`` (plain autodiff; for tests and sampling)."""
    valid = targets != ignore_index
    safe = jnp.where(valid, targets, 0)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.where(valid, jnp.take_along_axis(logp, safe[:, None], axis=-1)[:, 0], 0.0)
