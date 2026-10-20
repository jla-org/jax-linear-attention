# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from flash-linear-attention (MIT), fla/layers/nsa.py; the DSA indexer follows
# deepseek-ai/DeepSeek-V3.2-Exp (MIT), inference/model.py (Indexer).
"""Sparse softmax-attention mixers: Native Sparse Attention (``"nsa"``) and DeepSeek Sparse Attention (``"dsa"``).

Both reuse :class:`jla.layers.attention.Attention` for the grouped-query q/k/v/o
projections, QK-norm, RoPE and the optional output gate (``config.attention``);
``config.sparse_attention`` holds the sparse settings.

* NSA (FLA's layer): a ``d_model -> 3 * num_heads`` projection gives sigmoid gates
  for the compressed, selected and sliding-window branches.
* DSA with GQA projections (DeepSeek-V3.2 uses MLA): the lightning indexer has its
  own query heads ``x W_qi``, one LayerNorm-ed index key ``x W_ki`` per token, partial
  RoPE on both, and per-head weights ``x W_w / sqrt(index_heads)``. The indexer reads
  a stop-gradient copy of the input and learns only from DeepSeek-V3.2's KL loss
  between the main attention distribution (summed over heads, over the selected
  keys) and its own softmax scores, injected into the backward pass with weight
  ``indexer_loss_coef`` (summed over queries, the convention of the train step's
  summed objective; 0 gives no indexer gradient, though weight decay still applies);
  the language-model loss never reaches it.
  The FP8 quantization and Hadamard rotation of DeepSeek's inference indexer are
  not used. ``dsa_implementation="auto"`` (the default; on a Hopper GPU, for the shapes
  they take, unless ``attention.implementation="xla"`` pins plain JAX) or ``"mosaic_gpu"``
  runs the indexer, the top-k and the selected-key
  attention on the kernels of :mod:`jla.ops.mosaic_gpu.dsa` (same selection and math;
  bf16/fp16 compute), in query blocks, and the KL over the selected keys with its
  indexer gradient on backward kernels (``jla.ops.mosaic_gpu.dsa.dsa_layer``);
  ``"xla"`` runs the plain ops (dense ``[T, T]`` scores).

Decoding with a KV cache is not implemented for these mixers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from jla.layers.attention import Attention
from jla.layers.linear import Linear
from jla.layers.rope import apply_rope
from jla.ops.sparse_attention import attach_indexer_grad, dsa, dsa_attention_target, dsa_index_scores, dsa_select, nsa

if TYPE_CHECKING:  # layers never import jla.models at runtime (it imports layers)
    from jla.models.config import AttentionConfig, SparseAttentionConfig

__all__ = ["SPARSE_MIXERS", "DeepSeekSparseAttention", "NativeSparseAttention"]

Array = jax.Array


def _layer_norm(x: Array, weight: Array, bias: Array, eps: float) -> Array:
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * jax.lax.rsqrt(var + eps) * weight + bias).astype(x.dtype)


def _check_attention(attention: AttentionConfig, kind: str) -> None:
    if attention.sliding_window is not None:
        raise ValueError(f"{kind} layers do not take attention.sliding_window (NSA has window_size)")
    if attention.implementation not in ("auto", "xla"):
        paths = ("run plain JAX only" if kind != "dsa" else
                 "pick their kernels with sparse_attention.dsa_implementation ('xla' here keeps 'auto' on plain JAX)")
        raise ValueError(f"{kind} layers {paths}; attention.implementation must be 'auto' or 'xla', "
                         f"got {attention.implementation!r}")


class NativeSparseAttention(eqx.Module):
    """NSA token mixer: compressed, selected and sliding-window branches mixed by sigmoid gates."""

    attn: Attention
    branch_gate_proj: Linear
    cfg: SparseAttentionConfig = eqx.field(static=True)

    def __init__(self, d_model: int, attention: AttentionConfig, cfg: SparseAttentionConfig, *, key: Array,
                 std: float, norm_eps: float, param_dtype: jnp.dtype = jnp.float32,
                 compute_dtype: str = "bfloat16"):
        _check_attention(attention, "nsa")
        k_attn, k_gate = jax.random.split(key)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.attn = Attention(d_model, attention, key=k_attn, norm_eps=norm_eps, **common)
        self.branch_gate_proj = Linear(d_model, 3 * attention.num_heads, key=k_gate, axes=("embed", "heads"),
                                       **common)
        self.cfg = cfg

    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        batch, length, _ = x.shape
        q, k, v = self.attn._project(x, positions)
        gates = jax.nn.sigmoid(self.branch_gate_proj(x).astype(jnp.float32))
        gates = gates.reshape(batch, length, self.attn.cfg.num_heads, 3)
        o = nsa(q, k, v, gates[..., 0], gates[..., 1], gates[..., 2], block_size=self.cfg.block_size,
                block_count=self.cfg.block_count, window_size=self.cfg.window_size,
                scale=self.attn.cfg.softmax_scale, segment_ids=segment_ids)
        return self.attn._output(x, o)

    def init_cache(self, batch: int, dtype: jnp.dtype):
        raise NotImplementedError("decoding with a KV cache is not implemented for nsa layers")

    def step(self, x: Array, cache):
        raise NotImplementedError("decoding with a KV cache is not implemented for nsa layers")


class DeepSeekSparseAttention(eqx.Module):
    """DSA token mixer: lightning indexer, top-k keys per query, GQA attention over them."""

    attn: Attention
    index_q_proj: Linear
    index_k_proj: Linear
    index_w_proj: Linear
    index_k_norm_weight: Array
    index_k_norm_bias: Array
    cfg: SparseAttentionConfig = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)

    def __init__(self, d_model: int, attention: AttentionConfig, cfg: SparseAttentionConfig, *, key: Array,
                 std: float, norm_eps: float, param_dtype: jnp.dtype = jnp.float32,
                 compute_dtype: str = "bfloat16"):
        _check_attention(attention, "dsa")
        k_attn, k_q, k_k, k_w = jax.random.split(key, 4)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        heads, dim = cfg.index_heads, cfg.index_head_dim
        self.attn = Attention(d_model, attention, key=k_attn, norm_eps=norm_eps, **common)
        self.index_q_proj = Linear(d_model, heads * dim, key=k_q, axes=("embed", "heads"), **common)
        self.index_k_proj = Linear(d_model, dim, key=k_k, axes=("embed", None), **common)
        self.index_w_proj = Linear(d_model, heads, key=k_w, axes=("embed", None), **common)
        self.index_k_norm_weight = jnp.ones((dim,), param_dtype)
        self.index_k_norm_bias = jnp.zeros((dim,), param_dtype)
        self.cfg = cfg
        self.norm_eps = float(norm_eps)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"index_k_norm_weight": (None,), "index_k_norm_bias": (None,)}

    def _index_inputs(self, x: Array, positions: Array) -> tuple[Array, Array, Array]:
        """The indexer's queries ``[B, T, HI, DI]``, shared keys ``[B, T, DI]`` and head weights ``[B, T, HI]``
        (float32, times ``index_heads ** -0.5``), from a stop-gradient copy of ``x``."""
        batch, length, _ = x.shape
        cfg, attention = self.cfg, self.attn.cfg
        x = jax.lax.stop_gradient(x)
        q_idx = self.index_q_proj(x).reshape(batch, length, cfg.index_heads, cfg.index_head_dim)
        k_idx = self.index_k_proj(x)
        k_idx = _layer_norm(k_idx, self.index_k_norm_weight, self.index_k_norm_bias, self.norm_eps)
        if attention.use_rope:
            rope = dict(theta=attention.rope_theta, rotary_fraction=cfg.index_rotary_fraction,
                        scaling=attention.rope_scaling, exact_angles=attention.rope_exact_angles)
            q_idx = apply_rope(q_idx, positions, **rope)
            k_idx = apply_rope(k_idx[:, :, None], positions, **rope)[:, :, 0]
        w_idx = self.index_w_proj(x).astype(jnp.float32) * cfg.index_heads ** -0.5
        return q_idx, k_idx, w_idx

    def _uses_kernels(self, q: Array, k: Array, v: Array, q_idx: Array, k_idx: Array, w_idx: Array) -> bool:
        """``dsa_implementation``: ``"mosaic_gpu"`` always (unsupported shapes raise), ``"auto"`` when the
        Hopper kernels take these operands and ``attention.implementation`` is not pinned to ``"xla"``,
        ``"xla"`` never."""
        if self.cfg.dsa_implementation == "xla":
            return False
        if self.cfg.dsa_implementation == "mosaic_gpu":
            return True
        if jax.default_backend() != "gpu" or self.attn.cfg.implementation == "xla":
            return False
        from jla.ops.mosaic_gpu import dsa as dsa_kernels  # GPU-only module, imported on use

        return (dsa_kernels.supported_attention(q, k, v) is None
                and dsa_kernels.supported_index(q_idx, k_idx, w_idx) is None)

    def index_scores(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        """Lightning-indexer scores ``[B, T, T]`` (float32, ``-inf`` outside the causal document)."""
        q_idx, k_idx, w_idx = self._index_inputs(x, positions)
        if self.cfg.dsa_implementation == "mosaic_gpu" or (self.cfg.dsa_implementation == "auto"
                                                           and jax.default_backend() == "gpu"
                                                           and self.attn.cfg.implementation != "xla"):
            from jla.ops.mosaic_gpu import dsa as dsa_kernels  # GPU-only module, imported on use

            if (self.cfg.dsa_implementation == "mosaic_gpu"
                    or dsa_kernels.supported_index(q_idx, k_idx, w_idx) is None):
                return dsa_kernels.index_scores(q_idx, k_idx, w_idx, segment_ids=segment_ids)
        return dsa_index_scores(q_idx, k_idx, w_idx, segment_ids=segment_ids)

    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        q_idx, k_idx, w_idx = self._index_inputs(x, positions)
        q, k, v = self.attn._project(x, positions)
        if self._uses_kernels(q, k, v, q_idx, k_idx, w_idx):
            from jla.ops.mosaic_gpu import dsa as dsa_kernels  # GPU-only module, imported on use

            o = dsa_kernels.dsa_layer(q, k, v, q_idx, k_idx, w_idx, top_k=self.cfg.top_k,
                                      scale=self.attn.cfg.softmax_scale,
                                      indexer_loss_coef=self.cfg.indexer_loss_coef, segment_ids=segment_ids,
                                      index_fp8=self.cfg.dsa_index_fp8)
            return self.attn._output(x, o)
        if self.cfg.dsa_index_fp8:
            raise ValueError("dsa_index_fp8 needs the DSA kernels (dsa_implementation 'mosaic_gpu', or 'auto' on a "
                             "Hopper GPU with shapes they take); the plain path selects exactly")
        scores = dsa_index_scores(q_idx, k_idx, w_idx, segment_ids=segment_ids)
        indices = dsa_select(scores, self.cfg.top_k)
        o, probs, selected = dsa(q, k, v, indices, scale=self.attn.cfg.softmax_scale, segment_ids=segment_ids,
                                 return_probs=True)
        if self.cfg.indexer_loss_coef > 0:
            o = attach_indexer_grad(o, scores, dsa_attention_target(probs), selected.astype(jnp.float32),
                                    float(self.cfg.indexer_loss_coef))
        return self.attn._output(x, o)

    def init_cache(self, batch: int, dtype: jnp.dtype):
        raise NotImplementedError("decoding with a KV cache is not implemented for dsa layers")

    def step(self, x: Array, cache):
        raise NotImplementedError("decoding with a KV cache is not implemented for dsa layers")


SPARSE_MIXERS = {"nsa": NativeSparseAttention, "dsa": DeepSeekSparseAttention}
