# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Hugging Face transformers (Apache-2.0), src/transformers/models/minimax/modular_minimax.py
# (MiniMaxLightningAttention); decay schedule of Lightning Attention-2 (arXiv 2401.04658).
"""MiniMax-01 lightning attention (``layer_types`` entry ``"lightning"``).

The linear-attention layer of MiniMax-Text-01 (arXiv 2501.08313), as in
``MiniMaxLightningAttention``:

* ``q, k, v = silu(x W_qkv)``, one projection whose output is laid out per head
  (``[H, (q | k | v), head_dim]``), no QK normalisation and no ``1/sqrt(d)`` scale;
* the ``simple_gla`` recurrence ``S_t = exp(-s_h) S_{t-1} + k_t v_t^T``,
  ``o_t = q_t^T S_t`` with a **fixed** per-head decay rate
  ``s_h = 2^(-8 (h + 1) / H) * (1 - l / (L - 1 + 1e-5) + 1e-5)`` for layer ``l`` of ``L``
  (the ALiBi slopes, smaller in deeper layers; TransNormerLLM / Lightning Attention-2);
* an RMSNorm over all ``H * head_dim`` channels (eps ``1e-6``, MiniMax's default), the
  output gate ``sigmoid(x W_g)`` and the output projection.

The decay rates are not parameters. The layer stores only its depth, as an int32
``[1]`` array ``layer_index`` (so layers of one type keep one structure and stack under
``scan_layers``), and computes ``s_h`` from it in float32 in the graph. An integer
leaf is not an inexact array, so the train step, the optimizers (weight decay,
Muon) and the parameter dtype never touch the schedule.

Reads ``LinearAttentionConfig``: ``num_k_heads == num_v_heads`` heads with
``head_k_dim == head_v_dim``, ``conv_size=0``, ``output_gate=True``,
``qk_l2norm=False``, ``chunk_size`` and ``implementation``. ``step`` continues from a
:class:`~jla.layers.cache.RecurrentCache`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name

from jla import ops
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.linear import Linear
from jla.layers.norm import RMSNorm

if TYPE_CHECKING:
    from jla.models.config import LinearAttentionConfig

__all__ = ["LightningAttention", "lightning_decay_rates"]

Array = jax.Array

NORM_EPS = 1e-6  # MiniMaxRMSNorm's default epsilon, used by the lightning-attention output norm


def lightning_decay_rates(num_heads: int, layer_idx: int | Array, n_layers: int) -> Array:
    """``[..., H]`` float32 decay rates ``s_h`` (the per-token decay is ``exp(-s_h)``) of layer ``layer_idx``.

    ``layer_idx`` may be an integer array (a stack of layers gives ``[layers, H]``). The
    factor ``1 - l / (L - 1 + 1e-5) + 1e-5`` is evaluated as ``((L - 1 - l) + 1e-5) / (L - 1 + 1e-5) + 1e-5``,
    so the integer difference stays exact in float32 and deep layers keep full relative precision.
    """
    base = 2.0 ** (-8.0 / num_heads)
    remaining = (n_layers - 1 - jnp.asarray(layer_idx, jnp.int32)).astype(jnp.float32)
    factor = (remaining + 1e-5) / (n_layers - 1 + 1e-5) + 1e-5
    return (base ** jnp.arange(1, num_heads + 1, dtype=jnp.float32)) * factor[..., None]


class LightningAttention(eqx.Module):
    """MiniMax-01 lightning attention: ``simple_gla`` with a fixed per-head decay."""

    qkv_proj: Linear  # D -> H * 3 * head_dim, per-head (q, k, v) layout
    gate_proj: Linear  # D -> H * head_dim
    norm: RMSNorm  # over H * head_dim
    o_proj: Linear
    layer_index: Array  # int32 [1]: the layer's depth, which fixes its decay rates (not a parameter)
    cfg: LinearAttentionConfig = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, layer_idx: int, n_layers: int, key: Array,
                 std: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        if cfg.num_k_heads != cfg.num_v_heads or cfg.head_k_dim != cfg.head_v_dim:
            raise ValueError("lightning attention needs num_k_heads == num_v_heads and head_k_dim == head_v_dim")
        if cfg.conv_size != 0:
            raise ValueError(f"lightning attention uses no short convolution; set conv_size=0 (got {cfg.conv_size})")
        if not cfg.output_gate:
            raise ValueError("lightning attention always gates its output: set output_gate=True")
        if cfg.qk_l2norm:
            raise ValueError("lightning attention does not normalize q and k: set qk_l2norm=False")
        if cfg.context_parallel_axis is not None:
            raise NotImplementedError("context parallelism (context_parallel_axis) does not support lightning layers")
        kqkv, kg, ko = jax.random.split(key, 3)
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        width = cfg.num_v_heads * cfg.head_v_dim
        self.qkv_proj = Linear(d_model, 3 * width, key=kqkv, axes=("embed", "heads"), **common)
        self.gate_proj = Linear(d_model, width, key=kg, axes=("embed", "heads"), **common)
        self.norm = RMSNorm(width, eps=NORM_EPS, param_dtype=param_dtype)
        self.o_proj = Linear(width, d_model, key=ko, axes=("heads", "embed"), **common)
        if not 0 <= layer_idx < n_layers:
            raise ValueError(f"lightning attention needs 0 <= layer_idx < n_layers (got {layer_idx} of {n_layers})")
        self.layer_index = jnp.full((1,), layer_idx, jnp.int32)
        self.cfg = cfg
        self.n_layers = n_layers

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"layer_index": (None,)}

    @property
    def decay_rate(self) -> Array:
        """``[H]`` float32 decay rates of this layer (``[layers, H]`` for a scanned stack), from its depth."""
        return lightning_decay_rates(self.cfg.num_v_heads, self.layer_index[..., 0], self.n_layers)

    def _forward(self, x: Array, *, segment_ids: Array | None,
                 cache: RecurrentCache | None) -> tuple[Array, RecurrentCache | None]:
        cfg = self.cfg
        batch, seq_len, _ = x.shape
        heads, dim = cfg.num_v_heads, cfg.head_v_dim
        qkv = jax.nn.silu(self.qkv_proj(x).astype(jnp.float32)).astype(x.dtype)
        qkv = qkv.reshape(batch, seq_len, heads, 3, dim)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        rate = lax.stop_gradient(self.decay_rate)  # float32, computed from the integer depth
        g = jnp.broadcast_to(-rate, (batch, seq_len, heads))
        options: dict = {"chunk_size": cfg.chunk_size, "g": g, "scale": 1.0}
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = cfg.implementation
        if cache is not None:
            options.update(initial_state=cache.state, output_final_state=True)
            if seq_len == 1:
                impl = "reference"  # a single recurrent step; chunking would pad it to a full chunk
        o, final = ops.linear_attention("simple_gla", q, k, v, impl=impl, **options)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        o = self.norm(o.reshape(batch, seq_len, heads * dim))
        o = jax.nn.sigmoid(self.gate_proj(x).astype(jnp.float32)) * o.astype(jnp.float32)
        y = self.o_proj(o.astype(x.dtype))
        if cache is None:
            return y, None
        return y, RecurrentCache(state=final, conv=None)

    def __call__(self, x: Array, segment_ids: Array | None = None) -> Array:
        return self._forward(x, segment_ids=segment_ids, cache=None)[0]

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        cfg = self.cfg
        return RecurrentCache(state=jnp.zeros((batch, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim), jnp.float32))

    def step(self, x: Array, cache: RecurrentCache) -> tuple[Array, RecurrentCache]:
        """Prefill or decode ``x [B, T, D]`` from ``cache``; returns the output and the new cache."""
        return self._forward(x, segment_ids=None, cache=cache)
