# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear token mixer (Sun et al. 2024, arXiv 2407.04620), on the shared mixer scaffold.

``x`` is projected to ``q, k, v [B, T, H, D]`` (short causal conv plus SiLU when
``conv_size > 0``), the inner learning rate is
``eta = ttt_base_lr * sigmoid(x W_lr + b_lr) / D`` per head, and
:func:`jla.ops.ttt.ttt_linear` runs the inner model ``f(x) = x + LN(x W + b)`` from the
learnable initial weights ``(w0, b0)`` (``w0 ~ N(0, 0.02)``, ``b0 = 0``, layer norm
``ln_weight = 1``, ``ln_bias = 0``). The per-position factor is
``max(1 / (i + 1) + token_offset_i, 0)`` with a learnable ``token_offset`` (zero at
init). The layer was written from the paper (arXiv 2407.04620) without reading upstream TTT
code. The output goes through the shared output (gated) RMSNorm and projection. Needs
``num_k_heads == num_v_heads`` and ``head_k_dim == head_v_dim``.

Importing this module registers the mixer as ``make_mixer("ttt_linear", ...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers import linear_attention as _la
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME
from jla.layers.cache import RecurrentCache
from jla.layers.linear import Linear, truncated_normal
from jla.ops import ttt as ttt_ops

if TYPE_CHECKING:
    from jla.models.config import LinearAttentionConfig

__all__ = ["TTTLinear"]

Array = jax.Array


class TTTLinear(_la.LinearAttentionMixer):
    """TTT-Linear mixer; ``step`` continues from a :class:`RecurrentCache` holding a ``TTTState``."""

    lr_proj: Linear
    ln_weight: Array
    ln_bias: Array
    w0: Array
    b0: Array
    token_offset: Array

    def __init__(self, d_model: int, cfg: LinearAttentionConfig, *, key: Array, std: float,
                 norm_eps: float, param_dtype: jnp.dtype = jnp.float32, compute_dtype: str = "bfloat16"):
        if cfg.num_k_heads != cfg.num_v_heads or cfg.head_k_dim != cfg.head_v_dim:
            raise ValueError("TTT-Linear needs num_k_heads == num_v_heads and head_k_dim == head_v_dim")
        k0, klr, kw = jax.random.split(key, 3)
        self._init_common("ttt_linear", d_model, cfg, k0, std, norm_eps, param_dtype, compute_dtype)
        heads, dim = cfg.num_v_heads, cfg.head_v_dim
        self.lr_proj = Linear(d_model, heads, key=klr, axes=("embed", "heads"), std=std, use_bias=True,
                              param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.ln_weight = jnp.ones((heads, dim), param_dtype)
        self.ln_bias = jnp.zeros((heads, dim), param_dtype)
        self.w0 = truncated_normal(kw, (heads, dim, dim), 0.02, param_dtype)
        self.b0 = jnp.zeros((heads, dim), param_dtype)
        self.token_offset = jnp.zeros((cfg.ttt_mini_batch_size,), param_dtype)

    def gates(self, x: Array) -> dict[str, Array]:
        cfg = self.cfg
        logits = self._heads(self.lr_proj(x), cfg.num_v_heads)[..., 0].astype(jnp.float32)  # [B, T, H]
        return {"eta": cfg.ttt_base_lr * jax.nn.sigmoid(logits) / cfg.head_v_dim}

    def _options(self) -> dict:
        base = ttt_ops.default_token_scale(self.cfg.ttt_mini_batch_size)
        return dict(
            ln_weight=self.ln_weight.astype(jnp.float32), ln_bias=self.ln_bias.astype(jnp.float32),
            w0=self.w0.astype(jnp.float32), b0=self.b0.astype(jnp.float32),
            mini_batch_size=self.cfg.ttt_mini_batch_size,
            token_scale=jnp.maximum(base + self.token_offset.astype(jnp.float32), 0.0),
        )

    def _forward(self, x: Array, *, segment_ids: Array | None,
                 cache: RecurrentCache | None) -> tuple[Array, RecurrentCache | None]:
        cfg = self.cfg
        if cfg.context_parallel_axis is not None and cache is None:
            raise NotImplementedError("context parallelism (context_parallel_axis) does not support ttt_linear layers")
        conv_state = (None, None, None) if cache is None or cache.conv is None else cache.conv
        q, cq = self._activate(self.q_proj(x), self.q_conv, segment_ids, conv_state[0])
        k, ck = self._activate(self.k_proj(x), self.k_conv, segment_ids, conv_state[1])
        v, cv = self._activate(self.v_proj(x), self.v_conv, segment_ids, conv_state[2])
        q, k, v = (self._heads(t, cfg.num_v_heads) for t in (q, k, v))
        options = {**self._options(), **self.gates(x)}
        eta = options.pop("eta")
        if segment_ids is not None:
            options["segment_ids"] = segment_ids
        impl = {"auto": "auto", "reference": "reference", "chunked": "dual"}.get(cfg.implementation)
        if impl is None:
            raise ValueError(f"TTT-Linear has no {cfg.implementation!r} implementation (auto, reference, chunked)")
        w0, b0 = options["w0"][None], options["b0"][None]
        if cache is not None:  # decode and prefill-from-cache run the stateful reference recurrence
            st = cache.state
            options.update(initial_state=st._replace(w=st.w + w0, b=st.b + b0), output_final_state=True)
            impl = "reference"
        o, final = ttt_ops.ttt_linear(q, k, v, eta, impl=impl, **options)
        if final is not None:
            final = final._replace(w=final.w - w0, b=final.b - b0)
        o = checkpoint_name(o, ATTENTION_CHECKPOINT_NAME)
        if self.g_proj is not None:
            o = self.o_norm(o, self._heads(self.g_proj(x), cfg.num_v_heads))
        else:
            o = self.o_norm(o)
        y = self.o_proj(o.reshape(*o.shape[:2], -1))
        if cache is None:
            return y, None
        conv = None if self.q_conv is None else (cq, ck, cv)
        return y, RecurrentCache(state=final, conv=conv)

    def init_cache(self, batch: int, dtype: jnp.dtype) -> RecurrentCache:
        # Shapes come from the config only (works on layer-stacked modules): the cached
        # weights are stored as offsets from this layer's (w0, b0).
        cfg = self.cfg
        heads, dim = cfg.num_v_heads, cfg.head_v_dim
        w = jnp.zeros((batch, heads, dim, dim), jnp.float32)
        b = jnp.zeros((batch, heads, dim), jnp.float32)
        zeros = jnp.zeros((batch,), jnp.int32)
        state = ttt_ops.TTTState(w, b, w, b, zeros, zeros)
        conv = None
        if cfg.conv_size > 0:
            width = heads * dim
            conv = tuple(jnp.zeros((batch, cfg.conv_size - 1, width), dtype) for _ in range(3))
        return RecurrentCache(state=state, conv=conv)

    def param_axes(self):
        return {"ln_weight": (None, None), "ln_bias": (None, None), "w0": (None, None, None),
                "b0": (None, None), "token_offset": (None,)}


_la._MIXERS["ttt_linear"] = TTTLinear
