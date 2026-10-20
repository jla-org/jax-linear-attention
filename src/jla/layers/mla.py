# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from DeepSeek-V3 (MIT), inference/model.py (class MLA, naive and absorb paths).
"""Multi-head Latent Attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear).

Per token: ``q = W_qb rmsnorm(W_qa x)`` (``W_q x`` without ``q_lora_rank``), split
per head into ``qk_nope_head_dim`` channels and ``qk_rope_head_dim`` channels that
carry RoPE. Keys and values come from a compressed latent: ``[c, k_pe] = W_kva x``,
``c = rmsnorm(c)`` (``kv_lora_rank`` channels), ``[k_nope, v] = W_kvb c`` per head,
and one rotated ``k_pe`` shared by every head. Head ``h`` attends with
``q = [q_nope, q_pe]`` and ``k = [k_nope, k_pe]``. With ``use_rope=False`` the
``k_pe`` / ``q_pe`` channels stay unrotated (NoPE, Kimi Linear's MLA layers).

The full-sequence call materializes per-head keys and values. On Hopper with bf16/fp16
compute and DeepSeek-V3's head shape (q/k 192 = 128 + 64, v 128), ``implementation``
``"auto"`` or ``"mosaic_gpu"`` runs the FA3-style Mosaic GPU kernel
(:mod:`jla.ops.mosaic_gpu.attention`) on the unpadded heads, with packed documents and
sliding windows, per batch and head shard of the context mesh. Otherwise the shared
:func:`jla.layers.attention.attention` runs (cuDNN flash attention or XLA):
``jax.nn.dot_product_attention`` needs one head dim for q, k and v, so the narrower of
(q/k, v) is zero-padded (exact: zero channels add nothing to scores or outputs) and the
output sliced back to ``v_head_dim``.

Decoding caches only ``c`` and ``k_pe`` per token (``kv_lora_rank +
qk_rope_head_dim`` numbers instead of ``num_heads * (qk_head_dim + v_head_dim)``)
and absorbs ``W_kvb`` into the query and the output, as DeepSeek-V3's
``attn_impl="absorb"``: ``score = (q_nope W_uk) . c + q_pe . k_pe`` and
``out = (p . c) W_uv``. RoPE uses the half-rotation layout of
:func:`jla.layers.rope.apply_rope` (DeepSeek-V3's reference rotates interleaved
pairs; the two are a fixed permutation of the rope channels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers.attention import (
    ATTENTION_CHECKPOINT_NAME,
    _mosaic_gpu_attention,
    attention,
    resolve_attention_implementation,
)
from jla.layers.cache import KVCache
from jla.layers.linear import Linear
from jla.layers.norm import RMSNorm
from jla.layers.rope import apply_rope

if TYPE_CHECKING:  # layers never import jla.models at runtime (it imports layers)
    from jla.models.config import MLAConfig

__all__ = ["MLA"]

Array = jax.Array


class MLA(eqx.Module):
    """Multi-head latent attention token mixer with a compressed-latent decode cache."""

    q_proj: Linear | None  # [D, H * qk_head_dim] when q_lora_rank is None
    q_a_proj: Linear | None  # [D, q_lora_rank]
    q_a_norm: RMSNorm | None
    q_b_proj: Linear | None  # [q_lora_rank, H * qk_head_dim]
    kv_a_proj: Linear  # [D, kv_lora_rank + qk_rope_head_dim]
    kv_a_norm: RMSNorm
    kv_b_proj: Linear  # [kv_lora_rank, H * (qk_nope_head_dim + v_head_dim)]
    o_proj: Linear  # [H * v_head_dim, D]
    cfg: MLAConfig = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        cfg: MLAConfig,
        *,
        key: Array,
        std: float,
        norm_eps: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        kq, kqa, kqb, kkva, kkvb, ko = jax.random.split(key, 6)
        heads, qk = cfg.num_heads, cfg.qk_head_dim
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        if cfg.q_lora_rank:
            self.q_proj = None
            self.q_a_proj = Linear(d_model, cfg.q_lora_rank, key=kqa, axes=("embed", None), **common)
            self.q_a_norm = RMSNorm(cfg.q_lora_rank, eps=norm_eps, param_dtype=param_dtype)
            self.q_b_proj = Linear(cfg.q_lora_rank, heads * qk, key=kqb, axes=(None, "heads"), **common)
        else:
            self.q_proj = Linear(d_model, heads * qk, key=kq, axes=("embed", "heads"), **common)
            self.q_a_proj = self.q_a_norm = self.q_b_proj = None
        self.kv_a_proj = Linear(d_model, cfg.kv_lora_rank + cfg.qk_rope_head_dim, key=kkva,
                                axes=("embed", None), **common)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, eps=norm_eps, param_dtype=param_dtype)
        self.kv_b_proj = Linear(cfg.kv_lora_rank, heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), key=kkvb,
                                axes=(None, "heads"), **common)
        self.o_proj = Linear(heads * cfg.v_head_dim, d_model, key=ko, axes=("heads", "embed"), **common)
        self.cfg = cfg

    # Projections ---------------------------------------------------------------------------
    def _rope(self, x: Array, positions: Array) -> Array:
        if not self.cfg.use_rope:
            return x
        return apply_rope(x, positions, theta=self.cfg.rope_theta, scaling=self.cfg.rope_scaling)

    def _queries(self, x: Array, positions: Array) -> tuple[Array, Array]:
        """``q_nope [B, T, H, qk_nope]`` and rotated ``q_pe [B, T, H, qk_rope]``."""
        batch, length, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x) if self.q_proj is not None else self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.reshape(batch, length, cfg.num_heads, cfg.qk_head_dim)
        q_nope, q_pe = q[..., : cfg.qk_nope_head_dim], q[..., cfg.qk_nope_head_dim :]
        return q_nope, self._rope(q_pe, positions)

    def _latents(self, x: Array, positions: Array) -> tuple[Array, Array]:
        """Normalized latent ``c [B, T, kv_lora_rank]`` and the shared rotated key ``k_pe [B, T, qk_rope]``."""
        cfg = self.cfg
        kv = self.kv_a_proj(x)
        c = self.kv_a_norm(kv[..., : cfg.kv_lora_rank])
        k_pe = self._rope(kv[..., cfg.kv_lora_rank :][:, :, None, :], positions)[:, :, 0, :]
        return c, k_pe

    def _output(self, o: Array) -> Array:
        batch, length = o.shape[:2]
        o = checkpoint_name(o.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        return self.o_proj(o)

    def _mosaic_gpu(self, q: Array, k: Array, v: Array) -> bool:
        """Whether the full-sequence call runs the Mosaic GPU kernel on the unpadded heads.

        ``"mosaic_gpu"`` always does (an unsupported shape raises); ``"auto"`` does when the kernel
        supports the operands (Hopper, bf16/fp16, head dims ``(192, 128)`` or one of 64/128).
        """
        if self.cfg.implementation == "mosaic_gpu":
            return True
        if self.cfg.implementation != "auto":
            return False
        from jla.ops.mosaic_gpu.attention import supported

        return supported(q, k, v) is None

    # Full sequence -------------------------------------------------------------------------
    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        batch, length, _ = x.shape
        cfg = self.cfg
        heads, nope, rope, dv = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        q_nope, q_pe = self._queries(x, positions)
        c, k_pe = self._latents(x, positions)
        kv = self.kv_b_proj(c).reshape(batch, length, heads, nope + dv)
        k_nope, v = kv[..., :nope], kv[..., nope:]
        q = jnp.concatenate([q_nope, q_pe.astype(q_nope.dtype)], axis=-1)
        k_pe = jnp.broadcast_to(k_pe[:, :, None, :].astype(k_nope.dtype), (batch, length, heads, rope))
        k = jnp.concatenate([k_nope, k_pe], axis=-1)
        if self._mosaic_gpu(q, k, v):  # the FA3-style kernel takes the narrower values as they are
            o = _mosaic_gpu_attention(q, k, v, segment_ids=segment_ids, sliding_window=cfg.sliding_window,
                                      scale=cfg.scale, sinks=None)
            return self._output(o)
        impl = resolve_attention_implementation(cfg.implementation, q, k, v=v, segment_ids=segment_ids)
        if impl == "cudnn" and _cudnn_takes_value_head_dim(q, v):  # cuDNN's MLA mode: no padding
            o = attention(q, k, v, segment_ids=segment_ids, sliding_window=cfg.sliding_window, scale=cfg.scale,
                          implementation="cudnn")
            return self._output(o)
        width = max(cfg.qk_head_dim, dv)
        q, k, v = (_pad_last(t, width) for t in (q, k, v))
        o = attention(q, k, v, segment_ids=segment_ids, sliding_window=cfg.sliding_window,
                      scale=cfg.scale, implementation=impl)
        return self._output(o[..., :dv])

    # Decoding ------------------------------------------------------------------------------
    def init_cache(self, batch: int, max_len: int, dtype: jnp.dtype) -> KVCache:
        """Latent cache: ``keys`` hold ``c [B, S, 1, kv_lora_rank]``, ``values`` hold ``k_pe [B, S, 1, qk_rope]``."""
        cfg = self.cfg
        ring = cfg.sliding_window is not None and cfg.sliding_window < max_len
        size = cfg.sliding_window if ring else max_len
        if size < 1:
            raise ValueError(f"cache size must be >= 1; got {size}")
        return KVCache(
            keys=jnp.zeros((batch, size, 1, cfg.kv_lora_rank), dtype),
            values=jnp.zeros((batch, size, 1, cfg.qk_rope_head_dim), dtype),
            positions=jnp.full((batch, size), -1, jnp.int32),
            ring=ring,
        )

    def _absorbed(self, q_nope: Array, q_pe: Array, latents: Array, k_pe: Array, q_positions: Array,
                  kv_positions: Array) -> Array:
        """Attention over cached latents with ``W_kvb`` absorbed; float32, returns ``[B, T, H, v]``."""
        cfg = self.cfg
        f32 = jnp.float32
        w = self.kv_b_proj.weight.astype(f32).reshape(cfg.kv_lora_rank, cfg.num_heads, -1)
        w_uk, w_uv = w[..., : cfg.qk_nope_head_dim], w[..., cfg.qk_nope_head_dim :]
        q_lat = jnp.einsum("bthd,rhd->bthr", q_nope.astype(f32), w_uk)
        lat = latents.astype(f32)
        scores = jnp.einsum("bthr,bsr->bhts", q_lat, lat) + jnp.einsum(
            "bthe,bse->bhts", q_pe.astype(f32), k_pe.astype(f32))
        scores = scores * cfg.scale
        kv = kv_positions[:, None, :]
        qp = q_positions[:, :, None]
        mask = (kv >= 0) & (kv <= qp)
        if cfg.sliding_window is not None:
            mask = mask & (qp - kv < cfg.sliding_window)
        scores = jnp.where(mask[:, None], scores, jnp.finfo(f32).min)
        p = jax.nn.softmax(scores, axis=-1)
        o_lat = jnp.einsum("bhts,bsr->bthr", p, lat)
        return jnp.einsum("bthr,rhd->bthd", o_lat, w_uv)

    def step(self, x: Array, positions: Array, cache: KVCache) -> tuple[Array, KVCache]:
        """Prefill or decode ``x [B, T, D]`` at ``positions [B, T]`` against the latent ``cache``."""
        q_nope, q_pe = self._queries(x, positions)
        c, k_pe = self._latents(x, positions)
        c4, pe4 = c[:, :, None, :], k_pe[:, :, None, :]
        positions = positions.astype(jnp.int32)
        if cache.ring:
            # Attend before writing: a long block can overwrite slots its first queries need.
            lat = jnp.concatenate([cache.keys, c4.astype(cache.keys.dtype)], axis=1)
            pe = jnp.concatenate([cache.values, pe4.astype(cache.values.dtype)], axis=1)
            kv_pos = jnp.concatenate([cache.positions, positions], axis=1)
            o = self._absorbed(q_nope, q_pe, lat[:, :, 0], pe[:, :, 0], positions, kv_pos)
            cache = cache.write(c4, pe4, positions)
        else:
            cache = cache.write(c4, pe4, positions)
            o = self._absorbed(q_nope, q_pe, cache.keys[:, :, 0], cache.values[:, :, 0], positions,
                               cache.positions)
        return self._output(o.astype(q_nope.dtype)), cache


def _cudnn_takes_value_head_dim(q: Array, v: Array) -> bool:
    """Whether cuDNN takes this value head dim unpadded: JAX's cuDNN SDPA runs ``Dv != Dqk`` (MLA mode) with
    cuDNN >= 9.10 on Hopper and later, both head dims multiples of 8 (q/k at most 256)."""
    from jla.layers.attention import _cuda_compute_capability

    if v.shape[-1] == q.shape[-1]:
        return True
    capability = _cuda_compute_capability()
    if capability is None or capability < (9, 0) or q.shape[-1] % 8 or v.shape[-1] % 8 or q.shape[-1] > 256:
        return False
    try:
        from jax._src.cudnn.fused_attention_stablehlo import check_cudnn_version

        return check_cudnn_version() >= 91000
    except Exception:  # noqa: BLE001 (no cuDNN: the padded path)
        return False


def _pad_last(x: Array, width: int) -> Array:
    pad = width - x.shape[-1]
    if pad == 0:
        return x
    return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
