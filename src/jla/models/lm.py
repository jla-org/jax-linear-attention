# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decoder-only language model: embedding -> pre-norm blocks -> final norm -> LM head.

Each block is ``x + mixer(norm(x))`` followed by ``x + mlp(norm(x))``; the mixer
is softmax attention, a linear-attention kind or a Mamba mixer
(``config.layer_types``) and the MLP a dense SwiGLU, an MoE or absent
(``config.mlp_types``). With ``scan_layers`` the
stack is stored as the minimal repeating ``(layer_type, mlp_type)`` group with
every parameter stacked over the repeats, and the forward pass is one
``lax.scan`` over repeats (compile time independent of depth); otherwise the
blocks are a plain list run by a Python loop. Both layouts compute the same
function from the same key.

Rematerialization (``config.remat``) wraps every block in ``jax.checkpoint``:
``"full"`` saves nothing, ``"dots"`` / ``"dots_with_no_batch_dims"`` save matmul
outputs, ``"save_attention"`` saves the attention outputs (including the cuDNN
flash-attention residuals, so the attention forward never reruns in the
backward pass).
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jla.layers import ttt as _ttt_layers  # noqa: F401  (registers the ttt_linear mixer)
from jla.layers.attention import ATTENTION_CHECKPOINT_NAME, Attention
from jla.layers.cache import KVCache, RecurrentCache
from jla.layers.forgetting_attention import ForgettingAttention
from jla.layers.linear import Embedding, truncated_normal
from jla.layers.linear_attention import UPSTREAM_NORMAL_CONV_KINDS, LinearAttentionMixer, make_mixer
from jla.layers.mamba1 import Mamba1
from jla.layers.mamba3_mimo import Mamba3MIMO
from jla.layers.mla import MLA
from jla.layers.mlp import SquaredReLUMLP, SwiGLU
from jla.layers.moe import MoE, MoEStats
from jla.layers.norm import RMSNorm
from jla.layers.rwkv6 import RWKV6ChannelMix
from jla.layers.sparse_attention import SPARSE_MIXERS
from jla.layers.ssm import Mamba2, Mamba3
from jla.models.config import ModelConfig
from jla.ops.reference import segment_resets
from jla.distributed.activation_sharding import constrain_batch

SSM_MIXERS = {"mamba2": Mamba2, "mamba3": Mamba3, "mamba1": Mamba1}
ATTENTION_TYPES = ("attention", "local_attention")  # softmax mixers (they take positions and a KV cache)
POSITIONAL_MIXERS = (*ATTENTION_TYPES, "mla", "fox")  # mixers that take positions and keep a KV cache
TOKEN_SHIFT_MLPS = ("rwkv_cmix", "rwkv7_cmix")  # channel mixers that read the previous token (decode caches it)
POSITIONAL_CACHES = (*POSITIONAL_MIXERS, "path")  # mixers whose decode cache is addressed by position

__all__ = ["Block", "Cache", "LMOutput", "LanguageModel", "build_model", "layer_period", "remat_policy"]

Array = jax.Array


class LMOutput(NamedTuple):
    hidden: Array  # [B, T, D] after the final norm (compute dtype)
    pre_norm_hidden: Array  # [B, T, D] before the final norm (for MTP)
    aux_loss: Array  # f32 scalar, already weighted (MoE balance + router z-loss), 0 if none
    metrics: dict[str, Array]  # small scalars for logging


def _zero_stats() -> MoEStats:
    zero = jnp.zeros((), jnp.float32)
    return MoEStats(zero, zero, zero, zero)


def _dropout(x: Array, rate: float, key: Array | None) -> Array:
    if rate == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), jnp.zeros((), x.dtype)).astype(x.dtype)


def _save_cudnn_attention(prim, *_, **__) -> bool:
    # The cuDNN flash-attention forward primitive of jax.nn.dot_product_attention: saving
    # its outputs keeps the softmax statistics, so the backward does not rerun the forward.
    return prim.name == "dot_product_attention_fwd_wrapper"


def remat_policy(name: str, names: tuple[str, ...] = (), offload_names: tuple[str, ...] = ()) -> Any:
    """``jax.checkpoint`` policy for ``ModelConfig.remat`` (``None`` for ``"none"``).

    ``names`` and ``offload_names`` are ``ModelConfig.remat_names`` and ``remat_offload_names``, used
    by ``"names"`` and ``"offload"`` (see :func:`_named_policy`).
    """
    policies = jax.checkpoint_policies
    if name == "none":
        return None
    if name == "full":
        return policies.nothing_saveable
    if name == "dots":
        return policies.dots_saveable
    if name == "dots_with_no_batch_dims":
        return policies.dots_with_no_batch_dims_saveable
    if name == "save_attention":
        return policies.save_from_both_policies(
            policies.save_only_these_names(ATTENTION_CHECKPOINT_NAME), _save_cudnn_attention
        )
    if name == "kernels":
        return _named_policy(("attention", "kernels"), ())
    if name == "dots_kernels":
        return policies.save_from_both_policies(policies.dots_saveable, _named_policy(("attention", "kernels"), ()))
    if name == "names":
        if not names or offload_names:
            raise ValueError("remat policy 'names' needs checkpoint names (remat_names) and no offload names")
        return _named_policy(names, ())
    if name == "offload":
        if not offload_names:
            raise ValueError("remat policy 'offload' needs checkpoint names to offload (remat_offload_names)")
        return _named_policy(names, offload_names)
    raise ValueError(f"unknown remat policy {name!r}")


def layer_period(config: ModelConfig) -> int:
    """Length of the minimal repeating ``(layer_type, mlp_type)`` group."""
    pattern = list(zip(config.resolved_layer_types, config.resolved_mlp_types))
    n = len(pattern)
    for period in range(1, n + 1):
        if n % period == 0 and all(pattern[i] == pattern[i % period] for i in range(n)):
            return period
    return n


class Block(eqx.Module):
    """Pre-norm residual block: token mixer, then channel mixer (absent for ``mlp_type="none"``)."""

    mixer_norm: RMSNorm
    mixer: Attention | LinearAttentionMixer | Mamba2 | Mamba3
    mlp_norm: RMSNorm | None
    mlp: SwiGLU | MoE | None
    layer_type: str = eqx.field(static=True)
    mlp_type: str = eqx.field(static=True)
    resid_dropout: float = eqx.field(static=True)

    def __init__(self, config: ModelConfig, layer_type: str, mlp_type: str, *, key: Array, layer_idx: int = 0):
        # layer_idx (the block's depth) only sets depth-dependent initial values (RWKV-7) and the lightning
        # layers' int32 depth leaf (their decay rates); no static field depends on it, so blocks of one
        # type keep one structure.
        k_mixer, k_mlp = jax.random.split(key)
        param_dtype = jnp.dtype(config.param_dtype)
        common = dict(std=config.initializer_range, param_dtype=param_dtype, compute_dtype=config.compute_dtype)
        d = config.d_model
        self.mixer_norm = RMSNorm(d, eps=config.norm_eps, param_dtype=param_dtype)
        if layer_type in ATTENTION_TYPES:
            attn_cfg = config.attention if layer_type == "attention" else config.local_attention
            self.mixer = Attention(d, attn_cfg, key=k_mixer, norm_eps=config.norm_eps, **common)
        elif layer_type == "mla":
            self.mixer = MLA(d, config.mla, key=k_mixer, norm_eps=config.norm_eps, **common)
        elif layer_type in SSM_MIXERS:
            mixer_cls = SSM_MIXERS[layer_type]
            if config.ssm.mimo_rank > 1:
                if layer_type != "mamba3":
                    raise ValueError(f"mimo_rank > 1 is a Mamba-3 option; {layer_type} layers need mimo_rank=1")
                mixer_cls = Mamba3MIMO
            # init_scheme "upstream": out_proj U(+-1/sqrt(inner)) / sqrt(residual branches); "jla": N(0, std).
            upstream = config.init_scheme == "upstream" and layer_type != "mamba1"
            depth = {"n_residuals": residual_branches(config)} if upstream else {}
            self.mixer = mixer_cls(d, config.ssm, key=k_mixer, norm_eps=config.norm_eps, **depth, **common)
        elif layer_type == "fox":
            self.mixer = ForgettingAttention(d, config.forgetting_attention, key=k_mixer,
                                             norm_eps=config.norm_eps, **common)
        elif layer_type == "path":
            from jla.layers.path_attention import PathAttention

            self.mixer = PathAttention(d, config.path_attention, key=k_mixer, norm_eps=config.norm_eps, **common)
        elif layer_type in SPARSE_MIXERS:
            if config.attention.context_parallel_axis is not None:
                raise NotImplementedError(
                    f"context parallelism (context_parallel_axis) does not support {layer_type} layers")
            self.mixer = SPARSE_MIXERS[layer_type](d, config.attention, config.sparse_attention, key=k_mixer,
                                                   norm_eps=config.norm_eps, **common)
        elif layer_type == "rwkv7":
            from jla.layers.rwkv7 import RWKV7Mixer

            self.mixer = RWKV7Mixer(d, config.linear_attention, layer_idx=layer_idx, n_layers=config.n_layers,
                                    key=k_mixer, norm_eps=config.norm_eps, param_dtype=param_dtype,
                                    compute_dtype=config.compute_dtype, init_scheme=config.init_scheme)
        elif layer_type == "lightning":
            from jla.layers.lightning_attention import LightningAttention

            self.mixer = LightningAttention(d, config.linear_attention, layer_idx=layer_idx, n_layers=config.n_layers,
                                            key=k_mixer, **common)
        else:
            la_cfg = config.linear_attention
            if (config.init_scheme == "upstream" and la_cfg.conv_init is None
                    and layer_type in UPSTREAM_NORMAL_CONV_KINDS):
                la_cfg = dataclasses.replace(la_cfg, conv_init="normal")  # FLA's N(0, initializer_range) convs
            self.mixer = make_mixer(layer_type, d, la_cfg, key=k_mixer, norm_eps=config.norm_eps, **common)
        self.mlp_norm = None if mlp_type == "none" else RMSNorm(d, eps=config.norm_eps, param_dtype=param_dtype)
        if mlp_type == "moe":
            self.mlp = MoE(d, config.moe, key=k_mlp, **common)
        elif mlp_type == "dense":
            self.mlp = SwiGLU(d, config.d_ff, key=k_mlp, **common)
        elif mlp_type == "rwkv_cmix":
            self.mlp = RWKV6ChannelMix(d, config.d_ff, key=k_mlp, **common)
        elif mlp_type == "relu2":
            self.mlp = SquaredReLUMLP(d, config.d_ff, key=k_mlp, **common)
        elif mlp_type == "rwkv7_cmix":
            from jla.layers.rwkv7 import RWKV7FeedForward

            self.mlp = RWKV7FeedForward(d, config.d_ff, layer_idx=layer_idx, n_layers=config.n_layers, key=k_mlp,
                                        param_dtype=param_dtype, compute_dtype=config.compute_dtype,
                                        init_scheme=config.init_scheme)
        else:
            self.mlp = None
        self.layer_type = layer_type
        self.mlp_type = mlp_type
        self.resid_dropout = float(config.resid_dropout)

    def _mix(self, h: Array, positions: Array, segment_ids: Array | None) -> Array:
        if self.layer_type in POSITIONAL_MIXERS or self.layer_type in SPARSE_MIXERS:
            return self.mixer(h, positions, segment_ids)
        return self.mixer(h, segment_ids)

    def _channel(self, x: Array, *, dropless: bool = False,
                 segment_ids: Array | None = None) -> tuple[Array | None, MoEStats]:
        if self.mlp is None:
            return None, _zero_stats()
        h = self.mlp_norm(x)
        if self.mlp_type in TOKEN_SHIFT_MLPS:
            return self.mlp(h, segment_ids), _zero_stats()
        if self.mlp_type == "moe":
            return self.mlp(h, dropless=dropless)
        return self.mlp(h), _zero_stats()

    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None, *,
                 key: Array | None = None, value_residual=None):
        """``(x, stats)``; with a ``value_residual`` carry (RWKV-7 stacks) ``(x, stats, value_residual)``."""
        k1, k2 = (None, None) if key is None else jax.random.split(key)
        if value_residual is not None and self.layer_type == "rwkv7":
            h, _, value_residual = self.mixer.run(self.mixer_norm(x), segment_ids=segment_ids,
                                                  value_residual=value_residual)
        else:
            h = self._mix(self.mixer_norm(x), positions, segment_ids)
        x = x + _dropout(h, self.resid_dropout, k1)
        h, stats = self._channel(x, segment_ids=segment_ids)
        if h is not None:
            x = x + _dropout(h, self.resid_dropout, k2)
        return (x, stats) if value_residual is None else (x, stats, value_residual)

    def init_cache(self, batch: int, max_len: int, dtype: jnp.dtype) -> KVCache | RecurrentCache:
        if self.mlp_type in TOKEN_SHIFT_MLPS:  # (mixer cache, last normed token of the channel mix)
            return (self._mixer_cache(batch, max_len, dtype), jnp.zeros((batch, 1, self.mlp_norm.weight.shape[-1]), dtype))
        return self._mixer_cache(batch, max_len, dtype)

    def _mixer_cache(self, batch: int, max_len: int, dtype: jnp.dtype) -> KVCache | RecurrentCache:
        if self.layer_type in POSITIONAL_CACHES:
            return self.mixer.init_cache(batch, max_len, dtype)
        return self.mixer.init_cache(batch, dtype)

    def step(self, x: Array, positions: Array, cache, *, value_residual=None):
        """``(x, cache)``; with a ``value_residual`` carry (RWKV-7 stacks) ``(x, cache, value_residual)``."""
        if self.mlp_type in TOKEN_SHIFT_MLPS:
            cache, last = cache
            x, cache, value_residual = self._mixer_step(x, positions, cache, value_residual=value_residual)
            h = self.mlp_norm(x)
            out = (x + self.mlp(h, None, last), (cache, h[:, -1:].astype(last.dtype)))
        else:
            *out, value_residual = self._mixer_step(x, positions, cache, channel=True, value_residual=value_residual)
        return tuple(out) if value_residual is None else (*out, value_residual)

    def _mixer_step(self, x: Array, positions: Array, cache, *, channel: bool = False, value_residual=None):
        h = self.mixer_norm(x)
        if self.layer_type in POSITIONAL_CACHES:
            h, cache = self.mixer.step(h, positions, cache)
        elif value_residual is not None and self.layer_type == "rwkv7":
            h, cache, value_residual = self.mixer.run(h, cache=cache, value_residual=value_residual)
        else:
            h, cache = self.mixer.step(h, cache)
        x = x + h
        if channel:
            h, _ = self._channel(x, dropless=True)
            x = x if h is None else x + h
        return x, cache, value_residual


class Cache(eqx.Module):
    """Per-block decode caches, laid out like ``LanguageModel.blocks`` (stacked when scanned)."""

    layers: tuple


def _document_ids(segment_ids: Array | None) -> Array | None:
    """Renumber ``segment_ids`` by runs: a document starts wherever the id changes.

    The recurrent ops already reset on every change; numbering the runs gives the
    attention masks (which compare ids) the same meaning when an id is reused.
    """
    if segment_ids is None:
        return None
    return jnp.cumsum(segment_resets(segment_ids).astype(jnp.int32), axis=1)


def _stack(trees: list) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def _tree_add(a, b):
    return jax.tree.map(jnp.add, a, b)


class LanguageModel(eqx.Module):
    embed: Embedding
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    head: Array | None  # [V, D] untied LM head, None when tied to the embedding
    mtp: Any  # jla.models.mtp.MTPHead | None
    config: ModelConfig = eqx.field(static=True)

    def __init__(self, config: ModelConfig, *, key: Array):
        from jla.models.mtp import MTPHead  # mtp builds Blocks, so import it lazily

        k_embed, k_blocks, k_head, k_mtp = jax.random.split(key, 4)
        param_dtype = jnp.dtype(config.param_dtype)
        self.embed = Embedding(config.vocab_size, config.d_model, key=k_embed, std=config.initializer_range,
                               param_dtype=param_dtype, compute_dtype=config.compute_dtype)
        types = list(zip(config.resolved_layer_types, config.resolved_mlp_types))
        keys = jax.random.split(k_blocks, config.n_layers)
        layers = [Block(config, lt, mt, key=keys[i], layer_idx=i) for i, (lt, mt) in enumerate(types)]
        if config.scan_layers:
            period = layer_period(config)
            self.blocks = tuple(_stack(layers[j::period]) for j in range(period))
        else:
            self.blocks = tuple(layers)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps, param_dtype=param_dtype)
        self.head = (
            None if config.tie_embeddings
            else truncated_normal(k_head, (config.vocab_size, config.d_model), config.initializer_range, param_dtype)
        )
        self.mtp = None if config.mtp is None else MTPHead(config, key=k_mtp)
        # Module swaps, in this order. FP8 and the fused SwiGLU GEMMs each take over the projections / MLPs
        # they convert (ModelConfig rejects using both). The lean VJPs come last: apply_lean swaps exact base
        # types only, so it never reaches a module FP8 or the fused GEMMs own, and ModelConfig rejects
        # lean_vjp with fp8 (unless fp8_fused, whose FP8 MLP VJP is lean) or mlp_kernel != "xla" instead of
        # letting their MLPs silently stay non-lean. The
        # batch-sharding constraints (constrain_batch) are not swaps: they run in the forward pass whatever
        # module classes the blocks ended up with.
        if config.fp8 != "off":
            from jla.layers.fp8 import apply_fp8  # FP8 projections and MLPs (jla.layers.fp8)

            self.blocks, self.mtp = apply_fp8((self.blocks, self.mtp), config.fp8, amax_history=config.fp8_amax_history,
                                              min_features=config.fp8_min_features, fused=config.fp8_fused,
                                              weight_once=config.fp8_weight_once)
        if config.mlp_kernel == "mosaic_gpu":
            from jla.layers.fused_mlp import apply_fused_mlp  # fused SwiGLU GEMMs (jla.ops.mosaic_gpu.fused_gemm)

            self.blocks, self.mtp = apply_fused_mlp((self.blocks, self.mtp), recompute=config.remat == "full")
        if config.lean_vjp:
            from jla.layers.lean import apply_lean  # norms, SwiGLU, convs and gates that save only inputs

            self.blocks, self.final_norm, self.mtp = apply_lean((self.blocks, self.final_norm, self.mtp),
                                                                remat_core=config.lean_remat_core)
        self.config = config

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"head": ("vocab", "embed")}

    # Helpers ---------------------------------------------------------------------------
    @property
    def compute_dtype(self) -> jnp.dtype:
        return jnp.dtype(self.config.compute_dtype)

    def lm_head_weight(self) -> Array:
        """``[V, D]`` output projection (the embedding table when tied)."""
        return self.embed.weight if self.head is None else self.head

    def logits(self, hidden: Array) -> Array:
        """``hidden [..., D]`` -> float32 logits ``[..., V]`` (generation and evaluation)."""
        dtype = self.compute_dtype
        w = self.lm_head_weight().astype(dtype)
        return lax.dot_general(hidden.astype(dtype), w, (((hidden.ndim - 1,), (1,)), ((), ())),
                               preferred_element_type=jnp.float32)

    def _value_residual(self, batch: int, length: int):
        """The empty RWKV-7 value-residual carry, or ``None`` when the stack has no ``rwkv7`` layer."""
        if "rwkv7" not in self.config.resolved_layer_types:
            return None
        from jla.layers.rwkv7 import ValueResidual

        return ValueResidual.empty(batch, length, self.config.d_model, self.compute_dtype)

    def _block_fn(self):
        def run(block, x, positions, segment_ids, key, value_residual=None):
            # The residual stream keeps its batch split over the data axes under a mesh (FSDP shards the
            # weights along embed over the same axis; see jla.distributed.activation_sharding).
            out = block(constrain_batch(x), positions, segment_ids, key=key, value_residual=value_residual)
            out = (constrain_batch(out[0]), *out[1:])
            return out if value_residual is not None else (*out, None)

        policy = remat_policy(self.config.remat, self.config.remat_names, self.config.remat_offload_names)
        if policy is None:
            return run
        # Inside lax.scan, CSE across iterations cannot defeat the checkpoint. A scan of one iteration (a
        # layer pattern that never repeats, such as a dense first layer before MoE layers) is inlined by
        # XLA, and there CSE merged the recomputed forward with the original one: every layer's
        # activations stayed live (kimi-linear-1p3b asked for 121 GiB). Such stacks prevent CSE as
        # unscanned ones do.
        repeats = self.config.n_layers // layer_period(self.config) if self.config.scan_layers else 1
        return eqx.filter_checkpoint(run, policy=policy, prevent_cse=repeats < 2)

    def _layer_keys(self, key: Array | None) -> Array | None:
        return None if key is None else jax.random.split(key, self.config.n_layers)

    # Forward ---------------------------------------------------------------------------
    def __call__(
        self,
        input_ids: Array,
        *,
        positions: Array | None = None,
        segment_ids: Array | None = None,
        key: Array | None = None,
        inference: bool = False,
    ) -> LMOutput:
        """``input_ids [B, T]``; ``positions [B, T]`` (default ``arange``) feed RoPE.

        ``segment_ids [B, T]`` packs documents: a new document starts wherever the
        id changes along ``T``, and no token attends to, convolves with or carries
        recurrent state from another document. ``key`` drives dropout unless
        ``inference``.
        """
        cfg = self.config
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T]; got {input_ids.shape}")
        batch, length = input_ids.shape
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
        if positions.shape != (batch, length):
            raise ValueError(f"positions must be [B, T] = {(batch, length)}; got {positions.shape}")
        if segment_ids is not None and segment_ids.shape != (batch, length):
            raise ValueError(f"segment_ids must be [B, T] = {(batch, length)}; got {segment_ids.shape}")
        segment_ids = _document_ids(segment_ids)
        uses_dropout = not inference and (cfg.embed_dropout > 0.0 or cfg.resid_dropout > 0.0)
        if uses_dropout and key is None:
            raise ValueError("dropout is enabled: pass key= (or inference=True)")
        if not uses_dropout:
            key = None
        k_embed, k_layers = (None, None) if key is None else jax.random.split(key)

        x = constrain_batch(_dropout(self.embed(input_ids), cfg.embed_dropout, k_embed))
        layer_keys = self._layer_keys(k_layers)
        block_fn = self._block_fn()
        stats = _zero_stats()
        residual = self._value_residual(batch, length)  # RWKV-7's v_first, threaded through the blocks
        if cfg.scan_layers:
            period = len(self.blocks)
            params, static = eqx.partition(self.blocks, eqx.is_array)
            xs_keys = None if layer_keys is None else layer_keys.reshape(-1, period, *layer_keys.shape[1:])
            # RWKV-7 stacks that start with an RWKV-7 layer: v_first enters the scan as a constant (saved
            # once under remat) instead of riding in the carry (saved once per layer).
            v_first = None
            if residual is not None and cfg.resolved_layer_types[0] == "rwkv7":
                from jla.layers.rwkv7 import hoisted_value_residual

                first = eqx.combine(jax.tree.map(lambda a: a[0], params[0]), static[0])
                v_first = hoisted_value_residual(first, x, segment_ids, residual.v_first.dtype)
                residual = residual._replace(v_first=jnp.zeros((), v_first.dtype))

            def body(carry, xs):
                x, acc, residual = carry
                if v_first is not None:
                    residual = residual._replace(v_first=v_first)
                group_params, group_keys = xs
                for j in range(period):
                    block = eqx.combine(group_params[j], static[j])
                    x, s, residual = block_fn(block, x, positions, segment_ids,
                                              None if group_keys is None else group_keys[j], residual)
                    acc = _tree_add(acc, s)
                if v_first is not None:
                    residual = residual._replace(v_first=jnp.zeros((), v_first.dtype))
                return (x, acc, residual), None

            (x, stats, _), _ = lax.scan(body, (x, stats, residual), (params, xs_keys))
        else:
            for i, block in enumerate(self.blocks):
                x, s, residual = block_fn(block, x, positions, segment_ids,
                                          None if layer_keys is None else layer_keys[i], residual)
                stats = _tree_add(stats, s)

        num_moe = sum(t == "moe" for t in cfg.resolved_mlp_types)
        metrics: dict[str, Array] = {}
        aux = jnp.zeros((), jnp.float32)
        if num_moe:
            stats = jax.tree.map(lambda v: v / num_moe, stats)  # mean over MoE layers
            aux = stats.aux_loss
            metrics = {
                "moe/balance_loss": stats.balance_loss,
                "moe/router_z_loss": stats.z_loss,
                "moe/dropped_fraction": stats.dropped_fraction,
            }
        return LMOutput(hidden=self.final_norm(x), pre_norm_hidden=x, aux_loss=aux, metrics=metrics)

    # Multi-token prediction -----------------------------------------------------------------
    def mtp_loss(
        self,
        out: LMOutput,
        input_ids: Array,
        targets: Array,
        *,
        positions: Array | None = None,
        segment_ids: Array | None = None,
        aux_loss_cap: float = 0.0,
        objective: Any = None,
    ) -> tuple[Array, dict[str, Array]]:
        """Weighted MTP loss from the trunk output; ``(0, {})`` when the model has no MTP.

        ``targets`` are the next-token targets of the main head (``-100`` = ignored);
        depth ``d`` at position ``t`` is trained on ``targets[t + d]``. For packed
        documents pass the forward pass's ``positions`` and ``segment_ids``: the
        refiner attention stays inside each document and a label counts only when
        positions ``t`` and ``t + d`` share a document (so with document-final
        targets set to ``-100``, as for the main loss, no label crosses a boundary).
        ``aux_loss_cap > 0`` clips finite MoE refiner aux losses to ``+-cap``, like the trunk's.
        ``objective`` replaces the heads' cross entropy (:meth:`jla.models.mtp.MTPHead.loss`).
        """
        if self.mtp is None:
            return jnp.zeros((), jnp.float32), {}
        batch, length = input_ids.shape
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
        return self.mtp.loss(
            out.pre_norm_hidden, input_ids, targets, embed=self.embed, head_weight=self.lm_head_weight(),
            positions=positions, segment_ids=_document_ids(segment_ids), block_fn=self._mtp_block_fn(),
            aux_loss_cap=aux_loss_cap, objective=objective,
        )

    def _mtp_block_fn(self):
        def run(block, x, positions, segment_ids):
            return block(constrain_batch(x), positions, segment_ids)  # (x, MoE stats of an "moe" refiner block)

        policy = remat_policy(self.config.remat, self.config.remat_names, self.config.remat_offload_names)
        return run if policy is None else eqx.filter_checkpoint(run, policy=policy)

    # Decoding ----------------------------------------------------------------------------
    def init_cache(self, batch: int, max_len: int) -> Cache:
        """Empty caches for ``batch`` rows of up to ``max_len`` positions."""
        dtype = self.compute_dtype
        if self.config.scan_layers:
            repeats = self.config.n_layers // len(self.blocks)
            layers = tuple(_stack([b.init_cache(batch, max_len, dtype)] * repeats) for b in self.blocks)
        else:
            layers = tuple(b.init_cache(batch, max_len, dtype) for b in self.blocks)
        return Cache(layers=layers)

    def decode_step(self, cache: Cache, input_ids: Array, positions: Array) -> tuple[Array, Cache]:
        """Run ``input_ids [B, T_new]`` at ``positions [B, T_new]`` through the caches.

        Works for prompt prefill (``T_new > 1``) and single-token decoding; returns
        float32 logits ``[B, T_new, V]`` and the updated cache.
        """
        if input_ids.shape != positions.shape or input_ids.ndim != 2:
            raise ValueError(f"input_ids and positions must both be [B, T]; got {input_ids.shape}, {positions.shape}")
        x = self.embed(input_ids)
        positions = positions.astype(jnp.int32)
        residual = self._value_residual(*input_ids.shape)

        def step(block, x, layer_cache, residual):
            out = block.step(x, positions, layer_cache, value_residual=residual)
            return out if residual is not None else (*out, None)

        if self.config.scan_layers:
            period = len(self.blocks)
            params, static = eqx.partition(self.blocks, eqx.is_array)

            def body(carry, xs):
                x, residual = carry
                group_params, group_cache = xs
                new = []
                for j in range(period):
                    x, c, residual = step(eqx.combine(group_params[j], static[j]), x, group_cache[j], residual)
                    new.append(c)
                return (x, residual), tuple(new)

            (x, _), layers = lax.scan(body, (x, residual), (params, cache.layers))
        else:
            layers = []
            for block, layer_cache in zip(self.blocks, cache.layers):
                x, c, residual = step(block, x, layer_cache, residual)
                layers.append(c)
            layers = tuple(layers)
        return self.logits(self.final_norm(x)), Cache(layers=layers)


def build_model(config: ModelConfig, *, key: Array) -> LanguageModel:
    """Initialize a :class:`LanguageModel` (parameters in ``config.param_dtype``)."""
    return LanguageModel(config, key=key)


# Remat policies built from checkpoint names (ModelConfig.remat = "kernels" / "dots_kernels" / "names" /
# "offload"). The names are set in the layers: "attention" on every mixer's core output, "linear" on
# every Linear and StackedLinear projection, "mlp_hidden" on the SwiGLU gate/up product. "kernels" is not a name but the
# kernel launches themselves: Pallas kernels (the Mosaic GPU kernels) are opaque to XLA, so their
# outputs (the mixer output, chunk states and the other kernel-computed tensors a custom VJP saves) are
# kept instead of launching the forward kernels again in the backward pass. Saved tensors that XLA
# computes (the kernel inputs, cumulative gates) are recomputed unless another name keeps them. Under
# jax.checkpoint a custom VJP's primal call and its forward rule are separate launches, so a kernel whose
# primal call drops outputs still runs twice, both times in the forward pass (benchmarks/fp8_1p3b.md).
_KERNEL_PRIMITIVES = frozenset({"pallas_call", "core_map", "mpmd_map"})


def _sub_jaxprs(params: dict) -> Any:
    """The jaxprs (``Jaxpr`` or ``ClosedJaxpr``) among a primitive's parameters."""
    for value in params.values():
        for sub in value if isinstance(value, (tuple, list)) else (value,):
            if hasattr(getattr(sub, "jaxpr", sub), "eqns"):
                yield sub


def _launches_kernel(jaxpr: Any) -> bool:
    """Whether ``jaxpr`` launches a kernel, directly or inside a nested call."""
    for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        if eqn.primitive.name in _KERNEL_PRIMITIVES or any(map(_launches_kernel, _sub_jaxprs(eqn.params))):
            return True
    return False


def _save_kernel_outputs(prim, *_, **params) -> bool:
    if prim.name in _KERNEL_PRIMITIVES:
        return True
    # plgpu.kernel wraps every kernel launch in a custom_vmap, which an op may wrap again (the attention
    # kernels run inside custom_partitioning, which batch-shards them under SPMD jit).
    return any(map(_launches_kernel, _sub_jaxprs(params)))


def _named_policy(names: tuple[str, ...], offload_names: tuple[str, ...]) -> Any:
    policies = jax.checkpoint_policies
    kept = tuple(n for n in names if n != "kernels")
    if offload_names:
        policy = policies.save_and_offload_only_these_names(
            names_which_can_be_saved=list(kept), names_which_can_be_offloaded=list(offload_names),
            offload_src="device", offload_dst="pinned_host",
        )
    else:
        policy = policies.save_only_these_names(*kept)
    if "attention" in names:
        policy = _or_save(policy, _save_cudnn_attention)
    if "kernels" in names:
        policy = _or_save(policy, _save_kernel_outputs)
    return policy


def _or_save(policy: Any, extra: Any) -> Any:
    """``policy`` plus whatever the boolean policy ``extra`` saves (``policy`` may offload, which
    ``jax.checkpoint_policies.save_from_both_policies`` rejects)."""

    def combined(prim, *args, **params):
        return True if extra(prim, *args, **params) else policy(prim, *args, **params)

    return combined


def residual_branches(config: ModelConfig) -> int:
    """Residual branches of the trunk: one per token mixer plus one per channel mixer (``mlp_type != "none"``).

    With ``init_scheme="upstream"`` Mamba-2/3 ``out_proj`` starts divided by its square root
    (``n_residuals_per_layer * n_layer`` of state-spaces/mamba ``MixerModel`` and FLA's
    ``rescale_prenorm_residual``; the same count for a uniform stack).
    """
    return sum(1 + (mlp != "none") for mlp in config.resolved_mlp_types)
