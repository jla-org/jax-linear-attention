# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers (Apache-2.0), models/deepseek_v3/modeling_deepseek_v3.py (DeepseekV3TopkRouter),
# and DeepSeek-V3 (MIT), inference/model.py (Gate): the group-limited, bias-corrected routing in _choose.
"""Token-choice mixture of SwiGLU experts.

Routing: a softmax or sigmoid router picks ``top_k`` experts per token; the
selected gate values are optionally renormalized to sum to one. Two dispatch
paths give identical results whenever no route is dropped:

* capacity routing (``capacity_factor`` set): every expert has
  ``ceil(capacity_factor * tokens * top_k / experts)`` slots, routes claim slots
  in order with every token's first choice ahead of all second choices, and
  overflowing routes are dropped (the residual connection carries the token);
  the experts run as one batched GEMM over ``[E, C, D]``;
* dropless routing (``capacity_factor=None``): routes are sorted by expert and
  run through grouped GEMMs (``ragged_dot_impl``: by default the Mosaic GPU kernels
  on an H100 and ``jax.lax.ragged_dot`` elsewhere); under a multi-device mesh each
  device dispatches its own tokens.

Tokens move to and from their expert rows with the gathers of
:mod:`jla.ops.moe_permute` in both directions (no scatter-adds in the backward pass).

``num_shared_experts`` adds an always-on SwiGLU of width
``num_shared_experts * d_ff_expert`` (DeepSeek-MoE). The layer reports the
Switch-Transformer load-balancing loss ``E * sum_e f_e P_e`` (``f_e`` the
fraction of routes sent to expert ``e``, ``counts_e / (tokens * top_k)``, ``P_e`` its
mean router probability; DeepSeek-V3's normalization), the router z-loss
``mean(logsumexp(logits) ** 2)`` and the fraction of dropped routes. Hugging Face's
``load_balancing_loss_func`` (Mixtral, Qwen2/3-MoE, Qwen3-Next) divides the counts by
``tokens`` instead, so its loss is ``top_k`` times ours and its ``router_aux_loss_coef``
equals ``MoEConfig.aux_loss_coef / top_k`` (:func:`jla.models.config.aux_loss_coef_from_hf`). The
expert-choice ``router_bias`` (DeepSeek-V3) is applied, never updated: it serves imported
checkpoints (aux-loss-free bias updates are not implemented).

``expert_parallel_axis`` shards the experts over that mesh axis and exchanges the
routed tokens with all-to-all (:mod:`jla.distributed.expert_parallel`); routing,
losses and capacity decisions are unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jla.layers.linear import truncated_normal
from jla.layers.mlp import SwiGLU, swiglu
from jla.ops.moe_permute import combine_rows, dispatch_rows, group_counts

if TYPE_CHECKING:
    from jla.models.config import MoEConfig

__all__ = ["MoE", "MoEStats"]

Array = jax.Array


class MoEStats(NamedTuple):
    aux_loss: Array  # aux_loss_coef * balance_loss + z_loss_coef * z_loss
    balance_loss: Array
    z_loss: Array
    dropped_fraction: Array


class _Routing(NamedTuple):
    logits: Array  # [N, E] float32
    experts: Array  # [N, k] int32
    gates: Array  # [N, k] float32
    probs: Array  # [N, E] float32, the distribution used by the balance loss


class MoE(eqx.Module):
    router: Array  # [D, E]
    gate_up: Array  # [E, D, 2, F]
    down: Array  # [E, F, D]
    shared: SwiGLU | None
    cfg: MoEConfig = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    router_bias: Array | None  # [E] float32, DeepSeek-V3's expert-choice bias (cfg.router_bias)
    shared_gate: Array | None = None  # [D, 1]: sigmoid(x W) scales the shared experts (cfg.shared_expert_gate)

    def __init__(
        self,
        d_model: int,
        cfg: MoEConfig,
        *,
        key: Array,
        std: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        kr, kg, kd, ks = jax.random.split(key, 4)
        experts, d_ff = cfg.num_experts, cfg.d_ff_expert
        self.router = truncated_normal(kr, (d_model, experts), std, param_dtype)
        self.gate_up = truncated_normal(kg, (experts, d_model, 2, d_ff), std, param_dtype)
        self.down = truncated_normal(kd, (experts, d_ff, d_model), std, param_dtype)
        self.shared = (
            SwiGLU(d_model, cfg.num_shared_experts * d_ff, key=ks, std=std, param_dtype=param_dtype,
                   compute_dtype=compute_dtype)
            if cfg.num_shared_experts else None
        )
        self.cfg = cfg
        self.compute_dtype = str(jnp.dtype(compute_dtype))
        self.router_bias = jnp.zeros((experts,), jnp.float32) if cfg.router_bias else None
        # Qwen2-MoE / Qwen3-Next shared_expert_gate (nn.Linear(hidden, 1, bias=False)). fold_in(key, 4) is not
        # one of the keys above (with partitionable threefry fold_in(k, i) == split(k, n)[i], so fold_in(ks, 1)
        # would repeat the shared expert's down key): the other parameters keep their initialization.
        self.shared_gate = (truncated_normal(jax.random.fold_in(key, 4), (d_model, 1), std, param_dtype)
                            if cfg.shared_expert_gate else None)

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {
            "router": ("embed", None),
            "gate_up": ("experts", "embed", None, "mlp"),
            "down": ("experts", "mlp", "embed"),
            "router_bias": (None,),
            "shared_gate": ("embed", None),
        }

    def _shared_out(self, x: Array) -> Array:
        """The shared experts on ``x [N, D]``, scaled by ``sigmoid(x W_gate)`` when ``cfg.shared_expert_gate``."""
        y = self.shared(x)
        if self.shared_gate is None:
            return y
        dtype = jnp.dtype(self.compute_dtype)
        logit = lax.dot_general(x.astype(dtype), self.shared_gate.astype(dtype), (((1,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32)
        return (jax.nn.sigmoid(logit) * y.astype(jnp.float32)).astype(y.dtype)

    # Routing -------------------------------------------------------------------------
    def _route(self, x: Array) -> _Routing:
        dtype = jnp.dtype(self.compute_dtype)
        # Router GEMM on compute-dtype operands with float32 accumulation and logits.
        precision = None
        if self.router_bias is not None:  # DeepSeek-V3 routing: float32 operands, as transformers computes it
            dtype, precision = jnp.dtype(jnp.float32), lax.Precision.HIGHEST
        logits = lax.dot_general(x.astype(dtype), self.router.astype(dtype), (((1,), (0,)), ((), ())),
                                 precision=precision, preferred_element_type=jnp.float32)
        if self.cfg.router == "softmax":
            scores = jax.nn.softmax(logits, axis=-1)
            probs = scores
        else:
            scores = jax.nn.sigmoid(logits)
            probs = scores / jnp.sum(scores, axis=-1, keepdims=True)
        if self.router_bias is None and self.cfg.num_groups == 1:
            gates, experts = lax.top_k(scores, self.cfg.top_k)
        else:
            experts = self._choose(scores)
            gates = jnp.take_along_axis(scores, experts, axis=-1)
        if self.cfg.normalize_top_k:
            gates = gates / jnp.sum(gates, axis=-1, keepdims=True)
        if self.cfg.routed_scaling_factor != 1.0:
            gates = gates * self.cfg.routed_scaling_factor
        return _Routing(logits, experts.astype(jnp.int32), gates, probs)

    # The routing of DeepSeek-V3 (MIT) inference/model.py (Gate) and transformers (Apache-2.0)
    # models/deepseek_v3/modeling_deepseek_v3.py (DeepseekV3TopkRouter), written for JAX.
    def _choose(self, scores: Array) -> Array:
        """DeepSeek-V3 expert choice ``[N, k]``: top-k of the biased scores inside the best ``top_k_groups`` groups."""
        cfg = self.cfg
        choice = lax.stop_gradient(scores)
        if self.router_bias is not None:
            choice = choice + lax.stop_gradient(self.router_bias.astype(jnp.float32))
        if cfg.num_groups > 1:
            tokens, per_group = choice.shape[0], cfg.num_experts // cfg.num_groups
            group_scores = jnp.sum(lax.top_k(choice.reshape(tokens, cfg.num_groups, per_group), 2)[0], axis=-1)
            groups = lax.top_k(group_scores, cfg.top_k_groups)[1]
            keep = jnp.zeros((tokens, cfg.num_groups), bool).at[jnp.arange(tokens)[:, None], groups].set(True)
            choice = jnp.where(jnp.repeat(keep, per_group, axis=1), choice, -jnp.inf)
        return lax.top_k(choice, cfg.top_k)[1]

    def _stats(self, routing: _Routing, dropped_fraction: Array) -> MoEStats:
        experts = self.cfg.num_experts
        routes = routing.experts.size
        counts = group_counts(routing.experts.reshape(-1), experts).astype(jnp.float32)
        balance = experts * jnp.sum((counts / routes) * jnp.mean(routing.probs, axis=0))
        z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(routing.logits, axis=-1)))
        aux = self.cfg.aux_loss_coef * balance + self.cfg.z_loss_coef * z_loss
        return MoEStats(aux, balance, z_loss, dropped_fraction.astype(jnp.float32))

    # Dispatch --------------------------------------------------------------------------
    # Token <-> expert-row moves are gathers both ways (jla.ops.moe_permute): the transposes of
    # plain x[rows] / y[rows] gathers are scatter-adds, which ran slower than the expert GEMMs.
    def _capacity(self, x: Array, routing: _Routing) -> tuple[Array, Array]:
        tokens, d_model = x.shape
        experts, top_k = self.cfg.num_experts, self.cfg.top_k
        dtype = jnp.dtype(self.compute_dtype)
        routes = tokens * top_k
        capacity = max(1, min(routes, math.ceil(self.cfg.capacity_factor * routes / experts)))
        # Slot-major order: all first choices claim capacity before any second choice; a route's
        # position at its expert is its rank among that expert's routes in this order.
        expert = routing.experts.T.reshape(-1)
        order = jnp.argsort(expert, stable=True)
        counts = group_counts(expert, experts)
        rank = jnp.zeros((routes,), jnp.int32).at[order].set(jnp.arange(routes, dtype=jnp.int32), unique_indices=True)
        position = rank - (jnp.cumsum(counts) - counts)[expert]
        kept = position < capacity
        # Row of every route in the flat [E * C] expert buffer (E * C: dropped), and the
        # token-major route held by every row (routes: an empty slot).
        row = jnp.where(kept, expert * capacity + position, experts * capacity)
        slot_major = jnp.arange(routes, dtype=jnp.int32)
        route_tm = (slot_major % tokens) * top_k + slot_major // tokens
        dst = row.reshape(top_k, tokens).T.reshape(-1)  # token-major routes -> rows
        # Admitted rows are unique; the dropped routes all write the sink entry, sliced off.
        holder = jnp.full((experts * capacity + 1,), routes, jnp.int32).at[row].set(route_tm)[:-1]
        buffer = dispatch_rows(x.astype(dtype), holder // top_k, dst, top_k).reshape(experts, capacity, d_model)
        weight = dispatch_rows(routing.gates.reshape(-1, 1), holder, dst, 1).reshape(experts, capacity)
        h = jnp.einsum("ecd,edgf->ecgf", buffer, self.gate_up.astype(dtype))
        # The gate multiplies before the (linear) down projection: same result, smaller backward.
        a = swiglu(h) * weight[..., None].astype(dtype)
        y = jnp.einsum("ecf,efd->ecd", a, self.down.astype(dtype))
        out = combine_rows(y.reshape(experts * capacity, d_model), holder // top_k, dst, top_k)
        return out.astype(dtype), 1.0 - jnp.mean(kept.astype(jnp.float32))

    def _dropless(self, x: Array, routing: _Routing, gate_up: Array | None = None,
                  down: Array | None = None) -> tuple[Array, Array]:
        tokens, d_model = x.shape
        experts, top_k, d_ff = self.cfg.num_experts, self.cfg.top_k, self.cfg.d_ff_expert
        dtype = jnp.dtype(self.compute_dtype)
        gate_up = self.gate_up if gate_up is None else gate_up
        down = self.down if down is None else down
        routes = tokens * top_k
        expert = routing.experts.reshape(-1)  # token-major routes
        order = jnp.argsort(expert, stable=True)  # row s of the expert-sorted rows holds route order[s]
        rank = jnp.zeros((routes,), jnp.int32).at[order].set(jnp.arange(routes, dtype=jnp.int32), unique_indices=True)
        group_sizes = group_counts(expert, experts)
        xs = dispatch_rows(x.astype(dtype), order // top_k, rank, top_k)
        gate = dispatch_rows(routing.gates.reshape(-1, 1), order, rank, 1)
        w_in = gate_up.astype(dtype).reshape(experts, d_model, 2 * d_ff)
        h = self._ragged_dot(xs, w_in, group_sizes).reshape(routes, 2, d_ff)
        a = swiglu(h) * gate.astype(dtype)
        y_sorted = self._ragged_dot(a, down.astype(dtype), group_sizes)
        out = combine_rows(y_sorted, order // top_k, rank, top_k)
        return out.astype(dtype), jnp.zeros((), jnp.float32)

    def _dropless_per_device(self, x: Array, routing: _Routing) -> tuple[Array, Array]:
        """Dropless routing with each device dispatching its own tokens (``jax.shard_map``).

        The Mosaic GPU grouped GEMM is an opaque custom call: under a multi-device mesh XLA would
        gather every route to every device and run the whole layer on each. Every token's routes
        are independent, so the tokens split over the mesh like a batch and the expert weights
        enter replicated in the compute dtype (FSDP / TP shards are all-gathered; their gradients
        are summed over the mesh by the ``shard_map`` transpose), as
        :func:`jla.distributed.kernel_sharding.shard_layout_call` does for the other kernels.
        """
        if self._grouped_gemm_impl() == "xla":  # XLA partitions jax.lax.ragged_dot itself
            return self._dropless(x, routing)
        from jla.distributed.kernel_sharding import shard_layout_call

        dtype = jnp.dtype(self.compute_dtype)

        def body(x, experts, gates, gate_up, down):
            return self._dropless(x, _Routing(None, experts, gates, None), gate_up, down)[0]

        y = shard_layout_call(
            body,
            dict(x=x, experts=routing.experts, gates=routing.gates, gate_up=self.gate_up.astype(dtype),
                 down=self.down.astype(dtype)),
            dict(x="batch", experts="batch", gates="batch", gate_up="replicated", down="replicated"),
            out_layouts="batch",
        )
        return y, jnp.zeros((), jnp.float32)

    def __call__(self, x: Array, *, dropless: bool = False) -> tuple[Array, MoEStats]:
        """``x [..., D]`` -> ``(y [..., D], stats)``.

        ``dropless=True`` overrides capacity routing (decoding uses it: a capacity
        computed from a handful of tokens would drop routes the full pass keeps).
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        routing = self._route(flat)
        if self.cfg.expert_parallel_axis is not None:
            from jla.distributed.expert_parallel import expert_parallel_moe

            y, dropped = expert_parallel_moe(self, flat, routing, dropless=dropless or self.cfg.capacity_factor is None)
        elif dropless or self.cfg.capacity_factor is None:
            y, dropped = self._dropless_per_device(flat, routing)
        else:
            y, dropped = self._capacity(flat, routing)
        if self.shared is not None:
            y = y + self._shared_out(flat)
        return y.reshape(shape[:-1] + (y.shape[-1],)), self._stats(routing, dropped)

    def reference(self, x: Array) -> Array:
        """Dense oracle: every expert runs on every token, weighted by the routed gates (no drops)."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        routing = self._route(flat)
        dtype = jnp.dtype(self.compute_dtype)
        combine = jnp.zeros((flat.shape[0], self.cfg.num_experts), jnp.float32)
        combine = combine.at[jnp.arange(flat.shape[0])[:, None], routing.experts].set(routing.gates)
        h = jnp.einsum("nd,edgf->negf", flat.astype(dtype), self.gate_up.astype(dtype))
        y = jnp.einsum("nef,efd->ned", swiglu(h), self.down.astype(dtype))
        out = jnp.einsum("ne,ned->nd", combine, y.astype(jnp.float32)).astype(dtype)
        if self.shared is not None:
            out = out + self._shared_out(flat)
        return out.reshape(shape[:-1] + (out.shape[-1],))

    def _ragged_dot(self, lhs: Array, rhs: Array, group_sizes: Array) -> Array:
        """Grouped GEMM of dropless routing by ``cfg.ragged_dot_impl`` (every row is in a group).

        ``"mosaic_gpu"`` and ``"auto"`` run the Mosaic GPU kernels where they support the shapes and
        ``jax.lax.ragged_dot`` elsewhere (other devices, small batches, unaligned widths; ``"auto"``
        falls back without a warning).
        """
        if self._grouped_gemm_impl() == "xla":
            return lax.ragged_dot(lhs, rhs, group_sizes)
        from jla.ops.mosaic_gpu import ragged_dot as mosaic_ragged_dot

        return mosaic_ragged_dot.grouped_matmul(lhs, rhs, group_sizes, impl=self.cfg.ragged_dot_impl,
                                                full_groups=True)

    def _grouped_gemm_impl(self) -> str:
        """``cfg.ragged_dot_impl`` with ``"auto"`` resolved: ``"mosaic_gpu"`` on an H100, else ``"xla"``."""
        impl = self.cfg.ragged_dot_impl
        if impl == "auto":
            from jla.ops.mosaic_gpu._common import is_hopper

            return "mosaic_gpu" if is_hopper() else "xla"
        return impl
