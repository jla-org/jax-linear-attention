# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped-query softmax attention with QK-norm, sliding windows, an output gate and KV caches.

Training and full-sequence calls use ``jax.nn.dot_product_attention`` or our FA3-style
Mosaic GPU kernel. With ``implementation="auto"`` the backend is chosen at call time:
cuDNN flash attention on GPU for bf16/fp16 inputs whose geometry cuDNN accepts, as
long as one device's query shard has at most ``CUDNN_MAX_QUERY_ELEMENTS`` (2**31)
elements; past that the Mosaic GPU kernel on Hopper (when it supports the call) and
XLA otherwise. Packed documents (``segment_ids``) run the Mosaic GPU kernel on Hopper
when it accepts the operands (it reads the ids per tile); elsewhere they become a
boolean mask, which cuDNN consumes as an additive bias. Sliding windows use cuDNN's
native left window.
Cached prefill and decode build an explicit position mask over the cache and
always run the XLA implementation.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from jla.layers.cache import KVCache
from jla.layers.linear import Linear
from jla.layers.norm import RMSNorm
from jla.layers.rope import apply_rope
from jla.ops.moba import moba_attention, moba_cached_mask

if TYPE_CHECKING:  # layers never import jla.models at runtime (it imports layers)
    from jla.models.config import AttentionConfig

__all__ = ["Attention", "attention", "batch_partition", "resolve_attention_implementation"]

Array = jax.Array

# Name of the attention output for ``jax.checkpoint_policies.save_only_these_names``.
ATTENTION_CHECKPOINT_NAME = "attention"


def _cuda_compute_capability() -> tuple[int, int] | None:
    """``(major, minor)`` of the default CUDA device, or ``None`` when it is not one."""
    if jax.default_backend() != "gpu":
        return None
    capability = getattr(jax.devices()[0], "compute_capability", None)
    if capability is None:
        return None
    major, _, minor = str(capability).partition(".")
    return int(major), int(minor or 0)


def resolve_attention_implementation(requested: str, q: Array, k: Array, *, v: Array | None = None,
                                     segment_ids: Array | None = None) -> str:
    """Pick ``"cudnn"``, ``"mosaic_gpu"`` or ``"xla"`` for this call (never cached on the module).

    An explicit ``"cudnn"``, ``"xla"`` or ``"mosaic_gpu"`` is returned unchanged.

    ``"auto"`` with packed documents (``segment_ids``) on Hopper selects the Mosaic GPU kernel
    whenever it accepts the operands (bf16/fp16, head dim 64 or 128, any length): it reads the ids
    per tile and skips blocks across documents, where cuDNN would get a dense ``[B, 1, T, T]`` mask.
    Otherwise ``"auto"`` selects cuDNN only on CUDA GPUs, for bf16/fp16 inputs with a head dim
    that is a multiple of 8 (at most 256 on Hopper and later, 128 before), even
    query and key lengths (the cuDNN backward rejects odd lengths) and at most
    ``CUDNN_MAX_QUERY_ELEMENTS`` query elements per device (see
    :func:`query_elements_per_device`). Past that limit it selects the Mosaic GPU
    kernel on Hopper when that kernel supports the call, and XLA otherwise.
    """
    if requested in ("cudnn", "xla", "mosaic_gpu"):
        return requested
    if requested != "auto":
        raise ValueError(f"unknown attention implementation {requested!r}")
    capability = _cuda_compute_capability()
    if capability is None or q.dtype not in (jnp.bfloat16, jnp.float16):
        return "xla"
    if segment_ids is not None and capability[0] == 9:
        from jla.ops.mosaic_gpu.attention import supported

        if supported(q, k, k if v is None else v) is None:
            return "mosaic_gpu"
    head_dim = q.shape[-1]
    max_head_dim = 256 if capability >= (9, 0) else 128
    if head_dim % 8 or head_dim > max_head_dim or q.shape[1] % 2 or k.shape[1] % 2:
        return "xla"
    if query_elements_per_device(q)[0] > CUDNN_MAX_QUERY_ELEMENTS:  # cuDNN's backward is silently wrong here
        if capability[0] == 9:
            from jla.ops.mosaic_gpu.attention import supported

            if supported(q, k, k if v is None else v) is None:
                return "mosaic_gpu"
        return "xla"
    return "cudnn"


# cuDNN as shipped with JAX 0.11.2 (9.27) returns wrong attention gradients, with no error, once one call's
# query tensor has more than 2**31 elements: right at 1,048,576 tokens x 16 heads x 128 (exactly 2**31), but at
# 1,064,960 tokens its dq was off by up to 3.5x the gradient's magnitude and dk by 12% while its forward and dv
# stayed right (benchmarks/long_context.md). The limit applies to one device's call (cuDNN is partitioned over
# the batch): past it "auto" takes the Mosaic GPU kernel (or XLA), and an explicit "cudnn" keeps its forward
# but raises when differentiated.
CUDNN_MAX_QUERY_ELEMENTS = 2**31

# Mesh axes that shard the batch dim of the activations traced inside :func:`batch_partition` (GSPMD traces
# see global shapes, so the per-device size of a query tensor is not visible in its shape).
_BATCH_PARTITION: contextvars.ContextVar = contextvars.ContextVar("jla_attention_batch_partition", default=None)


@contextlib.contextmanager
def batch_partition(mesh, axes):
    """Declare that attention calls traced inside have their batch dim ``B`` sharded over mesh ``axes``.

    The trainer enters this around its loss with the data-parallel axes of its batch sharding. The cuDNN
    size limit is then checked per device (``q.size`` divided by the product of those axis sizes, when it
    divides ``B``), and every cuDNN call whose per-device query tensor is within the limit pins
    ``q, k, v`` and its document mask to that batch sharding (other dims unconstrained). That keeps the
    per-device size cuDNN was chosen for, and gives the tensors one sharding, which JAX's cuDNN
    partitioner requires under FSDP (and a replicated mask would not fit the per-device call). ``mesh``
    needs only ``.shape`` for the size check (a concrete ``Mesh`` for the pin).
    """
    names = (axes,) if isinstance(axes, str) else tuple(a for a in (axes or ()) if a is not None)
    token = _BATCH_PARTITION.set((mesh, names) if names else None)
    try:
        yield
    finally:
        _BATCH_PARTITION.reset(token)


def _explicit_shards(x) -> int:
    """Devices that ``x``'s explicitly sharded type splits it over (1 under GSPMD's auto axes)."""
    try:
        sharding = jax.typeof(x).sharding
        mesh, spec = sharding.mesh, sharding.spec
        explicit = jax.sharding.AxisType.Explicit
        types = dict(zip(mesh.axis_names, mesh.axis_types))
    except (AttributeError, TypeError):  # no sharding in this value's type (or not a JAX value)
        return 1
    shards = 1
    for entry in spec:
        for name in entry if isinstance(entry, tuple) else (entry,):
            if isinstance(name, str) and types.get(name) == explicit:
                shards *= mesh.shape[name]
    return shards


def query_elements_per_device(q: Array) -> tuple[int, jax.sharding.NamedSharding | None]:
    """``(elements of one device's shard of q, sharding to pin q, k, v to for that or None)``.

    Explicitly sharded types give the exact count. Under GSPMD (global shapes) the count comes from the
    enclosing :func:`batch_partition`, whose returned sharding must be applied for it to hold; without
    one, the global size (an upper bound) is returned.
    """
    shards = _explicit_shards(q)
    if shards > 1:
        return q.size // shards, None
    partition = _BATCH_PARTITION.get()
    if partition is not None:
        mesh, names = partition
        auto = jax.sharding.AxisType.Auto
        types = dict(zip(mesh.axis_names, getattr(mesh, "axis_types", ())))
        shards = math.prod(mesh.shape[name] for name in names)
        # Explicit axes are authoritative in q's type (handled above); only GSPMD's auto axes need the pin.
        # Inside shard_map over these axes (manual in the current mesh) q's shape is already per device.
        current = jax.sharding.get_abstract_mesh()
        current_types = dict(zip(getattr(current, "axis_names", ()), getattr(current, "axis_types", ())))
        auto_axes = all(types.get(n, auto) == auto and current_types.get(n, auto) == auto for n in names)
        if shards > 1 and q.shape[0] % shards == 0 and auto_axes:
            free = jax.sharding.PartitionSpec.UNCONSTRAINED
            spec = jax.sharding.PartitionSpec(names if len(names) > 1 else names[0], *([free] * (q.ndim - 1)))
            return q.size // shards, jax.sharding.NamedSharding(mesh, spec)
    return q.size, None


def _cudnn_sdpa(q, k, v, mask, *, scale, window) -> Array:
    """Causal cuDNN attention as ``jax.nn.dot_product_attention(implementation="cudnn")`` computes it.

    A value head dim different from the q/k one (MLA: q/k 192, v 128) goes to JAX's cuDNN SDPA
    (``jax._src.cudnn.fused_attention_stablehlo``, cuDNN >= 9.10 on Hopper) directly: the public
    wrapper takes one head dim, so the values would otherwise have to be zero-padded.
    """
    if v.shape[-1] == q.shape[-1]:
        return jax.nn.dot_product_attention(q, k, v, mask=mask, scale=scale, is_causal=True,
                                            local_window_size=window, implementation="cudnn")
    from jax._src.cudnn import fused_attention_stablehlo as cudnn_sdpa

    return cudnn_sdpa.dot_product_attention(
        q, k, v, None, mask, scale=q.shape[-1] ** -0.5 if scale is None else scale,
        mask_type=cudnn_sdpa.MaskType.CAUSAL, sliding_window_length=None if window is None else window[0] + 1,
        qkv_layout="BTNH")


def _cudnn_forward_only(q, k, v, mask, *, scale, window, per_device: int) -> Array:
    """cuDNN attention past ``CUDNN_MAX_QUERY_ELEMENTS`` per device: the forward (measured right there) runs,
    differentiating it raises (its backward is silently wrong there)."""

    @jax.custom_vjp
    def forward(q, k, v, mask):
        return _cudnn_sdpa(q, k, v, mask, scale=scale, window=window)

    def forward_fwd(q, k, v, mask):
        return forward(q, k, v, mask), None

    def forward_bwd(_, cotangent):
        raise ValueError(f"cuDNN attention computes wrong gradients for query tensors of more than "
                         f"{CUDNN_MAX_QUERY_ELEMENTS} elements per device (got {per_device}); use "
                         "implementation='mosaic_gpu' or 'auto'")

    forward.defvjp(forward_fwd, forward_bwd)
    return forward(q, k, v, mask)


def attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    segment_ids: Array | None = None,
    sliding_window: int | None = None,
    scale: float | None = None,
    implementation: str = "auto",
    sinks: Array | None = None,
    mosaic_gpu_backward: str = "ranged",
    persistent_forward: bool | None = None,
) -> Array:
    """Causal GQA over a full sequence: ``q [B, T, H, D]``, ``k [B, T, Hkv, D]``, ``v [B, T, Hkv, Dv]``.

    ``Dv`` may differ from ``D`` (MLA) on the cuDNN path (Hopper) and the Mosaic GPU kernel.

    ``segment_ids [B, T]`` restricts every query to keys of its own document;
    ``sliding_window`` counts the visible tokens including the query itself.
    ``sinks [H]`` adds a learned per-head logit to the softmax denominator (GPT-OSS
    attention sinks); it runs the XLA path or, with ``implementation="mosaic_gpu"``, the
    Mosaic GPU kernel (``implementation="cudnn"`` raises). ``mosaic_gpu_backward`` picks the
    Mosaic GPU kernel's backward (``"ranged"`` or ``"fused"``) whenever that kernel runs
    (``"fused"`` needs ``Dv == D``) and ``persistent_forward`` its forward
    (``AttentionConfig.persistent_forward``).
    """
    if sinks is not None and implementation != "mosaic_gpu":
        return _full_sink_attention(q, k, v, sinks, segment_ids=segment_ids, sliding_window=sliding_window,
                                    scale=scale, implementation=implementation)
    impl = resolve_attention_implementation(implementation, q, k, v=v, segment_ids=segment_ids)
    forward_only, pin = False, None
    if impl == "cudnn":
        per_device, pin = query_elements_per_device(q)
        forward_only = per_device > CUDNN_MAX_QUERY_ELEMENTS
        if forward_only:
            pin = None
        # Make the per-device size cuDNN was chosen for hold, and give q, k, v and the document mask below one
        # sharding: JAX's cuDNN partitioner rejects differing q, k, v ones ("Query, key and value should have
        # same sharding"), and a mask not split like them fails to build cuDNN's graph ("dimensions mismatch").
        if pin is not None:
            q, k, v = (jax.lax.with_sharding_constraint(x, pin) for x in (q, k, v))
    if impl == "mosaic_gpu":  # FA3-style Hopper kernel: windows and documents without a dense mask
        return _mosaic_gpu_attention(q, k, v, segment_ids=segment_ids, sliding_window=sliding_window,
                                     scale=scale, sinks=sinks, backward=mosaic_gpu_backward,
                                     persistent_forward=persistent_forward)
    mask = None
    if segment_ids is not None:
        if segment_ids.shape != q.shape[:2]:
            raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
        mask = (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]
        if pin is not None:
            mask = jax.lax.with_sharding_constraint(mask, pin)
    window = None if sliding_window is None else (sliding_window - 1, 0)
    if forward_only:
        return _cudnn_forward_only(q, k, v, mask, scale=scale, window=window, per_device=per_device)
    if impl == "cudnn":
        return _cudnn_sdpa(q, k, v, mask, scale=scale, window=window)
    return jax.nn.dot_product_attention(
        q, k, v, mask=mask, scale=scale, is_causal=True, local_window_size=window, implementation=impl
    )


def _mosaic_gpu_attention(q, k, v, *, segment_ids, sliding_window, scale, sinks, backward="ranged",
                          persistent_forward=None) -> Array:
    """The FA3-style kernel, per batch and head shard of the context mesh (``jax.set_mesh``).

    A custom call XLA cannot partition: under a mesh it runs per batch and head shard
    (:func:`jla.distributed.kernel_sharding.shard_layout_call`). The sinks ``[H]`` enter as
    ``[B, H]`` rows so that they split with the heads; each shard reads its heads from its first row
    and the broadcast sums their gradient over the batch.
    """
    from jla.distributed.kernel_sharding import shard_layout_call
    from jla.ops.mosaic_gpu.attention import flash_attention, supported

    if sinks is not None:
        if sinks.shape != (q.shape[2],):
            raise ValueError(f"mosaic_gpu attention: sinks must be [H] = {(q.shape[2],)}; got {sinks.shape}")
        sinks = jnp.broadcast_to(sinks[None], (q.shape[0], q.shape[2]))

    def kernel(q, k, v, segment_ids, sinks):
        return flash_attention(q, k, v, segment_ids=segment_ids, sliding_window=sliding_window, scale=scale,
                               sinks=None if sinks is None else sinks[0], backward=backward,
                               persistent_forward=persistent_forward)

    return shard_layout_call(
        kernel,
        {"q": q, "k": k, "v": v, "segment_ids": segment_ids, "sinks": sinks},
        {"q": "token", "k": "token", "v": "token", "segment_ids": "batch", "sinks": "state"},
        out_layouts="token", heads=math.gcd(q.shape[2], k.shape[2]),
        supports=lambda q, k, v, segment_ids, sinks: supported(q, k, v) is None,
    )


def _cached_attention(q, k, v, q_positions, kv_positions, *, sliding_window, scale, sinks=None) -> Array:
    """Attention of ``q`` over cache slots, masked by stored positions."""
    kv = kv_positions[:, None, :]
    qp = q_positions[:, :, None]
    mask = (kv >= 0) & (kv <= qp)
    if sliding_window is not None:
        mask = mask & (qp - kv < sliding_window)
    if sinks is not None:
        return sink_attention(q, k, v, mask[:, None], sinks, scale=scale)
    return jax.nn.dot_product_attention(q, k, v, mask=mask[:, None], scale=scale, implementation="xla")


class Attention(eqx.Module):
    """Llama / Qwen3-style attention block mixer (separate q and kv projections).

    Optional: per-head RMSNorm on q and k (``qk_norm``), partial RoPE, a sliding
    window, and the Qwen3-Next elementwise sigmoid output gate
    ``o * sigmoid(x W_gate)`` before the output projection.
    """

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    gate_proj: Linear | None
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None
    cfg: AttentionConfig = eqx.field(static=True)
    sinks: Array | None  # [H] per-head sink logits (``cfg.sinks``), None otherwise

    def __init__(
        self,
        d_model: int,
        cfg: AttentionConfig,
        *,
        key: Array,
        std: float,
        norm_eps: float,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: str = "bfloat16",
    ):
        if cfg.moba_block_size is not None and (cfg.sinks or cfg.implementation == "mosaic_gpu"):
            raise ValueError("MoBA attention supports neither sinks nor implementation='mosaic_gpu'")
        kq, kk, kv, ko, kg = jax.random.split(key, 5)
        heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        common = dict(std=std, param_dtype=param_dtype, compute_dtype=compute_dtype)
        self.q_proj = Linear(d_model, heads * head_dim, key=kq, axes=("embed", "heads"), **common)
        self.k_proj = Linear(d_model, kv_heads * head_dim, key=kk, axes=("embed", "kv_heads"), **common)
        self.v_proj = Linear(d_model, kv_heads * head_dim, key=kv, axes=("embed", "kv_heads"), **common)
        self.o_proj = Linear(heads * head_dim, d_model, key=ko, axes=("heads", "embed"), **common)
        if cfg.qkv_bias:  # Qwen2: the same projections plus zero-initialized biases
            self.q_proj, self.k_proj, self.v_proj = (
                Linear(d_model, n * head_dim, key=k, axes=("embed", axis), use_bias=True, **common)
                for n, k, axis in ((heads, kq, "heads"), (kv_heads, kk, "kv_heads"), (kv_heads, kv, "kv_heads"))
            )
        self.gate_proj = (
            Linear(d_model, heads * head_dim, key=kg, axes=("embed", "heads"), **common)
            if cfg.output_gate else None
        )
        if cfg.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps, param_dtype=param_dtype)
        else:
            self.q_norm = self.k_norm = None
        self.cfg = cfg
        # GPT-OSS attention sinks: one learned logit per query head (zero init = one virtual key).
        self.sinks = jnp.zeros((heads,), param_dtype) if cfg.sinks else None

    def _project(self, x: Array, positions: Array) -> tuple[Array, Array, Array]:
        batch, length, _ = x.shape
        cfg = self.cfg
        from jla.layers.fp8 import shared_input_projections  # one FP8 cast of x for q, k and v (training)

        q, k, v = shared_input_projections(x, self.q_proj, self.k_proj, self.v_proj)
        q = q.reshape(batch, length, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        v = v.reshape(batch, length, cfg.num_kv_heads, cfg.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if cfg.use_rope:
            rope = dict(theta=cfg.rope_theta, rotary_fraction=cfg.rotary_fraction, scaling=cfg.rope_scaling,
                        exact_angles=cfg.rope_exact_angles)
            q, k = apply_rope(q, positions, **rope), apply_rope(k, positions, **rope)
        return q, k, v

    def _output(self, x: Array, o: Array) -> Array:
        batch, length = x.shape[:2]
        o = checkpoint_name(o.reshape(batch, length, -1), ATTENTION_CHECKPOINT_NAME)
        if self.gate_proj is not None:
            gate = jax.nn.sigmoid(self.gate_proj(x).astype(jnp.float32))
            o = (o.astype(jnp.float32) * gate).astype(o.dtype)
        return self.o_proj(o)

    def __call__(self, x: Array, positions: Array, segment_ids: Array | None = None) -> Array:
        if self.cfg.fp8_grad_epilogue:  # FP8 q/k/v gradients from the attention backward (jla.layers.fp8_attention)
            from jla.layers.fp8_attention import fused_fp8_attention

            o = fused_fp8_attention(self, x, positions, segment_ids)
            if o is not None:
                return self._output(x, o)
        q, k, v = self._project(x, positions)
        if self.cfg.context_parallel_axis is not None:  # T sharded over a mesh axis (ring or Ulysses)
            return self._output(x, self._context_parallel(q, k, v, segment_ids))
        if self.cfg.moba_block_size is not None:
            o = moba_attention(q, k, v, block_size=self.cfg.moba_block_size, top_k=self.cfg.moba_top_k,
                               segment_ids=segment_ids, scale=self.cfg.softmax_scale)
            return self._output(x, o)
        o = attention(
            q, k, v, segment_ids=segment_ids, sliding_window=self.cfg.sliding_window,
            scale=self.cfg.softmax_scale, implementation=self.cfg.implementation, sinks=self.sinks,
            mosaic_gpu_backward=self.cfg.mosaic_gpu_backward, persistent_forward=self.cfg.persistent_forward,
        )
        return self._output(x, o)

    def init_cache(self, batch: int, max_len: int, dtype: jnp.dtype) -> KVCache:
        cfg = self.cfg
        ring = cfg.sliding_window is not None and cfg.sliding_window < max_len
        size = cfg.sliding_window if ring else max_len
        return KVCache.create(batch, size, cfg.num_kv_heads, cfg.head_dim, dtype, ring=ring)

    def step(self, x: Array, positions: Array, cache: KVCache) -> tuple[Array, KVCache]:
        """Prefill or decode ``x [B, T, D]`` at ``positions [B, T]`` against ``cache``."""
        if self.cfg.moba_block_size is not None:  # MoBA forbids sliding windows: never a ring cache
            q, k, v = self._project(x, positions)
            cache = cache.write(k, v, positions)
            mask = moba_cached_mask(q, cache.keys, positions, cache.positions,
                                    block_size=self.cfg.moba_block_size, top_k=self.cfg.moba_top_k)
            o = jax.nn.dot_product_attention(q.astype(cache.keys.dtype), cache.keys, cache.values, mask=mask,
                                             scale=self.cfg.softmax_scale, implementation="xla")
            return self._output(x, o.astype(q.dtype)), cache
        q, k, v = self._project(x, positions)
        common = dict(sliding_window=self.cfg.sliding_window, scale=self.cfg.softmax_scale, sinks=self.sinks)
        if cache.ring:
            # Attend before writing: a long block can overwrite slots its first queries need.
            keys = jnp.concatenate([cache.keys, k.astype(cache.keys.dtype)], axis=1)
            values = jnp.concatenate([cache.values, v.astype(cache.values.dtype)], axis=1)
            kv_pos = jnp.concatenate([cache.positions, positions.astype(jnp.int32)], axis=1)
            o = _cached_attention(q.astype(keys.dtype), keys, values, positions, kv_pos, **common)
            cache = cache.write(k, v, positions)
        else:
            cache = cache.write(k, v, positions)
            o = _cached_attention(q.astype(cache.keys.dtype), cache.keys, cache.values, positions,
                                  cache.positions, **common)
        return self._output(x, o.astype(q.dtype)), cache

    def param_axes(self) -> dict[str, tuple[str | None, ...]]:
        return {"sinks": ("heads",)}

    def _context_parallel(self, q: Array, k: Array, v: Array, segment_ids: Array | None) -> Array:
        from jla.distributed.context_parallel_attention import context_parallel_attention

        cfg = self.cfg
        return context_parallel_attention(
            q, k, v, axis=cfg.context_parallel_axis, batch_axis="auto", strategy=cfg.context_parallel_strategy,
            load_balance=cfg.context_parallel_load_balance, segment_ids=segment_ids,
            sliding_window=cfg.sliding_window, scale=cfg.softmax_scale, implementation=cfg.implementation,
            sinks=self.sinks,
        )


# Attention sinks (GPT-OSS) ---------------------------------------------------------------------


def sink_attention(q: Array, k: Array, v: Array, mask: Array, sinks: Array, *, scale: float | None) -> Array:
    """GQA softmax attention with a per-head sink logit in the denominator (XLA, float32 softmax).

    ``q [B, T, H, D]``, ``k, v [B, S, Hkv, D]``, ``mask [B, 1 | H, T, S]`` (True = visible),
    ``sinks [H]``: ``o = sum_j exp(s_j) v_j / (exp(sink_h) + sum_j exp(s_j))``. Query head
    ``h`` reads key head ``h // (H / Hkv)``, as in ``jax.nn.dot_product_attention``.
    """
    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    if heads % kv_heads or sinks.shape != (heads,):
        raise ValueError(f"need H % Hkv == 0 and sinks [H]; got H={heads}, Hkv={kv_heads}, {sinks.shape}")
    group = heads // kv_heads
    scale = head_dim ** -0.5 if scale is None else scale
    f32 = jnp.float32
    qg = q.reshape(batch, length, kv_heads, group, head_dim).astype(f32)
    s = jnp.einsum("btkgd,bskd->bkgts", qg, k.astype(f32)) * scale
    s = s.reshape(batch, heads, length, k.shape[1])
    s = jnp.where(mask, s, -jnp.inf)
    sink = sinks.astype(f32)[None, :, None, None]
    m = jax.lax.stop_gradient(jnp.maximum(jnp.max(s, axis=-1, keepdims=True), sink))
    p = jnp.exp(s - m)
    p = p / (jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(sink - m))
    p = p.reshape(batch, kv_heads, group, length, k.shape[1])
    o = jnp.einsum("bkgts,bskd->btkgd", p, v.astype(f32)).reshape(batch, length, heads, head_dim)
    return o.astype(q.dtype)


def _full_sink_attention(q, k, v, sinks, *, segment_ids, sliding_window, scale, implementation) -> Array:
    if implementation not in ("auto", "xla"):
        raise ValueError("attention sinks run on the XLA path or the Mosaic GPU kernel; use "
                         "implementation='auto', 'xla' or 'mosaic_gpu'")
    length = q.shape[1]
    i = jnp.arange(length)[:, None]
    j = jnp.arange(length)[None, :]
    allowed = j <= i
    if sliding_window is not None:
        allowed = allowed & (i - j < sliding_window)
    mask = allowed[None, None]
    if segment_ids is not None:
        if segment_ids.shape != q.shape[:2]:
            raise ValueError(f"segment_ids must be [B, T] = {q.shape[:2]}; got {segment_ids.shape}")
        mask = mask & (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]
    return sink_attention(q, k, v, mask, sinks, scale=scale)
