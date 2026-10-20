# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model configuration: frozen dataclasses shared by layers, models and the trainer.

A model is a stack of ``n_layers`` blocks. Each block has a token mixer
(``layer_types[i]``: softmax ``"attention"``, one of the linear-attention kinds,
or a state-space mixer ``"mamba2"`` / ``"mamba3"``) followed by a channel mixer
(``mlp_types[i]``: ``"dense"`` SwiGLU, ``"moe"``, or ``"none"`` for pure Mamba
stacks). Hybrid models such as Qwen3-Next (three Gated DeltaNet layers per
attention layer), Kimi Linear (KDA plus attention) or the Gated DeltaNet-2
hybrid (GDN-2 and sliding-window attention) are expressed with ``layer_types``;
``hybrid_pattern`` builds such tuples.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Literal

AttentionImpl = Literal["auto", "cudnn", "xla", "mosaic_gpu"]
LinearAttentionKind = Literal[
    "gated_deltanet", "kda", "gla", "deltanet", "simple_gla", "gdn2",
    "gated_deltaproduct", "mlstm", "rwkv6", "hgrn2", "ttt_linear",
]
LinearAttentionImpl = Literal["auto", "reference", "chunked", "mosaic_gpu"]
LayerType = Literal[
    "attention", "gated_deltanet", "kda", "gla", "deltanet", "simple_gla", "gdn2", "mamba2", "mamba3",
    "gated_deltaproduct", "mlstm", "rwkv6", "hgrn2", "ttt_linear", "mamba1",
    "local_attention", "mla", "fox", "path", "nsa", "dsa",
    "rwkv7", "lightning",
]
MlpType = Literal["dense", "moe", "none", "relu2", "rwkv_cmix", "rwkv7_cmix"]
RematPolicy = Literal[
    "none", "full", "dots", "dots_with_no_batch_dims", "save_attention",
    "kernels", "dots_kernels", "names", "offload",  # named-checkpoint policies (jla.models.lm.remat_policy)
]

LINEAR_ATTENTION_KINDS: tuple[str, ...] = (
    "gated_deltanet", "kda", "gla", "deltanet", "simple_gla", "gdn2",
    "gated_deltaproduct",
    "mlstm",
    "rwkv6", "hgrn2",  # RWKV-6 and HGRN2 mixers (jla.layers.rwkv6) read the linear-attention config
    "ttt_linear",  # TTT-Linear (jla.layers.ttt)
    "rwkv7",  # RWKV-7 time mix (jla.layers.rwkv7.RWKV7Mixer)
    "lightning",  # MiniMax-01 lightning attention (jla.layers.lightning_attention)
)
SSM_KINDS: tuple[str, ...] = ("mamba2", "mamba3", "mamba1")
SPARSE_ATTENTION_KINDS: tuple[str, ...] = ("nsa", "dsa")


@dataclass(frozen=True)
class AttentionConfig:
    """Softmax attention with grouped KV heads (GQA) and rotary positions."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float = 10_000.0
    rotary_fraction: float = 1.0  # fraction of head_dim that is rotated (partial RoPE)
    rope_exact_angles: bool = False  # reduce RoPE angles mod 2 pi (exact past ~256K positions; layers.rope)
    qk_norm: bool = True  # per-head RMSNorm on q and k (Qwen3, Gemma 3)
    output_gate: bool = False  # sigmoid gate on the attention output (Qwen3-Next)
    sliding_window: int | None = None  # tokens visible to each query, None = global
    softmax_scale: float | None = None  # default head_dim ** -0.5
    # "auto" on GPU: Mosaic GPU for packed documents on Hopper, else cuDNN up to 2**31 query elements per
    # device, then Mosaic GPU/XLA (layers.attention.resolve_attention_implementation)
    implementation: AttentionImpl = "auto"
    use_rope: bool = True  # False = no positional encoding (NoPE, as in Kimi Linear's global layers)
    moba_block_size: int | None = None  # MoBA block-sparse attention with this block size; None = dense
    moba_top_k: int = 3  # MoBA: blocks each query attends, counting its own block
    sinks: bool = False  # learned per-head logit in the softmax denominator (GPT-OSS); XLA or mosaic_gpu
    qkv_bias: bool = False  # learned biases on the q, k and v projections (Qwen2 / Qwen2.5)
    rope_scaling: RoPEScalingConfig | None = None  # RoPE frequency scaling (Llama 3.1+); None = plain RoPE
    # Shard T over this mesh axis of the context mesh (jla.distributed.context_parallel_attention); None = off.
    context_parallel_axis: str | None = None
    context_parallel_strategy: Literal["ring", "ulysses"] = "ring"  # ring attention or Ulysses all-to-all
    context_parallel_load_balance: bool = True  # ring, global causal: zigzag layout (equal work per device)
    # Backward of the Mosaic GPU kernel (ops.mosaic_gpu.attention.flash_attention): "ranged" (dq and dk, dv
    # kernels, bitwise reproducible) or "fused" (one kernel, dq added up with fp32 TMA reductions)
    mosaic_gpu_backward: Literal["ranged", "fused"] = "ranged"
    # FP8 (delayed scaling with the train step's weight copies): the Mosaic GPU attention backward kernels write
    # the q/k/v projections' e5m2 gradients and their amaxes in their epilogues (RoPE backward included), which
    # the projection GEMMs read directly (jla.layers.fp8_attention; other calls run the plain path)
    fp8_grad_epilogue: bool = False
    # Forward of the Mosaic GPU kernel from one persistent CTA per SM with a static tile schedule
    # (jla.ops.mosaic_gpu.attention_persistent): None = wherever it applies, True = required (raises where it
    # does not apply), False = one CTA per query tile
    persistent_forward: bool | None = None

    def __post_init__(self) -> None:
        if self.mosaic_gpu_backward not in ("ranged", "fused"):
            raise ValueError(f"unknown mosaic_gpu_backward {self.mosaic_gpu_backward!r}")
        if self.persistent_forward not in (None, True, False):
            raise ValueError(f"persistent_forward must be None, true or false; got {self.persistent_forward!r}")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if not 0.0 < self.rotary_fraction <= 1.0:
            raise ValueError("rotary_fraction must be in (0, 1]")
        if self.rope_exact_angles and self.rope_scaling is not None:
            raise ValueError("rope_exact_angles supports plain RoPE only; it cannot be combined with rope_scaling")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.implementation not in ("auto", "cudnn", "xla", "mosaic_gpu"):
            raise ValueError(f"unknown attention implementation {self.implementation!r}")
        if self.moba_block_size is not None:
            if self.moba_block_size < 1 or self.moba_top_k < 1:
                raise ValueError("moba_block_size and moba_top_k must be >= 1")
            if self.sliding_window is not None:
                raise ValueError("MoBA attention does not support sliding_window")
        if self.context_parallel_strategy not in ("ring", "ulysses"):
            raise ValueError(f"unknown context_parallel_strategy {self.context_parallel_strategy!r}")
        if self.context_parallel_axis is not None:
            if self.moba_block_size is not None:
                raise ValueError("context parallelism (context_parallel_axis) does not support MoBA attention")
            if self.context_parallel_strategy == "ring" and self.implementation in ("cudnn", "mosaic_gpu"):
                raise ValueError("ring attention runs blockwise XLA attention; "
                                 "use context_parallel_strategy='ulysses' for cuDNN or Mosaic GPU")


@dataclass(frozen=True)
class LinearAttentionConfig:
    """Shared settings for the linear-attention token mixers.

    ``num_v_heads`` may exceed ``num_k_heads`` (grouped value heads, as in
    Qwen3-Next): q and k heads are then shared by groups of value heads.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_size: int = 4  # depthwise causal conv on q, k, v; 0 disables it
    output_gate: bool = True  # gated RMSNorm on the mixer output
    qk_l2norm: bool = True  # L2-normalize q and k per head (delta-rule kinds)
    gate_low_rank_dim: int | None = None  # low-rank per-channel gate projection (KDA, GLA)
    chunk_size: int = 64
    implementation: LinearAttentionImpl = "auto"
    gate_logit_normalizer: float = 16.0  # GLA / simple GLA: g = logsigmoid(x W_g) / normalizer
    allow_neg_eigval: bool = False  # delta-rule kinds: write strength in [0, 2] (GDN-2: the erase gate)
    # Shard T over this mesh axis of the context mesh (jla.distributed.context_parallel); None = off.
    context_parallel_axis: str | None = None
    num_householder: int = 1  # gated_deltaproduct: delta-rule (Householder) steps per token (FLA name)
    gate_soft_cap: float | None = 15.0  # mlstm: cap * tanh(x / cap) on the input/forget gate pre-activations
    ttt_mini_batch_size: int = 16  # TTT-Linear: tokens per inner gradient step
    ttt_base_lr: float = 1.0  # TTT-Linear: eta = base_lr * sigmoid(x W_lr + b_lr) / head_dim
    rwkv7_lora_dims: tuple[int, int, int, int] | None = None  # RWKV-7 (decay, a, v, gate) ranks; None = FLA's
    # Delta-rule kinds with qk_l2norm: materialize the conv outputs q, k before the op normalizes them, so XLA
    # cannot fuse the three short convolutions into the norm's reduction (same values up to bf16 rounding).
    # It matters where the convs run on XLA (conv_impl = "xla", or "auto" where the kernels decline: off
    # Hopper, float32, several devices): there, on an H100, 1.11x end to end for gdn-hd128-1p3b, 1.10x
    # kda-1p3b, 1.12x gdn2-1p3b (benchmarks/zero_copy_layouts.md). XLA cannot fuse the Mosaic GPU conv
    # kernels either, so next to them the barrier is redundant and the two speedups do not add up.
    # False lets XLA fuse the convolutions into the reduction.
    qk_conv_barrier: bool = True
    # Short conv on q, k, v: "auto" (the fused Mosaic GPU kernels of jla.ops.mosaic_gpu.short_conv on Hopper,
    # per shard on several devices; XLA where they decline), "mosaic_gpu" (the kernels or an error) or "xla".
    conv_impl: str = "auto"
    # Delta-rule kinds with qk_l2norm: L2-normalize q and k inside the short-conv kernel when it runs
    # (jla.layers.conv_l2norm) and call the op with use_qk_l2norm=False; False keeps the norm in the op.
    qk_l2norm_in_conv: bool = True
    # Starting weights of the q/k/v short convs: "normal" = N(0, initializer_range), FLA's init of every
    # nn.Conv1d (ShortConvolution, no bias); "identity" = current-token tap 1, older taps 0 (a fresh mixer
    # then behaves like its conv-free variant). None = ModelConfig.init_scheme decides: "jla" (and a mixer
    # built on its own) keeps the identity; "upstream" draws "normal" for gated_deltanet, deltanet,
    # gated_deltaproduct and kda (as FLA) and keeps the identity for gla, simple_gla and gdn2 (its paper
    # states no conv init). An explicit value wins over the scheme.
    conv_init: str | None = None

    def __post_init__(self) -> None:
        if self.conv_init not in (None, "normal", "identity"):
            raise ValueError(f"unknown conv_init {self.conv_init!r}; expected None, 'normal' or 'identity'")
        if self.context_parallel_axis is not None and self.implementation not in ("auto", "chunked"):
            raise ValueError("context_parallel_axis runs the chunked ops; implementation must be 'auto' or 'chunked'")
        if self.gate_soft_cap is not None and self.gate_soft_cap <= 0:
            raise ValueError("gate_soft_cap must be positive or None")
        if self.num_v_heads % self.num_k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.conv_size < 0:
            raise ValueError("conv_size must be >= 0")
        if self.chunk_size < 16 or self.chunk_size % 16:
            raise ValueError("chunk_size must be a positive multiple of 16 (the chunked ops' sub-chunk)")
        if self.implementation not in ("auto", "reference", "chunked", "mosaic_gpu"):
            raise ValueError(f"unknown linear-attention implementation {self.implementation!r}")
        if self.gate_logit_normalizer <= 0:
            raise ValueError("gate_logit_normalizer must be positive")
        if not isinstance(self.num_householder, int) or self.num_householder < 1:
            raise ValueError("num_householder must be a positive int")
        if self.conv_impl not in ("auto", "mosaic_gpu", "xla"):
            raise ValueError(f"unknown conv_impl {self.conv_impl!r}; expected 'auto', 'mosaic_gpu' or 'xla'")


@dataclass(frozen=True)
class SSMConfig:
    """Mamba-2 / Mamba-3 mixer settings; the defaults are state-spaces/mamba's.

    The inner width is ``expand * d_model`` split into heads of ``head_dim``; B and
    C have ``n_groups`` groups, each shared by ``num_heads / n_groups`` heads.
    """

    d_state: int = 128
    expand: int = 2
    head_dim: int = 64
    n_groups: int = 1
    conv_size: int = 4  # Mamba-2: depthwise causal conv on x, B, C (Mamba-3 has none)
    a_init_range: tuple[float, float] = (1.0, 16.0)  # Mamba-2: A = -U(a_init_range)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_floor: float = 1e-4  # Mamba-3: A = min(-heavy_tail(x W_A), -a_floor)
    rope_fraction: float = 0.5  # Mamba-3: rotated fraction of d_state (0.5 or 1.0)
    outproj_norm: bool = False  # Mamba-3: rmsnorm(y) * silu(z) per head instead of y * silu(z)
    chunk_size: int = 64
    implementation: LinearAttentionImpl = "auto"
    log_linear_levels: int = 0  # Mamba-2 only: > 0 runs log-linear Mamba-2 with this many Fenwick levels
    dt_rank: int | None = None  # Mamba-1: rank of the dt projection, None = ceil(d_model / 16)
    dt_scale: float = 1.0  # Mamba-1: dt_proj weights ~ U(+-dt_scale / sqrt(dt_rank))
    mimo_rank: int = 1  # Mamba-3: 1 = SISO; R > 1 = MIMO of rank R (jla.layers.mamba3_mimo)
    # Mamba-2 on the Mosaic GPU kernels: sum dB, dC over each group's heads inside the backward kernel
    # (jla.ops.mosaic_gpu.scalar_gate_grouped) instead of writing them per head for XLA to sum (about 1.03x
    # end to end, 1.02x and 1.04x in two A/B rounds, and 0.2-1.7 GiB less peak memory for mamba2-1p3b on an
    # H100, benchmarks/zero_copy_layouts.md). False restores the per-head kernel.
    grouped_key_grads: bool = True
    # Mamba-2 short conv on x, B, C: "auto" (fused Mosaic GPU kernels on Hopper, per shard on several devices;
    # XLA where they decline), "mosaic_gpu" or "xla".
    conv_impl: str = "auto"
    # Mamba-2 training calls on GPUs (one B/C group, power-of-two heads and head dim, a bias-free bf16 or FP8
    # out-projection): the D skip, gate and gated RMSNorm as Pallas Triton row kernels in one custom VJP with the
    # out-projection (jla.layers.mamba2_fused; benchmarks/end_to_end_1p3b.md); on several GPUs the row kernels run
    # per shard of the trainer's mesh. Other calls, and False, run the unfused layer.
    fused_glue: bool = True

    def __post_init__(self) -> None:
        if self.mimo_rank < 1:
            raise ValueError(f"mimo_rank must be >= 1; got {self.mimo_rank}")
        if min(self.d_state, self.expand, self.head_dim, self.n_groups) < 1 or self.conv_size < 1:
            raise ValueError("d_state, expand, head_dim, n_groups and conv_size must be >= 1")
        if not 0 < self.a_init_range[0] <= self.a_init_range[1]:
            raise ValueError("a_init_range must satisfy 0 < low <= high")
        if not 0 < self.dt_min <= self.dt_max or self.dt_init_floor <= 0 or self.a_floor <= 0:
            raise ValueError("need 0 < dt_min <= dt_max and positive dt_init_floor and a_floor")
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0 (as in Mamba-3)")
        if self.chunk_size < 16 or self.chunk_size % 16:
            raise ValueError("chunk_size must be a positive multiple of 16")
        if self.implementation not in ("auto", "reference", "chunked", "mosaic_gpu"):
            raise ValueError(f"unknown SSM implementation {self.implementation!r}")
        if self.conv_impl not in ("auto", "mosaic_gpu", "xla"):
            raise ValueError(f"unknown conv_impl {self.conv_impl!r}; expected 'auto', 'mosaic_gpu' or 'xla'")

    def num_heads(self, d_model: int) -> int:
        inner = self.expand * d_model
        if inner % self.head_dim:
            raise ValueError(f"expand * d_model = {inner} must be a multiple of head_dim {self.head_dim}")
        heads = inner // self.head_dim
        if heads % self.n_groups:
            raise ValueError(f"{heads} heads cannot be split into {self.n_groups} B/C groups")
        return heads


@dataclass(frozen=True)
class MoEConfig:
    """Token-choice mixture of SwiGLU experts."""

    num_experts: int
    top_k: int
    d_ff_expert: int
    num_shared_experts: int = 0
    router: Literal["softmax", "sigmoid"] = "softmax"
    normalize_top_k: bool = True  # renormalize the selected gate weights
    capacity_factor: float | None = None  # None = dropless
    # Load-balancing loss weight. The loss is Switch / DeepSeek-V3's E * sum_e f_e P_e with f_e the fraction of
    # routes (counts / (tokens * top_k)); Hugging Face's load_balancing_loss_func (Mixtral, Qwen2/3-MoE,
    # Qwen3-Next, MiniMax) divides the same counts by tokens, so its router_aux_loss_coef c equals
    # aux_loss_coef = top_k * c here (aux_loss_coef_from_hf; jla.convert maps it that way). DeepSeek-V3's
    # aux_loss_alpha uses our normalization and maps 1:1.
    aux_loss_coef: float = 1e-2  # load-balancing loss
    z_loss_coef: float = 0.0  # router logit z-loss
    # Grouped GEMM of dropless routing: "auto" = the Mosaic GPU kernels on H100 where they take the
    # shapes, jax.lax.ragged_dot elsewhere; "mosaic_gpu" = the same, warning on a fallback; "xla".
    ragged_dot_impl: Literal["auto", "xla", "mosaic_gpu"] = "auto"
    # DeepSeek-V3 routing (jla.convert imports it): ``router_bias`` adds a per-expert bias to the
    # scores for the top-k choice only (the gates stay unbiased; the bias gets no gradient and no
    # weight decay). It is for imported checkpoints only: nothing updates it during training
    # (DeepSeek-V3's aux-loss-free bias updates are not implemented), so from scratch it stays zero.
    # ``num_groups`` > 1 limits each token's experts to the ``top_k_groups`` groups with the largest
    # sum of their two best biased scores; the routed output is scaled by ``routed_scaling_factor``.
    router_bias: bool = False
    num_groups: int = 1
    top_k_groups: int = 1
    routed_scaling_factor: float = 1.0
    # Expert parallelism (jla.distributed.expert_parallel): experts sharded over this axis of the
    # context mesh, tokens exchanged with all-to-all; None = off.
    expert_parallel_axis: str | None = None
    # Per-peer dispatch block of ceil(f * local routes / ep) rows (a full block keeps its routes
    # slot-major and drops the rest, second choices first); None = worst-case blocks that never
    # drop (equal to the layer without EP).
    expert_parallel_buffer_factor: float | None = None
    shared_expert_gate: bool = False  # sigmoid(x W) [D -> 1] scales the shared experts (Qwen2-MoE, Qwen3-Next)

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("MoE needs 1 <= top_k <= num_experts")
        if self.num_groups < 1 or self.num_experts % self.num_groups or not 1 <= self.top_k_groups <= self.num_groups:
            raise ValueError("MoE groups need num_experts % num_groups == 0 and 1 <= top_k_groups <= num_groups")
        if self.num_groups > 1 and (self.num_experts // self.num_groups < 2
                                    or self.top_k > self.top_k_groups * (self.num_experts // self.num_groups)):
            raise ValueError("group-limited routing needs >= 2 experts per group and top_k <= the experts "
                             "of the top_k_groups chosen groups")
        if not self.routed_scaling_factor > 0:
            raise ValueError("routed_scaling_factor must be positive")
        if self.router not in ("softmax", "sigmoid"):
            raise ValueError(f"unknown router {self.router!r}")
        if self.capacity_factor is not None and self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive (or None for dropless routing)")
        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0")
        if self.ragged_dot_impl not in ("auto", "xla", "mosaic_gpu"):
            raise ValueError(f"unknown ragged_dot_impl {self.ragged_dot_impl!r}")
        if self.ragged_dot_impl == "mosaic_gpu" and self.capacity_factor is not None and self.expert_parallel_axis is None:
            raise ValueError("ragged_dot_impl applies to dropless routing only: set capacity_factor=None")
        # Expert parallelism (its grouped GEMMs serve capacity routing too).
        if self.expert_parallel_axis is not None and not (isinstance(self.expert_parallel_axis, str) and self.expert_parallel_axis):
            raise ValueError("expert_parallel_axis must be a mesh axis name or None")
        if self.expert_parallel_buffer_factor is not None and (
            self.expert_parallel_axis is None or not self.expert_parallel_buffer_factor > 0
        ):
            raise ValueError("expert_parallel_buffer_factor must be positive and needs expert_parallel_axis")
        if self.shared_expert_gate and self.num_shared_experts < 1:
            raise ValueError("shared_expert_gate needs num_shared_experts >= 1")


@dataclass(frozen=True)
class MTPConfig:
    """Multi-token prediction heads trained next to the main LM head."""

    depth: int = 1
    kind: Literal["deepseek", "chained", "independent"] = "deepseek"
    loss_weight: float = 0.3
    vocab_chunk: int = 4096  # vocabulary tile of the MTP cross entropy
    mlp_type: Literal["dense", "moe"] = "dense"  # channel mixer of the refiner blocks (Qwen3-Next: "moe")

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("MTP depth must be >= 1 (use mtp=None to disable MTP)")
        if self.kind not in ("deepseek", "chained", "independent"):
            raise ValueError(f"unknown MTP kind {self.kind!r}")
        if self.mlp_type not in ("dense", "moe"):
            raise ValueError(f"MTP refiner blocks take mlp_type 'dense' or 'moe'; got {self.mlp_type!r}")
        if self.kind == "chained" and self.mlp_type != "dense":
            raise ValueError("chained MTP has no refiner blocks, so mlp_type must stay 'dense'")


@dataclass(frozen=True)
class MLAConfig:
    """Multi-head latent attention (DeepSeek-V2/V3, Kimi K2, Kimi Linear); defaults are DeepSeek-V3's head shape.

    Queries come from a low-rank ``q_lora_rank`` bottleneck (``None`` = one full
    projection, as in Kimi K2 and Kimi Linear); keys and values from a compressed
    ``kv_lora_rank`` latent plus one ``qk_rope_head_dim`` RoPE key shared by all
    heads. ``use_rope=False`` leaves the rope channels unrotated (NoPE, Kimi
    Linear). The softmax scale defaults to ``qk_head_dim ** -0.5``, multiplied by
    YaRN's ``mscale ** 2`` with ``mscale = 0.1 * yarn_mscale * ln(yarn_factor) + 1``
    when ``yarn_factor > 1`` (only the scale: ``rope_scaling`` interpolates the RoPE frequencies).
    """

    num_heads: int
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10_000.0
    use_rope: bool = True
    sliding_window: int | None = None  # tokens visible to each query, None = global
    softmax_scale: float | None = None  # overrides the default (and the YaRN correction)
    yarn_factor: float = 1.0
    yarn_mscale: float = 1.0
    # "auto": the Mosaic GPU kernel on Hopper for the head shapes it supports (q/k 192 with v 128, or equal
    # q/k/v head dims of 64 or 128), else cuDNN flash attention on GPU (unpadded heads in cuDNN's MLA mode on
    # Hopper with cuDNN >= 9.10, heads zero-padded to one width otherwise) and XLA elsewhere
    implementation: AttentionImpl = "auto"
    rope_scaling: RoPEScalingConfig | None = None  # RoPE frequency scaling (YaRN in DeepSeek-V3); None = plain RoPE

    def __post_init__(self) -> None:
        if min(self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim) < 1:
            raise ValueError("num_heads, kv_lora_rank, qk_nope_head_dim and v_head_dim must be >= 1")
        if self.q_lora_rank is not None and self.q_lora_rank < 1:
            raise ValueError("q_lora_rank must be >= 1 (or None for a full-rank query projection)")
        if self.qk_rope_head_dim < 2 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.yarn_factor < 1.0 or self.yarn_mscale <= 0.0:
            raise ValueError("yarn_factor must be >= 1 and yarn_mscale > 0")
        if self.implementation not in ("auto", "cudnn", "xla", "mosaic_gpu"):
            raise ValueError(f"unknown attention implementation {self.implementation!r}")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        if self.softmax_scale is not None:
            return self.softmax_scale
        mscale = 0.1 * self.yarn_mscale * math.log(self.yarn_factor) + 1.0 if self.yarn_factor > 1.0 else 1.0
        return self.qk_head_dim ** -0.5 * mscale * mscale


@dataclass(frozen=True)
class ForgettingAttentionConfig:
    """Forgetting Transformer (FoX) attention: softmax attention with a per-head forget gate.

    The defaults are the paper's "Pro" block: QK-norm, a sigmoid output gate, a
    per-head output RMSNorm and the KV shift, without positional encoding.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    qk_norm: bool = True  # per-head RMSNorm on q and k
    output_gate: bool = True  # o * sigmoid(x W_g) before the output projection
    output_norm: bool = True  # per-head RMSNorm of the attention output
    kv_shift: bool = True  # data-dependent token shift of k and v (one mixing gate per KV head)
    use_rope: bool = False  # FoX needs no positional encoding; True adds RoPE on q and k
    rope_theta: float = 10_000.0
    sliding_window: int | None = None  # tokens visible to each query, None = global
    softmax_scale: float | None = None  # default head_dim ** -0.5
    implementation: Literal["xla"] = "xla"  # the only path so far (no fused kernel yet)

    def __post_init__(self) -> None:
        if min(self.num_heads, self.num_kv_heads, self.head_dim) < 1:
            raise ValueError("num_heads, num_kv_heads and head_dim must be >= 1")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.implementation != "xla":
            raise ValueError(f"unknown forgetting-attention implementation {self.implementation!r}")


@dataclass(frozen=True)
class PathAttentionConfig:
    """PaTH attention (``layer_types`` entry ``"path"``): softmax attention with Householder positions.

    FLA's ``PaTHAttention`` defaults: a rank-32 ``w`` projection (``None`` = full,
    FLA's choice for grouped KV heads), a 3-tap SiLU conv on ``w``, no forget gate
    and no QK-norm. ``implementation`` is ``"chunked"`` (``"auto"``) or ``"reference"``.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    w_low_rank_dim: int | None = 32
    w_conv_size: int = 3  # 0 disables the conv on w
    forget_gate: bool = False  # FoX-style log forget gate per query head (PaTH-FoX)
    qk_norm: bool = False
    softmax_scale: float | None = None  # default head_dim ** -0.5
    chunk_size: int = 64
    implementation: Literal["auto", "chunked", "reference"] = "auto"

    def __post_init__(self) -> None:
        if min(self.num_heads, self.num_kv_heads, self.head_dim) < 1:
            raise ValueError("num_heads, num_kv_heads and head_dim must be >= 1")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.w_low_rank_dim is not None and self.w_low_rank_dim < 1:
            raise ValueError("w_low_rank_dim must be >= 1 (or None for a full projection)")
        if self.w_conv_size < 0:
            raise ValueError("w_conv_size must be >= 0")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.implementation not in ("auto", "chunked", "reference"):
            raise ValueError(f"unknown PaTH attention implementation {self.implementation!r}")


@dataclass(frozen=True)
class SparseAttentionConfig:
    """Native Sparse Attention (``"nsa"``) and DeepSeek Sparse Attention (``"dsa"``) settings.

    Both use ``ModelConfig.attention`` for heads, head_dim, RoPE and QK-norm.
    NSA: ``block_size`` tokens per compressed / selected block, ``block_count``
    selected blocks per query, a ``window_size`` sliding window (0 disables it).
    DSA: ``top_k`` keys per query chosen by a lightning indexer with ``index_heads``
    heads of ``index_head_dim`` (RoPE on ``index_rotary_fraction`` of it), trained by
    the indexer KL loss with weight ``indexer_loss_coef`` (0 gives the indexer no
    gradient; its weights stay in the optimizer, so weight decay still applies).
    """

    block_size: int = 64
    block_count: int = 16
    window_size: int = 512
    top_k: int = 2048
    index_heads: int = 64
    index_head_dim: int = 128
    index_rotary_fraction: float = 0.5
    indexer_loss_coef: float = 1.0
    # DSA kernels: "auto" (the kernels on a Hopper GPU when they take the shapes and attention.implementation
    # is not "xla", else xla), "mosaic_gpu"
    # (the Hopper indexer, top-k, selected-key attention and indexer-backward kernels of
    # jla.ops.mosaic_gpu.dsa; bf16/fp16 compute, head dims 64/128; unsupported shapes raise) or "xla"
    # (plain JAX, dense [T, T] math)
    dsa_implementation: str = "auto"
    # DSA kernels only: select the keys with the APPROXIMATE FP8 lightning indexer (float8_e4m3fn index queries
    # scaled per token and head, keys per sequence, as DeepSeek-V3.2 runs it in FP8; see benchmarks/dsa.md)
    dsa_index_fp8: bool = False

    def __post_init__(self) -> None:
        if min(self.block_size, self.block_count, self.top_k, self.index_heads, self.index_head_dim) < 1:
            raise ValueError("block_size, block_count, top_k, index_heads and index_head_dim must be >= 1")
        if self.window_size < 0:
            raise ValueError("window_size must be >= 0 (0 disables the NSA sliding window)")
        if not 0.0 < self.index_rotary_fraction <= 1.0:
            raise ValueError("index_rotary_fraction must be in (0, 1]")
        if self.indexer_loss_coef < 0:
            raise ValueError("indexer_loss_coef must be >= 0")
        if self.dsa_implementation not in ("xla", "mosaic_gpu", "auto"):
            raise ValueError(f"unknown dsa_implementation {self.dsa_implementation!r} "
                             "(use 'xla', 'mosaic_gpu' or 'auto')")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    d_ff: int
    attention: AttentionConfig
    linear_attention: LinearAttentionConfig | None = None
    moe: MoEConfig | None = None
    mtp: MTPConfig | None = None
    layer_types: tuple[LayerType, ...] | None = None  # None = all "attention"
    mlp_types: tuple[MlpType, ...] | None = None  # None = all "moe" if moe else all "dense"
    ssm: SSMConfig | None = None  # settings of "mamba2" / "mamba3" layers
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    remat: RematPolicy = "full"
    scan_layers: bool = True  # scan over repeating layer groups (fast compile at depth)
    embed_dropout: float = 0.0
    resid_dropout: float = 0.0
    # Settings of "local_attention" layers: a second softmax-attention config for models that
    # interleave sliding-window and global layers (GPT-OSS, Gemma 3).
    local_attention: AttentionConfig | None = None
    mla: MLAConfig | None = None  # settings of "mla" (multi-head latent attention) layers
    forgetting_attention: ForgettingAttentionConfig | None = None  # settings of "fox" layers
    path_attention: PathAttentionConfig | None = None  # settings of "path" (PaTH attention) layers
    sparse_attention: SparseAttentionConfig | None = None  # settings of "nsa" / "dsa" layers
    # FP8 training of the dense projections and MLPs (jla.layers.fp8; H100-class GPUs): "off",
    # "delayed" (per-tensor scales from an amax history kept in the model) or "current".
    fp8: Literal["off", "delayed", "current"] = "off"
    fp8_amax_history: int = 1024  # optimizer steps of amax history behind each delayed scale
    # Checkpoint names for remat = "names" (kept on the device) and "offload" (remat_offload_names go to
    # pinned host memory between the passes): "attention" (mixer core outputs), "linear" (every Linear and
    # StackedLinear projection; not the MoE experts), "mlp_hidden" (the SwiGLU gate/up product) and, kept
    # on the device only, "kernels" (every Pallas / Mosaic GPU kernel's outputs, the fused short conv of
    # conv_impl and the fused SwiGLU GEMMs of mlp_kernel included; most forward kernels still launch twice
    # under it, see benchmarks/fp8_1p3b.md). Where the fused SwiGLU GEMMs run, their gate/up product is a
    # kernel output: "kernels" keeps it, "mlp_hidden" does not, and their down projection carries no "linear".
    remat_names: tuple[str, ...] = ()
    remat_offload_names: tuple[str, ...] = ()
    # SwiGLU MLPs (dense layers, MoE shared experts, MTP refiner blocks): "xla" or "mosaic_gpu"
    # (jla.layers.fused_mlp: SwiGLU fused into the gate/up GEMM epilogue and its backward into the down
    # projection's dgrad epilogue; calls the kernels do not support, such as decode steps, CPU or
    # multi-device calls, run the XLA path). Under a remat policy other than "none" and "full" that does
    # not keep "kernels" (such as "dots"), the gate/up kernel runs again in the backward pass.
    mlp_kernel: Literal["xla", "mosaic_gpu"] = "xla"
    # FP8 only for projections whose input and output widths are both >= this (and multiples of 16):
    # smaller GEMMs (gates, low-rank factors, dt / B-C projections) cost more in quantize and
    # transpose passes than FP8 saves, so they keep the compute dtype (benchmarks/fp8_1p3b.md).
    fp8_min_features: int = 16
    # Delayed scaling only: SwiGLU MLPs run their activation, FP8 casts, amaxes and the transposed FP8
    # copies for the weight gradients in two fused passes (jla.layers.fp8_swiglu); same state, and in
    # a jitted train step on a GPU the values of fp8_fused = false up to GEMM accumulation order.
    fp8_fused: bool = False
    # Memory-lean custom VJPs (jla.layers.lean): RMSNorms, gated norms, SwiGLU + down projection, short
    # convolutions, the KDA / GDN-2 decay gates and the Mamba-2/3 SSD epilogues save only their inputs
    # for the backward pass instead of their float32 intermediates (same forward values). For
    # remat = "none" (benchmarks/fp8_1p3b.md); "dots" already drops those intermediates.
    # The Mosaic GPU short-conv kernels (conv_impl) and the fused conv + q/k L2-norm kernels
    # (LinearAttentionConfig.qk_l2norm_in_conv) are custom VJPs that already keep only their inputs, so
    # they keep running under lean_vjp and lean_remat_core: the lean short-conv VJP is for the
    # convolutions that run on XLA, and the op then skips its own q/k L2 norm as without lean_vjp.
    # Not combined with mlp_kernel = "mosaic_gpu" or with fp8 without fp8_fused yet (config errors, see __post_init__).
    lean_vjp: bool = False
    # With lean_vjp: the Gated DeltaNet, KDA, GDN-2 and Mamba-3 mixers rerun their op (kernels and
    # its XLA prologue) in the backward pass instead of keeping its residuals (chunk states, WY
    # blocks and, where the op normalizes q and k itself, their float32 L2-norm copies): a forward
    # kernel launch per layer for ~0.1-0.35 GiB per layer and 4096-token sequence.
    lean_remat_core: bool = False
    # Delayed scaling: the train step quantizes every FP8 weight (and its transposed copy for the forward GEMM)
    # once per optimizer step, outside the micro-step loop and the rematerialized forward, instead of in every
    # forward call (jla.training.fp8.prepare_fp8_weights). Same values; false quantizes per call as before.
    fp8_weight_once: bool = True
    # Starting weights. "upstream" (the default; chosen per family by the 125M FineWeb-Edu A/B,
    # benchmarks/init_parity.md): the reference implementations' choices where they differ: Mamba-2/3 (SISO, MIMO)
    # out_proj U(-1/sqrt(inner), 1/sqrt(inner)) / sqrt(residual branches) (state-spaces/mamba, FLA), the q/k/v
    # short convs of gated_deltanet, deltanet, gated_deltaproduct and kda N(0, initializer_range) (FLA), and
    # RWKV-7's zero o_proj, channel-mix value and LoRA downs with orthogonal (gain 0.1) LoRA ups (FLA,
    # BlinkDL). "jla": jla's original initialisation (bit-identical to it for the same key). An explicit
    # LinearAttentionConfig.conv_init wins over the scheme: the presets with gated_deltanet or kda layers set
    # "identity" (FLA's convs tied for the Gated DeltaNet hybrid and lost for the KDA hybrid in that A/B), except
    # the all-linear ones (there FLA's convs won) and the tiny test presets (FLA's convs, for test coverage).
    init_scheme: Literal["jla", "upstream"] = "upstream"

    def __post_init__(self) -> None:
        if self.init_scheme not in ("jla", "upstream"):
            raise ValueError(f"unknown init_scheme {self.init_scheme!r}; expected 'jla' or 'upstream'")
        if self.layer_types is not None and len(self.layer_types) != self.n_layers:
            raise ValueError("layer_types must have n_layers entries")
        if self.mlp_types is not None and len(self.mlp_types) != self.n_layers:
            raise ValueError("mlp_types must have n_layers entries")
        layer_types = set(self.resolved_layer_types)
        unknown = layer_types - {
            "attention", "local_attention", "mla", "fox", "path",
            *LINEAR_ATTENTION_KINDS, *SSM_KINDS, *SPARSE_ATTENTION_KINDS,
        }
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if set(self.resolved_mlp_types) - {"dense", "moe", "none", "relu2", "rwkv_cmix", "rwkv7_cmix"}:
            raise ValueError(
                "mlp_types must be 'dense', 'moe', 'relu2', 'rwkv_cmix', 'rwkv7_cmix' or 'none'; "
                f"got {self.resolved_mlp_types}"
            )
        if self.linear_attention is None and layer_types & set(LINEAR_ATTENTION_KINDS):
            raise ValueError("linear-attention layers need linear_attention settings")
        if self.ssm is None and layer_types & set(SSM_KINDS):
            raise ValueError("mamba2 / mamba3 layers need ssm settings")
        if self.ssm is not None and layer_types & set(SSM_KINDS):
            self.ssm.num_heads(self.d_model)  # validates the head and group split
        mtp_moe = self.mtp is not None and self.mtp.mlp_type == "moe"
        if self.moe is None and ("moe" in self.resolved_mlp_types or mtp_moe):
            raise ValueError("moe layers (and MoE MTP refiner blocks) need moe settings")
        dense_mtp = self.mtp is not None and not mtp_moe
        if self.d_ff < 1 and ({"dense", "rwkv_cmix", "rwkv7_cmix"} & set(self.resolved_mlp_types) or dense_mtp):
            raise ValueError("dense MLP layers (and MTP refiner blocks) need d_ff >= 1")
        if self.remat not in RematPolicy.__args__:
            raise ValueError(f"unknown remat policy {self.remat!r}")
        for name in ("param_dtype", "compute_dtype"):
            dtype = getattr(self, name)
            if dtype not in ("float32", "bfloat16", "float16"):
                raise ValueError(f"{name} must be float32, bfloat16 or float16; got {dtype!r}")
        if not (0.0 <= self.embed_dropout < 1.0 and 0.0 <= self.resid_dropout < 1.0):
            raise ValueError("dropout rates must be in [0, 1)")
        if self.local_attention is None and "local_attention" in layer_types:
            raise ValueError("local_attention layers need local_attention settings")
        if self.d_ff < 1 and "relu2" in self.resolved_mlp_types:
            raise ValueError("relu2 MLP layers need d_ff >= 1")
        if self.mla is None and "mla" in layer_types:
            raise ValueError("mla layers need mla settings")
        if self.forgetting_attention is None and "fox" in layer_types:
            raise ValueError("fox layers need forgetting_attention settings")
        if self.path_attention is None and "path" in layer_types:
            raise ValueError("path layers need path_attention settings")
        if self.sparse_attention is None and layer_types & set(SPARSE_ATTENTION_KINDS):
            raise ValueError("nsa / dsa layers need sparse_attention settings")
        if self.fp8 not in ("off", "delayed", "current"):
            raise ValueError(f"fp8 must be 'off', 'delayed' or 'current'; got {self.fp8!r}")
        if self.fp8_amax_history < 1:
            raise ValueError("fp8_amax_history must be >= 1")
        if self.fp8_min_features < 1:
            raise ValueError("fp8_min_features must be >= 1")
        if self.fp8_fused and self.fp8 != "delayed":
            raise ValueError("fp8_fused needs fp8 = 'delayed'")
        if self.fp8 != "off" and self.compute_dtype == "float16":
            raise ValueError("fp8 needs compute_dtype bfloat16 or float32 (FP8 scales underflow float16)")
        softmax_cp = {name: getattr(self, name).context_parallel_axis
                      for name in ("attention", "local_attention") if name in layer_types}
        if len(set(softmax_cp.values())) > 1:  # else the unsharded layers would gather T silently
            raise ValueError(f"attention and local_attention layers must set the same context_parallel_axis; "
                             f"got {softmax_cp}")
        _check_remat_names(self.remat, self.remat_names, self.remat_offload_names)
        if self.mlp_kernel not in ("xla", "mosaic_gpu"):
            raise ValueError(f"mlp_kernel must be 'xla' or 'mosaic_gpu'; got {self.mlp_kernel!r}")
        if self.mlp_kernel != "xla" and self.fp8 != "off":
            raise ValueError("mlp_kernel = 'mosaic_gpu' runs bf16/fp16 GEMMs; it cannot be combined with fp8")
        # FP8 (jla.layers.fp8) and the fused SwiGLU GEMMs of mlp_kernel (jla.layers.fused_mlp) take over the
        # SwiGLU MLPs before apply_lean runs (jla.models.lm), and apply_lean swaps exact base types only, so
        # with either of them lean_vjp would silently leave those MLPs non-lean: explicit errors until they
        # combine (benchmarks/fp8_1p3b.md: apply_fused_mlp before apply_lean, FusedSwiGLU recompute=False).
        # With fp8_fused the FP8 SwiGLU MLPs run jla.layers.fp8_swiglu, whose VJP keeps only its inputs' FP8
        # copies, the bf16 gate/up product and the FP8 activation (no float32 intermediates, as LeanSwiGLU), so
        # lean_vjp may take the norms, convolutions, gates and mixer cores (benchmarks/fp8_1p3b.md); an
        # MLP the fused path cannot take keeps the unfused FP8 path, which apply_fp8 logs.
        if self.lean_vjp and self.fp8 != "off" and not self.fp8_fused:
            raise ValueError("lean_vjp combines with fp8 only with fp8_fused = true (the unfused FP8 SwiGLU keeps "
                             "float32 intermediates; both replace the SwiGLU module)")
        if self.lean_vjp and self.mlp_kernel != "xla":
            raise ValueError("lean_vjp does not combine with mlp_kernel = 'mosaic_gpu' yet (both replace the "
                             "SwiGLU module's VJP); set one of them")
        if self.lean_remat_core and not self.lean_vjp:
            raise ValueError("lean_remat_core needs lean_vjp = true")

    @property
    def resolved_layer_types(self) -> tuple[str, ...]:
        return self.layer_types or ("attention",) * self.n_layers

    @property
    def resolved_mlp_types(self) -> tuple[str, ...]:
        if self.mlp_types is not None:
            return self.mlp_types
        return ("moe" if self.moe is not None else "dense",) * self.n_layers

    def replace(self, **changes) -> "ModelConfig":
        return dataclasses.replace(self, **changes)


def hybrid_pattern(n_layers: int, mixer: LayerType, attention_every: int) -> tuple[LayerType, ...]:
    """``attention_every - 1`` ``mixer`` layers, then one attention layer, repeated.

    ``hybrid_pattern(8, "gated_deltanet", 4)`` gives three Gated DeltaNet
    layers and one attention layer, twice (the Qwen3-Next layout).
    """
    if attention_every < 1:
        raise ValueError("attention_every must be >= 1")
    return tuple("attention" if (i + 1) % attention_every == 0 else mixer for i in range(n_layers))


@dataclass(frozen=True)
class RoPEScalingConfig:
    """RoPE frequency scaling, as Hugging Face ``rope_type`` ``"linear"``, ``"llama3"`` and ``"yarn"`` define it.

    ``"linear"`` divides every frequency by ``factor`` (position interpolation);
    ``"llama3"`` (Llama 3.1 and later) divides frequencies whose wavelength exceeds
    ``original_max_position_embeddings / low_freq_factor`` by ``factor``, keeps those
    below ``original_max_position_embeddings / high_freq_factor`` and interpolates
    smoothly in between. ``"yarn"`` (YaRN: Qwen2.5 long context, DeepSeek-V3) blends
    divided and kept frequencies with a linear ramp between the channels that turn
    ``beta_fast`` and ``beta_slow`` times over ``original_max_position_embeddings`` and
    multiplies the rotated channels by ``attention_factor``.
    """

    kind: Literal["linear", "llama3", "yarn"] = "llama3"
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    beta_fast: float = 32.0  # yarn
    beta_slow: float = 1.0  # yarn
    attention_factor: float = 1.0  # yarn: scale of the rotated channels (HF's resolved attention_factor)
    truncate: bool = True  # yarn: round the ramp's ends to whole channels

    def __post_init__(self) -> None:
        if self.kind not in ("linear", "llama3", "yarn"):
            raise ValueError(f"unknown RoPE scaling kind {self.kind!r}")
        if self.factor <= 0:
            raise ValueError("RoPE scaling factor must be positive")
        if self.kind == "llama3" and not (
            0 < self.low_freq_factor < self.high_freq_factor and self.original_max_position_embeddings > 0
        ):
            raise ValueError("llama3 RoPE scaling needs 0 < low_freq_factor < high_freq_factor and a positive context")
        if self.kind == "yarn" and not (self.beta_fast > 0 and self.beta_slow > 0 and self.attention_factor > 0
                                        and self.original_max_position_embeddings > 0):
            raise ValueError("yarn RoPE scaling needs positive beta_fast, beta_slow, attention_factor and context")
        if self.kind != "yarn" and self.attention_factor != 1.0:
            raise ValueError("attention_factor applies to yarn RoPE scaling only")


__all__ = [
    "AttentionConfig",
    "ForgettingAttentionConfig",
    "LINEAR_ATTENTION_KINDS",
    "LinearAttentionConfig",
    "MLAConfig",
    "MTPConfig",
    "MoEConfig",
    "ModelConfig",
    "PathAttentionConfig",
    "RoPEScalingConfig",
    "SPARSE_ATTENTION_KINDS",
    "SSMConfig",
    "SSM_KINDS",
    "SparseAttentionConfig",
    "hybrid_pattern",
]


# Checkpoint names a "names" / "offload" remat policy may list (jla.models.lm.remat_policy).
REMAT_CHECKPOINT_NAMES: tuple[str, ...] = ("attention", "linear", "mlp_hidden", "kernels")


def _check_remat_names(remat: str, names: tuple[str, ...], offload_names: tuple[str, ...]) -> None:
    unknown = sorted(set(names) - set(REMAT_CHECKPOINT_NAMES))
    if unknown:
        raise ValueError(f"unknown remat_names {unknown}; expected some of {list(REMAT_CHECKPOINT_NAMES)}")
    unknown = sorted(set(offload_names) - (set(REMAT_CHECKPOINT_NAMES) - {"kernels"}))
    if unknown:
        raise ValueError(f"remat_offload_names {unknown} cannot be offloaded; expected some of "
                         "['attention', 'linear', 'mlp_hidden']")
    if set(names) & set(offload_names):
        raise ValueError("a checkpoint name cannot be both kept (remat_names) and offloaded (remat_offload_names)")
    if remat == "names" and (not names or offload_names):
        raise ValueError("remat = 'names' needs remat_names and no remat_offload_names")
    if remat == "offload" and not offload_names:
        raise ValueError("remat = 'offload' needs remat_offload_names")
    if remat not in ("names", "offload") and (names or offload_names):
        raise ValueError(f"remat_names / remat_offload_names need remat = 'names' or 'offload'; got {remat!r}")


def aux_loss_coef_from_hf(router_aux_loss_coef: float, top_k: int) -> float:
    """``MoEConfig.aux_loss_coef`` for a Hugging Face ``router_aux_loss_coef`` (Mixtral, Qwen2/3-MoE, Qwen3-Next).

    HF's ``load_balancing_loss_func`` is ``E * sum_e (counts_e / tokens) P_e``; ours divides the same counts
    by the routes (``tokens * top_k``, the Switch / DeepSeek-V3 form), so it is ``1 / top_k`` of HF's and the
    same balancing weight needs ``top_k`` times HF's coefficient.
    """
    return float(router_aux_loss_coef) * int(top_k)


def aux_loss_coef_to_hf(aux_loss_coef: float, top_k: int) -> float:
    """HF ``router_aux_loss_coef`` for a ``MoEConfig.aux_loss_coef`` (inverse of :func:`aux_loss_coef_from_hf`)."""
    return float(aux_loss_coef) / int(top_k)
