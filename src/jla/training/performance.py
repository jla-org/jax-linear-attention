# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throughput accounting: model FLOPs, MFU, remat-aware HFU and MBU.

FLOPs are analytical (2 per multiply-accumulate) for one token of a
``ModelConfig`` at a given sequence length:

* softmax attention: q/k/v/o projections with grouped KV heads, an optional
  output gate, and the causal score/value matmuls over the average visible
  context (the sliding window caps it);
* recurrent mixers, counted under ``linear_attention_*``: the linear-attention
  kinds (``gated_deltanet``, ``gdn2``, ``kda``, ``gla``, ``deltanet``,
  ``simple_gla``) and the Mamba-2 / Mamba-3 SSMs: projections, gates, short
  convolution and the chunked recurrence (state read-out/update, delta-rule
  correction, and the causal intra-chunk products; Mamba-3 writes two streams);
* SwiGLU MLPs (``mlp_types="none"`` counts zero); MoE layers count the router and
  the *active* experts only;
* the LM head and multi-token-prediction heads.

Training FLOPs are ``3 x`` forward. HFU adds the recomputation implied by the
remat policy. MBU uses the compiler's ``bytes accessed`` estimate when
available and a transparent hand model otherwise; both are labelled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "DEVICE_PROFILES",
    "DeviceProfile",
    "FlopEstimate",
    "StepPerformance",
    "compiled_cost_bytes_accessed",
    "device_profile_for",
    "estimate_flops_per_token",
    "estimate_hfu_flops_per_token",
    "estimate_step_bytes",
    "step_performance",
]

_BETA_KINDS = ("deltanet", "gated_deltanet", "kda")
_DELTA_KINDS = (*_BETA_KINDS, "gdn2")
_SSM_KINDS = ("mamba2", "mamba3", "mamba1")
_SPARSE_KINDS = ("nsa", "dsa")


@dataclass(frozen=True)
class DeviceProfile:
    """Dense peak compute (TFLOP/s per device) and HBM bandwidth (decimal GB/s)."""

    name: str
    bf16_tflops: float
    hbm_gb_s: float
    fp8_tflops: float = 0.0
    fp32_tflops: float = 0.0

    def peak_tflops(self, dtype: str) -> float:
        key = dtype.lower()
        if key in ("bfloat16", "bf16", "float16", "fp16"):
            return self.bf16_tflops
        if key in ("float8", "fp8", "e4m3", "e5m2"):
            return self.fp8_tflops or self.bf16_tflops
        if key in ("float32", "fp32"):
            return self.fp32_tflops or self.bf16_tflops / 2.0
        raise ValueError(f"unsupported compute dtype {dtype!r}")


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    profile.name: profile
    for profile in (
        DeviceProfile("A10", 125.0, 600.0),
        DeviceProfile("A100", 312.0, 1935.0),
        DeviceProfile("A100-SXM", 312.0, 2039.0),
        DeviceProfile("H100", 989.0, 3352.0, fp8_tflops=1979.0),
        DeviceProfile("H200", 989.0, 4800.0, fp8_tflops=1979.0),
        DeviceProfile("B200", 2250.0, 8000.0, fp8_tflops=4500.0),
        DeviceProfile("B300", 2500.0, 8000.0, fp8_tflops=5000.0),
        DeviceProfile("RTX PRO 6000", 1000.0, 1597.0, fp8_tflops=2000.0),
        DeviceProfile("MI300X", 1307.4, 5300.0, fp8_tflops=2614.9),
        DeviceProfile("MI325X", 1307.4, 6000.0, fp8_tflops=2614.9),
        DeviceProfile("TPU v4", 275.0, 1200.0),
        DeviceProfile("TPU v5 lite", 197.0, 819.0),
        DeviceProfile("TPU v5p", 459.0, 2765.0),
    )
}


def device_profile_for(device_kind: str) -> DeviceProfile | None:
    """Profile whose name appears in a JAX ``device_kind`` (longest match); ``None`` if unknown."""
    upper = device_kind.upper().replace("_", " ")
    if "A100" in upper and "SXM" in upper:
        return DEVICE_PROFILES["A100-SXM"]
    for name in sorted(DEVICE_PROFILES, key=len, reverse=True):
        if name.upper() in upper:
            return DEVICE_PROFILES[name]
    return None


@dataclass(frozen=True)
class FlopEstimate:
    """Forward FLOPs per token by component; ``training`` = 3 x ``forward``."""

    attention_projections: float
    attention_core: float
    linear_attention_projections: float
    linear_attention_core: float
    mlp: float
    lm_head: float
    mtp: float

    @property
    def blocks(self) -> float:
        return (
            self.attention_projections
            + self.attention_core
            + self.linear_attention_projections
            + self.linear_attention_core
            + self.mlp
        )

    @property
    def forward(self) -> float:
        return self.blocks + self.lm_head + self.mtp

    @property
    def training(self) -> float:
        return 3.0 * self.forward

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "forward": self.forward, "training": self.training}


def _average_context(seq_len: int, window: int | None) -> float:
    """Mean number of keys a causal query sees (window-capped)."""
    if window is None or window >= seq_len:
        return (seq_len + 1) / 2.0
    # Positions t < window see t + 1 keys; the rest see window keys.
    return (window * (window + 1) / 2.0 + (seq_len - window) * window) / seq_len


def _attention_flops(config: Any, attention: Any, seq_len: int) -> tuple[float, float]:
    d = config.d_model
    q_width = attention.num_heads * attention.head_dim
    kv_width = attention.num_kv_heads * attention.head_dim
    projections = 2.0 * d * q_width + 2.0 * 2.0 * d * kv_width + 2.0 * q_width * d
    if attention.output_gate:
        projections += 2.0 * d * q_width
    core = 4.0 * q_width * _average_context(seq_len, attention.sliding_window)
    if getattr(attention, "moba_block_size", None) is not None:
        core = _moba_core_flops(q_width, seq_len, attention.moba_block_size, attention.moba_top_k)
    return projections, core


def _linear_attention_flops(config: Any, kind: str, seq_len: int) -> tuple[float, float]:
    """Projections follow FLA's layer defaults (as ``jla.layers.linear_attention`` does)."""
    if kind == "ttt_linear":
        return _ttt_linear_flops(config, seq_len)
    if kind == "gated_deltaproduct":
        return _gated_deltaproduct_flops(config, seq_len)
    if kind in ("rwkv6", "hgrn2"):
        return _rwkv6_hgrn2_flops(config, kind, seq_len)
    if kind == "rwkv7":
        return _rwkv7_flops(config, seq_len)
    if kind == "lightning":
        return _lightning_flops(config, seq_len)
    la = config.linear_attention
    d = config.d_model
    k_width = la.num_k_heads * la.head_k_dim
    v_width = la.num_v_heads * la.head_v_dim
    projections = 2.0 * 2.0 * d * k_width + 2.0 * d * v_width + 2.0 * v_width * d
    kda_rank = la.gate_low_rank_dim or la.head_v_dim
    if la.output_gate:  # KDA's output gate is low-rank
        projections += 2.0 * (d * kda_rank + kda_rank * v_width) if kind == "kda" else 2.0 * d * v_width
    if la.conv_size:
        projections += 2.0 * la.conv_size * (2 * k_width + v_width)
    if kind == "gla":  # low-rank per-channel decay gate, rank 16 by default
        rank = la.gate_low_rank_dim or 16
        projections += 2.0 * d * rank + 2.0 * rank * k_width
    elif kind == "kda":  # low-rank per-channel decay gate over the value heads' key channels
        projections += 2.0 * d * kda_rank + 2.0 * kda_rank * la.num_v_heads * la.head_k_dim
    elif kind == "gdn2":  # low-rank channel-wise decay, channel-wise erase (keys) and write (values)
        projections += 2.0 * d * kda_rank + 2.0 * kda_rank * k_width + 2.0 * d * k_width + 2.0 * d * v_width
    elif kind in ("simple_gla", "gated_deltanet"):
        projections += 2.0 * d * la.num_v_heads
    if kind in _BETA_KINDS:
        projections += 2.0 * d * la.num_v_heads  # beta
    if kind == "mlstm":  # input and forget gate per head (its sigmoid output gate is the output_gate term)
        projections += 2.0 * 2.0 * d * la.num_v_heads
    state = la.head_k_dim * la.head_v_dim
    per_head = 4.0 * state + (2.0 * state if kind in _DELTA_KINDS else 0.0)
    chunk = min(la.chunk_size, seq_len)
    per_head += (la.head_k_dim + la.head_v_dim) * chunk  # causal intra-chunk q.k^T and attn.v
    if kind == "mlstm":  # the normaliser n: a key-sized state row (update + q.n) and the row sums of the scores
        per_head += 4.0 * la.head_k_dim + chunk
    return projections, la.num_v_heads * per_head


def _ssm_flops(config: Any, kind: str, seq_len: int) -> tuple[float, float]:
    """Mamba-2 / Mamba-3 SISO mixers (``jla.layers.ssm``, the official module layouts)."""
    if kind == "mamba1":
        return _mamba1_flops(config)
    ssm = config.ssm
    d = config.d_model
    inner = ssm.expand * d
    heads = inner // ssm.head_dim
    bc_width = ssm.n_groups * ssm.d_state
    rank = getattr(ssm, "mimo_rank", 1) if kind == "mamba3" else 1  # Mamba-3 MIMO: R copies of B, C
    bc_width *= rank
    projections = 2.0 * d * 2 * inner + 2.0 * d * 2 * bc_width + 2.0 * inner * d  # z, x, B, C, out
    levels = getattr(ssm, "log_linear_levels", 0) if kind == "mamba2" else 0
    projections += 2.0 * d * heads * levels  # log-linear Mamba-2 level logits
    if kind == "mamba2":
        projections += 2.0 * d * heads  # dt
        projections += 2.0 * ssm.conv_size * (inner + 2 * bc_width)
        streams = 1
    else:
        rotated = int(ssm.d_state * ssm.rope_fraction) // 2 * 2
        projections += 2.0 * d * 3 * heads + 2.0 * d * (rotated // 2)  # dt, A, trap; angles
        streams = 2  # the trapezoidal rule writes the current and the previous token
    state = ssm.d_state * ssm.head_dim
    chunk = min(ssm.chunk_size, seq_len)
    if rank > 1:  # the MIMO layer caps its chunks at 64 / R tokens (jla.layers.mamba3_mimo.mimo_chunk_size)
        chunk = max(1, min(chunk, 64 // rank))
    # MIMO: R reads and R writes per stream and token; intra-chunk scores are (C R) x (C R).
    per_head = rank * (2.0 * state + streams * (2.0 * state + rank * (ssm.d_state + ssm.head_dim) * chunk))
    per_head += 2.0 * state * max(levels - 1, 0)  # log-linear: queries read every Fenwick level
    return projections, heads * per_head


def _mlp_flops(config: Any, kind: str) -> float:
    d = config.d_model
    if kind == "none":
        return 0.0
    if kind == "dense":
        return 6.0 * d * config.d_ff
    if kind == "relu2":  # squared-ReLU MLP: up and down projections only
        return 4.0 * d * config.d_ff
    if kind == "rwkv_cmix":
        # RWKV-6 channel mix (jla.layers.rwkv6.RWKV6ChannelMix): key D x F, value F x D and
        # receptance D x D matmuls; the two token-shift mixes are elementwise.
        return 2.0 * d * (2 * config.d_ff + d)
    if kind == "rwkv7_cmix":  # RWKV-7 channel mix (jla.layers.rwkv7.RWKV7FeedForward): key D x F, value F x D
        return 4.0 * d * config.d_ff
    moe = config.moe
    experts = moe.top_k + moe.num_shared_experts
    shared_gate = 2.0 * d if getattr(moe, "shared_expert_gate", False) else 0.0  # D -> 1 sigmoid gate
    return 2.0 * d * moe.num_experts + experts * 6.0 * d * moe.d_ff_expert + shared_gate


def estimate_flops_per_token(config: Any, seq_len: int) -> FlopEstimate:
    """Analytical forward FLOPs per token of a ``jla.models.ModelConfig``."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    attn_proj = attn_core = lin_proj = lin_core = mlp = 0.0
    for layer_type, mlp_type in zip(config.resolved_layer_types, config.resolved_mlp_types):
        if layer_type in ("attention", "local_attention"):
            attn_cfg = config.attention if layer_type == "attention" else config.local_attention
            p, c = _attention_flops(config, attn_cfg, seq_len)
            attn_proj, attn_core = attn_proj + p, attn_core + c
        elif layer_type == "fox":
            p, c = _forgetting_attention_flops(config, seq_len)
            attn_proj, attn_core = attn_proj + p, attn_core + c
        elif layer_type == "path":
            p, c, householder = _path_attention_flops(config, seq_len)
            attn_proj, attn_core, lin_core = attn_proj + p, attn_core + c, lin_core + householder
        elif layer_type == "mla":
            p, c = _mla_flops(config, seq_len)
            attn_proj, attn_core = attn_proj + p, attn_core + c
        elif layer_type in _SPARSE_KINDS:
            p, c = _sparse_attention_flops(config, layer_type, seq_len)
            attn_proj, attn_core = attn_proj + p, attn_core + c
        else:
            mixer = _ssm_flops if layer_type in _SSM_KINDS else _linear_attention_flops
            p, c = mixer(config, layer_type, seq_len)
            lin_proj, lin_core = lin_proj + p, lin_core + c
        mlp += _mlp_flops(config, mlp_type)
    lm_head = 2.0 * config.d_model * config.vocab_size
    mtp = 0.0
    if config.mtp is not None:
        block = sum(_attention_flops(config, config.attention, seq_len)) + _mlp_flops(
            config, getattr(config.mtp, "mlp_type", "dense"))  # "dense" = 6 d d_ff
        projection = 2.0 * 2 * config.d_model * config.d_model
        per_depth = {
            "deepseek": projection + block + lm_head,
            "chained": projection + lm_head,
            "independent": block + lm_head,
        }[config.mtp.kind]
        mtp = config.mtp.depth * per_depth
    return FlopEstimate(attn_proj, attn_core, lin_proj, lin_core, mlp, lm_head, mtp)


def estimate_hfu_flops_per_token(flops: FlopEstimate, remat: str, names: tuple[str, ...] = ()) -> float:
    """Executed FLOPs per training token including recomputation for ``remat``.

    ``none``/``dots`` recompute (almost) no matmuls; ``full`` replays every
    block forward; ``dots_with_no_batch_dims`` replays the batched attention
    and recurrence products; ``save_attention`` replays everything except them.
    ``kernels`` saves the kernel outputs but, as measured on the Mosaic GPU kernels (GDN, KDA,
    GDN-2, Mamba-3; benchmarks/fp8_1p3b.md), the forward kernels still launch twice, so it is
    counted as ``full`` and ``dots_kernels`` as ``dots`` plus the mixer cores (Mamba-2's
    kernel is not replayed, so its HFU is overstated by its core). For
    ``names`` / ``offload``, ``names`` lists the saved and offloaded checkpoint names
    (``ModelConfig.remat_names + remat_offload_names``): ``linear`` keeps the ``Linear`` and
    ``StackedLinear`` projections and the MLP down projection (a third of a SwiGLU),
    ``mlp_hidden`` the gate/up product (the other two thirds), ``attention`` the mixer cores
    (``kernels`` saves no replay, as above).
    """
    core = flops.attention_core + flops.linear_attention_core
    extra = {
        "none": 0.0,
        "dots": 0.0,
        "full": flops.blocks,
        "dots_with_no_batch_dims": core,
        "save_attention": flops.blocks - core,
        "kernels": flops.blocks,
        "dots_kernels": core,
        "names": _named_replay(flops, core, names),
        "offload": _named_replay(flops, core, names),
    }
    if remat not in extra:
        raise ValueError(f"unknown remat policy {remat!r}; expected one of {sorted(extra)}")
    return flops.training + extra[remat]


def estimate_step_bytes(
    *,
    config: Any,
    total_params: int,
    optimizer_state_bytes: int,
    tokens_per_microbatch: int,
    grad_accum: int,
) -> float:
    """Hand-model HBM bytes per optimizer step (labelled ``hand_model`` in logs).

    Parameter reads per microbatch (forward, backward and remat replay),
    float32 gradient accumulation, one optimizer-state read and write, the
    parameter update, and a coarse activation stream.
    """
    param_bytes = total_params * _dtype_size(config.param_dtype)
    compute_bytes = _dtype_size(config.compute_dtype)
    remat = config.remat != "none"
    reads_per_micro = 3 if remat else 2
    activation = tokens_per_microbatch * config.d_model * config.n_layers * compute_bytes * (4 if remat else 8)
    return float(
        grad_accum * (reads_per_micro * param_bytes + activation)
        + total_params * 4.0 * (grad_accum + 1)
        + 2.0 * optimizer_state_bytes
        + param_bytes
    )


def _dtype_size(name: str) -> int:
    sizes = {"float32": 4, "bfloat16": 2, "float16": 2, "float64": 8}
    if name not in sizes:
        raise ValueError(f"unsupported dtype {name!r}")
    return sizes[name]


def compiled_cost_bytes_accessed(cost_analysis: Any) -> float | None:
    """Aggregate ``bytes accessed`` from a JAX cost analysis (mapping or per-device list).

    Operand-specific ``bytes accessed0{}`` style fields are not summed again.
    This is a compiler estimate; opaque custom calls may be under-counted.
    """

    def normalized(key: Any) -> str:
        return "".join(ch for ch in str(key).lower() if ch.isalnum())

    if cost_analysis is None:
        return None
    if isinstance(cost_analysis, Mapping):
        for key, value in cost_analysis.items():
            if normalized(key) == "bytesaccessed":
                result = float(value)
                if not math.isfinite(result) or result < 0:
                    raise ValueError(f"bytes accessed must be finite and >= 0, got {value!r}")
                return result
        for key, value in cost_analysis.items():
            if normalized(key) == "costanalysis":
                return compiled_cost_bytes_accessed(value)
        return None
    if isinstance(cost_analysis, Sequence) and not isinstance(cost_analysis, (str, bytes)):
        values = [compiled_cost_bytes_accessed(item) for item in cost_analysis]
        present = [value for value in values if value is not None]
        return sum(present) if present else None
    return None


@dataclass(frozen=True)
class StepPerformance:
    """Per-window throughput and utilization (aggregate over ``num_devices``)."""

    tokens_per_sec: float
    tokens_per_sec_per_device: float
    step_time: float
    model_tflops_per_sec_per_device: float
    mfu_percent: float
    hfu_percent: float
    mbu_percent: float
    bandwidth_gb_s_per_device: float
    bytes_per_step_source: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def step_performance(
    *,
    tokens_per_step: int,
    step_time: float,
    flops_per_token: float,
    hfu_flops_per_token: float,
    num_devices: int,
    bytes_per_step: float,
    bytes_per_step_source: str,
    profile: DeviceProfile | None,
    compute_dtype: str,
) -> StepPerformance:
    """MFU/HFU/MBU for one step; utilizations are 0 when the device profile is unknown."""
    if step_time <= 0 or num_devices < 1:
        raise ValueError("step_time must be > 0 and num_devices >= 1")
    tokens_per_sec = tokens_per_step / step_time
    model_tflops = tokens_per_sec * flops_per_token / 1e12
    hfu_tflops = tokens_per_sec * hfu_flops_per_token / 1e12
    bandwidth = bytes_per_step / step_time / 1e9
    peak = profile.peak_tflops(compute_dtype) * num_devices if profile else 0.0
    peak_bandwidth = profile.hbm_gb_s * num_devices if profile else 0.0
    return StepPerformance(
        tokens_per_sec=tokens_per_sec,
        tokens_per_sec_per_device=tokens_per_sec / num_devices,
        step_time=step_time,
        model_tflops_per_sec_per_device=model_tflops / num_devices,
        mfu_percent=100.0 * model_tflops / peak if peak else 0.0,
        hfu_percent=100.0 * hfu_tflops / peak if peak else 0.0,
        mbu_percent=100.0 * bandwidth / peak_bandwidth if peak_bandwidth else 0.0,
        bandwidth_gb_s_per_device=bandwidth / num_devices,
        bytes_per_step_source=bytes_per_step_source,
    )


def _ttt_linear_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """TTT-Linear (``jla.layers.ttt``): q/k/v/out (+ gate) projections, short convs, the
    per-head learning-rate projection and the read-out of the initial inner weights ``w0``;
    the core is the dual form per token and head: ``k W``, ``q W`` and the end-of-mini-batch
    update (``2 K V`` each, one of them counted with ``w0``) plus the ``[b, b]`` intra
    mini-batch products ``q k^T``, ``(E * q k^T) gz`` and ``E gz``."""
    la = config.linear_attention
    d = config.d_model
    heads, dim = la.num_v_heads, la.head_v_dim
    width = heads * dim
    projections = 2.0 * 3.0 * d * width + 2.0 * width * d + 2.0 * d * heads + 2.0 * heads * dim * dim
    if la.output_gate:
        projections += 2.0 * d * width
    if la.conv_size:
        projections += 2.0 * la.conv_size * 3 * width
    block = min(la.ttt_mini_batch_size, seq_len)
    per_head = 4.0 * dim * dim + 6.0 * block * dim
    return projections, heads * per_head


def _moba_core_flops(q_width: int, seq_len: int, block_size: int, top_k: int) -> float:
    """MoBA core FLOPs per token: gate dots with earlier block means plus q.k^T and attn.v.

    A query at position t sees ``t % block_size + 1`` keys of its own block and
    ``min(top_k - 1, t // block_size)`` full earlier blocks, and scores all
    ``t // block_size`` earlier block means (one document of ``seq_len`` tokens).
    """
    keys = gates = 0.0
    for block in range(-(-seq_len // block_size)):
        m = min(block_size, seq_len - block * block_size)  # tokens in this block
        keys += m * (m + 1) / 2.0 + m * min(top_k - 1, block) * block_size
        gates += m * block
    return (4.0 * q_width * keys + 2.0 * q_width * gates) / seq_len


def _forgetting_attention_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """FoX layers (``jla.layers.forgetting_attention``): attention plus the per-head forget gate.

    Projections add the forget gate (``d -> heads``), the optional output gate and the
    KV-shift mixing gates (``d -> kv_heads`` for k and for v); the core is the causal
    score/value matmuls plus the ``c_i - c_j`` bias over the visible context.
    """
    fox = config.forgetting_attention
    d = config.d_model
    q_width = fox.num_heads * fox.head_dim
    kv_width = fox.num_kv_heads * fox.head_dim
    projections = 2.0 * d * q_width + 2.0 * 2.0 * d * kv_width + 2.0 * q_width * d + 2.0 * d * fox.num_heads
    if fox.output_gate:
        projections += 2.0 * d * q_width
    if fox.kv_shift:
        projections += 2.0 * 2.0 * d * fox.num_kv_heads
    context = _average_context(seq_len, fox.sliding_window)
    core = 4.0 * q_width * context + fox.num_heads * context
    return projections, core


def _path_attention_flops(config: Any, seq_len: int) -> tuple[float, float, float]:
    """PaTH attention (``jla.layers.path_attention``): projections, softmax core, Householder transform.

    The projections and the causal score/value matmuls are counted as softmax
    attention; the chunked UT transform (``W_beta W^T``, the triangular solve,
    the reflected queries and keys, the whole-chunk reflections and the
    transport of every query across the earlier chunks) is returned separately
    and counted with the recurrent mixers' core.
    """
    path = config.path_attention
    d = config.d_model
    q_width = path.num_heads * path.head_dim
    kv_width = path.num_kv_heads * path.head_dim
    projections = 2.0 * d * q_width + 2.0 * 2.0 * d * kv_width + 2.0 * q_width * d
    rank = path.w_low_rank_dim
    projections += 2.0 * d * kv_width if rank is None else 2.0 * (d * rank + rank * kv_width)
    projections += 2.0 * path.w_conv_size * kv_width + 2.0 * d * path.num_kv_heads  # w conv, beta
    if path.forget_gate:
        projections += 2.0 * d * path.num_heads
    core = 4.0 * q_width * _average_context(seq_len, None)
    chunk = min(path.chunk_size, seq_len)
    dim = path.head_dim
    earlier_chunks = (math.ceil(seq_len / chunk) - 1) / 2.0  # mean number of earlier chunks per query
    macs = 6.0 * chunk * dim + 1.5 * chunk * chunk + dim * dim + earlier_chunks * dim * dim
    return projections, core, path.num_heads * 2.0 * macs


def _mamba1_flops(config: Any) -> tuple[float, float]:
    """Mamba-1 mixer (``jla.layers.mamba1``, state-spaces/mamba ``Mamba``): projections and selective scan."""
    ssm = config.ssm
    d = config.d_model
    inner = ssm.expand * d
    rank = -(-d // 16) if ssm.dt_rank is None else ssm.dt_rank
    projections = 2.0 * d * 2 * inner + 2.0 * inner * (rank + 2 * ssm.d_state) + 2.0 * rank * inner
    projections += 2.0 * inner * d + 2.0 * ssm.conv_size * inner  # out_proj; depthwise conv
    # Per channel and state: delta * A, (delta u) * B, h = exp(.) h + ., y += C h.
    return projections, 6.0 * inner * ssm.d_state


# RWKV-6 time mixing and HGRN2 (jla.layers.rwkv6, FLA's layer layouts).
_RWKV6_MIX_RANK = 32  # jla.layers.rwkv6.MIX_RANK: five ddlerp mixes of this rank
_RWKV6_DECAY_RANK = 64  # jla.layers.rwkv6.DECAY_RANK: default decay LoRA rank


def _rwkv6_hgrn2_flops(config: Any, kind: str, seq_len: int) -> tuple[float, float]:
    la = config.linear_attention
    d = config.d_model
    k_width = la.num_k_heads * la.head_k_dim
    v_width = la.num_v_heads * la.head_v_dim
    if kind == "rwkv6":
        rank = la.gate_low_rank_dim or _RWKV6_DECAY_RANK
        mix = 5 * _RWKV6_MIX_RANK
        projections = 2.0 * d * mix + 2.0 * mix * d  # ddlerp down and up
        projections += 2.0 * 2.0 * d * k_width + 2.0 * 2.0 * d * v_width  # r, k; v, g
        projections += 2.0 * d * rank + 2.0 * rank * k_width  # decay LoRA
    else:
        projections = 2.0 * 2.0 * d * k_width + 2.0 * d * v_width  # q, f; i
    projections += 2.0 * v_width * d  # output projection
    chunk = min(la.chunk_size, seq_len)
    per_head = 4.0 * la.head_k_dim * la.head_v_dim + (la.head_k_dim + la.head_v_dim) * chunk
    return projections, la.num_v_heads * per_head


def _gated_deltaproduct_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """Gated DeltaProduct (``jla.layers.linear_attention.GatedDeltaProduct``).

    k, v, beta and their convs are ``num_householder`` (``n_h``) times as wide as
    Gated DeltaNet's; the core runs the gated delta rule over ``T * n_h`` sub-steps.
    """
    la = config.linear_attention
    n = la.num_householder
    d = config.d_model
    k_width = la.num_k_heads * la.head_k_dim
    v_width = la.num_v_heads * la.head_v_dim
    projections = 2.0 * d * k_width * (1 + n) + 2.0 * d * v_width * n + 2.0 * v_width * d  # q, k, v, out
    if la.output_gate:
        projections += 2.0 * d * v_width
    if la.conv_size:
        projections += 2.0 * la.conv_size * (k_width + n * (k_width + v_width))
    projections += 2.0 * d * la.num_v_heads * (1 + n)  # decay, beta per sub-step
    state = la.head_k_dim * la.head_v_dim
    chunk = min(la.chunk_size, seq_len * n)
    per_head = n * (6.0 * state + (la.head_k_dim + la.head_v_dim) * chunk)
    return projections, la.num_v_heads * per_head


def _mla_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """Multi-head latent attention (``jla.layers.mla``): low-rank q, latent kv, shared RoPE key.

    The core counts the model's q.k (``qk_head_dim``) and p.v (``v_head_dim``) products,
    not the zero padding the full-sequence kernel call adds to the narrower one.
    """
    mla = config.mla
    d = config.d_model
    heads, qk = mla.num_heads, mla.qk_nope_head_dim + mla.qk_rope_head_dim
    if mla.q_lora_rank:
        projections = 2.0 * d * mla.q_lora_rank + 2.0 * mla.q_lora_rank * heads * qk
    else:
        projections = 2.0 * d * heads * qk
    projections += 2.0 * d * (mla.kv_lora_rank + mla.qk_rope_head_dim)
    projections += 2.0 * mla.kv_lora_rank * heads * (mla.qk_nope_head_dim + mla.v_head_dim)
    projections += 2.0 * heads * mla.v_head_dim * d
    core = 2.0 * heads * (qk + mla.v_head_dim) * _average_context(seq_len, mla.sliding_window)
    return projections, core


def _sparse_attention_flops(config: Any, kind: str, seq_len: int) -> tuple[float, float]:
    """NSA / DSA: GQA projections plus the gates or the indexer; the core counts the sparse work only.

    NSA attends to the complete compressed blocks, ``block_count`` selected blocks
    and a sliding window; DSA scores every causal key with the indexer and attends
    to ``top_k`` keys. The plain-JAX ops evaluate these with dense ``[T, T]`` masks.
    """
    attention, sparse, d = config.attention, config.sparse_attention, config.d_model
    projections, _ = _attention_flops(config, attention, seq_len)
    q_width = attention.num_heads * attention.head_dim
    context = _average_context(seq_len, None)
    if kind == "nsa":
        projections += 2.0 * d * 3 * attention.num_heads
        compressed = context / sparse.block_size
        selected = _average_context(seq_len, sparse.block_size * sparse.block_count)
        window = _average_context(seq_len, sparse.window_size) if sparse.window_size > 0 else 0.0
        core = 4.0 * q_width * (compressed + selected + window)
    else:
        index_width = sparse.index_heads * sparse.index_head_dim
        projections += 2.0 * d * (index_width + sparse.index_head_dim + sparse.index_heads)
        core = 2.0 * index_width * context + 4.0 * q_width * _average_context(seq_len, sparse.top_k)
    return projections, core


def _rwkv7_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """RWKV-7 time mix (``jla.layers.rwkv7.RWKV7Mixer``): projections, LoRAs and the chunked DPLR recurrence.

    Projections: ``r, k, v, o`` (``D x D``) and the decay, ``a``, value-residual and gate
    LoRAs (every layer computes its value LoRA; the first one's result is unused). The
    core per head and token: the state terms ``a^T S``, the rank-1 ``b`` and ``k v^T``
    writes and ``q^T S`` (``8 K V``), plus the chunked algorithm's four causal pair
    matrices, the products with ``v`` / ``u`` and the WY triangular solves
    (``(5 K + 4 V) C`` for chunk ``C``).
    """
    from jla.layers.rwkv7 import rwkv7_lora_dims

    la = config.linear_attention
    d = config.d_model
    ranks = la.rwkv7_lora_dims or rwkv7_lora_dims(d, la.head_k_dim)
    projections = 2.0 * 4 * d * d + 2.0 * 2 * d * sum(ranks)
    chunk = min(la.chunk_size, seq_len)
    per_head = 8.0 * la.head_k_dim * la.head_v_dim + (5.0 * la.head_k_dim + 4.0 * la.head_v_dim) * chunk
    return projections, la.num_v_heads * per_head


def _lightning_flops(config: Any, seq_len: int) -> tuple[float, float]:
    """MiniMax-01 lightning attention (``jla.layers.lightning_attention``): ``qkv``, gate and output
    projections, and the ``simple_gla`` core (state read-out and update, causal intra-chunk products)."""
    la = config.linear_attention
    d = config.d_model
    width = la.num_v_heads * la.head_v_dim
    projections = 2.0 * 5 * d * width  # q, k, v, gate (D -> W) and out (W -> D)
    chunk = min(la.chunk_size, seq_len)
    per_head = 4.0 * la.head_k_dim * la.head_v_dim + (la.head_k_dim + la.head_v_dim) * chunk
    return projections, la.num_v_heads * per_head


def _named_replay(flops: FlopEstimate, core: float, names: tuple[str, ...]) -> float:
    """Forward FLOPs per token a ``names`` / ``offload`` remat policy replays (see estimate_hfu_flops_per_token)."""
    replay = flops.blocks
    if "linear" in names:
        replay -= flops.attention_projections + flops.linear_attention_projections + flops.mlp / 3
    if "mlp_hidden" in names:
        replay -= 2 * flops.mlp / 3
    if "attention" in names:
        replay -= core
    return max(replay, 0.0)
