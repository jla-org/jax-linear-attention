# Architectures and presets

[README](../README.md) · [Architectures](architectures.md) · [Kernels](kernels.md) · [Training](training.md) · [Results](results.md) · [Benchmarks](../benchmarks/README.md)

What the library implements, the model presets, and the ports beyond the release architectures.

- [Overview](#overview): [Release architectures](#release-architectures) · [Models](#models) · [Ops](#ops) · [Training](#training)
- [Presets](#presets)
- [Other ports](#other-ports): [GLA](#gla) · [Mamba-1](#mamba-1) · [Multi-head Latent Attention](#multi-head-latent-attention) · [DeepSeek Sparse Attention](#deepseek-sparse-attention) · [mLSTM](#mlstm) · [Gated DeltaProduct](#gated-deltaproduct) · [RWKV-6, HGRN2, TTT-Linear, log-linear Mamba-2](#rwkv-6-hgrn2-ttt-linear-log-linear-mamba-2) · [RWKV-7](#rwkv-7) · [Lightning attention](#lightning-attention) · [FoX, PaTH, MoBA, NSA](#fox-path-moba-nsa) · [MoE options](#moe-options) · [MoE balance loss](#moe-balance-loss)

JAX and Pallas Mosaic GPU kernels for linear attention and state-space models on NVIDIA Hopper
(Mamba-2, Mamba-3, Gated DeltaNet and DeltaNet, Gated DeltaNet-2, KDA, RWKV-7, GLA, Mamba-1 and
more), every recurrence with a token-by-token reference and a chunk-parallel JAX version behind
one dispatcher; an FA3-style softmax attention kernel; and the training stack around them: FSDP, tensor,
expert and pipeline parallelism, FP8 training, checkpoints with exact resume, generation, evaluation
and Hugging Face import and export. The Python package and the command-line tool are `jla`. On any
other device every op falls back to plain JAX.

## Overview

A JAX LLM training template with **Pallas Mosaic GPU** kernels for linear attention, state-space
models and softmax attention on NVIDIA Hopper. The Python package and the command-line tool are `jla`.

JAX has strong kernels for softmax attention, but not for the recurrent layers that newer
models use: in JAX they have mostly run as plain JAX ops (MaxText's Gated DeltaNet, for example).
jax-linear-attention adds Mosaic GPU kernels for these layers, and the training stack around them.

### Release architectures

**Release architectures.** Mamba-2, Mamba-3 (SISO and MIMO), Gated DeltaNet and DeltaNet, Gated
DeltaNet-2, KDA, RWKV-7, softmax attention (an FA3-style kernel, with an opt-in fused one-kernel
backward) and sliding-window attention. Each has a Mosaic GPU kernel for H100, tested against a
token-by-token reference and (except the clean-room GDN-2) the latest upstream code, with packed
documents (`segment_ids`), grouped heads, initial and final states and decode parity.
[Kernels](kernels.md#kernels) lists what `impl="auto"` picks, and [Results](results.md#results-13b-end-to-end) gives
the 1.3B end-to-end numbers (`transformer-1p3b` in FP8 with the TOML as written: 72.3% MFU on one H100 by
the wall clock, and 68.2% MFU per GPU on 8 GPUs (FSDP 8), where the TOML reduces the gradients once per step;
measured on 20 Oct 2026 with no other jobs on the GPUs).

### Models

- **Models**: Llama/Qwen3-style transformer (GQA, QK-norm, RoPE with Llama 3 / YaRN scaling,
  sliding window, output gate, attention sinks, mixed local and global layers), Mamba-2, Mamba-3,
  Gated DeltaNet, Gated DeltaNet-2, KDA, RWKV-7, hybrids of any of them with attention (including
  Kimi Linear's and Qwen3-Next's layouts at 1.3B), mixture-of-experts and multi-token prediction.
  Also ported (see [other ports](architectures.md#other-ports)): Mamba-1 and GLA with Mosaic GPU kernels that
  `impl="auto"` picks, Multi-head Latent Attention and DSA sparse attention with Mosaic GPU kernels
  on by default (their layers' `"auto"` implementation), mLSTM (xLSTM) with an opt-in kernel path,
  and as plain-JAX or opt-in paths log-linear Mamba-2, Gated DeltaProduct, HGRN2, RWKV-6,
  TTT-Linear, MiniMax-01 lightning attention, FoX, PaTH attention, MoBA and NSA sparse attention.

### Ops

- **Ops**: every recurrence ships as a token-by-token reference (the oracle), a chunk-parallel
  JAX version (FLA's stage decomposition, with exact packed-document masks) and, where
  available, a Mosaic GPU kernel. One fail-closed dispatcher picks the fastest implementation that
  supports the call.

### Training

- **Training**: one compiled train step with fp32 gradient accumulation and non-finite step
  rejection; data, FSDP, tensor, expert and pipeline parallelism over logical axis names, plus
  context-parallel attention as a library; dropless MoE on grouped-GEMM kernels with per-device
  dispatch; FP8 training (each weight quantized once per optimizer step, and opt-in FP8 q/k/v gradients
  written by the attention backward kernels); opt-in host-side and FSDP options for several GPUs (dispatch
  ahead, a select instead of the rejected-step conditional, one gradient reduce-scatter per step); an LM-head
  cross entropy that can form its gradient in the forward pass; remat policies and memory-lean VJPs (the
  1.3B transformer configs train without recompute); upstream-parity initialisation by default
  (FLA, state-spaces/mamba and RWKV-LM; the attention hybrids with Gated DeltaNet or KDA layers keep
  identity short convs, and jla's own initialisation stays selectable); crash-consistent
  checkpoints with exact resume; memory-mapped, streamed and SFT data; DPO preference tuning; JSONL,
  Weights & Biases and TensorBoard logging; generation, perplexity and lm-evaluation-harness evaluation; Hugging Face
  import and export. Details in [Training](training.md#training).

Status: early development (0.1.0). The APIs will change, and the Mosaic GPU APIs in JAX change
with every JAX release, so the JAX version is pinned (0.11.2).

## Presets

A config picks its architecture with `preset = "<name>"` in its `[model]` table. Many presets
also have a `tiny-*` variant for CPU tests; `jla.models.presets` defines them.

| family | presets |
|---|---|
| transformer, MoE | `transformer-125m`, `transformer-350m`, `transformer-1b`, `moe-125m` |
| Mamba | `mamba1-130m`, `mamba2-130m`, `mamba3-130m`, `mamba3-mimo-130m`, `loglinear-mamba2-130m` |
| delta rule, GLA, RWKV | `gdn-hybrid-125m`, `gdn2-125m`, `gdn2-hybrid-125m`, `kda-hybrid-125m`, `gated-deltaproduct-125m`, `deltanet-125m`, `gla-125m`, `hgrn2-125m`, `rwkv6-0.1b`, `rwkv7-0.1b` |
| other recurrent | `xlstm-125m`, `ttt-linear-125m` |
| attention variants | `fox-125m`, `path-125m`, `moba-hybrid-125m` (3 Gated DeltaNet : 1 MoBA), `tiny-mla`, `tiny-nsa`, `tiny-dsa` |
| 1.3B, FLA's published shapes | `transformer-1p3b`, `transformer-swa-1p3b`, `gdn-1p3b`, `gdn-hd128-1p3b`, `gdn2-1p3b`, `kda-1p3b`, `mamba2-1p3b`, `mamba3-1p3b`, `rwkv7-1p3b`, `gdn-hybrid-1p3b`, `kda-hybrid-1p3b` |
| real hybrid layouts at ~1.3B ([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts)) | `kimi-linear-1p3b` (Kimi-Linear-48B-A3B's layout: 3 KDA : 1 NoPE MLA, dense first layer then sigmoid-routed MoE, 32 x 512 experts top-8 + 1 shared; 1.20B total, 0.63B active), `qwen3-next-1p3b` (Qwen3-Next-80B-A3B's layout: 3 Gated DeltaNet : 1 gated attention, MoE in every layer, 64 x 256 experts top-10 + gated shared expert, one MTP layer; 1.27B total, 0.55B active) |
| published layouts | the table below, plus `deepseek-v3-tiny`, `kimi-linear-1b` (3 KDA : 1 NoPE MLA, MoE) and `tiny-jamba` (Mamba-1 : attention = 7 : 1, MoE every other layer) |

The 1.3B presets take their shapes from flame's `*_1B.json` configs and FLA main's model-config
defaults, with the Llama-3.1 vocabulary of 128,256 tokens and untied embeddings. FLA's models of
the same names have exactly the same parameter counts, except RWKV-7 (0.002% fewer here: RMSNorm
blocks instead of LayerNorms with bias); GDN-2 (clean-room) and the two hybrids have no FLA
counterpart here.
`benchmarks/end_to_end/1p3b/` has a TOML for each ([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md)).
`benchmarks/end_to_end/hybrids_1p3b/` has the TOMLs of the two real-hybrid presets; `bench_e2e_1p3b.py --arch kimi-linear-1p3b` finds them.

The presets below keep the per-head shapes, layer ratios and mixer settings of each model's
Hugging Face `config.json`, with the template's 50,304-token vocabulary. Each docstring lists what
differs from the original model.

| preset | after | layout |
|---|---|---|
| `qwen3-next-300m` (322M active, 874M total) | Qwen3-Next-80B-A3B | 3 Gated DeltaNet : 1 gated attention; MoE in every layer: 64 experts, top-10, plus a sigmoid-gated shared expert; one MTP layer (gated attention + MoE) |
| `nemotron-h-600m` (581M) | Nemotron-H-8B | the full 52-layer pattern: 24 Mamba-2, 4 NoPE attention, 24 squared-ReLU MLPs |
| `granite-4.0-h-125m` (122M active, 464M total) | Granite-4.0-H-Tiny | 9 Mamba-2 : 1 NoPE attention; MoE: 64 experts, top-6, plus a shared SwiGLU |
| `gpt-oss-180m` (184M active, 778M total) | gpt-oss-20b | alternating 128-token sliding and full attention with attention sinks; MoE: 32 experts, top-4 |
| `gemma-3-750m` (756M) | Gemma-3-1B | 5 local (512-token window) : 1 global attention, QK-norm |
| `minimax-01-1b` (193M active, 1.04B total) | MiniMax-Text-01 | 7 lightning attention : 1 softmax attention; MoE in every layer: 32 experts, top-2 |

The ratios of `deepseek-v3-tiny` and `kimi-linear-1b` have not yet been checked against the published
configs. `kimi-linear-1p3b` and `qwen3-next-1p3b` are compared field by field with
Kimi-Linear-48B-A3B-Instruct's and Qwen3-Next-80B-A3B-Instruct's `config.json`
([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts)).

RWKV-7 is a `LanguageModel` layer type: the `layer_types` entry `"rwkv7"` is the time mix (the
first RWKV-7 layer's value is carried to every later layer) and the `mlp_types` entry
`"rwkv7_cmix"` is its channel mix, in RMSNorm pre-norm blocks. `rwkv7-0.1b` (129M parameters) is
BlinkDL's 0.1B shape (L12-D768, head size 64) with the tied 50,304-token vocabulary. `tiny-rwkv7`,
`rwkv7-0.1b` and `rwkv7-1p3b` train with `jla train` and decode with the model cache. The standalone
`jla.models.rwkv7.RWKV7LM` keeps BlinkDL's exact LayerNorm layout (its presets end in `-blinkdl`).

## Other ports

Mosaic GPU kernels picked by default on H100 (`impl="auto"`; for MLA and DSA, the layer config's
`"auto"` implementation; rows in [Kernels](kernels.md#kernels)):

### GLA

- **GLA** (`gla`, also the core of the HGRN2 layer): dedicated kernels with FLA's `chunk_gla` stage
  split (`Aqk` only, a warp-specialized state pass fused with the output, a backward without
  delta-rule stages), exact intra-chunk decays, `segment_ids`, initial and final states. The first
  version (the KDA / GDN-2 kernels with constant gates) stays selectable as `pipeline="vector_gate"`
  ([gla.md](../benchmarks/gla.md)).

### Mamba-1

- **Mamba-1** (`mamba1`): a fused selective-scan forward (one channel per thread, states in registers,
  softplus, `D` skip and `z` gate fused) and a backward kernel (`backward="kernel"`, the default: states and
  decays stay in registers per 2-token sub-tile, `dB` and `dC` are reduced by warp shuffles without atomics,
  so it is deterministic, and an exact sequence split runs the segments in parallel; a slower first backward
  kernel stays selectable as `backward="kernel_v1"`), for `d_state` 4, 8, 16 and 32 (32 runs as two 16-state
  slices, and `impl="auto"` sends it to the kernels) and `D` a multiple of 128. Forward + backward is 1.85x
  the official CUDA kernel at the mamba-130m shape `(8, 2048, 1536, 16)`, 1.72x at the Falcon-Mamba-7B width
  `(1, 4096, 8192, 16)` and 1.60x at `d_state` 32 (provisional); the layer's conv runs on the short-conv
  kernels.
  Calls the kernels decline fall back to `chunked`, which on a GPU runs the chunked scan forward and takes
  the reference op's exact (slower) gradient, because differentiating the chunked scan is wrong on
  the H100 for some inputs. Building the kernels for training takes 15.9 s of lowering and
  1.4 s of compiling at the mamba-130m shape (22.7 s and 1.7 s at `d_state` 32; about two minutes with
  the first backward kernel), shared by every layer of that shape
  ([mamba1.md](../benchmarks/mamba1.md)).

### Multi-head Latent Attention

- **Multi-head Latent Attention** (`mla`): with `MLAConfig.implementation` `"auto"` (the default) or
  `"mosaic_gpu"`, on Hopper in bf16/fp16 with DeepSeek-V3's head shape (q/k 192 = 128 no-RoPE + 64
  RoPE, v 128; also Kimi K2, Kimi Linear, `kimi-linear-1b` and `kimi-linear-1p3b`), the training path
  runs our FA3-style Mosaic GPU attention kernel on the unpadded heads with its two-kernel backward
  (the fused backward takes symmetric heads only): causal, packed documents (`segment_ids`), sliding
  windows, per batch and head shard under a mesh ([mla.md](../benchmarks/mla.md)).
  Other head shapes (the tiny presets' 24/16) and dtypes run cuDNN or XLA attention: cuDNN on the
  unpadded heads (its MLA mode, cuDNN >= 9.10 on Hopper), XLA and older cuDNN with the value
  zero-padded to the query/key head dim (JAX 0.11.2's public `dot_product_attention` takes one head
  dim for q, k and v). Against cuDNN on the unpadded heads the kernel is 1.06-1.19x forward and
  1.01-1.05x forward + backward on causal rows (FA3 built for these heads: 0.95-0.97x and 1.01-1.06x);
  with packed documents it is 1.53-1.86x forward and 1.71-1.83x forward + backward against cuDNN's
  packed-sequence layout (which needs the values padded at these head dims in JAX 0.11.2), and 0.91-1.05x
  and 1.03-1.27x against cuDNN on equal documents split into rows. Latent-cache decode stays in XLA.

### DeepSeek Sparse Attention

- **DeepSeek Sparse Attention** (`dsa`, DeepSeek-V3.2's lightning indexer and top-k attention):
  on H100 with bf16 and head dims 64/128, `SparseAttentionConfig(dsa_implementation="auto")` (the
  default) runs the indexer, an exact top-k, the selected-key attention and the gradient of
  DeepSeek-V3.2's indexer KL loss on Mosaic GPU kernels, per query block; `"xla"` (and every other
  device) builds full `[T, T]` scores ([dsa.md](../benchmarks/dsa.md), all provisional). The exact top-k
  is a radix-select Mosaic GPU kernel (`jla.ops.mosaic_gpu.dsa_topk`, k = 2048): ties keep the lower
  column, as DeepSeek's `torch.topk` does, and +inf and NaN scores are never selected. In kernel time our
  exact bf16 indexer + top-k is 1.16-1.43x faster than DeepGEMM's FP8 indexer + `torch.topk` at 16K-128K
  tokens. The lead comes only from the top-k: DeepGEMM's FP8 logits are 1.5-1.6x faster than our bf16
  indexer, and `torch.topk` is DeepSeek's reference selection, not the fastest top-k there is. The radix
  path is bound by kernel launches: by wall clock our 16K call takes 10.4 ms against 5.1 ms of kernel
  time. With it the layer's forward at 16K is 1.44-1.46x faster in kernel time and 1.43-1.45x by wall
  clock, and its training step 1.30-1.31x; `index_topk` and the forward fit at 64K-128K tokens, and the
  128K training step does not fit in a 20 GB share. An approximate FP8 indexer is an explicit option
  (`SparseAttentionConfig(dsa_index_fp8=True)`): 1.71x / 1.61x DeepGEMM + `torch.topk` at 16K / 32K in
  kernel time, keeping 97.8% / 97.3% of the exact top-2048 keys. The selected-key attention runs as
  masked-dense kernels, at 0.08-0.12x the rate of FlashMLA's sparse prefill.
  `jla.ops.dsa_gather.gather_attention` is a tested plain-JAX gathering version (O(T × top_k) memory, any
  head dims; 1.1-1.6x faster than the masked-dense kernels at 128K tokens and slower at shorter lengths),
  not wired into the layer; a gathering kernel is the next step.

Opt-in Mosaic GPU paths (`auto` does not pick them):

### mLSTM

- **mLSTM** (`impl="mosaic_gpu"`): the stabilised mLSTM mapped exactly onto the simple-GLA kernels
  (`g' = g + m_{t-1} - m_t`, `k' = exp(i - m) k`), with the stabiliser and the denominator in float32
  XLA. Against NX-AI's TFLA Triton kernels it is 1.38-1.42x faster forward and 1.03-1.06x forward +
  backward (TFLA's backward holds the normaliser and the stabiliser constant, an approximate gradient;
  ours is exact), but rows where `q n` nearly cancels are less accurate than TFLA's, so `auto` keeps
  the float32 `chunked` op ([mlstm.md](../benchmarks/mlstm.md)).

### Gated DeltaProduct

- **Gated DeltaProduct** (`impl="mosaic_gpu"`): an exact route through the Gated DeltaNet kernels
  (interleaved sub-steps), 11.4x faster forward + backward than the plain-JAX `chunked` op
  ([gated_deltaproduct.md](../benchmarks/gated_deltaproduct.md), provisional). Mamba-3 MIMO also has a route
  through the SISO kernels, `kernel="siso"` ([mamba3.md](../benchmarks/mamba3.md#mimo)).

Plain JAX (reference and chunked ops, or XLA layers):

### RWKV-6, HGRN2, TTT-Linear, log-linear Mamba-2

- **RWKV-6** (`rwkv6`) and **HGRN2** (`hgrn2`, without HGRN2's layer-wise forget-gate lower bound;
  its GLA core runs on the GLA kernels above), **TTT-Linear** (`ttt_linear`; Titans is not started),
  **log-linear Mamba-2** (`log_linear_simple_gla`; no decode cache or initial state, so
  `loglinear-mamba2-130m` and `tiny-loglinear-mamba2` are training-only, and no log-linear Gated
  DeltaNet yet). Timings: [log_linear.md](../benchmarks/log_linear.md).

### RWKV-7

- **RWKV-7** with `chunk_size` other than 32 (the `tiny-rwkv7` preset uses 16) runs the chunked op,
  which builds exact pairwise-decay tensors. `rwkv7-0.1b` and `rwkv7-1p3b` set `chunk_size = 32`, so
  `impl = "auto"` runs the Mosaic GPU kernels, document masking included: a `rwkv7-0.1b` training step
  on FineWeb-Edu at 4 × 4096 tokens × 2 microbatches takes 276 ms (118,800 tok/s) on one H100, against
  1,146 ms (28,600 tok/s) on the chunked op at `chunk_size = 16` (provisional), with losses equal to
  within 1e-4 ([models.md](../benchmarks/models.md#results); the 290.1 ms against 116.0 ms for
  `rwkv6-0.1b` on the same page is the chunked op at 4 × 2048). Off Hopper, or with f32 compute, these
  presets fall back to the chunked op, which needs about twice the memory at chunk 32 as at 16; set
  `chunk_size = 16` there.

### Lightning attention

- **Lightning attention** (`lightning`, MiniMax-01): `simple_gla` with MiniMax-Text-01's fixed
  per-head decay; in bf16 with head dim 64 or 128 it runs the `simple_gla` Mosaic GPU kernel.

### FoX, PaTH, MoBA, NSA

- **FoX** (`fox`), **PaTH** (`path`), **MoBA** (`AttentionConfig(moba_block_size, moba_top_k)`) and
  **NSA** (`nsa`): XLA builds full `[T, T]` scores or masks, so memory grows as T²
  ([fox.md](../benchmarks/fox.md)). PaTH decodes with the key-reflection recurrence (each new token
  reflects the cached keys, O(T·D) per token). NSA and DSA have no decode cache yet: `tiny-nsa` and
  `tiny-dsa` are training-only, and `LanguageModel.init_cache` / `decode_step` raise for them.

### MoE options

- **MoE options**: `MoEConfig(shared_expert_gate=True)` scales the shared experts by `sigmoid(x W)`
  (Qwen2-MoE, Qwen3-Next), and `MTPConfig(mlp_type="moe")` gives the MTP blocks an MoE.

### MoE balance loss

- **MoE balance loss**: ours is the Switch / DeepSeek-V3 form, E·Σ f_e·P_e with f_e the fraction of
  routes. Hugging Face's `load_balancing_loss_func` (Mixtral, Qwen3-MoE, Qwen3-Next, MiniMax) divides
  the same counts by tokens, so it is top_k times ours, and an HF `router_aux_loss_coef` c becomes
  `aux_loss_coef = top_k·c` (`jla.models.config.aux_loss_coef_from_hf`). The `qwen3-next-300m`,
  `qwen3-next-1p3b` and `minimax-01-1b` presets (and the tiny variants) and `jla convert` in both
  directions apply this; DeepSeek-V3's `aux_loss_alpha` maps 1:1. `router_bias` (DeepSeek-V3) is
  applied but never updated: it serves imported checkpoints only, since aux-loss-free bias updates
  are not implemented. `MoEConfig.aux_loss_coef` defaults to 1e-2, and `qwen3-next-1p3b` sets
  1e-2 = top-10 × Qwen3-Next's 1e-3.

