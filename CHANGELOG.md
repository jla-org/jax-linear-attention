# Changelog

Notable changes to jax-linear-attention. Versions follow [semantic versioning](https://semver.org/);
before 1.0 a minor release may change APIs.

## 0.1.0 - 2026-10-21

First public release.

Requirements: Python 3.12 or newer and JAX 0.11.2 (pinned). The Mosaic GPU kernels need an NVIDIA
Hopper GPU (H100/H200, sm_90a) and a CUDA 13 driver; on any other device every op runs its plain-JAX
implementation. `requirements/jax-0.11.2-cuda13.txt` pins the tested CUDA 13 stack as a pip
constraints file.

### Kernels

- Pallas Mosaic GPU kernels, forward and backward, for Mamba-2, Mamba-3 (SISO and MIMO), Mamba-1,
  Gated DeltaNet and DeltaNet, Gated DeltaNet-2, KDA, RWKV-7, GLA and mLSTM, with packed documents
  (`segment_ids`), grouped heads, and initial and final states.
- An FA3-style softmax attention kernel (causal, sliding windows, packed documents, GQA, sinks) with a
  deterministic default backward and an opt-in fused one-kernel backward, an FP8 attention backward,
  Multi-head Latent Attention, and DeepSeek Sparse Attention with an exact top-k kernel.
- A grouped-GEMM kernel for dropless MoE, a fused cross-entropy kernel and fused FP8 SwiGLU kernels.
- One fail-closed dispatcher (`impl="auto"`): every recurrence also ships a token-by-token reference
  and a chunk-parallel JAX op, and a kernel runs only for the calls its GPU tests cover.
- Plain-JAX or opt-in paths for log-linear Mamba-2, Gated DeltaProduct, HGRN2, RWKV-6, TTT-Linear,
  MiniMax-01 lightning attention, FoX, PaTH attention, MoBA and NSA sparse attention.

### Training

- Models: a Llama/Qwen3-style transformer, Mamba-2, Mamba-3, Gated DeltaNet, Gated DeltaNet-2, KDA,
  RWKV-7 and hybrids of any of them with attention (including Kimi Linear's and Qwen3-Next's layouts),
  mixture-of-experts and multi-token prediction; presets from `tiny` CPU models to FLA's published
  1.3B shapes.
- One compiled train step with fp32 gradient accumulation and non-finite step rejection; data, FSDP,
  tensor, expert and pipeline parallelism, and context-parallel attention as a library; FP8 training
  with delayed scaling; remat policies and memory-lean VJPs; crash-consistent checkpoints with exact
  resume; memory-mapped, streamed and SFT data; DPO preference tuning; JSONL, Weights & Biases and
  TensorBoard logging.
- Generation, perplexity and lm-evaluation-harness evaluation; Hugging Face import and export.
- Command line: `jla train`, `show-config`, `tokenize`, `generate`, `eval`, `convert` and `dpo`.

### Performance

Measured on H100s with JAX 0.11.2 on 20 Oct 2026, against the latest upstream code of that day. Kernel
ratios are forward + backward, the other implementation's time over ours.

- `transformer-1p3b` in FP8: 72.3% MFU on one H100 (77,094 tokens/s) and 68.2% MFU per GPU on 8 H100s
  with FSDP 8 (72,709 tokens/s per GPU, 1.47x flame).
- Against flame (FLA's trainer) on one H100 with the same tokens: 1.04-1.67x in bf16 and 1.10-1.77x in
  FP8 for the 1.3B transformer, sliding-window, Gated DeltaNet, KDA and Mamba-2 presets, and 1.55x for
  Mamba-3 in bf16.
- In the Gated DeltaNet-2 paper's Figure 2 setting (1.3B hybrids on one H100, 16K tokens per step):
  every model of the figure 1.14-1.41x the paper's numbers with a 32,000-token vocabulary. Gated DeltaNet-2 runs at 52.1 and 49.1 thousand tokens/s at 2K × 8 and 16K × 1, against
  38.0 and 36.1 ([gdn2_paper.md](benchmarks/gdn2_paper.md)).
- Kernels: the Mamba-2 scan 1.20-1.96x the official Triton kernels (the whole Mamba-2 layer 1.17-1.41x
  the official fused training path), Mamba-3 SISO 1.58-2.32x and MIMO 2.70-5.37x the official
  recurrence kernels, Mamba-1 1.72-1.85x the official CUDA selective scan (`d_state` 16), GLA 1.52-1.77x
  FLA (dense), KDA 1.47-1.78x FLA, RWKV-7 1.06-1.46x FLA, Gated DeltaNet 1.71-2.68x FLA's Triton kernels
  and 0.73-1.26x FlashQLA, and MLA 1.01-1.05x cuDNN.

Every shape, baseline commit and method: [docs/results.md](docs/results.md),
[docs/kernels.md](docs/kernels.md) and [benchmarks/](benchmarks/README.md).

### Known limitations

- Slower than FlashQLA for Gated DeltaNet on long single sequences and with packed documents at few
  heads, slower than FLA for Gated DeltaNet at head dim 64 with packed documents in short rows, slower
  than FA3 for causal softmax attention, and slower than FlashKDA for KDA inference; see
  [Where we lose](README.md#where-we-lose).
- Hopper only, with one pinned JAX version. The Gated DeltaNet piece-scan kernel, the Mamba-2 row
  kernels and the FP8 SwiGLU kernels use Pallas Triton, which JAX 0.11.2 runs with a
  DeprecationWarning; they will move to Mosaic GPU before the JAX pin moves.
