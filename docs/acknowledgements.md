# Acknowledgements

[README](../README.md) · [Architectures](architectures.md) · [Kernels](kernels.md) · [Training](training.md) · [Results](results.md) · [Benchmarks](../benchmarks/README.md)

## Acknowledgements and licenses

jax-linear-attention is Apache-2.0 (see `LICENSE`, and `NOTICE` for every adapted project with its
licence text in `licenses/`). The Mosaic GPU kernels port the Triton kernels of
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (MIT) and
[Mamba](https://github.com/state-spaces/mamba) (Apache-2.0), including Mamba's TileLang MIMO
kernels; the Mamba layers follow the official modules. Gated DeltaNet-2 is implemented from the
equations of [arXiv:2605.22791](https://arxiv.org/abs/2605.22791). The fused Gated DeltaNet kernels
follow [FlashQLA](https://github.com/QwenLM/FlashQLA) (MIT), the KDA forward adapts ideas from
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) (MIT), and the short-convolution kernels adapt
[FLAT](https://github.com/trainstation-project/flat) (Apache-2.0). The softmax attention kernel
adapts the Pallas Mosaic GPU examples of [JAX](https://github.com/jax-ml/jax) (Apache-2.0) and
[Tokamax](https://github.com/openxla/tokamax) (Apache-2.0), with its named barriers adapted from Tokamax's, and its forward's intra-warpgroup overlap,
warp-scheduler turns and 192-row tiles, its backward block ranges and its fused backward after
[FlashAttention-3](https://github.com/Dao-AILab/flash-attention) (BSD-3-Clause, design only); the grouped GEMMs (bf16 and the FP8 forward), fused SwiGLU GEMM and fused cross-entropy kernels adapt JAX's
ragged-dot and Hopper matmul examples, and the grouped GEMM's aligned row schedule adapts Tokamax's
(its persistent schedule follows the persistent-kernel design of
[CUTLASS](https://github.com/NVIDIA/cutlass), BSD-3-Clause, as an idea only). Forming the LM-head
gradient in the forward pass follows the design of
[Liger Kernel](https://github.com/linkedin/Liger-Kernel) (BSD-2-Clause) and FLA's fused linear
cross entropy, and the cross-entropy page compares against
[Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy) (no code copied from either). The
Mamba-1 kernels (the forward and both backward kernels) follow the design of Mamba's CUDA selective scan, and the GLA kernels the stage split
of FLA's `chunk_gla`. The DPO losses and reward metrics follow
[TRL](https://github.com/huggingface/trl) (Apache-2.0) and are checked against it; the DPO options
and preference preprocessing follow MaxText. The `"upstream"` initialisation reimplements the
schemes of Mamba, FLA and RWKV-LM. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and
[FlashMLA](https://github.com/deepseek-ai/FlashMLA) (MIT) are called as DSA baselines only.
The FP8 recipe follows [MaxText](https://github.com/AI-Hypercomputer/maxtext)
(Apache-2.0) and [Flax](https://github.com/google/flax) (Apache-2.0), and the fused FP8 SwiGLU
passes follow the design of [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
(Apache-2.0, no code copied). Pipeline parallelism follows MaxText, and the pure-JAX Gated DeltaNet
baseline in `benchmarks/` is copied from it. Hugging Face import/export, RoPE scaling, the
MiniMax-01 lightning attention and the DeepSeek-V3 router follow
[transformers](https://github.com/huggingface/transformers) (Apache-2.0), and the MoE balance-loss
mapping follows its `load_balancing_loss_func`. The PyTorch end-to-end
baseline runs [flame](https://github.com/fla-org/flame) (MIT), and the RWKV-7 test oracles follow
[RWKV-LM](https://github.com/BlinkDL/RWKV-LM) (Apache-2.0).

The RWKV-6, RWKV-7, HGRN2, Gated DeltaProduct, FoX, PaTH, NSA, DSA, log-linear and TTT-Linear
layers or reference ops, and the context-parallel state passing, adapt code or designs from Flash
Linear Attention; the Mamba-1 and Mamba-3 MIMO layers follow the official Mamba modules.
Multi-head Latent Attention follows [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
(MIT), the DSA indexer [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) (MIT; the top-k's
tie order and the FP8 indexer option's per-token scaling follow its `Indexer` and `fp8_index_kernel`),
MoBA [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) (MIT) and the mLSTM layer
[xLSTM](https://github.com/NX-AI/xlstm) (Apache-2.0). The mLSTM recurrence is written from the
equations of the xLSTM papers; no code from `mlstm_kernels` (NXAI Community License) is used.
