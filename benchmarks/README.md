# Benchmarks

[README](../README.md) · [Architectures](../docs/architectures.md) · [Kernels](../docs/kernels.md) · [Training](../docs/training.md) · [Results](../docs/results.md) · [Benchmarks](README.md)

Speed and correctness of the jax-linear-attention 0.1.0 kernels and training stack on NVIDIA H100, each
against the latest upstream code. Every page below has one layout (setup, results, where we lose,
correctness and the commands that reproduce it, most with a short note on how it works); this page
summarises them.

- [Method](#method) · [Kernels](#kernels) · [End to end](#end-to-end) · [Pages](#pages) · [Reproduce](#reproduce)

## Method

- **Hardware and software**: H100 80GB HBM3 (SXM), JAX 0.11.2, bf16 inputs unless a page says otherwise.
  The PyTorch baselines (torch 2.13.0+cu129, Triton 3.7.1) run in their own environments, as a subprocess
  (`torch_baselines.py`; `torch_fa3.py` for FlashAttention-3).
- **Kernel timing**: GPU kernel time per call, which leaves out host dispatch and idle gaps, so XLA, Mosaic
  GPU, Triton and torch code compare on the same terms. JAX is timed with CUPTI (`harness.py`: the median of
  at least 30 calls after 3 warmups), torch with `torch.profiler` device times; every FA3 time is
  cross-checked against a CUDA-event time per call. The implementations run interleaved, round-robin on one
  GPU. `fwd+bwd` is one forward plus the backward for a fixed output cotangent.
- **Ratio** = baseline time / ours, so > 1 means ours is faster.
- **End to end**: `bench_e2e_1p3b.py` (jla) and `bench_e2e_flame.py` (flame, FLA's trainer) on FineWeb-Edu
  with document masking, the same tokens in the same order, tokens/s by the wall clock from the completion of
  step 5 to that of step 19; MFU is model FLOPs over the 989 TFLOP/s dense bf16 peak, the same count for both.
- **Baselines**: FLA main `6dff3e1` (main `9d478fa` changes none of the compared kernels), FlashQLA main
  `5538c35`, FlashKDA main `7afb9f4`, state-spaces/mamba main `e9594ce`, FlashAttention-3 main `965c921`,
  cuDNN 9.27 as shipped with JAX 0.11.2, and flame `24dea91` with torchtitan `0b44d4c`. Each page gives the
  exact commits and versions of its baselines.
- **When**: measured on 20 Oct 2026 with no other jobs on the GPUs. Numbers marked provisional were measured
  while other jobs could share the machine, on the date their page gives.

## Kernels

Ranges over each page's shapes; "a / b" in a cell follows the two baselines of its row, and "-" means not timed.

**Mamba family**

| Kernel | Baseline (commit) | Forward | Forward + backward |
|---|---|---|---|
| Mamba-2 (SSD) scan, [mamba2.md](mamba2.md) | official Triton `mamba_chunk_scan_combined` (`e9594ce`) | 1.31-2.60x | 1.20-1.96x |
| | FLA `chunk_simple_gla`, Triton (`6dff3e1`) | 1.49-3.18x | 1.58-2.20x |
| Mamba-2 whole layer, [mamba2.md](mamba2.md#the-whole-layer-against-the-official-fused-training-path) | the official `Mamba2` module's fused training path (`e9594ce`) | 1.12-1.34x | 1.17-1.41x |
| Mamba-3 SISO, [mamba3.md](mamba3.md) | official Triton `mamba3_siso_combined` (`e9594ce`) | 1.45-2.60x | 1.58-2.32x |
| Mamba-3 MIMO, [mamba3.md](mamba3.md#mimo) | official TileLang kernels (`e9594ce`): the full call / its recurrence kernels only | 1.85-5.57x / 1.57-4.70x | 3.44-6.09x / 2.70-5.37x |
| Mamba-1, `d_state` 16, [mamba1.md](mamba1.md) | official CUDA selective scan (`e9594ce`) | 1.19-1.63x | 1.72-1.85x |

**Delta rule, RWKV-7, GLA and mLSTM**

| Kernel | Baseline (commit) | Forward | Forward + backward |
|---|---|---|---|
| Gated DeltaNet, `K, V` in {64, 128}, [gdn.md](gdn.md) | FlashQLA main `5538c35` (FLA's default dispatch at `K = V = 128`) | 0.79-1.69x | 0.73-1.26x |
| | FLA Triton (`6dff3e1`) | 1.74-3.14x | 1.71-2.68x |
| | MaxText main `34ffea83`, pure JAX, dense (provisional) | 14.9-20.2x | 17.4-23.2x |
| Gated DeltaNet, `K = 256, V = 512`, [gdn.md](gdn.md#head-dim-256) | FLA default dispatch / Triton (`6dff3e1`) | 2.37-3.05x | 1.17-1.24x / 1.23-1.29x |
| KDA, [kda.md](kda.md) | FLA default dispatch / Triton (`6dff3e1`), training | 1.25-1.57x dense (provisional), 1.37-1.52x with 16 packed documents | 1.47-1.78x / 1.51-1.91x dense, 1.61-1.63x / 1.73-1.77x with 16 packed documents |
| | FlashKDA main `7afb9f4`, inference contract (forward only) | 0.57-2.30x | - |
| Gated DeltaNet-2, [kda.md](kda.md) | our plain-JAX `chunked` op (clean-room: no upstream code) | 6.3-8.1x | 6.7-10.8x |
| RWKV-7, [rwkv7.md](rwkv7.md) | FLA `chunk_rwkv7` as `RWKV7Attention` calls it (`6dff3e1`) | 1.09-1.52x | 1.06-1.46x |
| GLA, [gla.md](gla.md) | FLA `chunk_gla`, Triton (`6dff3e1`) | 1.69-2.19x dense, 0.88-1.39x packed | 1.52-1.77x dense, 1.01-1.41x packed |
| mLSTM, opt-in, [mlstm.md](mlstm.md) | NX-AI `mlstm_kernels` 2.0.6 TFLA Triton | 1.38-1.42x | 1.03-1.06x |
| Gated DeltaProduct, opt-in, [gated_deltaproduct.md](gated_deltaproduct.md) | our plain-JAX `chunked` op (provisional) | - | 11.4x |

**Attention**

| Kernel | Baseline (commit) | Forward | Forward + backward (default backward; opt-in fused backward vs FA3 at `D` 128 / `D` 64) |
|---|---|---|---|
| Softmax attention, causal, [attention.md](attention.md) | FA3 main `965c921` / cuDNN 9.27 | 0.90-0.92x / 0.96-1.09x | 0.77-0.92x / 0.94-1.07x; fused 0.87-0.89x / 0.87-1.00x |
| 4 packed documents per row | FA3 with `cu_seqlens` / cuDNN with the dense mask | 0.80-1.04x / 3.48-7.85x | 0.79-1.29x / 3.48-7.61x; fused 0.82-0.94x / 0.89-1.31x |
| 128-token window | FA3 / cuDNN | 0.88-1.09x / 0.86-0.94x | 1.10-1.38x / 1.15-1.27x; fused 0.92-0.96x / 1.21-1.34x |
| 1024- and 4096-token windows | FA3 / cuDNN | 0.83-1.04x / 0.96-1.16x | 0.81-0.96x / 0.93-1.12x; fused 0.81-0.87x / 0.88-1.02x |
| MLA (q/k 192, v 128), causal, [mla.md](mla.md) | cuDNN 9.27 in MLA mode / FA3 `965c921` built for 192/128 | 1.06-1.19x / 0.95-0.97x | 1.01-1.05x / 1.01-1.06x |
| MLA, packed documents | cuDNN's packed layout / FA3 | 1.53-1.86x / 0.88-0.96x | 1.71-1.83x / 1.08-1.20x |
| DSA indexer + top-k, [dsa.md](dsa.md) (provisional) | DeepGEMM `1fc9c66` FP8 logits + `torch.topk`, kernel time, 16K-128K tokens | 1.16-1.43x | - |

**Kernels around the mixers** (all provisional, measured 17-19 Oct 2026)

| Kernel | Baseline | Forward | Forward + backward |
|---|---|---|---|
| Short causal conv, [short_conv.md](short_conv.md) | XLA / causal-conv1d 1.7.0 | 1.5-1.8x / 0.76-1.27x | 2.4-2.9x / 1.2-1.7x |
| Short conv + q/k L2 norm, [short_conv.md](short_conv.md) | the faster of FLA's Triton conv and causal-conv1d, each with FLA's `l2norm` | 1.78x dense, 1.68x packed | 2.35x dense, 2.06x packed |
| LM-head cross entropy, [cross_entropy.md](cross_entropy.md) | the XLA op, `D` 2048, 128,256-token vocabulary | - | 1.57-1.67x (gradient in the forward pass) |
| Fused SwiGLU GEMMs, opt-in, [fused_gemm.md](fused_gemm.md) | XLA (cuBLASLt GEMM + fusion), 1.3B MLP block | 1.10x | 1.05x |
| Grouped GEMM for MoE, [ragged_dot.md](ragged_dot.md) | `torch._grouped_mm` (CUTLASS, torch 2.13), six MoE shapes | 0.90-1.04x over all GEMMs | 0.94-1.04x |

`impl="auto"` runs every kernel above on H100 for the calls its tests cover, except the opt-in rows and
softmax attention, where it runs our kernel for packed documents and past 2^31 query elements per device
and cuDNN otherwise ([attention.md](attention.md#implementationauto)). Every loss, with its reason:
[docs/kernels.md](../docs/kernels.md#where-we-lose).

## End to end

1.3B presets on one H100, tokens/s (MFU), 65,536 tokens per step, against flame on the same shapes and
tokens ([end_to_end_1p3b.md](end_to_end_1p3b.md)):

| preset | ours, bf16 | vs flame | ours, FP8 | vs flame FP8 |
|---|---|---|---|---|
| `transformer-1p3b` | 60,178 (56.4%) | 1.15x | **77,094 (72.3%)** | 1.41x |
| `transformer-swa-1p3b` | 55,188 (50.1%) | 1.04x | 68,072 (61.8%) | 1.21x |
| `gdn-hd128-1p3b` | 42,995 (42.3%) | 1.67x | 55,956 (55.0%) | 1.52x |
| `gdn-1p3b` (d) | 37,759 (36.0%) | 1.34x | 46,971 (44.8%) | 1.22x |
| `kda-1p3b` | 35,110 (32.9%) | 1.49x | 42,756 (40.1%) | 1.77x |
| `mamba2-1p3b` | 38,899 (37.1%) | 1.05x (a) | 43,688 (41.7%) | 1.10x (a) |
| `mamba3-1p3b` | 32,976 (32.6%) | 1.55x | 36,344 (36.0%) | 3.41x (b) |
| `rwkv7-1p3b` | 28,484 (26.2%) | - (c) | 30,651 (28.2%) | 1.90x |

On 8 H100s (FSDP 8, FP8, 524,288 tokens per step; [multi_gpu.md](multi_gpu.md)): `transformer-1p3b` at
72,709 tokens/s per GPU (68.2% MFU), 1.47x flame, and `gdn-hd128-1p3b` at 53,208 (52.3%), 1.57x.

In the Gated DeltaNet-2 paper's Figure 2 setting (1.3B hybrids, one H100, 16,384 tokens per step), every
model of the figure trains 1.14-1.41x faster than the paper's numbers with a 32,000-token vocabulary, and
0.99-1.23x with Llama-3's 128,256. The Gated DeltaNet-2 hybrid runs at 52.1 and 49.1 thousand tokens/s at
2K × 8 and 16K × 1, against the paper's 38.0 and 36.1. [gdn2_paper.md](gdn2_paper.md) has every model, both
vocabularies and the settings.

(a) flame's Mamba-2 trains without document masking. (b) flame's Mamba-3 runs without `torch.compile`, its
only working mode, and its FP8 run is half as fast as its bf16 run. (c) flame's compiled RWKV-7 in bf16
trains to a much worse loss, so there is no bf16 comparison. (d) Measured the same day on the code just
before the last kernel changes of this release, which leave its `K = 256` kernels as they were.

## Pages

| Kernels | |
|---|---|
| [mamba2.md](mamba2.md) | the Mamba-2 (SSD) scan and the whole Mamba-2 layer |
| [mamba3.md](mamba3.md) | Mamba-3 SISO and MIMO |
| [mamba1.md](mamba1.md) | the Mamba-1 selective scan |
| [gdn.md](gdn.md) | Gated DeltaNet and DeltaNet: the exact sequence split, packed documents, `K = 256` |
| [kda.md](kda.md) | KDA and Gated DeltaNet-2, training and inference |
| [rwkv7.md](rwkv7.md) | RWKV-7, with the cause and fix of NaN losses on its kernel path |
| [gla.md](gla.md) | GLA (and HGRN2's GLA core) |
| [mlstm.md](mlstm.md) | the opt-in mLSTM (xLSTM) path |
| [gated_deltaproduct.md](gated_deltaproduct.md) | Gated DeltaProduct on the Gated DeltaNet kernels |
| [log_linear.md](log_linear.md) | log-linear Mamba-2 (plain JAX) |
| [attention.md](attention.md) | the FA3-style softmax attention kernel, and context-parallel attention |
| [mla.md](mla.md) | Multi-head Latent Attention heads (192 / 128) |
| [dsa.md](dsa.md) | DeepSeek Sparse Attention |
| [fox.md](fox.md) | FoX attention (plain XLA) |
| [long_context.md](long_context.md) | single calls past 2^31 elements |

| Kernels around the mixers | |
|---|---|
| [short_conv.md](short_conv.md) | the fused short convolution and the q/k L2-norm prologue |
| [cross_entropy.md](cross_entropy.md) | the fused LM-head cross entropy |
| [fused_gemm.md](fused_gemm.md) | the fused SwiGLU GEMMs |
| [ragged_dot.md](ragged_dot.md) | the grouped GEMM and the MoE layers |
| [fp8.md](fp8.md) | FP8 dense projections and MLPs |
| [zero_copy_layouts.md](zero_copy_layouts.md) | layout traffic around the kernels, `qk_conv_barrier` and `grouped_key_grads` |

| End to end and training | |
|---|---|
| [end_to_end_1p3b.md](end_to_end_1p3b.md) | 1.3B training against flame: the release table, the presets, the flame baseline, Mamba-2, RWKV-7, the Qwen3-Next and Kimi Linear layouts |
| [gdn2_paper.md](gdn2_paper.md) | training throughput in the Gated DeltaNet-2 paper's Figure 2 setting: 1.3B hybrids at 2K × 8 to 16K × 1 |
| [fp8_1p3b.md](fp8_1p3b.md) | FP8 at 1.3B, activation checkpointing and the XLA settings |
| [profile_1p3b.md](profile_1p3b.md) | where a 1.3B step spends its time, the H100's measured ceilings, a 125M step |
| [multi_gpu.md](multi_gpu.md) | 8 GPUs: FSDP against flame, the train options for several GPUs, every layout against one GPU |
| [end_to_end.md](end_to_end.md) | 125M training runs |
| [models.md](models.md) | RWKV-7 in `LanguageModel`: one training step |
| [init_parity.md](init_parity.md) | the initialisation A/B |
| [pipeline.md](pipeline.md) | pipeline parallelism |
| [expert_parallel.md](expert_parallel.md) | expert parallelism |
| [convert_hf.md](convert_hf.md) | Hugging Face import and export |
| [dpo.md](dpo.md) | DPO against TRL |
| [sft_packing.md](sft_packing.md) | SFT packing |
| [streaming.md](streaming.md) | streaming data |

Scripts without a page: `bench_generate.py` times the prefill memory of `jla.inference.generate`, and
`mla_head_dims_probe.py` checks that `jax.nn.dot_product_attention` in JAX 0.11.2 rejects unequal q/k and v
head dims (why the MLA layer pads the value on its XLA path).

## Reproduce

Install from a clone (`pip install -e ".[cuda,test]"`, or pinned with `-c requirements/jax-0.11.2-cuda13.txt`).
The PyTorch baselines need their own environments: FLA main alone (its Triton path), FLA main with its
optional backends (TileLang 0.1.15, FlashQLA and FlashKDA), state-spaces/mamba built from source, and
FlashAttention-3 built from `hopper/`; the scripts take them as `--torch-python <venv>/bin/python` and
similar flags. Run one benchmark per H100. Each page's Reproduce section has its exact commands, for example:

```bash
python benchmarks/bench_gated_delta_rule.py --kind gated_deltanet --shapes "B,T,H,K,V;..." \
    --impls mosaic_gpu,flashqla,fla,fla_full --rounds 2 --out results/gdn.json \
    --torch-python VENV_FLA_MAIN/bin/python --fla-full-python VENV_FLA_MAIN_FULL/bin/python \
    --qla-python VENV_QLA_MAIN/bin/python --baseline-commit fla=6dff3e1 --baseline-commit flashqla=5538c35
python benchmarks/bench_attention_fa3.py --impls mosaic_gpu,cudnn,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_d128.json
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    python benchmarks/bench_e2e_1p3b.py --arch kda-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --dp 1 --fsdp 1 --steps 20 --skip 5 --allocator bfc --json results/kda-1p3b-1gpu.json
```

By default the scripts write their JSON to `benchmarks/results/`, which is git-ignored; the published
numbers are the tables on these pages.
