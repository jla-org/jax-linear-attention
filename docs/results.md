# Results

[README](../README.md) · [Architectures](architectures.md) · [Kernels](kernels.md) · [Training](training.md) · [Results](results.md) · [Benchmarks](../benchmarks/README.md)

The 1.3B end-to-end results on one and eight H100s against flame, FP8 against bf16, MFU, the hybrid
layouts, and the index of every benchmark page.

- [Summary](#summary): [Headlines](#headlines) · [Where we lose](#where-we-lose)
- [Results: 1.3B end to end](#results-13b-end-to-end): [One H100](#one-h100) · [FP8 against bf16](#fp8-against-bf16) · [MFU and ceilings](#mfu-and-ceilings) · [8 GPUs](#8-gpus) · [Hybrid layouts](#hybrid-layouts)
- [Other results](#other-results)
- [Benchmarks](#benchmarks)

## Summary

### Headlines

H100, JAX 0.11.2, measured on 20 Oct 2026 with no other jobs on the GPUs (details below and in
[Kernels](kernels.md#kernels)):

- `transformer-1p3b` in FP8 on one H100, the TOML as written: **72.3% MFU** (77,094 tokens/s) by the
  wall clock.
- The same TOML on 8 H100s (FSDP 8): **68.2% MFU per GPU** (72,709 tokens/s per GPU), 1.47x flame. MFU is
  model FLOPs (bf16-equivalent for FP8 runs) over the 989 TFLOP/s bf16 spec.
- Against flame (FLA's trainer, with `torch.compile` where it works) on one H100, the same tokens and 65,536
  tokens per step: 1.04-1.67x in bf16 and 1.10-1.77x in FP8 against flame's FP8 for the transformer,
  sliding-window, Gated DeltaNet, KDA and Mamba-2 presets, and 1.55x for Mamba-3 in bf16.
- Kernels against the strongest latest upstream code, forward + backward (the other implementation's time
  over ours): the Mamba-2 scan 1.20-1.96x the official Triton kernels (the whole Mamba-2 layer 1.17-1.41x
  the official fused training path), Mamba-3 SISO 1.58-2.32x and MIMO 2.70-5.37x the official
  recurrence kernels, Mamba-1 1.72-1.85x the official CUDA selective scan (`d_state` 16), GLA 1.52-1.77x
  FLA (dense), KDA 1.47-1.78x FLA, RWKV-7 1.06-1.46x FLA, Gated DeltaNet 1.71-2.68x FLA's Triton kernels
  and 0.73-1.26x FlashQLA.

### Where we lose

Softmax attention against FA3 (the forward is 0.90-0.92x FA3 on causal attention, forward + backward
0.77-0.92x with the default backward and 0.87-1.00x with the opt-in fused one); Gated DeltaNet against
FlashQLA at long single sequences (one row of 16K-64K tokens: 0.69-0.78x forward + backward at 8 heads,
0.68-0.73x at 16), with packed documents at small `B * H` (0.58-0.77x at one row of 8 heads) and in the
backward at large `B * H` (0.83x forward + backward at `(8, 2048, 32)`, where the forward is 1.07x); and KDA
inference against FlashKDA (0.57x at `(8, 2048, 32)`). [Kernels](kernels.md#where-we-lose) has every case
and the reason.

## Results: 1.3B end to end

### One H100

One H100 80GB SXM per run, measured on 20 Oct 2026 with no other jobs on the GPUs (the one-GPU runs side by
side, one per GPU, sharing the host): FineWeb-Edu tokenized with the Llama-3.1 tokenizer, 4096-token
sequences with document masking, 65,536 tokens per step, Mosaic GPU kernels on
(`benchmarks/bench_e2e_1p3b.py --impl kernels`), each TOML as written (the two transformer TOMLs train
without recompute at 4 × 4096 × `grad_accum` 4, the recurrent presets at 16 × 4096 with full
rematerialization), XLA's BFC allocator, 20 steps, tokens/s by the wall clock from the completion of step 5
to that of step 19. FP8 adds `model.fp8 = "delayed"`, `fp8_min_features = 512` and `fp8_fused = true`. One
run per configuration; every loss falls from about 12.1 to 7.8-8.0 in 20 steps.

flame: fla-org/flame main `24dea91` with FLA main `6dff3e1` and torchtitan `0b44d4c`, on the same day, the
same shapes, the same tokens in the same order and the same 65,536 tokens per step, in flame's recommended
mode: `torch.compile`, no activation checkpointing, the step split into 4, 8 or 16 micro-steps (the fewest
that fit). flame's FP8 is torchtitan's float8 converter (torchao 0.18.0 `Float8Linear`, tensorwise dynamic
scaling, every linear layer but the LM head), run twice: on every linear layer, and with layers narrower
than 512 features left in bf16 as ours are; the faster of the two is the baseline. Every commit and setting:
[the flame baseline](../benchmarks/end_to_end_1p3b.md#the-flame-baseline).

| preset | ours, bf16: tokens/s (MFU) | flame, bf16 | bf16 / flame | ours, FP8: tokens/s (MFU) | flame, FP8 | FP8 / flame FP8 |
|---|---|---|---|---|---|---|
| `transformer-1p3b` | 60,178 (56.4%) | 52,165 (48.9%) | 1.15x | **77,094 (72.3%)** | 54,871 (51.5%) | 1.41x |
| `transformer-swa-1p3b` | 55,188 (50.1%) | 53,162 (48.2%) | 1.04x | 68,072 (61.8%) | 56,339 (51.1%) | 1.21x |
| `gdn-hd128-1p3b` | 42,995 (42.3%) | 25,723 (25.3%) | 1.67x | 55,956 (55.0%) | 36,694 (36.1%) | 1.52x |
| `gdn-1p3b` (`K = 256`, `V = 512`) (d) | 37,759 (36.0%) | 28,172 (26.9%) | 1.34x | 46,971 (44.8%) | 38,482 (36.7%) | 1.22x |
| `kda-1p3b` | 35,110 (32.9%) | 23,487 (22.0%) | 1.49x | 42,756 (40.1%) | 24,168 (22.6%) | 1.77x |
| `mamba2-1p3b` | 38,899 (37.1%) | 37,144 (35.4%) (a) | 1.05x (a) | 43,688 (41.7%) | 39,845 (38.0%) (a) | 1.10x (a) |
| `mamba3-1p3b` | 32,976 (32.6%) | 21,221 (21.0%) (b) | 1.55x | 36,344 (36.0%) | 10,661 (10.5%) (b) | 3.41x (b) |
| `rwkv7-1p3b` | 28,484 (26.2%) | not a valid baseline (c) | - | 30,651 (28.2%) | 16,143 (14.9%) (c) | 1.90x (c) |

- (a) flame's Mamba-2 trains **without document masking**: FLA's Mamba-2 layer ignores `cu_seqlens`, while
  ours masks documents.
- (b) flame's Mamba-3 cannot compile (inductor fails on the Mamba-3 block) and fails under activation
  checkpointing (the official SISO kernel unpacks its saved tensors twice), so it runs in its only working
  mode: no compile, no checkpointing, one sequence per micro-step. Its FP8 run is half as fast as its bf16
  run; against that bf16 run our FP8 is 1.71x.
- (c) flame's compiled RWKV-7 in bf16 trains to a much worse loss (9.31 at step 20, against 7.53 for its FP8
  run and 7.88 for ours), so it is not a valid baseline and there is no bf16 comparison (it ran at 23,094
  tokens/s). Its FP8 run trains normally but is slower than that bf16 run; the 1.90x is against the FP8 run.
  flame's records carry no FLOP count for RWKV-7, so its MFU uses ours (the same model).
- (d) `gdn-1p3b` ran the same day on the code just before the last kernel changes of this release, which
  leave its `K = 256` kernels as they were (on one GPU those changes moved the other presets by 0.98-1.03x).
- flame's `kda-1p3b` runs were measured separately on 20 Oct, with the same harness and protocol and no
  other jobs on the GPUs.
- GDN-2 is implemented clean-room from the equations of arXiv 2605.22791, so it has no flame baseline.
  `gdn2-1p3b` trains at 30,300 tokens/s (33.70% MFU) in bf16 and 37,010 (41.16%) in FP8 (provisional,
  interleaved pairs on one H100).

### FP8 against bf16

FP8 against bf16 in the table above: 1.28x for the transformer, 1.23x with the sliding window, 1.30x for
`gdn-hd128-1p3b`, 1.24x for `gdn-1p3b`, 1.22x for KDA, 1.12x for Mamba-2, 1.10x for Mamba-3 and 1.08x for
RWKV-7. Our runs peak at 41.9-54.7 GiB in bf16 and 44.5-52.0 GiB in FP8; flame's at 50.9-66.9 GiB and
49.9-69.6 GiB of the 80 GB. FP8 makes the projection and MLP GEMMs 1.9x faster, but they are about half of a
bf16 step, and it gains least on Mamba-2, Mamba-3 and RWKV-7 (1.08-1.12x; the Mamba presets have 48 layers
without an MLP). The recipe, what stays in bf16 and the fused FP8 passes:
[fp8_1p3b.md](../benchmarks/fp8_1p3b.md). On `gdn-1p3b` the `K = 256` kernels train 2.08x faster than the
plain-JAX chunked ops (provisional).

### MFU and ceilings

MFU here is model FLOPs per second (jla's analytic count, the same for flame) over 989 TFLOP/s, the H100
SXM dense bf16 spec, which assumes a 1,830 MHz clock. Under sustained tensor-core load at its 700 W power
limit an H100 sustains 678-692 TFLOP/s in bf16 (median 681, 68.5-70.0% of the spec) and 1,236 TFLOP/s in
FP8, 125% of the bf16 spec ([ceilings on H100](../benchmarks/profile_1p3b.md#ceilings-on-h100)); FP8 runs
report bf16-equivalent MFU. The same page models an ideal 1.3B step (no recompute, every elementwise pass
fused, every op at its measured ceiling): 69.7% bf16 MFU for the transformer, 52.7-62.7% for the recurrent
presets, and 77-119% bf16-equivalent with FP8 GEMMs. At 9.28 GFLOP per token, 70% MFU is a 0.878 s
`transformer-1p3b` step; its TOML reaches 72.3% ([crossing 70% MFU](../benchmarks/fp8_1p3b.md#crossing-70-mfu)).
The FLOP count assumes full causal context, so with about 4 documents per 4096-token record the MFU of
`transformer-1p3b` is about 8% (relative) higher than the work done; tokens/s is unaffected
([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#where-we-lose-and-limits)).

### 8 GPUs

One 8x H100 80GB HBM3 machine (NVLink), FSDP 8, 16 × 4096 tokens per GPU (524,288 per step), measured on 20
Oct 2026 with no other jobs on the GPUs: the same harness, timing window and data as the one-GPU table, each
TOML as written, FP8 as above, every kernel per shard of the mesh. Both rows ran on the code just before the
last kernel changes of this release (whose one-GPU FP8 runs, 76,584 tokens/s for `transformer-1p3b` and
56,099 for `gdn-hd128-1p3b`, are the references of "per GPU vs ours on 1 GPU"). flame: the same commits
with FSDP2, `torch.compile` and 4 micro-steps, FP8 as above (its float8 all-gather for FSDP failed to run, so
flame gathers its weights in bf16).

| preset | ours, FP8: tokens/s per GPU (MFU) | per GPU vs ours on 1 GPU | flame, FP8: tokens/s per GPU (MFU) | ours / flame |
|---|---|---|---|---|
| `transformer-1p3b` | 72,709 (68.2%) | 0.95x | 49,475 (46.4%) | 1.47x |
| `gdn-hd128-1p3b` | 53,208 (52.3%) | 0.95x | 33,997 (33.4%) | 1.57x |

The `transformer-1p3b` row ran with the CUDA async allocator and the `gdn-hd128-1p3b` row with XLA's default
BFC allocator, timed by the harness's mean step over steps 6-20. With the CUDA async allocator
(`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) `gdn-hd128-1p3b` runs 1.80x slower on 8 GPUs (29,520 tokens/s per
GPU, 29.0% MFU, 0.87x flame), so keep BFC.

- **Every 1.3B preset on 8 GPUs** (provisional, bf16 against flame's bf16, before the train options below):
  8 GPUs keep 0.91-0.96x of one GPU's tokens/s per GPU in every row, from 0.95x flame for the transformer at
  full rematerialization to 1.48x for KDA
  ([multi_gpu.md](../benchmarks/multi_gpu.md#every-13b-preset-on-8-gpus)).
- **Train options for several GPUs** (off by default; the `transformer-1p3b` TOML sets all three;
  provisional, [multi_gpu.md](../benchmarks/multi_gpu.md#train-options-for-several-gpus)):
  `train.grad_reduce = "step"` (one gradient reduce-scatter per step) took the TOML from 64.4% to 68.2-68.6%
  MFU per GPU (1.060x and 1.064x in two pairs); `train.update_guard = "select"` (the rejected-step rule as an
  elementwise select, so the host does not wait for every step); `train.dispatch_ahead = true` alone 0.992x
  and 1.004x (1.017x on one GPU). `"step"` keeps about 10 bytes per parameter on every GPU whatever the FSDP
  size (about 30 GB at 3B; 7B does not fit on 80 GB), so it is for models up to about 3B.
- **XLA settings**: on several GPUs set `runtime.xla_tuning_level = 2` (pipelined collectives) for
  `mamba2-1p3b` (+9% at FSDP 8); for the other presets it measured -4.9% to +2.2% against tier 1, which every
  1.3B TOML sets.

### Hybrid layouts

`qwen3-next-1p3b` and `kimi-linear-1p3b` follow Qwen3-Next-80B-A3B's and Kimi-Linear-48B-A3B's layouts at
half their hidden size, with MoE layers on the Mosaic GPU grouped GEMM and per-device dispatch
([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts)). flame has no
model for either. Provisional (other jobs could share the machine), tokens/s per GPU (MFU of the active
parameters' FLOPs):

| preset | GPUs | setting | tokens/s per GPU (MFU) |
|---|---|---|---|
| `qwen3-next-1p3b` | 1 | the TOML (forward-gradient cross entropy on the main and MTP heads) | 73,248 (26.7%); 77,797 (28.4%) with no other jobs on the machine |
| `qwen3-next-1p3b` | 8, FSDP 8 | the TOML | 71,926 (26.2%) |
| `kimi-linear-1p3b` | 1 | the TOML (8 × 4096 × 2, XLA cross entropy) | 67,674 (22.7%) |
| `kimi-linear-1p3b` | 1 | 16 × 4096, forward-gradient cross entropy | 73,480 (24.6%) |
| `kimi-linear-1p3b` | 8, FSDP 8 | the TOML | 61,553 (20.6%) |
| `kimi-linear-1p3b` | 8, FSDP 8 | 16 × 4096, forward-gradient cross entropy | 68,672 (23.0%) |

The MoE layers take 29.2% (`qwen3-next-1p3b`) and 30.5% (`kimi-linear-1p3b`) of the step, about half of it
XLA gathers into and out of expert order, so both stay below the 35% MFU target. FP8 (the dense projections
only) is about even with bf16 (1.003x and 1.017x). Expert parallelism over 8 GPUs, measured on the MoE layer
without the per-device dispatch, trained `qwen3-next-1p3b` at 19,866 tokens/s per GPU (7.2% MFU, 0.38x the
one-GPU rate), where FSDP 8 trains it at 71,926.

### In the Gated DeltaNet-2 paper's setting

The Gated DeltaNet-2 paper's Figure 2 times seven 1.3B models (hybrids of a recurrent mixer, MLP, 2K
sliding-window attention and MLP, and a transformer) on one H100 at 16,384 tokens per step.
- **Against the figure:** in that setting every model trains 1.14-1.41x faster with a 32,000-token
  vocabulary, and 0.99-1.23x with Llama-3's 128,256. The paper does not state its tokenizer.
- **Gated DeltaNet-2:** its hybrid runs at 52.1 and 49.1 thousand tokens/s at 2K × 8 and 16K × 1, against
  the 38.0 and 36.1 the paper's text prints. The paper's other values are read from its figure's vector
  data.
- **Mamba-3 MIMO:** its hybrid is the slowest, as in the paper: 1.14-1.39x the paper's numbers with
  32,000 tokens and 0.99-1.23x with 128,256.
- **Caveat:** different frameworks and harnesses produced the two sets of numbers
  ([gdn2_paper.md](../benchmarks/gdn2_paper.md) has every model and setting).

## Other results

- **125M training** ([end_to_end.md](../benchmarks/end_to_end.md), provisional): six release architectures
  train 300 steps on FineWeb-Edu at 140,241-206,945 tokens/s on one H100, and the kernels end within 0.01 of
  the chunked ops' eval losses.
- **Initialisation** (a loss comparison, not a speed one; [init_parity.md](../benchmarks/init_parity.md)):
  at 125M on FineWeb-Edu after 2000 steps the upstream-parity initialisation lowers the held-out eval loss by
  0.0972 for DeltaNet, 0.1088 for Gated DeltaProduct, 0.100 and 0.103 for RWKV-7 (two seeds), 0.0387 for
  Mamba-3 SISO and 0.0195 for Mamba-2. FLA's N(0, 0.02) short convs did not help the attention hybrids
  (+0.0160 ± 0.0131 for the Gated DeltaNet hybrid, a tie, and +0.0172 ± 0.0042 for the KDA hybrid, worse at
  every evaluation), so those presets keep identity short convs; without attention layers the same convs
  help (0.103 lower for Gated DeltaNet, 0.047 for KDA), so the all-linear `gdn-1p3b`, `gdn-hd128-1p3b` and
  `kda-1p3b` keep them.
- **MoE** ([ragged_dot.md](../benchmarks/ragged_dot.md), provisional): with the persistent grouped GEMM and
  the gather-only token permutation the train steps of `qwen3-next-300m`, `kimi-linear-1b` and
  `granite-4.0-h-125m` are 1.24x, 1.07x and 6.48x faster than the faster of the other two MoE paths.
- **DPO** ([dpo.md](../benchmarks/dpo.md)): the loss, its gradient and the reward metrics match TRL 1.16's
  `DPOTrainer` on fixed inputs; `configs/post_train/dpo-tiny.toml` (a toy run that checks the objective
  trains) reaches a train loss of 0.0145, an eval loss of 0.0137 and reward accuracy 1.0 at step 100.

## Benchmarks

[benchmarks/README.md](../benchmarks/README.md) explains the timing method, summarises every kernel and
gives the commands.

- Kernels: [Mamba-2](../benchmarks/mamba2.md), [Mamba-3 SISO and MIMO](../benchmarks/mamba3.md),
  [Mamba-1](../benchmarks/mamba1.md), [Gated DeltaNet](../benchmarks/gdn.md) (with
  [packed documents](../benchmarks/gdn.md#packed-documents) and [`K = 256`](../benchmarks/gdn.md#head-dim-256)),
  [KDA and Gated DeltaNet-2](../benchmarks/kda.md), [RWKV-7](../benchmarks/rwkv7.md), [GLA](../benchmarks/gla.md),
  [mLSTM](../benchmarks/mlstm.md), [Gated DeltaProduct](../benchmarks/gated_deltaproduct.md),
  [log-linear Mamba-2](../benchmarks/log_linear.md).
- Attention: [softmax attention and context parallelism](../benchmarks/attention.md),
  [Multi-head Latent Attention](../benchmarks/mla.md), [DeepSeek Sparse Attention](../benchmarks/dsa.md),
  [FoX](../benchmarks/fox.md), [long contexts](../benchmarks/long_context.md).
- Kernels around the mixers: [short convolution](../benchmarks/short_conv.md),
  [fused cross entropy](../benchmarks/cross_entropy.md), [fused SwiGLU GEMMs](../benchmarks/fused_gemm.md),
  [grouped GEMM for MoE](../benchmarks/ragged_dot.md), [FP8 layers](../benchmarks/fp8.md),
  [layout copies and glue](../benchmarks/zero_copy_layouts.md).
- End to end: [1.3B against flame](../benchmarks/end_to_end_1p3b.md), [FP8 at 1.3B](../benchmarks/fp8_1p3b.md),
  [step profiles and ceilings](../benchmarks/profile_1p3b.md), [8 GPUs](../benchmarks/multi_gpu.md),
  [125M training runs](../benchmarks/end_to_end.md), [model steps](../benchmarks/models.md),
  [initialisation A/B](../benchmarks/init_parity.md).
- Training features: [pipeline parallelism](../benchmarks/pipeline.md),
  [expert parallelism](../benchmarks/expert_parallel.md), [Hugging Face conversion](../benchmarks/convert_hf.md),
  [DPO](../benchmarks/dpo.md), [SFT packing](../benchmarks/sft_packing.md), [streaming data](../benchmarks/streaming.md).
