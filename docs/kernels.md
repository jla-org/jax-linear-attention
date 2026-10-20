# Kernels and ops

[README](../README.md) · [Architectures](architectures.md) · [Kernels](kernels.md) · [Training](training.md) · [Results](results.md) · [Benchmarks](../benchmarks/README.md)

The ops API, every Mosaic GPU kernel with its speed against the strongest latest upstream code, the kernels
around the mixers, and where we lose.

- [Ops](#ops)
- [Kernels](#kernels): [Mamba family](#mamba-family) · [Delta rule, RWKV-7, GLA and mLSTM](#delta-rule-rwkv-7-gla-and-mlstm) · [Attention](#attention) · [Kernels around the mixers](#kernels-around-the-mixers)
- [Design notes](#design-notes): [Softmax attention routing](#softmax-attention-routing) · [Mamba-3 MIMO baseline](#mamba-3-mimo-baseline) · [Packed documents and states](#packed-documents-and-states) · [Long contexts](#long-contexts) · [Fused short convolution](#fused-short-convolution) · [Fused q/k prologue](#fused-qk-prologue) · [`qk_conv_barrier`, `grouped_key_grads`](#qk_conv_barrier-grouped_key_grads) · [Mamba-2 fused output glue](#mamba-2-fused-output-glue) · [Fused LM-head cross entropy](#fused-lm-head-cross-entropy) · [Fused SwiGLU GEMMs](#fused-swiglu-gemms) · [Grouped GEMM for MoE](#grouped-gemm-for-moe)
- [Where we lose](#where-we-lose)

## Ops

```python
from jla import ops

# q, k: [B, T, H, K]; v: [B, T, H, V]; g: log-decay; beta: write strength
o, state = ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta,
                                output_final_state=True, segment_ids=segment_ids)
```

| kind | recurrence | gates |
|---|---|---|
| `simple_gla` | `S = exp(g) S + k^T v` (the Mamba-2 SSD core) | `g [B,T,H]` |
| `mamba3` | `S = exp(g) S + beta k_{t-1}^T v_{t-1} + gamma k^T v` (Mamba-3 SISO) | `g, beta, gamma` |
| `mamba3_mimo` | Mamba-3 with `R` input and output ranks per head: `q, k [B,T,H,R,K]`, `v [B,T,H,R,V]` | `g, beta, gamma` |
| `gated_deltanet` | `S' = exp(g) S;  S = S' + beta k^T (v - k S')` | `g, beta` |
| `deltanet` | `S = S + beta k^T (v - k S)` | `beta` |
| `kda` | `S' = diag(exp g) S;  S = S' + beta k^T (v - k S')` | `g [B,T,H,K], beta` |
| `gdn2` | `S' = diag(exp g) S;  S = S' + k^T (w v - (b k) S')` | `g, b [B,T,H,K], w [B,T,H,V]` |
| `rwkv7` | `S = diag(exp w) S + b^T (a S) + k^T v` (diagonal plus low rank) | `w, a, b [B,T,H,K]` |
| `gla` | gated linear attention | `g [B,T,H,K]` |
| `gated_deltaproduct` | `n_h` delta-rule steps per token after one decay `exp(g)`; `k, v, beta` are `[B, T*n_h, ...]` | `g, beta` |
| `mamba1` | Mamba-1 selective scan, per channel `h = exp(dt A) h + dt B u` (`q = C, k = B, v = u`) | `dt [B,T,D], a = A [D,N]` |
| `log_linear_simple_gla` | `simple_gla` on a Fenwick tree of O(log T) states, each level weighted | `g, level_scales [B,T,H,L]` |
| `rwkv6` | `o = q (S + diag(u) k^T v);  S = diag(exp g) S + k^T v` | `g [B,T,H,K], u [H,K]` |
| `mlstm` | `C = exp(g) C + exp(i) k^T v`, normalizer `n`; `o = q C / max(abs(q n), 1)`, max-stabilized | `g, i` |

`impl="reference" | "chunked" | "mosaic_gpu" | "auto"`. `segment_ids [B, T]` restarts the state
at every document boundary, with the same semantics in every implementation. A Mosaic GPU kernel
registers a strict `supports` check, and `"auto"` uses it only for the calls its GPU tests cover;
everything else runs the chunked op.

## Kernels

Mosaic GPU kernels for H100 (sm_90a) and JAX 0.11.2, forward and backward, exact algorithms (fp32 states and
accumulation; no approximate gates; `exp` and similar functions use the GPU's fast approximate instructions,
Mosaic GPU's `approx_math`). Ratios are kernel time against the strongest latest upstream code, the other
implementation's time over ours (> 1 means ours is faster), interleaved on one H100, median kernel time per
call, measured on 20 Oct 2026 with no other jobs on the GPUs (FLA main `9d478fa` changes none of the compared
kernels, so `6dff3e1` is the newest code for them). Cells marked provisional were measured while other jobs
could share the GPU. Every shape, version and method is on the linked page.

### Mamba family

| Architecture | What `impl="auto"` runs on H100 | Baseline (latest code) | Forward | Forward + backward |
|---|---|---|---|---|
| Mamba-2 (SSD) scan, [mamba2.md](../benchmarks/mamba2.md) | `simple_gla` kernels, `K, V` in {64, 128} | official Mamba-2 Triton scan (state-spaces/mamba `e9594ce`, `mamba_chunk_scan_combined`) | 1.31-2.60x | 1.20-1.96x |
| | | FLA main `6dff3e1` (Triton = its default dispatch here) | 1.49-3.18x | 1.58-2.20x |
| Mamba-2 whole layer (input projection, causal conv1d, scan, `D` skip, gated RMSNorm, output projection), [mamba2.md](../benchmarks/mamba2.md#the-whole-layer-against-the-official-fused-training-path) | the `Mamba2` mixer as the presets run it | the official `Mamba2` module's training path, `use_mem_eff_path=True` (`mamba_split_conv1d_scan_combined`, one fused autograd function; `e9594ce`), `d_state` 128, head dim 64, at `(B, T, d_model)` = (4, 4096, 2048), (16, 4096, 2048), (1, 16384, 2048) and (8, 2048, 768) | 1.12-1.34x | 1.17-1.41x |
| Mamba-3 SISO, [mamba3.md](../benchmarks/mamba3.md) | the fused layer core (biases, angles, rotation and recurrence on Mosaic GPU) | official Mamba-3 Triton (`mamba3_siso_combined`, `e9594ce`) | 1.45-2.60x | 1.58-2.32x |
| Mamba-3 MIMO, [mamba3.md](../benchmarks/mamba3.md#mimo) | dedicated `mamba3_mimo` kernels, ranks 1/2/4/8, chunk `64/R` | official TileLang MIMO kernels (`e9594ce`): the full call / its recurrence kernels only | 1.85-5.57x / 1.57-4.70x | 3.44-6.09x / 2.70-5.37x |
| Mamba-1 (Jamba, Falcon-Mamba), [mamba1.md](../benchmarks/mamba1.md) | selective-scan kernels, `d_state` 4, 8, 16 or 32 | official CUDA selective scan (state-spaces/mamba `e9594ce`) | 1.19-1.63x (`d_state` 16); 1.81x at `d_state` 32 (provisional) | 1.72-1.85x; 1.60x at `d_state` 32 (provisional) |

### Delta rule, RWKV-7, GLA and mLSTM

| Architecture | What `impl="auto"` runs on H100 | Baseline (latest code) | Forward | Forward + backward |
|---|---|---|---|---|
| Gated DeltaNet / DeltaNet, `K, V` in {64, 128}, [gdn.md](../benchmarks/gdn.md) | FlashQLA-style kernels with an exact sequence split | FlashQLA main `5538c35` (= FLA main's default dispatch) | 0.79-1.69x | 0.73-1.26x |
| | | FLA main `6dff3e1` Triton | 1.74-3.14x | 1.71-2.68x |
| Gated DeltaNet / DeltaNet, `K = 256` (`V` up to 512), [gdn.md](../benchmarks/gdn.md#head-dim-256) | K-blocked kernels (also `K = 128, V = 256`) | FLA main default dispatch / Triton, at `(4, 4096, 6, 256, 512)` and FLA's 1.3B layer `(16, 4096, 6, 256, 512)` | 2.37-3.05x / 2.37-3.05x | 1.17-1.24x / 1.23-1.29x |
| KDA, [kda.md](../benchmarks/kda.md) | vector-gate kernels | FLA main default dispatch / Triton (training) | 1.25-1.57x / 1.25-1.57x dense (provisional); 1.37-1.52x / 1.37-1.52x with 16 packed documents | 1.47-1.78x / 1.51-1.91x dense, 1.61-1.63x / 1.73-1.77x with 16 packed documents |
| | | FlashKDA main `7afb9f4` (inference contract, forward only) | 0.57-2.30x | - |
| Gated DeltaNet-2, [kda.md](../benchmarks/kda.md) | vector-gate kernels with the erase and write gates | no upstream run (clean-room); our plain-JAX `chunked` op | 6.3-8.1x | 6.7-10.8x |
| RWKV-7, [rwkv7.md](../benchmarks/rwkv7.md) | the layers' bounded mode (`lower_bound = -0.61`; bf16 `r, k, v`, `chunk_size = 32`, head dim 64): Mosaic GPU kernels for the per-chunk work forward (`rwkv7_prep`) and backward (`rwkv7_bwd`), two warpgroups per CTA, around the state passes | FLA main as `RWKV7Attention` calls it (safe gate, chunk 64): at `(B, T, H)` = (1, 4096, 12), (2, 4096, 32) and (8, 4096, 12) / at the 1.3B layer shape (16 × 4096, 32 heads) and with 16 packed documents (1 × 65,536 × 32 heads, 1 × 16,384 × 8) | 1.21-1.52x / 1.09-1.29x | 1.18-1.46x / 1.06-1.29x |
| GLA (and HGRN2's GLA core), [gla.md](../benchmarks/gla.md) | dedicated GLA kernels (FLA's `chunk_gla` stage split), `K, V` in {64, 128} | FLA main `6dff3e1` Triton (= its default dispatch: no other backend serves GLA) | 1.69-2.19x dense; 0.88-1.39x with 16 packed documents | 1.52-1.77x dense; 1.01-1.41x with 16 packed documents |
| mLSTM (xLSTM), [mlstm.md](../benchmarks/mlstm.md) | the float32 `chunked` op; `impl="mosaic_gpu"` opts in to the stabilised mLSTM on the simple-GLA kernels (less accurate on rows where `q n` nearly cancels) | NX-AI `mlstm_kernels` 2.0.6 TFLA Triton (`chunkwise--triton_xl_chunk`, its fastest configuration; its backward is approximate), the opt-in path | 1.38-1.42x | 1.03-1.06x |

### Attention

The ratios are our FA3-style kernel's; the second column says what `"auto"` actually runs.

| Architecture | What `impl="auto"` runs on H100 | Baseline (latest code) | Forward | Forward + backward |
|---|---|---|---|---|
| Softmax attention, causal, [attention.md](../benchmarks/attention.md) | cuDNN | FA3 main `965c921` / cuDNN 9.27 | 0.90-0.92x / 0.96-1.09x | 0.77-0.92x / 0.94-1.07x with the default two-kernel backward; with the opt-in fused backward 0.87-0.89x FA3 at head dim 128 and 0.87-1.00x FA3 at the 1.3B presets' 32 query heads of 64 over 8 KV heads |
| Softmax attention, packed documents | our FA3-style kernel | FA3 main with `cu_seqlens` / cuDNN per document / cuDNN with the dense mask | 0.80-1.04x / 0.66-0.94x (provisional) / 3.48-7.85x | 0.79-1.29x / 0.89-1.08x (provisional) / 3.48-7.61x; with the fused backward 0.82-0.94x FA3 at head dim 128 and 0.89-1.31x at the 1.3B presets' heads, 2.9-6.6x the dense mask (provisional) |
| Sliding window, 128 tokens | cuDNN | FA3 main / cuDNN 9.27 | 0.88-1.09x / 0.86-0.94x | 1.10-1.38x / 1.15-1.27x; with the fused backward 0.92-0.96x FA3 at head dim 128 and 1.21-1.34x at 32 heads of 64 |
| Sliding window, 1024-4096 tokens | cuDNN | FA3 main / cuDNN 9.27 | 0.83-1.04x / 0.96-1.16x | 0.81-0.96x / 0.93-1.12x (two-kernel backward); with the fused backward 0.81-0.87x FA3 at head dim 128 and 0.88-1.02x FA3 at 32 heads of 64 |
| MLA attention core (q/k 192 = 128 + 64 RoPE, v 128; DeepSeek-V2/V3, Kimi K2, Kimi Linear), [mla.md](../benchmarks/mla.md) | our FA3-style kernel on the unpadded heads (causal, packed documents, sliding windows) | cuDNN 9.27 on the unpadded heads (JAX's cuDNN SDPA in MLA mode; packed documents: its packed-sequence layout with values padded to 192, or each equal document as its own row) / FA3 main `965c921` built with head dim 192 and the 192/128 forward | 1.06-1.19x / 0.95-0.97x causal; 1.53-1.86x (packed layout), 0.91-1.05x (documents as rows) / 0.88-0.96x with packed documents | 1.01-1.05x / 1.01-1.06x causal (three shapes, FA3 at two); 1.71-1.83x, 1.03-1.27x / 1.08-1.20x with packed documents |
| DeepSeek Sparse Attention (DSA), [dsa.md](../benchmarks/dsa.md), provisional | `dsa_implementation="auto"`: indexer, exact radix-select top-k, selected-key attention and indexer-KL backward kernels (bf16, head dims 64/128; MLA's 576/512 run plain JAX) | our plain-JAX DSA (out of memory from 4K tokens) / DeepGEMM main `1fc9c66` `fp8_mqa_logits` (FP8) + `torch.topk`, indexer + top-k in kernel time, 16K-128K / FlashMLA `ba89a34` sparse prefill (MLA shapes, forward only; rate of selected-key work) | 3.7-6.0x (with `lax.top_k`) / 1.16-1.43x / 0.08-0.12x | 2.8-3.9x / - / - |

### Kernels around the mixers

All provisional (17-19 Oct 2026, other jobs could share the machine).

| Kernel | What runs by default on H100 | Baseline | Forward | Forward + backward |
|---|---|---|---|---|
| Short causal conv + SiLU, [short_conv.md](../benchmarks/short_conv.md) | the fused kernels (`conv_impl = "auto"`) | XLA / causal-conv1d 1.7.0 CUDA, at the 1.3B shapes | 1.5-1.8x / 0.76-1.27x | 2.4-2.9x / 1.2-1.7x |
| Short conv + q/k L2 norm, [short_conv.md](../benchmarks/short_conv.md) | the fused prologue (`qk_l2norm_in_conv = true`) | the faster of FLA's Triton conv and causal-conv1d, each with FLA's `l2norm` | 1.78x dense, 1.68x packed | 2.35x dense, 2.06x packed |
| LM-head cross entropy, [cross_entropy.md](../benchmarks/cross_entropy.md) | XLA (the kernels are opt-in: `train.ce_impl`) | the XLA op, `D` 2048, 128,256-token vocabulary | - | 1.57-1.67x with the gradient formed in the forward pass |
| Fused SwiGLU GEMMs, [fused_gemm.md](../benchmarks/fused_gemm.md) | XLA (opt-in: `model.mlp_kernel = "mosaic_gpu"`) | XLA (cuBLASLt GEMM + fusion), the 1.3B MLP block | 1.10x | 1.05x |
| Grouped GEMM for MoE, [ragged_dot.md](../benchmarks/ragged_dot.md) | `ragged_dot` with the `PERSISTENT` tiling (`MoEConfig.ragged_dot_impl = "auto"`) | `torch._grouped_mm` (CUTLASS, torch 2.13), six shapes of the shipped MoE presets | 0.90-1.04x over all GEMMs, forward and backward | 0.94-1.04x |

## Design notes

### Softmax attention routing

`AttentionConfig(implementation="auto")` picks our kernel on Hopper for packed documents (bf16/fp16, head dim
64 or 128, any length) and, because cuDNN's backward as shipped with JAX 0.11.2 returns wrong gradients there,
once one device's query shard has more than 2^31 elements (`implementation="cudnn"` then raises when
differentiated). Dense causal and windowed calls stay on cuDNN, and XLA runs everywhere else.

The kernel's forward is persistent: one CTA per SM loops over the query tiles in FA3's longest-first order
inside L2-sized sections of heads, with packed-document tiles sorted by their work
(`AttentionConfig.persistent_forward = false` runs one CTA per tile; every variant is bitwise equal). Its
consumer warpgroups take turns and release key/value slots on named barriers, with an overlapped
`S_j` / `P_{j-1} V_{j-1}` loop and 192-row tiles of three consumer warpgroups for head dim 64 without
`segment_ids`. The default backward is block-skipping (separate `dq` and `dk, dv` kernels that visit only the
blocks a tile can see, bitwise reproducible); `AttentionConfig.mosaic_gpu_backward = "fused"` selects one
backward kernel in FA3's design (5 matrix products per block instead of 7; `dq` is added up in fp32 by TMA
reductions, so it is not bitwise reproducible, and it needs one head dim for q, k and v, so MLA heads keep the
two-kernel backward). The two transformer 1.3B TOMLs use the fused backward, and the library default stays the
reproducible two-kernel one. The kernel takes GQA, head dims 64 and 128 (and MLA's 192/128, see
[Other ports](architectures.md#other-ports)), `segment_ids` in any order as long as each document is one run of
tokens, and GPT-OSS sinks with gradients. Under `jax.jit` each GPU runs it on its own batch rows and, under
tensor parallelism, its own heads (when every mesh axis divides `Hkv`); it also runs inside `jax.shard_map`.

### Mamba-3 MIMO baseline

The official call also runs the angle cumsum, the rotation, the Q/K biases and the value up-projection inside
its kernels; our op leaves those to XLA in the layer, and they are not in our time. The second ratio counts only
the official recurrence kernels, the closest like-for-like comparison.

### Packed documents and states

Every recurrent kernel takes `segment_ids` (exact masks, no dense `[T, T]` tensor) and initial and final
states, and the attention kernel reads `segment_ids` per tile; the Mamba-2, Gated DeltaNet and KDA rows above
include packed-document shapes.

### Long contexts

[long_context.md](../benchmarks/long_context.md): the kernels address memory only through TMA descriptors, so
they have no 32-bit offset limits. Single calls of 1,064,960 tokens (2,181,038,080 > 2^31 elements per tensor)
match chained halves and float32 windows. Calls are capped by CUDA's 65,535-block grid limit (about 4.19M
tokens for Mamba-2, Gated DeltaNet, DeltaNet, attention and the fused Mamba-3 core, 2,097,088 for KDA and
GDN-2 at head dim 128); `supports()` rejects longer calls, so chain them through `initial_state`. For long
contexts set `AttentionConfig(rope_exact_angles=True)`. The `K = 256` Gated DeltaNet kernels are not yet
tested past 2^31 elements, and they have the same grid limit (4,194,240 tokens per differentiated call),
which their `supports()` does not check yet, so chain longer sequences there too.

### Fused short convolution

`conv_impl = "auto"` in `LinearAttentionConfig` and `SSMConfig` (the default) runs the width-4 depthwise
causal conv + SiLU before the recurrence on two Mosaic GPU kernels adapted from FLAT, with `segment_ids`
resets and initial states, per shard under FSDP or data parallelism. At the 1.3B shapes its forward + backward
is 2.4-2.9x faster than XLA and 1.2-1.7x faster than the causal-conv1d CUDA kernel
([short_conv.md](../benchmarks/short_conv.md), provisional).

### Fused q/k prologue

`LinearAttentionConfig.qk_l2norm_in_conv = true` (the default; Gated DeltaNet, DeltaNet, GDN-2 and KDA at head
dims 64 and 128, bf16): the q and k conv kernels also apply SiLU and the per-head L2 norm, so the recurrent
kernels skip their own. At the 1.3B q/k shape it is 1.78x / 2.35x faster (forward / forward + backward) than
causal-conv1d plus FLA's `l2norm` ([short_conv.md](../benchmarks/short_conv.md), provisional).

### `qk_conv_barrier`, `grouped_key_grads`

`qk_conv_barrier` (`LinearAttentionConfig`) keeps XLA from fusing the short convolutions into the q/k L2-norm
reduction of the recomputed forward, and `grouped_key_grads` (`SSMConfig`) sums Mamba-2's `dB`, `dC` per group
inside the backward kernel ([zero_copy_layouts.md](../benchmarks/zero_copy_layouts.md), provisional). These
gains were each measured against the code without that switch, so they do not multiply; the end-to-end
results include all of them.

### Mamba-2 fused output glue

`SSMConfig.fused_glue` (on by default for training calls on GPUs): the `D` skip, gate and gated RMSNorm run as
Pallas Triton row kernels in one custom VJP with the out-projection. On `mamba2-1p3b` the glue drops from 198
to 126 ms per step, 1.039x end to end in bf16, and on several GPUs the row kernels run per shard of the
trainer's mesh (1.036x at FSDP 8 in bf16); under a mesh that splits the heads (tensor parallelism) the layer
keeps the unfused glue. With FP8, on one device, the conv and the glue kernels read the z|x rows of the
stacked FP8 in-projection in place instead of copying them out (`jla.layers.ssm.IN_PLACE_ZX`, 1.022x). These
gains are interleaved A/B pairs, provisional ([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#mamba-2)).
Decoding runs the unfused layer. The row kernels take any head count (masked rows, padded to the next power of
two of heads, so `d_model` 768 with 24 heads takes the fused glue too), and a call that is not differentiated
forms the SSD values `dt x` inside the SSD kernel (the same output bit for bit). The whole layer at
`(8, 2048, 768)` takes 0.435 ms forward and 1.613 ms forward + backward, 1.24x and 1.41x the official fused
training path ([mamba2.md](../benchmarks/mamba2.md#the-whole-layer-against-the-official-fused-training-path)).

### Fused LM-head cross entropy

Opt-in: `train.ce_impl = "mosaic_gpu"`, or `"auto"` to use the kernels where they apply; the default is
`"xla"` ([cross_entropy.md](../benchmarks/cross_entropy.md), provisional). The forward never writes logits, and
the backward recomputes them per vocabulary chunk (on one GPU): at 16,384 tokens and a 128,256-token
vocabulary its forward + backward is 1.20-1.57x faster than the XLA op at hidden sizes 2048-768.
`mamba2-1p3b.toml` sets `"auto"` with 16,384-token chunks.

With `train.ce_grad_in_forward = true` the logits kernel also stores `exp(logit - tile max)` in float16 and the
head's gradient is formed in the forward pass: three GEMMs instead of four (no logits recompute), at the bf16
accuracy floor, and per shard under a multi-GPU mesh. At 65,536 tokens, `D` = 2048 and a 128,256-token
vocabulary its forward + backward takes 166.8 ms, against 255.9 ms for the XLA op and 210.6 ms for the
recompute kernels. The 1.3B steps get 1.027-1.099x faster (`transformer-1p3b` in FP8: 55.3% -> 60.8% MFU), and
the loss matches the XLA op's over 600 steps. With every other change of the release in the code it is 1.105x
and 1.060x in bf16 on one H100 for `transformer-1p3b` and `gdn-hd128-1p3b` (1.088x and 1.070x in FP8), and
1.114x and 1.076x on 8 GPUs (FSDP 8, FP8), so their TOMLs turn it on (`ce_impl = "auto"`,
`ce_grad_in_forward = true`). `transformer-swa-1p3b`, `kda-1p3b` and `mamba2-1p3b` were faster on one GPU too
(1.028-1.119x over every pair) but keep it off: `kda-1p3b`'s one 8-GPU pair was 0.954x (1,876.7 against 1,790.3
ms per step; [multi_gpu.md](../benchmarks/multi_gpu.md#train-options-for-several-gpus)), and the other two have
no 8-GPU measurement. `gdn2-1p3b.toml` keeps it off too: the option measured 1.043x (bf16) and 1.052x (FP8),
but with the CE kernels the 20-step bf16 losses of `gdn2-1p3b` end about 0.02 below the XLA cross entropy's,
which is not understood. The MTP heads use it too: it makes `qwen3-next-1p3b` 1.091-1.118x faster (its TOML
sets it) and `kimi-linear-1p3b` 1.066x by median step (1.092x with one 16 × 4096 micro-step; its TOML keeps it
off). Under FSDP the per-shard cross entropy gathers the LM-head weight in bf16 and reduce-scatters its
float32 gradient (exact): 1.84 GB less traffic per GPU and step for `transformer-1p3b`. `train.ce_fp8 = true`
runs the head's two gradient GEMMs in FP8: 1.068-1.140x in all (63.1% MFU), but it costs 0.02 in loss after
600 steps, so it is lossy and not recommended.

### Fused SwiGLU GEMMs

Opt-in: `model.mlp_kernel = "mosaic_gpu"` (under a multi-device mesh per shard of the batch, the XLA path
under `tp`). The activation runs in the gate/up GEMM's epilogue and its derivative in the down projection's;
1.05x for the 1.3B MLP block forward + backward ([fused_gemm.md](../benchmarks/fused_gemm.md), provisional).

### Grouped GEMM for MoE

The H100 default (`MoEConfig.ragged_dot_impl = "auto"`, with `jax.lax.ragged_dot` on shapes the kernels do not
take), adapted from JAX's ragged-dot example, with Tokamax's aligned row schedule and one persistent TMA
pipeline per SM ([ragged_dot.md](../benchmarks/ragged_dot.md), [expert_parallel.md](../benchmarks/expert_parallel.md),
provisional). At the shipped MoE presets' shapes its grouped GEMMs are within 0.90-1.04x of `torch._grouped_mm`
(CUTLASS), and inside each preset's MoE layer the `PERSISTENT` tiling makes them 1.28-1.37x faster than the
`BLOCK_ALIGNED` one. MoE layers move tokens to and from expert order with gathers in both directions (no
scatter-add atomics in the backward pass; pipeline stages keep the scatter-add transposes), and under a
multi-device mesh each device dispatches its own tokens. Together the dropless MoE layer runs 1.40-2.07x faster
forward + backward than with the scatter-add permutation and `BLOCK_ALIGNED` tiles, and 1.29-1.39x faster than
our PyTorch port of the same layer under `torch.compile` (Megatron-Core and TransformerEngine MoE layers not
measured). On one H100 the `qwen3-next-300m` train step is 1.24x faster than with the scatter-add permutation
on the kernels (XLA's `ragged_dot` runs out of memory there at a 0.45 memory fraction), and the
`granite-4.0-h-125m` step 6.48x faster than on XLA's `ragged_dot`.

## Where we lose

Every case with its numbers is on the linked page, with the reason and what is next.

- **Gated DeltaNet against FlashQLA** ([gdn.md](../benchmarks/gdn.md#where-we-lose-and-limits)): long single
  sequences (one row of 16K-64K tokens: 0.71-0.87x forward and 0.69-0.78x forward + backward at 8 heads,
  0.68-0.79x and 0.68-0.73x at 16), packed documents at small `B * H` (one row of 8 heads, 16 documents:
  0.80-0.87x / 0.74-0.77x from 16K tokens, where the split runs 8 pieces, and 0.45-0.54x / 0.58-0.60x at 2K-8K
  tokens, where it runs 4) and the backward at large `B * H` (0.83x forward + backward at `(8, 2048, 32)`, where
  the forward is 1.07x). FlashQLA starts each sequence piece from zero a few chunks early (an approximation; our
  split is exact) and runs its backward as one fused kernel, while ours splits it into a reverse state pass and a
  per-chunk gradient kernel. At head dim 64, which FlashQLA does not support, packed documents in short rows
  lose to FLA's default dispatch: one row of 8 heads with 16 documents runs 0.84-0.88x forward and 0.83-0.93x
  forward + backward at 2K-8K tokens (1.43-1.86x from 16K tokens).
- **KDA inference against FlashKDA** at large `B * H` (0.57x at `(8, 2048, 32)`): FlashKDA keeps a bf16 state
  and approximate `exp`, and fuses the gate activation, cumsum, decays and 16-token inverses into one prepare
  kernel ([kda.md](../benchmarks/kda.md#where-we-lose-and-limits)).
- **Softmax attention against FA3** ([attention.md](../benchmarks/attention.md#where-we-lose-and-limits)): the
  forward is 0.90-0.92x FA3 on causal attention, 0.83-1.04x with 1024- and 4096-token windows and 0.80-1.04x
  with packed documents (0.81-0.82x when documents start inside key blocks, provisional), so the 0.95x target
  is not reached on causal attention. FA3's intra-warpgroup overlap, named-barrier turns and slot releases and
  its 192-row tiles for head dim 64 are in the kernel; measured, they do not close the gap. Not built, with
  measurements: TMA multicast of keys and values across a CTA cluster (at most 1.4-2.0% of the forward waits
  for keys and values, provisional; FA3 enables clusters only for non-causal, non-varlen calls) and FA3-style
  varlen tiles (JAX 0.11.2's TMA needs tile-aligned offsets). Forward + backward with the default two-kernel
  backward is 0.77-0.92x FA3 causal (0.73x on one row of 8 heads of 64 at 32K-64K tokens); with the fused
  backward 0.87-1.00x causal, 0.88-1.02x with windows and 0.89-1.31x with packed documents at the 1.3B
  presets' grouped heads of 64, and 0.87-0.89x, 0.81-0.87x and 0.82-0.94x at head dim 128. Kernel by kernel
  (provisional, `B = 4, T = 4096`) the fused backward at the presets' heads is 0.93x FA3's on causal attention
  (0.945x with the per-parity `dS^T` barriers; its main kernel takes 1.14x FA3's time, the passes around it are
  0.065 ms cheaper), the two main kernels tie with packed documents, and at head dim 128 our main kernel takes
  1.22-1.51x FA3's time; the float32 TMA reductions of `dq` are 11% of the head-dim-64 backward, and FA3's
  128-row tiles for head dim 64 take 8-18% longer in our `S^T` layout.
- **MLA heads against FA3** built for them ([mla.md](../benchmarks/mla.md#where-we-lose-and-limits)): forward
  0.95-0.97x causal and 0.88-0.96x with packed documents, for the same reasons, and 0.91-1.05x cuDNN's forward
  when equal documents are split into rows; the fused backward does not take MLA heads, and the ranged backward
  keeps the serial `dk, dv` body at their 192-wide keys, where the overlapped one is 14-16% slower.
- **Mamba-1 forward at the Falcon-Mamba-7B / Jamba width** (`B = 1, T = 4096, D = 8192`): 1.19x the official
  kernel against 1.63x at the mamba-130m shape (64 CTAs on 132 SMs; the forward has no sequence split yet;
  [mamba1.md](../benchmarks/mamba1.md#where-we-lose-and-limits)).
- **GLA forward with packed documents at few heads** (0.88x at one row of 16,384 tokens and 6 heads): our state
  pass walks the row in order, FLA runs every document as its own sequence
  ([gla.md](../benchmarks/gla.md#where-we-lose-and-limits)).
- **DSA** (provisional; [dsa.md](../benchmarks/dsa.md#where-we-lose-and-limits)): our indexer is bf16 where
  DeepGEMM's is FP8, and DeepGEMM's FP8 logits are 1.5-1.6x faster than it; our exact indexer + top-k leads
  DeepGEMM + `torch.topk` only through the radix top-k and only in kernel time: the radix path is bound by
  kernel launches, so by wall clock our 16K call takes 10.4 ms against 5.1 ms of kernel time (and DeepGEMM +
  `torch.topk`'s 7.16 ms of kernel time). The 128K training step does not fit in a 20 GB share, and the
  selected-key attention is masked-dense (0.48-0.55x the speed of dense causal cuDNN attention at 16K-32K, and
  0.08-0.12x the useful-work rate of FlashMLA's gathering sparse prefill); the gathering kernel is not built.
- **MoE** (provisional; [ragged_dot.md](../benchmarks/ragged_dot.md#where-we-lose-and-limits)): our grouped GEMMs
  trail `torch._grouped_mm` on the `qwen3-next-300m` down projection (0.91x forward) and the up projection's
  backward (0.94x); the FP8 grouped-GEMM forward kernel (1.32-1.47x the bf16 one) is not yet used by the MoE
  layer, and the expert-parallel all-to-all does not overlap the expert GEMMs. Expert parallelism is measured on
  2 GPUs only, where it is slower than per-device data parallelism for `qwen3-next-300m`; at 1.3B, on the MoE
  layer without the per-device dispatch, it ran `qwen3-next-1p3b` at 0.38x its one-GPU rate per GPU. The 1.3B
  hybrids stay below the 35% MFU target (`qwen3-next-1p3b` 26.7% with its TOML and 28.4% with no other jobs on
  the machine; `kimi-linear-1p3b` 22.7% with its TOML and 24.6% with one 16 × 4096 micro-step and the
  forward-gradient cross entropy), with the MoE layers at 29.2% and 30.5% of the step and about half of that in
  XLA gathers into and out of expert order
  ([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts)).
- **Hopper only, one JAX version**: the kernels need an H100/H200, and JAX is pinned to 0.11.2. The Mamba-2 row
  kernels, the Gated DeltaNet piece scan and the FP8 SwiGLU passes are Pallas Triton, which JAX 0.11.2
  deprecates; they move to Mosaic GPU before the pin does.
