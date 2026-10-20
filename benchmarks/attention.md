# Softmax attention: the FA3-style kernel against FA3 and cuDNN

`jla.ops.mosaic_gpu.attention.flash_attention`, reached through `jla.layers.attention.attention(...,
implementation="mosaic_gpu")`, is a FlashAttention-3-style Mosaic GPU kernel for causal softmax attention with
sliding windows, packed documents (`segment_ids`), GQA and GPT-OSS sinks. Its forward is persistent, and it has two
backwards: the default `backward="ranged"` (a `dq` kernel and a `dk, dv` kernel, bitwise reproducible) and the
opt-in `backward="fused"` (one kernel in FA3's design). It is compared against FlashAttention-3 main and cuDNN.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 `q, k, v`.
- FA3: Dao-AILab/flash-attention main `965c921` (`hopper/`; main `e7a6402` changes only `flash_attn/cute`), built
  as `flash_attn_3` 3.0.1 for torch 2.13.0+cu129 with the instantiations these shapes use (bf16, head dims 64 and
  128, sm90, causal and local, varlen, PackGQA, cluster, backward). Called as `flash_attn_func(..., causal=True,
  window_size=(W - 1, 0))`, and for packed documents `flash_attn_varlen_func` with `cu_seqlens`.
- cuDNN 9.27.0 as shipped with JAX 0.11.2, through `jla.layers.attention.attention(..., implementation="cudnn")`:
  native left window; packed documents as the dense `[B, 1, T, T]` mask the layer builds.
- `benchmarks/bench_attention_fa3.py`: all implementations interleaved per shape, two rounds of 30 calls after 3
  warmups (the order alternates), the faster round's median kernel time: CUPTI for ours and cuDNN, `torch.profiler`
  for FA3 (pre- and post-processing kernels included; a fresh FA3 worker per shape, cross-checked against a
  gap-free CUDA-event time per call). Forward, and forward + backward for a fixed output cotangent.
- 16K tokens per call (`B = 16384 / T`); `documents` = 4 equal documents per row; `swa<W>` = a window of `W` tokens
  including the query.
- Measured on 20 Oct 2026 with no other jobs on the GPUs, one job per H100.

## Results

Ratio = baseline time / ours (> 1: ours is faster). "Default" is the ranged backward, "fused" the opt-in one.
Ranges over `T` = 2K-32K at both head layouts: `D = 128` with 16 query and 16 KV heads, and `D = 64` with 32 query
heads over 8 KV heads (the 1.3B presets' attention):

| mask | forward, vs FA3 / vs cuDNN | fwd + bwd, default, vs FA3 / vs cuDNN | fwd + bwd, fused, vs FA3 (`D = 128` / `D = 64`) |
|---|---|---|---|
| causal | 0.90-0.92x / 0.96-1.09x | 0.77-0.92x / 0.94-1.07x | 0.87-0.89x / 0.87-1.00x |
| 4 packed documents per row | 0.80-1.04x / 3.48-7.85x | 0.79-1.29x / 3.48-7.61x | 0.82-0.94x / 0.89-1.31x |
| 128-token window | 0.88-1.09x / 0.86-0.94x | 1.10-1.38x / 1.15-1.27x | 0.92-0.96x / 1.21-1.34x |
| 1024- and 4096-token windows | 0.83-1.04x / 0.96-1.16x | 0.81-0.96x / 0.93-1.12x | 0.81-0.87x / 0.88-1.02x |

Every shape (`swa4096` only where `T > 4096`):

| T | B | mask | `D = 128`: fwd vs FA3 | fwd + bwd, default | fwd + bwd, fused | default vs cuDNN | `D = 64`, 32 / 8 heads: fwd vs FA3 | fwd + bwd, default | fwd + bwd, fused | default vs cuDNN |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 8 | causal | 0.90 | 0.89 | 0.89 | 1.01 | 0.92 | 0.92 | 1.00 | 1.07 |
| 4096 | 4 | causal | 0.91 | 0.87 | 0.89 | 1.00 | 0.91 | 0.85 | 0.94 | 1.03 |
| 8192 | 2 | causal | 0.92 | 0.83 | 0.87 | 0.97 | 0.90 | 0.80 | 0.90 | 1.00 |
| 16384 | 1 | causal | 0.92 | 0.82 | 0.87 | 0.94 | 0.91 | 0.77 | 0.87 | 0.96 |
| 32768 | 1 | causal | 0.92 | 0.84 | 0.89 | 0.94 | 0.92 | 0.79 | 0.89 | 1.00 |
| 2048 | 8 | documents | 0.80 | 1.00 | 0.94 | 3.48 | 1.04 | 1.29 | 1.31 | 3.81 |
| 4096 | 4 | documents | 0.89 | 0.97 | 0.90 | 5.41 | 1.01 | 1.10 | 1.17 | 4.67 |
| 8192 | 2 | documents | 0.93 | 0.91 | 0.85 | 6.80 | 0.93 | 0.91 | 0.98 | 5.74 |
| 16384 | 1 | documents | 0.95 | 0.85 | 0.82 | 7.27 | 0.92 | 0.81 | 0.94 | 6.20 |
| 32768 | 1 | documents | 0.91 | 0.81 | 0.82 | 7.61 | 0.87 | 0.79 | 0.89 | 6.75 |
| 2048 | 8 | swa128 | 0.88 | 1.10 | 0.92 | 1.15 | 1.06 | 1.25 | 1.21 | 1.27 |
| 4096 | 4 | swa128 | 0.90 | 1.11 | 0.93 | 1.16 | 1.09 | 1.26 | 1.21 | 1.27 |
| 8192 | 2 | swa128 | 0.90 | 1.11 | 0.93 | 1.16 | 1.07 | 1.25 | 1.21 | 1.27 |
| 16384 | 1 | swa128 | 0.92 | 1.12 | 0.94 | 1.16 | 1.05 | 1.25 | 1.21 | 1.27 |
| 32768 | 1 | swa128 | 0.95 | 1.16 | 0.96 | 1.20 | 1.06 | 1.38 | 1.34 | 1.27 |
| 2048 | 8 | swa1024 | 0.91 | 0.94 | 0.87 | 1.04 | 0.99 | 0.96 | 1.02 | 1.12 |
| 4096 | 4 | swa1024 | 0.86 | 0.90 | 0.85 | 1.01 | 1.04 | 0.93 | 1.00 | 1.10 |
| 8192 | 2 | swa1024 | 0.86 | 0.90 | 0.85 | 1.00 | 1.04 | 0.92 | 0.99 | 1.08 |
| 16384 | 1 | swa1024 | 0.90 | 0.88 | 0.84 | 0.98 | 1.03 | 0.92 | 0.96 | 1.08 |
| 32768 | 1 | swa1024 | 0.91 | 0.89 | 0.85 | 0.98 | 1.04 | 0.93 | 1.00 | 1.08 |
| 8192 | 2 | swa4096 | 0.91 | 0.87 | 0.86 | 0.99 | 0.96 | 0.81 | 0.89 | 1.02 |
| 16384 | 1 | swa4096 | 0.83 | 0.82 | 0.81 | 0.93 | 0.96 | 0.81 | 0.88 | 1.02 |
| 32768 | 1 | swa4096 | 0.85 | 0.82 | 0.82 | 0.93 | 0.95 | 0.82 | 0.89 | 1.02 |

At head dim 128 the fused backward's forward + backward with packed documents is 2.9-6.6x cuDNN with the dense mask
(19 Oct 2026, provisional).

Length sweep, causal, FA3 time / ours, at one row of 8 heads (`B = 1`, `H = Hkv = 8`) and at four rows of 16 heads
(`B = 4`, `H = Hkv = 16`). Over both layouts the default backward's forward + backward is 0.73-0.90x FA3 (the
README's sweep chart); its worst shapes are one row of 8 heads of 64 at 32K-64K tokens (0.73x):

| D | T | 1 x 8 heads: fwd | fwd + bwd, default | fwd + bwd, fused | 4 x 16 heads: fwd + bwd, default |
|---|---|---|---|---|---|
| 64 | 2048 | 0.92 | 0.79 | 0.90 | 0.82 |
| 64 | 4096 | 0.90 | 0.80 | 0.87 | 0.78 |
| 64 | 8192 | 0.92 | 0.77 | 0.86 | 0.76 |
| 64 | 16384 | 0.91 | 0.75 | 0.85 | 0.76 |
| 64 | 32768 | 0.89 | 0.73 | 0.85 | 0.74 |
| 64 | 65536 | 0.89 | 0.73 | 0.86 | 0.75 |
| 128 | 2048 | 0.91 | 0.85 | 0.86 | 0.90 |
| 128 | 4096 | 0.91 | 0.85 | 0.87 | 0.86 |
| 128 | 8192 | 0.92 | 0.83 | 0.87 | 0.83 |
| 128 | 16384 | 0.91 | 0.82 | 0.87 | 0.84 |
| 128 | 32768 | 0.91 | 0.81 | 0.88 | 0.82 |
| 128 | 65536 | 0.91 | 0.80 | 0.92 | 0.80 |

### Fused backward end to end

The two transformer 1.3B TOMLs set `[model.attention] mosaic_gpu_backward = "fused"`; the library default stays
`"ranged"`. `bench_e2e_ab.py --arch <preset> --micro-batch 4 --b-set model.attention.mosaic_gpu_backward=fused
--rounds 2` (A = default, B = fused, A1 B1 A2 B2 in fresh processes on one GPU; the FP8 pairs add
`model.fp8="delayed"`, `fp8_min_features=512` and `fp8_fused=true` to both sides), 18 Oct 2026, provisional:

| preset | precision | B / A, median | MFU, A -> B |
|---|---|---|---|
| `transformer-1p3b` | bf16 | 1.014x and 1.029x (two pairs) | 49.1% -> 49.8% and 47.7% -> 49.1% |
| `transformer-swa-1p3b` | bf16 | 1.042x | 46.2% -> 48.2% |
| both presets, four pairs | FP8 | 0.977-1.041x, within the run-to-run spread | |

The forward's FA3-style schedule and the persistent forward each moved these presets end to end by under 1% (19 Oct
2026, provisional): with no recompute the attention forward runs 96 times per step (24 layers x 4 micro-batches) at
0.25-0.4 ms each.

## `implementation="auto"`

`AttentionConfig(implementation="auto")` picks this kernel on Hopper for packed documents (bf16/fp16, head dim 64 or
128, any length), where the layer's alternative is cuDNN with the dense mask, and once one device's query shard has
more than 2^31 elements, where cuDNN 9.27's backward returns wrong gradients ([long_context.md](long_context.md)).
Dense causal and windowed calls stay on cuDNN, which this kernel does not beat consistently, and XLA runs everywhere
else. `AttentionConfig.mosaic_gpu_backward = "fused"` selects the fused backward, and
`AttentionConfig.persistent_forward = false` one CTA per query tile.

## Where we lose, and limits

- **Causal attention against FA3**: the forward is 0.90-0.92x FA3, short of the 0.95x target; forward + backward
  is 0.77-0.92x with the default backward (0.73x on one row of 8 heads of 64 at 32K-64K tokens) and 0.87-1.00x with
  the fused one. The default backward is two kernels, so `S = QK^T` and `dP = dO V^T` are computed twice: 7 matrix
  products per block against FA3's 5. With head dim 128 and windows or packed documents it is the faster of the two.
- **The fused backward's main loop** (kernel by kernel, `B = 4, T = 4096`, 19 Oct 2026, provisional): at the 1.3B
  presets' 32 query heads of 64 over 8 KV heads the whole backward is 0.93x FA3's on causal attention (0.945x in an
  interleaved run with the per-parity `dS^T` barriers): the main kernel takes 1.14x FA3's time, and the passes
  around it are 0.065 ms cheaper than FA3's preprocess, conversion and GQA accumulation kernels. With packed
  documents the two main kernels tie; at head dim 128 ours takes 1.22-1.51x FA3's time. The float32 TMA reductions
  of `dq` are 11% of the head-dim-64 backward (2.1 GB of `reduction_op="add"` traffic per call), and FA3's 128-row
  query tiles for head dim 64 take 8-18% longer in this kernel's `S^T` layout, where the per-query LSE and `D`
  vectors need 32 registers each; FA3's layout (`S` with queries as rows, `P` and `dS` staged in shared memory)
  needs a new main loop.
- **The fused backward against the default one** (18 Oct 2026, provisional; the tables above give both against
  FA3): at head dim 128 it is 1.01-1.07x on causal attention from 4K tokens but 0.73-0.92x with windows and
  0.81-0.95x with packed documents; at the presets' heads it is 1.12-1.20x on causal attention. Its `dq` is added
  in fp32 by TMA reductions in an order that depends on CTA timing, so it is not bitwise reproducible (as in FA3),
  and it needs one head dim for `q`, `k` and `v`: MLA heads take the default backward ([mla.md](mla.md)).
- **The forward's issue rate**: Mosaic GPU 0.11.2 always issues `wgmma` with scale-d = 1, so every `QK^T`
  accumulator is first zero-filled (64 `mov.b32` per step per thread at `D = 128`; FA3 issues the first k-slice
  with scale-d = 0), and the row sums of `P` are reduced across the quad every step, where FA3 reduces once per
  tile; a non-accumulating first product needs an option in `plgpu.wgmma` (a JAX change). FA3's intra-warpgroup
  overlap, named-barrier turns, 192-row tiles for head dim 64 and longest-first persistent schedule are in the
  kernel; measured, they do not close the gap.
- **No TMA multicast of keys and values across a CTA cluster**: at most 1.4-2.0% of the forward waits for key/value
  data (19 Oct 2026, provisional), and FA3 enables clusters only for non-causal, non-local, non-varlen calls.
- **Packed documents**: the tiles are dense, so a document that starts inside a key block costs each 64-row
  warpgroup up to one key block more than FA3's document-aligned varlen tiles. With 5 documents per row (`D = 64`,
  `T` = 4K-8K) the forward is 0.81-0.82x FA3, against 0.86-0.93x with 4 block-aligned documents at the same lengths
  (19 Oct 2026, provisional); varlen tiles need unaligned TMA offsets and a per-row masked store, which JAX 0.11.2's Mosaic
  GPU lacks. Against cuDNN on each document as its own row (the same work without a mask, possible only for equal
  documents) the forward is 0.66-0.94x and forward + backward 0.89-1.08x (17 Oct 2026, provisional). A document id
  that returns after another document in the same row (which `cu_seqlens` cannot express) turns block skipping off
  for that row: at `T = 8192` 1.172 / 4.909 ms forward / forward + backward against 0.300 / 1.208 ms with one run
  per id (17 Oct 2026, provisional).
- **Sharding**: each GPU runs the kernel on its own batch rows, and on its own heads when every mesh axis divides
  `Hkv`; with MQA or an `Hkv` that a mesh axis does not divide (for example `Hkv = 2` on `tp = 4`) every TP rank
  computes every head, and a sharded sequence axis is gathered whole. The linear-attention Mosaic GPU kernels run
  per shard only under a mesh context (`jax.set_mesh`, as the trainer sets it); under a plain `jax.jit` without one
  they gather (Gated DeltaNet with the batch over two GPUs: 6 all-gathers).
- **Tokamax** main `b01b0d2a` (0.0.16) was measured only on 17 Oct 2026 (provisional), on the kernel without its
  persistent forward: our forward was 0.94-0.98x Tokamax's and causal forward + backward 0.97-1.06x; Tokamax's
  window backward visits every block (3.4-55x slower than ours).

## Correctness

`benchmarks/attention_fa3_diff.py` sends identical bf16 inputs through this kernel, FA3, cuDNN and an fp32 dense
reference: over 12 cases (`T` up to 8192, odd `T`, GQA up to 8:1, `D` 64 and 128, windows of 1 to 1000 tokens,
packed documents, logits scaled 8x) the worst relative errors equal FA3's (out 3.3e-3, `dq` 6.8e-3, `dk` 6.0e-3,
`dv` 6.5e-3), and sinks match GPT-OSS's formula. `sliding_window=W` is FA3's `window_size=(W - 1, 0)` with
`causal=True`, and `segment_ids` with one run per document is FA3's `cu_seqlens`. Every forward variant (persistent
or one CTA per tile, either loop, two or three warpgroups) gives bitwise equal outputs, the default backward is
bitwise reproducible, and the fused backward's `dk, dv` equal the default's bit for bit with `dq` equal up to its
fp32 addition order (across processes with `--xla_gpu_autotune_level=0`). `tests/test_mosaic_gpu_attention.py`,
`tests/test_mosaic_gpu_attention_persistent.py`, `tests/test_fp8_attention.py` and `tests/test_layers_attention.py`
cover these, the fp32 dense references and the sharded calls on two GPUs; they also bound each row's error of `out`
(< 2e-2) and `dv` (< 4e-2), while rows of `dq` and `dk` cancel to near zero, where FA3, cuDNN and this kernel all
reach 1.2-1.5.

## Reproduce

```
# One H100 per command; <fa3-venv> has flash_attn_3 built from FA3 main 965c921
python benchmarks/bench_attention_fa3.py --impls mosaic_gpu,cudnn,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_d128.json
python benchmarks/bench_attention_fa3.py --impls mosaic_gpu:fused,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_d128_fused.json
python benchmarks/bench_attention_fa3.py --heads 32 --kv-heads 8 --head-dim 64 --impls mosaic_gpu,cudnn,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_d64.json
python benchmarks/bench_attention_fa3.py --heads 32 --kv-heads 8 --head-dim 64 --impls mosaic_gpu:fused,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_d64_fused.json
# Length sweep, one row of 8 heads (and again with --head-dim 128)
python benchmarks/bench_attention_fa3.py --heads 8 --kv-heads 8 --head-dim 64 --tokens 2048 \
    --lengths 2048,4096,8192,16384,32768,65536 --masks causal --impls mosaic_gpu,mosaic_gpu:fused,fa3 \
    --fa3-python <fa3-venv>/bin/python --fa3-commit 965c9212e6be --out results/attention_sweep_d64.json
# four rows of 16 heads: one command per length T, with --heads 16 --kv-heads 16 --tokens <4 T> --lengths <T>
# Errors against FA3, cuDNN and fp32
python benchmarks/attention_fa3_diff.py --fa3-python <fa3-venv>/bin/python --out results/attention_diff.json
```

## How it works

The forward is persistent: one CTA per SM loops over (batch, head, query tile) items in FA3's longest-first order
inside L2-sized sections of heads, with packed-document tiles sorted by their real key-block count and each tile's
epilogue overlapped with the next tile's loads; over 20 shapes it is 1.00-1.26x faster than one CTA per tile, with
bitwise equal outputs (19 Oct 2026, provisional). Each tile runs two ping-pong consumer warpgroups of 64 rows, or,
for head dim 64 without `segment_ids`, FA3's intra-warpgroup overlapped loop on 192-row tiles of three warpgroups,
where that measured faster. Every tile visits only the key blocks its window, document and causal diagonal allow,
and masks only the blocks that cross an edge. The default backward's `dq` and `dk, dv` kernels (GQA `dk, dv` summed
in the kernel) start their CTAs longest first inside L2-sized sections of (batch, head) pairs, FA3's
`SingleTileBwdLPTScheduler` order; the fused backward forms all three gradients in one kernel. The FP8
attention-gradient epilogue (`attention_fp8_bwd.py`, `[model.attention] fp8_grad_epilogue`) runs variants of these
backward kernels with the same CTA order and masks, 64-key `dq` blocks and the serial `dk, dv` body;
[fp8_1p3b.md](fp8_1p3b.md) measures it.

## Context parallelism

`jla.distributed.context_parallel_attention` shards `T` of causal softmax attention over the `cp` axis of the
context mesh: ring attention (blockwise float32 online softmax over `ppermute`, zigzag load balancing) or Ulysses
(`all_to_all` from sequence to head shards around any local attention), with packed documents, windows, sinks and
GQA, forward and backward. It is a library API; the trainer has no `cp` mesh axis.

`benchmarks/bench_context_parallel_attention.py` on 2 H100s, bf16, `B = 1, H = 16, Hkv = 4, D = 128`, three
packed-document boundaries (one on the shard boundary), forward + backward, median wall clock of 10 jitted calls in
ms (17 Oct 2026, provisional: other jobs' tests may have shared the GPUs); the baselines run the whole sequence on
one GPU:

| T | mask | 1 GPU, cuDNN | 1 GPU, this kernel | ring, zigzag | ring, contiguous | Ulysses + cuDNN | Ulysses + this kernel |
|---|---|---|---|---|---|---|---|
| 8192 | causal + documents | 4.52 | 4.85 | 11.01 | 13.20 | **2.99** | 3.05 |
| 8192 | + window 1024 | 1.67 | 1.16 | 8.72 | - | 1.42 | **1.06** |
| 32768 | causal + documents | 57.6 | 69.8 | 122.7 | 157.2 | **31.1** | 37.3 |
| 32768 | + window 1024 | 7.03 | 3.91 | 29.3 | - | 5.24 | **3.07** |

- Ulysses runs a fused kernel locally on `H / P` heads plus two `all_to_all` per tensor (1.85x faster than one GPU
  at 32K and 1.5x at 8K with cuDNN inside) and needs `H % P == 0`. For long packed sequences use
  `implementation="mosaic_gpu"` inside it: cuDNN materializes the `[B, 1, T, T]` document mask (16 GB per device
  at 128K).
- The ring scores blocks with XLA einsums, not a fused kernel: about 2.1x slower than one cuDNN GPU at 32K, and
  7-10x slower with a window than a fused windowed kernel or Ulysses. It keeps `T / P` queries and two key/value
  blocks per device and takes any `H`; zigzag balancing makes the causal ring 1.2-1.3x faster than the contiguous
  one, and windowed rings run contiguous, truncated to the blocks a window reaches.
- `tests/test_context_parallel_attention.py` (8 CPU devices) and `tests/_context_parallel_attention_checks.py` (8
  H100s) compare ring and Ulysses on `cp = 8` and `dp x cp` meshes, and one SGD step of `tiny-hybrid`, with
  single-device attention: float32 errors at most 1.1e-6 (ops) and 6.2e-6 (train step), bf16 ring 6.3e-3. On the
  GPUs every bf16 variant's output sits at the bf16 rounding floor of the reference (2.2e-3 against 2.0e-3).

```
CUDA_VISIBLE_DEVICES=<two GPUs> python benchmarks/bench_context_parallel_attention.py --seq 8192 32768 --window 1024 \
    --check-seq 4096
```
