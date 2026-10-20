# DeepSeek Sparse Attention (DSA) kernels

`src/jla/ops/mosaic_gpu/dsa.py` runs the DSA layer (`"dsa"` layer type: DeepSeek-V3.2's lightning indexer plus top-k
sparse attention) on Hopper: the indexer scores, an exact radix-select top-k (`src/jla/ops/mosaic_gpu/dsa_topk.py`),
softmax attention over the selected keys (forward and backward) and the gradient of DeepSeek-V3.2's indexer KL loss
on two backward kernels. It is compared against the plain-JAX path, against DeepSeek's production inference kernels
(DeepGEMM's FP8 indexer logits with `torch.topk`, FlashMLA's sparse prefill) and against dense causal cuDNN
attention. `SparseAttentionConfig.dsa_implementation = "auto"` (the default) runs the kernels on Hopper for bf16/fp16
with attention head dim 64 or 128 (`value_dim == head_dim`) and index head dim 64 or 128; other shapes, such as MLA's
576/512 head dims, run the plain-JAX ops of `jla.ops.sparse_attention` (dense `[B, T, T]` scores).

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16. DeepSeek-V3.2-like shapes per GPU: `B = 1`, 16 query heads sharing one
  key/value head of 128 (MQA, as DeepSeek-V3.2 runs MLA for DSA under 8-way tensor parallelism), an indexer of 64
  heads of 128, `top_k = 2048`, the indexer KL with coefficient 1. Forward + backward = the gradients of `q, k, v`
  and of the indexer's `q, k, w`.
- DeepGEMM main `1fc9c66` (14 Oct 2026, version 2.8.2), `fp8_fp4_mqa_logits`: the indexer logits in FP8 e4m3 (one
  float scale per key) with causal per-query key ranges, then `-inf` outside the range and `torch.topk` per chunk of
  8192 queries, as DeepSeek-V3.2's inference code selects; the FP8 casts of `q` and `k` are not timed.
- FlashMLA `ba89a34` (15 Sep 2026), `flash_mla_sparse_fwd`: the sparse MLA prefill in DeepSeek-V3.2's absorbed MQA
  form (128 query heads, one latent key/value head, `d_qk = 576`, `d_v = 512`, 2048 indices per query), forward
  only; the last commit that builds with CUDA 12.9 (main `0f11e17` needs NVCC 13.1). Our kernels do not take MLA's
  head dims, so attention is compared through the rate of selected-key work, `2 * (selected query-key pairs) * heads
  * (d_qk + d_v)` FLOP/s.
- DeepSeek-V3.2-Exp `87e509a`: `inference/kernel.py`'s TileLang `fp8_index` (TileLang 0.1.15) and
  `inference/model.py` for the correctness check; torch 2.13.0+cu129, CUDA 12.9.
- Ours: median CUPTI kernel time. DeepGEMM and FlashMLA: summed CUDA kernel time per call from `torch.profiler`
  (CUDA-event times agree within 8%); DeepSeek's TileLang indexer: median CUDA-event time. Our rows of a length are
  interleaved in one run on one H100; DeepGEMM and FlashMLA ran right after them on the same GPU.
- Measured on 18-19 Oct 2026 under a per-GPU benchmark lock while other jobs' test runs could share the GPU, so every
  number on this page is provisional.

## Provisional results

The layer (`dsa_layer`), ms, forward / forward + backward. The radix top-k is the default; `topk_method="sort"`
selects with `lax.top_k` on causal prefixes:

| T | radix top-k | `lax.top_k` | plain-JAX path (`"xla"`) | kernels vs plain JAX |
|---|---|---|---|---|
| 1024 | | 0.19 / 0.54 | 0.70 / 1.49 | 3.7x / 2.8x |
| 2048 | | 0.44 / 1.41 | 2.63 / 5.43 | 6.0x / 3.9x |
| 4096 | | 1.30 / 4.14 | out of memory | |
| 8192 | | 5.05 / 14.9 | out of memory | |
| 16384 | 12.88 / 36.76 | 18.53 / 47.78 | out of memory | |
| 32768 | | 68.2 / 175.2 | out of memory | |
| 65536 | 174.1 / 535.1 | 259.5 / out of memory | | |
| 131072 | 444.0 / out of memory | | | |

1K-32K with `lax.top_k` and the plain path: 18 Oct; the radix rows and the 16K `lax.top_k` pair: 19 Oct. "Out of
memory" is within a 20 GB share of the GPU. At 16K the radix forward is 1.44-1.46x `lax.top_k`'s in kernel time
(three runs) and 1.43-1.45x by wall clock (13.5-13.8 against 19.6-19.7 ms), and the training step 1.30-1.31x.

Forward against DeepSeek's kernels, ms (19 Oct; the TileLang and attention columns 18 Oct):

| T | ours: bf16 indexer + radix top-k (`index_topk`) | ours: bf16 indexer alone | DeepGEMM FP8 logits + `torch.topk` (logits alone) | ours vs DeepGEMM + `torch.topk` | DeepSeek's TileLang `fp8_index` + `torch.topk` (without the top-k) | FlashMLA sparse prefill, MLA (TFLOP/s of selected-key work) | ours: selected-key attention (TFLOP/s of selected-key work) |
|---|---|---|---|---|---|---|---|
| 16384 | 5.00 | 3.66 | 7.16 (2.24) | 1.43x | 8.70 (5.26) | 14.8 (592) | 3.60 (72) |
| 32768 | 18.4 | 14.0 | 23.8 (9.37) | 1.30x | 29.8 (19.4) | 32.4 (558) | 12.0 (44) |
| 65536 | 71.2 | 59.5 | 85.4 (37.4) | 1.20x | 105.6 (74.1) | 66.8 (551) | |
| 131072 | 278.3 | | 323.2 (152.2) | 1.16x | 414.5 | 123.3 (601) | |

DeepSeek's TileLang `fp8_index` scores every key of a 4096-query chunk (the whole square); DeepGEMM skips the
causal future.

Our indexer + top-k is exact (bf16 indexer, float32 accumulation); DeepGEMM's indexer is FP8. The approximate FP8
indexer option (`SparseAttentionConfig(dsa_index_fp8=True)`, off by default) takes 4.18 / 14.8 ms at 16K / 32K
(1.71x / 1.61x DeepGEMM + `torch.topk`) and keeps 97.8% / 97.3% of the exact top-2048 keys on average per query.

## Where we lose, and limits

- **The indexer**: DeepGEMM's FP8 logits are 1.5-1.6x faster than our bf16 indexer (ours runs at 0.61-0.67x of
  them), so the lead of our indexer + top-k comes entirely from the radix top-k (3.3-4.1x faster than `torch.topk` as
  DeepSeek-V3.2's inference code calls it) and shrinks with `T`. `torch.topk` is DeepSeek's reference selection, not
  the fastest top-k there is: the radix top-k kernels of serving engines and TileLang were not measured.
- **Wall clock**: the radix path launches a few hundred small kernels and `lax.cond` host round trips per call, so by
  wall clock `index_topk` takes 10.4 ms at 16K against 5.1 ms of kernel time, above DeepGEMM + `torch.topk`'s 7.16 ms
  of kernel time (their wall clock was not measured). Its kernel time also has two modes (8.1 and 12.95 ms for the
  same compiled 16K layer forward); the wall clock matches the slower one, which the layer table uses.
- **Selected-key attention is masked-dense** (`O(T^2)`): 0.48-0.55x the speed of dense causal cuDNN attention at
  16K-32K, and 0.08-0.12x the selected-key-work rate of FlashMLA's gathering sparse prefill (550-600 TFLOP/s). A
  gathering kernel (FlashMLA's design) is not built yet; on Hopper it needs one row copy per selected key, since
  JAX 0.11.2's gathered TMA copy lowers to the Blackwell-only `tile::gather4`.
  Its plain-JAX formulation (`jla.ops.dsa_gather.gather_attention`, tested, not wired into the layer) runs at 14-17
  TFLOP/s at every length: 5x slower than the masked-dense kernels at 16K, 1.2-1.7x slower at 64K and 1.1-1.6x
  faster at 128K (19 Oct).
- **Memory**: the 128K training step does not fit in a 20 GB share (a 12.2 GB allocation on top of the rest); its
  forward runs. The radix select's candidate buffer grows with the block width (`max(2048, W / 16)` slots per row,
  256 MB per 4096-row block at 128K). At DeepSeek-V3.2's real MLA head dims the plain path's float32 `[B, H, T, T]`
  scores and probabilities take 8.6 GB each at 4K tokens with 128 heads.
- **Compile time** grows with `T`: each 4096-row query block has its own kernels, so a 64K layer compiles in minutes
  (a 2.4-minute forward compile).
- Packed documents compute the cross-document tiles (the masks keep them exact); there is no decode cache. Neither
  FlashMLA nor DeepGEMM has a DSA backward, so the training side has no DeepSeek baseline.

## Correctness

`benchmarks/dsa_reference_torch.py` runs DeepSeek-V3.2-Exp's inference math (`inference/model.py`, `87e509a`,
without FP8 quantization and the Hadamard rotation, differentiated by torch autograd with the sparse-stage indexer
KL) on the bf16 inputs of `benchmarks/dsa_reference_jax.py`: over four cases (GQA and MQA, head dims 64 and 128,
packed documents, top-k 64-1024, several query blocks) DeepSeek's top-k selects exactly our keys in every row, and
with identical selections the kernel path's largest relative errors are bf16 rounding (2.6e-3 out, 4.8e-3 `dq` and
`dk`, 4.1e-3 `dv`, 4.2e-3 / 3.3e-3 / 2.3e-4 for the indexer's `dq, dk, dw`). `tests/test_mosaic_gpu_dsa.py` checks
every kernel and the layer against the float32 plain path (one and two GPUs, scan over layers with remat),
`tests/test_mosaic_gpu_dsa_topk.py` checks `radix_top_k` against an exact top-k (ties to the lower column, as
`torch.topk` and `lax.top_k`; `+inf` and NaN never selected), and `tests/test_dsa_gather.py` the gathering op.
[`dsa_torch_topk_ties.py`](dsa_torch_topk_ties.py) checks which keys `torch.topk` keeps among tied scores on CUDA.

## Reproduce

```
python benchmarks/bench_dsa_long.py --lengths 1024,2048,4096,8192 --out results/dsa_long_1k_8k.json
python benchmarks/bench_dsa_long.py --out results/dsa_long.json
python benchmarks/bench_dsa_topk.py --out results/dsa_topk.json
python benchmarks/bench_dsa_gather.py --out results/dsa_gather.json
python benchmarks/dsa_layer_wallclock.py results/dsa_wallclock.json 16384 layer,index_topk
# DeepSeek's kernels, in a torch environment with DeepGEMM 1fc9c66 and FlashMLA ba89a34 built from source
python benchmarks/bench_dsa_production.py --out results/dsa_production.json
python benchmarks/bench_dsa_deepseek.py --deepseek-inference <DeepSeek-V3.2-Exp>/inference --out results/dsa_deepseek.json
# Against DeepSeek-V3.2's reference math (the first case; --length, --batch, --kv-heads, --head-dim, --top-k,
# --block-rows and --packed give the others)
python benchmarks/dsa_reference_jax.py --out results/dsa_ref.npz
python benchmarks/dsa_reference_torch.py --npz results/dsa_ref.npz --out results/dsa_ref.json
```

## How it works

`dsa_layer` runs per query block of 4096 rows, so memory grows with `T` per block instead of `T^2`. The indexer runs
one CTA per 128 x 128 (query, key) tile on or below the diagonal, one bf16 WGMMA per index head with ReLU and the
head weights applied in registers. The top-k is a most-significant-digit radix select on the order-preserving
32-bit key of each float32 score, 4 bits per pass in 8 passes, compacting the candidates after 3 passes, with no
sort. The selected indices become a bf16 0/1 mask applied tile by tile inside FA3-style attention kernels, and the
indexer KL's target and gradient run on the indexer kernel's tiles. Under a mesh the kernels run per batch shard.
