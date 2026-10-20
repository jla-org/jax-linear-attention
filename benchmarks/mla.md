# MLA attention: the FA3-style kernel on 192 / 128 heads

Multi-head Latent Attention (DeepSeek-V2/V3, Kimi K2 and the attention layers of Kimi Linear) attends with queries
and keys of 192 channels (128 without RoPE + 64 with RoPE) and values of 128. The MLA layer (`jla.layers.mla`) runs
its training path on the FA3-style Mosaic GPU attention kernel ([attention.md](attention.md)) on these unpadded
heads, with causal masks, packed documents and sliding windows, and is compared here against cuDNN 9.27 in its MLA
mode and against FlashAttention-3 main built for these heads.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16; `Dqk` 192, `Dv` 128; `H` query heads with `H` key/value heads (MLA has no GQA),
  `H` in {16, 32}, `T` in {2048, 4096, 8192}, 16,384 tokens per call (`B = 16384 / T`); `documents` = 4 equal
  documents per row.
- cuDNN 9.27 as shipped with JAX 0.11.2: its SDPA in MLA mode on the unpadded heads (packed documents as one batch
  row per document, exact here because the documents have equal lengths), and its packed-sequence layout with the
  values zero-padded to 192 (at 192 / 128 that layout returns wrong values in JAX 0.11.2, which scales the value
  offsets by the q/k head dim).
- FA3: Dao-AILab/flash-attention main `965c921` (`hopper/`) built with head dim 192 and the 192/128 forward by
  `benchmarks/build_fa3_mla.sh` (FA3's default build compiles head dim 192 out); documents through
  `flash_attn_varlen_func`. FA3's backward for `Dv` 128 runs its 192-wide kernel with the values padded.
- `benchmarks/bench_mla_attention.py`: three interleaved rounds per shape on one GPU (the order alternates), the
  fastest round's median of 30 CUDA kernel times; forward, and forward + backward for a fixed cotangent.
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

Ratio = baseline time / ours (> 1: ours is faster); ranges over the measured shapes (`H` 16 and 32, `T` 2K-8K),
except causal forward + backward, which covers the three shapes of the next table (FA3 at two):

| mask | forward, vs cuDNN | forward, vs FA3 | fwd + bwd, vs cuDNN | fwd + bwd, vs FA3 |
|---|---|---|---|---|
| causal | 1.06-1.19x | 0.95-0.97x | 1.01-1.05x | 1.01-1.06x |
| packed documents | 1.53-1.86x (packed layout), 0.91-1.05x (documents as rows) | 0.88-0.96x | 1.71-1.83x (packed layout), 1.03-1.27x (documents as rows) | 1.08-1.20x |

Causal forward + backward with the default backward (the serial `dk, dv` body), at the three shapes where that
backward was timed on its own: the forward of the main run plus that backward, against the same run's cuDNN and FA3
(FA3's backward at `(4, 4096, 32)` has no kernel time), in ms:

| `(B, T, H)` | forward | backward | forward + backward | cuDNN | FA3 | cuDNN / ours | FA3 / ours |
|---|---|---|---|---|---|---|---|
| (8, 2048, 16) | 0.301 | 1.143 | 1.444 | 1.518 | 1.537 | 1.05x | 1.06x |
| (4, 4096, 32) | 1.024 | 3.916 | 4.939 | 5.009 | - | 1.01x | - |
| (2, 8192, 16) | 0.958 | 3.630 | 4.588 | 4.761 | 4.634 | 1.04x | 1.01x |

## Where we lose, and limits

- **The forward against FA3** (0.95-0.97x causal, 0.88-0.96x with packed documents) has the causes of the symmetric
  heads' forward gap ([attention.md](attention.md#where-we-lose-and-limits)), and each score tile here is the sum of
  a 128-wide and a 64-wide product. Against cuDNN with equal documents split into rows (the same work without a
  mask, which a real packed batch cannot use) the forward is 0.91-1.05x.
- **Backward**: the fused and `"full"` backwards need one head dim for `q`, `k` and `v` and raise for MLA heads, so
  MLA runs the ranged backward: two kernels with 7 matrix products per block against FA3's one kernel with 5 (FA3
  pads the values to 192 in its backward, which roughly cancels the difference). At these 192-wide keys the
  overlapped `dk, dv` body, the default at head dims up to 128, is 14-16% slower (1.327, 4.545 and 4.278 ms against
  1.143, 3.916 and 3.630 ms), so MLA keeps the serial one.

## Correctness

`tests/test_mosaic_gpu_mla_attention.py` checks the kernel's output and `dq, dk, dv` against an independent fp32
einsum attention (causal, packed documents with sorted, unsorted and reused ids, sliding windows, lengths 200 to
2048, fp16 too; global relative error < 2e-2 for the output and < 3e-2 for the gradients, worst row < 2e-2 for the
output and < 4e-2 for `dv`), and the MLA layer on the kernel against the same layer on XLA (output and every
parameter's gradient, low-rank and full-rank queries, RoPE and NoPE, packed documents, a sliding window), with the
sharded call on two GPUs; `tests/test_mla.py::test_cudnn_takes_the_value_head_dim_unpadded` covers the cuDNN path.
In the benchmark every JAX path's output agrees with the kernel's within 0.9e-3 to 1.9e-3 relative to the largest
value.

## Reproduce

```
SRC=<flash-attention clone> VENV=<fa3-mla-venv> bash benchmarks/build_fa3_mla.sh
python benchmarks/bench_mla_attention.py --lengths 2048,4096,8192 --heads 16,32 --masks causal,documents --rounds 3 \
    --fa3-python <fa3-mla-venv>/bin/python --out results/mla.json
```

## How it works

The kernel accepts the head-dim pairs (64, 64), (128, 128) and (192, 128). In the forward the first `Dv` query/key
channels and the remaining 64 sit in separate shared-memory tiles, each score tile is the sum of two WGMMA products,
and the query tile, which has the output's width, stages the output (210 KiB of shared memory are in use at `Dqk`
192 with two 128-key stages). The ranged backward uses `Dqk` for `q, k, dq, dk` and `Dv` for `v, do, dP, dv`, with
64-key `dq` blocks (128-key blocks do not fit in shared memory). `MLAConfig.implementation` `"auto"` (the default)
or `"mosaic_gpu"` runs the kernel on Hopper in bf16/fp16 at these heads, per batch and head shard under a mesh;
other shapes run cuDNN's MLA mode (cuDNN >= 9.10) or XLA with the values zero-padded to the q/k head dim, and the
absorbed latent-cache decode stays in XLA.
