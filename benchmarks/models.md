# RWKV-7 in `LanguageModel`: one training step

`rwkv7-0.1b` is a `LanguageModel` preset, so `jla train` runs it. This page times one training step of it (the
step `jla train` builds: AdamW, remat `"full"`, the LM loss) next to two presets of the same size, on synthetic
tokens, with [`bench_end_to_end_step.py`](bench_end_to_end_step.py).

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 compute.
- The end-to-end config [`end_to_end/fw-gdn-hybrid-125m.toml`](end_to_end/fw-gdn-hybrid-125m.toml) with
  `model.preset` overridden: Llama-3 vocabulary (128,256 ids, tied embedding), sequence 2048, micro batch 4.
- Median of 10 timed steps after 3 warm-up steps.
- Measured on 17 Oct 2026, back to back on one H100 with no other timed run on it. Test jobs of other work could
  still use the GPU, so the step times are provisional.

## Results

Provisional step times, one H100:

| preset | parameters | mixer op | ms/step | tok/s | vs `rwkv7-0.1b` |
|---|---|---|---|---|---|
| `rwkv7-0.1b` at `chunk_size = 16` | 188.9M | `rwkv7` chunked, plain JAX | 290.1 | 28,243 | 1.00 |
| `rwkv6-0.1b` | 194.8M | `rwkv6` chunked, plain JAX | 116.0 | 70,609 | 2.50 |
| `gdn-hybrid-125m` | 190.1M | Gated DeltaNet Mosaic GPU kernels, cuDNN attention | 51.5 | 159,170 | 5.64 |

The `rwkv7-0.1b` row is the exact chunked DPLR op of `jla.ops.rwkv7`. The preset sets `chunk_size = 32`, which
runs the Mosaic GPU RWKV-7 kernels ([rwkv7.md](rwkv7.md); `impl="auto"` takes them for bf16 `r, k, v` at
`chunk_size = 32`). On FineWeb-Edu at 4 × 4096 tokens × 2 microbatches with document masking it trains at 276 ms
per step (118,800 tok/s) against 1,146 ms (28,600 tok/s) on the chunked op at `chunk_size = 16` (the median of
steps 2-20 of two 20-step runs on one H100, provisional).

## Where we lose, and limits

- On the chunked op RWKV-7 is 2.5x slower per step than RWKV-6: every chunk builds four causal pair matrices
  from `[C, C, K]` pairwise-decay tensors and runs two triangular solves (the WY form of the low-rank state
  reads), where RWKV-6's chunked op has one pair matrix and no solve.
- `rwkv6-0.1b` has no Mosaic GPU kernel: it always runs the chunked op.
- `minimax-01-1b` (1.04B parameters) is not timed: with AdamW it does not fit the 20 GB memory share of a timed
  run. Its lightning-attention layers run the `simple_gla` Mosaic GPU kernel in bf16 (head dim 128).

## Correctness

On the kernel path the 20-step losses of `rwkv7-0.1b` equal the chunked op's to within 1e-4. `tests/test_models_lm.py`
checks the RWKV-7 `LanguageModel` scanned against looped and under every remat policy (`tiny-rwkv7`).

## Reproduce

```bash
python benchmarks/bench_end_to_end_step.py benchmarks/end_to_end/fw-gdn-hybrid-125m.toml \
    --steps 10 --warmup 3 --set model.preset=<preset> --set train.micro_batch_per_replica=4
# the chunked RWKV-7 row: --set model.preset=rwkv7-0.1b --set model.linear_attention.chunk_size=16
```
