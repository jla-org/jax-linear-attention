# RWKV-7

`ops.linear_attention("rwkv7", ..., impl="mosaic_gpu")` (`jla.ops.mosaic_gpu.rwkv7`) computes RWKV-7's
recurrence `S_t = diag(exp w_t) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T`, `o_t = r_t^T S_t`, forward and
backward. The RWKV-7 layers call it in the bounded mode (`lower_bound = -0.61`, bf16 `r, k, v`,
`chunk_size = 32`): at head dim 64 the per-chunk work runs in two Mosaic GPU kernels, `rwkv7_prep` forward
and `rwkv7_bwd` backward (two warpgroups per CTA), around the vector-gate state passes of the KDA kernels.
This page compares that call with FLA main's `chunk_rwkv7` on the path FLA's `RWKV7Attention` runs in
training.

## Setup

- H100 80GB HBM3, JAX 0.11.2; RWKV-7 inputs, all bf16, head size 64 (`K = V = 64`).
- FLA main `6dff3e1` (torch 2.13.0+cu129, Triton 3.7.1):
  `chunk_rwkv7(r, w, k, v, a, b, scale=1.0, safe_gate=True, lower_bound=-0.6065306597126334, chunk_size=64)`,
  which `fla/layers/rwkv7.py` takes whenever `self.training or seq_len >= 64 or torch.is_grad_enabled()`
  (`bench_rwkv7.py`'s `fla_layer`). With every optional backend installed, FLA's default dispatch picks the
  same Triton kernels for this call (its TileLang DPLR backend declines chunk 64 on sm_90). FLA main `9d478fa`
  changes no RWKV-7 code. The official reference is BlinkDL RWKV-LM `20be0f8c` (`RWKV-v7/train_temp`).
- Median kernel time per call (CUPTI for JAX, `torch.profiler` for FLA), 30 calls after 3 warm-ups, the
  faster of the rounds, ms. Ratio = FLA time / ours, so > 1 means ours is faster. `fwd+bwd` is the forward plus
  the backward for a fixed output cotangent. Packed documents are our `segment_ids` against FLA's
  `cu_seqlens`.
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

| B, T, H (K = V = 64) | pass | ours | FLA layer path | FLA / ours |
|---|---|---|---|---|
| 1, 4096, 12 | fwd | 0.203 | 0.309 | **1.52x** |
| 1, 4096, 12 | fwd+bwd | 0.677 | 0.989 | **1.46x** |
| 2, 4096, 32 | fwd | 0.627 | 0.803 | **1.28x** |
| 2, 4096, 32 | fwd+bwd | 2.461 | 2.967 | **1.21x** |
| 8, 4096, 12 | fwd | 0.921 | 1.112 | **1.21x** |
| 8, 4096, 12 | fwd+bwd | 3.581 | 4.240 | **1.18x** |
| 16, 4096, 32 (the 1.3B layer) | fwd | 4.751 | 5.195 | **1.09x** |
| 16, 4096, 32 (the 1.3B layer) | fwd+bwd | 19.318 | 20.432 | **1.06x** |
| 1, 65536, 32, 16 documents | fwd | 4.774 | 5.486 | **1.15x** |
| 1, 65536, 32, 16 documents | fwd+bwd | 18.342 | 21.276 | **1.16x** |
| 1, 16384, 8, 16 documents | fwd | 0.346 | 0.448 | **1.29x** |
| 1, 16384, 8, 16 documents | fwd+bwd | 1.261 | 1.627 | **1.29x** |

**The FLA baseline at the 1.3B layer shape** was timed with no other jobs on the GPU. FLA main `c6e4f00` and
`6dff3e1` time the same within 0.2%, interleaved in one process: 5.196 against 5.193 ms forward and 20.37
against 20.40 ms forward + backward at `(16, 4096, 32)`, 5.497 against 5.498 and 21.28 against 21.27 ms with 16
documents at `(1, 65536, 32)`. An earlier FLA time of 5.91 and 22.5 ms at `(16, 4096, 32)` came from a GPU that
another process shared (15 GB of its memory in use) and overstates FLA's time.

## Where we lose, and limits

- **The lead is smallest at the 1.3B layer shape** (1.09x forward, 1.06x forward + backward). In a
  provisional profile there the forward is `rwkv7_prep` 3.45 + state pass 1.01 + chunk decays 0.29 ms, and forward +
  backward adds `rwkv7_bwd` 11.72, the saving prep 3.75, the reverse state pass 1.22 and the `d_carry` reduce
  0.36 ms. The `d_carry` reduce over `states * d_states` is the one XLA pass left in the backward (0.073 ms at
  `(8, 4096, 12)`, 0.364 at the 1.3B shape).
- **Chunk size.** `RWKV7Config.chunk_size` defaults to 16, so `impl="auto"` keeps the `chunked` op there; with
  `chunk_size = 32` (the `rwkv7-0.1b` and `rwkv7-1p3b` presets) and bf16 the layers run the kernels. Off
  Hopper, or with f32 compute, the `chunked` op's pairwise-decay tensors grow with the chunk size (about twice
  the memory at 32 as at 16).
- **Head dim 128:** the bounded mode's per-chunk work runs in XLA; only head dim 64 has the Mosaic GPU
  per-chunk kernels.
- **Decode:** single-token decode in the language model's RWKV-7 mixer runs the reference recurrence; there
  is no recurrent decode kernel.
- **Not supported by the kernel path:** `cu_seqlens` (use `segment_ids`), head dims other than 64 and 128,
  fp32 or fp16 `r, k, v`. The bounded mode does not check its bound on the values.

## NaN losses in training on the kernel path: cause and fix

Training `rwkv7-0.1b` on FineWeb-Edu with `chunk_size = 32` (so that the kernels run)
and document masking gave NaN losses on some batches, with either initialisation, in the initialisation A/B of
[init_parity.md](init_parity.md); the `chunked` op had none. The cause was the forward kernel `rwkv7_prep`: it formed the WY inverse `T = (I - A_ab)^-1` as the power
series `(I + A)(I + A^2)(I + A^4)(I + A^8)(I + A^16)` over the whole 32-token chunk in split-bf16 products.
When consecutive keys are nearly parallel, `a` is large and the decay slow, the powers grow like
`C(30, 15) a^16` (4e6 at `a = 0.8`) before they cancel, so the rounding left errors of order 10-100 in `T`,
the per-chunk transition became expanding and the state overflowed. The fix runs the series on the 8 x 8
diagonal blocks only (`(I + A)(I + A^2)(I + A^4)`) and doubles the inverse to 16 and 32 rows by the exact
block formula `T_21 = T_22 A_21 T_11` (`rwkv7_prep._inverse`, and `rwkv7._block_inverse` on the XLA path).
With the fix, four 2000-step runs (both initialisations, two seeds) had no rejected update, and
`test_bounded_mode_nearly_parallel_keys` covers that regime. Cost (provisional): the prep kernel is about 2%
slower and the layer's forward 1.2-1.6% slower; forward + backward is unchanged.

## Correctness

`benchmarks/rwkv7_check.py op` runs ours, FLA main `c6e4f00` (`chunk_rwkv7` with its defaults and as its
`RWKV7Attention` calls it) and a float64 token-by-token oracle written from the official CUDA kernel's update
on identical inputs: everything agrees to bf16 noise (ours against FLA at most 1.1e-2, `db`), with no drift at
4096 tokens and packed documents matching FLA's varlen kernels; in fp32 our ops match the oracle to 4e-7. With
the same weights copied into FLA's `RWKV7Attention` and `RWKV7FeedForward` (`rwkv7_check.py layer`), our
layers agree with FLA's float32 recurrent kernel to 5e-7. Tests: `tests/test_mosaic_gpu_rwkv7.py` (H100,
against the token-by-token f32 oracle and autodiff of the f32 `chunked` op), `tests/test_rwkv7_kernel_glue.py`
(bitwise), `tests/test_rwkv7.py` and `tests/test_rwkv7_official.py` (the official formulas in float64).

## Reproduce

```
python benchmarks/bench_rwkv7.py --torch-python FLA_MAIN/bin/python --torch-python-full FLA_MAIN_FULL/bin/python \
    --impls mosaic_gpu_layer,fla_layer,full_fla_layer --out results/rwkv7.json
# the 1.3B layer and 16 packed documents (a sixth shape entry packs N documents into the row):
#   --shapes "16,4096,32,64,64;1,65536,32,64,64,16;1,16384,8,64,64,16"
python benchmarks/rwkv7_check.py op --torch-python FLA_MAIN/bin/python      # ours vs FLA vs a float64 oracle
python benchmarks/rwkv7_check.py layer --torch-python FLA_MAIN/bin/python   # our layers vs FLA's, same weights
```

## How it works

Per chunk, FLA's DPLR algorithm gives the low-rank reads `U = W S + U_wy` (`W = T (a exp(G_{t-1}))`,
`U_wy = T A_ak V`), the output `o = (q exp(G)) S + A_qk V + A_qb U` and the next state. Stacking the chunk's 32
`k` rows over its 32 `b` rows makes this exactly the vector-gate state pass of the KDA kernels on one 64-row
chunk, so the serial passes run on those kernels unchanged. In the bounded mode the caller guarantees
`lower_bound <= w` (`-0.61` covers RWKV-7 after bf16 rounding), and each chunk's four score matrices are one
dense product through the chunk's middle token, as split-bf16 tensor-core products (`BF16X3`); the default
mode is exact for any `w <= 0`. With packed documents both state passes are cut at document starts into up
to 16 pieces, as KDA's ([kda.md](kda.md#how-it-works)).
