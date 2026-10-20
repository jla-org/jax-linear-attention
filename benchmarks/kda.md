# KDA and Gated DeltaNet-2

`ops.linear_attention("kda" | "gdn2", ..., impl="mosaic_gpu")` (`jla.ops.mosaic_gpu.kda` / `gdn2`) runs the
vector-gate Mosaic GPU kernels: a delta rule with a per-key-channel decay, forward and backward, with packed
documents through `segment_ids`. KDA is compared with FLA main's `chunk_kda` for training and with FlashKDA
main's fused forward on FlashKDA's inference contract. Gated DeltaNet-2 (GDN-2) is implemented clean-room from
the equations of arXiv 2605.22791: no upstream GDN-2 code is read or run, so it has no upstream baseline, and
it is compared with our plain-JAX `chunked` op and with our KDA kernels on the same shapes.

## Setup

- H100 80GB HBM3, JAX 0.11.2; bf16 `q, k, v` (unit-norm `k`) with float32 gates: `g = logsigmoid(N(0, 1))`
  per key channel, `beta, b, w = sigmoid(N(0, 1))` (the training contract: activated gates).
- FLA main `6dff3e1` (torch 2.13.0+cu129, Triton 3.7.1): `chunk_kda` on its Triton path (`FLA_FLASH_KDA=0`,
  `FLA_TILELANG=0`), and FLA's default dispatch in an environment with TileLang 0.1.15, FlashQLA `5538c35`
  and FlashKDA `7afb9f4`, where a training call runs FLA's Triton forward with the TileLang fused WY backward.
  FLA main `9d478fa` changes none of these kernels.
- FlashKDA main `7afb9f4` (MoonshotAI, MIT): a fused CUTLASS forward for `K = V = 128` with no backward,
  timed on its contract: raw bf16 `q, k` (L2 norm in the kernel), raw bf16 gate logits with
  `A_log = log(linspace(1, 16, H))`, `dt_bias = 0.1 N(0, 1)` and `lower_bound = -5`, beta logits. Ours runs the
  gate activation and the beta sigmoid in XLA, inside the timed call.
- Kernel time of one call (the sum of its CUDA kernel durations: CUPTI for JAX, one measurement per call;
  `torch.profiler` for FLA and FlashKDA), 30 calls after 3 warm-ups, implementations round-robin; cells are
  the faster of the rounds' medians, in ms. Ratio = baseline time / ours, so > 1 means ours is faster.
  `fwd+bwd` is the forward plus the backward for a fixed output cotangent. `--docs 16` packs 16 documents of
  pseudo-random lengths into one row (`segment_ids` for ours, `cu_seqlens` for FLA and FlashKDA).
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

KDA forward + backward is 1.47-1.78x FLA main's default dispatch and 1.51-1.91x its Triton path at the six
dense shapes of the table below, and 1.61-1.63x and 1.73-1.77x with 16 packed documents; the forward with 16
packed documents is 1.37-1.52x both. In a length sweep against FLA's Triton path (`T` 2K-64K, `B * H` 8 and
64, `K = V` 64 and 128), dense inputs give 1.18-1.59x forward and 1.48-1.76x forward + backward. On FlashKDA's
inference contract the forward is 0.57-2.30x FlashKDA main at the dense shapes. GDN-2 is 6.3-8.1x forward
and 6.7-10.8x forward + backward against `chunked` XLA at the six dense shapes.

**KDA, 16 packed documents in one row:**

| B, T, H, K, V | pass | ours | FLA default | FLA Triton | default / ours | Triton / ours |
|---|---|---|---|---|---|---|
| 1, 16384, 6, 128, 128 | fwd | 0.365 | 0.499 | 0.498 | **1.37x** | 1.37x |
| 1, 16384, 6, 128, 128 | fwd+bwd | 1.334 | 2.177 | 2.360 | **1.63x** | 1.77x |
| 1, 16384, 16, 128, 128 | fwd | 0.740 | 1.127 | 1.125 | **1.52x** | 1.52x |
| 1, 16384, 16, 128, 128 | fwd+bwd | 3.348 | 5.393 | 5.786 | **1.61x** | 1.73x |

The length sweep with 16 packed documents in one row, FLA Triton time / ours:

| heads, `K = V` | forward | forward + backward |
|---|---|---|
| 8, 64 | **1.08-1.14x** | **1.49-1.57x** |
| 8, 128 | **1.27-1.35x** | **1.81-1.93x** |
| 64, 64 | **1.22-1.28x** | **1.48-1.53x** |
| 64, 128 | **1.47-1.61x** | **1.82-1.90x** (64K tokens: both sides out of memory with two processes on one GPU) |

**KDA, dense, per shape** (provisional: the shipped kernels on 20 Oct 2026 in a run that other jobs' tests could
share, against FLA main `c6e4f00`, whose `fla/ops/kda` is unchanged up to FLA main `8e625ef`; the fastest of 30
calls, ms):

| B, T, H, K, V | pass | ours | FLA main (Triton) | FLA main (default dispatch) |
|---|---|---|---|---|
| 1, 4096, 8, 128, 128 | fwd | **0.169** | 0.229 (1.35x) | 0.229 (1.36x) |
| 1, 4096, 8, 128, 128 | fwd+bwd | **0.569** | 0.964 (1.69x) | 0.917 (1.61x) |
| 4, 4096, 16, 128, 128 | fwd | **0.704** | 1.077 (1.53x) | 1.078 (1.53x) |
| 4, 4096, 16, 128, 128 | fwd+bwd | **3.310** | 5.489 (1.66x) | 5.103 (1.54x) |
| 8, 2048, 6, 128, 128 | fwd | **0.306** | 0.443 (1.45x) | 0.443 (1.45x) |
| 8, 2048, 6, 128, 128 | fwd+bwd | **1.289** | 2.219 (1.72x) | 2.056 (1.60x) |
| 1, 16384, 16, 128, 128 | fwd | **0.951** | 1.435 (1.51x) | 1.430 (1.50x) |
| 1, 16384, 16, 128, 128 | fwd+bwd | **3.498** | 6.616 (1.89x) | 6.211 (1.78x) |
| 8, 2048, 32, 128, 128 | fwd | **1.353** | 2.131 (1.57x) | 2.123 (1.57x) |
| 8, 2048, 32, 128, 128 | fwd+bwd | **6.565** | 10.814 (1.65x) | 10.007 (1.52x) |
| 4, 4096, 16, 64, 64 | fwd | **0.489** | 0.610 (1.25x) | 0.612 (1.25x) |
| 4, 4096, 16, 64, 64 | fwd+bwd | **1.747** | 2.633 (1.51x) | 2.570 (1.47x) |

**KDA inference on FlashKDA's contract** (forward only; provisional, 17 Oct 2026 with FLA main `c6e4f00`, the
fastest of 30 calls, ms; FLA's default dispatch hands these calls to FlashKDA):

| B, T, H, K, V | ours | FLA main (Triton) | FLA main (default dispatch) | FlashKDA main |
|---|---|---|---|---|
| 1, 4096, 8, 128, 128 | **0.187** | 0.216 (1.15x) | 0.431 (2.30x) | 0.434 (2.31x) |
| 4, 4096, 16, 128, 128 | 0.801 | 1.030 (1.29x) | 0.658 (0.82x) | **0.656 (0.82x)** |
| 8, 2048, 6, 128, 128 | 0.340 | 0.430 (1.26x) | 0.308 (0.90x) | **0.307 (0.90x)** |
| 1, 16384, 16, 128, 128 | **1.040** | 1.339 (1.29x) | 1.828 (1.76x) | 1.829 (1.76x) |
| 8, 2048, 32, 128, 128 | 1.543 | 2.008 (1.30x) | 0.876 (0.57x) | **0.877 (0.57x)** |

**GDN-2** (provisional, the same run as the dense KDA table; no upstream code is run), against our KDA
kernels on the same shape, KDA time / GDN-2 time in parentheses:

| B, T, H, K, V | pass | ours | our KDA |
|---|---|---|---|
| 1, 4096, 8, 128, 128 | fwd | 0.195 | 0.169 (0.87x) |
| 1, 4096, 8, 128, 128 | fwd+bwd | 0.650 | 0.560 (0.86x) |
| 4, 4096, 16, 128, 128 | fwd | 0.866 | 0.705 (0.81x) |
| 4, 4096, 16, 128, 128 | fwd+bwd | 3.852 | 3.303 (0.86x) |
| 8, 2048, 6, 128, 128 | fwd | 0.337 | 0.304 (0.90x) |
| 8, 2048, 6, 128, 128 | fwd+bwd | 1.510 | 1.285 (0.85x) |
| 1, 16384, 16, 128, 128 | fwd | 1.093 | 0.944 (0.86x) |
| 1, 16384, 16, 128, 128 | fwd+bwd | 4.070 | 3.497 (0.86x) |
| 8, 2048, 32, 128, 128 | fwd | 1.852 | 1.356 (0.73x) |
| 8, 2048, 32, 128, 128 | fwd+bwd | 7.688 | 6.569 (0.85x) |
| 4, 4096, 16, 64, 64 | fwd | 0.571 | 0.487 (0.85x) |
| 4, 4096, 16, 64, 64 | fwd+bwd | 2.035 | 1.740 (0.85x) |

With 16 packed documents GDN-2 takes 0.442 ms forward and 1.539 ms forward + backward at
`(1, 16384, 6, 128, 128)`, and 0.912 and 3.917 ms at `(1, 16384, 16, 128, 128)`.

End to end on one H100, `kda-1p3b` trains at 35,110 tokens/s (32.9% MFU) in BF16 and 42,756 (40.1%) in FP8,
1.49x and 1.77x flame measured the same day ([end_to_end_1p3b.md](end_to_end_1p3b.md)); it runs 16 x 16 heads
per call, so the document split does not run there.

## Where we lose, and limits

- **KDA inference against FlashKDA at large `B * H`:** 0.57x at `(8, 2048, 32)`, 0.82x at `(4, 4096, 16)` and
  0.90x at `(8, 2048, 6)`; ours is faster with few heads in flight (`(1, 4096, 8)` and
  `(1, 16384, 16)`). At `(8, 2048, 32)` ours takes 1.552 ms: the intra-chunk work with the L2 norm 0.672, the
  state pass 0.287, the solve 0.248, the cumsum 0.204 and the XLA gate activation 0.133. FlashKDA (0.880 ms)
  fuses the gate activation, cumsum, decays and 16-token inverses into one prepare kernel with bf16
  intermediates, and keeps a bf16 state with an approximate `exp`; ours keeps exact semantics, an f32 state
  and no bound on the gate. Closing the gap needs an exact fused raw-gate prepare kernel first.
- **GDN-2 costs more than KDA:** 11-36% more forward and 15-17% more forward + backward on the same shapes
  (two more gate tensors to read and two more gradients to write).
- **GDN-2 gate-parameter gradient at mild decays:** at the layers' default `dt_bias` init (mean log-decay
  -0.23 per token) GDN-2's `dA_log` is 2.7% off on one of four heads (0.4-0.5% on the others; `chunked`
  0.0-0.3%). KDA there is at `chunked`'s level (0.06-0.6% per head).
- **Causality at rounding level:** the kernels pick a path per chunk from the data, so changing gates later
  in the same chunk can change earlier outputs by rounding (at most 3.7e-3 of the max for the outputs and
  5.6e-3 for the gradients, the size of the kernels' bf16 error); a leak of a future token would move them by
  O(1). The reference and `chunked` are causal bit for bit.
- **Not supported by the kernels:** `cu_seqlens` (use `segment_ids`; `chunked` does not take it either) and
  float32 inputs; the kernels take `K, V` in {64, 128} with `H K` a multiple of 128.

## Correctness

A differential check (`benchmarks/diff_vector_gate.py`) runs the same inputs through a float64 recurrence
written from the equations, our f32 reference, `chunked`, `mosaic_gpu` and FLA main's `chunk_kda`: ours and FLA
agree to bf16 noise in every case (at most 9.5e-3 of the max over every tensor), with packed documents, initial
and final states, extreme gates, odd `T`, `K = 64`, grouped value heads and the gate parameters (`dA_log`,
`ddt_bias`), and the reference matches the float64 recurrence to 1e-7; GDN-2 is checked the same way against
the paper's equations. The document split is bit for bit one chain per row, and the opt-in solve inside
`vg_intra` (`fuse_solve=True`, no faster) is bit for bit the separate `vg_solve`. Tests: `tests/test_mosaic_gpu_kda.py`, `tests/test_mosaic_gpu_gdn2.py`,
`tests/test_vector_gate_doc_split.py`, `tests/test_vector_gate_bwd_schedule.py`,
`tests/test_vector_gate_gate_grads.py`, `tests/test_vector_gate_causality.py` and
`tests/test_mosaic_gpu_vector_gate_fused.py`.

## Reproduce

```
P=--impl-python; MAIN=VENV_FLA_MAIN/bin/python; FULL=VENV_FLA_MAIN_FULL/bin/python; FKDA=VENV_FLASHKDA/bin/python
python benchmarks/bench_kda.py --kind kda --impls mosaic_gpu,fla,fla_default --torch-python $MAIN $P fla_default=$FULL \
    --out results/kda.json
python benchmarks/bench_kda.py --kind kda --modes fwd --contract raw --impls mosaic_gpu,fla,fla_default,flashkda \
    --torch-python $MAIN $P fla_default=$FULL,flashkda=$FKDA
python benchmarks/bench_kda.py --kind gdn2 --impls mosaic_gpu,mosaic_gpu_kda,chunked
# packed documents: add --docs 16 --shapes '1,16384,16,128,128;1,16384,6,128,128'
python benchmarks/diff_vector_gate.py --kind kda --torch-python $MAIN   # differential check
python benchmarks/diff_vector_gate.py --kind gdn2
```

## How it works

Per 64-token chunk, `vg_cumsum` forms the chunk-local cumulative log-decay and flags chunks whose 16-token
spans decay by at most 80; `vg_intra` builds the decayed score matrices (flagged chunks fully on the tensor
cores, through each 32-token block's middle token, adapted from FlashKDA; other chunks with exact 8x8 diagonal
blocks, so no exponential can overflow for any gate); `vg_solve` inverts `I + Akk` exactly in float32;
`vg_wy` forms the WY products; and `vg_state_output` carries the f32 state in registers in one warpgroup
while a second computes the output. The backward runs six kernels and recomputes nothing; it computes every
token pair at most 7 apart exactly in f32, so the gate-parameter gradients stay accurate under strong decays.
A chunk in which a document starts passes no state (a segment start adds a log decay of -128), so with packed
documents and few heads the state passes cut each row at such chunks into up to 16 pieces that run in
parallel. GDN-2 runs the same kernels with the erase gate on the read side of `Akk` and in `w`, and the write
gate on `v`.
