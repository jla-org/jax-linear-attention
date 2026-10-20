# Fused SwiGLU GEMMs at the 1.3B MLP shape

`jla.ops.mosaic_gpu.fused_gemm` runs the SwiGLU activation in the gate/up GEMM's epilogue and its
backward in the down projection's dgrad epilogue, so neither the `[M, 2F]` pre-activations' read nor
the separate activation passes touch HBM. It is opt-in (`model.mlp_kernel = "mosaic_gpu"`; the default
is `"xla"`) and is compared with what XLA runs: a cuBLASLt GEMM plus an XLA fusion.

## Setup

- One H100 80GB SXM, JAX 0.11.2; bf16 operands with f32 accumulation, the 1.3B MLP shape (65,536 tokens
  = 16 x 4096, `d_model` 2048, `d_ff` 5632).
- Baseline: XLA (cuBLASLt) GEMMs and XLA's fusions of the same operations.
- Timing: CUPTI kernel time, the median of 32 calls after 3 warm-up calls, two interleaved rounds.
  Percentages of peak use 989 TFLOP/s, the dense bf16 peak.
- Measured on 17 Oct 2026, each timed run on its own GPU while other jobs could share the machine, so
  the numbers are provisional.

## Results

| operation | XLA (cuBLASLt GEMM + fusion) | fused kernel | speedup |
|---|---|---|---|
| gate/up GEMM + SwiGLU, `h` only | 4.48 ms | 3.90 ms | 1.15x |
| gate/up GEMM + SwiGLU, also storing the pre-activations for the backward | 4.49 ms | 4.17 ms | 1.08x |
| down-projection dgrad + SwiGLU backward | 3.33 ms | 2.56 ms | 1.30x |
| whole MLP block, forward | 6.40 ms | 5.84 ms | 1.10x |
| whole MLP block, forward + backward | 21.1 ms | 20.1 ms | 1.05x |

Counting only GEMM FLOPs, the fused gate/up kernel runs at 774 TFLOP/s (78% of peak) with `h` only and
725 TFLOP/s when it also stores the pre-activations; cuBLASLt's GEMM alone runs at 800 TFLOP/s at this
shape, so the kernel's GEMM is about 3% slower and the kernel wins by removing the activation pass.

For context, through XLA the 1.3B presets' projection and MLP GEMMs already run at 743-832 TFLOP/s
(75-85% of peak) in all three forms (forward, dgrad, wgrad), except Mamba-2's 256-wide B/C projection
(580-611 TFLOP/s); the best bf16 GEMM measured on this GPU, XLA at 8192^3, runs at 811 TFLOP/s
(82.0%). The memory-bound passes next to them already run near the HBM bandwidth (RMSNorm forward
2851 GB/s, SwiGLU forward 3087 GB/s, SwiGLU backward 2506 GB/s), so removing their traffic by fusing
them into the GEMMs is the way to speed them up.

## Where we lose, and limits

- **The fused dgrad** (2.56 ms) is slower than the bare cuBLASLt dgrad (1.84 ms): its epilogue moves
  3 GB (pre-activations in, gradients out), and only the first tile load overlaps the k-loop. Both
  kernels run their epilogue after the k-loop; a ping-pong schedule (each consumer warpgroup owning
  whole tiles) would hide most of the dgrad's 0.7 ms and the forward's 0.27 ms for storing the
  pre-activations.
- **No end-to-end measurement.** With full rematerialization the kernel times above save 1.67 ms per
  layer (12.30 against 10.63 ms), about 40 ms of a 1978 ms `transformer-1p3b` step over 24 layers, an
  expected 1.02x in tokens/s for the presets with the 5632-wide SwiGLU MLP. The A/B run
  (`bench_fused_gemm_e2e.py`) has not run, so that is an estimate; the training losses of the fused
  path are covered by the model-level gradient test, not by a training run. Mamba-2 and Mamba-3 have
  no MLP.
- **Not fused:** the RMSNorm before the MLP (at most about 9 ms of a 1978 ms step), the residual add
  (XLA already fuses it into the next norm's reduction) and Mamba-2's gated RMSNorm before `out_proj`
  (as a GEMM prologue its sigmoid per element would make the GEMM ALU-bound).
- `unsupported_reason` needs an H100, tokens % 128 == 0, `d_model` % 64 == 0, `d_ff` % 256 == 0 and
  bf16 or fp16; `FusedSwiGLU` runs the XLA path for anything else (decode steps, CPU).

## Correctness

`tests/test_mosaic_gpu_fused_gemm.py` checks both kernels against a float32 oracle on the bf16-rounded
operands (relative error < 1e-2 for `h`, the pre-activations, `dgate` and `dup`) over several tile
orders, warpgroup splits, pipeline depths and grids with more tiles than SMs, the MLP's value and all
three gradients in bf16 and fp16 (with and without `recompute`, also inside `jax.checkpoint`) within
2e-2 of the oracle and no worse than 1.5x the XLA path's own error, and a tiny model with
`mlp_kernel = "mosaic_gpu"` against the same model on XLA (loss and every parameter gradient).

## Reproduce

```bash
python benchmarks/bench_fused_gemm.py --parts gemms,passes --iterations 32 --json results/gemms.json
python benchmarks/bench_fused_gemm.py --parts fused --sweep --iterations 32 --json results/fused.json
python benchmarks/bench_fused_gemm_e2e.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B \
    --variants xla,fused,xla,fused --json results/e2e_transformer.json
pytest tests/test_mosaic_gpu_fused_gemm.py
```

## How it works

The kernels are adapted from JAX's `hopper_matmul_mgpu.py` (Apache-2.0): persistent (one CTA per SM,
`planar_snake` tile order), warp-specialized (one TMA producer warpgroup, two WGMMA consumer
warpgroups), 128 x 256 CTA tiles and 64-wide k-steps. The forward, `swiglu_gate_up`, multiplies
`x [M, D]` by the gate/up weight with gate and up columns interleaved in blocks of 64 features, so each
consumer warpgroup's accumulator holds the gate and up pre-activations of the same features and the
epilogue writes `h = silu(gate) * up` from the f32 accumulator. The backward, `swiglu_down_dgrad`,
computes `dh = dy @ W_down^T` with an epilogue that loads the matching gate and up tiles by TMA and
writes `dgate = dh * up * silu'(gate)` and `dup = dh * silu(gate)`. `swiglu_mlp` wraps the block in a
`custom_vjp` with cuBLAS for the other GEMMs; with `remat = "full"` it saves only its inputs and
recomputes the pre-activations in the backward (`recompute=True`), so the primal pass can run the
cheaper `h`-only kernel.
