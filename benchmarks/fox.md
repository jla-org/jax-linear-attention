# FoX attention: the XLA path against cuDNN flash attention

Forgetting attention (FoX) adds a per-token forget-gate bias, formed from cumulative log gates, to causal softmax
attention. `jla.layers.forgetting_attention.forgetting_attention` runs it as plain XLA ops (full `[T, T]` scores;
there is no fused FoX kernel yet). It is compared against `jax.nn.dot_product_attention(..., is_causal=True,
implementation="cudnn")` on the same causal shape without a forget gate, the speed target for a fused FoX kernel.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 `q, k, v`; float32 log forget gates `log_sigmoid(N(0, 1) + 3)`.
- Baseline: cuDNN flash attention as shipped with JAX 0.11.2, no forget gate.
- Mean wall-clock time per call over 20 back-to-back calls after 3 warmups (`time.perf_counter`, one
  `block_until_ready` after the last call); `fwd+bwd` is `jax.grad` of the summed output with respect to `q, k, v`
  (and the log gates for FoX). TFLOP/s count the causal half, `4 B H D T (T + 1) / 2`.
- Measured on 17 Oct 2026 as the only timed job on the GPU, next to CPU-only test runs, so the numbers are
  provisional.

## Provisional results

| B, T, H, D | pass | FoX, XLA path ms | cuDNN, no forget gate ms | FoX / cuDNN time |
|---|---|---|---|---|
| 4, 2048, 12, 64 | fwd | 1.420 (18.2 TFLOP/s) | 0.097 (267 TFLOP/s) | 14.7x |
| | fwd+bwd | 4.290 | 0.377 | 11.4x |

## Where we lose, and limits

- The XLA path takes 14.7x the cuDNN time forward and 11.4x forward + backward at this shape.
- It builds the full `[T, T]` scores, so memory grows as `T^2`. A fused kernel (flash attention with the forget-gate
  bias formed per tile from the cumulative log gates) is the next step; this table is its baseline.

## Correctness

`tests/test_forgetting_attention.py` checks the op against naive loops (GQA, packed documents, a sliding
window, a custom scale), its gate gradients against finite differences of a float64 oracle, an open forget gate
against plain causal attention, the layer against a naive layer, and prefill then decode against the full forward
pass.

## Reproduce

```
python benchmarks/bench_fox.py --batch 4 --seq 2048 --heads 12 --dim 64 --out results/fox.json
```
