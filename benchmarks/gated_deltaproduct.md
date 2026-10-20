# Gated DeltaProduct

`ops.linear_attention("gated_deltaproduct", ..., use_qk_l2norm=True)` with `impl="mosaic_gpu"` runs the Gated
DeltaNet Mosaic GPU kernels on the interleaved sequence of `n_h` Householder steps per token: one decay per
token, then `n_h` delta-rule steps. This page compares it with our plain-JAX op (`impl="chunked"`); no
upstream kernel is timed.

## Setup

- H100 80GB HBM3, JAX 0.11.2; `B = 4, T = 2048, H = 8, K = V = 128`, `n_h = 2` (so 2T = 4096 key/value rows
  per sequence); bf16 `q`, `k` (l2-normalised) and `v`, float32 `g = -softplus(N(0, 1) - 1)` and
  `beta = sigmoid(N(0, 1))`.
- Only forward + backward is timed: `jax.grad` of `<o, do>` with respect to all inputs, the median wall-clock
  time of 10 calls after one warm-up (host dispatch included).
- Measured on 17 Oct 2026 as the only timed job on the GPU, beside CPU-only test runs, so the
  numbers are provisional.

## Results

| B, T, H, K, V, n_h | pass | Mosaic GPU ms | chunked ms | chunked / Mosaic GPU time |
|---|---|---|---|---|
| 4, 2048, 8, 128, 128, 2 | fwd+bwd | 0.772 | 8.836 | **11.4x** |

## Correctness

`tests/test_gated_delta_product.py` checks the reference, `chunked` and Mosaic GPU paths against an
independent float64 loop of the recurrence (FLA's `naive_recurrent_gated_delta_product`) with packed
documents, an initial state and the final state, the Mosaic GPU gradients against autodiff of the reference,
and that `n_h = 1` equals Gated DeltaNet exactly; `tests/test_layers_gated_deltaproduct.py` covers the layer (documents, prefill then decode).

## Reproduce

```
python benchmarks/bench_gated_deltaproduct.py --batch 4 --seq 2048 --heads 8 --num-householder 2
```

The command prints the result as JSON.
