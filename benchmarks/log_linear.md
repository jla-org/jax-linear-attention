# Log-linear Mamba-2

`ops.linear_attention("log_linear_simple_gla", ..., impl="chunked")` runs log-linear attention on simple GLA
(the Mamba-2 recurrence): `simple_gla` on a Fenwick tree of O(log T) states, each level weighted per token.
It is a plain-JAX chunked op with no Mosaic GPU kernel yet, so this page compares it with the linear-time
`ops.linear_attention("simple_gla", ..., impl="chunked")`.

## Setup

- H100 80GB HBM3, JAX 0.11.2; chunk size 64, `precision=DEFAULT`; bf16 `q, k, v`, float32 decays
  `-softplus(N(0, 1))` and per-level scales `softplus(N(0, 1))` (12 levels at T = 2048).
- The mean wall-clock time of 10 calls, each followed by `block_until_ready`, after 3 warm-ups. `fwd+bwd` is
  `jax.grad` of the summed output with respect to `q, k, v, g`.
- Measured on 17 Oct 2026 as the only timed job on the GPU, beside CPU-only test runs, so the
  numbers are provisional.

## Results

| B, T, H, K, V | pass | log-linear simple GLA ms | simple GLA ms | log-linear / linear time |
|---|---|---|---|---|
| 1, 2048, 24, 128, 64 | fwd | 1.936 | 0.640 | 3.03x |
| 1, 2048, 24, 128, 64 | fwd+bwd | 8.801 | 1.783 | 4.93x |

## Limits

- The 12 levels cost 3.0x the linear-time op forward and 4.9x forward + backward on the chunked JAX path. A
  kernel that keeps the per-level states on chip is not written yet.
- There is no decode cache: generation recomputes the forward pass.

## Correctness

`tests/test_log_linear_ops.py` checks the reference and `chunked` ops against an independent float64 dense
oracle built from FLA's level masks, with and without packed documents, the gradients against the reference,
and that unit level scales reduce the op to `simple_gla`; `tests/test_log_linear_layer.py` covers the layer.

## Reproduce

```
python benchmarks/bench_log_linear.py --batch 1 --seq-len 2048 --heads 24 --dk 128 --dv 64
```

The command prints the result as JSON.
