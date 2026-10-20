# GLA

`ops.linear_attention("gla", ...)` runs dedicated Mosaic GPU kernels for GLA, `S_t = diag(exp g_t) S_{t-1} +
k_t v_t^T`, `o_t = scale q_t^T S_t`, with a per-key-channel decay, forward and backward
(`jla.ops.mosaic_gpu.gla_kernels`, `pipeline="gla"`, the default; `impl="auto"` runs it on an H100). HGRN2's
GLA core runs on the same kernels. This page compares them with FLA main's `chunk_gla` (Triton; no other FLA
backend serves GLA, so FLA's default dispatch runs the same kernels) and with our plain-JAX `chunked` op.

## Setup

- H100 80GB HBM3, JAX 0.11.2; bf16 `q, k, v` (unit-norm `k`), a float32 log-decay `g = logsigmoid(N(0, 1))`
  per key channel.
- FLA main `6dff3e1` Triton `chunk_gla`, torch 2.13.0+cu129. `fla_full` is the same call in FLA main with its
  optional backends installed (TileLang, FlashQLA, FlashKDA); none of them serves GLA.
- Kernel time of one call (the sum of its CUDA kernel durations: CUPTI for JAX, one measurement per call;
  `torch.profiler` for FLA), 30 calls after 3 warm-ups, the implementations interleaved per shape; ratio =
  FLA time / ours, so > 1 means ours is faster. `fwd+bwd` is the forward plus the backward for a fixed output
  cotangent; packed documents are our `segment_ids` against FLA's `cu_seqlens`.
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

FLA main Triton time / ours, at the five dense shapes and the two packed-document shapes of the table below:

| rows | forward | forward + backward |
|---|---|---|
| dense | **1.69-2.19x** | **1.52-1.77x** |
| 16 packed documents per row | **0.88-1.39x** | **1.01-1.41x** |

Per shape (provisional, 18 Oct 2026: FLA main `c6e4f00`, with no change in `fla/ops/gla` since our port source
`423cff91`, in runs that other jobs' tests could share; the lower of two rounds' medians, ms; FLA time / ours
in parentheses):

| B, T, H, K, V | pass | ours | FLA main Triton | FLA main default dispatch | `chunked` |
|---|---|---|---|---|---|
| 4, 4096, 16, 128, 128 | fwd | **0.414** | 0.948 (2.29x) | 0.952 (2.30x) | 3.967 (9.57x) |
| 4, 4096, 16, 128, 128 | fwd+bwd | **2.029** | 3.457 (1.70x) | 3.454 (1.70x) | 13.731 (6.77x) |
| 8, 2048, 32, 128, 128 | fwd | **0.812** | 1.870 (2.30x) | 1.878 (2.31x) | 7.243 (8.92x) |
| 8, 2048, 32, 128, 128 | fwd+bwd | **4.078** | 6.854 (1.68x) | 6.850 (1.68x) | out of memory |
| 1, 4096, 8, 128, 128 | fwd | **0.091** | 0.185 (2.03x) | 0.185 (2.03x) | 0.959 (10.55x) |
| 1, 4096, 8, 128, 128 | fwd+bwd | **0.371** | 0.660 (1.78x) | 0.660 (1.78x) | 3.550 (9.58x) |
| 4, 4096, 16, 64, 64 | fwd | **0.259** | 0.496 (1.91x) | 0.499 (1.92x) | 2.081 (8.03x) |
| 4, 4096, 16, 64, 64 | fwd+bwd | **1.066** | 1.728 (1.62x) | 1.724 (1.62x) | 7.740 (7.26x) |
| 8, 2048, 6, 64, 128 (`gla-125m` heads) | fwd | **0.117** | 0.228 (1.95x) | 0.228 (1.96x) | 1.016 (8.70x) |
| 8, 2048, 6, 64, 128 (`gla-125m` heads) | fwd+bwd | **0.484** | 0.818 (1.69x) | 0.819 (1.69x) | 3.478 (7.18x) |
| 1, 32768, 16, 128, 128, 16 documents | fwd | **1.183** | 1.876 (1.59x) | | |
| 1, 32768, 16, 128, 128, 16 documents | fwd+bwd | **4.706** | 7.327 (1.56x) | | |
| 1, 16384, 6, 64, 128, 16 documents | fwd | 0.242 | **0.224 (0.93x)** | | |
| 1, 16384, 6, 64, 128, 16 documents | fwd+bwd | **0.787** | 0.831 (1.06x) | | |

## Where we lose, and limits

- **The forward with packed documents at few heads** (`(1, 16384, 6, 64, 128)` with 16 documents: 0.88x FLA's
  forward in the release measurement; the forward + backward still leads). Our state pass walks each row's chunks in order with
  one CTA per (row, head, value tile), so a single row of 6 heads keeps few SMs busy whatever its documents;
  FLA's `cu_seqlens` path runs every document as its own sequence, so 16 documents give it 16 times the
  parallel work.
- **The backward** is about 80% of forward + backward and where the lead over FLA is smallest: `bwd_inter`
  writes three f32 `[B, T, H, K]` partial gradients (`dq, dk, dG`) that `bwd_intra` reads back.
- **Saturated decays:** with every log-decay near -30 the true `dg` is about 3e-13, and the kernels return
  values of up to about 2e-7 (the reverse sum `dG = q dq - k dk` cancels terms of order one in float32, as in
  FLA's `chunk_gla`); at log-decays of -5 and -10 `dg` is within bf16 noise.
- **Not supported by the kernels:** float32 inputs, `cu_seqlens`, head dims other than 64 and 128 and other
  chunk sizes run `chunked`; `T` is limited by the CUDA grid to about 2.1M tokens at `K = 128` and 4.2M at
  `K = 64`.

## Correctness

`tests/test_mosaic_gpu_gla.py` (H100) checks outputs, final states and every gradient (`q, k, v, g`, initial
state) against the token-by-token f32 oracle `jla.ops.reference.gla` within 1.5e-2 (outputs) and 2e-2
(gradients) of its max magnitude, for `K, V` in {64, 128}, odd `T`, huge decays and packed documents; the GLA
and HGRN2 layers on the kernels match the layers on the oracle, and `tests/test_kernel_sharding_gla_mlstm_mamba1.py`
checks two GPUs against one. Against FLA main on the same values (`benchmarks/diff_upstream.py`, packed
documents as `segment_ids` against `cu_seqlens`) ours agrees to bf16 noise on every tensor: at most 6.5e-3 of
the largest magnitude, the same size as each one's own error against the oracle.

## Reproduce

```
python benchmarks/bench_gla.py --torch-python VENV_FLA_MAIN/bin/python --torch-python-full VENV_FLA_MAIN_FULL/bin/python \
  --impls mosaic_gpu,fla,fla_full,chunked --rounds 2 --out results/gla.json
python benchmarks/bench_gla.py --torch-python VENV_FLA_MAIN/bin/python --torch-python-full VENV_FLA_MAIN_FULL/bin/python \
  --impls mosaic_gpu,fla,fla_full --shapes "1,32768,16,128,128;1,16384,6,64,128" --docs 16   # packed rows
python benchmarks/diff_upstream.py --ops gla --fla-python VENV_FLA_MAIN/bin/python
```

## How it works

The kernels follow the stage split of FLA's `chunk_gla` on the building blocks of the vector-gate kernels, in
chunks of 64 tokens: `chunk_cumsum` forms the chunk-local cumulative log-decay `G`; `intra` the decayed scores
`Aqk` and the decayed `q` and `k` tiles; `state_output` carries the f32 state in registers in one warpgroup
while a second computes the output; the backward runs `bwd_dav`, `bwd_state`, `bwd_inter`, `bwd_intra` and
`chunk_reverse_cumsum`. Blocks below the diagonal factor the decay through a reference inside the chunk so
that every exponent is at most zero, and the 8x8 diagonal blocks are exact, so no exponential can overflow for
any gate; the backward computes every token pair at most 7 apart exactly in f32, as the KDA backward does.
`pipeline="vector_gate"` keeps the first version, which ran GLA on the KDA kernels with constant gates.
