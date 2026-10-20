# mLSTM (xLSTM)

`ops.linear_attention("mlstm", ..., impl="mosaic_gpu")` (`jla.ops.mosaic_gpu.mlstm`) is an opt-in path for
xLSTM's max-stabilised mLSTM: it runs the numerator on the simple-GLA Mosaic GPU kernels and the stabiliser and
denominator in XLA. `impl="auto"` keeps the float32 `chunked` op, because the path is less accurate on rows
where `q n` nearly cancels (below). This page compares the path with NX-AI's TFLA Triton kernels and with
`chunked`.

## Setup

- H100 80GB HBM3, JAX 0.11.2; bf16 `q, k, v ~ N(0, 1)` with float32 gates `g = logsigmoid(3 N(0, 1) + 2)` and
  `i = 3 N(0, 1)`.
- NX-AI `mlstm_kernels` 2.0.6 TFLA, `chunkwise--triton_xl_chunk` (the mLSTMexp TFLA kernel of xLSTM-7B) with
  chunk 128 and 256 (`torch_mlstm.py`; torch 2.13.0+cu129, Triton 3.7.1). `mlstm_kernels` is under the NXAI
  Community License Agreement, which is not a permissive licence; it is not a dependency of `jla`:
  `torch_mlstm.py` imports it from a separate environment through its public API, and none of its code was
  read or copied.
- Kernel time of one call (CUPTI for JAX, one measurement per call; `torch.profiler` for torch), 30 calls
  after 3 warm-ups, the implementations interleaved, two rounds, median ms. Ratio = baseline time / ours, so
  > 1 means ours is faster. `fwd+bwd` is the forward plus the backward for a fixed output cotangent
  (gradients for `q, k, v` and both gates).
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

Against TFLA's fastest configuration per shape, the opt-in path is **1.38-1.42x** forward and **1.03-1.06x**
forward + backward. TFLA's backward is not the exact gradient of its forward (see Correctness), so the forward +
backward comparison is our exact gradient against TFLA's approximate one.

Per shape (provisional, 18 Oct 2026: the shipped kernels, on a GPU that a test job could share; TFLA time / ours in
parentheses):

| B, T, H, K, V | pass | ours | TFLA xl 128 | TFLA xl 256 | `chunked` |
|---|---|---|---|---|---|
| 1, 8192, 8, 256, 512 (xLSTM-7B heads) | fwd | **0.484** | 0.668 (1.38x) | 0.711 (1.47x) | 2.989 (6.18x) |
| 1, 8192, 8, 256, 512 (xLSTM-7B heads) | fwd+bwd | **2.203** | 2.369 (1.08x) | 2.327 (1.06x) | 9.654 (4.38x) |
| 4, 4096, 8, 128, 256 | fwd | **0.322** | 0.462 (1.43x) | 0.528 (1.64x) | 2.206 (6.84x) |
| 4, 4096, 8, 128, 256 | fwd+bwd | **1.413** | 1.484 (1.05x) | 1.510 (1.07x) | 6.284 (4.45x) |

## Where we lose, and limits

- **Per-row precision.** On the same bf16 inputs at these head dims with the benchmark's random gates, our worst
  (token, head) row is 4% to 27% off the float32 result where TFLA's is 0.4% to 1.5%, and the error relative to
  max |o| is 3.3x larger; with xLSTM-7B-like gates the worst rows are 0.9% to 1.5% (TFLA 0.4% to 0.6%), and a
  bf16 layer on this path stays within 0.9% per row of the same layer on `chunked`. The speed is bought with
  this tail, which is why the path is opt-in.
- **XLA glue.** About half of every call is XLA around kernels written for `K, V <= 128` heads. At the xLSTM-7B
  heads the simple-GLA kernel takes 0.149 ms of the 0.48 ms forward, and the Mosaic GPU kernels 1.24 ms of the
  2.37 ms forward + backward (XLA 1.13 ms: the per-value-head `dq, dk` sums, the `dv` reductions, the
  duplicated values and the autodiff of the denominator).
- **Not supported by the path** (its `supports` rejects them): `K = 96` (the `xlstm-125m` preset), float32
  layers and `cu_seqlens`; it takes `K` in {64, 128, 256} and `V` in {64, 128, 256, 512}.

## Correctness

`tests/test_mosaic_gpu_mlstm.py` (H100) compares against the float32 scan reference on the same bf16 or fp16
inputs: outputs and final states `(c, n, m)` for all 12 supported `(K, V)` pairs with an initial state and
packed documents, odd and partial-chunk lengths, gates up to |40|, every gradient including the initial state's,
and a bf16 mLSTM layer on this path against the same layer on `chunked`. The bounds are 2e-2 of its max for
`o * D_ref` against the reference `q c`, 1e-1 of max |o| for `o`, 2e-2 for final states, and 2e-2 (`K <= 128`)
and 8e-2 (`K = 256`) for gradients. TFLA computes the same forward, but its `dq, dk, di, df` are the gradient of
the float32 reference with the denominator and the stabiliser held constant, 99% to 123% of their max off the
exact gradient (`benchmarks/diff_upstream.py` records both); ours is autodiff of the forward.

## Reproduce

```
python benchmarks/bench_mlstm.py --torch-python VENV/bin/python --impls mosaic_gpu,tfla_xl128,tfla_xl256,chunked \
    --breakdown --out results/mlstm.json
python benchmarks/diff_upstream.py --ops mlstm --mlstm-python VENV/bin/python
```

## How it works

The max-stabilised mLSTM (`m_t = max(g_t + m_{t-1}, i_t)`, `c_t = exp(g_t + m_{t-1} - m_t) c_{t-1} +
exp(i_t - m_t) k_t^T v_t`, the same for `n_t`, `o_t = q_t c_t / (max(|q_t n_t|, exp(-m_t)) + eps)`) is, in exact
arithmetic, a simple GLA with the log-decay `g'_t = g_t + m_{t-1} - m_t <= 0` and the keys
`k'_t = exp(i_t - m_t) k_t`. The path runs the simple-GLA kernels for the numerator `q c_t` and computes the
stabiliser (a max-plus `lax.associative_scan`) and the denominator `q n_t` (from float32-accumulated scores) in
XLA. `K = 256` runs as two 128-wide key heads whose numerators are summed in float32, and `V` is split into
value heads of 128 columns. A dedicated mLSTM kernel, which would remove most of the glue and could bring the
path into `impl="auto"`, is not written yet.
