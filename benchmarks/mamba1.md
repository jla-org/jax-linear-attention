# Mamba-1 selective scan: Mosaic GPU kernels against the official CUDA kernel

`linear_attention("mamba1", ...)` is the selective scan of Mamba, Jamba and Falcon-Mamba (`q = C`,
`k = B`, `v = u`). It has three implementations: the token-by-token `reference` (the oracle), the
`chunked` XLA op and, on Hopper, the Mosaic GPU kernels of `jla.ops.mosaic_gpu.mamba1` (forward) and
`mamba1_bwd2` (backward), which `impl="auto"` takes for every call they support. This page compares
the kernels with the official CUDA selective scan of state-spaces/mamba.

## Setup

- One H100 80GB HBM3, JAX 0.11.2, torch 2.13.0+cu129. bf16 `u, delta, z, B, C`, float32 `A, D,
  delta_bias`, `delta_softplus=True`; every case has `D`, `z` and `dt_bias`.
- Baseline: state-spaces/mamba main `e9594ce`, built for sm_90 from source. `selective_scan_cuda.fwd`
  and `.bwd` are called exactly as `selective_scan_fn` calls them, in their `[B, D, T]` layout
  (`benchmarks/torch_mamba1.py`); ours runs in the `[B, T, D]` layout with the same dtypes.
- Timing: the median CUDA kernel time per call over 30 calls after 3 warmups (CUPTI through
  `harness.median_kernel_ms` for JAX, `torch.profiler` for the official kernel). Every implementation
  runs interleaved in one locked benchmark session on one GPU, the official one in a subprocess on the
  same GPU. `fwd+bwd` is one forward plus the backward for a fixed output cotangent.
- Measured on 20 Oct 2026 with no other jobs on the GPUs. The times per call and the `d_state = 32`
  shape are from 19 Oct 2026 and provisional.

## Results

Official time over ours (> 1 means ours is faster):

| B, T, D, N | forward | forward + backward |
|---|---|---|
| 8, 2048, 1536, 16 (mamba-130m layer) | **1.63x** | **1.85x** |
| 1, 4096, 8192, 16 (Falcon-Mamba-7B / Jamba width) | **1.19x** | **1.72x** |
| 8, 2048, 1536, 32 (provisional) | 1.81x | 1.60x |

At `d_state = 16` the kernels are 1.19-1.63x the official kernel forward and 1.72-1.85x forward +
backward.

Milliseconds per call (19 Oct 2026, provisional), with the official time over ours:

| B, T, D, N | pass | ours | official CUDA |
|---|---|---|---|
| 8, 2048, 1536, 16 | fwd | 0.272 | 0.446 (1.64x) |
| | fwd+bwd | 1.233 | 2.281 (1.85x) |
| | bwd alone | 0.961 | 1.835 (1.91x) |
| 1, 4096, 8192, 16 | fwd | 0.505 | 0.603 (1.19x) |
| | fwd+bwd | 1.833 | 3.132 (1.71x) |
| | bwd alone | 1.329 | 2.529 (1.90x) |
| 8, 2048, 1536, 32 | fwd | 0.436 | 0.789 (1.81x) |
| | fwd+bwd | 2.695 | 4.319 (1.60x) |
| | bwd alone | 2.259 | 3.530 (1.56x) |

## Where we lose, and limits

- **Forward at small grids.** At the 7B width the forward is 1.19x the official kernel, against 1.63x
  at the mamba-130m shape: one CTA per 128 channels and batch row gives 64 CTAs for the H100's 132
  SMs, and the forward has no sequence split yet (the backward has one).
- **Occupancy.** One CTA (four warps) per SM, at 241 to 252 registers per thread (no spills) and about
  220 KB of shared memory.
- **`d_state = 32`** runs as two 16-state slices. Each slice re-reads `dt, u, z, dy` and writes float32
  partial `ddt, du, dz` that XLA adds, so forward + backward is 1.60x there against 1.85x at 16 states.
- **Segment count.** `num_segments` breaks near-ties towards fewer segments: at the 7B width 4 segments
  measured 2 % faster than the 2 it picks.
- **Calls the kernels decline** (`D` not a multiple of 128, grouped `B, C`, fp16, `d_state` other than
  4, 8, 16 or 32, `dt_softplus=False` with `T` not a multiple of 32, a GPU other than sm_90) fall back
  to `chunked`. JAX's gradient of the chunked scan (one `lax.associative_scan` per chunk inside a
  `jax.checkpoint`-ed `lax.scan`) is wrong on the H100 for some inputs, though exact on CPU, and the
  cause is not isolated. On a GPU `chunked` therefore runs the chunked scan forward and takes the
  reference op's exact gradient (a `jax.custom_vjp` that recomputes the reference forward), which is
  slow: at the mamba-130m shape `chunked` takes 6.468 ms forward and 83.371 ms forward + backward, and
  `reference` 6.358 and 76.988 ms (provisional). At `B = 4, T = 2048, D = 1536, N = 16`
  (17 Oct 2026, provisional) the forward took 3.448 ms (`chunked`) and 5.969 ms (`reference`), and
  forward + backward 18.000 ms through the chunked scan's own gradient (the path that is wrong on the
  H100 and no longer runs there) and 38.796 ms for `reference`.

## Correctness

`tests/test_mosaic_gpu_mamba1.py` (H100) checks the forward (float32 to 1e-4) and every backward path
against the reference op: every gradient (`C, B, u, dt, A, D, z, dt_bias` and the initial state) in
float32 and bf16, odd `T`, packed documents (`segment_ids`), `d_state` 4, 8, 16 and 32, the sequence
split (including documents that start inside a segment or exactly at its first token), the layer on
the kernels against the layer on the reference, and that `impl="auto"` returns exactly the kernels'
result; `tests/test_kernel_sharding_gla_mlstm_mamba1.py` runs the kernels on two GPUs against one, and
`tests/test_mamba1.py` covers the reference-gradient path of `chunked` on CPU. Against the official
kernel on the same bf16 values (`benchmarks/diff_upstream.py`), ours and the official agree to bf16
noise on the output, the final state and every gradient, both are as close to the float32 reference as
the token-by-token reference op is, and in float32 all are within 3e-6; the largest difference is `dz`
(up to 6.7e-3 of its maximum), where the official kernel is the less accurate of the two.

## Reproduce

```
python benchmarks/bench_mamba1.py --batch 8 --seq-len 2048 --dim 1536 --d-state 16 \
    --impls mosaic_gpu,mosaic_gpu:kernel@1 --modes fwd,fwdbwd \
    --torch-python <venv with selective_scan_cuda>/bin/python --out results/mamba1_130m.json
python benchmarks/bench_mamba1.py --batch 1 --seq-len 4096 --dim 8192 --d-state 16 \
    --impls mosaic_gpu,mosaic_gpu:kernel@1 --modes fwd,fwdbwd \
    --torch-python <venv with selective_scan_cuda>/bin/python --out results/mamba1_7b.json
python benchmarks/bench_mamba1.py --batch 8 --seq-len 2048 --dim 1536 --d-state 32 --impls mosaic_gpu \
    --modes fwd,fwdbwd --torch-python <venv with selective_scan_cuda>/bin/python --out results/mamba1_n32.json
python benchmarks/build_time_mamba1.py                 # lowering and compile time; --d-state 32 for 32 states
python benchmarks/diff_upstream.py --ops mamba1 --mamba-python <venv with selective_scan_cuda>/bin/python \
    --out results/mamba1_diff.json
pytest tests/test_mosaic_gpu_mamba1.py tests/test_mamba1.py
```

`mosaic_gpu:kernel@S` forces the backward's sequence split to `S` segments (`@1`: no split).

## How it works

The forward kernel gives each CTA 128 channels, one per thread, with the channel's `N` states in
registers, and fuses the softplus, the `D` skip and the `z` gate. The backward keeps that layout and
walks each row's 32-token tiles in reverse from the state the forward saved at every tile start: it
stores the state entering every 2-token sub-tile in shared memory, recomputes each sub-tile's states in
registers and runs the adjoint `g_{t-1} = exp(delta_t A) (dy_t C_t + g_t)` backwards over them, two
`exp2` per state element and token. The per-token `dB, dC` sums over the channels go through a
butterfly reduce-scatter of warp shuffles and one TMA store per tile, with no atomics, so the backward
is deterministic (the official kernel adds them with global atomics). When `D / 128 * B` CTAs would
leave SMs idle, the backward splits each row into up to 8 segments that run in parallel: a pre-pass
returns each segment's adjoint at its start and its decay, and XLA chains them into the cotangent
entering every segment, which is exact up to float32 rounding. The split is worth 1.07x at the
mamba-130m shape (96 CTAs become 384) and 1.43x at the 7B width (64 CTAs become 128).

- **Dispatch.** `impl="auto"` takes the kernels on an H100 when `D` is a multiple of 128, `B, C` are
  shared by all channels (`[B, T, N]`), `u, dt, z` are bf16 or float32, `d_state` is 4, 8, 16 or 32
  (32 as two 16-state slices whose float32 partials XLA adds before rounding) and, for
  `dt_softplus=False`, `T` is a multiple of 32. `backward="kernel_v1"` selects the first backward
  kernel (`mamba1_bwd`), which is slower and takes about two minutes to build per configuration.
- **Build time.** One `jax.jit(jax.grad(...))` at the mamba-130m shape lowers in 15.9 s and compiles in
  1.4 s, and in 22.7 s and 1.7 s at `d_state = 32`. The kernel calls are jitted with their
  configuration static, so every call of one configuration in a traced program shares one lowering.
  The forward alone (inference and prefill) lowers in about 4 s.
- **Several devices.** `mamba1_sharded` splits the batch over the `batch` rule's mesh axes and the
  channels over the `heads` rule's axes (`jla.distributed.kernel_sharding.shard_layout_call`).
- **The layer.** `jla.layers.mamba1` runs its depthwise conv on the fused short-conv kernels
  ([short_conv.md](short_conv.md)) for bf16 layers with `SSMConfig.conv_impl = "auto"`.
