# Mamba-3: SISO and MIMO kernels against the official Triton and TileLang kernels

Mamba-3 SISO runs `jla.layers.ssm.mamba3_siso(..., impl="mosaic_gpu")`, the fused core
`jla.ops.mosaic_gpu.mamba3_fused.mamba3_siso_fused`: Mosaic GPU kernels do the biases, the angle scan,
the rotation and the recurrence, and XLA does the per-token gate math, `D` and `Z`. It is compared with
the official Triton `mamba3_siso_combined`, and the whole layer with the official `Mamba3` module.
Mamba-3 MIMO runs dedicated kernels and is compared with the official TileLang kernels
([MIMO](#mimo)).

## Setup

- One H100 80GB HBM3, JAX 0.11.2, torch 2.13.0+cu129, Triton 3.7.1; bf16 inputs.
- Baseline: state-spaces/mamba main `e9594ce` (`mamba_ssm` 2.3.2.post1), imported from a source
  checkout. `mamba3_siso_combined(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)` runs an angle
  kernel and a forward kernel (bias, rotation, recurrence, `D`, `Z`), and five more kernels in the
  backward. The layer baseline is the eager `mamba_ssm.modules.mamba3.Mamba3`. MIMO runs on TileLang
  0.1.8 (the version state-spaces/mamba pins) with apache-tvm-ffi 0.1.8 and nvcc 13.4.
- Inputs: one seeded numpy generator with the official test's distributions (RMS-normalised `q, k` in
  one group shared by all heads, `dt` log-uniform in [1e-3, 1e-1], `A` uniform in [-16, -1], `S = K/4`
  angles), bf16-representable so that both frameworks see the same values.
- Timing: CUPTI device time summed over all kernels of one call, through Mosaic `profiler.measure` for
  JAX. The official side is timed with launch-counted `torch.profiler` sessions: each kernel's launches
  per call times its mean time per recorded launch, with lost records reported (`lost_records`), so a
  lost record cannot make the official call look faster. `fwd+bwd` is one forward and one backward
  that returns gradients for every input.
- Measured on 20 Oct 2026 with no other jobs on the GPUs. The MIMO times per shape are from 17 Oct 2026
  and provisional.

## Results

### SISO kernel

| B, T, H, K, V | ours, fwd (ms) | official / ours, fwd | official / ours, fwd+bwd |
|---|---|---|---|
| 4, 4096, 24, 128, 64 | 0.423 | **1.78x** | **1.78x** |
| 1, 16384, 48, 128, 64 | 1.048 | **2.60x** | **2.32x** |
| 8, 2048, 32, 128, 64 | 0.519 | **1.45x** | **1.58x** |

No record was lost. In a sweep over lengths (`T` 2K-64K, `B * H` 8 and 64, head dim 64; one official
record lost in 24 entries) the fused core is 1.93-3.61x the official forward and 1.90-2.97x forward +
backward.

### SISO layer against the official module

`benchmarks/bench_mamba3_layer.py`: ours is `Mamba3` under `jit` with `implementation="mosaic_gpu"`,
the official the eager module, both with bf16 parameters and the official defaults (`d_state` 128, head
dim 64, expand 2, one B/C group, `rope_fraction` 0.5, no output norm, chunk 64); `fwd+bwd` returns the
gradients of the input and of every parameter.

| B, T, d_model | pass | ours (ms) | official (ms) | official / ours |
|---|---|---|---|---|
| 8, 2048, 768 | fwd | 0.695 | 1.079 | **1.55x** |
| | fwd+bwd | 2.076 | 3.031 | **1.46x** |
| 4, 4096, 1536 | fwd | 1.530 | 2.348 | **1.53x** |
| | fwd+bwd | 4.492 | 6.544 | **1.46x** |

By the wall clock of back-to-back calls the official forward at `(8, 2048, 768)` takes 1.150 ms
against its 1.079 ms of kernels, and ours 0.724 against 0.695 ms.

## Where we lose, and limits

- **Extreme gates** (`dt` log-uniform in [0.01, 1], `A` down to −64, trapezoid logits ×8, angle logits
  ×4): our angle gradient is 1.0e-1 from float32 (9.1e-2 for MIMO), where the official's is 3.7e-2
  (2.1e-2); every other tensor stays at or below 1.1e-2. The XLA-glue path shows the same error, so it
  comes from the recurrence kernels, whose bf16 `dq, dk` feed the rotation backward; float32 `dq, dk`
  for the rotary backward is the next step.
- **A `z|x` copy in the layer.** At `d_model` 768 XLA's dot merger fuses the layer's z and x
  projections and copies them back out, a 0.034 ms copy at `(8, 2048, 768)`; the Mamba-2 layer keeps
  the two GEMMs apart with an optimization barrier, and the Mamba-3 layer does not yet.
- **MIMO**: the official kernels' shared-Q/K configuration and our saved-state traffic are under
  [MIMO limits](#mimo-limits). MIMO layers (`R` 2 and 4) were not timed against the official module.

## Correctness

`benchmarks/check_mamba3_official.py` gives identical inputs to the official code
(`mamba3_siso_combined`, the TileLang `mamba3_mimo` with its varlen kernels, and the `Mamba3` module
with random values in every parameter) and to ours, and compares every output, gradient and final
state, with a float32 token-by-token oracle: in every case (dense, odd lengths, grouped heads, full
rotation, initial and final states, our `segment_ids` against its `cu_seqlens`, ranks 1, 2 and 4, the
module's cache after a prefill) ours and the official agree within bf16 noise, at most 2.9e-2 of a
tensor's maximum for SISO, about the official's own distance from float32 (up to 2.2e-2), and our
decode steps after a prefill match the official full forward to 5-8e-3 (the official `Mamba3.step`
could not run next to TileLang 0.1.8). `tests/test_mosaic_gpu_mamba3.py` and
`tests/test_mosaic_gpu_mamba3_fused.py` check the SISO kernels and the fused core against the
reference and chunked ops (values, final states and every gradient, grouped B/C, packed documents,
initial states), and `tests/test_mosaic_gpu_mamba3_mimo_kernels.py` and
`tests/test_mamba3_mimo_release.py` the MIMO kernels (below 2 % of the output scale and 3 % of each
gradient's scale against the float32 reference) and the layer's chunking.

## Reproduce

```
python benchmarks/bench_mamba3.py jax --out results/mamba3_jax.json
python benchmarks/bench_mamba3.py torch --mamba-root /path/to/state-spaces/mamba \
    --memory-fraction 0.2 --out results/mamba3_torch.json
python benchmarks/bench_mamba3_layer.py jax --out results/mamba3_layer_jax.json
python benchmarks/bench_mamba3_layer.py torch --out results/mamba3_layer_torch.json      # official venv
python benchmarks/bench_mamba3_mimo.py jax --torch-python <torch+tilelang venv>/bin/python \
    --mamba-root <state-spaces/mamba checkout> --out results/mamba3_mimo.json
python benchmarks/check_mamba3_official.py torch --dir <dir>                             # official venv
python benchmarks/check_mamba3_official.py jax --dir <dir> --out results/mamba3_official_check.json
```

The JAX runs of `bench_mamba3.py` use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2`, and the torch side needs
`einops`. Repeat each jax/torch pair for the second and third rounds.

## How it works

- **Channel order.** The rotated channel pairs are de-interleaved (`[evens | odds | rest]`), so a
  rotation pairs two column blocks; XLA permutes `C`, `B`, the biases and the states at group size, and
  the recurrence is invariant to the key-channel order.
- **Angles.** Within a chunk the angle prefix sums are exact (three bf16 tensor-core passes,
  `x = hi + mid + lo`); the carry crosses chunks in float32, wrapped to [-π, π]. The angle scan uses
  exact `tanh`, and the rotary kernels `sin.approx`/`cos.approx` on the wrapped angle (absolute error
  ≤ 2^-20.9, far below the bf16 rounding of the rotated `q, k` that follows).
- **Grouped heads.** The CTA sums `dq`, `dk` over the heads of a B/C group, so there are no atomics and
  the results are deterministic.
- **Where the time goes** at `(8, 2048, 32, 128, 64)`: our fused forward (0.519 ms) is recurrence
  0.158, rotary 0.130, angle scan 0.093, the XLA `D`/`Z` gate 0.087 and 0.05 of small XLA glue; the
  official forward is `mamba3_siso_fwd_kernel` 0.647, `angle_dt_fwd_kernel` 0.071 and two bf16 copies
  0.032.

## MIMO

`ops.linear_attention("mamba3_mimo", ..., impl="mosaic_gpu")` runs the dedicated kernels of
`jla.ops.mosaic_gpu.mamba3_mimo_kernels` (chunks of `64 / R` tokens, one state per head) plus the XLA
gate rows. The official call is `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo.mamba3_mimo` with
`chunk_size = 64 / R`, `MIMO_Out=None` (per-rank output, like our op), no `D`, `Z` or norm, and one
Q/K group per head. It also runs its angle cumsum (zero angles), rotation, Q/K biases (zero) and the
value up-projection `v_r = Psi_r v` inside its kernels and reads one `[B, T, H, V]` value tensor, where
our op reads the projected `[B, T, H, R, V]` values and leaves the projection, biases and rotation to
the layer (XLA). The "official recurrence" column therefore counts only its three recurrence kernels
(`mamba_mimo_fwd`, `mamba_mimo_bwd_fwd`, `mamba_mimo_bwd_bwd`), the closest like-for-like comparison.

Inputs: `dt` log-uniform in [1e-3, 1e-1], `A` uniform in [-16, -1], trapezoid logits N(0, 1),
`q, k ~ N(0, 1) / sqrt(R K)`, `v ~ N(0, 1)`, `Psi ~ 1 + N(0, 0.1)`. Ours is the median of 30 calls
after 3 warmups in two rounds with the official run between them, the minimum of the two medians.

Release measurements (20 Oct 2026): forward 1.85-5.57x the official full call and 1.57-4.70x its
recurrence kernels, forward + backward 3.44-6.09x and 2.70-5.37x.

Times per shape, CUPTI kernel ms per call (17 Oct 2026, provisional):

| B, T, H, R, K, V | pass | official | official recurrence | **ours** | SISO route | chunked | official / ours |
|---|---|---|---|---|---|---|---|
| 8, 2048, 32, 4, 128, 64 | fwd | 1.089 | 0.919 | **0.626** | | | 1.74x |
| | fwd+bwd | 7.567 | 5.912 | **2.214** | | | 3.42x |
| 4, 2048, 24, 4, 128, 64 | fwd | 0.840 | 0.703 | **0.254** | 2.461 | 6.946 | 3.31x |
| | fwd+bwd | 4.231 | 3.426 | **0.850** | 6.005 | 23.44 | 4.98x |
| 1, 8192, 32, 4, 128, 64 | fwd | 3.112 | 2.626 | **0.558** | | | 5.58x |
| | fwd+bwd | 14.62 | 12.89 | **2.327** | | | 6.28x |
| 4, 2048, 32, 2, 128, 64 | fwd | 0.508 | 0.407 | **0.172** | 0.842 | 5.796 | 2.95x |
| | fwd+bwd | 2.507 | 1.868 | **0.597** | 2.087 | 18.53 | 4.20x |

The SISO route (`kernel="siso"`: `H R^2` heads of the SISO kernels) and the chunked path (XLA,
16-token chunks) ran only where their forward + backward fit a 25 % memory cap. The kernels are
4.9-9.7x the SISO route forward and 3.5-7.1x forward + backward, and 27-34x the chunked path; by
the wall clock the SISO route itself is 2.7x the chunked path forward and 3.5x forward + backward at
`(4, 2048, 24, 4, 128, 64)`. The
gap to the official kernels is smallest at their tuned shape `(8, 2048, 32, 4, 128, 64)` (256 CTAs)
and largest with few CTAs (`(1, 8192, 32)`: 32 CTAs), where the official kernel, one 128-thread CTA
per (head, batch) without warp specialisation, is bound by its serial chunk loop.

### MIMO limits

- **Shared Q/K and the value projection.** The official kernels' recommended configuration shares one
  Q/K group across all heads (`nheads_qk = 1`; `--official-shared-qk`), which reads 1/32 of the Q/K
  bytes at the main shape, and applies the biases, rotation and `Psi` in the kernel. Our op takes
  per-head, per-rank `q, k, v` (the layer's per-head biases and rotation make them per head), so the
  layer materialises them in XLA. Against that cheaper problem ours is 1.63x forward and 3.34x forward
  + backward at `(8, 2048, 32, 4, 128, 64)` (17 Oct 2026, provisional). A fused rotary kernel as for
  SISO, plus `Psi` applied in the forward kernel, would remove that traffic.
- **Saved states.** The bf16 states per 16-token chunk (`R = 4`) are 0.54 GB written and read at the
  main shape; recomputing them in the backward, as the official kernels do, would trade that traffic
  for a second state pass.

### How the MIMO kernels work

A chunk is `64 / R` tokens times `R` ranks, 64 rows, the SISO kernels' tile height, so the rank-`R`
state update is one `[K, 64] x [64, V]` tensor-core product per chunk and the cost is `R` times one
SISO recurrence over the same tokens. Rows are rank-major (row `r C + i` is rank `r` of token `i`), so
each rank's `[C, D]` block is one TMA box of the free `[B, T, H R, D]` view of the inputs. The kernels
keep the SISO kernels' warp-specialised structure, with 4 chunks in flight when at most 64 CTAs
(`B H`) run and 2 otherwise. At the main shape the forward kernel moves about 1.69 GB in 0.580 ms,
2.9 TB/s or about 87 % of the H100's HBM3 peak, so it is bandwidth-bound. They take ranks 1, 2, 4 and
8, `(K, V)` in (128, 64), (64, 128) and (64, 64), `segment_ids`, initial and final states and any
length; `impl="auto"` picks them when `chunk_size` is None or `64 / R`, and `Mamba3MIMO` uses
`min(SSMConfig.chunk_size, 64 / R)` tokens per chunk.
