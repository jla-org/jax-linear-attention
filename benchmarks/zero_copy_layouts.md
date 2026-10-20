# Layout traffic around the recurrent kernels (1.3B, one H100)

This page measures the layout traffic around the Mosaic GPU kernels in 1.3B training (transposes to
head-major tiles, padding, q/k/v splits, gradient concatenations) for Mamba-2, Gated DeltaNet, KDA and
Mamba-3, and the two switches that remove the largest costs found next to the kernels:
`LinearAttentionConfig.qk_conv_barrier` (GDN, GDN-2, KDA) and `SSMConfig.grouped_key_grads` (Mamba-2).
Both are on by default, and `false` restores the other path, which is the comparison here. FLAT
(Apache-2.0, `flat/linear_attention/op.py` `linear_attention_packed_jax`) avoids such copies by reading
q, k and v straight out of the fused projection and returning one fused gradient.

## Setup

- One H100, JAX 0.11.2, 16 × 4096 tokens per step, real FineWeb-Edu batches with document masking:
  [`bench_e2e_1p3b.py`](bench_e2e_1p3b.py) with `--impl kernels`, 20 steps of which 5 are untimed. A
  (switch off) and B (switch on) alternate in fresh processes on the same GPU
  ([`bench_e2e_ab.py`](bench_e2e_ab.py)); throughput is the median over rounds, loss the loss at step 20.
- Profiles: [`profile_layout_copies.py`](profile_layout_copies.py) builds the same step with the first
  16 × 4096 batch, times it, then profiles 2 steps with XLA command buffers off so that every kernel
  maps to its HLO instruction, and classifies every fusion by its opcodes: pure data movement
  (`transpose`, `copy`, `pad`, `concatenate`, slices, `broadcast`, `convert`, `reshape`, `bitcast`) or
  computation, with the JAX source line and the phase (`fwd`, `bwd`, or `remat`).
- Each run had its GPU to itself while other benchmarks ran on the host's other GPUs (CPU and
  PCIe shared), so the numbers are provisional.

## Results

| preset | switch (B) | rounds | A tok/s | B tok/s | B / A | MFU % | loss, step 20 | peak GiB |
|---|---|---|---|---|---|---|---|---|
| `gdn-hd128-1p3b` | `qk_conv_barrier` | 2 | 31,755 | 35,239 | **1.110** | 31.2 → 34.6 | 7.827 → 7.812 | 42.6 → 42.4 |
| `kda-1p3b` | `qk_conv_barrier` | 2 | 27,918 | 30,849 | **1.105** | 26.2 → 28.9 | 7.864 → 7.869 | 45.9 → 45.9 |
| `gdn2-1p3b` | `qk_conv_barrier` | 1 | 24,215 | 27,225 | **1.124** | 26.9 → 30.3 | 7.845 → 7.820 | 51.2 → 51.2 |
| `mamba2-1p3b` | `grouped_key_grads` | 2 | 31,996 | 32,953 | **1.030** | 30.5 → 31.4 | 7.804 → 7.803 | 45.3-46.8 → 45.0-45.1 |

Pure data movement per step without the switches (ms; step times from the profiling run):

| preset | step | device busy | pure data movement | of which next to our kernels |
|---|---|---|---|---|
| `mamba2-1p3b` | 2041 ms | 2038 ms | 16.8 ms (0.8%) | 2.7 ms: gate and segment-mask transposes, `[B, T, H]` to `[B, H, T]` (`scalar_gate.py`) |
| `gdn-hd128-1p3b` | 2057 ms | 2052 ms | 19.8 ms (1.0%) | 0.0 ms |
| `kda-1p3b` | 2525 ms | 2583 ms | 25.1 ms (1.0%) | 0.0 ms |
| `mamba3-1p3b` | 2034 ms | 2019 ms | 42.8 ms (2.1%) | 17.2 ms: per-head broadcast of the rotation angles (`ssm.py:453`, `f32[16,4096,64,32]`, forward + remat); 6.5 ms of `_deinterleave`/`angle_rows` transposes and concatenations in `mamba3_fused.py` |

The kernels do not transpose or split their operands. The scalar-gate kernels (simple GLA, the Mamba-2
SSD core, and Gated DeltaNet) and the vector-gate kernels (KDA, GDN-2) read FLA's `[B, T, H, D]` with
TMA boxes over 64 tokens and one head's `D` slice, write the output as `[B, T, H, V]` (`[B, T, H·V]` for
the output projection by a free reshape) and return `dq, dk, dv` in the same layout; head-major tiles
exist only for intermediates the kernels write and read themselves, and at `T = 4096` nothing is
padded. The layers use separate q, k and v projections, each with its own short convolution, so there
is no fused projection output to split and no gradient to concatenate. The large costs next to the
kernels were two computing fusions, which the switches remove.

## Where we lose, and limits

- **The grouped key-gradient kernel** runs one CTA per SM (about 200 KB of shared memory) with one
  warpgroup walking 64 heads, and moves data at about half of `sg_dqkwg`'s bandwidth: 95.0 ms per step
  against `sg_dqkwg`'s 93.6 ms, so the gain comes from the XLA group sums it removes (35.3 → 0.4 ms
  per step). `register_sums=True` (the sums in registers instead of shared memory) is tested but not
  timed; splitting K across two warpgroups is the next step.
- **Remaining glue** in the profiles without the switches: the Mamba-2 x-conv and `v = dt · x` fusion
  with both outputs materialized (`ssm.py:201`, 63 ms forward, 44 ms remat and 62 ms backward per step,
  about 1.2 TB/s), and Mamba-3's angle broadcast per head (17 ms) and `mamba3_fused.py` glue (142 ms,
  including 69 ms of `loop_add_convert_fusion` in remat), which per-group angles in the rotary kernel
  would remove. The short-conv fusions of the same profiles are replaced by the fused kernels of
  [short_conv.md](short_conv.md).
- The barrier's gains and those of the short-conv kernels on [short_conv.md](short_conv.md) were each
  measured against the code before that change, so they do not multiply; with the conv on its kernels
  (`conv_impl = "auto"`, the default on one H100) the barrier is redundant and costs nothing.
- `gdn-hybrid-1p3b` was not measured: its A run does not fit this harness at a 0.85 memory fraction.

## Correctness

`tests/test_zero_copy_layouts.py` checks that the outputs and every parameter gradient with the barrier
match the fused XLA path to bf16 noise (Gated DeltaNet and KDA, on the chunked ops and on the Mosaic
GPU kernels, with packed documents and grouped value heads), and
`tests/test_mosaic_gpu_scalar_gate_grouped.py` the grouped kernel against per-head `dqkwg` plus a
float32 group sum at 1e-5 (with and without packed documents, both buffering modes, split groups), the
whole op's gradients against the default path and chunked autodiff, and the Mamba-2 layer with the
switch against the default. The barrier rounds the conv output to bf16 before the norm, as `ShortConv`
returns it (the first step's loss differs by 1.7e-5 relative); the in-kernel float32 group sum is at
least as accurate as the bf16 per-head terms it replaces.

## Reproduce

```bash
# layout profiles (one GPU each; add --set to profile a switch)
python benchmarks/profile_layout_copies.py --arch gdn-hd128-1p3b --data $DATA --profile-dir $T/gdn \
    --json results/layout_gdn.json --set model.linear_attention.qk_conv_barrier=false
python benchmarks/bench_end_to_end_step.py --analyze-only <trace-dir> --json results/layout_all.json  # every kernel
# interleaved end-to-end A/B, 2 rounds of A then B on one GPU
python benchmarks/bench_e2e_ab.py --arch kda-1p3b --data $DATA --a-set model.linear_attention.qk_conv_barrier=false \
    --b-set model.linear_attention.qk_conv_barrier=true --rounds 2 --out-dir $T/ab --json results/kda_ab.json
python benchmarks/bench_e2e_ab.py --arch mamba2-1p3b --data $DATA --a-set model.ssm.grouped_key_grads=false \
    --b-set model.ssm.grouped_key_grads=true --rounds 2 --out-dir $T/ab --json results/mamba2_ab.json
```

## How it works

- **`qk_conv_barrier`.** With `qk_l2norm` (the default for the delta-rule mixers) and
  `model.remat = "full"`, XLA fused the recomputed forward of all three short convolutions into the
  q/k norm's sum-of-squares reduction: one multi-output reduction fusion that the reduction emitter
  runs far below bandwidth (241.0 ms per step in remat plus 59.7 ms of forward norm fusions for
  `gdn-hd128-1p3b`, 14.7% of device time; 263.5 and 59.7 ms for `kda-1p3b`, 12.5%). The layer puts
  `jax.lax.optimization_barrier((q, k))` between the convolutions and the op, so the convolutions stay
  loop fusions and the norm becomes a small reduction over materialized q and k. In the profiles the
  KDA step drops from 2525 to 2160 ms and the GDN step from 2057 to 1844 ms.
- **`grouped_key_grads`.** Mamba-2 hands its one group of `B, C` to the simple-GLA kernels, and 64
  value heads read it; the per-head backward `sg_dqkwg` wrote `dq, dk` per head (1 GiB each per layer)
  for XLA to sum. The switch routes the backward through
  [`scalar_gate_grouped.dqkwg_grouped`](../src/jla/ops/mosaic_gpu/scalar_gate_grouped.py), FLAT's
  one-fused-gradient idea applied to the B/C groups: one CTA per (batch, chunk, group, split) loads the
  chunk's q and k once, walks the group's heads with their `v`, `do`, states and decays double-buffered
  by TMA, and adds each head's `dq, dk` and gate gradient into float32 accumulators in shared memory.
  Only `[B, T, G·splits, N]` float32 sums leave the kernel, and peak memory falls by up to 1.7 GiB.
