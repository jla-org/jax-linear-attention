# FP8 training at 1.3B

`model.fp8 = "delayed"` runs the projections and MLPs in FP8 on Hopper: e4m3 inputs and weights, e5m2
gradients and per-tensor delayed scaling from amax histories, after MaxText's GPU recipe and Flax. This page
compares the FP8 and bf16 training steps of the 1.3B presets on one H100, lists what stays in bf16, describes
the fused FP8 paths, and gives the activation-checkpointing and XLA settings of the 1.3B TOMLs. flame's FP8
runs are compared in [end_to_end_1p3b.md](end_to_end_1p3b.md), the FP8 layers in [fp8.md](fp8.md).

## Setup

- One H100 80GB HBM3 (SXM) per run, JAX 0.11.2, bf16 compute, float32 parameters and AdamW state.
- FP8 recipe: `model.fp8 = "delayed"`, `model.fp8_min_features = 512`, `model.fp8_fused = true`; every weight
  is quantized once per optimizer step (`model.fp8_weight_once`, on by default). FP8 is off by default.
- The protocol of [end_to_end_1p3b.md](end_to_end_1p3b.md): `benchmarks/bench_e2e_1p3b.py --impl kernels`,
  FineWeb-Edu tokenized with the Llama-3.1 tokenizer, 4096-token sequences with document masking, 65,536
  tokens per step, each TOML as written (the two transformer TOMLs train without recompute at 4 × 4096 ×
  `grad_accum` 4, the recurrent presets at 16 × 4096 with full rematerialization), XLA's BFC allocator,
  20 steps, tokens/s by the wall clock from the completion of step 5 to that of step 19, one run each.
- MFU: model FLOPs (bf16-equivalent for FP8 runs) over the 989 TFLOP/s bf16 spec.
- Measured on 20 Oct 2026 with no other jobs on the GPUs (the one-GPU runs side by side, one per GPU,
  sharing the host). Rows marked provisional ran while other jobs could share the machine.

## Results

### FP8 against bf16, one H100

| preset | bf16: tokens/s (MFU) | FP8: tokens/s (MFU) | FP8 / bf16 |
|---|---|---|---|
| `transformer-1p3b` | 60,178 (56.4%) | **77,094 (72.3%)** | **1.28x** |
| `transformer-swa-1p3b` | 55,188 (50.1%) | 68,072 (61.8%) | 1.23x |
| `gdn-hd128-1p3b` | 42,995 (42.3%) | 55,956 (55.0%) | **1.30x** |
| `gdn-1p3b` (`K = 256`, `V = 512`) (a) | 37,759 (36.0%) | 46,971 (44.8%) | 1.24x |
| `kda-1p3b` | 35,110 (32.9%) | 42,756 (40.1%) | 1.22x |
| `mamba2-1p3b` | 38,899 (37.1%) | 43,688 (41.7%) | 1.12x |
| `mamba3-1p3b` | 32,976 (32.6%) | 36,344 (36.0%) | 1.10x |
| `rwkv7-1p3b` | 28,484 (26.2%) | 30,651 (28.2%) | 1.08x |

The runs peak at 41.9-54.7 GiB in bf16 and 44.5-52.0 GiB in FP8, and every loss falls from about 12.1 to
7.8-8.0 in 20 steps. (a) `gdn-1p3b` ran the same day on the code just before the release's last kernel
changes, which leave its `K = 256` kernels as they are (on one GPU they moved the other presets by
0.98-1.03x). `gdn2-1p3b` has provisional runs only ([end_to_end_1p3b.md](end_to_end_1p3b.md)); the 8-GPU FP8
runs are in [multi_gpu.md](multi_gpu.md).

### Crossing 70% MFU

At 9.28 GFLOP per token, 70% MFU is a 0.878 s step (74,600 tokens/s). `transformer-1p3b`'s TOML (with
`[model.attention] fp8_grad_epilogue = true`, `train.dispatch_ahead = true` and `train.update_guard =
"select"`) reaches 72.3%. Without the last two, in an interleaved A/B on one H100 (19 Oct, provisional;
completion-to-completion wall clock of steps 5-19), `fp8_grad_epilogue` with the merged weight-gradient operand
and the fused backward's per-parity `dS^T` barrier and key-blocks-first grid runs at 75,211-75,328 tokens/s,
**70.55-70.66% MFU** (median step 0.865-0.868 s), 1.024x the plain FP8 path without those kernel changes
(73,374-73,660 tokens/s, 68.83-69.10%; both with the persistent attention forward). About 1.3 points of that
come from the fused backward's new defaults at head dim 64, on with the option off too; the option alone is
1.011x (70.71-70.74% MFU against 69.9-70.0%), and 1.017x on `transformer-swa-1p3b`.

### The early stall and `train.gc_freeze`

`train.gc_freeze` (on by default) runs one full garbage collection and `gc.freeze()` before the step loop of
`jla train` and `jla dpo`, and `gc.unfreeze()` after it. Without it Python's first full collection after
compiling, a one-off host stall of 0.2-0.85 s here, lands in an early step that a 20-step protocol times. It
does not change throughput: over steps 7-20 the FP8 steady state of `transformer-1p3b` without the three
options above is 0.906-0.916 s per step (67.1-67.9% MFU) with and without it in all runs but one (provisional,
a busy host), 27-38 ms from a 0.878 s step; two 65.6% MFU runs that timed the stall read 67.6% and 68.0%.

## What stays in bf16

- **Small projections** (`fp8_min_features = 512`; the default, 16, converts every 16-aligned projection):
  only GEMMs whose input and output widths are both at least 512 run in FP8. Each FP8 matmul pays its input's
  cast, a transposed copy and the gradient's cast and transpose whatever its width, so 512 keeps in bf16
  exactly the 1.3B GEMMs that cost more in those passes than FP8 saves: Gated DeltaNet's `a_proj` / `b_proj`
  (16 outputs), the rank-128 gate factors of KDA (`f_proj`, `g_proj`) and GDN-2 (`f_proj`), KDA's `b_proj`,
  Mamba's `dt_proj` (64) and `bc_proj` (256), and Mamba-3's angle and scalar projections. Every attention,
  in/out, gate-output and MLP projection runs in FP8 (the transformer's k/v projections are 512 wide).
- **The LM head and cross entropy** stay bf16 by design (logit precision; [cross_entropy.md](cross_entropy.md)
  covers the opt-in FP8 head GEMMs). The mixer and attention kernels and the optimizer are unchanged by FP8.

## The recipe and the fused paths

- **Delayed scaling.** Every FP8 matmul keeps 1024-entry amax histories for its input, kernel and gradient;
  each scale is `max(history) / fp8_max`. Current scaling needs each tensor's amax before its cast, an extra
  pass: 1.5-7% slower than delayed on five presets, about even on the transformer (17 Oct, provisional).
- **Weights once per optimizer step** (`fp8_weight_once`). Before the micro-step loop the train step makes
  every delayed-scaling weight's e4m3 copy, its amax and, with the parameters on one device, a transposed
  e4m3 copy (the forward GEMM's TN operand); every micro-step, rematerialized forward and backward reads them.
  When the parameters span several devices only the e4m3 copy is kept and gathered where a GEMM uses it. An
  optimization barrier materializes each amax before the history update (`jla.layers.fp8.push_amax`).
- **One cast per shared input** (`jla.layers.fp8.shared_input_projections`): the attention q/k/v, the
  linear-attention q/k/v/output-gate and GDN-2's erase and write gate projections read one e4m3 copy of their
  common input. With the two items above and the 8-warp SwiGLU backward below: 1.06-1.07x on
  `transformer-1p3b` (62,743-63,499 tokens/s, 58.9-60.0% MFU, without the fused attention backward and the
  forward-gradient cross entropy), and on `gdn-hd128-1p3b` with FSDP over 8 GPUs 1.03-1.05x on the best steps
  and 1.074x in tokens/s (18 Oct, provisional).
- **`fp8_fused = true`** (delayed scaling only): each SwiGLU MLP whose two GEMMs are FP8 runs as one custom
  VJP (`jla.layers.fp8_swiglu`) with two Pallas Triton passes after Transformer Engine's cast-transpose and
  activation-cast kernels. The forward computes `silu(gate) * up` in float32, its amax, the e4m3 copy and its
  transpose in one pass; the backward forms the SwiGLU gradient, rounds it to bf16, records its amax and
  writes the e5m2 gradient and its transpose (8 warps). At the 1.3B shape they run at 2.5 TB/s forward and
  2.4 TB/s backward. Under a multi-device mesh they run per shard; plain-JAX passes run where no split tiles.
- **`[model.attention] fp8_grad_epilogue = true`** (off by default, on in both transformer 1.3B TOMLs): the
  Mosaic GPU attention backward kernels apply the softmax scale and the RoPE backward to each finished gradient
  tile, record the amaxes and write the e5m2 q/k/v projection gradients that the input-gradient GEMMs read,
  and (fused backward, one device) the transposed operand of the merged q/k/v weight-gradient GEMM. Kernel
  time per call, `B = 4, T = 4096`, 32 query heads over 8, `D = 64`, CUPTI median of 30 calls, the faster of
  2 rounds, 19 Oct:

| mask | plain fused backward + XLA's passes | with the FP8 epilogues | speedup |
|---|---|---|---|
| causal | 1.687 ms | 1.619 ms | 1.042x |
| 4 documents per row | 0.767 ms | 0.704 ms | 1.090x |
| sliding window 2048 | 1.473 ms | 1.469 ms | 1.003x |

## Where an FP8 step spends its time

`profile_1p3b.py` (CUDA graphs off) and `profile_fp8_passes.py`, the FP8 recipe and each TOML's split, before
`fp8_grad_epilogue` (19 Oct, provisional), ms per optimizer step:

| | `transformer-1p3b` | `gdn-hd128-1p3b` | `kda-1p3b` | `mamba2-1p3b` |
|---|---|---|---|---|
| step, profiled / GPU kernel time | 967 / 911 | 1170 / 1172 | 1609 / 1603 | 1575 / 1574 |
| GEMMs (FP8 and bf16) | 396 | 576 | 649 | 533 |
| all FP8 passes | 165.9 | 206.3 | 190.6 | 171.6 |
| · fused SwiGLU kernels (fwd / remat / bwd) | 20.9 / – / 37.3 | 21.0 / 21.3 / 38.1 | 20.5 / 20.6 / 36.8 | – |
| · activation casts and amaxes (fwd / remat / bwd) | 12.3 / – / 40.3 | 19.8 / 18.7 / 38.1 | 23.0 / 19.5 / 29.2 | 24.0 / 51.7 / 38.8 |
| · transposed copies for the weight-gradient GEMMs | 23.7 | 38.4 | 31.2 | 49.1 |
| · weight copies, scale arithmetic, amax histories | 24.2 | 10.9 | 9.6 | 8.1 |

With `fp8_grad_epilogue` the transformer step's device time is 886.9 ms: the fused attention backward with
its epilogues takes 120.9 ms and the `dq` pass 8.0 ms, and the transposed e4m3 inputs (6.8 ms) and e5m2
gradients (6.9 ms) of the other projections are the layout copies left. FP8 makes the projection and MLP
GEMMs 1.9x faster, but they are about half of a bf16 step, so FP8 is worth 1.08-1.30x end to end.

## Activation checkpointing at 1.3B

| `model.remat` | saved for the backward pass | recomputed in the backward pass |
|---|---|---|
| `full` (default) | the block inputs | every block's forward |
| `none` | everything | nothing |
| `dots` | every matmul output | elementwise ops, norms, convolutions, every kernel's forward |
| `dots_with_no_batch_dims` | projections and MLPs | as `dots`, plus batched (attention-style) matmuls |
| `save_attention` | each mixer's core output, cuDNN attention residuals | the rest |
| `kernels` | every Pallas / Mosaic GPU kernel output, each mixer's core output | projections, MLPs, norms, convolutions, XLA glue |
| `dots_kernels` | `dots` + `kernels` | elementwise ops, norms, convolutions, XLA glue |
| `names` | the checkpoint names in `model.remat_names` | everything else |
| `offload` | `remat_names` on the device, `remat_offload_names` in pinned host memory | everything else |

The checkpoint names are `attention`, `linear`, `mlp_hidden` and `kernels`; FP8 projections and MLPs carry
them too. The settings of the 1.3B TOMLs (18 Oct, provisional, interleaved pairs on one GPU against full
remat at 16 × 4096):

- **The two transformer TOMLs** train with `remat = "none"`, `model.lean_vjp = true` and unrolled layers
  (`model.scan_layers = false`) at 4 × 4096 × `grad_accum` 4 with a rolled micro-step loop
  (`train.scan_unroll = 1`). In bf16 that is 1.114x and 1.128x full remat for the transformer (53,497 tokens/s,
  50.2% MFU as written) and 1.07-1.13x for SWA (51,695-53,997 tokens/s, 46.9-49.0% MFU); on 8 GPUs (FSDP 8,
  bf16) 1.102x and 1.087x for the transformer (49,112 and 46,960 tokens/s per GPU, 46.1% and 44.0% MFU; 28.6
  GiB peak per GPU against 14.9) and 1.100x for SWA. With FP8 the transformer gains 1.01-1.02x (54.9-55.9% MFU)
  and SWA is about even (0.985-1.03x). With scanned layers the same step loses (0.89x in FP8, 0.96x in bf16):
  the scan zero-fills and copies stacked residual buffers and launches the attention forward twice per layer.
- **The recurrent presets** keep full remat at 16 × 4096. Without recompute they do not fit on one 80 GB GPU
  with preallocation off: the full-remat steps already peak at 42-56 GiB next to 22 GiB of float32
  parameters, gradients and AdamW state, and every `none` step at 4 × 4096 or more asks BFC for a 52-79 GiB
  block. With `lean_vjp`, `model.lean_remat_core = true` and one preallocated pool
  (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`) `none` at 4 × 4096 × 4 fits for Gated DeltaNet and KDA but measured
  0.98x and 0.93-0.95x of full remat (17 Oct, provisional). Across the recurrent presets no policy or
  micro-batch split that fits is more than 5% faster than full remat; plain `none` at 2 × 4096 tokens per
  micro-step compiles to 97-142 GiB for their stacks, about 3-6x flame's activation memory per token.
- **The lean VJPs** (`jla.layers.lean`) save only inputs and per-row statistics for the norms, SwiGLU, short
  convolutions, gates and Mamba epilogues: per layer and 4096-token sequence they cut the activations kept for
  the backward by 1.7-5.1x (transformer 0.77 → 0.27 GiB, Gated DeltaNet 2.10 → 0.46, KDA 2.25 → 0.59, Mamba-2
  1.47 → 0.29). With FP8 they need `fp8_fused = true`, whose SwiGLU VJP is already lean.

## XLA settings at 1.3B

`runtime.xla_tuning_level` is 0 (the default, no tuning flags), 1 (XLA's latency-hiding scheduler, set by
every 1.3B TOML) or 2 (plus cuBLASLt, while-loop double buffering and pipelined collectives, for several
GPUs: [multi_gpu.md](multi_gpu.md)). A sweep on one H100 against tier 1 (17 Oct,
provisional; bf16, full remat, scanned layers, best steps):

- **Within about 1%** (JAX 0.11.2's defaults stay): command buffers off or extended to `WHILE,CONDITIONAL`,
  programmatic dependent launch (`--xla_gpu_enable_pdl`), tier 2 on one GPU, `--xla_gpu_triton_gemm_any`,
  copy-insertion region analysis, while-loop auto unrolling, and a bf16 first moment (`optim.mu_dtype`, the
  same loss and 3.3 GiB less on KDA: a memory option).
- **Slower or not safe:** Triton GEMM fusions off (Mamba-2 2%, the transformer 1%); autotuning off
  (`--xla_gpu_autotune_level=0`, which `jla` warns about): a 31.2 s Mamba-2 step instead of 2.0 s; bf16
  parameters move the step-20 loss by 0.02 and round away AdamW updates smaller than half a bf16 step.
- **Unrolled layers** (`model.scan_layers = false`): KDA 5% faster (2.24 s per step instead of 2.35 s) with
  7 GiB less memory and a 5.3 min compile, the transformer 1.3% faster with 4.7 GiB less (78 s compile), Mamba-2
  3% slower. The transformer TOMLs unroll (above); the other TOMLs keep the scan.

## Where we lose, and limits

- FP8 gains least on Mamba-2, Mamba-3 and RWKV-7 (1.08-1.12x); the Mamba presets have 48 layers without an MLP.
- The FP8 passes left are memory-bound (165.9 ms per transformer step above). cuBLASLt FP8 GEMMs are TN only,
  so the weight-gradient GEMMs need transposed copies; a standalone cast-transpose re-reads the bf16 tensor and
  costs more than it saves. The delayed scales are recomputed per micro-step (8,736 tiny kernels per step).
- FP8 lengthens the first compile: the FP8 `transformer-1p3b` steps compiled in 137-153 s, 163-165 s with
  `fp8_grad_epilogue`; at full remat 54-71 s instead of 12-23 s for the attention and delta-rule presets and
  180-270 s instead of 10-14 s for the Mamba presets (17 Oct).
- The `fp8_fused` passes are Pallas Triton kernels, which JAX 0.11.2 marks deprecated (a warning per compile).
- On a GPU a step with the weight copies is not bitwise the per-call step: XLA may pick other GEMM algorithms
  for the new layouts, and e5m2 gradients amplify the last-bit differences in the warm-up spike.
- `fp8_grad_epilogue` covers the plain attention layer with delayed-scaling FP8 q/k/v projections that carry
  the step's weight copies, no biases, QK-norm, sinks, MoBA or context parallelism, plain full RoPE, the Mosaic
  GPU kernel, head dim 64 or 128 (64 with the fused backward) and `T` a multiple of 128; other calls take the
  plain path (logged once per reason). The merged transposed operand needs the fused backward and no mesh axis
  with more than one device.
- `fp8_weight_once` and the shared input cast work in steps built by `jla.training.step.build_train_step`
  (the trainer and these benchmarks); a model called directly quantizes per call. `train.gc_freeze` covers
  only what compiles before the loop, and `remat = "offload"` is tested for correctness but not timed.

## Exactness and tests

The weight copies and the shared input cast are bitwise the per-call FP8 path on CPU through whole train
steps, also with FSDP and pipeline parallelism on 4 CPU devices (`tests/test_fp8_weight_once.py`); the fused
SwiGLU path is bitwise the unfused one on an H100 for `y`, `dx` and the down-projection weight gradient, the
gate/up weight gradient within GEMM accumulation order (`tests/test_fp8_swiglu.py`); the attention epilogues
are bitwise the plain layer with the XLA reference backend on CPU, and on an H100 0.004-0.012% of the e5m2 `q`
and `k` values (none of `v`) differ from XLA's RoPE backward (`tests/test_fp8_attention.py`). Also:
`tests/test_fp8_min_features.py`, `test_fp8_training.py`, `test_fp8_multidevice.py`, `test_fp8_remat_names.py`,
`test_lean_fp8.py`, `test_lean_vjp.py`, `test_remat_policies.py` and `test_trainer_gc_freeze.py`.

## Reproduce

```bash
# one run, the TOML as written (the bf16 rows leave out the three --set model.fp8* overrides)
python benchmarks/bench_e2e_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --dp 1 --fsdp 1 --steps 20 --skip 5 --allocator bfc \
    --set model.fp8=delayed --set model.fp8_min_features=512 --set model.fp8_fused=true --json results/fp8_1p3b.json
# interleaved A/B of one option, A B A B in fresh processes on one GPU
python benchmarks/bench_e2e_ab.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B --micro-batch 4 --rounds 2 \
    --set model.fp8=delayed --set model.fp8_min_features=512 --set model.fp8_fused=true \
    --b-set model.attention.fp8_grad_epilogue=true --json results/fp8_ab.json
# every FP8 pass of one step (CUDA graphs off)
python benchmarks/profile_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --micro-batch 4 --grad-accum 4 --set model.fp8=delayed --set model.fp8_min_features=512 \
    --set model.fp8_fused=true --profile-dir traces/fp8 --json results/profile_fp8.json
python benchmarks/profile_fp8_passes.py traces/fp8 --json results/fp8_passes.json
# the attention-gradient epilogue kernels
python benchmarks/bench_attention_fp8_bwd.py --breakdown --json results/attention_fp8_bwd.json
# recompute policies (<dtype>:<remat>[=<names>]:<micro batch>x<grad accum>[:<flags>]) and one XLA flag
python benchmarks/bench_remat_fp8_1p3b.py --arch transformer-1p3b --data /path/to/fineweb_edu_3B \
    --variant bf16:full:16x1 --variant bf16:none:4x4:lean+noscan --json results/remat_1p3b.json
python benchmarks/bench_e2e_1p3b.py --arch mamba2-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --micro-batch 16 --set 'runtime.xla_flags=["--xla_gpu_enable_command_buffer="]' --json results/xla_flags.json
# what the backward keeps per block under remat = "none", with and without the lean VJPs (one --arch per preset)
python benchmarks/lean_residuals.py --arch kda-1p3b --arch mamba2-1p3b --lean --json results/residuals.json
```

## How it works

With delayed scales a cast never waits for a reduction over its own tensor, so XLA fuses each tensor's amax,
scale and cast into one pass. The fused paths add what XLA does not fuse: transposed FP8 copies, the SwiGLU's
amax with its cast, and the gradient casts with the backward kernels that produce them.
