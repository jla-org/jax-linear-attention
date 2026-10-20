# Multi-GPU training: data parallelism and FSDP on 8 H100s

The trainer's data-parallel (`dp`), FSDP (`fsdp`) and tensor-parallel (`tp`) meshes on eight H100s: the 1.3B
presets at FSDP 8 against flame's FSDP2, the throughput per GPU against one GPU, the agreement of every layout
with one GPU, checkpoints resumed under another mesh, multi-process runs, and the train options for several
GPUs. Pipeline and expert parallelism have their own pages: [pipeline.md](pipeline.md),
[expert_parallel.md](expert_parallel.md).

## Setup

- One 8x H100 80GB HBM3 machine (NVLink), JAX 0.11.2, NCCL through XLA, preallocation off
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), the Mosaic GPU kernels on (`--impl kernels`), each running per
  shard of the mesh ([How it works](#how-it-works)).
- 1.3B: `benchmarks/bench_e2e_1p3b.py` on FineWeb-Edu, 4096-token sequences with document masking,
  16 x 4096 tokens per GPU per step (65,536 on one GPU, 524,288 on eight), 20 steps, each TOML of
  `benchmarks/end_to_end/1p3b/` as written. FP8 adds `model.fp8 = "delayed"`, `fp8_min_features = 512` and
  `fp8_fused = true`.
- Baseline: flame `24dea91` with FLA main `6dff3e1` and torchtitan `0b44d4c` (FLA main `c6e4f00` in the
  [every-preset table](#every-13b-preset-on-8-gpus)), FSDP2 over the eight GPUs in its strongest mode that
  runs: `torch.compile`, no activation checkpointing, 4 micro-steps of 4 records per GPU, the transformer's
  attention through FlashAttention-3. Same records in the same order, documents masked through `cu_seqlens`
  (except Mamba-2: FLA's Mamba-2 layer takes no `cu_seqlens`). flame's FP8 setup is under "The flame baseline" in
  [end_to_end_1p3b.md](end_to_end_1p3b.md).
- Timing: tokens/s per GPU by the wall clock from the completion of step 5 to that of step 19, one run per
  configuration. MFU is jla's analytic model FLOPs over the 989 TFLOP/s dense bf16 peak for every row, FP8
  included; flame's rows use the same FLOPs.
- Measured on 20 Oct 2026 with no other jobs on the GPUs. The other tables are
  [provisional](../README.md#status-and-limitations) (17-19 Oct 2026, other jobs' processes could share the GPUs).

## Results

### 1.3B at FSDP 8 against flame

FP8, FSDP 8, each TOML as written (ratio = flame's time / ours, so > 1 means ours is faster):

| preset | ours: tokens/s per GPU (MFU) | per GPU vs ours on 1 GPU | flame FP8: tokens/s per GPU | ours / flame | allocator |
|---|---|---|---|---|---|
| `transformer-1p3b` | 72,709 (68.2%) | 0.95x | 49,475 | **1.47x** | CUDA async |
| `gdn-hd128-1p3b` | 53,208 (52.3%) | 0.95x | 33,997 | **1.57x** | BFC (XLA's default) |

Both rows ran on the code just before the last kernel changes of 0.1.0, and "per GPU vs ours on 1 GPU"
divides by that code's one-GPU rate. flame's FP8 is torchtitan's float8 converter; its float8 all-gather for FSDP
failed to run, so flame gathers its weights in bf16. The `gdn-hd128-1p3b` row is timed by the harness's mean step over steps
6-20. With the CUDA async allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) `gdn-hd128-1p3b` runs 1.80x
slower on 8 GPUs (29,520 tokens/s per GPU, 29.0% MFU, 0.87x flame).

### Every 1.3B preset on 8 GPUs

Provisional (18 Oct 2026): each 8-GPU batch was the only timed job on the eight GPUs, but other jobs' test
processes held 0.9-2.8 GiB on one to four GPUs before some runs. Every preset at full rematerialization and
16 x 4096 per GPU (the transformer TOMLs train without recompute), BFC, XLA tuning tier 1 (tier 2 for
`mamba2-1p3b`), none of the [train options](#train-options-for-several-gpus), the harness's mean over steps
6-20; the faster of fsdp 8 and dp 2 x fsdp 4; flame in bf16. The FP8 rows of `transformer-1p3b` and
`gdn-hd128-1p3b` are the table above.

| preset | precision | layout | tokens/s per GPU (MFU) | per GPU vs 1 GPU (1-GPU tokens/s) | flame bf16, 8 GPUs | vs flame |
|---|---|---|---|---|---|---|
| `transformer-1p3b` | bf16 | fsdp 8 | 44,965 (42.2%) | 0.96x (46,641) | 47,117 | 0.95x |
| `gdn-hd128-1p3b` | bf16 | fsdp 8 | 39,030 (38.4%) | 0.96x (40,860) | 32,135 | 1.21x |
| `gdn-1p3b` | bf16 | fsdp 8 | 34,879 (33.3%) | 0.93x (37,368) | 34,210 | 1.02x |
| `kda-1p3b` | bf16 | dp 2 x fsdp 4 | 32,225 (30.2%) | 0.94x (34,411) | 21,729 | 1.48x |
| `kda-1p3b` | FP8 | fsdp 8 | 39,017 (36.6%) | 0.96x (40,462) | 21,729 | 1.80x |
| `gdn2-1p3b` | bf16 | fsdp 8 | 28,502 (31.7%) | 0.96x (29,762) | - (a) | - |
| `gdn2-1p3b` | FP8 | fsdp 8 | 34,474 (38.3%) | 0.96x (35,964) | - (a) | - |
| `mamba2-1p3b` | bf16 | fsdp 8 | 32,423 (30.9%) | 0.91x (35,669) | 31,740 (b) | 1.02x (b) |
| `mamba3-1p3b` | bf16 | fsdp 8 | 30,358 (30.0%) | 0.93x (32,582) | - (c) | - |
| `mamba3-1p3b` | FP8 | fsdp 8 | 34,421 (34.1%) | 0.95x (36,365) | - (c) | - |

(a) GDN-2 is clean-room (no upstream code), so it has no flame run. (b) flame's Mamba-2 trains without
document masking, so it does less work than ours. (c) FLA main's Mamba-3 fails before its first step under
FSDP2 (its `_init_weights` copies a plain tensor into the DTensor `dt_bias`); `bench_e2e_flame.py` patches
that (`_patch_fla_mamba3_dtensor_init`, recorded as `fla_patches`), but no patched 8-GPU run was measured.

8 GPUs keep 0.91-0.96x of one GPU's tokens/s per GPU in every row. dp 2 x fsdp 4 was 0.7-7.5% slower than
fsdp 8 except for KDA in bf16 (+0.9%; at fsdp 8 31,953 tokens/s per GPU, 29.9%, 0.93x) and the transformer in
FP8 at full rematerialization (+2.3%). The Mamba-2 row ran its unfused output glue, which also runs per shard
in the shipped code (the Mamba-2 section of [end_to_end_1p3b.md](end_to_end_1p3b.md)); RWKV-7 has no 8-GPU run.

### Train options for several GPUs

The `transformer-1p3b` TOML turns on the first three options (off by default) and pins the Mosaic GPU
attention, which `train.grad_reduce = "step"` needs on GPUs. Provisional (17-19 Oct 2026): FSDP 8, FP8 (bf16 for
the XLA tier), A/B runs on the same eight GPUs, with other jobs' processes possible beside and between the runs.

| setting | what it does | measured on 8 GPUs |
|---|---|---|
| `train.grad_reduce = "step"` | each GPU runs its micro-steps as one GPU does (the whole weights gathered once per step, the FP8 copies made once, no collective inside the micro-step loop), then one gradient reduce-scatter per step | `transformer-1p3b` 1.060x and 1.064x: 72,751-73,104 against 68,626-68,695 tokens/s per GPU (68.2-68.6% against 64.4% MFU), 38.3 against 34.4 GiB per GPU at peak |
| `train.update_guard = "select"` | the rejected-step rule (non-finite loss or gradient) as an elementwise select instead of a `lax.cond`, whose predicate, the gradient norm, made the host wait for every step | with `"step"`: 935 against 984 ms per step in one pair (that arm also ran the CUDA async allocator); on 2 GPUs 1.061x for the plain step; on one GPU no change (881 against 877 ms) |
| `train.dispatch_ahead = true` | dispatches the next step before reading a log step's metrics | alone: 0.992x and 1.004x (1.017x on one GPU) |
| all three, against none | | `transformer-1p3b` 1.154x (66.8-66.9% against 56.9-58.9% MFU per GPU, one GPU 74,753 tokens/s and 70.1%; 38.2 against 34.6 GiB), measured before the persistent attention forward and the FP8 gradient epilogue; `gdn-hd128-1p3b`, whose TOML leaves them off, 1.015x (53.5% against 52.7%, one GPU 56,007 tokens/s and 55.1%; 28.4 against 19.9 GiB) |
| persistent attention forward (the default) | one CTA per SM in the attention forward | 1.020x and 0.998x against one CTA per tile, so it stays on (`[model.attention] persistent_forward = false` turns it off) |
| `[model.attention] fp8_grad_epilogue = true` | the attention backward writes the FP8 q/k/v projection gradients | 1.038x and 1.013x |
| `train.ce_grad_in_forward = true` | the LM-head gradient formed in the forward pass | 1.114x for `transformer-1p3b` and 1.076x for `gdn-hd128-1p3b`, whose TOMLs set it; 0.954x for `kda-1p3b` in its one pair (1,876.7 against 1,790.3 ms per step, the arm without it spread over 1,666-1,971 ms), whose TOML keeps it off |
| bf16 LM-head gather (always on) | the per-shard cross entropy gathers the LM head in bf16 and reduce-scatters its float32 gradient (exact) | 1.84 GB less traffic per GPU and step for `transformer-1p3b`, 0.46 GB for `gdn-hd128-1p3b` |
| `runtime.xla_tuning_level` | tier 1, set by every 1.3B TOML: XLA's latency-hiding scheduler; tier 2 adds pipelined all-gather, reduce-scatter and all-reduce, while-loop double buffering and cuBLASLt | tier 2 at fsdp 8, back to back: +9% for `mamba2-1p3b` (use tier 2 for it on several GPUs), -4.9% to +2.2% for the other presets, 0.9-3.9 GiB more peak memory per GPU; it does not compile the TOML's `transformer-1p3b` step (unrolled layers, 4 micro-steps) within 5 minutes |

### Every layout against one GPU (125M to 1B)

`benchmarks/bench_multi_gpu.py` (17 Oct 2026): seven models (`transformer-125m` with a 256-token window,
`gdn-hybrid-125m`, `kda-hybrid-125m`, `gdn2-hybrid-125m`, `mamba2-130m`, `mamba3-130m`, and `transformer-1b`
with a 512-token window), the FA3-style Mosaic GPU attention in every attention layer, synthetic tokens,
1024-token rows, the same global batch of 16 rows on every layout, AdamW, bf16 compute with fp32 parameters,
`remat = "full"`, 16 steps. Layouts (not every model ran every one): one GPU, dp 2/4/8, fsdp 2/4/8, tp 2,
dp 2 x tp 2, dp 4 x tp 2, 4 and 8 `jax.distributed` processes with one GPU each, and resumes of an fsdp 8
checkpoint. The noise reference is one GPU with two micro-steps of 8 rows (the same batch summed in another
order).

- **Loss.** At step 1 every layout matches one GPU to at most 2.9e-5 relative in the loss (1.6e-5 except
  `transformer-1b` under dp 4 x tp 2) and 5.2e-4 in the gradient norm (`kda-hybrid-125m` at fsdp 4). Over the
  16 steps the transformer stays within 3e-5 relative on every layout and Mamba-2 within 1.3e-4; the hybrids,
  Mamba-3 and the 1B model drift by 3e-4 to 4.6e-3, 1.5x to 3.4x their own noise reference (2.8e-4 to 2.7e-3).
- **Resume.** The fsdp 8 run's step-8 checkpoint, resumed under dp 8, fsdp 4 x dp 2 and on one GPU (the
  restore places every array straight into the new shardings): the first loss after the restore is bitwise
  equal to the uninterrupted run in 10 of the 20 resumes and within 7.5e-5 in the others, the gradient norm
  within 2e-4 relative except `kda-hybrid-125m` on one GPU (1.2e-3).
- **Processes.** 4 and 8 processes with one GPU each (`gdn-hybrid-125m`, fsdp 4 and fsdp 8): every worker
  logs the same losses; against one process on the same GPUs the step-1 loss differs by 0 and 3.0e-5 and the
  gradient norm by 1.8e-6 and 4.7e-5 relative; the checkpoint is the multi-process format
  (`payload_format = "orbax"`).
- **1.3B.** fsdp 8 against one GPU with 8 micro-steps of 16 records (the same 128 records per step): the
  losses of six presets agree within 0.001-0.027 over six steps.
- Loss trajectories cannot show a small gradient error (the step-1 loss is a forward pass, and by step 16
  the noise reference's gradient norms differ from one GPU by up to 47%); the evidence for exact gradients is
  the per-shard kernel tests under [Correctness](#correctness).

Throughput at 125M is provisional (two runs of one configuration differed by up to 30%). At 16 rows per step,
dp 8 or fsdp 8 gave 1.9-3.4x one GPU for the 125M models and 3.5-4.1x for the 1B model, and dp 4 x tp 2
1.7-2.8x; 2 rows per GPU leave a 125M model at 4-8% MFU. With 16 rows per GPU (global batch 128, fsdp 8):

| model | 1 GPU, 16 rows: tokens/s | fsdp 8, 16 rows per GPU: tokens/s | per GPU vs 1 GPU | MFU |
|---|---|---|---|---|
| `gdn-hybrid-125m` | 256k | 1,755k | 0.86 | 18.2% |
| `transformer-125m`, 256-token window | 322k | 2,201k | 0.86 | 20.9% |

## Where we lose, and limits

- **Per GPU against one GPU: 0.95x** for `transformer-1p3b` and `gdn-hd128-1p3b`, 0.91-0.96x for every preset
  in the provisional table. In a profiled `transformer-1p3b` step with the three options (provisional), the
  two float32 collectives of `"step"` leave 57 ms exposed (the end-of-step reduce-scatter 43.2 ms, the
  start-of-step all-gather 13.7 ms) and every GPU waits 30 ms at each step boundary (not isolated).
- **flame in bf16:** `transformer-1p3b` at full rematerialization is 0.95x flame on 8 GPUs (flame's compiled
  transformer was already faster on one GPU, 52,351 against 46,641 tokens/s in those runs); `mamba2-1p3b` is
  1.02x against a flame run without document masking; `gdn-1p3b` is 1.02x (FlashQLA declines K = 256, and
  flame runs FLA's Triton kernels with TileLang's `chunk_bwd_dqkwg`). No flame Mamba-3 number on 8 GPUs.
- **`gdn-hd128-1p3b`'s layer scan** holds 72 tiny all-reduces per step (short-conv weight-gradient sums and
  FP8 amax maxima) that act as barriers; XLA's pipelined collectives alone did not help (0.995x at FSDP 4).
- **The CUDA async allocator** runs `gdn-hd128-1p3b` 1.80x slower on 8 GPUs; keep XLA's default BFC.
- **`train.grad_reduce = "step"`** holds about 10 bytes per parameter on every GPU whatever the FSDP size
  (about 30 GB at 3B; 7B does not fit on an 80 GB GPU), so it is for models up to about 3B. It refuses MoE
  layers, MTP heads, dropout, FP8 current scaling, the DSA indexer loss, Mamba-1 layers, tensor and pipeline
  parallelism, and cuDNN or `"auto"` attention on GPU.
- **Small models at small batches** scale poorly (above). Tensor parallelism costs more than it saves at
  d = 768: tp 2 splits matmuls that are already small and adds two activation all-reduces per layer (538 MiB
  of bf16 all-reduces in the compiled `gdn-hybrid-125m` dp 2 x tp 2 step); it pays only for models that do not
  fit on one GPU.
- **One machine only:** the `dcn_*` mesh axes, multi-node `jax.distributed` and InfiniBand are untested, and
  multi-process runs were measured for one 125M model. On hosts that refuse TCP to 127.0.0.1 and to IPv4 ports
  below the ephemeral range, give the coordinator a port in the ephemeral range;
  `bench_multi_gpu.py multiprocess --coordinator auto` (the default) probes one.
- **Gathered inputs** (exact, no speedup) under a multi-device mesh: the dispatcher kinds outside
  `TOKEN_LAYOUT_KINDS` (Mamba-1, RWKV-6, Gated DeltaProduct, log-linear), and the fused Mamba-3 core when the
  batch has fewer rows than the mesh has devices (its heads do not split, since its B/C groups do not split
  with them).
- **Resumed runs** can peak higher than uninterrupted ones (+0.4-0.8 GiB for the 125M models under dp 8,
  +6.3 GiB for `transformer-1b` on one GPU, whose dp 8 resume ran out of memory at a 0.40 memory fraction).
- The trainer rejects `linear_attention.context_parallel_axis` and `attention.context_parallel_axis`; kernel
  splits over the `ep` and `pp` axes are tested on CPU with an abstract mesh only.

## Correctness

The per-shard kernels match one GPU bitwise: on 4 and 8 H100s the attention output and its three gradients,
and the fused Mamba-3 output and per-row gradients, with only the replicated parameters' gradients (summed
over the devices in another order) within 1.7e-7 of their largest value (`tests/test_kernel_sharding.py`); on
2 H100s the short conv, the conv + q/k L2 norm, the FP8 SwiGLU passes and RWKV-7's bounded mode per row and
channel, weight-gradient sums within 2.2e-7, and the fused SwiGLU GEMMs to bf16 noise
(`tests/test_kernel_sharding_rules.py`), with no all-gather of a kernel's activations in the compiled steps.
The 2-GPU Gated DeltaNet test is bf16-close, not bitwise: that kernel splits a sequence into a number of pieces
that depends on the per-device `B * H` (36 of 1M outputs differed by one bf16 ulp). `train.grad_reduce =
"step"` is one GPU's step to float32 noise on two H100s (the loss to 1e-7, the gradients bit for bit except the
LM head's, 1.9e-6, the FP8 states bit for bit; `tests/test_grad_reduce_gpu.py`, `tests/test_grad_reduce.py`).
Also `tests/test_update_guard.py`, `tests/test_activation_sharding.py`, `tests/test_cudnn_fsdp_pin.py`,
`tests/test_distributed_multiprocess.py` and `tests/test_dispatch_ahead_multiprocess.py`. The tests with
several GPUs run by hand: the one-GPU test runner never sees two GPUs.

## Reproduce

```bash
FP8="--set model.fp8=delayed --set model.fp8_min_features=512 --set model.fp8_fused=true"
# 1.3B, FSDP 8, the TOML as written (one GPU: CUDA_VISIBLE_DEVICES=0 ... --fsdp 1; dp 2 x fsdp 4: --dp 2 --fsdp 4)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
  python benchmarks/bench_e2e_1p3b.py --data /path/to/fineweb_edu_3B --arch transformer-1p3b --impl kernels \
  --dp 1 --fsdp 8 --steps 20 --skip 5 $FP8 --output-dir runs/transformer-1p3b-fsdp8 --json results/multi_gpu_1p3b.json
# flame, FSDP2 over 8 GPUs (transformers: add --fa3-shim)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmarks/bench_e2e_flame.py --data /path/to/fineweb_edu_3B \
  --nproc 8 --flame-src /path/to/flame --arch gdn-hd128-1p3b --micro-batch 16 --grad-accum 4 --ac none \
  --steps 20 --skip 5 --json results/flame_8gpu.json
# How it works: where a profiled FSDP step's time goes per GPU (the two profiled runs it reads are in the script's
# header), and the host timeline of the trainer loop
python benchmarks/fsdp_trace_1p3b.py --profile RUN8/profile --result run8.json \
  --single-profile RUN1/profile --single-result run1.json --json results/fsdp_trace.json
python benchmarks/trainer_loop_probe.py --data /path/to/fineweb_edu_3B --arch transformer-1p3b --impl kernels \
  --dp 1 --fsdp 8 --steps 20 --skip 5 $FP8
# 125M to 1B layouts, one per process (1gpu, dp8, fsdp8, dp4-tp2, ...); a resume; 8 processes; the tables (CPU)
python benchmarks/bench_multi_gpu.py run --preset gdn-hybrid-125m --attention-impl mosaic_gpu --steps 16 \
  --seq-len 1024 --global-batch 16 --record-kernels --layout fsdp8 --save-at 8 --out runs/mgpu/gdn/fsdp8.run \
  --json results/multi_gpu/gdn/fsdp8.json
python benchmarks/bench_multi_gpu.py run ... --layout dp8 --resume-from runs/mgpu/gdn/fsdp8.run --resume-step 8 \
  --json results/multi_gpu/gdn/resume-dp8.json
python benchmarks/bench_multi_gpu.py multiprocess ... --layout fsdp8 --num-processes 8 --coordinator auto \
  --json results/multi_gpu/gdn/mp8.json
python benchmarks/bench_multi_gpu.py compare --dir results/multi_gpu --json results/multi_gpu_summary.json
# kernel tests on every visible GPU, with nothing else on them
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 pytest tests/test_kernel_sharding.py -k gpus -s
CUDA_VISIBLE_DEVICES=0,1 pytest -q tests/test_kernel_sharding_rules.py tests/test_grad_reduce_gpu.py -s
```

## How it works

XLA's SPMD partitioner cannot split a Mosaic GPU or Pallas custom call: under a multi-device mesh it would
all-gather the operands and run the whole kernel on every device. So every kernel runs inside a `shard_map`
over the trainer's mesh (`jla.distributed.kernel_sharding`: `shard_kernel_call` for the dispatcher's
linear-attention and SSM kernels, `shard_layout_call` for the attention kernel, the fused Mamba-3 core, the
short convs, the FP8 SwiGLU passes and the fused SwiGLU GEMMs). The batch splits over the data axes, heads,
channels or `d_ff` over `tp` where valid, and every other mesh axis larger than 1 splits the batch too, so no
device repeats another's work; the kernel's own constraints are checked on the per-device shapes, and the
plain-JAX path or the unsplit call runs when no split fits. Activations are pinned to the batch split
(`jla.distributed.activation_sharding.constrain_batch`), and FSDP gathers bf16 weights (XLA moves the cast in
front of the all-gather) and reduce-scatters the gradients. With the three train options off (the library
default), a profiled 8-GPU `transformer-1p3b` FP8 step ran at 0.78x of one GPU per GPU (52.8% against 67.7%
MFU, provisional): the GPUs idled 134 ms longer per step than on one GPU while the host dispatched the next
8-device step, and 115 ms of communication was exposed, mostly the float32 gradient reduce-scatter that runs
once per micro-step. `update_guard`, `dispatch_ahead` and `grad_reduce` target those two costs.
