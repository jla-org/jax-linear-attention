# Where a 1.3B training step spends its time

This page attributes the GPU time of one 1.3B training step to what produced it (GEMMs, recurrent
kernels, softmax attention, the remaining XLA glue and the recomputed forward), for jla and for flame
(FLA's trainer) in the flame mode whose recomputation matches ours. It also gives the measured H100
ceilings (the sustained clock and tensor-core rate under the power cap, GEMM and HBM rates) with a
speed-of-light model of the step, and the profile of a 125M step.

## Setup

- One H100 80GB HBM3 (SXM) per run, JAX 0.11.2 (torch 2.13.0+cu129 for flame), bf16 compute, float32
  parameters and AdamW state.
- Both trainers: 16 × 4096 tokens per step (65,536), FineWeb-Edu packed records with document masking
  (except flame's Mamba-2: FLA's Mamba-2 layer ignores `cu_seqlens`), full activation recomputation (ours
  `model.remat = "full"`, flame `--activation_checkpoint.mode full`), AdamW with gradient clipping, a 128,256-token
  vocabulary with an untied head.
- jla: [`profile_1p3b.py`](profile_1p3b.py) builds the step of [`bench_e2e_1p3b.py`](bench_e2e_1p3b.py)
  `--impl kernels` (every mixer on its Mosaic GPU kernel), times 6 steps (median) after 2 warm-up steps on the
  first batch of the harness's loader and records 2 steps with `jax.profiler`. XLA command buffers are off so
  that every kernel, cuBLAS included, carries its HLO instruction; category, source line and phase (forward,
  recomputed forward, backward) come from the HLO metadata. Idle = wall time minus the union of kernel
  intervals. The release harness with command buffers on gave KDA and Mamba-2 steps within about 1% of these.
- flame: fla-org/flame `24dea91` with FLA main `c6e4f00` and torchtitan `0b44d4c`, FLA's default backend
  dispatch (TileLang for the GDN and KDA backward kernels; FlashQLA and FlashKDA reject these calls), the
  official `mamba_ssm` Triton kernels and `causal_conv1d` for Mamba-2, FA3 (flash-attention `965c921`) for
  attention; profiled by [`profile_1p3b_flame.py`](profile_1p3b_flame.py) in flame's full activation
  checkpointing mode without `torch.compile`, with every `nn.Linear` call labelled by its module path. flame's
  compiled mode is not profiled: its Inductor kernels carry no module labels.
- Measured on 17 Oct 2026, **provisional**: each run was the only timed job on its GPU while other jobs used
  the other GPUs. The profiled code runs every preset in bf16 at full remat, without the release's conv
  barrier, fused short-convolution kernels, fused attention backward, forward-gradient cross entropy, FP8 and
  no-recompute transformer TOMLs; the release step times are in [end_to_end_1p3b.md](end_to_end_1p3b.md) and
  an FP8 step profile in [fp8_1p3b.md](fp8_1p3b.md).

## Results

### One step by category, ours and flame

Provisional (17 Oct), ms per step:

| architecture | trainer | ms/step | tok/s | idle | GEMMs | recurrent kernels | softmax attention | everything else (XLA fusions / aten + Triton glue) | of which recomputed forward |
|---|---|---|---|---|---|---|---|---|---|
| transformer-1p3b | ours | 1978 | 33,137 | 25 | 873 (44%) | – | **825 (42%)** | 254 (13%) | 276 (14%) |
| transformer-1p3b | flame | 2075 | 32,255 | 268 | 905 (44%) | – | 500 (24%) | 402 (19%) | 381 (18%) |
| transformer-swa-1p3b | ours | 1716 | 38,195 | 15 | 869 (51%) | – | 578 (34%) | 253 (15%) | 268 (16%) |
| gdn-hd128-1p3b | ours | 2056 | 31,879 | 2 | 1096 (53%) | 114 (6%) | – | **843 (41%)** | 503 (24%) |
| gdn-hd128-1p3b | flame | 2142 | 29,681 | 100 | 1102 (51%) | 200 (9%) | – | 739 (35%) | 545 (25%) |
| gdn2-1p3b | ours | 2664 | 24,598 | 3 | 1208 (45%) | 529 (20%) | – | **924 (35%)** | 626 (24%) |
| mamba2-1p3b | ours | 2043 | 32,078 | 6 | 993 (49%) | 290 (14%) | – | 753 (37%) | 299 (15%) |
| mamba2-1p3b | flame (no doc masking) | 2206 | 28,020 | 23 | 990 (45%) | 663 (30%) | – | 530 (24%) | 565 (26%) |
| mamba3-1p3b | ours | 2048 | 31,993 | 17 | 980 (48%) | 632 (31%) | – | 419 (20%) | 425 (21%) |
| kda-1p3b | ours | 2382 | 27,514 | 20 | 999 (42%) | 486 (20%) | – | **877 (37%)** | 556 (23%) |
| kda-1p3b | flame | 3138 | 20,307 | 130 | 1648 (53%) | 480 (15%) | – | 881 (28%) | 598 (19%) |
| gdn-hybrid-1p3b | ours | 1788 | 36,650 | 20 | 1026 (57%) | 86 (5%) | 155 (9%) | 501 (28%) | 267 (15%) |
| kda-hybrid-1p3b | ours | 2184 | 30,008 | 30 | 979 (45%) | 365 (17%) | 147 (7%) | 663 (30%) | 381 (17%) |

ms per step; the recomputed forward overlaps the other columns. flame's tok/s are over its timed steps,
profiler window included; its ms/step excludes the window, and its transformer row's 268 ms of idle time is
an upper bound.

- **GEMMs are 42-57% of every step** in both trainers, at 700-830 TFLOP/s per projection GEMM: with
  everything but the GEMMs free, a step would be only 1.74x (gdn-hybrid) to 2.39x (KDA) faster.
- **The recomputed forward is 14-25% of the step** in both trainers (ours 267-626 ms; its GEMMs 127-214 ms).
- **The recurrent kernels are 5-20% of the jla step** (Mamba-3: 31%) and beat FLA's inside it: 114 against 200 ms
  for GDN-128 and 290 against the official kernels' 663 ms for Mamba-2. KDA's are even (486 against 480 ms;
  `vg_bwd_intra` alone takes 191 ms).
- **Softmax attention is 42% of the jla transformer step**: 825 ms (forward 106, recomputed 105, backward 614)
  against FA3's 500 ms in flame (36, 33, 431). FineWeb-Edu's short documents leave most causal blocks fully
  masked; FA3's varlen path never visits them, while the profiled kernel visits every causal block.
- **XLA glue that flame does not have**: in GDN-128, GDN-2 and KDA one recomputed-forward fusion of the three
  short convs, their SiLU and the q/k L2 norm takes 239-241 ms per step (10 ms per layer, at 160 GB/s); the
  short-conv backward reductions take 153-216 ms, against 117-137 ms for flame's `causal_conv1d` forward,
  recomputed forward and backward together; the LM head and cross entropy take 245-255 ms against flame's
  153-163 ms (the chunked cross entropy recomputes the logits in the backward); Mamba-2's `dt_proj` fusion
  takes 76 ms and its mixer glue 177 ms. flame has its own: 166-447 ms of aten elementwise ops and layout
  copies and 23-268 ms of idle time per step, against 2-30 ms of idle time for ours.
- **flame's KDA step loses 742 ms in its LM head**: FLA main's KDA model runs `FusedLinearCrossEntropyFunction`
  by default, whose weight-gradient `addmm` runs at 52 TFLOP/s (134-144 ms in flame's other models). That is
  most of the KDA difference in this table.

### Ceilings on H100

The bf16 spec's 989.4 TFLOP/s is 4096 dense FLOPs per clock per SM × 132 SMs × 1,830 MHz. With 8192^3 GEMMs
back to back for 30-60 s every GPU sat at its 700 W limit, the software power cap active in every
`nvidia-smi` sample (clocks, power limit and application clocks at their defaults):

| GEMM loop | sustained TFLOP/s | SM clock, median | power, median |
|---|---|---|---|
| bf16 8192^3, `torch.mm` (cuBLASLt), 30 s | 678.2 (68.5% of 989.4) | 1,365 MHz | 698.2 W |
| bf16 8192^3, XLA, 60 s, a second GPU | 681.7 (68.9%) | 1,350 MHz | 698.1 W |
| the same, a third GPU | 681.2 (68.8%) | 1,350 MHz | 697.7 W |
| the same, a fourth GPU | 692.4 (70.0%) | 1,365 MHz | 698.0 W |
| FP8 8192^3, `torch._scaled_mm`, 30 s | 1,235.9 (124.9% of 989.4; 62.5% of 1,978.9) | 1,260 MHz | 698.2 W |

- A power-capped H100 sustains **678-692 TFLOP/s in bf16** (median 681, 68.5-70.0% of the spec; 92-94% of the
  tensor-core peak at that clock) and **1,236 TFLOP/s in FP8**, 125% of the bf16 spec.
- Bursts are faster: at the 1.3B GEMM shapes (M = 65,536 tokens, CUPTI medians of up to 30 calls) bf16 runs at
  760-850 TFLOP/s and FP8 with both operands K-major at 1,520-1,610 TFLOP/s (77-81% of the FP8 spec). XLA's
  own FP8 weight-gradient GEMMs, which transpose both activations in separate kernels, deliver 70-350 TFLOP/s
  effective, slower than bf16. Narrow GEMMs are slower (N = 512: 740-760, N = 320: 540-670 TFLOP/s in bf16).
  At M = 16,384 (a 4 × 4096 micro-step) the GEMMs run at 0.91-1.14x of their M = 65,536 rates.
- HBM (spec 3,352 GB/s): read-only streams 3.1-3.2 TB/s, read + write 2.9-3.0 TB/s.
- A cuBLAS GEMM fills every SM, so a memory-bound kernel on a second CUDA stream overlaps it by 0%; a
  residual add costs +0.256 ms as its own pass and +0.06-0.09 ms in cuBLASLt's epilogue (MLP down and o_proj).

**The ideal step** (estimated): [`speed_of_light_step.py`](speed_of_light_step.py) applies these ceilings to
the profiles above op by op. The ideal step recomputes nothing (the cross entropy's logits included), fuses
every elementwise pass away, runs every GEMM and kernel at its ceiling and executes only the attention tiles
that document masking leaves (0.567 of the counted causal work on the profiled batch); power-capped scales the
tensor-core work to the median sustained rate.

| preset | measured step (MFU) | ideal, power-capped | ideal, FP8 GEMMs, power-capped (bf16-eq. MFU) | tensor work at the sustained rate |
|---|---|---|---|---|
| `transformer-1p3b` | 1,978 ms (31.1%) | 882 ms (**69.7%**) | 516 ms (119.0%) | 72.7% |
| `transformer-swa-1p3b` | 1,716 ms (34.6%) | 874 ms (68.0%) | 508 ms (117.0%) | 71.0% |
| `gdn-hd128-1p3b` | 2,056 ms (31.3%) | 1,028 ms (62.7%) | 599 ms (107.6%) | 68.7% |
| `mamba2-1p3b` | 2,043 ms (30.6%) | 1,079 ms (57.9%) | 696 ms (89.8%) | 68.7% |
| `gdn2-1p3b` | 2,664 ms (27.3%) | 1,311 ms (55.6%) | 828 ms (87.9%) | 68.7% |
| `kda-1p3b` | 2,382 ms (25.8%) | 1,125 ms (54.6%) | 718 ms (85.4%) | 68.6% |
| `mamba3-1p3b` | 2,043 ms (31.7%) | 1,230 ms (52.7%) | 845 ms (76.7%) | 68.6% |

In bf16 the transformer's ideal step sits at the 70% line (69.7%; 70.4% with the average document mix of 20
batches, 70.8% at the fastest GPU's rate) and the recurrent presets at 52.7-62.7%: their tensor-core FLOPs
equal their model FLOPs, so they cannot pass the sustained rate's 68.9%, and 100% is impossible in bf16. With
FP8 GEMMs whose quantize, amax and transposes are fused into the producers, the ideal step scores 77-119%
bf16-equivalent MFU, 38-60% of the FP8 peak.

### A 125M step

[`bench_end_to_end_step.py`](bench_end_to_end_step.py) builds the train step as `jla train` does on one device
(the same TOML, model builder, optimizer, loss and `build_train_step`) on uniform random tokens and, with
`--profile-dir`, attributes every kernel to its HLO instruction and JAX source line. One step of 8 × 2048
tokens, vocabulary 128,256, 15 timed steps, provisional:

| | mamba2-130m | mamba3-130m | kda-hybrid-125m |
|---|---|---|---|
| step, median / min (a) | 117.7 / 117.4 ms (139k tok/s) | 110.5 / 109.7 ms (148k tok/s) | 96.8 / 96.1 ms (169k tok/s) |
| matmuls (cuBLAS projections, MLP and LM head), ms (b) | 29.5 | 33.6 | 33.7 |
| Mosaic GPU kernels (b) | 20.7 | 36.6 | 16.7 |
| cross entropy: non-GEMM passes + Triton logits-recompute GEMM (b) | 11.3 + 7.4 | 11.9 + 7.4 | 12.9 + 9.1 |
| mixer glue (gates, B/C broadcast, D skip, layout) (b) | 19.0 | 8.9 | 1.5 |
| norms / short conv1d (b) | 11.1 / 7.5 | 5.5 / – | 6.9 / 5.5 |

(a) No other timed job on the GPU, with `jla.layers.ssm.GROUPED_BC` and `SPLIT_ZX` off. Both are on by
default: Mamba-2 then reads its one B/C group in the kernel instead of repeating it to 24 heads, and both Mamba
layers compute the `z` and `x` rows as two GEMMs. In interleaved runs on one GPU (3 repeats, a second machine
whose host was busy) that takes mamba2-130m from 119.5 to 105.5 ms per step (137k to 155k tok/s, 1.13x);
mamba3-130m is unchanged (0.996x) and kda-hybrid-125m uses neither. (b) Device time per step from a profile
with both options off, on a GPU that other jobs' tests shared.

## Where we lose, and limits

- The 1.3B profiles are of 17 Oct code (bf16, full remat; see Setup). The release's conv barrier, short-conv
  kernels, fused attention backward and forward-gradient cross entropy address the largest gaps they show; the
  release steps are not profiled by category here.
- The jla softmax attention backward was the largest single gap of the transformer step: `kernel_dkv` and
  `kernel_dq` ran at 83-90 TFLOP/s of model FLOPs (357 and 257 ms per step) against 249 TFLOP/s for the
  forward. [attention.md](attention.md) has the release kernels against FA3.
- The KDA / GDN-2 intra-chunk backward kernel (`vg_bwd_intra`) moved 105-131 GB per step at 550-650 GB/s:
  191-202 ms against 35-43 ms at bandwidth.
- At 700 W the H100 cannot hold the 1,830 MHz that the bf16 spec assumes, so a bf16 step made only of GEMMs
  tops out near 69% MFU against 989 TFLOP/s. The speed-of-light model keeps the profiled structure (one custom
  call per mixer kernel, a separate optimizer pass, narrow projections) and credits FLOPs only to GEMMs and the
  attention and recurrence kernels.
- At 125M the cross entropy is the one sink all three models share (18-22 ms per step plus its cuBLAS GEMMs).

## Exactness and tests

The profilers only read traces and compiled HLO; `tests/test_profile_1p3b.py`, `tests/test_speed_of_light_step.py`
and `tests/test_bench_end_to_end_step.py` cover the attribution and the model. The Mamba-2 grouped B/C path
gives bit-identical forward outputs and final states to the repeated path, and the split `z`/`x` GEMMs match
the stacked GEMM to 1e-5 in float32 (`tests/test_mixer_fast_paths.py`).

## Reproduce

```bash
# the jla step (CUDA graphs off), then flame's full-checkpointing step, then the tables
python benchmarks/profile_1p3b.py --arch gdn-hd128-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --steps 6 --warmup 2 --profile-steps 2 --profile-dir traces/gdn-hd128 --json results/profile_1p3b.json
python benchmarks/profile_1p3b_flame.py --runner benchmarks/bench_e2e_flame.py --json results/profile_flame.json \
    -- --arch gdn-hd128-1p3b --data /path/to/fineweb_edu_3B --nproc 1 --micro-batch 16 --steps 14 --skip 5 \
    --no-compile --flame-src <flame checkout> --dump-dir dump/gdn-hd128 --json results/flame.json
python benchmarks/profile_1p3b_report.py --ours results/profile_1p3b.json --flame results/profile_flame.json \
    --json results/profile_report.json
# hardware ceilings, then the speed-of-light model over a profile
python benchmarks/bench_speed_of_light.py clock --seconds 60 --json results/clock.json
python benchmarks/bench_speed_of_light.py gemm --impl xla --json results/gemm_xla.json
python benchmarks/bench_speed_of_light.py hbm --json results/hbm.json
python benchmarks/speed_of_light_step.py --bench-json results/gemm_xla.json --bench-json results/clock.json \
    --attention-work-fraction transformer-1p3b=0.5666 --profile-dir traces/transformer-1p3b --json results/sol.json
# a 125M step, profiled
python benchmarks/bench_end_to_end_step.py benchmarks/end_to_end/fw-mamba2-130m.toml --steps 15 \
    --profile-dir traces/mamba2-130m --json results/end_to_end_step.json
```
