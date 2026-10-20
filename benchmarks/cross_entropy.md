# Fused LM-head cross entropy on Mosaic GPU

`jla.ops.cross_entropy.linear_cross_entropy(hidden [N, D], weight [V, D], targets [N], ...)` computes
the summed token cross entropy of `hidden @ weight.T` (plus the z-loss sum, with label smoothing and
`ignore_index`) without materialising `[N, V]` logits. `impl="xla"` (the default) scans over
vocabulary tiles and writes each tile's f32 logits to HBM; `impl="mosaic_gpu"` runs fused Hopper
kernels that keep the logits on chip and recompute them in the backward (and raises when they cannot
run the call); `impl="auto"` takes the kernels where they apply. `linear_cross_entropy_objective`
(`train.ce_grad_in_forward = true`) forms the head's gradient in the forward pass instead: three GEMMs
instead of four. Both are compared with the XLA op, Liger Kernel, Cut Cross-Entropy and FLA's fused
linear cross entropy, and in 1.3B training.

## Setup

- One H100, JAX 0.11.2. Kernel tables: bf16 `hidden` and `weight`, V = 128,256 (the Llama-3.1
  vocabulary), the mean loss, uniform random targets, no label smoothing, no z-loss.
- Baselines (torch 2.13.0+cu129, Triton 3.7.1): Liger-Kernel 0.8.4
  (`LigerFusedLinearCrossEntropyLoss`), Cut Cross-Entropy main `3de376c` (25.9.3; `cce` with its
  default gradient filter, `cce_exact` with `filter_eps=None`) and flash-linear-attention main
  `c6e4f00` (0.6.0; `FusedLinearCrossEntropyLoss` with `accumulate_grad_in_fp32` True, the default, or
  False).
- Timing: the median GPU kernel time of 20 or 30 calls after 3 warmups (CUPTI for JAX,
  `torch.profiler` kernel sums for torch), one timed run per GPU at a time. Training: `jla train`
  through `bench_e2e_1p3b.py` on FineWeb-Edu, the median ms per step of steps 6-20, MFU at the median
  step.
- Measured on 17-19 Oct 2026 while other jobs' test processes could share the machine, so every number
  on this page is provisional.

## Results

### Kernel time

The LM head at D = 2048, forward + backward in ms (two runs, run 1 / run 2). N = 16,384 is one
micro-batch of `transformer-1p3b` (4 x 4096) and N = 65,536 one step of `gdn-hd128-1p3b` and
`mamba2-1p3b` (16 x 4096):

| N | XLA op | recompute, `vocab_chunk` 4096 | recompute, `vocab_chunk` 16384 | forward gradient, 8192 rows (default) | forward gradient, 16384 rows | forward gradient + FP8 head GEMMs, 16384 rows |
|---|---|---|---|---|---|---|
| 16,384 | 68.09 | 55.95 / 54.71 | 52.49 / 52.67 | 40.86 / 40.16 | 40.67 / **39.02** | **31.34** |
| 65,536 | 255.88 | 226.49 / 223.90 | 210.56 / 208.71 | 166.77 / 166.12 | 163.15 / **162.92** | **131.06** |

The forward gradient is 1.28-1.35x faster than the best recompute path and 1.57-1.67x faster than the
XLA op. Per 65,536 tokens the head's `6 N D V` = 1.03e14 FLOPs run at 404 TFLOP/s in the XLA op
(255.9 ms), 491 in the recompute kernels (210.6 ms), 619 in the forward gradient (166.8 ms: the
Mosaic GPU logits kernel at about 640 TFLOP/s and two cuBLAS GEMMs at about 740) and 750
bf16-equivalent with the FP8 head GEMMs (137.8 ms in 8,192-row blocks). The loss alone (no
differentiation) runs the same statistics kernel on every Mosaic GPU path: 11.4-11.6 ms at N = 16,384
and 53.0-55.0 ms at 65,536, against 18.25 and 71.39 ms for the XLA op.

Against the strongest torch baselines in one run, N = 16,384, forward + backward in ms:

| D | ours, 8192 rows | ours, 16384 rows | ours + FP8, 16384 rows | recompute, chunk 16384 | Liger 0.8.4 | FLA main, bf16 `dweight` |
|---|---|---|---|---|---|---|
| 2048 | 41.25 | 39.79 | **31.64** | 52.29 | 45.28 | 38.82 |
| 768 | 17.56 | **17.14** | **15.68** | 19.02 | 51.46 | 18.42 |

The bf16 forward gradient is 1.10-1.14x Liger at D = 2048 and 2.9-3.0x at D = 768, and 1.07x FLA's
bf16-accumulation mode at D = 768. The recompute path (`linear_cross_entropy(impl="mosaic_gpu")`) is
1.51-1.57x the XLA op forward + backward at D = 768, 1.24-1.47x at 1024 and 1.20-1.25x at 2048;
2.28-2.34x Liger at D = 768 and 1.33-1.36x at 1024; 2.78-3.96x Cut Cross-Entropy; and 3.66x (D = 768)
and 3.53x (D = 2048) FLA's default mode. Its forward alone is 2.2-2.6x, 1.7-1.9x and 1.3-1.5x the XLA
op at D = 768, 1024 and 2048, with the forward kernel at 570-665 TFLOP/s.

### Training at 1.3B

One H100, 16 x 4096 tokens per step (`transformer-1p3b` 4 x 4 x 4096), each preset's TOML against the
same run with `train.ce_impl = "mosaic_gpu"` and `train.ce_grad_in_forward = true`, back to back on one
GPU. FP8 configurations use FP8 GEMMs in the blocks; the LM head stays bf16 unless `train.ce_fp8` is
set. Median ms per step:

| preset | precision | the TOML's cross entropy | forward gradient (8192 rows) | forward gradient + FP8 head GEMMs | change | MFU |
|---|---|---|---|---|---|---|
| `transformer-1p3b` | FP8 | XLA: 1,111.6 | 1,011.5 | **974.7** | 1.099x; 1.140x with the FP8 head | 55.3% -> 60.8% -> 63.1% |
| `gdn-hd128-1p3b` | FP8 | XLA: 1,280.6 | 1,195.9 | **1,180.0** | 1.071x; 1.085x | 50.3% -> 53.9% -> 54.6% |
| `mamba2-1p3b` | FP8 | `auto` = recompute: 1,585.8 | 1,516.6 | **1,484.6** | 1.046x; 1.068x | 39.4% -> 41.2% -> 42.1% |
| `transformer-1p3b` | bf16 | XLA: 1,218.5 | 1,121.5 | **1,102.1** | 1.086x; 1.106x | 50.5% -> 54.8% -> 55.8% |
| `gdn-hd128-1p3b` | bf16 | XLA: 1,618.4 | **1,526.5** | not measured | 1.060x | 39.8% -> 42.2% |
| `mamba2-1p3b` | bf16 | `auto` = recompute: 1,723.2 | **1,678.5** | not measured | 1.027x | 36.3% -> 37.2% |

The forward gradient makes these steps 1.027-1.099x faster, and 1.068-1.140x with the FP8 head GEMMs.

With every other change of the release in the code (FP8 weights quantized once per step, the fused
attention backward and the rest), each TOML as written against the same TOML plus `train.ce_impl =
"auto"` and `train.ce_grad_in_forward = true`, interleaved (A B A B; one GPU) or, on 8 GPUs, FSDP 8
with 524,288 tokens per step and the per-shard path; median tokens/s (per GPU on 8 GPUs):

| preset | 1 GPU, bf16 | 1 GPU, FP8 | 8 GPUs, FP8 |
|---|---|---|---|
| `transformer-1p3b` | 53,127 -> 58,692 (**1.105x**, 49.8% -> 55.1% MFU) | 64,322 -> 69,981 (**1.088x**, 60.3% -> 65.6%) | 53,967 -> 60,144 (**1.114x**, 50.6% -> 56.4%) |
| `transformer-swa-1p3b` | 54,641 -> 59,619 (**1.091x**) | 65,503 -> 73,268 (**1.119x**, 59.4% -> 66.5%) | not run |
| `gdn-hd128-1p3b` | 41,152 -> 43,613 (**1.060x**) | 52,374 -> 56,047 (**1.070x**) | 49,464 -> 53,204 (**1.076x**, 48.6% -> 52.3%) |
| `kda-1p3b` | not run | 40,708 -> 43,018 (**1.057x**) | 0.954x (one pair) |
| `mamba2-1p3b` (TOML: `ce_impl = "auto"`, recompute) | not run | 42,548 -> 43,751 (**1.028x**) | not run |

On one GPU every pair is faster (1.028-1.119x). `transformer-1p3b.toml` and `gdn-hd128-1p3b.toml`
set `ce_impl = "auto"` and `ce_grad_in_forward = true`: faster on one GPU, not slower on eight, and
the 20-step losses inside the run-to-run noise.

### Loss parity

`transformer-1p3b`, bf16, the same FineWeb-Edu batches in the same order and the same initialization.
The noise reference is the XLA op with `vocab_chunk` 8192 against the baseline's 4096 (the same math in
another summation order): over 200 steps it differs from the baseline by 0.0039 per step on average
(steps 101-200), and the bf16 forward gradient by 0.0068, with step 1 identical in every run. Over
600 steps every 100-step window of the bf16 forward gradient is within 0.0012 of the baseline, with no
drift. The FP8 head GEMMs sit 0.014-0.040 above the baseline from step 50 on, 0.021 (0.4%) in steps
501-600: the baseline reaches their mean loss of steps 551-600 about 57 steps (10%) earlier.

## Where we lose, and limits

- **FLA's bf16-accumulation mode.** FLA's `FusedLinearCrossEntropyLoss` with a bf16 `dweight` beats
  the recompute path (0.82x at D = 768, 0.69x at 2048), and at D = 2048 the bf16 forward gradient is
  0.98x of it (16384 rows; 0.94x at 8192 rows; 1.23x with the FP8 head GEMMs). That mode is less exact:
  its `dweight` error is 37-44% above the bf16 floor, where ours stays on it.
- **Liger at D = 2048** beats the recompute path (0.78-0.82x): its row chunks are large enough there,
  and the recompute's fourth GEMM (about 13 ms) is the gap. The forward gradient removes that GEMM.
- **The FP8 head GEMMs cost loss**: 0.02 (0.4%) after 600 steps, for a 1.3-3.8% faster step than the
  bf16 forward gradient (1.068-1.140x in all against the TOMLs). `train.ce_fp8` is lossy and not
  recommended.
- **Memory.** The forward gradient holds `4 * row_chunk * V` bytes of transient buffers per block (4.2
  GB at 8,192 rows and V = 128,256): `transformer-1p3b`'s peak rises from 50.4 to 54.8 GiB in bf16 and
  from 52.1 to 56.4 GiB in FP8, and from 28.4 to 34.6 GiB per GPU on 8 GPUs. 16,384-row blocks ran
  out of memory in a 0.92 preallocated pool on `transformer-1p3b`, so the default stays 8192.
- **Off in three TOMLs.** `transformer-swa-1p3b`, `kda-1p3b` and `mamba2-1p3b` keep it off:
  `kda-1p3b`'s one 8-GPU pair measured 0.954x, and the other two have no 8-GPU measurement.
  `mamba2-1p3b` in FP8 also moved the 20-step loss by up to 0.083 at one step against a run-to-run
  spread of 0.002, which was not judged without a reordering reference.
- **What is left in the forward gradient**: the elementwise gradient pass (12.3 ms of 163 ms per
  65,536 rows), a separate float32 `dweight` add per row block (0.8-0.9 ms per block at D = 2048) and,
  with FP8, XLA's transposes to the K-major layout (9.3 ms).
- **The recompute op runs on one device.** With several devices `linear_cross_entropy` uses XLA under
  `"auto"` and raises under `"mosaic_gpu"`, unless it runs inside a `shard_map` whose mesh axes are all
  manual; `linear_cross_entropy_objective` runs per shard itself under a context mesh.
- **Cut Cross-Entropy** is measured on random inputs with a flat softmax, where its filter skips almost
  no block; on a trained model's peaked softmax its filtered backward would be faster (and
  approximate). Its `dhidden` error is 13-20x the bf16 floor even with `filter_eps=None`.

## Correctness

`tests/test_mosaic_gpu_linear_cross_entropy.py` and `tests/test_mosaic_gpu_linear_cross_entropy_objective.py`
compare the values and both gradients with the XLA op and a naive f32 full-logits oracle (label
smoothing, z-loss, ignored tokens, padded vocabularies of 50,304 and 1,000, uneven blocks and chunks,
2-CTA clusters, float16, an all-ignored batch, the multi-device dispatch and the per-shard path on two
H100s): the recompute kernels are within 1e-3 (`dhidden`) and 1e-4 (`dweight`) of the XLA op, and the
forward gradient's error against the f32 oracle is at most 1.05x the XLA op's. Against float32 logits
(N = 4096) ours and the XLA op sit on the bf16 floor (`dhidden` / `dweight` relative errors
1.656e-3 - 1.663e-3 / 1.449e-3 - 1.505e-3, the floor 1.655e-3 - 1.662e-3 / 1.448e-3 - 1.504e-3);
FLA's default mode is 1.5-2 % above it and Liger's `dweight` 43-55 %. The FP8 head GEMMs have 3.0-5.5%
relative gradient error against 0.17-0.30% for bf16, from the e5m2 gradient's 2 mantissa bits.

## Reproduce

```
python benchmarks/bench_cross_entropy.py --iterations 20 --shapes 16384,2048,128256 65536,2048,128256 \
    --mosaic-chunks 16384 --objective-rows 4096 8192 16384 --breakdown --json results/cross_entropy_fwdgrad.json
python benchmarks/bench_cross_entropy.py --iterations 20 --shapes 16384,2048,128256 65536,2048,128256 \
    --skip-xla --mosaic-chunks 16384 --objective-rows 8192 16384 --objective-fp8-rows 8192 16384 --breakdown \
    --json results/cross_entropy_fwdgrad_fp8.json
python benchmarks/bench_cross_entropy.py --iterations 30 --torch-python <torch venv> --breakdown \
    --mosaic-chunks 16384 --json results/cross_entropy.json
python benchmarks/bench_cross_entropy.py --accuracy-rows 4096 --torch-python <torch venv> \
    --json results/cross_entropy_accuracy.json
python benchmarks/bench_e2e_1p3b.py --arch <preset> --data <FineWeb-Edu CIDX prefix> --impl kernels --dp 1 --fsdp 1 \
    --steps 20 --skip 5 --json <out> \
    [--set train.ce_impl=\"mosaic_gpu\" --set train.ce_grad_in_forward=true [--set train.ce_fp8=true]]
# FP8 configurations add: --set model.fp8=\"delayed\" --set model.fp8_min_features=512 --set model.fp8_fused=true
```

`--torch-kinds liger,fla_bf16acc` restricts the torch side to the two strongest baselines.

## How it works

Both kernels are one persistent, warp-specialised GEMM over `hidden [N, D] x weight [V, D]^T`, adapted
from JAX's `hopper_matmul_mgpu.py` (Apache-2.0): a TMA producer warpgroup streams `128 x 64` hidden
and `256 x 64` weight blocks through a 3-4 stage pipeline, and two WGMMA consumer warpgroups each keep
a `128 x 128` f32 logit tile in registers. The forward epilogue writes only a per-row max and exp-sum
per 128-column tile, which XLA combines into the logsumexp. The recompute backward runs the same GEMM
per vocabulary chunk and forms the logit gradient in its epilogue, stored once in bf16 for the
`dhidden` and `dweight` cuBLAS GEMMs. The forward gradient instead writes the target logit and
`e = exp(logit - m)` in float16 (11 significant bits in (0, 1]) from the forward kernel; one XLA
elementwise pass turns them into the bf16 (or e5m2) logit gradient, and two cuBLAS GEMMs form
`dhidden` and `dweight`, so the backward only scales them; row blocks are chained with
`optimization_barrier`, so one block's buffers are live at a time. Forming the gradient during the
forward pass is the design of Liger Kernel's fused linear cross entropy and FLA's
`FusedLinearCrossEntropyLoss`; no code was copied. The weight is zero-padded to a multiple of 256 rows
for any vocabulary size. Under a context mesh the objective runs per shard in a `shard_map`, and a
float32 weight whose columns split over the devices enters it in the compute dtype, so the all-gather
moves bf16 (1.84 GB less over NVLink per GPU and step for `transformer-1p3b` on 8 GPUs) and the
float32 gradient is reduce-scattered to each device's columns.
