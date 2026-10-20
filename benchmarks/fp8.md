# FP8 training of the dense projections and MLPs

`ModelConfig.fp8` switches the dense projections and MLPs to FP8 matmuls (`jla.layers.fp8`): e4m3
inputs and weights, e5m2 output gradients, per-tensor scales and float32 accumulation. It is off by
default, and is compared here with the same train step in bf16.

```toml
[model]
fp8 = "delayed"          # "off" (bf16), "delayed" or "current"
fp8_amax_history = 1024  # delayed scaling: optimizer steps of amax history behind each scale
```

## Setup

- One H100 80GB HBM3, JAX 0.11.2.
- Train step (`bench_fp8.py`): `build_train_step` + `lm_loss`, the preset defaults with
  `remat = "full"` and `scan_layers`, random tokens; the recipes run one after another in one process,
  bf16 first and again last. One matmul (`bench_fp8_gemm.py`): the forward and both backward GEMMs,
  `x [M, K]` and `g` bfloat16, `w [K, N]` float32.
- Timing: the median of 10 timed steps after 3 warmups (train step) or of 30 calls after 3 warmups
  (one matmul).
- Measured on 17 Oct 2026, one benchmark per GPU while other jobs' tests could share the GPU, so the
  numbers are provisional.

## Results

| preset | tokens / step | parameters, optimizer | bf16 | FP8 delayed | FP8 current |
|---|---|---|---|---|---|
| `transformer-125m` (d 768) | 8 x 2048 | float32, AdamW | 54.5 ms, 300.9k tok/s | 60.1 ms, 272.5k tok/s (**0.91x**) | 60.9 ms, 269.0k tok/s (**0.89x**) |
| `transformer-1b` (d 2048, d_ff 8192) | 4 x 2048 | bfloat16, SGD | 173.3 ms, 47.3k tok/s | 159.4 ms, 51.4k tok/s (**1.09x**) | 152.7 ms, 53.7k tok/s (**1.13x**) |

One FP8 matmul, forward and both backward GEMMs:

| M x K x N | bf16 | FP8 (delayed / current) |
|---|---|---|
| 16384 x 2048 x 8192 (1B MLP) | 2.26 ms | 1.64 / 1.70 ms (**1.38x / 1.33x**) |
| 16384 x 768 x 2304 (125M MLP) | 0.27 ms | 0.38 / 0.34 ms (**0.70x / 0.78x**) |

The compiled 1B FP8 step holds 18 (delayed) / 16 (current) cuBLASLt FP8 GEMM calls and 5 / 1 Triton
FP8 GEMM fusions; its only other GEMMs are 2 bfloat16 cuBLASLt calls (the linear cross-entropy head),
against 17 in the bf16 step.

## Where we lose, and limits

- **Small models.** H100 FP8 tensor cores have twice the bfloat16 peak, but every FP8 matmul adds
  memory-bound passes: quantizing `x`, `w` and `g` (with their amax reductions) and transposing the
  FP8 `x` and `g` for the weight-gradient GEMM (cuBLASLt FP8 takes only the TN layout). At `d_model` 768
  these cost more than the GEMMs save, so FP8 is a loss for 125M-class models (0.89-0.91x). Fusing
  the quantize, amax and transpose passes into one kernel per tensor, as Transformer Engine does, is
  the next speed step.
- **The 1B rows** use bfloat16 parameters and plain SGD to fit a 20 GiB memory cap (0.25 of the GPU);
  with the preset's float32 parameters the FP8 step did not fit that cap, which was not investigated.
  FP8 compiles take longer (56 and 82 s against 9 s for bf16).
- **Coverage.** Per-tensor scaling only; block-scaled MXFP8 (`jax.nn.scaled_dot_general`) runs
  natively only on Blackwell-class GPUs. The MoE routed experts, the embedding, the LM head and the
  attention core stay in bfloat16.
- **Delayed scaling** records the sum of the `n` amaxes of a matmul applied `n` times in one loss
  evaluation, so its scales can be up to `n` times larger than needed; a pipeline that runs each
  stage's layers M + S - 1 times per forward goes through this path, and FP8 with pipeline parallelism
  has not been tested. A matmul that never gets a cotangent (the DSA indexer with
  `indexer_loss_coef = 0`) never records an amax and quantizes with unit scales for the whole run.
- **No conversion.** FP8 cannot be switched on for an existing run or for imported weights: strict
  resume and the Hugging Face converter reject the extra `Fp8State` leaves. The logged parameter count
  includes that state (3 + 3 x `fp8_amax_history` floats per FP8 matmul, 0.02% of `transformer-1b`).
- On GPUs without FP8 tensor cores (before Ada/Hopper) XLA upcasts the FP8 dots: the numerics are the
  same, the speed is not.

## Correctness

`tests/test_fp8_training.py` (CPU and GPU) and `tests/test_fp8_multidevice.py` (4 CPU devices) check
that the forward and both backward GEMMs of `fp8_dot` match a NumPy emulation (ml_dtypes rounding,
float64 GEMMs) to 2e-3 of the output's max with float32 activations and to 1e-2 with bfloat16 ones,
that FP8 loss tracks bf16 on the `tiny` preset (40 AdamW steps, every step within `0.05 + 1%`; largest
gaps 0.008 delayed and 0.010 current on an H100), the delayed state's updates (microbatches, rejected
steps, matmuls without a cotangent or used twice), checkpoint and resume, sharded training against one
device, and that the compiled H100 step contains FP8 GEMMs, including e5m2 ones.

## Reproduce

```bash
python benchmarks/bench_fp8.py --preset transformer-125m --recipes off,delayed,current,off \
    --batch 8 --seq 2048 --steps 10 --json results/fp8_125m.json
python benchmarks/bench_fp8.py --preset transformer-1b --recipes off,delayed,current,off --batch 4 --seq 2048 \
    --steps 10 --param-dtype bfloat16 --optimizer sgd --json results/fp8_1b.json
python benchmarks/bench_fp8_gemm.py                    # 16384 2048 8192, the 1B MLP shape
python benchmarks/bench_fp8_gemm.py 16384 768 2304     # the 125M MLP shape
pytest tests/test_fp8_training.py tests/test_fp8_multidevice.py
```

## How it works

`apply_fp8` (called by `build_model` when `fp8 != "off"`) converts every `Linear`, `StackedLinear` and
the gate/up GEMM of `SwiGLU` whose input and output widths are multiples of 16 (the cuBLASLt FP8
alignment) into `Fp8Linear` / `Fp8StackedLinear` / `Fp8SwiGLU`, keeping the parameters, their dtypes
and their logical axes. Each FP8 matmul is a `jax.custom_vjp` in the form of Flax's
`Fp8DirectDotGeneralOp` (which MaxText's GPU `fp8` recipe uses): the forward quantizes `x / s_x` and
`w / s_w` to e4m3 and runs an FP8 dot that writes the compute dtype (cuBLASLt fast accumulation), then
multiplies by `s_x s_w`; the backward quantizes the output gradient to e5m2 and runs both gradient GEMMs
with fast accumulation off. Writing float32 from the dot and casting afterwards would move twice the
output bytes, and dequantize-then-dot is not rewritten to an FP8 GEMM by this XLA at all. `"delayed"`
scaling (Transformer Engine's default) takes `s = max(amax history) / fp8_max` over the previous
`fp8_amax_history` optimizer steps, from an `Fp8State` that is sharded, checkpointed and restored like
a parameter; `"current"` takes the amax of the tensor being quantized and keeps no state. The matmul's
VJP returns the new scales and histories as the "gradient" of its `Fp8State` (MaxText's
`_overwrite_with_gradient` variables); the train step combines them over microbatches with `max`,
keeps them out of the optimizer and writes them back once per applied step, so a rejected step keeps
the old state and a matmul applied `n` times advances once.
