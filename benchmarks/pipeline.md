# Pipeline parallelism (`mesh.pp`)

`jla.distributed.pipeline` splits the scanned layer stack into `pp` contiguous stages and runs a GPipe
schedule (`M` microbatches, `M + S - 1` iterations, `lax.ppermute` between stages) inside the compiled train
and eval steps, after MaxText's `layers/pipeline.py`. It is checked against the plain single-device model and
timed against FSDP and data parallelism of the same model on two H100s; there is no upstream baseline. The
idle fraction `(S - 1) / (M + S - 1)` is the step metric `pipeline/bubble_fraction`.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 compute. `benchmarks/bench_pipeline.py`: one compiled train step of
  `transformer-125m` (12 layers, d 768, vocab 50304), global batch 16 x 1024 tokens, AdamW, the same init key
  and batch for every layout, so the first-step loss and gradient norm must agree.
- Timing: the median step time per layout after two warm-up steps. Provisional: measured on 17 Oct 2026 on two
  GPUs while other benchmarks ran beside them; medians moved by up to 2x between runs, so compare layouts within
  one run only.

## Provisional results

| attention | layout | ms per step (median) | first loss | gradient norm |
|---|---|---|---|---|
| Mosaic GPU | dp=2 | 68.7 (min 44.5) | 10.94317 | 1.1173 |
| Mosaic GPU | fsdp=2 | 82.7 | 10.94319 | 1.1184 |
| Mosaic GPU | pp=2, M=4 (bubble 0.2) | 186.4 (min 152.5) | 10.94317 | 1.1177 |
| cuDNN | dp=2 | 33.3 | 10.94310 | 1.1175 |
| cuDNN | fsdp=2 | 78.0 | 10.94318 | 1.1182 |
| XLA | dp=2 | 66.3 | 10.94315 | 1.1179 |
| XLA | fsdp=2 | 176.8 | 10.94319 | 1.1180 |

The `pp=2` row was timed while the fully manual stages still saved every activation instead of following
`model.remat`; it has not been re-timed. The stages wrap every layer in a checkpoint that re-gathers the FSDP
weights in the backward pass, and that checkpoint now carries the model's remat policy minus the gathered
weights. Compile-time temporary memory of the gradient, `transformer-125m` cut to 4 layers, batch 8 x 1024,
`M = 4`, Mosaic GPU attention, one H100:

| forward | `remat = "full"` | `remat = "none"` |
|---|---|---|
| plain model | 0.369 GiB | 2.796 GiB |
| pipeline, manual over `pp` only | 0.409 GiB | 3.35 GiB |
| pipeline, fully manual | 0.415 GiB | 3.48 GiB |

## Where we lose, and limits

- On one machine the pipeline is slower than FSDP and DP: `pp=2, M=4` took 2.3x FSDP's step and 2.7x DP's, and
  DP with cuDNN attention, the GPU default that the pipeline cannot use, took 33.3 ms. Pipeline parallelism is
  for models whose layers do not fit FSDP across slower multi-host links, which one 8-GPU machine cannot show.
- GPipe only (no circular, interleaved or 1F1B schedule): every stage keeps the activations of its `M`
  microbatches until the backward pass. Stages are contiguous slices of the scanned repeats
  (`scan_layers = true`; `pp` must divide `n_layers / period`).
- The embedding runs on every stage; the final norm, LM head, loss and MTP run sharded over `pp` only when every
  batch shard holds a multiple of `pp_microbatches * pp` rows (otherwise every stage recomputes them).
- On GPU, attention inside the pipeline (and `model.mla`) must use `implementation = "xla"` or `"mosaic_gpu"`:
  cuDNN attention fails inside a partially manual `shard_map` and its backward crashes the XLA GPU compiler inside
  a fully manual one (JAX 0.11.2); the pipeline raises a clear error instead.
- MoE layers route each pipeline microbatch on its own tokens (equal to the plain model for `M = 1`), and with the
  batch split over `dp` and `fsdp` their head stays replicated (XLA's SPMD partitioner aborts on the sharded head).
  Residual dropout masks differ from the plain model's (same distribution).
- The stages also run on the bubble iterations, which adds no gradient only for layers whose backward pass is
  linear in the cotangent; the DSA indexer's KL gradient is not, so `pp > 1` refuses
  `sparse_attention.indexer_loss_coef > 0`.
- Mamba-1 layers run in a `shard_map` manual over `pp` only (XLA segfaults compiling their gradient fully manual,
  seen on CPU); on GPU that path cannot partition Pallas kernels.
- Not combined with expert or context parallelism, and not run across hosts (`dcn_pp` is untested on several slices).

## Correctness

`tests/test_pipeline_parallel.py` (CPU, 8 host devices, float32) compares the pipelined loss and every
parameter gradient with the plain single-device model for attention, a Gated DeltaNet + attention hybrid with
MTP, Mamba-2, Mamba-1, GDN-2 + sliding-window attention, MoE and DeepSeek sparse attention, on meshes that
combine `pp` with `dp`, `fsdp` and `tp`: gradients within 2.9e-6 relative, objectives within 1e-7. The `Trainer`
at `pp = 2` and `pp = 4` matches the single-device run within 1.2e-7 (also saved at `pp = 2` and resumed at
`pp = 4`), and a `pp = 2` checkpoint restores bit-exactly at `pp = 1`. On two H100s the pipelined step agrees
with FSDP and DP (loss within 6e-6 relative, gradient norm within 8e-4, bf16).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/bench_pipeline.py --preset transformer-125m --batch 16 --seq 1024 \
  --layouts fsdp=2 dp=2 pp=2,M=4 --attention mosaic_gpu --steps 10 --json results/pipeline.json
pytest tests/test_pipeline_parallel.py tests/test_activation_sharding.py
```
