# Hugging Face checkpoint import and export (`jla convert`)

`jla.convert` maps Hugging Face `transformers` checkpoints (a local directory with `config.json` and
`*.safetensors`; nothing is downloaded) onto `LanguageModel` and back, for Llama 2/3/3.1+, Mistral, Qwen2/2.5,
Qwen3, Phi-3, Mixtral, Qwen3-MoE, DeepSeek-V3 and Mamba-2, in the `transformers` 4.x and 5.x config layouts.
It is compared with `transformers` itself: the same logits, the same tensors on export, and the load and save
times of `from_pretrained` and `save_pretrained`.

## Setup

- H100 80GB HBM3, JAX 0.11.2; `benchmarks/bench_convert_hf.py` builds each model at its real shape with random
  weights and saves it as a bfloat16 checkpoint in 500 MB shards.
- Baseline: `transformers`' `from_pretrained` (float32 on the host) and `save_pretrained`; the tests build their
  models with `transformers` 5.20.1.
- Timing: median of 3, warm page cache; `load_hf_checkpoint` until every float32 parameter is ready on one H100,
  `save_hf_checkpoint` in bfloat16 with 500 MB shards and fsync. Logits: float32 on 64 tokens, JAX matmuls at
  `highest` precision.
- Provisional: measured on 17 Oct 2026 on one H100 while the other GPUs ran other benchmarks.

## Provisional results

| shape | params | HF `from_pretrained` (host) | `load_hf_checkpoint` (H100) | max abs logit diff (fp32) | argmax agree | HF `save_pretrained` | `save_hf_checkpoint` | export bytes equal |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 596 M | 0.38 s | 3.42 s | 6.9e-6 (logits up to 3.6) | 100% | 3.22 s | 6.91 s | yes |
| Llama-3.2-1B | 1.24 B | 0.66 s | 7.83 s | 2.3e-5 (logits up to 4.7) | 100% | 3.17 s | 14.2 s | yes |

The timed loader copied the arrays to the GPU one at a time and the timed saver cast to bfloat16 on the host.
The shipped loader copies all arrays in one batched `jax.device_put`, and the shipped saver casts on the device
before one batched copy back (half the bytes); both pass the same tests but are not re-timed.

## Where we lose, and limits

- Load and save are slower than `transformers`, whose loader stays on the host and memory-maps the shards. The
  Qwen3-0.6B load spends 0.14-0.25 s mapping the tensors, 0.07-0.5 s tracing the model shapes, 0.26-0.44 s
  stacking the scanned layers, 0.9 s casting bfloat16 to float32 on the host and 2.0-2.4 s copying 2.4 GB to
  the GPU one array at a time; doing the casts, transposes and stacking on the device is not built yet.
- The importer builds the whole model on one device and holds up to three host copies while stacking and
  casting, so a model must fit on one accelerator (or import with `JAX_PLATFORMS=cpu` on a host with that much
  memory); sharded import onto a mesh is not implemented.
- Export refuses what the target family cannot express: sliding windows in Llama, mixed windows in Qwen3-MoE,
  DeepSeek-V3 routing (`router_bias`, expert groups, `routed_scaling_factor`) in Mixtral or Qwen3-MoE, RoPE
  scaling in Phi-3. Every import option without an exact equivalent raises.
- DeepSeek-V3's router GEMM runs on float32 operands (as `transformers` computes it) whatever `compute_dtype` is;
  `router_bias` is imported and applied but never updated in training (no aux-loss-free bias updates).

## Correctness

`tests/test_convert_hf.py` builds tiny random models with `transformers` 5.20.1 for 17 configurations (Llama with
`llama3` RoPE scaling and tied embeddings, Mistral with a sliding window, Qwen2 with biases, mixed windows and
YaRN, Qwen3, Phi-3 with partial rotary, Mixtral, Qwen3-MoE, DeepSeek-V3 with q LoRA, expert groups and YaRN, and
Mamba-2). For each, the imported model's float32 logits match `transformers` within 1e-4 (observed 1.1e-6 to
2.1e-6 on CPU and 9.5e-7 to 2.3e-6 on an H100) for the scanned and unscanned stack and for cached decoding;
export reproduces every original tensor bit for bit and `transformers` reloads it to identical logits. Negative
controls (dropping a RoPE scaling, the sliding window, partial rotary or a DeepSeek-V3 routing detail) move the
logits by 0.26 to 3.05.

## Reproduce

```bash
python benchmarks/bench_convert_hf.py --shape qwen3-0.6b --workdir /path/to/scratch --out results/convert_hf.json
#   --shape llama-3.2-1b for the second row
pytest tests/test_convert_hf.py
```

## How it works

Tensors are renamed and re-laid out (`[out, in]` to `[in, out]`, gate and up fused as `[D, 2, F]`, Phi-3's fused
projections split, MoE experts stacked, DeepSeek-V3's interleaved RoPE pairs permuted to the rotate-half layout,
Mamba-2's `in_proj` and `conv1d` split) and stacked for scanned layers; sharded safetensors are read and written by
a NumPy implementation of the format. `transformers`' MoE balance loss is `top_k` times ours, so import sets
`aux_loss_coef = top_k * router_aux_loss_coef` and export divides by `top_k`.
