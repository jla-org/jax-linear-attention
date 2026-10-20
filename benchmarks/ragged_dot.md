# Grouped GEMM (ragged dot) for MoE on Mosaic GPU

`jla.ops.mosaic_gpu.ragged_dot` is a differentiable Mosaic GPU grouped GEMM with the semantics of
`jax.lax.ragged_dot`: `lhs [M, K]` rows sorted by group, `rhs [G, K, N]`, `group_sizes [G]`. Dropless
MoE layers use it by default on an H100 (`MoEConfig.ragged_dot_impl = "auto"`), and move tokens into
and out of expert order with gathers in both directions. This page compares the kernel with
`torch._grouped_mm` (CUTLASS) and `jax.lax.ragged_dot`, and the whole MoE layer with the same layer in
PyTorch, eager and under `torch.compile`.

## Setup

- One H100, JAX 0.11.2, bf16 operands with f32 accumulation; group sizes from top-k routing of random
  router logits (uneven; some experts can be empty).
- Baselines: `torch._grouped_mm` (PyTorch 2.13, its CUTLASS sm90 grouped GEMM) with the faster of its
  two weight layouts (`[G, K, N]` or `[G, N, K]`); our own PyTorch port of the dropless layer (router,
  top-k, sort, `torch._grouped_mm` up, SwiGLU x gate, `torch._grouped_mm` down, unsort and sum, shared
  expert), eager and under `torch.compile`; and `jax.lax.ragged_dot`, which on GPU computes a dense
  product per group.
- Timing: median CUPTI kernel time per call over 30 calls after 3 warmups (`benchmarks/harness.py`);
  MoE layers forward + backward over 2 or 3 rotated rounds of 20 or 30 calls, the torch layers right
  after them on the same GPU. TFLOP/s count the useful `2 routes K N` per GEMM (3x that for forward +
  backward).
- Measured on 17-19 Oct 2026 while other jobs' tests could share the machine, so every number on this
  page is provisional.

## Results

### Per GEMM against `torch._grouped_mm`

The default `PERSISTENT` tiling, the shipped presets' expert GEMMs at 16,384 tokens and two more
shapes; ours / torch in ms, and torch's time over ours (> 1 means ours is faster):

| shape (tokens, `d_model`, experts, top-k, `d_ff`) | up fwd | up fwd+bwd | down fwd | down fwd+bwd |
|---|---|---|---|---|
| `qwen3-next-300m` (16384, 1024, 64, 10, 256) | 0.2956 / 0.3016, 1.02x | 1.0150 / 0.9584, 0.94x | 0.2448 / 0.2226, 0.91x | 0.6483 / 0.6294, 0.97x |
| `kimi-linear-1b` (16384, 1024, 32, 4, 512) | 0.2349 / 0.2359, 1.00x | 0.7335 / 0.7331, 1.00x | 0.1492 / 0.1470, 0.99x | 0.4352 / 0.4509, 1.04x |
| `granite-4.0-h-125m` (16384, 768, 64, 6, 256) | 0.1564 / 0.1548, 0.99x | 0.4899 / 0.4960, 1.01x | 0.1228 / 0.1102, 0.90x | 0.3342 / 0.3386, 1.01x |
| `gpt-oss-180m` (16384, 768, 32, 4, 768) | 0.2859 / 0.2690, 0.94x | 0.8028 / 0.7898, 0.98x | 0.1542 / 0.1492, 0.97x | 0.4604 / 0.4569, 0.99x |
| 64 experts, top-8, `d_ff` 512 (8192, 768, 64, 8, 512) | 0.2002 / 0.2022, 1.01x | 0.6069 / 0.5936, 0.98x | 0.1236 / 0.1248, 1.01x | 0.3486 / 0.3607, 1.03x |
| Qwen3-Next-80B EP8 shard (4096, 2048, 64 local, 10, 512) | 0.2934 / 0.3027, 1.03x | 0.9917 / 0.9438, 0.95x | 0.2000 / 0.1967, 0.98x | 0.5591 / 0.5434, 0.97x |

Over these six shapes our grouped GEMMs are 0.90-1.04x `torch._grouped_mm` (forward + backward
0.94-1.04x), with 585 TFLOP/s on the `kimi-linear-1b` up forward. The whole dropless layer forward
+ backward takes 3.181 ms (`qwen3-next-300m`) and 2.320 ms (`kimi-linear-1b`), against 4.407 and
2.983 ms for the PyTorch port under `torch.compile` (**1.39x** and **1.29x**) and 7.415 and 5.269 ms
eager.

### The MoE layer of each shipped preset

`benchmarks/profile_moe.py` builds each preset's MoE layer as the preset configures it and times its
forward + backward on 16,384 bf16 tokens (ms), with the grouped GEMMs' share in parentheses. Every
column is an option that still exists, from the slowest to the default: XLA's `ragged_dot` for
dropless routing (`ragged_dot_impl = "xla"`), the kernels with the scatter-add token permutation
(autodiff's transposes of plain indexing, which pipeline stages still use) and the `BLOCK_ALIGNED`
tiling, the gather-only permutation, and the `PERSISTENT` tiling:

| preset (experts, top-k, `d_ff`; routing) | XLA `ragged_dot` | scatter-add, `BLOCK_ALIGNED` | gathers, `BLOCK_ALIGNED` | gathers, `PERSISTENT` (default) | `PERSISTENT` / `BLOCK_ALIGNED`: layer / GEMMs | capacity 1.25: scatter-add / gathers |
|---|---|---|---|---|---|---|
| `qwen3-next-300m` (64, 10, 256; dropless) | out of memory | 6.682 | 3.776 (2.080) | **3.229** (1.519) | 1.17x / 1.37x | 7.478 / 3.398 |
| `kimi-linear-1b` (32, 4, 512; capacity 1.25) | - | 3.834 | 2.628 (1.401) | **2.310** (1.074) | 1.14x / 1.30x | 4.045 / 2.387 |
| `granite-4.0-h-125m` (64, 6, 256; dropless) | out of memory | 3.486 | 2.152 (1.042) | **1.896** (0.776) | 1.14x / 1.34x | 3.798 / 1.990 |
| `gpt-oss-180m` (32, 4, 768; dropless) | out of memory | 3.495 | 2.588 (1.573) | **2.222** (1.194) | 1.16x / 1.32x | 3.530 / 2.327 |
| `minimax-01-1b` (32, 2, 768; dropless) | 23.955 | 2.043 | 1.584 (0.879) | **1.390** (0.675) | 1.14x / 1.30x | 1.982 / 1.408 |
| `moe-125m` (16, 2, 768; capacity 1.25) | - | 2.194 | 1.727 (0.792) | **1.563** (0.618) | 1.10x / 1.28x | 2.222 / 1.613 |

"Out of memory": XLA's dense per-group products at a 0.25 memory fraction (20 GB); on `minimax-01-1b`
XLA takes 17.2x the default's time. The gather-only permutation makes the dropless layers 1.27-1.77x
faster on the same kernel, the `PERSISTENT` tiling its GEMMs 1.28-1.37x faster, and together the
dropless layer is 1.40-2.07x faster than with the scatter-add permutation and `BLOCK_ALIGNED`. With
`PERSISTENT` the dropless layer (no dropped routes) is also faster than the capacity-1.25 einsum layer
on every preset. The opt-in tilings `ALIGNED` and `PERSISTENT_WS` measure between `BLOCK_ALIGNED` and
`PERSISTENT` on every preset.

Train steps on one H100 (`benchmarks/bench_end_to_end_step.py`, AdamW, bf16, synthetic tokens, 2
sequences of 2048 tokens per step, 10 timed steps after 2 warmups, a 0.45 memory fraction):

| preset | layer with the scatter-add permutation, preset's `ragged_dot_impl` (`"xla"`) | the same on the kernels (`BLOCK_ALIGNED`) | default layer, against the faster of the other two |
|---|---|---|---|
| `qwen3-next-300m` (dropless) | out of memory (single 26.8 and 24.3 GiB requests in two runs) | 88.22 ms (46.4k tok/s) | **71.11 ms (57.6k tok/s), 1.24x** |
| `kimi-linear-1b` (capacity 1.25) | 76.76 ms (53.4k tok/s) | (the same path) | **71.61 ms (57.2k tok/s), 1.07x** |
| `granite-4.0-h-125m` (dropless) | 232.04 ms (17.7k tok/s) | not run | **35.83 ms (114.3k tok/s), 6.48x** |

The losses at step 10 agree: 4.8444 against 4.8458, 7.0192 against 7.0170, and 8.0693 for both.

Against `jax.lax.ragged_dot` itself (moe-125m-like layer: 8192 tokens, `d_model` 768, 64 experts,
top-8; 17 Oct 2026, the kernel's `BLOCK_ALIGNED` tiling; ms):

| `d_ff` | up fwd: XLA / ours, speedup | down fwd: XLA / ours, speedup | MoE layer fwd+bwd: capacity 1.25 einsum / dropless on the kernels |
|---|---|---|---|
| 256 | 5.746 / 0.155, 37.1x | 2.518 / 0.123, 20.5x | 2.464 / 2.263 (0.92x the time) |
| 512 | 9.124 / 0.300, 30.4x | 5.010 / 0.185, 27.1x | 3.086 / 3.093 (1.00x) |
| 1024 | 17.538 / 0.591, 29.7x | XLA out of memory (8.0 GiB request) / 0.280 | 4.341 / 4.728 (1.09x) |

XLA computes every group densely, so these ratios say more about that baseline than about the kernel.
At a DeepSeek-V3-like EP8 shard (4096 tokens, `d_model` 7168, 32 local experts, top-8, `d_ff` 2048) the
down-projection forward takes 2.138 ms against XLA's 36.497 ms (17.1x), and the up-projection forward
3.861 ms (XLA ran out of memory while autotuning).

### The hybrid presets' MoE layers

The MoE layers of `qwen3-next-1p3b` (64 experts, top-10, `d_ff` 256, `d_model` 1024) and
`kimi-linear-1p3b` (32 experts, top-8, `d_ff` 512), forward + backward on 16,384 tokens on the
default path (19 Oct 2026):

| MoE layer | forward + backward, ms | grouped GEMMs | everything else | largest other kernels |
|---|---|---|---|---|
| `qwen3-next-1p3b` | 3.235 | 1.523 (47%) | 1.712 | 0.442, 0.287, 0.177 |
| `kimi-linear-1p3b` | 3.828 | 2.029 (53%) | 1.799 | 0.367, 0.280, 0.240 |

In the `qwen3-next-1p3b` layer the three largest other kernels are XLA fusions: the dispatch gather
with the combine's backward gather (0.442 ms, about 1.5 TB/s), the combine's forward sum over each
token's 10 routes with the dispatch's backward sum (0.287 ms), and the SwiGLU backward with the gate.
The rest is the SwiGLU forward (0.16 ms), the float32 <-> bf16 casts of the expert weights (0.21 ms per
call), routing (0.072 ms) and the shared expert. In the training step these layers are 25.7%
(`qwen3-next-1p3b`; 29.2% with the forward-gradient cross entropy) and 27.7% (`kimi-linear-1p3b`;
30.5%) of the kernel time ([end_to_end_1p3b.md](end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts)).

## Where we lose, and limits

- **Per GEMM against `torch._grouped_mm`**: the `qwen3-next-300m` down projection (contraction 256:
  0.91x forward, 0.97x forward + backward) and its up projection's backward (0.94x), and the
  `granite-4.0-h-125m` down forward (0.90x). The transposed kernel that computes `d_rhs` still fills and
  drains a pipeline per tile (it runs at 469 TFLOP/s on `kimi-linear-1p3b`'s up projection, 1.20 ms per
  call at 32,768 tokens).
- **The padded einsum** of the capacity path (cuBLAS, 25% more rows) is faster per GEMM than both
  grouped GEMMs, but drops the overflow routes.
- **Gathers outside the GEMM.** The two gather fusions are 0.73 ms of the 3.24 ms `qwen3-next-1p3b`
  layer (23%; about 6% of its step). Mosaic GPU in JAX 0.11.2 gathers rows in an async copy only through
  TMA `gather4` (Blackwell), so on an H100 a gather fused into the GEMM's prologue needs per-row
  register loads, and the combine a scatter-add of 10 routes per token; that kernel is not built.
- **FP8 expert GEMMs are not used yet.** `jla.ops.mosaic_gpu.ragged_dot_fp8` (e4m3 or e5m2 x e4m3,
  weights as `[G, N, K]`) passes its tests, and its forward is 1.32-1.47x the bf16 kernel at the hybrid
  layers' shapes (16,384 tokens, 3 rotated rounds of 30 calls):

  | GEMM (routes x K x N) | bf16 `PERSISTENT` | FP8 e4m3 | speedup |
  |---|---|---|---|
  | `qwen3-next-1p3b` up (163,840 x 1024 x 512) | 0.296 ms (580 TFLOP/s) | 0.205 ms (836) | 1.44x |
  | `qwen3-next-1p3b` down (163,840 x 256 x 1024) | 0.246 ms (349) | 0.185 ms (463) | 1.33x |
  | `kimi-linear-1p3b` up (131,072 x 1024 x 1024) | 0.442 ms (622) | 0.301 ms (914) | 1.47x |
  | `kimi-linear-1p3b` down (131,072 x 512 x 1024) | 0.282 ms (488) | 0.214 ms (642) | 1.32x |

  The GEMMs FP8 can take (the forward, its rematerialization and `d_lhs`) are about 70% of the grouped
  GEMMs, which are 13.3% and 15.7% of these steps, so FP8 would save about 2.5-3% of the step before its
  quantization passes. Wiring it into the MoE layer needs the delayed-scaling state of `jla.layers.fp8`
  per expert GEMM and a loss check.
- **Several GPUs.** Expert parallelism's all-to-all does not overlap the expert GEMMs; with the
  per-device dispatch, FSDP trains the hybrid presets 3.3x faster per GPU than expert parallelism over
  8 GPUs ([end_to_end_1p3b.md](end_to_end_1p3b.md#qwen3-next-and-kimi-linear-layouts), [expert_parallel.md](expert_parallel.md)). Inside a
  pipeline stage the permutation keeps the scatter-add transposes (XLA's SPMD partitioner in JAX 0.11.2
  aborts on the gather-only ones there); that path was not timed on GPU.
- **The layer baseline** is our own PyTorch port: Megatron-Core's and TransformerEngine's MoE layers
  were not run, so the comparison bounds what PyTorch's compiler does with the same algorithm.

## Correctness

`tests/test_mosaic_gpu_ragged_dot.py` compares values and both gradients with a float32 oracle
(per-group dense matmuls) for every tiling: uneven groups, empty first, middle and last groups, one
group, 64 small groups, rows in no group, float16, a 64-expert top-8 routing, and larger cases in which
every SM runs several tiles, checked against an exact per-group float64 oracle (output, both gradients,
exact zeros). `tests/test_moe.py`, `tests/test_moe_permute.py` (the gather permutations against plain
indexing, and the layer against the scatter formulation, with dropped routes),
`tests/test_moe_per_device.py`, `tests/test_expert_parallel.py` and
`tests/test_mosaic_gpu_ragged_dot_fp8.py` (against a float64 oracle) cover the layer and the FP8 kernel,
and `tests/test_moe_two_gpus.py` runs the per-device dispatch and expert parallelism on two H100s:
outputs bit-identical to one device, gradients within 0.41-0.63% (bfloat16 sums in another order).

## Reproduce

```
python benchmarks/bench_ragged_dot.py --iterations 30 --layer-rounds 3 --no-xla --torch-python TORCH_VENV/bin/python \
    --shapes qwen3-next-300m,kimi-linear-1b --out results/ragged_dot.json
python benchmarks/bench_ragged_dot.py --iterations 30 --no-layer --no-xla --torch-python TORCH_VENV/bin/python \
    --shapes granite-4.0-h-125m,gpt-oss-180m,moe125m-like-f512,qwen3-next-80b-ep8 --out results/ragged_dot_more.json
python benchmarks/bench_ragged_dot.py --shapes <shape> --no-xla-bwd --out results/ragged_dot_xla.json   # vs jax.lax.ragged_dot
python benchmarks/profile_moe.py --tokens 16384 --out results/moe_profile.json
python benchmarks/profile_moe.py --presets qwen3-next-1p3b,kimi-linear-1p3b --tokens 16384 \
    --only dropless_mosaic_gpu --out results/moe_profile_hybrids.json
python benchmarks/bench_ragged_dot_fp8.py --out results/ragged_dot_fp8.json
python benchmarks/bench_end_to_end_step.py benchmarks/end_to_end/qwen3-next-300m.toml --steps 10 --warmup 2 \
    --set train.micro_batch_per_replica=2 [--set model.preset="kimi-linear-1b"] --json results/moe_step.json
```

## How it works

The kernels are adapted from the JAX 0.11.2 Pallas examples `ragged_dot_mgpu.py` and
`transposed_ragged_dot_mgpu.py` (Apache-2.0). The forward is a persistent kernel over (row tile,
column tile) pairs: 128 x 128 tiles with 64-wide contraction blocks and one TMA pipeline per SM over
all of its (tile, k) steps, so the next tile's loads are in flight during the current tile's epilogue
(CUTLASS's persistent design). Its rows follow Tokamax's aligned schedule (Apache-2.0): each group's
row tiles start at its first row rounded down to 8, found from a `[5, G]` per-group table.
`d_lhs` is the same kernel with the transposed weights read straight from `[G, K, N]`, and `d_rhs` the
transposed ragged dot, which masks the neighbouring groups' rows in each group's first and last row
block and leaves empty groups at zero. `supports` needs an H100, `M >= 128`, `K` and `N` multiples of
128 and at most 512 groups; `ragged_dot_impl = "auto"` uses `jax.lax.ragged_dot` elsewhere (other
devices, decoding), `"mosaic_gpu"` does the same with a warning, and `config=ragged_dot.BLOCK_ALIGNED`
(or `ALIGNED`, `WARP_SPECIALIZED`, `PERSISTENT_WS`) selects another tiling. In the layer,
`jla.ops.moe_permute`'s `dispatch_rows` and `combine_rows` are each other's transpose (`custom_vjp`),
so both directions are gathers plus a float32 sum over a token's `k` routes, with no scatter-add
atomics; under a data or FSDP mesh each device dispatches its own tokens inside a `shard_map`
(`MoE._dropless_per_device`).
