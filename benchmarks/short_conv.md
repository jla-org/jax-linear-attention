# Short causal convolution: fused Mosaic GPU kernels

Mamba-2, Gated DeltaNet, GDN-2 and KDA run a depthwise causal convolution of width 4, followed by SiLU,
on the projection outputs before the recurrence (`jla.layers.conv.ShortConv`; Mamba-2 on x and on B/C,
the linear-attention mixers on q, k and v). [`jla.ops.mosaic_gpu.short_conv`](../src/jla/ops/mosaic_gpu/short_conv.py)
replaces XLA's version with two Mosaic GPU kernels (forward; backward with `dx`, `dweight`, `dbias`),
adapted from FLAT's depthwise causal convolution (Mayank Mishra, Apache-2.0,
`flat/depthwise_causal_convolution/pallas_implementation/{forward,backward}.py`, a TPU Pallas kernel),
and [`jla.ops.mosaic_gpu.short_conv_l2norm`](../src/jla/ops/mosaic_gpu/short_conv_l2norm.py) also folds
in the per-head q/k L2 norm of the delta-rule mixers. They are compared with XLA, the causal-conv1d
CUDA kernel and FLA's Triton convolution.

## Setup

- One H100 SXM, JAX 0.11.2; bf16 input `[8, 4096, C]` (B = 8 and T = 4096, the shape of FLAT's README
  measurements), width 4, SiLU, with a bias where the preset has one; the q/k norm at `[16, 4096,
  2048]` (16 heads of 128, no bias). "Packed" means documents of 1024 tokens.
- Baselines (torch 2.13.0+cu129): causal-conv1d 1.7.0, the official CUDA kernel, with the
  channel-last `[B, T, C]` input as Mamba-2 passes it and `seq_idx` for documents; FLA main `c6e4f00`
  (`fla.modules.causal_conv1d`, Triton backend, then `fla.modules.norm.l2norm` per head) with
  `cu_seqlens`.
- Timing: the median CUPTI kernel time of 30 calls after 3 warm-ups (`torch.profiler` for torch);
  `fwd+bwd` returns the output and the gradients of x, the weight and the bias. The floor is every
  tensor moved once at 3.35 TB/s (x and y forward; x, y, dy and dx forward + backward).
- Measured on 17 Oct 2026, each run the only timed job on its GPU while other jobs could share the
  machine, so the numbers are provisional.

## Results

Kernel time at the 1.3B shapes (ms):

| conv | C | docs | XLA fwd | **ours fwd** | causal-conv1d fwd | XLA fwd+bwd | **ours fwd+bwd** | causal-conv1d fwd+bwd | ours vs XLA (fwd / fwd+bwd) | ours vs causal-conv1d (fwd / fwd+bwd) | ours, fraction of floor (fwd / fwd+bwd) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mamba-2 x | 4096 | one per row | 0.335 | **0.204** | 0.207 | 1.498 | **0.562** | 0.726 | 1.65× / 2.66× | 1.02× / 1.29× | 0.79 / 0.71 |
| Mamba-2 x | 4096 | 1024 tokens | 0.361 | **0.222** | 0.283 | 1.693 | **0.589** | 1.002 | 1.63× / 2.87× | 1.27× / 1.70× | 0.72 / 0.68 |
| Mamba-2 B/C | 256 | one per row | 0.024 | **0.014** | 0.010 | 0.115 | **0.048** | 0.060 | 1.80× / 2.42× | **0.76×** / 1.26× | 0.74 / 0.53 |
| Mamba-2 B/C | 256 | 1024 tokens | 0.026 | **0.016** | 0.019 | 0.126 | **0.051** | 0.088 | 1.60× / 2.44× | 1.19× / 1.70× | 0.62 / 0.49 |
| GDN / GDN-2 / KDA q, k, v | 2048 | one per row | 0.163 | **0.108** | 0.102 | 0.701 | **0.291** | 0.356 | 1.51× / 2.41× | **0.95×** / 1.22× | 0.74 / 0.69 |
| GDN / GDN-2 / KDA q, k, v | 2048 | 1024 tokens | 0.181 | **0.114** | 0.142 | 0.852 | **0.301** | 0.511 | 1.59× / 2.83× | 1.25× / 1.70× | 0.71 / 0.67 |

Forward + backward is 2.4–2.9× faster than XLA and 1.2–1.7× faster than causal-conv1d at every shape;
the forward alone is 1.5–1.8× faster than XLA and ties or beats causal-conv1d except on dense rows at
2048 and 256 channels.

With the q/k L2 norm fused into the conv kernel, at the 1.3B q/k shape (ms):

| documents | path | fwd | fwd+bwd | fused speedup (fwd / fwd+bwd) |
|---|---|---|---|---|
| one per row | **fused** | **0.213** | **0.566** | – |
| one per row | conv kernel + XLA norm | 0.385 | 1.112 | 1.81× / 1.96× |
| one per row | FLA Triton conv + FLA `l2norm` | 0.401 | 1.686 | 1.88× / 2.98× |
| one per row | causal-conv1d CUDA + FLA `l2norm` | 0.380 | 1.328 | 1.78× / 2.35× |
| 1024 tokens | **fused** | **0.238** | **0.742** | – |
| 1024 tokens | conv kernel + XLA norm | 0.402 | 1.138 | 1.69× / 1.53× |
| 1024 tokens | FLA Triton conv + FLA `l2norm` | 0.400 | 1.912 | 1.68× / 2.58× |
| 1024 tokens | causal-conv1d CUDA + FLA `l2norm` | 0.459 | 1.531 | 1.93× / 2.06× |

Against the faster baseline at each shape the fused kernels are 1.78× forward and 2.35× forward +
backward on dense rows, and 1.68× and 2.06× with documents.

**1.3B training on one GPU** (`bench_e2e_1p3b.py --impl kernels`, 16 × 4096 tokens per step,
FineWeb-Edu with document masking, full remat, steps 6–20 timed; the two runs of a pair differ only
in `model.<mixer>.conv_impl`): `mamba2-1p3b` trains at 35,144 instead of 32,312 tok/s with the conv on
the kernels (**1.09×**) and `kda-1p3b` at 33,893 instead of 27,825 (**1.22×**), with final losses of
7.806 / 7.805 and 7.865 / 7.869 (XLA / kernels) and peak memory of 45.3 / 45.1 and 45.9 / 45.9 GiB.
Fusing the q/k norm into the conv saves a further 1.5–2.5% per median step on `gdn-hd128-1p3b`,
1.1–1.5% on `gdn2-1p3b` and 1.0–2.9% on `kda-1p3b`
(pairs in both orders; a Mamba-2 pair of identical code puts the noise at about 0.4%), and lowers the
peak memory by 1.3 GiB on KDA and 0.5 GiB on GDN-2.

## Where we lose, and limits

- **Dense forward against causal-conv1d at 2048 and 256 channels (0.95× and 0.76×).** Our forward
  reaches 74% of the HBM floor there. It loads rows straight from HBM into registers, so the registers
  and the resident CTAs bound the bytes in flight per SM: at 2048 channels it uses 120 registers per
  thread (4 CTAs per SM), and at 256 channels the grid itself is small (8 sequences × 64 blocks, 512
  CTAs of 64 tokens). With packed documents causal-conv1d slows by 37–90% (its `seq_idx` path) while
  our per-token mask costs 6–14%, so ours wins those cases by 1.19–1.27×. Staging the tiles through
  shared memory with TMA, as the FA3-style kernels do, is the next step.
- **Backward at 256 channels (0.49–0.53 of the floor):** 167 registers per thread at 2 channels per
  thread (3 CTAs per SM) on a 512-CTA grid, and the weight-gradient partials add about 10% to its HBM
  traffic. At 2048 and 4096 channels the backward reaches 0.67–0.71 of the floor.
- **The fused q/k norm with documents.** Its backward takes 0.498 ms against 0.358 ms for the plain
  conv backward (210 registers per thread, 2 CTAs per SM, where the dense one runs 3), so the fused
  forward + backward gains 1.53× with documents against 1.96× on dense rows. Gated DeltaProduct and
  the context-parallel and multi-device paths keep the unfused norm.
- **Decode.** A one-token decode step is one small kernel launch per conv; it is bit-identical to the
  full sequence but neither tuned nor timed.
- `gdn-hd128-1p3b` has no end-to-end pair of the conv on XLA against the conv on the kernels.

## Correctness

[`tests/test_mosaic_gpu_short_conv.py`](../tests/test_mosaic_gpu_short_conv.py) and
[`tests/test_mosaic_gpu_short_conv_l2norm.py`](../tests/test_mosaic_gpu_short_conv_l2norm.py) (GPU)
compare the kernels in bf16 with an f32 oracle on the same values: for the output, the new decode state
and the gradients of x, the weight, the bias and the initial state, the error stays within a few bf16
ulps of the largest value and within 1.5× the error of the unfused bf16 path (for the fused norm,
within the larger of the two), over widths 2–4, packed and reused segment ids, initial states, block
edges from `T` = 1 to 191, the 1.3B presets' channel counts and float16; the new state is
bit-identical, prefill plus token-by-token decode is bit-identical
to the full sequence, no document leaks into another, and one layer of `mamba2-1p3b`, `kda-1p3b` and
`gdn-hd128-1p3b` gives the same loss and parameter gradients on either path.
[`tests/test_short_conv_impl.py`](../tests/test_short_conv_impl.py) and
[`tests/test_conv_l2norm_plumbing.py`](../tests/test_conv_l2norm_plumbing.py) check the plumbing on any
backend.

## Reproduce

```bash
# kernels vs XLA vs causal-conv1d (the torch side runs in an environment with causal-conv1d)
python benchmarks/bench_short_conv.py --batch 8 --json results/short_conv.json \
    --torch-python /path/to/venv-with-causal-conv1d/bin/python
# fused q/k norm vs conv kernel + XLA norm at [16, 4096, 2048], with FLA's prologue as well
python benchmarks/bench_short_conv_l2norm.py --batch 16 --json results/conv_l2norm.json \
    --torch-python /path/to/venv-with-fla-and-causal-conv1d/bin/python
# 1.3B training, conv on XLA and on the kernels; then the norm in the op and in the conv kernel
python benchmarks/bench_short_conv_e2e.py --arch kda-1p3b --data /path/to/fineweb_edu_3B \
    --order mosaic_gpu,xla --json results/short_conv_e2e.json
python benchmarks/bench_short_conv_l2norm_e2e.py --arch gdn-hd128-1p3b --data /path/to/fineweb_edu_3B \
    --order off,on --json results/conv_l2norm_e2e.json
pytest tests/test_mosaic_gpu_short_conv.py tests/test_short_conv_impl.py \
    tests/test_mosaic_gpu_short_conv_l2norm.py tests/test_conv_l2norm_plumbing.py
```

## How it works

Each CTA is one warpgroup that owns `128 · vec` channels of one sequence and a block of tokens; every
thread streams its `vec` contiguous channels along time, keeps the last `K - 1` inputs in registers,
and loads and stores straight between HBM and registers, with a group of rows loaded ahead of its use.
The backward follows FLAT's decomposition (the SiLU gradient once in f32 from the recomputed
pre-activation, `dx` as the sum of the gradient taps shifted forward in time, `dweight` and `dbias` as
per-block f32 partials that XLA adds), but every block recomputes its `K - 1` future rows instead of
carrying them, so all blocks run in parallel; it prefetches the next row group and picks the longest
block (256, 128 or 64 tokens) whose grid still has at least 3 CTAs per SM. `segment_ids` become one
int32 mask per token (bit `l` set when the token `l` steps back shares the document). In the fused
norm a 128-channel head is held by the 32 threads of one warp, whose sum of squares is a 5-step
butterfly shuffle; the output `y * rsqrt(sum + 1e-6)` is rounded to bf16 once, so it is more accurate
than the unfused path, which rounds the conv output first.

`ShortConv(..., impl=...)` takes `"xla"`, `"mosaic_gpu"` (raises on calls the kernels do not support)
or `"auto"`; the mixers pass `LinearAttentionConfig.conv_impl` and `SSMConfig.conv_impl`, which
default to `"auto"`: the kernels on Hopper for bf16 or f16 inputs, widths 2–4 and channel counts
divisible by 128, XLA anywhere else. Under FSDP or data parallelism `ShortConv` runs them per shard
(`short_conv_sharded`, through `jla.distributed.kernel_sharding.shard_layout_call`).
`LinearAttentionConfig.qk_l2norm_in_conv` (default `True`) folds the q/k norm into the conv kernel for
the delta-rule kinds at head dims 64 and 128 when a head fits one warp (channels a multiple of 512 at
head dim 128 and of 256 at head dim 64), and the recurrent op then runs with `use_qk_l2norm=False`.
