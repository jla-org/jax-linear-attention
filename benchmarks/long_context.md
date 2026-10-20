# Long contexts: single calls past 2^31 elements

A kernel that forms `row * row_stride` (or any flattened element index) in 32-bit arithmetic reads or writes the
wrong memory once one tensor passes 2^31 elements, and returns NaN or finite wrong values with no error.
state-spaces/mamba 2.3.2's Triton Mamba-3 kernels, for example, address their inputs as `row * stride_x_row` with a
32-bit `row`: with a 2384-wide in_proj row (Mamba-3, d_model 512, 16 heads of 64, d_state 128), every single call
longer than 900,790 tokens reads wrapped memory from token 900,791 on. This page lists what the release kernels were
audited and tested for, the single-call limits, and the results of one-call runs of 1,064,960 tokens.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 inputs. Every release kernel (Mamba-2 / simple GLA, Mamba-3 SISO with the fused
  angle and rotary kernels, Mamba-3 MIMO, Gated DeltaNet, DeltaNet, GDN-2, KDA, RWKV-7, the FA3-style causal and
  sliding-window attention, the fused LM-head cross entropy) and the XLA glue around it.
- `tests/test_long_context.py` (opt-in) makes one call of `T` = 1,064,960 tokens (2^20 + 2^14): with 16 heads of 128
  every `q, k, v` and output tensor has 2,181,038,080 > 2^31 elements, and the 2^31-element boundary falls at token
  1,048,576. Each one-call result is compared with the same op over two halves of 532,480 tokens chained through the
  final / initial state (every tensor of a half stays below 2^31 elements), and with the float32 token-by-token
  reference on three 256-token windows (the start, across the 2^31 boundary, the end), each started from a state
  computed below the limit. Errors are relative to the largest reference magnitude.
- Measured on 17 Oct 2026 on one H100 with no other benchmark on it; the release code's opt-in suite was rerun on 18
  Oct 2026 as 8 jobs of one H100 each (provisional). Every output was finite.

## Results

What the audit covered:

- **32-bit offsets in kernels**: none. Every kernel moves global memory with TMA, whose descriptors Mosaic GPU builds
  on the host from 64-bit sizes and strides, with one int32 coordinate per dimension; a folded coordinate stays below
  2^31 unless `B * T * H` itself passes it (about 2^27 tokens at 16 heads). TMA does not bounds-check, so a wrapped
  coordinate would be silent.
- **The XLA glue**: no flattened gathers, scatters or `arange`; gathers and scatters index per dimension.
- **Cumulative sums**: every log-decay cumsum is chunk-local, and the fused Mamba-3 angle kernel wraps the carried
  angle to [-pi, pi] after every chunk.
- **Segment ids**: the Mamba-3 SISO, fused and MIMO kernels compare chunk-local run ids (global run ids stored as
  float32 would be exact only below 2^24 documents); the other kernels use chunk-local or int32 ids, or reset by
  decay.
- **cuDNN's attention backward** (not ours; the `"auto"` path below 2^31): cuDNN 9.27 as shipped with JAX 0.11.2
  returns wrong `dq` and `dk` once one call's query tensor has more than 2^31 elements (`dq` from token 1,048,576 on,
  a 32-bit element offset), while its forward and `dv` stay right. Past 2^31 query elements per device `"auto"`
  picks the Mosaic GPU kernel on Hopper (XLA elsewhere), and an explicit `"cudnn"` raises when differentiated.
- **RoPE angles** `pos * inv_freq` in fp32 lose precision with the position: against float64 (theta 1e4, head dim
  128) the rotation between neighbouring tokens is off by 0.0018 rad at [2^14, 2^15), 0.062 rad at [2^19, 2^20),
  0.125 rad at [2^20, 2^21) and 0.25 rad at [2^21, 2^22). `AttentionConfig.rope_exact_angles` (`apply_rope(...,
  exact_angles=True)`) stays within about 1e-3 rad at any int32 position; the default stays the fp32 product, for
  checkpoint parity with Hugging Face.

Single-call limits (CUDA caps grid y and z at 65,535 blocks; a longer launch fails loudly with
`CUDA_ERROR_INVALID_VALUE`, and the `supports()` predicates reject such calls, except the `K = 256` Gated
DeltaNet kernels', below):

| kernel | longest single call | beyond it |
|---|---|---|
| Mamba-3 SISO recurrence, Mamba-3 MIMO, RWKV-7, fused cross entropy | no chunk limit | n/a |
| Mamba-3 fused angle + rotary | 4,194,240 tokens (65,535 chunks) | the layer runs the XLA glue around the unfused `mamba3` kernel |
| simple GLA (Mamba-2), Gated DeltaNet, DeltaNet | 65,535 chunks after padding to `64 * pieces`: about 4.19M tokens | `impl="auto"` uses the chunked XLA op; `impl="mosaic_gpu"` raises |
| KDA, GDN-2 | 65,535 / max(1, min(K, V) / 64) chunks: 2,097,088 tokens at K = V = 128, 4,194,240 at 64 | as above |
| FA3-style attention (causal, SWA, documents) | 4,194,176 tokens | `flash_attention` raises; chain such calls or shard the sequence |
| cuDNN attention (not ours) | 2^31 query elements per device: wrong gradients past it | `"auto"` picks the Mosaic GPU kernel |

Longer sequences run as calls chained through `initial_state` / `output_final_state`.

One call of 1,064,960 tokens:

| op (heads x K, V) | one call vs two chained halves | vs float32 reference (start / across 2^31 / end) |
|---|---|---|
| simple GLA, the Mamba-2 SSD core (16 x 128, 128) | identical outputs and final state | 7.3e-3 / 3.9e-3 / 3.7e-3 |
| Gated DeltaNet (16 x 128, 128) | identical outputs and final state | 6.6e-3 / 6.6e-3 / 6.0e-3 |
| DeltaNet (16 x 128, 128) | 5.5e-3 max, final state 3.7e-3 (the sequence split places its pieces by length) | 5.0e-3 / 7.4e-3 / 7.2e-3 |
| KDA (16 x 128, 128) | identical outputs and final state | 6.8e-3 / 6.7e-3 / 5.9e-3 |
| Mamba-3 SISO (16 x 128, 64) | first half identical, second 1.8e-3 max (4e-8 mean), final state identical | 5.7e-3 / 5.7e-3 / 2.7e-3 |
| Mamba-3 SISO layer core, fused angle / rotary kernels (32 heads, K 128, V 64) | first half identical, second 7.2e-4 max, final angle 1.9e-6 | 4.5e-3 / 3.6e-3 / 4.2e-3 |
| Mamba-3 MIMO, dedicated kernels (8 heads, rank 2) | first half identical, second 1.6e-3 max (4e-8 mean), final state identical | 4.7e-3 / 5.4e-3 / 4.4e-3 |
| FA3-style attention, causal (16 / 16 heads x 128) | n/a | 2.5e-3 / 3.6e-3 / 3.1e-3 |
| FA3-style attention, sliding window 4096 | second half recomputed from its window: identical | 2.5e-3 / 3.2e-3 / 2.3e-3 |

The attention errors are the bf16 rounding of the output, the same as cuDNN's forward. The GDN-2 forward passes too
(the KDA and GDN-2 one-call forwards: worst float32-window error 5.9e-3 to 7.6e-3). An SWA attention layer (d_model
2048, window 4096) is identical to chunked recomputation, and a Mamba-3 layer at 900,790, 900,800 and 1,064,960
tokens matches a two-chunk prefill within 3.5e-3 max (at most 8.6e-6 mean); the fused LM-head cross entropy (N =
1,064,960, D = 2048, V = 32,768) gives `dhidden` identical to two halves and the loss within 8.7e-8.

Gradients, one call against two chained halves: simple GLA and Gated DeltaNet identical, Mamba-3 SISO at most 2.6e-3
max (4e-8 mean), DeltaNet at most 1.1e-2 max (`dk`, 3.7e-3 mean, the bf16 rounding of its length-placed split).
Attention (16 query / 4 KV heads x 128) against float32 on windows of 256 query rows:

| backend | `dq` start / across 2^31 / end | `dk`, `dv` of the last 256 keys |
|---|---|---|
| ours (FA3-style Mosaic GPU kernel) | 5.7e-3 / 3.1e-3 / 4.3e-3 max (2.0e-3 to 2.1e-3 mean) | 6.1e-3, 4.1e-3 max |
| cuDNN 9.27 | 5.7e-3 / **3.5** / **2.0** max (**3.8** / **1.0** mean) | **0.12**, 4.1e-3 max |

cuDNN agrees with our kernel within 3.2e-3 at every tested length from 131,072 to 1,048,576 tokens (exactly 2^31
query elements), and at 1,052,672 its `dq` is off from token 1,048,576 on.

## Where we lose, and limits

- Not measured past 2^31 elements: the one-call gradients of KDA, GDN-2 and RWKV-7 and the RWKV-7 one-call forward
  run out of memory inside the kernel executables on one H100 (KDA one 66.5 GiB buffer, GDN-2 62.1 GiB, RWKV-7
  174.9 GiB; the RWKV-7 forward 37.1 GiB of temporaries). They need a smaller per-chunk state in the backward, or
  chained calls.
- The `K = 256` Gated DeltaNet kernels have no test past 2^31 elements yet. Two of them put the chunk index on a grid
  axis capped at 65,535 blocks (`gdr_kkt_fwd`, `gdr_kb_chunk_grads`), so one differentiated call is limited to
  4,194,240 tokens, and `scalar_gate_kb`'s `supports()` does not check that limit yet: chain longer sequences through
  `initial_state`.
- With the default RoPE (fp32 angles), long positions lose precision (above); set `AttentionConfig.rope_exact_angles`
  for long contexts.

## Correctness

This page is a correctness check: `tests/test_long_context.py` (opt-in, one H100, up to about 64 GB) runs every
comparison above, `tests/test_mosaic_gpu_grid_limits.py` (default GPU suite, shapes only) pins the `supports()`
limits, the chunk-local Mamba-3 run ids (2^24 + 128 one-token documents) and the cuDNN guard per device, including a
data-parallel `jit(grad(attention))` on two GPUs, and `tests/test_rope_exact_angles.py` the exact RoPE angles.

## Reproduce

```
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  pytest tests/test_long_context.py --run-long-context -s -rfE
pytest tests/test_mosaic_gpu_grid_limits.py tests/test_rope_exact_angles.py
```
