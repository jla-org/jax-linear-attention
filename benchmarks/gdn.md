# Gated DeltaNet and DeltaNet

`jla.ops.linear_attention("gated_deltanet" | "deltanet", ..., impl="mosaic_gpu")` runs Mosaic GPU kernels in
the style of FlashQLA's Hopper kernels for `K, V` in {64, 128}, with an exact sequence split when a call has
too few heads to fill the GPU, and K-blocked kernels for `K = 256` (FLA's 1.3B head shape), forward and
backward. This page compares them with FlashQLA main, with FLA main's Triton kernels and its default
dispatch (which sends Gated DeltaNet at `K = V = 128` to FlashQLA), and with MaxText's pure-JAX op. Simple
GLA and the Mamba-2 SSD core run on the same kernels ([mamba2.md](mamba2.md)).

## Setup

- H100 80GB HBM3, JAX 0.11.2; bf16 `q, k, v` (unit-norm `k`, scale `K^-0.5`), float32 gates
  `g = logsigmoid(N(0, 1))` and `beta = sigmoid(N(0, 1))`, no initial state.
- FlashQLA main `5538c35` (`0.1.3+git5538c35`, TileLang 0.1.15): `flash_qla.chunk_gated_delta_rule` with its
  automatic intra-card CP (its default); it takes `K = V = 128` only.
- FLA main `6dff3e1` (torch 2.13.0+cu129, Triton 3.7.1): its Triton kernels, and its default dispatch with
  FlashQLA `5538c35`, TileLang 0.1.15 and FlashKDA `7afb9f4` installed, which runs FlashQLA at `K = V = 128`;
  at `K = V = 64` its forward matches the Triton path and its backward is slower. FLA main `9d478fa` changes
  no Gated DeltaNet code.
- MaxText main `34ffea83`: `jax_chunk_gated_delta_rule`, unchanged since our copy (`ae6e4eb2`) in
  `benchmarks/maxtext_gdn.py`.
- CUPTI kernel time for JAX and `torch.profiler` device time for torch, round-robin, 2 rounds of 30 calls after
  3 warm-ups (64K-token rows: 1 round of 12). Cells: the median over rounds of the per-round median, fastest
  call in parentheses, ms; ratios compare fastest calls, baseline time / ours (> 1: ours is faster).
  `fwd+bwd` is one forward plus the backward for a fixed output cotangent.
- Measured on 20 Oct 2026 with no other jobs on the GPUs.

## Results

At five dense shapes (`(1, 4096, 8)`, `(4, 4096, 16)`, `(1, 16384, 16)` and `(8, 2048, 32)` at
`K = V = 128`, and `(4, 4096, 16)` at `K = V = 64`), ours is 0.79-1.69x forward and 0.73-1.26x forward +
backward against FlashQLA main (at the four with `K = V = 128`), and 1.74-3.14x and 1.71-2.68x against FLA
main's Triton kernels. In a length sweep (`T` 2K-64K, one row of 8 heads, `K = V = 128`) ours is 0.71-0.87x
forward and 0.69-0.78x forward + backward against FlashQLA from 16K tokens; with 16 packed documents in the
row, 0.80-0.87x and 0.74-0.77x from 16K tokens and 0.45-0.54x and 0.58-0.60x at 2K-8K. The tables below, a
separate run, add the long rows and packed documents where the sequence split runs. **Dense** (at `K = V = 64` FlashQLA does not run):

| B, T, H, K, V | pass | ours | FlashQLA main | FLA main default | FlashQLA / ours | FLA default / ours |
|---|---|---|---|---|---|---|
| 8, 2048, 32, 128, 128 | fwd | 0.336 (0.334) | 0.361 (0.356) | 0.360 (0.356) | **1.07x** | 1.06x |
| 8, 2048, 32, 128, 128 | fwd+bwd | 1.619 (1.609) | 1.361 (1.342) | 1.366 (1.343) | **0.83x** | 0.83x |
| 1, 4096, 8, 128, 128 | fwd | 0.077 (0.076) | 0.129 (0.128) | 0.129 (0.129) | **1.68x** | 1.69x |
| 1, 4096, 8, 128, 128 | fwd+bwd | 0.253 (0.250) | 0.261 (0.258) | 0.262 (0.259) | **1.03x** | 1.04x |
| 4, 4096, 16, 128, 128 | fwd | 0.173 (0.172) | 0.214 (0.210) | 0.214 (0.210) | **1.22x** | 1.22x |
| 4, 4096, 16, 128, 128 | fwd+bwd | 0.836 (0.830) | 1.057 (1.051) | 1.049 (1.039) | **1.27x** | 1.25x |
| 1, 16384, 8, 128, 128 | fwd | 0.195 (0.193) | 0.169 (0.167) | 0.169 (0.166) | **0.86x** | 0.86x |
| 1, 16384, 8, 128, 128 | fwd+bwd | 0.664 (0.658) | 0.509 (0.503) | 0.509 (0.503) | **0.76x** | 0.76x |
| 1, 16384, 16, 128, 128 | fwd | 0.297 (0.295) | 0.234 (0.229) | 0.234 (0.230) | **0.78x** | 0.78x |
| 1, 16384, 16, 128, 128 | fwd+bwd | 1.091 (1.084) | 0.793 (0.779) | 0.794 (0.779) | **0.72x** | 0.72x |
| 1, 32768, 8, 128, 128 | fwd | 0.327 (0.325) | 0.250 (0.247) | 0.250 (0.247) | **0.76x** | 0.76x |
| 1, 32768, 8, 128, 128 | fwd+bwd | 1.156 (1.151) | 0.832 (0.818) | 0.832 (0.818) | **0.71x** | 0.71x |
| 1, 65536, 8, 128, 128 | fwd | 0.585 (0.583) | 0.416 (0.413) | 0.418 (0.412) | **0.71x** | 0.71x |
| 1, 65536, 8, 128, 128 | fwd+bwd | 2.130 (2.117) | 1.472 (1.453) | 1.469 (1.460) | **0.69x** | 0.69x |
| 1, 65536, 16, 128, 128 | fwd | 1.071 (1.066) | 0.729 (0.725) | 0.730 (0.726) | **0.68x** | 0.68x |
| 1, 65536, 16, 128, 128 | fwd+bwd | 4.008 (3.994) | 2.737 (2.707) | 2.726 (2.705) | **0.68x** | 0.68x |
| 4, 4096, 16, 64, 64 | fwd | 0.146 (0.145) | - | 0.291 (0.288) | - | **1.99x** |
| 4, 4096, 16, 64, 64 | fwd+bwd | 0.532 (0.528) | - | 0.992 (0.984) | - | **1.86x** |
| 1, 16384, 8, 64, 64 | fwd | 0.136 (0.135) | - | 0.316 (0.314) | - | **2.32x** |
| 1, 16384, 8, 64, 64 | fwd+bwd | 0.420 (0.414) | - | 1.129 (1.116) | - | **2.69x** |
| 1, 16384, 16, 64, 64 | fwd | 0.205 (0.203) | - | 0.434 (0.432) | - | **2.12x** |
| 1, 16384, 16, 64, 64 | fwd+bwd | 0.644 (0.640) | - | 1.533 (1.524) | - | **2.38x** |
| 1, 32768, 8, 64, 64 | fwd | 0.232 (0.231) | - | 0.628 (0.625) | - | **2.71x** |
| 1, 32768, 8, 64, 64 | fwd+bwd | 0.740 (0.734) | - | 2.238 (2.226) | - | **3.03x** |

### Packed documents

FlashQLA and FLA get the `B` rows as one `cu_seqlens` row of `B * T` tokens, ours `segment_ids`; documents of
the given length fill each row (1000-token documents do not align with the chunks or the sequence pieces).

| document length | B, T, H, K, V | pass | ours | FlashQLA main | FLA main default | FlashQLA / ours | FLA default / ours |
|---|---|---|---|---|---|---|---|
| 128 | 1, 2048, 8, 128, 128 | fwd | 0.047 (0.047) | 0.021 (0.021) | 0.021 (0.021) | **0.45x** | 0.45x |
| 128 | 1, 2048, 8, 128, 128 | fwd+bwd | 0.148 (0.147) | 0.089 (0.087) | 0.089 (0.088) | **0.60x** | 0.60x |
| 1024 | 1, 16384, 8, 128, 128 | fwd | 0.124 (0.123) | 0.099 (0.097) | 0.099 (0.097) | **0.79x** | 0.79x |
| 1024 | 1, 16384, 8, 128, 128 | fwd+bwd | 0.526 (0.520) | 0.386 (0.379) | 0.387 (0.381) | **0.73x** | 0.73x |
| 1000 | 1, 16384, 8, 128, 128 | fwd | 0.132 (0.130) | 0.120 (0.118) | 0.121 (0.118) | **0.91x** | 0.91x |
| 1000 | 1, 16384, 8, 128, 128 | fwd+bwd | 0.560 (0.554) | 0.417 (0.410) | 0.417 (0.411) | **0.74x** | 0.74x |
| 2048 | 1, 32768, 8, 128, 128 | fwd | 0.216 (0.215) | 0.180 (0.177) | 0.180 (0.178) | **0.82x** | 0.83x |
| 2048 | 1, 32768, 8, 128, 128 | fwd+bwd | 0.941 (0.935) | 0.712 (0.695) | 0.708 (0.696) | **0.74x** | 0.74x |
| 4096 | 1, 65536, 8, 128, 128 | fwd | 0.388 (0.387) | 0.340 (0.338) | 0.340 (0.337) | **0.87x** | 0.87x |
| 4096 | 1, 65536, 8, 128, 128 | fwd+bwd | 1.760 (1.755) | 1.347 (1.332) | 1.350 (1.331) | **0.76x** | 0.76x |
| 512 | 1, 16384, 16, 128, 128 | fwd | 0.217 (0.215) | 0.198 (0.194) | 0.198 (0.195) | **0.91x** | 0.91x |
| 512 | 1, 16384, 16, 128, 128 | fwd+bwd | 0.942 (0.935) | 0.796 (0.781) | 0.789 (0.778) | **0.84x** | 0.83x |
| 2048 | 1, 16384, 16, 128, 128 | fwd | 0.217 (0.216) | 0.184 (0.180) | 0.184 (0.181) | **0.83x** | 0.84x |
| 2048 | 1, 16384, 16, 128, 128 | fwd+bwd | 0.944 (0.937) | 0.708 (0.693) | 0.708 (0.690) | **0.74x** | 0.74x |
| 4096 | 1, 65536, 16, 128, 128 | fwd | 0.736 (0.733) | 0.668 (0.663) | 0.666 (0.662) | **0.90x** | 0.90x |
| 4096 | 1, 65536, 16, 128, 128 | fwd+bwd | 3.399 (3.384) | 2.627 (2.607) | 2.617 (2.611) | **0.77x** | 0.77x |
| 512 | 4, 4096, 16, 128, 128 | fwd | 0.192 (0.190) | 0.198 (0.194) | 0.198 (0.195) | **1.02x** | 1.03x |
| 512 | 4, 4096, 16, 128, 128 | fwd+bwd | 0.887 (0.878) | 0.797 (0.785) | 0.791 (0.777) | **0.89x** | 0.89x |
| 2048 | 4, 4096, 16, 128, 128 | fwd | 0.192 (0.190) | 0.183 (0.181) | 0.184 (0.182) | **0.95x** | 0.96x |
| 2048 | 4, 4096, 16, 128, 128 | fwd+bwd | 0.886 (0.877) | 0.711 (0.700) | 0.705 (0.694) | **0.80x** | 0.79x |
| 1024 | 1, 16384, 8, 64, 64 | fwd | 0.097 (0.097) | - | 0.149 (0.148) | - | **1.53x** |
| 1024 | 1, 16384, 8, 64, 64 | fwd+bwd | 0.333 (0.332) | - | 0.526 (0.524) | - | **1.58x** |

### Head dim 256

FLA's Gated DeltaNet 1.3B config has 6 heads with `K = 256, V = 512` (the K-blocked kernels take `K = 256`
with `V` in {128, 256, 512}, and `K = 128, V = 256`). FlashQLA does not take `K = 256`, so FLA's default
dispatch runs the Triton forward and `chunk_bwd_dqkwg` on TileLang in the backward. FLA main time / ours:

| B, T, H, K, V | forward, FLA default and Triton | fwd+bwd, FLA default | fwd+bwd, FLA Triton |
|---|---|---|---|
| 4, 4096, 6, 256, 512 | **2.37x** | **1.17x** | 1.23x |
| 16, 4096, 6, 256, 512 (FLA's 1.3B layer) | **3.05x** | **1.24x** | 1.29x |

### Against MaxText

Provisional: MaxText main, dense only (it has no packed documents), measured on 17 Oct 2026 with the
kernels of that date, which are slower than the shipped ones at these shapes:

| B, T, H, K, V | pass | ours | MaxText main | MaxText / ours |
|---|---|---|---|---|
| 1, 4096, 8, 128, 128 | fwd | 0.096 (0.095) | 1.472 (1.409) | 14.9x |
| 1, 4096, 8, 128, 128 | fwd+bwd | 0.276 (0.274) | 7.203 (6.339) | 23.2x |
| 4, 4096, 16, 128, 128 | fwd | 0.189 (0.187) | 3.924 (3.782) | 20.2x |
| 4, 4096, 16, 128, 128 | fwd+bwd | 0.869 (0.861) | 16.135 (14.937) | 17.4x |

## Where we lose, and limits

- **Long single sequences at few heads.** One row of 16K-64K tokens: 0.71-0.87x forward and 0.69-0.78x
  forward + backward against FlashQLA at 8 heads, 0.68-0.79x and 0.68-0.73x at 16 heads. The exact split's
  forward map pass is a second, twice-as-wide state pass over every piece but the last (54 us at
  `(1, 16384, 8)`, table below), where FlashQLA's warm-up walks one chunk per piece and drops the rest (an
  approximation); the backward pays the same for its reverse map, and without documents the piece scan stays
  in XLA (28 us at 8 pieces forward, 38 backward).
- **Packed documents at few heads.** One row of 8 heads with 16 documents: 0.80-0.87x forward and
  0.74-0.77x forward + backward from 16K tokens (8 pieces), 0.45-0.54x and 0.58-0.60x at 2K-8K (4 pieces,
  where launches and XLA glue dominate); FlashQLA runs every document as its own sequence and needs no split.
  The short rows vary by up to a third between GPUs and runs (0.036-0.049 ms forward on two H100s with no
  other jobs, FlashQLA 0.021 on both). The piece-scan kernel used with packed documents and 8 pieces is
  Pallas Triton, which JAX 0.11.2 deprecates; a Mosaic GPU port is open.
- **Head dim 64 with packed documents, short rows.** FlashQLA does not support head dim 64, so the baseline is
  FLA main's default dispatch. One row of 8 heads with 16 documents: 0.88x / 0.84x / 0.84x forward and
  0.83x / 0.85x / 0.93x forward + backward at 2K / 4K / 8K tokens; from 16K tokens ours is 1.53-1.86x forward
  and 1.43-1.64x forward + backward.
- **The backward at large `B * H`** (0.83x forward + backward at `(8, 2048, 32)`, where the forward is
  1.07x): ours is a reverse state pass (`state_bwd`) and a per-chunk gradient kernel (`chunk_grads`) against
  FlashQLA's single fused backward. There `state_bwd` (318 us) is within 10% of its HBM traffic and
  `chunk_grads` needs at least 470 us for its 1.4 GB even with every load hidden, so the split backward cannot
  go much below 790 us, against 752 us for FlashQLA's `fused_bwd`. The opt-in fused backward
  (`fused_bwd=True`) is exact but slower at every shape measured (0.20-0.64x the default's speed, provisional).
- **Accuracy against FLA Triton.** Ours is within bf16 noise of FLA and FlashQLA on every tensor, but `dv` is
  10-16% above FLA Triton's rms error for Gated DeltaNet (15-22% for DeltaNet), and the initial-state gradient
  is 1.3-4.1x FLA Triton's for Gated DeltaNet and 1.1-2.2x for DeltaNet (equal to FlashQLA's); at `K = 256`
  it is 1.2e-3 to 2.2e-3 against FLA's 5e-4 to 1.2e-3.
- **`K = 256`:** in a provisional per-kernel profile the per-chunk gradient kernel is 63-69% of our forward +
  backward (single-buffered state blocks in 210 KB of shared memory; the two key-block CTAs of a chunk repeat
  the value-side products). No sequence split runs at these head dims, and `cu_seqlens` falls back to `chunked`.
- **Grid limit:** one row takes at most about 4.19M tokens (the chunk axis is CUDA's grid `y`, 65535 blocks).

Per stage at `(1, 16384, 8, 128, 128)` with 8 pieces, us per call (`profile_gdn_stages.py`, medians of 10
calls; provisional):

| stage | dense: ours | dense: FlashQLA | 16 documents: ours | 16 documents: FlashQLA |
|---|---|---|---|---|
| `T0` (`kkt`) | 26 | 32 | 29 | 32 |
| forward maps | 54 | 16 (`prepare_h`, 1-chunk warm-up) + 3 | 6 | - (no split) |
| piece scan | 28 (XLA) | 39 (`correct_h0`) | 10 (kernel) | - |
| forward main pass | 62 | 73 | 67 | 64 |
| rest (XLA glue) | 23 | 5 | 12 | 3 |
| **forward total** | **193** | **168** | **125** | **99** |
| **forward + backward total** | **654-664** | **511** | **526** | **386** |

## Correctness

The sequence split is exact (outputs, final state and every gradient equal the full walks bit for bit), and
all paths match autodiff of the float32 `chunked` op within 2e-2, kernel against kernel 1e-2:
`tests/test_mosaic_gpu_gated_delta_rule.py`, `test_mosaic_gpu_gdn_split.py`, `test_mosaic_gpu_scalar_gate_docs.py`,
`test_mosaic_gpu_scalar_gate_grouped.py` and `test_mosaic_gpu_gdn_hd256.py` (dense and packed documents,
initial and final states, grouped key heads, odd lengths, extreme gates, DeltaNet). A differential check
against a float64 token-loop oracle (`benchmarks/xcheck_scalar_gate.py`: ours, FLA main `c6e4f00` and
FlashQLA main on the same inputs) gives bf16-sized errors for every kernel (rms 2e-3 - 6e-3), and our float32
reference op agrees with the oracle to 2e-7 - 9e-7: the document, state and grouped-head semantics are FLA's.

## Reproduce

```
python benchmarks/bench_gated_delta_rule.py --kind gated_deltanet --shapes "B,T,H,K,V;..." \
    --impls mosaic_gpu,flashqla,fla,fla_full --rounds 2 --out results/gdn.json \
    --torch-python VENV_FLA_MAIN/bin/python --fla-full-python VENV_FLA_MAIN_FULL/bin/python \
    --qla-python VENV_QLA_MAIN/bin/python --baseline-commit fla=6dff3e1 --baseline-commit flashqla=5538c35
# packed documents: --doc-len 1024 (and so on); 64K-token rows: --rounds 1 --iterations 12
# K = 256: --shapes "4,4096,6,256,512;16,4096,6,256,512" --impls mosaic_gpu,fla,fla_full; per kernel:
python benchmarks/bench_gdn_hd256.py --shapes "4,4096,6,256,512;16,4096,6,256,512"
python benchmarks/profile_gdn_stages.py --shapes "1,16384,8,128,128" --modes fwd,fwdbwd \
    --torch qla_gdn=VENV_QLA_MAIN/bin/python   # per stage; add --doc-len 1024 for 16 documents
python benchmarks/xcheck_scalar_gate.py gen --dir DIR && python benchmarks/xcheck_scalar_gate.py ours --dir DIR
VENV_FLA_MAIN/bin/python benchmarks/xcheck_scalar_gate_torch.py --dir DIR --impls oracle,fla
python benchmarks/xcheck_scalar_gate.py report --dir DIR --out results/xcheck_scalar_gate.json
```

## How it works

When the serial state passes would run at most 32 CTAs (`B * H * V / 64`), `scalar_gate.pieces` cuts each
sequence into up to 4 pieces, or 8 when they would run at most 16 CTAs and every piece keeps at least 32
chunks. A map pass gives each piece's affine map `S_end = M S_start + E`, a float32 scan over the pieces gives
every piece's entering state, and the main pass runs the pieces in parallel; the backward mirrors this. A
chunk in which a document starts lets no state through, so with packed documents the map passes walk only
from a piece's last such chunk and skip maps and products that are exactly zero. `K = 256` runs K-blocked
kernels that carry the state as `K / 64` register tiles, as FLA does. **Opt-in approximate split:**
`scalar_gate.run(..., cp_warmup=-10.0)` uses FlashQLA's warm-up: each piece's map walks only the chunks until
their summed log decay falls below `cp_warmup` and takes `M = 0` (pieces and heads that never get there are
split exactly); with unit-norm keys and `0 <= beta <= 2` the dropped share of every state and state gradient
is at most `e^cp_warmup` (4.5e-5 at -10). It is off by default; `benchmarks/gdn_warmup_error.py` measures it.
