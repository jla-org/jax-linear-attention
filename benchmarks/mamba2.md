# Mamba-2 (SSD) scan and layer: Mosaic GPU kernels against the official kernels and FLA

Ours is `jla.ops.linear_attention("simple_gla", ..., impl="mosaic_gpu")` with the Mamba-2 SSD mapping
`q = C`, `k = B`, `v = dt x`, `g = dt A`, `scale = 1`; it shares the scalar-gate kernels with Gated
DeltaNet ([gdn.md](gdn.md)). The scan is compared with the official Mamba-2 Triton scan
(`mamba_chunk_scan_combined`) and with FLA's `chunk_simple_gla`, and the whole Mamba-2 layer with the
official `Mamba2` module's fused training path.

## Setup

- One H100, JAX 0.11.2, torch 2.13.0+cu129, Triton 3.7.1; bf16 `q, k, v` (`C, B, dt x`) and float32
  gates.
- Baselines:
  - **Official Mamba-2**: state-spaces/mamba `e9594ce` (still its latest commit), `mamba_ssm`
    2.3.2.post1 built from source with its CUDA extensions. The scan gets `x, dt, A, B, C` with
    `ngroups = 1`, chunk 256 and `seq_idx` for documents. The layer is the `Mamba2` module with
    `use_mem_eff_path=True` (its default): `mamba_split_conv1d_scan_combined` fuses the causal conv1d,
    the scan, the `D` skip, the gated RMSNorm and the output projection into one autograd function
    (the input projection is a separate GEMM).
  - **FLA**: main `6dff3e1` for the release ratios and `c6e4f00` (0.6.0) for the table of times,
    `chunk_simple_gla` with `scale = 1` and `cu_seqlens` for documents. FLA's default dispatch (with
    FlashQLA, TileLang 0.1.15 and FlashKDA installed) runs the same Triton path for simple GLA
    (within 0.4%).
- **Timing, scan**: GPU kernel time per call (CUPTI for JAX, `torch.profiler` device time for torch),
  the implementations round-robin on one GPU, 2 rounds dense and 1 at 64K tokens or with documents,
  each of 30 calls after 3 warmups (12 at 64K tokens). Cells are the median over rounds of the
  per-round median, with the fastest call in parentheses; ratios compare the fastest calls.
- **Timing, layer**: the same configuration on both sides (`d_state` 128, expand 2, head dim 64, one
  B/C group, chunk 256, bf16), forward and forward + backward with the gradients of the input and of
  every parameter. Median kernel time per call: CUPTI for ours; for the official module
  `torch.profiler` over 3 sessions of 10 calls, each kernel's mean time per launch times its launches
  per call (a missing record would be reported as `dropped_records`; none was), checked against the
  CUDA-event wall clock of 20 back-to-back calls.
- Measured on 20 Oct 2026 with no other jobs on the GPUs. The scan's times per shape are from 17 Oct
  2026 and provisional.

## Results

### The scan

Release measurements (20 Oct 2026): forward 1.31-2.60x and forward + backward 1.20-1.96x the
official kernel, and 1.49-3.18x and 1.58-2.20x FLA main.

Times per shape (17 Oct 2026, provisional; FLA main `c6e4f00`). Packed documents are our
`segment_ids`, one `cu_seqlens` row of `B * T` tokens for FLA and `seq_idx` for Mamba-2:

| documents | B, T, H, K, V | pass | ours | FLA main Triton | Mamba-2 (official) | FLA / ours | Mamba-2 / ours |
|---|---|---|---|---|---|---|---|
| none | 4, 4096, 24, 128, 64 | fwd | 0.123 (0.120) | 0.254 (0.251) | 0.229 (0.227) | 2.10x | 1.90x |
| | | fwd+bwd | 0.741 (0.737) | 1.189 (1.180) | 1.112 (1.105) | 1.60x | 1.50x |
| | 1, 16384, 48, 128, 64 | fwd | 0.293 (0.291) | 0.555 (0.550) | 0.435 (0.431) | 1.89x | 1.48x |
| | | fwd+bwd | 1.666 (1.657) | 2.622 (2.611) | 2.004 (1.992) | 1.58x | 1.20x |
| | 1, 4096, 8, 128, 128 | fwd | 0.032 (0.031) | 0.047 (0.046) | 0.042 (0.041) | 1.47x | 1.30x |
| | | fwd+bwd | 0.142 (0.140) | 0.229 (0.226) | 0.280 (0.279) | 1.61x | 1.99x |
| | 1, 16384, 16, 128, 128 | fwd | 0.174 (0.172) | 0.327 (0.324) | 0.259 (0.257) | 1.88x | 1.49x |
| | | fwd+bwd | 0.856 (0.851) | 1.578 (1.570) | 1.236 (1.230) | 1.85x | 1.45x |
| | 4, 4096, 16, 128, 128 | fwd | 0.096 (0.095) | 0.304 (0.301) | 0.250 (0.247) | 3.18x | 2.61x |
| | | fwd+bwd | 0.703 (0.699) | 1.547 (1.539) | 1.191 (1.184) | 2.20x | 1.69x |
| | 1, 65536, 16, 128, 128 | fwd | 0.632 (0.629) | 1.247 (1.237) | 1.136 (1.133) | 1.97x | 1.80x |
| | | fwd+bwd | 3.271 (3.265) | 6.284 (6.244) | 4.836 (4.830) | 1.91x | 1.48x |
| 512 tokens | 4, 4096, 24, 128, 64 | fwd | 0.124 (0.121) | 0.255 (0.253) | 0.231 (0.228) | 2.10x | 1.89x |
| | | fwd+bwd | 0.747 (0.743) | 1.199 (1.195) | 1.127 (1.123) | 1.61x | 1.51x |
| | 1, 16384, 16, 128, 128 | fwd | 0.125 (0.124) | 0.272 (0.270) | 0.265 (0.263) | 2.18x | 2.12x |
| | | fwd+bwd | 0.747 (0.744) | 1.354 (1.349) | 1.258 (1.254) | 1.81x | 1.69x |
| | 1, 65536, 16, 128, 128 | fwd | 0.431 (0.428) | 1.068 (1.064) | 1.153 (1.147) | 2.48x | 2.68x |
| | | fwd+bwd | 2.817 (2.808) | 5.259 (5.247) | 4.893 (4.885) | 1.87x | 1.74x |
| 2048 tokens | 4, 4096, 24, 128, 64 | fwd | 0.124 (0.120) | 0.257 (0.254) | 0.229 (0.228) | 2.11x | 1.89x |
| | | fwd+bwd | 0.746 (0.744) | 1.203 (1.198) | 1.127 (1.120) | 1.61x | 1.51x |
| | 1, 16384, 16, 128, 128 | fwd | 0.153 (0.150) | 0.275 (0.270) | 0.264 (0.262) | 1.79x | 1.74x |
| | | fwd+bwd | 0.772 (0.768) | 1.359 (1.351) | 1.257 (1.251) | 1.76x | 1.63x |

### The whole layer against the official fused training path

The `Mamba2` mixer as the presets run it (input projection, causal conv1d, scan, `D` skip, gated
RMSNorm, output projection; 20 Oct 2026):

| B, T, d_model | pass | ours (ms) | official fused path (ms) | official / ours |
|---|---|---|---|---|
| 4, 4096, 2048 | fwd | 1.609 | 1.928 | **1.20x** |
| | fwd+bwd | 5.671 | 6.962 | **1.23x** |
| 16, 4096, 2048 | fwd | 6.201 | 8.323 | **1.34x** |
| | fwd+bwd | 22.172 | 29.694 | **1.34x** |
| 1, 16384, 2048 | fwd | 1.751 | 1.961 | **1.12x** |
| | fwd+bwd | 6.005 | 7.010 | **1.17x** |
| 8, 2048, 768 | fwd | 0.435 | 0.538 | **1.24x** |
| | fwd+bwd | 1.613 | 2.281 | **1.41x** |

The layer is faster than the official fused path at every shape: 1.12-1.34x forward and 1.17-1.41x
forward + backward. At `(8, 2048, 768)` the official call is bound by its launches: 0.857 ms per
forward on the wall clock (20 back-to-back calls) against 0.538 ms of kernels, where ours takes 0.455
and 0.435 ms.

## Where we lose, and limits

- **The layer gains less than the scan.** Forward + backward the layer is 1.17-1.41x the official
  fused path, against the scan's 1.20-1.96x: the scan is only part of the layer's time, and the
  official path fuses the conv1d, the gating and the norm into its scan's autograd function. Forward
  at `(1, 16384, 2048)` (one provisional round), ours takes 1.742 ms (GEMMs 1.040, SSD 0.339, gated
  norm 0.178, short conv 0.111, slices and weight casts 0.07) and the official 1.956 ms (GEMMs 1.090,
  SSD scan 0.606 in five Triton kernels, gated norm 0.132, conv1d 0.110). Left there: the scaled SSD
  kernel is 0.05 ms slower than the unscaled one (the `dt` multiply sits on the state warpgroup's
  serial chain), the norm reads `x` for the `D` skip (+0.046 ms against the official norm), and each
  call slices the z|x weight (0.035 ms) and folds the norm weight into the output projection
  (0.018 ms).
- **Smallest scan margins**: forward + backward at `(1, 16384, 48, 128, 64)` (1.20x) and the forward at
  `(1, 4096, 8, 128, 128)` (1.30x, 31 us per call).
- **`dA` and `ddt` accuracy.** Against the official kernel, the layer path's rms errors are 1.2-1.3x
  the official's for `y`, 1.25-1.4x for `dx`, 1.4-1.5x for `dB, dC`, 1.3-1.4x for the final state,
  1.6-1.7x for `ddt` (12x with extreme gates) and 2.3-15x for `dA` (43x with extreme gates: 2.1e-3
  against 4.9e-5); our `dh0` is more accurate (1.6e-3 - 1.7e-3 against 2.3e-3 - 2.4e-3). FLA main
  through the same mapping has the same gaps, so they come from the gate-gradient formulation (`dg`
  from `q . dq - k . dk` and a reverse cumsum, as FLA computes it) and the bf16 `v = dt x`, not from
  the kernels. The errors stay bf16-sized (rms 3.1e-3 at most); Mamba-2's own `ddA` formulation is
  the next step.
- **Simple GLA `dv`** has a 12-17% larger rms error than FLA's (2.7e-3 - 2.8e-3 against 2.4e-3):
  `sg_dv_local` stores the intra-chunk part of `dv` in bf16 and `sg_state_bwd` adds the inter-chunk
  part and rounds again. A float32 intermediate (16 KB more SMEM in `sg_state_bwd`) would fix it.
- **Grouped B/C kernels** (`q`/`k` with fewer heads than `v`): their forward is faster (0.64-0.98x our
  time), but their backward writes the key-side gradients per value head in float32 and sums them in
  XLA, so their forward + backward is 0-13% slower. The layer therefore repeats B/C to the value heads
  (through float32) and calls the per-head kernels.

## Correctness

The differential harness of [gdn.md](gdn.md) (`benchmarks/xcheck_scalar_gate.py`, a float64 torch
oracle) feeds the same `x, dt, A, B, C` and initial state `[B, H, P, N]` to the official kernel and to
ours: our reference op agrees with the oracle to 1.8e-7 - 4.2e-7 on every simple GLA and Mamba-2 case
(odd lengths, packed documents, grouped heads, extreme gates, up to 16K tokens), and the bf16 layer
path's worst tensor is 4.4e-3 - 6.1e-3 of its maximum (official kernel 3.6e-3 - 5.7e-3).
`tests/test_layers_grouped_heads_precision.py` covers the B/C repeat, `tests/test_mamba2_fused.py` the
fused glue (masked row kernels at 3, 5 and 6 heads, the fused against the unfused layer, the folded SSD
forward bitwise against the unfolded one) and `tests/test_mamba2_in_place_zx.py` the in-place z rows;
against a float32 reference the fused glue's error is at most 1.23x the unfused glue's at `d_model`
768 (three seeds), and its gradients' at most 1.33x at 6 to 96 heads.

## Reproduce

```
python benchmarks/bench_gated_delta_rule.py --kind simple_gla [--doc-len 512|2048] --shapes "B,T,H,K,V;..." \
    --impls mosaic_gpu,mosaic_gpu_grouped,fla,fla_full,mamba2 --rounds 2 \
    --torch-python VENV_FLA_MAIN/bin/python --fla-full-python VENV_FLA_MAIN_FULL/bin/python \
    --mamba-python VENV_MAMBA/bin/python --out results/mamba2_scan.json
python benchmarks/bench_mamba2_layer.py jax --out results/mamba2_layer_jax.json
VENV_MAMBA/bin/python benchmarks/bench_mamba2_layer.py torch --out results/mamba2_layer_torch.json
```

## How it works

The layer (`jla.layers.ssm.mamba2_ssd`) computes the gate and the conv input as two GEMMs (an
optimization barrier keeps XLA's dot merger from fusing them back into one), runs the causal conv on
the fused short-conv kernels ([short_conv.md](short_conv.md)) and the scan on the simple-GLA kernels,
with the `ngroups` B/C repeated to the value heads through float32. The `D` skip, the gate and the
gated RMSNorm run as Pallas Triton row kernels in one custom VJP with the output projection
(`SSMConfig.fused_glue`); masked row kernels pad each row to the next power of two of heads, so any
head count takes this path (`d_model` 768 with 24 heads included). A call that is not differentiated
(inference, evaluation) hands `x` and `dt` to the SSD kernel, whose state warpgroup forms
`(x dt).astype(bf16)` per chunk, rounded as XLA rounds it, so the output is the same bit for bit.
