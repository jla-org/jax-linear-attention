# End-to-end training at 1.3B on FineWeb-Edu

`jla train` trains each 1.3B preset for 20 steps on FineWeb-Edu at FLA's published ~1.3B shapes
([`bench_e2e_1p3b.py`](bench_e2e_1p3b.py)) and reports tokens/s, MFU, peak memory and the loss. The baseline
is FLA's own trainer, fla-org/flame, with FLA's models and kernels ([`bench_e2e_flame.py`](bench_e2e_flame.py)),
on the same shapes, the same tokens in the same order and the same 65,536 tokens per step. Both runners score
MFU with jla's analytic FLOP count, so their MFU columns compare directly. The same harness trains two MoE
hybrids with the layouts of Qwen3-Next and Kimi Linear, which flame has no model for.

## Setup

- H100 80GB HBM3, JAX 0.11.2; fp32 parameters, gradients and AdamW state, bf16 compute. FP8 rows add
  `model.fp8 = "delayed"`, `fp8_min_features = 512` and `fp8_fused = true`.
- FineWeb-Edu, 3B tokens, Llama-3.1 tokenizer (128,256 ids, EOS 128001), CIDX records of exactly 4096 tokens,
  with document masking: `segment_ids` and positions restart after every EOS, so attention and every recurrent
  state stay within a document.
- Each TOML as written (`end_to_end/1p3b/`, `end_to_end/hybrids_1p3b/`), 65,536 tokens per GPU and step: the
  recurrent presets at 16 × 4096 with full rematerialization, the two transformers without recompute
  (`remat = "none"`, `lean_vjp = true`, `scan_layers = false`) at 4 × 4096 × `grad_accum` 4, `kimi-linear-1p3b`
  at 8 × 4096 × 2. 20 AdamW steps (b1 0.9, b2 0.95, weight decay 0.1, clipping 1.0, lr 3e-4 with 5 warm-up
  steps, then cosine to 0.1×).
- `--impl kernels`: every mixer on its Mosaic GPU kernel, attention on the FA3-style kernel; a shape a kernel
  does not support fails instead of falling back. XLA's BFC allocator, preallocation off, memory fraction 0.92.
- Timing: the wall clock from the completion of step 5 to that of step 19 for both trainers (with
  `train.dispatch_ahead`, set by the `transformer-1p3b` TOML, the last step has nothing dispatched behind it).
  MFU = tokens/s × training FLOPs per token / (GPUs × 989 TFLOP/s, the H100 SXM dense bf16 peak), with training
  FLOPs = 3 × the forward count of `jla.training.performance.estimate_flops_per_token`, bf16-equivalent in FP8.
- Baseline: flame main `24dea91` with FLA main `6dff3e1` and torchtitan `0b44d4c`, in flame's recommended mode
  ([The flame baseline](#the-flame-baseline) has every commit).
- Measured on 20 Oct 2026 with no other jobs on the GPUs: one run per configuration, the one-GPU runs side by
  side, one per GPU.

## Results

One H100, tokens/s (MFU). "vs flame" is flame's step time over ours (> 1: ours is faster); the FP8 column
compares with flame's FP8.

| preset | ours, bf16 | flame, bf16 | vs flame | ours, FP8 | flame, FP8 | vs flame FP8 |
|---|---|---|---|---|---|---|
| `transformer-1p3b` | 60,178 (56.4%) | 52,165 (48.9%) | 1.15x | **77,094 (72.3%)** | 54,871 (51.5%) | 1.41x |
| `transformer-swa-1p3b` | 55,188 (50.1%) | 53,162 (48.2%) | 1.04x | 68,072 (61.8%) | 56,339 (51.1%) | 1.21x |
| `gdn-hd128-1p3b` | 42,995 (42.3%) | 25,723 (25.3%) | 1.67x | 55,956 (55.0%) | 36,694 (36.1%) | 1.52x |
| `gdn-1p3b` (d) | 37,759 (36.0%) | 28,172 (26.9%) | 1.34x | 46,971 (44.8%) | 38,482 (36.7%) | 1.22x |
| `kda-1p3b` | 35,110 (32.9%) | 23,487 (22.0%) | 1.49x | 42,756 (40.1%) | 24,168 (22.6%) | 1.77x |
| `mamba2-1p3b` | 38,899 (37.1%) | 37,144 (35.4%) (a) | 1.05x (a) | 43,688 (41.7%) | 39,845 (38.0%) (a) | 1.10x (a) |
| `mamba3-1p3b` | 32,976 (32.6%) | 21,221 (21.0%) (b) | 1.55x | 36,344 (36.0%) | 10,661 (10.5%) (b) | 3.41x (b) |
| `rwkv7-1p3b` | 28,484 (26.2%) | not a valid baseline (c) | - | 30,651 (28.2%) | 16,143 (14.9%) (c) | 1.90x (c) |
| `gdn2-1p3b`, provisional (e) | 30,300 (33.70%) | no baseline (e) | - | 37,010 (41.16%) | - | - |

- (a) flame's Mamba-2 trains **without document masking**: FLA's Mamba-2 layer ignores `cu_seqlens`
  (`fla/layers/mamba2.py` passes `seq_idx=None`), while ours masks documents.
- (b) flame's Mamba-3 cannot compile (inductor fails on the Mamba-3 block) and fails under activation
  checkpointing (the official SISO kernel unpacks its saved tensors twice), so it runs in its only working mode:
  no compile, no checkpointing, one sequence per micro-step. Its FP8 run is half as fast as its bf16 run; against
  that bf16 run our FP8 is 1.71x.
- (c) flame's compiled RWKV-7 in bf16 trains to a much worse loss (9.31 at step 20, against 7.53 for its FP8 run
  and 7.88 for ours), so it is not a valid baseline and there is no bf16 comparison (it ran at 23,094 tokens/s).
  Its FP8 run trains normally but is slower than that bf16 run; the 1.90x is against the FP8 run. flame's records
  carry no FLOP count for RWKV-7, so its MFU uses ours (the same model).
- (d) Measured the same day on the code just before the last kernel changes of this release, which leave the
  `K = 256` kernels as they were (on one GPU those changes moved the other presets by 0.98-1.03x).
- (e) GDN-2 is implemented clean-room from the equations of arXiv 2605.22791: no upstream GDN-2 code is read or
  run, so there is no baseline. Provisional: interleaved pairs on one H100 while other jobs could share the
  machine.

flame's FP8 is torchtitan's float8 converter (torchao 0.18.0 `Float8Linear`, tensorwise dynamic scaling, every
linear layer but the LM head), run twice: on every linear layer, and with the layers narrower than 512 features
left in bf16 as ours are. The faster run is the baseline (the narrower-in-bf16 one for RWKV-7, the other one
elsewhere).

- **FP8 against bf16** (ours): 1.28x for the transformer, 1.23x with the sliding window, 1.30x for
  `gdn-hd128-1p3b`, 1.24x for `gdn-1p3b`, 1.22x for KDA, 1.12x for Mamba-2, 1.10x for Mamba-3 and 1.08x for
  RWKV-7.
- **Peak memory**: ours 41.9-54.7 GiB in bf16 and 44.5-52.0 GiB in FP8; flame's 50.9-66.9 GiB and 49.9-69.6 GiB
  of the 80 GB. Every loss falls from about 12.1 to 7.8-8.0 in 20 steps.
- **Against the plain-JAX ops** (`--impl chunked`, provisional): on `gdn-1p3b` the `K = 256` kernels train 2.08x
  faster; GDN-2, KDA, Mamba-2, Mamba-3 and the dense hybrids do not fit one GPU at 16 × 4096 with the chunked ops.
- **Elsewhere**: 8 GPUs on [multi_gpu.md](multi_gpu.md), FP8 on [fp8_1p3b.md](fp8_1p3b.md), where the step time
  goes on [profile_1p3b.md](profile_1p3b.md), and 1.3B hybrids in the Gated DeltaNet-2 paper's throughput
  setting on [gdn2_paper.md](gdn2_paper.md).

## Presets

The shapes come from flame's `configs/{transformer,gated_deltanet,mamba2}_1B.json` and, where flame has no 1B
config, from the defaults of FLA main `c6e4f00` `configuration_<model>.py`. Every preset has hidden size 2048,
untied embeddings over the corpus's 128,256 ids (525M parameters in the embedding and LM head; FLA's configs
say 32,000), FLA's SwiGLU MLP of width 5632 (`hidden_ratio` 4) and RMSNorm with FLA's `norm_eps`.

| preset | source | layers | mixer | params, total / non-embedding | training GFLOPs/token |
|---|---|---|---|---|---|
| `transformer-1p3b` | flame `transformer_1B.json` + `num_kv_heads` 8 | 24 | 32 heads × 64, 8 KV heads (Llama-3.2-1B's 32:8), RoPE θ 1e4 | 1608M / 1082M | 9.28 |
| `transformer-swa-1p3b` | the same + `window_size` 2048 | 24 | as above, sliding window 2048 on every layer | 1608M / 1082M | 8.98 |
| `gdn-1p3b` | flame `gated_deltanet_1B.json` | 21 | 6 heads, `head_dim` 256, `expand_v` 2 (values 512), conv 4, output gate | 1782M / 1256M | 9.43 |
| `gdn-hd128-1p3b` | the same with 16 heads × 128, `expand_v` 1 | 24 | 16 heads × 128, conv 4, output gate | 1861M / 1336M | 9.72 |
| `gdn2-1p3b` | clean-room, arXiv 2605.22791 | 24 | Gated DeltaNet-2, 16 heads × 128, conv 4 | 2074M / 1548M | 11.00 |
| `kda-1p3b` | FLA `KDAConfig` | 24 | 16 heads × 128, conv 4, low-rank gates of rank 128 | 1785M / 1260M | 9.27 |
| `mamba2-1p3b` | flame `mamba2_1B.json` | 48 | `state_size` 128, `expand` 2, 64 heads of 64, one group, no MLP | 1766M / 1241M | 9.43 |
| `mamba3-1p3b` | FLA `Mamba3Config` | 48 | SISO, `state_size` 128, `expand` 2, `head_dim` 64, `rope_fraction` 0.5, no MLP | 1781M / 1256M | 9.79 |
| `rwkv7-1p3b` | FLA `RWKV7Config` | 24 | 32 heads of 64, LoRA ranks 64/64/16/128 (decay/a/v/gate), channel mix 8192 | 1,761M / 1,236M | 9.08 |
| `gdn-hybrid-1p3b` | this template's layout | 24 | 18 GDN (8 key / 16 value heads of 128) + 6 attention (16 × 128, 4 KV heads, QK-norm, output gate, RoPE θ 1e7 on 1/4 of each head) | 1748M / 1222M | 9.31 |
| `kda-hybrid-1p3b` | this template's layout | 24 | 18 KDA + 6 attention (16 × 128, 4 KV heads, QK-norm, no positional encoding) | 1741M / 1216M | 9.27 |

`gdn-1p3b` is FLA's exact shape, on the K-blocked kernels of [gdn.md](gdn.md). Our kernels use their
native chunk (64 tokens; 32 for RWKV-7) where `mamba2_1B.json` says 256, which does not change the model.
`rwkv7-1p3b` keeps this template's RMSNorm pre-norm blocks where FLA uses LayerNorms with bias: 1,760,872,448
parameters against FLA's 1,760,909,312 (36,864 fewer, 0.002%). FLA's hybrid attention layer has no output gate,
partial RoPE or NoPE, so the two dense hybrids have no flame equivalent.

## The flame baseline

[`bench_e2e_flame.py`](bench_e2e_flame.py) runs fla-org/flame's own trainer on the configs in
[`end_to_end/1p3b/flame/`](end_to_end/1p3b/flame/), each one the jla preset of the same name: flame main
`24dea91`, FLA main `6dff3e1` (0.6.0, its default backend dispatch on, with TileLang 0.1.15, FlashQLA main
`5538c35` and FlashKDA main `7afb9f4` installed), torchtitan `0b44d4c` (the commit flame's README pins), torch
2.13.0+cu129 with Triton 3.7.1, torchao 0.18.0, FlashAttention-3 3.0.1 (Dao-AILab/flash-attention main
`965c9212`, `hopper/`) and state-spaces/mamba main `e9594ce` (`mamba_ssm` 2.3.2.post1) with causal-conv1d 1.7.0.

- **Mode.** `torch.compile` with no activation checkpointing (flame's `activation_checkpoint.mode = "none"` with
  its README's `--training.compile`), each step split into 4, 8 or 16 micro-steps (the fewest that fit). Compile
  and checkpointing cannot be combined here: a compiled checkpointed block fails at the first step, because FLA's
  fused norm `layer_norm_fwd` is `torch.compiler.disable`d and dynamo cannot graph-break inside the checkpoint
  operator. AMP bf16 with fp32 parameters and optimizer state on one GPU, FSDP2 with bf16 parameters on 8.
- **FP8** adds `--set=--model.converters=float8`; flame's `Float8` job config lacks the `filter_fqns` and
  `emulate` fields that torchtitan's converter reads, so a small shim sets them (the LM head skipped, no
  emulation).
- **Kernels that ran**, from FLA's backend log of each run: FlashQLA rejects both Gated DeltaNet shapes
  (`requires K=128, got 256`; `does not support use_gate_in_kernel`, which FLA's `GatedDeltaNet` layer always
  passes) and FlashKDA rejects KDA (`only supports inference mode`), so those rows run FLA's Triton kernels with one
  backward kernel each on TileLang (`chunk_bwd_dqkwg`, `chunk_kda_bwd_wy_dqkg_fused`). RWKV-7 rejects TileLang
  (its `a`, `b` stay fp32 under AMP) and runs FLA's Triton DPLR kernels; the transformers run FA3, and Mamba-2 and
  Mamba-3 state-spaces/mamba's kernels inside FLA's layers. No run measures FlashQLA or FlashKDA.
- **Two adjustments**, outside the training step: `--fa3-shim` takes FLA's `flash_attn_func` and
  `flash_attn_varlen_func` from FA3 ([`fa3_shim/flash_attn/__init__.py`](end_to_end/1p3b/flame/fa3_shim/flash_attn/__init__.py)),
  since FlashAttention-2 has no wheel for torch 2.13; `--flame-src` puts the flame checkout on `sys.path`.
- **Same tokens.** The runner draws the records with jla's `TokenWindowDataset(exact_packed_records=True)` and
  `BatchLoader` with the same seed and shuffle. Each GPU's records become one varlen sequence (flame's
  `--training.varlen`) with a `cu_seqlens` boundary after every EOS and at every record start, so FLA restarts
  attention, states, convolutions and positions at every document; Mamba-2 gets separate 4096-token rows (note
  (a)). FLA's loss leaves out the target that jla scores against the next record's first token: about 1 in 4096.
- flame's `skip_nan_inf` stays on, and the harness fails a run with any skipped step.

## Mamba-2

`mamba2-1p3b.toml` runs the fused output glue and the fused LM-head cross entropy (`train.ce_impl = "auto"`,
16,384-token vocabulary chunks). The fused glue (`model.ssm.fused_glue`, on by default) runs the D skip, the gate
and the gated RMSNorm as two Pallas Triton row kernels, forward and backward, in one custom VJP with the
out-projection; the skip stays in float32, and with a bf16 out-projection the norm weight is folded into the
out-projection weight. The glue drops from 198 to 126 ms per step, 1.039x end to end in bf16. In FP8 the short
conv and the glue kernels read the two rows of the stacked z|x in-projection in place instead of copying them out
(`jla.layers.ssm.IN_PLACE_ZX`, 1.022x). On several GPUs the row kernels run per shard of the mesh (1.036x at FSDP 8
in bf16); under tensor parallelism the layer keeps its unfused glue. The three gains are interleaved A/B pairs
against the code without each change (provisional). Without document masking (flame's setting) ours ran at 38,691
tokens/s against 38,508 with it (provisional): the kernel's document resets cost nothing measurable.

## RWKV-7

The RWKV-7 layers call the op's bounded mode (`lower_bound = -0.61`; FLA's `RWKV7Attention` passes a safe gate
with `lower_bound = -0.6065`). At head dim 64 its per-chunk work runs in two Mosaic GPU kernels around the
vector-gate state passes, `rwkv7_prep` forward and `rwkv7_bwd` backward, each with two warpgroups per CTA (1.102x
and 1.104x end to end against one warpgroup, two interleaved pairs, provisional). Products with a float32 operand
run as three split-bf16 tensor-core products (`BF16X3`), and no score matrix or intermediate gradient reaches HBM.
A scanned stack computes the value residual `v_first` before the scan and saves it once instead of once per layer
(6 GiB at this size). `rwkv7-1p3b` fits one H100 at 16 × 4096 under BFC (48.52 GiB peak, provisional). At the
1.3B layer shape the op is 1.09x FLA main forward and 1.06x forward + backward (4.75 / 19.32 ms against
5.19 / 20.43 ms), and 1.15x / 1.16x with 16 packed documents at 1 × 65,536 ([rwkv7.md](rwkv7.md)).

## Qwen3-Next and Kimi Linear layouts

Two presets mirror published hybrid models at half their hidden size, with the same per-head shapes, layer
pattern, routing and MoE granularity, and the corpus's untied 128,256-token vocabulary (262.7M parameters in the
embedding and LM head). FLA main has no model with either layout (none of its configs has an MoE, and its hybrids
take only standard softmax attention layers), so there is no flame row.

| | `qwen3-next-1p3b` (Qwen3-Next-80B-A3B-Instruct) | `kimi-linear-1p3b` (Kimi-Linear-48B-A3B-Instruct) |
|---|---|---|
| parameters | 1.27B total, 0.55B active | 1.20B total, 0.63B active |
| layers, hidden | 16 (48), 1024 (2048) | 16 (27), 1024 (2304) |
| linear mixer | Gated DeltaNet, 8 key / 16 value heads of 128 (16 / 32), conv 4 | KDA, 16 heads × 128 (32), conv 4, low-rank decay and output gates |
| every 4th layer | gated attention, 8 / 1 heads of 256 (16 / 2), sigmoid output gate, QK-norm, `partial_rotary_factor` 0.25, `rope_theta` 1e7 | MLA, 16 heads (32), `kv_lora_rank` 512, nope 128 / rope 64 / v 128, full-rank queries, NoPE |
| MLP | MoE in every layer: 64 experts × 256 (512 × 512), top-10, sigmoid-gated shared expert | dense first layer of 4096 (4 × hidden), then MoE: 32 experts × 512 (256 × 1024), top-8, sigmoid scores, `routed_scaling_factor` 2.446, 1 shared expert |
| other | one MTP layer; `aux_loss_coef` 1e-2 = top-10 × 1e-3 (our balance loss divides by routes, Hugging Face's by tokens) | load-balancing loss (no expert-bias update rule) |

Provisional (other jobs could share the machine): the protocol above with memory fraction 0.92 on one GPU and
0.85-0.92 on eight, tokens/s per GPU from the harness's mean step over steps 6-20. Qwen3-Next's 256-wide attention heads run
on cuDNN (`--attention cudnn`: the Mosaic GPU kernel takes head dims 64 and 128, and 192/128 for MLA); the MoE runs
dropless on the Mosaic GPU grouped GEMM with per-device dispatch. MFU counts the active parameters' FLOPs (routed
experts at top-k, the shared expert and the MTP layer included).

| preset | GPUs | setting | tokens/s per GPU (MFU) | peak memory |
|---|---|---|---|---|
| `qwen3-next-1p3b` | 1 | the TOML (forward-gradient cross entropy on the main and MTP heads) | 73,248 (26.7%); 77,797 (28.4%) with no other jobs on the machine | 45.0 GiB |
| `qwen3-next-1p3b` | 8, FSDP 8 | the TOML | 71,926 (26.2%) | 33.9 GiB |
| `kimi-linear-1p3b` | 1 | the TOML (8 × 4096 × 2, XLA cross entropy) | 67,674 (22.7%) | 28.3 GiB |
| `kimi-linear-1p3b` | 1 | 16 × 4096, forward-gradient cross entropy | 73,480 (24.6%) | 28.8 GiB |
| `kimi-linear-1p3b` | 8, FSDP 8 | the TOML | 61,553 (20.6%) | 15.8 GiB |
| `kimi-linear-1p3b` | 8, FSDP 8 | 16 × 4096, forward-gradient cross entropy | 68,672 (23.0%) | 17.2 GiB |

- The forward-gradient cross entropy on the main and MTP heads (`train.ce_impl = "auto"`,
  `train.ce_grad_in_forward = true`) gives `qwen3-next-1p3b` 1.118x on one H100, 1.091x at FSDP 4 and 1.112x at
  FSDP 8 (4.7 GiB more per GPU, the same loss), 26.7-28.0% MFU on one GPU over its A/B rounds; the TOML sets it. For
  `kimi-linear-1p3b` it is 1.066x by median step on one H100, 1.092x together with one 16 × 4096 micro-step; that
  TOML keeps two micro-steps and the XLA cross entropy until an interleaved 8-GPU A/B checks them. FP8 (the dense
  projections only; the experts stay bf16) is about even with bf16: 1.003x and 1.017x.
- **Where the step goes** (`profile_1p3b.py`, one H100, forward-gradient cross entropy): the MoE layers take
  245.0 ms of the 855.2 ms `qwen3-next-1p3b` step (29.2%) and 270.4 ms of `kimi-linear-1p3b`'s 909.9 ms step at
  16 × 4096 (30.5%, level with its KDA kernels; 24.1% MFU), about half of it XLA gathers into and out of expert order
  ([ragged_dot.md](ragged_dot.md)); next in `qwen3-next-1p3b` come cuDNN attention (14.3%), the Gated DeltaNet
  kernels (8.9%) and the norms (7.3%).

## Where we lose, and limits

- **Narrow margins in bf16**: `transformer-swa-1p3b` is 1.04x flame, and `mamba2-1p3b` 1.05x against a flame
  Mamba-2 that skips document masking.
- **MFU credits attention work the kernel skips.** The FLOP count assumes full causal context, but the records hold
  about 4 documents each (3.97 EOS tokens per 4096-token record over the first 20,000 records), so masked attention
  does 45% of that work. For `transformer-1p3b` the attention core is 13% of the counted FLOPs, which makes MFU about
  8% (relative) higher than the work done. Tokens/s is unaffected.
- **Memory.** fp32 parameters, gradients and AdamW state take 24-31 GiB at this size. The recurrent presets keep
  full rematerialization: without it every step asks BFC for a 52-79 GiB block and fails with preallocation off.
  For Mamba-2 the replayed forward is about 16% of the bf16 step, but `remat = "kernels"` needs 92.1 GiB and
  `"dots"` 95.9 GiB (at 8 × 4096 × 2).
- **Compile time.** The no-recompute transformer step needs unrolled layers (scanned ones run 0.96x in bf16 and
  0.89x in FP8), which compile in about 150 s with FP8 and 70 s in bf16 (provisional). `kimi-linear-1p3b` compiles
  in 277 s: its dense first layer keeps the layer pattern from repeating, so the layer scan holds all 16 blocks in
  one iteration and XLA compiles them one by one.
- **GDN-2** has no baseline. Its FP8 step profiles at 1,784 ms (40.9% MFU) against 1,182 ms for `gdn-hd128-1p3b`,
  most of the 602 ms in its chunk kernels. `gdn2-1p3b.toml` keeps the
  forward-gradient cross entropy off: with the CE kernels its 20-step bf16 losses end about 0.02 below the XLA cross
  entropy's (7.814 against 7.833), which is not understood; the option measured 1.043x (bf16) and 1.052x (FP8).
- **The forward-gradient cross entropy** is set by `transformer-1p3b.toml`, `gdn-hd128-1p3b.toml` and
  `qwen3-next-1p3b.toml` only: `kda-1p3b` measured 0.954x on 8 GPUs, and `transformer-swa-1p3b` and `mamba2-1p3b`
  have no 8-GPU check.
- **RWKV-7** has no bf16 comparison (note (c)). Its layer glue (token shift and six mixes, the key update, group norm
  with bonus and gate) runs as XLA fusions, where FLA fuses it into Triton kernels.
- **The MoE hybrids** stay below the 35% MFU target: 35% is a 683 ms `qwen3-next-1p3b` step, which needs about 70%
  of the MoE's time removed, and for `kimi-linear-1p3b` even a free MoE would leave about 34% MFU. Expert
  parallelism at 1.3B has one measurement, on the MoE layer without the per-device dispatch and the persistent
  grouped GEMM: `qwen3-next-1p3b` at EP 8 with bounded blocks (buffer factor 2.0) trained at 19,866 tokens/s per
  GPU (7.2% MFU, 0.38x the one-GPU rate; provisional), where FSDP 8 with the per-device dispatch trains it at
  71,926.
- **The dense hybrids** (`gdn-hybrid-1p3b`, `kda-hybrid-1p3b`) are not in the release table.

## Correctness

Parameter counts are exact: `tests/test_models_lm.py::test_1p3b_preset_parameter_count_is_exact` pins each preset's
total (1,607,567,360 for `transformer-1p3b`), [`check_params.py`](end_to_end/1p3b/flame/check_params.py) finds
flame's count equal to jla's for every preset but RWKV-7 (above), and `tests/test_hybrid_presets_1p3b.py` pins the
hybrids'. Each run stores the sum of
the input ids it drew, which [`token_checksum.py`](end_to_end/1p3b/flame/token_checksum.py) recomputes from the
CIDX files (11,848,261,411 for 20 steps on one GPU, 95,816,696,411 on 8); `tests/test_bench_e2e_flame_data.py`
checks flame's rows, document boundaries and positions against jla's `DocumentFields`. The optimized train-step HLO
of each preset contains the expected kernels (fused conv + L2-norm in `gdn-hd128-1p3b`, KDA and GDN-2, `gdr_kb_*`
in `gdn-1p3b`, vector-gate kernels in KDA and GDN-2, the fused FP8 SwiGLU passes in FP8). A run is flagged if any
loss is non-finite or its last 5 losses are not below its first 5 on average. Runs are not bitwise reproducible:
the fused attention backward adds `dq` with TMA reductions, and final losses of repeated linear-attention runs
differ by up to 0.03. `tests/test_bench_e2e_1p3b.py` pins which TOMLs set the forward-gradient cross entropy, and
`tests/test_mtp_grad_in_forward.py`, `tests/test_mamba2_fused_multidevice.py`, `tests/test_mamba2_in_place_zx.py`,
`tests/test_rwkv7_layer_bounded.py` and `tests/test_gdn2_fast_paths.py` cover the 1.3B-specific paths above.

## Reproduce

```bash
# one run: 1 GPU, Mosaic GPU kernels, the TOML as written (--set a.b=value passes any further override)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    python benchmarks/bench_e2e_1p3b.py --arch kda-1p3b --data /path/to/fineweb_edu_3B --impl kernels \
    --dp 1 --fsdp 1 --steps 20 --skip 5 --allocator bfc --json results/kda-1p3b-1gpu.json
# FP8: add --set 'model.fp8="delayed"' --set model.fp8_min_features=512 --set model.fp8_fused=true
# qwen3-next-1p3b: add --attention cudnn; kimi-linear-1p3b: --micro-batch 8 --grad-accum 2
# the matrix, one JSON and one log per run (modes 1gpu, fsdp8, dp2fsdp4 and their -chunked forms)
export DATA=/path/to/fineweb_edu_3B OUT=results/e2e_1p3b
CUDA_VISIBLE_DEVICES=0 bash benchmarks/end_to_end/1p3b/run_matrix.sh 1gpu

# flame: the environment once, then flame's recommended mode
VENV=/path/to/venv-flame WORK=/path/to/build FA3_WHEEL=/path/to/flash_attn_3-*.whl \
    MAMBA_WHEEL=/path/to/mamba_ssm-*.whl bash benchmarks/end_to_end/1p3b/flame/setup_flame_env.sh
CUDA_VISIBLE_DEVICES=0 /path/to/venv-flame/bin/python benchmarks/bench_e2e_flame.py \
    --arch kda-1p3b --data /path/to/fineweb_edu_3B --nproc 1 --micro-batch 16 --steps 20 --skip 5 \
    --ac none --grad-accum 4 --flame-src /path/to/build/flame-src \
    --jla-python /path/to/jax-venv/bin/python --json results/flame-kda-1p3b.json
# transformers: add --fa3-shim; Mamba-3: --no-compile --grad-accum 16; FP8: --set=--model.converters=float8

# checks (CPU)
/path/to/venv-flame/bin/python benchmarks/end_to_end/1p3b/flame/check_params.py /path/to/jax-venv/bin/python src
python benchmarks/end_to_end/1p3b/flame/token_checksum.py /path/to/fineweb_edu_3B 128 20
```
