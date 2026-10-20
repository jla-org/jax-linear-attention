# Initialisation A/B on FineWeb-Edu: upstream-parity inits against jla's own

The upstream-parity initialisation (`ModelConfig.init_scheme = "upstream"`, the library default) follows FLA,
state-spaces/mamba and BlinkDL; `init_scheme = "jla"` keeps jla's own. This A/B trains 125M-130M presets on real
data with both and picks each family's default. It compares held-out loss, not speed.

| family (preset) | jla's own init (OLD) | upstream-parity init (NEW) |
|---|---|---|
| Mamba-2, Mamba-3 SISO, Mamba-3 MIMO (`mamba2-130m`, `mamba3-130m`, `mamba3-mimo-130m`) | `out_proj` truncated normal, std 0.02 | `out_proj` kaiming-uniform `U(±1/sqrt(inner))` divided by `sqrt(residual branches)` (std 0.0030 at 130M) |
| DeltaNet, Gated DeltaProduct, Gated DeltaNet, KDA (`deltanet-125m`, `gated-deltaproduct-125m`, `gdn-hybrid-125m`, `kda-hybrid-125m`, and all-linear Gated DeltaNet and KDA stacks) | q/k/v short convs start as the identity filter | q/k/v short convs `N(0, 0.02)`, no bias (FLA's `ShortConvolution` init) |
| RWKV-7 (`rwkv7-0.1b`) | `o_proj`, channel-mix `value` and LoRA projections truncated normals | `o_proj`, channel-mix `value` and the LoRA down projections zero; LoRA up projections orthogonal with gain 0.1 |

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 compute with f32 parameters; one H100 per run, every mixer at its preset's
  `auto` implementation (the Mosaic GPU kernels where they apply). The two arms differ only in the
  initialisation.
- Data: the corpus of [end_to_end_1p3b.md](end_to_end_1p3b.md) (FineWeb-Edu, 3B tokens, Llama-3 tokenizer with
  128,256 ids, 4096-token packed records, document masking at EOS 128001), `model.vocab_size = 128256`.
- Recipe, identical for both arms: AdamW, lr 6e-4, b2 0.95, weight decay 0.1, gradient clipping 1.0, z-loss 1e-4;
  linear warmup over 200 steps, then cosine decay to 6e-5 at step 2000; 2000 steps of 32,768 tokens (65.5M tokens
  per run). RWKV-7 runs micro-batch 4 with 2 accumulation steps (the same tokens) and `chunk_size = 32`, so `auto`
  runs its Mosaic GPU kernels.
- The seed sets both the parameter-init key and the data order, so the two arms of a seed see the same batches in
  the same order. Seeds per family: 2, except the two attention hybrids (7), Mamba-2 (3) and the all-linear stacks
  (3).
- Evaluation: the last 2048 records of the corpus never reach training. Every 500 steps the trainer evaluates the
  first 512 of them, unshuffled: 2,095,155 tokens, the same in every run.
- Noise: the same seed run twice differs by up to 0.009 at step 2000 (the GPU kernels are not bitwise
  deterministic). A difference counts as real when it has the same sign in every seed and its mean is more than
  two standard errors from zero; a single-seed difference of at most 0.02 counts as noise (the rule was set before
  the runs).
- Trained on 18-19 Oct 2026.

## Results

Held-out eval loss at step 2000, NEW - OLD (negative: the upstream-parity init is better), mean ± standard error
over the seeds, and how many paired evaluations (steps 500, 1000, 1500 and 2000 of every seed) NEW was lower at.

| family (preset) | seeds | NEW - OLD at step 2000 | per seed | NEW lower | default |
|---|---|---|---|---|---|
| Mamba-2 (`mamba2-130m`) | 3 | -0.0195 ± 0.0015 | -0.022, -0.017, -0.020 | 12 of 12 | upstream |
| Mamba-3 SISO (`mamba3-130m`) | 2 | -0.0387 ± 0.0068 | -0.032, -0.046 | 8 of 8 | upstream |
| Mamba-3 MIMO (`mamba3-mimo-130m`) | 2 | -0.0202 ± 0.0150 | -0.005, -0.035 | 7 of 8 | upstream (not established on its own) |
| DeltaNet (`deltanet-125m`) | 2 | -0.0972 ± 0.0044 | -0.102, -0.093 | 8 of 8 | upstream |
| Gated DeltaProduct (`gated-deltaproduct-125m`) | 2 | -0.1088 ± 0.0026 | -0.106, -0.111 | 7 of 8 | upstream |
| RWKV-7 (`rwkv7-0.1b`) | 2 | -0.1003 ± 0.0040 (a) | -0.096, -0.104 | 8 of 8 | upstream |
| Gated DeltaNet hybrid (`gdn-hybrid-125m`) | 7 | +0.0160 ± 0.0131 | -0.030 to +0.063, NEW lower in 3 of 7 | 17 of 28 | identity convs (a tie) |
| KDA hybrid (`kda-hybrid-125m`) | 7 | +0.0172 ± 0.0042 | +0.002 to +0.034 | 0 of 28 | identity convs (NEW worse) |
| Gated DeltaNet, all-linear (`gdn-hybrid-125m` with 12 Gated DeltaNet layers) | 3 | -0.1031 ± 0.0073 | -0.094, -0.098, -0.118 | 12 of 12 | FLA's convs |
| KDA, all-linear (`kda-hybrid-125m` with 12 KDA layers) | 3 | -0.0475 ± 0.0012 | -0.047, -0.050, -0.046 | 11 of 12 | FLA's convs |

(a) The NEW seed-0 run went through NaN batches of the RWKV-7 kernel (23 rejected updates); that kernel bug is
fixed in the shipped kernel ([rwkv7.md](rwkv7.md#nan-losses-in-training-on-the-kernel-path-cause-and-fix)). On the fixed kernel both inits trained 2000 steps at both seeds
without a rejected update, and NEW is lower by 0.100 (seed 0) and 0.103 (seed 1).

**The `N(0, 0.02)` short convs help the all-linear stacks, not the attention hybrids.** DeltaNet and Gated
DeltaProduct gain about 0.10 at step 2000. The Gated DeltaNet hybrid's NEW runs lead by about 0.05 at step 500 in
every seed and tie by step 2000, and the KDA hybrid's NEW runs trail at every evaluation of every seed (0.053 at
step 1000). The same two mixers without the attention layers gain 0.103 (Gated DeltaNet) and 0.047 (KDA), so the
attention layers, not the mixer kinds, take the gain away.

The defaults are applied through the presets. Every preset with attention (or MLA) layers and Gated DeltaNet or
KDA layers sets `LinearAttentionConfig.conv_init = "identity"`: `gdn-hybrid-125m`, `kda-hybrid-125m`,
`moba-hybrid-125m`, `gdn-hybrid-1p3b`, `kda-hybrid-1p3b`, `kimi-linear-1b`, `kimi-linear-1p3b`, `qwen3-next-300m`
and `qwen3-next-1p3b`. The all-linear `gdn-1p3b`, `gdn-hd128-1p3b` and `kda-1p3b` leave `conv_init` unset and keep
FLA's convs, as do the DeltaNet and Gated DeltaProduct presets. In any config `conv_init = "normal"` gives FLA's
init and `conv_init = "identity"` jla's own.

## Limits

- 2000 steps of 32K tokens (about a third of a token per parameter) measure early training: an init that wins here
  can still wash out over a full run, and the ranking at step 500 differs from the one at step 2000 for some
  families.
- One learning rate (6e-4) for every family, and a second seed changes both the init draw and the data order.
- The all-linear runs are 125M stacks with the hybrids' mixer settings (Gated DeltaNet with 4 key and 8 value heads
  of 128, KDA with 6 heads of 128); the 1.3B presets they decide (`gdn-1p3b`, `gdn-hd128-1p3b`, `kda-1p3b`) carry
  the 125M result over.

## Correctness

Each run records a SHA-256 of its `src/jla` sources (`source_sha256`) and its recipe; within each preset every run
has the same recipe and overrides except the seed (`checks.recipe_identical_within_preset_except_seed`). The test
presets `tiny-hybrid`, `kimi-linear-tiny` and `tiny-qwen3-next` keep FLA's convs whatever the result: identity convs
have zero older taps at init, so the model-level cache and packed-document tests would pass with a lost conv cache
or a missed document reset (with FLA's convs 7 and 3 of those tests fail when each is broken on purpose).
`tests/test_init_parity.py` guards those convs.

## Reproduce

```bash
python benchmarks/bench_init_ab.py --preset <preset> --data /path/to/fineweb_edu_3B --seed <seed> --tag <old|new> \
    --json results/init_ab.json
# OLD arm: add --set model.init_scheme=jla
# NEW arm for the Gated DeltaNet and KDA hybrid presets, which keep identity short convs:
#   add --set model.linear_attention.conv_init=normal
# RWKV-7 adds: --micro-batch 4 --set train.grad_accum=2 --eval-batches 128
# all-linear Gated DeltaNet / KDA: --preset gdn-hybrid-125m / kda-hybrid-125m with
#   --set 'model.layer_types=["gated_deltanet", ...]' / '["kda", ...]' (12 entries), and both arms explicit:
#   --set model.linear_attention.conv_init=identity (OLD) or --set model.linear_attention.conv_init=normal (NEW)
```
