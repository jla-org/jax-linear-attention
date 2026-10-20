# The training stack

[README](../README.md) · [Architectures](architectures.md) · [Kernels](kernels.md) · [Training](training.md) · [Results](results.md) · [Benchmarks](../benchmarks/README.md)

How `jla train` is configured, and everything the trainer supports.

- [Configs and tokenizers](#configs-and-tokenizers)
- [Training](#training): [Parallelism](#parallelism) · [Context parallelism](#context-parallelism) · [FP8](#fp8) · [Rematerialization](#rematerialization) · [Initialisation](#initialisation) · [Data](#data) · [Supervised fine-tuning](#supervised-fine-tuning) · [Preference tuning](#preference-tuning) · [Logging](#logging) · [Checkpoints](#checkpoints) · [Generation](#generation) · [Evaluation](#evaluation) · [Hugging Face import and export](#hugging-face-import-and-export) · [Host-side step loop](#host-side-step-loop) · [XLA settings](#xla-settings)

## Configs and tokenizers

The presets' 50,304-token vocabulary fits the GPT-NeoX tokenizer. With another tokenizer, set
`--set model.vocab_size=<its vocabulary size>`. At startup the trainer checks a sample of the corpus
and refuses token ids that do not fit the model's vocabulary.

Every experiment is one TOML file (`configs/`); `--set a.b=value` overrides any field. The config
decides all behaviour. The only environment inputs are JAX's own (`CUDA_VISIBLE_DEVICES`,
`XLA_FLAGS`, `XLA_PYTHON_CLIENT_*`) and, when `runtime.cluster_env` is enabled, the cluster
launcher's variables (SLURM, Open MPI or torchrun).

## Training

### Parallelism

- **Parallelism** (`[mesh]`): data (`dp`), FSDP (`fsdp`) and tensor (`tp`) parallel meshes over
  logical axis names; expert parallelism (`ep`, all-to-all dispatch of routed experts,
  [expert_parallel.md](../benchmarks/expert_parallel.md)); pipeline parallelism (`pp`, a GPipe
  schedule inside the compiled step, after MaxText, [pipeline.md](../benchmarks/pipeline.md)). Under a
  multi-device mesh the Mosaic GPU kernels run per shard of the trainer's mesh, and so do the fused
  short convolutions (with the q/k L2 norm), the FP8 SwiGLU passes and the fused SwiGLU GEMMs
  ([multi_gpu.md](../benchmarks/multi_gpu.md#how-it-works)), and so do Mamba-2's fused output glue
  row kernels (except under a mesh that splits the heads, which keeps the unfused glue); the fused cross entropy's recompute path runs on one GPU only (XLA's cross entropy
  runs on several), while the forward-gradient cross entropy (`train.ce_grad_in_forward`)
  runs per shard, and under a multi-device mesh each device dispatches its own MoE tokens.
  Multi-process runs
  set `runtime.distributed = true` (one process per GPU). [multi_gpu.md](../benchmarks/multi_gpu.md)
  checks `dp`, `fsdp` and `dp x tp` against one GPU and resumes checkpoints across meshes.

### Context parallelism

- **Context parallelism**: ring or Ulysses softmax attention over a `cp` axis
  (`jla.distributed.context_parallel_attention`, with packed documents, windows, sinks and GQA;
  [attention.md](../benchmarks/attention.md#context-parallelism)), and state passing for
  `simple_gla`, `gla`, `deltanet`, `gated_deltanet` and `kda` (`jla.distributed.context_parallel`,
  on the chunked ops). Both are library APIs only: the trainer has no `cp` mesh axis and refuses
  `context_parallel_axis`.

### FP8

- **FP8** (`model.fp8 = "delayed"`): e4m3 inputs and weights, e5m2 gradients, per-tensor delayed scaling, after
  MaxText's GPU recipe and Flax. `model.fp8_min_features = 512` keeps small gate and low-rank projections in bf16.
  `model.fp8_fused = true` (delayed scaling only) runs each SwiGLU MLP with fused activation + FP8 cast + amax +
  transposed-copy passes (Pallas Triton, per shard under a multi-device mesh, one GPU's kernels in the per-device
  step of `train.grad_reduce = "step"`; plain-JAX passes where no split fits, as in the pipeline-parallel stages).
  Use delayed scaling, not `"current"` ([fp8_1p3b.md](../benchmarks/fp8_1p3b.md), [fp8.md](../benchmarks/fp8.md)).
  With `[model.attention] fp8_grad_epilogue = true` (off by default; on in the two transformer 1.3B TOMLs) the
  Mosaic GPU attention backward kernels write the q/k/v projections' e5m2 gradients and their amaxes (through the
  RoPE backward), which the projection GEMMs read directly: 1.011x for `transformer-1p3b` and 1.017x for
  `transformer-swa-1p3b` on one H100, and 1.038x and 1.013x for `transformer-1p3b` on 8 GPUs (FSDP 8,
  `grad_reduce = "step"`; provisional; `transformer-swa-1p3b` not yet timed on 8 GPUs). Each FP8 weight is
  quantized once per optimizer step (`model.fp8_weight_once`, default on) and that copy is reused across
  micro-steps, rematerialization and the backward pass; with the parameters on one device a transposed copy is kept
  too, and projections that share an input (attention q/k/v, linear-attention q/k/v/gate) read one e4m3 cast of it.
  This is exact: bitwise equal to quantizing per call on CPU through whole train steps, also with FSDP and pipeline
  parallelism (on GPU XLA may pick other GEMM algorithms), and delayed-scaling and checkpoint semantics are
  unchanged. With `fp8_fused = true` and `fp8_min_features = 512`, the weight copies, the shared input cast and the
  8-warp SwiGLU backward together are 1.06-1.07x on `transformer-1p3b` (62,743-63,499 tokens/s on one H100, 58.9-60.0%
  MFU, without the fused attention backward and the forward-gradient cross entropy), and on `gdn-hd128-1p3b` with FSDP
  over eight H100s 1.03-1.05x on the best steps and 1.074x in tokens/s (provisional;
  [fp8_1p3b.md](../benchmarks/fp8_1p3b.md#the-recipe-and-the-fused-paths)). With the TOML as written
  `transformer-1p3b` trains at 72.3% MFU in FP8 on one H100 ([Results](results.md#one-h100)). Against bf16, FP8 is
  1.08-1.30x on the 1.3B presets, 1.12x for Mamba-2 (43,688 against 38,899 tokens/s), whose TOML runs the fused
  output glue and the fused cross entropy ([end_to_end_1p3b.md](../benchmarks/end_to_end_1p3b.md#mamba-2)).

### Rematerialization

- **Rematerialization** (`model.remat`): `"full"` (the default, and the fastest policy measured at
  1.3B for the recurrent presets), `"none"`, `"dots"`, `"dots_with_no_batch_dims"`, `"save_attention"`, `"kernels"`,
  `"dots_kernels"`, and `"names"` / `"offload"`, which save, or move to pinned host memory, the
  checkpoint names in `model.remat_names` / `model.remat_offload_names` (`attention`, `linear`,
  `mlp_hidden`, `kernels`; FP8 projections and MLPs carry the same names)
  ([fp8_1p3b.md](../benchmarks/fp8_1p3b.md#activation-checkpointing-at-13b)). `model.lean_vjp = true` (memory-lean
  custom VJPs for norms, SwiGLU, short convs, gates and Mamba epilogues; with FP8 it needs `fp8_fused = true`, whose
  SwiGLU VJP is already lean) lets the 1.3B transformer configs drop the recompute: `remat = "none"`
  with lean VJPs and unrolled layers (`scan_layers = false`) at 4 × 4096 × 4 micro-steps is
  1.114x and 1.128x the full-remat step in bf16 on one H100 for the transformer (1.07-1.13x for SWA), 1.087x and
  1.102x on 8 GPUs and 1.01-1.02x with FP8, and both transformer TOMLs use it (provisional).
  For the recurrent presets the same with `model.lean_remat_core = true` fits at 4 × 4096 × 4 only in
  one preallocated pool (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` with preallocation on; with
  preallocation off it runs out of memory). It measured 0.98x (Gated DeltaNet) and 0.93-0.95x (KDA)
  of full remat (provisional, recorded in the comments of the two 1.3B TOMLs), so those configs keep full remat.
  `train.grad_accum` and `train.scan_unroll` set the micro-steps.

### Initialisation

- **Initialisation** (`model.init_scheme`): `"upstream"` (the default) follows FLA,
  state-spaces/mamba and BlinkDL's RWKV-LM; `"jla"` keeps jla's own initialisation. Under `"upstream"` the Mamba-2 and Mamba-3
  (SISO and MIMO) `out_proj` start at PyTorch's `kaiming_uniform_(a=sqrt(5))` divided by sqrt(the
  number of residual branches); the q/k/v short convs of Gated DeltaNet, DeltaNet, Gated
  DeltaProduct and KDA start at N(0, initializer_range) instead of an identity filter
  (`LinearAttentionConfig.conv_init = "normal"` or `"identity"` sets this per layer config and wins
  over the scheme; GLA, simple GLA and GDN-2 keep the identity filter); and RWKV-7 zero-initialises
  `o_proj`, the channel-mix value and the LoRA down projections, with orthogonal LoRA up projections
  of gain 0.1, so every RWKV-7 block is the identity at init (with the tied head of `rwkv7-0.1b` the
  step-0 loss is then about 13 instead of ln V). Only training from scratch changes: the same
  parameters give bit-identical outputs. A real-data A/B (FineWeb-Edu, 125M-130M presets, 2000
  steps of 32,768 tokens, 2 to 7 seeds per family; [init_parity.md](../benchmarks/init_parity.md))
  chose the initialisation per family. The upstream one lowers the held-out eval loss at step 2000
  by 0.0972 for DeltaNet, 0.1088 for Gated DeltaProduct, 0.100 and 0.103 for RWKV-7 (seeds 0 and 1),
  0.0387 for Mamba-3 SISO and 0.0195 for Mamba-2 (3 seeds), so the presets use it for the Mamba-2/3
  `out_proj`, the DeltaNet and Gated DeltaProduct convs and RWKV-7; Mamba-3 MIMO (0.0202 lower, not
  established on its own) follows the other Mamba variants. FLA's N(0, 0.02) short convs did not help
  the attention hybrids: +0.0160 ± 0.0131 for the Gated DeltaNet hybrid (7 seeds, the sign depends on
  the seed) and +0.0172 ± 0.0042 for the KDA hybrid (worse in 7 of 7 seeds and at every evaluation).
  So every preset with attention (or MLA) and Gated DeltaNet or KDA
  layers (the GDN and KDA hybrids at 125M and 1.3B, `moba-hybrid-125m`, `kimi-linear-1b`,
  `kimi-linear-1p3b`, `qwen3-next-300m` and `qwen3-next-1p3b`) sets `conv_init = "identity"`. Without
  the attention layers the same convs help: all-linear 125M stacks of the hybrids' Gated DeltaNet and
  KDA layers end 0.103 and 0.047 lower at step 2000 with them (three seeds each, lower in every
  seed), so the all-linear `gdn-1p3b`, `gdn-hd128-1p3b` and `kda-1p3b` leave `conv_init` unset
  and keep FLA's convs (a 125M result carried over; the 1.3B shapes were not trained in the A/B). The
  tiny test presets `tiny-hybrid`, `tiny-qwen3-next` and `kimi-linear-tiny` also keep FLA's convs, so
  that their model-level decode and packed-document tests see nonzero older conv taps.

### Data

- **Data**: memory-mapped token datasets (`data.source = "cidx"`, from `jla tokenize`) with
  `data.document_masking`; streamed text (`data.source = "stream"`: local `.jsonl` / `.parquet`
  files via `data.stream_files`, or a Hugging Face dataset via `data.hf_dataset`, tokenized on the
  fly, [streaming.md](../benchmarks/streaming.md)); supervised fine-tuning (below); synthetic tokens.

### Supervised fine-tuning

- **Supervised fine-tuning** (`data.source = "sft"`): `jla tokenize --format chat` (the tokenizer's
  chat template) or `--format instruction` (prompt/completion fields) writes tokens plus a mask
  sidecar of the trained tokens; by default the loss covers assistant turns only. The trainer packs
  whole conversations into rows by best-fit decreasing (`data.sft_packing`) with per-conversation
  `segment_ids` and positions; packed and unpacked rows give the same loss and gradients
  ([sft_packing.md](../benchmarks/sft_packing.md)). `configs/sft-chat-125m.toml` is an example; the
  trainer cannot yet start from a pretrained checkpoint, so it trains from scratch.

### Preference tuning

- **Preference tuning** (`jla dpo --config configs/post_train/dpo-tiny.toml`): DPO with the sigmoid
  (optionally conservative, `dpo.label_smoothing`), IPO and hinge losses on preference JSONL (text or
  tokens), starting from a `jla` checkpoint (`dpo.init_checkpoint`); the frozen reference model runs
  in the same compiled step with no gradient or optimizer state. The loss, its gradient and the
  reward metrics match TRL 1.16's `DPOTrainer` on fixed inputs; resume, gradient accumulation and
  multi-device runs work as in `jla train` ([dpo.md](../benchmarks/dpo.md)).

### Logging

- **Logging**: `train.loggers = ["jsonl", "wandb", "tensorboard"]` (JSONL is the default; `"wandb"`
  needs `train.wandb_project`, and TensorBoard writes to `train.tensorboard_dir`, by default
  `<output_dir>/tensorboard`); MFU, HFU and MBU in every log line; compile preflight, sharding audit
  and HLO graph-health checks.

### Checkpoints

- **Checkpoints**: crash-consistent checkpoints with a data-position sidecar; resume reproduces
  the uninterrupted run (bit for bit on CPU, and on GPU with `runtime.deterministic = true`).

### Generation

- **Generation** (`jla generate`, `jla.inference.generate`): prefill of the shared prompt prefix,
  then a jit-compiled decode loop with every layer family's cache (KV, ring, convolution and
  recurrent states). Each row is fed its own prompt, so padding never enters a cache. Greedy, or
  temperature, top-k and top-p sampling. NSA, DSA and log-linear Mamba-2 have no decode cache, so
  generation recomputes the forward pass for them.

### Evaluation

- **Evaluation**: `jla eval ppl` scores a text or token stream in strided windows (with
  `--document-eos`, it masks documents like training does); `jla eval lm-eval` runs
  lm-evaluation-harness tasks through `jla.eval.lm_eval_adapter.JLALM`.

### Hugging Face import and export

- **Hugging Face import and export** (`jla convert import | export | show-config`,
  [convert_hf.md](../benchmarks/convert_hf.md)): Llama 2/3/3.1+, Mistral, Qwen2/2.5, Qwen3, Phi-3,
  Mixtral, Qwen3-MoE, DeepSeek-V3 and Mamba-2 checkpoints to `LanguageModel` parameters and back.

### Host-side step loop

- **Host-side step loop**: `train.gc_freeze` (on by default): before the step loop, `jla train` and
  `jla dpo` run one `gc.collect()` and `gc.freeze()`, and call `gc.unfreeze()` when the loop ends.
  Python's first full garbage collection after compiling is a one-off host stall of a few tenths of a
  second that otherwise lands at some step of the run; with the option it happens before the first
  step. Training throughput does not change: short timed runs, such as the 20-step release protocol,
  read the steady state instead of the steady state plus the stall. Steps and results are unchanged,
  `train.gc_freeze = false` turns it off, and only what compiles before the loop is
  covered (with `train.compile_preflight = false` the train step, and always the eval step, compile
  after the freeze) ([fp8_1p3b.md](../benchmarks/fp8_1p3b.md#the-early-stall-and-traingc_freeze)).
  `train.dispatch_ahead` (off by default, exact; on in the `transformer-1p3b` TOML): a log step dispatches the
  next step before reading its own metrics (eval, save and profiler steps excluded; a stop request takes effect
  one step later; with several processes the stop check runs on the hosts, so it does not wait for the step in
  flight; `jla dpo` rejects it). On its own it measured 0.992x and 1.004x on 8 GPUs and 1.017x on one GPU
  for `transformer-1p3b` in FP8 (provisional; [multi_gpu.md](../benchmarks/multi_gpu.md#train-options-for-several-gpus)):
  on several GPUs it helps only together with `train.update_guard = "select"`, since with the default
  `lax.cond` the host waits for every step. With it on, the last step's metrics are
  read without the lag of the others, so the 1.3B harness leaves that step out of its timing.
  `train.update_guard` (`"cond"` by default; `"select"` in the `transformer-1p3b` TOML): how a rejected step
  (non-finite loss or gradient) keeps the parameters and optimizer state, a `lax.cond` around the update or
  the update computed and every leaf selected elementwise (the same rule; on GPU a few float32 ulps of the
  update differ; `optim.skip_spikes` keeps its own `lax.cond`).
  `train.grad_reduce` (`"micro_step"` by default; `"step"` in the `transformer-1p3b` TOML): `"step"` runs one GPU's micro-steps on every
  GPU (whole weights, the FP8 copies made once) and reduce-scatters the gradients once per step, one GPU's
  arithmetic up to the order of the float32 gradient sums. It refuses MoE, MTP, dropout, FP8 current scaling,
  the DSA indexer loss, Mamba-1 layers, tensor and pipeline parallelism and, on GPU, cuDNN or `"auto"`
  attention (set `"mosaic_gpu"`), and keeps about 10 bytes per parameter on every GPU whatever the FSDP size
  (about 30 GB at 3B; 7B does not fit on 80 GB), so it is for models up to about 3B. With the other two
  options it gives `transformer-1p3b` 68.2% MFU per GPU on 8 H100s ([Results](results.md#8-gpus)).

### XLA settings

- **XLA settings**: `runtime.xla_tuning_level` 0 (the default, no tuning flags), 1 (the
  latency-hiding scheduler, set by the 1.3B TOMLs and every config in `configs/` except
  `tiny-synthetic`, `tiny-stream` and `sft-chat-125m`) or 2 (plus pipelined collectives). No other
  XLA flag measured faster at 1.3B ([fp8_1p3b.md](../benchmarks/fp8_1p3b.md#xla-settings-at-13b)).

