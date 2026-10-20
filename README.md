<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/banner-dark.png">
  <img alt="jax-linear-attention: fast linear attention, SSM and attention kernels for JAX on NVIDIA Hopper. 72.3% MFU on one H100 (1.3B transformer, FP8), 68.2% MFU per GPU on 8 H100s, kernels up to 5.37x faster (Mamba-3 MIMO against the official recurrence kernels), 1.04-1.67x flame end to end in bf16" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/banner-light.png">
</picture>

[![CI](https://github.com/jla-org/jax-linear-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/jla-org/jax-linear-attention/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jax-linear-attention.svg)](https://pypi.org/project/jax-linear-attention/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](#install)
[![JAX 0.11.2](https://img.shields.io/badge/JAX-0.11.2-8A2BE2.svg)](#install)
[![NVIDIA Hopper](https://img.shields.io/badge/NVIDIA-Hopper-76B900.svg)](#install)
[![Chat on Discord](https://img.shields.io/badge/chat-Discord-3D5AFE?logo=discord&logoColor=white)](https://discord.com/invite/rWuU7cWU4E)

**jax-linear-attention** brings FLA-class linear-attention kernels to JAX: Pallas Mosaic
GPU kernels for NVIDIA Hopper, forward and backward, for Mamba-2, Mamba-3, Gated DeltaNet, KDA, RWKV-7,
GLA and more, with an FA3-style attention kernel and a full LLM training stack around them. It is built
for researchers and engineers who train linear-attention, state-space and hybrid models in JAX. Every
kernel is exact by default and measured against the latest upstream code, and this page shows where we
lose as plainly as where we win.

<p align="center">
  <a href="#quick-start">Quick start</a> · <a href="#supported-architectures">Architectures</a> ·
  <a href="#benchmarks">Benchmarks</a> · <a href="#where-we-lose">Where we lose</a> ·
  <a href="#faq">FAQ</a> · <a href="docs/">Docs</a> · <a href="https://discord.com/invite/rWuU7cWU4E">Discord</a>
</p>

## Highlights

Measured on H100 GPUs on 20 Oct 2026 with no other jobs on the GPUs, unless noted. Kernel ratios are forward +
backward, the other implementation's time over ours.

- **72.3% MFU on one H100**: `transformer-1p3b` in FP8 with the TOML as written, 77,094 tokens/s
  ([results](docs/results.md#one-h100)).
- **68.2% MFU per GPU on 8 H100s** (FSDP 8): 72,709 tokens/s per GPU, **1.47x flame**
  ([results](docs/results.md#8-gpus)).
- **1.04-1.67x flame in bf16 and 1.10-1.77x in FP8** on one H100 for the transformer, sliding-window,
  Gated DeltaNet, KDA and Mamba-2 1.3B presets, and 1.55x for Mamba-3 in bf16, on the same tokens
  ([results](docs/results.md#one-h100)).
- **In the Gated DeltaNet-2 paper's own throughput setting** (its Figure 2: 1.3B hybrids on one H100,
  16K tokens per step), every model of the figure trains 1.14-1.41x faster than the paper reports with a
  32,000-token vocabulary, and 0.99-1.23x with Llama-3's 128,256. The Gated DeltaNet-2 hybrid runs at 52.1
  against 38.0 thousand tokens/s at 2K × 8 ([gdn2_paper.md](benchmarks/gdn2_paper.md)).
- **Mamba kernels**: the Mamba-2 scan 1.20-1.96x the official Triton kernels and the whole Mamba-2 layer
  1.17-1.41x the official fused training path, Mamba-3 SISO 1.58-2.32x and MIMO 2.70-5.37x the official
  recurrence kernels, Mamba-1 1.72-1.85x the official CUDA selective scan (`d_state` 16)
  ([Mamba-2](benchmarks/mamba2.md), [Mamba-3](benchmarks/mamba3.md),
  [MIMO](benchmarks/mamba3.md#mimo), [Mamba-1](benchmarks/mamba1.md)).
- **Delta-rule, RWKV-7 and GLA kernels**: GLA 1.52-1.77x FLA (dense), KDA 1.47-1.78x FLA, RWKV-7
  1.06-1.46x FLA, Gated DeltaNet 1.71-2.68x FLA's Triton kernels and 0.73-1.26x FlashQLA
  ([GLA](benchmarks/gla.md), [KDA](benchmarks/kda.md), [RWKV-7](benchmarks/rwkv7.md),
  [Gated DeltaNet](benchmarks/gdn.md)).
- **Exact, packed and long**: fp32 states and accumulation, exact packed-document masks in every kernel,
  and single calls of 1,064,960 tokens that match chained halves ([long contexts](benchmarks/long_context.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/kernels-dark.png">
  <img alt="Kernel speed against the strongest latest upstream code, forward + backward, on one H100" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/kernels-light.png">
</picture>

*Forward + backward on one H100 against the latest upstream code named on each row: bars are medians over
the benchmark shapes, whiskers the range, and right of 1x ours is faster. Every shape and version:
[docs/kernels.md](docs/kernels.md#kernels).*

## Start here

| You want to | Go to |
|---|---|
| Call a kernel from your own JAX model | [Quick start](#quick-start): `jla.ops.linear_attention(..., impl="auto")` |
| Train a model from one TOML file | [Quick start](#quick-start): `jla train --config ...`, then [docs/training.md](docs/training.md) |
| See which layers have kernels and how fast they are | [Supported architectures](#supported-architectures) and [docs/kernels.md](docs/kernels.md) |
| Check the speed claims or reproduce them | [Benchmarks](#benchmarks) and [benchmarks/README.md](benchmarks/README.md) |
| Know where it is slower than the alternatives | [Where we lose](#where-we-lose) |
| Contribute a kernel, a layer or a fix | [Contributing](#contributing) |

## Why jax-linear-attention

The layers behind newer language models (Mamba, Gated DeltaNet, KDA, RWKV-7, GLA) are recurrences, and
they need dedicated kernels to train fast. PyTorch has them in Flash Linear Attention's Triton kernels.
In JAX these layers have mostly run as plain JAX ops (MaxText's Gated DeltaNet, for example), which XLA
cannot fuse into one efficient pass. jax-linear-attention brings kernels of FLA's class to JAX on
NVIDIA Hopper, with the training stack to use them, and keeps every kernel exact, so switching the
implementation never changes the model.

## Supported architectures

Kernels are Pallas Mosaic GPU kernels for H100 (sm_90a), picked by `impl="auto"` unless marked opt-in.
Speed is forward + backward against the strongest latest upstream code (the other implementation's time
over ours); "-" means no upstream comparison, and "provisional" marks a cell measured while other jobs could
share the machine.

| Architecture | Paper | Kernels | Speed against upstream | Notes |
|---|---|---|---|---|
| **Mamba-2** (SSD) | [2405.21060](https://arxiv.org/abs/2405.21060) | fwd + bwd | 1.20-1.96x official Triton scan | whole layer 1.17-1.41x the official fused path (forward + backward; forward 1.12-1.34x) |
| **Mamba-3** SISO | [2603.15569](https://arxiv.org/abs/2603.15569) | fwd + bwd | 1.58-2.32x official Triton | fused layer core (biases, angles, rotation) |
| **Mamba-3** MIMO | [2603.15569](https://arxiv.org/abs/2603.15569) | fwd + bwd | 2.70-5.37x official TileLang recurrence kernels | ranks 1, 2, 4 and 8 |
| **Mamba-1** | [2312.00752](https://arxiv.org/abs/2312.00752) | fwd + bwd | 1.72-1.85x official CUDA scan (`d_state` 16) | `d_state` 4-32; Jamba, Falcon-Mamba |
| **Gated DeltaNet**, DeltaNet | [2412.06464](https://arxiv.org/abs/2412.06464), [2406.06484](https://arxiv.org/abs/2406.06484) | fwd + bwd | 0.73-1.26x FlashQLA, 1.71-2.68x FLA Triton | exact sequence split; `K = 256` kernels 1.17-1.24x FLA |
| **Gated DeltaNet-2** | [2605.22791](https://arxiv.org/abs/2605.22791) | fwd + bwd | - (clean-room, no upstream code) | 6.7-10.8x our plain-JAX op |
| **KDA** | [2510.26692](https://arxiv.org/abs/2510.26692) | fwd + bwd | 1.47-1.78x FLA | 1.47-1.78x dense, 1.61-1.63x with 16 packed documents |
| **RWKV-7** | [2503.14456](https://arxiv.org/abs/2503.14456) | fwd + bwd | 1.06-1.46x FLA | 1.06x at the 1.3B layer shape, 1.16-1.29x with 16 packed documents |
| **GLA** | [2312.06635](https://arxiv.org/abs/2312.06635) | fwd + bwd | 1.52-1.77x FLA (dense) | 1.01-1.41x with 16 packed documents |
| **HGRN2** | [2404.07904](https://arxiv.org/abs/2404.07904) | GLA core | as GLA | no layer-wise forget-gate lower bound |
| **mLSTM** (xLSTM) | [2405.04517](https://arxiv.org/abs/2405.04517) | opt-in fwd + bwd | 1.03-1.06x TFLA Triton | `auto` keeps the float32 chunked op |
| **Gated DeltaProduct** | [2502.10297](https://arxiv.org/abs/2502.10297) | opt-in (GDN kernels) | - | 11.4x our plain-JAX op (provisional) |
| **RWKV-6** | [2404.05892](https://arxiv.org/abs/2404.05892) | plain JAX | - | |
| **Log-linear Mamba-2** | [2506.04761](https://arxiv.org/abs/2506.04761) | plain JAX | - | training only |
| **TTT-Linear** | [2407.04620](https://arxiv.org/abs/2407.04620) | plain JAX | - | |
| **Lightning attention** (MiniMax-01) | [2501.08313](https://arxiv.org/abs/2501.08313) | `simple_gla` kernels | - | bf16, head dim 64 or 128 |
| **Softmax attention** (FA3-style kernel) | [2407.08608](https://arxiv.org/abs/2407.08608) | fwd + bwd | 0.77-0.92x FA3 (causal; 0.87-1.00x with the opt-in fused backward) | `auto` runs it for packed documents, cuDNN for dense causal and windowed calls |
| **Sliding window** | - | fwd + bwd | 1.10-1.38x FA3 at 128 tokens, 0.81-0.96x at 1K-4K | GQA, sinks, mixed local and global layers |
| **MLA** | [2405.04434](https://arxiv.org/abs/2405.04434) | fwd + bwd | 1.01-1.05x cuDNN, 1.01-1.06x FA3 (causal) | DeepSeek-V3, Kimi K2, Kimi Linear heads |
| **DSA** (DeepSeek Sparse Attention) | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | fwd + bwd | indexer + top-k 1.16-1.43x DeepGEMM + `torch.topk` (provisional) | forward, kernel time; selected-key attention is masked-dense |
| **FoX**, **PaTH** | [2503.02130](https://arxiv.org/abs/2503.02130), [2505.16381](https://arxiv.org/abs/2505.16381) | plain JAX | - | full `[T, T]` scores |
| **MoBA**, **NSA** | [2502.13189](https://arxiv.org/abs/2502.13189), [2502.11089](https://arxiv.org/abs/2502.11089) | plain JAX | - | NSA is training only |
| **MoE** (dropless) | - | grouped GEMM fwd + bwd | 0.94-1.04x `torch._grouped_mm` (provisional) | shared experts, expert parallelism, MTP |
| **Qwen3-Next-like** hybrid | [model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | GDN + attention + MoE | - | 3 Gated DeltaNet : 1 gated attention, MoE, MTP (`qwen3-next-1p3b`) |
| **Kimi-Linear-like** hybrid | [2510.26692](https://arxiv.org/abs/2510.26692) | KDA + MLA + MoE | - | 3 KDA : 1 NoPE MLA, MoE (`kimi-linear-1p3b`) |

Presets also follow the layouts of Nemotron-H, Granite-4.0-H, gpt-oss, Gemma 3, MiniMax-01, Jamba and
DeepSeek-V3 at small scale, and the 1.3B presets take FLA's published shapes. Every preset, the ops API
and the ports: [docs/architectures.md](docs/architectures.md) and [docs/kernels.md](docs/kernels.md).

## Install

Requirements: Python 3.12 or newer and JAX 0.11.2 (the version is pinned). The Mosaic GPU
kernels need an NVIDIA Hopper GPU (H100/H200, sm_90a) and a driver for CUDA 13. On any other device
every op falls back to its plain-JAX implementation.

```bash
pip install "jax-linear-attention[cuda]"   # Linux + NVIDIA GPU (CUDA 13 wheels)
# the same, pinned to the tested CUDA 13 stack:
pip install -c https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/requirements/jax-0.11.2-cuda13.txt "jax-linear-attention[cuda]"
pip install jax-linear-attention           # CPU only (reference and chunked ops, small models)
```

The wheel holds the `jla` package and command-line tool. For the configs, benchmarks and tests, install
from a clone:

```bash
git clone https://github.com/jla-org/jax-linear-attention.git && cd jax-linear-attention
pip install -e ".[cuda,test]"          # Linux + NVIDIA GPU (CUDA 13 wheels)
pip install -c requirements/jax-0.11.2-cuda13.txt -e ".[cuda,test]"   # the same, pinned to the tested CUDA 13 stack
pip install -e ".[test]"               # CPU only (reference and chunked ops, small models)
```

Optional extras:

| extra | installs | needed for |
|---|---|---|
| `orbax` | `orbax-checkpoint` | sharded multi-host checkpoints |
| `tokenize` | `transformers`, `jinja2` | Hugging Face tokenizers: `jla tokenize` (chat templates for SFT need `jinja2`), `jla generate`, `jla eval` |
| `stream` | `datasets`, `pyarrow` | `data.source = "stream"` from a Hugging Face dataset or `.parquet` files |
| `loggers` | `wandb`, `tensorboardX` | `train.loggers = ["wandb"]` or `["tensorboard"]` |
| `eval` | `lm-eval` | `jla eval lm-eval` (with `tokenize`) |

## Quick start

The Python package and the command-line tool are `jla`. Call a kernel; `impl="auto"` runs the Mosaic
GPU kernel on an H100 and the chunked JAX op anywhere else:

```python
import jax
import jax.numpy as jnp
from jla import ops

B, T, H, K, V = 2, 2048, 8, 128, 128
kq, kk, kv, kg, kb = jax.random.split(jax.random.key(0), 5)
q = jax.random.normal(kq, (B, T, H, K), jnp.bfloat16)
k = jax.random.normal(kk, (B, T, H, K), jnp.bfloat16)
v = jax.random.normal(kv, (B, T, H, V), jnp.bfloat16)
g = jax.nn.log_sigmoid(jax.random.normal(kg, (B, T, H))) / 4  # log-decay
beta = jax.nn.sigmoid(jax.random.normal(kb, (B, T, H)))        # write strength
segment_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32) // 512, (B, T))  # 4 packed documents per row

o, state = ops.linear_attention("gated_deltanet", q, k, v, g=g, beta=beta, use_qk_l2norm=True,
                                output_final_state=True, segment_ids=segment_ids)
print(o.shape, state.shape)  # (2, 2048, 8, 128) (2, 8, 128, 128)
```

Train, evaluate and convert from the command line, in a clone of the repository (the configs are not in
the wheel). Every experiment is one TOML file in `configs/`, and `--set a.b=value` overrides any field:

```bash
# Train a tiny model on synthetic tokens (runs on CPU in a few minutes)
jla train --config configs/tiny-synthetic.toml

# Tokenize a corpus into the memory-mapped format, then train a 125M model on it
jla tokenize --input train.jsonl --output data/train --tokenizer EleutherAI/gpt-neox-20b
jla train --config configs/transformer-125m.toml --set 'data.paths=["data/train"]'

# Generate, score and evaluate a trained run
jla generate --run-dir runs/gdn --tokenizer EleutherAI/gpt-neox-20b --prompt "The capital of France is" --max-new-tokens 32
jla eval ppl --run-dir runs/gdn --cidx data/val --seq-len 2048 --document-eos 0
jla eval lm-eval --run-dir runs/gdn --tokenizer EleutherAI/gpt-neox-20b --tasks hellaswag,arc_easy --batch-size 16

# Import a Hugging Face checkpoint (a local directory; nothing is downloaded)
jla convert import --hf path/to/Qwen3-0.6B --output ckpt/qwen3-0.6b
```

Tokenizers, vocabulary sizes and the config rules: [docs/training.md](docs/training.md#configs-and-tokenizers).

## Training stack

One compiled train step with fp32 gradient accumulation and non-finite step rejection, configured by one
TOML file:

- **Parallelism**: data (`dp`), FSDP (`fsdp`), tensor (`tp`), expert (`ep`, all-to-all dispatch) and
  pipeline (`pp`, GPipe inside the compiled step) over logical axis names; context parallelism (ring or
  Ulysses attention, recurrent state passing) as library APIs, not yet a trainer mesh axis. Kernels run
  per shard of the mesh; one process per GPU with `runtime.distributed = true`.
- **FP8** with delayed scaling (`model.fp8 = "delayed"`): e4m3 inputs and weights, e5m2 gradients, each
  weight quantized once per optimizer step, fused SwiGLU FP8 passes, and opt-in FP8 q/k/v gradients
  written by the attention backward kernels.
- **Memory**: remat policies (`full`, `none`, `dots`, `save_attention`, `kernels`, named saves and
  host offload) and memory-lean VJPs; the 1.3B transformer configs train without recompute.
- **Packed documents** everywhere: `segment_ids` in every kernel and document masking in the data path.
- **MoE and MTP**: dropless MoE on grouped-GEMM kernels with per-device dispatch, shared experts and
  expert parallelism; multi-token prediction heads.
- **Fused LM-head cross entropy** that never writes the logits and can form its gradient in the forward
  pass.
- **8-GPU options**: `train.grad_reduce = "step"` (one gradient reduce-scatter per step),
  `train.update_guard = "select"` and `train.dispatch_ahead = true`, which take `transformer-1p3b` to
  68.2% MFU per GPU ([details](docs/training.md#host-side-step-loop)).
- **Data**: memory-mapped token datasets with document masking, streamed `.jsonl` / `.parquet` files
  or Hugging Face datasets, synthetic tokens.
- **Post-training**: SFT with chat templates and best-fit packing; DPO, IPO and hinge preference tuning
  (`jla dpo`), matched against TRL.
- **Checkpoints**: crash-consistent, with exact resume (bit for bit on CPU and with
  `runtime.deterministic = true` on GPU); Orbax for sharded multi-host checkpoints.
- **Generation and evaluation**: `jla generate` with every layer family's decode cache and greedy,
  temperature, top-k and top-p sampling; perplexity and lm-evaluation-harness (`jla eval`).
- **Hugging Face import and export**: Llama 2/3/3.1+, Mistral, Qwen2/2.5, Qwen3, Phi-3, Mixtral,
  Qwen3-MoE, DeepSeek-V3 and Mamba-2 checkpoints (`jla convert`).
- **Upstream-parity initialisation** (FLA, state-spaces/mamba and RWKV-LM), JSONL, Weights & Biases and
  TensorBoard logging with MFU, HFU and MBU in every line, and compile and sharding checks.

Every option, default and measurement: [docs/training.md](docs/training.md).

## How it works

- **Three implementations of every recurrence**: a token-by-token reference (the oracle), a
  chunk-parallel JAX op that runs on any JAX backend, and a Pallas Mosaic GPU kernel for Hopper. The
  tests hold each kernel to the reference on dense inputs, packed documents, initial and final states
  and every gradient.
- **Fail-closed dispatch**: `impl="auto"` runs a kernel only for the shapes, dtypes and options its GPU
  tests cover; every other call runs the chunked JAX op, so an unsupported call is slower, never wrong.
- **Packed documents everywhere**: `segment_ids` reset the state at document boundaries inside every
  kernel, so a packed batch trains exactly as its documents would one by one.
- **Layers and trainer**: the layers are Equinox modules that follow FLA's layer definitions, and the
  trainer adds parallelism, FP8, remat and the rest of the [training stack](#training-stack) from one
  TOML file per experiment.

## jax-linear-attention, FLA on PyTorch, or plain JAX

| | Plain JAX ops | FLA on PyTorch | jax-linear-attention |
|---|---|---|---|
| Framework | JAX | PyTorch | JAX |
| Linear-attention GPU kernels | none (XLA ops) | Triton, plus TileLang, FlashQLA and FlashKDA backends | Pallas Mosaic GPU for H100, forward and backward |
| Speed on one H100 | the baseline | the strongest PyTorch option | the Mamba-2 scan 1.20-1.96x and Mamba-3 MIMO 2.70-5.37x the official recurrence kernels, KDA 1.47-1.78x FLA, RWKV-7 1.06-1.46x FLA, Gated DeltaNet 1.71-2.68x FLA Triton; slower than FlashQLA and FA3 in [some cases](#where-we-lose) |
| Results | exact | exact, with approximate options in some backends (FlashQLA's sequence split) | exact by default |
| Training stack | bring your own | flame (torchtitan) | built in: FSDP, TP, EP, PP, FP8, MoE, SFT, DPO, Hugging Face conversion |
| 1.3B training on one H100 | | flame | 1.04-1.67x flame in bf16, 1.10-1.77x in FP8 |
| Hardware for the kernels | any JAX backend | GPUs Triton supports | NVIDIA Hopper (H100, H200); plain JAX elsewhere |

## Benchmarks

Every number comes from a linked page with its shapes, versions, method and commands.
[benchmarks/README.md](benchmarks/README.md) explains the timing method, lists every page and gives the
commands that reproduce the published numbers.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/e2e-dark.png">
  <img alt="1.3B training throughput against flame on one H100, bf16 and FP8, for the 1.3B presets" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/e2e-light.png">
</picture>

**End to end at 1.3B.** On one H100 and the same tokens, the transformer, sliding-window, Gated
DeltaNet, KDA and Mamba-2 presets train 1.04-1.67x faster than flame (FLA's trainer, with `torch.compile`
where it works) in bf16 and 1.10-1.77x in FP8, and Mamba-3 1.55x in bf16.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/mfu-dark.png">
  <img alt="MFU of the 1.3B transformer and Gated DeltaNet presets against flame: one H100 in bf16 and FP8, eight H100s in FP8" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/mfu-light.png">
</picture>

**MFU.** `transformer-1p3b` reaches 72.3% MFU in FP8 on one H100 and 68.2% per GPU on eight, and
`gdn-hd128-1p3b` 55.0% and 52.3% against flame's 36.1% and 33.4%. MFU is model FLOPs over the 989 TFLOP/s
bf16 spec; a power-capped H100 sustains about 681 TFLOP/s in bf16.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/gdn2-paper-dark.png">
  <img alt="Training throughput in the Gated DeltaNet-2 paper's setting: every 1.3B model of the paper's Figure 2 at 4K x 4 tokens on one H100, the paper's numbers against ours with a 32K and a 128K vocabulary" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/gdn2-paper-light.png">
</picture>

**In the Gated DeltaNet-2 paper's setting.** The paper's Figure 2 times 1.3B hybrids on one H100 at 16,384
tokens per step. In that setting, every model of the figure trains 1.14-1.41x faster than the paper's numbers
with a 32,000-token vocabulary, and 0.99-1.23x with Llama-3's 128,256. The Gated DeltaNet-2 hybrid runs at
52.1 and 49.1 thousand tokens/s at 2K × 8 and 16K × 1, against the paper's 38.0 and 36.1
([gdn2_paper.md](benchmarks/gdn2_paper.md) has the settings and the caveats).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/sweep-dark.png">
  <img alt="Forward + backward speedup across sequence lengths from 2K to 64K tokens for Mamba-2, Gated DeltaNet, GLA and causal softmax attention, against the fastest upstream code at each shape" src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/sweep-light.png">
</picture>

**Across sequence lengths.** Forward + backward from 2K to 64K tokens against the fastest upstream code
that runs each shape: Mamba-2 and GLA lead at every length and head layout, Gated DeltaNet trails FlashQLA
on long single rows of 8 heads of dim 128, and causal attention stays below FA3 (0.73-0.90x with the default
backward; [attention.md](benchmarks/attention.md#results)).

1.3B presets, tokens/s per GPU (MFU), 65,536 tokens per step on one H100 and 524,288 on eight (FSDP 8),
against flame on the same shapes and tokens:

| preset | 1 GPU, bf16 | vs flame | 1 GPU, FP8 | vs flame FP8 | 8 GPUs, FP8 | vs flame FP8 |
|---|---|---|---|---|---|---|
| `transformer-1p3b` | 60,178 (56.4%) | 1.15x | 77,094 (72.3%) | 1.41x | 72,709 (68.2%) (d) | 1.47x |
| `transformer-swa-1p3b` | 55,188 (50.1%) | 1.04x | 68,072 (61.8%) | 1.21x | | |
| `gdn-hd128-1p3b` | 42,995 (42.3%) | 1.67x | 55,956 (55.0%) | 1.52x | 53,208 (52.3%) (d) | 1.57x |
| `gdn-1p3b` (d) | 37,759 (36.0%) | 1.34x | 46,971 (44.8%) | 1.22x | | |
| `kda-1p3b` | 35,110 (32.9%) | 1.49x | 42,756 (40.1%) | 1.77x | | |
| `mamba2-1p3b` | 38,899 (37.1%) | 1.05x (a) | 43,688 (41.7%) | 1.10x (a) | | |
| `mamba3-1p3b` | 32,976 (32.6%) | 1.55x | 36,344 (36.0%) | 3.41x (b) | | |
| `rwkv7-1p3b` | 28,484 (26.2%) | - (c) | 30,651 (28.2%) | 1.90x (c) | | |

Measured on 20 Oct 2026 with no other jobs on the GPUs, one run per configuration, flame on the same day.
(a) flame's Mamba-2 trains without document masking. (b) flame's Mamba-3 cannot compile, and its FP8 run is half as
fast as its bf16 run (our FP8 is 1.71x that). (c) flame's compiled RWKV-7 in bf16 trains to a much worse loss, so we
claim no bf16 comparison. (d) Measured the same day on the code just before the last kernel changes of this
release, which leave `gdn-1p3b`'s `K = 256` kernels as they were (on one GPU those changes moved the other presets
by 0.98-1.03x). All footnotes, GDN-2 and the 8-GPU protocol: [docs/results.md](docs/results.md#one-h100).

## Where we lose

- **Softmax attention against FA3**: the forward is 0.90-0.92x FA3 on causal attention, short of the
  0.95x target, and forward + backward 0.77-0.92x with the default backward (0.87-1.00x with the opt-in
  fused one); `impl="auto"` keeps cuDNN for dense causal and windowed calls.
- **Gated DeltaNet against FlashQLA** on long single sequences (one row of 16K-64K tokens: 0.69-0.78x
  forward + backward at 8 heads, 0.68-0.73x at 16) and with packed documents at small `B * H` (one row
  of 8 heads with 16 documents: 0.74-0.77x from 16K tokens, 0.58-0.60x at 2K-8K). FlashQLA starts each
  sequence piece a few chunks early, from zero (an approximation); our split is exact, and its backward
  is two passes against FlashQLA's one fused kernel. At head dim 64, which FlashQLA does not support,
  packed documents in short rows also trail FLA (0.83-0.93x forward + backward at 2K-8K tokens).
- **KDA inference against FlashKDA** at large `B * H` (0.57x at `(8, 2048, 32)`; FlashKDA keeps a bf16
  state and approximate `exp`, and fuses its per-chunk preparation into one kernel).
- **Hopper only, one JAX version**: the kernels need an H100/H200 (everything else runs plain JAX), and
  JAX is pinned to 0.11.2 because Mosaic GPU's API changes with every JAX release. A few kernels (the
  Mamba-2 row kernels, the Gated DeltaNet piece scan, `fp8_swiglu`) are Pallas Triton, which JAX 0.11.2
  deprecates; they move to Mosaic GPU before the pin does.

Every case with its numbers and the reason: [docs/kernels.md](docs/kernels.md#where-we-lose).

## Status and limitations

- Early development (0.1.0): the APIs will change.
- The release numbers were measured on 20 Oct 2026 with no other jobs on the GPUs. Numbers marked provisional were
  measured while other jobs could share the machine.
- Context parallelism is a library API; the trainer has no `cp` mesh axis yet.
- SFT trains from scratch: the trainer cannot yet start from a pretrained checkpoint (DPO starts from a
  `jla` checkpoint).
- NSA, DSA and log-linear Mamba-2 have no decode cache; generation recomputes the forward pass for them.
- The 1.3B MoE hybrids stay below the 35% MFU target on one H100 (provisional): `qwen3-next-1p3b` at 26.7%
  with its TOML (28.4% with no other jobs on the machine), `kimi-linear-1p3b` at 22.7% with its TOML and 24.6%
  with one 16 × 4096 micro-step and the forward-gradient cross entropy; about half of their MoE time is XLA
  gathers into and out of expert order.

## FAQ

**Does it run without an H100?** Yes. Every op has a chunked JAX implementation that runs on any JAX
backend, and `impl="auto"` uses it wherever the Mosaic GPU kernels don't apply. The tests run on CPU and
H100; the kernel speedups need Hopper (H100 or H200).

**Why is JAX pinned to 0.11.2?** Mosaic GPU's Python API changes between JAX releases, and every kernel
is tested against one version. Moving to a newer JAX means testing every kernel again, so it comes with a
new release of this library.

**Are the kernels exact?** They compute the same recurrence as the reference, with float32 states and
accumulation: outputs match the token-by-token reference to bf16 rounding, and several kernel paths are
bitwise equal to each other. The few approximate options are opt-in, off by default and labelled.

**How is MFU measured?** Model FLOPs per second, counted analytically from the model (the same count for
flame), over the H100's dense bf16 peak of 989 TFLOP/s, timed by wall clock from one completed step to
the next. FP8 runs use the same bf16 peak.

**Can I use the kernels without the trainer?** Yes. `jla.ops` takes plain `(B, T, H, D)` arrays and
works under `jax.jit`, `jax.grad`, `shard_map` and FSDP sharding.

**Does it do inference?** Yes: prefill, decode with each layer's recurrent state, and `jla generate`.
The kernels are tuned for training; inference-only kernels such as FlashKDA's forward are faster in some
cases.

**How do I add a layer or a kernel?** Start with a reference, a chunked op and their tests; a kernel joins
`impl="auto"` only once its GPU tests match the reference. [CONTRIBUTING.md](CONTRIBUTING.md) has the
details.

## Documentation

| page | what it covers |
|---|---|
| [docs/architectures.md](docs/architectures.md) | overview, every preset (1.3B and published layouts), the other ports |
| [docs/kernels.md](docs/kernels.md) | the ops API, the kernel tables per family, attention routing, long contexts, the kernels around the mixers, where we lose |
| [docs/training.md](docs/training.md) | configs, parallelism, FP8, rematerialization, initialisation, data, SFT, DPO, checkpoints, generation, evaluation, conversion, the 8-GPU options |
| [docs/results.md](docs/results.md) | 1.3B results on one and eight H100s against flame, FP8, MFU and the H100's ceilings, the hybrid layouts, the index of benchmark pages |
| [benchmarks/README.md](benchmarks/README.md) | timing method, every kernel benchmark, reproduction |
| [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md) | release notes; setup, tests and the DCO sign-off |

## Development

All tests run on CPU except the GPU-only kernel and cuDNN tests, which skip there:

```bash
pytest tests/
JAX_PLATFORMS=cpu pytest tests/   # on a GPU host: also runs the kernels' CPU interpreter tests
pytest tests/test_long_context.py --run-long-context   # opt-in, one H100 (calls past 2^31 elements)
pytest tests/ --run-slow   # opt-in, H100: also the slow GPU kernel cases (minutes each)
```

The benchmarks compare against PyTorch libraries, so they need a separate torch environment;
see [benchmarks/README.md](benchmarks/README.md).

## Contributing

Contributions are welcome, with a DCO sign-off on every commit (`git commit -s`):
[CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the tests and the kernel conventions.

- **Questions and ideas**: [Discussions](https://github.com/jla-org/jax-linear-attention/discussions)
- **Chat**: [Discord](https://discord.com/invite/rWuU7cWU4E)
- **Bugs**: [open an issue](https://github.com/jla-org/jax-linear-attention/issues/new/choose)
- **Security problems**: [report them privately](https://github.com/jla-org/jax-linear-attention/security/advisories/new), never in a public issue
- **Everything else**: [vikramlex@karlexai.com](mailto:vikramlex@karlexai.com) · [@vikram_lex on X](https://x.com/vikram_lex)

## Citation

If you use jax-linear-attention, please cite it. [CITATION.cff](CITATION.cff) has the metadata, and
GitHub's "Cite this repository" button reads it.

```bibtex
@software{jax_linear_attention,
  title   = {jax-linear-attention},
  author  = {{KarLex AI, Inc.}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jla-org/jax-linear-attention},
  license = {Apache-2.0}
}
```

## Acknowledgements

This project stands on the work of others, and we are grateful to them.
Our linear-attention kernels port the Triton kernels of
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention), our linear-attention layers follow
its layer definitions, and its trainer, flame, is our end-to-end baseline.
[FlashAttention-3](https://github.com/Dao-AILab/flash-attention) shaped the design of our softmax attention
kernel. [Mamba](https://github.com/state-spaces/mamba)'s Triton and TileLang kernels are what our Mamba
kernels port, and its modules are the reference for every Mamba layer.
[FlashQLA](https://github.com/QwenLM/FlashQLA) showed the way for the fused Gated DeltaNet kernels, and
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) for the KDA forward.
[Tokamax](https://github.com/openxla/tokamax) and [JAX](https://github.com/jax-ml/jax)'s Pallas Mosaic GPU
examples are where our attention and GEMM kernels began, and JAX and Pallas make all of it possible. The
full list, with what each project contributed: [docs/acknowledgements.md](docs/acknowledgements.md) and
[NOTICE](NOTICE).

## License

Apache-2.0: see [LICENSE](LICENSE). Code adapted from other projects keeps its licence, listed in
[NOTICE](NOTICE) with the licence texts in [licenses/](licenses/).

jax-linear-attention is an independent project. It is not affiliated with or endorsed by Google or the JAX
team; "JAX" names the framework it is built on.

<sub>jax-linear-attention is made by <a href="https://www.karlex.ai"><img src="https://raw.githubusercontent.com/jla-org/jax-linear-attention/main/docs/assets/karlex-ai-logo.png" width="16" height="16" align="absmiddle" alt="KarLex AI logo"></a> <a href="https://www.karlex.ai">KarLex AI</a>.</sub>
