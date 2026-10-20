# Expert parallelism for MoE (`mesh.ep`)

`jla.distributed.expert_parallel` shards the routed experts of `jla.layers.moe.MoE` over an `ep` mesh axis and
moves the routed tokens to their experts with all-to-all. It is checked against the same layer without expert
parallelism and timed against data parallelism and FSDP with per-device dispatch; the grouped GEMM itself is
compared with `torch._grouped_mm` in [ragged_dot.md](ragged_dot.md).

```toml
[model.moe]
expert_parallel_axis = "ep"           # the layer's all-to-all dispatch
expert_parallel_buffer_factor = 2.0   # optional: bounded per-peer blocks
[mesh]
fsdp = 2
ep = 4                                # experts -> ep, batch -> (dp, fsdp, ep)
```

`configs/moe-125m-ep.toml` is the 8-GPU example. `mesh.ep` adds a mesh axis that, like `dp`, splits the batch
(`micro_batch_per_replica * dp * fsdp * ep` rows); without `expert_parallel_axis`, XLA partitions the layer.

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16, `benchmarks/bench_expert_parallel.py` on 2 GPUs: the routed experts of
  `qwen3-next-300m` (64 experts, top-10, `d_ff` 256, `d_model` 1024), dropless routing on the Mosaic GPU
  grouped GEMMs (`ragged_dot_impl = "auto"`, the default on H100).
- Timing: the MoE layer forward + backward on 16,384 tokens (wall-clock median of 20 calls), and the Trainer
  step (AdamW, 2 x 2048 tokens per GPU, 8 steps, the median after step 2).
- Provisional: measured on 18 Oct 2026 with no other timed run on the two GPUs, but other jobs' test processes
  on them.

## Provisional results

| layout | MoE layer forward + backward, ms | Trainer step, ms |
|---|---|---|
| dp (experts replicated, per-device dispatch) | **2.677** | 86.49 |
| FSDP (`fsdp = 2`, per-device dispatch) | | **81.94** |
| gspmd (`ep = 2` mesh, no EP dispatch) | 2.816 | |
| ep (worst-case blocks) | 6.483 | |
| ep, buffer factor 2 | 7.957 | 109.48 |
| ep, buffer factor 1.25 | 5.083 | |

The FSDP step over 2 GPUs runs at 87% of twice the one-GPU throughput (71.1 ms for 4096 tokens on one GPU), and
the losses at step 8 agree (10.9226 with FSDP, 10.9235 with DP and with EP).

## Where we lose, and limits

- While every expert fits on every GPU, EP is slower than data parallelism with per-device dispatch: its
  all-to-all moves 10 routes of 1024 values per token each way, so even with a buffer factor of 1.25 the EP layer
  takes 1.9x the time of `dp`. For the 1.3B hybrids FSDP stays the layout to train with: at FSDP 8 `qwen3-next-1p3b`
  trains at 71.9k tokens/s per GPU and `kimi-linear-1p3b` at 61.6k (68.7k with one 16 x 4096 micro-step and the
  forward-gradient cross entropy), against 19.9k for `qwen3-next-1p3b` with EP 8 and bounded blocks (the Qwen3-Next
  and Kimi Linear section of [end_to_end_1p3b.md](end_to_end_1p3b.md)). EP is for experts that do not fit on one GPU; its
  all-to-all does not overlap the expert GEMMs yet, and the layer and step timings above ran on 2 GPUs only.
- The XLA `ragged_dot` path ran out of memory at this shape (a 20 GB allocation request during autotuning).
- Tokens per step must divide over `(dp, fsdp, ep)`, and `num_experts` over `ep`. Worst-case blocks are exact but
  carry up to `ep` times the routed rows; a smaller `expert_parallel_buffer_factor` (the trade of MaxText's
  `ragged_buffer_factor`) drops routes, second choices first, counted in `moe/dropped_fraction` with capacity drops.
- FSDP or TP shards of the expert weights are all-gathered per layer in the compute dtype. Per-device dispatch
  without EP replicates the expert weights, so under `tp > 1` every device holds every expert in full (not
  measured; `ragged_dot_impl = "xla"` keeps `d_ff` split over `tp`).
- `jax.lax.ragged_all_to_all` is not used: JAX 0.11.2 has no XLA:CPU lowering for it, and on H100 it fails with a
  CUDA illegal-address error, in a one-op `shard_map` on one GPU and in the EP layer on 2 and 8 GPUs.
- EP needs the trainer's context mesh (without one, as in decoding, the layer runs its own dispatch), and it is
  not combined with context or pipeline parallelism.

## Correctness

`tests/test_expert_parallel.py` (8 CPU devices) checks that the EP layer's outputs, losses, dropped fraction and
gradients equal the layer without EP on `fsdp2 x ep4`, `dp2 x fsdp2 x ep2`, `dp4 x ep2`, `ep8` and
`fsdp2 x tp2 x ep2`, for dropless and capacity routing, bfloat16 and bounded blocks, and that `Trainer` runs with
`mesh.ep` match one device step for step (losses and parameters to 1e-7). `tests/test_moe_two_gpus.py` runs the
Mosaic GPU kernels inside the `shard_map`s on two H100s with Qwen3-Next's and Kimi-Linear's routing: outputs
bit-identical in all six cases, the router statistics equal, and gradients within 0.41% (per-device dispatch)
and 0.62-0.63% (EP) relative norm, since the bfloat16 partial gradients are summed in another order.

## Reproduce

```bash
python benchmarks/bench_expert_parallel.py --gpus 2 --sections timing --timing-preset qwen3-next-300m \
  --timing-impls mosaic_gpu --tokens-per-gpu 8192 --iters 20 --out results/expert_parallel_layer.json
python benchmarks/bench_expert_parallel.py --gpus 2 --sections trainer_125m --trainer-preset qwen3-next-300m \
  --only-125m fsdp_baseline,dp_baseline,ep_mosaic_f2 --seq-len 2048 --micro 2 --steps 8 \
  --out results/expert_parallel_trainer.json
pytest tests/test_expert_parallel.py
pytest tests/test_moe_two_gpus.py -s   # two GPUs; JLA_TEST_GPUS="a,b" picks them
```

## How it works

The router, the balance and z-losses and the capacity decisions run on the global token set. Inside
`jax.shard_map` every device packs its routes, sorted by expert, into one block of `S` rows per `ep` peer,
exchanges the blocks with `jax.lax.all_to_all`, runs its `E / ep` local experts as grouped GEMMs, sends the
results back with the inverse all-to-all and sums each token's routes in float32. `S` defaults to the worst case
`local_tokens * min(top_k, E / ep)`, so no route is dropped; `expert_parallel_buffer_factor = f` bounds it to
`ceil(f * local_tokens * top_k / ep)` rows in the capacity rule's slot-major order. Without
`expert_parallel_axis`, dropless routing on the Mosaic GPU grouped GEMM runs per device (`MoE._dropless_per_device`).
