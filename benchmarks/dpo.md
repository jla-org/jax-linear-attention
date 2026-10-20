# Direct preference optimization (`jla dpo`)

`jla dpo --config configs/post_train/dpo-tiny.toml` trains a policy with DPO (Rafailov et al. 2023) on the
compiled train step of `jla train`, under the trainer's mesh and kernel-sharding rules, so on several GPUs the
Mosaic GPU kernels run per shard and MoE layers dispatch per device. Its loss, gradient and reward metrics are
compared with TRL's `DPOTrainer`. The experiment file is a `jla train` file plus a `[dpo]` table
(`jla.training.dpo.DPOConfig`):

| key | default | meaning |
|---|---|---|
| `beta` | 0.1 | strength of the KL constraint to the reference |
| `loss_type` | `"sigmoid"` | `"sigmoid"` (DPO), `"ipo"` (Azar et al. 2023, on length-normalized log-ratios) or `"hinge"` (SLiC) |
| `label_smoothing` | 0.0 | conservative DPO (cDPO) for the sigmoid loss: `-(1 - eps) log sigmoid(beta h) - eps log sigmoid(-beta h)` |
| `train_path`, `eval_path` | none | preference JSONL; without `train_path` the run uses synthetic toy pairs |
| `tokenizer` | none | Hugging Face tokenizer for text records (needs `transformers`) |
| `max_prompt_len` | 0 | 0 = `train.seq_len // 2`; longer prompts keep their last tokens, responses are cut on the right |
| `pad_id`, `eos_id` | 0, none | padding id; `eos_id` is appended to every response when set |
| `init_checkpoint` | none | a `jla` checkpoint (run directory, step directory or `.safetensors`) for the policy and the reference |

Records are `{"prompt", "chosen", "rejected"}` as text or token ids; `train.micro_batch_per_replica` counts
pairs. Each log line reports the loss and the rewards (`beta * (log pi - log ref)` of the summed
log-probabilities, as TRL reports them): chosen, rejected, margin and accuracy.

## Setup

- JAX 0.11.2; the parity check is a float32 test against stored TRL outputs, the toy run uses an H100 80GB HBM3.
- Baseline: TRL 1.16.0's own `DPOTrainer._compute_loss` with torch 2.13 in float64, run unmodified by
  `benchmarks/dpo_trl_reference.py` on fixed inputs: six pairs of per-token policy log-probabilities with prompt
  and padding masks, and precomputed reference log-probabilities.
- Toy run: the `tiny` preset (0.11M trainable parameters), synthetic pairs whose chosen responses follow a fixed
  rule, 8 pairs of 64 tokens per step, 100 steps, one H100, 18 Oct 2026 (other GPUs ran timed jobs, so the run
  time is provisional).

## Results

Parity with TRL (`tests/test_dpo.py::test_loss_rewards_and_gradients_match_trl`, sigmoid at beta 0.1 and 0.5,
hinge at 0.2, IPO at 0.1 and 0.4): the loss to a relative error below 2e-6 (float32 against TRL's float64), the
gradient with respect to every response token's log-probability below 1e-5, and `rewards/chosen`,
`rewards/rejected`, `rewards/margins`, `rewards/accuracies`, `logps/chosen` and `logps/rejected` within 1e-5.

The toy run (a check that the objective trains, not a quality result):

| step | 10 | 20 | 40 | 60 | 100 |
|---|---|---|---|---|---|
| train loss | 0.542 | 0.202 | 0.047 | 0.023 | 0.0145 |
| reward accuracy | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |
| reward margin | 0.36 | 1.56 | 3.08 | 3.76 | 4.24 |

The loss starts at `log 2` (policy = reference), and the held-out evaluation at step 100 gives a loss of 0.0137
and reward accuracy 1.0. On the final code the run reproduces these values to three significant digits and takes
9.5 s for the 100 steps.

## Limits

- The reference forward runs in every step (no precomputed reference log-probabilities), pairs are padded rows,
  and KTO, ORPO, SimPO and the robust and f-divergence losses are not implemented. TRL 1.16's sigmoid loss has no
  label smoothing, so `label_smoothing` (cDPO) is checked against a NumPy oracle instead.
- Refused at load time: models with MTP heads, model dropout, `train.z_loss`, token label smoothing, the profiler
  window, FP8 delayed scaling (the step does not lower with the frozen reference's delayed-scaling state) and
  `train.ce_grad_in_forward`. FP8 current scaling trains.

## Correctness

`tests/test_dpo.py` also checks the per-token log-prob op against `log_softmax`, the losses against a NumPy
oracle, the DPO gradient against autodiff through explicit logits, a frozen reference, resume equal to the
uninterrupted run (bit for bit on CPU, and on GPU with `runtime.deterministic = true`) and 8 CPU devices
splitting pairs like one device.

## Reproduce

```bash
jla dpo --config configs/post_train/dpo-tiny.toml
python benchmarks/dpo_trl_reference.py --out results/dpo_trl_reference.json   # in a venv with torch and TRL 1.16.0
pytest tests/test_dpo.py
```

## How it works

A frozen copy of the starting point enters the compiled step as a traced, sharded input with no gradient or
optimizer state (`PolicyAndReference`), and its log-probabilities are computed in the same step with
`jla.ops.token_logprobs.linear_token_logprobs`, which never materializes `[N, V]` logits. With
`h = (log pi(c) - log pi(r)) - (log ref(c) - log ref(r))` the loss is the mean over pairs (plus the MoE auxiliary loss).
