# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DPO loss, rewards and gradients from TRL's own ``DPOTrainer._compute_loss`` on fixed inputs.

    TRL_VENV/bin/python benchmarks/dpo_trl_reference.py --out tests/data/dpo_trl_reference.json

Runs in an environment with torch and TRL (not a dependency of ``jla``). The inputs are
per-token policy log-probabilities, reference per-sequence log-probabilities and completion masks
drawn from ``numpy.random.default_rng(0)`` (stored in the output). TRL's method runs unmodified on
a stand-in trainer and model: the model returns the fixed per-token log-probabilities (zero on
prompt and padding, as TRL's fused LM head returns them), the reference log-probabilities enter as
precomputed (``precompute_ref_log_probs``), and TRL's chosen-then-rejected batch layout is used.
For each case (loss type, beta) the file records TRL's loss, its gradient with respect to the
per-token policy log-probabilities (torch autograd, float64) and TRL's reward metrics.
``tests/test_dpo.py`` checks ``jla.training.dpo.preference_terms`` against it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

PAIRS = 6
LENGTH = 12  # completion positions per row (the TRL mask has one more leading prompt position)
CASES = (("sigmoid", 0.1), ("sigmoid", 0.5), ("hinge", 0.2), ("ipo", 0.1), ("ipo", 0.4))


def inputs() -> dict[str, np.ndarray]:
    """Rows in jla order: pair i -> row 2i chosen, row 2i + 1 rejected."""
    rng = np.random.default_rng(0)
    rows = 2 * PAIRS
    prompt = rng.integers(1, 5, rows)  # leading prompt positions (masked)
    response = rng.integers(1, LENGTH - 4, rows)  # response tokens, then padding (masked)
    mask = np.zeros((rows, LENGTH), np.float64)
    for r in range(rows):
        mask[r, prompt[r]:prompt[r] + response[r]] = 1.0
    token_logps = -rng.gamma(2.0, 1.5, (rows, LENGTH)) * mask
    ref_logps = (token_logps * mask).sum(-1) + rng.normal(0.0, 3.0, rows)
    return dict(token_logps=token_logps, mask=mask, ref_logps=ref_logps)


def trl_case(data: dict[str, np.ndarray], loss_type: str, beta: float) -> dict:
    from trl import DPOTrainer

    order = np.concatenate([np.arange(0, 2 * PAIRS, 2), np.arange(1, 2 * PAIRS, 2)])  # TRL: chosen rows, then rejected
    lp = torch.tensor(data["token_logps"][order], dtype=torch.float64, requires_grad=True)
    mask = data["mask"][order]
    completion = torch.tensor(np.concatenate([np.zeros((2 * PAIRS, 1)), mask], axis=1), dtype=torch.int64)
    ref = torch.tensor(data["ref_logps"][order], dtype=torch.float64)

    class Model:
        training = True

        def __call__(self, **kwargs):
            zeros = torch.zeros_like(lp)
            return SimpleNamespace(log_probs=lp, entropy=zeros, mean_logits=zeros, is_top1=zeros.bool(), aux_loss=None)

    accelerator = SimpleNamespace(device=torch.device("cpu"), gather=lambda x: x, gather_for_metrics=lambda x: x)
    trainer = SimpleNamespace(
        model=Model(), accelerator=accelerator, aux_loss_enabled=False, ld_alpha=None, precompute_ref_logps=True,
        f_divergence_type="reverse_kl", f_alpha_divergence_coef=1.0, loss_types=[loss_type], loss_weights=[1.0],
        beta=beta, label_smoothing=0.0, use_weighting=False, _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens=0, _tp_size=1,
    )
    batch = dict(input_ids=torch.ones_like(completion), attention_mask=torch.ones_like(completion),
                 completion_mask=completion, ref_chosen_logps=ref[:PAIRS], ref_rejected_logps=ref[PAIRS:])
    loss = DPOTrainer._compute_loss(trainer, trainer.model, batch, return_outputs=False)
    loss.backward()
    grad = np.empty_like(data["token_logps"])
    grad[order] = lp.grad.numpy()
    metrics = {k: v[-1] for k, v in trainer._metrics["train"].items()
               if k.startswith("rewards/") or k.startswith("logps/")}
    return dict(loss_type=loss_type, beta=beta, loss=float(loss), grad_token_logps=grad.tolist(), metrics=metrics)


def main() -> None:
    import trl

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = inputs()
    out = dict(trl_version=trl.__version__, torch_version=torch.__version__, pairs=PAIRS,
               inputs={k: v.tolist() for k, v in data.items()},
               cases=[trl_case(data, lt, beta) for lt, beta in CASES])
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({c["loss_type"] + f"@{c['beta']}": c["loss"] for c in out["cases"]}))


if __name__ == "__main__":
    main()
