# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-data A/B of model initialisations (benchmarks/init_parity.md).

    python benchmarks/bench_init_ab.py --preset deltanet-125m --data /path/to/fineweb_edu_3B \
        --seed 0 --tag new --json results/deltanet-125m-new-s0.json

Trains one model preset through :class:`jla.training.trainer.Trainer` (what ``jla train`` runs) on a
CIDX corpus with a fixed recipe, so that two source trees that differ only in their initialisation
can be compared run for run: the same seed (parameter-init key and data order), batches, optimizer
and schedule. The recipe is the FineWeb-Edu end-to-end one (``benchmarks/end_to_end/fw-*.toml``:
AdamW, lr 6e-4, b2 0.95, weight decay 0.1, gradient clipping 1.0, z-loss 1e-4) with a linear warmup
over 10% of the steps and a cosine decay to 10% of the peak lr, on 4096-token pre-packed records
with document masking (the corpus of ``bench_e2e_1p3b.py``: Llama-3 tokenizer, 128,256 ids, EOS
128001). Every mixer keeps its preset's ``auto`` implementation.

The last ``--eval-holdout`` records of the corpus never reach training. Every ``--eval-every``
steps (and after the last step) the trainer evaluates the first ``--eval-batches`` batches of them,
unshuffled, so every run is scored on the same tokens. The train loss of each record is the
trainer's mean over its ``--log-every``-step window (cross entropy plus z-loss, as optimised).

The JSON holds the train curve, the evaluations, the recipe as resolved after the ``--set``
overrides, the tokens per optimizer step (sequences per replica x replicas x ``train.grad_accum`` x
sequence length), the parameter count and a SHA-256 of the tree's ``src/jla`` sources (which
identifies the tree that ran); no machine paths. ``status`` is ``ok`` when the run reached
``--steps`` with a finite final evaluation, ``halted`` otherwise; ``rejected_nonfinite_steps``
counts the updates the trainer skipped because the loss or gradients were not finite. Non-finite
numbers are written as ``null``, so the file is strict JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", required=True, help="model preset (jla.models.get_preset)")
    p.add_argument("--data", required=True, help="CIDX prefix (without .bin/.idx) of pre-packed records")
    p.add_argument("--tag", default="", help="label stored in the JSON, e.g. the tree (old / new)")
    p.add_argument("--seed", type=int, default=0, help="train.seed: parameter init and data order")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=None, help="linear warmup steps (default: 10%% of --steps)")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine floor as a fraction of --lr")
    p.add_argument("--micro-batch", type=int, default=8, help="sequences per step")
    p.add_argument("--seq-len", type=int, default=4096, help="the corpus's record length")
    p.add_argument("--vocab-size", type=int, default=128256)
    p.add_argument("--eos-id", type=int, default=128001)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--eval-holdout", type=int, default=2048, help="tail records held out for evaluation")
    p.add_argument("--output-dir", default=None, help="trainer output dir (default: a temporary directory)")
    p.add_argument("--json", default=None, help="write the result JSON here")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="extra config override a.b=value")
    args = p.parse_args(argv)
    if args.warmup is None:
        args.warmup = max(1, round(0.1 * args.steps))
    return args


def _source_sha256() -> str:
    """SHA-256 over the package sources (``src/jla/**/*.py``, path and content)."""
    digest = hashlib.sha256()
    root = HERE.parent / "src" / "jla"
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _recipe(args, output_dir: str) -> dict:
    return {
        "name": f"init-ab-{args.preset}",
        "model": {"preset": args.preset, "vocab_size": args.vocab_size},
        "optim": {
            "name": "adamw", "lr": args.lr, "schedule": "cosine", "warmup_steps": args.warmup,
            "min_lr_ratio": args.min_lr_ratio, "b2": 0.95, "weight_decay": 0.1, "max_grad_norm": 1.0,
        },
        "train": {
            "seed": args.seed, "max_steps": args.steps, "seq_len": args.seq_len,
            "micro_batch_per_replica": args.micro_batch, "grad_accum": 1, "z_loss": 1e-4,
            "log_every": args.log_every, "eval_every": args.eval_every, "eval_batches": args.eval_batches,
            "final_eval": True, "save_every": 0, "save_final": False, "sharding_audit": False,
            "output_dir": output_dir,
        },
        "data": {
            "source": "cidx", "paths": [args.data], "exact_packed_records": True, "eos_id": args.eos_id,
            "document_masking": True, "eval_holdout_records": args.eval_holdout, "num_workers": 4, "prefetch": 8,
        },
        "mesh": {"dp": 1, "fsdp": -1, "tp": 1},
        "runtime": {"xla_tuning_level": 1},
    }


def _strict(value):
    """``value`` with every non-finite float replaced by ``None`` (strict JSON has no NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    return value


def _resolve(recipe: dict, cfg) -> None:
    """Overwrite the recipe's scalar fields with the values of the resolved config (after ``--set``)."""
    for section in ("optim", "train", "data"):
        resolved = getattr(cfg, section)
        for key, value in recipe[section].items():
            new = getattr(resolved, key, value)
            if key != "paths" and isinstance(new, (bool, int, float, str)):
                recipe[section][key] = new


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    from jla.config import config_from_mapping
    from jla.launch_env import launch

    output_dir = args.output_dir
    if output_dir is None:
        import tempfile

        output_dir = tempfile.mkdtemp(prefix=f"init-ab-{args.preset}-")
    raw = _recipe(args, output_dir)
    cfg = config_from_mapping(raw, args.overrides)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    launch(cfg.runtime)

    import jax

    from jla.models import parameter_count
    from jla.training.trainer import Trainer

    recipe = json.loads(json.dumps(raw))
    _resolve(recipe, cfg)
    recipe["data"]["paths"] = [Path(args.data).name]
    recipe["train"].pop("output_dir")
    counts = parameter_count(cfg.model)
    # The mesh is dp 1, fsdp -1 (every device), tp 1: one data-parallel replica per device.
    replicas = jax.device_count()
    out = {
        "preset": args.preset,
        "tag": args.tag,
        "seed": args.seed,
        "steps": cfg.train.max_steps,
        "devices": replicas,
        "tokens_per_step": cfg.train.micro_batch_per_replica * replicas * cfg.train.grad_accum * cfg.train.seq_len,
        "recipe": recipe,
        "overrides": args.overrides,
        "parameters": {"total": counts.total, "embedding": counts.embedding, "non_embedding": counts.total - counts.embedding},
        "device_kind": str(jax.devices()[0].device_kind),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "source_sha256": _source_sha256(),
    }

    def write(record: dict) -> None:
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(_strict(record), indent=2, allow_nan=False))

    started = time.time()
    try:
        result = Trainer(cfg).run()
    except Exception as exc:  # record the failure, then fail
        write({**out, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
        raise
    keys = ("loss", "grad_norm", "lr", "tokens_per_sec", "skipped_steps")
    train = [{"step": int(h["step"]), **{k: float(h[k]) for k in keys if k in h}} for h in result.history]
    evals = [
        {"step": int(e["step"]), "loss": float(e["eval_loss"]), "perplexity": float(e["eval_perplexity"]),
         "tokens": int(e["eval_tokens"])}
        for e in result.eval_history
    ]
    finished = (result.stop_reason == "max_steps" and bool(evals) and evals[-1]["step"] == cfg.train.max_steps
                and math.isfinite(evals[-1]["loss"]))
    out.update({
        "status": "ok" if finished else "halted",
        "stop_reason": result.stop_reason,
        "rejected_nonfinite_steps": int(sum(r.get("skipped_steps", 0.0) for r in train)),
        "nonfinite_train_windows": sum(not math.isfinite(r["loss"]) for r in train),
        "wall_seconds_incl_compile": time.time() - started,
        "final_train_loss": train[-1]["loss"] if train else None,
        "final_eval_loss": evals[-1]["loss"] if evals else None,
        "train": train,
        "eval": evals,
    })
    write(out)
    print("INIT_AB_RESULT " + json.dumps(_strict({k: out[k] for k in ("preset", "tag", "seed", "status", "final_eval_loss",
                                                                       "final_train_loss", "wall_seconds_incl_compile")})))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
