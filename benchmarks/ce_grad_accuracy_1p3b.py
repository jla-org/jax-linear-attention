# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head cross entropy variants against a float32 CE on identical weights and batches (gdn2-1p3b, bf16).

    python benchmarks/ce_grad_accuracy_1p3b.py --data /path/to/fineweb_edu_3B [--micro-batch 4] [--train-steps 10]

Trains --train-steps steps with the TOML's cross entropy (ce_impl "xla"), checking at step 0 and at the end,
on the next batch of the loader: the loss and the gradient of every parameter for
  R  ce_impl "xla", ce_compute_dtype float32 (the float32 CE reference; run twice: R2 gives the noise floor)
  A  ce_impl "xla" (bf16, the TOML as written)
  B  ce_impl "auto" + ce_grad_in_forward (the forward-gradient CE)
  C  ce_impl "auto" without the forward gradient (the CE kernels with the logits recompute)
Prints, per variant, the relative L2 error against R of the LM-head, embedding, final-norm and trunk
gradients and of the whole gradient, and the norm ratio |g| / |g_R| (a systematic scale shows there).
Matmuls run at "highest" precision, so R's float32 CE GEMMs are not TF32 (the bf16 GEMMs are unaffected).
Run for gdn2-1p3b on an H100 on 20 Oct 2026 (benchmarks/end_to_end_1p3b.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CIDX prefix (without .bin/.idx) of 4096-token packed records")
    p.add_argument("--arch", default="gdn2-1p3b")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=10)
    args = p.parse_args()
    import bench_e2e_1p3b as harness
    from jla.config import load_config
    from jla.launch_env import launch

    overrides = [f'data.paths=["{args.data}"]', "train.seq_len=4096",
                 f"train.micro_batch_per_replica={args.micro_batch}", "train.grad_accum=1", "mesh.dp=1", "mesh.fsdp=1"]
    cfg = load_config(str(harness._config_path(args.arch)), overrides)
    cfg = dataclasses.replace(cfg, model=harness._with_impl(cfg.model, *harness._IMPLS["kernels"]))
    launch(cfg.runtime)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_default_matmul_precision", "highest")  # R: float32 CE GEMMs, not TF32

    from jla.data.datasets import TokenWindowDataset
    from jla.data.loader import BatchLoader
    from jla.data.packing import DocumentFields
    from jla.models import build_model
    from jla.training.optimizers import optimizer_from_config
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss

    dataset = TokenWindowDataset(cfg.data.paths[0], 4096, exact_packed_records=cfg.data.exact_packed_records)
    loader = BatchLoader(dataset, args.micro_batch, seed=getattr(cfg.train, "seed", 0), shuffle=True, num_workers=0,
                         transform=DocumentFields(cfg.data.eos_id) if cfg.data.document_masking else None)

    def next_batch():
        return {k: jnp.asarray(np.asarray(v)) for k, v in next(loader).items()}

    base = LossConfig.from_train_config(cfg.train)
    variants = {
        "R": dataclasses.replace(base, impl="xla", grad_in_forward=False, compute_dtype="float32"),
        "A": dataclasses.replace(base, impl="xla", grad_in_forward=False),
        "B": dataclasses.replace(base, impl="auto", grad_in_forward=True),
        "C": dataclasses.replace(base, impl="auto", grad_in_forward=False),
    }
    grad_fns = {}
    for name, lc in variants.items():
        fn = lm_loss(lc)
        grad_fns[name] = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, b, k, fn=fn: fn(m, b, k)[0]))

    def leaves(grads):
        out = []
        for path, leaf in jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_inexact_array)):
            name = jax.tree_util.keystr(path)
            g = ("lm_head" if name.startswith(".head") else "embed" if name.startswith(".embed") else
                 "final_norm" if name.startswith(".final_norm") else "trunk")
            out.append((g, leaf))
        return out

    @jax.jit
    def leaf_stats(a, r):
        a, r = a.astype(jnp.float32), r.astype(jnp.float32)
        return jnp.stack([jnp.sum((a - r) ** 2), jnp.sum(r * r), jnp.sum(a * a), jnp.sum(a * r)])

    def stats(grads, ref):
        acc = {}
        for (g, a), (_, r) in zip(leaves(grads), leaves(ref), strict=True):
            acc[g] = acc.get(g, 0.0) + np.asarray(leaf_stats(a, r), np.float64)
        acc["all"] = sum(acc.values())
        return acc

    def check(model, tag):
        batch = next_batch()
        key = jax.random.PRNGKey(7)
        ref_loss, ref = grad_fns["R"](model, batch, key)
        print(f"[{tag}] R loss {float(ref_loss):.6f}", flush=True)
        for name in ("R", "A", "B", "C"):
            loss, g = grad_fns[name](model, batch, key)
            acc = stats(g, ref)
            del g
            parts = []
            for k in ("all", "lm_head", "embed", "final_norm", "trunk"):
                if k not in acc:
                    continue
                d2, r2, a2, ar = acc[k]
                parts.append(f"{k}: err {np.sqrt(d2 / r2):.3e} norm {np.sqrt(a2 / r2):.6f} "
                             f"cos {ar / np.sqrt(a2 * r2):.7f}")
            print(f"[{tag}] {name}{'2' if name == 'R' else ' '} loss {float(loss):.6f} "
                  f"(d {float(loss - ref_loss):+.2e}) | " + "; ".join(parts), flush=True)

    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps,
                               warmup_steps=cfg.optim.warmup_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                               stable_fraction=cfg.optim.stable_fraction, decay_type=cfg.optim.decay_type,
                               d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_jit(lambda k: build_model(cfg.model, key=k))(key)
    check(model, "step 0")
    opt_state = jax.jit(optimizer.init)(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(base), scan_unroll=cfg.train.scan_unroll)
    for i in range(args.train_steps):
        batch = {k: v[None] for k, v in next_batch().items()}
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        print(f"train step {i + 1} loss {float(metrics['loss']):.4f}", flush=True)
    check(model, f"step {args.train_steps}")
    loader.close()


if __name__ == "__main__":
    main()
