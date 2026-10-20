# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Remat-policy and micro-batch sweep of one 1.3B preset (benchmarks/fp8_1p3b.md).

    python benchmarks/bench_remat_sweep.py --arch kda-1p3b --json out.json \\
        --variant full:16x1 --variant none:4x4 --variant dots:8x2 \\
        --variant names=linear+kernels:8x2 --variant offload=kernels/linear+mlp_hidden:16x1

Each ``--variant`` is ``<remat>[=<names>[/<offload names>]]:<micro batch>x<grad accum>[:noscan]``.
The model is the preset of ``benchmarks/end_to_end/1p3b/<arch>.toml`` with every mixer on its Mosaic
GPU kernel (as ``bench_e2e_1p3b.py --impl kernels``), built once (twice with ``noscan``); every variant
compiles the real train step (``jla.training.step.build_train_step``, the config's AdamW, the default
linear cross entropy) and times ``--steps`` optimizer steps after ``--warmup`` on one synthetic packed
batch: random token ids, 4 documents per 4096-token row (segment ids and positions restart at each
boundary, as with ``data.document_masking``). The data loader is not part of the timing, so the
numbers rank the policies; ``bench_e2e_1p3b.py`` measures the chosen ones end to end. Each record
has the step time, tokens/s, the compiled step's memory and the first and last loss.

Every variant runs in a fresh subprocess of this script (``--in-process`` runs them all in one), so a
variant that ran out of memory leaves neither its arrays nor the allocator's grown pool to the next.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _parse_variant(text: str) -> dict:
    parts = text.split(":")
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "noscan"):
        raise ValueError(f"variant {text!r}: expected <remat>[=<names>[/<offload>]]:<mb>x<ga>[:noscan]")
    policy, batch = parts[0], parts[1]
    remat, _, names = policy.partition("=")
    kept, _, offload = names.partition("/")
    mb, _, ga = batch.partition("x")
    return {
        "label": text, "remat": remat,
        "remat_names": tuple(n for n in kept.split("+") if n),
        "remat_offload_names": tuple(n for n in offload.split("+") if n),
        "micro_batch": int(mb), "grad_accum": int(ga or 1), "scan_layers": len(parts) < 3,
    }


def _packed_batch(np, grad_accum: int, rows: int, seq_len: int, vocab: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(grad_accum, rows, seq_len + 1), dtype=np.int32)
    segment = np.zeros((grad_accum, rows, seq_len), np.int32)
    positions = np.zeros((grad_accum, rows, seq_len), np.int32)
    for a in range(grad_accum):
        for r in range(rows):
            cuts = np.sort(rng.choice(np.arange(64, seq_len - 64), size=3, replace=False))
            starts = np.concatenate([[0], cuts])
            ends = np.concatenate([cuts, [seq_len]])
            for doc, (s, e) in enumerate(zip(starts, ends)):
                segment[a, r, s:e] = doc
                positions[a, r, s:e] = np.arange(e - s)
    labels = ids[..., 1:].copy()
    last = np.concatenate([segment[..., 1:] != segment[..., :-1], np.ones_like(segment[..., :1], bool)], axis=-1)
    labels[last[..., :seq_len]] = -100  # a document's last token does not predict the next document
    return {"input_ids": ids[..., :-1], "labels": labels, "segment_ids": segment, "positions": positions}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--json", required=True)
    p.add_argument("--in-process", action="store_true",
                   help="run every variant in this process (default: each in a fresh subprocess)")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    variants = [_parse_variant(v) for v in args.variant]
    if not args.in_process:
        return _run_isolated(args)

    import equinox as eqx
    import jax
    import numpy as np

    from bench_e2e_1p3b import _config_path, _gpu_snapshot, _with_impl
    from jla.config import load_config
    from jla.models import build_model
    from jla.training.optimizers import optimizer_from_config
    from jla.training.performance import estimate_flops_per_token, estimate_hfu_flops_per_token
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss

    cfg = load_config(str(_config_path(args.arch)), [f"train.seq_len={args.seq_len}"])
    base_model = _with_impl(cfg.model, "mosaic_gpu", "mosaic_gpu")
    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=100, warmup_steps=cfg.optim.warmup_steps,
                               min_lr_ratio=cfg.optim.min_lr_ratio, stable_fraction=cfg.optim.stable_fraction,
                               decay_type=cfg.optim.decay_type, d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    flops = estimate_flops_per_token(base_model, args.seq_len)
    peak = 989e12  # dense bf16 peak of an H100 SXM, as bench_e2e_1p3b.py
    records = []
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    init_fns: dict = {}  # one jitted parameter initializer per layout (scan_layers on / off)
    n_params = []

    def fresh_model(model_cfg):
        """The preset's initial parameters with this variant's (static) config, as the Trainer builds them."""
        key = jax.random.key(0)
        abstract = eqx.filter_eval_shape(build_model, model_cfg, key=key)
        is_abstract = lambda x: isinstance(x, jax.ShapeDtypeStruct)  # noqa: E731
        if model_cfg.scan_layers not in init_fns:
            init_fns.clear()
            init_fns[model_cfg.scan_layers] = jax.jit(
                lambda k: jax.tree.leaves(eqx.filter(build_model(model_cfg, key=k), eqx.is_array)))
        # The arrays do not depend on the remat fields, only their tree's static metadata does.
        leaves = init_fns[model_cfg.scan_layers](key)
        arrays = jax.tree.unflatten(jax.tree.structure(eqx.filter(abstract, is_abstract)), leaves)
        return eqx.combine(arrays, eqx.filter(abstract, is_abstract, inverse=True))

    def save():
        out_path.write_text(json.dumps({
            "arch": args.arch, "seq_len": args.seq_len, "warmup": args.warmup, "steps": args.steps,
            "params": n_params[0] if n_params else None,
            "training_flops_per_token": flops.training, "jax_version": jax.__version__,
            "device": str(jax.devices()[0].device_kind), "gpus": _gpu_snapshot(), "records": records,
        }, indent=2))

    for v in variants:
        rec = dict(v)
        model = opt_state = compiled = owner = step = batch = None
        try:
            model_cfg = base_model.replace(remat=v["remat"], remat_names=v["remat_names"],
                                           remat_offload_names=v["remat_offload_names"], scan_layers=v["scan_layers"])
            model = fresh_model(model_cfg)
            if not n_params:
                n_params.append(sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))))
            opt_state = jax.jit(optimizer.init)(eqx.filter(model, eqx.is_inexact_array))
            # A rolled microbatch loop (scan_unroll = 1): unrolled, XLA keeps several microbatches'
            # activations alive at once, which defeats the point of a smaller micro-batch.
            train_cfg = dataclasses.replace(cfg.train, micro_batch_per_replica=v["micro_batch"], grad_accum=v["grad_accum"],
                                            scan_unroll=1)
            rec["scan_unroll"] = 1
            step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(train_cfg)), scan_unroll=1)
            batch = {k: jax.device_put(a) for k, a in _packed_batch(np, v["grad_accum"], v["micro_batch"], args.seq_len,
                                                                    model_cfg.vocab_size, 0).items()}
            key = jax.random.key(1)
            t0 = time.time()
            compiled = step.lower(model, opt_state, batch, key).compile()
            rec["compile_s"] = time.time() - t0
            owner = compiled  # equinox wraps the jax Compiled object
            while not hasattr(owner, "memory_analysis") and hasattr(owner, "compiled"):
                owner = owner.compiled
            mem = owner.memory_analysis()
            if mem is not None:
                rec["compiled_gib"] = {
                    "temp": mem.temp_size_in_bytes / 2**30, "argument": mem.argument_size_in_bytes / 2**30,
                    "output": mem.output_size_in_bytes / 2**30, "alias": mem.alias_size_in_bytes / 2**30,
                    "host_temp": getattr(mem, "host_temp_size_in_bytes", 0) / 2**30,
                }
                rec["compiled_total_gib"] = (mem.temp_size_in_bytes + mem.argument_size_in_bytes
                                             + mem.output_size_in_bytes - mem.alias_size_in_bytes) / 2**30
            limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 2**30
            rec["device_limit_gib"] = limit
            if limit and rec.get("compiled_total_gib", 0) > limit:
                raise MemoryError(f"out of memory: the compiled step needs {rec['compiled_total_gib']:.1f} GiB "
                                  f"of the {limit:.1f} GiB this process may use")
            step.use_compiled(compiled)
            times, losses = [], []
            for i in range(args.warmup + args.steps):
                t0 = time.time()
                model, opt_state, metrics, key = step(model, opt_state, batch, key)
                loss = float(metrics["loss"])
                times.append(time.time() - t0)
                losses.append(loss)
            timed = times[args.warmup:]
            tokens = v["micro_batch"] * v["grad_accum"] * args.seq_len
            mean = sum(timed) / len(timed)
            hfu_flops = estimate_hfu_flops_per_token(flops, v["remat"], v["remat_names"] + v["remat_offload_names"])
            rec.update(status="ok", step_s_mean=mean, step_s_min=min(timed), step_s_max=max(timed),
                       tokens_per_step=tokens, tokens_per_s=tokens / mean, mfu=tokens / mean * flops.training / peak,
                       hfu=tokens / mean * hfu_flops / peak, loss_first=losses[0], loss_last=losses[-1],
                       peak_gib_process=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        except Exception as exc:  # out of memory or a policy that does not lower: record it, go on
            msg = f"{type(exc).__name__}: {exc}"
            rec.update(status="oom" if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() else "failed",
                       error=msg[:1500])
            traceback.print_exc()
        finally:  # drop this variant's arrays whether it ran or failed
            model = opt_state = compiled = owner = step = batch = None
            gc.collect()
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("label", "status", "step_s_mean", "tokens_per_s", "mfu",
                                                  "compiled_total_gib", "compile_s")}), flush=True)
        save()
    return 0


def _run_isolated(args) -> int:
    """Each ``--variant`` in a fresh process of this script; the records are merged into ``--json``."""
    import subprocess
    import tempfile

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: dict = {}
    records = []
    for text in args.variant:
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "variant.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--arch", args.arch, "--variant", text,
                   "--seq-len", str(args.seq_len), "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--json", str(child), "--in-process"]
            proc = subprocess.run(cmd, check=False)
            data = json.loads(child.read_text()) if child.exists() else {}
        if data.get("records"):
            header = header or {k: val for k, val in data.items() if k != "records"}
            records.extend(data["records"])
        else:  # the process died before it could record the variant (a hard crash or abort)
            rec = _parse_variant(text)
            rec.update(status="failed", error=f"the variant's process exited with code {proc.returncode} "
                                                  "without a record")
            records.append(rec)
        out_path.write_text(json.dumps({**header, "arch": args.arch, "isolated": True, "records": records},
                                       indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
