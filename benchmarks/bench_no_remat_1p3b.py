# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No recomputation + gradient accumulation at 1.3B with the memory-lean VJPs (benchmarks/fp8_1p3b.md).

    python benchmarks/bench_no_remat_1p3b.py --arch kda-1p3b --json out.json \\
        --variant full:16x1 --variant none+lean:4x4 --variant none+lean:2x8 --variant dots+lean:8x2

Each ``--variant`` is ``<remat>[+lean[+core]]:<micro batch>x<grad accum>``: ``model.remat``, ``model.lean_vjp``
(``jla.layers.lean``), ``model.lean_remat_core`` and the split of the 65,536 tokens of one optimizer step (``train.micro_batch_per_replica``
x ``train.grad_accum`` rows of 4096 tokens, a rolled micro-step loop, ``scan_unroll = 1``). The model is the
preset of ``benchmarks/end_to_end/1p3b/<arch>.toml`` with every mixer on its Mosaic GPU kernel (as
``bench_e2e_1p3b.py --impl kernels``). Every variant runs in a fresh subprocess, compiles the real train
step (``jla.training.step.build_train_step``, the config's AdamW, the default linear cross entropy) and times
``--steps`` optimizer steps after ``--warmup`` on one synthetic packed batch of the same 16 rows for every
split (random ids, 4 documents per row, positions and segment ids restarting at each document, no target
across a document end), so the first and last losses of all variants are comparable. A variant whose
compiled step needs more memory than the process may use is recorded without running.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _parse_variant(text: str) -> dict:
    policy, sep, split = text.partition(":")
    mb, x, ga = split.partition("x")
    remat, *flags = policy.split("+")
    if not sep or not x or flags not in ([], ["lean"], ["lean", "core"]):
        raise ValueError(f"variant {text!r}: expected <remat>[+lean[+core]]:<micro batch>x<grad accum>")
    return {"label": text, "remat": remat, "lean_vjp": "lean" in flags, "lean_remat_core": "core" in flags,
            "micro_batch": int(mb), "grad_accum": int(ga)}


def _packed_batch(np, grad_accum: int, rows: int, seq_len: int, vocab: int, seed: int) -> dict:
    """``grad_accum * rows`` packed rows drawn in one order, so every split holds the same tokens."""
    rng = np.random.default_rng(seed)
    total = grad_accum * rows
    ids = rng.integers(0, vocab, size=(total, seq_len + 1), dtype=np.int32)
    segment = np.zeros((total, seq_len), np.int32)
    positions = np.zeros((total, seq_len), np.int32)
    for r in range(total):
        cuts = np.sort(rng.choice(np.arange(64, seq_len - 64), size=3, replace=False))
        for doc, (s, e) in enumerate(zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [seq_len]]))):
            segment[r, s:e] = doc
            positions[r, s:e] = np.arange(e - s)
    labels = ids[:, 1:].copy()
    last = np.concatenate([segment[:, 1:] != segment[:, :-1], np.ones_like(segment[:, :1], bool)], axis=-1)
    labels[last] = -100  # a document's last token does not predict the next document
    shape = (grad_accum, rows, seq_len)
    return {"input_ids": ids[:, :-1].reshape(shape), "labels": labels.reshape(shape),
            "segment_ids": segment.reshape(shape), "positions": positions.reshape(shape)}


def _run_variant(args, v: dict) -> dict:
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
    model_cfg = _with_impl(cfg.model, "mosaic_gpu", "mosaic_gpu").replace(
        remat=v["remat"], lean_vjp=v["lean_vjp"], lean_remat_core=v["lean_remat_core"])
    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=100, warmup_steps=cfg.optim.warmup_steps,
                               min_lr_ratio=cfg.optim.min_lr_ratio, stable_fraction=cfg.optim.stable_fraction,
                               decay_type=cfg.optim.decay_type, d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    flops = estimate_flops_per_token(model_cfg, args.seq_len)
    rec = dict(v, arch=args.arch, seq_len=args.seq_len, jax_version=jax.__version__,
               device=str(jax.devices()[0].device_kind), gpus=_gpu_snapshot(), scan_unroll=1,
               training_flops_per_token=flops.training)
    train_cfg = dataclasses.replace(cfg.train, micro_batch_per_replica=v["micro_batch"], grad_accum=v["grad_accum"],
                                    scan_unroll=1)
    step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(train_cfg)), scan_unroll=1)
    host_batch = _packed_batch(np, v["grad_accum"], v["micro_batch"], args.seq_len, model_cfg.vocab_size, 0)
    key = jax.random.key(1)
    if args.compile_only:  # abstract model, optimizer state and batch: no device buffers, no timing
        model = eqx.filter_eval_shape(build_model, model_cfg, key=jax.random.key(0))
        opt_state = eqx.filter_eval_shape(  # traced, so eqx.is_inexact_array sees the parameters
            lambda k: optimizer.init(eqx.filter(build_model(model_cfg, key=k), eqx.is_inexact_array)),
            jax.random.key(0))
        batch = {k: jax.ShapeDtypeStruct(a.shape, a.dtype) for k, a in host_batch.items()}
    else:
        model = eqx.filter_jit(lambda k: build_model(model_cfg, key=k))(jax.random.key(0))  # same arrays every variant
        opt_state = jax.jit(optimizer.init)(eqx.filter(model, eqx.is_inexact_array))
        batch = {k: jax.device_put(a) for k, a in host_batch.items()}
    t0 = time.time()
    if args.compile_only:  # eqx.filter_jit would treat ShapeDtypeStructs as static: lower the step function itself
        rec["params"] = sum(int(x.size) for x in jax.tree.leaves(model) if isinstance(x, jax.ShapeDtypeStruct)
                            and jax.numpy.issubdtype(x.dtype, jax.numpy.inexact))
        is_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)  # noqa: E731
        dynamic, static = eqx.partition((model, opt_state), is_struct)
        train_step = step._jitted.__wrapped__

        def run(batch, dynamic, key):
            m, o = eqx.combine(dynamic, static)
            return train_step(batch, m, o, key)

        compiled = jax.jit(run, donate_argnums=(1,)).lower(batch, dynamic, key).compile()
    else:
        rec["params"] = sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))
        compiled = step.lower(model, opt_state, batch, key).compile()
    rec["compile_s"] = time.time() - t0
    owner = compiled  # equinox wraps the jax Compiled object
    while not hasattr(owner, "memory_analysis") and hasattr(owner, "compiled"):
        owner = owner.compiled
    mem = owner.memory_analysis()
    if mem is not None:
        rec["compiled_gib"] = {"temp": mem.temp_size_in_bytes / 2**30, "argument": mem.argument_size_in_bytes / 2**30,
                               "output": mem.output_size_in_bytes / 2**30, "alias": mem.alias_size_in_bytes / 2**30}
        rec["compiled_total_gib"] = (mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes
                                     - mem.alias_size_in_bytes) / 2**30
    limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) / 2**30
    rec["device_limit_gib"] = limit
    if args.compile_only:
        rec.update(status="compiled")
        return rec
    if limit and rec.get("compiled_total_gib", 0) > limit:
        rec.update(status="does_not_fit")
        return rec
    step.use_compiled(compiled)
    times, losses = [], []
    for _ in range(args.warmup + args.steps):
        t0 = time.time()
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
        times.append(time.time() - t0)
    timed = times[args.warmup:]
    tokens = v["micro_batch"] * v["grad_accum"] * args.seq_len
    mean = sum(timed) / len(timed)
    peak = 989e12  # dense bf16 peak of an H100 SXM, as bench_e2e_1p3b.py
    rec.update(status="ok", step_s_mean=mean, step_s_min=min(timed), step_s_max=max(timed), step_times=times,
               tokens_per_step=tokens, tokens_per_s=tokens / mean, mfu=tokens / mean * flops.training / peak,
               hfu=tokens / mean * estimate_hfu_flops_per_token(flops, v["remat"]) / peak, losses=losses,
               peak_gib_process=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--json", required=True)
    p.add_argument("--compile-only", action="store_true",
                   help="compile every variant from abstract arrays and record its memory; run nothing")
    p.add_argument("--preallocate", action="store_true",
                   help="variant processes preallocate their memory fraction as one pool (XLA_PYTHON_CLIENT_PREALLOCATE),"
                        " so a step whose temporaries fit is not refused for the growth regions the parameters left;"
                        " only for a process that holds its GPU alone")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.child:  # one variant in this process
        v = _parse_variant(args.variant[0])
        try:
            rec = _run_variant(args, v)
        except Exception as exc:  # out of memory or a variant that does not lower: record it
            msg = f"{type(exc).__name__}: {exc}"
            rec = dict(v, status="oom" if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() else "failed",
                       error=msg[:1500])
            traceback.print_exc()
        Path(args.json).write_text(json.dumps(rec, indent=1))
        return 0
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for text in args.variant:
        _parse_variant(text)
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "variant.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--arch", args.arch, "--variant", text,
                   "--seq-len", str(args.seq_len), "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--json", str(child)] + (["--compile-only"] if args.compile_only else [])
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="true") if args.preallocate else None
            proc = subprocess.run(cmd, check=False, env=env)
            rec = json.loads(child.read_text()) if child.exists() else dict(
                _parse_variant(text), status="failed", error=f"process exited with code {proc.returncode}, no record")
            rec["preallocate"] = bool(args.preallocate)
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("label", "status", "step_s_mean", "tokens_per_s", "mfu",
                                                  "compiled_total_gib", "compile_s")}), flush=True)
        out_path.write_text(json.dumps({"arch": args.arch, "records": records}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
