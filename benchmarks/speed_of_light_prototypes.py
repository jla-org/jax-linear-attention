# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B prototypes of speed-of-light roadmap items on the real 1.3B step (benchmarks/profile_1p3b.md).

    python benchmarks/speed_of_light_prototypes.py --arch gdn-hd128-1p3b --data /path/to/fineweb_edu_3B \
        --variants base,l2norm_barrier --json out.json

Each variant runs in its own subprocess, one after the other on the same GPU. The step is the 1.3B
harness's (``benchmarks/bench_e2e_1p3b.py`` config, ``--impl kernels``: every mixer on its Mosaic GPU
kernel, one GPU, 16 x 4096 tokens, full remat, document masking from EOS), built as
``bench_end_to_end_step.py`` builds it, on the first batch of the real CIDX loader, timed as the median
of ``--steps`` steps after ``--warmup``. Variants change what XLA fuses, not the program (a fused
kernel may keep XLA's excess precision where the program rounds to bf16, so losses can differ in
the last bits):

* ``base``: the step as shipped.
* ``l2norm_barrier``: ``lax.optimization_barrier`` on the input of the q/k L2 norm in the Mosaic GPU
  delta-rule wrappers (``jla.ops.mosaic_gpu.scalar_gate`` and ``vector_gate_ops``), so that XLA cannot
  fuse the short convolution, its SiLU and the L2-norm reductions of q, k (and v's convolution) into one
  multi-output reduction fusion in the recomputed forward, which the profile shows running at ~160 GB/s.
* ``ce_mosaic_gpu``: ``train.ce_impl = "mosaic_gpu"``, the fused LM-head cross entropy kernels
  (``jla.ops.mosaic_gpu.linear_cross_entropy``) in place of the XLA op, whose backward recomputes the
  logits in a Triton GEMM fusion and makes float32 passes over every vocabulary chunk.

``--doc-stats`` (CPU only, no step is built) instead reads the first ``--batches`` batches of the same
loader and prints the share of the counted causal attention work that document masking leaves: exact
query-key pairs, and the ``--tile`` x ``--tile`` tiles a kernel that skips fully masked tiles visits,
both against the causal (window-capped) count of the MFU numerator, which ignores documents.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# Variants that are config overrides rather than patches.
_OVERRIDES = {"ce_mosaic_gpu": ["train.ce_impl=mosaic_gpu"]}


def _patch(name: str) -> None:
    if name == "base" or name in _OVERRIDES:
        return
    if name == "l2norm_barrier":
        from jax import lax

        import jla.ops.mosaic_gpu.scalar_gate as sg
        import jla.ops.mosaic_gpu.vector_gate_ops as vgo
        from jla.ops.reference import l2norm

        def barrier_l2norm(x, eps: float = 1e-6):
            return l2norm(lax.optimization_barrier(x), eps)

        sg.l2norm = barrier_l2norm
        vgo.l2norm = barrier_l2norm
        return
    raise SystemExit(f"unknown variant {name!r}")


def _first_batch(cfg, micro_batch: int, seq_len: int) -> dict:
    import numpy as np

    from jla.data.datasets import TokenWindowDataset
    from jla.data.loader import BatchLoader
    from jla.data.packing import DocumentFields

    dataset = TokenWindowDataset(cfg.data.paths[0], seq_len, exact_packed_records=cfg.data.exact_packed_records)
    transform = DocumentFields(cfg.data.eos_id) if cfg.data.document_masking else None
    loader = BatchLoader(dataset, micro_batch, seed=getattr(cfg.train, "seed", 0), shuffle=True, num_workers=0,
                         transform=transform)
    batch = next(loader)
    loader.close()
    return {k: np.asarray(v)[None] for k, v in batch.items()}


def attention_work(segment_ids, window: int | None, tile: int) -> dict:
    """Causal attention work of rows of ``segment_ids`` ([rows, T], contiguous non-decreasing ids):
    ``counted`` query-key pairs as the MFU numerator counts them (causal, window-capped, documents
    ignored), ``exact`` pairs within each document, and ``tiles``: the pairs of every ``tile`` x ``tile``
    tile on or below the diagonal that holds at least one visible pair (a fully masked tile is skipped,
    a partly masked one costs the whole tile)."""
    import numpy as np

    seg = np.asarray(segment_ids)
    rows, t = seg.shape
    if t % tile:
        raise ValueError(f"sequence length {t} is not a multiple of the tile {tile}")
    win = t + 1 if window is None else int(window)
    pos = np.arange(t)
    nb = t // tile
    qi, kj = np.tril_indices(nb)
    counted = exact = tiles = 0
    for s in seg:
        start = np.maximum.accumulate(np.where(np.r_[True, s[1:] != s[:-1]], pos, 0))
        counted += int(np.minimum(pos + 1, win).sum())
        exact += int(np.minimum(pos - start + 1, win).sum())
        first, last = s[::tile], s[tile - 1::tile]
        visible = (qi == kj) | ((first[qi] == last[kj]) & ((qi - kj - 1) * tile + 1 < win))
        tiles += int(visible.sum()) * tile * tile
    docs = sum(int((np.diff(s) != 0).sum()) + 1 for s in seg)
    return {"rows": rows, "seq_len": t, "window": window, "tile": tile, "documents": docs,
            "counted_pairs": counted, "exact_pairs": exact, "tile_pairs": tiles,
            "exact_fraction": round(exact / counted, 4), "tile_fraction": round(tiles / counted, 4)}


def doc_stats(args) -> dict:
    import bench_e2e_1p3b as harness
    import numpy as np

    from jla.config import load_config
    from jla.data.datasets import TokenWindowDataset
    from jla.data.loader import BatchLoader
    from jla.data.packing import DocumentFields

    cfg = load_config(str(harness._config_path(args.arch)), [
        f'data.paths=["{args.data}"]', f"train.seq_len={args.seq_len}",
        f"train.micro_batch_per_replica={args.micro_batch}", "train.grad_accum=1", "mesh.dp=1", "mesh.fsdp=1",
        *args.overrides])
    if not cfg.data.document_masking:
        raise SystemExit(f"{args.arch}: document masking is off, every attention tile below the diagonal runs")
    window = cfg.model.attention.sliding_window
    dataset = TokenWindowDataset(cfg.data.paths[0], args.seq_len, exact_packed_records=cfg.data.exact_packed_records)
    loader = BatchLoader(dataset, args.micro_batch, seed=getattr(cfg.train, "seed", 0), shuffle=True, num_workers=0,
                         transform=DocumentFields(cfg.data.eos_id))
    per_batch = []
    for _ in range(args.batches):
        per_batch.append(attention_work(np.asarray(next(loader)["segment_ids"]), window, args.tile))
    loader.close()
    total = {k: sum(b[k] for b in per_batch) for k in ("rows", "documents", "counted_pairs", "exact_pairs",
                                                       "tile_pairs")}
    out = {"arch": args.arch, "window": window, "tile": args.tile, "batches": args.batches, **total,
           "documents_per_row": round(total["documents"] / total["rows"], 2),
           "exact_fraction": round(total["exact_pairs"] / total["counted_pairs"], 4),
           "tile_fraction": round(total["tile_pairs"] / total["counted_pairs"], 4),
           "first_batch": per_batch[0]}
    print("DOCSTATS " + json.dumps(out), flush=True)
    return out


def run_variant(args) -> dict:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=").strip() \
        if args.profile_dir else os.environ.get("XLA_FLAGS", "")
    import bench_e2e_1p3b as harness

    from jla.config import load_config
    from jla.launch_env import launch

    cfg = load_config(str(harness._config_path(args.arch)), [
        f'data.paths=["{args.data}"]', f"train.seq_len={args.seq_len}",
        f"train.micro_batch_per_replica={args.micro_batch}", "train.grad_accum=1", "mesh.dp=1", "mesh.fsdp=1",
        *args.overrides, *_OVERRIDES.get(args.variant, [])])
    mixer_impl, attention_impl = harness._IMPLS["kernels"]
    cfg = dataclasses.replace(cfg, model=harness._with_impl(cfg.model, mixer_impl, attention_impl))
    launch(cfg.runtime)
    _patch(args.variant)

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jla.models import build_model
    from jla.training.optimizers import optimizer_from_config
    from jla.training.performance import estimate_flops_per_token
    from jla.training.schedulers import create_schedule
    from jla.training.step import LossConfig, build_train_step, lm_loss

    schedule = create_schedule(cfg.optim.lr, cfg.optim.schedule, total_steps=cfg.schedule_steps,
                               warmup_steps=cfg.optim.warmup_steps, min_lr_ratio=cfg.optim.min_lr_ratio,
                               stable_fraction=cfg.optim.stable_fraction, decay_type=cfg.optim.decay_type,
                               d_model=cfg.model.d_model)
    optimizer = optimizer_from_config(cfg.optim, schedule)
    key = jax.random.PRNGKey(0)
    model = eqx.filter_jit(lambda k: build_model(cfg.model, key=k))(key)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = jax.jit(optimizer.init)(params)
    step = build_train_step(optimizer, lm_loss(LossConfig.from_train_config(cfg.train)),
                            scan_unroll=cfg.train.scan_unroll)
    batch = {k: jnp.asarray(v) for k, v in _first_batch(cfg, args.micro_batch, args.seq_len).items()}
    t0 = time.perf_counter()
    compiled = step.lower(model, opt_state, batch, key).compile()
    step.use_compiled(compiled)
    compile_s = time.perf_counter() - t0
    losses = []
    for _ in range(args.warmup):
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        jax.block_until_ready(metrics)
        times.append(time.perf_counter() - t)
        losses.append(float(metrics["loss"]))
    wall = statistics.median(times) * 1e3
    tokens = args.micro_batch * args.seq_len
    flops = estimate_flops_per_token(cfg.model, args.seq_len)
    out = {"arch": args.arch, "variant": args.variant, "ce_impl": getattr(cfg.train, "ce_impl", "xla"),
           "compile_s": round(compile_s, 1),
           "ms_per_step_median": round(wall, 2), "ms_per_step_all": [round(t * 1e3, 2) for t in times],
           "tok_per_s": round(tokens / (wall / 1e3)),
           "mfu_percent": round(100 * flops.training * tokens / (wall / 1e3) / 989.4e12, 2), "losses": losses,
           "peak_bytes_in_use": int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0))}
    if args.profile_dir:
        xla_compiled = compiled if hasattr(compiled, "as_text") else compiled.compiled
        pdir = Path(args.profile_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "train_step.hlo.txt").write_text(xla_compiled.as_text())
        with jax.profiler.trace(str(pdir), create_perfetto_trace=True):
            for _ in range(2):
                model, opt_state, metrics, key = step(model, opt_state, batch, key)
            jax.block_until_ready(metrics)
        result = {"arch": args.arch, "micro_batch": args.micro_batch, "seq_len": args.seq_len,
                  "flops_per_token": flops.as_dict()}
        (pdir / "profile_meta.json").write_text(json.dumps({"profile_steps": 2, "wall_ms_median": wall,
                                                            "result": result}))
    print("PROTOTYPE " + json.dumps(out), flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--variants", default="base,l2norm_barrier")
    p.add_argument("--variant", default=None, help=argparse.SUPPRESS)  # one variant, in a subprocess
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--profile-root", default=None, help="record a trace of each variant under <root>/<variant>")
    p.add_argument("--profile-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--doc-stats", action="store_true", help="print the attention work document masking leaves")
    p.add_argument("--batches", type=int, default=1, help="--doc-stats: batches to read")
    p.add_argument("--tile", type=int, default=128, help="--doc-stats: square attention tile")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    if args.doc_stats:
        out = doc_stats(args)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(out, indent=1))
        return 0
    if args.variant:
        run_variant(args)
        return 0
    results = []
    for variant in args.variants.split(","):
        cmd = [sys.executable, __file__, "--arch", args.arch, "--data", args.data, "--variant", variant,
               "--micro-batch", str(args.micro_batch), "--seq-len", str(args.seq_len), "--steps", str(args.steps),
               "--warmup", str(args.warmup)]
        for o in args.overrides:
            cmd += ["--set", o]
        if args.profile_root:
            cmd += ["--profile-dir", str(Path(args.profile_root) / f"{args.arch}-{variant}")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROTOTYPE ")), None)
        if line is None:
            results.append({"arch": args.arch, "variant": variant, "error": (proc.stderr or proc.stdout)[-1500:]})
        else:
            results.append(json.loads(line[len("PROTOTYPE "):]))
        print(json.dumps(results[-1])[:600], flush=True)
    base = next((r for r in results if r.get("variant") == "base" and "ms_per_step_median" in r), None)
    if base:
        for r in results:
            if "ms_per_step_median" in r:
                r["speedup_vs_base"] = round(base["ms_per_step_median"] / r["ms_per_step_median"], 4)
                r["saved_ms_per_step"] = round(base["ms_per_step_median"] - r["ms_per_step_median"], 2)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"results": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
