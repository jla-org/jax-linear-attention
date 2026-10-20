# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 MIMO op benchmark: the dedicated Mosaic GPU kernels vs the SISO-kernel route vs the
chunked JAX path vs the official TileLang MIMO kernels (state-spaces/mamba ``mamba3_mimo``).

    python benchmarks/bench_mamba3_mimo.py jax --out benchmarks/results/mamba3_mimo.json \\
        [--torch-python /path/to/venv/bin/python --mamba-root /path/to/state-spaces/mamba]

Labels (all on the same inputs; ``R`` ranks per token):

* ``mimo``: ``ops.linear_attention("mamba3_mimo", ..., impl="mosaic_gpu")`` (dedicated kernels,
  chunks of ``64 / R`` tokens);
* ``siso``: the same with ``kernel="siso"`` (``H R^2`` SISO heads, the wave-2 route);
* ``chunked``: ``impl="chunked"`` (XLA, ``--chunk`` tokens per chunk);
* ``official``: ``mamba_ssm.ops.tilelang.mamba3.mamba3_mimo.mamba3_mimo`` with
  ``chunk_size = 64 / R``, run in a torch subprocess on the same GPU between two rounds of ours. Its
  call also does the angle cumsum and rotation (zero angles here), the Q/K biases (zero) and the
  value up-projection ``v_r = Psi_r v``, and reads one ``[B, T, H, V]`` value tensor where our op
  reads the projected ``[B, T, H, R, V]`` values; there is no ``D``, ``Z`` or output projection.
  ``fwd+bwd`` differentiates ``Q, K, V, ADT, DT, Trap, Psi``.

The metric is the CUPTI kernel time of one call: the median of 30 calls after 3 warmups for JAX
(``benchmarks/harness.py``), and in torch the mean device time of 10 profiled calls after a wait
and a warmup step. ``fwd+bwd`` is one forward plus the backward with gradients for every input.
Inputs: ``dt`` log-uniform in [1e-3, 1e-1], ``A`` uniform in [-16, -1], trapezoid logits N(0, 1),
``q, k ~ N(0, 1) / sqrt(R K)``, ``v ~ N(0, 1)``, ``Psi ~ 1 + N(0, 0.1)``; bf16-representable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

SHAPES = [(8, 2048, 32, 4, 128, 64), (4, 2048, 24, 4, 128, 64), (1, 8192, 32, 4, 128, 64),
          (4, 2048, 32, 2, 128, 64)]


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), kept as float32."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def make_inputs(batch: int, seq_len: int, heads: int, rank: int, key_dim: int, value_dim: int, seed: int = 0):
    """Raw inputs (float32 numpy) in the official layout: ``q, k [B, T, R, H, K]``, ``v [B, T, H, V]``,
    ``adt, dt, trap [B, T, H]``, ``psi [H, R, V]``; ``do [B, T, H, R, V]`` is the output cotangent."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(rank * key_dim)
    q = _round_bf16(rng.standard_normal((batch, seq_len, rank, heads, key_dim), np.float32) * scale)
    k = _round_bf16(rng.standard_normal((batch, seq_len, rank, heads, key_dim), np.float32) * scale)
    v = _round_bf16(rng.standard_normal((batch, seq_len, heads, value_dim), np.float32))
    dt = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), (batch, seq_len, heads))).astype(np.float32)
    a = -rng.uniform(1.0, 16.0, (batch, seq_len, heads)).astype(np.float32)
    trap = _round_bf16(rng.standard_normal((batch, seq_len, heads), np.float32))
    psi = _round_bf16(1.0 + 0.1 * rng.standard_normal((heads, rank, value_dim), np.float32))
    do = _round_bf16(rng.standard_normal((batch, seq_len, heads, rank, value_dim), np.float32))
    return dict(q=q, k=k, v=v, adt=a * dt, dt=dt, trap=trap, psi=psi, do=do)


def gpu_status() -> list[dict]:
    """``nvidia-smi`` rows of the visible GPU only."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        idx, util, used = (int(x) for x in line.split(", "))
        if str(idx) in visible.split(","):
            rows.append(dict(gpu=idx, util_pct=util, mem_used_mib=used))
    return rows


# --------------------------------------------------------------------------------------- JAX


def run_jax(args) -> dict:
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness
    from jla import ops

    def op_inputs(raw):
        lam = 1.0 / (1.0 + np.exp(-raw["trap"]))
        g = raw["adt"]
        q = np.transpose(raw["q"], (0, 1, 3, 2, 4))  # [B, T, H, R, K]
        k = np.transpose(raw["k"], (0, 1, 3, 2, 4))
        v = raw["v"][:, :, :, None, :] * raw["psi"][None, None]  # v_r = Psi_r v
        x = dict(q=q, k=k, v=v, g=g, beta=(1.0 - lam) * raw["dt"] * np.exp(g), gamma=lam * raw["dt"])
        out = {n: jnp.asarray(a, jnp.bfloat16 if n in ("q", "k", "v") else jnp.float32) for n, a in x.items()}
        return out, jnp.asarray(raw["do"], jnp.bfloat16)

    names = ("q", "k", "v", "g", "beta", "gamma")
    impls = {
        "mimo": dict(impl="mosaic_gpu", kernel="mimo"),
        "siso": dict(impl="mosaic_gpu", kernel="siso"),
        "chunked": dict(impl="chunked", chunk_size=args.chunk),
    }
    results = []
    for shape in args.shapes:
        raw = make_inputs(*shape)
        x, do = op_inputs(raw)
        arrays = [x[n] for n in names]
        row = dict(shape=dict(zip(("B", "T", "H", "R", "K", "V"), shape)), gpu_before=gpu_status())
        batch, seq_len, heads, rank = shape[:4]
        fns = {}
        for label, opts in impls.items():
            if label != "mimo" and batch * seq_len * heads * rank * rank > args.baseline_limit:
                row[label] = "skipped: larger than --baseline-limit (its fwd+bwd does not fit the memory cap)"
                continue

            def call(q, k, v, g, beta, gamma, opts=opts):
                return ops.linear_attention("mamba3_mimo", q, k, v, g=g, beta=beta, gamma=gamma, **opts)

            fns[label] = (jax.jit(lambda *a, call=call: call(*a)[0]), harness.vjp_fn(call, tuple(range(6))))
        row["mimo kernels"] = dict(fwd=harness.kernel_breakdown_ms(fns["mimo"][0], *arrays, iterations=3),
                                   fwd_bwd=harness.kernel_breakdown_ms(fns["mimo"][1], *arrays, do, iterations=3))
        for rnd in range(args.rounds):
            if rnd == 1 and args.torch_python:  # the official kernels between our two rounds, same GPU
                row["official"] = run_official_subprocess(args, shape)
                if args.official_shared_qk:
                    row["official G=1"] = run_official_subprocess(args, shape, groups=1)
                for label in ("official", "official G=1"):
                    if label in row:
                        print(shape, label, {m: row[label].get(m) for m in ("fwd", "fwd+bwd", "error")}, flush=True)
            for label, (fwd, fwd_bwd) in fns.items():
                for mode, fn, fn_args in (("fwd", fwd, arrays), ("fwd+bwd", fwd_bwd, arrays + [do])):
                    try:
                        ms = kernel_ms(fn, *fn_args)
                    except Exception as err:  # noqa: BLE001
                        ms = f"{type(err).__name__}: {str(err)[:200]}"
                    row.setdefault(f"{label} {mode}", []).append(ms)
                    print(shape, label, mode, rnd, ms, flush=True)
        row["gpu_after"] = gpu_status()
        results.append(row)
        if args.out:  # partial results survive a timeout of a later shape
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(framework="jax", partial=True, results=results), indent=1, default=str))
        del x, do, arrays, fns
    return dict(framework="jax", jax=jax.__version__, device=jax.devices()[0].device_kind, results=results)


def kernel_ms(fn, *args, calls: int = 30, batch: int = 3) -> float:
    """Median CUPTI kernel time (ms) of ``calls`` calls of ``fn(*args)`` after 3 warmups.

    ``profiler.measure`` keeps every result of a measured batch alive, so the calls run in batches
    of ``batch`` (forward + backward outputs of the large shapes would not fit the memory cap).
    """
    import jax
    from jax.experimental.mosaic.gpu import profiler

    for _ in range(3):
        jax.block_until_ready(fn(*args))
    times: list[float] = []
    for _ in range(calls // batch):
        _, batch_times = profiler.measure(fn, iterations=batch)(*args)
        times += [t for t in batch_times if t is not None]
    return statistics.median(times)


def run_official_subprocess(args, shape, groups: int = 0) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", dir=args.tmp_dir, delete=False) as f:
        out = f.name
    cmd = [args.torch_python, str(Path(__file__).resolve()), "torch", "--mamba-root", args.mamba_root,
           "--shapes", ",".join(str(s) for s in shape), "--out", out, "--memory-fraction", str(args.memory_fraction),
           "--groups", str(groups)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.torch_timeout)
    if proc.returncode != 0:
        lines = [line for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
        return dict(error=(lines[-1] if lines else f"exit {proc.returncode}")[:300])
    data = json.loads(Path(out).read_text())
    os.unlink(out)
    return data["results"][0]


# ------------------------------------------------------------------------------------- torch


def import_official(mamba_root: str):
    """``mamba3_mimo`` from a source checkout, without running ``mamba_ssm/__init__.py``."""
    root = Path(mamba_root) / "mamba_ssm"
    for name, path in (
        ("mamba_ssm", root),
        ("mamba_ssm.ops", root / "ops"),
        ("mamba_ssm.ops.triton", root / "ops" / "triton"),
        ("mamba_ssm.ops.triton.mamba3", root / "ops" / "triton" / "mamba3"),
        ("mamba_ssm.ops.tilelang", root / "ops" / "tilelang"),
        ("mamba_ssm.ops.tilelang.mamba3", root / "ops" / "tilelang" / "mamba3"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo

    return mamba3_mimo


def run_torch(args) -> dict:
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile, schedule

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    official = import_official(args.mamba_root)
    dev = torch.device("cuda")
    results = []
    for shape in args.shapes:
        batch, seq_len, heads, rank, key_dim, value_dim = shape
        raw = make_inputs(*shape)
        t = {n: torch.from_numpy(a).to(dev) for n, a in raw.items()}
        groups = args.groups or heads  # B/C groups: one per head (our op's layout) or shared (official's best case)
        x = dict(
            q=t["q"][:, :, :, :groups].contiguous().to(torch.bfloat16),
            k=t["k"][:, :, :, :groups].contiguous().to(torch.bfloat16), v=t["v"].to(torch.bfloat16),
            adt=t["adt"].permute(0, 2, 1).contiguous(), dt=t["dt"].permute(0, 2, 1).contiguous(),
            trap=t["trap"].permute(0, 2, 1).contiguous().to(torch.bfloat16), psi=t["psi"],
        )
        do = t["do"].permute(0, 1, 3, 2, 4).contiguous().to(torch.bfloat16)  # official O [B, T, R, H, V]
        zeros_bias = torch.zeros((heads, rank, key_dim), device=dev)
        angles = torch.zeros((batch, seq_len, heads, key_dim // 4), device=dev)
        names = ("q", "k", "v", "adt", "dt", "trap", "psi")

        def call(a):
            return official(a["q"], a["k"], a["v"], a["adt"], a["dt"], a["trap"], zeros_bias, zeros_bias, a["psi"],
                            None, None, angles, None, None, chunk_size=64 // rank, rotary_dim_divisor=4,
                            dtype=torch.bfloat16)

        leaves = {n: x[n].detach().clone().requires_grad_(True) for n in names}

        def fwd():
            with torch.no_grad():
                return call(x)

        def fwd_bwd():
            out = call(leaves)
            return torch.autograd.grad(out, [leaves[n] for n in names], grad_outputs=do)

        def kernel_ms(fn) -> tuple[float, list]:
            active = 10
            with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=active)) as p:
                for _ in range(active + 2):
                    fn()
                    torch.cuda.synchronize()
                    p.step()
            # Mean time per recorded launch times launches per call: the profiler can lose whole calls' records
            # (bench_mamba3.py, kernel_times).
            totals: dict[str, tuple] = {}
            for e in p.events():
                if e.device_type == DeviceType.CUDA:
                    total, count = totals.get(e.name, (0.0, 0))
                    totals[e.name] = (total + e.device_time_total / 1e3, count + 1)
            per = {name: total / count * -(-count // active) for name, (total, count) in totals.items()}
            return sum(per.values()), sorted(per.items(), key=lambda kv: -kv[1])[:8]

        row = dict(shape=dict(zip(("B", "T", "H", "R", "K", "V"), shape)), gpu_before=gpu_status())
        for mode, fn in (("fwd", fwd), ("fwd+bwd", fwd_bwd)):
            try:
                for _ in range(3):  # compilation and warmup
                    fn()
                torch.cuda.synchronize()
                ms, kernels = kernel_ms(fn)
                row[mode] = dict(kernel_ms=ms, kernels=kernels)
            except Exception as err:  # noqa: BLE001
                row[mode] = dict(error=f"{type(err).__name__}: {str(err)[-400:]}")  # compiler errors end the text
        row["gpu_after"] = gpu_status()
        results.append(row)
    return dict(framework="torch", torch=torch.__version__, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("framework", choices=["jax", "torch"])
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--shapes", type=lambda s: [tuple(int(v) for v in t.split(",")) for t in s.split(";")],
                        default=SHAPES, help='"B,T,H,R,K,V;..."')
    parser.add_argument("--rounds", type=int, default=2, help="rounds of ours (the official runs after round 1)")
    parser.add_argument("--chunk", type=int, default=16, help="tokens per chunk of the chunked path")
    parser.add_argument("--baseline-limit", type=int, default=4 * 2048 * 24 * 16,
                        help="run siso and chunked only when B T H R^2 is at most this")
    parser.add_argument("--torch-python", default="", help="python of a torch + tilelang venv (official)")
    parser.add_argument("--mamba-root", default="", help="state-spaces/mamba source checkout (official)")
    parser.add_argument("--tmp-dir", default=None, help="directory for the official run's result file")
    parser.add_argument("--torch-timeout", type=int, default=420, help="seconds per official subprocess")
    parser.add_argument("--memory-fraction", type=float, default=0.25, help="torch per-process memory cap")
    parser.add_argument("--groups", type=int, default=0, help="torch: q/k groups of the official call (0: one per head)")
    parser.add_argument("--official-shared-qk", action="store_true",
                        help="jax: also time the official kernels with one q/k group shared by all heads")
    args = parser.parse_args()
    result = run_jax(args) if args.framework == "jax" else run_torch(args)
    result.update(time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print(json.dumps(result, indent=1, default=str)[:4000])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
