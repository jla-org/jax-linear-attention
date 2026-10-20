# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One train step of a preset in bf16 and with FP8 projections and MLPs (``ModelConfig.fp8``).

    python benchmarks/bench_fp8.py --preset transformer-125m --recipes off,delayed,current \
        --batch 8 --seq 2048 --steps 10 [--json out.json]
    python benchmarks/bench_fp8.py --gemms   # which GEMM kernels XLA picks for one FP8 matmul

The step is :func:`jla.training.step.build_train_step` with AdamW and :func:`lm_loss` on
uniform random tokens, one device. Reported: median ms/step and tok/s over ``--steps`` timed
steps after ``--warmup`` steps, and the FP8 GEMMs of the compiled step (cuBLASLt FP8 calls and
Triton GEMM fusions of FP8 dots).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jla.layers.fp8 import Fp8State, fp8_dot
from jla.models import build_model, get_preset
from jla.training.step import LossConfig, build_train_step, lm_loss

_FP8_GEMM = "__cublas$lt$matmul$f8"


def _hlo_text(compiled) -> str:
    return (compiled if hasattr(compiled, "as_text") else compiled.compiled).as_text()  # equinox wraps it


def _temp_gib(compiled) -> float | None:
    """The compiled step's temporary buffer size (GiB), if XLA reports it."""
    stats = (compiled if hasattr(compiled, "memory_analysis") else compiled.compiled).memory_analysis()
    temp = getattr(stats, "temp_size_in_bytes", None)
    return None if temp is None else round(temp / 2**30, 2)


_DEF = re.compile(r"^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*([a-z0-9]+)\[")
_DOT = re.compile(r"\sdot\((%[\w.\-]+),\s*(%[\w.\-]+)\)")


def fp8_dot_fusions(text: str) -> int:
    """Dots left in fusions (Triton GEMMs) whose two operands are FP8 (names are module-unique)."""
    types = {m.group(1): m.group(2) for m in map(_DEF.match, text.splitlines()) if m}
    return sum(
        1
        for m in map(_DOT.search, text.splitlines())
        if m and all(types.get(name, "").startswith("f8") for name in m.groups())
    )


_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?(%?[\w.\-]+)\s.*\{\s*$")


def fp8_input_gemm_fusions(text: str) -> int:
    """Fused computations (Triton GEMMs) that contain a dot and take FP8 values."""
    count, body, name = 0, [], None
    for line in text.splitlines():
        if name is None:
            m = _COMPUTATION.match(line)
            if m:
                name, body = m.group(1), []
            continue
        if line.strip() == "}":
            joined = "\n".join(body)
            if "fus" in name and " dot(" in joined and re.search(r"= f8e\w+\[[^\]]*\][^ ]* parameter\(", joined):
                count += 1
            name = None
        else:
            body.append(line)
    return count


def gemm_summary(text: str) -> dict[str, Any]:
    """GEMM kinds of the compiled program: custom-call targets, FP8 cuBLASLt output types, FP8 fusions."""
    counts: dict[str, Any] = {}
    for target in re.findall(r'custom_call_target="([^"]+)"', text):
        counts[target] = counts.get(target, 0) + 1
    counts["f8_dots_left_as_dot"] = fp8_dot_fusions(text)
    counts["f8_input_gemm_fusions"] = fp8_input_gemm_fusions(text)
    out_types: dict[str, int] = {}
    for line in text.splitlines():
        if _FP8_GEMM in line:
            m = re.search(r"=\s*\(?([a-z0-9]+)\[", line)
            if m:
                out_types[m.group(1)] = out_types.get(m.group(1), 0) + 1
    counts["f8_cublaslt_out_types"] = out_types
    return counts


def probe_gemms(m: int = 4096, k: int = 2048, n: int = 2048) -> None:
    """Compile the forward and both backward GEMMs of one FP8 matmul and print their kernels."""
    x = jnp.ones((m, k), jnp.bfloat16)
    w = jnp.ones((k, n), jnp.float32)
    g = jnp.ones((m, n), jnp.bfloat16)
    for recipe in ("delayed", "current"):
        state = Fp8State.init(16) if recipe == "delayed" else None

        def fwd_bwd(x, w, g, state):
            y, vjp = jax.vjp(lambda a, b, s: fp8_dot(a, b, s, recipe=recipe), x, w, state)
            return y, vjp(g)

        text = _hlo_text(jax.jit(fwd_bwd).lower(x, w, g, state).compile())
        print(recipe, json.dumps(gemm_summary(text)))
        for line in text.splitlines():
            if "custom_call_target" in line and "matmul" in line:
                print("   ", line.strip()[:300])


def bench(preset: str, recipe: str, batch: int, seq: int, steps: int, warmup: int, remat: str | None,
          optimizer_name: str = "adamw", param_dtype: str | None = None) -> dict:
    cfg = get_preset(preset).replace(fp8=recipe)
    if remat is not None:
        cfg = cfg.replace(remat=remat)
    if param_dtype is not None:
        cfg = cfg.replace(param_dtype=param_dtype)
    model = eqx.filter_jit(lambda k: build_model(cfg, key=k))(jax.random.key(0))
    optimizer = optax.adamw(1e-4, weight_decay=0.1) if optimizer_name == "adamw" else optax.sgd(1e-4)
    opt_state = jax.jit(optimizer.init)(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig()))
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(1, batch, seq + 1), dtype=np.int32)
    data = {"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])}
    key = jax.random.key(1)
    t0 = time.perf_counter()
    compiled = step.lower(model, opt_state, data, key).compile()
    step.use_compiled(compiled)
    compile_s = time.perf_counter() - t0
    gemms = gemm_summary(_hlo_text(compiled))
    temp_gib = _temp_gib(compiled)
    times, losses = [], []
    for i in range(warmup + steps):
        t = time.perf_counter()
        model, opt_state, metrics, key = step(model, opt_state, data, key)
        loss = float(metrics["loss"])  # blocks until the step is done
        if i >= warmup:
            times.append(time.perf_counter() - t)
            losses.append(loss)
    ms = 1e3 * statistics.median(times)
    return {
        "preset": preset,
        "recipe": recipe,
        "remat": cfg.remat,
        "param_dtype": cfg.param_dtype,
        "optimizer": optimizer_name,
        "batch": batch,
        "seq": seq,
        "device": str(jax.devices()[0].device_kind),
        "ms_per_step": round(ms, 2),
        "tok_per_s": round(batch * seq / (ms / 1e3)),
        "compile_s": round(compile_s, 1),
        "temp_gib": temp_gib,
        "fp8_cublaslt_gemms": gemms.get(_FP8_GEMM, 0),
        "fp8_cublaslt_out_types": gemms["f8_cublaslt_out_types"],
        "fp8_dots_in_fusions": gemms["f8_dots_left_as_dot"],
        "fp8_input_gemm_fusions": gemms["f8_input_gemm_fusions"],
        "gemm_calls": {name: count for name, count in gemms.items() if "matmul" in name or "gemm" in name},
        "last_loss": round(losses[-1], 4),
        "ms_all": [round(1e3 * t, 2) for t in times],
        "gpu_util": _gpu_utilization(),
    }


def _gpu_utilization() -> str | None:
    """``utilization.gpu, memory.used`` of this process's GPU from nvidia-smi (None off-GPU)."""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if jax.default_backend() != "gpu" or not device or "," in device:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", device, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gemms", action="store_true", help="print the GEMM kernels of one FP8 matmul and exit")
    p.add_argument("--preset", default="transformer-125m")
    p.add_argument("--recipes", default="off,delayed,current")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--remat", default=None)
    p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw",
                   help="sgd keeps no moments (fits a 1B model in a quarter of an H100)")
    p.add_argument("--param-dtype", default=None, help="override the preset's parameter dtype, e.g. bfloat16")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.gemms:
        probe_gemms()
        return
    results = []
    for recipe in args.recipes.split(","):
        result = bench(args.preset, recipe, args.batch, args.seq, args.steps, args.warmup, args.remat, args.optimizer,
                       args.param_dtype)
        print(json.dumps(result), flush=True)
        results.append(result)
        if args.json:  # rewritten after every recipe, so a later failure keeps the earlier rows
            with open(args.json, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
