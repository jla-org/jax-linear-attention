# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy generation throughput: ``jla.inference.generate`` vs Hugging Face ``generate``.

Random-init models of the same shapes, bf16, a batch of equal-length random prompts,
exactly ``--new`` new tokens (no EOS). Every implementation runs in its own
subprocess, one after another, on the same GPU:

* ``jla``: ``transformer-125m``, ``gdn-hybrid-125m`` and ``gdn-125m-all`` (the hybrid's shapes with
  Gated DeltaNet in all 12 layers), jit-compiled decode loop;
* ``hf``: transformers ``Qwen3ForCausalLM`` with ``transformer-125m``'s shapes (eager, dynamic cache)
  and, with ``--hf-static``, ``cache_implementation="static"``, for which ``generate`` compiles the
  decoding forward pass itself (``torch.compile``, CUDA graphs; HF's fastest generate path);
* ``fla``: flash-linear-attention ``GatedDeltaNetForCausalLM`` with ``gdn-125m-all``'s shapes, once per
  ``--fla-python`` (venvs with FLA; e.g. FLA main with Triton kernels only, and FLA main with its
  optional backends). FLA's hybrid attention layers need the ``flash-attn`` package, which the
  benchmark venvs do not have, so the FLA baseline has no attention layers.

    python benchmarks/bench_generate.py --batch 8 --prompt 128 --new 128 --hf-static \\
        --torch-python /path/to/torch-venv/bin/python --fla-python /path/to/fla-venv/bin/python \\
        --out benchmarks/results/generate.json

Reported: best of ``--rounds`` timed calls after one warmup call (which compiles);
``tokens_per_s = batch * new / seconds`` counts new tokens only (prefill included in the time).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

VOCAB = 50304


def _jla(preset: str, batch: int, prompt: int, new: int, rounds: int) -> dict:
    import jax
    import numpy as np

    from jla.inference import generate
    from jla.models import build_model, get_preset

    if preset == "gdn-125m-all":
        cfg = get_preset("gdn-hybrid-125m")
        cfg = cfg.replace(layer_types=("gated_deltanet",) * cfg.n_layers)
    else:
        cfg = get_preset(preset)
    model = build_model(cfg, key=jax.random.key(0))
    prompts = np.random.default_rng(0).integers(0, cfg.vocab_size, (batch, prompt)).tolist()
    start = time.perf_counter()
    generate(model, prompts, max_new_tokens=new)
    first = time.perf_counter() - start
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        out = generate(model, prompts, max_new_tokens=new)
        times.append(time.perf_counter() - start)
    assert int(out.lengths.min()) == new
    return {"impl": "jla", "model": preset, "first_call_s": first, "seconds": min(times), "all_s": times}


def _torch_generate(model, batch: int, prompt: int, new: int, rounds: int, **kwargs) -> tuple[float, list[float]]:
    import torch

    ids = torch.randint(0, VOCAB, (batch, prompt), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    mask = torch.ones_like(ids)
    args = dict(attention_mask=mask, max_new_tokens=new, min_new_tokens=new, do_sample=False, pad_token_id=0, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        model.generate(ids, **args)
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    times = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(ids, **args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    assert out.shape == (batch, prompt + new)
    return first, times


def _hf(batch: int, prompt: int, new: int, rounds: int, static: bool) -> dict:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=768, intermediate_size=2304, num_hidden_layers=12,
                      num_attention_heads=12, num_key_value_heads=4, head_dim=64, tie_word_embeddings=True,
                      max_position_embeddings=4096, rope_theta=10_000.0)
    model = Qwen3ForCausalLM(cfg).to("cuda", dtype=torch.bfloat16).eval()
    # The static cache makes generate() compile its decoding forward pass (CompileConfig defaults);
    # compiling model.forward by hand instead breaks on CUDA-graph output reuse.
    kwargs = {"cache_implementation": "static"} if static else {}
    first, times = _torch_generate(model, batch, prompt, new, rounds, **kwargs)
    name = "hf-qwen3-static-compiled" if static else "hf-qwen3-eager"
    import transformers

    return {"impl": name, "model": "transformer-125m shapes", "first_call_s": first, "seconds": min(times),
            "all_s": times, "torch": torch.__version__, "transformers": transformers.__version__}


def _fla(batch: int, prompt: int, new: int, rounds: int) -> dict:
    import fla
    import torch
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    torch.cuda.set_per_process_memory_fraction(0.25)
    torch.manual_seed(0)
    cfg = GatedDeltaNetConfig(
        hidden_size=768, num_hidden_layers=12, head_dim=128, num_heads=4, num_v_heads=8, expand_v=1,
        intermediate_size=2048, vocab_size=VOCAB, tie_word_embeddings=True, attn=None,
    )
    model = GatedDeltaNetForCausalLM(cfg).to("cuda", dtype=torch.bfloat16).eval()
    first, times = _torch_generate(model, batch, prompt, new, rounds)
    return {"impl": "fla-gated-deltanet", "model": "gdn-125m-all shapes", "first_call_s": first,
            "seconds": min(times), "all_s": times, "python": sys.executable, "fla": fla.__version__,
            "torch": torch.__version__}


def _run_child(python: str, impl: str, args: argparse.Namespace) -> list[dict]:
    cmd = [python, __file__, "--child", impl, "--batch", str(args.batch), "--prompt", str(args.prompt),
           "--new", str(args.new), "--rounds", str(args.rounds)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.child_timeout)
    if proc.returncode != 0:
        return [{"impl": impl, "python": python, "error": proc.stderr.strip().splitlines()[-3:] or ["failed"]}]
    return [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--new", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hf-static", action="store_true", help="also time HF's static cache + torch.compile")
    parser.add_argument("--torch-python", help="python of a venv with torch + transformers (the 'hf' baselines)")
    parser.add_argument("--fla-python", action="append", default=[],
                        help="python of a venv with FLA for the 'fla' baseline (repeatable)")
    parser.add_argument("--child-timeout", type=int, default=900)
    parser.add_argument("--out", help="write the results JSON here")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    shape = {"batch": args.batch, "prompt": args.prompt, "new": args.new}
    if args.child:
        if args.child == "jla":
            rows = [_jla(p, args.batch, args.prompt, args.new, args.rounds)
                    for p in ("transformer-125m", "gdn-hybrid-125m", "gdn-125m-all")]
        elif args.child == "hf":
            rows = [_hf(args.batch, args.prompt, args.new, args.rounds, static=False)]
        elif args.child == "hf-static":
            rows = [_hf(args.batch, args.prompt, args.new, args.rounds, static=True)]
        elif args.child == "fla":
            rows = [_fla(args.batch, args.prompt, args.new, args.rounds)]
        else:
            raise SystemExit(f"unknown child {args.child}")
        for row in rows:
            row.update(shape, tokens_per_s=args.batch * args.new / row["seconds"])
            print(json.dumps(row))
        return 0
    torch_python = args.torch_python or sys.executable
    rows = _run_child(sys.executable, "jla", args)
    rows += _run_child(torch_python, "hf", args)
    if args.hf_static:
        rows += _run_child(torch_python, "hf-static", args)
    for python in args.fla_python:
        rows += _run_child(python, "fla", args)
    smi = ""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and "," not in visible:  # only the GPU this run was pinned to
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                                  "--format=csv,noheader", "-i", visible],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            smi = ""
    report = {"shape": shape, "rows": rows, "nvidia_smi_after": smi}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
