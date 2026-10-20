# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face import/export at real model shapes (random weights): wall time and logit parity.

    python benchmarks/bench_convert_hf.py --shape qwen3-0.6b --workdir /path/to/workdir --out results/convert.json

Builds the model with ``transformers`` (random init, no download), saves it as a sharded
bfloat16 checkpoint like the published ones, then times ``transformers`` ``from_pretrained``
(the baseline loader, float32 on the host, and followed by ``.to("cuda")``) against
``jla.convert.load_hf_checkpoint`` (float32 parameters on the default JAX device, every
array ready, and on the JAX host CPU device), compares float32
logits (JAX matmuls at ``highest`` precision, so no TF32), and times ``save_hf_checkpoint``
and checks the exported tensors equal the original bytes. Times are medians over
``--repeats`` runs with a warm page cache. Needs ``torch`` and ``transformers``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers

from jla.convert import load_hf_checkpoint, save_hf_checkpoint
from jla.convert.safetensors_io import load_checkpoint_dir

SHAPES = {
    "qwen3-0.6b": dict(model_type="qwen3", vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                       num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                       rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=40960,
                       rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0}),
    "llama-3.2-1b": dict(model_type="llama", vocab_size=128256, hidden_size=2048, intermediate_size=8192,
                         num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, head_dim=64,
                         rms_norm_eps=1e-5, tie_word_embeddings=True, max_position_embeddings=131072,
                         rope_parameters={"rope_type": "llama3", "rope_theta": 500_000.0, "factor": 32.0,
                                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                          "original_max_position_embeddings": 8192}),
}


def _ready(model) -> None:
    jax.block_until_ready(eqx.filter(model, eqx.is_array))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), default="qwen3-0.6b")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    spec = dict(SHAPES[args.shape])
    model_type = spec.pop("model_type")
    work = Path(args.workdir)
    shutil.rmtree(work, ignore_errors=True)
    torch.manual_seed(0)
    cfg = transformers.AutoConfig.for_model(model_type, **spec)
    hf = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", dtype=torch.float32)
    hf.to(torch.bfloat16).save_pretrained(work / "hf", max_shard_size="500MB")
    del hf
    device = jax.devices()[0]
    jnp.zeros(8).block_until_ready()  # backend start-up outside the timings
    result: dict = {"shape": args.shape, "model_type": model_type, "jax_device": device.device_kind,
                    "torch_threads": torch.get_num_threads(), "repeats": args.repeats,
                    "checkpoint_bytes": sum(p.stat().st_size for p in (work / "hf").glob("*.safetensors"))}

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        hf = transformers.AutoModelForCausalLM.from_pretrained(work / "hf", attn_implementation="eager",
                                                               dtype=torch.float32).eval()
        times.append(time.perf_counter() - t0)
    result["hf_from_pretrained_s"] = statistics.median(times)
    result["parameters"] = sum(p.numel() for p in hf.parameters())
    if torch.cuda.is_available():  # the same load followed by the copy to the GPU
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            on_gpu = transformers.AutoModelForCausalLM.from_pretrained(work / "hf", attn_implementation="eager",
                                                                       dtype=torch.float32).to("cuda")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            del on_gpu
        torch.cuda.empty_cache()
        result["hf_from_pretrained_to_gpu_s"] = statistics.median(times)

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        model, config = load_hf_checkpoint(work / "hf", compute_dtype="float32")
        _ready(model)
        times.append(time.perf_counter() - t0)
    result["jla_load_hf_checkpoint_s"] = statistics.median(times)
    times = []
    with jax.default_device(jax.devices("cpu")[0]):  # host only, like the from_pretrained baseline
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            on_host, _ = load_hf_checkpoint(work / "hf", compute_dtype="float32")
            _ready(on_host)
            times.append(time.perf_counter() - t0)
            del on_host
    result["jla_load_hf_checkpoint_host_s"] = statistics.median(times)

    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, args.tokens)).astype(np.int32)
    with torch.no_grad():
        want = hf(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
    out = model(jnp.asarray(ids), inference=True)
    got = np.asarray(model.logits(out.hidden))
    result["max_abs_logit_diff_fp32"] = float(np.max(np.abs(got - want)))
    result["max_abs_logit"] = float(np.max(np.abs(want)))
    result["argmax_agreement"] = float(np.mean(np.argmax(got, -1) == np.argmax(want, -1)))

    times = []
    for _ in range(args.repeats):
        shutil.rmtree(work / "out", ignore_errors=True)
        t0 = time.perf_counter()
        save_hf_checkpoint(model, work / "out", model_type, dtype="bfloat16", base_config=work / "hf",
                           max_shard_bytes=500 * 1024**2)
        times.append(time.perf_counter() - t0)
    result["jla_save_hf_checkpoint_s"] = statistics.median(times)
    times = []
    for _ in range(args.repeats):
        shutil.rmtree(work / "hf_out", ignore_errors=True)
        t0 = time.perf_counter()
        hf.to(torch.bfloat16).save_pretrained(work / "hf_out", max_shard_size="500MB")
        times.append(time.perf_counter() - t0)
        hf.float()
    result["hf_save_pretrained_bf16_s"] = statistics.median(times)
    original, exported = load_checkpoint_dir(work / "hf"), load_checkpoint_dir(work / "out")
    result["export_identical"] = sorted(original) == sorted(exported) and all(
        np.array_equal(original[k].view(np.uint16), exported[k].view(np.uint16)) for k in original)
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
