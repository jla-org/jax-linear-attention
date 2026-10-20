#!/usr/bin/env bash
# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build the FlashAttention-3 baseline of benchmarks/mla.md: Dao-AILab flash-attention main 965c921
# (hopper/, BSD-3) with MLA's head dims. FA3's default build compiles out head dim 192 ("FlashAttention
# forward only supports head dimension at most 128"); this build keeps only bf16 sm90 head dim 192 (q/k/v
# 192, which FA3's backward also runs for v 128) and the 192/128 forward (HDIMDIFF192), with causal/local,
# varlen, PackGQA, cluster and backward; every other head dim, fp8/fp16, paged/append KV, split-KV and
# softcap are compiled out. The flags are recorded in the benchmark JSON (baselines.fa3.build).
#
#   SRC=<flash-attention clone> VENV=<new venv dir> CUDA_HOME=/usr/local/cuda-12.9 bash benchmarks/build_fa3_mla.sh
#   python benchmarks/bench_mla_attention.py ... --fa3-python "$VENV/bin/python"
set -euo pipefail
: "${SRC:?set SRC to a clone of https://github.com/Dao-AILab/flash-attention}"
: "${VENV:?set VENV to the venv to create}"
: "${CUDA_HOME:=/usr/local/cuda-12.9}"
COMMIT=965c9212e6be33c821ee9f2b461564c5c91a2503
git -C "$SRC" checkout -q "$COMMIT"
git -C "$SRC" submodule update --init csrc/cutlass
[ -x "$VENV/bin/python" ] || uv venv --python python3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cu129 "torch==2.13.0+cu129"
uv pip install --python "$VENV/bin/python" numpy einops packaging ninja setuptools wheel psutil
export CUDA_HOME PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
for flag in FP8 FP16 HDIM64 HDIM96 HDIM128 HDIM256 HDIMDIFF64 SM80 PAGEDKV APPENDKV SPLIT SOFTCAP; do
  export "FLASH_ATTENTION_DISABLE_$flag=TRUE"
done
export MAX_JOBS="${MAX_JOBS:-20}" NVCC_THREADS="${NVCC_THREADS:-2}"
cd "$SRC/hopper"
rm -rf build dist
"$VENV/bin/python" setup.py bdist_wheel
uv pip install --python "$VENV/bin/python" --no-deps --reinstall dist/flash_attn_3-*.whl
"$VENV/bin/python" -c "
import torch, flash_attn_interface as fa
q = torch.randn(1, 256, 4, 192, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, 256, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
o = fa.flash_attn_func(q, q, v, causal=True)
o = o[0] if isinstance(o, tuple) else o
o.float().sum().backward()
print('FA3 head dim 192 / 128 build ok', tuple(o.shape))
"
