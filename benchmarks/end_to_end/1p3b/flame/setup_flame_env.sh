#!/usr/bin/env bash
# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build the PyTorch environment of the flame baseline (benchmarks/end_to_end_1p3b.md):
#   torch 2.13.0+cu129 (and the triton it pins), FLA main c6e4f00 (fla-core + flash-linear-attention),
#   TileLang 0.1.15 + FlashQLA main 5538c35 and FlashKDA main 7afb9f45 (the optional backends FLA's
#   default dispatch uses), fla-org/flame main 24dea91 and torchtitan 0b44d4c (the commit flame's
#   README pins).
#
#   VENV=/path/to/venv-flame WORK=/path/to/build-dir bash setup_flame_env.sh
#
# Required: FA3_WHEEL=<FlashAttention-3 wheel built from Dao-AILab/flash-attention hopper/ at $FA3_COMMIT
# against torch 2.13.0+cu129> (the transformer configs need it) and MAMBA_WHEEL=<mamba_ssm wheel built from
# state-spaces/mamba at $MAMBA_COMMIT against the same torch> (FLA's Mamba-2/Mamba-3 fast paths). The script
# stops without them; SKIP_FA3=1 / SKIP_MAMBA=1 build a reduced environment on purpose (transformer configs
# fail without FA3; FLA's Mamba-2 falls back to a slow torch loop and Mamba-3 fails without mamba_ssm).
# Optional: FLASHKDA_WHEEL=<wheel of FlashKDA 7afb9f45 already built for sm_90a> skips the CUTLASS build;
# FLA_REPO / QLA_REPO / KDA_REPO / FLAME_REPO / TT_REPO may point at local clones (default: GitHub).
# Writes $VENV/FLAME_ENV_COMMITS.txt (every commit and the freeze). Needs uv, git, python3.12 and,
# without FLASHKDA_WHEEL, CUDA 12.9 at CUDA_HOME.
set -euo pipefail
: "${VENV:?set VENV}" "${WORK:?set WORK}"
if [ -z "${FA3_WHEEL:-}" ] && [ "${SKIP_FA3:-0}" != 1 ]; then
  echo "== ERROR: set FA3_WHEEL (FlashAttention-3 wheel, see the header) or SKIP_FA3=1"; exit 1
fi
if [ -z "${MAMBA_WHEEL:-}" ] && [ "${SKIP_MAMBA:-0}" != 1 ]; then
  echo "== ERROR: set MAMBA_WHEEL (mamba_ssm wheel, see the header) or SKIP_MAMBA=1"; exit 1
fi
UV=${UV:-uv}
PY=${PYTHON:-python3.12}
IDX=https://download.pytorch.org/whl/cu129
FLA_COMMIT=c6e4f00d92fa2e3ed7232159d7be6fbbcff5c69f
QLA_COMMIT=5538c35f0df33ef804eacd90c77d9d9087c1d4e4
KDA_COMMIT=7afb9f454f160a6c4bbc0999beca0a8c40a38934
FLAME_COMMIT=24dea9133e2166da89a563afbdcefb7c55615646
TT_COMMIT=0b44d4c437c424b6bf719661c0eb4283dc4068bc
FA3_COMMIT=965c9212e6be33c821ee9f2b461564c5c91a2503  # Dao-AILab/flash-attention main, hopper/
MAMBA_COMMIT=e9594ce  # state-spaces/mamba main
FLA_REPO=${FLA_REPO:-https://github.com/fla-org/flash-linear-attention.git}
QLA_REPO=${QLA_REPO:-https://github.com/QwenLM/FlashQLA.git}
KDA_REPO=${KDA_REPO:-https://github.com/MoonshotAI/FlashKDA.git}
FLAME_REPO=${FLAME_REPO:-https://github.com/fla-org/flame.git}
TT_REPO=${TT_REPO:-https://github.com/pytorch/torchtitan.git}
mkdir -p "$WORK"
cd "$WORK"
echo "== start $(date -u +%FT%TZ)"

clone() {  # clone <repo> <dir> <commit>
  rm -rf "$2" && git clone --quiet "$1" "$2" && git -C "$2" checkout --quiet "$3"
  echo "== $2 $(git -C "$2" rev-parse HEAD) $(git -C "$2" log -1 --format=%cs)"
}
clone "$FLA_REPO" fla-src "$FLA_COMMIT"
clone "$QLA_REPO" flashqla-src "$QLA_COMMIT"
clone "$FLAME_REPO" flame-src "$FLAME_COMMIT"
clone "$TT_REPO" torchtitan-src "$TT_COMMIT"

[ -x "$VENV/bin/python" ] || $UV venv --python "$PY" "$VENV"
# 1. torch and FLA main (fla-core holds the kernels, flash-linear-attention the layers and models).
$UV pip install --python "$VENV/bin/python" --extra-index-url $IDX "torch==2.13.0+cu129" pytest numpy einops
if [ -f fla-src/fla-core/pyproject.toml ] || [ -f fla-src/fla-core/setup.py ]; then
  $UV pip install --python "$VENV/bin/python" --no-deps ./fla-src/fla-core
fi
$UV pip install --python "$VENV/bin/python" --extra-index-url $IDX "torch==2.13.0+cu129" ./fla-src
# 2. TileLang 0.1.15 and FlashQLA main (torch stays pinned).
QLA_VERSION_SUFFIX=+git5538c35 $UV pip install --python "$VENV/bin/python" --extra-index-url $IDX \
  "torch==2.13.0+cu129" "tilelang==0.1.15" ./flashqla-src
# 3. FlashKDA main: a prebuilt wheel of the same commit, or the CUTLASS build for sm_90a.
if [ -n "${FLASHKDA_WHEEL:-}" ]; then
  $UV pip install --python "$VENV/bin/python" --no-deps "$FLASHKDA_WHEEL"
  echo "== FlashKDA from wheel $FLASHKDA_WHEEL sha256 $(sha256sum "$FLASHKDA_WHEEL" | cut -c1-16)"
else
  clone "$KDA_REPO" flashkda-src "$KDA_COMMIT"
  (cd flashkda-src && { git submodule update --init --depth 1 cutlass || git submodule update --init cutlass; })
  $UV pip install --python "$VENV/bin/python" --extra-index-url $IDX "torch==2.13.0+cu129" ninja setuptools wheel packaging
  (
    export CUDA_VISIBLE_DEVICES= CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9} FLASH_KDA_CUDA_ARCHS=90a
    export NVCC_THREADS=${NVCC_THREADS:-8} MAX_JOBS=${MAX_JOBS:-8} PATH=$CUDA_HOME/bin:$PATH
    cd flashkda-src && $UV pip install --python "$VENV/bin/python" --no-build-isolation .
  )
fi
# 4. torchtitan 0b44d4c and flame main, without letting their requirements move torch or FLA.
#    flame/data.py imports datasets.iterable_dataset.ShufflingConfig, which datasets 4 removed.
$UV pip install --python "$VENV/bin/python" --extra-index-url $IDX "torch==2.13.0+cu129" \
  "torchdata>=0.8.0" tomli tensorboard tiktoken blobfile tabulate wandb fsspec tyro "transformers>=4.45.0" ninja
$UV pip install --python "$VENV/bin/python" "datasets>=3.6.0,<4"
$UV pip install --python "$VENV/bin/python" --no-deps ./torchtitan-src ./flame-src
# 5. Kernels FLA's layers import at run time: FlashAttention-3 (hopper/ of Dao-AILab/flash-attention main
#    965c921; fla.layers.attn reaches it through end_to_end/1p3b/flame/fa3_shim, FA2 has no torch 2.13
#    wheel) and state-spaces/mamba (mamba_ssm + causal-conv1d: FLA's Mamba-2/Mamba-3 fast paths;
#    without them fla.layers.mamba2 falls back to a slow torch loop). Pass prebuilt wheels built against
#    this torch (FA3_WHEEL, MAMBA_WHEEL); causal-conv1d builds from PyPI.
if [ -n "${FA3_WHEEL:-}" ]; then $UV pip install --python "$VENV/bin/python" --no-deps "$FA3_WHEEL"; fi
if [ -n "${MAMBA_WHEEL:-}" ]; then
  $UV pip install --python "$VENV/bin/python" --no-deps "$MAMBA_WHEEL"
  $UV pip install --python "$VENV/bin/python" --no-deps --no-build-isolation "causal-conv1d==1.7.0"
fi

{
  echo "fla $FLA_COMMIT ($FLA_REPO)"
  echo "flashqla $QLA_COMMIT ($QLA_REPO)"
  echo "flashkda $KDA_COMMIT ($KDA_REPO) ${FLASHKDA_WHEEL:+wheel $(sha256sum "$FLASHKDA_WHEEL" | cut -d' ' -f1)}"
  echo "flame $FLAME_COMMIT ($FLAME_REPO)"
  echo "torchtitan $TT_COMMIT ($TT_REPO)"
  if [ -n "${FA3_WHEEL:-}" ]; then echo "flash-attention $FA3_COMMIT (Dao-AILab/flash-attention hopper/) wheel $(sha256sum "$FA3_WHEEL" | cut -d' ' -f1)"; fi
  if [ -n "${MAMBA_WHEEL:-}" ]; then
    echo "mamba $MAMBA_COMMIT (state-spaces/mamba) wheel $(sha256sum "$MAMBA_WHEEL" | cut -d' ' -f1)"
    echo "causal-conv1d 1.7.0 (PyPI sdist, built against torch 2.13.0+cu129)"
  fi
  echo "-- freeze"
  $UV pip freeze --python "$VENV/bin/python" | sed 's/\x1b\[[0-9;]*m//g'
} > "$VENV/FLAME_ENV_COMMITS.txt"
grep -q "^torch==2.13.0+cu129$" "$VENV/FLAME_ENV_COMMITS.txt" || { echo "== ERROR: torch is not 2.13.0+cu129"; exit 1; }
if grep -q -i -E "^nvidia-cuda-nvcc|^nvidia-nvvm|^nvidia-cuda-crt" "$VENV/FLAME_ENV_COMMITS.txt"; then
  echo "== ERROR: CUDA 13 nvcc wheels present (they break TileLang's compile)"; exit 1
fi
# flash_qla reads the GPU's compute capability on import, so it is imported in the GPU smoke test.
CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -I -c "import torch, triton, fla, tilelang, flash_kda, torchtitan, flame; print('import ok: torch', torch.__version__, 'triton', triton.__version__, 'fla', fla.__version__, 'tilelang', tilelang.__version__, 'flame', flame.__version__)"
if [ -n "${FA3_WHEEL:-}" ]; then CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -I -c "import flash_attn_interface; print('import ok: flash_attn_interface (FA3)')"; fi
if [ -n "${MAMBA_WHEEL:-}" ]; then CUDA_VISIBLE_DEVICES= "$VENV/bin/python" -I -c "import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined; print('import ok: mamba_ssm', mamba_ssm.__version__, '+ causal_conv1d')"; fi
echo "== done $(date -u +%FT%TZ)"
