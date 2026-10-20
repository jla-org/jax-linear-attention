#!/usr/bin/env bash
# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Training throughput of the 1.3B models in this directory in the setting of the Gated DeltaNet-2 paper's Fig. 2
# (arXiv 2605.22791): one GPU, bf16, 16,384 tokens per step as seq_len x batch = 2K x 8, 4K x 4, 8K x 2 and 16K x 1.
# One JSON and one log per run in $OUT:
#   OUT=/path/to/results bash benchmarks/end_to_end/gdn2_paper/run.sh <model> [<vocab> [<seq_len>x<batch> ...]]
# <model>: a TOML in this directory without .toml (gdn2-hybrid, kda-hybrid, gdn-hybrid, mamba2-hybrid, mamba3-hybrid,
# mamba3-mimo-hybrid, transformer); <vocab>: 32000 (the TOMLs' vocabulary) or another size, e.g. 128256; the splits
# default to all four. Run from the repository root; CUDA_VISIBLE_DEVICES selects the GPU. Each run is
# bench_e2e_1p3b.py with every mixer and the attention on its Mosaic GPU kernel (--impl kernels), 30 steps of which
# the first 10 are warm-up, synthetic uniform token ids (the TOMLs' data.source; --data only names the run's data),
# and the allocator settings below. EXTRA adds bench_e2e_1p3b.py arguments and TAG suffixes the file names.
set -uo pipefail
: "${OUT:?set OUT to the result directory}"
model=${1:?model: a TOML name in benchmarks/end_to_end/gdn2_paper}
vocab=${2:-32000}
shift $(( $# < 2 ? $# : 2 ))
splits=${*:-2048x8 4096x4 8192x2 16384x1}
here=benchmarks/end_to_end/gdn2_paper
[ -f "$here/$model.toml" ] || { echo "no $here/$model.toml (run from the repository root)" >&2; exit 2; }
# Allocator. Without recompute the step's temporaries are one block of 30-56 GiB next to the fp32 weights, gradients
# and AdamW state (21 GiB at 1.3B, 25 GiB with the 128,256-token vocabulary). XLA's BFC allocator places it with
# preallocation off and 0.92 of the GPU at 32,000 tokens; at 128,256 tokens it often cannot (a 41-56 GiB block next
# to 25 GiB fails in its growth regions, and a preallocated BFC pool fragments within a few steps), so every
# 128,256-token run, the transformer's included, uses the CUDA async allocator with one pool of 0.97 of the GPU.
allocator=()
prealloc=false
fraction=0.92
if [ "$vocab" -gt 32000 ]; then
  allocator=(--allocator cuda_async)
  prealloc=true
  fraction=0.97
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-$prealloc}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$fraction}
mkdir -p "$OUT"
status=0
for split in $splits; do
  seq=${split%x*}
  batch=${split#*x}
  name="$model-v$vocab-${seq}x$batch${TAG:-}"
  echo "== $name $(date -u +%FT%TZ)"
  python benchmarks/bench_e2e_1p3b.py --arch "$here/$model.toml" --data synthetic --set 'data.paths=[]' \
      --set "model.vocab_size=$vocab" --impl kernels --dp 1 --fsdp 1 --seq-len "$seq" --micro-batch "$batch" \
      --grad-accum 1 --steps 30 --skip 10 ${allocator[@]+"${allocator[@]}"} --json "$OUT/$name.json" ${EXTRA:-} > "$OUT/$name.log" 2>&1
  rc=$?
  grep -h "^E2E_RESULT" "$OUT/$name.log"
  if [ $rc != 0 ]; then
    echo "== FAILED $name (exit $rc)" >&2
    grep -m 3 -E "ran out of memory|RESOURCE_EXHAUSTED|E2E_ERROR|Error" "$OUT/$name.log" >&2
    status=1
  fi
done
exit $status
