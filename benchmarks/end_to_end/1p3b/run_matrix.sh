#!/usr/bin/env bash
# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The end-to-end matrix of benchmarks/end_to_end_1p3b.md, one JSON (and one log) per run in $OUT:
#   DATA=/path/to/fineweb_edu_3B OUT=/path/to/results bash benchmarks/end_to_end/1p3b/run_matrix.sh <mode> [arch ...]
# <mode>: 1gpu (one GPU, Mosaic GPU kernels), 1gpu-chunked (one GPU, plain-JAX chunked ops + cuDNN
# attention), fsdp8 / dp2fsdp4 (8 GPUs: FSDP over all 8, or 2 replicas of 4-way FSDP; Mosaic GPU
# kernels) and fsdp8-chunked / dp2fsdp4-chunked (the same meshes with the chunked ops).
# Default architectures: every config (gdn-1p3b's K=256, V=512 heads run the K=256 Mosaic GPU kernels in
# the kernel modes), except the attention-bearing ones in the 8-GPU chunked modes, which keep the recorded
# chunked baselines; cuDNN attention compiles under FSDP since layers/attention.py pins q, k, v and the
# document mask to one sharding, so name those configs to run them.
# Every run is 20 steps of 65,536 tokens per GPU (524,288 tokens per step on 8 GPUs); the first 5
# steps are warm-up. Every mesh uses XLA's default (BFC) allocator (bench_e2e_1p3b.py --allocator auto).
# Every mode runs at the TOMLs' XLA tuning tier 1; EXTRA="--set runtime.xla_tuning_level=2" gives tier 2
# (see benchmarks/multi_gpu.md for what the tiers measured at 8 GPUs).
# Run from the repository root; CUDA_VISIBLE_DEVICES selects the GPUs. EXTRA adds bench_e2e_1p3b.py
# arguments to every run and TAG suffixes the file names, e.g.
#   EXTRA="--set data.document_masking=false" TAG=-nodocmask ... run_matrix.sh fsdp8 mamba2-1p3b
set -uo pipefail
: "${DATA:?set DATA to the CIDX prefix}" "${OUT:?set OUT to the result directory}"
mode=${1:?mode: 1gpu | 1gpu-chunked | fsdp8 | dp2fsdp4 | fsdp8-chunked | dp2fsdp4-chunked}
shift
linear="gdn-hd128-1p3b gdn-1p3b gdn2-1p3b kda-1p3b mamba2-1p3b mamba3-1p3b"
attention="transformer-1p3b transformer-swa-1p3b gdn-hybrid-1p3b kda-hybrid-1p3b"
# The kernel modes take each config's split and remat (full remat at 16 x 4096 x 1, except the transformers:
# no recompute + lean VJPs + unrolled layers at 4 x 4096 x 4, benchmarks/fp8_1p3b.md;
# a TOML with remat = "none" + lean_vjp, benchmarks/fp8_1p3b.md, keeps its own split); the chunked
# baselines force full remat at 16 x 4096 (EXTRA comes after these on the command line, so it still
# overrides them).
chunked=(--micro-batch 16 --grad-accum 1 --set 'model.remat="full"' --set model.lean_vjp=false --set model.lean_remat_core=false
         --set model.scan_layers=true)
case $mode in
  1gpu) args=(--impl kernels --dp 1 --fsdp 1) ;;
  1gpu-chunked) args=(--impl chunked --dp 1 --fsdp 1 "${chunked[@]}") ;;
  fsdp8) args=(--impl kernels --dp 1 --fsdp 8) ;;
  dp2fsdp4) args=(--impl kernels --dp 2 --fsdp 4) ;;
  fsdp8-chunked) args=(--impl chunked --dp 1 --fsdp 8 "${chunked[@]}") ;;
  dp2fsdp4-chunked) args=(--impl chunked --dp 2 --fsdp 4 "${chunked[@]}") ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
case $mode in
  1gpu-chunked) default="$attention $linear" ;;
  *-chunked) default="$linear" ;;
  *) default="$attention $linear" ;;
esac
archs=${*:-$default}
mkdir -p "$OUT"
status=0
for arch in $archs; do
  name="$arch-$mode${TAG:-}"
  echo "== $arch $mode $(date -u +%FT%TZ)"
  (
    # One GPU without recomputation (a TOML line remat = "none"; the lean GDN / KDA opt-in compiles to
    # 66-70 GiB) needs one preallocated pool (JAX's default preallocation) above JAX's default fraction
    # 0.75. The full-remat configs keep the caller's settings: a 0.92 pool leaves the CUDA driver little
    # room for the step's command buffer (benchmarks/fp8_1p3b.md). An EXTRA override to remat none is
    # not seen here: set XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 yourself.
    if [ "$mode" = 1gpu ] && grep -qE '^remat *= *"none"' "benchmarks/end_to_end/1p3b/$arch.toml" 2>/dev/null; then
      export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
    fi
    exec python benchmarks/bench_e2e_1p3b.py --arch "$arch" --data "$DATA" "${args[@]}" \
        --steps 20 --skip 5 --json "$OUT/$name.json" ${EXTRA:-}
  ) 2>&1 | tee "$OUT/$name.log"
  if [ "${PIPESTATUS[0]}" != 0 ]; then
    echo "== FAILED $arch $mode" >&2
    grep -m 3 -E "ran out of memory|RESOURCE_EXHAUSTED|E2E_ERROR" "$OUT/$name.log" >&2
    status=1
  fi
done
exit $status
