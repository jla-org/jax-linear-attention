# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent check of the records both trainers draw: the sum of the input ids of the first S global batches.

    python benchmarks/end_to_end/1p3b/flame/token_checksum.py <CIDX prefix> <global batch G> <steps S> [seed]

Re-implements the jla data order without importing jla (numpy only): the order is
``RandomState(seed).shuffle(arange(n))``, global batch ``s`` is ``order[s*G:(s+1)*G]`` and record ``i`` is
``tokens[offsets[i]:offsets[i] + 4096]`` of the CIDX file (exact packed records). The result must equal the
``input_token_checksum`` that ``benchmarks/bench_e2e_flame.py`` writes into each result JSON.
"""

import struct
import sys

import numpy as np

CIDX_MAGIC = 0x58444943  # b"CIDX", see jla.data.cidx
INT32_CODE = 4


def main() -> int:
    prefix, global_batch, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    with open(prefix + ".idx", "rb") as f:
        magic, _version, code, ndocs, _ntok = struct.unpack("<IIBqq", f.read(25))
        offsets = np.frombuffer(f.read((ndocs + 1) * 8), "<i8")
    if magic != CIDX_MAGIC or code != INT32_CODE or not np.all(np.diff(offsets) == 4096):
        raise SystemExit("expected a CIDX int32 corpus of 4096-token records")
    data = np.memmap(prefix + ".bin", dtype="<i4", mode="r")
    order = np.arange(ndocs, dtype=np.int64)
    np.random.RandomState(seed).shuffle(order)
    total = 0
    for s in range(steps):
        for r in order[s * global_batch:(s + 1) * global_batch]:
            total += int(data[offsets[r]:offsets[r] + 4096].astype(np.int64).sum())
    print(f"records={ndocs} global_batch={global_batch} steps={steps} seed={seed} checksum={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
