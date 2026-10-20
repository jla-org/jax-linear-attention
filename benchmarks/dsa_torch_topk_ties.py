# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which keys does ``torch.topk`` on CUDA keep among tied scores? (DeepSeek-V3.2's selection.)

DeepSeek-V3.2's ``Indexer.forward`` selects with ``index_score.topk(min(index_topk, end_pos), dim=-1)[1]``
on float32 scores with ``-inf`` on masked keys; torch does not document the order among equal values.
This compares its key SET with the rule of our radix-select kernel and ``lax.top_k`` (the lower column
first among equal scores) on rows of heavily tied integer scores, all-equal rows and causal ``-inf``
suffixes, for ``k = 2048`` (and 64) at widths 2053 to 65536. Untimed; needs a CUDA GPU and torch.

    python benchmarks/dsa_torch_topk_ties.py benchmarks/results/dsa_torch_topk_ties.json
"""

import json
import sys

import torch


def main() -> None:
    torch.manual_seed(0)
    dev = "cuda"
    rows = []
    for width, k in [(2053, 2048), (4096, 2048), (16384, 2048), (65536, 2048), (4096, 64)]:
        for lo, hi in [(-4, 5), (0, 2), (0, 1)]:
            s = torch.randint(lo, hi, (64, width), device=dev).float()
            # half of the rows: a causal prefix (masked keys -inf), as the indexer's scores
            t = torch.arange(width, device=dev)[None, :]
            limit = torch.randint(k, width + 1, (64, 1), device=dev)
            s = torch.where((torch.arange(64, device=dev)[:, None] % 2 == 1) & (t >= limit), float("-inf"), s)
            idx = s.topk(k, dim=-1)[1]
            lower_first = torch.sort(-s, dim=-1, stable=True).indices[:, :k]
            a = torch.sort(idx, dim=-1).values
            b = torch.sort(lower_first, dim=-1).values
            differ = (a != b).any(-1)
            rows.append(dict(width=width, k=k, scores=f"integers in [{lo}, {hi})", rows=64,
                             rows_matching_lower_column_first=int((~differ).sum().item())))
            print(rows[-1], flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(dict(torch=torch.__version__, device=torch.cuda.get_device_name(), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
