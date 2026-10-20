# SFT packing: row fill and host throughput

`benchmarks/bench_sft_packing.py` measures the host side of `data.source = "sft"` (`jla.data.sft`): how full
best-fit-decreasing packing (`data.sft_packing = "bfd"`) fills the rows, against one example per row
(`"single"`) and against the lower bound on the row count, and how fast one loader process builds rows.

## Setup

- The host CPU of an H100 machine, Python 3.12, NumPy 2.5; no GPU work.
- Example lengths are lognormal (median 600 tokens, sigma 0.9: chat-like, with a long tail), clipped to the row.
- Timing: one run of each configuration, wall clock. Provisional: measured on 17 Oct 2026 while other jobs
  shared the host.

## Provisional results

Best-fit decreasing on 1M examples:

| seq_len | bfd rows | lower bound `ceil(sum / seq_len)` | bfd fill | single fill | bfd time |
|---:|---:|---:|---:|---:|---:|
| 2048 | 384,196 | 384,062 | 0.99965 | 0.38406 | 1.03 s |
| 4096 | 212,113 | 212,087 | 0.99987 | 0.21209 | 0.97 s |
| 8192 | 109,144 | 109,138 | 0.99994 | 0.10914 | 0.92 s |

`SftDataset` on a 50k-example corpus (44.7M tokens, seq_len 4096; 609 examples longer than 4097 tokens are
truncated):

| packing | rows | fill | plan time | rows/s (one process) | row tokens/s |
|---|---:|---:|---:|---:|---:|
| bfd | 10,392 | 0.99985 | 0.18 s | 21,826 | 89.4M |
| single | 49,794 | 0.20867 | 0.15 s | 43,321 | 177.4M |

BFD stays within 0.04% of the lower bound on the row count, so packing costs almost no padding; one example per
row would waste 62-89% of every batch on this length mix.

## Where we lose, and limits

- Packing builds fewer row tokens per second than one example per row (89.4M against 177.4M in one process),
  but still far more than a training step consumes, so it does not limit training speed.

## Correctness

Packing whole examples with per-example `segment_ids` and positions is exact for dense models: packed and
unpacked rows give the same loss and gradients
(`tests/test_trainer_sft.py::test_packed_and_unpacked_rows_give_the_same_loss_and_gradients`), and the rows hold
every example exactly once (`tests/test_data_sft.py`).

## Reproduce

```bash
python benchmarks/bench_sft_packing.py --out results/sft_packing.json
pytest tests/test_data_sft.py tests/test_trainer_sft.py
```
