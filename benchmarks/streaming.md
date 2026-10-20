# Streaming data throughput (`data.source = "stream"`)

Host-side throughput of the streamed data pipeline, which tokenizes `.jsonl` or `.parquet` text on the fly, on
one thread of the training process (the device-prefetch thread runs it during training). It is compared with the
pre-tokenized path (`jla tokenize` once, then `TokenWindowDataset`) as an upper bound, with Grain (MaxText's
recipe) and with Hugging Face `datasets` streaming.

## Setup

- The host CPU of an 8x H100 machine; no GPU work. Batches of 8 rows x 2048 tokens with document masking, from a
  26 MiB synthetic JSONL corpus (6000 documents, Zipf words) and the same texts as parquet.
- `bpe` is a byte-level BPE (8192 tokens) trained on the corpus and loaded as a Hugging Face fast tokenizer;
  `byte` is UTF-8 bytes. `targets/s` counts trained targets (no EOS-position or padding targets), so packings
  compare fairly.
- Baselines: Grain's `ConcatThenSplitIterDataset` and `FirstFitPackIterDataset` as MaxText uses them, with the
  JSONL lines held in memory (file reads not timed) and, for first-fit, each document trimmed to `seq_len` as
  MaxText's `TokenizeAndTrim` does; `datasets` streaming `.map(batched=True)` with `group_texts` (no
  `segment_ids` or positions).
- Provisional: measured on 17 Oct 2026 while about 25 other benchmark jobs shared the CPUs; the BPE tokenizer is
  multi-threaded, so the BPE rows carry about 2x noise.

## Provisional results

M tokens/s / M targets/s:

| pipeline | byte | bpe |
|---|---|---|
| pre-tokenized CIDX, upper bound | 77.6 / 77.6 | 86.2 / 86.1 |
| stream jsonl, concat_split, read_ahead 64 | 53.4 / 53.3 | 1.54 / 1.54 |
| stream jsonl, concat_split, read_ahead 1024 (default) | 53.6 / 53.6 | 2.94 / 2.94 |
| stream jsonl, first_fit, read_ahead 64 | 57.4 / 53.3 | 1.57 / 1.45 |
| stream jsonl, first_fit, read_ahead 1024 (default) | 56.0 / 52.0 | 3.60 / 3.31 |
| stream parquet, concat_split, read_ahead 1024 | 51.6 / 51.5 | 2.31 / 2.30 |
| stream parquet, first_fit, read_ahead 1024 | 56.0 / 52.0 | 1.95 / 1.79 |
| Grain (MaxText recipe), ConcatThenSplitIterDataset | 7.61 / 7.61 | 1.16 / 1.16 |
| Grain (MaxText recipe), FirstFitPackIterDataset | 3.16 / 2.90 | 1.09 / 1.00 |
| Hugging Face `datasets` streaming + `group_texts` | | 2.72 / 2.72 |

With bytes the stream reaches about 70% of the pre-tokenized path and runs 7-18x faster than Grain's per-record
pipeline. With BPE the tokenizer dominates: batching 1024 documents per tokenizer call (the default
`data.stream_read_ahead`) is about 2x faster than 64; on jsonl it is then 1.1-1.3x the `datasets` recipe and
2.5-3.3x Grain. The stream splits long documents instead of trimming them, so it keeps every target.

## Where we lose, and limits

- A single-threaded BPE stream does not keep up with training: at 40% MFU (about 400 dense bf16 TFLOP/s) and
  6 x 125M FLOPs per token, a 125M model trains at roughly 0.5M tokens/s per H100, about 4M tokens/s for one
  8-GPU host. Pre-tokenize (`jla tokenize`) for small models at full speed, and stream for larger models, quick
  experiments or data that changes.

## Correctness

The stream gives the pre-tokenized path's batches (`tests/test_data_streaming.py`: byte and Hugging Face
tokenizer streams equal the pre-tokenized path, parquet and `datasets` sources equal JSONL, first-fit matches
Grain's `FirstFitPackIterDataset`, `read_ahead` does not change the stream, and a resume mid-stream reproduces
the next batches).

## Reproduce

```bash
python benchmarks/bench_streaming.py --workdir /path/to/scratch --seconds 4 --read-ahead 64,256,1024 \
  --out results/streaming.json
pytest tests/test_data_streaming.py tests/test_trainer_stream.py
```
