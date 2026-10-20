# End-to-end training at 125M on one H100

Six 125M-scale models train through `jla train` for 300 steps on FineWeb-Edu with the Mosaic GPU kernels. The
page reports the trainer's throughput, MFU and memory, and checks that the kernels train to the same loss as the
plain-JAX chunked ops. Where the step time goes is on [profile_1p3b.md](profile_1p3b.md).

## Setup

- H100 80GB HBM3, JAX 0.11.2, bf16 compute.
- Each shipped preset with `model.vocab_size = 128256` (the Llama-3 tokenizer; the tied embedding holds 98.5M of
  each model's parameters). AdamW with a WSD schedule (lr 6e-4, warmup 100), 300 steps, micro batch 8 × 2048
  tokens. The configs are in [`end_to_end/`](end_to_end/), with the data path as a placeholder.
- Data: a FineWeb-Edu export (3B tokens) in the CIDX format that `jla tokenize` writes.
- tok/s, ms/step, MFU and HFU come from the trainer's step-300 log line; the eval loss from its step-300 evaluation
  (8 batches, 131,072 held-out tokens). Device memory is the compiled step's estimate (`compile_preflight.json`).
- Measured on 16 Oct 2026, all six on the same H100 within 45 minutes. Other jobs could share the machine, so the
  numbers are provisional.

## Results

Provisional, one H100, step 300 of each run:

| model | parameters | tok/s | ms/step | MFU | HFU | device memory | eval loss | eval ppl |
|---|---|---|---|---|---|---|---|---|
| transformer-125m | 181.1M | 206,945 | 79 | 25.1% | 29.4% | 3.67 GiB | 6.368 | 583 |
| gdn-hybrid-125m | 190.1M | 197,641 | 83 | 23.9% | 27.9% | 5.37 GiB | 6.074 | 434 |
| gdn2-hybrid-125m | 190.6M | 177,571 | 92 | 21.8% | 25.5% | 4.96 GiB | 6.055 | 426 |
| kda-hybrid-125m | 184.8M | 163,361 | 100 | 19.1% | 22.2% | 6.38 GiB | 6.111 | 451 |
| mamba3-130m | 190.2M | 149,770 | 109 | 19.2% | 22.6% | 4.52 GiB | 5.919 | 372 |
| mamba2-130m | 188.9M | 140,241 | 117 | 17.2% | 20.1% | 4.66 GiB | 5.947 | 383 |

## Limits

- At 300 steps and 2.5M tokens the losses only show that every model trains; they do not compare architectures.
- The chunked-op runs shared their GPUs with other training jobs, so their throughput is not compared here.

## Correctness

The same configs with `implementation = "chunked"` (the plain-JAX ops) end within 0.01 of these eval losses.
Evaluated on the same batches with both implementations, the step-300 checkpoints of the kernel runs give the same
loss ([`end_to_end/compare_impls.py`](end_to_end/compare_impls.py)):

| model | eval loss, chunked | eval loss, kernels | final hidden state, max / mean relative difference |
|---|---|---|---|
| kda-hybrid-125m | 6.11114 | 6.11114 | 1.4e-2 / 3.7e-3 |
| gdn-hybrid-125m | 6.07360 | 6.07361 | 1.2e-2 / 4.3e-3 |

## Reproduce

```bash
jla tokenize --input fineweb_edu.jsonl --output /path/to/fineweb_edu --tokenizer meta-llama/Llama-3.2-1B
for m in transformer-125m gdn-hybrid-125m gdn2-hybrid-125m kda-hybrid-125m mamba3-130m mamba2-130m; do
  jla train --config benchmarks/end_to_end/fw-$m.toml --set 'data.paths=["/path/to/fineweb_edu"]'
done
python benchmarks/end_to_end/compare_impls.py benchmarks/end_to_end/fw-kda-hybrid-125m.toml \
    runs/fw-kda-hybrid-125m/checkpoints/step_00000300
```
