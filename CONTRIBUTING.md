# Contributing

Bug reports, fixes, kernels and ports are welcome. For anything larger than a fix, please open an
issue first to agree on the approach.

## Setup

Python 3.12 or newer and JAX 0.11.2 (pinned):

```bash
git clone https://github.com/jla-org/jax-linear-attention.git && cd jax-linear-attention
pip install -e ".[cuda,test]"   # Linux + NVIDIA Hopper GPU (CUDA 13 wheels)
pip install -e ".[test]"        # CPU only
```

## Tests

All tests run on CPU except the GPU-only kernel and cuDNN tests, which skip there:

```bash
pytest tests/                                          # CPU machine: the whole CPU suite
JAX_PLATFORMS=cpu pytest tests/                        # GPU host: CPU only, plus the kernels' CPU interpreter tests
pytest tests/                                          # H100: also the Mosaic GPU kernel and cuDNN tests
pytest tests/test_long_context.py --run-long-context   # opt-in, one H100 (calls past 2^31 elements)
pytest tests/ --run-slow                               # opt-in, H100: also the slow GPU kernel cases (minutes each),
                                                       # such as the whole-op checks in tests/test_vector_gate_bwd_schedule.py
```

Tests that need an optional extra (`transformers`, `lm-eval`, `wandb`, ...) skip when it is not
installed, and multi-GPU tests skip on fewer GPUs. Please run the CPU suite before opening a pull
request, and say which GPU tests you ran. CI runs a fast set of the CPU tests on every pull request
and push to `main` (about 10 minutes; the list is in `.github/workflows/ci.yml`), and the whole CPU
suite nightly and on demand.

## Mosaic GPU kernels

- Kernels target H100 (sm_90a) with JAX 0.11.2. The Mosaic GPU APIs change with every JAX release,
  so the JAX version is pinned and kernels are written against it.
- Every recurrence ships as a token-by-token reference (the oracle), a chunk-parallel JAX version
  (`chunked`) and, where available, a Mosaic GPU kernel.
- A kernel registers a strict `supports` check (`jla.ops.register`). `impl="auto"` uses the kernel
  only for the calls its GPU tests cover; every other call runs the chunked op.
- `segment_ids [B, T]` restarts the state at every document boundary, with the same semantics in
  every implementation.
- Algorithms are exact: fp32 states and accumulation, no approximate gates. `exp` and similar
  functions may use Mosaic GPU's `approx_math`.
- Test a kernel against the reference and, where one exists, the latest upstream code, with
  packed documents, grouped heads, initial and final states and decode parity. GPU tests skip
  off Hopper.
- Benchmarks follow [benchmarks/README.md](benchmarks/README.md): GPU kernel time per call,
  interleaved with the baseline on one GPU, with the baseline's commit recorded.
- Minimal reproductions of Mosaic GPU issues go in `benchmarks/mosaic_gpu_repros/`.

## Adapted code and licences

The project is Apache-2.0. Code adapted from another project names that project and its source
files in a header comment, the project is listed in `NOTICE`, and its licence text goes in
`licenses/`. New source files start with a copyright line and
`# SPDX-License-Identifier: Apache-2.0`, as the existing files do.

## Sign-off (DCO)

Contributions are accepted under the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
Sign off every commit:

```bash
git commit -s
```

This adds a `Signed-off-by: Your Name <your email>` line, which certifies that you wrote the
change or have the right to submit it under the project's licence. Commits without a sign-off
cannot be merged.
