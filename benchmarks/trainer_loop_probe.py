# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host timeline of the trainer loop: when each step call, batch fetch and metrics read starts and returns.

    python benchmarks/trainer_loop_probe.py <benchmarks/bench_e2e_1p3b.py arguments...>

Runs ``benchmarks/bench_e2e_1p3b.py`` in this process with the train step's call
(:class:`jla.training.step.TrainStep`), the device prefetcher's ``__next__`` and ``jax.device_get`` (the trainer's
metrics read) wrapped in timers, and prints the last 60 events as ``PROBE <event> start <s> end <s> dur <ms>``
(seconds from the start of the probe). A step call that lasts about one step period means the host cannot dispatch
ahead of the devices (``benchmarks/multi_gpu.md``, ``train.dispatch_ahead`` on 8 GPUs).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    import jax

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_e2e_1p3b

    from jla.data import pipeline
    from jla.training import step as step_module

    start_time = time.perf_counter()
    events: list[tuple[str, float, float]] = []

    def wrap(owner, name: str, label: str) -> None:
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            begin = time.perf_counter()
            out = original(*args, **kwargs)
            events.append((label, begin - start_time, time.perf_counter() - start_time))
            return out

        setattr(owner, name, timed)

    wrap(step_module.TrainStep, "__call__", "step_call")
    wrap(pipeline.DevicePrefetcher, "__next__", "next_batch")
    wrap(jax, "device_get", "device_get")
    try:
        return bench_e2e_1p3b.main(sys.argv[1:])
    finally:
        print("PROBE_EVENTS", len(events))
        for label, begin, end in events[-60:]:
            print(f"PROBE {label:10s} start {begin:9.3f} end {end:9.3f} dur {1e3 * (end - begin):8.1f} ms")


if __name__ == "__main__":  # data-loader workers re-import this module (spawn)
    raise SystemExit(main())
