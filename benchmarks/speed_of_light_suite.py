# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run several speed-of-light benchmarks back to back on one GPU (benchmarks/profile_1p3b.md).

    python benchmarks/speed_of_light_suite.py "bench_speed_of_light.py clock --seconds 60 --json clock.json" \
        "speed_of_light_prototypes.py --arch gdn-hd128-1p3b --data DATA --json proto.json"

Each argument is one command line of a script in ``benchmarks/``, run with this interpreter in its own
process, one after the other, so that one reservation of the GPU covers all of them. A command that
fails is reported and the next one still runs; the exit status is the number of failed commands.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    commands = sys.argv[1:] if argv is None else argv
    failed = 0
    for command in commands:
        args = shlex.split(command)
        start = time.time()
        print(f"[suite] start {args[0]}", flush=True)
        rc = subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]]).returncode
        print(f"[suite] {args[0]} exit {rc} after {time.time() - start:.0f} s", flush=True)
        failed += rc != 0
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
