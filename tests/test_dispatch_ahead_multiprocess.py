# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``train.dispatch_ahead`` with two ``jax.distributed`` CPU processes: the hosts agree on a stop without the devices.

After dispatching a step ahead the trainer checks for a stop request with
:func:`jla.distributed.init.process_any_host` (the distributed runtime's key-value store) instead of a device
collective that would wait for the dispatched step. Two workers (2 CPU devices each, fsdp = 4, Gloo collectives)
check the OR of different flags, train 4 steps with and without dispatching ahead (the same losses) and stop on a
time limit at the same step. Skipped when no TCP address of this host accepts connections.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

WORKER = textwrap.dedent(
    r"""
    import json
    import sys

    import jax

    from _toy_model import experiment
    from jla.distributed.init import process_any_host
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    def config(out, max_steps, **train):
        model = get_preset("tiny").replace(compute_dtype="float32")
        train = {"max_steps": max_steps, "log_every": 1, "micro_batch_per_replica": 2, "grad_accum": 1,
                 "final_eval": False, "save_final": False, "ce_compute_dtype": "float32",
                 "checkpoint_format": "sharded", **train}
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0, "schedule_steps": 6}
        return experiment(out, model=model, train=train, optim=optim, mesh={"fsdp": 4})

    coordinator, rank, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2, process_id=rank,
                               initialization_timeout=120)
    assert jax.process_count() == 2 and len(jax.devices()) == 4
    flags = [process_any_host(rank == 1), process_any_host(False), process_any_host(rank == 0), process_any_host(True)]
    plain = Trainer(config(f"{out}/plain", 4)).run()
    ahead = Trainer(config(f"{out}/ahead", 4, dispatch_ahead=True)).run()
    limited = Trainer(config(f"{out}/limit", 6, dispatch_ahead=True, time_limit_minutes=1e-9)).run()
    print("WORKER " + json.dumps({
        "rank": rank, "flags": flags,
        "plain": [r["loss"] for r in plain.history], "ahead": [r["loss"] for r in ahead.history],
        "limit_step": limited.step, "limit_reason": limited.stop_reason,
    }), flush=True)
    jax.distributed.shutdown()
    """
)


def _reachable_address() -> str | None:
    """A local IPv4 address that accepts TCP connections (loopback first)."""
    candidates = ["127.0.0.1"]
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # no packet is sent; picks the outbound interface
        candidates.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    for host in candidates:
        server = socket.socket()
        try:
            server.bind((host, 0))
            server.listen(1)
            with socket.create_connection((host, server.getsockname()[1]), timeout=2):
                return host
        except OSError:
            continue
        finally:
            server.close()
    return None


def _free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def test_two_processes_dispatch_ahead_and_agree_on_a_stop(tmp_path: Path) -> None:
    host = _reachable_address()
    if host is None:
        pytest.skip("no local TCP address accepts connections")
    coordinator = f"{host}:{_free_port(host)}"
    (tmp_path / "worker.py").write_text(WORKER)
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }
    # Output goes to files (an undrained pipe could block a worker mid-collective); both workers are killed if
    # either hangs or fails.
    logs = [(tmp_path / f"worker{rank}.out", tmp_path / f"worker{rank}.err") for rank in range(2)]
    workers: list[subprocess.Popen] = []
    try:
        for rank, (out, err) in enumerate(logs):
            with out.open("w") as stdout, err.open("w") as stderr:
                workers.append(subprocess.Popen(
                    [sys.executable, str(tmp_path / "worker.py"), coordinator, str(rank), str(tmp_path / "mp")],
                    env=env, stdout=stdout, stderr=stderr, text=True))
        deadline = time.monotonic() + 600
        while any(proc.poll() is None for proc in workers) and time.monotonic() < deadline:
            if any(proc.poll() not in (None, 0) for proc in workers):
                break
            time.sleep(0.5)
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    outputs = [(out.read_text(), err.read_text()) for out, err in logs]
    if any(proc.returncode != 0 for proc in workers):
        pytest.fail("\n".join(
            f"worker {rank} exit {proc.returncode}\nstdout:\n{stdout[-3000:]}\nstderr:\n{stderr[-6000:]}"
            for rank, (proc, (stdout, stderr)) in enumerate(zip(workers, outputs))
        ))
    results = [json.loads(stdout.split("WORKER ", 1)[1].splitlines()[0]) for stdout, _ in outputs]
    for result in results:
        assert result["flags"] == [True, False, True, True]
        assert len(result["ahead"]) == 4
        assert result["ahead"] == pytest.approx(result["plain"], rel=1e-6)
    assert results[0]["ahead"] == results[1]["ahead"]
    # The limit is due at step 1: the step dispatched ahead runs too, and both processes stop after it.
    assert results[0]["limit_step"] == results[1]["limit_step"] == 2
    assert {r["limit_reason"] for r in results} <= {"time_limit", "stop requested by another process"}
