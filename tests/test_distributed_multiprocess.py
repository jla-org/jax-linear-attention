# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two ``jax.distributed`` CPU processes train like one process and their checkpoint resumes on one.

Each worker has 2 CPU devices (4 global, fsdp=4) and Gloo collectives. A third process trains the
same run on 4 local devices and then resumes the workers' per-process sharded checkpoint on one
device. Skipped when no TCP address of this host accepts connections (some sandboxes block loopback).
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

COMMON = textwrap.dedent(
    r"""
    import json
    import sys

    import jax

    from _toy_model import experiment
    from jla.models import get_preset
    from jla.training.trainer import Trainer

    def config(out, max_steps, fsdp, **train):
        # Global batch 8 on any layout.
        model = get_preset("tiny").replace(compute_dtype="float32")
        train = {"max_steps": max_steps, "log_every": 1, "micro_batch_per_replica": 8 // fsdp, "grad_accum": 1,
                 "final_eval": False, "ce_compute_dtype": "float32", "checkpoint_format": "sharded", **train}
        optim = {"name": "sgd", "lr": 0.5, "b1": 0.9, "warmup_steps": 0, "schedule_steps": 6}
        return experiment(out, model=model, train=train, optim=optim, mesh={"fsdp": fsdp})
    """
)

WORKER = COMMON + textwrap.dedent(
    r"""
    coordinator, rank, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2, process_id=rank,
                               initialization_timeout=120)
    assert jax.process_count() == 2 and len(jax.devices()) == 4 and len(jax.local_devices()) == 2
    result = Trainer(config(out, 4, 4)).run()
    print("WORKER " + json.dumps({"rank": rank, "losses": [r["loss"] for r in result.history]}), flush=True)
    jax.distributed.shutdown()
    """
)

REFERENCE = COMMON + textwrap.dedent(
    r"""
    import shutil

    out = sys.argv[1]
    assert len(jax.devices()) == 4
    full = Trainer(config(f"{out}/ref", 6, 4)).run()
    shutil.copytree(f"{out}/mp/checkpoints", f"{out}/resume/checkpoints")
    resumed = Trainer(config(f"{out}/resume", 6, 1, resume="required"), devices=jax.devices()[:1]).run()
    print("REFERENCE " + json.dumps({"losses": [r["loss"] for r in full.history],
                                     "resumed": [r["loss"] for r in resumed.history]}), flush=True)
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


def _env(devices: int) -> dict[str, str]:
    return {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"))),
    }


def _payload(stdout: str, marker: str) -> dict:
    return json.loads(stdout.split(marker + " ", 1)[1].splitlines()[0])


def test_two_processes_match_one_process_and_resume_on_one_device(tmp_path: Path) -> None:
    host = _reachable_address()
    if host is None:
        pytest.skip("no local TCP address accepts connections")
    coordinator = f"{host}:{_free_port(host)}"
    (tmp_path / "worker.py").write_text(WORKER)
    (tmp_path / "reference.py").write_text(REFERENCE)
    # Output goes to files (an undrained pipe could block a worker mid-collective); both workers are
    # killed if either hangs or fails, so no worker or coordinator port outlives the test.
    logs = [(tmp_path / f"worker{rank}.out", tmp_path / f"worker{rank}.err") for rank in range(2)]
    workers: list[subprocess.Popen] = []
    try:
        for rank, (out, err) in enumerate(logs):
            with out.open("w") as stdout, err.open("w") as stderr:
                workers.append(subprocess.Popen(
                    [sys.executable, str(tmp_path / "worker.py"), coordinator, str(rank), str(tmp_path / "mp")],
                    env=_env(2), stdout=stdout, stderr=stderr, text=True))
        deadline = time.monotonic() + 600
        while any(proc.poll() is None for proc in workers) and time.monotonic() < deadline:
            if any(proc.poll() not in (None, 0) for proc in workers):
                break  # one worker failed: the other would wait for it forever
            time.sleep(0.5)
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    outputs = [(out.read_text(), err.read_text()) for out, err in logs]
    if any(proc.returncode != 0 for proc in workers):
        pytest.fail("\n".join(
            f"worker {rank} exit {proc.returncode} (-9: killed after a timeout or the other worker's failure)\n"
            f"stdout:\n{stdout[-3000:]}\nstderr:\n{stderr[-6000:]}"
            for rank, (proc, (stdout, stderr)) in enumerate(zip(workers, outputs))
        ))
    worker_losses = [_payload(stdout, "WORKER")["losses"] for stdout, _ in outputs]
    assert worker_losses[0] == worker_losses[1]
    manifest = json.loads((tmp_path / "mp/checkpoints/step_00000004/manifest.json").read_text())
    assert manifest["payload_format"] == "sharded" and manifest["process_count"] == 2
    assert len(list((tmp_path / "mp/checkpoints/step_00000004").glob("model.p*-of-00002.safetensors"))) == 2

    ref = subprocess.run([sys.executable, str(tmp_path / "reference.py"), str(tmp_path)], env=_env(4),
                         capture_output=True, text=True, timeout=600, check=False)
    assert ref.returncode == 0, f"stdout:\n{ref.stdout[-3000:]}\nstderr:\n{ref.stderr[-6000:]}"
    reference = _payload(ref.stdout, "REFERENCE")
    # Gloo and the in-process reduction may sum the same partial gradients in another order.
    assert worker_losses[0] == pytest.approx(reference["losses"][:4], rel=1e-5)
    assert reference["resumed"] == pytest.approx(reference["losses"][4:], rel=1e-5)
