# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flame baseline's result summary (benchmarks/bench_e2e_flame.py), without torch or a GPU.

Rebuilds a result JSON with ``--from-raw`` from a hand-written dump folder and checks the summary's fields:
the timed-steps peak memory (warm-up excluded), the logged FLA backend dispatch, flame's skipped
optimizer steps failing the run, every commit of the environment file, and tree-relative paths only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_e2e_flame as bench  # noqa: E402

LOG = """\
[titan] 2026-10-17 12:00:00,000 - root - INFO - building model
[FLA Backend] gated_delta_rule.chunk_gated_delta_rule -> flash_qla rejected: FlashQLA requires K=128, got 256
[rank0]: [FLA Backend] common.chunk_bwd_dqkwg -> tilelang
[FLA Backend] common.chunk_bwd_dqkwg -> tilelang
"""


def _dump(tmp_path: Path, *, skipped: int = 0) -> Path:
    dump = tmp_path / "dump"
    dump.mkdir()
    records = [{"step": s, "end": 10.0 + 2.0 * s, "loss": 12.0 - 0.1 * s, "grad_norm": 1.0,
                "skipped_steps": skipped if s == 3 else 0} for s in (1, 2, 3)]
    raw = {
        "records": records, "loop_start": 10.0, "batches_drawn_per_rank": 3, "input_token_checksum_all_ranks": 5,
        "peak_allocated_gib_max_rank": 70.0, "peak_reserved_gib_max_rank": 72.0,
        "per_step_peak_gib_max_rank": [[70.0, 72.0], [50.0, 60.0], [55.0, 61.0]],  # step 1 = warm-up peak
        "params": 10, "embedding_params": 4, "flame_flops_per_token": 60.0, "device_name": "test",
        "flame_argv": ["--job.config_file", str(bench.FLAME_DIR / "job.toml"), "--job.dump_folder", str(dump),
                       "--model.config", "/elsewhere/checkout/benchmarks/end_to_end/1p3b/flame/kda-1p3b.json"],
        "varlen": True,
    }
    (dump / "bench_raw.json").write_text(json.dumps(raw))
    (dump / "rank0.log").write_text(LOG)
    return dump


def _summarise(tmp_path: Path, dump: Path) -> tuple[int, dict]:
    out = tmp_path / "out.json"
    rc = bench.main(["--arch", "kda-1p3b", "--data", "/data/fineweb_edu_3B", "--steps", "3", "--skip", "1",
                     "--micro-batch", "2", "--dump-dir", str(dump), "--from-raw", "--json", str(out)])
    return rc, json.loads(out.read_text())


def test_summary_fields(tmp_path: Path) -> None:
    rc, d = _summarise(tmp_path, _dump(tmp_path))
    assert rc == 0
    assert d["peak_memory_allocated_gib_max_gpu"] == 55.0  # timed steps 2-3 only
    assert d["peak_memory_reserved_gib_max_gpu"] == 61.0
    assert d["peak_memory_allocated_gib_max_gpu_incl_warmup"] == 70.0
    assert [s["peak_allocated_gib"] for s in d["per_step"]] == [70.0, 50.0, 55.0]
    assert d["timed_steps"] == 2 and d["step_time_mean_s"] == pytest.approx(2.0)
    dispatch = d["fla_backend_dispatch"]
    assert dispatch["served"] == {"common.chunk_bwd_dqkwg": "tilelang"}
    assert dispatch["rejected"] == {"gated_delta_rule.chunk_gated_delta_rule": "flash_qla: FlashQLA requires K=128, got 256"}
    assert len(dispatch["log_lines"]) == 2  # deduplicated
    assert d["optimizer_skipped_steps"] == 0 and d["grad_norm_finite"]
    text = json.dumps(d)
    assert str(tmp_path) not in text and str(bench.HERE.parent) not in text
    assert d["flame_argv"][1] == "benchmarks/end_to_end/1p3b/flame/job.toml"
    assert d["flame_argv"][3] == "<dump-dir>"
    assert d["flame_argv"][5] == "benchmarks/end_to_end/1p3b/flame/kda-1p3b.json"


def test_skipped_optimizer_step_fails_the_run(tmp_path: Path) -> None:
    rc, d = _summarise(tmp_path, _dump(tmp_path, skipped=1))
    assert rc == 1
    assert d["optimizer_skipped_steps"] == 1


def test_versions_keep_every_commit_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "FLAME_ENV_COMMITS.txt").write_text(
        "fla c6e4f00 (/src/fla)\nflashqla 5538c35 (/src/qla)\nflashkda 7afb9f4 (https://x) wheel ab\n"
        "flame 24dea91 (/src/flame)\ntorchtitan 0b44d4c (/src/tt)\n"
        "flash-attention 965c921 (Dao-AILab/flash-attention hopper/, /src/fa3) wheel cd\n"
        "mamba e9594ce (state-spaces/mamba main) wheel ef\ncausal-conv1d 1.7.0 (PyPI sdist)\n"
        "-- freeze (2026-10-17)\ntorch==2.13.0+cu129\n")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert bench._versions()["commits"] == ["fla c6e4f00", "flashqla 5538c35", "flashkda 7afb9f4", "flame 24dea91",
                                            "torchtitan 0b44d4c", "flash-attention 965c921", "mamba e9594ce",
                                            "causal-conv1d 1.7.0"]
