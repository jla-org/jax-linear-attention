# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flame baseline's data adapter (benchmarks/bench_e2e_flame.py) against the jla data pipeline.

Needs torch (the flame environment); skipped in the JAX environment. Checks that ``CidxBatches`` hands each rank
the jla BatchLoader's rows, that its labels drop exactly the targets jla's document masking drops (EOS targets)
plus the one target per record that crosses into the next record, and that its ``cu_seqlens`` and the position
ids flame derives from them equal jla's ``segment_ids`` and ``positions``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jla.data.cidx import write_cidx  # noqa: E402
from jla.data.datasets import TokenWindowDataset  # noqa: E402
from jla.data.loader import BatchLoader  # noqa: E402
from jla.data.packing import DocumentFields  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from bench_e2e_flame import CidxBatches  # noqa: E402

EOS = 7
T = 16


def _corpus(tmp_path: Path, records: int = 24) -> str:
    rng = np.random.default_rng(0)
    rows = rng.integers(8, 50, size=(records, T))
    rows[rng.random((records, T)) < 0.12] = EOS
    rows[0, 0] = rows[1, T - 1] = rows[2, 3] = rows[2, 4] = EOS  # EOS at a record start, at its end, twice in a row
    write_cidx(tmp_path / "c", list(rows))
    return str(tmp_path / "c")


@pytest.fixture(autouse=True)
def _no_cuda_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """CidxBatches syncs the GPU before its first batch (the timing origin); these tests run on the CPU."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a, **_kw: None)


def _global_rows(prefix: str, batch: int, steps: int) -> list[dict[str, np.ndarray]]:
    loader = BatchLoader(TokenWindowDataset(prefix, T, exact_packed_records=True), batch, seed=0, shuffle=True)
    return [next(loader) for _ in range(steps)]


@pytest.mark.parametrize("world", [1, 2])
def test_varlen_batches_match_jla_documents(tmp_path: Path, world: int) -> None:
    prefix, rows, steps = _corpus(tmp_path), 3, 3
    reference = _global_rows(prefix, rows * world, steps)
    fields = DocumentFields(EOS)
    for rank in range(world):
        batches = CidxBatches(prefix, seq_len=T, rows=rows, rank=rank, world=world, seed=0, shuffle=True,
                              eos_id=EOS, varlen=True)
        checksum = 0
        for step in range(steps):
            got = next(batches)
            ref = fields({k: v[rank * rows:(rank + 1) * rows] for k, v in reference[step].items()})
            tokens = ref["input_ids"]
            checksum += int(tokens.sum())
            assert got["input_ids"].shape == (1, rows * T)
            np.testing.assert_array_equal(got["input_ids"].numpy().reshape(rows, T), tokens)
            # FLA's target at t is labels[t + 1]: jla's loss mask (no EOS targets) minus each record's last target.
            labels = got["labels"].numpy().reshape(rows, T)
            kept = np.zeros((rows, T), dtype=bool)
            kept[:, :-1] = labels[:, 1:] != -100
            np.testing.assert_array_equal(kept, ref["loss_mask"] & (np.arange(T) < T - 1))
            np.testing.assert_array_equal(labels[:, 1:][kept[:, :-1]], tokens[:, 1:][kept[:, :-1]])
            assert np.all(labels[:, 0] == -100)
            # Documents: cu_seqlens starts every segment of every row, as jla's segment_ids do.
            seg = ref["segment_ids"]
            starts = np.ones((rows, T), dtype=bool)
            starts[:, 1:] = seg[:, 1:] != seg[:, :-1]
            expected = np.concatenate([np.flatnonzero(starts.reshape(-1)), [rows * T]])
            np.testing.assert_array_equal(got["cu_seqlens"].numpy(), expected)
            assert got["cu_seqlens"].dtype == torch.int32
            fla_utils = pytest.importorskip("fla.ops.utils")
            positions = fla_utils.prepare_position_ids(got["cu_seqlens"].long())
            np.testing.assert_array_equal(positions.numpy().reshape(rows, T), ref["positions"])
        assert batches.token_checksum == checksum and batches.batches == steps


def test_dense_batches_with_accumulation(tmp_path: Path) -> None:
    prefix, rows, world, accum = _corpus(tmp_path), 4, 2, 2
    reference = _global_rows(prefix, rows * world, 2)
    for rank in range(world):
        batches = CidxBatches(prefix, seq_len=T, rows=rows, rank=rank, world=world, seed=0, shuffle=True,
                              eos_id=EOS, varlen=False, accum=accum)
        for step in range(2):
            micro = [next(batches) for _ in range(accum)]
            assert all(m["input_ids"].shape == (rows // accum, T) and "cu_seqlens" not in m for m in micro)
            tokens = torch.cat([m["input_ids"] for m in micro]).numpy()
            np.testing.assert_array_equal(tokens, reference[step]["input_ids"][rank * rows:(rank + 1) * rows])
            labels = torch.cat([m["labels"] for m in micro]).numpy()
            after_eos = np.zeros_like(tokens, dtype=bool)
            after_eos[:, 1:] = tokens[:, :-1] == EOS
            np.testing.assert_array_equal(labels == -100, after_eos | (np.arange(T) == 0))
        assert batches.batches == 2


def test_rejects_uneven_accumulation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="micro-batches"):
        CidxBatches(_corpus(tmp_path), seq_len=T, rows=3, rank=0, world=1, seed=0, shuffle=True, eos_id=EOS,
                    varlen=False, accum=2)
