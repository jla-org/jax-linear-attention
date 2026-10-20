# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SFT training: packed == unpacked loss, the trainer on data.source = "sft", resume and config."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _toy_model import experiment
from jla.config import load_config
from jla.data.loader import index_batches
from jla.data.sft import SftDataset, write_sft
from jla.launch_env import DETERMINISTIC_FLAG
from jla.models import build_model, get_preset
from jla.training.step import LossConfig, lm_loss
from jla.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def _examples(seed: int, count: int, *, high: int = 40, vocab: int = 500):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = int(rng.integers(3, high + 1))
        tokens = rng.integers(0, vocab, size=n)
        mask = np.zeros(n, dtype=np.uint8)
        mask[int(rng.integers(1, n)) :] = 1  # a prompt, then a trained completion
        out.append((tokens, mask))
    return out


def _batch(dataset: SftDataset) -> dict[str, jax.Array]:
    rows = [dataset[i] for i in range(len(dataset))]
    return {key: jnp.asarray(np.stack([row[key] for row in rows])) for key in rows[0]}


@pytest.mark.parametrize("preset", ["tiny", "tiny-hybrid", "tiny-mamba2"])
def test_packed_and_unpacked_rows_give_the_same_loss_and_gradients(tmp_path: Path, preset: str) -> None:
    """Packing whole examples with segment_ids is exact for dense models: same summed CE, target
    count, MTP loss (tiny-hybrid has MTP depth 2), full training objective and its gradients.
    (MoE routers also see padding positions, so their aux loss depends on the padding per row.)"""
    write_sft(tmp_path / "s", _examples(0, 14))
    packed = SftDataset(str(tmp_path / "s"), 64, packing="bfd")
    unpacked = SftDataset(str(tmp_path / "s"), 64, packing="single")
    assert len(packed) < len(unpacked) == 14
    config = get_preset(preset).replace(compute_dtype="float32", param_dtype="float32")
    model = build_model(config, key=jax.random.key(0))
    loss_fn = lm_loss(LossConfig(compute_dtype="float32"))

    def objective(params, batch):
        total, aux = loss_fn(eqx.combine(params, static), batch, jax.random.key(1))
        return total, (aux["loss_sum"], aux["tokens"], aux["mtp_loss"])

    params, static = eqx.partition(model, eqx.is_inexact_array)
    grad_fn = jax.jit(jax.value_and_grad(objective, has_aux=True))
    with jax.default_matmul_precision("highest"):  # full float32 matmuls on GPU too (no TF32)
        (total_p, (loss_p, tokens_p, mtp_p)), grads_p = grad_fn(params, _batch(packed))
        (total_u, (loss_u, tokens_u, mtp_u)), grads_u = grad_fn(params, _batch(unpacked))
    assert float(tokens_p) == float(tokens_u) == sum(int(m[1:].sum()) for _, m in _examples(0, 14))
    np.testing.assert_allclose(float(loss_p), float(loss_u), rtol=1e-5)
    assert (float(mtp_u) > 0) == (config.mtp is not None)
    np.testing.assert_allclose(float(mtp_p), float(mtp_u), rtol=1e-5)  # MTP labels stay inside each example
    np.testing.assert_allclose(float(total_p), float(total_u), rtol=1e-5)
    flat_p, flat_u = jax.tree.leaves(grads_p), jax.tree.leaves(grads_u)
    num = np.sqrt(sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(flat_p, flat_u)))
    den = np.sqrt(sum(float(jnp.sum(b**2)) for b in flat_u))
    assert num <= 1e-4 * den, (num, den)


def _sft_experiment(
    output_dir: Path, corpus: Path, max_steps: int, *, runtime: dict | None = None, final_eval: bool = False, **data
):
    return experiment(
        str(output_dir),
        model=get_preset("tiny").replace(compute_dtype="float32"),
        train={"max_steps": max_steps, "log_every": 1, "final_eval": final_eval, "eval_batches": 2, "seq_len": 48,
               "micro_batch_per_replica": 4},
        optim={"schedule_steps": 8},
        data={"source": "sft", "paths": (str(corpus),), "synthetic_samples": 1, **data},
        runtime=runtime or {},
    )


def _host(tree) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def test_sft_train_steps_use_the_assistant_mask(tmp_path: Path) -> None:
    corpus = tmp_path / "sft"
    write_sft(corpus, _examples(1, 200))
    config = _sft_experiment(tmp_path / "run", corpus, 4)
    result = Trainer(config).run()
    assert result.stop_reason == "max_steps" and len(result.history) == 4
    # The loss sees exactly the masked targets of the rows the loader drew, step by step.
    dataset = SftDataset(str(corpus), 48)
    batch = config.train.micro_batch_per_replica * len(jax.devices())
    order = index_batches(len(dataset), batch, shuffle=True, seed=config.train.seed)
    for record in result.history:
        expected = sum(int(dataset[int(i)]["loss_mask"].sum()) for _ in range(2) for i in next(order))
        assert record["tokens"] == expected
        assert np.isfinite(record["loss"]) and record["skipped_steps"] == 0


def test_sft_holdout_evaluation_counts_only_masked_targets(tmp_path: Path) -> None:
    corpus = tmp_path / "sft"
    write_sft(corpus, _examples(4, 200))
    config = _sft_experiment(tmp_path / "run", corpus, 2, final_eval=True, eval_holdout_records=60)
    result = Trainer(config).run()
    holdout = SftDataset(str(corpus), 48, record_range=(140, 200))
    train_rows = SftDataset(str(corpus), 48, record_range=(0, 140))
    batch = config.train.micro_batch_per_replica * len(jax.devices())
    expected = sum(int(holdout[i]["loss_mask"].sum()) for i in range(2 * batch))
    assert result.eval_history[-1]["eval_tokens"] == expected
    assert {d for r in range(len(holdout)) for _, d in holdout.row_items(r)} <= set(range(140, 200))
    assert {d for r in range(len(train_rows)) for _, d in train_rows.row_items(r)} <= set(range(140))


def _check_sft_resume(tmp_path: Path, num_workers: int, runtime: dict | None = None) -> None:
    """8 steps == 3 steps + checkpoint + resume + 5 steps, bit for bit (model, optimizer, losses, data state)."""
    corpus = tmp_path / "sft"
    write_sft(corpus, _examples(2, 200))
    options = {"num_workers": num_workers, "runtime": runtime}
    full = Trainer(_sft_experiment(tmp_path / "full", corpus, 8, **options)).run()
    first = Trainer(_sft_experiment(tmp_path / "split", corpus, 3, **options)).run()
    second = Trainer(_sft_experiment(tmp_path / "split", corpus, 8, **options)).run()
    assert (first.step, second.step) == (3, 8)
    for expected, actual in zip(_host(full.model), _host(second.model), strict=True):
        np.testing.assert_array_equal(expected, actual)
    for expected, actual in zip(_host(full.opt_state), _host(second.opt_state), strict=True):
        np.testing.assert_array_equal(expected, actual)
    assert full.data_state == second.data_state and full.data_state.batches_consumed == 16
    np.testing.assert_array_equal(
        np.array([r["loss"] for r in full.history]), np.array([r["loss"] for r in first.history + second.history])
    )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_sft_resume_matches_the_uninterrupted_run(tmp_path: Path, num_workers: int) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("GPU scatter-adds are not deterministic; the GPU check runs in XLA's deterministic mode below")
    _check_sft_resume(tmp_path, num_workers)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU determinism check")
def test_sft_resume_is_bitwise_on_gpu_in_deterministic_mode(tmp_path: Path) -> None:
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); import test_trainer_sft as t; "
        f"t._check_sft_resume(Path({str(tmp_path)!r}), 0, runtime={{'deterministic': True}}); print('BITWISE')"
    )
    env = dict(os.environ, XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " " + DETERMINISTIC_FLAG).strip())
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0 and "BITWISE" in proc.stdout, proc.stderr[-3000:]


def test_sft_resume_refuses_changed_packing_or_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jla.data.sft as sft

    corpus = tmp_path / "sft"
    write_sft(corpus, _examples(3, 120))
    Trainer(_sft_experiment(tmp_path / "run", corpus, 2)).run()
    with pytest.raises(ValueError, match="data source/order changed"):
        Trainer(_sft_experiment(tmp_path / "run", corpus, 4, sft_packing="single")).run()
    with monkeypatch.context() as patch:  # a new packing-plan version (changed packer or row shuffle)
        patch.setattr(sft, "SFT_PLAN_VERSION", sft.SFT_PLAN_VERSION + 1)
        with pytest.raises(ValueError, match="data source/order changed"):
            Trainer(_sft_experiment(tmp_path / "run", corpus, 4)).run()
    remasked = [(t, np.ones_like(m)) for t, m in _examples(3, 120)]  # same tokens, new mask
    write_sft(corpus, remasked)
    with pytest.raises(ValueError, match="data source/order changed"):
        Trainer(_sft_experiment(tmp_path / "run", corpus, 4)).run()


def test_sft_config_validation_and_example_config(tmp_path: Path) -> None:
    from jla.config import DataConfig

    with pytest.raises(ValueError, match="data.paths is required"):
        DataConfig(source="sft")
    with pytest.raises(ValueError, match="segment_ids"):
        DataConfig(source="sft", paths=("x",), document_masking=True, eos_id=0)
    with pytest.raises(ValueError, match="weights"):
        DataConfig(source="sft", paths=("x", "y"), weights=(1.0, 2.0))
    with pytest.raises(ValueError, match="eval_holdout_records needs exactly one"):
        DataConfig(source="sft", paths=("x", "y"), eval_holdout_records=10)
    with pytest.raises(ValueError, match="needs data.eval_path or"):
        _sft_experiment(tmp_path / "run", tmp_path / "missing", 2, final_eval=True)
    config = load_config(ROOT / "configs" / "sft-chat-125m.toml", ["data.sft_packing=single"])
    assert config.data.source == "sft" and config.data.sft_packing == "single"
    assert config.data.sft_pad_id < config.model.vocab_size
