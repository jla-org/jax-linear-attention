# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DPO: per-token log-prob op, preference data, loss vs a NumPy oracle, training, resume, CLI."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.cli import main as cli_main
from jla.config import DataConfig, ExperimentConfig, MeshConfig, OptimConfig, TrainConfig
from jla.data.preference import (
    PairRows,
    PreferenceDataset,
    SyntheticPreferenceDataset,
    iter_pairs,
    load_preference_jsonl,
    split_common_prefix,
)
from jla.models import build_model, get_preset
from jla.ops.token_logprobs import linear_token_logprobs, token_logprobs_from_logits
from jla.training.dpo import (
    TRAINABLE,
    DPOConfig,
    DPOTrainer,
    PolicyAndReference,
    build_dpo_eval_step,
    dpo_loss,
    load_dpo_config,
    load_model_weights,
    preference_terms,
    sequence_logps,
)

SEQ_LEN = 16


def _model_config():
    return get_preset("tiny").replace(compute_dtype="float32")


def _config(output_dir: Path, *, max_steps: int = 6, schedule_steps: int = 0, lr: float = 3e-3, **train) -> ExperimentConfig:
    train_fields = {
        "max_steps": max_steps,
        "seq_len": SEQ_LEN,
        "micro_batch_per_replica": 4,
        "grad_accum": 1,
        "log_every": 1,
        "eval_every": 0,
        "eval_batches": 4,
        "final_eval": False,
        "output_dir": str(output_dir),
        "ce_compute_dtype": "float32",
        **train,
    }
    return ExperimentConfig(
        model=_model_config(),
        name="dpo-test",
        optim=OptimConfig(lr=lr, warmup_steps=0, weight_decay=0.0, schedule="cosine", schedule_steps=schedule_steps),
        train=TrainConfig(**train_fields),
        data=DataConfig(num_workers=0),
        mesh=MeshConfig(fsdp=-1),
    )


def _host(tree) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


# -- the op ----------------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_chunk", [16, 64])
def test_token_logprobs_match_log_softmax_values_and_gradients(vocab_chunk: int) -> None:
    keys = jax.random.split(jax.random.key(0), 4)
    n, d, v = 12, 8, 50  # 50 is not a multiple of the chunk
    hidden = jax.random.normal(keys[0], (n, d), jnp.float32)
    weight = jax.random.normal(keys[1], (v, d), jnp.float32)
    targets = jax.random.randint(keys[2], (n,), 0, v).at[jnp.array([1, 7])].set(-100)
    cotangent = jax.random.normal(keys[3], (n,), jnp.float32)

    def ours(h, w):
        return linear_token_logprobs(h, w, targets, vocab_chunk=vocab_chunk, compute_dtype=jnp.float32)

    def oracle(h, w):
        return token_logprobs_from_logits(h @ w.T, targets)

    np.testing.assert_allclose(ours(hidden, weight), oracle(hidden, weight), rtol=1e-5, atol=1e-5)
    assert float(ours(hidden, weight)[1]) == 0.0
    grads = jax.grad(lambda h, w: jnp.sum(cotangent * ours(h, w)), argnums=(0, 1))(hidden, weight)
    expected = jax.grad(lambda h, w: jnp.sum(cotangent * oracle(h, w)), argnums=(0, 1))(hidden, weight)
    for got, want in zip(grads, expected):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    assert np.all(np.asarray(grads[0])[[1, 7]] == 0.0)  # ignored rows get no gradient


# -- data ------------------------------------------------------------------------------------


def test_rows_mask_the_prompt_and_padding() -> None:
    pairs = [(np.array([5, 6, 7]), np.array([8, 9]), np.array([10]))]
    sample = PreferenceDataset(pairs, 8, max_prompt_len=4, pad_id=0, eos_id=2)[0]
    # chosen: 5 6 7 | 8 9 2 ; rejected: 5 6 7 | 10 2
    np.testing.assert_array_equal(sample["input_ids"], [[5, 6, 7, 8, 9, 0, 0, 0], [5, 6, 7, 10, 0, 0, 0, 0]])
    np.testing.assert_array_equal(sample["labels"], [[6, 7, 8, 9, 2, 0, 0, 0], [6, 7, 10, 2, 0, 0, 0, 0]])
    np.testing.assert_array_equal(sample["loss_mask"], [[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]])


def test_rows_truncate_prompt_from_the_left_and_response_from_the_right() -> None:
    pairs = [(np.arange(1, 11), np.arange(20, 30), np.arange(40, 42))]
    sample = PreferenceDataset(pairs, 8, max_prompt_len=3)[0]
    np.testing.assert_array_equal(sample["input_ids"][0], [8, 9, 10, 20, 21, 22, 23, 24])
    np.testing.assert_array_equal(sample["labels"][0], [9, 10, 20, 21, 22, 23, 24, 25])
    np.testing.assert_array_equal(sample["loss_mask"][0], [0, 0, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(sample["loss_mask"][1], [0, 0, 1, 1, 0, 0, 0, 0])


def test_pair_rows_interleave_chosen_and_rejected() -> None:
    dataset = SyntheticPreferenceDataset(64, 8, num_pairs=3)
    batch = {key: np.stack([dataset[i][key] for i in range(3)]) for key in dataset[0]}
    rows = PairRows()(batch)
    assert rows["input_ids"].shape == (6, 8)
    np.testing.assert_array_equal(rows["labels"][2], dataset[1]["labels"][0])
    np.testing.assert_array_equal(rows["labels"][3], dataset[1]["labels"][1])


def test_record_formats_tokens_text_and_common_prefix(tmp_path: Path) -> None:
    def encode(text: str, special: bool) -> list[int]:
        return ([1] if special else []) + [3 + ord(c) % 40 for c in text]

    assert split_common_prefix([1, 2, 3, 4], [1, 2, 5]) == ([1, 2], [3, 4], [5])
    records = [
        {"prompt_ids": [1, 2], "chosen_ids": [3], "rejected_ids": [4, 5]},
        {"chosen_ids": [1, 2, 3], "rejected_ids": [1, 2, 4]},
        {"prompt": "ab", "chosen": "c", "rejected": "dd"},
        {"chosen": "xy", "rejected": "xz"},
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n\n", encoding="utf-8")
    loaded = load_preference_jsonl(path, encode=encode)
    expected = [
        ([1, 2], [3], [4, 5]),
        ([1, 2], [3], [4]),
        (encode("ab", True), encode("c", False), encode("dd", False)),
        (encode("x", True), [3 + ord("y") % 40], [3 + ord("z") % 40]),
    ]
    for got, want in zip(loaded, expected, strict=True):
        for a, b in zip(got, want, strict=True):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="need a tokenizer"):
        iter_pairs([{"prompt": "a", "chosen": "b", "rejected": "c"}])
    with pytest.raises(ValueError, match="empty prompt"):
        PreferenceDataset([(np.array([], np.int32), np.array([1]), np.array([2]))], 8, max_prompt_len=4)


# -- loss ------------------------------------------------------------------------------------


def _numpy_terms(pi, ref, n, beta, loss_type, eps):
    pi, ref, n = (np.asarray(x, np.float64).reshape(-1, 2) for x in (pi, ref, n))
    pi_sum, ref_sum = pi, ref  # rewards use the summed log-probs for every loss type
    if loss_type == "ipo":
        pi, ref = pi / np.maximum(n, 1), ref / np.maximum(n, 1)
    h = (pi[:, 0] - pi[:, 1]) - (ref[:, 0] - ref[:, 1])
    if loss_type == "sigmoid":
        losses = (1 - eps) * np.logaddexp(0.0, -beta * h) + eps * np.logaddexp(0.0, beta * h)
    elif loss_type == "ipo":
        losses = (h - 1 / (2 * beta)) ** 2
    else:
        losses = np.maximum(0.0, 1 - beta * h)
    chosen, rejected = beta * (pi_sum[:, 0] - ref_sum[:, 0]), beta * (pi_sum[:, 1] - ref_sum[:, 1])
    valid = (n[:, 0] > 0) & (n[:, 1] > 0)
    return {
        "loss_sum": losses[valid].sum(),
        "tokens": valid.sum(),
        "reward_margin_sum": (chosen - rejected)[valid].sum(),
        "reward_accuracy_sum": (chosen > rejected)[valid].sum(),
        "reward_chosen_sum": chosen[valid].sum(),
    }


@pytest.mark.parametrize("loss_type,eps", [("sigmoid", 0.0), ("sigmoid", 0.1), ("ipo", 0.0), ("hinge", 0.0)])
def test_preference_terms_match_numpy(loss_type: str, eps: float) -> None:
    rng = np.random.default_rng(0)
    pi = rng.normal(-20, 5, 16).astype(np.float32)
    ref = rng.normal(-20, 5, 16).astype(np.float32)
    n = rng.integers(1, 9, 16).astype(np.float32)
    n[5] = 0  # pair 2 has an empty rejected row and does not count
    got = preference_terms(jnp.asarray(pi), jnp.asarray(ref), jnp.asarray(n), beta=0.3, loss_type=loss_type, label_smoothing=eps)
    want = _numpy_terms(pi, ref, n, 0.3, loss_type, eps)
    for name, value in want.items():
        np.testing.assert_allclose(float(got[name]), value, rtol=2e-5, atol=1e-5, err_msg=name)


def test_loss_rewards_and_gradients_match_trl() -> None:
    """TRL 1.16's own ``DPOTrainer._compute_loss`` (``benchmarks/dpo_trl_reference.py``, float64 torch).

    TRL 1.16's sigmoid loss ignores ``label_smoothing`` (it enters only its robust, EXO and AOT losses),
    so conservative DPO is checked against the cDPO formula of TRL's earlier sigmoid loss in
    test_preference_terms_match_numpy.
    """
    data = json.loads((Path(__file__).parent / "data" / "dpo_trl_reference.json").read_text(encoding="utf-8"))
    token_logps = jnp.asarray(np.asarray(data["inputs"]["token_logps"]), jnp.float32)
    mask = jnp.asarray(np.asarray(data["inputs"]["mask"]), jnp.float32)
    ref_logps = jnp.asarray(np.asarray(data["inputs"]["ref_logps"]), jnp.float32)
    pairs = data["pairs"]
    assert len(data["cases"]) == 5
    for case in data["cases"]:
        def mean_loss(lp, case=case):
            terms = preference_terms(jnp.sum(lp * mask, -1), ref_logps, jnp.sum(mask, -1), beta=case["beta"],
                                     loss_type=case["loss_type"])
            return terms["loss_sum"] / terms["tokens"], terms

        (loss, terms), grad = jax.value_and_grad(mean_loss, has_aux=True)(token_logps)
        label = f"{case['loss_type']} beta={case['beta']}"
        assert float(terms["tokens"]) == pairs
        np.testing.assert_allclose(float(loss), case["loss"], rtol=2e-6, err_msg=label)
        # TRL sums every position of its fused head's per-token log-probs (zero, with no gradient to the
        # logits, outside the response); compare the gradient on the response tokens.
        np.testing.assert_allclose(np.asarray(grad), np.asarray(case["grad_token_logps"]) * np.asarray(mask), rtol=1e-5,
                                   atol=1e-8, err_msg=label)
        metrics = case["metrics"]
        for ours, theirs in [("reward_chosen_sum", "rewards/chosen"), ("reward_rejected_sum", "rewards/rejected"),
                             ("reward_margin_sum", "rewards/margins"), ("reward_accuracy_sum", "rewards/accuracies"),
                             ("logp_chosen_sum", "logps/chosen"), ("logp_rejected_sum", "logps/rejected")]:
            np.testing.assert_allclose(float(terms[ours]) / pairs, metrics[theirs], rtol=1e-5, atol=1e-6,
                                       err_msg=f"{label} {ours}")


def _batch(num_pairs: int = 4, seed: int = 0) -> dict[str, jax.Array]:
    dataset = SyntheticPreferenceDataset(512, SEQ_LEN, num_pairs=num_pairs, seed=seed)
    batch = {key: np.stack([dataset[i][key] for i in range(num_pairs)]) for key in dataset[0]}
    return {key: jnp.asarray(value) for key, value in PairRows()(batch).items()}


def test_sequence_logps_match_explicit_log_softmax() -> None:
    model = build_model(_model_config(), key=jax.random.key(1))
    batch = _batch()
    logps, counts, _ = sequence_logps(model, batch, None, inference=True, compute_dtype=jnp.float32, vocab_chunk=128)
    logits = np.asarray(model.logits(model(batch["input_ids"], inference=True).hidden), np.float64)
    logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
    picked = np.take_along_axis(logp, np.asarray(batch["labels"])[..., None], axis=-1)[..., 0]
    mask = np.asarray(batch["loss_mask"]).astype(bool)
    np.testing.assert_allclose(np.asarray(logps), np.where(mask, picked, 0.0).sum(-1), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(counts), mask.sum(-1))


def test_policy_equal_to_reference_gives_log2() -> None:
    model = build_model(_model_config(), key=jax.random.key(2))
    pair = PolicyAndReference(policy=model, reference=jax.tree.map(lambda x: x + 0 if eqx.is_array(x) else x, model))
    terms = build_dpo_eval_step(DPOConfig(beta=0.5, label_smoothing=0.2), compute_dtype="float32")(pair, _batch(8))
    pairs = float(terms["tokens"])
    assert pairs == 8
    np.testing.assert_allclose(float(terms["loss_sum"]) / pairs, math.log(2.0), rtol=1e-6)
    assert float(terms["reward_margin_sum"]) == 0.0 and float(terms["reward_accuracy_sum"]) == 0.0


def test_dpo_gradient_matches_explicit_logits_autodiff() -> None:
    model = build_model(_model_config(), key=jax.random.key(3))
    reference = build_model(_model_config(), key=jax.random.key(4))
    pair = PolicyAndReference(policy=model, reference=reference)
    batch = _batch(4, seed=1)
    dpo = DPOConfig(beta=0.2)
    params, static = eqx.partition(pair, TRAINABLE)
    loss_fn = dpo_loss(dpo, compute_dtype="float32", vocab_chunk=128)

    def explicit(p):
        def seq(m):
            logits = m.logits(m(batch["input_ids"], inference=False).hidden)
            logp = jax.nn.log_softmax(logits, axis=-1)
            picked = jnp.take_along_axis(logp, batch["labels"][..., None], axis=-1)[..., 0]
            return jnp.sum(jnp.where(batch["loss_mask"] > 0, picked, 0.0), -1)

        full = eqx.combine(p, static)
        counts = jnp.sum(batch["loss_mask"], -1).astype(jnp.float32)
        return preference_terms(seq(full.policy), jax.lax.stop_gradient(seq(full.reference)), counts, beta=0.2, loss_type="sigmoid")["loss_sum"]

    with jax.default_matmul_precision("highest"):  # no TF32 on GPU: compare the math, not the rounding
        ours = jax.grad(lambda p: loss_fn(eqx.combine(p, static), batch, jax.random.key(0))[0])(params)
        want = jax.grad(explicit)(params)
    assert jax.tree.leaves(ours.reference) == []  # the reference has no trainable leaves
    for got, expected in zip(jax.tree.leaves(ours), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=1e-6)


# -- training ----------------------------------------------------------------------------------


def test_training_lowers_the_loss_and_keeps_the_reference_frozen(tmp_path: Path) -> None:
    config = _config(tmp_path / "run", max_steps=40, eval_every=20, final_eval=True, eval_batches=4)
    result = DPOTrainer(config, DPOConfig(beta=0.5, synthetic_pairs=256)).run()
    losses = [record["loss"] for record in result.history]
    assert abs(losses[0] - math.log(2.0)) < 1e-4  # policy == reference before the first update
    assert np.mean(losses[-5:]) < 0.35, losses
    assert result.eval_history[-1]["eval_reward_accuracy"] > 0.9, result.eval_history
    assert result.eval_history[-1]["eval_loss"] < 0.45, result.eval_history
    initial = build_model(_model_config(), key=jax.random.split(jax.random.key(0))[0])
    # The trainer initializes under jit; on GPU that differs from this eager init by an ulp.
    for got, want in zip(_host(result.pair.reference), _host(initial), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)
    assert any(not np.allclose(a, b, rtol=1e-3, atol=1e-5) for a, b in zip(_host(result.policy), _host(initial)))
    paths = ["/".join(map(str, p)) for p, _ in jax.tree_util.tree_flatten_with_path(result.opt_state)[0]]
    assert paths and not any("reference" in p for p in paths)


def test_fp8_current_scaling_trains_and_delayed_is_refused(tmp_path: Path) -> None:
    """Current-scaling FP8 trains (policy == reference at the start); delayed scaling is refused up front (its
    step does not lower with the frozen reference's FP8 state)."""
    config = _config(tmp_path / "current", max_steps=2)
    config = config.replace(model=config.model.replace(fp8="current"))
    losses = [record["loss"] for record in DPOTrainer(config, DPOConfig(beta=0.3, synthetic_pairs=64)).run().history]
    assert len(losses) == 2 and abs(losses[0] - math.log(2.0)) < 1e-4 and np.all(np.isfinite(losses))
    path = tmp_path / "dpo.toml"
    path.write_text(_TOML.format(out=tmp_path / "out"), encoding="utf-8")
    with pytest.raises(ValueError, match="fp8 = .delayed. is not supported by jla dpo"):
        load_dpo_config(path, ["model.fp8=delayed"])


def _check_resume(tmp_path: Path, runtime: dict | None = None) -> None:
    """6 steps == 3 steps + checkpoint + resume + 3 steps, bit for bit."""
    from jla.config import RuntimeConfig

    def config(name: str, max_steps: int) -> ExperimentConfig:
        return _config(tmp_path / name, max_steps=max_steps, schedule_steps=6).replace(runtime=RuntimeConfig(**(runtime or {})))

    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    full = DPOTrainer(config("full", 6), dpo).run()
    first = DPOTrainer(config("split", 3), dpo).run()
    second = DPOTrainer(config("split", 6), dpo).run()
    assert (first.step, second.step) == (3, 6)
    for want, got in zip(_host(full.policy), _host(second.policy), strict=True):
        np.testing.assert_array_equal(want, got)
    for want, got in zip(_host(full.opt_state), _host(second.opt_state), strict=True):
        np.testing.assert_array_equal(want, got)
    np.testing.assert_array_equal(jax.random.key_data(full.rng_key), jax.random.key_data(second.rng_key))
    assert full.data_state == second.data_state
    np.testing.assert_array_equal([r["loss"] for r in full.history], [r["loss"] for r in first.history + second.history])


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="bitwise in-process check on CPU; GPU runs it in deterministic mode")
def test_resumed_run_matches_the_uninterrupted_run(tmp_path: Path) -> None:
    _check_resume(tmp_path)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU determinism check")
def test_resumed_run_is_bitwise_identical_on_gpu_in_deterministic_mode(tmp_path: Path) -> None:
    """GPU scatter-adds use atomics; XLA's deterministic mode (flags apply before JAX starts) fixes that."""
    from jla.launch_env import DETERMINISTIC_FLAG

    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, 'tests'); import test_dpo as t; "
        f"t._check_resume(Path({str(tmp_path)!r}), runtime={{'deterministic': True}}); print('BITWISE')"
    )
    env = dict(os.environ, XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " " + DETERMINISTIC_FLAG).strip())
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0 and "BITWISE" in proc.stdout, proc.stderr[-3000:]


def test_resume_rejects_a_different_reference(tmp_path: Path) -> None:
    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    DPOTrainer(_config(tmp_path / "base", max_steps=2), dpo).run()
    DPOTrainer(_config(tmp_path / "split", max_steps=2, schedule_steps=4), dpo).run()
    other = DPOConfig(beta=0.3, synthetic_pairs=64, init_checkpoint=str(tmp_path / "base"))
    with pytest.raises(ValueError, match="reference model was"):
        DPOTrainer(_config(tmp_path / "split", max_steps=4), other).run()
    with pytest.raises(ValueError, match="cannot resume the data stream"):
        DPOTrainer(_config(tmp_path / "split", max_steps=4), DPOConfig(beta=0.3, synthetic_pairs=32)).run()


def test_shipped_dpo_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "post_train" / "dpo-tiny.toml"
    config, dpo = load_dpo_config(path)
    assert (config.model.vocab_size, dpo.loss_type, dpo.train_path) == (512, "sigmoid", None)


def test_init_checkpoint_sets_policy_and_reference(tmp_path: Path) -> None:
    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    base = DPOTrainer(_config(tmp_path / "base", max_steps=2), dpo).run()
    template = build_model(_model_config(), key=jax.random.key(9))
    loaded = load_model_weights(tmp_path / "base", template)
    for want, got in zip(_host(base.policy), _host(loaded), strict=True):
        np.testing.assert_array_equal(want, got)
    tuned = DPOTrainer(_config(tmp_path / "tuned", max_steps=1), DPOConfig(beta=0.3, synthetic_pairs=64, init_checkpoint=str(tmp_path / "base" / "checkpoints"))).run()
    for want, got in zip(_host(base.policy), _host(tuned.pair.reference), strict=True):
        np.testing.assert_array_equal(want, got)
    assert abs(tuned.history[0]["loss"] - math.log(2.0)) < 1e-4


def test_trainable_patterns_see_policy_paths_and_must_match(tmp_path: Path) -> None:
    from jla.training.optimizers import _path_string

    config = _config(tmp_path / "anchored", max_steps=1, save_final=False)
    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    with pytest.raises(ValueError, match="match no policy parameter"):
        DPOTrainer(config.replace(optim=dataclasses.replace(config.optim, trainable_patterns=("^embed",))), dpo).run()
    config = _config(tmp_path / "embed", max_steps=1, save_final=False)
    tuned = DPOTrainer(config.replace(optim=dataclasses.replace(config.optim, trainable_patterns=("^policy/embed",))), dpo).run()
    leaves = jax.tree_util.tree_flatten_with_path(eqx.filter(tuned.policy, eqx.is_array))[0]
    changed = [_path_string(path) for (path, got), want in zip(leaves, _host(tuned.pair.reference), strict=True) if not np.array_equal(np.asarray(got), want)]
    assert changed and all(text.startswith("embed") for text in changed), changed


def test_consecutive_rejected_steps_halt_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jla.training.dpo as dpo_module

    real_loss = dpo_module.dpo_loss

    def nan_loss(*args, **kwargs):
        loss_fn = real_loss(*args, **kwargs)

        def wrapped(pair, batch, key):
            objective, aux = loss_fn(pair, batch, key)
            return objective * jnp.nan, aux  # non-finite gradients: every update is rejected

        return wrapped

    monkeypatch.setattr(dpo_module, "dpo_loss", nan_loss)
    run = tmp_path / "run"
    result = DPOTrainer(_config(run, max_steps=10, max_consecutive_skips=2), DPOConfig(beta=0.3, synthetic_pairs=64)).run()
    assert result.stop_reason.startswith("halt:") and result.step == 2, result.stop_reason
    assert result.history[-1]["skipped_steps"] == 1 and result.history[-1]["skip_reason"] > 0
    assert (run / "flight_recorder.json").exists() and (run / "sharding_dump.json").exists()
    assert (run / "checkpoints" / "step_00000002" / "manifest.json").exists()


def test_grad_accum_splits_pairs_without_changing_the_update(tmp_path: Path) -> None:
    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    one = DPOTrainer(_config(tmp_path / "one", max_steps=1, micro_batch_per_replica=4, grad_accum=1, save_final=False), dpo).run()
    two = DPOTrainer(_config(tmp_path / "two", max_steps=1, micro_batch_per_replica=2, grad_accum=2, save_final=False), dpo).run()
    np.testing.assert_allclose(one.history[0]["loss"], two.history[0]["loss"], rtol=1e-5)
    np.testing.assert_allclose(one.history[0]["grad_norm"], two.history[0]["grad_norm"], rtol=1e-4)


def test_default_weight_decay_and_data_workers(tmp_path: Path) -> None:
    """Defaults the other tests turn off: weight decay 0.1 (policy only) and 2 data workers."""
    dpo = DPOConfig(beta=0.3, synthetic_pairs=64)
    config = _config(tmp_path / "w0", max_steps=3, save_final=False)
    config = config.replace(optim=dataclasses.replace(config.optim, weight_decay=0.1))
    zero = DPOTrainer(config, dpo).run()
    workers = config.replace(train=dataclasses.replace(config.train, output_dir=str(tmp_path / "w2")), data=DataConfig(num_workers=2))
    two = DPOTrainer(workers, dpo).run()
    assert all(record["step_applied"] == 1.0 for record in zero.history), zero.history
    assert abs(zero.history[0]["loss"] - math.log(2.0)) < 1e-4
    np.testing.assert_allclose([r["loss"] for r in zero.history], [r["loss"] for r in two.history], rtol=1e-5)
    initial = build_model(_model_config(), key=jax.random.split(jax.random.key(0))[0])
    for got, want in zip(_host(zero.pair.reference), _host(initial), strict=True):  # decay would move it by ~1e-3
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


def test_jsonl_data_eval_eos_and_safetensors_init(tmp_path: Path) -> None:
    from jla.training.checkpointing import save_pytree

    rng = np.random.default_rng(0)

    def write(name: str, n: int) -> str:
        lines = []
        for _ in range(n):
            prompt = rng.integers(3, 512, 5).tolist()
            chosen = [(prompt[-1] + 1 + k) % 509 + 3 for k in range(6)]  # ids >= 3: never pad (0) or eos (2)
            record = {"prompt_ids": prompt, "chosen_ids": chosen, "rejected_ids": rng.integers(3, 512, 4).tolist()}
            lines.append(json.dumps(record))
        (tmp_path / name).write_text("\n".join(lines), encoding="utf-8")
        return str(tmp_path / name)

    weights = build_model(_model_config(), key=jax.random.key(7))
    init = save_pytree(tmp_path / "init.safetensors", eqx.filter(weights, eqx.is_array))
    dpo = DPOConfig(beta=0.3, train_path=write("train.jsonl", 32), eval_path=write("eval.jsonl", 8), init_checkpoint=str(init), eos_id=2)
    result = DPOTrainer(_config(tmp_path / "run", max_steps=4, eval_every=2, final_eval=True), dpo).run()
    assert abs(result.history[0]["loss"] - math.log(2.0)) < 1e-4
    for got, want in zip(_host(result.pair.reference), _host(weights), strict=True):
        np.testing.assert_array_equal(got, want)
    assert [record["step"] for record in result.eval_history] == [2, 4], result.eval_history
    assert result.eval_history[-1]["eval_pairs"] == 8.0
    assert result.history[0]["response_tokens"] == (6 + 1) + (4 + 1)  # per pair, eos appended to each response


@pytest.mark.parametrize("preset", ["tiny-moe", "tiny-mamba2"])
def test_moe_and_ssm_presets_start_at_log2(tmp_path: Path, preset: str) -> None:
    config = _config(tmp_path / preset, max_steps=2, save_final=False).replace(model=get_preset(preset).replace(compute_dtype="float32"))
    result = DPOTrainer(config, DPOConfig(beta=0.3, synthetic_pairs=64)).run()
    assert abs(result.history[0]["loss"] - math.log(2.0)) < 1e-4, result.history[0]
    assert all(record["step_applied"] == 1.0 for record in result.history), result.history


@pytest.mark.parametrize("loss_type", ["ipo", "hinge"])
def test_ipo_and_hinge_losses_train(tmp_path: Path, loss_type: str) -> None:
    dpo = DPOConfig(beta=0.5, loss_type=loss_type, synthetic_pairs=128)
    result = DPOTrainer(_config(tmp_path / loss_type, max_steps=20, save_final=False), dpo).run()
    margins = [record["reward_margin"] for record in result.history]
    assert all(np.isfinite(record["loss"]) for record in result.history)
    assert margins[-1] > margins[0], margins


_MULTI_DEVICE = """
import sys
sys.path.insert(0, 'tests')
from pathlib import Path
import jax, numpy as np
import test_dpo as t
from jla.config import OptimConfig
from jla.training.dpo import DPOConfig, DPOTrainer
import contextlib
import jla.distributed.kernel_sharding as kernel_sharding
import jla.training.dpo as dpo_module
assert len(jax.devices()) == 8, jax.devices()
# The step traces under the trainer's mesh and kernel-sharding rules (as in jla train), so kernels XLA
# cannot partition run per shard and shard_map layers see the mesh.
entered, seen = [], []
use_rules, logps = kernel_sharding.use_rules, dpo_module.sequence_logps
@contextlib.contextmanager
def recording_use_rules(rules):
    entered.append(rules)
    with use_rules(rules):
        yield
def recording_logps(*args, **kwargs):
    mesh = jax.sharding.get_abstract_mesh()
    seen.append((() if mesh.empty else tuple(mesh.axis_names), dict(zip(mesh.axis_names, mesh.axis_sizes)), len(entered)))
    return logps(*args, **kwargs)
kernel_sharding.use_rules, dpo_module.sequence_logps = recording_use_rules, recording_logps
out, dpo = Path(sys.argv[1]), DPOConfig(beta=0.5, synthetic_pairs=64)
sgd = OptimConfig(name="sgd", lr=0.5, b1=0.9, warmup_steps=0, weight_decay=0.0)  # Adam would amplify reduction-order noise
losses = []
for name, pairs, devices in (("eight", 1, jax.devices()), ("one", 8, jax.devices()[:1])):
    run = DPOTrainer(t._config(out / name, max_steps=4, micro_batch_per_replica=pairs).replace(optim=sgd), dpo, devices=devices).run()
    losses.append([record["loss"] for record in run.history])
    if name == "eight":
        assert seen and all(axes and shape.get("fsdp") == 8 and rules for axes, shape, rules in seen), seen
np.testing.assert_allclose(losses[0], losses[1], rtol=2e-5, atol=2e-6)
print("MULTI_OK")
"""


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="8 forced host devices")
def test_eight_devices_split_pairs_like_one_device(tmp_path: Path) -> None:
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, "-c", _MULTI_DEVICE, str(tmp_path)], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0 and "MULTI_OK" in proc.stdout, proc.stderr[-4000:]


# -- config and CLI -----------------------------------------------------------------------------

_TOML = """
name = "dpo-cli"

[model]
preset = "tiny"
compute_dtype = "float32"

[train]
max_steps = 2
seq_len = 16
micro_batch_per_replica = 2
grad_accum = 1
log_every = 1
eval_batches = 1
output_dir = "{out}"
ce_compute_dtype = "float32"

[optim]
lr = 1e-3

[data]
num_workers = 0

[dpo]
beta = 0.2
synthetic_pairs = 32
"""


def test_config_loading_overrides_and_rejections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "dpo.toml"
    path.write_text(_TOML.format(out=tmp_path / "out"), encoding="utf-8")
    config, dpo = load_dpo_config(path, ["dpo.loss_type=ipo", "dpo.beta=0.5"])
    assert (dpo.loss_type, dpo.beta, config.train.seq_len) == ("ipo", 0.5, 16)
    assert cli_main(["dpo", "--config", str(path), "--dry-run"]) == 0
    assert "[dpo]" in capsys.readouterr().out
    for override, message in [
        ("train.z_loss=0.1", "z_loss"),
        ("data.synthetic_samples=10", r"\[data\]"),
        ("dpo.label_smoothing=0.1", None),  # valid with the sigmoid loss
        ("dpo.unknown=1", "unknown key"),
        ("dpo.train_path=pairs.jsonl", "needs dpo.eval_path"),
        ("model.resid_dropout=0.1", "dropout"),
        ("train.dispatch_ahead=true", "dispatch_ahead"),
        ('train.grad_reduce="step"', "grad_reduce"),
        ('train.update_guard="select"', "update_guard"),
    ]:
        if message is None:
            load_dpo_config(path, [override])
            continue
        with pytest.raises(ValueError, match=message):
            load_dpo_config(path, [override])
    with pytest.raises(ValueError, match="sigmoid"):
        load_dpo_config(path, ["dpo.loss_type=hinge", "dpo.label_smoothing=0.1"])
    with pytest.raises(ValueError, match="ce_grad_in_forward"):  # the LM cross entropy's option, unused by DPO
        load_dpo_config(path, ["train.ce_impl=auto", "train.ce_grad_in_forward=true"])


def test_cli_trains_and_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "dpo.toml"
    path.write_text(_TOML.format(out=tmp_path / "out"), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    proc = subprocess.run([sys.executable, "-m", "jla.cli", "dpo", "--config", str(path)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["step"] == 2 and summary["eval"] is not None
    assert (tmp_path / "out" / "checkpoints" / "step_00000002" / "manifest.json").exists()
