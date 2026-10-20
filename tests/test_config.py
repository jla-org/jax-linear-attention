# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strict experiment-config loading, overrides, serialization and hashing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _toy_model import experiment
from jla.config import (
    ExperimentConfig,
    apply_overrides,
    config_from_mapping,
    dump_toml,
    load_config,
    parse_override,
)
from jla.models import get_preset

ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE = """
[model]
vocab_size = 128
d_model = 64
n_layers = 2
d_ff = 128
compute_dtype = "float32"

[model.attention]
num_heads = 2
num_kv_heads = 1
head_dim = 32
"""


def _write(tmp_path: Path, text: str, name: str = "exp.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_toml_loading_builds_every_section(tmp_path: Path) -> None:
    text = 'name = "demo"\n' + MODEL_TABLE + '\n[optim]\nlr = 1e-3\nschedule = "wsd"\n\n[train]\nmax_steps = 50\nseq_len = 64\n\n[mesh.rules]\nexperts = ["fsdp"]\n'
    config = load_config(_write(tmp_path, text))
    assert config.name == "demo" and config.model.attention.num_kv_heads == 1
    assert config.optim.lr == 1e-3 and config.optim.schedule == "wsd" and config.train.seq_len == 64
    assert config.mesh.rules == (("experts", ("fsdp",)),)
    assert config.model.layer_types is None and config.model.resolved_layer_types == ("attention", "attention")


@pytest.mark.parametrize(
    ("extra", "error", "match"),
    [
        ("[train]\nmax_step = 5\n", ValueError, "unknown key"),
        ("[train]\nmax_steps = true\n", TypeError, "integer"),
        ("[train]\nlabel_smoothing = nan\n", ValueError, "finite"),
        ("[optim]\nname = \"adam\"\n", ValueError, "must be one of"),
        ("[optim]\nlr = \"fast\"\n", TypeError, "number"),
        ("[data]\ndocument_masking = true\n", ValueError, "requires data.eos_id"),
        ("[mesh]\nfsdp = 0\n", ValueError, "mesh.fsdp"),
        ("[model.attention]\nrope_theta = \"big\"\n", TypeError, "number"),
        ("bogus = 1\n", ValueError, "unknown top-level"),
    ],
)
def test_strict_validation_rejects_bad_values(tmp_path: Path, extra: str, error: type, match: str) -> None:
    if extra.startswith("[model.attention]"):
        text = MODEL_TABLE.replace("head_dim = 32", "head_dim = 32\n" + extra.split("\n", 1)[1])
    elif extra.startswith("["):
        text = MODEL_TABLE + extra
    else:
        text = extra + MODEL_TABLE  # top-level keys precede every table header
    with pytest.raises(error, match=match):
        load_config(_write(tmp_path, text))


def test_json_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    good = experiment("runs/x").to_json()
    assert load_config(_write(tmp_path, good, "ok.json")) == experiment("runs/x")
    with pytest.raises(ValueError, match="duplicate key"):
        load_config(_write(tmp_path, '{"name": "a", "name": "b"}', "dup.json"))
    with pytest.raises(ValueError, match="non-finite"):
        load_config(_write(tmp_path, '{"train": {"z_loss": NaN}}', "nan.json"))
    with pytest.raises(ValueError, match="must be .toml or .json"):
        load_config(_write(tmp_path, "", "exp.yaml"))


def test_overrides_parse_toml_values_and_bare_words() -> None:
    assert parse_override("optim.lr=3e-4") == (["optim", "lr"], 3e-4)
    assert parse_override("data.source=cidx") == (["data", "source"], "cidx")
    assert parse_override("data.paths=['a', 'b']") == (["data", "paths"], ["a", "b"])
    assert parse_override("train.output_dir=runs/x-1") == (["train", "output_dir"], "runs/x-1")
    assert parse_override("data.eos_id=none") == (["data", "eos_id"], None)
    with pytest.raises(ValueError, match="section.key=value"):
        parse_override("optim.lr")
    raw = {"train": {"max_steps": 5}}
    updated = apply_overrides(raw, ["train.max_steps=7", "optim.name=lion"])
    assert updated == {"train": {"max_steps": 7}, "optim": {"name": "lion"}} and raw == {"train": {"max_steps": 5}}
    with pytest.raises(ValueError, match="not a table"):
        apply_overrides(raw, ["train.max_steps.x=1"])


def test_overrides_are_validated_like_file_values(tmp_path: Path) -> None:
    path = _write(tmp_path, MODEL_TABLE)
    config = load_config(path, ["optim.lr=0.01", "train.grad_accum=4", "model.d_ff=256", "model.attention.head_dim=16"])
    assert config.optim.lr == 0.01 and config.train.grad_accum == 4
    assert config.model.d_ff == 256 and config.model.attention.head_dim == 16
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path, ["optim.learning_rate=0.1"])
    with pytest.raises(TypeError, match="integer"):
        load_config(path, ["train.max_steps=1.5"])


def test_toml_and_json_roundtrips_preserve_the_config_and_hash(tmp_path: Path) -> None:
    config = experiment(
        "runs/round",
        optim={"muon_adam_lr": 1e-3, "trainable_patterns": ("^embed",)},
        data={"eos_id": 0, "document_masking": True},
        mesh={"rules": (("experts", ("fsdp",)), ("vocab", ()))},
    )
    toml_path = config.save(tmp_path / "config.toml")
    json_path = config.save(tmp_path / "config.json")
    assert load_config(toml_path) == config == load_config(json_path)
    assert load_config(toml_path).config_hash() == config.config_hash()
    assert "eval_path" not in toml_path.read_text()  # None values are omitted


def test_hash_is_stable_across_processes_and_sensitive_to_changes(tmp_path: Path) -> None:
    config = experiment("runs/hash")
    assert config.config_hash() == experiment("runs/hash").config_hash()
    assert config.config_hash() != config.replace(name="other").config_hash()
    path = config.save(tmp_path / "c.toml")
    code = f"from jla.config import load_config; print(load_config({str(path)!r}).config_hash())"
    env_path = str(ROOT / "src")
    child = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": env_path, "PYTHONHASHSEED": "123"}
    )
    assert child.stdout.strip() == config.config_hash()


def test_cross_section_validation() -> None:
    with pytest.raises(ValueError, match="schedule_steps"):
        experiment("runs/x", optim={"schedule_steps": 5})
    with pytest.raises(ValueError, match="needs optim.warmup_steps"):
        experiment("runs/x", optim={"schedule": "inv_sqrt", "warmup_steps": 0})
    with pytest.raises(ValueError, match="eos_id must be < model.vocab_size"):
        experiment("runs/x", data={"eos_id": get_preset("tiny").vocab_size})
    with pytest.raises(ValueError, match="eval_path or data.eval_holdout_records"):
        experiment("runs/x", data={"source": "cidx", "paths": ("corpus",)})
    with pytest.raises(ValueError, match="profile_steps"):
        experiment("runs/x", train={"profile_start_step": 3})


def test_dump_toml_handles_nested_tables_and_quoting() -> None:
    text = dump_toml({"a": 1, "b": {"c": [1.5, 2.0], "d": "x\"y", "e": None, "weird key": True}})
    assert text == 'a = 1\n\n[b]\nc = [1.5, 2.0]\nd = "x\\"y"\n"weird key" = true\n'
    with pytest.raises(ValueError, match="cannot contain"):
        dump_toml({"x": float("inf")})


def test_model_presets_are_resolved_with_overrides(tmp_path: Path) -> None:
    text = '[model]\npreset = "tiny-hybrid"\nn_layers = 6\nlayer_types = ["gated_deltanet", "attention", "gated_deltanet", "attention", "gated_deltanet", "attention"]\n\n[model.mtp]\ndepth = 1\n'
    config = load_config(_write(tmp_path, text))
    preset = get_preset("tiny-hybrid")
    assert config.model.n_layers == 6 and config.model.mtp.depth == 1
    assert config.model.mtp.kind == preset.mtp.kind and config.model.linear_attention == preset.linear_attention
    with pytest.raises(KeyError, match="unknown model preset"):
        load_config(_write(tmp_path, '[model]\npreset = "nope"\n', "bad.toml"))


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "configs").glob("*.toml")))
def test_shipped_configs_load_and_roundtrip(name: str, tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / name)
    assert isinstance(config, ExperimentConfig)
    assert load_config(config.save(tmp_path / name)) == config


def test_cli_prints_the_resolved_config_and_hash() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jla.cli", "show-config", "--config", str(ROOT / "configs" / "tiny-synthetic.toml"), "--set", "optim.lr=0.002"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr
    expected = load_config(ROOT / "configs" / "tiny-synthetic.toml", ["optim.lr=0.002"])
    assert f"# config_hash = {expected.config_hash()}" in result.stdout
    assert "lr = 0.002" in result.stdout


def test_config_from_mapping_requires_a_model() -> None:
    with pytest.raises(ValueError, match=r"\[model\] section is required"):
        config_from_mapping({"name": "x"})
