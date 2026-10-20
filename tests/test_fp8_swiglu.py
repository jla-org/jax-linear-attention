# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused delayed-scaling FP8 SwiGLU MLP (jla.layers.fp8_swiglu) against the unfused FP8 path
(two jla.layers.fp8.fp8_dot calls around jla.layers.mlp.swiglu), its Pallas Triton passes against
their plain-JAX reference (GPU), and a model with ``fp8_fused`` against the unfused model and bf16."""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jla.layers import fp8 as F
from jla.layers import fp8_swiglu as S
from jla.layers.mlp import swiglu
from jla.models import build_model, get_preset
from jla.training.fp8 import without_fp8_state
from jla.training.step import LossConfig, build_train_step, lm_loss

ON_GPU = jax.default_backend() == "gpu"
IMPLS = ["xla", "pallas"] if ON_GPU else ["xla"]
gpu_only = pytest.mark.skipif(not ON_GPU, reason="the Pallas Triton passes run on GPUs")


def _close(actual, expected, rel) -> None:
    expected = np.asarray(expected, np.float64)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    np.testing.assert_allclose(np.asarray(actual, np.float64), expected, rtol=0, atol=rel * scale)


def _state(history: int, amaxes: tuple[float, float, float]) -> F.Fp8State:
    state = F.Fp8State.init(history)
    hist = lambda v: jnp.zeros(history).at[:2].set([0.5 * v, v])  # noqa: E731
    return eqx.tree_at(lambda s: (s.input_amax_history, s.kernel_amax_history, s.grad_amax_history), state,
                       tuple(hist(v) for v in amaxes))


def _operands(d=64, f=128, tokens=(2, 64), seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((*tokens, d)), jnp.bfloat16)
    w_gu = jnp.asarray(0.2 * rng.standard_normal((d, 2, f)), jnp.float32)
    w_dn = jnp.asarray(0.1 * rng.standard_normal((f, d)), jnp.float32)
    g = jnp.asarray(rng.standard_normal((*tokens, d)), jnp.bfloat16)
    # Histories near each tensor's range, so the scales are realistic (amax / fp8 max).
    st_gu = _state(8, (4.0, 0.8, 4.0))
    st_dn = _state(8, (6.0, 0.4, 4.0))
    return x, w_gu, w_dn, g, st_gu, st_dn


def _unfused(x, w_gu, w_dn, st_gu, st_dn):
    h = F.fp8_dot(x, w_gu, st_gu, recipe="delayed")
    return F.fp8_dot(swiglu(h), w_dn, st_dn, recipe="delayed")


@functools.partial(jax.jit, static_argnums=0)
def _value_and_vjp(fn, args, g):
    """``fn(*args)`` and its VJP for ``g``, jitted as in a train step (XLA's default flags)."""
    y, vjp = jax.vjp(fn, *args)
    return y, vjp(g)


def _fused(impl: str):
    return functools.partial(_fused_mlp, impl=impl)


def _fused_mlp(*args, impl: str):
    return S.fp8_swiglu_mlp(*args, impl=impl)


@pytest.mark.parametrize("impl", IMPLS)
def test_fused_mlp_matches_the_unfused_fp8_path(impl: str) -> None:
    """Jitted, as in training: on a GPU the fused path gives bitwise the unfused path's output and gradients.

    Both quantize the float32 SwiGLU straight to e4m3: on GPUs XLA drops the unfused SwiGLU's bfloat16
    rounding inside the fusion that quantizes it (``--xla_allow_excess_precision``, on by default).
    Exceptions: with the Pallas passes the gate/up weight gradient differs in FP8 GEMM accumulation
    order only (same FP8 operands; the transposed copy comes from the kernel, so cuBLASLt sees another
    layout), and XLA's CPU compiler keeps that bfloat16 rounding, so on CPU about 1% of the e4m3
    down-projection inputs differ by one unit in the last place, which moves ``y`` and ``dW_dn``."""
    x, w_gu, w_dn, g, st_gu, st_dn = _operands()
    args = (x, w_gu, w_dn, st_gu, st_dn)
    y_ref, grads_ref = _value_and_vjp(_unfused, args, g)
    y, grads = _value_and_vjp(_fused(impl), args, g)
    assert y.dtype == y_ref.dtype == jnp.bfloat16 and y.shape == y_ref.shape
    outs = {"y": (y, y_ref), **{n: (a, b) for n, a, b in zip(("dx", "dw_gu", "dw_dn"), grads[:3], grads_ref[:3])}}
    for name, (a, b) in outs.items():
        assert a.dtype == b.dtype and a.shape == b.shape, name
        if not ON_GPU and name in ("y", "dw_dn"):
            _close(a, b, 3e-2)
        elif name == "dw_gu" and (impl == "pallas" or not ON_GPU):  # accumulation order (CPU: GEMM layout)
            _close(a, b, 1e-3)
        else:
            np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32), err_msg=name)
    for new, ref in zip(grads[3:], grads_ref[3:]):  # the next delayed-scaling states
        for name in ("input", "kernel", "grad"):
            np.testing.assert_array_equal(np.asarray(getattr(new, f"{name}_scale")),
                                          np.asarray(getattr(ref, f"{name}_scale")))
            h_new = np.asarray(getattr(new, f"{name}_amax_history"))
            h_ref = np.asarray(getattr(ref, f"{name}_amax_history"))
            np.testing.assert_array_equal(h_new[1:], h_ref[1:])
            np.testing.assert_allclose(h_new[0], h_ref[0], rtol=1e-2)
            assert h_new[0] > 0


def _ulp_close(actual, expected, mantissa_bits: int, max_fraction: float) -> None:
    """FP8 values agree to one unit in the last place, and only a few differ at all."""
    a, b = (np.asarray(v, np.float32) for v in (actual, expected))
    diff = np.abs(a - b)
    tol = np.maximum(np.abs(a), np.abs(b)) * 2.0 ** -mantissa_bits + 1e-30
    assert np.all(diff <= tol + np.finfo(np.float32).tiny), float(np.max(diff - tol))
    assert np.mean(diff > 0) <= max_fraction, float(np.mean(diff > 0))


@gpu_only
def test_pallas_passes_match_their_plain_jax_reference() -> None:
    rng = np.random.default_rng(3)
    m, f = 256, 384
    h = jnp.asarray(2.0 * rng.standard_normal((m, 2, f)), jnp.bfloat16)
    da = jnp.asarray(rng.standard_normal((m, f)), jnp.bfloat16)
    scale = jnp.float32(4.0 / 448.0)  # some values saturate
    q, qt, amax = S.swiglu_quantize(h, scale, impl="pallas")
    q_ref, _, amax_ref = S.swiglu_quantize(h, scale, impl="xla")
    assert q.dtype == F.E4M3 and q.shape == (m, f) and qt.shape == (f, m)
    np.testing.assert_array_equal(np.asarray(qt, np.float32), np.asarray(q, np.float32).T)
    _ulp_close(q, q_ref, 3, 2e-3)
    np.testing.assert_allclose(float(amax), float(amax_ref), rtol=1e-4)
    # Jitted, the plain-JAX pass still quantizes the float32 SwiGLU (no rounding for XLA to drop).
    q_jit, _, _ = jax.jit(functools.partial(S.swiglu_quantize, impl="xla"))(h, scale)
    _ulp_close(q, q_jit, 3, 2e-3)
    gscale = jnp.float32(1.0 / 57344.0)
    qd, qdt, damax = S.swiglu_backward_quantize(h, da, gscale, jnp.bfloat16, impl="pallas")
    qd_ref, _, damax_ref = S.swiglu_backward_quantize(h, da, gscale, jnp.bfloat16, impl="xla")
    assert qd.dtype == F.E5M2 and qd.shape == (m, 2, f) and qdt.shape == (2, f, m)
    np.testing.assert_array_equal(np.asarray(qdt, np.float32), np.transpose(np.asarray(qd, np.float32), (1, 2, 0)))
    _ulp_close(qd, qd_ref, 2, 2e-3)
    np.testing.assert_allclose(float(damax), float(damax_ref), rtol=1e-2)
    with pytest.raises(ValueError, match="need tokens"):
        S.swiglu_quantize(h[:100], scale, impl="pallas")


def test_fused_mlp_rejects_bad_arguments() -> None:
    x, w_gu, w_dn, _, st_gu, st_dn = _operands()
    with pytest.raises(ValueError, match="impl"):
        S.fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn, impl="cuda")
    with pytest.raises(ValueError, match="needs x"):
        S.fp8_swiglu_mlp(x, w_gu[:, :1], w_dn, st_gu, st_dn)
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        S.fp8_swiglu_mlp(x.astype(jnp.float16), w_gu, w_dn, st_gu, st_dn)
    with pytest.raises(ValueError, match="delayed-scaling Fp8State"):
        S.fp8_swiglu_mlp(x, w_gu, w_dn, None, None)
    with pytest.raises(ValueError, match="fp8_fused"):
        get_preset("tiny").replace(fp8="current", fp8_fused=True)
    with pytest.raises(ValueError, match="delayed scaling"):
        F.apply_fp8(build_model(get_preset("tiny"), key=jax.random.key(0)), "current", fused=True)


def test_fused_fallbacks_are_logged(caplog) -> None:
    """``fp8_fused`` never falls back silently: a SwiGLU whose down projection has a bias keeps the
    unfused path with a warning, and on a GPU a shape the kernels do not tile logs the plain-JAX passes."""
    import logging

    from jla.layers.mlp import SwiGLU

    plain = F.apply_fp8(SwiGLU(64, 128, key=jax.random.key(0), std=0.02), "delayed", fused=True)
    assert isinstance(plain, F.Fp8SwiGLU) and plain.fused
    biased = eqx.tree_at(lambda m: m.down.bias, SwiGLU(64, 128, key=jax.random.key(0), std=0.02),
                         jnp.zeros(64), is_leaf=lambda v: v is None)
    with caplog.at_level(logging.WARNING, logger="jla.layers.fp8"):
        converted = F.apply_fp8(biased, "delayed", fused=True)
    assert isinstance(converted, F.Fp8SwiGLU) and not converted.fused
    assert "keeps the unfused FP8 path" in caplog.text
    if ON_GPU:
        caplog.clear()
        S._LOGGED.clear()
        with caplog.at_level(logging.WARNING, logger="jla.layers.fp8_swiglu"):
            q, _, _ = S.swiglu_quantize(jnp.ones((100, 2, 128), jnp.bfloat16), jnp.float32(1.0))
        assert q.shape == (100, 128) and "plain-JAX SwiGLU passes" in caplog.text


def _batches(steps: int, batch: int = 8, seq: int = 32, vocab: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        ids = [rng.integers(0, vocab, size=(1, batch, 1))]
        for _ in range(seq):
            ids.append((5 * ids[-1] + 3) % vocab)
        ids = np.concatenate(ids, axis=-1).astype(np.int32)
        out.append({"input_ids": jnp.asarray(ids[..., :-1]), "labels": jnp.asarray(ids[..., 1:])})
    return out


def _train(cfg, batches):
    model = build_model(cfg, key=jax.random.key(0))
    optimizer = optax.adamw(3e-3, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step = build_train_step(optimizer, lm_loss(LossConfig(compute_dtype=cfg.compute_dtype)), donate=False)
    key, losses = jax.random.key(1), []
    for batch in batches:
        model, opt_state, metrics, key = step(model, opt_state, batch, key)
        losses.append(float(metrics["loss"]))
    return model, np.array(losses)


def test_fused_model_trains_like_the_unfused_model_and_bf16() -> None:
    base = get_preset("tiny")  # 8 x 32 tokens and d_ff 128: the Pallas passes tile on a GPU
    batches = _batches(30)
    _, bf16 = _train(base, batches)
    unfused_model, unfused = _train(base.replace(fp8="delayed"), batches)
    fused_cfg = base.replace(fp8="delayed", fp8_fused=True)
    model = build_model(fused_cfg, key=jax.random.key(0))
    mlps = [m for m in F.fp8_modules(model) if isinstance(m, F.Fp8SwiGLU)]
    assert mlps and all(m.fused for m in mlps)
    # Same parameters and state leaves as the unfused model: checkpoints are interchangeable.
    paths = lambda mdl: [(jax.tree_util.keystr(p), v.shape) for p, v in  # noqa: E731
                         jax.tree_util.tree_flatten_with_path(eqx.filter(mdl, eqx.is_array))[0]]
    assert paths(model) == paths(build_model(base.replace(fp8="delayed"), key=jax.random.key(0)))
    fused_model, fused = _train(fused_cfg, batches)
    assert bf16[-1] < bf16[0] - 2.0
    assert np.all(np.abs(fused - bf16) <= 0.05 + 0.01 * bf16), (np.round(fused, 3), np.round(bf16, 3))
    assert np.all(np.abs(fused - unfused) <= 0.02 + 0.005 * unfused), (np.round(fused, 3), np.round(unfused, 3))
    for a, b in zip(F.fp8_modules(fused_model), F.fp8_modules(unfused_model), strict=True):
        for name in ("input", "kernel", "grad"):
            ha, hb = (np.asarray(getattr(m.state, f"{name}_amax_history")) for m in (a, b))
            assert np.all(ha[..., :30] > 0) and np.all(ha[..., 30:] == 0)
            if name == "kernel":  # step 1 (the oldest entry) saw the same weights
                np.testing.assert_array_equal(ha[..., 29], hb[..., 29])
            elif name == "input":
                # Same forward pass on the same weights, up to the last bits of the FP8 GEMMs (fast
                # accumulation, GEMM shapes [M, D] vs [B, T, D]) and of the SwiGLU, which move a later
                # layer's input amax by a few bf16 units. Gradient amaxes are not compared: at step 1
                # every scale is 1, so e5m2 flushes most of the (tiny) loss gradients and those last
                # bits decide which survive.
                np.testing.assert_allclose(ha[..., 29], hb[..., 29], rtol=5e-2)
    leaves = jax.tree.leaves(without_fp8_state(eqx.filter(fused_model, eqx.is_array)))
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in leaves)


def test_fused_fp8_state_survives_checkpoint_and_resume(tmp_path) -> None:
    """``fp8_fused`` + ``fp8_min_features`` through ``Trainer``: 4 steps equal 2 steps + resume + 2 steps."""
    from _toy_model import experiment
    from jla.training.trainer import Trainer

    def run(path, steps):
        model = get_preset("tiny").replace(fp8="delayed", fp8_amax_history=16, fp8_fused=True, fp8_min_features=48)
        cfg = experiment(str(path), model=model, train={"max_steps": steps, "log_every": 1, "final_eval": False},
                         optim={"schedule_steps": 4})
        return Trainer(cfg).run()

    full = run(tmp_path / "full", 4)
    first = run(tmp_path / "split", 2)
    second = run(tmp_path / "split", 4)
    assert first.step == 2 and second.step == 4
    modules = F.fp8_modules(full.model)
    assert any(isinstance(m, F.Fp8SwiGLU) and m.fused for m in modules)
    for module in modules:
        hist = np.asarray(module.state.grad_amax_history)
        assert np.all(np.count_nonzero(hist, axis=-1) == 4)  # the resumed run restored the first two
    exact = jax.default_backend() == "cpu"
    same = np.testing.assert_array_equal if exact else (lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5))
    for expected, actual in zip(jax.tree.leaves(eqx.filter(full.model, eqx.is_array)),
                                jax.tree.leaves(eqx.filter(second.model, eqx.is_array)), strict=True):
        same(np.asarray(expected), np.asarray(actual))


def test_fused_fp8_state_is_independent_of_the_device_layout(tmp_path) -> None:
    """The fused MLP under fsdp=4 and dp=2 x fsdp=2 (4 CPU devices) matches one device, like the unfused path."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import test_fp8_multidevice as multi

    old = 'fp8="delayed", fp8_amax_history=8)'
    assert old in multi.SCRIPT
    script = tmp_path / "fp8_fused_multidevice.py"
    script.write_text(multi.SCRIPT.replace(old, 'fp8="delayed", fp8_amax_history=8, fp8_fused=True)'))
    root = Path(multi.__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "tests")))}
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "runs")], env=env, capture_output=True,
                            text=True, timeout=900, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}"
    assert "FP8MULTI " in result.stdout
