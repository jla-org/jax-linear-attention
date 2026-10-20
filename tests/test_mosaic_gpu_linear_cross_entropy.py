# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused Mosaic GPU linear cross entropy against the XLA op and a naive full-logits oracle.

Values and gradients (``hidden`` and ``weight``) with ignored tokens, label smoothing and
z-loss; several and uneven backward vocabulary chunks; row counts that need padding; padded
vocabularies (50,304 and 1,000 columns, very negative logits); hidden sizes below the
pipeline depth; bfloat16 and float16; an all-ignored batch; and the ``impl`` dispatch
(``"mosaic_gpu"`` raises and ``"auto"`` falls back where the kernels do not apply, including
calls that may span several devices). Kernel tests need an H100; the dispatch tests run anywhere.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.cross_entropy import cross_entropy_from_logits, linear_cross_entropy
from jla.ops.mosaic_gpu._common import is_hopper

ON_HOPPER = is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    # The float32 oracle must not use TF32 matmuls.
    with jax.default_matmul_precision("highest"):
        yield


def _inputs(n, d, v, seed=0, ignore_every=5, dtype=jnp.bfloat16):
    ks = jax.random.split(jax.random.key(seed), 3)
    hidden = jax.random.normal(ks[0], (n, d), jnp.float32).astype(dtype)
    weight = jax.random.normal(ks[1], (v, d), jnp.float32) / np.sqrt(d) * 2.0
    targets = jax.random.randint(ks[2], (n,), 0, v)
    if ignore_every:
        targets = jnp.where(jnp.arange(n) % ignore_every == 0, -100, targets)
    return hidden, weight, targets


def _objective(impl, z_weight, **kw):
    def f(hidden, weight, targets):
        if impl == "naive":
            h = hidden.astype(kw.get("compute_dtype", jnp.bfloat16)).astype(jnp.float32)
            w = weight.astype(kw.get("compute_dtype", jnp.bfloat16)).astype(jnp.float32)
            out = cross_entropy_from_logits(h @ w.T, targets, label_smoothing=kw.get("label_smoothing", 0.0))
        else:
            out = linear_cross_entropy(hidden, weight, targets, impl=impl, **kw)
        return out.loss_sum + z_weight * out.z_loss_sum, out
    return f


def _value_and_grads(impl, hidden, weight, targets, z_weight=0.0, **kw):
    f = jax.jit(jax.value_and_grad(_objective(impl, z_weight, **kw), argnums=(0, 1), has_aux=True))
    (_, out), grads = f(hidden, weight, targets)
    return out, grads


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# Against the XLA op the kernels differ only in the exp2 approximation and the f32 summation
# order, which flip a few bf16 roundings of the logit gradient: measured 4e-5 to 3e-4
# (dhidden) and 1e-6 to 1e-5 (dweight) relative. Against the f32 oracle, whose logit gradient
# is not rounded to bf16, both ops are further away.
XLA_TOL = dict(value_rtol=1e-4, dh_rel=1e-3, dw_rel=1e-4)
NAIVE_TOL = dict(value_rtol=1e-4, dh_rel=2e-2, dw_rel=2e-2)


def _check(got, want, *, value_rtol, dh_rel, dw_rel):
    (out_g, (dh_g, dw_g)), (out_w, (dh_w, dw_w)) = got, want
    np.testing.assert_allclose(float(out_g.loss_sum), float(out_w.loss_sum), rtol=value_rtol, atol=1e-3)
    np.testing.assert_allclose(float(out_g.z_loss_sum), float(out_w.z_loss_sum), rtol=value_rtol, atol=1e-3)
    np.testing.assert_equal(float(out_g.num_valid), float(out_w.num_valid))
    assert dh_g.dtype == dh_w.dtype and dw_g.dtype == dw_w.dtype
    assert dh_g.shape == dh_w.shape and dw_g.shape == dw_w.shape
    assert np.isfinite(np.asarray(dh_g, np.float32)).all() and np.isfinite(np.asarray(dw_g, np.float32)).all()
    dh_err = _rel(dh_g.astype(jnp.float32), dh_w.astype(jnp.float32))
    dw_err = _rel(dw_g.astype(jnp.float32), dw_w.astype(jnp.float32))
    assert dh_err < dh_rel and dw_err < dw_rel, (dh_err, dw_err)


@gpu
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("z_weight", [0.0, 0.5])
def test_matches_xla_op_and_naive(label_smoothing, z_weight):
    hidden, weight, targets = _inputs(384, 192, 1024)
    kw = dict(label_smoothing=label_smoothing, vocab_chunk=512)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, z_weight, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, z_weight, **kw)
    naive = _value_and_grads("naive", hidden, weight, targets, z_weight, label_smoothing=label_smoothing)
    _check(got, xla, **XLA_TOL)
    _check(got, naive, **NAIVE_TOL)


@gpu
@pytest.mark.parametrize("n, d, v, chunk", [
    (200, 128, 1280, 512),   # rows padded to 256; chunks 512, 512, 256
    (128, 64, 256, 4096),    # one K step, one CTA column tile, chunk clipped to V
    (512, 768, 4096, 1000),  # chunk rounded up to 1024
])
def test_shapes_and_chunks(n, d, v, chunk):
    hidden, weight, targets = _inputs(n, d, v, seed=1)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 0.25, vocab_chunk=chunk)
    xla = _value_and_grads("xla", hidden, weight, targets, 0.25, vocab_chunk=chunk)
    _check(got, xla, **XLA_TOL)


@gpu
def test_many_tiles_per_sm():
    # 16 x 32 = 512 output tiles in the forward and in the one backward chunk: every SM of the
    # persistent kernels runs several tiles, reusing its pipeline and output buffers.
    hidden, weight, targets = _inputs(2048, 128, 8192, seed=6)
    kw = dict(label_smoothing=0.1, vocab_chunk=8192)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 1e-2, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, 1e-2, **kw)
    _check(got, xla, **XLA_TOL)
    # Every row's gradient matches (a tile stored to the wrong rows would fail here).
    dh_g, dh_x = (np.asarray(g[1][0].astype(jnp.float32)) for g in (got, xla))
    row_err = np.linalg.norm(dh_g - dh_x, axis=1) / np.maximum(np.linalg.norm(dh_x, axis=1), 1e-30)
    assert row_err.max() < 5e-2


@gpu
@pytest.mark.parametrize("cluster", [1, 2])
def test_cluster_sizes(cluster, monkeypatch):
    # Both CTA cluster sizes (2 multicasts each weight block to a pair of row tiles), with
    # 8 x 16 cluster tiles so every cluster runs several, and rows that need padding.
    import dataclasses

    from jla.ops.mosaic_gpu import linear_cross_entropy as fused

    monkeypatch.setattr(fused, "_STATS", dataclasses.replace(fused._STATS, cluster=cluster))
    monkeypatch.setattr(fused, "_GRAD", dataclasses.replace(fused._GRAD, cluster=cluster))
    hidden, weight, targets = _inputs(2000, 192, 4096, seed=7)
    kw = dict(label_smoothing=0.1, vocab_chunk=1536)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 1e-2, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, 1e-2, **kw)
    _check(got, xla, **XLA_TOL)
    dh_g, dh_x = (np.asarray(g[1][0].astype(jnp.float32)) for g in (got, xla))
    row_err = np.linalg.norm(dh_g - dh_x, axis=1) / np.maximum(np.linalg.norm(dh_x, axis=1), 1e-30)
    assert row_err.max() < 5e-2


@gpu
def test_backward_keeps_one_gradient_chunk_live():
    n, d, v, chunk = 2048, 64, 65536, 2048
    hidden, weight, targets = _inputs(n, d, v)
    f = jax.jit(jax.grad(_objective("mosaic_gpu", 0.1, vocab_chunk=chunk), argnums=(0, 1), has_aux=True))
    temp = f.lower(hidden, weight, targets).compile().memory_analysis().temp_size_in_bytes
    full_gradient = n * v * 2  # the bf16 [N, V] logit gradient
    assert temp < full_gradient / 4, (temp, full_gradient)


@gpu
def test_large_vocabulary_bf16_weight():
    hidden, weight, targets = _inputs(256, 256, 32768, seed=2, ignore_every=3)
    weight = weight.astype(jnp.bfloat16)
    kw = dict(label_smoothing=0.05, vocab_chunk=8192)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 1e-2, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, 1e-2, **kw)
    _check(got, xla, **XLA_TOL)


@gpu
def test_float16_compute():
    hidden, weight, targets = _inputs(256, 128, 512, seed=3, dtype=jnp.float16)
    kw = dict(compute_dtype=jnp.float16, vocab_chunk=256)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 0.1, **kw)
    naive = _value_and_grads("naive", hidden, weight, targets, 0.1, compute_dtype=jnp.float16)
    _check(got, naive, **NAIVE_TOL)


@gpu
@pytest.mark.parametrize("v, chunk", [
    (50304, 4096),  # the presets' vocabulary: the last 256-column tile is half padding (a whole warpgroup tile)
    (1000, 512),    # 24 padded columns inside the last warpgroup tile; chunks 512 and 512 (488 real)
])
def test_padded_vocabulary(v, chunk):
    hidden, weight, targets = _inputs(320, 128, v, seed=8)
    kw = dict(label_smoothing=0.1, vocab_chunk=chunk)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 0.1, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, 0.1, **kw)
    naive = _value_and_grads("naive", hidden, weight, targets, 0.1, label_smoothing=0.1)
    _check(got, xla, **XLA_TOL)
    _check(got, naive, **NAIVE_TOL)


@gpu
def test_padded_vocabulary_negative_logits():
    # Every real logit is about -100, below the exact 0 of a padded column: the forward must keep
    # padded columns out of the max, and the backward's exp(0 - lse) = e^90 would overflow f32.
    hidden, weight, targets = _inputs(256, 64, 1000, seed=9)
    hidden = hidden.at[:, 0].set(1.0)
    weight = weight.at[:, 0].set(-100.0)
    kw = dict(label_smoothing=0.1, vocab_chunk=1024)
    got = _value_and_grads("mosaic_gpu", hidden, weight, targets, 1e-3, **kw)
    xla = _value_and_grads("xla", hidden, weight, targets, 1e-3, **kw)
    assert float(got[0].z_loss_sum) > 1e3 * float(got[0].num_valid)  # lse is about -93
    _check(got, xla, **XLA_TOL)


@gpu
def test_all_ignored_and_auto():
    hidden, weight, targets = _inputs(128, 128, 512, seed=4)
    none = jnp.full_like(targets, -100)
    out, (dh, dw) = _value_and_grads("mosaic_gpu", hidden, weight, none, 0.3)
    assert float(out.loss_sum) == 0.0 and float(out.z_loss_sum) == 0.0 and float(out.num_valid) == 0.0
    assert not np.asarray(dh.astype(jnp.float32)).any() and not np.asarray(dw).any()
    auto = _value_and_grads("auto", hidden, weight, targets, 0.3)
    fused = _value_and_grads("mosaic_gpu", hidden, weight, targets, 0.3)
    _check(auto, fused, value_rtol=0.0, dh_rel=1e-7, dw_rel=1e-7)


def test_dispatch_rejects_and_falls_back():
    hidden, weight, targets = _inputs(64, 32, 1000)  # D % 64 != 0: kernels never apply
    with pytest.raises(ValueError, match="mosaic_gpu"):
        linear_cross_entropy(hidden, weight, targets, impl="mosaic_gpu")
    with pytest.raises(ValueError, match="unknown cross entropy impl"):
        linear_cross_entropy(hidden, weight, targets, impl="triton")
    auto = linear_cross_entropy(hidden, weight, targets, impl="auto", vocab_chunk=256)
    xla = linear_cross_entropy(hidden, weight, targets, impl="xla", vocab_chunk=256)
    np.testing.assert_array_equal(np.asarray(auto.loss_sum), np.asarray(xla.loss_sum))
    with pytest.raises(ValueError, match="mosaic_gpu"):  # float32 compute is XLA only
        linear_cross_entropy(*_inputs(64, 64, 256), impl="mosaic_gpu", compute_dtype=jnp.float32)
    from jla.ops.mosaic_gpu import linear_cross_entropy as fused

    rows = jax.ShapeDtypeStruct((8, 64), jnp.bfloat16)
    huge = jax.ShapeDtypeStruct((2**24 + 256, 64), jnp.bfloat16)  # target columns no longer exact in f32
    assert "2**24" in fused.unsupported_reason(rows, huge, jnp.bfloat16)


def test_train_config_selects_impl():
    from jla.config import TrainConfig
    from jla.training.step import LossConfig

    assert LossConfig.from_train_config(TrainConfig()).impl == "xla"
    assert LossConfig.from_train_config(TrainConfig(ce_impl="mosaic_gpu")).impl == "mosaic_gpu"


@gpu
def test_lm_loss_gradients_match_xla():
    import equinox as eqx

    from jla.models import build_model, get_preset
    from jla.training.step import LossConfig, lm_loss

    model = build_model(get_preset("tiny"), key=jax.random.key(0))  # vocabulary 512
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 512, size=(2, 97))
    batch = {"input_ids": jnp.asarray(tokens[:, :-1], jnp.int32), "labels": jnp.asarray(tokens[:, 1:], jnp.int32),
             "loss_mask": jnp.asarray(rng.random((2, 96)) > 0.2)}
    results = {}
    for impl in ("xla", "mosaic_gpu"):
        loss_fn = lm_loss(LossConfig(label_smoothing=0.1, z_loss=1e-2, impl=impl))
        f = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, jax.random.key(1)), has_aux=True))
        (objective, aux), grads = f(model)
        results[impl] = (float(objective), float(aux["tokens"]), jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
    (obj_x, tok_x, g_x), (obj_m, tok_m, g_m) = results["xla"], results["mosaic_gpu"]
    assert tok_x == tok_m and 0 < tok_m < 192
    np.testing.assert_allclose(obj_m, obj_x, rtol=1e-4)
    flat = lambda gs: np.concatenate([np.asarray(g, np.float64).ravel() for g in gs])  # noqa: E731
    assert _rel(flat(g_m), flat(g_x)) < 5e-3  # measured 1.7e-3: the bf16 model backward amplifies last-bit differences


MULTI_DEVICE_SCRIPT = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from jla.ops.cross_entropy import linear_cross_entropy
    from jla.ops.mosaic_gpu import linear_cross_entropy as fused

    assert jax.device_count() == 4, jax.devices()
    auto = (AxisType.Auto, AxisType.Auto)
    mesh = jax.make_mesh((2, 2), ("data", "model"), axis_types=auto)
    reason = fused.multi_device_reason()
    assert reason is not None and "4 devices" in reason, reason
    seen = {}

    def body(name):
        def f(x):
            seen[name] = fused.multi_device_reason()
            return x
        return f

    x = jnp.zeros((4, 4))
    jax.jit(jax.shard_map(body("all"), mesh=mesh, in_specs=P("data", "model"), out_specs=P("data", "model")))(x)
    jax.jit(jax.shard_map(body("data"), mesh=mesh, in_specs=P("data"), out_specs=P("data"), axis_names={"data"}))(x)
    one = jax.make_mesh((4, 1), ("data", "model"), axis_types=auto)
    jax.jit(jax.shard_map(body("size1"), mesh=one, in_specs=P("data"), out_specs=P("data"), axis_names={"data"}))(x)
    assert seen["all"] is None and seen["size1"] is None, seen
    assert seen["data"] is not None, seen  # "model" (size 2) stays automatic: XLA could partition the call

    ks = jax.random.split(jax.random.key(0), 3)
    hidden = jax.random.normal(ks[0], (256, 64), jnp.float32)
    weight = jax.random.normal(ks[1], (512, 64), jnp.float32) / 8.0
    targets = jax.random.randint(ks[2], (256,), 0, 512)
    rows = NamedSharding(mesh, P("data"))
    hidden, targets = jax.device_put(hidden, rows), jax.device_put(targets, rows)
    try:
        jax.jit(lambda h, w, t: linear_cross_entropy(h, w, t, impl="mosaic_gpu").loss_sum)(hidden, weight, targets)
    except ValueError as err:
        assert "devices are visible" in str(err), err
    else:
        raise AssertionError("impl='mosaic_gpu' must raise on a multi-device call")

    def grads(impl):
        f = lambda h, w, t: linear_cross_entropy(h, w, t, impl=impl, vocab_chunk=256).loss_sum  # noqa: E731
        return jax.jit(jax.value_and_grad(f, argnums=(0, 1)))(hidden, weight, targets)

    (va, (ha, wa)), (vx, (hx, wx)) = grads("auto"), grads("xla")
    np.testing.assert_array_equal(np.asarray(va), np.asarray(vx))
    np.testing.assert_array_equal(np.asarray(ha), np.asarray(hx))
    np.testing.assert_array_equal(np.asarray(wa), np.asarray(wx))
    print("MULTI_DEVICE_OK")
    """
)


def test_multi_device_calls_fail_closed(tmp_path: Path) -> None:
    # The kernels have no partitioning rule: with several visible devices "auto" must use XLA and
    # "mosaic_gpu" must raise, except inside a shard_map whose mesh axes are all manual.
    script = tmp_path / "multi_device.py"
    script.write_text(MULTI_DEVICE_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600,
                            check=False)
    assert result.returncode == 0 and "MULTI_DEVICE_OK" in result.stdout, (
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}")
