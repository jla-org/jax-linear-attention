# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The forward-gradient LM-head cross entropy (``linear_cross_entropy_objective``).

Values and gradients of ``loss_sum + z_loss_weight * z_loss_sum`` against the XLA op and a
naive float32 full-logits oracle: label smoothing, z-loss, ignored tokens, several and uneven
row blocks, padded rows and vocabularies (including very negative logits), float16, an
all-ignored batch, cotangent scaling, the compiled memory of the row blocks, ``lm_loss``
through a whole model, and the dispatch (XLA fallback, fail-closed multi-device calls,
configuration). Kernel tests need an H100; the dispatch tests run anywhere.
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

from jla.ops.cross_entropy import cross_entropy_from_logits, linear_cross_entropy, linear_cross_entropy_objective
from jla.ops.mosaic_gpu._common import is_hopper

ON_HOPPER = is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):  # the float32 oracle must not use TF32
        yield


def _inputs(n, d, v, seed=0, ignore_every=5, dtype=jnp.bfloat16):
    ks = jax.random.split(jax.random.key(seed), 3)
    hidden = jax.random.normal(ks[0], (n, d), jnp.float32).astype(dtype)
    weight = jax.random.normal(ks[1], (v, d), jnp.float32) / np.sqrt(d) * 2.0
    targets = jax.random.randint(ks[2], (n,), 0, v)
    if ignore_every:
        targets = jnp.where(jnp.arange(n) % ignore_every == 0, -100, targets)
    return hidden, weight, targets


def _run(impl, hidden, weight, targets, z_weight=0.0, scale=None, **kw):
    """``(objective, loss_sum, z_loss_sum, num_valid), (dhidden, dweight)`` of ``scale(sums) * objective``.

    ``impl``: ``"naive"`` (f32 full logits), ``"recompute"`` (the XLA op) or an impl of the objective.
    """
    def f(h, w):
        if impl == "naive":
            dtype = kw.get("compute_dtype", jnp.bfloat16)
            out = cross_entropy_from_logits(h.astype(dtype).astype(jnp.float32) @ w.astype(dtype).astype(jnp.float32).T,
                                            targets, label_smoothing=kw.get("label_smoothing", 0.0))
            obj = (out.loss_sum + z_weight * out.z_loss_sum, out.loss_sum, out.z_loss_sum, out.num_valid)
        elif impl == "recompute":  # the XLA op: plain autodiff of the two sums
            out = linear_cross_entropy(h, w, targets, impl="xla", **kw)
            obj = (out.loss_sum + z_weight * out.z_loss_sum, out.loss_sum, out.z_loss_sum, out.num_valid)
        else:
            obj = tuple(linear_cross_entropy_objective(h, w, targets, z_loss_weight=z_weight, impl=impl, **kw))
        factor = 1.0 if scale is None else scale(obj)
        return obj[0] * factor, obj
    (_, obj), grads = jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))(hidden, weight)
    return obj, grads


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _errors(got, want):
    (_, (dh_g, dw_g)), (_, (dh_w, dw_w)) = got, want
    assert dh_g.dtype == dh_w.dtype and dw_g.dtype == dw_w.dtype
    assert dh_g.shape == dh_w.shape and dw_g.shape == dw_w.shape
    assert np.isfinite(np.asarray(dh_g, np.float32)).all() and np.isfinite(np.asarray(dw_g, np.float32)).all()
    return _rel(dh_g.astype(jnp.float32), dh_w.astype(jnp.float32)), _rel(dw_g.astype(jnp.float32),
                                                                           dw_w.astype(jnp.float32))


def _check_values(got, want, rtol=1e-4, atol=1e-3):
    for g, w in zip(got[0], want[0]):
        np.testing.assert_allclose(float(g), float(w), rtol=rtol, atol=atol)


def _check(got, xla, naive, atol=1e-3):
    """Values as the XLA op; gradients near the XLA op and as close to the f32 oracle as it is."""
    _check_values(got, xla, atol=atol)
    _check_values(got, naive, atol=atol)
    dh_x, dw_x = _errors(got, xla)
    assert dh_x < 5e-3 and dw_x < 5e-3, (dh_x, dw_x)
    # The bf16 floor: the XLA op rounds the f32 logit gradient to bf16 once. Ours rounds
    # exp(logit - tile max) to f16 first (2^-11), which adds about 2 % to that error at most.
    (dh_g, dw_g), (dh_ref, dw_ref) = _errors(got, naive), _errors(xla, naive)
    assert dh_g <= 1.05 * dh_ref + 1e-5 and dw_g <= 1.05 * dw_ref + 1e-5, (dh_g, dh_ref, dw_g, dw_ref)


@gpu
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("z_weight", [0.0, 0.5])
def test_matches_xla_op_and_naive(label_smoothing, z_weight):
    hidden, weight, targets = _inputs(384, 192, 1024)
    kw = dict(label_smoothing=label_smoothing)
    got = _run("mosaic_gpu", hidden, weight, targets, z_weight, row_chunk=256, **kw)
    xla = _run("recompute", hidden, weight, targets, z_weight, vocab_chunk=512, **kw)
    naive = _run("naive", hidden, weight, targets, z_weight, **kw)
    _check(got, xla, naive)


@gpu
@pytest.mark.parametrize("n, d, v, row_chunk", [
    (200, 128, 1280, 256),     # rows padded to 256: one block
    (2000, 192, 4096, 768),    # padded to 2048: blocks of 768, 768 and 512
    (512, 768, 4096, 100000),  # the block clipped to the rows
    (2048, 128, 8192, 2048),   # 16 x 32 output tiles: every SM runs several
])
def test_shapes_and_row_blocks(n, d, v, row_chunk):
    hidden, weight, targets = _inputs(n, d, v, seed=1)
    kw = dict(label_smoothing=0.1)
    got = _run("mosaic_gpu", hidden, weight, targets, 0.25, row_chunk=row_chunk, **kw)
    xla = _run("recompute", hidden, weight, targets, 0.25, vocab_chunk=1024, **kw)
    naive = _run("naive", hidden, weight, targets, 0.25, **kw)
    _check(got, xla, naive)
    # Every row's gradient matches (a block or tile stored to the wrong rows would fail here).
    dh_g, dh_x = (np.asarray(r[1][0].astype(jnp.float32)) for r in (got, xla))
    row_err = np.linalg.norm(dh_g - dh_x, axis=1) / np.maximum(np.linalg.norm(dh_x, axis=1), 1e-30)
    assert row_err.max() < 5e-2


@gpu
@pytest.mark.parametrize("v", [50304, 1000])  # a warpgroup tile of padding only; 24 padded columns in a tile
def test_padded_vocabulary(v):
    hidden, weight, targets = _inputs(320, 128, v, seed=8)
    kw = dict(label_smoothing=0.1)
    got = _run("mosaic_gpu", hidden, weight, targets, 0.1, row_chunk=256, **kw)
    xla = _run("recompute", hidden, weight, targets, 0.1, vocab_chunk=4096, **kw)
    naive = _run("naive", hidden, weight, targets, 0.1, **kw)
    _check(got, xla, naive)


@gpu
def test_padded_vocabulary_negative_logits():
    # Every real logit is about -100, below the exact 0 of a padded column.
    hidden, weight, targets = _inputs(256, 64, 1000, seed=9)
    hidden = hidden.at[:, 0].set(1.0)
    weight = weight.at[:, 0].set(-100.0)
    kw = dict(label_smoothing=0.1)
    got = _run("mosaic_gpu", hidden, weight, targets, 1e-3, **kw)
    xla = _run("recompute", hidden, weight, targets, 1e-3, vocab_chunk=1024, **kw)
    assert float(got[0][2]) > 1e3 * float(got[0][3])  # lse is about -93
    _check_values(got, xla)
    dh, dw = _errors(got, xla)
    assert dh < 5e-3 and dw < 5e-3, (dh, dw)


@gpu
def test_confident_targets():
    # Peaked softmax (p(target) near 1): the target column's p - 1 cancels, so it is formed from
    # the f32 target logit and not from the f16 exp tile.
    hidden, weight, targets = _inputs(512, 128, 2048, seed=10, ignore_every=0)
    weight = weight.at[targets].add(4.0 * hidden.astype(jnp.float32) / np.sqrt(128))
    got = _run("mosaic_gpu", hidden, weight, targets, 1e-4)
    xla = _run("recompute", hidden, weight, targets, 1e-4, vocab_chunk=1024)
    naive = _run("naive", hidden, weight, targets, 1e-4)
    assert float(got[0][1]) / 512 < 0.5  # mean CE: most targets are confident
    # The loss is lse - target logit with both near 45: as in impl="mosaic_gpu", the target logit is a
    # row dot product and lse comes from the GEMM, so each token's CE is exact to about 1e-5 (f32 at 45).
    _check(got, xla, naive, atol=512 * 2e-5)


@gpu
def test_cotangent_scaling_and_eval():
    hidden, weight, targets = _inputs(512, 128, 2048, seed=3)
    mean = lambda obj: 3.0 / obj[3]  # noqa: E731  (the train step divides by the token count)
    got = _run("mosaic_gpu", hidden, weight, targets, 0.1, scale=mean, label_smoothing=0.05)
    xla = _run("recompute", hidden, weight, targets, 0.1, scale=mean, label_smoothing=0.05)
    naive = _run("naive", hidden, weight, targets, 0.1, scale=mean, label_smoothing=0.05)
    _check(got, xla, naive)
    # Without differentiation only the statistics kernel runs; the values agree.
    plain = jax.jit(lambda h, w: tuple(linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.1,
                                                                      label_smoothing=0.05, impl="mosaic_gpu")))
    for a, b in zip(plain(hidden, weight), got[0]):
        np.testing.assert_allclose(float(a), float(b), rtol=1e-6)


@gpu
def test_sums_carry_no_gradient():
    hidden, weight, targets = _inputs(256, 64, 512, seed=4)
    for impl in ("mosaic_gpu", "xla"):
        f = lambda h, w: linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.1,  # noqa: E731
                                                        impl=impl).loss_sum
        dh, dw = jax.jit(jax.grad(f, argnums=(0, 1)))(hidden, weight)
        assert not np.asarray(dh.astype(jnp.float32)).any() and not np.asarray(dw).any()


@gpu
def test_all_ignored_float16_and_auto():
    hidden, weight, targets = _inputs(128, 128, 512, seed=5)
    obj, (dh, dw) = _run("mosaic_gpu", hidden, weight, jnp.full_like(targets, -100), 0.3)
    assert all(float(x) == 0.0 for x in obj)
    assert not np.asarray(dh.astype(jnp.float32)).any() and not np.asarray(dw).any()
    auto = _run("auto", hidden, weight, targets, 0.3)
    fused = _run("mosaic_gpu", hidden, weight, targets, 0.3)
    for a, b in zip(jax.tree.leaves(auto), jax.tree.leaves(fused)):
        np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
    h16, w16, t16 = _inputs(256, 128, 512, seed=6, dtype=jnp.float16)
    got = _run("mosaic_gpu", h16, w16, t16, 0.1, compute_dtype=jnp.float16)
    xla = _run("recompute", h16, w16, t16, 0.1, compute_dtype=jnp.float16, vocab_chunk=256)
    naive = _run("naive", h16, w16, t16, 0.1, compute_dtype=jnp.float16)
    _check(got, xla, naive)


@gpu
def test_row_blocks_bound_the_memory():
    # One block's f16 exp and bf16 gradient (4 * rows * V bytes) are live at a time.
    n, d, v, rows = 4096, 64, 65536, 1024
    hidden, weight, targets = _inputs(n, d, v)
    f = jax.jit(jax.grad(lambda h, w: linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.1, row_chunk=rows,
                                                                     impl="mosaic_gpu").objective_sum, argnums=(0, 1)))
    temp = f.lower(hidden, weight).compile().memory_analysis().temp_size_in_bytes
    block = 4 * rows * v
    assert temp < 1.6 * block, (temp, block)  # two live blocks would be 2 * block


@gpu
@pytest.mark.parametrize("preset", ["tiny", "tiny-hybrid"])  # tiny-hybrid: Gated DeltaNet and two MTP heads
def test_lm_loss_grad_in_forward_matches_xla(preset):
    import equinox as eqx

    from jla.models import build_model, get_preset
    from jla.training.step import LossConfig, lm_loss

    model_cfg = get_preset(preset)
    if preset != "tiny":  # a float32 trunk: bf16 rounding through the Gated DeltaNet and MTP layers is not compared
        model_cfg = model_cfg.replace(compute_dtype="float32")
    model = build_model(model_cfg, key=jax.random.key(0))
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, model_cfg.vocab_size, size=(2, 97))
    batch = {"input_ids": jnp.asarray(tokens[:, :-1], jnp.int32), "labels": jnp.asarray(tokens[:, 1:], jnp.int32),
             "loss_mask": jnp.asarray(rng.random((2, 96)) > 0.2)}
    results = {}
    for name, cfg in {"xla": LossConfig(label_smoothing=0.1, z_loss=1e-2),
                      "fwd": LossConfig(label_smoothing=0.1, z_loss=1e-2, impl="mosaic_gpu", grad_in_forward=True,
                                        row_chunk=256)}.items():
        loss_fn = lm_loss(cfg)
        f = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, jax.random.key(1)), has_aux=True))
        (objective, aux), grads = f(model)
        results[name] = (float(objective), aux, jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
    (obj_x, aux_x, g_x), (obj_f, aux_f, g_f) = results["xla"], results["fwd"]
    assert float(aux_x["tokens"]) == float(aux_f["tokens"]) > 0
    np.testing.assert_allclose(obj_f, obj_x, rtol=1e-4)
    for key in ("loss_sum", "z_loss_sum", "mtp_loss"):
        np.testing.assert_allclose(float(aux_f[key]), float(aux_x[key]), rtol=1e-4)
    flat = lambda gs: np.concatenate([np.asarray(g, np.float64).ravel() for g in gs])  # noqa: E731
    assert _rel(flat(g_f), flat(g_x)) < 5e-3  # the bf16 model backward amplifies last-bit differences


def test_xla_fallback_and_validation():
    hidden, weight, targets = _inputs(64, 32, 1000)  # D % 64 != 0: the kernels never apply
    with pytest.raises(ValueError, match="mosaic_gpu"):
        linear_cross_entropy_objective(hidden, weight, targets, impl="mosaic_gpu")
    with pytest.raises(ValueError, match="unknown cross entropy impl"):
        linear_cross_entropy_objective(hidden, weight, targets, impl="triton")
    with pytest.raises(ValueError, match="z_loss_weight"):
        linear_cross_entropy_objective(hidden, weight, targets, z_loss_weight=-1.0)
    with pytest.raises(ValueError, match="row_chunk"):
        linear_cross_entropy_objective(hidden, weight, targets, row_chunk=0)

    def grads(fn):
        return jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(hidden, weight)

    zw = 0.3
    for impl in ("auto", "xla"):
        got = grads(lambda h, w: linear_cross_entropy_objective(h, w, targets, z_loss_weight=zw, vocab_chunk=256,
                                                                label_smoothing=0.1, impl=impl).objective_sum)
        want = grads(lambda h, w: (lambda o: o.loss_sum + zw * o.z_loss_sum)(
            linear_cross_entropy(h, w, targets, vocab_chunk=256, label_smoothing=0.1)))
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_config_plumbing():
    from jla.config import TrainConfig
    from jla.training.step import LossConfig, lm_loss

    with pytest.raises(ValueError, match="ce_grad_in_forward"):
        TrainConfig(ce_grad_in_forward=True)  # ce_impl "xla"
    with pytest.raises(ValueError, match="ce_row_chunk"):
        TrainConfig(ce_row_chunk=0)
    cfg = LossConfig.from_train_config(TrainConfig(ce_impl="auto", ce_grad_in_forward=True, ce_row_chunk=4096))
    assert cfg.grad_in_forward and cfg.row_chunk == 4096 and cfg.impl == "auto"
    assert not LossConfig.from_train_config(TrainConfig()).grad_in_forward
    with pytest.raises(ValueError, match="grad_in_forward"):
        lm_loss(LossConfig(grad_in_forward=True))  # impl "xla"
    with pytest.raises(ValueError, match="grad_in_forward"):
        lm_loss(LossConfig(grad_in_forward=True, impl="auto"), cross_entropy=linear_cross_entropy)
    with pytest.raises(ValueError, match="ce_fp8"):
        TrainConfig(ce_impl="auto", ce_fp8=True)  # without ce_grad_in_forward
    assert LossConfig.from_train_config(TrainConfig(ce_impl="auto", ce_grad_in_forward=True, ce_fp8=True)).fp8
    with pytest.raises(ValueError, match="fp8"):
        lm_loss(LossConfig(fp8=True, impl="auto"))


MULTI_DEVICE_SCRIPT = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from jla.ops.cross_entropy import linear_cross_entropy, linear_cross_entropy_objective

    assert jax.device_count() == 4, jax.devices()
    mesh = jax.make_mesh((4,), ("data",), axis_types=(AxisType.Auto,))
    ks = jax.random.split(jax.random.key(0), 3)
    hidden = jax.random.normal(ks[0], (256, 64), jnp.float32)
    weight = jax.random.normal(ks[1], (512, 64), jnp.float32) / 8.0
    targets = jax.random.randint(ks[2], (256,), 0, 512)
    rows = NamedSharding(mesh, P("data"))
    hidden, targets = jax.device_put(hidden, rows), jax.device_put(targets, rows)
    try:
        jax.jit(lambda h, w, t: linear_cross_entropy_objective(h, w, t, impl="mosaic_gpu").objective_sum)(
            hidden, weight, targets)
    except ValueError as err:
        assert "devices are visible" in str(err) or "sm_90" in str(err), err
    else:
        raise AssertionError("impl='mosaic_gpu' must raise on a multi-device call")

    def grads(fn):
        return jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(hidden, weight)

    want = grads(lambda h, w: (lambda o: o.loss_sum + 0.2 * o.z_loss_sum)(
        linear_cross_entropy(h, w, targets, vocab_chunk=256)))
    for mesh_context in (False, True):  # with a context mesh too: no kernels here, so no per-shard call
        def run():
            return grads(lambda h, w: linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.2,
                                                                     vocab_chunk=256, impl="auto").objective_sum)
        if mesh_context:
            with jax.set_mesh(mesh):
                got = run()
        else:
            got = run()
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    print("MULTI_DEVICE_OK")
    """
)


def test_multi_device_calls_fall_back(tmp_path: Path) -> None:
    # Several visible devices outside a fully manual shard_map: "auto" is the XLA op with plain
    # autodiff, bit for bit, and "mosaic_gpu" raises.
    script = tmp_path / "multi_device.py"
    script.write_text(MULTI_DEVICE_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600,
                            check=False)
    assert result.returncode == 0 and "MULTI_DEVICE_OK" in result.stdout, (
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}")


@pytest.mark.skipif(not ON_HOPPER or jax.device_count() < 2, reason="needs two H100s")
def test_per_shard_under_a_mesh():
    # Under a context mesh (as the trainer sets it) the call runs per shard in a shard_map: rows over
    # the mesh, the weight replicated with its gradient summed by the transpose, partial sums added.
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    hidden, weight, targets = _inputs(1024, 128, 4096, seed=12)
    rows = NamedSharding(mesh, P("data"))
    hidden, targets = jax.device_put(hidden, rows), jax.device_put(targets, rows)
    weight = jax.device_put(weight, NamedSharding(mesh, P()))

    def objective(h, w):
        out = linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.1, label_smoothing=0.1, row_chunk=256)
        return out.objective_sum, out

    f = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
    with jax.set_mesh(mesh):
        assert "mosaic_gpu" in f.lower(hidden, weight).as_text()  # the kernels run, per shard
        (_, got), (dh, dw) = f(hidden, weight)
    xla = _run("recompute", hidden, weight, targets, 0.1, label_smoothing=0.1, vocab_chunk=1024)
    naive = _run("naive", hidden, weight, targets, 0.1, label_smoothing=0.1)
    _check(((got.objective_sum, got.loss_sum, got.z_loss_sum, got.num_valid), (dh, dw)), xla, naive)


@gpu
@pytest.mark.parametrize("z_weight", [0.0, 0.1])
def test_fp8_gradient_gemms(z_weight):
    # The two gradient GEMMs in FP8 (e5m2 logit gradient, e4m3 hidden and weight): the forward and
    # the loss are those of the bf16 path; the gradients carry FP8 rounding (2-3 mantissa bits).
    hidden, weight, targets = _inputs(1024, 256, 4096, seed=13)
    kw = dict(label_smoothing=0.1, row_chunk=512)
    got = _run("mosaic_gpu", hidden, weight, targets, z_weight, fp8=True, **kw)
    bf16 = _run("mosaic_gpu", hidden, weight, targets, z_weight, **kw)
    naive = _run("naive", hidden, weight, targets, z_weight, label_smoothing=0.1)
    for a, b in zip(got[0], bf16[0]):
        assert float(a) == float(b)
    dh, dw = _errors(got, naive)
    dh_bf16, dw_bf16 = _errors(bf16, naive)
    print(f"fp8 gradient errors vs f32: dhidden {dh:.3e} dweight {dw:.3e} (bf16: {dh_bf16:.3e} {dw_bf16:.3e})")
    assert dh < 6e-2 and dw < 6e-2, (dh, dw)


def _collective_lines(hlo: str, op: str) -> list[str]:
    """Instruction lines of the compiled HLO that run collective ``op`` (synchronous or async start)."""
    import re

    pattern = re.compile(rf"\b{op}(-start)?\(")
    return [line for line in hlo.splitlines() if pattern.search(line)]


def _has_array(line: str, dtype: str, elements: int) -> bool:
    """Whether an HLO line names a ``dtype`` array of ``elements`` elements (in any layout or shape)."""
    import math
    import re

    for dims in re.findall(rf"\b{dtype}\[([0-9,]+)\]", line):
        if math.prod(int(d) for d in dims.split(",")) == elements:
            return True
    return False


@pytest.mark.skipif(not ON_HOPPER or jax.device_count() < 2, reason="needs two H100s")
def test_per_shard_fsdp_weight_gathers_bf16_and_reduce_scatters_its_fp32_gradient():
    # The float32 weight split over the mesh along its columns, as FSDP shards the LM head ("embed" ->
    # fsdp): the per-shard call all-gathers it in bf16 (the kernels' compute dtype, so the same values)
    # and reduce-scatters its float32 gradient over the columns. No float32 all-gather of the weight and no
    # all-reduce of its [V, D] gradient remain; values and gradients as the XLA op and the f32 oracle.
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((2,), ("fsdp",), axis_types=(AxisType.Auto,), devices=jax.devices()[:2])
    hidden, weight, targets = _inputs(1024, 256, 4096, seed=21)
    rows = NamedSharding(mesh, P("fsdp"))
    hidden, targets = jax.device_put(hidden, rows), jax.device_put(targets, rows)
    cols = jax.device_put(weight, NamedSharding(mesh, P(None, "fsdp")))

    def objective(h, w):
        out = linear_cross_entropy_objective(h, w, targets, z_loss_weight=0.1, label_smoothing=0.1, row_chunk=256)
        return out.objective_sum, out

    f = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
    with jax.set_mesh(mesh):
        hlo = f.lower(hidden, cols).compile().as_text()
        (_, got), (dh, dw) = f(hidden, cols)
    assert "mosaic_gpu" in hlo  # the kernels run, per shard
    gathers = _collective_lines(hlo, "all-gather")
    whole, half = 4096 * 256, 4096 * 128  # elements of the weight and of one device's columns
    assert any(_has_array(line, "bf16", whole) for line in gathers), gathers
    assert not any(_has_array(line, "f32", whole) for line in gathers), gathers
    assert not any(_has_array(line, "f32", whole) for line in _collective_lines(hlo, "all-reduce"))
    scatters = _collective_lines(hlo, "reduce-scatter")
    assert any(_has_array(line, "f32", half) for line in scatters), scatters
    assert dw.dtype == jnp.float32 and dw.shape == weight.shape
    xla = _run("recompute", hidden, weight, targets, 0.1, label_smoothing=0.1, vocab_chunk=1024)
    naive = _run("naive", hidden, weight, targets, 0.1, label_smoothing=0.1)
    _check(((got.objective_sum, got.loss_sum, got.z_loss_sum, got.num_valid), (dh, dw)), xla, naive)
    # The replicated weight of the same values takes the same path: the same values and gradients.
    with jax.set_mesh(mesh):
        (_, rep), (dh_rep, dw_rep) = f(hidden, jax.device_put(weight, NamedSharding(mesh, P())))
    for a, b in zip(rep, got):
        np.testing.assert_allclose(float(a), float(b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dh_rep, np.float32), np.asarray(dh, np.float32), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(dw_rep), np.asarray(dw), rtol=1e-6, atol=1e-9)
    print(f"FSDP_CE gathers={len(gathers)} reduce_scatters={len(_collective_lines(hlo, 'reduce-scatter'))} "
          f"all_reduces={len(_collective_lines(hlo, 'all-reduce'))}")


COLUMN_GRADIENT_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from jla.ops.cross_entropy import _column_gradient_copy, _gather_in_compute_dtype

    mesh = jax.make_mesh((4,), ("fsdp",), axis_types=(AxisType.Auto,))
    k1, k2 = jax.random.split(jax.random.key(0))
    weight = jax.random.normal(k1, (16, 8), jnp.float32)
    hidden = jax.random.normal(k2, (32, 8), jnp.float32)
    spec = dict(mesh=mesh, check_vma=False, axis_names=frozenset({"fsdp"}))

    def f(h, w):  # the bf16 values of the weight in float32 compute
        return jnp.sum(jnp.tanh(h @ w.T))

    def columns(weight, hidden):  # the new path: bf16 copy gathered, float32 gradient reduce-scattered
        def body(h, w_c, w_cols):
            return jnp.reshape(f(h, _column_gradient_copy(w_c, w_cols, ("fsdp",))), (1,))
        w_c = lax.stop_gradient(weight).astype(jnp.bfloat16)
        return jnp.sum(jax.shard_map(body, in_specs=(P("fsdp", None), P(), P(None, "fsdp")), out_specs=P("fsdp"),
                                     **spec)(hidden, w_c, weight))

    def replicated(weight, hidden):  # the previous path: the float32 weight replicated, the transpose sums
        def body(h, w):
            rounded = w + lax.stop_gradient(w.astype(jnp.bfloat16).astype(jnp.float32) - w)
            return jnp.reshape(f(h, rounded), (1,))
        return jnp.sum(jax.shard_map(body, in_specs=(P("fsdp", None), P()), out_specs=P("fsdp"), **spec)(hidden, weight))

    weight = jax.device_put(weight, NamedSharding(mesh, P(None, "fsdp")))
    hidden = jax.device_put(hidden, NamedSharding(mesh, P("fsdp", None)))
    with jax.set_mesh(mesh):
        got = jax.jit(jax.value_and_grad(columns))(weight, hidden)
        want = jax.jit(jax.value_and_grad(replicated))(weight, hidden)
    assert float(got[0]) == float(want[0]), (float(got[0]), float(want[0]))
    assert got[1].dtype == jnp.float32 and got[1].shape == (16, 8)
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), rtol=1e-6, atol=1e-6)
    assert _gather_in_compute_dtype(weight, jnp.bfloat16, 4)
    assert not _gather_in_compute_dtype(weight, jnp.float32, 4)  # nothing to save
    assert not _gather_in_compute_dtype(weight.astype(jnp.bfloat16), jnp.bfloat16, 4)
    assert not _gather_in_compute_dtype(weight, jnp.bfloat16, 3)  # 8 columns do not split over 3
    print("COLUMN_GRADIENT_OK")
    """
)


def test_column_gradient_copy_matches_the_replicated_weight_on_cpu(tmp_path: Path) -> None:
    # The custom VJP behind the per-shard objective's bf16 gather, without the kernels: on 4 CPU devices the
    # value equals, and the float32 gradient matches, the replicated float32 weight whose gradient the
    # shard_map transpose sums; and when the bf16 gather applies.
    script = tmp_path / "column_gradient.py"
    script.write_text(COLUMN_GRADIENT_SCRIPT)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600,
                            check=False)
    assert result.returncode == 0 and "COLUMN_GRADIENT_OK" in result.stdout, (
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-8000:]}")
