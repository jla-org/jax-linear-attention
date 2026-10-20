# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 fused glue (``SSMConfig.fused_glue``, ``jla.layers.mamba2_fused``): values and every gradient.

The Pallas row kernels run in the interpreter on the CPU against plain autodiff of their formulas (float32),
and compiled on the GPU; the fused layer is compared with the unfused layer (same parameters) on the GPU, with
packed documents, odd lengths and the Mosaic GPU kernels (grouped B/C key gradients included): its bf16 error
against a float32 reference stays within bf16 noise of the unfused layer's; with an FP8 out-projection the fused
layer matches the unfused FP8 layer to FP8 noise. Calls the fused path does not cover (several B/C groups,
a head dim that is not a power of two, decoding, the CPU, ``fused_glue = False``) keep the unfused layer; a head
count that is not a power of two runs the masked row kernels (rows padded to the next power of two of heads).
"""

from __future__ import annotations

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import mamba2_fused as fused
from jla.layers.ssm import Mamba2, _fused_glue_applies
from jla.models.config import SSMConfig


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _gpu() -> bool:
    try:
        return jax.devices()[0].platform == "gpu" and jax.device_count() == 1
    except RuntimeError:
        return False


def _hopper() -> bool:
    return _gpu() and getattr(jax.devices()[0], "compute_capability", "") == "9.0"


def _plain(y, x, z, d, norm_w, out_w, eps=1e-5):
    """The unfused formula in float32: ``(rmsnorm((y + D x) silu(z)) * w) @ W``."""
    width = z.shape[-1]
    u = (y + d[:, None] * x).reshape(*y.shape[:2], width) * jax.nn.silu(z)
    u = u * jax.lax.rsqrt(jnp.mean(u * u, axis=-1, keepdims=True) + eps) * norm_w
    return u @ out_w


def _inputs(seed, batch=2, length=7, heads=4, head_dim=8, d_model=16, dtype=jnp.float32):
    ks = jax.random.split(jax.random.key(seed), 7)
    y = jax.random.normal(ks[0], (batch, length, heads, head_dim), dtype)
    x = jax.random.normal(ks[1], (batch, length, heads, head_dim), dtype)
    z = jax.random.normal(ks[2], (batch, length, heads * head_dim), dtype)
    d = jax.random.normal(ks[3], (heads,))
    norm_w = 1.0 + 0.3 * jax.random.normal(ks[4], (heads * head_dim,))
    out_w = jax.random.normal(ks[5], (heads * head_dim, d_model)) * 0.2
    cot = jax.random.normal(ks[6], (batch, length, d_model))
    return (y, x, z, d, norm_w, out_w), cot


# ------------------------------------------------------------------ the kernels (interpreted on the CPU)


@pytest.mark.parametrize("heads", [4, 3, 6])  # 3 and 6: the masked kernels (padded to 4 and 8 heads)
def test_row_kernels_match_their_formulas(heads):
    (y, x, z, d, _, _), _ = _inputs(0, heads=heads)
    rows = y.shape[0] * y.shape[1]
    y3, x3, z3 = (t.reshape(rows, heads, 8) for t in (y, x, z))

    def ref_a(y3, x3, z3, d):
        g = (y3 + d[:, None] * x3) * jax.nn.silu(z3)
        return g * jax.lax.rsqrt(jnp.mean(g * g, axis=(1, 2), keepdims=True) + 1e-5)

    a, rstd = fused.norm_forward(y3, x3, z3, d, eps=1e-5, dtype=jnp.float32, interpret=True)
    np.testing.assert_allclose(a, ref_a(y3, x3, z3, d), rtol=1e-5, atol=1e-5)
    g = (y3 + d[:, None] * x3) * jax.nn.silu(z3)
    np.testing.assert_allclose(rstd, jax.lax.rsqrt(jnp.mean(g * g, axis=(1, 2)) + 1e-5), rtol=1e-5)
    da = jax.random.normal(jax.random.key(9), a.shape)
    _, vjp = jax.vjp(lambda y3, x3, z3: ref_a(y3, x3, z3, d), y3, x3, z3)
    want_dy, want_dx, want_dz = vjp(da)
    dy, dz, dd_rows = fused.norm_backward(da, y3, x3, z3, d, rstd, interpret=True)
    np.testing.assert_allclose(dy, want_dy, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dz, want_dz, rtol=1e-4, atol=1e-5)
    # dx of the skip is D * dy; dD is the sum of dy * x per head.
    np.testing.assert_allclose(dy * d[:, None], want_dx, rtol=1e-4, atol=1e-5)
    want_dd = jax.vjp(lambda d: ref_a(y3, x3, z3, d), d)[1](da)[0]
    np.testing.assert_allclose(jnp.sum(dd_rows, axis=0), want_dd, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("heads", [4, 3])
def test_gated_norm_out_matches_autodiff_interpreted(heads):
    args, cot = _inputs(1, heads=heads)

    def mine(*a):
        return fused.gated_norm_out(*a, eps=1e-5, compute_dtype=jnp.float32, interpret=True)

    np.testing.assert_allclose(mine(*args), _plain(*args), rtol=1e-5, atol=1e-5)
    got = jax.grad(lambda *a: jnp.sum(mine(*a) * cot), argnums=tuple(range(6)))(*args)
    want = jax.grad(lambda *a: jnp.sum(_plain(*a) * cot), argnums=tuple(range(6)))(*args)
    for name, g, w in zip(("y", "x", "z", "D", "norm_w", "out_w"), got, want):
        np.testing.assert_allclose(g, w, rtol=2e-5, atol=2e-5, err_msg=name)


def test_gated_norm_out_rejects_unsupported_shapes():
    args, _ = _inputs(2, heads=3, head_dim=6)
    with pytest.raises(ValueError, match="power-of-two head_dim"):
        fused.gated_norm_out(*args, eps=1e-5, compute_dtype=jnp.float32, interpret=True)


@pytest.mark.parametrize("heads", [4, 5])
def test_gated_norm_matches_autodiff_interpreted(heads):
    (y, x, z, d, norm_w, _), _ = _inputs(4, heads=heads)
    cot = jax.random.normal(jax.random.key(11), z.shape)

    def plain(y, x, z, d, norm_w):
        u = (y + d[:, None] * x).reshape(z.shape) * jax.nn.silu(z)
        return u * jax.lax.rsqrt(jnp.mean(u * u, axis=-1, keepdims=True) + 1e-5) * norm_w

    def mine(*a):
        return fused.gated_norm(*a, eps=1e-5, compute_dtype=jnp.float32, interpret=True)

    args = (y, x, z, d, norm_w)
    np.testing.assert_allclose(mine(*args), plain(*args), rtol=1e-5, atol=1e-5)
    got = jax.grad(lambda *a: jnp.sum(mine(*a) * cot), argnums=tuple(range(5)))(*args)
    want = jax.grad(lambda *a: jnp.sum(plain(*a) * cot), argnums=tuple(range(5)))(*args)
    for name, g, w in zip(("y", "x", "z", "D", "norm_w"), got, want):
        np.testing.assert_allclose(g, w, rtol=2e-5, atol=2e-5, err_msg=name)


# ------------------------------------------------------------------ the kernels compiled on the GPU


@pytest.mark.skipif(not _gpu(), reason="the compiled Pallas Triton kernels need one GPU")
@pytest.mark.parametrize("heads", [8, 24])  # 24 heads of 64: d_model 768, the masked kernels
def test_gated_norm_out_compiled_bf16(heads):
    args32, cot = _inputs(3, batch=2, length=200, heads=heads, head_dim=64, d_model=128)
    y, x, z = (t.astype(jnp.bfloat16) for t in args32[:3])
    args = (y, x, z, *args32[3:])
    ref32 = (y.astype(jnp.float32), x.astype(jnp.float32), z.astype(jnp.float32), *args32[3:])

    def mine(*a):
        return fused.gated_norm_out(*a, eps=1e-5, compute_dtype=jnp.bfloat16)

    def unfused_bf16(y, x, z, d, norm_w, out_w):  # the unfused layer's roundings
        skip = (y.astype(jnp.float32) + d[:, None] * x.astype(jnp.float32)).astype(jnp.bfloat16)
        u = skip.astype(jnp.float32).reshape(*y.shape[:2], -1) * jax.nn.silu(z.astype(jnp.float32))
        u = u * jax.lax.rsqrt(jnp.mean(u * u, axis=-1, keepdims=True) + 1e-5) * norm_w
        return (u.astype(jnp.bfloat16) @ out_w.astype(jnp.bfloat16)).astype(jnp.bfloat16)

    want = _plain(*ref32)
    got_out, base_out = jax.jit(mine)(*args), jax.jit(unfused_bf16)(*args)
    scale = float(jnp.max(jnp.abs(want)))
    err, base_err = (float(jnp.max(jnp.abs(o.astype(jnp.float32) - want))) for o in (got_out, base_out))
    assert err <= 2 * base_err + 1e-2 * scale, (err, base_err, scale)
    grads = [jax.jit(jax.grad(lambda *a, f=f: jnp.sum(f(*a).astype(jnp.float32) * cot), argnums=tuple(range(6))))(*args)
             for f in (mine, unfused_bf16)]
    ref = jax.grad(lambda *a: jnp.sum(_plain(*a) * cot), argnums=tuple(range(6)))(*ref32)
    for name, g, b, w in zip(("y", "x", "z", "D", "norm_w", "out_w"), grads[0], grads[1], ref):
        w = np.asarray(w, np.float32)
        e = float(np.max(np.abs(np.asarray(g, np.float32) - w)))
        eb = float(np.max(np.abs(np.asarray(b, np.float32) - w)))
        assert e <= 2 * eb + 2e-2 * float(np.max(np.abs(w))), (name, e, eb)


# ------------------------------------------------------------------ the layer, fused vs unfused


def _randomize(module, seed):
    """Random values for every float array except ``A_log`` / ``dt_bias`` (their inits are meaningful)."""
    keep = {name: getattr(module, name) for name in ("A_log", "dt_bias")}
    leaves, treedef = jax.tree.flatten(module)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [0.3 * jax.random.normal(k, x.shape, x.dtype) + (1.0 if x.ndim == 1 else 0.0)
              if eqx.is_inexact_array(x) else x for k, x in zip(keys, leaves)]
    module = jax.tree.unflatten(treedef, leaves)
    for name, value in keep.items():
        module = eqx.tree_at(lambda m, name=name: getattr(m, name), module, value)
    return module


def _layer(*, fused_glue, d_model=128, head_dim=64, d_state=64, n_groups=1, impl="mosaic_gpu",
           compute_dtype="bfloat16", seed=0, grouped_key_grads=False):
    cfg = SSMConfig(d_state=d_state, expand=2, head_dim=head_dim, n_groups=n_groups,
                    chunk_size=64 if impl == "mosaic_gpu" else 16, implementation=impl, fused_glue=fused_glue,
                    grouped_key_grads=grouped_key_grads)
    layer = Mamba2(d_model, cfg, key=jax.random.key(seed), std=0.2, norm_eps=1e-5, compute_dtype=compute_dtype)
    return _randomize(layer, seed + 50)


def _segments(batch, length, seed):
    """Packed documents: 1-3 boundaries per row at random positions."""
    rng = np.random.default_rng(seed)
    ids = np.zeros((batch, length), np.int32)
    for b in range(batch):
        for cut in sorted(rng.choice(np.arange(1, length), size=rng.integers(1, 4), replace=False)):
            ids[b, cut:] += 1
    return jnp.asarray(ids)


def _value_and_grads(layer, u, segment_ids, cot):
    params, static = eqx.partition(layer, eqx.is_inexact_array)

    @jax.jit
    def run(params, u):
        def loss(params, u):
            return jnp.sum(eqx.combine(params, static)(u, segment_ids=segment_ids).astype(jnp.float32) * cot)

        out = eqx.combine(params, static)(u, segment_ids=segment_ids)
        return out, jax.grad(loss, argnums=(0, 1))(params, u)

    out, (g_params, g_u) = run(params, u)
    return out, g_params, g_u


def _flat(tree):
    return {jax.tree_util.keystr(path): np.asarray(leaf, np.float32)
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree) if leaf is not None}


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU SSD kernels and the fused glue need one H100")
@pytest.mark.parametrize("packed,length,grouped,d_model", [(True, 200, True, 128), (False, 256, False, 128),
                                                            (True, 131, True, 128), (True, 131, True, 96),
                                                            (False, 200, False, 96)])
def test_fused_layer_matches_unfused_on_gpu(packed, length, grouped, d_model):
    """``d_model`` 96: three heads of 64 (the masked row kernels)."""
    batch = 2
    u = jax.random.normal(jax.random.key(3), (batch, length, d_model))
    seg = _segments(batch, length, 4) if packed else None
    cot = jax.random.normal(jax.random.key(5), (batch, length, d_model))
    shape = dict(d_model=d_model, grouped_key_grads=grouped)
    ref = _value_and_grads(_layer(fused_glue=False, impl="chunked", compute_dtype="float32", **shape), u, seg, cot)
    ref = (np.asarray(ref[0]), _flat(ref[1]), np.asarray(ref[2]))
    fused_layer = _layer(fused_glue=True, **shape)
    assert _fused_glue_applies(fused_layer)
    errs = {}
    for name, layer in (("fused", fused_layer), ("unfused", _layer(fused_glue=False, **shape))):
        out, gp, gu = _value_and_grads(layer, u.astype(jnp.bfloat16), seg, cot)
        e = {"out": float(np.max(np.abs(np.asarray(out, np.float32) - ref[0]))),
             "u": float(np.max(np.abs(np.asarray(gu, np.float32) - ref[2])))}
        for key, value in _flat(gp).items():
            e[key] = float(np.max(np.abs(value - ref[1][key])))
        errs[name] = e
    scales = {"out": float(np.max(np.abs(ref[0]))), "u": float(np.max(np.abs(ref[2])))}
    scales.update({key: float(np.max(np.abs(v))) for key, v in ref[1].items()})
    for key, scale in scales.items():
        f, b = errs["fused"][key], errs["unfused"][key]
        # bf16 noise: the fused error stays within 2x the unfused layer's (plus a floor of 2% of the scale).
        assert f <= 2.0 * b + 0.02 * scale, f"{key}: fused err {f:.3e}, unfused err {b:.3e}, scale {scale:.3e}"


def test_fused_path_applies_only_where_it_fuses():
    from jla.layers.fp8 import apply_fp8

    on = _layer(fused_glue=True, impl="chunked")
    assert _fused_glue_applies(on) == _gpu()
    assert not _fused_glue_applies(_layer(fused_glue=False, impl="chunked"))
    assert not _fused_glue_applies(_layer(fused_glue=True, impl="chunked", n_groups=2))
    assert _fused_glue_applies(apply_fp8(on, "delayed")) == _gpu()  # an FP8 out-projection runs as its module
    assert not _fused_glue_applies(_layer(fused_glue=True, impl="chunked", d_model=96, head_dim=48))
    assert _fused_glue_applies(_layer(fused_glue=True, impl="chunked", d_model=96)) == _gpu()  # 3 heads of 64
    # Decoding keeps the unfused path, bit for bit (same seed: same parameters).
    off = _layer(fused_glue=False, impl="chunked")
    assert dataclasses.replace(on.cfg, fused_glue=False) == off.cfg
    u = jax.random.normal(jax.random.key(7), (1, 5, 128))
    cache = on.init_cache(1, jnp.float32)
    got, _ = on.step(u, cache)
    want, _ = off.step(u, cache)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU SSD kernels and the fused glue need one H100")
def test_fused_layer_with_fp8_out_projection_on_gpu():
    """FP8 (delayed) layer: the fused glue feeding the FP8 out-projection module against the unfused FP8 layer.
    FP8 roundings (E4M3 activations, E5M2 gradients) flip on bf16-level input differences, so the bound is a
    relative L2 error per tensor."""
    from jla.layers.fp8 import apply_fp8

    batch, length, d_model = 2, 200, 128
    u = jax.random.normal(jax.random.key(3), (batch, length, d_model), jnp.bfloat16)
    seg = _segments(batch, length, 4)
    cot = jax.random.normal(jax.random.key(5), (batch, length, d_model))
    layers = [apply_fp8(_layer(fused_glue=flag, grouped_key_grads=True), "delayed", min_features=128)
              for flag in (True, False)]
    assert _fused_glue_applies(layers[0]) and not _fused_glue_applies(layers[1])
    (got_out, got_p, got_u), (want_out, want_p, want_u) = (_value_and_grads(m, u, seg, cot) for m in layers)

    def rel(g, w):
        g, w = np.asarray(g, np.float64).ravel(), np.asarray(w, np.float64).ravel()
        return float(np.linalg.norm(g - w) / (np.linalg.norm(w) or 1.0))

    assert rel(np.asarray(got_out, np.float32), np.asarray(want_out, np.float32)) < 3e-2
    assert rel(np.asarray(got_u, np.float32), np.asarray(want_u, np.float32)) < 5e-2
    for key, w in _flat(want_p).items():
        assert rel(_flat(got_p)[key], w) < 5e-2, (key, rel(_flat(got_p)[key], w))


# ------------------------------------------------------------------ the folded forward (v = dt x in the kernel)


def _ssd_inputs(seed, batch, length, heads, packed):
    ks = jax.random.split(jax.random.key(seed), 6)
    x = jax.random.normal(ks[0], (batch, length, heads, 64), jnp.bfloat16)
    dt = jax.nn.softplus(jax.random.normal(ks[1], (batch, length, heads)) - 2.0)
    b = (0.3 * jax.random.normal(ks[2], (batch, length, 1, 64))).astype(jnp.bfloat16)
    c = (0.3 * jax.random.normal(ks[3], (batch, length, 1, 64))).astype(jnp.bfloat16)
    a = -jnp.exp(jax.random.uniform(ks[4], (heads,), minval=0.0, maxval=2.0))
    v = (x.astype(jnp.float32) * dt[..., None]).astype(jnp.bfloat16)
    seg = _segments(batch, length, seed) if packed else None
    return v, dt * a, b, c, (x, dt), seg


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU SSD kernels need one H100")
@pytest.mark.parametrize("batch,length,heads,packed", [(2, 256, 4, False), (2, 200, 3, True), (1, 131, 8, True),
                                                       (3, 1000, 2, False)])
def test_ssd_core_folded_forward_is_bitwise_on_gpu(batch, length, heads, packed):
    """The forward that forms ``v = dt x`` in the kernel gives the unfolded forward's output bit for bit, and a
    differentiated call keeps the unfolded path: same output and gradients, bit for bit."""
    v, g, b, c, parts, seg = _ssd_inputs(length, batch, length, heads, packed)
    opts = dict(segment_ids=seg, impl="mosaic_gpu", chunk_size=64, grouped_key_grads=True)
    assert fused._fold_applies(v, g, b, c, parts, seg, "mosaic_gpu", 64, True)

    def run(v, g, b, c, x, dt, fold):
        return fused.ssd_core(v, g, b, c, v_parts=(x, dt) if fold else None, **opts)

    from jla.ops.mosaic_gpu import scalar_gate

    folded = jax.jit(functools.partial(run, fold=True))
    lowered = folded.lower(v, g, b, c, *parts).as_text()
    # A split sequence (few long rows, here 3 x 1000 with 2 heads) forms v in XLA for its piece maps.
    assert ("sg_fused_fwd_scaled" in lowered) == (scalar_gate.pieces(batch, heads, 64, length, delta=False) == 1)
    want = jax.jit(functools.partial(run, fold=False))(v, g, b, c, *parts)
    np.testing.assert_array_equal(np.asarray(folded(v, g, b, c, *parts), np.float32), np.asarray(want, np.float32))
    cot = jax.random.normal(jax.random.key(1), want.shape, jnp.float32)

    def loss(fold, v, g, b, c, x, dt):
        v = (x.astype(jnp.float32) * dt[..., None]).astype(jnp.bfloat16)  # the layer's product
        y = run(v, g, b, c, x, dt, fold)
        return jnp.sum(y.astype(jnp.float32) * cot), y

    grads = [jax.jit(jax.value_and_grad(functools.partial(loss, f), argnums=(1, 2, 3, 4, 5), has_aux=True))(
        v, g, b, c, *parts) for f in (True, False)]
    (_, y_f), g_f = grads[0]
    (_, y_u), g_u = grads[1]
    np.testing.assert_array_equal(np.asarray(y_f, np.float32), np.asarray(y_u, np.float32))
    for name, gf, gu in zip(("g", "b", "c", "x", "dt"), g_f, g_u):
        np.testing.assert_array_equal(np.asarray(gf, np.float32), np.asarray(gu, np.float32), err_msg=name)


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU SSD kernels and the fused glue need one H100")
@pytest.mark.parametrize("d_model,length,packed", [(128, 256, False), (96, 131, True)])
def test_fused_layer_forward_matches_its_training_forward_on_gpu(d_model, length, packed):
    """The layer's forward alone (folded SSD) against the forward of a differentiated call: bit for bit."""
    layer = _layer(fused_glue=True, d_model=d_model, grouped_key_grads=True)
    u = jax.random.normal(jax.random.key(3), (2, length, d_model), jnp.bfloat16)
    seg = _segments(2, length, 4) if packed else None
    params, static = eqx.partition(layer, eqx.is_inexact_array)

    def run(params, u):
        return eqx.combine(params, static)(u, segment_ids=seg)

    alone = jax.jit(run)(params, u)
    trained = jax.jit(lambda p, u: jax.vjp(run, p, u)[0])(params, u)
    np.testing.assert_array_equal(np.asarray(alone, np.float32), np.asarray(trained, np.float32))
