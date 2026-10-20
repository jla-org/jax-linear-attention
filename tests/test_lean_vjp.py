# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Memory-lean custom VJPs (jla.layers.lean): same values and gradients as plain autodiff, fewer residuals."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.ad_checkpoint import saved_residuals

from jla.layers import lean
from jla.layers.conv import ShortConv
from jla.layers.linear_attention import _mamba_gate
from jla.layers.mlp import swiglu
from jla.layers.norm import GatedRMSNorm, RMSNorm, rms_norm
from jla.models import build_model, get_preset
from jla.models.config import LinearAttentionConfig, hybrid_pattern

def _keys(n: int, seed: int = 0):
    return jax.random.split(jax.random.key(seed), n)


def _vjp_pair(f_ref, f_lean, args, *, dtype=jnp.float32, seed: int = 1):
    """Outputs and input cotangents of both functions for one random output cotangent."""
    y_ref, vjp_ref = jax.vjp(f_ref, *args)
    y_lean, vjp_lean = jax.vjp(f_lean, *args)
    ct = jax.random.normal(jax.random.key(seed), y_ref.shape, jnp.float32).astype(y_ref.dtype)
    return (y_ref, y_lean), (vjp_ref(ct), vjp_lean(ct))


def _close(a, b, tol):
    a = np.asarray(jnp.asarray(a, jnp.float32))
    b = np.asarray(jnp.asarray(b, jnp.float32))
    scale = max(1.0, float(np.max(np.abs(a))))
    np.testing.assert_allclose(a, b, atol=tol * scale, rtol=tol)


@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)])
@pytest.mark.parametrize("with_weight", [True, False])
def test_rms_norm(dtype, tol, with_weight):
    k1, k2 = _keys(2)
    x = jax.random.normal(k1, (2, 33, 48), jnp.float32).astype(dtype)
    w = 1.0 + 0.1 * jax.random.normal(k2, (48,), jnp.float32)
    if with_weight:
        (y0, y1), (g0, g1) = _vjp_pair(lambda x, w: rms_norm(x, w, 1e-6), lambda x, w: lean.rms_norm(x, w, 1e-6),
                                       (x, w))
    else:
        (y0, y1), (g0, g1) = _vjp_pair(lambda x: rms_norm(x, None, 1e-6), lambda x: lean.rms_norm(x, None, 1e-6),
                                       (x,))
    _close(y0, y1, 1e-6 if y0.dtype == jnp.float32 else 1e-2)
    for a, b in zip(g0, g1):
        assert a.dtype == b.dtype
        _close(a, b, tol)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("group_size", [None, 16])
def test_gated_rms_norm(activation, norm_before_gate, group_size):
    k1, k2, k3 = _keys(3)
    x = jax.random.normal(k1, (2, 17, 3, 32), jnp.float32)
    z = 2.0 * jax.random.normal(k2, (2, 17, 3, 32), jnp.float32)
    ref = GatedRMSNorm(32, eps=1e-5, activation=activation, norm_before_gate=norm_before_gate, group_size=group_size)
    ref = eqx.tree_at(lambda m: m.weight, ref, 1.0 + 0.1 * jax.random.normal(k3, (32,), jnp.float32))
    mod = lean.apply_lean(ref)
    assert type(mod) is lean.LeanGatedRMSNorm

    def run(m):
        return lambda x, z, w: eqx.tree_at(lambda n: n.weight, m, w)(x, z)

    (y0, y1), (g0, g1) = _vjp_pair(run(ref), run(mod), (x, z, ref.weight))
    _close(y0, y1, 1e-6)
    for a, b in zip(g0, g1):
        _close(a, b, 2e-5)


@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-5), (jnp.bfloat16, 3e-2)])
def test_swiglu_linear(dtype, tol):
    k1, k2 = _keys(2)
    h = (2.0 * jax.random.normal(k1, (2, 19, 2, 40), jnp.float32)).astype(dtype)
    w = (0.1 * jax.random.normal(k2, (40, 24), jnp.float32)).astype(dtype)

    def ref(h, w):
        p = swiglu(h)
        return jax.lax.dot_general(p, w, (((p.ndim - 1,), (0,)), ((), ())))

    (y0, y1), (g0, g1) = _vjp_pair(ref, lean.swiglu_linear, (h, w))
    _close(y0, y1, 1e-6 if y0.dtype == jnp.float32 else 1e-2)
    for a, b in zip(g0, g1):
        assert a.dtype == b.dtype
        _close(a, b, tol)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("use_bias,activation", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("with_state", [False, True])
def test_short_conv(packed, use_bias, activation, with_state):
    k1, k2, k3 = _keys(3)
    batch, length, channels, width = 2, 23, 12, 4
    conv = ShortConv(channels, width, activation=activation, use_bias=use_bias, init="uniform", key=k1)
    lean_conv = lean.apply_lean(conv)
    assert type(lean_conv) is lean.LeanShortConv
    x = jax.random.normal(k2, (batch, length, channels), jnp.float32)
    state = jax.random.normal(k3, (batch, width - 1, channels), jnp.float32) if with_state else None
    seg = None
    if packed:
        seg = jnp.asarray(np.repeat(np.array([[0, 1, 2], [5, 5, 6]]), [7, 1, 15], axis=1), jnp.int32)

    def run(m):
        def f(x, w, b, s):
            m2 = eqx.tree_at(lambda n: n.weight, m, w)
            if b is not None:
                m2 = eqx.tree_at(lambda n: n.bias, m2, b)
            y, new_state = m2(x, segment_ids=seg, state=s)
            return y, new_state
        return f

    args = (x, conv.weight, conv.bias, state)
    (y0, s0), vjp0 = jax.vjp(run(conv), *args)
    (y1, s1), vjp1 = jax.vjp(run(lean_conv), *args)
    np.testing.assert_array_equal(np.asarray(y0), np.asarray(y1))
    np.testing.assert_array_equal(np.asarray(s0), np.asarray(s1))
    ct = (jax.random.normal(jax.random.key(7), y0.shape), jax.random.normal(jax.random.key(8), s0.shape))
    for a, b in zip(vjp0(ct), vjp1(ct)):
        if a is None:
            assert b is None
            continue
        _close(a, b, 1e-5)


def test_mamba_gate():
    k1, k2, k3 = _keys(3)
    a = (3.0 * jax.random.normal(k1, (2, 9, 4, 8), jnp.float32)).astype(jnp.bfloat16)
    a_log = jax.random.normal(k2, (4, 1), jnp.float32)
    dt_bias = jax.random.normal(k3, (4, 8), jnp.float32)
    (y0, y1), (g0, g1) = _vjp_pair(_mamba_gate, lean.mamba_gate, (a, a_log, dt_bias))
    np.testing.assert_array_equal(np.asarray(y0), np.asarray(y1))
    for a0, a1 in zip(g0, g1):
        assert a0.dtype == a1.dtype and a0.shape == a1.shape
        _close(a0, a1, 1e-5 if a0.dtype == jnp.float32 else 2e-2)


@pytest.mark.parametrize("gated", [True, False])
def test_skip_gate(gated):
    k1, k2, k3, k4 = _keys(4)
    shape = (2, 11, 3, 8)
    y, x = (jax.random.normal(k, shape, jnp.float32).astype(jnp.bfloat16) for k in (k1, k2))
    z = (2.0 * jax.random.normal(k3, shape, jnp.float32)).astype(jnp.bfloat16)
    d = jax.random.normal(k4, (3,), jnp.float32)

    def ref(y, x, z, d):
        out = y.astype(jnp.float32) + d[:, None] * x.astype(jnp.float32)
        if gated:
            out = out * jax.nn.silu(z.astype(jnp.float32))
        return out.astype(x.dtype)

    def mod(y, x, z, d):
        return lean.skip_gate(y, x, z if gated else None, d)

    (y0, y1), (g0, g1) = _vjp_pair(ref, mod, (y, x, z, d))
    np.testing.assert_array_equal(np.asarray(y0, np.float32), np.asarray(y1, np.float32))
    for a, b in zip(g0, g1):
        assert a.dtype == b.dtype
        _close(a, b, 1e-5 if a.dtype == jnp.float32 else 2e-2)


def _kda_tiny():
    cfg = get_preset("tiny-hybrid")
    la = LinearAttentionConfig(num_k_heads=4, num_v_heads=4, head_k_dim=16, head_v_dim=16, chunk_size=16)
    return cfg.replace(linear_attention=la, layer_types=hybrid_pattern(4, "kda", 2), mtp=None)


_MODELS = {
    "tiny": lambda: get_preset("tiny"),
    "tiny-hybrid": lambda: get_preset("tiny-hybrid"),
    "tiny-mamba2": lambda: get_preset("tiny-mamba2"),
    "tiny-mamba3": lambda: get_preset("tiny-mamba3"),
    "tiny-gdn2": lambda: get_preset("tiny-gdn2"),
    "tiny-kda": _kda_tiny,
}


def _loss_and_grads(model, ids, seg, r):
    def loss(m):
        out = m(ids, segment_ids=seg)
        return jnp.sum(out.hidden.astype(jnp.float32) * r)

    return eqx.filter_value_and_grad(loss)(model)


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("remat,core", [("none", False), ("dots", False), ("none", True)])
def test_model_matches_plain_autodiff(name, remat, core):
    cfg = _MODELS[name]().replace(compute_dtype="float32", remat=remat)
    key = jax.random.key(0)
    ref = build_model(cfg, key=key)
    mod = build_model(cfg.replace(lean_vjp=True, lean_remat_core=core), key=key)
    leaves_ref, leaves_mod = jax.tree.leaves(eqx.filter(ref, eqx.is_array)), jax.tree.leaves(eqx.filter(mod, eqx.is_array))
    assert len(leaves_ref) == len(leaves_mod)
    for a, b in zip(leaves_ref, leaves_mod):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    batch, length = 2, 40
    ids = jax.random.randint(jax.random.key(1), (batch, length), 0, cfg.vocab_size)
    seg = jnp.asarray(np.repeat(np.array([[0, 1], [3, 4]]), [13, 27], axis=1), jnp.int32)
    r = jax.random.normal(jax.random.key(2), (batch, length, cfg.d_model), jnp.float32)
    # float32 GEMMs at full precision: on a GPU the default lets XLA use TF32 (about 1e-3 relative), and the
    # two models run the same math in a different order (recomputed products), which this tolerance would flag.
    with jax.default_matmul_precision("highest"):
        l0, g0 = eqx.filter_jit(_loss_and_grads)(ref, ids, seg, r)
        l1, g1 = eqx.filter_jit(_loss_and_grads)(mod, ids, seg, r)
    _close(l0, l1, 1e-5)
    flat0 = jax.tree_util.tree_leaves_with_path(eqx.filter(g0, eqx.is_array))
    flat1 = jax.tree_util.tree_leaves_with_path(eqx.filter(g1, eqx.is_array))
    assert [jax.tree_util.keystr(p) for p, _ in flat0] == [jax.tree_util.keystr(p) for p, _ in flat1]
    for (path, a), (_, b) in zip(flat0, flat1):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        scale = max(1e-6, float(np.max(np.abs(a))))
        err = float(np.max(np.abs(a - b))) / scale
        assert err < 1e-4, f"{jax.tree_util.keystr(path)}: relative error {err:.2e}"


@pytest.mark.parametrize("name,ratio", [("tiny-hybrid", 0.5), ("tiny-mamba2", 0.5), ("tiny-kda", 0.8)])
def test_lean_blocks_save_less(name, ratio):
    """The lean block keeps a fraction of plain autodiff's activation residuals (on CPU the chunked
    JAX op's own residuals, which the lean VJPs do not touch, dominate KDA's tiny block)."""
    from jla.models.lm import Block

    cfg = _MODELS[name]().replace(remat="none")
    # The ratio is against plain XLA autodiff, so both blocks run their short convs on XLA. With
    # conv_impl="auto" on a Hopper GPU both blocks would run the Mosaic GPU short-conv kernels, whose VJP
    # already keeps only its inputs (lean keeps using them and only covers the convs that run on XLA), so
    # the plain block would have no conv residuals left for lean to remove. On CPU "auto" is XLA anyway.
    if cfg.ssm is not None:
        cfg = cfg.replace(ssm=dataclasses.replace(cfg.ssm, conv_impl="xla"))
    if cfg.linear_attention is not None:
        cfg = cfg.replace(linear_attention=dataclasses.replace(cfg.linear_attention, conv_impl="xla"))
    lt, mt = cfg.resolved_layer_types[0], cfg.resolved_mlp_types[0]
    totals = []
    for use_lean in (False, True):
        block = Block(cfg, lt, mt, key=jax.random.key(0))
        if use_lean:
            block = lean.apply_lean(block)
        params, static = eqx.partition(block, eqx.is_array)
        x = jax.random.normal(jax.random.key(1), (2, 64, cfg.d_model), jnp.bfloat16)

        def f(params, x):
            positions = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), (2, 64))
            seg = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32) // 32, (2, 64))
            return eqx.combine(params, static)(x, positions, seg)[0]

        res = saved_residuals(f, params, x)
        totals.append(sum(int(a.size) * a.dtype.itemsize for a, src in res if not src.startswith("from the argument")))
    assert totals[1] < ratio * totals[0], totals


def _module_types(node, out: set) -> set:
    out.add(type(node))
    if isinstance(node, eqx.Module):
        for field in dataclasses.fields(node):
            _module_types(getattr(node, field.name), out)
    elif isinstance(node, (tuple, list)):
        for item in node:
            _module_types(item, out)
    return out


def test_apply_lean_types_and_axes():
    from jla.layers.mlp import SwiGLU
    from jla.models import logical_axes

    cfg = get_preset("tiny-hybrid")
    ref = build_model(cfg, key=jax.random.key(0))
    mod = build_model(cfg.replace(lean_vjp=True), key=jax.random.key(0))
    types = _module_types(mod, set())
    assert {lean.LeanRMSNorm, lean.LeanGatedRMSNorm, lean.LeanSwiGLU, lean.LeanShortConv} <= types
    assert not {RMSNorm, GatedRMSNorm, SwiGLU, ShortConv} & types
    def flat(model):
        def is_axes(x):
            return isinstance(x, tuple) and all(e is None or isinstance(e, str) for e in x)

        leaves = jax.tree_util.tree_leaves_with_path(logical_axes(model), is_leaf=is_axes)
        assert leaves and all(is_axes(axes) for _, axes in leaves)
        return [(jax.tree_util.keystr(path), axes) for path, axes in leaves]

    assert flat(ref) == flat(mod)


_RELEASE_1P3B = ["transformer-1p3b", "mamba2-1p3b", "gdn-hd128-1p3b", "kda-1p3b", "gdn2-1p3b", "mamba3-1p3b"]


@pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="needs the Mosaic GPU kernels")
@pytest.mark.parametrize("name,core", [(n, False) for n in _RELEASE_1P3B]
                         + [(n, True) for n in ("gdn-hd128-1p3b", "kda-1p3b", "gdn2-1p3b", "mamba3-1p3b")])
def test_release_blocks_on_kernels(name, core):
    """One block of each 1.3B preset on its Mosaic GPU kernel (bfloat16, packed rows): the lean block's
    output equals the default's and its gradients agree within bfloat16 rounding (also with the op
    recomputed in the backward pass, ``remat_core``)."""
    import dataclasses

    from jla.models.lm import Block

    cfg = get_preset(name).replace(remat="none")
    changes = {"attention": dataclasses.replace(cfg.attention, implementation="mosaic_gpu")}
    if cfg.linear_attention is not None:
        changes["linear_attention"] = dataclasses.replace(cfg.linear_attention, implementation="mosaic_gpu")
    if cfg.ssm is not None:
        changes["ssm"] = dataclasses.replace(cfg.ssm, implementation="mosaic_gpu")
    cfg = cfg.replace(**changes)
    lt, mt = cfg.resolved_layer_types[0], cfg.resolved_mlp_types[0]
    batch, length = 1, 512
    x = jax.random.normal(jax.random.key(1), (batch, length, cfg.d_model), jnp.float32).astype(jnp.bfloat16)
    positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32) % 200, (batch, length))
    seg = jnp.broadcast_to((jnp.arange(length, dtype=jnp.int32) >= 200).astype(jnp.int32), (batch, length))
    ct = jax.random.normal(jax.random.key(2), x.shape, jnp.float32).astype(jnp.bfloat16)
    outs = []
    for use_lean in (False, True):
        block = Block(cfg, lt, mt, key=jax.random.key(0))
        if use_lean:
            block = lean.apply_lean(block, remat_core=core)
        params, static = eqx.partition(block, eqx.is_array)

        @jax.jit
        def run(params, x):
            y, vjp = jax.vjp(lambda p, x: eqx.combine(p, static)(x, positions, seg)[0], params, x)
            return y, vjp(ct)

        outs.append(run(params, x))
    (y0, (gp0, gx0)), (y1, (gp1, gx1)) = outs
    _close(y0, y1, 1e-2)
    _close(gx0, gx1, 3e-2)
    for (path, a), (_, b) in zip(jax.tree_util.tree_leaves_with_path(gp0), jax.tree_util.tree_leaves_with_path(gp1)):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        err = float(np.max(np.abs(a - b))) / max(1e-6, float(np.max(np.abs(a))))
        assert err < 3e-2, f"{jax.tree_util.keystr(path)}: relative error {err:.2e}"


def test_lean_rejects_fp8():
    with pytest.raises(ValueError, match="lean_vjp"):
        get_preset("tiny").replace(lean_vjp=True, fp8="current")
    with pytest.raises(ValueError, match="lean_vjp"):
        get_preset("tiny").replace(lean_remat_core=True)


def test_lean_rejects_fused_mlp_kernel():
    """``lean_vjp`` with the fused SwiGLU GEMMs (``mlp_kernel = 'mosaic_gpu'``) is a config error."""
    from jla.models.config import ModelConfig

    if "mlp_kernel" in {f.name for f in dataclasses.fields(ModelConfig)}:
        cfg_cls = ModelConfig
    else:  # the field comes with jla.layers.fused_mlp: stand in for it

        @dataclasses.dataclass(frozen=True)
        class cfg_cls(ModelConfig):
            mlp_kernel: str = "xla"

    base = get_preset("tiny")
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(ModelConfig) if f.init}
    assert cfg_cls(**dict(kwargs, lean_vjp=True)).lean_vjp
    assert cfg_cls(**dict(kwargs, mlp_kernel="mosaic_gpu")).mlp_kernel == "mosaic_gpu"
    with pytest.raises(ValueError, match="mlp_kernel"):
        cfg_cls(**dict(kwargs, lean_vjp=True, mlp_kernel="mosaic_gpu"))


def test_remat_core_keeps_less():
    """With ``remat_core`` the GDN block's VJP keeps fewer residuals than the lean block (the op's are rerun)."""
    from jla.models.lm import Block

    cfg = _MODELS["tiny-hybrid"]().replace(remat="none")
    totals = []
    for core in (False, True):
        block = lean.apply_lean(Block(cfg, "gated_deltanet", "dense", key=jax.random.key(0)), remat_core=core)
        assert block.mixer.remat_core is core
        params, static = eqx.partition(block, eqx.is_array)
        x = jax.random.normal(jax.random.key(1), (2, 64, cfg.d_model), jnp.bfloat16)

        def f(params, x):
            positions = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), (2, 64))
            return eqx.combine(params, static)(x, positions, None)[0]

        res = saved_residuals(f, params, x)
        totals.append(sum(int(a.size) * a.dtype.itemsize for a, src in res if not src.startswith("from the argument")))
    assert totals[1] < totals[0], totals
