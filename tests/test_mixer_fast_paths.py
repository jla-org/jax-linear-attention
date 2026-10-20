# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The end-to-end fast paths of the Mamba mixers against the paths they replace.

* ``SPLIT_ZX``: the ``z`` and ``x`` rows of ``zx_proj`` as two GEMMs instead of slicing
  one stacked GEMM (Mamba-2 and Mamba-3): values and gradients of the input and every
  parameter, float32 and bfloat16, with documents;
* ``GROUPED_BC``: Mamba-2 with grouped ``B, C`` handed to the Mosaic GPU simple GLA
  against repeating them per head (GPU only): values, gradients, documents, initial
  and final states;
* both together: the Mamba-2 layer at the 130M head shape on the Mosaic GPU kernels
  against the previous paths (GPU only).
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import ssm as ssm_mod
from jla.models.config import SSMConfig


@pytest.fixture(autouse=True)
def _exact_float32_matmuls():
    with jax.default_matmul_precision("highest"):
        yield


def _segments(batch, length, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=min(3, length - 1), replace=False))
        rows.append(np.searchsorted(cuts, np.arange(length), side="right"))
    return jnp.asarray(np.stack(rows), jnp.int32)


def _layer(kind, dtype, seed=0, **cfg):
    base = dict(d_state=8, expand=2, head_dim=8, n_groups=2, chunk_size=16, implementation="chunked")
    base.update(cfg)
    cls = {"mamba2": ssm_mod.Mamba2, "mamba3": ssm_mod.Mamba3}[kind]
    return cls(32, SSMConfig(**base), key=jax.random.key(seed), std=0.2, norm_eps=1e-5, compute_dtype=dtype)


def _layer_value_and_grads(layer, u, segment_ids, cot, *, split, monkeypatch):
    monkeypatch.setattr(ssm_mod, "SPLIT_ZX", split)

    def loss(layer, u):
        y = layer(u, segment_ids=segment_ids)
        return jnp.sum(y.astype(jnp.float32) * cot), y

    return jax.jit(jax.grad(loss, argnums=(0, 1), has_aux=True))(layer, u)


@pytest.mark.parametrize("kind", ["mamba2", "mamba3"])
@pytest.mark.parametrize("dtype,documents", [("float32", False), ("float32", True), ("bfloat16", True)])
def test_split_zx_projection_matches_the_stacked_gemm(kind, dtype, documents, monkeypatch):
    layer = _layer(kind, dtype)
    ks = jax.random.split(jax.random.key(4), 2)
    u = jax.random.normal(ks[0], (2, 37, 32), jnp.dtype(dtype))
    cot = jax.random.normal(ks[1], (2, 37, 32))
    seg = _segments(2, 37, 1) if documents else None
    (wl, wu), wy = _layer_value_and_grads(layer, u, seg, cot, split=False, monkeypatch=monkeypatch)
    (gl, gu), gy = _layer_value_and_grads(layer, u, seg, cot, split=True, monkeypatch=monkeypatch)
    got = [gy, gu, *jax.tree.leaves(gl)]
    want = [wy, wu, *jax.tree.leaves(wl)]
    if dtype == "float32":  # the same products, accumulated by a different GEMM
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5 * max(1.0, float(jnp.abs(w).max())))
        return
    # bfloat16: against the float32-compute layer (same parameters), every tensor of the new path is
    # as close as the previous path's, or within the bf16 level of the layer's other weight gradients
    # (2% of the norm). On CPU the zx_proj weight gradient is 1.1% off against 0.6% for the stacked
    # GEMM, whose CPU lowering skips one bf16 rounding; out_proj's is 1.1% on both paths.
    (rl, ru), ry = _layer_value_and_grads(_layer(kind, "float32"), u.astype(jnp.float32), seg, cot, split=False,
                                          monkeypatch=monkeypatch)
    for g, w, r in zip(got, want, [ry, ru, *jax.tree.leaves(rl)]):
        assert _relerr(g, r) <= max(1.25 * _relerr(w, r), 0.02), (_relerr(g, r), _relerr(w, r))


def test_split_rows_are_the_stacked_rows(monkeypatch):
    layer = _layer("mamba2", "float32")
    u = jax.random.normal(jax.random.key(0), (2, 5, 32))
    stacked = layer.zx_proj(u)
    monkeypatch.setattr(ssm_mod, "SPLIT_ZX", True)
    rows = ssm_mod._split_rows(layer.zx_proj, u)
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert row.shape == stacked[..., i, :].shape and row.dtype == stacked.dtype
        np.testing.assert_allclose(row, stacked[..., i, :], rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------------------
# Mamba-2 grouped B/C (Mosaic GPU only)


def _on_hopper() -> bool:
    try:
        from jla.ops.mosaic_gpu._common import is_hopper
    except ImportError:
        return False
    return is_hopper()


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")


def _ssd_inputs(seed, batch, length, heads, groups, head_dim=64, d_state=128):
    ks = jax.random.split(jax.random.key(seed), 7)
    x = jax.random.normal(ks[0], (batch, length, heads, head_dim), jnp.bfloat16)
    dt = jax.nn.softplus(jax.random.normal(ks[1], (batch, length, heads)) - 1.0)
    a = -jnp.exp(jax.random.uniform(ks[2], (heads,), minval=0.0, maxval=2.0))
    b = (0.3 * jax.random.normal(ks[3], (batch, length, groups, d_state))).astype(jnp.bfloat16)
    c = (0.3 * jax.random.normal(ks[4], (batch, length, groups, d_state))).astype(jnp.bfloat16)
    d = jax.random.normal(ks[5], (heads,))
    h0 = 0.1 * jax.random.normal(ks[6], (batch, heads, d_state, head_dim))
    return x, dt, a, b, c, d, h0


def _ssd_value_and_grads(inputs, *, grouped, segment_ids, with_state, monkeypatch):
    monkeypatch.setattr(ssm_mod, "GROUPED_BC", grouped)
    x, dt, a, b, c, d, h0 = inputs
    cot = jax.random.normal(jax.random.key(99), x.shape)

    def loss(x, dt, b, c, h0):
        y, state = ssm_mod.mamba2_ssd(x, dt, a, b, c, d=d, segment_ids=segment_ids,
                                      initial_state=h0 if with_state else None,
                                      output_final_state=with_state, impl="auto", chunk_size=64)
        total = jnp.sum(y.astype(jnp.float32) * cot)
        if with_state:
            total = total + jnp.sum(state * 0.01)
        return total, (y, state)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))(x, dt, b, c, h0)


def _relerr(got, want) -> float:
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30))


def _close(got, want, rel):
    err = _relerr(got, want)
    assert err < rel, err


@gpu
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("length,documents,with_state", [(512, False, False), (333, True, False),
                                                          (256, False, True), (200, True, True)])
def test_mamba2_grouped_bc_matches_repeated(groups, length, documents, with_state, monkeypatch):
    inputs = _ssd_inputs(groups + length, 2, length, 8, groups)
    seg = _segments(2, length, length) if documents else None
    want_grads, (want_y, want_state) = _ssd_value_and_grads(inputs, grouped=False, segment_ids=seg,
                                                            with_state=with_state, monkeypatch=monkeypatch)
    got_grads, (got_y, got_state) = _ssd_value_and_grads(inputs, grouped=True, segment_ids=seg,
                                                         with_state=with_state, monkeypatch=monkeypatch)
    # Forward: the same kernel reads the same values, so the outputs match exactly.
    np.testing.assert_array_equal(np.asarray(got_y, np.float32), np.asarray(want_y, np.float32))
    if with_state:
        np.testing.assert_array_equal(got_state, want_state)
    # Backward: both paths sum bf16 per-head dB, dC over a group's heads, the grouped one in float32
    # (scalar_gate._key_grad), the repeated one in the transpose of jnp.repeat.
    names = ("dx", "ddt", "db", "dc", "dh0")
    for name, g, w in zip(names, got_grads, want_grads):
        if name == "dh0" and not with_state:
            continue
        _close(g, w, 1e-2 if name in ("db", "dc") else 1e-6)


@gpu
def test_mamba2_grouped_bc_is_used_only_where_the_kernel_takes_it(monkeypatch):
    inputs = _ssd_inputs(5, 1, 128, 8, 2)
    x, dt, a, b, c, d, _ = inputs
    v = x.astype(c.dtype)
    g = dt * a
    assert ssm_mod._grouped_simple_gla("auto", c, b, v, g, scale=1.0, chunk_size=64)
    assert not ssm_mod._grouped_simple_gla("chunked", c, b, v, g, scale=1.0, chunk_size=64)
    assert not ssm_mod._grouped_simple_gla("auto", c, b, v, g, scale=1.0, chunk_size=32)
    monkeypatch.setattr(ssm_mod, "GROUPED_BC", False)
    assert not ssm_mod._grouped_simple_gla("auto", c, b, v, g, scale=1.0, chunk_size=64)


@gpu
@pytest.mark.parametrize("documents", [False, True])
def test_mamba2_layer_fast_paths_match_the_previous_paths(documents, monkeypatch):
    """The Mamba-2 layer at the 130M head shape (d_state 128, heads of 64, one B/C group, bf16) on the
    Mosaic GPU kernels: both fast paths on against both off, the output and every gradient."""
    cfg = SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1, chunk_size=64, implementation="auto")
    layer = ssm_mod.Mamba2(128, cfg, key=jax.random.key(7), std=0.05, norm_eps=1e-5, compute_dtype="bfloat16")
    ks = jax.random.split(jax.random.key(8), 2)
    u = jax.random.normal(ks[0], (2, 333, 128), jnp.bfloat16)
    cot = jax.random.normal(ks[1], (2, 333, 128))
    seg = _segments(2, 333, 5) if documents else None

    def run(fast):
        monkeypatch.setattr(ssm_mod, "GROUPED_BC", fast)
        monkeypatch.setattr(ssm_mod, "SPLIT_ZX", fast)

        def loss(layer, u):
            y = layer(u, segment_ids=seg)
            return jnp.sum(y.astype(jnp.float32) * cot), y

        return jax.jit(jax.grad(loss, argnums=(0, 1), has_aux=True))(layer, u)

    (gl, gu), gy = run(True)
    (wl, wu), wy = run(False)
    _close(gy, wy, 1e-2)
    for g, w in zip([gu, *jax.tree.leaves(gl)], [wu, *jax.tree.leaves(wl)]):
        _close(g, w, 2e-2)


def _compiled_layer_hlo(kind, d_model, cfg, shape) -> str:
    cls = {"mamba2": ssm_mod.Mamba2, "mamba3": ssm_mod.Mamba3}[kind]
    layer = cls(d_model, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-5, compute_dtype="bfloat16")

    def loss(layer, u):
        return jnp.sum(layer(u).astype(jnp.float32))

    u = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
    return jax.jit(jax.grad(loss, argnums=(0, 1))).lower(layer, u).compile().as_text()


@gpu
def test_mamba2_grouped_bc_writes_no_per_head_copy_of_b_and_c(monkeypatch):
    """The fast path's point: no ``[B, T, H, N]`` broadcast of B and C in the compiled layer."""
    cfg = SSMConfig(d_state=128, expand=2, head_dim=64, n_groups=1, chunk_size=64, implementation="auto")
    per_head = r"= bf16\[2,256,8,128\]\S* broadcast\("
    counts = {}
    for grouped in (False, True):
        monkeypatch.setattr(ssm_mod, "GROUPED_BC", grouped)
        counts[grouped] = len(re.findall(per_head, _compiled_layer_hlo("mamba2", 256, cfg, (2, 256, 256))))
    assert counts[False] > 0 and counts[True] == 0, counts


@pytest.mark.parametrize("kind", ["mamba2", "mamba3"])
def test_split_zx_forms_no_stacked_zx_tensor(kind, monkeypatch):
    cfg = SSMConfig(d_state=16, expand=2, head_dim=16, n_groups=1, chunk_size=16, implementation="chunked")
    stacked = r"(?:bf16|f32)\[2,64,2,128\]"  # CPU may run the bf16 GEMM in float32
    counts = {}
    for split in (False, True):
        monkeypatch.setattr(ssm_mod, "SPLIT_ZX", split)
        counts[split] = len(re.findall(stacked, _compiled_layer_hlo(kind, 64, cfg, (2, 64, 64))))
    assert counts[False] > 0 and counts[True] == 0, counts


def test_mamba2_grouped_bc_falls_back_off_hopper(monkeypatch):
    """Off Hopper (or for the chunked/reference impls) the groups are repeated, as before."""
    if _on_hopper():
        pytest.skip("checks the fallback where the Mosaic GPU kernel cannot run")
    monkeypatch.setattr(ssm_mod, "GROUPED_BC", True)
    ks = jax.random.split(jax.random.key(3), 5)
    x = jax.random.normal(ks[0], (1, 20, 4, 8))
    dt = jax.nn.softplus(jax.random.normal(ks[1], (1, 20, 4)))
    a = -jnp.ones((4,))
    b = jax.random.normal(ks[2], (1, 20, 2, 8))
    c = jax.random.normal(ks[3], (1, 20, 2, 8))
    y_auto, _ = ssm_mod.mamba2_ssd(x, dt, a, b, c, impl="auto", chunk_size=16)
    y_ref, _ = ssm_mod.mamba2_ssd(x, dt, a, b, c, impl="reference")
    np.testing.assert_allclose(y_auto, y_ref, rtol=2e-4, atol=2e-4)


def test_module_flags_default_to_the_fast_paths():
    assert ssm_mod.GROUPED_BC is True
    assert ssm_mod.SPLIT_ZX is True
