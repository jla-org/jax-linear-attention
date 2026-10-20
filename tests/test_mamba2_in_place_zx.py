# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mamba-2 z|x rows read in place from one stacked projection GEMM (``jla.layers.ssm.IN_PLACE_ZX``).

With an FP8 ``zx_proj`` (one GEMM writing ``[B, T, 2, inner]``) and an FP8 out-projection, the fused layer hands the
GEMM output itself to the x conv (``short_conv_rows``, row 1) and to the glue kernels (``gated_norm(z_row=0)``)
instead of copying both rows out. Checked here: the glue kernels with a stacked ``z`` against the same kernels on
the sliced row (interpreted on the CPU: values and every gradient, the other row's gradient zero); on an H100 the
conv kernels reading a row in place against :func:`short_conv` on the sliced row (packed documents, odd lengths),
and the FP8 Mamba-2 layer with and without the in-place rows (the same values and gradients).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers import mamba2_fused as fused
from jla.layers import ssm


def _hopper() -> bool:
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and jax.device_count() == 1 and getattr(device, "compute_capability", "") == "9.0"


@pytest.mark.parametrize("heads,row", [(4, 0), (3, 0), (3, 1)])  # 3 heads: the masked kernels
def test_gated_norm_reads_a_stacked_gate_row_interpreted(heads, row):
    ks = jax.random.split(jax.random.key(0), 6)
    batch, length, head_dim = 2, 5, 8
    width = heads * head_dim
    y = jax.random.normal(ks[0], (batch, length, heads, head_dim))
    x = jax.random.normal(ks[1], (batch, length, heads, head_dim))
    zx = jax.random.normal(ks[2], (batch, length, 2, width))
    d = jax.random.normal(ks[3], (heads,))
    norm_w = 1.0 + 0.3 * jax.random.normal(ks[4], (width,))
    cot = jax.random.normal(ks[5], (batch, length, width))

    def loss(in_place, y, x, zx, d, norm_w):
        if not in_place:  # the gate sliced out of the stacked tensor
            out = fused.gated_norm(y, x, zx[:, :, row], d, norm_w, eps=1e-5, compute_dtype=jnp.float32,
                                   interpret=True)
        else:
            out = fused.gated_norm(y, x, zx, d, norm_w, eps=1e-5, compute_dtype=jnp.float32, interpret=True,
                                   z_row=row)
        return jnp.sum(out * cot), out

    args = (y, x, zx, d, norm_w)
    (_, want), want_g = jax.value_and_grad(lambda *a: loss(False, *a), argnums=tuple(range(5)), has_aux=True)(*args)
    (_, got), got_g = jax.value_and_grad(lambda *a: loss(True, *a), argnums=tuple(range(5)), has_aux=True)(*args)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    for name, g, w in zip(("y", "x", "zx", "D", "norm_w"), got_g, want_g):
        np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-6, err_msg=name)
    assert not np.any(np.asarray(got_g[2][:, :, 1 - row]))  # the other row's gradient is not the glue's
    with pytest.raises(ValueError, match="z_row"):
        fused.gated_norm(y, x, zx, d, norm_w, eps=1e-5, compute_dtype=jnp.float32, interpret=True, z_row=2)


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU short conv kernels need one H100")
@pytest.mark.parametrize("length,packed", [(256, True), (131, True), (200, False)])
def test_short_conv_rows_matches_the_sliced_row_on_gpu(length, packed):
    from jla.ops.mosaic_gpu.short_conv import short_conv, short_conv_rows

    batch, channels, width = 2, 512, 4
    ks = jax.random.split(jax.random.key(length), 5)
    xs = jax.random.normal(ks[0], (batch, length, 2, channels), jnp.bfloat16)
    weight = 0.5 * jax.random.normal(ks[1], (channels, width))
    bias = 0.1 * jax.random.normal(ks[2], (channels,))
    cot = jax.random.normal(ks[3], (batch, length, channels))
    seg = None
    if packed:
        cuts = np.sort(np.random.default_rng(length).choice(np.arange(1, length), size=(batch, 3)), axis=1)
        seg = jnp.asarray((np.arange(length)[None, None] >= cuts[..., None]).sum(axis=1), jnp.int32)
    for row in (0, 1):
        def rows(xs, weight, bias, row=row):
            out = short_conv_rows(xs, weight, bias, row=row, segment_ids=seg)
            return jnp.sum(out.astype(jnp.float32) * cot), out

        def sliced(xs, weight, bias, row=row):
            out = short_conv(xs[:, :, row], weight, bias, segment_ids=seg)
            return jnp.sum(out.astype(jnp.float32) * cot), out

        (_, got), got_g = jax.jit(jax.value_and_grad(rows, argnums=(0, 1, 2), has_aux=True))(xs, weight, bias)
        (_, want), want_g = jax.jit(jax.value_and_grad(sliced, argnums=(0, 1, 2), has_aux=True))(xs, weight, bias)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
        for name, g, w in zip(("xs", "weight", "bias"), got_g, want_g):
            np.testing.assert_array_equal(np.asarray(g, np.float32), np.asarray(w, np.float32), err_msg=name)
        assert not np.any(np.asarray(got_g[0][:, :, 1 - row], np.float32))


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU kernels and the fused glue need one H100")
def test_fp8_layer_with_in_place_rows_matches_sliced_rows_on_gpu(monkeypatch):
    """The FP8 fused layer reads the z|x rows in place; against the same layer slicing them: the same output and
    gradients (the same kernels on the same values; only where the rows are read from differs)."""
    from test_mamba2_fused import _flat, _layer, _segments, _value_and_grads

    from jla.layers.fp8 import apply_fp8

    batch, length, d_model = 2, 200, 128
    u = jax.random.normal(jax.random.key(3), (batch, length, d_model), jnp.bfloat16)
    seg = _segments(batch, length, 4)
    cot = jax.random.normal(jax.random.key(5), (batch, length, d_model))
    layer = apply_fp8(_layer(fused_glue=True, grouped_key_grads=True), "delayed", min_features=128)
    assert ssm._fused_glue_applies(layer) and ssm._zx_in_place(layer, u, seg)
    got = _value_and_grads(layer, u, seg, cot)
    monkeypatch.setattr(ssm, "IN_PLACE_ZX", False)
    assert not ssm._zx_in_place(layer, u, seg)
    want = _value_and_grads(layer, u, seg, cot)
    np.testing.assert_allclose(np.asarray(got[0], np.float32), np.asarray(want[0], np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(got[2], np.float32), np.asarray(want[2], np.float32), rtol=1e-6, atol=1e-6)
    want_p = _flat(want[1])
    for key, value in _flat(got[1]).items():
        np.testing.assert_allclose(value, want_p[key], rtol=1e-6, atol=1e-6, err_msg=key)


@pytest.mark.skipif(not _hopper(), reason="the Mosaic GPU kernels and the fused glue need one H100")
def test_fp8_layer_slices_the_rows_when_the_conv_kernels_decline_them_on_gpu(monkeypatch):
    """``short_conv_rows`` returning ``None`` (the kernels declining the stacked rows) makes the layer slice the rows
    out, as with ``IN_PLACE_ZX = False``: the same output and gradients."""
    from test_mamba2_fused import _flat, _layer, _segments, _value_and_grads

    from jla.layers.fp8 import apply_fp8
    from jla.ops.mosaic_gpu import short_conv

    batch, length, d_model = 2, 200, 128
    u = jax.random.normal(jax.random.key(3), (batch, length, d_model), jnp.bfloat16)
    seg = _segments(batch, length, 4)
    cot = jax.random.normal(jax.random.key(5), (batch, length, d_model))
    layer = apply_fp8(_layer(fused_glue=True, grouped_key_grads=True), "delayed", min_features=128)
    monkeypatch.setattr(short_conv, "short_conv_rows", lambda *args, **kwargs: None)
    assert ssm._fused_glue_applies(layer) and ssm._zx_in_place(layer, u, seg)
    got = _value_and_grads(layer, u, seg, cot)
    monkeypatch.setattr(ssm, "IN_PLACE_ZX", False)
    want = _value_and_grads(layer, u, seg, cot)
    np.testing.assert_allclose(np.asarray(got[0], np.float32), np.asarray(want[0], np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(got[2], np.float32), np.asarray(want[2], np.float32), rtol=1e-6, atol=1e-6)
    want_p = _flat(want[1])
    for key, value in _flat(got[1]).items():
        np.testing.assert_allclose(value, want_p[key], rtol=1e-6, atol=1e-6, err_msg=key)


def test_in_place_rows_apply_only_to_a_stacked_fp8_projection():
    """bf16 splits the z|x GEMM (``SPLIT_ZX``: nothing to read in place); an FP8 layer reads the rows in place on one
    H100 (the Mosaic GPU conv), and nowhere else; ``IN_PLACE_ZX = False`` turns it off."""
    from test_mamba2_fused import _layer

    from jla.layers.fp8 import apply_fp8

    u = jnp.ones((1, 64, 128), jnp.bfloat16)
    bf16 = _layer(fused_glue=True, impl="chunked")
    fp8 = apply_fp8(bf16, "delayed", min_features=128)
    assert not ssm._zx_in_place(bf16, u, None)
    assert ssm._zx_in_place(fp8, u, None) == _hopper()
    previous = ssm.IN_PLACE_ZX
    ssm.IN_PLACE_ZX = False
    try:
        assert not ssm._zx_in_place(fp8, u, None)
    finally:
        ssm.IN_PLACE_ZX = previous
