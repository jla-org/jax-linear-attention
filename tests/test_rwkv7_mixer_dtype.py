# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LanguageModel's RWKV-7 mixer hands the op ``r, k, v`` in the compute dtype (bf16 in a bf16 model).

The Mosaic GPU RWKV-7 kernel takes bf16 ``q, k, v`` only. A bf16 :class:`RWKV7Mixer` that passed f32
``r, k, v`` failed under ``implementation="mosaic_gpu"`` and silently ran the chunked op under ``"auto"``.
The CPU tests check the dtypes the op receives (training forward and the one-token decode step); the GPU
test runs a bf16 mixer on the kernel, output and every parameter gradient, against the chunked op.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jla.layers.rwkv7 as rwkv7_layer
from jla.layers.rwkv7 import RWKV7Mixer
from jla.models.config import LinearAttentionConfig
from jla.ops.mosaic_gpu.vector_gate_ops import is_hopper

gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100 (compute capability 9.0)")

D = 128  # 2 heads x 64: a head size and chunk (32) the Mosaic GPU kernel supports
_LA = LinearAttentionConfig(num_k_heads=2, num_v_heads=2, head_k_dim=64, head_v_dim=64, conv_size=0,
                            qk_l2norm=False, chunk_size=32, rwkv7_lora_dims=(16, 16, 16, 16))


def _mixer(impl: str, compute_dtype: str = "bfloat16") -> RWKV7Mixer:
    """A later-layer mixer with every float parameter moved off its initial value."""
    la = dataclasses.replace(_LA, implementation=impl)
    m = RWKV7Mixer(D, la, layer_idx=1, n_layers=3, key=jax.random.key(1), norm_eps=1e-5,
                   compute_dtype=compute_dtype)
    params, static = eqx.partition(m, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(7), len(leaves))
    leaves = [p + 0.05 * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def _spy(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    real = rwkv7_layer.linear_attention

    def spy(kind, q, k, v, **kw):
        calls.append({"kind": kind, "q": q.dtype, "k": k.dtype, "v": v.dtype,
                      "w": kw["w"].dtype, "a": kw["a"].dtype, "b": kw["b"].dtype})
        return real(kind, q, k, v, **kw)

    monkeypatch.setattr(rwkv7_layer, "linear_attention", spy)
    return calls


@pytest.mark.parametrize("compute_dtype", ["bfloat16", "float32"])
def test_mixer_passes_compute_dtype_rkv_and_f32_gates(monkeypatch, compute_dtype):
    calls = _spy(monkeypatch)
    m = _mixer("chunked", compute_dtype)
    x = jax.random.normal(jax.random.key(0), (1, 40, D)).astype(compute_dtype)
    y = m(x)
    want = jnp.dtype(compute_dtype)
    assert calls == [{"kind": "rwkv7", "q": want, "k": want, "v": want,
                      "w": jnp.dtype(jnp.float32), "a": jnp.dtype(jnp.float32), "b": jnp.dtype(jnp.float32)}]
    assert y.dtype == want and bool(jnp.all(jnp.isfinite(y.astype(jnp.float32))))


def test_decode_step_passes_the_same_dtypes(monkeypatch):
    calls = _spy(monkeypatch)
    m = _mixer("chunked")
    x = jax.random.normal(jax.random.key(0), (2, 9, D)).astype(jnp.bfloat16)
    cache = m.init_cache(2, jnp.bfloat16)
    _, cache = m.step(x[:, :8], cache)  # prefill (chunked)
    m.step(x[:, 8:], cache)  # one-token decode (reference op)
    assert len(calls) == 2
    assert all(c["q"] == c["k"] == c["v"] == jnp.dtype(jnp.bfloat16) for c in calls)


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@gpu
def test_bf16_mixer_runs_the_mosaic_gpu_kernel_and_matches_chunked():
    x = jax.random.normal(jax.random.key(2), (2, 100, D)).astype(jnp.bfloat16)
    seg = jnp.asarray(np.repeat([[0, 1, 2], [3, 3, 4]], [30, 41, 29], axis=1), jnp.int32)
    kernel, chunked = _mixer("mosaic_gpu"), _mixer("chunked")

    def loss(m):
        return jnp.sum(jnp.square(m(x, seg).astype(jnp.float32)))

    y_k, y_c = jax.jit(kernel.__call__)(x, seg), jax.jit(chunked.__call__)(x, seg)
    assert _rel(y_k, y_c) < 1e-2
    g_k = eqx.filter_jit(eqx.filter_grad(loss))(kernel)
    g_c = eqx.filter_jit(eqx.filter_grad(loss))(chunked)
    leaves_k = jax.tree.leaves(eqx.filter(g_k, eqx.is_inexact_array))
    leaves_c = jax.tree.leaves(eqx.filter(g_c, eqx.is_inexact_array))
    assert len(leaves_k) == len(leaves_c) > 10
    for a, b in zip(leaves_k, leaves_c):
        if float(jnp.max(jnp.abs(b))) == 0.0:  # parameters with no gradient path
            assert float(jnp.max(jnp.abs(a))) == 0.0
            continue
        assert _rel(a, b) < 3e-2
