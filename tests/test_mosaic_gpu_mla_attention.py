# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLA's asymmetric heads on the Mosaic GPU attention kernel (``jla.ops.mosaic_gpu.attention``).

Queries and keys of 192 channels (128 no-RoPE + 64 RoPE) with values of 128, as DeepSeek-V2/V3,
Kimi K2 and Kimi Linear use them. The kernel's output and ``dq, dk, dv`` are compared with an
independent fp32 attention written with einsums on the same bf16 inputs (causal, packed documents
with sorted, unsorted and reused ids, sliding windows, GQA, lengths that are not a multiple of the
block), with the internal kernel options (the overlapped ``dk, dv`` body, the launch order). The default
backward keeps the serial ``dk, dv`` body for these heads (bitwise; the default itself is checked on CPU). The MLA
layer on the kernel (``implementation="mosaic_gpu"`` and ``"auto"``) is compared with the layer on
XLA: outputs and the gradients of every parameter and of the input, packed documents and an odd
length included. With the batch or the heads sharded over two GPUs the kernels run per device.
The kernel tests need an H100 (the sharding test two); the dispatch checks run on CPU too.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.mla import MLA
from jla.models.config import MLAConfig


def _on_hopper() -> bool:
    if jax.default_backend() != "gpu":
        return False
    capability = getattr(jax.devices()[0], "compute_capability", None)
    return capability is not None and str(capability).split(".")[0] == "9"


gpu = pytest.mark.skipif(not _on_hopper(), reason="needs an H100 (compute capability 9.0)")

QK, V = 192, 128  # DeepSeek-V3 / Kimi heads: 128 no-RoPE + 64 RoPE query/key channels, values of 128


def _reference(q, k, v, segment_ids, window, scale):
    """fp32 causal attention with einsums and an explicit mask (independent of the kernel and of XLA's
    ``dot_product_attention``, which needs one head dim)."""
    f32 = jnp.float32
    group = q.shape[2] // k.shape[2]
    kk = jnp.repeat(k.astype(f32), group, axis=2)
    vv = jnp.repeat(v.astype(f32), group, axis=2)
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(f32), kk, precision=jax.lax.Precision.HIGHEST) * scale
    pos = jnp.arange(q.shape[1])
    mask = pos[:, None] >= pos[None, :]
    if window is not None:
        mask = mask & (pos[:, None] - pos[None, :] < window)
    mask = mask[None, None]
    if segment_ids is not None:
        mask = mask & (segment_ids[:, :, None] == segment_ids[:, None, :])[:, None]
    p = jax.nn.softmax(jnp.where(mask, s, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", p, vv, precision=jax.lax.Precision.HIGHEST)


def _segments(rng, batch, length, docs, order="sorted"):
    ids = np.zeros((batch, length), np.int32)
    for b in range(batch):
        cuts = np.sort(rng.choice(np.arange(1, length), size=docs - 1, replace=False))
        run = np.searchsorted(cuts, np.arange(length), side="right")
        if order == "shuffled":  # distinct corpus-style ids in no order: every id still one run
            run = (run * 856) % 1009
        elif order == "interleaved":  # ids reused across runs: no block may be skipped
            run = run % 2
        ids[b] = run + 10 * b
    return jnp.asarray(ids)


def _inputs(seed, batch, length, heads, kv_heads, dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(keys[0], (batch, length, heads, QK), jnp.float32).astype(dtype)
    k = jax.random.normal(keys[1], (batch, length, kv_heads, QK), jnp.float32).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, kv_heads, V), jnp.float32).astype(dtype)
    do = jax.random.normal(keys[3], (batch, length, heads, V), jnp.float32)
    return q, k, v, do


def _rel_err(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    return float(np.max(np.abs(actual - expected)) / (np.max(np.abs(expected)) + 1e-3))


def _row_rel_err(actual, expected):
    """Worst row (batch, position, head) error over that row's magnitude (floor: 5% of the median row)."""
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    magnitude = np.abs(expected).max(axis=-1)
    floor = 0.05 * float(np.median(magnitude)) + 1e-6
    return float((np.abs(actual - expected).max(axis=-1) / (magnitude + floor)).max())


def _check(batch, length, heads, kv_heads, window, docs, *, order="sorted", dtype=jnp.bfloat16):
    from jla.ops.mosaic_gpu.attention import flash_attention

    q, k, v, do = _inputs(length + docs, batch, length, heads, kv_heads, dtype)
    seg = _segments(np.random.default_rng(length + 7 * docs), batch, length, docs, order) if docs else None
    scale = QK**-0.5

    def loss(fn, q, k, v):
        out = fn(q, k, v)
        return jnp.sum(out.astype(jnp.float32) * do), out

    kernel = lambda q, k, v: flash_attention(q, k, v, segment_ids=seg, sliding_window=window)  # noqa: E731
    dense = lambda q, k, v: _reference(q, k, v, seg, window, scale)  # noqa: E731
    (dq, dk, dv), out = jax.jit(jax.grad(lambda *a: loss(kernel, *a), argnums=(0, 1, 2), has_aux=True))(q, k, v)
    (rdq, rdk, rdv), ref = jax.jit(jax.grad(lambda *a: loss(dense, *a), argnums=(0, 1, 2), has_aux=True))(q, k, v)
    assert out.dtype == q.dtype and out.shape == (batch, length, heads, V)
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    errors = {"out": _rel_err(out, ref), "dq": _rel_err(dq, rdq), "dk": _rel_err(dk, rdk), "dv": _rel_err(dv, rdv),
              "out rows": _row_rel_err(out, ref), "dv rows": _row_rel_err(dv, rdv)}
    # The bounds of the symmetric-head tests (tests/test_mosaic_gpu_attention.py): bf16 noise.
    assert errors["out"] < 2e-2 and errors["out rows"] < 2e-2, errors
    assert max(errors["dq"], errors["dk"], errors["dv"]) < 3e-2, errors
    assert errors["dv rows"] < 4e-2, errors
    return errors


CASES = [
    # batch, length, heads, kv_heads, sliding_window, docs
    pytest.param(2, 256, 4, 4, None, 0, id="causal"),
    pytest.param(1, 333, 4, 4, None, 4, id="documents-odd-length"),
    pytest.param(1, 200, 2, 2, None, 0, id="causal-odd-length"),
    pytest.param(2, 1024, 2, 2, None, 9, id="documents-9-1024"),
    pytest.param(1, 2048, 2, 2, None, 2, id="documents-2-2048"),
    pytest.param(1, 1024, 4, 1, None, 0, id="causal-1024-gqa4"),
    pytest.param(1, 777, 4, 2, 300, 6, id="swa-documents-777-gqa2"),
    pytest.param(1, 512, 2, 2, 100, 0, id="swa-100"),
]


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads,window,docs", CASES)
def test_forward_and_gradients_match_fp32_attention(batch, length, heads, kv_heads, window, docs):
    _check(batch, length, heads, kv_heads, window, docs)


@gpu
@pytest.mark.parametrize("order", ["interleaved", "shuffled"])
def test_unsorted_segment_ids_stay_exact(order):
    _check(1, 640, 2, 2, None, 7, order=order)


@gpu
def test_float16_matches():
    _check(1, 384, 2, 2, None, 3, dtype=jnp.float16)


@gpu
@pytest.mark.parametrize("options", [
    pytest.param(dict(overlap_dkv=True), id="overlap"),
    pytest.param(dict(heavy_first=True), id="heavy-first"),
    pytest.param(dict(unmask_documents=False), id="documents-always-masked"),
])
def test_kernel_options_match(monkeypatch, options):
    """The overlapped ``dk, dv`` body, the heavy-first launch order and the document-interior mask
    skipping turned off."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    monkeypatch.setattr(kernel_module, "_Config", functools.partial(kernel_module._Config, **options))
    _check(1, 777, 4, 2, None, 6)


@pytest.mark.parametrize("head_dim,window,documents,expected", [
    pytest.param(64, None, False, True, id="d64-causal"),
    pytest.param(128, None, False, True, id="d128-causal"),
    pytest.param(192, None, False, False, id="mla-causal"),
    pytest.param(192, None, True, False, id="mla-documents"),
    pytest.param(192, 300, False, False, id="mla-window"),
    pytest.param(128, None, True, False, id="d128-documents"),
    pytest.param(64, 129, False, False, id="d64-window"),
])
def test_overlapped_dkv_default_follows_the_head_dim(head_dim, window, documents, expected):
    """CPU: ``overlap_dkv=None`` resolves to the overlapped ``dk, dv`` body only for causal attention without
    documents or a window at q/k head dims 64 and 128. MLA's 192-wide q/k keep the serial body, because the
    overlapped one made their ranged backward 14-16% slower. An explicit value is always kept."""
    import dataclasses

    from jla.ops.mosaic_gpu.attention import _Config, _overlap_dkv

    cfg = _Config(scale=head_dim**-0.5, window=window, has_segments=documents)
    assert cfg.overlap_dkv is None and _overlap_dkv(cfg, head_dim) is expected
    for explicit in (True, False):
        assert _overlap_dkv(dataclasses.replace(cfg, overlap_dkv=explicit), head_dim) is explicit


@gpu
@pytest.mark.parametrize("batch,length,heads,kv_heads", [
    pytest.param(2, 256, 4, 4, id="causal"),
    pytest.param(1, 1024, 4, 1, id="causal-1024-gqa4"),
    pytest.param(1, 200, 2, 2, id="causal-odd-length"),
])
def test_default_backward_is_the_serial_dkv_body(monkeypatch, batch, length, heads, kv_heads):
    """Causal MLA without documents, where 64- and 128-wide heads take the overlapped ``dk, dv`` body by default:
    the default ``dq, dk, dv`` are bitwise those of ``overlap_dkv=False``."""
    import functools

    import jla.ops.mosaic_gpu.attention as kernel_module

    original = kernel_module._Config
    q, k, v, do = _inputs(length + 3, batch, length, heads, kv_heads)

    def grads(**options):
        monkeypatch.setattr(kernel_module, "_Config", functools.partial(original, **options))

        def loss(q, k, v):
            return jnp.sum(kernel_module.flash_attention(q, k, v).astype(jnp.float32) * do)

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)

    for name, default, serial in zip(("dq", "dk", "dv"), grads(), grads(overlap_dkv=False), strict=True):
        np.testing.assert_array_equal(np.asarray(default, np.float32), np.asarray(serial, np.float32),
                                      err_msg=name)


def test_unsupported_head_dims_fail_loudly():
    from jla.ops.mosaic_gpu.attention import flash_attention, supported

    q = jnp.zeros((1, 128, 2, 192), jnp.bfloat16)
    v = jnp.zeros((1, 128, 2, 96), jnp.bfloat16)
    assert "head dims (q/k 192, v 96)" in supported(q, q, v)
    with pytest.raises(ValueError, match="mosaic_gpu attention"):
        flash_attention(q, q, v)
    if _on_hopper():
        v = jnp.zeros((1, 128, 2, 128), jnp.bfloat16)
        assert supported(q, q, v) is None
        with pytest.raises(ValueError, match="backward='full'"):
            flash_attention(q, q, v, backward="full")


# The MLA layer ---------------------------------------------------------------------------------
def _layer(implementation, *, q_lora_rank=None, use_rope=True, window=None, heads=4, d_model=256):
    cfg = MLAConfig(num_heads=heads, kv_lora_rank=64, q_lora_rank=q_lora_rank, qk_nope_head_dim=128,
                    qk_rope_head_dim=64, v_head_dim=128, use_rope=use_rope, sliding_window=window,
                    implementation=implementation)
    layer = MLA(d_model, cfg, key=jax.random.key(0), std=0.05, norm_eps=1e-6, compute_dtype="bfloat16")
    return eqx.tree_at(lambda m: m.kv_a_norm.weight, layer, 1.0 + 0.1 * jnp.cos(jnp.arange(64, dtype=jnp.float32)))


def _layer_grads(layer, x, positions, seg, cotangent):
    params, static = eqx.partition(layer, eqx.is_inexact_array)

    def loss(params, x):
        out = eqx.combine(params, static)(x, positions, seg)
        return jnp.sum(out.astype(jnp.float32) * cotangent), out

    (grads, dx), out = jax.jit(jax.grad(loss, argnums=(0, 1), has_aux=True))(params, x)
    return out, grads, dx


@gpu
@pytest.mark.parametrize("implementation", ["mosaic_gpu", "auto"])
@pytest.mark.parametrize("variant", [
    pytest.param(dict(length=256, docs=0), id="causal"),
    pytest.param(dict(length=333, docs=3), id="documents-odd-length"),
    pytest.param(dict(length=512, docs=5, q_lora_rank=96), id="documents-lora-q"),
    pytest.param(dict(length=384, docs=0, use_rope=False), id="nope-kimi-linear"),
    pytest.param(dict(length=640, docs=4, window=200), id="swa-documents"),
])
def test_layer_on_the_kernel_matches_the_layer_on_xla(implementation, variant):
    variant = dict(variant)
    length, docs = variant.pop("length"), variant.pop("docs")
    batch, d_model = 2, 256
    kernel_layer = _layer(implementation, **variant)
    xla_layer = _layer("xla", **variant)  # same key: the same weights
    assert kernel_layer._mosaic_gpu(jnp.zeros((1, 128, 4, QK), jnp.bfloat16),
                                    jnp.zeros((1, 128, 4, QK), jnp.bfloat16),
                                    jnp.zeros((1, 128, 4, V), jnp.bfloat16))
    keys = jax.random.split(jax.random.key(length), 2)
    x = jax.random.normal(keys[0], (batch, length, d_model), jnp.float32)
    cotangent = jax.random.normal(keys[1], (batch, length, d_model), jnp.float32)
    seg = _segments(np.random.default_rng(docs), batch, length, docs) if docs else None
    if seg is None:
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
    else:  # positions restart in every document, as the packed data pipeline gives them
        starts = jnp.concatenate([jnp.ones((batch, 1), bool), seg[:, 1:] != seg[:, :-1]], axis=1)
        idx = jnp.arange(length)[None]
        positions = (idx - jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)).astype(jnp.int32)
    out, grads, dx = _layer_grads(kernel_layer, x, positions, seg, cotangent)
    ref_out, ref_grads, ref_dx = _layer_grads(xla_layer, x, positions, seg, cotangent)
    assert _rel_err(out, ref_out) < 2e-2
    assert _rel_err(dx, ref_dx) < 3e-2
    flat, _ = jax.tree_util.tree_flatten_with_path(grads)
    ref_flat = jax.tree.leaves(ref_grads)
    assert len(flat) == len(ref_flat) and len(flat) >= 5
    for (path, got), want in zip(flat, ref_flat, strict=True):
        assert _rel_err(got, want) < 3e-2, (jax.tree_util.keystr(path), _rel_err(got, want))


def test_layer_dispatch_without_a_hopper_gpu():
    """``"auto"`` takes the padded shared attention path where the kernel cannot run (here: fp32 or no
    H100), and ``"xla"`` / ``"cudnn"`` never take the kernel."""
    q = jnp.zeros((1, 128, 4, QK), jnp.float32)
    v = jnp.zeros((1, 128, 4, V), jnp.float32)
    assert not _layer("auto")._mosaic_gpu(q, q, v)
    assert not _layer("xla")._mosaic_gpu(q.astype(jnp.bfloat16), q.astype(jnp.bfloat16), v.astype(jnp.bfloat16))
    assert not _layer("cudnn")._mosaic_gpu(q.astype(jnp.bfloat16), q.astype(jnp.bfloat16), v.astype(jnp.bfloat16))
    assert _layer("mosaic_gpu")._mosaic_gpu(q, q, v)
    assert MLAConfig(num_heads=2, implementation="mosaic_gpu").implementation == "mosaic_gpu"


@pytest.mark.skipif(not _on_hopper() or jax.device_count() < 2, reason="needs two H100s")
@pytest.mark.parametrize("split", ["batch", "heads"])
def test_sharded_call_runs_per_device(split):
    """Under jit with the batch (data parallelism) or the heads (tensor parallelism) sharded over two
    GPUs, the asymmetric-head kernels run on each device's own rows or heads (no bf16 operand
    all-gather) and give the single-device values and gradients."""
    import re

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from jla.ops.mosaic_gpu.attention import flash_attention

    mesh = Mesh(np.array(jax.devices()[:2]).reshape(2, 1) if split == "batch"
                else np.array(jax.devices()[:2]).reshape(1, 2), ("data", "model"))
    spec = PartitionSpec("data") if split == "batch" else PartitionSpec("data", None, "model", None)
    tokens, rows = NamedSharding(mesh, spec), NamedSharding(mesh, PartitionSpec("data"))
    q, k, v, do = _inputs(11, 2, 1000, 4, 4)
    seg = _segments(np.random.default_rng(5), 2, 1000, 3)

    def grads(q, k, v, do, seg):
        def loss(q, k, v):
            out = flash_attention(q, k, v, segment_ids=seg)
            return jnp.sum(out.astype(jnp.float32) * do), out

        return jax.grad(loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)

    args = (q, k, v, do, seg)
    want = jax.jit(grads)(*args)
    shardings = (tokens,) * 4 + (rows,)
    sharded = jax.jit(grads, in_shardings=shardings)
    placed = [jax.device_put(x, s) for x, s in zip(args, shardings)]
    hlo = sharded.lower(*placed).compile().as_text()
    gathered = [g for g in re.findall(r"all-gather[^\n]*", hlo) if "bf16[" in g]
    assert not gathered, gathered[:2]
    got = sharded(*placed)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
        off = ~np.isclose(g, w, rtol=1e-5, atol=1e-5)
        assert off.mean() <= 1e-3 and _rel_err(g, w) < 2e-3, (int(off.sum()), _rel_err(g, w))


def test_reference_reduces_to_symmetric_heads():
    """The einsum reference equals XLA's attention when values have the query width (sanity check)."""
    keys = jax.random.split(jax.random.key(3), 3)
    q, k, v = (jax.random.normal(kk, (1, 40, 2, 16), jnp.float32) for kk in keys)
    seg = jnp.asarray([[0] * 15 + [1] * 25], jnp.int32)
    with jax.default_matmul_precision("highest"):  # no TF32 on GPU
        want = jax.nn.dot_product_attention(q, k, v, mask=(seg[:, :, None] == seg[:, None, :])[:, None],
                                            is_causal=True, implementation="xla")
    np.testing.assert_allclose(_reference(q, k, v, seg, None, 16**-0.5), want, rtol=1e-5, atol=1e-5)
