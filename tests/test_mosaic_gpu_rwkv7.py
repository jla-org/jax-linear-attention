# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RWKV-7 (``kind="rwkv7"``) with the Mosaic GPU state passes, against the reference op.

Outputs and final states against the token-by-token reference, every gradient (``q, k, v, w,
a, b`` and the initial state) against autodiff of the chunked op, for odd ``T``, ``K, V`` in
{64, 128}, packed documents, strong decays and no initial state; the sub-chunk score matrices
against the pairwise formula; the ``supports`` predicate and the registration. The kernel tests
need an H100; the score and predicate tests run anywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import assert_close

from jla import ops
from jla.ops import linear_attention
from jla.ops.mosaic_gpu import rwkv7 as mg
from jla.ops.mosaic_gpu.vector_gate_ops import is_hopper
from jla.ops.rwkv7 import rwkv7 as rwkv7_reference
from jla.ops.rwkv7 import rwkv7_chunked

F32, BF16 = jnp.float32, jnp.bfloat16
gpu = pytest.mark.skipif(not is_hopper(), reason="needs an H100 (compute capability 9.0)")


def _inputs(seed, B, T, H, K, V, *, strong=False, w_const=None):
    """RWKV-7 style inputs (``a = -kk``, ``b = kk * gate``), ``q, k, v`` bf16, the rest f32."""
    ks = jax.random.split(jax.random.key(seed), 9)
    shape = (B, T, H, K)
    q = jax.random.normal(ks[0], shape).astype(BF16)
    k = (0.5 * jax.random.normal(ks[1], shape)).astype(BF16)
    v = jax.random.normal(ks[2], (B, T, H, V)).astype(BF16)
    if w_const is not None:
        w = jnp.full(shape, w_const, F32)
    elif strong:
        w = -jnp.exp(1.5 * jax.random.normal(ks[3], shape))  # some channels decay by e^-20 per token
    else:
        w = -0.6065306597126334 * jax.nn.sigmoid(2.0 * jax.random.normal(ks[3], shape))
    kk = jax.random.normal(ks[4], shape)
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    gate = jax.nn.sigmoid(jax.random.normal(ks[5], shape))
    h0 = 0.3 * jax.random.normal(ks[6], (B, H, K, V))
    do = jax.random.normal(ks[7], (B, T, H, V))
    dht = jax.random.normal(ks[8], (B, H, K, V))
    return dict(q=q, k=k, v=v, w=w, a=-kk, b=kk * gate, h0=h0, do=do, dht=dht)


def _segments(B, T):
    """Documents starting mid-chunk, at a chunk boundary (token 32) and one token long."""
    ids = np.zeros((B, T), np.int32)
    for row in range(B):
        for start in (5 + row, 32, 33, 47, 64 + 3 * row):
            if start < T:
                ids[row, start:] += 1
    return jnp.asarray(ids)


def _args(x):
    return tuple(x[n] for n in ("q", "k", "v", "w", "a", "b"))


def test_decayed_scores_match_pairwise():
    """The sub-chunk factorisation equals the exact pairwise sum (strong decays included)."""
    ks = jax.random.split(jax.random.key(0), 3)
    shape = (2, 3, mg.CHUNK, 16)
    x, y = jax.random.normal(ks[0], shape), jax.random.normal(ks[1], shape)
    g = jnp.cumsum(-jnp.exp(1.5 * jax.random.normal(ks[2], shape)), axis=-2)
    g_exc = jnp.concatenate([jnp.zeros_like(g[..., :1, :]), g[..., :-1, :]], axis=-2)
    t = jnp.arange(mg.CHUNK)
    for strict, gx in ((False, g), (True, g_exc)):
        got = mg._decayed_scores(x, gx, y, g, strict=strict, precision=jax.lax.Precision.HIGHEST)
        mask = (t[:, None] > t[None, :]) if strict else (t[:, None] >= t[None, :])
        diff = jnp.where(mask[:, :, None], gx[..., :, None, :] - g[..., None, :, :], -jnp.inf)
        want = jnp.einsum("...tk,...tik,...ik->...ti", x, jnp.exp(diff), y, precision=jax.lax.Precision.HIGHEST)
        assert bool(jnp.isfinite(got).all())
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_diag_block_gradients_match_autodiff():
    """The hand-written VJP of the exact diagonal blocks equals autodiff of the pairwise formula."""
    ks = jax.random.split(jax.random.key(3), 5)
    shape = (2, 3, 4, mg.SUB_CHUNK, 16)
    x, y = jax.random.normal(ks[0], shape), jax.random.normal(ks[1], shape)
    ct = jax.random.normal(ks[2], shape[:-1] + (mg.SUB_CHUNK,))
    g = jnp.cumsum(-jnp.exp(1.5 * jax.random.normal(ks[3], shape)), axis=-2)
    gx = g + 0.3 * jax.random.uniform(ks[4], shape)  # any gx with gx_t - g_i <= 0 on the kept entries
    hi = jax.lax.Precision.HIGHEST
    for strict in (False, True):
        def plain(x, gx, y, g, strict=strict):
            return jnp.einsum("...tk,...tik,...ik->...ti", x, mg._block_decay(gx, g, strict), y, precision=hi)

        got = jax.vjp(lambda *a: mg._diag_blocks(*a, strict, hi), x, gx, y, g)[1](ct)
        want = jax.vjp(plain, x, gx, y, g)[1](ct)
        for name, a, b in zip(["dx", "dgx", "dy", "dg"], got, want):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4, err_msg=name)


def test_unit_lower_inverse_matches_triangular_solve():
    a = jnp.tril(0.4 * jax.random.normal(jax.random.key(1), (3, mg.CHUNK, mg.CHUNK)), -1)
    eye = jnp.eye(mg.CHUNK)
    got = mg._unit_lower_inverse(a, jax.lax.Precision.HIGHEST)
    want = jax.lax.linalg.triangular_solve(eye - a, jnp.broadcast_to(eye, a.shape), left_side=True, lower=True,
                                           unit_diagonal=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)


def _forward_check(x, *, segment_ids=None, initial_state=True, precision=None):
    h0 = x["h0"] if initial_state else None
    run = jax.jit(lambda *a: mg.rwkv7(*a, initial_state=h0, output_final_state=True, segment_ids=segment_ids,
                                      precision=precision))
    o, final = run(*_args(x))
    f32 = tuple(t.astype(F32) for t in _args(x))
    o_ref, final_ref = jax.jit(lambda *a: rwkv7_reference(*a, initial_state=h0, output_final_state=True,
                                                          segment_ids=segment_ids))(*f32)
    assert o.dtype == BF16 and final.dtype == F32
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")


@gpu
@pytest.mark.parametrize(
    "shape", [(2, 100, 2, 64, 64), (1, 256, 2, 128, 128), (2, 75, 3, 128, 64), (1, 1, 2, 64, 64)],
    ids=["t100_k64v64", "t256_k128v128", "t75_k128v64", "t1"],
)
def test_forward_matches_reference(shape):
    _forward_check(_inputs(1, *shape))


@gpu
@pytest.mark.parametrize("value_dim", [64, 128], ids=["v64", "v128"])
def test_wide_value_tiles(value_dim):
    """``B H = 36`` puts the forward state pass on its 64- (V = 64) or 128-wide (V = 128) value tiles."""
    x = _inputs(7, 3, 70, 12, 64, value_dim)
    _forward_check(x, segment_ids=_segments(3, 70))
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x), (0, 3, 6)))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16), (0, 3, 6)))(
        *(t.astype(F32) for t in _args(x)), x["h0"])
    for name, g_got, g_want in zip(["dq", "dw", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
def test_long_sequence():
    """64 chunks forward and 32 chunks of gradients: the bf16 state-pass operands do not drift."""
    _forward_check(_inputs(8, 1, 2048, 2, 64, 64))
    x = _inputs(9, 1, 1024, 2, 64, 64)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x), (0, 1, 2, 3, 4, 5, 6)))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16), (0, 1, 2, 3, 4, 5, 6)))(
        *(t.astype(F32) for t in _args(x)), x["h0"])
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


# The RWKV-7 layers' call (jla.layers.rwkv7._OP_OPTIONS: lower_bound=-0.61, the default bounded
# precision BF16X3) and the same call with f32 products (HIGHEST).
_BOUNDED_PRECISIONS = pytest.mark.parametrize("precision", [None, jax.lax.Precision.HIGHEST], ids=["layer", "highest"])


@gpu
@_BOUNDED_PRECISIONS
@pytest.mark.parametrize("case", ["plain", "packed", "k128v64_t33"])
def test_bounded_mode_matches_reference(case, precision):
    """The call the layers make (bounded mode at RWKV7_LOWER_BOUND): outputs, states, gradients."""
    from jla.layers.rwkv7 import _OP_OPTIONS

    (B, T, K, V) = (1, 33, 128, 64) if case == "k128v64_t33" else (2, 140, 64, 64)
    x = _inputs(10, B, T, 2, K, V)
    options = dict(_OP_OPTIONS, precision=precision)
    if case == "packed":
        options["segment_ids"] = _segments(B, T)
    ref_options = {k: v for k, v in options.items() if k == "segment_ids"}
    f32 = tuple(t.astype(F32) for t in _args(x))
    o, final = jax.jit(lambda *a: mg.rwkv7(*a, initial_state=x["h0"], output_final_state=True, **options))(*_args(x))
    o_ref, final_ref = rwkv7_reference(*f32, initial_state=x["h0"], output_final_state=True, **ref_options)
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, **options), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, **ref_options), argnums))(*f32, x["h0"])
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
@_BOUNDED_PRECISIONS
@pytest.mark.parametrize("dims", [64, 128], ids=["k64v64", "k128v128"])
def test_bounded_mode_at_most_negative_bound(dims, precision):
    """Every token at ``w = MIN_LOWER_BOUND``, the largest factors the bounded mode allows: outputs,
    states and every gradient (``dw`` included) within the default mode's tolerances."""
    B, T, bound = 2, 140, mg.MIN_LOWER_BOUND
    x = _inputs(18, B, T, 2, dims, dims, w_const=bound)
    f32 = tuple(t.astype(F32) for t in _args(x))
    options = dict(lower_bound=bound, precision=precision)
    o, final = jax.jit(lambda *a: mg.rwkv7(*a, initial_state=x["h0"], output_final_state=True, **options))(
        *_args(x))
    o_ref, final_ref = jax.jit(lambda *a: rwkv7_reference(*a, initial_state=x["h0"], output_final_state=True))(*f32)
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, **options), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16), argnums))(*f32, x["h0"])
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
@pytest.mark.parametrize("shape", [(2, 100, 2), (1, 33, 3), (2, 160, 2)], ids=["t100", "t33_h3", "t160_odd_chunks"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_prep_kernel_matches_xla_operands(shape, packed):
    """The bounded mode's forward kernel (rwkv7_prep) builds the XLA path's state-pass operands, and its
    VJP (the XLA one with the kernel's saved T and A_ak) is the XLA path's."""
    B, T, H = shape
    x = _inputs(31, B, T, H, 64, 64)
    doc = mg._doc_ids(_segments(B, T) if packed else None, B, T)
    assert mg._uses_prep_kernel(64, 64, None, True)
    assert not mg._uses_prep_kernel(64, 64, jax.lax.Precision.HIGHEST, True)
    got = jax.jit(mg._prep_operands_impl, static_argnums=7)(*_args(x), doc.astype(F32), 0.7)
    xla = lambda *a: mg._operands(*a, 0.7, doc, mg.BOUNDED_PRECISION, True)[0]  # noqa: E731
    want = jax.jit(xla)(*_args(x))
    for name, g_got, g_want in zip(["kg", "w", "u", "g_carry", "qg", "aqk"], got, want):
        assert g_got.shape == g_want.shape and g_got.dtype == g_want.dtype, name
        assert_close(g_got, g_want, 8e-3, name)
    cts = tuple(jax.random.normal(jax.random.key(40 + i), y.shape, F32).astype(y.dtype) for i, y in enumerate(want))

    def pullback(fn):
        return jax.jit(lambda *a: jax.vjp(fn, *a)[1](cts))(*_args(x))

    got_grads = pullback(lambda *a: mg._prep_operands(*a, doc.astype(F32), 0.7))
    for name, g_got, g_want in zip("qkvwab", got_grads, pullback(xla)):
        assert g_got.dtype == g_want.dtype, name
        assert_close(g_got, g_want, 8e-3, f"d{name}")  # bf16 dq, dk, dv: one rounding step is 4e-3 of max


@gpu
@pytest.mark.parametrize("shape", [(2, 100, 2), (1, 33, 3), (2, 160, 2), (1, 64, 1)],
                         ids=["t100", "t33_h3", "t160_odd_chunks", "t64"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("initial_state", [True, False], ids=["h0", "no_h0"])
def test_kernel_backward_matches_xla_backward(shape, packed, initial_state):
    """The bounded mode's backward kernel (rwkv7_bwd, with bwd_state) gives the gradients of the XLA VJP
    path (the same forward kernel, the per-chunk backward in XLA), with segment resets and initial and
    final states."""
    from jla.layers.rwkv7 import _OP_OPTIONS

    B, T, H = shape
    x = _inputs(41, B, T, H, 64, 64)
    seg = _segments(B, T) if packed else None
    h0 = x["h0"] if initial_state else None
    options = dict(scale=0.7, initial_state=h0, output_final_state=True, segment_ids=seg, precision=None,
                   lower_bound=_OP_OPTIONS["lower_bound"])

    def loss(kernel_backward):
        def run(q, k, v, w, a, b, h):
            o, final = mg._run(q, k, v, w, a, b, **(options | dict(initial_state=h)), kernel_backward=kernel_backward)
            return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(final * x["dht"])

        return jax.jit(jax.grad(run, tuple(range(7)) if initial_state else tuple(range(6))))

    got = loss(True)(*_args(x), h0)
    want = loss(False)(*_args(x), h0)
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert g_got.dtype == g_want.dtype and g_got.shape == g_want.shape, name
        assert bool(jnp.isfinite(g_got).all()), name
        assert_close(g_got, g_want, 8e-3, name)  # bf16 dq, dk, dv: one rounding step is 4e-3 of max


@gpu
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_kernels_are_deterministic(monkeypatch, packed):
    """rwkv7_prep's and rwkv7_bwd's two warpgroups share SMEM buffers and hand operands and partial results
    over behind mbarriers, so a missing wait would make their outputs depend on timing: run again and again
    on the same inputs (a grid of 1,024 CTAs, several waves per SM), the kernels give bit-identical tiles
    and gradients."""
    from jla.ops.mosaic_gpu import rwkv7_bwd, rwkv7_prep

    B, T, H = 2, 2048, 16
    x = _inputs(53, B, T, H, 64, 64)
    doc_f = mg._doc_ids(_segments(B, T) if packed else None, B, T).astype(F32)
    _, residuals = mg._kernel_path_fwd(*_args(x), doc_f, x["h0"], 0.7)
    tiles = [jax.block_until_ready(rwkv7_prep.prep(*residuals[0], scale=0.7, save=True)) for _ in range(4)]
    for i in range(9):  # kg, w, u, qg, aqk and the saved T and A_ak (hi, lo)
        first = np.asarray(tiles[0][i])
        assert np.abs(first.astype(np.float32)).max() > 0, f"prep output {i}"
        for again in tiles[1:]:
            np.testing.assert_array_equal(np.asarray(again[i]), first, err_msg=f"prep output {i}")
    real, runs = rwkv7_bwd.bwd, []

    def repeated(*args, **kwargs):
        runs.extend(jax.block_until_ready(real(*args, **kwargs)) for _ in range(4))
        return runs[0]

    monkeypatch.setattr(rwkv7_bwd, "bwd", repeated)
    mg._kernel_path_bwd(0.7, residuals, (x["do"], x["dht"]))
    assert len(runs) == 4
    for i, name in enumerate(["dq", "dk", "dv", "da", "db", "dw"]):
        first = np.asarray(runs[0][i])
        assert np.isfinite(first.astype(np.float32)).all() and np.abs(first.astype(np.float32)).max() > 0, name
        for again in runs[1:]:
            np.testing.assert_array_equal(np.asarray(again[i]), first, err_msg=name)


@gpu
@pytest.mark.parametrize("dims", [64, 128], ids=["k64_kernels", "k128_xla"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_bounded_mode_nearly_parallel_keys(dims, packed):
    """Consecutive ``kk`` nearly parallel with a large gate and slow decay put ``A_ab`` near ``-a`` on every
    entry. The WY inverse's power series over 32 rows then grows to ~1e7 before cancelling to a ``T`` of
    order one; in ~16-bit split products that gave wrong ``W``/``U`` and an overflowing state pass (NaN
    losses training rwkv7-0.1b on FineWeb-Edu; benchmarks/rwkv7.md). The layers' call must match the
    reference: the operands, outputs, final state and every gradient."""
    from jla.layers.rwkv7 import _OP_OPTIONS

    B, T, H = 2, 512, 2
    x = _inputs(51, B, T, H, dims, dims)
    ks = jax.random.split(jax.random.key(52), 2)
    kk = jax.random.normal(ks[0], (1, 1, H, dims)) + 0.05 * jax.random.normal(ks[1], (B, T, H, dims))
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    x.update(w=jnp.full((B, T, H, dims), -1e-3, F32), a=-kk, b=0.95 * kk)
    seg = _segments(B, T) if packed else None
    options = dict(_OP_OPTIONS, segment_ids=seg)
    f32 = tuple(t.astype(F32) for t in _args(x))
    doc = mg._doc_ids(seg, B, T)  # W and U against the XLA operands with f32 products
    want = jax.jit(lambda *a: mg._operands(*a, 1.0, doc, jax.lax.Precision.HIGHEST, True)[0])(*_args(x))
    if dims == 64:  # the forward kernel (the layers' path at head dims 64)
        got = jax.jit(mg._prep_operands_impl, static_argnums=7)(*_args(x), doc.astype(F32), 1.0)
    else:  # the XLA operands with the bounded mode's default products (BF16X3)
        got = jax.jit(lambda *a: mg._operands(*a, 1.0, doc, mg.BOUNDED_PRECISION, True)[0])(*_args(x))
    for name, i in (("w_rows", 1), ("u_rows", 2)):
        assert_close(got[i], want[i], 8e-3, name)
    o, final = jax.jit(lambda *a: mg.rwkv7(*a, initial_state=x["h0"], output_final_state=True, **options))(*_args(x))
    o_ref, final_ref = jax.jit(lambda *a: rwkv7_reference(*a, initial_state=x["h0"], output_final_state=True,
                                                          segment_ids=seg))(*f32)
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, **options), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, segment_ids=seg), argnums))(*f32, x["h0"])
    # The gradients reach ~1e3 here (slow decay over 512 tokens) and carry the bf16 state passes'
    # rounding: da, db within 1.8e-2 of their max on both paths (before the fix the kernels' W was off by
    # 600 times its magnitude and the XLA path's da by 2.1e-2).
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 2.5e-2, name)


@gpu
@pytest.mark.parametrize("kernel_backward", [True, False], ids=["kernel_bwd", "xla_bwd"])
def test_kernel_backward_bf16_gates_against_reference(kernel_backward):
    """bf16 ``w, a, b`` (as an FLA model passes them) on the head-dim-64 kernel path with either backward,
    packed, ``scale != 1``: outputs against the token oracle, every gradient (dtypes included) against
    autodiff of the f32 chunked op."""
    from jla.layers.rwkv7 import _OP_OPTIONS

    def op(q, k, v, w, a, b, **kw):
        kw = dict(initial_state=None, output_final_state=False, segment_ids=None, precision=None) | kw
        return mg._run(q, k, v, w, a, b, kernel_backward=kernel_backward, **kw)

    B, T, scale = 2, 130, 0.37
    x = _inputs(43, B, T, 2, 64, 64)
    x |= {n: x[n].astype(BF16) for n in ("w", "a", "b")}
    seg = _segments(B, T)
    f32 = tuple(t.astype(F32) for t in _args(x))
    options = dict(_OP_OPTIONS, scale=scale, segment_ids=seg)
    o, final = jax.jit(lambda *a: op(*a, initial_state=x["h0"], output_final_state=True, **options))(*_args(x))
    o_ref, final_ref = rwkv7_reference(*f32, initial_state=x["h0"], output_final_state=True, scale=scale,
                                       segment_ids=seg)
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(op, x, **options), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, scale=scale, segment_ids=seg), argnums))(
        *f32, x["h0"])
    for name, arg, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], (*_args(x), x["h0"]), got, want):
        assert g_got.dtype == arg.dtype, name
        assert_close(g_got, g_want, 1.5e-2, name)


def _group_norm(o):
    """RWKV-7's per-head group norm without affine terms (eps = head size x 1e-5)."""
    o = o.astype(F32)
    mu = o.mean(-1, keepdims=True)
    var = jnp.square(o - mu).mean(-1, keepdims=True)
    return (o - mu) * jax.lax.rsqrt(var + o.shape[-1] * 1e-5)


@gpu
@pytest.mark.parametrize("case", ["mixed", "saturated", "packed_k128"])
def test_layer_call_under_group_norm_cotangent(case):
    """The layers' call under the structured cotangent of the group norm after the op (orthogonal to
    each head's output), where bf16 score operands lost dq/dk accuracy: every gradient's norm error
    against the f32 chunked op is at most 1.25x the exact mode's plus 1e-3 (both about 4e-3 on the H100;
    the plain-bf16 bounded variant measured 1.1e-2 on dq/dk). Checked for the layers' call and, for heads
    of 64, for the rwkv7_bwd kernel backward explicitly."""
    from jla.layers.rwkv7 import _OP_OPTIONS

    B, T, K = (2, 777, 128) if case == "packed_k128" else (2, 1024, 64)
    x = _inputs(21, B, T, 2, K, K, w_const=-0.6065306597126334 if case == "saturated" else None)
    seg = None
    if case == "packed_k128":
        ids = np.zeros((B, T), np.int32)
        for start in (100, 333, 334, 512):
            ids[:, start:] += 1
        seg = jnp.asarray(ids)
    ct = jax.random.normal(jax.random.key(22), x["do"].shape)

    def loss(fn, **options):
        def run(q, k, v, w, a, b, h0):
            o, final = fn(q, k, v, w, a, b, initial_state=h0, output_final_state=True, segment_ids=seg, **options)
            return jnp.sum(_group_norm(o) * ct) + jnp.sum(final * x["dht"])

        return jax.jit(jax.grad(run, tuple(range(7))))

    f32 = tuple(t.astype(F32) for t in _args(x))
    want = loss(rwkv7_chunked, chunk_size=16)(*f32, x["h0"])
    def kernel_bwd(q, k, v, w, a, b, **kw):  # the head-dim-64 calls with the rwkv7_bwd kernel, whatever the default
        return mg._run(q, k, v, w, a, b, scale=1.0, precision=None, kernel_backward=True, **kw)

    exact = loss(mg.rwkv7)(*_args(x), x["h0"])
    rel = lambda a, b: float(jnp.linalg.norm((a.astype(F32) - b).ravel()) / jnp.linalg.norm(b.ravel()))  # noqa: E731
    for got in (loss(mg.rwkv7, **_OP_OPTIONS)(*_args(x), x["h0"]), loss(kernel_bwd, **_OP_OPTIONS)(*_args(x), x["h0"])):
        for name, g_got, g_exact, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, exact, want):
            assert rel(g_got, g_want) <= 1.25 * rel(g_exact, g_want) + 1e-3, (name, rel(g_got, g_want),
                                                                               rel(g_exact, g_want))


@gpu
def test_scale_and_bf16_gates():
    """``scale != 1`` and bf16 ``w, a, b`` (the dtypes an FLA model passes) with packed documents."""
    B, T, scale = 2, 100, 0.37
    x = _inputs(12, B, T, 2, 64, 64)
    x |= {n: x[n].astype(BF16) for n in ("w", "a", "b")}
    seg = _segments(B, T)
    f32 = tuple(t.astype(F32) for t in _args(x))
    o, final = jax.jit(lambda *a: mg.rwkv7(*a, scale=scale, initial_state=x["h0"], output_final_state=True,
                                           segment_ids=seg))(*_args(x))
    o_ref, final_ref = jax.jit(lambda *a: rwkv7_reference(*a, scale=scale, initial_state=x["h0"],
                                                          output_final_state=True, segment_ids=seg))(*f32)
    assert_close(o, o_ref, 1e-2, "o")
    assert_close(final, final_ref, 1e-2, "final")
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, scale=scale, segment_ids=seg), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, scale=scale, segment_ids=seg), argnums))(
        *f32, x["h0"])
    for name, arg, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], (*_args(x), x["h0"]), got, want):
        assert g_got.dtype == arg.dtype, name
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
def test_highest_precision_option():
    _forward_check(_inputs(6, 2, 100, 2, 64, 64, strong=True), segment_ids=_segments(2, 100),
                   precision=jax.lax.Precision.HIGHEST)


@gpu
def test_highest_precision_gradients():
    B, T = 2, 100
    x = _inputs(14, B, T, 2, 64, 64, strong=True)
    seg, hi = _segments(B, T), jax.lax.Precision.HIGHEST
    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, segment_ids=seg, precision=hi), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, segment_ids=seg), argnums))(
        *(t.astype(F32) for t in _args(x)), x["h0"])
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
def test_gradients_without_final_state_wide_k128():
    """Only ``o`` carries a cotangent (no final state), ``K = V = 128`` at ``B H = 39`` and a
    document that starts at the last real token of the padded last chunk."""
    B, T, H = 3, 75, 13
    x = _inputs(13, B, T, H, 128, 128)
    ids = np.asarray(_segments(B, T)).copy()
    ids[:, T - 1:] += 1
    seg = jnp.asarray(ids)

    def loss(fn, **options):
        def run(q, k, v, w, a, b, h0):
            o, final = fn(q, k, v, w, a, b, initial_state=h0, segment_ids=seg, **options)
            assert final is None
            return jnp.sum(o.astype(F32) * x["do"])

        return run

    argnums = (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(loss(mg.rwkv7), argnums))(*_args(x), x["h0"])
    want = jax.jit(jax.grad(loss(rwkv7_chunked, chunk_size=16), argnums))(*(t.astype(F32) for t in _args(x)), x["h0"])
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
@pytest.mark.parametrize("strong", [False, True], ids=["plain", "strong_decay"])
@pytest.mark.parametrize("initial_state", [True, False], ids=["h0", "no_h0"])
def test_segments_and_decays_match_reference(strong, initial_state):
    B, T = 2, 150
    x = _inputs(2, B, T, 2, 64, 64, strong=strong)
    _forward_check(x, segment_ids=_segments(B, T), initial_state=initial_state)


def _loss(fn, x, **options):
    def loss(q, k, v, w, a, b, h0):
        o, final = fn(q, k, v, w, a, b, initial_state=h0, output_final_state=True, **options)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(final * x["dht"])

    return loss


@gpu
@pytest.mark.parametrize(
    "case", ["plain", "packed", "strong", "no_h0", "k128v64"],
)
def test_gradients_match_chunked_autodiff(case):
    K, V = (128, 64) if case == "k128v64" else (64, 64)
    B, T = 2, 140
    x = _inputs(3, B, T, 2, K, V, strong=case == "strong")
    options = dict(segment_ids=_segments(B, T)) if case == "packed" else {}
    h0 = None if case == "no_h0" else x["h0"]
    argnums = (0, 1, 2, 3, 4, 5) if h0 is None else (0, 1, 2, 3, 4, 5, 6)
    got = jax.jit(jax.grad(_loss(mg.rwkv7, x, **options), argnums))(*_args(x), h0)
    f32 = tuple(t.astype(F32) for t in _args(x))
    want = jax.jit(jax.grad(_loss(rwkv7_chunked, x, chunk_size=16, **options), argnums))(*f32, h0)
    for name, g_got, g_want in zip(["dq", "dk", "dv", "dw", "da", "db", "dh0"], got, want):
        assert g_got.shape == g_want.shape, name
        assert_close(g_got, g_want, 1.5e-2, name)


@gpu
def test_dispatcher_runs_the_kernel_by_name():
    x = _inputs(4, 1, 64, 2, 64, 64)
    gates = dict(w=x["w"], a=x["a"], b=x["b"])
    o, _ = jax.jit(lambda q, k, v: linear_attention("rwkv7", q, k, v, impl="mosaic_gpu", **gates))(
        x["q"], x["k"], x["v"])
    o_ref, _ = rwkv7_reference(*(t.astype(F32) for t in _args(x)))
    assert_close(o, o_ref, 1e-2, "o")


def test_registered_for_auto():
    assert "mosaic_gpu" in ops.available("rwkv7")
    assert ops._REGISTRY["mosaic_gpu"]["rwkv7"][2] is True  # qualified: impl="auto" picks it when supported


@gpu
def test_auto_picks_the_kernel_only_for_supported_calls():
    """bf16 ``q, k, v`` without ``chunk_size`` run the kernel; ``chunk_size=16`` or f32 inputs the chunked op."""
    x = _inputs(6, 1, 70, 2, 64, 64)
    gates = dict(w=x["w"], a=x["a"], b=x["b"])
    seg = _segments(1, 70)

    def run(impl, q, k, v, **kw):
        return jax.jit(lambda q, k, v: linear_attention("rwkv7", q, k, v, impl=impl, segment_ids=seg,
                                                        initial_state=x["h0"], output_final_state=True,
                                                        **gates, **kw))(q, k, v)

    qkv = (x["q"], x["k"], x["v"])
    for got, want in zip(run("auto", *qkv), run("mosaic_gpu", *qkv)):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    for got, want in zip(run("auto", *qkv, chunk_size=16), run("chunked", *qkv, chunk_size=16)):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    f32 = tuple(t.astype(F32) for t in qkv)
    for got, want in zip(run("auto", *f32), run("chunked", *f32)):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


@gpu
def test_bf16_layer_runs_the_kernel(monkeypatch):
    """A bf16 RWKV-7 block with ``chunk_size=32`` runs the kernel under ``impl="auto"`` and matches the
    same block on the chunked op (outputs, final state and the gradients of every parameter)."""
    import equinox as eqx

    from jla.layers.rwkv7 import RWKV7Block

    fn, supports, auto = ops._REGISTRY["mosaic_gpu"]["rwkv7"]
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)

    monkeypatch.setitem(ops._REGISTRY["mosaic_gpu"], "rwkv7", (counted, supports, auto))
    keys = jax.random.split(jax.random.key(11), 2)
    x = jax.random.normal(keys[1], (2, 100, 128))
    seg = _segments(2, 100)

    def run(impl):
        block = RWKV7Block(128, 256, head_dim=64, layer_idx=0, n_layers=2, key=keys[0], lora_dims=(32, 32, 32, 32),
                           chunk_size=32, impl=impl)  # compute dtype bf16 (the default)
        # Every parameter off its initial value (o_proj, the channel-mix value and the LoRA downs start at
        # zero, as upstream), so the output and every gradient depend on the kernel.
        params, static = eqx.partition(block, eqx.is_inexact_array)
        leaves, treedef = jax.tree.flatten(params)
        pkeys = jax.random.split(jax.random.key(5), len(leaves))
        leaves = [p + 0.05 * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, pkeys)]
        block = eqx.combine(jax.tree.unflatten(treedef, leaves), static)

        def loss(block, x):
            y, _, cache = block(x, None, seg)
            return jnp.sum(jnp.sin(y.astype(F32))) + jnp.sum(cache.state**2), (y, cache.state)

        (_, out), grads = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))(block, x)
        return out, grads

    (y, state), grads = run("auto")
    assert calls, "impl='auto' did not run the Mosaic GPU kernel"
    n_calls = len(calls)
    (y_ref, state_ref), grads_ref = run("chunked")
    assert len(calls) == n_calls
    assert_close(y, y_ref, 2e-2, "y")
    assert_close(state, state_ref, 2e-2, "state")
    for g, g_ref in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(grads_ref)):
        assert_close(g, g_ref, 3e-2, "grad")


def test_supports_predicate():
    x = _inputs(5, 1, 64, 2, 64, 64)
    q, k, v, w, a, b = _args(x)
    reason = mg.unsupported_reason
    assert "unsupported options" in reason(q, k, v, w, a, b, {"use_qk_l2norm": True})
    if not is_hopper():
        assert "sm_90" in reason(q, k, v, w, a, b, {})
        assert not mg.supports_rwkv7(q, k, v, w, a, b)
        return
    assert reason(q, k, v, w, a, b, {}) is None
    assert "cu_seqlens" in reason(q, k, v, w, a, b, {"cu_seqlens": jnp.array([0, 64])})
    assert "chunk_size" in reason(q, k, v, w, a, b, {"chunk_size": 16})
    assert "lower_bound" in reason(q, k, v, w, a, b, {"lower_bound": -3.0})
    assert "lower_bound" in reason(q, k, v, w, a, b, {"lower_bound": 0.0})
    assert reason(q, k, v, w, a, b, {"lower_bound": -0.6}) is None
    assert "precision" in reason(q, k, v, w, a, b, {"precision": jax.lax.Precision.DEFAULT})
    assert reason(q, k, v, w, a, b, {"precision": jax.lax.Precision.HIGHEST}) is None
    assert "bounded mode" in reason(q, k, v, w, a, b, {"precision": mg.BF16X3})  # needs lower_bound
    assert reason(q, k, v, w, a, b, {"precision": mg.BF16X3, "lower_bound": -0.61}) is None
    assert "precision" in reason(q, k, v, w, a, b, {"precision": jax.lax.DotAlgorithmPreset.BF16_BF16_F32})
    with pytest.raises(ValueError, match="precision"):
        mg.rwkv7(q, k, v, w, a, b, precision=jax.lax.Precision.DEFAULT)
    assert "bfloat16" in reason(q.astype(F32), k, v, w, a, b, {})
    assert "float32 or bfloat16" in reason(q, k, v, w.astype(jnp.float16), a, b, {})  # untested gate dtype
    small = jnp.zeros((1, 64, 2, 32), BF16)
    assert "head dims" in reason(small, small, small, small, small, small, {})
    assert "segment_ids" in reason(q, k, v, w, a, b, {"segment_ids": jnp.zeros((1, 63), jnp.int32)})
    assert "initial_state" in reason(q, k, v, w, a, b, {"initial_state": jnp.zeros((1, 2, 64, 32))})
    with pytest.raises(ValueError, match="chunk_size"):
        mg.rwkv7(q, k, v, w, a, b, chunk_size=16)
