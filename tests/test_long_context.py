# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context safety of the release kernels: single calls past 2**31 elements (H100 only).

A kernel that forms ``row * row_stride`` (or any flattened index) in 32 bits reads or writes wrong
memory once a tensor passes 2**31 elements; the result is NaN or, worse, finite garbage. Every test
here runs ONE call at a size where a tensor the kernels touch has more than 2**31 elements (about
1.06M tokens at 2048 channels per token) and checks it against

* the same op chained over two halves with state passing (each half stays below 2**31 elements),
* the float32 reference op on windows at the start, across the 2**31-element boundary and at the end,
  started from a state computed below the limit (an independent oracle for the overflow region).

Opt-in (minutes per test and up to ~64 GB of GPU memory): ``pytest tests/test_long_context.py
--run-long-context``. Every comparison prints one ``LONGCTX`` line with its errors.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jla import ops
from jla.ops.reference import Mamba3State

pytestmark = pytest.mark.long_context

F32, BF16 = jnp.float32, jnp.bfloat16
HIGHEST = lax.Precision.HIGHEST
INT32_LIMIT = 2**31
T_BIG = 2**20 + 2**14  # 1,064,960 tokens: 2048 channels per token give 2,181,038,080 > 2**31 elements
SPLIT = T_BIG // 2  # 532,480, a multiple of 128: each half stays below 2**31 elements
WINDOW = 256  # reference windows
BOUNDARY = INT32_LIMIT // 2048  # token where a 2048-channel tensor passes 2**31 elements
# Mamba-3 with d_model 512, expand 2, 16 heads of 64, d_state 128, rope 0.5: the official module's fused
# in_proj row is 2 * 1024 + 2 * 128 + 3 * 16 + 32 = 2384 wide, so past 900,790 tokens that row-major tensor
# passes 2**31 elements (where state-spaces/mamba 2.3.2's Triton kernels, which form row * row_stride in
# 32 bits, read wrapped memory). Our layer never forms that row; the test runs both sides of the limit.
FUSED_ROW = 2384
FUSED_ROW_LIMIT = (INT32_LIMIT - 1) // FUSED_ROW  # 900,790


def _hopper() -> bool:
    try:
        device = jax.devices()[0]
    except RuntimeError:
        return False
    return device.platform == "gpu" and getattr(device, "compute_capability", None) == "9.0"


requires_hopper = pytest.mark.skipif(not _hopper(), reason="needs an sm_90 (H100) GPU")


# ---------------------------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------------------------


_STATS_BLOCK = 65536  # tokens per compared block (host float32 temporaries of ~0.5 GB)


def _host(x) -> np.ndarray:
    """A host copy: comparisons never run on the device, where a jitted block comparison taking the
    multi-GB operands compiled to 4-24 GiB allocations."""
    return np.asarray(jax.device_get(x))


def _stats(x, ref, *, start=None, stop=None, axis=1) -> np.ndarray:
    """``[max |x - ref|, max |ref|, mean |x - ref|, mean |ref|, all finite(x)]`` for ``x[start:stop]`` on
    ``axis``, block by block on host copies. NaN propagates through the maxima."""
    x, ref = _host(x), _host(ref)
    if x.ndim <= axis:  # scalars
        x, ref, axis = x.reshape(1), ref.reshape(1), 0
    if start is None:
        start, stop = 0, x.shape[axis]
    length = stop - start
    assert ref.shape[axis] == length, (ref.shape, axis, length)
    max_err, max_ref, sum_err, sum_ref, finite = 0.0, 0.0, 0.0, 0.0, True
    lead = (slice(None),) * axis
    for offset in range(0, length, _STATS_BLOCK):
        size = min(_STATS_BLOCK, length - offset)
        xs = x[lead + (slice(start + offset, start + offset + size),)].astype(np.float32)
        rs = ref[lead + (slice(offset, offset + size),)].astype(np.float32)
        diff = np.abs(xs - rs)
        block_max = float(diff.max())
        max_err = block_max if np.isnan(block_max) or np.isnan(max_err) else max(max_err, block_max)
        max_ref = max(max_ref, float(np.abs(rs).max()))
        sum_err += float(diff.sum(dtype=np.float64))
        sum_ref += float(np.abs(rs).sum(dtype=np.float64))
        finite = finite and bool(np.isfinite(xs).all())
    count = float(ref.size)
    return np.array([max_err, max_ref, sum_err / count, sum_ref / count, float(finite)])


def _close(name: str, x, ref, *, max_rel: float, mean_rel: float, start=None, stop=None, axis=1) -> None:
    max_err, max_ref, mean_err, mean_ref, finite = (float(v) for v in _stats(x, ref, start=start, stop=stop,
                                                                              axis=axis))
    rel_max, rel_mean = max_err / max(max_ref, 1e-30), mean_err / max(mean_ref, 1e-30)
    print(f"LONGCTX {name}: rel_max={rel_max:.3e} rel_mean={rel_mean:.3e} max_abs_err={max_err:.3e} "
          f"max_ref={max_ref:.3e} finite={bool(finite)}")
    assert finite == 1.0, f"{name}: non-finite values"
    assert rel_max <= max_rel, f"{name}: max error {rel_max:.3e} of max |ref| > {max_rel}"
    assert rel_mean <= mean_rel, f"{name}: mean error {rel_mean:.3e} of mean |ref| > {mean_rel}"


def _close_tree(name: str, x, ref, **tol) -> None:
    for i, (a, b) in enumerate(zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(ref), strict=True)):
        _close(f"{name}[{i}]", a, b, **tol)


# Same kernel, two halves vs one call: only the state hand-off and the summation order differ (the
# sequence-split kernels place their pieces by length), so bf16 rounding noise is allowed; an offset
# overflow gives O(1) errors or NaN.
SAME = dict(max_rel=2e-2, mean_rel=5e-3)
# bf16 kernel vs the float32 reference.
REF = dict(max_rel=5e-2, mean_rel=1e-2)


def _windows(length: int) -> tuple[int, ...]:
    """Reference windows: the start, across the 2**31-element boundary of a 2048-channel tensor, the end."""
    starts = [0, BOUNDARY - WINDOW // 2, length - WINDOW]
    return tuple(s for s in starts if 0 <= s <= length - WINDOW)


# ---------------------------------------------------------------------------------------------
# linear attention and state-space ops (ops.linear_attention)
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Spec:
    kind: str
    heads: int
    key_dim: int
    value_dim: int
    l2norm: bool = False
    rank: int = 0  # Mamba-3 MIMO rank (q, k [B, T, H, R, K])
    row_factor: int = 1  # kernel operand rows per token (RWKV-7: 2, its 64-row chunks hold 32 tokens)

    @property
    def per_token(self) -> int:
        """Channels per token of the largest tensor the kernels touch."""
        return self.row_factor * self.heads * max(self.rank, 1) * max(self.key_dim, self.value_dim)


SPECS = (
    Spec("simple_gla", 16, 128, 128),  # Mamba-2 SSD core
    Spec("gated_deltanet", 16, 128, 128, l2norm=True),
    Spec("deltanet", 16, 128, 128, l2norm=True),
    Spec("kda", 16, 128, 128, l2norm=True),
    Spec("gdn2", 16, 128, 128, l2norm=True),
    Spec("mamba3", 16, 128, 64),
    Spec("mamba3_mimo", 8, 128, 64, rank=2),
    Spec("rwkv7", 8, 128, 128, row_factor=2),
)


def _make_inputs(spec: Spec, first: int, count: int, seed: int = 0):
    """``(q, k, v, gates)`` for tokens ``[first * SPLIT, (first + count) * SPLIT)``, with gates as a dict of
    keyword inputs; bf16 operands, moderate decays. Each block of ``SPLIT`` tokens is drawn from its own
    ``fold_in`` key, so a half generated on its own equals that half of the whole sequence (no multi-GB
    slice copies)."""
    batch, heads, kd, vd = 1, spec.heads, spec.key_dim, spec.value_dim
    lead = (batch, SPLIT, heads) + ((spec.rank,) if spec.rank else ())
    head = (batch, SPLIT, heads)

    def block(key):
        ks = list(jax.random.split(key, 8))
        normal = lambda shape: jax.random.normal(ks.pop(), shape, F32)  # noqa: E731
        unit = lambda shape: jax.nn.sigmoid(normal(shape))  # noqa: E731
        decay = lambda shape: -0.02 - 0.2 * unit(shape)  # noqa: E731  log decay in (-0.22, -0.02)
        q = (normal(lead + (kd,)) * kd**-0.5).astype(BF16)
        k = (normal(lead + (kd,)) * kd**-0.5).astype(BF16)
        v = normal(lead + (vd,)).astype(BF16)
        if spec.kind == "simple_gla":
            gates = dict(g=decay(head))
        elif spec.kind == "gated_deltanet":
            gates = dict(g=decay(head), beta=unit(head))
        elif spec.kind == "deltanet":
            gates = dict(beta=unit(head))
        elif spec.kind == "kda":
            gates = dict(g=decay(head + (kd,)).astype(BF16), beta=unit(head))
        elif spec.kind == "gdn2":
            gates = dict(g=decay(head + (kd,)).astype(BF16), b=unit(head + (kd,)).astype(BF16),
                         w=unit(head + (vd,)).astype(BF16))
        elif spec.kind in ("mamba3", "mamba3_mimo"):
            dt = 0.01 + 0.09 * unit(head)
            lam = unit(head)
            g = -dt * (1.0 + 3.0 * unit(head))
            gates = dict(g=g, beta=(1.0 - lam) * dt * jnp.exp(g), gamma=lam * dt)
        elif spec.kind == "rwkv7":
            kk = normal(head + (kd,))
            kk = kk * lax.rsqrt(jnp.sum(kk * kk, axis=-1, keepdims=True))
            gates = dict(w=decay(head + (kd,)).astype(BF16), a=(-kk).astype(BF16),
                         b=(kk * unit(head + (1,))).astype(BF16))
        else:
            raise AssertionError(spec.kind)
        return q, k, v, gates

    @jax.jit
    def make(key):
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(first, first + count))
        blocks = jax.vmap(block)(keys)  # leaves [count, 1, SPLIT, ...]
        return jax.tree.map(lambda x: x.reshape((batch, count * SPLIT) + x.shape[3:]), blocks)

    return make(jax.random.PRNGKey(seed))


def _call(spec: Spec, impl: str, q, k, v, gates, initial_state=None):
    options = dict(use_qk_l2norm=True) if spec.l2norm else {}
    return ops.linear_attention(spec.kind, q, k, v, impl=impl, initial_state=initial_state,
                                output_final_state=True, **gates, **options)


@functools.partial(jax.jit, static_argnames=("spec", "impl", "start", "stop"))
def _call_slice(spec: Spec, impl: str, q, k, v, gates, initial_state, *, start: int, stop: int):
    """The op on tokens ``[start, stop)`` (sliced inside the jit) from ``initial_state``."""
    cut = lambda x: lax.slice_in_dim(x, start, stop, axis=1)  # noqa: E731
    return _call(spec, impl, cut(q), cut(k), cut(v), {n: cut(x) for n, x in gates.items()}, initial_state)


def _skip_unless_kernel(kind: str) -> None:
    if "mosaic_gpu" not in ops.available(kind):
        pytest.skip(f"no mosaic_gpu kernel registered for {kind!r} in this tree")


@requires_hopper
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.kind)
def test_linear_attention_one_call_past_int32(spec: Spec):
    _skip_unless_kernel(spec.kind)
    length = T_BIG
    assert length * spec.per_token > INT32_LIMIT and length == 2 * SPLIT
    o, final = _call_slice(spec, "mosaic_gpu", *_make_inputs(spec, 0, 2), None, start=0, stop=length)
    o = _host(o)  # the one call's output lives on the host from here; its inputs are freed
    # Chained halves, each generated on its own (each call stays below 2**31 elements), and float32
    # reference windows from states computed below the limit.
    half = _make_inputs(spec, 0, 1)
    o1, s1 = _call_slice(spec, "mosaic_gpu", *half, None, start=0, stop=SPLIT)
    _close(f"{spec.kind} one-call vs half 1", o, o1, start=0, stop=SPLIT, **SAME)
    del o1
    windows = _windows(length)
    for start in (w for w in windows if w + WINDOW <= SPLIT):
        state = None if start == 0 else _call_slice(spec, "mosaic_gpu", *half, None, start=0, stop=start)[1]
        ref, _ = _call_slice(spec, "reference", *half, state, start=start, stop=start + WINDOW)
        _close(f"{spec.kind} one-call vs f32 reference [{start}, {start + WINDOW})", o, ref, start=start,
               stop=start + WINDOW, **REF)
    del half
    half = _make_inputs(spec, 1, 1)
    o2, s2 = _call_slice(spec, "mosaic_gpu", *half, s1, start=0, stop=SPLIT)
    _close(f"{spec.kind} one-call vs half 2", o, o2, start=SPLIT, stop=length, **SAME)
    _close_tree(f"{spec.kind} final state one-call vs chained", final, s2, **SAME)
    del o2, s2
    for start in (w for w in windows if w >= SPLIT):
        local = start - SPLIT
        state = s1 if local == 0 else _call_slice(spec, "mosaic_gpu", *half, s1, start=0, stop=local)[1]
        ref, _ = _call_slice(spec, "reference", *half, state, start=local, stop=local + WINDOW)
        _close(f"{spec.kind} one-call vs f32 reference [{start}, {start + WINDOW})", o, ref, start=start,
               stop=start + WINDOW, **REF)


GRAD_KINDS = ("simple_gla", "gated_deltanet", "deltanet", "kda", "gdn2", "mamba3", "rwkv7")
# Backward memory at T_BIG: KDA at 16 heads of 128 needs more than 60 GB, so 32 heads of 64 keep 2048
# channels per token (every tensor still passes 2**31 elements) with half the per-chunk state memory. GDN-2
# has six [T, H, K] inputs: value dim 64 keeps q, k, g, b (2048 per token) past 2**31 and halves the states.
GRAD_SPECS = {"kda": Spec("kda", 32, 64, 64, l2norm=True), "gdn2": Spec("gdn2", 16, 128, 64, l2norm=True)}


def _loss(spec: Spec, q, k, v, gates, initial_state=None):
    """``sum(o * v)`` (float32) and the final state."""
    o, final = _call(spec, "mosaic_gpu", q, k, v, gates, initial_state)
    return jnp.sum(o.astype(F32) * lax.stop_gradient(v).astype(F32)), final


@functools.partial(jax.jit, static_argnames=("spec",))
def _grads_one_call(spec: Spec, inputs):
    """Gradients of ``sum(o * v)`` w.r.t. ``q, k, v`` and the gates of one call over the whole sequence."""
    return jax.grad(lambda x: _loss(spec, *x)[0])(inputs)


@functools.partial(jax.jit, static_argnames=("spec",))
def _grads_chained(spec: Spec, first, second):
    """The same gradients from two chained calls (the second starts from the first's final state)."""

    def loss(a, b):
        loss1, s1 = _loss(spec, *a)
        return loss1 + _loss(spec, *b, initial_state=s1)[0]

    return jax.grad(loss, argnums=(0, 1))(first, second)


@requires_hopper
@pytest.mark.parametrize("kind", GRAD_KINDS)
def test_linear_attention_grads_one_call_past_int32(kind: str):
    """The backward kernels on > 2**31-element inputs: one call vs two chained halves, compared block by
    block on the host (the one call's gradients leave the GPU before the halves are generated)."""
    _skip_unless_kernel(kind)
    spec = GRAD_SPECS.get(kind) or next(s for s in SPECS if s.kind == kind)
    assert T_BIG * spec.per_token > INT32_LIMIT and T_BIG == 2 * SPLIT
    one = [_host(g) for g in jax.tree_util.tree_leaves(_grads_one_call(spec, _make_inputs(spec, 0, 2)))]
    first, second = _grads_chained(spec, _make_inputs(spec, 0, 1), _make_inputs(spec, 1, 1))
    names = ["dq", "dk", "dv"] + [f"d{g}" for g in sorted(first[3])]  # tree_leaves orders dict keys
    failures = []
    for name, a, b1, b2 in zip(names, one, jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second),
                               strict=True):
        for label, ref, lo, hi in (("half 1", b1, 0, SPLIT), ("half 2", b2, SPLIT, T_BIG)):
            try:
                _close(f"{kind} {name} one-call vs chained {label}", a, ref, start=lo, stop=hi, **REF)
            except AssertionError as err:  # every comparison is printed before the test fails
                failures.append(str(err).splitlines()[0])
    assert not failures, failures


# ---------------------------------------------------------------------------------------------
# Mamba-3 SISO layer core with the fused angle / rotary kernels
# ---------------------------------------------------------------------------------------------


def _mamba3_core_inputs(length: int, heads: int, groups: int, key_dim: int, value_dim: int, angles: int,
                        seed: int = 0, angle_bias: float = 0.0):
    batch = 1

    @jax.jit
    def make(key):
        ks = list(jax.random.split(key, 12))
        normal = lambda shape: jax.random.normal(ks.pop(), shape, F32)  # noqa: E731
        q = normal((batch, length, groups, key_dim)).astype(BF16)  # C (RMS-normalized scale)
        k = normal((batch, length, groups, key_dim)).astype(BF16)  # B
        v = normal((batch, length, heads, value_dim)).astype(BF16)  # x
        z = normal((batch, length, heads, value_dim)).astype(BF16)
        dt = jax.nn.softplus(normal((batch, length, heads)) - 3.0)
        adt = -dt * (1.0 + 3.0 * jax.nn.sigmoid(normal((batch, length, heads))))
        trap = normal((batch, length, heads))
        qb = 1.0 + 0.1 * normal((heads, key_dim))
        kb = 1.0 + 0.1 * normal((heads, key_dim))
        ang = normal((batch, length, heads, angles)) + angle_bias
        d = normal((heads,))
        return q, k, v, adt, dt, trap, qb, kb, ang, d, z

    return make(jax.random.PRNGKey(seed))


@functools.partial(jax.jit, static_argnames=("impl", "start", "stop"))
def _core_slice(inputs, initial_state, *, impl: str, start: int, stop: int):
    from jla.layers.ssm import mamba3_siso

    q, k, v, adt, dt, trap, qb, kb, ang, d, z = inputs
    cut = lambda x: lax.slice_in_dim(x, start, stop, axis=1)  # noqa: E731
    return mamba3_siso(cut(q), cut(k), cut(v), cut(adt), cut(dt), cut(trap), qb, kb, cut(ang), d=d, z=cut(z),
                       initial_state=initial_state, output_final_state=True, impl=impl)


@requires_hopper
def test_mamba3_fused_core_one_call_past_int32():
    """``mamba3_siso`` on the fused angle/rotary + recurrence kernels; 32 heads of K 128, V 64: the
    rotated ``q, k`` (4096 per token) and ``x, z`` (2048 per token) pass 2**31 elements."""
    from jla.ops.mosaic_gpu import mamba3_fused

    length, heads, key_dim, value_dim = T_BIG, 32, 128, 64
    inputs = _mamba3_core_inputs(length, heads, 1, key_dim, value_dim, key_dim // 4)
    assert mamba3_fused.supports_fused(*inputs[:9], d=inputs[9], z=inputs[10])
    y, final = _core_slice(inputs, None, impl="mosaic_gpu", start=0, stop=length)
    y1, s1 = _core_slice(inputs, None, impl="mosaic_gpu", start=0, stop=SPLIT)
    _close("mamba3 fused one-call vs half 1", y, y1, start=0, stop=SPLIT, **SAME)
    del y1
    y2, s2 = _core_slice(inputs, s1, impl="mosaic_gpu", start=SPLIT, stop=length)
    _close("mamba3 fused one-call vs half 2", y, y2, start=SPLIT, stop=length, **SAME)
    del y2
    unit_circle = lambda a: jnp.stack([jnp.cos(a), jnp.sin(a)])  # noqa: E731  (angles are mod 2 pi)
    _close("mamba3 fused final angle one-call vs chained", unit_circle(final[0]), unit_circle(s2[0]),
           max_rel=1e-4, mean_rel=1e-4)
    _close_tree("mamba3 fused final state one-call vs chained", final[1], s2[1], **SAME)
    for start in _windows(length):
        state = None
        if 0 < start < SPLIT:
            _, state = _core_slice(inputs, None, impl="mosaic_gpu", start=0, stop=start)
        elif start >= SPLIT:
            _, state = _core_slice(inputs, s1, impl="mosaic_gpu", start=SPLIT, stop=start)
        ref, _ = _core_slice(inputs, state, impl="reference", start=start, stop=start + WINDOW)
        _close(f"mamba3 fused one-call vs f32 reference [{start}, {start + WINDOW})", y, ref, start=start,
               stop=start + WINDOW, **REF)


@requires_hopper
def test_mamba3_angle_does_not_drift():
    """The carried rotation angle is wrapped mod 2 pi: after ~1M tokens of ~0.1 rad steps (a total of
    ~1e5 rad) the fused kernel's final angle matches a float64 sum mod 2 pi."""
    length, heads, key_dim, value_dim = T_BIG, 1, 64, 64
    inputs = _mamba3_core_inputs(length, heads, 1, key_dim, value_dim, key_dim // 4, seed=1, angle_bias=1.5)
    _, (angle, _) = _core_slice(inputs, None, impl="mosaic_gpu", start=0, stop=length)
    _, _, _, _, dt, _, _, _, ang, _, _ = inputs
    inc = np.tanh(np.asarray(ang, np.float64)) * math.pi * np.asarray(dt, np.float64)[..., None]
    exact = np.mod(inc.sum(axis=1), 2 * math.pi)  # [B, H, S]
    err = np.abs(np.angle(np.exp(1j * (np.asarray(angle, np.float64) - exact))))
    total = float(np.abs(inc.sum(axis=1)).max())
    print(f"LONGCTX mamba3 angle drift: max |angle error| = {err.max():.3e} rad after {total:.3e} rad")
    assert total > 1e4
    assert err.max() < 1e-2


# ---------------------------------------------------------------------------------------------
# Softmax attention: our FA3-style kernel (causal, sliding window) and the default cuDNN path
# ---------------------------------------------------------------------------------------------


def _attention_inputs(length: int, heads: int, kv_heads: int, head_dim: int, seed: int = 0):
    @jax.jit
    def make(key):
        kq, kk, kv = jax.random.split(key, 3)
        q = (2.0 * jax.random.normal(kq, (1, length, heads, head_dim), F32)).astype(BF16)
        k = jax.random.normal(kk, (1, length, kv_heads, head_dim), F32).astype(BF16)
        v = jax.random.normal(kv, (1, length, kv_heads, head_dim), F32).astype(BF16)
        return q, k, v

    return make(jax.random.PRNGKey(seed))


@functools.partial(jax.jit, static_argnames=("start", "window"))
def _attention_rows(q, k, v, *, start: int, window: int | None):
    """Float32 causal (windowed) attention of the query rows ``[start, start + WINDOW)``, head by head."""
    heads, kv_heads, head_dim = q.shape[2], k.shape[2], q.shape[3]
    group = heads // kv_heads
    lo = 0 if window is None else max(0, start - window + 1)
    hi = start + WINDOW
    q_pos = start + jnp.arange(WINDOW)
    k_pos = lo + jnp.arange(hi - lo)
    visible = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        visible &= k_pos[None, :] > q_pos[:, None] - window
    rows = lax.slice_in_dim(q, start, hi, axis=1)

    def head(h):
        qh = lax.dynamic_index_in_dim(rows, h, axis=2, keepdims=False).astype(F32)
        kh = lax.slice_in_dim(lax.dynamic_index_in_dim(k, h // group, axis=2, keepdims=False), lo, hi, axis=1)
        vh = lax.slice_in_dim(lax.dynamic_index_in_dim(v, h // group, axis=2, keepdims=False), lo, hi, axis=1)
        s = jnp.einsum("btd,bld->btl", qh, kh.astype(F32), precision=HIGHEST) * head_dim**-0.5
        p = jax.nn.softmax(jnp.where(visible[None], s, -jnp.inf), axis=-1)
        return jnp.einsum("btl,bld->btd", p, vh.astype(F32), precision=HIGHEST)

    # Rematerialized per head: gradients through this would otherwise keep every head's [WINDOW, T] probabilities.
    return jnp.moveaxis(lax.map(jax.checkpoint(head), jnp.arange(heads)), 0, 2)  # [B, WINDOW, H, D]


@functools.partial(jax.jit, static_argnames=("impl", "window", "start", "stop"))
def _attention_slice(q, k, v, *, impl: str, window: int | None, start: int, stop: int):
    from jla.layers.attention import attention

    cut = lambda x: lax.slice_in_dim(x, start, stop, axis=1)  # noqa: E731
    return attention(cut(q), cut(k), cut(v), sliding_window=window, implementation=impl)


@requires_hopper
@pytest.mark.parametrize("impl,window", [("mosaic_gpu", None), ("mosaic_gpu", 4096), ("cudnn", None)],
                         ids=["mosaic_causal", "mosaic_swa4096", "cudnn_forward_causal"])
def test_attention_one_call_past_int32(impl: str, window: int | None):
    length, heads, kv_heads, head_dim = T_BIG, 16, 16, 128
    q, k, v = _attention_inputs(length, heads, kv_heads, head_dim)
    o = _attention_slice(q, k, v, impl=impl, window=window, start=0, stop=length)
    name = f"attention {impl} window={window}"
    for start in _windows(length):
        ref = _attention_rows(q, k, v, start=start, window=window)
        _close(f"{name} one-call vs f32 reference [{start}, {start + WINDOW})", o, ref, start=start,
               stop=start + WINDOW, **REF)
    if window is not None:  # a window needs only `window - 1` tokens of history: recompute the second half
        lo = SPLIT - window
        o2 = _attention_slice(q, k, v, impl=impl, window=window, start=lo, stop=length)
        _close(f"{name} one-call vs recomputed half 2", o, o2[:, window:], start=SPLIT, stop=length, **SAME)


@requires_hopper
def test_attention_grads_one_call_past_int32():
    """Our causal attention backward (dq and dk/dv kernels) in one 1M-token call against float32 gradients
    on windows (``dq`` at the start, across 2**31 and at the end; ``dk, dv`` of the last keys); the default
    path must not use cuDNN here, whose backward was wrong at this size. ``q``, ``o`` and ``dq`` pass 2**31
    elements (and ``dk, dv`` where the kernel forms them per query head)."""
    length, heads, kv_heads, head_dim = T_BIG, 16, 4, 128
    q, k, v = _attention_inputs(length, heads, kv_heads, head_dim)

    @functools.partial(jax.jit, static_argnames=("impl",))
    def grads(q, k, v, *, impl):
        from jla.layers.attention import attention

        weight = lax.stop_gradient(q).astype(F32)
        loss = lambda q, k, v: jnp.sum(attention(q, k, v, implementation=impl).astype(F32) * weight)  # noqa: E731
        return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    @functools.partial(jax.jit, static_argnames=("start",))
    def window_grads(q, k, v, *, start):
        """Float32 gradients of the same loss restricted to the query rows ``[start, start + WINDOW)``: exact
        ``dq`` for those rows and, for the last window, exact ``dk, dv`` of its keys (no later query sees them)."""
        weight = lax.stop_gradient(lax.slice_in_dim(q, start, start + WINDOW, axis=1)).astype(F32)
        loss = lambda q, k, v: jnp.sum(_attention_rows(q, k, v, start=start, window=None) * weight)  # noqa: E731
        dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        cut = lambda x: lax.slice_in_dim(x, start, start + WINDOW, axis=1)  # noqa: E731
        return cut(dq), cut(dk), cut(dv)

    failures = []

    def check(name, x, ref, **kwargs):  # every comparison is printed before any assertion fails
        try:
            _close(name, x, ref, **kwargs)
        except AssertionError as err:
            failures.append(str(err).splitlines()[0])

    from jla.layers.attention import CUDNN_MAX_QUERY_ELEMENTS, attention, resolve_attention_implementation

    # cuDNN's backward was wrong at this size (dq up to 3.5x its magnitude off, dk 12%): "auto" must avoid it
    # and an explicit request must fail loudly.
    assert q.size > CUDNN_MAX_QUERY_ELEMENTS
    assert resolve_attention_implementation("auto", q, k) == "mosaic_gpu"
    with pytest.raises(ValueError, match="cuDNN attention computes wrong gradients"):  # its forward is allowed
        jax.eval_shape(jax.grad(lambda q: attention(q, k, v, implementation="cudnn").astype(F32).sum()), q)
    results = {"mosaic_gpu": grads(q, k, v, impl="mosaic_gpu")}
    for start in _windows(length):
        ref = window_grads(q, k, v, start=start)
        names = ("dq", "dk", "dv") if start + WINDOW == length else ("dq",)
        for impl, got in results.items():
            for i, name in enumerate(names):
                check(f"attention {impl} {name} vs f32 reference [{start}, {start + WINDOW})", got[i], ref[i],
                      start=start, stop=start + WINDOW, **REF)
    assert not failures, failures


# ---------------------------------------------------------------------------------------------
# Layers: Mamba-3 (both sides of the 2384-wide row's 900,790-token limit, and past 2**31) and SWA attention
# ---------------------------------------------------------------------------------------------


@eqx.filter_jit
def _layer_call(layer, u):
    return layer(u)


@eqx.filter_jit
def _layer_step(layer, u, cache):
    return layer.step(u, cache)


@requires_hopper
@pytest.mark.parametrize("length", [FUSED_ROW_LIMIT, FUSED_ROW_LIMIT + 10, T_BIG],
                         ids=["row2384_limit", "row2384_limit_plus_10", "t_big"])
def test_mamba3_layer_one_call_vs_chunked_prefill(length: int):
    from jla.layers.ssm import Mamba3
    from jla.models.config import SSMConfig

    d_model = 512
    cfg = SSMConfig()  # d_state 128, expand 2, head_dim 64, one group, rope 0.5: a 2384-wide in_proj row
    layer = Mamba3(d_model, cfg, key=jax.random.PRNGKey(0), std=0.02, norm_eps=1e-5)
    assert 2 * 2 * d_model + 2 * cfg.d_state + 3 * layer.num_heads + layer.num_angles == FUSED_ROW
    u = jax.jit(lambda key: jax.random.normal(key, (1, length, d_model), F32).astype(BF16))(jax.random.PRNGKey(1))
    y = _layer_call(layer, u)
    half = (length // 2) // 64 * 64
    cache = layer.init_cache(1, BF16)
    y1, cache = _layer_step(layer, u[:, :half], cache)
    _close(f"mamba3 layer T={length} one-call vs prefill chunk 1", y, y1, start=0, stop=half, **SAME)
    del y1
    y2, cache = _layer_step(layer, u[:, half:], cache)
    _close(f"mamba3 layer T={length} one-call vs prefill chunk 2", y, y2, start=half, stop=length, **SAME)


@requires_hopper
def test_swa_attention_layer_one_call_vs_chunked():
    from jla.layers.attention import Attention
    from jla.models.config import AttentionConfig

    d_model, window, length = 2048, 4096, T_BIG  # the layer input itself passes 2**31 elements
    cfg = AttentionConfig(num_heads=16, num_kv_heads=4, head_dim=128, sliding_window=window,
                          implementation="mosaic_gpu", rope_theta=1_000_000.0)
    layer = Attention(d_model, cfg, key=jax.random.PRNGKey(0), std=0.02, norm_eps=1e-6)
    x = jax.jit(lambda key: jax.random.normal(key, (1, length, d_model), F32).astype(BF16))(jax.random.PRNGKey(1))
    positions = jnp.arange(length, dtype=jnp.int32)[None]
    call = eqx.filter_jit(lambda layer, x, p: layer(x, p))
    out = call(layer, x, positions)
    out1 = call(layer, x[:, :SPLIT], positions[:, :SPLIT])
    _close("swa layer one-call vs chunk 1", out, out1, start=0, stop=SPLIT, **SAME)
    del out1
    lo = SPLIT - window
    out2 = call(layer, x[:, lo:], positions[:, lo:])
    _close("swa layer one-call vs chunk 2 (window recomputed)", out, out2[:, window:], start=SPLIT, stop=length,
           **SAME)


# ---------------------------------------------------------------------------------------------
# LM-head cross entropy over a million tokens
# ---------------------------------------------------------------------------------------------


@requires_hopper
@pytest.mark.parametrize("impl", ["xla", "mosaic_gpu"])
def test_linear_cross_entropy_one_call_past_int32(impl: str):
    from jla.ops.cross_entropy import linear_cross_entropy

    if "impl" not in inspect.signature(linear_cross_entropy).parameters:
        if impl != "xla":
            pytest.skip("this tree has no fused cross-entropy kernel")
        options = {}
    else:
        options = dict(impl=impl)
    n, d_model, vocab = T_BIG, 2048, 32768  # hidden [N, D] passes 2**31 elements

    @jax.jit
    def make(key):
        kh, kw, kt = jax.random.split(key, 3)
        hidden = jax.random.normal(kh, (n, d_model), F32).astype(BF16)
        weight = (0.02 * jax.random.normal(kw, (vocab, d_model), F32)).astype(BF16)
        targets = jax.random.randint(kt, (n,), 0, vocab, jnp.int32)
        return hidden, weight, targets

    hidden, weight, targets = make(jax.random.PRNGKey(0))

    @functools.partial(jax.jit, static_argnames=("start", "stop"))
    def loss_and_grads(hidden, weight, targets, *, start, stop):
        def loss(h, w):
            return linear_cross_entropy(h, w, targets[start:stop], vocab_chunk=1024, **options).loss_sum

        return jax.value_and_grad(loss, argnums=(0, 1))(hidden[start:stop], weight)

    loss, (dh, dw) = loss_and_grads(hidden, weight, targets, start=0, stop=n)
    loss1, (dh1, dw1) = loss_and_grads(hidden, weight, targets, start=0, stop=SPLIT)
    _close(f"cross entropy {impl} dhidden one-call vs half 1", dh, dh1, start=0, stop=SPLIT, axis=0, **SAME)
    del dh1
    loss2, (dh2, dw2) = loss_and_grads(hidden, weight, targets, start=SPLIT, stop=n)
    _close(f"cross entropy {impl} dhidden one-call vs half 2", dh, dh2, start=SPLIT, stop=n, axis=0, **SAME)
    _close(f"cross entropy {impl} loss one-call vs halves", loss, loss1 + loss2, max_rel=1e-5, mean_rel=1e-5)
    _close(f"cross entropy {impl} dweight one-call vs halves", dw, dw1.astype(F32) + dw2.astype(F32), **SAME)
    # Exact per-token losses of the last tokens in float32 (beyond the 2**31-element boundary of hidden).
    tail = slice(n - WINDOW, n)
    logits = jnp.einsum("nd,vd->nv", hidden[tail].astype(F32), weight.astype(F32), precision=HIGHEST)
    ref = jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, targets[tail, None], axis=-1)[:, 0]
    last = loss_and_grads(hidden, weight, targets, start=n - WINDOW, stop=n)[0]
    _close(f"cross entropy {impl} tail loss vs f32", last, ref.sum(), max_rel=1e-3, mean_rel=1e-3)
