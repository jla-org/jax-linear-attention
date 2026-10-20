# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partitioning rules of the single-device kernels (``jla.distributed.kernel_sharding``).

``shard_layout_call``'s layouts beyond the attention / Mamba-3 ones (``"channels"`` parameters, ``(batch_dim,
head_dim)`` tuples, per-shard ``"shards"`` outputs, ``head_axes`` and the ``unsplit`` fallback) are checked on 4
CPU devices with plain-JAX stand-ins. With two or more GPUs visible (e.g. ``CUDA_VISIBLE_DEVICES=0,1
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/test_kernel_sharding_rules.py``) the real kernels run on a
2-device mesh, batch over ``dp`` or channels / ``d_ff`` / heads over ``tp``: the fused short conv, the fused conv +
q/k L2 norm, the FP8 fused SwiGLU passes (and the FP8 MLP around them), the fused SwiGLU GEMMs and the RWKV-7
bounded mode. Each compiled call has no all-gather, the kernel's custom call has the per-device shapes, and the
results match one GPU: bitwise for values computed per row and channel (the same arithmetic per element), to
float32 rounding for sums over the split dimension, to bf16 (FP8) noise where the per-device GEMM shapes differ
from one GPU's; the FP8 MLP is also checked against one GPU on each device's half of the batch (same shapes per
device), where its rows are bitwise equal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]

CPU_SCRIPT = textwrap.dedent(
    r"""
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from jla.distributed.kernel_sharding import (SHARDS, mesh_can_split, per_device_call, rule_axes,
                                                 shard_layout_call, use_rules)
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    assert len(jax.devices()) == 4, jax.devices()
    mesh = build_mesh(resolve_mesh_plan(4, dp=2, fsdp=1, tp=2), topology_aware=False)
    seen = []
    B, T, C, K = 4, 16, 8, 3
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(keys[0], (B, T, C))
    w = jax.random.normal(keys[1], (C, K))
    bias = jax.random.normal(keys[2], (C,))

    # A depthwise causal conv: weight and bias split with the channels ("channels"), the batch over dp.
    def conv(x, w, bias):
        seen.append(("conv", tuple(x.shape), tuple(w.shape)))
        xp = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
        return jnp.tanh(sum(xp[:, i:i + x.shape[1]] * w[:, i] for i in range(K)) + bias)

    def conv_call(x, w, bias, supports=None, unsplit=None):
        return shard_layout_call(conv, {"x": x, "w": w, "bias": bias},
                                 {"x": "token", "w": "channels", "bias": "channels"}, out_layouts="token",
                                 heads=C, head_axes=("tp",), supports=supports, unsplit=unsplit)

    def conv_loss(x, w, bias, supports=None):
        return jnp.sum(conv_call(x, w, bias, supports) ** 2)

    want = jax.value_and_grad(conv_loss, argnums=(0, 1, 2))(x, w, bias)
    placed = (jax.device_put(x, NamedSharding(mesh, P("dp", None, "tp"))),
              jax.device_put(w, NamedSharding(mesh, P("tp"))), jax.device_put(bias, NamedSharding(mesh, P("tp"))))
    seen.clear()
    with jax.set_mesh(mesh):
        assert mesh_can_split() and not per_device_call()
        got = jax.jit(jax.value_and_grad(conv_loss, argnums=(0, 1, 2)))(*placed)
        out = jax.jit(conv_call)(*placed)
    assert ("conv", (B // 2, T, C // 2), (C // 2, K)) in seen, seen
    assert out.sharding.spec == P("dp", None, "tp"), out.sharding.spec
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)

    # A predicate that rejects the channel split puts every axis on the batch (the weight then enters whole);
    # one that rejects every split runs ``unsplit`` on the global operands, which may return None.
    seen.clear()
    with jax.set_mesh(mesh):
        got = jax.jit(jax.value_and_grad(lambda *a: conv_loss(*a, supports=lambda x, w, bias: x.shape[2] == C),
                                         argnums=(0, 1, 2)))(*placed)
    assert ("conv", (B // 4, T, C), (C, K)) in seen, seen
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-5, atol=2e-5)
    calls = []

    def xla_conv(x, w, bias):
        calls.append(tuple(x.shape))
        return conv(x, w, bias)

    with jax.set_mesh(mesh):
        y = jax.jit(lambda *a: conv_call(*a, supports=lambda **_: False, unsplit=xla_conv))(*placed)
        none = conv_call(*placed, supports=lambda **_: False, unsplit=lambda **_: None)
    assert calls == [(B, T, C)] and none is None, (calls, none)
    np.testing.assert_allclose(np.asarray(y), np.asarray(conv(x, w, bias)), rtol=2e-5, atol=2e-5)

    # Like the FP8 SwiGLU pass: rows over dp, features over tp, a transposed copy as layout (1, 0) and the
    # amax as one partial per device ("shards").
    M, F = 32, 8
    h = jax.random.normal(keys[3], (M, 2, F))
    scale = jnp.float32(2.0)

    def quant(h, scale):
        seen.append(("quant", tuple(h.shape)))
        a = jnp.tanh(h[:, 0, :]) * h[:, 1, :]
        return a / scale, (a / scale).T, jnp.reshape(jnp.max(jnp.abs(a)), (1,))

    def quant_call(h, scale):
        return shard_layout_call(quant, {"h": h, "scale": scale}, {"h": "token", "scale": "replicated"},
                                 out_layouts=("state", (1, 0), SHARDS), heads=F, head_axes=rule_axes("mlp"))

    q0, qt0, amax0 = quant(h, scale)
    seen.clear()
    with jax.set_mesh(mesh):
        q, qt, amax = jax.jit(quant_call)(jax.device_put(h, NamedSharding(mesh, P("dp", None, "tp"))), scale)
    assert ("quant", (M // 2, 2, F // 2)) in seen, seen  # (and the global shapes, traced for the output layout)
    assert amax.shape == (4,), amax.shape
    assert q.sharding.spec == P("dp", "tp") and qt.sharding.spec == P("tp", "dp"), (q.sharding, qt.sharding)
    np.testing.assert_array_equal(np.asarray(q), np.asarray(q0))
    np.testing.assert_array_equal(np.asarray(qt), np.asarray(qt0))
    assert float(jnp.max(amax)) == float(amax0[0])
    with use_rules({"batch": ("dp",), "mlp": ()}):
        assert rule_axes("mlp") == () and rule_axes("batch") == ("dp",)
        seen.clear()
        with jax.set_mesh(mesh):  # no axis for d_ff: tp splits the rows too (a new function: jit caches traces)
            q, qt, amax = jax.jit(lambda h, s: quant_call(h, s))(jax.device_put(h, NamedSharding(mesh, P("dp"))), scale)
        assert ("quant", (M // 4, 2, F)) in seen, seen
        np.testing.assert_array_equal(np.asarray(qt), np.asarray(qt0))
    # No context mesh: one device's call, the function sees the global shapes.
    assert not mesh_can_split()
    print("RULESCPU " + json.dumps({"ok": True}))
    """
)


def test_layout_call_channels_transposed_and_shards_on_cpu(tmp_path: Path) -> None:
    script = tmp_path / "rules_cpu.py"
    script.write_text(CPU_SCRIPT)
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               XLA_FLAGS="--xla_force_host_platform_device_count=4", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RULESCPU "))
    assert json.loads(line.split(" ", 1)[1]) == {"ok": True}


# -- GPUs ------------------------------------------------------------------------------------------------------

two_gpus = pytest.mark.skipif(jax.default_backend() != "gpu" or len(jax.devices()) < 2, reason="needs two GPUs")


def _hopper_or_skip() -> None:
    if jax.devices()[0].compute_capability != "9.0":
        pytest.skip("the kernels need sm_90a")


def _mesh(axis: str):
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    return build_mesh(resolve_mesh_plan(2, fsdp=1, **{axis: 2}), jax.devices()[:2], topology_aware=False)


def _one_gpu(fn, args):
    """``jax.jit(fn)(*args)`` on the first GPU, under a one-device mesh (a single device's call)."""
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    mesh = build_mesh(resolve_mesh_plan(1, fsdp=1), jax.devices()[:1], topology_aware=False)
    with jax.set_mesh(mesh):
        return jax.jit(fn)(*(jax.device_put(a, jax.devices()[0]) for a in args))


def _on_mesh(fn, args, specs, axis: str):
    """``(results, compiled HLO text)`` of ``jax.jit(fn)`` on the 2-GPU mesh with ``args`` placed as ``specs``."""
    from jax.sharding import NamedSharding

    mesh = _mesh(axis)
    placed = [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs, strict=True)]
    with jax.set_mesh(mesh):
        jitted = jax.jit(fn)
        out = jitted(*placed)
        text = jitted.lower(*placed).compile().as_text()
    return out, text


def _kernel_lines(text: str, target: str) -> list[str]:
    return [line for line in text.splitlines() if "custom-call(" in line and target in line]


def _shape(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(d) for d in shape) + "]"


def _rel(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-30)


_MOSAIC = 'custom_call_target="mosaic_gpu'
_TRITON = 'custom_call_target="__gpu$xla.gpu.triton"'


@two_gpus
@pytest.mark.parametrize("fused_norm", [False, True], ids=["conv", "conv_l2norm"])
def test_short_conv_kernels_on_two_gpus_match_one_gpu(fused_norm: bool) -> None:
    """The fused short conv (``ShortConv(impl="mosaic_gpu")``) and the fused conv + q/k L2 norm with documents
    and an initial state, batch split over ``dp`` and channels (whole 128-channel heads) over ``tp``."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from jla.layers.conv import ShortConv
    from jla.layers.conv_l2norm import conv_l2norm

    _hopper_or_skip()
    B, T, K, HD = 4, 1024, 4, 128
    C = 1024 if fused_norm else 256  # the L2-norm kernels need 512 channels per device at head_dim 128
    ks = jax.random.split(jax.random.key(5), 5)
    conv = ShortConv(C, K, activation=True, use_bias=True, init="uniform", key=ks[0], impl="mosaic_gpu")
    x = jax.random.normal(ks[1], (B, T, C), jnp.bfloat16)
    state = jax.random.normal(ks[2], (B, K - 1, C), jnp.bfloat16)
    seg = jnp.asarray(np.repeat([[0] * 600 + [1] * 424], B, axis=0), jnp.int32)
    dy = jax.random.normal(ks[3], (B, T, C), jnp.float32)

    def loss(weight, bias, x, state, seg, dy):
        layer = eqx.tree_at(lambda m: (m.weight, m.bias), conv, (weight, bias))
        if fused_norm:
            out = conv_l2norm(layer, x, HD, segment_ids=seg, state=state)
            assert out is not None, "the fused conv + L2 norm declined the call"
            y, new_state = out
        else:
            y, new_state = layer(x, segment_ids=seg, state=state)
        return jnp.sum(y.astype(jnp.float32) * dy) + jnp.sum(new_state.astype(jnp.float32) ** 2), y

    grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)
    args = (conv.weight, conv.bias, x, state, seg, dy)
    one = _one_gpu(grad, args)
    report = {}
    for axis in ("dp", "tp"):
        if axis == "dp":
            specs = (P(), P(), P("dp"), P("dp"), P("dp"), P("dp"))
            local = (B // 2, T, C)
        else:
            specs = (P("tp"), P("tp"), P(None, None, "tp"), P(None, None, "tp"), P(), P(None, None, "tp"))
            local = (B, T, C // 2)
        many, text = _on_mesh(grad, args, specs, axis)
        lines = _kernel_lines(text, _MOSAIC)
        assert "all-gather" not in text, axis
        assert lines and any(_shape(local) in line for line in lines), (axis, lines[:2])
        assert not any(_shape((B, T, C)) in line for line in lines), (axis, lines[:2])
        diffs = {label: _rel(a, b) for label, a, b in zip(
            ("y", "dweight", "dbias", "dx", "dstate"), (one[0][1], *one[1]), (many[0][1], *many[1]), strict=True)}
        report[axis] = diffs
        np.testing.assert_allclose(float(many[0][0]), float(one[0][0]), rtol=1e-5)
        # Per row and channel the kernels do the same float32 arithmetic on each shard: bitwise equal.
        for label in ("y", "dx", "dstate"):
            assert diffs[label] == 0.0, (axis, label, diffs)
        # The weight and bias gradients are sums over rows (per-block partials summed by XLA, then over dp).
        assert diffs["dweight"] <= 1e-5 and diffs["dbias"] <= 1e-5, (axis, diffs)
    print(("CONVL2SHARD " if fused_norm else "CONVSHARD ") + json.dumps(report))


@two_gpus
def test_fp8_swiglu_passes_on_two_gpus_match_one_gpu() -> None:
    """The two fused FP8 SwiGLU passes (Pallas Triton) per shard: tokens over ``dp`` or ``d_ff`` over ``tp``,
    the transposed copies and the amax bitwise equal to one GPU; then the FP8 MLP around them (dp)."""
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from jla.layers import fp8_swiglu as fs
    from jla.layers.fp8 import Fp8State

    _hopper_or_skip()
    M, F = 4096, 1024
    ks = jax.random.split(jax.random.key(6), 8)
    h = (jax.random.normal(ks[0], (M, 2, F), jnp.float32) * 3).astype(jnp.bfloat16)
    da = jax.random.normal(ks[1], (M, F), jnp.bfloat16)
    scale, gscale = jnp.float32(0.02), jnp.float32(2e-3)

    def passes(h, da, scale, gscale):
        return fs.swiglu_quantize(h, scale), fs.swiglu_backward_quantize(h, da, gscale, jnp.bfloat16)

    def reference(h, da, scale, gscale):
        return fs._swiglu_quantize_pallas(h, scale), fs._swiglu_backward_quantize_pallas(h, da, gscale, jnp.bfloat16)

    one = _one_gpu(reference, (h, da, scale, gscale))
    report = {}
    for axis in ("dp", "tp"):
        specs = ((P("dp"), P("dp"), P(), P()) if axis == "dp" else (P(None, None, "tp"), P(None, "tp"), P(), P()))
        local = (M // 2, 2, F) if axis == "dp" else (M, 2, F // 2)
        many, text = _on_mesh(passes, (h, da, scale, gscale), specs, axis)
        lines = _kernel_lines(text, _TRITON)
        assert "all-gather" not in text, axis
        assert len(lines) >= 2 and all(_shape(local) in line for line in lines), (axis, lines)
        for label, a, b in zip(("q", "qt", "amax", "dq", "dqt", "damax"), jax.tree.leaves(one), jax.tree.leaves(many),
                               strict=True):
            np.testing.assert_array_equal(np.asarray(b, np.float32), np.asarray(a, np.float32), err_msg=f"{axis} {label}")
        report[axis] = {"kernel_lines": len(lines), "local_shape": list(local)}

    # The FP8 MLP (delayed scaling) on x [B, T, D], weights and states replicated, batch over dp.
    B, T, D = 4, 1024, 512
    x = jax.random.normal(ks[2], (B, T, D), jnp.bfloat16)
    w_gu = jax.random.normal(ks[3], (D, 2, F), jnp.float32) * 0.05
    w_dn = jax.random.normal(ks[4], (F, D), jnp.float32) * 0.05
    dy = jax.random.normal(ks[5], (B, T, D), jnp.float32)
    st = Fp8State.init(16)
    st_gu = Fp8State(st.input_scale * 0.05, st.input_amax_history + 4.0, st.kernel_scale * 0.001,
                     st.kernel_amax_history + 0.2, st.grad_scale * 1e-4, st.grad_amax_history + 1e-2)
    st_dn = Fp8State(st.input_scale * 0.02, st.input_amax_history + 8.0, st.kernel_scale * 0.001,
                     st.kernel_amax_history + 0.2, st.grad_scale * 1e-4, st.grad_amax_history + 1e-2)

    def mlp_loss(x, w_gu, w_dn, st_gu, st_dn, dy):
        y = fs.fp8_swiglu_mlp(x, w_gu, w_dn, st_gu, st_dn)
        return jnp.sum(y.astype(jnp.float32) * dy), y

    grad = jax.value_and_grad(mlp_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)
    args = (x, w_gu, w_dn, st_gu, st_dn, dy)
    one = _one_gpu(grad, args)
    many, text = _on_mesh(grad, args, (P("dp"), P(), P(), P(), P(), P("dp")), "dp")  # states: one sharding per pytree
    lines = _kernel_lines(text, _TRITON)
    assert "all-gather" not in text
    assert len(lines) >= 2 and all(_shape((B * T // 2, 2, F)) in line for line in lines), lines
    labels = ("y", "dx", "dw_gu", "dw_dn")
    diffs = {label: _rel(a, b) for label, a, b in zip(
        labels, (one[0][1], *one[1][:3]), (many[0][1], *many[1][:3]), strict=True)}
    # The next delayed-scaling state: the amaxes of sharded tensors are global maxima.
    for a, b in zip(jax.tree.leaves(one[1][3:]), jax.tree.leaves(many[1][3:]), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-2)
    # Against the full batch on one GPU the GEMMs differ in shape (half the tokens per device: other cuBLASLt
    # tiles), and a one-ulp change of a bf16 GEMM output can move its e4m3 copy by one FP8 step: FP8 noise.
    report["mlp_vs_full_batch"] = diffs
    assert all(value <= 5e-2 for value in diffs.values()), diffs
    # Against one GPU on each device's half of the batch (the same shapes per device): the rows bitwise equal,
    # the weight gradients the sum of the two halves'.
    halves = [_one_gpu(grad, (x[i * B // 2:(i + 1) * B // 2], w_gu, w_dn, st_gu, st_dn,
                              dy[i * B // 2:(i + 1) * B // 2])) for i in range(2)]
    ref = (jnp.concatenate([h[0][1] for h in halves]), jnp.concatenate([h[1][0] for h in halves]),
           halves[0][1][1] + halves[1][1][1], halves[0][1][2] + halves[1][1][2])
    same_shapes = {label: _rel(a, b) for label, a, b in zip(labels, ref, (many[0][1], *many[1][:3]), strict=True)}
    report["mlp_vs_halves"] = same_shapes
    assert same_shapes["y"] == 0.0 and same_shapes["dx"] == 0.0, same_shapes
    assert same_shapes["dw_gu"] <= 1e-5 and same_shapes["dw_dn"] <= 1e-5, same_shapes
    print("FP8SHARD " + json.dumps(report))


@two_gpus
def test_fused_swiglu_gemm_on_two_gpus_matches_one_gpu() -> None:
    """The fused SwiGLU GEMMs per shard (tokens over ``dp``, weights replicated); under ``tp`` the rule
    declines (the XLA path keeps the feature split)."""
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from jla.ops.mosaic_gpu import fused_gemm

    _hopper_or_skip()
    B, T, D, F = 4, 512, 512, 1280  # distinct sizes, so the operand shapes in the HLO are unambiguous
    ks = jax.random.split(jax.random.key(7), 4)
    x = jax.random.normal(ks[0], (B, T, D), jnp.bfloat16)
    gate_up = jax.random.normal(ks[1], (D, 2, F), jnp.float32) * 0.05
    w_down = jax.random.normal(ks[2], (F, D), jnp.float32) * 0.05
    dy = jax.random.normal(ks[3], (B, T, D), jnp.float32)

    def loss(x, gate_up, w_down, dy, sharded):
        if sharded:
            y = fused_gemm.swiglu_mlp_sharded(x, gate_up, w_down, compute_dtype=jnp.bfloat16, recompute=True)
            assert y is not None, "the per-shard rule declined the call"
        else:
            y = fused_gemm.swiglu_mlp(x, gate_up, w_down, compute_dtype=jnp.bfloat16, recompute=True)
        return jnp.sum(y.astype(jnp.float32) * dy), y

    args = (x, gate_up, w_down, dy)
    one = _one_gpu(jax.value_and_grad(lambda *a: loss(*a, False), argnums=(0, 1, 2), has_aux=True), args)
    many, text = _on_mesh(jax.value_and_grad(lambda *a: loss(*a, True), argnums=(0, 1, 2), has_aux=True), args,
                          (P("dp"), P(), P(), P("dp")), "dp")
    lines = _kernel_lines(text, _MOSAIC)
    assert "all-gather" not in text
    assert lines and not any(_shape((B * T, D)) in line for line in lines), lines[:2]
    assert any(_shape((B * T // 2, D)) in line for line in lines), lines[:2]
    diffs = {label: _rel(a, b) for label, a, b in zip(
        ("y", "dx", "dgate_up", "dw_down"), (one[0][1], *one[1]), (many[0][1], *many[1]), strict=True)}
    assert all(value <= 2e-2 for value in diffs.values()), diffs
    with jax.set_mesh(_mesh("tp")):
        declined = fused_gemm.swiglu_mlp_sharded(x, gate_up, w_down)
    assert declined is None
    print("GEMMSHARD " + json.dumps(diffs))


@two_gpus
def test_rwkv7_bounded_mode_on_two_gpus_matches_one_gpu() -> None:
    """RWKV-7's bounded mode (``lower_bound``, the layers' default) goes through the dispatcher's token-layout
    rule: per shard over ``dp`` (batch) or ``tp`` (heads), no all-gather, one GPU's result to bf16 noise."""
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from jla.ops import linear_attention

    _hopper_or_skip()
    B, T, H, K = 4, 512, 4, 64
    ks = jax.random.split(jax.random.key(8), 7)
    q, k, v = (jax.random.normal(ks[i], (B, T, H, K), jnp.float32).astype(jnp.bfloat16) for i in range(3))
    w = -0.6065 * jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, K)))
    kk = jax.random.normal(ks[4], (B, T, H, K))
    kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
    a_gate = jax.nn.sigmoid(jax.random.normal(ks[5], (B, T, H, K)))
    seg = jnp.asarray(np.repeat([[0] * 300 + [1] * 212], B, axis=0), jnp.int32)
    do = jax.random.normal(ks[6], (B, T, H, K), jnp.float32)

    def loss(q, k, v, w, kk, a_gate, seg, do):
        o, _ = linear_attention("rwkv7", q, k, v, w=w, a=-kk, b=kk * a_gate, scale=1.0, impl="mosaic_gpu",
                                chunk_size=32, lower_bound=-0.61, segment_ids=seg)
        return jnp.sum(o.astype(jnp.float32) * do), o

    grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)
    args = (q, k, v, w, kk, a_gate, seg, do)
    one = _one_gpu(grad, args)
    report = {}
    for axis in ("dp", "tp"):
        tok = P("dp") if axis == "dp" else P(None, None, "tp")
        many, text = _on_mesh(grad, args, (tok,) * 6 + (P("dp") if axis == "dp" else P(), tok), axis)
        assert "all-gather" not in text, axis
        assert _kernel_lines(text, _MOSAIC), axis
        diffs = {label: _rel(a, b) for label, a, b in zip(
            ("o", "dq", "dk", "dv", "dw", "dkk", "da"), (one[0][1], *one[1]), (many[0][1], *many[1]), strict=True)}
        report[axis] = diffs
        np.testing.assert_allclose(float(many[0][0]), float(one[0][0]), rtol=1e-4)
        assert all(value <= 2e-2 for value in diffs.values()), (axis, diffs)
    print("RWKV7SHARD " + json.dumps(report))


def test_kernel_lines_parser() -> None:
    text = ('%c = bf16[2,1024,256]{2,1,0} custom-call(bf16[2,1024,256]{2,1,0} %p), custom_call_target="mosaic_gpu_v2"\n'
            '%d = f32[4] all-reduce(f32[4] %e)\n')
    assert _kernel_lines(text, _MOSAIC) == [text.splitlines()[0]]
    assert _shape((2, 1024, 256)) in _kernel_lines(text, _MOSAIC)[0]
    assert re.search(r"all-reduce", text)
