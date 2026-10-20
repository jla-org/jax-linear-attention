# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layout and fusion switches of benchmarks/zero_copy_layouts.md: same values with the switch on and off.

``LinearAttentionConfig.qk_conv_barrier`` (the delta-rule mixers Gated DeltaNet, KDA, GDN-2 and DeltaNet, with
``qk_l2norm``) only changes how XLA fuses the short convolutions and the q/k normalization; outputs and
every parameter gradient
match the default (chunked ops on any backend, the Mosaic GPU kernels on an H100). The helpers of
``benchmarks/profile_layout_copies.py`` classify HLO data movement as the page describes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.linear_attention import KDA, DeltaNet, GatedDeltaNet, GatedDeltaNet2
from jla.models.config import LinearAttentionConfig

F32, BF16 = jnp.float32, jnp.bfloat16


def _hopper() -> bool:
    try:
        d = jax.devices()[0]
    except RuntimeError:
        return False
    return d.platform == "gpu" and getattr(d, "compute_capability", None) == "9.0"


def _close(a, b, tol, name):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    scale = max(float(np.abs(b).max()), 1e-6)
    err = float(np.abs(a - b).max()) / scale
    assert err <= tol, f"{name}: {err:.3e} > {tol}"


def _segments(B, T):
    ids = np.zeros((B, T), np.int32)
    ids[0, T // 3:] = 1
    ids[-1, T // 2:] = 2
    return jnp.asarray(ids)


@pytest.mark.parametrize("impl", ["chunked", pytest.param("mosaic_gpu", marks=pytest.mark.skipif(
    not _hopper(), reason="needs an H100"))])
@pytest.mark.parametrize("cls", [GatedDeltaNet, KDA, GatedDeltaNet2, DeltaNet],
                         ids=["gdn", "kda", "gdn2", "deltanet"])
@pytest.mark.parametrize("k_heads", [2, 1], ids=["mha", "grouped"])  # grouped: 2 value heads per q/k head
def test_qk_conv_barrier_keeps_values(cls, impl, k_heads):
    d_model, heads, dim = 256, 2, 128
    cfg = LinearAttentionConfig(num_k_heads=k_heads, num_v_heads=heads, head_k_dim=dim, head_v_dim=dim,
                                implementation=impl, gate_low_rank_dim=32 if cls in (KDA, GatedDeltaNet2) else None,
                                qk_conv_barrier=False)
    a = cls(d_model, cfg, key=jax.random.key(0), std=0.02, norm_eps=1e-6)
    b = cls(d_model, dataclasses.replace(cfg, qk_conv_barrier=True), key=jax.random.key(0), std=0.02,
            norm_eps=1e-6)
    B, T = 2, 160
    x = jax.random.normal(jax.random.key(1), (B, T, d_model)).astype(BF16)
    dy = jax.random.normal(jax.random.key(2), (B, T, d_model))
    seg = _segments(B, T)

    def loss(m, x):
        y = m(x, segment_ids=seg)
        return jnp.sum(y.astype(F32) * dy), y

    grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
    (_, ya), (ga, gxa) = grad(a, x)
    (_, yb), (gb, gxb) = grad(b, x)
    _close(yb, ya, 1e-2, "y")
    _close(gxb, gxa, 2e-2, "dx")
    # bf16 rounding only: with grouped heads the shared q/k gradients sum two value heads, and the barrier moves
    # where they are rounded (1.4-2.3% on dq_proj/dk_proj; in float32 compute every mixer matches to ~2e-7).
    proj_tol = 3e-2 if cls is DeltaNet else 2e-2
    for name in ("q_proj", "k_proj", "v_proj"):
        _close(getattr(gb, name).weight, getattr(ga, name).weight, proj_tol, f"d{name}")
    for name in ("q_conv", "k_conv"):
        _close(getattr(gb, name).weight, getattr(ga, name).weight, 2e-2, f"d{name}")


def _profiler_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "profile_layout_copies.py"
    spec = importlib.util.spec_from_file_location("profile_layout_copies", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


HLO = """HloModule m

%fused_transpose (p0: bf16[4,8]) -> bf16[8,4] {
  %p0 = bf16[4,8]{1,0} parameter(0)
  ROOT %t = bf16[8,4]{1,0} transpose(%p0), dimensions={1,0}
}

%fused_conv_norm (p0: bf16[4,8]) -> f32[4] {
  %p0 = bf16[4,8]{1,0} parameter(0)
  %c = bf16[] constant(0)
  %pad = bf16[4,11]{1,0} pad(%p0, %c), padding=0_0x3_0
  %s = bf16[4,8]{1,0} slice(%pad), slice={[0:4], [0:8]}
  %f = f32[4,8]{1,0} convert(%s)
  %m = f32[4,8]{1,0} multiply(%f, %f)
  %z = f32[] constant(0)
  ROOT %r = f32[4]{0} reduce(%m, %z), dimensions={1}, to_apply=%add
}

ENTRY %main (x: bf16[4,8]) -> (bf16[8,4], f32[4]) {
  %x = bf16[4,8]{1,0} parameter(0)
  %a = bf16[8,4]{1,0} fusion(%x), kind=kLoop, calls=%fused_transpose
  %b = f32[4]{0} fusion(%x), kind=kInput, calls=%fused_conv_norm
  ROOT %out = (bf16[8,4]{1,0}, f32[4]{0}) tuple(%a, %b)
}
"""


def test_layout_classes_of_fusions():
    prof = _profiler_module()
    comps = prof.fused_opcodes(HLO)
    lines = {"a": "bf16[8,4]{1,0} fusion(%x), kind=kLoop, calls=%fused_transpose",
             "b": "f32[4]{0} fusion(%x), kind=kInput, calls=%fused_conv_norm"}
    assert prof.layout_class({"opcode": "fusion"}, lines["a"], comps) == "transpose"
    assert prof.layout_class({"opcode": "fusion"}, lines["b"], comps) == "compute+pad"
    assert prof.layout_class({"opcode": "copy"}, "", comps) == "copy"
    assert prof.layout_class({"opcode": "dot"}, "", comps) is None
