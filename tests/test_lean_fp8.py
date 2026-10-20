# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``lean_vjp`` with FP8 (``fp8 = "delayed"`` + ``fp8_fused = true``): same forward, same gradients.

The fused FP8 SwiGLU (``jla.layers.fp8_swiglu``) already keeps no float32 intermediates, so with
``fp8_fused`` the lean VJPs take the rest of the block (norms, convolutions, gates, the mixers' cores)
and the model computes what the non-lean FP8 model computes. Without ``fp8_fused`` the combination stays a
config error (the unfused FP8 SwiGLU is not lean).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.layers.fp8 import Fp8State, Fp8SwiGLU, fp8_modules
from jla.layers.lean import LeanRMSNorm
from jla.models import build_model, get_preset
from jla.models.config import LinearAttentionConfig, hybrid_pattern
from jla.training.fp8 import apply_fp8_state, fp8_state_mask


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
    "tiny-rwkv7": lambda: get_preset("tiny-rwkv7"),  # scanned: v_first hoisted (lean norm, FP8 value projection)
}


def _fp8(cfg):
    return cfg.replace(compute_dtype="float32", fp8="delayed", fp8_fused=True, fp8_amax_history=16)


def test_lean_fp8_needs_fp8_fused():
    with pytest.raises(ValueError, match="fp8_fused"):
        get_preset("tiny").replace(lean_vjp=True, fp8="delayed")
    with pytest.raises(ValueError, match="fp8_fused"):
        get_preset("tiny").replace(lean_vjp=True, fp8="current")
    cfg = get_preset("tiny").replace(lean_vjp=True, fp8="delayed", fp8_fused=True)
    assert cfg.lean_vjp and cfg.fp8_fused


def test_lean_fp8_module_types():
    model = build_model(_fp8(get_preset("tiny")).replace(lean_vjp=True), key=jax.random.key(0))
    mlps = [m for m in fp8_modules(model) if isinstance(m, Fp8SwiGLU)]
    assert mlps and all(m.fused for m in mlps)  # the FP8 MLPs stay FP8 (fused), not LeanSwiGLU
    norms = [n for n in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, LeanRMSNorm))
             if isinstance(n, LeanRMSNorm)]
    assert norms  # the block norms are lean


def _loss_and_grads(model, ids, seg, r):
    def loss(m):
        out = m(ids, segment_ids=seg)
        return jnp.sum(out.hidden.astype(jnp.float32) * r)

    return eqx.filter_value_and_grad(loss)(model)


def _with_states_of(model, ref, ids, seg, r):
    """``model`` with the FP8 states ``ref`` reaches after one step (as the train step writes them back).

    A fresh delayed-scaling state quantizes with scale 1, so this tiny model's small gradients land on e5m2's
    subnormal grid, where float32 rounding differences flip many of them; after one step the scales follow
    each tensor's amax, as in training.
    """
    _, g = eqx.filter_jit(_loss_and_grads)(ref, ids, seg, r)
    ref_params, ref_static = eqx.partition(ref, eqx.is_inexact_array)
    mask = fp8_state_mask(ref_params)
    new_ref = eqx.combine(apply_fp8_state(ref_params, ref_params, g, mask, jnp.asarray(True)), ref_static)
    is_state = lambda n: isinstance(n, Fp8State)  # noqa: E731
    states = [n for n in jax.tree.leaves(new_ref, is_leaf=is_state) if is_state(n)]
    assert states
    it = iter(states)  # the lean model has the same modules in the same order, only some classes differ
    out = jax.tree.map(lambda n: next(it) if is_state(n) else n, model, is_leaf=is_state)
    assert next(it, None) is None
    return out


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("remat,core,scan", [("none", False, True), ("none", True, True), ("dots", False, True),
                                             ("none", False, False)])  # the last: the transformer TOMLs' setting
def test_lean_fp8_matches_fp8(name, remat, core, scan):
    cfg = _fp8(_MODELS[name]()).replace(remat=remat, scan_layers=scan)
    key = jax.random.key(0)
    ref = build_model(cfg, key=key)
    mod = build_model(cfg.replace(lean_vjp=True, lean_remat_core=core), key=key)
    if name == "tiny-rwkv7":  # zero o_proj / value: identity blocks at init; move both models' weights alike
        ref, mod = _moved_weights(ref), _moved_weights(mod)
    batch, length = 2, 40
    ids = jax.random.randint(jax.random.key(1), (batch, length), 0, cfg.vocab_size)
    seg = jnp.asarray(np.repeat(np.array([[0, 1], [3, 4]]), [13, 27], axis=1), jnp.int32)
    r = jax.random.normal(jax.random.key(2), (batch, length, cfg.d_model), jnp.float32)
    with jax.default_matmul_precision("highest"):
        ref, mod = _with_states_of(ref, ref, ids, seg, r), _with_states_of(mod, ref, ids, seg, r)
        l0, g0 = eqx.filter_jit(_loss_and_grads)(ref, ids, seg, r)
        l1, g1 = eqx.filter_jit(_loss_and_grads)(mod, ids, seg, r)
    np.testing.assert_allclose(np.asarray(l1), np.asarray(l0), rtol=1e-5, atol=1e-5)
    flat0 = jax.tree_util.tree_leaves_with_path(eqx.filter(g0, eqx.is_array))
    flat1 = jax.tree_util.tree_leaves_with_path(eqx.filter(g1, eqx.is_array))
    assert [jax.tree_util.keystr(p) for p, _ in flat0] == [jax.tree_util.keystr(p) for p, _ in flat1]
    # With the warmed scales (_with_states_of) the lean model's gradients match to 1e-4 of each tensor's
    # largest value on an AVX-512 CPU host, as without FP8 (tests/test_lean_vjp.py). Elsewhere the two backward
    # passes can round float32 differently: on a GPU (about 4e-7 without FP8, differently from one process to
    # the next) and on other CPUs (1.3-1.5e-4 of a tensor's largest value on GitHub's runners, tiny-hybrid and
    # tiny-gdn2, 20 Oct 2026). An element that this moves across an e4m3/e5m2 rounding step changes by up to a
    # quarter: measured up to 6.5e-2 of a tensor's largest value on tiny-hybrid on a GPU, scanned or unrolled,
    # in some processes and nothing above 1.5e-6 in others (repeated runs of one program agree to 1e-7). The
    # forward (the loss, above) stays exact, and on every backend the gradients are held to 0.25, which a wrong
    # VJP (an error of order 1) still fails.
    tol = 0.25
    for (path, a), (_, b) in zip(flat0, flat1):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        scale = max(1e-6, float(np.max(np.abs(a))))
        err = float(np.max(np.abs(a - b))) / scale
        assert err < tol, f"{jax.tree_util.keystr(path)}: relative error {err:.2e}"


def _moved_weights(model, scale=0.05, seed=11):
    """Every float parameter, not the FP8 states, moved off its initial value (the same draws for the lean
    and reference models, whose parameters share paths and shapes)."""
    is_state = lambda n: isinstance(n, Fp8State)  # noqa: E731
    params, static = eqx.partition(model, eqx.is_inexact_array, is_leaf=is_state)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    leaves = [p + scale * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static, is_leaf=is_state)
