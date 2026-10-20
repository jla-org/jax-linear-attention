# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
# GDN-2: implemented from arXiv 2605.22791 (equations only).
"""The Gated DeltaNet-2 (and KDA) training fast paths against the paths they replace.

* The erase and write gate projections read the shared FP8 cast of the mixer input with q, k, v and the
  output gate (``GatedDeltaNet2._gate_input_projections``): two e4m3 casts of ``x`` fewer per forward, and the
  output, every gradient and every next FP8 state bitwise those of separate casts on the CPU (the FP8 train
  step against the per-call path: ``tests/test_fp8_weight_once.py``, ``tiny-gdn2``).
* The Mosaic GPU vector-gate op (GDN-2 and KDA) under ``jax.checkpoint`` runs its no-residual primal in the
  forward pass (``vector_gate_ops._op_remat``, ``optimize_remat``): the op's gradients bitwise those without
  remat (``jax.checkpoint`` around the whole loss), the forward value the no-gradient call's, and one ``wy``
  kernel per gradient instead of two (the kernels on an H100, dense and packed; two H100s with batch and head
  splits). That value differs from the residual path's by rounding (relative L2 about 4e-3 at B * H = 32,
  T = 4096, slightly closer to a float32 reference), so under per-layer remat, where layers outside the
  checkpoint read it, a model's gradients differ from those without remat at rounding level (dq up to about
  1.5e-3 of its max; ``test_model_remat_runs_the_primal_forward``). The Mosaic GPU interpreter keeps the plain
  custom VJP: its kernels are host callbacks, which neither JAX's remat optimization nor ``jax.checkpoint``
  takes, so it has no remat case.
"""

from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _vector_gate import assert_close, inputs

from jla.layers import fp8 as F
from jla.layers.linear_attention import GatedDeltaNet2, make_mixer
from jla.models.presets import get_preset
from jla.ops import chunked
from jla.ops.mosaic_gpu import vector_gate_ops as vgo
from jla.training.fp8 import attach_fp8_weights, prepare_fp8_weights

F32, BF16 = jnp.float32, jnp.bfloat16
ON_GPU = jax.default_backend() == "gpu"
ON_HOPPER = vgo.is_hopper()
gpu = pytest.mark.skipif(not ON_HOPPER, reason="needs an H100 (compute capability 9.0)")
B, T = 2, 32


def _with_history(state: F.Fp8State) -> F.Fp8State:
    """Nonzero amax histories (the same input history everywhere, as every projection of one input records)."""
    hist = lambda v: jnp.zeros_like(state.input_amax_history).at[..., :2].set(jnp.asarray([0.5 * v, v]))  # noqa: E731
    return eqx.tree_at(lambda s: (s.input_amax_history, s.kernel_amax_history, s.grad_amax_history), state,
                       (hist(3.0), hist(0.2), hist(1e-3)))


def _fp8_mixer():
    cfg = get_preset("tiny-gdn2")  # Hk 2, Hv 4 (grouped value heads), K = V = 16, d_model 64
    mixer = make_mixer("gdn2", cfg.d_model, cfg.linear_attention, key=jax.random.key(0), std=0.02, norm_eps=1e-6)
    mixer = F.apply_fp8(mixer, "delayed", amax_history=8, min_features=16)
    is_state = lambda n: isinstance(n, F.Fp8State)  # noqa: E731
    mixer = jax.tree.map(lambda n: _with_history(n) if is_state(n) else n, mixer, is_leaf=is_state)
    assert type(mixer.b_proj) is F.Fp8Linear and type(mixer.w_proj) is F.Fp8Linear
    return attach_fp8_weights(mixer, prepare_fp8_weights(mixer)), cfg.d_model


def _e4m3_casts(text: str, d_model: int) -> int:
    """e4m3 casts of a ``[B, T, d_model]`` tensor in lowered StableHLO (the mixer input and the out projection's)."""
    target = re.compile(rf"-> tensor<{B}x{T}x{d_model}xf8E4M3FN>")
    return sum(1 for line in text.splitlines() if "stablehlo.convert" in line and target.search(line))


def test_gdn2_gate_projections_share_the_fp8_input_cast(monkeypatch) -> None:
    mixer, d_model = _fp8_mixer()
    x = jax.random.normal(jax.random.key(1), (B, T, d_model), F32).astype(BF16)
    seg = jnp.asarray([[0] * 20 + [1] * 12, [0] * T], jnp.int32)
    cot = jax.random.normal(jax.random.key(2), (B, T, d_model), F32)

    def run():
        def loss(m, x):
            return jnp.sum(m(x, segment_ids=seg).astype(F32) * cot)

        text = jax.jit(loss).lower(mixer, x).as_text()
        return _e4m3_casts(text, d_model), eqx.filter_jit(eqx.filter_value_and_grad(loss))(mixer, x)

    shared_casts, (shared_loss, shared_grads) = run()
    monkeypatch.setattr(GatedDeltaNet2, "_gate_input_projections", lambda self: ())  # one cast per gate projection
    separate_casts, (separate_loss, separate_grads) = run()
    # Shared: one cast of x for q, k, v, g, b, w; separate: one for q, k, v, g and one each for b and w (both also
    # cast x for the low-rank decay projection's first factor, FP8 at these widths, and the out projection's input).
    assert (shared_casts, separate_casts) == (3, 5)
    a_leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(shared_grads, eqx.is_array))
    b_leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(separate_grads, eqx.is_array))
    assert [jax.tree_util.keystr(p) for p, _ in a_leaves] == [jax.tree_util.keystr(p) for p, _ in b_leaves]
    assert any("b_proj" in jax.tree_util.keystr(p) and "state" in jax.tree_util.keystr(p) for p, _ in a_leaves)
    for (path, a), (_, b) in zip(a_leaves, b_leaves, strict=True):
        a, b = np.asarray(a.astype(F32) if a.dtype != F32 else a), np.asarray(b.astype(F32) if b.dtype != F32 else b)
        # The same GEMMs on the same e4m3 inputs. XLA may fuse the epilogues differently, on a GPU and on some
        # CPUs (bitwise equal on an AVX-512 host), so every backend allows float32 rounding.
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=jax.tree_util.keystr(path))
    np.testing.assert_allclose(float(shared_loss), float(separate_loss), rtol=1e-6)


def test_gdn2_gate_logits_given_or_computed_match() -> None:
    """``gates(x)`` (decode-style callers) and ``gates(x, x W_b, x W_w)`` (the forward's shared projections)."""
    cfg = get_preset("tiny-gdn2")
    mixer = make_mixer("gdn2", cfg.d_model, cfg.linear_attention, key=jax.random.key(3), std=0.02, norm_eps=1e-6)
    x = jax.random.normal(jax.random.key(4), (B, T, cfg.d_model), F32).astype(BF16)
    own = mixer.gates(x)
    given = mixer.gates(x, mixer.b_proj(x), mixer.w_proj(x))
    assert sorted(own) == ["b", "g", "w"]
    for name in own:
        np.testing.assert_array_equal(np.asarray(own[name]), np.asarray(given[name]), err_msg=name)


def _wy_kernels(fn, args) -> int:
    text = jax.jit(fn).lower(*args).compile().as_text()
    return sum(1 for line in text.splitlines() if "custom-call" in line and "vg_wy" in line)


@gpu
@pytest.mark.parametrize("kind", ["gdn2", "kda"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("shape", [(2, 256, 8, 128, 128), (4, 192, 4, 64, 128)], ids=["k128", "k64v128"])
def test_remat_forward_runs_the_primal_on_the_kernels(kind, shape, packed) -> None:
    # B H = 16 > _WY_HEADS: the primal takes the fused state pass, _forward the wy path.
    x = inputs(12, kind, *shape)
    args = (x["q"], x["k"], x["v"], x["g"], x["gates"], x["h0"])
    batch, seq_len = shape[:2]
    seg = (jnp.arange(seq_len)[None, :] >= jnp.asarray([[seq_len // 3], [seq_len // 2]] * (batch // 2))).astype(
        jnp.int32) if packed else None

    def forward(q, k, v, g, gates, h0):
        return vgo.chunked_vector_gate(q, k, v, g, gates, scale=q.shape[-1] ** -0.5, initial_state=h0,
                                       segment_ids=seg)

    def loss(*a):
        o, final = forward(*a)
        return jnp.sum(o.astype(F32) * x["do"]) + jnp.sum(final), (o, final)

    argnums = (0, 1, 2, 3, 4, 5)
    (_, wy_out), grads = jax.jit(jax.value_and_grad(loss, argnums, has_aux=True))(*args)  # _forward: the wy path
    (_, remat_out), remat_grads = jax.jit(jax.value_and_grad(jax.checkpoint(loss), argnums, has_aux=True))(*args)
    primal_out = jax.jit(forward)(*args)  # the no-gradient call: the fused state pass, no wy
    for i, (a, b) in enumerate(zip(jax.tree.leaves(grads), jax.tree.leaves(remat_grads), strict=True)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"grad{i}")
    for name, a, b in zip(("o", "final"), remat_out, primal_out, strict=True):  # the forward pass ran the primal
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    # Both forward paths against the f32 chunked op, at test_mosaic_gpu_gdn2.py's tolerance, and against each other.
    q32, k32, v32 = (x[n].astype(F32) for n in ("q", "k", "v"))
    want = getattr(chunked, kind)(q32, k32, v32, x["g"], *x["gates"], initial_state=x["h0"], output_final_state=True,
                                  segment_ids=seg)
    for name, fused, wy, ref in zip(("o", "final"), primal_out, wy_out, want, strict=True):
        assert_close(fused, ref, 1.5e-2, f"{name} (fused)")
        assert_close(wy, ref, 1.5e-2, f"{name} (wy)")
        assert_close(fused, wy, 1.5e-2, f"{name} (fused vs wy)")
    loss_only = lambda *a: loss(*a)[0]  # noqa: E731
    plain = _wy_kernels(jax.grad(loss_only, argnums), args)
    remat = _wy_kernels(jax.grad(jax.checkpoint(loss_only), argnums), args)
    assert plain >= 1 and remat == plain  # without optimize_remat the remat step also ran wy in its forward pass


@pytest.mark.skipif(not ON_HOPPER or len(jax.devices()) < 2, reason="needs two H100s")
@pytest.mark.parametrize("kind", ["gdn2", "kda"])
def test_remat_on_two_gpus_matches_one_gpu(kind) -> None:
    """The remat-optimized op inside the kernels' ``shard_map`` (batch and head splits), under ``jax.checkpoint``.

    Needs a process that sees two H100s, e.g. ``CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest
    tests/test_gdn2_fast_paths.py -k two_gpus``."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.ops import linear_attention

    x = inputs(13, kind, 4, 512, 8, 128, 128)  # B H = 16 per shard either way: the fused primal
    seg = jnp.repeat((jnp.arange(512) >= 200)[None].astype(jnp.int32), 4, axis=0)
    gate_names = ("b", "w") if kind == "gdn2" else ("beta",)
    arrays = (x["q"], x["k"], x["v"], x["g"], tuple(x["gates"]), x["do"])

    def fwd_bwd(q, k, v, g, gates, do, seg):
        def loss(q, k, v, g, gates):
            o, _ = linear_attention(kind, q, k, v, g=g, **dict(zip(gate_names, gates)), impl="mosaic_gpu",
                                    segment_ids=seg)
            return jnp.sum(o.astype(F32) * do), o
        return jax.value_and_grad(jax.checkpoint(loss), argnums=(0, 1, 2, 3, 4), has_aux=True)(q, k, v, g, gates)

    one = jax.jit(fwd_bwd)(*jax.device_put((*arrays, seg), jax.devices()[0]))
    for spec_axes in ({"dp": 2}, {"tp": 2}):
        mesh = build_mesh(resolve_mesh_plan(2, fsdp=1, **spec_axes), jax.devices()[:2], topology_aware=False)
        tok = NamedSharding(mesh, P("dp", None, "tp"))
        placed = [*jax.device_put(arrays, tok), jax.device_put(seg, NamedSharding(mesh, P("dp", None)))]
        with jax.set_mesh(mesh):
            fn = jax.jit(fwd_bwd)
            two = fn(*placed)
            text = fn.lower(*placed).compile().as_text()
        assert "all-gather" not in text, spec_axes
        # The loss sums two partial sums plus an all-reduce add in another order; the shards' XLA code around
        # the kernels fuses differently (tests/test_kernel_sharding.py), so the rest is bf16-close.
        np.testing.assert_allclose(float(two[0][0]), float(one[0][0]), rtol=1e-5)
        names = ("o", "q", "k", "v", "g", *gate_names)
        leaves_one, leaves_two = jax.tree.leaves((one[0][1], one[1])), jax.tree.leaves((two[0][1], two[1]))
        for name, a, b in zip(names, leaves_one, leaves_two, strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            np.testing.assert_allclose(b, a, rtol=2e-2, atol=2e-2 * float(np.abs(a).max()),
                                       err_msg=f"{spec_axes} {name}")


def _kernel_model(kind: str, **changes):
    """A two-layer model of the 1.3B preset's mixer on the Mosaic GPU kernels (4 heads of 128, d_model 256)."""
    import dataclasses

    cfg = get_preset(f"{kind}-1p3b")
    la = dataclasses.replace(cfg.linear_attention, num_k_heads=4, num_v_heads=4, implementation="mosaic_gpu")
    return cfg.replace(n_layers=2, vocab_size=512, d_model=256, d_ff=512, layer_types=(kind,) * 2,
                       linear_attention=la, **changes)


def _lm_batch(rows: int, seq: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 512, size=(rows, seq)).astype(np.int32)
    cut = rng.integers(seq // 4, 3 * seq // 4, size=(rows, 1))
    t = np.arange(seq)
    return {"input_ids": jnp.asarray(ids), "labels": jnp.asarray(np.roll(ids, -1, axis=1)),
            "segment_ids": jnp.asarray((t >= cut).astype(np.int32)),
            "positions": jnp.asarray(np.where(t >= cut, t - cut, t).astype(np.int32))}


def _model_grads(model, batch, forward=None, mesh=None, rules=None):
    from jla.distributed.sharding import grouped_batch_sharding, param_shardings
    from jla.models import logical_axes
    from jla.training.step import LossConfig, lm_loss

    params, static = eqx.partition(model, eqx.is_inexact_array)
    loss_fn = lm_loss(LossConfig(), forward=forward)
    if mesh is not None:
        params = jax.device_put(params, param_shardings(params, logical_axes(model), mesh, rules))
        sharding = grouped_batch_sharding(mesh, rules, grouped=False)
        batch = {k: jax.device_put(v, sharding) for k, v in batch.items()}

    def f(p):
        return loss_fn(eqx.combine(p, static), batch, jax.random.key(1))[0]

    value, grads = jax.jit(jax.value_and_grad(f))(params)
    return float(value), jax.tree.leaves(grads), float(jax.jit(f)(params))


def _max_rel_errors(got, want):
    return [float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
            / max(float(np.max(np.abs(np.asarray(b, np.float32)))), 1e-12) for a, b in zip(got, want, strict=True)]


@gpu
@pytest.mark.parametrize("kind", ["gdn2", "kda"])
@pytest.mark.parametrize("scan_layers", [True, False], ids=["scanned", "unrolled"])
def test_model_remat_runs_the_primal_forward(kind, scan_layers) -> None:
    """Full remat against none, layers scanned or unrolled: the loss of the remat step is the no-gradient forward's
    (the primal ran in its forward pass), and its gradients within rounding of those without remat (the layers after
    each checkpoint read the primal's output)."""
    from jla.models import build_model

    batch = _lm_batch(4, 256)  # B H = 16 > _WY_HEADS: the fused primal
    runs = {}
    for remat in ("none", "full"):
        model = build_model(_kernel_model(kind, remat=remat, scan_layers=scan_layers), key=jax.random.key(7))
        runs[remat] = _model_grads(model, batch)
    (none_value, none_grads, primal), (full_value, full_grads, primal_full) = runs["none"], runs["full"]
    np.testing.assert_allclose(primal_full, primal, rtol=1e-6)  # the no-gradient forward, with or without remat
    np.testing.assert_allclose(full_value, primal, rtol=1e-6)  # the remat step's forward ran the primal
    np.testing.assert_allclose(none_value, primal, rtol=1e-3)  # the residual forward: another rounding
    # bf16 compute: the recomputed forward rounds a little differently from the first one (as in
    # tests/test_remat_policies.py, whose 5e-2 bound on the relative L2 error this max-error bound mirrors).
    assert max(_max_rel_errors(full_grads, none_grads)) < 5e-2


@pytest.mark.skipif(not ON_HOPPER or len(jax.devices()) < 2, reason="needs two H100s")
@pytest.mark.parametrize("kind", ["gdn2", "kda"])
def test_remat_in_the_pipeline_on_two_gpus_matches_one_gpu(kind) -> None:
    """Two pipeline stages (pp 2, two microbatches) of the remat-optimized op under full remat, against one GPU."""
    from jla.distributed.mesh import build_mesh, resolve_mesh_plan
    from jla.distributed.pipeline import make_pipeline_forward, pipeline_rules
    from jla.distributed.sharding import DEFAULT_RULES
    from jla.models import build_model

    model = build_model(_kernel_model(kind, remat="full"), key=jax.random.key(7))
    batch = _lm_batch(8, 256, seed=1)  # 4 rows per microbatch: B H = 16 per stage call, the fused primal
    ref_value, ref_grads, _ = _model_grads(model, batch)
    mesh = build_mesh(resolve_mesh_plan(2, fsdp=1, pp=2), jax.devices()[:2], topology_aware=False)
    rules = pipeline_rules(DEFAULT_RULES)
    value, grads, _ = _model_grads(model, batch, make_pipeline_forward(mesh, num_microbatches=2), mesh, rules)
    np.testing.assert_allclose(value, ref_value, rtol=1e-4)
    assert max(_max_rel_errors(grads, ref_grads)) < 5e-2
