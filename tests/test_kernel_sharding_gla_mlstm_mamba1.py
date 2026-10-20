# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The GLA, mLSTM and Mamba-1 Mosaic GPU kernels per shard of the context mesh, against one GPU.

GLA goes through the dispatcher's split (``jla.distributed.kernel_sharding.shard_kernel_call``:
batch over ``dp``, heads over ``tp``), mLSTM through the same split of its simple-GLA kernel call only
(the XLA code around it is left to the SPMD partitioner); Mamba-1 through its own ``shard_layout_call`` rule
(``jla.ops.mosaic_gpu.mamba1.mamba1_sharded``: batch over ``dp``, channels over ``tp``, ``B, C``
replicated over the channel shards). Needs a process that sees two H100s, e.g.
``CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest tests/test_kernel_sharding_gla_mlstm_mamba1.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu" or len(jax.devices()) < 2,
                                reason="needs two GPUs")

F32 = jnp.float32


def _inputs(kind: str):
    """``(names, values, placement, loss)``: the differentiable inputs, their sharding names
    (``"tok"`` ``[B, T, H, ...]``, ``"rows"`` ``[B, ...]``, ``"state"`` ``[B, H, ...]``, ``"chan"``
    ``[D, ...]``) and a loss of the outputs."""
    from jla import ops
    from jla.ops.reference import MLSTMState

    ks = jax.random.split(jax.random.key(7), 12)
    seg = jnp.concatenate([jnp.zeros((4, 150), jnp.int32), jnp.ones((4, 106), jnp.int32)], axis=1)
    if kind == "gla":
        B, T, H, K, V = 4, 256, 4, 128, 128
        vals = dict(q=jax.random.normal(ks[0], (B, T, H, K)).astype(jnp.bfloat16),
                    k=jax.random.normal(ks[1], (B, T, H, K)).astype(jnp.bfloat16),
                    v=jax.random.normal(ks[2], (B, T, H, V)).astype(jnp.bfloat16),
                    g=jax.nn.log_sigmoid(jax.random.normal(ks[3], (B, T, H, K))) / 16,
                    h0=jax.random.normal(ks[4], (B, H, K, V)))
        place = dict(q="tok", k="tok", v="tok", g="tok", h0="state")

        def loss(q, k, v, g, h0):
            o, ht = ops.linear_attention("gla", q, k, v, g=g, initial_state=h0, output_final_state=True,
                                         segment_ids=seg, impl="mosaic_gpu")
            return jnp.sum(o.astype(F32) * jnp.cos(o.astype(F32))) + jnp.sum(ht * ht), o

    elif kind == "mlstm":
        B, T, H, K, V = 4, 256, 4, 128, 128
        vals = dict(q=jax.random.normal(ks[0], (B, T, H, K)).astype(jnp.bfloat16),
                    k=jax.random.normal(ks[1], (B, T, H, K)).astype(jnp.bfloat16),
                    v=jax.random.normal(ks[2], (B, T, H, V)).astype(jnp.bfloat16),
                    g=jax.nn.log_sigmoid(3.0 * jax.random.normal(ks[3], (B, T, H)) + 2.0),
                    i=3.0 * jax.random.normal(ks[4], (B, T, H)),
                    c0=0.1 * jax.random.normal(ks[5], (B, H, K, V)), n0=0.1 * jax.random.normal(ks[6], (B, H, K)),
                    m0=jax.random.normal(ks[7], (B, H)))
        place = dict(q="tok", k="tok", v="tok", g="tok", i="tok", c0="state", n0="state", m0="state")

        def loss(q, k, v, g, i, c0, n0, m0):
            o, st = ops.linear_attention("mlstm", q, k, v, g=g, i=i, initial_state=MLSTMState(c0, n0, m0),
                                         output_final_state=True, segment_ids=seg, impl="mosaic_gpu")
            return jnp.sum(o.astype(F32) * jnp.cos(o.astype(F32))) + jnp.sum(st.c * st.c) + jnp.sum(st.n), o

    else:
        B, T, D, N = 4, 256, 256, 8
        vals = dict(q=jax.random.normal(ks[0], (B, T, N)).astype(jnp.bfloat16),
                    k=jax.random.normal(ks[1], (B, T, N)).astype(jnp.bfloat16),
                    v=jax.random.normal(ks[2], (B, T, D)).astype(jnp.bfloat16),
                    dt=(0.5 * jax.random.normal(ks[3], (B, T, D))).astype(jnp.bfloat16),
                    a=-jnp.exp(jax.random.uniform(ks[4], (D, N), F32, -1.0, 1.5)),
                    d=jax.random.normal(ks[5], (D,)),
                    z=jax.random.normal(ks[6], (B, T, D)).astype(jnp.bfloat16),
                    dt_bias=jax.random.uniform(ks[7], (D,), F32, -3.0, -1.0),
                    h0=jax.random.normal(ks[8], (B, D, N)))
        place = dict(q="rows", k="rows", v="tok", dt="tok", a="chan", d="chan", z="tok", dt_bias="chan", h0="state")

        def loss(q, k, v, dt, a, d, z, dt_bias, h0):
            o, ht = ops.linear_attention("mamba1", q, k, v, dt=dt, a=a, d=d, z=z, dt_bias=dt_bias, dt_softplus=True,
                                         initial_state=h0, output_final_state=True, segment_ids=seg,
                                         impl="mosaic_gpu")
            return jnp.sum(o.astype(F32) * jnp.cos(o.astype(F32))) + jnp.sum(ht * ht), o

    return tuple(vals), vals, place, loss, seg


@pytest.mark.parametrize("kind", ["gla", "mlstm", "mamba1"])
def test_kernel_on_two_gpus_matches_one_gpu(kind: str) -> None:
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jla.distributed.mesh import build_mesh, resolve_mesh_plan

    if jax.devices()[0].compute_capability != "9.0":
        pytest.skip("Mosaic GPU kernels need sm_90a")
    names, vals, place, loss, seg = _inputs(kind)
    grad = jax.value_and_grad(loss, argnums=tuple(range(len(names))), has_aux=True)
    first = jax.devices()[0]
    one = jax.jit(grad)(*(jax.device_put(vals[n], first) for n in names))
    specs = dict(tok=P("dp", None, "tp"), rows=P("dp"), state=P("dp", "tp"), chan=P("tp"))
    for axes in ({"dp": 2}, {"tp": 2}):
        mesh = build_mesh(resolve_mesh_plan(2, fsdp=1, **axes), jax.devices()[:2], topology_aware=False)
        placed = [jax.device_put(vals[n], NamedSharding(mesh, specs[place[n]])) for n in names]
        with jax.set_mesh(mesh):
            fn = jax.jit(grad)
            two = fn(*placed)
            text = fn.lower(*placed).compile().as_text()
        # The kernels run on each device's own rows / heads / channels: nothing is gathered.
        assert "all-gather" not in text, (kind, axes)
        np.testing.assert_allclose(float(two[0][0]), float(one[0][0]), rtol=1e-4)
        # Same kernels on smaller shards; the XLA code around them may fuse and sum in another order.
        for name, a, b in zip(("o", *names), (one[0][1], *one[1]), (two[0][1], *two[1]), strict=True):
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            err = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-30)
            assert err <= 2e-2, f"{kind} {axes}: {name} differs from one GPU by {err:.2e} of its max"
