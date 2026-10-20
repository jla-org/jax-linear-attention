# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN attention on a multi-device mesh: under FSDP, q, k and v can reach the attention with different
shardings (GSPMD propagates them from the projections' sharded weights), and JAX's cuDNN partitioner rejects
that ("Query, key and value should have same sharding"). Inside the trainer's batch partition the layer pins
all three, and the document mask, to the batch sharding for every cuDNN call (not only past cuDNN's size
limit), so the call compiles and stays batch-partitioned. The packed case passes replicated segment ids: a
mask left replicated failed to build cuDNN's graph ("dimensions mismatch"). Needs two GPUs: on one device
there is nothing to pin."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.default_backend() != "gpu" or jax.device_count() < 2, reason="needs two GPUs")

F32 = jnp.float32


@pytest.mark.parametrize("packed", [False, True])
def test_cudnn_pins_differently_sharded_qkv_to_the_batch(packed):
    from jla.layers.attention import attention, batch_partition

    mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",), axis_types=(jax.sharding.AxisType.Auto,))
    b, t, h, hkv, d = 4, 512, 8, 2, 128
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (b, t, h, d), F32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (b, t, hkv, d), F32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (b, t, hkv, d), F32).astype(jnp.bfloat16)
    cot = jax.random.normal(kc, (b, t, h, d), F32)
    seg = jnp.broadcast_to((jnp.arange(t) >= t // 3).astype(jnp.int32), (b, t)) if packed else None

    def loss(q, k, v, implementation):
        return (attention(q, k, v, segment_ids=seg, implementation=implementation).astype(F32) * cot).sum()

    def sharded(q, k, v):
        with batch_partition(mesh, "fsdp"):
            out = attention(q, k, v, segment_ids=seg, implementation="cudnn")
            grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v, "cudnn")
        return out, grads

    # Three different shardings: q over the batch, k over its heads, v over its head dim.
    specs = (P("fsdp"), P(None, None, "fsdp"), P(None, None, None, "fsdp"))
    args = [jax.device_put(x, NamedSharding(mesh, s)) for x, s in zip((q, k, v), specs)]
    fn = jax.jit(sharded)
    hlo = fn.lower(*args).compile().as_text()
    assert "cudnn" in hlo.lower()
    out, grads = fn(*args)

    want = attention(q, k, v, segment_ids=seg, implementation="xla")
    want_grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v, "xla")
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(want, np.float32), rtol=2e-2, atol=2e-2)
    for got, ref in zip(grads, want_grads):
        got, ref = np.asarray(got, np.float32), np.asarray(ref, np.float32)
        scale = max(float(np.abs(ref).max()), 1.0)
        np.testing.assert_allclose(got / scale, ref / scale, atol=2e-2)
