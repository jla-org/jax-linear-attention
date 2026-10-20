# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long single calls fail closed: kernels whose CUDA grid would pass 65535 blocks on y/z are rejected by
their ``supports`` predicate (``impl="auto"`` falls back) and raise a clear ``ValueError`` when requested by
name, instead of a launch failure (``CUDA_ERROR_INVALID_VALUE``). Shapes only, nothing is allocated, except
``test_mamba3_segment_runs_are_chunk_local``, which runs the gate preparation on 2**24 + 128 tokens."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jla.ops.mosaic_gpu._common import CHUNK, MAX_GRID_YZ, is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="needs an sm_90 (H100) GPU")

sds = jax.ShapeDtypeStruct
BF16, F32 = jnp.bfloat16, jnp.float32


def _x(length, heads, dim, dtype=BF16):
    return sds((1, length, heads, dim), dtype)


def _gate(length, heads):
    return sds((1, length, heads), F32)


def test_scalar_gate_kinds_reject_too_many_chunks():
    from jla.ops.mosaic_gpu import gated_delta_rule as gdn
    from jla.ops.mosaic_gpu import scalar_gate

    ok, bad = 32768 * CHUNK, (MAX_GRID_YZ + 1) * CHUNK
    for length, want in ((ok, True), (bad, False)):
        q, v, g = _x(length, 1, 64), _x(length, 1, 64), _gate(length, 1)
        assert scalar_gate.supports_simple_gla(q, q, v, g) is want
        assert gdn.supports_gated_delta_rule(q, q, v, g, g) is want
        assert gdn.supports_delta_rule(q, q, v, g) is want
    q, v, g = _x(bad, 1, 64), _x(bad, 1, 64), _gate(bad, 1)
    with pytest.raises(ValueError, match="CUDA grid"):
        scalar_gate.simple_gla(q, q, v, g)
    with pytest.raises(ValueError, match="CUDA grid"):
        gdn.gated_delta_rule(q, q, v, g, g)


@pytest.mark.parametrize("dim", [64, 128])
def test_vector_gate_kinds_reject_too_many_chunks(dim):
    from jla.ops.mosaic_gpu import gdn2, kda

    parts = dim // 64  # the backward WY kernel folds K / 64 channel slices into the chunk grid dimension
    heads = 128 // dim  # H * K must be a multiple of 128
    ok, bad = (MAX_GRID_YZ // parts) * CHUNK, (MAX_GRID_YZ // parts + 1) * CHUNK
    for length, want in ((ok, True), (bad, False)):
        q, g = _x(length, heads, dim), _x(length, heads, dim, F32)
        assert kda.supports_kda(q, q, q, g, _gate(length, heads)) is want
        assert gdn2.supports_gdn2(q, q, q, g, q, q) is want
    q, g = _x(bad, heads, dim), _x(bad, heads, dim, F32)
    with pytest.raises(ValueError, match="CUDA grid"):
        kda.kda(q, q, q, g, _gate(bad, heads))


def test_mamba3_fused_rejects_too_many_chunks_and_unfused_kernel_has_no_limit():
    from jla.ops.mosaic_gpu import mamba3, mamba3_fused

    def args(length):
        c = _x(length, 1, 64)
        v = _x(length, 1, 64)
        h = _gate(length, 1)
        bias = sds((1, 64), F32)
        return (c, c, v, h, h, h, bias, bias, _x(length, 1, 16, F32))

    assert mamba3_fused.supports_fused(*args(MAX_GRID_YZ * CHUNK))
    bad = MAX_GRID_YZ * CHUNK + 1
    assert not mamba3_fused.supports_fused(*args(bad))
    with pytest.raises(ValueError, match="CUDA grid"):
        mamba3_fused.mamba3_siso_fused(*args(bad))
    # The recurrence kernels' grid is (heads, batch): the layer's fallback for long calls stays on GPU.
    q, h = _x(bad, 1, 64), _gate(bad, 1)
    assert mamba3.supports(q, q, q, h, h, h)


def test_attention_rejects_too_many_blocks():
    from jla.ops.mosaic_gpu import attention

    ok = 32767 * 128  # T is padded to 128 and the backward kernels run one block per 64 tokens
    assert attention.supported(_x(ok, 2, 128), _x(ok, 2, 128), _x(ok, 2, 128)) is None
    bad = ok + 1
    reason = attention.supported(_x(bad, 2, 128), _x(bad, 2, 128), _x(bad, 2, 128))
    assert reason is not None and "CUDA grid" in reason
    with pytest.raises(ValueError, match="CUDA grid"):
        attention.flash_attention(_x(bad, 2, 128), _x(bad, 2, 128), _x(bad, 2, 128))


def _cudnn_loss(q, k):
    from jla.layers.attention import attention

    return attention(q, k, k, implementation="cudnn").astype(F32).sum()


def test_default_attention_avoids_cudnn_for_large_queries():
    """cuDNN's attention backward is silently wrong for large single calls: "auto" takes the Mosaic GPU kernel
    there, and an explicit "cudnn" keeps its (correct) forward but raises when differentiated
    (benchmarks/long_context.md)."""
    from jla.layers.attention import CUDNN_MAX_QUERY_ELEMENTS, attention, resolve_attention_implementation

    assert resolve_attention_implementation("auto", _x(4096, 16, 128), _x(4096, 4, 128)) == "cudnn"
    length = CUDNN_MAX_QUERY_ELEMENTS // (16 * 128)  # exactly at the limit: still cuDNN (measured right)
    assert resolve_attention_implementation("auto", _x(length, 16, 128), _x(length, 4, 128)) == "cudnn"
    length += 2  # one more (even) step past 2**31 query elements
    q, k = _x(length, 16, 128), _x(length, 4, 128)
    assert resolve_attention_implementation("auto", q, k) == "mosaic_gpu"
    assert jax.eval_shape(lambda q, k: attention(q, k, k, implementation="cudnn"), q, k).shape == q.shape
    with pytest.raises(ValueError, match="cuDNN attention computes wrong gradients"):
        jax.eval_shape(jax.grad(_cudnn_loss), q, k)


def _abstract_mesh(size: int, name: str = "data"):  # GSPMD (auto) axes, as the trainer's Mesh
    return jax.sharding.AbstractMesh((size,), (name,), axis_types=(jax.sharding.AxisType.Auto,))


def test_cudnn_limit_is_per_device_under_a_batch_partition():
    """GSPMD traces see global shapes: inside the trainer's batch partition the limit is checked on one
    device's batch shard, so data-parallel runs keep batch-partitioned cuDNN (no all-gather to our kernel)."""
    from jla.layers.attention import (CUDNN_MAX_QUERY_ELEMENTS, batch_partition, query_elements_per_device,
                                      resolve_attention_implementation)

    length = CUDNN_MAX_QUERY_ELEMENTS // (16 * 128)
    q, k = sds((2, length, 16, 128), BF16), sds((2, length, 4, 128), BF16)  # 2**32 global, 2**31 per device
    assert resolve_attention_implementation("auto", q, k) == "mosaic_gpu"  # no partition: the global size
    with batch_partition(_abstract_mesh(2), "data"):
        per_device, pin = query_elements_per_device(q)
        assert per_device == CUDNN_MAX_QUERY_ELEMENTS and pin is not None
        assert resolve_attention_implementation("auto", q, k) == "cudnn"
        assert jax.eval_shape(jax.grad(_cudnn_loss), q, k).shape == q.shape  # no false raise
    q3, k3 = sds((3, length, 16, 128), BF16), sds((3, length, 4, 128), BF16)  # B not divisible: global size
    with batch_partition(_abstract_mesh(2), "data"):
        assert query_elements_per_device(q3) == (q3.size, None)
        assert resolve_attention_implementation("auto", q3, k3) == "mosaic_gpu"
    with batch_partition(_abstract_mesh(1), "data"):
        assert resolve_attention_implementation("auto", q, k) == "mosaic_gpu"
    if jax.device_count() >= 2:  # inside shard_map over the batch axes q is already one device's shard
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
        seen = []

        def local(x):
            with batch_partition(mesh, "data"):
                seen.append(query_elements_per_device(x))
            return x

        spec = jax.sharding.PartitionSpec("data")
        jax.eval_shape(jax.shard_map(local, mesh=mesh, in_specs=spec, out_specs=spec), sds((4, 8, 16, 128), BF16))
        assert seen == [(2 * 8 * 16 * 128, None)]
    explicit = jax.sharding.AbstractMesh((2,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    with batch_partition(explicit, "data"):  # explicit axes: q's (unsharded) type decides, not the context
        assert query_elements_per_device(q) == (q.size, None)


@pytest.mark.skipif(jax.device_count() < 2, reason="needs two GPUs")
def test_data_parallel_cudnn_attention_compiles_without_all_gathers():
    """Compile only: global q (160, 8192, 16, 128) = 2.68e9 elements sharded over two GPUs (1.34e9 each) keeps
    batch-partitioned cuDNN inside the batch partition (it all-gathered q, k, v into our kernel before)."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from jla.layers.attention import attention, batch_partition

    mesh = Mesh(np.array(jax.devices()[:2]), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    sharding = NamedSharding(mesh, P("data"))
    q = sds((160, 8192, 16, 128), BF16, sharding=sharding)
    k = sds((160, 8192, 4, 128), BF16, sharding=sharding)

    def loss(q, k):
        with batch_partition(mesh, "data"):
            return attention(q, k, k).astype(F32).sum()

    hlo = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(q, k).compile().as_text()
    assert "all-gather" not in hlo
    assert "cudnn" in hlo.lower()


def test_mamba3_segment_runs_are_chunk_local():
    """More than 2**24 documents in one call: the run ids the kernels compare stay exact in float32."""
    from jla.ops.mosaic_gpu import mamba3, mamba3_fused

    length = 2**24 + 2 * CHUNK
    num_chunks = length // CHUNK
    segment_ids = jnp.arange(length, dtype=jnp.int32)[None]  # every token its own document
    zeros = jnp.zeros((1, length, 1), F32)
    rows = jax.jit(lambda s: mamba3.prepare_gates(zeros, zeros, zeros, num_chunks=num_chunks, segment_ids=s))(
        segment_ids)
    last = np.asarray(rows[0, 0, -1, mamba3.ROW_RUN])
    np.testing.assert_array_equal(last, np.arange(CHUNK, dtype=np.float32))
    angle = jax.jit(lambda s: mamba3_fused.angle_rows(zeros, num_chunks=num_chunks, segment_ids=s))(segment_ids)
    np.testing.assert_array_equal(np.asarray(angle[0, 0, -1, mamba3_fused.RA_RUN]), np.arange(CHUNK, dtype=np.float32))
