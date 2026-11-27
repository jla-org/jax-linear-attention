# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Blackwell (tcgen05) path of ``_common.matmul`` against a float32 reference (runs only on sm_100/sm_103)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jla.ops.mosaic_gpu import _common
from jla.ops.mosaic_gpu._common import WGMMA, matmul, smem

M = 64
CASES = ("smem", "rhs_t", "regs_init", "regs_t", "loop")
STEPS = 3


def build(case: str, k: int, n: int, m: int = M, ws: bool = False, ctas: int = 1):
    """``(kernel, args, want)``: one warpgroup loads A (M,K) and B (K,N) [or B^T (N,K)] and runs ``case``."""
    M = m  # noqa: N806 (rows of the product)
    key = jax.random.key(hash((case, k, n, m)) % 2**31)
    ka, kb, kc = jax.random.split(key, 3)
    a = jax.random.normal(ka, (M, k), jnp.float32).astype(jnp.bfloat16)
    b_shape = (n, k) if case in ("rhs_t", "regs_t") else (k, n)
    b = jax.random.normal(kb, b_shape, jnp.float32).astype(jnp.bfloat16)
    c = jax.random.normal(kc, (M, n), jnp.float32)

    def work(a_ref, b_ref, c_ref, o_ref, a_smem, b_smem, c_smem, o_smem, bar):
        plgpu.copy_gmem_to_smem(a_ref, a_smem, bar)
        plgpu.copy_gmem_to_smem(b_ref, b_smem, bar)
        plgpu.copy_gmem_to_smem(c_ref, c_smem, bar)
        plgpu.barrier_wait(bar)
        rhs = b_smem.T if case in ("rhs_t", "regs_t") else b_smem
        if case in ("smem", "rhs_t"):
            out = matmul(a_smem, rhs, (M, n))
        elif case in ("regs_init", "regs_t"):
            a_regs = plgpu.load(a_smem, layout=WGMMA).astype(jnp.float32) * 2.0
            init = plgpu.load(c_smem, layout=WGMMA, optimized=False)
            out = matmul(a_regs.astype(jnp.bfloat16), rhs, (M, n), init=init)
        elif case == "loop":
            def step(i, acc):
                return matmul(a_smem, rhs, (M, n), init=acc)
            out = jax.lax.fori_loop(0, STEPS, step, plgpu.load(c_smem, layout=WGMMA, optimized=False))
        else:
            raise ValueError(case)
        o_smem[...] = out
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_ref)
        plgpu.wait_smem_to_gmem(0)

    scratch = dict(a_smem=smem((M, k), jnp.bfloat16), b_smem=smem(b_shape, jnp.bfloat16),
                   c_smem=plgpu.SMEM((M, n), jnp.float32), o_smem=plgpu.SMEM((M, n), jnp.float32),
                   bar=plgpu.Barrier(num_arrivals=3))
    if ws:  # warp-specialized: warpgroup 1 computes, warpgroup 0 idles (exercises the two-pass TMEM plan)
        def body(*refs, **scr):
            @pl.when(lax.axis_index("wg") == 1)
            def _():
                work(*refs, **scr)

        threads = dict(num_threads=2, thread_name="wg")
    else:
        body, threads = work, {}
    fn = _common.build_kernel(body, name=f"bw_{case}", out_type=jax.ShapeDtypeStruct((M, n), jnp.float32),
                              scratch=scratch, grid=(ctas,), grid_names=("x",), **threads)
    a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
    rhs32 = b32.T if case in ("rhs_t", "regs_t") else b32
    dot = lambda x, y: jnp.matmul(x, y, precision=lax.Precision.HIGHEST)  # no TF32 in the reference
    if case in ("smem", "rhs_t"):
        want = dot(a32, rhs32)
    elif case in ("regs_init", "regs_t"):
        want = c + dot(a32 * 2.0, rhs32)  # a * 2 is exact in bfloat16
    else:
        want = c + STEPS * dot(a32, rhs32)
    return fn, (a, b, c), want


def lower(case: str, k: int, n: int, m: int = M, ws: bool = False) -> str:
    """Lower one case for CUDA without running it (works on a CPU-only host)."""
    fn, args, _ = build(case, k, n, m, ws)
    return jax.jit(fn).trace(*args).lower(lowering_platforms=("cuda",)).as_text()


@pytest.mark.skipif(not _common.is_blackwell(), reason="needs a datacenter Blackwell GPU")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("k,n", [(64, 64), (128, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("ws", [False, True])
def test_tcgen05_matmul(case, k, n, ws):
    fn, args, want = build(case, k, n, M, ws)
    got = np.asarray(jax.jit(fn)(*args))
    want = np.asarray(want)
    assert np.abs(want).max() > 1.0  # the reference is not trivially small
    err = float(np.abs(got - want).max())
    print(f"{case} k={k} n={n} ws={ws}: max abs error {err:.3e}")
    # bfloat16 inputs are exact in float32, so only the float32 summation order differs from the reference.
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3 * np.sqrt(k))
