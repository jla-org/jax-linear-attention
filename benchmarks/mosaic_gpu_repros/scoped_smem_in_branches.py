# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mosaic GPU (JAX 0.11.2, H100): SMEM ``pl.run_scoped`` inside warpgroup-specific ``pl.when``.

Each of two warpgroups allocates "its own" buffer with ``pl.run_scoped`` in its branch. The
lowering rejects this ("Only thread-collective allocations are supported in multithreaded
kernels"), and the GPU interpreter likewise asks for ``collective_axes``: per-warpgroup buffers
have to be kernel scratch (or one collective allocation that every warpgroup enters).
"""

import functools

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

LINE = plgpu.Layout.WG_STRIDED((128,), vec_size=1)


@functools.partial(plgpu.kernel, out_type=jax.ShapeDtypeStruct((2, 128), jnp.float32), num_threads=2,
                   thread_name="wg", compiler_params=plgpu.CompilerParams(
                       lowering_semantics=plgpu.LoweringSemantics.Lane))
def kernel(o_ref):
    wg = lax.axis_index("wg")
    for w in range(2):

        @pl.when(wg == w)
        def _():
            def scoped(buf):
                buf[...] = plgpu.layout_cast(jnp.full((128,), w + 1.0, jnp.float32), LINE)
                o_ref.at[w][...] = plgpu.load(buf, layout=LINE)

            pl.run_scoped(scoped, plgpu.SMEM((128,), jnp.float32))


try:
    print(jax.jit(kernel)())
except NotImplementedError as err:
    print("NotImplementedError:", str(err).splitlines()[0])
