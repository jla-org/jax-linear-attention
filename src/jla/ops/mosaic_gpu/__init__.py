# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pallas Mosaic GPU (Hopper) kernels for the linear-attention family.

Importing this package registers every available kernel with
:func:`jla.ops.register` under ``impl="mosaic_gpu"``; each kernel's
``supports`` predicate sends calls it cannot handle (no sm_90 GPU, other
shapes or options) to the next implementation under ``impl="auto"``. A JAX
build without the Mosaic GPU toolchain registers nothing.
"""

import importlib

try:
    import jax.experimental.pallas.mosaic_gpu  # noqa: F401
except ImportError:
    _HAVE_MOSAIC_GPU = False
else:
    _HAVE_MOSAIC_GPU = True

if _HAVE_MOSAIC_GPU:
    from jla.ops.mosaic_gpu import gated_delta_rule, scalar_gate  # noqa: F401  (register on import)

    for _name in ("kda", "gdn2", "mamba3", "gla"):
        try:
            importlib.import_module(f"jla.ops.mosaic_gpu.{_name}")
        except ModuleNotFoundError as err:
            if err.name != f"jla.ops.mosaic_gpu.{_name}":
                raise
    try:  # Mamba-3 MIMO on the SISO kernels
        importlib.import_module("jla.ops.mosaic_gpu.mamba3_mimo")
    except ModuleNotFoundError as err:
        if err.name not in ("jla.ops.mosaic_gpu.mamba3_mimo", "jla.ops.mosaic_gpu.mamba3"):
            raise

if _HAVE_MOSAIC_GPU:
    from jla.ops.mosaic_gpu import gated_delta_product  # noqa: F401,E402  (register on import)

if _HAVE_MOSAIC_GPU:
    from jla.ops.mosaic_gpu import rwkv7  # noqa: F401,E402  (register on import)
    from jla.ops.mosaic_gpu import mlstm  # noqa: F401,E402  (register on import)
    from jla.ops.mosaic_gpu import mamba1  # noqa: F401,E402  (register on import)
