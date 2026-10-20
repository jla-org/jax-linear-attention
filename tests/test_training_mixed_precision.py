# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision policy validation and dtype-preserving casts."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _toy_model import TOY_MODEL
from jla.training import mixed_precision as mp


def test_policy_defaults_are_safe_and_serializable() -> None:
    policy = mp.MixedPrecisionPolicy()
    assert policy.param_dtype == jnp.float32 and policy.compute_dtype == jnp.bfloat16
    assert policy.accum_dtype == jnp.float32 and policy.loss_dtype == jnp.float32
    metadata = policy.as_dict()
    assert json.loads(json.dumps(metadata)) == metadata
    assert mp.MixedPrecisionPolicy.from_model_config(TOY_MODEL).compute_dtype == jnp.float32


def test_replace_revalidates_and_accumulation_needs_fp32() -> None:
    original = mp.MixedPrecisionPolicy()
    changed = original.replace(compute_dtype=jnp.float16)
    assert original.compute_dtype == jnp.bfloat16 and changed.compute_dtype == jnp.float16
    for field in ("accum_dtype", "loss_dtype"):
        for unsafe in (jnp.float16, jnp.bfloat16):
            with pytest.raises(TypeError, match=field):
                mp.MixedPrecisionPolicy(**{field: unsafe})
    with pytest.raises(TypeError, match="real floating"):
        mp.MixedPrecisionPolicy(compute_dtype=jnp.complex64)


def test_generic_fp8_casting_is_rejected() -> None:
    for field in ("param_dtype", "compute_dtype"):
        with pytest.raises(TypeError, match="scaled-matmul"):
            mp.MixedPrecisionPolicy(**{field: jnp.float8_e4m3fn})


def test_casts_preserve_integer_boolean_complex_and_none_leaves() -> None:
    tree = {
        "real": jnp.array([1.0, 2.0], jnp.float32),
        "complex": jnp.array([1.0 + 2.0j], jnp.complex64),
        "integer": jnp.array([1, 2], jnp.int32),
        "boolean": jnp.array([True, False]),
        "none": None,
    }
    cast = mp.cast_floating(tree, jnp.bfloat16)
    assert cast["real"].dtype == jnp.bfloat16 and cast["complex"].dtype == jnp.complex64
    assert cast["integer"].dtype == jnp.int32 and cast["boolean"].dtype == jnp.bool_ and cast["none"] is None
    restored = jax.jit(mp.restore_dtypes)(jax.tree.map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, cast), cast)
    assert restored["real"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(restored["complex"], tree["complex"])


def test_parameter_dtype_check_and_fp8_capability() -> None:
    mp.check_param_dtypes({"w": jnp.ones(2), "step": jnp.zeros((), jnp.int32)}, "float32")
    with pytest.raises(TypeError, match="must be stored in float32"):
        mp.check_param_dtypes({"w": jnp.ones(2, jnp.bfloat16)}, "float32")
    assert mp.is_fp8_capable("NVIDIA H100 80GB HBM3") and not mp.is_fp8_capable("NVIDIA A100-SXM4-80GB")
