# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Linear-attention layers that context parallelism does not support fail loudly.

``LinearAttentionConfig.context_parallel_axis`` shards ``T`` for the kinds in
``jla.distributed.context_parallel.CONTEXT_PARALLEL_KINDS``. Every other layer that
reads this config must raise instead of silently running unsharded.
"""

from __future__ import annotations

import dataclasses

import jax
import pytest

from jla.distributed.context_parallel import CONTEXT_PARALLEL_KINDS
from jla.models import build_model, get_preset


@pytest.mark.parametrize("name", ["tiny-rwkv6", "tiny-hgrn2", "tiny-ttt", "tiny-mlstm", "tiny-deltaproduct"])
def test_unsupported_kinds_raise_with_context_parallel_axis(name):
    cfg = get_preset(name).replace(compute_dtype="float32")
    kinds = {kind for kind in cfg.resolved_layer_types if kind != "attention"}
    assert kinds and not kinds & set(CONTEXT_PARALLEL_KINDS), kinds
    cfg = cfg.replace(linear_attention=dataclasses.replace(cfg.linear_attention, context_parallel_axis="cp"))
    model = build_model(cfg, key=jax.random.key(0))
    ids = jax.random.randint(jax.random.key(1), (1, 32), 0, cfg.vocab_size)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        model(ids)
