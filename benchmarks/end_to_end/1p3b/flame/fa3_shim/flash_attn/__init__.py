# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-2 import names over FlashAttention-3 (hopper), for FLA's softmax-attention layer.

``fla.layers.attn`` imports ``flash_attn_func`` and ``flash_attn_varlen_func`` from the FA2 package.
FA3's ``flash_attn_interface`` functions take the keyword arguments FLA passes (``causal``,
``window_size``, ``cu_seqlens_q/k``, ``max_seqlen_q/k``) and return the output tensor, so they are
re-exported unchanged. ``benchmarks/bench_e2e_flame.py --fa3-shim`` puts this directory first on
``sys.path`` (FA2 publishes no wheel for torch 2.13; FA3 is the faster kernel on H100).
"""

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func  # noqa: F401

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
