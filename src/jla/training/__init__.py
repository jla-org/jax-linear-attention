# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Training stack: compiled step, trainer loop, optimizers, checkpoints, diagnostics.

Submodules are imported explicitly (``from jla.training.step import ...``);
this package does not import JAX so that host-only tools such as
:mod:`jla.training.graph_health` stay importable without it.
"""
