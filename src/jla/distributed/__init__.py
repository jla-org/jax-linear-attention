# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device meshes, multi-host initialization and logical-axis sharding.

* :mod:`jla.distributed.mesh`: ``(dp, fsdp, tp)`` mesh planning, ``fsdp=-1``
  fill, hybrid ICI/DCN meshes;
* :mod:`jla.distributed.init`: ``jax.distributed`` setup from explicit
  settings, with opt-in SLURM/Open MPI/torchrun environment discovery, and
  host-coordination helpers;
* :mod:`jla.distributed.sharding`: MaxText-style logical-axis rules,
  parameter/optimizer-state shardings, the replicated-parameter audit and
  per-host batch placement;
* :mod:`jla.distributed.expert_parallel`: MoE experts sharded over an ``ep``
  mesh axis with all-to-all token dispatch;
* :mod:`jla.distributed.pipeline`: GPipe pipeline parallelism over the ``pp``
  mesh axis (stages are slices of the scanned layer stack).
"""
