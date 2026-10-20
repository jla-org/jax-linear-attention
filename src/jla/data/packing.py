# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Document boundaries inside packed windows.

Corpora store documents back to back with an EOS token after each one. For
document-masked training every token gets the id of the document it belongs
to (the EOS token closes its own document, the next token starts a new one)
and a position that restarts at 0 for every document. The model then attends
and carries recurrent state only within a segment.

The target at an EOS position is the first token of the *next* document,
which a document-masked model cannot see; :class:`DocumentFields` drops those
targets from the loss (``loss_mask``), as Megatron's ``--eod-mask-loss`` does,
so no prediction (main head or MTP) crosses a document boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DocumentFields", "positions_from_segment_ids", "segment_ids_from_eos"]


def segment_ids_from_eos(tokens: np.ndarray, eos_id: int) -> np.ndarray:
    """``int32`` segment ids along the last axis, starting at 0 and incremented after each EOS."""
    starts = np.zeros(tokens.shape, dtype=bool)
    starts[..., 1:] = tokens[..., :-1] == eos_id
    return np.cumsum(starts, axis=-1, dtype=np.int32)


def positions_from_segment_ids(segment_ids: np.ndarray) -> np.ndarray:
    """``int32`` position of every token inside its segment."""
    length = segment_ids.shape[-1]
    index = np.broadcast_to(np.arange(length, dtype=np.int64), segment_ids.shape)
    starts = np.ones(segment_ids.shape, dtype=bool)
    starts[..., 1:] = segment_ids[..., 1:] != segment_ids[..., :-1]
    start_index = np.maximum.accumulate(np.where(starts, index, 0), axis=-1)
    return (index - start_index).astype(np.int32)


class DocumentFields:
    """Batch transform adding ``segment_ids``, ``positions`` and an EOS-target ``loss_mask`` (picklable)."""

    def __init__(self, eos_id: int) -> None:
        self.eos_id = int(eos_id)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        tokens = batch["input_ids"]
        segment_ids = segment_ids_from_eos(tokens, self.eos_id)
        within_document = tokens != self.eos_id
        if "loss_mask" in batch:
            within_document = within_document & batch["loss_mask"].astype(bool)
        return {
            **batch,
            "segment_ids": segment_ids,
            "positions": positions_from_segment_ids(segment_ids),
            "loss_mask": within_document,
        }
