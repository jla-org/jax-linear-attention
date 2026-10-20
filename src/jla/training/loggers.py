# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Metric loggers selected by ``train.loggers``: JSONL (default), Weights & Biases, TensorBoard.

Every sink takes the records the trainer already writes to its JSONL log
(``append(event, step=, metrics=)``). The JSONL sink is the durable record
(:class:`jla.training.metrics_sink.MetricSink`); W&B and TensorBoard receive
the numeric scalars of each record as ``<event>/<name>`` (nested mappings join
with ``/``; strings are skipped). The optional packages are imported only when
selected, and nothing is configured through environment variables: project,
entity, mode and directories are explicit config fields.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from jla.training.metrics_sink import MetricSink

__all__ = ["MetricLoggers", "TensorBoardLogger", "WandbLogger", "metric_loggers", "scalar_items"]


def scalar_items(metrics: Mapping[str, Any] | None, prefix: str = "") -> Iterator[tuple[str, float]]:
    """``(name, float)`` for every numeric scalar of a (nested) metric mapping."""
    for key, value in (metrics or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from scalar_items(value, f"{name}/")
            continue
        item = getattr(value, "item", None)
        if callable(item) and getattr(value, "ndim", 0) == 0:
            value = item()
        if isinstance(value, (bool, numbers.Real)) and not isinstance(value, str):
            yield name, float(value)


def _tagged(event: str, metrics: Mapping[str, Any] | None) -> Iterator[tuple[str, float]]:
    """``(<event>/<name>, value)``; names that already start with ``<event>/`` keep their prefix once."""
    for name, value in scalar_items(metrics):
        yield (name if name.startswith(f"{event}/") else f"{event}/{name}"), value


class TensorBoardLogger:
    """Scalars to a TensorBoard event file (optional ``tensorboardX`` package)."""

    def __init__(self, logdir: str | Path) -> None:
        try:
            from tensorboardX import SummaryWriter
        except ImportError as exc:
            raise ImportError("train.loggers 'tensorboard' needs the optional 'tensorboardX' package") from exc
        self.logdir = Path(logdir)
        self.writer = SummaryWriter(logdir=str(self.logdir))

    def append(self, event: str, *, step: int, metrics: Mapping[str, Any] | None = None) -> None:
        for tag, value in _tagged(event, metrics):
            self.writer.add_scalar(tag, value, global_step=int(step))

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class WandbLogger:
    """Scalars to a Weights & Biases run (optional ``wandb`` package); ``mode="offline"`` writes local files only.

    Non-finite values are not sent (the JSONL record keeps them); steps must not decrease within a run.
    """

    def __init__(
        self,
        *,
        project: str,
        directory: str | Path,
        mode: str = "online",
        entity: str | None = None,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
        run_id: str = "",
    ) -> None:
        if mode not in ("online", "offline"):
            raise ValueError(f"wandb mode must be 'online' or 'offline', got {mode!r}")
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("train.loggers 'wandb' needs the optional 'wandb' package") from exc
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            dir=str(directory),
            mode=mode,
            config=dict(config or {}),
            id=run_id or None,
            resume="allow" if run_id else None,
            settings=wandb.Settings(console="off"),
        )

    def append(self, event: str, *, step: int, metrics: Mapping[str, Any] | None = None) -> None:
        values = {tag: value for tag, value in _tagged(event, metrics) if math.isfinite(value)}
        if values:
            self.run.log(values, step=int(step))

    def close(self) -> None:
        self.run.finish()


class MetricLoggers:
    """Fan-out of each record to every configured sink, in ``train.loggers`` order."""

    def __init__(self, sinks: Sequence[Any]) -> None:
        if not sinks:
            raise ValueError("MetricLoggers needs at least one sink")
        self.sinks = list(sinks)

    def append(self, event: str, *, step: int, metrics: Mapping[str, Any] | None = None) -> None:
        for sink in self.sinks:
            sink.append(event, step=step, metrics=metrics)

    def close(self) -> None:
        errors = []
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # close every sink, then report the first failure
                errors.append(exc)
        if errors:
            raise errors[0]


def metric_loggers(config: Any, output_dir: str | Path, *, run_id: str) -> MetricLoggers | None:
    """The sinks of ``config.train.loggers`` for a run writing to ``output_dir`` (``None`` if empty)."""
    train = config.train
    output = Path(output_dir)
    sinks: list[Any] = []
    for kind in train.loggers:
        if kind == "jsonl":
            sinks.append(MetricSink(output / "metrics.jsonl", run_id=run_id))
        elif kind == "tensorboard":
            sinks.append(TensorBoardLogger(train.tensorboard_dir or output / "tensorboard"))
        elif kind == "wandb":
            sinks.append(
                WandbLogger(
                    project=train.wandb_project,
                    entity=train.wandb_entity,
                    mode=train.wandb_mode,
                    directory=output,
                    name=config.name,
                    config=config.to_dict(),
                    run_id=train.wandb_run_id,
                )
            )
        else:
            raise ValueError(f"unknown logger {kind!r}; expected 'jsonl', 'wandb' or 'tensorboard'")
    return MetricLoggers(sinks) if sinks else None
