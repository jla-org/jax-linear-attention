# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Durable run records: an fsynced JSONL metric log and an atomic run manifest.

Console logs are not a reliable record of a long run. Every metric record is
one ``O_APPEND`` write followed by ``fsync``, so a killed process leaves only
complete lines; manifests are replaced atomically. The sink is host-only:
callers pass values already fetched at a logging boundary.
"""

from __future__ import annotations

import dataclasses
import enum
import errno
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["METRIC_SCHEMA", "RUN_SCHEMA", "MetricSink", "atomic_write_json", "json_safe", "write_run_manifest"]

METRIC_SCHEMA = "jla.training-metrics.v1"
RUN_SCHEMA = "jla.training-run.v1"


def json_safe(value: Any) -> Any:
    """Strict-JSON form of ``value``: non-finite floats become strings, scalars unwrap."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, enum.Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        offset += os.write(fd, payload[offset:])


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably replace a JSON object without ever exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    encoded = (json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary.unlink(missing_ok=True)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, destination)
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            except OSError:
                pass  # some network filesystems reject directory fsync after a durable file fsync
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class MetricSink:
    """Append strict JSON records (one write + fsync each) to ``path``."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not str(run_id).strip():
            raise ValueError("run_id must be non-empty")
        self.path = Path(path)
        self.run_id = str(run_id).strip()
        self._lock = threading.Lock()
        self._rewrite = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append_by_rewrite(self, encoded: bytes) -> None:
        # For filesystems that reject O_APPEND: rewrite the file atomically.
        existing = self.path.read_bytes() if self.path.exists() else b""
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                _write_all(fd, existing + encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def append(self, event: str, *, step: int, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not str(event).strip():
            raise ValueError("event must be non-empty")
        if isinstance(step, bool) or int(step) < 0:
            raise ValueError("step must be a non-negative integer")
        record = {
            "schema": METRIC_SCHEMA,
            "run_id": self.run_id,
            "event": str(event).strip(),
            "step": int(step),
            "time_unix_ns": time.time_ns(),
            "metrics": json_safe(dict(metrics or {})),
        }
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with self._lock:
            if self._rewrite:
                self._append_by_rewrite(encoded)
                return record
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            failure: OSError | None = None
            try:
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise OSError(f"short JSONL append: wrote {written} of {len(encoded)} bytes")
                os.fsync(fd)
            except OSError as exc:
                failure = exc
            finally:
                os.close(fd)
            if failure is not None:
                if failure.errno not in (errno.ESPIPE, errno.EINVAL, errno.ENOTSUP):
                    raise failure
                self._rewrite = True
                self._append_by_rewrite(encoded)
        return record


def write_run_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish the resolved run description (config, hash, devices) next to the checkpoints."""
    return atomic_write_json(path, {"schema": RUN_SCHEMA, "created_unix_ns": time.time_ns(), **dict(payload)})
