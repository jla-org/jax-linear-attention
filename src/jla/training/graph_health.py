# Copyright 2026 KarLex AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, JAX-free graph-health analysis of XLA HLO dumps and XProf traces.

The training step stays free of host callbacks; this module inspects the
artifacts produced beside it (``XLA_FLAGS=--xla_dump_to=...`` text dumps or a
profiler trace) and counts patterns that usually mean lost throughput:
device-to-device copy storms, dynamic-update-slice and tiny loop-fusion
fragmentation, ``while``/``scan`` fragments, host callbacks, custom and Pallas
calls, materialized attention matrices and whether XLA command buffers (CUDA
graphs) were used. Reads are bounded per file, in total, in file count and in
directory entries; symlinks and special files are never followed.

    python -m jla.training.graph_health TRACE_OR_DIR [--strict] [--policy release]
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import operator
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

__all__ = [
    "GraphHealthArtifactError",
    "GraphHealthMetrics",
    "GraphHealthPolicy",
    "analyze_graph_health_path",
    "analyze_xprof_trace_attribution",
    "graph_health_policy",
    "graph_health_release_warnings",
    "graph_health_warnings",
    "main",
    "merge_graph_health_metrics",
    "metrics_as_dict",
]

_DEFAULT_MAX_BYTES_PER_FILE = 32 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_FILES = 4096
_DEFAULT_MAX_ENTRIES = 65_536

_TEXT_SUFFIXES = {".hlo", ".html", ".json", ".mlir", ".pbtxt", ".txt"}
_GZIP_SUFFIXES = {".gz"}

_COPY_RE = re.compile(r"MemcpyD2D|^\s*%[\w.-]+\s*=.*\b(?:copy|copy-start|copy-done)\s*\(", re.I | re.M)
_DYNAMIC_UPDATE_RE = re.compile(r"loop_dynamic_update_slice_fusion|dynamic[-_]update[-_]slice", re.I)
_LOOP_ADD_RE = re.compile(r"loop_add_fusion", re.I)
_SCAN_WHILE_RE = re.compile(r"\b(?:scan|while)\b|while\.", re.I)
_HOST_CALLBACK_RE = re.compile(
    r"host[-_ ]callback|pure_callback|io_callback|debug_callback|outfeed|infeed|outside[-_ ]compilation", re.I
)
_CUSTOM_CALL_RE = re.compile(r"custom[-_]?call|ffi_call|\bcublas(?:lt)?\b|\bcudnn\b", re.I)
_PALLAS_CALL_RE = re.compile(r"\bpallas(?:_call)?\b|mosaic", re.I)
_ATTENTION_MATERIALIZATION_RE = re.compile(
    r"attention.*materiali[sz]e|materiali[sz]ed.*attention|full.*attention.*matrix|\bT\s*\*\s*T\b|\bT\s*x\s*T\b", re.I
)
_TINY_KERNEL_FRAGMENT_RE = re.compile(
    r"loop_dynamic_update_slice_fusion|loop_add_fusion|fusion\([^)]*\),\s*kind=kLoop|MemcpyD2D", re.I
)
_COMMAND_BUFFER_RE = re.compile(r"command[-_ ]buffer|cuda[-_ ]graph", re.I)
_HOST_DISPATCH_RE = re.compile(
    r"PjRtLoadedExecutable::Execute|ExecuteHelper|TfrtCpuClient::Compile|xla::LocalExecutable::Run|host dispatch", re.I
)
_INSTRUCTION_RE = re.compile(r"^\s*%[\w.-]+\s*=", re.M)
_TRACE_MEMCPY_RE = re.compile(r"MemcpyD2D|memcpy.*d2d", re.I)
_HLO_INSTRUCTION_RE = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.-]+)\s*=", re.MULTILINE)
_HLO_SOURCE_FILE_RE = re.compile(r'\bsource_file="((?:\\.|[^"\\])*)"')
_HLO_SOURCE_LINE_RE = re.compile(r"\bsource_line=(\d+)")
_HLO_STACK_FRAME_ID_RE = re.compile(r"\bstack_frame_id=(\d+)")
_HLO_TABLE_STRING_RE = re.compile(r'^(\d+) "((?:\\.|[^"\\])*)"$')
_HLO_FILE_LOCATION_RE = re.compile(r"^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+)\b")
_HLO_STACK_FRAME_RE = re.compile(r"^(\d+) \{file_location_id=(\d+) parent_frame_id=(\d+)\}")
_HLO_OPERAND_RE = re.compile(r"%([A-Za-z0-9_.-]+)")
_INSTANCE_SUFFIX_RE = re.compile(r"(?:__\d+|\.\d+)(?=\b|$)")
_LAYER_SCOPE_RE = re.compile(r"\blayer_\d+\b")

_COUNTED = (
    ("memcpy_d2d_count", _TRACE_MEMCPY_RE),
    ("loop_add_fusion_count", _LOOP_ADD_RE),
    ("loop_dynamic_update_slice_fusion_count", _DYNAMIC_UPDATE_RE),
    ("scan_while_fragment_count", _SCAN_WHILE_RE),
    ("host_dispatch_count", _HOST_DISPATCH_RE),
    ("host_callback_count", _HOST_CALLBACK_RE),
    ("custom_call_count", _CUSTOM_CALL_RE),
    ("pallas_call_count", _PALLAS_CALL_RE),
    ("attention_materialization_count", _ATTENTION_MATERIALIZATION_RE),
    ("tiny_kernel_fragment_count", _TINY_KERNEL_FRAGMENT_RE),
)


class GraphHealthArtifactError(RuntimeError):
    """Strict analysis could not produce complete evidence."""


@dataclass
class GraphHealthMetrics:
    """Counters for one artifact set plus parser status (so empty never looks clean)."""

    total_events: int = 0
    memcpy_d2d_count: int = 0
    loop_add_fusion_count: int = 0
    loop_dynamic_update_slice_fusion_count: int = 0
    scan_while_fragment_count: int = 0
    host_dispatch_count: int = 0
    host_callback_count: int = 0
    custom_call_count: int = 0
    pallas_call_count: int = 0
    attention_materialization_count: int = 0
    tiny_kernel_fragment_count: int = 0
    command_buffer_present: bool | None = None  # None = no evidence either way
    artifact_files_discovered: int = 0
    artifact_files_parsed: int = 0
    artifact_bytes_read: int = 0
    artifact_read_error_count: int = 0
    artifact_parse_error_count: int = 0
    artifact_truncated: bool = False
    artifact_limit_reached: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GraphHealthMetrics:
        """Build from a mapping of field names; unknown command-buffer text stays unproven."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in values:
                continue
            value = values[f.name]
            if f.name == "command_buffer_present":
                kwargs[f.name] = _optional_bool(value)
            elif f.name in ("artifact_truncated", "artifact_limit_reached"):
                kwargs[f.name] = bool(_optional_bool(value))
            else:
                kwargs[f.name] = int(value)
        return cls(**kwargs)


_METRIC_COUNT_FIELDS = tuple(
    f.name for f in fields(GraphHealthMetrics) if f.name not in ("command_buffer_present", "artifact_truncated", "artifact_limit_reached")
)


def _optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "present"):
        return True
    if text in ("0", "false", "no", "off", "absent"):
        return False
    return None


@dataclass(frozen=True)
class GraphHealthPolicy:
    """Warning thresholds (``None`` disables a check)."""

    memcpy_d2d_warn: int = 10_000
    dynamic_update_slice_warn: int = 1_000
    loop_add_warn: int = 10_000
    scan_while_warn: int = 1_000
    host_dispatch_warn: int = 1
    host_callback_warn: int = 0
    pallas_call_warn: int | None = None
    attention_materialization_warn: int | None = None
    tiny_kernel_fragment_warn: int = 10_000
    total_events_warn: int | None = None
    require_command_buffer: bool = False
    require_artifacts: bool = False


_POLICIES: dict[str, GraphHealthPolicy] = {
    "default": GraphHealthPolicy(),
    "release": GraphHealthPolicy(
        attention_materialization_warn=0,
        total_events_warn=100_000,
        require_command_buffer=True,
        require_artifacts=True,
    ),
}


def graph_health_policy(name: str | None = None, *, strict: bool = False) -> GraphHealthPolicy:
    """Named policy; unknown names fall back to ``default`` unless ``strict``."""
    key = (name or "default").strip().lower()
    if key in _POLICIES:
        return _POLICIES[key]
    if strict:
        raise ValueError(f"unknown graph-health policy {name!r}; expected one of {sorted(_POLICIES)}")
    return _POLICIES["default"]


def _exact_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    parsed = operator.index(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return int(parsed)


def _supported_artifact(path: Path) -> bool:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(suffixes & _TEXT_SUFFIXES or suffixes & _GZIP_SUFFIXES)


@dataclass
class _Traversal:
    entries_scanned: int = 0
    limit_reached: bool = False


def _iter_artifact_files(root: Path, *, errors: list[OSError], max_entries: int, state: _Traversal) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries: list[os.DirEntry[str]] = []
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if state.entries_scanned >= max_entries:
                        state.limit_reached = True
                        break
                    state.entries_scanned += 1
                    entries.append(entry)
        except OSError as exc:
            errors.append(exc)
            continue
        subdirectories: list[Path] = []
        for entry in sorted(entries, key=lambda item: item.name):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    subdirectories.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and _supported_artifact(Path(entry.path)):
                    yield Path(entry.path)
            except OSError as exc:
                errors.append(exc)
        if state.limit_reached:
            return
        stack.extend(reversed(subdirectories))


def _open_regular(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened, current = os.fstat(descriptor), os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode):
            raise OSError(f"artifact is not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise OSError(f"artifact changed while it was being opened: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_text(path: Path, *, max_bytes: int) -> tuple[str, int, bool]:
    with _open_regular(path) as raw:
        if path.suffix.lower() == ".gz":
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                payload = stream.read(max_bytes + 1)
        else:
            payload = raw.read(max_bytes + 1)
    truncated = len(payload) > max_bytes
    payload = payload[:max_bytes]
    return payload.decode("utf-8", errors="replace"), len(payload), truncated


def _count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def _event_name(event: Any) -> str:
    if isinstance(event, Mapping):
        pieces = (str(event.get("name", "")), str(event.get("cat", "")), json.dumps(event.get("args", {}), sort_keys=True, default=str))
        return " ".join(piece for piece in pieces if piece)
    return str(event)


def _metrics_from_trace_json(value: Any) -> GraphHealthMetrics | None:
    events = value if isinstance(value, list) else value.get("traceEvents") if isinstance(value, Mapping) else None
    if not isinstance(events, list):
        return None
    metrics = GraphHealthMetrics(total_events=len(events), command_buffer_present=False)
    for event in events:
        name = _event_name(event)
        for name_field, pattern in _COUNTED:
            setattr(metrics, name_field, getattr(metrics, name_field) + _count(pattern, name))
        if _COMMAND_BUFFER_RE.search(name):
            metrics.command_buffer_present = True
    return metrics


def _metrics_from_text(text: str) -> GraphHealthMetrics:
    command_buffer = bool(_COMMAND_BUFFER_RE.search(text))
    return GraphHealthMetrics(
        total_events=_count(_INSTRUCTION_RE, text),
        memcpy_d2d_count=_count(_COPY_RE, text),
        loop_add_fusion_count=_count(_LOOP_ADD_RE, text),
        loop_dynamic_update_slice_fusion_count=_count(_DYNAMIC_UPDATE_RE, text),
        scan_while_fragment_count=_count(_SCAN_WHILE_RE, text),
        host_dispatch_count=_count(_HOST_DISPATCH_RE, text),
        host_callback_count=_count(_HOST_CALLBACK_RE, text),
        custom_call_count=_count(_CUSTOM_CALL_RE, text),
        pallas_call_count=_count(_PALLAS_CALL_RE, text),
        attention_materialization_count=_count(_ATTENTION_MATERIALIZATION_RE, text),
        tiny_kernel_fragment_count=_count(_TINY_KERNEL_FRAGMENT_RE, text),
        command_buffer_present=True if command_buffer else None,
    )


def merge_graph_health_metrics(metrics: Iterable[GraphHealthMetrics]) -> GraphHealthMetrics:
    """Sum counters; any positive command-buffer evidence wins over absence."""
    merged = GraphHealthMetrics()
    evidence: bool | None = None
    for item in metrics:
        for name in _METRIC_COUNT_FIELDS:
            setattr(merged, name, getattr(merged, name) + getattr(item, name))
        merged.artifact_truncated = merged.artifact_truncated or item.artifact_truncated
        merged.artifact_limit_reached = merged.artifact_limit_reached or item.artifact_limit_reached
        if item.command_buffer_present is True:
            evidence = True
        elif evidence is None and item.command_buffer_present is False:
            evidence = False
    merged.command_buffer_present = evidence
    return merged


def analyze_graph_health_path(
    path: str | Path,
    *,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES_PER_FILE,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
    max_files: int = _DEFAULT_MAX_FILES,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    strict: bool = False,
) -> GraphHealthMetrics:
    """Parse one artifact file or directory under explicit resource bounds.

    Non-strict mode returns lower-bound counters with parser status; strict
    mode raises :class:`GraphHealthArtifactError` when evidence is missing or
    incomplete. Missing paths and explicit symlinks are always rejected.
    """
    max_bytes_per_file = _exact_positive_int("max_bytes_per_file", max_bytes_per_file)
    max_total_bytes = _exact_positive_int("max_total_bytes", max_total_bytes)
    max_files = _exact_positive_int("max_files", max_files)
    max_entries = _exact_positive_int("max_entries", max_entries)
    root = Path(path).expanduser()
    if root.is_symlink():
        raise GraphHealthArtifactError(f"artifact path must not be a symlink: {root}")
    if not root.exists():
        raise FileNotFoundError(f"graph-health artifact path does not exist: {root}")
    if not root.is_file() and not root.is_dir():
        raise GraphHealthArtifactError(f"artifact path is not a regular file or directory: {root}")

    per_file: list[GraphHealthMetrics] = []
    walk_errors: list[OSError] = []
    traversal = _Traversal()
    discovered = parsed = bytes_read = read_errors = parse_errors = 0
    truncated = limit_reached = False
    for artifact in _iter_artifact_files(root, errors=walk_errors, max_entries=max_entries, state=traversal):
        discovered += 1
        if discovered > max_files or bytes_read >= max_total_bytes:
            limit_reached = True
            break
        read_limit = min(max_bytes_per_file, max_total_bytes - bytes_read)
        try:
            text, size, artifact_truncated = _read_text(artifact, max_bytes=read_limit)
        except (OSError, EOFError) as exc:
            read_errors += 1
            if strict:
                raise GraphHealthArtifactError(f"could not read graph-health artifact {artifact}: {exc}") from exc
            continue
        bytes_read += size
        truncated = truncated or artifact_truncated
        if artifact_truncated and read_limit < max_bytes_per_file:
            limit_reached = True
        if not text:
            continue
        parsed += 1
        decoded: Any = None
        if ".json" in {suffix.lower() for suffix in artifact.suffixes} and not artifact_truncated:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                parse_errors += 1
                if strict:
                    raise GraphHealthArtifactError(f"invalid JSON graph-health artifact: {artifact}") from None
        trace = _metrics_from_trace_json(decoded)
        per_file.append(trace if trace is not None else _metrics_from_text(text))
        if limit_reached:
            break

    merged = merge_graph_health_metrics(per_file)
    merged.artifact_files_discovered = discovered
    merged.artifact_files_parsed = parsed
    merged.artifact_bytes_read = bytes_read
    merged.artifact_read_error_count = read_errors + len(walk_errors)
    merged.artifact_parse_error_count = parse_errors
    merged.artifact_truncated = truncated
    merged.artifact_limit_reached = limit_reached or traversal.limit_reached
    if strict:
        problems = []
        if parsed == 0:
            problems.append("no non-empty supported artifacts were parsed")
        if merged.artifact_read_error_count:
            problems.append(f"{merged.artifact_read_error_count} artifact read error(s)")
        if parse_errors:
            problems.append(f"{parse_errors} JSON parse error(s)")
        if truncated:
            problems.append("one or more artifacts were truncated")
        if merged.artifact_limit_reached:
            problems.append("an artifact resource limit was reached")
        if problems:
            raise GraphHealthArtifactError("; ".join(problems))
    return merged


def graph_health_warnings(metrics: GraphHealthMetrics, policy: GraphHealthPolicy | None = None, **overrides: Any) -> list[str]:
    """Warnings for counters above the policy thresholds and for incomplete parsing."""
    policy = GraphHealthPolicy(**{**asdict(policy or GraphHealthPolicy()), **overrides})
    warnings: list[str] = []
    if metrics.artifact_read_error_count:
        warnings.append(f"artifact read errors: {metrics.artifact_read_error_count}")
    if metrics.artifact_parse_error_count:
        warnings.append(f"artifact JSON parse errors: {metrics.artifact_parse_error_count}")
    if metrics.artifact_truncated:
        warnings.append("artifact scan was truncated; counters are lower bounds")
    if metrics.artifact_limit_reached:
        warnings.append("artifact resource limit reached; scan is incomplete")
    if policy.require_artifacts and metrics.artifact_files_parsed <= 0:
        warnings.append("no non-empty graph-health artifacts were parsed")
    checks = (
        ("total trace events", metrics.total_events, policy.total_events_warn),
        ("MemcpyD2D storm", metrics.memcpy_d2d_count, policy.memcpy_d2d_warn),
        ("dynamic-update-slice storm", metrics.loop_dynamic_update_slice_fusion_count, policy.dynamic_update_slice_warn),
        ("loop_add_fusion storm", metrics.loop_add_fusion_count, policy.loop_add_warn),
        ("scan/while fragmentation", metrics.scan_while_fragment_count, policy.scan_while_warn),
        ("host dispatch count", metrics.host_dispatch_count, policy.host_dispatch_warn),
        ("host callback/outfeed/infeed count", metrics.host_callback_count, policy.host_callback_warn),
        ("Pallas call count", metrics.pallas_call_count, policy.pallas_call_warn),
        ("full attention materialization markers", metrics.attention_materialization_count, policy.attention_materialization_warn),
        ("tiny kernel/loop fragment storm", metrics.tiny_kernel_fragment_count, policy.tiny_kernel_fragment_warn),
    )
    for label, value, limit in checks:
        if limit is not None and value > limit:
            warnings.append(f"{label}: {value} > {limit}")
    if policy.require_command_buffer and metrics.command_buffer_present is not True:
        state = "absent" if metrics.command_buffer_present is False else "unproven"
        warnings.append(f"command-buffer/CUDA-graph replay evidence not found (evidence={state})")
    return warnings


def graph_health_release_warnings(metrics: GraphHealthMetrics, *, policy: str | None = None, strict: bool = False) -> list[str]:
    return graph_health_warnings(metrics, graph_health_policy(policy, strict=strict))


def metrics_as_dict(metrics: GraphHealthMetrics) -> dict[str, Any]:
    return asdict(metrics)


# ---------------------------------------------------------------------------
# XProf attribution: which device events (and which source lines) dominate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class _SourceLocation:
    file: str
    function: str | None
    line: int


class _SourceIndexVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.symbols: dict[str, list[_SourceLocation]] = {}
        self.functions: list[tuple[int, int, str]] = []

    def _location(self, line: int) -> _SourceLocation:
        return _SourceLocation(self.relative_path, ".".join(self.scope) if self.scope else None, line)

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.symbols.setdefault(node.name, []).append(self._location(node.lineno))
            self.functions.append((node.lineno, int(node.end_lineno or node.lineno), ".".join(self.scope)))
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node, node.name)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, str) and 2 <= len(value) <= 160 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
            self.symbols.setdefault(value, []).append(self._location(node.lineno))


def _build_source_index(source_root: str | Path) -> tuple[dict[str, list[_SourceLocation]], dict[str, list[tuple[int, int, str]]]]:
    root = Path(source_root).expanduser().resolve()
    symbols: dict[str, list[_SourceLocation]] = {}
    functions: dict[str, list[tuple[int, int, str]]] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        visitor = _SourceIndexVisitor(relative)
        visitor.visit(tree)
        functions[relative] = visitor.functions
        for symbol, locations in visitor.symbols.items():
            symbols.setdefault(symbol, []).extend(locations)
    return symbols, functions


def _relative_source_file(source_file: str, known_files: Iterable[str]) -> str:
    normalized = source_file.replace("\\", "/")
    matches = [item for item in known_files if normalized.endswith(item)]
    return max(matches, key=len) if matches else normalized


def _function_at_line(relative_path: str, line: int, functions: Mapping[str, list[tuple[int, int, str]]]) -> str | None:
    candidates = [(end - start, name) for start, end, name in functions.get(relative_path, ()) if start <= line <= end]
    return min(candidates)[1] if candidates else None


def _hlo_source_locations(hlo_path: str | Path | None, functions: Mapping[str, list[tuple[int, int, str]]]) -> dict[str, list[_SourceLocation]]:
    """Map HLO instruction names to Python source locations from a source-preserving dump."""
    if hlo_path is None:
        return {}
    root = Path(hlo_path).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.txt"))
    known = tuple(functions)
    result: dict[str, list[_SourceLocation]] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for path in paths:
        file_names: dict[int, str] = {}
        function_names: dict[int, str] = {}
        file_locations: dict[int, tuple[int, int, int]] = {}
        stack_frames: dict[int, tuple[int, int]] = {}
        section: str | None = None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for text in stream:
                    stripped = text.rstrip("\n")
                    if stripped in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
                        section = stripped
                        continue
                    if section in ("FileNames", "FunctionNames") and (match := _HLO_TABLE_STRING_RE.match(stripped)):
                        table = file_names if section == "FileNames" else function_names
                        table[int(match.group(1))] = bytes(match.group(2), "utf-8").decode("unicode_escape")
                    elif section == "FileLocations" and (match := _HLO_FILE_LOCATION_RE.match(stripped)):
                        file_locations[int(match.group(1))] = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
                    elif section == "StackFrames" and (match := _HLO_STACK_FRAME_RE.match(stripped)):
                        stack_frames[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for text in stream:
                    instruction = _HLO_INSTRUCTION_RE.match(text)
                    if instruction is None:
                        continue
                    name = instruction.group(1)
                    expression = text.split("=", 1)[1] if "=" in text else ""
                    dependencies[name] = tuple(dict.fromkeys(_HLO_OPERAND_RE.findall(expression)))
                    location: _SourceLocation | None = None
                    if stack := _HLO_STACK_FRAME_ID_RE.search(text):
                        frame = stack_frames.get(int(stack.group(1)))
                        file_location = file_locations.get(frame[0]) if frame else None
                        if file_location:
                            file_id, function_id, line = file_location
                            relative = _relative_source_file(file_names.get(file_id, ""), known)
                            location = _SourceLocation(
                                relative, function_names.get(function_id) or _function_at_line(relative, line, functions), line
                            )
                    if location is None:
                        file_match, line_match = _HLO_SOURCE_FILE_RE.search(text), _HLO_SOURCE_LINE_RE.search(text)
                        if file_match and line_match:
                            relative = _relative_source_file(bytes(file_match.group(1), "utf-8").decode("unicode_escape"), known)
                            line = int(line_match.group(1))
                            location = _SourceLocation(relative, _function_at_line(relative, line, functions), line)
                    if location is not None:
                        result.setdefault(name, []).append(location)
        except OSError:
            continue
    # Copies inserted by XLA inherit the location of their operand.
    unresolved = {name for name in dependencies if name.startswith("copy.") and name not in result}
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            inherited = next((result[op] for op in dependencies.get(name, ()) if op in result), None)
            if inherited is not None:
                result[name] = list(inherited)
                unresolved.remove(name)
                progressed = True
        if not progressed:
            break
    return result


def _gpu_process_ids(events: Iterable[Any]) -> set[Any]:
    return {
        event.get("pid")
        for event in events
        if isinstance(event, Mapping)
        and event.get("ph") == "M"
        and event.get("name") == "process_name"
        and isinstance(event.get("args"), Mapping)
        and str(event["args"].get("name", "")).startswith("/device:GPU:")
    }


def analyze_xprof_trace_attribution(
    path: str | Path,
    *,
    source_root: str | Path | None = None,
    hlo_path: str | Path | None = None,
    module: str | None = None,
    limit_per_class: int = 50,
) -> dict[str, Any]:
    """Attribute accelerator events of one XProf JSON trace to framework scopes and source lines."""
    limit_per_class = _exact_positive_int("limit_per_class", limit_per_class)
    trace_path = Path(path).expanduser()
    text, _, truncated = _read_text(trace_path, max_bytes=_DEFAULT_MAX_TOTAL_BYTES)
    if truncated:
        raise GraphHealthArtifactError(f"XProf trace exceeds {_DEFAULT_MAX_TOTAL_BYTES} bytes: {trace_path}")
    value = json.loads(text)
    events = value if isinstance(value, list) else value.get("traceEvents")
    if not isinstance(events, list):
        raise GraphHealthArtifactError(f"not an XProf JSON trace: {trace_path}")
    symbols, functions = _build_source_index(source_root) if source_root is not None else ({}, {})
    hlo_sources = _hlo_source_locations(hlo_path, functions)
    classes = {
        "pallas": _PALLAS_CALL_RE,
        "device_copy": _TRACE_MEMCPY_RE,
        "tiny_fragment": _TINY_KERNEL_FRAGMENT_RE,
        "scan_while_scope": _SCAN_WHILE_RE,
    }
    gpu_pids = _gpu_process_ids(events)
    if gpu_pids:
        gpu_events = [e for e in events if isinstance(e, Mapping) and e.get("ph") == "X" and e.get("pid") in gpu_pids]
    else:  # older exports lack process metadata; HLO-tagged events are still device records
        gpu_events = [
            e
            for e in events
            if isinstance(e, Mapping) and e.get("ph") == "X" and isinstance(e.get("args"), Mapping) and e["args"].get("hlo_op")
        ]
    device_events = [
        e
        for e in gpu_events
        if isinstance(e.get("args"), Mapping) and e["args"].get("hlo_op") and (module is None or e["args"].get("hlo_module") == module)
    ]

    def duration(items: Iterable[Mapping[str, Any]]) -> float:
        return sum(float(item.get("dur", 0.0) or 0.0) for item in items)

    gpu_time, device_time = duration(gpu_events), duration(device_events)
    summaries: dict[str, Any] = {}
    top: dict[str, list[dict[str, Any]]] = {}
    for event_class, pattern in classes.items():
        matching = [e for e in device_events if pattern.search(_event_name(e))]
        aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
        for event in matching:
            args = event["args"]
            kernel = _INSTANCE_SUFFIX_RE.sub("", str(event.get("name", "")))
            scope = _LAYER_SCOPE_RE.sub("layer_N", str(args.get("name") or args.get("prefix") or ""))
            hlo_op = _INSTANCE_SUFFIX_RE.sub("", str(args.get("hlo_op", "")))
            row = aggregates.setdefault(
                (kernel, scope, hlo_op),
                {"event_class": event_class, "kernel_name": kernel, "framework_op": scope, "hlo_op": hlo_op, "occurrences": 0, "total_duration_us": 0.0, "source_locations": set()},
            )
            row["occurrences"] += 1
            row["total_duration_us"] += float(event.get("dur", 0.0) or 0.0)
            row["source_locations"].update(hlo_sources.get(str(args.get("hlo_op", "")), ()))
            if event_class == "pallas":
                row["source_locations"].update(symbols.get(kernel, ()))
        rows = sorted(aggregates.values(), key=lambda item: item["total_duration_us"], reverse=True)[:limit_per_class]
        for row in rows:
            row["average_duration_us"] = row["total_duration_us"] / row["occurrences"]
            row["percent_of_device_time"] = 100.0 * row["total_duration_us"] / device_time if device_time else 0.0
            row["source_locations"] = [asdict(location) for location in sorted(row["source_locations"])]
        matched = duration(matching)
        summaries[event_class] = {
            "device_event_count": len(matching),
            "device_duration_us": matched,
            "percent_of_device_time": 100.0 * matched / device_time if device_time else 0.0,
        }
        top[event_class] = rows
    return {
        "trace_event_count": len(events),
        "gpu_event_count": len(gpu_events),
        "gpu_duration_us": gpu_time,
        "device_event_count": len(device_events),
        "device_duration_us": device_time,
        "module_filter": module,
        "class_summary": summaries,
        "top_device_events": top,
    }


def _positive_cli_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graph-health counters for HLO dumps and XProf traces.")
    parser.add_argument("path", help="HLO dump, JSON trace, or a directory of them")
    parser.add_argument("--max-bytes-per-file", type=_positive_cli_int, default=_DEFAULT_MAX_BYTES_PER_FILE)
    parser.add_argument("--max-total-bytes", type=_positive_cli_int, default=_DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--max-files", type=_positive_cli_int, default=_DEFAULT_MAX_FILES)
    parser.add_argument("--max-entries", type=_positive_cli_int, default=_DEFAULT_MAX_ENTRIES)
    parser.add_argument("--strict", action="store_true", help="fail on missing, unreadable or incomplete artifacts")
    parser.add_argument("--policy", default=None, help=f"warning thresholds: {sorted(_POLICIES)}")
    parser.add_argument("--require-command-buffer", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--attribution", action="store_true", help="attribute XProf device events")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--hlo-path", default=None)
    parser.add_argument("--attribution-module", default=None)
    parser.add_argument("--attribution-limit", type=_positive_cli_int, default=50)
    args = parser.parse_args(argv)
    try:
        metrics = analyze_graph_health_path(
            args.path,
            max_bytes_per_file=args.max_bytes_per_file,
            max_total_bytes=args.max_total_bytes,
            max_files=args.max_files,
            max_entries=args.max_entries,
            strict=args.strict,
        )
        policy = graph_health_policy(args.policy, strict=args.strict)
        warnings = graph_health_warnings(metrics, policy, require_command_buffer=policy.require_command_buffer or args.require_command_buffer)
        attribution = (
            analyze_xprof_trace_attribution(
                args.path,
                source_root=args.source_root,
                hlo_path=args.hlo_path,
                module=args.attribution_module,
                limit_per_class=args.attribution_limit,
            )
            if args.attribution
            else None
        )
    except (FileNotFoundError, GraphHealthArtifactError, OSError, ValueError) as exc:
        parser.error(str(exc))
    output: dict[str, Any] = {"metrics": metrics_as_dict(metrics), "warnings": warnings}
    if attribution is not None:
        output["attribution"] = attribution
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if args.fail_on_warning and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
