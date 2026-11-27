# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified artifact index and typed viewers for every file in a session directory."""

from __future__ import annotations

import difflib
import json
import os
from enum import Enum
from pathlib import Path

from hyperloom.inspector.collectors import resolve_inside
from hyperloom.inspector.graph import producer_for
from hyperloom.inspector.redact import redact, redact_text
from hyperloom.inspector.session_layout import PRIVATE_SESSION_SUBDIRS

_KIB = 1024
_MIB = _KIB * _KIB
MAX_FILES = 20000
MAX_JSON_BYTES = 16 * _MIB
MAX_TEXT_BYTES = 256 * _KIB
MAX_DIFF_BYTES = 2 * _MIB
BULK_MIN_FILES = 200
_SKIP_DIRS = {"__pycache__", ".git", "aiter_jit_backup", "node_modules"}


class Kind(str, Enum):
    """An artifact's ``type`` in the index and in a view payload."""

    TRACE = "trace"
    BINARY = "binary"
    JSONL = "jsonl"
    JSON = "json"
    LOG = "log"
    TEXT = "text"
    CONFIG = "config"
    OTHER = "other"


class Mode(str, Enum):
    """How :func:`view` renders an artifact: its payload's ``mode``."""

    JSON = "json"
    JSONL = "jsonl"
    TEXT = "text"
    META = "meta"


_OPAQUE_KINDS = (Kind.TRACE, Kind.BINARY)
_DEFAULT_MODE = {
    Kind.JSON: Mode.JSON,
    Kind.JSONL: Mode.JSONL,
    Kind.LOG: Mode.TEXT,
    Kind.TEXT: Mode.TEXT,
    Kind.CONFIG: Mode.TEXT,
    Kind.OTHER: Mode.TEXT,
}


def classify(name: str) -> Kind:
    lower = name.lower()
    if ".trace.json" in lower or lower.endswith(
        (".pt", ".gz", ".npz", ".bin", ".safetensors", ".db", ".db-wal", ".db-shm", ".sqlite")
    ):
        return Kind.TRACE if ".trace.json" in lower else Kind.BINARY
    if lower.endswith(".jsonl"):
        return Kind.JSONL
    if lower.endswith(".json"):
        return Kind.JSON
    if lower.endswith((".log", ".out", ".err")):
        return Kind.LOG
    if lower.endswith((".md", ".txt", ".rst")):
        return Kind.TEXT
    if lower.endswith((".yaml", ".yml", ".sh", ".toml", ".ini", ".cfg", ".patch", ".diff", ".py")):
        return Kind.CONFIG
    return Kind.OTHER


class ArtifactIndex(list):
    """The index's entries; ``truncated`` says the walk stopped at ``MAX_FILES`` with more files left to list."""

    truncated = False


def index(session_dir: Path, task_nodes: dict[str, str]) -> ArtifactIndex:
    """Every file below the session dir, up to ``MAX_FILES``, with type and producer; ``.env*``, the private
    subdirectories (``archive``) and links that lead out of the session or into one of those never listed."""
    out = ArtifactIndex()
    for dirpath, dirs, files in os.walk(session_dir):
        skip = _SKIP_DIRS.union(PRIVATE_SESSION_SUBDIRS) if Path(dirpath) == Path(session_dir) else _SKIP_DIRS
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith(".env"))
        for name in sorted(files):
            if name.startswith(".env"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(session_dir).as_posix()
            if path.is_symlink() and resolve_inside(session_dir, rel) is None:
                continue
            try:
                st = path.stat()
            except (FileNotFoundError, PermissionError):
                continue
            if len(out) >= MAX_FILES:
                out.truncated = True
                return _mark_bulk(out)
            producer, task_id = producer_for(rel, task_nodes)
            out.append(
                {
                    "path": rel,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "type": classify(name),
                    "producer": producer,
                    "task_id": task_id,
                }
            )
    return _mark_bulk(out)


def _mark_bulk(files: ArtifactIndex) -> ArtifactIndex:
    """Tag files below a large subtree (a source-tree copy, say) with ``bulk``: that subtree's root.

    The root is the outermost directory below the task dir (or below the first path segment for files that belong
    to no task) that holds more than ``BULK_MIN_FILES`` files, so a client can show it as one collapsed entry.
    """
    counts: dict[str, int] = {}
    for a in files:
        parts = a["path"].split("/")[:-1]
        for i in range(1, len(parts) + 1):
            key = "/".join(parts[:i])
            counts[key] = counts.get(key, 0) + 1
    for a in files:
        parts = a["path"].split("/")[:-1]
        first = 3 if a["task_id"] else 1  # runs/<kind>/<task>
        for i in range(first + 1, len(parts) + 1):
            key = "/".join(parts[:i])
            if counts[key] > BULK_MIN_FILES:
                a["bulk"] = key
                break
    return files


def _meta(path: Path, kind: Kind, rel: str) -> dict:
    st = path.stat()
    return {"path": rel, "type": kind, "size": st.st_size, "mtime": st.st_mtime}


def _read_text(path: Path, offset: int | None, max_bytes: int) -> tuple[str, int, int, bool]:
    size = path.stat().st_size
    start = max(0, size - max_bytes) if offset is None else min(max(0, offset), size)
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read(min(max_bytes, size - start))
    return _decode(data), start, start + len(data), start > 0 and offset is None


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def view(session_dir: Path, rel: str, mode: str | None = None, offset: int | None = None, limit: int = 500) -> dict:
    """Typed payload for one artifact. Raises ``PermissionError`` / ``FileNotFoundError``."""
    path = resolve_inside(session_dir, rel)
    if path is None:
        raise PermissionError(rel)
    if not path.is_file():
        raise FileNotFoundError(rel)
    kind = classify(path.name)
    mode = mode or _DEFAULT_MODE.get(kind, Mode.META)
    out = _meta(path, kind, rel)
    out["mode"] = mode
    if mode == Mode.META or kind in _OPAQUE_KINDS:
        out["mode"] = Mode.META
        return out
    if mode == Mode.JSON:
        if out["size"] > MAX_JSON_BYTES:
            out["mode"], out["note"] = Mode.TEXT, "too large to parse; showing the tail"
        else:
            try:
                out["data"] = redact(json.loads(_decode(path.read_bytes())))
                return out
            except json.JSONDecodeError as exc:
                out["mode"], out["note"] = Mode.TEXT, f"invalid JSON ({exc.msg}); showing text"
    if mode == Mode.JSONL:
        return {**out, **_jsonl_page(path, offset or 0, limit)}
    text, start, end, truncated = _read_text(path, offset, MAX_TEXT_BYTES)
    out.update({"mode": Mode.TEXT, "text": redact_text(text), "start": start, "end": end, "truncated": truncated})
    return out


def _jsonl_page(path: Path, offset: int, limit: int) -> dict:
    rows, total = [], 0
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            if offset <= total < offset + limit:
                try:
                    rows.append(redact(json.loads(line)))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    rows.append({"_unparsed": redact_text(_decode(line)[:2000])})
            total += 1
    return {"rows": rows, "offset": offset, "total": total}


def diff(session_dir: Path, rel_a: str, rel_b: str, context: int = 3) -> dict:
    """Side-by-side diff rows ``[tag, a_line_no, a_text, b_line_no, b_text]`` of two text artifacts."""
    texts = []
    for rel in (rel_a, rel_b):
        path = resolve_inside(session_dir, rel)
        if path is None:
            raise PermissionError(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        if classify(path.name) in _OPAQUE_KINDS or path.stat().st_size > MAX_DIFF_BYTES:
            raise ValueError(f"{rel}: not a diffable text file")
        texts.append(redact_text(_decode(path.read_bytes())).splitlines())
    a, b = texts
    rows = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for group in matcher.get_grouped_opcodes(context):
        rows.append(["hunk", None, "", None, ""])
        for tag, i1, i2, j1, j2 in group:
            for k in range(max(i2 - i1, j2 - j1)):
                ai, bj = i1 + k, j1 + k
                rows.append(
                    [
                        tag,
                        ai + 1 if ai < i2 else None,
                        a[ai] if ai < i2 else "",
                        bj + 1 if bj < j2 else None,
                        b[bj] if bj < j2 else "",
                    ]
                )
    return {"a": rel_a, "b": rel_b, "rows": rows, "identical": not rows}
