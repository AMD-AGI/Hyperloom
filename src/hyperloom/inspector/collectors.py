# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cursor-based, read-only readers for the artefacts in a session directory.

Every reader tolerates a file that does not exist yet, a JSONL line that is
still being written, and a JSON file replaced mid-read. None of them writes:
SQLite is opened with ``mode=ro`` so a live Coordinator is never blocked.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess  # nosec B404 - runs amd-smi with a fixed argv only.
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hyperloom.inspector.session_layout import PRIVATE_SESSION_SUBDIRS

# A task's states with no way out, the orchestrator's TERMINAL_STATES: spelled here, as the Inspector runs without
# the orchestrator; a test keeps them equal.
ENDED_STATES = frozenset({"succeeded", "failed", "cancelled"})
_KIB = 1024
_MIB = _KIB * _KIB
_ENCODING = "utf-8"
_DECODE_ERRORS = "replace"
_ASSISTANT = "assistant"


def parse_ts(value: Any) -> float | None:
    """ISO-8601 string or unix number -> unix seconds; ``None`` when unparsable."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def read_json(path: Path) -> Any | None:
    """Read a JSON file written by atomic replace; retry once on a torn read."""
    for attempt in range(2):
        try:
            with path.open("r", encoding=_ENCODING) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            if attempt == 0:
                time.sleep(0.05)
    return None


@dataclass
class JsonlTail:
    """Yields only complete, new JSONL records; restarts if the file shrinks."""

    path: Path
    offset: int = 0

    def read_new(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read(size - self.offset)
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        out = []
        for line in chunk[: end + 1].splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out


@dataclass
class ChangedFiles:
    """Tracks a glob of JSON files and returns the ones added or modified since the last call."""

    root: Path
    pattern: str
    seen: dict[str, float] = field(default_factory=dict)

    def read_changed(self) -> list[tuple[Path, Any]]:
        out = []
        for path in sorted(self.root.glob(self.pattern)):
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            key = str(path)
            if self.seen.get(key) == mtime:
                continue
            data = read_json(path)
            if data is None:
                continue
            self.seen[key] = mtime
            out.append((path, data))
        return out


def _loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


class CoordinatorDb:
    """Read-only view of a session's coordinator database; each call uses a short-lived connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.event_seq = 0

    def _connect(self) -> sqlite3.Connection | None:
        if not self.path.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=2.0)
        except sqlite3.Error:
            return None
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            return conn.execute(sql, args).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def new_events(self, limit: int = 2000) -> list[dict]:
        rows = self._query(
            "SELECT seq, msg_id, from_agent, to_agent, topic, in_reply_to, payload, ts "
            "FROM events WHERE seq > ? ORDER BY seq LIMIT ?",
            (self.event_seq, limit),
        )
        out = []
        for r in rows:
            self.event_seq = max(self.event_seq, r["seq"])
            rec = dict(r)
            rec["payload"] = _loads(rec["payload"], rec["payload"])
            out.append(rec)
        return out

    def tasks(self) -> list[dict]:
        rows = self._query(
            "SELECT task_id, kind, state, params, requires_lanes, lease_ttl_sec, history, created_at, updated_at "
            "FROM tasks ORDER BY created_at"
        )
        out = []
        for r in rows:
            rec = dict(r)
            rec["params"] = _loads(rec["params"], {})
            rec["requires_lanes"] = _loads(rec["requires_lanes"], [])
            rec["history"] = _loads(rec["history"], [])
            out.append(rec)
        return out


def resolve_inside(session_dir: Path, rel: str) -> Path | None:
    """Resolve *rel* against the session dir; ``None`` if it escapes it, lands in a private subdirectory (``archive``)
    or names a ``.env`` file. Links are followed first, so none leads out of it or into a private one."""
    root = session_dir.resolve()
    try:
        target = (root / rel).resolve()
    except (OSError, RuntimeError):
        return None
    if root != target and root not in target.parents:
        return None
    if target.name.startswith(".env"):
        return None
    if target != root and target.relative_to(root).parts[0] in PRIVATE_SESSION_SUBDIRS:
        return None
    return target


def tail_file(path: Path, max_bytes: int = 64 * _KIB) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return ""
    text = data.decode(_ENCODING, errors=_DECODE_ERRORS)
    if size > max_bytes:
        text = text.split("\n", 1)[-1]
    return text


_EXCEPTION = re.compile(r"\b[A-Z][\w.]*(?:Error|Exception): \S.*")


def first_exception(path: Path, max_bytes: int = 8 * _MIB) -> str | None:
    """The first ``SomeError: message`` in a log, without its logger prefix; in a crashed vLLM/SGLang server log
    that is the root cause, and the later ones re-raise it wrapped."""
    read = 0
    try:
        with path.open("r", encoding=_ENCODING, errors=_DECODE_ERRORS) as f:
            for line in f:
                read += len(line)
                if read > max_bytes:
                    return None
                m = _EXCEPTION.search(line)
                if m:
                    return m.group(0).strip()[:400]
    except OSError:
        return None
    return None


_TURN_TEXT_CAP = 20_000
_TOOL_RESULT_CAP = 4_000


class _Turn:
    """One specialist turn as its log is read: the assistant message's blocks and the tool results after it."""

    def __init__(self, turn: int, msg: dict) -> None:
        self.record = {
            "turn": turn,
            "model": msg.get("model"),
            "usage": msg.get("usage"),
            "response": [],
            "tool_results": [],
        }

    def add(self, kind: str, msg: dict) -> None:
        blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
        if kind == _ASSISTANT:
            self.record["usage"] = msg.get("usage") or self.record["usage"]
            self.record["response"].extend(_cut_block(b, _TURN_TEXT_CAP) for b in blocks if isinstance(b, dict))
        else:
            self.record["tool_results"].extend(_cut_block(b, _TOOL_RESULT_CAP) for b in blocks if isinstance(b, dict))


def specialist_turn(process_log: Path, turn: int, max_bytes: int = 32 * _MIB) -> dict | None:
    """Turn *turn* (1-based) of a specialist's stream-JSON ``process.log``; ``None`` when it has no such turn.

    A turn is one assistant message (its content arrives over several lines sharing a message id) plus the tool
    results that follow it. Long text and tool results are cut.
    """
    seen: list[str] = []
    out: _Turn | None = None
    try:
        if process_log.stat().st_size > max_bytes:
            return None
        fh = process_log.open("r", encoding=_ENCODING, errors=_DECODE_ERRORS)
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            kind = rec.get("type") if isinstance(rec, dict) else None
            msg = rec.get("message") if kind in (_ASSISTANT, "user") else None
            if not isinstance(msg, dict):
                continue
            if kind == _ASSISTANT and msg.get("id") not in seen:
                seen.append(msg.get("id"))
                if len(seen) > turn:
                    break
                if len(seen) == turn:
                    out = _Turn(turn, msg)
            if out is None or len(seen) != turn:
                continue
            out.add(kind, msg)
    return out.record if out is not None else None


def _cut_block(block: dict, cap: int) -> dict:
    out = {}
    for key, value in block.items():
        if key == "signature":
            continue
        if not isinstance(value, str):
            value = json.dumps(value) if key == "content" else value
        if isinstance(value, str) and len(value) > cap:
            value = value[:cap] + f"… [{len(value) - cap} more chars]"
        out[key] = value
    return out


def newest_mtime(path: Path) -> float | None:
    """Newest mtime of *path* or anything below it (bounded walk)."""
    if not path.exists():
        return None
    newest = path.stat().st_mtime
    seen = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            seen += 1
            if seen > 2000:
                return newest
            try:
                newest = max(newest, (Path(dirpath) / name).stat().st_mtime)
            except FileNotFoundError:
                continue
    return newest


def container_memory() -> dict:
    """cgroup v2 (or v1) memory usage and limit of this container, in bytes."""
    candidates = [
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
    ]
    for cur_p, max_p in candidates:
        try:
            current = int(cur_p.read_text().strip())
            raw_max = max_p.read_text().strip()
        except (OSError, ValueError):
            continue
        limit = None if raw_max == "max" else int(raw_max)
        if limit is not None and limit > 1 << 60:
            limit = None
        return {"used_bytes": current, "limit_bytes": limit, "source": str(cur_p.parent)}
    return {}


def _metric_value(obj: Any) -> float | None:
    if isinstance(obj, dict):
        obj = obj.get("value")
    return float(obj) if isinstance(obj, (int, float)) else None


def gpu_metrics(timeout: float = 8.0) -> list[dict]:
    """Per-GPU utilisation and VRAM from ``amd-smi metric --json``; empty when unavailable."""
    exe = shutil.which("amd-smi") or ("/opt/rocm/bin/amd-smi" if Path("/opt/rocm/bin/amd-smi").exists() else None)
    if not exe:
        return []
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no caller input.
            [exe, "metric", "-u", "-m", "--json"], capture_output=True, text=True, timeout=timeout, check=False
        )
        data = json.loads(proc.stdout)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return []
    gpus = data.get("gpu_data", data) if isinstance(data, dict) else data
    out = []
    for g in gpus if isinstance(gpus, list) else []:
        usage, mem = g.get("usage") or {}, g.get("mem_usage") or {}
        out.append(
            {
                "gpu": g.get("gpu"),
                "gfx_pct": _metric_value(usage.get("gfx_activity")),
                "umc_pct": _metric_value(usage.get("umc_activity")),
                "vram_used_mb": _metric_value(mem.get("used_vram")),
                "vram_total_mb": _metric_value(mem.get("total_vram")),
            }
        )
    return out
