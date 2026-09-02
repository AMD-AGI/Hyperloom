# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared stdlib-only helpers for the standalone kernel-agent tools.

Deduplicates run-status, log, JSON, and source helpers used by the standalone
kernel tools.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now(*, timespec: str | None = None) -> str:
    """Return the current UTC time as an ISO-8601 string."""
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec=timespec) if timespec else now.isoformat()


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    trailing_newline: bool = True,
) -> None:
    """Write JSON to ``path`` via a temp file then rename, creating parents.

    The temp file is opened UTF-8 explicitly, like every other writer here: with
    ``ensure_ascii=False`` callers the payload carries non-ASCII, which a
    locale-derived default encoding cannot always represent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
        if trailing_newline:
            tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    """Write text to ``path`` using UTF-8, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(log_path: Path, message: str) -> None:
    """Append one line to ``log_path`` (rstripped + newline), creating parents."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def append_jsonl(path: Path, row: Any, *, sort_keys: bool = True, ensure_ascii: bool = True) -> None:
    """Append one JSON value as a JSONL row, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n")


def read_json(path: str | Path | None, default: Any = None, *, require_dict: bool = False) -> Any:
    """Parse JSON from ``path``; return ``default`` on missing/malformed input.

    Kernel-local, stdlib-only mirror of ``common.jsonio.read_json`` tolerant
    mode: a falsy path, an ``OSError`` / ``JSONDecodeError``, or — under
    ``require_dict`` — a non-object payload all yield ``default``.
    """
    if not path:
        return default
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if require_dict and not isinstance(data, dict):
        return default
    return data


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    """Return the last ``limit`` lines of ``log_path``, empty when missing."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def kernel_row_matches(row: dict[str, Any], target_kernel: str) -> bool:
    """Return whether a result row matches ``target_kernel`` (empty matches any)."""
    if not target_kernel:
        return True
    target = target_kernel.strip()
    names = (
        str(row.get("matched_kernel_name") or "").strip(),
        str(row.get("name") or "").strip(),
    )
    return any(name == target for name in names)


def safe_float(
    value: Any,
    default: float | None = 0.0,
    *,
    strip_percent: bool = False,
    strip_commas: bool = False,
) -> float | None:
    """Coerce int/float/numeric-str to float; None/empty/malformed -> default."""
    if isinstance(value, bool):
        return default
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            text = value.strip()
            if strip_percent:
                text = text.rstrip("%")
            if strip_commas:
                text = text.replace(",", "")
            if not text:
                return default
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_last_json(text: str) -> dict[str, Any] | None:
    """Return the last top-level JSON object embedded in ``text``.

    The scanner respects quoted strings and escapes, so braces inside JSON
    string values do not disturb depth tracking.
    """
    if not text:
        return None
    start: int | None = None
    depth = 0
    in_str = False
    esc = False
    found: dict[str, Any] | None = None
    for idx, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    data = json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(data, dict):
                    found = data
                start = None
    return found


def truthy(val: Any) -> bool:
    """Interpret common truthy spellings from JSON or env strings.

    A real ``bool`` is returned as-is; any other value is stringified,
    stripped, lower-cased, and matched against ``1/true/yes/on``.
    """
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


_COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}


def source_text_looks_complete(text: str, suffix: str) -> bool:
    """Heuristically decide whether ``text`` is a complete source file.

    Python must compile and carry a top-level marker; compiled sources must
    carry a C/C++/HIP marker. Fenced text is rejected.
    """
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(marker in stripped for marker in ("def ", "class ", "import ", "@triton.jit", "torch."))
    if suffix in _COMPILED_SOURCE_SUFFIXES:
        return any(
            marker in stripped
            for marker in (
                "#include",
                "__global__",
                "__device__",
                "extern ",
                "namespace ",
                "template",
                "void ",
                "int ",
                "float ",
                "half",
                "torch::",
            )
        )
    return False


__all__ = [
    "append_log",
    "append_jsonl",
    "atomic_write_json",
    "extract_last_json",
    "kernel_row_matches",
    "read_json",
    "read_last_lines",
    "safe_float",
    "source_text_looks_complete",
    "truthy",
    "utc_now",
    "write_text",
]
