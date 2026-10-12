# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared stdlib-only helpers for the kernel-agent tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_text(path: Path, text: str) -> None:
    """Write text to ``path`` using UTF-8, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(log_path: Path, message: str) -> None:
    """Append one line to ``log_path`` (rstripped + newline), creating parents."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def read_last_lines(log_path: Path, limit: int = 20) -> list[str]:
    """Return the last ``limit`` lines of ``log_path``, empty when missing."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


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


__all__ = [
    "append_log",
    "read_last_lines",
    "safe_float",
    "write_text",
]
