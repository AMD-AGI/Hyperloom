# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared stdlib-only helpers for the kernel-agent tools."""

from __future__ import annotations

from pathlib import Path


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


__all__ = [
    "append_log",
    "read_last_lines",
    "write_text",
]
