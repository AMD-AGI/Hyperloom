# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared stdlib-only helpers for the standalone kernel-agent tools.

Deduplicates run-status / log / JSON helpers and small source heuristics
copied across kernel_optimization.py, tracelens_analysis.py, and siblings.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to ``path`` via a temp file then rename, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


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
    "atomic_write_json",
    "kernel_row_matches",
    "read_last_lines",
    "source_text_looks_complete",
    "utc_now",
]
