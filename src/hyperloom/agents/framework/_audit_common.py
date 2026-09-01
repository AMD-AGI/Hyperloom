# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified-diff parsing, kept free of framework-agent imports.

The orchestrator reads the file list out of a candidate diff through this
module, so it must stay importable without pulling the agent runtime in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileChange:
    """One file's hunks parsed out of a unified diff."""

    path: str
    is_new: bool = False
    is_deleted: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


def parse_unified_diff(patch_text: str) -> list[FileChange]:
    """Parse a unified diff into per-file added/removed/context line groups.

    Args:
        patch_text: The unified diff (``diff --git`` / ``--- a/`` / ``+++ b/``).

    Returns:
        One :class:`FileChange` per file section, in first-seen order.
    """
    changes: list[FileChange] = []
    current: FileChange | None = None
    for raw in (patch_text or "").splitlines():
        if raw.startswith("diff --git"):
            # New file section; the +++ line below sets the canonical path.
            current = FileChange(path="")
            changes.append(current)
            continue
        if current is None:
            # Tolerate a diff with no leading "diff --git".
            if raw.startswith("--- ") or raw.startswith("+++ "):
                current = FileChange(path="")
                changes.append(current)
            else:
                continue
        if raw.startswith("new file mode"):
            current.is_new = True
            continue
        if raw.startswith("deleted file mode"):
            current.is_deleted = True
            continue
        if raw.startswith("+++ "):
            new_path = _strip_diff_path(raw[4:].strip())
            if new_path == "/dev/null":
                current.is_deleted = True
            else:
                current.path = new_path
            continue
        if raw.startswith("--- "):
            old_path = _strip_diff_path(raw[4:].strip())
            if old_path and old_path != "/dev/null" and not current.path:
                current.path = old_path
            continue
        if raw.startswith("@@"):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            current.added.append(raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            current.removed.append(raw[1:])
        elif raw.startswith(" "):
            current.context.append(raw[1:])
    # Drop empty/placeholder sections.
    return [c for c in changes if c.path and c.path != "/dev/null"]


def _strip_diff_path(token: str) -> str:
    """Normalize a diff path token (strip ``a/``/``b/`` prefix + tab suffix)."""
    token = token.split("\t", 1)[0].strip()
    if token in ("/dev/null", ""):
        return token
    if token.startswith(("a/", "b/")):
        return token[2:]
    return token
