# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified-diff parsing (stdlib-only, no framework-agent imports)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum

_DEV_NULL = "/dev/null"
_OLD_FILE = "--- "
_NEW_FILE = "+++ "


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

    A ``diff --git`` header seeds the section's path, so binary, mode-only and
    pure-rename sections, which carry no ``---``/``+++`` pair, still name their file.
    """
    changes: list[FileChange] = []
    current: FileChange | None = None
    for raw in (patch_text or "").splitlines():
        if raw.startswith("diff --git"):
            header = raw.split()
            current = FileChange(path=strip_diff_path(header[3]) if len(header) >= 4 else "")
            changes.append(current)
            continue
        if current is None:
            # Tolerate a diff with no leading "diff --git".
            if not raw.startswith((_OLD_FILE, _NEW_FILE)):
                continue
            current = FileChange(path="")
            changes.append(current)
        if raw.startswith("new file mode"):
            current.is_new = True
            continue
        if raw.startswith("deleted file mode"):
            current.is_deleted = True
            continue
        if raw.startswith(_NEW_FILE):
            new_path = strip_diff_path(raw[4:])
            if new_path == _DEV_NULL:
                current.is_deleted = True
            else:
                current.path = new_path
            continue
        if raw.startswith(_OLD_FILE):
            old_path = strip_diff_path(raw[4:])
            if old_path and old_path != _DEV_NULL and not current.path:
                current.path = old_path
            continue
        lines = {"+": current.added, "-": current.removed, " ": current.context}.get(raw[:1])
        if lines is not None and not raw.startswith(("+++", "---")):
            lines.append(raw[1:])
    # Drop empty/placeholder sections.
    return [c for c in changes if c.path and c.path != _DEV_NULL]


def touched_paths(patch_text: str) -> list[str]:
    """Return the distinct paths a unified diff touches, in diff order (a deletion by its old path)."""
    return list(dict.fromkeys(change.path for change in parse_unified_diff(patch_text)))


def strip_diff_path(token: str) -> str:
    """Normalize a diff path token (strip ``a/``/``b/`` prefix + tab suffix)."""
    token = token.strip().split("\t", 1)[0].strip()
    if token in (_DEV_NULL, ""):
        return token
    if token.startswith(("a/", "b/")):
        return token[2:]
    return token


def strip_path_components(path: str, level: int) -> str:
    """Drop ``level`` leading components like ``git apply -p<level>``, keeping at least the basename."""
    if level <= 0:
        return path
    parts = path.split("/")
    return "/".join(parts[level:]) if len(parts) > level else parts[-1]


class PatchOp(str, Enum):
    """What a patch does to one path."""

    WRITE = "write"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class PatchTarget:
    """One filesystem change a unified diff intends.

    ``path`` is the header path, still ``a/``/``b/``-prefixed; strip it with :func:`strip_diff_path` or
    :func:`strip_path_components`. ``is_new`` marks a path the patch creates, so a caller staging a base
    content must leave it absent.
    """

    op: PatchOp
    path: str
    mode: str = ""
    is_new: bool = False


def _unquote_git_path(raw: str) -> str:
    """Decode a C-quoted git header path (``"b/\\303\\251.py"``), so later ``..``/absolute checks see the real path."""
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
    return raw


def _header_path(block: str, marker: str) -> str:
    """The path following ``marker`` on its own line, unquoted and without a trailing timestamp."""
    found = re.search(rf"(?m)^{re.escape(marker)}(.+)$", block)
    return _unquote_git_path(found.group(1).split("\t", 1)[0].strip()) if found else ""


def _moved_header_paths(block: str, verb: str) -> tuple[str, str]:
    """The ``diff --git`` token pair of a ``rename``/``copy`` section, located by its own from/to names.

    The header prefix is whatever the producer's gitconfig chose (``a/``/``b/``, none, ``c/``/``w/``),
    so the names come from the ``<verb> from``/``<verb> to`` lines and the header supplies the prefix
    each one carries.
    """
    destination = re.search(rf"(?m)^{verb} to (.+)$", block)
    header = re.search(r"(?m)^diff --git (.+)$", block)
    if not destination or not header:
        return "", ""
    line = header.group(1)
    at = line.rfind(destination.group(1).strip().strip('"'))
    cut = line.rfind(" ", 0, at) if at > 0 else -1
    if cut <= 0:
        return "", ""
    return _unquote_git_path(line[:cut]), _unquote_git_path(line[cut + 1 :])


def _section_targets(block: str) -> list[PatchTarget]:
    """The targets of one diff section, covering the whole git header vocabulary."""
    new_file_m = re.search(r"(?m)^new file mode (\d+)$", block)
    mode_m = new_file_m or re.search(r"(?m)^new mode (\d+)$", block)
    mode = mode_m.group(1)[-4:] if mode_m else ""

    old_path, new_path = _moved_header_paths(block, "rename")
    if old_path and new_path:
        return [
            PatchTarget(PatchOp.DELETE, old_path),
            PatchTarget(PatchOp.WRITE, new_path, mode=mode, is_new=True),
        ]
    copied = _moved_header_paths(block, "copy")[1]
    if copied:
        return [PatchTarget(PatchOp.WRITE, copied, mode=mode, is_new=True)]

    minus = _header_path(block, _OLD_FILE)
    plus = _header_path(block, _NEW_FILE)
    if plus == _DEV_NULL or re.search(r"(?m)^deleted file mode", block):
        if not minus or minus == _DEV_NULL:
            raise ValueError(f"deletion section missing source path:\n{block[:200]}")
        return [PatchTarget(PatchOp.DELETE, minus)]

    # Header-only sections (pure chmod) carry no ``+++`` line and name their path on the ``diff --git`` line.
    header = re.search(r"(?m)^diff --git a/.+? (b/.+)$", block)
    target = plus or (_unquote_git_path(header.group(1).strip()) if header else "")
    if not target:
        raise ValueError(f"cannot determine target path for section:\n{block[:200]}")
    return [
        PatchTarget(
            PatchOp.WRITE,
            target,
            mode=mode,
            is_new=minus == _DEV_NULL or bool(new_file_m),
        )
    ]


def parse_patch_manifest(patch_text: str) -> list[PatchTarget]:
    """Read a unified diff as a manifest of intended filesystem changes, never as an instruction to replay.

    A rename yields a :attr:`PatchOp.DELETE` of the source and a :attr:`PatchOp.WRITE` of the destination.

    Raises:
        ValueError: The patch is empty, or a section's target path cannot be determined.
    """
    if not patch_text.strip():
        raise ValueError("empty patch: nothing to apply")

    marker = r"diff --git " if re.search(r"(?m)^diff --git ", patch_text) else re.escape(_OLD_FILE)
    blocks = [b for b in re.split(rf"(?m)^(?={marker})", patch_text) if b.strip()]
    return [target for block in blocks for target in _section_targets(block)]


def repo_relative_manifest(patch_text: str) -> list[PatchTarget]:
    """:func:`parse_patch_manifest` for a caller that works in repo-relative paths rather than at a strip level."""
    return [replace(target, path=strip_diff_path(target.path)) for target in parse_patch_manifest(patch_text)]


__all__ = [
    "FileChange",
    "PatchOp",
    "PatchTarget",
    "parse_patch_manifest",
    "parse_unified_diff",
    "repo_relative_manifest",
    "strip_diff_path",
    "strip_path_components",
    "touched_paths",
]
