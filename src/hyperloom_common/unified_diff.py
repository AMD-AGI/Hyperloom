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


_GIT_HEADER_LINE = re.compile(r"(?m)^diff --git (.+)$")
_QUOTED_NAME = re.compile(r'"(?:[^"\\]|\\.)*"')
_QUOTED_PIECE = re.compile(r"\\([0-3][0-7]{2}|.?)|[^\\]+")
_C_ESCAPES = dict(zip('abtnvfr"\\', b'\a\b\t\n\v\f\r"\\'))
_HUNK_HEADER = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
# Old- and new-side lines each hunk body line uses up, by its first character; ``git apply`` reads an empty line
# as empty context.
_HUNK_LINE_COUNTS = {" ": (1, 1), "": (1, 1), "-": (1, 0), "+": (0, 1), "\\": (0, 0)}


def _unquote_git_path(raw: str) -> str:
    """Decode a C-quoted git path (``"b/caf\\303\\251.py"``) to its name; an unquoted path is returned unchanged.

    An octal escape is one byte of the name and a character git left unescaped (``core.quotePath=false``) stands
    for its UTF-8 bytes; the bytes are read back as UTF-8, so later ``..``/absolute checks see the real path.

    Raises:
        ValueError: The quoted path holds an escape git never writes, or its bytes are not UTF-8.
    """
    if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
        return raw
    name = bytearray()
    for piece in _QUOTED_PIECE.finditer(raw[1:-1]):
        escape = piece.group(1)
        if escape is None:
            name += piece.group().encode("utf-8")
        elif len(escape) == 3:
            name.append(int(escape, 8))
        elif escape in _C_ESCAPES:
            name.append(_C_ESCAPES[escape])
        else:
            raise ValueError(f"unknown escape {piece.group()!r} in quoted path {raw!r}")
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"quoted path {raw!r} is not valid UTF-8") from exc


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
    header = _GIT_HEADER_LINE.search(block)
    if not destination or not header:
        return "", ""
    line = header.group(1)
    at = line.rfind(destination.group(1).strip().strip('"'))
    cut = line.rfind(" ", 0, at) if at > 0 else -1
    if cut <= 0:
        return "", ""
    return _unquote_git_path(line[:cut]), _unquote_git_path(line[cut + 1 :])


def _same_path(old: str, new: str) -> bool:
    """Whether two unquoted ``diff --git`` names are one path: equal, or equal below their first ``/`` component."""
    return old == new or ("/" in old and "/" in new and old.partition("/")[2] == new.partition("/")[2])


def _git_header_new_name(block: str) -> str:
    """The new-side name on the section's ``diff --git`` line, still prefixed; empty when it has none to give.

    A name git had to quote starts with ``"``. Two unquoted names may hold spaces, so they are split the way
    ``git apply`` splits them: where both halves name the same path, whole (``--no-prefix``) or below their first
    ``/`` component (``a/`` and ``b/``, ``i/`` and ``w/``, ...).
    """
    header = _GIT_HEADER_LINE.search(block)
    if not header:
        return ""
    names = header.group(1)
    first = _QUOTED_NAME.match(names)
    if first:
        return _unquote_git_path(names[first.end() :].lstrip())
    quoted_second = names.find(' "')
    if quoted_second != -1:
        return _unquote_git_path(names[quoted_second + 1 :])
    for at, char in enumerate(names):
        if char == " " and _same_path(names[:at], names[at + 1 :]):
            return names[at + 1 :]
    return ""


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
        # A binary or empty file's deletion has no ``---`` line; git names the deleted path on both sides of its
        # ``diff --git`` line.
        source = minus or _git_header_new_name(block)
        if not source or source == _DEV_NULL:
            raise ValueError(f"deletion section missing source path:\n{block[:200]}")
        return [PatchTarget(PatchOp.DELETE, source)]

    # Header-only sections (binary, mode-only) carry no ``+++`` line and name their path on the ``diff --git`` line.
    target = plus or _git_header_new_name(block)
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


def _non_git_sections(patch_text: str) -> list[str]:
    """Split a diff that has no ``diff --git`` lines into per-file sections, each opened by a ``---`` line.

    A hunk's body is walked by the line counts in its header, so a body line that reads ``--- x`` (the removed
    line ``-- x``) stays in the hunk rather than opening a section.
    """
    sections: list[list[str]] = [[]]
    old_left = new_left = 0
    for line in patch_text.split("\n"):
        counts = _HUNK_LINE_COUNTS.get(line[:1]) if old_left > 0 or new_left > 0 else None
        if counts is not None:
            old_left, new_left = old_left - counts[0], new_left - counts[1]
        else:
            hunk = _HUNK_HEADER.match(line)
            old_left, new_left = (int(hunk[1] or 1), int(hunk[2] or 1)) if hunk else (0, 0)
            if line.startswith(_OLD_FILE):
                sections.append([])
        sections[-1].append(line)
    return ["\n".join(lines) for lines in sections]


def parse_patch_manifest(patch_text: str) -> list[PatchTarget]:
    """Read a unified diff as a manifest of intended filesystem changes, never as an instruction to replay.

    A rename yields a :attr:`PatchOp.DELETE` of the source and a :attr:`PatchOp.WRITE` of the destination.

    Raises:
        ValueError: The patch is empty, a section's target path cannot be determined, or a quoted path does not
            decode.
    """
    if not patch_text.strip():
        raise ValueError("empty patch: nothing to apply")

    if _GIT_HEADER_LINE.search(patch_text):
        blocks = re.split(r"(?m)^(?=diff --git )", patch_text)
    else:
        blocks = _non_git_sections(patch_text)
    return [target for block in blocks if block.strip() for target in _section_targets(block)]


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
