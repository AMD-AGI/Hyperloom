# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The allowlist that decides which agent-created files a KEEP may carry.

One module because two independent readers have to agree on it: the campaign
configuration validates and stores the patterns, and the loop matches workspace
paths against them to decide what is committed and what a REVERT deletes. A
pattern that meant one thing at configuration time and another at deletion time
would delete a file nobody allowlisted.
"""

from __future__ import annotations

from pathlib import PurePosixPath


class AllowlistPatternError(ValueError):
    """A ``commit_new_paths`` pattern this loop refuses to interpret."""


def normalize_commit_new_paths(patterns) -> list[str]:
    """Validate and canonicalize the new-file allowlist patterns.

    Entries are workspace-relative POSIX paths or single-segment globs
    (``configs/*.json``). Blank entries are dropped, duplicates collapse, and
    order is preserved.

    ``**`` is NOT supported and is rejected rather than accepted and silently
    treated as a single ``*``: the allowlist decides both what a KEEP commits
    and what a REVERT deletes, so a pattern whose reach the operator and the
    loop disagree about is the one failure this whole path exists to prevent.
    Absolute paths and ``..`` are rejected for the same reason -- an allowlist
    only ever names something inside the workspace.
    """
    normalized: list[str] = []
    for raw in patterns or []:
        pattern = str(raw).strip()
        if not pattern:
            continue
        if "**" in pattern:
            raise AllowlistPatternError(
                "commit_new_paths does not support '**' (a '*' never crosses a "
                f"directory separator); name each directory level: {pattern}"
            )
        candidate = PurePosixPath(pattern)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AllowlistPatternError(
                f"commit_new_paths entries must be workspace-relative paths without '..': {pattern}"
            )
        posix = candidate.as_posix()
        if posix not in normalized:
            normalized.append(posix)
    return normalized


def matches_commit_new_paths(path: str, patterns) -> bool:
    """Whether a workspace-relative path is admitted by the allowlist.

    Matching is anchored glob, not :func:`fnmatch.fnmatch`: ``fnmatch`` treats
    ``/`` as an ordinary character, so ``configs/*.json`` would also admit
    ``configs/generated/tmp.json`` and ``*.py`` would admit every ``.py`` file
    at any depth. Both sides are rooted at ``/`` so :meth:`PurePosixPath.match`
    compares whole paths instead of matching the pattern against the tail.

    Patterns must already have been through :func:`normalize_commit_new_paths`.
    An entry this function cannot interpret raises rather than being skipped:
    a silent skip is a pattern the operator wrote and the loop ignored, which
    is the disagreement this module exists to prevent, and it would show up as
    a file that was never committed or never removed with nothing said.
    """
    target = PurePosixPath("/") / PurePosixPath(path)
    for pattern in patterns or []:
        text = str(pattern).strip()
        if not text or "**" in text:
            raise AllowlistPatternError(
                f"commit_new_paths reached matching unnormalized: {pattern!r}; normalize_commit_new_paths first"
            )
        if target.match(str(PurePosixPath("/") / PurePosixPath(text))):
            return True
    return False
