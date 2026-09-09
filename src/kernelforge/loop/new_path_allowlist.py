# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The allowlist that decides which agent-created files a KEEP may carry."""

from __future__ import annotations

from pathlib import PurePosixPath


class AllowlistPatternError(ValueError):
    """A ``commit_new_paths`` pattern this loop refuses to interpret."""


def normalize_commit_new_paths(patterns) -> list[str]:
    """Validate and canonicalize the new-file allowlist patterns."""
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
    """Whether a workspace-relative path is admitted by the allowlist."""
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
