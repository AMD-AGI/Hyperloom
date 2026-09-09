# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR KB slug helpers (consumer side)."""

from __future__ import annotations

import os
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

#: Default slug prefix root; must match the worker's ``PR_KB_SLUG_PREFIX``.
DEFAULT_PREFIX = "pr-kb"


def slug_prefix() -> str:
    """Return the active slug prefix root (``PR_KB_SLUG_PREFIX`` or default)."""
    return (os.environ.get("PR_KB_SLUG_PREFIX", "") or "").strip() or DEFAULT_PREFIX


def normalise_repo(repo: str) -> str:
    """Normalize a repo string to ``owner/name``."""
    r = (repo or "").strip()
    if not r:
        return ""
    if "://" in r:
        r = r.split("://", 1)[1]
        parts = r.split("/", 1)
        r = parts[1] if len(parts) == 2 else parts[0]
    if r.endswith(".git"):
        r = r[: -len(".git")]
    return r.strip("/")


def repo_slug(repo_full_name: str) -> str:
    """Convert ``owner/name`` into a gbrain-safe slug segment."""
    lowered = normalise_repo(repo_full_name).lower()
    return _NON_ALNUM.sub("-", lowered).strip("-")


def files_slug(repo_full_name: str, pr_number: int | str, *, prefix: str | None = None) -> str:
    """Return the ``<prefix>-files/<repo-slug>/pr/<n>`` slug."""
    p = prefix or slug_prefix()
    return f"{p}-files/{repo_slug(repo_full_name)}/pr/{pr_number}"


def index_slug(repo_full_name: str, *, prefix: str | None = None) -> str:
    """Return the ``<prefix>-index/<repo-slug>`` slug."""
    p = prefix or slug_prefix()
    return f"{p}-index/{repo_slug(repo_full_name)}"


__all__ = [
    "DEFAULT_PREFIX",
    "slug_prefix",
    "normalise_repo",
    "repo_slug",
    "files_slug",
    "index_slug",
]
