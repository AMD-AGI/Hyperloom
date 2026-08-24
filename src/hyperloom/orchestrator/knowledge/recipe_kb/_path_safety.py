# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared path-safety helpers for the recipe_kb package.

Extracted from :mod:`local_graph_store` so both the graph store and the recipe
store can validate path components against the same closed character set and
containment invariant without duplicating the logic.
"""

from __future__ import annotations

import re
from pathlib import Path


SLUG_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def validated_slug(value: object) -> str:
    """Return *value* after confirming it is a safe filesystem path slug.

    A safe slug is a non-empty string whose ``/``-separated parts each match
    :data:`SLUG_PART_RE` and are not the special names ``""``, ``"."``, or
    ``".."``.  Leading/trailing ``/``, backslashes, and null bytes are also
    rejected.

    Args:
        value: Candidate slug.

    Returns:
        The validated slug (same object, as a ``str``).

    Raises:
        ValueError: If the slug fails any safety check.
    """
    if not isinstance(value, str):
        raise ValueError("slug must be a string")
    slug = value.strip()
    if not slug or len(slug) > 512:
        raise ValueError("slug must be non-empty and at most 512 characters")
    if slug.startswith("/") or slug.endswith("/") or "\\" in slug or "\x00" in slug:
        raise ValueError(f"unsafe slug: {value!r}")
    parts = slug.split("/")
    if any(part in ("", ".", "..") or not SLUG_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(f"unsafe slug: {value!r}")
    return slug


def assert_within_root(path: Path, root: Path) -> None:
    """Raise ``ValueError`` when *path* resolves outside *root*.

    Args:
        path: The candidate path (need not exist).
        root: The expected ancestor directory.

    Raises:
        ValueError: If the resolved *path* is not under *root*.
    """
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path {path!r} escapes root {root!r}") from exc


__all__ = [
    "SLUG_PART_RE",
    "assert_within_root",
    "validated_slug",
]
