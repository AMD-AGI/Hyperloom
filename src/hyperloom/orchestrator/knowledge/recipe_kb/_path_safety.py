# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path-component validation shared by the recipe_kb stores."""

from __future__ import annotations

import re
from pathlib import Path


SLUG_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def validated_slug(value: object) -> str:
    """Return *value* if every ``/``-separated part is a safe path component.

    Raises:
        ValueError: If the slug is not a string, is empty or over 512
            characters, or holds a part that is not matched by
            :data:`SLUG_PART_RE`.
    """
    if not isinstance(value, str):
        raise ValueError("slug must be a string")
    slug = value.strip()
    if not slug or len(slug) > 512:
        raise ValueError("slug must be non-empty and at most 512 characters")
    if slug.startswith("/") or slug.endswith("/") or "\\" in slug or "\x00" in slug:
        raise ValueError(f"unsafe slug: {value!r}")
    if any(not SLUG_PART_RE.fullmatch(part) for part in slug.split("/")):
        raise ValueError(f"unsafe slug: {value!r}")
    return slug


def is_within_root(path: Path, root: Path) -> bool:
    """Return whether *path* resolves inside *root*, following symlinks."""
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        return False
    return True


__all__ = [
    "SLUG_PART_RE",
    "is_within_root",
    "validated_slug",
]
