# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Normalization shared by KernelForge knowledge-record identities."""

from __future__ import annotations

import re
from importlib import metadata

from kernelforge.knowledge.kb_store.identity.implementation import canonical_framework_version

UNKNOWN_SEGMENT = "unknown"

_DISALLOWED = re.compile(r"[^a-z0-9._+-]+")
_LEADING = re.compile(r"^[^a-z0-9_]+")


def segment(value: str, *, fallback: str) -> str:
    """Fold a free-form value into one identity dimension."""
    folded = _DISALLOWED.sub("-", str(value or "").strip().lower())
    folded = _LEADING.sub("", folded).strip("-")
    if not folded:
        folded = fallback
    return folded.encode("ascii", "ignore").decode("ascii")[:256] or fallback


def framework_version(framework: str) -> str:
    """Return a canonical installed-framework version for a record identity."""
    name = str(framework or "").strip().lower()
    try:
        installed = metadata.version(name) if name and name != UNKNOWN_SEGMENT else ""
    except metadata.PackageNotFoundError:
        installed = ""
    return segment(canonical_framework_version(installed), fallback=UNKNOWN_SEGMENT)
