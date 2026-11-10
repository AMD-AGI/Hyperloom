# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declared deliverable a specialist round hands back."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _clean_seq(values: Any) -> tuple[str, ...]:
    """Return a de-duplicated tuple of non-empty strings, order preserved."""
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def declared_targets(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the tree-relative paths a ``specialist_done`` payload declares it touched.

    The nested ``deliverable`` mapping is authoritative; a payload without one
    is read flat.
    """
    if not payload:
        return ()
    declared = payload.get("deliverable")
    source: Mapping[str, Any] = declared if isinstance(declared, Mapping) else payload
    return _clean_seq(source.get("targets"))


__all__ = ["declared_targets"]
