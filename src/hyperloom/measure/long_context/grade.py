# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grading rule of the long-context check: an answer passes when it contains every reference, ignoring case."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["passes"]


def passes(text: str, references: Sequence[str]) -> bool:
    """Whether ``text`` holds each of ``references`` as a substring; no reference means nothing to find, which never passes."""
    lowered = text.lower()
    return bool(references) and all(reference.lower() in lowered for reference in references)
