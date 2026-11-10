"""Conflict error for writes that would mutate an immutable Experience."""

from __future__ import annotations


class ExperienceConflictError(ValueError):
    """Raised when a write would mutate immutable Experience facts."""


__all__ = [
    "ExperienceConflictError",
]
