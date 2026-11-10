"""Errors of the canonical Experience submission path."""

from __future__ import annotations

from hyperloom_kb.storage import StorageContractError


class CompleteExperienceRequired(StorageContractError):
    """Raised when immutable submission receives a non-terminal Experience."""


__all__ = [
    "CompleteExperienceRequired",
]
