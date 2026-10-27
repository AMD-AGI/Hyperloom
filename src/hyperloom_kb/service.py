"""Errors of the canonical Experience submission path."""

from __future__ import annotations

from hyperloom_kb.storage import StorageContractError


class CompleteExperienceRequired(StorageContractError):
    """Raised when immutable submission receives a non-terminal Experience."""


class UnknownSchemaRef(StorageContractError):
    """Raised when a request names, or an Experience references, a schema the service has not registered."""


__all__ = [
    "CompleteExperienceRequired",
    "UnknownSchemaRef",
]
