# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared error type for the recipe-snapshot KB remote clients.

Writes are local-only by design (:class:`recipe_kb.LocalRecipeStore` is the
source of truth); the read-side remote is the gbrain page store
(:class:`recipe_kb.gbrain_remote_client.GbrainRemoteRecipeClient`). That
client raises :class:`RemoteRecipeClientError` (via its ``GbrainRemoteError``
subclass) on any unrecoverable interaction so the :class:`recipe_kb.RecipeKB`
dispatcher can degrade to the local store with a single ``except``.
"""

from __future__ import annotations

from typing import Any, Mapping


# Errors
class RemoteRecipeClientError(RuntimeError):
    """Raised on any unrecoverable interaction with a remote recipe KB.

    The dispatcher catches this and degrades to the local store. Carries
    a ``category`` discriminator (``transport`` / ``business`` /
    ``validation`` / ``unknown``) so a future smarter dispatcher can
    decide between "retry with backoff" and "fall through immediately".
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = "unknown",
        code: str = "",
        status: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the error with a category discriminator and context.

        Args:
            message (str): Human-readable error description.
            category (str): Failure class — one of ``transport`` /
                ``business`` / ``validation`` / ``unknown``.
            code (str): Machine-readable error code from the server
                envelope, if any.
            status (int | None): HTTP status code, if a response was
                received.
            details (Mapping[str, Any] | None): Extra structured error
                context; copied into ``self.details``.
        """
        super().__init__(message)
        self.category = category
        self.code = code
        self.status = status
        self.details = dict(details or {})


__all__ = [
    "RemoteRecipeClientError",
]
