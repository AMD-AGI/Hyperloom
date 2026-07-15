# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local-first recipe-snapshot KB for the inference optimizer.

* Writes go LOCAL ONLY — never to the central kb-service. The store
  on disk is the single source of truth in degraded / offline mode
  and the authoritative source in healthy mode (the central service
  becomes a read-side cache, not a write target).
* Reads are dispatched at a higher layer (``recipe_kb.dispatcher.RecipeKB``);
  the local store implemented here is the read fallback.

The on-disk layout maps the canonical id
``inference:{model}:{hardware}:{framework}:{framework_version}:{precision}``
to a directory tree under :data:`LocalRecipeStore.root`. Each leaf directory
holds ``recipe.json`` (live), ``history/v{N}.json`` (archived prior
versions), ``attempts.ndjson`` (append-only attempts log), and ``.lock``
(flock target).

The wire shapes (``Recipe`` / ``Attempt`` dataclasses) mirror the central
kb-service v2 contract, so a dispatcher consumer sees identical dicts whether
they come from the local store or a central GET.
"""

from __future__ import annotations

from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Shared error type for the recipe-snapshot KB remote clients.
# ---------------------------------------------------------------------------
# Writes are local-only by design (:class:`LocalRecipeStore` is the source of
# truth); the read-side remote is the gbrain page store
# (:class:`gbrain_remote_client.GbrainRemoteRecipeClient`). That client raises
# :class:`RemoteRecipeClientError` (via its ``GbrainRemoteError`` subclass) on
# any unrecoverable interaction so the :class:`RecipeKB` dispatcher can degrade
# to the local store with a single ``except``.
#
# Defined here (before the relative imports below) because ``dispatcher`` and
# ``gbrain_remote_client`` import this class back from the package root; putting
# it above ``from .dispatcher import RecipeKB`` keeps that partial-init cycle
# safe.
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


from .canonical_id import (  # noqa: E402
    CANONICAL_ID_DIMENSIONS,
    CANONICAL_ID_PREFIX,
    DEFAULT_FRAMEWORK_SLUG,
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    DEFAULT_HARDWARE_SLUG,
    DEFAULT_MODEL_SLUG,
    DEFAULT_PRECISION_SLUG,
    InvalidCanonicalIdError,
    canonical_id_for_path,
    canonical_id_from_components,
    canonical_labels,
    cid_to_path_components,
    detect_framework_version,
    recipe_canonical_id,
)
from .dispatcher import RecipeKB  # noqa: E402
from .local_store import (  # noqa: E402
    ATTEMPTS_FILENAME,
    HISTORY_DIRNAME,
    LOCK_FILENAME,
    LocalRecipeStore,
    LocalRecipeStoreError,
    RECIPE_FILENAME,
)
from .schema import Attempt, Recipe  # noqa: E402


__all__ = [
    "ATTEMPTS_FILENAME",
    "Attempt",
    "CANONICAL_ID_DIMENSIONS",
    "CANONICAL_ID_PREFIX",
    "DEFAULT_FRAMEWORK_SLUG",
    "DEFAULT_FRAMEWORK_VERSION_SLUG",
    "DEFAULT_HARDWARE_SLUG",
    "DEFAULT_MODEL_SLUG",
    "DEFAULT_PRECISION_SLUG",
    "HISTORY_DIRNAME",
    "InvalidCanonicalIdError",
    "LOCK_FILENAME",
    "LocalRecipeStore",
    "LocalRecipeStoreError",
    "RECIPE_FILENAME",
    "Recipe",
    "RecipeKB",
    "RemoteRecipeClientError",
    "canonical_id_for_path",
    "canonical_id_from_components",
    "canonical_labels",
    "cid_to_path_components",
    "detect_framework_version",
    "recipe_canonical_id",
]
