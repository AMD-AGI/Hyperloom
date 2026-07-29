"""Chain several read-side recipe clients into one duck-typed remote.

:class:`recipe_kb.RecipeKB` accepts a single ``remote``, but a deployment may
want to try a cheap-but-lossy source ahead of an authoritative one. This wrapper
walks its members in order and returns the first non-empty result, so
``RecipeKB(local, ChainedRemoteRecipeClient([memo, gbrain]))`` resolves reads as
``memo -> gbrain -> local``.

A member that raises is treated as a miss: one unreachable backend must not hide
the rows a later one can still serve.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class ChainedRemoteRecipeClient:
    """Try each member remote in order; the first non-empty answer wins."""

    def __init__(self, members: list[Any]) -> None:
        """Build the chain, dropping members that are absent or disabled.

        Args:
            members: Read-side clients in priority order. ``None`` entries and
                members reporting ``enabled is False`` are skipped.
        """
        self._members: list[Any] = [
            member
            for member in members
            if member is not None and bool(getattr(member, "enabled", False))
        ]

    @property
    def enabled(self) -> bool:
        """Whether any member can serve reads.

        Returns:
            ``True`` when the chain holds at least one enabled member.
        """
        return bool(self._members)

    @property
    def members(self) -> list[Any]:
        """The active members, in priority order.

        Returns:
            A shallow copy of the member list.
        """
        return list(self._members)

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the first member's non-empty row for ``canonical_id``.

        Args:
            canonical_id: Canonical recipe identity to look up.
            version: Forwarded verbatim to each member.

        Returns:
            The first non-empty arbor-shape row, or ``None`` when every member
            misses.
        """
        for member in self._members:
            try:
                row = member.get_recipe(canonical_id=canonical_id, version=version)
            except Exception:  # noqa: BLE001 - a broken member is just a miss
                log.warning(
                    "recipe_kb chain: %s.get_recipe raised",
                    type(member).__name__,
                    exc_info=True,
                )
                continue
            if isinstance(row, dict) and row:
                return row
        return None

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return the first member's non-empty search result.

        Args:
            **kwargs: Forwarded verbatim to each member's ``search``.

        Returns:
            The first non-empty row list, or ``[]`` when every member misses.
        """
        for member in self._members:
            try:
                rows = member.search(**kwargs)
            except Exception:  # noqa: BLE001 - a broken member is just a miss
                log.warning(
                    "recipe_kb chain: %s.search raised",
                    type(member).__name__,
                    exc_info=True,
                )
                continue
            if rows:
                return list(rows)
        return []

    def close(self) -> None:
        """Close every member, best-effort. Idempotent."""
        for member in self._members:
            try:
                member.close()
            except Exception:  # noqa: BLE001
                log.warning(
                    "recipe_kb chain: %s.close raised",
                    type(member).__name__,
                    exc_info=True,
                )


__all__ = [
    "ChainedRemoteRecipeClient",
]
