# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional Remote Recipe KB V2 integration.

The module is inert unless both ``KB_STORE_URL`` and ``KB_STORE_TOKEN`` are
configured, so the existing RecipeKB local/remote dispatcher remains unchanged.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

from .client import (
    KBStoreClient,
    KBStoreError,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
)
from .models import RemoteRecipeValidationError, RemoteWriteResult
from .values import (
    build_remote_knowledge,
    convert_v1_recipe_to_knowledge,
    envelope_to_v1_recipe,
    has_new_keep,
)


def read_remote_recipe(
    canonical_id: str,
    destination: str | Path,
    *,
    client: RemoteRecipeClient | None = None,
) -> dict[str, Any] | None:
    """Download the champion as flattened recipe.json + files/."""
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return None
    return resolved.read(canonical_id, destination)


# Kept as a standalone API compatibility alias; it is not wired into T0/replay.
read_remote_champion = read_remote_recipe


def write_final_remote_recipe(
    state: Any,
    canonical_id: str,
    session_id: str,
    *,
    client: RemoteRecipeClient | None = None,
) -> RemoteWriteResult:
    """Build and conditionally write one final E2E/CLOSE session record."""
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return RemoteWriteResult("disabled", "KB_STORE_URL/TOKEN not configured")
    if not has_new_keep(state):
        return RemoteWriteResult("skipped", "no_new_keep_or_pure_warm_replay", canonical_id, session_id)
    current_best = getattr(state, "current_best", {}) or {}
    try:
        throughput = float(current_best.get("tput") or 0.0) if isinstance(current_best, dict) else 0.0
    except (TypeError, ValueError):
        throughput = 0.0
    if not math.isfinite(throughput):
        return RemoteWriteResult(
            "skipped",
            "nonfinite_optimized_throughput",
            canonical_id,
            session_id,
        )
    if throughput <= 0:
        return RemoteWriteResult("skipped", "missing_optimized_throughput", canonical_id, session_id)
    with tempfile.TemporaryDirectory(prefix="hyperloom-remote-recipe-") as temporary:
        files_dir = Path(temporary) / "files"
        bundle = build_remote_knowledge(state, files_dir)
        return resolved.write_if_better(
            canonical_id,
            session_id,
            bundle,
            optimized_throughput=throughput,
            files_dir=files_dir,
        )


class HyperloomRemoteKB:
    """Public facade for Hyperloom's remote inference knowledge."""

    def __init__(self, client: RemoteRecipeClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "HyperloomRemoteKB":
        """Build a configured facade, requiring both KB Store variables."""
        client = RemoteRecipeClient.from_env_optional()
        if client is None:
            raise RemoteRecipeConfigurationError(
                "KB_STORE_URL and KB_STORE_TOKEN are required for HyperloomRemoteKB"
            )
        return cls(client)

    def read(
        self,
        identity: str,
        destination: str | Path,
    ) -> dict[str, Any] | None:
        """Download the champion for a Hyperloom inference canonical id."""
        return read_remote_recipe(identity, destination, client=self._client)

    def write(
        self,
        identity: str,
        state: Any,
        session_id: str | None = None,
    ) -> RemoteWriteResult:
        """Write final E2E knowledge, resolving the session id from state."""
        resolved_session_id = session_id
        if resolved_session_id is None:
            resolved_session_id = (
                str(getattr(state, "recipe_kb_session_id", "") or "").strip()
                or str(getattr(state, "session_id", "") or "").strip()
            )
        resolved_session_id = str(resolved_session_id or "").strip()
        if not resolved_session_id:
            raise RemoteRecipeValidationError(
                "session_id is required; set state.recipe_kb_session_id or state.session_id"
            )
        return write_final_remote_recipe(
            state,
            identity,
            resolved_session_id,
            client=self._client,
        )


__all__ = [
    "HyperloomRemoteKB",
    "KBStoreClient",
    "KBStoreError",
    "RemoteRecipeClient",
    "RemoteRecipeConfigurationError",
    "build_remote_knowledge",
    "convert_v1_recipe_to_knowledge",
    "envelope_to_v1_recipe",
    "has_new_keep",
    "read_remote_champion",
    "read_remote_recipe",
    "write_final_remote_recipe",
]
