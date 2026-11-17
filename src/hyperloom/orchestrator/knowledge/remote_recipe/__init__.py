# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom Recipe contract, reader, and CLOSE writer."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._vendor.kb_store_client import KBStoreError, KnowledgeSections
from .client import (
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
    _deactivate_destination,
)
from .models import (
    KBSelectionProfile,
    RecipeScope,
    RemoteRecipeValidationError,
    RemoteWriteResult,
)
from .values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
    build_remote_knowledge,
    has_new_keep,
    knowledge_to_warm_recipe,
)

log = logging.getLogger(__name__)


def write_final_remote_recipe(
    state: Any,
    canonical_id: str,
    session_id: str,
    *,
    client: RemoteRecipeClient,
) -> RemoteWriteResult:
    """Build and conditionally write one final E2E/CLOSE session record."""
    if not has_new_keep(state):
        return RemoteWriteResult("skipped", "no_new_keep_or_pure_warm_replay", canonical_id, session_id)
    try:
        scope = RecipeScope.from_state(state)
    except RemoteRecipeValidationError:
        return RemoteWriteResult("skipped", "invalid_recipe_scope", canonical_id, session_id)
    try:
        profile = KBSelectionProfile.from_state(state, scope=scope)
    except RemoteRecipeValidationError as exc:
        return RemoteWriteResult(
            "skipped",
            exc.reason or "invalid_recipe_selection_profile",
            canonical_id,
            session_id,
        )
    with tempfile.TemporaryDirectory(prefix="hyperloom-remote-recipe-") as temporary:
        files_dir = Path(temporary) / "files"
        bundle = build_remote_knowledge(
            state,
            files_dir,
            sections=KnowledgeSections.from_env(),
            profile=profile,
            session_id=session_id,
        )
        return client.write_if_better(
            canonical_id,
            session_id,
            bundle,
            profile,
            files_dir=files_dir,
        )


class RemoteWarmRecipeAdapter:
    """Read-only adapter that gives T0 current-record metadata and advisories."""

    mode = "remote"
    backend_name = "kb-store"
    search_candidate_cap = 200
    search_page_cap = 10

    def __init__(
        self,
        client: RemoteRecipeClient,
        destination: str | Path,
        scope: RecipeScope,
    ) -> None:
        self._client = client
        self._destination = Path(destination)
        self._scope = scope
        self._cache: dict[str, dict[str, Any] | None] = {}
        self._candidate_views: dict[str, dict[str, Any]] = {}
        self._candidate_rows: dict[str, dict[str, Any]] = {}
        self._scanned_candidate_ids: set[str] = set()

    def _read(self, canonical_id: str) -> dict[str, Any] | None:
        if canonical_id not in self._cache:
            try:
                document = self._client.read(canonical_id, self._destination, scope=self._scope)
                row = None if document is None else knowledge_to_warm_recipe(document)
            except Exception:
                _deactivate_destination(self._destination)
                raise
            if row is None:
                _deactivate_destination(self._destination)
                self._cache[canonical_id] = None
            else:
                try:
                    replay_material = row.get(
                        "replay_material_available"
                    ) is True and self._candidate_has_replay_material(self._destination)
                except (
                    KBStoreError,
                    OSError,
                    RemoteRecipeValidationError,
                    ValueError,
                ) as exc:
                    log.warning(
                        "exact remote Recipe material rejected %s: %s",
                        canonical_id,
                        exc,
                    )
                    _deactivate_destination(self._destination)
                    replay_material = False
                row["replay_material_available"] = replay_material
                self._cache[canonical_id] = row
        return self._cache[canonical_id]

    def get_recipe(self, *, canonical_id: str) -> dict[str, Any] | None:
        """Return the exact record; degraded tiers are ranked by T0 via ``search``."""
        return self._read(canonical_id)

    @staticmethod
    def _candidate_has_replay_material(candidate_dir: Path) -> bool:
        """Ask each column's facade whether it carries anything replayable."""
        from ..agent_kb import ConfigKB, KernelAgentKB, PatchKB

        sections = KnowledgeSections(
            candidate_dir / ".selection-sdk",
            warm_start_dir=candidate_dir,
        )
        config_kb = ConfigKB(sections)
        config = config_kb.read()
        if str(config.get("extra_server_args") or "").strip() or config.get("extra_envs"):
            return True
        # A launch whose only change removes options of the launch the framework resolves, as a recipe's launch
        # document carries them.
        if config_kb.prior().get("remove_args"):
            return True
        if PatchKB(sections).read_patches():
            return True
        kernel = KernelAgentKB(sections)
        for column in (
            kernel.read_gemm(),
            kernel.read_fusion(),
            kernel.read_rewrite(),
        ):
            if isinstance(column, Mapping) and any(column.values()):
                return True
        return False

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        hardware_in: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Scan bounded candidate metadata while leaving ranking to T0."""
        translated: dict[str, str] = {}
        for raw_key, raw_value in (label_match or {}).items():
            value = str(raw_value or "").strip()
            if not value:
                continue
            translated[str(raw_key)] = value

        page_size = min(100, max(1, int(limit or 10)))
        if len(self._scanned_candidate_ids) >= self.search_candidate_cap:
            return []
        offset = 0
        rows: list[dict[str, Any]] = []
        for _ in range(self.search_page_cap):
            result = self._client.search_identities(
                scheme=self._scope.identity_scheme,
                match=translated,
                hardware_in=hardware_in,
                offset=offset,
                limit=page_size,
            )
            items = result.get("items")
            if not isinstance(items, list):
                raise RemoteRecipeValidationError("KB Store identity search items must be a list")
            if not items:
                break
            rows.extend(self._page_rows(items))
            if len(self._scanned_candidate_ids) >= self.search_candidate_cap:
                break
            next_offset = result.get("next_offset")
            if next_offset is None:
                break
            try:
                resolved_offset = int(next_offset)
            except (TypeError, ValueError) as exc:
                raise RemoteRecipeValidationError(
                    "KB Store identity search next_offset must be integer or null"
                ) from exc
            if resolved_offset <= offset:
                raise RemoteRecipeValidationError("KB Store identity search pagination did not advance")
            offset = resolved_offset
        else:
            log.warning(
                "remote Recipe candidate search stopped after %d pages",
                self.search_page_cap,
            )
        return rows

    def _page_rows(self, items: list[Any]) -> list[dict[str, Any]]:
        """Project one page of identity hits onto warm rows, fetching each new candidate's view until the scan cap."""
        rows: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            canonical_id = str(item.get("canonical_id") or "").strip()
            if not canonical_id.startswith(f"{self._scope.identity_scheme}:"):
                continue
            cached = self._candidate_rows.get(canonical_id)
            if cached is not None:
                rows.append(dict(cached))
                continue
            if len(self._scanned_candidate_ids) >= self.search_candidate_cap:
                break
            self._scanned_candidate_ids.add(canonical_id)
            try:
                envelope = self._client.get_view(
                    canonical_id,
                    self._scope,
                )
                if envelope is None:
                    continue
                row = knowledge_to_warm_recipe(envelope)
            except (
                KBStoreError,
                OSError,
                RemoteRecipeValidationError,
                ValueError,
            ) as exc:
                log.warning(
                    "remote Recipe candidate rejected %s: %s",
                    canonical_id,
                    exc,
                )
                continue
            dimensions = item.get("dimensions")
            if isinstance(dimensions, dict):
                for key, value in dimensions.items():
                    row[str(key)] = str(value)
            row["updated_at"] = str(item.get("updated_at") or "")
            row["identity_source"] = str(item.get("source") or "")
            self._candidate_views[canonical_id] = envelope
            self._candidate_rows[canonical_id] = dict(row)
            rows.append(dict(row))
        return rows

    def select_candidate(self, row: Mapping[str, Any]) -> bool:
        """Materialize a candidate only after T0 accepts and ranks it."""
        canonical_id = str(row.get("canonical_id") or "").strip()
        envelope = self._candidate_views.get(canonical_id)
        if not canonical_id or row.get("replayable") is not True or envelope is None:
            return False
        document = self._client.read(
            canonical_id,
            self._destination,
            scope=self._scope,
            envelope=envelope,
        )
        if document is None:
            _deactivate_destination(self._destination)
            return False
        try:
            if not self._candidate_has_replay_material(self._destination):
                _deactivate_destination(self._destination)
                return False
            selected = knowledge_to_warm_recipe(document)
        except Exception:
            _deactivate_destination(self._destination)
            raise
        selected.update(
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "model",
                    "hardware",
                    "framework_name",
                    "model_type",
                    "architectures",
                    "framework_version",
                    "precision",
                    "updated_at",
                    "identity_source",
                }
            }
        )
        self._cache[canonical_id] = selected
        return True


__all__ = [
    "CURRENT_KNOWLEDGE_SCHEMA_VERSION",
    "RECORD_KIND_HYPERLOOM_RECIPE",
    "KBStoreError",
    "KnowledgeSections",
    "RemoteRecipeClient",
    "RemoteRecipeConfigurationError",
    "RecipeScope",
    "RemoteWarmRecipeAdapter",
    "build_remote_knowledge",
    "knowledge_to_warm_recipe",
    "has_new_keep",
    "write_final_remote_recipe",
]
