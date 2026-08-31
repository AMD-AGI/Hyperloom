# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Current Hyperloom inference Recipe contract, reader, and CLOSE writer."""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._vendor.kb_store_client import KnowledgeSections, SectionContent
from .client import (
    KBStoreClient,
    KBStoreError,
    RemoteRecipeClient,
    RemoteRecipeConfigurationError,
    _deactivate_destination,
)
from .models import (
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


def read_remote_recipe(
    canonical_id: str,
    destination: str | Path,
    *,
    scope: RecipeScope,
    client: RemoteRecipeClient | None = None,
) -> dict[str, Any] | None:
    """Download the exact record and validate it against the current Recipe View."""
    resolved = client or RemoteRecipeClient.from_env_optional()
    if resolved is None:
        return None
    document = resolved.read(canonical_id, destination, scope=scope)
    if document is not None:
        try:
            knowledge_to_warm_recipe(document)
        except Exception:  # noqa: BLE001 — cleanup then preserve original error
            _deactivate_destination(Path(destination))
            raise
    return document


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
    try:
        scope = RecipeScope.from_state(state)
    except RemoteRecipeValidationError:
        return RemoteWriteResult("skipped", "invalid_recipe_scope", canonical_id, session_id)
    with tempfile.TemporaryDirectory(prefix="hyperloom-remote-recipe-") as temporary:
        files_dir = Path(temporary) / "files"
        bundle = build_remote_knowledge(
            state,
            files_dir,
            sections=KnowledgeSections.from_env(),
        )
        return resolved.write_if_better(
            canonical_id,
            session_id,
            bundle,
            scope=scope,
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
            raise RemoteRecipeConfigurationError("KB_STORE_URL and KB_STORE_TOKEN are required for HyperloomRemoteKB")
        return cls(client)

    def read(
        self,
        identity: str,
        destination: str | Path,
        scope: RecipeScope,
    ) -> dict[str, Any] | None:
        """Download the selected Recipe View for an inference identity."""
        return read_remote_recipe(
            identity,
            destination,
            scope=scope,
            client=self._client,
        )

    def get_view(
        self,
        identity: str,
        scope: RecipeScope,
    ) -> dict[str, Any] | None:
        """Read normalized metadata without downloading its artifact bundle."""
        return self._client.get_view(identity, scope)

    def materialize_view(
        self,
        identity: str,
        destination: str | Path,
        envelope: dict[str, Any],
        scope: RecipeScope,
    ) -> dict[str, Any] | None:
        """Download and activate the exact previously selected View."""
        return self._client.read(
            identity,
            destination,
            scope=scope,
            envelope=envelope,
        )

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str],
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Discover exact identities through the KB Store search endpoint."""
        return self._client.search_identities(
            scheme=scheme,
            match=match,
            hardware_in=hardware_in,
            offset=offset,
            limit=limit,
        )

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


class RemoteWarmRecipeAdapter:
    """Read-only adapter that gives T0 current-record metadata and advisories."""

    enabled = True
    mode = "remote"
    backend_name = "kb-store"
    search_candidate_cap = 200
    search_page_cap = 10

    def __init__(
        self,
        remote_kb: HyperloomRemoteKB,
        destination: str | Path,
        scope: RecipeScope,
    ) -> None:
        self._remote_kb = remote_kb
        self._destination = Path(destination)
        self._scope = scope
        self._cache: dict[str, dict[str, Any] | None] = {}
        self._materialized_identity = ""
        self._candidate_views: dict[str, dict[str, Any]] = {}
        self._candidate_rows: dict[str, dict[str, Any]] = {}
        self._scanned_candidate_ids: set[str] = set()
        self._deactivate_path(self._destination.parent / f".{self._destination.name}-candidates")
        self._deactivate_path(self._destination.with_name(f".{self._destination.name}.selected"))

    @staticmethod
    def _deactivate_path(path: Path) -> None:
        """Remove a path without following a symlink."""
        _deactivate_destination(path)

    def _read(self, canonical_id: str) -> dict[str, Any] | None:
        if canonical_id not in self._cache:
            try:
                document = self._remote_kb.read(
                    canonical_id,
                    self._destination,
                    self._scope,
                )
            except Exception:  # noqa: BLE001 — deactivate before propagating
                self._deactivate_path(self._destination)
                raise
            if document is None:
                self._deactivate_path(self._destination)
                self._cache[canonical_id] = None
            else:
                try:
                    row = knowledge_to_warm_recipe(document)
                except (
                    RemoteRecipeValidationError,
                    TypeError,
                    ValueError,
                ):
                    self._deactivate_path(self._destination)
                    raise
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
                    self._deactivate_path(self._destination)
                    replay_material = False
                row["replay_material_available"] = replay_material
                self._cache[canonical_id] = row
                if row["replay_material_available"]:
                    self._materialized_identity = canonical_id
        return self._cache[canonical_id]

    def get_authoritative_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the exact record projected into the current Recipe contract."""
        del version
        return self._read(canonical_id)

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
        prefer: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return the exact record; degraded tiers are ranked by T0 via ``search``."""
        del version, prefer
        return self._read(canonical_id)

    @staticmethod
    def _candidate_has_replay_material(candidate_dir: Path) -> bool:
        """Ask each column's facade whether it carries anything replayable."""
        from ..agent_kb import ConfigKB, KernelAgentKB, PatchKB

        sections = KnowledgeSections(
            candidate_dir / ".selection-sdk",
            warm_start_dir=candidate_dir,
        )
        config = ConfigKB(sections).read()
        if str(config.get("extra_server_args") or "").strip() or config.get("extra_envs"):
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
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Scan bounded candidate metadata while leaving ranking to T0."""
        translated: dict[str, str] = {}
        aliases = {
            "hardware": "hardware",
            "framework": "framework_name",
        }
        for raw_key, raw_value in (label_match or {}).items():
            value = str(raw_value or "").strip()
            if not value:
                continue
            key = aliases.get(str(raw_key), str(raw_key))
            translated[key] = value

        page_size = min(100, max(1, int(limit or 10)))
        offset = 0
        pages_scanned = 0
        has_more = False
        rows: list[dict[str, Any]] = []
        while len(self._scanned_candidate_ids) < self.search_candidate_cap and pages_scanned < self.search_page_cap:
            has_more = False
            result = self._remote_kb.search_identities(
                scheme="inference",
                match=translated,
                hardware_in=hardware_in,
                offset=offset,
                limit=page_size,
            )
            pages_scanned += 1
            items = result.get("items")
            if not isinstance(items, list):
                raise RemoteRecipeValidationError("KB Store identity search items must be a list")
            if not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                canonical_id = str(item.get("canonical_id") or "").strip()
                if not canonical_id.startswith("inference:"):
                    continue
                cached = self._candidate_rows.get(canonical_id)
                if cached is not None:
                    rows.append(dict(cached))
                    continue
                if len(self._scanned_candidate_ids) >= self.search_candidate_cap:
                    break
                self._scanned_candidate_ids.add(canonical_id)
                try:
                    envelope = self._remote_kb.get_view(
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
            has_more = True
        if pages_scanned >= self.search_page_cap and has_more:
            log.warning(
                "remote Recipe candidate search stopped after %d pages",
                self.search_page_cap,
            )
        return rows

    def select_candidate(self, row: Mapping[str, Any]) -> bool:
        """Materialize a candidate only after T0 accepts and ranks it."""
        canonical_id = str(row.get("canonical_id") or "").strip()
        envelope = self._candidate_views.get(canonical_id)
        if not canonical_id or row.get("replayable") is not True or envelope is None:
            return False
        document = self._remote_kb.materialize_view(
            canonical_id,
            self._destination,
            envelope,
            self._scope,
        )
        if document is None:
            self._deactivate_path(self._destination)
            return False
        try:
            if not self._candidate_has_replay_material(self._destination):
                self._deactivate_path(self._destination)
                return False
            selected = knowledge_to_warm_recipe(document)
        except Exception:  # noqa: BLE001 — deactivate before rejecting donor
            self._deactivate_path(self._destination)
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
        self._materialized_identity = canonical_id
        return True

    def close(self) -> None:
        """The wrapped blocking client has no explicit lifecycle."""


__all__ = [
    "CURRENT_KNOWLEDGE_SCHEMA_VERSION",
    "RECORD_KIND_HYPERLOOM_RECIPE",
    "HyperloomRemoteKB",
    "KBStoreClient",
    "KBStoreError",
    "KnowledgeSections",
    "RemoteRecipeClient",
    "RemoteRecipeConfigurationError",
    "RecipeScope",
    "RemoteWarmRecipeAdapter",
    "SectionContent",
    "build_remote_knowledge",
    "knowledge_to_warm_recipe",
    "has_new_keep",
    "read_remote_recipe",
    "write_final_remote_recipe",
]
