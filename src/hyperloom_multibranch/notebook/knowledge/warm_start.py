# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE warm start: what the Recipe KB says about this session's task, as leads for the first specialists.

The read takes the single-lineage PRELUDE's two steps:

1. ``search_identities`` finds the canonical ids that match the session on six of the seven serving dimensions,
   labelled as ``canonical_labels`` labels them; the framework version is left free, so a recipe measured on the
   previous release still counts.
2. ``get_view`` reads each one's champion in exactly this session's scope.

Only exact scopes are read: the KB Store does not document whether ``get_rollup`` without a scope lists a record's
other scopes, so finding the nearest concurrency and sequence lengths waits until it does.

Every record goes through ``knowledge_to_warm_recipe``, so records the single-lineage CLOSE wrote read the same way.
A lead names the dimensions where it differs from this session. It is a lead, not a starting point: specialists still
measure it on this session's root.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from hyperloom.inference_optimizer.recipe_snapshot_constants import F_LABEL_FRAMEWORK_VERSION, canonical_labels
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.models import (
    CONFIG_SECTION,
    RecipeScope,
    RemoteRecipeValidationError,
)
from hyperloom.orchestrator.knowledge.remote_recipe.values import knowledge_to_warm_recipe
from hyperloom_multibranch.contracts.notebook.records import SessionProfile
from hyperloom_multibranch.notebook.knowledge.lessons import listed
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import recipe_identity

log = logging.getLogger(__name__)

MAX_CANDIDATES = 20
PAGE_SIZE = 50


class RecipeReader(Protocol):
    """What the warm start needs of a Recipe KB client, such as ``RemoteRecipeClient``."""

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str],
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]: ...

    def get_view(self, canonical_id: str, scope: RecipeScope) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class WarmLead:
    """One Recipe KB champion in this session's scope; ``differing`` maps a dimension to (theirs, ours)."""

    canonical_id: str
    session_id: str
    scope: Mapping[str, object]
    gain_pct: float | None
    best_throughput: float | None
    server_args: str
    envs: Mapping[str, str]
    replayable: bool
    lessons: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    what_failed: tuple[str, ...] = ()
    differing: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def describe(self) -> str:
        gain = "n/a" if self.gain_pct is None else f"{self.gain_pct:+.2f}%"
        where = "".join(f", {key} {theirs} (here {ours})" for key, (theirs, ours) in self.differing.items())
        args = f" `{self.server_args}`" if self.server_args else ""
        return f"{self.canonical_id}{where}: {gain}{args} [{self.session_id}]"

    def render(self) -> str:
        return f"- {self.describe()}"


def identity_labels(profile: SessionProfile) -> dict[str, str]:
    """The session's seven serving dimensions, as the KB labels them."""

    return canonical_labels(
        model=profile.model,
        hardware=profile.hardware,
        framework_name=profile.framework,
        model_type=profile.model_type,
        architectures=list(profile.architectures),
        framework_version=profile.framework_version,
        precision=profile.precision,
    )


def _config(envelope: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """The launch arguments and environment of a record ``knowledge_to_warm_recipe`` accepted."""

    config = envelope["knowledge"]["value"][CONFIG_SECTION]
    envs = config.get("extra_envs")
    return str(config.get("extra_server_args") or ""), (
        {str(key): str(item) for key, item in envs.items()} if isinstance(envs, Mapping) else {}
    )


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def lead_of(
    envelope: dict[str, Any], scope: RecipeScope, labels: Mapping[str, str], dimensions: Mapping[str, Any]
) -> WarmLead:
    """The lead one champion View gives; raises ``RemoteRecipeValidationError`` on a record of another contract."""

    row = knowledge_to_warm_recipe(envelope)
    server_args, envs = _config(envelope)
    return WarmLead(
        canonical_id=str(envelope.get("canonical_id") or ""),
        session_id=str(row.get("remote_session_id") or ""),
        scope=scope.as_dict(),
        gain_pct=_number(row.get("validated_gain_pct")),
        best_throughput=_number(row.get("best_throughput")),
        server_args=server_args,
        envs=envs,
        replayable=row.get("replay_material_available") is True,
        lessons=listed(map(str, row.get("lessons") or ())),
        pitfalls=listed(map(str, row.get("pitfalls") or ())),
        what_failed=listed(map(str, row.get("what_failed") or ())),
        differing={
            key: (str(dimensions[key]), ours)
            for key, ours in labels.items()
            if key in dimensions and str(dimensions[key]) != ours
        },
    )


def candidates(reader: RecipeReader, scheme: str, match: dict[str, str], limit: int) -> list[Mapping[str, Any]]:
    """The identity search results for ``match``, page by page: at most ``limit`` distinct ids of ``scheme``."""

    found: dict[str, Mapping[str, Any]] = {}
    offset: int | None = 0
    while offset is not None and len(found) < limit:
        page = reader.search_identities(scheme=scheme, match=match, offset=offset, limit=PAGE_SIZE)
        items = [item for item in page.get("items") or () if isinstance(item, Mapping)]
        for item in items:
            canonical_id = str(item.get("canonical_id") or "")
            if canonical_id.startswith(f"{scheme}:") and len(found) < limit:
                found.setdefault(canonical_id, item)
        next_offset = page.get("next_offset")
        offset = next_offset if isinstance(next_offset, int) and items and next_offset > offset else None
    return list(found.values())


def _read_lead(
    reader: RecipeReader, item: Mapping[str, Any], scope: RecipeScope, labels: Mapping[str, str]
) -> WarmLead | None:
    canonical_id = str(item["canonical_id"])
    try:
        envelope = reader.get_view(canonical_id, scope)
        dimensions = item.get("dimensions")
        return (
            lead_of(envelope, scope, labels, dimensions if isinstance(dimensions, Mapping) else {})
            if envelope is not None
            else None
        )
    except (KBStoreError, RemoteRecipeValidationError, ValueError) as exc:
        log.warning("Recipe KB record %s is no warm-start lead: %s", canonical_id, exc)
        return None


def warm_leads(reader: RecipeReader, profile: SessionProfile, *, limit: int = MAX_CANDIDATES) -> tuple[WarmLead, ...]:
    """Champions of this session's scope on its task, exact identity first, then best gain first.

    None when the session's scope is not one the KB files.
    """

    try:
        _, scope = recipe_identity(profile)
    except RemoteRecipeValidationError:
        return ()
    labels = identity_labels(profile)
    match = {key: value for key, value in labels.items() if key != F_LABEL_FRAMEWORK_VERSION}
    found = (
        _read_lead(reader, item, scope, labels) for item in candidates(reader, scope.identity_scheme, match, limit)
    )
    leads = [lead for lead in found if lead is not None]
    return tuple(sorted(leads, key=lambda lead: (bool(lead.differing), -(lead.gain_pct or 0.0))))


__all__ = ["MAX_CANDIDATES", "RecipeReader", "WarmLead", "candidates", "identity_labels", "lead_of", "warm_leads"]
