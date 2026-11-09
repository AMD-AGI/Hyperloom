# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An in-memory Recipe KB Store with the calls the multi-branch write-back and warm start make."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom_kb import RemoteClientError, RemoteReadResult, RenderedRef

SEARCH_PAGE = 2


def experience_summary(
    experience_id: str,
    identity: Mapping[str, Any],
    *,
    decision: str = "keep",
    outcome: float = 1100.0,
    change: str = "--kv-cache-dtype fp8",
) -> dict[str, Any]:
    """One Experience as the Experience KB's read summaries carry it, measured against a 1000 tokens/s baseline."""

    return {
        "experience_id": experience_id,
        "source_run_id": "run-old",
        "identity": dict(identity),
        "change_summary": change,
        "decision": decision,
        "baseline_value": 1000.0,
        "outcome_value": outcome,
    }


def completed_read(*experiences: dict[str, Any]) -> RemoteReadResult:
    refs = tuple(RenderedRef(item["experience_id"], "representative") for item in experiences)
    return RemoteReadResult("read-1", "completed", "=== block ===", refs, (), experiences)


class FakeExperienceReader:
    """Answers every read with ``result`` and records what was asked."""

    def __init__(self, result: RemoteReadResult) -> None:
        self.result = result
        self.calls: list[tuple[str, Mapping[str, Any], int | None]] = []

    def read(self, decision: str, context: Mapping[str, Any], *, limit: int | None = None) -> RemoteReadResult:
        self.calls.append((decision, context, limit))
        return self.result


@dataclass(frozen=True)
class ExportPage:
    items: tuple[dict[str, Any], ...]
    next_cursor: int
    has_more: bool


@dataclass
class FakeExport:
    """A service's export in write order, ``page`` records at a time; ``failing`` makes every page fail."""

    records: list[dict[str, Any]] = field(default_factory=list)
    page: int = 2
    failing: bool = False
    afters: list[int] = field(default_factory=list)

    def export_page(self, *, after: int = 0, limit: int = 100, schema_ref: str | None = None) -> ExportPage:
        self.afters.append(after)
        if self.failing:
            raise RemoteClientError("the Experience KB service is unreachable")
        chunk = self.records[after : after + min(limit, self.page)]
        items = tuple({"sequence": after + n + 1, "experience": record} for n, record in enumerate(chunk))
        return ExportPage(items, after + len(chunk), after + len(chunk) < len(self.records))


@dataclass
class FakeRecipeStore:
    """Holds one partition's champion and the identities and Views a search finds; records every write."""

    champion: dict[str, Any] | None = None
    identities: list[dict[str, Any]] = field(default_factory=list)
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    uploads: dict[str, bytes] = field(default_factory=dict)
    knowledge: dict[str, Any] | None = None
    objective_schema: str = ""
    promoted: tuple[str, str, float] | None = None
    scopes: list[dict[str, Any] | None] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str] | None = None,
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        self.searches.append({"scheme": scheme, "match": dict(match or {}), "offset": offset})
        end = offset + SEARCH_PAGE
        return {"items": self.identities[offset:end], "next_offset": end if end < len(self.identities) else None}

    def get_hyperloom_recipe_view(
        self, canonical_id: str, *, scope: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        view = self.views.get(canonical_id)
        return copy.deepcopy(view) if view is not None else None

    def get_rollup(self, canonical_id: str, *, scope: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.scopes.append(scope)
        if self.champion is None:
            return None
        return {"champion": self.champion, "sessions": [{"session_id": self.champion["session_id"]}]}

    def put_dir(self, canonical_id: str, session_id: str, local_dir: str | Path, **_: Any) -> dict[str, str]:
        root = Path(local_dir)
        self.uploads = {
            path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        return {rel: f"kb://{canonical_id}/{session_id}/{rel}" for rel in self.uploads}

    def put_knowledge(
        self,
        canonical_id: str,
        knowledge: dict[str, Any],
        *,
        session_id: str = "",
        mode: str = "merge",
        scope: dict[str, Any] | None = None,
        objective_schema: str = "",
    ) -> dict[str, Any]:
        self.knowledge, self.objective_schema = knowledge, objective_schema
        return {"ok": True}

    def set_champion(
        self, canonical_id: str, session_id: str, *, metric: str, value: float, scope: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.promoted = (session_id, metric, value)
        return {"ok": True}
