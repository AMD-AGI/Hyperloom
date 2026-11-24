"""Policy-neutral retrieval primitives over one immutable Query View."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from hyperloom_kb.query_view import (
    QueryView,
    RepeatAnnotations,
    RetrievalCapability,
)
from hyperloom_kb.schema import (
    CATEGORIES,
    Experience,
    FieldValue,
    FileRef,
    JsonScalar,
    JsonValue,
    RenderedRef,
)
from hyperloom_kb.storage import ExperienceStore


class RetrievalError(RuntimeError):
    """Base error for policy-neutral retrieval primitives."""


class ViewUnavailable(RetrievalError):
    pass


@dataclass(frozen=True)
class CandidateHit:
    experience_id: str
    capability: RetrievalCapability
    score: float | None = None


@dataclass(frozen=True)
class GroupCandidate:
    group_key: str
    annotations: RepeatAnnotations
    member_ids: tuple[str, ...]
    hits: tuple[CandidateHit, ...]


@dataclass(frozen=True)
class RenderedResult:
    text: str
    rendered_refs: tuple[RenderedRef, ...]
    truncated: bool


ExperienceRenderer: TypeAlias = Callable[[Experience, QueryView], str]


class LocalRetrievalService:
    """Exact recall, grouping, and rendering over the one Query View it is built on."""

    def __init__(
        self,
        experiences: ExperienceStore,
        view: QueryView,
        *,
        renderer: ExperienceRenderer | None = None,
    ) -> None:
        self._experiences = experiences
        self.view = view
        self._renderer = renderer or render_complete_experience

    def exact(self, where: dict[str, JsonScalar]) -> tuple[CandidateHit, ...]:
        """The visible Experiences holding every value of ``where`` under its lookup field."""

        candidate_ids = set(self.view.visible_experience_ids)
        for field_name, value in sorted(where.items()):
            values = self.view.field_lookup.get(field_name)
            if values is None:
                candidate_ids.clear()
                break
            candidate_ids.intersection_update(values.get(_value_key(value), ()))
        return tuple(CandidateHit(experience_id, RetrievalCapability.EXACT) for experience_id in sorted(candidate_ids))

    def organize(self, candidate_sets: tuple[tuple[CandidateHit, ...], ...]) -> tuple[GroupCandidate, ...]:
        by_experience: dict[str, list[CandidateHit]] = {}
        for candidates in candidate_sets:
            for hit in candidates:
                if hit.experience_id not in self.view.experience_groups:
                    continue
                by_experience.setdefault(hit.experience_id, []).append(hit)
        grouped_hits: dict[str, list[CandidateHit]] = {}
        for experience_id, hits in by_experience.items():
            group_key = self.view.experience_groups[experience_id]
            grouped_hits.setdefault(group_key, []).extend(hits)
        return tuple(
            GroupCandidate(
                group_key=group_key,
                annotations=self.view.groups[group_key].annotations,
                member_ids=self.view.groups[group_key].member_ids,
                hits=tuple(
                    sorted(
                        hits,
                        key=lambda hit: (
                            hit.experience_id,
                            hit.capability.value,
                            -(hit.score or 0.0),
                        ),
                    )
                ),
            )
            for group_key, hits in sorted(grouped_hits.items())
        )

    def render(
        self,
        representatives: tuple[str, ...],
        *,
        budget_chars: int | None = 4_000,
    ) -> RenderedResult:
        """Render representatives whole, in order, while they fit ``budget_chars``; ``None`` renders every one.

        A record is never cut: one that does not fit is left out and the ones after it that fit are still rendered,
        and only the records rendered are named in ``rendered_refs``, so nothing reads as shown that the prompt did
        not carry.
        """

        if budget_chars is not None and budget_chars <= 0:
            raise ValueError("budget_chars must be positive")
        visible = set(self.view.visible_experience_ids)
        experiences: list[Experience] = []
        for experience_id in representatives:
            if experience_id not in visible:
                raise ViewUnavailable(f"{experience_id} is not visible in {self.view.ref.view_id}")
            record = self._experiences.get_experience(experience_id)
            if record is None:
                raise ViewUnavailable(f"{experience_id} is missing from ExperienceStore")
            if record.content_hash != self.view.experience_hashes.get(experience_id):
                raise ViewUnavailable(f"{experience_id} content does not match pinned Query View")
            experiences.append(record.experience)
        blocks: list[str] = []
        shown: list[Experience] = []
        used = 0
        for experience in experiences:
            block = self._renderer(experience, self.view)
            size = len(block) + (len("\n\n") if blocks else 0)
            if budget_chars is not None and used + size > budget_chars:
                continue
            blocks.append(block)
            shown.append(experience)
            used += size
        return RenderedResult(
            text="\n\n".join(blocks),
            rendered_refs=tuple(RenderedRef(experience.id, "representative") for experience in shown),
            truncated=len(shown) < len(experiences),
        )


def _value_key(value: JsonScalar) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


#: Where a file's content can be read on this host.
FilePath: TypeAlias = Callable[[FileRef], Path]


def _rendered(value: FieldValue, file_path: FilePath | None) -> JsonValue:
    if isinstance(value, tuple):
        return [_rendered(item, file_path) for item in value]
    if not isinstance(value, FileRef):
        return value
    if file_path is None:
        return {"file": value.name, "bytes": value.bytes, "sha256": value.sha256}
    return {"file": value.name, "bytes": value.bytes, "path": str(file_path(value))}


def render_complete_experience(
    experience: Experience,
    view: QueryView,
    *,
    file_path: FilePath | None = None,
) -> str:
    """Render every knowledge field of the record, never condensed, plus full Repeat Group annotations.

    The record's metadata, such as its provenance and the reads that shaped it, stays in the record and out of the
    prompt; the heading names the Experience so an agent can cite it. A text field renders whole. A file field renders
    as its name, size, and the ``file_path`` an agent reads its content from, never as the content itself.
    """

    group_key = view.experience_groups[experience.id]
    annotations = view.groups[group_key].annotations
    record: dict[str, JsonValue] = {"objective": experience.objective}
    for category in CATEGORIES:
        record[category] = {name: _rendered(value, file_path) for name, value in getattr(experience, category).items()}
    if experience.raw_logs:
        record["raw_logs"] = [_rendered(ref, file_path) for ref in experience.raw_logs]
    record["notes"] = dict(experience.notes)
    return "\n".join(
        (
            f"Experience {experience.id}",
            f"Repeat Group: {group_key}",
            "Repeat Group Annotations:",
            json.dumps(annotations.to_dict(), ensure_ascii=False, indent=2),
            "Record:",
            json.dumps(record, ensure_ascii=False, indent=2),
        )
    )


__all__ = [
    "CandidateHit",
    "ExperienceRenderer",
    "FilePath",
    "GroupCandidate",
    "LocalRetrievalService",
    "RenderedResult",
    "RetrievalError",
    "ViewUnavailable",
    "render_complete_experience",
]
