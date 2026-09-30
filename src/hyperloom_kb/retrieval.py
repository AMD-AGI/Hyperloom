"""Policy-neutral retrieval primitives over one immutable Query View."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast, runtime_checkable

from hyperloom_kb.query_view import (
    CapabilityState,
    QueryView,
    QueryViewRef,
    QueryViewStore,
    RepeatAnnotations,
    RetrievalCapability,
)
from hyperloom_kb.schema import Experience, JsonScalar, JsonValue, RenderedRef
from hyperloom_kb.storage import ExperienceStore


class RetrievalError(RuntimeError):
    """Base error for policy-neutral retrieval primitives."""


class ViewUnavailable(RetrievalError):
    pass


class LeaseExpired(RetrievalError):
    pass


class CapabilityUnavailable(RetrievalError):
    pass


class MixedQueryViews(RetrievalError):
    pass


@dataclass(frozen=True)
class CandidateHit:
    experience_id: str
    capability: RetrievalCapability
    score: float | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class CandidateProvider(Protocol):
    capability: RetrievalCapability

    def recall(
        self,
        request: dict[str, JsonValue],
        view: QueryView,
        *,
        limit: int,
    ) -> tuple[CandidateHit, ...]:
        """Return source-labelled candidate hits for one exact Query View."""


@dataclass(frozen=True)
class ReadLease:
    lease_id: str
    view: QueryViewRef
    expires_at_monotonic: float


class ReadLeaseManager:
    def __init__(self, *, max_ttl_seconds: float = 300.0) -> None:
        if max_ttl_seconds <= 0:
            raise ValueError("max_ttl_seconds must be positive")
        self._max_ttl = max_ttl_seconds
        self._leases: dict[str, ReadLease] = {}
        self._lock = threading.RLock()

    def acquire(self, view: QueryViewRef, ttl_seconds: float | None = None) -> ReadLease:
        ttl = self._ttl(ttl_seconds)
        lease = ReadLease(
            lease_id=f"lease-{uuid.uuid4().hex}",
            view=view,
            expires_at_monotonic=time.monotonic() + ttl,
        )
        with self._lock:
            self._leases[lease.lease_id] = lease
        return lease

    def renew(self, lease_id: str, ttl_seconds: float | None = None) -> ReadLease:
        with self._lock:
            current = self.require(lease_id)
            lease = ReadLease(
                lease_id=current.lease_id,
                view=current.view,
                expires_at_monotonic=time.monotonic() + self._ttl(ttl_seconds),
            )
            self._leases[lease_id] = lease
            return lease

    def release(self, lease_id: str) -> None:
        with self._lock:
            self._leases.pop(lease_id, None)

    def require(self, lease_id: str, view: QueryViewRef | None = None) -> ReadLease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.expires_at_monotonic <= time.monotonic():
                self._leases.pop(lease_id, None)
                raise LeaseExpired(lease_id)
            if view is not None and lease.view != view:
                raise MixedQueryViews("lease does not belong to the requested Query View")
            return lease

    def _ttl(self, requested: float | None) -> float:
        ttl = self._max_ttl if requested is None else float(requested)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        return min(ttl, self._max_ttl)


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


@dataclass(frozen=True)
class RetrievalResult:
    view: QueryViewRef
    lease_id: str
    groups: tuple[GroupCandidate, ...]
    rendered: RenderedResult
    executed_capabilities: tuple[RetrievalCapability, ...]
    unavailable_capabilities: tuple[RetrievalCapability, ...]


ExperienceRenderer: TypeAlias = Callable[[Experience, QueryView], str]


class LocalRetrievalService:
    """Expose independent recall, organize, and render capabilities."""

    def __init__(
        self,
        experiences: ExperienceStore,
        views: QueryViewStore,
        *,
        providers: tuple[CandidateProvider, ...] = (),
        leases: ReadLeaseManager | None = None,
        renderer: ExperienceRenderer | None = None,
    ) -> None:
        self._experiences = experiences
        self._views = views
        self._renderer = renderer or _render_experience
        self._providers = {provider.capability: provider for provider in providers}
        if any(
            capability not in {RetrievalCapability.FUZZY, RetrievalCapability.SEMANTIC}
            for capability in self._providers
        ):
            raise ValueError("external providers may implement only fuzzy or semantic recall")
        self.leases = leases or ReadLeaseManager()

    def acquire_view(
        self,
        schema_ref: str,
        *,
        view: QueryViewRef | None = None,
        ttl_seconds: float | None = None,
    ) -> ReadLease:
        resolved = self._resolve_view(schema_ref, view)
        return self.leases.acquire(resolved.ref, ttl_seconds)

    def query(
        self,
        where: dict[str, JsonScalar],
        *,
        view: QueryViewRef,
        lease_id: str,
        limit: int = 200,
    ) -> tuple[CandidateHit, ...]:
        return self._structured_query(
            where,
            capability=RetrievalCapability.FILTER,
            view=view,
            lease_id=lease_id,
            limit=limit,
        )

    def exact(
        self,
        where: dict[str, JsonScalar],
        *,
        view: QueryViewRef,
        lease_id: str,
        limit: int = 200,
    ) -> tuple[CandidateHit, ...]:
        return self._structured_query(
            where,
            capability=RetrievalCapability.EXACT,
            view=view,
            lease_id=lease_id,
            limit=limit,
        )

    def _structured_query(
        self,
        where: dict[str, JsonScalar],
        *,
        capability: RetrievalCapability,
        view: QueryViewRef,
        lease_id: str,
        limit: int,
    ) -> tuple[CandidateHit, ...]:
        snapshot = self._leased_view(view, lease_id)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        candidate_ids = set(snapshot.visible_experience_ids)
        for field_name, value in sorted(where.items()):
            values = snapshot.field_lookup.get(field_name)
            if values is None:
                candidate_ids.clear()
                break
            candidate_ids.intersection_update(values.get(_value_key(value), ()))
        return tuple(CandidateHit(experience_id, capability) for experience_id in sorted(candidate_ids)[:limit])

    def recall(
        self,
        capability: RetrievalCapability,
        request: dict[str, JsonValue],
        *,
        view: QueryViewRef,
        lease_id: str,
        limit: int = 20,
    ) -> tuple[CandidateHit, ...]:
        if capability not in {RetrievalCapability.FUZZY, RetrievalCapability.SEMANTIC}:
            raise ValueError("recall capability must be fuzzy or semantic")
        snapshot = self._leased_view(view, lease_id)
        if snapshot.capabilities.get(capability) is not CapabilityState.READY:
            raise CapabilityUnavailable(capability.value)
        provider = self._providers.get(capability)
        if provider is None:
            raise CapabilityUnavailable(capability.value)
        visible = set(snapshot.visible_experience_ids)
        return tuple(
            hit
            for hit in provider.recall(request, snapshot, limit=limit)
            if hit.experience_id in visible and hit.capability is capability
        )[:limit]

    def organize(
        self,
        candidate_sets: tuple[tuple[CandidateHit, ...], ...],
        *,
        view: QueryViewRef,
        lease_id: str,
    ) -> tuple[GroupCandidate, ...]:
        snapshot = self._leased_view(view, lease_id)
        by_experience: dict[str, list[CandidateHit]] = {}
        for candidates in candidate_sets:
            for hit in candidates:
                if hit.experience_id not in snapshot.experience_groups:
                    continue
                by_experience.setdefault(hit.experience_id, []).append(hit)
        grouped_hits: dict[str, list[CandidateHit]] = {}
        for experience_id, hits in by_experience.items():
            group_key = snapshot.experience_groups[experience_id]
            grouped_hits.setdefault(group_key, []).extend(hits)
        return tuple(
            GroupCandidate(
                group_key=group_key,
                annotations=snapshot.groups[group_key].annotations,
                member_ids=snapshot.groups[group_key].member_ids,
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
        view: QueryViewRef,
        lease_id: str,
        budget_chars: int | None = 4_000,
    ) -> RenderedResult:
        """Render representatives; ``budget_chars=None`` never truncates."""

        snapshot = self._leased_view(view, lease_id)
        if budget_chars is not None and budget_chars <= 0:
            raise ValueError("budget_chars must be positive")
        visible = set(snapshot.visible_experience_ids)
        experiences: list[Experience] = []
        for experience_id in representatives:
            if experience_id not in visible:
                raise ViewUnavailable(f"{experience_id} is not visible in {view.view_id}")
            record = self._experiences.get_experience(experience_id)
            if record is None:
                raise ViewUnavailable(f"{experience_id} is missing from ExperienceStore")
            if record.content_hash != snapshot.experience_hashes.get(experience_id):
                raise ViewUnavailable(f"{experience_id} content does not match pinned Query View")
            experiences.append(record.experience)
        blocks = [self._renderer(experience, snapshot) for experience in experiences]
        text = "\n\n".join(blocks)
        truncated = budget_chars is not None and len(text) > budget_chars
        if budget_chars is not None and truncated:
            text = text[: max(0, budget_chars - 14)].rstrip() + "\n… [truncated]"
        return RenderedResult(
            text=text,
            rendered_refs=tuple(RenderedRef(experience.id, "representative") for experience in experiences),
            truncated=truncated,
        )

    def result(
        self,
        groups: tuple[GroupCandidate, ...],
        rendered: RenderedResult,
        *,
        view: QueryViewRef,
        lease_id: str,
        executed_capabilities: tuple[RetrievalCapability, ...],
        unavailable_capabilities: tuple[RetrievalCapability, ...] = (),
    ) -> RetrievalResult:
        self._leased_view(view, lease_id)
        return RetrievalResult(
            view=view,
            lease_id=lease_id,
            groups=groups,
            rendered=rendered,
            executed_capabilities=tuple(dict.fromkeys(executed_capabilities)),
            unavailable_capabilities=tuple(dict.fromkeys(unavailable_capabilities)),
        )

    def _resolve_view(
        self,
        schema_ref: str,
        requested: QueryViewRef | None,
    ) -> QueryView:
        if requested is None:
            snapshot = self._views.current_view(schema_ref)
        else:
            snapshot = self._views.get_view(requested.view_id)
            if snapshot is not None and snapshot.ref != requested:
                raise MixedQueryViews("requested QueryViewRef does not match stored view")
        if snapshot is None or snapshot.ref.schema_ref != schema_ref:
            raise ViewUnavailable(schema_ref)
        return snapshot

    def _leased_view(self, view: QueryViewRef, lease_id: str) -> QueryView:
        self.leases.require(lease_id, view)
        snapshot = self._views.get_view(view.view_id)
        if snapshot is None:
            raise ViewUnavailable(view.view_id)
        if snapshot.ref != view:
            raise MixedQueryViews("stored Query View does not match requested reference")
        return snapshot


def _value_key(value: JsonScalar) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_experience(experience: Experience, view: QueryView) -> str:
    group_key = view.experience_groups[experience.id]
    annotations = view.groups[group_key].annotations
    outcome = experience.outcome
    return "\n".join(
        (
            f"Experience {experience.id}",
            f"Repeat Group: {group_key}",
            f"Conditions: {dict(sorted(experience.identity.items()))}",
            f"Objective: {experience.objective}",
            (
                "Baseline: "
                f"identity={dict(sorted(experience.baseline_identity.items()))}, "
                f"value={experience.baseline_value}"
            ),
            (
                "Annotations: "
                f"members={annotations.member_count}, "
                f"runs={annotations.distinct_run_count}, "
                f"decisions={dict(sorted(annotations.decision_counts.items()))}"
            ),
            f"Reasoning: {experience.reasoning}",
            f"Change: {experience.change.summary if experience.change else ''}",
            (f"Outcome: decision={outcome.decision if outcome else ''}, value={outcome.value if outcome else None}"),
            f"Reflection: {experience.reflection}",
        )
    )


def content_ref(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def render_complete_experience(
    experience: Experience,
    view: QueryView,
    *,
    inline_limit: int | None = None,
    external: dict[str, str] | None = None,
) -> str:
    """Render every canonical field plus full Repeat Group annotations.

    A ``change.content`` longer than ``inline_limit`` bytes renders as a reference to its text, which is put in
    ``external`` under that reference: the record stays complete while the prompt carries only its size.
    """

    group_key = view.experience_groups[experience.id]
    annotations = view.groups[group_key].annotations
    record = experience.to_dict()
    content = experience.change.content if experience.change is not None else ""
    if inline_limit is not None and external is not None and len(content.encode()) > inline_limit:
        ref = content_ref(content)
        external[ref] = content
        cast(dict[str, JsonValue], record["change"])["content"] = (
            f"<external content {ref}, {len(content.encode())} bytes>"
        )
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
    "CandidateProvider",
    "CapabilityUnavailable",
    "ExperienceRenderer",
    "GroupCandidate",
    "LeaseExpired",
    "LocalRetrievalService",
    "MixedQueryViews",
    "ReadLease",
    "ReadLeaseManager",
    "RenderedResult",
    "RetrievalError",
    "RetrievalResult",
    "ViewUnavailable",
    "content_ref",
    "render_complete_experience",
]
