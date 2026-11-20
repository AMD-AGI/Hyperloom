"""Immutable query-view snapshots, Repeat Groups, and Annotations."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from hyperloom_kb.schema import (
    CATEGORIES,
    OUTCOME,
    Experience,
    ExperienceDeclaration,
    FieldKind,
    FieldRole,
    JsonScalar,
    JsonValue,
)
from hyperloom_kb.storage import StoredExperience

QUERY_VIEW_VERSION = 2
GROUP_BUILDER_VERSION = "repeat-group-v2"
ANNOTATION_BUILDER_VERSION = "annotations-v2"


class QueryViewError(RuntimeError):
    """Raised when a query-view snapshot violates its immutable contract."""


class RetrievalCapability(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(kind: str, value: Any) -> str:
    return hashlib.sha256(f"{kind}\0{_canonical(value)}".encode()).hexdigest()


def _field_key(value: JsonScalar) -> str:
    return _canonical(value)


@dataclass(frozen=True)
class QueryViewRef:
    schema_ref: str
    local_page_sequence: int
    view_id: str

    def __post_init__(self) -> None:
        if not self.schema_ref.startswith("schema:sha256:"):
            raise QueryViewError("QueryViewRef schema_ref is invalid")
        if self.local_page_sequence < 0:
            raise QueryViewError("local_page_sequence must be non-negative")
        if not self.view_id.startswith("view:sha256:"):
            raise QueryViewError("view_id is invalid")


@dataclass(frozen=True)
class RepeatAnnotations:
    member_count: int
    distinct_run_count: int
    decision_counts: dict[str, int]
    measured_count: int
    measurement_min: float | None
    measurement_max: float | None
    measurement_median: float | None
    measurement_mean: float | None
    measurement_variance: float | None
    #: For each boolean outcome field, how many members hold it ``true`` and how many ``false``.
    flag_counts: dict[str, dict[str, int]]
    builder_version: str = ANNOTATION_BUILDER_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "member_count": self.member_count,
            "distinct_run_count": self.distinct_run_count,
            "decision_counts": dict(sorted(self.decision_counts.items())),
            "measured_count": self.measured_count,
            "measurement_min": self.measurement_min,
            "measurement_max": self.measurement_max,
            "measurement_median": self.measurement_median,
            "measurement_mean": self.measurement_mean,
            "measurement_variance": self.measurement_variance,
            "flag_counts": {name: dict(sorted(counts.items())) for name, counts in sorted(self.flag_counts.items())},
            "builder_version": self.builder_version,
        }


@dataclass(frozen=True)
class RepeatGroup:
    group_key: str
    member_ids: tuple[str, ...]
    annotations: RepeatAnnotations

    def __post_init__(self) -> None:
        if len(self.group_key) != 64:
            raise QueryViewError("Repeat Group key is invalid")
        object.__setattr__(self, "member_ids", tuple(sorted(self.member_ids)))
        if not self.member_ids:
            raise QueryViewError("Repeat Group must contain at least one member")
        if len(self.member_ids) != self.annotations.member_count:
            raise QueryViewError("Repeat Group member count does not match Annotations")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "group_key": self.group_key,
            "member_ids": list(self.member_ids),
            "annotations": self.annotations.to_dict(),
        }


@dataclass(frozen=True)
class QueryView:
    ref: QueryViewRef
    visible_experience_ids: tuple[str, ...]
    experience_hashes: dict[str, str]
    field_lookup: dict[str, dict[str, tuple[str, ...]]]
    groups: dict[str, RepeatGroup]
    experience_groups: dict[str, str]
    version: int = QUERY_VIEW_VERSION

    def to_manifest(self) -> dict[str, JsonValue]:
        return {
            "version": self.version,
            "schema_ref": self.ref.schema_ref,
            "local_page_sequence": self.ref.local_page_sequence,
            "visible_experience_ids": list(self.visible_experience_ids),
            "experience_hashes": dict(sorted(self.experience_hashes.items())),
            "field_lookup": {
                field: {value: list(ids) for value, ids in sorted(values.items())}
                for field, values in sorted(self.field_lookup.items())
            },
            "groups": {key: group.to_dict() for key, group in sorted(self.groups.items())},
            "experience_groups": dict(sorted(self.experience_groups.items())),
        }


class QueryViewBuilder:
    """Deterministically build one complete query-view snapshot."""

    @staticmethod
    def view_id(manifest: dict[str, JsonValue]) -> str:
        return f"view:sha256:{_digest('query-view', manifest)}"

    def build(self, declaration: ExperienceDeclaration, records: tuple[StoredExperience, ...]) -> QueryView:
        experiences = tuple(
            sorted(
                (record.experience for record in records),
                key=lambda experience: experience.id,
            )
        )
        for experience in experiences:
            declaration.validate(experience)
        experience_hashes = {record.experience.id: record.content_hash for record in records}
        lookup = _build_lookup(declaration, experiences)
        groups, experience_groups = _build_groups(declaration, experiences)
        manifest: dict[str, JsonValue] = {
            "version": QUERY_VIEW_VERSION,
            "schema_ref": declaration.schema_ref,
            "local_page_sequence": len(experiences),
            "visible_experience_ids": [item.id for item in experiences],
            "experience_hashes": dict(sorted(experience_hashes.items())),
            "field_lookup": {
                field: {value: list(ids) for value, ids in sorted(values.items())}
                for field, values in sorted(lookup.items())
            },
            "groups": {key: group.to_dict() for key, group in sorted(groups.items())},
            "experience_groups": dict(sorted(experience_groups.items())),
        }
        ref = QueryViewRef(
            declaration.schema_ref,
            len(experiences),
            self.view_id(manifest),
        )
        return QueryView(
            ref=ref,
            visible_experience_ids=tuple(item.id for item in experiences),
            experience_hashes=experience_hashes,
            field_lookup=lookup,
            groups=groups,
            experience_groups=experience_groups,
        )

    def restrict(self, view: QueryView, experience_ids: Iterable[str]) -> QueryView:
        """Narrow recall visibility; Repeat Groups keep full-corpus annotations."""

        allowed = set(experience_ids)
        restricted = replace(
            view,
            visible_experience_ids=tuple(item for item in view.visible_experience_ids if item in allowed),
        )
        return replace(
            restricted,
            ref=QueryViewRef(
                view.ref.schema_ref,
                view.ref.local_page_sequence,
                self.view_id(restricted.to_manifest()),
            ),
        )


def _lookup_fields(declaration: ExperienceDeclaration, experience: Experience) -> dict[str, JsonScalar]:
    """The ``category.field`` paths exact lookup finds this record under, with its values."""

    values: dict[str, JsonScalar] = {
        "schema_ref": experience.schema_ref,
        "objective": experience.objective,
        "status": experience.status.value,
    }
    paths = declaration.lookup_fields()
    for category in CATEGORIES:
        for name, value in getattr(experience, category).items():
            # Identity fields no declaration names are indexed too, so a consumer can look them up.
            if category == "identity" or f"{category}.{name}" in paths:
                values[f"{category}.{name}"] = value
    return values


def _build_lookup(
    declaration: ExperienceDeclaration,
    experiences: tuple[Experience, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    staged: dict[str, dict[str, list[str]]] = {}
    for experience in experiences:
        for field, value in _lookup_fields(declaration, experience).items():
            staged.setdefault(field, {}).setdefault(_field_key(value), []).append(experience.id)
    return {
        field: {value: tuple(sorted(ids)) for value, ids in sorted(values.items())}
        for field, values in sorted(staged.items())
    }


def repeat_group_key(declaration: ExperienceDeclaration, experience: Experience) -> str:
    """Records that repeat one attempt share this key: the schema, identity, objective, and declared group fields."""

    grouped = {
        f"{category}.{name}": getattr(experience, category).get(name) for category, name in declaration.group_fields()
    }
    return _digest(
        GROUP_BUILDER_VERSION,
        {
            "schema_ref": experience.schema_ref,
            "identity": experience.identity,
            "objective": experience.objective,
            "fields": grouped,
        },
    )


def _annotations(declaration: ExperienceDeclaration, experiences: tuple[Experience, ...]) -> RepeatAnnotations:
    decision = declaration.role_field(OUTCOME, FieldRole.DECISION)
    measurement = declaration.role_field(OUTCOME, FieldRole.MEASUREMENT)
    flag_fields = tuple(
        item.name for item in declaration.fields(OUTCOME) if item.kind is FieldKind.BOOLEAN and not item.many
    )
    decisions: dict[str, int] = {}
    run_ids: set[str] = set()
    measurements: list[float] = []
    flags: dict[str, dict[str, int]] = {}
    for experience in experiences:
        run_ids.add(experience.run_id)
        outcome = experience.outcome
        value = outcome.get(decision.name) if decision is not None else None
        if isinstance(value, str):
            decisions[value] = decisions.get(value, 0) + 1
        measured = outcome.get(measurement.name) if measurement is not None else None
        if isinstance(measured, (int, float)) and not isinstance(measured, bool):
            measurements.append(float(measured))
        for name in flag_fields:
            if isinstance(outcome.get(name), bool):
                counts = flags.setdefault(name, {"true": 0, "false": 0})
                counts["true" if outcome[name] else "false"] += 1
    return RepeatAnnotations(
        member_count=len(experiences),
        distinct_run_count=len(run_ids),
        decision_counts=decisions,
        measured_count=len(measurements),
        measurement_min=min(measurements) if measurements else None,
        measurement_max=max(measurements) if measurements else None,
        measurement_median=statistics.median(measurements) if measurements else None,
        measurement_mean=statistics.fmean(measurements) if measurements else None,
        measurement_variance=(statistics.pvariance(measurements) if len(measurements) >= 2 else None),
        flag_counts=flags,
    )


def _build_groups(
    declaration: ExperienceDeclaration,
    experiences: tuple[Experience, ...],
) -> tuple[dict[str, RepeatGroup], dict[str, str]]:
    staged: dict[str, list[Experience]] = {}
    experience_groups: dict[str, str] = {}
    for experience in experiences:
        key = repeat_group_key(declaration, experience)
        staged.setdefault(key, []).append(experience)
        experience_groups[experience.id] = key
    groups = {
        key: RepeatGroup(
            group_key=key,
            member_ids=tuple(item.id for item in members),
            annotations=_annotations(declaration, tuple(members)),
        )
        for key, members in staged.items()
    }
    return dict(sorted(groups.items())), dict(sorted(experience_groups.items()))


__all__ = [
    "ANNOTATION_BUILDER_VERSION",
    "GROUP_BUILDER_VERSION",
    "QUERY_VIEW_VERSION",
    "QueryView",
    "QueryViewBuilder",
    "QueryViewError",
    "QueryViewRef",
    "RepeatAnnotations",
    "RepeatGroup",
    "RetrievalCapability",
    "repeat_group_key",
]
