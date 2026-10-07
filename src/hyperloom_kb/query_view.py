"""Immutable query-view snapshots, Repeat Groups, and Annotations."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hyperloom_kb.schema import (
    CATEGORIES,
    Experience,
    ExperienceDeclaration,
    FieldKind,
    FieldRole,
    JsonScalar,
    JsonValue,
)
from hyperloom_kb.storage import ExperienceStore, SchemaRegistry, StoredExperience

QUERY_VIEW_VERSION = 2
GROUP_BUILDER_VERSION = "repeat-group-v2"
ANNOTATION_BUILDER_VERSION = "annotations-v2"


class QueryViewError(RuntimeError):
    """Raised when a query-view snapshot violates its immutable contract."""


class RetrievalCapability(str, Enum):
    EXACT = "exact"
    FILTER = "filter"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


class CapabilityState(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"


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

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_ref": self.schema_ref,
            "local_page_sequence": self.local_page_sequence,
            "view_id": self.view_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QueryViewRef:
        return cls(
            schema_ref=str(value.get("schema_ref") or ""),
            local_page_sequence=int(value.get("local_page_sequence", -1)),
            view_id=str(value.get("view_id") or ""),
        )


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepeatAnnotations:
        decisions = value.get("decision_counts")
        flags = value.get("flag_counts")
        if not isinstance(decisions, dict) or not isinstance(flags, dict):
            raise QueryViewError("Annotations contain invalid distributions")
        return cls(
            member_count=int(value.get("member_count", -1)),
            distinct_run_count=int(value.get("distinct_run_count", -1)),
            decision_counts={str(key): int(count) for key, count in decisions.items()},
            measured_count=int(value.get("measured_count", -1)),
            measurement_min=_optional_float(value.get("measurement_min")),
            measurement_max=_optional_float(value.get("measurement_max")),
            measurement_median=_optional_float(value.get("measurement_median")),
            measurement_mean=_optional_float(value.get("measurement_mean")),
            measurement_variance=_optional_float(value.get("measurement_variance")),
            flag_counts={
                str(name): {str(key): int(count) for key, count in dict(counts).items()}
                for name, counts in flags.items()
                if isinstance(counts, dict)
            },
            builder_version=str(value.get("builder_version") or ""),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RepeatGroup:
        members = value.get("member_ids")
        annotations = value.get("annotations")
        if not isinstance(members, list) or not isinstance(annotations, dict):
            raise QueryViewError("Repeat Group is invalid")
        return cls(
            group_key=str(value.get("group_key") or ""),
            member_ids=tuple(str(item) for item in members),
            annotations=RepeatAnnotations.from_dict(annotations),
        )


@dataclass(frozen=True)
class QueryView:
    ref: QueryViewRef
    visible_experience_ids: tuple[str, ...]
    experience_hashes: dict[str, str]
    field_lookup: dict[str, dict[str, tuple[str, ...]]]
    groups: dict[str, RepeatGroup]
    experience_groups: dict[str, str]
    capabilities: dict[RetrievalCapability, CapabilityState]
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
            "capabilities": {
                capability.value: state.value
                for capability, state in sorted(
                    self.capabilities.items(),
                    key=lambda item: item[0].value,
                )
            },
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {**self.to_manifest(), "view_id": self.ref.view_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QueryView:
        lookup = value.get("field_lookup")
        groups = value.get("groups")
        experience_groups = value.get("experience_groups")
        experience_hashes = value.get("experience_hashes")
        capabilities = value.get("capabilities")
        visible = value.get("visible_experience_ids")
        if (
            not isinstance(lookup, dict)
            or not isinstance(groups, dict)
            or not isinstance(experience_groups, dict)
            or not isinstance(experience_hashes, dict)
            or not isinstance(capabilities, dict)
            or not isinstance(visible, list)
        ):
            raise QueryViewError("Query View contains invalid structures")
        ref = QueryViewRef(
            schema_ref=str(value.get("schema_ref") or ""),
            local_page_sequence=int(value.get("local_page_sequence", -1)),
            view_id=str(value.get("view_id") or ""),
        )
        view = cls(
            version=int(value.get("version", 0)),
            ref=ref,
            visible_experience_ids=tuple(str(item) for item in visible),
            experience_hashes={
                str(experience_id): str(content_hash) for experience_id, content_hash in experience_hashes.items()
            },
            field_lookup={
                str(field): {
                    str(encoded): tuple(str(item) for item in ids)
                    for encoded, ids in values.items()
                    if isinstance(ids, list)
                }
                for field, values in lookup.items()
                if isinstance(values, dict)
            },
            groups={str(key): RepeatGroup.from_dict(group) for key, group in groups.items() if isinstance(group, dict)},
            experience_groups={
                str(experience_id): str(group_key) for experience_id, group_key in experience_groups.items()
            },
            capabilities={
                RetrievalCapability(str(capability)): CapabilityState(str(state))
                for capability, state in capabilities.items()
            },
        )
        expected = QueryViewBuilder.view_id(view.to_manifest())
        if expected != ref.view_id:
            raise QueryViewError("Query View content does not match view_id")
        return view


class QueryViewBuilder:
    """Deterministically build one complete query-view snapshot."""

    @staticmethod
    def view_id(manifest: dict[str, JsonValue]) -> str:
        return f"view:sha256:{_digest('query-view', manifest)}"

    def build(
        self,
        declaration: ExperienceDeclaration,
        records: tuple[StoredExperience, ...],
        *,
        fuzzy_ready: bool = False,
        semantic_ready: bool = False,
    ) -> QueryView:
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
        capabilities = {
            RetrievalCapability.EXACT: CapabilityState.READY,
            RetrievalCapability.FILTER: CapabilityState.READY,
            RetrievalCapability.FUZZY: (CapabilityState.READY if fuzzy_ready else CapabilityState.UNAVAILABLE),
            RetrievalCapability.SEMANTIC: (CapabilityState.READY if semantic_ready else CapabilityState.UNAVAILABLE),
        }
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
            "capabilities": {
                capability.value: state.value
                for capability, state in sorted(
                    capabilities.items(),
                    key=lambda item: item[0].value,
                )
            },
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
            capabilities=capabilities,
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
    decision = declaration.role_field("outcome", FieldRole.DECISION)
    measurement = declaration.role_field("outcome", FieldRole.MEASUREMENT)
    flag_fields = tuple(
        item.name for item in declaration.fields("outcome") if item.kind is FieldKind.BOOLEAN and not item.many
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


@runtime_checkable
class QueryViewStore(Protocol):
    def publish_view(self, view: QueryView) -> QueryViewRef:
        """Atomically publish an immutable view and advance current."""

    def get_view(self, view_id: str) -> QueryView | None:
        """Return one immutable view by id."""

    def current_view(self, schema_ref: str) -> QueryView | None:
        """Return the current view for one exact schema_ref."""


class InMemoryQueryViewStore:
    def __init__(self) -> None:
        self._views: dict[str, QueryView] = {}
        self._current: dict[str, str] = {}

    def publish_view(self, view: QueryView) -> QueryViewRef:
        existing = self._views.get(view.ref.view_id)
        if existing is not None and existing != view:
            raise QueryViewError("view_id collision")
        self._views[view.ref.view_id] = view
        self._current[view.ref.schema_ref] = view.ref.view_id
        return view.ref

    def get_view(self, view_id: str) -> QueryView | None:
        return self._views.get(view_id)

    def current_view(self, schema_ref: str) -> QueryView | None:
        current = self._current.get(schema_ref)
        return self._views.get(current) if current is not None else None


class LocalQueryViewStore:
    """Filesystem query-view generations with an atomic current pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "query-views"

    @staticmethod
    def _schema_digest(schema_ref: str) -> str:
        if not schema_ref.startswith("schema:sha256:"):
            raise QueryViewError("schema_ref is invalid")
        return schema_ref.rsplit(":", 1)[-1]

    @staticmethod
    def _view_digest(view_id: str) -> str:
        if not view_id.startswith("view:sha256:"):
            raise QueryViewError("view_id is invalid")
        return view_id.rsplit(":", 1)[-1]

    def _schema_root(self, schema_ref: str) -> Path:
        return self.root / self._schema_digest(schema_ref)

    def _view_path(self, schema_ref: str, view_id: str) -> Path:
        return self._schema_root(schema_ref) / "generations" / f"{self._view_digest(view_id)}.json"

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def publish_view(self, view: QueryView) -> QueryViewRef:
        path = self._view_path(view.ref.schema_ref, view.ref.view_id)
        data = (_canonical(view.to_dict()) + "\n").encode()
        if path.exists():
            if path.read_bytes() != data:
                raise QueryViewError("view_id collision")
        else:
            self._atomic_write(path, data)
        pointer = self._schema_root(view.ref.schema_ref) / "current"
        self._atomic_write(pointer, f"{view.ref.view_id}\n".encode())
        return view.ref

    def get_view(self, view_id: str) -> QueryView | None:
        digest = self._view_digest(view_id)
        for path in self.root.glob(f"*/generations/{digest}.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise QueryViewError(f"cannot read Query View: {exc}") from exc
            if not isinstance(value, dict):
                raise QueryViewError("Query View document is not an object")
            return QueryView.from_dict(value)
        return None

    def current_view(self, schema_ref: str) -> QueryView | None:
        pointer = self._schema_root(schema_ref) / "current"
        try:
            view_id = pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise QueryViewError(f"cannot read current view pointer: {exc}") from exc
        view = self.get_view(view_id)
        if view is None or view.ref.schema_ref != schema_ref:
            raise QueryViewError("current view pointer does not resolve for schema_ref")
        return view


class QueryViewMaintainer:
    """Rebuild and atomically publish a query view from canonical storage."""

    def __init__(
        self,
        schemas: SchemaRegistry,
        experiences: ExperienceStore,
        views: QueryViewStore,
    ) -> None:
        self._schemas = schemas
        self._experiences = experiences
        self._views = views

    def rebuild(
        self,
        schema_ref: str,
        *,
        fuzzy_ready: bool = False,
        semantic_ready: bool = False,
    ) -> QueryViewRef:
        declaration = self._schemas.get_schema(schema_ref)
        if declaration is None:
            raise QueryViewError("cannot build a view for an unknown schema_ref")
        records = self._experiences.list_experiences(schema_ref)
        view = QueryViewBuilder().build(
            declaration,
            records,
            fuzzy_ready=fuzzy_ready,
            semantic_ready=semantic_ready,
        )
        return self._views.publish_view(view)


__all__ = [
    "ANNOTATION_BUILDER_VERSION",
    "GROUP_BUILDER_VERSION",
    "QUERY_VIEW_VERSION",
    "CapabilityState",
    "InMemoryQueryViewStore",
    "LocalQueryViewStore",
    "QueryView",
    "QueryViewBuilder",
    "QueryViewError",
    "QueryViewMaintainer",
    "QueryViewRef",
    "QueryViewStore",
    "RepeatAnnotations",
    "RepeatGroup",
    "RetrievalCapability",
    "repeat_group_key",
]
