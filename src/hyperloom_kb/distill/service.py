# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a user does with Distills through a KB: describe a schema, explore its Experiences, register forms, preview
a Distill, create and edit it, and read it back with the Experiences it rests on.

The author writes a draft: the scope, the claim, and optionally a check, edits to the group, and annotations. The KB
selects the group, classifies each member, and records everything a version holds, so a preview shows exactly what a
create or an update would store.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

from hyperloom_kb.database import Connection, Database, WriteSource, record_write
from hyperloom_kb.distill.drift import NEEDS_ATTENTION, Drift, EvidenceStatus, class_counts, drift
from hyperloom_kb.distill.explore import GroupView, classify, newest_first, representatives, select
from hyperloom_kb.distill.model import (
    DISTILL_KIND,
    OBJECTIVE_PATH,
    Derivation,
    DistillEntry,
    Evidence,
    EvidenceClass,
    EvidenceMember,
    EvidenceStats,
    FormDeclaration,
    Relations,
    Scope,
    new_entry_id,
    path_field,
    require_entry_id,
    require_form_ref,
)
from hyperloom_kb.distill.store import CREATED, UNCHANGED, CatalogRow, DistillStore
from hyperloom_kb.local_state import LocalState, UnknownStateItem
from hyperloom_kb.observability import Metrics, event
from hyperloom_kb.schema import (
    BASELINE,
    CATEGORIES,
    CHANGE,
    OUTCOME,
    SCALAR_KINDS,
    Experience,
    ExperienceDeclaration,
    FieldRole,
    FileRef,
    JsonValue,
    Provenance,
    SchemaValidationError,
    _list,
    _mapping,
    _reject_unknown,
    _required,
)

log = logging.getLogger(__name__)

PRODUCER = "hyperloom-kb"
MEMBER_LIMIT, MAX_MEMBER_LIMIT = 20, 500
EVIDENCE_PAGE, MAX_EVIDENCE_PAGE = 50, 500
REPRESENTATIVES, MAX_REPRESENTATIVES = 5, 50
MAX_GROUP_BY = 2
EVIDENCE_HISTOGRAM = "hyperloom_kb_distill_evidence_seconds"
_AUDIT = "audit"
_DRAFT_FIELDS = frozenset(
    {
        "schema_ref",
        "form_ref",
        "scope",
        "claim",
        "check",
        "add",
        "remove",
        "annotations",
        "representatives",
        "relations",
        "tags",
        "notes",
        "author",
    }
)


class DistillHost(Protocol):
    """What the Distill operations need of the Experience service that hosts them."""

    kb_id: str
    metrics: Metrics

    @property
    def code_digest(self) -> str: ...

    @property
    def schema_refs(self) -> tuple[str, ...]: ...

    def declaration_for(self, schema_ref: str) -> ExperienceDeclaration: ...

    def visible_experiences(self, schema_ref: str) -> tuple[Experience, ...]: ...

    def held(self, experience_id: str) -> Experience: ...

    def is_visible(self, experience: Experience) -> bool: ...

    def missing_files(self, refs: Iterable[FileRef]) -> tuple[str, ...]: ...

    def file_path(self, ref: FileRef) -> Path: ...

    def kb_state(self, schema_ref: str) -> str: ...

    def lock_schema(self, connection: Connection, schema_ref: str) -> None: ...


def bounded(value: Any, name: str, *, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise SchemaValidationError(f"{name} must be an integer from 1 to {maximum}")
    return value


@dataclass(frozen=True)
class Draft:
    """A Distill as its author writes it; the KB adds the group, the classes, and everything else a version records.

    ``add`` and ``remove`` edit the group the scope selects, and ``annotations`` classify members by hand, overriding
    ``check``.
    """

    form_ref: str
    scope: Scope
    claim: dict[str, Any]
    check: Scope | None = None
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    annotations: dict[str, EvidenceClass] = field(default_factory=dict)
    representatives: tuple[str, ...] = ()
    relations: Relations = field(default_factory=Relations)
    tags: tuple[str, ...] = ()
    notes: dict[str, str] = field(default_factory=dict)
    author: str = ""
    schema_ref: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> Draft:
        data = _mapping(value, "draft")
        _reject_unknown(data, _DRAFT_FIELDS, "draft")
        annotations: dict[str, EvidenceClass] = {}
        for experience_id, classification in _mapping(data.get("annotations", {}), "draft.annotations").items():
            try:
                annotations[experience_id] = EvidenceClass(classification)
            except ValueError as exc:
                raise SchemaValidationError(
                    f"draft.annotations.{experience_id} must be one of: {', '.join(c.value for c in EvidenceClass)}"
                ) from exc
        check = data.get("check")
        schema_ref = data.get("schema_ref")
        return cls(
            form_ref=require_form_ref(_required(data, "form_ref", "draft.form_ref"), "draft.form_ref"),
            scope=Scope.from_dict(data.get("scope", {}), "draft.scope"),
            claim=_mapping(data.get("claim", {}), "draft.claim"),
            check=None if check is None else Scope.from_dict(check, "draft.check"),
            add=tuple(_list(data.get("add", []), "draft.add")),
            remove=tuple(_list(data.get("remove", []), "draft.remove")),
            annotations=annotations,
            representatives=tuple(_list(data.get("representatives", []), "draft.representatives")),
            relations=Relations.from_dict(data.get("relations", {})),
            tags=tuple(_list(data.get("tags", []), "draft.tags")),
            notes=_mapping(data.get("notes", {}), "draft.notes"),
            author=data.get("author", ""),
            schema_ref=None if schema_ref is None else str(schema_ref),
        )


@dataclass(frozen=True)
class _Prepared:
    """A draft checked against its schema and form, with the group it selects and the evidence the KB drew from it."""

    schema_ref: str
    declaration: ExperienceDeclaration
    form: FormDeclaration
    draft: Draft
    population: int
    evidence: Evidence
    kb_state: str


def _draft_of(entry: DistillEntry, visible: Mapping[str, Experience]) -> dict[str, JsonValue]:
    """The draft that writes ``entry`` again over the KB as it is: its scope, claim, check, and everything else the
    author wrote, with the edits and annotations that still name a visible Experience, so an update of it freezes
    today's evidence under the same judgement."""

    evidence = entry.evidence
    classes = {member.experience_id: member.classification.value for member in evidence.members}
    return {
        "schema_ref": entry.schema_ref,
        "form_ref": entry.form_ref,
        "scope": entry.scope.to_dict(),
        "claim": entry.to_dict()["claim"],
        "check": None if evidence.check is None else evidence.check.to_dict(),
        "add": [experience_id for experience_id in evidence.added if experience_id in visible],
        "remove": [experience_id for experience_id in evidence.removed if experience_id in visible],
        "annotations": {
            experience_id: classes[experience_id] for experience_id in evidence.annotated if experience_id in visible
        },
        "representatives": [experience_id for experience_id in evidence.representatives if experience_id in visible],
        "relations": entry.relations.to_dict(),
        "tags": list(entry.tags),
        "notes": dict(entry.notes),
        "author": entry.author,
    }


@dataclass(frozen=True)
class Check:
    """What the last pass over a schema's visible Distills found: how many stand in each evidence status."""

    at: datetime
    counts: dict[EvidenceStatus, int]

    @property
    def needs_attention(self) -> int:
        return sum(count for status, count in self.counts.items() if status in NEEDS_ATTENTION)


class DistillService:
    """The Distill operations of one KB, over its ``store``, its schemas' ``state``, and the service that hosts them.

    Evidence drift is computed when a listing, a read, or the changes inbox asks for it; ``checks`` keeps what each
    schema's last full pass found, for the health report and the metrics a scrape samples without computing.
    """

    def __init__(self, host: DistillHost, database: Database, state: LocalState, store: DistillStore) -> None:
        self._host = host
        self._database = database
        self._state = state
        self._store = store
        self._checks_lock = threading.Lock()
        self._checks: dict[str, Check] = {}

    # Describing and exploring a schema.

    def describe_schema(self, schema_ref: str) -> dict[str, JsonValue]:
        """The schema's declaration, the paths a scope or a grouping may name, its roles, and what it holds."""

        declaration = self._host.declaration_for(schema_ref)
        visible = self._host.visible_experiences(schema_ref)
        with self._database.transaction() as connection:
            stored = len(self._state.stored(connection, schema_ref))
            members = self._state.entry_members(connection, schema_ref)
            deleted = members & self._state.excluded(connection, schema_ref)
            forms = self._store.forms(connection, schema_ref)

        def role(category: str, field_role: FieldRole) -> str | None:
            declared = declaration.role_field(category, field_role)
            return None if declared is None else f"{category}.{declared.name}"

        paths = [OBJECTIVE_PATH] + sorted(
            f"{category}.{item.name}"
            for category in CATEGORIES
            for item in declaration.fields(category)
            if item.kind in SCALAR_KINDS and not item.many
        )
        return {
            "schema_ref": schema_ref,
            "declaration": declaration.to_dict(),
            "paths": paths,
            "roles": {
                "decision": role(OUTCOME, FieldRole.DECISION),
                "baseline_measurement": role(BASELINE, FieldRole.MEASUREMENT),
                "outcome_measurement": role(OUTCOME, FieldRole.MEASUREMENT),
                "summary": role(CHANGE, FieldRole.SUMMARY),
                "group": [f"{category}.{name}" for category, name in declaration.group_fields()],
            },
            "experiences": {"stored": stored, "visible": len(visible)},
            "distills": {"visible": len(members - deleted), "deleted": len(deleted)},
            "forms": [{"name": form.name, "form_ref": form.form_ref} for form in forms],
        }

    def explore(
        self, schema_ref: str, scope: Scope, group_by: tuple[str, ...] = (), member_limit: int | None = None
    ) -> dict[str, JsonValue]:
        """The group ``scope`` selects among the schema's visible Experiences, and its cross table over ``group_by``."""

        declaration = self._host.declaration_for(schema_ref)
        scope.validate(declaration, "scope")
        if len(group_by) > MAX_GROUP_BY or len(set(group_by)) != len(group_by):
            raise SchemaValidationError(f"group_by names at most {MAX_GROUP_BY} distinct paths")
        for path in group_by:
            path_field(declaration, path, "group_by")
        limit = bounded(member_limit, "member_limit", default=MEMBER_LIMIT, maximum=MAX_MEMBER_LIMIT)
        view = GroupView(declaration)
        with self._host.metrics.timed(EVIDENCE_HISTOGRAM, operation="explore"):
            members = newest_first(select(self._host.visible_experiences(schema_ref), scope))
            result: dict[str, JsonValue] = {
                "schema_ref": schema_ref,
                "scope": scope.to_dict(),
                **view.summarize(members, limit),
            }
            if group_by:
                result["group_by"] = list(group_by)
                result["cells"] = view.cross(members, group_by, limit)  # type: ignore[assignment]
        return result

    # Forms.

    def forms(self, schema_ref: str) -> dict[str, JsonValue]:
        self._host.declaration_for(schema_ref)
        with self._database.transaction() as connection:
            forms = self._store.forms(connection, schema_ref)
        return {
            "schema_ref": schema_ref,
            "forms": [{"name": form.name, "form_ref": form.form_ref, "form": form.to_dict()} for form in forms],
        }

    def register_form(self, schema_ref: str, form: FormDeclaration) -> dict[str, JsonValue]:
        self._host.declaration_for(schema_ref)
        with self._database.transaction() as connection:
            created = self._store.register_form(connection, schema_ref, form)
        if created:
            event(log, _AUDIT, action="register_distill_form", schema_ref=schema_ref, form_ref=form.form_ref)
        return {
            "schema_ref": schema_ref,
            "form_ref": form.form_ref,
            "name": form.name,
            "status": CREATED if created else UNCHANGED,
        }

    # Writing.

    def _schema_for(self, draft: Draft, entry_id: str | None) -> str:
        if entry_id is None:
            if draft.schema_ref is None:
                raise SchemaValidationError("draft.schema_ref is required to create a Distill")
            return draft.schema_ref
        with self._database.transaction() as connection:
            schema_ref = self._store.schema_of(connection, require_entry_id(entry_id))
        if draft.schema_ref is not None and draft.schema_ref != schema_ref:
            raise SchemaValidationError(f"{entry_id} is a Distill of {schema_ref}, not {draft.schema_ref}")
        return schema_ref

    def _evidence(self, visible: tuple[Experience, ...], population: list[Experience], draft: Draft) -> Evidence:
        in_group = {experience.id for experience in population}
        outside_group = sorted(set(draft.remove) - in_group)
        if outside_group:
            raise SchemaValidationError(
                f"remove names Experiences outside the scope's group: {', '.join(outside_group)}"
            )
        by_id = {experience.id: experience for experience in visible}
        added: list[Experience] = []
        for experience_id in draft.add:
            if experience_id in in_group:
                raise SchemaValidationError(f"add names {experience_id}, which the scope already selects")
            if experience_id not in by_id:
                raise SchemaValidationError(f"add names {experience_id}, which is no visible Experience of this schema")
            added.append(by_id[experience_id])
        removed = set(draft.remove)
        members = [experience for experience in population if experience.id not in removed] + added
        strays = sorted(set(draft.annotations) - {experience.id for experience in members})
        if strays:
            raise SchemaValidationError(f"annotations name Experiences that are not members: {', '.join(strays)}")
        classified = classify(members, draft.check, draft.annotations)
        counts = Counter(classification for _, classification in classified)
        return Evidence(
            members=tuple(EvidenceMember(experience.id, classification) for experience, classification in classified),
            stats=EvidenceStats(
                counts[EvidenceClass.SUPPORT],
                counts[EvidenceClass.CONTRADICT],
                counts[EvidenceClass.IRRELEVANT],
                len({experience.run_id for experience in members}),
            ),
            check=draft.check,
            added=draft.add,
            removed=draft.remove,
            representatives=draft.representatives,
            annotated=tuple(sorted(draft.annotations)),
        )

    def _prepare(self, schema_ref: str, draft: Draft, operation: str) -> _Prepared:
        declaration = self._host.declaration_for(schema_ref)
        with self._database.transaction() as connection:
            form = self._store.form(connection, schema_ref, draft.form_ref)
            for relation in draft.relations.entries:
                if self._store.schema_of(connection, relation.target) != schema_ref:
                    raise SchemaValidationError(f"relation target {relation.target} is a Distill of another schema")
        draft.scope.validate(declaration, "scope")
        if draft.check is not None:
            draft.check.validate(declaration, "check")
        with self._host.metrics.timed(EVIDENCE_HISTOGRAM, operation=operation):
            visible = self._host.visible_experiences(schema_ref)
            population = newest_first(select(visible, draft.scope))
            evidence = self._evidence(visible, population, draft)
        return _Prepared(
            schema_ref, declaration, form, draft, len(population), evidence, self._host.kb_state(schema_ref)
        )

    def _assemble(self, prepared: _Prepared, entry_id: str, version: int) -> DistillEntry:
        draft = prepared.draft
        entry = DistillEntry(
            id=entry_id,
            version=version,
            schema_ref=prepared.schema_ref,
            form_ref=prepared.form.form_ref,
            created_at=datetime.now(timezone.utc),
            provenance=Provenance(PRODUCER, self._host.code_digest),
            scope=draft.scope,
            claim=draft.claim,
            evidence=prepared.evidence,
            author=draft.author,
            derivation=Derivation(prepared.kb_state),
            relations=draft.relations,
            tags=draft.tags,
            notes=draft.notes,
        )
        self._check_entry(entry, prepared.form, prepared.declaration)
        return entry

    def _check_entry(self, entry: DistillEntry, form: FormDeclaration, declaration: ExperienceDeclaration) -> None:
        entry.validate(form, declaration)
        missing = self._host.missing_files(entry.files())
        if missing:
            raise SchemaValidationError(f"the claim names files this KB does not hold: {', '.join(missing)}")

    def _overlaps(self, schema_ref: str, scope: Scope, exclude: str | None) -> list[JsonValue]:
        with self._database.transaction() as connection:
            visible = self._state.visible_entries(connection, schema_ref)
            rows = self._store.catalog(connection, schema_ref)
        return [
            {"entry_id": row.entry_id, "version": row.version, "form_name": row.form_name, "summary": row.summary}
            for row in rows
            if row.entry_id in visible and row.entry_id != exclude and row.scope.overlaps(scope)
        ]

    def preview(self, value: Any, *, entry_id: str | None = None) -> dict[str, JsonValue]:
        """What a create of the draft ``value``, or with ``entry_id`` an update, would store, without storing anything.

        A draft the KB would refuse answers ``valid: false`` with the reason instead of failing.
        """

        try:
            draft = Draft.from_dict(value)
            schema_ref = self._schema_for(draft, entry_id)
            prepared = self._prepare(schema_ref, draft, "preview")
            entry = self._assemble(prepared, entry_id or new_entry_id(), 1)
        except (ValueError, UnknownStateItem) as exc:
            return {"valid": False, "errors": [str(exc)]}
        return {
            "valid": True,
            "errors": [],
            "schema_ref": schema_ref,
            "form_ref": entry.form_ref,
            "population": prepared.population,
            "evidence": entry.evidence.stats.to_dict(),
            "knowledge": entry.knowledge(),
            "overlaps": self._overlaps(schema_ref, draft.scope, entry_id),
        }

    def _write(
        self, connection: Connection, entry: DistillEntry, form: FormDeclaration, source: WriteSource
    ) -> dict[str, JsonValue]:
        status = self._store.store(connection, entry, form)
        record_write(
            connection,
            self._host.kb_id,
            record_id=entry.id,
            schema_ref=entry.schema_ref,
            result=status,
            source=source,
            record_kind=DISTILL_KIND,
        )
        return {
            "status": status,
            "entry_id": entry.id,
            "version": entry.version,
            "evidence": entry.evidence.stats.to_dict(),
        }

    def _written(self, result: dict[str, JsonValue], operation: str, schema_ref: str) -> dict[str, JsonValue]:
        self._host.metrics.count("hyperloom_kb_distill_writes_total", operation=operation, result=str(result["status"]))
        event(
            log,
            _AUDIT,
            action=f"distill_{operation}",
            schema_ref=schema_ref,
            entry_id=result["entry_id"],
            version=result["version"],
            result=result["status"],
        )
        return result

    def create(self, value: Any, *, source: WriteSource = WriteSource()) -> dict[str, JsonValue]:
        """Store the draft ``value`` as version 1 of a new Distill."""

        draft = Draft.from_dict(value)
        schema_ref = self._schema_for(draft, None)
        prepared = self._prepare(schema_ref, draft, "create")
        entry = self._assemble(prepared, new_entry_id(), 1)
        with self._database.transaction() as connection:
            self._host.lock_schema(connection, schema_ref)
            result = self._write(connection, entry, prepared.form, source)
        return self._written(result, "create", schema_ref)

    def update(self, entry_id: str, value: Any, *, source: WriteSource = WriteSource()) -> dict[str, JsonValue]:
        """Add a version to ``entry_id``: the draft ``value`` in full, its evidence drawn again from the KB as it is."""

        draft = Draft.from_dict(value)
        schema_ref = self._schema_for(draft, entry_id)
        prepared = self._prepare(schema_ref, draft, "update")
        with self._database.transaction() as connection:
            version = self._store.latest_version(connection, entry_id) + 1
        # An update that took this version first makes the store refuse this one as a conflict.
        entry = self._assemble(prepared, entry_id, version)
        with self._database.transaction() as connection:
            self._host.lock_schema(connection, schema_ref)
            result = self._write(connection, entry, prepared.form, source)
        return self._written(result, "update", schema_ref)

    # Syncing with another KB.

    def receive(
        self, entry: DistillEntry, form: FormDeclaration, *, source: WriteSource = WriteSource()
    ) -> dict[str, JsonValue]:
        """Store a version another KB wrote, exactly as it wrote it: its evidence is that KB's snapshot."""

        declaration = self._host.declaration_for(entry.schema_ref)
        self._check_entry(entry, form, declaration)
        with self._database.transaction() as connection:
            self._host.lock_schema(connection, entry.schema_ref)
            result = self._write(connection, entry, form, source)
        return self._written(result, "receive", entry.schema_ref)

    def pushable(self, schema_ref: str) -> list[tuple[str, int]]:
        """Each visible Distill of ``schema_ref`` that shows its latest version, with that version: a sync sends only
        latest versions, so one a restore set back to an older version waits until it shows its latest again."""

        with self._database.transaction() as connection:
            visible = self._state.visible_entries(connection, schema_ref)
            shown = self._state.current_versions(connection, schema_ref)
            return sorted(
                (entry_id, version)
                for entry_id, version in shown.items()
                if entry_id in visible and version == self._store.latest_version(connection, entry_id)
            )

    def version(self, entry_id: str, version: int) -> tuple[DistillEntry, FormDeclaration]:
        """Version ``version`` of ``entry_id`` and the form it is written in."""

        with self._database.transaction() as connection:
            entry = self._store.read(connection, entry_id, version)
            return entry, self._store.form(connection, entry.schema_ref, entry.form_ref)

    def latest_version(self, entry_id: str) -> int:
        with self._database.transaction() as connection:
            return self._store.latest_version(connection, entry_id)

    def show_at_least(self, schema_ref: str, entry_id: str, version: int) -> None:
        """Make ``entry_id`` show ``version`` when it shows an older one."""

        with self._database.transaction() as connection:
            self._host.lock_schema(connection, schema_ref)
            if self._state.current_versions(connection, schema_ref).get(entry_id, 0) < version:
                self._state.set_current(connection, entry_id, version)

    # Reading.

    def get(
        self,
        entry_id: str,
        *,
        version: int | None = None,
        follow_evidence: bool = False,
        after: int = 0,
        limit: int | None = None,
        representative_limit: int | None = None,
    ) -> dict[str, JsonValue]:
        """One version of an entry, the one it shows by default: how its evidence moved, the visible members that
        represent it, and with ``follow_evidence`` a page of the Experiences it rests on."""

        require_entry_id(entry_id)
        shown = bounded(representative_limit, "representatives", default=REPRESENTATIVES, maximum=MAX_REPRESENTATIVES)
        with self._database.transaction() as connection:
            schema_ref = self._store.schema_of(connection, entry_id)
            entry = self._store.read(connection, entry_id, version)
            versions = self._store.versions(connection, entry_id)
            current = self._state.current_versions(connection, schema_ref).get(entry_id)
            deleted = entry_id in self._state.excluded(connection, schema_ref)
            outside = entry_id not in self._state.entry_members(connection, schema_ref)
        with self._host.metrics.timed(EVIDENCE_HISTOGRAM, operation="get"):
            visible = self._visible(schema_ref)
            found = drift(entry, visible)
        members = [
            (visible[member.experience_id], member.classification)
            for member in entry.evidence.members
            if member.experience_id in visible
        ]
        result: dict[str, JsonValue] = {
            "entry": entry.to_dict(),
            "current_version": current,
            "versions": list(versions),
            "deleted": deleted,
            "outside": outside,
            "drift": self._detail(found, GroupView(self._host.declaration_for(schema_ref))),
            "representatives": representatives(members, entry.evidence.representatives, shown),
            "draft": _draft_of(entry, visible),
        }
        if follow_evidence:
            result["evidence"] = self._evidence_page(entry, after, limit)
        return result

    def _evidence_page(self, entry: DistillEntry, after: int, limit: int | None) -> dict[str, JsonValue]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise SchemaValidationError("after must be a non-negative integer")
        size = bounded(limit, "limit", default=EVIDENCE_PAGE, maximum=MAX_EVIDENCE_PAGE)
        members = entry.evidence.members[after : after + size]
        items: list[JsonValue] = []
        for member in members:
            item: dict[str, JsonValue] = {"id": member.experience_id, "class": member.classification.value}
            try:
                experience = self._host.held(member.experience_id)
            except UnknownStateItem:
                item["held"] = False
            else:
                item["held"] = True
                item["visible"] = self._host.is_visible(experience)
                item["experience"] = experience.knowledge()
                item["files"] = {ref.sha256: str(self._host.file_path(ref)) for ref in experience.files()}
            items.append(item)
        following = after + len(members)
        return {
            "items": items,
            "next": following,
            "has_more": following < len(entry.evidence.members),
            "total": len(entry.evidence.members),
        }

    def _visible(self, schema_ref: str) -> dict[str, Experience]:
        return {experience.id: experience for experience in self._host.visible_experiences(schema_ref)}

    def _collection(self, schema_ref: str, *, include_deleted: bool) -> list[tuple[CatalogRow, DistillEntry, bool]]:
        """Each Distill of ``schema_ref`` in its state, newest first, at the version it shows, and whether it is
        deleted."""

        with self._database.transaction() as connection:
            members = self._state.entry_members(connection, schema_ref)
            excluded = self._state.excluded(connection, schema_ref)
            return [
                (row, self._store.read(connection, row.entry_id, row.version), row.entry_id in excluded)
                for row in self._store.catalog(connection, schema_ref)
                if row.entry_id in members and (include_deleted or row.entry_id not in excluded)
            ]

    def _check(self, schema_ref: str, collection: list[tuple[CatalogRow, DistillEntry, bool]]) -> list[Drift]:
        """The drift of each of ``collection``, and what it says of the visible ones kept as the schema's check."""

        drifts: list[Drift] = []
        if collection:
            with self._host.metrics.timed(EVIDENCE_HISTOGRAM, operation="check"):
                visible = self._visible(schema_ref)
                drifts = [drift(entry, visible) for _, entry, _ in collection]
        counts = Counter(found.status for (_, _, deleted), found in zip(collection, drifts) if not deleted)
        check = Check(datetime.now(timezone.utc), {status: counts[status] for status in EvidenceStatus})
        with self._checks_lock:
            self._checks[schema_ref] = check
        return drifts

    def _detail(self, found: Drift, view: GroupView) -> dict[str, JsonValue]:
        """``found``'s status and counts, with the members a user should look at: the first ``MEMBER_LIMIT`` of each
        kind, newest first."""

        def described(experience: Experience) -> JsonValue:
            return {"id": experience.id, "summary": view.summary(experience)}

        def lost(experience_id: str) -> JsonValue:
            try:
                return described(self._host.held(experience_id))
            except UnknownStateItem:
                return {"id": experience_id, "summary": None}

        def first(cls: EvidenceClass | None) -> list[JsonValue]:
            return list(islice((described(e) for e, judged in found.gained if judged is cls), MEMBER_LIMIT))

        lost_supporting = (member for member, cls in found.lost if cls is EvidenceClass.SUPPORT)
        return {
            **found.summary(),
            "gained": class_counts(cls for _, cls in found.gained),
            "lost": class_counts(cls for _, cls in found.lost),
            "gained_contradicting": first(EvidenceClass.CONTRADICT),
            "gained_undetermined": first(None),
            "lost_supporting": [lost(member) for member in islice(lost_supporting, MEMBER_LIMIT)],
        }

    def list(self, schema_ref: str, *, include_deleted: bool = False) -> dict[str, JsonValue]:
        """The Distills of ``schema_ref`` in its current state, newest first, each with how its evidence stands now;
        deleted ones only on request."""

        self._host.declaration_for(schema_ref)
        collection = self._collection(schema_ref, include_deleted=include_deleted)
        drifts = self._check(schema_ref, collection)
        items: list[JsonValue] = [
            {**row.to_dict(), "state": "deleted" if deleted else "visible", **found.summary()}
            for (row, _, deleted), found in zip(collection, drifts, strict=True)
        ]
        return {"schema_ref": schema_ref, "items": items}

    def changes(self, schema_ref: str | None = None) -> dict[str, JsonValue]:
        """Every visible Distill of ``schema_ref``, or of every schema, whose evidence moved in a way its author should
        look at, with what moved; and how many of each schema's Distills stand in each status."""

        schemas = self._host.schema_refs if schema_ref is None else (self._host.declaration_for(schema_ref).schema_ref,)
        items: list[JsonValue] = []
        checked: dict[str, JsonValue] = {}
        for schema in schemas:
            view = GroupView(self._host.declaration_for(schema))
            collection = self._collection(schema, include_deleted=False)
            drifts = self._check(schema, collection)
            for (row, _, _), found in zip(collection, drifts, strict=True):
                if found.needs_attention:
                    items.append({**row.to_dict(), **self._detail(found, view)})
            counts = Counter(found.status for found in drifts)
            checked[schema] = {status.value: counts[status] for status in EvidenceStatus}
        return {"items": items, "checked": checked}

    def checks(self) -> dict[str, Check]:
        """What the last listing or changes pass over each schema found, for reports that must not compute it."""

        with self._checks_lock:
            return dict(self._checks)


__all__ = ["EVIDENCE_HISTOGRAM", "PRODUCER", "Check", "DistillHost", "DistillService", "Draft"]
