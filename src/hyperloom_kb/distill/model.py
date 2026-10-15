# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Distill: a pattern a user drew from a group of one schema's Experiences, kept as versions of one entry.

An entry names the conditions it holds under (its scope), what it concludes (its claim, whose fields a form declares),
and the Experiences that bear on it (its evidence), each classified as supporting, contradicting, or irrelevant. Every
version is immutable; editing an entry adds a version under the same id.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from hyperloom_kb.schema import (
    CATEGORIES,
    SCALAR_KINDS,
    Experience,
    ExperienceDeclaration,
    FieldDeclaration,
    FieldRole,
    FieldValue,
    FileRef,
    JsonValue,
    Provenance,
    SchemaValidationError,
    _aware_datetime,
    _encode,
    _field_values,
    _files,
    _format_datetime,
    _list,
    _mapping,
    _notes,
    _parse_datetime,
    _reject_unknown,
    _require_name,
    _require_schema_ref,
    _require_string,
    _require_token,
    _required,
    _single_value,
)

DISTILL_KIND = "distill"
FORM_VERSION = 1
#: A condition on the objective an Experience pursues; every other path is ``<category>.<field>``.
OBJECTIVE_PATH = "objective"
_ENTRY_ID_RE = re.compile(r"^dst-[0-9a-f]{32}$")
_FORM_REF_RE = re.compile(r"^form:sha256:[0-9a-f]{64}$")
_NAME, _DESCRIPTION, _CLAIM = "name", "description", "claim"
_SCOPE, _EVIDENCE, _RELATIONS, _DERIVATION = "scope", "evidence", "relations", "derivation"
_KIND, _VERSION, _SCHEMA_REF, _FORM_REF, _CREATED_AT = "kind", "version", "schema_ref", "form_ref", "created_at"
_MEMBERS, _TARGET, _URI = "members", "target", "uri"
_STATS = ("supports", "contradicts", "irrelevant", "runs")
_CREATED_AT_WHERE = "distill.created_at"

Scalar = str | int | float | bool


def new_entry_id() -> str:
    return f"dst-{uuid.uuid4().hex}"


def require_entry_id(value: Any, name: str = "entry id") -> str:
    if not isinstance(value, str) or not _ENTRY_ID_RE.fullmatch(value):
        raise SchemaValidationError(f"{name} must be dst-<32 lowercase hex>")
    return value


def require_form_ref(value: Any, name: str = _FORM_REF) -> str:
    if not isinstance(value, str) or not _FORM_REF_RE.fullmatch(value):
        raise SchemaValidationError(f"{name} must be form:sha256:<64 lowercase hex>")
    return value


def _canonical(value: JsonValue) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaValidationError(f"{name} must be an integer of at least {minimum}")
    return value


def _same(left: Scalar, right: Scalar) -> bool:
    """Equal as a condition reads it: a boolean equals only a boolean, and numbers compare by value."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _unique(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise SchemaValidationError(f"{name} contains duplicates")
    return values


def path_field(declaration: ExperienceDeclaration, path: str, where: str) -> FieldDeclaration | None:
    """The declared single-scalar field ``path`` names, ``None`` for the objective; any other path is refused."""

    if path == OBJECTIVE_PATH:
        return None
    category, _, name = path.partition(".")
    declared = declaration.field(category, name) if category in CATEGORIES else None
    if declared is None:
        raise SchemaValidationError(f"{where}: {path} is not a declared field")
    if declared.kind not in SCALAR_KINDS or declared.many:
        raise SchemaValidationError(f"{where}: {path} does not hold one scalar")
    return declared


def value_at(experience: Experience, path: str) -> Scalar | None:
    """The scalar ``experience`` holds at ``path``, ``None`` when it holds none there."""

    if path == OBJECTIVE_PATH:
        return experience.objective
    category, _, name = path.partition(".")
    value = getattr(experience, category).get(name)
    return None if value is None or isinstance(value, (tuple, FileRef)) else value


class EvidenceClass(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    IRRELEVANT = "irrelevant"


class RelationKind(str, Enum):
    REFINES = "refines"
    RELATED = "related"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True)
class Condition:
    """One field an Experience must hold one of ``values`` in; ``path`` is ``objective`` or ``<category>.<field>``."""

    path: str
    values: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        path = _require_string(self.path, "condition path")
        if path != OBJECTIVE_PATH:
            category, _, name = path.partition(".")
            if category not in CATEGORIES:
                raise SchemaValidationError(f"condition path {path!r} must be objective or <category>.<field>")
            _require_name(name, f"condition path {path!r} field")
        object.__setattr__(self, "path", path)
        values: list[Scalar] = []
        for item in tuple(self.values):
            value = _single_value(item, f"condition {path} value")
            if isinstance(value, FileRef) or value == "":
                raise SchemaValidationError(f"condition {path} values must be non-empty scalars")
            if any(_same(value, seen) for seen in values):
                raise SchemaValidationError(f"condition {path} lists a value twice")
            values.append(value)
        if not values:
            raise SchemaValidationError(f"condition {path} names no value")
        object.__setattr__(self, "values", tuple(values))

    def validate(self, declaration: ExperienceDeclaration, where: str) -> None:
        """Check that ``declaration`` declares the field as one scalar and that every value fits its kind."""

        declared = path_field(declaration, self.path, where)
        for value in self.values:
            if declared is None:
                if not isinstance(value, str) or declaration.objective(value) is None:
                    raise SchemaValidationError(f"{where}: objective {value!r} is not declared")
            else:
                declared.validate(value, f"{where}.{self.path}")

    def holds(self, experience: Experience) -> bool | None:
        """Whether ``experience`` holds one of the values; ``None`` when it has no value at this path."""

        value = value_at(experience, self.path)
        if value is None:
            return None
        return any(_same(value, wanted) for wanted in self.values)

    def admits(self, other: Condition) -> bool:
        """Whether an Experience could meet both this condition and ``other`` on the same path."""

        return any(_same(mine, theirs) for mine in self.values for theirs in other.values)


@dataclass(frozen=True)
class Scope:
    """Conditions an Experience must all meet; with none, every Experience of the schema meets it."""

    conditions: tuple[Condition, ...] = ()

    def __post_init__(self) -> None:
        conditions = tuple(sorted(self.conditions, key=lambda item: item.path))
        if not all(isinstance(item, Condition) for item in conditions):
            raise SchemaValidationError("scope contains an invalid condition")
        _unique(tuple(item.path for item in conditions), "scope paths")
        object.__setattr__(self, "conditions", conditions)

    def validate(self, declaration: ExperienceDeclaration, where: str = _SCOPE) -> None:
        for condition in self.conditions:
            condition.validate(declaration, where)

    def holds(self, experience: Experience) -> bool | None:
        """``True`` when every condition holds, ``False`` when one does not, ``None`` when one has no value."""

        results = [condition.holds(experience) for condition in self.conditions]
        if False in results:
            return False
        return None if None in results else True

    def overlaps(self, other: Scope) -> bool:
        """Whether one Experience could meet both scopes: every path both name admits a common value."""

        theirs = {condition.path: condition for condition in other.conditions}
        return all(
            condition.admits(theirs[condition.path]) for condition in self.conditions if condition.path in theirs
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {condition.path: list(condition.values) for condition in self.conditions}

    @classmethod
    def from_dict(cls, value: Any, name: str = _SCOPE) -> Scope:
        data = _mapping(value, name)
        return cls(
            tuple(
                Condition(path, tuple(values) if isinstance(values, list) else (values,))
                for path, values in data.items()
            )
        )


@dataclass(frozen=True)
class FormDeclaration:
    """The fields of a claim in one form, such as a rule or a parameter curve; ``name`` groups its versions."""

    name: str
    description: str
    claim: tuple[FieldDeclaration, ...]
    form_version: int = FORM_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "form.name"))
        object.__setattr__(self, "description", _require_string(self.description, f"form {self.name!r} description"))
        if self.form_version != FORM_VERSION:
            raise SchemaValidationError(f"unsupported form_version {self.form_version!r}; expected {FORM_VERSION}")
        claim = tuple(self.claim)
        if not claim or not all(isinstance(item, FieldDeclaration) for item in claim):
            raise SchemaValidationError(f"form {self.name!r} must declare at least one claim field")
        _unique(tuple(item.name for item in claim), f"form {self.name!r} claim fields")
        for item in claim:
            if item.group:
                raise SchemaValidationError(f"form {self.name!r}: claim field {item.name!r} cannot join a group")
            if item.role not in (None, FieldRole.SUMMARY):
                raise SchemaValidationError(f"form {self.name!r}: a claim field takes only the summary role")
        if Counter(item.role for item in claim)[FieldRole.SUMMARY] > 1:
            raise SchemaValidationError(f"form {self.name!r} gives the summary role to several fields")
        object.__setattr__(self, "claim", claim)

    @property
    def form_ref(self) -> str:
        return f"form:sha256:{hashlib.sha256(_canonical(self.to_dict())).hexdigest()}"

    @property
    def summary_field(self) -> FieldDeclaration | None:
        return next((item for item in self.claim if item.role is FieldRole.SUMMARY), None)

    def validate_claim(self, claim: dict[str, FieldValue], where: str = _CLAIM) -> None:
        declared = {item.name: item for item in self.claim}
        unknown = sorted(set(claim) - set(declared))
        if unknown:
            raise SchemaValidationError(
                f"{where} holds fields form {self.name!r} does not declare: {', '.join(unknown)}"
            )
        for name, item in declared.items():
            if name in claim:
                item.validate(claim[name], f"{where}.{name}")
            elif item.required:
                raise SchemaValidationError(f"{where} is missing required field {name!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "form_version": self.form_version,
            "name": self.name,
            "description": self.description,
            "claim": [item.to_dict() for item in self.claim],
        }

    @classmethod
    def from_dict(cls, value: Any) -> FormDeclaration:
        data = _mapping(value, "form")
        _reject_unknown(data, {"form_version", _NAME, _DESCRIPTION, _CLAIM}, "form")
        return cls(
            form_version=data.get("form_version", FORM_VERSION),
            name=_required(data, _NAME, "form.name"),
            description=_required(data, _DESCRIPTION, "form.description"),
            claim=tuple(FieldDeclaration.from_dict(item) for item in _list(data.get("claim", []), "form.claim")),
        )


@dataclass(frozen=True)
class EvidenceMember:
    experience_id: str
    classification: EvidenceClass

    def __post_init__(self) -> None:
        object.__setattr__(self, "experience_id", _require_token(self.experience_id, "evidence member id"))
        if not isinstance(self.classification, EvidenceClass):
            raise SchemaValidationError(f"evidence member {self.experience_id} has an invalid classification")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"id": self.experience_id, "class": self.classification.value}

    @classmethod
    def from_dict(cls, value: Any) -> EvidenceMember:
        data = _mapping(value, "evidence member")
        _reject_unknown(data, {"id", "class"}, "evidence member")
        try:
            classification = EvidenceClass(_required(data, "class", "evidence member class"))
        except ValueError as exc:
            raise SchemaValidationError(
                f"evidence member class must be one of: {', '.join(c.value for c in EvidenceClass)}"
            ) from exc
        return cls(_required(data, "id", "evidence member id"), classification)


@dataclass(frozen=True)
class EvidenceStats:
    """What the evidence held when the version was written: members of each class, and the runs they came from."""

    supports: int
    contradicts: int
    irrelevant: int
    runs: int

    def __post_init__(self) -> None:
        for name in _STATS:
            _int(getattr(self, name), f"evidence stats {name}", minimum=0)

    def to_dict(self) -> dict[str, JsonValue]:
        return {name: getattr(self, name) for name in _STATS}

    @classmethod
    def from_dict(cls, value: Any) -> EvidenceStats:
        data = _mapping(value, "evidence stats")
        _reject_unknown(data, set(_STATS), "evidence stats")
        return cls(**{name: _required(data, name, f"evidence stats {name}") for name in _STATS})


@dataclass(frozen=True)
class Evidence:
    """The Experiences an entry rests on, as its version froze them.

    By default the members are every Experience the scope selected; ``added`` and ``removed`` record where the author
    departed from that. ``check``, when given, is the condition that classified each member automatically, and
    ``annotated`` names the members whose class the author set instead.
    """

    members: tuple[EvidenceMember, ...]
    stats: EvidenceStats
    check: Scope | None = None
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    representatives: tuple[str, ...] = ()
    annotated: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if not members or not all(isinstance(item, EvidenceMember) for item in members):
            raise SchemaValidationError("evidence must name at least one Experience")
        ids = _unique(tuple(item.experience_id for item in members), "evidence members")
        object.__setattr__(self, "members", members)
        if not isinstance(self.stats, EvidenceStats):
            raise SchemaValidationError("evidence stats are invalid")
        counts = Counter(item.classification for item in members)
        if (self.stats.supports, self.stats.contradicts, self.stats.irrelevant) != (
            counts[EvidenceClass.SUPPORT],
            counts[EvidenceClass.CONTRADICT],
            counts[EvidenceClass.IRRELEVANT],
        ):
            raise SchemaValidationError("evidence stats do not count the members' classes")
        if not 1 <= self.stats.runs <= len(members):
            raise SchemaValidationError("evidence stats runs must be between 1 and the number of members")
        if self.check is not None and not isinstance(self.check, Scope):
            raise SchemaValidationError("evidence check is invalid")
        for name in ("added", "removed", "representatives", "annotated"):
            values = _unique(
                tuple(_require_token(item, f"evidence {name} id") for item in getattr(self, name)), f"evidence {name}"
            )
            object.__setattr__(self, name, values)
        member_ids = set(ids)
        if not set(self.added) <= member_ids:
            raise SchemaValidationError("evidence added names an Experience that is not a member")
        if set(self.removed) & member_ids:
            raise SchemaValidationError("evidence removed names an Experience that is still a member")
        if not set(self.representatives) <= member_ids:
            raise SchemaValidationError("evidence representatives must be members")
        if not set(self.annotated) <= member_ids:
            raise SchemaValidationError("evidence annotated names an Experience that is not a member")

    def ids(self) -> tuple[str, ...]:
        return tuple(item.experience_id for item in self.members)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "members": [item.to_dict() for item in self.members],
            "stats": self.stats.to_dict(),
            "check": None if self.check is None else self.check.to_dict(),
            "added": list(self.added),
            "removed": list(self.removed),
            "representatives": list(self.representatives),
            "annotated": list(self.annotated),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Evidence:
        data = _mapping(value, "evidence")
        _reject_unknown(
            data, {_MEMBERS, "stats", "check", "added", "removed", "representatives", "annotated"}, _EVIDENCE
        )
        check = data.get("check")
        return cls(
            members=tuple(
                EvidenceMember.from_dict(item)
                for item in _list(_required(data, _MEMBERS, "evidence.members"), "evidence.members")
            ),
            stats=EvidenceStats.from_dict(_required(data, "stats", "evidence.stats")),
            check=None if check is None else Scope.from_dict(check, "evidence.check"),
            added=tuple(_list(data.get("added", []), "evidence.added")),
            removed=tuple(_list(data.get("removed", []), "evidence.removed")),
            representatives=tuple(_list(data.get("representatives", []), "evidence.representatives")),
            annotated=tuple(_list(data.get("annotated", []), "evidence.annotated")),
        )


@dataclass(frozen=True)
class Relation:
    kind: RelationKind
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RelationKind):
            raise SchemaValidationError("relation kind is invalid")
        object.__setattr__(self, "target", require_entry_id(self.target, "relation target"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"kind": self.kind.value, "target": self.target}

    @classmethod
    def from_dict(cls, value: Any) -> Relation:
        data = _mapping(value, "relation")
        _reject_unknown(data, {_KIND, _TARGET}, "relation")
        try:
            kind = RelationKind(_required(data, _KIND, "relation.kind"))
        except ValueError as exc:
            raise SchemaValidationError(
                f"relation kind must be one of: {', '.join(k.value for k in RelationKind)}"
            ) from exc
        return cls(kind, _required(data, _TARGET, "relation.target"))


@dataclass(frozen=True)
class ExternalRef:
    """Something outside the KB an entry points at, such as a Curated Wiki page; the KB never resolves it."""

    uri: str
    version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _require_string(self.uri, "external reference uri"))
        if self.version:
            object.__setattr__(self, _VERSION, _require_string(self.version, "external reference version"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"uri": self.uri, "version": self.version}

    @classmethod
    def from_dict(cls, value: Any) -> ExternalRef:
        data = _mapping(value, "external reference")
        _reject_unknown(data, {_URI, _VERSION}, "external reference")
        return cls(_required(data, _URI, "external reference uri"), data.get("version", ""))


@dataclass(frozen=True)
class Relations:
    entries: tuple[Relation, ...] = ()
    external: tuple[ExternalRef, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(item, Relation) for item in self.entries):
            raise SchemaValidationError("relations.entries contains an invalid relation")
        if not all(isinstance(item, ExternalRef) for item in self.external):
            raise SchemaValidationError("relations.external contains an invalid reference")
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "external", tuple(self.external))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "entries": [item.to_dict() for item in self.entries],
            "external": [item.to_dict() for item in self.external],
        }

    @classmethod
    def from_dict(cls, value: Any) -> Relations:
        data = _mapping(value, _RELATIONS)
        _reject_unknown(data, {"entries", "external"}, _RELATIONS)
        return cls(
            tuple(Relation.from_dict(item) for item in _list(data.get("entries", []), "relations.entries")),
            tuple(ExternalRef.from_dict(item) for item in _list(data.get("external", []), "relations.external")),
        )


@dataclass(frozen=True)
class Derivation:
    """How a version came to be: ``kb_state`` is the state of the schema's Experiences its evidence was drawn at."""

    kb_state: str = ""

    def __post_init__(self) -> None:
        if self.kb_state:
            object.__setattr__(self, "kb_state", _require_string(self.kb_state, "derivation.kb_state"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"kb_state": self.kb_state}

    @classmethod
    def from_dict(cls, value: Any) -> Derivation:
        data = _mapping(value, _DERIVATION)
        _reject_unknown(data, {"kb_state"}, _DERIVATION)
        return cls(data.get("kb_state", ""))


#: What a version says: when it holds, what it concludes, what it rests on, what it relates to, and how it is found.
DISTILL_KNOWLEDGE_FIELDS = frozenset({_SCOPE, _CLAIM, _EVIDENCE, _RELATIONS, "tags", "notes"})
#: What keeps a version: its identity, the schema and form it is written under, and how it came to be.
DISTILL_METADATA_FIELDS = frozenset(
    {_KIND, "id", _VERSION, _SCHEMA_REF, _FORM_REF, _CREATED_AT, "author", "provenance", _DERIVATION}
)


@dataclass(frozen=True)
class DistillEntry:
    """One version of one Distill of the Experiences of ``schema_ref``, its claim written in the form ``form_ref``."""

    id: str
    version: int
    schema_ref: str
    form_ref: str
    created_at: datetime
    provenance: Provenance
    scope: Scope
    claim: dict[str, FieldValue]
    evidence: Evidence
    author: str = ""
    derivation: Derivation = field(default_factory=Derivation)
    relations: Relations = field(default_factory=Relations)
    tags: tuple[str, ...] = ()
    notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", require_entry_id(self.id))
        object.__setattr__(self, _VERSION, _int(self.version, "distill version", minimum=1))
        object.__setattr__(self, "schema_ref", _require_schema_ref(self.schema_ref, "distill.schema_ref"))
        object.__setattr__(self, "form_ref", require_form_ref(self.form_ref, "distill.form_ref"))
        object.__setattr__(self, _CREATED_AT, _aware_datetime(self.created_at, _CREATED_AT_WHERE))
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("distill.provenance is invalid")
        for name, kind in (
            (_SCOPE, Scope),
            (_EVIDENCE, Evidence),
            (_DERIVATION, Derivation),
            (_RELATIONS, Relations),
        ):
            if not isinstance(getattr(self, name), kind):
                raise SchemaValidationError(f"distill.{name} is invalid")
        object.__setattr__(self, _CLAIM, _field_values(self.claim, _CLAIM))
        if self.author:
            object.__setattr__(self, "author", _require_string(self.author, "distill.author"))
        tags = tuple(_require_string(tag, "distill tag") for tag in self.tags)
        object.__setattr__(self, "tags", _unique(tags, "distill tags"))
        object.__setattr__(self, "notes", _notes(self.notes, "distill.notes"))
        if any(item.target == self.id for item in self.relations.entries):
            raise SchemaValidationError("a Distill cannot relate to itself")

    def validate(self, form: FormDeclaration, declaration: ExperienceDeclaration) -> None:
        """Check this version against the form its claim is written in and the schema its evidence comes from."""

        if self.schema_ref != declaration.schema_ref:
            raise SchemaValidationError("distill schema_ref does not match the declaration")
        if self.form_ref != form.form_ref:
            raise SchemaValidationError("distill form_ref does not match the form")
        self.scope.validate(declaration, _SCOPE)
        if self.evidence.check is not None:
            self.evidence.check.validate(declaration, "evidence.check")
        form.validate_claim(self.claim)

    def files(self) -> tuple[FileRef, ...]:
        found: dict[str, FileRef] = {}
        for value in self.claim.values():
            for ref in _files(value):
                found.setdefault(ref.sha256, ref)
        return tuple(found.values())

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": DISTILL_KIND,
            "id": self.id,
            "version": self.version,
            "schema_ref": self.schema_ref,
            "form_ref": self.form_ref,
            "created_at": _format_datetime(self.created_at),
            "author": self.author,
            "provenance": self.provenance.to_dict(),
            "derivation": self.derivation.to_dict(),
            "scope": self.scope.to_dict(),
            "claim": {name: _encode(value) for name, value in self.claim.items()},
            "evidence": self.evidence.to_dict(),
            "relations": self.relations.to_dict(),
            "tags": list(self.tags),
            "notes": dict(self.notes),
        }

    def knowledge(self) -> dict[str, JsonValue]:
        return {name: value for name, value in self.to_dict().items() if name in DISTILL_KNOWLEDGE_FIELDS}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> DistillEntry:
        data = _mapping(value, DISTILL_KIND)
        _reject_unknown(data, DISTILL_KNOWLEDGE_FIELDS | DISTILL_METADATA_FIELDS, DISTILL_KIND)
        if data.get("kind") != DISTILL_KIND:
            raise SchemaValidationError(f"distill.kind must be {DISTILL_KIND!r}")
        return cls(
            id=_required(data, "id", "distill.id"),
            version=_required(data, _VERSION, "distill.version"),
            schema_ref=_required(data, _SCHEMA_REF, "distill.schema_ref"),
            form_ref=_required(data, _FORM_REF, "distill.form_ref"),
            created_at=_parse_datetime(_required(data, _CREATED_AT, _CREATED_AT_WHERE), _CREATED_AT_WHERE),
            author=data.get("author", ""),
            provenance=Provenance.from_dict(_required(data, "provenance", "distill.provenance")),
            derivation=Derivation.from_dict(data.get("derivation", {})),
            scope=Scope.from_dict(data.get("scope", {})),
            claim=data.get("claim", {}),
            evidence=Evidence.from_dict(_required(data, _EVIDENCE, "distill.evidence")),
            relations=Relations.from_dict(data.get("relations", {})),
            tags=tuple(_list(data.get("tags", []), "distill.tags")),
            notes=data.get("notes", {}),
        )


__all__ = [
    "DISTILL_KIND",
    "DISTILL_KNOWLEDGE_FIELDS",
    "DISTILL_METADATA_FIELDS",
    "FORM_VERSION",
    "OBJECTIVE_PATH",
    "Condition",
    "Derivation",
    "DistillEntry",
    "Evidence",
    "EvidenceClass",
    "EvidenceMember",
    "EvidenceStats",
    "ExternalRef",
    "FormDeclaration",
    "Relation",
    "RelationKind",
    "Relations",
    "Scope",
    "new_entry_id",
    "path_field",
    "require_entry_id",
    "require_form_ref",
    "value_at",
]
