"""Versioned Experience and declaration models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, TypeAlias

from hyperloom_kb.identity import derive_experience_id

CURRENT_SCHEMA_VERSION = 1
MAX_SAFE_INTEGER = (1 << 53) - 1

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]*$")
_SCHEMA_REF_RE = re.compile(r"^schema:sha256:[0-9a-f]{64}$")


class SchemaValidationError(ValueError):
    """Raised when a schema object violates the v0 contract."""


class UnsupportedSchemaVersion(SchemaValidationError):
    """Raised when a document uses a schema version this package cannot read."""


class ExperienceStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class FieldKind(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    VERSION = "version"
    JSON = "json"


class ObjectiveDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_token(value: Any, name: str) -> str:
    token = _require_string(value, name)
    if not _TOKEN_RE.fullmatch(token):
        raise SchemaValidationError(f"{name} contains unsupported characters")
    return token


def _require_schema_ref(value: Any, name: str = "schema_ref") -> str:
    schema_ref = _require_string(value, name)
    if not _SCHEMA_REF_RE.fullmatch(schema_ref):
        raise SchemaValidationError(f"{name} must be schema:sha256:<64 lowercase hex>")
    return schema_ref


def _string_or_empty(value: Any, name: str) -> str:
    if value == "":
        return ""
    return _require_string(value, name)


def _require_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError("schema_version must be an integer")
    if value != CURRENT_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(f"unsupported schema_version {value}; expected {CURRENT_SCHEMA_VERSION}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaValidationError(f"{name} must be finite")
    return number


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _finite_number(value, name)


def _json_value(value: Any, name: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{name}[]") for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = _require_string(key, f"{name} key")
            result[normalized_key] = _json_value(item, f"{name}.{normalized_key}")
        return result
    raise SchemaValidationError(f"{name} contains a non-JSON value")


def _json_object(value: Any, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{name} must be an object")
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise AssertionError("object normalization returned a non-object")
    return normalized


def _identity_object(value: Any, name: str) -> dict[str, JsonScalar]:
    if not isinstance(value, dict) or not value:
        raise SchemaValidationError(f"{name} must be a non-empty object")
    result: dict[str, JsonScalar] = {}
    for key, item in value.items():
        normalized_key = _require_string(key, f"{name} key")
        if isinstance(item, (dict, list)):
            raise SchemaValidationError(f"{name}.{normalized_key} must be a scalar, not a nested value")
        normalized = _json_value(item, f"{name}.{normalized_key}")
        if isinstance(normalized, (dict, list)):
            raise AssertionError("identity normalization returned a nested value")
        result[normalized_key] = normalized
    return result


def _aware_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SchemaValidationError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any, name: str) -> datetime:
    raw = _require_string(value, name)
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as exc:
        raise SchemaValidationError(f"{name} must be an RFC3339 datetime") from exc
    return _aware_datetime(parsed, name)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{name} must be an object")
    return value


def _required(data: dict[str, Any], key: str, name: str) -> Any:
    if key not in data:
        raise SchemaValidationError(f"{name} is required")
    return data[key]


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{name} must be a list")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{name} must be a boolean")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SchemaValidationError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SchemaValidationError(f"{name} must be a list")
    return tuple(_require_string(item, f"{name}[]") for item in value)


@dataclass(frozen=True)
class FieldDeclaration:
    name: str
    description: str
    kind: FieldKind = FieldKind.STRING
    required: bool = True
    sensitive: bool = False

    def __post_init__(self) -> None:
        name = _require_string(self.name, "field.name")
        if not _FIELD_NAME_RE.fullmatch(name):
            raise SchemaValidationError(f"invalid declaration field name: {name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            _require_string(self.description, f"field {name!r} description"),
        )
        if not isinstance(self.kind, FieldKind):
            raise SchemaValidationError(f"field {name!r} has an invalid kind")
        if not isinstance(self.required, bool) or not isinstance(self.sensitive, bool):
            raise SchemaValidationError(f"field {name!r} required/sensitive flags must be booleans")

    def validate(self, value: JsonScalar) -> None:
        if value is None:
            if self.required:
                raise SchemaValidationError(f"required field {self.name!r} is unknown")
            return
        if self.kind in {FieldKind.STRING, FieldKind.VERSION} and not isinstance(value, str):
            raise SchemaValidationError(f"field {self.name!r} must be a string")
        if self.kind is FieldKind.BOOLEAN and not isinstance(value, bool):
            raise SchemaValidationError(f"field {self.name!r} must be a boolean")
        if self.kind is FieldKind.NUMBER:
            _finite_number(value, f"field {self.name!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "required": self.required,
            "sensitive": self.sensitive,
        }

    @classmethod
    def from_dict(cls, value: Any) -> FieldDeclaration:
        data = _mapping(value, "field declaration")
        _reject_unknown(
            data,
            {"name", "description", "kind", "required", "sensitive"},
            "field declaration",
        )
        try:
            kind = FieldKind(data.get("kind", FieldKind.STRING.value))
        except ValueError as exc:
            raise SchemaValidationError("field declaration has an invalid kind") from exc
        return cls(
            name=_require_string(
                _required(data, "name", "field declaration.name"),
                "field declaration.name",
            ),
            description=_require_string(
                _required(data, "description", "field declaration.description"),
                "field declaration.description",
            ),
            kind=kind,
            required=_boolean(
                data.get("required", True),
                "field declaration.required",
            ),
            sensitive=_boolean(
                data.get("sensitive", False),
                "field declaration.sensitive",
            ),
        )


@dataclass(frozen=True)
class ObjectiveDeclaration:
    id: str
    direction: ObjectiveDirection
    description: str
    unit: str = ""
    failure_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "objective.id"))
        if not isinstance(self.direction, ObjectiveDirection):
            raise SchemaValidationError(f"objective {self.id!r} has an invalid direction")
        object.__setattr__(
            self,
            "description",
            _require_string(self.description, f"objective {self.id!r} description"),
        )
        if self.unit:
            object.__setattr__(self, "unit", _require_string(self.unit, "objective.unit"))
        object.__setattr__(
            self,
            "failure_value",
            _optional_number(self.failure_value, "objective.failure_value"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "direction": self.direction.value,
            "description": self.description,
            "unit": self.unit,
            "failure_value": self.failure_value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ObjectiveDeclaration:
        data = _mapping(value, "objective declaration")
        _reject_unknown(
            data,
            {"id", "direction", "description", "unit", "failure_value"},
            "objective declaration",
        )
        try:
            direction = ObjectiveDirection(data.get("direction"))
        except ValueError as exc:
            raise SchemaValidationError("objective declaration has an invalid direction") from exc
        return cls(
            id=_require_token(
                _required(data, "id", "objective declaration.id"),
                "objective declaration.id",
            ),
            direction=direction,
            description=_require_string(
                _required(
                    data,
                    "description",
                    "objective declaration.description",
                ),
                "objective declaration.description",
            ),
            unit=_string_or_empty(
                data.get("unit", ""),
                "objective declaration.unit",
            ),
            failure_value=_optional_number(
                data.get("failure_value"),
                "objective declaration.failure_value",
            ),
        )


@dataclass(frozen=True)
class Alternative:
    option: str
    why_not: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "option", _require_string(self.option, "alternative.option"))
        object.__setattr__(
            self,
            "why_not",
            _require_string(self.why_not, "alternative.why_not"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {"option": self.option, "why_not": self.why_not}

    @classmethod
    def from_dict(cls, value: Any) -> Alternative:
        data = _mapping(value, "alternative")
        _reject_unknown(data, {"option", "why_not"}, "alternative")
        return cls(
            option=_require_string(
                _required(data, "option", "alternative.option"),
                "alternative.option",
            ),
            why_not=_require_string(
                _required(data, "why_not", "alternative.why_not"),
                "alternative.why_not",
            ),
        )


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    value: JsonValue = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_token(self.name, "constraint.name"))
        if not isinstance(self.passed, bool):
            raise SchemaValidationError("constraint.passed must be a boolean")
        object.__setattr__(self, "value", _json_value(self.value, "constraint.value"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "passed": self.passed, "value": self.value}

    @classmethod
    def from_dict(cls, value: Any) -> ConstraintResult:
        data = _mapping(value, "constraint")
        _reject_unknown(data, {"name", "passed", "value"}, "constraint")
        return cls(
            name=_require_token(
                _required(data, "name", "constraint.name"),
                "constraint.name",
            ),
            passed=_boolean(
                _required(data, "passed", "constraint.passed"),
                "constraint.passed",
            ),
            value=_json_value(data.get("value"), "constraint.value"),
        )


@dataclass(frozen=True)
class RenderedRef:
    """A record rendered to the decision context, not proof of actual use."""

    id: str
    purpose: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "rendered_ref.id"))
        if self.purpose:
            object.__setattr__(
                self,
                "purpose",
                _require_token(self.purpose, "rendered_ref.purpose"),
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {"id": self.id, "purpose": self.purpose}

    @classmethod
    def from_dict(cls, value: Any) -> RenderedRef:
        data = _mapping(value, "rendered_ref")
        _reject_unknown(data, {"id", "purpose"}, "rendered_ref")
        return cls(
            id=_require_token(
                _required(data, "id", "rendered_ref.id"),
                "rendered_ref.id",
            ),
            purpose=_string_or_empty(data.get("purpose", ""), "rendered_ref.purpose"),
        )


@dataclass(frozen=True)
class Change:
    identity: dict[str, JsonScalar]
    summary: str
    kind: str = ""
    content: str = ""
    resource_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity",
            _identity_object(self.identity, "change.identity"),
        )
        object.__setattr__(self, "summary", _require_string(self.summary, "change.summary"))
        if self.kind:
            object.__setattr__(self, "kind", _require_token(self.kind, "change.kind"))
        object.__setattr__(
            self,
            "content",
            _string_or_empty(self.content, "change.content"),
        )
        refs: list[str] = []
        for ref in self.resource_refs:
            normalized = _require_string(ref, "change.resource_refs[]")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts:
                raise SchemaValidationError("change.resource_refs[] must be a safe relative path")
            refs.append(path.as_posix())
        if len(set(refs)) != len(refs):
            raise SchemaValidationError("change.resource_refs must not contain duplicates")
        object.__setattr__(self, "resource_refs", tuple(refs))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "identity": dict(self.identity),
            "summary": self.summary,
            "kind": self.kind,
            "content": self.content,
            "resource_refs": list(self.resource_refs),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Change:
        data = _mapping(value, "change")
        _reject_unknown(
            data,
            {"identity", "summary", "kind", "content", "resource_refs"},
            "change",
        )
        return cls(
            identity=_identity_object(
                _required(data, "identity", "change.identity"),
                "change.identity",
            ),
            summary=_require_string(
                _required(data, "summary", "change.summary"),
                "change.summary",
            ),
            kind=_string_or_empty(data.get("kind", ""), "change.kind"),
            content=_string_or_empty(data.get("content", ""), "change.content"),
            resource_refs=_string_tuple(data.get("resource_refs"), "change.resource_refs"),
        )


@dataclass(frozen=True)
class Outcome:
    decision: str
    value: float | None = None
    constraints: tuple[ConstraintResult, ...] = ()
    error_class: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", _require_token(self.decision, "outcome.decision"))
        object.__setattr__(self, "value", _optional_number(self.value, "outcome.value"))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not all(isinstance(item, ConstraintResult) for item in self.constraints):
            raise SchemaValidationError("outcome.constraints contains an invalid item")
        if self.error_class:
            object.__setattr__(
                self,
                "error_class",
                _require_token(self.error_class, "outcome.error_class"),
            )
        if self.value is None and not self.error_class:
            raise SchemaValidationError("outcome requires either a numeric value or an error_class")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "decision": self.decision,
            "value": self.value,
            "constraints": [item.to_dict() for item in self.constraints],
            "error_class": self.error_class,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Outcome:
        data = _mapping(value, "outcome")
        _reject_unknown(
            data,
            {"decision", "value", "constraints", "error_class"},
            "outcome",
        )
        raw_constraints = _list(data.get("constraints", []), "outcome.constraints")
        return cls(
            decision=_require_token(
                _required(data, "decision", "outcome.decision"),
                "outcome.decision",
            ),
            value=_optional_number(data.get("value"), "outcome.value"),
            constraints=tuple(ConstraintResult.from_dict(item) for item in raw_constraints),
            error_class=_string_or_empty(
                data.get("error_class", ""),
                "outcome.error_class",
            ),
        )


@dataclass(frozen=True)
class Provenance:
    producer: str
    producer_version: str
    model: str = ""
    prompt_version: str = ""
    snapshot_version: str = ""
    source_ref: str = ""
    extra: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer", _require_token(self.producer, "provenance.producer"))
        object.__setattr__(
            self,
            "producer_version",
            _require_string(self.producer_version, "provenance.producer_version"),
        )
        for name in ("model", "prompt_version", "snapshot_version", "source_ref"):
            value = getattr(self, name)
            if value:
                object.__setattr__(
                    self,
                    name,
                    _require_string(value, f"provenance.{name}"),
                )
        object.__setattr__(self, "extra", _json_object(self.extra, "provenance.extra"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "producer": self.producer,
            "producer_version": self.producer_version,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "snapshot_version": self.snapshot_version,
            "source_ref": self.source_ref,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Provenance:
        data = _mapping(value, "provenance")
        _reject_unknown(
            data,
            {
                "producer",
                "producer_version",
                "model",
                "prompt_version",
                "snapshot_version",
                "source_ref",
                "extra",
            },
            "provenance",
        )
        return cls(
            producer=_require_token(
                _required(data, "producer", "provenance.producer"),
                "provenance.producer",
            ),
            producer_version=_require_string(
                _required(
                    data,
                    "producer_version",
                    "provenance.producer_version",
                ),
                "provenance.producer_version",
            ),
            model=_string_or_empty(data.get("model", ""), "provenance.model"),
            prompt_version=_string_or_empty(
                data.get("prompt_version", ""),
                "provenance.prompt_version",
            ),
            snapshot_version=_string_or_empty(
                data.get("snapshot_version", ""),
                "provenance.snapshot_version",
            ),
            source_ref=_string_or_empty(
                data.get("source_ref", ""),
                "provenance.source_ref",
            ),
            extra=_json_object(data.get("extra", {}), "provenance.extra"),
        )


#: Top-level record fields that say what was learned: the conditions, the decision and why, what changed, and
#: what came of it. A read shows an agent these.
KNOWLEDGE_FIELDS = frozenset(
    {
        "identity",
        "objective",
        "baseline_identity",
        "baseline_value",
        "preconditions",
        "reasoning",
        "alternatives",
        "change",
        "outcome",
        "reflection",
    }
)
#: Top-level record fields that keep the record: its identity, lifecycle, lineage, and how it came to be,
#: including the reads that shaped its decision. They stay in the record and never reach a prompt.
METADATA_FIELDS = frozenset(
    {
        "kind",
        "id",
        "schema_ref",
        "schema_version",
        "status",
        "created_at",
        "completed_at",
        "run_id",
        "seq",
        "parent_id",
        "supersedes",
        "rendered_refs",
        "provenance",
    }
)


@dataclass(frozen=True)
class Experience:
    id: str
    run_id: str
    seq: int
    created_at: datetime
    identity: dict[str, JsonScalar]
    objective: str
    baseline_identity: dict[str, JsonScalar]
    baseline_value: float
    provenance: Provenance
    schema_ref: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    status: ExperienceStatus = ExperienceStatus.IN_PROGRESS
    completed_at: datetime | None = None
    parent_id: str = ""
    supersedes: str = ""
    preconditions: tuple[str, ...] = ()
    reasoning: str = ""
    alternatives: tuple[Alternative, ...] = ()
    change: Change | None = None
    rendered_refs: tuple[RenderedRef, ...] = ()
    outcome: Outcome | None = None
    reflection: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "experience.id"))
        object.__setattr__(self, "run_id", _require_token(self.run_id, "experience.run_id"))
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0 or self.seq > MAX_SAFE_INTEGER:
            raise SchemaValidationError("experience.seq must be a non-negative IEEE-754-safe integer")
        object.__setattr__(self, "schema_version", _require_version(self.schema_version))
        object.__setattr__(self, "schema_ref", _require_schema_ref(self.schema_ref, "experience.schema_ref"))
        if not isinstance(self.status, ExperienceStatus):
            raise SchemaValidationError("experience.status is invalid")
        object.__setattr__(
            self,
            "created_at",
            _aware_datetime(self.created_at, "experience.created_at"),
        )
        object.__setattr__(
            self,
            "identity",
            _identity_object(self.identity, "experience.identity"),
        )
        object.__setattr__(
            self,
            "objective",
            _require_token(self.objective, "experience.objective"),
        )
        object.__setattr__(
            self,
            "baseline_identity",
            _identity_object(self.baseline_identity, "experience.baseline_identity"),
        )
        object.__setattr__(
            self,
            "baseline_value",
            _finite_number(self.baseline_value, "experience.baseline_value"),
        )
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("experience.provenance is invalid")
        expected_id = derive_experience_id(
            self.provenance.producer,
            self.run_id,
            self.seq,
        )
        if self.id != expected_id:
            raise SchemaValidationError("experience.id does not match producer, run_id, and seq")
        for name in ("parent_id", "supersedes"):
            value = getattr(self, name)
            if value:
                object.__setattr__(
                    self,
                    name,
                    _require_token(value, f"experience.{name}"),
                )
        object.__setattr__(
            self,
            "preconditions",
            tuple(_require_string(item, "experience.preconditions[]") for item in self.preconditions),
        )
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        if not all(isinstance(item, Alternative) for item in self.alternatives):
            raise SchemaValidationError("experience.alternatives contains an invalid item")
        object.__setattr__(self, "rendered_refs", tuple(self.rendered_refs))
        if not all(isinstance(item, RenderedRef) for item in self.rendered_refs):
            raise SchemaValidationError("experience.rendered_refs contains an invalid item")
        rendered_ids = [item.id for item in self.rendered_refs]
        if len(set(rendered_ids)) != len(rendered_ids):
            raise SchemaValidationError("experience.rendered_refs contains duplicate ids")
        object.__setattr__(
            self,
            "reasoning",
            _string_or_empty(self.reasoning, "experience.reasoning"),
        )
        object.__setattr__(
            self,
            "reflection",
            _string_or_empty(self.reflection, "experience.reflection"),
        )
        if self.change is not None and not isinstance(self.change, Change):
            raise SchemaValidationError("experience.change is invalid")
        if (self.change is None) != (not self.reasoning):
            raise SchemaValidationError("experience.change and reasoning must be captured together")
        if self.change is None and (self.alternatives or self.rendered_refs):
            raise SchemaValidationError("alternatives and rendered_refs require a captured decision")

        if self.status is ExperienceStatus.IN_PROGRESS:
            if self.completed_at is not None or self.outcome is not None or self.reflection:
                raise SchemaValidationError("in_progress Experience cannot have completion fields")
            return

        if self.completed_at is None:
            raise SchemaValidationError("complete Experience requires completed_at")
        completed_at = _aware_datetime(self.completed_at, "experience.completed_at")
        object.__setattr__(self, "completed_at", completed_at)
        if completed_at < self.created_at:
            raise SchemaValidationError("completed_at cannot be before created_at")
        if self.change is None:
            raise SchemaValidationError("complete Experience requires change")
        if not isinstance(self.outcome, Outcome):
            raise SchemaValidationError("complete Experience requires outcome")
        if not self.reflection:
            raise SchemaValidationError("complete Experience requires reflection")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "kind": "experience",
            "id": self.id,
            "schema_ref": self.schema_ref,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "created_at": _format_datetime(self.created_at),
            "completed_at": (_format_datetime(self.completed_at) if self.completed_at is not None else None),
            "run_id": self.run_id,
            "seq": self.seq,
            "parent_id": self.parent_id,
            "supersedes": self.supersedes,
            "identity": dict(self.identity),
            "objective": self.objective,
            "baseline_identity": dict(self.baseline_identity),
            "baseline_value": self.baseline_value,
            "preconditions": list(self.preconditions),
            "reasoning": self.reasoning,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "change": self.change.to_dict() if self.change is not None else None,
            "rendered_refs": [item.to_dict() for item in self.rendered_refs],
            "outcome": self.outcome.to_dict() if self.outcome is not None else None,
            "reflection": self.reflection,
            "provenance": self.provenance.to_dict(),
        }

    def knowledge(self) -> dict[str, JsonValue]:
        """The record's ``KNOWLEDGE_FIELDS``, exactly as ``to_dict`` holds them."""

        return {name: value for name, value in self.to_dict().items() if name in KNOWLEDGE_FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> Experience:
        data = _mapping(value, "experience")
        allowed = {
            "kind",
            "id",
            "schema_ref",
            "schema_version",
            "status",
            "created_at",
            "completed_at",
            "run_id",
            "seq",
            "parent_id",
            "supersedes",
            "identity",
            "objective",
            "baseline_identity",
            "baseline_value",
            "preconditions",
            "reasoning",
            "alternatives",
            "change",
            "rendered_refs",
            "outcome",
            "reflection",
            "provenance",
        }
        _reject_unknown(data, allowed, "experience")
        if data.get("kind") != "experience":
            raise SchemaValidationError("experience.kind must be 'experience'")
        schema_version = _require_version(_required(data, "schema_version", "experience.schema_version"))
        try:
            status = ExperienceStatus(_required(data, "status", "experience.status"))
        except ValueError as exc:
            raise SchemaValidationError("experience.status is invalid") from exc
        alternatives = _list(
            data.get("alternatives", []),
            "experience.alternatives",
        )
        rendered_refs = _list(
            data.get("rendered_refs", []),
            "experience.rendered_refs",
        )
        completed_at = data.get("completed_at")
        seq = _required(data, "seq", "experience.seq")
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise SchemaValidationError("experience.seq must be an integer")
        return cls(
            id=_require_token(
                _required(data, "id", "experience.id"),
                "experience.id",
            ),
            run_id=_require_token(
                _required(data, "run_id", "experience.run_id"),
                "experience.run_id",
            ),
            seq=seq,
            schema_ref=_require_schema_ref(
                _required(data, "schema_ref", "experience.schema_ref"),
                "experience.schema_ref",
            ),
            schema_version=schema_version,
            status=status,
            created_at=_parse_datetime(
                _required(data, "created_at", "experience.created_at"),
                "experience.created_at",
            ),
            completed_at=(
                _parse_datetime(completed_at, "experience.completed_at") if completed_at is not None else None
            ),
            parent_id=_string_or_empty(
                data.get("parent_id", ""),
                "experience.parent_id",
            ),
            supersedes=_string_or_empty(
                data.get("supersedes", ""),
                "experience.supersedes",
            ),
            identity=_identity_object(
                _required(data, "identity", "experience.identity"),
                "experience.identity",
            ),
            objective=_require_token(
                _required(data, "objective", "experience.objective"),
                "experience.objective",
            ),
            baseline_identity=_identity_object(
                _required(data, "baseline_identity", "experience.baseline_identity"),
                "experience.baseline_identity",
            ),
            baseline_value=_finite_number(
                _required(
                    data,
                    "baseline_value",
                    "experience.baseline_value",
                ),
                "experience.baseline_value",
            ),
            preconditions=_string_tuple(
                data.get("preconditions"),
                "experience.preconditions",
            ),
            reasoning=_string_or_empty(
                data.get("reasoning", ""),
                "experience.reasoning",
            ),
            alternatives=tuple(Alternative.from_dict(item) for item in alternatives),
            change=Change.from_dict(data["change"]) if data.get("change") is not None else None,
            rendered_refs=tuple(RenderedRef.from_dict(item) for item in rendered_refs),
            outcome=(Outcome.from_dict(data["outcome"]) if data.get("outcome") is not None else None),
            reflection=_string_or_empty(
                data.get("reflection", ""),
                "experience.reflection",
            ),
            provenance=Provenance.from_dict(_required(data, "provenance", "experience.provenance")),
        )


@dataclass(frozen=True)
class ExperienceDeclaration:
    identity: tuple[FieldDeclaration, ...]
    baseline_identity: tuple[FieldDeclaration, ...]
    change_identity: tuple[FieldDeclaration, ...]
    objectives: tuple[ObjectiveDeclaration, ...]
    decisions: tuple[str, ...]
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_version(self.schema_version))
        object.__setattr__(self, "identity", tuple(self.identity))
        object.__setattr__(self, "baseline_identity", tuple(self.baseline_identity))
        object.__setattr__(self, "change_identity", tuple(self.change_identity))
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(
            self,
            "decisions",
            tuple(_require_token(item, "declaration.decisions[]") for item in self.decisions),
        )
        if not self.identity:
            raise SchemaValidationError("declaration.identity must not be empty")
        if not self.baseline_identity:
            raise SchemaValidationError("declaration.baseline_identity must not be empty")
        if not self.change_identity:
            raise SchemaValidationError("declaration.change_identity must not be empty")
        if not self.objectives:
            raise SchemaValidationError("declaration.objectives must not be empty")
        if not self.decisions:
            raise SchemaValidationError("declaration.decisions must not be empty")
        if not all(isinstance(item, FieldDeclaration) for item in self.identity):
            raise SchemaValidationError("declaration.identity contains an invalid field")
        if not all(isinstance(item, FieldDeclaration) for item in self.baseline_identity):
            raise SchemaValidationError("declaration.baseline_identity contains an invalid field")
        if not all(isinstance(item, FieldDeclaration) for item in self.change_identity):
            raise SchemaValidationError("declaration.change_identity contains an invalid field")
        if not all(isinstance(item, ObjectiveDeclaration) for item in self.objectives):
            raise SchemaValidationError("declaration.objectives contains an invalid objective")
        self._require_unique((item.name for item in self.identity), "identity field")
        self._require_unique(
            (item.name for item in self.baseline_identity),
            "baseline identity field",
        )
        self._require_unique(
            (item.name for item in self.change_identity),
            "change identity field",
        )
        self._require_unique((item.id for item in self.objectives), "objective")
        self._require_unique(iter(self.decisions), "decision")

    @staticmethod
    def _require_unique(values: Iterable[str], name: str) -> None:
        materialized = list(values)
        if len(set(materialized)) != len(materialized):
            raise SchemaValidationError(f"declaration contains duplicate {name}s")

    @property
    def schema_ref(self) -> str:
        """Return the exact content-addressed reference for this declaration."""

        return derive_schema_ref(self)

    def validate(self, experience: Experience) -> None:
        if experience.schema_version != self.schema_version:
            raise SchemaValidationError("Experience and declaration schema versions differ")
        if experience.schema_ref != self.schema_ref:
            raise SchemaValidationError("Experience schema_ref does not match declaration")

        objectives = {item.id: item for item in self.objectives}
        if experience.objective not in objectives:
            raise SchemaValidationError(f"objective {experience.objective!r} is not declared")
        self._validate_fields(self.identity, experience.identity, "identity")
        self._validate_fields(
            self.baseline_identity,
            experience.baseline_identity,
            "baseline_identity",
        )

        if experience.change is not None:
            self._validate_fields(
                self.change_identity,
                experience.change.identity,
                "change.identity",
            )
        if experience.outcome is not None:
            if experience.outcome.decision not in self.decisions:
                raise SchemaValidationError(f"decision {experience.outcome.decision!r} is not declared")
            objective = objectives[experience.objective]
            if experience.outcome.value is None and experience.outcome.error_class and objective.failure_value is None:
                raise SchemaValidationError(f"objective {objective.id!r} requires failure_value for error outcomes")

    @staticmethod
    def _validate_fields(
        declarations: tuple[FieldDeclaration, ...],
        values: dict[str, JsonScalar],
        name: str,
    ) -> None:
        for declaration in declarations:
            if declaration.name not in values:
                if declaration.required:
                    raise SchemaValidationError(f"{name} is missing required field {declaration.name!r}")
                continue
            declaration.validate(values[declaration.name])

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "identity": [item.to_dict() for item in self.identity],
            "baseline_identity": [item.to_dict() for item in self.baseline_identity],
            "change_identity": [item.to_dict() for item in self.change_identity],
            "objectives": [item.to_dict() for item in self.objectives],
            "decisions": list(self.decisions),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExperienceDeclaration:
        data = _mapping(value, "declaration")
        _reject_unknown(
            data,
            {
                "schema_version",
                "identity",
                "baseline_identity",
                "change_identity",
                "objectives",
                "decisions",
            },
            "declaration",
        )
        schema_version = _require_version(_required(data, "schema_version", "declaration.schema_version"))
        identity = _list(
            _required(data, "identity", "declaration.identity"),
            "declaration.identity",
        )
        baseline_identity = _list(
            _required(data, "baseline_identity", "declaration.baseline_identity"),
            "declaration.baseline_identity",
        )
        change_identity = _list(
            _required(data, "change_identity", "declaration.change_identity"),
            "declaration.change_identity",
        )
        objectives = _list(
            _required(data, "objectives", "declaration.objectives"),
            "declaration.objectives",
        )
        decisions = _list(
            _required(data, "decisions", "declaration.decisions"),
            "declaration.decisions",
        )
        return cls(
            schema_version=schema_version,
            identity=tuple(FieldDeclaration.from_dict(item) for item in identity),
            baseline_identity=tuple(FieldDeclaration.from_dict(item) for item in baseline_identity),
            change_identity=tuple(FieldDeclaration.from_dict(item) for item in change_identity),
            objectives=tuple(ObjectiveDeclaration.from_dict(item) for item in objectives),
            decisions=tuple(_require_token(item, "declaration.decisions[]") for item in decisions),
        )


def derive_schema_ref(declaration: ExperienceDeclaration) -> str:
    """Return a stable global reference for one exact declaration document."""

    if not isinstance(declaration, ExperienceDeclaration):
        raise SchemaValidationError("schema_ref requires an ExperienceDeclaration")
    encoded = json.dumps(
        declaration.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"schema:sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "KNOWLEDGE_FIELDS",
    "METADATA_FIELDS",
    "Alternative",
    "Change",
    "ConstraintResult",
    "Experience",
    "ExperienceDeclaration",
    "ExperienceStatus",
    "FieldDeclaration",
    "FieldKind",
    "JsonScalar",
    "JsonValue",
    "ObjectiveDeclaration",
    "ObjectiveDirection",
    "Outcome",
    "Provenance",
    "RenderedRef",
    "SchemaValidationError",
    "UnsupportedSchemaVersion",
    "derive_schema_ref",
]
