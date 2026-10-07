"""Versioned Experience and declaration models.

The KB fixes an Experience's categories and the kinds of value a field may hold. A declaration names each
category's fields, their kinds, and the roles in which KB functions read them: grouping, exact lookup, search
weights, the outcome a read filters and counts by, and the measurement its statistics take.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, TypeAlias

from hyperloom_kb.identity import derive_experience_id

CURRENT_SCHEMA_VERSION = 2
MAX_SAFE_INTEGER = (1 << 53) - 1
#: The most one text value holds; a read injects text whole, so a longer one belongs in a file.
TEXT_MAX_BYTES = 32 * 1024

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]*$")
_SCHEMA_REF_RE = re.compile(r"^schema:sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SchemaValidationError(ValueError):
    """Raised when a schema object violates the contract."""


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
    TEXT = "text"
    FILE = "file"


SCALAR_KINDS = frozenset({FieldKind.STRING, FieldKind.NUMBER, FieldKind.BOOLEAN, FieldKind.VERSION})


class FieldRole(str, Enum):
    """What a KB function reads a field as."""

    #: The outcome a read filters by and a Repeat Group counts: a string field with declared values.
    DECISION = "decision"
    #: The number a Repeat Group's statistics take.
    MEASUREMENT = "measurement"
    #: The change's one-line description, listed and compared with a read's text.
    SUMMARY = "summary"


class ObjectiveDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


#: The categories whose fields a declaration names, in record order.
CATEGORIES = ("identity", "baseline", "rationale", "change", "outcome", "reflection")
#: Where a field may join the Repeat Group key; every identity field always does.
_GROUP_CATEGORIES = frozenset({"baseline", "change"})
#: Where each role may be declared; each category holds at most one field per role.
_ROLE_CATEGORIES = {
    FieldRole.DECISION: frozenset({"outcome"}),
    FieldRole.MEASUREMENT: frozenset({"baseline", "outcome"}),
    FieldRole.SUMMARY: frozenset({"change"}),
}
#: The search weight a field gets when its declaration names none.
_DEFAULT_SEARCH = {"identity": 3.0}


def default_search_weight(category: str) -> float:
    """How heavily fuzzy matching weighs a field of ``category`` whose declaration names no weight."""

    return _DEFAULT_SEARCH.get(category, 0.0)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_token(value: Any, name: str) -> str:
    token = _require_string(value, name)
    if not _TOKEN_RE.fullmatch(token):
        raise SchemaValidationError(f"{name} contains unsupported characters")
    return token


def _require_name(value: Any, name: str) -> str:
    text = _require_string(value, name)
    if not _FIELD_NAME_RE.fullmatch(text):
        raise SchemaValidationError(f"{name} {text!r} must be a lowercase name such as 'framework_version'")
    return text


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


def _reject_unknown(value: dict[str, Any], allowed: AbstractSet[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SchemaValidationError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _notes(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{name} must be an object")
    notes: dict[str, str] = {}
    for label, text in value.items():
        notes[_require_name(label, f"{name} label")] = _require_string(text, f"{name}.{label}")
    return notes


@dataclass(frozen=True)
class FileRef:
    """A file an Experience carries: its name in the record, and the content the KB stores under its digest."""

    name: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        name = _require_string(self.name, "file.name")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or name.endswith("/"):
            raise SchemaValidationError(f"file.name {name!r} must be a relative path")
        object.__setattr__(self, "name", path.as_posix())
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise SchemaValidationError("file.sha256 must be 64 lowercase hex characters")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or not 0 <= self.bytes <= MAX_SAFE_INTEGER:
            raise SchemaValidationError("file.bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, value: Any) -> FileRef:
        data = _mapping(value, "file")
        _reject_unknown(data, {"name", "sha256", "bytes"}, "file")
        return cls(
            name=_required(data, "name", "file.name"),
            sha256=_required(data, "sha256", "file.sha256"),
            bytes=_required(data, "bytes", "file.bytes"),
        )


#: One value of a category field: a scalar, a text, a file, or a list of strings, texts, or files.
FieldValue: TypeAlias = str | int | float | bool | FileRef | tuple["str | FileRef", ...]


def _single_value(value: Any, name: str) -> str | int | float | bool | FileRef:
    if isinstance(value, FileRef):
        return value
    if isinstance(value, dict):
        return FileRef.from_dict(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = _finite_number(value, name)
        if isinstance(value, int):
            if abs(value) > MAX_SAFE_INTEGER:
                raise SchemaValidationError(f"{name} exceeds the IEEE-754-safe integer range")
            return value
        return number
    if isinstance(value, str):
        return value
    raise SchemaValidationError(f"{name} must be a string, number, boolean, or file")


def _field_values(value: Any, name: str, *, scalars_only: bool = False) -> dict[str, FieldValue]:
    """A category's fields, structurally valid; their kinds are the declaration's to check."""

    data = _mapping(value, name)
    result: dict[str, FieldValue] = {}
    for key, item in data.items():
        field_name = _require_name(key, f"{name} field")
        where = f"{name}.{field_name}"
        if isinstance(item, (list, tuple)):
            if scalars_only:
                raise SchemaValidationError(f"{where} must be a scalar, not a list")
            items = tuple(_single_value(entry, f"{where}[]") for entry in item)
            if not all(isinstance(entry, (str, FileRef)) for entry in items):
                raise SchemaValidationError(f"{where} must list strings, texts, or files")
            result[field_name] = items  # type: ignore[assignment]
            continue
        single = _single_value(item, where)
        if scalars_only and (isinstance(single, FileRef) or single == ""):
            raise SchemaValidationError(f"{where} must be a non-empty scalar")
        result[field_name] = single
    return result


def _encode(value: FieldValue) -> JsonValue:
    if isinstance(value, FileRef):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


def _files(value: FieldValue) -> Iterator[FileRef]:
    if isinstance(value, FileRef):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            if isinstance(item, FileRef):
                yield item


@dataclass(frozen=True)
class FieldDeclaration:
    """One field of a category: its kind, whether it must be present, and the roles KB functions read it in."""

    name: str
    description: str
    kind: FieldKind = FieldKind.STRING
    required: bool = False
    sensitive: bool = False
    many: bool = False
    group: bool = False
    search: float | None = None
    values: tuple[str, ...] = ()
    role: FieldRole | None = None

    def __post_init__(self) -> None:
        name = _require_name(self.name, "field.name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", _require_string(self.description, f"field {name!r} description"))
        if not isinstance(self.kind, FieldKind):
            raise SchemaValidationError(f"field {name!r} has an invalid kind")
        for flag in ("required", "sensitive", "many", "group"):
            _boolean(getattr(self, flag), f"field {name!r} {flag}")
        if self.many and self.kind not in {FieldKind.STRING, FieldKind.TEXT, FieldKind.FILE}:
            raise SchemaValidationError(f"field {name!r}: only string, text, and file fields may hold many values")
        values = tuple(_require_string(item, f"field {name!r} values[]") for item in self.values)
        if values and self.kind is not FieldKind.STRING:
            raise SchemaValidationError(f"field {name!r}: only a string field declares values")
        if len(set(values)) != len(values):
            raise SchemaValidationError(f"field {name!r} declares duplicate values")
        object.__setattr__(self, "values", values)
        if self.group and (self.kind not in SCALAR_KINDS or self.many):
            raise SchemaValidationError(f"field {name!r}: only a single scalar joins the Repeat Group key")
        if self.search is not None:
            weight = _finite_number(self.search, f"field {name!r} search")
            if weight < 0:
                raise SchemaValidationError(f"field {name!r} search weight must not be negative")
            if weight and self.kind is FieldKind.FILE:
                raise SchemaValidationError(f"field {name!r}: a file's content is not searched")
            object.__setattr__(self, "search", weight)
        if self.role is not None:
            if not isinstance(self.role, FieldRole):
                raise SchemaValidationError(f"field {name!r} has an invalid role")
            if self.many:
                raise SchemaValidationError(f"field {name!r}: a field with a role holds one value")
            if self.role is FieldRole.DECISION and (self.kind is not FieldKind.STRING or not values):
                raise SchemaValidationError(f"field {name!r}: the decision is a string field with declared values")
            if self.role is FieldRole.MEASUREMENT and self.kind is not FieldKind.NUMBER:
                raise SchemaValidationError(f"field {name!r}: the measurement is a number field")
            if self.role is FieldRole.SUMMARY and self.kind not in {FieldKind.STRING, FieldKind.TEXT}:
                raise SchemaValidationError(f"field {name!r}: the summary is a string or text field")

    def validate(self, value: FieldValue, where: str) -> None:
        """Check one present value against this field's kind."""

        if self.many:
            if not isinstance(value, tuple):
                raise SchemaValidationError(f"{where} must be a list")
            for index, item in enumerate(value):
                self._validate_one(item, f"{where}[{index}]")
            return
        if isinstance(value, tuple):
            raise SchemaValidationError(f"{where} holds one value, not a list")
        self._validate_one(value, where)

    def _validate_one(self, value: Any, where: str) -> None:
        kind = self.kind
        if kind is FieldKind.FILE:
            if not isinstance(value, FileRef):
                raise SchemaValidationError(f"{where} must be a file")
            return
        if isinstance(value, FileRef):
            raise SchemaValidationError(f"{where} must be a {kind.value}, not a file")
        if kind is FieldKind.NUMBER:
            _finite_number(value, where)
        elif kind is FieldKind.BOOLEAN:
            if not isinstance(value, bool):
                raise SchemaValidationError(f"{where} must be a boolean")
        elif not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(f"{where} must be a non-empty string")
        elif kind is FieldKind.TEXT and len(value.encode()) > TEXT_MAX_BYTES:
            raise SchemaValidationError(
                f"{where} holds {len(value.encode())} bytes of text, over the {TEXT_MAX_BYTES} a text field "
                "holds; declare the field as a file"
            )
        elif self.values and value not in self.values:
            raise SchemaValidationError(f"{where} must be one of: {', '.join(self.values)}")

    def search_weight(self, category: str) -> float:
        return self.search if self.search is not None else default_search_weight(category)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "required": self.required,
            "sensitive": self.sensitive,
            "many": self.many,
            "group": self.group,
            "search": self.search,
            "values": list(self.values),
            "role": None if self.role is None else self.role.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> FieldDeclaration:
        data = _mapping(value, "field declaration")
        _reject_unknown(
            data,
            {"name", "description", "kind", "required", "sensitive", "many", "group", "search", "values", "role"},
            "field declaration",
        )
        try:
            kind = FieldKind(data.get("kind", FieldKind.STRING.value))
        except ValueError as exc:
            raise SchemaValidationError(f"field declaration has an invalid kind {data.get('kind')!r}") from exc
        raw_role = data.get("role")
        try:
            role = None if raw_role is None else FieldRole(raw_role)
        except ValueError as exc:
            raise SchemaValidationError(f"field declaration has an invalid role {raw_role!r}") from exc
        return cls(
            name=_required(data, "name", "field declaration.name"),
            description=_required(data, "description", "field declaration.description"),
            kind=kind,
            required=_boolean(data.get("required", False), "field declaration.required"),
            sensitive=_boolean(data.get("sensitive", False), "field declaration.sensitive"),
            many=_boolean(data.get("many", False), "field declaration.many"),
            group=_boolean(data.get("group", False), "field declaration.group"),
            search=_optional_number(data.get("search"), "field declaration.search"),
            values=tuple(_list(data.get("values", []), "field declaration.values")),
            role=role,
        )


#: The rationale fields every schema has; a declaration adds its own beside them.
RATIONALE_DEFAULTS = (
    FieldDeclaration("preconditions", "Facts already known before the action.", FieldKind.TEXT, many=True),
    FieldDeclaration("reasoning", "Why the action was chosen.", FieldKind.TEXT, search=1.5),
    FieldDeclaration("alternatives", "Options considered, and why each was not chosen.", FieldKind.TEXT, many=True),
)


@dataclass(frozen=True)
class ObjectiveDeclaration:
    """What an attempt pursues, in words: for example maximizing one metric while another stays within a bound.

    ``unit``, ``direction``, and ``failure_value`` describe the measurement the objective is judged by, when it has one.
    """

    id: str
    description: str
    unit: str = ""
    direction: ObjectiveDirection | None = None
    failure_value: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "objective.id"))
        object.__setattr__(self, "description", _require_string(self.description, f"objective {self.id!r} description"))
        object.__setattr__(self, "unit", _string_or_empty(self.unit, f"objective {self.id!r} unit"))
        if self.direction is not None and not isinstance(self.direction, ObjectiveDirection):
            raise SchemaValidationError(f"objective {self.id!r} has an invalid direction")
        object.__setattr__(
            self, "failure_value", _optional_number(self.failure_value, f"objective {self.id!r} failure_value")
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "description": self.description,
            "unit": self.unit,
            "direction": None if self.direction is None else self.direction.value,
            "failure_value": self.failure_value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ObjectiveDeclaration:
        data = _mapping(value, "objective declaration")
        _reject_unknown(data, {"id", "description", "unit", "direction", "failure_value"}, "objective declaration")
        raw_direction = data.get("direction")
        try:
            direction = None if raw_direction is None else ObjectiveDirection(raw_direction)
        except ValueError as exc:
            raise SchemaValidationError(f"objective declaration has an invalid direction {raw_direction!r}") from exc
        return cls(
            id=_required(data, "id", "objective declaration.id"),
            description=_required(data, "description", "objective declaration.description"),
            unit=data.get("unit", ""),
            direction=direction,
            failure_value=data.get("failure_value"),
        )


@dataclass(frozen=True)
class ExperienceDeclaration:
    """One schema: the objectives an Experience pursues and the fields of each category."""

    objectives: tuple[ObjectiveDeclaration, ...]
    identity: tuple[FieldDeclaration, ...] = ()
    baseline: tuple[FieldDeclaration, ...] = ()
    rationale: tuple[FieldDeclaration, ...] = ()
    change: tuple[FieldDeclaration, ...] = ()
    outcome: tuple[FieldDeclaration, ...] = ()
    reflection: tuple[FieldDeclaration, ...] = ()
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _require_version(self.schema_version))
        object.__setattr__(self, "objectives", tuple(self.objectives))
        if not self.objectives or not all(isinstance(item, ObjectiveDeclaration) for item in self.objectives):
            raise SchemaValidationError("declaration.objectives must name at least one objective")
        if len({item.id for item in self.objectives}) != len(self.objectives):
            raise SchemaValidationError("declaration contains duplicate objectives")
        defaults = {item.name for item in RATIONALE_DEFAULTS}
        for category in CATEGORIES:
            fields = tuple(getattr(self, category))
            object.__setattr__(self, category, fields)
            if not all(isinstance(item, FieldDeclaration) for item in fields):
                raise SchemaValidationError(f"declaration.{category} contains an invalid field")
            names = [item.name for item in fields]
            if len(set(names)) != len(names):
                raise SchemaValidationError(f"declaration.{category} contains duplicate fields")
            if category == "rationale" and defaults & set(names):
                raise SchemaValidationError(
                    f"declaration.rationale redeclares a default field: {', '.join(sorted(defaults & set(names)))}"
                )
            for item in fields:
                self._check_field(category, item)
            for role in FieldRole:
                if sum(1 for item in fields if item.role is role) > 1:
                    raise SchemaValidationError(f"declaration.{category} gives role {role.value} to several fields")

    @staticmethod
    def _check_field(category: str, item: FieldDeclaration) -> None:
        where = f"declaration.{category}.{item.name}"
        if category == "identity" and (item.kind not in SCALAR_KINDS or item.many):
            raise SchemaValidationError(f"{where}: an identity field holds one scalar")
        if item.group and category not in _GROUP_CATEGORIES:
            raise SchemaValidationError(f"{where}: only baseline and change fields declare group")
        if item.role is not None and category not in _ROLE_CATEGORIES[item.role]:
            raise SchemaValidationError(f"{where}: role {item.role.value} belongs in another category")

    def fields(self, category: str) -> tuple[FieldDeclaration, ...]:
        """Every field of ``category``, the rationale defaults included."""

        declared: tuple[FieldDeclaration, ...] = getattr(self, category)
        return (*RATIONALE_DEFAULTS, *declared) if category == "rationale" else declared

    def field(self, category: str, name: str) -> FieldDeclaration | None:
        return next((item for item in self.fields(category) if item.name == name), None)

    def role_field(self, category: str, role: FieldRole) -> FieldDeclaration | None:
        return next((item for item in self.fields(category) if item.role is role), None)

    def objective(self, objective_id: str) -> ObjectiveDeclaration | None:
        return next((item for item in self.objectives if item.id == objective_id), None)

    @property
    def decision_values(self) -> tuple[str, ...]:
        """The values the outcome's decision field takes; empty when the schema declares none."""

        decision = self.role_field("outcome", FieldRole.DECISION)
        return decision.values if decision is not None else ()

    def group_fields(self) -> tuple[tuple[str, str], ...]:
        """The baseline and change fields of the Repeat Group key, beside the identity, objective, and schema."""

        return tuple(
            (category, item.name) for category in ("baseline", "change") for item in self.fields(category) if item.group
        )

    def lookup_fields(self) -> frozenset[str]:
        """The paths exact lookup indexes: every single-scalar field, as ``category.field``."""

        paths = {"schema_ref", "objective", "status"}
        for category in CATEGORIES:
            for item in self.fields(category):
                if item.kind in SCALAR_KINDS and not item.many:
                    paths.add(f"{category}.{item.name}")
        return frozenset(paths)

    def search_fields(self) -> tuple[tuple[str, FieldDeclaration, float], ...]:
        """Every field fuzzy matching reads, with its weight."""

        return tuple(
            (category, item, item.search_weight(category))
            for category in CATEGORIES
            for item in self.fields(category)
            if item.search_weight(category) > 0
        )

    @property
    def schema_ref(self) -> str:
        """Return the exact content-addressed reference for this declaration."""

        return derive_schema_ref(self)

    def validate(self, experience: Experience) -> None:
        if experience.schema_version != self.schema_version:
            raise SchemaValidationError("Experience and declaration schema versions differ")
        if experience.schema_ref != self.schema_ref:
            raise SchemaValidationError("Experience schema_ref does not match declaration")
        if self.objective(experience.objective) is None:
            raise SchemaValidationError(f"objective {experience.objective!r} is not declared")
        complete = experience.status is ExperienceStatus.COMPLETE
        for category in CATEGORIES:
            values: dict[str, FieldValue] = getattr(experience, category)
            declared = {item.name: item for item in self.fields(category)}
            # Identity keeps keys no field declares, so another consumer can index them later.
            if category != "identity":
                unknown = sorted(set(values) - set(declared))
                if unknown:
                    raise SchemaValidationError(f"{category} holds undeclared fields: {', '.join(unknown)}")
            for name, item in declared.items():
                if name in values:
                    item.validate(values[name], f"{category}.{name}")
                elif item.required and complete:
                    raise SchemaValidationError(f"{category} is missing required field {name!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "objectives": [item.to_dict() for item in self.objectives],
            **{category: [item.to_dict() for item in getattr(self, category)] for category in CATEGORIES},
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExperienceDeclaration:
        data = _mapping(value, "declaration")
        _reject_unknown(data, {"schema_version", "objectives", *CATEGORIES}, "declaration")
        objectives = _list(_required(data, "objectives", "declaration.objectives"), "declaration.objectives")
        return cls(
            schema_version=_require_version(_required(data, "schema_version", "declaration.schema_version")),
            objectives=tuple(ObjectiveDeclaration.from_dict(item) for item in objectives),
            **{
                category: tuple(
                    FieldDeclaration.from_dict(item)
                    for item in _list(data.get(category, []), f"declaration.{category}")
                )
                for category in CATEGORIES
            },
        )


@dataclass(frozen=True)
class RenderedRef:
    """A record rendered to the decision context, not proof of actual use."""

    id: str
    purpose: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "rendered_ref.id"))
        if self.purpose:
            object.__setattr__(self, "purpose", _require_token(self.purpose, "rendered_ref.purpose"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {"id": self.id, "purpose": self.purpose}

    @classmethod
    def from_dict(cls, value: Any) -> RenderedRef:
        data = _mapping(value, "rendered_ref")
        _reject_unknown(data, {"id", "purpose"}, "rendered_ref")
        return cls(
            id=_require_token(_required(data, "id", "rendered_ref.id"), "rendered_ref.id"),
            purpose=_string_or_empty(data.get("purpose", ""), "rendered_ref.purpose"),
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
                object.__setattr__(self, name, _require_string(value, f"provenance.{name}"))
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
            {"producer", "producer_version", "model", "prompt_version", "snapshot_version", "source_ref", "extra"},
            "provenance",
        )
        return cls(
            producer=_require_token(_required(data, "producer", "provenance.producer"), "provenance.producer"),
            producer_version=_require_string(
                _required(data, "producer_version", "provenance.producer_version"),
                "provenance.producer_version",
            ),
            model=_string_or_empty(data.get("model", ""), "provenance.model"),
            prompt_version=_string_or_empty(data.get("prompt_version", ""), "provenance.prompt_version"),
            snapshot_version=_string_or_empty(data.get("snapshot_version", ""), "provenance.snapshot_version"),
            source_ref=_string_or_empty(data.get("source_ref", ""), "provenance.source_ref"),
            extra=_json_object(data.get("extra", {}), "provenance.extra"),
        )


#: Top-level record fields that say what was learned: the conditions and objective, the baseline, why the action
#: was chosen, what changed, what came of it, how it reads, and notes outside the schema. A read shows these.
KNOWLEDGE_FIELDS = frozenset({"objective", *CATEGORIES, "notes"})
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
    """One attempt and what came of it, its fields grouped by category as its schema declares them."""

    id: str
    run_id: str
    seq: int
    created_at: datetime
    objective: str
    provenance: Provenance
    schema_ref: str
    identity: dict[str, FieldValue] = field(default_factory=dict)
    baseline: dict[str, FieldValue] = field(default_factory=dict)
    rationale: dict[str, FieldValue] = field(default_factory=dict)
    change: dict[str, FieldValue] = field(default_factory=dict)
    outcome: dict[str, FieldValue] = field(default_factory=dict)
    reflection: dict[str, FieldValue] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION
    status: ExperienceStatus = ExperienceStatus.IN_PROGRESS
    completed_at: datetime | None = None
    parent_id: str = ""
    supersedes: str = ""
    rendered_refs: tuple[RenderedRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_token(self.id, "experience.id"))
        object.__setattr__(self, "run_id", _require_token(self.run_id, "experience.run_id"))
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0 or self.seq > MAX_SAFE_INTEGER:
            raise SchemaValidationError("experience.seq must be a non-negative IEEE-754-safe integer")
        object.__setattr__(self, "schema_version", _require_version(self.schema_version))
        object.__setattr__(self, "schema_ref", _require_schema_ref(self.schema_ref, "experience.schema_ref"))
        if not isinstance(self.status, ExperienceStatus):
            raise SchemaValidationError("experience.status is invalid")
        object.__setattr__(self, "created_at", _aware_datetime(self.created_at, "experience.created_at"))
        object.__setattr__(self, "objective", _require_token(self.objective, "experience.objective"))
        if not isinstance(self.provenance, Provenance):
            raise SchemaValidationError("experience.provenance is invalid")
        if self.id != derive_experience_id(self.provenance.producer, self.run_id, self.seq):
            raise SchemaValidationError("experience.id does not match producer, run_id, and seq")
        for name in ("parent_id", "supersedes"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _require_token(value, f"experience.{name}"))
        for category in CATEGORIES:
            values = _field_values(getattr(self, category), category, scalars_only=category == "identity")
            object.__setattr__(self, category, values)
        object.__setattr__(self, "notes", _notes(self.notes, "experience.notes"))
        object.__setattr__(self, "rendered_refs", tuple(self.rendered_refs))
        if not all(isinstance(item, RenderedRef) for item in self.rendered_refs):
            raise SchemaValidationError("experience.rendered_refs contains an invalid item")
        rendered_ids = [item.id for item in self.rendered_refs]
        if len(set(rendered_ids)) != len(rendered_ids):
            raise SchemaValidationError("experience.rendered_refs contains duplicate ids")
        if self.status is ExperienceStatus.IN_PROGRESS:
            if self.completed_at is not None:
                raise SchemaValidationError("an in_progress Experience has no completed_at")
            return
        if self.completed_at is None:
            raise SchemaValidationError("a complete Experience requires completed_at")
        completed_at = _aware_datetime(self.completed_at, "experience.completed_at")
        if completed_at < self.created_at:
            raise SchemaValidationError("completed_at cannot be before created_at")
        object.__setattr__(self, "completed_at", completed_at)

    def files(self) -> tuple[FileRef, ...]:
        """Every file the record carries, each content once."""

        found: dict[str, FileRef] = {}
        for category in CATEGORIES:
            for value in getattr(self, category).values():
                for ref in _files(value):
                    found.setdefault(ref.sha256, ref)
        return tuple(found.values())

    def to_dict(self) -> dict[str, JsonValue]:
        record: dict[str, JsonValue] = {
            "kind": "experience",
            "id": self.id,
            "schema_ref": self.schema_ref,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "created_at": _format_datetime(self.created_at),
            "completed_at": _format_datetime(self.completed_at) if self.completed_at is not None else None,
            "run_id": self.run_id,
            "seq": self.seq,
            "parent_id": self.parent_id,
            "supersedes": self.supersedes,
            "objective": self.objective,
            "rendered_refs": [item.to_dict() for item in self.rendered_refs],
            "notes": dict(self.notes),
            "provenance": self.provenance.to_dict(),
        }
        for category in CATEGORIES:
            record[category] = {name: _encode(value) for name, value in getattr(self, category).items()}
        return record

    def knowledge(self) -> dict[str, JsonValue]:
        """The record's ``KNOWLEDGE_FIELDS``, exactly as ``to_dict`` holds them."""

        return {name: value for name, value in self.to_dict().items() if name in KNOWLEDGE_FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> Experience:
        data = _mapping(value, "experience")
        _reject_unknown(data, KNOWLEDGE_FIELDS | METADATA_FIELDS, "experience")
        if data.get("kind") != "experience":
            raise SchemaValidationError("experience.kind must be 'experience'")
        try:
            status = ExperienceStatus(_required(data, "status", "experience.status"))
        except ValueError as exc:
            raise SchemaValidationError("experience.status is invalid") from exc
        completed_at = data.get("completed_at")
        seq = _required(data, "seq", "experience.seq")
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise SchemaValidationError("experience.seq must be an integer")
        return cls(
            id=_required(data, "id", "experience.id"),
            run_id=_required(data, "run_id", "experience.run_id"),
            seq=seq,
            created_at=_parse_datetime(_required(data, "created_at", "experience.created_at"), "experience.created_at"),
            objective=_required(data, "objective", "experience.objective"),
            provenance=Provenance.from_dict(_required(data, "provenance", "experience.provenance")),
            schema_ref=_required(data, "schema_ref", "experience.schema_ref"),
            schema_version=_require_version(_required(data, "schema_version", "experience.schema_version")),
            status=status,
            completed_at=_parse_datetime(completed_at, "experience.completed_at") if completed_at is not None else None,
            parent_id=_string_or_empty(data.get("parent_id", ""), "experience.parent_id"),
            supersedes=_string_or_empty(data.get("supersedes", ""), "experience.supersedes"),
            rendered_refs=tuple(
                RenderedRef.from_dict(item) for item in _list(data.get("rendered_refs", []), "experience.rendered_refs")
            ),
            notes=data.get("notes", {}),
            **{category: data.get(category, {}) for category in CATEGORIES},
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
    "CATEGORIES",
    "CURRENT_SCHEMA_VERSION",
    "KNOWLEDGE_FIELDS",
    "METADATA_FIELDS",
    "RATIONALE_DEFAULTS",
    "SCALAR_KINDS",
    "TEXT_MAX_BYTES",
    "Experience",
    "ExperienceDeclaration",
    "ExperienceStatus",
    "FieldDeclaration",
    "FieldKind",
    "FieldRole",
    "FieldValue",
    "FileRef",
    "JsonScalar",
    "JsonValue",
    "ObjectiveDeclaration",
    "ObjectiveDirection",
    "Provenance",
    "RenderedRef",
    "SchemaValidationError",
    "UnsupportedSchemaVersion",
    "default_search_weight",
    "derive_schema_ref",
]
