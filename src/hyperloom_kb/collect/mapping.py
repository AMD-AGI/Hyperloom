"""Load and compile a collection mapping.

A mapping names the declaration it produces, the units of a source document
that each become one Experience, and one expression per Experience field.
Names are bound in declaration order -- ``doc``, then each unit step, then
lookups, then ``let`` values -- and every expression may read only the names
bound before it.

``identity`` is one expression that evaluates to the whole identity. Each other
category is a mapping from the fields its declaration names to one expression
each; a file field's expression evaluates to the path of a local file, relative
to the source document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom_kb.collect.expressions import (
    Each,
    Expression,
    MappingError,
    compile_expression,
)
from hyperloom_kb.config import ConfigurationError, load_declaration
from hyperloom_kb.schema import ExperienceDeclaration

MAPPING_FORMAT = "hyperloom-kb.collect.v2"
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PACKAGED_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOP_LEVEL = frozenset(
    {
        "format",
        "description",
        "declaration",
        "producer",
        "skip",
        "units",
        "unit_id",
        "lookup",
        "let",
        "require",
        "experience",
    }
)
_PRODUCER_FIELDS = {
    "name": True,
    "version": True,
    "snapshot_version": False,
    "model": False,
    "prompt_version": False,
}
_EXPERIENCE_FIELDS: dict[str, bool | dict[str, bool]] = {
    "run_id": True,
    "seq": True,
    "created_at": False,
    "completed_at": True,
    "identity": False,
    "objective": True,
    "provenance": {"source_ref": False, "extra": False},
    "rendered_refs": False,
    "notes": False,
    "parent_id": False,
    "supersedes": False,
}
#: The categories whose fields a mapping maps one by one, as the declaration names them.
DECLARED_CATEGORIES = ("baseline", "rationale", "change", "outcome", "reflection")


@dataclass(frozen=True)
class Producer:
    name: str
    version: str
    snapshot_version: str = ""
    model: str = ""
    prompt_version: str = ""


@dataclass(frozen=True)
class Rule:
    reason: str
    condition: Expression


@dataclass(frozen=True)
class UnitStep:
    name: str
    items: Expression
    where: Expression | None


@dataclass(frozen=True)
class CollectMapping:
    """A compiled mapping from one source document shape to one declaration."""

    reference: str
    description: str
    declaration: ExperienceDeclaration
    producer: Producer
    skip: tuple[Rule, ...]
    units: tuple[UnitStep, ...]
    unit_id: Expression | None
    lookups: tuple[tuple[str, Each], ...]
    lets: tuple[tuple[str, Expression], ...]
    require: tuple[Rule, ...]
    #: One expression per record field; a category field is keyed ``category.field``.
    experience: dict[str, Expression]


def packaged_path(directory: str, name: str) -> Path | None:
    """Return a packaged ``<directory>/<name>.yaml`` resource when ``name`` is a bare name."""

    if not _PACKAGED_NAME.fullmatch(name):
        return None
    candidate = _PACKAGE_ROOT / directory / f"{name}.yaml"
    return candidate if candidate.is_file() else None


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MappingError(f"{where} must be a mapping")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MappingError(f"{where} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, where: str, bound: set[str]) -> str:
    name = _string(value, where)
    if not _IDENTIFIER.fullmatch(name):
        raise MappingError(f"{where}: {name!r} is not an identifier")
    if name in bound:
        raise MappingError(f"{where}: {name!r} is already bound")
    return name


def _reject_unknown(data: Mapping[str, Any], allowed: frozenset[str] | set[str], where: str) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise MappingError(f"{where} has unknown keys: {', '.join(unknown)}")


def _declaration(value: Any, base: Path | None) -> ExperienceDeclaration:
    reference = _string(value, "declaration")
    path = packaged_path("declarations", reference)
    if path is None:
        candidate = Path(reference).expanduser()
        path = candidate if candidate.is_absolute() or base is None else base / candidate
    try:
        return load_declaration(path)
    except ConfigurationError as exc:
        raise MappingError(f"declaration {reference!r}: {exc}") from exc


def _producer(value: Any) -> Producer:
    data = _mapping(value, "producer")
    _reject_unknown(data, set(_PRODUCER_FIELDS), "producer")
    fields: dict[str, str] = {}
    for name, required in _PRODUCER_FIELDS.items():
        if name in data:
            fields[name] = _string(data[name], f"producer.{name}")
        elif required:
            raise MappingError(f"producer.{name} is required")
    return Producer(**fields)


def _rules(value: Any, where: str, key: str, names: frozenset[str]) -> tuple[Rule, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MappingError(f"{where} must be a list")
    rules: list[Rule] = []
    for index, item in enumerate(value):
        location = f"{where}[{index}]"
        data = _mapping(item, location)
        _reject_unknown(data, {"reason", key}, location)
        if key not in data:
            raise MappingError(f"{location}.{key} is required")
        rules.append(
            Rule(
                _string(data.get("reason"), f"{location}.reason"),
                compile_expression(data[key], names, f"{location}.{key}"),
            )
        )
    return tuple(rules)


def _experience(value: Any, names: frozenset[str], declaration: ExperienceDeclaration) -> dict[str, Expression]:
    data = _mapping(value, "experience")
    _reject_unknown(data, {*_EXPERIENCE_FIELDS, *DECLARED_CATEGORIES}, "experience")
    compiled: dict[str, Expression] = {}
    for category in DECLARED_CATEGORIES:
        section = _mapping(data.get(category) or {}, f"experience.{category}")
        declared = {item.name: item for item in declaration.fields(category)}
        _reject_unknown(section, set(declared), f"experience.{category}")
        for name, field in declared.items():
            if name in section:
                compiled[f"{category}.{name}"] = compile_expression(
                    section[name], names, f"experience.{category}.{name}"
                )
            elif field.required:
                raise MappingError(f"experience.{category}.{name} is required by the declaration")
    for name, spec in _EXPERIENCE_FIELDS.items():
        if isinstance(spec, dict):
            if name not in data:
                if any(spec.values()):
                    raise MappingError(f"experience.{name} is required")
                continue
            section = _mapping(data[name], f"experience.{name}")
            _reject_unknown(section, set(spec), f"experience.{name}")
            for child, required in spec.items():
                if child in section:
                    compiled[f"{name}.{child}"] = compile_expression(
                        section[child], names, f"experience.{name}.{child}"
                    )
                elif required:
                    raise MappingError(f"experience.{name}.{child} is required")
            continue
        if name in data:
            compiled[name] = compile_expression(data[name], names, f"experience.{name}")
        elif spec:
            raise MappingError(f"experience.{name} is required")
    return compiled


def compile_mapping(
    value: Any,
    *,
    reference: str = "<memory>",
    base: Path | None = None,
) -> CollectMapping:
    """Compile an already-parsed mapping document."""

    data = _mapping(value, "mapping")
    _reject_unknown(data, _TOP_LEVEL, "mapping")
    if data.get("format") != MAPPING_FORMAT:
        raise MappingError(f"mapping format must be {MAPPING_FORMAT!r}")
    declaration = _declaration(data.get("declaration"), base)
    producer = _producer(data.get("producer"))

    bound = {"doc"}
    skip = _rules(data.get("skip"), "skip", "when", frozenset(bound))

    raw_units = data.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise MappingError("units must be a non-empty list")
    units: list[UnitStep] = []
    for index, item in enumerate(raw_units):
        location = f"units[{index}]"
        step = _mapping(item, location)
        _reject_unknown(step, {"each", "as", "where"}, location)
        if "each" not in step:
            raise MappingError(f"{location}.each is required")
        items = compile_expression(step["each"], frozenset(bound), f"{location}.each")
        name = _identifier(step.get("as"), f"{location}.as", bound)
        bound.add(name)
        where = compile_expression(step["where"], frozenset(bound), f"{location}.where") if "where" in step else None
        units.append(UnitStep(name, items, where))

    unit_id = compile_expression(data["unit_id"], frozenset(bound), "unit_id") if "unit_id" in data else None

    lookups: list[tuple[str, Each]] = []
    for name, spec in _mapping(data.get("lookup") or {}, "lookup").items():
        location = f"lookup.{name}"
        result_name = _identifier(name, location, bound)
        entry = _mapping(spec, location)
        _reject_unknown(entry, {"from", "as", "where"}, location)
        if "from" not in entry or "where" not in entry:
            raise MappingError(f"{location} requires from and where")
        item_name = _identifier(entry.get("as"), f"{location}.as", bound | {result_name})
        lookups.append(
            (
                result_name,
                Each(
                    compile_expression(entry["from"], frozenset(bound), f"{location}.from"),
                    item_name,
                    compile_expression(entry["where"], frozenset(bound | {item_name}), f"{location}.where"),
                    None,
                ),
            )
        )
        bound.add(result_name)

    lets: list[tuple[str, Expression]] = []
    for name, expression in _mapping(data.get("let") or {}, "let").items():
        location = f"let.{name}"
        let_name = _identifier(name, location, bound)
        lets.append((let_name, compile_expression(expression, frozenset(bound), location)))
        bound.add(let_name)

    names = frozenset(bound)
    return CollectMapping(
        reference=reference,
        description=str(data.get("description") or "").strip(),
        declaration=declaration,
        producer=producer,
        skip=skip,
        units=tuple(units),
        unit_id=unit_id,
        lookups=tuple(lookups),
        lets=tuple(lets),
        require=_rules(data.get("require"), "require", "check", names),
        experience=_experience(data.get("experience"), names, declaration),
    )


def load_mapping(reference: str | Path) -> CollectMapping:
    """Load a mapping by packaged name (for example ``hyperloom-sbd-v6``) or file path."""

    text = str(reference)
    path = packaged_path("mappings", text) if isinstance(reference, str) else None
    if path is None:
        path = Path(reference).expanduser()
    try:
        value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MappingError(f"cannot read collection mapping {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MappingError(f"cannot parse collection mapping {path}: {exc}") from exc
    return compile_mapping(value, reference=text, base=path.parent)


__all__ = [
    "DECLARED_CATEGORIES",
    "MAPPING_FORMAT",
    "CollectMapping",
    "Producer",
    "Rule",
    "UnitStep",
    "compile_mapping",
    "load_mapping",
    "packaged_path",
]
