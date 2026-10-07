"""Turn one source document into complete Experiences through a mapping."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from hyperloom_kb.collect.expressions import (
    EvaluationError,
    is_present,
    truth,
)
from hyperloom_kb.collect.mapping import DECLARED_CATEGORIES, CollectMapping, UnitStep, load_mapping
from hyperloom_kb.collect.sensitive import FREE_TEXT_FIELDS, find_sensitive, find_sensitive_in_file
from hyperloom_kb.config import ConfigurationError
from hyperloom_kb.files import file_ref
from hyperloom_kb.identity import derive_experience_id
from hyperloom_kb.remote import RemoteWriteResult, SessionValue
from hyperloom_kb.runtime import experience_kb_from_env
from hyperloom_kb.schema import (
    Experience,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    FieldValue,
    FileRef,
    JsonValue,
    Provenance,
    RenderedRef,
    SchemaValidationError,
)
from hyperloom_kb.transitions import ExperienceConflictError

REPORT_FORMAT = "hyperloom-kb.collect-report.v1"


class SourceDocumentError(ValueError):
    """Raised when the source document cannot be read or has the wrong shape."""


class _Skip(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CollectionTarget(Protocol):
    """What collection needs of a configured Experience KB."""

    @property
    def enabled(self) -> bool:
        """Whether the KB is configured to take writes."""

    @property
    def schema_ref(self) -> str:
        """The schema this KB's Experiences are written under."""

    def begin(
        self,
        *,
        run_id: str,
        seq: int,
        objective: str,
        provenance: Provenance,
        identity: Mapping[str, FieldValue] | None = None,
        baseline: Mapping[str, SessionValue] | None = None,
        parent_id: str = "",
        supersedes: str = "",
        created_at: datetime | None = None,
    ) -> Any:
        """Open the Experience session for ``run_id``/``seq``, through which its decision and outcome are recorded."""


@dataclass(frozen=True)
class _Projected:
    experience: Experience
    #: Where each file the record names is read from, by its sha256.
    files: dict[str, Path]


@dataclass(frozen=True)
class CollectedExperience:
    unit_id: str
    experience_id: str
    status: str
    experience: Experience


@dataclass(frozen=True)
class SkippedUnit:
    unit_id: str
    reason: str


@dataclass(frozen=True)
class FailedUnit:
    unit_id: str
    error: str


@dataclass(frozen=True)
class CollectReport:
    mapping: str
    schema_ref: str
    document: str
    dry_run: bool
    enabled: bool
    blocked_reason: str = ""
    collected: tuple[CollectedExperience, ...] = ()
    skipped: tuple[SkippedUnit, ...] = ()
    errors: tuple[FailedUnit, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize; a dry run carries each projected Experience for review."""

        by_status: dict[str, int] = {}
        for item in self.collected:
            by_status[item.status] = by_status.get(item.status, 0) + 1
        counts: dict[str, JsonValue] = {
            "units": len(self.collected) + len(self.skipped) + len(self.errors),
            "collected": len(self.collected),
            "skipped": len(self.skipped),
            "errors": len(self.errors),
            "by_status": dict(sorted(by_status.items())),
        }
        collected: list[JsonValue] = []
        for item in self.collected:
            row: dict[str, JsonValue] = {
                "unit_id": item.unit_id,
                "experience_id": item.experience_id,
                "status": item.status,
            }
            if self.dry_run:
                row["experience"] = item.experience.to_dict()
            collected.append(row)
        return {
            "format": REPORT_FORMAT,
            "mapping": self.mapping,
            "schema_ref": self.schema_ref,
            "document": self.document,
            "dry_run": self.dry_run,
            "enabled": self.enabled,
            "blocked_reason": self.blocked_reason,
            "counts": counts,
            "collected": collected,
            "skipped": [{"unit_id": item.unit_id, "reason": item.reason} for item in self.skipped],
            "errors": [{"unit_id": item.unit_id, "error": item.error} for item in self.errors],
        }


def load_document(path: str | Path) -> Mapping[str, Any]:
    """Read one JSON source document."""

    source = Path(path).expanduser()
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SourceDocumentError(f"cannot read source document {source}: {exc}") from exc
    except ValueError as exc:
        raise SourceDocumentError(f"cannot parse source document {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SourceDocumentError(f"source document {source} is not a JSON object")
    return value


def _units(steps: tuple[UnitStep, ...], scope: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if not steps:
        yield scope
        return
    step = steps[0]
    items = step.items.evaluate(scope)
    if items is None:
        return
    if not isinstance(items, list):
        raise EvaluationError(f"unit step {step.name!r} expects a list, got {type(items).__name__}")
    for item in items:
        local = {**scope, step.name: item}
        if step.where is None or truth(step.where.evaluate(local)):
            yield from _units(steps[1:], local)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None or value == "":
        if required:
            raise EvaluationError(f"{name} is missing")
        return ""
    if not isinstance(value, str):
        raise EvaluationError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{name} must be an integer")
    return value


def _time(value: Any, name: str) -> datetime:
    text = _text(value, name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvaluationError(f"{name} is not an ISO-8601 timestamp: {value!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _items(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationError(f"{name} must be a list")
    return value


def _local_file(value: Any, base: Path, files: dict[str, Path], name: str) -> FileRef:
    path = Path(_text(value, name)).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise EvaluationError(f"{name} names {path}, which is not a file")
    ref = file_ref(path)
    files[ref.sha256] = path
    return ref


def _field_value(field: FieldDeclaration, value: Any, base: Path, files: dict[str, Path], name: str) -> FieldValue:
    """One category field as the record holds it: a file field's local path becomes the ``FileRef`` of its file."""

    if field.many:
        items = _items(value, name)
        if field.kind is FieldKind.FILE:
            return tuple(_local_file(item, base, files, f"{name}[]") for item in items)
        return tuple(items)
    if field.kind is FieldKind.FILE:
        return _local_file(value, base, files, name)
    return value  # type: ignore[no-any-return]


def _category(
    mapping: CollectMapping, category: str, scope: Mapping[str, Any], base: Path, files: dict[str, Path]
) -> dict[str, FieldValue]:
    values: dict[str, FieldValue] = {}
    for field in mapping.declaration.fields(category):
        expression = mapping.experience.get(f"{category}.{field.name}")
        value = None if expression is None else expression.evaluate(scope)
        if is_present(value):
            values[field.name] = _field_value(field, value, base, files, f"{category}.{field.name}")
    return values


def _project(mapping: CollectMapping, scope: Mapping[str, Any], base: Path) -> _Projected:
    fields = mapping.experience

    def value(name: str) -> Any:
        expression = fields.get(name)
        return None if expression is None else expression.evaluate(scope)

    producer = mapping.producer
    run_id = _text(value("run_id"), "run_id")
    seq = _integer(value("seq"), "seq")
    completed_at = _time(value("completed_at"), "completed_at")
    created_raw = value("created_at")
    created_at = _time(created_raw, "created_at") if is_present(created_raw) else completed_at
    files: dict[str, Path] = {}

    def category(name: str) -> dict[str, FieldValue]:
        return _category(mapping, name, scope, base, files)

    experience = Experience(
        id=derive_experience_id(producer.name, run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=created_at,
        identity={key: item for key, item in _object(value("identity"), "identity").items() if item is not None},
        objective=_text(value("objective"), "objective"),
        baseline=category("baseline"),
        rationale=category("rationale"),
        change=category("change"),
        outcome=category("outcome"),
        reflection=category("reflection"),
        provenance=Provenance(
            producer=producer.name,
            producer_version=producer.version,
            model=producer.model,
            prompt_version=producer.prompt_version,
            snapshot_version=producer.snapshot_version,
            source_ref=_text(value("provenance.source_ref"), "provenance.source_ref", required=False),
            extra=_object(value("provenance.extra"), "provenance.extra"),
        ),
        schema_ref=mapping.declaration.schema_ref,
        status=ExperienceStatus.COMPLETE,
        completed_at=completed_at,
        parent_id=_text(value("parent_id"), "parent_id", required=False),
        supersedes=_text(value("supersedes"), "supersedes", required=False),
        rendered_refs=tuple(RenderedRef.from_dict(item) for item in _items(value("rendered_refs"), "rendered_refs")),
        notes={
            str(label): _text(text, f"notes.{label}")
            for label, text in _object(value("notes"), "notes").items()
            if is_present(text)
        },
    )
    mapping.declaration.validate(experience)
    return _Projected(experience, files)


def _screen(mapping: CollectMapping, projected: _Projected) -> None:
    summary = mapping.declaration.role_field("change", FieldRole.SUMMARY)
    free_text = FREE_TEXT_FIELDS | ({f"change.{summary.name}"} if summary is not None else set())
    experience = projected.experience
    finding = find_sensitive(experience.to_dict(), free_text=free_text)
    if finding:
        raise _Skip(f"sensitive content in {finding}")
    for category in DECLARED_CATEGORIES:
        for name, value in getattr(experience, category).items():
            path = f"{category}.{name}"
            prose = any(path == item or path.startswith(f"{item}.") for item in free_text)
            for ref in value if isinstance(value, tuple) else (value,):
                if isinstance(ref, FileRef):
                    finding = find_sensitive_in_file(projected.files[ref.sha256], free_text=prose)
                    if finding:
                        raise _Skip(f"sensitive content in {path} file {ref.name}, {finding}")


def _prepare(mapping: CollectMapping, unit: dict[str, Any], base: Path) -> _Projected:
    scope = dict(unit)
    for name, lookup in mapping.lookups:
        matches = lookup.evaluate(scope)
        scope[name] = matches[0] if matches else None
    for name, expression in mapping.lets:
        scope[name] = expression.evaluate(scope)
    for rule in mapping.require:
        if not truth(rule.condition.evaluate(scope)):
            raise _Skip(rule.reason)
    projected = _project(mapping, scope, base)
    _screen(mapping, projected)
    return projected


def _with_paths(values: Mapping[str, FieldValue], files: Mapping[str, Path]) -> dict[str, SessionValue]:
    """``values`` as a session takes them: each ``FileRef`` as the local file it was made from."""

    def one(value: str | FileRef) -> str | FileRef | Path:
        return files[value.sha256] if isinstance(value, FileRef) else value

    return {
        name: tuple(one(item) for item in value) if isinstance(value, tuple) else one(value)  # type: ignore[arg-type]
        for name, value in values.items()
    }


def _publish(target: CollectionTarget, projected: _Projected) -> str:
    experience, files = projected.experience, projected.files
    session = target.begin(
        run_id=experience.run_id,
        seq=experience.seq,
        objective=experience.objective,
        provenance=experience.provenance,
        identity=dict(experience.identity),
        baseline=_with_paths(experience.baseline, files),
        parent_id=experience.parent_id,
        supersedes=experience.supersedes,
        created_at=experience.created_at,
    )
    record = session.record
    if not isinstance(record, Experience):
        raise RuntimeError("collection target did not open an Experience session")
    if record.status is ExperienceStatus.COMPLETE:
        if record != experience:
            raise ExperienceConflictError("a different complete Experience already has this id")
        return "unchanged"
    if not record.change:
        session.decide(
            change=_with_paths(experience.change, files),
            rationale=_with_paths(experience.rationale, files),
            rendered_refs=experience.rendered_refs,
        )
    elif record.change != experience.change or record.rationale != experience.rationale:
        raise ExperienceConflictError("an in-progress Experience holds a different decision")
    session.complete(
        outcome=_with_paths(experience.outcome, files),
        reflection=_with_paths(experience.reflection, files),
        completed_at=experience.completed_at,
        notes=experience.notes,
    )
    result = session.publish()
    if result is None:
        raise RuntimeError(f"collection target degraded: {getattr(target, 'reason', '')}")
    if isinstance(result, RemoteWriteResult):
        return result.status or "published"
    return "written"


def write_report(path: str | Path, report: CollectReport) -> None:
    """Atomically write a collection report as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def collect(
    mapping: CollectMapping | str | Path,
    document: Mapping[str, Any] | str | Path,
    *,
    kb: CollectionTarget | None = None,
    dry_run: bool = False,
    receipt: str | Path | None = None,
) -> CollectReport:
    """Project every unit of ``document`` through ``mapping`` and publish the results.

    Without ``kb`` the target comes from :func:`experience_kb_from_env` and
    writes the mapping's own declaration; an unconfigured environment returns a
    disabled report and writes nothing. A dry run projects and validates without
    touching any KB. A file field's relative path is read from the document's
    directory, or from the working directory for a document given in memory.
    """

    compiled = mapping if isinstance(mapping, CollectMapping) else load_mapping(mapping)
    if isinstance(document, Mapping):
        source, document_ref, base = document, "<memory>", Path.cwd()
    else:
        source, document_ref = load_document(document), str(document)
        base = Path(document).expanduser().resolve().parent
    schema_ref = compiled.declaration.schema_ref

    target: CollectionTarget | None = None
    if not dry_run:
        target = experience_kb_from_env(declaration=compiled.declaration) if kb is None else kb
        if not target.enabled:
            report = CollectReport(compiled.reference, schema_ref, document_ref, dry_run, False)
            if receipt is not None:
                write_report(receipt, report)
            return report
        if target.schema_ref != schema_ref:
            raise ConfigurationError(
                f"mapping {compiled.reference!r} produces {schema_ref}, but the configured "
                f"Experience KB accepts {target.schema_ref}"
            )

    root: dict[str, Any] = {"doc": source}
    blocked = next((rule.reason for rule in compiled.skip if truth(rule.condition.evaluate(root))), "")
    collected: list[CollectedExperience] = []
    skipped: list[SkippedUnit] = []
    errors: list[FailedUnit] = []
    try:
        units = list(_units(compiled.units, root))
    except EvaluationError as exc:
        raise SourceDocumentError(f"document does not match mapping units: {exc}") from exc
    for index, unit in enumerate(units):
        unit_id = str((compiled.unit_id.evaluate(unit) if compiled.unit_id is not None else None) or f"unit-{index}")
        if blocked:
            skipped.append(SkippedUnit(unit_id, blocked))
            continue
        try:
            projected = _prepare(compiled, unit, base)
        except _Skip as skip:
            skipped.append(SkippedUnit(unit_id, skip.reason))
            continue
        except EvaluationError as exc:
            skipped.append(SkippedUnit(unit_id, f"mapping evaluation failed: {exc}"))
            continue
        except SchemaValidationError as exc:
            skipped.append(SkippedUnit(unit_id, f"schema validation failed: {exc}"))
            continue
        experience = projected.experience
        if target is None:
            collected.append(CollectedExperience(unit_id, experience.id, "dry_run", experience))
            continue
        try:
            status = _publish(target, projected)
        except (LookupError, OSError, RuntimeError, ValueError) as exc:
            errors.append(FailedUnit(unit_id, f"{type(exc).__name__}: {exc}"))
            continue
        collected.append(CollectedExperience(unit_id, experience.id, status, experience))

    report = CollectReport(
        mapping=compiled.reference,
        schema_ref=schema_ref,
        document=document_ref,
        dry_run=dry_run,
        enabled=True,
        blocked_reason=blocked,
        collected=tuple(collected),
        skipped=tuple(skipped),
        errors=tuple(errors),
    )
    if receipt is not None:
        write_report(receipt, report)
    return report


__all__ = [
    "REPORT_FORMAT",
    "CollectReport",
    "CollectedExperience",
    "CollectionTarget",
    "FailedUnit",
    "SkippedUnit",
    "SourceDocumentError",
    "collect",
    "load_document",
    "write_report",
]
