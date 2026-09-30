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
from hyperloom_kb.collect.mapping import CollectMapping, UnitStep, load_mapping
from hyperloom_kb.collect.sensitive import find_sensitive
from hyperloom_kb.config import ConfigurationError
from hyperloom_kb.identity import derive_experience_id
from hyperloom_kb.remote import RemoteWriteResult
from hyperloom_kb.runtime import experience_kb_from_env
from hyperloom_kb.schema import (
    Change,
    ConstraintResult,
    Experience,
    ExperienceStatus,
    JsonScalar,
    JsonValue,
    Outcome,
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
        identity: dict[str, JsonScalar],
        objective: str,
        baseline_identity: dict[str, JsonScalar],
        baseline_value: float,
        provenance: Provenance,
        preconditions: tuple[str, ...] = (),
        parent_id: str = "",
        supersedes: str = "",
        created_at: datetime | None = None,
    ) -> Any:
        """Open the Experience session for ``run_id``/``seq``, through which its decision and outcome are recorded."""


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


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a number")
    return float(value)


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


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


def _project(mapping: CollectMapping, scope: Mapping[str, Any]) -> Experience:
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
    change = Change(
        identity=_object(value("change.identity"), "change.identity"),
        summary=_text(value("change.summary"), "change.summary"),
        kind=_text(value("change.kind"), "change.kind", required=False),
        content=_text(value("change.content"), "change.content", required=False),
        resource_refs=tuple(
            _text(item, "change.resource_refs[]")
            for item in _items(value("change.resource_refs"), "change.resource_refs")
        ),
    )
    outcome = Outcome(
        decision=_text(value("outcome.decision"), "outcome.decision"),
        value=_optional_number(value("outcome.value"), "outcome.value"),
        constraints=tuple(
            ConstraintResult.from_dict(item) for item in _items(value("outcome.constraints"), "outcome.constraints")
        ),
        error_class=_text(value("outcome.error_class"), "outcome.error_class", required=False),
    )
    experience = Experience(
        id=derive_experience_id(producer.name, run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=created_at,
        identity=_object(value("identity"), "identity"),
        objective=_text(value("objective"), "objective"),
        baseline_identity=_object(value("baseline_identity"), "baseline_identity"),
        baseline_value=_number(value("baseline_value"), "baseline_value"),
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
        preconditions=tuple(_text(item, "preconditions[]") for item in _items(value("preconditions"), "preconditions")),
        reasoning=_text(value("reasoning"), "reasoning"),
        change=change,
        rendered_refs=tuple(RenderedRef.from_dict(item) for item in _items(value("rendered_refs"), "rendered_refs")),
        outcome=outcome,
        reflection=_text(value("reflection"), "reflection"),
    )
    mapping.declaration.validate(experience)
    return experience


def _screen(experience: Experience) -> None:
    payload = experience.to_dict()
    finding = find_sensitive(payload)
    if finding:
        raise _Skip(f"sensitive content in {finding}")


def _prepare(mapping: CollectMapping, unit: dict[str, Any]) -> Experience:
    scope = dict(unit)
    for name, lookup in mapping.lookups:
        matches = lookup.evaluate(scope)
        scope[name] = matches[0] if matches else None
    for name, expression in mapping.lets:
        scope[name] = expression.evaluate(scope)
    for rule in mapping.require:
        if not truth(rule.condition.evaluate(scope)):
            raise _Skip(rule.reason)
    experience = _project(mapping, scope)
    _screen(experience)
    return experience


def _publish(target: CollectionTarget, experience: Experience) -> str:
    if experience.change is None or experience.outcome is None:
        raise ValueError("only a complete projected Experience can be published")
    session = target.begin(
        run_id=experience.run_id,
        seq=experience.seq,
        identity=dict(experience.identity),
        objective=experience.objective,
        baseline_identity=dict(experience.baseline_identity),
        baseline_value=experience.baseline_value,
        provenance=experience.provenance,
        preconditions=experience.preconditions,
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
    if record.change is None:
        session.decide(
            reasoning=experience.reasoning,
            change=experience.change,
            rendered_refs=experience.rendered_refs,
        )
    elif record.change != experience.change or record.reasoning != experience.reasoning:
        raise ExperienceConflictError("an in-progress Experience holds a different decision")
    session.complete(
        outcome=experience.outcome,
        reflection=experience.reflection,
        completed_at=experience.completed_at,
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
    touching any KB.
    """

    compiled = mapping if isinstance(mapping, CollectMapping) else load_mapping(mapping)
    if isinstance(document, Mapping):
        source, document_ref = document, "<memory>"
    else:
        source, document_ref = load_document(document), str(document)
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
            experience = _prepare(compiled, unit)
        except _Skip as skip:
            skipped.append(SkippedUnit(unit_id, skip.reason))
            continue
        except EvaluationError as exc:
            skipped.append(SkippedUnit(unit_id, f"mapping evaluation failed: {exc}"))
            continue
        except SchemaValidationError as exc:
            skipped.append(SkippedUnit(unit_id, f"schema validation failed: {exc}"))
            continue
        if target is None:
            collected.append(CollectedExperience(unit_id, experience.id, "dry_run", experience))
            continue
        try:
            status = _publish(target, experience)
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
