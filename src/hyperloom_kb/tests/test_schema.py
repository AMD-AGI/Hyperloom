from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom_kb import (
    KNOWLEDGE_FIELDS,
    METADATA_FIELDS,
    Alternative,
    Change,
    ConstraintResult,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    Provenance,
    RenderedRef,
    SchemaValidationError,
    UnsupportedSchemaVersion,
    derive_experience_id,
)

NOW = datetime(2026, 9, 4, 20, 0, tzinfo=timezone.utc)


def declaration(*, failure_value: float | None = 0.0) -> ExperienceDeclaration:
    return ExperienceDeclaration(
        identity=(
            FieldDeclaration("model", "Model identifier."),
            FieldDeclaration("framework", "Serving framework."),
            FieldDeclaration("gpu", "GPU model."),
            FieldDeclaration(
                "framework_version",
                "Serving framework version.",
                kind=FieldKind.VERSION,
                required=False,
            ),
        ),
        baseline_identity=(FieldDeclaration("config", "Baseline configuration."),),
        change_identity=(FieldDeclaration("knob", "Stable name of the changed setting."),),
        objectives=(
            ObjectiveDeclaration(
                id="throughput@v1",
                direction=ObjectiveDirection.HIGHER_IS_BETTER,
                description="Output token throughput.",
                unit="token/s",
                failure_value=failure_value,
            ),
        ),
        decisions=("keep", "revert", "failed"),
    )


def in_progress_experience() -> Experience:
    return Experience(
        id=derive_experience_id("hyperloom", "run-456", 2),
        run_id="run-456",
        seq=2,
        created_at=NOW,
        identity={
            "model": "qwen3-8b",
            "framework": "sglang",
            "gpu": "mi355x",
            "tenant": "extra-dimension-is-preserved",
        },
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=2400,
        schema_ref=declaration().schema_ref,
        preconditions=("Baseline correctness passed.",),
        provenance=Provenance(
            producer="hyperloom",
            producer_version="1.0.0",
            model="claude-sonnet",
            prompt_version="7",
        ),
    )


def complete_experience() -> Experience:
    return replace(
        in_progress_experience(),
        status=ExperienceStatus.COMPLETE,
        completed_at=NOW + timedelta(minutes=12),
        reasoning="KV cache bandwidth dominates decode.",
        alternatives=(
            Alternative(
                option="Increase tensor parallelism.",
                why_not="The workload fits on one GPU.",
            ),
        ),
        change=Change(
            kind="config_delta",
            identity={"knob": "kv_cache_dtype"},
            summary="Use fp8_e4m3 for the KV cache.",
            content="--kv-cache-dtype fp8_e4m3",
            resource_refs=("artifacts/measurement.json",),
        ),
        rendered_refs=(RenderedRef("exp-100", "starting_point"),),
        outcome=Outcome(
            decision="keep",
            value=2610.5,
            constraints=(ConstraintResult("accuracy_delta", True, -0.2),),
        ),
        reflection="The change improved throughput without violating accuracy.",
    )


def test_complete_experience_round_trips_as_json() -> None:
    original = complete_experience()

    encoded = json.dumps(original.to_dict(), sort_keys=True)
    restored = Experience.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.to_dict()["created_at"] == "2026-09-04T20:00:00Z"
    assert restored.identity["tenant"] == "extra-dimension-is-preserved"
    declaration().validate(restored)


def test_every_record_field_is_either_knowledge_or_metadata() -> None:
    record = complete_experience().to_dict()

    assert KNOWLEDGE_FIELDS | METADATA_FIELDS == set(record)
    assert not KNOWLEDGE_FIELDS & METADATA_FIELDS
    assert complete_experience().knowledge() == {name: record[name] for name in KNOWLEDGE_FIELDS}


def test_schema_ref_is_content_addressed_and_required() -> None:
    current = declaration()
    same = declaration()
    changed = declaration(failure_value=-1)

    assert current.schema_ref == same.schema_ref
    assert current.schema_ref.startswith("schema:sha256:")
    assert current.schema_ref != changed.schema_ref

    missing_ref = complete_experience().to_dict()
    missing_ref.pop("schema_ref")
    with pytest.raises(SchemaValidationError, match="schema_ref"):
        Experience.from_dict(missing_ref)


def test_declaration_rejects_mismatched_schema_ref() -> None:
    with pytest.raises(SchemaValidationError, match="schema_ref does not match"):
        declaration(failure_value=-1).validate(complete_experience())


def test_in_progress_experience_has_no_synthetic_outcome() -> None:
    experience = in_progress_experience()

    assert experience.status is ExperienceStatus.IN_PROGRESS
    assert experience.completed_at is None
    assert experience.outcome is None
    declaration().validate(experience)

    with pytest.raises(SchemaValidationError, match="cannot have completion fields"):
        replace(experience, outcome=Outcome(decision="failed", error_class="crash"))


def test_complete_failure_requires_declared_failure_value() -> None:
    failed = replace(
        in_progress_experience(),
        status=ExperienceStatus.COMPLETE,
        completed_at=NOW + timedelta(seconds=5),
        reasoning="The server failed during warmup.",
        change=Change(
            identity={"knob": "torch_compile"},
            summary="Enable torch compile.",
        ),
        outcome=Outcome(decision="failed", error_class="warmup_failed"),
        reflection="No throughput measurement was produced.",
    )

    declaration().validate(failed)
    no_failure_value = declaration(failure_value=None)
    with pytest.raises(SchemaValidationError, match="requires failure_value"):
        no_failure_value.validate(replace(failed, schema_ref=no_failure_value.schema_ref))


def test_complete_experience_requires_decision_time_fields() -> None:
    with pytest.raises(SchemaValidationError, match="requires change"):
        replace(
            in_progress_experience(),
            status=ExperienceStatus.COMPLETE,
            completed_at=NOW,
            outcome=Outcome(decision="keep", value=2500),
            reflection="Improved.",
        )


def test_declaration_enforces_required_fields_types_and_vocabulary() -> None:
    valid = complete_experience()

    with pytest.raises(SchemaValidationError, match="missing required field 'gpu'"):
        declaration().validate(replace(valid, identity={"model": "qwen3", "framework": "sglang"}))

    with pytest.raises(SchemaValidationError, match="'model' must be a string"):
        declaration().validate(replace(valid, identity={**valid.identity, "model": 3}))

    with pytest.raises(SchemaValidationError, match="'unknown' is not declared"):
        declaration().validate(replace(valid, objective="unknown"))

    with pytest.raises(SchemaValidationError, match="'skip' is not declared"):
        declaration().validate(replace(valid, outcome=replace(valid.outcome, decision="skip")))


def test_declaration_round_trip_preserves_sensitive_and_version_metadata() -> None:
    original = ExperienceDeclaration(
        identity=(
            FieldDeclaration(
                "tenant",
                "Stable tenant identifier.",
                sensitive=True,
            ),
        ),
        baseline_identity=(
            FieldDeclaration(
                "config",
                "Baseline configuration.",
            ),
        ),
        change_identity=(
            FieldDeclaration(
                "framework_version",
                "Version being changed.",
                kind=FieldKind.VERSION,
            ),
        ),
        objectives=(
            ObjectiveDeclaration(
                "latency@v2",
                ObjectiveDirection.LOWER_IS_BETTER,
                "Request latency.",
                unit="ms",
            ),
        ),
        decisions=("keep", "revert"),
    )

    assert ExperienceDeclaration.from_dict(original.to_dict()) == original


def test_schema_rejects_unknown_versions_and_fields() -> None:
    document = complete_experience().to_dict()
    document["schema_version"] = 3
    with pytest.raises(UnsupportedSchemaVersion):
        Experience.from_dict(document)

    document = complete_experience().to_dict()
    document["confidence"] = 0.9
    with pytest.raises(SchemaValidationError, match="unknown fields: confidence"):
        Experience.from_dict(document)


def test_schema_rejects_non_finite_and_nested_identity_values() -> None:
    with pytest.raises(SchemaValidationError, match="must be finite"):
        replace(in_progress_experience(), baseline_value=float("nan"))

    with pytest.raises(SchemaValidationError, match="must be a scalar"):
        replace(in_progress_experience(), identity={"model": {"name": "qwen3"}})

    with pytest.raises(SchemaValidationError, match="non-finite"):
        Provenance(
            producer="hyperloom",
            producer_version="1",
            extra={"bad": float("inf")},
        )


def test_experience_sequence_must_round_trip_through_javascript_numbers() -> None:
    too_large = 1 << 53
    with pytest.raises(SchemaValidationError, match="IEEE-754-safe"):
        replace(
            complete_experience(),
            seq=too_large,
            id=derive_experience_id("hyperloom", "run-456", too_large),
        )


def test_change_rejects_unsafe_or_duplicate_resource_paths() -> None:
    with pytest.raises(SchemaValidationError, match="safe relative path"):
        Change(
            identity={"knob": "kernel"},
            summary="Replace kernel.",
            resource_refs=("../outside.patch",),
        )

    with pytest.raises(SchemaValidationError, match="must not contain duplicates"):
        Change(
            identity={"knob": "kernel"},
            summary="Replace kernel.",
            resource_refs=("solution.patch", "solution.patch"),
        )


def test_duplicate_declaration_entries_are_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="duplicate decision"):
        replace(declaration(), decisions=("keep", "keep"))

    with pytest.raises(SchemaValidationError, match="duplicate identity field"):
        replace(
            declaration(),
            identity=(
                FieldDeclaration("gpu", "GPU."),
                FieldDeclaration("gpu", "GPU again."),
            ),
        )
