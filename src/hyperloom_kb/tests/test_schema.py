from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom_kb import (
    KNOWLEDGE_FIELDS,
    METADATA_FIELDS,
    RATIONALE_DEFAULTS,
    MAX_FILE_BYTES,
    TEXT_MAX_BYTES,
    Experience,
    ExperienceDeclaration,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    FileRef,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Provenance,
    RenderedRef,
    SchemaValidationError,
    UnsupportedSchemaVersion,
    derive_experience_id,
)

NOW = datetime(2026, 9, 4, 20, 0, tzinfo=timezone.utc)
PATCH = FileRef("patches/kv.diff", "a" * 64, 812)


def declaration(*, failure_value: float | None = 0.0) -> ExperienceDeclaration:
    return ExperienceDeclaration(
        objectives=(
            ObjectiveDeclaration(
                id="throughput@v1",
                description="Maximize output token throughput while accuracy stays within its gate.",
                unit="token/s",
                direction=ObjectiveDirection.HIGHER_IS_BETTER,
                failure_value=failure_value,
            ),
        ),
        identity=(
            FieldDeclaration("model", "Model identifier.", required=True),
            FieldDeclaration("framework", "Serving framework.", required=True),
            FieldDeclaration("gpu", "GPU model.", required=True),
            FieldDeclaration("framework_version", "Serving framework version.", kind=FieldKind.VERSION),
        ),
        baseline=(
            FieldDeclaration("config", "Baseline configuration.", required=True, group=True),
            FieldDeclaration("value", "Baseline throughput.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
        ),
        change=(
            FieldDeclaration("knob", "Stable name of the changed setting.", required=True, group=True),
            FieldDeclaration("summary", "What the change does.", required=True, role=FieldRole.SUMMARY, search=4),
            FieldDeclaration("content", "The complete change.", kind=FieldKind.TEXT),
            FieldDeclaration("patches", "Patch files.", kind=FieldKind.FILE, many=True),
        ),
        outcome=(
            FieldDeclaration(
                "decision",
                "Kept, reverted, or failed.",
                required=True,
                role=FieldRole.DECISION,
                values=("keep", "revert", "failed"),
            ),
            FieldDeclaration("value", "Throughput with the change.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
            FieldDeclaration("accuracy_passed", "Whether the accuracy gate passed.", kind=FieldKind.BOOLEAN),
        ),
        reflection=(FieldDeclaration("text", "How the outcome reads.", kind=FieldKind.TEXT, required=True),),
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
        baseline={"config": "default", "value": 2400},
        schema_ref=declaration().schema_ref,
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
        rationale={
            "preconditions": ("Baseline correctness passed.",),
            "reasoning": "KV cache bandwidth dominates decode.",
            "alternatives": ("Increase tensor parallelism: the workload fits on one GPU.",),
        },
        change={
            "knob": "kv_cache_dtype",
            "summary": "Use fp8_e4m3 for the KV cache.",
            "content": "--kv-cache-dtype fp8_e4m3",
            "patches": (PATCH,),
        },
        rendered_refs=(RenderedRef("exp-100", "starting_point"),),
        outcome={"decision": "keep", "value": 2610.5, "accuracy_passed": True},
        reflection={"text": "The change improved throughput without violating accuracy."},
        notes={"interconnect": "XGMI links saturate during the all-reduce."},
    )


def test_complete_experience_round_trips_as_json() -> None:
    original = complete_experience()

    encoded = json.dumps(original.to_dict(), sort_keys=True)
    restored = Experience.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.to_dict()["created_at"] == "2026-09-04T20:00:00Z"
    assert restored.change["patches"] == (PATCH,)
    assert restored.files() == (PATCH,)
    declaration().validate(restored)


def test_every_record_field_is_either_knowledge_or_metadata() -> None:
    record = complete_experience().to_dict()

    assert KNOWLEDGE_FIELDS | METADATA_FIELDS == set(record)
    assert not KNOWLEDGE_FIELDS & METADATA_FIELDS
    assert complete_experience().knowledge() == {name: record[name] for name in KNOWLEDGE_FIELDS}


def test_identity_keeps_undeclared_scalars_and_holds_no_list_or_file() -> None:
    experience = in_progress_experience()

    declaration().validate(experience)
    assert experience.identity["tenant"] == "extra-dimension-is-preserved"
    for value, message in (
        (["a", "b"], "must be a scalar, not a list"),
        (PATCH.to_dict(), "non-empty scalar"),
        ("", "non-empty scalar"),
        ({"name": "qwen3"}, "file"),
    ):
        with pytest.raises(SchemaValidationError, match=message):
            replace(experience, identity={**experience.identity, "model": value})


def test_other_categories_hold_only_declared_fields() -> None:
    experience = complete_experience()

    with pytest.raises(SchemaValidationError, match="change holds undeclared fields: kind"):
        declaration().validate(replace(experience, change={**experience.change, "kind": "config"}))


def test_rationale_carries_kb_defaults_and_rejects_their_redeclaration() -> None:
    assert [item.name for item in RATIONALE_DEFAULTS] == ["preconditions", "reasoning", "alternatives"]
    extended = replace(
        declaration(), rationale=(FieldDeclaration("evidence", "Profiles consulted.", kind=FieldKind.TEXT),)
    )
    assert [item.name for item in extended.fields("rationale")] == [
        "preconditions",
        "reasoning",
        "alternatives",
        "evidence",
    ]

    with pytest.raises(SchemaValidationError, match="redeclares a default field: reasoning"):
        replace(declaration(), rationale=(FieldDeclaration("reasoning", "Why.", kind=FieldKind.TEXT),))


def test_required_fields_are_checked_only_once_the_record_is_complete() -> None:
    started = replace(in_progress_experience(), baseline={"value": 2400})

    declaration().validate(started)
    with pytest.raises(SchemaValidationError, match="baseline is missing required field 'config'"):
        declaration().validate(replace(complete_experience(), baseline={"value": 2400}))
    with pytest.raises(SchemaValidationError, match="outcome is missing required field 'decision'"):
        declaration().validate(replace(complete_experience(), outcome={"value": 1.0}))


def test_declaration_enforces_kinds_and_values() -> None:
    valid = complete_experience()

    for category, values, message in (
        ("identity", {**valid.identity, "model": 3}, "identity.model must be a non-empty string"),
        ("outcome", {**valid.outcome, "decision": "skip"}, "must be one of: keep, revert, failed"),
        ("outcome", {**valid.outcome, "value": "fast"}, "outcome.value must be a number"),
        ("outcome", {**valid.outcome, "accuracy_passed": "yes"}, "must be a boolean"),
        ("change", {**valid.change, "patches": PATCH}, "change.patches must be a list"),
        ("change", {**valid.change, "content": PATCH}, "must be a text, not a file"),
        ("change", {**valid.change, "patches": ("x.diff",)}, r"change.patches\[0\] must be a file"),
    ):
        with pytest.raises(SchemaValidationError, match=message):
            declaration().validate(replace(valid, **{category: values}))
    with pytest.raises(SchemaValidationError, match="'unknown' is not declared"):
        declaration().validate(replace(valid, objective="unknown"))


def test_a_text_field_holds_at_most_32_kib_and_points_larger_content_to_a_file() -> None:
    valid = complete_experience()
    at_limit = replace(valid, change={**valid.change, "content": "x" * TEXT_MAX_BYTES})

    declaration().validate(at_limit)
    with pytest.raises(SchemaValidationError, match="declare the field as a file"):
        declaration().validate(replace(valid, change={**valid.change, "content": "x" * (TEXT_MAX_BYTES + 1)}))


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"kind": FieldKind.NUMBER, "many": True}, "only string, text, and file"),
        ({"kind": FieldKind.NUMBER, "values": ("a",)}, "only a string field declares values"),
        ({"kind": FieldKind.TEXT, "group": True}, "only a single scalar"),
        ({"kind": FieldKind.FILE, "search": 1}, "a file's content is not searched"),
        ({"role": FieldRole.DECISION}, "string field with declared values"),
        ({"role": FieldRole.MEASUREMENT}, "the measurement is a number field"),
        ({"kind": FieldKind.FILE, "role": FieldRole.SUMMARY}, "string or text field"),
        ({"kind": FieldKind.TEXT, "many": True, "role": FieldRole.SUMMARY}, "holds one value"),
    ],
)
def test_field_declaration_rejects_attribute_combinations_no_kb_function_reads(attributes, message) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        FieldDeclaration("x", "X.", **attributes)


def test_declaration_places_roles_and_groups_only_where_kb_functions_read_them() -> None:
    base = declaration()
    for overrides, message in (
        ({"identity": (FieldDeclaration("notes", "N.", kind=FieldKind.TEXT),)}, "identity field holds one scalar"),
        ({"outcome": (*base.outcome, FieldDeclaration("phase", "P.", group=True))}, "only baseline and change"),
        (
            {"change": (*base.change, FieldDeclaration("v", "V.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT))},
            "role measurement belongs in another category",
        ),
        (
            {
                "outcome": (
                    *base.outcome,
                    FieldDeclaration("v2", "V.", kind=FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
                )
            },
            "gives role measurement to several fields",
        ),
        ({"identity": (FieldDeclaration("gpu", "GPU."), FieldDeclaration("gpu", "Again."))}, "duplicate fields"),
        ({"objectives": ()}, "at least one objective"),
    ):
        with pytest.raises(SchemaValidationError, match=message):
            replace(base, **overrides)


def test_declaration_derives_what_kb_functions_read() -> None:
    schema = declaration()

    assert schema.group_fields() == (("baseline", "config"), ("change", "knob"))
    assert schema.decision_values == ("keep", "revert", "failed")
    assert {"identity.gpu", "baseline.config", "outcome.decision", "outcome.accuracy_passed"} <= schema.lookup_fields()
    assert not {"change.content", "change.patches", "rationale.reasoning"} & schema.lookup_fields()
    weights = {(category, item.name): weight for category, item, weight in schema.search_fields()}
    assert weights[("identity", "gpu")] == 3.0
    assert weights[("change", "summary")] == 4.0
    assert weights[("rationale", "reasoning")] == 1.5
    assert ("change", "content") not in weights


def test_declaration_round_trips_and_its_schema_ref_is_content_addressed() -> None:
    current = declaration()

    assert ExperienceDeclaration.from_dict(json.loads(json.dumps(current.to_dict()))) == current
    assert current.schema_ref == declaration().schema_ref
    assert current.schema_ref.startswith("schema:sha256:")
    assert current.schema_ref != declaration(failure_value=-1).schema_ref
    with pytest.raises(SchemaValidationError, match="schema_ref does not match"):
        declaration(failure_value=-1).validate(complete_experience())
    missing_ref = complete_experience().to_dict()
    missing_ref.pop("schema_ref")
    with pytest.raises(SchemaValidationError, match="schema_ref"):
        Experience.from_dict(missing_ref)


def test_a_file_over_the_size_a_kb_stores_is_refused_by_the_record_naming_it() -> None:
    FileRef("profile.bin", "a" * 64, MAX_FILE_BYTES)

    with pytest.raises(SchemaValidationError, match="file.bytes"):
        FileRef("profile.bin", "a" * 64, MAX_FILE_BYTES + 1)


def test_a_string_that_is_not_utf_8_fails_validation_like_any_invalid_value() -> None:
    valid = complete_experience()
    split_emoji = "the patch \ud83d helped"

    for changed in (
        {"rationale": {**valid.rationale, "reasoning": split_emoji}},
        {"identity": {**valid.identity, "model": split_emoji}},
        {"notes": {"interconnect": split_emoji}},
    ):
        with pytest.raises(SchemaValidationError, match="not valid UTF-8"):
            declaration().validate(replace(valid, **changed))


def test_notes_take_labelled_text_only() -> None:
    noted = complete_experience()

    for notes, message in (
        ({"Interconnect": "text"}, "lowercase name"),
        ({"interconnect": ""}, "non-empty string"),
        ({"interconnect": {"nested": "text"}}, "non-empty string"),
        (["interconnect"], "must be an object"),
    ):
        with pytest.raises(SchemaValidationError, match=message):
            replace(noted, notes=notes)


def test_completion_fields_follow_the_status() -> None:
    started = in_progress_experience()

    with pytest.raises(SchemaValidationError, match="has no completed_at"):
        replace(started, completed_at=NOW)
    with pytest.raises(SchemaValidationError, match="requires completed_at"):
        replace(started, status=ExperienceStatus.COMPLETE)
    with pytest.raises(SchemaValidationError, match="cannot be before created_at"):
        replace(started, status=ExperienceStatus.COMPLETE, completed_at=NOW - timedelta(seconds=1))


def test_file_refs_name_a_relative_path_and_a_sha256() -> None:
    for kwargs, message in (
        ({"name": "../outside.diff"}, "relative path"),
        ({"name": "/abs.diff"}, "relative path"),
        ({"sha256": "A" * 64}, "64 lowercase hex"),
        ({"bytes": -1}, "non-negative integer"),
    ):
        with pytest.raises(SchemaValidationError, match=message):
            replace(PATCH, **kwargs)


def test_schema_rejects_unknown_versions_and_fields() -> None:
    document = complete_experience().to_dict()
    document["schema_version"] = 3
    with pytest.raises(UnsupportedSchemaVersion):
        Experience.from_dict(document)

    document = complete_experience().to_dict()
    document["confidence"] = 0.9
    with pytest.raises(SchemaValidationError, match="unknown fields: confidence"):
        Experience.from_dict(document)


def test_schema_rejects_non_finite_values() -> None:
    with pytest.raises(SchemaValidationError, match="must be finite"):
        replace(in_progress_experience(), baseline={"value": float("nan")})

    with pytest.raises(SchemaValidationError, match="non-finite"):
        Provenance(producer="hyperloom", producer_version="1", extra={"bad": float("inf")})


def test_experience_sequence_must_round_trip_through_javascript_numbers() -> None:
    too_large = 1 << 53
    with pytest.raises(SchemaValidationError, match="IEEE-754-safe"):
        replace(
            complete_experience(),
            seq=too_large,
            id=derive_experience_id("hyperloom", "run-456", too_large),
        )
