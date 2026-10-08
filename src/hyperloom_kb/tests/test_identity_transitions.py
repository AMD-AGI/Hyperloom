from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom_kb import (
    Change,
    Experience,
    ExperienceConflictError,
    ExperienceStatus,
    Outcome,
    Provenance,
    SchemaValidationError,
    TransitionKind,
    classify_transition,
    derive_experience_id,
    validate_supersession,
)

NOW = datetime(2026, 9, 4, 21, 0, tzinfo=timezone.utc)
SCHEMA_REF = f"schema:sha256:{'a' * 64}"


def beginning(*, seq: int = 0, supersedes: str = "") -> Experience:
    return Experience(
        id=derive_experience_id("hyperloom", "run-1", seq),
        run_id="run-1",
        seq=seq,
        created_at=NOW + timedelta(minutes=seq),
        identity={"model": "qwen3-8b", "gpu": "mi355x"},
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=2400,
        schema_ref=SCHEMA_REF,
        provenance=Provenance("hyperloom", "1.0.0"),
        supersedes=supersedes,
        preconditions=("Baseline passed.",),
    )


def decided() -> Experience:
    return replace(
        beginning(),
        reasoning="Decode is memory bound.",
        change=Change(
            identity={"knob": "kv_cache_dtype"},
            summary="Use fp8 KV cache.",
        ),
    )


def completed() -> Experience:
    return replace(
        decided(),
        status=ExperienceStatus.COMPLETE,
        completed_at=NOW + timedelta(minutes=1),
        outcome=Outcome(decision="keep", value=2600),
        reflection="The measured result improved.",
    )


def test_experience_id_is_stable_and_separates_identity_inputs() -> None:
    first = derive_experience_id("hyperloom", "run-1", 0)

    assert first == derive_experience_id("hyperloom", "run-1", 0)
    assert first.startswith("exp-")
    assert len(first) == 36
    assert (
        len(
            {
                first,
                derive_experience_id("other", "run-1", 0),
                derive_experience_id("hyperloom", "run-2", 0),
                derive_experience_id("hyperloom", "run-1", 1),
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    ("producer", "run_id", "seq"),
    [
        ("", "run", 0),
        ("producer", "", 0),
        ("producer", "run", -1),
        ("producer", "run", True),
    ],
)
def test_experience_id_rejects_invalid_inputs(
    producer: str,
    run_id: str,
    seq: int,
) -> None:
    with pytest.raises(ValueError):
        derive_experience_id(producer, run_id, seq)


def test_experience_rejects_caller_supplied_non_deterministic_id() -> None:
    with pytest.raises(SchemaValidationError, match="does not match"):
        replace(beginning(), id="exp-arbitrary")


def test_monotonic_begin_decide_complete_transitions() -> None:
    start = beginning()
    choice = decided()
    result = completed()

    assert classify_transition(start, start) is TransitionKind.NOOP
    assert classify_transition(start, choice) is TransitionKind.DECIDED
    assert classify_transition(choice, result) is TransitionKind.COMPLETED
    assert classify_transition(start, result) is TransitionKind.COMPLETED
    assert classify_transition(result, result) is TransitionKind.NOOP


def test_begin_time_facts_cannot_change() -> None:
    changed = replace(beginning(), identity={"model": "other", "gpu": "mi355x"})

    with pytest.raises(ExperienceConflictError, match="identity"):
        classify_transition(beginning(), changed)


def test_decision_time_facts_cannot_change_during_completion() -> None:
    changed = replace(
        decided(),
        status=ExperienceStatus.COMPLETE,
        completed_at=NOW + timedelta(minutes=1),
        reasoning="A different rationale.",
        outcome=Outcome(decision="keep", value=2600),
        reflection="Improved.",
    )

    with pytest.raises(ExperienceConflictError, match="reasoning"):
        classify_transition(decided(), changed)


def test_complete_record_rejects_non_identical_replay() -> None:
    mutated = replace(completed(), reflection="Edited after publication.")

    with pytest.raises(ExperienceConflictError, match="immutable"):
        classify_transition(completed(), mutated)


def test_correction_requires_new_id_and_supersedes_link() -> None:
    previous = completed()
    correction = beginning(seq=1, supersedes=previous.id)

    validate_supersession(previous, correction)

    with pytest.raises(ExperienceConflictError, match="new sequence"):
        validate_supersession(previous, replace(previous, supersedes=previous.id))
    with pytest.raises(ExperienceConflictError, match="must name"):
        validate_supersession(previous, beginning(seq=1))
