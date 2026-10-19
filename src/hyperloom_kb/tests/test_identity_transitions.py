from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from hyperloom_kb import (
    Experience,
    Provenance,
    SchemaValidationError,
    derive_experience_id,
)

NOW = datetime(2026, 9, 4, 21, 0, tzinfo=timezone.utc)
SCHEMA_REF = f"schema:sha256:{'a' * 64}"


def beginning() -> Experience:
    return Experience(
        id=derive_experience_id("hyperloom", "run-1", 0),
        run_id="run-1",
        seq=0,
        created_at=NOW,
        identity={"model": "qwen3-8b", "gpu": "mi355x"},
        objective="throughput@v1",
        baseline={"config": "default", "value": 2400},
        schema_ref=SCHEMA_REF,
        provenance=Provenance("hyperloom", "1.0.0"),
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
