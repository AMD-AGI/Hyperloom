# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from importlib import resources
from typing import Any

import pytest
from hyperloom.measure.tests.bundle_factory import DEFAULT_FILES, manifest_document
from jsonschema import Draft202012Validator

from hyperloom.contracts.measurement import PhaseTiming
from hyperloom.measure import signing
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.manifest import (
    CardOccupancyFacts,
    ResultManifest,
    RulePackFacts,
    encode_manifest,
    manifest_sha256,
)
from hyperloom.measure.phases import PhaseRecorder, manifest_document as phases_document

SCHEMA = json.loads(resources.files("hyperloom.contracts").joinpath("measurement_result.v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
PUBLIC = signing.public_key_pem(signing.generate_private_key())
PACK_DIGEST = "sha256:" + "ab" * 32


def _document() -> dict[str, Any]:
    return manifest_document(DEFAULT_FILES, PUBLIC)


def _edit(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    document = copy.deepcopy(_document())
    change(document)
    return document


def test_a_manifest_round_trips_and_passes_the_schema() -> None:
    document = _document()
    assert list(VALIDATOR.iter_errors(document)) == []
    manifest = ResultManifest.from_bytes(encode_manifest(document))
    assert manifest.request_id == document["request_id"] and manifest.total_size() == sum(
        map(len, DEFAULT_FILES.values())
    )


def test_a_manifest_states_the_rule_pack_its_runner_read_and_one_from_an_older_runner_states_none() -> None:
    stated = _edit(lambda d: d["runner"].update(rule_pack={"version": "2-test", "digest": PACK_DIGEST}))
    assert list(VALIDATOR.iter_errors(stated)) == []
    assert ResultManifest.from_document(stated).runner.rule_pack == RulePackFacts("2-test", PACK_DIGEST)
    assert "rule_pack" not in _document()["runner"]
    assert ResultManifest.from_document(_document()).runner.rule_pack is None


@pytest.mark.parametrize(
    ("stated", "expected"),
    [
        ({"state": "clean", "reason": None, "checks": 7}, CardOccupancyFacts("clean", None, 7)),
        ({"state": "clean", "checks": 3}, CardOccupancyFacts("clean", None, 3)),
        (
            {"state": "unknown", "reason": "no AMD GPU driver is loaded", "checks": 0},
            CardOccupancyFacts("unknown", "no AMD GPU driver is loaded", 0),
        ),
        (
            {"state": "foreign", "reason": "1 held the test's cards: pid 9 on card 0", "checks": 4},
            CardOccupancyFacts("foreign", "1 held the test's cards: pid 9 on card 0", 4),
        ),
        (
            {"state": "clean", "reason": None, "checks": 9, "method": "census+fdinfo"},
            CardOccupancyFacts("clean", None, 9, "census+fdinfo"),
        ),
        (
            {"state": "unknown", "reason": "no driver", "checks": 0, "method": None},
            CardOccupancyFacts("unknown", "no driver", 0),
        ),
    ],
)
def test_a_manifest_states_whether_its_cards_were_free_of_other_processes_and_why_not_when_it_could_not_say(
    stated: dict[str, Any], expected: CardOccupancyFacts
) -> None:
    document = _edit(lambda d: d["runner"].update(card_occupancy=stated))

    assert list(VALIDATOR.iter_errors(document)) == []
    assert ResultManifest.from_document(document).runner.card_occupancy == expected


def test_a_manifest_of_an_older_runner_or_of_a_runner_with_no_card_states_no_occupancy() -> None:
    assert "card_occupancy" not in _document()["runner"]
    assert ResultManifest.from_document(_document()).runner.card_occupancy is None
    null = _edit(lambda d: d["runner"].update(card_occupancy=None))
    assert list(VALIDATOR.iter_errors(null)) == []
    assert ResultManifest.from_document(null).runner.card_occupancy is None


def _timed(phases: list[dict[str, Any]]) -> dict[str, Any]:
    return _edit(lambda d: d["timing"].update(phases=phases))


def test_a_manifest_states_the_phases_its_runner_went_through_and_one_from_an_older_runner_states_none() -> None:
    recorder = PhaseRecorder(iter([10.0, 14.0, 20.0]).__next__)
    recorder.begin("sheet_wait")
    recorder.begin("server_boot", point=0)
    stated = _timed(phases_document(recorder.finish()))
    assert list(VALIDATOR.iter_errors(stated)) == []
    phases = ResultManifest.from_bytes(encode_manifest(stated)).timing.phases
    assert phases == (PhaseTiming("sheet_wait", 10.0, 4.0), PhaseTiming("server_boot", 14.0, 6.0, point=0))
    assert "phases" not in _document()["timing"]
    assert ResultManifest.from_document(_document()).timing.phases == ()


PHASE = {"name": "warmup", "started_unix": 1.0, "seconds": 2.0, "point": 0}
BAD_PHASES = {
    "a phase of another shape": [{**PHASE, "extra": 1}],
    "a phase with no length": [{"name": "warmup", "started_unix": 1.0}],
    "a negative length": [{**PHASE, "seconds": -1.0}],
    "a name that is not a token": [{**PHASE, "name": "Server Boot"}],
    "a name with a line break": [{**PHASE, "name": "boot\nforged"}],
    "a negative point": [{**PHASE, "point": -1}],
    "phases that are not a list": {"warmup": 2.0},
    "a phase that lasts years": [{**PHASE, "seconds": 1e9}],
    "a phase that starts past the year 5000": [{**PHASE, "started_unix": 1e12}],
    "more phases than a test can have": [PHASE] * 4097,
}


def test_a_manifest_whose_phases_add_up_to_more_than_a_year_is_refused_by_the_reader() -> None:
    day = 24 * 3600
    document = _timed([{**PHASE, "seconds": 200.0 * day}, {**PHASE, "seconds": 200.0 * day}])

    with pytest.raises(DocumentError, match="add up to more than a year"):
        ResultManifest.from_document(document)


def test_a_manifest_may_state_its_phases_as_null_as_it_may_its_other_optional_timing() -> None:
    document = _edit(lambda d: d["timing"].update(phases=None))

    assert list(VALIDATOR.iter_errors(document)) == []
    assert ResultManifest.from_document(document).timing.phases == ()


@pytest.mark.parametrize("name", sorted(BAD_PHASES))
def test_a_manifest_whose_phases_break_the_contract_is_refused_by_the_reader_and_the_schema(name: str) -> None:
    document = _timed(BAD_PHASES[name])  # type: ignore[arg-type]
    with pytest.raises(DocumentError):
        ResultManifest.from_document(document)
    assert list(VALIDATOR.iter_errors(document)) != [], "the schema accepts what the reader refuses"


def test_the_manifest_hash_is_the_hash_of_its_raw_bytes() -> None:
    raw = encode_manifest(_document())
    assert manifest_sha256(raw) == manifest_sha256(raw) != manifest_sha256(raw + b" ")


def test_the_encoding_is_canonical() -> None:
    document = _document()
    assert encode_manifest(document) == encode_manifest(dict(reversed(list(document.items()))))


NEGATIVES: dict[str, Callable[[dict[str, Any]], None]] = {
    "unknown key": lambda d: d.update(extra=1),
    "short spec fingerprint": lambda d: d.update(spec_fingerprint="abc"),
    "bad spec_sha256": lambda d: d.update(spec_sha256="zz"),
    "bad key fingerprint": lambda d: d["runner"].update(signing_key_fingerprint="sha256:abc"),
    "negative gpu id": lambda d: d["runner"].update(gpu_ids=[-1]),
    "bad failure class": lambda d: d["outcome"].update(failure_class="oops"),
    "failure class on a passing test": lambda d: d["outcome"].update(failure_class="infra", failure_reason="x"),
    "failed test with no class": lambda d: d["outcome"].update(returncode=1),
    "class without reason": lambda d: d["outcome"].update(returncode=1, failure_class="infra"),
    "negative timing": lambda d: d["timing"].update(finished_unix=-1),
    "negative size": lambda d: d["files"][0].update(size=-1),
    "duplicate file": lambda d: d["files"].append(copy.deepcopy(d["files"][0])),
    "file under a file": lambda d: d["files"].append({"path": "server.log/inner", "size": 0, "sha256": "0" * 64}),
    "absolute path": lambda d: d["files"][0].update(path="/abs"),
    "dotdot path": lambda d: d["files"][0].update(path="a/../../b"),
    "reserved name": lambda d: d["files"][0].update(path="manifest.sig"),
    "backslash path": lambda d: d["files"][0].update(path="a\\b"),
    "rule pack digest without its prefix": lambda d: d["runner"].update(
        rule_pack={"version": "1", "digest": "ab" * 32}
    ),
    "rule pack without a version": lambda d: d["runner"].update(rule_pack={"digest": PACK_DIGEST}),
    "rule pack with an unknown key": lambda d: d["runner"].update(
        rule_pack={"version": "1", "digest": PACK_DIGEST, "path": "/x"}
    ),
    "card occupancy that is another state": lambda d: d["runner"].update(
        card_occupancy={"state": "other", "reason": None, "checks": 1}
    ),
    "card occupancy foreign with no process named": lambda d: d["runner"].update(
        card_occupancy={"state": "foreign", "reason": None, "checks": 1}
    ),
    "card occupancy unknown with no reason": lambda d: d["runner"].update(
        card_occupancy={"state": "unknown", "reason": None, "checks": 0}
    ),
    "card occupancy clean with a reason": lambda d: d["runner"].update(
        card_occupancy={"state": "clean", "reason": "all is well", "checks": 2}
    ),
    "card occupancy with a negative count of looks": lambda d: d["runner"].update(
        card_occupancy={"state": "clean", "reason": None, "checks": -1}
    ),
    "card occupancy with an unknown key": lambda d: d["runner"].update(
        card_occupancy={"state": "clean", "reason": None, "checks": 1, "holders": []}
    ),
}


@pytest.mark.parametrize("name", sorted(NEGATIVES))
def test_a_manifest_that_breaks_the_contract_is_refused_by_the_reader_and_the_schema(name: str) -> None:
    document = _edit(NEGATIVES[name])
    with pytest.raises(DocumentError):
        ResultManifest.from_document(document)
    cross_field = {
        "failure class on a passing test",
        "failed test with no class",
        "class without reason",
        "file under a file",
        "duplicate file",
        "card occupancy unknown with no reason",
        "card occupancy foreign with no process named",
        "card occupancy clean with a reason",
    }
    if name not in cross_field:
        assert list(VALIDATOR.iter_errors(document)) != [], "the schema accepts what the reader refuses"


def test_a_failed_test_may_carry_a_failure_class_and_a_timeout_counts_as_failed() -> None:
    def change(d: dict[str, Any]) -> None:
        d["outcome"].update(returncode=1, failure_class="candidate", failure_reason="import_error")

    ResultManifest.from_document(_edit(change))

    def timeout(d: dict[str, Any]) -> None:
        d["outcome"].update(timed_out=True, failure_class="unclear", failure_reason="timeout")

    ResultManifest.from_document(_edit(timeout))


def test_duplicate_keys_in_the_manifest_are_refused() -> None:
    raw = encode_manifest(_document()).decode().replace('"timed_out":false', '"timed_out":false,"timed_out":true')
    with pytest.raises(DocumentError, match="duplicate"):
        ResultManifest.from_bytes(raw.encode())
