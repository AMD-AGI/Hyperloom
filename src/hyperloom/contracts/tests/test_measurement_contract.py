# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyperloom.contracts.measurement import (
    AccuracyValue,
    AttachmentRef,
    Attempt,
    FailureClass,
    MeasurementError,
    NOT_REPRODUCED,
    Purpose,
    REPRODUCED,
    RoundValue,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta

DELTA = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
ATTACH = AttachmentRef("case-1", "file", "a" * 64)


def _request(**overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.EXPLORE,
        "branch_id": "b1",
        "parent_commit": "c0",
        "delta": DELTA,
    }
    return TestRequest(**{**fields, **overrides})


def _sandbox(**overrides: object) -> SandboxFacts:
    fields: dict[str, object] = {"sandbox_id": "sb-1", "node_name": "n1", "gpu_model": "mi355x", "gpu_count": 2}
    return SandboxFacts(**{**fields, **overrides})


def _result(**overrides: object) -> TestResult:
    fields: dict[str, object] = {
        "request_id": "r1",
        "validity": Validity.SUCCEEDED,
        "rounds": (RoundValue("measure", 64, {"output_tps": 1000.5}, 100, 0),),
        "sandbox": _sandbox(),
        "manifest_sha256": "b" * 64,
        "output_dir": "tests/r1",
    }
    return TestResult(**{**fields, **overrides})


def _roundtrip(value):
    document = json.loads(json.dumps(value.to_document()))
    again = type(value).from_document(document)
    assert again == value
    assert again.to_document() == value.to_document()
    return again


# --- request: round trip -----------------------------------------------------------------------------------------


def test_request_round_trips_through_json() -> None:
    request = _request(
        intent_fingerprint="i1",
        priority=2,
        avoid_nodes=("n0", "n3"),
        concurrency=(8, 64),
        attachments=(ATTACH,),
        diff_ref="diff/1",
    )
    _roundtrip(request)
    _roundtrip(_request(purpose=Purpose.CONFIRM, retest_of="r0"))
    _roundtrip(_request(purpose=Purpose.BASELINE, delta=RecipeDelta()))


def test_request_has_no_id_until_accepted() -> None:
    request = _request()
    assert request.request_id is None
    accepted = request.accepted("r1")
    assert accepted.request_id == "r1"
    assert _roundtrip(accepted).request_id == "r1"
    assert request.request_id is None


def test_request_document_with_unknown_enum_is_refused_naming_the_field() -> None:
    document = _request().to_document()
    with pytest.raises(MeasurementError, match="purpose must be one of"):
        TestRequest.from_document({**document, "purpose": "wander"})
    with pytest.raises(MeasurementError, match="kind must be one of"):
        TestRequest.from_document({**document, "kind": "bench"})


def test_request_document_with_bad_delta_is_a_measurement_error() -> None:
    with pytest.raises(MeasurementError, match="delta"):
        TestRequest.from_document({**_request().to_document(), "delta": {"config": {"args_mode": "merge"}}})


def test_request_document_must_be_an_object_with_lists_for_lists() -> None:
    with pytest.raises(MeasurementError, match="a test request must be an object"):
        TestRequest.from_document([])
    with pytest.raises(MeasurementError, match="avoid_nodes must be a list"):
        TestRequest.from_document({**_request().to_document(), "avoid_nodes": "n1"})
    with pytest.raises(MeasurementError, match="accuracy must be true or false"):
        TestRequest.from_document({**_request().to_document(), "accuracy": "yes"})


# --- request: every validation rule, each with its counter-example ----------------------------------------------


def test_confirm_needs_retest_of() -> None:
    with pytest.raises(MeasurementError, match="retest_of is required for purpose 'confirm'"):
        _request(purpose=Purpose.CONFIRM)
    assert _request(purpose=Purpose.CONFIRM, retest_of="r0").retest_of == "r0"


@pytest.mark.parametrize("purpose", [p for p in Purpose if p is not Purpose.CONFIRM])
def test_only_confirm_may_name_a_retest(purpose: Purpose) -> None:
    fields: dict[str, object] = {"purpose": purpose, "retest_of": "r0"}
    if purpose in (Purpose.BASELINE, Purpose.HEALTH):
        fields["delta"] = RecipeDelta()
    with pytest.raises(MeasurementError, match="allowed for no other purpose"):
        _request(**fields)


@pytest.mark.parametrize("purpose", [Purpose.BASELINE, Purpose.HEALTH])
def test_baseline_and_health_have_an_empty_delta(purpose: Purpose) -> None:
    with pytest.raises(MeasurementError, match="delta must be empty"):
        _request(purpose=purpose, delta=DELTA)
    assert _request(purpose=purpose, delta=RecipeDelta()).delta == RecipeDelta()


def test_explore_may_carry_an_empty_delta() -> None:
    assert _request(delta=RecipeDelta()).purpose is Purpose.EXPLORE


def test_accuracy_kind_needs_the_accuracy_flag() -> None:
    with pytest.raises(MeasurementError, match="accuracy must be true"):
        _request(kind=TestKind.ACCURACY, purpose=Purpose.ACCURACY)
    assert _request(kind=TestKind.ACCURACY, purpose=Purpose.ACCURACY, accuracy=True).accuracy


def test_repro_kind_needs_an_attachment() -> None:
    with pytest.raises(MeasurementError, match="needs the reproduction case as an attachment"):
        _request(kind=TestKind.REPRO, purpose=Purpose.REPRO)
    assert _request(kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(ATTACH,)).attachments == (ATTACH,)


@pytest.mark.parametrize("bad", [(0,), (-4,), (8, 8), (True,), (1.5,)])
def test_concurrency_is_positive_and_distinct(bad: tuple[object, ...]) -> None:
    with pytest.raises(MeasurementError, match="concurrency"):
        _request(concurrency=bad)


@pytest.mark.parametrize("kind", [TestKind.PROFILE, TestKind.ACCURACY, TestKind.REPRO])
def test_several_concurrency_values_are_only_for_test_requests(kind: TestKind) -> None:
    with pytest.raises(MeasurementError, match="only for 'test' requests"):
        _request(kind=kind, purpose=Purpose.EXPLORE, concurrency=(8, 16), accuracy=True, attachments=(ATTACH,))
    assert _request(concurrency=(8, 16)).concurrency == (8, 16)
    assert _request(kind=kind, concurrency=(8,), accuracy=True, attachments=(ATTACH,)).concurrency == (8,)


@pytest.mark.parametrize("bad", [-1, 5, True, 2.0, "1"])
def test_priority_is_an_integer_from_0_to_4(bad: object) -> None:
    with pytest.raises(MeasurementError, match="priority"):
        _request(priority=bad)
    assert [_request(priority=p).priority for p in range(5)] == [0, 1, 2, 3, 4]
    assert _request().priority is None


def test_text_fields_are_not_blank() -> None:
    for name in ("branch_id", "parent_commit"):
        with pytest.raises(MeasurementError, match=name):
            _request(**{name: " "})
    with pytest.raises(MeasurementError, match="an avoided node"):
        _request(avoid_nodes=("",))


def test_attachments_are_content_addressed_and_distinct() -> None:
    with pytest.raises(MeasurementError, match="lowercase hex sha256"):
        AttachmentRef("x", "file", "ABC")
    with pytest.raises(MeasurementError, match="'file' or 'dir'"):
        AttachmentRef("x", "link", "a" * 64)
    with pytest.raises(MeasurementError, match="distinct ids"):
        _request(attachments=(ATTACH, ATTACH))
    _roundtrip(AttachmentRef("d", "dir", "c" * 64))


def test_request_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _request().priority = 1  # type: ignore[misc]


def test_test_classes_are_not_collected_by_pytest() -> None:
    assert TestRequest.__test__ is False
    assert TestResult.__test__ is False
    assert TestKind.__test__ is False


# --- compatibility with the notebook's names ---------------------------------------------------------------------


def test_request_answers_to_the_notebooks_names() -> None:
    request = _request(accuracy=True)
    assert request.branch == "b1"
    assert request.accuracy_rounds is True
    assert request.action_fingerprint == DELTA.fingerprint()


def test_enums_keep_the_notebooks_member_names() -> None:
    assert Validity.SUCCESS is Validity.SUCCEEDED
    assert Validity.CANDIDATE_FAILURE is Validity.CANDIDATE_FAILED
    assert TestKind.SWEEP.value == "sweep"
    assert Purpose.EXPLORE == "explore"


def test_result_answers_to_the_notebooks_names() -> None:
    result = _result(validity=Validity.CANDIDATE_FAILED, failure_class=FailureClass.CANDIDATE)
    assert result.failure is FailureClass.CANDIDATE
    assert result.sandbox.node == "n1"
    facts = _sandbox(power_cap_w=750.0)
    assert facts.power_cap_watts == 750.0
    assert Attempt(sandbox=facts, failure_class=FailureClass.INFRA).failure is FailureClass.INFRA
    assert Attempt(sandbox=facts).node == "n1"
    assert Attempt().node is None


# --- result ------------------------------------------------------------------------------------------------------


def test_result_round_trips_through_json() -> None:
    attempt_1 = Attempt(_sandbox(sandbox_id="sb-0", node_name="n0"), FailureClass.INFRA, "preflight_mismatch", 1.0, 2.0)
    result = _result(
        accuracy=(AccuracyValue("gsm8k", "exact_match", 0.93),),
        attempts=(attempt_1, Attempt(_sandbox(), started_at=3.0, finished_at=9.0)),
        signing_key_fingerprint="fp",
        spec_fingerprint="sp",
        surface_sha256="c" * 64,
        queued_at=0.0,
        started_at=1.0,
        finished_at=9.0,
        gpu_seconds=12.5,
        profile_location="profiles/r1",
    )
    again = _roundtrip(result)
    assert again.rounds[0].metrics == {"output_tps": 1000.5}
    _roundtrip(_result(validity=Validity.MEASUREMENT_FAILED, failure_class=FailureClass.INFRA, sandbox=None))


def test_result_document_with_unknown_values_is_refused() -> None:
    document = _result().to_document()
    with pytest.raises(MeasurementError, match="validity must be one of"):
        TestResult.from_document({**document, "validity": "ok"})
    with pytest.raises(MeasurementError, match="failure_class must be one of"):
        TestResult.from_document({**document, "failure_class": "flaky"})
    with pytest.raises(MeasurementError, match="a round phase"):
        TestResult.from_document({**document, "rounds": [{"phase": "cooldown"}]})


@pytest.mark.parametrize("verdict", ["", REPRODUCED, NOT_REPRODUCED])
def test_a_repro_verdict_round_trips(verdict: str) -> None:
    assert _roundtrip(_result(repro_verdict=verdict)).repro_verdict == verdict


def test_a_document_written_before_the_repro_verdict_loads_with_none() -> None:
    document = _result().to_document()
    del document["repro_verdict"]
    assert TestResult.from_document(document).repro_verdict == ""


def test_a_repro_verdict_is_one_of_the_two_verdicts() -> None:
    with pytest.raises(MeasurementError, match="repro_verdict must be"):
        _result(repro_verdict="fixed")
    with pytest.raises(MeasurementError, match="repro_verdict must be"):
        TestResult.from_document({**_result().to_document(), "repro_verdict": "maybe"})


@pytest.mark.parametrize(
    ("validity", "failure_class"),
    [
        (Validity.CANDIDATE_FAILED, FailureClass.CANDIDATE),
        (Validity.MEASUREMENT_FAILED, FailureClass.INFRA),
    ],
)
def test_only_a_succeeded_result_has_a_repro_verdict(validity: Validity, failure_class: FailureClass) -> None:
    with pytest.raises(MeasurementError, match="only a succeeded test has a repro_verdict"):
        _result(validity=validity, failure_class=failure_class, repro_verdict=REPRODUCED)


def test_a_succeeded_result_has_no_failure_class() -> None:
    with pytest.raises(MeasurementError, match="succeeded test has no failure_class"):
        _result(failure_class=FailureClass.INFRA)


def test_candidate_failed_fails_as_candidate_or_unclear_never_infra() -> None:
    for cls in (FailureClass.CANDIDATE, FailureClass.UNCLEAR, None):
        assert _result(validity=Validity.CANDIDATE_FAILED, failure_class=cls).failure_class is cls
    with pytest.raises(MeasurementError, match="candidate or unclear"):
        _result(validity=Validity.CANDIDATE_FAILED, failure_class=FailureClass.INFRA)


def test_measurement_failed_is_not_a_candidate_failure() -> None:
    with pytest.raises(MeasurementError, match="not a failure of the candidate"):
        _result(validity=Validity.MEASUREMENT_FAILED, failure_class=FailureClass.CANDIDATE)


def test_a_failure_reason_needs_a_failure_class() -> None:
    with pytest.raises(MeasurementError, match="needs a failure_class"):
        _result(failure_reason="import_error")
    with pytest.raises(MeasurementError, match="needs a failure_class"):
        Attempt(failure_reason="oom")
    assert _result(failure_reason="").failure_reason == ""


def test_result_checks_digests_and_paths() -> None:
    with pytest.raises(MeasurementError, match="manifest_sha256"):
        _result(manifest_sha256="short")
    with pytest.raises(MeasurementError, match="surface_sha256"):
        _result(surface_sha256="Z" * 64)
    for path in ("/abs/dir", "../escape", "a/../../b"):
        with pytest.raises(MeasurementError, match="relative path inside the session directory"):
            _result(output_dir=path)
    with pytest.raises(MeasurementError, match="gpu_seconds cannot be negative"):
        _result(gpu_seconds=-1.0)


def test_attempt_cannot_finish_before_it_starts() -> None:
    with pytest.raises(MeasurementError, match="cannot finish"):
        Attempt(started_at=5.0, finished_at=4.0)


def test_round_and_accuracy_values_are_finite_numbers() -> None:
    with pytest.raises(MeasurementError, match="finite number"):
        RoundValue("measure", metrics={"ttft_ms": float("nan")})
    with pytest.raises(MeasurementError, match="finite number"):
        AccuracyValue("gsm8k", "em", float("inf"))
    with pytest.raises(MeasurementError, match="concurrency"):
        RoundValue("measure", concurrency=0)
    with pytest.raises(MeasurementError, match="failed_requests"):
        RoundValue("measure", failed_requests=-1)


def test_an_accuracy_value_round_trips_its_version_and_an_old_document_loads_without_one() -> None:
    value = AccuracyValue("gsm8k", "em", 0.5, "lm_eval=0.4.13;gsm8k=3")
    assert AccuracyValue.from_document(value.to_document()) == value
    old = {"task": "gsm8k", "metric": "em", "score": 0.5}
    assert AccuracyValue.from_document(old) == AccuracyValue("gsm8k", "em", 0.5, "")
    assert AccuracyValue.from_document(old).to_document() == old, "an empty version is not written"
    assert AccuracyValue.from_document({**old, "version": ""}).version == ""


@pytest.mark.parametrize("bad", [" lm_eval=1", "lm_eval=1\n", 3, None, ["x"]])
def test_an_accuracy_version_must_be_a_string_without_surrounding_whitespace(bad: object) -> None:
    with pytest.raises(MeasurementError, match="accuracy version must be a string without surrounding whitespace"):
        AccuracyValue.from_document({"task": "gsm8k", "metric": "em", "score": 0.5, "version": bad})


def test_a_result_document_without_the_accuracy_version_still_loads() -> None:
    document = _result().to_document()
    document["accuracy"] = [{"task": "gsm8k", "metric": "em", "score": 0.5}]
    assert TestResult.from_document(document).accuracy == (AccuracyValue("gsm8k", "em", 0.5),)


def test_sandbox_facts_validate_and_flag_same_node() -> None:
    with pytest.raises(MeasurementError, match="gpu_count"):
        _sandbox(gpu_count=-1)
    with pytest.raises(MeasurementError, match="a sandbox id"):
        _sandbox(sandbox_id="")
    assert _roundtrip(_sandbox(same_node=True, gpu_ids=(0, 1), backend="hosted")).same_node


def test_result_is_replaceable_but_revalidated() -> None:
    result = _result()
    with pytest.raises(MeasurementError, match="succeeded test has no failure_class"):
        replace(result, failure_class=FailureClass.INFRA)


def test_nonfatal_warnings_round_trip_in_order_without_dedup() -> None:
    warnings = ("warmup_round_seed:1", "run_grid_warmup_discarded_first", "warmup_round_seed:1")
    assert _roundtrip(_result(nonfatal_warnings=warnings)).nonfatal_warnings == warnings


def test_a_document_written_before_the_warnings_loads_with_none() -> None:
    document = _result().to_document()
    del document["nonfatal_warnings"]
    assert TestResult.from_document(document).nonfatal_warnings == ()


def test_a_document_written_while_the_decode_pass_existed_loads_and_drops_its_two_keys() -> None:
    plain = _result(nonfatal_warnings=("decode_pass_failed:http_500",)).to_document()
    old = {**plain, "decode_outputs": "decode/outputs.jsonl", "decode_identity": "sha256:ab|temperature=0|top_k=1"}

    loaded = TestResult.from_document(old)

    assert loaded == TestResult.from_document(plain)
    assert "decode_outputs" not in loaded.to_document() and "decode_identity" not in loaded.to_document()
    # the old validation ("both set or both empty") judged a half-set pair; nothing reads the pair now
    assert TestResult.from_document({**plain, "decode_outputs": "decode/outputs.jsonl"}) == loaded
    assert TestResult.from_document({**plain, "decode_outputs": "", "decode_identity": ""}) == loaded


@pytest.mark.parametrize("bad", [("ok", 3), "a string", ["list"], (None,)])
def test_warnings_must_be_a_tuple_of_strings(bad: object) -> None:
    with pytest.raises(MeasurementError, match="nonfatal_warnings must be a tuple of strings"):
        _result(nonfatal_warnings=bad)


def test_a_document_with_a_non_string_warning_is_refused() -> None:
    with pytest.raises(MeasurementError, match="nonfatal_warnings must be a tuple of strings"):
        TestResult.from_document({**_result().to_document(), "nonfatal_warnings": ["ok", 3]})
    with pytest.raises(MeasurementError, match="nonfatal_warnings must be a list"):
        TestResult.from_document({**_result().to_document(), "nonfatal_warnings": "oops"})
