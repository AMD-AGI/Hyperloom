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
    BuildCharge,
    BuildPayment,
    FailureClass,
    MeasurementError,
    NOT_REPRODUCED,
    PhaseTiming,
    Purpose,
    REPRODUCED,
    RequestClasses,
    RoundValue,
    SandboxFacts,
    TABLE_EXERCISED,
    TABLE_NOT_EXERCISED,
    TABLE_NOT_OBSERVABLE,
    TableHits,
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


def test_sandbox_facts_state_whether_the_cards_were_free_of_other_processes_and_round_trip() -> None:
    clean = _roundtrip(_sandbox(card_occupancy="clean"))
    unknown = _roundtrip(_sandbox(card_occupancy="unknown", card_occupancy_reason="no AMD GPU driver is loaded"))
    foreign = _roundtrip(
        _sandbox(card_occupancy="foreign", card_occupancy_reason="1 held the test's cards: pid 9 on card 0")
    )
    unobservable = _roundtrip(
        _sandbox(card_occupancy="unobservable", card_occupancy_reason="every GPU node is refused")
    )
    older = SandboxFacts.from_document({"sandbox_id": "sb-1"})

    assert (unobservable.card_occupancy, unobservable.card_occupancy_reason) == (
        "unobservable",
        "every GPU node is refused",
    )
    assert (clean.card_occupancy, clean.card_occupancy_reason) == ("clean", "")
    assert (unknown.card_occupancy, unknown.card_occupancy_reason) == ("unknown", "no AMD GPU driver is loaded")
    assert (foreign.card_occupancy, foreign.card_occupancy_reason) == (
        "foreign",
        "1 held the test's cards: pid 9 on card 0",
    )
    assert (older.card_occupancy, older.card_occupancy_reason) == ("", ""), "an attempt that did not say, says nothing"
    assert older.card_occupancy_method == ""
    assert _roundtrip(
        _sandbox(card_occupancy="clean", card_occupancy_method="census+fdinfo")
    ).card_occupancy_method == ("census+fdinfo")


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"card_occupancy": "other"}, "card_occupancy must be"),
        ({"card_occupancy": "unknown"}, "needs its reason"),
        ({"card_occupancy": "foreign"}, "needs its reason"),
        ({"card_occupancy": "unobservable"}, "needs its reason"),
        ({"card_occupancy": "clean", "card_occupancy_reason": "why?"}, "needs its reason"),
        ({"card_occupancy_reason": "a reason with no state"}, "needs its reason"),
        ({"card_occupancy": "unknown", "card_occupancy_reason": 7}, "must be a string"),
        ({"card_occupancy": "clean", "card_occupancy_method": 7}, "card_occupancy_method must be a string"),
    ],
)
def test_sandbox_facts_refuse_a_card_occupancy_that_is_not_one_of_the_states_or_lacks_its_reason(
    fields: dict[str, object], why: str
) -> None:
    with pytest.raises(MeasurementError, match=why):
        _sandbox(**fields)


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


# --- compatibility with the blackboard's names ---------------------------------------------------------------------


def test_request_answers_to_the_blackboards_names() -> None:
    request = _request(accuracy=True)
    assert request.branch == "b1"
    assert request.accuracy_rounds is True
    assert request.action_fingerprint == DELTA.fingerprint()


def test_enums_keep_the_blackboards_member_names() -> None:
    assert Validity.SUCCESS is Validity.SUCCEEDED
    assert Validity.CANDIDATE_FAILURE is Validity.CANDIDATE_FAILED
    assert TestKind.SWEEP.value == "sweep"
    assert Purpose.EXPLORE == "explore"


def test_result_answers_to_the_blackboards_names() -> None:
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


def test_a_cold_start_round_trips_and_a_document_written_before_it_loads_with_none() -> None:
    assert _roundtrip(_result(cold_start_seconds=82.2)).cold_start_seconds == 82.2
    document = _result().to_document()
    del document["cold_start_seconds"]
    assert TestResult.from_document(document).cold_start_seconds is None


def test_a_cold_start_is_a_finite_number_that_is_not_negative() -> None:
    with pytest.raises(MeasurementError, match="cold_start_seconds cannot be negative"):
        _result(cold_start_seconds=-0.5)
    with pytest.raises(MeasurementError, match="cold_start_seconds must be a finite number"):
        _result(cold_start_seconds=float("inf"))


PHASES = (
    PhaseTiming("sandbox_start", 100.0, 12.5),
    PhaseTiming("server_boot", 120.0, 82.221, gpu_seconds=82.221, point=0),
    PhaseTiming("server_boot.weights_load", 120.0, 40.0, gpu_seconds=40.0, point=0),
)


def test_phases_round_trip_and_a_document_written_before_them_loads_without() -> None:
    result = _roundtrip(_result(phases=PHASES, unaccounted_seconds=-0.25))
    assert result.phases == PHASES and result.unaccounted_seconds == -0.25
    document = _result().to_document()
    del document["phases"], document["unaccounted_seconds"]
    loaded = TestResult.from_document(document)
    assert (loaded.phases, loaded.unaccounted_seconds) == ((), None), "no phases is not a test that took no time"
    assert _result().to_document()["phases"] == [] and _result().to_document()["unaccounted_seconds"] is None


def test_a_result_has_an_unaccounted_time_exactly_when_it_has_phases() -> None:
    with pytest.raises(MeasurementError, match="unaccounted_seconds is set exactly when the result has phases"):
        _result(phases=PHASES)
    with pytest.raises(MeasurementError, match="unaccounted_seconds is set exactly when the result has phases"):
        _result(unaccounted_seconds=0.0)
    with pytest.raises(MeasurementError, match="unaccounted_seconds must be a finite number"):
        _result(phases=PHASES, unaccounted_seconds=float("nan"))
    with pytest.raises(MeasurementError, match="phases must be a tuple of PhaseTiming"):
        _result(phases=({"name": "x"},), unaccounted_seconds=0.0)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"name": "Server Boot"}, "a phase name is lowercase words joined by '_'"),
        ({"name": "boot\nforged: line"}, "a phase name is lowercase words joined by '_'"),
        ({"name": "a.b.c"}, "a phase name is lowercase words joined by '_'"),
        ({"seconds": -1.0}, "a phase's seconds cannot be negative"),
        ({"gpu_seconds": -1.0}, "a phase's gpu_seconds cannot be negative"),
        ({"started_at": -1.0}, "a phase's start cannot be negative"),
        ({"seconds": float("inf")}, "a phase's seconds must be a finite number"),
        ({"seconds": 1e9}, "a phase's seconds cannot be more than 3.16224e\\+07"),
        ({"started_at": 1e12}, "a phase's start cannot be more than 1e\\+11"),
        ({"gpu_seconds": 1e15}, "a phase's gpu_seconds cannot be more than"),
        ({"point": -1}, "a phase's point must be an integer of at least 0"),
    ],
)
def test_a_phase_that_breaks_the_contract_is_refused(fields: dict[str, object], message: str) -> None:
    document = {"name": "warmup", "started_at": 1.0, "seconds": 2.0, "gpu_seconds": 2.0, "point": 0, **fields}
    with pytest.raises(MeasurementError, match=message):
        PhaseTiming.from_document(document)


def test_a_phase_document_without_card_time_or_point_loads_with_none_of_either() -> None:
    phase = PhaseTiming.from_document({"name": "warmup", "started_at": 1.0, "seconds": 2.0})
    assert (phase.gpu_seconds, phase.point) == (0.0, None)


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


# --- the hits of a kernel table ------------------------------------------------------------------------------------

TABLE_SHA = "c" * 64
EXERCISED = TableHits(
    "AITER_CONFIG_GEMM_BF16", "gemm_bf16", TABLE_SHA, TABLE_EXERCISED, rows=4, rows_hit=2, hit_lines=11
)
NOT_EXERCISED = TableHits(
    "AITER_CONFIG_GEMM_BF16",
    "gemm_bf16",
    TABLE_SHA,
    TABLE_NOT_EXERCISED,
    rows=4,
    rows_hit=0,
    hit_lines=0,
    reason="none",
)
NOT_OBSERVABLE = TableHits(
    "AITER_CONFIG_GEMM_BF16", "gemm_bf16", TABLE_SHA, TABLE_NOT_OBSERVABLE, rows=4, reason="no hit logging"
)


@pytest.mark.parametrize("hits", [EXERCISED, NOT_EXERCISED, NOT_OBSERVABLE], ids=lambda hits: hits.status)
def test_the_hits_of_a_table_round_trip_and_an_unknown_count_stays_unknown(hits: TableHits) -> None:
    again = _roundtrip(_result(table_hits=(hits,), runtime_findings=("a finding",)))
    assert again.table_hits == (hits,) and again.runtime_findings == ("a finding",)
    assert (again.table_hits[0].rows_hit is None) == (hits.status == TABLE_NOT_OBSERVABLE), "none is not zero"


def test_a_document_written_before_the_table_hits_loads_with_none() -> None:
    document = _result(table_hits=(EXERCISED,), runtime_findings=("x",)).to_document()
    del document["table_hits"], document["runtime_findings"]
    loaded = TestResult.from_document(document)
    assert loaded.table_hits == () and loaded.runtime_findings == ()


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"status": "flat"}, "status must be one of"),
        ({"status": TABLE_EXERCISED, "rows_hit": 0, "hit_lines": 0}, "an exercised table has at least one row hit"),
        ({"status": TABLE_EXERCISED, "rows_hit": 5, "hit_lines": 5}, "5 rows hit of 4 rows"),
        ({"status": TABLE_EXERCISED, "rows_hit": None, "hit_lines": None}, "counts its rows hit and its hit lines"),
        ({"status": TABLE_NOT_EXERCISED, "rows_hit": 1, "hit_lines": 1}, "a not_exercised table has no row hit"),
        ({"status": TABLE_NOT_EXERCISED, "rows_hit": None, "hit_lines": None}, "counts its rows hit"),
        ({"status": TABLE_NOT_OBSERVABLE, "rows_hit": 0, "hit_lines": 0, "reason": "r"}, "has no counts and says why"),
        ({"status": TABLE_NOT_OBSERVABLE, "reason": ""}, "has no counts and says why"),
        (
            {"status": TABLE_EXERCISED, "rows": -1, "rows_hit": 1, "hit_lines": 1},
            "rows must be an integer of at least 0",
        ),
        ({"sha256": "not-a-sha"}, "sha256 must be a lowercase hex sha256"),
        ({"env": " "}, "env must be a non-empty string"),
    ],
)
def test_a_table_that_contradicts_its_own_status_is_refused_naming_the_contradiction(
    fields: dict[str, object], match: str
) -> None:
    document = {**EXERCISED.to_document(), **fields}
    with pytest.raises(MeasurementError, match=match):
        TableHits.from_document(document)


@pytest.mark.parametrize("bad", [[EXERCISED], "text", (EXERCISED.to_document(),), (None,)])
def test_table_hits_must_be_a_tuple_of_table_hits(bad: object) -> None:
    with pytest.raises(MeasurementError, match="table_hits must be a tuple of TableHits"):
        _result(table_hits=bad)


def test_runtime_findings_must_be_a_tuple_of_strings() -> None:
    with pytest.raises(MeasurementError, match="runtime_findings must be a tuple of strings"):
        _result(runtime_findings=("ok", 3))
    with pytest.raises(MeasurementError, match="table_hits must be a list"):
        TestResult.from_document({**_result().to_document(), "table_hits": "oops"})


# --- the failed requests of an AgentX round, by class ----------------------------------------------------------------

COUNTED = RequestClasses(
    0,
    "p0/aiperf_artifacts/profile_export.jsonl",
    128,
    requests_total=100,
    failed_total=7,
    failed_transport=3,
    failed_no_content=4,
    failed_empty=1,
    failed_short=3,
)
UNSPLIT = RequestClasses(
    1,
    "p1/aiperf_artifacts/profile_export.jsonl",
    256,
    requests_total=100,
    failed_total=7,
    failed_transport=3,
    failed_no_content=4,
    reason="no output token count",
)
NOT_COUNTED = RequestClasses(2, reason="the round left no per-request export")


@pytest.mark.parametrize("classes", [COUNTED, UNSPLIT, NOT_COUNTED], ids=["counted", "unsplit", "not-counted"])
def test_the_classes_of_a_rounds_failed_requests_round_trip_and_an_unknown_count_stays_unknown(
    classes: RequestClasses,
) -> None:
    again = _roundtrip(_result(request_classes=(classes,)))
    assert again.request_classes == (classes,)
    unknown = [
        name
        for name, value in classes.to_document().items()
        if name.startswith(("requests_", "failed_")) and value is None
    ]
    assert [getattr(again.request_classes[0], name) for name in unknown] == [None] * len(unknown), "none is not zero"


def test_a_document_written_before_the_request_classes_loads_with_none_and_that_is_not_zero_failures() -> None:
    document = _result(request_classes=(COUNTED,)).to_document()
    del document["request_classes"]
    assert TestResult.from_document(document).request_classes == ()


def test_the_result_document_names_every_class_count_and_where_the_export_is() -> None:
    (shown,) = _result(request_classes=(COUNTED,)).to_document()["request_classes"]
    assert shown == {
        "point": 0,
        "export": "p0/aiperf_artifacts/profile_export.jsonl",
        "concurrency": 128,
        "requests_total": 100,
        "failed_total": 7,
        "failed_transport": 3,
        "failed_no_content": 4,
        "failed_empty": 1,
        "failed_short": 3,
        "reason": "",
    }


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"failed_total": 8}, "8 failed requests of 100 are 3 transport and 4 no-content"),
        ({"failed_transport": 2}, "7 failed requests of 100 are 2 transport and 4 no-content"),
        ({"requests_total": 5}, "7 failed requests of 5 are 3 transport and 4 no-content"),
        ({"failed_empty": 2}, "the empty and short failures add up to the no-content failures"),
        ({"failed_short": None, "reason": "why"}, "the empty and the short failures come together"),
        ({"requests_total": None, "reason": "why"}, "the requests, the failed ones and their .* parts come together"),
        ({"failed_transport": -1}, "failed_transport must be an integer of at least 0"),
        ({"failed_total": True}, "failed_total must be an integer"),
        ({"point": -1}, "point must be an integer of at least 0"),
        ({"concurrency": 0}, "concurrency must be an integer of at least 1"),
        ({"export": "/abs/profile_export.jsonl"}, "export must be a relative path"),
        ({"export": "../profile_export.jsonl"}, "export must be a relative path"),
        ({"reason": 3}, "reason must be a string"),
    ],
)
def test_classes_that_contradict_themselves_are_refused_naming_the_contradiction(
    fields: dict[str, object], match: str
) -> None:
    with pytest.raises(MeasurementError, match=match):
        RequestClasses.from_document({**COUNTED.to_document(), **fields})


@pytest.mark.parametrize(
    "unknown",
    [
        ("failed_empty", "failed_short"),
        ("requests_total", "failed_total", "failed_transport", "failed_no_content", "failed_empty", "failed_short"),
    ],
    ids=["the-split", "everything"],
)
def test_a_count_that_could_not_be_computed_must_say_why(unknown: tuple[str, ...]) -> None:
    document = {**COUNTED.to_document(), **dict.fromkeys(unknown), "reason": ""}
    with pytest.raises(MeasurementError, match="a count that could not be computed says why"):
        RequestClasses.from_document(document)
    assert RequestClasses.from_document({**document, "reason": "why"}).reason == "why"


@pytest.mark.parametrize("bad", [[COUNTED], "text", (COUNTED.to_document(),), (None,)])
def test_request_classes_must_be_a_tuple_of_request_classes(bad: object) -> None:
    with pytest.raises(MeasurementError, match="request_classes must be a tuple of RequestClasses"):
        _result(request_classes=bad)
    with pytest.raises(MeasurementError, match="request_classes must be a list"):
        TestResult.from_document({**_result().to_document(), "request_classes": "oops"})


# --- result: native build charges ---------------------------------------------------------------------------------

PAID = BuildCharge("k" * 64, "sgl_kernel", 812.5, BuildPayment.PAID)
REUSED = BuildCharge("j" * 64, "vllm", 0.0, BuildPayment.REUSED)


def test_build_charges_round_trip_and_a_result_without_builds_writes_none() -> None:
    again = _roundtrip(_result(builds=(PAID, REUSED)))
    assert again.builds == (PAID, REUSED)
    assert again.to_document()["builds"][0] == {
        "key": "k" * 64,
        "component": "sgl_kernel",
        "build_seconds": 812.5,
        "payment": "paid",
    }
    assert "builds" not in _result().to_document()


def test_a_document_written_before_builds_loads_with_none() -> None:
    document = _result().to_document()
    document.pop("builds", None)
    assert TestResult.from_document(document).builds == ()


def test_a_reused_build_is_charged_no_seconds_and_no_build_is_charged_negative_seconds() -> None:
    with pytest.raises(MeasurementError, match="a reused build is charged no build_seconds, got 3.0"):
        BuildCharge("k", "sgl_kernel", 3.0, BuildPayment.REUSED)
    with pytest.raises(MeasurementError, match="build_seconds cannot be negative"):
        BuildCharge("k", "sgl_kernel", -1.0, BuildPayment.PAID)
    with pytest.raises(MeasurementError, match="payment must be one of"):
        BuildCharge.from_document({"key": "k", "component": "c", "build_seconds": 0.0, "payment": "free"})
    with pytest.raises(MeasurementError, match="builds must be a tuple of BuildCharge"):
        _result(builds=[PAID])
