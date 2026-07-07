# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for pure helpers in ``kernel_attempt_summary``.

Targets the small, file-/dict-only functions (artifact path check, failure
classification, kernel-agent results harvesting, per-kernel classification)
that the higher-level ``build_kernel_optimization_summary`` tests do not
exercise directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.kernel import attempt_summary as kas


def test_is_real_artifact_path_variants():
    assert kas._is_real_artifact_path("") is False
    assert kas._is_real_artifact_path("   ") is False
    assert kas._is_real_artifact_path("runs/k/out_stdout.log") is False
    assert kas._is_real_artifact_path("runs/k/build.txt") is False
    assert kas._is_real_artifact_path("runs/k/foo_stderr.json") is False
    assert kas._is_real_artifact_path("runs/k/optimized.py") is True


def test_classify_attempt_failure_priority():
    assert kas._classify_attempt_failure({"status": "succeeded"}) == ("", "")

    cls, msg = kas._classify_attempt_failure(
        {"status": "failed", "error_message": "Timed out after 42s"},
    )
    assert cls == kas.ERROR_CLASS_TIMEOUT and "42s" in msg

    cls, _ = kas._classify_attempt_failure(
        {"status": "failed", "stdout_tail": "preprocess success=False errors=3"},
    )
    assert cls == kas.ERROR_CLASS_PREPROCESS_FAILED

    cls, _ = kas._classify_attempt_failure(
        {"status": "failed", "stdout_tail": "build failed: undefined reference"},
    )
    assert cls == kas.ERROR_CLASS_COMPILE_FAILED

    cls, _ = kas._classify_attempt_failure(
        {"status": "failed", "stdout_tail": "correctness mismatch detected"},
    )
    assert cls == kas.ERROR_CLASS_CORRECTNESS_FAILED

    cls, msg = kas._classify_attempt_failure(
        {"status": "failed", "returncode": 2},
    )
    assert cls == kas.ERROR_CLASS_AGENT_ERROR and "2" in msg

    assert kas._classify_attempt_failure({"status": "failed"}) == (
        kas.ERROR_CLASS_UNKNOWN,
        "",
    )


def test_backend_results_dir_lookup(tmp_path: Path):
    # No runs root at all.
    assert kas._backend_results_dir(tmp_path, "sid") is None

    # Exact key match.
    sd = tmp_path / "sess"
    results = sd / "kernel-agent" / "runs" / "sess" / "results"
    results.mkdir(parents=True)
    assert kas._backend_results_dir(sd, "") == results

    # Single-subdir recovery when neither key matches.
    sd2 = tmp_path / "sess2"
    other = sd2 / "kernel-agent" / "runs" / "migrated-key" / "results"
    other.mkdir(parents=True)
    assert kas._backend_results_dir(sd2, "no-match") == other


def test_load_kernel_result_cases(tmp_path: Path):
    assert kas._load_kernel_result(None, "k")[1] == "kernel_agent_results_dir_missing"
    assert kas._load_kernel_result(tmp_path, "k")[1] == "kernel_agent_result_file_missing"

    bad = tmp_path / "k.json"
    bad.write_text("{not json", encoding="utf-8")
    assert kas._load_kernel_result(tmp_path, "k")[1] == "parse_error"

    notdict = tmp_path / "list.json"
    notdict.write_text("[1, 2]", encoding="utf-8")
    assert kas._load_kernel_result(tmp_path, "list")[1] == "parse_error"

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"attempts": []}), encoding="utf-8")
    data, reason = kas._load_kernel_result(tmp_path, "ok")
    assert reason == "" and data == {"attempts": []}


def test_load_backend_ladder(tmp_path: Path):
    no_attempts = tmp_path / "na.json"
    no_attempts.write_text(json.dumps({"attempts": []}), encoding="utf-8")
    assert kas._load_backend_ladder(tmp_path, "na") == ([], "no_attempts_recorded")

    payload = {
        "attempts": [
            "not-a-dict",
            {
                "backend": "geak",
                "status": "failed",
                "attempt_id": "a1",
                "optimized_path": "runs/k/out_stdout.log",
                "elapsed_s": 1.5,
                "error_message": "Timed out after 10s",
            },
        ],
    }
    f = tmp_path / "k.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    ladder, reason = kas._load_backend_ladder(tmp_path, "k")
    assert reason == "" and len(ladder) == 1
    row = ladder[0]
    assert row["backend"] == "geak"
    assert row["produced_artifact"] is False
    assert row["elapsed_sec"] == 1.5
    assert row["error_class"] == kas.ERROR_CLASS_TIMEOUT


def test_relative_to_session(tmp_path: Path):
    inside = tmp_path / "a" / "b"
    assert kas._relative_to_session(inside, tmp_path) == "a/b"
    outside = Path("/somewhere/else")
    assert kas._relative_to_session(outside, tmp_path) == str(outside)


def test_classify_attempted():
    entry = {"last_decision": "keep"}
    assert (
        kas._classify_attempted(
            entry,
            integrated_ids={"x"},
            rejected_ids=set(),
            kernel_id="x",
        )
        == kas.CATEGORY_INTEGRATED
    )
    assert (
        kas._classify_attempted(
            entry,
            integrated_ids=set(),
            rejected_ids={"x"},
            kernel_id="x",
        )
        == kas.CATEGORY_ATTEMPTED_REJECTED
    )
    assert (
        kas._classify_attempted(
            entry,
            integrated_ids=set(),
            rejected_ids=set(),
            kernel_id="x",
        )
        == kas.CATEGORY_KEEP_PENDING
    )
    assert (
        kas._classify_attempted(
            {"last_decision": "partial"},
            integrated_ids=set(),
            rejected_ids=set(),
            kernel_id="x",
        )
        == kas.CATEGORY_IN_FLIGHT
    )


def test_unattempted_reason_order():
    assert kas._unattempted_reason({})[0] == kas.UNATTEMPTED_NO_SOURCE
    assert (
        kas._unattempted_reason(
            {"source_file": "f.py", "reusable_native_kernel": False},
        )[0]
        == kas.UNATTEMPTED_NOT_REUSABLE
    )
    assert (
        kas._unattempted_reason(
            {"source_file": "f.py", "reusable_native_kernel": True, "recommended_backends": []},
        )[0]
        == kas.UNATTEMPTED_NO_BACKEND
    )
    assert (
        kas._unattempted_reason(
            {
                "source_file": "f.py",
                "reusable_native_kernel": True,
                "recommended_backends": ["geak"],
            },
        )[0]
        == kas.UNATTEMPTED_BELOW_CUTOFF
    )


def test_load_backend_ladder_skipped_flag(tmp_path: Path):
    # A forge self-skip attempt carries skipped=True; it must ride into the row.
    payload = {
        "attempts": [
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "forge-1",
                "returncode": 2,
                "skipped": True,
            },
            {"backend": "geak", "status": "failed", "attempt_id": "geak-1"},
        ],
    }
    f = tmp_path / "k.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    ladder, reason = kas._load_backend_ladder(tmp_path, "k")
    assert reason == "" and len(ladder) == 2
    assert ladder[0]["skipped"] is True
    # Non-skip rows must not gain a skipped key.
    assert "skipped" not in ladder[1]


def test_kernel_outcome_class_mapping():
    # success: kept / integrated.
    assert kas._kernel_outcome_class(kas.CATEGORY_INTEGRATED, []) == kas.OUTCOME_SUCCESS
    assert kas._kernel_outcome_class(kas.CATEGORY_KEEP_PENDING, []) == kas.OUTCOME_SUCCESS

    # skip: never dispatched.
    assert kas._kernel_outcome_class(kas.CATEGORY_UNATTEMPTED, []) == kas.OUTCOME_SKIP

    # skip: every recorded attempt self-skipped.
    all_skipped = [{"skipped": True, "error_class": kas.ERROR_CLASS_AGENT_ERROR}]
    assert kas._kernel_outcome_class(kas.CATEGORY_ATTEMPTED_REJECTED, all_skipped) == kas.OUTCOME_SKIP

    # mixed ladder (one skipped, one really failed) is NOT a skip -> fail.
    mixed = [
        {"skipped": True},
        {"error_class": kas.ERROR_CLASS_COMPILE_FAILED},
    ]
    assert kas._kernel_outcome_class(kas.CATEGORY_ATTEMPTED_REJECTED, mixed) == kas.OUTCOME_FAIL

    # timeout: any attempt timed out (and not all-skipped).
    to = [{"error_class": kas.ERROR_CLASS_TIMEOUT}]
    assert kas._kernel_outcome_class(kas.CATEGORY_ATTEMPTED_REJECTED, to) == kas.OUTCOME_TIMEOUT

    # fail: attempted, real error, no skip/timeout.
    fail = [{"error_class": kas.ERROR_CLASS_AGENT_ERROR}]
    assert kas._kernel_outcome_class(kas.CATEGORY_ATTEMPTED_REJECTED, fail) == kas.OUTCOME_FAIL

    # IN_FLIGHT folds into fail.
    assert kas._kernel_outcome_class(kas.CATEGORY_IN_FLIGHT, []) == kas.OUTCOME_FAIL


def test_session_kernel_opt_outcome_rollup():
    out = kas._session_kernel_opt_outcome
    # No kernels -> skip.
    assert out([]) == kas.OUTCOME_SKIP
    # Any success wins.
    assert out([
        {"outcome_class": kas.OUTCOME_FAIL},
        {"outcome_class": kas.OUTCOME_SUCCESS},
    ]) == kas.OUTCOME_SUCCESS
    # All skip -> skip.
    assert out([
        {"outcome_class": kas.OUTCOME_SKIP},
        {"outcome_class": kas.OUTCOME_SKIP},
    ]) == kas.OUTCOME_SKIP
    # timeout only when present and no real fail.
    assert out([
        {"outcome_class": kas.OUTCOME_SKIP},
        {"outcome_class": kas.OUTCOME_TIMEOUT},
    ]) == kas.OUTCOME_TIMEOUT
    # fail dominates a co-occurring timeout.
    assert out([
        {"outcome_class": kas.OUTCOME_TIMEOUT},
        {"outcome_class": kas.OUTCOME_FAIL},
    ]) == kas.OUTCOME_FAIL
