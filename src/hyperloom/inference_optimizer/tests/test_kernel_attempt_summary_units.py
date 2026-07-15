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
                "backend": "forge",
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
    assert row["backend"] == "forge"
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
                "recommended_backends": ["forge"],
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
            {"backend": "forge", "status": "failed", "attempt_id": "forge-1"},
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


# CATEGORY_DISPATCH — single source of truth consumed by the summary builder
# and both count sites; pin the count-key mapping + per-category summary output
# so the three formerly-duplicated dispatch sites can never drift apart.
def test_category_dispatch_count_keys():
    # The table covers exactly the four terminal categories.
    assert set(kas.CATEGORY_DISPATCH) == {
        kas.CATEGORY_INTEGRATED,
        kas.CATEGORY_KEEP_PENDING,
        kas.CATEGORY_ATTEMPTED_REJECTED,
        kas.CATEGORY_IN_FLIGHT,
    }
    # Each category maps to the ``totals`` counter the old if/elif ladder used.
    assert kas._category_count_key(kas.CATEGORY_INTEGRATED) == "integrated"
    assert kas._category_count_key(kas.CATEGORY_KEEP_PENDING) == "keep_pending"
    assert kas._category_count_key(kas.CATEGORY_ATTEMPTED_REJECTED) == "rejected"
    assert kas._category_count_key(kas.CATEGORY_IN_FLIGHT) == "in_flight"
    # Unknown/blank category falls back to the ``in_flight`` counter (the old
    # ``else`` branch), never a KeyError.
    assert kas._category_count_key("NOT_A_CATEGORY") == "in_flight"
    assert kas._category_count_key("") == "in_flight"


def test_summary_one_line_per_category():
    integrated = kas._summary_one_line(
        category=kas.CATEGORY_INTEGRATED,
        entry={"last_micro_speedup": 1.25},
        backend_ladder=[],
        artifact_error="",
    )
    assert integrated == "integrated into optimization_stack; micro_speedup=1.250x"

    keep = kas._summary_one_line(
        category=kas.CATEGORY_KEEP_PENDING,
        entry={"last_micro_speedup": 1.2},
        backend_ladder=[],
        artifact_error="",
    )
    assert keep == "KEEP awaiting integrate; micro_speedup=1.200x (pending integrate action)"

    in_flight = kas._summary_one_line(
        category=kas.CATEGORY_IN_FLIGHT,
        entry={"attempts": 3},
        backend_ladder=[],
        artifact_error="",
    )
    assert in_flight == "in-flight; 3 attempt(s) recorded, no terminal decision yet"

    # ATTEMPTED_REJECTED: all-failed ladder branch wins over the decision fallback.
    all_failed = kas._summary_one_line(
        category=kas.CATEGORY_ATTEMPTED_REJECTED,
        entry={"last_decision": "REVERT", "rejected_reason": "revert_decision"},
        backend_ladder=[
            {"backend": "geak_v3", "status": "failed", "produced_artifact": False},
            {"backend": "claude", "status": "failed", "produced_artifact": False},
        ],
        artifact_error="no usable artifact",
    )
    assert all_failed == (
        "kernel-agent ladder (geak_v3/claude) all 2 backends failed to produce a "
        "usable patch; verification: no usable artifact"
    )

    # ATTEMPTED_REJECTED: decision/reason fallback when not all-failed.
    rejected = kas._summary_one_line(
        category=kas.CATEGORY_ATTEMPTED_REJECTED,
        entry={"last_decision": "revert", "rejected_reason": "max_failures_without_keep"},
        backend_ladder=[],
        artifact_error="",
    )
    assert rejected == "REVERT; rejected_reason=max_failures_without_keep"

    # Unknown category -> empty string (the old trailing ``return ""``).
    assert (
        kas._summary_one_line(
            category="NOT_A_CATEGORY",
            entry={},
            backend_ladder=[],
            artifact_error="",
        )
        == ""
    )


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
