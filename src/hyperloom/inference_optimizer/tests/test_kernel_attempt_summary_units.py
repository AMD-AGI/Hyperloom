# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for pure helpers in ``kernel_attempt_summary``.

Targets the small, file-/dict-only functions (artifact path check, failure
classification, kernel-agent results harvesting, per-kernel classification)
that the higher-level ``build_kernel_optimization_summary`` tests do not
exercise directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    assert kas._backend_results_dir(tmp_path, "sid") is None

    sd = tmp_path / "sess"
    results = sd / "kernel-agent" / "runs" / "sess" / "results"
    results.mkdir(parents=True)
    assert kas._backend_results_dir(sd, "") == results

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


def test_classify_attempted_reads_task_terminal_rejection():
    """A rejected group task is out of ``rejected_kernel_ids`` by design, not in flight."""
    grouped = {
        "last_decision": "REVERT",
        "task_group_key": "grp-1",
        "integration_status": "rejected",
        "integration_rejected_reason": "revert_decision",
    }
    assert (
        kas._classify_attempted(
            grouped,
            integrated_ids=set(),
            rejected_ids=set(),
            kernel_id="k001",
        )
        == kas.CATEGORY_ATTEMPTED_REJECTED
    )
    # The reason travels under either spelling.
    assert kas._rejected_reason_of(grouped) == "revert_decision"
    assert kas._rejected_reason_of({"rejected_reason": "max_e2e_attempts_3_without_keep"}) == (
        "max_e2e_attempts_3_without_keep"
    )


def test_classify_attempted_reads_task_terminal_integration():
    """A KEEP already integrated on the task row is not left pending."""
    assert (
        kas._classify_attempted(
            {"last_decision": "KEEP", "integration_status": "integrated"},
            integrated_ids=set(),
            rejected_ids=set(),
            kernel_id="k001",
        )
        == kas.CATEGORY_INTEGRATED
    )


def test_load_backend_ladder_skipped_flag(tmp_path: Path):
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
    assert "skipped" not in ladder[1]


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
    assert out([]) == kas.OUTCOME_SKIP
    assert (
        out(
            [
                {"outcome_class": kas.OUTCOME_FAIL},
                {"outcome_class": kas.OUTCOME_SUCCESS},
            ]
        )
        == kas.OUTCOME_SUCCESS
    )
    assert (
        out(
            [
                {"outcome_class": kas.OUTCOME_SKIP},
                {"outcome_class": kas.OUTCOME_SKIP},
            ]
        )
        == kas.OUTCOME_SKIP
    )
    assert (
        out(
            [
                {"outcome_class": kas.OUTCOME_SKIP},
                {"outcome_class": kas.OUTCOME_TIMEOUT},
            ]
        )
        == kas.OUTCOME_TIMEOUT
    )
    assert (
        out(
            [
                {"outcome_class": kas.OUTCOME_TIMEOUT},
                {"outcome_class": kas.OUTCOME_FAIL},
            ]
        )
        == kas.OUTCOME_FAIL
    )


def test_collective_attempt_identity_normalizes_and_tolerates_absence():
    """A blank identity degrades to ``""`` instead of aborting the report."""
    assert (
        kas._stored_collective_attempt_id({"collective_attempt_id": " collective-attempt-1 "}) == "collective-attempt-1"
    )

    for record in ({}, {"collective_attempt_id": "   "}):
        assert kas._stored_collective_attempt_id(record) == ""


def test_collective_attempt_records_drops_unusable_history():
    """Unusable Collective rows are skipped, never raised."""
    assert kas._collective_attempt_records(SimpleNamespace()) == []
    assert kas._collective_attempt_records(SimpleNamespace(collective_attempts={"not": "a list"})) == []
    assert kas._collective_attempt_records(
        SimpleNamespace(collective_attempts=[{"collective_attempt_id": "a"}, 1])
    ) == [{"collective_attempt_id": "a"}]
    assert kas._collective_attempt_records(SimpleNamespace(collective_attempts=[{"status": "complete"}])) == []

    original = [
        {"collective_attempt_id": "a", "status": "complete"},
        {"collective_attempt_id": "b", "status": "failed"},
    ]
    records = kas._collective_attempt_records(SimpleNamespace(collective_attempts=original))
    assert records == original
    assert records[0] is not original[0]
    records[0]["status"] = "changed"
    assert original[0]["status"] == "complete"

    assert kas._collective_attempt_records(
        SimpleNamespace(
            collective_attempts=[
                {"collective_attempt_id": "duplicate", "status": "kept"},
                {"collective_attempt_id": " duplicate ", "status": "stale"},
            ]
        )
    ) == [{"collective_attempt_id": "duplicate", "status": "kept"}]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            {"integration_decision": "KEEP", "integration_status": "complete"},
            kas.CATEGORY_INTEGRATED,
        ),
        (
            {"integration_decision": "keep", "integration_status": "pending"},
            kas.CATEGORY_KEEP_PENDING,
        ),
        (
            {"integration_decision": "REVERT", "integration_status": "complete"},
            kas.CATEGORY_ATTEMPTED_REJECTED,
        ),
        (
            {"integration_status": "pending"},
            kas.CATEGORY_KEEP_PENDING,
        ),
        (
            {"kept": True, "requires_e2e_validation": True},
            kas.CATEGORY_KEEP_PENDING,
        ),
        (
            {"status": "failed"},
            kas.CATEGORY_ATTEMPTED_REJECTED,
        ),
        (
            {"decision": "NEEDS_REVIEW"},
            kas.CATEGORY_ATTEMPTED_REJECTED,
        ),
        (
            {"decision": "KEEP"},
            kas.CATEGORY_KEEP_PENDING,
        ),
        (
            {"kept": True},
            kas.CATEGORY_KEEP_PENDING,
        ),
        (
            {"status": "succeeded"},
            kas.CATEGORY_ATTEMPTED_REJECTED,
        ),
        (
            {"status": "running"},
            kas.CATEGORY_IN_FLIGHT,
        ),
    ],
    ids=[
        "integrated",
        "keep-integration-pending",
        "integration-complete-without-keep",
        "integration-pending",
        "kept-requires-e2e",
        "run-failed",
        "needs-review",
        "run-keep",
        "run-kept",
        "run-succeeded-without-keep",
        "in-flight",
    ],
)
def test_classify_collective_attempt_return_paths(record, expected):
    """Classify every terminal and nonterminal Collective return path."""
    assert kas._classify_collective_attempt(record) == expected


def test_collective_row_marks_a_microbenchmark_only_speedup():
    """A micro ratio with no E2E number must not read as a measured gain.

    The 8-GPU run landed 1.108x micro on a kernel holding 27.8% of GPU time and
    still only moved E2E by 0.39%, so an unvalidated row needs to say so.
    """
    row = kas._render_collective_attempt_row(
        {
            "collective_attempt_id": "collective-1",
            "kernel_id": "kernel-1",
            "kept": True,
            "status": "succeeded",
            "kernel_speedup": 1.1189,
        },
        kas.CATEGORY_KEEP_PENDING,
    )

    assert row["speedup_basis"] == "microbenchmark"
    assert row["e2e_gain_pct"] is None
    assert "micro_speedup=1.119x" in row["summary"]
    assert "not E2E validated" in row["summary"]


def test_render_collective_attempt_row_integrated_fields():
    """Render an integrated Collective row with metrics and provenance."""
    record = {
        "collective_attempt_id": "collective-1",
        "integration_id": "integration-1",
        "experiment_id": "experiment-1",
        "kernel_id": "kernel-1",
        "kernel_name": "all_reduce",
        "source_file": "kernels/all_reduce.py",
        "gpu_pct": "42.125",
        "engine": "forge_collective_v2",
        "integration_decision": "keep",
        "integration_result_status": "accepted",
        "integration_ts": "2026-08-11T06:00:00Z",
        "status": "succeeded",
        "kept": True,
        "kernel_speedup": "1.23456",
        "integration_gain_pct": "2.34567",
        "patch_path": "artifacts/all_reduce.patch",
        "duration_sec": "3.25",
        "collective_op": "all_reduce",
        "world_size": 8,
        "iterations": 20,
        "salvaged": True,
        "integration_workspace": "workspaces/integration-1",
    }

    row = kas._render_collective_attempt_row(record, kas.CATEGORY_INTEGRATED)

    assert row["kernel_id"] == "kernel-1"
    assert row["kernel_category"] == "collective"
    assert row["lane"] == "collective"
    assert row["engine"] == "forge_collective_v2"
    assert row["speedup_basis"] == "e2e"
    assert row["category"] == kas.CATEGORY_INTEGRATED
    assert row["outcome_class"] == kas.OUTCOME_SUCCESS
    assert row["summary"] == ("collective E2E KEEP integrated; micro_speedup=1.235x; e2e_gain=2.346%")
    assert row["last_decision"] == "KEEP"
    assert row["last_status"] == "accepted"
    assert row["last_micro_speedup"] == 1.2346
    assert row["verification"] == {
        "compile_passed": None,
        "correctness_passed": True,
        "micro_speedup": 1.2346,
        "e2e_gain_pct": 2.3457,
        "integration_decision": "KEEP",
    }
    assert row["workspace"] == "workspaces/integration-1"
    assert row["collective_op"] == "all_reduce"
    assert row["world_size"] == 8
    assert row["iterations"] == 20
    assert row["salvaged"] is True
    assert row["backend_ladder"] == [
        {
            "backend": "forge_collective",
            "status": "succeeded",
            "attempt_id": "experiment-1",
            "produced_artifact": True,
            "elapsed_sec": 3.25,
        }
    ]


@pytest.mark.parametrize(
    ("record", "category", "expected_summary", "expected_rejection"),
    [
        (
            {
                "collective_attempt_id": "keep-recovery",
                "integration_decision": "KEEP",
                "integration_recovery_action": "retry integration",
            },
            kas.CATEGORY_KEEP_PENDING,
            "collective integration recovery pending: retry integration",
            "",
        ),
        (
            {
                "collective_attempt_id": "keep-awaiting-e2e",
                "integration_decision": "KEEP",
            },
            kas.CATEGORY_KEEP_PENDING,
            "collective KEEP awaiting E2E integration",
            "",
        ),
        (
            {
                "collective_attempt_id": "e2e-revert",
                "integration_decision": "REVERT",
                "status": "complete",
                "patch": "artifacts/reverted.patch",
            },
            kas.CATEGORY_ATTEMPTED_REJECTED,
            "collective E2E REVERT",
            "collective_e2e_revert",
        ),
        (
            {
                "collective_attempt_id": "run-failed",
                "status": "failed",
                "error": "runner failed",
            },
            kas.CATEGORY_ATTEMPTED_REJECTED,
            "collective campaign failed",
            "collective_run_failed",
        ),
        (
            {
                "collective_attempt_id": "no-keep",
                "status": "complete",
                "target_file": "kernels/all_gather.py",
            },
            kas.CATEGORY_ATTEMPTED_REJECTED,
            "collective campaign did not integrate",
            "collective_no_keep",
        ),
    ],
    ids=[
        "keep-with-recovery",
        "keep-without-recovery",
        "e2e-revert",
        "run-failed",
        "no-keep",
    ],
)
def test_render_collective_attempt_row_summary_paths(
    record,
    category,
    expected_summary,
    expected_rejection,
):
    """Render Collective integration and rejection summary variants."""
    row = kas._render_collective_attempt_row(record, category)

    assert row["summary"] == expected_summary
    assert row["rejected_reason"] == expected_rejection


@pytest.mark.parametrize(
    ("record", "expected_error_class", "expected_backend_status"),
    [
        (
            {
                "integration_error_class": "validation_timeout",
                "error_class": "build_failed",
                "status": "ok",
            },
            kas.ERROR_CLASS_TIMEOUT,
            "succeeded",
        ),
        (
            {"error_class": "build_failed", "status": "failed"},
            kas.ERROR_CLASS_COMPILE_FAILED,
            "failed",
        ),
        (
            {"error_class": "correctness_mismatch", "status": "running"},
            kas.ERROR_CLASS_CORRECTNESS_FAILED,
            "running",
        ),
        (
            {"error_class": "worker_failure", "status": "crashed"},
            kas.ERROR_CLASS_AGENT_ERROR,
            "failed",
        ),
        (
            {"status": ""},
            None,
            "unknown",
        ),
    ],
    ids=[
        "timeout-success",
        "compile-failed",
        "correctness-running",
        "agent-crashed",
        "no-error-unknown",
    ],
)
def test_render_collective_attempt_row_error_and_backend_status(
    record,
    expected_error_class,
    expected_backend_status,
):
    """Map Collective error classes and backend status branches."""
    record = {
        "collective_attempt_id": "collective-status",
        **record,
    }

    row = kas._render_collective_attempt_row(
        record,
        kas.CATEGORY_ATTEMPTED_REJECTED,
    )
    backend_row = row["backend_ladder"][0]

    assert backend_row["status"] == expected_backend_status
    if expected_error_class is None:
        assert "error_class" not in backend_row
    else:
        assert backend_row["error_class"] == expected_error_class
